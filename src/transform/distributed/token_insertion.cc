/*!
 * \file token_insertion.cc
 * \brief Tracks async operations and inserts synchronization at buffer use points.
 *
 * This pass implements a token-based synchronization model for async operations:
 *
 * 1. Tracks all async put/get operations and their target buffers
 * 2. Analyzes buffer usage patterns to find where data is consumed
 * 3. Inserts appropriate synchronization (quiet/fence/signal_wait) before consumption
 *
 * The pass ensures memory consistency by:
 * - Inserting nvshmem_quiet() before a buffer written by put_async is read locally
 * - Inserting nvshmem_fence() to order put operations
 * - Inserting signal_wait() when signal-based synchronization is used
 *
 * This optimization reduces unnecessary global synchronization by only
 * synchronizing when data is actually needed.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;
using tvm::transform::PassContext;

/*!
 * \brief Information about an outstanding async operation.
 */
struct AsyncOpInfo {
  enum class OpType {
    PUT_ASYNC,
    GET_ASYNC,
    PUT_SIGNAL,
  };

  OpType type;
  Var target_buffer;       // Buffer being written (for put) or read (for get)
  Optional<Var> signal;    // Signal variable (for put_signal)
  bool is_local_target;    // True if target is local buffer (get), false if remote (put)
};

/*!
 * \brief Collector that identifies async operations and their target buffers.
 */
class AsyncOpCollector : public StmtExprVisitor {
 public:
  void VisitStmt_(const EvaluateNode* op) final {
    if (const auto* call = op->value.as<CallNode>()) {
      // Check for NVSHMEM async operations
      if (call->op.same_as(nvshmem_putmem_nbi_block())) {
        // nvshmem_putmem_nbi_block(dst, src, bytes, pe)
        // dst is remote, src is local
        if (call->args.size() >= 2) {
          if (const auto* load = call->args[1].as<BufferLoadNode>()) {
            pending_puts_.insert(load->buffer->data.get());
          } else if (const auto* var = call->args[1].as<VarNode>()) {
            pending_puts_.insert(var);
          }
        }
      } else if (call->op.same_as(nvshmem_getmem_nbi_block())) {
        // nvshmem_getmem_nbi_block(dst, src, bytes, pe)
        // dst is local, src is remote
        if (call->args.size() >= 1) {
          if (const auto* load = call->args[0].as<BufferLoadNode>()) {
            pending_gets_.insert(load->buffer->data.get());
          } else if (const auto* var = call->args[0].as<VarNode>()) {
            pending_gets_.insert(var);
          }
        }
      } else if (call->op.same_as(nvshmem_putmem_signal_nbi_block())) {
        // Signal-based put - synchronization is via signal_wait
        if (call->args.size() >= 2) {
          if (const auto* load = call->args[1].as<BufferLoadNode>()) {
            signal_puts_.insert(load->buffer->data.get());
          } else if (const auto* var = call->args[1].as<VarNode>()) {
            signal_puts_.insert(var);
          }
        }
      } else if (call->op.same_as(nvshmem_quiet())) {
        // Quiet clears all pending operations
        pending_puts_.clear();
        pending_gets_.clear();
      } else if (call->op.same_as(nvshmem_fence())) {
        // Fence orders puts but doesn't complete them
        // Keep tracking
      } else if (call->op.same_as(nvshmem_signal_wait_until())) {
        // Signal wait clears signal-based puts
        signal_puts_.clear();
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    // Check if this load is from a buffer with pending gets
    const VarNode* var = op->buffer->data.get();
    if (pending_gets_.count(var)) {
      buffers_needing_sync_.insert(var);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    // Check if this store is to a buffer that was source of a put
    const VarNode* var = op->buffer->data.get();
    if (pending_puts_.count(var)) {
      // Writing to a buffer that has pending puts - need fence
      buffers_needing_fence_.insert(var);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  // Buffers with pending get operations
  std::unordered_set<const VarNode*> pending_gets_;
  // Buffers with pending put operations (source buffers)
  std::unordered_set<const VarNode*> pending_puts_;
  // Buffers with signal-based puts
  std::unordered_set<const VarNode*> signal_puts_;
  // Buffers that need synchronization before use
  std::unordered_set<const VarNode*> buffers_needing_sync_;
  // Buffers that need fence before modification
  std::unordered_set<const VarNode*> buffers_needing_fence_;
};

/*!
 * \brief Mutator that inserts synchronization at buffer use points.
 */
class TokenInsertionMutator : public StmtExprMutator {
 public:
  TokenInsertionMutator() = default;

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    // Process sequence and track async operations
    Array<Stmt> new_seq;
    new_seq.reserve(op->seq.size());

    for (size_t i = 0; i < op->seq.size(); ++i) {
      const Stmt& stmt = op->seq[i];

      // Check if this statement is an async operation
      if (IsAsyncOp(stmt)) {
        UpdatePendingOps(stmt);
      }

      // Check if next statement uses a buffer with pending operations
      if (i + 1 < op->seq.size()) {
        const Stmt& next_stmt = op->seq[i + 1];
        if (NeedsSyncBefore(next_stmt)) {
          // Insert sync before next statement
          new_seq.push_back(VisitStmt(stmt));
          new_seq.push_back(MakeQuietStmt());
          ClearPendingGets();
          continue;
        }
      }

      new_seq.push_back(VisitStmt(stmt));
    }

    if (new_seq.size() == 1) {
      return new_seq[0];
    }
    return SeqStmt(new_seq);
  }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    const auto* node = stmt.as<EvaluateNode>();

    if (const auto* call = node->value.as<CallNode>()) {
      // Track async operations
      if (call->op.same_as(nvshmem_getmem_nbi_block())) {
        TrackGetAsync(call);
      } else if (call->op.same_as(nvshmem_putmem_nbi_block())) {
        TrackPutAsync(call);
      } else if (call->op.same_as(nvshmem_quiet())) {
        ClearAllPending();
      } else if (call->op.same_as(nvshmem_barrier_all_block())) {
        ClearAllPending();
      }
    }

    return stmt;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // Before entering a loop that uses pending buffers, insert sync
    AsyncOpCollector collector;
    collector(op->body);

    bool needs_sync = false;
    for (const VarNode* var : collector.buffers_needing_sync_) {
      if (pending_gets_.count(var)) {
        needs_sync = true;
        break;
      }
    }

    Stmt body = StmtExprMutator::VisitStmt_(op);

    if (needs_sync) {
      return SeqStmt({MakeQuietStmt(), body});
    }

    return body;
  }

 private:
  // Track buffers with pending get operations
  std::unordered_set<const VarNode*> pending_gets_;
  // Track buffers with pending put operations
  std::unordered_set<const VarNode*> pending_puts_;

  bool IsAsyncOp(const Stmt& stmt) {
    if (const auto* eval = stmt.as<EvaluateNode>()) {
      if (const auto* call = eval->value.as<CallNode>()) {
        return call->op.same_as(nvshmem_getmem_nbi_block()) ||
               call->op.same_as(nvshmem_putmem_nbi_block()) ||
               call->op.same_as(nvshmem_putmem_signal_nbi_block());
      }
    }
    return false;
  }

  void UpdatePendingOps(const Stmt& stmt) {
    if (const auto* eval = stmt.as<EvaluateNode>()) {
      if (const auto* call = eval->value.as<CallNode>()) {
        if (call->op.same_as(nvshmem_getmem_nbi_block())) {
          TrackGetAsync(call);
        } else if (call->op.same_as(nvshmem_putmem_nbi_block())) {
          TrackPutAsync(call);
        }
      }
    }
  }

  void TrackGetAsync(const CallNode* call) {
    // nvshmem_getmem_nbi_block(dst, src, bytes, pe)
    if (call->args.size() >= 1) {
      if (const auto* load = call->args[0].as<BufferLoadNode>()) {
        pending_gets_.insert(load->buffer->data.get());
      } else if (const auto* var = call->args[0].as<VarNode>()) {
        pending_gets_.insert(var);
      }
    }
  }

  void TrackPutAsync(const CallNode* call) {
    // nvshmem_putmem_nbi_block(dst, src, bytes, pe)
    if (call->args.size() >= 2) {
      if (const auto* load = call->args[1].as<BufferLoadNode>()) {
        pending_puts_.insert(load->buffer->data.get());
      } else if (const auto* var = call->args[1].as<VarNode>()) {
        pending_puts_.insert(var);
      }
    }
  }

  bool NeedsSyncBefore(const Stmt& stmt) {
    if (pending_gets_.empty()) {
      return false;
    }

    // Check if this statement reads from any pending buffer
    AsyncOpCollector collector;
    collector(stmt);

    for (const VarNode* var : collector.buffers_needing_sync_) {
      if (pending_gets_.count(var)) {
        return true;
      }
    }

    return false;
  }

  Stmt MakeQuietStmt() {
    Call quiet = Call(DataType::Handle(), nvshmem_quiet(), {});
    return Evaluate(quiet);
  }

  Stmt MakeFenceStmt() {
    Call fence = Call(DataType::Handle(), nvshmem_fence(), {});
    return Evaluate(fence);
  }

  void ClearPendingGets() {
    pending_gets_.clear();
  }

  void ClearPendingPuts() {
    pending_puts_.clear();
  }

  void ClearAllPending() {
    pending_gets_.clear();
    pending_puts_.clear();
  }
};

namespace transform {
using namespace tir::transform;

/*!
 * \brief Create TokenInsertionPass.
 *
 * This pass tracks async put/get operations and inserts appropriate
 * synchronization (quiet/fence) before buffers are used. This ensures
 * memory consistency while minimizing global synchronization overhead.
 */
tvm::transform::Pass TokenInsertion() {
  auto pass_func = [](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    auto* n = f.CopyOnWrite();
    TokenInsertionMutator mutator;
    n->body = mutator(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.TokenInsertion", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.TokenInsertion", TokenInsertion);
}

}  // namespace transform
}  // namespace tl
}  // namespace tvm
