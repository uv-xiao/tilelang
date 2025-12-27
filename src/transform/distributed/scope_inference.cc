/*!
 * \file scope_inference.cc
 * \brief Infers communication scope (INTRA_NODE vs INTER_NODE) from PE expressions.
 *
 * This pass analyzes PE expressions in distributed operations and determines
 * whether the communication is intra-node (NVLink) or inter-node (InfiniBand).
 *
 * The pass uses the following heuristics:
 * 1. If target PE is within the same node (pe / local_size == my_pe / local_size),
 *    the scope is INTRA_NODE
 * 2. If target PE is on a different node, the scope is INTER_NODE
 * 3. If the relationship cannot be determined statically, the scope remains GLOBAL
 *
 * This information enables transport-specific optimizations:
 * - INTRA_NODE: Use NVLink-optimized paths, reduce synchronization overhead
 * - INTER_NODE: Use IB-optimized paths, batch small transfers
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

#include "../../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;
using tvm::transform::PassContext;
using tvm::ffi::GetRef;

// Communication scope constants
enum class CommScope : int {
  GPU = 0,
  INTRA_NODE = 1,
  INTER_NODE = 2,
  GLOBAL = 3,
};

// Operations that have a scope parameter to optimize
static const std::unordered_set<std::string> kScopedOperations = {
    "nvshmemx_putmem_nbi_block",
    "nvshmemx_getmem_nbi_block",
    "nvshmemx_putmem_signal_nbi_block",
    "nvshmemx_putmem_nbi_warp",
    "nvshmemx_getmem_nbi_warp",
};

/*!
 * \brief Analyzer that infers scope from PE expressions.
 */
class ScopeAnalyzer {
 public:
  ScopeAnalyzer() = default;

  /*!
   * \brief Infer scope from source and destination PE expressions.
   *
   * \param my_pe Expression representing current PE
   * \param target_pe Expression representing target PE
   * \param local_size Number of PEs per node (if known)
   * \return Inferred communication scope
   */
  CommScope InferScope(const PrimExpr& my_pe, const PrimExpr& target_pe,
                       Optional<PrimExpr> local_size) {
    arith::Analyzer analyzer;

    // Check if target_pe equals my_pe (self communication)
    if (analyzer.CanProve(target_pe == my_pe)) {
      return CommScope::GPU;
    }

    // If local_size is known, check if same node
    if (local_size.defined()) {
      PrimExpr my_node = floordiv(my_pe, local_size.value());
      PrimExpr target_node = floordiv(target_pe, local_size.value());

      if (analyzer.CanProve(my_node == target_node)) {
        return CommScope::INTRA_NODE;
      }

      if (analyzer.CanProve(my_node != target_node)) {
        return CommScope::INTER_NODE;
      }
    }

    // Check common patterns for intra-node communication
    // Pattern: (pe + 1) % local_size or similar
    // This requires more sophisticated pattern matching

    // Default to GLOBAL if scope cannot be determined
    return CommScope::GLOBAL;
  }

  /*!
   * \brief Check if an expression references local PE index.
   */
  bool IsLocalPePattern(const PrimExpr& expr) {
    // Check for patterns like local_pe, pe % local_size, etc.
    if (const auto* call = expr.as<CallNode>()) {
      if (call->op.same_as(nvshmem_local_pe())) {
        return true;
      }
    }
    return false;
  }
};

/*!
 * \brief Mutator that annotates operations with inferred scope.
 */
class ScopeInferenceMutator : public StmtExprMutator {
 public:
  ScopeInferenceMutator() : analyzer_() {}

  Stmt VisitStmt_(const EvaluateNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    const auto* node = stmt.as<EvaluateNode>();

    if (const auto* call = node->value.as<CallNode>()) {
      // Handle NVSHMEM operations
      if (call->op.same_as(nvshmem_putmem_nbi_block()) ||
          call->op.same_as(nvshmem_getmem_nbi_block())) {
        return OptimizeScopedOperation(call);
      }

      if (call->op.same_as(nvshmem_putmem_signal_nbi_block())) {
        return OptimizePutSignal(call);
      }

      // Handle call_extern for operations that may have scope
      if (call->op.same_as(builtin::call_extern())) {
        std::string func_name = ExtractFunctionName(call);
        if (kScopedOperations.count(func_name)) {
          return OptimizeCallExtern(call, func_name);
        }
      }
    }

    return stmt;
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    // Track let bindings for PE-related values
    if (IsPeRelated(op->var)) {
      pe_bindings_[op->var.get()] = op->value;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

 private:
  ScopeAnalyzer analyzer_;
  std::unordered_map<const VarNode*, PrimExpr> pe_bindings_;
  Optional<PrimExpr> current_local_size_;

  std::string ExtractFunctionName(const CallNode* call) {
    if (call->args.size() >= 2) {
      if (const auto* str = call->args[1].as<StringImmNode>()) {
        return str->value;
      }
    }
    return "";
  }

  bool IsPeRelated(const Var& var) {
    std::string name = var->name_hint;
    return name.find("pe") != std::string::npos ||
           name.find("rank") != std::string::npos ||
           name.find("local") != std::string::npos;
  }

  /*!
   * \brief Try to infer local_size from the IR context.
   */
  Optional<PrimExpr> TryGetLocalSize() {
    if (current_local_size_.defined()) {
      return current_local_size_;
    }

    // Look for nvshmem_local_size() calls in bindings
    for (const auto& kv : pe_bindings_) {
      if (const auto* call = kv.second.as<CallNode>()) {
        if (call->op.same_as(nvshmem_local_size())) {
          current_local_size_ = kv.second;
          return current_local_size_;
        }
      }
    }

    return std::nullopt;
  }

  /*!
   * \brief Optimize a scoped put/get operation by inferring scope.
   */
  Stmt OptimizeScopedOperation(const CallNode* call) {
    // For nvshmem_putmem_nbi_block(dst, src, bytes, pe)
    // or nvshmem_getmem_nbi_block(dst, src, bytes, pe)
    if (call->args.size() < 4) {
      return Evaluate(GetRef<Call>(call));
    }

    PrimExpr target_pe = call->args[3];

    // Get my_pe
    PrimExpr my_pe = Call(DataType::Int(32), nvshmem_my_pe(), {});

    // Try to get local_size
    Optional<PrimExpr> local_size = TryGetLocalSize();

    // Infer scope
    CommScope scope = analyzer_.InferScope(my_pe, target_pe, local_size);

    // If we inferred a more specific scope, we could potentially
    // use optimized transport paths. For now, we just return the
    // original call since NVSHMEM handles routing internally.
    //
    // In the future, we could:
    // 1. Use different intrinsics for intra-node vs inter-node
    // 2. Batch inter-node transfers for better efficiency
    // 3. Use direct NVLink paths for intra-node

    // Add scope annotation as an attribute for downstream passes
    if (scope == CommScope::INTRA_NODE) {
      return AttrStmt(GetRef<Call>(call), "comm_scope",
                      StringImm("intra_node"),
                      Evaluate(GetRef<Call>(call)));
    } else if (scope == CommScope::INTER_NODE) {
      return AttrStmt(GetRef<Call>(call), "comm_scope",
                      StringImm("inter_node"),
                      Evaluate(GetRef<Call>(call)));
    }

    return Evaluate(GetRef<Call>(call));
  }

  /*!
   * \brief Optimize put_signal operation.
   */
  Stmt OptimizePutSignal(const CallNode* call) {
    // For nvshmem_putmem_signal_nbi_block(dst, src, bytes, sig_addr, signal, sig_op, pe)
    if (call->args.size() < 7) {
      return Evaluate(GetRef<Call>(call));
    }

    PrimExpr target_pe = call->args[6];
    PrimExpr my_pe = Call(DataType::Int(32), nvshmem_my_pe(), {});
    Optional<PrimExpr> local_size = TryGetLocalSize();

    CommScope scope = analyzer_.InferScope(my_pe, target_pe, local_size);

    if (scope == CommScope::INTRA_NODE) {
      return AttrStmt(GetRef<Call>(call), "comm_scope",
                      StringImm("intra_node"),
                      Evaluate(GetRef<Call>(call)));
    } else if (scope == CommScope::INTER_NODE) {
      return AttrStmt(GetRef<Call>(call), "comm_scope",
                      StringImm("inter_node"),
                      Evaluate(GetRef<Call>(call)));
    }

    return Evaluate(GetRef<Call>(call));
  }

  /*!
   * \brief Optimize call_extern operations with scope.
   */
  Stmt OptimizeCallExtern(const CallNode* call, const std::string& func_name) {
    // Extract PE argument based on operation type
    // Most operations have PE as the last argument before any optional params
    size_t pe_arg_idx = 0;

    if (func_name.find("putmem") != std::string::npos ||
        func_name.find("getmem") != std::string::npos) {
      // (dst, src, bytes, pe) or (dst, src, bytes, sig, signal, sig_op, pe)
      if (func_name.find("signal") != std::string::npos) {
        pe_arg_idx = 8;  // After "handle", func_name, dst, src, bytes, sig, signal, sig_op
      } else {
        pe_arg_idx = 5;  // After "handle", func_name, dst, src, bytes
      }
    }

    if (pe_arg_idx > 0 && call->args.size() > pe_arg_idx) {
      PrimExpr target_pe = call->args[pe_arg_idx];
      PrimExpr my_pe = Call(DataType::Int(32), nvshmem_my_pe(), {});
      Optional<PrimExpr> local_size = TryGetLocalSize();

      CommScope scope = analyzer_.InferScope(my_pe, target_pe, local_size);

      if (scope != CommScope::GLOBAL) {
        std::string scope_str = (scope == CommScope::INTRA_NODE) ? "intra_node" : "inter_node";
        return AttrStmt(GetRef<Call>(call), "comm_scope",
                        StringImm(scope_str),
                        Evaluate(GetRef<Call>(call)));
      }
    }

    return Evaluate(GetRef<Call>(call));
  }
};

namespace transform {
using namespace tir::transform;

/*!
 * \brief Create ScopeInferencePass.
 *
 * This pass analyzes PE expressions in distributed operations and infers
 * whether communication is intra-node or inter-node. This information
 * can be used by downstream passes or code generation to optimize
 * transport selection.
 */
tvm::transform::Pass ScopeInference() {
  auto pass_func = [](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    auto* n = f.CopyOnWrite();
    ScopeInferenceMutator mutator;
    n->body = mutator(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ScopeInference", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ScopeInference", ScopeInference);
}

}  // namespace transform
}  // namespace tl
}  // namespace tvm
