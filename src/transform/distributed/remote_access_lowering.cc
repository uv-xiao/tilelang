/*!
 * \file remote_access_lowering.cc
 * \brief Converts high-level remote memory operations to low-level async primitives.
 *
 * This pass transforms:
 * - remote_load(src, dst) -> nvshmem_getmem_nbi_block + nvshmem_quiet
 * - remote_store(src, dst) -> nvshmem_putmem_nbi_block + nvshmem_quiet
 * - put_signal -> nvshmem_putmem_signal_nbi_block
 *
 * The pass identifies call_extern nodes with distributed operation names
 * and rewrites them to use NVSHMEM intrinsics.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

#include "../../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;
using tvm::transform::PassContext;
using tvm::ffi::GetRef;

// High-level distributed operation names that need lowering
static const std::unordered_set<std::string> kRemoteLoadCalls = {
    "tl_dist_remote_load",
    "remote_load",
};

static const std::unordered_set<std::string> kRemoteStoreCalls = {
    "tl_dist_remote_store",
    "remote_store",
};

static const std::unordered_set<std::string> kPutSignalCalls = {
    "tl_dist_put_signal",
    "put_signal",
};

static const std::unordered_set<std::string> kGetAsyncCalls = {
    "tl_dist_get_async",
    "get_async",
};

static const std::unordered_set<std::string> kPutAsyncCalls = {
    "tl_dist_put_async",
    "put_async",
};

/*!
 * \brief Mutator that transforms high-level remote access calls to NVSHMEM intrinsics.
 */
class RemoteAccessLoweringMutator : public StmtExprMutator {
 public:
  RemoteAccessLoweringMutator() = default;

  Stmt VisitStmt_(const EvaluateNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    const auto* node = stmt.as<EvaluateNode>();

    if (const auto* call = node->value.as<CallNode>()) {
      // Check if this is a call_extern
      if (call->op.same_as(builtin::call_extern())) {
        std::string func_name = ExtractFunctionName(call);

        if (kRemoteLoadCalls.count(func_name)) {
          return LowerRemoteLoad(call);
        } else if (kRemoteStoreCalls.count(func_name)) {
          return LowerRemoteStore(call);
        } else if (kPutSignalCalls.count(func_name)) {
          return LowerPutSignal(call);
        } else if (kGetAsyncCalls.count(func_name)) {
          return LowerGetAsync(call);
        } else if (kPutAsyncCalls.count(func_name)) {
          return LowerPutAsync(call);
        }
      }
    }

    return stmt;
  }

 private:
  /*!
   * \brief Extract function name from call_extern arguments.
   */
  std::string ExtractFunctionName(const CallNode* call) {
    // call_extern format: call_extern(dtype, func_name, args...)
    // or: call_extern("handle", func_name, args...)
    if (call->args.size() >= 2) {
      if (const auto* str = call->args[1].as<StringImmNode>()) {
        return str->value;
      }
    }
    if (call->args.size() >= 1) {
      if (const auto* str = call->args[0].as<StringImmNode>()) {
        return str->value;
      }
    }
    return "";
  }

  /*!
   * \brief Lower remote_load to nvshmem_getmem_nbi_block + nvshmem_quiet.
   *
   * Input: remote_load(dst_ptr, src_ptr, size, src_pe, scope_id)
   * Output:
   *   nvshmem_getmem_nbi_block(dst_ptr, src_ptr, size, src_pe)
   *   nvshmem_quiet()  // For blocking semantics
   */
  Stmt LowerRemoteLoad(const CallNode* call) {
    // Parse arguments: [dtype/"handle", func_name, dst, src, size, pe, ...]
    size_t arg_offset = 2;  // Skip dtype and func_name
    if (call->args.size() < arg_offset + 4) {
      // Not enough arguments, return original
      return Evaluate(GetRef<Call>(call));
    }

    PrimExpr dst_ptr = call->args[arg_offset];
    PrimExpr src_ptr = call->args[arg_offset + 1];
    PrimExpr size = call->args[arg_offset + 2];
    PrimExpr src_pe = call->args[arg_offset + 3];

    // Generate nvshmem_getmem_nbi_block call
    Call get_call = Call(DataType::Handle(), nvshmem_getmem_nbi_block(),
                         {dst_ptr, src_ptr, size, src_pe});

    // Generate nvshmem_quiet call for blocking semantics
    Call quiet_call = Call(DataType::Handle(), nvshmem_quiet(), {});

    return SeqStmt({Evaluate(get_call), Evaluate(quiet_call)});
  }

  /*!
   * \brief Lower remote_store to nvshmem_putmem_nbi_block + nvshmem_quiet.
   *
   * Input: remote_store(src_ptr, dst_ptr, size, dst_pe, scope_id)
   * Output:
   *   nvshmem_putmem_nbi_block(dst_ptr, src_ptr, size, dst_pe)
   *   nvshmem_quiet()  // For blocking semantics
   */
  Stmt LowerRemoteStore(const CallNode* call) {
    size_t arg_offset = 2;
    if (call->args.size() < arg_offset + 4) {
      return Evaluate(GetRef<Call>(call));
    }

    PrimExpr src_ptr = call->args[arg_offset];
    PrimExpr dst_ptr = call->args[arg_offset + 1];
    PrimExpr size = call->args[arg_offset + 2];
    PrimExpr dst_pe = call->args[arg_offset + 3];

    // Generate nvshmem_putmem_nbi_block call
    Call put_call = Call(DataType::Handle(), nvshmem_putmem_nbi_block(),
                         {dst_ptr, src_ptr, size, dst_pe});

    // Generate nvshmem_quiet call for blocking semantics
    Call quiet_call = Call(DataType::Handle(), nvshmem_quiet(), {});

    return SeqStmt({Evaluate(put_call), Evaluate(quiet_call)});
  }

  /*!
   * \brief Lower put_signal to nvshmem_putmem_signal_nbi_block.
   *
   * Input: put_signal(src_ptr, dst_ptr, size, sig_addr, signal, sig_op, dst_pe, scope_id)
   * Output:
   *   nvshmem_putmem_signal_nbi_block(dst_ptr, src_ptr, size, sig_addr, signal, sig_op, dst_pe)
   */
  Stmt LowerPutSignal(const CallNode* call) {
    size_t arg_offset = 2;
    if (call->args.size() < arg_offset + 7) {
      return Evaluate(GetRef<Call>(call));
    }

    PrimExpr src_ptr = call->args[arg_offset];
    PrimExpr dst_ptr = call->args[arg_offset + 1];
    PrimExpr size = call->args[arg_offset + 2];
    PrimExpr sig_addr = call->args[arg_offset + 3];
    PrimExpr signal = call->args[arg_offset + 4];
    PrimExpr sig_op = call->args[arg_offset + 5];
    PrimExpr dst_pe = call->args[arg_offset + 6];

    // Generate nvshmem_putmem_signal_nbi_block call
    Call put_signal_call = Call(DataType::Handle(), nvshmem_putmem_signal_nbi_block(),
                                {dst_ptr, src_ptr, size, sig_addr, signal, sig_op, dst_pe});

    return Evaluate(put_signal_call);
  }

  /*!
   * \brief Lower get_async to nvshmem_getmem_nbi_block (non-blocking).
   *
   * Input: get_async(src_ptr, dst_ptr, size, src_pe, scope_id, exec_scope)
   * Output:
   *   nvshmem_getmem_nbi_block(dst_ptr, src_ptr, size, src_pe)
   */
  Stmt LowerGetAsync(const CallNode* call) {
    size_t arg_offset = 2;
    if (call->args.size() < arg_offset + 4) {
      return Evaluate(GetRef<Call>(call));
    }

    PrimExpr src_ptr = call->args[arg_offset];
    PrimExpr dst_ptr = call->args[arg_offset + 1];
    PrimExpr size = call->args[arg_offset + 2];
    PrimExpr src_pe = call->args[arg_offset + 3];

    // Generate nvshmem_getmem_nbi_block call (non-blocking, no quiet)
    Call get_call = Call(DataType::Handle(), nvshmem_getmem_nbi_block(),
                         {dst_ptr, src_ptr, size, src_pe});

    return Evaluate(get_call);
  }

  /*!
   * \brief Lower put_async to nvshmem_putmem_nbi_block (non-blocking).
   *
   * Input: put_async(src_ptr, dst_ptr, size, dst_pe, scope_id, exec_scope)
   * Output:
   *   nvshmem_putmem_nbi_block(dst_ptr, src_ptr, size, dst_pe)
   */
  Stmt LowerPutAsync(const CallNode* call) {
    size_t arg_offset = 2;
    if (call->args.size() < arg_offset + 4) {
      return Evaluate(GetRef<Call>(call));
    }

    PrimExpr src_ptr = call->args[arg_offset];
    PrimExpr dst_ptr = call->args[arg_offset + 1];
    PrimExpr size = call->args[arg_offset + 2];
    PrimExpr dst_pe = call->args[arg_offset + 3];

    // Generate nvshmem_putmem_nbi_block call (non-blocking, no quiet)
    Call put_call = Call(DataType::Handle(), nvshmem_putmem_nbi_block(),
                         {dst_ptr, src_ptr, size, dst_pe});

    return Evaluate(put_call);
  }
};

namespace transform {
using namespace tir::transform;

/*!
 * \brief Create RemoteAccessLoweringPass.
 *
 * This pass converts high-level remote memory operations (remote_load, remote_store,
 * put_signal, get_async, put_async) to low-level NVSHMEM intrinsics.
 */
tvm::transform::Pass RemoteAccessLowering() {
  auto pass_func = [](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    auto* n = f.CopyOnWrite();
    RemoteAccessLoweringMutator mutator;
    n->body = mutator(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.RemoteAccessLowering", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.RemoteAccessLowering", RemoteAccessLowering);
}

}  // namespace transform
}  // namespace tl
}  // namespace tvm
