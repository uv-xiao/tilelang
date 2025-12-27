/*!
 * \file collective_lowering.cc
 * \brief Expands high-level collective operations to NVSHMEM primitives.
 *
 * This pass transforms high-level collective operations (allreduce, allgather,
 * reduce_scatter, broadcast) into sequences of low-level NVSHMEM operations.
 *
 * Supported algorithms:
 * - hierarchical: 3-phase algorithm optimized for multi-node (intra-node, inter-node, intra-node)
 * - ring: Ring algorithm for bandwidth-optimal collectives
 * - tree: Binary tree algorithm for latency-optimal collectives
 *
 * The pass generates NVSHMEM team-based collective calls that are topology-aware.
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

// NVSHMEM reduce operation constants
static constexpr int NVSHMEM_REDUCE_SUM = 0;
static constexpr int NVSHMEM_REDUCE_MAX = 1;
static constexpr int NVSHMEM_REDUCE_MIN = 2;
static constexpr int NVSHMEM_REDUCE_PROD = 3;
static constexpr int NVSHMEM_REDUCE_AND = 4;
static constexpr int NVSHMEM_REDUCE_OR = 5;
static constexpr int NVSHMEM_REDUCE_XOR = 6;

// NVSHMEM team constants
static constexpr int NVSHMEM_TEAM_WORLD = 0;
static constexpr int NVSHMEM_TEAM_NODE = 1;
static constexpr int NVSHMEM_TEAM_SHARED = 2;

// NVSHMEM signal operation constants
static constexpr int NVSHMEM_SIGNAL_SET = 0;
static constexpr int NVSHMEM_SIGNAL_ADD = 1;

// Communication scope constants
static constexpr int SCOPE_GPU = 0;
static constexpr int SCOPE_INTRA_NODE = 1;
static constexpr int SCOPE_INTER_NODE = 2;
static constexpr int SCOPE_GLOBAL = 3;

// High-level collective operation names
static const std::unordered_set<std::string> kAllReduceCalls = {
    "tl_dist_allreduce_hierarchical",
    "tl_dist_allreduce_ring",
    "tl_dist_allreduce_tree",
};

static const std::unordered_set<std::string> kAllGatherCalls = {
    "tl_dist_allgather_hierarchical",
    "tl_dist_allgather_ring",
};

static const std::unordered_set<std::string> kReduceScatterCalls = {
    "tl_dist_reduce_scatter_hierarchical",
    "tl_dist_reduce_scatter_ring",
};

static const std::unordered_set<std::string> kBroadcastCalls = {
    "tl_dist_broadcast_hierarchical",
    "tl_dist_broadcast_binomial",
};

static const std::unordered_set<std::string> kTeamCollectiveCalls = {
    "tl_dist_team_allreduce",
    "tl_dist_team_allgather",
    "tl_dist_team_broadcast",
};

static const std::unordered_set<std::string> kAllToAllCalls = {
    "tl_dist_alltoall",
};

/*!
 * \brief Mutator that expands collective operations to NVSHMEM primitives.
 */
class CollectiveLoweringMutator : public StmtExprMutator {
 public:
  CollectiveLoweringMutator() = default;

  Stmt VisitStmt_(const EvaluateNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    const auto* node = stmt.as<EvaluateNode>();

    if (const auto* call = node->value.as<CallNode>()) {
      if (call->op.same_as(builtin::call_extern())) {
        std::string func_name = ExtractFunctionName(call);

        if (kAllReduceCalls.count(func_name)) {
          return LowerAllReduce(call, func_name);
        } else if (kAllGatherCalls.count(func_name)) {
          return LowerAllGather(call, func_name);
        } else if (kReduceScatterCalls.count(func_name)) {
          return LowerReduceScatter(call, func_name);
        } else if (kBroadcastCalls.count(func_name)) {
          return LowerBroadcast(call, func_name);
        } else if (kTeamCollectiveCalls.count(func_name)) {
          return LowerTeamCollective(call, func_name);
        } else if (kAllToAllCalls.count(func_name)) {
          return LowerAllToAll(call, func_name);
        }
      }
    }

    return stmt;
  }

 private:
  std::string ExtractFunctionName(const CallNode* call) {
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
   * \brief Lower allreduce to NVSHMEM reduce operations.
   *
   * For hierarchical algorithm, generates:
   * 1. Team reduce on TEAM_NODE (intra-node)
   * 2. Team sync on TEAM_NODE
   * 3. Global reduce on TEAM_WORLD (inter-node)
   * 4. Global barrier
   */
  Stmt LowerAllReduce(const CallNode* call, const std::string& algorithm) {
    size_t arg_offset = 2;
    if (call->args.size() < arg_offset + 2) {
      return Evaluate(GetRef<Call>(call));
    }

    PrimExpr buf_ptr = call->args[arg_offset];
    PrimExpr nelems = call->args[arg_offset + 1];
    PrimExpr op_id = (call->args.size() > arg_offset + 2)
                         ? call->args[arg_offset + 2]
                         : IntImm(DataType::Int(32), NVSHMEM_REDUCE_SUM);
    PrimExpr dtype_id = (call->args.size() > arg_offset + 3)
                            ? call->args[arg_offset + 3]
                            : IntImm(DataType::Int(32), 1);  // float32

    Array<Stmt> stmts;

    if (algorithm.find("hierarchical") != std::string::npos) {
      // Phase 1: Intra-node reduce using TEAM_NODE
      Call intra_reduce = Call(
          DataType::Handle(), builtin::call_extern(),
          {StringImm("handle"), StringImm("nvshmemx_float_sum_reduce_block"),
           buf_ptr, buf_ptr, nelems, IntImm(DataType::Int(32), NVSHMEM_TEAM_NODE)});
      stmts.push_back(Evaluate(intra_reduce));

      // Team sync on TEAM_NODE
      Call team_sync = Call(
          DataType::Handle(), builtin::call_extern(),
          {StringImm("handle"), StringImm("nvshmemx_team_sync_block"),
           IntImm(DataType::Int(32), NVSHMEM_TEAM_NODE)});
      stmts.push_back(Evaluate(team_sync));

      // Phase 2: Global reduce using TEAM_WORLD
      Call global_reduce = Call(
          DataType::Handle(), builtin::call_extern(),
          {StringImm("handle"), StringImm("nvshmemx_float_sum_reduce_block"),
           buf_ptr, buf_ptr, nelems, IntImm(DataType::Int(32), NVSHMEM_TEAM_WORLD)});
      stmts.push_back(Evaluate(global_reduce));

      // Global barrier
      Call barrier = Call(DataType::Handle(), nvshmem_barrier_all_block(), {});
      stmts.push_back(Evaluate(barrier));
    } else {
      // For ring/tree, use NVSHMEM's optimized reduce
      Call reduce = Call(
          DataType::Handle(), builtin::call_extern(),
          {StringImm("handle"), StringImm("nvshmemx_float_sum_reduce_block"),
           buf_ptr, buf_ptr, nelems, IntImm(DataType::Int(32), NVSHMEM_TEAM_WORLD)});
      stmts.push_back(Evaluate(reduce));

      Call barrier = Call(DataType::Handle(), nvshmem_barrier_all_block(), {});
      stmts.push_back(Evaluate(barrier));
    }

    return SeqStmt(stmts);
  }

  /*!
   * \brief Lower allgather to NVSHMEM fcollect.
   */
  Stmt LowerAllGather(const CallNode* call, const std::string& algorithm) {
    size_t arg_offset = 2;
    if (call->args.size() < arg_offset + 3) {
      return Evaluate(GetRef<Call>(call));
    }

    PrimExpr dst_ptr = call->args[arg_offset];
    PrimExpr src_ptr = call->args[arg_offset + 1];
    PrimExpr src_nelems = call->args[arg_offset + 2];

    Array<Stmt> stmts;

    // Use NVSHMEM's fcollect (allgather equivalent)
    Call fcollect = Call(
        DataType::Handle(), builtin::call_extern(),
        {StringImm("handle"), StringImm("nvshmemx_float_fcollect_block"),
         dst_ptr, src_ptr, src_nelems, IntImm(DataType::Int(32), NVSHMEM_TEAM_WORLD)});
    stmts.push_back(Evaluate(fcollect));

    Call barrier = Call(DataType::Handle(), nvshmem_barrier_all_block(), {});
    stmts.push_back(Evaluate(barrier));

    return SeqStmt(stmts);
  }

  /*!
   * \brief Lower reduce_scatter.
   */
  Stmt LowerReduceScatter(const CallNode* call, const std::string& algorithm) {
    size_t arg_offset = 2;
    if (call->args.size() < arg_offset + 3) {
      return Evaluate(GetRef<Call>(call));
    }

    PrimExpr dst_ptr = call->args[arg_offset];
    PrimExpr src_ptr = call->args[arg_offset + 1];
    PrimExpr dst_nelems = call->args[arg_offset + 2];
    PrimExpr op_id = (call->args.size() > arg_offset + 3)
                         ? call->args[arg_offset + 3]
                         : IntImm(DataType::Int(32), NVSHMEM_REDUCE_SUM);

    Array<Stmt> stmts;

    // For reduce-scatter, we can use ring algorithm implementation
    // First, implement as a placeholder that calls a runtime helper
    Call reduce_scatter = Call(
        DataType::Handle(), builtin::call_extern(),
        {StringImm("handle"), StringImm("tl_nvshmem_reduce_scatter_ring"),
         dst_ptr, src_ptr, dst_nelems, op_id});
    stmts.push_back(Evaluate(reduce_scatter));

    Call barrier = Call(DataType::Handle(), nvshmem_barrier_all_block(), {});
    stmts.push_back(Evaluate(barrier));

    return SeqStmt(stmts);
  }

  /*!
   * \brief Lower broadcast.
   */
  Stmt LowerBroadcast(const CallNode* call, const std::string& algorithm) {
    size_t arg_offset = 2;
    if (call->args.size() < arg_offset + 3) {
      return Evaluate(GetRef<Call>(call));
    }

    PrimExpr buf_ptr = call->args[arg_offset];
    PrimExpr nelems = call->args[arg_offset + 1];
    PrimExpr root_pe = call->args[arg_offset + 2];

    Array<Stmt> stmts;

    // Use NVSHMEM's broadcast
    Call broadcast = Call(
        DataType::Handle(), builtin::call_extern(),
        {StringImm("handle"), StringImm("nvshmemx_float_broadcast_block"),
         buf_ptr, buf_ptr, nelems, root_pe,
         IntImm(DataType::Int(32), NVSHMEM_TEAM_WORLD)});
    stmts.push_back(Evaluate(broadcast));

    Call barrier = Call(DataType::Handle(), nvshmem_barrier_all_block(), {});
    stmts.push_back(Evaluate(barrier));

    return SeqStmt(stmts);
  }

  /*!
   * \brief Lower team-based collective operations.
   */
  Stmt LowerTeamCollective(const CallNode* call, const std::string& func_name) {
    size_t arg_offset = 2;
    if (call->args.size() < arg_offset + 3) {
      return Evaluate(GetRef<Call>(call));
    }

    PrimExpr buf_ptr = call->args[arg_offset];
    PrimExpr nelems = call->args[arg_offset + 1];
    PrimExpr team_id = call->args[arg_offset + 2];

    Array<Stmt> stmts;

    if (func_name.find("allreduce") != std::string::npos) {
      Call reduce = Call(
          DataType::Handle(), builtin::call_extern(),
          {StringImm("handle"), StringImm("nvshmemx_float_sum_reduce_block"),
           buf_ptr, buf_ptr, nelems, team_id});
      stmts.push_back(Evaluate(reduce));
    } else if (func_name.find("allgather") != std::string::npos) {
      PrimExpr src_ptr = call->args[arg_offset + 1];
      PrimExpr dst_ptr = call->args[arg_offset];
      PrimExpr src_nelems = call->args[arg_offset + 2];

      Call fcollect = Call(
          DataType::Handle(), builtin::call_extern(),
          {StringImm("handle"), StringImm("nvshmemx_float_fcollect_block"),
           dst_ptr, src_ptr, src_nelems, team_id});
      stmts.push_back(Evaluate(fcollect));
    } else if (func_name.find("broadcast") != std::string::npos) {
      PrimExpr root_pe = (call->args.size() > arg_offset + 3)
                             ? call->args[arg_offset + 3]
                             : IntImm(DataType::Int(32), 0);

      Call broadcast = Call(
          DataType::Handle(), builtin::call_extern(),
          {StringImm("handle"), StringImm("nvshmemx_float_broadcast_block"),
           buf_ptr, buf_ptr, nelems, root_pe, team_id});
      stmts.push_back(Evaluate(broadcast));
    }

    // Team sync
    Call team_sync = Call(
        DataType::Handle(), builtin::call_extern(),
        {StringImm("handle"), StringImm("nvshmemx_team_sync_block"), team_id});
    stmts.push_back(Evaluate(team_sync));

    return SeqStmt(stmts);
  }

  /*!
   * \brief Lower all-to-all personalized exchange.
   */
  Stmt LowerAllToAll(const CallNode* call, const std::string& algorithm) {
    size_t arg_offset = 2;
    if (call->args.size() < arg_offset + 3) {
      return Evaluate(GetRef<Call>(call));
    }

    PrimExpr dst_ptr = call->args[arg_offset];
    PrimExpr src_ptr = call->args[arg_offset + 1];
    PrimExpr total_nelems = call->args[arg_offset + 2];

    Array<Stmt> stmts;

    // All-to-all is implemented as a runtime helper
    Call alltoall = Call(
        DataType::Handle(), builtin::call_extern(),
        {StringImm("handle"), StringImm("tl_nvshmem_alltoall"),
         dst_ptr, src_ptr, total_nelems});
    stmts.push_back(Evaluate(alltoall));

    Call barrier = Call(DataType::Handle(), nvshmem_barrier_all_block(), {});
    stmts.push_back(Evaluate(barrier));

    return SeqStmt(stmts);
  }
};

namespace transform {
using namespace tir::transform;

/*!
 * \brief Create CollectiveLoweringPass.
 *
 * This pass expands high-level collective operations (allreduce, allgather,
 * reduce_scatter, broadcast) to low-level NVSHMEM primitives.
 */
tvm::transform::Pass CollectiveLowering() {
  auto pass_func = [](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    auto* n = f.CopyOnWrite();
    CollectiveLoweringMutator mutator;
    n->body = mutator(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.CollectiveLowering", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.CollectiveLowering", CollectiveLowering);
}

}  // namespace transform
}  // namespace tl
}  // namespace tvm
