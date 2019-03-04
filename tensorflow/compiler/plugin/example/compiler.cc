#include "tensorflow/compiler/plugin/example/compiler.h"
#include "tensorflow/compiler/plugin/example/platform_id.h"

#include <stddef.h>
#include <string.h>
#include <map>
#include <mutex>  // NOLINT(build/c++11): only using std::call_once, not mutex.
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/batch_dot_simplification.h"
#include "tensorflow/compiler/xla/service/batchnorm_expander.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/buffer_liveness.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/conditional_simplifier.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_element_type_converter.h"
#include "tensorflow/compiler/xla/service/hlo_get_dimension_size_rewriter.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/indexed_array_analysis.h"
#include "tensorflow/compiler/xla/service/map_inliner.h"
#include "tensorflow/compiler/xla/service/reduce_precision_insertion.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/scatter_expander.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
#include "tensorflow/compiler/xla/service/while_loop_invariant_code_motion.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace mydevplugin {

MyDevCompiler::MyDevCompiler() {}

stream_executor::Platform::Id MyDevCompiler::PlatformId() const {
  return stream_executor::mydevplugin::kMyDevPlatformId;
}

StatusOr<std::unique_ptr<HloModule>> MyDevCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, stream_executor::StreamExecutor* stream_exec,
    DeviceMemoryAllocator* device_allocator) {
  HloPassPipeline pipeline("MyDev HLO Optimization");

  pipeline.Run(module.get()).status();

  return std::move(module);
}

Status MyDevCompiler::RunHloPassesOnModuleGroup(
    HloModuleGroup* module_group,
    absl::Span<stream_executor::StreamExecutor* const> executors,
    DeviceMemoryAllocator* device_allocator) {
  return Unimplemented("Model partitioning not implemented");
}

StatusOr<std::unique_ptr<Executable>> MyDevCompiler::RunBackend(
    std::unique_ptr<HloModule> module, stream_executor::StreamExecutor* stream_exec,
    DeviceMemoryAllocator* device_allocator) {
  return Unimplemented("Need to implement");
}

StatusOr<std::vector<std::unique_ptr<Executable>>> MyDevCompiler::RunBackendOnModuleGroup(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<stream_executor::StreamExecutor*>> stream_exec,
    DeviceMemoryAllocator* device_allocator) {
  return Unimplemented("Model partitioning not implemented");
}

StatusOr<std::vector<std::unique_ptr<Executable>>> MyDevCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<stream_executor::StreamExecutor*>> stream_exec,
    DeviceMemoryAllocator* device_allocator) {
  return Unimplemented("Need to implement");
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
MyDevCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                   const AotCompilationOptions& options) {
  return Unimplemented("Need to implement");
}


int64 ShapeSizeBytes(const Shape& shape) {
  if (shape.IsOpaque()) {
    return sizeof(void*);
  }
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}

HloCostAnalysis::ShapeSizeFunction MyDevCompiler::ShapeSizeBytesFunction() const {
  return ShapeSizeBytes;
}

}  // namespace mydevplugin
}  // namespace xla

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(
      stream_executor::mydevplugin::kMyDevPlatformId,
      []() { return absl::make_unique<xla::mydevplugin::MyDevCompiler>(); });
  return true;
}
static bool module_initialized = InitModule();
