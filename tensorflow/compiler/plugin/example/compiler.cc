#include "tensorflow/compiler/plugin/example/compiler.h"
#include "tensorflow/compiler/plugin/example/platform_id.h"
#include "tensorflow/compiler/plugin/example/executable.h"

#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"


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
  TF_RET_CHECK(stream_exec != nullptr);

  VLOG(1) << "Run MyDev backend " << module->name();

  auto evaluator = absl::make_unique<HloEvaluator>();

  std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data;
  std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map;

  std::unique_ptr<Executable> executable =
    absl::make_unique<MyDevExecutable>(std::move(module),
                                       std::move(hlo_profile_printer_data),
                                       std::move(hlo_profile_index_map));

  return std::move(executable);
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
