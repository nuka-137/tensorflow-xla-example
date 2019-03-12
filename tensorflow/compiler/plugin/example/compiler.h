#ifndef TENSORFLOW_COMPILER_EXAMPLE_COMPILER_H_
#define TENSORFLOW_COMPILER_EXAMPLE_COMPILER_H_

#include "tensorflow/compiler/xla/service/compiler.h"


namespace xla {
namespace mydevplugin {

class MyDevCompiler : public Compiler {
 public:
  MyDevCompiler();
  ~MyDevCompiler() override {}

  se::Platform::Id PlatformId() const override;

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      DeviceMemoryAllocator* device_allocator) override;

  // Unimplemented
  Status RunHloPassesOnModuleGroup(
      HloModuleGroup* module_group,
      absl::Span<se::StreamExecutor* const> executors,
      DeviceMemoryAllocator* device_allocator) override;

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      DeviceMemoryAllocator* device_allocator) override;

  // Unimplemented
  StatusOr<std::vector<std::unique_ptr<Executable>>> RunBackendOnModuleGroup(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_exec,
      DeviceMemoryAllocator* device_allocator) override;

  StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_exec,
      DeviceMemoryAllocator* device_allocator) override;

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     const AotCompilationOptions& options) override;


  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

 private:

  TF_DISALLOW_COPY_AND_ASSIGN(MyDevCompiler);
};

}  // namespace mydevplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_EXAMPLE_COMPILER_H_
