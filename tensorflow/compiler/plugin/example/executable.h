#ifndef TENSORFLOW_COMPILER_PLUGIN_MYDEV_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_PLUGIN_MYDEV_EXECUTABLE_H_

#include "tensorflow/compiler/xla/service/executable.h"

namespace xla {
namespace mydevplugin {

class MyDevExecutable : public Executable {
 public:
  MyDevExecutable(std::unique_ptr<HloModule> hlo_module,
                  std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
                  std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map);
  ~MyDevExecutable() override;

  StatusOr<ScopedShapedBuffer> ExecuteOnStream(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<const ShapedBuffer* const> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  StatusOr<ScopedShapedBuffer> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<const ShapedBuffer* const> arguments) override;
};

}  // namespace mydevplugin
}  // namespace xla

#endif
