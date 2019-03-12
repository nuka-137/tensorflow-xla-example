#include "tensorflow/compiler/plugin/example/executable.h"

namespace xla {
namespace mydevplugin {

MyDevExecutable::MyDevExecutable(
        std::unique_ptr<HloModule> hlo_module,
        std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
        std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map)
    : Executable(std::move(hlo_module),
                 std::move(hlo_profile_printer_data),
                 std::move(hlo_profile_index_map)) {
}

MyDevExecutable::~MyDevExecutable() {}

StatusOr<ScopedShapedBuffer> MyDevExecutable::ExecuteOnStream(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<const ShapedBuffer* const> arguments,
      HloExecutionProfile* hlo_execution_profile) {
  return tensorflow::errors::Unimplemented("ExecuteOnStream is not yet supported.");
}

StatusOr<ScopedShapedBuffer> MyDevExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments) {
  return tensorflow::errors::Unimplemented("ExecuteOnStream is not yet supported.");
}

}  // namespace mydevplugin
}  // namespace xla
