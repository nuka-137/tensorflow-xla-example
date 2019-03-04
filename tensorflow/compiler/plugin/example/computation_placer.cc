#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/plugin/example/platform_id.h"

static std::unique_ptr<xla::ComputationPlacer> CreateComputationPlacer() {
  return absl::make_unique<xla::ComputationPlacer>();
}

static bool InitModule() {
  xla::ComputationPlacer::RegisterComputationPlacer(
      stream_executor::mydevplugin::kMyDevPlatformId, &CreateComputationPlacer);
  return true;
}

static bool module_initialized = InitModule();

