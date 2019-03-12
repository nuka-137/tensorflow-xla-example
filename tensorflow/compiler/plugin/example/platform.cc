#include "tensorflow/compiler/plugin/example/platform.h"
#include "tensorflow/compiler/plugin/example/platform_id.h"
#include "tensorflow/compiler/plugin/example/executor.h"

#include "tensorflow/stream_executor/lib/ptr_util.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"


namespace stream_executor {
namespace mydevplugin {

PLATFORM_DEFINE_ID(kMyDevPlatformId);

MyDevPlatform::MyDevPlatform() : name_("MyDev") {}

MyDevPlatform::~MyDevPlatform() {}

Platform::Id MyDevPlatform::id() const { return kMyDevPlatformId; }

int MyDevPlatform::VisibleDeviceCount() const {
  return 1;
}

const string& MyDevPlatform::Name() const { return name_; }


port::StatusOr<StreamExecutor*> MyDevPlatform::ExecutorForDevice(int ordinal) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = PluginConfig();
  config.device_options = DeviceOptions::Default();

  return GetExecutor(config);
}


port::StatusOr<StreamExecutor*> MyDevPlatform::ExecutorForDeviceWithPluginConfig(
    int device_ordinal, const PluginConfig& plugin_config) {
  StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  config.plugin_config = plugin_config;
  config.device_options = DeviceOptions::Default();

  return GetExecutor(config);
}


port::StatusOr<StreamExecutor*> MyDevPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}


port::StatusOr<std::unique_ptr<StreamExecutor>>
MyDevPlatform::GetUncachedExecutor(const StreamExecutorConfig& config) {
  auto executor = MakeUnique<StreamExecutor>(
      this, MakeUnique<MyDevExecutor>(config.plugin_config));
  auto init_status = executor->Init(config.ordinal, config.device_options);
  if (!init_status.ok()) {
    return port::Status(
        port::error::INTERNAL,
        port::Printf(
          "failed initializing StreamExecutor for device ordinal %d: %s",
          config.ordinal, init_status.ToString().c_str()));
  }

  return std::move(executor);
}


void MyDevPlatform::RegisterTraceListener(
    std::unique_ptr<TraceListener> listener) {
  LOG(FATAL) << "unimplemented: MyDevPlatform::RegisterTraceListener";
}


void MyDevPlatform::UnregisterTraceListener(TraceListener* listener) {
  LOG(FATAL) << "unimplemented: MyDevPlatform::UnRegisterTraceListener";
}


static void InitializeMyDevPlatform() {
  std::unique_ptr<Platform> platform(new MyDevPlatform);
  SE_CHECK_OK(MultiPlatformManager::RegisterPlatform(std::move(platform)));
}

} // mydevplugin
} // stream_executor


REGISTER_MODULE_INITIALIZER(
    mydev_platform, stream_executor::mydevplugin::InitializeMyDevPlatform());
REGISTER_MODULE_INITIALIZER_SEQUENCE(mydev_platform, multi_platform_manager);
REGISTER_MODULE_INITIALIZER_SEQUENCE(multi_platform_manager_listener, mydev_platform);
