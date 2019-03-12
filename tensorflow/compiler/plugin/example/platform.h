#ifndef TENSORFLOW_COMPILER_PLUGIN_MYDEV_PLATFORM_H_
#define TENSORFLOW_COMPILER_PLUGIN_MYDEV_PLATFORM_H_

#include "tensorflow/stream_executor/executor_cache.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/mutex.h"


namespace stream_executor {
namespace mydevplugin {

class MyDevPlatform : public Platform {
 public:
  MyDevPlatform();
  ~MyDevPlatform() override;

  Platform::Id id() const override;

  int VisibleDeviceCount() const override;

  const string& Name() const override;

  port::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override;

  port::StatusOr<StreamExecutor*> ExecutorForDeviceWithPluginConfig(
      int ordinal, const PluginConfig& config) override;

  port::StatusOr<StreamExecutor*> GetExecutor(
      const StreamExecutorConfig& config) override;

  port::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      const StreamExecutorConfig& config) override;

  void RegisterTraceListener(std::unique_ptr<TraceListener> listener) override;

  void UnregisterTraceListener(TraceListener* listener) override;

 private:
  string name_;
  mutable mutex executors_mutex_;
  ExecutorCache executor_cache_;
  SE_DISALLOW_COPY_AND_ASSIGN(MyDevPlatform);
};

} // mydevplugin
} // stream_executor

#endif  // TENSORFLOW_COMPILER_PLUGIN_MYDEV_PLATFORM_H_
