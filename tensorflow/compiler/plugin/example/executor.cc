#include "tensorflow/compiler/plugin/example/executor.h"
#include "tensorflow/compiler/plugin/example/platform_id.h"

#include "tensorflow/core/platform/profile_utils/cpu_utils.h"


namespace stream_executor {
namespace mydevplugin {

host::HostStream *AsMyDevStream(Stream *stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<host::HostStream *>(stream->implementation());
}

MyDevExecutor::MyDevExecutor(const PluginConfig &plugin_config)
    : plugin_config_(plugin_config) {}

MyDevExecutor::~MyDevExecutor() {}

void *MyDevExecutor::Allocate(uint64 size) { return new char[size]; }

void *MyDevExecutor::AllocateSubBuffer(DeviceMemoryBase *parent,
                                       uint64 offset_bytes, uint64 size_bytes) {
  return reinterpret_cast<char *>(parent->opaque()) + offset_bytes;
}

void MyDevExecutor::Deallocate(DeviceMemoryBase *mem) {
  if (!mem->is_sub_buffer()) {
    delete[] static_cast<char *>(mem->opaque());
  }
}

bool MyDevExecutor::SynchronousMemZero(DeviceMemoryBase *location, uint64 size) {
  memset(location->opaque(), 0, size);
  return true;
}

bool MyDevExecutor::SynchronousMemSet(DeviceMemoryBase *location, int value,
                                      uint64 size) {
  memset(location->opaque(), value, size);
  return true;
}

bool MyDevExecutor::Memcpy(Stream *stream, void *host_dst,
                           const DeviceMemoryBase &dev_src, uint64 size) {
  void *src_mem = const_cast<void *>(dev_src.opaque());
  AsMyDevStream(stream)->EnqueueTask(
      [host_dst, src_mem, size]() { memcpy(host_dst, src_mem, size); });
  return true;
}

bool MyDevExecutor::Memcpy(Stream *stream, DeviceMemoryBase *dev_dst,
                           const void *host_src, uint64 size) {
  void *dst_mem = dev_dst->opaque();
  AsMyDevStream(stream)->EnqueueTask(
      [dst_mem, host_src, size]() { memcpy(dst_mem, host_src, size); });
  return true;
}

bool MyDevExecutor::MemcpyDeviceToDevice(Stream *stream,
                                         DeviceMemoryBase *dev_dst,
                                         const DeviceMemoryBase &dev_src,
                                         uint64 size) {
  void *dst_mem = dev_dst->opaque();
  void *src_mem = const_cast<void *>(dev_src.opaque());
  AsMyDevStream(stream)->EnqueueTask(
      [src_mem, dst_mem, size]() { memcpy(dst_mem, src_mem, size); });
  return true;
}

bool MyDevExecutor::MemZero(Stream *stream, DeviceMemoryBase *location,
                            uint64 size) {
  void *dev_mem = location->opaque();
  AsMyDevStream(stream)->EnqueueTask(
      [dev_mem, size]() { memset(dev_mem, 0, size); });
  return true;
}

bool MyDevExecutor::Memset(Stream *stream, DeviceMemoryBase *location,
                           uint8 pattern, uint64 size) {
  void *dev_mem = location->opaque();
  AsMyDevStream(stream)->EnqueueTask(
      [dev_mem, size, pattern]() { memset(dev_mem, pattern, size); });
  return true;
}

bool MyDevExecutor::Memset32(Stream *stream, DeviceMemoryBase *location,
                             uint32 pattern, uint64 size) {
  void *dev_mem = location->opaque();
  AsMyDevStream(stream)->EnqueueTask(
      [dev_mem, size, pattern]() { memset(dev_mem, pattern, size); });
  return true;
}

port::Status MyDevExecutor::SynchronousMemcpy(DeviceMemoryBase *dev_dst,
                                              const void *host_src,
                                              uint64 size) {
  memcpy(dev_dst->opaque(), host_src, size);
  return port::Status::OK();
}

port::Status MyDevExecutor::SynchronousMemcpy(void *host_dst,
                                              const DeviceMemoryBase &dev_src,
                                              uint64 size) {
  memcpy(host_dst, dev_src.opaque(), size);
  return port::Status::OK();
}

port::Status MyDevExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase *dev_dst, const DeviceMemoryBase &dev_src, uint64 size) {
  memcpy(dev_dst->opaque(), dev_src.opaque(), size);
  return port::Status::OK();
}

bool MyDevExecutor::HostCallback(Stream *stream,
                                 std::function<port::Status()> callback) {
  AsMyDevStream(stream)->EnqueueTask([callback]() {
    port::Status s = callback();
    if (!s.ok()) {
      LOG(WARNING) << "Host callback failed: " << s;
    }
  });
  return true;
}

bool MyDevExecutor::AllocateStream(Stream *stream) { return true; }

void MyDevExecutor::DeallocateStream(Stream *stream) {}

bool MyDevExecutor::CreateStreamDependency(Stream *dependent, Stream *other) {
  AsMyDevStream(dependent)->EnqueueTask(
      [other]() { SE_CHECK_OK(other->BlockHostUntilDone()); });
  AsMyDevStream(dependent)->BlockUntilDone();
  return true;
}

bool MyDevExecutor::StartTimer(Stream *stream, Timer *timer) {
  dynamic_cast<host::HostTimer *>(timer->implementation())->Start(stream);
  return true;
}

bool MyDevExecutor::StopTimer(Stream *stream, Timer *timer) {
  dynamic_cast<host::HostTimer *>(timer->implementation())->Stop(stream);
  return true;
}

port::Status MyDevExecutor::BlockHostUntilDone(Stream *stream) {
  AsMyDevStream(stream)->BlockUntilDone();
  return port::Status::OK();
}

DeviceDescription *MyDevExecutor::PopulateDeviceDescription() const {
  internal::DeviceDescriptionBuilder builder;

  builder.set_name("MyDev");
  builder.set_device_vendor("nuka");
  builder.set_platform_version("1.0");
  builder.set_driver_version("1.0");
  builder.set_runtime_version("1.0");
  builder.set_device_address_bits(64);
  builder.set_device_memory_size(static_cast<uint64>(4) * 1024 * 1024 * 1024);

  float cycle_counter_frequency = static_cast<float>(
      tensorflow::profile_utils::CpuUtils::GetCycleCounterFrequency());
  builder.set_clock_rate_ghz(cycle_counter_frequency / 1e9);

  auto built = builder.Build();
  return built.release();
}

bool MyDevExecutor::SupportsBlas() const {
  return PluginRegistry::Instance()
      ->GetFactory<PluginRegistry::BlasFactory>(kMyDevPlatformId,
                                                plugin_config_.blas())
      .ok();
}

blas::BlasSupport *MyDevExecutor::CreateBlas() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::BlasFactory> status =
      registry->GetFactory<PluginRegistry::BlasFactory>(kMyDevPlatformId,
                                                        plugin_config_.blas());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve BLAS factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

bool MyDevExecutor::SupportsFft() const {
  return PluginRegistry::Instance()
      ->GetFactory<PluginRegistry::FftFactory>(kMyDevPlatformId,
                                               plugin_config_.fft())
      .ok();
}

fft::FftSupport *MyDevExecutor::CreateFft() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::FftFactory> status =
      registry->GetFactory<PluginRegistry::FftFactory>(kMyDevPlatformId,
                                                       plugin_config_.fft());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve FFT factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

bool MyDevExecutor::SupportsRng() const {
  return PluginRegistry::Instance()
      ->GetFactory<PluginRegistry::RngFactory>(kMyDevPlatformId,
                                               plugin_config_.rng())
      .ok();
}

rng::RngSupport *MyDevExecutor::CreateRng() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::RngFactory> status =
      registry->GetFactory<PluginRegistry::RngFactory>(kMyDevPlatformId,
                                                       plugin_config_.rng());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve RNG factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

}  // namespace mydevplugin
}  // namespace stream_executor
