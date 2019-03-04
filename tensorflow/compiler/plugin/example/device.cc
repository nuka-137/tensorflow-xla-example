#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/jit/xla_compile_on_demand_op.h"
#include "tensorflow/compiler/jit/kernels/xla_ops.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"

namespace tensorflow {

const char* const DEVICE_XLA_MYDEV = "XLA_MYDEV";
const char* const DEVICE_MYDEV_XLA_JIT = "MYDEV_XLA_JIT";

constexpr std::array<DataType, 5> kMyDevAllTypes = 
        {{DT_INT32, DT_FLOAT, DT_BOOL}};

class XlaMyDevFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& session_options,
                       const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override;
};

Status XlaMyDevFactory::CreateDevices(const SessionOptions& session_options,
                                      const string& name_prefix,
                                      std::vector<std::unique_ptr<Device>>* devices) {
  XlaOpRegistry::DeviceRegistration registration;
  registration.compilation_device_name = DEVICE_MYDEV_XLA_JIT;
  registration.autoclustering_policy = XlaOpRegistry::AutoclusteringPolicy::kAlways;
  registration.compile_resource_ops = true;
  XlaOpRegistry::RegisterCompilationDevice(DEVICE_XLA_MYDEV, registration);

  static XlaDeviceOpRegistrations* registrations =
    RegisterXlaDeviceKernels(DEVICE_XLA_MYDEV, DEVICE_MYDEV_XLA_JIT);
  (void)registrations;

  TF_ASSIGN_OR_RETURN(auto platform, stream_executor::MultiPlatformManager::PlatformWithName("MyDev"));

  XlaDevice::Options options;
  options.platform = platform;
  options.device_name_prefix = name_prefix;
  options.device_name = DEVICE_XLA_MYDEV;
  options.device_ordinal = 0;
  options.compilation_device_name = DEVICE_MYDEV_XLA_JIT;
  options.use_multiple_streams = false;

  auto device = absl::make_unique<XlaDevice>(session_options, options);

  devices->push_back(std::move(device));

  return Status::OK();
}


bool MyDevOpFilter(KernelDef* kdef) {
  if (kdef->op() == "Const") {
    AddDtypeToKernelDefConstraint("dtype", DT_STRING, kdef);
  }
  if (kdef->op() == "Assert") {
    AddDtypeToKernelDefConstraint("T", DT_STRING, kdef);
  }

  return true;
}


REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_MYDEV, XlaMyDevFactory, 210);

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_MYDEV, XlaLocalLaunchOp, kMyDevAllTypes);
REGISTER_XLA_COMPILE_KERNEL(DEVICE_XLA_MYDEV, XlaCompileOp, kMyDevAllTypes);
REGISTER_XLA_RUN_KERNEL(DEVICE_XLA_MYDEV, XlaRunOp, kMyDevAllTypes);

REGISTER_XLA_BACKEND(DEVICE_MYDEV_XLA_JIT, kMyDevAllTypes, MyDevOpFilter);

REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_MYDEV, kMyDevAllTypes);

}  // namespace tensorflow
