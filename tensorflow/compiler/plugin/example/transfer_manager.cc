#include "tensorflow/compiler/plugin/example/transfer_manager.h"
#include "tensorflow/compiler/plugin/example/platform_id.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace xla {
namespace mydevplugin {

MyDevTransferManager::MyDevTransferManager()
    : GenericTransferManager(stream_executor::mydevplugin::kMyDevPlatformId,
                             /*pointer_size=*/sizeof(void*)) {}

Status MyDevTransferManager::TransferLiteralToInfeed(
    se::StreamExecutor* executor, const LiteralSlice& literal) {
  const Shape& shape = literal.shape();

  VLOG(1) << "transferring literal shape to infeed: "
          << ShapeUtil::HumanString(shape);

  return Status::OK();
}

Status MyDevTransferManager::TransferLiteralFromOutfeed(
    se::StreamExecutor* executor, const Shape& literal_shape,
    MutableBorrowingLiteral literal) {
  const Shape& shape = literal_shape;
  VLOG(1) << "transferring literal shape from outfeed: "
          << ShapeUtil::HumanString(shape);
  return Status::OK();
}

Status MyDevTransferManager::ResetDevices(
    absl::Span<stream_executor::StreamExecutor* const> executors) {
  return Unimplemented("Device reset not supported");
}


}  // namespace mydevplugin
}  // namespace xla

static std::unique_ptr<xla::TransferManager> CreateMyDevTransferManager() {
  return absl::make_unique<xla::mydevplugin::MyDevTransferManager>();
}

static bool InitModule() {
  xla::TransferManager::RegisterTransferManager(
      stream_executor::mydevplugin::kMyDevPlatformId, &CreateMyDevTransferManager);
  return true;
}
static bool module_initialized = InitModule();
