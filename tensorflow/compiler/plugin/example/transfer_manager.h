#ifndef TENSORFLOW_COMPILER_EXAMPLE_TRANSFER_MANAGER_H_
#define TENSORFLOW_COMPILER_EXAMPLE_TRANSFER_MANAGER_H_

#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/cpu/xfeed_manager.h"
#include "tensorflow/compiler/xla/service/generic_transfer_manager.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace mydevplugin {

class MyDevTransferManager : public GenericTransferManager {
 public:
  MyDevTransferManager();
  ~MyDevTransferManager() override {}

  Status TransferLiteralToInfeed(se::StreamExecutor* executor,
                                 const LiteralSlice& literal) override;
  Status TransferLiteralFromOutfeed(se::StreamExecutor* executor,
                                    const Shape& literal_shape,
                                    MutableBorrowingLiteral literal) override;

  Status ResetDevices(absl::Span<stream_executor::StreamExecutor* const> executors) override;
 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MyDevTransferManager);
};

}  // namespace mydevplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_EXAMPLE_TRANSFER_MANAGER_H_
