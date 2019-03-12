#ifndef TENSORFLOW_COMPILER_EXAMPLE_TRANSFER_MANAGER_H_
#define TENSORFLOW_COMPILER_EXAMPLE_TRANSFER_MANAGER_H_

#include "tensorflow/compiler/xla/service/generic_transfer_manager.h"


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
