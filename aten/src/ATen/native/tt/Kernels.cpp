#include <ATen/ops/add_native.h>
#include <ATen/tt/TTDevice.h>

#include <tt-metalium/host_api.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/storage.hpp>
#include <ttnn/operations/eltwise/binary_ng/binary_ng.hpp>

using namespace tt::tt_metal;

namespace at::native {

static ::tt::tt_metal::Tensor convert(const at::Tensor & tensor) {
  auto* allocator = tt::GetTTAllocator();
  auto storage = ::tt::tt_metal::DeviceStorage(allocator->get_buffer(tensor.mutable_data_ptr()));
  auto shape = ttnn::Shape({static_cast<unsigned int>(tensor.numel())});
  auto dtype = ::tt::tt_metal::DataType::FLOAT32;
  auto layout = ::tt::tt_metal::Layout::ROW_MAJOR;
  return ::tt::tt_metal::Tensor(storage, shape, dtype, layout);
}

at::Tensor & add_out_tt(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  
  auto a = convert(self);
  auto b = convert(other);
  auto c = convert(out);

  ttnn::operations::binary_ng::BinaryNg<ttnn::operations::binary_ng::BinaryOpType::ADD>::invoke(ttnn::DefaultQueueId, a, b, std::nullopt, std::nullopt, c);
  
  return out;
}

// static void sum_kernel_tt(TensorIterator& iter) {
// }

// REGISTER_TT_DISPATCH(sum_stub, &sum_kernel_tt)

}
