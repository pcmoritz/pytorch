#include <ATen/tt/Context.h>
#include <torch/script.h>

namespace at {
namespace native {
namespace tt {

/*
at::Tensor& copy_to_tt_(at::Tensor& dst, const at::Tensor& src) {
  TORCH_INTERNAL_ASSERT(
      dst.device().type() == DeviceType::TT,
      "copy_to_tt_ output tensor's device is not tt");
  TORCH_INTERNAL_ASSERT(
      src.device().type() == DeviceType::CPU,
      "copy_to_tt_ is implemented only for CPU device input");
  TORCH_INTERNAL_ASSERT(
      src.layout() == Layout::Strided,
      "copy_to_tt_ is implemented only for Strided layout input");
  TORCH_INTERNAL_ASSERT(
      src.scalar_type() == ScalarType::Float,
      "copy_to_tt_ is implemented only for float dtype");
  auto cpu_tensor_contiguous = src.contiguous();
  TTTensor& tttensor = TTTensor::fromTensor(dst);
  tttensor.set_data_from_host(cpu_tensor_contiguous.data_ptr<float>());
  return dst;
}
*/

} // namespace tt
} // namespace native

/*
struct TTImpl : public at::tt::TTInterface {
  bool is_tt_available() const override {
    return true;
  }
  at::Tensor& tt_copy_(at::Tensor& input, const at::Tensor& src)
      const override {
    TORCH_CHECK(
        is_tt_available(), "TT is not available on the current device");
    return native::tt::tt_copy_impl_(input, src);
  }
};

static at::tt::TTImplRegistrar g_tt_impl(new TTImpl());
*/

} // namespace at
