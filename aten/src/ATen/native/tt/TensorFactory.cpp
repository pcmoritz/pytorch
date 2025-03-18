#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <torch/library.h>

#include <ATen/tt/EmptyTensor.h>

namespace at::native {

Tensor empty_tt(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {

  return at::detail::empty_tt(size, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

Tensor empty_strided_tt(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  check_size_nonnegative(size);
  // empty memory formatempty
  auto t = at::native::empty_tt(
      {0},
      dtype_opt,
      layout_opt,
      device_opt,
      pin_memory_opt);
  // resize_impl_mps_(t.unsafeGetTensorImpl(), size, stride);
  return t;
}
  
}
