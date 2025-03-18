#include <ATen/EmptyTensor.h>
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <torch/library.h>
#include <ATen/native/Resize.h>
#include <ATen//tt/TTDevice.h>
// #include <ATen/native/mps/Copy.h>

namespace at { namespace detail {

TensorBase empty_tt(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  auto* allocator = at::tt::GetTTAllocator();
  int64_t nelements = c10::multiply_integers(size);
  auto dtype = dtype_or_default(dtype_opt);
  
  auto dtype_meta = scalarTypeToTypeMeta(dtype);
  int64_t size_bytes = nelements * dtype_meta.itemsize();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        size_bytes,
        allocator->allocate(size_bytes),
        allocator,
        /*resizeable=*/true);
  auto tensor =
        detail::make_tensor<TensorImpl>(storage_impl, DispatchKey::TT, dtype_meta);
  return tensor;
}

TensorBase empty_strided_tt(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    c10::optional<Device> device_opt) {
  auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_tt());
  const DeviceGuard device_guard(device);
  auto* allocator = at::tt::GetTTAllocator();
  constexpr c10::DispatchKeySet tt_dks(c10::DispatchKey::TT);
  return at::detail::empty_strided_generic(
      size, stride, allocator, tt_dks, dtype);
}

TensorBase empty_strided_tt(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options) {
  return at::native::empty_strided_tt(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

} // namespace detail
} // namespace at
