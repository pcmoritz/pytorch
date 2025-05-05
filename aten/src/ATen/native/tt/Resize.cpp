#include <ATen/ops/resize_native.h>

#include <tt-metalium/host_api.hpp>

#include <ATen/native/ResizeCommon.h>
#include <ATen/tt/TTDevice.h>

namespace at::native {

static void resize_bytes_tt(StorageImpl* storage, size_t size_bytes, size_t page_size) {
  TORCH_CHECK(storage->resizable(), "Trying to resize storage that is not resizable");
  // auto allocator = storage->allocator();
  auto* allocator = at::tt::GetTTAllocator();
  TORCH_CHECK(allocator != nullptr, "Trying to resize storage without an allocator");

  c10::Device device = storage->device();

  if (size_bytes == 0) {
    storage->set_data_ptr_noswap(at::DataPtr(nullptr, device));
    storage->set_nbytes(0);
    return;
  }

  // c10::cuda::CUDAGuard guard(device.index());
  at::DataPtr data = allocator->allocate_with_page_size(size_bytes, page_size);
  // if (storage->data_ptr()) {
  //  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  //
  // TODO(pcm): Adapt MemcpyFromOffset to be able to handle an offset for both source
  // and destination and replace the following call.
  //
  //  C10_CUDA_CHECK(
  //      cudaMemcpyAsync(
  //          data.get(),
  //          storage->data(),
  //          std::min(storage->nbytes(), size_bytes),
  //          cudaMemcpyDeviceToDevice,
  //          c10::cuda::getCurrentCUDAStream()));
  // }

  // Destructively overwrite data_ptr
  storage->set_data_ptr_noswap(std::move(data));
  storage->set_nbytes(size_bytes);
}

static inline void maybe_resize_storage_tt(TensorImpl* self, size_t new_size_bytes, size_t page_size) {
  // It does not make sense to try to resize a storage
  // to hold 0 elements, and this can break
  // if storage_offset is positive but
  // new_size is 0, so just bail in that case
  // (same comment is in Resize.h)
  if (self->numel() == 0) {
    return;
  }

  const Storage &storage = self->unsafe_storage();
  TORCH_CHECK(storage, "Tensor: invalid null storage");
  if (new_size_bytes > storage.nbytes()) {
    resize_bytes_tt(storage.unsafeGetStorageImpl(), new_size_bytes, page_size);
  }
}

inline TensorImpl* resize_impl_tt_(
    TensorImpl* self,
    IntArrayRef size,
    at::OptionalIntArrayRef stride) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }
  const auto itemsize = self->dtype().itemsize();
  const auto storage_offset = self->storage_offset();
  size_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    storage_size = at::detail::computeStorageNbytes(
        size, *stride, itemsize, storage_offset);
  } else {
    self->set_sizes_contiguous(size);
    storage_size = at::detail::computeStorageNbytesContiguous(
        size, itemsize, storage_offset);
  }
  // Make sure the memory is padded to the tile size
  size_t tile_size_bytes = ::tt::constants::TILE_HW * itemsize;
  storage_size = ((storage_size + tile_size_bytes - 1) / tile_size_bytes) * tile_size_bytes;
  size_t page_size = itemsize * ::tt::constants::FACE_WIDTH;
  maybe_resize_storage_tt(self, storage_size, page_size);

  return self;
}

const Tensor& resize_tt_(
    const Tensor& self,
    IntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  // if (self.has_names()) {
  //   return resize_named_tensor_(self, size, optional_memory_format);
  // }
  auto* self_ = self.unsafeGetTensorImpl();
  auto old_storage_nbytes = self_->unsafe_storage() ? self_->unsafe_storage().nbytes() : 0;
  resize_impl_tt_(self_, size, /*stride=*/std::nullopt);
  AT_ASSERT(!optional_memory_format.has_value());

  // See Note [Enabling Deterministic Operations]
  // if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
  //  at::native::fill_resize_deterministic_(self, static_cast<int64_t>(old_storage_nbytes));
  // }
  return self;
}

}
