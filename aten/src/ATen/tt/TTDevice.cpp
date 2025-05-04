#include <ATen/tt/TTDevice.h>

#include <c10/util/Logging.h>
#include <c10/util/Registry.h>

#include <tt-metalium/host_api.hpp>

#include <ATen/ATen.h>
#include <ATen/native/tt/Kernels.h>

using namespace tt;
using namespace tt::tt_metal;

namespace at::tt {

TTAllocator::TTAllocator() {
  device_ = CreateDevice(0);
}

IntArrayRef TTAllocator::getBufferShape(const void* ptr) const {
  return IntArrayRef();
}

static std::shared_ptr<Buffer> MakeBuffer(IDevice* device, uint32_t size, uint32_t page_size, BufferType buffer_type) {
  InterleavedBufferConfig config{
      .device = device,
      .size = size,
      .page_size = page_size,
      .buffer_type = buffer_type
  };
  return CreateBuffer(config);
}

DataPtr TTAllocator::allocate(size_t n) {
  AT_ASSERT(false);
}

DataPtr TTAllocator::allocate_with_page_size(size_t n, size_t page_size_bytes) {
  LOG(INFO) << "allocating " << n << " bytes with page size " << page_size_bytes << " bytes.";
  auto buffer = MakeBuffer(device_, n, page_size_bytes, BufferType::DRAM);
  auto address = reinterpret_cast<void*>(buffer->address());
  buffers_[address] = std::move(buffer);
  return DataPtr(address, DeviceType::TT);
}

void TTAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
  AT_ASSERT(false);
}

std::shared_ptr<Buffer> TTAllocator::get_buffer(const at::Tensor& tensor) const {
  auto it = buffers_.find(tensor.data_ptr());
  if (it == buffers_.end()) {
    AT_ASSERT(tensor.storage_offset() != 0);
    auto src_buf = get_buffer(tensor.storage().data_ptr());
    // In this case we make a copy of the tensor starting at storage_offset.
    // Once we make the kernels support such offsets, this copy can be removed.
    auto new_tensor = at::empty_like(tensor);
    native::MemcpyFromOffset(new_tensor, src_buf, tensor.element_size() * tensor.storage_offset());
    return get_buffer(new_tensor);
  }
  return it->second;
}

TTAllocator* GetTTAllocator(bool useSharedAllocator) {
  static TTAllocator allocator;
  return &allocator;
}
  
}  // namespace at::tt
