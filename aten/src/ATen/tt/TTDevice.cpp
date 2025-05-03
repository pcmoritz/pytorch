#include <ATen/tt/TTDevice.h>

#include <c10/util/Logging.h>
#include <c10/util/Registry.h>

#include <tt-metalium/host_api.hpp>

#include <ATen/ATen.h>

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
  AT_ASSERT(it != buffers_.end());
  return it->second;
}

TTAllocator* GetTTAllocator(bool useSharedAllocator) {
  static TTAllocator allocator;
  return &allocator;
}
  
}  // namespace at::tt
