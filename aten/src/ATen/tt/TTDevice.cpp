#include <ATen/tt/TTDevice.h>

#include <c10/util/Registry.h>

#include <tt-metalium/host_api.hpp>

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
  std::cout << "allocating " << n << std::endl;
  auto buffer = MakeBuffer(device_, n, n, BufferType::DRAM);
  auto address = reinterpret_cast<void*>(buffer->address());
  buffers_[address] = std::move(buffer);
  return DataPtr(address, DeviceType::TT);
}

void TTAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
}

std::shared_ptr<Buffer> TTAllocator::get_buffer(void* data) const {
  auto it = buffers_.find(data);
  AT_ASSERT(it != buffers_.end());
  return it->second;
}

TTAllocator* GetTTAllocator(bool useSharedAllocator) {
  static TTAllocator allocator;
  return &allocator;
}
  
}  // namespace at::tt
