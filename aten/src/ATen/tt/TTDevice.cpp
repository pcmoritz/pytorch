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

DataPtr TTAllocator::allocate(size_t n) {
  std::cout << "allocating " << n << std::endl;
  InterleavedBufferConfig config{
      .device = device_,
      .size = n,
      .page_size = n,
      .buffer_type = BufferType::DRAM
  };
  auto buffer = CreateBuffer(config);
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
