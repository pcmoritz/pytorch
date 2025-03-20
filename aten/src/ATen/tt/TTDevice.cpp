#include <ATen/tt/TTDevice.h>

#include <ATen/core/ATen_fwd.h>
#include <c10/core/Allocator.h>
#include <c10/util/Registry.h>
#include <c10/core/Storage.h>

#include <tt-metalium/host_api.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace at::tt {

TTAllocator::TTAllocator() {
  device_ = CreateDevice(0);
}

virtual DataPtr TTAllocator::allocate(size_t n) {
  std::cout << "allocating " << n << std::endl;
  InterleavedBufferConfig config{
      .device = device_,
      .size = n,
      .page_size = n,
      .buffer_type = BufferType::DRAM
  };
  buffers_.push_back(CreateBuffer(config));
  return DataPtr(reinterpret_cast<void*>(buffers_.back()->address()), DeviceType::TT);
}

virtual void TTAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
  CommandQueue& cq = device_->command_queue();
  EnqueueWriteBuffer(cq, buffers_[0], src, false); // TODO(pcm): Fix this
  Finish(cq);
}

TTAllocator* GetTTAllocator(bool useSharedAllocator) {
  static TTAllocator allocator;
  return &allocator;
}
  
}  // namespace at::tt
