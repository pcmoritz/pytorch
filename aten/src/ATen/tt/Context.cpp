#include <atomic>

#include <ATen/Tensor.h>
#include <ATen/tt/Context.h>
#include <ATen/tt/TTDevice.h>

#include <tt-metalium/host_api.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace at {
namespace tt {

at::Tensor& tt_copy_(at::Tensor& self, const at::Tensor& src) {
  auto* allocator = GetTTAllocator();
  CommandQueue& cq = allocator->device()->command_queue();
  if (src.device().type() == at::kCPU) {
    auto cpu_tensor_contiguous = src.contiguous();
    EnqueueWriteBuffer(cq, allocator->get_buffer(self.mutable_data_ptr()), cpu_tensor_contiguous.data_ptr(), false);
    Finish(cq);
  }
  else if (self.device().type() == at::kCPU) {
    AT_ASSERT(self.is_contiguous());
    EnqueueReadBuffer(cq, allocator->get_buffer(src.data_ptr()), self.mutable_data_ptr(), true);
    Finish(cq);
  } else {
    // TODO: Implement copy TT -> TT
  }
  return self;
}
} // namespace tt

namespace native {
bool is_tt_available() {
  return true;
}

} // namespace native
} // namespace at
