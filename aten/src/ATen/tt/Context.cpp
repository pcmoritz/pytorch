#include <atomic>

#include <ATen/Tensor.h>
#include <ATen/tt/Context.h>
#include <ATen/tt/TTDevice.h>

namespace at {
namespace tt {

std::atomic<const TTInterface*> g_tt_impl_registry;

TTImplRegistrar::TTImplRegistrar(TTInterface* impl) {
  g_tt_impl_registry.store(impl);
}

at::Tensor& tt_copy_(at::Tensor& self, const at::Tensor& src) {
  auto cpu_tensor_contiguous = src.contiguous();
  auto* allocator = GetTTAllocator();
  allocator->copy_data(self.mutable_data_ptr(), cpu_tensor_contiguous.const_data_ptr(), src.nbytes());
  return self;
}
} // namespace tt

namespace native {
bool is_tt_available() {
  auto p = at::tt::g_tt_impl_registry.load();
  return p ? p->is_tt_available() : false;
}

} // namespace native
} // namespace at
