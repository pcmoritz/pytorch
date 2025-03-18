#include <atomic>

#include <ATen/Tensor.h>
#include <ATen/tt/Context.h>

namespace at {
namespace tt {

std::atomic<const TTInterface*> g_tt_impl_registry;

TTImplRegistrar::TTImplRegistrar(TTInterface* impl) {
  g_tt_impl_registry.store(impl);
}

at::Tensor& tt_copy_(at::Tensor& self, const at::Tensor& src) {
  auto p = at::tt::g_tt_impl_registry.load();
  if (p) {
    return p->tt_copy_(self, src);
  }
  AT_ERROR("TT backend was not linked to the build");
}
} // namespace tt

namespace native {
bool is_tt_available() {
  auto p = at::tt::g_tt_impl_registry.load();
  return p ? p->is_tt_available() : false;
}

} // namespace native
} // namespace at
