#ifndef TTContext_h
#define TTContext_h

#include <atomic>

#include <ATen/Tensor.h>

namespace at {
namespace tt {

struct TTInterface {
  virtual ~TTInterface() = default;
  virtual bool is_tt_available() const = 0;
  virtual at::Tensor& tt_copy_(at::Tensor& self, const at::Tensor& src)
      const = 0;
};

extern std::atomic<const TTInterface*> g_tt_impl_registry;

class TTImplRegistrar {
 public:
  explicit TTImplRegistrar(TTInterface*);
};

at::Tensor& tt_copy_(at::Tensor& self, const at::Tensor& src);

} // namespace tt
} // namespace at

#endif /* TTContext_h */
