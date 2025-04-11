#ifndef TTContext_h
#define TTContext_h

#include <atomic>

#include <ATen/Tensor.h>

namespace at {
namespace tt {

at::Tensor& tt_copy_(at::Tensor& self, const at::Tensor& src);

} // namespace tt

namespace native {
bool is_tt_available();

} // namespace native

} // namespace at

#endif /* TTContext_h */
