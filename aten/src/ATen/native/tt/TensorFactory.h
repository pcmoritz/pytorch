#include <ATen/ATen.h>

namespace at::native {

Tensor empty_tt(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt);

Tensor empty_strided_tt(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt);

}

#define AT_DISPATCH_TT_TYPES(TYPE, NAME, ...)                                  \
  AT_DISPATCH_SWITCH(                                                          \
      TYPE,                                                                    \
      NAME,                                                                    \
      AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                     \
      AT_DISPATCH_CASE(at::ScalarType::Half,  __VA_ARGS__)                     \
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)		       \
      AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)                      \
      AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)                       \
      AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)                     \
      AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)                      \
      AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__))
