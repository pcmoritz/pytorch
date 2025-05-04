#pragma once

#include <ATen/ATen.h>

namespace at::native {

void MemcpyOp(const at::Tensor& a, const at::Tensor& b);

}