#pragma once

#include <ATen/ATen.h>

namespace at::native {

void MemcpyFromOffset(const at::Tensor& dst, const std::shared_ptr<::tt::tt_metal::Buffer>& src_buf, int64_t src_offset);

}
