#pragma once

#include <c10/core/Allocator.h>
#include <c10/macros/Macros.h>

namespace at::tt {

TORCH_API at::Allocator* GetTTAllocator(bool useSharedAllocator = false);

} // namespace at::tt
