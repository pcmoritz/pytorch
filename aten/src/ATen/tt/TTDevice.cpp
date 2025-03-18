#include <ATen/tt/TTDevice.h>

#include <iostream>

namespace at::tt {

at::Allocator* GetTTAllocator(bool useSharedAllocator) {
  std::cout << "XXX" << std::endl;
  return nullptr; // TODO(pcm)
}
  
}  // namespace at::tt
