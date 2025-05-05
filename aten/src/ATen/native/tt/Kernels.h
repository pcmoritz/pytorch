#pragma once

namespace at::native {

void MemcpyWithOffsets(uint32_t dst_addr, uint32_t dst_offset, uint32_t src_addr, uint32_t src_offset, uint32_t num_tiles);

}
