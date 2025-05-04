#include "dataflow_api.h"

constexpr uint32_t FACE_WIDTH = 16;

void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_tile_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t datum_size_bytes = 2;

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    const uint32_t tile_size_bytes = get_tile_size(cb_out);

    constexpr uint32_t page_size_bytes = datum_size_bytes * FACE_WIDTH;

    const uint32_t pages_per_tile = tile_size_bytes / page_size_bytes;
    uint32_t current_page_idx = start_tile_id * pages_per_tile;

    const InterleavedAddrGen<true> dst = {
        .bank_base_address = dst_addr, .page_size = page_size_bytes};

    const uint32_t end_tile_id = start_tile_id + n_tiles;
    for (uint32_t i = start_tile_id; i < end_tile_id; i++) {
        cb_wait_front(cb_out, 1);
        uint32_t cb_out_addr = get_read_ptr(cb_out);
        uint32_t bytes_read = 0;
        while (bytes_read < tile_size_bytes) {
            uint32_t dst_noc_addr = get_noc_addr(current_page_idx, dst);
            noc_async_write(cb_out_addr, dst_noc_addr, page_size_bytes);
            bytes_read += page_size_bytes;
            cb_out_addr += page_size_bytes;
            current_page_idx += 1;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}