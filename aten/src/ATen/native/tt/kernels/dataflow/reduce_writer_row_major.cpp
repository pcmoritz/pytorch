// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;

constexpr uint32_t FACE_HEIGHT = 16;
constexpr uint32_t FACE_WIDTH = 16;

void kernel_main() {
    uint32_t c_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_tile_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t datum_size_bytes = 2;

    // The circular buffer that we are going to read from and write to DRAM
    constexpr uint32_t cb_out0 = get_compile_time_arg_val(0);
    const uint32_t tile_size_bytes = get_tile_size(cb_out0);

    const InterleavedAddrGen<true> c = {
        .bank_base_address = c_addr, .page_size = datum_size_bytes * FACE_WIDTH
    };

    // Calculate the range of tiles this core should process
    const uint32_t end_tile_id = start_tile_id + n_tiles;

    // Loop over the assigned tiles and write them to the output buffer
    for (uint32_t i = start_tile_id; i < end_tile_id; i++) {
        // Make sure there is a tile in the circular buffer
        cb_wait_front(cb_out0, 1);
        uint32_t cb_out0_addr = get_read_ptr(cb_out0);

        // Copy out only the first row of the result tile, which contains the means
        // (all other entries of the tile are zero)
        for (uint32_t h = 0; h < 2; ++h) {
            uint64_t c_noc_addr = get_noc_addr(i * 2 + h, c);
            noc_async_write(cb_out0_addr, c_noc_addr, FACE_WIDTH * datum_size_bytes);
            cb_out0_addr += FACE_HEIGHT * FACE_WIDTH * datum_size_bytes;
        }

        // This will wait until the write is done. As an alternative, noc_async_writes_flushed()
        // can be faster because it waits until the write request is sent. In that case, you
        // have to use noc_async_write_barrier() at least once at the end of data movement
        // kernel to make sure all writes are done.
        noc_async_write_barrier();
        // Mark the tile as consumed
        cb_pop_front(cb_out0, 1);
    }
}
