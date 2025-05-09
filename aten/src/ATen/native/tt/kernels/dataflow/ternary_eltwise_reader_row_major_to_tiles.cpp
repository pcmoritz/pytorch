// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include <cstdint>

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;

constexpr uint32_t FACE_HEIGHT = 16;
constexpr uint32_t FACE_WIDTH = 16;


void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);
    uint32_t c_addr = get_arg_val<uint32_t>(2);
    uint32_t n_tiles = get_arg_val<uint32_t>(3);
    uint32_t start_tile_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t datum_size_bytes = 2;

    // The circular buffers to read the tiles into
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in2 = get_compile_time_arg_val(2);

    const InterleavedAddrGen<true> a = {
        .bank_base_address = a_addr, .page_size = 1 * FACE_WIDTH};

    const InterleavedAddrGen<true> b = {
        .bank_base_address = b_addr, .page_size = datum_size_bytes * FACE_WIDTH};

    const InterleavedAddrGen<true> c = {
        .bank_base_address = c_addr, .page_size = datum_size_bytes * FACE_WIDTH};

    // Calculate the range of tiles this core should process
    const uint32_t end_tile_id = start_tile_id + n_tiles;

    // Now we loop over the assigned tiles and read them into the circular
    // buffers
    for (uint32_t i = start_tile_id; i < end_tile_id; i++) {
        cb_reserve_back(cb_in0, 1);
        uint32_t cb_in0_addr = get_write_ptr(cb_in0);
        cb_reserve_back(cb_in1, 1);
        uint32_t cb_in1_addr = get_write_ptr(cb_in1);
        cb_reserve_back(cb_in2, 1);
        uint32_t cb_in2_addr = get_write_ptr(cb_in2);

	    for (uint32_t h = 0; h < TILE_HEIGHT * 2; ++h) {
	        uint64_t a_noc_addr = get_noc_addr(i * TILE_HEIGHT * 2 + h, a);
	        noc_async_read(a_noc_addr, cb_in0_addr, FACE_WIDTH * 1);
	        cb_in0_addr += FACE_WIDTH * 1;

	        uint64_t b_noc_addr = get_noc_addr(i * TILE_HEIGHT * 2 + h, b);
	        noc_async_read(b_noc_addr, cb_in1_addr, FACE_WIDTH * datum_size_bytes);
	        cb_in1_addr += FACE_WIDTH * datum_size_bytes;

	        uint64_t c_noc_addr = get_noc_addr(i * TILE_HEIGHT * 2 + h, c);
	        noc_async_read(c_noc_addr, cb_in2_addr, FACE_WIDTH * datum_size_bytes);
	        cb_in2_addr += FACE_WIDTH * datum_size_bytes;
	    }

        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);
        cb_push_back(cb_in1, 1);
        cb_push_back(cb_in2, 1);
    }
}
