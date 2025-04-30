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

// Fills one full tile of bfloat16 with a scalar value
// Scalar is assumed to be a 16-bit value double packed into a u32
FORCE_INLINE void fill_with_val_bfloat16(uint32_t cb_id, uint32_t packed_scalar) {
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < TILE_HEIGHT * TILE_WIDTH / 2; ++i) {
        ptr[i] = packed_scalar;
    }
}

void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t a_addr = get_arg_val<uint32_t>(0);
#ifndef BINARY_ELTWISE_SCALAR_OP
    uint32_t b_addr = get_arg_val<uint32_t>(1);
#else
    uint32_t b_packed_scalar = get_arg_val<uint32_t>(1);
#endif
    uint32_t n_tiles = get_arg_val<uint32_t>(2);
    uint32_t start_tile_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t datum_size_bytes = 2;

    // The circular buffers to read the tiles into
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);

    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    const InterleavedAddrGen<true> a = {
        .bank_base_address = a_addr, .page_size = datum_size_bytes * FACE_WIDTH};

#ifndef BINARY_ELTWISE_SCALAR_OP
    const InterleavedAddrGen<true> b = {
        .bank_base_address = b_addr, .page_size = datum_size_bytes * FACE_WIDTH};
#else
    // We only need to fill the tile with the scalar value once
    cb_reserve_back(cb_in1, 1);
    fill_with_val_bfloat16(cb_in1, b_packed_scalar);
    cb_push_back(cb_in1, 1);
#endif

    // Calculate the range of tiles this core should process
    const uint32_t end_tile_id = start_tile_id + n_tiles;

    // Now we loop over the assigned tiles and read them into the circular
    // buffers
    for (uint32_t i = start_tile_id; i < end_tile_id; i++) {
        cb_reserve_back(cb_in0, 1);
        uint32_t cb_in0_addr = get_write_ptr(cb_in0);
#ifndef BINARY_ELTWISE_SCALAR_OP
        cb_reserve_back(cb_in1, 1);
        uint32_t cb_in1_addr = get_write_ptr(cb_in1);
#endif

	    for (uint32_t h = 0; h < TILE_HEIGHT * 2; ++h) {
	        uint64_t a_noc_addr = get_noc_addr(i * TILE_HEIGHT * 2 + h, a);
	        noc_async_read(a_noc_addr, cb_in0_addr, FACE_WIDTH * datum_size_bytes);
	        cb_in0_addr += FACE_WIDTH * datum_size_bytes;
#ifndef BINARY_ELTWISE_SCALAR_OP
	        uint64_t b_noc_addr = get_noc_addr(i * TILE_HEIGHT * 2 + h, b);
	        noc_async_read(b_noc_addr, cb_in1_addr, FACE_WIDTH * datum_size_bytes);
	        cb_in1_addr += FACE_WIDTH * datum_size_bytes;
#endif
	    }

        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);
#ifndef BINARY_ELTWISE_SCALAR_OP
        cb_push_back(cb_in1, 1);
#endif
    }
}
