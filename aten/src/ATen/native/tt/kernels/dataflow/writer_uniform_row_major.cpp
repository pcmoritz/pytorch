// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "dataflow_api.h"

using namespace tt;

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;

constexpr uint32_t FACE_HEIGHT = 16;
constexpr uint32_t FACE_WIDTH = 16;

void kernel_main() {
    constexpr uint32_t intermed_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);
    constexpr bool output_is_dram = get_compile_time_arg_val(2) == 1;

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t end_id = start_id + num_tiles;

    constexpr uint32_t datum_size_bytes = 2;

    const InterleavedAddrGen<true> c = {
        .bank_base_address = dst_addr, .page_size = datum_size_bytes * FACE_WIDTH
    };

    cb_reserve_back(dst_cb_id, 1);
    uint32_t dst_cb_write_ptr = get_write_ptr(dst_cb_id);

    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(intermed_cb_id, 1);

        uint32_t intermed_cb_read_ptr = get_read_ptr(intermed_cb_id);
        auto intermed_cb_addr = reinterpret_cast<float*>(intermed_cb_read_ptr);

#ifdef OUTPUT_DTYPE_FLOAT32
	assert(false);
#endif

#ifdef OUTPUT_DTYPE_BFLOAT16
        auto dst_cb_addr = reinterpret_cast<uint8_t*>(dst_cb_write_ptr);
        for (uint32_t k = 0; k < constants::TILE_WIDTH; k++) {
            for (uint32_t j = 0; j < constants::TILE_HEIGHT; j++) {
                float rand_float = *intermed_cb_addr;

                uint16_t* uint16_ptr = reinterpret_cast<uint16_t*>(&rand_float) + 1;
                *(uint16_t*)dst_cb_addr = *uint16_ptr;
                dst_cb_addr += 2;
                intermed_cb_addr += 1;
            }
        }
        cb_pop_front(intermed_cb_id, 1);

        for (uint32_t h = 0; h < TILE_HEIGHT * 2; ++h) {
            uint64_t c_noc_addr = get_noc_addr(i * TILE_HEIGHT * 2 + h, c);
            noc_async_write(dst_cb_write_ptr, c_noc_addr, FACE_WIDTH * datum_size_bytes);
            dst_cb_write_ptr += FACE_WIDTH * datum_size_bytes;
        }
        noc_async_write_barrier();
#endif
    }
}

