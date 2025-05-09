// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"

void gtz_lez_block(uint32_t in_cb, uint32_t out0_cb, uint32_t out1_cb, uint32_t num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Postcondition: out_cb has num_tiles produced
    // Postcondition: in_cb has num_tiles consumed
    copy_tile_to_dst_init_short(in_cb);
    gtz_tile_init();
    lez_tile_init();

    cb_wait_front(in_cb, num_tiles);
    cb_reserve_back(out0_cb, num_tiles);
    cb_reserve_back(out1_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        copy_tile(in_cb, 0, 0);
        cb_pop_front(in_cb, 1);
        gtz_tile(0);
        pack_tile(0, out0_cb, i);
        lez_tile(0);
        pack_tile(0, out1_cb, i);
        release_dst();
    }
    cb_push_back(out0_cb, num_tiles);
    cb_push_back(out1_cb, num_tiles);
}

void add_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles consumed

    add_tiles_init(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        add_tiles(in0_cb, in1_cb, i, i, 0);
        pack_tile(0, out_cb, i);
        release_dst();
    }
    cb_push_back(out_cb, num_tiles);

    cb_pop_front(in0_cb, num_tiles);
    cb_pop_front(in1_cb, num_tiles);
}

void mul_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles produced

    mul_tiles_init(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        mul_tiles(in0_cb, in1_cb, 0, i, 0);
        cb_pop_front(in0_cb, 1);
        cb_reserve_back(in0_cb, 1);
        pack_tile(0, in0_cb);
        cb_push_back(in0_cb, 1);
        release_dst();
    }
}

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    uint32_t start_tile_id = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = get_compile_time_arg_val(0);
    constexpr auto cb_in1 = get_compile_time_arg_val(1);
    constexpr auto cb_in2 = get_compile_time_arg_val(2);

    constexpr auto cb_tmp1 = tt::CBIndex::c_16;
    constexpr auto cb_tmp2 = tt::CBIndex::c_17;

    constexpr auto cb_out0 = get_compile_time_arg_val(3);

    // Calculate the range of tiles this core should process
    const uint32_t end_tile_id = start_tile_id + n_tiles;

    // Loop over the assigned tiles and perform the computation
    for (uint32_t i = start_tile_id; i < end_tile_id; i++) {
        gtz_lez_block(cb_in0, cb_tmp1, cb_tmp2, 1);
        mul_block_inplace(cb_tmp1, cb_in1, 1);
        mul_block_inplace(cb_tmp2, cb_in2, 1);
        add_block(cb_tmp1, cb_tmp2, cb_out0, 1);
    }
}
}  // namespace NAMESPACE
