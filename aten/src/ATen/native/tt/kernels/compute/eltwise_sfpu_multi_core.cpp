// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    uint32_t start_tile_id = get_arg_val<uint32_t>(1);

    // We are going to apply the relu to this circular buffer
    constexpr auto cb_in0 = get_compile_time_arg_val(0);
    constexpr auto cb_out0 = get_compile_time_arg_val(1);
    init_sfpu(cb_in0, cb_out0);

    // Calculate the range of tiles this core should process
    const uint32_t end_tile_id = start_tile_id + n_tiles;

    // Loop over the assigned tiles and perform the computation
    for (uint32_t i = start_tile_id; i < end_tile_id; i++) {
        cb_reserve_back(cb_out0, 1);
        tile_regs_acquire();
        
        cb_wait_front(cb_in0, 1);
	copy_tile(cb_in0, 0, 0);
#ifdef SFPU_OP_CHAIN_0
        SFPU_OP_CHAIN_0
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out0);

        cb_pop_front(cb_in0, 1);
        tile_regs_release();
	cb_push_back(cb_out0, 1);
    }
}
}  // namespace NAMESPACE
