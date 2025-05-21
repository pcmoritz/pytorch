// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"


#ifdef TRISC_MATH
#define ITERATIONS (8)
inline void gate1(const uint dst_offset) {
  constexpr uint dst_tile_size = 32;
  for(int _ = 0; _ < ITERATIONS; _++) {
    vFloat pred = dst_reg[0];
    vFloat values = dst_reg[dst_offset * dst_tile_size];
    v_if (pred > 0.0) {
      dst_reg[dst_offset * dst_tile_size] = values;
    } v_else {
      dst_reg[dst_offset * dst_tile_size] = 0.0;
    } v_endif;
    dst_reg++;
  }
}

inline void gate2(const uint dst_offset) {
  constexpr uint dst_tile_size = 32;
  for(int _ = 0; _ < ITERATIONS; _++) {
    vFloat pred = dst_reg[0];
    vFloat values = dst_reg[dst_offset * dst_tile_size];
    v_if (pred > 0.0) {
      dst_reg[dst_offset * dst_tile_size] = 0.0;
    } v_else {
      dst_reg[dst_offset * dst_tile_size] = values;
    } v_endif;
    dst_reg++;
  }
}
#endif


namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    uint32_t start_tile_id = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = get_compile_time_arg_val(0);
    constexpr auto cb_in1 = get_compile_time_arg_val(1);
    constexpr auto cb_in2 = get_compile_time_arg_val(2);

    constexpr auto cb_tmp1 = get_compile_time_arg_val(3);
    constexpr auto cb_tmp2 = get_compile_time_arg_val(4);

    constexpr auto cb_out0 = get_compile_time_arg_val(5);

    // Calculate the range of tiles this core should process
    const uint32_t end_tile_id = start_tile_id + n_tiles;

    init_sfpu(cb_in1, cb_out0);

    // Loop over the assigned tiles and perform the computation
    for (uint32_t i = start_tile_id; i < end_tile_id; i++) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        cb_wait_front(cb_in2, 1);

        acquire_dst();

        reconfig_data_format_srca<true>(cb_in0);
        copy_tile_to_dst_init_short(cb_in0);
        copy_tile(cb_in0, 0, 0);
	reconfig_data_format_srca<true>(cb_in1);

        copy_tile_init(cb_in1);
        copy_tile(cb_in1, 0, 1);
        MATH(llk_math_eltwise_binary_sfpu_params<false>(gate1, 0, 1, VectorMode::RC);)

        copy_tile_init(cb_in2);
        copy_tile(cb_in2, 0, 2);
        MATH(llk_math_eltwise_binary_sfpu_params<false>(gate2, 0, 2, VectorMode::RC);)

        cb_reserve_back(cb_tmp1, 1);
	cb_reserve_back(cb_tmp2, 1);
	pack_tile(1, cb_tmp1);
	pack_tile(2, cb_tmp2);
	cb_push_back(cb_tmp1, 1);
	cb_push_back(cb_tmp2, 1);
	release_dst();

	cb_wait_front(cb_tmp1, 1);
	cb_wait_front(cb_tmp2, 1);

	tile_regs_acquire();
	cb_reserve_back(cb_out0, 1);
	add_tiles_init(cb_tmp1, cb_tmp2); add_tiles(cb_tmp1, cb_tmp2, 0, 0, 0);
	tile_regs_commit();
	tile_regs_wait();
	pack_tile(0, cb_out0);
        tile_regs_release();

	cb_pop_front(cb_tmp1, 1);
	cb_pop_front(cb_tmp2, 1);

	cb_push_back(cb_out0, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
        cb_pop_front(cb_in2, 1);
    }
}
}  // namespace NAMESPACE
