// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
namespace NAMESPACE {
void MAIN {
    constexpr int onetile = 1;
    constexpr uint32_t is_b_transposed = get_compile_time_arg_val(0);

    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t Kt = get_arg_val<uint32_t>(1);

    mm_init(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16, is_b_transposed);

    // the simplest possible version of outer product blocked matmul
    // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
    // the outer loops over batch, n and m dimension are combined into a single loop
    for (uint32_t tile = 0; tile < num_tiles; tile++) {
      acquire_dst();
      for (uint32_t kt = 0; kt < Kt; kt++) {
	cb_wait_front(tt::CBIndex::c_0, onetile);
	cb_wait_front(tt::CBIndex::c_1, onetile);

	matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0, false);

	cb_pop_front(tt::CBIndex::c_0, onetile);
	cb_pop_front(tt::CBIndex::c_1, onetile);
      }

      cb_reserve_back(tt::CBIndex::c_16, onetile);
      pack_tile(0, tt::CBIndex::c_16);
      cb_push_back(tt::CBIndex::c_16, onetile);

      release_dst();
    }
}
}  // namespace NAMESPACE

