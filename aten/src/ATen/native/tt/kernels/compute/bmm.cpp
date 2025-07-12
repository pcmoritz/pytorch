// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack_untilize.h"

#include "compute_kernel_api/eltwise_binary.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
namespace NAMESPACE {

inline void tilizeA_B_matmul_init(
    uint32_t icb0, uint32_t icb1, uint32_t block, uint32_t ocb, uint32_t num_faces = 4, uint32_t face_r_dim = 16) {

  UNPACK((llk_unpack_AB_matmul_hw_configure_disaggregated<DST_ACCUM_MODE>(icb0, icb1)));
  UNPACK((llk_unpack_AB_matmul_init(icb0, icb1, 0)));
  // UNPACK((llk_unpack_tilizeA_B_hw_configure_disaggregated<DST_ACCUM_MODE>(icb0, icb1)));
  // UNPACK((llk_unpack_tilizeA_B_init<true, true>(icb0, icb1, block, num_faces, face_r_dim, face_r_dim)));

  MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(icb0, icb1, 0)));
  MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
  MATH((llk_math_hw_configure_disaggregated(icb0, icb1)));
  // MATH((llk_math_eltwise_binary_init<ELWADD, NONE>(0 /*transpose*/, 0 /*acc_to_dest*/)));

  PACK((llk_pack_hw_configure_disaggregated<DST_ACCUM_MODE, false>(ocb)));
  PACK((llk_pack_init(ocb)));
  PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>(ocb)));
}

void MAIN {
    constexpr int onetile = 1;
    constexpr uint32_t is_b_transposed = get_compile_time_arg_val(0);

    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t Kt = get_arg_val<uint32_t>(1);

    // mm_init(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16, is_b_transposed);
    tilizeA_B_matmul_init(tt::CBIndex::c_0, tt::CBIndex::c_1, 1, tt::CBIndex::c_16);

    // pack_untilize_dst_init_short(tt::CBIndex::c_16);

    // the simplest possible version of outer product blocked matmul
    // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
    // the outer loops over batch, n and m dimension are combined into a single loop
    for (uint32_t tile = 0; tile < num_tiles; tile++) {
      acquire_dst();
      for (uint32_t kt = 0; kt < Kt; kt++) {
	cb_wait_front(tt::CBIndex::c_0, onetile);
	cb_wait_front(tt::CBIndex::c_1, onetile);

	// unpack_tilizeA_B_block(tt::CBIndex::c_0, tt::CBIndex::c_1, onetile, 0);
	UNPACK((llk_unpack_AB_matmul(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0)));

	// matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0, false);
	MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(0, false)));

	cb_pop_front(tt::CBIndex::c_1, onetile);
	cb_pop_front(tt::CBIndex::c_0, onetile);
      }

      cb_reserve_back(tt::CBIndex::c_16, onetile);
      pack_tile(0, tt::CBIndex::c_16);
      // pack_untilize_dst(tt::CBIndex::c_16);
      cb_push_back(tt::CBIndex::c_16, onetile);

      release_dst();
    }
}
}  // namespace NAMESPACE

