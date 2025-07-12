// SPDX-FileCopyrightText: © 2023 Philipp Moritz
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "lltt.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_math_common.h"

#include "compute_kernel_api/matmul.h"

using namespace ckernel;
using std::uint32_t;

#ifndef HF
#define HF 0
#endif

template <int MATH_FIDELITY_DESC, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor>
inline void gemm_configure_addrmod(
    const bool transpose,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim)
{
    constexpr int NUM_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);
    constexpr bool high_fidelity = (NUM_FIDELITY_PHASES > 0);
    constexpr int FIDELITY_INCREMENT = high_fidelity ? get_math_fidelity_increment(MATH_FIDELITY_DESC) : 0;

    static_assert(FaceLayout == DstTileFaceLayout::RowMajor, "FaceLayout must be RowMajor");

    // Inner Loop --> 32/8 = 4 times for the full 32x16 face
    // DEST -- 8 rows are calculated each time
    // SRCB -- 8 rows are needed
    // SRCA -- full 16x16 gets used -- hardware will pair cols of A with rows of B
    // D[8,16] = B[8,16] * A[16,16]
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 8, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    // reset all, increment fidelity if we have more fidelity phases
    addr_mod_t {
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 0, .clr = 1, .cr = 1},
        .fidelity = {.incr = FIDELITY_INCREMENT, .clr = 0},
    }
        .set(ADDR_MOD_5);

    if (transpose) {
        addr_mod_t {
            .srca = {.incr = 32, .clr = 0, .cr = 0},
            .srcb = {.incr = 0, .clr = 0, .cr = 1},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_1);
    } else {
        addr_mod_t {
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr = 0, .clr = 0, .cr = 1},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_1);
    }

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 1},
        .srcb = {.incr = 32, .clr = 0, .cr = 1},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_2);

    if (transpose) {
        addr_mod_t {
            .srca = {.incr = 16, .clr = 0, .cr = 1},
            .srcb = {.incr = 48, .clr = 0, .cr = 1},
            .dest = {.incr = 0, .clr = 0, .cr = 1},
        }
            .set(ADDR_MOD_4);
    } else {
        addr_mod_t {
            .srca = {.incr = 32, .clr = 0, .cr = 1},
            .srcb = {.incr = 48, .clr = 0, .cr = 1},
            .dest = {.incr = 0, .clr = 0, .cr = 1},
        }
            .set(ADDR_MOD_4);
    }
}

template <int NUM_FIDELITY_PHASES, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor>
inline void gemm_configure_mop(
    bool transpose,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim)
{
    constexpr bool high_fidelity = NUM_FIDELITY_PHASES > 0;

    const bool reuse_a = ct_dim >= rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;

    const std::uint32_t replay_buf_len = 16;

    load_replay_buf(
        ckernel::math::replay_buf_offset,
        replay_buf_len,
        false,
        [high_fidelity, reuse_a, transpose]
        {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B0A0
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A1
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B0A1

            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A0
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B2A0
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A1
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B2A1

            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A2
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B1A2
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A3
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B1A3

            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A2
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B3A2
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A3

            if constexpr (high_fidelity) {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0);
            } else {
                if (reuse_a) {
                    TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_5, 0);
                } else {
                    TTI_MVMUL(p_setrwc::CLR_B, 0, ADDR_MOD_5, 0);
                }
            }
        });

    constexpr uint inner_loops = high_fidelity ? NUM_FIDELITY_PHASES : 1;
    ckernel_template tmp(1, inner_loops, lltt::replay_insn(ckernel::math::replay_buf_offset, replay_buf_len));

    if constexpr (high_fidelity) {
        if (reuse_a) {
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
        } else {
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
        }
    }
    tmp.program(instrn_buffer);
}

template <int MATH_FIDELITY_DESC, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor>
inline void gemm_init(
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1)
{
    gemm_configure_addrmod<MATH_FIDELITY_DESC, FaceLayout>(
        transpose, ct_dim, rt_dim, kt_dim);

    constexpr int MATH_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);
    gemm_configure_mop<MATH_FIDELITY_PHASES, FaceLayout>(
        transpose > 0, ct_dim, rt_dim, kt_dim);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <int MATH_FIDELITY_DESC, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor>
inline void gemm_compute(
    uint dst_index, 
    const bool transpose = false, 
    const std::uint32_t ct_dim = 1, 
    const std::uint32_t rt_dim = 1, 
    const std::uint32_t kt_dim = 1)
{
    const bool reuse_a = ct_dim >= rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;
    const std::uint32_t rut_dim = reuse_a ? ct_dim : rt_dim;
    constexpr int NUM_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);

    for (uint t = 0; t < t_dim; t++) {
        for (uint rut = 0; rut < rut_dim; rut++) {
            math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(
                dst_index + (reuse_a ? ct_dim * t + rut : t + rut * ct_dim));

            ckernel_template::run(instrn_buffer);

            // Clear srcB or srcA at end of re-use (once per u block row)
            if (rut == (rut_dim - 1)) {
                if (reuse_a) {
                    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                } else {
                    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                }
            }
        }
    }
}

namespace NAMESPACE {
void MAIN {
    constexpr int onetile = 1;
    constexpr uint32_t is_b_transposed = get_compile_time_arg_val(0);

    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t Kt = get_arg_val<uint32_t>(1);

    // Initialize the simplified GEMM kernel
    gemm_init<HF>(is_b_transposed);

    // Main computation loop
    for (uint32_t tile = 0; tile < num_tiles; tile++) {
        acquire_dst();
        
        for (uint32_t kt = 0; kt < Kt; kt++) {
            cb_wait_front(tt::CBIndex::c_0, onetile);
            cb_wait_front(tt::CBIndex::c_1, onetile);

            gemm_compute<HF>(0, is_b_transposed);

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
