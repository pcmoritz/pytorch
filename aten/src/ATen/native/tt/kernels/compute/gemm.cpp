// SPDX-FileCopyrightText: © 2023 Philipp Moritz
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "lltt.h"
#include "circular_buffer.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "cunpack_common.h"
#include "llk_math_common.h"

#include "compute_kernel_api/matmul.h"

using namespace ckernel;
using std::uint32_t;

#ifndef HF
#define HF 0
#endif

inline void gemm_unpack_AB_configure_mop(
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim
) {
    const bool reuse_a = ct_dim >= rt_dim;
    const std::uint32_t replay_buf_prog_len = 12;
    const std::uint32_t replay_buf_run_len  = replay_buf_prog_len / 2;
    if (reuse_a) {
        load_replay_buf(
        0,
	    replay_buf_prog_len,
	    false,
	    // Lambda function to set up replay buffer
	    []{
	        TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
	        TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC0_REG3_Base_address_ADDR32);
	        TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A);
	        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
	        TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
	        // Added to ensure WRCFG instruction has finished, since it takes 2 cycles.
	        TTI_NOP;

	        TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
            TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
            TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A);
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
            TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
	        // Added to ensure WRCFG instruction has finished, since it takes 2 cycles.
	        TTI_NOP;
	    }
        );
    } else {
        load_replay_buf(
	    0,
	    replay_buf_prog_len,
	    false,
	    // Lambda function to set up replay buffer
	    []{
	        TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
		TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC1_REG3_Base_address_ADDR32);
		TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP_LO);
		TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
		TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC1_REG3_Base_address_ADDR32);
		// Added to ensure WRCFG instruction has finished, since it takes 2 cycles.
		TTI_NOP;

		TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
		TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
		TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP_LO);
		TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
		TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
		// Added to ensure WRCFG instruction has finished, since it takes 2 cycles.
		TTI_NOP;
	    }
	);
    }
    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,                                    // src B
	false,                                    // halo - just used for 4 unpacks
	lltt::replay_insn(0, replay_buf_run_len), // runs when context is 0
	0,
	0,
	0,
	lltt::replay_insn(replay_buf_run_len, replay_buf_run_len), // runs when context is 1
	0,
	0
    );
    tmp.program(instrn_buffer);
}

#ifdef TRISC_UNPACK
inline void gemm_unpack_init(
    const std::uint32_t A_id,
    const std::uint32_t B_id,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1
) {
    const std::uint32_t A_num_faces = 4;
    const std::uint32_t B_num_faces = 4;
 
    const std::uint32_t A_face_r_dim = FACE_R_DIM;
    const std::uint32_t B_face_r_dim = FACE_R_DIM;

    const std::uint32_t A_tile_size = get_local_cb_interface(A_id).fifo_page_size;
    const std::uint32_t B_tile_size = get_local_cb_interface(B_id).fifo_page_size;
 
    ckernel::unpacker::configure_unpack_AB<true>(
        unpack_src_format[A_id],
        unpack_src_format[B_id],
        unpack_dst_format[A_id],
        unpack_dst_format[B_id],
        A_face_r_dim,
        B_face_r_dim,
        transpose,
        A_num_faces,
        B_num_faces
    );

    // Configure tile size in datums
    const uint32_t A_x_end = A_num_faces * A_face_r_dim * FACE_C_DIM - 1;
    const uint32_t B_x_end = B_num_faces * B_face_r_dim * FACE_C_DIM - 1;
    TT_SETADCXX(p_setadc::UNP_A, A_x_end, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, B_x_end, 0x0);

    regfile[p_gpr_unpack::TILE_SIZE_A] = A_tile_size;
    regfile[p_gpr_unpack::TILE_SIZE_B] = B_tile_size;
    sync_regfile_write(p_gpr_unpack::TILE_SIZE_B);

    // also turn on within_face_16x16_transpose if it was turned off by datacopy at runtime
    // on WH, the unpacker performs both transpose of faces as well as transpose each face.
    // the former is configured in mop, the latter is configured in cfg register in hw_configure
    // in large matmul, datacopy will disable the transpose of faces, so we need it turn it back on for matmul.
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose);

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    TT_SETDMAREG(0, LOWER_HALFWORD(kt_dim), 0, LO_16(p_gpr_unpack::KT_DIM)); // store kt_dim to gpr for scaling tile size

    gemm_unpack_AB_configure_mop(ct_dim, rt_dim, kt_dim);
}
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

#ifdef TRISC_MATH
template <int MATH_FIDELITY_DESC, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor, int THROTTLE_LEVEL = 0>
inline void gemm_math_init(
    const std::uint32_t A_id,
    const std::uint32_t B_id,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    gemm_configure_addrmod<MATH_FIDELITY_DESC, FaceLayout>(transpose, ct_dim, rt_dim, kt_dim);

    constexpr int MATH_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);
    gemm_configure_mop<MATH_FIDELITY_PHASES, FaceLayout>(transpose > 0, ct_dim, rt_dim, kt_dim);
    math::reset_counters(p_setrwc::SET_ABD_F);

    _llk_math_pack_sync_init_<DST_SYNC_MODE, DST_ACCUM_MODE>();

    _llk_math_hw_configure_<false, false>(unpack_dst_format[A_id], unpack_dst_format[B_id]);
}
#endif

#ifdef TRISC_PACK
#include "llk_outputs.h"

inline void gemm_pack_init(const std::uint32_t C_id, const std::uint32_t transpose = 0) {
    const std::uint32_t face_r_dim = get_output_face_r_dim(C_id);
    const std::uint32_t tile_c_dim = get_output_tile_c_dim(C_id);
    const std::uint32_t num_faces = get_output_num_faces(C_id);
    const bool partial_face = get_output_partial_face(C_id);
    const bool narrow_tile = get_output_narrow_tile(C_id);

    const std::uint32_t tile_size = get_local_cb_interface(output_id).fifo_page_size;

    ckernel::packer::configure_pack<DST_ACCUM_MODE, false, false>(
        pack_src_format[C_id],
        pack_dst_format[C_id],
        tile_size,
        face_r_dim,
        tile_c_dim,
        num_faces,
        partial_face,
        narrow_tile,
        0
    );

    llk_pack_init(C_id);
    llk_pack_dest_init<DST_ACCUM_MODE, false>();
}
#endif

template <int MATH_FIDELITY_DESC, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor>
inline void gemm_init(
    uint32_t A_id, uint32_t B_id, uint32_t C_id,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1)
{
    UNPACK(gemm_unpack_init(A_id, B_id, transpose));
    MATH(gemm_math_init<MATH_FIDELITY>(A_id, B_id, transpose));
    PACK(gemm_pack_init(C_id, transpose));
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

    DPRINT << "XX gemm" << ENDL();

    // Initialize the simplified GEMM kernel
    gemm_init<HF>(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16, is_b_transposed);

    DPRINT << "XX gemm after init" << ENDL();

    // Main computation loop
    for (uint32_t tile = 0; tile < num_tiles; tile++) {
        acquire_dst();
        
        for (uint32_t kt = 0; kt < Kt; kt++) {
            cb_wait_front(tt::CBIndex::c_0, onetile);
            cb_wait_front(tt::CBIndex::c_1, onetile);

	    DPRINT << "XX gemm before compute" << ENDL();

            // TODO: Add call to _llk_unpack_AB_matmul_
            // gemm_compute<HF>(0, is_b_transposed);

	    DPRINT << "XX gemm after compute" << ENDL();

            cb_pop_front(tt::CBIndex::c_0, onetile);
            cb_pop_front(tt::CBIndex::c_1, onetile);
        }

	DPRINT << "XX gemm ater loop" << ENDL();

        cb_reserve_back(tt::CBIndex::c_16, onetile);
        pack_tile(0, tt::CBIndex::c_16);
        cb_push_back(tt::CBIndex::c_16, onetile);

        release_dst();
    }

    DPRINT << "XX finish" << ENDL();
}
}  // namespace NAMESPACE
