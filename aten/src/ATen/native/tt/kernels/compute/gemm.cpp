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

template <bool is_fp32_dest_acc_en, bool row_pool = false, bool fpu_srnd_en = false, bool pack_srnd_en = false, bool disable_src_zero_flag = false>
inline void gemm_configure_unpack_AB(
    const uint unpA_src_format,
    const uint unpB_src_format,
    const uint unpA_dst_format,
    const uint unpB_dst_format,
    const uint unpA_face_r_dim      = FACE_R_DIM,
    const uint unpB_face_r_dim      = FACE_R_DIM,
    const bool transpose_xy_srca_en = false,
    const uint unpA_num_faces       = 4,
    const uint unpB_num_faces       = 4)
{
    // Check that unpacker is done (all contexts freed up) before starting hw configuration
    ckernel::unpacker::wait_for_idle();

    // Reset address counters
    ckernel::unpacker::unpacker_addr_counter_init();

    const uint unpA_src_format_masked = (uint)unpA_src_format & 0x0F;
    const uint unpB_src_format_masked = (uint)unpB_src_format & 0x0F;
    const uint unpA_dst_format_masked = (uint)unpA_dst_format & 0x0F;
    const uint unpB_dst_format_masked = (uint)unpB_dst_format & 0x0F;

    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();

    uint unpA_ch1_x_stride = (uint)(unpA_dst_format_masked & 0x3) == (uint)DataFormat::Float32   ? 4
                             : (uint)(unpA_dst_format_masked & 0x3) == (uint)DataFormat::Float16 ? 2
                                                                                                 : 1;
    uint unpB_ch1_x_stride = (uint)(unpB_dst_format_masked & 0x3) == (uint)DataFormat::Float32   ? 4
                             : (uint)(unpB_dst_format_masked & 0x3) == (uint)DataFormat::Float16 ? 2
                                                                                                 : 1;
    uint unpA_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpA_ch1_x_stride;
    uint unpB_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpB_ch1_x_stride;
    uint exp_width         = ((uint)unpA_dst_format_masked >> 2) & 0x1; // 0=5-bit, 1=8-bit

    // Strides for incrementing ch1 address to srcA and srcB
    cfg[UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] =
        (0 << UNP0_ADDR_CTRL_ZW_REG_1_Wstride_SHAMT) |
        (unpA_ch1_z_stride << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT); // Z and W(not used) stride for dest address (ch1)

    cfg[UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] =
        (0 << UNP1_ADDR_CTRL_ZW_REG_1_Wstride_SHAMT) |
        (unpB_ch1_z_stride << UNP1_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT); // Z and W(not used) stride for dest address (ch1)

    // Math ALU_FORMAT_REG
    t6_mutex_acquire(mutex::REG_RMW);
    uint alu_src_format = (0x0 << ALU_FORMAT_SPEC_REG_SrcA_val_SHAMT);

    constexpr uint mask0 = (1 << (ALU_FORMAT_SPEC_REG_Dstacc_override_SHAMT + 1)) - 1;
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32, ALU_FORMAT_SPEC_REG_SrcA_val_SHAMT, mask0>(alu_src_format);

    ckernel::unpacker::alu_config_u alu_payload = {.val = 0};

    uint32_t fp32_dest_acc_en  = (is_fp32_dest_acc_en) ? (1) : (0);
    uint32_t int8_math_enabled = ((uint)unpA_dst_format_masked == (uint)DataFormat::Int8) || ((uint)unpB_dst_format_masked == (uint)DataFormat::Int8) ||
                                 ((uint)unpA_dst_format_masked == (uint)DataFormat::Int32) || ((uint)unpB_dst_format_masked == (uint)DataFormat::Int32);

    constexpr uint alu_format_mask = ALU_FORMAT_SPEC_REG0_SrcAUnsigned_MASK | ALU_FORMAT_SPEC_REG0_SrcBUnsigned_MASK;

    if ((uint)unpA_src_format == (uint)DataFormat::UInt8)
    {
        alu_payload.f.ALU_FORMAT_SPEC_REG0_SrcAUnsigned = 1;
    }
    if ((uint)unpB_src_format == (uint)DataFormat::UInt8)
    {
        alu_payload.f.ALU_FORMAT_SPEC_REG0_SrcBUnsigned = 1;
    }

    // FP32 accumulation and SFPU to read dest as FP32
    // NOTE: This assumes these config fields are adjacent and in same register!!
    static_assert(ALU_ACC_CTRL_Fp32_enabled_ADDR32 == ALU_FORMAT_SPEC_REG0_SrcA_ADDR32);
    static_assert(ALU_ACC_CTRL_Fp32_enabled_ADDR32 == ALU_ACC_CTRL_SFPU_Fp32_enabled_ADDR32);
    constexpr uint alu_dest_format_mask          = ALU_ACC_CTRL_SFPU_Fp32_enabled_MASK | ALU_ACC_CTRL_Fp32_enabled_MASK;
    alu_payload.f.ALU_ACC_CTRL_Fp32_enabled      = fp32_dest_acc_en;
    alu_payload.f.ALU_ACC_CTRL_SFPU_Fp32_enabled = fp32_dest_acc_en;
    constexpr uint alu_stoch_rnd_mask = ALU_ROUNDING_MODE_Fpu_srnd_en_MASK | ALU_ROUNDING_MODE_Gasket_srnd_en_MASK | ALU_ROUNDING_MODE_Packer_srnd_en_MASK;
    alu_payload.f.ALU_ROUNDING_MODE_Fpu_srnd_en    = fpu_srnd_en;
    alu_payload.f.ALU_ROUNDING_MODE_Gasket_srnd_en = pack_srnd_en;
    alu_payload.f.ALU_ROUNDING_MODE_Packer_srnd_en = pack_srnd_en;

    constexpr uint alu_mask = alu_format_mask | alu_dest_format_mask | alu_stoch_rnd_mask;

    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, alu_mask>(alu_payload.val);

    uint32_t src_zeroflags_disable = ((uint)unpA_dst_format == (uint)DataFormat::UInt16) || ((uint)unpB_dst_format == (uint)DataFormat::UInt16);
    if constexpr (disable_src_zero_flag)
    {
        src_zeroflags_disable = true;
    }
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(src_zeroflags_disable);

    // Set FP8 E4M3 mode, bit is accessible by unpacker/packer
    if ((unpA_src_format & 0x1F) == (uint)DataFormat::Fp8_e4m3)
    {
        cfg_reg_rmw_tensix<THCON_SEC0_REG1_Unp_LF8_4b_exp_RMW>(1);
    }

    if ((unpB_src_format & 0x1F) == (uint)DataFormat::Fp8_e4m3)
    {
        cfg_reg_rmw_tensix<THCON_SEC1_REG1_Unp_LF8_4b_exp_RMW>(1);
    }

    t6_mutex_release(mutex::REG_RMW);

    // Set tile descriptor
    ckernel::unpacker::unpack_tile_descriptor_u tile_descriptor;
    for (uint i = 0; i < ckernel::unpacker::TILE_DESC_SIZE; i++)
    {
        tile_descriptor.val[i] = 0;
    }
    tile_descriptor.f.in_data_format = (uint)unpA_src_format_masked;
    tile_descriptor.f.uncompressed   = 1; // Input tile is uncompressed
    tile_descriptor.f.x_dim          = 0; // Not used for unpA as value is overridden by per context x_dim set below. Used for unpB
    tile_descriptor.f.y_dim          = 1;
    tile_descriptor.f.z_dim          = unpA_num_faces;
    // tile_descriptor.f.blobs_per_xy_plane = 0;
    // tile_descriptor.f.blobs_y_start = 0;
    for (uint i = 0; i < ckernel::unpacker::TILE_DESC_SIZE; i++)
    {
        cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    }
    tile_descriptor.f.in_data_format = row_pool ? (uint)DataFormat::Float32 : unpB_src_format_masked;
    tile_descriptor.f.x_dim          = unpB_face_r_dim * FACE_C_DIM;
    tile_descriptor.f.z_dim          = unpB_num_faces;
    for (uint i = 0; i < ckernel::unpacker::TILE_DESC_SIZE; i++)
    {
        cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    }

    // Set unpacker config
    ckernel::unpacker::unpack_config_u config;
    for (uint i = 0; i < ckernel::unpacker::CONFIG_SIZE; i++)
    {
        config.val[i] = 0;
    }
    config.f.out_data_format = unpA_dst_format_masked;
    config.f.throttle_mode   = 2;
    config.f.context_count   = 0;
    config.f.haloize_mode    = transpose_xy_srca_en ? 1 : 0;
    // config.f.upsample_rate   = 0;
    // config.f.upsamle_and_interlave  = 0;
    // config.f.shift_amount = 0;
    config.f.uncompress_cntx0_3 = 0xf;
    config.f.uncompress_cntx4_7 = 0xf;
    // config.f.limit_addr = 0; // Set dynamically
    // config.f.fifo_size = 0; // Set dynamically
    for (uint i = 0; i < ckernel::unpacker::CONFIG_SIZE; i++)
    {
        cfg[THCON_SEC0_REG2_Out_data_format_ADDR32 + i] = config.val[i];
    }

    config.f.out_data_format = row_pool ? ((uint)DataFormat::Float16 | (exp_width << 2)) : unpB_dst_format_masked;
    config.f.haloize_mode    = 0;

    for (uint i = 0; i < ckernel::unpacker::CONFIG_SIZE; i++)
    {
        cfg[THCON_SEC1_REG2_Out_data_format_ADDR32 + i] = config.val[i];
    }

    uint unpA_x_end = (unpA_face_r_dim == 0) ? 1 : (unpA_face_r_dim << 4) - 1;
    TTI_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
    TTI_SETADCXX(p_setadc::UNP_B, (unpB_face_r_dim << 4) - 1, 0x0);

    // Program base address for all 2 sections (each section address is loaded to corresponding context)
    // Load dummy data to unused location if face height is 0
    const uint Dest_cntx0_address                  = unpA_face_r_dim == 0 ? 22 * 16 : 4 * 16;
    const uint Dest_cntx1_address                  = unpA_face_r_dim == 0 ? 22 * 16 : 4 * 16;
    cfg[THCON_SEC0_REG5_Dest_cntx0_address_ADDR32] = Dest_cntx0_address | (Dest_cntx1_address << 16);

    // Program unpacker0 per context x_dim (face size in l1)
    // Overrides value set by tile descriptor when thread override bit is set in unpack instruction
    const uint face_dim                          = unpA_face_r_dim * FACE_C_DIM;
    cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32] = face_dim | (face_dim << 16);

    constexpr uint face_dim_16x16         = FACE_R_DIM * FACE_C_DIM;
    regfile[p_gpr_unpack::FACE_DIM_16x16] = (face_dim_16x16 / 1) | ((face_dim_16x16 / 1) << 16);
    regfile[p_gpr_unpack::FACE_DIM_8x16]  = (face_dim_16x16 / 2) | ((face_dim_16x16 / 2) << 16);
    regfile[p_gpr_unpack::FACE_DIM_4x16]  = (face_dim_16x16 / 4) | ((face_dim_16x16 / 4) << 16);
    regfile[p_gpr_unpack::FACE_DIM_2x16]  = (face_dim_16x16 / 8) | ((face_dim_16x16 / 8) << 16);
    regfile[p_gpr_unpack::FACE_DIM_1x16]  = (face_dim_16x16 / 16) | ((face_dim_16x16 / 16) << 16);
    sync_regfile_write(p_gpr_unpack::FACE_DIM_1x16);

    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4);

    // Enable address counter for unpacker ch1/dst address
    // final address is calculated as: Dest_cntx0/1_address + address_counter_ch1
    // used for face by face unpacking of entire tile into srcA
    cfg[UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_ADDR32] = 0x1 << UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_SHAMT;

    /*
    // Workaround for HW bug (fp32 dest and movd2a/b is used with srcA/B configured with 5-bit exponent)
    if (is_fp32_dest_acc_en && (exp_width == 0)) {
        reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 1<<11); // Set debug feature disable bit 11
                                                               // workaround for bug tenstorrent/budabackend#1372
    }
    */
    // Workaround for HW bug (int32 dest and movd2a/b is used with srcA/B configured as int8)
    if (int8_math_enabled || (fp32_dest_acc_en && ((uint)unpA_dst_format == (uint)DataFormat::UInt16)))
    {
        reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 1 << 11); // Set debug feature disable bit 11
                                                                 // workaround for bug tenstorrent/budabackend#1948
    }

    // Clear context ID
    ckernel::unpacker::reset_config_context();
}

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
 
    gemm_configure_unpack_AB<true>(
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

    const std::uint32_t tile_size = get_local_cb_interface(C_id).fifo_page_size;

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

inline void gemm_unpack_AB(
    uint32_t A_id, uint32_t B_id,
    const std::uint32_t A_tile_index,
    const std::uint32_t B_tile_index,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1
) {
    volatile uint* cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    std::uint32_t base_address_a = get_local_cb_interface(A_id).fifo_rd_ptr - 1;
    std::uint32_t base_address_b = get_local_cb_interface(B_id).fifo_rd_ptr - 1;

    std::uint32_t tile_size_a = get_local_cb_interface(A_id).fifo_page_size;
    std::uint32_t tile_size_b = get_local_cb_interface(B_id).fifo_page_size;

    const bool reuse_a        = ct_dim >= rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;

    if (!reuse_a) {
        TTI_MULDMAREG(0, p_gpr_unpack::TMP_LO, p_gpr_unpack::TILE_SIZE_B, p_gpr_unpack::KT_DIM);
    }

    for (uint t = 0; t < t_dim; t++)
    {
        std::uint32_t offset_address_a      = tile_size_a * (A_tile_index + (reuse_a ? (t * kt_dim) : (0)));
        std::uint32_t next_offset_address_a = tile_size_a * (A_tile_index + (reuse_a ? ((t + 1) * kt_dim) : (0)));

        std::uint32_t offset_address_b      = tile_size_b * (B_tile_index + (reuse_a ? (0) : (t)));
        std::uint32_t next_offset_address_b = tile_size_b * (B_tile_index + (reuse_a ? (0) : (t + 1)));

        std::uint32_t address_a      = base_address_a + offset_address_a;
        std::uint32_t next_address_a = base_address_a + next_offset_address_a;
        std::uint32_t address_b      = base_address_b + offset_address_b;
        std::uint32_t next_address_b = base_address_b + next_offset_address_b;

        // Wait for free context
        ckernel::unpacker::wait_for_next_context(2);

        // Program unpacker 1 base address
        if (0 == unp_cfg_context)
        {
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_b;
            cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_a;
        }
        else
        {
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_b;
            cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_a;
        }

        semaphore_post(semaphore::UNPACK_SYNC); // Trisc::SEMPOST for context acquire

        // Stall unpacker until pending CFG writes from Trisc have completed
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

        if (reuse_a)
        {
            TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);

            if ((t + 1) < t_dim)
            {
                // Let's load one more tile into srcB
                TT_SETDMAREG(0, LOWER_HALFWORD(next_address_a), 0, LO_16(p_gpr_unpack::TMP0));
                TT_SETDMAREG(0, UPPER_HALFWORD(next_address_a), 0, HI_16(p_gpr_unpack::TMP0));
                if (0 == unp_cfg_context)
                {
                    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG3_Base_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
                }
                else
                {
                    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG3_Base_cntx1_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
                }
                TTI_DMANOP;
                TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                t++;
            }
        }
        else
        {
            TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);

            if ((t + 1) < t_dim)
            {
                // Let's load one more tile into srcB
                TT_SETDMAREG(0, LOWER_HALFWORD(next_address_b), 0, LO_16(p_gpr_unpack::TMP0));
                TT_SETDMAREG(0, UPPER_HALFWORD(next_address_b), 0, HI_16(p_gpr_unpack::TMP0));
                if (0 == unp_cfg_context)
                {
                    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
                }
                else
                {
                    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
                }
                TTI_DMANOP;
                TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                t++;
            }
        }

        TT_MOP(0, (reuse_a ? ct_dim : rt_dim) - 1, unp_cfg_context == 0 ? 0 : 0xff); // Run the MOP

        // T6::SEMGET for context release
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // Switch unpacker config context
        ckernel::unpacker::switch_config_context(unp_cfg_context);
    }
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

            UNPACK(gemm_unpack_AB(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0));
            MATH(gemm_compute<HF>(0, is_b_transposed));

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
