// TODO: Consider unifying this with reduce.cpp

#include "compute_kernel_api/matmul.h"

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/sfpu_int_sum.h"

#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"

#ifdef TRISC_MATH
#define ITERATIONS (8)

// Implement the operation T[0] <- T[0] > 0.0 ? 1.0 : 0.0
void convert(const uint dst_offset) {
  constexpr uint dst_tile_size = 32;
  for(int _ = 0; _ < ITERATIONS; _++) {
    vFloat values = dst_reg[0];
    v_if (values > 0.0) {
      dst_reg[dst_offset * dst_tile_size] = 1.0;
    } v_else {
      dst_reg[dst_offset * dst_tile_size] = 0.0;
    } v_endif;
    dst_reg++;
  }
}

inline void calculate_sum_float_row() {
  vFloat a = dst_reg[0];
  for (unsigned i = 1; i < 32; ++i) {
    a += dst_reg[i];
  }
  dst_reg[0] = a;
}

#endif

namespace NAMESPACE {

void MAIN {
  uint32_t Kt = get_arg_val<uint32_t>(0);
  uint32_t n_tiles = get_arg_val<uint32_t>(1);
  constexpr int onetile = 1;

  constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
  //  The second CB is used as a scale, e.g. if we want to compute
  //  the minimum everything will be scaled by -1
  constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);
  constexpr uint32_t cb_tmp0 = get_compile_time_arg_val(2);
  constexpr uint32_t cb_out0 = get_compile_time_arg_val(3);

  init_sfpu(cb_in0, cb_tmp0);

  cb_wait_front(cb_in1, onetile); // scaler tile from the reader
  for (uint32_t i = 0; i < n_tiles; ++i) {
    for (uint32_t kt = 0; kt < Kt; ++kt) {
      cb_wait_front(cb_in0, onetile);

      acquire_dst();

      reconfig_data_format_srca<true>(cb_in0);
      copy_tile_to_dst_init_short(cb_in0);
      copy_tile(cb_in0, 0, 0);
      reconfig_data_format_srca<true>(cb_in1);

      MATH(llk_math_eltwise_binary_sfpu_params<false>(convert, 0, 1, VectorMode::RC);)
      MATH(llk_math_eltwise_unary_sfpu_params<false>(calculate_sum_float_row, 1, VectorMode::C);)
      cb_reserve_back(cb_tmp0, 1);
      pack_tile(1, cb_tmp0);
      cb_push_back(cb_tmp0, 1);

      release_dst();

      cb_wait_front(cb_tmp0, 1);
      // tt::compute::common::print_full_tile(cb_tmp0);
	
      // dprint_tensix_dest_reg(1);
      cb_pop_front(cb_in0, onetile);
    }
    cb_reserve_back(cb_out0, onetile);
    
    acquire_dst();
    copy_tile(cb_tmp0, 0, 0);
    unary_gt_tile(0, 1107034112); // 31.5f
    dprint_tensix_dest_reg(0);
    pack_reconfig_data_format(cb_out0);
    pack_tile(0, cb_out0);
    release_dst();
    cb_push_back(cb_out0, onetile);
  }
}
  
}
