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

// TODO: Write comment
void count_pos(const uint dst_offset) {
  constexpr uint dst_tile_size = 32;
  vInt a = 0;
  for (unsigned i = 0; i < 32; ++i) {
    vFloat values = dst_reg[i];
    v_if (values > 0.0) {
      a += 1;
    } v_else {
      a += 0;
    } v_endif;
  }
  dst_reg[dst_offset * dst_tile_size] = a;
}

#endif

namespace NAMESPACE {

void MAIN {
  uint32_t Kt = get_arg_val<uint32_t>(0);
  uint32_t n_tiles = get_arg_val<uint32_t>(1);
  constexpr int onetile = 1;

  constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
  constexpr uint32_t cb_out0 = get_compile_time_arg_val(1);

  init_sfpu(cb_in0, cb_out0);

  for (uint32_t i = 0; i < n_tiles; ++i) {
    for (uint32_t kt = 0; kt < Kt; ++kt) {
      cb_wait_front(cb_in0, onetile);

      acquire_dst();

      reconfig_data_format_srca<true>(cb_in0);
      copy_tile_to_dst_init_short(cb_in0);
      copy_tile(cb_in0, 0, 0);

      MATH(llk_math_eltwise_binary_sfpu_params<false>(count_pos, 0, 1, VectorMode::C);)
      cb_reserve_back(cb_out0, 1);
      pack_reconfig_data_format(cb_out0);
      pack_tile(1, cb_out0);
      // dprint_tensix_dest_reg(1);
      cb_push_back(cb_out0, 1);

      release_dst();

      cb_pop_front(cb_in0, onetile);
    }
    // tt::compute::common::print_full_tile(cb_out0);
  }
}
  
}
