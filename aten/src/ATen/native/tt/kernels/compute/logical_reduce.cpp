// TODO: Consider unifying this with reduce.cpp

#include "compute_kernel_api/matmul.h"

#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"


#ifdef TRISC_MATH
#define ITERATIONS (8)

// Implement the operation T[0] <- T[0] > 0.0 ? 1.0 : 0.0
inline void convert() {
  for(int _ = 0; _ < ITERATIONS; _++) {
    vFloat values = dst_reg[0];
    v_if (values > 0.0) {
      dst_reg[0] = 1.0;
    } v_else {
      dst_reg[0] = 0.0;
    } v_endif;
    dst_reg++;
  }
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

  // If we use the matrix we are reducing as the second operand of the MM
  // and transpose it, the sums in the result matrix will be in the rows
  // (i.e. consecutive) so they can be read with a single read instruction.
  constexpr uint32_t is_b_transposed = 1;
  mm_init(cb_in1, cb_in0, cb_out0, is_b_transposed);

  cb_wait_front(cb_in1, onetile); // scaler tile from the reader
  for (uint32_t i = 0; i < n_tiles; ++i) {
    acquire_dst();

    for (uint32_t kt = 0; kt < Kt; ++kt) {
      cb_wait_front(cb_in0, onetile);

      reconfig_data_format_srca<true>(cb_in0);
      copy_tile_to_dst_init_short(cb_in0);
      copy_tile(cb_in0, 0, 1);
      cb_pop_front(cb_in0, onetile);
      reconfig_data_format_srca<true>(cb_in1);

      MATH(llk_math_eltwise_unary_sfpu_params<false>(convert, 1, VectorMode::RC);)

      cb_reserve_back(cb_tmp0, onetile);
      pack_tile(1, cb_tmp0);
      cb_push_back(cb_tmp0, onetile);

      cb_wait_front(cb_tmp0, onetile);
      
      matmul_tiles(cb_in1, cb_tmp0, 0, 0, 0, false);
      cb_pop_front(cb_tmp0, onetile);
    }
    cb_reserve_back(cb_out0, onetile);
    pack_reconfig_data_format(cb_out0);
    dprint_tensix_dest_reg(0);
    pack_tile(0, cb_out0);
    // PACK(tt::compute::common::print_full_tile(cb_out0);)
    cb_push_back(cb_out0, onetile);
    release_dst();
  }
}
  
}
