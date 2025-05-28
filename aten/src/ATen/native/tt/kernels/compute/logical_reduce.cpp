// TODO: Consider unifying this with reduce.cpp

#include "compute_kernel_api/reduce.h"

namespace NAMESPACE {

void MAIN {
  uint32_t Kt = get_arg_val<uint32_t>(0);
  uint32_t n_tiles = get_arg_val<uint32_t>(1);
  constexpr int onetile = 1;

  constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
  //  The second CB is used as a scale, e.g. if we want to compute
  //  the minimum everything will be scaled by -1
  constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);
  constexpr uint32_t cb_out0 = get_compile_time_arg_val(2);

  reduce_init<true>(cb_in0, cb_in1, cb_out0);

  cb_wait_front(cb_in1, onetile); // scaler tile from the reader
  for (uint32_t i = 0; i < n_tiles; ++i) {
    acquire_dst();
    // Enable using uint8 elements
    reconfig_data_format_srca<true>(cb_in0);
    reconfig_data_format_srcb<true>(cb_in1);
    for (uint32_t kt = 0; kt < Kt; ++kt) {
      cb_wait_front(cb_in0, onetile);
      reduce_tile(cb_in0, cb_in1, 0, 0, 0);
      cb_pop_front(cb_in0, onetile);
    }
    cb_reserve_back(cb_out0, onetile);
    pack_tile(0, cb_out0);
    cb_push_back(cb_out0, onetile);
    release_dst();
  }
}
  
}
