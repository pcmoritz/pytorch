#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t MtKt = get_arg_val<uint32_t>(5);
    uint32_t KtNt = get_arg_val<uint32_t>(6);

    constexpr uint32_t datum_size_bytes = get_compile_time_arg_val(0);

    // For now, we only write the code to work for a single tile, will adapt it later
    constexpr uint32_t ld = TILE_WIDTH;
    
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    constexpr uint32_t onetile = 1;
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat in0_data_format = get_dataformat(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);
    const DataFormat in1_data_format = get_dataformat(cb_id_in1);

    constexpr bool src0_is_dram = 1;
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = in0_tile_bytes, .data_format = in0_data_format};

    constexpr bool src1_is_dram = 1;
    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr, .page_size = in1_tile_bytes, .data_format = in1_data_format};

    for (unint32_t n = 0; n < num_output_tiles; ++n) {
      for (uint32_t kt = 0; kt < Kt; kt++) {
	cb_reserve_back(cb_id_in0, onetile);
	uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
	constexpr uint32_t face_offset[4] = {0, FACE_WIDTH, N * FACE_HEIGHT, N * FACE_HEIGHT + FACE_WIDTH};
#pragma GCC unroll 4
	for (uint32_t f = 0; f < 4; ++f) {
#pragma GCC unroll FACE_HEIGHT
	  for (uint32_t h = 0; h < FACE_HEIGHT; ++h) {
	    noc_async_read(s0 + face_offset[f] + h * ld, l1_write_addr_in0 + face_offset[f] + h * TILE_WIDTH);
	  }
	}
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
      }
    }
}
