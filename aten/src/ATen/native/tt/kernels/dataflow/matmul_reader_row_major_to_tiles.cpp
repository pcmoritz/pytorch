#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;

constexpr uint32_t FACE_HEIGHT = 16;
constexpr uint32_t FACE_WIDTH = 16;

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t M = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t K = Kt * TILE_WIDTH;
    uint32_t N = get_arg_val<uint32_t>(4);
    uint32_t start_id = get_arg_val<uint32_t>(5);

    uint32_t Nt = N / TILE_WIDTH;
    uint32_t start_id_h = start_id / Nt;
    uint32_t start_id_w = start_id % Nt;

    constexpr uint32_t datum_size_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t is_b_transposed = get_compile_time_arg_val(1);

    // For now, we only write the code to work for a single tile, will adapt it later
    constexpr uint32_t num_output_tiles = 1;
    
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    constexpr uint32_t onetile = 1;
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat in0_data_format = get_dataformat(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);
    const DataFormat in1_data_format = get_dataformat(cb_id_in1);

    constexpr bool src0_is_dram = 1;
    const InterleavedAddrGen<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = datum_size_bytes * FACE_WIDTH};

    constexpr bool src1_is_dram = 1;
    const InterleavedAddrGen<src1_is_dram> s1 = {
        .bank_base_address = src1_addr, .page_size = datum_size_bytes * FACE_WIDTH};

    const uint32_t a_face_offset[4] = {0, FACE_WIDTH, K * FACE_HEIGHT, K * FACE_HEIGHT + FACE_WIDTH};
    const uint32_t b_face_offset[4] = {0, FACE_WIDTH, N * FACE_HEIGHT, N * FACE_HEIGHT + FACE_WIDTH};

    for (uint32_t n = 0; n < num_output_tiles; ++n) {
      for (uint32_t kt = 0; kt < Kt; ++kt) {
	// Read A tile
	cb_reserve_back(cb_id_in0, onetile);
	uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
	for (uint32_t f = 0; f < 4; ++f) {
#pragma GCC unroll FACE_HEIGHT
	  for (uint32_t h = 0; h < FACE_HEIGHT; ++h) {
	    uint64_t offset = start_id_h * TILE_HEIGHT * K + a_face_offset[f] + kt * TILE_WIDTH + K * h;
	    uint64_t s0_noc_addr = get_noc_addr(offset / FACE_WIDTH, s0);
	    noc_async_read(s0_noc_addr,
			   l1_write_addr_in0,
			   FACE_WIDTH * datum_size_bytes);
	    l1_write_addr_in0 += FACE_WIDTH * datum_size_bytes;
	  }
	}
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);

	// Read B tile
	cb_reserve_back(cb_id_in1, onetile);
	uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
	for (uint32_t f = 0; f < 4; ++f) {
#pragma GCC unroll FACE_HEIGHT
	  for (uint32_t h = 0; h < FACE_HEIGHT; ++h) {
	    uint32_t offset;
	    if constexpr (is_b_transposed) {
	      offset = start_id_w * TILE_HEIGHT * K + a_face_offset[f] + kt * TILE_WIDTH + K * h;
	    } else {
	      offset = start_id_w * TILE_WIDTH + b_face_offset[f] + N * h + kt * TILE_HEIGHT * N;
	    }
	    uint64_t s1_noc_addr = get_noc_addr(offset / FACE_WIDTH, s1);
	    noc_async_read(s1_noc_addr,
			   l1_write_addr_in1,
			   FACE_WIDTH * datum_size_bytes);
	    l1_write_addr_in1 += FACE_WIDTH * datum_size_bytes;
	  }
	}
	noc_async_read_barrier();
	cb_push_back(cb_id_in1, onetile);
      }
    }
}
