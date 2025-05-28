// SPDX-FileCopyrightText: (c) 2025 Philipp Moritz
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"

#include "util.h"

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;

void kernel_main() {
  // Read parameters from the kernel arguments
  uint32_t a_addr = get_arg_val<uint32_t>(0);
  // TODO: Consider unifying this with matmul
  uint32_t K = get_arg_val<uint32_t>(1);
  uint32_t n_tiles = get_arg_val<uint32_t>(2);
  uint32_t start_tile_id = get_arg_val<uint32_t>(3);
  uint32_t Kt = K / TILE_WIDTH;

  constexpr uint32_t scaler = get_compile_time_arg_val(0);

  // The circular buffer to read the tiles nto
  constexpr uint32_t cb_in0 = get_compile_time_arg_val(1);
  // The cirecular buffer for the scale
  constexpr uint32_t cb_in1 = get_compile_time_arg_val(2);

  generate_mm_scaler(cb_in1, scaler);

  const InterleavedAddrGen<true> a = {
    .bank_base_address = a_addr, .page_size = TILE_WIDTH};

  // Calculate the range of tiles this core should process
  const uint32_t end_tile_id = start_tile_id + n_tiles;

  // Now we loop over the assigned tiles and read them into the circular buffers
  for (uint32_t i = start_tile_id; i < end_tile_id; i++) {
    for (uint32_t kt = 0; kt < Kt; ++kt) {
      cb_reserve_back(cb_in0, 1);
      uint32_t cb_in0_addr = get_write_ptr(cb_in0);

      for (uint32_t h = 0; h < TILE_HEIGHT; ++h) {
	uint64_t offset = start_tile_id * TILE_HEIGHT * K + kt * TILE_WIDTH + K * h;
	uint64_t a_noc_addr = get_noc_addr(offset / TILE_WIDTH, a);
	noc_async_read(a_noc_addr, cb_in0_addr, TILE_WIDTH);
	cb_in0_addr += TILE_WIDTH;
      }
      noc_async_read_barrier();
      cb_push_back(cb_in0, 1);
    }
  }
}
