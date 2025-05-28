#include <cstdint>

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;

void kernel_main() {
  uint32_t c_addr = get_arg_val<uint32_t>(0);
  uint32_t n_tiles = get_arg_val<uint32_t>(1);
  uint32_t start_tile_id = get_arg_val<uint32_t>(2);

  // The circular buffer that we are going to read from and write to DRAM
  constexpr uint32_t cb_out0 = get_compile_time_arg_val(0);

  const InterleavedAddrGen<true> c = {
    .bank_base_address = c_addr, .page_size = TILE_WIDTH
  };

  // Calculate the range of tiles this core should process
  const uint32_t end_tile_id = start_tile_id + n_tiles;

  // Loop over the assigned tiles and write them to the output buffer
  for (uint32_t i = start_tile_id; i < end_tile_id; i++) {
    cb_wait_front(cb_out0, 1);
    uint32_t cb_out0_addr = get_read_ptr(cb_out0);

    // Copy out only the first row of the result tile, which contains the reduced values
    for (uint32_t h = 0; h < TILE_HEIGHT; ++h) {
      uint64_t offset = start_tile_id * TILE_HEIGHT + h;
      uint64_t c_noc_addr = get_noc_addr(offset, c);
      noc_async_write(cb_out0_addr, c_noc_addr, TILE_WIDTH);
      cb_out0_addr += TILE_HEIGHT * TILE_WIDTH;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out0, 1);
  }
}
