#include <cstdint>

constexpr uint32_t FACE_WIDTH = 16;

void kernel_main() {
  // Tile of indices of elements to copy
  constexpr uint32_t cb_indices = get_compile_time_arg_val(0);

  uint32_t indices_addr = get_arg_val<uint32_t>(0);
  uint32_t n_pages = get_arg_val<uint32_t>(1);
  uint32_t start_page_id = get_arg_val<uint32_t>(2);

  constexpr uint32_t datum_size_bytes = sizeof(uint32_t);

  const InterleavedAddrGen<true> indices = {
    .bank_base_address = indices_addr, .page_size = datum_size_bytes * FACE_WIDTH};

  // Calculate the range of pages this core should process
  const uint32_t end_page_id = start_page_id + n_pages;

  for (uint32_t i = start_page_id; i < end_page_id; ++i) {
    cb_reserve_back(cb_indices, 1);
    uint32_t cb_indices_addr = get_write_ptr(cb_indices);
    uint64_t indices_noc_addr = get_noc_addr(i * FACE_WIDTH * datum_size_bytes, indices);
    noc_async_read(indices_noc_addr, cb_indices_addr, FACE_WIDTH * datum_size_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_indices, 1);
  }
}
