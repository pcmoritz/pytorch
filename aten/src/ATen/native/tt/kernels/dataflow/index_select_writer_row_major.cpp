// SPDX-FileCopyrightText: (c) 2024 Philipp Moritz
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

constexpr uint32_t FACE_WIDTH = 16;

void kernel_main() {
  constexpr uint32_t cb_indices = get_compile_time_arg_val(0);
  
  uint32_t in_addr = get_arg_val<uint32_t>(0);
  uint32_t out_addr = get_arg_val<uint32_t>(1);
  uint32_t indices_addr = get_arg_val<uint32_t>(2);
  uint32_t n_pages = get_arg_val<uint32_t>(3);
  uint32_t start_page_id = get_arg_val<uint32_t>(4);

  constexpr uint32_t datum_size_bytes = 2;

  const InterleavedAddrGen<true> in = {
    .bank_base_address = in_addr, .page_size = datum_size_bytes * FACE_WIDTH
  };
  const InterleavedAddrGen<true> out = {
    .bank_base_address = out_addr, .page_size = datum_size_bytes * FACE_WIDTH
  };

  const uint32_t end_page_id = start_page_id + n_pages;

  for (uint32_t i = start_page_id; i < end_page_id; ++i) {
    cb_wait_front(cb_indices, 1);
    uint32_t cb_indices_addr = get_read_ptr(cb_indices);
    volatile tt_l1_ptr uint32_t* index = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_indices_addr);
    for (uint32_t j = 0; j < FACE_WIDTH; ++j) {
      // TODO: read embedding with index index[j] and write it to the output
    }
  }
}

