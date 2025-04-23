// SPDX-FileCopyrightText: (c) 2024 Philipp Moritz
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

constexpr uint32_t FACE_WIDTH = 16;

void kernel_main() {
    constexpr uint32_t num_tensors = get_compile_time_arg_val(0);

    uint32_t dim = get_arg_val<uint32_t>(0);
    uint32_t n_pages = get_arg_val<uint32_t>(1);
    uint32_t start_page_id = get_arg_val<uint32_t>(2);
    uint32_t dst_addr = get_arg_val<uint32_t>(3);
    uint32_t buffer_addr = get_arg_val<uint32_t>(4);
    tt_l1_ptr uint32_t* arg_ptr = (tt_l1_ptr uint32_t*)get_arg_addr(5);

    const uint32_t end_page_id = start_page_id + n_pages;

    uint8_t src_addr_gens_memblk[sizeof(InterleavedAddrGen<true>) * num_tensors];
    InterleavedAddrGen<true>* src_addr_gens = reinterpret_cast<InterleavedAddrGen<true>*>(src_addr_gens_memblk);
    const InterleavedAddrGen<true> dst_addr_gen = {
      .bank_base_address = dst_addr, .page_size = datum_size_bytes * FACE_WIDTH
    };
    
    uint32_t num_pages_per_block[num_tensors];
    uint32_t src_page_id[num_tensors];
    for (uint32_t i = 0; i < num_tensors; ++i) {
        uint32_t src_addr = arg_ptr[i];
        new (&src_addr_gens[i]) InterleavedAddrGen<true>{
            .bank_base_address = src_addr, .page_size = FACE_WIDTH};
        num_pages_per_block[i] = arg_ptr[num_tensors + i]
        src_page_id[i] = arg_ptr[2 * num_tensors + i];
    }

    uint32_t curr_tensor_id = start_tensor_id;
    for (uint32_t dst_page_id = start_page_id; dst_page_id < end_page_id; ) {
        InterleavedAddrGen<true>& src_addr_gen = src_addr_gens[curr_tensor_id];
        for (uint32_t i = 0; i < num_pages_per_block[curr_tensor_id]; ++i) {
            uint64_t src_noc_addr = get_noc_addr(src_page_id[curr_tensor_id], src_addr_gen);
            uint64_t dst_noc_addr = get_noc_addr(dst_page_id, dst_addr_gen);
            noc_async_read(src_noc_addr, buffer_addr, FACE_WIDTH * datum_size_bytes);
	        noc_async_read_barrier();
	        noc_async_write(buffer_addr, dst_noc_addr, FACE_WIDTH * datum_size_bytes);
	        noc_async_write_barrier();
            src_page_id[curr_tensor_id] += 1;
            dst_page_id += 1;
        }
        curr_tensor_id = (curr_tensor_id + 1) % num_tensors;
    }
}
