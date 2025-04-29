#include <stdint.h>
#include "dataflow_api.h"

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;

constexpr uint32_t FACE_HEIGHT = 16;
constexpr uint32_t FACE_WIDTH = 16;

// Tile is assumed to have 16-bit elements
// Scaler is assumed to be a 16-bit value double packed into a u32
FORCE_INLINE void generate_mm_scaler(const uint32_t cb_id, const uint32_t scaler) {
    cb_reserve_back(cb_id, 1);

    constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t write_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

    // Fill tile with zeros
    // TODO: src addr does not need to be rewritten. Update/add api for this
    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();

    // Fill the first row of the tile with scales
    for (int i = 0; i < 8; ++i) {
        ptr[i] = scaler;
        ptr[i + 128] = scaler;
    }

    cb_push_back(cb_id, 1);
}

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t K = get_arg_val<uint32_t>(1);
    uint32_t Kt = K / TILE_WIDTH; // TODO: Unify with matmul
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t start_tile_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t datum_size_bytes = 2;

    constexpr uint32_t scaler = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in2 = 2;
    generate_mm_scaler(cb_id_in2, scaler);

    constexpr uint32_t cb_id_in0 = 0;
    const uint32_t tile_size_bytes = get_tile_size(cb_id_in0);

    const InterleavedAddrGen<true> src = {
        .bank_base_address = src_addr, .page_size = datum_size_bytes * FACE_WIDTH};

    const uint32_t end_tile_id = start_tile_id + num_tiles;

    const uint32_t src_face_offset[4] = {0, FACE_WIDTH, K * FACE_HEIGHT, K * FACE_HEIGHT + FACE_WIDTH};

    for (uint32_t i = start_tile_id; i < end_tile_id; i++) {
      for (uint32_t kt = 0; kt < Kt; ++kt) {
        cb_reserve_back(cb_id_in0, 1);
        uint32_t cb_in0_addr = get_write_ptr(cb_id_in0);
	for (uint32_t f = 0; f < 4; ++f) {
	  for (uint32_t h = 0; h < FACE_HEIGHT; ++h) {
	    uint64_t offset = 0 * TILE_HEIGHT * K + src_face_offset[f] + kt * TILE_WIDTH + K * h;
            uint64_t a_noc_addr = get_noc_addr(offset / FACE_WIDTH, src);
            noc_async_read(a_noc_addr, cb_in0_addr, FACE_WIDTH * datum_size_bytes);
            cb_in0_addr += FACE_WIDTH * datum_size_bytes;
	  }
	}
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);
      }
    }
}
