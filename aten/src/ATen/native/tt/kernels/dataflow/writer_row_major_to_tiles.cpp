#include "dataflow_api.h"

#include "debug/dprint.h"

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;

constexpr uint32_t FACE_HEIGHT = 16;
constexpr uint32_t FACE_WIDTH = 16;

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;

    constexpr uint32_t N = 32;
    constexpr uint32_t datum_size_bytes = 2;
    constexpr uint32_t ld = TILE_WIDTH;

#ifdef OUT_SHARDED
    cb_wait_front(cb_id_out, num_tiles);
    return;
#endif

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGen<dst_is_dram> s = {
	.bank_base_address = dst_addr, .page_size = datum_size_bytes * FACE_WIDTH};

    constexpr uint32_t face_offset[4] = {0, FACE_WIDTH, N * FACE_HEIGHT, N * FACE_HEIGHT + FACE_WIDTH};
    constexpr uint32_t l1_face_offset[4] = {0, FACE_WIDTH, TILE_WIDTH * FACE_HEIGHT, TILE_WIDTH * FACE_HEIGHT + FACE_WIDTH};

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_wait_front(cb_id_out, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);

#pragma GCC unroll 4
	for (uint32_t f = 0; f < 4; ++f) {
#pragma GCC unroll FACE_HEIGHT
          for (uint32_t h = 0; h < FACE_HEIGHT; ++h) {
            uint64_t offset = face_offset[f] + h * ld;
            uint64_t s_noc_addr = get_noc_addr(offset / FACE_WIDTH, s);

            noc_async_write(l1_read_addr,
                            s_noc_addr,
                            FACE_WIDTH * datum_size_bytes);
	    l1_read_addr += FACE_WIDTH * datum_size_bytes;
          }
	}

        noc_async_write_barrier();  // This will wait until the write is done. As an alternative,
                                    // noc_async_write_flushed() can be faster because it waits
                                    // until the write request is sent. In that case, you have to
                                    // use noc_async_write_barrier() at least once at the end of
                                    // data movement kernel to make sure all writes are done.
        cb_pop_front(cb_id_out, onetile);
    }
}

