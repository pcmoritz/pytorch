#include <ATen/ops/add_native.h>
#include <ATen/ops/relu_native.h>
#include <ATen/tt/TTDevice.h>

#include <tt-metalium/host_api.hpp>

using namespace tt::tt_metal;

namespace at::native {

CBHandle MakeCircularBufferF32(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles) {
  constexpr uint32_t tile_size = sizeof(float) * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
  return MakeCircularBuffer(program, core, cb, n_tiles * tile_size, tile_size, tt::DataFormat::Float32);
}

at::Tensor & add_out_tt(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();
  CommandQueue& cq = device->command_queue();
  Program program = CreateProgram();

  auto a = allocator->get_buffer(self.data_ptr());
  auto b = allocator->get_buffer(other.data_ptr());
  auto c = allocator->get_buffer(out.data_ptr());

  auto grid_size = device->compute_with_storage_grid_size();
  uint32_t num_cores_x = grid_size.x;
  uint32_t num_cores_y = grid_size.y;
  uint32_t num_cores_total = num_cores_x * num_cores_y;
  auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

  const uint32_t cir_buf_num_title = 4;
  CBHandle cb_a = MakeCircularBufferF32(program, all_device_cores, tt::CBIndex::c_0, cir_buf_num_title);
  CBHandle cb_b = MakeCircularBufferF32(program, all_device_cores, tt::CBIndex::c_1, cir_buf_num_title);
  CBHandle cb_c = MakeCircularBufferF32(program, all_device_cores, tt::CBIndex::c_2, cir_buf_num_title);

  std::vector<uint32_t> reader_compile_time_args = {(uint32_t)tt::CBIndex::c_0, (uint32_t)tt::CBIndex::c_1};
  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)tt::CBIndex::c_2};
  std::vector<uint32_t> compute_compile_time_args = {(uint32_t)tt::CBIndex::c_0, (uint32_t)tt::CBIndex::c_1, (uint32_t)tt::CBIndex::c_2};

  auto reader = CreateKernel(
      program,
      "tt_metal/programming_examples/vecadd_multi_core/kernels/"
      "interleaved_tile_read_multi_core.cpp",
      all_device_cores,
      DataMovementConfig{
          .processor = DataMovementProcessor::RISCV_0,
          .noc = NOC::RISCV_0_default,
          .compile_args = reader_compile_time_args});
  auto writer = CreateKernel(
      program,
      "tt_metal/programming_examples/vecadd_multi_core/kernels/"
      "tile_write_multi_core.cpp",
      all_device_cores,
      DataMovementConfig{
          .processor = DataMovementProcessor::RISCV_1,
          .noc = NOC::RISCV_1_default,
          .compile_args = writer_compile_time_args});
  auto compute = CreateKernel(
      program,
      "tt_metal/programming_examples/vecadd_multi_core/"
      "kernels/add_multi_core.cpp",
      all_device_cores,
      ComputeConfig{.math_approx_mode = false, .compile_args = compute_compile_time_args, .defines = {}});

  constexpr bool row_major = true;
  auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
      tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, n_tiles, row_major);

  auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);
  for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
      const auto& core = cores[i];
       uint32_t num_tiles_per_core;

      if (core_group_1.contains(core)) {
          num_tiles_per_core = num_tiles_per_core_group_1;
      } else if (core_group_2.contains(core)) {
          num_tiles_per_core = num_tiles_per_core_group_2;
      } else {
          SetRuntimeArgs(program, reader, core, std::array<uint32_t, 10>{0});
          SetRuntimeArgs(program, writer, core, std::array<uint32_t, 11>{0});
          SetRuntimeArgs(program, compute, core, std::array<uint32_t, 3>{0});
          continue;
      }
      SetRuntimeArgs(program, reader, core, {a->address(), b->address(), num_tiles_per_core, start_tile_id});
      SetRuntimeArgs(program, writer, core, {c->address(), num_tiles_per_core, start_tile_id});
      SetRuntimeArgs(program, compute, core, {num_tiles_per_core, start_tile_id});
      start_tile_id += num_tiles_per_core;
  }

  EnqueueProgram(cq, program, true);
  Finish(cq);
  
  return out;
}

// RELU

Tensor relu_tt(const Tensor& self) {
  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();
  CommandQueue& cq = device->command_queue();
  Program program = CreateProgram();

  return self;
}

// static void sum_kernel_tt(TensorIterator& iter) {
// }

// REGISTER_TT_DISPATCH(sum_stub, &sum_kernel_tt)

}
