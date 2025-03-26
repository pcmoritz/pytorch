#include <ATen/ops/add_native.h>
#include <ATen/ops/relu_native.h>
#include <ATen/ops/mm_native.h>
#include <ATen/tt/TTDevice.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include <ATen/ATen.h>

using namespace tt;
using namespace tt::tt_metal;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

namespace at::native {

static CBHandle MakeCircularBuffer(
    Program& program, const CoreSpec& core, CBIndex cb, uint32_t size, uint32_t page_size, DataFormat format) {
    CircularBufferConfig cb_src0_config = CircularBufferConfig(size, {{cb, format}}).set_page_size(cb, page_size);
    return CreateCircularBuffer(program, core, cb_src0_config);
}

static CBHandle MakeCircularBufferBF16(Program& program, const CoreSpec& core, CBIndex cb, uint32_t n_tiles) {
  constexpr uint32_t tile_size = sizeof(bfloat16) * constants::TILE_HW;
  return MakeCircularBuffer(program, core, cb, n_tiles * tile_size, tile_size, DataFormat::Float16_b);
}

at::Tensor & add_out_tt(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();
  CommandQueue& cq = device->command_queue();
  Program program = CreateProgram();

  const uint32_t n_tiles = (self.numel() + ::tt::constants::TILE_HW - 1) / ::tt::constants::TILE_HW;

  auto a = allocator->get_buffer(self.data_ptr());
  auto b = allocator->get_buffer(other.data_ptr());
  auto c = allocator->get_buffer(out.data_ptr());

  auto grid_size = device->compute_with_storage_grid_size();
  uint32_t num_cores_x = grid_size.x;
  uint32_t num_cores_y = grid_size.y;
  uint32_t num_cores_total = num_cores_x * num_cores_y;
  auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

  const uint32_t cir_buf_num_title = 4;
  CBHandle cb_a = MakeCircularBufferBF16(program, all_device_cores, CBIndex::c_0, cir_buf_num_title);
  CBHandle cb_b = MakeCircularBufferBF16(program, all_device_cores, CBIndex::c_1, cir_buf_num_title);
  CBHandle cb_c = MakeCircularBufferBF16(program, all_device_cores, CBIndex::c_2, cir_buf_num_title);

  std::vector<uint32_t> reader_compile_time_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1};
  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)CBIndex::c_2};
  std::vector<uint32_t> compute_compile_time_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1, (uint32_t)CBIndex::c_2};

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
      split_work_to_cores(grid_size, n_tiles, row_major);

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

  const uint32_t n_tiles = (self.numel() + ::tt::constants::TILE_HW - 1) / ::tt::constants::TILE_HW;
  auto out = at::empty_like(self);

  auto a = allocator->get_buffer(self.data_ptr());
  auto b = allocator->get_buffer(out.data_ptr());

  auto grid_size = device->compute_with_storage_grid_size();
  uint32_t num_cores_x = grid_size.x;
  uint32_t num_cores_y = grid_size.y;
  uint32_t num_cores_total = num_cores_x * num_cores_y;
  auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

  const uint32_t cir_buf_num_title = 2;
  CBHandle cb_a = MakeCircularBufferBF16(program, all_device_cores, CBIndex::c_0, cir_buf_num_title);
  CBHandle cb_b = MakeCircularBufferBF16(program, all_device_cores, CBIndex::c_1, cir_buf_num_title);

  std::vector<uint32_t> reader_compile_time_args = {(uint32_t)CBIndex::c_0};
  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)CBIndex::c_1};
  std::vector<uint32_t> compute_compile_time_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1};

  auto reader = CreateKernel(
      program,
      "tt_metal/programming_examples/vecadd_multi_core/kernels/"
      "interleaved_tile_read_multi_core_unary.cpp",
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
      "kernels/sfpu_multi_core.cpp",
      all_device_cores,
      ComputeConfig{.math_approx_mode = false, .compile_args = compute_compile_time_args, .defines = {
         {"SFPU_OP_RELU_FAMILY_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "relu_tile_init(); relu_tile(0);"}}});

  constexpr bool row_major = true;
  auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
      split_work_to_cores(grid_size, n_tiles, row_major);

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
      SetRuntimeArgs(program, reader, core, {a->address(), num_tiles_per_core, start_tile_id});
      SetRuntimeArgs(program, writer, core, {b->address(), num_tiles_per_core, start_tile_id});
      SetRuntimeArgs(program, compute, core, {num_tiles_per_core, start_tile_id});
      start_tile_id += num_tiles_per_core;
  }

  EnqueueProgram(cq, program, true);
  Finish(cq);

  return out;
}

// matmul -- this is a very naive but also simple implementation and not optimized yet at all
at::Tensor& mm_out_tt(const at::Tensor & self, const at::Tensor & mat2, at::Tensor &result) {
  int64_t M = self.size(0);
  int64_t K = self.size(1);
  AT_ASSERT(mat2.size(0) == K);
  int64_t N = mat2.size(1);

  uint32_t Mt = M / constants::TILE_HEIGHT;
  uint32_t Kt = K / constants::TILE_WIDTH;
  uint32_t Nt = N / constants::TILE_WIDTH;
  uint32_t KtNt = Kt * Nt;
  uint32_t MtKt = Mt * Kt;
  uint32_t MtNt = Mt * Nt;

  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();
  CommandQueue& cq = device->command_queue();
  Program program = CreateProgram();

  auto grid_size = device->compute_with_storage_grid_size();
  uint32_t num_cores_x = grid_size.x;
  uint32_t num_cores_y = grid_size.y;

  auto num_output_tiles_total = (M * N) / constants::TILE_HW;
  auto [num_cores, all_cores, core_group_1, core_group_2,
        num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] =
            split_work_to_cores(grid_size, num_output_tiles_total);

  auto a = allocator->get_buffer(self.data_ptr());
  auto b = allocator->get_buffer(mat2.data_ptr());
  auto c = allocator->get_buffer(result.data_ptr());

  const uint32_t num_input_tiles = 2;
  CBHandle cb_a = MakeCircularBufferBF16(program, all_cores, CBIndex::c_0, num_input_tiles);
  CBHandle cb_b = MakeCircularBufferBF16(program, all_cores, CBIndex::c_1, num_input_tiles);
  const uint32_t num_output_tiles = 2;
  CBHandle cb_c = MakeCircularBufferBF16(program, all_cores, CBIndex::c_16, num_output_tiles);

  std::vector<uint32_t> reader_compile_time_args = {(uint32_t)1, (uint32_t)1};
  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)CBIndex::c_16, (uint32_t)1};

  auto reader_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp",
    all_cores,
    tt_metal::DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = reader_compile_time_args});

  auto writer_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
    all_cores,
    tt_metal::DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = writer_compile_time_args});

  MathFidelity math_fidelity = MathFidelity::HiFi4;

  std::vector<uint32_t> compute_args_group_1 = {
    1,                                 // B
    1,                                 // Mt
    Kt,                                // Kt
    num_output_tiles_per_core_group_1  // Nt
  };  // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set Nt
      // for simplicity

  auto matmul_multi_core_kernel_group_1_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/compute/bmm.cpp",
    core_group_1,
    tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args_group_1});

  if (!core_group_2.ranges().empty()) {
     std::vector<uint32_t> compute_args_group_2 = {
        1,                                 // B
        1,                                 // Mt
        Kt,                                // Kt
        num_output_tiles_per_core_group_2  // Nt
     };  // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set
         // Nt for simplicity

     auto matmul_multi_core_kernel_group_2_id = tt_metal::CreateKernel(
         program,
         "tt_metal/programming_examples/matmul_common/kernels/compute/bmm.cpp",
         core_group_2,
         tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args_group_2});
  }

  for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
    CoreCoord core = {i / num_cores_y, i % num_cores_y};

    uint32_t num_output_tiles_per_core = 0;
    if (core_group_1.contains(core)) {
        num_output_tiles_per_core = num_output_tiles_per_core_group_1;
    } else if (core_group_2.contains(core)) {
        num_output_tiles_per_core = num_output_tiles_per_core_group_2;
    } else {
        TT_ASSERT(false, "Core not in specified core ranges");
    }

    tt_metal::SetRuntimeArgs(
        program,
        reader_id,
        core,
        {a->address(),
         b->address(),
         Mt,
         Kt,
         Nt,
         MtKt,
         KtNt,
         1,
         uint32_t(0),
         num_tiles_written,
         num_output_tiles_per_core,
         MtNt});
    tt_metal::SetRuntimeArgs(program, writer_id, core, {c->address(), num_output_tiles_per_core, num_tiles_written});
    num_tiles_written += num_output_tiles_per_core;
  }

  EnqueueProgram(cq, program, false);
  Finish(cq);

  return result;
}

// static void sum_kernel_tt(TensorIterator& iter) {
// }

// REGISTER_TT_DISPATCH(sum_stub, &sum_kernel_tt)

}
