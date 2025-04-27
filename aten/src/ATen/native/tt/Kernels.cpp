#include <ATen/ops/add_native.h>
#include <ATen/ops/cat_native.h>
#include <ATen/ops/cos_native.h>
#include <ATen/ops/mean_native.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/pow_native.h>
#include <ATen/ops/relu_native.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/tt/TTDevice.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include <ATen/ATen.h>
#include <ATen/native/Resize.h>

using namespace tt;
using namespace tt::tt_metal;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

namespace at::native {

static CBHandle MakeCircularBuffer(
    Program& program, const CoreSpec& core, CBIndex cb, uint32_t size, uint32_t page_size, DataFormat format) {
    CircularBufferConfig cb_config = CircularBufferConfig(size, {{cb, format}}).set_page_size(cb, page_size);
    return CreateCircularBuffer(program, core, cb_config);
}

static CBHandle MakeCircularBufferBF16(Program& program, const CoreSpec& core, CBIndex cb, uint32_t n_tiles) {
  constexpr uint32_t tile_size = sizeof(bfloat16) * constants::TILE_HW;
  return MakeCircularBuffer(program, core, cb, n_tiles * tile_size, tile_size, DataFormat::Float16_b);
}

static CBHandle MakeCircularBufferF32(Program& program, const CoreSpec& core, CBIndex cb, uint32_t n_tiles) {
  constexpr uint32_t tile_size = sizeof(float) * constants::TILE_HW;
  return MakeCircularBuffer(program, core, cb, n_tiles * tile_size, tile_size, DataFormat::Float32);
}

static CoreRange AllDeviceCores(IDevice* device) {
  auto grid_size = device->compute_with_storage_grid_size();
  return CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
}

class ProgramBuilder {
public:
  ProgramBuilder(IDevice* device)
  : device_(device), program_(CreateProgram()), all_device_cores_(AllDeviceCores(device)) {}

  template<typename SetRuntimeArgsFn>
  void CreateKernels(
    uint32_t n_tiles,
    const std::string& reader_kernel_path,
    const std::string& writer_kernel_path,
    const std::string& compute_kernel_path,
    const std::vector<uint32_t>& reader_compile_time_args,
    const std::vector<uint32_t>& writer_compile_time_args,
    const std::vector<uint32_t>& compute_compile_time_args,
    const std::map<std::string, std::string>& compute_defines,
    SetRuntimeArgsFn set_runtime_args
  ) {
    auto reader = CreateKernel(
      program_,
      reader_kernel_path,
      all_device_cores_,
      DataMovementConfig{
          .processor = DataMovementProcessor::RISCV_0,
          .noc = NOC::RISCV_0_default,
          .compile_args = reader_compile_time_args});

    auto writer = CreateKernel(
      program_,
      writer_kernel_path,
      all_device_cores_,
      DataMovementConfig{
          .processor = DataMovementProcessor::RISCV_1,
          .noc = NOC::RISCV_1_default,
          .compile_args = writer_compile_time_args});

    MathFidelity math_fidelity = MathFidelity::HiFi4;
    auto compute = CreateKernel(
      program_,
      compute_kernel_path,
      all_device_cores_,
      ComputeConfig{.math_fidelity = math_fidelity, .math_approx_mode = false, .compile_args = compute_compile_time_args, .defines = compute_defines});

    auto grid_size = all_device_cores_.grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(grid_size, n_tiles, true);

    for (uint32_t i = 0, start_tile_id = 0; i < all_device_cores_.size(); i++) {
      CoreCoord core = {i % grid_size.x, i / grid_size.x};
      uint32_t num_tiles_per_core;

      if (core_group_1.contains(core)) {
        num_tiles_per_core = num_tiles_per_core_group_1;
      } else if (core_group_2.contains(core)) {
        num_tiles_per_core = num_tiles_per_core_group_2;
      } else {
        num_tiles_per_core = 0;
      }
      set_runtime_args(program_, core, reader, writer, compute, num_tiles_per_core, start_tile_id);
      start_tile_id += num_tiles_per_core;
    }
  }

  CBHandle AddCircularBuffer(CBIndex cb, DataFormat format, uint32_t n_tiles) {
    const uint32_t tile_size = datum_size(format) * constants::TILE_HW;
    return MakeCircularBuffer(program_, all_device_cores_, cb, n_tiles * tile_size, tile_size, format);
  }

  void Execute() {
    CommandQueue& cq = device_->command_queue();
    EnqueueProgram(cq, program_, true);
    Finish(cq);
  }

private:
  IDevice* device_;
  Program program_;
  CoreRange all_device_cores_;
};

enum class BinaryOpType {
  ADD,
  MUL,
};

static std::map<std::string, std::string> get_binary_op_defines(BinaryOpType op) {
  switch (op) {
  case BinaryOpType::ADD:
    return {{"ELTWISE_OP", "add_tiles"}, {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD"}};
  case BinaryOpType::MUL:
    return {{"ELTWISE_OP", "mul_tiles"}, {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWMUL"}};
  default:
    TORCH_INTERNAL_ASSERT(false, "Unrecognized BinaryOpType: ", static_cast<int64_t>(op));
  }
}

// Compute c <- a <op> b for tensors a, b, c with numel elements
static void EltwiseBinaryOp(BinaryOpType op, const std::shared_ptr<Buffer>& a, const std::shared_ptr<Buffer>& b, const std::shared_ptr<Buffer>& c, int64_t numel, IDevice* device) {
  ProgramBuilder builder(device);

  const uint32_t cb_num_tiles = 4;
  builder.AddCircularBuffer(CBIndex::c_0, DataFormat::Float16_b, cb_num_tiles);
  builder.AddCircularBuffer(CBIndex::c_1, DataFormat::Float16_b, cb_num_tiles);
  builder.AddCircularBuffer(CBIndex::c_2, DataFormat::Float16_b, cb_num_tiles);

  std::vector<uint32_t> reader_compile_time_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1};
  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)CBIndex::c_2};
  std::vector<uint32_t> compute_compile_time_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1, (uint32_t)CBIndex::c_2};
  auto compute_defines = get_binary_op_defines(op);

  const uint32_t n_tiles = (numel + ::tt::constants::TILE_HW - 1) / ::tt::constants::TILE_HW;

  builder.CreateKernels(
    n_tiles,
    // TODO: The paths are currently hard-coded, figure out how to fix it
    "/root/pytorch/aten/src/ATen/native/tt/kernels/dataflow/binary_eltwise_reader_row_major_to_tiles.cpp",
    "/root/pytorch/aten/src/ATen/native/tt/kernels/dataflow/eltwise_writer_row_major_to_tiles.cpp",
    "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp",
    reader_compile_time_args,
    writer_compile_time_args,
    compute_compile_time_args,
    compute_defines,
    [a, b, c](const Program& program, const CoreCoord& core, KernelHandle reader, KernelHandle writer, KernelHandle compute, uint32_t num_tiles, uint32_t start_tile_id) {
      SetRuntimeArgs(program, reader, core, {a->address(), b->address(), num_tiles, start_tile_id});
      SetRuntimeArgs(program, writer, core, {c->address(), num_tiles, start_tile_id});
      SetRuntimeArgs(program, compute, core, {num_tiles, 1});
    }
  );

  builder.Execute();
}

enum class UnaryOpType {
  COS,
  SIN,
  RELU,
  POW,
};

static std::map<std::string, std::string> get_unary_op_defines(UnaryOpType op, const std::vector<float>& params) {
  switch (op) {
  case UnaryOpType::COS:
    return {{"SFPU_OP_TRIG_FAMILY_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "cos_tile_init(); cos_tile(0);"}};
  case UnaryOpType::SIN:
    return {{"SFPU_OP_TRIG_FAMILY_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "sin_tile_init(); sin_tile(0);"}};
  case UnaryOpType::RELU:
    return {{"SFPU_OP_RELU_FAMILY_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "relu_tile_init(); relu_tile(0);"}};
  case UnaryOpType::POW:
    return {{"SFPU_OP_COMPUTE_KERNEL_API_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", std::format("power_tile_init(); power_tile(0, {}u);", (uint32_t)params[0])}};
  default:
    TORCH_INTERNAL_ASSERT(false, "Unrecognized UnaryOpType: ", static_cast<int64_t>(op));
  }
}

static void EltwiseUnaryOp(UnaryOpType op, const std::shared_ptr<Buffer>& a, const std::shared_ptr<Buffer>& b, int64_t numel, const std::vector<float>& params, IDevice* device) {
  ProgramBuilder builder(device);

  const uint32_t cb_num_tiles = 2;
  builder.AddCircularBuffer(CBIndex::c_0, DataFormat::Float16_b, cb_num_tiles);
  builder.AddCircularBuffer(CBIndex::c_1, DataFormat::Float16_b, cb_num_tiles);

  std::vector<uint32_t> reader_compile_time_args = {(uint32_t)CBIndex::c_0};
  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)CBIndex::c_1};
  std::vector<uint32_t> compute_compile_time_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1};
  auto compute_defines = get_unary_op_defines(op, params);

  const uint32_t n_tiles = (numel + ::tt::constants::TILE_HW - 1) / ::tt::constants::TILE_HW;

  builder.CreateKernels(
    n_tiles,
    // TODO: The paths are currently hard-coded, figure out how to fix it
    "/root/pytorch/aten/src/ATen/native/tt/kernels/dataflow/unary_eltwise_reader_row_major_to_tiles.cpp",
    "/root/pytorch/aten/src/ATen/native/tt/kernels/dataflow/eltwise_writer_row_major_to_tiles.cpp",
    "/root/pytorch/aten/src/ATen/native/tt/kernels/compute/eltwise_sfpu_multi_core.cpp",
    reader_compile_time_args,
    writer_compile_time_args,
    compute_compile_time_args,
    compute_defines,
    [a, b](const Program& program, const CoreCoord& core, KernelHandle reader, KernelHandle writer, KernelHandle compute, uint32_t num_tiles, uint32_t start_tile_id) {
      SetRuntimeArgs(program, reader, core, {a->address(), num_tiles, start_tile_id});
      SetRuntimeArgs(program, writer, core, {b->address(), num_tiles, start_tile_id});
      SetRuntimeArgs(program, compute, core, {num_tiles, start_tile_id});
    }
  );

  builder.Execute();
}

// Elementwise addition

at::Tensor & add_out_tt(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();

  auto a = allocator->get_buffer(self.data_ptr());
  auto b = allocator->get_buffer(other.data_ptr());
  auto c = allocator->get_buffer(out.data_ptr());

  EltwiseBinaryOp(BinaryOpType::ADD, a, b, c, self.numel(), device);

  return out;
}

at::Tensor & mul_out_tt(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();

  auto a = allocator->get_buffer(self.data_ptr());
  auto b = allocator->get_buffer(other.data_ptr());
  auto c = allocator->get_buffer(out.data_ptr());

  EltwiseBinaryOp(BinaryOpType::MUL, a, b, c, self.numel(), device);

  return out;
}

// RELU

Tensor relu_tt(const Tensor& self) {
  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();

  auto out = at::empty_like(self);
  auto a = allocator->get_buffer(self.data_ptr());
  auto b = allocator->get_buffer(out.data_ptr());

  EltwiseUnaryOp(UnaryOpType::RELU, a, b, self.numel(), {}, device);

  return out;
}

// COS

at::Tensor & cos_out_tt(const at::Tensor & self, at::Tensor & out) {
  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();

  auto a = allocator->get_buffer(self.data_ptr());
  auto b = allocator->get_buffer(out.data_ptr());

  EltwiseUnaryOp(UnaryOpType::COS, a, b, self.numel(), {}, device);

  return out;
}

// SIN

at::Tensor& at::native::sin_out_tt(at::Tensor const& self, at::Tensor& out) {
  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();

  auto a = allocator->get_buffer(self.data_ptr());
  auto b = allocator->get_buffer(out.data_ptr());

  EltwiseUnaryOp(UnaryOpType::SIN, a, b, self.numel(), {}, device);
  
  return out;
}

at::Tensor & pow_tensor_scalar_out_tt(const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out) {
  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();

  auto a = allocator->get_buffer(self.data_ptr());
  auto b = allocator->get_buffer(out.data_ptr());

  EltwiseUnaryOp(UnaryOpType::POW, a, b, self.numel(), {exponent.to<float>()}, device);
  
  return out;
}

// matmul -- this is a very naive but also simple implementation and not optimized yet at all
at::Tensor& mm_out_tt(const at::Tensor & self, const at::Tensor & mat2, at::Tensor &result) {
  uint32_t M = self.size(0);
  int64_t K = self.size(1);
  AT_ASSERT(mat2.size(0) == K);
  uint32_t N = mat2.size(1);

  uint32_t Mt = M / constants::TILE_HEIGHT;
  uint32_t Kt = K / constants::TILE_WIDTH;
  uint32_t Nt = N / constants::TILE_WIDTH;

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

  std::vector<uint32_t> reader_compile_time_args = {(uint32_t)2 /* bytes in bfloat16 */, (uint32_t) !mat2.is_contiguous() /* whether b is transposed */};
  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)CBIndex::c_16, (uint32_t)1};

  auto reader_id = tt_metal::CreateKernel(
    program,
    // TODO: The path is currently hard-coded, figure out how to fix it
    "/root/pytorch/aten/src/ATen/native/tt/kernels/dataflow/matmul_reader_row_major_to_tiles.cpp",
    all_cores,
    tt_metal::DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = reader_compile_time_args});

  auto writer_id = tt_metal::CreateKernel(
    program,
    // TODO: The path is currently hard-coded, figure out how to fix it
    "/root/pytorch/aten/src/ATen/native/tt/kernels/dataflow/matmul_writer_row_major_to_tiles.cpp",
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
    num_output_tiles_per_core_group_1, // Nt
    (uint32_t) !mat2.is_contiguous() // whether b is transposed
  };  // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set Nt
      // for simplicity

  auto matmul_multi_core_kernel_group_1_id = tt_metal::CreateKernel(
    program,
    // TODO: The path is currently hard-coded, figure out how to fix it
    "/root/pytorch/aten/src/ATen/native/tt/kernels/compute/bmm.cpp",
    core_group_1,
    tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .fp32_dest_acc_en = true, .compile_args = compute_args_group_1});

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
         M,
         Kt,
         N,
         num_tiles_written,
        });
    tt_metal::SetRuntimeArgs(program, writer_id, core, {c->address(), num_output_tiles_per_core, num_tiles_written, (uint32_t)M, (uint32_t)N});
    num_tiles_written += num_output_tiles_per_core;
  }

  EnqueueProgram(cq, program, false);
  Finish(cq);

  return result;
}

at::Tensor & addmm_out_tt(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");
  TORCH_CHECK(
    mat1.dtype() == mat2.dtype(),
    "expected mat1 and mat2 to have the same dtype, but got: ", mat1.dtype(), " != ", mat2.dtype()
  );
  // We first start with the very naive implementation here
  // TODO: handle alpha, beta != 1.0
  mm_out_tt(mat1, mat2, out);
  add_out_tt(out, self, beta, out);
  return out;
}

Tensor& uniform_tt_(Tensor& self, double from, double to, std::optional<Generator> gen) {
  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();
  CommandQueue& cq = device->command_queue();
  Program program = CreateProgram();

  const uint32_t n_tiles = (self.numel() + ::tt::constants::TILE_HW - 1) / ::tt::constants::TILE_HW;
  auto a = allocator->get_buffer(self.data_ptr());

  auto grid_size = device->compute_with_storage_grid_size();
  uint32_t num_cores_x = grid_size.x;
  uint32_t num_cores_y = grid_size.y;
  uint32_t num_cores_total = num_cores_x * num_cores_y;
  auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

  constexpr auto intermed_cb_id = CBIndex::c_24;
  constexpr uint32_t intermed_num_tiles = 2;
  CBHandle cb_intermed = MakeCircularBufferF32(program, all_device_cores, intermed_cb_id, intermed_num_tiles);
  constexpr auto dst_cb_id = CBIndex::c_0;
  constexpr uint32_t in_out_num_tiles = 1;
  CBHandle cb_output = MakeCircularBufferBF16(program, all_device_cores, dst_cb_id, in_out_num_tiles);

  const uint32_t output_is_dram = 1;
  const std::vector<uint32_t> writer_compile_time_args{intermed_cb_id, dst_cb_id, output_is_dram};
  std::map<string, string> writer_defines = {{"OUTPUT_DTYPE_BFLOAT16", "1"}};
  const std::vector<uint32_t> compute_compile_time_args{intermed_cb_id};

  auto writer = CreateKernel(
    program,
    // TODO: The path is currently hard-coded, figure out how to fix it
    "/root/pytorch/aten/src/ATen/native/tt/kernels/dataflow/writer_uniform_row_major.cpp",
    all_device_cores,
    WriterDataMovementConfig(writer_compile_time_args, writer_defines));

  auto compute = CreateKernel(
    program,
    "ttnn/cpp/ttnn/operations/uniform/device/kernels/compute_uniform.cpp",
    all_device_cores,
    ComputeConfig{
      .fp32_dest_acc_en = true,  // if fp32_dest_acc_en set to false a precision error may occur which makes
                                 // generated number out of range [from, to)
      .math_approx_mode = false, .compile_args = compute_compile_time_args, .defines = {}
    });

  auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
    split_work_to_cores(grid_size, n_tiles);
  auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);

  uint32_t tile_offset = 0;
  for (int i = 0; i < cores.size(); ++i) {
    const auto& core = cores[i];
    uint32_t units_per_core;
    if (core_group_1.contains(core)) {
        units_per_core = units_per_core_group_1;
    } else if (core_group_2.contains(core)) {
        units_per_core = units_per_core_group_2;
    } else {
      AT_ASSERT(false, "Core not in specified core ranges");
    }

    const float eps = 1e-6;
    union {
        float f;
        uint32_t u;
    } f2u_from, f2u_to;
    f2u_from.f = static_cast<float>(from);
    f2u_to.f = static_cast<float>(to) - eps;  // -eps make sure that generated number is < operation_attributes.to

    // Each core has its own seed to increase the number of generated random numbers
    uint32_t seed = 42 + i;

    std::vector<uint32_t> compute_runtime_args = {seed, f2u_from.u, f2u_to.u, tile_offset, units_per_core};
    SetRuntimeArgs(program, compute, core, compute_runtime_args);

    std::vector<uint32_t> writer_runtime_args = {a->address(), tile_offset, units_per_core};
    SetRuntimeArgs(program, writer, core, writer_runtime_args);

    tile_offset += units_per_core;
  }

  EnqueueProgram(cq, program, true);

  Finish(cq);

  return self;
}

Tensor index_select_tt(const Tensor& self, int64_t dim, const Tensor& index) {
  TORCH_CHECK(index.dim() == 1, "Index is supposed to be a vector");
  TORCH_CHECK(self.stride(dim) % constants::FACE_WIDTH == 0, "Size of vectors to be selected currently needs to be divisible by FACE_WIDTH");

  auto contiguous_index = index.contiguous();
  uint64_t num_indices = index.numel();
  std::vector<int64_t> new_size = self.sizes().vec();
  new_size[dim] = num_indices;
  Tensor out = at::empty(new_size, self.options());

  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();
  CommandQueue& cq = device->command_queue();
  Program program = CreateProgram();

  auto input = allocator->get_buffer(self.data_ptr());
  auto output = allocator->get_buffer(out.data_ptr());
  auto indices = allocator->get_buffer(index.data_ptr());

  auto grid_size = device->compute_with_storage_grid_size();
  uint32_t num_cores_x = grid_size.x;
  uint32_t num_cores_y = grid_size.y;
  uint32_t num_cores_total = num_cores_x * num_cores_y;
  auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

  CBHandle cb_indices = MakeCircularBuffer(program, all_device_cores, CBIndex::c_0, 2 * constants::FACE_WIDTH, 2 * constants::FACE_WIDTH, DataFormat::UInt32);

  // Distribute the indices onto the cores
  uint64_t num_pages = num_indices / constants::FACE_WIDTH; // TODO: Use ceil here and adapt boundary
  auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
    split_work_to_cores(grid_size, num_pages);

  constexpr uint32_t datum_size_bytes = sizeof(uint32_t); // TODO: Fix this!

  // Create a buffer in SRAM which will be used as temporary storage to copy over data from input to output
  // For now we will just make it size FACE_WIDTH for simplicity but we might need to optimize that later
  tt_metal::InterleavedBufferConfig l1_config{
    .device = device,
    .size = datum_size_bytes * constants::FACE_WIDTH,
    .page_size = datum_size_bytes * constants::FACE_WIDTH,
    .buffer_type = tt_metal::BufferType::L1};
  auto l1_buffer = CreateBuffer(l1_config);

  std::vector<uint32_t> reader_compile_time_args = {(uint32_t)CBIndex::c_0};
  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)CBIndex::c_0};

  auto reader_id = tt_metal::CreateKernel(
    program,
    // TODO: The path is currently hard-coded, figure out how to fix it
    "/root/pytorch/aten/src/ATen/native/tt/kernels/dataflow/index_select_reader_row_major.cpp",
    all_device_cores,
    tt_metal::DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = reader_compile_time_args});

  auto writer_id = tt_metal::CreateKernel(
    program,
    // TODO: The path is currently hard-coded, figure out how to fix it
    "/root/pytorch/aten/src/ATen/native/tt/kernels/dataflow/index_select_writer_row_major.cpp",
    all_device_cores,
    tt_metal::DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = writer_compile_time_args});

  auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y);
  for (uint32_t i = 0, start_page_id = 0; i < num_cores_total; i++) {
    CoreCoord core = {i / num_cores_y, i % num_cores_y};

    uint32_t num_pages_per_core;
    if (core_group_1.contains(core)) {
      num_pages_per_core = num_pages_per_core_group_1;
    } else if (core_group_2.contains(core)) {
      num_pages_per_core = num_pages_per_core_group_2;
    } else {
      num_pages_per_core = 0;
    }

    std::vector<uint32_t> reader_args = {indices->address(), num_pages_per_core, start_page_id};
    tt_metal::SetRuntimeArgs(program, reader_id, core, reader_args);
    std::vector<uint32_t> writer_args = {input->address(), output->address(), l1_buffer->address(), num_pages_per_core, start_page_id, (uint32_t) self.stride(dim)};
    tt_metal::SetRuntimeArgs(program, writer_id, core, writer_args);

    start_page_id += num_pages_per_core;
  }

  EnqueueProgram(cq, program, true);

  Finish(cq);

  return out;
}

at::Tensor & cat_out_tt(const at::ITensorListRef & tensors, int64_t dim, at::Tensor & out) {
  auto inputs = tensors.materialize();

  int64_t num_tensors = inputs.size();
  uint32_t num_pages = out.numel() / constants::FACE_WIDTH;
  uint32_t num_output_pages_per_block = out.size(dim) * out.stride(dim) / constants::FACE_WIDTH;
  std::vector<uint32_t> num_pages_per_block(num_tensors);

  for (int i = 0; i < num_tensors; ++i) {
    auto& tensor = inputs[i].get();
    num_pages_per_block[i] = tensor.size(dim) * tensor.stride(dim) / constants::FACE_WIDTH;
  }

  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();
  CommandQueue& cq = device->command_queue();
  Program program = CreateProgram();

  auto output = allocator->get_buffer(out.data_ptr());

  auto grid_size = device->compute_with_storage_grid_size();
  uint32_t num_cores_x = grid_size.x;
  uint32_t num_cores_y = grid_size.y;
  uint32_t num_cores_total = num_cores_x * num_cores_y;
  auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

  auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
    split_work_to_cores(grid_size, num_pages);

  constexpr uint32_t datum_size_bytes = sizeof(bfloat16);

  // Create a buffer in SRAM which will be used as temporary storage to copy over data from input to output
  // For now we will just make it size FACE_WIDTH for simplicity but we might need to optimize that later
  tt_metal::InterleavedBufferConfig l1_config{
    .device = device,
    .size = datum_size_bytes * constants::FACE_WIDTH,
    .page_size = datum_size_bytes * constants::FACE_WIDTH,
    .buffer_type = tt_metal::BufferType::L1};
  auto l1_buffer = CreateBuffer(l1_config);

  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)num_tensors};

  auto writer_id = tt_metal::CreateKernel(
    program,
    // TODO: The path is currently hard-coded, figure out how to fix it
    "/root/pytorch/aten/src/ATen/native/tt/kernels/dataflow/writer_cat_row_major.cpp",
    all_device_cores,
    tt_metal::DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = writer_compile_time_args});

  std::vector<uint32_t> common_writer_args = {(uint32_t)dim, 0, 0, 0, 0, output->address(), l1_buffer->address()};
  for (int i = 0; i < num_tensors; ++i) {
    auto src = allocator->get_buffer(inputs[i].get().data_ptr());
    common_writer_args.push_back(src->address());
  }
  common_writer_args.insert(common_writer_args.end(), num_pages_per_block.begin(), num_pages_per_block.end());

  std::vector<uint32_t> src_page_id(num_tensors);
  auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y);
  for (uint32_t i = 0, start_page_id = 0; i < num_cores_total; i++) {
    CoreCoord core = {i / num_cores_y, i % num_cores_y};

    uint32_t num_pages_per_core;
    if (core_group_1.contains(core)) {
      num_pages_per_core = num_pages_per_core_group_1;
    } else if (core_group_2.contains(core)) {
      num_pages_per_core = num_pages_per_core_group_2;
    } else {
      std::vector<uint32_t> writer_args(7 + 3 * num_tensors, 0);
      SetRuntimeArgs(program, writer_id, core, writer_args);
      continue;
    }

    uint32_t block_id = start_page_id / num_output_pages_per_block;
    uint32_t page_id_within_block = start_page_id % num_output_pages_per_block;
    uint32_t curr_tensor = 0;
    uint32_t curr_tensor_page_id = 0;
    for (int i = 0; i < num_tensors; ++i) {
      src_page_id[i] = block_id * num_pages_per_block[i];
      if (page_id_within_block == 0) {
        continue;
      } else if (page_id_within_block >= num_pages_per_block[i]) {
        src_page_id[i] += num_pages_per_block[i];
        page_id_within_block -= num_pages_per_block[i];
        curr_tensor = i + 1;
      } else {
        src_page_id[i] += page_id_within_block;
        curr_tensor = i;
        curr_tensor_page_id = page_id_within_block;
        page_id_within_block = 0;
      }
    }

    std::vector<uint32_t> writer_args = common_writer_args;
    writer_args[1] = num_pages_per_core;
    writer_args[2] = start_page_id;
    writer_args[3] = curr_tensor;
    writer_args[4] = curr_tensor_page_id;
    writer_args.insert(writer_args.end(), src_page_id.begin(), src_page_id.end());

    tt_metal::SetRuntimeArgs(program, writer_id, core, writer_args);
    start_page_id += num_pages_per_core;
  }

  EnqueueProgram(cq, program, true);

  Finish(cq);

  return out;
}

at::Tensor & mean_out_tt(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  bfloat16 bfloat_scale_value = bfloat16(1.0f); // TODO: Put the right scale for the mean here
  uint32_t packed_scale_value = pack_two_bfloat16_into_uint32({bfloat_scale_value, bfloat_scale_value});

  uint32_t W = self.size(3);
  uint32_t H = self.size(2);
  uint32_t NC = self.size(1) * self.size(0);
  uint32_t HW = H * W;

  uint32_t Wt = W / constants::TILE_WIDTH;
  uint32_t Ht = H / constants::TILE_HEIGHT;

  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();

  auto a = allocator->get_buffer(self.data_ptr());
  auto b = allocator->get_buffer(out.data_ptr());

  ProgramBuilder builder(device);

  const uint32_t cb_num_tiles = 2;
  builder.AddCircularBuffer(CBIndex::c_0, DataFormat::Float16_b, cb_num_tiles);
  builder.AddCircularBuffer(CBIndex::c_2, DataFormat::Float16_b, cb_num_tiles);
  builder.AddCircularBuffer(CBIndex::c_3, DataFormat::Float16_b, cb_num_tiles);

  std::vector<uint32_t> reader_compile_time_args = {packed_scale_value};
  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)CBIndex::c_3};
  std::vector<uint32_t> compute_compile_time_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1};

  const uint32_t num_rows = NC * Ht;

  builder.CreateKernels(
    num_rows,
    // TODO: The paths are currently hard-coded, figure out how to fix it
    "/root/pytorch/aten/src/ATen/native/tt/kernels/dataflow/reduce_reader_row_major_to_tiles.cpp",
    "/root/pytorch/aten/src/ATen/native/tt/kernels/dataflow/reduce_writer_row_major.cpp",
    "/root/pytorch/aten/src/ATen/native/tt/kernels/compute/reduce.cpp",
    reader_compile_time_args,
    writer_compile_time_args,
    compute_compile_time_args,
    {{}},
    [a, b, Wt](const Program& program, const CoreCoord& core, KernelHandle reader, KernelHandle writer, KernelHandle compute, uint32_t num_rows_per_core, uint32_t start_row_id) {
      uint32_t num_tiles_per_core = num_rows_per_core * Wt;
      uint32_t start_tile_id = start_row_id * Wt;
      SetRuntimeArgs(program, reader, core, {a->address(), num_tiles_per_core, start_tile_id});
      SetRuntimeArgs(program, writer, core, {b->address(), num_tiles_per_core, start_tile_id});
      SetRuntimeArgs(program, compute, core, {num_rows_per_core, Wt, 1});
    }
  );

  builder.Execute();

  return out;
}

// static void sum_kernel_tt(TensorIterator& iter) {
// }

// REGISTER_TT_DISPATCH(sum_stub, &sum_kernel_tt)

}
