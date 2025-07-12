#include <ATen/ops/add_native.h>
#include <ATen/ops/cat_native.h>
#include <ATen/ops/cos_native.h>
#include <ATen/ops/mean_native.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/pow_native.h>
#include <ATen/ops/relu_native.h>
#include <ATen/ops/where_native.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/tt/TTDevice.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include <ATen/native/tt/Kernels.h>
#include <ATen/native/TensorCompare.h>

#include <ATen/ATen.h>
#include <ATen/native/Resize.h>

#define TT_NOT_IMPLEMENTED() AT_ASSERT(false && "not implemented")

using namespace tt;
using namespace tt::tt_metal;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

namespace at::native {

static CBHandle MakeCircularBuffer(
    Program& program, const CoreSpec& core, CBIndex cb, uint32_t size, uint32_t page_size, DataFormat format) {
    CircularBufferConfig cb_config = CircularBufferConfig(size, {{cb, format}}).set_page_size(cb, page_size);
    return CreateCircularBuffer(program, core, cb_config);
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
    // The defines are shared among all kernels (reader, writer, compute) but it could be split up if needed
    const std::map<std::string, std::string>& defines,
    SetRuntimeArgsFn set_runtime_args
  ) {
    KernelHandle reader;
    if (!reader_kernel_path.empty()) {
      reader = CreateKernel(
        program_,
        reader_kernel_path,
        all_device_cores_,
        DataMovementConfig{
          .processor = DataMovementProcessor::RISCV_0,
          .noc = NOC::RISCV_0_default,
          .compile_args = reader_compile_time_args,
          .defines = defines});
    }

    auto writer = CreateKernel(
      program_,
      writer_kernel_path,
      all_device_cores_,
      DataMovementConfig{
          .processor = DataMovementProcessor::RISCV_1,
          .noc = NOC::RISCV_1_default,
          .compile_args = writer_compile_time_args,
          .defines = defines});

    MathFidelity math_fidelity = MathFidelity::HiFi4;
    KernelHandle compute;
    if (!compute_kernel_path.empty()) {
      compute = CreateKernel(
        program_,
        compute_kernel_path,
        all_device_cores_,
        ComputeConfig{
          .math_fidelity = math_fidelity,
          .fp32_dest_acc_en = true,
          .math_approx_mode = false,
          .compile_args = compute_compile_time_args,
          .defines = defines});
    }

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
static void EltwiseBinaryOp(BinaryOpType op, const at::Tensor& a, const at::Tensor& b, const at::Tensor& c) {
  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();
  ProgramBuilder builder(device);

  auto defines = get_binary_op_defines(op);
  // Either the address of b if b is a tensor or a packed value if b is a scalar
  uint32_t b_val;
  auto a_buf = allocator->get_buffer(a);
  if (b.dim() == 0 && b.device().is_cpu()) {
    auto val = bfloat16(b.item().to<float>());
    b_val = pack_two_bfloat16_into_uint32({val, val});
    defines["BINARY_ELTWISE_SCALAR_OP"] = "1";
  } else {
    auto b_buf = allocator->get_buffer(b);
    b_val = b_buf->address();
  }
  auto c_buf = allocator->get_buffer(c);

  const uint32_t cb_num_tiles = 4;
  builder.AddCircularBuffer(CBIndex::c_0, DataFormat::Float16_b, cb_num_tiles);
  builder.AddCircularBuffer(CBIndex::c_1, DataFormat::Float16_b, cb_num_tiles);
  builder.AddCircularBuffer(CBIndex::c_2, DataFormat::Float16_b, cb_num_tiles);

  std::vector<uint32_t> reader_compile_time_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1};
  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)CBIndex::c_2};
  std::vector<uint32_t> compute_compile_time_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1, (uint32_t)CBIndex::c_2};

  const uint32_t n_tiles = (a.numel() + ::tt::constants::TILE_HW - 1) / ::tt::constants::TILE_HW;

  builder.CreateKernels(
    n_tiles,
    // TODO: The paths are currently hard-coded, figure out how to fix it
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/dataflow/binary_eltwise_reader_row_major_to_tiles.cpp",
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/dataflow/eltwise_writer_row_major_to_tiles.cpp",
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/compute/eltwise_binary_kernel.cpp",
    reader_compile_time_args,
    writer_compile_time_args,
    compute_compile_time_args,
    defines,
    [a_buf, b_val, c_buf](const Program& program, const CoreCoord& core, KernelHandle reader, KernelHandle writer, KernelHandle compute, uint32_t num_tiles, uint32_t start_tile_id) {
      SetRuntimeArgs(program, reader, core, {a_buf->address(), b_val, num_tiles, start_tile_id});
      SetRuntimeArgs(program, writer, core, {c_buf->address(), num_tiles, start_tile_id});
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
  RSQRT,
  NEG,
  FILL,
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
    return {{"SFPU_OP_COMPUTE_KERNEL_API_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", fmt::format("power_tile_init(); power_tile(0, {}u);", (uint32_t)params[0])}};
  case UnaryOpType::RSQRT:
    return {{"SFPU_OP_COMPUTE_KERNEL_API_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "rsqrt_tile_init(); rsqrt_tile(0);"}};
  case UnaryOpType::NEG:
    return {{"SFPU_OP_NEG_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "negative_tile_init(); negative_tile(0);"}};
  case UnaryOpType::FILL:
    return {{"SFPU_OP_FILL_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", fmt::format("fill_tile_init(); fill_tile_bitcast(0, {}u);", std::bit_cast<uint32_t>(params[0]))}};
  default:
    TORCH_INTERNAL_ASSERT(false, "Unrecognized UnaryOpType: ", static_cast<int64_t>(op));
  }
}

static void EltwiseUnaryOp(UnaryOpType op, const at::Tensor& a, const at::Tensor& b, const std::vector<float>& params) {
  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();
  ProgramBuilder builder(device);

  auto a_buf = allocator->get_buffer(a);
  auto b_buf = allocator->get_buffer(b);

  const uint32_t cb_num_tiles = 2;
  builder.AddCircularBuffer(CBIndex::c_0, DataFormat::Float16_b, cb_num_tiles);
  builder.AddCircularBuffer(CBIndex::c_1, DataFormat::Float16_b, cb_num_tiles);

  std::vector<uint32_t> reader_compile_time_args = {(uint32_t)CBIndex::c_0};
  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)CBIndex::c_1};
  std::vector<uint32_t> compute_compile_time_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1};
  auto defines = get_unary_op_defines(op, params);

  const uint32_t n_tiles = (a.numel() + ::tt::constants::TILE_HW - 1) / ::tt::constants::TILE_HW;

  builder.CreateKernels(
    n_tiles,
    // TODO: The paths are currently hard-coded, figure out how to fix it
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/dataflow/unary_eltwise_reader_row_major_to_tiles.cpp",
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/dataflow/eltwise_writer_row_major_to_tiles.cpp",
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/compute/eltwise_sfpu_multi_core.cpp",
    reader_compile_time_args,
    writer_compile_time_args,
    compute_compile_time_args,
    defines,
    [a_buf, b_buf](const Program& program, const CoreCoord& core, KernelHandle reader, KernelHandle writer, KernelHandle compute, uint32_t num_tiles, uint32_t start_tile_id) {
      SetRuntimeArgs(program, reader, core, {a_buf->address(), num_tiles, start_tile_id});
      SetRuntimeArgs(program, writer, core, {b_buf->address(), num_tiles, start_tile_id});
      SetRuntimeArgs(program, compute, core, {num_tiles, start_tile_id});
    }
  );

  builder.Execute();
}

// Elementwise addition

at::Tensor& add_out_tt(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, at::Tensor& out) {
  EltwiseBinaryOp(BinaryOpType::ADD, self, other, out);
  return out;
}

at::Tensor& mul_out_tt(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
  EltwiseBinaryOp(BinaryOpType::MUL, self, other, out);
  return out;
}

// RELU

Tensor relu_tt(const Tensor& self) {
  auto out = at::empty_like(self);
  EltwiseUnaryOp(UnaryOpType::RELU, self, out, {});
  return out;
}

// COS

at::Tensor & cos_out_tt(const at::Tensor & self, at::Tensor & out) {
  EltwiseUnaryOp(UnaryOpType::COS, self, out, {});
  return out;
}

// SIN

at::Tensor& at::native::sin_out_tt(at::Tensor const& self, at::Tensor& out) {
  EltwiseUnaryOp(UnaryOpType::SIN, self, out, {});
  return out;
}

at::Tensor & pow_tensor_scalar_out_tt(const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out) {
  EltwiseUnaryOp(UnaryOpType::POW, self, out, {exponent.to<float>()});
  return out;
}

at::Tensor & rsqrt_out_tt(const at::Tensor & self, at::Tensor & out) {
  EltwiseUnaryOp(UnaryOpType::RSQRT, self, out, {});
  return out;
}

at::Tensor & neg_out_tt(const at::Tensor & self, at::Tensor & out) {
  EltwiseUnaryOp(UnaryOpType::NEG, self, out, {});
  return out;
}

at::Tensor & fill_scalar_tt(at::Tensor & self, const at::Scalar & value) {
  // TODO: This is currently reading self as an argument, which is not neccessary.
  // We should extend the kernel so it can skip reading the input.
  EltwiseUnaryOp(UnaryOpType::FILL, self, self, {value.to<float>()});
  return self;
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
  ProgramBuilder builder(device);

  auto a_buf = allocator->get_buffer(self);
  auto b_buf = allocator->get_buffer(mat2);
  auto c_buf = allocator->get_buffer(result);

  const uint32_t cb_num_tiles = 2;
  builder.AddCircularBuffer(CBIndex::c_0, DataFormat::Float16_b, cb_num_tiles);
  builder.AddCircularBuffer(CBIndex::c_1, DataFormat::Float16_b, cb_num_tiles);
  builder.AddCircularBuffer(CBIndex::c_16, DataFormat::Float16_b, cb_num_tiles);

  std::vector<uint32_t> reader_compile_time_args = {(uint32_t)2 /* bytes in bfloat16 */, (uint32_t) !mat2.is_contiguous() /* whether b is transposed */};
  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)CBIndex::c_16, (uint32_t)1};
  std::vector<uint32_t> compute_compile_time_args = {(uint32_t) !mat2.is_contiguous() /* whether b is transposed */};

  const uint32_t n_tiles = (M * N) / constants::TILE_HW;

  builder.CreateKernels(
    n_tiles,
    // TODO: The paths are currently hard-coded, figure out how to fix it
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/dataflow/matmul_reader_row_major_to_tiles.cpp",
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/dataflow/matmul_writer_row_major_to_tiles.cpp",
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/compute/gemm.cpp",
    reader_compile_time_args,
    writer_compile_time_args,
    compute_compile_time_args,
    {},
    [a_buf, b_buf, c_buf, M, N, Kt](const Program& program, const CoreCoord& core, KernelHandle reader, KernelHandle writer, KernelHandle compute, uint32_t num_tiles, uint32_t start_tile_id) {
      SetRuntimeArgs(program, reader, core, {a_buf->address(), b_buf->address(), M, Kt, N, start_tile_id, num_tiles});
      SetRuntimeArgs(program, writer, core, {c_buf->address(), num_tiles, start_tile_id, M, N});
      SetRuntimeArgs(program, compute, core, {num_tiles, Kt});
    }
  );

  builder.Execute();

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

at::Tensor & bmm_out_tt(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
  TT_NOT_IMPLEMENTED();
  return out;
}

Tensor& uniform_tt_(Tensor& self, double from, double to, std::optional<Generator> gen) {
  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();
  ProgramBuilder builder(device);

  auto a_buf = allocator->get_buffer(self);

  builder.AddCircularBuffer(CBIndex::c_0, DataFormat::Float16_b, 1);
  builder.AddCircularBuffer(CBIndex::c_24, DataFormat::Float32, 2);

  const uint32_t output_is_dram = 1;
  const std::vector<uint32_t> writer_compile_time_args = {(uint32_t)CBIndex::c_24, (uint32_t)CBIndex::c_0, output_is_dram};
  const std::vector<uint32_t> compute_compile_time_args = {(uint32_t)CBIndex::c_24};
  std::map<string, string> defines = {{"OUTPUT_DTYPE_BFLOAT16", "1"}};

  const uint32_t n_tiles = (self.numel() + ::tt::constants::TILE_HW - 1) / ::tt::constants::TILE_HW;

  builder.CreateKernels(
    n_tiles,
    // TODO: The paths are currently hard-coded, figure out how to fix it
    "",
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/dataflow/writer_uniform_row_major.cpp",
    "ttnn/cpp/ttnn/operations/uniform/device/kernels/compute_uniform.cpp",
    {},
    writer_compile_time_args,
    compute_compile_time_args,
    defines,
    [a_buf, from, to](const Program& program, const CoreCoord& core, KernelHandle reader, KernelHandle writer, KernelHandle compute, uint32_t num_tiles, uint32_t start_tile_id) {
      const float eps = 1e-6;
      union {
          float f;
          uint32_t u;
      } f2u_from, f2u_to;
      f2u_from.f = static_cast<float>(from);
      f2u_to.f = static_cast<float>(to) - eps;  // -eps make sure that generated number is < operation_attributes.to

      // Each core has its own seed to increase the number of generated random numbers
      uint32_t seed = 42 + start_tile_id;

      SetRuntimeArgs(program, writer, core, {a_buf->address(), start_tile_id, num_tiles});
      SetRuntimeArgs(program, compute, core, {seed, f2u_from.u, f2u_to.u, start_tile_id, num_tiles});
    }
  );

  builder.Execute();

  return self;
}

Tensor index_select_tt(const Tensor& self, int64_t dim, const Tensor& index) {
  TORCH_CHECK(index.dim() == 1, "Index is supposed to be a vector");
  TORCH_CHECK(self.stride(dim) % constants::TILE_WIDTH == 0, "Size of vectors to be selected currently needs to be divisible by TILE_WIDTH");

  auto contiguous_index = index.contiguous();
  uint64_t num_indices = index.numel();
  std::vector<int64_t> new_size = self.sizes().vec();
  new_size[dim] = num_indices;
  Tensor out = at::empty(new_size, self.options());

  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();
  CommandQueue& cq = device->command_queue();
  Program program = CreateProgram();

  auto input = allocator->get_buffer(self);
  auto output = allocator->get_buffer(out);
  auto indices = allocator->get_buffer(index);

  auto grid_size = device->compute_with_storage_grid_size();
  uint32_t num_cores_x = grid_size.x;
  uint32_t num_cores_y = grid_size.y;
  uint32_t num_cores_total = num_cores_x * num_cores_y;
  auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

  CBHandle cb_indices = MakeCircularBuffer(program, all_device_cores, CBIndex::c_0, constants::TILE_WIDTH, constants::TILE_WIDTH, DataFormat::UInt32);

  // Distribute the indices onto the cores
  uint64_t num_pages = num_indices / constants::TILE_WIDTH; // TODO: Use ceil here and adapt boundary
  auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
    split_work_to_cores(grid_size, num_pages);

  constexpr uint32_t datum_size_bytes = sizeof(uint32_t); // TODO: Fix this!

  // Create a buffer in SRAM which will be used as temporary storage to copy over data from input to output
  // For now we will just make it size FACE_WIDTH for simplicity but we might need to optimize that later
  tt_metal::InterleavedBufferConfig l1_config{
    .device = device,
    .size = datum_size_bytes * constants::TILE_WIDTH,
    .page_size = datum_size_bytes * constants::TILE_WIDTH,
    .buffer_type = tt_metal::BufferType::L1};
  auto l1_buffer = CreateBuffer(l1_config);

  std::vector<uint32_t> reader_compile_time_args = {(uint32_t)CBIndex::c_0};
  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)CBIndex::c_0};

  auto reader_id = tt_metal::CreateKernel(
    program,
    // TODO: The path is currently hard-coded, figure out how to fix it
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/dataflow/index_select_reader_row_major.cpp",
    all_device_cores,
    tt_metal::DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = reader_compile_time_args});

  auto writer_id = tt_metal::CreateKernel(
    program,
    // TODO: The path is currently hard-coded, figure out how to fix it
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/dataflow/index_select_writer_row_major.cpp",
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
  uint32_t num_pages = out.numel() / constants::TILE_WIDTH;
  uint32_t num_output_pages_per_block = out.size(dim) * out.stride(dim) / constants::TILE_WIDTH;
  std::vector<uint32_t> num_pages_per_block(num_tensors);

  for (int i = 0; i < num_tensors; ++i) {
    auto& tensor = inputs[i].get();
    num_pages_per_block[i] = tensor.size(dim) * tensor.stride(dim) / constants::TILE_WIDTH;
  }

  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();
  CommandQueue& cq = device->command_queue();
  Program program = CreateProgram();

  auto output = allocator->get_buffer(out);

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
    .size = datum_size_bytes * constants::TILE_WIDTH,
    .page_size = datum_size_bytes * constants::TILE_WIDTH,
    .buffer_type = tt_metal::BufferType::L1};
  auto l1_buffer = CreateBuffer(l1_config);

  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)num_tensors};

  auto writer_id = tt_metal::CreateKernel(
    program,
    // TODO: The path is currently hard-coded, figure out how to fix it
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/dataflow/writer_cat_row_major.cpp",
    all_device_cores,
    tt_metal::DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = writer_compile_time_args});

  std::vector<uint32_t> common_writer_args = {(uint32_t)dim, 0, 0, 0, 0, output->address(), l1_buffer->address()};
  for (int i = 0; i < num_tensors; ++i) {
    auto src = allocator->get_buffer(inputs[i].get());
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
  // Currently there is a bunch of assumptions on the parameters, none of them is hard to lift
  TORCH_CHECK(dim, "dim currently needs to be specified");
  TORCH_CHECK(dim->size() == 1, "dim currently needs to be of size 1, got ", dim->size());
  int64_t d = (*dim)[0];
  TORCH_CHECK(d == -1, "dim[0] currently must be -1, got dim[0] == ", d);

  // K is the inner dimension of the reduction
  uint32_t K = self.size(d);
  // num_blocks is the number of matrix multiplications of shape (TILE_HEIGHT x K) by (K x 1)
  // we need to do to compute all the means. The second operand is the constant matrix (1.0 / K).
  uint32_t num_blocks = self.numel() / (K * constants::TILE_HEIGHT);

  bfloat16 bfloat_scale_value = bfloat16(1.0f / K);
  uint32_t packed_scale_value = pack_two_bfloat16_into_uint32({bfloat_scale_value, bfloat_scale_value});

  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();

  auto a = allocator->get_buffer(self);
  auto b = allocator->get_buffer(out);

  ProgramBuilder builder(device);

  const uint32_t cb_num_tiles = 2;
  builder.AddCircularBuffer(CBIndex::c_0, DataFormat::Float16_b, cb_num_tiles);
  builder.AddCircularBuffer(CBIndex::c_2, DataFormat::Float16_b, cb_num_tiles);
  builder.AddCircularBuffer(CBIndex::c_3, DataFormat::Float16_b, cb_num_tiles);

  std::vector<uint32_t> reader_compile_time_args = {packed_scale_value};
  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)CBIndex::c_3};
  std::vector<uint32_t> compute_compile_time_args = {};

  builder.CreateKernels(
    num_blocks,
    // TODO: The paths are currently hard-coded, figure out how to fix it
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/dataflow/reduce_reader_row_major_to_tiles.cpp",
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/dataflow/reduce_writer_row_major.cpp",
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/compute/reduce.cpp",
    reader_compile_time_args,
    writer_compile_time_args,
    compute_compile_time_args,
    {},
    [a, b, K](const Program& program, const CoreCoord& core, KernelHandle reader, KernelHandle writer, KernelHandle compute, uint32_t num_blocks_per_core, uint32_t start_block_id) {
      SetRuntimeArgs(program, reader, core, {a->address(), K, num_blocks_per_core, start_block_id});
      SetRuntimeArgs(program, writer, core, {b->address(), num_blocks_per_core, start_block_id});
      SetRuntimeArgs(program, compute, core, {num_blocks_per_core, K / constants::TILE_WIDTH, 1});
    }
  );

  builder.Execute();

  return out;
}

at::Tensor & tril_tt_out(const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
  TT_NOT_IMPLEMENTED();
  return out;
}

static void where_kernel_tt(TensorIterator& iter) {
  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();
  ProgramBuilder builder(device);

  auto a_buf = allocator->get_buffer(iter.input(0));
  auto b_buf = allocator->get_buffer(iter.input(1));
  auto c_buf = allocator->get_buffer(iter.input(2));
  auto d_buf = allocator->get_buffer(iter.output(0));

  const uint32_t cb_num_tiles = 2;
  builder.AddCircularBuffer(CBIndex::c_0, DataFormat::UInt8, cb_num_tiles);
  builder.AddCircularBuffer(CBIndex::c_1, DataFormat::Float16_b, cb_num_tiles);
  builder.AddCircularBuffer(CBIndex::c_2, DataFormat::Float16_b, cb_num_tiles);
  builder.AddCircularBuffer(CBIndex::c_3, DataFormat::Float16_b, cb_num_tiles);
  builder.AddCircularBuffer(CBIndex::c_15, DataFormat::Float16_b, cb_num_tiles);
  builder.AddCircularBuffer(CBIndex::c_16, DataFormat::Float16_b, cb_num_tiles);

  std::vector<uint32_t> reader_compile_time_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1, (uint32_t)CBIndex::c_2};
  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)CBIndex::c_3};
  std::vector<uint32_t> compute_compile_time_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1, (uint32_t)CBIndex::c_2, (uint32_t)CBIndex::c_15, (uint32_t)CBIndex::c_16, (uint32_t)CBIndex::c_3};

  const uint32_t n_tiles = (iter.input(0).numel() + ::tt::constants::TILE_HW - 1) / ::tt::constants::TILE_HW;

  builder.CreateKernels(
    n_tiles,
    // TODO: The paths are currently hard-coded, figure out how to fix it
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/dataflow/ternary_eltwise_reader_row_major_to_tiles.cpp",
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/dataflow/eltwise_writer_row_major_to_tiles.cpp",
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/compute/eltwise_where_kernel.cpp",
    reader_compile_time_args,
    writer_compile_time_args,
    compute_compile_time_args,
    {},
    [a_buf, b_buf, c_buf, d_buf](const Program& program, const CoreCoord& core, KernelHandle reader, KernelHandle writer, KernelHandle compute, uint32_t num_tiles, uint32_t start_tile_id) {
      SetRuntimeArgs(program, reader, core, {a_buf->address(), b_buf->address(), c_buf->address(), num_tiles, start_tile_id});
      SetRuntimeArgs(program, writer, core, {d_buf->address(), num_tiles, start_tile_id});
      SetRuntimeArgs(program, compute, core, {num_tiles, start_tile_id});
    }
  );

  builder.Execute();
}

REGISTER_TT_DISPATCH(where_kernel, &where_kernel_tt)

void MemcpyWithOffsets(uint32_t dst_addr, uint32_t dst_offset, uint32_t src_addr, uint32_t src_offset, uint32_t num_tiles) {
  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();
  ProgramBuilder builder(device);

  const uint32_t cb_num_tiles = 2;
  builder.AddCircularBuffer(CBIndex::c_0, DataFormat::Float16_b, cb_num_tiles);

  std::vector<uint32_t> reader_compile_time_args = {(uint32_t)CBIndex::c_0};
  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)CBIndex::c_0};

  builder.CreateKernels(
    num_tiles,
    // TODO: The paths are currently hard-coded, figure out how to fix it
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/dataflow/memcpy_reader.cpp",
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/dataflow/memcpy_writer.cpp",
    "",
    reader_compile_time_args,
    writer_compile_time_args,
    {},
    {},
    [src_addr, src_offset, dst_addr, dst_offset](const Program& program, const CoreCoord& core, KernelHandle reader, KernelHandle writer, KernelHandle compute, uint32_t num_tiles, uint32_t start_tile_id) {
      SetRuntimeArgs(program, reader, core, {src_addr, src_offset, num_tiles, start_tile_id});
      SetRuntimeArgs(program, writer, core, {dst_addr, dst_offset, num_tiles, start_tile_id});
    }
  );

  builder.Execute();
}

Scalar _local_scalar_dense_tt(const Tensor& self) {
  Scalar r;
  TORCH_CHECK(self.numel() > 0, "_local_scalar_dense: Empty tensor not supported");
  // TODO: We should make this copy async so no sync between device and CPU is needed
  auto cpu_self = self.cpu();
  AT_DISPATCH_V2(
    self.scalar_type(), "_local_scalar_dense_tt", AT_WRAP([&] {
        r = Scalar(*cpu_self.const_data_ptr<scalar_t>());
     }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kComplexHalf, kHalf, kBool, kBFloat16, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  return r;
}

at::Tensor & softmax_tt_out(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out) {
  TT_NOT_IMPLEMENTED();
  return out;
}

at::Tensor & isneginf_out_tt(const at::Tensor & self, at::Tensor & out) {
  TT_NOT_IMPLEMENTED();
  return out;
}

at::Tensor & all_out_tt(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out) {
  TORCH_CHECK(dim == -1, "dim currently must be -1, got ", dim);

  // K is the inner dimension of the reduction
  uint32_t K = self.size(dim);
  // num_tiles is the number of output tiles that need to be computed
  // TODO: This is most likely not correct yet
  uint32_t num_tiles = self.numel() / (K * ::tt::constants::TILE_HEIGHT);

  auto* allocator = at::tt::GetTTAllocator();
  auto* device = allocator->device();

  auto a = allocator->get_buffer(self);
  auto b = allocator->get_buffer(out);

  ProgramBuilder builder(device);

  const uint32_t cb_num_tiles = 2;
  builder.AddCircularBuffer(CBIndex::c_0, DataFormat::UInt8, cb_num_tiles);
  builder.AddCircularBuffer(CBIndex::c_1, DataFormat::UInt8, cb_num_tiles);

  std::vector<uint32_t> reader_compile_time_args = {(uint32_t)CBIndex::c_0};
  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)CBIndex::c_1};
  std::vector<uint32_t> compute_compile_time_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1};

  builder.CreateKernels(
    num_tiles,
    // TODO: The paths are currently hard-coded, figure out how to fix it
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/dataflow/logical_reduce_reader_row_major_to_tiles.cpp",
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/dataflow/logical_reduce_writer_row_major.cpp",
    "/home/pcmoritz/pytorch/aten/src/ATen/native/tt/kernels/compute/logical_reduce.cpp",
    reader_compile_time_args,
    writer_compile_time_args,
    compute_compile_time_args,
    {},
    [a, b, K](const Program& program, const CoreCoord& core, KernelHandle reader, KernelHandle writer, KernelHandle compute, uint32_t num_tiles_per_core, uint32_t start_tile_id) {
      SetRuntimeArgs(program, reader, core, {a->address(), K, num_tiles_per_core, start_tile_id});
      SetRuntimeArgs(program, writer, core, {b->address(), num_tiles_per_core, start_tile_id});
      SetRuntimeArgs(program, compute, core, {K / constants::TILE_WIDTH, num_tiles_per_core});
    });

  builder.Execute();

  return out;
}

at::Tensor & silu_out_tt(const at::Tensor & self, at::Tensor & out) {
  TT_NOT_IMPLEMENTED();
  return out;
}

// static void sum_kernel_tt(TensorIterator& iter) {
// }

// REGISTER_TT_DISPATCH(sum_stub, &sum_kernel_tt)

}
