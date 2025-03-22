#include <ATen/ops/add_native.h>
#include <ATen/ops/relu_native.h>
#include <ATen/tt/TTDevice.h>

#include <tt-metalium/host_api.hpp>

using namespace tt::tt_metal;

namespace at::native {

// TODO(pcm): This is currently just putting the example
// https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/eltwise_binary/eltwise_binary.cpp
// in. Going forward we should figure out how to adapt / run the existing tt_metal kernels.

struct BinaryOpType {
    enum Enum { ADD = 0, SUB = 1, MUL = 2 };
    static const auto all() { return magic_enum::enum_values<Enum>(); }
};

static std::map<std::string, std::string> get_defines(BinaryOpType::Enum op_type) {
    std::map<std::string, std::string> defines;
    // TODO(AP): remove duplication
    std::string op_name, op_binary_type;
    switch (op_type) {
        case BinaryOpType::ADD:
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            break;
        case BinaryOpType::SUB:
            op_name = "sub_tiles";
            op_binary_type = "EltwiseBinaryType::ELWSUB";
            break;
        case BinaryOpType::MUL:
            op_name = "mul_tiles";
            op_binary_type = "EltwiseBinaryType::ELWMUL";
            break;
        default: TT_ASSERT(false && "Undefined op type");
    }
    defines["ELTWISE_OP"] = op_name.c_str();
    defines["ELTWISE_OP_TYPE"] = op_binary_type.c_str();
    return defines;
}

at::Tensor & add_out_tt(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  CommandQueue& cq = tt::GetTTAllocator()->device()->command_queue();
  Program program = CreateProgram();
  constexpr CoreCoord core = {0, 0};

  constexpr uint32_t single_tile_size = 2 * 1024;
  constexpr uint32_t num_tiles = 64;

  constexpr uint32_t src0_cb_index = ::tt::CBIndex::c_0;
  constexpr uint32_t num_input_tiles = 2;
  CircularBufferConfig cb_src0_config =
    CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, ::tt::DataFormat::Float32}})
        .set_page_size(src0_cb_index, single_tile_size);
  CBHandle cb_src0 = ::tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

  constexpr uint32_t src1_cb_index = ::tt::CBIndex::c_1;
  CircularBufferConfig cb_src1_config =
  CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, ::tt::DataFormat::Float32}})
    .set_page_size(src1_cb_index, single_tile_size);
  CBHandle cb_src1 = ::tt::tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

  constexpr uint32_t output_cb_index = ::tt::CBIndex::c_16;
  constexpr uint32_t num_output_tiles = 2;
  CircularBufferConfig cb_output_config =
    CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, ::tt::DataFormat::Float32}})
        .set_page_size(output_cb_index, single_tile_size);
  CBHandle cb_output = ::tt::tt_metal::CreateCircularBuffer(program, core, cb_output_config);

  uint32_t src0_bank_id = 0;
  uint32_t src1_bank_id = 0;
  uint32_t dst_bank_id = 0;

  auto* allocator = at::tt::GetTTAllocator();
  auto src0_dram_buffer = allocator->get_buffer(self.data_ptr());
  auto src1_dram_buffer = allocator->get_buffer(other.data_ptr());
  auto dst_dram_buffer = allocator->get_buffer(out.data_ptr());
  
  KernelHandle binary_reader_kernel_id = CreateKernel(
    program,
    "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
    core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

  KernelHandle unary_writer_kernel_id = CreateKernel(
    program,
    "tt_metal/kernels/dataflow/writer_unary.cpp",
    core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

  std::vector<uint32_t> compute_kernel_args = {};
  constexpr bool fp32_dest_acc_en = false;
  constexpr bool math_approx_mode = false;

  KernelHandle eltwise_binary_kernel_id = CreateKernel(
    program,
    "tt_metal/kernels/compute/eltwise_binary.cpp",
    core,
    ComputeConfig{
       .math_fidelity = MathFidelity::HiFi4,
       .fp32_dest_acc_en = fp32_dest_acc_en,
       .math_approx_mode = math_approx_mode,
       .compile_args = compute_kernel_args,
       .defines = get_defines(BinaryOpType::ADD)});

   SetRuntimeArgs(
     program,
     binary_reader_kernel_id,
     core,
     {src0_dram_buffer->address(),
      src0_bank_id,
      num_tiles,
      src1_dram_buffer->address(),
      src1_bank_id,
      num_tiles,
      0});

  SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {num_tiles, 1});
  SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_dram_buffer->address(), dst_bank_id, num_tiles});
  EnqueueProgram(cq, program, false);
  Finish(cq);
  
  return out;
}

// RELU

static std::shared_ptr<Buffer> MakeBuffer(IDevice* device, uint32_t size, uint32_t page_size, BufferType buffer_type) {
    InterleavedBufferConfig config{
        .device = device,
        .size = size,
        .page_size = page_size,
        .buffer_type = buffer_type
    };
    return CreateBuffer(config);
}

Tensor relu_tt(const Tensor& self) {
  CommandQueue& cq = tt::GetTTAllocator()->device()->command_queue();
  Program program = CreateProgram();
  return self;
}

// static void sum_kernel_tt(TensorIterator& iter) {
// }

// REGISTER_TT_DISPATCH(sum_stub, &sum_kernel_tt)

}
