#include <ATen/ops/add_native.h>

#include <tt-metalium/host_api.hpp>

using namespace tt::tt_metal;

namespace at::native {

at::Tensor & add_out_tt(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  Program program = CreateProgram();
  constexpr CoreCoord core = {0, 0};
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
  
  return out;
}

// static void sum_kernel_tt(TensorIterator& iter) {
// }

// REGISTER_TT_DISPATCH(sum_stub, &sum_kernel_tt)

}
