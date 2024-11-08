#include <torch/extension.h>

#include "libxsmm.h"
#include <omp.h>

bool fastCopy2D(torch::Tensor &input, torch::Tensor &output) {
  auto inSizes = input.sizes();
  auto outSizes = output.sizes();
  auto byteSize = input.element_size();
  if (!input.is_floating_point() || inSizes.size() != 2 ||
      outSizes.size() != 2 || outSizes[0] < inSizes[0] ||
      outSizes[1] < inSizes[1] || byteSize != output.element_size())
    return false;

  libxsmm_datatype dtype =
      byteSize == 4 ? LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_BF16;
  libxsmm_meltw_unary_shape shape;
  // Fliped to libxsmm's column-major convention.
  shape.m = inSizes[1];
  shape.n = 1;
  shape.ldi = inSizes[1];
  shape.ldo = outSizes[1];
  shape.in0_type = dtype;
  shape.out_type = dtype;
  shape.comp_type = dtype;
  libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltwfunction_unary identityFn = libxsmm_dispatch_meltw_unary(
      LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, shape, flags);

  void *baseIn = input.data_ptr();
  void *outIn = output.data_ptr();

#pragma omp parallel for
  for (int i = 0; i < inSizes[0]; ++i) {
    libxsmm_meltw_unary_param param;
    param.in.primary = baseIn + i * inSizes[1] * byteSize;
    param.out.primary = outIn + i * outSizes[1] * byteSize;
    identityFn(&param);
  }

  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fastCopy2D", &fastCopy2D, "Fast 2D tensor copy");
}
