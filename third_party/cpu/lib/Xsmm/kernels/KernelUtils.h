//===- KernelUtils.h - ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef XSMM_KERNELS_KERNELUTILS_H
#define XSMM_KERNELS_KERNELUTILS_H

#include <stdint.h>

namespace xsmm {
namespace kernel {

enum class ComputeType { GEMM, BRGEMM };

enum class DataType { F32, BF16 };

// Returns true if there is a static kernel available for the given
// configuration.
bool isConfigSupported(ComputeType comp, DataType data, int64_t m, int64_t n,
                       int64_t k);

} // namespace kernel
} // namespace xsmm

#endif // XSMM_KERNELS_KERNELUTILS_H
