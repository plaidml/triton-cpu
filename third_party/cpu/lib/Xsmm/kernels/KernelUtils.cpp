//===- KernelUtils.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KernelUtils.h"

namespace xsmm {
namespace kernel {

bool isConfigSupported(ComputeType comp, DataType data, int64_t m, int64_t n,
                       int64_t k) {
  // Currently all size combinations support all data types - no `data` checks.
  if (comp == ComputeType::GEMM) {
    if (m == 32 && n == 32 && k == 32)
      return true;
    if (m == 64 && n == 64)
      return k == 32 || k == 64 || k == 512;
  }

  if (comp == ComputeType::BRGEMM) {
    if (m == 64 && n == 64 && k == 32)
      return true;
  }

  return false;
}

} // namespace kernel
} // namespace xsmm
