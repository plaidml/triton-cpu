//===- MlirKernelWrappers.cpp - Static kernel wrappers for MLIR -------=---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements basic functions to manipulate structured MLIR types at
// runtime. Entities in this file are meant to be retargetable, including on
// targets without a C++ runtime, and must be kept C compatible.
//
//===----------------------------------------------------------------------===//

#include "MlirKernelWrappers.h"
#include "XsmmKernels.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Helper to compute target pointer address.
static void *get_base_ptr(const libxsmm_datatype dType, void *alignedPtr,
                          int64_t offset) {
  if (dType == LIBXSMM_DATATYPE_F32) {
    float *base_ptr = (float *)alignedPtr + offset;
    return (void *)base_ptr;
  } else if (dType == LIBXSMM_DATATYPE_BF16) {
    uint16_t *base_ptr = (uint16_t *)alignedPtr + offset;
    return (void *)base_ptr;
  } else if (dType == LIBXSMM_DATATYPE_BF8) {
    uint8_t *base_ptr = (uint8_t *)alignedPtr + offset;
    return (void *)base_ptr;
  }

  return nullptr;
}

// Helper to populate libxsmm GEMM parameters.
static libxsmm_gemm_param
getXsmmGemmParam(const libxsmm_datatype dType, const libxsmm_datatype out_dtype,
                 void *alignedPtrA, int64_t offsetA, void *alignedPtrB,
                 int64_t offsetB, void *alignedPtrC, int64_t offsetC,
                 int64_t *lda, int64_t *ldb, int64_t *ldc, int64_t *numBatches,
                 int64_t *l_stride_a, int64_t *l_stride_b) {
  libxsmm_gemm_param gemm_param;

  // LIBXSMM col-major change A with B.
  gemm_param.a.primary = get_base_ptr(dType, alignedPtrB, offsetB);
  gemm_param.b.primary = get_base_ptr(dType, alignedPtrA, offsetA);
  gemm_param.c.primary = get_base_ptr(out_dtype, alignedPtrC, offsetC);

  // Pass LDs at runtime.
  // Switch A with B for col-major.
  gemm_param.a.quinary = (void *)ldb;
  gemm_param.b.quinary = (void *)lda;
  gemm_param.c.quinary = (void *)ldc;

  if (numBatches) {
    gemm_param.op.tertiary = (void *)numBatches;
  }

  if (l_stride_a && l_stride_a) {
    // Switch A with B for col-major.
    gemm_param.a.secondary = (void *)l_stride_b;
    gemm_param.b.secondary = (void *)l_stride_a;
  }

  return gemm_param;
}

void xsmm_gemm_f32_m32_n32_k32(const libxsmm_datatype dType,
                               const libxsmm_datatype out_dtype,
                               void *alignedPtrA, int64_t offsetA,
                               void *alignedPtrB, int64_t offsetB,
                               void *alignedPtrC, int64_t offsetC, int64_t lda,
                               int64_t ldb, int64_t ldc) {
  libxsmm_gemm_param gemm_param = getXsmmGemmParam(
      dType, out_dtype, alignedPtrA, offsetA, alignedPtrB, offsetB, alignedPtrC,
      offsetC, &lda, &ldb, &ldc, NULL, NULL, NULL);
  libxsmm_gemm_f32_m32_n32_k32(&gemm_param);
}

void xsmm_gemm_f32_m64_n64_k64(const libxsmm_datatype dType,
                               const libxsmm_datatype out_dtype,
                               void *alignedPtrA, int64_t offsetA,
                               void *alignedPtrB, int64_t offsetB,
                               void *alignedPtrC, int64_t offsetC, int64_t lda,
                               int64_t ldb, int64_t ldc) {
  libxsmm_gemm_param gemm_param = getXsmmGemmParam(
      dType, out_dtype, alignedPtrA, offsetA, alignedPtrB, offsetB, alignedPtrC,
      offsetC, &lda, &ldb, &ldc, NULL, NULL, NULL);
  libxsmm_gemm_f32_m64_n64_k64(&gemm_param);
}

void xsmm_gemm_f32_m64_n64_k32(const libxsmm_datatype dType,
                               const libxsmm_datatype out_dtype,
                               void *alignedPtrA, int64_t offsetA,
                               void *alignedPtrB, int64_t offsetB,
                               void *alignedPtrC, int64_t offsetC, int64_t lda,
                               int64_t ldb, int64_t ldc) {
  libxsmm_gemm_param gemm_param = getXsmmGemmParam(
      dType, out_dtype, alignedPtrA, offsetA, alignedPtrB, offsetB, alignedPtrC,
      offsetC, &lda, &ldb, &ldc, NULL, NULL, NULL);
  libxsmm_gemm_f32_m64_n64_k32(&gemm_param);
}

void xsmm_gemm_f32_m64_n64_k512(const libxsmm_datatype dType,
                                const libxsmm_datatype out_dtype,
                                void *alignedPtrA, int64_t offsetA,
                                void *alignedPtrB, int64_t offsetB,
                                void *alignedPtrC, int64_t offsetC, int64_t lda,
                                int64_t ldb, int64_t ldc) {
  libxsmm_gemm_param gemm_param = getXsmmGemmParam(
      dType, out_dtype, alignedPtrA, offsetA, alignedPtrB, offsetB, alignedPtrC,
      offsetC, &lda, &ldb, &ldc, NULL, NULL, NULL);
  libxsmm_gemm_f32_m64_n64_k512(&gemm_param);
}

void xsmm_gemm_bf16_m32_n32_k32(const libxsmm_datatype dType,
                                const libxsmm_datatype out_dtype,
                                void *alignedPtrA, int64_t offsetA,
                                void *alignedPtrB, int64_t offsetB,
                                void *alignedPtrC, int64_t offsetC, int64_t lda,
                                int64_t ldb, int64_t ldc) {
  libxsmm_gemm_param gemm_param = getXsmmGemmParam(
      dType, out_dtype, alignedPtrA, offsetA, alignedPtrB, offsetB, alignedPtrC,
      offsetC, &lda, &ldb, &ldc, NULL, NULL, NULL);
  libxsmm_gemm_bf16_m32_n32_k32(&gemm_param);
}

void xsmm_gemm_bf16_m64_n64_k64(const libxsmm_datatype dType,
                                const libxsmm_datatype out_dtype,
                                void *alignedPtrA, int64_t offsetA,
                                void *alignedPtrB, int64_t offsetB,
                                void *alignedPtrC, int64_t offsetC, int64_t lda,
                                int64_t ldb, int64_t ldc) {
  libxsmm_gemm_param gemm_param = getXsmmGemmParam(
      dType, out_dtype, alignedPtrA, offsetA, alignedPtrB, offsetB, alignedPtrC,
      offsetC, &lda, &ldb, &ldc, NULL, NULL, NULL);
  libxsmm_gemm_bf16_m64_n64_k64(&gemm_param);
}

void xsmm_gemm_bf16_m64_n64_k32(const libxsmm_datatype dType,
                                const libxsmm_datatype out_dtype,
                                void *alignedPtrA, int64_t offsetA,
                                void *alignedPtrB, int64_t offsetB,
                                void *alignedPtrC, int64_t offsetC, int64_t lda,
                                int64_t ldb, int64_t ldc) {
  libxsmm_gemm_param gemm_param = getXsmmGemmParam(
      dType, out_dtype, alignedPtrA, offsetA, alignedPtrB, offsetB, alignedPtrC,
      offsetC, &lda, &ldb, &ldc, NULL, NULL, NULL);
  libxsmm_gemm_bf16_m64_n64_k32(&gemm_param);
}

void xsmm_gemm_bf16_m64_n64_k512(const libxsmm_datatype dType,
                                 const libxsmm_datatype out_dtype,
                                 void *alignedPtrA, int64_t offsetA,
                                 void *alignedPtrB, int64_t offsetB,
                                 void *alignedPtrC, int64_t offsetC,
                                 int64_t lda, int64_t ldb, int64_t ldc) {
  libxsmm_gemm_param gemm_param = getXsmmGemmParam(
      dType, out_dtype, alignedPtrA, offsetA, alignedPtrB, offsetB, alignedPtrC,
      offsetC, &lda, &ldb, &ldc, NULL, NULL, NULL);
  libxsmm_gemm_bf16_m64_n64_k512(&gemm_param);
}

void xsmm_brgemm_f32_m64_n64_k32(
    const libxsmm_datatype dType, const libxsmm_datatype out_dtype,
    void *alignedPtrA, int64_t offsetA, void *alignedPtrB, int64_t offsetB,
    void *alignedPtrC, int64_t offsetC, int64_t numBatches, int64_t lda,
    int64_t ldb, int64_t ldc, int64_t stride_a, int64_t stride_b) {
  size_t typeSize;
  if (dType == LIBXSMM_DATATYPE_F32)
    typeSize = sizeof(float);
  else if (dType == LIBXSMM_DATATYPE_BF16)
    typeSize = sizeof(uint16_t);
  else if (dType == LIBXSMM_DATATYPE_BF8)
    typeSize = sizeof(uint8_t);
  else
    assert(false && "unsupported datatype");

  // Switch A with B for col-major.
  int64_t l_stride_a = stride_a * typeSize;
  int64_t l_stride_b = stride_b * typeSize;

  libxsmm_gemm_param gemm_param = getXsmmGemmParam(
      dType, out_dtype, alignedPtrA, offsetA, alignedPtrB, offsetB, alignedPtrC,
      offsetC, &lda, &ldb, &ldc, &numBatches, &l_stride_a, &l_stride_b);

  libxsmm_brgemm_f32_m64_n64_k32(&gemm_param);
}

void xsmm_brgemm_bf16_m64_n64_k32(
    const libxsmm_datatype dType, const libxsmm_datatype out_dtype,
    void *alignedPtrA, int64_t offsetA, void *alignedPtrB, int64_t offsetB,
    void *alignedPtrC, int64_t offsetC, int64_t numBatches, int64_t lda,
    int64_t ldb, int64_t ldc, int64_t stride_a, int64_t stride_b) {
  size_t typeSize;
  if (dType == LIBXSMM_DATATYPE_F32)
    typeSize = sizeof(float);
  else if (dType == LIBXSMM_DATATYPE_BF16)
    typeSize = sizeof(uint16_t);
  else if (dType == LIBXSMM_DATATYPE_BF8)
    typeSize = sizeof(uint8_t);
  else
    assert(false && "unsupported datatype");

  // Switch A with B for col-major.
  int64_t l_stride_a = stride_a * typeSize;
  int64_t l_stride_b = stride_b * typeSize;

  libxsmm_gemm_param gemm_param = getXsmmGemmParam(
      dType, out_dtype, alignedPtrA, offsetA, alignedPtrB, offsetB, alignedPtrC,
      offsetC, &lda, &ldb, &ldc, &numBatches, &l_stride_a, &l_stride_b);

  libxsmm_brgemm_bf16_m64_n64_k32(&gemm_param);
}

#ifdef __cplusplus
} /* extern "C" */
#endif
