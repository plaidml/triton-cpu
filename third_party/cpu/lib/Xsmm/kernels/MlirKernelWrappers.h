//===- MlirKernelWrappers.h - Static kernel wrappers for MLIR -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares basic classes and functions to manipulate structured MLIR
// types at runtime. Entities in this file must be compliant with C++11 and be
// retargetable, including on targets without a C++ runtime.
//
//===----------------------------------------------------------------------===//

#ifndef XSMM_KERNELS_MLIRKERNELWRAPPERS_H
#define XSMM_KERNELS_MLIRKERNELWRAPPERS_H

#include "mlir/ExecutionEngine/RunnerUtils.h"

#ifdef __cplusplus
extern "C" {
#endif

// Generated XSMM static library header is not compatible with libxsmm.h include
// as the generated header duplicates some of the definitions.
// Redefine locally libxsmm datatype needed for MLIR wrappers.
typedef enum libxsmm_datatype {
  LIBXSMM_DATATYPE_F64,
  LIBXSMM_DATATYPE_F32,
  LIBXSMM_DATATYPE_BF16,
  LIBXSMM_DATATYPE_F16,
  LIBXSMM_DATATYPE_BF8,
  LIBXSMM_DATATYPE_HF8,
  LIBXSMM_DATATYPE_I64,
  LIBXSMM_DATATYPE_U64,
  LIBXSMM_DATATYPE_I32,
  LIBXSMM_DATATYPE_U32,
  LIBXSMM_DATATYPE_I16,
  LIBXSMM_DATATYPE_U16,
  LIBXSMM_DATATYPE_I8,
  LIBXSMM_DATATYPE_U8,
  LIBXSMM_DATATYPE_IMPLICIT,
  LIBXSMM_DATATYPE_UNSUPPORTED
} libxsmm_datatype;

MLIR_RUNNERUTILS_EXPORT void xsmm_gemm_f32_m32_n32_k32(const libxsmm_datatype dType,
  const libxsmm_datatype out_dtype,
  void *alignedPtrA, int64_t offsetA,
  void *alignedPtrB, int64_t offsetB,
  void *alignedPtrC, int64_t offsetC, int64_t lda,
  int64_t ldb, int64_t ldc);

MLIR_RUNNERUTILS_EXPORT void xsmm_gemm_f32_m64_n64_k64(const libxsmm_datatype dType,
  const libxsmm_datatype out_dtype,
  void *alignedPtrA, int64_t offsetA,
  void *alignedPtrB, int64_t offsetB,
  void *alignedPtrC, int64_t offsetC, int64_t lda,
  int64_t ldb, int64_t ldc);

MLIR_RUNNERUTILS_EXPORT void xsmm_gemm_f32_m64_n64_k32(const libxsmm_datatype dType,
  const libxsmm_datatype out_dtype,
  void *alignedPtrA, int64_t offsetA,
  void *alignedPtrB, int64_t offsetB,
  void *alignedPtrC, int64_t offsetC, int64_t lda,
  int64_t ldb, int64_t ldc);

MLIR_RUNNERUTILS_EXPORT void xsmm_gemm_f32_m64_n64_k512(const libxsmm_datatype dType,
   const libxsmm_datatype out_dtype,
   void *alignedPtrA, int64_t offsetA,
   void *alignedPtrB, int64_t offsetB,
   void *alignedPtrC, int64_t offsetC, int64_t lda,
   int64_t ldb, int64_t ldc);

MLIR_RUNNERUTILS_EXPORT void xsmm_gemm_bf16_m32_n32_k32(const libxsmm_datatype dType,
   const libxsmm_datatype out_dtype,
   void *alignedPtrA, int64_t offsetA,
   void *alignedPtrB, int64_t offsetB,
   void *alignedPtrC, int64_t offsetC, int64_t lda,
   int64_t ldb, int64_t ldc);

MLIR_RUNNERUTILS_EXPORT void xsmm_gemm_bf16_m64_n64_k64(const libxsmm_datatype dType,
   const libxsmm_datatype out_dtype,
   void *alignedPtrA, int64_t offsetA,
   void *alignedPtrB, int64_t offsetB,
   void *alignedPtrC, int64_t offsetC, int64_t lda,
   int64_t ldb, int64_t ldc);

MLIR_RUNNERUTILS_EXPORT void xsmm_gemm_bf16_m64_n64_k32(const libxsmm_datatype dType,
   const libxsmm_datatype out_dtype,
   void *alignedPtrA, int64_t offsetA,
   void *alignedPtrB, int64_t offsetB,
   void *alignedPtrC, int64_t offsetC, int64_t lda,
   int64_t ldb, int64_t ldc);

MLIR_RUNNERUTILS_EXPORT void xsmm_gemm_bf16_m64_n64_k512(const libxsmm_datatype dType,
    const libxsmm_datatype out_dtype,
    void *alignedPtrA, int64_t offsetA,
    void *alignedPtrB, int64_t offsetB,
    void *alignedPtrC, int64_t offsetC,
    int64_t lda, int64_t ldb, int64_t ldc);

MLIR_RUNNERUTILS_EXPORT void xsmm_brgemm_f32_m64_n64_k32(
const libxsmm_datatype dType, const libxsmm_datatype out_dtype,
void *alignedPtrA, int64_t offsetA, void *alignedPtrB, int64_t offsetB,
void *alignedPtrC, int64_t offsetC, int64_t numBatches, int64_t lda,
int64_t ldb, int64_t ldc, int64_t stride_a, int64_t stride_b);

MLIR_RUNNERUTILS_EXPORT void xsmm_brgemm_bf16_m64_n64_k32(
const libxsmm_datatype dType, const libxsmm_datatype out_dtype,
void *alignedPtrA, int64_t offsetA, void *alignedPtrB, int64_t offsetB,
void *alignedPtrC, int64_t offsetC, int64_t numBatches, int64_t lda,
int64_t ldb, int64_t ldc, int64_t stride_a, int64_t stride_b);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // XSMM_KERNELS_MLIRKERNELWRAPPERS_H
