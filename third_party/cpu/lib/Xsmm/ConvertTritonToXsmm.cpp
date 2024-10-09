//===- ConvertTritonToXsmm.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cpu/include/Xsmm/Passes.h"

#include "ValueUtils.h"
#include "VnniUtils.h"
#include "XsmmUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::func;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTTRITONTOXSMM
#include "cpu/include/Xsmm/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

namespace {

static Value getMemrefSource(PatternRewriter &rewriter, Operation *op,
                             TypedValue<RankedTensorType> operand) {
  Location loc = op->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  RankedTensorType tensorTy = operand.getType();
  MemRefType memTy =
      MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
  auto alloca = rewriter.create<memref::AllocaOp>(loc, memTy);
  rewriter.create<triton::cpu::StoreOp>(loc, operand, alloca);

  return alloca;
}

struct DotToXsmm : public OpRewritePattern<triton::DotOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    Location loc = dotOp.getLoc();
    MLIRContext *ctx = dotOp.getContext();

    // Dot op computes standard (batch) GEMM.
    SmallVector<AffineMap> indexingMaps;
    TypedValue<RankedTensorType> res = dotOp.getD();
    uint32_t rank = res.getType().getRank();
    if (rank == 2) {
      indexingMaps.push_back(
          AffineMap::getMultiDimMapWithTargets(3, {0, 2}, ctx));
      indexingMaps.push_back(
          AffineMap::getMultiDimMapWithTargets(3, {2, 1}, ctx));
      indexingMaps.push_back(
          AffineMap::getMultiDimMapWithTargets(3, {0, 1}, ctx));
    } else if (rank == 3) {
      indexingMaps.push_back(
          AffineMap::getMultiDimMapWithTargets(4, {0, 1, 3}, ctx));
      indexingMaps.push_back(
          AffineMap::getMultiDimMapWithTargets(4, {0, 3, 2}, ctx));
      indexingMaps.push_back(
          AffineMap::getMultiDimMapWithTargets(4, {0, 1, 2}, ctx));
    }
    if (indexingMaps.size() == 0)
      return rewriter.notifyMatchFailure(dotOp, "unsupported indexing maps");

    TypedValue<RankedTensorType> lhs = dotOp.getA();
    TypedValue<RankedTensorType> rhs = dotOp.getB();
    TypedValue<RankedTensorType> acc = dotOp.getC();

    SmallVector<Attribute> flags;
    Value lhsBuf = getMemrefSource(rewriter, dotOp, lhs);
    Value rhsBuf = getMemrefSource(rewriter, dotOp, rhs);
    Value accBuf = getMemrefSource(rewriter, dotOp, acc);
    SmallVector<Value> inputs{lhsBuf, rhsBuf, accBuf};
    SmallVector<Value> outputs{nullptr};

    auto brgemmInfo = xsmm::utils::isMappableToBrgemm(rewriter, dotOp, inputs,
                                                      outputs, indexingMaps);
    if (failed(brgemmInfo))
      return rewriter.notifyMatchFailure(dotOp, "not mappable to XSMM");
    if (brgemmInfo->isVnni)
      return rewriter.notifyMatchFailure(dotOp, "VNNI support NYI");

    auto xsmmFuncs = xsmm::utils::buildBrgemmCalls(
        rewriter, dotOp, ValueRange{lhsBuf, rhsBuf, accBuf}, *brgemmInfo,
        flags);
    auto loadOp =
        rewriter.create<triton::cpu::LoadOp>(loc, res.getType(), accBuf);

    rewriter.replaceOp(dotOp, loadOp);

    return success();
  }
};

struct ConvertTritonToXsmm
    : public triton::cpu::impl::ConvertTritonToXsmmBase<ConvertTritonToXsmm> {
  using ConvertTritonToXsmmBase::ConvertTritonToXsmmBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<DotToXsmm>(context);
    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
