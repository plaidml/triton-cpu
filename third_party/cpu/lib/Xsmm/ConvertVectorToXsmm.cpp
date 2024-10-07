//===- ConvertVectorToXsmm.cpp ----------------------------------*- C++ -*-===//
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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/TypeSwitch.h"
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
#define GEN_PASS_DEF_CONVERTVECTORTOXSMM
#include "cpu/include/Xsmm/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

namespace {

static std::pair<Operation *, Operation *>
buildBrgemm(PatternRewriter &rewriter, Operation *contractOp, ValueRange inputs,
            xsmm::BrgemmInfo brgemmInfo, SmallVector<Attribute> flags) {
  assert(inputs.size() == 3 && "Expects three inputs for BRGEMM call");
  auto m = brgemmInfo.m;
  auto n = brgemmInfo.n;
  auto k = brgemmInfo.k;
  auto batch = brgemmInfo.batch;
  int64_t lda = brgemmInfo.lda;
  int64_t ldb = brgemmInfo.ldb;
  int64_t ldc = brgemmInfo.ldc;
  int64_t strideA = brgemmInfo.strideA;
  int64_t strideB = brgemmInfo.strideB;
  auto loc = contractOp->getLoc();
  auto dtype = xsmm::utils::getDataType(rewriter, inputs[0].getType());
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  SmallVector<Value, 10> dispatchOperands;
  SmallVector<Type, 10> dispatchOperandTypes;
  // Dispatch the data type.
  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, cast<TypedAttr>(dtype)));
  dispatchOperandTypes.push_back(integer64);

  ArrayAttr brgemmFlags = rewriter.getArrayAttr(flags);
  SmallVector<Value, 10> invokeOperands;
  std::string dispatchName = "xsmm_gemm_dispatch";
  std::string invokeName = "xsmm_gemm_invoke";

  if (batch != 0) {
    dispatchName = "xsmm_brgemm_dispatch";
    invokeName = "xsmm_brgemm_invoke";
  }

  auto dims = SmallVector<int64_t>{m, n, k, lda, ldb, ldc, strideA, strideB};
  for (size_t idx = 0; idx < dims.size(); idx++) {
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, dims[idx])));
    dispatchOperandTypes.push_back(integer64);
  }
  // Dispatch the flags. Pass to the library the already ored-flag to
  // avoid changing the interface every time we add a new flag. Flags
  // are assumed to be verified before (i.e., op verifier).
  int64_t oredFlag = xsmm::utils::getOredFlags(brgemmFlags);

  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), oredFlag)));
  dispatchOperandTypes.push_back(integer64);
  ModuleOp module = contractOp->getParentOfType<ModuleOp>();
  auto dispatched = xsmm::utils::buildDispatchCall(
      rewriter, loc, dispatchOperands, dispatchOperandTypes, module,
      SymbolRefAttr::get(contractOp->getContext(), dispatchName));
  SmallVector<Value, 6> operandRange;
  operandRange.push_back(dispatched.getResult(0));
  for (auto operand : inputs) {
    operandRange.push_back(operand);
  }
  Value batchDim = rewriter.create<arith::ConstantOp>(
      loc, integer64, rewriter.getIntegerAttr(integer64, batch));
  operandRange.push_back(batchDim);
  auto invokeCall = xsmm::utils::buildInvokeCall(
      rewriter, loc, module, operandRange, invokeName, dtype);
  return std::make_pair(&*dispatched, &*invokeCall);
}

static Value getMemrefSource(PatternRewriter &rewriter, Operation *op,
                             TypedValue<Type> operand) {
  Location loc = op->getLoc();
  OpBuilder::InsertionGuard g(rewriter);

  if (auto readOp =
          dyn_cast_or_null<vector::TransferReadOp>(operand.getDefiningOp())) {
    return readOp.getSource();
  }

  rewriter.setInsertionPoint(op);

  auto vecTy = dyn_cast<VectorType>(operand.getType());
  assert(vecTy && "Expect vector type operand");
  MemRefType memTy = MemRefType::get(vecTy.getShape(), vecTy.getElementType());
  auto alloca = rewriter.create<memref::AllocaOp>(loc, memTy);
  Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> indices(memTy.getRank(), zeroIdx);
  auto write =
      rewriter.create<vector::TransferWriteOp>(loc, operand, alloca, indices);

  return alloca;
}

struct ContractToXsmm : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    Location loc = contractOp.getLoc();

    TypedValue<VectorType> lhs = contractOp.getLhs();
    TypedValue<VectorType> rhs = contractOp.getRhs();
    TypedValue<Type> acc = contractOp.getAcc();

    auto vecTy = dyn_cast<VectorType>(acc.getType());
    if (!vecTy)
      return rewriter.notifyMatchFailure(contractOp,
                                         "expects to accumulate on vector");

    SmallVector<Attribute> flags;
    Value lhsBuf = getMemrefSource(rewriter, contractOp, lhs);
    Value rhsBuf = getMemrefSource(rewriter, contractOp, rhs);
    Value accBuf = getMemrefSource(rewriter, contractOp, acc);
    SmallVector<Value> inputs{lhsBuf, rhsBuf, accBuf};
    SmallVector<Value> outputs{nullptr};
    auto brgemmInfo =
        xsmm::utils::isMappableToBrgemm(rewriter, contractOp, inputs, outputs,
                                        contractOp.getIndexingMapsArray());
    if (failed(brgemmInfo))
      return rewriter.notifyMatchFailure(contractOp, "not mappable to XSMM");
    if (brgemmInfo->isVnni)
      return rewriter.notifyMatchFailure(contractOp, "VNNI support NYI");

    auto xsmmFuncs =
        buildBrgemm(rewriter, contractOp, ValueRange{lhsBuf, rhsBuf, accBuf},
                    *brgemmInfo, flags);

    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices(dyn_cast<MemRefType>(accBuf.getType()).getRank(),
                               zeroIdx);
    auto readOp =
        rewriter.create<vector::TransferReadOp>(loc, vecTy, accBuf, indices);

    rewriter.replaceOp(contractOp, readOp);

    return success();
  }
};

struct ConvertVectorToXsmm
    : public triton::cpu::impl::ConvertVectorToXsmmBase<ConvertVectorToXsmm> {
  using ConvertVectorToXsmmBase::ConvertVectorToXsmmBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ContractToXsmm>(context);
    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
