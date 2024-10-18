//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/AnalysisStructured/OpFoldResultUtils.h"
#include "triton/AnalysisStructured/PtrAnalysis.h"
#include "triton/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <cstdint>
#include <optional>

#define DEBUG_TYPE "structured-to-triton"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton/Conversion/TritonToStructured/Passes.h.inc"

namespace {

struct ConvertLoad : public OpRewritePattern<tts::LoadOp> {
  ConvertLoad(mlir::MLIRContext *context)
      : OpRewritePattern<tts::LoadOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tts::LoadOp op,
                                PatternRewriter &rewriter) const override {
    auto ptr = op.getPtr();
    auto boundaryCheck = SmallVector<int32_t>();
    std::optional<PaddingOption> padding = std::nullopt;
    auto cache = CacheModifier::NONE;
    auto evict = EvictionPolicy::NORMAL;
    auto isVolatile = false;

    LoadOp loadOp = rewriter.create<LoadOp>(op->getLoc(), ptr, boundaryCheck,
                                            padding, cache, evict, isVolatile);
    rewriter.replaceOp(op, loadOp);
    return success();
  }
};

struct ConvertStore : public OpRewritePattern<tts::StoreOp> {
  ConvertStore(mlir::MLIRContext *context)
      : OpRewritePattern<tts::StoreOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tts::StoreOp op,
                                PatternRewriter &rewriter) const override {
    auto ptr = op.getPtr();
    auto value = op.getValue();
    auto boundaryCheck = SmallVector<int32_t>();
    auto cache = CacheModifier::NONE;
    auto evict = EvictionPolicy::NORMAL;
    StoreOp storeOp = rewriter.create<StoreOp>(op->getLoc(), ptr, value,
                                               boundaryCheck, cache, evict);
    rewriter.replaceOp(op, storeOp);
    return success();
  }
};

struct ConvertMakeTensorPtr : public OpRewritePattern<tts::MakeTensorPtrOp> {
  ConvertMakeTensorPtr(mlir::MLIRContext *context)
      : OpRewritePattern<tts::MakeTensorPtrOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tts::MakeTensorPtrOp op,
                                PatternRewriter &rewriter) const override {
    auto base = op.getBase();
    SmallVector<Value> shapeAtI64;
    for (auto dimSize : op.getMixedSizes()) {
      Value dimSizeOp =
          getValueOrCreateConstantIntOp(rewriter, op->getLoc(), dimSize);
      shapeAtI64.emplace_back(getValueOrCreateCastToIndexLike(
          rewriter, op->getLoc(), rewriter.getIntegerType(64), dimSizeOp));
    }
    SmallVector<Value> stridesAtI64;
    for (auto stride : op.getMixedStrides()) {
      Value strideOp =
          getValueOrCreateConstantIntOp(rewriter, op->getLoc(), stride);
      stridesAtI64.emplace_back(getValueOrCreateCastToIndexLike(
          rewriter, op->getLoc(), rewriter.getIntegerType(64), strideOp));
    }
    SmallVector<Value> offsetsAtI32;
    for (auto offset : op.getMixedOffsets()) {
      Value offsetOp =
          getValueOrCreateConstantIntOp(rewriter, op->getLoc(), offset);
      offsetsAtI32.emplace_back(getValueOrCreateCastToIndexLike(
          rewriter, op->getLoc(), rewriter.getIntegerType(32), offsetOp));
    }
    auto resultType = dyn_cast<ShapedType>(op->getResultTypes()[0]);
    assert(resultType != nullptr); // TODO: fail gracefully
    auto tensorShape = SmallVector<int32_t>(resultType.getShape());
    MakeTensorPtrOp makeTensorPtrOp = rewriter.create<MakeTensorPtrOp>(
        op->getLoc(), base, shapeAtI64, stridesAtI64, offsetsAtI32, tensorShape,
        op.getOrder());
    rewriter.replaceOp(op, makeTensorPtrOp);
    return success();
  }
};

class StructuredToTritonPass
    : public StructuredToTritonBase<StructuredToTritonPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto moduleOp = getOperation();

    // TODO: make into a proper conversion pass
    // ConversionTarget target(*context);
    // target.addLegalDialect<TritonDialect>();
    // target.addLegalDialect<arith::ArithDialect>();
    // target.addIllegalDialect<tts::TritonStructuredDialect>();

    RewritePatternSet patterns(context);
    patterns.addWithLabel<ConvertLoad>({"convertLoad"}, context);
    patterns.addWithLabel<ConvertStore>({"convertStore"}, context);
    patterns.addWithLabel<ConvertMakeTensorPtr>({"convertMakeTensorPtr"},
                                                context);

    (void)applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
    // TODO: make into a proper conversion pass
    // if (failed(applyFullConversion(moduleOp, target, std::move(patterns)))) {
    // // <-- new thing
    //  signalPassFailure();
    //}
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createStructuredToTritonPass() {
  return std::make_unique<StructuredToTritonPass>();
}
