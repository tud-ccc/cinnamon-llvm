//===- FactorReductionScale.cpp - Hoist a sum's common factor -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrites a reduction loop whose every term is scaled by a loop-invariant
// value, `acc += a[k] * (x[k] * s)`, into `acc += s * sum_k a[k] * x[k]`: the
// loop sums the unscaled terms and the scale is applied once, after it.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/Debug.h"

#include <optional>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINEFACTORREDUCTIONSCALE
#include "mlir/Dialect/Affine/Transforms/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-factor-reduction-scale"

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Which adds and multiplies may be reassociated: integer ones always, since
/// they are exact modulo 2^n; floating-point ones only when the pass is told
/// so or the op itself carries the `reassoc` fast-math flag.
struct Reassociation {
  bool allowFloat;

  bool permits(Operation *op) const {
    if (isa<arith::AddIOp, arith::MulIOp>(op))
      return true;
    if (!isa<arith::AddFOp, arith::MulFOp>(op))
      return false;
    if (allowFloat)
      return true;
    auto fmf = cast<arith::ArithFastMathInterface>(op).getFastMathFlagsAttr();
    return fmf && arith::bitEnumContainsAll(fmf.getValue(),
                                            arith::FastMathFlags::reassoc);
  }
  bool isAdd(Operation *op) const {
    return isa<arith::AddIOp, arith::AddFOp>(op) && permits(op);
  }
  bool isMul(Operation *op) const {
    return isa<arith::MulIOp, arith::MulFOp>(op) && permits(op);
  }
};

/// One term of the chain with the scale in it: either `outer = x * s`, or
/// `outer = a * inner` with `inner = x * s`, in any operand order.
struct Term {
  Operation *outer;
  Operation *inner; // Null for `x * s`.
  unsigned operand; // outer's operand that is `s`, resp. `inner`.
};

/// `t` as a term scaled by `s`. Every multiply involved must have no other
/// use, since the rewrite changes its value.
static std::optional<Term> matchTerm(Value t, Value s,
                                     const Reassociation &reassoc) {
  Operation *outer = t.getDefiningOp();
  if (!outer || !reassoc.isMul(outer) || !t.hasOneUse())
    return std::nullopt;
  for (unsigned i = 0; i < 2; ++i)
    if (outer->getOperand(i) == s)
      return Term{outer, nullptr, i};
  for (unsigned i = 0; i < 2; ++i) {
    Value side = outer->getOperand(i);
    Operation *inner = side.getDefiningOp();
    if (!inner || inner->getName() != outer->getName() || !side.hasOneUse() ||
        !reassoc.permits(inner))
      continue;
    if (llvm::is_contained(inner->getOperands(), s))
      return Term{outer, inner, i};
  }
  return std::nullopt;
}

/// The values a chain ending at `end` could be factored by: whatever a
/// multiply feeding its last add multiplies by, directly or one level down.
static SmallVector<Value> scaleCandidates(Value end,
                                          const Reassociation &reassoc) {
  SmallVector<Value> result;
  Operation *add = end.getDefiningOp();
  if (!add || !reassoc.isAdd(add))
    return result;
  for (Value side : add->getOperands()) {
    Operation *outer = side.getDefiningOp();
    if (!outer || !reassoc.isMul(outer))
      continue;
    for (Value v : outer->getOperands()) {
      result.push_back(v);
      if (Operation *inner = v.getDefiningOp();
          inner && inner->getName() == outer->getName())
        llvm::append_range(result, inner->getOperands());
    }
  }
  return result;
}

/// `acc + t_0 + ... + t_n` ending at `end`, every `t_i` scaled by `scale`.
/// `adds` and `terms` run from the last add back to the first.
struct Chain {
  Value scale;
  SmallVector<Operation *> adds;
  SmallVector<Term> terms;
};

/// `acc`, `end` and every add in between must have a single use -- the next
/// add, or for `end` the loop's yield -- since each of their values changes.
static std::optional<Chain> matchChain(Value end, Value acc, Value scale,
                                       const Reassociation &reassoc) {
  if (!acc.hasOneUse())
    return std::nullopt;
  Chain chain;
  chain.scale = scale;
  Value node = end;
  while (node != acc) {
    Operation *add = node.getDefiningOp();
    if (!add || !reassoc.isAdd(add) || !node.hasOneUse())
      return std::nullopt;
    std::optional<Term> term;
    Value rest;
    for (unsigned i = 0; i < 2 && !term; ++i) {
      term = matchTerm(add->getOperand(i), scale, reassoc);
      rest = add->getOperand(1 - i);
    }
    if (!term)
      return std::nullopt;
    chain.adds.push_back(add);
    chain.terms.push_back(*term);
    node = rest;
  }
  if (chain.terms.empty())
    return std::nullopt;
  return chain;
}

/// Drops the scale from every term, and the no-wrap flags from every op whose
/// value changes.
static void unscaleChain(RewriterBase &rewriter, Chain &chain) {
  auto dropOverflowFlags = [&](Operation *op) {
    auto ovf = dyn_cast<arith::ArithIntegerOverflowFlagsInterface>(op);
    if (!ovf)
      return;
    rewriter.modifyOpInPlace(op, [&] {
      op->setAttr(ovf.getIntegerOverflowAttrName(),
                  arith::IntegerOverflowFlagsAttr::get(
                      op->getContext(), arith::IntegerOverflowFlags::none));
    });
  };
  for (Operation *add : chain.adds)
    dropOverflowFlags(add);
  for (Term &term : chain.terms) {
    if (!term.inner) {
      rewriter.replaceOp(term.outer, term.outer->getOperand(1 - term.operand));
      continue;
    }
    Value x = term.inner->getOperand(0) == chain.scale
                  ? term.inner->getOperand(1)
                  : term.inner->getOperand(0);
    rewriter.modifyOpInPlace(term.outer,
                             [&] { term.outer->setOperand(term.operand, x); });
    dropOverflowFlags(term.outer);
    rewriter.eraseOp(term.inner);
  }
}

class Factorizer {
public:
  Factorizer(AliasAnalysis &aliasAnalysis, bool allowFloat)
      : aliasAnalysis(aliasAnalysis), reassoc{allowFloat} {}

  void run(AffineForOp loop) {
    for (unsigned i = 0; i < loop.getNumRegionIterArgs(); ++i)
      factorIterArg(loop, i);
  }

  /// `x + r`, with `r` a sum a loop carries from the identity and `x`
  /// available before the loop: the loop starts from `x` instead, and `r`
  /// replaces the add.
  ///
  /// Factoring a nest leaves this behind. The inner loop is factored first,
  /// starting from the identity with its old initial value -- the outer
  /// loop's accumulator -- added after it together with the scale; factoring
  /// the outer loop then removes the scale, and what remains is the add. As
  /// the last add of the inner sum, it has the accumulator enter after every
  /// term, and a code generator that reorders the terms of a sum by rank --
  /// LLVM's Reassociate puts the loop-carried value first -- rebuilds the
  /// whole unrolled sum at its root, so that every term is live at once.
  void seedFromSum(AffineForOp loop, unsigned i, DominanceInfo &dominance) {
    Value result = loop.getResult(i);
    if (!result.hasOneUse())
      return;
    Operation *add = *result.getUsers().begin();
    Operation *yielded = loop.getYieldedValues()[i].getDefiningOp();
    if (!reassoc.isAdd(add) || !yielded ||
        yielded->getName() != add->getName())
      return;
    Value x = add->getOperand(0) == result ? add->getOperand(1)
                                           : add->getOperand(0);
    if (!dominance.properlyDominates(x, loop))
      return;
    std::optional<TypedAttr> identity = arith::getNeutralElement(add);
    Attribute init;
    if (!identity || !matchPattern(loop.getInits()[i], m_Constant(&init)) ||
        init != *identity)
      return;
    LLVM_DEBUG(llvm::dbgs() << "seeding iter_arg " << i << " of " << loop
                            << " from " << x << "\n");
    IRRewriter rewriter(loop->getContext());
    rewriter.modifyOpInPlace(loop,
                             [&] { loop.getInitsMutable()[i].assign(x); });
    rewriter.replaceOp(add, result);
  }

private:
  AliasAnalysis &aliasAnalysis;
  Reassociation reassoc;

  /// Whether no op in `loop`'s body may write `memref`. An op whose effects
  /// are unknown, or that writes without saying where, may.
  bool isUnwrittenIn(AffineForOp loop, Value memref) {
    WalkResult walk = loop.getBody()->walk([&](Operation *op) {
      if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        return WalkResult::advance();
      auto iface = dyn_cast<MemoryEffectOpInterface>(op);
      if (!iface)
        return WalkResult::interrupt();
      SmallVector<MemoryEffects::EffectInstance> effects;
      iface.getEffects(effects);
      for (const MemoryEffects::EffectInstance &effect : effects) {
        if (!isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()))
          continue;
        if (!effect.getValue() ||
            !aliasAnalysis.alias(effect.getValue(), memref).isNo())
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return !walk.wasInterrupted();
  }

  /// Whether `scale` has one value throughout `loop`: defined outside it, or
  /// loaded in its body from a loop-invariant address nothing in the loop
  /// writes. `load` is set in the second case: the scale is applied after the
  /// loop, where that load has to be repeated. The address is invariant when
  /// every operand of the load is defined outside the loop -- the memref as
  /// well as the indices, since a view taken in the body moves with it.
  bool isInvariantScale(AffineForOp loop, Value scale, AffineLoadOp &load) {
    load = nullptr;
    if (loop.isDefinedOutsideOfLoop(scale))
      return true;
    auto candidate = scale.getDefiningOp<AffineLoadOp>();
    if (!candidate || candidate->getBlock() != loop.getBody() ||
        !llvm::all_of(candidate->getOperands(),
                      [&](Value v) { return loop.isDefinedOutsideOfLoop(v); }))
      return false;
    if (!isUnwrittenIn(loop, candidate.getMemRef()))
      return false;
    load = candidate;
    return true;
  }

  /// `yield acc + t_0 + ...` over iteration argument `i`, every term scaled
  /// by the same invariant value `s`: the loop starts from the identity and
  /// sums the unscaled terms, and `init + result * s` replaces its result.
  void factorIterArg(AffineForOp loop, unsigned i) {
    Value acc = loop.getRegionIterArgs()[i];
    Value end = loop.getYieldedValues()[i];
    std::optional<Chain> chain;
    AffineLoadOp scaleLoad;
    for (Value scale : scaleCandidates(end, reassoc)) {
      if (!isInvariantScale(loop, scale, scaleLoad))
        continue;
      if ((chain = matchChain(end, acc, scale, reassoc)))
        break;
    }
    if (!chain)
      return;

    Operation *lastAdd = chain->adds.front();
    Operation *lastMul = chain->terms.front().outer;
    // A floating-point sum may only start from the identity if the loop runs:
    // `init + (-0.0) * s` is not `init` for an infinite `s`.
    std::optional<uint64_t> trip = getConstantTripCount(loop);
    if (isa<arith::AddFOp>(lastAdd) && (!trip || *trip == 0))
      return;
    std::optional<TypedAttr> identity = arith::getNeutralElement(lastAdd);
    if (!identity)
      return;
    LLVM_DEBUG(llvm::dbgs() << "factoring iter_arg " << i << " of " << loop
                            << "\n");

    IRRewriter rewriter(loop->getContext());
    Value init = loop.getInits()[i];
    rewriter.setInsertionPoint(loop);
    Value zero = arith::ConstantOp::create(rewriter, loop.getLoc(), *identity);
    rewriter.modifyOpInPlace(loop,
                             [&] { loop.getInitsMutable()[i].assign(zero); });

    rewriter.setInsertionPointAfter(loop);
    Value scale = scaleLoad ? rewriter.clone(*scaleLoad)->getResult(0)
                            : chain->scale;
    Value result = loop.getResult(i);
    Location loc = loop.getLoc();
    Operation *scaled = rewriter.create(
        OperationState(loc, lastMul->getName(), ValueRange{result, scale},
                       result.getType()));
    Operation *total = rewriter.create(
        OperationState(loc, lastAdd->getName(),
                       ValueRange{init, scaled->getResult(0)},
                       result.getType()));
    rewriter.replaceAllUsesExcept(result, total->getResult(0), scaled);

    unscaleChain(rewriter, *chain);
  }
};

struct AffineFactorReductionScalePass
    : public affine::impl::AffineFactorReductionScaleBase<
          AffineFactorReductionScalePass> {
  using Base::Base;

  void runOnOperation() override {
    Factorizer factorizer(getAnalysis<AliasAnalysis>(),
                          allowFloatReassociation);
    getOperation()->walk([&](AffineForOp loop) { factorizer.run(loop); });
    // After every loop is factored: the outer loop of a nest is what turns
    // the inner one's result into a plain add.
    // Collected first: seeding erases the add that follows a loop, which a
    // walk may be about to visit.
    SmallVector<AffineForOp> loops;
    getOperation()->walk([&](AffineForOp loop) { loops.push_back(loop); });
    DominanceInfo dominance(getOperation());
    for (AffineForOp loop : loops)
      for (unsigned i = 0; i < loop.getNumRegionIterArgs(); ++i)
        factorizer.seedFromSum(loop, i, dominance);
  }
};

} // namespace
