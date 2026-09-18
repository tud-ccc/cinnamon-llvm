// RUN: mlir-opt %s -linalg-fuse-elementwise-ops -split-input-file | FileCheck %s --check-prefix=RECOMPUTE
// RUN: mlir-opt %s -linalg-fuse-elementwise-ops="fuse-with-recompute=false" -split-input-file | FileCheck %s --check-prefix=NORECOMPUTE

#id = affine_map<(d0, d1) -> (d0, d1)>
#lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
#out = affine_map<(d0, d1, d2) -> (d0, d1)>

// The producer feeds the LHS of a contraction, which indexes it with two of
// its three loops: fusing it recomputes its two ops once per reduction step.
// By default that fusion happens; with fuse-with-recompute=false a producer
// with more than max-recompute-body-ops (default 1) ops stays a separate op.

// RECOMPUTE-LABEL: func @producer_into_contraction
//       RECOMPUTE:   linalg.generic
//   RECOMPUTE-NOT:   linalg.generic

// NORECOMPUTE-LABEL: func @producer_into_contraction
//       NORECOMPUTE:   linalg.generic
//  NORECOMPUTE-SAME:     iterator_types = ["parallel", "parallel"]
//       NORECOMPUTE:   linalg.generic
//  NORECOMPUTE-SAME:     iterator_types = ["parallel", "parallel", "reduction"]
func.func @producer_into_contraction(%a: tensor<4x8xf32>, %b: tensor<8x6xf32>,
                                     %init: tensor<4x6xf32>) -> tensor<4x6xf32> {
  %cst = arith.constant 2.0 : f32
  %empty = tensor.empty() : tensor<4x8xf32>
  %scaled = linalg.generic {indexing_maps = [#id, #id],
                            iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<4x8xf32>) outs(%empty : tensor<4x8xf32>) {
  ^bb0(%in: f32, %o: f32):
    %s = arith.mulf %in, %cst : f32
    %t = arith.addf %s, %cst : f32
    linalg.yield %t : f32
  } -> tensor<4x8xf32>
  %res = linalg.generic {indexing_maps = [#lhs, #rhs, #out],
                         iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%scaled, %b : tensor<4x8xf32>, tensor<8x6xf32>)
      outs(%init : tensor<4x6xf32>) {
  ^bb0(%l: f32, %r: f32, %acc: f32):
    %m = arith.mulf %l, %r : f32
    %add = arith.addf %acc, %m : f32
    linalg.yield %add : f32
  } -> tensor<4x6xf32>
  return %res : tensor<4x6xf32>
}

// -----

#id = affine_map<(d0, d1) -> (d0, d1)>
#lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
#out = affine_map<(d0, d1, d2) -> (d0, d1)>

// A producer of a single op -- here a scaling -- is cheap enough to redo per
// reduction step, and cheaper than materializing its result: it is fused
// either way, being within max-recompute-body-ops.

// RECOMPUTE-LABEL: func @cheap_producer_into_contraction
//       RECOMPUTE:   linalg.generic
//   RECOMPUTE-NOT:   linalg.generic

// NORECOMPUTE-LABEL: func @cheap_producer_into_contraction
//       NORECOMPUTE:   linalg.generic
//  NORECOMPUTE-SAME:     iterator_types = ["parallel", "parallel", "reduction"]
//   NORECOMPUTE-NOT:   linalg.generic
func.func @cheap_producer_into_contraction(%a: tensor<4x8xf32>, %b: tensor<8x6xf32>,
                                           %init: tensor<4x6xf32>) -> tensor<4x6xf32> {
  %cst = arith.constant 2.0 : f32
  %empty = tensor.empty() : tensor<4x8xf32>
  %scaled = linalg.generic {indexing_maps = [#id, #id],
                            iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<4x8xf32>) outs(%empty : tensor<4x8xf32>) {
  ^bb0(%in: f32, %o: f32):
    %s = arith.mulf %in, %cst : f32
    linalg.yield %s : f32
  } -> tensor<4x8xf32>
  %res = linalg.generic {indexing_maps = [#lhs, #rhs, #out],
                         iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%scaled, %b : tensor<4x8xf32>, tensor<8x6xf32>)
      outs(%init : tensor<4x6xf32>) {
  ^bb0(%l: f32, %r: f32, %acc: f32):
    %m = arith.mulf %l, %r : f32
    %add = arith.addf %acc, %m : f32
    linalg.yield %add : f32
  } -> tensor<4x6xf32>
  return %res : tensor<4x6xf32>
}

// -----

#id = affine_map<(d0, d1) -> (d0, d1)>

// An elementwise chain where the consumer indexes the producer with all of
// its loops is fused either way.

// RECOMPUTE-LABEL: func @elementwise_chain
//       RECOMPUTE:   linalg.generic
//   RECOMPUTE-NOT:   linalg.generic

// NORECOMPUTE-LABEL: func @elementwise_chain
//       NORECOMPUTE:   linalg.generic
//   NORECOMPUTE-NOT:   linalg.generic
func.func @elementwise_chain(%a: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %cst = arith.constant 2.0 : f32
  %empty = tensor.empty() : tensor<4x8xf32>
  %scaled = linalg.generic {indexing_maps = [#id, #id],
                            iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<4x8xf32>) outs(%empty : tensor<4x8xf32>) {
  ^bb0(%in: f32, %o: f32):
    %s = arith.mulf %in, %cst : f32
    linalg.yield %s : f32
  } -> tensor<4x8xf32>
  %empty2 = tensor.empty() : tensor<4x8xf32>
  %res = linalg.generic {indexing_maps = [#id, #id],
                         iterator_types = ["parallel", "parallel"]}
      ins(%scaled : tensor<4x8xf32>) outs(%empty2 : tensor<4x8xf32>) {
  ^bb0(%in: f32, %o: f32):
    %s = arith.addf %in, %cst : f32
    linalg.yield %s : f32
  } -> tensor<4x8xf32>
  return %res : tensor<4x8xf32>
}

// -----

#bcast_in = affine_map<(d0, d1) -> (d1)>
#bcast_out = affine_map<(d0, d1) -> (d0, d1)>
#lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
#out = affine_map<(d0, d1, d2) -> (d0, d1)>

// A broadcast feeding the RHS of a contraction is indexed by two of the
// consumer's three loops too, but its body computes nothing: fused, the
// consumer loads the vector element it would have loaded anyway; unfused, a
// whole 8x6 tensor is materialized. It is fused either way.

// RECOMPUTE-LABEL: func @broadcast_into_contraction
//       RECOMPUTE:   linalg.generic
//   RECOMPUTE-NOT:   linalg.generic

// NORECOMPUTE-LABEL: func @broadcast_into_contraction
//       NORECOMPUTE:   linalg.generic
//  NORECOMPUTE-SAME:     iterator_types = ["parallel", "parallel", "reduction"]
//   NORECOMPUTE-NOT:   linalg.generic
func.func @broadcast_into_contraction(%a: tensor<4x8xf32>, %v: tensor<6xf32>,
                                      %init: tensor<4x6xf32>) -> tensor<4x6xf32> {
  %empty = tensor.empty() : tensor<8x6xf32>
  %b = linalg.generic {indexing_maps = [#bcast_in, #bcast_out],
                       iterator_types = ["parallel", "parallel"]}
      ins(%v : tensor<6xf32>) outs(%empty : tensor<8x6xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  } -> tensor<8x6xf32>
  %res = linalg.generic {indexing_maps = [#lhs, #rhs, #out],
                         iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<4x8xf32>, tensor<8x6xf32>)
      outs(%init : tensor<4x6xf32>) {
  ^bb0(%l: f32, %r: f32, %acc: f32):
    %m = arith.mulf %l, %r : f32
    %add = arith.addf %acc, %m : f32
    linalg.yield %add : f32
  } -> tensor<4x6xf32>
  return %res : tensor<4x6xf32>
}

// -----

#fill_out = affine_map<(d0, d1) -> (d0, d1)>
#lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
#out = affine_map<(d0, d1, d2) -> (d0, d1)>

// A fill (a generic yielding a scalar from outside its body) feeding a
// contraction input computes nothing either and is fused either way.

// RECOMPUTE-LABEL: func @fill_into_contraction
//       RECOMPUTE:   linalg.generic
//   RECOMPUTE-NOT:   linalg.generic

// NORECOMPUTE-LABEL: func @fill_into_contraction
//       NORECOMPUTE:   linalg.generic
//  NORECOMPUTE-SAME:     iterator_types = ["parallel", "parallel", "reduction"]
//   NORECOMPUTE-NOT:   linalg.generic
func.func @fill_into_contraction(%a: tensor<4x8xf32>, %init: tensor<4x6xf32>)
    -> tensor<4x6xf32> {
  %cst = arith.constant 1.0 : f32
  %empty = tensor.empty() : tensor<8x6xf32>
  %ones = linalg.generic {indexing_maps = [#fill_out],
                          iterator_types = ["parallel", "parallel"]}
      outs(%empty : tensor<8x6xf32>) {
  ^bb0(%o: f32):
    linalg.yield %cst : f32
  } -> tensor<8x6xf32>
  %res = linalg.generic {indexing_maps = [#lhs, #rhs, #out],
                         iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %ones : tensor<4x8xf32>, tensor<8x6xf32>)
      outs(%init : tensor<4x6xf32>) {
  ^bb0(%l: f32, %r: f32, %acc: f32):
    %m = arith.mulf %l, %r : f32
    %add = arith.addf %acc, %m : f32
    linalg.yield %add : f32
  } -> tensor<4x6xf32>
  return %res : tensor<4x6xf32>
}
