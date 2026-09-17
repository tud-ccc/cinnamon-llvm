// RUN: mlir-opt %s -linalg-fuse-elementwise-ops -split-input-file | FileCheck %s --check-prefix=RECOMPUTE
// RUN: mlir-opt %s -linalg-fuse-elementwise-ops="fuse-with-recompute=false" -split-input-file | FileCheck %s --check-prefix=NORECOMPUTE

#id = affine_map<(d0, d1) -> (d0, d1)>
#lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
#out = affine_map<(d0, d1, d2) -> (d0, d1)>

// The producer feeds the LHS of a contraction, which indexes it with two of
// its three loops: fusing it recomputes the scaling once per reduction step.
// By default that fusion happens; with fuse-with-recompute=false the
// producer stays a separate op.

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
