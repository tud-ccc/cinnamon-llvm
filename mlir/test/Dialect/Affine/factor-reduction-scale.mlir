// RUN: mlir-opt %s --split-input-file --affine-factor-reduction-scale | FileCheck %s
// RUN: mlir-opt %s --split-input-file --affine-factor-reduction-scale=allow-float-reassociation | FileCheck %s --check-prefix=FLOAT

// A dot product scaled by a value defined outside the loop: the loop sums the
// unscaled products from zero, the scale is applied once after it.

// CHECK-LABEL: func @scaled_dot
//  CHECK-SAME:   (%[[A:.*]]: memref<64xi32>, %[[X:.*]]: memref<64xi32>, %[[S:.*]]: i32, %[[INIT:.*]]: i32)
//       CHECK:   %[[ZERO:.*]] = arith.constant 0 : i32
//       CHECK:   %[[R:.*]] = affine.for %{{.*}} = 0 to 64 iter_args(%[[ACC:.*]] = %[[ZERO]]) -> (i32) {
//       CHECK:     %[[AV:.*]] = affine.load %[[A]]
//       CHECK:     %[[XV:.*]] = affine.load %[[X]]
//   CHECK-NOT:     arith.muli {{.*}}%[[S]]
//       CHECK:     %[[P:.*]] = arith.muli %[[AV]], %[[XV]] : i32
//       CHECK:     %[[N:.*]] = arith.addi %[[ACC]], %[[P]] : i32
//       CHECK:     affine.yield %[[N]]
//       CHECK:   %[[RS:.*]] = arith.muli %[[R]], %[[S]] : i32
//       CHECK:   %[[T:.*]] = arith.addi %[[INIT]], %[[RS]] : i32
//       CHECK:   return %[[T]]
func.func @scaled_dot(%A: memref<64xi32>, %X: memref<64xi32>, %s: i32,
                      %init: i32) -> i32 {
  %r = affine.for %k = 0 to 64 iter_args(%acc = %init) -> (i32) {
    %a = affine.load %A[%k] : memref<64xi32>
    %x = affine.load %X[%k] : memref<64xi32>
    %xs = arith.muli %x, %s overflow<nsw> : i32
    %t = arith.muli %a, %xs overflow<nsw> : i32
    %next = arith.addi %acc, %t overflow<nsw> : i32
    affine.yield %next : i32
  }
  return %r : i32
}

// -----

// The scale loaded in the body from an address nothing in the loop writes:
// the load is repeated after the loop. Operand orders are swapped throughout.

// CHECK-LABEL: func @scale_loaded_in_body
//  CHECK-SAME:   (%[[A:.*]]: memref<64xi32>, %[[X:.*]]: memref<64xi32>, %[[SM:.*]]: memref<i32>, %[[INIT:.*]]: i32)
//       CHECK:   %[[R:.*]] = affine.for
//       CHECK:     arith.muli
//   CHECK-NOT:     arith.muli
//       CHECK:     affine.yield
//       CHECK:   %[[S:.*]] = affine.load %[[SM]][]
//       CHECK:   %[[RS:.*]] = arith.muli %[[R]], %[[S]] : i32
//       CHECK:   arith.addi %[[INIT]], %[[RS]] : i32
func.func @scale_loaded_in_body(%A: memref<64xi32>, %X: memref<64xi32>,
                                %S: memref<i32>, %init: i32) -> i32 {
  %r = affine.for %k = 0 to 64 iter_args(%acc = %init) -> (i32) {
    %a = affine.load %A[%k] : memref<64xi32>
    %x = affine.load %X[%k] : memref<64xi32>
    %s = affine.load %S[] : memref<i32>
    %xs = arith.muli %s, %x : i32
    %t = arith.muli %xs, %a : i32
    %next = arith.addi %t, %acc : i32
    affine.yield %next : i32
  }
  return %r : i32
}

// -----

// The same, but the loop writes the scale's buffer: the scale is not
// invariant.

// CHECK-LABEL: func @scale_written_in_body
//       CHECK:   affine.for
//  CHECK-COUNT-2: arith.muli
//       CHECK:     affine.yield
//   CHECK-NOT:   arith.muli
func.func @scale_written_in_body(%A: memref<64xi32>, %X: memref<64xi32>,
                                 %S: memref<i32>, %init: i32) -> i32 {
  %r = affine.for %k = 0 to 64 iter_args(%acc = %init) -> (i32) {
    %a = affine.load %A[%k] : memref<64xi32>
    %x = affine.load %X[%k] : memref<64xi32>
    %s = affine.load %S[] : memref<i32>
    %xs = arith.muli %x, %s : i32
    %t = arith.muli %a, %xs : i32
    %next = arith.addi %acc, %t : i32
    affine.store %a, %S[] : memref<i32>
    affine.yield %next : i32
  }
  return %r : i32
}

// -----

// A load at a constant index of a view taken in the body is not invariant:
// the view moves with the loop. Nothing here is factored -- in particular the
// load must not be repeated after the loop, where its view is out of scope.

// CHECK-LABEL: func @view_in_body
//       CHECK:   affine.for
//  CHECK-COUNT-2: arith.muli
//       CHECK:     affine.yield
//   CHECK-NOT:   arith.muli
func.func @view_in_body(%A: memref<64xi32>, %X: memref<64xi32>,
                        %Y: memref<64xi32>, %init: i32) -> i32 {
  %r = affine.for %k = 0 to 64 iter_args(%acc = %init) -> (i32) {
    %view = memref.subview %A[%k] [1] [1]
        : memref<64xi32> to memref<1xi32, strided<[1], offset: ?>>
    %a = affine.load %view[0] : memref<1xi32, strided<[1], offset: ?>>
    %x = affine.load %X[%k] : memref<64xi32>
    %y = affine.load %Y[%k] : memref<64xi32>
    %xa = arith.muli %x, %a : i32
    %t = arith.muli %y, %xa : i32
    %next = arith.addi %acc, %t : i32
    affine.yield %next : i32
  }
  return %r : i32
}

// -----

// Two terms per iteration, one of them a bare `x * s`.

// CHECK-LABEL: func @two_terms
//  CHECK-SAME:   (%[[A:.*]]: memref<64xi32>, %[[X:.*]]: memref<64xi32>, %[[S:.*]]: i32, %[[INIT:.*]]: i32)
//       CHECK:   %[[R:.*]] = affine.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (i32) {
//       CHECK:     %[[AV:.*]] = affine.load %[[A]]
//       CHECK:     %[[XV:.*]] = affine.load %[[X]]
//       CHECK:     %[[P:.*]] = arith.muli %[[AV]], %[[XV]] : i32
//       CHECK:     %[[N0:.*]] = arith.addi %[[ACC]], %[[P]] : i32
//       CHECK:     %[[N1:.*]] = arith.addi %[[N0]], %[[XV]] : i32
//       CHECK:     affine.yield %[[N1]]
//       CHECK:   arith.muli %[[R]], %[[S]] : i32
func.func @two_terms(%A: memref<64xi32>, %X: memref<64xi32>, %s: i32,
                     %init: i32) -> i32 {
  %r = affine.for %k = 0 to 64 iter_args(%acc = %init) -> (i32) {
    %a = affine.load %A[%k] : memref<64xi32>
    %x = affine.load %X[%k] : memref<64xi32>
    %xs = arith.muli %x, %s : i32
    %t = arith.muli %a, %xs : i32
    %n0 = arith.addi %acc, %t : i32
    %xs2 = arith.muli %x, %s : i32
    %n1 = arith.addi %n0, %xs2 : i32
    affine.yield %n1 : i32
  }
  return %r : i32
}

// -----

// A nest: the reduction is split over two loops, the inner one starting from
// the outer one's accumulator. The scale leaves both, and the inner loop still
// starts from the outer accumulator, so that the accumulator heads the sum
// rather than being added after it.

// CHECK-LABEL: func @nest
//  CHECK-SAME:   (%[[A:.*]]: memref<4x64xi32>, %[[X:.*]]: memref<4x64xi32>, %[[S:.*]]: i32, %[[INIT:.*]]: i32)
//       CHECK:   %[[ZERO:.*]] = arith.constant 0 : i32
//       CHECK:   %[[R:.*]] = affine.for %{{.*}} = 0 to 4 iter_args(%[[OACC:.*]] = %[[ZERO]]) -> (i32) {
//       CHECK:     %[[IR:.*]] = affine.for %{{.*}} = 0 to 64 iter_args(%[[IACC:.*]] = %[[OACC]]) -> (i32) {
//       CHECK:       %[[P:.*]] = arith.muli
//       CHECK:       %[[N:.*]] = arith.addi %[[IACC]], %[[P]] : i32
//       CHECK:       affine.yield %[[N]]
//   CHECK-NOT:     arith
//       CHECK:     affine.yield %[[IR]]
//       CHECK:   %[[RS:.*]] = arith.muli %[[R]], %[[S]] : i32
//       CHECK:   arith.addi %[[INIT]], %[[RS]] : i32
func.func @nest(%A: memref<4x64xi32>, %X: memref<4x64xi32>, %s: i32,
                %init: i32) -> i32 {
  %r = affine.for %j = 0 to 4 iter_args(%oacc = %init) -> (i32) {
    %ir = affine.for %k = 0 to 64 iter_args(%acc = %oacc) -> (i32) {
      %a = affine.load %A[%j, %k] : memref<4x64xi32>
      %x = affine.load %X[%j, %k] : memref<4x64xi32>
      %xs = arith.muli %x, %s : i32
      %t = arith.muli %a, %xs : i32
      %next = arith.addi %acc, %t : i32
      affine.yield %next : i32
    }
    affine.yield %ir : i32
  }
  return %r : i32
}

// -----

// No invariant factor: a plain dot product is left alone.

// CHECK-LABEL: func @plain_dot
//       CHECK:   affine.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32) {
//       CHECK:     arith.muli
//       CHECK:     affine.yield
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return
func.func @plain_dot(%A: memref<64xi32>, %X: memref<64xi32>,
                     %init: i32) -> i32 {
  %r = affine.for %k = 0 to 64 iter_args(%acc = %init) -> (i32) {
    %a = affine.load %A[%k] : memref<64xi32>
    %x = affine.load %X[%k] : memref<64xi32>
    %t = arith.muli %a, %x : i32
    %next = arith.addi %acc, %t : i32
    affine.yield %next : i32
  }
  return %r : i32
}

// -----

// A product with another use inside the loop cannot change value.

// CHECK-LABEL: func @term_used_twice
//       CHECK:   affine.for
//  CHECK-COUNT-2: arith.muli
//       CHECK:     affine.yield
//   CHECK-NOT:   arith.muli
func.func @term_used_twice(%A: memref<64xi32>, %X: memref<64xi32>, %s: i32,
                           %init: i32) -> i32 {
  %r = affine.for %k = 0 to 64 iter_args(%acc = %init) -> (i32) {
    %a = affine.load %A[%k] : memref<64xi32>
    %x = affine.load %X[%k] : memref<64xi32>
    %xs = arith.muli %x, %s : i32
    %t = arith.muli %a, %xs : i32
    %next = arith.addi %acc, %t : i32
    affine.store %xs, %X[%k] : memref<64xi32>
    affine.yield %next : i32
  }
  return %r : i32
}

// -----

// Floating point needs an opt-in: by default the loop is left alone; the
// pass option allows it, and so do `reassoc` flags on the ops (next case).

// CHECK-LABEL: func @float_dot
//       CHECK:   affine.for
//  CHECK-COUNT-2: arith.mulf
//       CHECK:     affine.yield
//   CHECK-NOT:   arith.mulf

// FLOAT-LABEL: func @float_dot
//  FLOAT-SAME:   (%[[A:.*]]: memref<64xf32>, %[[X:.*]]: memref<64xf32>, %[[S:.*]]: f32, %[[INIT:.*]]: f32)
//       FLOAT:   %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
//       FLOAT:   %[[R:.*]] = affine.for {{.*}} iter_args(%{{.*}} = %[[ZERO]]) -> (f32) {
//       FLOAT:     arith.mulf
//   FLOAT-NOT:     arith.mulf
//       FLOAT:     affine.yield
//       FLOAT:   %[[RS:.*]] = arith.mulf %[[R]], %[[S]] : f32
//       FLOAT:   arith.addf %[[INIT]], %[[RS]] : f32
func.func @float_dot(%A: memref<64xf32>, %X: memref<64xf32>, %s: f32,
                     %init: f32) -> f32 {
  %r = affine.for %k = 0 to 64 iter_args(%acc = %init) -> (f32) {
    %a = affine.load %A[%k] : memref<64xf32>
    %x = affine.load %X[%k] : memref<64xf32>
    %xs = arith.mulf %x, %s : f32
    %t = arith.mulf %a, %xs : f32
    %next = arith.addf %acc, %t : f32
    affine.yield %next : f32
  }
  return %r : f32
}

// -----

// CHECK-LABEL: func @float_reassoc_flags
//       CHECK:   %[[R:.*]] = affine.for
//       CHECK:     arith.mulf
//   CHECK-NOT:     arith.mulf
//       CHECK:     affine.yield
//       CHECK:   arith.mulf %[[R]]
func.func @float_reassoc_flags(%A: memref<64xf32>, %X: memref<64xf32>,
                               %s: f32, %init: f32) -> f32 {
  %r = affine.for %k = 0 to 64 iter_args(%acc = %init) -> (f32) {
    %a = affine.load %A[%k] : memref<64xf32>
    %x = affine.load %X[%k] : memref<64xf32>
    %xs = arith.mulf %x, %s fastmath<reassoc> : f32
    %t = arith.mulf %a, %xs fastmath<reassoc> : f32
    %next = arith.addf %acc, %t fastmath<reassoc> : f32
    affine.yield %next : f32
  }
  return %r : f32
}

// -----

// A floating-point loop that may not run keeps its initial value exactly
// only if left alone, even with the opt-in.

// FLOAT-LABEL: func @float_maybe_empty
//       FLOAT:   affine.for
//  FLOAT-COUNT-2: arith.mulf
//       FLOAT:     affine.yield
//   FLOAT-NOT:   arith.mulf
func.func @float_maybe_empty(%A: memref<?xf32>, %X: memref<?xf32>, %s: f32,
                             %init: f32, %n: index) -> f32 {
  %r = affine.for %k = 0 to %n iter_args(%acc = %init) -> (f32) {
    %a = affine.load %A[%k] : memref<?xf32>
    %x = affine.load %X[%k] : memref<?xf32>
    %xs = arith.mulf %x, %s : f32
    %t = arith.mulf %a, %xs : f32
    %next = arith.addf %acc, %t : f32
    affine.yield %next : f32
  }
  return %r : f32
}
