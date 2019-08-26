// RUN: %target-run-simple-swift
// REQUIRES: executable_test

import StdlibUnittest

var AnyDerivativeTests = TestSuite("AnyDerivative")

struct Vector : Differentiable {
  var x, y: Float
}
struct Generic<T: Differentiable> : Differentiable {
  var x: T
}

extension AnyDerivative {
  // This exists only to faciliate testing.
  func moved(along direction: TangentVector) -> Self {
    var result = self
    result.move(along: direction)
    return result
  }
}

AnyDerivativeTests.test("Vector") {
  var tan = AnyDerivative(Vector.TangentVector(x: 1, y: 1))
  tan += tan
  expectEqual(AnyDerivative(Vector.TangentVector(x: 2, y: 2)), tan)
  expectEqual(AnyDerivative(Vector.TangentVector(x: 4, y: 4)), tan + tan)
  expectEqual(AnyDerivative(Vector.TangentVector(x: 0, y: 0)), tan - tan)
  expectEqual(AnyDerivative(Vector.TangentVector(x: 4, y: 4)), tan.moved(along: tan))
  expectEqual(AnyDerivative(Vector.TangentVector(x: 2, y: 2)), tan)
}

AnyDerivativeTests.test("Generic") {
  var tan = AnyDerivative(Generic<Float>.TangentVector(x: 1))
  let cotan = AnyDerivative(Generic<Float>.TangentVector(x: 1))
  tan += tan
  expectEqual(AnyDerivative(Generic<Float>.TangentVector(x: 2)), tan)
  expectEqual(AnyDerivative(Generic<Float>.TangentVector(x: 4)), tan + tan)
  expectEqual(AnyDerivative(Generic<Float>.TangentVector(x: 0)), tan - tan)
  expectEqual(AnyDerivative(Generic<Float>.TangentVector(x: 4)), tan.moved(along: tan))
  expectEqual(AnyDerivative(Generic<Float>.TangentVector(x: 1)), cotan)
}

AnyDerivativeTests.test("Zero") {
  var zero = AnyDerivative.zero
  zero += zero
  zero -= zero
  expectEqual(zero, zero + zero)
  expectEqual(zero, zero - zero)
  expectEqual(zero, zero.moved(along: zero))

  var tan = AnyDerivative(Vector.TangentVector(x: 1, y: 1))
  expectEqual(zero, zero)
  expectEqual(AnyDerivative(Vector.TangentVector.zero), tan - tan)
  expectNotEqual(AnyDerivative(Vector.TangentVector.zero), zero)
  expectNotEqual(AnyDerivative.zero, tan - tan)
  tan += zero
  tan -= zero
  expectEqual(tan, tan + zero)
  expectEqual(tan, tan - zero)
  expectEqual(tan, tan.moved(along: zero))
  expectEqual(tan, zero.moved(along: tan))
  expectEqual(zero, zero)
  expectEqual(tan, tan)
}

AnyDerivativeTests.test("Casting") {
  let tan = AnyDerivative(Vector.TangentVector(x: 1, y: 1))
  expectEqual(Vector.TangentVector(x: 1, y: 1), tan.base as? Vector.TangentVector)

  let genericTan = AnyDerivative(Generic<Float>.TangentVector(x: 1))
  expectEqual(Generic<Float>.TangentVector(x: 1),
              genericTan.base as? Generic<Float>.TangentVector)
  expectEqual(nil, genericTan.base as? Generic<Double>.TangentVector)

  let zero = AnyDerivative.zero
  expectEqual(nil, zero.base as? Float)
  expectEqual(nil, zero.base as? Vector.TangentVector)
  expectEqual(nil, zero.base as? Generic<Float>.TangentVector)
}

AnyDerivativeTests.test("Derivatives") {
  // Test `AnyDerivative` operations.
  func tripleSum(_ x: AnyDerivative, _ y: AnyDerivative) -> AnyDerivative {
    let sum = x + y
    return sum + sum + sum
  }

  do {
    let x = AnyDerivative(Float(4))
    let y = AnyDerivative(Float(-2))
    let v = AnyDerivative(Float(1))
    let expectedVJP: Float = 3

    let (𝛁x, 𝛁y) = pullback(at: x, y, in: tripleSum)(v)
    expectEqual(expectedVJP, 𝛁x.base as? Float)
    expectEqual(expectedVJP, 𝛁y.base as? Float)
  }

  do {
    let x = AnyDerivative(Vector.TangentVector(x: 4, y: 5))
    let y = AnyDerivative(Vector.TangentVector(x: -2, y: -1))
    let v = AnyDerivative(Vector.TangentVector(x: 1, y: 1))
    let expectedVJP = Vector.TangentVector(x: 3, y: 3)

    let (𝛁x, 𝛁y) = pullback(at: x, y, in: tripleSum)(v)
    expectEqual(expectedVJP, 𝛁x.base as? Vector.TangentVector)
    expectEqual(expectedVJP, 𝛁y.base as? Vector.TangentVector)
  }

  do {
    let x = AnyDerivative(Generic<Double>.TangentVector(x: 4))
    let y = AnyDerivative(Generic<Double>.TangentVector(x: -2))
    let v = AnyDerivative(Generic<Double>.TangentVector(x: 1))
    let expectedVJP = Generic<Double>.TangentVector(x: 3)

    let (𝛁x, 𝛁y) = pullback(at: x, y, in: tripleSum)(v)
    expectEqual(expectedVJP, 𝛁x.base as? Generic<Double>.TangentVector)
    expectEqual(expectedVJP, 𝛁y.base as? Generic<Double>.TangentVector)
  }

  // Test `AnyDerivative` initializer.
  func typeErased<T>(_ x: T) -> AnyDerivative
    where T : Differentiable, T.TangentVector == T
  {
    let any = AnyDerivative(x)
    return any + any
  }

  do {
    let x: Float = 3
    let v = AnyDerivative(Float(1))
    let 𝛁x = pullback(at: x, in: { x in typeErased(x) })(v)
    let expectedVJP: Float = 2
    expectEqual(expectedVJP, 𝛁x)
  }

  do {
    let x = Vector.TangentVector(x: 4, y: 5)
    let v = AnyDerivative(Vector.TangentVector(x: 1, y: 1))
    let 𝛁x = pullback(at: x, in: { x in typeErased(x) })(v)
    let expectedVJP = Vector.TangentVector(x: 2, y: 2)
    expectEqual(expectedVJP, 𝛁x)
  }

  do {
    let x = Generic<Double>.TangentVector(x: 4)
    let v = AnyDerivative(Generic<Double>.TangentVector(x: 1))
    let 𝛁x = pullback(at: x, in: { x in typeErased(x) })(v)
    let expectedVJP = Generic<Double>.TangentVector(x: 2)
    expectEqual(expectedVJP, 𝛁x)
  }
}

runAllTests()
