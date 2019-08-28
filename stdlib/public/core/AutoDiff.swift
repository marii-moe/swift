//===--- AutoDiff.swift ---------------------------------------*- swift -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2017 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//
//
// SWIFT_ENABLE_TENSORFLOW
//
// This file defines support for automatic differentiation.
//
//===----------------------------------------------------------------------===//

infix operator .* : MultiplicationPrecedence
infix operator .*= : AssignmentPrecedence

//===----------------------------------------------------------------------===//
// Compiler Protocols
//===----------------------------------------------------------------------===//

/// A type with values that support pointwise multiplication.
// TODO: Add API documentation.
public protocol PointwiseMultiplicative : AdditiveArithmetic {
  /// The one value.
  ///
  /// One is the identity element for multiplication. For any value,
  /// `x .* .one == x` and `.one .* x == x`.
  static var one: Self { get }

  /// The multiplicative inverse of self.
  ///
  /// For any value, `x .* x.reciprocal == .one` and
  /// `x.reciprocal .* x == .one`.
  var reciprocal: Self { get }

  /// Multiplies two values and produces their product.
  ///
  /// - Parameters:
  ///   - lhs: The first value to multiply.
  ///   - rhs: The second value to multiply.
  static func .*(lhs: Self, rhs: Self) -> Self

  /// Multiplies two values and produces their product.
  ///
  /// - Parameters:
  ///   - lhs: The first value to multiply.
  ///   - rhs: The second value to multiply.
  static func .*=(lhs: inout Self, rhs: Self)
}

public extension PointwiseMultiplicative {
  static func .*=(lhs: inout Self, rhs: Self) {
    lhs = lhs .* rhs
  }
}

public extension PointwiseMultiplicative
  where Self : ExpressibleByIntegerLiteral {
  static var one: Self {
    return 1
  }
}

/// A type that represents an unranked vector space. Values of this type are
/// elements in this vector space and have either no shape or a static shape.
public protocol VectorProtocol : AdditiveArithmetic {
  /// The type of scalars in the vector space.
  associatedtype VectorSpaceScalar : AdditiveArithmetic

  func adding(_ x: VectorSpaceScalar) -> Self

  mutating func add(_ x: VectorSpaceScalar)

  func subtracting(_ x: VectorSpaceScalar) -> Self

  mutating func subtract(_ x: VectorSpaceScalar)

  /// Returns `self` multiplied by the given scalar.
  func scaled(by scalar: VectorSpaceScalar) -> Self

  /// Multiplies `self` by the given scalar.
  mutating func scale(by scalar: VectorSpaceScalar)
}

public extension VectorProtocol {
  mutating func add(_ x: VectorSpaceScalar) {
    self = adding(x)
  }

  mutating func subtract(_ x: VectorSpaceScalar) {
    self = subtracting(x)
  }

  mutating func scale(by scalar: VectorSpaceScalar) {
    self = scaled(by: scalar)
  }
}

/* Note: These default-implemented operators will slow down type-checking
   performance and break existing code.

public extension VectorProtocol {
  static func + (lhs: Self, rhs: VectorSpaceScalar) -> Self {
    lhs.adding(rhs)
  }

  static func + (lhs: VectorSpaceScalar, rhs: Self) -> Self {
    rhs.adding(lhs)
  }

  static func += (lhs: inout Self, rhs: VectorSpaceScalar) {
    lhs.add(rhs)
  }

  static func - (lhs: Self, rhs: VectorSpaceScalar) -> Self {
    lhs.subtracting(rhs)
  }

  static func -= (lhs: inout Self, rhs: VectorSpaceScalar) {
    lhs.subtract(rhs)
  }

  static func * (lhs: Self, rhs: VectorSpaceScalar) -> Self {
    lhs.scaled(by: rhs)
  }

  static func * (lhs: VectorSpaceScalar, rhs: Self) -> Self {
    rhs.scaled(by: lhs)
  }

  static func *= (lhs: inout Self, rhs: VectorSpaceScalar) {
    lhs.scale(by: rhs)
  }
}

public extension VectorProtocol where VectorSpaceScalar : SignedNumeric {
  static func - (lhs: VectorSpaceScalar, rhs: Self) -> Self {
    -rhs.adding(lhs)
  }

  static prefix func - (x: Self) -> Self {
    .zero - x
  }
}
*/

/// A type that mathematically represents a differentiable manifold whose
/// tangent spaces are finite-dimensional.
public protocol Differentiable {
  /// A type representing a differentiable value’s derivatives.
  ///
  /// Mathematically, this is equivalent to the tangent bundle of the
  /// differentiable manifold represented by the differentiable type.
  associatedtype TangentVector: Differentiable & AdditiveArithmetic
    where TangentVector.TangentVector == TangentVector

  /// Moves `self` along the given direction. In Riemannian geometry, this is
  /// equivalent to exponential map, which moves `self` on the geodesic surface
  /// along the given tangent vector.
  mutating func move(along direction: TangentVector)

  /// A tangent vector such that `move(along: zeroTangentVector)` will not
  /// modify `self`.
  /// - Note: `zeroTangentVector` can be `TangentVector.zero` in most cases,
  ///   but types whose tangent vectors depend on instance properties of `self`
  ///   need to provide a different implementation. For example, the tangent
  ///   vector of an `Array` depends on the array’s `count`.
  @available(*, deprecated, message: """
      `zeroTangentVector` derivation has not been implemented; do not use \
      this property
      """)
  var zeroTangentVector: TangentVector { get }

  @available(*, deprecated, message: """
    'AllDifferentiableVariables' is now equal to 'Self' and will be removed
    """)
  typealias AllDifferentiableVariables = Self
}

public extension Differentiable {
  @available(*, deprecated, message: """
    'allDifferentiableVariables' is now equal to 'self' and will be removed
    """)
  var allDifferentiableVariables: AllDifferentiableVariables {
    get { return self }
    set { self = newValue }
  }

  // This is a temporary solution that allows us to add `zeroTangentVector`
  // without implementing derived conformances. This property is marked
  // unavailable because it will produce incorrect results when tangent vectors
  // depend on instance properties of `self`.
  // FIXME: Implement derived conformance and remove this default
  // implementation.
  var zeroTangentVector: TangentVector { .zero }
}

public extension Differentiable where TangentVector == Self {
  mutating func move(along direction: TangentVector) {
    self += direction
  }
}

/// A type that is differentiable in the Euclidean space.
/// The type may represent a vector space, or consist of a vector space and some
/// other non-differentiable component.
///
/// Mathematically, this represents a product manifold that consists of
/// a differentiable vector space and some arbitrary manifold, where the tangent
/// bundle of the entire product manifold is equal to the vector space
/// component.
///
/// This abstraction is useful for representing common differentiable data
/// structures that contain both differentiable vector properties and other
/// stored properties that do not have a derivative, e.g.
///
/// ```swift
/// struct Perceptron: @memberwise EuclideanDifferentiable {
///     var weight: SIMD16<Float>
///     var bias: Float
///     @noDerivative var useBias: Bool
/// }
/// ```
///
/// - Note: Conform a type to `EuclideanDifferentiable` if it is differentiable
///   only with respect to its vector space component and when its
///   `TangentVector` is equal to its vector space component.
public protocol EuclideanDifferentiable: Differentiable {
  /// The differentiable vector component of `self`.
  var differentiableVectorView: TangentVector { get }
}

public extension EuclideanDifferentiable where TangentVector == Self {
  var differentiableVectorView: TangentVector { _read { yield self } }
}

/// Returns `x` like an identity function. When used in a context where `x` is
/// being differentiated with respect to, this function will not produce any 
/// derivative at `x`.
@inlinable
@inline(__always)
@_semantics("autodiff.nonvarying")
public func withoutDerivative<T>(at x: T) -> T {
  x
}

/// Applies the given closure `body` to `x`. When used in a context where `x` is
/// being differentiated with respect to, this function will not produce any
/// derivative at `x`.
// FIXME: Support throws-rethrows.
@inlinable
@inline(__always)
@_semantics("autodiff.nonvarying")
public func withoutDerivative<T, R>(at x: T, in body: (T) -> R) -> R {
  body(x)
}

//===----------------------------------------------------------------------===//
// Functional utilities
//===----------------------------------------------------------------------===//

/// Create a differentiable function from a vector-Jacobian products function.
@inlinable
public func differentiableFunction<T : Differentiable, R : Differentiable>(
  from vjp: @escaping (T)
           -> (value: R, pullback: (R.TangentVector) -> T.TangentVector)
) -> @differentiable (T) -> R {
  func original(_ x: T) -> R {
    return vjp(x).value
  }
  @differentiating(original)
  func derivative(_ x: T)
    -> (value: R, pullback: (R.TangentVector) -> T.TangentVector) {
    return vjp(x)
  }
  return original
}

/// Create a differentiable function from a vector-Jacobian products function.
@inlinable
public func differentiableFunction<T, U, R>(
  from vjp: @escaping (T, U)
           -> (value: R, pullback: (R.TangentVector)
             -> (T.TangentVector, U.TangentVector))
) -> @differentiable (T, U) -> R
  where T : Differentiable, U : Differentiable, R : Differentiable {
  func original(_ x: T, _ y: U) -> R {
    return vjp(x, y).value
  }
  @differentiating(original)
  func derivative(_ x: T, _ y: U)
    -> (value: R,
        pullback: (R.TangentVector)
                    -> (T.TangentVector, U.TangentVector)) {
    return vjp(x, y)
  }
  return original
}

public extension Differentiable {
  @differentiable(wrt: self, vjp: _vjpWithDerivative)
  func withDerivative(_ body: @escaping (inout TangentVector) -> Void) -> Self {
    return self
  }

  @inlinable
  internal func _vjpWithDerivative(
    _ body: @escaping (inout TangentVector) -> Void
  ) -> (Self, (TangentVector) -> TangentVector) {
    return (self, { grad in
      var grad = grad
      body(&grad)
      return grad
    })
  }
}

/// Make a function be recomputed in its pullback, known as "checkpointing" in
/// traditional automatic differentiation.
@inlinable
public func withRecomputationInPullbacks<T, U>(
  _ body: @escaping @differentiable (T) -> U
) -> @differentiable (T) -> U where T : Differentiable, U : Differentiable {
  return differentiableFunction { x in
    (value: body(x), pullback: { v in pullback(at: x, in: body)(v) })
  }
}

public extension Differentiable {
  @inlinable
  @differentiable(wrt: self, vjp: _vjp_withRecomputationInPullbacks)
  func withRecomputationInPullbacks<Result : Differentiable>(
    _ body: @escaping @differentiable (Self) -> Result
  ) -> Result {
    return body(self)
  }

  @inlinable
  internal func _vjp_withRecomputationInPullbacks<Result : Differentiable>(
    _ body: @escaping @differentiable (Self) -> Result
  ) -> (Result, (Result.TangentVector) -> TangentVector) {
    return valueWithPullback(in: Swift.withRecomputationInPullbacks(body))
  }
}

//===----------------------------------------------------------------------===//
// Method-style differential operators
//===----------------------------------------------------------------------===//

public extension Differentiable {
  @inlinable
  func valueWithPullback<R>(
    in f: @differentiable (Self) -> R
  ) -> (value: R, pullback: (R.TangentVector) -> TangentVector) {
    return Builtin.autodiffApply_vjp_arity1(f, self)
  }

  @inlinable
  func pullback<R>(
    in f: @differentiable (Self) -> R
  ) -> (R.TangentVector) -> TangentVector {
    return Builtin.autodiffApply_vjp_arity1(f, self).1
  }

  @inlinable
  func gradient<R>(
    in f: @differentiable (Self) -> R
  ) -> TangentVector
    where R : FloatingPoint, R.TangentVector == R {
    return self.pullback(in: f)(R(1))
  }

  @inlinable
  func valueWithGradient<R>(
    in f: @differentiable (Self) -> R
  ) -> (value: R, gradient: TangentVector)
    where R : FloatingPoint, R.TangentVector == R {
    let (y, pb) = self.valueWithPullback(in: f)
    return (y, pb(R(1)))
  }

  @inlinable
  func valueWithPullback<T, R>(
    at x: T, in f: @differentiable (Self, T) -> R
  ) -> (value: R,
        pullback: (R.TangentVector) -> (TangentVector, T.TangentVector)) {
    return Builtin.autodiffApply_vjp_arity2(f, self, x)
  }

  @inlinable
  func pullback<T, R>(
    at x: T, in f: @differentiable (Self, T) -> R
  ) -> (R.TangentVector) -> (TangentVector, T.TangentVector) {
    return Builtin.autodiffApply_vjp_arity2(f, self, x).1
  }

  @inlinable
  func gradient<T, R>(
    at x: T, in f: @differentiable (Self, T) -> R
  ) -> (TangentVector, T.TangentVector)
    where R : FloatingPoint, R.TangentVector == R {
    return self.pullback(at: x, in: f)(R(1))
  }

  @inlinable
  func valueWithGradient<T, R>(
    at x: T, in f: @differentiable (Self, T) -> R
  ) -> (value: R, gradient: (TangentVector, T.TangentVector))
    where R : FloatingPoint, R.TangentVector == R {
    let (y, pb) = self.valueWithPullback(at: x, in: f)
    return (y, pb(R(1)))
  }
}

//===----------------------------------------------------------------------===//
// Free-function-style differential operators
//===----------------------------------------------------------------------===//

// Transpose

@available(*, unavailable)
@inlinable
public func transpose<T, R>(
  of body: @escaping @differentiable/*(linear)*/ (T) -> R
) -> @differentiable/*(linear)*/ (R) -> T {
  fatalError()
}

// Value with differential

@inlinable
public func valueWithDifferential<T, R>(
  at x: T, in f: @differentiable (T) -> R
) -> (value: R, differential: (T.TangentVector) -> R.TangentVector) {
  return Builtin.autodiffApply_jvp(f, x)
}

@inlinable
public func valueWithDifferential<T, U, R>(
  at x: T, _ y: U, in f: @differentiable (T, U) -> R
) -> (value: R,
      differential: (T.TangentVector, U.TangentVector) -> R.TangentVector) {
  return Builtin.autodiffApply_jvp_arity2(f, x, y)
}

@inlinable
public func valueWithDifferential<T, U, V, R>(
  at x: T, _ y: U, _ z: V, in f: @differentiable (T, U, V) -> R
) -> (value: R,
      differential: (T.TangentVector, U.TangentVector, V.TangentVector)
        -> (R.TangentVector)) {
  return Builtin.autodiffApply_jvp_arity3(f, x, y, z)
}

// Value with pullback

@inlinable
public func valueWithPullback<T, R>(
  at x: T, in f: @differentiable (T) -> R
) -> (value: R, pullback: (R.TangentVector) -> T.TangentVector) {
  return Builtin.autodiffApply_vjp(f, x)
}

@inlinable
public func valueWithPullback<T, U, R>(
  at x: T, _ y: U, in f: @differentiable (T, U) -> R
) -> (value: R,
      pullback: (R.TangentVector) -> (T.TangentVector, U.TangentVector)) {
  return Builtin.autodiffApply_vjp_arity2(f, x, y)
}

@inlinable
public func valueWithPullback<T, U, V, R>(
  at x: T, _ y: U, _ z: V, in f: @differentiable (T, U, V) -> R
) -> (value: R,
      pullback: (R.TangentVector)
        -> (T.TangentVector, U.TangentVector, V.TangentVector)) {
  return Builtin.autodiffApply_vjp_arity3(f, x, y, z)
}

// Differential

@inlinable
public func differential<T, R>(
  at x: T, in f: @differentiable (T) -> R
) -> (T.TangentVector) -> R.TangentVector {
  return valueWithDifferential(at: x, in: f).1
}

@inlinable
public func differential<T, U, R>(
  at x: T, _ y: U, in f: @differentiable (T, U) -> R
) -> (T.TangentVector, U.TangentVector) -> R.TangentVector {
  return valueWithDifferential(at: x, y, in: f).1
}

@inlinable
public func differential<T, U, V, R>(
  at x: T, _ y: U, _ z: V, in f: @differentiable (T, U, V) -> R
) -> (T.TangentVector, U.TangentVector, V.TangentVector) -> (R.TangentVector) {
  return valueWithDifferential(at: x, y, z, in: f).1
}


// Pullback

@inlinable
public func pullback<T, R>(
  at x: T, in f: @differentiable (T) -> R
) -> (R.TangentVector) -> T.TangentVector {
  return Builtin.autodiffApply_vjp(f, x).1
}

@inlinable
public func pullback<T, U, R>(
  at x: T, _ y: U, in f: @differentiable (T, U) -> R
) -> (R.TangentVector) -> (T.TangentVector, U.TangentVector) {
  return Builtin.autodiffApply_vjp_arity2(f, x, y).1
}

@inlinable
public func pullback<T, U, V, R>(
  at x: T, _ y: U, _ z: V, in f: @differentiable (T, U, V) -> R
) -> (R.TangentVector)
    -> (T.TangentVector, U.TangentVector, V.TangentVector) {
  return Builtin.autodiffApply_vjp_arity3(f, x, y, z).1
}

// Derivative

@inlinable
public func derivative<T: FloatingPoint, R>(
  at x: T, in f: @differentiable (T) -> R
) ->  R.TangentVector
  where T.TangentVector == T {
  return differential(at: x, in: f)(T(1))
}

@inlinable
public func derivative<T: FloatingPoint, U: FloatingPoint, R>(
  at x: T, _ y: U, in f: @differentiable (T, U) -> R
) -> R.TangentVector
  where T.TangentVector == T,
        U.TangentVector == U {
  return differential(at: x, y, in: f)(T(1), U(1))
}

@inlinable
public func derivative<T: FloatingPoint, U: FloatingPoint, V: FloatingPoint, R>(
  at x: T, _ y: U, _ z: V, in f: @differentiable (T, U, V) -> R
) -> R.TangentVector
  where T.TangentVector == T,
        U.TangentVector == U,
        V.TangentVector == V {
  return differential(at: x, y, z, in: f)(T(1), U(1), V(1))
}

// Gradient

@inlinable
public func gradient<T, R>(
  at x: T, in f: @differentiable (T) -> R
) -> T.TangentVector
  where R : FloatingPoint, R.TangentVector == R {
  return pullback(at: x, in: f)(R(1))
}

@inlinable
public func gradient<T, U, R>(
  at x: T, _ y: U, in f: @differentiable (T, U) -> R
) -> (T.TangentVector, U.TangentVector)
  where R : FloatingPoint, R.TangentVector == R {
  return pullback(at: x, y, in: f)(R(1))
}

@inlinable
public func gradient<T, U, V, R>(
  at x: T, _ y: U, _ z: V, in f: @differentiable (T, U, V) -> R
) -> (T.TangentVector, U.TangentVector, V.TangentVector)
  where R : FloatingPoint, R.TangentVector == R {
  return pullback(at: x, y, z, in: f)(R(1))
}

// Value with derivative

@inlinable
public func valueWithDerivative<T: FloatingPoint, R>(
  at x: T, in f: @escaping @differentiable (T) -> R
) -> (value: R, derivative: R.TangentVector)
  where T.TangentVector == T {
  let (y, differential) = valueWithDifferential(at: x, in: f)
  return (y, differential(T(1)))
}

@inlinable
public func valueWithDerivative<T: FloatingPoint, U: FloatingPoint, R>(
  at x: T, _ y: U, in f: @escaping @differentiable (T, U) -> R
) -> (value: R, derivative: R.TangentVector)
  where T.TangentVector == T,
        U.TangentVector == U {
  let (y, differential) = valueWithDifferential(at: x, y, in: f)
  return (y, differential(T(1), U(1)))
}

@inlinable
public func valueWithDerivative<
  T: FloatingPoint, U: FloatingPoint, V: FloatingPoint, R>(
  at x: T, _ y: U, _ z: V, in f: @escaping @differentiable (T, U, V) -> R
) -> (value: R, derivative: R.TangentVector)
  where T.TangentVector == T,
        U.TangentVector == U,
        V.TangentVector == V {
  let (y, differential) = valueWithDifferential(at: x, y, z, in: f)
  return (y, differential(T(1), U(1), V(1)))
}

// Value with gradient

@inlinable
public func valueWithGradient<T, R>(
  at x: T, in f: @differentiable (T) -> R
) -> (value: R, gradient: T.TangentVector)
  where R : FloatingPoint, R.TangentVector == R {
  let (y, pullback) = valueWithPullback(at: x, in: f)
  return (y, pullback(R(1)))
}

@inlinable
public func valueWithGradient<T, U, R>(
  at x: T, _ y: U, in f: @differentiable (T, U) -> R
) -> (value: R, gradient: (T.TangentVector, U.TangentVector))
  where R : FloatingPoint, R.TangentVector == R {
  let (y, pullback) = valueWithPullback(at: x, y, in: f)
  return (y, pullback(R(1)))
}

@inlinable
public func valueWithGradient<T, U, V, R>(
  at x: T, _ y: U, _ z: V, in f: @differentiable (T, U, V) -> R
) -> (value: R,
      gradient: (T.TangentVector, U.TangentVector, V.TangentVector))
  where R : FloatingPoint, R.TangentVector == R {
  let (y, pullback) = valueWithPullback(at: x, y, z, in: f)
  return (y, pullback(R(1)))
}

// Derivative (curried)

@inlinable 
public func derivative<T: FloatingPoint, R>(
  of f: @escaping @differentiable (T) -> R
) -> (T) -> R.TangentVector
  where T.TangentVector == T {
  return { x in derivative(at: x, in: f) }
}

@inlinable 
public func derivative<T: FloatingPoint, U: FloatingPoint, R>(
  of f: @escaping @differentiable (T, U) -> R
) -> (T, U) -> R.TangentVector
  where T.TangentVector == T,
        U.TangentVector == U {
  return { (x, y) in derivative(at: x, y, in: f) }
}

@inlinable
public func derivative<T: FloatingPoint, U: FloatingPoint, V: FloatingPoint, R>(
  of f: @escaping @differentiable (T, U, V) -> R
) -> (T, U, V) -> R.TangentVector
  where T.TangentVector == T,
        U.TangentVector == U,
        V.TangentVector == V {
  return { (x, y, z) in derivative(at: x, y, z, in: f) }
}

// Gradient (curried)

@inlinable
public func gradient<T, R>(
  of f: @escaping @differentiable (T) -> R
) -> (T) -> T.TangentVector
  where R : FloatingPoint, R.TangentVector == R {
  return { x in gradient(at: x, in: f) }
}

@inlinable
public func gradient<T, U, R>(
  of f: @escaping @differentiable (T, U) -> R
) -> (T, U) -> (T.TangentVector, U.TangentVector)
  where R : FloatingPoint, R.TangentVector == R {
  return { x, y in gradient(at: x, y, in: f) }
}

@inlinable
public func gradient<T, U, V, R>(
  of f: @escaping @differentiable (T, U, V) -> R
) -> (T, U, V) -> (T.TangentVector, U.TangentVector, V.TangentVector)
  where R : FloatingPoint, R.TangentVector == R {
  return { x, y, z in gradient(at: x, y, z, in: f) }
}

// Value with derivative (curried)

@inlinable
public func valueWithDerivative<T: FloatingPoint, R>(
  of f: @escaping @differentiable (T) -> R
) -> (T) -> (value: R, derivative: R.TangentVector)
  where T.TangentVector == T {
  return { x in valueWithDerivative(at: x, in: f) }
}

@inlinable
public func valueWithDerivative<T: FloatingPoint, U: FloatingPoint, R>(
  of f: @escaping @differentiable (T, U) -> R
) -> (T, U) -> (value: R, derivative: R.TangentVector)
  where T.TangentVector == T,
        U.TangentVector == U {
  return { (x, y) in valueWithDerivative(at: x, y, in: f) }
}

@inlinable
public func valueWithDerivative<
  T: FloatingPoint, U: FloatingPoint, V: FloatingPoint, R>(
  of f: @escaping @differentiable (T, U, V) -> R
) -> (T, U, V) -> (value: R, derivative: R.TangentVector)
  where T.TangentVector == T,
        U.TangentVector == U,
        V.TangentVector == V {
  return { (x, y, z) in valueWithDerivative(at: x, y, z, in: f) }
}

// Value with gradient (curried)

@inlinable
public func valueWithGradient<T, R>(
  of f: @escaping @differentiable (T) -> R
) -> (T) -> (value: R, gradient: T.TangentVector)
  where R : FloatingPoint, R.TangentVector == R {
  return { x in valueWithGradient(at: x, in: f) }
}

@inlinable
public func valueWithGradient<T, U, R>(
  of f: @escaping @differentiable (T, U) -> R
) -> (T, U) -> (value: R, gradient: (T.TangentVector, U.TangentVector))
  where R : FloatingPoint, R.TangentVector == R {
  return { x, y in valueWithGradient(at: x, y, in: f) }
}

@inlinable
public func valueWithGradient<T, U, V, R>(
  of f: @escaping @differentiable (T, U, V) -> R
) -> (T, U, V)
  -> (value: R,
      gradient: (T.TangentVector, U.TangentVector, V.TangentVector))
  where R : FloatingPoint, R.TangentVector == R {
  return { x, y, z in valueWithGradient(at: x, y, z, in: f) }
}

//===----------------------------------------------------------------------===//
// Type-erased `AnyDerivative`
//===----------------------------------------------------------------------===//

internal protocol _AnyDerivativeBox {
  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  func _isEqual(to other: _AnyDerivativeBox) -> Bool
  func _isNotEqual(to other: _AnyDerivativeBox) -> Bool

  // `AdditiveArithmetic` requirements.
  static var _zero: _AnyDerivativeBox { get }
  func _adding(_ x: _AnyDerivativeBox) -> _AnyDerivativeBox
  func _subtracting(_ x: _AnyDerivativeBox) -> _AnyDerivativeBox

  // `Differentiable` requirements.
  mutating func _move(along direction: _AnyDerivativeBox)

  // `EuclideanDifferentiable` requirements.
  var _differentiableVectorView: _AnyDerivativeBox { get }

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any { get }

  /// Returns the underlying value unboxed to the given type, if possible.
  func _unboxed<U>(to type: U.Type) -> U?
    where U : Differentiable, U.TangentVector == U
}

extension _AnyDerivativeBox {
  /// Returns true if the underlying value has type `AnyDerivative.OpaqueZero`.
  func _isOpaqueZero() -> Bool {
    return _unboxed(to: AnyDerivative.OpaqueZero.self) != nil
  }
}

@inline(never)
@usableFromInline
internal func _derivativeTypeMismatch(
  _ x: Any.Type, _ y: Any.Type, file: StaticString = #file, line: UInt = #line
) -> Never {
  preconditionFailure("""
    Derivative type mismatch: \
    \(String(reflecting: x)) and \(String(reflecting: y))
    """, file: file, line: line)
}

internal struct _ConcreteDerivativeBox<T> : _AnyDerivativeBox
  where T : Differentiable, T.TangentVector == T
{
  /// The underlying base value.
  var _base: T

  init(_ base: T) {
    self._base = base
  }

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any {
    return _base
  }

  func _unboxed<U>(to type: U.Type) -> U?
    where U : Differentiable, U.TangentVector == U
  {
    return (self as? _ConcreteDerivativeBox<U>)?._base
  }

  // `Equatable` requirements (implied by `AdditiveArithmetic`).

  func _isEqual(to other: _AnyDerivativeBox) -> Bool {
    return _base == other._unboxed(to: T.self)
  }

  func _isNotEqual(to other: _AnyDerivativeBox) -> Bool {
    return _base != other._unboxed(to: T.self)
  }

  // `AdditiveArithmetic` requirements.

  static var _zero: _AnyDerivativeBox {
    return _ConcreteDerivativeBox(T.zero)
  }

  func _adding(_ x: _AnyDerivativeBox) -> _AnyDerivativeBox {
    // 0 + x = x
    if _isOpaqueZero() {
      return x
    }
    // y + 0 = y
    if x._isOpaqueZero() {
      return self
    }
    guard let xBase = x._unboxed(to: T.self) else {
      _derivativeTypeMismatch(T.self, type(of: x._typeErasedBase))
    }
    return _ConcreteDerivativeBox(_base + xBase)
  }

  func _subtracting(_ x: _AnyDerivativeBox) -> _AnyDerivativeBox {
    // y - 0 = y
    if x._isOpaqueZero() {
      return self
    }
    // 0 - x = -x
    if _isOpaqueZero() {
      return type(of: x)._zero._subtracting(x)
    }
    guard let xBase = x._unboxed(to: T.self) else {
      _derivativeTypeMismatch(T.self, type(of: x._typeErasedBase))
    }
    return _ConcreteDerivativeBox(_base - xBase)
  }

  // `Differentiable` requirements.

  mutating func _move(along direction: _AnyDerivativeBox) {
    if direction._isOpaqueZero() {
      return
    }
    // The case where `self._isOpaqueZero()` returns true is handled in
    // `AnyDerivative.move(along:)`.
    guard let directionBase =
      direction._unboxed(to: T.TangentVector.self) else {
      _derivativeTypeMismatch(T.self, type(of: direction._typeErasedBase))
    }
    _base.move(along: directionBase)
  }

  // `EuclideanDifferentiable` requirements.
  var _differentiableVectorView: _AnyDerivativeBox {
    return self
  }
}

/// A type-erased derivative value.
///
/// The `AnyDerivative` type forwards its operations to an arbitrary underlying
/// base derivative value conforming to `Differentiable` and
/// `AdditiveArithmetic`, hiding the specifics of the underlying value.
public struct AnyDerivative : EuclideanDifferentiable & AdditiveArithmetic {
  internal var _box: _AnyDerivativeBox

  internal init(_box: _AnyDerivativeBox) {
    self._box = _box
  }

  /// The underlying base value.
  public var base: Any {
    return _box._typeErasedBase
  }

  /// Creates a type-erased derivative from the given derivative.
  @differentiable(vjp: _vjpInit(_:))
  public init<T>(_ base: T) where T : Differentiable, T.TangentVector == T {
    self._box = _ConcreteDerivativeBox<T>(base)
  }

  @usableFromInline internal static func _vjpInit<T>(
    _ base: T
  ) -> (AnyDerivative, (AnyDerivative) -> T.TangentVector)
    where T : Differentiable, T.TangentVector == T
  {
    return (AnyDerivative(base), { v in v.base as! T.TangentVector })
  }

  public typealias TangentVector = AnyDerivative

  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  public static func == (lhs: AnyDerivative, rhs: AnyDerivative) -> Bool {
    return lhs._box._isEqual(to: rhs._box)
  }
  public static func != (lhs: AnyDerivative, rhs: AnyDerivative) -> Bool {
    return lhs._box._isNotEqual(to: rhs._box)
  }

  // `AdditiveArithmetic` requirements.

  /// Internal struct representing an opaque zero value.
  @frozen
  @usableFromInline
  internal struct OpaqueZero : EuclideanDifferentiable & AdditiveArithmetic {}

  public static var zero: AnyDerivative {
    return AnyDerivative(
      _box: _ConcreteDerivativeBox<OpaqueZero>(OpaqueZero.zero))
  }

  public static func + (
    lhs: AnyDerivative, rhs: AnyDerivative
  ) -> AnyDerivative {
    return AnyDerivative(_box: lhs._box._adding(rhs._box))
  }

  @differentiating(+)
  @usableFromInline internal static func _vjpAdd(
    lhs: AnyDerivative, rhs: AnyDerivative
  ) -> (value: AnyDerivative,
        pullback: (AnyDerivative) -> (AnyDerivative, AnyDerivative)) {
    return (lhs + rhs, { v in (v, v) })
  }

  public static func - (
    lhs: AnyDerivative, rhs: AnyDerivative
  ) -> AnyDerivative {
    return AnyDerivative(_box: lhs._box._subtracting(rhs._box))
  }

  @differentiating(-)
  @usableFromInline internal static func _vjpSubtract(
    lhs: AnyDerivative, rhs: AnyDerivative
  ) -> (value: AnyDerivative,
        pullback: (AnyDerivative) -> (AnyDerivative, AnyDerivative)) {
    return (lhs - rhs, { v in (v, .zero - v) })
  }

  // `Differentiable` requirements.
  public mutating func move(along direction: TangentVector) {
    if _box._isOpaqueZero() {
      _box = direction._box
      return
    }
    _box._move(along: direction._box)
  }

  // `EuclideanDifferentiable` requirements.
  public var differentiableVectorView: TangentVector {
    return self
  }
}

//===----------------------------------------------------------------------===//
// Differentiable higher order functions for collections
//===----------------------------------------------------------------------===//

public extension Array where Element: Differentiable {
  @differentiable(wrt: (self, initialResult), vjp: _vjpDifferentiableReduce)
  func differentiableReduce<Result: Differentiable>(
    _ initialResult: Result,
    _ nextPartialResult: @differentiable (Result, Element) -> Result
  ) -> Result {
    reduce(initialResult, nextPartialResult)
  }

  @usableFromInline
  internal func _vjpDifferentiableReduce<Result: Differentiable>(
    _ initialResult: Result,
    _ nextPartialResult: @differentiable (Result, Element) -> Result
  ) -> (value: Result,
        pullback: (Result.TangentVector)
          -> (Array.TangentVector, Result.TangentVector)) {
    var pullbacks:
      [(Result.TangentVector) -> (Result.TangentVector, Element.TangentVector)]
        = []
    let count = self.count
    pullbacks.reserveCapacity(count)
    var result = initialResult
    for element in self {
      let (y, pb) =
        Swift.valueWithPullback(at: result, element, in: nextPartialResult)
      result = y
      pullbacks.append(pb)
    }
    return (value: result, pullback: { tangent in
      var resultTangent = tangent
      var elementTangents = TangentVector([])
      elementTangents.base.reserveCapacity(count)
      for pullback in pullbacks.reversed() {
        let (newResultTangent, elementTangent) = pullback(resultTangent)
        resultTangent = newResultTangent
        elementTangents.base.append(elementTangent)
      }
      return (TangentVector(elementTangents.base.reversed()), resultTangent)
    })
  }
}

public extension Array where Element: Differentiable {
  @differentiable(wrt: self, vjp: _vjpDifferentiableMap)
  func differentiableMap<Result: Differentiable>(
    _ body: @differentiable (Element) -> Result
  ) -> [Result] {
    map(body)
  }

  @usableFromInline
  internal func _vjpDifferentiableMap<Result: Differentiable>(
    _ body: @differentiable (Element) -> Result
  ) -> (value: [Result],
        pullback: (Array<Result>.TangentVector) -> Array.TangentVector) {
    var values: [Result] = []
    var pullbacks: [(Result.TangentVector) -> Element.TangentVector] = []
    for x in self {
      let (y, pb) = Swift.valueWithPullback(at: x, in: body)
      values.append(y)
      pullbacks.append(pb)
    }
    func pullback(_ tans: Array<Result>.TangentVector) -> Array.TangentVector {
      .init(zip(tans.base, pullbacks).map { tan, pb in pb(tan) })
    }
    return (value: values, pullback: pullback)
  }
}

//===----------------------------------------------------------------------===//
// JVP Diagnostics
//===----------------------------------------------------------------------===//
@_silgen_name("_printJVPErrorAndExit")
public func _printJVPErrorAndExit() -> Never {
    fatalError("""
        JVP does not exist. Differential-first differentiation APIs are \
        experimental and should not be used.
        """)
}
