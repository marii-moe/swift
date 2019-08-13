// RUN: %target-swift-frontend -emit-sil -verify %s | %FileCheck %s -check-prefix=CHECK-SIL

@_silgen_name("identity")
func identity<T : Differentiable>(_ x: T) -> T {
  return x
}
_ = gradient(at: Float(1), in: { x in identity(x) })

// Test AdjointEmitter local buffer allocation.
// Verify that local buffers are immediately set to zero.

// CHECK-SIL-LABEL: // pullback wrt 0 source 0 for identity
// CHECK-SIL-NEXT: sil hidden @$s8identityTUp0r0
// CHECK-SIL:      [[ORIG_COTAN:%.*]] = alloc_stack $τ_0_0.TangentVector
// CHECK-SIL-NEXT: [[ZERO_WITNESS:%.*]] = witness_method $τ_0_0.TangentVector, #AdditiveArithmetic.zero!getter.1
// CHECK-SIL-NEXT: [[ORIG_COTAN_METATYPE:%.*]] = metatype $@thick τ_0_0.TangentVector.Type
// CHECK-SIL-NEXT: [[EMIT_ZERO_INDIRECT:%.*]] = apply [[ZERO_WITNESS]]<τ_0_0.TangentVector>([[ORIG_COTAN]], [[ORIG_COTAN_METATYPE]])
// CHECK-SIL: }

struct Tensor<Scalar : FloatingPoint & Differentiable> : VectorProtocol, Differentiable {
  // NOTE: `value` must have type with known size (e.g. `Float`, not `Scalar`)
  // until differentiation has indirect passing support.
  var value: Float
  init(_ value: Float) { self.value = value }
}

func generic<T : FloatingPoint & Differentiable>(_ x: Tensor<T>) -> Float {
  return x.value + x.value
}
_ = gradient(at: Tensor<Float>(1), in: generic)

// Test case where associated derivative function's requirements are met.
extension Tensor where Scalar : Numeric {
  @differentiable(wrt: self where Scalar : Differentiable & FloatingPoint)
  func mean() -> Tensor {
    return self
  }

  @differentiable(wrt: self where Scalar : Differentiable & FloatingPoint)
  func variance() -> Tensor {
    return mean() // ok
  }
}
_ = pullback(at: Tensor<Float>(1), in: { $0.variance() })

// Tests TF-277.
protocol Layer : Differentiable {
  associatedtype Output : Differentiable
}
struct SupervisedTrainer<Model : Layer> {
  var model: Model
  var lossFunction: @differentiable (Model.Output, Model.Output) -> Float
  func fit(y: Model.Output) {
    _ = gradient(at: y) { y in return self.lossFunction(y, y) }
  }
}

// Tests TF-440.
struct TF_440_Input<Input: Differentiable, State: Differentiable>
  : Differentiable {
  var input: Input
  var state: State
}
struct TF_440<T : Differentiable> {
  @differentiable
  func applied(to input: TF_440_Input<Float, Float>) -> Float {
    return input.state
  }

  @differentiable
  func applied(to input: TF_440_Input<T, Float>) -> Float {
    return input.state
  }

  @differentiable
  func applied(to input: TF_440_Input<T, Float>) -> T {
    return input.input
  }
}

// Tests TF-508: differentiation requirements with dependent member types.
protocol TF_508_Proto {
  associatedtype Scalar
}
extension TF_508_Proto where Scalar : FloatingPoint {
  @differentiable(
    vjp: vjpAdd
    where Self : Differentiable, Scalar : Differentiable,
          // Conformance requirement with dependent member type.
          Self.TangentVector : TF_508_Proto
  )
  static func +(lhs: Self, rhs: Self) -> Self {
    return lhs
  }

  @differentiable(
    vjp: vjpSubtract
    where Self : Differentiable, Scalar : Differentiable,
          // Same-type requirement with dependent member type.
          Self.TangentVector == Float
  )
  static func -(lhs: Self, rhs: Self) -> Self {
    return lhs
  }
}
extension TF_508_Proto where Self : Differentiable,
                             Scalar : FloatingPoint & Differentiable,
                             Self.TangentVector : TF_508_Proto {
  static func vjpAdd(lhs: Self, rhs: Self)
      -> (Self, (TangentVector) -> (TangentVector, TangentVector)) {
    return (lhs, { v in (v, v) })
  }
}
extension TF_508_Proto where Self : Differentiable,
                             Scalar : FloatingPoint & Differentiable,
                             Self.TangentVector == Float {
  static func vjpSubtract(lhs: Self, rhs: Self)
      -> (Self, (TangentVector) -> (TangentVector, TangentVector)) {
    return (lhs, { v in (v, v) })
  }
}

struct TF_508_Struct<Scalar : AdditiveArithmetic>
  : TF_508_Proto, AdditiveArithmetic {}
extension TF_508_Struct : Differentiable where Scalar : Differentiable {
  typealias TangentVector = TF_508_Struct
}

func TF_508() {
  let x = TF_508_Struct<Float>()
  // Test conformance requirement with dependent member type.
  _ = pullback(at: x, in: { (x: TF_508_Struct<Float>) -> TF_508_Struct<Float> in
    return x + x
  })
  // Test same-type requirement with dependent member type.
  _ = pullback(at: x, in: { (x: TF_508_Struct<Float>) -> TF_508_Struct<Float> in
    return x - x
  })
}

// TF-523
struct TF_523_Struct : Differentiable & AdditiveArithmetic {
  var a: Float = 1
  typealias TangentVector = TF_523_Struct
}

@differentiable
func TF_523_f(_ x: TF_523_Struct) -> Float {
  return x.a * 2
}

// TF-534: Thunk substitution map remapping.
protocol TF_534_Layer : Differentiable {
  associatedtype Input : Differentiable
  associatedtype Output : Differentiable

  @differentiable
  func callAsFunction(_ input: Input) -> Output
}
struct TF_534_Tensor<Scalar> : Differentiable {}

func TF_534<Model: TF_534_Layer>(
  _ model: inout Model, inputs: Model.Input
) -> TF_534_Tensor<Float> where Model.Output == TF_534_Tensor<Float> {
  return valueWithPullback(at: model) { model -> Model.Output in
    return model(inputs)
  }.0
}

// TF-546: Test that SILGen linear map thunk performs correct reabstraction.
struct TF_546<T: FloatingPoint>: AdditiveArithmetic {
  var real: T
  var imaginary: T

  @differentiable(vjp: _vjpInit where T: Differentiable, T == T.TangentVector)
  init(real: T = 0, imaginary: T = 0) {
    self.real = real
    self.imaginary = imaginary
  }
}
extension TF_546: Differentiable where T: Differentiable {
  typealias TangentVector = TF_546
}
extension TF_546 where T: Differentiable, T == T.TangentVector {
  static func _vjpInit(real: T, imaginary: T) -> (TF_546, (TF_546) -> (T, T)) {
    return (TF_546(real: real, imaginary: imaginary), { ($0.real, $0.imaginary) })
  }
}
let _: @differentiable(Float, Float) -> TF_546<Float> = { r, i in
  TF_546(real: r, imaginary: i)
}

// TF-652: Test VJPEmitter substitution map generic signature.
// The substitution map should have the VJP's generic signature, not the
// original function's.
struct TF_652<Scalar> {}
extension TF_652 : Differentiable where Scalar : FloatingPoint {}

@differentiable(wrt: x where Scalar: FloatingPoint)
func test<Scalar: Numeric>(x: TF_652<Scalar>) -> TF_652<Scalar> {
  for _ in 0..<10 {
    let _ = x
  }
  return x
}

// TF-682: Test that SILGen linear map thunk performs correct reabstraction.
protocol TF_682_Proto {
  associatedtype Scalar
}
extension TF_682_Proto where Scalar : FloatingPoint {
  @differentiable(
    vjp: vjpFoo
    where Self : Differentiable, Scalar : Differentiable,
          // Same-type requirement with dependent member type.
          Self.TangentVector == Float
  )
  func foo(lhs: Self) -> Self {
    return lhs
  }
}
extension TF_682_Proto where Self : Differentiable,
                             Scalar : FloatingPoint & Differentiable,
                             Self.TangentVector == Float {
  func vjpFoo(lhs: Self)
      -> (Self, (TangentVector) -> (TangentVector, TangentVector)) {
    return (lhs, { v in (v, v) })
  }
}

// TF-688: Test generic curry thunk cloning.
public struct TF_688_Struct<Scalar> {
  var x: Scalar
}
extension TF_688_Struct: Differentiable where Scalar: Differentiable {
  @differentiable
  public static func id(x: Self) -> Self {
    return x
  }
}
@differentiable(wrt: x)
public func TF_688<Scalar: Differentiable>(
  _ x: TF_688_Struct<Scalar>,
  reduction: @differentiable (TF_688_Struct<Scalar>) -> TF_688_Struct<Scalar> = TF_688_Struct.id
) -> TF_688_Struct<Scalar> {
  reduction(x)
}

// TF-697: Test generic requirements of generated AD associated function.
protocol TF_697_Module: Differentiable {
    associatedtype Input
    associatedtype Output: Differentiable

    @differentiable(wrt: self)
    func callModule(_ input: Input) -> Output
}
protocol TF_697_Layer: TF_697_Module where Input: Differentiable {
    @differentiable
    func callLayer(_ input: Input) -> Output
}
struct TF_697_Sequential<Layer1: TF_697_Module, Layer2: TF_697_Layer>: TF_697_Module
    where Layer1.Output == Layer2.Input {
    var layer1: Layer1
    var layer2: Layer2

    @differentiable(wrt: self)
    func callModule(_ input: Layer1.Input) -> Layer2.Output {
        layer2.callLayer(layer1.callModule(input))
    }
}
extension TF_697_Sequential: TF_697_Layer where Layer1: TF_697_Layer {
    @differentiable
    func callLayer(_ input: Layer1.Input) -> Layer2.Output {
        layer2.callLayer(layer1.callLayer(input))
    }
}

// TODO: add more tests.
