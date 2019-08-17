// RUN: %target-run-simple-swift
// REQUIRES: executable_test
import StdlibUnittest
#if os(macOS)
import Darwin.C
#else
import Glibc
#endif

var OptionalTests = TestSuite("Optional")

typealias OptionalGrad = Optional<Float>.TangentVector

extension Optional where Wrapped: Differentiable {
    @differentiable(wrt: self, vjp: _vjpUnWrap)
    func forceUnwrapped() -> Wrapped {
        return self.unsafelyUnwrapped2
    }
    @inlinable
    public var unsafelyUnwrapped2: Wrapped {
        @inline(__always)
        get {
            if let x = self {
                return x
            }
            fatalError("Move to move `.none`?")
        }
    }
    

    @usableFromInline
    func _vjpUnWrap() ->
      (Wrapped/* TangentVector */, (Wrapped.TangentVector) -> Optional<Wrapped>.TangentVector) {
        return (self!, { Optional<Wrapped>.TangentVector(value: .some($0)) })
    }
    

}

OptionalTests.test("Optional DifferentialView Math") {
    let twoGrad = OptionalGrad(value: 2)
    let zeroGrad = OptionalGrad.zero
    expectEqual(zeroGrad,twoGrad-twoGrad)
    let fourGrad=OptionalGrad(value: 4)
    expectEqual(twoGrad+twoGrad,fourGrad)
    let noneGrad = OptionalGrad(value: Optional<Float>.none)
    expectEqual(noneGrad+noneGrad,noneGrad)
    expectEqual(noneGrad+twoGrad,twoGrad)
    expectEqual(twoGrad+noneGrad,twoGrad)
    expectEqual(noneGrad-noneGrad,noneGrad)
    expectEqual(twoGrad-noneGrad,twoGrad)
    let negTwoGrad = OptionalGrad(value: -2)
    expectEqual(noneGrad-twoGrad,negTwoGrad)
}

OptionalTests.test("Optional Dense") {
    struct Square : Differentiable {
        @differentiable
        func callAsFunction(_ x: Float) -> Float {
          return x * x
        }
    }
    struct optFunc : Differentiable {
        @noDerivative var f1: Square?
        init(f1 fopt: Square ) {
            f1=Optional.some(fopt)
        }
        @differentiable
        func callAsFunction(_ x: Float) -> Float {
            if let _ = f1 {
                let f = f1.forceUnwrapped()
                return f(x)
            }
            return x
        }
    }
    //let four:Optional<Float>.TangentVector = Optional<Float>.TangentVector(value: 4.0)
    let grad = optFunc(f1: Square()).gradient(at: 4, in: { dense, x in dense(x) })
    expectEqual((optFunc.TangentVector(),8.0),grad)
}

runAllTests()
