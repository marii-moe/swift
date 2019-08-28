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

OptionalTests.test("Optional Dense") {
    struct optFunc : Differentiable {
        /*@noDerivative*/ var f1: Float?
        init(f1 f: Float ) {
            f1=Optional.some(f)
        }
        //@differentiable
        func callAsFunction(_ x: Float) -> Float {
            if f1 != nil {
                let f = f1!
                return f*x
            }
            return x
        }
    }
    let opt = optFunc(f1: 2.0)
    let grad = opt.gradient(at: 4, in: { dense, x in dense(x) })
    //let backprop = opt.pullback(at: 100, in: { dense, x in dense(x) })
    expectEqual(grad,(optFunc.TangentVector(f1: OptionalGrad(value: 1.0) ), 2.0))
    //expectEqual(backprop(-2.0),(optFunc.TangentVector(f1: OptionalGrad(value: 1.0) ),-4.0))
}

runAllTests()
