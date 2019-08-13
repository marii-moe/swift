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

OptionalTests.test("Optional DifferentialView Math") {
    let twoGrad = OptionalGrad(2)
    let zeroGrad = OptionalGrad.zero
    expectEqual(zeroGrad,twoGrad-twoGrad)
    let fourGrad=OptionalGrad(4)
    expectEqual(twoGrad+twoGrad,fourGrad)
    let noneGrad = OptionalGrad(Optional<Float>.none)
    expectEqual(noneGrad+noneGrad,noneGrad)
    expectEqual(noneGrad+twoGrad,twoGrad)
    expectEqual(twoGrad+noneGrad,twoGrad)
    expectEqual(noneGrad-noneGrad,noneGrad)
    expectEqual(twoGrad-noneGrad,twoGrad)
    let negTwoGrad = OptionalGrad(-2)
    expectEqual(noneGrad-twoGrad,negTwoGrad)
}

OptionalTests.test("Optional identity") {
    func optionalIdentity(_ x: Optional<Float>) -> Optional<Float> {
        return x
    }
    let one:Optional<Float> = Float("1")
    let _ = pullback(at: one, in: optionalIdentity)
}

runAllTests()
