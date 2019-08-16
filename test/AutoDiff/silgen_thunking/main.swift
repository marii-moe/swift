// RUN: %target-swift-frontend -emit-silgen -verify %s %S/../Inputs/silgen_thunking_other_module.swift | %FileCheck %s

// RUN: %empty-directory(%t)
// RUN: %target-build-swift %S/../Inputs/silgen_thunking_other_module.swift %s -o %t/a.out
// RUN: %target-codesign %t/a.out
// RUN: %target-run %t/a.out

// REQUIRES: executable_test

import StdlibUnittest
import DifferentiationUnittest

var DerivativeSILGenThunkTests = TestSuite("DerivativeSILGenThunks")

// TF-619: Test cross-module import of `@differentiable` methods with
// self-ordering thunks.
DerivativeSILGenThunkTests.testWithLeakChecking("CrossModuleMethodSelfReorderingThunk") {
  expectEqual(1, gradient(at: 0) { x in TF_619().foo(x) })
}

// TF-698, TF-742: Test thunks that perform self-ordering but not reabstraction.
struct SelfReordering : Differentiable & AdditiveArithmetic {
  var x: Tracked<Float>
  init(_ x: Tracked<Float>) {
    self.x = x
  }

  // TF-742: Test method with three parameters (including `self`).
  // Note: pullback returns direct `Self.TangentVector`.
  @differentiable(jvp: jvpThreeParameterMethod, vjp: vjpThreeParameterMethod)
  func threeParameterMethod(_: Self, _: Self) -> Self {
    return self
  }
  func jvpThreeParameterMethod(_ x: Self, _ y: Self) -> (Self, (Self, Self, Self) -> Self) {
    let value = threeParameterMethod(x, y)
    return (value, { dself, dx, dy in Self(dself.x + dx.x * 2 + dy.x * 3) })
  }
  func vjpThreeParameterMethod(_ x: Self, _ y: Self) -> (Self, (Self) -> (Self, Self, Self)) {
    let value = threeParameterMethod(x, y)
    return (value, { v in (Self(1), Self(2), Self(3)) })
  }

// CHECK-LABEL: sil hidden @AD__$s4main14SelfReorderingV20threeParameterMethodyA2C_ACtF__jvp_src_0_wrt_0_1_2 : $@convention(method) (@guaranteed SelfReordering, @guaranteed SelfReordering, @guaranteed SelfReordering) -> (@owned SelfReordering, @owned @callee_guaranteed (@guaranteed SelfReordering, @guaranteed SelfReordering, @guaranteed SelfReordering) -> @owned SelfReordering)
// CHECK: bb0([[X:%.*]] : $SelfReordering, [[Y:%.*]] : $SelfReordering, [[SELF:%.*]] : $SelfReordering):
// CHECK: [[JVP:%.*]] = function_ref @$s4main14SelfReorderingV23jvpThreeParameterMethodyAC_A2C_A2CtctAC_ACtF
// CHECK: [[JVP_RESULT:%.*]] = apply [[JVP]]([[X]], [[Y]], [[SELF]])
// CHECK: [[JVP_ORIG_RESULT:%.*]] = tuple_extract [[JVP_RESULT]] : {{.*}}, 0
// CHECK: [[DF:%.*]] = tuple_extract [[JVP_RESULT]] : {{.*}}, 1
// CHECK: [[DF_SELF_REORDER_THUNK:%.*]] = function_ref @AD__$s4main14SelfReorderingVA3CIeggggo_A4CIeggggo_TR_differential_self_reordering_thunk
// CHECK: [[THUNKED_DF:%.*]] = partial_apply [callee_guaranteed] [[DF_SELF_REORDER_THUNK]]([[DF]])
// CHECK: [[RESULT:%.*]] = tuple ([[JVP_ORIG_RESULT]] : $SelfReordering, [[THUNKED_DF]] : {{.*}})
// CHECK: return [[RESULT]]

// CHECK-LABEL: sil shared [transparent] [serialized] [reabstraction_thunk] [ossa] @AD__$s4main14SelfReorderingVA3CIeggggo_A4CIeggggo_TR_differential_self_reordering_thunk : $@convention(thin) (@guaranteed SelfReordering, @guaranteed SelfReordering, @guaranteed SelfReordering, @guaranteed @callee_guaranteed (@guaranteed SelfReordering, @guaranteed SelfReordering, @guaranteed SelfReordering) -> @owned SelfReordering) -> @owned SelfReordering
// CHECK: bb0([[DX:%.*]] : @guaranteed $SelfReordering, [[DY:%.*]] : @guaranteed $SelfReordering, [[DSELF:%.*]] : @guaranteed $SelfReordering, [[DF:%.*]] : @guaranteed $@callee_guaranteed (@guaranteed SelfReordering, @guaranteed SelfReordering, @guaranteed SelfReordering) -> @owned SelfReordering)
// CHECK: [[DF_RESULT:%.*]] = apply [[DF]]([[DSELF]], [[DX]], [[DY]])
// CHECK: return [[DF_RESULT]]

// CHECK-LABEL: sil hidden @AD__$s4main14SelfReorderingV20threeParameterMethodyA2C_ACtF__vjp_src_0_wrt_0_1_2 : $@convention(method) (@guaranteed SelfReordering, @guaranteed SelfReordering, @guaranteed SelfReordering) -> (@owned SelfReordering, @owned @callee_guaranteed (@guaranteed SelfReordering) -> (@owned SelfReordering, @owned SelfReordering, @owned SelfReordering))
// CHECK: bb0([[X:%.*]] : $SelfReordering, [[Y:%.*]] : $SelfReordering, [[SELF:%.*]] : $SelfReordering):
// CHECK: [[VJP:%.*]] = function_ref @$s4main14SelfReorderingV23vjpThreeParameterMethodyAC_AC_A2CtACctAC_ACtF
// CHECK: [[VJP_RESULT:%.*]] = apply [[VJP]]([[X]], [[Y]], [[SELF]])
// CHECK: [[VJP_ORIG_RESULT:%.*]] = tuple_extract [[VJP_RESULT]] : {{.*}}, 0
// CHECK: [[PB:%.*]] = tuple_extract [[VJP_RESULT]] : {{.*}}, 1
// CHECK: [[PB_SELF_REORDER_THUNK:%.*]] = function_ref @AD__$s4main14SelfReorderingVA3CIeggooo_A4CIeggooo_TR_pullback_self_reordering_thunk
// CHECK: [[THUNKED_PB:%.*]] = partial_apply [callee_guaranteed] [[PB_SELF_REORDER_THUNK]]([[PB]])
// CHECK: [[RESULT:%.*]] = tuple ([[VJP_ORIG_RESULT]] : $SelfReordering, [[THUNKED_PB]] : {{.*}})
// CHECK: return [[RESULT]]

// CHECK-LABEL: sil shared [transparent] [serialized] [reabstraction_thunk] [ossa] @AD__$s4main14SelfReorderingVA3CIeggooo_A4CIeggooo_TR_pullback_self_reordering_thunk : $@convention(thin) (@guaranteed SelfReordering, @guaranteed @callee_guaranteed (@guaranteed SelfReordering) -> (@owned SelfReordering, @owned SelfReordering, @owned SelfReordering)) -> (@owned SelfReordering, @owned SelfReordering, @owned SelfReordering)
// CHECK: bb0([[SEED:%.*]] : @guaranteed $SelfReordering, [[PB:%.*]] : @guaranteed $@callee_guaranteed (@guaranteed SelfReordering) -> (@owned SelfReordering, @owned SelfReordering, @owned SelfReordering)):
// CHECK: [[PB_RESULT:%.*]] = apply [[PB]]([[SEED]])
// CHECK: ([[SELF_ADJ:%.*]], [[X_ADJ:%.*]], [[Y_ADJ:%.*]]) = destructure_tuple %2 : $(SelfReordering, SelfReordering, SelfReordering)
// CHECK: [[RESULT:%.*]] = tuple ([[X_ADJ]] : $SelfReordering, [[Y_ADJ]] : $SelfReordering, [[SELF_ADJ]] : $SelfReordering)
// CHECK: return [[RESULT]]
}

// TF-742: Test thunks that perform self-ordering but not reabstraction.
struct SelfReorderingGeneric<Dummy>: Differentiable
where Dummy: Differentiable & ExpressibleByIntegerLiteral {
  // The property with type `Dummy` makes `Self` be indirect.
  var indirectDummy: Dummy = 0
  var x: Tracked<Float>
  init(_ x: Tracked<Float>) {
    self.x = x
  }

  // TF-742: Test method with three parameters (including `self`).
  // Note: pullback returns indirect `Self.TangentVector`.
  @differentiable(jvp: jvpThreeParameterMethod, vjp: vjpThreeParameterMethod)
  func threeParameterMethod<T: Differentiable, U: Differentiable>(_: T, _: U) -> Self
  where T.TangentVector: ExpressibleByFloatLiteral, U.TangentVector: ExpressibleByFloatLiteral {
    return self
  }
  func jvpThreeParameterMethod<T: Differentiable, U: Differentiable>(_ x: T, _ y: U)
    -> (Self, (Self.TangentVector, T.TangentVector, U.TangentVector) -> Self.TangentVector)
  where T.TangentVector: ExpressibleByFloatLiteral, U.TangentVector: ExpressibleByFloatLiteral {
    let value = threeParameterMethod(x, y)
    // TODO: Make this test meaningful/robust.
    return (value, { dself, dx, dy in dself })
  }
  func vjpThreeParameterMethod<T: Differentiable, U: Differentiable>(_ x: T, _ y: U)
    -> (Self, (Self.TangentVector) -> (Self.TangentVector, T.TangentVector, U.TangentVector))
  where T.TangentVector: ExpressibleByFloatLiteral, U.TangentVector: ExpressibleByFloatLiteral {
    let value = threeParameterMethod(x, y)
    return (value, { v in (v, 2.0, 3.0) })
  }

// CHECK-LABEL: sil hidden @AD__$s4main21SelfReorderingGenericV20threeParameterMethodyACyxGqd___qd_0_ts14DifferentiableRd__sAFRd_0_s25ExpressibleByFloatLiteral13TangentVectorRpd__sAgHRpd_0_r0_lF__jvp_src_0_wrt_0_1_2 : $@convention(method) <τ_0_0 where τ_0_0 : Differentiable, τ_0_0 : ExpressibleByIntegerLiteral><τ_1_0, τ_1_1 where τ_1_0 : Differentiable, τ_1_1 : Differentiable, τ_1_0.TangentVector : ExpressibleByFloatLiteral, τ_1_1.TangentVector : ExpressibleByFloatLiteral> (@in_guaranteed τ_1_0, @in_guaranteed τ_1_1, @in_guaranteed SelfReorderingGeneric<τ_0_0>) -> (@out SelfReorderingGeneric<τ_0_0>, @owned @callee_guaranteed (@in_guaranteed τ_1_0.TangentVector, @in_guaranteed τ_1_1.TangentVector, @in_guaranteed SelfReorderingGeneric<τ_0_0>.TangentVector) -> @out SelfReorderingGeneric<τ_0_0>.TangentVector)
// CHECK: bb0([[JVP_RESULT:%.*]] : $*SelfReorderingGeneric<τ_0_0>, [[X:%.*]] : $*τ_1_0, [[Y:%.*]] : $*τ_1_1, [[SELF:%.*]] : $*SelfReorderingGeneric<τ_0_0>):
// CHECK: [[JVP:%.*]] = function_ref @$s4main21SelfReorderingGenericV23jvpThreeParameterMethodyACyxG_AC13TangentVectorVyx_GAH_AFQyd__AFQyd_0_tctqd___qd_0_ts14DifferentiableRd__sAKRd_0_s25ExpressibleByFloatLiteralAIRQsAlJRQr0_lF
// CHECK: [[DF:%.*]] = apply [[JVP]]<τ_0_0, τ_1_0, τ_1_1>([[JVP_RESULT]], [[X]], [[Y]], [[SELF]])
// CHECK: [[DF_SELF_REORDER_THUNK:%.*]] = function_ref @AD__$s4main21SelfReorderingGenericV13TangentVectorVyx_GADQyd__ADQyd_0_AFIegnnnr_Agh2FIegnnnr_s14DifferentiableRzs27ExpressibleByIntegerLiteralRzsAIRd__sAIRd_0_s0hi5FloatK0AGRQsAkHRQr_0_lTR_differential_self_reordering_thunk
// CHECK: [[THUNKED_DF:%.*]] = partial_apply [callee_guaranteed] [[DF_SELF_REORDER_THUNK]]<τ_0_0, τ_1_0, τ_1_1>([[DF]])
// CHECK: return [[THUNKED_DF]]

// CHECK-LABEL: sil shared [transparent] [serialized] [reabstraction_thunk] [ossa] @AD__$s4main21SelfReorderingGenericV13TangentVectorVyx_GADQyd__ADQyd_0_AFIegnnnr_Agh2FIegnnnr_s14DifferentiableRzs27ExpressibleByIntegerLiteralRzsAIRd__sAIRd_0_s0hi5FloatK0AGRQsAkHRQr_0_lTR_differential_self_reordering_thunk : $@convention(thin) <τ_0_0 where τ_0_0 : Differentiable, τ_0_0 : ExpressibleByIntegerLiteral><τ_1_0, τ_1_1 where τ_1_0 : Differentiable, τ_1_1 : Differentiable, τ_1_0.TangentVector : ExpressibleByFloatLiteral, τ_1_1.TangentVector : ExpressibleByFloatLiteral> (@in_guaranteed τ_1_0.TangentVector, @in_guaranteed τ_1_1.TangentVector, @in_guaranteed SelfReorderingGeneric<τ_0_0>.TangentVector, @guaranteed @callee_guaranteed (@in_guaranteed SelfReorderingGeneric<τ_0_0>.TangentVector, @in_guaranteed τ_1_0.TangentVector, @in_guaranteed τ_1_1.TangentVector) -> @out SelfReorderingGeneric<τ_0_0>.TangentVector) -> @out SelfReorderingGeneric<τ_0_0>.TangentVector
// CHECK: bb0([[DF_RESULT:%.*]] : $*SelfReorderingGeneric<τ_0_0>.TangentVector, [[DX:%.*]] : $*τ_1_0.TangentVector, [[DY:%.*]] : $*τ_1_1.TangentVector, [[DSELF:%.*]] : $*SelfReorderingGeneric<τ_0_0>.TangentVector, [[DF:%.*]] : @guaranteed $@callee_guaranteed (@in_guaranteed SelfReorderingGeneric<τ_0_0>.TangentVector, @in_guaranteed τ_1_0.TangentVector, @in_guaranteed τ_1_1.TangentVector) -> @out SelfReorderingGeneric<τ_0_0>.TangentVector):
// CHECK: {{%.*}} = apply [[DF]]([[DF_RESULT]], [[DSELF]], [[DX]], [[DY]])
// CHECK: [[VOID:%.*]] = tuple ()
// CHECK: return [[VOID]]

// CHECK-LABEL: sil hidden @AD__$s4main21SelfReorderingGenericV20threeParameterMethodyACyxGqd___qd_0_ts14DifferentiableRd__sAFRd_0_s25ExpressibleByFloatLiteral13TangentVectorRpd__sAgHRpd_0_r0_lF__vjp_src_0_wrt_0_1_2 : $@convention(method) <τ_0_0 where τ_0_0 : Differentiable, τ_0_0 : ExpressibleByIntegerLiteral><τ_1_0, τ_1_1 where τ_1_0 : Differentiable, τ_1_1 : Differentiable, τ_1_0.TangentVector : ExpressibleByFloatLiteral, τ_1_1.TangentVector : ExpressibleByFloatLiteral> (@in_guaranteed τ_1_0, @in_guaranteed τ_1_1, @in_guaranteed SelfReorderingGeneric<τ_0_0>) -> (@out SelfReorderingGeneric<τ_0_0>, @owned @callee_guaranteed (@in_guaranteed SelfReorderingGeneric<τ_0_0>.TangentVector) -> (@out τ_1_0.TangentVector, @out τ_1_1.TangentVector, @out SelfReorderingGeneric<τ_0_0>.TangentVector))
// CHECK: bb0([[VJP_RESULT:%.*]] : $*SelfReorderingGeneric<τ_0_0>, [[X:%.*]] : $*τ_1_0, [[Y:%.*]] : $*τ_1_1, [[SELF:%.*]] : $*SelfReorderingGeneric<τ_0_0>):
// CHECK: [[VJP:%.*]] = function_ref @$s4main21SelfReorderingGenericV23vjpThreeParameterMethodyACyxG_AC13TangentVectorVyx_G_AFQyd__AFQyd_0_tAHctqd___qd_0_ts14DifferentiableRd__sAKRd_0_s25ExpressibleByFloatLiteralAIRQsAlJRQr0_lF
// CHECK: [[PB:%.*]] = apply [[VJP]]<τ_0_0, τ_1_0, τ_1_1>([[VJP_RESULT]], [[X]], [[Y]], [[SELF]])
// CHECK: [[PB_SELF_REORDER_THUNK:%.*]] = function_ref @AD__$s4main21SelfReorderingGenericV13TangentVectorVyx_GAfDQyd__ADQyd_0_Iegnrrr_AfghFIegnrrr_s14DifferentiableRzs27ExpressibleByIntegerLiteralRzsAIRd__sAIRd_0_s0hi5FloatK0AGRQsAkHRQr_0_lTR_pullback_self_reordering_thunk
// CHECK: [[THUNKED_PB:%.*]] = partial_apply [callee_guaranteed] [[PB_SELF_REORDER_THUNK]]<τ_0_0, τ_1_0, τ_1_1>([[PB]])
// CHECK: return [[THUNKED_PB]]

// CHECK-LABEL: sil shared [transparent] [serialized] [reabstraction_thunk] [ossa] @AD__$s4main21SelfReorderingGenericV13TangentVectorVyx_GAfDQyd__ADQyd_0_Iegnrrr_AfghFIegnrrr_s14DifferentiableRzs27ExpressibleByIntegerLiteralRzsAIRd__sAIRd_0_s0hi5FloatK0AGRQsAkHRQr_0_lTR_pullback_self_reordering_thunk : $@convention(thin) <τ_0_0 where τ_0_0 : Differentiable, τ_0_0 : ExpressibleByIntegerLiteral><τ_1_0, τ_1_1 where τ_1_0 : Differentiable, τ_1_1 : Differentiable, τ_1_0.TangentVector : ExpressibleByFloatLiteral, τ_1_1.TangentVector : ExpressibleByFloatLiteral> (@in_guaranteed SelfReorderingGeneric<τ_0_0>.TangentVector, @guaranteed @callee_guaranteed (@in_guaranteed SelfReorderingGeneric<τ_0_0>.TangentVector) -> (@out SelfReorderingGeneric<τ_0_0>.TangentVector, @out τ_1_0.TangentVector, @out τ_1_1.TangentVector)) -> (@out τ_1_0.TangentVector, @out τ_1_1.TangentVector, @out SelfReorderingGeneric<τ_0_0>.TangentVector)
// CHECK: bb0([[X_ADJ:%.*]] : $*τ_1_0.TangentVector, [[Y_ADJ:%.*]] : $*τ_1_1.TangentVector, [[SELF_ADJ:%.*]] : $*SelfReorderingGeneric<τ_0_0>.TangentVector, [[SEED:%.*]] : $*SelfReorderingGeneric<τ_0_0>.TangentVector, [[PB:%.*]] : @guaranteed $@callee_guaranteed (@in_guaranteed SelfReorderingGeneric<τ_0_0>.TangentVector) -> (@out SelfReorderingGeneric<τ_0_0>.TangentVector, @out τ_1_0.TangentVector, @out τ_1_1.TangentVector)):
// CHECK: {{%.*}} = apply [[PB]]([[SELF_ADJ]], [[X_ADJ]], [[Y_ADJ]], [[SEED]])
// CHECK: [[VOID:%.*]] = tuple ()
// CHECK: return [[VOID]]
}

extension SelfReorderingGeneric.TangentVector : ExpressibleByFloatLiteral {}

DerivativeSILGenThunkTests.testWithLeakChecking("SelfReorderingNonReabstractingThunk") {
  do {
    let v = SelfReordering(1)
    // TODO: Add JVP/differential tests.
    expectEqual((SelfReordering(1), SelfReordering(2), SelfReordering(3)),
                pullback(at: v, v, v) { x, y, z in x.threeParameterMethod(y, z) }(v))
  }
  do {
    let dummy: Float = 0
    let x = SelfReorderingGeneric<Float>(1)
    let v = SelfReorderingGeneric<Float>.TangentVector(indirectDummy: dummy, x: 1)
    let tracked = Tracked<Float>(1.0)
    // TODO: Add JVP/differential tests.
    expectEqual((v, 2, 3),
                pullback(at: x, tracked, tracked) { x, y, z in x.threeParameterMethod(y, z) }(v))
  }
}

runAllTests()
