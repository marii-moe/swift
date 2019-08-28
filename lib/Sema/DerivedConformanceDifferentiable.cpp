//===--- DerivedConformanceDifferentiable.cpp - Derived Differentiable ----===//
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
// This file implements explicit derivation of the Differentiable protocol for
// struct and class types.
//
//===----------------------------------------------------------------------===//

#include "CodeSynthesis.h"
#include "TypeChecker.h"
#include "swift/AST/AutoDiff.h"
#include "swift/AST/Decl.h"
#include "swift/AST/Expr.h"
#include "swift/AST/Module.h"
#include "swift/AST/ParameterList.h"
#include "swift/AST/Pattern.h"
#include "swift/AST/ProtocolConformance.h"
#include "swift/AST/Stmt.h"
#include "swift/AST/Types.h"
#include "DerivedConformances.h"

using namespace swift;

/// Return the protocol requirement with the specified name.
/// TODO: Move function to shared place for use with other derived conformances.
static ValueDecl *getProtocolRequirement(ProtocolDecl *proto, Identifier name) {
  auto lookup = proto->lookupDirect(name);
  // Erase declarations that are not protocol requirements.
  // This is important for removing default implementations of the same name.
  llvm::erase_if(lookup, [](ValueDecl *v) {
    return !isa<ProtocolDecl>(v->getDeclContext()) ||
           !v->isProtocolRequirement();
  });
  assert(lookup.size() <= 1 && "Ambiguous protocol requirement");
  return lookup.front();
}

/// Get the stored properties of a nominal type that are relevant for
/// differentiation, except the ones tagged `@noDerivative`.
static void
getStoredPropertiesForDifferentiation(NominalTypeDecl *nominal,
                                      DeclContext *DC,
                                      SmallVectorImpl<VarDecl *> &result) {
  auto &C = nominal->getASTContext();
  auto *diffableProto = C.getProtocol(KnownProtocolKind::Differentiable);
  for (auto *vd : nominal->getStoredProperties()) {
    if (vd->getAttrs().hasAttribute<NoDerivativeAttr>())
      continue;
    if (vd->isLet())
      continue;
    if (!vd->hasInterfaceType())
      C.getLazyResolver()->resolveDeclSignature(vd);
    if (!vd->hasInterfaceType())
      continue;
    auto varType = DC->mapTypeIntoContext(vd->getValueInterfaceType());
    if (!TypeChecker::conformsToProtocol(varType, diffableProto, nominal,
                                         None))
      continue;
    result.push_back(vd);
  }
}

/// Convert the given `ValueDecl` to a `StructDecl` if it is a `StructDecl` or a
/// `TypeDecl` with an underlying struct type. Otherwise, return `nullptr`.
static StructDecl *convertToStructDecl(ValueDecl *v) {
  if (auto *structDecl = dyn_cast<StructDecl>(v))
    return structDecl;
  auto *typeDecl = dyn_cast<TypeDecl>(v);
  if (!typeDecl)
    return nullptr;
  return dyn_cast_or_null<StructDecl>(
      typeDecl->getDeclaredInterfaceType()->getAnyNominal());
}

/// Get the `Differentiable` protocol `TangentVector` associated type for the
/// given `VarDecl`.
/// TODO: Generalize and move function to shared place for use with other derived
/// conformances.
static Type getTangentVectorType(VarDecl *decl, DeclContext *DC) {
  auto &C = decl->getASTContext();
  auto *diffableProto = C.getProtocol(KnownProtocolKind::Differentiable);
  if (!decl->hasInterfaceType())
    C.getLazyResolver()->resolveDeclSignature(decl);
  auto varType = DC->mapTypeIntoContext(decl->getValueInterfaceType());
  auto conf = TypeChecker::conformsToProtocol(varType, diffableProto, DC,
                                              None);
  if (!conf)
    return nullptr;
  Type tangentType = conf->getTypeWitnessByName(varType, C.Id_TangentVector);
  return tangentType;
}

// Get the `Differentiable` protocol associated `TangentVector` struct for the
// given nominal `DeclContext`. Asserts that the `TangentVector` struct type
// exists.
static StructDecl *getTangentVectorStructDecl(DeclContext *DC) {
  assert(DC->getSelfNominalTypeDecl() && "Must be a nominal `DeclContext`");
  auto &C = DC->getASTContext();
  auto *diffableProto = C.getProtocol(KnownProtocolKind::Differentiable);
  assert(diffableProto && "`Differentiable` protocol not found");
  auto conf = TypeChecker::conformsToProtocol(DC->getSelfTypeInContext(),
                                              diffableProto, DC, None);
  assert(conf && "Nominal must conform to `Differentiable`");
  auto assocType = conf->getTypeWitnessByName(
      DC->getSelfTypeInContext(), C.Id_TangentVector);
  assert(assocType && "`Differentiable.TangentVector` type not found");
  auto *structDecl = dyn_cast<StructDecl>(assocType->getAnyNominal());
  assert(structDecl && "Associated type must be a struct type");
  return structDecl;
}

bool DerivedConformance::canDeriveDifferentiable(NominalTypeDecl *nominal,
                                                 DeclContext *DC) {
  // Nominal type must be a struct or class. (No stored properties is okay.)
  if (!isa<StructDecl>(nominal) && !isa<ClassDecl>(nominal))
    return false;
  auto &C = nominal->getASTContext();
  auto *lazyResolver = C.getLazyResolver();
  auto *diffableProto = C.getProtocol(KnownProtocolKind::Differentiable);
  auto *addArithProto = C.getProtocol(KnownProtocolKind::AdditiveArithmetic);

  // Nominal type must not customize `TangentVector` to anything other than
  // `Self`. Otherwise, synthesis is semantically unsupported.
  auto tangentDecls = nominal->lookupDirect(C.Id_TangentVector);
  auto nominalTypeInContext =
      DC->mapTypeIntoContext(nominal->getDeclaredInterfaceType());

  auto isValidAssocTypeCandidate = [&](ValueDecl *v) -> StructDecl * {
    // Valid candidate must be a struct or a typealias to a struct.
    auto *structDecl = convertToStructDecl(v);
    if (!structDecl)
      return nullptr;
    // Valid candidate must either:
    // 1. Be implicit (previously synthesized).
    if (structDecl->isImplicit())
      return structDecl;
    // 2. Equal nominal's implicit parent.
    //    This can occur during mutually recursive constraints. Example:
    //   `X == X.TangentVector`.
    if (nominal->isImplicit() && structDecl == nominal->getDeclContext() &&
        TypeChecker::conformsToProtocol(structDecl->getDeclaredInterfaceType(),
                                        diffableProto, DC, None))
      return structDecl;
    // 3. Equal nominal and conform to `AdditiveArithmetic`.
    if (structDecl == nominal) {
      // Check conformance to `AdditiveArithmetic`.
      if (TypeChecker::conformsToProtocol(nominalTypeInContext, addArithProto,
                                          DC, None))
        return structDecl;
    }
    // Otherwise, candidate is invalid.
    return nullptr;
  };

  auto invalidTangentDecls = llvm::partition(tangentDecls, [&](ValueDecl *v) {
    return isValidAssocTypeCandidate(v);
  });

  auto validTangentDeclCount =
      std::distance(tangentDecls.begin(), invalidTangentDecls);
  auto invalidTangentDeclCount =
      std::distance(invalidTangentDecls, tangentDecls.end());

  // There cannot be any invalid `TangentVector` types.
  // There can be at most one valid `TangentVector` type.
  if (invalidTangentDeclCount != 0 || validTangentDeclCount > 1)
    return false;

  // All stored properties not marked with `@noDerivative`:
  // - Must conform to `Differentiable`.
  // - Must not have any `let` stored properties with an initial value.
  //   - This restriction may be lifted later with support for "true" memberwise
  //     initializers that initialize all stored properties, including initial
  //     value information.
  SmallVector<VarDecl *, 16> diffProperties;
  getStoredPropertiesForDifferentiation(nominal, DC, diffProperties);
  return llvm::all_of(diffProperties, [&](VarDecl *v) {
    if (!v->hasInterfaceType())
      lazyResolver->resolveDeclSignature(v);
    if (!v->hasInterfaceType())
      return false;
    auto varType = DC->mapTypeIntoContext(v->getValueInterfaceType());
    return (bool)TypeChecker::conformsToProtocol(varType, diffableProto, DC,
                                                 None);
  });
}

/// Determine if a EuclideanDifferentiable requirement can be derived for a type.
///
/// \returns True if the requirement can be derived.
bool DerivedConformance::canDeriveEuclideanDifferentiable(
    NominalTypeDecl *nominal, DeclContext *DC) {
  if (!canDeriveDifferentiable(nominal, DC))
    return false;
  auto &C = nominal->getASTContext();
  auto *lazyResolver = C.getLazyResolver();
  auto *eucDiffProto =
      C.getProtocol(KnownProtocolKind::EuclideanDifferentiable);
  // Return true if all differentiation stored properties conform to
  // `AdditiveArithmetic` and their `TangentVector` equals themselves.
  SmallVector<VarDecl *, 16> diffProperties;
  getStoredPropertiesForDifferentiation(nominal, DC, diffProperties);
  return llvm::all_of(diffProperties, [&](VarDecl *member) {
    if (!member->hasInterfaceType())
      lazyResolver->resolveDeclSignature(member);
    if (!member->hasInterfaceType())
      return false;
    auto varType = DC->mapTypeIntoContext(member->getValueInterfaceType());
    return (bool)TypeChecker::conformsToProtocol(
        varType, eucDiffProto, DC, None);
  });
}

/// Synthesize body for a `Differentiable` method requirement.
static std::pair<BraceStmt *, bool>
deriveBodyDifferentiable_method(AbstractFunctionDecl *funcDecl,
                                Identifier methodName,
                                Identifier methodParamLabel) {
  auto *parentDC = funcDecl->getParent();
  auto *nominal = parentDC->getSelfNominalTypeDecl();
  auto &C = nominal->getASTContext();

  // Get method protocol requirement.
  auto *diffProto = C.getProtocol(KnownProtocolKind::Differentiable);
  auto *methodReq = getProtocolRequirement(diffProto, methodName);

  // Get references to `self` and parameter declarations.
  auto *selfDecl = funcDecl->getImplicitSelfDecl();
  auto *selfDRE =
      new (C) DeclRefExpr(selfDecl, DeclNameLoc(), /*Implicit*/ true);
  auto *paramDecl = funcDecl->getParameters()->get(0);
  auto *paramDRE =
      new (C) DeclRefExpr(paramDecl, DeclNameLoc(), /*Implicit*/ true);

  SmallVector<VarDecl *, 8> diffProperties;
  getStoredPropertiesForDifferentiation(nominal, parentDC, diffProperties);

  // Create call expression applying a member method to a parameter member.
  // Format: `<member>.method(<parameter>.<member>)`.
  // Example: `x.move(along: direction.x)`.
  auto createMemberMethodCallExpr = [&](VarDecl *member) -> Expr * {
    auto *module = nominal->getModuleContext();
    auto memberType =
        parentDC->mapTypeIntoContext(member->getValueInterfaceType());
    auto confRef = module->lookupConformance(memberType, diffProto);
    assert(confRef && "Member does not conform to `Differentiable`");

    // Get member type's method, e.g. `Member.move(along:)`.
    // Use protocol requirement declaration for the method by default: this
    // will be dynamically dispatched.
    ValueDecl *memberMethodDecl = methodReq;
    // If conformance reference is concrete, then use concrete witness
    // declaration for the operator.
    if (confRef->isConcrete())
      memberMethodDecl = confRef->getConcrete()->getWitnessDecl(
          methodReq);
    assert(memberMethodDecl && "Member method declaration must exist");
    auto memberMethodDRE =
        new (C) DeclRefExpr(memberMethodDecl, DeclNameLoc(), /*Implicit*/ true);
    memberMethodDRE->setFunctionRefKind(FunctionRefKind::SingleApply);

    // Create reference to member method: `x.move(along:)`.
    auto memberExpr =
        new (C) MemberRefExpr(selfDRE, SourceLoc(), member, DeclNameLoc(),
                              /*Implicit*/ true);
    auto memberMethodExpr =
        new (C) DotSyntaxCallExpr(memberMethodDRE, SourceLoc(), memberExpr);

    // Create reference to parameter member: `direction.x`.
    VarDecl *paramMember = nullptr;
    auto *paramNominal = paramDecl->getType()->getAnyNominal();
    assert(paramNominal && "Parameter should have a nominal type");
    // Find parameter member corresponding to returned nominal member.
    for (auto *candidate : paramNominal->getStoredProperties()) {
      if (candidate->getName() == member->getName()) {
        paramMember = candidate;
        break;
      }
    }
    assert(paramMember && "Could not find corresponding parameter member");
    auto *paramMemberExpr =
        new (C) MemberRefExpr(paramDRE, SourceLoc(), paramMember, DeclNameLoc(),
                              /*Implicit*/ true);
    // Create expression: `x.move(along: direction.x)`.
    return CallExpr::createImplicit(C, memberMethodExpr, {paramMemberExpr},
                                    {methodParamLabel});
  };

  // Create array of member method call expressions.
  llvm::SmallVector<ASTNode, 2> memberMethodCallExprs;
  llvm::SmallVector<Identifier, 2> memberNames;
  for (auto *member : diffProperties) {
    memberMethodCallExprs.push_back(createMemberMethodCallExpr(member));
    memberNames.push_back(member->getName());
  }
  auto *braceStmt = BraceStmt::create(C, SourceLoc(), memberMethodCallExprs,
                                      SourceLoc(), true);
  return std::pair<BraceStmt *, bool>(braceStmt, false);
}

/// Synthesize body for `move(along:)`.
static std::pair<BraceStmt *, bool>
deriveBodyDifferentiable_move(AbstractFunctionDecl *funcDecl, void *) {
  auto &C = funcDecl->getASTContext();
  return deriveBodyDifferentiable_method(funcDecl, C.Id_move,
                                         C.getIdentifier("along"));
}

/// Synthesize function declaration for a `Differentiable` method requirement.
static ValueDecl *deriveDifferentiable_method(
    DerivedConformance &derived, Identifier methodName, Identifier argumentName,
    Identifier parameterName, Type parameterType, Type returnType,
    AbstractFunctionDecl::BodySynthesizer bodySynthesizer) {
  auto *nominal = derived.Nominal;
  auto &C = derived.TC.Context;
  auto *parentDC = derived.getConformanceContext();

  auto *param =
      new (C) ParamDecl(ParamDecl::Specifier::Default, SourceLoc(), SourceLoc(),
                        argumentName, SourceLoc(), parameterName, parentDC);
  param->setInterfaceType(parameterType);
  ParameterList *params = ParameterList::create(C, {param});

  DeclName declName(C, methodName, params);
  auto *funcDecl = FuncDecl::create(C, SourceLoc(), StaticSpellingKind::None,
                                    SourceLoc(), declName, SourceLoc(),
                                    /*Throws*/ false, SourceLoc(),
                                    /*GenericParams=*/nullptr, params,
                                    TypeLoc::withoutLoc(returnType), parentDC);
  if (!nominal->getSelfClassDecl())
    funcDecl->setSelfAccessKind(SelfAccessKind::Mutating);
  funcDecl->setImplicit();
  funcDecl->setBodySynthesizer(bodySynthesizer.Fn, bodySynthesizer.Context);

  if (auto *env = parentDC->getGenericEnvironmentOfContext())
    funcDecl->setGenericEnvironment(env);
  funcDecl->computeType();
  funcDecl->copyFormalAccessFrom(nominal, /*sourceIsParentContext*/ true);
  funcDecl->setValidationToChecked();

  derived.addMembersToConformanceContext({funcDecl});
  C.addSynthesizedDecl(funcDecl);

  return funcDecl;
}

/// Synthesize the `move(along:)` function declaration.
static ValueDecl *deriveDifferentiable_move(DerivedConformance &derived) {
  auto &C = derived.TC.Context;
  auto *parentDC = derived.getConformanceContext();

  auto *tangentDecl = getTangentVectorStructDecl(parentDC);
  auto tangentType = tangentDecl->getDeclaredInterfaceType();

  return deriveDifferentiable_method(
      derived, C.Id_move, C.getIdentifier("along"),
      C.getIdentifier("direction"), tangentType, C.TheEmptyTupleType,
      {deriveBodyDifferentiable_move, nullptr});
}

/// Synthesize the `differentiableVectorView` property declaration.
static ValueDecl *deriveEuclideanDifferentiable_differentiableVectorView(
    DerivedConformance &derived) {
  auto &C = derived.TC.Context;
  auto *parentDC = derived.getConformanceContext();

  auto *tangentDecl = getTangentVectorStructDecl(parentDC);
  auto tangentType = tangentDecl->getDeclaredInterfaceType();
  auto tangentContextualType = parentDC->mapTypeIntoContext(tangentType);

  VarDecl *vectorViewDecl;
  PatternBindingDecl *pbDecl;
  std::tie(vectorViewDecl, pbDecl) = derived.declareDerivedProperty(
      C.Id_differentiableVectorView, tangentType, tangentContextualType,
      /*isStatic*/ false, /*isFinal*/ true);

  struct GetterSynthesizerContext {
    StructDecl *tangentDecl;
    Type tangentContextualType;
  };

  auto getterSynthesizer = [](AbstractFunctionDecl *getterDecl, void *ctx)
      -> std::pair<BraceStmt *, bool> {
    auto *context = reinterpret_cast<GetterSynthesizerContext *>(ctx);
    assert(context && "Invalid context");
    auto *parentDC = getterDecl->getParent();
    auto *nominal = parentDC->getSelfNominalTypeDecl();
    auto *module = nominal->getModuleContext();
    auto &C = nominal->getASTContext();
    auto *eucDiffProto =
        C.getProtocol(KnownProtocolKind::EuclideanDifferentiable);
    auto *vectorViewReq =
        eucDiffProto->lookupDirect(C.Id_differentiableVectorView).front();

    SmallVector<VarDecl *, 8> diffProperties;
    getStoredPropertiesForDifferentiation(nominal, parentDC, diffProperties);

    // Create a reference to the memberwise initializer: `TangentVector.init`.
    auto *memberwiseInitDecl =
        context->tangentDecl->getEffectiveMemberwiseInitializer();
    assert(memberwiseInitDecl && "Memberwise initializer must exist");
    assert(diffProperties.size() ==
               memberwiseInitDecl->getParameters()->size());
    // `TangentVector`
    auto *tangentTypeExpr =
        TypeExpr::createImplicit(context->tangentContextualType, C);
    // `TangentVector.init`
    auto *initDRE = new (C) DeclRefExpr(memberwiseInitDecl, DeclNameLoc(),
                                        /*Implicit*/ true);
    initDRE->setFunctionRefKind(FunctionRefKind::SingleApply);
    auto *initExpr = new (C) ConstructorRefCallExpr(initDRE, tangentTypeExpr);
    initExpr->setThrows(false);
    initExpr->setImplicit();

    // Create a call:
    //   TangentVector.init(
    //     <property_name_1...>:
    //        self.<property_name_1>.differentiableVectorView,
    //     <property_name_2...>:
    //        self.<property_name_2>.differentiableVectorView,
    //     ...
    //   )
    SmallVector<Identifier, 8> argLabels;
    SmallVector<Expr *, 8> memberRefs;
    for (auto *member : diffProperties) {
      auto *selfDRE = new (C) DeclRefExpr(getterDecl->getImplicitSelfDecl(),
                                          DeclNameLoc(),
                                          /*Implicit*/ true);
      auto *memberExpr = new (C) MemberRefExpr(
          selfDRE, SourceLoc(), member, DeclNameLoc(), /*Implicit*/ true);
      auto memberType =
          parentDC->mapTypeIntoContext(member->getValueInterfaceType());
      auto confRef = module->lookupConformance(memberType, eucDiffProto);
      assert(confRef &&
             "Member missing conformance to `EuclideanDifferentiable`");
      ConcreteDeclRef memberDeclRef = vectorViewReq;
      if (confRef->isConcrete())
        memberDeclRef = confRef->getConcrete()->getWitnessDecl(vectorViewReq);
      argLabels.push_back(member->getName());
      memberRefs.push_back(new (C) MemberRefExpr(
          memberExpr, SourceLoc(), memberDeclRef, DeclNameLoc(),
          /*Implicit*/ true));
    }
    assert(memberRefs.size() == argLabels.size());
    CallExpr *callExpr =
        CallExpr::createImplicit(C, initExpr, memberRefs, argLabels);

    // Create a return statement: `return TangentVector.init(...)`.
    ASTNode retStmt =
        new (C) ReturnStmt(SourceLoc(), callExpr, /*implicit*/ true);
    auto *braceStmt = BraceStmt::create(C, SourceLoc(), retStmt, SourceLoc(),
                                        /*implicit*/ true);
    return std::make_pair(braceStmt, false);
  };
  auto *getterDecl = derived.addGetterToReadOnlyDerivedProperty(
      vectorViewDecl, tangentContextualType);
  getterDecl->setBodySynthesizer(
      getterSynthesizer, /*context*/ C.AllocateObjectCopy(
          GetterSynthesizerContext{tangentDecl, tangentContextualType}));
  derived.addMembersToConformanceContext({vectorViewDecl, pbDecl});
  return vectorViewDecl;
}

/// Return associated `TangentVector` struct for a nominal type, if it exists.
/// If not, synthesize the struct.
static StructDecl *
getOrSynthesizeTangentVectorStruct(DerivedConformance &derived, Identifier id) {
  auto &TC = derived.TC;
  auto *parentDC = derived.getConformanceContext();
  auto *nominal = derived.Nominal;
  auto &C = nominal->getASTContext();

  // If the associated struct already exists, return it.
  auto lookup = nominal->lookupDirect(C.Id_TangentVector);
  assert(lookup.size() < 2 &&
         "Expected at most one associated type named `TangentVector`");
  if (lookup.size() == 1) {
    auto *structDecl = convertToStructDecl(lookup.front());
    assert(structDecl && "Expected lookup result to be a struct");
    return structDecl;
  }

  // Otherwise, synthesize a new struct.
  auto *diffableProto = C.getProtocol(KnownProtocolKind::Differentiable);
  auto diffableType = TypeLoc::withoutLoc(diffableProto->getDeclaredType());
  auto *addArithProto = C.getProtocol(KnownProtocolKind::AdditiveArithmetic);
  auto addArithType = TypeLoc::withoutLoc(addArithProto->getDeclaredType());
  auto *pointMulProto =
      C.getProtocol(KnownProtocolKind::PointwiseMultiplicative);
  auto pointMulType = TypeLoc::withoutLoc(pointMulProto->getDeclaredType());
  auto *mathProto = C.getProtocol(KnownProtocolKind::ElementaryFunctions);
  auto mathType = TypeLoc::withoutLoc(mathProto->getDeclaredType());
  auto *vectorProto = C.getProtocol(KnownProtocolKind::VectorProtocol);
  auto vectorType = TypeLoc::withoutLoc(vectorProto->getDeclaredType());
  auto *kpIterableProto = C.getProtocol(KnownProtocolKind::KeyPathIterable);
  auto kpIterableType = TypeLoc::withoutLoc(kpIterableProto->getDeclaredType());

  // By definition, `TangentVector` must conform to `Differentiable` and
  // `AdditiveArithmetic`.
  SmallVector<TypeLoc, 4> inherited{diffableType, addArithType};

  // Cache original members and their associated types for later use.
  SmallVector<VarDecl *, 8> diffProperties;
  getStoredPropertiesForDifferentiation(nominal, parentDC, diffProperties);

  // Add ad-hoc implicit conformances for `TangentVector`.
  // TODO(TF-632): Remove this implicit conformance logic when synthesized
  // member types can be extended.

  // `TangentVector` struct can derive `PointwiseMultiplicative` if the
  // `TangentVector` types of all stored properties conform to
  // `PointwiseMultiplicative`.
  bool canDerivePointwiseMultiplicative =
      llvm::all_of(diffProperties, [&](VarDecl *vd) {
        return TC.conformsToProtocol(getTangentVectorType(vd, parentDC),
                                     pointMulProto, parentDC, None);
      });

  // `TangentVector` struct can derive `ElementaryFunctions` if the
  // `TangentVector` types of all stored properties conform to
  // `ElementaryFunctions`.
  bool canDeriveElementaryFunctions =
      llvm::all_of(diffProperties, [&](VarDecl *vd) {
        return TC.conformsToProtocol(getTangentVectorType(vd, parentDC),
                                     mathProto, parentDC, None);
      });

  // `TangentVector` struct can derive `VectorProtocol` if the `TangentVector`
  // types of all members conform to `VectorProtocol` and share the same
  // `VectorSpaceScalar` type.
  Type sameScalarType;
  bool canDeriveVectorProtocol = !diffProperties.empty() &&
      llvm::all_of(diffProperties, [&](VarDecl *vd) {
        auto conf = TC.conformsToProtocol(getTangentVectorType(vd, parentDC),
                                          vectorProto, nominal, None);
        if (!conf)
          return false;
        auto scalarType =
            conf->getTypeWitnessByName(vd->getType(), C.Id_VectorSpaceScalar);
        if (!sameScalarType) {
          sameScalarType = scalarType;
          return true;
        }
        return scalarType->isEqual(sameScalarType);
      });

  // `TangentVector` struct should derive `KeyPathIterable` if the parent struct
  // conforms to `KeyPathIterable`.
  bool shouldDeriveKeyPathIterable =
      TC.conformsToProtocol(nominal->getDeclaredInterfaceType(),
                            kpIterableProto, parentDC, None).hasValue();

  // If all members conform to `PointwiseMultiplicative`, make the
  // `TangentVector` struct conform to `PointwiseMultiplicative`.
  if (canDerivePointwiseMultiplicative)
    inherited.push_back(pointMulType);
  // If all members conform to `ElementaryFunctions`, make the `TangentVector`
  // struct conform to `ElementaryFunctions`.
  if (canDeriveElementaryFunctions)
    inherited.push_back(mathType);
  // If all members also conform to `VectorProtocol` with the same `Scalar`
  // type, make the `TangentVector` struct conform to `VectorProtocol`.
  if (canDeriveVectorProtocol)
    inherited.push_back(vectorType);
  // If parent type conforms to `KeyPathIterable`, make the `TangentVector`
  // struct conform to `KeyPathIterable`.
  if (shouldDeriveKeyPathIterable)
    inherited.push_back(kpIterableType);

  auto *structDecl =
      new (C) StructDecl(SourceLoc(), C.Id_TangentVector, SourceLoc(),
                         /*Inherited*/ C.AllocateCopy(inherited),
                         /*GenericParams*/ {}, parentDC);
  structDecl->setImplicit();
  structDecl->copyFormalAccessFrom(nominal, /*sourceIsParentContext*/ true);

  // Add members to `TangentVector` struct.
  for (auto *member : diffProperties) {
    // Add this member's corresponding `TangentVector` type to the parent's
    // `TangentVector` struct.
    auto *newMember = new (C) VarDecl(
        member->isStatic(), member->getIntroducer(), member->isCaptureList(),
        /*NameLoc*/ SourceLoc(), member->getName(), structDecl);
    // NOTE: `newMember` is not marked as implicit here, because that affects
    // memberwise initializer synthesis.

    auto memberAssocType = getTangentVectorType(member, parentDC);
    auto memberAssocInterfaceType = memberAssocType->hasArchetype()
                                        ? memberAssocType->mapTypeOutOfContext()
                                        : memberAssocType;
    auto memberAssocContextualType =
        parentDC->mapTypeIntoContext(memberAssocInterfaceType);
    newMember->setInterfaceType(memberAssocInterfaceType);
    newMember->setType(memberAssocContextualType);
    Pattern *memberPattern =
        new (C) NamedPattern(newMember, /*implicit*/ true);
    memberPattern->setType(memberAssocContextualType);
    memberPattern = TypedPattern::createImplicit(
        C, memberPattern, memberAssocContextualType);
    memberPattern->setType(memberAssocContextualType);
    auto *memberBinding = PatternBindingDecl::createImplicit(
        C, StaticSpellingKind::None, memberPattern, /*initExpr*/ nullptr,
        structDecl);
    structDecl->addMember(newMember);
    structDecl->addMember(memberBinding);
    newMember->copyFormalAccessFrom(member, /*sourceIsParentContext*/ true);
    newMember->setValidationToChecked();
    newMember->setSetterAccess(member->getFormalAccess());
    C.addSynthesizedDecl(newMember);
    C.addSynthesizedDecl(memberBinding);

    // Now that this member is in the `TangentVector` type, it should be marked
    // `@differentiable` so that the differentiation transform will synthesize
    // associated functions for it. We only add this to public stored
    // properties, because their access outside the module will go through a
    // call to the getter.
    if (member->getEffectiveAccess() > AccessLevel::Internal &&
        !member->getAttrs().hasAttribute<DifferentiableAttr>()) {
      if (!member->getSynthesizedAccessor(AccessorKind::Get)
               ->hasInterfaceType())
        TC.resolveDeclSignature(member->getAccessor(AccessorKind::Get));
      // If member or its getter already has a `@differentiable` attribute,
      // continue.
      if (member->getAttrs().hasAttribute<DifferentiableAttr>() ||
          member->getAccessor(AccessorKind::Get)
              ->getAttrs()
              .hasAttribute<DifferentiableAttr>())
        continue;
      ArrayRef<Requirement> requirements;
      // If the parent declaration context is an extension, the nominal type may
      // conditionally conform to `Differentiable`. Use the conditional
      // conformance requirements in getter `@differentiable` attributes.
      if (auto *extDecl = dyn_cast<ExtensionDecl>(parentDC->getAsDecl()))
        requirements = extDecl->getGenericRequirements();
      auto *diffableAttr = DifferentiableAttr::create(
          C, /*implicit*/ true, SourceLoc(), SourceLoc(),
          /*linear*/ false, {}, None, None, requirements);
      member->getAttrs().add(diffableAttr);
      // Compute getter parameter indices.
      auto *getterType = member->getAccessor(AccessorKind::Get)
                             ->getInterfaceType()
                             ->castTo<AnyFunctionType>();
      AutoDiffParameterIndicesBuilder builder(getterType);
      builder.setParameter(0);
      diffableAttr->setParameterIndices(builder.build(C));
    }
  }

  // If nominal type has `@_fixed_layout` attribute, mark `TangentVector` struct
  // as `@_fixed_layout` as well.
  if (nominal->getAttrs().hasAttribute<FixedLayoutAttr>())
    structDecl->addFixedLayoutAttr();

  // The implicit memberwise constructor must be explicitly created so that it
  // can called in `AdditiveArithmetic` and `Differentiable` methods. Normally,
  // the memberwise constructor is synthesized during SILGen, which is too late.
  auto *initDecl = createMemberwiseImplicitConstructor(TC, structDecl);
  structDecl->addMember(initDecl);
  C.addSynthesizedDecl(initDecl);

  // After memberwise initializer is synthesized, mark members as implicit.
  for (auto *member : structDecl->getStoredProperties())
    member->setImplicit();

  derived.addMembersToConformanceContext({structDecl});
  C.addSynthesizedDecl(structDecl);

  return structDecl;
}

/// Add a typealias declaration with the given name and underlying target
/// struct type to the given source nominal declaration context.
static void addAssociatedTypeAliasDecl(Identifier name,
                                       DeclContext *sourceDC,
                                       StructDecl *target,
                                       TypeChecker &TC) {
  auto &C = TC.Context;
  auto *nominal = sourceDC->getSelfNominalTypeDecl();
  assert(nominal && "Expected `DeclContext` to be a nominal type");
  auto lookup = nominal->lookupDirect(name);
  assert(lookup.size() < 2 &&
         "Expected at most one associated type named member");
  // If implicit type declaration with the given name already exists in source
  // struct, return it.
  if (lookup.size() == 1) {
    auto existingTypeDecl = dyn_cast<TypeDecl>(lookup.front());
    assert(existingTypeDecl && existingTypeDecl->isImplicit() &&
           "Expected lookup result to be an implicit type declaration");
    return;
  }
  // Otherwise, create a new typealias.
  auto *aliasDecl = new (C)
      TypeAliasDecl(SourceLoc(), SourceLoc(), name, SourceLoc(), {}, sourceDC);
  aliasDecl->setUnderlyingType(target->getDeclaredInterfaceType());
  aliasDecl->setImplicit();
  if (auto env = sourceDC->getGenericEnvironmentOfContext())
    aliasDecl->setGenericEnvironment(env);
  cast<IterableDeclContext>(sourceDC->getAsDecl())->addMember(aliasDecl);
  aliasDecl->copyFormalAccessFrom(nominal, /*sourceIsParentContext*/ true);
  aliasDecl->setValidationToChecked();
  TC.validateDecl(aliasDecl);
  C.addSynthesizedDecl(aliasDecl);
};

/// Diagnose stored properties in the nominal that do not have an explicit
/// `@noDerivative` attribute, but either:
/// - Do not conform to `Differentiable`.
/// - Are a `let` stored property.
/// Emit a warning and a fixit so that users will make the attribute explicit.
static void checkAndDiagnoseImplicitNoDerivative(TypeChecker &TC,
                                                 NominalTypeDecl *nominal,
                                                 DeclContext* DC) {
  auto *diffableProto =
      TC.Context.getProtocol(KnownProtocolKind::Differentiable);
  bool nominalCanDeriveAdditiveArithmetic =
      DerivedConformance::canDeriveAdditiveArithmetic(nominal, DC);
  for (auto *vd : nominal->getStoredProperties()) {
    if (!vd->hasInterfaceType())
      TC.resolveDeclSignature(vd);
    if (!vd->hasInterfaceType())
      continue;
    auto varType = DC->mapTypeIntoContext(vd->getValueInterfaceType());
    if (vd->getAttrs().hasAttribute<NoDerivativeAttr>())
      continue;
    // Check whether to diagnose stored property.
    bool conformsToDifferentiable =
        TC.conformsToProtocol(varType, diffableProto, nominal, None).hasValue();
    // If stored property should not be diagnosed, continue.
    if (conformsToDifferentiable && !vd->isLet())
      continue;
    // Otherwise, add an implicit `@noDerivative` attribute.
    vd->getAttrs().add(
        new (TC.Context) NoDerivativeAttr(/*Implicit*/ true));
    auto loc = vd->getAttributeInsertionLoc(/*forModifier*/ false);
    assert(loc.isValid() && "Expected valid source location");
    // If nominal type can conform to `AdditiveArithmetic`, suggest conforming
    // adding a conformance to `AdditiveArithmetic`.
    // `Differentiable` protocol requirements all have default implementations
    // when `Self` conforms to `AdditiveArithmetic`, so `Differentiable`
    // derived conformances will no longer be necessary.
    if (!conformsToDifferentiable) {
      TC.diagnose(loc,
                  diag::differentiable_nondiff_type_implicit_noderivative_fixit,
                  vd->getName(), nominal->getName(),
                  nominalCanDeriveAdditiveArithmetic)
          .fixItInsert(loc, "@noDerivative ");
      continue;
    }
    TC.diagnose(loc,
                diag::differentiable_let_property_implicit_noderivative_fixit,
                vd->getName(), nominal->getName(),
                nominalCanDeriveAdditiveArithmetic)
        .fixItInsert(loc, "@noDerivative ");
  }
}

/// Get or synthesize `TangentVector` struct type.
static Type
getOrSynthesizeTangentVectorStructType(DerivedConformance &derived) {
  auto &TC = derived.TC;
  auto *parentDC = derived.getConformanceContext();
  auto *nominal = derived.Nominal;
  auto &C = nominal->getASTContext();

  // Get or synthesize `TangentVector` struct.
  auto *tangentStruct =
      getOrSynthesizeTangentVectorStruct(derived, C.Id_TangentVector);
  if (!tangentStruct)
    return nullptr;
  // Check and emit warnings for implicit `@noDerivative` members.
  checkAndDiagnoseImplicitNoDerivative(TC, nominal, parentDC);
  // Add `TangentVector` typealias for `TangentVector` struct.
  addAssociatedTypeAliasDecl(C.Id_TangentVector,
                             tangentStruct, tangentStruct, TC);
  TC.validateDecl(tangentStruct);

  // Sanity checks for synthesized struct.
  assert(DerivedConformance::canDeriveAdditiveArithmetic(tangentStruct,
                                                         parentDC) &&
         "Should be able to derive `AdditiveArithmetic`");
  assert(DerivedConformance::canDeriveDifferentiable(tangentStruct, parentDC) &&
         "Should be able to derive `Differentiable`");

  // Return the `TangentVector` struct type.
  return parentDC->mapTypeIntoContext(
      tangentStruct->getDeclaredInterfaceType());
}

/// Synthesize the `TangentVector` struct type.
static Type
deriveDifferentiable_TangentVectorStruct(DerivedConformance &derived) {
  auto &TC = derived.TC;
  auto *parentDC = derived.getConformanceContext();
  auto *nominal = derived.Nominal;
  auto &C = nominal->getASTContext();

  // Get all stored properties for differentation.
  SmallVector<VarDecl *, 16> diffProperties;
  getStoredPropertiesForDifferentiation(nominal, parentDC, diffProperties);

  // If any member has an invalid `TangentVector` type, return nullptr.
  for (auto *member : diffProperties)
    if (!getTangentVectorType(member, parentDC))
      return nullptr;

  // Prevent re-synthesis during repeated calls.
  // FIXME: Investigate why this is necessary to prevent duplicate synthesis.
  auto lookup = nominal->lookupDirect(C.Id_TangentVector);
  if (lookup.size() == 1)
    if (auto *structDecl = convertToStructDecl(lookup.front()))
      if (structDecl->isImplicit())
        return structDecl->getDeclaredInterfaceType();

  // Check whether at least one `@noDerivative` stored property exists.
  unsigned numStoredProperties =
      std::distance(nominal->getStoredProperties().begin(),
                    nominal->getStoredProperties().end());
  bool hasNoDerivativeStoredProp = diffProperties.size() != numStoredProperties;

  // Check conditions for returning `Self`.
  // - `Self` is not a class type.
  // - No `@noDerivative` stored properties exist.
  // - All stored properties must have `TangentVector` type equal to `Self`.
  // - Parent type must also conform to `AdditiveArithmetic`.
  bool allMembersAssocTypeEqualsSelf =
      llvm::all_of(diffProperties, [&](VarDecl *member) {
        auto memberAssocType = getTangentVectorType(member, parentDC);
        return member->getType()->isEqual(memberAssocType);
      });

  auto *addArithProto = C.getProtocol(KnownProtocolKind::AdditiveArithmetic);
  auto nominalConformsToAddArith =
      TC.conformsToProtocol(parentDC->getSelfTypeInContext(), addArithProto,
                            parentDC, None);

  // Return `Self` if conditions are met.
  if (!hasNoDerivativeStoredProp && !nominal->getSelfClassDecl() &&
      allMembersAssocTypeEqualsSelf && nominalConformsToAddArith) {
    auto selfType = parentDC->getSelfTypeInContext();
    auto *aliasDecl =
        new (C) TypeAliasDecl(SourceLoc(), SourceLoc(), C.Id_TangentVector,
                              SourceLoc(), {}, parentDC);
    aliasDecl->setUnderlyingType(selfType);
    aliasDecl->setImplicit();
    aliasDecl->copyFormalAccessFrom(nominal, /*sourceIsParentContext*/ true);
    aliasDecl->setValidationToChecked();
    TC.validateDecl(aliasDecl);
    derived.addMembersToConformanceContext({aliasDecl});
    C.addSynthesizedDecl(aliasDecl);
    return selfType;
  }

  // Otherwise, get or synthesize `TangentVector` struct type.
  return getOrSynthesizeTangentVectorStructType(derived);
}

ValueDecl *DerivedConformance::deriveDifferentiable(ValueDecl *requirement) {
  // Diagnose conformances in disallowed contexts.
  if (checkAndDiagnoseDisallowedContext(requirement))
    return nullptr;
  if (requirement->getBaseName() == TC.Context.Id_move)
    return deriveDifferentiable_move(*this);
  TC.diagnose(requirement->getLoc(), diag::broken_differentiable_requirement);
  return nullptr;
}

Type DerivedConformance::deriveDifferentiable(AssociatedTypeDecl *requirement) {
  // Diagnose conformances in disallowed contexts.
  if (checkAndDiagnoseDisallowedContext(requirement))
    return nullptr;
  if (requirement->getBaseName() == TC.Context.Id_TangentVector)
    return deriveDifferentiable_TangentVectorStruct(*this);
  TC.diagnose(requirement->getLoc(), diag::broken_differentiable_requirement);
  return nullptr;
}

/// Derive a EuclideanDifferentiable requirement for a nominal type.
///
/// \returns the derived member, which will also be added to the type.
ValueDecl *DerivedConformance::deriveEuclideanDifferentiable(
    ValueDecl *requirement) {
  // Diagnose conformances in disallowed contexts.
  if (checkAndDiagnoseDisallowedContext(requirement))
    return nullptr;
  if (requirement->getFullName() == TC.Context.Id_differentiableVectorView)
    return deriveEuclideanDifferentiable_differentiableVectorView(*this);
  TC.diagnose(requirement->getLoc(),
              diag::broken_euclidean_differentiable_requirement);
  return nullptr;
}
