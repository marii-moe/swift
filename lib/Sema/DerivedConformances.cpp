//===--- DerivedConformances.cpp - Derived conformance utilities ----------===//
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

#include "TypeChecker.h"
#include "swift/AST/Decl.h"
#include "swift/AST/Stmt.h"
#include "swift/AST/Expr.h"
#include "swift/AST/Pattern.h"
#include "swift/AST/ParameterList.h"
#include "swift/AST/ProtocolConformance.h"
#include "swift/AST/Types.h"
#include "swift/ClangImporter/ClangModule.h"
#include "DerivedConformances.h"

using namespace swift;

DerivedConformance::DerivedConformance(TypeChecker &tc, Decl *conformanceDecl,
                                       NominalTypeDecl *nominal,
                                       ProtocolDecl *protocol)
    : TC(tc), ConformanceDecl(conformanceDecl), Nominal(nominal),
      Protocol(protocol) {
  assert(getConformanceContext()->getSelfNominalTypeDecl() == nominal);
}

DeclContext *DerivedConformance::getConformanceContext() const {
  return cast<DeclContext>(ConformanceDecl);
}

void DerivedConformance::addMembersToConformanceContext(
    ArrayRef<Decl *> children) {
  auto IDC = cast<IterableDeclContext>(ConformanceDecl);
  for (auto child : children) {
    IDC->addMember(child);
  }
}

Type DerivedConformance::getProtocolType() const {
  return Protocol->getDeclaredType();
}

bool DerivedConformance::derivesProtocolConformance(DeclContext *DC,
                                                    NominalTypeDecl *Nominal,
                                                    ProtocolDecl *Protocol) {
  // Only known protocols can be derived.
  auto knownProtocol = Protocol->getKnownProtocolKind();
  if (!knownProtocol)
    return false;

  if (*knownProtocol == KnownProtocolKind::Hashable) {
    // We can always complete a partial Hashable implementation, and we can
    // synthesize a full Hashable implementation for structs and enums with
    // Hashable components.
    return canDeriveHashable(Nominal);
  }

  // SWIFT_ENABLE_TENSORFLOW
  if (*knownProtocol == KnownProtocolKind::AdditiveArithmetic)
    return canDeriveAdditiveArithmetic(Nominal, DC);

  // SWIFT_ENABLE_TENSORFLOW
  if (*knownProtocol == KnownProtocolKind::PointwiseMultiplicative)
    return canDerivePointwiseMultiplicative(Nominal, DC);

  // SWIFT_ENABLE_TENSORFLOW
  if (*knownProtocol == KnownProtocolKind::ElementaryFunctions)
    return canDeriveElementaryFunctions(Nominal, DC);

  // SWIFT_ENABLE_TENSORFLOW
  if (*knownProtocol == KnownProtocolKind::KeyPathIterable)
    return canDeriveKeyPathIterable(Nominal);

  // SWIFT_ENABLE_TENSORFLOW
  if (*knownProtocol == KnownProtocolKind::TensorArrayProtocol)
    return canDeriveTensorArrayProtocol(Nominal, DC);

  // SWIFT_ENABLE_TENSORFLOW
  if (*knownProtocol == KnownProtocolKind::TensorGroup)
    return canDeriveTensorGroup(Nominal, DC);

  // SWIFT_ENABLE_TENSORFLOW
  if (*knownProtocol == KnownProtocolKind::VectorProtocol)
    return canDeriveVectorProtocol(Nominal, DC);

  // SWIFT_ENABLE_TENSORFLOW
  if (*knownProtocol == KnownProtocolKind::Differentiable)
    return canDeriveDifferentiable(Nominal, DC);

  // SWIFT_ENABLE_TENSORFLOW
  if (*knownProtocol == KnownProtocolKind::EuclideanDifferentiable)
    return canDeriveEuclideanDifferentiable(Nominal, DC);

  if (auto *enumDecl = dyn_cast<EnumDecl>(Nominal)) {
    switch (*knownProtocol) {
        // The presence of a raw type is an explicit declaration that
        // the compiler should derive a RawRepresentable conformance.
      case KnownProtocolKind::RawRepresentable:
        return enumDecl->hasRawType();

        // Enums without associated values can implicitly derive Equatable
        // conformance.
      case KnownProtocolKind::Equatable:
        return canDeriveEquatable(DC, Nominal);

        // "Simple" enums without availability attributes can explicitly derive
        // a CaseIterable conformance.
        //
        // FIXME: Lift the availability restriction.
      case KnownProtocolKind::CaseIterable:
        return !enumDecl->hasPotentiallyUnavailableCaseValue()
            && enumDecl->hasOnlyCasesWithoutAssociatedValues();

        // @objc enums can explicitly derive their _BridgedNSError conformance.
      case KnownProtocolKind::BridgedNSError:
        return enumDecl->isObjC() && enumDecl->hasCases()
            && enumDecl->hasOnlyCasesWithoutAssociatedValues();

        // Enums without associated values and enums with a raw type of String
        // or Int can explicitly derive CodingKey conformance.
      case KnownProtocolKind::CodingKey: {
        Type rawType = enumDecl->getRawType();
        if (rawType) {
          auto parentDC = enumDecl->getDeclContext();
          ASTContext &C = parentDC->getASTContext();

          auto nominal = rawType->getAnyNominal();
          return nominal == C.getStringDecl() || nominal == C.getIntDecl();
        }

        // hasOnlyCasesWithoutAssociatedValues will return true for empty enums;
        // empty enumas are allowed to conform as well.
        return enumDecl->hasOnlyCasesWithoutAssociatedValues();
      }

      default:
        return false;
    }
  } else if (isa<StructDecl>(Nominal) || isa<ClassDecl>(Nominal)) {
    // Structs and classes can explicitly derive Encodable and Decodable
    // conformance (explicitly meaning we can synthesize an implementation if
    // a type conforms manually).
    if (*knownProtocol == KnownProtocolKind::Encodable ||
        *knownProtocol == KnownProtocolKind::Decodable) {
      // FIXME: This is not actually correct. We cannot promise to always
      // provide a witness here for all structs and classes. Unfortunately,
      // figuring out whether this is actually possible requires much more
      // context -- a TypeChecker and the parent decl context at least -- and is
      // tightly coupled to the logic within DerivedConformance.
      // This unfortunately means that we expect a witness even if one will not
      // be produced, which requires DerivedConformance::deriveCodable to output
      // its own diagnostics.
      return true;
    }

    // Structs can explicitly derive Equatable conformance.
    if (isa<StructDecl>(Nominal)) {
      switch (*knownProtocol) {
        case KnownProtocolKind::Equatable:
          return canDeriveEquatable(DC, Nominal);
        default:
          return false;
      }
    }
  }
  return false;
}

void DerivedConformance::tryDiagnoseFailedDerivation(DeclContext *DC,
                                                     NominalTypeDecl *nominal,
                                                     ProtocolDecl *protocol) {
  auto knownProtocol = protocol->getKnownProtocolKind();
  if (!knownProtocol)
    return;

  if (*knownProtocol == KnownProtocolKind::Equatable) {
    tryDiagnoseFailedEquatableDerivation(DC, nominal);
  }

  if (*knownProtocol == KnownProtocolKind::Hashable) {
    tryDiagnoseFailedHashableDerivation(DC, nominal);
  }
}

ValueDecl *DerivedConformance::getDerivableRequirement(NominalTypeDecl *nominal,
                                                       ValueDecl *requirement) {
  // Note: whenever you update this function, also update
  // TypeChecker::deriveProtocolRequirement.
  ASTContext &ctx = nominal->getASTContext();
  auto name = requirement->getFullName();

  // Local function that retrieves the requirement with the same name as
  // the provided requirement, but within the given known protocol.
  // SWIFT_ENABLE_TENSORFLOW
  auto getRequirement = [&](KnownProtocolKind kind,
                            llvm::function_ref<bool(ValueDecl *)> filter =
                                nullptr) -> ValueDecl * {
    // Dig out the protocol.
    auto proto = ctx.getProtocol(kind);
    if (!proto) return nullptr;

    if (auto conformance = TypeChecker::conformsToProtocol(
            nominal->getDeclaredInterfaceType(), proto, nominal,
            ConformanceCheckFlags::SkipConditionalRequirements)) {
      auto DC = conformance->getConcrete()->getDeclContext();
      // Check whether this nominal type derives conformances to the protocol.
      if (!DerivedConformance::derivesProtocolConformance(DC, nominal, proto))
        return nullptr;
    }

    // Retrieve the requirement.
    auto results = proto->lookupDirect(name);
    // SWIFT_ENABLE_TENSORFLOW
    // Filter requirements, if `filter` function is specified.
    if (filter) {
      llvm::erase_if(results, [&](ValueDecl *v) {
        return !isa<ProtocolDecl>(v->getDeclContext()) ||
               !v->isProtocolRequirement() || !filter(v);
      });
    }
    return results.empty() ? nullptr : results.front();
  };

  // Properties.
  if (isa<VarDecl>(requirement)) {
    // RawRepresentable.rawValue
    if (name.isSimpleName(ctx.Id_rawValue))
      return getRequirement(KnownProtocolKind::RawRepresentable);

    // Hashable.hashValue
    if (name.isSimpleName(ctx.Id_hashValue))
      return getRequirement(KnownProtocolKind::Hashable);

    // CaseIterable.allValues
    if (name.isSimpleName(ctx.Id_allCases))
      return getRequirement(KnownProtocolKind::CaseIterable);

    // _BridgedNSError._nsErrorDomain
    if (name.isSimpleName(ctx.Id_nsErrorDomain))
      return getRequirement(KnownProtocolKind::BridgedNSError);

    // CodingKey.stringValue
    if (name.isSimpleName(ctx.Id_stringValue))
      return getRequirement(KnownProtocolKind::CodingKey);

    // CodingKey.intValue
    if (name.isSimpleName(ctx.Id_intValue))
      return getRequirement(KnownProtocolKind::CodingKey);

    // SWIFT_ENABLE_TENSORFLOW
    // AdditiveArithmetic.zero
    if (name.isSimpleName(ctx.Id_zero))
      return getRequirement(KnownProtocolKind::AdditiveArithmetic);

    // SWIFT_ENABLE_TENSORFLOW
    // EuclideanDifferentiable.differentiableVectorView
    if (name.isSimpleName(ctx.Id_differentiableVectorView))
      return getRequirement(KnownProtocolKind::EuclideanDifferentiable);

    // SWIFT_ENABLE_TENSORFLOW
    // PointwiseMultiplicative.one
    if (name.isSimpleName(ctx.Id_one))
      return getRequirement(KnownProtocolKind::PointwiseMultiplicative);

    // SWIFT_ENABLE_TENSORFLOW
    // PointwiseMultiplicative.reciprocal
    if (name.isSimpleName(ctx.Id_reciprocal))
      return getRequirement(KnownProtocolKind::PointwiseMultiplicative);

    // SWIFT_ENABLE_TENSORFLOW
    // KeyPathIterable.allKeyPaths
    if (name.isSimpleName(ctx.Id_allKeyPaths))
      return getRequirement(KnownProtocolKind::KeyPathIterable);

    // SWIFT_ENABLE_TENSORFLOW
    // TensorArrayProtocol._tensorHandleCount
    if (name.isSimpleName(ctx.Id_tensorHandleCount))
      return getRequirement(KnownProtocolKind::TensorArrayProtocol);
    
    // SWIFT_ENABLE_TENSORFLOW
    // TensorArrayProtocol._typeList
    if (name.isSimpleName(ctx.Id_typeList) && !requirement->isStatic())
      return getRequirement(KnownProtocolKind::TensorArrayProtocol);

    // SWIFT_ENABLE_TENSORFLOW
    // TensorGroup._typeList
    if (name.isSimpleName(ctx.Id_typeList))
      return getRequirement(KnownProtocolKind::TensorGroup);

    return nullptr;
  }

  // Functions.
  if (auto func = dyn_cast<FuncDecl>(requirement)) {
    if (func->isOperator() && name.getBaseName() == "==")
      return getRequirement(KnownProtocolKind::Equatable);

    // Encodable.encode(to: Encoder)
    if (name.isCompoundName() && name.getBaseName() == ctx.Id_encode) {
      auto argumentNames = name.getArgumentNames();
      if (argumentNames.size() == 1 && argumentNames[0] == ctx.Id_to)
        return getRequirement(KnownProtocolKind::Encodable);
    }

    // Hashable.hash(into: inout Hasher)
    if (name.isCompoundName() && name.getBaseName() == ctx.Id_hash) {
      auto argumentNames = name.getArgumentNames();
      if (argumentNames.size() == 1 && argumentNames[0] == ctx.Id_into)
        return getRequirement(KnownProtocolKind::Hashable);
    }

    // SWIFT_ENABLE_TENSORFLOW
    // AdditiveArithmetic.+
    // AdditiveArithmetic.-
    if (func->isOperator() && (name.getBaseName() == "+" ||
                               name.getBaseName() == "-")) {
      auto argumentNames = name.getArgumentNames();
      if (argumentNames.size() == 2)
        return getRequirement(KnownProtocolKind::AdditiveArithmetic);
    }

    // SWIFT_ENABLE_TENSORFLOW
    // PointwiseMultiplicative.(.*)
    if (func->isOperator() && name.getBaseName() == ".*") {
      auto argumentNames = name.getArgumentNames();
      if (argumentNames.size() == 2)
        return getRequirement(KnownProtocolKind::PointwiseMultiplicative);
    }

    // SWIFT_ENABLE_TENSORFLOW
    // ElementaryFunctions requirements
    if (name.isCompoundName()) {
      auto argumentNames = name.getArgumentNames();
      if (argumentNames.size() == 1 && (false
#define ELEMENTARY_FUNCTION_UNARY(ID, NAME) || name.getBaseName() == NAME
#include "DerivedConformanceElementaryFunctions.def"
#undef ELEMENTARY_FUNCTION_UNARY
                                        )) {
        return getRequirement(KnownProtocolKind::ElementaryFunctions);
      }
      if (argumentNames.size() == 2) {
        if (name.getBaseName() == "root")
          return getRequirement(KnownProtocolKind::ElementaryFunctions);
        if (name.getBaseName() == "pow") {
          return getRequirement(
              KnownProtocolKind::ElementaryFunctions,
              [&](ValueDecl *v) {
                auto *funcDecl = dyn_cast<FuncDecl>(v);
                if (!funcDecl)
                  return false;
                return funcDecl->getParameters()->get(1)->getName() ==
                       func->getParameters()->get(1)->getName();
              });
        }
      }
    }

    // SWIFT_ENABLE_TENSORFLOW
    // VectorProtocol.scaled(by:)
    if (name.isCompoundName() && name.getBaseName() == ctx.Id_scaled) {
      auto argumentNames = name.getArgumentNames();
      if (argumentNames.size() == 1 &&
          argumentNames[0] == ctx.getIdentifier("by"))
        return getRequirement(KnownProtocolKind::VectorProtocol);
    }

    // SWIFT_ENABLE_TENSORFLOW
    // VectorProtocol.adding(_:)
    // VectorProtocol.subtracting(_:)
    if (name.isCompoundName() &&
        (name.getBaseName() == ctx.Id_adding ||
         name.getBaseName() == ctx.Id_subtracting)) {
      auto argumentNames = name.getArgumentNames();
      if (argumentNames.size() == 1 && argumentNames[0].empty())
        return getRequirement(KnownProtocolKind::VectorProtocol);
    }

    // SWIFT_ENABLE_TENSORFLOW
    // TensorArrayProtocol._unpackTensorHandles(into:)
    if (name.isCompoundName() && 
        name.getBaseName() == ctx.Id_unpackTensorHandles) {
      auto argumentNames = name.getArgumentNames();
      if (argumentNames.size() == 1 &&
          argumentNames[0] == ctx.getIdentifier("into")) {
        return getRequirement(KnownProtocolKind::TensorArrayProtocol);
      }
    }

    // SWIFT_ENABLE_TENSORFLOW
    // Differentiable.move(along:)
    if (name.isCompoundName() &&
        name.getBaseName() == ctx.Id_move) {
      auto argumentNames = name.getArgumentNames();
      if (argumentNames.size() == 1 &&
          argumentNames[0] == ctx.getIdentifier("along")) {
        return getRequirement(KnownProtocolKind::Differentiable);
      }
    }

    return nullptr;
  }

  // Initializers.
  if (auto ctor = dyn_cast<ConstructorDecl>(requirement)) {
    auto argumentNames = name.getArgumentNames();
    if (argumentNames.size() == 1) {
      if (argumentNames[0] == ctx.Id_rawValue)
        return getRequirement(KnownProtocolKind::RawRepresentable);

      // CodingKey.init?(stringValue:), CodingKey.init?(intValue:)
      if (ctor->getFailability() == OTK_Optional &&
          (argumentNames[0] == ctx.Id_stringValue ||
           argumentNames[0] == ctx.Id_intValue))
        return getRequirement(KnownProtocolKind::CodingKey);

      // Decodable.init(from: Decoder)
      if (argumentNames[0] == ctx.Id_from)
        return getRequirement(KnownProtocolKind::Decodable);

      // SWIFT_ENABLE_TENSORFLOW
      // TensorGroup.init(_owning:)
      if (argumentNames[0] == ctx.getIdentifier("_owning")) {
        return getRequirement(KnownProtocolKind::TensorGroup);
      }
    } else if (argumentNames.size() == 2) {
      // SWIFT_ENABLE_TENSORFLOW
      // TensorArrayProtocol.init(_owning:count)
      if (argumentNames[0] == ctx.getIdentifier("_owning") && 
          argumentNames[1] == ctx.getIdentifier("count")) {
        return getRequirement(KnownProtocolKind::TensorArrayProtocol);
      }
    }

    return nullptr;
  }

  // Associated types.
  if (isa<AssociatedTypeDecl>(requirement)) {
    // RawRepresentable.RawValue
    if (name.isSimpleName(ctx.Id_RawValue))
      return getRequirement(KnownProtocolKind::RawRepresentable);

    // CaseIterable.AllCases
    if (name.isSimpleName(ctx.Id_AllCases))
      return getRequirement(KnownProtocolKind::CaseIterable);

    // SWIFT_ENABLE_TENSORFLOW
    // KeyPathIterable.AllKeyPaths
    if (name.isSimpleName(ctx.Id_AllKeyPaths))
      return getRequirement(KnownProtocolKind::KeyPathIterable);

    // SWIFT_ENABLE_TENSORFLOW
    // Differentiable.TangentVector
    if (name.isSimpleName(ctx.Id_TangentVector))
      return getRequirement(KnownProtocolKind::Differentiable);

    // SWIFT_ENABLE_TENSORFLOW
    // VectorProtocol.VectorSpaceScalar
    if (name.isSimpleName(ctx.Id_VectorSpaceScalar))
      return getRequirement(KnownProtocolKind::VectorProtocol);

    return nullptr;
  }

  return nullptr;
}

DeclRefExpr *
DerivedConformance::createSelfDeclRef(AbstractFunctionDecl *fn) {
  ASTContext &C = fn->getASTContext();

  auto selfDecl = fn->getImplicitSelfDecl();
  return new (C) DeclRefExpr(selfDecl, DeclNameLoc(), /*implicit*/true);
}

AccessorDecl *DerivedConformance::
addGetterToReadOnlyDerivedProperty(VarDecl *property,
                                   Type propertyContextType) {
  auto getter =
    declareDerivedPropertyGetter(property, propertyContextType);

  property->setImplInfo(StorageImplInfo::getImmutableComputed());
  property->setAccessors(SourceLoc(), {getter}, SourceLoc());

  return getter;
}

std::pair<AccessorDecl *, AccessorDecl *>
DerivedConformance::addGetterAndSetterToMutableDerivedProperty(
    VarDecl *property, Type propertyContextType) {
  auto *getter = declareDerivedPropertyGetter(property, propertyContextType);
  auto *setter = declareDerivedPropertySetter(property, propertyContextType);
  property->setImplInfo(StorageImplInfo::getMutableComputed());
  property->setAccessors(SourceLoc(), {getter, setter}, SourceLoc());
  return std::make_pair(getter, setter);
}

AccessorDecl *
DerivedConformance::declareDerivedPropertyGetter(VarDecl *property,
                                                 Type propertyContextType) {
  bool isStatic = property->isStatic();

  auto &C = property->getASTContext();
  auto parentDC = property->getDeclContext();
  ParameterList *params = ParameterList::createEmpty(C);

  Type propertyInterfaceType = property->getInterfaceType();
  
  auto getterDecl = AccessorDecl::create(C,
    /*FuncLoc=*/SourceLoc(), /*AccessorKeywordLoc=*/SourceLoc(),
    AccessorKind::Get, property,
    /*StaticLoc=*/SourceLoc(), StaticSpellingKind::None,
    /*Throws=*/false, /*ThrowsLoc=*/SourceLoc(),
    /*GenericParams=*/nullptr, params,
    TypeLoc::withoutLoc(propertyInterfaceType), parentDC);
  getterDecl->setImplicit();
  getterDecl->setStatic(isStatic);
  getterDecl->setIsTransparent(false);

  // Compute the interface type of the getter.
  if (auto env = parentDC->getGenericEnvironmentOfContext())
    getterDecl->setGenericEnvironment(env);
  getterDecl->computeType();

  getterDecl->copyFormalAccessFrom(property);
  getterDecl->setValidationToChecked();

  C.addSynthesizedDecl(getterDecl);

  return getterDecl;
}

// SWIFT_ENABLE_TENSORFLOW
AccessorDecl *
DerivedConformance::declareDerivedPropertySetter(VarDecl *property,
                                                 Type propertyContextType) {
  bool isStatic = property->isStatic();
  bool isFinal = property->isFinal();

  auto &C = property->getASTContext();
  auto parentDC = property->getDeclContext();

  auto propertyInterfaceType = property->getInterfaceType();
  auto propertyParam = new (C)
    ParamDecl(ParamDecl::Specifier::Default, SourceLoc(), SourceLoc(),
              Identifier(), property->getLoc(), C.getIdentifier("newValue"),
              parentDC);
  propertyParam->setInterfaceType(propertyInterfaceType);

  ParameterList *params = ParameterList::create(C, propertyParam);

  auto setterDecl = AccessorDecl::create(C,
    /*FuncLoc*/ SourceLoc(), /*AccessorKeywordLoc*/ SourceLoc(),
    AccessorKind::Set, property, /*StaticLoc*/ SourceLoc(),
    StaticSpellingKind::None, /*Throws*/ false, /*ThrowsLoc*/ SourceLoc(),
    /*GenericParams*/ nullptr, params, TypeLoc(), parentDC);
  setterDecl->setImplicit();
  setterDecl->setStatic(isStatic);
  // Set mutating if parent is not a class.
  if (!parentDC->getSelfClassDecl())
    setterDecl->setSelfAccessKind(SelfAccessKind::Mutating);

  // If this is supposed to be a final method, mark it as such.
  assert(isFinal || !parentDC->getSelfClassDecl());
  if (isFinal && parentDC->getSelfClassDecl() &&
      !setterDecl->isFinal())
    setterDecl->getAttrs().add(new (C) FinalAttr(/*Implicit*/ true));

  // Compute the interface type of the setter.
  if (auto env = parentDC->getGenericEnvironmentOfContext())
    setterDecl->setGenericEnvironment(env);
  setterDecl->computeType();
  setterDecl->copyFormalAccessFrom(property);
  setterDecl->setValidationToChecked();

  C.addSynthesizedDecl(setterDecl);
  return setterDecl;
}

std::pair<VarDecl *, PatternBindingDecl *>
DerivedConformance::declareDerivedProperty(Identifier name,
                                           Type propertyInterfaceType,
                                           Type propertyContextType,
                                           bool isStatic, bool isFinal) {
  auto &C = TC.Context;
  auto parentDC = getConformanceContext();

  VarDecl *propDecl = new (C) VarDecl(/*IsStatic*/isStatic, VarDecl::Introducer::Var,
                                      /*IsCaptureList*/false, SourceLoc(), name,
                                      parentDC);
  // SWIFT_ENABLE_TENSORFLOW
  // TODO: Upstream this change to master.
  if (isFinal && parentDC->getSelfClassDecl())
    propDecl->getAttrs().add(new (C) FinalAttr(/*Implicit*/ true));
  propDecl->setImplicit();
  propDecl->copyFormalAccessFrom(Nominal, /*sourceIsParentContext*/ true);
  propDecl->setInterfaceType(propertyInterfaceType);
  propDecl->setValidationToChecked();

  Pattern *propPat = new (C) NamedPattern(propDecl, /*implicit*/ true);
  propPat->setType(propertyContextType);

  propPat = TypedPattern::createImplicit(C, propPat, propertyContextType);
  propPat->setType(propertyContextType);

  auto *pbDecl = PatternBindingDecl::createImplicit(
      C, StaticSpellingKind::None, propPat, /*InitExpr*/ nullptr, parentDC);
  return {propDecl, pbDecl};
}

bool DerivedConformance::checkAndDiagnoseDisallowedContext(
    ValueDecl *synthesizing) const {
  // In general, conformances can't be synthesized in extensions across files;
  // but we have to allow it as a special case for Equatable and Hashable on
  // enums with no associated values to preserve source compatibility.
  bool allowCrossfileExtensions = false;
  if (Protocol->isSpecificProtocol(KnownProtocolKind::Equatable) ||
      Protocol->isSpecificProtocol(KnownProtocolKind::Hashable)) {
    auto ED = dyn_cast<EnumDecl>(Nominal);
    allowCrossfileExtensions = ED && ED->hasOnlyCasesWithoutAssociatedValues();
  }

  if (!allowCrossfileExtensions &&
      Nominal->getModuleScopeContext() !=
          getConformanceContext()->getModuleScopeContext()) {
    TC.diagnose(ConformanceDecl->getLoc(),
                diag::cannot_synthesize_in_crossfile_extension,
                getProtocolType());
    TC.diagnose(Nominal->getLoc(), diag::kind_declared_here,
                DescriptiveDeclKind::Type);
    return true;
  }

  // A non-final class can't have an protocol-witnesss initializer in an
  // extension.
  if (auto CD = dyn_cast<ClassDecl>(Nominal)) {
    if (!CD->isFinal() && isa<ConstructorDecl>(synthesizing) &&
        isa<ExtensionDecl>(ConformanceDecl)) {
      TC.diagnose(ConformanceDecl->getLoc(),
                  diag::cannot_synthesize_init_in_extension_of_nonfinal,
                  getProtocolType(), synthesizing->getFullName());
      return true;
    }
  }

  return false;
}
