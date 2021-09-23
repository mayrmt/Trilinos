// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef MUELU_MUELUSMOOTHER_DEF_HPP
#define MUELU_MUELUSMOOTHER_DEF_HPP

#include "MueLu_MueLuSmoother_decl.hpp"

#include "MueLu_Hierarchy.hpp"
#include "MueLu_HierarchyManager.hpp"
#include "MueLu_Level.hpp"

#include <Teuchos_ParameterList.hpp>

#include <Xpetra_Matrix.hpp>
#include <Xpetra_MultiVector.hpp>

// #include "MueLu_CreateXpetraPreconditioner.hpp"

namespace MueLu {

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::MueLuSmoother(const Teuchos::ParameterList& paramList)
  {

  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<MueLu::SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
  MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Copy() const {
    RCP<MueLuSmoother> smoother = rcp(new MueLuSmoother(*this));
    smoother->SetParameterList(this->GetParameterList());
    return Teuchos::rcp_dynamic_cast<MueLu::SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal, Node>>(smoother);
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& currentLevel) const
  {

  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Setup(Level& currentLevel)
  {
#include "MueLu_UseShortNames.hpp"
    Teuchos::ParameterList mueluList;
    A_ = Factory::Get<RCP<Matrix>>(currentLevel, "A");

    const std::string label = "label";
    RCP<HierarchyManager> mueLuFactory;
    hierarchy_ = mueLuFactory->CreateHierarchy(label);

    hierarchy_->setlib(A_->getDomainMap()->lib());
    hierarchy_->GetLevel(0)->Set("A", A_);
    hierarchy_->SetProcRankVerbose(A_->getDomainMap()->getComm()->getRank());

  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Apply(
      MultiVector& X, const MultiVector& B, bool InitialGuessIsZero) const
  {

  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  size_t MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getNodeSmootherComplexity() const
  {

    // ToDo: Does it make sense to return the operator complexity of the underlying MueLu hierarchy?
    return Teuchos::OrdinalTraits<size_t>::invalid();
  }

} // namespace MueLu

#endif // MUELU_MUELUSMOOTHER_DEF_HPP