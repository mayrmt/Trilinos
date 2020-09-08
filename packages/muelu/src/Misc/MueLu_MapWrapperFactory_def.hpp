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
#ifndef MUELU_MAPWRAPPERFACTORY_DEF_HPP
#define MUELU_MAPWRAPPERFACTORY_DEF_HPP

#include "MueLu_MapWrapperFactory_decl.hpp"

#include "MueLu_ConfigDefs.hpp"
#include "MueLu_Level.hpp"
#include "MueLu_SingleLevelFactoryBase.hpp"

#include <Xpetra_Map.hpp>
#include <Xpetra_MapFactory.hpp>

namespace MueLu {

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<const ParameterList> MapWrapperFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const
  {
    RCP<ParameterList> validParams = rcp(new ParameterList());

    validParams->set<std::string>("map: name", "", "Name of map, under which it will stored in the level class");
    validParams->set<RCP<const Map>>("map: object to wrap", Teuchos::null, "Precomputed map object to be wrapped by this factory");

    return validParams;
  };

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void MapWrapperFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(Level& currentLevel) const
  {
    const ParameterList& pL = GetParameterList();
    const std::string mapName = pL.get<std::string>("map: name");
    RCP<const Map> precomputedMap = pL.get<RCP<const Map>>("map: object to wrap");

    TEUCHOS_TEST_FOR_EXCEPTION(mapName=="", Exceptions::InvalidArgument, "Name of map not specified. Please specify a name, which is used to store this map in the level class.");
    TEUCHOS_TEST_FOR_EXCEPTION(precomputedMap.is_null(), Exceptions::RuntimeError, "Map object to be wrapped by MapWrapperFactory is Teuchos::null.");

    currentLevel.Set(mapName, precomputedMap, this);
  };

}

#endif //ifndef MUELU_MAPWRAPPERFACTORY_DEF_HPP
