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
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>

#include <MueLu_TestHelpers.hpp>
#include <MueLu_Version.hpp>

#include <Xpetra_Map.hpp>
#include <Xpetra_MapFactory.hpp>

#include <MueLu_MapWrapperFactory.hpp>

namespace MueLuTests {

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(MapWrapperFactory, Constructor, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
    out << "version: " << MueLu::Version() << std::endl;

    RCP<MapWrapperFactory> mapWrapperFactory = rcp(new MapWrapperFactory());
    TEST_EQUALITY(mapWrapperFactory != Teuchos::null, true);
  } //Constructor

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(MapWrapperFactory, WrapExistingMap, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
    out << "version: " << MueLu::Version() << std::endl;

    // Create a small input map to be wrapped by the MapWrapperFactory
    const GO numGlobalElements = 10;
    RCP<const Map> inputMap = MapFactory::Build(TestHelpers::Parameters::getLib(), numGlobalElements,
        Teuchos::ScalarTraits<GO>::zero(), TestHelpers::Parameters::getDefaultComm());

    // Create the MapWrapperFactory
    const std::string mapName = "Simple Map";
    RCP<MapWrapperFactory> mapWrapperFactory = rcp(new MapWrapperFactory());
    mapWrapperFactory->SetParameter("map: name", Teuchos::ParameterEntry(mapName));
    mapWrapperFactory->SetParameter("map: object to wrap", Teuchos::ParameterEntry(inputMap));

    Level currentLevel;
    currentLevel.Request(mapName, mapWrapperFactory.get());

    mapWrapperFactory->Build(currentLevel);

    // Extract the output map for result testing
    RCP<const Map> outputMap = currentLevel.Get<RCP<const Map>>(mapName, mapWrapperFactory.get());

    TEST_ASSERT(!outputMap.is_null());
    TEST_ASSERT(outputMap->isSameAs(*inputMap));
  } // WrapExistingMap

#define MUELU_ETI_GROUP(Scalar,LocalOrdinal,GlobalOrdinal,Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(MapWrapperFactory, Constructor, Scalar, LocalOrdinal, GlobalOrdinal, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(MapWrapperFactory, WrapExistingMap, Scalar, LocalOrdinal, GlobalOrdinal, Node)

#include <MueLu_ETI_4arg.hpp>

} // namespace MueLuTests
