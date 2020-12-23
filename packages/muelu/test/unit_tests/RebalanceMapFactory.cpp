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
#include <vector>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_ScalarTraits.hpp>

#include <Xpetra_Map.hpp>
#include <Xpetra_MapFactory.hpp>

#include <MueLu_TestHelpers.hpp>
#include <MueLu_Version.hpp>

#include <MueLu_RebalanceMapFactory.hpp>
#include <MueLu_RepartitionHeuristicFactory.hpp>
#include <MueLu_RepartitionFactory.hpp>
#include <MueLu_ZoltanInterface.hpp>

namespace MueLuTests {

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(RebalanceMapFactory, Constructor, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
    out << "version: " << MueLu::Version() << std::endl;

    RCP<RebalanceMapFactory> rebalanceMapFactory = rcp(new RebalanceMapFactory());
    TEST_ASSERT(!rebalanceMapFactory.is_null());

  }

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(RebalanceMapFactory, RebalanceMap, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);
    out << "version: " << MueLu::Version() << std::endl;

    RCP<const Teuchos::Comm<int>> comm = Parameters::getDefaultComm();
    const int numProcs = comm->getSize();
    const int myRank = comm->getRank();

    if (numProcs != 4) { // Test is tailored to numProcs == 4. Skip it otherwise.
      std::cout << "\nThis is a rebalancing test. It must be run on 4 processors.\n" << std::endl;
      return;
    }

    Xpetra::UnderlyingLib lib = TestHelpers::Parameters::getLib();

    const std::string mapName = "input map";
    const LO numLocalElements = 10;
    const GO numGlobalElements = Teuchos::as<GO>(numLocalElements * comm->getSize());

    // Create source and target map for the fake rebalancing importer
    RCP<const Map> sourceMap = MapFactory::Build(lib, numGlobalElements, Teuchos::ScalarTraits<GO>::zero(), comm);

    Array<GO> targetGIDs;
    if (myRank == 0)
    {
      for (GO i = 0; i < Teuchos::as<GO>(2*numLocalElements); ++i)
        targetGIDs.push_back(i);
    }
    else if (myRank == 1)
    {
      for (GO i = 0; i < Teuchos::as<GO>(numLocalElements) - 7; ++i)
        targetGIDs.push_back(Teuchos::as<GO>(2*numLocalElements) + i);
    }
    else if (myRank == 2)
    {
      for (GO i = 0; i < Teuchos::as<GO>(numLocalElements - 7); ++i)
        targetGIDs.push_back(Teuchos::as<GO>(3*numLocalElements) - 7 + i);
    }
    else if (myRank == 3)
    {
      for (GO i = 0; i < Teuchos::as<GO>(14); ++i)
        targetGIDs.push_back(26 + i);
    }
    RCP<const Map> targetMap = MapFactory::Build(lib, numGlobalElements, targetGIDs, Teuchos::ScalarTraits<GO>::zero(), comm);
    TEST_ASSERT(!targetMap.is_null());

    RCP<const Import> importer = ImportFactory::Build(sourceMap, targetMap);
    TEST_ASSERT(!importer.is_null());

    const int levelID = 3; // Pick a random level ID
    Level myLevel;
    myLevel.SetLevelID(levelID);
    TEST_EQUALITY_CONST(myLevel.GetLevelID(), levelID);

    myLevel.Set("Importer", importer, MueLu::NoFactory::get());
    TEST_ASSERT(myLevel.IsAvailable("Importer", MueLu::NoFactory::get()));

    // Create a map (portion of sourceMap) to be rebalanced
    Array<GO> inputGIDs;
    for (GO dof = 0; dof < numLocalElements-4; ++dof)
      inputGIDs.push_back(myRank*numLocalElements + dof);
    RCP<const Map> inputMap = MapFactory::Build(lib, numProcs * (numLocalElements-4), inputGIDs, Teuchos::ScalarTraits<GO>::zero(), comm);
    TEST_ASSERT(!inputMap.is_null());

    myLevel.Set(mapName, inputMap);
    TEST_ASSERT(myLevel.IsAvailable(mapName, MueLu::NoFactory::get()));

    RCP<FactoryManager> factoryManager = rcp(new FactoryManager());
    factoryManager->SetKokkosRefactor(false);
    factoryManager->SetFactory(mapName, MueLu::NoFactory::getRCP());
    myLevel.SetFactoryManager(factoryManager);

    RCP<RebalanceMapFactory> rebalanceMapFactory = rcp(new RebalanceMapFactory());
    TEST_ASSERT(!rebalanceMapFactory.is_null());

    rebalanceMapFactory->SetParameter("map: factory", Teuchos::ParameterEntry(mapName));
    rebalanceMapFactory->SetParameter("map: name", Teuchos::ParameterEntry(mapName));
    rebalanceMapFactory->SetParameter("repartition: use subcommunicators", Teuchos::ParameterEntry(bool(false)));
    rebalanceMapFactory->SetFactory("Importer", MueLu::NoFactory::getRCP());

    myLevel.Request(mapName, MueLu::NoFactory::get());
    myLevel.Request("Importer", MueLu::NoFactory::get(), rebalanceMapFactory.get());
    myLevel.Request(*rebalanceMapFactory); // This calls DeclareInput() on rebalanceMapFactory

    TEST_ASSERT(myLevel.IsRequested(mapName, MueLu::NoFactory::get()));
    TEST_ASSERT(myLevel.IsRequested("Importer", MueLu::NoFactory::get()));

    rebalanceMapFactory->Build(myLevel);

    RCP<const Map> rebalancedMap = myLevel.Get<RCP<const Map>>(mapName, MueLu::NoFactory::get());
    TEST_ASSERT(!rebalancedMap.is_null());
    TEST_EQUALITY_CONST(rebalancedMap->getGlobalNumElements(), inputMap->getGlobalNumElements());

    ArrayView<const GO> rebalancedGIDs = rebalancedMap->getNodeElementList();
    if (myRank == 0)
    {
      TEST_EQUALITY_CONST(rebalancedGIDs[0], 0);
      TEST_EQUALITY_CONST(rebalancedGIDs[1], 1);
      TEST_EQUALITY_CONST(rebalancedGIDs[2], 2);
      TEST_EQUALITY_CONST(rebalancedGIDs[3], 3);
      TEST_EQUALITY_CONST(rebalancedGIDs[4], 4);
      TEST_EQUALITY_CONST(rebalancedGIDs[5], 5);

      TEST_EQUALITY_CONST(rebalancedGIDs[6], 10);
      TEST_EQUALITY_CONST(rebalancedGIDs[7], 11);
      TEST_EQUALITY_CONST(rebalancedGIDs[8], 12);
      TEST_EQUALITY_CONST(rebalancedGIDs[9], 13);
      TEST_EQUALITY_CONST(rebalancedGIDs[10], 14);
      TEST_EQUALITY_CONST(rebalancedGIDs[11], 15);
    }
    else if (myRank == 1)
    {
      TEST_EQUALITY_CONST(rebalancedGIDs[0], 20);
      TEST_EQUALITY_CONST(rebalancedGIDs[1], 21);
      TEST_EQUALITY_CONST(rebalancedGIDs[2], 22);
    }
    else if (myRank == 2)
    {
      TEST_EQUALITY_CONST(rebalancedGIDs[0], 23);
      TEST_EQUALITY_CONST(rebalancedGIDs[1], 24);
      TEST_EQUALITY_CONST(rebalancedGIDs[2], 25);
    }
    else if (myRank == 3)
    {
      TEST_EQUALITY_CONST(rebalancedGIDs[0], 30);
      TEST_EQUALITY_CONST(rebalancedGIDs[1], 31);
      TEST_EQUALITY_CONST(rebalancedGIDs[2], 32);
      TEST_EQUALITY_CONST(rebalancedGIDs[3], 33);
      TEST_EQUALITY_CONST(rebalancedGIDs[4], 34);
      TEST_EQUALITY_CONST(rebalancedGIDs[5], 35);
    }
  }

#define MUELU_ETI_GROUP(Scalar, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(RebalanceMapFactory,Constructor,Scalar,LO,GO,Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(RebalanceMapFactory,RebalanceMap,Scalar,LO,GO,Node)

#include <MueLu_ETI_4arg.hpp>

} // namespace MueLuTests
