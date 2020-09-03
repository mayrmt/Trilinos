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

#include <MueLu_AmalgamationFactory.hpp>
#include <MueLu_CoalesceDropFactory.hpp>
#include <MueLu_CoarseMapFactory.hpp>
#include <MueLu_FactoryManager.hpp>
#include <MueLu_Level.hpp>
#include <MueLu_MapTransferFactory.hpp>
#include <MueLu_MapWrapperFactory.hpp>
#include <MueLu_NoFactory.hpp>
#include <MueLu_NullspaceFactory.hpp>
#include <MueLu_TentativePFactory.hpp>
#include <MueLu_UncoupledAggregationFactory.hpp>
#include <MueLu_Utilities.hpp>

#include <Xpetra_Import.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_Map.hpp>
#include <Xpetra_MapFactory.hpp>
#include <Xpetra_Vector.hpp>
#include <Xpetra_VectorFactory.hpp>

namespace MueLuTests {

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(MapTransferFactory, Constructor, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#   include "MueLu_UseShortNames.hpp"
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);

    out << "version: " << MueLu::Version() << std::endl;

    RCP<MapTransferFactory> mapTransferFactory = rcp(new MapTransferFactory());
    TEST_EQUALITY(mapTransferFactory != Teuchos::null, true);
  } // Constructor

  /* This tests coarsens the row map of the fine level operator, so the result from the MapTransferFactory
   * needs to match the domain map of the prolongator.
   *
   * Assume a 1D Poisson discretization.
   */
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(MapTransferFactory, TransferFullMap1D, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#   include "MueLu_UseShortNames.hpp"
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);

    using test_factory = TestHelpers::TestFactory<SC, LO, GO, NO>;

    out << "version: " << MueLu::Version() << std::endl;
    out << "Test transfer of a map with the MapTransferFactory" << std::endl;

    // Manual setup of a two-level hierarchy
    Level fineLevel;
    Level coarseLevel;
    coarseLevel.SetPreviousLevel(Teuchos::rcpFromRef(fineLevel));
    fineLevel.SetLevelID(0);
    coarseLevel.SetLevelID(1);

    // Create a dummy matrix needed to build a prolongator
    const GO nx = 199;
    RCP<Matrix> A = test_factory::Build1DPoisson(nx);
    fineLevel.Set("A", A);

    // Use factory to create a dummy map to be used for the MapTransferFactory
    const std::string mapName = "Dummy Map";
    RCP<MapWrapperFactory> mapWrapperFactory = rcp(new MapWrapperFactory());
    mapWrapperFactory->SetParameter("map: name", Teuchos::ParameterEntry(mapName));
    mapWrapperFactory->SetParameter("map: object to wrap", Teuchos::ParameterEntry(A->getRowMap()));

    // Register the map generating factory as factory in the coarse level
    RCP<FactoryManager> factoryManager = rcp(new FactoryManager());
    factoryManager->SetKokkosRefactor(false);
    factoryManager->SetFactory(mapName, mapWrapperFactory);
    fineLevel.SetFactoryManager(factoryManager);
    coarseLevel.SetFactoryManager(factoryManager);

    // Create a default TentativePFactory required by the MapTransferFactory
    RCP<TentativePFactory> tentativePFact = rcp(new TentativePFactory());

    // Create the MapTransferFactory (the one, we actually want to test here)
    RCP<MapTransferFactory> mapTransferFactory = rcp(new MapTransferFactory());
    mapTransferFactory->SetParameter("map: factory", Teuchos::ParameterEntry(mapName));
    mapTransferFactory->SetParameter("map: name", Teuchos::ParameterEntry(mapName));
    mapTransferFactory->SetFactory("P", tentativePFact);

    // Request the necessary data on both levels
    fineLevel.Request(mapName, mapWrapperFactory.get(), mapTransferFactory.get());
    coarseLevel.Request(mapName, mapWrapperFactory.get());
    coarseLevel.Request("P", tentativePFact.get(), mapTransferFactory.get());
    coarseLevel.Request(*mapTransferFactory); // This calls DeclareInput() on mapTransferFactory

    // Call Build() on all factories in the right order
    mapWrapperFactory->Build(fineLevel);
    tentativePFact->Build(fineLevel, coarseLevel);
    mapTransferFactory->Build(fineLevel, coarseLevel);

    // Get some quantities form levels to perform result checks
    RCP<Matrix> Ptent = coarseLevel.Get<RCP<Matrix>>("P", tentativePFact.get());
    RCP<const Map> coarsenedMap = coarseLevel.Get<RCP<const Map>>(mapName, mapWrapperFactory.get());

    TEST_ASSERT(coarsenedMap->isSameAs(*Ptent->getDomainMap()));
  } // TransferFullMap1D

  /* This tests coarsens the row map of the fine level operator, so the result from the MapTransferFactory
   * needs to match the domain map of the prolongator.
   *
   * Assume a 1D Poisson discretization.
   */
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(MapTransferFactory, TransferPartialMap1D, Scalar, LocalOrdinal, GlobalOrdinal, Node)
  {
#   include "MueLu_UseShortNames.hpp"
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_SCOPE(Scalar,GlobalOrdinal,Node);

    using test_factory = TestHelpers::TestFactory<SC, LO, GO, NO>;

    out << "version: " << MueLu::Version() << std::endl;
    out << "Test transfer of a map with the MapTransferFactory" << std::endl;

    RCP<const Teuchos::Comm<int> > comm = TestHelpers::Parameters::getDefaultComm();

    // Manual setup of a two-level hierarchy
    Level fineLevel;
    Level coarseLevel;
    coarseLevel.SetPreviousLevel(Teuchos::rcpFromRef(fineLevel));
    fineLevel.SetLevelID(0);
    coarseLevel.SetLevelID(1);

    // Create a dummy matrix needed to build a prolongator
    const GO nx = 49;
    RCP<Matrix> A = test_factory::Build1DPoisson(4*nx);
    fineLevel.Set("A", A);

    // Extract a subset of A->getRowMap()'s GIDs to create the map to be transferred
    ArrayView<const GO> allRowGIDs = A->getRowMap()->getNodeElementList();
    Array<GO> myMapGIDs;
    for (LO lid = 0; lid < Teuchos::as<LO>(nx); ++lid) {
      if (lid % 3 == 0)
        myMapGIDs.push_back(allRowGIDs[lid]);
    }
    GO gNumFineEntries = 0;
    reduceAll(*comm, Teuchos::REDUCE_SUM, Teuchos::as<GO>(myMapGIDs.size()), Teuchos::outArg(gNumFineEntries));
    RCP<const Map> mapWithHoles = MapFactory::Build(TestHelpers::Parameters::getLib(), gNumFineEntries, myMapGIDs(), Teuchos::ScalarTraits<GO>::zero(), comm);

    // Use factory to create a dummy map to be used for the MapTransferFactory
    const std::string mapName = "Dummy Map";
    RCP<MapWrapperFactory> mapWrapperFactory = rcp(new MapWrapperFactory());
    mapWrapperFactory->SetParameter("map: name", Teuchos::ParameterEntry(mapName));
    mapWrapperFactory->SetParameter("map: object to wrap", Teuchos::ParameterEntry(mapWithHoles));

    // Register the map generating factory as factory in the coarse level
    RCP<FactoryManager> factoryManager = rcp(new FactoryManager());
    factoryManager->SetKokkosRefactor(false);
    factoryManager->SetFactory(mapName, mapWrapperFactory);
    fineLevel.SetFactoryManager(factoryManager);
    coarseLevel.SetFactoryManager(factoryManager);

    // Create a default TentativePFactory required by the MapTransferFactory
    RCP<TentativePFactory> tentativePFact = rcp(new TentativePFactory());

    // Create the MapTransferFactory (the one, we actually want to test here)
    RCP<MapTransferFactory> mapTransferFactory = rcp(new MapTransferFactory());
    mapTransferFactory->SetParameter("map: factory", Teuchos::ParameterEntry(mapName));
    mapTransferFactory->SetParameter("map: name", Teuchos::ParameterEntry(mapName));
    mapTransferFactory->SetFactory("P", tentativePFact);

    // Request the necessary data on both levels
    fineLevel.Request(mapName, mapWrapperFactory.get(), mapTransferFactory.get());
    coarseLevel.Request(mapName, mapWrapperFactory.get());
    coarseLevel.Request("P", tentativePFact.get(), mapTransferFactory.get());
    coarseLevel.Request(*mapTransferFactory); // This calls DeclareInput() on mapTransferFactory

    // Call Build() on all factories in the right order
    mapWrapperFactory->Build(fineLevel);
    tentativePFact->Build(fineLevel, coarseLevel);
    mapTransferFactory->Build(fineLevel, coarseLevel);

    // Get some quantities form levels to perform result checks
    RCP<Matrix> Ptent = coarseLevel.Get<RCP<Matrix>>("P", tentativePFact.get());
    RCP<const Map> fineMap = fineLevel.Get<RCP<const Map>>(mapName, mapWrapperFactory.get()); // map to be transferred
    RCP<const Map> coarsenedMap = coarseLevel.Get<RCP<const Map>>(mapName, mapWrapperFactory.get());

    // Populate a fine level vector w/ ones according to fineMap
    RCP<Vector> fullFineVec = VectorFactory::Build(Ptent->getRangeMap(), true);
    RCP<Vector> partialFineVec = VectorFactory::Build(fineMap, true);
    partialFineVec->putScalar(Teuchos::ScalarTraits<Scalar>::one());
    RCP<Import> fineImporter = ImportFactory::Build(fineMap, fullFineVec->getMap());
    fullFineVec->doImport(*partialFineVec, *fineImporter, Xpetra::INSERT);

    // Restrict to coarse level manually
    RCP<Vector> fullCoarseVec = VectorFactory::Build(Ptent->getDomainMap(), true);
    Ptent->apply(*fullFineVec, *fullCoarseVec, Teuchos::TRANS);

    // Reconstruct coarse map for result checking
    ArrayRCP<const Scalar> coarseVecEntries = fullCoarseVec->getData(0);
    Array<GO> myCoarseGIDs;
    for (LO lid = 0; lid < coarseVecEntries.size(); ++lid) {
      if (coarseVecEntries[lid] > Teuchos::ScalarTraits<Scalar>::zero())
        myCoarseGIDs.push_back(Ptent->getDomainMap()->getGlobalElement(lid));
    }
    GO gNumCoarseEntries = 0;
    reduceAll(*comm, Teuchos::REDUCE_SUM, Teuchos::as<GO>(myCoarseGIDs.size()), Teuchos::outArg(gNumCoarseEntries));
    RCP<const Map> mapForComparison = MapFactory::Build(fineMap->lib(), gNumCoarseEntries, myCoarseGIDs, Teuchos::ScalarTraits<GO>::zero(), comm);

    TEST_ASSERT(coarsenedMap->isSameAs(*mapForComparison));
  } // TransferPartialMap1D

  #  define MUELU_ETI_GROUP(Scalar, LO, GO, Node) \
      TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(MapTransferFactory,Constructor,Scalar,LO,GO,Node) \
      TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(MapTransferFactory,TransferFullMap1D,Scalar,LO,GO,Node) \
      TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(MapTransferFactory,TransferPartialMap1D,Scalar,LO,GO,Node)


#include <MueLu_ETI_4arg.hpp>

} // namespace MueLuTests