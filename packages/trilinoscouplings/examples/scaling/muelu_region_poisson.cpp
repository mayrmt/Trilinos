// Standard headers
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <unordered_map>

// TrilinosCouplings headers
#include "TrilinosCouplings_config.h"

// Teuchos headers
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_YamlParameterListHelpers.hpp"

// Belos headers
#include "BelosBiCGStabSolMgr.hpp"
#include "BelosBlockCGSolMgr.hpp"
#include "BelosBlockGmresSolMgr.hpp"
#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosMueLuAdapter.hpp"      // => This header defines Belos::MueLuOp
#include "BelosPseudoBlockCGSolMgr.hpp"
#include "BelosPseudoBlockGmresSolMgr.hpp"
#include "BelosTpetraAdapter.hpp"
#ifdef HAVE_MUELU_TPETRA
#include <BelosTpetraAdapter.hpp>    // => This header defines Belos::TpetraOp
#endif

// MueLu headers
#include "MueLu.hpp"
#include "MueLu_BaseClass.hpp"
#include "MueLu_CreateTpetraPreconditioner.hpp"
#include "MueLu_Level.hpp"
#include "MueLu_MutuallyExclusiveTime.hpp"
#include "MueLu_ParameterListInterpreter.hpp"
#include "MueLu_TpetraOperator.hpp"
#include "MueLu_Utilities.hpp"

#include <Xpetra_IO.hpp>

#ifdef HAVE_MUELU_EXPLICIT_INSTANTIATION
#include <MueLu_ExplicitInstantiation.hpp>
#endif

#ifdef HAVE_MUELU_CUDA
#include "cuda_profiler_api.h"
#endif

// MueLu and Xpetra Tpetra stack
#ifdef HAVE_MUELU_TPETRA
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <KokkosBlas1_abs.hpp>
#include <Tpetra_leftAndOrRightScaleCrsMatrix.hpp>
#include <Tpetra_computeRowAndColumnOneNorms.hpp>
#endif

#if defined(HAVE_MUELU_TPETRA) && defined(HAVE_MUELU_AMESOS2)
#include <Amesos2_config.h>
#include <Amesos2.hpp>
#endif

// Region MG headers
#include "SetupRegionUtilities.hpp"
#include "SetupRegionVector_def.hpp"
#include "SetupRegionMatrix_def.hpp"
#include "SetupRegionHierarchy_def.hpp"
#include "SolveRegionHierarchy_def.hpp"

// Shards headers
#include "Shards_CellTopology.hpp"

// Panzer headers
#include "Panzer_AssemblyEngine.hpp"
#include "Panzer_AssemblyEngine_InArgs.hpp"
#include "Panzer_AssemblyEngine_TemplateManager.hpp"
#include "Panzer_AssemblyEngine_TemplateBuilder.hpp"
#include "Panzer_BlockedTpetraLinearObjFactory.hpp"
#include "Panzer_CheckBCConsistency.hpp"
#include "Panzer_DOFManagerFactory.hpp"
#include "Panzer_FieldManagerBuilder.hpp"
#include "Panzer_GlobalData.hpp"
#include "Panzer_LinearObjFactory.hpp"
#include "Panzer_ResponseEvaluatorFactory_Functional.hpp"
#include "Panzer_ResponseLibrary.hpp"
#include "Panzer_Response_Functional.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_MeshFactory.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "Panzer_STK_ExodusReaderFactory.hpp"
#include "Panzer_STK_WorksetFactory.hpp"
#include "Panzer_STKConnManager.hpp"
#include "Panzer_STK_Version.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "Panzer_STK_Utilities.hpp"
#include "Panzer_TpetraLinearObjContainer.hpp"

// // STK headers
// #include "stk_mesh/base/Types.hpp"

// Percept headers
#include <percept/PerceptMesh.hpp>
#include <adapt/UniformRefinerPattern.hpp>
#include <adapt/UniformRefiner.hpp>

// Tpetra headers
#include "Tpetra_Core.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_FECrsMatrix.hpp"
#include "Tpetra_Import.hpp"
#include "MatrixMarket_Tpetra.hpp"


#include <Xpetra_IO.hpp>

// Include factories for boundary conditions and other Panzer setup
// Most of which is taken from PoissonExample in Panzer_STK
#include "muelu_region_poisson.hpp"

#include "muelu_region_exodus_utils.hpp"

int main(int argc, char *argv[]) {

  // The following typedefs are used so that the two codes will work together.
  // TODO: change everything to use the same types to avoid the following silly code...
  // i.e. check what Drekar does first before making changes
  // Panzer types
  using ST = double;
  using LO = panzer::LocalOrdinal;
  using GO = panzer::GlobalOrdinal;
  using NT = panzer::TpetraNodeType;
  // using OP = Tpetra::Operator<ST,LO,GO,NT>;
  // using MV = Tpetra::MultiVector<ST,LO,GO,NT>;

  // MueLu types
  using Scalar = ST;
  using LocalOrdinal = LO;
  using GlobalOrdinal = GO;
  using Node = NT;
  using SC = Scalar;
  using NO = Node;

  using Teuchos::RCP;
  using Teuchos::Array;
  using Teuchos::ArrayRCP;
  using Teuchos::rcp_dynamic_cast;

  using Map = Xpetra::Map<LO,GO,NO>;
  using MapFactory = Xpetra::MapFactory<LO,GO,NO>;
  using Import = Xpetra::Import<LO,GO,NO>;
  using ImportFactory = Xpetra::ImportFactory<LO,GO,NO>;
  using Vector = Xpetra::Vector<SC,LO,GO,NO>;
  using VectorFactory = Xpetra::VectorFactory<SC,LO,GO,NO>;
  using MultiVector = Xpetra::MultiVector<SC,LO,GO,NO>;
  using MultiVectorFactory = Xpetra::MultiVectorFactory<SC,LO,GO,NO>;
  using Matrix = Xpetra::Matrix<SC,LO,GO,NO>;
  using CrsMatrixWrap = Xpetra::CrsMatrixWrap<SC,LO,GO,NO>;

  using Hierarchy = MueLu::Hierarchy<SC,LO,GO,NO>;

// #include <MueLu_UseShortNames.hpp>

  Kokkos::initialize(argc,argv);
  { // Kokkos scope


    /**********************************************************************************/
    /************************************** SETUP *************************************/
    /**********************************************************************************/

    // TODO: comb back through everything and make sure I'm using MPI comms properly when necessary
    Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
    Teuchos::RCP<const Teuchos::MpiComm<int> > comm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));

    const int numRanks = comm->getSize();
    const int myRank = comm->getRank();

    // Setup output streams
    Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
    fancy->setProcRankAndSize (myRank, numRanks);
    fancy->setOutputToRootOnly(0);
    Teuchos::FancyOStream& out = *fancy;
    // out.setOutputToRootOnly(0); // use out on rank 0

    Teuchos::RCP<Teuchos::FancyOStream> fancydebug = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
    Teuchos::FancyOStream& debug = *fancydebug; // use on all ranks

    out << "Running TrilinosCouplings region multigrid driver on " << numRanks << " ranks... \n";

    // Parse command line arguments
    Teuchos::CommandLineProcessor clp(false);
    std::string exodusFileName        = "";                  clp.setOption("exodus-mesh",           &exodusFileName,          "Exodus hex mesh filename (overrides a pamgen-mesh if both specified)");
    std::string pamgenFileName        = "cylinder.rtp";      clp.setOption("pamgen-mesh",           &pamgenFileName,          "Pamgen hex mesh filename");
    std::string xmlFileName           = "";                  clp.setOption("xml",                   &xmlFileName,             "MueLu parameters from an xml file");
    std::string yamlFileName          = "";                  clp.setOption("yaml",                  &yamlFileName,            "MueLu parameters from a yaml file");
    int mesh_refinements              = 0;                   clp.setOption("mesh-refinements",      &mesh_refinements,        "Uniform mesh refinements");
    bool delete_parent_elements       = false;               clp.setOption("delete-parent-elements", "keep-parent-elements", &delete_parent_elements,"Save the parent elements in the perceptMesh");

    // Multigrid options
    std::string convergenceLog        = "residual_norm.txt"; clp.setOption("convergence-log",       &convergenceLog,        "file in which the convergence history of the linear solver is stored");
    int         maxIts                = 200;                 clp.setOption("its",                   &maxIts,                "maximum number of solver iterations");
    std::string smootherType          = "Jacobi";            clp.setOption("smootherType",          &smootherType,          "smoother to be used: (None | Jacobi | Gauss | Chebyshev)");
    int         smootherIts           = 2;                   clp.setOption("smootherIts",           &smootherIts,           "number of smoother iterations");
    double      smootherDamp          = 0.67;                clp.setOption("smootherDamp",          &smootherDamp,          "damping parameter for the level smoother");
    double      smootherChebyEigRatio = 2.0;                 clp.setOption("smootherChebyEigRatio", &smootherChebyEigRatio, "eigenvalue ratio max/min used to approximate the smallest eigenvalue for Chebyshev relaxation");
    double      smootherChebyBoostFactor = 1.1;              clp.setOption("smootherChebyBoostFactor", &smootherChebyBoostFactor, "boost factor for Chebyshev smoother");
    double      tol                   = 1e-12;               clp.setOption("tol",                   &tol,                   "solver convergence tolerance");
    bool        scaleResidualHist     = true;                clp.setOption("scale", "noscale",      &scaleResidualHist,     "scaled Krylov residual history");
    bool        serialRandom          = false;               clp.setOption("use-serial-random", "no-use-serial-random", &serialRandom, "generate the random vector serially and then broadcast it");
    bool        keepCoarseCoords      = false;               clp.setOption("keep-coarse-coords", "no-keep-coarse-coords", &keepCoarseCoords, "keep coordinates on coarsest level of region hierarchy");
    bool        coarseSolverRebalance = false;               clp.setOption("rebalance-coarse", "no-rebalance-coarse", &coarseSolverRebalance, "rebalance before AMG coarse grid solve");
    int         rebalanceNumPartitions = -1;                 clp.setOption("numPartitions",         &rebalanceNumPartitions, "number of partitions for rebalancing the coarse grid AMG solve");
    std::string coarseSolverType      = "direct";            clp.setOption("coarseSolverType",      &coarseSolverType,      "Type of solver for (composite) coarse level operator (smoother | direct | amg)");
    std::string unstructured          = "{}";                clp.setOption("unstructured",          &unstructured,          "List of ranks to be treated as unstructured, e.g. {0, 2, 5}");
    std::string coarseAmgXmlFile      = "";                  clp.setOption("coarseAmgXml",          &coarseAmgXmlFile,      "Read parameters for AMG as coarse level solve from this xml file.");
    std::string coarseSmootherXMLFile = "";                  clp.setOption("coarseSmootherXML",     &coarseSmootherXMLFile, "File containing the parameters to use with the coarse level smoother.");
    int  cacheSize = 0;                                      clp.setOption("cachesize",               &cacheSize,           "cache size (in KB)"); // what does this do?
    std::string cycleType = "V";                             clp.setOption("cycleType", &cycleType, "{Multigrid cycle type. Possible values: V, W.");
#ifdef HAVE_MUELU_TPETRA
    std::string equilibrate = "no" ;                         clp.setOption("equilibrate",           &equilibrate,           "equilibrate the system (no | diag | 1-norm)");
#endif
#ifdef HAVE_MUELU_CUDA
    bool profileSetup = false;                               clp.setOption("cuda-profile-setup", "no-cuda-profile-setup", &profileSetup, "enable CUDA profiling for setup");
    bool profileSolve = false;                               clp.setOption("cuda-profile-solve", "no-cuda-profile-solve", &profileSolve, "enable CUDA profiling for solve");
#endif

    // debug options
    bool print_percept_mesh           = false;               clp.setOption("print-percept-mesh", "no-print-percept-mesh", &print_percept_mesh, "Calls perceptMesh's print_info routine");
    bool print_debug_info             = false;               clp.setOption("print-debug-info", "no-print-debug-info", &print_debug_info, "Print more debugging information");
    bool dump_element_vertices        = false;               clp.setOption("dump-element-vertices", "no-dump-element-vertices", &dump_element_vertices, "Dump the panzer_stk mesh vertices, element-by-element");

    // timer options
    bool useStackedTimer              = false;               clp.setOption("stacked-timer","no-stacked-timer", &useStackedTimer, "use stacked timer");
    bool showTimerSummary             = false;               clp.setOption("show-timer-summary", "no-show-timer-summary", &showTimerSummary, "Switch on/off the timer summary at the end of the run.");

    TEUCHOS_ASSERT(mesh_refinements >= 0); // temporarily do this instead of typing as unsigned int to get around the expected 7 arguments error for clp.setOption(...unsigned int...)

    clp.recogniseAllOptions(true);
    switch (clp.parse(argc, argv)) {
    case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:        return EXIT_SUCCESS;
    case Teuchos::CommandLineProcessor::PARSE_ERROR:
    case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION: return EXIT_FAILURE;
    case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:          break;
    }

    TEUCHOS_TEST_FOR_EXCEPTION(xmlFileName != "" && yamlFileName != "", std::runtime_error,
                               "Cannot provide both xml and yaml input files");

    // get xml file from command line if provided, otherwise use default
    std::string  xmlSolverInFileName(xmlFileName);

    // Read xml file into parameter list
    Teuchos::ParameterList inputSolverList;

    if(xmlSolverInFileName.length()) {
        out << "\nReading parameter list from the XML file \""<<xmlSolverInFileName<<"\" ...\n" << std::endl;
      Teuchos::updateParametersFromXmlFile (xmlSolverInFileName, Teuchos::ptr(&inputSolverList));
    }
    else {
      out << "Using default solver values ..." << std::endl;
    }

    /**********************************************************************************/
    /******************************* MESH AND WORKSETS ********************************/
    /**********************************************************************************/

    const int numDofsPerNode = 1;

    // TODO: due to #8475, we may need to create two meshes while we explore ways to correct it

    Teuchos::RCP<Teuchos::Time> meshTimer = Teuchos::TimeMonitor::getNewCounter("Step 1: Mesh generation");
    Teuchos::RCP<Teuchos::StackedTimer> stacked_timer;
    if(useStackedTimer)
      stacked_timer = rcp(new Teuchos::StackedTimer("MueLu_Driver"));
    Teuchos::TimeMonitor::setStackedTimer(stacked_timer);
    RCP<Teuchos::TimeMonitor> globalTimeMonitor = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Driver: S - Global Time")));
    RCP<Teuchos::TimeMonitor> tm                = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Driver: 1 - Build Mesh and Assign Physics")));

    Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory;
    Teuchos::RCP<Teuchos::ParameterList> mesh_pl = Teuchos::rcp(new Teuchos::ParameterList);


    if(exodusFileName.length())
    {
      // set the filename and type
      mesh_factory = Teuchos::rcp(new panzer_stk::STK_ExodusReaderFactory);
      mesh_pl->set("File Name",exodusFileName);
      mesh_pl->set("File Type","Exodus");

      if(mesh_refinements>0)
      {
        mesh_pl->set("Levels of Uniform Refinement",mesh_refinements); // this multiplies the number of elements by 2^(dimension*level)
        mesh_pl->set("Keep Percept Data",true); // this is necessary to gather mesh hierarchy information
        mesh_pl->set("Keep Percept Parent Elements",!delete_parent_elements); // this is necessary to gather mesh hierarchy information
      }
    }
    else if(pamgenFileName.length())
    {
      // set the filename and type
      mesh_factory = Teuchos::rcp(new panzer_stk::STK_ExodusReaderFactory);
      mesh_pl->set("File Name",pamgenFileName);
      mesh_pl->set("File Type","Pamgen");

      if(mesh_refinements>0)
      {
        mesh_pl->set("Levels of Uniform Refinement",mesh_refinements); // this multiplies the number of elements by 2^(dimension*level)
        mesh_pl->set("Keep Percept Data",true); // this is necessary to gather mesh hierarchy information
        mesh_pl->set("Keep Percept Parent Elements",!delete_parent_elements); // this is necessary to gather mesh hierarchy information
      }
    }
    else
      throw std::runtime_error("no mesh file name found!");

    // set the parameters
    mesh_factory->setParameterList(mesh_pl);

    // build the mesh
    Teuchos::RCP<panzer_stk::STK_Interface> mesh;
    mesh = mesh_factory->buildUncommitedMesh(MPI_COMM_WORLD);

    // setup the physics block
    Teuchos::RCP<Example::EquationSetFactory> eqset_factory = Teuchos::rcp(new Example::EquationSetFactory);
    Example::BCStrategyFactory bc_factory;
    const std::size_t workset_size = 100; // TODO: this may be much larger in practice. experiment with it.
    const int discretization_order = 1;

    // grab the number and names of mesh blocks
    std::vector<std::string> eBlocks;
    mesh->getElementBlockNames(eBlocks);
    for (typename std::vector<std::string>::size_type blockId = 0; blockId < eBlocks.size(); ++blockId)
      out << "eBlocks [" << blockId << "] is named: " << eBlocks[blockId] << std::endl;
    std::vector<bool> unstructured_eBlocks(eBlocks.size(), false);
    out << "After initialization, we expect 'number of element blocks' entries with 'false'." << std::endl;
    for (std::vector<bool>::size_type blockId = 0; blockId < unstructured_eBlocks.size(); ++blockId)
      out << "unstructured_eBlocks [" << blockId << "] is unstructured: " << unstructured_eBlocks[blockId] << std::endl;
    // TODO: set unstructured blocks based on some sort of input information; for example, using the Exodus ex_get_var* functions

    // grab the number and names of sidesets
    std::vector<std::string> sidesets;
    mesh->getSidesetNames(sidesets);
    for (std::vector<std::string>::size_type sidesetId = 0; sidesetId < sidesets.size(); ++sidesetId)
      out << "sidesets [" << sidesetId << "] is named: " << sidesets[sidesetId] << std::endl;

    // grab the number and names of nodesets
    std::vector<std::string> nodesets;
    mesh->getNodesetNames(nodesets);
    for (std::vector<std::string>::size_type nodesetId = 0; nodesetId < nodesets.size(); ++nodesetId)
      out << "nodesets [" << nodesetId << "] is named: " << nodesets[nodesetId] << std::endl;

    // create a physics blocks parameter list
    Teuchos::RCP<Teuchos::ParameterList> ipb = Teuchos::parameterList("Physics Blocks");
    std::vector<panzer::BC> bcs;
    std::vector<Teuchos::RCP<panzer::PhysicsBlock> > physicsBlocks;

    // set physics and boundary conditions on each block
    {
      bool build_transient_support = false;

      const int integration_order = 10;
      Teuchos::ParameterList& p = ipb->sublist("Poisson Physics");
      p.set("Type","Poisson");
      p.set("Model ID","solid");
      p.set("Basis Type","HGrad");
      p.set("Basis Order",discretization_order);
      p.set("Integration Order",integration_order);

      // TODO: double-check. this assumes we impose Dirichlet BCs on all boundaries of all physics blocks
      // It may potentially assign Dirichlet BCs to internal block boundaries, which is undesirable
      for(size_t i=0; i<eBlocks.size(); ++i)
      {
        for(size_t j=0; j<sidesets.size(); ++j)
        {
          std::size_t bc_id = j;
          panzer::BCType bctype = panzer::BCT_Dirichlet;
          std::string sideset_id = sidesets[j];
          std::string element_block_id = eBlocks[i];
          std::string dof_name = "TEMPERATURE";
          std::string strategy = "Constant";
          double value = 0.0;
	        Teuchos::ParameterList p; // this is how the official example does it, so I'll leave it alone for now
          p.set("Value",value);
          panzer::BC bc(bc_id, bctype, sideset_id, element_block_id, dof_name,
                        strategy, p);
          bcs.push_back(bc);
          std::cout<<"Dirichlet size: "<<sidesets.size()<<std::endl;
        }
        const panzer::CellData volume_cell_data(workset_size, mesh->getCellTopology(eBlocks[i]));

        // GobalData sets ostream and parameter interface to physics
        Teuchos::RCP<panzer::GlobalData> gd = panzer::createGlobalData();

        // Can be overridden by the equation set
        int default_integration_order = 1;

        // the physics block nows how to build and register evaluator with the field manager
        Teuchos::RCP<panzer::PhysicsBlock> pb
        = Teuchos::rcp(new panzer::PhysicsBlock(ipb,
                                                eBlocks[i],
                                                default_integration_order,
                                                volume_cell_data,
                                                eqset_factory,
                                                gd,
                                                build_transient_support));

        // we can have more than one physics block, one per element block
        physicsBlocks.push_back(pb);
      }
    }
    panzer::checkBCConsistency(eBlocks,sidesets,bcs);


    // finish building mesh, set required field variables and mesh bulk data
    ////////////////////////////////////////////////////////////////////////

    for(size_t i=0; i<physicsBlocks.size(); ++i)
    {
      Teuchos::RCP<panzer::PhysicsBlock> pb = physicsBlocks[i]; // we are assuming only one physics block

      const std::vector<panzer::StrPureBasisPair> & blockFields = pb->getProvidedDOFs();

      // insert all fields into a set
      std::set<panzer::StrPureBasisPair,panzer::StrPureBasisComp> fieldNames;
      fieldNames.insert(blockFields.begin(),blockFields.end());

      // add basis to DOF manager: block specific
      std::set<panzer::StrPureBasisPair,panzer::StrPureBasisComp>::const_iterator fieldItr;
      for (fieldItr=fieldNames.begin();fieldItr!=fieldNames.end();++fieldItr)
        mesh->addSolutionField(fieldItr->first,pb->elementBlockID());
    }
    mesh_factory->completeMeshConstruction(*mesh,MPI_COMM_WORLD); // this is where the mesh refinements are applied

    const unsigned int numDimensions = mesh->getDimension();
    if(print_debug_info)
      out << "Using dimension = " << numDimensions << std::endl;

    // build DOF Manager and linear object factory
    /////////////////////////////////////////////////////////////

    tm = Teuchos::null;
    tm = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Driver: 2 - Build DOF Manager and Worksets")));
    // build the connection manager
    const Teuchos::RCP<panzer::ConnManager> conn_manager = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));

    panzer::DOFManagerFactory globalIndexerFactory;
    Teuchos::RCP<panzer::GlobalIndexer> dofManager = globalIndexerFactory.buildGlobalIndexer(Teuchos::opaqueWrapper(MPI_COMM_WORLD),physicsBlocks,conn_manager);

    // construct some linear algebra object, build object to pass to evaluators
    Teuchos::RCP<panzer::LinearObjFactory<panzer::Traits> > linObjFactory = Teuchos::rcp(new panzer::TpetraLinearObjFactory<panzer::Traits,ST,LO,GO>(comm.getConst(),dofManager));

    // build worksets
    ////////////////////////////////////////////////////////

    // build STK workset factory and attach it to a workset container (uses lazy evaluation)
    Teuchos::RCP<panzer_stk::WorksetFactory> wkstFactory = Teuchos::rcp(new panzer_stk::WorksetFactory(mesh));
    Teuchos::RCP<panzer::WorksetContainer> wkstContainer = Teuchos::rcp(new panzer::WorksetContainer);
    wkstContainer->setFactory(wkstFactory);
    for(size_t i=0;i<physicsBlocks.size();i++)
      wkstContainer->setNeeds(physicsBlocks[i]->elementBlockID(),physicsBlocks[i]->getWorksetNeeds());
    wkstContainer->setWorksetSize(workset_size);
    wkstContainer->setGlobalIndexer(dofManager);


    /**********************************************************************************/
    /********************************** CONSTRUCT REGIONS *****************************/
    /**********************************************************************************/

    // The code in this section assumes that a region hierarchy can be established with Percept.
    // In the case where a region hierarchy is constructed from an exodus data input, for example,
    // this implementation will need to be updated.
    // TODO: Assign MPI rank p to region p and collect element IDs. If this region is assigned via
    // percept, reorder the element IDs lexicographically using the utility in the header.
    // Then collect mesh->identifier(node) for the nodes in lexicographic order for region p,
    // and put those in quasiRegionGIDs. Coordinates should be able to be extracted from the
    // stk::mesh::entity node as well.

    tm = Teuchos::null;
    tm = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Driver: 3 - Setup Region Information")));

    unsigned int children_per_element = 1 << (numDimensions*mesh_refinements);
    if(print_debug_info)
      out << "Number of mesh children = " << children_per_element << std::endl;

    // These will store the GIDs to be put into the region maps
    Teuchos::Array<GlobalOrdinal> quasiRegionNodeGIDs;
    Teuchos::Array<GlobalOrdinal> quasiRegionDofGIDs;
    // initialize data here that we will use for the region MG solver
    std::vector<GO> child_element_gids; // these don't always start at 0, and changes I'm making to Panzer keep changing this, so I'll store them for now
    std::vector<GO> child_element_region_gids;
    Array<GO>  sendGIDs, receiveGIDs, interfaceGIDs;
    Array<int> sendPIDs, receivePIDs, interfaceLIDsData;
    Array<LO>  sendLIDs, receiveLIDs;
    Array<GO>  quasiRegionGIDs;
    // do not run region MG if we delete parent elements or if we do not refine the mesh regularly
    if(mesh_refinements>0 && !delete_parent_elements)
    {
      // get the Percept mesh from Panzer
      Teuchos::RCP<percept::PerceptMesh> refinedMesh = mesh->getRefinedMesh();
      if(print_percept_mesh)
        refinedMesh->print_info(out,"",1,true);

      // ids are linear within stk, but we need an offset because the original mesh info comes first
      size_t node_id_start = 0;
      {
        const stk::mesh::BucketVector & local_buckets = refinedMesh->get_bulk_data()->get_buckets(stk::topology::ELEM_RANK,refinedMesh->get_fem_meta_data()->locally_owned_part());
        //const stk::mesh::BucketVector & buckets = refinedMesh->get_bulk_data()->buckets(refinedMesh->node_rank());
        stk::mesh::Bucket & bucket = **local_buckets.begin() ;
        node_id_start = refinedMesh->id(bucket[0]);
        if(print_debug_info)
          debug << "Starting node id = " << node_id_start << std::endl;
      }

      size_t elem_id_start = 0;
      {
        const stk::mesh::BucketVector & local_buckets = refinedMesh->get_bulk_data()->get_buckets(stk::topology::ELEM_RANK,refinedMesh->get_fem_meta_data()->locally_owned_part());
        //const stk::mesh::BucketVector & buckets = refinedMesh->get_bulk_data()->buckets(refinedMesh->element_rank());
        stk::mesh::Bucket & bucket = **local_buckets.begin() ;
        elem_id_start = refinedMesh->id(bucket[0]);
        if(print_debug_info)
          debug << "Starting element id = " << elem_id_start << std::endl;
      }
      //panzer_stk::workset_utils::getIdsAndVertices

      {
        const stk::mesh::BucketVector & buckets = refinedMesh->get_bulk_data()->buckets(refinedMesh->element_rank());
        // int npar=0;
        int nchild=0;
        for (stk::mesh::BucketVector::const_iterator k = buckets.begin(); k != buckets.end(); ++k)
        {
          stk::mesh::Bucket & bucket = **k ;
          if(print_debug_info)
            debug << "New bucket" << std::endl;

          const unsigned num_elements_in_bucket = bucket.size();
          for (unsigned iElement = 0; iElement < num_elements_in_bucket; iElement++)
          {
            stk::mesh::Entity element = bucket[iElement];
            if (!refinedMesh->isParentElement(element, false))
            {
              ++nchild;

              // this is the important part here. take the id of the element and the id of the element's root
              child_element_gids.push_back(refinedMesh->id(element));
              child_element_region_gids.push_back(refinedMesh->id(refinedMesh->rootOfTree(element)));

              if(print_debug_info)
                debug << "Stk Element = " << element << std::endl;

              percept::MyPairIterRelation elem_nodes ( *refinedMesh, element,  stk::topology::NODE_RANK);

              for (unsigned i_node = 0; i_node < elem_nodes.size(); i_node++)
              {
                stk::mesh::Entity node = elem_nodes[i_node].entity();
                // push_back(mesh->id(node))
                if(print_debug_info)
                  debug << "Stk Node = " << node << std::endl;



              }
            }
            else
            {
              if(print_debug_info)
                debug << "parent= " << refinedMesh->id(element) << std::endl;
            }
          }
        }
      }
    }
    else
    {
      out << "Looks like you're running from an Exodus mesh w/o Percept mesh refinement..." << std::endl;

    } // if(mesh_refinements>0 && !delete_parent_elements)

    comm->barrier();
    out << "Done working on mesh refinement and blocks detection" << std::endl;

    // Probably need to map indices of elements from the Percept indices back to the Panzer indices
    std::vector<stk::mesh::Entity> elements;
    Kokkos::DynRankView<double,PHX::Device> vertices;
    std::vector<std::size_t> localIds;
    panzer_stk::workset_utils::getIdsAndVertices(*mesh,eBlocks[myRank],localIds,vertices);

    // std::cout<<"Printing LID panzer to GID stk mapping: "<<std::endl;
    Array<LO> panzerLID2stkLID, localPanzerLID2stkLID;
    Array<GO> panzerLID2stkGID, localPanzerLID2stkGID;
    Array<GO> panzerLID2panzerGID, localPanzerLID2panzerGID;
    findPanzer2StkMapping(mesh, dofManager, vertices,
                          panzerLID2stkLID, panzerLID2stkGID, panzerLID2panzerGID);
    findPanzer2StkMappingOwned(mesh, dofManager, vertices,
                               localPanzerLID2stkLID,
                               localPanzerLID2stkGID,
                               localPanzerLID2panzerGID);
    std::unordered_map<LO,LO> localStkLID2panzerLID;
    for(typename Array<LO>::size_type idx = 0; idx < localPanzerLID2stkLID.size(); ++idx) {
      localStkLID2panzerLID[localPanzerLID2stkLID[idx]] = idx;
    }
    std::unordered_map<GO,LO> stkLID2panzerLID;
    for(typename Array<GO>::size_type idx = 0; idx < panzerLID2stkLID.size(); ++idx) {
      stkLID2panzerLID[panzerLID2stkLID[idx]] = idx;
    }
      //std::cout << "p=" << myRank << " | localPanzerLID2stkLID = " << localPanzerLID2stkLID << std::endl;


    if(dump_element_vertices)
    {
      for(unsigned int ielem=0; ielem<vertices.extent(0); ++ielem)
        for(unsigned int ivert=0; ivert<vertices.extent(1); ++ivert)
        {
          out << "element " << ielem << " vertex " << ivert << " = (" << vertices(ielem,ivert,0);
          for(unsigned int idim=1; idim<vertices.extent(2); ++idim) // fenceposting the output
            out << ", " << vertices(ielem,ivert,idim);
          out << ")" << std::endl;
        }
    }


    if(print_debug_info)
    {
      for(unsigned int i=0; i<child_element_gids.size(); ++i)
      {
        out << "child= " << child_element_gids[i] << " parent= " << child_element_region_gids[i] << std::endl;
      }
    }

    if (print_debug_info)
      printNodeCoordinates(mesh);

    if (myRank == 0 && mesh_refinements)
      perceptrenumbertest(mesh_refinements);

    // next we need to get the LIDs in order
    out << "Get Elements in order." << std::endl;
    auto dofLID = dofManager->getLIDs();
    const int numElm = dofLID.extent(0);
    Teuchos::Array<LO> elemRemap(numElm,-1);
    Teuchos::Array<LO> elemIJK(3,1);// IJK counts for elements (one less than nodes).
    Teuchos::Array<LO> regionIJK(3,1);// IJK counts for region format.

    if(myRank != 5){//TODO: update to useUnstructured
      reorderLexElem(vertices, elemRemap, elemIJK, regionIJK);
    }
    if (print_debug_info)
    {
      comm->barrier();
      std::cout << "p=" << myRank << " | elemRemap = " << elemRemap << std::endl;
    }

    std::cout<<"elemIJK: "<<elemIJK<<std::endl;

    LO numElmInRegion = (regionIJK[0])*(regionIJK[1])*(regionIJK[2]);

    if(myRank == 5) {// TODO: update to use useUnstructured (defined below line 893)
      for( int i = 0; i< numElm; i++){
        elemRemap[i] = i;
      }
      numElmInRegion = panzerLID2stkLID.size();
          //std::cout<<"asdf "<<numElmInRegion<<" local: "<<localPanzerLID2stkLID.size()<<std::endl;
      regionIJK[0] = 18;
      regionIJK[1] = 18;
      regionIJK[2] = 18;
    }
          //std::cout<<myRank<<" asdf "<<panzerLID2stkLID.size()<<" local: "<<localPanzerLID2stkLID.size()<<std::endl;


    Teuchos::Array<LO> lidRemap;
    Teuchos::Array<GO> gidRemap;
    grabLIDsGIDsLexOrder(elemIJK, elemRemap, dofLID, dofManager, numElmInRegion, lidRemap, gidRemap);
    if(myRank == 5) {// TODO: update to use useUnstructured (defined below line 893)
      lidRemap.resize(numElmInRegion, -1);
      gidRemap.resize(numElmInRegion, -1);
      for( int i = 0; i< numElmInRegion; i++){
        lidRemap[i] = i;
        gidRemap[i] = panzerLID2panzerGID[i];
      }
    }
    Teuchos::Array<GO> gidStkRemap( lidRemap.size(), -1 );
    Teuchos::Array<GO> lidStkRemap( lidRemap.size(), -1 );
    for( int i=0; i<gidStkRemap.size(); i++){
      if(lidRemap[i] < panzerLID2stkGID.size() ){
        lidStkRemap[i] = panzerLID2stkLID[ lidRemap[i] ];
        gidStkRemap[i] = panzerLID2stkGID[ lidRemap[i] ];
      }
    }

    if (print_debug_info)
    {
      comm->barrier();
      std::cout << "p=" << myRank << " | lidRemap = " << lidRemap << std::endl;
      std::cout << "p=" << myRank << " | gidRemap = " << gidRemap << std::endl;
      std::cout << "p=" << myRank << " | lidStkRemap = " << lidStkRemap << std::endl;
      std::cout << "p=" << myRank << " | gidStkRemap = " << gidStkRemap << std::endl;
      std::cout << "p=" << myRank << " | elemIJK = " << elemIJK << std::endl;
    }



    // Setup response library for checking the error in this manufactured solution
    ////////////////////////////////////////////////////////////////////////

    tm = Teuchos::null;
    tm = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Driver: 4 - Other Panzer Setup")));
    Teuchos::RCP<panzer::ResponseLibrary<panzer::Traits> > errorResponseLibrary = Teuchos::rcp(new panzer::ResponseLibrary<panzer::Traits>(wkstContainer,dofManager,linObjFactory));

    {
      const int integration_order = 10;

      panzer::FunctionalResponse_Builder<int,int> builder;
      builder.comm = MPI_COMM_WORLD;
      builder.cubatureDegree = integration_order;
      builder.requiresCellIntegral = true;
      builder.quadPointField = "TEMPERATURE_L2_ERROR";

      errorResponseLibrary->addResponse("L2 Error",eBlocks,builder);

      // TODO: uncomment the H1 errors once things look correct in the L2 norm
      /*
      builder.comm = MPI_COMM_WORLD;
      builder.cubatureDegree = integration_order;
      builder.requiresCellIntegral = true;
      builder.quadPointField = "TEMPERATURE_H1_ERROR";

      errorResponseLibrary->addResponse("H1 Error",eBlocks,builder);
       */
    }


    // setup closure model
    /////////////////////////////////////////////////////////////

    // Add in the application specific closure model factory
    panzer::ClosureModelFactory_TemplateManager<panzer::Traits> cm_factory;
    Example::ClosureModelFactory_TemplateBuilder cm_builder;
    cm_factory.buildObjects(cm_builder);

    Teuchos::ParameterList closure_models("Closure Models");
    {
      closure_models.sublist("solid").sublist("SOURCE_TEMPERATURE").set<std::string>("Type","SIMPLE SOURCE"); // a constant source
      // SOURCE_TEMPERATURE field is required by the PoissonEquationSet
      // required for error calculation
      closure_models.sublist("solid").sublist("TEMPERATURE_L2_ERROR").set<std::string>("Type","L2 ERROR_CALC");
      closure_models.sublist("solid").sublist("TEMPERATURE_L2_ERROR").set<std::string>("Field A","TEMPERATURE");
      closure_models.sublist("solid").sublist("TEMPERATURE_L2_ERROR").set<std::string>("Field B","TEMPERATURE_EXACT");

      // TODO: uncomment the H1 errors once things look correct in the L2 norm
      /*
      closure_models.sublist("solid").sublist("TEMPERATURE_H1_ERROR").set<std::string>("Type","H1 ERROR_CALC");
      closure_models.sublist("solid").sublist("TEMPERATURE_H1_ERROR").set<std::string>("Field A","TEMPERATURE");
      closure_models.sublist("solid").sublist("TEMPERATURE_H1_ERROR").set<std::string>("Field B","TEMPERATURE_EXACT");
       */
      closure_models.sublist("solid").sublist("TEMPERATURE_EXACT").set<std::string>("Type","TEMPERATURE_EXACT");
    }

    Teuchos::ParameterList user_data("User Data"); // user data can be empty here


    // setup field manager builder
    /////////////////////////////////////////////////////////////

    Teuchos::RCP<panzer::FieldManagerBuilder> fmb = Teuchos::rcp(new panzer::FieldManagerBuilder);
    fmb->setWorksetContainer(wkstContainer);
    fmb->setupVolumeFieldManagers(physicsBlocks,cm_factory,closure_models,*linObjFactory,user_data);
    fmb->setupBCFieldManagers(bcs,physicsBlocks,*eqset_factory,cm_factory,bc_factory,closure_models,
                              *linObjFactory,user_data);
    fmb->writeVolumeGraphvizDependencyFiles("Poisson", physicsBlocks);


    // setup assembly engine
    /////////////////////////////////////////////////////////////

    panzer::AssemblyEngine_TemplateManager<panzer::Traits> ae_tm;
    panzer::AssemblyEngine_TemplateBuilder builder(fmb,linObjFactory);
    ae_tm.buildObjects(builder);


    // Finalize construction of STK writer response library
    /////////////////////////////////////////////////////////////
    {
      user_data.set<int>("Workset Size",workset_size);
      errorResponseLibrary->buildResponseEvaluators(physicsBlocks,
                                                    cm_factory,
                                                    closure_models,
                                                    user_data);
    }


    // assemble linear system
    /////////////////////////////////////////////////////////////

    Teuchos::RCP<panzer::LinearObjContainer> ghostCont = linObjFactory->buildGhostedLinearObjContainer();
    Teuchos::RCP<panzer::LinearObjContainer> container = linObjFactory->buildLinearObjContainer();
    linObjFactory->initializeGhostedContainer(panzer::LinearObjContainer::X |
                                              panzer::LinearObjContainer::F |
                                              panzer::LinearObjContainer::Mat,*ghostCont);
    linObjFactory->initializeContainer(panzer::LinearObjContainer::X |
                                       panzer::LinearObjContainer::F |
                                       panzer::LinearObjContainer::Mat,*container);
    ghostCont->initialize();
    container->initialize();

    panzer::AssemblyEngineInArgs input(ghostCont,container);
    input.alpha = 0;
    input.beta = 1;

    // evaluate physics: This does both the Jacobian and residual at once
    ae_tm.getAsObject<panzer::Traits::Jacobian>()->evaluate(input);


    // /**********************************************************************************/
    // /************************************ LINEAR SOLVER *******************************/
    // /**********************************************************************************/

    // // TODO: this goes away once we finish getting the runtime errors in the region driver section sorted
    // tm = Teuchos::null;
    // tm = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Driver: 5 - Linear Solver")));

    // // convert generic linear object container to tpetra container
    // Teuchos::RCP<panzer::TpetraLinearObjContainer<ST,LO,GO> > tp_container = Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<ST,LO,GO> >(container);


    // Teuchos::RCP<MueLu::TpetraOperator<ST,LO,GO,NT> > mueLuPreconditioner;

    // if(xmlFileName.size())
    // {
    //   mueLuPreconditioner = MueLu::CreateTpetraPreconditioner(Teuchos::rcp_dynamic_cast<Tpetra::Operator<ST,LO,GO,NT> >(tp_container->get_A()), xmlFileName);
    // }
    // else
    // {
    //   Teuchos::ParameterList mueLuParamList;
    //   if(print_debug_info)
    //   {
    //     mueLuParamList.set("verbosity", "high");
    //   }
    //   else
    //   {
    //     mueLuParamList.set("verbosity", "low");
    //   }
    //   mueLuParamList.set("max levels", 3);
    //   mueLuParamList.set("coarse: max size", 10);
    //   mueLuParamList.set("multigrid algorithm", "sa");
    //   mueLuPreconditioner = MueLu::CreateTpetraPreconditioner(Teuchos::rcp_dynamic_cast<Tpetra::Operator<ST,LO,GO,NT> >(tp_container->get_A()), mueLuParamList);
    // }

    // // Setup the linear solve
    // Belos::LinearProblem<ST,MV,OP> problem(tp_container->get_A(), tp_container->get_x(), tp_container->get_f());
    // problem.setLeftPrec(mueLuPreconditioner);
    // problem.setProblem();

    // Teuchos::RCP<Teuchos::ParameterList> pl_belos = Teuchos::rcp(new Teuchos::ParameterList());
    // pl_belos->set("Maximum Iterations", 1000);
    // pl_belos->set("Convergence Tolerance", 1e-9);

    // // build the solver
    // Belos::PseudoBlockGmresSolMgr<ST,MV,OP> solver(Teuchos::rcpFromRef(problem), pl_belos);

    // // solve the linear system
    // solver.solve();

    // // scale by -1 since we solved a residual correction
    // tp_container->get_x()->scale(-1.0);
    // if(print_debug_info)
    // {
    //   debug << "Solution local length: " << tp_container->get_x()->getLocalLength() << std::endl;
    //   out << "Solution norm: " << tp_container->get_x()->norm2() << std::endl;
    // }

    /**********************************************************************************/
    /************************************ REGION DRIVER *******************************/
    /**********************************************************************************/
    std::cout<<"p = "<<myRank<<" | Begin REGION DRIVER"<<std::endl;
    {
      using Teuchos::RCP;
      using Teuchos::rcp;
      using Teuchos::ArrayRCP;
      using Teuchos::TimeMonitor;
      using Teuchos::ParameterList;

      RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();

      // =========================================================================
      // Convenient definitions
      // =========================================================================
      using STS = Teuchos::ScalarTraits<SC>;
      SC zero = STS::zero(), one = STS::one();
      // using magnitude_type = typename Teuchos::ScalarTraits<Scalar>::magnitudeType;
      using real_type = typename STS::coordinateType;
      using RealValuedMultiVector = Xpetra::MultiVector<real_type,LO,GO,NO>;

      ParameterList paramList;
      //auto inst = xpetraParameters.GetInstantiation();

      if (yamlFileName != "") {
        Teuchos::updateParametersFromYamlFileAndBroadcast(yamlFileName, Teuchos::Ptr<ParameterList>(&paramList), *comm);
      } else {
        //if (inst == Xpetra::COMPLEX_INT_INT)
        //  xmlFileName = (xmlFileName != "" ? xmlFileName : "muelu_region_poisson_input-complex.xml");
        //else
          xmlFileName = (xmlFileName != "" ? xmlFileName : "muelu_region_poisson_input.xml");
        Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName, Teuchos::Ptr<ParameterList>(&paramList), *comm);
      }

      Array<RCP<Teuchos::ParameterList> > smootherParams(1); //TODO: this is good, resized to numlevel
      smootherParams[0] = rcp(new Teuchos::ParameterList());
      smootherParams[0]->set("smoother: type",    smootherType);
      smootherParams[0]->set("smoother: sweeps",  smootherIts);
      smootherParams[0]->set("smoother: damping", smootherDamp);
      smootherParams[0]->set("smoother: Chebyshev eigRatio", smootherChebyEigRatio);
      smootherParams[0]->set("smoother: Chebyshev boost factor", smootherChebyBoostFactor);

      bool useUnstructured = false;
      Array<LO> unstructuredRanks = Teuchos::fromStringToArray<LO>(unstructured);
      for(int idx = 0; idx < unstructuredRanks.size(); ++idx) {
        if(unstructuredRanks[idx] == myRank) {useUnstructured = true;}
      } //TODO: track unstructured
      Array<LO> lNodesPerDim(3);
      for(int idx = 0; idx < 3; ++idx) {
        lNodesPerDim[idx] = regionIJK[idx];//TODO: regionIJK is region format. fix lNodesPerDim to be composite format?
      }

      // Extract matrix, vectors and other auxiliary data from Panzer
      Teuchos::RCP<panzer::TpetraLinearObjContainer<ST,LO,GO> > tp_container =
        Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<ST,LO,GO> >(container);

      ////TODO: This replaces A with a diagonal matrix.
      //tp_container->get_A()->resumeFill();
      //tp_container->get_A()->setAllToScalar(0.0);
      //tp_container->get_x()->putScalar(1.0);
      //tp_container->get_f()->putScalar(1.0);
      //for(int i=0; i<tp_container->get_x()->getLocalLength(); i++){
      //  tp_container->get_x()->replaceLocalValue(i,1.0/(i+1));
      //}
      //Tpetra::replaceDiagonalCrsMatrix(*tp_container->get_A(),*tp_container->get_x());
      //tp_container->get_A()->fillComplete();

      RCP<Matrix> A = MueLu::TpetraCrs_To_XpetraMatrix<SC,LO,GO,NO>(tp_container->get_A());
      RCP<Vector> X = Xpetra::toXpetra(tp_container->get_x());
      RCP<Vector> B = Xpetra::toXpetra(tp_container->get_f());
      //X->putScalar(1.0);
      //B->putScalar(0.0);

    std::cout<<"p = "<<myRank<<" | Matrix now Xpetra."<<std::endl;

      // The map of X, B and rowMap of A should all be the same
      // and correspond to the "dofMap" of the structured region
      // driver. The nodeMap is built assuming this map stores
      // dof associated with a single node consecutively.
      RCP<const Map> dofMap = X->getMap();

      if (print_debug_info)
      {
        comm->barrier();
        std::cout<<"dofMap: "<<std::endl;
        RCP<Teuchos::FancyOStream> my_out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
        dofMap->describe(*my_out, Teuchos::VERB_EXTREME);
      }

      Array<GO> dofGIDs = dofMap->getNodeElementList();
  Teuchos::Array<GlobalOrdinal> interfaceGIDs;

    std::cout<<"p = "<<myRank<<" | computeInterfaceNoddes."<<std::endl;

      computeInterfaceNodes(mesh, /* true */ print_debug_info, numDofsPerNode,
                            sendGIDs, sendPIDs, sendLIDs, receiveGIDs, receivePIDs, receiveLIDs,
                            quasiRegionNodeGIDs, quasiRegionDofGIDs,interfaceGIDs);
    dofMap->getComm()->barrier();
    std::cout<<"p = "<<myRank<<" | computeInterfaceNoddes Done."<<std::endl;
    dofMap->getComm()->barrier();
      for(int sendIdx = 0; sendIdx < static_cast<int>(sendGIDs.size()); ++sendIdx) {
        sendLIDs[sendIdx] = stkLID2panzerLID[sendLIDs[sendIdx]];
        sendGIDs[sendIdx] = dofGIDs[sendLIDs[sendIdx]];
      }
    //std::cout<<"p = "<<myRank<<" | receiveLIDs: "<<receiveLIDs<<std::endl;
      for(int receiveIdx = 0; receiveIdx < static_cast<int>(receiveGIDs.size()); ++receiveIdx) {
        receiveLIDs[receiveIdx] = stkLID2panzerLID[receiveLIDs[receiveIdx]];
        receiveGIDs[receiveIdx] = A->getColMap()->getGlobalElement(receiveLIDs[receiveIdx]);
      }
    //std::cout<<"p = "<<myRank<<" | receiveLIDs: "<<receiveLIDs<<std::endl;

//      Array<GO> quasiRegionDofGIDsPanzer(quasiRegionDofGIDs);
//      for(typename Array<GO>::size_type idx = 0; idx < quasiRegionDofGIDsPanzer.size(); ++idx) {
//        LO idxpanzer = -1;
//        for(typename Array<GO>::size_type i = 0; i < panzerLID2stkGID.size(); ++i) {
//          if( quasiRegionDofGIDs[idx] == panzerLID2stkGID[i] ){
//            idxpanzer = i;
//            break;
//          }
//        }
//        quasiRegionDofGIDsPanzer[idxpanzer] = panzerLID2panzerGID[idxpanzer];
//      }

      Array<GO> interfaceGIDsPanzer(interfaceGIDs.size());
      Array<LO> interfacesLIDsPanzer(interfaceGIDs.size());
      for(typename Array<GO>::size_type idx = 0; idx < interfaceGIDs.size(); ++idx) {
        LO idxpanzer = -1;
        for(typename Array<GO>::size_type i = 0; i < panzerLID2stkGID.size(); ++i) {
          if( interfaceGIDs[idx] == panzerLID2stkGID[i] ){
            idxpanzer = i;
            break;
          }
        }
        interfaceGIDsPanzer[idx] = panzerLID2panzerGID[idxpanzer];
        interfacesLIDsPanzer[idx] = idxpanzer; // NOTE: THESE VALUES NOT USED.
      }

      // RCP<Map> quasiRegionRowMap = Teuchos::null;
      // RCP<Map> regionRowMap = Teuchos::null;
      // RCP<Map> quasiRegionColMap = Teuchos::null;
      // RCP<Map> regionColMap = Teuchos::null;
      // setupRegionMaps(comm, quasiRegionDofGIDs, quasiRegionRowMap, quasiRegionColMap, regionRowMap, regionColMap);

      // if (print_debug_info)
      //   {
      //     comm->barrier();
      //     RCP<Teuchos::FancyOStream> my_out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
      //     quasiRegionRowMap->describe(*my_out, Teuchos::VERB_EXTREME);
      //     regionRowMap->describe(*my_out, Teuchos::VERB_EXTREME);
      //   }

      Array<GO> nodeGIDs(dofGIDs.size() / numDofsPerNode);
      for(typename Array<GO>::size_type nodeIdx = 0; nodeIdx < nodeGIDs.size(); ++nodeIdx) {
        nodeGIDs[nodeIdx] = dofGIDs[nodeIdx*numDofsPerNode] / numDofsPerNode;
      }
      RCP<Map> nodeMap = MapFactory::Build(dofMap->lib(),
                                           Teuchos::OrdinalTraits<Xpetra::global_size_t>::invalid(),
                                           nodeGIDs.view(0, nodeGIDs.size()),
                                           dofMap->getIndexBase(),
                                           dofMap->getComm());
      // TODO: now we need to loop over the local coordinates
      // they will then go through the composite to region
      // mechanism. Although we could probably go directly to
      // the region format?
      RCP<Xpetra::MultiVector<SC,LO,GO,NO> > nullspace =
        Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(dofMap, 1, false);
      nullspace->putScalar(one);
      RCP<Xpetra::MultiVector<double, LO, GO, NO> > coordinates =
        Xpetra::MultiVectorFactory<double, LO, GO, NO>::Build(nodeMap, numDimensions, false);
      Array<ArrayRCP<double> > coordsData(numDimensions);
      for(unsigned int dimIdx = 0; dimIdx < numDimensions; ++dimIdx) {
        coordsData[dimIdx] = coordinates->getDataNonConst(dimIdx);
      }
      stk::mesh::Part* myRegion = mesh->getElementBlockPart(eBlocks[myRank]);
      stk::mesh::EntityVector localNodes;
      stk::mesh::FieldBase *localCoordinatesField =
        mesh->getMetaData()->get_field(stk::topology::NODE_RANK, "coordinates");
      stk::mesh::get_entities(*mesh->getBulkData(), stk::topology::NODE_RANK,
                              mesh->getMetaData()->locally_owned_part(), localNodes);
      for(int nodeIdx = 0; nodeIdx < static_cast<int>(localNodes.size()); ++nodeIdx) {
        double *nodeCoord = static_cast<double *>(stk::mesh::field_data(*localCoordinatesField,
                                                                        localNodes[nodeIdx]));
        LO stkLID = getLIDfromSTKNode(mesh->getBulkData(), localNodes[nodeIdx]);
        LO panzerLID = localStkLID2panzerLID[stkLID];
        for(unsigned int dimIdx = 0; dimIdx < numDimensions; ++dimIdx) {
          coordsData[dimIdx][panzerLID] = nodeCoord[dimIdx];
        }
      }
      // std::cout << "p=" << myRank << " | lidstkremap: " << lidStkRemap() << std::endl;
      if( myRank == 5 ){
      stk::mesh::EntityVector regNodes;
      stk::mesh::get_entities(*mesh->getBulkData(), stk::topology::NODE_RANK, *myRegion, regNodes);
      int bdyCnt = 0;
      for(int nodeIdx = 0; nodeIdx < static_cast<int>(regNodes.size()); ++nodeIdx) {
        double *nodeCoord = static_cast<double *>(stk::mesh::field_data(*localCoordinatesField, regNodes[nodeIdx]));
        LO stkLID = getLIDfromSTKNode(mesh->getBulkData(), regNodes[nodeIdx]);
        LO panzerLID = stkLID2panzerLID[stkLID];
        if( nodeCoord[0] > 0.707 && nodeCoord[0] < 0.708 ){
            std::cout<<"RANK: "<<myRank<<" panzerid "<<panzerLID<<" ("<<nodeCoord[0]<<", "<<nodeCoord[1]<<", "<<nodeCoord[2]<<")"<<std::endl;
            int zijk=-1,yijk=-1;
            if( lNodesPerDim[0] == 4 ){
              if( nodeCoord[2] < -4.99)
                  zijk++;
              if( nodeCoord[2] <= 0.00001)
                  zijk++;
              if( nodeCoord[2] < -3.33)
                  zijk++;
              if( nodeCoord[2] < -1.66)
                  zijk++;

              if( nodeCoord[1] < 0.708)//
                  yijk++;
              if( nodeCoord[1] < 0.236)//
                  yijk++;
              if( nodeCoord[1] < -0.235)//
                  yijk++;
              if( nodeCoord[1] < -0.707)//
                  yijk++;
            }
            if( lNodesPerDim[0] == 10 ){
              if( nodeCoord[2] <= 0.0000001)
                zijk++;
              if( nodeCoord[2] < -0.55)
                zijk++;
              if( nodeCoord[2] < -1.11)
                zijk++;
              if( nodeCoord[2] < -1.66)
                zijk++;
              if( nodeCoord[2] < -2.22)
                zijk++;
              if( nodeCoord[2] < -2.77)
                zijk++;
              if( nodeCoord[2] < -3.33)
                zijk++;
              if( nodeCoord[2] < -3.88)
                zijk++;
              if( nodeCoord[2] < -4.44)
                zijk++;
              if( nodeCoord[2] < -4.999)
                zijk++;



              if( nodeCoord[1] < 0.7072)
                yijk++;
              if( nodeCoord[1] < 0.5500)
                yijk++;
              if( nodeCoord[1] < 0.3929)
                yijk++;
              if( nodeCoord[1] < 0.2358)
                yijk++;
              if( nodeCoord[1] < 0.0786)
                yijk++;
              if( nodeCoord[1] < -0.078)
                yijk++;
              if( nodeCoord[1] < -0.235)
                yijk++;
              if( nodeCoord[1] < -0.392)
                yijk++;
              if( nodeCoord[1] < -0.549)
                yijk++;
              if( nodeCoord[1] < -0.707)
                yijk++;
            }
            if( lNodesPerDim[0] == 16 ){
              if( nodeCoord[2] < -4.99)
                  zijk++;
              if( nodeCoord[2] <= 0.00001)
                  zijk++;
              if( nodeCoord[2] < -4.666)
                  zijk++;
              if( nodeCoord[2] < -4.333)
                  zijk++;
              if( nodeCoord[2] < -3.999)
                  zijk++;
              if( nodeCoord[2] < -3.66)
                  zijk++;
              if( nodeCoord[2] < -3.33)
                  zijk++;
              if( nodeCoord[2] < -2.999)
                  zijk++;
              if( nodeCoord[2] < -2.66)
                  zijk++;
              if( nodeCoord[2] < -2.33)
                  zijk++;
              if( nodeCoord[2] < -1.999)
                  zijk++;
              if( nodeCoord[2] < -1.66)
                  zijk++;
              if( nodeCoord[2] < -1.33)
                  zijk++;
              if( nodeCoord[2] < -0.999)
                  zijk++;
              if( nodeCoord[2] < -0.66)
                  zijk++;
              if( nodeCoord[2] < -0.33)
                  zijk++;

              if( nodeCoord[1] < 0.708)
                  yijk++;
              if( nodeCoord[1] < 0.613)
                  yijk++;
              if( nodeCoord[1] < 0.519)
                  yijk++;
              if( nodeCoord[1] < 0.425)
                  yijk++;
              if( nodeCoord[1] < 0.330)
                  yijk++;
              if( nodeCoord[1] < 0.236)
                  yijk++;
              if( nodeCoord[1] < 0.142)
                  yijk++;
              if( nodeCoord[1] < 0.0472)
                  yijk++;
              if( nodeCoord[1] < -0.047)
                  yijk++;
              if( nodeCoord[1] < -0.141)
                  yijk++;
              if( nodeCoord[1] < -0.235)
                  yijk++;
              if( nodeCoord[1] < -0.329)
                  yijk++;
              if( nodeCoord[1] < -0.424)
                  yijk++;
              if( nodeCoord[1] < -0.518)
                  yijk++;
              if( nodeCoord[1] < -0.612)
                  yijk++;
              if( nodeCoord[1] < -0.707)
                  yijk++;

            }
            if( lNodesPerDim[0] == 18 ){
              if( nodeCoord[1] < 0.624)
                  yijk++;
              if( nodeCoord[1] < 0.541)
                  yijk++;
              if( nodeCoord[1] < 0.458)
                  yijk++;
              if( nodeCoord[1] < 0.375)
                  yijk++;
              if( nodeCoord[1] < 0.292)
                  yijk++;
              if( nodeCoord[1] < 0.208)
                  yijk++;
              if( nodeCoord[1] < 0.125)
                  yijk++;
              if( nodeCoord[1] < 0.0416)
                  yijk++;
              if( nodeCoord[1] < -0.041)
                  yijk++;
              if( nodeCoord[1] < -0.124)
                  yijk++;
              if( nodeCoord[1] < -0.207)
                  yijk++;
              if( nodeCoord[1] < -0.291)
                  yijk++;
              if( nodeCoord[1] < -0.374)
                  yijk++;
              if( nodeCoord[1] < -0.457)
                  yijk++;
              if( nodeCoord[1] < -0.540)
                  yijk++;
              if( nodeCoord[1] < -0.623)
                  yijk++;
              if( nodeCoord[1] < -0.707)
                  yijk++;
              if( nodeCoord[1] < 0.708)
                  yijk++;

              if( nodeCoord[2] < 0.000001)
                  zijk++;
              if( nodeCoord[2] < -4.99)
                  zijk++;
              if( nodeCoord[2] < -4.70)
                  zijk++;
              if( nodeCoord[2] < -4.41)
                  zijk++;
              if( nodeCoord[2] < -4.11)
                  zijk++;
              if( nodeCoord[2] < -3.82)
                  zijk++;
              if( nodeCoord[2] < -3.52)
                  zijk++;
              if( nodeCoord[2] < -3.23)
                  zijk++;
              if( nodeCoord[2] < -2.94)
                  zijk++;
              if( nodeCoord[2] < -2.64)
                  zijk++;
              if( nodeCoord[2] < -2.35)
                  zijk++;
              if( nodeCoord[2] < -2.05)
                  zijk++;
              if( nodeCoord[2] < -1.76)
                  zijk++;
              if( nodeCoord[2] < -1.47)
                  zijk++;
              if( nodeCoord[2] < -1.17)
                  zijk++;
              if( nodeCoord[2] < -0.88)
                  zijk++;
              if( nodeCoord[2] < -0.58)
                  zijk++;
              if( nodeCoord[2] < -0.29)
                  zijk++;
            }
            if( lNodesPerDim[0] == 34 ){
             if( nodeCoord[1] < 0.708)
                 yijk++;
             if( nodeCoord[1] < 0.665)
                 yijk++;
             if( nodeCoord[1] < 0.622)
                 yijk++;
             if( nodeCoord[1] < 0.579)
                 yijk++;
             if( nodeCoord[1] < 0.536)
                 yijk++;
             if( nodeCoord[1] < 0.493)
                 yijk++;
             if( nodeCoord[1] < 0.450)
                 yijk++;
             if( nodeCoord[1] < 0.408)
                 yijk++;
             if( nodeCoord[1] < 0.365)
                 yijk++;
             if( nodeCoord[1] < 0.322)
                 yijk++;
             if( nodeCoord[1] < 0.279)
                 yijk++;
             if( nodeCoord[1] < 0.236)
                 yijk++;
             if( nodeCoord[1] < 0.193)
                 yijk++;
             if( nodeCoord[1] < 0.150)
                 yijk++;
             if( nodeCoord[1] < 0.108)
                 yijk++;
             if( nodeCoord[1] < 0.0643)
                 yijk++;
             if( nodeCoord[1] < 0.0215)
                 yijk++;
             if( nodeCoord[1] < -0.0214)
                 yijk++;
             if( nodeCoord[1] < -0.0642)
                 yijk++;
             if( nodeCoord[1] < -0.107)
                 yijk++;
             if( nodeCoord[1] < -0.149)
                 yijk++;
             if( nodeCoord[1] < -0.192)
                 yijk++;
             if( nodeCoord[1] < -0.235)
                 yijk++;
             if( nodeCoord[1] < -0.278)
                 yijk++;
             if( nodeCoord[1] < -0.321)
                 yijk++;
             if( nodeCoord[1] < -0.364)
                 yijk++;
             if( nodeCoord[1] < -0.407)
                 yijk++;
             if( nodeCoord[1] < -0.449)
                 yijk++;
             if( nodeCoord[1] < -0.492)
                 yijk++;
             if( nodeCoord[1] < -0.535)
                 yijk++;
             if( nodeCoord[1] < -0.578)
                 yijk++;
             if( nodeCoord[1] < -0.621)
                 yijk++;
             if( nodeCoord[1] < -0.664)
                 yijk++;
             if( nodeCoord[1] < -0.707)
                 yijk++;

             if( nodeCoord[2] < -4.99)
                 zijk++;
             if( nodeCoord[2] < -4.848)
                 zijk++;
             if( nodeCoord[2] < -4.696)
                 zijk++;
             if( nodeCoord[2] < -4.545)
                 zijk++;
             if( nodeCoord[2] < -4.393)
                 zijk++;
             if( nodeCoord[2] < -4.242)
                 zijk++;
             if( nodeCoord[2] < -4.090)
                 zijk++;
             if( nodeCoord[2] < -3.939)
                 zijk++;
             if( nodeCoord[2] < -3.787)
                 zijk++;
             if( nodeCoord[2] < -3.636)
                 zijk++;
             if( nodeCoord[2] < -3.484)
                 zijk++;
             if( nodeCoord[2] < -3.333)
                 zijk++;
             if( nodeCoord[2] < -3.181)
                 zijk++;
             if( nodeCoord[2] < -3.030)
                 zijk++;
             if( nodeCoord[2] < -2.878)
                 zijk++;
             if( nodeCoord[2] < -2.727)
                 zijk++;
             if( nodeCoord[2] < -2.575)
                 zijk++;
             if( nodeCoord[2] < -2.424)
                 zijk++;
             if( nodeCoord[2] < -2.272)
                 zijk++;
             if( nodeCoord[2] < -2.121)
                 zijk++;
             if( nodeCoord[2] < -1.969)
                 zijk++;
             if( nodeCoord[2] < -1.818)
                 zijk++;
             if( nodeCoord[2] < -1.666)
                 zijk++;
             if( nodeCoord[2] < -1.515)
                 zijk++;
             if( nodeCoord[2] < -1.363)
                 zijk++;
             if( nodeCoord[2] < -1.212)
                 zijk++;
             if( nodeCoord[2] < -1.060)
                 zijk++;
             if( nodeCoord[2] < -0.909)
                 zijk++;
             if( nodeCoord[2] < -0.757)
                 zijk++;
             if( nodeCoord[2] < -0.606)
                 zijk++;
             if( nodeCoord[2] < -0.454)
                 zijk++;
             if( nodeCoord[2] < -0.303)
                 zijk++;
             if( nodeCoord[2] < -0.151)
                 zijk++;
             if( nodeCoord[2] < 0.0001)
                 zijk++;

            }
            if( lNodesPerDim[0] == 66 ){
              if( nodeCoord[1] < 0.686)
                  yijk++;
              if( nodeCoord[1] < 0.664)
                  yijk++;
              if( nodeCoord[1] < 0.642)
                  yijk++;
              if( nodeCoord[1] < 0.621)
                  yijk++;
              if( nodeCoord[1] < 0.599)
                  yijk++;
              if( nodeCoord[1] < 0.577)
                  yijk++;
              if( nodeCoord[1] < 0.555)
                  yijk++;
              if( nodeCoord[1] < 0.534)
                  yijk++;
              if( nodeCoord[1] < 0.512)
                  yijk++;
              if( nodeCoord[1] < 0.490)
                  yijk++;
              if( nodeCoord[1] < 0.468)
                  yijk++;
              if( nodeCoord[1] < 0.447)
                  yijk++;
              if( nodeCoord[1] < 0.425)
                  yijk++;
              if( nodeCoord[1] < 0.403)
                  yijk++;
              if( nodeCoord[1] < 0.381)
                  yijk++;
              if( nodeCoord[1] < 0.359)
                  yijk++;
              if( nodeCoord[1] < 0.338)
                  yijk++;
              if( nodeCoord[1] < 0.316)
                  yijk++;
              if( nodeCoord[1] < 0.294)
                  yijk++;
              if( nodeCoord[1] < 0.272)
                  yijk++;
              if( nodeCoord[1] < 0.251)
                  yijk++;
              if( nodeCoord[1] < 0.229)
                  yijk++;
              if( nodeCoord[1] < 0.207)
                  yijk++;
              if( nodeCoord[1] < 0.185)
                  yijk++;
              if( nodeCoord[1] < 0.164)
                  yijk++;
              if( nodeCoord[1] < 0.142)
                  yijk++;
              if( nodeCoord[1] < 0.120)
                  yijk++;
              if( nodeCoord[1] < 0.098)
                  yijk++;
              if( nodeCoord[1] < 0.077)
                  yijk++;
              if( nodeCoord[1] < 0.055)
                  yijk++;
              if( nodeCoord[1] < 0.033)
                  yijk++;
              if( nodeCoord[1] < 0.011)
                  yijk++;
              if( nodeCoord[1] < -0.010)
                  yijk++;
              if( nodeCoord[1] < -0.032)
                  yijk++;
              if( nodeCoord[1] < -0.054)
                  yijk++;
              if( nodeCoord[1] < -0.076)
                  yijk++;
              if( nodeCoord[1] < -0.097)
                  yijk++;
              if( nodeCoord[1] < -0.119)
                  yijk++;
              if( nodeCoord[1] < -0.141)
                  yijk++;
              if( nodeCoord[1] < -0.163)
                  yijk++;
              if( nodeCoord[1] < -0.184)
                  yijk++;
              if( nodeCoord[1] < -0.206)
                  yijk++;
              if( nodeCoord[1] < -0.228)
                  yijk++;
              if( nodeCoord[1] < -0.250)
                  yijk++;
              if( nodeCoord[1] < -0.271)
                  yijk++;
              if( nodeCoord[1] < -0.293)
                  yijk++;
              if( nodeCoord[1] < -0.315)
                  yijk++;
              if( nodeCoord[1] < -0.337)
                  yijk++;
              if( nodeCoord[1] < -0.358)
                  yijk++;
              if( nodeCoord[1] < -0.380)
                  yijk++;
              if( nodeCoord[1] < -0.402)
                  yijk++;
              if( nodeCoord[1] < -0.424)
                  yijk++;
              if( nodeCoord[1] < -0.446)
                  yijk++;
              if( nodeCoord[1] < -0.467)
                  yijk++;
              if( nodeCoord[1] < -0.489)
                  yijk++;
              if( nodeCoord[1] < -0.511)
                  yijk++;
              if( nodeCoord[1] < -0.533)
                  yijk++;
              if( nodeCoord[1] < -0.554)
                  yijk++;
              if( nodeCoord[1] < -0.576)
                  yijk++;
              if( nodeCoord[1] < -0.598)
                  yijk++;
              if( nodeCoord[1] < -0.620)
                  yijk++;
              if( nodeCoord[1] < -0.641)
                  yijk++;
              if( nodeCoord[1] < -0.663)
                  yijk++;
              if( nodeCoord[1] < -0.685)
                  yijk++;
              if( nodeCoord[1] < -0.707)
                  yijk++;
              if( nodeCoord[1] < 0.708)
                  yijk++;



              if( nodeCoord[2] < 0.00001)
                  zijk++;
              if( nodeCoord[2] < -4.923)
                  zijk++;
              if( nodeCoord[2] < -4.846)
                  zijk++;
              if( nodeCoord[2] < -4.769)
                  zijk++;
              if( nodeCoord[2] < -4.692)
                  zijk++;
              if( nodeCoord[2] < -4.615)
                  zijk++;
              if( nodeCoord[2] < -4.538)
                  zijk++;
              if( nodeCoord[2] < -4.461)
                  zijk++;
              if( nodeCoord[2] < -4.384)
                  zijk++;
              if( nodeCoord[2] < -4.307)
                  zijk++;
              if( nodeCoord[2] < -4.230)
                  zijk++;
              if( nodeCoord[2] < -4.153)
                  zijk++;
              if( nodeCoord[2] < -4.076)
                  zijk++;
              if( nodeCoord[2] < -3.999)
                  zijk++;
              if( nodeCoord[2] < -3.923)
                  zijk++;
              if( nodeCoord[2] < -3.846)
                  zijk++;
              if( nodeCoord[2] < -3.769)
                  zijk++;
              if( nodeCoord[2] < -3.692)
                  zijk++;
              if( nodeCoord[2] < -3.615)
                  zijk++;
              if( nodeCoord[2] < -3.538)
                  zijk++;
              if( nodeCoord[2] < -3.461)
                  zijk++;
              if( nodeCoord[2] < -3.384)
                  zijk++;
              if( nodeCoord[2] < -3.307)
                  zijk++;
              if( nodeCoord[2] < -3.230)
                  zijk++;
              if( nodeCoord[2] < -3.153)
                  zijk++;
              if( nodeCoord[2] < -3.076)
                  zijk++;
              if( nodeCoord[2] < -2.999)
                  zijk++;
              if( nodeCoord[2] < -2.923)
                  zijk++;
              if( nodeCoord[2] < -2.846)
                  zijk++;
              if( nodeCoord[2] < -2.769)
                  zijk++;
              if( nodeCoord[2] < -2.692)
                  zijk++;
              if( nodeCoord[2] < -2.615)
                  zijk++;
              if( nodeCoord[2] < -2.538)
                  zijk++;
              if( nodeCoord[2] < -2.461)
                  zijk++;
              if( nodeCoord[2] < -2.384)
                  zijk++;
              if( nodeCoord[2] < -2.307)
                  zijk++;
              if( nodeCoord[2] < -2.230)
                  zijk++;
              if( nodeCoord[2] < -2.153)
                  zijk++;
              if( nodeCoord[2] < -2.076)
                  zijk++;
              if( nodeCoord[2] < -1.999)
                  zijk++;
              if( nodeCoord[2] < -1.923)
                  zijk++;
              if( nodeCoord[2] < -1.846)
                  zijk++;
              if( nodeCoord[2] < -1.769)
                  zijk++;
              if( nodeCoord[2] < -1.692)
                  zijk++;
              if( nodeCoord[2] < -1.615)
                  zijk++;
              if( nodeCoord[2] < -1.538)
                  zijk++;
              if( nodeCoord[2] < -1.461)
                  zijk++;
              if( nodeCoord[2] < -1.384)
                  zijk++;
              if( nodeCoord[2] < -1.307)
                  zijk++;
              if( nodeCoord[2] < -1.230)
                  zijk++;
              if( nodeCoord[2] < -1.153)
                  zijk++;
              if( nodeCoord[2] < -1.076)
                  zijk++;
              if( nodeCoord[2] < -0.999)
                  zijk++;
              if( nodeCoord[2] < -0.923)
                  zijk++;
              if( nodeCoord[2] < -0.846)
                  zijk++;
              if( nodeCoord[2] < -0.769)
                  zijk++;
              if( nodeCoord[2] < -0.692)
                  zijk++;
              if( nodeCoord[2] < -0.615)
                  zijk++;
              if( nodeCoord[2] < -0.538)
                  zijk++;
              if( nodeCoord[2] < -0.461)
                  zijk++;
              if( nodeCoord[2] < -0.384)
                  zijk++;
              if( nodeCoord[2] < -0.307)
                  zijk++;
              if( nodeCoord[2] < -0.230)
                  zijk++;
              if( nodeCoord[2] < -0.153)
                  zijk++;
              if( nodeCoord[2] < -0.076)
                  zijk++;
              if( nodeCoord[2] < -4.999)
                  zijk++;

            }

            std::cout<<yijk<<" "<<zijk<<std::endl;
            lidRemap[zijk + lNodesPerDim[2]*yijk] = panzerLID;
        } else {
            //std::cout<<bdyCnt<<std::endl;
            lidRemap[lNodesPerDim[1]*lNodesPerDim[2] + bdyCnt] = panzerLID;
            bdyCnt++;
        }
      }
      std::cout<<"DONE"<<std::endl;
      for( int i=0; i<gidStkRemap.size(); i++){
          lidStkRemap[i] = panzerLID2stkLID[ lidRemap[i] ];
          gidStkRemap[i] = panzerLID2stkGID[ lidRemap[i] ];
          gidRemap[i] = panzerLID2panzerGID[ lidRemap[i] ];
      }
      std::cout<<"DONE2"<<std::endl;
      for(typename Array<GO>::size_type idx = 0; idx < interfaceGIDs.size(); ++idx) {
        interfacesLIDsPanzer[idx] = idx;//lidRemap[idx];
        interfaceGIDsPanzer[idx] = gidRemap[idx];//lidRemap[idx];
      }
      //std::cout << "p=" << myRank << " | lidRemap = " << lidRemap << std::endl;
      //std::cout<<"donedone"<<std::endl;
      }// Rank5
      //std::cout << "p=" << myRank << " | interfaceGIDsPanzer: " << interfaceGIDsPanzer << std::endl;
      //std::cout << "p=" << myRank << " | interfaceLIDsPanzer: " << interfacesLIDsPanzer << std::endl;
      //

      Array<LO> localLIDstkRemap(nodeGIDs.size()), localPanzerLIDRemap(nodeGIDs.size());
      int countLocal = 0;
      for(int regionIdx = 0; regionIdx < static_cast<int>(lidStkRemap.size()); ++regionIdx) {
        bool is_local = (localStkLID2panzerLID.find(lidStkRemap[regionIdx]) == localStkLID2panzerLID.end()) ? false : true;
        if(is_local) {
          localLIDstkRemap[countLocal] = lidStkRemap[regionIdx];
          localPanzerLIDRemap[countLocal] = localStkLID2panzerLID[lidStkRemap[regionIdx]];
          ++countLocal;
        }
      }
    dofMap->getComm()->barrier();
    std::cout<<"p = "<<myRank<<" | num local region nodes "<<countLocal<<std::endl;
    dofMap->getComm()->barrier();
      // std::cout << "p=" << myRank << " | lidRemap: " << lidRemap() << std::endl;
      // std::cout << "p=" << myRank << " | gidRemap: " << gidRemap() << std::endl;
      //std::cout << "p=" << myRank << " | localPanzerLIDRemap: " << localPanzerLIDRemap() << std::endl;

      LO numLocalRegionNodes = 1;
      for(int dimIdx = 0; dimIdx < static_cast<int>(numDimensions); ++dimIdx) {
        numLocalRegionNodes = numLocalRegionNodes*regionIJK[dimIdx];
      }
    if(myRank == 5) {// TODO: update to use useUnstructured (defined below line 893)
      numLocalRegionNodes = panzerLID2stkLID.size();
    }

      // {
      //   std::ostringstream msg;
      //   msg << "p=" << myRank << " | coordinates: { ";
      //   for(int nodeIdx = 0; nodeIdx < static_cast<int>(numLocalCompositeNodes); ++nodeIdx) {
      //     msg << "(" << coordsData[0][localPanzerLIDRemap[nodeIdx]] << ", "
      //         << coordsData[1][localPanzerLIDRemap[nodeIdx]] << ", "
      //         << coordsData[2][localPanzerLIDRemap[nodeIdx]]
      //         << " -- LID: " << localPanzerLIDRemap[nodeIdx]
      //         << ", GID: " << A->getRowMap()->getGlobalElement(localPanzerLIDRemap[nodeIdx]) << ") ";
      //   }
      //   std::cout << msg.str() << "}" << std::endl;
      // }

      // std::cout << "p=" << myRank << " | panzerLID2stkGID: " << panzerLID2stkGID << std::endl;
      // std::cout << "p=" << myRank << " | localPanzerLID2stkGID: "
      //           << localPanzerLID2stkGID << std::endl;

      // {
      //   std::ostringstream msg;
      //   msg << "p=" << myRank << " | localStkLID2panzerLIDs ("
      //       << localStkLID2panzerLID.size() << "): { ";
      //   for(auto& it : localStkLID2panzerLID) {
      //     msg << "(" << it.first << ", " << it.second << ") ";
      //   }
      //   std::cout << msg.str() << "}" << std::endl;
      // }
      // stk::mesh::EntityVector nodes;
      stk::mesh::FieldBase *coordinatesField = mesh->getMetaData()->get_field(stk::topology::NODE_RANK, "coordinates");
      // stk::mesh::Part* myRegion = mesh->getElementBlockPart(eBlocks[myRank]);
      // mesh->getBulkData()->get_entities(stk::topology::NODE_RANK, *myRegion, nodes);
      RCP<Map> regionCoordMap = MapFactory::Build(dofMap->lib(),
                                                  Teuchos::OrdinalTraits<Xpetra::global_size_t>::invalid(),
                                                  gidStkRemap.size(),
                                                  dofMap->getIndexBase(),
                                                  dofMap->getComm());
      // RCP<Xpetra::MultiVector<double,LO,GO,NO> > regionCoordinates =
      //   Xpetra::MultiVectorFactory<double,LO,GO,NO>::Build(regionCoordMap, numDimensions, false);
      // Array<ArrayRCP<double> > regionCoordsData(numDimensions);
      // for(unsigned int dimIdx = 0; dimIdx < numDimensions; ++dimIdx) {
      //   regionCoordsData[dimIdx] = regionCoordinates->getDataNonConst(dimIdx);
      // }
      // stk::mesh::Entity node;
      // for(typename Array<GO>::size_type nodeIdx = 0; nodeIdx < gidStkRemap.size(); ++nodeIdx){
      //   node = mesh->getBulkData()->get_entity(stk::topology::NODE_RANK, gidStkRemap[nodeIdx] + 1);
      //   double *nodeCoord = static_cast<double *>(stk::mesh::field_data(*coordinatesField, node));
      //   for(unsigned int dimIdx = 0; dimIdx < numDimensions; ++dimIdx) {
      //     regionCoordsData[dimIdx][nodeIdx] = nodeCoord[dimIdx];
      //   }
      // }

      // {
      //   std::ostringstream msg;
      //   msg << "p=" << myRank << " | coordinates: { ";
      //   for(int nodeIdx = 0; nodeIdx < static_cast<int>(numLocalRegionNodes); ++nodeIdx) {
      //     msg << "(" << regionCoordsData[0][nodeIdx] << ", "
      //         << regionCoordsData[1][nodeIdx] << ", "
      //         << regionCoordsData[2][nodeIdx]
      //         << " -- LID: " << nodeIdx
      //         << ", GID: " << nodeIdx << ") ";
      //   }
      //   std::cout << msg.str() << "}" << std::endl;
      // }

      // regionCoordinates->describe(out, Teuchos::VERB_EXTREME);

      // X = VectorFactory::Build(dofMap);
      // X->putScalar(zero);
      // B = VectorFactory::Build(dofMap);

      Teuchos::Array<typename STS::magnitudeType> norms(1);
      B->norm2(norms);
      if( norms[0] > 0 ){
        B->scale(one/norms[0]);
      }

      comm->barrier();
      tm = Teuchos::null;

    dofMap->getComm()->barrier();
    std::cout<<"p = "<<myRank<<" | compute region data"<<std::endl;
    dofMap->getComm()->barrier();

      tm = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: 2 - Compute region data")));
      if(myRank == 1) { std::cout << "Driver: 2 - Compute region data" << std::endl;}

      // Set aggregation type for each region
      std::string aggregationRegionType;
      RCP<ParameterList> interfaceParams = rcp(new ParameterList());
      if(useUnstructured) {
        aggregationRegionType = "uncoupled";
      } else {
        aggregationRegionType = "structured";
      }

      const LO numLocalCompositeNodes = localPanzerLID2stkLID.size();
       std::cout << "p=" << myRank << " | numLocalCompositeNodes: " << numLocalCompositeNodes
                 << ", lNodesPerDim: " << lNodesPerDim << std::endl;

      // Rule for boundary duplication
      // For any two ranks that share an interface:
      // the lowest rank owns the interface and the highest rank gets extra nodes

      // 1D example of the relation between Composite, Quasi Region, and Region formats
      //
      // Composite:
      // Rank 0   Rank 1
      // [0 1 2]  [3 4]
      //
      // Quasi Region:
      // Rank 0   Rank 1
      // [0 1 2]  [2 3 4]
      //
      // Region:
      // Rank 0   Rank 1
      // [0 1 2]  [5 3 4]

      // Count the number of interface LIDs that this rank owns and sends.
      int count = 0;
      for(int i = 0; i<sendLIDs.size(); i++){
        for(int j = 0; j<i; j++){
          if(sendLIDs[i]==sendLIDs[j]){
            count++;
          }
        }
      }
      int numMyInterfaceNodes = sendLIDs.size() - count;

    dofMap->getComm()->barrier();
    std::cout<<"p = "<<myRank<<" | asdf1 myInterfaceNodes: "<<numMyInterfaceNodes<<std::endl;
    dofMap->getComm()->barrier();

      // First we count how many nodes the region needs to send and receive
      // and allocate arrays accordingly
      Array<int> boundaryConditions;
      const int maxRegPerGID = eBlocks.size();
      const int numInterfaceNodes = numLocalRegionNodes - numLocalCompositeNodes;
    dofMap->getComm()->barrier();
    std::cout<<"p = "<<myRank<<" | numInterfaceNodes: "<<numInterfaceNodes<<std::endl;
    dofMap->getComm()->barrier();
      Array<LO>  rNodesPerDim = lNodesPerDim;
      Array<LO>  compositeToRegionLIDs(numLocalCompositeNodes*numDofsPerNode);
      Array<GO>  quasiRegionGIDs(numLocalRegionNodes*numDofsPerNode);
      Array<GO>  quasiRegionCoordGIDs(numLocalRegionNodes);
      interfaceLIDsData.resize((numLocalRegionNodes - numLocalCompositeNodes + numMyInterfaceNodes)*numDofsPerNode);
      //interfaceGIDs.resize((numLocalRegionNodes - numLocalCompositeNodes + numMyInterfaceNodes)*numDofsPerNode);
    dofMap->getComm()->barrier();
    //std::cout<<"p = "<<myRank<<" | asdf2"<<std::endl;
    dofMap->getComm()->barrier();

      Array<LO>  myReceiveLIDs( numInterfaceNodes );
      int rIdx = 0;
      Array<LO>  compositeToRegionLIDsNoRemap(numLocalCompositeNodes*numDofsPerNode);

      {
        int compositeNodeIdx = 0, interfaceNodeIdx = 0;
        for(int regionIdx = 0; regionIdx < numLocalRegionNodes; ++regionIdx) {
          if((compositeNodeIdx < numLocalCompositeNodes) &&
             (lidRemap[regionIdx] == localPanzerLIDRemap[compositeNodeIdx])) {
            for(int dofIdx = 0; dofIdx < numDofsPerNode; ++dofIdx) {
              compositeToRegionLIDsNoRemap[compositeNodeIdx*numDofsPerNode + dofIdx] =
                regionIdx*numDofsPerNode + dofIdx;
            }
            for(int ii = 0; ii<sendLIDs.size(); ii++){ // Add the owned interface nodes
              if( sendLIDs[ii] == lidRemap[regionIdx]){
                for(int dofIdx = 0; dofIdx < numDofsPerNode; ++dofIdx) {
                  interfaceLIDsData[interfaceNodeIdx*numDofsPerNode + dofIdx] = regionIdx*numDofsPerNode + dofIdx;
                }
                ++interfaceNodeIdx;
                break;
              }
            }
            ++compositeNodeIdx;
          } else {
            for(int dofIdx = 0; dofIdx < numDofsPerNode; ++dofIdx) {
              interfaceLIDsData[interfaceNodeIdx*numDofsPerNode + dofIdx] = regionIdx*numDofsPerNode + dofIdx;
            }
            myReceiveLIDs[rIdx] = regionIdx;
            rIdx++;
            ++interfaceNodeIdx;
          }
        }
        std::cout << "p=" << myRank << " | numLocalCompositeNodes (" << numLocalCompositeNodes
                  << ") == compositeNodeIdx (" << compositeNodeIdx
                  << "), numInterfaceNodes (" << numInterfaceNodes
                  << ") == interfaceNodeIdx (" << interfaceNodeIdx << ")" << std::endl;
      }
    dofMap->getComm()->barrier();
    //std::cout<<"p = "<<myRank<<" | asdf3"<<std::endl;
    dofMap->getComm()->barrier();

      std::cout << "p=" << myRank << " | compositeToRegionLIDs" << std::endl;
      for(int nodeIdx = 0; nodeIdx < numLocalCompositeNodes; ++nodeIdx) {
        for(int dofIdx = 0; dofIdx < numDofsPerNode; ++dofIdx) {
          compositeToRegionLIDs[nodeIdx*numDofsPerNode + dofIdx] =
            compositeToRegionLIDsNoRemap[localPanzerLIDRemap[nodeIdx]*numDofsPerNode + dofIdx];
        }
      }
      //std::cout << "p=" << myRank << " | compositeToRegionLIDs: " << compositeToRegionLIDs << std::endl;

      std::cout << "p=" << myRank << " | quasiRegionGIDs" << std::endl;
      for(int nodeIdx = 0; nodeIdx < numLocalRegionNodes; ++nodeIdx) {
        // quasiRegionCoordGIDs[nodeIdx] = A->getColMap()->getGlobalElement(lidRemap[nodeIdx]);
        quasiRegionCoordGIDs[nodeIdx] = gidRemap[nodeIdx];
        for(int dofIdx = 0; dofIdx < numDofsPerNode; ++dofIdx) {
          quasiRegionGIDs[nodeIdx*numDofsPerNode + dofIdx] =
            quasiRegionCoordGIDs[nodeIdx]*numDofsPerNode + dofIdx;
        }
      }
      //for(int interfaceIdx = 0; interfaceIdx < (numInterfaceNodes+numMyInterfaceNodes)*numDofsPerNode; ++interfaceIdx) {
      //  interfaceGIDs[interfaceIdx] = quasiRegionGIDs[interfaceLIDsData[interfaceIdx]];
      //}

      // std::cout << "p=" << myRank << " | createInterfaceData" << std::endl;
      // createInterfaceData(static_cast<const int>(numDofsPerNode),
      //                     sendLIDs(), sendGIDs(),
      //                     receiveLIDs(), receiveGIDs(),
      //                     compositeToRegionLIDs(),
      //                     interfaceLIDsData, interfaceGIDs);
      std::cout << "p=" << myRank << " | createInterfaceData done!" << std::endl;

      const LO numSend = static_cast<LO>(sendGIDs.size());

      // std::cout << "p=" << myRank << " | numSend=" << numSend << std::endl;
      // std::cout << "p=" << myRank << " | sendGIDs: " << sendGIDs << std::endl;
      // std::cout << "p=" << myRank << " | sendPIDs: " << sendPIDs << std::endl;

      int leftBC = 0, rightBC = 1, frontBC = 1, backBC = 1, bottomBC = 1, topBC = 1;
      topBC = 1;
      bottomBC = 1;
  boundaryConditions.resize(6);
  boundaryConditions[0] = leftBC  ;
  boundaryConditions[1] = rightBC ;
  boundaryConditions[2] = frontBC ;
  boundaryConditions[3] = backBC  ;
  boundaryConditions[4] = bottomBC;
  boundaryConditions[5] = topBC   ;
      // Second we actually fill the send and receive arrays with appropriate data
      // which will allow us to compute the region and composite maps.
      // Now we can construct a list of GIDs that corresponds to rowMap
      Array<LO>  interfacesDimensions, interfacesLIDs;
      if(useUnstructured) {
        findInterface(numDimensions, rNodesPerDim, boundaryConditions,
                      interfacesDimensions, interfacesLIDs);
//interfacesLIDs = myReceiveLIDs;
interfacesLIDs = interfacesLIDsPanzer;

        // std::cout << "p=" << myRank << " | numLocalRegionNodes=" << numLocalRegionNodes
        //           << ", rNodesPerDim: " << rNodesPerDim << std::endl;
        // std::cout << "p=" << myRank << " | boundaryConditions: " << boundaryConditions << std::endl
        //           << "p=" << myRank << " | rNodesPerDim: " << rNodesPerDim << std::endl
        //           << "p=" << myRank << " | interfacesDimensions: " << interfacesDimensions << std::endl
        //           << "p=" << myRank << " | interfacesLIDs: " << interfacesLIDs << std::endl;
      }

      interfaceParams->set<Array<LO> >("interfaces: nodes per dimensions", interfacesDimensions); // nodesPerDimensions);
      interfaceParams->set<Array<LO> >("interfaces: interface nodes",      interfacesLIDs); // interfaceLIDs);

      // std::cout << "p=" << myRank << " | compositeGIDs (" << dofGIDs.size() << "): " << dofGIDs() << std::endl;
      // std::cout << "p=" << myRank << " | quasiRegionGIDs (" << quasiRegionGIDs.size() << "): " << quasiRegionGIDs << std::endl;
      // std::cout << "p=" << myRank << " | colMapGIDs: (" << A->getColMap()->getNodeElementList().size() << ")" << A->getColMap()->getNodeElementList() << std::endl;
      //std::cout << "p=" << myRank << " | interfaceGIDs: " << interfaceGIDs << std::endl;
      // std::cout << "p=" << myRank << " | interfaceLIDsData("<<interfaceLIDsData.size()<<"): " << interfaceLIDsData << std::endl;
      // std::cout << "p=" << myRank << " | interfaceLIDs: " << interfaceLIDs << std::endl;
      // std::cout << "p=" << myRank << " | quasiRegionCoordGIDs: " << quasiRegionCoordGIDs() << std::endl;

      // In our very particular case we know that a node is at most shared by 4 (8) regions in 2D (3D) problems.
      // Other geometries will certainly have different constrains and a parallel reduction using MAX
      // would be appropriate.

      comm->barrier();
      tm = Teuchos::null;

      tm = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: 3 - Build Region Matrix")));
      if(myRank == 1) { std::cout << "" << std::endl;}

      RCP<TimeMonitor> tmLocal = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: 3.1 - Build Region Maps")));
      if(myRank == 1) { std::cout << "" << std::endl;}

      Teuchos::RCP<const Xpetra::Map<LO,GO,NO> > quasiRowMap, quasiColMap;
      Teuchos::RCP<const Xpetra::Map<LO,GO,NO> > regionRowMap, regionColMap;
      quasiRowMap = MapFactory::Build(dofMap->lib(),
                                 Teuchos::OrdinalTraits<GO>::invalid(),
                                 quasiRegionGIDs(),
                                 dofMap->getIndexBase(),
                                 dofMap->getComm());
      quasiColMap = quasiRowMap;
      regionRowMap = MapFactory::Build(dofMap->lib(),
                                        Teuchos::OrdinalTraits<GO>::invalid(),
                                        numLocalRegionNodes*numDofsPerNode,
                                        dofMap->getIndexBase(),
                                        dofMap->getComm());
      regionColMap = regionRowMap;

      // Build objects needed to construct the region coordinates
      Teuchos::RCP<Xpetra::Map<LO,GO,NO> > quasiRegCoordMap = MapFactory::
          Build(nodeMap->lib(),
                Teuchos::OrdinalTraits<GO>::invalid(),
                quasiRegionCoordGIDs(),
                nodeMap->getIndexBase(),
                nodeMap->getComm());
      Teuchos::RCP<Xpetra::Map<LO,GO,NO> > regCoordMap = MapFactory::
          Build(nodeMap->lib(),
                Teuchos::OrdinalTraits<GO>::invalid(),
                numLocalRegionNodes,
                nodeMap->getIndexBase(),
                nodeMap->getComm());

      comm->barrier();
      tmLocal = Teuchos::null;
      tmLocal = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: 3.2 - Build Region Importers")));
      if(myRank == 1) { std::cout << "Driver: 3.2 - Build Region Importers" << std::endl;}

      // Setup importers
      RCP<Import> rowImport;
      RCP<Import> colImport;
      rowImport = ImportFactory::Build(dofMap, quasiRowMap);
      colImport = ImportFactory::Build(dofMap, quasiColMap);
      RCP<Import> coordImporter = ImportFactory::Build(nodeMap, quasiRegCoordMap);

      //rowImport->print(std::cout);
      //coordImporter->print(std::cout);

      comm->barrier();
      tmLocal = Teuchos::null;
      tmLocal = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: 3.3 - Import ghost GIDs")));
      if(myRank == 1) { std::cout << "Driver: 3.3 - Import ghost GIDs" << std::endl;}

      Array<GO>  interfaceCompositeGIDs, interfaceRegionGIDs;
      ExtractListOfInterfaceRegionGIDs(regionRowMap, interfaceLIDsData, interfaceRegionGIDs);
      //std::cout << "p=" << myRank << " | ExtractListOfInterfaceRegionGIDs: done " << interfaceRegionGIDs << std::endl;

      RCP<Xpetra::MultiVector<LO, LO, GO, NO> > regionsPerGIDWithGhosts;
      RCP<Xpetra::MultiVector<GO, LO, GO, NO> > interfaceGIDsMV;
      MakeRegionPerGIDWithGhosts(nodeMap, regionRowMap, rowImport,
                                 maxRegPerGID, numDofsPerNode,
                                 numLocalCompositeNodes, sendGIDs,
                                 sendPIDs, interfaceLIDsData,
                                 regionsPerGIDWithGhosts, interfaceGIDsMV);

//       {
//         comm->barrier();
//         RCP<Teuchos::FancyOStream> my_out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
//         interfaceGIDsMV->describe(*my_out, Teuchos::VERB_EXTREME);
//         regionsPerGIDWithGhosts->describe(*my_out, Teuchos::VERB_EXTREME);
//         std::cout<<"MV"<<std::endl;
//       }

      if(myRank == 1) { std::cout << "MakeRegionPerGIDWithGhosts: done" << std::endl;}

      Teuchos::ArrayRCP<LO> regionMatVecLIDs;
      RCP<Import> regionInterfaceImporter;
      SetupMatVec(interfaceGIDsMV, regionsPerGIDWithGhosts, regionRowMap, rowImport,
                  regionMatVecLIDs, regionInterfaceImporter);

      if(myRank == 1) { std::cout << "SetupMatVec: done" << std::endl;}

       //regionInterfaceImporter->print(std::cout);

      comm->barrier();
      tmLocal = Teuchos::null;
      tmLocal = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: 3.4 - Build QuasiRegion Matrix")));
      if(myRank == 1) { std::cout << "Driver: 3.4 - Build QuasiRegion Matrix" << std::endl;}

      std::cout << "p=" << myRank << " | About to create quasi region matrix" << std::endl;
      RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > quasiRegionMats;
      MakeQuasiregionMatrices(Teuchos::rcp_dynamic_cast<CrsMatrixWrap>(A),
                              regionsPerGIDWithGhosts, quasiRowMap, quasiColMap, rowImport,
                              quasiRegionMats, regionMatVecLIDs);
      std::cout << "p=" << myRank << " | Done creating quasi region matrix" << std::endl;

      comm->barrier();
      tmLocal = Teuchos::null;
      tmLocal = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: 3.5 - Build Region Matrix")));
      if(myRank == 1) { std::cout << "Driver: 3.5 - Build Region Matrix" << std::endl;}

      RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > regionMats;
      MakeRegionMatrices(Teuchos::rcp_dynamic_cast<CrsMatrixWrap>(A), A->getRowMap(), quasiRowMap,
                         regionRowMap, regionColMap,
                         rowImport, quasiRegionMats, regionMats);
      std::cout << "p=" << myRank << " | regionMats->getRowMap()->getNodeNumElements() "
                << regionMats->getRowMap()->getNodeNumElements()
                << ", rNodePerDim: " << rNodesPerDim << std::endl;

      ////////////////////////////////////////
      // {
      //   sleep(1);
      //   std::cout<<"Amat:"<<std::endl;
      //   RCP<Teuchos::FancyOStream> fancy2 = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
      //   Teuchos::FancyOStream& out2 = *fancy2;
      //   comm->barrier();
      //   A->describe(out2, Teuchos::VERB_EXTREME);
      //   comm->barrier();
      //   std::cout<<"________________________________________________"<<std::endl;
      //   comm->barrier();
      //   regionMats->describe(out2, Teuchos::VERB_EXTREME);
      //   comm->barrier();
      // }
//{
//  using TST            = Teuchos::ScalarTraits<SC>;
//  using magnitude_type = typename TST::magnitudeType;
//  using TMT            = Teuchos::ScalarTraits<magnitude_type>;
//  RCP<Vector> Xtest = VectorFactory::Build(X->getMap());
//  RCP<Vector> Btest = VectorFactory::Build(X->getMap());
//
//  // Generate a random composite X vector
//  Xtest->randomize();
//  Btest->randomize();
//
//  // Now build the region X vector
//  RCP<Vector> quasiRegX = Teuchos::null;
//  RCP<Vector> quasiRegB = Teuchos::null;
//  RCP<Vector> regX = Teuchos::null;
//  RCP<Vector> regB = Teuchos::null;
//  compositeToRegional(Xtest, quasiRegX, regX,
//                      regionMats->getMap(), rowImport);
//
//  regB = VectorFactory::Build(regionRowMap, true);
//
//  // Perform composite MatVec
//  A->apply(*Xtest, *Btest, Teuchos::NO_TRANS, TST::one(), TST::zero());
//
//  // Perform regional MatVec
//  ApplyMatVec( one, regionMats, regX, zero, regionInterfaceImporter, regionMatVecLIDs, regB, Teuchos::NO_TRANS, true);
//      RCP<Level> levelS = Hierarchy->GetLevel(0);
//      RCP<Vector> regInterfaceScalings = level->Get<RCP<Vector> >("regInterfaceScalings");
//  scaleInterfaceDOFs(regB, regInterfaceScalings, true);
//  //regionMats->apply(*regX, *regB, Teuchos::NO_TRANS, TST::one(), TST::zero(), false, regionInterfaceImporter, regionMatVecLIDs);
//
//  // Bring the result of the region MatVec
//  // to composite format so it can be compared
//  // with the original composite B vector.
//  RCP<Vector> compB = VectorFactory::Build(X->getMap());
//  regionalToComposite(regB, compB, rowImport);
//
//  // Extract the data from B and compB to compare it
//  ArrayRCP<const SC> dataB     = Btest->getData(0);
//  ArrayRCP<const SC> dataCompB = compB->getData(0);
//  for(size_t idx = 0; idx < Btest->getLocalLength(); ++idx) {
//      if( abs(TST::magnitude(dataB[idx]) - TST::magnitude(dataCompB[idx])) > 1e-6 ){
//          std::cout<<"p="<<myRank<<" | Index: "<< idx << " vals: "<< TST::magnitude(dataB[idx]) << " versus " << TST::magnitude(dataCompB[idx]) << std::endl;
//      }
//  }
//        std::cout<<"NORM        A: "<< Btest->norm1() <<std::endl;
//        std::cout<<"NORM region A: "<< regB->norm1() <<std::endl;
//        std::cout<<"NORM   comp A: "<< compB->norm1() <<std::endl;
//}


      //Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Write("fineCmpA",* A);
      //Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Write("fineRegA",* regionMats);

      // We don't need the composite operator on the fine level anymore. Free it!
      if (!A.is_null()) Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Write("A_00.m", *A);
//      A = Teuchos::null;

      comm->barrier();
      tmLocal = Teuchos::null;

      comm->barrier();
      tm = Teuchos::null;

      tm = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: 4 - Build Region Hierarchy")));
      if(myRank == 1) { std::cout << "Driver: 4 - Build Region Hierarchy" << std::endl;}

      // Setting up parameters before hierarchy construction
      // These need to stay in the driver as they would be provide by an app
      Array<int> regionNodesPerDim;
      RCP<MultiVector> regionNullspace;
      RCP<RealValuedMultiVector> regionCoordinates;

      // Set mesh structure data
      regionNodesPerDim = lNodesPerDim;

      // create nullspace vector
      regionNullspace = MultiVectorFactory::Build(quasiRowMap, nullspace->getNumVectors());
      regionNullspace->doImport(*nullspace, *rowImport, Xpetra::INSERT);
      regionNullspace->replaceMap(regionRowMap);

      // regionNullspace->describe(*fancydebug, Teuchos::VERB_EXTREME);

      // create region coordinates vector
      regionCoordinates = Xpetra::MultiVectorFactory<real_type,LO,GO,NO>::Build(quasiRegCoordMap, // TODO: this can't remain commented
                                                                                coordinates->getNumVectors());
      regionCoordinates->doImport(*coordinates, *coordImporter, Xpetra::INSERT);
      regionCoordinates->replaceMap(regCoordMap);
      // regionCoordinates->describe(*fancydebug, Teuchos::VERB_EXTREME);

      // using Tpetra_CrsMatrix = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
      // using Tpetra_MultiVector = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

      // Stuff for multi-level algorithm
      //
      // To allow for multi-level schemes with more than two levels, we need to store
      // maps, matrices, vectors, and stuff like that on each level. Since we call the
      // multi-level scheme recursively, this should be reflected in the design of
      // variables.
      //
      // We use MueLu::Hierarchy and MueLu:Level to store each quantity on each level.
      //
      RCP<ParameterList> coarseSolverData = rcp(new ParameterList());
      coarseSolverData->set<std::string>("coarse solver type", coarseSolverType);
      coarseSolverData->set<bool>("coarse solver rebalance", coarseSolverRebalance);
      coarseSolverData->set<int>("coarse rebalance num partitions", rebalanceNumPartitions);
      coarseSolverData->set<std::string>("amg xml file", coarseAmgXmlFile);
      coarseSolverData->set<std::string>("smoother xml file", coarseSmootherXMLFile);
      RCP<ParameterList> hierarchyData = rcp(new ParameterList());


      // Create MueLu Hierarchy Initially...
      // Read MueLu parameter list form xml file
      RCP<ParameterList> mueluParams = Teuchos::rcp(new ParameterList());
      Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName, mueluParams.ptr(), *dofMap->getComm());

      // Insert region-specific data into parameter list
      const std::string userName = "user data";
      Teuchos::ParameterList& userParamList = mueluParams->sublist(userName);
      userParamList.set<int>        ("int numDimensions", numDimensions);
      userParamList.set<Array<LO> > ("Array<LO> lNodesPerDim", regionNodesPerDim);
      userParamList.set<std::string>("string aggregationRegionType", aggregationRegionType);
      userParamList.set<Array<LO> > ("Array<LO> nodeOnInterface", interfaceParams->get<Array<LO> >("interfaces: interface nodes"));
      userParamList.set<Array<LO> > ("Array<LO> interfacesDimensions", interfaceParams->get<Array<LO> >("interfaces: nodes per dimensions"));
      if(Teuchos::nonnull(regionCoordinates)) {
        userParamList.set("Coordinates", regionCoordinates);
      }
      if(Teuchos::nonnull(regionNullspace)) {
        userParamList.set("Nullspace", regionNullspace);
      }

      tmLocal = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("CreateXpetraPreconditioner: Hierarchy")));
      if(myRank == 1) { std::cout << "CreateXpetraPreconditioner: Hierarchy" << std::endl;}

      // Create multigrid hierarchy part 1
      RCP<Hierarchy> regHierarchy  = MueLu::CreateXpetraPreconditioner(regionMats, *mueluParams);

      {
        RCP<MueLu::Level> level = regHierarchy->GetLevel(0);
        level->Set<RCP<Xpetra::Import<LocalOrdinal, GlobalOrdinal, Node> > >("rowImport",rowImport);
        level->Set<ArrayView<LocalOrdinal> > ("compositeToRegionLIDs", compositeToRegionLIDs() );
        level->Set<ArrayView<LocalOrdinal> > ("compositeToRegionLIDsNoRemap", compositeToRegionLIDsNoRemap() );
        level->Set<RCP<Xpetra::MultiVector<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node> > >("interfaceGIDs", interfaceGIDsMV);
        level->Set<RCP<Xpetra::MultiVector<LocalOrdinal, LocalOrdinal, GlobalOrdinal, Node> > >("regionsPerGIDWithGhosts", regionsPerGIDWithGhosts);
        level->Set<Teuchos::ArrayRCP<LocalOrdinal> >("regionMatVecLIDs", regionMatVecLIDs);
        level->Set<RCP<Xpetra::Import<LocalOrdinal, GlobalOrdinal, Node> > >("regionInterfaceImporter", regionInterfaceImporter);
        //level->print( std::cout, MueLu::Extreme );

        level->Set<Teuchos::Array<LO>>( "lidRemap",  lidRemap);
        level->Set<Teuchos::Array<LO>>( "localLIDsRemap",  localPanzerLIDRemap);
        level->Set<Teuchos::Array<GO>>( "gidRemap",  gidRemap);
      }

      tmLocal = Teuchos::null;
      if(myRank == 1) { std::cout << "Create region hierarchy" << std::endl;}


      // Create multigrid hierarchy part 2
      createRegionHierarchy(numDimensions,
                            regionNodesPerDim,
                            aggregationRegionType,
                            interfaceParams,
                            maxRegPerGID,
                            coarseSolverData,
                            smootherParams,
                            hierarchyData,
                            regHierarchy,
                            keepCoarseCoords,
                            /*debug flags, remove later! */ true);

      // hierarchyData->print();
{
  using TST            = Teuchos::ScalarTraits<SC>;
  using magnitude_type = typename TST::magnitudeType;
  using TMT            = Teuchos::ScalarTraits<magnitude_type>;
  RCP<Vector> Xtest = VectorFactory::Build(X->getMap());
  RCP<Vector> Btest = VectorFactory::Build(X->getMap());

  // Generate a random composite X vector
  Xtest->randomize();
  Btest->randomize();

  // Now build the region X vector
  RCP<Vector> quasiRegX = Teuchos::null;
  RCP<Vector> quasiRegB = Teuchos::null;
  RCP<Vector> regX = Teuchos::null;
  RCP<Vector> regB = Teuchos::null;
  compositeToRegional(Xtest, quasiRegX, regX,
                      regionMats->getMap(), rowImport);

  regB = VectorFactory::Build(regionRowMap, true);

  // Perform composite MatVec
  A->apply(*Xtest, *Btest, Teuchos::NO_TRANS, TST::one(), TST::zero());

  // Perform regional MatVec
  ApplyMatVec( one, regionMats, regX, zero, regionInterfaceImporter, regionMatVecLIDs, regB, Teuchos::NO_TRANS, true);
  RCP<MueLu::Level> levelS = regHierarchy->GetLevel(0);
  RCP<Vector> regInterfaceScalings = levelS->Get<RCP<Vector> >("regInterfaceScalings");
  scaleInterfaceDOFs(regB, regInterfaceScalings, true);
  //regionMats->apply(*regX, *regB, Teuchos::NO_TRANS, TST::one(), TST::zero(), false, regionInterfaceImporter, regionMatVecLIDs);

  // Bring the result of the region MatVec
  // to composite format so it can be compared
  // with the original composite B vector.
  RCP<Vector> compB = VectorFactory::Build(X->getMap());
  regionalToComposite(regB, compB, rowImport);

  // Extract the data from B and compB to compare it
  ArrayRCP<const SC> dataB     = Btest->getData(0);
  ArrayRCP<const SC> dataCompB = compB->getData(0);
  for(size_t idx = 0; idx < Btest->getLocalLength(); ++idx) {
      if( abs(TST::magnitude(dataB[idx]) - TST::magnitude(dataCompB[idx])) > 1e-6 ){
          std::cout<<"p="<<myRank<<" | Index: "<< idx << " vals: "<< TST::magnitude(dataB[idx]) << " versus " << TST::magnitude(dataCompB[idx]) << std::endl;
      }
  }
        std::cout<<"NORM        A: "<< Btest->norm1() <<std::endl;
        std::cout<<"NORM region A: "<< regB->norm1() <<std::endl;
        std::cout<<"NORM   comp A: "<< compB->norm1() <<std::endl;

}



      comm->barrier();
      tm = Teuchos::null;

      // Extract the number of levels from the prolongator data structure
      const int numLevels = regHierarchy->GetNumLevels();

      // Set data for fast MatVec
      for(LO levelIdx = 0; levelIdx < numLevels; ++levelIdx) {
        RCP<MueLu::Level> level = regHierarchy->GetLevel(levelIdx);
        RCP<Xpetra::Import<LO, GO, NO> > regionInterfaceImport = level->Get<RCP<Xpetra::Import<LocalOrdinal, GlobalOrdinal, Node> > >("regionInterfaceImporter");
        Teuchos::ArrayRCP<LO>            regionMatVecLIDs1     = level->Get<Teuchos::ArrayRCP<LO> >("regionMatVecLIDs");
        smootherParams[levelIdx]->set("Fast MatVec: interface LIDs",
                                      regionMatVecLIDs1);
        smootherParams[levelIdx]->set("Fast MatVec: interface importer",
                                      regionInterfaceImport);
      }

      // RCP<Teuchos::FancyOStream> fancy2 = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
      // Teuchos::FancyOStream& out2 = *fancy2;
      // for(LO levelIdx = 0; levelIdx < numLevels; ++levelIdx) {
      //   out2 << "p=" << myRank << " | regionMatVecLIDs on level " << levelIdx << std::endl;
      //   regionMatVecLIDsPerLevel[levelIdx]->describe(out2, Teuchos::VERB_EXTREME);
      // }

      tm = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: 5 - Solve with V-cycle")));
      if(myRank == 1) { std::cout << "Driver: 5 - Solve with V-cycle" << std::endl;}

      solveRegionProblem(tol, scaleResidualHist, maxIts, cycleType, convergenceLog,
                         coarseSolverData, smootherParams, hierarchyData,
                         regHierarchy, X, B);

      comm->barrier();
      tm = Teuchos::null;
      globalTimeMonitor = Teuchos::null;

      //sleep(myRank);
      //std::cout<<"p= "<<myRank<<" | X: "<<X->getDataNonConst(0)()<<std::endl;

      //sleep(myRank);
      //for(int i=0; i<X->getLocalLength(); i++){
      //  X->replaceLocalValue(i,(i+1));
      //}
      //sleep(myRank);
      //std::cout<<"p= "<<myRank<<" | X expected: "<<X->getDataNonConst(0)()<<std::endl;

      if (showTimerSummary)
      {
        RCP<ParameterList> reportParams = rcp(new ParameterList);
        const std::string filter = "";
        if (useStackedTimer) {
          Teuchos::StackedTimer::OutputOptions options;
          options.output_fraction = options.output_histogram = options.output_minmax = true;
          stacked_timer->report(out, comm, options);
        } else {
          std::ios_base::fmtflags ff(out.flags());
          TimeMonitor::report(comm.ptr(), out, filter, reportParams);
          out << std::setiosflags(ff);
        }
      }

      TimeMonitor::clearCounters();

      return EXIT_SUCCESS;
      /**/
    }

    /**********************************************************************************/
    /************************************ OUTPUT RESULTS ******************************/
    /**********************************************************************************/

    tm = Teuchos::null;
    tm = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Driver: 6 - Output Data")));

    // TODO: MueLu ordering and Panzer ordering will likely not match here... we'll need to run a conversion
    // write the solution to matrix
    {
      // redistribute solution vector to ghosted vector
      linObjFactory->globalToGhostContainer(*container,*ghostCont, panzer::TpetraLinearObjContainer<ST,LO,GO>::X
                                            | panzer::TpetraLinearObjContainer<ST,LO,GO>::DxDt);

      // get X Tpetra_Vector from ghosted container
      // TODO: there is some magic here with Tpetra objects that needs to be fixed
      //Teuchos::RCP<panzer::TpetraLinearObjContainer<ST,LO,GO> > tp_ghostCont = Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<ST,LO,GO> >(ghostCont);
      //panzer_stk::write_solution_data(*dofManager,*mesh,*tp_ghostCont->get_x());

      std::ostringstream filename;
      filename << "regionMG_output" << discretization_order << ".exo";
      mesh->writeToExodus(filename.str());
    }


    tm = Teuchos::null;
    globalTimeMonitor = Teuchos::null;

    if (showTimerSummary)
    {
      RCP<ParameterList> reportParams = rcp(new ParameterList);
      const std::string filter = "";
      if (useStackedTimer)
      {
        Teuchos::StackedTimer::OutputOptions options;
        options.output_fraction = options.output_histogram = options.output_minmax = true;
        stacked_timer->report(out, comm, options);
      }
      else
      {
        std::ios_base::fmtflags ff(out.flags());
        Teuchos::TimeMonitor::report(comm.ptr(), out, filter, reportParams);
        out << std::setiosflags(ff);
      }
    }

  } // Kokkos scope
  Kokkos::finalize();

  return 0;
} // main
