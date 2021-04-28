// Standard headers
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>

// TrilinosCouplings headers
#include "TrilinosCouplings_config.h"

// Teuchos headers
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_TimeMonitor.hpp"
// Teuchos headers (copied from MueLu side)
#include <Teuchos_YamlParameterListHelpers.hpp>
#include <Teuchos_StandardCatchMacros.hpp>

// Belos headers
#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosPseudoBlockGmresSolMgr.hpp"
#include "BelosTpetraAdapter.hpp"
// Belos headers (copied from MueLu side)
#include <BelosConfigDefs.hpp>
#include <BelosBiCGStabSolMgr.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosMueLuAdapter.hpp>      // => This header defines Belos::MueLuOp
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
// MueLu headers (copied from MueLu side)
#include <MueLu.hpp>
#include <MueLu_BaseClass.hpp>
#ifdef HAVE_MUELU_EXPLICIT_INSTANTIATION
#include <MueLu_ExplicitInstantiation.hpp>
#endif
#include <MueLu_Level.hpp>
#include <MueLu_MutuallyExclusiveTime.hpp>
#include <MueLu_ParameterListInterpreter.hpp>
#include <MueLu_Utilities.hpp>
#ifdef HAVE_MUELU_CUDA
#include "cuda_profiler_api.h"
#endif

// Region MG headers
#include "SetupRegionUtilities.hpp"
#include "SetupRegionVector_def.hpp"
#include "SetupRegionMatrix_def.hpp"
#include "SetupRegionHierarchy_def.hpp"

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

/////////////////////////////////////////////////////////////////////////
// Headers copied from MueLu driver

// This is only going to be based on the Tpetra stack...
// TODO: be careful about use of Xpetra in the MueLu side vs the use of Tpetra on this side

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

/////////////////////////////////////////////////////////////////////////

// Include factories for boundary conditions and other Panzer setup
// Most of which is taken from PoissonExample in Panzer_STK
#include "muelu_region_poisson.hpp"


int main(int argc, char *argv[]) {

  // The following typedefs are used so that the two codes will work together.
  // TODO: change everything to use the same types... i.e. check what Drekar does first before making the final decision
  // Panzer types
  using ST = double;
  using LO = panzer::LocalOrdinal;
  using GO = panzer::GlobalOrdinal;
  using NT = panzer::TpetraNodeType;
  using OP = Tpetra::Operator<ST,LO,GO,NT>;
  using MV = Tpetra::MultiVector<ST,LO,GO,NT>; 

  // MueLu types
  using Scalar = ST; 
  using LocalOrdinal = LO; 
  using GlobalOrdinal = GO;
  using Node = NT;

  Kokkos::initialize(argc,argv);
  { // Kokkos scope


    /**********************************************************************************/
    /************************************** SETUP *************************************/
    /**********************************************************************************/


    // Setup output stream, MPI, and grab info
    Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));

    // TODO: comb back through everything and make sure I'm using MPI comms properly when necessary
    Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
    Teuchos::RCP<const Teuchos::MpiComm<int> > comm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));

    const int numRanks = comm->getSize();
    const int myRank = comm->getRank();

    if (myRank == 0) {
      *out << "Running MueLu region driver on " << numRanks << " ranks... \n";
    }

    // Parse command line arguments
    Teuchos::CommandLineProcessor clp(false);
    std::string exodusFileName        = "";                  clp.setOption("exodus-mesh",           &exodusFileName,          "Exodus hex mesh filename (overrides a pamgen-mesh if both specified)");
    std::string pamgenFileName        = "cylinder.rtp";      clp.setOption("pamgen-mesh",           &pamgenFileName,          "Pamgen hex mesh filename");
    std::string xmlFileName           = "";                  clp.setOption("xml",                   &xmlFileName,             "MueLu parameters file");
    int mesh_refinements              = 1;                   clp.setOption("mesh-refinements",      &mesh_refinements,        "Uniform mesh refinements");
    bool delete_parent_elements       = false;                clp.setOption("delete-parent-elements", "keep-parent-elements", &delete_parent_elements,"Save the parent elements in the perceptMesh");

    // debug options
    bool print_percept_mesh           = false;                clp.setOption("print-percept-mesh", "no-print-percept-mesh", &print_percept_mesh, "Calls perceptMesh's print_info routine");
    bool print_debug_info             = false;                clp.setOption("print-debug-info", "no-print-debug-info", &print_debug_info, "Print more debugging information");
    bool dump_element_vertices        = false;                clp.setOption("dump-element-vertices", "no-dump-element-vertices", &dump_element_vertices, "Dump the panzer_stk mesh vertices, element-by-element");

    // timer options
    bool useStackedTimer              = false;                clp.setOption("stacked-timer","no-stacked-timer", &useStackedTimer, "use stacked timer");
    bool showTimerSummary             = false;                clp.setOption("show-timer-summary", "no-show-timer-summary", &showTimerSummary, "Switch on/off the timer summary at the end of the run.");

    clp.recogniseAllOptions(true);
    switch (clp.parse(argc, argv)) {
      case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:        return EXIT_SUCCESS;
      case Teuchos::CommandLineProcessor::PARSE_ERROR:
      case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION: return EXIT_FAILURE;
      case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:          break;
    }
    TEUCHOS_ASSERT(mesh_refinements >= 0); // get around the expected 7 arguments error for clp.setOption(...unsigned int...) for now

    // get xml file from command line if provided, otherwise use default
    std::string  xmlSolverInFileName(xmlFileName);

    // Read xml file into parameter list
    Teuchos::ParameterList inputSolverList;

    if(xmlSolverInFileName.length()) {
      if (myRank == 0)
        *out << "\nReading parameter list from the XML file \""<<xmlSolverInFileName<<"\" ...\n" << std::endl;
      Teuchos::updateParametersFromXmlFile (xmlSolverInFileName, Teuchos::ptr(&inputSolverList));
    }
    else if (myRank == 0)
      *out << "Using default solver values ..." << std::endl;


    /**********************************************************************************/
    /******************************* MESH AND WORKSETS ********************************/
    /**********************************************************************************/


    Teuchos::RCP<Teuchos::Time> meshTimer = Teuchos::TimeMonitor::getNewCounter("Step 1: Mesh generation");
    Teuchos::RCP<Teuchos::StackedTimer> stacked_timer;
    if(useStackedTimer)
      stacked_timer = rcp(new Teuchos::StackedTimer("MueLu_Driver"));
    Teuchos::TimeMonitor::setStackedTimer(stacked_timer);
    RCP<Teuchos::TimeMonitor> globalTimeMonitor = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Driver: S - Global Time")));
    RCP<Teuchos::TimeMonitor> tm                = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Driver: 1 - Build Mesh and Assign Physics")));

    Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory;
    Teuchos::RCP<Teuchos::ParameterList> mesh_pl = Teuchos::rcp(new Teuchos::ParameterList);
    std::string celltype = "Hex";

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
    const std::size_t workset_size = 10; // TODO: this may be much larger in practice. experiment with it.
    const int discretization_order = 1;

    // grab the number and names of mesh blocks
    std::vector<std::string> eBlocks;
    mesh->getElementBlockNames(eBlocks);
    std::vector<bool> unstructured_eBlocks(eBlocks.size(), false);
    // TODO: set unstructured blocks based on some sort of input information

    // grab the number and names of sidesets
    std::vector<std::string> sidesets;
    mesh->getSidesetNames(sidesets);

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
	  Teuchos::ParameterList p; // this is how the official example does it, so I'll leave it alone
          p.set("Value",value);
          panzer::BC bc(bc_id, bctype, sideset_id, element_block_id, dof_name,
                        strategy, p);
          bcs.push_back(bc);
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

    unsigned int dimension = mesh->getDimension();
    if(print_debug_info)
      std::cout << "Using dimension = " << dimension << std::endl;

    // build DOF Manager and linear object factory
    /////////////////////////////////////////////////////////////

    tm = Teuchos::null;
    tm = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Driver: 2 - Build DOF Manager and Worksets")));
    // build the connection manager
    const Teuchos::RCP<panzer::ConnManager> conn_manager
    = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));

    panzer::DOFManagerFactory globalIndexerFactory;
    Teuchos::RCP<panzer::GlobalIndexer> dofManager
    = globalIndexerFactory.buildGlobalIndexer(Teuchos::opaqueWrapper(MPI_COMM_WORLD),physicsBlocks,conn_manager);

    // construct some linear algebra object, build object to pass to evaluators
    Teuchos::RCP<panzer::LinearObjFactory<panzer::Traits> > linObjFactory
    = Teuchos::rcp(new panzer::TpetraLinearObjFactory<panzer::Traits,ST,LO,GO>(comm.getConst(),dofManager));

    // build worksets
    ////////////////////////////////////////////////////////

    Teuchos::RCP<panzer_stk::WorksetFactory> wkstFactory
    = Teuchos::rcp(new panzer_stk::WorksetFactory(mesh)); // build STK workset factory
    Teuchos::RCP<panzer::WorksetContainer> wkstContainer     // attach it to a workset container (uses lazy evaluation)
    = Teuchos::rcp(new panzer::WorksetContainer);
    wkstContainer->setFactory(wkstFactory);
    for(size_t i=0;i<physicsBlocks.size();i++)
      wkstContainer->setNeeds(physicsBlocks[i]->elementBlockID(),physicsBlocks[i]->getWorksetNeeds());
    wkstContainer->setWorksetSize(workset_size);
    wkstContainer->setGlobalIndexer(dofManager);


    /**********************************************************************************/
    /********************************** CONSTRUCT REGIONS *****************************/
    /**********************************************************************************/

    tm = Teuchos::null;
    tm = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Driver: 3 - Setup Region Information")));
    unsigned int children_per_element = 1 << (dimension*mesh_refinements);
    if(print_debug_info)
      std::cout << "Number of mesh children = " << children_per_element << std::endl;

    // initialize data here that we will use for the region MG solver
    std::vector<GO> child_element_gids; // these don't always start at 0, and changes I'm making to Panzer keep changing this, so I'll store them for now
    std::vector<GO> child_element_region_gids;

    // do not run region MG if we delete parent elements or if we do not refine the mesh regularly
    if(mesh_refinements>0 && !delete_parent_elements)
    {
      // get the Percept mesh from Panzer
      Teuchos::RCP<percept::PerceptMesh> refinedMesh = mesh->getRefinedMesh();
      if(print_percept_mesh)
        refinedMesh->print_info(*out,"",1,true);

      //    Teuchos::RCP<percept::URP_Heterogeneous_3D> breakPattern = Teuchos::rcp(new percept::URP_Heterogeneous_3D(*refinedMesh));
      //    percept::UniformRefiner breaker(*refinedMesh,*breakPattern);
      //    breaker.deleteParentElements();

      // ids are linear within stk, but we need an offset because the original mesh info comes first
      size_t node_id_start = 0;
      {
        const stk::mesh::BucketVector & buckets = refinedMesh->get_bulk_data()->buckets(refinedMesh->node_rank());
        stk::mesh::Bucket & bucket = **buckets.begin() ;
        node_id_start = refinedMesh->id(bucket[0]);
        if(print_debug_info)
          std::cout << "Starting node id = " << node_id_start << std::endl;
      }

      size_t elem_id_start = 0;
      {
        const stk::mesh::BucketVector & buckets = refinedMesh->get_bulk_data()->buckets(refinedMesh->element_rank());
        stk::mesh::Bucket & bucket = **buckets.begin() ;
        elem_id_start = refinedMesh->id(bucket[0]);
        if(print_debug_info)
          std::cout << "Starting element id = " << elem_id_start << std::endl;
      }
      //panzer_stk::workset_utils::getIdsAndVertices

      {
        const stk::mesh::BucketVector & buckets = refinedMesh->get_bulk_data()->buckets(refinedMesh->element_rank());
        int npar=0;
        int nchild=0;
        for (stk::mesh::BucketVector::const_iterator k = buckets.begin(); k != buckets.end(); ++k)
        {
          stk::mesh::Bucket & bucket = **k ;
          if(print_debug_info)
            std::cout << "New bucket" << std::endl;

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
                std::cout << "Stk Element = " << element << std::endl;

              percept::MyPairIterRelation elem_nodes ( *refinedMesh, element,  stk::topology::NODE_RANK);

              for (unsigned i_node = 0; i_node < elem_nodes.size(); i_node++)
              {
                stk::mesh::Entity node = elem_nodes[i_node].entity();
                if(print_debug_info)
                  std::cout << "Stk Node = " << node << std::endl;



              }
            }
            else
            {
              if(print_debug_info)
                std::cout << "parent= " << refinedMesh->id(element) << std::endl;
            }
          }
        }
      }
    }

    // Probably need to map indices of elements from the Percept indices back to the Panzer indices


    std::vector<stk::mesh::Entity> elements;
    Kokkos::DynRankView<double,PHX::Device> vertices;
    std::vector<std::size_t> localIds;

    panzer_stk::workset_utils::getIdsAndVertices(*mesh,eBlocks[0],localIds,vertices);
    //mesh->getElementVertices(elements,0,vertices);

    if(dump_element_vertices && myRank == 0)
    {
      for(unsigned int ielem=0; ielem<vertices.extent(0); ++ielem)
        for(unsigned int ivert=0; ivert<vertices.extent(1); ++ivert)
        {
          std::cout << "element " << ielem << " vertex " << ivert << " = (" << vertices(ielem,ivert,0);
          for(unsigned int idim=1; idim<vertices.extent(2); ++idim) // fenceposting the output
            std::cout << ", " << vertices(ielem,ivert,idim);
          std::cout << ")" << std::endl;
        }
    }



    if(print_debug_info && myRank == 0)
    {
      for(unsigned int i=0; i<child_element_gids.size(); ++i)
      {
        std::cout << "child= " << child_element_gids[i] << " parent= " << child_element_region_gids[i] << std::endl;
      }
    }

    if(myRank == 0)
      perceptrenumbertest(mesh_refinements);

    // next we need to get the LIDs

    // Setup response library for checking the error in this manufactered solution
    ////////////////////////////////////////////////////////////////////////

    tm = Teuchos::null;
    tm = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Driver: 4 - Other Panzer Setup")));
    Teuchos::RCP<panzer::ResponseLibrary<panzer::Traits> > errorResponseLibrary
    = Teuchos::rcp(new panzer::ResponseLibrary<panzer::Traits>(wkstContainer,dofManager,linObjFactory));

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

    // solve the linear system
    /////////////////////////////////////////////////////////////

    tm = Teuchos::null;
    tm = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Driver: 5 - Linear Solver")));

    // convert generic linear object container to tpetra container
    Teuchos::RCP<panzer::TpetraLinearObjContainer<ST,LO,GO> > tp_container = Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<ST,LO,GO> >(container);

    Teuchos::RCP<MueLu::TpetraOperator<ST,LO,GO,NT> > mueLuPreconditioner;

    if(xmlFileName.size())
      mueLuPreconditioner = MueLu::CreateTpetraPreconditioner(Teuchos::rcp_dynamic_cast<Tpetra::Operator<ST,LO,GO,NT> >(tp_container->get_A()), xmlFileName);
    else
    {
      Teuchos::ParameterList mueLuParamList;
      if(print_debug_info)
        mueLuParamList.set("verbosity", "high");
      else
        mueLuParamList.set("verbosity", "low");
      mueLuParamList.set("max levels", 3);
      mueLuParamList.set("coarse: max size", 10);
      mueLuParamList.set("multigrid algorithm", "sa");
      mueLuPreconditioner = MueLu::CreateTpetraPreconditioner(Teuchos::rcp_dynamic_cast<Tpetra::Operator<ST,LO,GO,NT> >(tp_container->get_A()), mueLuParamList);
    }

    // Setup the linear solve
    Belos::LinearProblem<ST,MV,OP> problem(tp_container->get_A(), tp_container->get_x(), tp_container->get_f());
    problem.setLeftPrec(mueLuPreconditioner);
    problem.setProblem();

    Teuchos::RCP<Teuchos::ParameterList> pl_belos = Teuchos::rcp(new Teuchos::ParameterList());
    pl_belos->set("Maximum Iterations", 1000);
    pl_belos->set("Convergence Tolerance", 1e-9);

    // build the solver
    Belos::PseudoBlockGmresSolMgr<ST,MV,OP> solver(Teuchos::rcpFromRef(problem), pl_belos);

    // solve the linear system
    solver.solve();

    // scale by -1 since we solved a residual correction
    tp_container->get_x()->scale(-1.0);
    std::cout << "Solution local length: " << tp_container->get_x()->getLocalLength() << std::endl;
    std::cout << "Solution norm: " << tp_container->get_x()->norm2() << std::endl;

    /**********************************************************************************/
    /************************************ REGION DRIVER *******************************/
    /**********************************************************************************/
    {

      #include <MueLu_UseShortNames.hpp>
      using Teuchos::RCP;
      using Teuchos::rcp;
      using Teuchos::ArrayRCP;
      using Teuchos::TimeMonitor;
      using Teuchos::ParameterList;

      // =========================================================================
      // MPI initialization using Teuchos
      // =========================================================================
      RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();
      // const int numRanks = comm->getSize();
      const int myRank   = comm->getRank();

      // =========================================================================
      // Convenient definitions
      // =========================================================================
      using STS = Teuchos::ScalarTraits<SC>;
      SC zero = STS::zero(), one = STS::one();
      using magnitude_type = typename Teuchos::ScalarTraits<Scalar>::magnitudeType;
      using real_type = typename STS::coordinateType;
      using RealValuedMultiVector = Xpetra::MultiVector<real_type,LO,GO,NO>;

      // =========================================================================
      // Parameters initialization
      // =========================================================================
      //GO nx = 10, ny = 10, nz = 10;
      //Galeri::Xpetra::Parameters<GO> galeriParameters(clp, nx, ny, nz, "Laplace2D"); // manage parameters of the test case
      //Xpetra::Parameters             xpetraParameters(clp);                          // manage parameters of Xpetra

      std::string xmlFileName           = "";                  clp.setOption("xml",                   &xmlFileName,           "read parameters from an xml file");
      std::string yamlFileName          = "";                  clp.setOption("yaml",                  &yamlFileName,          "read parameters from a yaml file");
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
      int         rebalanceNumPartitions = -1;                   clp.setOption("numPartitions",         &rebalanceNumPartitions, "number of partitions for rebalancing the coarse grid AMG solve");
      std::string coarseSolverType      = "direct";            clp.setOption("coarseSolverType",      &coarseSolverType,      "Type of solver for (composite) coarse level operator (smoother | direct | amg)");
      std::string unstructured          = "{}";                clp.setOption("unstructured",          &unstructured,          "List of ranks to be treated as unstructured, e.g. {0, 2, 5}");
      std::string coarseAmgXmlFile      = "";                  clp.setOption("coarseAmgXml",          &coarseAmgXmlFile,      "Read parameters for AMG as coarse level solve from this xml file.");
      std::string coarseSmootherXMLFile = "";                  clp.setOption("coarseSmootherXML",     &coarseSmootherXMLFile, "File containing the parameters to use with the coarse level smoother.");
#ifdef HAVE_MUELU_TPETRA
      std::string equilibrate = "no" ;                         clp.setOption("equilibrate",           &equilibrate,           "equilibrate the system (no | diag | 1-norm)");
#endif
#ifdef HAVE_MUELU_CUDA
      bool profileSetup = false;                               clp.setOption("cuda-profile-setup", "no-cuda-profile-setup", &profileSetup, "enable CUDA profiling for setup");
      bool profileSolve = false;                               clp.setOption("cuda-profile-solve", "no-cuda-profile-solve", &profileSolve, "enable CUDA profiling for solve");
#endif
      int  cacheSize = 0;                                      clp.setOption("cachesize",               &cacheSize,           "cache size (in KB)");
      bool useStackedTimer   = false;                          clp.setOption("stacked-timer","no-stacked-timer", &useStackedTimer, "use stacked timer");
      bool showTimerSummary = true;                            clp.setOption("show-timer-summary", "no-show-timer-summary", &showTimerSummary, "Switch on/off the timer summary at the end of the run.");
      std::string cycleType = "V";                             clp.setOption("cycleType", &cycleType, "{Multigrid cycle type. Possible values: V, W.");

      clp.recogniseAllOptions(true);
      switch (clp.parse(argc, argv)) {
      case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:        return EXIT_SUCCESS;
      case Teuchos::CommandLineProcessor::PARSE_ERROR:
      case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION: return EXIT_FAILURE;
      case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:          break;
      }

      
      TEUCHOS_TEST_FOR_EXCEPTION(xmlFileName != "" && yamlFileName != "", std::runtime_error,
                                 "Cannot provide both xml and yaml input files");
      

      // Instead of checking each time for rank, create a rank 0 stream
      RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
      Teuchos::FancyOStream& out = *fancy;
      out.setOutputToRootOnly(0);

      ParameterList paramList;
      //auto inst = xpetraParameters.GetInstantiation();

      if (yamlFileName != "") {
        Teuchos::updateParametersFromYamlFileAndBroadcast(yamlFileName, Teuchos::Ptr<ParameterList>(&paramList), *comm);
      } else {
        //if (inst == Xpetra::COMPLEX_INT_INT)
        //  xmlFileName = (xmlFileName != "" ? xmlFileName : "structured_1dof-complex.xml");
        //else
          xmlFileName = (xmlFileName != "" ? xmlFileName : "structured_1dof.xml");
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
      }
      
      // Retrieve matrix parameters (they may have been changed on the command line)
      // [for instance, if we changed matrix type from 2D to 3D we need to update nz]
      //ParameterList galeriList = galeriParameters.GetParameterList();

      // =========================================================================
      // Problem construction
      // =========================================================================
      //std::ostringstream galeriStream;
#ifdef HAVE_MUELU_OPENMP
      //std::string node_name = Node::name();
      //if(!comm->getRank() && !node_name.compare("OpenMP/Wrapper"))
      //  galeriStream<<"OpenMP Max Threads = "<<omp_get_max_threads()<<std::endl;
#endif


      comm->barrier();
      Teuchos::RCP<Teuchos::StackedTimer> stacked_timer;
      if(useStackedTimer)
        stacked_timer = rcp(new Teuchos::StackedTimer("MueLu_Driver"));
      Teuchos::TimeMonitor::setStackedTimer(stacked_timer);
      RCP<TimeMonitor> globalTimeMonitor = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: S - Global Time")));
      RCP<TimeMonitor> tm                = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: 1 - Build Composite Matrix")));


      RCP<Matrix> A;
      RCP<Map>    nodeMap, dofMap;
      RCP<Vector> X, B;
      RCP<MultiVector>           nullspace;
      RCP<RealValuedMultiVector> coordinates;

      //galeriStream << "========================================================\n" << xpetraParameters << galeriParameters;

      // Galeri will attempt to create a square-as-possible distribution of subdomains di, e.g.,
      //                                 d1  d2  d3
      //                                 d4  d5  d6
      //                                 d7  d8  d9
      //                                 d10 d11 d12
      // A perfect distribution is only possible when the #processors is a perfect square.
      // This *will* result in "strip" distribution if the #processors is a prime number or if the factors are very different in
      // size. For example, np=14 will give a 7-by-2 distribution.
      // If you don't want Galeri to do this, specify mx or my on the galeriList.
      //std::string matrixType = galeriParameters.GetMatrixType();
      int numDimensions  = 0;
      int numDofsPerNode = 0;
      Teuchos::Array<GO> procsPerDim(3);
      Teuchos::Array<GO> gNodesPerDim(3);
      Teuchos::Array<LO> lNodesPerDim(3);

      // Create map and coordinates
      // In the future, we hope to be able to first create a Galeri problem, and then request map and coordinates from it
      // At the moment, however, things are fragile as we hope that the Problem uses same map and coordinates inside
      //if (matrixType == "Laplace1D") {
      //  numDimensions = 1;
      //  nodeMap = Galeri::Xpetra::CreateMap<LO, GO, Node>(xpetraParameters.GetLib(), "Cartesian1D", comm, galeriList);
      //  coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<double,LO,GO,Map,RealValuedMultiVector>("1D", nodeMap, galeriList);

      //} else if (matrixType == "Laplace2D" || matrixType == "Star2D" ||
      //    matrixType == "BigStar2D" || matrixType == "Elasticity2D") {
      //  numDimensions = 2;
      //  nodeMap = Galeri::Xpetra::CreateMap<LO, GO, Node>(xpetraParameters.GetLib(), "Cartesian2D", comm, galeriList);
      //  coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<double,LO,GO,Map,RealValuedMultiVector>("2D", nodeMap, galeriList);

      //} else if (matrixType == "Laplace3D" || matrixType == "Brick3D" || matrixType == "Elasticity3D") {
      //  numDimensions = 3;
      //  nodeMap = Galeri::Xpetra::CreateMap<LO, GO, Node>(xpetraParameters.GetLib(), "Cartesian3D", comm, galeriList);
      //  coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<double,LO,GO,Map,RealValuedMultiVector>("3D", nodeMap, galeriList);
      // }

      // Expand map to do multiple DOF per node for block problems
      //if (matrixType == "Elasticity2D") {
      //  numDofsPerNode = 2;
      //} else if (matrixType == "Elasticity3D") {
      //  numDofsPerNode = 3;
      //} else {
      //  numDofsPerNode = 1;
      //}
      
      dofMap = Xpetra::MapFactory<LO,GO,Node>::Build(nodeMap, numDofsPerNode);
      

      //RCP<Tpetra::Problem<Map,CrsMatrixWrap,MultiVector> > Pr =
      //    Tpetra::BuildProblem<SC,LO,GO,Map,CrsMatrixWrap,MultiVector>(galeriParameters.GetMatrixType(), dofMap, galeriList);
      //A = Pr->BuildMatrix();
      
      //A = tp_container->get_A(); // TODO: convert this
      //A->SetFixedBlockSize(numDofsPerNode);

      //nullspace = Pr->BuildNullspace(); // TODO: double-check this

      X = VectorFactory::Build(dofMap);
      B = VectorFactory::Build(dofMap);

      if(serialRandom) {
        //Build the seed on rank zero and broadcast it.
        size_t localNumElements = 0;
        if(comm->getRank() == 0) {
          localNumElements = static_cast<size_t>(dofMap->getGlobalNumElements());
        }
        RCP<Map> serialMap = MapFactory::Build(dofMap->lib(),
                                               dofMap->getGlobalNumElements(),
                                               localNumElements,
                                               0,
                                               comm);
        RCP<Vector> Xserial = VectorFactory::Build(serialMap);
        Xserial->setSeed(251743369);
        Xserial->randomize();
        RCP<Import> randomnessImporter = ImportFactory::Build(serialMap, dofMap);
        X->doImport(*Xserial, *randomnessImporter, Xpetra::INSERT);
      } else {
        // we set seed for reproducibility
        Utilities::SetRandomSeed(*comm);
        X->randomize();
      }


      A->apply(*X, *B, Teuchos::NO_TRANS, one, zero);

      Teuchos::Array<typename STS::magnitudeType> norms(1);
      B->norm2(norms);
      B->scale(one/norms[0]);
      //galeriStream << "Galeri complete.\n========================================================" << std::endl;

#ifdef MATLAB_COMPARE
      Xpetra::IO<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Write("Ax.mm",*B);
      Xpetra::IO<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Write("A.mm",*A);
      B->putScalar(zero);
      Xpetra::IO<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Write("rhs.mm",*B);
      Xpetra::IO<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Write("x.mm",*X);
#endif
      //out << galeriStream.str();

      comm->barrier();
      tm = Teuchos::null;

      tm = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: 2 - Compute region data")));

      // Set aggregation type for each region
      std::string aggregationRegionType;
      RCP<ParameterList> interfaceParams = rcp(new ParameterList());
      if(useUnstructured) {
        aggregationRegionType = "uncoupled";
      } else {
        aggregationRegionType = "structured";
      }

      // Loading geometric info from galeri
//      if(numDimensions == 1) {
//        gNodesPerDim[0] = galeriList.get<GO>("nx");
//        gNodesPerDim[1] = 1;
//        gNodesPerDim[2] = 1;
//
//        lNodesPerDim[0] = galeriList.get<LO>("lnx");
//        lNodesPerDim[1] = 1;
//        lNodesPerDim[2] = 1;
//
//        procsPerDim[0] = galeriList.get<GO>("mx");
//        procsPerDim[1] = 1;
//        procsPerDim[2] = 1;
//      } else if(numDimensions == 2) {
//        gNodesPerDim[0] = galeriList.get<GO>("nx");
//        gNodesPerDim[1] = galeriList.get<GO>("ny");
//        gNodesPerDim[2] = 1;
//
//        lNodesPerDim[0] = galeriList.get<LO>("lnx");
//        lNodesPerDim[1] = galeriList.get<LO>("lny");
//        lNodesPerDim[2] = 1;
//
//        procsPerDim[0] = galeriList.get<GO>("mx");
//        procsPerDim[1] = galeriList.get<GO>("my");
//        procsPerDim[2] = 1;
//      } else if(numDimensions == 3) {
//        gNodesPerDim[0] = galeriList.get<GO>("nx");
//        gNodesPerDim[1] = galeriList.get<GO>("ny");
//        gNodesPerDim[2] = galeriList.get<GO>("nz");
//
//        lNodesPerDim[0] = galeriList.get<LO>("lnx");
//        lNodesPerDim[1] = galeriList.get<LO>("lny");
//        lNodesPerDim[2] = galeriList.get<LO>("lnz");
//
//        procsPerDim[0] = galeriList.get<GO>("mx");
//        procsPerDim[1] = galeriList.get<GO>("my");
//        procsPerDim[2] = galeriList.get<GO>("mz");
//      }

      const LO numLocalCompositeNodes = lNodesPerDim[0]*lNodesPerDim[1]*lNodesPerDim[2];

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

      // First we count how many nodes the region needs to send and receive
      // and allocate arrays accordingly
      Array<int> boundaryConditions;
      int maxRegPerGID = 0;
      int numInterfaces = 0;
      LO numLocalRegionNodes = 0;
      Array<GO>  sendGIDs;
      Array<int> sendPIDs;
      Array<LO>  rNodesPerDim(3);
      Array<LO>  compositeToRegionLIDs(nodeMap->getNodeNumElements()*numDofsPerNode);
      Array<GO>  quasiRegionGIDs;
      Array<GO>  quasiRegionCoordGIDs;
      Array<GO>  interfaceGIDs;
      Array<LO>  interfaceLIDsData;

//      createRegionData(numDimensions, useUnstructured, numDofsPerNode,
//                       gNodesPerDim(), lNodesPerDim(), procsPerDim(), nodeMap, dofMap,
//                       maxRegPerGID, numLocalRegionNodes, boundaryConditions,
//                       sendGIDs, sendPIDs, numInterfaces, rNodesPerDim,
//                       quasiRegionGIDs, quasiRegionCoordGIDs, compositeToRegionLIDs,
//                       interfaceGIDs, interfaceLIDsData);

      const LO numSend = static_cast<LO>(sendGIDs.size());

      // std::cout << "p=" << myRank << " | numSend=" << numSend << std::endl;
      // << ", numReceive=" << numReceive << std::endl;
      // std::cout << "p=" << myRank << " | receiveGIDs: " << receiveGIDs << std::endl;
      // std::cout << "p=" << myRank << " | receivePIDs: " << receivePIDs << std::endl;
      // std::cout << "p=" << myRank << " | sendGIDs: " << sendGIDs << std::endl;
      // std::cout << "p=" << myRank << " | sendPIDs: " << sendPIDs << std::endl;

      // Second we actually fill the send and receive arrays with appropriate data
      // which will allow us to compute the region and composite maps.
      // Now we can construct a list of GIDs that corresponds to rowMap
      Array<LO>  interfacesDimensions, interfacesLIDs;
      if(useUnstructured) {
        findInterface(numDimensions, rNodesPerDim, boundaryConditions,
                      interfacesDimensions, interfacesLIDs);

        // std::cout << "p=" << myRank << " | numLocalRegionNodes=" << numLocalRegionNodes
        //           << ", rNodesPerDim: " << rNodesPerDim << std::endl;
        // std::cout << "p=" << myRank << " | boundaryConditions: " << boundaryConditions << std::endl
        //           << "p=" << myRank << " | rNodesPerDim: " << rNodesPerDim << std::endl
        //           << "p=" << myRank << " | interfacesDimensions: " << interfacesDimensions << std::endl
        //           << "p=" << myRank << " | interfacesLIDs: " << interfacesLIDs << std::endl;
      }

      interfaceParams->set<Array<LO> >("interfaces: nodes per dimensions", interfacesDimensions); // nodesPerDimensions);
      interfaceParams->set<Array<LO> >("interfaces: interface nodes",      interfacesLIDs); // interfaceLIDs);

      // std::cout << "p=" << myRank << " | compositeToRegionLIDs: " << compositeToRegionLIDs << std::endl;
      // std::cout << "p=" << myRank << " | quasiRegionGIDs: " << quasiRegionGIDs << std::endl;
      // std::cout << "p=" << myRank << " | interfaceGIDs: " << interfaceGIDs << std::endl;
      // std::cout << "p=" << myRank << " | interfaceLIDsData: " << interfaceLIDsData << std::endl;
      // std::cout << "p=" << myRank << " | interfaceLIDs: " << interfaceLIDs << std::endl;
      // std::cout << "p=" << myRank << " | quasiRegionCoordGIDs: " << quasiRegionCoordGIDs() << std::endl;

      // In our very particular case we know that a node is at most shared by 4 (8) regions in 2D (3D) problems.
      // Other geometries will certainly have different constrains and a parallel reduction using MAX
      // would be appropriate.

      comm->barrier();
      tm = Teuchos::null;

      tm = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: 3 - Build Region Matrix")));

      RCP<TimeMonitor> tmLocal = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: 3.1 - Build Region Maps")));

      Teuchos::RCP<const Xpetra::Map<LO,GO,NO> > rowMap, colMap;
      Teuchos::RCP<const Xpetra::Map<LO,GO,NO> > revisedRowMap, revisedColMap;
      rowMap = Xpetra::MapFactory<LO,GO,Node>::Build(dofMap->lib(),
                                                     Teuchos::OrdinalTraits<GO>::invalid(),
                                                     quasiRegionGIDs(),
                                                     dofMap->getIndexBase(),
                                                     dofMap->getComm());
      colMap = rowMap;
      revisedRowMap = Xpetra::MapFactory<LO,GO,Node>::Build(dofMap->lib(),
                                                            Teuchos::OrdinalTraits<GO>::invalid(),
                                                            numLocalRegionNodes*numDofsPerNode,
                                                            dofMap->getIndexBase(),
                                                            dofMap->getComm());
      revisedColMap = revisedRowMap;

      // Build objects needed to construct the region coordinates
      Teuchos::RCP<Xpetra::Map<LO,GO,NO> > quasiRegCoordMap = Xpetra::MapFactory<LO,GO,Node>::
          Build(nodeMap->lib(),
                Teuchos::OrdinalTraits<GO>::invalid(),
                quasiRegionCoordGIDs(),
                nodeMap->getIndexBase(),
                nodeMap->getComm());
      Teuchos::RCP<Xpetra::Map<LO,GO,NO> > regCoordMap = Xpetra::MapFactory<LO,GO,Node>::
          Build(nodeMap->lib(),
                Teuchos::OrdinalTraits<GO>::invalid(),
                numLocalRegionNodes,
                nodeMap->getIndexBase(),
                nodeMap->getComm());

      comm->barrier();
      tmLocal = Teuchos::null;
      tmLocal = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: 3.2 - Build Region Importers")));

      // Setup importers
      RCP<Import> rowImport;
      RCP<Import> colImport;
      rowImport = ImportFactory::Build(dofMap, rowMap);
      colImport = ImportFactory::Build(dofMap, colMap);
      RCP<Import> coordImporter = ImportFactory::Build(nodeMap, quasiRegCoordMap);

      comm->barrier();
      tmLocal = Teuchos::null;
      tmLocal = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: 3.3 - Import ghost GIDs")));

      Array<GO>  interfaceCompositeGIDs, interfaceRegionGIDs;
      ExtractListOfInterfaceRegionGIDs(revisedRowMap, interfaceLIDsData, interfaceRegionGIDs);

      RCP<Xpetra::MultiVector<LO, LO, GO, NO> > regionsPerGIDWithGhosts;
      RCP<Xpetra::MultiVector<GO, LO, GO, NO> > interfaceGIDsMV;
      MakeRegionPerGIDWithGhosts(nodeMap, revisedRowMap, rowImport,
                                 maxRegPerGID, numDofsPerNode,
                                 lNodesPerDim, sendGIDs, sendPIDs, interfaceLIDsData,
                                 regionsPerGIDWithGhosts, interfaceGIDsMV);

      Teuchos::ArrayRCP<LO> regionMatVecLIDs;
      RCP<Import> regionInterfaceImporter;
      SetupMatVec(interfaceGIDsMV, regionsPerGIDWithGhosts, revisedRowMap, rowImport,
                  regionMatVecLIDs, regionInterfaceImporter);

      comm->barrier();
      tmLocal = Teuchos::null;
      tmLocal = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: 3.4 - Build QuasiRegion Matrix")));

      std::cout << "About to create quasi region matrix" << std::endl;
      RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > quasiRegionMats;
      MakeQuasiregionMatrices(Teuchos::rcp_dynamic_cast<CrsMatrixWrap>(A),
                              regionsPerGIDWithGhosts, rowMap, colMap, rowImport,
                              quasiRegionMats, regionMatVecLIDs);
      std::cout << "Done creating quasi region matrix" << std::endl;

      comm->barrier();
      tmLocal = Teuchos::null;
      tmLocal = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: 3.5 - Build Region Matrix")));

      RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > regionMats;
      MakeRegionMatrices(Teuchos::rcp_dynamic_cast<CrsMatrixWrap>(A), A->getRowMap(), rowMap,
                         revisedRowMap, revisedColMap,
                         rowImport, quasiRegionMats, regionMats);

      // We don't need the composite operator on the fine level anymore. Free it!
      A = Teuchos::null;

      comm->barrier();
      tmLocal = Teuchos::null;

      comm->barrier();
      tm = Teuchos::null;

      tm = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Driver: 4 - Build Region Hierarchy")));

      // Setting up parameters before hierarchy construction
      // These need to stay in the driver as they would be provide by an app
      Array<int> regionNodesPerDim;
      RCP<MultiVector> regionNullspace;
      RCP<RealValuedMultiVector> regionCoordinates;

      // Set mesh structure data
      regionNodesPerDim = rNodesPerDim;

      // create nullspace vector
      regionNullspace = MultiVectorFactory::Build(rowMap, nullspace->getNumVectors());
      regionNullspace->doImport(*nullspace, *rowImport, Xpetra::INSERT);
      regionNullspace->replaceMap(revisedRowMap);

      // create region coordinates vector
      regionCoordinates = Xpetra::MultiVectorFactory<real_type,LO,GO,NO>::Build(quasiRegCoordMap, // TODO: this can't remain commented
                                                                                coordinates->getNumVectors());
      regionCoordinates->doImport(*coordinates, *coordImporter, Xpetra::INSERT);
      regionCoordinates->replaceMap(regCoordMap);

      using Tpetra_CrsMatrix = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
      using Tpetra_MultiVector = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

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

      // Create multigrid hierarchy part 1
      RCP<Hierarchy> regHierarchy  = MueLu::CreateXpetraPreconditioner(regionMats, *mueluParams);

      {
        RCP<MueLu::Level> level = regHierarchy->GetLevel(0);
        level->Set<RCP<Xpetra::Import<LocalOrdinal, GlobalOrdinal, Node> > >("rowImport",rowImport);
        level->Set<ArrayView<LocalOrdinal> > ("compositeToRegionLIDs", compositeToRegionLIDs() );
        level->Set<RCP<Xpetra::MultiVector<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node> > >("interfaceGIDs", interfaceGIDsMV);
        level->Set<RCP<Xpetra::MultiVector<LocalOrdinal, LocalOrdinal, GlobalOrdinal, Node> > >("regionsPerGIDWithGhosts", regionsPerGIDWithGhosts);
        level->Set<Teuchos::ArrayRCP<LocalOrdinal> >("regionMatVecLIDs", regionMatVecLIDs);
        level->Set<RCP<Xpetra::Import<LocalOrdinal, GlobalOrdinal, Node> > >("regionInterfaceImporter", regionInterfaceImporter);
        level->print( std::cout, MueLu::Extreme );
      }

      tmLocal = Teuchos::null;


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
                            keepCoarseCoords);

      hierarchyData->print();



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

      {
        //    std::cout << myRank << " | Running V-cycle ..." << std::endl;

        TEUCHOS_TEST_FOR_EXCEPT_MSG(!(numLevels>0), "We require numLevel > 0. Probably, numLevel has not been set, yet.");

        // We first use the non-level container variables to setup the fine grid problem.
        // This is ok since the initial setup just mimics the application and the outer
        // Krylov method.
        //
        // We switch to using the level container variables as soon as we enter the
        // recursive part of the algorithm.
        //

        // Composite residual vector
        RCP<Vector> compRes = VectorFactory::Build(dofMap, true);

        // transform composite vectors to regional layout
        Teuchos::RCP<Vector> quasiRegX;
        Teuchos::RCP<Vector> regX;
        compositeToRegional(X, quasiRegX, regX,
                            revisedRowMap, rowImport);

        RCP<Vector> quasiRegB;
        RCP<Vector> regB;
        compositeToRegional(B, quasiRegB, regB,
                            revisedRowMap, rowImport);
#ifdef DUMP_LOCALX_AND_A
        FILE *fp;
        char str[80];
        sprintf(str,"theMatrix.%d",myRank);
        fp = fopen(str,"w");
        fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
        LO numNzs = 0;
        for (size_t kkk = 0; kkk < regionMats->getNodeNumRows(); kkk++) {
          ArrayView<const LO> AAcols;
          ArrayView<const SC> AAvals;
          regionMats->getLocalRowView(kkk, AAcols, AAvals);
          const int *Acols    = AAcols.getRawPtr();
          const SC  *Avals = AAvals.getRawPtr();
          numNzs += AAvals.size();
        }
        fprintf(fp, "%d %d %d\n",regionMats->getNodeNumRows(),regionMats->getNodeNumRows(),numNzs);

        for (size_t kkk = 0; kkk < regionMats->getNodeNumRows(); kkk++) {
          ArrayView<const LO> AAcols;
          ArrayView<const SC> AAvals;
          regionMats->getLocalRowView(kkk, AAcols, AAvals);
          const int *Acols    = AAcols.getRawPtr();
          const SC  *Avals = AAvals.getRawPtr();
          LO RowLeng = AAvals.size();
          for (LO kk = 0; kk < RowLeng; kk++) {
            fprintf(fp, "%d %d %22.16e\n",kkk+1,Acols[kk]+1,Avals[kk]);
          }
        }
        fclose(fp);
        sprintf(str,"theX.%d",myRank);
        fp = fopen(str,"w");
        ArrayRCP<SC> lX= regX->getDataNonConst(0);
        for (size_t kkk = 0; kkk < regionMats->getNodeNumRows(); kkk++) fprintf(fp, "%22.16e\n",lX[kkk]);
        fclose(fp);
#endif

        RCP<Vector> regRes;
        regRes = VectorFactory::Build(revisedRowMap, true);

        /////////////////////////////////////////////////////////////////////////
        // SWITCH TO RECURSIVE STYLE --> USE LEVEL CONTAINER VARIABLES
        /////////////////////////////////////////////////////////////////////////

        // Prepare output of residual norm to file
        RCP<std::ofstream> log;
        if (myRank == 0)
        {
          log = rcp(new std::ofstream(convergenceLog.c_str()));
          (*log) << "# num procs = " << dofMap->getComm()->getSize() << "\n"
              << "# iteration | res-norm (scaled=" << scaleResidualHist << ")\n"
              << "#\n";
          *log << std::setprecision(16) << std::scientific;
        }

        // Print type of residual norm to the screen
        if (scaleResidualHist)
          out << "Using scaled residual norm." << std::endl;
        else
          out << "Using unscaled residual norm." << std::endl;


        // Richardson iterations
        magnitude_type normResIni = Teuchos::ScalarTraits<magnitude_type>::zero();
        const int old_precision = std::cout.precision();
        std::cout << std::setprecision(8) << std::scientific;
        int cycle = 0;

        Teuchos::RCP<Vector> regCorrect;
        regCorrect = VectorFactory::Build(revisedRowMap, true);
        for (cycle = 0; cycle < maxIts; ++cycle)
        {
          const Scalar SC_ZERO = Teuchos::ScalarTraits<SC>::zero();
          regCorrect->putScalar(SC_ZERO);
          // Get Stuff out of Hierarchy
          RCP<MueLu::Level> level = regHierarchy->GetLevel(0);
          RCP<Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal> > regInterfaceScalings = level->Get<RCP<Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal> > >("regInterfaceScalings");
          // check for convergence
          {
            ////////////////////////////////////////////////////////////////////////
            // SWITCH BACK TO NON-LEVEL VARIABLES
            ////////////////////////////////////////////////////////////////////////
            computeResidual(regRes, regX, regB, regionMats, *smootherParams[0]);
            scaleInterfaceDOFs(regRes, regInterfaceScalings, true);

            compRes = VectorFactory::Build(dofMap, true);
            regionalToComposite(regRes, compRes, rowImport);

            typename Teuchos::ScalarTraits<Scalar>::magnitudeType normRes = compRes->norm2();
            if(cycle == 0) { normResIni = normRes; }

            if (scaleResidualHist)
              normRes /= normResIni;

            // Output current residual norm to screen (on proc 0 only)
            out << cycle << "\t" << normRes << std::endl;
            if (myRank == 0)
              (*log) << cycle << "\t" << normRes << "\n";

            if (normRes < tol)
              break;
          }

          /////////////////////////////////////////////////////////////////////////
          // SWITCH TO RECURSIVE STYLE --> USE LEVEL CONTAINER VARIABLES
          /////////////////////////////////////////////////////////////////////////

          bool zeroInitGuess = true;
          scaleInterfaceDOFs(regRes, regInterfaceScalings, false);
          vCycle(0, numLevels, cycleType, regHierarchy,
                 regCorrect, regRes,
                 smootherParams, zeroInitGuess, coarseSolverData, hierarchyData);

          regX->update(one, *regCorrect, one);
        }
        out << "Number of iterations performed for this solve: " << cycle << std::endl;

        std::cout << std::setprecision(old_precision);
        std::cout.unsetf(std::ios::fixed | std::ios::scientific);
      }

      comm->barrier();
      tm = Teuchos::null;
      globalTimeMonitor = Teuchos::null;

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
    /*********************************** END REGION DRIVER*****************************/
    /**********************************************************************************/

    // output data (optional)
    /////////////////////////////////////////////////////////////

    tm = Teuchos::null;
    tm = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Driver: 6 - Output Data")));

    // write the solution to matrix
    {
      // redistribute solution vector to ghosted vector
      linObjFactory->globalToGhostContainer(*container,*ghostCont, panzer::TpetraLinearObjContainer<ST,LO,GO>::X
                                            | panzer::TpetraLinearObjContainer<ST,LO,GO>::DxDt);

      // get X Tpetra_Vector from ghosted container
      //Teuchos::RCP<panzer::TpetraLinearObjContainer<ST,LO,GO> > tp_ghostCont = Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<ST,LO,GO> >(ghostCont);
      //panzer_stk::write_solution_data(*dofManager,*mesh,*tp_ghostCont->get_x());

      // Due to multiple instances of this test being run at the same
      // time (one for each celltype and each order), we need to
      // differentiate output to prevent race conditions on output
      // file. Multiple runs for different mesh refinement levels for
      // the same celltype/order are ok as they are staged one after
      // another in the ADD_ADVANCED_TEST cmake macro.
      std::ostringstream filename;
      filename << "output_" << celltype << "_p" << discretization_order << ".exo";
      mesh->writeToExodus(filename.str());
    }

    // compute the error of the finite element solution
    /////////////////////////////////////////////////////////////

    if(true)
    {
      panzer::AssemblyEngineInArgs respInput(ghostCont,container);
      respInput.alpha = 0;
      respInput.beta = 1;

      Teuchos::RCP<panzer::ResponseBase> l2_resp = errorResponseLibrary->getResponse<panzer::Traits::Residual>("L2 Error");
      Teuchos::RCP<panzer::Response_Functional<panzer::Traits::Residual> > l2_resp_func
      = Teuchos::rcp_dynamic_cast<panzer::Response_Functional<panzer::Traits::Residual> >(l2_resp);
      Teuchos::RCP<Thyra::VectorBase<double> > l2_respVec = Thyra::createMember(l2_resp_func->getVectorSpace());
      l2_resp_func->setVector(l2_respVec);

      /*
      Teuchos::RCP<panzer::ResponseBase> h1_resp = errorResponseLibrary->getResponse<panzer::Traits::Residual>("H1 Error");
      Teuchos::RCP<panzer::Response_Functional<panzer::Traits::Residual> > h1_resp_func
        = Teuchos::rcp_dynamic_cast<panzer::Response_Functional<panzer::Traits::Residual> >(h1_resp);
      Teuchos::RCP<Thyra::VectorBase<double> > h1_respVec = Thyra::createMember(h1_resp_func->getVectorSpace());
      h1_resp_func->setVector(h1_respVec);
       */

      errorResponseLibrary->addResponsesToInArgs<panzer::Traits::Residual>(respInput);
      errorResponseLibrary->evaluate<panzer::Traits::Residual>(respInput);

      if(myRank == 0)
      {
        *out << "This is the Basis Order" << std::endl;
        *out << "Basis Order = " << discretization_order << std::endl;
        *out << "This is the L2 Error" << std::endl;
        *out << "L2 Error = " << sqrt(l2_resp_func->value) << std::endl;
        //*out << "This is the H1 Error" << std::endl;
        //*out << "H1 Error = " << sqrt(h1_resp_func->value) << std::endl;
      }
    }

    tm = Teuchos::null;
    globalTimeMonitor = Teuchos::null;

    if (showTimerSummary)
    {
      RCP<ParameterList> reportParams = rcp(new ParameterList);
      const std::string filter = "";
      if (useStackedTimer) {
        Teuchos::StackedTimer::OutputOptions options;
        options.output_fraction = options.output_histogram = options.output_minmax = true;
        stacked_timer->report(*out, comm, options);
      } else {
        std::ios_base::fmtflags ff(out->flags());
        Teuchos::TimeMonitor::report(comm.ptr(), *out, filter, reportParams);
        *out << std::setiosflags(ff);
      }
    }

  } // Kokkos scope
  Kokkos::finalize();

  return 0;
} // main
