#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_TimeMonitor.hpp>

using Teuchos::RCP;
using Teuchos::rcp;

#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Panzer_STK_Version.hpp"
#include "Panzer_STK_config.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "Panzer_STK_WorksetFactory.hpp"
#include "Panzer_Workset_Builder.hpp"
#include "Panzer_FieldManagerBuilder.hpp"
#include "Panzer_STKConnManager.hpp"
#include "Panzer_DOFManagerFactory.hpp"
#include "Panzer_EpetraLinearObjFactory.hpp"
#include "Panzer_ParameterList_ObjectBuilders.hpp"
#include "Panzer_GlobalData.hpp"
#include "user_app_EquationSetFactory.hpp"
#include "user_app_ClosureModel_Factory_TemplateBuilder.hpp"
#include "user_app_BCStrategy_Factory.hpp"

#include "Panzer_ResponseContainer.hpp"
#include "Panzer_ResponseLibrary.hpp"
#include "Panzer_WorksetContainer.hpp"

#include "Panzer_ResponseAggregator_IPCoordinates.hpp"

#include "TestEvaluators.hpp"

#include "Epetra_MpiComm.h"

#include <vector>
#include <map>
#include <string>

namespace panzer {

  void testInitialzation(panzer::InputPhysicsBlock& ipb,
			 std::vector<panzer::BC>& bcs);

  TEUCHOS_UNIT_TEST(response_library_stk, test)
  {
    typedef Traits::Residual EvalT;

    using Teuchos::RCP;

  #ifdef HAVE_MPI
     Teuchos::RCP<Teuchos::Comm<int> > tcomm = Teuchos::rcp(new Teuchos::MpiComm<int>(Teuchos::opaqueWrapper(MPI_COMM_WORLD)));
  #else
     Teuchos::RCP<Teuchos::Comm<int> > tcomm = Teuchos::rcp(new Teuchos::SerialComm<int>);
  #endif

    panzer_stk::SquareQuadMeshFactory mesh_factory;
    user_app::MyFactory eqset_factory;
    user_app::BCFactory bc_factory;
    const std::size_t workset_size = 20;

    panzer::FieldManagerBuilder<int,int> fmb;

    // setup mesh
    /////////////////////////////////////////////
    RCP<panzer_stk::STK_Interface> mesh;
    {
       RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
       pl->set("X Blocks",2);
       pl->set("Y Blocks",1);
       pl->set("X Elements",2);
       pl->set("Y Elements",2);
       pl->set("X0",0.0);
       pl->set("Y0",0.0);
       pl->set("Xf",8.0);
       pl->set("Yf",4.0);
       mesh_factory.setParameterList(pl);
       mesh = mesh_factory.buildMesh(MPI_COMM_WORLD);
    }

    // setup physic blocks
    /////////////////////////////////////////////
    panzer::InputPhysicsBlock ipb;
    std::vector<panzer::BC> bcs;
    std::vector<Teuchos::RCP<panzer::PhysicsBlock> > physics_blocks;
    {
       std::map<std::string,panzer::InputPhysicsBlock> 
             physics_id_to_input_physics_blocks;

       testInitialzation(ipb, bcs);

       std::map<std::string,std::string> block_ids_to_physics_ids;
       block_ids_to_physics_ids["eblock-0_0"] = "test physics";
       block_ids_to_physics_ids["eblock-1_0"] = "test physics";

       std::map<std::string,Teuchos::RCP<const shards::CellTopology> > block_ids_to_cell_topo;
       block_ids_to_cell_topo["eblock-0_0"] = mesh->getCellTopology("eblock-0_0");
       block_ids_to_cell_topo["eblock-1_0"] = mesh->getCellTopology("eblock-1_0");
    
       physics_id_to_input_physics_blocks["test physics"] = ipb;

       Teuchos::RCP<panzer::GlobalData> gd = panzer::createGlobalData();

       panzer::buildPhysicsBlocks(block_ids_to_physics_ids,
                                  block_ids_to_cell_topo,
                                  physics_id_to_input_physics_blocks,
                                  2,workset_size,
                                  eqset_factory,
				  gd,
		    	          false,
                                  physics_blocks);
    }

    // setup worksets
    /////////////////////////////////////////////
 
     std::vector<std::string> validEBlocks;
     mesh->getElementBlockNames(validEBlocks);

    // build WorksetContainer
    Teuchos::RCP<panzer_stk::WorksetFactory> wkstFactory 
       = Teuchos::rcp(new panzer_stk::WorksetFactory(mesh)); // build STK workset factory
    Teuchos::RCP<panzer::WorksetContainer> wkstContainer     // attach it to a workset container (uses lazy evaluation)
       = Teuchos::rcp(new panzer::WorksetContainer(wkstFactory,physics_blocks,workset_size));
 
    // setup DOF manager
    /////////////////////////////////////////////
    const Teuchos::RCP<panzer::ConnManager<int,int> > conn_manager 
           = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));

    Teuchos::RCP<const panzer::UniqueGlobalIndexerFactory<int,int,int,int> > indexerFactory
          = Teuchos::rcp(new panzer::DOFManagerFactory<int,int>);
    const Teuchos::RCP<panzer::UniqueGlobalIndexer<int,int> > dofManager 
          = indexerFactory->buildUniqueGlobalIndexer(MPI_COMM_WORLD,physics_blocks,conn_manager);

    // and linear object factory
    Teuchos::RCP<const Epetra_Comm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
    Teuchos::RCP<panzer::EpetraLinearObjFactory<panzer::Traits,int> > elof 
          = Teuchos::rcp(new panzer::EpetraLinearObjFactory<panzer::Traits,int>(comm.getConst(),dofManager));

    Teuchos::RCP<panzer::LinearObjFactory<panzer::Traits> > lof = elof;

    // setup field manager builder
    /////////////////////////////////////////////
      
    // Add in the application specific closure model factory
    user_app::MyModelFactory_TemplateBuilder cm_builder;
    panzer::ClosureModelFactory_TemplateManager<panzer::Traits> cm_factory; 
    cm_factory.buildObjects(cm_builder);

    Teuchos::ParameterList closure_models("Closure Models");
    closure_models.sublist("solid").sublist("SOURCE_TEMPERATURE").set<double>("Value",1.0);
    closure_models.sublist("ion solid").sublist("SOURCE_ION_TEMPERATURE").set<double>("Value",1.0);
    closure_models.sublist("Response Model");

    Teuchos::ParameterList user_data("User Data");
    user_data.sublist("Panzer Data").set("Mesh", mesh);
    user_data.sublist("Panzer Data").set("DOF Manager", dofManager);
    user_data.sublist("Panzer Data").set("Linear Object Factory", lof);
    user_data.set<int>("Workset Size",workset_size);

    // setup and evaluate ResponseLibrary
    ///////////////////////////////////////////////////
 
    out << "Adding responses" << std::endl;

    RCP<ResponseLibrary<Traits> > rLibrary 
          = Teuchos::rcp(new ResponseLibrary<Traits>(wkstContainer,dofManager,lof));

    // register a user defined aggregator
    {
      ResponseAggregator_Manager<Traits> & agMgr= rLibrary->getAggregatorManager();
      ResponseAggregator_IPCoordinates_Builder ipCoordBuilder;
      ipCoordBuilder.setGlobalIndexer(dofManager);
      ipCoordBuilder.setLinearObjFactory(lof);
      agMgr.defineAggregatorTypeFromBuilder("IP Coordinates",ipCoordBuilder);
    }



    //rLibrary->defineDefaultAggregators();
  
    out << "reserving responses" << std::endl;
    std::list<std::string> aggregated_block_names;
    std::list<std::string> aggregated_block_eval_types;
    aggregated_block_names.push_back("eblock-0_0");
    aggregated_block_names.push_back("eblock-1_0");
    aggregated_block_eval_types.push_back("Residual");
    ResponseId aggIpcResp  = buildResponse("Source Term Coordinates","IP Coordinates");
    rLibrary->reserveLabeledVolumeResponse("Aggregated Source Term Coordinates",aggIpcResp,aggregated_block_names,aggregated_block_eval_types);

    rLibrary->printVolumeContainers(out);
  
    out << "building VFM" << std::endl;
    rLibrary->buildVolumeFieldManagersFromResponses(physics_blocks,
						    cm_factory,
						    closure_models,
						    *elof,
						    user_data,
						    true);

    Teuchos::RCP<panzer::LinearObjContainer> loc = elof->buildLinearObjContainer();
    elof->initializeContainer(panzer::EpetraLinearObjContainer::X,*loc);
    Teuchos::RCP<panzer::EpetraLinearObjContainer> eloc = Teuchos::rcp_dynamic_cast<EpetraLinearObjContainer>(loc);
    eloc->x->PutScalar(0.0);

    out << "evaluating VFM" << std::endl;
    panzer::AssemblyEngineInArgs ae_inargs(loc,loc);
    rLibrary->evaluateVolumeFieldManagers<EvalT>(ae_inargs,*tcomm);

    
    Teuchos::RCP<const Response<Traits> > coord_response =
      rLibrary->getVolumeResponseByLabel("Aggregated Source Term Coordinates");
    
    TEST_ASSERT(coord_response->hasParameterList());
    
    Teuchos::RCP<Teuchos::ParameterList> pl = coord_response->getParameterList();
    Teuchos::RCP<std::vector<panzer::IPCoordinates<panzer::Traits::Residual, Traits>::Coordinate> > coords = pl->get<Teuchos::RCP<std::vector<panzer::IPCoordinates<panzer::Traits::Residual, Traits>::Coordinate> > >("IP Coordinates");
    
    // NOTE: if the ordering of elements changes in the STK 2D object, this test may break
    out << "\nPrinting element data for block: eblock-1_0" << std::endl;
    for (std::vector<panzer::IPCoordinates<panzer::Traits::Residual, Traits>::Coordinate>::const_iterator i = coords->begin(); i != coords->end(); ++i)
      out << "pid = " << comm->MyPID() << ", lid = " << i->lid << ", coords(x,y,z) = (" << i->val[0] << "," << i->val[1] << "," << i->val[2] << ")" << std::endl;

    const double double_tol = 10.0*std::numeric_limits<double>::epsilon();
    if (comm->MyPID() == 0) {
      for (std::vector<panzer::IPCoordinates<panzer::Traits::Residual, Traits>::Coordinate>::const_iterator i = coords->begin(); i != coords->end(); ++i) {
	if (i->lid == 0) {
	  TEST_FLOATING_EQUALITY(i->val[0], 1.0, double_tol);
	  TEST_FLOATING_EQUALITY(i->val[1], 1.0, double_tol);
	}
      }
    }
    else if (comm->MyPID() == 1) {
      for (std::vector<panzer::IPCoordinates<panzer::Traits::Residual, Traits>::Coordinate>::const_iterator i = coords->begin(); i != coords->end(); ++i) {
	if (i->lid == 0) {
	  TEST_FLOATING_EQUALITY(i->val[0], 3.0, double_tol);
	  TEST_FLOATING_EQUALITY(i->val[1], 1.0, double_tol);
	}
      }
    }
    

  }

  void testInitialzation(panzer::InputPhysicsBlock& ipb,
			 std::vector<panzer::BC>& bcs)
  {
    panzer::InputEquationSet ies_1;
    ies_1.name = "Energy";
    ies_1.basis = "Q2";
    ies_1.integration_order = 1;
    ies_1.model_id = "solid";
    ies_1.prefix = "";

    panzer::InputEquationSet ies_2;
    ies_2.name = "Energy";
    ies_2.basis = "Q1";
    ies_2.integration_order = 1;
    ies_2.model_id = "ion solid";
    ies_2.prefix = "ION_";

    ipb.physics_block_id = "4";
    ipb.eq_sets.push_back(ies_1);
    ipb.eq_sets.push_back(ies_2);


    {
      std::size_t bc_id = 0;
      panzer::BCType neumann = BCT_Dirichlet;
      std::string sideset_id = "left";
      std::string element_block_id = "eblock-0_0";
      std::string dof_name = "TEMPERATURE";
      std::string strategy = "Constant";
      double value = 5.0;
      Teuchos::ParameterList p;
      p.set("Value",value);
      panzer::BC bc(bc_id, neumann, sideset_id, element_block_id, dof_name, 
		    strategy, p);
      bcs.push_back(bc);
    }    
    {
      std::size_t bc_id = 1;
      panzer::BCType neumann = BCT_Dirichlet;
      std::string sideset_id = "right";
      std::string element_block_id = "eblock-1_0";
      std::string dof_name = "TEMPERATURE";
      std::string strategy = "Constant";
      double value = 5.0;
      Teuchos::ParameterList p;
      p.set("Value",value);
      panzer::BC bc(bc_id, neumann, sideset_id, element_block_id, dof_name, 
		    strategy, p);
      bcs.push_back(bc);
    }   
    {
      std::size_t bc_id = 2;
      panzer::BCType neumann = BCT_Dirichlet;
      std::string sideset_id = "top";
      std::string element_block_id = "eblock-1_0";
      std::string dof_name = "TEMPERATURE";
      std::string strategy = "Constant";
      double value = 5.0;
      Teuchos::ParameterList p;
      p.set("Value",value);
      panzer::BC bc(bc_id, neumann, sideset_id, element_block_id, dof_name, 
		    strategy, p);
      bcs.push_back(bc);
    }
  }

}
