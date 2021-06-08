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

int checkNodeNeighbor(Kokkos::DynRankView<double,PHX::Device> vertices, const unsigned int elmj, const unsigned int elmk)
{
    unsigned int dim = 3;

    // check x+ (right)
    unsigned int sharedVert = 0;
    for(unsigned int i = 0; i<dim; i++){
       sharedVert += (vertices(elmj,4,i) == vertices(elmk,0,i)) &&
                     (vertices(elmj,5,i) == vertices(elmk,1,i)) &&
                     (vertices(elmj,6,i) == vertices(elmk,2,i)) &&
                     (vertices(elmj,7,i) == vertices(elmk,3,i));
    }
    if( sharedVert == dim )
    { /////std::cout<<"elm "<<elmj<<" is +x from elm "<<elmk<<std::endl;
        return 1;
    }

    // check x- (left)
    sharedVert = 0;
    for(unsigned int i = 0; i<dim; i++){
       sharedVert += (vertices(elmj,0,i) == vertices(elmk,4,i)) &&
                     (vertices(elmj,1,i) == vertices(elmk,5,i)) &&
                     (vertices(elmj,2,i) == vertices(elmk,6,i)) &&
                     (vertices(elmj,3,i) == vertices(elmk,7,i));
    }
    if( sharedVert == dim )
    { //std::cout<<"elm "<<elmj<<" is -x from elm "<<elmk<<std::endl;
        return 2;
    }

    // check y+ (front)
    sharedVert = 0;
    for(unsigned int i = 0; i<dim; i++){
       sharedVert += (vertices(elmj,0,i) == vertices(elmk,3,i)) &&
                     (vertices(elmj,1,i) == vertices(elmk,2,i)) &&
                     (vertices(elmj,4,i) == vertices(elmk,7,i)) &&
                     (vertices(elmj,5,i) == vertices(elmk,6,i));
    }
    if( sharedVert == dim )
    { //std::cout<<"elm "<<elmj<<" is +y from elm "<<elmk<<std::endl;
        return 3;
    }

    // check y- (back)
    sharedVert = 0;
    for(unsigned int i = 0; i<dim; i++){
       sharedVert += (vertices(elmj,3,i) == vertices(elmk,0,i)) &&
                     (vertices(elmj,2,i) == vertices(elmk,1,i)) &&
                     (vertices(elmj,7,i) == vertices(elmk,4,i)) &&
                     (vertices(elmj,6,i) == vertices(elmk,5,i));
    }
    if( sharedVert == dim )
    { //std::cout<<"elm "<<elmj<<" is -y from elm "<<elmk<<std::endl;
        return 4;
    }

    // check z+ (top)
    sharedVert = 0;
    for(unsigned int i = 0; i<dim; i++){
       sharedVert += (vertices(elmj,0,i) == vertices(elmk,1,i)) &&
                     (vertices(elmj,3,i) == vertices(elmk,2,i)) &&
                     (vertices(elmj,4,i) == vertices(elmk,5,i)) &&
                     (vertices(elmj,7,i) == vertices(elmk,6,i));
    }
    if( sharedVert == dim )
    {//std::cout<<"elm "<<elmj<<" is +z from elm "<<elmk<<std::endl;
        return 5;
    }

    // check z- (bottom)
    sharedVert = 0;
    for(unsigned int i = 0; i<dim; i++){
       sharedVert += (vertices(elmj,1,i) == vertices(elmk,0,i)) &&
                     (vertices(elmj,2,i) == vertices(elmk,3,i)) &&
                     (vertices(elmj,5,i) == vertices(elmk,4,i)) &&
                     (vertices(elmj,6,i) == vertices(elmk,7,i));
    }
    if( sharedVert == dim )
    { //std::cout<<"elm "<<elmj<<" is -z from elm "<<elmk<<std::endl;
        return 6;
    }

    // no shared face
    return -1;
}

void reorderLexElem(Kokkos::DynRankView<double,PHX::Device> vertices, Teuchos::Array<panzer::LocalOrdinal> &elemRemap, Teuchos::Array<panzer::LocalOrdinal> &IJK)
{
  int iter = 0;
  int ielem = 0;
  int ielemStart = ielem;
  int ielemStart0 = ielem;
  bool doThis = true, nextElem = true;
  bool icount = true, jcount = true, kcount = true;
  while( doThis == true){
    nextElem = true;
    ielemStart = ielem;
    elemRemap[iter++] = ielem;

    while(nextElem){
       nextElem = false;
       for( int en=0; en<vertices.extent(0); ++en){
         int output = checkNodeNeighbor( vertices, ielem, en);
         if(output == 6){
           if(en == ielemStart){
             nextElem = false; // We have looped in a circle, so stop. (put this check on the others as well.
             break;
           } else {
             ielem = en;
             elemRemap[iter++] = ielem;
             if( kcount )
               IJK[0] += 1;
             nextElem = true;
             break;
           }
         }
       }
    }
    kcount = false;
    if( iter > elemRemap.size() )
        doThis = false;

    //find next start:
    bool moveZ = true;
    for( int en=0; en<vertices.extent(0); ++en){// Move Y
      int output = checkNodeNeighbor( vertices, ielemStart, en);
      if(output == 4){
        ielem = en;
        if( jcount )
          IJK[1] += 1;
        moveZ = false;
        break;
      }
    }
    if(moveZ){
      jcount = false;
      for( int en=0; en<vertices.extent(0); ++en){// If no Move Y, then move Z.
        int output = checkNodeNeighbor( vertices, ielemStart0, en);
        if(output == 1){
          ielem = en;
          IJK[2] += 1;
          ielemStart0 = ielem;
          doThis = true;
          break;
        }
        doThis = false;
      }
    }
  }
  return;
}

Teuchos::Array<panzer::LocalOrdinal> grabLIDsGIDsLexOrder(Teuchos::Array<panzer::LocalOrdinal> IJK,
        Teuchos::Array<panzer::LocalOrdinal> elemRemap,
        const Kokkos::View< const panzer::LocalOrdinal**, Kokkos::LayoutRight, PHX::Device > dofLID,
        Teuchos::RCP<panzer::GlobalIndexer> dofManager,
        int nLID )
{
  using LO = panzer::LocalOrdinal;
  using GO = panzer::GlobalOrdinal;

  std::cout<<"Grab LID in order from Elements"<<std::endl;
  const int numLID = dofLID.extent(1);
  Teuchos::Array<LO> lidRemap(nLID, -1);
  Teuchos::Array<LO> gidRemap(nLID, -1);
  std::vector< GO > elmGIDs;
  int ind = 0;
  //out <<"i "<<i<<" r "<<r<<" lids "<< dofLID(i,r)<<std::endl;
  for(int k=0; k<IJK[2]; k++){
    for(int j=0; j<IJK[1]; j++){
      for(int i=0; i<IJK[0]; i++){
        dofManager->getElementGIDs(elemRemap[i+IJK[0]*j+IJK[0]*IJK[1]*k],elmGIDs);
        gidRemap[ind] = elmGIDs[0];
        lidRemap[ind++] = dofLID(elemRemap[i+IJK[0]*j+IJK[0]*IJK[1]*k], 0);
        if(i+1==IJK[0]){
          gidRemap[ind] = elmGIDs[1];
          lidRemap[ind++] = dofLID(elemRemap[i+IJK[0]*j+IJK[0]*IJK[1]*k], 1);
        }
      } // i
      if(j+1==IJK[1]){
        for(int i=0; i<IJK[0]; i++){
          dofManager->getElementGIDs(elemRemap[i+IJK[0]*j+IJK[0]*IJK[1]*k],elmGIDs);
          gidRemap[ind] = elmGIDs[3];
          lidRemap[ind++] = dofLID(elemRemap[i+IJK[0]*j+IJK[0]*IJK[1]*k], 3);
          if(i+1==IJK[0]){
            gidRemap[ind] = elmGIDs[2];
            lidRemap[ind++] = dofLID(elemRemap[i+IJK[0]*j+IJK[0]*IJK[1]*k], 2);
          }
        } // i
      }
    } // j
    if(k+1==IJK[2]){
      for(int j=0; j<IJK[1]; j++){
        for(int i=0; i<IJK[0]; i++){
          dofManager->getElementGIDs(elemRemap[i+IJK[0]*j+IJK[0]*IJK[1]*k],elmGIDs);
          gidRemap[ind] = elmGIDs[4];
          lidRemap[ind++] = dofLID(elemRemap[i+IJK[0]*j+IJK[0]*IJK[1]*k], 4);
          if(i+1==IJK[0]){
            gidRemap[ind] = elmGIDs[5];
            lidRemap[ind++] = dofLID(elemRemap[i+IJK[0]*j+IJK[0]*IJK[1]*k], 5);
          }
        } // i
        if(j+1==IJK[1]){
          for(int i=0; i<IJK[0]; i++){
            dofManager->getElementGIDs(elemRemap[i+IJK[0]*j+IJK[0]*IJK[1]*k],elmGIDs);
            gidRemap[ind] = elmGIDs[7];
            lidRemap[ind++] = dofLID(elemRemap[i+IJK[0]*j+IJK[0]*IJK[1]*k], 7);
            if(i+1==IJK[0]){
              gidRemap[ind] = elmGIDs[6];
              lidRemap[ind++] = dofLID(elemRemap[i+IJK[0]*j+IJK[0]*IJK[1]*k], 6);
            }
          } // i
        }
      } // j
    }
  } // k
  //std::cout<<"    | "<<gidRemap<<std::endl;
  return lidRemap;
}

// Convert a STK node entity into a node LID
panzer::LocalOrdinal getLIDfromSTKNode(Teuchos::RCP<stk::mesh::BulkData> bulk_data,
    const stk::mesh::Entity& node)
{
  return bulk_data->local_id(node);
}

// Convert a STK node entity into a node GID
panzer::GlobalOrdinal getGIDfromSTKNode(Teuchos::RCP<stk::mesh::BulkData> bulk_data,
    const stk::mesh::Entity& node)
{
  // Need to subtract 1 since STK starts node numbering with 1 (instead of 0)
  return bulk_data->identifier(node) - 1;
}

void findPanzer2StkMapping(Teuchos::RCP<const panzer_stk::STK_Interface> mesh,
        Teuchos::RCP<panzer::GlobalIndexer> dofManager,
        Kokkos::DynRankView<double,PHX::Device> vertices)
{
  using LO = panzer::GlobalOrdinal;

  auto dofLID = dofManager->getLIDs();

  Teuchos::Array<LO> panzer2stk(8*dofLID.extent(0),-1);

  mesh->getComm()->barrier();
  const int myRank = mesh->getComm()->getRank();

  stk::mesh::EntityVector nodes;
  stk::mesh::FieldBase *coordinatesField = mesh->getMetaData()->get_field(stk::topology::NODE_RANK, "coordinates");
  stk::mesh::get_entities(*mesh->getBulkData(), stk::topology::NODE_RANK, mesh->getMetaData()->locally_owned_part(), nodes);
      for(unsigned int ielem=0; ielem<vertices.extent(0); ++ielem){
        for(unsigned int ivert=0; ivert<vertices.extent(1); ++ivert){
          for(size_t nodeIdx = 0; nodeIdx < nodes.size(); ++nodeIdx)
          {
            double *nodeCoord = static_cast<double *>(stk::mesh::field_data(*coordinatesField, nodes[nodeIdx]));
            if(nodeCoord[0] == vertices(ielem,ivert,0) && nodeCoord[1] == vertices(ielem,ivert,1) && nodeCoord[2] == vertices(ielem,ivert,2) ){
              panzer2stk[ dofLID(ielem,ivert) ] = getLIDfromSTKNode(mesh->getBulkData(), nodes[nodeIdx]);
            }
          }
        }
      }
      std::cout << "p=" << myRank <<" | "<< panzer2stk << std::endl;
}

void printNodeCoordinates(Teuchos::RCP<const panzer_stk::STK_Interface> mesh)
{
  using GO = panzer::GlobalOrdinal;

  mesh->getComm()->barrier();
  const int myRank = mesh->getComm()->getRank();

  stk::mesh::EntityVector nodes;
  stk::mesh::FieldBase *coordinatesField = mesh->getMetaData()->get_field(stk::topology::NODE_RANK, "coordinates");
  stk::mesh::get_entities(*mesh->getBulkData(), stk::topology::NODE_RANK, mesh->getMetaData()->locally_owned_part(), nodes);
  for(size_t nodeIdx = 0; nodeIdx < nodes.size(); ++nodeIdx)
  {
    const GO node_gid = getGIDfromSTKNode(mesh->getBulkData(), nodes[nodeIdx]);
    double *nodeCoord = static_cast<double *>(stk::mesh::field_data(*coordinatesField, nodes[nodeIdx]));
    std::cout << "p=" << myRank << " | node " << node_gid << ": (" << nodeCoord[0] << ", " << nodeCoord[1] << ", " << nodeCoord[2] << ")" << std::endl;
  }
}

/* Compute list of nodes to be sent/received by each rank when duplicating the region interface nodes

We use STK's Selector to find nodes at region interfaces.
It enables boolean operations on element blocks and, thus,
is used to find interface nodes, i.e. nodes that belong to two regions. We find the
intersection of all possible region pairs to identify all interface nodes of a given region.
*/
void computeInterfaceNodes(
    Teuchos::RCP<const panzer_stk::STK_Interface> mesh, ///< STK mesh object with all element blocks etc.
    const bool print_debug_info, ///< flag to switch on/off debug outpu
    Teuchos::FancyOStream& out, ///< output stream
    const int numDofsPerNode, ///< number of DOFs per mesh node
    Teuchos::Array<panzer::GlobalOrdinal>& quasiRegionNodeGIDs, ///< This rank's node GIDs in quasiRegion format
    Teuchos::Array<panzer::GlobalOrdinal>& quasiRegionDofGIDs ///< This rank's DOF GIDs in quasiRegion format
    )
{
  // Panzer types
  using ST = double;
  using LO = panzer::LocalOrdinal;
  using GO = panzer::GlobalOrdinal;
  using NT = panzer::TpetraNodeType;

  // MueLu types
  using Scalar = ST;
  using LocalOrdinal = LO;
  using GlobalOrdinal = GO;
  using Node = NT;

  using Teuchos::Array;

  auto comm = mesh->getComm();

  const int myRank = comm->getRank();
  const int numProcs = comm->getSize();

  std::vector<std::string> eBlocks;
  mesh->getElementBlockNames(eBlocks);

  if (numProcs != eBlocks.size()) throw("Number of MPI ranks and number of regions do not match.");

  Teuchos::RCP<stk::mesh::BulkData> bulk_data = mesh->getBulkData();
  Teuchos::RCP<stk::mesh::MetaData> meta_data = mesh->getMetaData();
  const size_t num_regions = mesh->getNumElementBlocks();

  Array<Array<GO>> interface_nodes;
  interface_nodes.resize(num_regions);
  Array<stk::mesh::EntityVector> interface_nodes_stk;
  interface_nodes_stk.resize(num_regions);
  Array<Array<int>> interface_node_pids;
  interface_node_pids.resize(num_regions);

  for (size_t my_region_id = 0; my_region_id < num_regions; ++my_region_id)
  {
    stk::mesh::Part* my_region = mesh->getElementBlockPart(eBlocks[my_region_id]);
    Array<GO>& my_interface_nodes = interface_nodes[my_region_id];
    stk::mesh::EntityVector& my_interface_nodes_stk = interface_nodes_stk[my_region_id];
    Array<int>& my_interface_node_pids = interface_node_pids[my_region_id];

    for (size_t other_region_id = 0; other_region_id < num_regions; ++other_region_id)
    {
      if (my_region_id != other_region_id)
      {
        // Grab another region and define the intersection operation
        stk::mesh::Part* other_region = mesh->getElementBlockPart(eBlocks[other_region_id]);
        stk::mesh::Selector block_intersection = *my_region & *other_region;

        // Compute intersection and add to list of my interface nodes
        stk::mesh::EntityVector current_interface_nodes_stk;
        bulk_data->get_entities(stk::topology::NODE_RANK, block_intersection, current_interface_nodes_stk);
        my_interface_nodes_stk.insert(my_interface_nodes_stk.end(), current_interface_nodes_stk.begin(), current_interface_nodes_stk.end());

        // Convert STK entities to unique GIDs
        Array<GO> current_interface_node_GIDs;
        current_interface_node_GIDs.reserve(current_interface_nodes_stk.size());
        for (const auto& node : current_interface_nodes_stk)
          current_interface_node_GIDs.push_back(getGIDfromSTKNode(bulk_data, node));
        my_interface_nodes.insert(my_interface_nodes.end(), current_interface_node_GIDs.begin(), current_interface_node_GIDs.end());

        Array<int> pid_list(current_interface_nodes_stk.size(), static_cast<int>(other_region_id));
        my_interface_node_pids.insert(my_interface_node_pids.end(), pid_list.begin(), pid_list.end());
      }
    }
  }

  // Print region interface nodes
  if (print_debug_info)
  {
    comm->barrier();
    std::cout << "p=" << myRank << " | Interface nodes of region " << eBlocks[myRank] << ":\n";
    for (const auto& node : interface_nodes[myRank])
      std::cout << "  " << node;
    std::cout << "\n" << std::endl;
  }

  Array<GO> sendGIDs; // GIDs of nodes
  Array<LO> sendLIDs; // LIDs of nodes
  Array<int> sendPIDs; // Target processor

  Array<GO> receiveGIDs; // GIDs of nodes
  Array<LO> receiveLIDs; // LIDs of nodes
  Array<int> receivePIDs; // Source processor

  Array<LO> interfaceLIDs;

  stk::mesh::Part* my_own_region = mesh->getElementBlockPart(eBlocks[myRank]);
  stk::mesh::Selector select_myself = *my_own_region;
  stk::mesh::EntityVector my_region_nodes;
  bulk_data->get_entities(stk::topology::NODE_RANK, select_myself, my_region_nodes);
  LO new_lid = static_cast<LO>(my_region_nodes.size() - 1);

  size_t regionIdx = myRank;
  {
    const Array<GO>& my_interface_nodes = interface_nodes[regionIdx];
    const stk::mesh::EntityVector& my_interface_nodes_stk = interface_nodes_stk[regionIdx];
    const Array<int>& my_interface_node_pids = interface_node_pids[regionIdx];
    for (size_t node_idx = 0; node_idx < my_interface_nodes.size(); ++node_idx)
    {
      const GO node_gid = my_interface_nodes[node_idx];
      const stk::mesh::Entity& node = my_interface_nodes_stk[node_idx];
      const unsigned node_owner = bulk_data->parallel_owner_rank(node);
      std::cout << "p=" << myRank << " | node " << node_gid << " owned by proc " << node_owner << std::endl;

      if (myRank != node_owner)
      {
        // Make sure to add GIDs only once
        bool found_it = false;
        for (const auto& node_gid : receiveGIDs)
        {
          if (my_interface_nodes[node_idx] == node_gid)
          {
            found_it = true;
            break;
          }
        }

        if (!found_it)
        {
          // LID needs to be created, so increment the last LID to append the received DOF/Node.
          ++new_lid;
          receiveLIDs.push_back(new_lid);
          receiveGIDs.push_back(my_interface_nodes[node_idx]);
          receivePIDs.push_back(node_owner);
        }
      }
      // else if (myRank == node_owner && getLIDfromSTKNode(bulk_data, node) != -Teuchos::ScalarTraits<LO>::one())
      else //if (getLIDfromSTKNode(bulk_data, node) != -Teuchos::ScalarTraits<LO>::one())
      {
        sendLIDs.push_back(getLIDfromSTKNode(bulk_data, node));
        sendGIDs.push_back(my_interface_nodes[node_idx]);
        sendPIDs.push_back(my_interface_node_pids[node_idx]);
      }
    }
  }

  const LO numSend = static_cast<LO>(sendGIDs.size());
  const LO numReceive = static_cast<LO>(receiveGIDs.size());

  if (print_debug_info)
  {
    out << std::endl;
    for (int rank = 0; rank < num_regions; ++rank)
    {
      comm->barrier();
      if (rank == myRank)
      {
        std::cout << "p=" << myRank << " | sendLIDs = " << sendLIDs << std::endl;
        std::cout << "p=" << myRank << " | sendGIDs = " << sendGIDs << std::endl;
        std::cout << "p=" << myRank << " | sendPIDs = " << sendPIDs << std::endl;

        std::cout << "p=" << myRank << " | receiveLIDs = " << receiveLIDs << std::endl;
        std::cout << "p=" << myRank << " | receiveGIDs = " << receiveGIDs << std::endl;
        std::cout << "p=" << myRank << " | receivePIDs = " << receivePIDs << std::endl;
        std::cout << std::endl;
      }
    }
  }

  stk::mesh::Part* my_region = mesh->getElementBlockPart(eBlocks[myRank]);
  stk::mesh::Selector local_region = *my_region & meta_data->locally_owned_part();
  stk::mesh::EntityVector my_nodes;
  bulk_data->get_entities(stk::topology::NODE_RANK, local_region, my_nodes);
  const LO numLocalCompositeNodes = static_cast<LO>(my_nodes.size());

  LO numLocalRegionNodes = -1;

  numLocalRegionNodes = numLocalCompositeNodes + numReceive;
  quasiRegionNodeGIDs.resize(numLocalRegionNodes);
  quasiRegionDofGIDs.resize(numLocalRegionNodes*numDofsPerNode);

  if (print_debug_info)
  {
    comm->barrier();
    std::cout << "p=" << myRank << " | numLocalCompositeNodes = " << numLocalCompositeNodes
        << "\tnumLocalRegionNodes = " << numLocalRegionNodes
        << std::endl;
    comm->barrier();
  }

  {
    GO nodeGID = -Teuchos::ScalarTraits<GO>::one();
    for (size_t localNodeIdx = 0; localNodeIdx < numLocalCompositeNodes; ++localNodeIdx)
    {
      // Double-check for STK's weird numbering scheme which does NOT start with zero
      TEUCHOS_ASSERT(my_nodes[localNodeIdx].local_offset()!=Teuchos::ScalarTraits<LO>::zero());
      TEUCHOS_ASSERT(bulk_data->identifier(my_nodes[localNodeIdx])!=Teuchos::ScalarTraits<GO>::zero());

      quasiRegionNodeGIDs[localNodeIdx] = getGIDfromSTKNode(bulk_data, my_nodes[localNodeIdx]);
      // for (int dof = 0; dof < numDofsPerNode; ++dof)
      //   quasiRegionDofGIDs[localNodeIdx * numDofsPerNode + dof] = nodeGID * numDofsPerNode + dof;
    }
    for (size_t receiveNodeIdx = 0; receiveNodeIdx < numReceive; ++ receiveNodeIdx)
    {
      quasiRegionNodeGIDs[numLocalCompositeNodes + receiveNodeIdx] = receiveGIDs[receiveNodeIdx];
      // for (int dof = 0; dof < numDofsPerNode; ++dof)
      //   quasiRegionDofGIDs[numLocalCompositeNodes * numDofsPerNode + receiveNodeIdx * numDofsPerNode + dof]
      //       = receiveGIDs[receiveNodeIdx];
    }

    if (print_debug_info)
    {
      comm->barrier();
      std::cout << "p=" << myRank << " | quasiRegionNodeGIDs = " << quasiRegionNodeGIDs << std::endl;
    }
  }

  // For now, we're limited to one Dof per node, so (i) assert and (ii) copy node GIDs to DOF GIDs
  TEUCHOS_TEST_FOR_EXCEPTION(numDofsPerNode!=1, std::runtime_error,
      "Case with numDofsPerNode != 1 not supported, yet.");
  quasiRegionDofGIDs = quasiRegionNodeGIDs;

  // Here we gather the interface GIDs (in composite layout)
  // and the interface LIDs (in region layout) for the local rank
  Teuchos::Array<LocalOrdinal> interfaceLIDsData;
  Teuchos::Array<GlobalOrdinal> interfaceGIDs;
  interfaceLIDsData.resize((sendGIDs.size() + receiveGIDs.size()) * numDofsPerNode);
  interfaceGIDs.resize((sendGIDs.size() + receiveGIDs.size()) * numDofsPerNode);
  using size_type = typename Teuchos::Array<GO>::size_type;
  for(size_type nodeIdx = 0; nodeIdx < sendGIDs.size(); ++nodeIdx)
  {
    for(int dof = 0; dof < numDofsPerNode; ++dof)
    {
      LO dofIdx = nodeIdx*numDofsPerNode + dof;
      interfaceGIDs[dofIdx] = sendGIDs[nodeIdx] * numDofsPerNode + dof;
      // interfaceLIDsData[dofIdx] = compositeToRegionLIDs[sendLIDs[nodeIdx] * numDofsPerNode + dof];
    }
  }
  for(size_type nodeIdx = 0; nodeIdx < receiveGIDs.size(); ++nodeIdx)
  {
    for(int dof = 0; dof < numDofsPerNode; ++dof)
    {
      LO dofIdx = nodeIdx*numDofsPerNode + dof;
      interfaceGIDs[dofIdx + sendGIDs.size() * numDofsPerNode] = receiveGIDs[nodeIdx] * numDofsPerNode + dof;
      // interfaceLIDsData[dofIdx + sendLIDs.size() * numDofsPerNode] = receiveLIDs[nodeIdx] * numDofsPerNode + dof;
    }
  }

  // Have all the GIDs and LIDs we stort them in place with std::sort()
  // Subsequently we bring unique values to the beginning of the array with
  // std::unique() and delete the duplicates with erase.
  // std::sort(interfaceLIDsData.begin(), interfaceLIDsData.end());
  // interfaceLIDsData.erase(std::unique(interfaceLIDsData.begin(), interfaceLIDsData.end()),
  //                         interfaceLIDsData.end());
  std::sort(interfaceGIDs.begin(), interfaceGIDs.end());
  interfaceGIDs.erase(std::unique(interfaceGIDs.begin(), interfaceGIDs.end()),
                      interfaceGIDs.end());

  if (print_debug_info)
  {
    comm->barrier();
    std::cout << "p=" << myRank << " | interfaceGIDs = " << interfaceGIDs << std::endl;
  }
}

void setupRegionMaps(
    RCP<const Teuchos::Comm<int>> comm,
    const Teuchos::Array<panzer::GlobalOrdinal>& quasiRegionDofGIDs, ///< This rank's DOF GIDs in quasiRegion format
    RCP<Xpetra::Map<panzer::LocalOrdinal, panzer::GlobalOrdinal, panzer::TpetraNodeType>>& quasiRegionRowMap,
    RCP<Xpetra::Map<panzer::LocalOrdinal, panzer::GlobalOrdinal, panzer::TpetraNodeType>>& quasiRegionColMap,
    RCP<Xpetra::Map<panzer::LocalOrdinal, panzer::GlobalOrdinal, panzer::TpetraNodeType>>& regionRowMap,
    RCP<Xpetra::Map<panzer::LocalOrdinal, panzer::GlobalOrdinal, panzer::TpetraNodeType>>& regionColMap
    )
{
  // Panzer types
  using ST = double;
  using LO = panzer::LocalOrdinal;
  using GO = panzer::GlobalOrdinal;
  using NT = panzer::TpetraNodeType;

  // MueLu types
  using Scalar = ST;
  using LocalOrdinal = LO;
  using GlobalOrdinal = GO;
  using Node = NT;

#include <Xpetra_UseShortNames.hpp>

  quasiRegionRowMap = Xpetra::MapFactory<LO,GO,Node>::Build(Xpetra::UseTpetra, Teuchos::OrdinalTraits<GO>::invalid(),
      quasiRegionDofGIDs(), Teuchos::OrdinalTraits<GO>::zero(), comm);

  regionRowMap = Xpetra::MapFactory<LO,GO,Node>::Build(quasiRegionRowMap->lib(), Teuchos::OrdinalTraits<GO>::invalid(),
      quasiRegionDofGIDs.size(), quasiRegionRowMap->getIndexBase(), quasiRegionRowMap->getComm());

  // For now, column map = row map
  quasiRegionColMap = quasiRegionRowMap;
  regionColMap = regionRowMap;

}
