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

/* Compute list of nodes to be sent/received by each rank when duplicating the region interface nodes

We use STK's Selector to find nodes at region interfaces.
It enables boolean operations on element blocks and, thus,
is used to find interface nodes, i.e. nodes that belong to two regions. We find the
intersection of all possible region pairs to identify all interface nodes of a given region.
*/
void computeInterfaceNodes(Teuchos::RCP<const panzer_stk::STK_Interface> mesh,
    const bool print_debug_info, Teuchos::FancyOStream& out)
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

  auto comm = mesh->getComm();

  const int myRank = comm->getRank();
  const int numProcs = comm->getSize();

  std::vector<std::string> eBlocks;
  mesh->getElementBlockNames(eBlocks);

  if (numProcs != eBlocks.size()) throw("Number of MPI ranks and number of regions do not match.");

  Teuchos::RCP<stk::mesh::BulkData> bulk_data = mesh->getBulkData();
  Teuchos::RCP<stk::mesh::MetaData> meta_data = mesh->getMetaData();
  const size_t num_regions = mesh->getNumElementBlocks();

  std::vector<stk::mesh::EntityVector> interface_nodes;
  interface_nodes.resize(num_regions);
  std::vector<Teuchos::Array<int>> interface_node_pids;
  interface_node_pids.resize(num_regions);

  for (size_t my_region_id = 0; my_region_id < num_regions; ++my_region_id)
  {
    stk::mesh::Part* my_region = mesh->getElementBlockPart(eBlocks[my_region_id]);
    stk::mesh::EntityVector& my_interface_nodes = interface_nodes[my_region_id];
    Teuchos::Array<int>& my_interface_node_pids = interface_node_pids[my_region_id];

    for (size_t other_region_id = 0; other_region_id < num_regions; ++other_region_id)
    {
      if (my_region_id != other_region_id)
      {
        // Grab another region and define the intersection operation
        stk::mesh::Part* other_region = mesh->getElementBlockPart(eBlocks[other_region_id]);
        stk::mesh::Selector block_intersection = *my_region & *other_region;

        // Compute intersection and add to list of my interface nodes
        stk::mesh::EntityVector current_interface_nodes;
        bulk_data->get_entities(stk::topology::NODE_RANK, block_intersection, current_interface_nodes);
        my_interface_nodes.insert(my_interface_nodes.end(), current_interface_nodes.begin(), current_interface_nodes.end());

        Teuchos::Array<int> pid_list(current_interface_nodes.size(), static_cast<int>(other_region_id));
        my_interface_node_pids.insert(my_interface_node_pids.end(), pid_list.begin(), pid_list.end());
      }
    }
  }

  // Print region interface nodes
  if (print_debug_info)
  {
    for (size_t region_id_one = 0; region_id_one < num_regions; ++region_id_one)
    {
      std::cout << "Interface nodes of region " << eBlocks[region_id_one] << ":\n";
      for (const auto& node : interface_nodes[region_id_one])
        std::cout << "  " << node;
      std::cout << "\n" << std::endl;
    }
  }

  Teuchos::Array<GlobalOrdinal> sendGIDs; // GIDs of nodes
  Teuchos::Array<int> sendPIDs; // Target

  Teuchos::Array<GO> receiveGIDs;
  Teuchos::Array<int> receivePIDs;

  Teuchos::Array<LO> receiveLIDs;
  Teuchos::Array<LO> sendLIDs;
  Teuchos::Array<LO> interfaceLIDs;

  size_t regionIdx = myRank;
  {
    const stk::mesh::EntityVector& my_interface_nodes = interface_nodes[regionIdx];
    const Teuchos::Array<int>& my_interface_node_pids = interface_node_pids[regionIdx];
    for (size_t node_idx = 0; node_idx < my_interface_nodes.size(); ++node_idx)
    {
      const stk::mesh::Entity& node = my_interface_nodes[node_idx];
      const unsigned node_owner = mesh->entityOwnerRank(node);
      // std::cout << "p=" << myRank << " | node " << node << " owned by proc " << node_owner << std::endl;

      if (myRank != node_owner)
      {
        // Make sure to add GIDs only once
        bool found_it = false;
        for (const auto& node_gid : receiveGIDs)
        {
          if (static_cast<GO>(mesh->elementGlobalId(node)) == node_gid)
          {
            found_it = true;
            break;
          }
        }

        if (!found_it)
        {
          receiveLIDs.push_back(node.local_offset());
          receiveGIDs.push_back(static_cast<GO>(mesh->elementGlobalId(node)));
          receivePIDs.push_back(node_owner);
        }
      }
      else if (myRank == node_owner)
      {
        sendLIDs.push_back(node.local_offset());
        sendGIDs.push_back(static_cast<GO>(mesh->elementGlobalId(node)));
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

  const int numDofsPerNode = 1;
  LO numLocalRegionNodes = -1;
  Teuchos::Array<GlobalOrdinal> quasiRegionNodeGIDs;
  Teuchos::Array<GlobalOrdinal> quasiRegionDofGIDs;

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
      nodeGID = static_cast<GO>(mesh->elementGlobalId(my_nodes[localNodeIdx]));
      quasiRegionNodeGIDs[localNodeIdx] = nodeGID;
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
}
