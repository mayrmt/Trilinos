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
               IJK[2] += 1;
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
          IJK[0] += 1;
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
        dofManager->getElementGIDs(i+IJK[0]*j+IJK[0]*IJK[1]*k,elmGIDs);
        gidRemap[ind] = elmGIDs[0];
        lidRemap[ind++] = dofLID(i+IJK[0]*j+IJK[0]*IJK[1]*k, 0);
        if(i+1==IJK[0]){
          gidRemap[ind] = elmGIDs[1];
          lidRemap[ind++] = dofLID(i+IJK[0]*j+IJK[0]*IJK[1]*k, 1);
        }
      } // i
      if(j+1==IJK[1]){
        for(int i=0; i<IJK[0]; i++){
          dofManager->getElementGIDs(i+IJK[0]*j+IJK[0]*IJK[1]*k,elmGIDs);
          gidRemap[ind] = elmGIDs[3];
          lidRemap[ind++] = dofLID(i+IJK[0]*j+IJK[0]*IJK[1]*k, 3);
          if(i+1==IJK[0]){
            gidRemap[ind] = elmGIDs[2];
            lidRemap[ind++] = dofLID(i+IJK[0]*j+IJK[0]*IJK[1]*k, 2);
          }
        } // i
      }
    } // j
    if(k+1==IJK[2]){
      for(int j=0; j<IJK[1]; j++){
        for(int i=0; i<IJK[0]; i++){
          dofManager->getElementGIDs(i+IJK[0]*j+IJK[0]*IJK[1]*k,elmGIDs);
          gidRemap[ind] = elmGIDs[4];
          lidRemap[ind++] = dofLID(i+IJK[0]*j+IJK[0]*IJK[1]*k, 4);
          if(i+1==IJK[0]){
            gidRemap[ind] = elmGIDs[5];
            lidRemap[ind++] = dofLID(i+IJK[0]*j+IJK[0]*IJK[1]*k, 5);
          }
        } // i
        if(j+1==IJK[1]){
          for(int i=0; i<IJK[0]; i++){
            dofManager->getElementGIDs(i+IJK[0]*j+IJK[0]*IJK[1]*k,elmGIDs);
            gidRemap[ind] = elmGIDs[7];
            lidRemap[ind++] = dofLID(i+IJK[0]*j+IJK[0]*IJK[1]*k, 7);
            if(i+1==IJK[0]){
              gidRemap[ind] = elmGIDs[6];
              lidRemap[ind++] = dofLID(i+IJK[0]*j+IJK[0]*IJK[1]*k, 6);
            }
          } // i
        }
      } // j
    }
  } // k
  //std::cout<<"    | "<<gidRemap<<std::endl;
  return lidRemap;
}
