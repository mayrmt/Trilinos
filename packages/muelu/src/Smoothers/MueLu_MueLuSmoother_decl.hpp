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
#ifndef MUELU_MUELUSMOOTHER_DECL_HPP
#define MUELU_MUELUSMOOTHER_DECL_HPP

#include <Teuchos_ParameterList.hpp>

#include <Xpetra_Matrix_fwd.hpp>
#include <Xpetra_MultiVectorFactory_fwd.hpp>

#include "MueLu_ConfigDefs.hpp"

#include "MueLu_FactoryBase_fwd.hpp"
#include "MueLu_FactoryManagerBase_fwd.hpp"
#include "MueLu_Hierarchy_fwd.hpp"
#include "MueLu_Level_fwd.hpp"
#include "MueLu_SmootherPrototype.hpp"
#include "MueLu_Utilities_fwd.hpp"

namespace MueLu {

  /*!
    @class MueLuSmoother
    @ingroup MueLuSmootherClasses
    @brief Class that encapsulates MueLu as a smoother.

    This class creates an MueLu preconditioner factory. The factory creates a smoother based
    on the type and ParameterList passed into the constructor. See the constructor for more
    information.
    */

  template <class Scalar = SmootherPrototype<>::scalar_type,
            class LocalOrdinal = typename SmootherPrototype<Scalar>::local_ordinal_type,
            class GlobalOrdinal = typename SmootherPrototype<Scalar, LocalOrdinal>::global_ordinal_type,
            class Node = typename SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal>::node_type>
  class MueLuSmoother : public SmootherPrototype<Scalar,LocalOrdinal,GlobalOrdinal,Node> {
#undef MUELU_MUELUSMOOTHER_SHORT
#include "MueLu_UseShortNames.hpp"

   public:

    //! @name Constructors / destructors
    //@{

    //! Constructor
    MueLuSmoother(const Teuchos::ParameterList& paramList);

    //! Destructor
    virtual ~MueLuSmoother() = default;

    //@}

    //! Input
    //! @{

    void DeclareInput(Level& currentLevel) const;

    // void SetParameterList(const Teuchos::ParameterList& paramList);

    //!@}

    //! @name Computational methods.
    //! @{

    /*! @brief Set up the smoother.

    This creates the underlying MueLu hierarchy object, copies any parameter list options
    supplied to the constructor to the MueLu object, and computes the preconditioner.
    */
    void Setup(Level& currentLevel);

    /*! @brief Apply the preconditioner.

    Solves the linear system <tt>AX=B</tt> using the constructed smoother.

    @param X initial guess
    @param B right-hand side
    @param InitialGuessIsZero (optional) If false, some work can be avoided. Whether this actually saves any work depends on the underlying implementation.
    */
    void Apply(MultiVector& X, const MultiVector& B, bool InitialGuessIsZero = false) const;

    //! @}

    //! @name Utilities
    //! @{

    RCP<SmootherPrototype> Copy() const;

    //! @}

    //! @name Overridden from Teuchos::Describable
    //! @{

    //! Return a simple one-line description of this object.
    std::string description() const { return std::string("MueLuSmoother"); };

    //! Print the object to the output stream \c out with some verbosity level \c verbLevel
    void print(Teuchos::FancyOStream &out, const VerbLevel verbLevel = Default) const {};

    //! @}

    //! Get a rough estimate of cost per iteration
    size_t getNodeSmootherComplexity() const;

   private:
    //! Internal AMG hierarchy to be used as smoother
    RCP<Hierarchy> hierarchy_ = Teuchos::null;

    //! matrix, used in apply if solving residual equation
    RCP<Matrix> A_ = Teuchos::null;

  }; // class MueLuSmoother

} // namespace MueLu

#define MUELU_MUELUSMOOTHER_SHORT
#endif // MUELU_MUELUSMOOTHER_DECL_HPP