// @HEADER
// ************************************************************************
//
//               Rapid Optimization Library (ROL) Package
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact lead developers:
//              Drew Kouri   (dpkouri@sandia.gov) and
//              Denis Ridzal (dridzal@sandia.gov)
//
// ************************************************************************
// @HEADER

#ifndef ROL_TRUSTREGIONPSGALGORITHM_B_H
#define ROL_TRUSTREGIONPSGALGORITHM_B_H

#include "ROL_Algorithm_B.hpp"
#include "ROL_TrustRegionModel_U.hpp"
#include "ROL_TrustRegionUtilities.hpp"

/** \class ROL::TrustRegionPSGAlgorithm_B
    \brief Provides an interface to run the trust-region algorithm of Lin and More.
*/

namespace ROL {

template<typename Real>
class TrustRegionPSGAlgorithm_B : public Algorithm_B<Real> {
private:
  Ptr<TrustRegionModel_U<Real>> model_;  ///< Container for trust-region model

  // TRUST REGION PARAMETERS
  Real delMax_; ///< Maximum trust-region radius (default: ROL_INF)
  Real eta0_;   ///< Step acceptance threshold (default: 0.05)
  Real eta1_;   ///< Radius decrease threshold (default: 0.05)
  Real eta2_;   ///< Radius increase threshold (default: 0.9)
  Real gamma0_; ///< Radius decrease rate (negative rho) (default: 0.0625)
  Real gamma1_; ///< Radius decrease rate (positive rho) (default: 0.25)
  Real gamma2_; ///< Radius increase rate (default: 2.5)
  Real TRsafe_; ///< Safeguard size for numerically evaluating ratio (default: 1e2)
  Real eps_;    ///< Safeguard for numerically evaluating ratio
  bool interpRad_; ///< Interpolate the trust-region radius if ratio is negative (default: false)

  // ITERATION FLAGS/INFORMATION
  TRUtils::ETRFlag TRflag_; ///< Trust-region exit flag
  int SPflag_;              ///< Subproblem solver termination flag
  int SPiter_;              ///< Subproblem solver iteration count

  // SECANT INFORMATION
  ESecant esec_;          ///< Secant type (default: Limited-Memory BFGS)
  bool useSecantPrecond_; ///< Flag to use secant as a preconditioner (default: false)
  bool useSecantHessVec_; ///< Flag to use secant as Hessian (default: false)

  // TRUNCATED CG INFORMATION
  Real tol1_; ///< Absolute tolerance for truncated CG (default: 1e-4)
  Real tol2_; ///< Relative tolerance for truncated CG (default: 1e-2)
  int maxit_; ///< Maximum number of CG iterations (default: 20)

  // ALGORITHM SPECIFIC PARAMETERS
  Real mu0_;       ///< Sufficient decrease parameter (default: 1e-2)
  Real spexp_;     ///< Relative tolerance exponent for subproblem solve (default: 1, range: [1,2])
  int  redlim_;    ///< Maximum number of Cauchy point reduction steps (default: 10)
  int  explim_;    ///< Maximum number of Cauchy point expansion steps (default: 10)
  Real alpha_;     ///< Initial Cauchy point step length (default: 1.0)
  bool normAlpha_; ///< Normalize initial Cauchy point step length (default: false)
  Real interpf_;   ///< Backtracking rate for Cauchy point computation (default: 1e-1)
  Real extrapf_;   ///< Extrapolation rate for Cauchy point computation (default: 1e1)
  Real qtol_;      ///< Relative tolerance for computed decrease in Cauchy point computation (default: 1-8)
  Real lambdaMin_;
  Real lambdaMax_;
  Real gamma_;
  Real rhodec_;
  Real sigma1_;
  Real sigma2_;
  int lsmax_;
  int maxSize_;

  mutable int nhess_;  ///< Number of Hessian applications
  unsigned verbosity_; ///< Output level (default: 0)
  bool printHeader_;   ///< Flag to print header at every iteration

  using Algorithm_B<Real>::state_;
  using Algorithm_B<Real>::status_;
  using Algorithm_B<Real>::proj_;

public:
  TrustRegionPSGAlgorithm_B(ParameterList &list, const Ptr<Secant<Real>> &secant = nullPtr);

  using Algorithm_B<Real>::run;
  std::vector<std::string> run( Vector<Real>          &x,
                                const Vector<Real>    &g, 
                                Objective<Real>       &obj,
                                BoundConstraint<Real> &bnd,
                                std::ostream          &outStream = std::cout);

  std::string printHeader( void ) const;

  std::string printName( void ) const;

  std::string print( const bool print_header = false ) const;

private:
  void initialize(Vector<Real>          &x,
                  const Vector<Real>    &g,
                  Objective<Real>       &obj,
                  BoundConstraint<Real> &bnd,
                  std::ostream &outStream = std::cout);

  // Compute the projected step s = P(x + alpha*w) - x
  // Returns the norm of the projected step s
  //    s     -- The projected step upon return
  //    w     -- The direction vector w (unchanged)
  //    x     -- The anchor vector x (unchanged)
  //    alpha -- The step size (unchanged)
  Real dgpstep(Vector<Real> &s, const Vector<Real> &w,
         const Vector<Real> &x, const Real alpha,
         std::ostream &outStream = std::cout) const;

  // Compute Cauchy point, i.e., the minimizer of q(P(x - alpha*g)-x)
  // subject to the trust region constraint ||P(x - alpha*g)-x|| <= del
  //   s     -- The Cauchy step upon return: Primal optimization space vector
  //   alpha -- The step length for the Cauchy point upon return
  //   x     -- The anchor vector x (unchanged): Primal optimization space vector
  //   g     -- The (dual) gradient vector g (unchanged): Primal optimization space vector
  //   del   -- The trust region radius (unchanged)
  //   model -- Trust region model
  //   dwa   -- Dual working array, stores Hessian applied to step
  //   dwa1  -- Dual working array
  Real dcauchy(Vector<Real> &s, Real &alpha, Real &q,
               const Vector<Real> &x, const Vector<Real> &g,
               const Real del, TrustRegionModel_U<Real> &model,
               Vector<Real> &dwa, Vector<Real> &dwa1,
               std::ostream &outStream = std::cout);

  void dpsg(Vector<Real> &y, Real &q, Vector<Real> &gmod,
            const Vector<Real> &x, Real del, TrustRegionModel_U<Real> &model,
            Vector<Real> &pwa, Vector<Real> &pwa1, Vector<Real> &pwa2,
            Vector<Real> &pwa3, Vector<Real> &dwa,
            std::ostream &outStream = std::cout);

  void dproj(Vector<Real> &x, const Vector<Real> &x0, Real del,
            Vector<Real> &y, Vector<Real> &p,
            std::ostream &outStream = std::cout) const;

}; // class ROL::TrustRegionPSGAlgorithm_B

} // namespace ROL

#include "ROL_TrustRegionPSGAlgorithm_B_Def.hpp"

#endif
