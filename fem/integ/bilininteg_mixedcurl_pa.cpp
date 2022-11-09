// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../../general/forall.hpp"
#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "../qfunction.hpp"
#include "bilininteg_hcurl_kernels.hpp"
#include "bilininteg_hcurlhdiv_kernels.hpp"

using namespace std;

namespace mfem
{

void MixedScalarCurlIntegrator::AssemblePA(const FiniteElementSpace &trial_fes,
                                           const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *fel = trial_fes.GetFE(0); // In H(curl)
   const FiniteElement *eltest = test_fes.GetFE(0); // In scalar space

   const VectorTensorFiniteElement *el =
      dynamic_cast<const VectorTensorFiniteElement*>(fel);
   MFEM_VERIFY(el != NULL, "Only VectorTensorFiniteElement is supported!");

   if (el->GetDerivType() != mfem::FiniteElement::CURL)
   {
      MFEM_ABORT("Unknown kernel.");
   }

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*eltest, *eltest,
                                                     *mesh->GetElementTransformation(0));

   const int dims = el->GetDim();
   MFEM_VERIFY(dims == 2, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2, "");

   ne = test_fes.GetNE();
   mapsC = &el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   if (el->GetOrder() == eltest->GetOrder())
   {
      dofs1Dtest = dofs1D;
   }
   else
   {
      dofs1Dtest = dofs1D - 1;
   }

   pa_data.SetSize(nq * ne, Device::GetMemoryType());

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::FULL);

   if (dim == 2)
   {
      internal::PAHcurlL2Setup2D(quad1D, ne, ir->GetWeights(), coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

void MixedScalarCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 2)
   {
      internal::PAHcurlL2Apply2D(dofs1D, dofs1Dtest, quad1D, ne, mapsO->B, mapsO->Bt,
                                 mapsC->Bt, mapsC->G, pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

void MixedScalarCurlIntegrator::AddMultTransposePA(const Vector &x,
                                                   Vector &y) const
{
   if (dim == 2)
   {
      internal::PAHcurlL2ApplyTranspose2D(dofs1D, dofs1Dtest, quad1D, ne, mapsO->B,
                                          mapsO->Bt,
                                          mapsC->B, mapsC->Gt, pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

// Apply to x corresponding to DOFs in H(curl) (trial), whose curl is
// integrated against H(div) test functions corresponding to y.
template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
static void PAHcurlHdivApply3D(const int D1D,
                               const int D1Dtest,
                               const int Q1D,
                               const int NE,
                               const Array<double> &bo,
                               const Array<double> &bc,
                               const Array<double> &bot,
                               const Array<double> &bct,
                               const Array<double> &gc,
                               const Vector &pa_data,
                               const Vector &x,
                               Vector &y)
{
   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   // Using Piola transformations (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u}
   // for u in H(curl) and w = (1 / det (dF)) dF \hat{w} for w in H(div), we get
   // (\nabla\times u) \cdot w = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{w}
   // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
   // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
   // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

   constexpr static int VDIM = 3;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1Dtest-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1Dtest, Q1D);
   auto Gc = Reshape(gc.Read(), Q1D, D1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto X = Reshape(x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1Dtest-1)*(D1Dtest-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double curl[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];
      // curl[qz][qy][qx] will be computed as the vector curl at each quadrature point.

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  curl[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      // We treat x, y, z components separately for optimization specific to each.

      int osc = 0;

      {
         // x component
         const int D1Dz = D1D;
         const int D1Dy = D1D;
         const int D1Dx = D1D - 1;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double gradXY[MAX_Q1D][MAX_Q1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massX[MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * Bo(qx,dx);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = Bc(qy,dy);
                  const double wDy = Gc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = massX[qx];
                     gradXY[qy][qx][0] += wx * wDy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = Bc(qz,dz);
               const double wDz = Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     curl[qz][qy][qx][1] += gradXY[qy][qx][1] * wDz; // (u_0)_{x_2}
                     curl[qz][qy][qx][2] -= gradXY[qy][qx][0] * wz;  // -(u_0)_{x_1}
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // y component
         const int D1Dz = D1D;
         const int D1Dy = D1D - 1;
         const int D1Dx = D1D;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double gradXY[MAX_Q1D][MAX_Q1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               double massY[MAX_Q1D];
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  massY[qy] = 0.0;
               }

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     massY[qy] += t * Bo(qy,dy);
                  }
               }

               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx = Bc(qx,dx);
                  const double wDx = Gc(qx,dx);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wy = massY[qy];
                     gradXY[qy][qx][0] += wDx * wy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = Bc(qz,dz);
               const double wDz = Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     curl[qz][qy][qx][0] -= gradXY[qy][qx][1] * wDz; // -(u_1)_{x_2}
                     curl[qz][qy][qx][2] += gradXY[qy][qx][0] * wz;  // (u_1)_{x_0}
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // z component
         const int D1Dz = D1D - 1;
         const int D1Dy = D1D;
         const int D1Dx = D1D;

         for (int dx = 0; dx < D1Dx; ++dx)
         {
            double gradYZ[MAX_Q1D][MAX_Q1D][2];
            for (int qz = 0; qz < Q1D; ++qz)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradYZ[qz][qy][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massZ[MAX_Q1D];
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  massZ[qz] = 0.0;
               }

               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     massZ[qz] += t * Bo(qz,dz);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = Bc(qy,dy);
                  const double wDy = Gc(qy,dy);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const double wz = massZ[qz];
                     gradYZ[qz][qy][0] += wz * wy;
                     gradYZ[qz][qy][1] += wz * wDy;
                  }
               }
            }

            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double wx = Bc(qx,dx);
               const double wDx = Gc(qx,dx);

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     curl[qz][qy][qx][0] += gradYZ[qz][qy][1] * wx;  // (u_2)_{x_1}
                     curl[qz][qy][qx][1] -= gradYZ[qz][qy][0] * wDx; // -(u_2)_{x_0}
                  }
               }
            }
         }
      }

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double O11 = op(qx,qy,qz,0,e);
               const double O12 = op(qx,qy,qz,1,e);
               const double O13 = op(qx,qy,qz,2,e);
               const double O22 = op(qx,qy,qz,3,e);
               const double O23 = op(qx,qy,qz,4,e);
               const double O33 = op(qx,qy,qz,5,e);

               const double c1 = (O11 * curl[qz][qy][qx][0]) + (O12 * curl[qz][qy][qx][1]) +
                                 (O13 * curl[qz][qy][qx][2]);
               const double c2 = (O12 * curl[qz][qy][qx][0]) + (O22 * curl[qz][qy][qx][1]) +
                                 (O23 * curl[qz][qy][qx][2]);
               const double c3 = (O13 * curl[qz][qy][qx][0]) + (O23 * curl[qz][qy][qx][1]) +
                                 (O33 * curl[qz][qy][qx][2]);

               curl[qz][qy][qx][0] = c1;
               curl[qz][qy][qx][1] = c2;
               curl[qz][qy][qx][2] = c3;
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         double massXY[HCURL_MAX_D1D][HCURL_MAX_D1D];  // Assuming HDIV_MAX_D1D <= HCURL_MAX_D1D

         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1Dtest : D1Dtest - 1;
            const int D1Dy = (c == 1) ? D1Dtest : D1Dtest - 1;
            const int D1Dx = (c == 0) ? D1Dtest : D1Dtest - 1;

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massXY[dy][dx] = 0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massX[HCURL_MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massX[dx] += curl[qz][qy][qx][c] *
                                  ((c == 0) ? Bct(dx,qx) : Bot(dx,qx));
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = (c == 1) ? Bct(dy,qy) : Bot(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = (c == 2) ? Bct(dz,qz) : Bot(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) +=
                        massXY[dy][dx] * wz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop
}

// Apply to x corresponding to DOFs in H(div) (test), integrated against the
// curl of H(curl) trial functions corresponding to y.
template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
static void PAHcurlHdivApply3DTranspose(const int D1D,
                                        const int D1Dtest,
                                        const int Q1D,
                                        const int NE,
                                        const Array<double> &bo,
                                        const Array<double> &bc,
                                        const Array<double> &bot,
                                        const Array<double> &bct,
                                        const Array<double> &gct,
                                        const Vector &pa_data,
                                        const Vector &x,
                                        Vector &y)
{
   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   // Using Piola transformations (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u}
   // for u in H(curl) and w = (1 / det (dF)) dF \hat{w} for w in H(div), we get
   // (\nabla\times u) \cdot w = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{w}
   // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
   // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
   // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

   constexpr static int VDIM = 3;

   auto Bo = Reshape(bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(bc.Read(), Q1D, D1D);
   auto Bot = Reshape(bot.Read(), D1Dtest-1, Q1D);
   auto Bct = Reshape(bct.Read(), D1Dtest, Q1D);
   auto Gct = Reshape(gct.Read(), D1D, Q1D);
   auto op = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto X = Reshape(x.Read(), 3*(D1Dtest-1)*(D1Dtest-1)*D1D, NE);
   auto Y = Reshape(y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double mass[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];  // Assuming HDIV_MAX_D1D <= HCURL_MAX_D1D

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  mass[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D : D1D - 1;
         const int D1Dy = (c == 1) ? D1D : D1D - 1;
         const int D1Dx = (c == 0) ? D1D : D1D - 1;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double massXY[HDIV_MAX_Q1D][HDIV_MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massX[HDIV_MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = X(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * ((c == 0) ? Bc(qx,dx) : Bo(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 1) ? Bc(qy,dy) : Bo(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = massX[qx];
                     massXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = (c == 2) ? Bc(qz,dz) : Bo(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     mass[qz][qy][qx][c] += massXY[qy][qx] * wz;
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop (c) over components

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double O11 = op(qx,qy,qz,0,e);
               const double O12 = op(qx,qy,qz,1,e);
               const double O13 = op(qx,qy,qz,2,e);
               const double O22 = op(qx,qy,qz,3,e);
               const double O23 = op(qx,qy,qz,4,e);
               const double O33 = op(qx,qy,qz,5,e);
               const double massX = mass[qz][qy][qx][0];
               const double massY = mass[qz][qy][qx][1];
               const double massZ = mass[qz][qy][qx][2];
               mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
               mass[qz][qy][qx][1] = (O12*massX)+(O22*massY)+(O23*massZ);
               mass[qz][qy][qx][2] = (O13*massX)+(O23*massY)+(O33*massZ);
            }
         }
      }

      // x component
      osc = 0;
      {
         const int D1Dz = D1D;
         const int D1Dy = D1D;
         const int D1Dx = D1D - 1;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            double gradXY12[MAX_D1D][MAX_D1D];
            double gradXY21[MAX_D1D][MAX_D1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradXY12[dy][dx] = 0.0;
                  gradXY21[dy][dx] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massX[MAX_D1D][2];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     massX[dx][n] = 0.0;
                  }
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     const double wx = Bot(dx,qx);

                     massX[dx][0] += wx * mass[qz][qy][qx][1];
                     massX[dx][1] += wx * mass[qz][qy][qx][2];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = Bct(dy,qy);
                  const double wDy = Gct(dy,qy);

                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     gradXY21[dy][dx] += massX[dx][0] * wy;
                     gradXY12[dy][dx] += massX[dx][1] * wDy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = Bct(dz,qz);
               const double wDz = Gct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     // (u_0)_{x_2} * (op * curl)_1 - (u_0)_{x_1} * (op * curl)_2
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                       e) += (gradXY21[dy][dx] * wDz) - (gradXY12[dy][dx] * wz);
                  }
               }
            }
         }  // loop qz

         osc += D1Dx * D1Dy * D1Dz;
      }

      // y component
      {
         const int D1Dz = D1D;
         const int D1Dy = D1D - 1;
         const int D1Dx = D1D;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            double gradXY02[MAX_D1D][MAX_D1D];
            double gradXY20[MAX_D1D][MAX_D1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradXY02[dy][dx] = 0.0;
                  gradXY20[dy][dx] = 0.0;
               }
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               double massY[MAX_D1D][2];
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  massY[dy][0] = 0.0;
                  massY[dy][1] = 0.0;
               }
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     const double wy = Bot(dy,qy);

                     massY[dy][0] += wy * mass[qz][qy][qx][2];
                     massY[dy][1] += wy * mass[qz][qy][qx][0];
                  }
               }
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double wx = Bct(dx,qx);
                  const double wDx = Gct(dx,qx);

                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     gradXY02[dy][dx] += massY[dy][0] * wDx;
                     gradXY20[dy][dx] += massY[dy][1] * wx;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = Bct(dz,qz);
               const double wDz = Gct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     // -(u_1)_{x_2} * (op * curl)_0 + (u_1)_{x_0} * (op * curl)_2
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                       e) += (-gradXY20[dy][dx] * wDz) + (gradXY02[dy][dx] * wz);
                  }
               }
            }
         }  // loop qz

         osc += D1Dx * D1Dy * D1Dz;
      }

      // z component
      {
         const int D1Dz = D1D - 1;
         const int D1Dy = D1D;
         const int D1Dx = D1D;

         for (int qx = 0; qx < Q1D; ++qx)
         {
            double gradYZ01[MAX_D1D][MAX_D1D];
            double gradYZ10[MAX_D1D][MAX_D1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  gradYZ01[dz][dy] = 0.0;
                  gradYZ10[dz][dy] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massZ[MAX_D1D][2];
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     massZ[dz][n] = 0.0;
                  }
               }
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     const double wz = Bot(dz,qz);

                     massZ[dz][0] += wz * mass[qz][qy][qx][0];
                     massZ[dz][1] += wz * mass[qz][qy][qx][1];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = Bct(dy,qy);
                  const double wDy = Gct(dy,qy);

                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     gradYZ01[dz][dy] += wy * massZ[dz][1];
                     gradYZ10[dz][dy] += wDy * massZ[dz][0];
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const double wx = Bct(dx,qx);
               const double wDx = Gct(dx,qx);

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     // (u_2)_{x_1} * (op * curl)_0 - (u_2)_{x_0} * (op * curl)_1
                     Y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                       e) += (gradYZ10[dz][dy] * wx) - (gradYZ01[dz][dy] * wDx);
                  }
               }
            }
         }  // loop qx
      }
   }); // end of element loop
}

void MixedVectorCurlIntegrator::AssemblePA(const FiniteElementSpace &trial_fes,
                                           const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with vector test and trial spaces.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetFE(0);
   const FiniteElement *test_fel = test_fes.GetFE(0);

   const VectorTensorFiniteElement *trial_el =
      dynamic_cast<const VectorTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetElementTransformation(0));
   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 3, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 3, "");

   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder(), "");

   ne = trial_fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &trial_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &trial_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   mapsCtest = &test_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsOtest = &test_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;
   dofs1Dtest = mapsCtest->ndof;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   testType = test_el->GetDerivType();
   trialType = trial_el->GetDerivType();

   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   coeffDim = (DQ ? 3 : 1);

   const bool curlSpaces = (testType == mfem::FiniteElement::CURL &&
                            trialType == mfem::FiniteElement::CURL);

   const int ndata = curlSpaces ? (coeffDim == 1 ? 1 : 9) : symmDims;
   pa_data.SetSize(ndata * nq * ne, Device::GetMemoryType());

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(qs, CoefficientStorage::FULL);
   if (Q) { coeff.Project(*Q); }
   else if (DQ) { coeff.Project(*DQ); }
   else { coeff.SetConstant(1.0); }

   if (testType == mfem::FiniteElement::CURL &&
       trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      if (coeffDim == 1)
      {
         internal::PAHcurlL2Setup(nq, coeffDim, ne, ir->GetWeights(), coeff, pa_data);
      }
      else
      {
         internal::PAHcurlHdivMassSetup3D(quad1D, coeffDim, ne, false, ir->GetWeights(),
                                          geom->J, coeff, pa_data);
      }
   }
   else if (testType == mfem::FiniteElement::DIV &&
            trialType == mfem::FiniteElement::CURL && dim == 3 &&
            test_fel->GetOrder() == trial_fel->GetOrder())
   {
      internal::PACurlCurlSetup3D(quad1D, coeffDim, ne, ir->GetWeights(), geom->J,
                                  coeff,
                                  pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

void MixedVectorCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (testType == mfem::FiniteElement::CURL &&
       trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      const int ndata = coeffDim == 1 ? 1 : 9;

      if (Device::Allows(Backend::DEVICE_MASK))
      {
         const int ID = (dofs1D << 4) | quad1D;
         switch (ID)
         {
            case 0x23: return internal::SmemPAHcurlL2Apply3D<2,3>(dofs1D, quad1D, ndata, ne,
                                                                     mapsO->B, mapsC->B,
                                                                     mapsC->G, pa_data, x, y);
            case 0x34: return internal::SmemPAHcurlL2Apply3D<3,4>(dofs1D, quad1D, ndata, ne,
                                                                     mapsO->B, mapsC->B,
                                                                     mapsC->G, pa_data, x, y);
            case 0x45: return internal::SmemPAHcurlL2Apply3D<4,5>(dofs1D, quad1D, ndata, ne,
                                                                     mapsO->B, mapsC->B,
                                                                     mapsC->G, pa_data, x, y);
            case 0x56: return internal::SmemPAHcurlL2Apply3D<5,6>(dofs1D, quad1D, ndata, ne,
                                                                     mapsO->B, mapsC->B,
                                                                     mapsC->G, pa_data, x, y);
            default: return internal::SmemPAHcurlL2Apply3D(dofs1D, quad1D, ndata, ne,
                                                              mapsO->B, mapsC->B, mapsC->G,
                                                              pa_data, x, y);
         }
      }
      else
         internal::PAHcurlL2Apply3D(dofs1D, quad1D, ndata, ne, mapsO->B, mapsC->B,
                                    mapsO->Bt, mapsC->Bt, mapsC->G, pa_data, x, y);
   }
   else if (testType == mfem::FiniteElement::DIV &&
            trialType == mfem::FiniteElement::CURL && dim == 3)
      PAHcurlHdivApply3D(dofs1D, dofs1Dtest, quad1D, ne, mapsO->B,
                         mapsC->B, mapsOtest->Bt, mapsCtest->Bt, mapsC->G,
                         pa_data, x, y);
   else
   {
      MFEM_ABORT("Unsupported dimension or space!");
   }
}

void MixedVectorCurlIntegrator::AddMultTransposePA(const Vector &x,
                                                   Vector &y) const
{
   if (testType == mfem::FiniteElement::DIV &&
       trialType == mfem::FiniteElement::CURL && dim == 3)
      PAHcurlHdivApply3DTranspose(dofs1D, dofs1Dtest, quad1D, ne, mapsO->B,
                                  mapsC->B, mapsOtest->Bt, mapsCtest->Bt,
                                  mapsC->Gt, pa_data, x, y);
   else
   {
      MFEM_ABORT("Unsupported dimension or space!");
   }
}

void MixedVectorWeakCurlIntegrator::AssemblePA(const FiniteElementSpace
                                               &trial_fes,
                                               const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with vector test and trial spaces.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetFE(0);
   const FiniteElement *test_fel = test_fes.GetFE(0);

   const VectorTensorFiniteElement *trial_el =
      dynamic_cast<const VectorTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetElementTransformation(0));
   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 3, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 3, "");

   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder(), "");

   ne = trial_fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &test_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &test_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   testType = test_el->GetDerivType();
   trialType = trial_el->GetDerivType();

   const bool curlSpaces = (testType == mfem::FiniteElement::CURL &&
                            trialType == mfem::FiniteElement::CURL);

   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6

   coeffDim = DQ ? 3 : 1;
   const int ndata = curlSpaces ? (DQ ? 9 : 1) : symmDims;

   pa_data.SetSize(ndata * nq * ne, Device::GetMemoryType());

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(qs, CoefficientStorage::FULL);
   if (Q) { coeff.Project(*Q); }
   else if (DQ) { coeff.Project(*DQ); }
   else { coeff.SetConstant(1.0); }

   if (trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      if (coeffDim == 1)
      {
         internal::PAHcurlL2Setup(nq, coeffDim, ne, ir->GetWeights(), coeff, pa_data);
      }
      else
      {
         internal::PAHcurlHdivMassSetup3D(quad1D, coeffDim, ne, false, ir->GetWeights(),
                                          geom->J, coeff, pa_data);
      }
   }
   else if (trialType == mfem::FiniteElement::DIV && dim == 3 &&
            test_el->GetOrder() == trial_el->GetOrder())
   {
      internal::PACurlCurlSetup3D(quad1D, coeffDim, ne, ir->GetWeights(), geom->J,
                                  coeff,
                                  pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

void MixedVectorWeakCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (testType == mfem::FiniteElement::CURL &&
       trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      const int ndata = coeffDim == 1 ? 1 : 9;
      if (Device::Allows(Backend::DEVICE_MASK))
      {
         const int ID = (dofs1D << 4) | quad1D;
         switch (ID)
         {
            case 0x23: return internal::SmemPAHcurlL2Apply3DTranspose<2,3>(dofs1D, quad1D,
                                                                              ndata,
                                                                              ne, mapsO->B, mapsC->B,
                                                                              mapsC->G, pa_data, x, y);
            case 0x34: return internal::SmemPAHcurlL2Apply3DTranspose<3,4>(dofs1D, quad1D,
                                                                              ndata,
                                                                              ne, mapsO->B, mapsC->B,
                                                                              mapsC->G, pa_data, x, y);
            case 0x45: return internal::SmemPAHcurlL2Apply3DTranspose<4,5>(dofs1D, quad1D,
                                                                              ndata,
                                                                              ne, mapsO->B, mapsC->B,
                                                                              mapsC->G, pa_data, x, y);
            case 0x56: return internal::SmemPAHcurlL2Apply3DTranspose<5,6>(dofs1D, quad1D,
                                                                              ndata,
                                                                              ne, mapsO->B, mapsC->B,
                                                                              mapsC->G, pa_data, x, y);
            default: return internal::SmemPAHcurlL2Apply3DTranspose(dofs1D, quad1D, ndata,
                                                                       ne,
                                                                       mapsO->B, mapsC->B,
                                                                       mapsC->G, pa_data, x, y);
         }
      }
      else
         internal::PAHcurlL2Apply3DTranspose(dofs1D, quad1D, ndata, ne, mapsO->B,
                                             mapsC->B, mapsO->Bt, mapsC->Bt, mapsC->Gt, pa_data, x, y);
   }
   else if (testType == mfem::FiniteElement::CURL &&
            trialType == mfem::FiniteElement::DIV && dim == 3)
   {
      PAHcurlHdivApply3DTranspose(dofs1D, dofs1D, quad1D, ne, mapsO->B,
                                  mapsC->B, mapsO->Bt, mapsC->Bt,
                                  mapsC->Gt, pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension or space!");
   }
}

void MixedVectorWeakCurlIntegrator::AddMultTransposePA(const Vector &x,
                                                       Vector &y) const
{
   if (testType == mfem::FiniteElement::CURL &&
       trialType == mfem::FiniteElement::DIV && dim == 3)
   {
      PAHcurlHdivApply3D(dofs1D, dofs1D, quad1D, ne, mapsO->B,
                         mapsC->B, mapsO->Bt, mapsC->Bt, mapsC->G,
                         pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension or space!");
   }
}

} // namespace mfem
