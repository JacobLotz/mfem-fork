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
//
//   -----------------------------------------------------------------------
//       Stix2D_DH Miniapp: Cold Plasma Electromagnetic Simulation Code
//   -----------------------------------------------------------------------
//
//   Assumes that all sources and boundary conditions oscillate with the same
//   frequency although not necessarily in phase with one another.  This
//   assumption implies that we can factor out the time dependence which we
//   take to be of the form exp(-i omega t).  With these assumptions we can
//   write the Maxwell equations for D and H in the form:
//
//   -i omega D    = Curl H - J
//    i omega mu H = Curl epsilon^{-1} D
//
//   Which combine to yield:
//
//   Curl epsilon^{-1} Curl H - omega^2 mu H = Curl epsilon^{-1} J
//
//   In a cold plasma the dielectric tensor, epsilon, is complex-valued and
//   anisotropic.  The anisotropy aligns with the external magnetic field and
//   the values depend on the properties of the plasma including the masses and
//   charges of its constituent ion species.
//
//   For a magnetic field aligned with the z-axis the dielectric tensor has
//   the form:
//              | S  -iD 0 |
//    epsilon = |iD   S  0 |
//              | 0   0  P |
//
//   Where:
//      S = 1 - Sum_species omega_p^2 / (omega^2 - omega_c^2)
//      D = Sum_species omega_p^2 omega_c / (omega^2 - omega_c^2)
//      P = 1 - Sum_species omega_p^2 / omega^2
//
//   and:
//      omega_p is the plasma frequency
//      omega_c is the cyclotron frequency
//      omega   is the driving frequency
//
//   The plasma and cyclotron frequencies depend primarily on the properties
//   of the ion species.  We also include a complex-valued mass correction
//   which depends on the plasma temperature.
//
//   We discretize this equation with H(Curl) a.k.a Nedelec basis
//   functions.  The curl curl operator must be handled with
//   integration by parts which yields a surface integral:
//
//   (W, Curl epsilon^{-1} Curl H) = (Curl W, epsilon^{-1} Curl H)
//               + (W, n x (epsilon^{-1} Curl H))_{\Gamma}
//
//   or
//
//   (W, Curl epsilon^{-1} Curl H) = (Curl W, epsilon^{-1} Curl H)
//               - i omega (W, n x E)_{\Gamma}
//
//   Assuming J = 0 on the boundary
//
//   The non-linear sheath boundary condition can be used to set the
//   components of E that are tangent to the boundary. The governing
//   equations are:
//
//      E_t = - Grad Phi_{RF} (where Phi_{RF} is the sheath potential)
//      Phi_{RF} = i omega z_{sh} D_n
//
//   Where D_n is the normal component of D = epsilon E and z_{sh}
//   is the sheath impedance. The impedance z_{sh} is a function of the
//   plasma density, plasma temperature, ion charges, ion masses,
//   magnetic field strength, and the sheath potential itself. Clearly the
//   dependence on the potential is the source of the non-linearity.
//
//   The sheath boundary condition can be easily incorporated into the
//   weak form of the curl-curl operator:
//
//   (W, Curl epsilon^{-1} Curl H) = (Curl W, epsilon^{-1} Curl H)
//               + i omega (W, n x Grad Phi_{RF})_{\Gamma}
//
//   To compute Phi_{RF} we augment the Maxwell equations with the
//   relation between D_n and Phi_{RF}:
//
//      - i omega z_{sh} D_n + Phi_{RF} = 0
//
//   or
//
//      z_{sh} Curl H + Phi_{RF} = 0
//
// (By default the sources and fields are all zero)
//
// Compile with: make stix2d_dh
//
// Sample runs:
//   ./stix2d -rod '0 0 1 0 0 0.1' -o 3 -s 1 -rs 0 -maxit 1 -f 1e6
//
// Sample runs with partial assembly:
//   ./stix2d -rod '0 0 1 0 0 0.1' -o 3 -s 1 -rs 0 -maxit 1 -f 1e6 -pa
//
// Device sample runs:
//   ./stix2d -rod '0 0 1 0 0 0.1' -o 3 -s 1 -rs 0 -maxit 1 -f 1e6 -pa -d cuda
//
// Parallel sample runs:
//   mpirun -np 4 ./stix2d -rod '0 0 1 0 0 0.1' -dbcs '1' -w Z -o 3 -s 1 -rs 0 -maxit 1 -f 1e6
//

#include "cold_plasma_dielectric_coefs.hpp"
#include "cold_plasma_dielectric_dh_solver.hpp"
#include "../common/mesh_extras.hpp"
#include "plasma.hpp"
#include "g_eqdsk_data.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <complex>

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace mfem::plasma;

class MeshTransformCoefficient : public VectorCoefficient
{
private:
   double hphi_rad_;

   mutable Vector uvw_;

public:
   MeshTransformCoefficient(double hphi_deg)
      : VectorCoefficient(3), hphi_rad_(hphi_deg * M_PI / 180.0),
        uvw_(3)
   {}

   void Eval(Vector &xyz, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      T.Transform(ip, uvw_);

      const double r   = uvw_[0];
      const double phi = hphi_rad_ * uvw_[2];
      const double z   = uvw_[1];

      xyz[0] = r * cos(phi);
      xyz[1] = r * sin(phi);
      xyz[2] = z;
   }
};

class VectorConstantCylCoefficient : public VectorCoefficient
{
private:
   bool cyl;
   Vector vec;
   mutable Vector x;
public:
   /** The constant vector v is defined in either cartesian or cylindrical
       coordinates.

       If cyl == true
          v = (v_r, v_phi, v_z)
       Else
          v = (v_x, v_y, v_z)
   */
   VectorConstantCylCoefficient(bool cyl_, const Vector &v)
      : VectorCoefficient(3), cyl(cyl_), vec(v), x(3) {}
   using VectorCoefficient::Eval;

   ///  Evaluate the vector coefficient at @a ip.
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      if (cyl)
      {
         V.SetSize(3);

         T.Transform(ip, x);

         double r = sqrt(x[0] * x[0] + x[1] * x[1]);
         double cosphi = x[0] / r;
         double sinphi = x[1] / r;

         V[0] = vec[0] * cosphi - vec[1] * sinphi;
         V[1] = vec[0] * sinphi + vec[1] * cosphi;
         V[2] = vec[2];
      }
      else
      {
         V = vec;
      }
   }

   /// Return a reference to the constant vector in this class.
   const Vector& GetVec() { return vec; }
};

// Admittance for Absorbing Boundary Condition
Coefficient * SetupImpedanceCoefficient(const Mesh & mesh,
                                        const Array<int> & abcs);

// Storage for user-supplied, real-valued impedance
static Vector pw_eta_(0);      // Piecewise impedance values
static Vector pw_bdr_eta_(0);  // Piecewise impedance values (by bdr attr)

// Storage for user-supplied, complex-valued impedance
//static Vector pw_eta_re_(0);      // Piecewise real impedance
//static Vector pw_eta_inv_re_(0);  // Piecewise inverse real impedance
//static Vector pw_eta_im_(0);      // Piecewise imaginary impedance
//static Vector pw_eta_inv_im_(0);  // Piecewise inverse imaginary impedance

// Current Density Function
static bool j_cyl_ = false;
static Vector rod_params_
(0); // Amplitude of x, y, z current source, position in 2D, and radius
static Vector slab_params_
(0); // Amplitude of x, y, z current source, position in 2D, and size in 2D
static Vector curve_params_
(0); // Text here
static int slab_profile_;

void rod_current_source_r(const Vector &x, Vector &j);
void rod_current_source_i(const Vector &x, Vector &j);
void slab_current_source_r(const Vector &x, Vector &j);
void slab_current_source_i(const Vector &x, Vector &j);
void curve_current_source_r(const Vector &x, Vector &j);
void curve_current_source_i(const Vector &x, Vector &j);
void j_src_r(const Vector &x, Vector &j)
{
   if (rod_params_.Size() > 0)
   {
      rod_current_source_r(x, j);
   }
   else if (slab_params_.Size() > 0)
   {
      slab_current_source_r(x, j);
   }
   else if (curve_params_.Size() > 0)
   {
      curve_current_source_r(x, j);
   }
}
void j_src_i(const Vector &x, Vector &j)
{
   if (rod_params_.Size() > 0)
   {
      rod_current_source_i(x, j);
   }
   else if (slab_params_.Size() > 0)
   {
      slab_current_source_i(x, j);
   }
   else if (curve_params_.Size() > 0)
   {
      curve_current_source_i(x, j);
   }
}

// Electric Field Boundary Condition: The following function returns zero but
// any function could be used.
void e_bc_r(const Vector &x, Vector &E);
void e_bc_i(const Vector &x, Vector &E);

class ColdPlasmaPlaneWaveH: public VectorCoefficient
{
public:
   ColdPlasmaPlaneWaveH(char type,
                        double omega,
                        const Vector & B,
                        const Vector & number,
                        const Vector & charge,
                        const Vector & mass,
                        const Vector & temp,
                        int nuprof,
                        bool realPart);

   void SetCurrentSlab(double Jy, double xJ, double delta, double Lx)
   { Jy_ = Jy; xJ_ = xJ; dx_ = delta, Lx_ = Lx; }

   void SetPhaseShift(const Vector & beta)
   { beta_r_ = beta; beta_i_ = 0.0; }
   void SetPhaseShift(const Vector & beta_r,
                      const Vector & beta_i)
   { beta_r_ = beta_r; beta_i_ = beta_i; }

   void GetWaveVector(Vector & k_r, Vector & k_i) const
   { k_r = k_r_; k_i = k_i_; }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);

private:
   char type_;
   bool realPart_;
   int nuprof_;
   double omega_;
   double Bmag_;
   double Jy_;
   double xJ_;
   double dx_;
   double Lx_;
   complex<double> kappa_;
   Vector b_;   // Normalized vector in direction of B
   Vector bc_;  // Normalized vector perpendicular to b_, (by-bz,bz-bx,bx-by)
   Vector bcc_; // Normalized vector perpendicular to b_ and bc_
   Vector h_r_;
   Vector h_i_;
   Vector k_r_;
   Vector k_i_;
   Vector beta_r_;
   Vector beta_i_;

   // const Vector & B_;
   const Vector & numbers_;
   const Vector & charges_;
   const Vector & masses_;
   const Vector & temps_;

   complex<double> S_;
   complex<double> D_;
   complex<double> P_;
};

class ColdPlasmaPlaneWaveE: public VectorCoefficient
{
public:
   ColdPlasmaPlaneWaveE(char type,
                        double omega,
                        const Vector & B,
                        const Vector & number,
                        const Vector & charge,
                        const Vector & mass,
                        const Vector & temp,
                        int nuprof,
                        bool realPart);

   void SetCurrentSlab(double Jy, double xJ, double delta, double Lx)
   { Jy_ = Jy; xJ_ = xJ; dx_ = delta, Lx_ = Lx; }

   void SetPhaseShift(const Vector & beta)
   { beta_r_ = beta; beta_i_ = 0.0; }
   void SetPhaseShift(const Vector & beta_r,
                      const Vector & beta_i)
   { beta_r_ = beta_r; beta_i_ = beta_i; }

   void GetWaveVector(Vector & k_r, Vector & k_i) const
   { k_r = k_r_; k_i = k_i_; }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);

private:
   char type_;
   bool realPart_;
   int nuprof_;
   double omega_;
   double Bmag_;
   double Jy_;
   double xJ_;
   double dx_;
   double Lx_;
   complex<double> kappa_;
   Vector b_;   // Normalized vector in direction of B
   Vector bc_;  // Normalized vector perpendicular to b_, (by-bz,bz-bx,bx-by)
   Vector bcc_; // Normalized vector perpendicular to b_ and bc_
   Vector e_r_;
   Vector e_i_;
   Vector k_r_;
   Vector k_i_;
   Vector beta_r_;
   Vector beta_i_;

   // const Vector & B_;
   const Vector & numbers_;
   const Vector & charges_;
   const Vector & masses_;
   const Vector & temps_;

   complex<double> S_;
   complex<double> D_;
   complex<double> P_;
};

/** Vector coefficient to compute H field wrapping around
    current-carrying rectangular antenna straps.
    Any number of straps can be supported but the straps are assumed
    to be rectangular and aligned with the x and y axes..
    Each strap requires 6 parameters:
      x0: x-position of the first corner of the strap
      y0: y-position of the first corner of the strap
      x1: x-position of the second corner of the strap
      y1: y-position of the second corner of the strap
      x2: "
      y2: "
      x3: "
      y3: "
      Re(I): Real part of the current in the strap
      Im(I): Imaginary part of the current in the strap
    The parameters should be grouped by strap so that the ten params
    for strap 1 are first then the six for strap 2, etc..
*/
class MultiStrapAntennaH : public VectorCoefficient
{
private:
   bool real_part_;
   int num_straps_;
   double tol_;
   Vector params_;
   Vector current_vals_;
   Vector x_;

public:
   MultiStrapAntennaH(int n, const Vector &params, const Vector &current_vals,
                      bool real_part, double tol = 1e-6)
      : VectorCoefficient(3), real_part_(real_part), num_straps_(n),
        tol_(tol), params_(params), current_vals_(current_vals), x_(2)
   {
      MFEM_ASSERT(params.Size() == 8 * n,
                  "Incorrect number of parameters provided to "
                  "MultiStrapAntennaH");
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      V.SetSize(3); V = 0.0;
      T.Transform(ip, x_);
      for (int i=0; i<num_straps_; i++)
      {
         double x0  = params_[8 * i + 0];
         double y0  = params_[8 * i + 1];
         double x1  = params_[8 * i + 2];
         double y1  = params_[8 * i + 3];
         double x2  = params_[8 * i + 4];
         double y2  = params_[8 * i + 5];
         double x3  = params_[8 * i + 6];
         double y3  = params_[8 * i + 7];

         double ReI = current_vals_[2 * i + 0];
         double ImI = current_vals_[2 * i + 1];

         double d01 = sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2));
         double d12 = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
         double d23 = sqrt(pow(x3 - x2, 2) + pow(y3 - y2, 2));
         double d30 = sqrt(pow(x0 - x3, 2) + pow(y0 - y3, 2));

         double   H = (real_part_ ? ReI : ImI) / (d01 + d12 + d23 + d30);

         // *** The following will break on any vertical sides ***
         // Bottom of Antenna Strap:
         double s1 = (y1-y0)/(x1-x0);
         double b1 = y1 - s1*x1;
         // Right of Antenna Strap:
         double s2 = (y2-y1)/(x2-x1);
         double b2 = y2 - s2*x2;
         // Top of Antenna Strap:
         double s3 = (y3-y2)/(x3-x2);
         double b3 = y3 - s3*x3;
         // Left of Antenna Strap:
         double s4 = (y3-y0)/(x3-x0);
         double b4 = y3 - s4*x3;

         if (fabs(x_[1] - (s1*x_[0]+b1)) <= tol_
             && x_[0] >= x0 && x_[0] <= x1)
         {
            V[0] = (x1 - x0) * H / d01;
            V[1] = (y1 - y0) * H / d01;
            break;
         }
         else if (fabs(x_[1] - (s2*x_[0]+b2)) <= tol_
                  && x_[1] >= y1 && x_[1] <= y2)
         {
            V[0] = (x2 - x1) * H / d12;
            V[1] = (y2 - y1) * H / d12;
            break;
         }
         else if (fabs(x_[1] - (s3*x_[0]+b3)) <= tol_
                  && x_[0] >= x3 && x_[0] <= x2)
         {
            V[0] = (x3 - x2) * H / d23;
            V[1] = (y3 - y2) * H / d23;
            break;
         }
         else if (fabs(x_[1] - (s4*x_[0]+b4)) <= tol_
                  && x_[1] >= y0 && x_[1] <= y3)
         {
            V[0] = (x0 - x3) * H / d30;
            V[1] = (y0 - y3) * H / d30;
            break;
         }
      }
   }
};

void AdaptInitialMesh(MPI_Session &mpi,
                      ParMesh &pmesh,
                      ParFiniteElementSpace &err_fespace,
                      ParFiniteElementSpace & H1FESpace,
                      ParFiniteElementSpace & HCurlFESpace,
                      ParFiniteElementSpace & HDivFESpace,
                      ParFiniteElementSpace & L2FESpace,
                      VectorCoefficient & BCoef,
                      Coefficient & rhoCoef,
                      Coefficient & TCoef,
                      Coefficient & nueCoef,
                      Coefficient & nuiCoef,
                      int & size_h1,
                      int & size_l2,
                      Array<int> & density_offsets,
                      Array<int> & temperature_offsets,
                      BlockVector & density,
                      BlockVector & temperature,
                      ParGridFunction & BField,
                      ParGridFunction & density_gf,
                      ParGridFunction & temperature_gf,
                      ParGridFunction & nue_gf,
                      ParGridFunction & nui_gf,
                      Coefficient &ReCoef,
                      Coefficient &ImCoef,
                      int p, double tol, int max_its, int max_dofs,
                      bool visualization);

void Update(ParFiniteElementSpace & H1FESpace,
            ParFiniteElementSpace & HCurlFESpace,
            ParFiniteElementSpace & HDivFESpace,
            ParFiniteElementSpace & L2FESpace,
            VectorCoefficient & BCoef,
            Coefficient & rhoCoef,
            Coefficient & TCoef,
            Coefficient & nueCoef,
            Coefficient & nuiCoef,
            int & size_h1,
            int & size_l2,
            Array<int> & density_offsets,
            Array<int> & temperature_offsets,
            BlockVector & density,
            BlockVector & temperature,
            ParGridFunction & BField,
            ParGridFunction & density_gf,
            ParGridFunction & temperature_gf,
            ParGridFunction & nue_gf,
            ParGridFunction & nui_gf);

//static double freq_ = 1.0e9;

// Mesh Size
//static Vector mesh_dim_(0); // x, y, z dimensions of mesh

// Prints the program's logo to the given output stream
void display_banner(ostream & os);

void record_cmd_line(int argc, char *argv[]);

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);
   if (!mpi.Root()) { mfem::out.Disable(); mfem::err.Disable(); }

   display_banner(mfem::out);

   if (mpi.Root()) { record_cmd_line(argc, argv); }

   int logging = 1;

   // Parse command-line options.
   const char *mesh_file = "ellipse_origin_h0pt0625_o3.mesh";
   int mesh_order = -1;
   double init_amr_tol = 1e-5;
   int init_amr_max_its = 10;
   int init_amr_max_dofs = 100000;
   int ser_ref_levels = 0;
   int order = 1;
   int maxit = 100;
   int sol = 2;
   int prec = 1;
   // int nspecies = 2;
   bool amr_s = true;
   bool herm_conv = false;
   bool vis_u = false;
   bool visualization = true;
   bool visit = true;

   double freq = 1.0e6;
   const char * wave_type = " ";

   Vector BVec(3);
   BVec = 0.0;

   bool phase_shift = false;
   Vector kVec;
   Vector kReVec;
   Vector kImVec;

   double hz = -1.0; // Extruded mesh thickness in meters
   double hphi = -1.0; // Cylindrically extruded mesh thickness in degrees

   Vector numbers;
   Vector charges;
   Vector masses;
   Vector temps;
   Vector minority;
   Vector temp_charges;
   double nue = 0;
   double nui = 0;

   PlasmaProfile::Type dpt = PlasmaProfile::CONSTANT;
   PlasmaProfile::Type tpt = PlasmaProfile::CONSTANT;
   PlasmaProfile::Type nept = PlasmaProfile::CONSTANT;
   PlasmaProfile::Type nipt = PlasmaProfile::CONSTANT;
   BFieldProfile::Type bpt = BFieldProfile::CONSTANT;
   Vector dpp;
   Vector tpp;
   Vector bpp;
   Vector nepp;
   Vector nipp;
   int nuprof = 0;

   Array<int> abcs; // Absorbing BC attributes
   Array<int> sbca; // Sheath BC attributes
   Array<int> peca; // Perfect Electric Conductor BC attributes
   Array<int> dbca1; // Dirichlet BC attributes
   Array<int> dbca2; // Dirichlet BC attributes
   Array<int> dbcas; // Dirichlet BC attributes for multi-strap antenna source
   Array<int> dbcaw; // Dirichlet BC attributes for plane wave source
   Array<int> nbca1; // Neumann BC attributes
   Array<int> nbca2; // Neumann BC attributes
   Array<int> nbcaw; // Neumann BC attributes for plane wave source
   Vector dbcv1; // Dirichlet BC values
   Vector dbcv2; // Dirichlet BC values
   Vector nbcv1; // Neumann BC values
   Vector nbcv2; // Neumann BC values

   int msa_n = 0;
   Vector msa_p(0);
   Vector msa_c(0);

   int num_elements = 10;

   SolverOptions solOpts;
   solOpts.maxIter = 1000;
   solOpts.kDim = 50;
   solOpts.printLvl = 1;
   solOpts.relTol = 1e-4;
   solOpts.euLvl = 1;

   bool logo = false;
   bool cyl = false;
   bool per_y = false;
   bool check_eps_inv = false;
   bool pa = false;
   const char *device_config = "cpu";
   const char *eqdsk_file = "";

   OptionsParser args(argc, argv);
   args.AddOption(&logo, "-logo", "--print-logo", "-no-logo",
                  "--no-print-logo", "Print logo and exit.");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_order, "-mo", "--mesh-order",
                  "Geometry order for cylindrically symmetric mesh.");
   args.AddOption(&cyl, "-cyl", "--cylindrical-coords", "-cart",
                  "--cartesian-coords",
                  "Cartesian (x, y, z) coordinates or "
                  "Cylindrical (z, rho, phi).");
   args.AddOption(&per_y, "-per-y", "--periodic-in-y", "-no-per-y",
                  "--not-periodic-in-y",
                  "The input mesh is periodic in the y-direction.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&amr_s, "-amr-s", "--init-amr-s", "-no-amr-s",
                  "--no-init-amr-s",
                  "Initial AMR to capture Stix S coefficient.");
   args.AddOption(&init_amr_tol, "-iatol", "--init-amr-tol",
                  "Initial AMR tolerance.");
   args.AddOption(&init_amr_max_its, "-iamit", "--init-amr-max-its",
                  "Initial AMR Maximum Number of Iterations.");
   args.AddOption(&init_amr_max_dofs, "-iamdof", "--init-amr-max-dofs",
                  "Initial AMR Maximum Number of DoFs.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   // args.AddOption(&nspecies, "-ns", "--num-species",
   //               "Number of ion species.");
   args.AddOption(&freq, "-f", "--frequency",
                  "Frequency in Hertz (of course...)");
   args.AddOption(&hz, "-mh", "--mesh-height",
                  "Thickness of extruded mesh in meters.");
   args.AddOption(&hphi, "-mhc", "--mesh-height-cyl",
                  "Thickness of cylindrically extruded mesh in degrees.");
   args.AddOption((int*)&dpt, "-dp", "--density-profile",
                  "Density Profile Type (for ions): \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyprebolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&dpp, "-dpp", "--density-profile-params",
                  "Density Profile Parameters:\n"
                  "   CONSTANT: density value\n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption((int*)&bpt, "-bp", "--Bfield-profile",
                  "BField Profile Type: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyprebolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&bpp, "-bpp", "--Bfield-profile-params",
                  "BField Profile Parameters:\n"
                  "  B_P: value at -1, value at 1, "
                  "radius in x, radius in y, location of center, Bz, placeholder.");
   args.AddOption((int*)&tpt, "-tp", "--temperature-profile",
                  "Temperature Profile Type: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyperbolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&tpp, "-tpp", "--temperature-profile-params",
                  "Temperature Profile Parameters: \n"
                  "   CONSTANT: temperature value \n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption((int*)&nept, "-nep", "--electron-collision-profile",
                  "Electron Collisions Profile Type: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyperbolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&nepp, "-nepp", "--electron-collisions-profile-params",
                  "Electron Collisions Profile Parameters: \n"
                  "   CONSTANT: temperature value \n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption((int*)&nipt, "-nip", "--ion-collision-profile",
                  "Ion Collisions Profile Type: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyperbolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&nipp, "-nipp", "--ion-collisions-profile-params",
                  "Ion Collisions Profile Parameters: \n"
                  "   CONSTANT: temperature value \n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption(&nuprof, "-nuprof", "--collisional-profile",
                  "Temperature Profile Type: \n"
                  "0 - Standard e-i Collision Freq, 1 - Custom Freq.");
   args.AddOption(&wave_type, "-w", "--wave-type",
                  "Wave type: 'R' - Right Circularly Polarized, "
                  "'L' - Left Circularly Polarized, "
                  "'O' - Ordinary, 'X' - Extraordinary, "
                  "'J' - Current Slab (in conjunction with -slab), "
                  "'Z' - Zero");
   // args.AddOption(&BVec, "-B", "--magnetic-flux",
   //                "Background magnetic flux vector");
   args.AddOption(&kVec, "-k-vec", "--phase-vector",
                  "Phase shift vector across periodic directions."
                  " For complex phase shifts input 3 real phase shifts "
                  "followed by 3 imaginary phase shifts");
   args.AddOption(&msa_n, "-ns", "--num-straps","");
   args.AddOption(&msa_p, "-sp", "--strap-params","");
   args.AddOption(&msa_c, "-sc", "--strap-currents","");
   //args.AddOption(&numbers, "-num", "--number-densites",
   //               "Number densities of the various species");
   args.AddOption(&charges, "-q", "--charges",
                  "Charges of the various species "
                  "(in units of electron charge)");
   //args.AddOption(&masses, "-mass", "--masses",
   //               "Masses of the various species (in amu)");
   args.AddOption(&minority, "-min", "--minority",
                  "Minority Ion Species: charge, mass (amu), concentration."
                  " Concentration being: n_min/n_i");
   args.AddOption(&prec, "-pc", "--precond",
                  "Preconditioner: 1 - Diagonal Scaling, 2 - ParaSails, "
                  "3 - Euclid, 4 - AMS");
   args.AddOption(&sol, "-s", "--solver",
                  "Solver: 1 - GMRES, 2 - FGMRES, 3 - MINRES"
#ifdef MFEM_USE_SUPERLU
                  ", 4 - SuperLU"
#endif
#ifdef MFEM_USE_STRUMPACK
                  ", 5 - STRUMPACK"
#endif
                 );
   args.AddOption(&solOpts.maxIter, "-sol-it", "--solver-iterations",
                  "Maximum number of solver iterations.");
   args.AddOption(&solOpts.kDim, "-sol-k-dim", "--solver-krylov-dimension",
                  "Krylov space dimension for GMRES and FGMRES.");
   args.AddOption(&solOpts.relTol, "-sol-tol", "--solver-tolerance",
                  "Relative tolerance for GMRES or FGMRES.");
   args.AddOption(&solOpts.printLvl, "-sol-prnt-lvl", "--solver-print-level",
                  "Logging level for solvers.");
   args.AddOption(&solOpts.euLvl, "-eu-lvl", "--euclid-level",
                  "Euclid factorization level for ILU(k).");
   args.AddOption(&pw_eta_, "-pwz", "--piecewise-eta",
                  "Piecewise values of Impedance (one value per abc surface)");
   /*
   args.AddOption(&pw_eta_re_, "-pwz-r", "--piecewise-eta-r",
                  "Piecewise values of Real part of Complex Impedance "
                  "(one value per abc surface)");
   args.AddOption(&pw_eta_im_, "-pwz-i", "--piecewise-eta-i",
                  "Piecewise values of Imaginary part of Complex Impedance "
                  "(one value per abc surface)");
   */
   args.AddOption(&rod_params_, "-rod", "--rod_params",
                  "3D Vector Amplitude (Real x,y,z, Imag x,y,z), "
                  "2D Position, Radius");
   args.AddOption(&slab_params_, "-slab", "--slab_params",
                  "3D Vector Amplitude (Real x,y,z, Imag x,y,z), "
                  "2D Position, 2D Size");
   args.AddOption(&curve_params_, "-curve", "--curve_params",
                  "2D Vector Amplitude (Real theta,phi, theta,phi)");
   args.AddOption(&slab_profile_, "-slab-prof", "--slab_profile",
                  "0 (Constant) or 1 (Sin Function)");
   args.AddOption(&abcs, "-abcs", "--absorbing-bc-surf",
                  "Absorbing Boundary Condition Surfaces");
   args.AddOption(&sbca, "-sbcs", "--sheath-bc-surf",
                  "Sheath Boundary Condition Surfaces");
   args.AddOption(&peca, "-pecs", "--pec-bc-surf",
                  "Perfect Electrical Conductor Boundary Condition Surfaces");
   args.AddOption(&dbcaw, "-dbcs-pw", "--dirichlet-bc-pw-surf",
                  "Dirichlet Boundary Condition Surfaces Using Plane Wave");
   args.AddOption(&dbca1, "-dbcs1", "--dirichlet-bc-1-surf",
                  "Dirichlet Boundary Condition Surfaces Using Value 1");
   args.AddOption(&dbca2, "-dbcs2", "--dirichlet-bc-2-surf",
                  "Dirichlet Boundary Condition Surfaces Using Value 2");
   args.AddOption(&dbcv1, "-dbcv1", "--dirichlet-bc-1-vals",
                  "Dirichlet Boundary Condition Value 1 (v_x v_y v_z)"
                  " or (Re(v_x) Re(v_y) Re(v_z) Im(v_x) Im(v_y) Im(v_z))");
   args.AddOption(&dbcv2, "-dbcv2", "--dirichlet-bc-2-vals",
                  "Dirichlet Boundary Condition Value 2 (v_x v_y v_z)"
                  " or (Re(v_x) Re(v_y) Re(v_z) Im(v_x) Im(v_y) Im(v_z))");
   args.AddOption(&dbcas, "-dbcs-msa", "--dirichlet-bc-straps",
                  "Dirichlet Boundary Condition Surfaces Using "
                  "Multi-Strap Antenna");
   args.AddOption(&nbca1, "-nbcs1", "--neumann-bc-1-surf",
                  "Neumann Boundary Condition Surfaces Using Value 1");
   args.AddOption(&nbca2, "-nbcs2", "--neumann-bc-2-surf",
                  "Neumann Boundary Condition Surfaces Using Value 2");
   args.AddOption(&nbcaw, "-nbcs-pw", "--neumann-bc-pw-surf",
                  "Neumann Boundary Condition Surfaces Using Plane Wave");
   args.AddOption(&nbcv1, "-nbcv1", "--neumann-bc-1-vals",
                  "Neuamnn Boundary Condition (surface current) "
                  "Value 1 (v_x v_y v_z) or "
                  "(Re(v_x) Re(v_y) Re(v_z) Im(v_x) Im(v_y) Im(v_z))");
   args.AddOption(&nbcv2, "-nbcv2", "--neumann-bc-2-vals",
                  "Neumann Boundary Condition (surface current) "
                  "Value 2 (v_x v_y v_z) or "
                  "(Re(v_x) Re(v_y) Re(v_z) Im(v_x) Im(v_y) Im(v_z))");
   // args.AddOption(&num_elements, "-ne", "--num-elements",
   //             "The number of mesh elements in x");
   args.AddOption(&maxit, "-maxit", "--max-amr-iterations",
                  "Max number of iterations in the main AMR loop.");
   args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
                  "--no-hermitian", "Use convention for Hermitian operators.");
   args.AddOption(&vis_u, "-vis-u", "--visualize-energy", "-no-vis-u",
                  "--no-visualize-energy",
                  "Enable or disable visualization of energy density.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&eqdsk_file, "-eqdsk", "--eqdsk-file",
                  "G EQDSK input file.");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (logo)
   {
      return 1;
   }
   Device device(device_config);
   if (mpi.Root())
   {
      device.Print();
   }

   if (dpp.Size() == 0)
   {
      dpp.SetSize(1);
      dpp[0] = 1.0e19;
   }

   if (nepp.Size() == 0)
   {
      nepp.SetSize(1);
      nepp[0] = 0;
   }
   if (nipp.Size() == 0)
   {
      nipp.SetSize(1);
      nipp[0] = 0;
   }

   if (bpp.Size() == 0)
   {
      bpt = BFieldProfile::CONSTANT;
      bpp.SetSize(3);
      bpp[0] = 0.0; bpp[1] = 0.1; bpp[2] = 0.0;
   }
   if (bpt == BFieldProfile::CONSTANT)
   {
      BVec = bpp;
   }

   if (charges.Size() == 0)
   {
      if (minority.Size() == 0)
      {
         charges.SetSize(2);
         charges[0] = -1.0;
         charges[1] =  1.0;

         masses.SetSize(2);
         masses[0] = me_u_;
         masses[1] = 2.01410178;
      }
      else
      {
         charges.SetSize(3);
         charges[0] = -1.0;
         charges[1] =  1.0;
         charges[2] = minority[0];

         masses.SetSize(3);
         masses[0] = me_u_;
         masses[1] = 2.01410178;
         masses[2] = minority[1];
      }
   }
   if (minority.Size() == 0)
   {
      if (charges.Size() == 2)
      {
         numbers.SetSize(2);
         masses.SetSize(2);
         masses[0] = me_u_;
         masses[1] = 2.01410178;
         switch (dpt)
         {
            case PlasmaProfile::CONSTANT:
               numbers[0] = dpp[0];
               numbers[1] = dpp[0];
               break;
            case PlasmaProfile::GRADIENT:
               numbers[0] = dpp[0];
               numbers[1] = dpp[0];
               break;
            case PlasmaProfile::TANH:
               numbers[0] = dpp[1];
               numbers[1] = dpp[1];
               break;
            case PlasmaProfile::ELLIPTIC_COS:
               numbers[0] = dpp[1];
               numbers[1] = dpp[1];
               break;
            case PlasmaProfile::PARABOLIC:
               numbers[0] = dpp[1];
               numbers[1] = dpp[1];
               break;
            default:
               numbers[0] = 1.0e19;
               numbers[1] = 1.0e19;
               break;
         }
      }
      else
      {
         numbers.SetSize(3);
         masses.SetSize(3);
         masses[0] = me_u_;
         masses[1] = 2.01410178;
         masses[2] = 3.01604928;
         switch (dpt)
         {
            case PlasmaProfile::CONSTANT:
               numbers[0] = dpp[0];
               numbers[1] = 0.5*dpp[0];
               numbers[2] = 0.5*dpp[0];
               break;
            case PlasmaProfile::GRADIENT:
               numbers[0] = dpp[0];
               numbers[1] = 0.5*dpp[0];
               numbers[2] = 0.5*dpp[0];
               break;
            case PlasmaProfile::TANH:
               numbers[0] = dpp[1];
               numbers[1] = 0.5*dpp[1];
               numbers[2] = 0.5*dpp[1];
               break;
            case PlasmaProfile::ELLIPTIC_COS:
               numbers[0] = dpp[1];
               numbers[1] = 0.5*dpp[1];
               numbers[2] = 0.5*dpp[1];
               break;
            case PlasmaProfile::PARABOLIC:
               numbers[0] = dpp[1];
               numbers[1] = 0.5*dpp[1];
               numbers[2] = 0.5*dpp[1];
               break;
            default:
               numbers[0] = 1.0e19;
               numbers[1] = 0.5*1.0e19;
               numbers[2] = 0.5*1.0e19;
               break;
         }
      }
   }
   if (minority.Size() > 0)
   {
      if (charges.Size() == 2)
      {
         temp_charges.SetSize(3);
         temp_charges[0] = charges[0];
         temp_charges[1] = charges[1];
         temp_charges[2] = minority[0];
         charges.SetSize(3);
         charges = temp_charges;

         numbers.SetSize(3);
         masses.SetSize(3);
         masses[0] = me_u_;
         masses[1] = 2.01410178;
         masses[2] = minority[1];
         switch (dpt)
         {
            case PlasmaProfile::CONSTANT:
               numbers[0] = dpp[0];
               numbers[1] = (1.0/(1.0+minority[0]*minority[2]))*dpp[0];
               numbers[2] = ((minority[0]*minority[2])/(1.0+minority[0]*minority[2]))*dpp[0];
               break;
            case PlasmaProfile::GRADIENT:
               numbers[0] = dpp[0];
               numbers[1] = (1.0/(1.0+minority[0]*minority[2]))*dpp[0];
               numbers[2] = ((minority[0]*minority[2])/(1.0+minority[0]*minority[2]))*dpp[0];
               break;
            case PlasmaProfile::TANH:
               numbers[0] = dpp[1];
               numbers[1] = (1.0/(1.0+minority[0]*minority[2]))*dpp[1];
               numbers[2] = ((minority[0]*minority[2])/(1.0+minority[0]*minority[2]))*dpp[1];
               break;
            case PlasmaProfile::ELLIPTIC_COS:
               numbers[0] = dpp[1];
               numbers[1] = (1.0/(1.0+minority[0]*minority[2]))*dpp[1];
               numbers[2] = ((minority[0]*minority[2])/(1.0+minority[0]*minority[2]))*dpp[1];
               break;
            case PlasmaProfile::PARABOLIC:
               numbers[0] = dpp[1];
               numbers[1] = (1.0/(1.0+minority[0]*minority[2]))*dpp[1];
               numbers[2] = ((minority[0]*minority[2])/(1.0+minority[0]*minority[2]))*dpp[1];
               break;
            default:
               numbers[0] = 1.0e19;
               numbers[1] = (1.0/(1.0+minority[0]*minority[2]))*1.0e19;
               numbers[2] = ((minority[0]*minority[2])/(1.0+minority[0]*minority[2]))*1.0e19;
               break;
         }
      }
      else
      {
         temp_charges.SetSize(4);
         temp_charges[0] = charges[0];
         temp_charges[1] = charges[1];
         temp_charges[2] = charges[2];
         temp_charges[3] = minority[0];
         charges.SetSize(4);
         charges = temp_charges;

         numbers.SetSize(4);
         masses.SetSize(4);
         masses[0] = me_u_;
         masses[1] = 2.01410178;
         masses[2] = 3.01604928;
         masses[3] = minority[1];
         switch (dpt)
         {
            case PlasmaProfile::CONSTANT:
               numbers[0] = dpp[0];
               numbers[1] = 0.5*(1.0/(1.0+minority[0]*minority[2]))*dpp[0];
               numbers[2] = 0.5*(1.0/(1.0+minority[0]*minority[2]))*dpp[0];
               numbers[3] = ((minority[0]*minority[2])/(1.0+minority[0]*minority[2]))*dpp[0];
               break;
            case PlasmaProfile::GRADIENT:
               numbers[0] = dpp[0];
               numbers[1] = 0.5*(1.0/(1.0+minority[0]*minority[2]))*dpp[0];
               numbers[2] = 0.5*(1.0/(1.0+minority[0]*minority[2]))*dpp[0];
               numbers[3] = ((minority[0]*minority[2])/(1.0+minority[0]*minority[2]))*dpp[0];
               break;
            case PlasmaProfile::TANH:
               numbers[0] = dpp[1];
               numbers[1] = 0.5*(1.0/(1.0+minority[0]*minority[2]))*dpp[1];
               numbers[2] = 0.5*(1.0/(1.0+minority[0]*minority[2]))*dpp[1];
               numbers[3] = ((minority[0]*minority[2])/(1.0+minority[0]*minority[2]))*dpp[1];
               break;
            case PlasmaProfile::ELLIPTIC_COS:
               numbers[0] = dpp[1];
               numbers[1] = 0.5*(1.0/(1.0+minority[0]*minority[2]))*dpp[1];
               numbers[2] = 0.5*(1.0/(1.0+minority[0]*minority[2]))*dpp[1];
               numbers[3] = ((minority[0]*minority[2])/(1.0+minority[0]*minority[2]))*dpp[1];
               break;
            case PlasmaProfile::PARABOLIC:
               numbers[0] = dpp[1];
               numbers[1] = 0.5*(1.0/(1.0+minority[0]*minority[2]))*dpp[1];
               numbers[2] = 0.5*(1.0/(1.0+minority[0]*minority[2]))*dpp[1];
               numbers[3] = ((minority[0]*minority[2])/(1.0+minority[0]*minority[2]))*dpp[1];
               break;
            default:
               numbers[0] = 1.0e19;
               numbers[1] = 0.5*(1.0/(1.0+minority[0]*minority[2]))*1.0e19;
               numbers[2] = 0.5*(1.0/(1.0+minority[0]*minority[2]))*1.0e19;
               numbers[3] = ((minority[0]*minority[2])/(1.0+minority[0]*minority[2]))*1.0e19;
               break;
         }
      }
   }
   if (temps.Size() == 0)
   {
      temps.SetSize(numbers.Size());
      if (tpp.Size() == 0)
      {
         tpp.SetSize(1);
         tpp[0] = 1.0e3;
         for (int i=0; i<numbers.Size(); i++) {temps[i] = tpp[0];}
      }
      else
      {
         switch (tpt)
         {
            case PlasmaProfile::CONSTANT:
               for (int i=0; i<numbers.Size(); i++) {temps[i] = tpp[0];}
               break;
            case PlasmaProfile::GRADIENT:
               for (int i=0; i<numbers.Size(); i++) {temps[i] = tpp[0];}
               break;
            case PlasmaProfile::TANH:
               for (int i=0; i<numbers.Size(); i++) {temps[i] = tpp[1];}
               break;
            case PlasmaProfile::ELLIPTIC_COS:
               for (int i=0; i<numbers.Size(); i++) {temps[i] = tpp[1];}
               break;
            case PlasmaProfile::PARABOLIC:
               for (int i=0; i<numbers.Size(); i++) {temps[i] = tpp[1];}
               break;
            default:
               for (int i=0; i<numbers.Size(); i++) {temps[i] = 1e3;}
               break;
         }
      }
   }
   if (num_elements <= 0)
   {
      num_elements = 10;
   }
   if (hz < 0.0 && !cyl)
   {
      hz = 0.1;
   }
   if (cyl)
   {
      if (mesh_order <= 0)
      {
         mesh_order = 1;
      }
      if (hphi < 0.0)
      {
         hphi = 3;
      }
      hz = 1.0;

      j_cyl_ = cyl;
   }
   double omega = 2.0 * M_PI * freq;
   if (kVec.Size() != 0)
   {
      phase_shift = true;
   }

   if (mpi.Root())
   {
      args.PrintOptions(cout);
   }

   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   if (mpi.Root())
   {
      double lam0 = c0_ / freq;
      double Bmag = BVec.Norml2();
      std::complex<double> S = S_cold_plasma(omega, Bmag, nue, nui, numbers,
                                             charges, masses, temps, nuprof);
      std::complex<double> P = P_cold_plasma(omega, nue, numbers,
                                             charges, masses, temps, nuprof);
      std::complex<double> D = D_cold_plasma(omega, Bmag, nue, nui, numbers,
                                             charges, masses, temps, nuprof);
      std::complex<double> R = R_cold_plasma(omega, Bmag, nue, nui, numbers,
                                             charges, masses, temps, nuprof);
      std::complex<double> L = L_cold_plasma(omega, Bmag, nue, nui, numbers,
                                             charges, masses, temps, nuprof);

      cout << "\nConvenient Terms:\n";
      cout << "R = " << R << ",\tL = " << L << endl;
      cout << "S = " << S << ",\tD = " << D << ",\tP = " << P << endl;

      cout << "\nSpecies Properties (number, charge, mass):\n";
      for (int i=0; i<numbers.Size(); i++)
      {
         cout << numbers[i] << '\t' << charges[i] << '\t' << masses[i] << '\n';
      }
      cout << "\nPlasma and Cyclotron Frequencies by Species (GHz):\n";
      for (int i=0; i<numbers.Size(); i++)
      {
         cout << omega_p(numbers[i], charges[i], masses[i]) / (2.0e9 * M_PI)
              << '\t'
              << omega_c(Bmag, charges[i], masses[i]) / (2.0e9 * M_PI) << '\n';
      }

      cout << "\nWavelengths (meters):\n";
      cout << "   Free Space Wavelength: " << lam0 << '\n';
      complex<double> lamL = lam0 / sqrt(S-D);
      complex<double> lamR = lam0 / sqrt(S+D);
      complex<double> lamO = lam0 / sqrt(P);
      complex<double> lamX = lam0 * sqrt(S/(S*S-D*D));
      if (fabs(lamL.real()) > fabs(lamL.imag()))
      {
         cout << "   Oscillating L mode:    " << lamL << '\n';
      }
      else
      {
         cout << "   Decaying L mode:       " << lamL << '\n';
      }
      if (fabs(lamR.real()) > fabs(lamR.imag()))
      {
         cout << "   Oscillating R mode:    " << lamR << '\n';
      }
      else
      {
         cout << "   Decaying R mode:       " << lamR << '\n';
      }
      if (fabs(lamO.real()) > fabs(lamO.imag()))
      {
         cout << "   Oscillating O mode:    " << lamO << '\n';
      }
      else
      {
         cout << "   Decaying O mode:       " << lamO << '\n';
      }
      if (fabs(lamX.real()) > fabs(lamX.imag()))
      {
         cout << "   Oscillating X mode:    " << lamX << '\n';
      }
      else
      {
         cout << "   Decaying X mode:       " << lamX << '\n';
      }
      cout << endl;
   }

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   if ( mpi.Root() && logging > 0 )
   {
      cout << "Building Extruded 2D Mesh ..." << endl;
   }

   tic_toc.Clear();
   tic_toc.Start();

   Mesh * mesh2d = new Mesh(mesh_file, 1, 1);
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh2d->UniformRefinement();
   }
   Mesh * mesh = Extrude2D(mesh2d, 3, hz);
   delete mesh2d;
   if (cyl)
   {
      mesh->SetCurvature(mesh_order);

      MeshTransformCoefficient mtc(hphi);
      mesh->Transform(mtc);
   }
   {
      std::vector<int> v2v(mesh->GetNV());
      for (int i=0; i<v2v.size(); i++) { v2v[i] = i; }
      for (int i=0; i<mesh->GetNV() / 4; i++) { v2v[4 * i + 3] = 4 * i; }

      Mesh per_mesh = Mesh::MakePeriodic(*mesh, v2v);
      delete mesh;
      mesh = new Mesh(per_mesh);
   }
   tic_toc.Stop();

   if (mpi.Root() && logging > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }

   // Ensure that quad and hex meshes are treated as non-conforming.
   if (maxit > 1 || amr_s)
   {
      mesh->EnsureNCMesh();
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   if ( mpi.Root() && logging > 0 )
   { cout << "Building Parallel Mesh ..." << endl; }
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   if (mpi.Root())
   {
      cout << "Starting initialization." << endl;
   }
   /*
      double Bmag = BVec.Norml2();
      Vector BUnitVec(3);
      BUnitVec(0) = BVec(0)/Bmag;
      BUnitVec(1) = BVec(1)/Bmag;
      BUnitVec(2) = BVec(2)/Bmag;

      VectorConstantCoefficient BCoef(BVec);
      VectorConstantCoefficient BUnitCoef(BUnitVec);
    */

   H1_ParFESpace H1FESpace(&pmesh, order, pmesh.Dimension());
   ND_ParFESpace HCurlFESpace(&pmesh, order, pmesh.Dimension());
   RT_ParFESpace HDivFESpace(&pmesh, order, pmesh.Dimension());
   L2_ParFESpace L2FESpace(&pmesh, order, pmesh.Dimension());

   ParGridFunction BField(&HDivFESpace);
   ParGridFunction temperature_gf;
   ParGridFunction density_gf;
   ParGridFunction nue_gf(&H1FESpace);
   ParGridFunction nui_gf(&H1FESpace);

   G_EQDSK_Data *eqdsk = NULL;
   {
      named_ifgzstream ieqdsk(eqdsk_file);
      if (ieqdsk)
      {
         eqdsk = new G_EQDSK_Data(ieqdsk);
         if (mpi.Root())
         {
            eqdsk->PrintInfo();
            if (logging > 0)
            {
               eqdsk->DumpGnuPlotData("stix2d_dh_eqdsk");
            }
         }
      }
   }

   /*
   if (eqdsk)
     {
        if (mpi.Root())
         {
            cout << "Using B field from " << eqdsk_file << endl;
         }
         G_EQDSK_BField_VecCoefficient BCoef(*eqdsk, false);
         G_EQDSK_BField_VecCoefficient BUnitCoef(*eqdsk, true);
     }
   else
   {
        BFieldProfile BCoef(bpt, bpp, false);
        BFieldProfile BUnitCoef(bpt, bpp, true);
   }
   */

   BFieldProfile::CoordSystem b_coord_sys =
      cyl ? BFieldProfile::POLOIDAL : BFieldProfile::CARTESIAN_3D;
   BFieldProfile BCoef(bpt, bpp, false, b_coord_sys, eqdsk);
   BFieldProfile BUnitCoef(bpt, bpp, true, b_coord_sys, eqdsk);

   BField.ProjectCoefficient(BCoef);

   PlasmaProfile::CoordSystem coord_sys =
      cyl ? PlasmaProfile::POLOIDAL : PlasmaProfile::CARTESIAN_3D;
   PlasmaProfile nueCoef(nept, nepp, coord_sys, eqdsk);
   nue_gf.ProjectCoefficient(nueCoef);
   PlasmaProfile nuiCoef(nipt, nipp, coord_sys, eqdsk);
   nui_gf.ProjectCoefficient(nuiCoef);

   int size_h1 = H1FESpace.GetVSize();
   int size_l2 = L2FESpace.GetVSize();

   Array<int> density_offsets(numbers.Size() + 1);
   Array<int> temperature_offsets(numbers.Size() + 2);

   density_offsets[0] = 0;
   temperature_offsets[0] = 0;
   temperature_offsets[1] = size_h1;

   for (int i=1; i<=numbers.Size(); i++)
   {
      density_offsets[i]     = density_offsets[i - 1] + size_l2;
      temperature_offsets[i + 1] = temperature_offsets[i] + size_h1;
   }

   BlockVector density(density_offsets);
   BlockVector temperature(temperature_offsets);

   if (mpi.Root())
   {
      cout << "Creating plasma profile." << endl;
   }

   PlasmaProfile tempCoef(tpt, tpp, coord_sys, eqdsk);
   PlasmaProfile rhoCoef(dpt, dpp, coord_sys, eqdsk);

   for (int i=0; i<=numbers.Size(); i++)
   {
      temperature_gf.MakeRef(&H1FESpace, temperature.GetBlock(i));
      temperature_gf.ProjectCoefficient(tempCoef);
   }
   for (int i=0; i<charges.Size(); i++)
   {
      density_gf.MakeRef(&L2FESpace, density.GetBlock(i));
      density_gf.ProjectCoefficient(rhoCoef);
      density_gf *= numbers[i]/numbers[0];
   }

   if (amr_s)
   {
      if (mpi.Root())
      {
         cout << "Adapting mesh to Stix 'P' coefficient." << endl;
      }

      StixPCoef RePCoef(BField, nue_gf, nui_gf, density, temperature,
                        L2FESpace, H1FESpace,
                        omega, charges, masses, nuprof,
                        true);
      StixPCoef ImPCoef(BField, nue_gf, nui_gf, density, temperature,
                        L2FESpace, H1FESpace,
                        omega, charges, masses, nuprof,
                        false);

      L2_ParFESpace err_fes(&pmesh, 0, pmesh.Dimension());

      AdaptInitialMesh(mpi, pmesh, err_fes,
                       H1FESpace, HCurlFESpace, HDivFESpace, L2FESpace,
                       BCoef, rhoCoef, tempCoef, nueCoef, nuiCoef,
                       size_h1, size_l2,
                       density_offsets, temperature_offsets,
                       density, temperature,
                       BField, density_gf, temperature_gf, nue_gf, nui_gf,
                       RePCoef, ImPCoef,
                       order,
                       init_amr_tol, init_amr_max_its, init_amr_max_dofs,
                       visualization);
   }

   if (mpi.Root())
   {
      cout << "Creating coefficients for Maxwell equations." << endl;
   }

   // Create a coefficient describing the magnetic permeability
   ConstantCoefficient muCoef(mu0_);

   // Create a coefficient describing the surface admittance
   Coefficient * etaCoef = SetupImpedanceCoefficient(pmesh, abcs);

   // Create tensor coefficients describing the dielectric permittivity
   InverseDielectricTensor epsilonInv_real(BField, nue_gf, nui_gf, density,
                                           temperature,
                                           L2FESpace, H1FESpace,
                                           omega, charges, masses, nuprof,
                                           true);
   InverseDielectricTensor epsilonInv_imag(BField, nue_gf, nui_gf, density,
                                           temperature,
                                           L2FESpace, H1FESpace,
                                           omega, charges, masses, nuprof,
                                           false);
   SPDDielectricTensor epsilon_abs(BField, nue_gf, nui_gf, density, temperature,
                                   L2FESpace, H1FESpace,
                                   omega, charges, masses, nuprof);
   SheathImpedance z_r(BField, density, temperature,
                       L2FESpace, H1FESpace,
                       omega, charges, masses, true);
   SheathImpedance z_i(BField, density, temperature,
                       L2FESpace, H1FESpace,
                       omega, charges, masses, false);

   MultiStrapAntennaH HReStrapCoef(msa_n, msa_p, msa_c, true);
   MultiStrapAntennaH HImStrapCoef(msa_n, msa_p, msa_c, false);

   ColdPlasmaPlaneWaveH HReCoef(wave_type[0], omega, BVec,
                                numbers, charges, masses, temps, nuprof, true);
   ColdPlasmaPlaneWaveH HImCoef(wave_type[0], omega, BVec,
                                numbers, charges, masses, temps, nuprof, false);

   ColdPlasmaPlaneWaveE EReCoef(wave_type[0], omega, BVec,
                                numbers, charges, masses, temps, nuprof, true);
   ColdPlasmaPlaneWaveE EImCoef(wave_type[0], omega, BVec,
                                numbers, charges, masses, temps, nuprof, false);

   if (check_eps_inv)
   {
      DielectricTensor epsilon_real(BField, nue_gf, nui_gf, density, temperature,
                                    L2FESpace, H1FESpace,
                                    omega, charges, masses, nuprof, true);
      DielectricTensor epsilon_imag(BField, nue_gf, nui_gf, density, temperature,
                                    L2FESpace, H1FESpace,
                                    omega, charges, masses, nuprof, false);
      DenseMatrix epsInvRe(3,3);
      DenseMatrix epsInvIm(3,3);
      DenseMatrix epsRe(3,3);
      DenseMatrix epsIm(3,3);

      DenseMatrix IRe(3,3);
      DenseMatrix IIm(3,3);

      for (int i=0; i<pmesh.GetNE(); i++)
      {
         ElementTransformation *T = pmesh.GetElementTransformation(i);
         Geometry::Type g = pmesh.GetElementBaseGeometry(i);
         const IntegrationPoint &ip = Geometries.GetCenter(g);

         epsilonInv_real.Eval(epsInvRe, *T, ip);
         epsilonInv_imag.Eval(epsInvIm, *T, ip);

         epsilon_real.Eval(epsRe, *T, ip);
         epsilon_imag.Eval(epsIm, *T, ip);

         Mult(epsInvRe, epsRe, IRe);
         AddMult_a(-1.0, epsInvIm, epsIm, IRe);

         Mult(epsInvRe, epsIm, IIm);
         AddMult(epsInvIm, epsRe, IIm);

         IRe(0,0) -= 1.0;
         IRe(1,1) -= 1.0;
         IRe(2,2) -= 1.0;

         double nrmRe = IRe.MaxMaxNorm();
         double nrmIm = IIm.MaxMaxNorm();

         if (nrmRe + nrmIm > 1e-13)
         {
            cout << "element " << i << " on processor "
                 << mpi.WorldRank() << endl;
            IRe.Print(cout);
            IIm.Print(cout);
            cout << endl;
         }
      }
   }

   if (wave_type[0] != ' ')
   {
      Vector kr(3), ki(3);
      HReCoef.GetWaveVector(kr, ki);

      mfem::out << "Plane wave propagation vector: ("
                << complex<double>(kr(0),ki(0)) << ","
                << complex<double>(kr(1),ki(1)) << ","
                << complex<double>(kr(2),ki(2)) << ")" << endl;

      if (!phase_shift)
      {
         kVec.SetSize(6);
         kVec = 0.0;

         if (per_y)
         {
            kVec[1] = kr[1];
            kVec[4] = ki[1];
         }

         kVec[2] = kr[2];
         kVec[5] = ki[2];

         phase_shift = true;
      }

      kReVec.SetDataAndSize(&kVec[0], 3);
      kImVec.SetDataAndSize(&kVec[3], 3);

      HReCoef.SetPhaseShift(kReVec, kImVec);
      HImCoef.SetPhaseShift(kReVec, kImVec);
      EReCoef.SetPhaseShift(kReVec, kImVec);
      EImCoef.SetPhaseShift(kReVec, kImVec);
   }
   else
   {
      if (phase_shift)
      {
         if (kVec.Size() >= 3)
         {
            kReVec.SetDataAndSize(&kVec[0], 3);
         }
         else
         {
            kReVec.SetSize(3);
            kReVec = 0.0;
         }
         if (kVec.Size() >= 6)
         {
            kImVec.SetDataAndSize(&kVec[3], 3);
         }
         else
         {
            kImVec.SetSize(3);
            kImVec = 0.0;
         }
      }
   }

   mfem::out << "Setting phase shift of ("
             << complex<double>(kReVec[0],kImVec[0]) << ","
             << complex<double>(kReVec[1],kImVec[1]) << ","
             << complex<double>(kReVec[2],kImVec[2]) << ")" << endl;

   VectorConstantCylCoefficient kReCoef(cyl, kReVec);
   VectorConstantCylCoefficient kImCoef(cyl, kImVec);

   /*
   if (wave_type[0] == 'J' && slab_params_.Size() == 5)
   {
      EReCoef.SetCurrentSlab(slab_params_[1], slab_params_[3], slab_params_[4],
                             mesh_dim_[0]);
      EImCoef.SetCurrentSlab(slab_params_[1], slab_params_[3], slab_params_[4],
                             mesh_dim_[0]);
   }
   */

   if (visualization && wave_type[0] != ' ')
   {
      if (mpi.Root())
      {
         cout << "Visualize input fields." << endl;
      }
      ParComplexGridFunction HField(&HCurlFESpace);
      HField.ProjectCoefficient(HReCoef, HImCoef);
      ParComplexGridFunction EField(&HCurlFESpace);
      EField.ProjectCoefficient(EReCoef, EImCoef);

      Vector zeroVec(3); zeroVec = 0.0;
      VectorConstantCoefficient zeroCoef(zeroVec);
      double max_Hr = HField.real().ComputeMaxError(zeroCoef);
      double max_Hi = HField.imag().ComputeMaxError(zeroCoef);
      double max_Er = EField.real().ComputeMaxError(zeroCoef);
      double max_Ei = EField.imag().ComputeMaxError(zeroCoef);

      char vishost[] = "localhost";
      int  visport   = 19916;

      int Wx = 0, Wy = 0; // window position
      int Ww = 350, Wh = 350; // window size
      int offx = Ww+10, offy = Wh+45; // window offsets

      socketstream sock_Hr, sock_Hi, sock_Er, sock_Ei, sock_B;
      sock_Hr.precision(8);
      sock_Hi.precision(8);
      sock_Er.precision(8);
      sock_Ei.precision(8);
      sock_B.precision(8);

      ostringstream hr_keys, hi_keys;
      hr_keys << "aaAcPPPPvvv valuerange 0.0 " << max_Hr;
      hi_keys << "aaAcPPPPvvv valuerange 0.0 " << max_Hi;

      ostringstream er_keys, ei_keys;
      er_keys << "aaAcpppppvvv valuerange 0.0 " << max_Er;
      ei_keys << "aaAcpppppvvv valuerange 0.0 " << max_Ei;

      Wy += offy;
      VisualizeField(sock_Hr, vishost, visport,
                     HField.real(), "Exact Magnetic Field, Re(H)",
                     Wx, Wy, Ww, Wh, hr_keys.str().c_str());

      Wx += offx;
      VisualizeField(sock_Hi, vishost, visport,
                     HField.imag(), "Exact Magnetic Field, Im(H)",
                     Wx, Wy, Ww, Wh, hi_keys.str().c_str());

      Wx += offx;
      VisualizeField(sock_Er, vishost, visport,
                     EField.real(), "Exact Electric Field, Re(E)",
                     Wx, Wy, Ww, Wh, er_keys.str().c_str());
      Wx += offx;
      VisualizeField(sock_Ei, vishost, visport,
                     EField.imag(), "Exact Electric Field, Im(E)",
                     Wx, Wy, Ww, Wh, ei_keys.str().c_str());

      Wx -= offx;
      Wy += offy;

      VisualizeField(sock_B, vishost, visport,
                     BField, "Background Magnetic Field",
                     Wx, Wy, Ww, Wh);

      for (int i=0; i<charges.Size(); i++)
      {
         Wx += offx;

         socketstream sock;
         sock.precision(8);

         stringstream oss;
         oss << "Density Species " << i;
         density_gf.MakeRef(&L2FESpace, density.GetBlock(i));
         VisualizeField(sock, vishost, visport,
                        density_gf, oss.str().c_str(),
                        Wx, Wy, Ww, Wh);
      }
   }

   if (mpi.Root())
   {
      cout << "Setup boundary conditions." << endl;
   }

   // Setup coefficients for Dirichlet BC
   int dbcsSize = (peca.Size() > 0) + (dbca1.Size() > 0) + (dbca2.Size() > 0) +
                  (dbcas.Size() > 0) + (dbcaw.Size() > 0);

   Array<ComplexVectorCoefficientByAttr*> dbcs(dbcsSize);

   Vector zeroVec(3); zeroVec = 0.0;
   Vector dbc1ReVec;
   Vector dbc1ImVec;
   Vector dbc2ReVec;
   Vector dbc2ImVec;

   if (dbcv1.Size() >= 3)
   {
      dbc1ReVec.SetDataAndSize(&dbcv1[0], 3);
   }
   else
   {
      dbc1ReVec.SetDataAndSize(&zeroVec[0], 3);
   }
   if (dbcv1.Size() >= 6)
   {
      dbc1ImVec.SetDataAndSize(&dbcv1[3], 3);
   }
   else
   {
      dbc1ImVec.SetDataAndSize(&zeroVec[0], 3);
   }
   if (dbcv2.Size() >= 3)
   {
      dbc2ReVec.SetDataAndSize(&dbcv2[0], 3);
   }
   else
   {
      dbc2ReVec.SetDataAndSize(&zeroVec[0], 3);
   }
   if (dbcv2.Size() >= 6)
   {
      dbc2ImVec.SetDataAndSize(&dbcv2[3], 3);
   }
   else
   {
      dbc2ImVec.SetDataAndSize(&zeroVec[0], 3);
   }

   VectorConstantCoefficient zeroCoef(zeroVec);
   VectorConstantCoefficient dbc1ReCoef(dbc1ReVec);
   VectorConstantCoefficient dbc1ImCoef(dbc1ImVec);
   VectorConstantCoefficient dbc2ReCoef(dbc2ReVec);
   VectorConstantCoefficient dbc2ImCoef(dbc2ImVec);

   if (dbcsSize > 0)
   {
      int c = 0;
      if (peca.Size() > 0)
      {
         dbcs[c] = new ComplexVectorCoefficientByAttr;
         dbcs[c]->attr = peca;
         dbcs[c]->real = &zeroCoef;
         dbcs[c]->imag = &zeroCoef;
         c++;
      }
      if (dbca1.Size() > 0)
      {
         dbcs[c] = new ComplexVectorCoefficientByAttr;
         dbcs[c]->attr = dbca1;
         dbcs[c]->real = &dbc1ReCoef;
         dbcs[c]->imag = &dbc1ImCoef;
         c++;
      }
      if (dbca2.Size() > 0)
      {
         dbcs[c] = new ComplexVectorCoefficientByAttr;
         dbcs[c]->attr = dbca2;
         dbcs[c]->real = &dbc2ReCoef;
         dbcs[c]->imag = &dbc2ImCoef;
         c++;
      }
      if (dbcas.Size() > 0)
      {
         dbcs[c] = new ComplexVectorCoefficientByAttr;
         dbcs[c]->attr = dbcas;
         dbcs[c]->real = &HReStrapCoef;
         dbcs[c]->imag = &HImStrapCoef;
         c++;
      }
      if (dbcaw.Size() > 0)
      {
         dbcs[c] = new ComplexVectorCoefficientByAttr;
         dbcs[c]->attr = dbcaw;
         dbcs[c]->real = &HReCoef;
         dbcs[c]->imag = &HImCoef;
         c++;
      }
   }

   int nbcsSize = (nbca1.Size() > 0) + (nbca2.Size() > 0) + (nbcaw.Size() > 0);

   Array<ComplexVectorCoefficientByAttr*> nbcs(nbcsSize);

   Vector nbc1ReVec;
   Vector nbc1ImVec;
   Vector nbc2ReVec;
   Vector nbc2ImVec;

   if (nbcv1.Size() >= 3)
   {
      nbc1ReVec.SetDataAndSize(&nbcv1[0], 3);
   }
   else
   {
      nbc1ReVec.SetDataAndSize(&zeroVec[0], 3);
   }
   if (nbcv1.Size() >= 6)
   {
      nbc1ImVec.SetDataAndSize(&nbcv1[3], 3);
   }
   else
   {
      nbc1ImVec.SetDataAndSize(&zeroVec[0], 3);
   }
   if (nbcv2.Size() >= 3)
   {
      nbc2ReVec.SetDataAndSize(&nbcv2[0], 3);
   }
   else
   {
      nbc2ReVec.SetDataAndSize(&zeroVec[0], 3);
   }
   if (nbcv2.Size() >= 6)
   {
      nbc2ImVec.SetDataAndSize(&nbcv2[3], 3);
   }
   else
   {
      nbc2ImVec.SetDataAndSize(&zeroVec[0], 3);
   }

   VectorConstantCoefficient nbc1ReCoef(nbc1ReVec);
   VectorConstantCoefficient nbc1ImCoef(nbc1ImVec);
   VectorConstantCoefficient nbc2ReCoef(nbc2ReVec);
   VectorConstantCoefficient nbc2ImCoef(nbc2ImVec);

   if (nbcsSize > 0)
   {
      int c = 0;
      if (nbca1.Size() > 0)
      {
         nbcs[c] = new ComplexVectorCoefficientByAttr;
         nbcs[c]->attr = nbca1;
         nbcs[c]->real = &nbc1ReCoef;
         nbcs[c]->imag = &nbc1ImCoef;
         c++;
      }
      if (nbca2.Size() > 0)
      {
         nbcs[c] = new ComplexVectorCoefficientByAttr;
         nbcs[c]->attr = nbca2;
         nbcs[c]->real = &nbc2ReCoef;
         nbcs[c]->imag = &nbc2ImCoef;
         c++;
      }
      if (nbcaw.Size() > 0)
      {
         nbcs[c] = new ComplexVectorCoefficientByAttr;
         nbcs[c]->attr = nbcaw;
         nbcs[c]->real = &EReCoef;
         nbcs[c]->imag = &EImCoef;
         c++;
      }
   }

   Array<ComplexCoefficientByAttr*> sbcs((sbca.Size() > 0)? 1 : 0);
   if (sbca.Size() > 0)
   {
      sbcs[0] = new ComplexCoefficientByAttr;
      sbcs[0]->real = &z_r;
      sbcs[0]->imag = &z_i;
      sbcs[0]->attr = sbca;
      AttrToMarker(pmesh.bdr_attributes.Max(), sbcs[0]->attr,
                   sbcs[0]->attr_marker);
   }

   if (mpi.Root())
   {
      cout << "Creating Cold Plasma Dielectric solver." << endl;
   }

   // Create the cold plasma EM solver
   CPDSolverDH CPD(pmesh, order, omega,
                   (CPDSolverDH::SolverType)sol, solOpts,
                   (CPDSolverDH::PrecondType)prec,
                   conv, BUnitCoef,
                   epsilonInv_real, epsilonInv_imag, epsilon_abs,
                   muCoef, etaCoef,
                   (phase_shift) ? &kReCoef : NULL,
                   (phase_shift) ? &kImCoef : NULL,
                   abcs,
                   dbcs, nbcs, sbcs,
                   // e_bc_r, e_bc_i,
                   // EReCoef, EImCoef,
                   (rod_params_.Size() > 0 ||slab_params_.Size() > 0 ||curve_params_.Size() > 0) ?
                   j_src_r : NULL,
                   (rod_params_.Size() > 0 ||slab_params_.Size() > 0 ||curve_params_.Size() > 0) ?
                   j_src_i : NULL, vis_u, pa);

   // Initialize GLVis visualization
   if (visualization)
   {
      CPD.InitializeGLVis();
   }

   // Initialize VisIt visualization
   VisItDataCollection visit_dc("STIX2D-DH-AMR-Parallel", &pmesh);

   Array<ParComplexGridFunction*> auxFields;

   if ( visit )
   {
      CPD.RegisterVisItFields(visit_dc);

      /*
      auxFields.SetSize(2);
      auxFields[0] = new ParComplexGridFunction(&HCurlFESpace);
      auxFields[1] = new ParComplexGridFunction(&HCurlFESpace);

      auxFields[0]->ProjectCoefficient(HReCoef, HImCoef);
      auxFields[1]->ProjectCoefficient(EReCoef, EImCoef);

      visit_dc.RegisterField("Re_H_Exact", &auxFields[0]->real());
      visit_dc.RegisterField("Im_H_Exact", &auxFields[0]->imag());

      visit_dc.RegisterField("Re_E_Exact", &auxFields[1]->real());
      visit_dc.RegisterField("Im_E_Exact", &auxFields[1]->imag());
       */
      temperature_gf.MakeRef(&H1FESpace, temperature.GetBlock(0));
      visit_dc.RegisterField("Electron_Temp", &temperature_gf);

      density_gf.MakeRef(&L2FESpace, density.GetBlock(0));
      visit_dc.RegisterField("Electron_Density", &density_gf);

      //nue_gf *= 1/omega;
      visit_dc.RegisterField("Collisional_Profile", &nue_gf);

      visit_dc.RegisterField("B_background", &BField);

      visit_dc.SetCycle(0);
      visit_dc.Save();
   }
   if (mpi.Root()) { cout << "Initialization done." << endl; }

   // The main AMR loop. In each iteration we solve the problem on the current
   // mesh, visualize the solution, estimate the error on all elements, refine
   // the worst elements and update all objects to work with the new mesh. We
   // refine until the maximum number of dofs in the Nedelec finite element
   // space reaches 10 million.
   const int max_dofs = 10000000;
   for (int it = 1; it <= maxit; it++)
   {
      if (mpi.Root())
      {
         cout << "\nAMR Iteration " << it << endl;
      }

      // Display the current number of DoFs in each finite element space
      CPD.PrintSizes();

      // Assemble all forms
      CPD.Assemble();

      // Solve the system and compute any auxiliary fields
      CPD.Solve();

      if (wave_type[0] != ' ')
      {
         // Compute error
         double glb_error_H = CPD.GetHFieldError(HReCoef, HImCoef);
         if (mpi.Root())
         {
            cout << "Global L2 Error in H field " << glb_error_H << endl;
         }

         double glb_error_E = CPD.GetEFieldError(EReCoef, EImCoef);
         if (mpi.Root())
         {
            cout << "Global L2 Error in E field " << glb_error_E << endl;
         }
      }

      // Determine the current size of the linear system
      int prob_size = CPD.GetProblemSize();

      // Write fields to disk for VisIt
      if ( visit )
      {
         CPD.WriteVisItFields(it);
      }

      // Send the solution by socket to a GLVis server.
      if (visualization)
      {
         CPD.DisplayToGLVis();
      }

      if (mpi.Root())
      {
         cout << "AMR iteration " << it << " complete." << endl;
      }

      // Check stopping criteria
      if (prob_size > max_dofs)
      {
         if (mpi.Root())
         {
            cout << "Reached maximum number of dofs, exiting..." << endl;
         }
         break;
      }
      if ( it == maxit )
      {
         break;
      }

      // Wait for user input. Ask every 10th iteration.
      char c = 'c';
      if (mpi.Root() && (it % 10 == 0))
      {
         cout << "press (q)uit or (c)ontinue --> " << flush;
         cin >> c;
      }
      MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

      if (c != 'c')
      {
         break;
      }

      // Estimate element errors using the Zienkiewicz-Zhu error estimator.
      Vector errors(pmesh.GetNE());
      CPD.GetErrorEstimates(errors);

      double local_max_err = errors.Max();
      double global_max_err;
      MPI_Allreduce(&local_max_err, &global_max_err, 1,
                    MPI_DOUBLE, MPI_MAX, pmesh.GetComm());

      // Refine the elements whose error is larger than a fraction of the
      // maximum element error.
      const double frac = 0.3;
      double threshold = frac * global_max_err;
      if (mpi.Root()) { cout << "Refining ..." << endl; }
      {
         pmesh.RefineByError(errors, threshold);
      }

      // Update the magnetostatic solver to reflect the new state of the mesh.
      Update(H1FESpace, HCurlFESpace, HDivFESpace, L2FESpace, BCoef,
             rhoCoef, tempCoef, nueCoef, nuiCoef,
             size_h1, size_l2,
             density_offsets, temperature_offsets,
             density, temperature,
             BField, density_gf, temperature_gf, nue_gf, nui_gf);
      CPD.Update();

      if (pmesh.Nonconforming() && mpi.WorldSize() > 1 && false)
      {
         if (mpi.Root()) { cout << "Rebalancing ..." << endl; }
         pmesh.Rebalance();

         // Update again after rebalancing
         Update(H1FESpace, HCurlFESpace, HDivFESpace, L2FESpace, BCoef,
                rhoCoef, tempCoef, nueCoef, nuiCoef,
                size_h1, size_l2,
                density_offsets, temperature_offsets,
                density, temperature,
                BField, density_gf, temperature_gf, nue_gf, nui_gf);
         CPD.Update();
      }
   }

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      CPD.DisplayAnimationToGLVis();
   }

   for (int i=0; i<auxFields.Size(); i++)
   {
      delete auxFields[i];
   }

   return 0;
}

void AdaptInitialMesh(MPI_Session &mpi,
                      ParMesh &pmesh, ParFiniteElementSpace &err_fespace,
                      ParFiniteElementSpace & H1FESpace,
                      ParFiniteElementSpace & HCurlFESpace,
                      ParFiniteElementSpace & HDivFESpace,
                      ParFiniteElementSpace & L2FESpace,
                      VectorCoefficient & BCoef,
                      Coefficient & rhoCoef,
                      Coefficient & tempCoef,
                      Coefficient & nueCoef,
                      Coefficient & nuiCoef,
                      int & size_h1,
                      int & size_l2,
                      Array<int> & density_offsets,
                      Array<int> & temperature_offsets,
                      BlockVector & density,
                      BlockVector & temperature,
                      ParGridFunction & BField,
                      ParGridFunction & density_gf,
                      ParGridFunction & temperature_gf,
                      ParGridFunction & nue_gf,
                      ParGridFunction & nui_gf,
                      Coefficient &ReCoef,
                      Coefficient &ImCoef,
                      int p, double tol, int max_its, int max_dofs,
                      bool visualization)
{
   ConstantCoefficient zeroCoef(0.0);

   ParComplexGridFunction gf(&L2FESpace);

   ComplexLpErrorEstimator estimator(p, ReCoef, ImCoef, gf);

   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0);
   refiner.SetTotalErrorNormP(p);
   refiner.SetLocalErrorGoal(tol);

   vector<socketstream> sout(2);
   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 275, Wh = 250; // window size
   int offx = Ww + 3;

   for (int it = 0; it < max_its; it++)
   {
      HYPRE_Int global_dofs = L2FESpace.GlobalTrueVSize();
      if (mpi.Root())
      {
         cout << "\nAMR iteration " << it << endl;
         cout << "Number of L2 unknowns: " << global_dofs << endl;
      }

      gf.ProjectCoefficient(ReCoef, ImCoef);

      double l2_nrm = gf.ComputeL2Error(zeroCoef, zeroCoef);
      double l2_err = gf.ComputeL2Error(ReCoef, ImCoef);
      if (mpi.Root())
      {
         if (l2_nrm > 0.0)
         {
            cout << "Relative L2 Error: " << l2_err << " / " << l2_nrm
                 << " = " << l2_err / l2_nrm
                 << endl;
         }
         else
         {
            cout << "L2 Error: " << l2_err << endl;
         }
      }

      // 19. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         VisualizeField(sout[0], vishost, visport, gf.real(),
                        "Stix S Real",
                        Wx, Wy, Ww, Wh);
         Wx += offx;

         VisualizeField(sout[1], vishost, visport, gf.imag(),
                        "Stix S Imaginary",
                        Wx, Wy, Ww, Wh);
      }

      if (global_dofs > max_dofs)
      {
         if (mpi.Root())
         {
            cout << "Reached the maximum number of dofs. Stop." << endl;
         }
         break;
      }

      // 20. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.
      refiner.Apply(pmesh);
      if (refiner.Stop())
      {
         if (mpi.Root())
         {
            cout << "Stopping criterion satisfied. Stop." << endl;
         }
         break;
      }

      // 21. Update the finite element space (recalculate the number of DOFs,
      //     etc.) and create a grid function update matrix. Apply the matrix
      //     to any GridFunctions over the space. In this case, the update
      //     matrix is an interpolation matrix so the updated GridFunction will
      //     still represent the same function as before refinement.
      Update(H1FESpace, HCurlFESpace, HDivFESpace, L2FESpace,
             BCoef, rhoCoef, tempCoef, nueCoef, nuiCoef,
             size_h1, size_l2,
             density_offsets, temperature_offsets,
             density, temperature,
             BField, density_gf, temperature_gf, nue_gf, nui_gf);

      err_fespace.Update();
      gf.Update();

      // 22. Load balance the mesh, and update the space and solution. Currently
      //     available only for nonconforming meshes.
      if (pmesh.Nonconforming())
      {
         pmesh.Rebalance();

         // Update the space and the GridFunction. This time the update matrix
         // redistributes the GridFunction among the processors.
         Update(H1FESpace, HCurlFESpace, HDivFESpace, L2FESpace,
                BCoef, rhoCoef, tempCoef, nueCoef, nuiCoef,
                size_h1, size_l2,
                density_offsets, temperature_offsets,
                density, temperature,
                BField, density_gf, temperature_gf, nue_gf, nui_gf);

         err_fespace.Update();
         gf.Update();
      }

   }
   if (mpi.Root())
   {
      cout << endl;
   }
}

void Update(ParFiniteElementSpace & H1FESpace,
            ParFiniteElementSpace & HCurlFESpace,
            ParFiniteElementSpace & HDivFESpace,
            ParFiniteElementSpace & L2FESpace,
            VectorCoefficient & BCoef,
            Coefficient & rhoCoef,
            Coefficient & TCoef,
            Coefficient & nueCoef,
            Coefficient & nuiCoef,
            int & size_h1,
            int & size_l2,
            Array<int> & density_offsets,
            Array<int> & temperature_offsets,
            BlockVector & density,
            BlockVector & temperature,
            ParGridFunction & BField,
            ParGridFunction & density_gf,
            ParGridFunction & temperature_gf,
            ParGridFunction & nue_gf,
            ParGridFunction & nui_gf)
{
   H1FESpace.Update();
   HCurlFESpace.Update();
   HDivFESpace.Update();
   L2FESpace.Update();

   BField.Update();
   BField.ProjectCoefficient(BCoef);

   nue_gf.Update();
   nue_gf.ProjectCoefficient(nueCoef);
   nui_gf.Update();
   nui_gf.ProjectCoefficient(nuiCoef);

   size_l2 = L2FESpace.GetVSize();
   for (int i=1; i<density_offsets.Size(); i++)
   {
      density_offsets[i] = density_offsets[i - 1] + size_l2;
   }
   density.Update(density_offsets);
   for (int i=0; i<density_offsets.Size()-1; i++)
   {
      density_gf.MakeRef(&L2FESpace, density.GetBlock(i));
      density_gf.ProjectCoefficient(rhoCoef);
   }

   size_h1 = H1FESpace.GetVSize();
   for (int i=1; i<temperature_offsets.Size(); i++)
   {
      temperature_offsets[i] = temperature_offsets[i - 1] + size_h1;
   }
   temperature.Update(temperature_offsets);
   for (int i=0; i<temperature_offsets.Size()-1; i++)
   {
      temperature_gf.MakeRef(&H1FESpace, temperature.GetBlock(i));
      temperature_gf.ProjectCoefficient(TCoef);
   }
}

const char * banner[6] =
{
   R"(  _________ __   __       ________      ___________    ___ ___  )",
   R"( /   _____//  |_|__|__  __\_____  \  __| _/\______ \  /   |   \ )",
   R"( \_____  \\   __\  \  \/  //  ____/ / __ |  |    |  \/    ~    \)",
   R"( /        \|  | |  |>    </       \/ /_/ |  |    `   \    Y    /)",
   R"(/_______  /|__| |__/__/\_ \_______ \____ | /_______  /\___|_  / )",
   R"(        \/               \/       \/    \/         \/       \/  )"
};

// Print the stix2d_dh ascii logo to the given ostream
void display_banner(ostream & os)
{
   for (int i=0; i<6; i++)
   {
      os << banner[i] << endl;
   }
   os << endl
      << "* Thomas H. Stix was a pioneer in the use of radio frequency"
      << " waves to heat" << endl
      << "  terrestrial plasmas to solar temperatures. He made important"
      << " contributions" << endl
      << "  to experimental and theoretic plasma physics. In the Stix"
      << " applications, the" << endl
      << "  plasma dielectric for the wave equation is formulated using"
      << " the \"Stix\"" << endl
      << "  notation, \"S, D, P\"." << endl<< endl << flush;
}

void record_cmd_line(int argc, char *argv[])
{
   ofstream ofs("stix2d_dh_cmd.txt");

   for (int i=0; i<argc; i++)
   {
      ofs << argv[i] << " ";
      if (strcmp(argv[i], "-bm"     ) == 0 ||
          strcmp(argv[i], "-cm"     ) == 0 ||
          strcmp(argv[i], "-sm"     ) == 0 ||
          strcmp(argv[i], "-bpp"    ) == 0 ||
          strcmp(argv[i], "-dpp"    ) == 0 ||
          strcmp(argv[i], "-tpp"    ) == 0 ||
          strcmp(argv[i], "-nepp"   ) == 0 ||
          strcmp(argv[i], "-nipp"   ) == 0 ||
          strcmp(argv[i], "-B"      ) == 0 ||
          strcmp(argv[i], "-k-vec"  ) == 0 ||
          strcmp(argv[i], "-q"      ) == 0 ||
          strcmp(argv[i], "-jsrc"   ) == 0 ||
          strcmp(argv[i], "-rod"    ) == 0 ||
          strcmp(argv[i], "-slab"   ) == 0 ||
          strcmp(argv[i], "-sp"     ) == 0 ||
          strcmp(argv[i], "-abcs"   ) == 0 ||
          strcmp(argv[i], "-sbcs"   ) == 0 ||
          strcmp(argv[i], "-pecs"   ) == 0 ||
          strcmp(argv[i], "-dbcs-msa") == 0 ||
          strcmp(argv[i], "-dbcs-pw") == 0 ||
          strcmp(argv[i], "-dbcs1"  ) == 0 ||
          strcmp(argv[i], "-dbcs2"  ) == 0 ||
          strcmp(argv[i], "-dbcv1"  ) == 0 ||
          strcmp(argv[i], "-dbcv2"  ) == 0 ||
          strcmp(argv[i], "-nbcs"   ) == 0 ||
          strcmp(argv[i], "-nbcs1"  ) == 0 ||
          strcmp(argv[i], "-nbcs2"  ) == 0 ||
          strcmp(argv[i], "-nbcv"   ) == 0 ||
          strcmp(argv[i], "-nbcv1"  ) == 0 ||
          strcmp(argv[i], "-nbcv2"  ) == 0)
      {
         ofs << "'" << argv[++i] << "' ";
      }
   }
   ofs << endl << flush;
   ofs.close();
}

// The Impedance is an optional coefficient defined on boundary surfaces which
// can be used in conjunction with absorbing boundary conditions.
Coefficient *
SetupImpedanceCoefficient(const Mesh & mesh, const Array<int> & abcs)
{
   Coefficient * coef = NULL;

   if ( pw_eta_.Size() > 0 )
   {
      MFEM_VERIFY(pw_eta_.Size() == abcs.Size(),
                  "Each impedance value must be associated with exactly one "
                  "absorbing boundary surface.");

      pw_bdr_eta_.SetSize(mesh.bdr_attributes.Size());

      if ( abcs[0] == -1 )
      {
         pw_bdr_eta_ = pw_eta_[0];
      }
      else
      {
         pw_bdr_eta_ = 0.0;

         for (int i=0; i<pw_eta_.Size(); i++)
         {
            pw_bdr_eta_[abcs[i]-1] = pw_eta_[i];
         }
      }
      coef = new PWConstCoefficient(pw_bdr_eta_);
   }

   return coef;
}

void rod_current_source_r(const Vector &x, Vector &j)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   bool cmplx = rod_params_.Size() == 9;

   int o = 3 + (cmplx ? 3 : 0);

   double x0 = rod_params_(o+0);
   double y0 = rod_params_(o+1);
   double radius = rod_params_(o+2);

   double r2 = (x(0) - x0) * (x(0) - x0) + (x(1) - y0) * (x(1) - y0);

   if (r2 <= radius * radius)
   {
      j(0) = rod_params_(0);
      j(1) = rod_params_(1);
      j(2) = rod_params_(2);
   }
   // j *= height;
}

void rod_current_source_i(const Vector &x, Vector &j)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   bool cmplx = rod_params_.Size() == 9;

   int o = 3 + (cmplx ? 3 : 0);

   double x0 = rod_params_(o+0);
   double y0 = rod_params_(o+1);
   double radius = rod_params_(o+2);

   double r2 = (x(0) - x0) * (x(0) - x0) + (x(1) - y0) * (x(1) - y0);

   if (r2 <= radius * radius)
   {
      if (cmplx)
      {
         j(0) = rod_params_(3);
         j(1) = rod_params_(4);
         j(2) = rod_params_(5);
      }
   }
   // j *= height;
}

void slab_current_source_r(const Vector &x, Vector &j)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   bool cmplx = slab_params_.Size() == 10;

   int o = 3 + (cmplx ? 3 : 0);

   double x0 = slab_params_(o+0);
   double y0 = slab_params_(o+1);
   double dx = slab_params_(o+2);
   double dy = slab_params_(o+3);

   if (x[0] >= x0-0.5*dx && x[0] <= x0+0.5*dx &&
       x[1] >= y0-0.5*dy && x[1] <= y0+0.5*dy)
   {
      j(0) = slab_params_(0);
      j(1) = slab_params_(1);
      j(2) = slab_params_(2);
      if (slab_profile_ == 1)
      { j *= 0.5 * (1.0 + sin(M_PI*((2.0 * (x[1] - y0) + dy)/dy - 0.5))); }
   }
}

void slab_current_source_i(const Vector &x, Vector &j)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   bool cmplx = slab_params_.Size() == 10;

   int o = 3 + (cmplx ? 3 : 0);

   double x0 = slab_params_(o+0);
   double y0 = slab_params_(o+1);
   double dx = slab_params_(o+2);
   double dy = slab_params_(o+3);

   if (x[0] >= x0-0.5*dx && x[0] <= x0+0.5*dx &&
       x[1] >= y0-0.5*dy && x[1] <= y0+0.5*dy)
   {
      if (cmplx)
      {
         j(0) = slab_params_(3);
         j(1) = slab_params_(4);
         j(2) = slab_params_(5);
         if (slab_profile_ == 1)
         { j *= 0.5 * (1.0 + sin(M_PI*((2.0 * (x[1] - y0) + dy)/dy - 0.5))); }
      }
   }
}

void curve_current_source_r(const Vector &x, Vector &j)
{
   MFEM_ASSERT(x.Size() == 2, "current source requires 2D space (theta, phi).");

   j.SetSize(x.Size());
   j = 0.0;

   bool cmplx = curve_params_.Size() == 4;

   int o = 2 + (cmplx ? 2 : 0);

   double a = -0.50721437;
   double b = -0.00295982;
   double c = 2.41569303;

   double zmin = -0.196;
   double zmax = 0.1936;

   double r = (j_cyl_) ? sqrt(x[0] * x[0] + x[1] * x[1]) : x[0];
   double z = (j_cyl_) ? x[2] : x[1];

   double thickness = 0.05;
   double rsol = 0.034;
   double rmin = a*pow((z+b),2.0) + c + rsol;
   double rmax = a*pow((z+b),2.0) + c + rsol + thickness;
   double rmajor = 1.85;

   double theta = atan2(z, r-rmajor);

   if (r >= rmin && r <= rmax &&
       z >= zmin && z <= zmax)
   {
      if (!j_cyl_)
      {
         j(0) = -1.0*curve_params_(0)*sin(theta);
         j(1) = curve_params_(0)*cos(theta);
         j(2) = curve_params_(1);
         //if (slab_profile_ == 1)
         //{ j *= 0.5 * (1.0 + sin(M_PI*((2.0 * (x[1] - y0) + dy)/dy - 0.5))); }
      }
      else
      {
         double cosphi = x[0] / r;
         double sinphi = x[1] / r;

         double j_r   = -curve_params_(0)*sin(theta);
         double j_phi = curve_params_(1);
         double j_z   = curve_params_(0)*cos(theta);

         j(0) = j_r * cosphi - j_phi * sinphi;
         j(1) = j_r * sinphi - j_phi * cosphi;
         j(2) = j_z;
      }
   }
}

void curve_current_source_i(const Vector &x, Vector &j)
{
   MFEM_ASSERT(x.Size() == 2, "current source requires 2D space (theta, phi).");

   j.SetSize(x.Size());
   j = 0.0;

   bool cmplx = curve_params_.Size() == 4;

   if (!cmplx) { return; } // Return with Im(j) = 0

   int o = 2 + (cmplx ? 2 : 0);

   double a = -0.50721437;
   double b = -0.00295982;
   double c = 2.41569303;

   double zmin = -0.196;
   double zmax = 0.1936;

   double r = (j_cyl_) ? sqrt(x[0] * x[0] + x[1] * x[1]) : x[0];
   double z = (j_cyl_) ? x[2] : x[1];

   double thickness = 0.05;
   double rsol = 0.034;
   double rmin = a*pow((z+b),2.0) + c + rsol;
   double rmax = a*pow((z+b),2.0) + c + rsol + thickness;
   double rmajor = 1.85;

   double theta = atan2(z, r-rmajor);

   if (r >= rmin && r <= rmax &&
       z >= zmin && z <= zmax)
   {
      if (!j_cyl_)
      {
         j(0) = -1.0*curve_params_(2)*sin(theta);
         j(1) = curve_params_(2)*cos(theta);
         j(2) = curve_params_(3);
         //if (slab_profile_ == 1)
         //{ j *= 0.5 * (1.0 + sin(M_PI*((2.0 * (x[1] - y0) + dy)/dy - 0.5))); }
      }
      else
      {
         double cosphi = x[0] / r;
         double sinphi = x[1] / r;

         double j_r   = -curve_params_(2)*sin(theta);
         double j_phi = curve_params_(3);
         double j_z   = curve_params_(2)*cos(theta);

         j(0) = j_r * cosphi - j_phi * sinphi;
         j(1) = j_r * sinphi - j_phi * cosphi;
         j(2) = j_z;
      }
   }
}

void e_bc_r(const Vector &x, Vector &E)
{
   E.SetSize(3);
   E = 0.0;

}

void e_bc_i(const Vector &x, Vector &E)
{
   E.SetSize(3);
   E = 0.0;
}

ColdPlasmaPlaneWaveH::ColdPlasmaPlaneWaveH(char type,
                                           double omega,
                                           const Vector & B,
                                           const Vector & number,
                                           const Vector & charge,
                                           const Vector & mass,
                                           const Vector & temp,
                                           int nuprof,
                                           bool realPart)
   : VectorCoefficient(3),
     type_(type),
     realPart_(realPart),
     nuprof_(nuprof),
     omega_(omega),
     Bmag_(B.Norml2()),
     Jy_(0.0),
     xJ_(0.5),
     dx_(0.05),
     Lx_(1.0),
     kappa_(0.0),
     b_(B),
     bc_(3),
     bcc_(3),
     h_r_(3),
     h_i_(3),
     k_r_(3),
     k_i_(3),
     beta_r_(3),
     beta_i_(3),
     numbers_(number),
     charges_(charge),
     masses_(mass),
     temps_(temp)
{
   b_ *= 1.0 / Bmag_;

   {
      double bx = b_(0);
      double by = b_(1);
      double bz = b_(2);

      bc_(0) = by - bz;
      bc_(1) = bz - bx;
      bc_(2) = bx - by;

      bcc_(0) = by*by + bz*bz - bx*(by + bz);
      bcc_(1) = bz*bz + bx*bx - by*(bz + bx);
      bcc_(2) = bx*bx + by*by - bz*(bx + by);

      bc_  *= 1.0 / bc_.Norml2();
      bcc_ *= 1.0 / bcc_.Norml2();
   }

   beta_r_ = 0.0;
   beta_i_ = 0.0;

   double nue_ = 0;
   double nui_ = 0;

   S_ = S_cold_plasma(omega_, Bmag_, nue_, nui_, numbers_, charges_, masses_,
                      temps_,
                      nuprof_);
   D_ = D_cold_plasma(omega_, Bmag_, nue_, nui_, numbers_, charges_, masses_,
                      temps_,
                      nuprof_);
   P_ = P_cold_plasma(omega_, nue_, numbers_, charges_, masses_, temps_, nuprof_);

   switch (type_)
   {
      case 'L':
      {
         kappa_ = omega_ * sqrt(S_ - D_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), b_);
         k_i_.Set(kappa_.imag(), b_);

         complex<double> h = sqrt((S_ - D_) * (epsilon0_ / mu0_));

         h_r_.Set(-M_SQRT1_2 * h.real(), bcc_);
         h_r_.Add(-M_SQRT1_2 * h.imag(), bc_);
         h_i_.Set( M_SQRT1_2 * h.real(), bc_);
         h_i_.Add(-M_SQRT1_2 * h.imag(), bcc_);
      }
      break;
      case 'R':
      {
         kappa_ = omega_ * sqrt(S_ + D_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), b_);
         k_i_.Set(kappa_.imag(), b_);

         complex<double> h = sqrt((S_ + D_) * (epsilon0_ / mu0_));

         h_r_.Set(-M_SQRT1_2 * h.real(), bcc_);
         h_r_.Add( M_SQRT1_2 * h.imag(), bc_);
         h_i_.Set(-M_SQRT1_2 * h.real(), bc_);
         h_i_.Add(-M_SQRT1_2 * h.imag(), bcc_);
      }
      break;
      case 'O':
      {
         kappa_ = omega_ * sqrt(P_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), bc_);
         k_i_.Set(kappa_.imag(), bc_);

         complex<double> h = sqrt(P_ * (epsilon0_ / mu0_));

         h_r_.Set(h.real(), bcc_);
         h_i_.Set(h.imag(), bcc_);
      }
      break;
      case 'X':
      {
         kappa_ = omega_ * sqrt(S_ - D_ * D_ / S_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), bc_);
         k_i_.Set(kappa_.imag(), bc_);

         complex<double> h = S_ * sqrt(S_ - D_ * D_ / S_);
         h *= sqrt((epsilon0_ / mu0_) / (S_ * S_ + D_ * D_));

         h_r_.Set(-h.real(), b_);
         h_i_.Set(-h.imag(), b_);
      }
      break;
      case 'J':
         // MFEM_VERIFY(fabs(B_[2]) == Bmag_,
         //             "Current slab require a magnetic field in the z-direction.");
         break;
   }
}

void ColdPlasmaPlaneWaveH::Eval(Vector &V, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   V.SetSize(3);

   double x_data[3];
   Vector x(x_data, 3);
   T.Transform(ip, x);

   complex<double> i = complex<double>(0.0,1.0);

   switch (type_)
   {
      case 'L': // Left Circularly Polarized, propagating along B
      case 'R': // Right Circularly Polarized, propagating along B
      case 'O': // Ordinary wave propagating perpendicular to B
      case 'X': // eXtraordinary wave propagating perpendicular to B
      {
         complex<double> kx = 0.0;
         for (int d=0; d<3; d++)
         {
            kx += (k_r_[d] - beta_r_[d] + i * (k_i_[d] - beta_i_[d])) * x[d];
         }
         complex<double> phase = exp(i * kx);
         double phase_r = phase.real();
         double phase_i = phase.imag();

         if (realPart_)
         {
            for (int d=0; d<3; d++)
            {
               V[d] = h_r_[d] * phase_r - h_i_[d] * phase_i;
            }
         }
         else
         {
            for (int d=0; d<3; d++)
            {
               V[d] = h_r_[d] * phase_i + h_i_[d] * phase_r;
            }
         }
      }
      break;
      case 'J':  // Slab of current density perpendicular to propagation
      {
         if (true/* k_.Size() == 0 */)
         {
            complex<double> kE = omega_ * sqrt(S_ - D_ * D_ / S_) / c0_;

            complex<double> skL = sin(kE * Lx_);
            complex<double> E0 = i * Jy_ /
                                 (omega_ * epsilon0_ * skL *
                                  (S_ * S_ - D_ * D_));

            complex<double> Ex = i * D_ * E0;
            complex<double> Ey = S_ * E0;

            if (x[0] <= xJ_ - 0.5 * dx_)
            {
               complex<double> skLJ = sin(kE * (Lx_ - xJ_));
               complex<double> skd  = sin(kE * 0.5 * dx_);
               complex<double> skx  = sin(kE * x[0]);

               Ex *= -2.0 * skLJ * skd * skx;
               Ey *= -2.0 * skLJ * skd * skx;
            }
            else if (x[0] <= xJ_ + 0.5 * dx_)
            {
               complex<double> ck1  = cos(kE * (Lx_ - xJ_ - 0.5 * dx_));
               complex<double> ck2  = cos(kE * (xJ_ - 0.5 * dx_));
               complex<double> skx  = sin(kE * x[0]);
               complex<double> skLx = sin(kE * (Lx_ - x[0]));

               Ex *= skL - ck1 * skx - ck2 * skLx;
               Ey *= skL - ck1 * skx - ck2 * skLx;
            }
            else
            {
               complex<double> skJ  = sin(kE * xJ_);
               complex<double> skd  = sin(kE * 0.5 * dx_);
               complex<double> skLx = sin(kE * (Lx_ - x[0]));

               Ex *= -2.0 * skJ * skd * skLx;
               Ey *= -2.0 * skJ * skd * skLx;
            }

            if (realPart_)
            {
               V[0] = Ex.real();
               V[1] = Ey.real();
               V[2] = 0.0;
            }
            else
            {
               V[0] = Ex.imag();
               V[1] = Ey.imag();
               V[2] = 0.0;
            }
         }
         else
         {
            // General phase shift
            V = 0.0; // For now...
         }
      }
      break;
      case 'Z':
         V = 0.0;
         break;
   }
}

ColdPlasmaPlaneWaveE::ColdPlasmaPlaneWaveE(char type,
                                           double omega,
                                           const Vector & B,
                                           const Vector & number,
                                           const Vector & charge,
                                           const Vector & mass,
                                           const Vector & temp,
                                           int nuprof,
                                           bool realPart)
   : VectorCoefficient(3),
     type_(type),
     realPart_(realPart),
     nuprof_(nuprof),
     omega_(omega),
     Bmag_(B.Norml2()),
     Jy_(0.0),
     xJ_(0.5),
     dx_(0.05),
     Lx_(1.0),
     kappa_(0.0),
     b_(B),
     bc_(3),
     bcc_(3),
     e_r_(3),
     e_i_(3),
     k_r_(3),
     k_i_(3),
     beta_r_(3),
     beta_i_(3),
     numbers_(number),
     charges_(charge),
     masses_(mass),
     temps_(temp)
{
   b_ *= 1.0 / Bmag_;

   {
      double bx = b_(0);
      double by = b_(1);
      double bz = b_(2);

      bc_(0) = by - bz;
      bc_(1) = bz - bx;
      bc_(2) = bx - by;

      bcc_(0) = by*by + bz*bz - bx*(by + bz);
      bcc_(1) = bz*bz + bx*bx - by*(bz + bx);
      bcc_(2) = bx*bx + by*by - bz*(bx + by);

      bc_  *= 1.0 / bc_.Norml2();
      bcc_ *= 1.0 / bcc_.Norml2();
   }

   beta_r_ = 0.0;
   beta_i_ = 0.0;

   double nue_ = 0;
   double nui_ = 0;

   S_ = S_cold_plasma(omega_, Bmag_, nue_, nui_, numbers_, charges_, masses_,
                      temps_,
                      nuprof_);
   D_ = D_cold_plasma(omega_, Bmag_, nue_, nui_, numbers_, charges_, masses_,
                      temps_,
                      nuprof_);
   P_ = P_cold_plasma(omega_, nue_, numbers_, charges_, masses_, temps_, nuprof_);

   switch (type_)
   {
      case 'L':
      {
         kappa_ = omega_ * sqrt(S_ - D_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), b_);
         k_i_.Set(kappa_.imag(), b_);

         e_r_.Set(M_SQRT1_2, bc_);
         e_i_.Set(M_SQRT1_2, bcc_);
      }
      break;
      case 'R':
      {
         kappa_ = omega_ * sqrt(S_ + D_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), b_);
         k_i_.Set(kappa_.imag(), b_);

         e_r_.Set( M_SQRT1_2, bc_);
         e_i_.Set(-M_SQRT1_2, bcc_);
      }
      break;
      case 'O':
      {
         kappa_ = omega_ * sqrt(P_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), bc_);
         k_i_.Set(kappa_.imag(), bc_);

         e_r_.Set(1.0, b_);
         e_i_ = 0.0;
      }
      break;
      case 'X':
      {
         kappa_ = omega_ * sqrt(S_ - D_ * D_ / S_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), bc_);
         k_i_.Set(kappa_.imag(), bc_);

         complex<double> den = sqrt(S_ * S_ + D_ * D_);
         complex<double> ec  = D_ / den;
         complex<double> ecc = S_ / den;

         e_r_.Set(ecc.real(), bcc_);
         e_r_.Add(ec.imag(), bc_);
         e_i_.Set(-ec.real(), bc_);
         e_i_.Add(ecc.imag(), bcc_);
      }
      break;
      case 'J':
         // MFEM_VERIFY(fabs(B_[2]) == Bmag_,
         //           "Current slab require a magnetic field in the z-direction.");
         break;
   }
}

void ColdPlasmaPlaneWaveE::Eval(Vector &V, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   V.SetSize(3);

   double x_data[3];
   Vector x(x_data, 3);
   T.Transform(ip, x);

   complex<double> i = complex<double>(0.0,1.0);

   switch (type_)
   {
      case 'L': // Left Circularly Polarized, propagating along B
      case 'R': // Right Circularly Polarized, propagating along B
      case 'O': // Ordinary wave propagating perpendicular to B
      case 'X': // eXtraordinary wave propagating perpendicular to B
      {
         complex<double> kx = 0.0;
         for (int d=0; d<3; d++)
         {
            kx += (k_r_[d] - beta_r_[d] + i * (k_i_[d] - beta_i_[d])) * x[d];
         }
         complex<double> phase = exp(i * kx);
         double phase_r = phase.real();
         double phase_i = phase.imag();

         if (realPart_)
         {
            for (int d=0; d<3; d++)
            {
               V[d] = e_r_[d] * phase_r - e_i_[d] * phase_i;
            }
         }
         else
         {
            for (int d=0; d<3; d++)
            {
               V[d] = e_r_[d] * phase_i + e_i_[d] * phase_r;
            }
         }
      }
      break;
      case 'J':  // Slab of current density perpendicular to propagation
      {
         /*
          if (k_.Size() == 0)
               {
                  complex<double> kE = omega_ * sqrt(S_ - D_ * D_ / S_) / c0_;

                  complex<double> skL = sin(kE * Lx_);
                  complex<double> E0 = i * Jy_ /
                                       (omega_ * epsilon0_ * skL *
                                        (S_ * S_ - D_ * D_));

                  complex<double> Ex = i * D_ * E0;
                  complex<double> Ey = S_ * E0;

                  if (x[0] <= xJ_ - 0.5 * dx_)
                  {
                     complex<double> skLJ = sin(kE * (Lx_ - xJ_));
                     complex<double> skd  = sin(kE * 0.5 * dx_);
                     complex<double> skx  = sin(kE * x[0]);

                     Ex *= -2.0 * skLJ * skd * skx;
                     Ey *= -2.0 * skLJ * skd * skx;
                  }
                  else if (x[0] <= xJ_ + 0.5 * dx_)
                  {
                     complex<double> ck1  = cos(kE * (Lx_ - xJ_ - 0.5 * dx_));
                     complex<double> ck2  = cos(kE * (xJ_ - 0.5 * dx_));
                     complex<double> skx  = sin(kE * x[0]);
                     complex<double> skLx = sin(kE * (Lx_ - x[0]));

                     Ex *= skL - ck1 * skx - ck2 * skLx;
                     Ey *= skL - ck1 * skx - ck2 * skLx;
                  }
                  else
                  {
                     complex<double> skJ  = sin(kE * xJ_);
                     complex<double> skd  = sin(kE * 0.5 * dx_);
                     complex<double> skLx = sin(kE * (Lx_ - x[0]));

                     Ex *= -2.0 * skJ * skd * skLx;
                     Ey *= -2.0 * skJ * skd * skLx;
                  }

                  if (realPart_)
                  {
                     V[0] = Ex.real();
                     V[1] = Ey.real();
                     V[2] = 0.0;
                  }
                  else
                  {
                     V[0] = Ex.imag();
                     V[1] = Ey.imag();
                     V[2] = 0.0;
                  }
               }
               else
               {
                  // General phase shift
                  V = 0.0; // For now...
               }
         */
      }
      break;
      case 'Z':
         V = 0.0;
         break;
   }
}
