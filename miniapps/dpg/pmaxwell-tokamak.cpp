
// srun -n 256 ./pmaxwell-tokamak -o 3 -sc -rnum
// srun -n 448 ./pmaxwell-tokamak -o 4 -sc -rnum 11.0 -sigma 2.0 -paraview

// Description:
// This example code demonstrates the use of MFEM to define and solve
// the "ultraweak" (UW) DPG formulation for the Maxwell problem

//      ∇×(1/μ ∇×E) - (ω^2 ϵ + i ω σ) E = Ĵ ,   in Ω
//                E×n = E_0, on ∂Ω

// It solves the following kinds of problems
// 1) Known exact solutions with error convergence rates
//    a) A manufactured solution problem where E is a plane beam
// 2) Fichera "microwave" problem
// 3) PML problems
//    a) Generic PML problem with point source given by the load
//    b) Plane wave scattering from a square
//    c) PML problem with a point source prescribed on the boundary

//
// The DPG UW deals with the First Order System
//        i ω μ H + ∇ × E = 0,   in Ω (Faraday's law)
// -(i ω ϵ + σ) E + ∇ × H = J,   in Ω (Ampere's law)
//            E × n = E_0, on ∂Ω
// Note: Ĵ = -iωJ

// The ultraweak-DPG formulation is obtained by integration by parts of both
// equations and the introduction of trace unknowns on the mesh skeleton

// in 3D
// E,H ∈ (L^2(Ω))^3
// Ê ∈ H_0^1/2(Ω)(curl, Γ_h), Ĥ ∈ H^-1/2(curl, Γ_h)
//  i ω μ (H,F) + (E,∇ × F) + < Ê, F × n > = 0,      ∀ F ∈ H(curl,Ω)
// -i ω ϵ (E,G) + (H,∇ × G) + < Ĥ, G × n > = (J,G)   ∀ G ∈ H(curl,Ω)
//                                   Ê × n = E_0     on ∂Ω
// -------------------------------------------------------------------------
// |   |       E      |      H      |      Ê       |       Ĥ      |  RHS    |
// -------------------------------------------------------------------------
// | F |  (E,∇ × F)   | i ω μ (H,F) | < n × Ê, F > |              |         |
// |   |              |             |              |              |         |
// | G |-(iωϵ+σ)(E,G) |  (H,∇ × G)  |              | < n × Ĥ, G > |  (J,G)  |
// where (F,G) ∈  H(curl,Ω) × H(curl,Ω)

// Here we use the "Adjoint Graph" norm on the test space i.e.,
// ||(F,G)||^2_V = ||A^*(F,G)||^2 + ||(F,G)||^2 where A is the
// maxwell operator defined by (1)


#include "mfem.hpp"
#include "util/pcomplexweakform.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


int dim;
double omega;
double mu = 1.0;
double epsilon = 1.0;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // const char *mesh_file = "tokamak_100k.msh";
   // const char *mesh_file = "tokamak_200k.msh";
   const char *mesh_file = "meshes/tokamak_300k.msh";
   int order = 1;
   int delta_order = 1;
   bool visualization = false;
   double rnum=1.0;
   bool static_cond = false;
   int sr = 0;
   int pr = 0;
   bool exact_known = false;
   bool paraview = false;
   double sigma = 1.0;


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&rnum, "-rnum", "--number_of_wavelenths",
                  "Number of wavelengths");
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&epsilon, "-eps", "--permittivity",
                  "Permittivity of free space (or mass constant).");
   args.AddOption(&sigma, "-sigma", "--sigma",
                  "conductivity");
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&sr, "-sref", "--serial_ref",
                  "Number of serial refinements.");
   args.AddOption(&pr, "-pref", "--parallel_ref",
                  "Number of parallel refinements.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   omega = 2.*M_PI*rnum;

   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();

   for (int i = 0; i<sr; i++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Define spaces
   // L2 space for E
   FiniteElementCollection *E_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *E_fes = new ParFiniteElementSpace(&pmesh,E_fec,dim);

   // Vector L2 space for H
   FiniteElementCollection *H_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *H_fes = new ParFiniteElementSpace(&pmesh,H_fec, dim);

   // H^-1/2 (curl) space for Ê
   FiniteElementCollection * hatE_fec = nullptr;
   FiniteElementCollection * hatH_fec = nullptr;
   FiniteElementCollection * F_fec = nullptr;
   int test_order = order+delta_order;
   hatE_fec = new ND_Trace_FECollection(order,dim);
   hatH_fec = new ND_Trace_FECollection(order,dim);
   F_fec = new ND_FECollection(test_order, dim);

   ParFiniteElementSpace *hatE_fes = new ParFiniteElementSpace(&pmesh,hatE_fec);
   ParFiniteElementSpace *hatH_fes = new ParFiniteElementSpace(&pmesh,hatH_fec);
   FiniteElementCollection * G_fec = new ND_FECollection(test_order, dim);

   Array<ParFiniteElementSpace * > trial_fes;
   Array<FiniteElementCollection * > test_fec;
   trial_fes.Append(E_fes);
   trial_fes.Append(H_fes);
   trial_fes.Append(hatE_fes);
   trial_fes.Append(hatH_fes);
   test_fec.Append(F_fec);
   test_fec.Append(G_fec);

   // Bilinear form coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient eps2omeg2(epsilon*epsilon*omega*omega);
   ConstantCoefficient mu2omeg2(mu*mu*omega*omega);
   ConstantCoefficient muomeg(mu*omega);
   ConstantCoefficient negepsomeg(-epsilon*omega);
   ConstantCoefficient epsomeg(epsilon*omega);
   ConstantCoefficient negmuomeg(-mu*omega);
   ConstantCoefficient sigma_cf(sigma);
   ConstantCoefficient negsigma_cf(-sigma);
   ConstantCoefficient sigma2_cf(sigma*sigma);

   Vector z_one(3); z_one = 0.0; z_one(2) = 1.0;
   Vector zero(3); zero = 0.0;
   Vector z_negone(3); z_negone = 0.0; z_negone(2) = -1.0;
   VectorConstantCoefficient z_one_cf(z_one);
   VectorConstantCoefficient z_negone_cf(z_negone);
   VectorConstantCoefficient zero_cf(zero);

   if (myid == 0)
   {
      std::cout << "Assembling matrix" << endl;
   }


   ParComplexDPGWeakForm * a = new ParComplexDPGWeakForm(trial_fes,test_fec);

   // (E,∇ × F)
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one)),
                         nullptr,0,0);
   // -(i ω ϵ + σ) (E , G) = i (- ω ϵ E, G) - (σ E, G)
   a->AddTrialIntegrator(
      new TransposeIntegrator(new VectorFEMassIntegrator(negsigma_cf)),
      new TransposeIntegrator(new VectorFEMassIntegrator(negepsomeg)),0,1);
   //  (H,∇ × G)
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one)),
                         nullptr,1,1);
   // < n×Ĥ ,G>
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,3,1);
   // test integrators
   // (∇×G ,∇× δG)
   a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,1,1);
   // (G,δG)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,1,1);

   // i ω μ (H, F)
   a->AddTrialIntegrator(nullptr,
                         new TransposeIntegrator(new VectorFEMassIntegrator(muomeg)),1,0);
   // < n×Ê,F>
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,2,0);
   // test integrators
   // (∇×F,∇×δF)
   a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,0,0);
   // (F,δF)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,0,0);

   // μ^2 ω^2 (F,δF)
   a->AddTestIntegrator(new VectorFEMassIntegrator(mu2omeg2),nullptr,0,0);
   // -i ω μ (F,∇ × δG) = i (F, -ω μ ∇ × δ G)
   a->AddTestIntegrator(nullptr,new MixedVectorWeakCurlIntegrator(negmuomeg),0,1);
   // -i ω ϵ (∇ × F, δG) - σ (∇ × F, δG)
   a->AddTestIntegrator(new MixedVectorCurlIntegrator(negsigma_cf),
                        new MixedVectorCurlIntegrator(negepsomeg),0,1);
   // i ω μ (∇ × G,δF)
   a->AddTestIntegrator(nullptr,new MixedVectorCurlIntegrator(muomeg),1,0);
   // i ω ϵ (G, ∇ × δF ) - σ (G, ∇ × δF )
   a->AddTestIntegrator(new MixedVectorWeakCurlIntegrator(negsigma_cf),
                        new MixedVectorWeakCurlIntegrator(epsomeg),1,0);
   // ϵ^2 ω^2 (G,δG)
   a->AddTestIntegrator(new VectorFEMassIntegrator(eps2omeg2),nullptr,1,1);
   // σ^2(G,δG)
   a->AddTestIntegrator(new VectorFEMassIntegrator(sigma2_cf),nullptr,1,1);

   socketstream E_out_r;
   socketstream H_out_r;

   ParComplexGridFunction E(E_fes);
   ParComplexGridFunction H(H_fes);
   E.real() = 0.0;
   E.imag() = 0.0;
   H.real() = 0.0;
   H.imag() = 0.0;

   ParaViewDataCollection * paraview_dc = nullptr;

   if (paraview)
   {
      paraview_dc = new ParaViewDataCollection(mesh_file, &pmesh);
      paraview_dc->SetPrefixPath("ParaView");
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetCycle(0);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetTime(0.0); // set the time
      paraview_dc->RegisterField("E_r",&E.real());
      paraview_dc->RegisterField("E_i",&E.imag());
      paraview_dc->RegisterField("H_r",&H.real());
      paraview_dc->RegisterField("H_i",&H.imag());
      paraview_dc->Save();
   }

   // internal bdr attributes
   Array<int> internal_bdr({1, 3, 6, 9, 17, 157, 185, 75, 210, 211,
                            212, 213, 214, 215, 216, 217, 218, 219,
                            220, 221, 222, 223, 224, 225, 226, 227,
                            228, 229, 230, 231, 232, 233, 234, 125});

   for (int it = 0; it<=pr; it++)
   {
      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      Array<int> one_bdr;
      Array<int> negone_bdr;


      if (myid == 0)
      {
         std::cout << "Attributes" << endl;
      }

      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         one_bdr.SetSize(pmesh.bdr_attributes.Max());
         negone_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         // need to exclude these attributes
         for (int i = 0; i<internal_bdr.Size(); i++)
         {
            ess_bdr[internal_bdr[i]-1] = 0;
         }
         hatE_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
         one_bdr = 0;
         negone_bdr = 0;
         one_bdr[234] = 1;
         negone_bdr[235] = 1;
      }

      if (myid == 0)
      {
         std::cout << "Attributes 2" << endl;
      }

      // Set up bdr conditions
      // shift the ess_tdofs
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         ess_tdof_list[j] += E_fes->GetTrueVSize() + H_fes->GetTrueVSize();
      }


      Array<int> offsets(5);
      offsets[0] = 0;
      offsets[1] = E_fes->GetVSize();
      offsets[2] = H_fes->GetVSize();
      offsets[3] = hatE_fes->GetVSize();
      offsets[4] = hatH_fes->GetVSize();
      offsets.PartialSum();

      Vector x(2*offsets.Last());
      x = 0.;
      double * xdata = x.GetData();

      ParComplexGridFunction hatE_gf(hatE_fes);
      hatE_gf.real().MakeRef(hatE_fes,&xdata[offsets[2]]);
      hatE_gf.imag().MakeRef(hatE_fes,&xdata[offsets.Last()+ offsets[2]]);

      hatE_gf.ProjectBdrCoefficientTangent(z_one_cf,zero_cf, one_bdr);
      hatE_gf.ProjectBdrCoefficientTangent(z_negone_cf,zero_cf, negone_bdr);





      if (myid == 0)
      {
         std::cout << "Assembly start" << endl;
      }

      if (static_cond) { a->EnableStaticCondensation(); }
      a->Assemble();

      if (myid == 0)
      {
         std::cout << "Assembly finished" << endl;
      }

      OperatorPtr Ah;
      Vector X,B;
      a->FormLinearSystem(ess_tdof_list,x,Ah, X,B);

      ComplexOperator * Ahc = Ah.As<ComplexOperator>();

      BlockOperator * BlockA_r = dynamic_cast<BlockOperator *>(&Ahc->real());
      BlockOperator * BlockA_i = dynamic_cast<BlockOperator *>(&Ahc->imag());

      int num_blocks = BlockA_r->NumRowBlocks();
      Array<int> tdof_offsets(2*num_blocks+1);

      tdof_offsets[0] = 0;
      int skip = (static_cond) ? 0 : 2;
      int k = (static_cond) ? 2 : 0;
      for (int i=0; i<num_blocks; i++)
      {
         tdof_offsets[i+1] = trial_fes[i+k]->GetTrueVSize();
         tdof_offsets[num_blocks+i+1] = trial_fes[i+k]->GetTrueVSize();
      }
      tdof_offsets.PartialSum();

      BlockOperator blockA(tdof_offsets);
      for (int i = 0; i<num_blocks; i++)
      {
         for (int j = 0; j<num_blocks; j++)
         {
            blockA.SetBlock(i,j,&BlockA_r->GetBlock(i,j));
            blockA.SetBlock(i,j+num_blocks,&BlockA_i->GetBlock(i,j), -1.0);
            blockA.SetBlock(i+num_blocks,j+num_blocks,&BlockA_r->GetBlock(i,j));
            blockA.SetBlock(i+num_blocks,j,&BlockA_i->GetBlock(i,j));
         }
      }

      X = 0.;
      BlockDiagonalPreconditioner M(tdof_offsets);

      if (!static_cond)
      {
         HypreBoomerAMG * solver_E = new HypreBoomerAMG((HypreParMatrix &)
                                                        BlockA_r->GetBlock(0,0));
         solver_E->SetPrintLevel(0);
         solver_E->SetSystemsOptions(dim);
         HypreBoomerAMG * solver_H = new HypreBoomerAMG((HypreParMatrix &)
                                                        BlockA_r->GetBlock(1,1));
         solver_H->SetPrintLevel(0);
         solver_H->SetSystemsOptions(dim);
         M.SetDiagonalBlock(0,solver_E);
         M.SetDiagonalBlock(1,solver_H);
         M.SetDiagonalBlock(num_blocks,solver_E);
         M.SetDiagonalBlock(num_blocks+1,solver_H);
      }


      HypreAMS * solver_hatE = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(skip,
                                                                                 skip), hatE_fes);
      HypreAMS * solver_hatH = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(
                                               skip+1,skip+1), hatH_fes);
      solver_hatE->SetPrintLevel(0);
      solver_hatH->SetPrintLevel(0);

      M.SetDiagonalBlock(skip,solver_hatE);
      M.SetDiagonalBlock(skip+1,solver_hatH);
      M.SetDiagonalBlock(skip+num_blocks,solver_hatE);
      M.SetDiagonalBlock(skip+num_blocks+1,solver_hatH);

      if (myid == 0)
      {
         std::cout << "PCG iterations" << endl;
      }


      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-5);
      cg.SetMaxIter(10000);
      cg.SetPrintLevel(1);
      cg.SetPreconditioner(M);
      cg.SetOperator(blockA);
      cg.Mult(B, X);

      for (int i = 0; i<num_blocks; i++)
      {
         delete &M.GetDiagonalBlock(i);
      }

      int num_iter = cg.GetNumIterations();

      a->RecoverFEMSolution(X,x);

      E.real().MakeRef(E_fes,x.GetData());
      E.imag().MakeRef(E_fes,&x.GetData()[offsets.Last()]);

      H.real().MakeRef(H_fes,&x.GetData()[offsets[1]]);
      H.imag().MakeRef(H_fes,&x.GetData()[offsets.Last()+offsets[1]]);

      int dofs = 0;
      for (int i = 0; i<trial_fes.Size(); i++)
      {
         dofs += trial_fes[i]->GlobalTrueVSize();
      }

      if (visualization)
      {
         const char * keys = (it == 0 && dim == 2) ? "jRcml\n" : nullptr;
         char vishost[] = "localhost";
         int  visport   = 19916;
         common::VisualizeField(E_out_r,vishost, visport, E.real(),
                                "Numerical Electric field (real part)", 0, 0, 500, 500, keys);
         common::VisualizeField(H_out_r,vishost, visport, H.real(),
                                "Numerical Magnetic field (real part)", 501, 0, 500, 500, keys);
      }

      if (paraview)
      {
         paraview_dc->SetCycle(it);
         paraview_dc->SetTime((double)it);
         paraview_dc->Save();
      }

      if (it == pr)
      {
         break;
      }

      pmesh.UniformRefinement();

      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();
   }

   if (paraview)
   {
      delete paraview_dc;
   }

   delete a;
   delete F_fec;
   delete G_fec;
   delete hatH_fes;
   delete hatH_fec;
   delete hatE_fes;
   delete hatE_fec;
   delete H_fec;
   delete E_fec;
   delete H_fes;
   delete E_fes;

   return 0;
}
