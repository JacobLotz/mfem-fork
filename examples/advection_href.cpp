
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;


void Prefine(FiniteElementSpace & fes_old,
             GridFunction &u, Coefficient &gf_ex, GridFunction &orders_gf,
             double min_thresh, double max_thresh);

void Hrefine(GridFunction &u, Coefficient &gf_ex, double min_thresh,
             double max_thresh);

void Hrefine2(GridFunction &u, Coefficient &gf_ex, double min_thresh,
              double max_thresh);

void Refine(Array<int> ref_actions, GridFunction &u, int depth_limit = 100);

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x, double);

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;

class FE_Evolution : public TimeDependentOperator
{
private:
   BilinearForm &M, &K;
   const Vector &b;
   Solver *M_prec;
   CGSolver M_solver;

   mutable Vector z;

public:
   FE_Evolution(BilinearForm &M_, BilinearForm &K_, const Vector &b_);
   void Update();
   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~FE_Evolution();
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int ref_levels = 2;
   int order = 1;
   double t_final = 10.0;
   double dt = 0.0005;
   bool visualization = true;
   int vis_steps = 5;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh0 = Mesh::MakeCartesian2D(64, 1, mfem::Element::QUADRILATERAL,false, 2,
                                      1);


   std::vector<Vector> translations = {Vector({2.0,0.0}), };


   Mesh mesh = Mesh::MakePeriodic(mesh0,
                                  mesh0.CreatePeriodicVertexMapping(translations));


   mesh.EnsureNCMesh();

   int dim = mesh.Dimension();

   ODESolver *ode_solver = new RK4Solver;


   mesh.GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);
   FiniteElementSpace fes_old(&mesh, &fec);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);

   BilinearForm m(&fes);
   BilinearForm k(&fes);
   m.AddDomainIntegrator(new MassIntegrator);
   constexpr double alpha = -1.0;
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
   k.AddInteriorFaceIntegrator(
      new NonconservativeDGTraceIntegrator(velocity, alpha));
   k.AddBdrFaceIntegrator(
      new NonconservativeDGTraceIntegrator(velocity, alpha));

   LinearForm b(&fes);
   b.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, alpha,-0.5));

   m.Assemble();
   int skip_zeros = 0;
   k.Assemble(skip_zeros);
   b.Assemble();
   m.Finalize();
   k.Finalize(skip_zeros);

   // 7. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction u(&fes);
   u0.SetTime(0.);
   u.ProjectCoefficient(u0);


   L2_FECollection orders_fec(0,dim);
   FiniteElementSpace orders_fes(&mesh,&orders_fec);
   GridFunction orders_gf(&orders_fes);
   for (int i = 0; i<mesh.GetNE(); i++) { orders_gf(i) = order; }

   socketstream sout;
   socketstream meshout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      meshout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << mesh << u;
         sout << flush;
         meshout.precision(precision);
         // meshout << "solution\n" << mesh << orders_gf;
         meshout << "mesh\n" << mesh ;
         meshout << flush;
         cin.get();
      }
   }


   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution adv(m, k, b);

   double t = 0.0;
   adv.SetTime(t);

   GridFunction gf_ex(&fes);
   FunctionCoefficient u_ex(u0_function);

   ode_solver->Init(adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);

      ode_solver->Step(u, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;
         u_ex.SetTime(t);
         // Prefine(fes_old,u,u_ex, orders_gf, 5e-5, 5e-4);
         Hrefine2(u,u_ex, 5e-5, 5e-4);
         mfem::out << "number of elements = " << mesh.GetNE() << endl;


         m.Update();
         m.Assemble();
         m.Finalize();
         k.Update();
         k.Assemble(skip_zeros);
         k.Finalize(skip_zeros);
         b.Update();
         b.Assemble();
         adv.Update();
         ode_solver->Init(adv);
         if (visualization)
         {
            GridFunction * pr_u = ProlongToMaxOrder(&u);
            sout << "solution\n" << mesh << *pr_u << flush;
            // meshout << "solution\n" << mesh << orders_gf << flush;
            meshout << "mesh\n" << mesh << flush;
         }
      }
   }

   // 10. Free the used memory.
   delete ode_solver;

   return 0;
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(BilinearForm &M_, BilinearForm &K_, const Vector &b_)
   : TimeDependentOperator(M_.Height()), M(M_), K(K_), b(b_), z(M_.Height())
{
   Array<int> ess_tdof_list;
   M_prec = new OperatorJacobiSmoother(M, ess_tdof_list);
   M_solver.SetPreconditioner(*M_prec);
   M_solver.SetOperator(M);
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void FE_Evolution::Update()
{
   height = M.Height();
   width = M.Width();
   z.SetSize(M.Height());

   Array<int> ess_tdof_list;
   delete M_prec;
   M_prec = new OperatorJacobiSmoother(M, ess_tdof_list);
   M_solver.SetPreconditioner(*M_prec);
   M_solver.SetOperator(M);
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}

FE_Evolution::~FE_Evolution()
{
   delete M_prec;
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   v.SetSize(2);
   v(0) = 1.;
   v(1) = 0.;
}

// Initial condition
double u0_function(const Vector &x, double t)
{
   // give x0, y0;
   double x0 = 0.5;
   // double y0 = 0.5;
   double w = 100.;
   double c = 1.;
   double ds = c*t;

   double xx = x(0) - ds;
   double yy = x(1) - ds;

   double tol = 1e-6;
   if (xx>= 2.0+tol || xx<= 0.0-tol)
   {
      xx -= (int)xx;
   }
   if (yy>= 1.0+tol || yy<= 0.0-tol)
   {
      yy -= (int)yy;
   }

   double dr2 = (xx-x0)*(xx-x0);
   return 1. + exp(-w*dr2);
}

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(const Vector &x)
{
   return 1.0;
}


void Prefine(FiniteElementSpace & fes_old,
             GridFunction &u, Coefficient &ex, GridFunction &orders_gf,
             double min_thresh, double max_thresh)
{
   // get element errors
   FiniteElementSpace * fes = u.FESpace();
   int ne = fes->GetMesh()->GetNE();
   Vector errors(ne);
   u.ComputeElementL2Errors(ex,errors);
   for (int i = 0; i<ne; i++)
   {
      double error = errors(i);
      int order = fes->GetElementOrder(i);
      if (error < min_thresh && order > 1)
      {
         fes->SetElementOrder(i,order-1);
      }
      else if (error > max_thresh && order < 2)
      {
         fes->SetElementOrder(i, order+1);
      }
      else
      {
         // do nothing
      }
   }

   fes->Update(false);

   PRefinementTransferOperator * T = new PRefinementTransferOperator(fes_old,*fes);

   GridFunction u_fine(fes);
   T->Mult(u,u_fine);

   // copy the orders to the old space
   for (int i = 0; i<ne; i++)
   {
      int order = fes->GetElementOrder(i);
      fes_old.SetElementOrder(i,order);
      orders_gf(i) = order;
   }
   fes_old.Update(false);

   delete T;

   // update old gridfuntion;
   u = u_fine;

}

void Hrefine2(GridFunction &u, Coefficient & ex_coeff, double min_thresh,
              double max_thresh)
{
   FiniteElementSpace * fes = u.FESpace();
   Mesh * mesh = fes->GetMesh();
   int ne = mesh->GetNE();
   Vector errors(ne);
   u.ComputeElementL2Errors(ex_coeff,errors);

   Array<int> actions(ne);
   for (int i = 0; i<ne; i++)
   {
      double error = errors(i);
      if (error > max_thresh)
      {
         actions[i] = 1;
      }
      else if (error < min_thresh)
      {
         actions[i] = -1;
      }
      else
      {
         actions[i] = 0;
      }
   }
   Refine(actions,u,1);

   // construct a list of possible ref actions
   // Array<int> actions(ne);
   // for (int i = 0; i<ne; i++)
   // {
   //    double error = errors(i);
   //    if (error > max_thresh && mesh->ncmesh->GetElementDepth(i) < 1)
   //    {
   //       actions[i] = 1;
   //    }
   //    else
   //    {
   //       actions[i] = 0;
   //    }
   // }

   // // list of possible dref actions
   // Array<int> derefactions(ne); derefactions = 0;
   // const Table & dref_table = mesh->ncmesh->GetDerefinementTable();
   // for (int i = 0; i<dref_table.Size(); i++)
   // {
   //    int size = dref_table.RowSize(i);
   //    const int * row = dref_table.GetRow(i);
   //    double error = 0.;
   //    for (int j = 0; j<size; j++)
   //    {
   //       error += errors[row[j]];
   //    }
   //    if (error < min_thresh)
   //    {
   //       for (int j = 0; j<size; j++)
   //       {
   //          actions[row[j]] += -1;
   //       }
   //    }
   // }

   // // now refine the elements that have score >0 and deref the elements that have score < 0
   // Array<Refinement> elements_to_refine;
   // for (int i = 0; i<ne; i++)
   // {
   //    if (actions[i] > 0)
   //    {
   //       elements_to_refine.Append(Refinement(i,0b01));
   //    }
   // }

   // mesh->GeneralRefinement(elements_to_refine);
   // fes->Update();
   // u.Update();

   // // map old actions to new mesh
   // Array<int> new_actions(mesh->GetNE());
   // if (mesh->GetLastOperation() == mesh->REFINE)
   // {
   //    const CoarseFineTransformations &tr = mesh->GetRefinementTransforms();
   //    Table coarse2fine;
   //    tr.MakeCoarseToFineTable(coarse2fine);
   //    new_actions = 1;
   //    for (int i = 0; i<coarse2fine.Size(); i++)
   //    {
   //       if (coarse2fine.RowSize(i) == 1)
   //       {
   //          int * el = coarse2fine.GetRow(i);
   //          new_actions[el[0]] = actions[i];
   //       }
   //    }
   // }
   // else
   // {
   //    new_actions = actions;
   // }

   // // create a dummy error vector
   // Vector new_errors(mesh->GetNE());
   // new_errors = infinity();
   // for (int i = 0; i< new_errors.Size(); i++)
   // {
   //    if (new_actions[i] < 0)
   //    {
   //       new_errors[i] = 0.;
   //    }
   // }

   // // any threshold would do here
   // mesh->DerefineByError(new_errors,min_thresh);

   // fes->Update();
   // u.Update();
}


void Refine(Array<int> ref_actions, GridFunction &u, int depth_limit)
{
   FiniteElementSpace * fes = u.FESpace();
   Mesh * mesh = fes->GetMesh();
   int ne = mesh->GetNE();


   //  ovewrite to no action if an element is marked for refinement but it exceeds the depth limit
   for (int i = 0; i<ne; i++)
   {
      int depth = mesh->ncmesh->GetElementDepth(i);
      if (depth >= depth_limit && ref_actions[i] == 1)
      {
         ref_actions[i] = 0;
      }
   }

   // current policy to map agent_actions to actions
   // 1. All elements that are marked for refinement are to perform the refinement
   // 2. All of the "siblings" (i) of a marked element for refinement are assigned action=max(0,agent_actions[i])
   //   i.e., a) if the action is to be refined then they are refined
   //         b) if the action is to be derefined or no action then they get no action
   // 3. If among the "siblings" there is no refinement action then the group is marked
   //    for derefinement if the majority (including a tie) of the siblings are marked for derefinement
   //    otherwise they are marked for no action
   //  h-refine:   action =  1
   //  h-derefine: action = -1
   //  do nothing: action =  0

   Array<int> actions(ne);
   Array<int> actions_marker(ne);
   actions_marker = 0;

   const Table & dref_table = mesh->ncmesh->GetDerefinementTable();

   for (int i = 0; i<dref_table.Size(); i++)
   {
      int n = dref_table.RowSize(i);
      const int * row = dref_table.GetRow(i);
      int sum_of_actions = 0;
      bool ref_flag = false;
      for (int j = 0; j<n; j++)
      {
         int action = ref_actions[row[j]];
         sum_of_actions+=action;
         if (action == 1)
         {
            ref_flag = true;
            break;
         }
      }
      if (ref_flag)
      {
         for (int j = 0; j<n; j++)
         {
            actions[row[j]] = max(0,ref_actions[row[j]]);
            actions_marker[row[j]] = 1;
         }
      }
      else
      {
         bool dref_flag = (2*abs(sum_of_actions) >= n) ? true : false;
         for (int j = 0; j<n; j++)
         {
            actions[row[j]] = (dref_flag) ? -1 : 0;
            actions_marker[row[j]] = 1;
         }
      }
   }

   for (int i = 0; i<ne; i++)
   {
      if (actions_marker[i] != 1)
      {
         if (ref_actions[i] == -1)
         {
            actions[i] = 0;
         }
         else
         {
            actions[i] = ref_actions[i];
         }
      }
   }

   // now the actions array holds feasible actions of -1,0,1
   Array<Refinement> refinements;
   for (int i = 0; i<ne; i++)
   {
      if (actions[i] == 1) {refinements.Append(Refinement(i,0b01));}
   }
   if (refinements.Size())
   {
      mesh->GeneralRefinement(refinements);
      fes->Update();
      u.Update();
      ne = mesh->GetNE();
   }
   // now the derefinements
   Array<int> new_actions(ne);
   if (refinements.Size())
   {
      new_actions = 1;
      const CoarseFineTransformations & tr = mesh->GetRefinementTransforms();
      Table coarse_to_fine;
      tr.MakeCoarseToFineTable(coarse_to_fine);
      for (int i = 0; i<coarse_to_fine.Size(); i++)
      {
         int n = coarse_to_fine.RowSize(i);
         if (n == 1)
         {
            int * row = coarse_to_fine.GetRow(i);
            new_actions[row[0]] = actions[i];
         }
      }
   }
   else
   {
      new_actions = actions;
   }

   Vector dummy_errors(ne);
   dummy_errors = 1.0;
   for (int i = 0; i<ne; i++)
   {
      if (new_actions[i] < 0)
      {
         dummy_errors[i] = 0.;
      }
   }
   mesh->DerefineByError(dummy_errors,0.5);

   fes->Update();
   u.Update();
}