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

#ifndef MFEM_JIT_HPP
#define MFEM_JIT_HPP

#include "../../config/config.hpp"

#ifdef MFEM_USE_JIT

#include <cassert>
#include <functional> // for std::hash
#include  <unordered_map> // needed for the kernel maps

namespace mfem
{

struct Jit
{
   /// Initialize JIT, used in communication Mpi singleton.
   static int Init(int *argc, char ***argv);

   /// Finalize JIT, used in communication Mpi singleton.
   static void Finalize();

   /// Ask the JIT process to update the shared library.
   static void* DLOpen();

   /// Ask the JIT process to update the archive.
   static void AROpen(void* &handle);

   /// Ask the JIT process to compile and update the libraries.
   static int Compile(const uint64_t hash, const char *src, const char *symbol,
                      void *&handle);

   /// Ask the JIT process the address of the symbol.
   static void* DlSym(void* handle, const char* symbol);

   /// Kernel class
   template<typename kernel_t> struct Kernel
   {
      void *handle;
      kernel_t ker;

      /// \brief Kernel constructor
      Kernel(const size_t hash, const char *src, const char *symbol):
         handle(Jit::DLOpen()) // shared cache ?
      {
         if (!handle) { AROpen(handle); } // if no so, try to the ar
         if (!handle) { Compile(hash, src, symbol, handle); }
         auto Symbol = [&]() { return ker = (kernel_t) DlSym(handle, symbol); };
         // having a handle, we look for the kernel symbol
         if (!Symbol()) { Compile(hash, src, symbol, handle); Symbol(); }
         assert(handle); assert(ker);
      }

      /// Kernel launch
      template<typename... Args>
      void operator()(Args... args) { assert(ker); ker(args...); }

      //Kernel(const Kernel&) = default; // for std::unordered_map emplace
   };

   /// \brief Binary hash combine function
   template <typename T> static inline
   size_t Hash(const size_t &h, const T &arg) noexcept
   { return h ^ (std::hash<T> {}(arg) + 0x9e3779b9ull + (h<<6) + (h>>2));}

   /// \brief Ternary hash combine function
   template<typename T, typename... Args> static inline
   size_t Hash(const size_t &h, const T &arg, Args... args) noexcept
   { return Hash(Hash(h, arg), args...); }
};

} // namespace mfem

#endif // MFEM_USE_JIT

#endif // MFEM_JIT_HPP