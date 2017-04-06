#ifndef MULTIGRID_HPP
#define MULTIGRID_HPP

//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gravity.hpp
//  \brief defines Gravity class which implements data and functions for gravitational potential

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../task_list/mg_task_list.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

class MeshBlock;
class ParameterInput;
class Coordinates;

typedef void (*MGBoundaryFunc_t)(AthenaArray<Real> &dst,Real time, Real dt,
             int nvar, int is, int ie, int js, int je, int ks, int ke, int ngh,
             Real x0, Real y0, Real z0, Real dx, Real dy, Real dz);

void MGPeriodicInnerX1(AthenaArray<Real> &dst,Real time, Real dt,
                    int nvar, int is, int ie, int js, int je, int ks, int ke, int ngh,
                    Real x0, Real y0, Real z0, Real dx, Real dy, Real dz);
void MGPeriodicOuterX1(AthenaArray<Real> &dst,Real time, Real dt,
                    int nvar, int is, int ie, int js, int je, int ks, int ke, int ngh,
                    Real x0, Real y0, Real z0, Real dx, Real dy, Real dz);
void MGPeriodicInnerX2(AthenaArray<Real> &dst,Real time, Real dt,
                    int nvar, int is, int ie, int js, int je, int ks, int ke, int ngh,
                    Real x0, Real y0, Real z0, Real dx, Real dy, Real dz);
void MGPeriodicOuterX2(AthenaArray<Real> &dst,Real time, Real dt,
                    int nvar, int is, int ie, int js, int je, int ks, int ke, int ngh,
                    Real x0, Real y0, Real z0, Real dx, Real dy, Real dz);
void MGPeriodicInnerX3(AthenaArray<Real> &dst,Real time, Real dt,
                    int nvar, int is, int ie, int js, int je, int ks, int ke, int ngh,
                    Real x0, Real y0, Real z0, Real dx, Real dy, Real dz);
void MGPeriodicOuterX3(AthenaArray<Real> &dst,Real time, Real dt,
                    int nvar, int is, int ie, int js, int je, int ks, int ke, int ngh,
                    Real x0, Real y0, Real z0, Real dx, Real dy, Real dz);

//! \class Multigrid
//  \brief gravitational block

class Multigrid {
public:
  Multigrid(MeshBlock *pmb, int invar, int nx, int ny, int nz,
            RegionSize isize, MGBoundaryFunc_t *MGBoundary);
  virtual ~Multigrid();

  const enum MGBoundaryType btype;

  void LoadFinestData(const AthenaArray<Real> &src, int ns, int ngh);
  void LoadSource(const AthenaArray<Real> &src, int ns, int ngh, Real fac);
  void RetrieveResult(AthenaArray<Real> &dst, int ns, int ngh);
  void ZeroClearData(void);
  void ApplyPhysicalBoundaries(void);
  void Restrict(void);
  void ProlongateAndCorrect(void);
  void FMGProlongate(void);
  void RestrictFMGSource(void);
  void SetFMGSource(void);
  Real CalculateDefectNorm(int n, int nrm);
  Real CalculateTotalSource(int n);
  void SubtractAverageSource(int type, int n, Real ave);

  // small functions
  int GetCurrentLevel(void) { return current_level_; };
  void SetCurrentLevel(int level) { current_level_=level; return; };
  int GetCurrentNumberOfCells(void) { return 1<<current_level_; };
  AthenaArray<Real>& GetCurrentData(void) { return u_[current_level_]; };
  AthenaArray<Real>& GetCurrentSource(void) { return src_[current_level_]; };
  Real GetRootSource(int n) { return src_[0].(n,ngh_,ngh_,ngh_); };

  // pure virtual functions
  virtual void Smooth(int color) = 0;
  virtual void CalculateDefect(void) = 0;

  friend class MultigridDriver;

protected:
  RegionSize size_;
  int nlevel_, nx_, ny_, nz_, ngh_, nvar_;
  int current_level_;
  Real rdx_, rdy_, rdz_;
  AthenaArray<Real> *u_, *def_, *src_, *fmgsrc_;

private:
  MeshBlock *pmy_block_;
  TaskState ts_;
  MGBoundaryFunc_t MGBoundaryFunction_[6];
};


//! \class MultigridDriver
//  \brief Multigrid driver

class MultigridDriver
{
public:
  MultigridDriver(Mesh *pm, MeshBlock *pmb, MGBoundaryFunc_t *MGBoundary,
                  ParameterInput *pin);
  virtual ~MultigridDriver();
  void SetupMultigrid(void);
  void CollectSource(void);
  void FillRootGrid(void);

  // small functions
  int GetNumMeshBlocks(void) { return nblocks_; };

  virtual Multigrid* GetMultigridBlock(MeshBlock *) = 0;
  virtual void LoadSourceAndData(void) = 0;
  virtual void SolveCoarsestGrid(void);

  friend class Multigrid;

protected:
  int nvar_, nblocks_, nrootlevel_, nmblevel_, ntotallevel_, fmglevel_;
  Mesh *pmy_mesh_;
  MeshBlock *pblock_;
  Multigrid *mgroot_;
private:
  MultigridTaskList *mgtlist_;
  MGBoundaryFunc_t MGBoundaryFunction_[6];
  int *nslist_, *nblist_;
  Real *rootbuf_;
  AthenaArray<Real> rootsrc_;
#ifdef MPI_PARALLEL
  MPI_Comm MPI_COMM_MULTIGRID;
#endif
};


#endif // MULTIGRID_HPP
