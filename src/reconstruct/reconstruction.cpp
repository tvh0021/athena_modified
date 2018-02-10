//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file reconstruction.cpp
//  \brief

// C/C++ headers
#include <sstream>
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "reconstruction.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"
#include "../eos/eos.hpp"

// constructor

Reconstruction::Reconstruction(MeshBlock *pmb, ParameterInput *pin)
{
  pmy_block_ = pmb;

  // read and set type of spatial reconstruction
  characteristic_reconstruction = false;
  std::string input_recon = pin->GetOrAddString("time","xorder","2");
  if (input_recon == "1") {
    xorder = 1;
  } else if (input_recon == "2") {
    xorder = 2;
  } else if (input_recon == "2c") {
    xorder = 2;
    characteristic_reconstruction = true;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in Reconstruction constructor" << std::endl
        << "xorder=" << input_recon << " not valid choice for reconstruction"<< std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  // Allocate memory for scratch arrays
//  int ncells1 = ((pmy_block->ie)-(pmy_block->is) + 1) + 2*(NGHOST);
//  dwl_.NewAthenaArray(NWAVE,ncells1);
//  dwr_.NewAthenaArray(NWAVE,ncells1);
//  dw2_.NewAthenaArray(NWAVE,ncells1);
//  dwm_.NewAthenaArray(NWAVE,ncells1);
//  wc_.NewAthenaArray(NWAVE,ncells1);
}

// destructor

Reconstruction::~Reconstruction()
{
//  dwl_.DeleteAthenaArray();
//  dwr_.DeleteAthenaArray();
//  dw2_.DeleteAthenaArray();
//  dwm_.DeleteAthenaArray();
//  wc_.DeleteAthenaArray();
}
