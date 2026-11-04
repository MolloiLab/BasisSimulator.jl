// Geant4's own photon cross sections, for BasisMonteCarlo.
//
// Geant4.jl does not wrap G4EmCalculator, so this file exposes the one call BasisMonteCarlo
// needs: the macroscopic cross section (1/length, Geant4 internal units) of a named gamma
// process in a material, computed by the same models the simulation samples from. It is
// compiled on first use against the Geant4 that Geant4.jl ships (see `src/cross_sections.jl`).
#include "G4EmCalculator.hh"
#include "G4Gamma.hh"
#include "G4Material.hh"

extern "C" double basismc_cross_section_per_volume(double energy, const char* process,
                                                   const G4Material* material) {
  G4EmCalculator calculator;
  return calculator.ComputeCrossSectionPerVolume(energy, G4Gamma::Gamma(), process, material);
}
