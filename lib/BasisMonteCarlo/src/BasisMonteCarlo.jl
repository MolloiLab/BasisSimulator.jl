"""
    BasisMonteCarlo

Monte Carlo detector physics for BasisSimulator.jl, built on Geant4 through Geant4.jl.

A photon of known energy is fired into a detector of known geometry and material; Geant4 follows
it and every particle it makes (photoelectric absorption, Compton and Rayleigh scattering,
K-fluorescence, the electrons) until each is absorbed or leaves. Per photon, BasisMonteCarlo records the
energy left in the detector, whether the photon interacted, and every particle that escaped.
From those event records come the quantities a CT simulator needs: interaction probability,
quantum efficiency, energy-absorption efficiency, the absorbed-energy spectrum, the Swank factor
and the escape fractions, each on a 1-keV grid if asked.

The public API:

- materials: [`DetectorMaterial`](@ref), the presets in [`DETECTOR_MATERIALS`](@ref),
  [`linear_attenuation`](@ref) (NIST XCOM), [`beer_lambert_efficiency`](@ref);
- Geant4's own cross sections: [`geant4_attenuation`](@ref);
- geometry and beam: [`SlabDetector`](@ref), [`PencilBeam`](@ref);
- physics: [`PhysicsSettings`](@ref) (fixed for the lifetime of the process, as in Geant4);
- running: [`simulate_monoenergetic`](@ref), [`record_tracks`](@ref);
- results: [`MonoenergeticRun`](@ref) and the metrics on it ([`interaction_probability`](@ref),
  [`quantum_efficiency`](@ref), [`energy_absorption_efficiency`](@ref), [`swank_factor`](@ref),
  [`photopeak_fraction`](@ref), [`absorbed_energy_spectrum`](@ref), [`escape_spectrum`](@ref)).
"""
module BasisMonteCarlo

using Geant4
using Geant4.SystemOfUnits: mm, cm, keV, eV, MeV, g, cm3, ns
import XrayAttenuation as XA
using Unitful: @u_str, ustrip
using Statistics: mean, var
using Random: Random, Xoshiro
using SpecialFunctions: erf
using Scratch: @get_scratch!
import Libdl
import Distributed

include("materials.jl")
include("geometry.jl")
include("physics.jl")
include("engine.jl")
include("results.jl")
include("cross_sections.jl")
include("parallel.jl")
include("dexels.jl")
include("arc.jl")
include("beam.jl")
include("phantom.jl")
include("scatter.jl")
include("forced.jl")
include("checks.jl")
include("basissimulator_v0_15_1.jl")

export DetectorMaterial, DETECTOR_MATERIALS, linear_attenuation, beer_lambert_efficiency,
    geant4_attenuation,
    k_edges
export AbstractDetector, SlabDetector, Collimator, Septa, PencilBeam, AreaBeam, FocalSpot
export ArcDetector, ArcReadout, ArcCounts, Spectrum, simulate_spectrum, arc_counts, focal_spot
export ArcEnergyReadout, ArcSignal, arc_signal
export air_mu_en, air_kerma, half_value_layer
export RayBeam, PhantomInsert, CylinderPhantom, material_lengths, water_equivalent_length, water_equivalent_diameter
export ScatterScanner, ScatterReadout, ScatterTally, scatter_tally, merge_tallies, column_edges, row_edges
export ForcedScatterReadout, VertexLog, forced_attenuation, scattering_tables, exit_length
export ScatterLibrary, kernel_ray, scatter_library, library_kernel, library_scan
export Check, check_relative, check_below, check_range, scorecard
export BASISSIMULATOR_V0_15_1_EFFICIENCY
export cell_pitch, cell_count, face_size_mm, fan_angle_deg, dexel_centre, dexel_edges
export start_workers, stop_workers, workers, simulate_sweep
export DexelEnergyReadout, DexelEnergy, dexel_energy, dexel_efficiency, dexel_swank
export DexelReadout, DexelResponse, dexel_response, dexel_beam, threshold_counts, dexel_spectrum, DepositLog,
    response_table
export PhysicsSettings
export simulate_monoenergetic, record_tracks, MonoenergeticRun, TrackRecord, TrackStep
export interaction_probability, quantum_efficiency, energy_absorption_efficiency,
    swank_factor, photopeak_fraction, absorbed_energy_spectrum, escape_spectrum, fluorescence_lines,
    energy_budget, standard_error, interpolate_linear

end
