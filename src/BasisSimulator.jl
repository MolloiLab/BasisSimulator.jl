"""
    BasisSimulator.jl

CT simulation with backend-agnostic GPU/CPU execution via AcceleratedKernels.jl.

Core algorithms ported from TIGRE (CERN/TIGRE).
Works on: Metal (Apple), CUDA (NVIDIA), ROCm (AMD), or CPU.

# Quick Start
```julia
using BasisSimulator

# Create phantom and geometry
phantom = create_gammex_472(n_voxels=128, fov_cm=35.0, z_cm=4.0)
geom = create_aquilion_one(n_angles=180, n_rows=32, n_cols=256, fov_cm=35.0)

# Get μ at desired energy (v20.0-pivot: compute on demand)
μ = compute_μ(phantom, 60.0)  # 60 keV

# Forward projection
sinogram = siddon_forward_project(μ, geom)

# FDK reconstruction
recon = fdk_reconstruct(sinogram, geom, size(phantom.mask))
```
"""
module BasisSimulator

using LinearAlgebra
using Statistics
using Random
using DelimitedFiles
using Unitful
using FFTW
import AcceleratedKernels as AK
import XrayAttenuation as XA

# Re-export for convenience
export XA

# =============================================================================
# Physics
# =============================================================================

# Materials (Gammex 472)
include("physics/materials.jl")

# Spectrum loading (from .dat files)
include("physics/spectrum.jl")

# Attenuation coefficient computation
include("physics/attenuation.jl")

# PCCT detector physics: CdTe material constants, charge transport, fluorescence
# Reference: Koch-Mehrin 2020 (NIM-A 976:164241), Konrad 2025 (PMB 70:065004)
include("physics/pcct/cdte_constants.jl")
include("physics/pcct/charge_transport.jl")
include("physics/pcct/k_fluorescence.jl")
include("physics/pcct/charge_collection.jl")
include("physics/pcct/pileup_model.jl")

# =============================================================================
# Geometry
# =============================================================================

# Phantom generation with semantic masks
include("geometry/phantom.jl")

# CT scanner geometry
include("geometry/scanner.jl")


# =============================================================================
# Forward Projection - Core (TIGRE port via AcceleratedKernels.jl)
# =============================================================================

# Siddon forward projection (exact ray-voxel intersections)
# Reference: CERN/TIGRE/Common/CUDA/Siddon_projection.cu
include("forward/siddon.jl")

# =============================================================================
# Physics Effects (GPU-native via AcceleratedKernels.jl)
# =============================================================================

# Scatter simulation (spatial convolution)
include("forward/scatter.jl")

# Protocol definitions
include("forward/protocol.jl")

# Detector response and noise modeling
include("forward/detector_noise.jl")

# Detector absorption efficiency and DQE
include("forward/detector_efficiency.jl")

# Bowtie filter modeling
include("forward/bowtie_filter.jl")

# Flat (inherent) filter modeling
include("forward/flat_filter.jl")

# Finite focal spot modeling
include("forward/focal_spot.jl")

# Detector crosstalk modeling (electronic and optical)
include("forward/crosstalk.jl")

# Detector lag (afterglow) modeling
include("forward/detector_lag.jl")

# Detector fill factor modeling
include("forward/fill_factor.jl")


# =============================================================================
# CatSim-style Signal Processing (types without PhysicsConfig dependency)
# Must be loaded before PhysicsPipeline which references these types
# =============================================================================

# Heel effect (anode self-attenuation)
include("forward/heel_effect.jl")

# Data Acquisition System model (signal chain, noise, quantization)
include("forward/das_model.jl")

# Beam hardening correction (water-based polynomial)
include("forward/beam_hardening_correction.jl")

# =============================================================================
# Unified Physics Pipeline
# Depends on all physics effects above including CatSim-style
# =============================================================================

include("forward/physics_pipeline.jl")

# =============================================================================
# Calibration Pipeline (needs PhysicsConfig from PhysicsPipeline)
# =============================================================================

# Calibration pipeline: air scan, offset, gain correction, log transform
include("forward/calibration.jl")

# =============================================================================
# Forward Projection - Unified API (includes physics)
# =============================================================================

# Beer-Lambert spectral physics: I = Σ wₑ × exp(-∫μₑ dl)
# Unified API for mono/poly projection + optional physics effects
include("forward/polychromatic.jl")

# =============================================================================
# Reconstruction (TIGRE port via AcceleratedKernels.jl)
# Organized by clinical terminology: FBP, Hybrid IR, MBIR
# =============================================================================

# --- Core Components ---
# Voxel-driven backprojection
# Reference: CERN/TIGRE/Common/CUDA/voxel_backprojection.cu
include("reconstruction/core/backprojection.jl")

# FDK filtering (ramp filter, cosine weighting)
include("reconstruction/core/filtering.jl")

# --- FBP (Filtered Back-Projection) ---
# FDK reconstruction (filter + backproject)
include("reconstruction/fbp/fdk.jl")

# Helical (spiral) CT reconstruction
include("reconstruction/fbp/helical_recon.jl")

# --- Classic Iterative Reconstruction ---
# SIRT iterative reconstruction
# Reference: CERN/TIGRE/MATLAB/Algorithms/SIRT.m
include("reconstruction/ir/sirt.jl")

# CGLS iterative reconstruction
# Reference: CERN/TIGRE/MATLAB/Algorithms/CGLS.m
include("reconstruction/ir/cgls.jl")

# --- Regularization ---
# Total Variation regularization for iterative reconstruction
# Reference: Rudin-Osher-Fatemi (ROF) model
include("reconstruction/regularization/tv_regularization.jl")

# --- Statistical IR (PWLS core) ---
# Statistical Iterative Reconstruction (ASIR-style)
# Reference: GE ASIR/ASIR-V, Penalized Weighted Least Squares
include("reconstruction/statistical_ir.jl")

# --- Hybrid IR (TRUE Hybrid IR) ---
# Vendor-general Hybrid IR with PWLS refinement
# Reference: Geyer et al. 2015, Willemink & Noël 2019, SAFIRE clinical studies
include("reconstruction/hybrid_ir/hybrid_ir.jl")

# --- Model-Based IR ---
# Model-Based Iterative Reconstruction (TrueFidelity/ADMIRE/QIR-style)
# Reference: Thibault et al. Med Phys 2007, Siemens ADMIRE/QIR
include("reconstruction/mbir/mbir.jl")

# =============================================================================
# Clinical Scanner Configurations
# =============================================================================

# Scanner specifications (GE, Siemens, Canon, etc.)
include("scanners/scanners.jl")

# Helical protocol integration with scanner specs (must come after scanners.jl)
include("scanners/helical_protocols.jl")

# =============================================================================
# Dual-Energy CT (Spectral Imaging)
# =============================================================================

# Dual kVp (GSI) forward projection, material decomposition, VMI
include("dual_energy/dual_energy.jl")

# =============================================================================
# Photon-Counting CT (Spectral Imaging)
# =============================================================================

# Photon-counting detector model: energy binning, charge sharing, pile-up
include("forward/photon_counting.jl")

# Unified Detector Response Matrix (DRM) combining all energy-dependent physics
# Must be after photon_counting.jl (uses get_detector_material_properties, etc.)
include("physics/pcct/detector_response.jl")

# PCCT spectral imaging: native VMI, K-edge, effective Z, multi-material decomposition
include("forward/pcct_spectral.jl")

# =============================================================================
# Image Quality Metrics (AAPM TG-233)
# =============================================================================

# Modulation Transfer Function (MTF) - spatial resolution
include("metrics/mtf.jl")

# Noise Power Spectrum (NPS) - noise texture characterization
include("metrics/nps.jl")

# Point Spread Function (PSF) - spatial resolution in real space
include("metrics/psf.jl")

# =============================================================================
# High-Level Simulation Driver
# =============================================================================

include("simulation/options.jl")
include("simulation/driver.jl")

end # module
