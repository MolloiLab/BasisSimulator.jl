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
# Source (X-ray source + beam shaping)
# =============================================================================

# Spectrum loading (from .dat files)
include("source/spectrum.jl")

# =============================================================================
# Object (scanned object: phantoms, materials, attenuation)
# =============================================================================

# Materials (Gammex 472)
include("object/materials.jl")

# Attenuation coefficient computation
include("object/attenuation.jl")

# PCCT detector physics: CdTe material constants, charge transport, fluorescence
# Reference: Koch-Mehrin 2020 (NIM-A 976:164241), Konrad 2025 (PMB 70:065004)
include("detector/pcct/cdte_constants.jl")
include("detector/pcct/charge_transport.jl")
include("detector/pcct/k_fluorescence.jl")
include("detector/pcct/charge_collection.jl")
include("detector/pcct/pileup_model.jl")

# =============================================================================
# Geometry
# =============================================================================

# Phantom generation with semantic masks
include("object/phantom.jl")

# CT scanner geometry
include("geometry/scanner.jl")

# Affine transforms: phantom ↔ world ↔ recon coordinate mapping
include("geometry/affine.jl")


# =============================================================================
# Projection (ray tracing)
# =============================================================================

# Siddon forward projection (exact ray-voxel intersections)
# Reference: CERN/TIGRE/Common/CUDA/Siddon_projection.cu
include("projection/siddon.jl")

# =============================================================================
# Detector (ALL detector effects + noise)
# =============================================================================

# Scatter simulation (spatial convolution)
include("detector/scatter.jl")

# Protocol definitions
include("source/protocol.jl")


# Detector absorption efficiency and DQE
include("detector/detector_efficiency.jl")

# Bowtie filter modeling
include("source/bowtie_filter.jl")

# Flat (inherent) filter modeling
include("source/flat_filter.jl")

# Finite focal spot modeling
include("source/focal_spot.jl")

# Detector optical crosstalk modeling
include("detector/optical_crosstalk.jl")

# Detector lag (afterglow) modeling
include("detector/detector_lag.jl")

# Detector fill factor modeling
include("detector/fill_factor.jl")


# =============================================================================
# CatSim-style Signal Processing (types without PhysicsConfig dependency)
# Must be loaded before PhysicsPipeline which references these types
# =============================================================================

# Heel effect (anode self-attenuation)
include("source/heel_effect.jl")

# Beam hardening correction (water-based polynomial)
include("correction/beam_hardening_correction.jl")

# =============================================================================
# Unified Physics Pipeline
# Depends on all physics effects above including CatSim-style
# =============================================================================

include("detector/physics_pipeline.jl")

# =============================================================================
# Correction (post-detection corrections)
# =============================================================================

# Calibration pipeline: air scan, offset, gain correction, log transform
include("correction/calibration.jl")


# =============================================================================
# Projection - Unified API (includes physics)
# =============================================================================

# Beer-Lambert spectral physics: I = Σ wₑ × exp(-∫μₑ dl)
# Unified API for mono/poly projection + optional physics effects
include("projection/polychromatic.jl")

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
# Photon-Counting CT (Spectral Imaging)
# =============================================================================

# Photon-counting detector model: energy binning, charge sharing, pile-up
include("detector/photon_counting.jl")

# Unified Detector Response Matrix (DRM) combining all energy-dependent physics
# Must be after photon_counting.jl (uses get_detector_material_properties, etc.)
include("detector/pcct/detector_response.jl")

# PCCT spectral imaging: native VMI, K-edge, effective Z, multi-material decomposition
include("spectral/pcct_spectral.jl")

# Unified VMI pipeline (shared by dual-kVp and PCCT)
include("spectral/vmi.jl")

# =============================================================================
# API (top-level orchestration)
# =============================================================================

include("api/options.jl")
include("api/workspace.jl")
include("api/driver.jl")

end # module
