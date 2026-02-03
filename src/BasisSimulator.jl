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
include("Physics/Materials.jl")

# Spectrum loading (from .dat files)
include("Physics/Spectrum.jl")

# Attenuation coefficient computation
include("Physics/Attenuation.jl")

# =============================================================================
# Geometry
# =============================================================================

# Phantom generation with semantic masks
include("Geometry/Phantom.jl")

# CT scanner geometry
include("Geometry/Scanner.jl")


# =============================================================================
# Forward Projection - Core (TIGRE port via AcceleratedKernels.jl)
# =============================================================================

# Siddon forward projection (exact ray-voxel intersections)
# Reference: CERN/TIGRE/Common/CUDA/Siddon_projection.cu
include("Forward/Siddon.jl")

# =============================================================================
# Physics Effects (GPU-native via AcceleratedKernels.jl)
# =============================================================================

# Scatter simulation (spatial convolution)
include("Forward/Scatter.jl")

# Protocol definitions
include("Forward/Protocol.jl")

# Detector response and noise modeling
include("Forward/DetectorNoise.jl")

# Detector absorption efficiency and DQE
include("Forward/DetectorEfficiency.jl")

# Bowtie filter modeling
include("Forward/BowtieFilter.jl")

# Flat (inherent) filter modeling
include("Forward/FlatFilter.jl")

# Finite focal spot modeling
include("Forward/FocalSpot.jl")

# Detector crosstalk modeling (electronic and optical)
include("Forward/Crosstalk.jl")

# Detector lag (afterglow) modeling
include("Forward/DetectorLag.jl")

# Detector fill factor modeling
include("Forward/FillFactor.jl")


# =============================================================================
# CatSim-style Signal Processing (types without PhysicsConfig dependency)
# Must be loaded before PhysicsPipeline which references these types
# =============================================================================

# Heel effect (anode self-attenuation)
include("Forward/HeelEffect.jl")

# Data Acquisition System model (signal chain, noise, quantization)
include("Forward/DASModel.jl")

# Beam hardening correction (water-based polynomial)
include("Forward/BeamHardeningCorrection.jl")

# =============================================================================
# Unified Physics Pipeline
# Depends on all physics effects above including CatSim-style
# =============================================================================

include("Forward/PhysicsPipeline.jl")

# =============================================================================
# Calibration Pipeline (needs PhysicsConfig from PhysicsPipeline)
# =============================================================================

# Calibration pipeline: air scan, offset, gain correction, log transform
include("Forward/Calibration.jl")

# =============================================================================
# Forward Projection - Unified API (includes physics)
# =============================================================================

# Beer-Lambert spectral physics: I = Σ wₑ × exp(-∫μₑ dl)
# Unified API for mono/poly projection + optional physics effects
include("Forward/Polychromatic.jl")

# =============================================================================
# Reconstruction (TIGRE port via AcceleratedKernels.jl)
# =============================================================================

# Voxel-driven backprojection
# Reference: CERN/TIGRE/Common/CUDA/voxel_backprojection.cu
include("Reconstruction/Backprojection.jl")

# FDK filtering (ramp filter, cosine weighting)
include("Reconstruction/Filtering.jl")

# FDK reconstruction (filter + backproject)
include("Reconstruction/FDK.jl")

# SIRT iterative reconstruction
# Reference: CERN/TIGRE/MATLAB/Algorithms/SIRT.m
include("Reconstruction/SIRT.jl")

# CGLS iterative reconstruction
# Reference: CERN/TIGRE/MATLAB/Algorithms/CGLS.m
include("Reconstruction/CGLS.jl")

# Total Variation regularization for iterative reconstruction
# Reference: Rudin-Osher-Fatemi (ROF) model
include("Reconstruction/TVRegularization.jl")

# Statistical Iterative Reconstruction (ASIR-style)
# Reference: GE ASIR/ASIR-V, Penalized Weighted Least Squares
include("Reconstruction/StatisticalIR.jl")

# Model-Based Iterative Reconstruction (TrueFidelity/ADMIRE/QIR-style)
# Reference: Thibault et al. Med Phys 2007, Siemens ADMIRE/QIR
include("Reconstruction/MBIR.jl")

# Helical (spiral) CT reconstruction
include("Reconstruction/HelicalRecon.jl")

# =============================================================================
# Clinical Scanner Configurations
# =============================================================================

# Scanner specifications (GE, Siemens, Canon, etc.)
include("Scanners/Scanners.jl")

# Helical protocol integration with scanner specs (must come after Scanners.jl)
include("Scanners/HelicalProtocols.jl")

# =============================================================================
# Dual-Energy CT (Spectral Imaging)
# =============================================================================

# Dual kVp (GSI) forward projection, material decomposition, VMI
include("DualEnergy/DualEnergy.jl")

# =============================================================================
# Photon-Counting CT (Spectral Imaging)
# =============================================================================

# Photon-counting detector model: energy binning, charge sharing, pile-up
include("Forward/PhotonCounting.jl")

# PCCT spectral imaging: native VMI, K-edge, effective Z, multi-material decomposition
include("Forward/PCCTSpectral.jl")

# =============================================================================
# Image Quality Metrics (AAPM TG-233)
# =============================================================================

# Modulation Transfer Function (MTF) - spatial resolution
include("Metrics/MTF.jl")

# Noise Power Spectrum (NPS) - noise texture characterization
include("Metrics/NPS.jl")

# Point Spread Function (PSF) - spatial resolution in real space
include("Metrics/PSF.jl")

# =============================================================================
# High-Level Simulation Driver
# =============================================================================

include("Simulation/Options.jl")
include("Simulation/Driver.jl")

end # module
