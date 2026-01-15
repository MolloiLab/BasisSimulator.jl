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

# Forward projection
sinogram = siddon_forward_project(Float32.(phantom.μ), geom)

# FDK reconstruction
recon = fdk_reconstruct(sinogram, geom, size(phantom.μ))
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

# Helical (spiral) scanning mode
include("Geometry/Helical.jl")

# =============================================================================
# Forward Projection (TIGRE port via AcceleratedKernels.jl)
# =============================================================================

# Siddon forward projection (exact ray-voxel intersections)
# Reference: CERN/TIGRE/Common/CUDA/Siddon_projection.cu
include("Forward/Siddon.jl")

# =============================================================================
# Polychromatic Forward Projection (NOT from TIGRE)
# =============================================================================

# Beer-Lambert spectral physics: I = Σ wₑ × exp(-∫μₑ dl)
# TIGRE is monochromatic only - this is our own implementation
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

# =============================================================================
# Physics Effects (not from TIGRE, but useful)
# =============================================================================

# Scatter simulation (analytic kernel)
include("Forward/Scatter.jl")

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

# Flying focal spot modeling
include("Forward/FlyingFocalSpot.jl")

# =============================================================================
# Clinical Scanner Configurations
# =============================================================================

# Scanner specifications (GE, Siemens, Canon, etc.)
include("Scanners/Scanners.jl")

end # module
