"""
    BasisSimulator.jl

Differentiable CT simulator for inverse problems.

Built with Reactant/Enzyme compatibility from the ground up.
"""
module BasisSimulator

using LinearAlgebra
using Statistics
using Random
using FFTW
using DelimitedFiles
using Unitful
import XrayAttenuation as XA

# Re-export XrayAttenuation for convenience
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
# Forward Projection
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

# Ray marching projector (on-the-fly geometry, single compiled kernel, polychromatic support)
include("Forward/RayMarching.jl")

# =============================================================================
# Reconstruction
# =============================================================================

# Reconstruction kernels (filters)
include("Reconstruction/Kernels.jl")

# Beam hardening correction
include("Reconstruction/BeamHardeningCorrection.jl")

# FDK cone-beam reconstruction
include("Reconstruction/FDK.jl")

# Ray marching backprojection (on-the-fly geometry, single compiled kernel)
include("Reconstruction/RayMarchingBackproj.jl")

# =============================================================================
# Clinical Scanner Configurations
# =============================================================================

# Scanner specifications (GE, Siemens, Canon, etc.)
include("Scanners/Scanners.jl")

# =============================================================================
# Optimization (Gradients & Iterative Reconstruction)
# =============================================================================

# Loss functions
include("Optimization/Loss.jl")

# Gradient computation and optimization
include("Optimization/Gradients.jl")

# =============================================================================
# Unified High-Level API
# =============================================================================

# Main API: simulate_sinogram() and reconstruct()
include("API.jl")

end # module
