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

# =============================================================================
# Forward Projection
# =============================================================================

# Forward projector (ray-driven, Siddon's method)
include("Forward/Projector.jl")

# Polychromatic simulation (energy-dependent attenuation)
include("Forward/Polychromatic.jl")

# Scatter simulation (analytic kernel)
include("Forward/Scatter.jl")

# Detector response and noise modeling
include("Forward/DetectorNoise.jl")

# Bowtie filter modeling
include("Forward/BowtieFilter.jl")

# =============================================================================
# Reconstruction
# =============================================================================

# FDK cone-beam reconstruction
include("Reconstruction/FDK.jl")

# =============================================================================
# Optimization (Gradients & Iterative Reconstruction)
# =============================================================================

# Loss functions
include("Optimization/Loss.jl")

# Gradient computation and optimization
include("Optimization/Gradients.jl")

end # module
