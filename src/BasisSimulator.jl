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
# Materials (Gammex 472)
# =============================================================================
include("Physics/Materials.jl")

end # module
