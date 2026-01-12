# BasisSimulator.jl

Differentiable CT simulator for inverse problems.

## Status

**v0.2.0** - Clean rebuild in progress.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/MolloiLab/BasisSimulator.jl")
```

## Quick Start

```julia
using BasisSimulator

# Gammex 472 materials available
Ca_100  # 100 mg/ml calcium insert
I_10_0  # 10 mg/ml iodine insert

# Get material by symbol
mat = get_material(:Ca_50)

# Query attenuation via XrayAttenuation.jl
using Unitful
μ = XA.linear_attenuation_coeff(mat, 60u"keV")
```

## Architecture

Built for Reactant/Enzyme compatibility:
- Pure functional forward projection
- FDK reconstruction
- End-to-end differentiability

## License

MIT
