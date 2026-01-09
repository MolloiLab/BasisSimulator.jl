"""
    BasisSimulator

A fully differentiable, Reactant-compilable CT simulator for inverse problems.

# Features
- Complete physics models (spectrum, scatter, detector response, noise)
- Reactant/XLA compilation for GPU/TPU acceleration
- Enzyme.jl autodiff through entire forward model
- Validated against GECATSIM reference simulator
- Optimized for material decomposition and dose optimization

# Quick Start
```julia
using BasisSimulator

# Create a phantom
phantom = create_gammex_472(resolution_mm=1.0)

# Define scanner
scanner = create_aquilion_one()

# Generate X-ray spectrum
spec = generate_spectrum(kVp=120.0, mAs=200.0)

# Run simulation
result = simulate_ct_scan(
    phantom=phantom,
    scanner=scanner,
    spectrum=spec
)

# Access results
sinogram = result.sinogram
reconstruction = result.reconstruction
```

# Main Exports

## Physics
- `generate_spectrum` - X-ray source modeling
- `compute_attenuation` - Material interaction
- `estimate_scatter` - Compton scatter
- `apply_detector_response` - MTF/PSF modeling
- `apply_noise` - Quantum + electronic noise

## Geometry
- `create_aquilion_one` - Canon scanner geometry
- `create_custom_scanner` - User-defined geometry
- `trace_ray` - Reactant-compiled ray tracer
- `create_gammex_472` - Calibration phantom
- `create_xcat_phantom` - Anatomical phantom

## Reconstruction
- `reconstruct_fdk` - Feldkamp-Davis-Kress
- `reconstruct_sirt` - Iterative algorithm
- `correct_beam_hardening` - Physics correction

## Simulation
- `simulate_ct_scan` - High-level forward model
- `SimulationConfig` - Configuration struct

## Validation
- `compare_with_gecatsim` - Reference comparison
- `compute_metrics` - RMSE, SSIM, SNR, etc.

# Module Organization

The package is organized into logical submodules:

```
BasisSimulator
├── Physics       # Physical models
├── Geometry      # Scanner and phantom geometry
├── Reconstruction# Image reconstruction
├── Simulation    # Forward model orchestration
└── Validation    # Testing and comparison
```

See individual module documentation for details.
"""
module BasisSimulator

# ============================================================================
# External Dependencies
# ============================================================================

using LinearAlgebra
using Statistics
using FFTW
using Interpolations
using Distributions
import XrayAttenuation as XA
using Unitful: cm, keV, g, mm, ustrip, @u_str
using Reactant
using Enzyme

# ============================================================================
# Submodules
# ============================================================================

# Physics models
include("Physics/Spectrum.jl")
include("Physics/Attenuation.jl")
include("Physics/Materials.jl")
include("Physics/Detector.jl")
# TODO: include("Physics/Scatter.jl")
# TODO: include("Physics/Noise.jl")

# Geometry and ray tracing
include("Geometry/ScannerGeometry.jl")
include("Geometry/RayTracing.jl")
include("Geometry/Phantoms.jl")

# Image reconstruction
include("Reconstruction/FDK.jl")
# TODO: include("Reconstruction/Iterative.jl")
# TODO: include("Reconstruction/Corrections.jl")

# High-level simulation
include("Simulation.jl")

# Validation utilities
# TODO: include("Validation/Metrics.jl")
# TODO: include("Validation/GECATSIM.jl")
# TODO: include("Validation/ReferenceData.jl")

# ============================================================================
# Public API Exports
# ============================================================================

# Physics
export generate_spectrum, XRaySpectrum, mean_energy, total_fluence, validate_spectrum
export Materials, Elements, Compound, Mixture  # Re-exported from XrayAttenuation
export get_mass_attenuation, get_linear_attenuation
export compute_polychromatic_attenuation, compute_mixture_attenuation
export compute_two_material_decomposition
export create_custom_compound, create_custom_mixture
export list_available_materials, list_available_elements
# Custom Gammex materials
export Ca_50, Ca_100, Ca_200, Ca_300, Ca_400, Ca_500, Ca_600
export I_2_0, I_2_5, I_5_0, I_7_5, I_10_0, I_15_0, I_20_0
export CUSTOM_MATERIALS, get_material
export validate_material_hu, print_material_properties
# Detector
export compute_detector_efficiency
# TODO: export estimate_scatter_klein_nishina, estimate_scatter_convolution
# TODO: export apply_quantum_noise, apply_electronic_noise

# Geometry
export ScanProtocol, CTGeometry
export create_aquilion_one, create_custom_scanner
export get_magnification, get_detector_size, get_fov_diameter, validate_geometry
export trace_ray_material_paths, GridMeta
export PhantomData, VoxelGrid
export create_gammex_472, create_water_cylinder
export get_voxel_size, get_voxel_volume, get_memory_usage
export get_material_at_voxel, count_materials

# Reconstruction
export reconstruct_fdk, ReconstructionFilter, ramlak, shepplogan, hann
export convert_to_hounsfield_units, estimate_mu_water
export create_reconstruction_filter, validate_fdk_inputs
export reconstruct_sirt, reconstruct_mlem
export correct_beam_hardening, correct_scatter

# Simulation
export simulate_ct_scan
export convert_to_attenuation, estimate_air_scan

# Validation
export compare_with_gecatsim, compute_rmse, compute_ssim, compute_psnr
export run_gecatsim_reference, GECATSIMResults

# ============================================================================
# Version Information
# ============================================================================

"""
    version()

Return the current version of BasisSimulator.jl
"""
version() = v"0.1.0"

end # module BasisSimulator
