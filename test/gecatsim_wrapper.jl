"""
GECATSIM wrapper functions for cross-validation with BasisSimulator.

This module provides utilities to:
1. Create GECATSIM configurations matching BasisSimulator parameters
2. Run GECATSIM simulations
3. Convert between GECATSIM and BasisSimulator data formats

Usage:
    using PythonCall
    include("test/gecatsim_wrapper.jl")

    # Create water cylinder in GECATSIM
    cfg = create_gecatsim_water_cylinder(
        diameter_mm=100.0,
        height_mm=20.0,
        kVp=120.0,
        mAs=200.0
    )

    # Run simulation
    sinogram = run_gecatsim_simulation(cfg)
"""

using PythonCall
using BasisSimulator

# ==============================================================================
# GECATSIM Configuration Creation
# ==============================================================================

"""
    create_gecatsim_water_cylinder(;
        diameter_mm::Float64 = 100.0,
        height_mm::Float64 = 20.0,
        kVp::Float64 = 120.0,
        mAs::Float64 = 200.0,
        num_projections::Int = 360
    ) -> PythonCall.Py

Create a GECATSIM configuration for a water cylinder phantom.

This configuration matches the water cylinder phantom used in BasisSimulator
for validation purposes.

# Arguments
- `diameter_mm`: Cylinder diameter in mm
- `height_mm`: Cylinder height in mm
- `kVp`: X-ray tube peak voltage in kV
- `mAs`: Tube current-time product in mAs
- `num_projections`: Number of projection angles

# Returns
- GECATSIM configuration dictionary (Python dict)

# Example
```julia
cfg = create_gecatsim_water_cylinder(
    diameter_mm=100.0,
    height_mm=20.0,
    kVp=120.0,
    mAs=200.0
)
```
"""
function create_gecatsim_water_cylinder(;
    diameter_mm::Float64 = 100.0,
    height_mm::Float64 = 20.0,
    kVp::Float64 = 120.0,
    mAs::Float64 = 200.0,
    num_projections::Int = 360,
    detector_rows::Int = 320,
    detector_cols::Int = 800
)
    # Import GECATSIM
    gecatsim = pyimport("gecatsim")

    # Create configuration dictionary
    cfg = pydict()

    # Source configuration (X-ray tube)
    cfg["src"] = pydict()
    cfg["src"]["kVp"] = kVp
    cfg["src"]["mAs"] = mAs
    cfg["src"]["focal_spot"] = 0.6  # mm
    cfg["src"]["type"] = "rotating_anode"

    # Detector configuration
    cfg["det"] = pydict()
    cfg["det"]["rows"] = detector_rows
    cfg["det"]["cols"] = detector_cols
    cfg["det"]["pixel_width"] = 0.5  # mm (matches Canon Aquilion ONE ~0.5mm)
    cfg["det"]["pixel_height"] = 0.5  # mm

    # Scanner geometry
    cfg["scanner"] = pydict()
    cfg["scanner"]["SAD"] = 600.0  # mm (matches Canon Aquilion ONE)
    cfg["scanner"]["SDD"] = 1000.0  # mm

    # Protocol (scan parameters)
    cfg["protocol"] = pydict()
    cfg["protocol"]["num_projections"] = num_projections
    cfg["protocol"]["start_angle"] = 0.0
    cfg["protocol"]["end_angle"] = 360.0

    # Phantom configuration (water cylinder)
    cfg["phantom"] = pydict()
    cfg["phantom"]["type"] = "cylinder"
    cfg["phantom"]["material"] = "water"
    cfg["phantom"]["diameter"] = diameter_mm
    cfg["phantom"]["height"] = height_mm

    # Reconstruction configuration
    cfg["recon"] = pydict()
    cfg["recon"]["fov"] = diameter_mm * 1.2  # Slightly larger than phantom
    cfg["recon"]["image_size"] = 256

    return cfg
end

"""
    run_gecatsim_simulation(cfg::PythonCall.Py) -> Array{Float64, 3}

Run a GECATSIM simulation with the given configuration.

# Arguments
- `cfg`: GECATSIM configuration dictionary (from create_gecatsim_water_cylinder)

# Returns
- Sinogram data as Julia array (rows × cols × angles)

# Example
```julia
cfg = create_gecatsim_water_cylinder(kVp=120.0, mAs=200.0)
sinogram = run_gecatsim_simulation(cfg)
```

# Notes
- This is a placeholder implementation
- Full GECATSIM simulation requires understanding their API structure
- Will be implemented based on GECATSIM documentation
"""
function run_gecatsim_simulation(cfg::PythonCall.Py)
    # Import GECATSIM
    gecatsim = pyimport("gecatsim")

    # Placeholder: Full implementation requires understanding GECATSIM API
    error("""
    GECATSIM simulation wrapper not yet fully implemented.

    To implement:
    1. Create phantom from cfg["phantom"]
    2. Set up scanner geometry from cfg["scanner"]
    3. Run forward projection: gecatsim.CatSim.run(cfg)
    4. Extract sinogram data
    5. Convert to Julia array

    See GECATSIM documentation:
    - https://github.com/MolloiLab/main
    - Example scripts in GECATSIM repository

    Next steps:
    1. Study GECATSIM example configurations
    2. Understand CatSim.run() inputs/outputs
    3. Map BasisSimulator phantom → GECATSIM phantom
    4. Implement data conversion
    """)
end

"""
    compare_sinograms(
        basis_sinogram::Array{Float64, 3},
        gecatsim_sinogram::Array{Float64, 3}
    ) -> NamedTuple

Compare sinograms from BasisSimulator and GECATSIM.

# Arguments
- `basis_sinogram`: Sinogram from BasisSimulator
- `gecatsim_sinogram`: Sinogram from GECATSIM

# Returns
- Named tuple with comparison metrics:
  - `rmse`: Root mean square error
  - `mae`: Mean absolute error
  - `max_diff`: Maximum absolute difference
  - `correlation`: Pearson correlation coefficient

# Example
```julia
metrics = compare_sinograms(basis_sino, gecatsim_sino)
println("RMSE: ", metrics.rmse)
println("Correlation: ", metrics.correlation)
```
"""
function compare_sinograms(
    basis_sinogram::Array{Float64, 3},
    gecatsim_sinogram::Array{Float64, 3}
)
    # Check dimensions match
    if size(basis_sinogram) != size(gecatsim_sinogram)
        error("Sinogram dimensions do not match: $(size(basis_sinogram)) vs $(size(gecatsim_sinogram))")
    end

    # Compute metrics
    diff = basis_sinogram .- gecatsim_sinogram
    rmse = sqrt(mean(diff.^2))
    mae = mean(abs.(diff))
    max_diff = maximum(abs.(diff))

    # Correlation
    basis_flat = vec(basis_sinogram)
    gecatsim_flat = vec(gecatsim_sinogram)
    correlation = cor(basis_flat, gecatsim_flat)

    return (
        rmse = rmse,
        mae = mae,
        max_diff = max_diff,
        correlation = correlation
    )
end

"""
    compare_reconstructions(
        basis_volume::Array{Float64, 3},
        gecatsim_volume::Array{Float64, 3}
    ) -> NamedTuple

Compare reconstructed volumes from BasisSimulator and GECATSIM.

# Arguments
- `basis_volume`: Reconstructed volume from BasisSimulator
- `gecatsim_volume`: Reconstructed volume from GECATSIM

# Returns
- Named tuple with comparison metrics:
  - `rmse`: Root mean square error (cm^-1)
  - `mae`: Mean absolute error (cm^-1)
  - `hu_rmse`: Root mean square error in HU
  - `hu_mae`: Mean absolute error in HU
  - `ssim`: Structural similarity index (if available)
  - `correlation`: Pearson correlation coefficient

# Example
```julia
metrics = compare_reconstructions(basis_vol, gecatsim_vol)
println("HU RMSE: ", metrics.hu_rmse)
println("SSIM: ", metrics.ssim)
```
"""
function compare_reconstructions(
    basis_volume::Array{Float64, 3},
    gecatsim_volume::Array{Float64, 3}
)
    # Check dimensions match
    if size(basis_volume) != size(gecatsim_volume)
        error("Volume dimensions do not match: $(size(basis_volume)) vs $(size(gecatsim_volume))")
    end

    # Compute metrics in attenuation units (cm^-1)
    diff = basis_volume .- gecatsim_volume
    rmse = sqrt(mean(diff.^2))
    mae = mean(abs.(diff))

    # Convert to HU for clinical interpretation
    # HU = (μ - μ_water) / μ_water * 1000
    # Assuming water attenuation ~0.2 cm^-1 at 70 keV
    mu_water = 0.2  # cm^-1 (approximate)
    basis_hu = (basis_volume .- mu_water) ./ mu_water .* 1000
    gecatsim_hu = (gecatsim_volume .- mu_water) ./ mu_water .* 1000

    diff_hu = basis_hu .- gecatsim_hu
    hu_rmse = sqrt(mean(diff_hu.^2))
    hu_mae = mean(abs.(diff_hu))

    # Correlation
    basis_flat = vec(basis_volume)
    gecatsim_flat = vec(gecatsim_volume)
    correlation = cor(basis_flat, gecatsim_flat)

    # SSIM (structural similarity index) - placeholder
    # Requires ImageQualityIndexes.jl or similar
    ssim = missing  # TODO: implement if needed

    return (
        rmse = rmse,
        mae = mae,
        hu_rmse = hu_rmse,
        hu_mae = hu_mae,
        ssim = ssim,
        correlation = correlation
    )
end

# ==============================================================================
# Utility Functions
# ==============================================================================

"""
    print_comparison_metrics(metrics::NamedTuple; name::String="Comparison")

Pretty-print comparison metrics.

# Arguments
- `metrics`: Named tuple from compare_sinograms or compare_reconstructions
- `name`: Name for the comparison (e.g., "Sinogram", "Reconstruction")

# Example
```julia
metrics = compare_sinograms(basis_sino, gecatsim_sino)
print_comparison_metrics(metrics, name="Sinogram")
```
"""
function print_comparison_metrics(metrics::NamedTuple; name::String="Comparison")
    println("\n" * "="^70)
    println("$name COMPARISON METRICS")
    println("="^70)

    for (key, value) in pairs(metrics)
        if value isa Missing
            println("  $(String(key)): N/A")
        elseif value isa Float64
            println("  $(String(key)): $(round(value, digits=4))")
        else
            println("  $(String(key)): $value")
        end
    end

    println("="^70 * "\n")
end

# Export functions
export create_gecatsim_water_cylinder
export run_gecatsim_simulation
export compare_sinograms
export compare_reconstructions
export print_comparison_metrics
