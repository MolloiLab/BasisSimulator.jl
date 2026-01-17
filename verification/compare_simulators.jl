# =============================================================================
# Simulator Comparison Module
# =============================================================================
#
# Compares BasisSimulator and CatSim outputs against ground truth HU values
# from XrayAttenuation.jl (NIST/Geant4 data).
#
# Three-way comparison:
#   1. Ground Truth (XrayAttenuation.jl) - PHYSICS REFERENCE
#   2. CatSim (GE reference simulator)
#   3. BasisSimulator (our implementation)
#
# Ground truth is the tiebreaker if CatSim and BasisSimulator disagree.
#
# =============================================================================

module CompareSimulators

using Statistics
using Printf
using JSON
using Dates
using NPZ

# Include BasisSimulator (assumes we're running from the package)
using BasisSimulator
import XrayAttenuation as XA

export ComparisonConfig, ComparisonResult, RegionComparison
export run_comparison, compare_reconstructions, compute_region_stats
export generate_comparison_report, print_comparison_summary

# =============================================================================
# Configuration
# =============================================================================

"""
Scale configurations matching CatSim runner.
"""
const SCALE_CONFIGS = Dict(
    :dev => (
        phantom_n_voxels = 64,
        phantom_n_slices = 8,
        n_views = 90,
        n_rows = 16,
        n_cols = 128,
        recon_size = 64,
        fov_cm = 35.0,
        z_cm = 4.0,
    ),
    :integration => (
        phantom_n_voxels = 128,
        phantom_n_slices = 16,
        n_views = 180,
        n_rows = 32,
        n_cols = 256,
        recon_size = 128,
        fov_cm = 35.0,
        z_cm = 4.0,
    ),
    :verification => (
        phantom_n_voxels = 256,
        phantom_n_slices = 32,
        n_views = 360,
        n_rows = 64,
        n_cols = 512,
        recon_size = 256,
        fov_cm = 35.0,
        z_cm = 4.0,
    ),
    :publication => (
        phantom_n_voxels = 512,
        phantom_n_slices = 64,
        n_views = 900,
        n_rows = 64,
        n_cols = 736,
        recon_size = 512,
        fov_cm = 35.0,
        z_cm = 4.0,
    ),
)

"""
HU tolerance thresholds (from guardrails.md).
"""
const HU_TOLERANCES = Dict(
    :water => (acceptable = 20.0, failure = 50.0),
    :air => (acceptable = 20.0, failure = 50.0),
    :soft_tissue => (acceptable = 20.0, failure = 40.0),
    :bone => (acceptable = 50.0, failure = 100.0),
    :calcium => (acceptable = 30.0, failure = 60.0),
    :iodine => (acceptable = 30.0, failure = 60.0),
)

"""
    ComparisonConfig

Configuration for simulator comparison.
"""
struct ComparisonConfig
    scale::Symbol
    kvp::Int
    phantom_type::Symbol  # :water or :gammex
    output_dir::String
    catsim_results_dir::Union{String, Nothing}
    use_gpu::Bool
    noise_seed::Int
end

function ComparisonConfig(;
    scale::Symbol = :dev,
    kvp::Int = 120,
    phantom_type::Symbol = :water,
    output_dir::String = "verification_results",
    catsim_results_dir::Union{String, Nothing} = nothing,
    use_gpu::Bool = true,
    noise_seed::Int = 42
)
    return ComparisonConfig(scale, kvp, phantom_type, output_dir, catsim_results_dir, use_gpu, noise_seed)
end

# =============================================================================
# Comparison Results
# =============================================================================

"""
    RegionComparison

HU comparison for a single region (material).
"""
struct RegionComparison
    region_label::RegionLabel
    material_symbol::Symbol
    # Ground truth (XrayAttenuation.jl)
    ground_truth_hu::Float64
    ground_truth_uncertainty::Float64
    # BasisSimulator results
    basis_hu_mean::Float64
    basis_hu_std::Float64
    basis_n_voxels::Int
    # CatSim results (may be nothing if not available)
    catsim_hu_mean::Union{Float64, Nothing}
    catsim_hu_std::Union{Float64, Nothing}
    catsim_n_voxels::Union{Int, Nothing}
    # Deviations from ground truth
    basis_deviation::Float64      # basis_hu_mean - ground_truth_hu
    catsim_deviation::Union{Float64, Nothing}
    # Pass/fail status
    basis_acceptable::Bool
    basis_failure::Bool
    catsim_acceptable::Union{Bool, Nothing}
    catsim_failure::Union{Bool, Nothing}
end

"""
    ComparisonResult

Complete comparison result for a phantom.
"""
struct ComparisonResult
    config::ComparisonConfig
    timestamp::String
    # Per-region results
    region_comparisons::Vector{RegionComparison}
    # Overall metrics
    water_hu_basis::Float64
    water_hu_catsim::Union{Float64, Nothing}
    n_regions_compared::Int
    n_basis_acceptable::Int
    n_basis_failures::Int
    n_catsim_acceptable::Union{Int, Nothing}
    n_catsim_failures::Union{Int, Nothing}
    # Ordering checks (for Gammex phantom)
    calcium_ordering_basis::Bool
    calcium_ordering_catsim::Union{Bool, Nothing}
    iodine_ordering_basis::Bool
    iodine_ordering_catsim::Union{Bool, Nothing}
    # Uniformity (for water phantom)
    cupping_hu_basis::Union{Float64, Nothing}  # center-edge HU difference
    cupping_hu_catsim::Union{Float64, Nothing}
end

# =============================================================================
# Ground Truth Computation
# =============================================================================

"""
Load or compute ground truth HU values from expected_hu.jl.
"""
function get_ground_truth_hu(kvp::Int)
    # Include the ground truth module
    ground_truth_path = joinpath(@__DIR__, "..", "test", "ground_truth", "expected_hu.jl")

    if isfile(ground_truth_path)
        # Include the file to get EXPECTED_HU
        include(ground_truth_path)
        return EXPECTED_HU[kvp]
    else
        # Compute on the fly using BasisSimulator's functions
        @warn "Ground truth file not found, computing from XrayAttenuation.jl"
        return compute_ground_truth_from_xray_attenuation(kvp)
    end
end

"""
Compute ground truth HU from XrayAttenuation.jl directly.
"""
function compute_ground_truth_from_xray_attenuation(kvp::Int)
    energies, weights = load_spectrum(kvp)
    energies, weights = downsample_spectrum(energies, weights, 60)

    # Compute water reference
    μ_water = compute_effective_μ_material(XA.Materials.water, energies, weights)

    # Material mapping
    materials_to_compute = [
        (:air, XA.Materials.air),
        (:water, XA.Materials.water),
        (:solid_water, XA.Materials.water),
    ]

    # Add Gammex materials
    for (sym, mat) in [
        (:Ca_50, get_material(:Ca_50)),
        (:Ca_100, get_material(:Ca_100)),
        (:Ca_200, get_material(:Ca_200)),
        (:Ca_300, get_material(:Ca_300)),
        (:Ca_400, get_material(:Ca_400)),
        (:Ca_500, get_material(:Ca_500)),
        (:Ca_600, get_material(:Ca_600)),
        (:I_2_0, get_material(:I_2_0)),
        (:I_2_5, get_material(:I_2_5)),
        (:I_5_0, get_material(:I_5_0)),
        (:I_7_5, get_material(:I_7_5)),
        (:I_10_0, get_material(:I_10_0)),
        (:I_15_0, get_material(:I_15_0)),
        (:I_20_0, get_material(:I_20_0)),
    ]
        push!(materials_to_compute, (sym, mat))
    end

    result = Dict{Symbol, NamedTuple}()

    for (sym, mat) in materials_to_compute
        μ_eff = compute_effective_μ_material(mat, energies, weights)
        expected_hu = 1000.0 * (μ_eff - μ_water) / μ_water

        # Estimate uncertainty
        uncertainty = max(10.0, abs(expected_hu) * 0.03)
        if startswith(string(sym), "I_")
            uncertainty = max(15.0, abs(expected_hu) * 0.05)
        end

        result[sym] = (expected_hu = expected_hu, uncertainty_hu = uncertainty)
    end

    return result
end

"""
Helper to compute effective μ for a material over a spectrum.
"""
function compute_effective_μ_material(material, energies, weights)
    total_weight = sum(weights)
    μ_weighted_sum = 0.0

    for (E, w) in zip(energies, weights)
        μ = compute_μ_at_energy(material, Float64(E))
        μ_weighted_sum += w * μ
    end

    return μ_weighted_sum / total_weight
end

# =============================================================================
# BasisSimulator Runner
# =============================================================================

"""
Run BasisSimulator and return reconstruction in HU.
"""
function run_basissimulator(config::ComparisonConfig)
    scale_cfg = SCALE_CONFIGS[config.scale]

    println("Running BasisSimulator...")
    println("  Scale: $(config.scale)")
    println("  Phantom: $(config.phantom_type)")
    println("  kVp: $(config.kvp)")

    # Create phantom
    phantom = if config.phantom_type == :water
        create_water_phantom(
            n_voxels = scale_cfg.phantom_n_voxels,
            n_slices = scale_cfg.phantom_n_slices,
            fov_cm = scale_cfg.fov_cm,
            z_cm = scale_cfg.z_cm
        )
    else
        create_gammex_472(
            n_voxels = scale_cfg.phantom_n_voxels,
            n_slices = scale_cfg.phantom_n_slices,
            fov_cm = scale_cfg.fov_cm,
            z_cm = scale_cfg.z_cm
        )
    end

    # Create geometry
    geom = create_aquilion_one(
        n_angles = scale_cfg.n_views,
        n_rows = scale_cfg.n_rows,
        n_cols = scale_cfg.n_cols,
        fov_cm = scale_cfg.fov_cm,
        z_cm = scale_cfg.z_cm
    )

    # Load spectrum
    energies, weights = load_spectrum(config.kvp)
    energies, weights = downsample_spectrum(energies, weights, 30)
    materials = get_region_materials()

    # Use CPU for comparison (GPU requires Metal to be loaded in caller)
    mask = phantom.mask

    # Configure physics (minimal for comparison - just noise)
    physics = minimal_physics_config(
        noise_level = 0.01,
        noise_seed = config.noise_seed
    )

    # Forward projection
    println("  Forward projecting...")
    sinogram = forward_project(
        mask, geom;
        energies = energies,
        weights = weights,
        materials = materials,
        physics = physics
    )

    # Reconstruction
    println("  Reconstructing...")
    recon_size = (scale_cfg.recon_size, scale_cfg.recon_size, scale_cfg.phantom_n_slices)
    recon = fdk_reconstruct(sinogram, geom, recon_size)

    # Convert to CPU if needed
    recon_cpu = Array(recon)

    # Compute empirical water reference from solid water region
    center_z = scale_cfg.phantom_n_slices ÷ 2 + 1
    mask_recon = downsample_mask(phantom.mask, recon_size)

    water_mask = mask_recon[:, :, center_z] .== UInt8(REGION_SOLID_WATER)
    if sum(water_mask) > 0
        μ_water_empirical = mean(recon_cpu[:, :, center_z][water_mask])
    else
        # Fallback to NIST value
        μ_water_empirical = Float32(compute_effective_μ_material(
            XA.Materials.water, energies, weights
        ))
    end

    # Convert to HU
    recon_hu = 1000.0f0 .* (recon_cpu .- μ_water_empirical) ./ μ_water_empirical

    println("  Done. HU range: [$(round(minimum(recon_hu))), $(round(maximum(recon_hu)))]")

    return (
        recon_hu = recon_hu,
        recon_μ = recon_cpu,
        μ_water = μ_water_empirical,
        mask = mask_recon,
        phantom = phantom
    )
end

"""
Create a simple water cylinder phantom.
"""
function create_water_phantom(; n_voxels::Int=64, n_slices::Int=8, fov_cm::Float64=35.0, z_cm::Float64=4.0)
    dx = fov_cm / n_voxels
    dz = z_cm / n_slices

    # Coordinate arrays
    x = range(-fov_cm/2 + dx/2, fov_cm/2 - dx/2, length=n_voxels)
    y = range(-fov_cm/2 + dx/2, fov_cm/2 - dx/2, length=n_voxels)

    # Water cylinder radius (100mm = 10cm)
    water_radius = 10.0  # cm

    # Initialize
    μ = zeros(Float32, n_voxels, n_voxels, n_slices)
    mask = zeros(UInt8, n_voxels, n_voxels, n_slices)

    μ_water = Float32(compute_μ_at_energy(XA.Materials.water, 60.0))
    μ_air = Float32(compute_μ_at_energy(XA.Materials.air, 60.0))

    for k in 1:n_slices
        for j in 1:n_voxels
            for i in 1:n_voxels
                r = sqrt(x[i]^2 + y[j]^2)
                if r <= water_radius
                    μ[i, j, k] = μ_water
                    mask[i, j, k] = UInt8(REGION_SOLID_WATER)
                else
                    μ[i, j, k] = μ_air
                    mask[i, j, k] = UInt8(REGION_BACKGROUND)
                end
            end
        end
    end

    return Phantom(μ, mask, (dx, dx, dz), (-fov_cm/2 + dx/2, -fov_cm/2 + dx/2, -z_cm/2 + dz/2), (fov_cm, fov_cm, z_cm))
end

"""
Downsample mask to reconstruction resolution.
"""
function downsample_mask(mask, new_size)
    old_size = size(mask)
    if old_size == new_size
        return mask
    end
    scale = old_size ./ new_size
    result = similar(mask, new_size)
    for k in 1:new_size[3], j in 1:new_size[2], i in 1:new_size[1]
        oi = clamp(round(Int, (i - 0.5) * scale[1] + 0.5), 1, old_size[1])
        oj = clamp(round(Int, (j - 0.5) * scale[2] + 0.5), 1, old_size[2])
        ok = clamp(round(Int, (k - 0.5) * scale[3] + 0.5), 1, old_size[3])
        result[i, j, k] = mask[oi, oj, ok]
    end
    return result
end

# =============================================================================
# CatSim Results Loader
# =============================================================================

"""
Load CatSim reconstruction results from numpy files.
"""
function load_catsim_results(catsim_dir::String)
    recon_path = joinpath(catsim_dir, "catsim_recon.npy")
    config_path = joinpath(catsim_dir, "catsim_config.json")

    if !isfile(recon_path)
        @warn "CatSim reconstruction not found at $recon_path"
        return nothing
    end

    println("Loading CatSim results from $catsim_dir...")

    # Load reconstruction (numpy format)
    recon = npzread(recon_path)

    # Load config
    config = if isfile(config_path)
        JSON.parsefile(config_path)
    else
        Dict()
    end

    # CatSim outputs in HU already (typically)
    # If μ, need to convert

    println("  CatSim recon shape: $(size(recon))")
    println("  CatSim HU range: [$(round(minimum(recon))), $(round(maximum(recon)))]")

    return (
        recon_hu = recon,
        config = config
    )
end

# =============================================================================
# Comparison Functions
# =============================================================================

"""
Compute region statistics from reconstruction.
"""
function compute_region_stats(recon_hu::Array{<:Real, 3}, mask::Array{UInt8, 3}, label::RegionLabel)
    region_mask = mask .== UInt8(label)
    n_voxels = sum(region_mask)

    if n_voxels == 0
        return nothing
    end

    hu_vals = recon_hu[region_mask]

    return (
        mean = mean(hu_vals),
        std = std(hu_vals),
        min = minimum(hu_vals),
        max = maximum(hu_vals),
        n_voxels = n_voxels
    )
end

"""
Compare reconstructions from BasisSimulator and CatSim against ground truth.
"""
function compare_reconstructions(
    basis_result::NamedTuple,
    catsim_result::Union{NamedTuple, Nothing},
    config::ComparisonConfig
)
    # Get ground truth
    ground_truth = get_ground_truth_hu(config.kvp)

    # Regions to compare
    regions_to_compare = if config.phantom_type == :water
        [(REGION_SOLID_WATER, :water)]
    else
        [
            (REGION_SOLID_WATER, :solid_water),
            (REGION_CA_50, :Ca_50),
            (REGION_CA_100, :Ca_100),
            (REGION_CA_200, :Ca_200),
            (REGION_CA_300, :Ca_300),
            (REGION_CA_400, :Ca_400),
            (REGION_CA_500, :Ca_500),
            (REGION_CA_600, :Ca_600),
            (REGION_I_2_0, :I_2_0),
            (REGION_I_2_5, :I_2_5),
            (REGION_I_5_0, :I_5_0),
            (REGION_I_7_5, :I_7_5),
            (REGION_I_10_0, :I_10_0),
            (REGION_I_15_0, :I_15_0),
            (REGION_I_20_0, :I_20_0),
        ]
    end

    region_comparisons = RegionComparison[]

    for (label, symbol) in regions_to_compare
        # Ground truth
        gt = get(ground_truth, symbol, nothing)
        if gt === nothing
            continue
        end
        gt_hu = gt.expected_hu
        gt_uncertainty = gt.uncertainty_hu

        # BasisSimulator stats
        basis_stats = compute_region_stats(basis_result.recon_hu, basis_result.mask, label)
        if basis_stats === nothing
            continue
        end

        # CatSim stats (if available)
        catsim_stats = if catsim_result !== nothing
            # Need to create mask at CatSim resolution
            # For now, assume same resolution
            compute_region_stats(catsim_result.recon_hu, basis_result.mask, label)
        else
            nothing
        end

        # Compute deviations
        basis_deviation = basis_stats.mean - gt_hu
        catsim_deviation = catsim_stats !== nothing ? catsim_stats.mean - gt_hu : nothing

        # Get tolerance for this material type
        tol_key = if startswith(string(symbol), "Ca_")
            :calcium
        elseif startswith(string(symbol), "I_")
            :iodine
        elseif symbol == :water || symbol == :solid_water
            :water
        elseif symbol == :air
            :air
        else
            :soft_tissue
        end
        tol = HU_TOLERANCES[tol_key]

        # Pass/fail
        basis_acceptable = abs(basis_deviation) <= tol.acceptable
        basis_failure = abs(basis_deviation) > tol.failure

        catsim_acceptable = catsim_deviation !== nothing ? abs(catsim_deviation) <= tol.acceptable : nothing
        catsim_failure = catsim_deviation !== nothing ? abs(catsim_deviation) > tol.failure : nothing

        push!(region_comparisons, RegionComparison(
            label, symbol,
            gt_hu, gt_uncertainty,
            basis_stats.mean, basis_stats.std, basis_stats.n_voxels,
            catsim_stats !== nothing ? catsim_stats.mean : nothing,
            catsim_stats !== nothing ? catsim_stats.std : nothing,
            catsim_stats !== nothing ? catsim_stats.n_voxels : nothing,
            basis_deviation,
            catsim_deviation,
            basis_acceptable,
            basis_failure,
            catsim_acceptable,
            catsim_failure
        ))
    end

    # Compute overall metrics
    water_regions = filter(r -> r.material_symbol in [:water, :solid_water], region_comparisons)
    water_hu_basis = isempty(water_regions) ? NaN : water_regions[1].basis_hu_mean
    water_hu_catsim = isempty(water_regions) || water_regions[1].catsim_hu_mean === nothing ?
                      nothing : water_regions[1].catsim_hu_mean

    n_basis_acceptable = count(r -> r.basis_acceptable, region_comparisons)
    n_basis_failures = count(r -> r.basis_failure, region_comparisons)
    n_catsim_acceptable = catsim_result !== nothing ?
                          count(r -> r.catsim_acceptable === true, region_comparisons) : nothing
    n_catsim_failures = catsim_result !== nothing ?
                        count(r -> r.catsim_failure === true, region_comparisons) : nothing

    # Check ordering (Gammex phantom only)
    calcium_ordering_basis = check_calcium_ordering(region_comparisons, :basis)
    calcium_ordering_catsim = catsim_result !== nothing ? check_calcium_ordering(region_comparisons, :catsim) : nothing
    iodine_ordering_basis = check_iodine_ordering(region_comparisons, :basis)
    iodine_ordering_catsim = catsim_result !== nothing ? check_iodine_ordering(region_comparisons, :catsim) : nothing

    # Compute cupping (water phantom)
    cupping_basis = config.phantom_type == :water ?
                    compute_cupping(basis_result.recon_hu, basis_result.mask) : nothing
    cupping_catsim = config.phantom_type == :water && catsim_result !== nothing ?
                     compute_cupping(catsim_result.recon_hu, basis_result.mask) : nothing

    return ComparisonResult(
        config,
        string(now()),
        region_comparisons,
        water_hu_basis,
        water_hu_catsim,
        length(region_comparisons),
        n_basis_acceptable,
        n_basis_failures,
        n_catsim_acceptable,
        n_catsim_failures,
        calcium_ordering_basis,
        calcium_ordering_catsim,
        iodine_ordering_basis,
        iodine_ordering_catsim,
        cupping_basis,
        cupping_catsim
    )
end

"""
Check if calcium HU values are monotonically increasing.
"""
function check_calcium_ordering(comparisons::Vector{RegionComparison}, simulator::Symbol)
    ca_symbols = [:Ca_50, :Ca_100, :Ca_200, :Ca_300, :Ca_400, :Ca_500, :Ca_600]
    ca_comparisons = filter(r -> r.material_symbol in ca_symbols, comparisons)

    if length(ca_comparisons) < 2
        return true  # Not enough data
    end

    # Sort by concentration
    sort!(ca_comparisons, by = r -> findfirst(==(r.material_symbol), ca_symbols))

    # Get HU values
    hu_values = if simulator == :basis
        [r.basis_hu_mean for r in ca_comparisons]
    else
        vals = [r.catsim_hu_mean for r in ca_comparisons]
        any(isnothing, vals) && return nothing
        vals
    end

    return issorted(hu_values)
end

"""
Check if iodine HU values are monotonically increasing.
"""
function check_iodine_ordering(comparisons::Vector{RegionComparison}, simulator::Symbol)
    i_symbols = [:I_2_0, :I_2_5, :I_5_0, :I_7_5, :I_10_0, :I_15_0, :I_20_0]
    i_comparisons = filter(r -> r.material_symbol in i_symbols, comparisons)

    if length(i_comparisons) < 2
        return true  # Not enough data
    end

    # Sort by concentration
    sort!(i_comparisons, by = r -> findfirst(==(r.material_symbol), i_symbols))

    # Get HU values
    hu_values = if simulator == :basis
        [r.basis_hu_mean for r in i_comparisons]
    else
        vals = [r.catsim_hu_mean for r in i_comparisons]
        any(isnothing, vals) && return nothing
        vals
    end

    return issorted(hu_values)
end

"""
Compute cupping artifact magnitude (center - edge HU difference).
"""
function compute_cupping(recon_hu::Array{<:Real, 3}, mask::Array{UInt8, 3})
    center_z = size(recon_hu, 3) ÷ 2 + 1
    slice = recon_hu[:, :, center_z]
    mask_slice = mask[:, :, center_z]

    water_mask = mask_slice .== UInt8(REGION_SOLID_WATER)
    if sum(water_mask) == 0
        return nothing
    end

    # Find center and edge regions within water
    nx, ny = size(slice)
    cx, cy = nx ÷ 2, ny ÷ 2

    # Center region (inner 20%)
    center_radius = min(nx, ny) * 0.1
    center_mask = similar(water_mask, Bool)
    for j in 1:ny, i in 1:nx
        r = sqrt((i - cx)^2 + (j - cy)^2)
        center_mask[i, j] = r <= center_radius && water_mask[i, j]
    end

    # Edge region (outer 20-30%)
    inner_edge_radius = min(nx, ny) * 0.35
    outer_edge_radius = min(nx, ny) * 0.45
    edge_mask = similar(water_mask, Bool)
    for j in 1:ny, i in 1:nx
        r = sqrt((i - cx)^2 + (j - cy)^2)
        edge_mask[i, j] = inner_edge_radius <= r <= outer_edge_radius && water_mask[i, j]
    end

    if sum(center_mask) == 0 || sum(edge_mask) == 0
        return nothing
    end

    center_hu = mean(slice[center_mask])
    edge_hu = mean(slice[edge_mask])

    return center_hu - edge_hu  # Positive = cupping artifact
end

# =============================================================================
# Reporting
# =============================================================================

"""
Print comparison summary to console.
"""
function print_comparison_summary(result::ComparisonResult)
    println()
    println("=" ^ 80)
    println("SIMULATOR COMPARISON REPORT")
    println("=" ^ 80)
    println()
    println("Configuration:")
    println("  Scale: $(result.config.scale)")
    println("  kVp: $(result.config.kvp)")
    println("  Phantom: $(result.config.phantom_type)")
    println()

    # Water HU
    println("-" ^ 80)
    println("WATER HU (Expected: 0 HU, Tolerance: ±20 HU)")
    println("-" ^ 80)
    water_status = abs(result.water_hu_basis) <= 20 ? "PASS" : "FAIL"
    println("  BasisSimulator: $(round(result.water_hu_basis, digits=1)) HU [$water_status]")
    if result.water_hu_catsim !== nothing
        catsim_status = abs(result.water_hu_catsim) <= 20 ? "PASS" : "FAIL"
        println("  CatSim:         $(round(result.water_hu_catsim, digits=1)) HU [$catsim_status]")
    end
    println()

    # Region comparison table
    println("-" ^ 80)
    println("REGION COMPARISON (vs Ground Truth)")
    println("-" ^ 80)
    println()
    println(@sprintf("%-15s | %10s | %10s | %10s | %10s | %6s",
                     "Material", "GT HU", "Basis HU", "Basis Dev", "CatSim Dev", "Status"))
    println("-" ^ 80)

    for r in result.region_comparisons
        gt_hu = @sprintf("%10.1f", r.ground_truth_hu)
        basis_hu = @sprintf("%10.1f", r.basis_hu_mean)
        basis_dev = @sprintf("%+10.1f", r.basis_deviation)
        catsim_dev = r.catsim_deviation !== nothing ?
                     @sprintf("%+10.1f", r.catsim_deviation) : "    N/A   "

        status = if r.basis_failure
            "FAIL"
        elseif r.basis_acceptable
            "OK"
        else
            "WARN"
        end

        println(@sprintf("  %-13s | %s | %s | %s | %s | %6s",
                        string(r.material_symbol), gt_hu, basis_hu, basis_dev, catsim_dev, status))
    end

    println("-" ^ 80)
    println()

    # Summary
    println("SUMMARY:")
    println("  Regions compared: $(result.n_regions_compared)")
    println("  BasisSimulator: $(result.n_basis_acceptable) OK, $(result.n_basis_failures) FAIL")
    if result.n_catsim_acceptable !== nothing
        println("  CatSim:         $(result.n_catsim_acceptable) OK, $(result.n_catsim_failures) FAIL")
    end
    println()

    # Ordering checks
    if result.config.phantom_type == :gammex
        println("ORDERING CHECKS (must be monotonic):")
        ca_status = result.calcium_ordering_basis ? "PASS" : "FAIL"
        i_status = result.iodine_ordering_basis ? "PASS" : "FAIL"
        println("  Calcium (Basis): $ca_status")
        println("  Iodine (Basis):  $i_status")
        if result.calcium_ordering_catsim !== nothing
            ca_catsim = result.calcium_ordering_catsim ? "PASS" : "FAIL"
            i_catsim = result.iodine_ordering_catsim ? "PASS" : "FAIL"
            println("  Calcium (CatSim): $ca_catsim")
            println("  Iodine (CatSim):  $i_catsim")
        end
    end

    # Cupping
    if result.cupping_hu_basis !== nothing
        println()
        println("CUPPING ARTIFACT (center-edge HU, <20 HU acceptable):")
        cupping_status = abs(result.cupping_hu_basis) <= 20 ? "PASS" : "FAIL"
        println("  BasisSimulator: $(round(result.cupping_hu_basis, digits=1)) HU [$cupping_status]")
        if result.cupping_hu_catsim !== nothing
            catsim_status = abs(result.cupping_hu_catsim) <= 20 ? "PASS" : "FAIL"
            println("  CatSim:         $(round(result.cupping_hu_catsim, digits=1)) HU [$catsim_status]")
        end
    end

    println()
    println("=" ^ 80)

    # Overall pass/fail
    overall_pass = result.n_basis_failures == 0 &&
                   result.calcium_ordering_basis &&
                   result.iodine_ordering_basis &&
                   (result.cupping_hu_basis === nothing || abs(result.cupping_hu_basis) <= 20)

    if overall_pass
        println("OVERALL: PASS")
    else
        println("OVERALL: FAIL")
    end
    println("=" ^ 80)
    println()

    return overall_pass
end

"""
Generate JSON comparison report.
"""
function generate_comparison_report(result::ComparisonResult, output_path::String)
    mkpath(dirname(output_path))

    report = Dict(
        "timestamp" => result.timestamp,
        "config" => Dict(
            "scale" => string(result.config.scale),
            "kvp" => result.config.kvp,
            "phantom_type" => string(result.config.phantom_type),
        ),
        "summary" => Dict(
            "water_hu_basis" => result.water_hu_basis,
            "water_hu_catsim" => result.water_hu_catsim,
            "n_regions" => result.n_regions_compared,
            "n_basis_acceptable" => result.n_basis_acceptable,
            "n_basis_failures" => result.n_basis_failures,
            "calcium_ordering_basis" => result.calcium_ordering_basis,
            "iodine_ordering_basis" => result.iodine_ordering_basis,
            "cupping_hu_basis" => result.cupping_hu_basis,
        ),
        "regions" => [
            Dict(
                "material" => string(r.material_symbol),
                "ground_truth_hu" => r.ground_truth_hu,
                "basis_hu_mean" => r.basis_hu_mean,
                "basis_hu_std" => r.basis_hu_std,
                "basis_deviation" => r.basis_deviation,
                "basis_acceptable" => r.basis_acceptable,
                "basis_failure" => r.basis_failure,
                "catsim_hu_mean" => r.catsim_hu_mean,
                "catsim_deviation" => r.catsim_deviation,
            )
            for r in result.region_comparisons
        ]
    )

    open(output_path, "w") do f
        JSON.print(f, report, 2)
    end

    println("Report saved to: $output_path")
end

# =============================================================================
# Main Entry Point
# =============================================================================

"""
Run full comparison pipeline.
"""
function run_comparison(config::ComparisonConfig)
    println()
    println("=" ^ 80)
    println("STARTING COMPARISON PIPELINE")
    println("=" ^ 80)
    println()

    # Run BasisSimulator
    basis_result = run_basissimulator(config)

    # Load CatSim results if available
    catsim_result = nothing
    if config.catsim_results_dir !== nothing && isdir(config.catsim_results_dir)
        catsim_result = load_catsim_results(config.catsim_results_dir)
    else
        println()
        println("NOTE: CatSim results not provided. Running BasisSimulator-only comparison.")
        println()
    end

    # Compare
    result = compare_reconstructions(basis_result, catsim_result, config)

    # Print summary
    passed = print_comparison_summary(result)

    # Save report
    mkpath(config.output_dir)
    report_path = joinpath(config.output_dir, "comparison_report.json")
    generate_comparison_report(result, report_path)

    return (result = result, passed = passed)
end

end # module
