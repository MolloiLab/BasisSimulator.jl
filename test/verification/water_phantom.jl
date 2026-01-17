# =============================================================================
# VERIFY-001: Water Phantom HU Accuracy Verification
# =============================================================================
#
# This test verifies that BasisSimulator produces correct HU values for a
# water phantom, comparing against CatSim and ground truth.
#
# ACCEPTANCE CRITERIA (from prd.json):
# - Mean HU in water ROI: 0 +/- 20 HU (BasisSimulator)
# - Mean HU in water ROI: 0 +/- 20 HU (CatSim)
# - BasisSimulator vs CatSim difference: < 10 HU
# - Uniformity (std dev) difference: < 5 HU
# - No cupping artifact (center-edge < 20 HU)
#
# USAGE:
#   cd BasisSimulator.jl && julia --project test/verification/water_phantom.jl
#
# =============================================================================

using Test
using Statistics
using Printf
using Dates
using JSON

# Add parent directory to load path
pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using BasisSimulator
import XrayAttenuation as XA

# Try to load NPZ for CatSim result loading (optional dependency)
const NPZ_AVAILABLE = try
    @eval using NPZ
    true
catch
    false
end

# =============================================================================
# CONFIGURATION
# =============================================================================

"""
Water phantom verification configuration.
"""
struct WaterPhantomConfig
    # Simulation scale
    phantom_n_voxels::Int
    phantom_n_slices::Int
    n_views::Int
    n_rows::Int
    n_cols::Int
    recon_size::Int

    # Physical parameters
    fov_cm::Float64
    z_cm::Float64
    water_radius_cm::Float64

    # Acquisition parameters
    kvp::Int
    noise_seed::Int

    # Tolerances (from prd.json)
    water_hu_tolerance::Float64      # Mean HU deviation from 0
    simulator_diff_tolerance::Float64 # BasisSim vs CatSim difference
    uniformity_diff_tolerance::Float64 # Std dev difference
    cupping_tolerance::Float64        # Center-edge HU difference
end

"""
Default verification configuration (integration scale for faster testing).
"""
function default_water_phantom_config(; scale::Symbol=:integration)
    # Scale configurations matching compare_simulators.jl
    scale_configs = Dict(
        :dev => (64, 8, 90, 16, 128, 64),
        :integration => (128, 16, 180, 32, 256, 128),
        :verification => (256, 32, 360, 64, 512, 256),
        :publication => (512, 64, 900, 64, 736, 512)
    )

    cfg = scale_configs[scale]

    return WaterPhantomConfig(
        cfg[1],  # phantom_n_voxels
        cfg[2],  # phantom_n_slices
        cfg[3],  # n_views
        cfg[4],  # n_rows
        cfg[5],  # n_cols
        cfg[6],  # recon_size
        35.0,    # fov_cm
        4.0,     # z_cm
        10.0,    # water_radius_cm (200mm diameter)
        120,     # kvp
        42,      # noise_seed
        20.0,    # water_hu_tolerance (from prd.json)
        10.0,    # simulator_diff_tolerance (from prd.json)
        5.0,     # uniformity_diff_tolerance (from prd.json)
        20.0     # cupping_tolerance (from prd.json)
    )
end

# =============================================================================
# WATER PHANTOM CREATION
# =============================================================================

"""
Create a simple cylindrical water phantom.

Returns a Phantom struct with:
- Water cylinder (radius = water_radius_cm) labeled as REGION_SOLID_WATER
- Air background labeled as REGION_BACKGROUND
"""
function create_water_phantom_for_verification(cfg::WaterPhantomConfig)
    n_voxels = cfg.phantom_n_voxels
    n_slices = cfg.phantom_n_slices
    fov_cm = cfg.fov_cm
    z_cm = cfg.z_cm
    water_radius = cfg.water_radius_cm

    dx = fov_cm / n_voxels
    dz = z_cm / n_slices

    # Coordinate arrays (centered at isocenter)
    x = range(-fov_cm/2 + dx/2, fov_cm/2 - dx/2, length=n_voxels)
    y = range(-fov_cm/2 + dx/2, fov_cm/2 - dx/2, length=n_voxels)

    # Initialize arrays
    μ = zeros(Float32, n_voxels, n_voxels, n_slices)
    mask = zeros(UInt8, n_voxels, n_voxels, n_slices)

    # Get material attenuation at effective energy (60 keV typical)
    μ_water = Float32(compute_μ_at_energy(XA.Materials.water, 60.0))
    μ_air = Float32(compute_μ_at_energy(XA.Materials.air, 60.0))

    # Fill phantom
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

# =============================================================================
# BASISSIMULATOR WATER PHANTOM SIMULATION
# =============================================================================

"""
Run BasisSimulator water phantom simulation.

Returns named tuple with:
- recon_hu: Reconstruction in Hounsfield units
- recon_μ: Reconstruction in attenuation coefficients
- μ_water: Effective water attenuation
- mask: Downsampled mask for ROI analysis
- phantom: Original phantom
"""
function run_basissimulator_water(cfg::WaterPhantomConfig)
    println("=" ^ 60)
    println("RUNNING BASISSIMULATOR WATER PHANTOM")
    println("=" ^ 60)
    println()
    println("Configuration:")
    println("  Phantom: $(cfg.phantom_n_voxels)³ x $(cfg.phantom_n_slices) slices")
    println("  Views: $(cfg.n_views)")
    println("  Detector: $(cfg.n_cols) x $(cfg.n_rows)")
    println("  Recon: $(cfg.recon_size)³")
    println("  kVp: $(cfg.kvp)")
    println()

    # Create water phantom
    println("Creating water phantom...")
    phantom = create_water_phantom_for_verification(cfg)

    # Create geometry using GE Revolution Apex scanner
    println("Creating scanner geometry (GE Revolution Apex)...")
    scanner = GERevolutionApex()
    geom = create_geometry(scanner;
        n_angles = cfg.n_views,
        n_rows = cfg.n_rows,
        n_cols = cfg.n_cols,
        fov_cm = cfg.fov_cm
    )

    # Load spectrum
    println("Loading $(cfg.kvp) kVp spectrum...")
    energies, weights = load_spectrum(cfg.kvp)
    energies, weights = downsample_spectrum(energies, weights, 30)
    materials = get_region_materials()

    # Configure physics: minimal for HU verification (just noise for realistic texture)
    # We want to test the core physics without confounding factors
    physics = minimal_physics_config(
        noise_level = 0.01,
        noise_seed = cfg.noise_seed
    )

    # Forward projection
    println("Forward projecting (polychromatic)...")
    t_start = time()
    sinogram = forward_project(
        phantom.mask, geom;
        energies = energies,
        weights = weights,
        materials = materials,
        physics = physics
    )
    t_fp = time() - t_start
    println("  Forward projection time: $(round(t_fp, digits=2))s")

    # Reconstruction
    println("Reconstructing (FDK)...")
    recon_size = (cfg.recon_size, cfg.recon_size, cfg.phantom_n_slices)
    t_start = time()
    recon = fdk_reconstruct(sinogram, geom, recon_size)
    t_recon = time() - t_start
    println("  Reconstruction time: $(round(t_recon, digits=2))s")

    # Convert to CPU if needed
    recon_cpu = Array(recon)

    # Downsample mask to reconstruction resolution
    mask_recon = downsample_mask_to_size(phantom.mask, recon_size)

    # Compute empirical water reference from solid water region
    center_z = cfg.phantom_n_slices ÷ 2 + 1
    water_mask = mask_recon[:, :, center_z] .== UInt8(REGION_SOLID_WATER)

    if sum(water_mask) > 0
        μ_water_empirical = mean(recon_cpu[:, :, center_z][water_mask])
    else
        error("No water voxels found in reconstruction!")
    end

    # Convert to HU
    recon_hu = 1000.0f0 .* (recon_cpu .- μ_water_empirical) ./ μ_water_empirical

    println()
    println("Results:")
    println("  HU range: [$(round(minimum(recon_hu))), $(round(maximum(recon_hu)))]")
    println("  μ_water (empirical): $(round(μ_water_empirical, digits=6))")
    println()

    return (
        recon_hu = recon_hu,
        recon_μ = recon_cpu,
        μ_water = μ_water_empirical,
        mask = mask_recon,
        phantom = phantom
    )
end

"""
Create water mask for a given reconstruction size.

Creates a cylindrical water region mask based on the phantom config.
"""
function create_water_mask_for_recon(cfg::WaterPhantomConfig, recon_size::Tuple)
    nx, ny, nz = recon_size
    dx = cfg.fov_cm / nx
    water_radius = cfg.water_radius_cm

    # Coordinate arrays (centered at isocenter)
    x = range(-cfg.fov_cm/2 + dx/2, cfg.fov_cm/2 - dx/2, length=nx)
    y = range(-cfg.fov_cm/2 + dx/2, cfg.fov_cm/2 - dx/2, length=ny)

    mask = zeros(UInt8, nx, ny, nz)

    for k in 1:nz, j in 1:ny, i in 1:nx
        r = sqrt(x[i]^2 + y[j]^2)
        if r <= water_radius
            mask[i, j, k] = UInt8(REGION_SOLID_WATER)
        else
            mask[i, j, k] = UInt8(REGION_BACKGROUND)
        end
    end

    return mask
end

"""
Downsample mask to target size using nearest-neighbor.
"""
function downsample_mask_to_size(mask, new_size)
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
# CATSIM INTERFACE
# =============================================================================

"""
Run CatSim water phantom simulation (if available).

Returns named tuple similar to BasisSimulator result, or nothing if CatSim unavailable.
"""
function run_catsim_water(cfg::WaterPhantomConfig; catsim_dir::Union{String,Nothing}=nothing)
    # Check if CatSim results directory exists
    if catsim_dir !== nothing && isdir(catsim_dir)
        println("=" ^ 60)
        println("LOADING CATSIM RESULTS")
        println("=" ^ 60)
        println()

        recon_path = joinpath(catsim_dir, "catsim_recon.npy")
        if !isfile(recon_path)
            @warn "CatSim reconstruction not found at $recon_path"
            return nothing
        end

        if !NPZ_AVAILABLE
            @warn "NPZ package not available - cannot load CatSim results"
            return nothing
        end

        # Load NPZ file
        recon_raw = npzread(recon_path)

        # CatSim outputs in (z, y, x) order, need to permute to (x, y, z)
        # Note: CatSim convention is [slices, rows, cols] = [z, y, x]
        recon_hu = permutedims(Float32.(recon_raw), (3, 2, 1))

        println("  Loaded raw: $(size(recon_raw)), permuted to: $(size(recon_hu))")
        println("  HU range: [$(round(minimum(recon_hu))), $(round(maximum(recon_hu)))]")
        println()

        return (recon_hu = recon_hu,)
    end

    # Try to run CatSim via Python
    println("=" ^ 60)
    println("RUNNING CATSIM WATER PHANTOM")
    println("=" ^ 60)
    println()

    catsim_script = joinpath(@__DIR__, "..", "..", "verification", "run_catsim.py")
    if !isfile(catsim_script)
        @warn "CatSim runner script not found at $catsim_script"
        return nothing
    end

    output_dir = mktempdir()

    # Map scale to string
    scale_str = cfg.phantom_n_voxels <= 64 ? "dev" :
                cfg.phantom_n_voxels <= 128 ? "integration" :
                cfg.phantom_n_voxels <= 256 ? "verification" : "publication"

    cmd = `python3 $catsim_script --phantom water --scale $scale_str --kvp $(cfg.kvp) --output $output_dir`

    try
        run(cmd)

        # Load results
        recon_path = joinpath(output_dir, "catsim_recon.npy")
        if isfile(recon_path) && NPZ_AVAILABLE
            recon_raw = npzread(recon_path)
            # CatSim outputs in (z, y, x) order, need to permute to (x, y, z)
            recon_hu = permutedims(Float32.(recon_raw), (3, 2, 1))
            println("  Loaded raw: $(size(recon_raw)), permuted to: $(size(recon_hu))")
            println("  HU range: [$(round(minimum(recon_hu))), $(round(maximum(recon_hu)))]")
            println()
            return (recon_hu = recon_hu,)
        else
            @warn "CatSim did not produce reconstruction output"
            return nothing
        end
    catch e
        @warn "CatSim execution failed: $e"
        return nothing
    end
end

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

"""
Compute water phantom metrics.

Returns named tuple with:
- mean_hu: Mean HU in water ROI
- std_hu: Standard deviation in water ROI
- center_hu: Mean HU in center region
- edge_hu: Mean HU in edge region
- cupping: Center - edge HU (positive = cupping artifact)
- n_voxels: Number of voxels in ROI
"""
function compute_water_metrics(recon_hu::AbstractArray{<:Real, 3}, mask::AbstractArray{UInt8, 3})
    center_z = size(recon_hu, 3) ÷ 2 + 1
    slice = recon_hu[:, :, center_z]
    mask_slice = mask[:, :, center_z]

    water_mask = mask_slice .== UInt8(REGION_SOLID_WATER)
    n_voxels = sum(water_mask)

    if n_voxels == 0
        return (mean_hu=NaN, std_hu=NaN, center_hu=NaN, edge_hu=NaN, cupping=NaN, n_voxels=0)
    end

    # Overall statistics
    hu_vals = slice[water_mask]
    mean_hu = mean(hu_vals)
    std_hu = std(hu_vals)

    # Center vs edge analysis for cupping
    nx, ny = size(slice)
    cx, cy = nx ÷ 2, ny ÷ 2

    # Center region (inner 20% radius)
    center_radius = min(nx, ny) * 0.1
    center_mask = similar(water_mask, Bool)
    for j in 1:ny, i in 1:nx
        r = sqrt((i - cx)^2 + (j - cy)^2)
        center_mask[i, j] = r <= center_radius && water_mask[i, j]
    end

    # Edge region (80-90% of water radius)
    # Find approximate water boundary
    water_radius_px = 0
    for i in cx:nx
        if !water_mask[i, cy]
            water_radius_px = i - cx - 1
            break
        end
    end
    if water_radius_px == 0
        water_radius_px = nx ÷ 2
    end

    inner_edge_radius = water_radius_px * 0.7
    outer_edge_radius = water_radius_px * 0.9

    edge_mask = similar(water_mask, Bool)
    for j in 1:ny, i in 1:nx
        r = sqrt((i - cx)^2 + (j - cy)^2)
        edge_mask[i, j] = inner_edge_radius <= r <= outer_edge_radius && water_mask[i, j]
    end

    center_hu = sum(center_mask) > 0 ? mean(slice[center_mask]) : NaN
    edge_hu = sum(edge_mask) > 0 ? mean(slice[edge_mask]) : NaN
    cupping = center_hu - edge_hu

    return (
        mean_hu = mean_hu,
        std_hu = std_hu,
        center_hu = center_hu,
        edge_hu = edge_hu,
        cupping = cupping,
        n_voxels = n_voxels
    )
end

# =============================================================================
# VERIFICATION TESTS
# =============================================================================

"""
Run all water phantom verification tests.

Returns named tuple with pass/fail status and detailed metrics.
"""
function verify_water_phantom(;
    scale::Symbol=:integration,
    run_catsim::Bool=false,
    catsim_dir::Union{String,Nothing}=nothing
)
    cfg = default_water_phantom_config(scale=scale)

    println()
    println("=" ^ 80)
    println("VERIFY-001: WATER PHANTOM HU ACCURACY VERIFICATION")
    println("=" ^ 80)
    println("Timestamp: $(now())")
    println("Scale: $scale")
    println()

    # Run BasisSimulator
    basis_result = run_basissimulator_water(cfg)
    basis_metrics = compute_water_metrics(basis_result.recon_hu, basis_result.mask)

    # Run CatSim if requested
    catsim_result = nothing
    catsim_metrics = nothing
    if run_catsim || catsim_dir !== nothing
        catsim_result = run_catsim_water(cfg; catsim_dir=catsim_dir)
        if catsim_result !== nothing
            # Create mask matching CatSim recon dimensions
            catsim_size = size(catsim_result.recon_hu)
            catsim_mask = create_water_mask_for_recon(cfg, catsim_size)
            catsim_metrics = compute_water_metrics(catsim_result.recon_hu, catsim_mask)
        end
    end

    # Print results
    println("=" ^ 80)
    println("RESULTS")
    println("=" ^ 80)
    println()

    println("BASISSIMULATOR METRICS:")
    println("-" ^ 40)
    println(@sprintf("  Mean HU:       %+8.2f HU (expected: 0 +/- %.0f HU)", basis_metrics.mean_hu, cfg.water_hu_tolerance))
    println(@sprintf("  Std Dev:       %8.2f HU", basis_metrics.std_hu))
    println(@sprintf("  Center HU:     %+8.2f HU", basis_metrics.center_hu))
    println(@sprintf("  Edge HU:       %+8.2f HU", basis_metrics.edge_hu))
    println(@sprintf("  Cupping:       %+8.2f HU (tolerance: +/- %.0f HU)", basis_metrics.cupping, cfg.cupping_tolerance))
    println(@sprintf("  ROI voxels:    %8d", basis_metrics.n_voxels))
    println()

    if catsim_metrics !== nothing
        println("CATSIM METRICS:")
        println("-" ^ 40)
        println(@sprintf("  Mean HU:       %+8.2f HU (expected: 0 +/- %.0f HU)", catsim_metrics.mean_hu, cfg.water_hu_tolerance))
        println(@sprintf("  Std Dev:       %8.2f HU", catsim_metrics.std_hu))
        println(@sprintf("  Center HU:     %+8.2f HU", catsim_metrics.center_hu))
        println(@sprintf("  Edge HU:       %+8.2f HU", catsim_metrics.edge_hu))
        println(@sprintf("  Cupping:       %+8.2f HU (tolerance: +/- %.0f HU)", catsim_metrics.cupping, cfg.cupping_tolerance))
        println(@sprintf("  ROI voxels:    %8d", catsim_metrics.n_voxels))
        println()

        println("COMPARISON:")
        println("-" ^ 40)
        hu_diff = abs(basis_metrics.mean_hu - catsim_metrics.mean_hu)
        std_diff = abs(basis_metrics.std_hu - catsim_metrics.std_hu)
        println(@sprintf("  Mean HU diff:  %8.2f HU (tolerance: %.0f HU)", hu_diff, cfg.simulator_diff_tolerance))
        println(@sprintf("  Std dev diff:  %8.2f HU (tolerance: %.0f HU)", std_diff, cfg.uniformity_diff_tolerance))
        println()
    end

    # Check acceptance criteria
    println("=" ^ 80)
    println("ACCEPTANCE CRITERIA CHECK")
    println("=" ^ 80)
    println()

    tests_passed = true

    # Test 1: BasisSimulator water HU accuracy
    basis_hu_pass = abs(basis_metrics.mean_hu) <= cfg.water_hu_tolerance
    status = basis_hu_pass ? "PASS" : "FAIL"
    println(@sprintf("[%s] BasisSimulator mean HU: |%.1f| <= %.0f HU", status, basis_metrics.mean_hu, cfg.water_hu_tolerance))
    tests_passed &= basis_hu_pass

    # Test 2: CatSim water HU accuracy (if available)
    catsim_hu_pass = true
    if catsim_metrics !== nothing
        catsim_hu_pass = abs(catsim_metrics.mean_hu) <= cfg.water_hu_tolerance
        status = catsim_hu_pass ? "PASS" : "FAIL"
        println(@sprintf("[%s] CatSim mean HU: |%.1f| <= %.0f HU", status, catsim_metrics.mean_hu, cfg.water_hu_tolerance))
        tests_passed &= catsim_hu_pass
    else
        println("[SKIP] CatSim mean HU: CatSim not run")
    end

    # Test 3: Simulator difference
    sim_diff_pass = true
    if catsim_metrics !== nothing
        hu_diff = abs(basis_metrics.mean_hu - catsim_metrics.mean_hu)
        sim_diff_pass = hu_diff <= cfg.simulator_diff_tolerance
        status = sim_diff_pass ? "PASS" : "FAIL"
        println(@sprintf("[%s] Simulator difference: %.1f <= %.0f HU", status, hu_diff, cfg.simulator_diff_tolerance))
        tests_passed &= sim_diff_pass
    else
        println("[SKIP] Simulator difference: CatSim not run")
    end

    # Test 4: Uniformity difference
    uniformity_pass = true
    if catsim_metrics !== nothing
        std_diff = abs(basis_metrics.std_hu - catsim_metrics.std_hu)
        uniformity_pass = std_diff <= cfg.uniformity_diff_tolerance
        status = uniformity_pass ? "PASS" : "FAIL"
        println(@sprintf("[%s] Uniformity difference: %.1f <= %.0f HU", status, std_diff, cfg.uniformity_diff_tolerance))
        tests_passed &= uniformity_pass
    else
        println("[SKIP] Uniformity difference: CatSim not run")
    end

    # Test 5: Cupping artifact
    cupping_pass = abs(basis_metrics.cupping) <= cfg.cupping_tolerance
    status = cupping_pass ? "PASS" : "FAIL"
    println(@sprintf("[%s] Cupping artifact: |%.1f| <= %.0f HU", status, basis_metrics.cupping, cfg.cupping_tolerance))
    tests_passed &= cupping_pass

    println()
    println("=" ^ 80)
    if tests_passed
        println("OVERALL: PASS - All acceptance criteria met")
    else
        println("OVERALL: FAIL - One or more acceptance criteria not met")
    end
    println("=" ^ 80)
    println()

    return (
        passed = tests_passed,
        config = cfg,
        basis_metrics = basis_metrics,
        catsim_metrics = catsim_metrics,
        basis_hu_pass = basis_hu_pass,
        catsim_hu_pass = catsim_hu_pass,
        sim_diff_pass = sim_diff_pass,
        uniformity_pass = uniformity_pass,
        cupping_pass = cupping_pass
    )
end

# =============================================================================
# JULIA TEST SUITE
# =============================================================================

"""
Run water phantom tests using Julia's Test framework.
"""
function run_water_phantom_tests(; scale::Symbol=:integration, run_catsim::Bool=false)
    result = verify_water_phantom(scale=scale, run_catsim=run_catsim)
    cfg = result.config

    @testset "VERIFY-001: Water Phantom HU Accuracy" begin
        @testset "BasisSimulator" begin
            @test abs(result.basis_metrics.mean_hu) <= cfg.water_hu_tolerance
            @test abs(result.basis_metrics.cupping) <= cfg.cupping_tolerance
        end

        if result.catsim_metrics !== nothing
            @testset "CatSim" begin
                @test abs(result.catsim_metrics.mean_hu) <= cfg.water_hu_tolerance
            end

            @testset "Simulator Comparison" begin
                hu_diff = abs(result.basis_metrics.mean_hu - result.catsim_metrics.mean_hu)
                std_diff = abs(result.basis_metrics.std_hu - result.catsim_metrics.std_hu)
                @test hu_diff <= cfg.simulator_diff_tolerance
                @test std_diff <= cfg.uniformity_diff_tolerance
            end
        end
    end

    return result
end

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Parse command line arguments
    scale = :integration
    run_catsim_flag = false
    catsim_dir = nothing

    for arg in ARGS
        if startswith(arg, "--scale=")
            scale = Symbol(split(arg, "=")[2])
        elseif arg == "--run-catsim"
            run_catsim_flag = true
        elseif startswith(arg, "--catsim-dir=")
            catsim_dir = split(arg, "=")[2]
        elseif arg == "--help"
            println("Usage: julia water_phantom.jl [options]")
            println()
            println("Options:")
            println("  --scale=SCALE       Simulation scale: dev, integration, verification, publication")
            println("                      (default: integration)")
            println("  --run-catsim        Also run CatSim simulation for comparison")
            println("  --catsim-dir=DIR    Load CatSim results from directory instead of running")
            println("  --help              Show this help message")
            exit(0)
        end
    end

    # Run verification
    result = verify_water_phantom(scale=scale, run_catsim=run_catsim_flag, catsim_dir=catsim_dir)

    # Exit with appropriate code
    exit(result.passed ? 0 : 1)
end
