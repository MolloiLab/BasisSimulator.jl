# =============================================================================
# Clinical Dose Comparison Example
# =============================================================================
#
# This example demonstrates CT simulation at three clinical dose levels:
# - LOW: 80 kVp (lower dose, more noise)
# - STANDARD: 120 kVp (typical clinical scan)
# - HIGH: 140 kVp (higher dose, less noise)
#
# Key features:
# - SIRT reconstruction with exactly 3 iterations (as per EXAMPLE-FIX spec)
# - Grouped bar plots showing Expected vs Measured HU for Ca and I series
# - Metal GPU verification for all operations
# - Demonstration of noise relationship: σ_low > σ_std > σ_high
#
# =============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BasisSimulator
using Statistics
using CairoMakie
import XrayAttenuation as XA

# =============================================================================
# GPU Setup - REQUIRED
# =============================================================================

using Metal

# Verify Metal GPU is functional and print device info
@assert Metal.functional() "Metal GPU required for this example"
println("=" ^ 70)
println("Clinical Dose Comparison Example")
println("=" ^ 70)
println("GPU: $(Metal.current_device())")
println("Metal.functional(): $(Metal.functional())")
println()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Three dose levels with different kVp settings
DOSE_LEVELS = [
    (name = "LOW",      kvp = 80,  I0 = 5e5,  color = :steelblue),
    (name = "STANDARD", kvp = 120, I0 = 1e6,  color = :forestgreen),
    (name = "HIGH",     kvp = 140, I0 = 2e6,  color = :coral),
]

CONFIG = (
    # Phantom (moderate resolution for reasonable runtime)
    phantom_n_voxels = 256,
    phantom_n_slices = 16,
    fov_cm = 35.0,
    z_cm = 2.0,

    # Detector (clinical 64-slice CT)
    n_cols = 512,
    n_rows = 32,
    n_angles = 360,

    # Reconstruction output
    recon_n_voxels = 256,
    recon_n_slices = 16,

    # Spectrum
    n_energy_bins = 30,

    # SIRT reconstruction: exactly 3 iterations (per acceptance criteria)
    sirt_iterations = 3,

    # Reproducibility
    noise_seed = 42,
)

println("Configuration:")
println("  Phantom: $(CONFIG.phantom_n_voxels)³ × $(CONFIG.phantom_n_slices) slices")
println("  Detector: $(CONFIG.n_cols) × $(CONFIG.n_rows)")
println("  Projections: $(CONFIG.n_angles) angles")
println("  SIRT iterations: $(CONFIG.sirt_iterations)")
println()

# =============================================================================
# STEP 1: Create Geometry and Phantom (shared across all dose levels)
# =============================================================================
println("-" ^ 70)
println("STEP 1: Create Scanner Geometry and Phantom")
println("-" ^ 70)

geom = create_aquilion_one(
    n_angles = CONFIG.n_angles,
    n_rows = CONFIG.n_rows,
    n_cols = CONFIG.n_cols,
    fov_cm = CONFIG.fov_cm,
    z_cm = CONFIG.z_cm
)

println("Scanner Geometry:")
println("  SAD: $(geom.SAD) cm, SDD: $(geom.SDD) cm")
println("  Detector: $(geom.n_cols) × $(geom.n_rows)")
println()

phantom = create_gammex_472(
    n_voxels = CONFIG.phantom_n_voxels,
    n_slices = CONFIG.phantom_n_slices,
    fov_cm = CONFIG.fov_cm,
    z_cm = CONFIG.z_cm
)

println("Phantom: Gammex 472 Multi-Energy CT")
println("  Size: $(size(phantom.μ))")
println("  Materials: $(length(unique(phantom.mask))) regions")
println()

# Transfer phantom mask to GPU and verify
mask_gpu = MtlArray(phantom.mask)
println("Phantom mask transferred to GPU: $(typeof(mask_gpu))")
@assert mask_gpu isa MtlArray "Phantom mask must be on Metal GPU"
println()

# =============================================================================
# STEP 2: Define materials and regions for validation
# =============================================================================

# Calcium series (inner ring of Gammex 472)
ca_regions = [
    (name = "Ca-50",  id = REGION_CA_50,  symbol = :Ca_50),
    (name = "Ca-100", id = REGION_CA_100, symbol = :Ca_100),
    (name = "Ca-200", id = REGION_CA_200, symbol = :Ca_200),
    (name = "Ca-300", id = REGION_CA_300, symbol = :Ca_300),
    (name = "Ca-400", id = REGION_CA_400, symbol = :Ca_400),
    (name = "Ca-500", id = REGION_CA_500, symbol = :Ca_500),
    (name = "Ca-600", id = REGION_CA_600, symbol = :Ca_600),
]

# Iodine series (outer ring of Gammex 472)
iodine_regions = [
    (name = "I-2.0",  id = REGION_I_2_0,  symbol = :I_2_0),
    (name = "I-2.5",  id = REGION_I_2_5,  symbol = :I_2_5),
    (name = "I-5.0",  id = REGION_I_5_0,  symbol = :I_5_0),
    (name = "I-7.5",  id = REGION_I_7_5,  symbol = :I_7_5),
    (name = "I-10.0", id = REGION_I_10_0, symbol = :I_10_0),
    (name = "I-15.0", id = REGION_I_15_0, symbol = :I_15_0),
    (name = "I-20.0", id = REGION_I_20_0, symbol = :I_20_0),
]

# =============================================================================
# STEP 3: Helper functions
# =============================================================================

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

"""
Compute expected HU values for materials at a given kVp using NIST data.
Returns dictionaries mapping region names to expected HU values.
"""
function compute_expected_hu(kvp::Int; n_bins::Int=30)
    energies, weights = load_spectrum(kvp)
    energies, weights = downsample_spectrum(energies, weights, n_bins)

    # Compute μ_water for this spectrum
    μ_water = compute_effective_μ_material(XA.Materials.water, energies, weights)

    ca_expected = Dict{String, Float64}()
    iodine_expected = Dict{String, Float64}()

    for region in ca_regions
        mat = get_material(region.symbol)
        μ_mat = compute_effective_μ_material(mat, energies, weights)
        ca_expected[region.name] = 1000.0 * (μ_mat - μ_water) / μ_water
    end

    for region in iodine_regions
        mat = get_material(region.symbol)
        μ_mat = compute_effective_μ_material(mat, energies, weights)
        iodine_expected[region.name] = 1000.0 * (μ_mat - μ_water) / μ_water
    end

    return ca_expected, iodine_expected
end

"""
Measure HU values from reconstruction for all regions.
"""
function measure_hu_values(recon_hu, mask_recon, center_z)
    center_mask = mask_recon[:, :, center_z]
    center_hu = recon_hu[:, :, center_z]

    ca_measured = Dict{String, Tuple{Float64, Float64}}()  # (mean, std)
    iodine_measured = Dict{String, Tuple{Float64, Float64}}()

    for region in ca_regions
        region_mask = center_mask .== UInt8(region.id)
        if sum(region_mask) > 0
            vals = center_hu[region_mask]
            ca_measured[region.name] = (mean(vals), std(vals))
        end
    end

    for region in iodine_regions
        region_mask = center_mask .== UInt8(region.id)
        if sum(region_mask) > 0
            vals = center_hu[region_mask]
            iodine_measured[region.name] = (mean(vals), std(vals))
        end
    end

    return ca_measured, iodine_measured
end

# =============================================================================
# STEP 4: Run simulation at each dose level
# =============================================================================

results = []
recon_size = (CONFIG.recon_n_voxels, CONFIG.recon_n_voxels, CONFIG.recon_n_slices)
mask_recon = downsample_mask(phantom.mask, recon_size)
center_z = Int(recon_size[3] / 2 + 1)

for dose in DOSE_LEVELS
    println()
    println("-" ^ 70)
    println("Processing $(dose.name) dose ($(dose.kvp) kVp)")
    println("-" ^ 70)

    # Load spectrum for this kVp
    energies, weights = load_spectrum(dose.kvp)
    energies, weights = downsample_spectrum(energies, weights, CONFIG.n_energy_bins)
    materials = get_region_materials()

    mean_energy = sum(energies .* weights) / sum(weights)
    println("  Spectrum: $(dose.kvp) kVp, mean energy $(round(mean_energy, digits=1)) keV")

    # Configure physics with noise model
    physics_config = default_physics_config(
        fill_factor = fill_factor_standard(),
        flat_filter = flat_filter_al(3.0),
        bowtie_filter = bowtie_filter_large_body(),
        noise = default_detector_model(I0=dose.I0, seed=CONFIG.noise_seed),
        energy_keV = Float64(mean_energy),
        noise_seed = CONFIG.noise_seed
    )

    # Forward projection on GPU
    println("  Forward projection (GPU)...")
    @assert mask_gpu isa MtlArray "Forward projection input must be on Metal GPU"

    @time sinogram_gpu = forward_project(
        mask_gpu, geom;
        energies = energies,
        weights = weights,
        materials = materials,
        physics = physics_config
    )

    @assert sinogram_gpu isa MtlArray "Sinogram must be on Metal GPU"
    println("  Sinogram on GPU: $(typeof(sinogram_gpu))")

    # SIRT reconstruction with exactly 3 iterations (per acceptance criteria)
    println("  SIRT reconstruction ($(CONFIG.sirt_iterations) iterations, GPU)...")
    @assert sinogram_gpu isa MtlArray "SIRT input sinogram must be on Metal GPU"

    @time recon_sirt_gpu = sirt_reconstruct(sinogram_gpu, geom, recon_size; niter=CONFIG.sirt_iterations)

    @assert recon_sirt_gpu isa MtlArray "SIRT reconstruction must be on Metal GPU"
    println("  SIRT reconstruction on GPU: $(typeof(recon_sirt_gpu))")

    recon_cpu = Array(recon_sirt_gpu)

    # HU conversion using empirical water calibration
    water_mask = mask_recon[:, :, center_z] .== UInt8(REGION_SOLID_WATER)
    μ_water_empirical = mean(recon_cpu[:, :, center_z][water_mask])

    recon_hu = 1000.0f0 .* (recon_cpu .- μ_water_empirical) ./ μ_water_empirical

    # Measure HU and noise
    ca_measured, iodine_measured = measure_hu_values(recon_hu, mask_recon, center_z)
    ca_expected, iodine_expected = compute_expected_hu(dose.kvp; n_bins=CONFIG.n_energy_bins)

    # Measure noise (std in water region)
    water_vals = recon_hu[:, :, center_z][water_mask]
    noise_std = std(water_vals)

    println("  Water HU: $(round(mean(water_vals), digits=1)) ± $(round(noise_std, digits=1)) HU")

    push!(results, (
        dose = dose,
        recon_hu = recon_hu,
        ca_measured = ca_measured,
        iodine_measured = iodine_measured,
        ca_expected = ca_expected,
        iodine_expected = iodine_expected,
        noise_std = noise_std,
    ))
end

# =============================================================================
# STEP 5: Verify noise relationship
# =============================================================================
println()
println("-" ^ 70)
println("Noise Relationship Verification")
println("-" ^ 70)

noise_values = [(r.dose.name, r.dose.kvp, r.noise_std) for r in results]
println()
println("Dose Level    | kVp  | Noise σ (HU)")
println("-" ^ 40)
for (name, kvp, σ) in noise_values
    println("  $(rpad(name, 10)) | $(lpad(kvp, 3)) | $(lpad(round(σ, digits=1), 8))")
end

# Verify: σ_low > σ_std > σ_high
σ_low = results[1].noise_std
σ_std = results[2].noise_std
σ_high = results[3].noise_std

if σ_low > σ_std > σ_high
    println()
    println("✓ Noise relationship verified: σ_LOW ($(round(σ_low, digits=1))) > σ_STD ($(round(σ_std, digits=1))) > σ_HIGH ($(round(σ_high, digits=1)))")
else
    println()
    println("⚠ Noise relationship: σ_LOW=$(round(σ_low, digits=1)), σ_STD=$(round(σ_std, digits=1)), σ_HIGH=$(round(σ_high, digits=1))")
end

# =============================================================================
# STEP 6: Create visualization with grouped bar plots
# =============================================================================
println()
println("-" ^ 70)
println("Creating Visualization")
println("-" ^ 70)

fig = Figure(size=(1600, 1200), fontsize=12)

# Use STANDARD (120 kVp) for the main displays
std_result = results[2]

# --- Row 1: Reconstruction images for each dose level ---
for (i, r) in enumerate(results)
    ax = Axis(fig[1, i],
        title="$(r.dose.name) ($(r.dose.kvp) kVp)\nσ = $(round(r.noise_std, digits=1)) HU",
        aspect=DataAspect())
    hm = heatmap!(ax, r.recon_hu[:, :, center_z]', colormap=:grays, colorrange=(-200, 800))
    if i == 3
        Colorbar(fig[1, 4], hm, label="HU")
    end
end

# --- Row 2: Calcium series - Expected vs Measured HU (grouped bar plot) ---
ax_ca = Axis(fig[2, 1:2],
    title="Calcium Series: Expected vs Measured HU (120 kVp)",
    xlabel="Material",
    ylabel="HU Value",
    xticks=(1:7, [r.name for r in ca_regions]),
    xticklabelrotation=π/6)

ca_names = [r.name for r in ca_regions]
ca_exp_vals = [std_result.ca_expected[n] for n in ca_names]
ca_meas_vals = [std_result.ca_measured[n][1] for n in ca_names]
ca_meas_std = [std_result.ca_measured[n][2] for n in ca_names]

x_pos = 1:7
bar_width = 0.35

barplot!(ax_ca, x_pos .- bar_width/2, ca_exp_vals, width=bar_width, color=:steelblue, label="Expected (NIST)")
barplot!(ax_ca, x_pos .+ bar_width/2, ca_meas_vals, width=bar_width, color=:coral, label="Measured (SIRT)")
errorbars!(ax_ca, x_pos .+ bar_width/2, ca_meas_vals, ca_meas_std, color=:black, whiskerwidth=6)
axislegend(ax_ca, position=:lt)

# --- Row 2: Iodine series - Expected vs Measured HU (grouped bar plot) ---
ax_i = Axis(fig[2, 3:4],
    title="Iodine Series: Expected vs Measured HU (120 kVp)",
    xlabel="Material",
    ylabel="HU Value",
    xticks=(1:7, [r.name for r in iodine_regions]),
    xticklabelrotation=π/6)

i_names = [r.name for r in iodine_regions]
i_exp_vals = [std_result.iodine_expected[n] for n in i_names]
i_meas_vals = [std_result.iodine_measured[n][1] for n in i_names]
i_meas_std = [std_result.iodine_measured[n][2] for n in i_names]

barplot!(ax_i, x_pos .- bar_width/2, i_exp_vals, width=bar_width, color=:steelblue, label="Expected (NIST)")
barplot!(ax_i, x_pos .+ bar_width/2, i_meas_vals, width=bar_width, color=:coral, label="Measured (SIRT)")
errorbars!(ax_i, x_pos .+ bar_width/2, i_meas_vals, i_meas_std, color=:black, whiskerwidth=6)
axislegend(ax_i, position=:lt)

# --- Row 3: Noise comparison across dose levels ---
ax_noise = Axis(fig[3, 1:2],
    title="Noise vs Dose Level",
    xlabel="Dose Level",
    ylabel="Noise σ (HU)",
    xticks=(1:3, [d.name for d in DOSE_LEVELS]))

noise_vals = [r.noise_std for r in results]
colors = [d.color for d in DOSE_LEVELS]
barplot!(ax_noise, 1:3, noise_vals, color=colors)

# Add value labels on bars
for (i, n) in enumerate(noise_vals)
    text!(ax_noise, i, n + 2, text="$(round(n, digits=1))", align=(:center, :bottom), fontsize=12)
end

# --- Row 3: HU linearity plot ---
ax_linear = Axis(fig[3, 3:4],
    title="HU Linearity (120 kVp)",
    xlabel="Expected HU (NIST)",
    ylabel="Measured HU (SIRT)")

# Combine Ca and I data
all_expected = vcat(ca_exp_vals, i_exp_vals)
all_measured = vcat(ca_meas_vals, i_meas_vals)

scatter!(ax_linear, ca_exp_vals, ca_meas_vals, color=:blue, markersize=12, label="Calcium")
scatter!(ax_linear, i_exp_vals, i_meas_vals, color=:red, markersize=12, label="Iodine")

# Add identity line
min_val = minimum(all_expected)
max_val = maximum(all_expected)
lines!(ax_linear, [min_val, max_val], [min_val, max_val], color=:gray, linestyle=:dash, label="y=x")

axislegend(ax_linear, position=:lt)

# --- Add title and save ---
Label(fig[0, :], "Clinical Dose Comparison: BasisSimulator.jl", fontsize=20, font=:bold)

output_path = joinpath(@__DIR__, "clinical_dose_comparison_output.png")
save(output_path, fig, px_per_unit=2)  # 300 DPI equivalent
println("Saved visualization to: $output_path")

# =============================================================================
# SUMMARY
# =============================================================================
println()
println("=" ^ 70)
println("Clinical Dose Comparison Complete")
println("=" ^ 70)
println()
println("Key Results:")
println("  - Three dose levels simulated: LOW (80 kVp), STANDARD (120 kVp), HIGH (140 kVp)")
println("  - SIRT reconstruction with $(CONFIG.sirt_iterations) iterations")
println("  - All forward projection on Metal GPU: ✓")
println("  - All reconstruction on Metal GPU: ✓")
println("  - Noise relationship σ_LOW > σ_STD > σ_HIGH: $(σ_low > σ_std > σ_high ? "✓" : "partial")")
println()
println("Output: $output_path")
println()
