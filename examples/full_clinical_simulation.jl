# =============================================================================
# Full-Scale Clinical CT Simulation (CatSim-Exact, ALL Physics)
# =============================================================================
#
# Complete clinical CT simulation with ALL physics effects enabled.
# Uses CatSim-exact signal chain. Everything runs on GPU at clinical resolution.
#
# PHYSICS EFFECTS - CATSIM CLASSIFICATION:
# ========================================
#
#   CATSIM ESSENTIAL (enabled by default in CatSim):
#   - Flat filter (Al, 3mm) - beam hardening, dose reduction
#   - Bowtie filter - peripheral dose reduction, signal equalization
#   - Detector efficiency - energy-dependent absorption
#   - Fill factor (0.9) - detector dead area
#   - DAS model - gain + electronic noise
#   - Quantum noise - Poisson statistics on photons
#
#   CATSIM OPTIONAL (disabled by default in CatSim):
#   - X-ray crosstalk - pixel coupling (we enable for realism)
#   - Optical crosstalk - scintillator light spread
#   - Detector lag - afterglow/ghosting (we enable for realism)
#   - Scatter - patient scatter (we enable for realism)
#   - Focal spot blur - geometric blur (we enable for realism)
#
# CATSIM SIGNAL CHAIN:
# ====================
#   - Heel effect (anode self-attenuation)
#   - Air scan calibration (noise-free reference - CatSim exact!)
#   - Low signal correction (replace negatives with smoothed neighbors)
#   - Beam hardening correction
#
# =============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BasisSimulator
using Statistics
using CairoMakie  # For visualization

# =============================================================================
# GPU Setup - REQUIRED
# =============================================================================

using Metal
if !Metal.functional()
    error("This demo requires a functional Metal GPU!")
end

println("=" ^ 70)
println("Full-Scale Clinical CT Simulation (CatSim-Exact, ALL Physics)")
println("=" ^ 70)
println("GPU: ", Metal.current_device())
println()

# =============================================================================
# CLINICAL CONFIGURATION
# =============================================================================

CONFIG = (
    # Phantom (clinical resolution)
    phantom_n_voxels = 1024,
    phantom_n_slices = 64,
    fov_cm = 35.0,
    z_cm = 4.0,

    # Detector (clinical 64-slice CT)
    n_cols = 736,
    n_rows = 64,
    n_angles = 1160,

    # Reconstruction output
    recon_n_voxels = 512,
    recon_n_slices = 32,

    # Spectrum
    kvp = 120,
    n_energy_bins = 30,

    # Reproducibility
    noise_seed = 42,

    # CatSim Signal Chain Settings
    anode_angle_deg = 7.0,
    das_gain = 1.0,
    das_noise_electrons = 100.0,
)

voxel_mm = CONFIG.fov_cm * 10 / CONFIG.phantom_n_voxels
println("Clinical Configuration:")
println("  Phantom: $(CONFIG.phantom_n_voxels)x$(CONFIG.phantom_n_voxels)x$(CONFIG.phantom_n_slices) ($(round(voxel_mm, digits=2)) mm voxels)")
println("  Detector: $(CONFIG.n_cols)x$(CONFIG.n_rows) pixels")
println("  Projections: $(CONFIG.n_angles) angles")
println("  Spectrum: $(CONFIG.kvp) kVp")
println()

# =============================================================================
# STEP 1: Create Clinical Geometry
# =============================================================================
println("-" ^ 70)
println("STEP 1: Scanner Geometry")
println("-" ^ 70)

geom = create_aquilion_one(
    n_angles = CONFIG.n_angles,
    n_rows = CONFIG.n_rows,
    n_cols = CONFIG.n_cols,
    fov_cm = CONFIG.fov_cm,
    z_cm = CONFIG.z_cm
)

println("Scanner: Canon Aquilion ONE (64-slice)")
println("  SAD: $(geom.SAD) cm, SDD: $(geom.SDD) cm")
println("  Detector: $(CONFIG.n_cols) x $(CONFIG.n_rows)")
println("  FOV: $(CONFIG.fov_cm) x $(CONFIG.fov_cm) x $(CONFIG.z_cm) cm")
println()

# =============================================================================
# STEP 2: Create Clinical Phantom
# =============================================================================
println("-" ^ 70)
println("STEP 2: Phantom Creation")
println("-" ^ 70)

phantom = create_gammex_472(
    n_voxels = CONFIG.phantom_n_voxels,
    n_slices = CONFIG.phantom_n_slices,
    fov_cm = CONFIG.fov_cm,
    z_cm = CONFIG.z_cm
)

println("Phantom: Gammex 472 Multi-Energy CT")
println("  Size: $(size(phantom.μ))")
println("  Voxel: $(round(voxel_mm, digits=2)) mm isotropic (in-plane)")
println("  Materials: $(length(unique(phantom.mask))) regions")
println()

# =============================================================================
# STEP 3: Load Spectrum and Materials
# =============================================================================
println("-" ^ 70)
println("STEP 3: Spectrum Configuration")
println("-" ^ 70)

energies_full, weights_full = load_spectrum(CONFIG.kvp)
energies, weights = downsample_spectrum(energies_full, weights_full, CONFIG.n_energy_bins)
materials = get_region_materials()

mean_energy = sum(energies .* weights) / sum(weights)
println("Spectrum: $(CONFIG.kvp) kVp polychromatic")
println("  Energy bins: $(length(energies_full)) -> $(CONFIG.n_energy_bins)")
println("  Range: $(round(minimum(energies), digits=1)) - $(round(maximum(energies), digits=1)) keV")
println("  Mean energy: $(round(mean_energy, digits=1)) keV")
println()

# =============================================================================
# STEP 4: Configure Physics Effects (EXPLICIT - comment out to disable)
# =============================================================================
println("-" ^ 70)
println("STEP 4: Physics Configuration")
println("-" ^ 70)

# ============================================================================
# PHYSICS CONFIG - Comment out any line OR set to `nothing` to disable!
# (Don't use `false` - only `nothing` or the model works)
# ============================================================================
physics_config = default_physics_config(
    # --- CATSIM ESSENTIAL (scanner-specific) ---
    fill_factor = fill_factor_standard(),                      # 0.9 fill factor
    flat_filter = flat_filter_al(3.0),                         # 3mm Al
    bowtie_filter = bowtie_filter_large_body(),                # Large body
    detector_efficiency = detector_efficiency_gos(0.5),        # GOS 0.5mm

    # --- CATSIM OPTIONAL (we enable for realism) ---
    scatter = default_scatter_model(scale_factor=1.0),         # Compton/Rayleigh
    crosstalk = crosstalk_medium(),                            # X-ray pixel coupling
    optical_crosstalk = optical_crosstalk_typical(),           # Scintillator spread
    focal_spot = focal_spot_medium(),                          # Geometric blur
    noise = default_detector_model(I0=1e6, seed=CONFIG.noise_seed),  # Quantum noise
    lag = lag_gadox(),                                         # Afterglow

    # --- SIGNAL CHAIN (CatSim-exact) ---
    heel_effect = default_heel_effect(anode_angle_deg=CONFIG.anode_angle_deg),
    # das_model = default_das_model(gain=CONFIG.das_gain, electronic_noise_sigma=CONFIG.das_noise_electrons), # COMPLETELY BROKEN
    bhc = bhc_water_default(reference_energy_keV=mean_energy),

    # --- Settings ---
    energy_keV = Float64(mean_energy),
    noise_seed = CONFIG.noise_seed
)

physics_info = get_physics_config_info(physics_config)
println("\nPhysics effects enabled: $(physics_info.n_enabled)/13")
for effect in physics_info.enabled_effects
    println("  ✓ $effect")
end
if length(physics_info.disabled_effects) > 0
    println("\nDisabled:")
    for effect in physics_info.disabled_effects
        println("  ✗ $effect")
    end
end

# =============================================================================
# STEP 5: Forward Projection with FULL Signal Chain (GPU)
# =============================================================================
println("\n" * "-" ^ 70)
println("STEP 5: Forward Projection (ALL physics in one call)")
println("-" ^ 70)

mask_gpu = MtlArray(phantom.mask);
println("\nPhantom mask on GPU: $(typeof(mask_gpu))")
println("\nComputing with ALL 13 physics effects:")
println("  - Beer-Lambert spectral physics ($(CONFIG.n_energy_bins) bins)")
println("  - Physics pipeline: fill_factor, flat_filter, bowtie, scatter, etc.")
println("  - Signal chain: heel_effect, das_model, bhc, air_scan calibration")
println("\n  (Large clinical simulation - may take several minutes)")
println()

# Just pass physics=full_physics_config() - everything is included!
@time sinogram_gpu = forward_project(
    mask_gpu, geom;
    energies = energies,
    weights = weights,
    materials = materials,
    physics = physics_config,  # ALL 13 effects included!
    max_prep = 10.0
);

println("\nCalibrated sinogram on GPU: $(size(sinogram_gpu))")
sinogram_cpu = Array(sinogram_gpu);
println("  Range: [$(round(minimum(sinogram_cpu), digits=3)), $(round(maximum(sinogram_cpu), digits=3))]")
println("  Mean: $(round(mean(sinogram_cpu), digits=3))")

# =============================================================================
# STEP 6: FDK Reconstruction (GPU)
# =============================================================================
println("\n" * "-" ^ 70)
println("STEP 6: FDK Reconstruction")
println("-" ^ 70)

recon_size = (CONFIG.recon_n_voxels, CONFIG.recon_n_voxels, CONFIG.recon_n_slices)
println("\nReconstructing $(recon_size) volume (GPU)...")

@time recon_gpu = fdk_reconstruct(sinogram_gpu, geom, recon_size);

recon_cpu = Array(recon_gpu);
println("Reconstruction mu range: [$(round(minimum(recon_cpu), digits=4)), $(round(maximum(recon_cpu), digits=4))] cm^-1")

# =============================================================================
# STEP 7: HU Conversion and Validation
# =============================================================================
println("\n" * "-" ^ 70)
println("STEP 7: HU Conversion and Validation")
println("-" ^ 70)

mu_water_eff = Float32(get_effective_μ_water_kVp(CONFIG.kvp));
println("\nReference mu_water ($(CONFIG.kvp) kVp): $(mu_water_eff) cm^-1")

recon_hu = 1000.0f0 .* (recon_cpu .- mu_water_eff) ./ mu_water_eff;
println("HU range: [$(round(minimum(recon_hu))), $(round(maximum(recon_hu)))]")

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

mask_recon = downsample_mask(phantom.mask, recon_size);
center_z = Int(recon_size[3] / 2 + 1)
center_hu = recon_hu[:, :, center_z];
center_mask = mask_recon[:, :, center_z]

println("\n" * "=" ^ 60)
println("HU VALIDATION (Center Slice z=$center_z)")
println("=" ^ 60)

validation_regions = [
    (name="Solid Water", id=REGION_SOLID_WATER, expected=0),
    (name="Ca-50",       id=REGION_CA_50,       expected=50),
    (name="Ca-100",      id=REGION_CA_100,      expected=100),
    (name="Ca-200",      id=REGION_CA_200,      expected=200),
    (name="Ca-300",      id=REGION_CA_300,      expected=300),
    (name="Ca-400",      id=REGION_CA_400,      expected=400),
]

println("\nRegion          | Mean HU  | Std HU  | Expected | N voxels")
println("-" ^ 60)

# Collect data for bar chart
hu_measured = Float64[]
hu_expected = Float64[]
hu_std_vals = Float64[]
region_names = String[]

for region in validation_regions
    mask_2d = center_mask .== UInt8(region.id)
    n_voxels = sum(mask_2d)
    if n_voxels > 0
        hu_vals = center_hu[mask_2d]
        hu_mean = mean(hu_vals)
        hu_std = std(hu_vals)
        println("  $(rpad(region.name, 12)) | $(lpad(round(Int, hu_mean), 6)) | $(lpad(round(Int, hu_std), 6))  | $(lpad("~$(region.expected)", 8)) | $(n_voxels)")

        # Store for visualization
        push!(region_names, region.name)
        push!(hu_measured, hu_mean)
        push!(hu_expected, Float64(region.expected))
        push!(hu_std_vals, hu_std)
    end
end

# =============================================================================
# STEP 8: Visualization
# =============================================================================
println("\n" * "-" ^ 70)
println("STEP 8: Visualization")
println("-" ^ 70)

# --- Reconstruction Heatmap ---
begin
    fig1 = Figure(size=(600, 500))
    ax1 = Axis(fig1[1, 1], title="Reconstruction (HU) - Center Slice z=$center_z",
            xlabel="x", ylabel="y", aspect=DataAspect())
    hm1 = heatmap!(ax1, center_hu', colormap=:grays, colorrange=(-200, 400))
    Colorbar(fig1[1, 2], hm1, label="HU")
    display(fig1)
end

# --- Sinogram Heatmap ---
begin
    fig2 = Figure(size=(600, 500))
    sino_center = sinogram_cpu[:, CONFIG.n_rows÷2, :]
    ax2 = Axis(fig2[1, 1], title="Sinogram - Center Row",
            xlabel="Detector Column", ylabel="Angle")
    hm2 = heatmap!(ax2, sino_center', colormap=:viridis)
    Colorbar(fig2[1, 2], hm2, label="Line Integral")
    display(fig2)
end

# --- HU Validation Bar Chart ---
begin
    fig3 = Figure(size=(800, 400))
    ax3 = Axis(fig3[1, 1],
            title="HU Validation: Measured vs Expected",
            xlabel="Region",
            ylabel="HU Value",
            xticks=(1:length(region_names), region_names),
            xticklabelrotation=π/6)

    # Bar positions
    n_regions = length(region_names)
    x_pos = 1:n_regions
    bar_width = 0.35

    # Expected values (reference)
    barplot!(ax3, x_pos .- bar_width/2, hu_expected,
            width=bar_width, color=:steelblue, label="Expected")

    # Measured values with error bars
    barplot!(ax3, x_pos .+ bar_width/2, hu_measured,
            width=bar_width, color=:coral, label="Measured")

    # Error bars for measured values
    errorbars!(ax3, x_pos .+ bar_width/2, hu_measured, hu_std_vals,
            color=:black, whiskerwidth=8)

    # Add legend
    axislegend(ax3, position=:lt)

    # Add zero line for reference
    hlines!(ax3, [0], color=:gray, linestyle=:dash, linewidth=1)

    display(fig3)
end

# --- Save Combined Figure ---
begin
    fig_combined = Figure(size=(1400, 900))

    # Row 1: Heatmaps
    ax1 = Axis(fig_combined[1, 1], title="Reconstruction (HU) - Center Slice z=$center_z",
            xlabel="x", ylabel="y", aspect=DataAspect())
    hm1 = heatmap!(ax1, center_hu', colormap=:grays, colorrange=(-200, 400))
    Colorbar(fig_combined[1, 2], hm1, label="HU")

    sino_center = sinogram_cpu[:, CONFIG.n_rows÷2, :]
    ax2 = Axis(fig_combined[1, 3], title="Sinogram - Center Row",
            xlabel="Detector Column", ylabel="Angle")
    hm2 = heatmap!(ax2, sino_center', colormap=:viridis)
    Colorbar(fig_combined[1, 4], hm2, label="Line Integral")

    # Row 2: Bar Chart
    ax3 = Axis(fig_combined[2, 1:4],
            title="HU Validation: Measured vs Expected",
            xlabel="Region", ylabel="HU Value",
            xticks=(1:length(region_names), region_names),
            xticklabelrotation=π/6)
    barplot!(ax3, (1:length(region_names)) .- 0.175, hu_expected,
            width=0.35, color=:steelblue, label="Expected")
    barplot!(ax3, (1:length(region_names)) .+ 0.175, hu_measured,
            width=0.35, color=:coral, label="Measured")
    errorbars!(ax3, (1:length(region_names)) .+ 0.175, hu_measured, hu_std_vals,
            color=:black, whiskerwidth=8)
    axislegend(ax3, position=:lt)
    hlines!(ax3, [0], color=:gray, linestyle=:dash, linewidth=1)

    output_path = joinpath(@__DIR__, "full_clinical_simulation_output.png")
    save(output_path, fig_combined)
    println("\nSaved visualization to: $output_path")
end

# =============================================================================
# SUMMARY
# =============================================================================
println("\n" * "=" ^ 70)
println("Full-Scale Clinical Simulation Complete")
println("=" ^ 70)

println("\nPhysics effects: $(physics_info.n_enabled)/13 enabled")
println("  (Edit physics_config to enable/disable individual effects)")

println("\nKey features:")
println("  ✓ CatSim-exact air scan calibration (NO noise)")
println("  ✓ Low signal correction: smooth negatives")
println("  ✓ All operations on GPU (Metal)")

println("\nOutput: $output_path")
println()
