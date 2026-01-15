# =============================================================================
# Full CatSim-Style CT Imaging Chain
# =============================================================================
#
# Complete polychromatic CT simulation with ALL physics effects and proper
# calibration workflow. This demonstrates a realistic clinical CT pipeline.
#
# IMAGING CHAIN:
# ==============
#
#   1. POLYCHROMATIC FORWARD PROJECTION
#      - Beer-Lambert law: I = Σ wₑ × exp(-∫μₑ dl)
#      - 120 kVp spectrum (30 energy bins)
#      - Material-specific attenuation from XrayAttenuation.jl
#
#   2. PHYSICS EFFECTS (via PhysicsConfig)
#      - Scatter (Compton/Rayleigh convolution)
#      - Detector crosstalk (electronic + optical coupling)
#      - Focal spot blur (geometric penumbra)
#      - Detector lag (afterglow/ghosting)
#
#   3. INTENSITY-DOMAIN SIGNAL CHAIN
#      - Heel effect (anode self-attenuation)
#      - DAS model (gain, electronic noise, quantization)
#
#   4. CALIBRATION
#      - Air scan acquisition (same effects applied)
#      - Normalization: I_normalized = I_object / I_air
#      - Log transform: sinogram = -log(I_normalized)
#
#   5. SINOGRAM-DOMAIN CORRECTIONS
#      - Beam hardening correction (water-based polynomial)
#
#   6. RECONSTRUCTION
#      - FDK (Feldkamp-Davis-Kress) cone-beam reconstruction
#
#   7. HU CONVERSION
#      - Using effective μ_water for polychromatic spectrum
#
# =============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BasisSimulator
using Statistics

# Check for GPU
const HAS_GPU = try
    using Metal
    Metal.functional()
catch
    false
end

println("=" ^ 70)
println("Full CatSim-Style CT Imaging Chain")
println("=" ^ 70)
println("GPU available: $HAS_GPU")

# =============================================================================
# CONFIGURATION
# =============================================================================

const CONFIG = (
    # Scanner geometry
    n_angles = 360,
    n_rows = 16,
    n_cols = 256,
    fov_cm = 35.0,
    z_cm = 4.0,

    # Phantom
    n_voxels = 64,

    # Spectrum
    kvp = 120,
    n_energy_bins = 30,

    # Physics mode: Always TRUE for realistic simulation
    # Set to false only for debugging clean signal chain without noise
    enable_physics_pipeline = true,

    # Noise settings (only used when enable_physics_pipeline = true)
    electronic_noise_sigma = 0.001,  # DAS electronic noise
    noise_level = 1.0,               # Quantum noise level
)

# =============================================================================
# STEP 1: Create Scanner Geometry
# =============================================================================
println("\n" * "-"^70)
println("STEP 1: Scanner Geometry")
println("-"^70)

geom = create_aquilion_one(
    n_angles = CONFIG.n_angles,
    n_rows = CONFIG.n_rows,
    n_cols = CONFIG.n_cols,
    fov_cm = CONFIG.fov_cm,
    z_cm = CONFIG.z_cm  # Must match phantom z_cm!
)

println("Scanner: Canon Aquilion ONE style")
println("  Source-Axis Distance (SAD): $(geom.SAD) cm")
println("  Source-Detector Distance (SDD): $(geom.SDD) cm")
println("  Detector: $(CONFIG.n_cols) × $(CONFIG.n_rows) pixels")
println("  Projections: $(CONFIG.n_angles) angles")
println("  FOV: $(CONFIG.fov_cm) × $(CONFIG.fov_cm) × $(CONFIG.z_cm) cm")

# =============================================================================
# STEP 2: Create Phantom
# =============================================================================
println("\n" * "-"^70)
println("STEP 2: Phantom Creation")
println("-"^70)

phantom = create_gammex_472(
    n_voxels = CONFIG.n_voxels,
    fov_cm = CONFIG.fov_cm,
    z_cm = CONFIG.z_cm  # Must match geometry z_cm!
)

println("Phantom: Gammex 472 Multi-Energy CT")
println("  Volume: $(size(phantom.μ))")
println("  Voxel size: $(round(CONFIG.fov_cm * 10 / CONFIG.n_voxels, digits=2)) mm")
println("  Materials: $(length(unique(phantom.mask))) regions")

# =============================================================================
# STEP 3: Load Spectrum and Materials
# =============================================================================
println("\n" * "-"^70)
println("STEP 3: Spectrum and Materials")
println("-"^70)

energies_full, weights_full = load_spectrum(CONFIG.kvp)
energies, weights = downsample_spectrum(energies_full, weights_full, CONFIG.n_energy_bins)
materials = get_region_materials()

mean_energy = sum(energies .* weights) / sum(weights)
println("Spectrum: $(CONFIG.kvp) kVp")
println("  Energy bins: $(length(energies_full)) → $(CONFIG.n_energy_bins) (downsampled)")
println("  Range: $(round(minimum(energies), digits=1)) - $(round(maximum(energies), digits=1)) keV")
println("  Mean energy: $(round(mean_energy, digits=1)) keV")

# =============================================================================
# STEP 4: Configure Physics Effects
# =============================================================================
println("\n" * "-"^70)
println("STEP 4: Physics Configuration")
println("-"^70)

# --- Physics Pipeline (applied during forward projection) ---
# Includes: ALL physics effects when enabled
if CONFIG.enable_physics_pipeline
    # full_physics_config() enables ALL 10 effects by default
    physics_config = full_physics_config(
        energy_keV = Float64(mean_energy),
        noise_seed = 42,
        scatter_scale = 1.0,
        noise_level = CONFIG.noise_level
    )
    physics_info = get_physics_config_info(physics_config)
    println("Physics Pipeline (full_physics_config - ALL enabled):")
    for effect in physics_info.enabled_effects
        println("  ✓ $effect")
    end
else
    physics_config = nothing
    println("Physics Pipeline: DISABLED (deterministic mode)")
    println("  (Set enable_physics_pipeline=true for full physics)")
end

# --- Heel Effect (intensity domain) ---
heel = default_heel_effect(
    anode_angle_deg = 7.0,
    target_material = :tungsten,
    effective_thickness_mm = 0.01
)
heel_info = get_heel_effect_info(heel)
println("\nHeel Effect:")
println("  ✓ Anode angle: $(heel_info.anode_angle_deg)°")
println("  ✓ Target: $(heel_info.target_material)")
println("  ✓ $(heel_info.expected_variation)")

# --- DAS Model (intensity domain) ---
das_noise = CONFIG.enable_physics_pipeline ? CONFIG.electronic_noise_sigma : 0.0
das = default_das_model(
    gain = 1.0,
    electronic_noise_sigma = das_noise,
    lsb = 0.0,      # No quantization for demo
    min_value = 0.0,
    max_value = 1e6,
    offset = 0.0
)
das_info = get_das_info(das)
println("\nDAS Model:")
println("  ✓ Gain: $(das_info.gain)")
println("  ✓ Electronic noise σ: $(das_info.electronic_noise_sigma)")

# --- Beam Hardening Correction (sinogram domain) ---
bhc = bhc_water_default(reference_energy_keV = mean_energy)
bhc_info = get_bhc_info(bhc)
println("\nBeam Hardening Correction:")
println("  ✓ $(bhc_info.order)-order polynomial")
println("  ✓ Reference energy: $(bhc_info.reference_energy_keV) keV")

# =============================================================================
# STEP 5: Polychromatic Forward Projection
# =============================================================================
println("\n" * "-"^70)
println("STEP 5: Polychromatic Forward Projection")
println("-"^70)

# Transfer to GPU if available
if HAS_GPU
    mask_gpu = MtlArray(phantom.mask)
    println("Phantom mask transferred to GPU")
else
    mask_gpu = phantom.mask
    println("Running on CPU")
end

println("\nComputing polychromatic projection...")
println("  (Beer-Lambert: I = Σ wₑ × exp(-∫μₑ dl))")

@time sinogram_gpu = if physics_config !== nothing
    forward_project(
        mask_gpu, geom;
        energies = energies,
        weights = weights,
        materials = materials,
        physics = physics_config
    )
else
    forward_project(
        mask_gpu, geom;
        energies = energies,
        weights = weights,
        materials = materials
    )
end

sinogram_physics = HAS_GPU ? Array(sinogram_gpu) : sinogram_gpu
println("Sinogram size: $(size(sinogram_physics))")
println("Sinogram range: [$(round(minimum(sinogram_physics), digits=3)), $(round(maximum(sinogram_physics), digits=3))]")

# =============================================================================
# STEP 6: Intensity-Domain Signal Chain
# =============================================================================
println("\n" * "-"^70)
println("STEP 6: Intensity-Domain Signal Chain")
println("-"^70)

# Convert sinogram (line integrals) to intensity domain
# I = exp(-sinogram) represents transmitted X-ray intensity
println("\n[6a] Converting to intensity domain: I = exp(-sinogram)")
intensity = exp.(-sinogram_physics)
println("  Intensity range: [$(round(minimum(intensity), digits=6)), $(round(maximum(intensity), digits=4))]")

# --- Simulate Air Scan ---
# Air scan = reference measurement with no object in beam
# Must have SAME physics effects applied for proper calibration
println("\n[6b] Simulating air scan (reference measurement)")
air_scan = ones(Float32, size(intensity))

# --- Apply Heel Effect ---
println("\n[6c] Applying heel effect to object and air scans")
if HAS_GPU
    intensity_gpu = MtlArray(intensity)
    air_gpu = MtlArray(air_scan)
    apply_heel_effect!(intensity_gpu, heel, geom)
    apply_heel_effect!(air_gpu, heel, geom)
    intensity = Array(intensity_gpu)
    air_scan = Array(air_gpu)
else
    apply_heel_effect!(intensity, heel, geom)
    apply_heel_effect!(air_scan, heel, geom)
end
println("  Object intensity range: [$(round(minimum(intensity), digits=6)), $(round(maximum(intensity), digits=4))]")
println("  Air scan range: [$(round(minimum(air_scan), digits=4)), $(round(maximum(air_scan), digits=4))]")

# --- Apply DAS Model ---
println("\n[6d] Applying DAS model")
if HAS_GPU
    intensity_gpu = MtlArray(intensity)
    apply_das_model!(intensity_gpu, das; seed=42)
    intensity = Array(intensity_gpu)
else
    apply_das_model!(intensity, das; seed=42)
end
# Air scan gets gain only (no random noise for calibration reference)
air_scan .*= Float32(das.gain)
println("  Object intensity range: [$(round(minimum(intensity), digits=6)), $(round(maximum(intensity), digits=4))]")

# =============================================================================
# STEP 7: Calibration (Air Scan Normalization)
# =============================================================================
println("\n" * "-"^70)
println("STEP 7: Calibration")
println("-"^70)

# Normalize by air scan: removes heel effect and DAS gain variations
# This is how real CT scanners calibrate - divide by air reference
println("\n[7a] Normalizing: I_norm = I_object / I_air")
normalized = intensity ./ max.(air_scan, Float32(1e-10))
println("  Normalized range: [$(round(minimum(normalized), digits=6)), $(round(maximum(normalized), digits=4))]")

# Log transform back to sinogram domain
println("\n[7b] Log transform: sinogram = -log(I_norm)")
sinogram_calibrated = -log.(max.(normalized, Float32(1e-10)))
println("  Calibrated sinogram range: [$(round(minimum(sinogram_calibrated), digits=3)), $(round(maximum(sinogram_calibrated), digits=3))]")

# =============================================================================
# STEP 8: Beam Hardening Correction
# =============================================================================
println("\n" * "-"^70)
println("STEP 8: Beam Hardening Correction")
println("-"^70)

# BHC corrects for beam hardening artifacts caused by polychromatic spectrum
# Uses water-based polynomial correction
println("\nApplying $(bhc_info.order)-order polynomial BHC...")
if HAS_GPU
    sino_gpu = MtlArray(sinogram_calibrated)
    apply_bhc!(sino_gpu, bhc)
    sinogram_corrected = Array(sino_gpu)
else
    sinogram_corrected = copy(sinogram_calibrated)
    apply_bhc!(sinogram_corrected, bhc)
end
println("  BHC sinogram range: [$(round(minimum(sinogram_corrected), digits=3)), $(round(maximum(sinogram_corrected), digits=3))]")

# =============================================================================
# STEP 9: FDK Reconstruction
# =============================================================================
println("\n" * "-"^70)
println("STEP 9: FDK Reconstruction")
println("-"^70)

recon_size = size(phantom.μ)
println("\nReconstructing $(recon_size) volume...")

@time recon = fdk_reconstruct(
    HAS_GPU ? MtlArray(sinogram_corrected) : sinogram_corrected,
    geom,
    recon_size
    # Using default RampFilter for accurate HU quantification
)

recon_cpu = HAS_GPU ? Array(recon) : recon
println("Reconstruction μ range: [$(round(minimum(recon_cpu), digits=4)), $(round(maximum(recon_cpu), digits=4))] cm⁻¹")

# =============================================================================
# STEP 10: HU Conversion and Validation
# =============================================================================
println("\n" * "-"^70)
println("STEP 10: HU Conversion and Validation")
println("-"^70)

# Use effective μ_water for polychromatic spectrum
# This accounts for beam hardening in the water reference
μ_water_eff = Float32(get_effective_μ_water_kVp(CONFIG.kvp))
println("\nReference μ_water ($(CONFIG.kvp) kVp effective): $(μ_water_eff) cm⁻¹")

# Convert to HU
recon_hu = 1000.0f0 .* (recon_cpu .- μ_water_eff) ./ μ_water_eff
println("HU range: [$(round(minimum(recon_hu))), $(round(maximum(recon_hu)))]")

# --- Measure HU in center slice (avoids cone-beam edge artifacts) ---
center_z = recon_size[3] ÷ 2 + 1
center_slice = recon_hu[:, :, center_z]

println("\n--- HU Validation (Center Slice z=$center_z) ---")
println()

# Define validation regions
validation_regions = [
    (name="Solid Water", id=REGION_SOLID_WATER, expected=0),
    (name="Ca-50",       id=REGION_CA_50,       expected=50),
    (name="Ca-100",      id=REGION_CA_100,      expected=100),
    (name="Ca-200",      id=REGION_CA_200,      expected=200),
    (name="Ca-300",      id=REGION_CA_300,      expected=300),
    (name="Ca-400",      id=REGION_CA_400,      expected=400),
]

println("Region          | Measured HU    | Expected (approx)")
println("-"^55)

for region in validation_regions
    mask_2d = phantom.mask[:, :, center_z] .== UInt8(region.id)
    if sum(mask_2d) > 0
        hu_vals = center_slice[mask_2d]
        hu_mean = mean(hu_vals)
        hu_std = std(hu_vals)
        println("  $(rpad(region.name, 12)) | $(lpad(round(Int, hu_mean), 5)) ± $(lpad(round(Int, hu_std), 3)) HU | ~$(region.expected) HU")
    end
end

# =============================================================================
# SUMMARY
# =============================================================================
println("\n" * "=" ^ 70)
println("Full CatSim-Style Imaging Chain Complete")
println("=" ^ 70)

println("\nImaging Chain Summary:")
println("  1. ✓ Polychromatic forward projection ($(CONFIG.kvp) kVp, $(CONFIG.n_energy_bins) bins)")
if CONFIG.enable_physics_pipeline
    println("  2. ✓ Physics pipeline: scatter, crosstalk, focal_spot, noise, lag")
else
    println("  2. ○ Physics pipeline: disabled (deterministic mode)")
end
println("  3. ✓ Heel effect ($(heel_info.anode_angle_deg)° anode)")
println("  4. ✓ DAS model (gain=$(das_info.gain), noise=$(das_info.electronic_noise_sigma))")
println("  5. ✓ Air scan calibration")
println("  6. ✓ Beam hardening correction ($(bhc_info.order)-order polynomial)")
println("  7. ✓ FDK reconstruction")
println("  8. ✓ HU conversion (μ_water_eff = $(μ_water_eff) cm⁻¹)")

println("\nKey Points:")
println("  • Polychromatic projection models energy-dependent attenuation")
println("  • Air scan calibration normalizes intensity-domain effects")
println("  • BHC corrects beam hardening artifacts from polychromatic spectrum")
println("  • Center slice HU measurement avoids cone-beam edge artifacts")
println("  • Solid water should be ~0 HU; calcium inserts should be positive")

if !CONFIG.enable_physics_pipeline
    println("\nTo enable full physics simulation, set in CONFIG:")
    println("  • enable_physics_pipeline = true")
    println("  (Adds scatter, crosstalk, focal spot blur, quantum noise, detector lag)")
end
println()
