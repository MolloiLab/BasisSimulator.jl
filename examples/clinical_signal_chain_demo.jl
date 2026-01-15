# =============================================================================
# Clinical CT Signal Chain Demo
# =============================================================================
#
# This script demonstrates a realistic clinical CT simulation workflow
# incorporating all signal processing features:
#
# 1. Polychromatic X-ray spectrum (120 kVp)
# 2. Heel effect (anode self-attenuation)
# 3. DAS model (detector signal chain, electronic noise)
# 4. Calibration pipeline (air scan, offset correction, log transform)
# 5. Beam hardening correction
# 6. Helical scanning mode
# 7. FDK reconstruction
#
# =============================================================================

using BasisSimulator
using Statistics

# Check for GPU
HAS_GPU = try
    using Metal
    Metal.functional()
catch
    false
end

println("=" ^ 70)
println("Clinical CT Signal Chain Demo")
println("=" ^ 70)
println("GPU available: $HAS_GPU")

# =============================================================================
# Step 1: Create Scanner Geometry (Canon Aquilion ONE style)
# =============================================================================
println("\n--- Step 1: Scanner Geometry ---")

# Clinical parameters
n_angles = 1000          # ~1000 views per rotation (typical clinical)
n_rows = 32              # Multi-slice detector
n_cols = 512             # Detector columns
fov_cm = 35.0            # 35 cm FOV (body scan)

geom = create_aquilion_one(
    n_angles = n_angles,
    n_rows = n_rows,
    n_cols = n_cols,
    fov_cm = fov_cm
)

println("Scanner: Canon Aquilion ONE style")
println("  SAD: $(geom.SAD) mm")
println("  SDD: $(geom.SDD) mm")
println("  Detector: $(n_cols) x $(n_rows)")
println("  Angles: $(n_angles)")
println("  FOV: $(fov_cm) cm")

# =============================================================================
# Step 2: Create Phantom (Gammex 472 with inserts)
# =============================================================================
println("\n--- Step 2: Phantom Creation ---")

# High-resolution phantom
phantom = create_gammex_472(
    n_voxels = 256,      # 256x256x64 volume
    fov_cm = fov_cm,
    z_cm = 8.0
)

println("Phantom: Gammex 472 Multi-Energy CT")
println("  Volume size: $(size(phantom.μ))")
println("  Regions: $(length(unique(phantom.mask))) materials")
println("  μ range: [$(round(minimum(phantom.μ), digits=4)), $(round(maximum(phantom.μ), digits=4))] cm⁻¹")

# Get materials for polychromatic simulation
materials = get_region_materials()

# =============================================================================
# Step 3: Load X-ray Spectrum (120 kVp)
# =============================================================================
println("\n--- Step 3: X-ray Spectrum ---")

energies, weights = load_spectrum(120)  # 120 kVp tungsten spectrum
energies, weights = downsample_spectrum(energies, weights, 30)  # 30 energy bins

println("Spectrum: 120 kVp tungsten")
println("  Energy range: $(minimum(energies)) - $(maximum(energies)) keV")
println("  Energy bins: $(length(energies))")
println("  Mean energy: $(round(sum(energies .* weights) / sum(weights), digits=1)) keV")

# =============================================================================
# Step 4: Configure Physics Effects
# =============================================================================
println("\n--- Step 4: Physics Configuration ---")

# Heel effect (anode self-attenuation)
heel = default_heel_effect(
    anode_angle_deg = 7.0,
    target_material = :tungsten
)
heel_info = get_heel_effect_info(heel)
println("Heel effect: $(heel_info.anode_angle_deg)° anode")

# DAS model (detector signal chain)
das = das_clinical(noise_level = 1.0)
das_info = get_das_info(das)
println("DAS model: gain=$(das_info.gain), noise_σ=$(das_info.electronic_noise_sigma)")

# Tube current modulation (angular mA modulation for dose reduction)
# Lower dose in lateral views (AP/PA views get more mA)
mA_base = 200.0  # Base tube current
mA_modulation = [mA_base * (1.0 + 0.3 * cos(2 * geom.angles[i])) for i in 1:n_angles]
println("Tube current: $(round(minimum(mA_modulation)))-$(round(maximum(mA_modulation))) mA (angular modulation)")

# Beam hardening correction
bhc = bhc_water_default(reference_energy_keV = 70.0)
bhc_info = get_bhc_info(bhc)
println("BHC: $(bhc_info.order)-order polynomial, ref=$(bhc_info.reference_energy_keV) keV")

# =============================================================================
# Step 5: Forward Projection (Polychromatic)
# =============================================================================
println("\n--- Step 5: Forward Projection ---")

# Convert phantom to appropriate array type
if HAS_GPU
    mask_gpu = MtlArray(phantom.mask)
    println("Using GPU (Metal) for forward projection")
else
    mask_gpu = phantom.mask
    println("Using CPU for forward projection")
end

# Polychromatic forward projection
# This computes: I = Σ wₑ × exp(-∫μₑ dl) for each energy
println("Computing polychromatic projection...")
@time sinogram_raw = forward_project(
    mask_gpu, geom;
    energies = energies,
    weights = weights,
    materials = materials
)

# Convert to CPU for processing
sinogram_cpu = HAS_GPU ? Array(sinogram_raw) : sinogram_raw
println("Raw sinogram range: [$(round(minimum(sinogram_cpu), digits=3)), $(round(maximum(sinogram_cpu), digits=3))]")

# =============================================================================
# Step 6: Apply Signal Chain Effects
# =============================================================================
println("\n--- Step 6: Signal Chain Processing ---")

# Convert sinogram to intensity domain for signal chain
# sinogram = -log(I/I₀), so I = exp(-sinogram)
intensity = exp.(-sinogram_cpu)
println("Intensity range: [$(round(minimum(intensity), digits=6)), $(round(maximum(intensity), digits=3))]")

# 6a. Apply heel effect (in intensity domain)
println("  Applying heel effect...")
if HAS_GPU
    intensity_gpu = MtlArray(intensity)
    apply_heel_effect!(intensity_gpu, heel, geom)
    intensity = Array(intensity_gpu)
else
    apply_heel_effect!(intensity, heel, geom)
end

# 6b. Apply tube current modulation
println("  Applying mA modulation...")
if HAS_GPU
    intensity_gpu = MtlArray(intensity)
    apply_tube_current!(intensity_gpu, mA_modulation)
    intensity = Array(intensity_gpu)
else
    apply_tube_current!(intensity, mA_modulation)
end

# 6c. Simulate air scan (reference for calibration)
println("  Simulating air scan...")
air_scan = simulate_air_scan(geom)
# Apply same heel effect to air scan
if HAS_GPU
    air_gpu = MtlArray(air_scan)
    apply_heel_effect!(air_gpu, heel, geom)
    apply_tube_current!(air_gpu, mA_modulation)
    air_scan = Array(air_gpu)
else
    apply_heel_effect!(air_scan, heel, geom)
    apply_tube_current!(air_scan, mA_modulation)
end

# 6d. Apply DAS model (electronic noise, gain)
println("  Applying DAS model (noise, gain)...")
if HAS_GPU
    intensity_gpu = MtlArray(intensity)
    apply_das_model!(intensity_gpu, das; seed=42)
    intensity = Array(intensity_gpu)

    air_gpu = MtlArray(air_scan)
    apply_das_model!(air_gpu, das; seed=43)  # Different seed for air
    air_scan = Array(air_gpu)
else
    apply_das_model!(intensity, das; seed=42)
    apply_das_model!(air_scan, das; seed=43)
end

# 6e. Calibration: offset correction + normalization + log transform
println("  Applying calibration pipeline...")
offset = 0.0f0  # Simplified: zero offset
if HAS_GPU
    intensity_gpu = MtlArray(intensity)
    air_gpu = MtlArray(air_scan)
    calibrate_sinogram!(intensity_gpu, air_gpu, offset)
    sinogram_calibrated = Array(intensity_gpu)
else
    sinogram_calibrated = calibrate_sinogram!(intensity, air_scan, offset)
end

println("Calibrated sinogram range: [$(round(minimum(sinogram_calibrated), digits=3)), $(round(maximum(sinogram_calibrated), digits=3))]")

# 6f. Apply beam hardening correction
println("  Applying beam hardening correction...")
if HAS_GPU
    sino_gpu = MtlArray(sinogram_calibrated)
    apply_bhc!(sino_gpu, bhc)
    sinogram_corrected = Array(sino_gpu)
else
    sinogram_corrected = apply_bhc!(copy(sinogram_calibrated), bhc)
end

println("BHC-corrected sinogram range: [$(round(minimum(sinogram_corrected), digits=3)), $(round(maximum(sinogram_corrected), digits=3))]")

# =============================================================================
# Step 7: Reconstruction (FDK)
# =============================================================================
println("\n--- Step 7: FDK Reconstruction ---")

# Reconstruction volume size (can be different from phantom)
recon_size = (256, 256, 32)  # Lower z-resolution for speed

println("Reconstructing $(recon_size) volume...")
@time recon = fdk_reconstruct(
    HAS_GPU ? MtlArray(sinogram_corrected) : sinogram_corrected,
    geom,
    recon_size;
    filter = SheppLoganFilter(),
    cutoff = 0.8
)

recon_cpu = HAS_GPU ? Array(recon) : recon
println("Reconstruction range: [$(round(minimum(recon_cpu), digits=4)), $(round(maximum(recon_cpu), digits=4))] cm⁻¹")

# =============================================================================
# Step 8: Convert to Hounsfield Units
# =============================================================================
println("\n--- Step 8: HU Conversion ---")

# Water attenuation at ~70 keV
μ_water = 0.019  # cm⁻¹

# Convert to HU: HU = 1000 × (μ - μ_water) / μ_water
recon_hu = 1000.0f0 .* (recon_cpu .- μ_water) ./ μ_water

println("HU range: [$(round(minimum(recon_hu))), $(round(maximum(recon_hu)))]")

# Sample HU values at center slice
center_z = recon_size[3] ÷ 2
center_slice = recon_hu[:, :, center_z]

# Find approximate center of phantom
cx, cy = recon_size[1] ÷ 2, recon_size[2] ÷ 2
center_region = center_slice[cx-5:cx+5, cy-5:cy+5]
println("Center region HU: $(round(mean(center_region))) ± $(round(std(center_region)))")

# =============================================================================
# Step 9: Helical Scanning Demo (Optional)
# =============================================================================
println("\n--- Step 9: Helical Scanning Demo ---")

# Create helical geometry
helical_geom = create_helical_geometry(
    geom;
    pitch = 1.0,          # Pitch factor
    rotation_time = 0.5,  # 0.5 second rotation
    z_start = 0.0
)

helical_info = get_helical_info(helical_geom)
println("Helical geometry:")
println("  Pitch: $(helical_info.pitch)")
println("  Table speed: $(round(helical_info.table_speed, digits=1)) mm/s")
println("  Z coverage: $(round(helical_info.z_coverage, digits=1)) mm")

# Helical reconstruction (using same sinogram for demo)
println("Helical FDK reconstruction...")
@time recon_helical = helical_fdk_reconstruct(
    HAS_GPU ? MtlArray(sinogram_corrected) : sinogram_corrected,
    helical_geom,
    recon_size;
    filter = SheppLoganFilter(),
    cutoff = 0.8
)

recon_helical_cpu = HAS_GPU ? Array(recon_helical) : recon_helical
recon_helical_hu = 1000.0f0 .* (recon_helical_cpu .- μ_water) ./ μ_water
println("Helical HU range: [$(round(minimum(recon_helical_hu))), $(round(maximum(recon_helical_hu)))]")

# =============================================================================
# Summary
# =============================================================================
println("\n" * "=" ^ 70)
println("Clinical Signal Chain Demo Complete")
println("=" ^ 70)
println()
println("Signal chain applied:")
println("  1. Polychromatic forward projection ($(length(energies)) energy bins)")
println("  2. Heel effect ($(heel_info.anode_angle_deg)° anode)")
println("  3. Angular mA modulation ($(round(minimum(mA_modulation)))-$(round(maximum(mA_modulation))) mA)")
println("  4. DAS model (gain=$(das_info.gain), noise=$(das_info.electronic_noise_sigma))")
println("  5. Calibration (air normalization, log transform)")
println("  6. Beam hardening correction ($(bhc_info.order)-order polynomial)")
println("  7. FDK reconstruction (Shepp-Logan filter)")
println("  8. Helical reconstruction demo")
println()
println("Results:")
println("  Sinogram size: $(size(sinogram_corrected))")
println("  Recon size: $(size(recon_cpu))")
println("  HU range: [$(round(minimum(recon_hu))), $(round(maximum(recon_hu)))]")
