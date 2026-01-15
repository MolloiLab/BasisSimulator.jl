# =============================================================================
# Clinical CT Signal Chain Demo
# =============================================================================
#
# This script demonstrates a realistic clinical CT simulation workflow
# incorporating signal processing features:
#
# 1. Monochromatic forward projection
# 2. Heel effect (anode self-attenuation)
# 3. DAS model (detector signal chain, electronic noise)
# 4. Air scan calibration (normalization)
# 5. FDK reconstruction
#
# Key: Proper air scan calibration ensures HU values are correct!
#
# Note on Beam Hardening Correction (BHC):
#   BHC is NOT applied here because this demo uses monochromatic projection.
#   BHC corrects for beam hardening artifacts that occur with polychromatic
#   X-ray spectra - applying it to monochromatic data is physically incorrect.
#   For polychromatic simulations, use forward_project with energies/weights.
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

# Clinical parameters (moderate resolution for demo speed)
n_angles = 360           # Projection angles
n_rows = 16              # Multi-slice detector
n_cols = 256             # Detector columns
fov_cm = 35.0            # 35 cm FOV (body scan)
z_cm = 4.0               # Z coverage

# CRITICAL: Match geometry z FOV with phantom z FOV!
geom = create_aquilion_one(
    n_angles = n_angles,
    n_rows = n_rows,
    n_cols = n_cols,
    fov_cm = fov_cm,
    z_cm = z_cm  # Must match phantom!
)

println("Scanner: Canon Aquilion ONE style")
println("  SAD: $(geom.SAD) cm")
println("  SDD: $(geom.SDD) cm")
println("  Detector: $(n_cols) x $(n_rows)")
println("  Angles: $(n_angles)")
println("  FOV: $(fov_cm) x $(fov_cm) x $(geom.fov[3]) cm")

# =============================================================================
# Step 2: Create Phantom (Gammex 472 with inserts)
# =============================================================================
println("\n--- Step 2: Phantom Creation ---")

# Phantom with matched z FOV
phantom = create_gammex_472(
    n_voxels = 64,      # 64x64 voxels (moderate resolution)
    fov_cm = fov_cm,
    z_cm = z_cm         # Must match geometry!
)

println("Phantom: Gammex 472 Multi-Energy CT")
println("  Volume size: $(size(phantom.μ))")
println("  Regions: $(length(unique(phantom.mask))) materials")
println("  μ range: [$(round(minimum(phantom.μ), digits=4)), $(round(maximum(phantom.μ), digits=4))] cm⁻¹")

# =============================================================================
# Step 3: Configure Physics Effects
# =============================================================================
println("\n--- Step 3: Physics Configuration ---")

# Heel effect (anode self-attenuation)
heel = default_heel_effect(
    anode_angle_deg = 7.0,
    target_material = :tungsten,
    effective_thickness_mm = 0.01  # Conservative setting
)
heel_info = get_heel_effect_info(heel)
println("Heel effect: $(heel_info.anode_angle_deg)° anode, $(heel_info.effective_thickness_mm) mm")

# DAS model (detector signal chain) - NO noise for clean, deterministic demo
# Note: For realistic noise simulation, set electronic_noise_sigma > 0
# and ensure air scan is acquired with same noise characteristics (averaged)
das = default_das_model(
    gain = 1.0,
    electronic_noise_sigma = 0.0,   # No noise for clean demo
    lsb = 0.0,
    min_value = 0.0,
    max_value = 1e6,
    offset = 0.0
)
das_info = get_das_info(das)
println("DAS model: gain=$(das_info.gain), noise_σ=$(das_info.electronic_noise_sigma) (deterministic)")

# Note: BHC not used for monochromatic simulation (see header comment)
println("BHC: disabled (monochromatic simulation)")

# =============================================================================
# Step 4: Forward Projection (Monochromatic)
# =============================================================================
println("\n--- Step 4: Forward Projection ---")

# Convert phantom to GPU if available
volume = Float32.(phantom.μ)
if HAS_GPU
    volume_gpu = MtlArray(volume)
    println("Using GPU (Metal) for forward projection")
else
    volume_gpu = volume
    println("Using CPU for forward projection")
end

# Monochromatic forward projection
println("Computing forward projection...")
@time sinogram = forward_project(volume_gpu, geom)

# Convert to CPU for processing
sinogram_cpu = HAS_GPU ? Array(sinogram) : sinogram
println("Sinogram size: $(size(sinogram_cpu))")
println("Sinogram range: [$(round(minimum(sinogram_cpu), digits=3)), $(round(maximum(sinogram_cpu), digits=3))]")

# =============================================================================
# Step 5: Apply Signal Chain Effects with PROPER CALIBRATION
# =============================================================================
println("\n--- Step 5: Signal Chain Processing ---")

# Convert sinogram to intensity domain: I = exp(-sinogram)
intensity = exp.(-sinogram_cpu)
println("Intensity range: [$(round(minimum(intensity), digits=6)), $(round(maximum(intensity), digits=3))]")

# Simulate air scan (reference for calibration) - SAME effects must be applied!
println("  Simulating air scan...")
air_scan = ones(Float32, size(intensity))

# === Apply intensity-domain effects to BOTH phantom and air ===
println("  Applying heel effect...")
if HAS_GPU
    intensity_gpu = MtlArray(intensity)
    air_gpu = MtlArray(air_scan)
    apply_heel_effect!(intensity_gpu, heel, geom)
    apply_heel_effect!(air_gpu, heel, geom)  # SAME effect to air!
    intensity = Array(intensity_gpu)
    air_scan = Array(air_gpu)
else
    apply_heel_effect!(intensity, heel, geom)
    apply_heel_effect!(air_scan, heel, geom)  # SAME effect to air!
end

println("  Applying DAS model...")
if HAS_GPU
    intensity_gpu = MtlArray(intensity)
    apply_das_model!(intensity_gpu, das; seed=42)
    intensity = Array(intensity_gpu)
    # For air scan, apply only gain (no random noise for calibration)
    air_scan .*= Float32(das.gain)
else
    apply_das_model!(intensity, das; seed=42)
    air_scan .*= Float32(das.gain)
end

# === Calibration: normalized = intensity / air ===
println("  Calibrating (air scan normalization)...")
normalized = intensity ./ max.(air_scan, Float32(1e-10))
println("  Normalized range: [$(round(minimum(normalized), digits=6)), $(round(maximum(normalized), digits=3))]")

# === Log transform: sinogram = -log(normalized) ===
sinogram_final = -log.(max.(normalized, Float32(1e-10)))
println("Final sinogram range: [$(round(minimum(sinogram_final), digits=3)), $(round(maximum(sinogram_final), digits=3))]")

# Note: BHC would be applied here for polychromatic simulations:
#   apply_bhc!(sinogram_final, bhc_water_default())
# But we skip it for monochromatic data (no beam hardening to correct)

# =============================================================================
# Step 6: Reconstruction (FDK)
# =============================================================================
println("\n--- Step 6: FDK Reconstruction ---")

# Reconstruction volume size (same as phantom)
recon_size = size(phantom.μ)

println("Reconstructing $(recon_size) volume...")
@time recon = fdk_reconstruct(
    HAS_GPU ? MtlArray(sinogram_final) : sinogram_final,
    geom,
    recon_size
    # Note: Using default RampFilter for correct HU quantification
    # SheppLoganFilter etc. have normalization issues affecting HU accuracy
)

recon_cpu = HAS_GPU ? Array(recon) : recon
println("Reconstruction range: [$(round(minimum(recon_cpu), digits=4)), $(round(maximum(recon_cpu), digits=4))] cm⁻¹")

# =============================================================================
# Step 7: Convert to Hounsfield Units
# =============================================================================
println("\n--- Step 7: HU Conversion ---")

# Use library-consistent μ_water (CRITICAL for correct HU!)
μ_water = Float32(get_reference_μ_water(60.0))
println("Reference μ_water at 60 keV: $μ_water cm⁻¹")

# Convert to HU: HU = 1000 × (μ - μ_water) / μ_water
recon_hu = 1000.0f0 .* (recon_cpu .- μ_water) ./ μ_water

println("HU range: [$(round(minimum(recon_hu))), $(round(maximum(recon_hu)))]")

# Sample HU values at CENTER SLICE only (avoid cone-beam edge artifacts)
center_z = recon_size[3] ÷ 2 + 1
center_slice = recon_hu[:, :, center_z]

# Check HU in solid water region
mask_center = phantom.mask[:, :, center_z] .== UInt8(REGION_SOLID_WATER)
if sum(mask_center) > 0
    water_hu = mean(center_slice[mask_center])
    water_std = std(center_slice[mask_center])
    println("Solid water HU (center slice): $(round(water_hu, digits=1)) ± $(round(water_std, digits=1))")
end

# Check calcium inserts
mask_ca100 = phantom.mask[:, :, center_z] .== UInt8(REGION_CA_100)
mask_ca200 = phantom.mask[:, :, center_z] .== UInt8(REGION_CA_200)
if sum(mask_ca100) > 0
    ca100_hu = mean(center_slice[mask_ca100])
    println("Ca-100 HU (center slice): $(round(ca100_hu, digits=1))")
end
if sum(mask_ca200) > 0
    ca200_hu = mean(center_slice[mask_ca200])
    println("Ca-200 HU (center slice): $(round(ca200_hu, digits=1))")
end

# =============================================================================
# Summary
# =============================================================================
println("\n" * "=" ^ 70)
println("Clinical Signal Chain Demo Complete")
println("=" ^ 70)
println()
println("Signal chain applied:")
println("  1. Monochromatic forward projection")
println("  2. Heel effect ($(heel_info.anode_angle_deg)° anode)")
println("  3. DAS model (gain=$(das_info.gain), noise=$(das_info.electronic_noise_sigma))")
println("  4. Air scan calibration (CRITICAL for correct HU!)")
println("  5. FDK reconstruction (Ramp filter)")
println()
println("Key insight: Without proper air scan calibration, heel effect")
println("and DAS model would cause large HU shifts. Calibration normalizes")
println("the intensity variations by dividing by the air scan reference.")
println()
println("Note: BHC is NOT applied to monochromatic simulations.")
println("For polychromatic spectra, BHC would correct beam hardening artifacts.")
println()
println("Results:")
println("  Sinogram size: $(size(sinogram_final))")
println("  Recon size: $(size(recon_cpu))")
println("  Water HU should be close to 0 (±50 HU)")
println()
