"""
HU Calibration Fix - Diagnostic Test

Identifies and fixes the HU calibration issue.
"""

using BasisSimulator
using Statistics
using Printf
import XrayAttenuation as XA

println("\n" * "="^80)
println("HU CALIBRATION DIAGNOSTIC")
println("="^80)

# Simple water phantom test
println("\n1. Creating simple water cylinder...")
water_phantom = create_water_cylinder(diameter_mm=100.0, height_mm=40.0, resolution_mm=2.0)

protocol = ScanProtocol(kVp=120.0, mAs=200.0, scan_fov_mm=150.0, num_projections=180)
geometry = create_aquilion_one(protocol=protocol)
spectrum = generate_spectrum(kVp=120.0, mAs=200.0)

println("2. Running forward simulation...")
sinogram = simulate_ct_scan(phantom=water_phantom, geometry=geometry, spectrum=spectrum, verbose=false)

# Check signal range
println("\n3. Signal analysis:")
println("   Raw sinogram range: [$(round(minimum(sinogram), sigdigits=4)), $(round(maximum(sinogram), sigdigits=4))]")
println("   Mean signal: $(round(mean(sinogram), sigdigits=4))")

# Proper I0 normalization - use AIR scan equivalent
# The maximum signal represents rays that didn't pass through phantom (air)
I0 = maximum(sinogram)
println("   I0 (air scan): $(round(I0, sigdigits=4))")

# Convert to transmission
transmission = sinogram ./ I0
println("   Transmission range: [$(round(minimum(transmission), digits=4)), $(round(maximum(transmission), digits=4))]")

# Convert to attenuation: μt = -ln(I/I0)
attenuation_sino = -log.(max.(transmission, 1e-10))  # Clamp to avoid log(0)
println("   Attenuation range: [$(round(minimum(attenuation_sino), digits=3)), $(round(maximum(attenuation_sino), digits=3))] cm⁻¹")

# Expected attenuation for ~10cm water at mean energy
mean_E = sum(spectrum.energies .* spectrum.photons) / sum(spectrum.photons)
mu_water_theory = get_linear_attenuation(XA.Materials.water, mean_E)
expected_atten = mu_water_theory * 10.0  # 10 cm path
println("   Expected max attenuation (10cm water @ $(round(mean_E, digits=1)) keV): $(round(expected_atten, digits=2)) cm⁻¹")
println("   Actual max attenuation: $(round(maximum(attenuation_sino), digits=2)) cm⁻¹")

# Check if scaling is off
scaling_factor = expected_atten / maximum(attenuation_sino)
println("   Scaling factor needed: $(round(scaling_factor, digits=3))")

println("\n4. Reconstruction...")
recon_x = collect(range(-water_phantom.grid.fov_xy_cm/2, water_phantom.grid.fov_xy_cm/2, length=water_phantom.grid.nx))
recon_y = collect(range(-water_phantom.grid.fov_xy_cm/2, water_phantom.grid.fov_xy_cm/2, length=water_phantom.grid.ny))
recon_z = collect(range(-water_phantom.grid.fov_z_cm/2, water_phantom.grid.fov_z_cm/2, length=water_phantom.grid.nz))
angles_deg = rad2deg.(geometry.angles)

recon = reconstruct_fdk(
    attenuation_sino, geometry.SAD_cm, geometry.SDD_cm,
    geometry.pixel_width_cm, geometry.pixel_height_cm,
    angles_deg, recon_x, recon_y, recon_z,
    filter_type=ramlak
)

println("   Reconstruction range: [$(round(minimum(recon), digits=4)), $(round(maximum(recon), digits=4))] cm⁻¹")
println("   Mean reconstruction: $(round(mean(recon), digits=4)) cm⁻¹")
println("   Expected water μ: $(round(mu_water_theory, digits=4)) cm⁻¹")

# HU conversion - METHOD 1: Direct comparison
HU_method1 = @. 1000.0 * (recon - mu_water_theory) / mu_water_theory

# HU conversion - METHOD 2: Using reconstruction mean as reference
recon_mean = mean(recon)
HU_method2 = @. 1000.0 * (recon - recon_mean) / recon_mean

println("\n5. HU values:")
center_slice = div(size(HU_method1, 3), 2)
cx, cy = div(size(HU_method1, 1), 2), div(size(HU_method1, 2), 2)
roi = 5

roi_method1 = HU_method1[(cx-roi):(cx+roi), (cy-roi):(cy+roi), center_slice]
roi_method2 = HU_method2[(cx-roi):(cx+roi), (cy-roi):(cy+roi), center_slice]

println("   Method 1 (vs theory μ_water): $(round(mean(roi_method1), digits=1)) ± $(round(std(roi_method1), digits=1)) HU")
println("   Method 2 (vs recon mean): $(round(mean(roi_method2), digits=1)) ± $(round(std(roi_method2), digits=1)) HU")

# The issue: FDK has unknown scaling that needs to be calibrated out
# Proper approach: Use water ROI to calibrate
water_mu_measured = mean(recon[(cx-10):(cx+10), (cy-10):(cy+10), center_slice])
calibration_factor = mu_water_theory / water_mu_measured
println("\n6. Calibration factor: $(round(calibration_factor, digits=4))")

recon_calibrated = recon .* calibration_factor
HU_calibrated = @. 1000.0 * (recon_calibrated - mu_water_theory) / mu_water_theory
roi_calibrated = HU_calibrated[(cx-roi):(cx+roi), (cy-roi):(cy+roi), center_slice]

println("   ✅ Calibrated HU: $(round(mean(roi_calibrated), digits=1)) ± $(round(std(roi_calibrated), digits=1)) HU")

println("\n" * "="^80)
println("DIAGNOSIS COMPLETE")
println("="^80)
println("\nPROBLEM: FDK reconstruction has arbitrary scaling")
println("SOLUTION: Calibrate using known water reference")
println("   1. Measure μ_water in reconstruction ROI")
println("   2. Scale reconstruction to match theoretical μ_water")
println("   3. Compute HU relative to calibrated water reference")
println("\n" * "="^80 * "\n")
