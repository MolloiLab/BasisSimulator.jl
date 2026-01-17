# =============================================================================
# Helical (Spiral) CT Simulation Example
# =============================================================================
#
# Demonstrates helical CT scanning and reconstruction with GE Revolution Apex
# scanner specifications. Compares axial vs helical acquisition modes.
#
# HELICAL CT KEY CONCEPTS:
# ========================
#
# Pitch Factor:
#   pitch = table_advance_per_rotation / beam_width
#   - pitch < 1.0: Overlapping coverage (oversampling, higher dose, better quality)
#   - pitch = 1.0: Adjacent coverage (standard)
#   - pitch > 1.0: Gap between rotations (faster scan, lower dose, more artifacts)
#
# Z-Interpolation Methods:
#   - 180LI: Uses conjugate rays (180 degrees apart), thinner slice profile
#   - 360LI: Uses full rotations, thicker slice but fewer artifacts at high pitch
#
# GE Revolution Apex Pitch Values (FDA K213715):
#   - 0.5:    Low pitch, best image quality
#   - 0.531:  Cardiac imaging
#   - 0.969:  Standard body
#   - 0.992:  Standard chest/abdomen
#   - 1.375:  Fast scanning
#   - 1.531:  Maximum speed (HyperDrive mode)
#
# =============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BasisSimulator
using Statistics
using CairoMakie

# =============================================================================
# Create Phantom
# =============================================================================

println("Creating Gammex 472 phantom...")
phantom = create_gammex_472(n_voxels=64, n_slices=32, fov_cm=35.0, z_cm=16.0)
volume = Float32.(phantom.μ)

println("  Phantom size: ", size(volume))
println("  Phantom μ range: ", extrema(volume))

# =============================================================================
# Scanner Specification
# =============================================================================

spec = GERevolutionApex()
println("\nScanner: GE Revolution Apex")
println("  SID: $(spec.geometry.SID) mm")
println("  SDD: $(spec.geometry.SDD) mm")
println("  Z-coverage: $(spec.detector.z_coverage_mm) mm")

# =============================================================================
# Axial (Step-and-Shoot) Acquisition
# =============================================================================

println("\n" * "="^60)
println("AXIAL (STEP-AND-SHOOT) ACQUISITION")
println("="^60)

# Create axial geometry
axial_geom = create_geometry(spec; n_angles=360, n_rows=32, n_cols=128, fov_cm=35.0)

# Forward projection (axial)
println("Running axial forward projection...")
sino_axial = siddon_forward_project(volume, axial_geom)
println("  Sinogram size: ", size(sino_axial))

# FDK reconstruction (axial)
println("Running axial FDK reconstruction...")
recon_axial = fdk_reconstruct(sino_axial, axial_geom, (64, 64, 32))
println("  Reconstruction size: ", size(recon_axial))

# Calibrate HU
water_mask = phantom.mask .== UInt8(REGION_SOLID_WATER)
μ_water_axial = mean(recon_axial[water_mask])
hu_axial = 1000f0 .* (recon_axial .- μ_water_axial) ./ μ_water_axial

println("  Water HU: ", round(mean(hu_axial[water_mask]), digits=2))

# =============================================================================
# Helical (Spiral) Acquisition
# =============================================================================

println("\n" * "="^60)
println("HELICAL (SPIRAL) ACQUISITION")
println("="^60)

# Create helical geometry with pitch = 1.0
protocol = HelicalProtocol(
    120,    # kVp
    400,    # mA
    0.5,    # rotation time (s)
    1.0,    # pitch
    4.0,    # number of rotations
    180,    # angles per rotation
    0.625   # slice thickness (mm)
)

helical_geom = create_helical_geometry_from_spec(spec, protocol;
    n_rows=32, n_cols=128, fov_cm=35.0)

# Print helical geometry info
info = get_helical_info(helical_geom)
println("Helical geometry:")
println("  Pitch: ", info.pitch)
println("  Table speed: ", round(helical_geom.table_speed, digits=2), " cm/s")
println("  Z-coverage: ", round(info.z_coverage, digits=2), " cm")
println("  Total angles: ", info.n_angles)

# Forward projection (helical)
println("\nRunning helical forward projection...")
sino_helical = helical_forward_project(volume, helical_geom)
println("  Sinogram size: ", size(sino_helical))

# Helical FDK reconstruction with 180LI interpolation
println("Running helical FDK reconstruction (180LI)...")
recon_helical_180li = helical_fdk_reconstruct_volume(sino_helical, helical_geom, (64, 64, 16);
    interpolation=:li180)
println("  Reconstruction size: ", size(recon_helical_180li))

# Helical FDK reconstruction with 360LI interpolation
println("Running helical FDK reconstruction (360LI)...")
recon_helical_360li = helical_fdk_reconstruct_volume(sino_helical, helical_geom, (64, 64, 16);
    interpolation=:li360)
println("  Reconstruction size: ", size(recon_helical_360li))

# Calibrate HU for helical (use central water region)
center = (32, 32)
radius = 10
water_vals_180li = Float32[]
water_vals_360li = Float32[]
for iz in 1:size(recon_helical_180li, 3)
    for iy in (center[2]-radius):(center[2]+radius)
        for ix in (center[1]-radius):(center[1]+radius)
            if (ix-center[1])^2 + (iy-center[2])^2 < radius^2
                push!(water_vals_180li, recon_helical_180li[ix, iy, iz])
                push!(water_vals_360li, recon_helical_360li[ix, iy, iz])
            end
        end
    end
end

μ_water_180li = mean(water_vals_180li)
μ_water_360li = mean(water_vals_360li)

hu_180li = 1000f0 .* (recon_helical_180li .- μ_water_180li) ./ μ_water_180li
hu_360li = 1000f0 .* (recon_helical_360li .- μ_water_360li) ./ μ_water_360li

println("\nHU Calibration:")
println("  180LI Water HU: ", round(mean(1000f0 .* (water_vals_180li .- μ_water_180li) ./ μ_water_180li), digits=2))
println("  360LI Water HU: ", round(mean(1000f0 .* (water_vals_360li .- μ_water_360li) ./ μ_water_360li), digits=2))

# =============================================================================
# Helical SIRT Reconstruction
# =============================================================================

println("\n" * "="^60)
println("HELICAL SIRT RECONSTRUCTION")
println("="^60)

# Helical SIRT reconstruction (iterative method)
println("Running helical SIRT reconstruction (10 iterations)...")
recon_helical_sirt = helical_sirt_reconstruct(sino_helical, helical_geom, (64, 64, 16);
    niter=10, init=:helical_fdk)
println("  Reconstruction size: ", size(recon_helical_sirt))

# Calibrate HU for SIRT
sirt_water_vals = Float32[]
for iz in 1:size(recon_helical_sirt, 3)
    for iy in (center[2]-radius):(center[2]+radius)
        for ix in (center[1]-radius):(center[1]+radius)
            if (ix-center[1])^2 + (iy-center[2])^2 < radius^2
                push!(sirt_water_vals, recon_helical_sirt[ix, iy, iz])
            end
        end
    end
end
μ_water_sirt = mean(sirt_water_vals)
hu_sirt = 1000f0 .* (recon_helical_sirt .- μ_water_sirt) ./ μ_water_sirt

println("  SIRT Water HU: ", round(mean(1000f0 .* (sirt_water_vals .- μ_water_sirt) ./ μ_water_sirt), digits=2))

# =============================================================================
# Compare Pitch Values
# =============================================================================

println("\n" * "="^60)
println("PITCH COMPARISON")
println("="^60)

pitch_values = [0.5, 1.0, 1.5]

println("\nPitch | Z-Travel (cm) | Total Coverage (cm)")
println("------|---------------|--------------------")

for pitch in pitch_values
    protocol_p = HelicalProtocol(120, 400, 0.5, pitch, 4.0, 180, 0.625)
    helical_p = create_helical_geometry_from_spec(spec, protocol_p;
        n_rows=32, n_cols=128, fov_cm=35.0)

    z_travel = helical_p.z_positions[end] - helical_p.z_positions[1]
    total_coverage = z_travel + helical_p.beam_width

    println("$(rpad(pitch, 5)) | $(rpad(round(z_travel, digits=2), 13)) | $(round(total_coverage, digits=2))")
end

# =============================================================================
# Visualization
# =============================================================================

println("\n" * "="^60)
println("GENERATING VISUALIZATION")
println("="^60)

# Create figure
fig = Figure(size=(1600, 900))

# Row 1: Reconstruction comparison
ax1 = Axis(fig[1, 1], title="Axial (Step-and-Shoot)", aspect=DataAspect())
heatmap!(ax1, hu_axial[:, :, 16]', colormap=:grays, colorrange=(-200, 400))

ax2 = Axis(fig[1, 2], title="Helical FDK (180LI)", aspect=DataAspect())
heatmap!(ax2, hu_180li[:, :, 8]', colormap=:grays, colorrange=(-200, 400))

ax3 = Axis(fig[1, 3], title="Helical FDK (360LI)", aspect=DataAspect())
heatmap!(ax3, hu_360li[:, :, 8]', colormap=:grays, colorrange=(-200, 400))

ax4 = Axis(fig[1, 4], title="Helical SIRT", aspect=DataAspect())
heatmap!(ax4, hu_sirt[:, :, 8]', colormap=:grays, colorrange=(-200, 400))

# Row 2: Sinograms and comparison
ax5 = Axis(fig[2, 1], title="Axial Sinogram (slice 16)", xlabel="Angle", ylabel="Column")
heatmap!(ax5, sino_axial[:, 16, :]', colormap=:viridis)

ax6 = Axis(fig[2, 2], title="Helical Sinogram (slice 16)", xlabel="Angle", ylabel="Column")
heatmap!(ax6, sino_helical[:, 16, :]', colormap=:viridis)

# Z-profile comparison
ax7 = Axis(fig[2, 3:4], title="Z-Profile Comparison", xlabel="Z slice", ylabel="HU")
z_profile_axial = [mean(hu_axial[28:36, 28:36, z]) for z in 1:32]
z_profile_180li = [mean(hu_180li[28:36, 28:36, z]) for z in 1:16]
z_profile_sirt = [mean(hu_sirt[28:36, 28:36, z]) for z in 1:16]

lines!(ax7, 1:32, z_profile_axial, label="Axial", color=:blue)
lines!(ax7, range(1, 32, length=16), z_profile_180li, label="Helical FDK (180LI)", color=:red)
lines!(ax7, range(1, 32, length=16), z_profile_sirt, label="Helical SIRT", color=:green)
axislegend(ax7, position=:rt)

# Add colorbar
Colorbar(fig[1, 5], limits=(-200, 400), colormap=:grays, label="HU")

# Save figure
output_path = joinpath(@__DIR__, "helical_ct_comparison.png")
save(output_path, fig)
println("Saved visualization to: ", output_path)

# =============================================================================
# Summary
# =============================================================================

println("\n" * "="^60)
println("SUMMARY")
println("="^60)

println("""
Helical CT Reconstruction Complete!

Key Results:
- Axial and helical reconstructions produce comparable image quality
- Water HU calibrates to 0 for all modes (axial, helical FDK, helical SIRT)
- 180LI interpolation: Thinner slice profile, may show more artifacts at high pitch
- 360LI interpolation: Thicker slice profile, fewer artifacts at high pitch
- SIRT reconstruction: Iterative method provides improved noise characteristics

Reconstruction Methods Demonstrated:
- Axial FDK (standard step-and-shoot)
- Helical FDK with 180° linear interpolation (180LI)
- Helical FDK with 360° linear interpolation (360LI)
- Helical SIRT (iterative reconstruction)

For clinical use:
- Use pitch < 1.0 for high-quality imaging (cardiac, pediatric)
- Use pitch ~ 1.0 for standard body imaging
- Use pitch > 1.0 for fast scanning (trauma, large coverage)

GE Revolution Apex pitch values (FDA K213715, AJR 2018):
- 0.5, 0.531, 0.969, 0.992, 1.375, 1.531
""")
