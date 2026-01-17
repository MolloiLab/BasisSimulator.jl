# =============================================================================
# Comprehensive Reconstruction Comparison Example
# =============================================================================
#
# This example demonstrates all major reconstruction pathways:
#
# 1. SINGLE-ENERGY POLYCHROMATIC (120 kVp)
#    - FDK reconstruction
#    - SIRT reconstruction (3 iterations)
#
# 2. DUAL-ENERGY (80/140 kVp GSI)
#    - FDK reconstruction of low energy (80 kVp)
#    - FDK reconstruction of high energy (140 kVp)
#
# 3. VIRTUAL MONOENERGETIC IMAGING (VMI)
#    - Material decomposition (water/iodine basis)
#    - VMI at 70 keV (balanced, ~120 kVp equivalent)
#
# Output: 6-panel figure comparing all reconstructions
#
# =============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BasisSimulator
using Statistics
using CairoMakie
import XrayAttenuation as XA

# =============================================================================
# GPU Setup
# =============================================================================

using Metal
if !Metal.functional()
    error("This example requires a functional Metal GPU!")
end

println("=" ^ 70)
println("COMPREHENSIVE RECONSTRUCTION COMPARISON")
println("=" ^ 70)
println("GPU: $(Metal.current_device())")
println()

# =============================================================================
# Configuration
# =============================================================================

# Use integration scale for reasonable runtime
CONFIG = (
    phantom_n = 128,
    phantom_slices = 16,
    n_views = 180,
    n_rows = 16,
    n_cols = 256,
    recon_n = 128,
    n_energy_bins = 30,
    sirt_iters = 3,
    fov_cm = 35.0,
    z_cm = 4.0,
)

println("Configuration:")
println("  Phantom: $(CONFIG.phantom_n)³ × $(CONFIG.phantom_slices) slices")
println("  Views: $(CONFIG.n_views)")
println("  Detector: $(CONFIG.n_cols) × $(CONFIG.n_rows)")
println("  Energy bins: $(CONFIG.n_energy_bins)")
println("  SIRT iterations: $(CONFIG.sirt_iters)")
println()

# =============================================================================
# Setup: Phantom and Scanner
# =============================================================================

println("-" ^ 70)
println("SETUP: Phantom and Scanner")
println("-" ^ 70)

# Create Gammex 472 phantom
phantom = create_gammex_472(
    n_voxels = CONFIG.phantom_n,
    n_slices = CONFIG.phantom_slices,
    fov_cm = CONFIG.fov_cm,
    z_cm = CONFIG.z_cm
)
println("Phantom: Gammex 472 ($(size(phantom.mask)))")

# Create scanner geometry using GE Revolution Apex
scanner = GERevolutionApex()
geom = create_geometry(scanner;
    n_angles = CONFIG.n_views,
    n_rows = CONFIG.n_rows,
    n_cols = CONFIG.n_cols,
    fov_cm = CONFIG.fov_cm
)
println("Scanner: GE Revolution Apex")
println("  SID: $(geometry(scanner).sid_mm[]) mm")
println("  SDD: $(geometry(scanner).sdd_mm[]) mm")

# Get materials
materials = get_region_materials()
println()

# Move phantom to GPU
mask_gpu = MtlArray(phantom.mask);
println("Phantom mask on GPU: $(typeof(mask_gpu))")
println()

# Reconstruction size
recon_size = (CONFIG.recon_n, CONFIG.recon_n, CONFIG.phantom_slices)

# Helper function for mask downsampling (needed for HU calibration)
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
# PART 1: Single-Energy Polychromatic (120 kVp)
# =============================================================================

println("=" ^ 70)
println("PART 1: Single-Energy Polychromatic (120 kVp)")
println("=" ^ 70)

# Load spectrum
energies_120, weights_120 = load_spectrum(120)
energies_120, weights_120 = downsample_spectrum(energies_120, weights_120, CONFIG.n_energy_bins)
mean_energy_120 = sum(energies_120 .* weights_120) / sum(weights_120)
println("Spectrum: 120 kVp → $(CONFIG.n_energy_bins) bins, mean $(round(mean_energy_120, digits=1)) keV")

# Configure physics (full polychromatic pipeline)
physics_120 = realistic_physics_config(
    energy_keV = Float64(mean_energy_120),
    noise_seed = 42,
    noise_level = 1.0,  # Standard dose
    scatter_scale = 0.3,
    scatter_correction_scale = 0.3,
    enable_scatter_correction = true
)

# Forward projection
println("\nForward projection (polychromatic, all physics)...")
println("  GPU: Metal ✓")
@time sino_120_gpu = forward_project(
    mask_gpu, geom;
    energies = energies_120,
    weights = weights_120,
    materials = materials,
    physics = physics_120
);
println("  Sinogram: $(size(sino_120_gpu))")

# FDK Reconstruction
println("\n[FDK] Reconstructing...")
println("  GPU: Metal ✓")
@time recon_120_fdk_gpu = fdk_reconstruct(sino_120_gpu, geom, recon_size);
recon_120_fdk = Array(recon_120_fdk_gpu);

# SIRT Reconstruction
println("\n[SIRT] Reconstructing ($(CONFIG.sirt_iters) iterations)...")
println("  GPU: Metal ✓")
@time recon_120_sirt_gpu = sirt_reconstruct(sino_120_gpu, geom, recon_size; niter=CONFIG.sirt_iters);
recon_120_sirt = Array(recon_120_sirt_gpu);

# Convert to HU
mask_recon = downsample_mask(phantom.mask, recon_size);
center_z = CONFIG.phantom_slices ÷ 2 + 1
water_mask = mask_recon[:, :, center_z] .== UInt8(REGION_SOLID_WATER);
μ_water_120 = mean(recon_120_fdk[:, :, center_z][water_mask])

recon_120_fdk_hu = 1000f0 .* (recon_120_fdk .- μ_water_120) ./ μ_water_120
recon_120_sirt_hu = 1000f0 .* (recon_120_sirt .- μ_water_120) ./ μ_water_120

println("\nSingle-energy results (center slice):")
println("  FDK:  water=$(round(mean(recon_120_fdk_hu[:,:,center_z][water_mask]), digits=1)) HU")
println("  SIRT: water=$(round(mean(recon_120_sirt_hu[:,:,center_z][water_mask]), digits=1)) HU")

# =============================================================================
# PART 2: Dual-Energy (80/140 kVp GSI)
# =============================================================================

println()
println("=" ^ 70)
println("PART 2: Dual-Energy (80/140 kVp GSI)")
println("=" ^ 70)

# Create GSI protocol
protocol = default_gsi_protocol(
    low_mA = 400.0,
    high_mA = 400.0
)
println("Protocol: $(protocol.low_kvp)/$(protocol.high_kvp) kVp")
println("Current: $(protocol.low_mA)/$(protocol.high_mA) mA")

# Dual-energy forward projection
println("\nDual-energy forward projection...")
println("  GPU: Metal ✓")
@time de_sino = forward_project_dual_energy(
    mask_gpu, geom, protocol;
    materials = materials,
    scanner = scanner
);
println("  Low energy ($(protocol.low_kvp) kVp): $(size(de_sino.low))")
println("  High energy ($(protocol.high_kvp) kVp): $(size(de_sino.high))")

# Reconstruct low energy
println("\n[FDK] Reconstructing low energy ($(protocol.low_kvp) kVp)...")
println("  GPU: Metal ✓")
sino_low_gpu = MtlArray(de_sino.low);
@time recon_low_gpu = fdk_reconstruct(sino_low_gpu, geom, recon_size);
recon_low = Array(recon_low_gpu);

# Reconstruct high energy
println("\n[FDK] Reconstructing high energy ($(protocol.high_kvp) kVp)...")
println("  GPU: Metal ✓")
sino_high_gpu = MtlArray(de_sino.high);
@time recon_high_gpu = fdk_reconstruct(sino_high_gpu, geom, recon_size);
recon_high = Array(recon_high_gpu);

# Convert to HU using respective water attenuations
μ_water_low = mean(recon_low[:, :, center_z][water_mask])
μ_water_high = mean(recon_high[:, :, center_z][water_mask])

recon_low_hu = 1000f0 .* (recon_low .- μ_water_low) ./ μ_water_low
recon_high_hu = 1000f0 .* (recon_high .- μ_water_high) ./ μ_water_high

println("\nDual-energy results (center slice):")
println("  $(protocol.low_kvp) kVp: water=$(round(mean(recon_low_hu[:,:,center_z][water_mask]), digits=1)) HU")
println("  $(protocol.high_kvp) kVp: water=$(round(mean(recon_high_hu[:,:,center_z][water_mask]), digits=1)) HU")

# =============================================================================
# PART 3: Virtual Monoenergetic Imaging (70 keV)
# =============================================================================

println()
println("=" ^ 70)
println("PART 3: Virtual Monoenergetic Imaging (70 keV)")
println("=" ^ 70)

# Material decomposition
println("\nMaterial decomposition (water/iodine basis)...")
mat_map = decompose_materials(de_sino; basis=(:water, :iodine))
println("  Water map: mean=$(round(mean(mat_map.material1), digits=3))")
println("  Iodine map: mean=$(round(mean(mat_map.material2), digits=5))")

# VMI at 70 keV
println("\n[VMI] Reconstructing at 70 keV...")
println("  GPU: Metal ✓ (via sinogram synthesis)")
@time recon_vmi_70_hu = reconstruct_vmi(mat_map, 70.0, geom, recon_size;
    method=:fdk, to_hu=true)

println("\nVMI results (center slice):")
println("  70 keV: water=$(round(mean(recon_vmi_70_hu[:,:,center_z][water_mask]), digits=1)) HU")

# =============================================================================
# PART 4: Generate Comparison Figure
# =============================================================================

println()
println("=" ^ 70)
println("PART 4: Generating Figure")
println("=" ^ 70)

# Create figure
fig = Figure(size=(1400, 900), fontsize=12)

# Common display window
window_center = 50
window_width = 400
clim = (window_center - window_width/2, window_center + window_width/2)

# Row 1: Single-Energy
Label(fig[1, 1:2], "SINGLE-ENERGY (120 kVp)", fontsize=14)

ax1 = Axis(fig[2, 1], title="120 kVp - FDK", aspect=DataAspect())
heatmap!(ax1, recon_120_fdk_hu[:, :, center_z]', colormap=:grays, colorrange=clim)
hidedecorations!(ax1)

ax2 = Axis(fig[2, 2], title="120 kVp - SIRT (3 iter)", aspect=DataAspect())
heatmap!(ax2, recon_120_sirt_hu[:, :, center_z]', colormap=:grays, colorrange=clim)
hidedecorations!(ax2)

# Row 2: Dual-Energy
Label(fig[1, 3:4], "DUAL-ENERGY (80/140 kVp)", fontsize=14)

ax3 = Axis(fig[2, 3], title="80 kVp - FDK", aspect=DataAspect())
heatmap!(ax3, recon_low_hu[:, :, center_z]', colormap=:grays, colorrange=clim)
hidedecorations!(ax3)

ax4 = Axis(fig[2, 4], title="140 kVp - FDK", aspect=DataAspect())
heatmap!(ax4, recon_high_hu[:, :, center_z]', colormap=:grays, colorrange=clim)
hidedecorations!(ax4)

# Row 3: VMI and Stats
Label(fig[3, 1:2], "VIRTUAL MONOENERGETIC", fontsize=14)

ax5 = Axis(fig[4, 1], title="VMI 70 keV - FDK", aspect=DataAspect())
heatmap!(ax5, recon_vmi_70_hu[:, :, center_z]', colormap=:grays, colorrange=clim)
hidedecorations!(ax5)

# Stats panel
ax6 = Axis(fig[4, 2], title="HU Statistics (Water Region)")
hidedecorations!(ax6)
hidespines!(ax6)

stats_text = """
WATER HU (should be 0):

Single-Energy:
  120 kVp FDK:    $(round(mean(recon_120_fdk_hu[:,:,center_z][water_mask]), digits=1)) HU
  120 kVp SIRT:   $(round(mean(recon_120_sirt_hu[:,:,center_z][water_mask]), digits=1)) HU

Dual-Energy:
  80 kVp FDK:     $(round(mean(recon_low_hu[:,:,center_z][water_mask]), digits=1)) HU
  140 kVp FDK:    $(round(mean(recon_high_hu[:,:,center_z][water_mask]), digits=1)) HU

VMI:
  70 keV FDK:     $(round(mean(recon_vmi_70_hu[:,:,center_z][water_mask]), digits=1)) HU

All reconstructions on GPU (Metal)
"""
text!(ax6, 0.05, 0.95, text=stats_text, align=(:left, :top), fontsize=10)

# Colorbar
Colorbar(fig[2:4, 5], colorrange=clim, colormap=:grays, label="HU", height=Relative(0.8))

# Title
Label(fig[0, :], text="Comprehensive CT Reconstruction Comparison\nGammex 472 Phantom | GE Revolution Apex | Metal GPU",
      fontsize=16)

# Save
output_path = joinpath(@__DIR__, "comprehensive_recon_comparison_output.png")
save(output_path, fig, px_per_unit=2)
println("\nFigure saved: $output_path")

# =============================================================================
# Summary
# =============================================================================

println()
println("=" ^ 70)
println("SUMMARY")
println("=" ^ 70)
println()
println("Reconstructions completed:")
println("  1. Single-energy 120 kVp with FDK        ✓")
println("  2. Single-energy 120 kVp with SIRT (3i)  ✓")
println("  3. Dual-energy 80 kVp with FDK           ✓")
println("  4. Dual-energy 140 kVp with FDK          ✓")
println("  5. VMI 70 keV with FDK                   ✓")
println()
println("All operations ran on Metal GPU.")
println()
println("Output: $output_path")
println("=" ^ 70)
