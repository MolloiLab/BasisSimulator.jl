# =============================================================================
# Realistic Clinical CT Simulation
# =============================================================================
#
# High-resolution polychromatic CT simulation with full physics modeling.
#
# KEY FEATURE: Avoids "inverse crime" by using different resolutions for
# ground truth phantom vs reconstruction output (like real CT systems).
#
# Parameters:
# - Ground truth phantom: 1024x1024x40 (0.34 mm voxels) - high-res "physical" phantom
# - Reconstruction output: 512x512x20 (0.68 mm voxels) - clinical output resolution
# - Detector: 736x64 (typical clinical 64-slice CT)
# - Projections: 1160 angles (full rotation)
# - Spectrum: 120 kVp polychromatic (30 energy bins)
# - Physics: scatter, noise, crosstalk, focal spot blur, lag
#
# =============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# %%
using BasisSimulator
using Statistics
using Printf
using CairoMakie
using Metal

# =============================================================================
# 1. Configuration - Clinical CT Parameters
# =============================================================================

# %%
println("="^70)
println("Realistic Clinical CT Simulation")
println("="^70)
println("\nGPU: ", Metal.current_device())

# Clinical-grade parameters
CONFIG = (
    # Ground truth phantom (high resolution - like real physical phantom)
    phantom_n_voxels = 1024,    # 1024x1024 in-plane (high-res ground truth)
    phantom_n_slices = 40,      # 40 slices
    fov_cm = 35.0,              # 35 cm FOV (body imaging)
    z_cm = 2.0,                 # 2 cm z-coverage

    # Reconstruction volume (lower resolution - typical clinical output)
    recon_n_voxels = 512,       # 512x512 in-plane
    recon_n_slices = 20,        # 20 slices

    # Detector geometry
    n_cols = 736,               # Detector columns (typical clinical)
    n_rows = 64,                # Detector rows (64-slice CT)
    n_angles = 1160,            # Projections per rotation

    # Spectrum
    kvp = 120,                  # Tube voltage
    n_energy_bins = 30,         # Downsampled energy bins
    effective_keV = 70.0,       # Effective energy for HU reference
)

# Compute voxel sizes
phantom_voxel_mm = CONFIG.fov_cm * 10 / CONFIG.phantom_n_voxels
recon_voxel_mm = CONFIG.fov_cm * 10 / CONFIG.recon_n_voxels

println("\nSimulation Configuration:")
println("  Ground Truth Phantom:")
println("    Size:       $(CONFIG.phantom_n_voxels) x $(CONFIG.phantom_n_voxels) x $(CONFIG.phantom_n_slices)")
println("    Voxel size: $(round(phantom_voxel_mm, digits=2)) mm")
println("  Reconstruction Output:")
println("    Size:       $(CONFIG.recon_n_voxels) x $(CONFIG.recon_n_voxels) x $(CONFIG.recon_n_slices)")
println("    Voxel size: $(round(recon_voxel_mm, digits=2)) mm")
println("  Detector:     $(CONFIG.n_cols) x $(CONFIG.n_rows)")
println("  Angles:       $(CONFIG.n_angles)")
println("  Spectrum:     $(CONFIG.kvp) kVp ($(CONFIG.n_energy_bins) bins)")

# =============================================================================
# 2. Create Phantom and Geometry
# =============================================================================

# %%
println("\n" * "="^70)
println("Creating Phantom and Geometry")
println("="^70)

# Create HIGH-RESOLUTION phantom (like a real physical phantom)
phantom = create_gammex_472(
    n_voxels = CONFIG.phantom_n_voxels,
    n_slices = CONFIG.phantom_n_slices,
    fov_cm = CONFIG.fov_cm,
    z_cm = CONFIG.z_cm,
    μ_effective_energy_keV = CONFIG.effective_keV
)
println("\nHigh-res phantom: $(size(phantom.μ)) ($(round(phantom_voxel_mm, digits=2)) mm voxels)")

# Create downsampled mask for HU validation at reconstruction resolution
function downsample_mask(mask, new_size)
    old_size = size(mask)
    scale = old_size ./ new_size
    result = similar(mask, new_size)
    for k in 1:new_size[3], j in 1:new_size[2], i in 1:new_size[1]
        # Nearest neighbor sampling
        oi = clamp(round(Int, (i - 0.5) * scale[1] + 0.5), 1, old_size[1])
        oj = clamp(round(Int, (j - 0.5) * scale[2] + 0.5), 1, old_size[2])
        ok = clamp(round(Int, (k - 0.5) * scale[3] + 0.5), 1, old_size[3])
        result[i, j, k] = mask[oi, oj, ok]
    end
    return result
end

recon_size = (CONFIG.recon_n_voxels, CONFIG.recon_n_voxels, CONFIG.recon_n_slices)
mask_recon = downsample_mask(phantom.mask, recon_size)
println("Recon-resolution mask: $(size(mask_recon)) (for HU validation)")

# %%
geom = create_aquilion_one(
    n_angles = CONFIG.n_angles,
    n_rows = CONFIG.n_rows,
    n_cols = CONFIG.n_cols,
    fov_cm = CONFIG.fov_cm,
    z_cm = CONFIG.z_cm
)
println("Geometry: $(geom.n_cols) x $(geom.n_rows) x $(geom.n_angles) projections")
println("  SAD: $(geom.SAD) mm, SDD: $(geom.SDD) mm")

# =============================================================================
# 3. Load Spectrum and Materials
# =============================================================================

# %%
println("\n" * "="^70)
println("Loading Spectrum")
println("="^70)

energies_full, weights_full = load_spectrum(CONFIG.kvp)
energies, weights = downsample_spectrum(energies_full, weights_full, CONFIG.n_energy_bins)
materials = get_region_materials()

println("\nSpectrum: $(length(energies_full)) -> $(CONFIG.n_energy_bins) energy bins")
println("  Range: $(round(minimum(energies), digits=1)) - $(round(maximum(energies), digits=1)) keV")
println("  Mean energy: $(round(sum(energies .* weights) / sum(weights), digits=1)) keV")

# =============================================================================
# 4. Physics Configuration
# =============================================================================

# %%
println("\n" * "="^70)
println("Physics Configuration")
println("="^70)

physics = realistic_physics_config(
    scatter_scale = 1.0,     # Nominal scatter (~15% SPR)
    noise_level = 1.0,       # Clinical noise level
    noise_seed = 42          # Reproducibility
)

info = get_physics_config_info(physics)
println("\nEnabled physics effects:")
for effect in info.enabled_effects
    println("  - $effect")
end

# =============================================================================
# 5. Forward Projection (Polychromatic + Full Physics)
# =============================================================================

# %%
println("\n" * "="^70)
println("Forward Projection (Polychromatic + Physics)")
println("="^70)

# Transfer HIGH-RES phantom to GPU for forward projection
mask_gpu = MtlArray(phantom.mask)
println("\nHigh-res phantom mask transferred to GPU")

# Full simulation: polychromatic + all physics effects
println("\nRunning polychromatic forward projection with full physics...")
println("  (This may take a minute for high-resolution simulation)")

@time sinogram_gpu = forward_project(mask_gpu, geom;
    energies = energies,
    weights = weights,
    materials = materials,
    physics = physics
);

sinogram = Array(sinogram_gpu)
println("\nSinogram: $(size(sinogram))")
println("  Range: $(round(minimum(sinogram), digits=3)) - $(round(maximum(sinogram), digits=3))")
println("  Mean:  $(round(mean(sinogram), digits=3))")

# =============================================================================
# 6. Reconstruction - All Three Methods
# =============================================================================

# %%
println("\n" * "="^70)
println("Reconstruction (FDK, SIRT, CGLS)")
println("="^70)

# Reconstruct at LOWER resolution than phantom (avoids inverse crime)
volume_size = recon_size
println("\nReconstructing to $(volume_size[1])x$(volume_size[2])x$(volume_size[3]) ($(round(recon_voxel_mm, digits=2)) mm voxels)")

# %%
println("\n[1/3] FDK Reconstruction...")
@time recon_fdk_gpu = fdk_reconstruct(sinogram_gpu, geom, volume_size);
recon_fdk = Array(recon_fdk_gpu)
println("  Complete")

# %%
println("\n[2/3] SIRT Reconstruction (FDK init, 30 iterations)...")
@time recon_sirt_gpu = sirt_reconstruct(sinogram_gpu, geom, volume_size;
    niter = 30, init = :fdk, verbose = false);
recon_sirt = Array(recon_sirt_gpu);
println("  Complete")

# %%
println("\n[3/3] CGLS Reconstruction (FDK init, 15 iterations)...")
@time recon_cgls_gpu = cgls_reconstruct(sinogram_gpu, geom, volume_size;
    niter = 15, init = :fdk, verbose = false);
recon_cgls = Array(recon_cgls_gpu);
println("  Complete")

# =============================================================================
# 7. Convert to HU
# =============================================================================

# %%
μ_water = get_reference_μ_water(CONFIG.effective_keV)

recon_fdk_hu = μ_to_HU(recon_fdk, μ_water)
recon_sirt_hu = μ_to_HU(recon_sirt, μ_water)
recon_cgls_hu = μ_to_HU(recon_cgls, μ_water)

# Downsample ground truth for visualization (to match recon resolution)
function downsample_volume(vol::AbstractArray{T,3}, new_size) where T
    old_size = size(vol)
    scale = old_size ./ new_size
    result = similar(vol, new_size)
    for k in 1:new_size[3], j in 1:new_size[2], i in 1:new_size[1]
        oi = clamp(round(Int, (i - 0.5) * scale[1] + 0.5), 1, old_size[1])
        oj = clamp(round(Int, (j - 0.5) * scale[2] + 0.5), 1, old_size[2])
        ok = clamp(round(Int, (k - 0.5) * scale[3] + 0.5), 1, old_size[3])
        result[i, j, k] = vol[oi, oj, ok]
    end
    return result
end

phantom_recon_res = downsample_volume(phantom.μ, recon_size)
phantom_hu = μ_to_HU(phantom_recon_res, μ_water)

println("\nHU Conversion (reference: μ_water = $(round(μ_water, digits=4)) cm⁻¹ at $(CONFIG.effective_keV) keV)")
println("Ground truth downsampled to $(size(phantom_hu)) for visualization")

# =============================================================================
# 8. HU Validation
# =============================================================================

# %%
println("\n" * "="^70)
println("HU Validation")
println("="^70)

# Define validation regions
const REGIONS = [
    (name="Solid Water", id=REGION_SOLID_WATER, material=:solid_water),
    (name="Ca 50",       id=REGION_CA_50,       material=:Ca_50),
    (name="Ca 100",      id=REGION_CA_100,      material=:Ca_100),
    (name="Ca 200",      id=REGION_CA_200,      material=:Ca_200),
    (name="Ca 300",      id=REGION_CA_300,      material=:Ca_300),
    (name="Ca 400",      id=REGION_CA_400,      material=:Ca_400),
]

# Compute expected HU from physics
function expected_hu(material::Symbol, energy_keV::Float64)
    mat = get_material(material)
    μ_mat = compute_μ_at_energy(mat, energy_keV)
    μ_w = get_reference_μ_water(energy_keV)
    return μ_to_HU(μ_mat, μ_w)
end

# Measure HU in reconstruction
function measure_hu(vol, mask, region_id, μ_water)
    m = mask .== UInt8(region_id)
    sum(m) < 100 && return (mean=NaN, std=NaN)
    vals = vol[m]
    hu = μ_to_HU.(vals, μ_water)
    (mean=mean(hu), std=std(hu))
end

# Print table
println("\nRegion          | Expected |    FDK          |    SIRT         |    CGLS")
println("-"^80)

for r in REGIONS
    exp = round(Int, expected_hu(r.material, CONFIG.effective_keV))
    fdk = measure_hu(recon_fdk, mask_recon, r.id, μ_water)
    sirt = measure_hu(recon_sirt, mask_recon, r.id, μ_water)
    cgls = measure_hu(recon_cgls, mask_recon, r.id, μ_water)

    fdk_str = @sprintf("%4d ± %3d", round(Int, fdk.mean), round(Int, fdk.std))
    sirt_str = @sprintf("%4d ± %3d", round(Int, sirt.mean), round(Int, sirt.std))
    cgls_str = @sprintf("%4d ± %3d", round(Int, cgls.mean), round(Int, cgls.std))

    println("  $(rpad(r.name, 12)) |   $(lpad(exp, 4)) | $(fdk_str) | $(sirt_str) | $(cgls_str)")
end

# =============================================================================
# 9. Noise Analysis
# =============================================================================

# %%
println("\n" * "="^70)
println("Noise Analysis (Solid Water Region)")
println("="^70)

sw_fdk = measure_hu(recon_fdk, mask_recon, REGION_SOLID_WATER, μ_water)
sw_sirt = measure_hu(recon_sirt, mask_recon, REGION_SOLID_WATER, μ_water)
sw_cgls = measure_hu(recon_cgls, mask_recon, REGION_SOLID_WATER, μ_water)

println("\n  FDK:  $(round(sw_fdk.std, digits=1)) HU")
println("  SIRT: $(round(sw_sirt.std, digits=1)) HU ($(round(100*sw_sirt.std/sw_fdk.std))% of FDK)")
println("  CGLS: $(round(sw_cgls.std, digits=1)) HU ($(round(100*sw_cgls.std/sw_fdk.std))% of FDK)")

# =============================================================================
# 10. Visualization - Reconstruction Comparison
# =============================================================================

# %%
println("\n" * "="^70)
println("Generating Figures")
println("="^70)
begin
    slice = CONFIG.recon_n_slices ÷ 2
    hu_range = (-200, 500)

    fig1 = Figure(size=(1400, 400))

    Label(fig1[0, 1:4], "Polychromatic CT Reconstruction ($(CONFIG.kvp) kVp, Full Physics)",
        fontsize=20, tellwidth=false)

    ax1 = Axis(fig1[1, 1], title="Ground Truth", aspect=DataAspect())
    heatmap!(ax1, phantom_hu[:, :, slice], colormap=:grays, colorrange=hu_range)
    hidedecorations!(ax1)

    ax2 = Axis(fig1[1, 2], title="FDK", aspect=DataAspect())
    heatmap!(ax2, recon_fdk_hu[:, :, slice], colormap=:grays, colorrange=hu_range)
    hidedecorations!(ax2)

    ax3 = Axis(fig1[1, 3], title="SIRT (30 iter)", aspect=DataAspect())
    heatmap!(ax3, recon_sirt_hu[:, :, slice], colormap=:grays, colorrange=hu_range)
    hidedecorations!(ax3)

    ax4 = Axis(fig1[1, 4], title="CGLS (15 iter)", aspect=DataAspect())
    hm = heatmap!(ax4, recon_cgls_hu[:, :, slice], colormap=:grays, colorrange=hu_range)
    hidedecorations!(ax4)

    Colorbar(fig1[1, 5], hm, label="HU")

    display(fig1)
end

# =============================================================================
# 11. Visualization - HU Accuracy Bar Chart
# =============================================================================

# %%
begin
    fig2 = Figure(size=(900, 500))

    ax = Axis(fig2[1, 1],
        title = "HU Accuracy: Expected vs Measured ($(CONFIG.kvp) kVp)",
        xlabel = "Material",
        ylabel = "HU Value"
    )

    names = [r.name for r in REGIONS]
    expected_vals = [expected_hu(r.material, CONFIG.effective_keV) for r in REGIONS]
    fdk_vals = [measure_hu(recon_fdk, mask_recon, r.id, μ_water).mean for r in REGIONS]
    sirt_vals = [measure_hu(recon_sirt, mask_recon, r.id, μ_water).mean for r in REGIONS]
    cgls_vals = [measure_hu(recon_cgls, mask_recon, r.id, μ_water).mean for r in REGIONS]

    x = 1:length(names)
    w = 0.2

    barplot!(ax, x .- 1.5w, expected_vals, width=w, label="Expected", color=:gray60)
    barplot!(ax, x .- 0.5w, fdk_vals, width=w, label="FDK", color=:steelblue)
    barplot!(ax, x .+ 0.5w, sirt_vals, width=w, label="SIRT", color=:coral)
    barplot!(ax, x .+ 1.5w, cgls_vals, width=w, label="CGLS", color=:seagreen)

    ax.xticks = (x, names)
    ax.xticklabelrotation = π/6
    axislegend(ax, position=:lt)

    display(fig2)
end

# =============================================================================
# 12. Visualization - Sinogram
# =============================================================================

# %%
begin
    fig3 = Figure(size=(1000, 400))

    Label(fig3[0, 1:2], "Sinogram (Central Row)", fontsize=18, tellwidth=false)

    row_mid = CONFIG.n_rows ÷ 2

    ax1 = Axis(fig3[1, 1], title="Full Sinogram", xlabel="Detector Column", ylabel="Angle")
    hm1 = heatmap!(ax1, sinogram[:, row_mid, :]', colormap=:inferno)
    Colorbar(fig3[1, 2], hm1, label="Line Integral")

    # Profile at one angle
    ax2 = Axis(fig3[2, 1:2], title="Profile (angle = $(CONFIG.n_angles÷2))",
            xlabel="Detector Column", ylabel="Line Integral")
    lines!(ax2, sinogram[:, row_mid, CONFIG.n_angles÷2], linewidth=1.5, color=:steelblue)

    display(fig3)
end

# =============================================================================
# 13. Summary
# =============================================================================

# %%
println("\n" * "="^70)
println("Simulation Complete")
println("="^70)

println("\nConfiguration:")
println("  Ground Truth: $(CONFIG.phantom_n_voxels) x $(CONFIG.phantom_n_voxels) x $(CONFIG.phantom_n_slices) ($(round(phantom_voxel_mm, digits=2)) mm)")
println("  Recon Output: $(CONFIG.recon_n_voxels) x $(CONFIG.recon_n_voxels) x $(CONFIG.recon_n_slices) ($(round(recon_voxel_mm, digits=2)) mm)")
println("  Detector:     $(CONFIG.n_cols) x $(CONFIG.n_rows)")
println("  Angles:       $(CONFIG.n_angles)")
println("  Spectrum:     $(CONFIG.kvp) kVp polychromatic")

println("\nPhysics Effects:")
for effect in info.enabled_effects
    println("  - $effect")
end

println("\nReconstruction Methods:")
println("  - FDK (analytical)")
println("  - SIRT (30 iterations, FDK init)")
println("  - CGLS (15 iterations, FDK init)")

println("\nNoise Reduction vs FDK:")
println("  - SIRT: $(round(100*sw_sirt.std/sw_fdk.std))%")
println("  - CGLS: $(round(100*sw_cgls.std/sw_fdk.std))%")

println("\nAll operations GPU-accelerated via Metal + AcceleratedKernels.jl")
