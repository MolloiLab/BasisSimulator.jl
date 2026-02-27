# ============================================================================
# BasisSimulator.jl — Unfiltered Spectrum + Filtering Pipeline Demo
# GE Revolution Apex (Gemstone) with configurable beam filtration
# Hamidreza Khodajou-Chokami, PhD
# Demonstrates:
#   Part A: Raw vs filtered spectrum visualization
#   Part B: Effect of filter materials on spectrum shape / mean energy
#   Part C: kVp comparison (80 vs 120 vs 140) with same filtration
#   Part D: Anode angle comparison (8° vs 10°)
#   Part E: Full CT simulation comparing real vs generic spectrum
#   Part F: Parameter sensitivity — how mA and filtration affect CT images
#   Part G: Distance scaling (SDD effect on flux)
#   Part H: Flat filter vs additional filters comparison
#
# Run with: julia --project=. run_demo_spectrum.jl
# ============================================================================

println("="^72)
println("  BasisSimulator.jl — Unfiltered Spectrum + Beam Filtration Demo")
println("  GE Revolution Apex with Gemstone Detector")
println("="^72)

# ─── Load packages ───────────────────────────────────────────────────────────
println("\n[1/10] Loading packages...")
using CUDA
using BasisSimulator
import BasisSimulator as BS
import XrayAttenuation as XA
using CairoMakie
using Statistics

# ─── Check GPU ───────────────────────────────────────────────────────────────
println("\n[2/10] Checking CUDA GPU...")
if CUDA.functional()
    println("  ✓ CUDA is functional")
    println("  GPU: ", CUDA.name(CUDA.device()))
    println("  Memory: ", round(CUDA.totalmem(CUDA.device()) / 1024^3, digits=1), " GB")
else
    println("  ⚠ CUDA not available — using CPU")
end

# Output directory
output_dir = joinpath(@__DIR__, "output_spectrum_demo")
mkpath(output_dir)

# ═══════════════════════════════════════════════════════════════════════════════
# PART A: Raw vs Filtered Spectrum Visualization
# ═══════════════════════════════════════════════════════════════════════════════
println("\n[3/10] Part A: Raw vs Filtered Spectrum @ 120 kVp...")

# Load unfiltered spectrum (raw tube output at 750mm)
e_raw, f_raw = BS.load_spectrum_unfiltered(120; anode_angle=10)
println("  Raw spectrum: $(length(e_raw)) bins, $(e_raw[1])–$(e_raw[end]) keV")
println("  Raw mean energy: $(round(BS.spectrum_mean_energy(e_raw, f_raw), digits=1)) keV")

# Typical clinical filtration: 2.5mm Al inherent + 0.1mm Cu added
clinical_filters = [("Al", 2.5), ("Cu", 0.1)]
e_filt1, f_filt1, flux1 = BS.filter_spectrum(e_raw, f_raw; filters=clinical_filters, sdd_mm=750.0)
println("  After 2.5mm Al + 0.1mm Cu: mean = $(round(BS.spectrum_mean_energy(e_filt1, f_filt1), digits=1)) keV")

# Heavy filtration: 2.5mm Al + 0.2mm Cu + 0.1mm Sn
heavy_filters = [("Al", 2.5), ("Cu", 0.2), ("Sn", 0.1)]
e_filt2, f_filt2, flux2 = BS.filter_spectrum(e_raw, f_raw; filters=heavy_filters, sdd_mm=750.0)
println("  After 2.5mm Al + 0.2mm Cu + 0.1mm Sn: mean = $(round(BS.spectrum_mean_energy(e_filt2, f_filt2), digits=1)) keV")

# --- Plot A1: Raw vs Filtered Spectra ---
fig_a = Figure(size=(1400, 500), backgroundcolor=:white)

ax_a1 = Axis(fig_a[1, 1],
    title="X-ray Spectra: Unfiltered vs Clinically Filtered",
    xlabel="Energy (keV)", ylabel="Photon Flux (photons/mA/s/mm²)",
    titlesize=18)

lines!(ax_a1, e_raw, f_raw, color=(:gray50, 0.5), linewidth=1.5, label="Unfiltered (raw tube)")
lines!(ax_a1, e_filt1, f_filt1, color=:steelblue, linewidth=2.0, label="2.5mm Al + 0.1mm Cu")
lines!(ax_a1, e_filt2, f_filt2, color=:crimson, linewidth=2.0, label="2.5mm Al + 0.2mm Cu + 0.1mm Sn")
xlims!(ax_a1, 0, 125)
axislegend(ax_a1, position=:rt)

# --- Plot A2: Normalized (spectral shape) ---
ax_a2 = Axis(fig_a[1, 2],
    title="Spectral Shape (Normalized)",
    xlabel="Energy (keV)", ylabel="Relative Intensity",
    titlesize=18)

w_raw = f_raw ./ maximum(f_raw)
w_filt1 = f_filt1 ./ maximum(f_filt1)
w_filt2 = f_filt2 ./ maximum(f_filt2)

lines!(ax_a2, e_raw, w_raw, color=(:gray50, 0.5), linewidth=1.5, label="Unfiltered")
lines!(ax_a2, e_filt1, w_filt1, color=:steelblue, linewidth=2.0, label="Clinical (Al+Cu)")
lines!(ax_a2, e_filt2, w_filt2, color=:crimson, linewidth=2.0, label="Heavy (Al+Cu+Sn)")

# Mark mean energies
mean_raw = BS.spectrum_mean_energy(e_raw, f_raw)
mean_f1 = BS.spectrum_mean_energy(e_filt1, f_filt1)
mean_f2 = BS.spectrum_mean_energy(e_filt2, f_filt2)
vlines!(ax_a2, [mean_raw], color=:gray50, linestyle=:dash, linewidth=1)
vlines!(ax_a2, [mean_f1], color=:steelblue, linestyle=:dash, linewidth=1)
vlines!(ax_a2, [mean_f2], color=:crimson, linestyle=:dash, linewidth=1)
text!(ax_a2, mean_raw + 1, 0.95, text="$(round(mean_raw, digits=0)) keV", fontsize=10, color=:gray50)
text!(ax_a2, mean_f1 + 1, 0.85, text="$(round(mean_f1, digits=0)) keV", fontsize=10, color=:steelblue)
text!(ax_a2, mean_f2 + 1, 0.75, text="$(round(mean_f2, digits=0)) keV", fontsize=10, color=:crimson)
xlims!(ax_a2, 0, 125)
axislegend(ax_a2, position=:rt)

save(joinpath(output_dir, "01_raw_vs_filtered_spectrum.png"), fig_a, px_per_unit=2)
display(fig_a)
println("  ✓ Saved: 01_raw_vs_filtered_spectrum.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PART B: Filter Material Comparison — Mean Energy Shift
# ═══════════════════════════════════════════════════════════════════════════════
println("\n[4/9] Part B: Filter material comparison...")

# Sweep filter thicknesses for Al, Cu, Sn
filter_configs = [
    ("Al", :steelblue, [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]),
    ("Cu", :darkorange, [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]),
    ("Sn", :mediumpurple, [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]),
]

fig_b = Figure(size=(1400, 900), backgroundcolor=:white)

# Top: Mean energy vs filter thickness
ax_b1 = Axis(fig_b[1, 1:2],
    title="Mean Energy vs Filter Thickness (120 kVp, Anode 10°)",
    xlabel="Filter Thickness (mm)", ylabel="Mean Energy (keV)",
    titlesize=18)

for (mat, color, thicknesses) in filter_configs
    mean_energies = Float64[]
    for t in thicknesses
        _, f_out, _ = BS.filter_spectrum(e_raw, f_raw; filters=[(mat, t)], sdd_mm=750.0)
        push!(mean_energies, BS.spectrum_mean_energy(e_raw, f_out))
    end
    scatterlines!(ax_b1, thicknesses, mean_energies, color=color, linewidth=2,
        markersize=10, label="$mat filter")
end
axislegend(ax_b1, position=:rb)

# Bottom left: Flux attenuation vs thickness
ax_b2 = Axis(fig_b[2, 1],
    title="Total Flux Attenuation vs Filter Thickness",
    xlabel="Filter Thickness (mm)", ylabel="Remaining Flux (%)",
    titlesize=16)

for (mat, color, thicknesses) in filter_configs
    _, f_base, flux_base = BS.filter_spectrum(e_raw, f_raw; filters=Tuple{String,Float64}[], sdd_mm=750.0)
    flux_pcts = Float64[]
    for t in thicknesses
        _, _, flux_out = BS.filter_spectrum(e_raw, f_raw; filters=[(mat, t)], sdd_mm=750.0)
        push!(flux_pcts, 100.0 * flux_out / flux_base)
    end
    scatterlines!(ax_b2, thicknesses, flux_pcts, color=color, linewidth=2,
        markersize=8, label="$mat")
end
axislegend(ax_b2, position=:rt)

# Bottom right: μ(E) curves for all filter materials
ax_b3 = Axis(fig_b[2, 2],
    title="Linear Attenuation μ(E) — Filter Materials",
    xlabel="Energy (keV)", ylabel="μ (cm⁻¹)",
    titlesize=16, yscale=log10)

energies_mu = collect(5.0:1.0:140.0)
for (mat, color, _) in filter_configs
    μ_vals = [BS.get_filter_mu(mat, E) for E in energies_mu]
    lines!(ax_b3, energies_mu, μ_vals, color=color, linewidth=2, label=mat)
end
axislegend(ax_b3, position=:rt)

save(joinpath(output_dir, "02_filter_material_comparison.png"), fig_b, px_per_unit=2)
display(fig_b)
println("  ✓ Saved: 02_filter_material_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PART C: kVp Comparison (80, 100, 120, 140) with Same Filtration
# ═══════════════════════════════════════════════════════════════════════════════
println("\n[5/9] Part C: kVp comparison with identical filtration...")

fig_c = Figure(size=(1400, 500), backgroundcolor=:white)

kvp_values = [80, 100, 120, 140]
kvp_colors = [:teal, :steelblue, :darkorange, :crimson]
std_filters = [("Al", 2.5)]  # Standard 2.5mm Al inherent

ax_c1 = Axis(fig_c[1, 1],
    title="Filtered Spectra at Different kVp (2.5mm Al, Anode 10°)",
    xlabel="Energy (keV)", ylabel="Photon Flux (photons/mA/s/mm²)",
    titlesize=18)

ax_c2 = Axis(fig_c[1, 2],
    title="Spectral Statistics Summary",
    xlabel="kVp", ylabel="Energy (keV)",
    titlesize=18,
    xticks=(1:4, string.(kvp_values)))

mean_e_list = Float64[]
peak_e_list = Float64[]
total_flux_list = Float64[]

for (i, kVp) in enumerate(kvp_values)
    e_kv, f_kv = BS.load_spectrum_unfiltered(kVp; anode_angle=10)
    _, f_out, flux_out = BS.filter_spectrum(e_kv, f_kv; filters=std_filters, sdd_mm=750.0)

    lines!(ax_c1, e_kv, f_out, color=kvp_colors[i], linewidth=2.0, label="$(kVp) kVp")

    me = BS.spectrum_mean_energy(e_kv, f_out)
    push!(mean_e_list, me)
    push!(peak_e_list, e_kv[argmax(f_out)])
    push!(total_flux_list, flux_out)

    println("  $(kVp) kVp: mean=$(round(me, digits=1)) keV, peak=$(e_kv[argmax(f_out)]) keV, flux=$(round(flux_out, sigdigits=3))")
end
xlims!(ax_c1, 0, 145)
axislegend(ax_c1, position=:rt)

# Bar chart of mean energy and peak energy
barplot!(ax_c2, 1:4, mean_e_list, color=kvp_colors, label="Mean Energy",
    bar_labels=:y, label_formatter=x -> "$(round(x, digits=0))")
scatter!(ax_c2, 1:4, peak_e_list, color=:black, markersize=12, marker=:diamond, label="Peak Energy")
axislegend(ax_c2, position=:lt)

save(joinpath(output_dir, "03_kvp_comparison.png"), fig_c, px_per_unit=2)
display(fig_c)
println("  ✓ Saved: 03_kvp_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PART D: Anode Angle Comparison (8° vs 10°)
# ═══════════════════════════════════════════════════════════════════════════════
println("\n[6/9] Part D: Anode angle comparison (8° vs 10°)...")

fig_d = Figure(size=(800, 500), backgroundcolor=:white)
ax_d = Axis(fig_d[1, 1],
    title="120 kVp Spectrum: Anode 8° vs 10° (2.5mm Al filter)",
    xlabel="Energy (keV)", ylabel="Photon Flux (photons/mA/s/mm²)",
    titlesize=18)

for (angle, color, lstyle) in [(8, :steelblue, :solid), (10, :darkorange, :solid)]
    e_a, f_a = BS.load_spectrum_unfiltered(120; anode_angle=angle)
    _, f_af, flux_af = BS.filter_spectrum(e_a, f_a; filters=[("Al", 2.5)], sdd_mm=750.0)
    me = BS.spectrum_mean_energy(e_a, f_af)

    lines!(ax_d, e_a, f_af, color=color, linewidth=2.0, linestyle=lstyle,
        label="Anode $(angle)° — mean $(round(me, digits=1)) keV")

    println("  Anode $(angle)°: mean=$(round(me, digits=1)) keV, total_flux=$(round(flux_af, sigdigits=3))")
end
xlims!(ax_d, 0, 125)
axislegend(ax_d, position=:rt)

save(joinpath(output_dir, "04_anode_angle_comparison.png"), fig_d, px_per_unit=2)
display(fig_d)
println("  ✓ Saved: 04_anode_angle_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PART E: CatSim Baseline vs New Implementation (Bone Window)
#
#   Run A — CatSim baseline: GOS detector, generic pre-filtered spectrum,
#           generic flux 2E6, no additional filters.
#   Run B — New pipeline:    Gemstone detector, real anode spectrum (10°),
#           clinical filtration (2.5mm Al + 0.1mm Cu).
# ═══════════════════════════════════════════════════════════════════════════════
println("\n[7/9] Part E: CatSim baseline vs New implementation (bone window)...")

# Bone window for all CT images (WW=2000, WL=500)
ww, wl = 2000.0, 500.0
wmin_bone = wl - ww / 2    # -500 HU
wmax_bone = wl + ww / 2    # 1500 HU

# Create phantom
println("  Creating Gammex 472 phantom (256×256×16 for speed)...")
phantom_cpu = BS.create_gammex_472(n_voxels=256, n_slices=16, fov_cm=35.0, z_cm=4.0)

if CUDA.functional()
    phantom_gpu_mask = CuArray(phantom_cpu.mask)
    phantom = BS.Phantom(
        phantom_gpu_mask, phantom_cpu.materials,
        phantom_cpu.voxel_size, phantom_cpu.origin, phantom_cpu.fov)
else
    phantom = phantom_cpu
end

recon_opts = BS.ReconOptions(
    algorithm=:fdk,
    matrix_size=(256, 256, 16),
    fov_cm=35.0,
)

# ─── Scanner A: CatSim Baseline (GOS detector, generic scanner) ─────────────
scanner_catsim = BS.Scanner(
    source_to_isocenter=626.0,
    source_to_detector=1097.0,
    detector_rows=16,
    detector_cols=832,
    detector_row_size=0.625,
    detector_col_size=0.625,
    detector_depth=3.0,
    detector_material=:GOS,          # ← GOS scintillator (old CatSim default)
    fill_factor_row=0.90,
    fill_factor_col=0.90,
    target_angle=10.0,
    focal_spot_width=1.0,
    focal_spot_length=0.7,
    flat_filter_material=:aluminum,
    flat_filter_thickness=7.0,
)

# ─── Scanner B: New Gemstone Implementation ──────────────────────────────────
scanner_gemstone = BS.Scanner(
    source_to_isocenter=626.0,
    source_to_detector=1097.0,
    detector_rows=16,
    detector_cols=832,
    detector_row_size=0.625,
    detector_col_size=0.625,
    detector_depth=3.0,
    detector_material=:lumex,        # ← Gemstone Ce:(Tb,Lu)₃Al₅O₁₂
    fill_factor_row=0.90,
    fill_factor_col=0.90,
    target_angle=10.0,
    focal_spot_width=1.0,
    focal_spot_length=0.7,
    flat_filter_material=:aluminum,
    flat_filter_thickness=7.0,
)

# --- Run A: CatSim Baseline (GOS, unfiltered pre-filtered spectrum, flux 2E6) ---
println("  Run A: CatSim baseline (GOS, generic spectrum, flux=2E6)...")
protocol_catsim = BS.CTProtocol(
    kVp=120.0, mA=300.0, views=1000, rotation_time=1.0,
    flux_density=2.0e6          # ← generic constant flux
)
sim_opts_catsim = BS.SimOptions(fidelity=:high, seed=42, use_real_spectrum=false)

t_a = @elapsed begin
    result_catsim = BS.simulate(phantom, scanner_catsim, protocol_catsim, sim_opts_catsim, recon_opts)
end
println("  Run A done in $(round(t_a, digits=2))s")

# --- Run B: New Pipeline (Gemstone, real spectrum, clinical filtration) ---
println("  Run B: New pipeline (Gemstone, real Anode 10°, 2.5mm Al + 0.1mm Cu)...")
protocol_new = BS.CTProtocol(
    kVp=120.0, mA=300.0, views=1000, rotation_time=1.0,
    anode_angle=10,
    additional_filters=[("Al", 2.5), ("Cu", 0.1)]
)
sim_opts_new = BS.SimOptions(fidelity=:high_plus, seed=42)

t_b = @elapsed begin
    result_new = BS.simulate(phantom, scanner_gemstone, protocol_new, sim_opts_new, recon_opts)
end
println("  Run B done in $(round(t_b, digits=2))s")

# Convert both to HU
μ_water = BS.get_reference_μ_water(60.0)
hu_catsim = BS.to_hounsfield(Array(result_catsim.reconstruction); μ_water=μ_water)
hu_new = BS.to_hounsfield(Array(result_new.reconstruction); μ_water=μ_water)

nz = size(hu_catsim, 3)
mid = nz ÷ 2
nx = size(hu_catsim, 1)

# --- Plot E: Side-by-side in Bone Window ---
fig_e = Figure(size=(1400, 900), backgroundcolor=:black)
Label(fig_e[0, 1:3],
    "Before vs After: CatSim Baseline → New Spectrum+Gemstone (Bone W=$(Int(ww))/L=$(Int(wl)))",
    fontsize=20, color=:white, font=:bold)

for (col, hu_data, title_text) in [
    (1, hu_catsim, "BEFORE: CatSim (GOS, generic flux)"),
    (2, hu_new, "AFTER: Gemstone + Real Spectrum")]
    ax = Axis(fig_e[1, col],
        title=title_text, titlecolor=:white, titlesize=16,
        aspect=DataAspect(),
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false,
        bottomspinevisible=false, topspinevisible=false,
        leftspinevisible=false, rightspinevisible=false)
    s = clamp.(hu_data[:, :, mid], wmin_bone, wmax_bone)
    s_norm = (s .- wmin_bone) ./ (wmax_bone - wmin_bone)
    heatmap!(ax, s_norm', colormap=:grays, colorrange=(0, 1))
end

# Difference map
ax_diff = Axis(fig_e[1, 3],
    title="Difference (New − CatSim)", titlecolor=:white, titlesize=16,
    aspect=DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false,
    bottomspinevisible=false, topspinevisible=false,
    leftspinevisible=false, rightspinevisible=false)
diff_map = hu_new[:, :, mid] .- hu_catsim[:, :, mid]
heatmap!(ax_diff, diff_map', colormap=:RdBu, colorrange=(-100, 100))
Colorbar(fig_e[1, 4], limits=(-100, 100), colormap=:RdBu, label="ΔHU",
    labelcolor=:white, ticklabelcolor=:white)

# HU profiles
ax_prof = Axis(fig_e[2, 1:3],
    title="Horizontal HU Profile Through Center (Bone Window)",
    titlecolor=:white, titlesize=16,
    xlabel="Pixel", ylabel="HU",
    xlabelcolor=:white, ylabelcolor=:white,
    xticklabelcolor=:white, yticklabelcolor=:white,
    backgroundcolor=:gray10)

profile_catsim = hu_catsim[nx÷2, :, mid]
profile_new = hu_new[nx÷2, :, mid]

lines!(ax_prof, 1:length(profile_catsim), profile_catsim, color=:steelblue, linewidth=2, label="CatSim (GOS, generic)")
lines!(ax_prof, 1:length(profile_new), profile_new, color=:darkorange, linewidth=2, label="New (Gemstone, real spectrum)")
hlines!(ax_prof, [0.0], color=:white, linestyle=:dash, linewidth=1, label="Water (0 HU)")
axislegend(ax_prof, position=:rt, backgroundcolor=(:gray20, 0.8), labelcolor=:white)

save(joinpath(output_dir, "05_catsim_vs_new_pipeline_ct.png"), fig_e, px_per_unit=2)
display(fig_e)
println("  ✓ Saved: 05_catsim_vs_new_pipeline_ct.png")

# Print HU statistics
println("  --- HU Statistics (center slice) ---")
mask_cpu = Array(phantom.mask)
for (name, hu) in [("CatSim (GOS)", hu_catsim), ("New (Gemstone)", hu_new)]
    center_val = hu[nx÷2, size(hu, 2)÷2, mid]
    println("    $name: center pixel = $(round(center_val, digits=1)) HU")
    println("    $name: range = [$(round(minimum(hu[:,:,mid]), digits=0)), $(round(maximum(hu[:,:,mid]), digits=0))]")
end


# ═══════════════════════════════════════════════════════════════════════════════
# PART F: Filtration Effect on CT Image — Bone Window
# ═══════════════════════════════════════════════════════════════════════════════
println("\n[8/10] Part F: Filtration effect on CT image quality (bone window)...")

# Light filtration: just 2.5mm Al
println("  Run: Light filtration (2.5mm Al only)...")
protocol_light = BS.CTProtocol(
    kVp=120.0, mA=300.0, views=1000, rotation_time=1.0,
    anode_angle=10,
    additional_filters=[("Al", 2.5)]
)

t_light = @elapsed begin
    result_light = BS.simulate(phantom, scanner_gemstone, protocol_light, sim_opts_new, recon_opts)
end
hu_light = BS.to_hounsfield(Array(result_light.reconstruction); μ_water=μ_water)
println("  Done in $(round(t_light, digits=2))s")

# Heavy filtration: 2.5mm Al + 0.3mm Cu + 0.1mm Sn
println("  Run: Heavy filtration (2.5mm Al + 0.3mm Cu + 0.1mm Sn)...")
protocol_heavy = BS.CTProtocol(
    kVp=120.0, mA=300.0, views=1000, rotation_time=1.0,
    anode_angle=10,
    additional_filters=[("Al", 2.5), ("Cu", 0.3), ("Sn", 0.1)]
)

t_heavy = @elapsed begin
    result_heavy = BS.simulate(phantom, scanner_gemstone, protocol_heavy, sim_opts_new, recon_opts)
end
hu_heavy = BS.to_hounsfield(Array(result_heavy.reconstruction); μ_water=μ_water)
println("  Done in $(round(t_heavy, digits=2))s")

# --- Plot F: Side-by-side with noise ROI analysis (Bone Window) ---
fig_f = Figure(size=(1400, 900), backgroundcolor=:black)
Label(fig_f[0, 1:3],
    "Effect of Beam Filtration on CT Image — 120 kVp, Gemstone Detector (Bone W=$(Int(ww))/L=$(Int(wl)))",
    fontsize=20, color=:white, font=:bold)

for (col, hu_data, lbl) in [
    (1, hu_light, "Light (2.5mm Al)"),
    (2, hu_new, "Clinical (2.5mm Al + 0.1mm Cu)"),
    (3, hu_heavy, "Heavy (2.5mm Al + 0.3mm Cu + 0.1mm Sn)")]
    ax = Axis(fig_f[1, col],
        title=lbl, titlecolor=:white, titlesize=14,
        aspect=DataAspect(),
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false,
        bottomspinevisible=false, topspinevisible=false,
        leftspinevisible=false, rightspinevisible=false)
    s = clamp.(hu_data[:, :, mid], wmin_bone, wmax_bone)
    s_norm = (s .- wmin_bone) ./ (wmax_bone - wmin_bone)
    heatmap!(ax, s_norm', colormap=:grays, colorrange=(0, 1))

    # Measure noise in center ROI
    cx, cy = nx ÷ 2, size(hu_data, 2) ÷ 2
    r = 10  # ROI half-width (pixels)
    x1, x2 = max(1, cx - r), min(nx, cx + r)
    y1, y2 = max(1, cy - r), min(size(hu_data, 2), cy + r)
    roi = hu_data[x1:x2, y1:y2, mid]
    σ = std(roi)
    μ_roi = mean(roi)

    # Draw ROI rectangle on the image
    lines!(ax, [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1],
        color=:lime, linewidth=1.5)

    # Display stats next to ROI with "Water ROI" label
    text!(ax, x2 + 3, y1, text="Water ROI\nσ = $(round(σ, digits=1)) HU\nμ = $(round(μ_roi, digits=1)) HU",
        fontsize=10, color=:lime)
    println("    $lbl: Water ROI mean=$(round(μ_roi, digits=1)) HU, noise σ=$(round(σ, digits=1)) HU")
end

# Spectra comparison subplot
ax_spec = Axis(fig_f[2, 1:3],
    title="Corresponding Filtered Spectra (normalized)",
    titlecolor=:white, titlesize=16,
    xlabel="Energy (keV)", ylabel="Relative Intensity",
    xlabelcolor=:white, ylabelcolor=:white,
    xticklabelcolor=:white, yticklabelcolor=:white,
    backgroundcolor=:gray10)

e_base, f_base = BS.load_spectrum_unfiltered(120; anode_angle=10)

spec_configs = [
    ([("Al", 2.5)], :cyan, "Light (2.5mm Al)"),
    ([("Al", 2.5), ("Cu", 0.1)], :steelblue, "Clinical (Al+Cu)"),
    ([("Al", 2.5), ("Cu", 0.3), ("Sn", 0.1)], :orange, "Heavy (Al+Cu+Sn)"),
]

for (filters, color, label) in spec_configs
    _, f_out, _ = BS.filter_spectrum(e_base, f_base; filters=filters, sdd_mm=750.0)
    f_norm = f_out ./ maximum(f_out)
    lines!(ax_spec, e_base, f_norm, color=color, linewidth=2, label=label)
    me = BS.spectrum_mean_energy(e_base, f_out)
    vlines!(ax_spec, [me], color=color, linestyle=:dash, linewidth=1)
end
xlims!(ax_spec, 0, 125)
axislegend(ax_spec, position=:rt, backgroundcolor=(:gray20, 0.8), labelcolor=:white)

save(joinpath(output_dir, "06_filtration_effect_ct.png"), fig_f, px_per_unit=2)
display(fig_f)
println("  ✓ Saved: 06_filtration_effect_ct.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PART G: Distance Scaling (SDD effect on flux)
# ═══════════════════════════════════════════════════════════════════════════════
println("\n[9/10] Part G: Distance scaling and summary tables...")

fig_g = Figure(size=(1000, 500), backgroundcolor=:white)
ax_g = Axis(fig_g[1, 1],
    title="Flux vs Source-to-Detector Distance (120 kVp, 2.5mm Al)",
    xlabel="SDD (mm)", ylabel="Total Flux (relative to 750mm)",
    titlesize=18)

sdd_values = collect(500.0:50.0:1500.0)

# Compute reference flux at 750mm (first SDD value that matches or use first entry)
_, _, flux_ref = BS.filter_spectrum(e_raw, f_raw; filters=[("Al", 2.5)], sdd_mm=750.0)

flux_ratios = Float64[]
for sdd in sdd_values
    _, _, flux_sdd = BS.filter_spectrum(e_raw, f_raw; filters=[("Al", 2.5)], sdd_mm=sdd)
    push!(flux_ratios, flux_sdd / flux_ref)
end

lines!(ax_g, sdd_values, flux_ratios, color=:steelblue, linewidth=2.5)
vlines!(ax_g, [750.0], color=:gray50, linestyle=:dash, linewidth=1, label="Reference (750mm)")
vlines!(ax_g, [1097.0], color=:crimson, linestyle=:dash, linewidth=1, label="GE Apex SDD (1097mm)")
scatter!(ax_g, [1097.0], [flux_ratios[findfirst(x -> x >= 1097.0, sdd_values)]],
    color=:crimson, markersize=12)
axislegend(ax_g, position=:rt)

# Theoretical curve overlay
theory_ratios = (750.0 ./ sdd_values) .^ 2
lines!(ax_g, sdd_values, theory_ratios, color=:gray50, linewidth=1, linestyle=:dot,
    label="1/r² law")

save(joinpath(output_dir, "07_flux_vs_sdd.png"), fig_g, px_per_unit=2)
display(fig_g)
println("  ✓ Saved: 07_flux_vs_sdd.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PART H: Flat Filter (Projection Domain) vs Additional Filters (Spectrum Domain)
#
#   This is the key comparison for retiring flat_filter.jl:
#
#   OLD approach (flat_filter):
#     - Applies a MONOCHROMATIC approximation at a single reference energy (60 keV)
#     - Adds fixed attenuation offset to SINOGRAM values (projection domain)
#     - Not physically meaningful: real filters shape the SPECTRUM, not the sinogram
#
#   NEW approach (additional_filters):
#     - Loads UNFILTERED spectra from Anode .TXT files
#     - Applies Beer-Lambert law energy-by-energy: T(E) = exp(-Σ μᵢ(E)×tᵢ)
#     - Physically correct: filters the spectrum BEFORE simulation
#
# ═══════════════════════════════════════════════════════════════════════════════
println("\n[10/10] Part H: Flat Filter (projection domain) vs Additional Filters (spectrum domain)...")

# --- OLD approach: projection-domain flat_filter ---
# Same scanner, but enable the deprecated flat_filter in SimOptions
println("  Run OLD: Projection-domain flat_filter (7mm Al, monochromatic @ 60 keV)...")
protocol_old = BS.CTProtocol(
    kVp=120.0, mA=300.0, views=1000, rotation_time=1.0,
    flux_density=2.0e6
)
sim_opts_old = BS.SimOptions(
    fidelity=:high, seed=42,
    use_real_spectrum=false,
    use_flat_filter=true    # ← OLD: projection-domain filtering
)

t_old = @elapsed begin
    result_old = BS.simulate(phantom, scanner_gemstone, protocol_old, sim_opts_old, recon_opts)
end
hu_old = BS.to_hounsfield(Array(result_old.reconstruction); μ_water=μ_water)
println("  OLD done in $(round(t_old, digits=2))s")

# --- NEW approach: spectrum-domain additional_filters ---
# Same scanner, but use the physics-based spectrum pipeline
println("  Run NEW: Spectrum-domain additional_filters (7mm Al, Beer-Lambert per energy)...")
protocol_phys = BS.CTProtocol(
    kVp=120.0, mA=300.0, views=1000, rotation_time=1.0,
    anode_angle=10,
    additional_filters=[("Al", 7.0)]   # ← same 7mm Al, but applied to spectrum
)
sim_opts_phys = BS.SimOptions(
    fidelity=:high_plus, seed=42    # ← NEW: :high_plus = real spectrum + spectrum-domain filtering
)

t_phys = @elapsed begin
    result_phys = BS.simulate(phantom, scanner_gemstone, protocol_phys, sim_opts_phys, recon_opts)
end
hu_phys = BS.to_hounsfield(Array(result_phys.reconstruction); μ_water=μ_water)
println("  NEW done in $(round(t_phys, digits=2))s")

# --- Figure H: 3-panel comparison ---
fig_h = Figure(size=(1800, 1100), backgroundcolor=:black)

Label(fig_h[0, 1:4],
    "Flat Filter Comparison: Projection-Domain (OLD) vs Spectrum-Domain (NEW)",
    fontsize=22, color=:white, font=:bold)

nz_h = size(hu_old, 3)
mid_h = nz_h ÷ 2
nx_h = size(hu_old, 1)
ny_h = size(hu_old, 2)

# Panel 1: OLD (projection-domain flat_filter)
ax_h1 = Axis(fig_h[1, 1],
    title="OLD: Projection-Domain Flat Filter\n(monochromatic μ at 60 keV → sinogram)",
    titlecolor=:white, titlesize=14,
    aspect=DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false,
    bottomspinevisible=false, topspinevisible=false,
    leftspinevisible=false, rightspinevisible=false)
s_old = clamp.(hu_old[:, :, mid_h], wmin_bone, wmax_bone)
s_old_norm = (s_old .- wmin_bone) ./ (wmax_bone - wmin_bone)
heatmap!(ax_h1, s_old_norm', colormap=:grays, colorrange=(0, 1))

# Water ROI for OLD
cx, cy = nx_h ÷ 2, ny_h ÷ 2
r = 10
x1, x2 = max(1, cx - r), min(nx_h, cx + r)
y1, y2 = max(1, cy - r), min(ny_h, cy + r)
roi_old = hu_old[x1:x2, y1:y2, mid_h]
lines!(ax_h1, [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=:lime, linewidth=1.5)
text!(ax_h1, x2 + 3, y1, text="Water ROI\nσ=$(round(std(roi_old),digits=1)) HU\nμ=$(round(mean(roi_old),digits=1)) HU",
    fontsize=10, color=:lime)

# Panel 2: NEW (spectrum-domain additional_filters)
ax_h2 = Axis(fig_h[1, 2],
    title="NEW: Spectrum-Domain Filtering\n(Beer-Lambert T(E) = exp(-μ(E)×t) per energy)",
    titlecolor=:white, titlesize=14,
    aspect=DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false,
    bottomspinevisible=false, topspinevisible=false,
    leftspinevisible=false, rightspinevisible=false)
s_phys = clamp.(hu_phys[:, :, mid_h], wmin_bone, wmax_bone)
s_phys_norm = (s_phys .- wmin_bone) ./ (wmax_bone - wmin_bone)
heatmap!(ax_h2, s_phys_norm', colormap=:grays, colorrange=(0, 1))

# Water ROI for NEW
roi_phys = hu_phys[x1:x2, y1:y2, mid_h]
lines!(ax_h2, [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=:lime, linewidth=1.5)
text!(ax_h2, x2 + 3, y1, text="Water ROI\nσ=$(round(std(roi_phys),digits=1)) HU\nμ=$(round(mean(roi_phys),digits=1)) HU",
    fontsize=10, color=:lime)

# Panel 3: Difference map (NEW − OLD)
ax_h3 = Axis(fig_h[1, 3],
    title="Difference (NEW − OLD)\nSpectrum vs Projection Domain",
    titlecolor=:white, titlesize=14,
    aspect=DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false,
    bottomspinevisible=false, topspinevisible=false,
    leftspinevisible=false, rightspinevisible=false)
diff_h = hu_phys[:, :, mid_h] .- hu_old[:, :, mid_h]
heatmap!(ax_h3, diff_h', colormap=:RdBu, colorrange=(-200, 200))
Colorbar(fig_h[1, 4], limits=(-200, 200), colormap=:RdBu, label="ΔHU",
    labelcolor=:white, ticklabelcolor=:white)

# Bottom left: HU profiles through center
ax_h_prof = Axis(fig_h[2, 1:2],
    title="Horizontal HU Profile Through Center — Same 7mm Al Filter",
    titlecolor=:white, titlesize=16,
    xlabel="Pixel", ylabel="HU",
    xlabelcolor=:white, ylabelcolor=:white,
    xticklabelcolor=:white, yticklabelcolor=:white,
    backgroundcolor=:gray10)

prof_old = hu_old[nx_h÷2, :, mid_h]
prof_phys = hu_phys[nx_h÷2, :, mid_h]

lines!(ax_h_prof, 1:length(prof_old), prof_old, color=:tomato, linewidth=2,
    label="OLD: Projection-domain flat_filter")
lines!(ax_h_prof, 1:length(prof_phys), prof_phys, color=:dodgerblue, linewidth=2,
    label="NEW: Spectrum-domain additional_filters")
hlines!(ax_h_prof, [0.0], color=:white, linestyle=:dash, linewidth=1, label="Water (0 HU)")
axislegend(ax_h_prof, position=:rt, backgroundcolor=(:gray20, 0.8), labelcolor=:white)

# Bottom right: Spectrum comparison
ax_h_spec = Axis(fig_h[2, 3:4],
    title="How Each Approach Handles 7mm Al Filtering",
    titlecolor=:white, titlesize=16,
    xlabel="Energy (keV)", ylabel="Relative Intensity",
    xlabelcolor=:white, ylabelcolor=:white,
    xticklabelcolor=:white, yticklabelcolor=:white,
    backgroundcolor=:gray10)

# Show the spectrum that the NEW approach actually uses
e_demo, f_demo = BS.load_spectrum_unfiltered(120; anode_angle=10)
_, f_7al, _ = BS.filter_spectrum(e_demo, f_demo; filters=[("Al", 7.0)], sdd_mm=750.0)
f_demo_norm = f_demo ./ maximum(f_demo)
f_7al_norm = f_7al ./ maximum(f_7al)

lines!(ax_h_spec, e_demo, f_demo_norm, color=(:gray50, 0.5), linewidth=1.5,
    label="Unfiltered tube output")
lines!(ax_h_spec, e_demo, f_7al_norm, color=:dodgerblue, linewidth=2.5,
    label="NEW: 7mm Al filtered spectrum (per-energy)")

# Show what OLD approach does: single μ at 60 keV
mu_al_60 = BS.get_filter_mu("Al", 60.0)
trans_60 = exp(-mu_al_60 * 0.7)  # 7mm = 0.7cm
vlines!(ax_h_spec, [60.0], color=:tomato, linewidth=2, linestyle=:dash,
    label="OLD: single μ(60 keV), T=$(round(trans_60, digits=3))")

# Annotate mean energies
me_raw = BS.spectrum_mean_energy(e_demo, f_demo)
me_7al = BS.spectrum_mean_energy(e_demo, f_7al)
vlines!(ax_h_spec, [me_7al], color=:dodgerblue, linewidth=1, linestyle=:dot)
text!(ax_h_spec, me_7al + 1, 0.9, text="mean=$(round(me_7al, digits=0)) keV",
    fontsize=10, color=:dodgerblue)

xlims!(ax_h_spec, 0, 125)
axislegend(ax_h_spec, position=:rt, backgroundcolor=(:gray20, 0.8), labelcolor=:white)

# Add explanation text
Label(fig_h[3, 1:4],
    "OLD flat_filter: adds fixed -log(exp(-μ(60keV)×t)) to sinogram — monochromatic, ignores energy dependence\n" *
    "NEW additional_filters: shapes spectrum before simulation — energy-dependent, physically correct Beer-Lambert",
    fontsize=13, color=:gray70, justification=:center)

save(joinpath(output_dir, "08_flat_vs_additional_filters.png"), fig_h, px_per_unit=2)
display(fig_h)
println("  ✓ Saved: 08_flat_vs_additional_filters.png")

# Print comparison stats
println("\n  --- Flat Filter vs Additional Filters Comparison ---")
println("    OLD (projection-domain): Water ROI mean=$(round(mean(roi_old), digits=1)) HU, σ=$(round(std(roi_old), digits=1)) HU")
println("    NEW (spectrum-domain):   Water ROI mean=$(round(mean(roi_phys), digits=1)) HU, σ=$(round(std(roi_phys), digits=1)) HU")
println("    Difference range: [$(round(minimum(diff_h), digits=0)), $(round(maximum(diff_h), digits=0))] HU")
println("    ⚠ OLD applies monochromatic attenuation at 60 keV to sinogram")
println("    ✓ NEW applies energy-dependent Beer-Lambert to unfiltered spectrum")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary Table
# ═══════════════════════════════════════════════════════════════════════════════
println("\n", "="^72)
println("  SUMMARY")
println("="^72)
println()
println("  📁 All images saved to: $(output_dir)/")
println()
println("  Output files:")
println("    01_raw_vs_filtered_spectrum.png    — Raw vs filtered spectra (120 kVp)")
println("    02_filter_material_comparison.png  — Al/Cu/Sn mean energy & flux curves")
println("    03_kvp_comparison.png              — 80/100/120/140 kVp comparison")
println("    04_anode_angle_comparison.png      — 8° vs 10° anode angle")
println("    05_real_vs_generic_ct.png          — CT images: real vs generic spectrum")
println("    06_filtration_effect_ct.png        — CT images: light vs clinical vs heavy filtration")
println("    07_flux_vs_sdd.png                 — Flux vs source-to-detector distance")
println("    08_flat_vs_additional_filters.png  — ⭐ OLD vs NEW filtering approach comparison")
println()
println("  Key findings:")
println("    • Added filtration shifts mean energy upward (beam hardening)")
println("    • Sn filter has strongest spectral shaping effect per mm")
println("    • Higher kVp → higher mean energy + dramatically more flux")
println("    • Anode 10° slightly different spectral shape than 8°")
println("    • Real spectrum provides physically accurate beam model")
println("    • ⭐ Projection-domain flat_filter is a monochromatic approximation")
println("    • ⭐ Spectrum-domain additional_filters is physically correct (Beer-Lambert)")
println("="^72)
