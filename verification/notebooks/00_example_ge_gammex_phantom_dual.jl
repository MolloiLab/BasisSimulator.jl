### A Pluto.jl notebook ###
# v0.19.0

# ╔═╡ 00010001-0000-4000-8000-000000000004
begin
    using Pkg: Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()
    using Revise
end

# ╔═╡ 14530566-ecaa-4345-9115-e75981f3837d
using Metal # Choose one or the other

# ╔═╡ 00010003-0000-4000-8000-000000000004
using LinearAlgebra

# ╔═╡ 00010004-0000-4000-8000-000000000004
using FFTW

# ╔═╡ 00010005-0000-4000-8000-000000000004
using Random

# ╔═╡ f12cb282-dfc1-4762-9851-7ef4225c8bd4
using Statistics: quantile

# ╔═╡ 35b71532-be75-4fe9-ada8-9d6d4bf21f7f
using Markdown

# ╔═╡ 00010012-0000-4000-8000-000000000004
using Roots

# ╔═╡ 00090041-0000-4000-8000-000000000004
using Unitful

# ╔═╡ 0b629a9c-722c-47c5-900e-5e5b311a743b
# using CUDA # Choose one or the other

# ╔═╡ 00010006-0000-4000-8000-000000000004
import BasisSimulator as BS

# ╔═╡ 00010007-0000-4000-8000-000000000004
import CairoMakie as CM

# ╔═╡ 00010008-0000-4000-8000-000000000004
import Statistics: mean, std, median

# ╔═╡ 00010009-0000-4000-8000-000000000004
import XrayAttenuation as XA

# ╔═╡ 00010013-0000-4000-8000-000000000004
import AcceleratedKernels as AK

# ╔═╡ 00010010-0000-4000-8000-000000000004
const RESULTS_DIR = joinpath(dirname(@__DIR__), "results", "example_ge_dual"); mkpath(RESULTS_DIR)

# ╔═╡ 00010011-0000-4000-8000-000000000004
begin
    """Move array to GPU. Tries CUDA then Metal, falls back to CPU."""
    function to_gpu(x::AbstractArray)
        if @isdefined(CUDA) && CUDA.functional()
            return CUDA.CuArray(x)
        elseif @isdefined(Metal) && Metal.functional()
            return Metal.MtlArray(x)
        else
            @warn "No GPU backend available, using CPU"
            return x
        end
    end

    """Force GPU memory cleanup."""
    function clear_gpu!()
        GC.gc(true)
        if @isdefined(CUDA) && CUDA.functional()
            CUDA.reclaim()
        end
        return nothing
    end
end

# ╔═╡ 00020001-0000-4000-8000-000000000004
md"""
# GE Revolution Apex Elite — Dual-Energy GSI Example

Rapid kVp switching dual-energy acquisition of the Gammex 472 phantom at
80 / 140 kVp, with projection-domain two-material (water + iodine) decomposition
and VMI synthesis at 40 / 70 / 100 / 140 keV. FBP reconstruction (no HIR).

All parameters verified against clinical GE Revolution Apex Elite data
(see `06_ge_apex_elite_clinical.jl` for full validation).
"""

# ╔═╡ 00030001-0000-4000-8000-000000000004
md"""
## 1. Phantom Setup — Gammex 472 (via `create_phantom_from_mask`)

Same Gammex 472 digital phantom as the single-kVp example — 1750×1750×300
voxels @ 0.2 mm, 35 cm FOV, 16 calcium + iodine insert rods.
"""

# ╔═╡ 00030002-0000-4000-8000-000000000004
begin
    GAMMEX_SOLID_WATER = XA.Materials.gammex_472_solidwater

    # Material mapping: label → XA.Material (generic Dict{Int, ...} API)
    PHANTOM_MATERIALS = Dict{Int, XA.Material}(
        0 => XA.Materials.air,
        1 => GAMMEX_SOLID_WATER,
        2 => XA.Materials.water,
        3 => GAMMEX_SOLID_WATER,
        10 => XA.Materials.gammex_472_ca50_0,
        11 => XA.Materials.gammex_472_ca100_0,
        12 => XA.Materials.gammex_472_ca200_0,
        13 => XA.Materials.gammex_472_ca300_0,
        14 => XA.Materials.gammex_472_ca400_0,
        20 => XA.Materials.gammex_472_i2_0,
        21 => XA.Materials.gammex_472_i2_5,
        22 => XA.Materials.gammex_472_i5_0,
        23 => XA.Materials.gammex_472_i7_5,
        24 => XA.Materials.gammex_472_i10_0,
        25 => XA.Materials.gammex_472_i15_0,
        26 => XA.Materials.gammex_472_i20_0,
    )

    MATERIAL_INFO = Dict(
        0 => (name = "Air", color = :gray15),
        1 => (name = "Solid Water", color = :lightskyblue),
        2 => (name = "Pure Water", color = :royalblue),
        3 => (name = "SW Reference", color = :paleturquoise),
        10 => (name = "Ca 50 mg/mL", color = :wheat),
        11 => (name = "Ca 100 mg/mL", color = :sandybrown),
        12 => (name = "Ca 200 mg/mL", color = :orange),
        13 => (name = "Ca 300 mg/mL", color = :darkorange),
        14 => (name = "Ca 400 mg/mL", color = :orangered),
        20 => (name = "I 2.0 mg/mL", color = :honeydew),
        21 => (name = "I 2.5 mg/mL", color = :palegreen),
        22 => (name = "I 5.0 mg/mL", color = :lightgreen),
        23 => (name = "I 7.5 mg/mL", color = :mediumseagreen),
        24 => (name = "I 10.0 mg/mL", color = :seagreen),
        25 => (name = "I 15.0 mg/mL", color = :forestgreen),
        26 => (name = "I 20.0 mg/mL", color = :darkgreen),
    )
end;

# ╔═╡ 00030003-0000-4000-8000-000000000004
"""Build a raw integer mask for the Gammex 472 geometry (no materials — just labels)."""
function build_gammex_472_mask(;
        n_voxels::Int = 1750,
        n_slices::Int = 250,
        fov_cm::Float64 = 35.0,
        z_cm::Float64 = 5.0,
    )
    dx = fov_cm / n_voxels
    dy = fov_cm / n_voxels
    dz = z_cm / n_slices

    x = range(-fov_cm / 2 + dx / 2, fov_cm / 2 - dx / 2, length = n_voxels)
    y = range(-fov_cm / 2 + dy / 2, fov_cm / 2 - dy / 2, length = n_voxels)

    body_radius = 16.5
    rod_radius = 1.4
    air_gap_water = 0.03
    air_gap_other = 0.01
    rod_radius_sq = rod_radius^2
    outer_ring_R = 10.5
    inner_ring_R = 5.5

    outer_start = pi / 2 - pi / 8
    outer_angles = [outer_start - (i - 1) * pi / 4 for i in 1:8]
    outer_labels = [11, 12, 13, 14, 2, 3, 3, 10]

    inner_start = pi / 2
    inner_angles = [inner_start - (i - 1) * pi / 4 for i in 1:8]
    inner_labels = [21, 22, 23, 24, 25, 26, 2, 20]

    outer_hole_r_sq = [(rod_radius + (outer_labels[i] == 2 ? air_gap_water : air_gap_other))^2 for i in 1:8]
    inner_hole_r_sq = [(rod_radius + (inner_labels[i] == 2 ? air_gap_water : air_gap_other))^2 for i in 1:8]

    outer_cx = [outer_ring_R * cos(a) for a in outer_angles]
    outer_cy = [outer_ring_R * sin(a) for a in outer_angles]
    inner_cx = [inner_ring_R * cos(a) for a in inner_angles]
    inner_cy = [inner_ring_R * sin(a) for a in inner_angles]

    slice = zeros(Int, n_voxels, n_voxels)

    for j in 1:n_voxels, i in 1:n_voxels
        xi, yj = x[i], y[j]

        if xi^2 + yj^2 <= body_radius^2
            slice[i, j] = 1

            for idx in 1:8
                d_sq = (xi - outer_cx[idx])^2 + (yj - outer_cy[idx])^2
                if d_sq <= outer_hole_r_sq[idx]
                    slice[i, j] = d_sq <= rod_radius_sq ? outer_labels[idx] : 0
                    @goto next_voxel
                end
            end

            for idx in 1:8
                d_sq = (xi - inner_cx[idx])^2 + (yj - inner_cy[idx])^2
                if d_sq <= inner_hole_r_sq[idx]
                    slice[i, j] = d_sq <= rod_radius_sq ? inner_labels[idx] : 0
                    break
                end
            end
        end
        @label next_voxel
    end

    slice = reverse(permutedims(rot180(slice)), dims = 2)

    mask = Array{Int, 3}(undef, n_voxels, n_voxels, n_slices)
    @views for k in 1:n_slices
        mask[:, :, k] .= slice
    end

    return mask, (dx, dy, dz), (fov_cm, fov_cm, z_cm)
end

# ╔═╡ 00030004-0000-4000-8000-000000000004
# Phantom: 1750 × 1750 × 300, 0.2 mm isotropic, 35 cm FOV, 6 cm z-extent
begin
    raw_mask, voxel_size, _ = build_gammex_472_mask(
        n_voxels = 1750,
        n_slices = 300,
        fov_cm = 35.0,
        z_cm = 6.0,
    )

    sim_phantom = BS.create_phantom_from_mask(raw_mask, PHANTOM_MATERIALS, voxel_size)

    sim_phantom_gpu = BS.Phantom(
        to_gpu(sim_phantom.mask),
        sim_phantom.materials,
        sim_phantom.voxel_size,
        sim_phantom.origin,
        sim_phantom.extent,
    )
    @info "Phantom: $(size(sim_phantom.mask)) voxels, eltype=$(eltype(sim_phantom.mask)), extent=$(sim_phantom.extent) cm"
end;

# ╔═╡ 00030005-0000-4000-8000-000000000004
# Phantom mid-slice visualization
let
    mid = size(sim_phantom.mask, 3) ÷ 2
    nz = size(sim_phantom.mask, 3)
    slice_data = sim_phantom.mask[:, :, mid]

    unique_labels = sort(unique(slice_data))
    n_labels = length(unique_labels)

    lut = zeros(Float32, 27)
    for (i, l) in enumerate(unique_labels)
        lut[Int(l) + 1] = Float32(i)
    end
    mapped = lut[Int.(slice_data) .+ 1]

    colors = [MATERIAL_INFO[Int(l)].color for l in unique_labels]
    cmap = CM.cgrad(colors, n_labels, categorical = true)
    names = [MATERIAL_INFO[Int(l)].name for l in unique_labels]

    fig = CM.Figure(size = (1000, 850), fontsize = 12)
    ax = CM.Axis(
        fig[1, 1];
        title = "Gammex 472 Phantom — Slice $mid / $nz (0.2 mm voxels)",
        aspect = CM.DataAspect()
    )
    hm = CM.heatmap!(ax, mapped; colormap = cmap, colorrange = (0.5, n_labels + 0.5))
    CM.Colorbar(fig[1, 2], hm; ticks = (1:n_labels, names), ticklabelsize = 11, width = 15)
    CM.save(joinpath(RESULTS_DIR, "phantom.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00040001-0000-4000-8000-000000000004
md"""
## 2. Scanner & X-ray Spectra (80 / 140 kVp)
"""

# ╔═╡ 00040002-0000-4000-8000-000000000004
# Additional filtration: 4.5 mm Al (empirically matched to clinical HVL)
additional_filters = [("Al", 4.5)]

# ╔═╡ 00040003-0000-4000-8000-000000000004
# GE Revolution Apex Elite geometry (verified against clinical data)
begin
    sim_electronic_noise = 0       # e⁻ (electronic noise disabled)
    sim_detection_gain = 10.0      # e⁻/keV

    sim_scanner = BS.Scanner(
        source_to_isocenter = 625.6,
        source_to_detector = 1100.0,
        detector_rows = 256,
        detector_cols = 834,
        detector_row_size = 0.625,
        detector_col_size = 0.6,
        detector_shape = BS.CURVED_DETECTOR,
        focal_spot_width = 1.0,
        focal_spot_length = 1.0,
        target_angle = 10.0,
        flat_filter_material = :aluminum,
        flat_filter_thickness = 2.5,
        bowtie_filter = :ge_revolution_large,
        detector_material = :lumex,
        detector_depth = 3.0,
        fill_factor_row = 0.9,
        fill_factor_col = 0.9,
        electronic_noise = sim_electronic_noise,
        detection_gain = sim_detection_gain,
    )
end

# ╔═╡ 00040004-0000-4000-8000-000000000004
sim_opts = BS.SimOptions(fidelity = :eict, seed = 1234)

# ╔═╡ 00040005-0000-4000-8000-000000000004
# X-ray spectra for both 80 and 140 kVp
let
    fig = CM.Figure(size = (1200, 500), fontsize = 13)

    for (col, kVp) in enumerate([80, 140])
        e_raw, w_raw = BS.load_spectrum_unfiltered(kVp; anode_angle = 10)
        all_filters = vcat([("Al", 2.5)], additional_filters)
        _, w_filt = BS.filter_spectrum(e_raw, w_raw; filters = all_filters, sdd_mm = 1100.0)
        prot = BS.CTProtocol(kVp = kVp, additional_filters = additional_filters)
        e_res, w_res = BS.resolve_source_spectrum_without_bowtie(sim_opts, prot; scanner = sim_scanner)
        E_eff = sum(e_res .* w_res) / sum(w_res)

        ax = CM.Axis(
            fig[1, col];
            title = "$(kVp) kVp Spectrum — GE Revolution Apex Elite",
            xlabel = "Energy (keV)", ylabel = "Relative Fluence (a.u.)"
        )

        CM.lines!(
            ax, e_raw, w_raw ./ maximum(w_raw);
            color = :gray60, linewidth = 1.5, label = "Raw tube"
        )
        CM.lines!(
            ax, e_raw, w_filt ./ maximum(w_filt);
            color = :steelblue, linewidth = 2, label = "After 7.0 mm Al"
        )
        CM.lines!(
            ax, e_res, w_res ./ maximum(w_res);
            color = :darkorange, linewidth = 2, label = "Resolved (+ bowtie + detector)"
        )
        CM.vlines!(
            ax, [E_eff]; color = :red, linestyle = :dash, linewidth = 1,
            label = "E_eff = $(round(E_eff, digits = 1)) keV"
        )

        CM.xlims!(ax, 10, 145)
        CM.ylims!(ax, 0, 1.05)
        CM.axislegend(ax; position = :rt, labelsize = 9)
    end

    CM.save(joinpath(RESULTS_DIR, "spectrum_80_140kVp.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00050001-0000-4000-8000-000000000004
md"""
## 3. Acquisition Protocol — Dual Energy

Rapid kVp switching at 0.5 s gantry rotation. Duty-cycle-weighted effective
mA values derived from clinical GE GSI DICOM tags: 80 kVp carries ~65% of
the rotation, 140 kVp ~35%.
"""

# ╔═╡ 00050002-0000-4000-8000-000000000004
begin
    de_rotation_time = 0.5           # seconds — clinical GSI gantry rotation
    de_collimation_mm = 10.0         # dev mode (clinical GSI = 40 mm)
    de_n_views = 984                 # views per kVp per rotation

    de_mA_80 = 407 * 0.65            # 80 kVp duty-cycle-weighted effective mA
    de_mA_140 = 405 * 0.35           # 140 kVp duty-cycle-weighted effective mA
end

# ╔═╡ 00050003-0000-4000-8000-000000000004
begin
    sim_recon_xy = 512
    sim_recon_fov_cm = 35.0
    sim_slice_thickness_mm = 0.625
    sim_recon_z_cm = de_collimation_mm / 10.0
    sim_n_recon_slices = round(Int, de_collimation_mm / sim_slice_thickness_mm)
    sim_matrix_size = (sim_recon_xy, sim_recon_xy, sim_n_recon_slices)

    sim_recon_geom = BS.ReconOptions(
        algorithm = :fdk,
        matrix_size = sim_matrix_size,
        fov_cm = sim_recon_fov_cm,
        z_cm = sim_recon_z_cm,
    )
    @info "Recon: $(sim_matrix_size), FOV=$(sim_recon_fov_cm) cm, z=$(sim_recon_z_cm) cm"
end

# ╔═╡ 00050004-0000-4000-8000-000000000004
protocol_low = BS.CTProtocol(
    kVp = 80, mA = de_mA_80,
    views = de_n_views,
    rotation_time = de_rotation_time,
    collimation_mm = de_collimation_mm,
    additional_filters = additional_filters,
)

# ╔═╡ 00050005-0000-4000-8000-000000000004
protocol_high = BS.CTProtocol(
    kVp = 140, mA = de_mA_140,
    views = de_n_views,
    rotation_time = de_rotation_time,
    collimation_mm = de_collimation_mm,
    additional_filters = additional_filters,
)

# ╔═╡ 00070001-0000-4000-8000-000000000004
md"""
## 4. Forward Projection — Both kVps

Two independent polychromatic forward projections through the digital phantom.
"""

# ╔═╡ 00070002-0000-4000-8000-000000000004
# 80 kVp sinogram
sim_sino_low = let
    @info "Simulating 80 kVp / $(round(de_mA_80, digits = 1)) mA / $(de_collimation_mm) mm collimation..."
    ws = BS.create_eict_workspace(
        sim_scanner, protocol_low, sim_opts, sim_recon_geom, sim_phantom_gpu
    )
    BS.simulate!(ws, sim_phantom_gpu, sim_scanner, protocol_low, sim_opts, sim_recon_geom)
    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom, kvp = 80)
    ws = nothing; clear_gpu!()
    result
end;

# ╔═╡ 00070003-0000-4000-8000-000000000004
# 140 kVp sinogram
sim_sino_high = let
    @info "Simulating 140 kVp / $(round(de_mA_140, digits = 1)) mA / $(de_collimation_mm) mm collimation..."
    ws = BS.create_eict_workspace(
        sim_scanner, protocol_high, sim_opts, sim_recon_geom, sim_phantom_gpu
    )
    BS.simulate!(ws, sim_phantom_gpu, sim_scanner, protocol_high, sim_opts, sim_recon_geom)
    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom, kvp = 140)
    ws = nothing; clear_gpu!()
    result
end;

# ╔═╡ 00070004-0000-4000-8000-000000000004
# Sinogram mid-view visualization (both kVps)
let
    mid_view = size(sim_sino_low.sino, 2) ÷ 2
    mid_row = size(sim_sino_low.sino, 3) ÷ 2
    fig = CM.Figure(size = (1200, 800), fontsize = 13)

    for (row, sino_data, kVp) in [(1, sim_sino_low, 80), (2, sim_sino_high, 140)]
        ax1 = CM.Axis(
            fig[row, 1]; title = "$(kVp) kVp — Mid-view (view $mid_view)",
            xlabel = "Detector column", ylabel = "Detector row"
        )
        CM.heatmap!(ax1, sino_data.sino[:, mid_view, :]'; colormap = :grays)

        ax2 = CM.Axis(
            fig[row, 2]; title = "$(kVp) kVp — Mid-row (row $mid_row)",
            xlabel = "Detector column", ylabel = "View angle"
        )
        CM.heatmap!(ax2, sino_data.sino[:, :, mid_row]'; colormap = :grays)
    end

    CM.save(joinpath(RESULTS_DIR, "sinogram.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080001-0000-4000-8000-000000000004
md"""
## 5. Mechlem2018 — Joint One-Step Recon

One-step joint reconstruction + material decomposition, strict port of
**Mechlem et al. 2018** (*IEEE TMI* 37:68–80) as evaluated by Mory et al. 2018
(*Phys. Med. Biol.* 63:235001), Algorithm 4 — the fastest of the 5 one-step
methods compared (1–3 orders of magnitude fewer iterations than Cai2013,
Long2014, Weidinger2016, or Barber2016).

**Core algorithm** (Mory2018 Alg. 4, Appendix A.4):
1. **SQS** (Separable Quadratic Surrogate) of the Poisson log-likelihood.
2. **Huber regularization** on the 2D spatial gradient, per material.
3. **Ordered subsets** (≤6, Mory §3.1: diverges above).
4. **Nesterov momentum** on the SQS step.

**Adaptation for our problem** (2-kVp rapid-switching CBCT vs paper's 5-bin PCCT):
- Drop the bin dimension (each kVp is a single "channel" — no detector bin response).
- Two separate projection geometries (`geom_low`, `geom_high`), views don't align
  due to duty-cycle split — gradient and Hessian accumulate contributions from
  both.
- 2 materials (water + iodine) instead of 3 → per-voxel **2×2 analytic Newton**
  (no generalized linear solve).
- Skip μ-preconditioning (T = I₂). 80/140 kVp spectra are 30–140 keV wide, so
  the attenuation matrix M is well-conditioned. Can add Fessler's method later.
- All kernels via `AK.foreachindex` for native CPU / Metal / CUDA dispatch.

Built in-notebook test-first. Each primitive has a parity check against an
explicit reference loop before being used downstream.
"""

# ╔═╡ 00090040-0000-4000-8000-000000000004
md"""
### 5.0 Cong 2022 warm start  →  `ρ_water(r)` / `ρ_iodine(r)`

Pure-zero init makes the first few Mechlem iterations travel through an
extreme-transient region where Newton steps overshoot and the non-negativity
projection clips into the phantom body. A physically-reasonable warm start
collapses that transient.

**Pipeline (1:1 from `06_ge_apex_elite_clinical.jl` §5 Cong):**
1. Build the *empirical K-edge basis* `(p̂(E), q̂(E))` via 2×2 exact fit to
   `{water, iodine}` mass-attenuation across both 80/140 kVp grids — captures
   iodine's 33 keV K-edge faithfully, unlike Cong's analytic `p ∝ √(32/ε⁷)`.
2. Run `BS.apply_cong(sino_L, sino_H; basis, water_basis)` per ray → projection
   pair `(sino_photo, sino_compton)` = `(∫a(r)dr, ∫c(r)dr)`.
3. FBP each basis sinogram separately → volumes `(vol_a, vol_c)`.
4. **Analytic 2×2 conversion** `(a, c) → (ρ_water, ρ_iodine)`:

       [a(r)]   [C_aw  C_ai] [ρ_water(r) ]
       [c(r)] = [C_cw  C_ci] [ρ_iodine(r)]

   where `C_*w = (a_water, c_water) = BS.water_basis_constants()` and
   `C_*i = (Z⁴/A, Z/A)` for elemental iodine at unit density. Invert once
   on the host (2×2, analytic) → per-voxel matmul kernel → `mech_z0_w, mech_z0_i`
   volumes in g/mL, ready to hand to the Mechlem orchestrator as `z0`.
"""

# ╔═╡ 00090042-0000-4000-8000-000000000004
# Empirical K-edge-aware basis (1:1 port of 06_ge_apex_elite_clinical.jl:3641).
# Uses 2-material {water, iodine} calibration so both water AND iodine µ(E) are
# bit-exact against NIST at every E — the iodine K-edge at 33 keV is captured
# in p̂(E) instead of smoothed through by the analytic p ∝ √(32/ε⁷).
mech_de_basis = let
    prot_L = BS.CTProtocol(kVp = 80,  additional_filters = additional_filters)
    prot_H = BS.CTProtocol(kVp = 140, additional_filters = additional_filters)
    e_L, w_L = BS.resolve_source_spectrum_without_bowtie(sim_opts, prot_L; scanner = sim_scanner)
    e_H, w_H = BS.resolve_source_spectrum_without_bowtie(sim_opts, prot_H; scanner = sim_scanner)

    ŵ_L = Float32.(Float64.(w_L) ./ sum(Float64.(w_L)))
    ŵ_H = Float32.(Float64.(w_H) ./ sum(Float64.(w_H)))

    atomic_mass = Dict{Int, Float64}(
        let el = getproperty(XA.Elements, n)
            el.Z => el.Z / el.ZA_ratio
        end
        for n in propertynames(XA.Elements)
    )

    function physical_ac(mat)
        ρ = Unitful.ustrip(Unitful.u"g/cm^3", mat.density)
        if hasproperty(mat, :composition)
            a = 0.0; c = 0.0
            for (Z, m_frac) in mat.composition
                A = atomic_mass[Z]
                a += m_frac * Z^4 / A
                c += m_frac * Z / A
            end
            (ρ * a, ρ * c)
        else
            Z = mat.Z;  A = Z / mat.ZA_ratio
            (ρ * Z^4 / A, ρ * Z / A)
        end
    end

    ref_materials = [XA.Materials.water, XA.Elements.Iodine]
    A_coef = Matrix{Float64}(undef, length(ref_materials), 2)
    for (i, m) in enumerate(ref_materials)
        A_coef[i, 1], A_coef[i, 2] = physical_ac(m)
    end
    function fit_pq(energies)
        p = zeros(Float32, length(energies))
        q = zeros(Float32, length(energies))
        for (i, E) in enumerate(energies)
            µ = Float64[BS.compute_μ_at_energy(m, Float64(E)) for m in ref_materials]
            sol = A_coef \ µ
            p[i] = Float32(sol[1]);  q[i] = Float32(sol[2])
        end
        p, q
    end

    p_L, q_L = fit_pq(e_L)
    p_H, q_H = fit_pq(e_H)

    @info "[warm-start empirical basis] fit to {water, I} (EXACT) at $(length(e_L)) L-bins + $(length(e_H)) H-bins"

    (ŵ_L = ŵ_L, p_L = p_L, q_L = q_L,
     ŵ_H = ŵ_H, p_H = p_H, q_H = q_H)
end

# ╔═╡ 00090043-0000-4000-8000-000000000004
# Physical water basis constants (Cong Eqs 3a–3b), ρ=1 baseline:
#   a_w = Σ m_i · Z_i⁴/A_i,   c_w = Σ m_i · Z_i/A_i
mech_water_basis = BS.water_basis_constants()

# ╔═╡ 00090044-0000-4000-8000-000000000004
# Per-ray Cong 2022 decomposition: Brent(water-L) → Newton(quintic h(y)) → Brent(y).
# Strict 1:1 port of clinical notebook's sim_de_decomp (line 3729).
mech_cong_sino = let
    sino_L_gpu = to_gpu(Float32.(sim_sino_low.sino))
    sino_H_gpu = to_gpu(Float32.(sim_sino_high.sino))

    t1 = time()
    sino_y_gpu, sino_c_gpu = BS.apply_cong(
        sino_L_gpu, sino_H_gpu;
        basis           = mech_de_basis,
        water_basis     = mech_water_basis,
        newton_max_iter = 5,
        newton_tol      = eps(Float32),
        y_max_factor    = 0.2,
        y_max_cap       = 1f7,
    )
    dt = time() - t1

    sino_photo   = Array(sino_y_gpu)
    sino_compton = Array(sino_c_gpu)
    sino_L_gpu = nothing; sino_H_gpu = nothing
    sino_y_gpu = nothing; sino_c_gpu = nothing
    clear_gpu!()

    @info "[CDW22] apply_cong done in $(round(dt, digits=1)) s"
    @info "  ⟨y⟩ = $(round(mean(sino_photo),   sigdigits=4))   photoelectric line integral"
    @info "  ⟨C⟩ = $(round(mean(sino_compton), sigdigits=4))   Compton line integral"

    (sino_photo = sino_photo, sino_compton = sino_compton, geom = sim_sino_low.geom)
end;

# ╔═╡ 00090045-0000-4000-8000-000000000004
# FBP each Cong basis sinogram into image space — no regularization.
# Standard Ram-Lak with soft rolloff (BS.CustomFilter), same shape as clinical.
mech_cong_recon = let
    filter_ctrl = (
        x = (0.0, 0.25, 0.5, 0.75, 1.0),
        y = (1.0, 0.95, 0.85, 0.65, 0.4),
    )

    function _fbp(sino::Array{Float32, 3})
        sino_gpu = to_gpu(sino)
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, mech_cong_sino.geom, sim_matrix_size;
            filter = BS.CustomFilter(filter_ctrl.x, filter_ctrl.y),
        )
        vol = Array(BS.reconstruct!(ws, sino_gpu, mech_cong_sino.geom, sim_matrix_size))
        ws = nothing;  sino_gpu = nothing;  clear_gpu!()
        Float32.(vol)
    end

    t1 = time()
    vol_a = _fbp(mech_cong_sino.sino_photo)
    vol_c = _fbp(mech_cong_sino.sino_compton)
    @info "[warm-start FBP] a(r), c(r) volumes done in $(round(time() - t1, digits=1)) s"

    (vol_a = vol_a, vol_c = vol_c)
end;

# ╔═╡ 00090046-0000-4000-8000-000000000004
# Convert (a, c) volumes → (ρ_water, ρ_iodine) volumes via the analytic
# 2×2 inverse of the basis transformation matrix:
#     C = [a_water  a_iodine_Z⁴/A;   c_water  c_iodine_Z/A]
# (all at unit density).  Per-voxel matmul kernel runs on the sinogram backend.
mech_z0 = let
    # Build C on the host (2×2, Float64 for conditioning, then cast to Float32).
    aw = Float64(mech_water_basis.a)
    cw = Float64(mech_water_basis.c)
    I_Z  = Float64(XA.Elements.Iodine.Z)
    I_A  = I_Z / Float64(XA.Elements.Iodine.ZA_ratio)
    ai = I_Z^4 / I_A
    ci = I_Z   / I_A

    C    = [aw ai; cw ci]
    C_inv = inv(C)
    Ci11, Ci12 = Float32(C_inv[1, 1]), Float32(C_inv[1, 2])
    Ci21, Ci22 = Float32(C_inv[2, 1]), Float32(C_inv[2, 2])

    @info "[warm-start conv]  C = [$(round(aw, sigdigits=4))  $(round(ai, sigdigits=4)); $(round(cw, sigdigits=4))  $(round(ci, sigdigits=4))]"
    @info "[warm-start conv]  C⁻¹ = [$(round(Ci11, sigdigits=4))  $(round(Ci12, sigdigits=4)); $(round(Ci21, sigdigits=4))  $(round(Ci22, sigdigits=4))]"

    z0_w = similar(mech_cong_recon.vol_a)
    z0_i = similar(mech_cong_recon.vol_a)
    vol_a = mech_cong_recon.vol_a
    vol_c = mech_cong_recon.vol_c
    @inbounds Threads.@threads for idx in eachindex(z0_w)
        a = vol_a[idx];  c = vol_c[idx]
        # Clamp negatives from FBP ringing to 0 before we send into Mechlem
        zw = Ci11 * a + Ci12 * c
        zi = Ci21 * a + Ci22 * c
        z0_w[idx] = max(zw, 0f0)
        z0_i[idx] = max(zi, 0f0)
    end

    @info "[warm-start]  ρ_water  ∈ [$(round(minimum(z0_w); digits=3)), $(round(maximum(z0_w); digits=3))] g/mL"
    @info "[warm-start]  ρ_iodine ∈ [$(round(minimum(z0_i); digits=5)), $(round(maximum(z0_i); digits=5))] g/mL"
    (ρ_water = z0_w, ρ_iodine = z0_i)
end;

# ╔═╡ 00090047-0000-4000-8000-000000000004
# Visualize the warm-start volumes — what Mechlem starts from instead of zeros.
let
    mid = size(mech_z0.ρ_water, 3) ÷ 2
    ρw  = mech_z0.ρ_water[:, :, mid]
    ρi  = mech_z0.ρ_iodine[:, :, mid]

    fig = CM.Figure(size = (1250, 560), fontsize = 13)
    ax1 = CM.Axis(fig[1, 1]; title = "ρ_water (g/mL) — Cong warm start  (slice $mid)",
                  aspect = CM.DataAspect())
    hm1 = CM.heatmap!(ax1, ρw; colormap = :viridis)
    CM.Colorbar(fig[1, 2], hm1)

    ax2 = CM.Axis(fig[1, 3]; title = "ρ_iodine (g/mL) — Cong warm start  (slice $mid)",
                  aspect = CM.DataAspect())
    hm2 = CM.heatmap!(ax2, ρi; colormap = :viridis)
    CM.Colorbar(fig[1, 4], hm2)

    CM.save(joinpath(RESULTS_DIR, "mechlem_warm_start.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00090001-0000-4000-8000-000000000004
md"""
### 5.1 Hyperparameters

Every tunable knob for the Mechlem2018 recon, grouped by role. Defaults are
the paper values (Mory 2018, Table 2, Mechlem2018 row) translated from the
3-material PCCT problem to our 2-material dual-kVp problem. Change anything
here and the whole pipeline downstream picks it up.
"""

# ╔═╡ 00090002-0000-4000-8000-000000000004
begin
    # ─── Algorithm iteration ──────────────────────────────────────────
    mech_n_iter      = 3           # outer iterations.    Mory Fig 6 converges by ~10–20.
    mech_n_subsets   = 2            # ordered subsets.     HARD CAP 6 — Mory §3.1: diverges above.

    # ─── Materials (fixed: water + iodine for dual-kVp CCTA) ──────────
    mech_materials   = (:water, :iodine)    # index 1 = water, index 2 = iodine. Everything
                                            # below is ordered [water, iodine].

    # ─── Huber regularization per material ────────────────────────────
    # Huber φ(t, δ) approximates ‖·‖₁ on the spatial gradient.
    #   δ = "soft-L1 → L2" transition width in material-concentration units [g/mL].
    #   λ = regularization weight per material; larger ⇒ smoother.
    # iodine gets 10⁴× the weight and 100× tighter δ — noise streaks live in the
    # iodine basis image, water basis is already smooth at this dose.
    mech_huber_δ     = Float32[0.1, 0.001]     # [water g/mL, iodine g/mL]   — Mory Table 2
    mech_huber_λ     = Float32[10.0, 10000]    # [water,       iodine]       — Mory Table 2

    # ─── Spectrum sampling (shared energy grid for both kVps) ─────────
    mech_n_energies  = 61           # energy grid points for M matrix + S interpolation
    mech_E_min_keV   = 20.0         # keV — below this, filtered spectrum is negligible
    mech_E_max_keV   = 140.0        # keV — top tube voltage

    # ─── Numerical stability ──────────────────────────────────────────
    mech_ybar_floor  = 1f-6         # floor on ȳ to prevent 1/ȳ blow-up in gradient
    mech_det_floor   = 1f-12        # floor on det(H) for per-voxel 2×2 solve

    # ─── Initialization ───────────────────────────────────────────────
    mech_init_z0     = :zero        # :zero (Mechlem default) | :fbp_warm_start (future)

    # ─── Positivity projection (z, α, v ← max(·, 0) after Nesterov) ───
    # Mory 2018 MATLAB: NO positivity (z can go negative).
    # Mechlem 2018 IEEE paper: projects z ≥ 0 every step.
    # `true` (default) matches the Mechlem paper and prevents exp(+M·|neg_proj|)
    # blow-up when iodine goes negative on noise.  `false` for strict Mory-MATLAB
    # parity.
    mech_nonneg      = true

    # ─── Fessler μ-preconditioning mode ───────────────────────────────
    # Mechlem's 2×2 per-voxel data Hessian scales as M². With unprecon M:
    # (M_i/M_w)² ≈ (30/0.2)² ≈ 22,500 condition-number gap → water runs away,
    # iodine crawls. Fessler's preconditioning builds a 2×2 T such that the
    # synth-basis spectral Hessian M_synth^T · diag(S_L + S_H) · M_synth = I
    # (Kim-Ramani-Fessler 2015; Long-Fessler 2014).
    #
    # Modes:
    #   :fessler — Fessler's sym-sqrt preconditioner, T = F^(-1/2),
    #              F = M^T · diag(S_L + S_H) · M   (recommended for dual-kVp)
    #   :none    — T = I₂, strict Mory 2018 MATLAB behavior (unconditioned;
    #              diverges on water+iodine without SBM rescaling)
    mech_precond     = :fessler

    # ─── Diagnostics ──────────────────────────────────────────────────
    mech_verbose         = true
    mech_store_iterates  = false    # keep every iteration (memory-heavy, for convergence plots)
end;

# ╔═╡ 00090003-0000-4000-8000-000000000004
md"""
### 5.2 Mass-attenuation matrix M + per-kVp spectra S

**M [Nₑ × 2]**: mass-attenuation coefficients [cm²/g] for water + iodine at
each sampled energy. Shared across both kVps — M is a physical constant table,
the kVp only changes which energies get weighted.

**S_L, S_H [Nₑ]**: per-energy incident spectrum × flat filters × bowtie (pixel-
averaged) × detector response, interpolated onto the shared `mech_E_grid`.
Unscaled here — the actual I₀-per-pixel gets multiplied in when we hook up the
Gammex sim counts (§5.4+). Suitable as-is for the single-ray parity test below.
"""

# ╔═╡ 00090004-0000-4000-8000-000000000004
begin
    mech_E_grid = Float32.(range(mech_E_min_keV, mech_E_max_keV, length = mech_n_energies))

    # M[e, m]: mass-attenuation [cm²/g] of material m at energy mech_E_grid[e]
    mech_M = let
        M = zeros(Float32, mech_n_energies, 2)
        for (i, E) in enumerate(mech_E_grid)
            M[i, 1] = Float32(BS.compute_mass_μ_at_energy(XA.Materials.water, Float64(E)))
            M[i, 2] = Float32(BS.compute_mass_μ_at_energy(XA.Elements.Iodine, Float64(E)))
        end
        M
    end

    @info "M [$(mech_n_energies)×2]  water: $(round(mech_M[1, 1], digits=3))–$(round(mech_M[end, 1], digits=3)) cm²/g,  iodine: $(round(mech_M[1, 2], digits=2))–$(round(mech_M[end, 2], digits=2)) cm²/g"
end;

# ╔═╡ 00090005-0000-4000-8000-000000000004
begin
    # Normalized spectra (sum to 1) — I₀-per-pixel factor applied later when
    # we wire up real Gammex counts. Used here only for the forward-model
    # parity test, where the scale is arbitrary.
    mech_S_L, mech_S_H = let
        eL, wL = BS.resolve_source_spectrum_without_bowtie(sim_opts, protocol_low;  scanner = sim_scanner)
        eH, wH = BS.resolve_source_spectrum_without_bowtie(sim_opts, protocol_high; scanner = sim_scanner)

        interp = function (e_src, w_src, e_target)
            w_out = zeros(Float32, length(e_target))
            for (i, E) in enumerate(e_target)
                E < first(e_src) && continue
                E > last(e_src)  && continue
                j = searchsortedfirst(e_src, E) - 1
                j = clamp(j, 1, length(e_src) - 1)
                t = (Float64(E) - Float64(e_src[j])) / (Float64(e_src[j + 1]) - Float64(e_src[j]))
                w_out[i] = Float32((1 - t) * Float64(w_src[j]) + t * Float64(w_src[j + 1]))
            end
            w_out
        end

        sL = interp(eL, wL, mech_E_grid);  sL ./= max(sum(sL), eps(Float32))
        sH = interp(eH, wH, mech_E_grid);  sH ./= max(sum(sH), eps(Float32))
        sL, sH
    end

    @info "S_L (normalized) @ 80 kVp:  E_eff = $(round(sum(mech_E_grid .* mech_S_L),  digits=1)) keV"
    @info "S_H (normalized) @ 140 kVp: E_eff = $(round(sum(mech_E_grid .* mech_S_H),  digits=1)) keV"
end;

# ╔═╡ 00090006-0000-4000-8000-000000000004
md"""
### 5.3 Forward model kernel + parity test

Port of `WeidingerForwardModel.m` (Mory 2018, Utilities/) to Julia + AK. For
each ray pixel `p`, compute the polychromatic Beer–Lambert factor and
accumulate three quantities needed by the SQS data term:

- `ȳ[p]       = Σ_e S[e] · exp(-Σ_m M[e,m]·proj_m[p])`  (expected counts)
- `fmu_m[p]   = -Σ_e S[e] · exp(...) · M[e,m]`           (∂ȳ/∂proj_m)
- `fmm_mn[p]  = +Σ_e S[e] · exp(...) · M[e,m]·M[e,n]`    (∂²ȳ/∂proj_m ∂proj_n)

Energy loop stays inside the kernel (one `exp` per (pixel, energy), no
intermediate allocations). For 2 materials fmm is symmetric → store 3 unique
entries (ww, wi, ii). Runs on whatever backend the arrays live on via
`AK.foreachindex`.

Parity test checks the kernel against an explicit Float64 reference loop at
a single ray — the whole `mechlem_reconstruct` orchestrator builds on this,
so any bug here would compound.
"""

# ╔═╡ 00090007-0000-4000-8000-000000000004
function mechlem_poly_forward!(
        ȳ::AbstractArray{Float32, 3},
        fmu_w::AbstractArray{Float32, 3},
        fmu_i::AbstractArray{Float32, 3},
        fmm_ww::AbstractArray{Float32, 3},
        fmm_wi::AbstractArray{Float32, 3},
        fmm_ii::AbstractArray{Float32, 3},
        proj_w::AbstractArray{Float32, 3},
        proj_i::AbstractArray{Float32, 3},
        S::AbstractVector{Float32},
        M::AbstractMatrix{Float32},
    )
    Ne = length(S)
    AK.foreachindex(ȳ) do p
        pw  = proj_w[p]
        pii = proj_i[p]
        s_sum = 0f0
        s_mw  = 0f0;  s_mi  = 0f0
        s_mww = 0f0;  s_mwi = 0f0;  s_mii = 0f0
        @inbounds for e in 1:Ne
            mw = M[e, 1]
            mi = M[e, 2]
            att = exp(-(mw * pw + mi * pii))
            w = S[e] * att
            s_sum += w
            s_mw  += w * mw
            s_mi  += w * mi
            s_mww += w * mw * mw
            s_mwi += w * mw * mi
            s_mii += w * mi * mi
        end
        ȳ[p]      = s_sum
        fmu_w[p]  = -s_mw
        fmu_i[p]  = -s_mi
        fmm_ww[p] = s_mww
        fmm_wi[p] = s_mwi
        fmm_ii[p] = s_mii
    end
    return nothing
end

# ╔═╡ 00090008-0000-4000-8000-000000000004
# Single-ray parity test — kernel vs Float64 reference loop.
let
    p_w = 2.0f0       # g/cm² water line integral (about 2 cm of water)
    p_i = 0.01f0      # g/cm² iodine line integral (about 1 cm of 10 mg/mL)
    ȳ = zeros(Float32, 1, 1, 1)
    fmu_w  = similar(ȳ);  fmu_i  = similar(ȳ)
    fmm_ww = similar(ȳ);  fmm_wi = similar(ȳ);  fmm_ii = similar(ȳ)
    proj_w = fill(p_w, 1, 1, 1)
    proj_i = fill(p_i, 1, 1, 1)

    mechlem_poly_forward!(ȳ, fmu_w, fmu_i, fmm_ww, fmm_wi, fmm_ii,
                           proj_w, proj_i, mech_S_L, mech_M)

    ȳ_ref = 0.0
    fmu_w_ref = 0.0;  fmu_i_ref = 0.0
    fmm_ww_ref = 0.0; fmm_wi_ref = 0.0; fmm_ii_ref = 0.0
    for e in 1:length(mech_S_L)
        mw = Float64(mech_M[e, 1])
        mi = Float64(mech_M[e, 2])
        att = exp(-(mw * Float64(p_w) + mi * Float64(p_i)))
        w = Float64(mech_S_L[e]) * att
        ȳ_ref      += w
        fmu_w_ref  += w * mw
        fmu_i_ref  += w * mi
        fmm_ww_ref += w * mw * mw
        fmm_wi_ref += w * mw * mi
        fmm_ii_ref += w * mi * mi
    end

    rtol = 5f-5
    rel(a, b) = abs(Float32(a) - Float32(b)) / max(abs(Float32(b)), eps(Float32))
    checks = (
        ȳ      = rel(ȳ[1],      ȳ_ref),
        fmu_w  = rel(fmu_w[1],  -fmu_w_ref),
        fmu_i  = rel(fmu_i[1],  -fmu_i_ref),
        fmm_ww = rel(fmm_ww[1],  fmm_ww_ref),
        fmm_wi = rel(fmm_wi[1],  fmm_wi_ref),
        fmm_ii = rel(fmm_ii[1],  fmm_ii_ref),
    )
    max_err = maximum(values(checks))
    if max_err < rtol
        @info "✔ mechlem_poly_forward! parity (Float32 rtol=$rtol)  max_rel_err = $(round(max_err, sigdigits=3))"
    else
        @error "✘ mechlem_poly_forward! parity FAILED" checks max_err
        error("Forward model parity test failed")
    end
end

# ╔═╡ 00090030-0000-4000-8000-000000000004
# Fessler μ-preconditioning (Kim–Ramani–Fessler 2015; Long–Fessler 2014).
#
# Form the 2×2 spectrum-weighted mass-attenuation Hessian
#   F_{m,n} = Σ_e  (S_L[e] + S_H[e]) · M[e,m] · M[e,n]
# then T = F^(-1/2) via 2×2 eigendecomposition (symmetric PSD ⇒ analytic).
#
# Passed to the orchestrator alongside M_real; orchestrator derives
#   M_synth = M_real · T
# so that M_synth^T · diag(S_L + S_H) · M_synth = I₂  (per-ray Hessian
# identity) and solves in synth basis.  Huber still applies in real basis:
# per voxel, z_real = T · z_synth, apply Huber on z_real, then pull grad and
# Hess back to synth via  g_synth = T^T · g_real,  H_synth = T^T · H_real · T.
#
# At the end, convert back:  ρ_water / ρ_iodine = T · (α_w, α_i).
begin
    mech_T = let
        if mech_precond === :none
            Float32[1 0; 0 1]
        elseif mech_precond === :fessler
            # Build F = M^T · diag(S_L + S_H) · M  (symmetric 2×2, PSD)
            w = Float64.(mech_S_L) .+ Float64.(mech_S_H)
            MF = Float64.(mech_M)
            F = MF' * (w .* MF)
            # Analytic 2×2 symmetric eigendecomp
            a = F[1, 1];  b = F[1, 2];  d = F[2, 2]
            tr = a + d;  det_F = a * d - b * b
            disc = sqrt(max((a - d)^2 + 4 * b * b, 0.0))
            λ1 = 0.5 * (tr + disc);  λ2 = 0.5 * (tr - disc)
            # Eigenvectors — pair each λ with the null-space vector of F - λI.
            # Guard against b ≈ 0 (already diagonal) to avoid NaN.
            v1, v2 = if abs(b) > 1e-12 * max(abs(a), abs(d))
                v1 = [λ1 - d, b];  v1 ./= sqrt(v1[1]^2 + v1[2]^2)
                v2 = [λ2 - d, b];  v2 ./= sqrt(v2[1]^2 + v2[2]^2)
                v1, v2
            else
                [1.0, 0.0], [0.0, 1.0]
            end
            V = hcat(v1, v2)              # 2×2 orthogonal
            D_inv_sqrt = [1/sqrt(λ1) 0; 0 1/sqrt(λ2)]
            T_sym = V * D_inv_sqrt * V'   # symmetric F^(-1/2)
            Float32.(T_sym)
        else
            error("mech_precond must be :fessler or :none (got $(mech_precond))")
        end
    end

    mech_M_synth = Float32.(mech_M * mech_T)   # [Nₑ × 2]

    # Sanity: print the resulting conditioning gain on F_synth = M_synth^T W M_synth.
    let w = Float64.(mech_S_L) .+ Float64.(mech_S_H), Mraw = Float64.(mech_M),
        Msyn = Float64.(mech_M_synth)
        F_raw  = Mraw' * (w .* Mraw)
        F_syn  = Msyn' * (w .* Msyn)
        κ_raw  = cond(F_raw);  κ_syn = cond(F_syn)
        @info "Fessler T [$(mech_precond)]:  cond(F_raw) = $(round(κ_raw, sigdigits=4))  →  cond(F_synth) = $(round(κ_syn, sigdigits=4))   (target 1.0)"
        @info "T matrix:  [$(round(mech_T[1,1], sigdigits=3))  $(round(mech_T[1,2], sigdigits=3)); $(round(mech_T[2,1], sigdigits=3))  $(round(mech_T[2,2], sigdigits=3))]"
    end
end;

# ╔═╡ 00090010-0000-4000-8000-000000000004
md"""
### 5.4 Counts + scaled spectra

`sim_sino_*.sino` stores air-calibrated log-integrals `-log(I/I₀)`. For
Mechlem's Poisson data term we need raw counts. Invert the log:
`y = I₀ · exp(-p)` with a 1-count floor (matches the DAS hardware minimum in
`driver.jl` STEP 3). Per-pixel I₀ is assumed pixel-independent here (no
bowtie-spectrum shaping modeled — can be lifted later by passing a 3D `S̃`).

Pre-scale the normalized spectra to `S̃ = I₀ · S` so the forward kernel
returns expected counts directly.
"""

# ╔═╡ 00090011-0000-4000-8000-000000000004
begin
    function mech_compute_I0(geom, protocol, sim_opts, scanner)
        _, w_src = BS.resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner = scanner)
        Float32(BS.compute_detector_I0(geom, protocol, sum(w_src)))
    end

    mech_I0_L = mech_compute_I0(sim_sino_low.geom,  protocol_low,  sim_opts, sim_scanner)
    mech_I0_H = mech_compute_I0(sim_sino_high.geom, protocol_high, sim_opts, sim_scanner)

    mech_y_L = Float32.(max.(mech_I0_L .* exp.(-sim_sino_low.sino),  1f0))
    mech_y_H = Float32.(max.(mech_I0_H .* exp.(-sim_sino_high.sino), 1f0))

    mech_S_L_scaled = Float32.(mech_I0_L .* mech_S_L)
    mech_S_H_scaled = Float32.(mech_I0_H .* mech_S_H)

    @info "I₀  80 kVp = $(round(mech_I0_L, sigdigits=4))  |  140 kVp = $(round(mech_I0_H, sigdigits=4))  counts/pixel/view"
    @info "y_L ∈ [$(round(minimum(mech_y_L), sigdigits=3)), $(round(maximum(mech_y_L), sigdigits=3))]  y_H ∈ [$(round(minimum(mech_y_H), sigdigits=3)), $(round(maximum(mech_y_H), sigdigits=3))]"
end;

# ╔═╡ 00090012-0000-4000-8000-000000000004
md"""
### 5.5 Data-term gradient + Hessian (per channel)

Port of the Mechlem2017.m inner loop for ONE channel. Computes, at every
voxel, the 2-vector gradient `(g_w, g_i)` and 3-unique-entry Hessian
`(h_ww, h_wi, h_ii)` of the Poisson-likelihood SQS surrogate:

    ray_grad_m = (1 - y/ȳ) · fmu_m
    grad_m     = Aᵀ · ray_grad_m
    ray_hess_mn = fmm_mn · (A·1)            [Sum_aijs in MATLAB]
    hess_mn    = Aᵀ · ray_hess_mn

`A·1` is the per-ray volume length — precomputed once per geometry by
forward-projecting an all-ones volume.
"""

# ╔═╡ 00090013-0000-4000-8000-000000000004
function mechlem_data_term!(
        grad_w, grad_i, hess_ww, hess_wi, hess_ii,
        z_w, z_i, y, S, M, geom, row_sum,
        proj_w, proj_i, ȳ, fmu_w, fmu_i, fmm_ww, fmm_wi, fmm_ii;
        ybar_floor::Float32 = 1f-6,
    )
    BS.siddon_forward_project!(proj_w, z_w, geom)
    BS.siddon_forward_project!(proj_i, z_i, geom)

    mechlem_poly_forward!(ȳ, fmu_w, fmu_i, fmm_ww, fmm_wi, fmm_ii,
                          proj_w, proj_i, S, M)

    let yb_floor = ybar_floor
        AK.foreachindex(ȳ) do p
            yb = max(ȳ[p], yb_floor)
            one_minus_r = 1f0 - y[p] / yb
            fmu_w[p]  = one_minus_r * fmu_w[p]
            fmu_i[p]  = one_minus_r * fmu_i[p]
            rs = row_sum[p]
            fmm_ww[p] = fmm_ww[p] * rs
            fmm_wi[p] = fmm_wi[p] * rs
            fmm_ii[p] = fmm_ii[p] * rs
        end
    end

    BS.backproject!(grad_w,  fmu_w,  geom; weighted = false)
    BS.backproject!(grad_i,  fmu_i,  geom; weighted = false)
    BS.backproject!(hess_ww, fmm_ww, geom; weighted = false)
    BS.backproject!(hess_wi, fmm_wi, geom; weighted = false)
    BS.backproject!(hess_ii, fmm_ii, geom; weighted = false)
    return nothing
end

# ╔═╡ 00090014-0000-4000-8000-000000000004
md"""
### 5.6 Huber regularization (2-D, per-slice, per-material) — SBM-aware

Port of `SQSregul.m` + `SQSdiffWithNeighbors.m` + `Huber.m` — 3×3 replicate-
padded neighborhood, per-slice (matches paper's 2-D Huber).  For each voxel
sums the 8 non-zero neighbor differences, then applies the SQSregul outer
factors `grad ×= 2λ/N_subsets`, `hess ×= 4λ/N_subsets`.

With Fessler preconditioning, `z` is in synth basis and `δ, λ` are physical
units — so the kernel:
1. Computes `z_real = T · z_synth` per voxel (MATLAB's `input * T'`)
2. Applies Huber to `z_real` differences
3. Pulls gradient back: `g_synth = T^T · g_real`       (MATLAB's `regul_grad * T`)
4. Pulls Hessian back: `H_synth = T^T · diag(h_real) · T`  — FULL 2×2 now.

With `T = I₂` the kernel collapses to per-material diagonal (the prior
behavior), so the unpreconditioned path is still supported.
"""

# ╔═╡ 00090015-0000-4000-8000-000000000004
# Computes real-basis z_real = T · z_synth, the Huber grad/hess in real basis,
# and pulls them back to synth basis:
#   rg_w, rg_i            = T^T · (g_real_w, g_real_i)
#   rh_ww, rh_wi, rh_ii   = (T^T · diag(h_real) · T) 2×2 entries
# Both `z_w, z_i` inputs are SYNTHESIS basis.
function mechlem_huber_2d!(
        rg_w, rg_i, rh_ww, rh_wi, rh_ii, z_w, z_i,
        δ_w::Float32, δ_i::Float32, λ_w::Float32, λ_i::Float32, nsubsets::Int,
        T11::Float32, T12::Float32, T21::Float32, T22::Float32,
    )
    nx = Int32(size(z_w, 1));  ny = Int32(size(z_w, 2))
    inv_ns = 1f0 / Float32(nsubsets)
    AK.foreachindex(z_w) do idx
        idx_0 = Int32(idx - 1)
        i = (idx_0 % nx) + Int32(1)
        idx_0 = idx_0 ÷ nx
        j = (idx_0 % ny) + Int32(1)
        k = (idx_0 ÷ ny) + Int32(1)

        zw0_s = z_w[i, j, k];  zi0_s = z_i[i, j, k]
        # synth → real at center voxel
        zw0 = T11 * zw0_s + T12 * zi0_s
        zi0 = T21 * zw0_s + T22 * zi0_s

        # Accumulate Huber' (g_real) and Huber'' (h_real), per material, 8 neighbors.
        gw_r = 0f0;  gi_r = 0f0;  hw_r = 0f0;  hi_r = 0f0
        @inbounds for dj in Int32(-1):Int32(1), di in Int32(-1):Int32(1)
            (di == Int32(0) && dj == Int32(0)) && continue
            ii = clamp(i + di, Int32(1), nx)
            jj = clamp(j + dj, Int32(1), ny)
            zwn_s = z_w[ii, jj, k];  zin_s = z_i[ii, jj, k]
            zwn = T11 * zwn_s + T12 * zin_s
            zin = T21 * zwn_s + T22 * zin_s
            tw = zw0 - zwn
            ti = zi0 - zin
            aw = abs(tw);  ai = abs(ti)
            gw_r += (aw < δ_w) ? 2f0 * tw : 2f0 * δ_w * sign(tw)
            gi_r += (ai < δ_i) ? 2f0 * ti : 2f0 * δ_i * sign(ti)
            hw_r += (aw < δ_w) ? 2f0 : 0f0
            hi_r += (ai < δ_i) ? 2f0 : 0f0
        end

        # Real-basis SQS factors (apply 2λ/N on grad, 4λ/N on hess — before T pull-back)
        g_real_w = 2f0 * gw_r * λ_w * inv_ns
        g_real_i = 2f0 * gi_r * λ_i * inv_ns
        h_real_w = 4f0 * hw_r * λ_w * inv_ns
        h_real_i = 4f0 * hi_r * λ_i * inv_ns

        # Pull back to synth basis:
        #   g_synth = T^T · g_real        (column vec convention)
        #   H_synth = T^T · diag(h_real) · T   (full 2×2)
        rg_w[i, j, k] = T11 * g_real_w + T21 * g_real_i
        rg_i[i, j, k] = T12 * g_real_w + T22 * g_real_i
        rh_ww[i, j, k] = h_real_w * T11 * T11 + h_real_i * T21 * T21
        rh_wi[i, j, k] = h_real_w * T11 * T12 + h_real_i * T21 * T22
        rh_ii[i, j, k] = h_real_w * T12 * T12 + h_real_i * T22 * T22
    end
    return nothing
end

# ╔═╡ 00090016-0000-4000-8000-000000000004
md"""
### 5.7 Per-voxel 2×2 analytic Newton update

Port of `GetNewtonUpdate.m`.  With Fessler preconditioning, regul Hessian
is FULL 2×2 (not diagonal) — per-voxel Newton becomes:

    H  = [h_ww + rh_ww,  h_wi + rh_wi;
          h_wi + rh_wi,  h_ii + rh_ii]
    u  = H⁻¹ · g         (2×2 analytic inverse)

`det(H)` is clamped at `mech_det_floor` to keep GPU kernels branch-free when
the SQS Hessian degenerates inside air voxels.
"""

# ╔═╡ 00090017-0000-4000-8000-000000000004
function mechlem_newton_2x2!(
        u_w, u_i, g_w, g_i, h_ww, h_wi, h_ii, rh_ww, rh_wi, rh_ii, det_floor::Float32,
    )
    AK.foreachindex(u_w) do idx
        H_ww = h_ww[idx] + rh_ww[idx]
        H_wi = h_wi[idx] + rh_wi[idx]
        H_ii = h_ii[idx] + rh_ii[idx]
        det_H = H_ww * H_ii - H_wi * H_wi
        det_H = (det_H < det_floor) ? det_floor : det_H
        gw = g_w[idx];  gi = g_i[idx]
        u_w[idx] = ( gw * H_ii - gi * H_wi) / det_H
        u_i[idx] = (-gw * H_wi + gi * H_ww) / det_H
    end
    return nothing
end

# ╔═╡ 00090018-0000-4000-8000-000000000004
md"""
### 5.8 Ordered-Subsets + Nesterov orchestrator — `mechlem_reconstruct`

Outer loop over iterations × subsets (Mory 2018 Alg. 4, Appendix A.4). Both
kVp channels contribute to the same voxel-gradient / voxel-Hessian via linear
summation (separate `A_L` / `A_H` projectors, shared `z_w`, `z_i`). Nesterov
coefficients `t_k` / `ratio_k` are pre-computed on the host as per
Mechlem2017.m lines 27–33 — strict port including the MATLAB convention
`s[k] = s[k-1] + t[k]` (not `t[k-1]`) and length `n_steps+1` for both arrays.

Ordered subsets is implemented by building per-subset sub-geometries (`A_L`,
`A_H`, `y_L`, `y_H`, `row_sum_L`, `row_sum_H` sliced along the view axis).
The subset index cycles `1..NbSubsets` within each iteration.

**Dual-kVp adaptations vs MATLAB Mechlem2017 (PCCT-oriented):**
- MATLAB uses **one** shared `A` + a bin index `b=1..N_bins` with shared
  spectra per bin. Dual-kVp has **two different scans** (different view sets
  per kVp). We process each channel end-to-end (FP → poly model → BP) and sum
  the resulting image-space gradients / Hessians — equivalent by linearity
  of BP to MATLAB's `squeeze(sum(ToBackProject, 1))` across the bin axis.
- View partition: **round-robin over each channel's view list independently**
  (not MATLAB's random shuffle then contiguous split). Round-robin gives
  better angular coverage per subset — standard Hudson-Larkin OS for CT.
- **Fessler μ-preconditioning** (`mech_T = F^(-1/2)`, see §5.2b): for
  water+iodine the unpreconditioned Hessian has κ ≈ 20,000; with Fessler
  T the per-ray spectral Hessian becomes I₂. All iteration happens in the
  synth basis; `z_synth = T^(-1) z_real` gets you out.  Strict `T = I₂`
  mode (Mory MATLAB default) still supported via `mech_precond = :none`.
- `nonneg = true` (default): project `z, α, v ≥ 0` every step — from the
  Mechlem 2018 IEEE paper (Mory's MATLAB drops this). Needed because iodine
  can dip slightly negative on noise and blow up `exp(+M·|proj|)`. Set
  `nonneg = false` for strict Mory-MATLAB parity.

Returns `(ρ_water, ρ_iodine)` on the host (Array), both in physical g/mL —
converts the synth-basis `α_k` iterate (MATLAB line 117) back to real basis
via `ρ = T · α`.
"""

# ╔═╡ 00090019-0000-4000-8000-000000000004
begin
    # Subset the view axis of a CTGeometry by index list.
    function mech_subset_geom(geom::BS.CTGeometry, indices::AbstractVector{<:Integer})
        BS.CTGeometry(
            geom.SAD, geom.SDD,
            length(indices),
            geom.n_rows, geom.n_cols,
            geom.pixel_size, geom.pixel_row_size,
            geom.angles[indices],
            geom.source_positions[:, indices],
            geom.detector_centers[:, indices],
            geom.detector_u[:, indices],
            geom.detector_v[:, indices],
            geom.fov,
        )
    end

    # Split 1:n_views into `nsub` round-robin subsets (Mechlem MATLAB shuffles
    # globally — we use round-robin for deterministic angular coverage per
    # subset, which also matches the Ordered-Subsets literature since Hudson &
    # Larkin 1994).
    mech_split_views(n_views::Int, nsub::Int) =
        [collect(s:nsub:n_views) for s in 1:nsub]

    # Pre-compute the Nesterov t_k / ratio_k sequences (Mechlem2017.m L27–33).
    #
    # Strict 1:1 port:
    #   t[1] = 1
    #   t[k] = ½(1 + √(1 + 4·t[k-1]²))         for k = 2..n_steps+1
    #   s[1] = 0
    #   s[k] = s[k-1] + t[k]                    for k = 2..n_steps+1   (NOTE: t[k], not t[k-1])
    #   ratios[k] = t[k] / s[k]                 (ratios[1] = 0, used only at i+1 in main loop)
    #
    # Main loop uses t[step] for v-update, ratios[step+1] for z-update, where
    # step = 1..n_steps.  Hence BOTH arrays must have length n_steps+1 so that
    # ratios[n_steps+1] is valid at the final step.
    function mech_nesterov_coeffs(n_steps::Int)
        N = n_steps + 1
        t = ones(Float32, N)
        for k in 2:N
            t[k] = 0.5f0 * (1f0 + sqrt(1f0 + 4f0 * t[k - 1]^2))
        end
        s = zeros(Float32, N)
        for k in 2:N
            s[k] = s[k - 1] + t[k]
        end
        ratios = zeros(Float32, N)
        for k in 2:N
            ratios[k] = t[k] / s[k]
        end
        t, ratios
    end
end

# ╔═╡ 00090020-0000-4000-8000-000000000004
function mechlem_reconstruct(
        y_L::AbstractArray{Float32, 3}, geom_L::BS.CTGeometry, S_L::AbstractVector{Float32},
        y_H::AbstractArray{Float32, 3}, geom_H::BS.CTGeometry, S_H::AbstractVector{Float32},
        M::AbstractMatrix{Float32},                # M_synth = M_real · T  (Fessler-preconditioned)
        T::AbstractMatrix{Float32},                # 2×2 preconditioner (I₂ for unpreconditioned)
        volume_size::NTuple{3, Int};
        n_iter::Int             = 30,
        n_subsets::Int          = 4,
        huber_δ::Vector{Float32} = Float32[0.1, 0.001],
        huber_λ::Vector{Float32} = Float32[3f0, 30_000f0],
        ybar_floor::Float32     = 1f-6,
        det_floor::Float32      = 1f-12,
        nonneg::Bool            = true,   # project z,α,v ≥ 0 (Mechlem IEEE 2018) — off for strict Mory parity
        # ── Warm start (in PHYSICAL / real basis, g/mL) ──────────────────
        # If supplied, converted to synth basis internally via z_synth = T⁻¹·z_real.
        # `nothing` ⇒ zero init (Mory 2017 default).
        z0_w::Union{Nothing, AbstractArray{Float32, 3}} = nothing,
        z0_i::Union{Nothing, AbstractArray{Float32, 3}} = nothing,
        verbose::Bool           = true,
        dev_ref::AbstractArray  = y_L,    # backend (CPU / MtlArray / CuArray) template
    )
    n_subsets <= 6 || error("n_subsets ≤ 6 required (Mory 2018 §3.1 — diverges above)")

    # ─── Host-side view partitions + sub-geoms ───
    subs_L = mech_split_views(size(y_L, 3), n_subsets)
    subs_H = mech_split_views(size(y_H, 3), n_subsets)
    subgeoms_L  = [mech_subset_geom(geom_L, sv) for sv in subs_L]
    subgeoms_H  = [mech_subset_geom(geom_H, sv) for sv in subs_H]
    ncols_L = size(y_L, 1);  nrows_L = size(y_L, 2)
    ncols_H = size(y_H, 1);  nrows_H = size(y_H, 2)

    # ─── Stage measurements + spectra + M onto backend ───
    stage(a) = (dev_ref isa Array ? a : (b = similar(dev_ref, eltype(a), size(a)); copyto!(b, a); b))
    y_L_d = stage(y_L);  y_H_d = stage(y_H)
    S_L_d = stage(S_L);  S_H_d = stage(S_H)
    M_d   = stage(M)

    # Slice measurements per subset (views = dim 3)
    y_L_subs = [stage(y_L[:, :, sv]) for sv in subs_L]
    y_H_subs = [stage(y_H[:, :, sv]) for sv in subs_H]

    # ─── Pre-compute A·1 (row sums) per subset, per channel ───
    row_sums_L = Vector{typeof(dev_ref)}(undef, n_subsets)
    row_sums_H = Vector{typeof(dev_ref)}(undef, n_subsets)
    let ones_vol = similar(dev_ref, Float32, volume_size);  fill!(ones_vol, 1f0)
        for s in 1:n_subsets
            rL = similar(dev_ref, Float32, (ncols_L, nrows_L, length(subs_L[s])))
            BS.siddon_forward_project!(rL, ones_vol, subgeoms_L[s]);  row_sums_L[s] = rL
            rH = similar(dev_ref, Float32, (ncols_H, nrows_H, length(subs_H[s])))
            BS.siddon_forward_project!(rH, ones_vol, subgeoms_H[s]);  row_sums_H[s] = rH
        end
    end

    # ─── Image-space state ───
    z_w = similar(dev_ref, Float32, volume_size);  fill!(z_w, 0f0)
    z_i = similar(dev_ref, Float32, volume_size);  fill!(z_i, 0f0)
    v_w = similar(z_w);  fill!(v_w, 0f0)
    v_i = similar(z_i);  fill!(v_i, 0f0)
    α_w = similar(z_w);  fill!(α_w, 0f0)
    α_i = similar(z_i);  fill!(α_i, 0f0)

    # ─── Warm start: convert physical (real) → synth basis via T⁻¹ ───────
    if z0_w !== nothing && z0_i !== nothing
        size(z0_w) == volume_size || error("z0_w shape $(size(z0_w)) ≠ volume_size $(volume_size)")
        size(z0_i) == volume_size || error("z0_i shape $(size(z0_i)) ≠ volume_size $(volume_size)")
        z0_w_d = stage(z0_w)
        z0_i_d = stage(z0_i)
        # T⁻¹ computed below, so do the conversion after; hold a temp flag.
        has_warm_start = true
    else
        z0_w_d = nothing;  z0_i_d = nothing
        has_warm_start = false
    end

    g_w  = similar(z_w); g_i  = similar(z_i)
    h_ww = similar(z_w); h_wi = similar(z_w); h_ii = similar(z_w)
    rg_w = similar(z_w); rg_i = similar(z_i)
    rh_ww = similar(z_w); rh_wi = similar(z_w); rh_ii = similar(z_w)   # FULL 2×2 reg Hess (Fessler T makes it non-diagonal)
    u_w  = similar(z_w); u_i  = similar(z_i)

    # Dual per-channel image accumulators (we reuse g/h but need a second for the other channel)
    g2_w  = similar(z_w); g2_i  = similar(z_i)
    h2_ww = similar(z_w); h2_wi = similar(z_w); h2_ii = similar(z_w)

    # ─── Sinogram scratch per channel (largest subset size) ───
    sub_max_L = maximum(length.(subs_L));  sub_max_H = maximum(length.(subs_H))
    proj_w_L = similar(dev_ref, Float32, (ncols_L, nrows_L, sub_max_L))
    proj_i_L = similar(proj_w_L)
    ȳ_L      = similar(proj_w_L)
    fmu_w_L  = similar(proj_w_L); fmu_i_L = similar(proj_w_L)
    fmm_ww_L = similar(proj_w_L); fmm_wi_L = similar(proj_w_L); fmm_ii_L = similar(proj_w_L)
    proj_w_H = similar(dev_ref, Float32, (ncols_H, nrows_H, sub_max_H))
    proj_i_H = similar(proj_w_H)
    ȳ_H      = similar(proj_w_H)
    fmu_w_H  = similar(proj_w_H); fmu_i_H = similar(proj_w_H)
    fmm_ww_H = similar(proj_w_H); fmm_wi_H = similar(proj_w_H); fmm_ii_H = similar(proj_w_H)

    # Views into the scratch matching each subset's actual size
    view3(buf, n) = @view buf[:, :, 1:n]

    # ─── Nesterov coefficients ───
    t_nest, rat_nest = mech_nesterov_coeffs(n_iter * n_subsets)

    # ─── Preconditioner / inverse for synth ↔ real basis roundtrips ───
    T11, T12, T21, T22 = T[1, 1], T[1, 2], T[2, 1], T[2, 2]
    det_T = T11 * T22 - T12 * T21
    abs(det_T) > 1f-30 || error("mechlem_reconstruct: preconditioner T is singular (det ≈ 0)")
    Ti11 =  T22 / det_T;  Ti12 = -T12 / det_T
    Ti21 = -T21 / det_T;  Ti22 =  T11 / det_T

    # ─── Apply warm-start conversion now that T⁻¹ is available ───
    # z_synth = T⁻¹ · z_real   (per voxel matmul, on the backend)
    if has_warm_start
        let z0w = z0_w_d, z0i = z0_i_d, zw = z_w, zi = z_i,
                αw = α_w, αi = α_i, vw = v_w, vi = v_i,
                a11 = Ti11, a12 = Ti12, a21 = Ti21, a22 = Ti22
            AK.foreachindex(zw) do idx
                rw = z0w[idx];  ri = z0i[idx]
                sw = a11 * rw + a12 * ri
                si = a21 * rw + a22 * ri
                zw[idx] = sw;  zi[idx] = si
                αw[idx] = sw;  αi[idx] = si
                vw[idx] = sw;  vi[idx] = si   # Nesterov: v₀ = z₀ (canonical warm start)
            end
        end
        verbose && @info "[mechlem] warm-started from physical (ρ_water, ρ_iodine) volumes (z_synth = T⁻¹ · z_real)"
    end

    # ─── Iterate ───
    δ_w, δ_i = huber_δ[1], huber_δ[2]
    λ_w, λ_i = huber_λ[1], huber_λ[2]
    step = 0
    t_start = time()
    for iter in 1:n_iter
        for subset in 1:n_subsets
            step += 1

            nL = length(subs_L[subset]);  nH = length(subs_H[subset])
            gL = subgeoms_L[subset];      gH = subgeoms_H[subset]

            # ── LOW channel: contributes to g_*, h_** ──
            mechlem_data_term!(
                g_w, g_i, h_ww, h_wi, h_ii,
                z_w, z_i, y_L_subs[subset], S_L_d, M_d, gL, row_sums_L[subset],
                view3(proj_w_L, nL), view3(proj_i_L, nL), view3(ȳ_L, nL),
                view3(fmu_w_L, nL),  view3(fmu_i_L, nL),
                view3(fmm_ww_L, nL), view3(fmm_wi_L, nL), view3(fmm_ii_L, nL);
                ybar_floor = ybar_floor,
            )

            # ── HIGH channel: contributes to g2_*, h2_** ──
            mechlem_data_term!(
                g2_w, g2_i, h2_ww, h2_wi, h2_ii,
                z_w, z_i, y_H_subs[subset], S_H_d, M_d, gH, row_sums_H[subset],
                view3(proj_w_H, nH), view3(proj_i_H, nH), view3(ȳ_H, nH),
                view3(fmu_w_H, nH),  view3(fmu_i_H, nH),
                view3(fmm_ww_H, nH), view3(fmm_wi_H, nH), view3(fmm_ii_H, nH);
                ybar_floor = ybar_floor,
            )

            # Sum channels (data gradient + data Hessian)
            AK.foreachindex(g_w) do idx
                g_w[idx]  += g2_w[idx];   g_i[idx]  += g2_i[idx]
                h_ww[idx] += h2_ww[idx];  h_wi[idx] += h2_wi[idx];  h_ii[idx] += h2_ii[idx]
            end

            # ── Huber regularization (in real basis via T) into synth-basis rg_*, rh_** ──
            mechlem_huber_2d!(rg_w, rg_i, rh_ww, rh_wi, rh_ii, z_w, z_i,
                              δ_w, δ_i, λ_w, λ_i, n_subsets,
                              T11, T12, T21, T22)

            # Add Huber (synth-basis) gradient to data gradient; full Hess added in Newton kernel.
            AK.foreachindex(g_w) do idx
                g_w[idx] += rg_w[idx]
                g_i[idx] += rg_i[idx]
            end

            # ── Per-voxel 2×2 Newton update: u = H⁻¹ · g  (H_data + H_reg, both full 2×2) ──
            mechlem_newton_2x2!(u_w, u_i, g_w, g_i, h_ww, h_wi, h_ii,
                                rh_ww, rh_wi, rh_ii, det_floor)

            # ── Nesterov update (MATLAB Mechlem2017.m L110–114) ──
            # MATLAB: NO positivity. Mechlem IEEE 2018: projects z_real ≥ 0.
            # With Fessler T, `z_synth ≥ 0` is NOT equivalent to `z_real ≥ 0`,
            # so the projection is done in real basis (T · z_synth ≥ 0) and
            # then pushed back via T^(-1).  Only `α` and `z` are projected —
            # the momentum aux `v` stays free (can have either sign).
            t_k   = t_nest[step]
            rat   = rat_nest[step + 1]
            if nonneg
                AK.foreachindex(z_w) do idx
                    uw = u_w[idx];  ui = u_i[idx]
                    # α in synth basis
                    αw_s = z_w[idx] - uw;  αi_s = z_i[idx] - ui
                    # project α ≥ 0 in real basis
                    αw_r = T11 * αw_s + T12 * αi_s
                    αi_r = T21 * αw_s + T22 * αi_s
                    αw_r = max(αw_r, 0f0);  αi_r = max(αi_r, 0f0)
                    αw_s = Ti11 * αw_r + Ti12 * αi_r
                    αi_s = Ti21 * αw_r + Ti22 * αi_r
                    # v (free — no projection)
                    vw = v_w[idx] - t_k * uw;  vi = v_i[idx] - t_k * ui
                    # z in synth basis, then real-basis positivity + pull back
                    zw_s = αw_s + rat * (vw - αw_s)
                    zi_s = αi_s + rat * (vi - αi_s)
                    zw_r = T11 * zw_s + T12 * zi_s
                    zi_r = T21 * zw_s + T22 * zi_s
                    zw_r = max(zw_r, 0f0);  zi_r = max(zi_r, 0f0)
                    z_w[idx] = Ti11 * zw_r + Ti12 * zi_r
                    z_i[idx] = Ti21 * zw_r + Ti22 * zi_r
                    α_w[idx] = αw_s;  α_i[idx] = αi_s
                    v_w[idx] = vw;     v_i[idx] = vi
                end
            else
                AK.foreachindex(z_w) do idx
                    uw = u_w[idx];  ui = u_i[idx]
                    αw = z_w[idx] - uw;  αi = z_i[idx] - ui
                    vw = v_w[idx] - t_k * uw;  vi = v_i[idx] - t_k * ui
                    z_w[idx] = αw + rat * (vw - αw)
                    z_i[idx] = αi + rat * (vi - αi)
                    α_w[idx] = αw;  α_i[idx] = αi
                    v_w[idx] = vw;  v_i[idx] = vi
                end
            end
        end
        if verbose
            elapsed = time() - t_start
            # Convert α (synth basis) → physical (g/mL) for monitoring
            αw_h = Array(α_w);  αi_h = Array(α_i)
            ρw_min = typemax(Float32);  ρw_max = typemin(Float32)
            ρi_min = typemax(Float32);  ρi_max = typemin(Float32)
            @inbounds for idx in eachindex(αw_h)
                ρw = T11 * αw_h[idx] + T12 * αi_h[idx]
                ρi = T21 * αw_h[idx] + T22 * αi_h[idx]
                ρw_min = min(ρw_min, ρw);  ρw_max = max(ρw_max, ρw)
                ρi_min = min(ρi_min, ρi);  ρi_max = max(ρi_max, ρi)
            end
            @info "Mechlem iter $(iter)/$(n_iter)  ($(round(elapsed; digits=1)) s)  |  ρ_water ∈ [$(round(ρw_min; digits=3)), $(round(ρw_max; digits=3))] g/mL   ρ_iodine ∈ [$(round(ρi_min; digits=5)), $(round(ρi_max; digits=5))] g/mL"
        end
    end

    # Convert final α (synth basis) → physical ρ (real basis) via ρ = T · α
    ρw_h = similar(α_w);  ρi_h = similar(α_i)
    AK.foreachindex(ρw_h) do idx
        αw_s = α_w[idx];  αi_s = α_i[idx]
        ρw_h[idx] = T11 * αw_s + T12 * αi_s
        ρi_h[idx] = T21 * αw_s + T22 * αi_s
    end
    (ρ_water = Array(ρw_h), ρ_iodine = Array(ρi_h))
end

# ╔═╡ 00090021-0000-4000-8000-000000000004
md"""
### 5.9 Run Mechlem + visualize output basis volumes

Finally calls the orchestrator on the Gammex 80/140 kVp counts with the
hyperparameters from §5.1.  Plots the two output volumes as 2-panel
heatmaps at a representative axial slice — same style as the Cong+GN
reference figure: `ρ_water(r)` on the left, `ρ_iodine(r)` on the right.
"""

# ╔═╡ 00090022-0000-4000-8000-000000000004
mech_result = let
    t0 = time()
    volume_size = sim_matrix_size

    # Stage y_L / y_H onto the same backend as the phantom (to_gpu).
    y_L_dev = to_gpu(mech_y_L)
    y_H_dev = to_gpu(mech_y_H)

    res = mechlem_reconstruct(
        y_L_dev, sim_sino_low.geom,  Float32.(mech_S_L_scaled),
        y_H_dev, sim_sino_high.geom, Float32.(mech_S_H_scaled),
        Float32.(mech_M_synth),                 # Fessler-preconditioned M
        Float32.(mech_T),                       # 2×2 preconditioner (I₂ if :none)
        volume_size;
        n_iter     = mech_n_iter,
        n_subsets  = mech_n_subsets,
        huber_δ    = mech_huber_δ,
        huber_λ    = mech_huber_λ,
        ybar_floor = mech_ybar_floor,
        det_floor  = mech_det_floor,
        nonneg     = mech_nonneg,
        z0_w       = mech_z0.ρ_water,      # §5.0 Cong warm start (physical g/mL)
        z0_i       = mech_z0.ρ_iodine,
        verbose    = mech_verbose,
        dev_ref    = y_L_dev,
    )
    @info "Mechlem done  ($(round(time() - t0; digits=1)) s total)"
    y_L_dev = nothing;  y_H_dev = nothing;  clear_gpu!()
    res
end;

# ╔═╡ 00090023-0000-4000-8000-000000000004
let
    mid = size(mech_result.ρ_water, 3) ÷ 2
    ρw = mech_result.ρ_water[:, :, mid]
    ρi = mech_result.ρ_iodine[:, :, mid]

    fig = CM.Figure(size = (1250, 560), fontsize = 13)
    ax1 = CM.Axis(fig[1, 1]; title = "ρ_water (g/mL) — Mechlem2018  (slice $mid)",
                  aspect = CM.DataAspect())
    hm1 = CM.heatmap!(ax1, ρw; colormap = :viridis)
    CM.Colorbar(fig[1, 2], hm1)

    ax2 = CM.Axis(fig[1, 3]; title = "ρ_iodine (g/mL) — Mechlem2018  (slice $mid)",
                  aspect = CM.DataAspect())
    hm2 = CM.heatmap!(ax2, ρi; colormap = :viridis)
    CM.Colorbar(fig[1, 4], hm2)

    CM.save(joinpath(RESULTS_DIR, "mechlem_basis_output.png"), fig, px_per_unit = 2)
    @info "Mechlem output: ρ_water ∈ [$(round(minimum(ρw); digits=3)), $(round(maximum(ρw); digits=3))] g/mL,  ρ_iodine ∈ [$(round(minimum(ρi); digits=4)), $(round(maximum(ρi); digits=4))] g/mL"
    fig
end

# ╔═╡ 00100001-0000-4000-8000-000000000004
md"""
## 6. Virtual Monochromatic Imaging (VMI)

Direct physical synthesis from the Mechlem basis volumes — **no scaling
constants, no per-energy calibration, no HU LUT**, just NIST mass-attenuation
look-ups combined via the material-density model:

    μ(r, E) = (μ/ρ)_water(E) · ρ_water(r)  +  (μ/ρ)_iodine(E) · ρ_iodine(r)     [cm⁻¹]

where `(μ/ρ)_m(E)` comes from `BS.compute_mass_μ_at_energy` (NIST XCOM
via XrayAttenuation.jl — same table the sim forward-projects through, so
self-consistent by construction).  HU then follows the canonical definition:

    HU(r, E) = 1000 · ( μ(r, E) − μ_water_ref(E) ) / μ_water_ref(E)

with `μ_water_ref(E) = 1.0 g/cm³ · (μ/ρ)_water(E)` — the reference at the
SAME energy, not a single global reference. This guarantees the physical
sanity checks hold exactly at every E:

- air (`ρ_w = ρ_i = 0`)                       →  HU = −1000
- pure water at 1 g/cm³ (`ρ_w = 1, ρ_i = 0`)  →  HU = 0
- 10 mg/mL iodine at 40 keV                   →  HU ≈ +1700–2000 (K-edge boost)
- 10 mg/mL iodine at 140 keV                  →  HU ≈ +200        (Compton only)

Each target-energy look-up is a single XCOM table hit — no polynomial fits,
no spectrum-averaged effective attenuation, no beam-hardening correction (the
basis volumes are already monochromatic by construction).
"""

# ╔═╡ 00100002-0000-4000-8000-000000000004
begin
    # Standard GE GSI VMI energy grid (40–140 keV).
    vmi_energies_keV = Float32[40, 50, 70, 100, 140]
end;

# ╔═╡ 00100003-0000-4000-8000-000000000004
# VMI synthesis. For each target energy E_k build:
#   μ_vols[k][v]  = μ(r_v, E_k)         [cm⁻¹]
#   hu_vols[k][v] = 1000·(μ/μ_w_ref−1)  [HU]
# Pull NIST (μ/ρ) per material at E_k ONCE on the host, then run a per-voxel
# threaded loop — one multiply-add + HU scale per voxel per energy, trivial
# on CPU; no reason to roundtrip through the GPU.
#
# **Circular-FOV mask**: the reconstruction grid is square (512×512) but the
# scan FOV is circular (diameter = sim_recon_geom.fov_cm).  Voxels in the
# corners of the grid are outside the circular FOV, receive incomplete ray
# coverage, and carry undefined reconstruction values (noise-dominated).
# We set those voxels to `NaN` in μ and HU so (a) downstream stats naturally
# skip them and (b) CairoMakie renders them blank instead of clipping the
# display window to garbage.  Inside the FOV the synthesis is pure physics.
vmi_result = let
    ρw = mech_result.ρ_water
    ρi = mech_result.ρ_iodine
    size(ρw) == size(ρi) || error("ρ_water and ρ_iodine shape mismatch")

    μρ_w = [Float32(BS.compute_mass_μ_at_energy(XA.Materials.water,  Float64(E)))
            for E in vmi_energies_keV]
    μρ_i = [Float32(BS.compute_mass_μ_at_energy(XA.Elements.Iodine,  Float64(E)))
            for E in vmi_energies_keV]
    μ_w_ref = μρ_w   # μ_water(E) = (μ/ρ)_w(E) × 1 g/cm³

    nx, ny, nz = size(ρw)
    pixel_cm      = Float32(sim_recon_geom.fov_cm / nx)
    fov_rad_sq_cm = Float32((sim_recon_geom.fov_cm / 2)^2)
    cx_f          = Float32((nx + 1) / 2)
    cy_f          = Float32((ny + 1) / 2)

    # Pre-compute the 2-D FOV mask once (broadcast to 3-D inside the loop).
    fov_mask_2d = Array{Bool, 2}(undef, nx, ny)
    @inbounds for iy in 1:ny, ix in 1:nx
        x = (Float32(ix) - cx_f) * pixel_cm
        y = (Float32(iy) - cy_f) * pixel_cm
        fov_mask_2d[ix, iy] = (x * x + y * y) <= fov_rad_sq_cm
    end

    n_E     = length(vmi_energies_keV)
    μ_vols  = Vector{Array{Float32, 3}}(undef, n_E)
    hu_vols = Vector{Array{Float32, 3}}(undef, n_E)

    t0 = time()
    for k in 1:n_E
        μ_k   = similar(ρw)
        hu_k  = similar(ρw)
        mw    = μρ_w[k]
        mi    = μρ_i[k]
        inv_w = 1f0 / μ_w_ref[k]
        @inbounds Threads.@threads for iz in 1:nz
            for iy in 1:ny, ix in 1:nx
                if fov_mask_2d[ix, iy]
                    μ = mw * ρw[ix, iy, iz] + mi * ρi[ix, iy, iz]
                    μ_k[ix, iy, iz]  = μ
                    hu_k[ix, iy, iz] = 1f3 * (μ * inv_w - 1f0)
                else
                    μ_k[ix, iy, iz]  = NaN32
                    hu_k[ix, iy, iz] = NaN32
                end
            end
        end
        μ_vols[k]  = μ_k
        hu_vols[k] = hu_k
    end
    dt = time() - t0

    @info "[VMI] synthesized $(n_E) energies $(Int.(Float64.(vmi_energies_keV))) keV in $(round(dt, digits=2)) s  ($(Threads.nthreads()) threads)"
    for k in 1:n_E
        hu_pred = 1f3 * ((μρ_w[k] * 1f0 + μρ_i[k] * 0.010f0) / μ_w_ref[k] - 1f0)
        @info "  $(Int(vmi_energies_keV[k])) keV: (μ/ρ)_w=$(round(μρ_w[k], digits=4))  (μ/ρ)_I=$(round(μρ_i[k], digits=4)) cm²/g   |   predicted HU @ 10 mg/mL I = $(round(hu_pred; digits=0))"
    end

    (energies_keV = vmi_energies_keV,
     μ_mass_water = μρ_w, μ_mass_iodine = μρ_i, μ_water_ref = μ_w_ref,
     μ = μ_vols, hu = hu_vols, fov_mask_2d = fov_mask_2d)
end;

# ╔═╡ 00100004-0000-4000-8000-000000000004
# Multi-energy HU panel — fixed W/L = 700/150 (display range -200..500 HU,
# standard soft-tissue review window).  NaN voxels (outside-FOV corners)
# render blank via `nan_color`.  Panels stacked vertically.
let
    mid = size(vmi_result.hu[1], 3) ÷ 2
    n_E = length(vmi_result.energies_keV)
    hu_lo, hu_hi = -200.0, 500.0

    fig = CM.Figure(size = (560, 440 * n_E), fontsize = 13)
    for (k, E) in enumerate(vmi_result.energies_keV)
        hu_slice = vmi_result.hu[k][:, :, mid]
        ax = CM.Axis(fig[k, 1];
            title = "$(Int(E)) keV VMI  (HU, slice $mid)  [$(Int(hu_lo)), $(Int(hu_hi))]",
            aspect = CM.DataAspect())
        hm = CM.heatmap!(ax, hu_slice;
            colormap   = :grays,
            colorrange = (hu_lo, hu_hi),
            nan_color  = CM.RGBAf(0, 0, 0, 0))
        CM.Colorbar(fig[k, 2], hm; width = 12)
    end

    CM.save(joinpath(RESULTS_DIR, "mechlem_vmi_hu.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ Cell order:
# ╠═00010001-0000-4000-8000-000000000004
# ╠═14530566-ecaa-4345-9115-e75981f3837d
# ╠═0b629a9c-722c-47c5-900e-5e5b311a743b
# ╠═00010003-0000-4000-8000-000000000004
# ╠═00010004-0000-4000-8000-000000000004
# ╠═00010005-0000-4000-8000-000000000004
# ╠═00010006-0000-4000-8000-000000000004
# ╠═00010007-0000-4000-8000-000000000004
# ╠═00010008-0000-4000-8000-000000000004
# ╠═f12cb282-dfc1-4762-9851-7ef4225c8bd4
# ╠═00010009-0000-4000-8000-000000000004
# ╠═35b71532-be75-4fe9-ada8-9d6d4bf21f7f
# ╠═00010012-0000-4000-8000-000000000004
# ╠═00010013-0000-4000-8000-000000000004
# ╠═00010010-0000-4000-8000-000000000004
# ╠═00010011-0000-4000-8000-000000000004
# ╟─00020001-0000-4000-8000-000000000004
# ╟─00030001-0000-4000-8000-000000000004
# ╠═00030002-0000-4000-8000-000000000004
# ╠═00030003-0000-4000-8000-000000000004
# ╠═00030004-0000-4000-8000-000000000004
# ╟─00030005-0000-4000-8000-000000000004
# ╟─00040001-0000-4000-8000-000000000004
# ╠═00040002-0000-4000-8000-000000000004
# ╠═00040003-0000-4000-8000-000000000004
# ╠═00040004-0000-4000-8000-000000000004
# ╟─00040005-0000-4000-8000-000000000004
# ╟─00050001-0000-4000-8000-000000000004
# ╠═00050002-0000-4000-8000-000000000004
# ╠═00050003-0000-4000-8000-000000000004
# ╠═00050004-0000-4000-8000-000000000004
# ╠═00050005-0000-4000-8000-000000000004
# ╟─00070001-0000-4000-8000-000000000004
# ╠═00070002-0000-4000-8000-000000000004
# ╠═00070003-0000-4000-8000-000000000004
# ╟─00070004-0000-4000-8000-000000000004
# ╟─00080001-0000-4000-8000-000000000004
# ╟─00090040-0000-4000-8000-000000000004
# ╠═00090041-0000-4000-8000-000000000004
# ╠═00090042-0000-4000-8000-000000000004
# ╠═00090043-0000-4000-8000-000000000004
# ╠═00090044-0000-4000-8000-000000000004
# ╠═00090045-0000-4000-8000-000000000004
# ╠═00090046-0000-4000-8000-000000000004
# ╟─00090047-0000-4000-8000-000000000004
# ╟─00090001-0000-4000-8000-000000000004
# ╠═00090002-0000-4000-8000-000000000004
# ╟─00090003-0000-4000-8000-000000000004
# ╠═00090004-0000-4000-8000-000000000004
# ╠═00090005-0000-4000-8000-000000000004
# ╟─00090006-0000-4000-8000-000000000004
# ╠═00090007-0000-4000-8000-000000000004
# ╠═00090008-0000-4000-8000-000000000004
# ╠═00090030-0000-4000-8000-000000000004
# ╟─00090010-0000-4000-8000-000000000004
# ╠═00090011-0000-4000-8000-000000000004
# ╟─00090012-0000-4000-8000-000000000004
# ╠═00090013-0000-4000-8000-000000000004
# ╟─00090014-0000-4000-8000-000000000004
# ╠═00090015-0000-4000-8000-000000000004
# ╟─00090016-0000-4000-8000-000000000004
# ╠═00090017-0000-4000-8000-000000000004
# ╟─00090018-0000-4000-8000-000000000004
# ╠═00090019-0000-4000-8000-000000000004
# ╠═00090020-0000-4000-8000-000000000004
# ╟─00090021-0000-4000-8000-000000000004
# ╠═00090022-0000-4000-8000-000000000004
# ╟─00090023-0000-4000-8000-000000000004
# ╟─00100001-0000-4000-8000-000000000004
# ╠═00100002-0000-4000-8000-000000000004
# ╠═00100003-0000-4000-8000-000000000004
# ╟─00100004-0000-4000-8000-000000000004
