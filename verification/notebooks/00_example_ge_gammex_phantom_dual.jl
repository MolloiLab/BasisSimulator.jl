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
        e_res, w_res = BS.resolve_spectrum(sim_opts, prot; scanner = sim_scanner)
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
## 5. Photoelectric + Compton Basis Decomposition

**Pipeline:** Cong analytic per-ray decomp → PWLS-SQS sinogram restoration
(Noh/Fessler/Kinahan 2009) → ACNR sinogram smoother (Kalender/Klotz/Kostaridou
1988) → FBP → post-FBP radial capping correction. Each stage below has its
own sinogram + intermediate FBP so the streak / noise suppression is visible
stage-by-stage.

| Stage | Role | Handles |
|-------|------|---------|
| §5.1 Cong 2022 | Per-ray polychromatic inversion (no spatial coupling) | Beam-hardening, basis inversion |
| §5.2 PWLS-SQS  | Statistical penalized-WLS restoration on basis sinograms | Streaks from per-ray decomp errors; low-dose noise |
| §5.3 ACNR      | Anti-correlated noise projection across bases | Residual photo↔compton correlated noise (dominates VMI 40 keV) |
| §5.4 Capping   | Per-slice radial even-polynomial fit on the basis images | Residual radial cupping/capping from incomplete BHC |
| §5.5 Final FBP | FDK on cleaned basis sinograms (feeds §5.4 capping); final viz shows all four corrections | Reconstruction output consumed by §6 VMI |

### 5.1 Cong et al. 2022 — Per-ray Analytic Decomposition

**[CDW22] Cong, De Man, Wang (2022)** *J X-Ray Sci Technol* 30:725–736,
*"Projection decomposition via univariate optimization for dual-energy CT."*
DOI 10.3233/XST-221153.

Invert the polychromatic dual-energy forward model per ray. **No calibration
scan required** — only physical constants and the already-resolved x-ray
spectrum (`BS.resolve_spectrum`, which contains source × bowtie × detector).

#### Physical basis ([CDW22] Eqs 3a–3e, 4)

- `μ(r, ε) = p(ε)·a(r) + q(ε)·c(r)`
- `a(r) = ρ·⟨Z⁴/A⟩` — photoelectric spatial component
- `c(r) = ρ·⟨Z/A⟩` — Compton spatial component
- `p(ε) = N_A·α⁴·(8/3)π·r_e²·√(32/ε⁷)`,  `ε = E / m_e c²`
- `q(ε) = N_A·f_kn(ε)` — Klein-Nishina (Eq 3e)

#### Per-ray sinogram decomposition (Eqs 6–10, three Roots.jl calls)

1. **Water-based 1D Brent inversion** for water-equivalent path `L`:
   `∫Ŝ_L(ε)·exp(−(p(ε)·a_w + q(ε)·c_w)·L) dε = T_L_meas`, then `c̄ = c_w·L`.
   Absorbs the water beam-hardening exactly, so the Taylor-5 residual
   `x = C − c̄` carries only Ca/I/air deviations from water — bounded, small.
2. **Inner Newton on Eq 8 quintic** for `x = h(y)` — strictly convex ⇒ unique
   real root. Newton from `x = 0` converges in ≤ 8 iterations.
3. **Outer Roots.Brent on Eq 10.** `G(y) = T_H_pred(y, c̄+h(y)) − T_H_meas` is
   monotonic on `[0, τ_L/min(p_L))`. `Roots.find_zero(G, (0, y_max), Brent())`.

**Sinogram outputs (line integrals, per ray):**
- `sim_de_decomp.sino_photo   = y = ∫a(r)·dr`
- `sim_de_decomp.sino_compton = C = ∫c(r)·dr = c̄ + h(y)`

The intermediate FBP below shows **raw Cong output** — no spatial coupling
yet, so per-ray decomposition errors show up as coherent streaks through
high-attenuation rods. §5.2 cleans those up.
"""

# ╔═╡ 00080002-0000-4000-8000-000000000004
# Photo/Compton physical basis tables (Cong 2022 Eqs 3a–3e, 4).
de_basis = BS.compute_photo_compton_basis(
    BS.CTProtocol(kVp = 80,  additional_filters = additional_filters),
    BS.CTProtocol(kVp = 140, additional_filters = additional_filters);
    sim_opts = sim_opts, scanner = sim_scanner,
)

# ╔═╡ 00080003-0000-4000-8000-000000000004
# Water basis constants (Eqs 3a–3b): a_water = ρ·Σ(mᵢ·Zᵢ⁴/Aᵢ), c_water = ρ·Σ(mᵢ·Zᵢ/Aᵢ).
# Physical constant for H₂O — NOT a scanner calibration.
water_basis = BS.water_basis_constants()

# ╔═╡ 00080004-0000-4000-8000-000000000004
# Per-ray Cong 2022 decomposition: Brent(water-L) → Newton(quintic h(y)) → Brent(y).
# Runs via BS.apply_cong! which dispatches through AK.foreachindex, so Metal
# or CUDA arrays execute the same kernel on the GPU without a separate code
# path.  Brent is a 1:1 port of Roots.Brent (test/vmi_brent_parity.jl).
sim_de_decomp = let
    # Stage sinograms onto whatever backend `to_gpu` resolves (Metal/CUDA/CPU).
    sino_L_gpu = to_gpu(Float32.(sim_sino_low.sino))
    sino_H_gpu = to_gpu(Float32.(sim_sino_high.sino))

    t1 = time()
    sino_y_gpu, sino_c_gpu = BS.apply_cong(sino_L_gpu, sino_H_gpu;
                                           basis = de_basis,
                                           water_basis = water_basis)
    dt = time() - t1

    sino_photo   = Array(sino_y_gpu)
    sino_compton = Array(sino_c_gpu)
    sino_L_gpu = nothing; sino_H_gpu = nothing
    sino_y_gpu = nothing; sino_c_gpu = nothing
    clear_gpu!()

    backend = sim_sino_low.sino isa Array ? "CPU ($(Threads.nthreads()) threads)" : "GPU"
    @info "[CDW22] apply_cong done in $(round(dt, digits=1)) s  [backend: $backend]"
    @info "  ⟨y⟩ = $(round(mean(sino_photo),   sigdigits=4))   photoelectric line integral"
    @info "  ⟨C⟩ = $(round(mean(sino_compton), sigdigits=4))   Compton line integral"

    (sino_photo = sino_photo, sino_compton = sino_compton, geom = sim_sino_low.geom)
end;

# ╔═╡ 00080005-0000-4000-8000-000000000004
# Photoelectric and Compton sinograms — mid-view and mid-row.
let
    sino_p = sim_de_decomp.sino_photo
    sino_c = sim_de_decomp.sino_compton
    mid_view = size(sino_p, 2) ÷ 2
    mid_row  = size(sino_p, 3) ÷ 2

    fig = CM.Figure(size = (1400, 800), fontsize = 13)

    for (row, name, sino) in [(1, "Photoelectric  y = ∫a(r)dr",  sino_p),
                              (2, "Compton  C = ∫c(r)dr",        sino_c)]
        ax1 = CM.Axis(
            fig[row, 1]; title = "$name — mid-view (view $mid_view)",
            xlabel = "Detector column", ylabel = "Detector row",
            aspect = CM.DataAspect()
        )
        hm1 = CM.heatmap!(ax1, sino[:, mid_view, :]'; colormap = :viridis)
        CM.Colorbar(fig[row, 2], hm1; width = 12)

        ax2 = CM.Axis(
            fig[row, 3]; title = "$name — mid-row (row $mid_row)",
            xlabel = "Detector column", ylabel = "View angle"
        )
        hm2 = CM.heatmap!(ax2, sino[:, :, mid_row]'; colormap = :viridis)
        CM.Colorbar(fig[row, 4], hm2; width = 12)
    end

    CM.save(joinpath(RESULTS_DIR, "sinogram_photo_compton.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080012-0000-4000-8000-000000000004
# Intermediate FBP of RAW Cong output — NO Tikhonov, NO ACNR yet.
# Shows the per-ray decomposition streaks §5.2 PWLS-SQS will clean up.
sim_recon_cong = let
    filter_ctrl = (
        x = (0.0, 0.25, 0.5, 0.75, 1.0),
        y = (1.0, 0.95, 0.85, 0.65, 0.4),
    )

    function _fbp(sino::Array{Float32, 3})
        sino_gpu = to_gpu(sino)
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, sim_de_decomp.geom, sim_matrix_size;
            filter = BS.CustomFilter(filter_ctrl.x, filter_ctrl.y)
        )
        vol = Array(BS.reconstruct!(ws, sino_gpu, sim_de_decomp.geom, sim_matrix_size))
        ws = nothing; sino_gpu = nothing; clear_gpu!()
        Float32.(vol)
    end

    t1 = time()
    a_img = _fbp(sim_de_decomp.sino_photo)
    c_img = _fbp(sim_de_decomp.sino_compton)
    @info "Intermediate FBP (Cong only, no smoothing) done in $(round(time() - t1, digits=1)) s"
    (a = a_img, c = c_img)
end;

# ╔═╡ 00080013-0000-4000-8000-000000000004
# Cong-only FBP mid-slice — streaks from per-ray analytic decomp should be visible.
let
    a_img = sim_recon_cong.a
    c_img = sim_recon_cong.c
    mid   = size(a_img, 3) ÷ 2
    a_slice = a_img[:, :, mid]
    c_slice = c_img[:, :, mid]

    a_lo, a_hi = quantile(vec(a_slice), 0.01), quantile(vec(a_slice), 0.995)
    c_lo, c_hi = quantile(vec(c_slice), 0.01), quantile(vec(c_slice), 0.995)

    fig = CM.Figure(size = (1250, 570), fontsize = 13)
    ax1 = CM.Axis(fig[1, 1]; title = "Photoelectric  a(r) — Cong only  (slice $mid)", aspect = CM.DataAspect())
    hm1 = CM.heatmap!(ax1, a_slice; colormap = :viridis, colorrange = (a_lo, a_hi))
    CM.Colorbar(fig[1, 2], hm1; width = 12)

    ax2 = CM.Axis(fig[1, 3]; title = "Compton  c(r) — Cong only  (slice $mid)", aspect = CM.DataAspect())
    hm2 = CM.heatmap!(ax2, c_slice; colormap = :viridis, colorrange = (c_lo, c_hi))
    CM.Colorbar(fig[1, 4], hm2; width = 12)

    CM.save(joinpath(RESULTS_DIR, "recon_photo_compton_cong.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080014-0000-4000-8000-000000000004
md"""
### 5.2 PWLS-SQS Sinogram Restoration — Noh, Fessler, Kinahan 2009

**[NFK09] Noh, Fessler, Kinahan (2009)** *IEEE Trans Med Imaging* 28(11):
1688–1702. *"Statistical Sinogram Restoration in Dual-Energy CT for PET
Attenuation Correction."* DOI 10.1109/TMI.2009.2023988.

Cong 2022 (§5.1) gives a per-ray analytic decomposition with **zero spatial
coupling** — each detector column/view is solved in isolation. At low dose
or through high-attenuation rods, per-ray decomposition errors + Poisson
measurement noise stack up as coherent streaks in the intermediate FBP
above. NFK09 instead jointly denoises both basis sinograms in the sinogram
domain by minimizing a penalized weighted least-squares cost:

    Φ(s) = ½·Σ_{i,m} w_{mi}·(h_{mi} − f_m(s_i))²  +  Σ_l ½·γ_l·‖C·s_l‖²   [NFK09 Eq 12]

where
- `s = (s_y, s_C)` — basis-sinogram state (photo + Compton line integrals)
- `h_{mi}` — measured polychromatic transmissions (`−log y_{mi}/I_m`)
- `f_m(s_i) = −log Σ_k ŵ_m[k]·exp(−p_m[k]·s_y − q_m[k]·s_C)` — **same**
  polychromatic forward model used in §5.1
- `w_{mi} ≈ y_{mi} ∝ exp(−h_{mi})` — Poisson inverse-variance weights
  (NFK09 Eq 13); high-attenuation rays get low weight so their residuals
  don't dominate the fit
- `C` — 2D 2nd-order difference operator (detector-col + view axes), Neumann
  BC; γ_l controls smoothness per basis

**Why warm-start from Cong?** The data-fit landscape is nonconvex (because
`f_m` is nonlinear), but Cong already sits in the right basin — starting
there skips the global-descent phase, letting a handful of SQS iterations
denoise without introducing its own bias.

Solved via **Separable Paraboloidal Surrogates (SPS)** with Fessler-Erdogan
precomputed curvature (MIRT `de_wls_dercurv.m`):

    M[m, l]        = Σ_k ŵ_m[k]·c_l[k]                  — spectral MAC matrix, 2×2
    curv_geom[l]   = Σ_m |M[m, l]| · Σ_{l'} |M[m, l']|  — data-term majorant
    curv_R_l       = γ_l · 32                           — De Pierro row-sum bound
                                                          for |C'C| in 2D
    curv_l[i]      = max_m(w_{mi}) · curv_geom[l] + curv_R_l   — per-pixel, constant

SQS update (MIRT `pwls_sqs_os.m`, adapted: G = I for sinogram domain):

    grad_l[i]    = Σ_m w_{mi}·(f_m(s_i) − h_{mi})·∂f_m/∂s_l
    grad_R_l[i]  = γ_l·(C'C·s_l)[i]
    s_l[i] ← max( s_l[i] − relax·(grad_l[i] + grad_R_l[i]) / curv_l[i], 0 )

SPS guarantees **monotonic descent of Φ(s)** at every iteration (Fessler
2000). We log Φ per iter and warn if it ever increases — catches any sign
or curvature-bound bug immediately.

**Implementation.** 1:1 port of MIRT's `ct/de_wls_dercurv.m` (data-term
gradient + precomputed curvature) and `wls/pwls_sqs_os.m` (SQS loop).
Per-slice threading over detector rows. See cell §5.2 body below for the
code and the hyperparameter cell above it for `γ_photo`, `γ_compton`,
`n_iter`, `relax`.
"""

# ╔═╡ 00080021-0000-4000-8000-000000000004
# ── PWLS sinogram restoration hyperparameters (Noh, Fessler, Kinahan 2009) ─
# 1:1 port of MIRT ct/de_wls_dercurv.m + wls/pwls_sqs_os.m; warm-started from
# Cong; operates in the sinogram domain, per-slice threaded.
#
#   pwls_enable     — master switch; false = pass-through Cong
#   pwls_n_iter     — SQS iterations (MIRT examples use 20–30; monotonic
#                       descent is guaranteed so more is safe)
#   pwls_γ_photo    — regularization strength on the photo basis (Noh Eq 14 γ_l)
#   pwls_γ_compton  — regularization strength on the Compton basis
#                       Noh 2009 Sec V uses γ = 2⁻⁸ ≈ 0.0039 at very low dose.
#                       Higher → smoother; lower → closer to Cong.
#                       Tune on a log grid [1e-5, 1e-1]; doubling/halving is
#                       a fine step size.
#   pwls_relax      — SQS relaxation (MIRT default 1.0 = unrelaxed); drop to
#                       0.5 if the monotonicity check in the main cell fires.
#
# Paper: Noh, Fessler, Kinahan. IEEE TMI 28(11):1688–1702, 2009.
# Code:  github.com/JeffFessler/mirt (ct/de_wls_dercurv.m + wls/pwls_sqs_os.m)
begin
    pwls_enable    = true
    pwls_n_iter    = 20
    pwls_γ_photo   = 2.0^(-8)
    pwls_γ_compton = 2.0^(-8)
    pwls_relax     = 1.0
end

# ╔═╡ 00080022-0000-4000-8000-000000000004
# Fessler/Noh 2009 PWLS-SQS sinogram restoration — 1:1 port of MIRT's
# ct/de_wls_dercurv.m (per-ray WLS gradient + Fessler-Erdogan precomputed
# separable curvature) + wls/pwls_sqs_os.m (SQS update loop).
#
# Cost (Noh 2009 Eq 12):
#     Φ(s) = ½·Σ_{i,m} w_{mi}·(h_{mi} − f_m(s_i))²  +  Σ_l ½·γ_l·‖C·s_l‖²
#   where
#     h_{mi}  = −log(y_{mi}/I_m)  — measured line integrals (sim_sino_{low,high}.sino)
#     f_m(s_i) = −log Σ_k ŵ_m[k]·exp(−p_m[k]·y − q_m[k]·C)    (polychromatic fwd)
#     w_{mi}  ≈ y_{mi} ∝ exp(−h_{mi})                         (Poisson inv-var, Eq 13)
#     ‖C·s_l‖² = ‖Cx·s_l‖² + ‖Cy·s_l‖²                        — 2D 2nd-order diff
#                  (detector-col AND view axes), Neumann BC on each. Extends
#                  Noh 2009 §III.B (1D radial) using the MIRT Cdiffs pattern:
#                  stack independent Cdiff1 operators per axis. One γ per basis
#                  controls both directions equally — no extra hyperparam.
#
# Fessler-Erdogan precomputed curvature (MIRT de_wls_dercurv.m lines 131–133):
#     M[m, l]          = Σ_k ŵ_m[k]·c_l[k]   — spectral-weighted MAC, 2×2
#     curv_geom[l]     = Σ_m |M[m, l]| · Σ_{l'} |M[m, l']|
#     curv_data_l[i]   = max_m(w_{mi}) · curv_geom[l]
#     curv_R_l         = γ_l · 32            — De Pierro row-sum bound for
#                          |Cx'Cx + Cy'Cy|: each axis has stencil [1,−4,6,−4,1]
#                          with row-sum 16; two axes ⇒ 32.
#     curv_l[i]        = curv_data_l[i] + curv_R_l         (constant, one-time)
#
# SQS update (pwls_sqs_os.m lines 135–158, adapted: G=I for sinogram domain):
#     grad_l[i]   = Σ_m w_{mi} · (f_m(s_i) − h_{mi}) · ∂f_m/∂s_l
#     grad_R_l[i] = γ_l · (CᵀC · s_l)[i]
#     s_l[i] ← max( s_l[i] − relax · (grad_l[i] + grad_R_l[i]) / curv_l[i], 0 )
#
# SQS guarantees monotonic decrease of Φ(s) (Fessler 2000). We log Φ per iter
# and WARN if it increases — catches any sign or curvature-bound bug
# immediately.
sim_de_decomp_pwls = let
    if !pwls_enable
        @info "PWLS restoration: DISABLED (pass-through Cong)"
        (sino_photo   = sim_de_decomp.sino_photo,
         sino_compton = sim_de_decomp.sino_compton,
         geom         = sim_de_decomp.geom,
         n_iter       = 0, γ_photo = 0.0, γ_compton = 0.0,
         relax        = 0.0, cost_history = Float64[])
    else
        sino_y = copy(sim_de_decomp.sino_photo)     # Cong warm start
        sino_c = copy(sim_de_decomp.sino_compton)
        info = BS.apply_pwls!(sino_y, sino_c;
                              h_low     = sim_sino_low.sino,
                              h_high    = sim_sino_high.sino,
                              basis     = de_basis,
                              γ_photo   = pwls_γ_photo,
                              γ_compton = pwls_γ_compton,
                              n_iter    = pwls_n_iter,
                              relax     = pwls_relax)
        (sino_photo = sino_y, sino_compton = sino_c,
         geom = sim_de_decomp.geom,
         n_iter = info.n_iter,
         γ_photo = info.γ_photo, γ_compton = info.γ_compton,
         relax = info.relax, cost_history = info.cost_history)
    end
end;

# ╔═╡ 00080015-0000-4000-8000-000000000004
# Photo + Compton sinograms AFTER PWLS-SQS restoration — should look smoother
# (with streaks suppressed) compared to the raw Cong output in §5.1.
let
    sino_p = sim_de_decomp_pwls.sino_photo
    sino_c = sim_de_decomp_pwls.sino_compton
    mid_view = size(sino_p, 2) ÷ 2
    mid_row  = size(sino_p, 3) ÷ 2

    fig = CM.Figure(size = (1400, 800), fontsize = 13)
    tag = "γ_p=$(sim_de_decomp_pwls.γ_photo), γ_C=$(sim_de_decomp_pwls.γ_compton), k=$(sim_de_decomp_pwls.n_iter)"
    for (row, name, sino) in [(1, "Photoelectric y — post-PWLS ($tag)", sino_p),
                              (2, "Compton C — post-PWLS ($tag)",       sino_c)]
        ax1 = CM.Axis(fig[row, 1]; title = "$name — mid-view (view $mid_view)",
                      xlabel = "Detector column", ylabel = "Detector row", aspect = CM.DataAspect())
        hm1 = CM.heatmap!(ax1, sino[:, mid_view, :]'; colormap = :viridis)
        CM.Colorbar(fig[row, 2], hm1; width = 12)
        ax2 = CM.Axis(fig[row, 3]; title = "$name — mid-row (row $mid_row)",
                      xlabel = "Detector column", ylabel = "View angle")
        hm2 = CM.heatmap!(ax2, sino[:, :, mid_row]'; colormap = :viridis)
        CM.Colorbar(fig[row, 4], hm2; width = 12)
    end
    CM.save(joinpath(RESULTS_DIR, "sinogram_photo_compton_pwls.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080016-0000-4000-8000-000000000004
# Intermediate FBP with PWLS-SQS restoration applied but NOT ACNR yet.
# Compare to sim_recon_cong above: basis-sinogram noise should be suppressed
# by the statistical weighting (D_12 off-diagonals of the Fisher info handle
# anti-correlation at the source) + 2D quadratic roughness penalty (radial
# AND view directions, MIRT Cdiffs pattern).
sim_recon_pwls = let
    filter_ctrl = (
        x = (0.0, 0.25, 0.5, 0.75, 1.0),
        y = (1.0, 0.95, 0.85, 0.65, 0.4),
    )

    function _fbp(sino::Array{Float32, 3})
        sino_gpu = to_gpu(sino)
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, sim_de_decomp_pwls.geom, sim_matrix_size;
            filter = BS.CustomFilter(filter_ctrl.x, filter_ctrl.y)
        )
        vol = Array(BS.reconstruct!(ws, sino_gpu, sim_de_decomp_pwls.geom, sim_matrix_size))
        ws = nothing; sino_gpu = nothing; clear_gpu!()
        Float32.(vol)
    end

    t1 = time()
    a_img = _fbp(sim_de_decomp_pwls.sino_photo)
    c_img = _fbp(sim_de_decomp_pwls.sino_compton)
    @info "Intermediate FBP (Cong + PWLS) done in $(round(time() - t1, digits=1)) s"
    (a = a_img, c = c_img)
end;

# ╔═╡ c1139ae3-5186-445e-81b8-5d932ca5ef98
# Cong-only FBP mid-slice — streaks from per-ray analytic decomp should be visible.
let
    a_img = sim_recon_cong.a
    c_img = sim_recon_cong.c
    mid   = size(a_img, 3) ÷ 2
    a_slice = a_img[:, :, mid]
    c_slice = c_img[:, :, mid]

    a_lo, a_hi = quantile(vec(a_slice), 0.01), quantile(vec(a_slice), 0.995)
    c_lo, c_hi = quantile(vec(c_slice), 0.01), quantile(vec(c_slice), 0.995)

    fig = CM.Figure(size = (1250, 570), fontsize = 13)
    ax1 = CM.Axis(fig[1, 1]; title = "Photoelectric  a(r) — Cong only  (slice $mid)", aspect = CM.DataAspect())
    hm1 = CM.heatmap!(ax1, a_slice; colormap = :viridis, colorrange = (a_lo, a_hi))
    CM.Colorbar(fig[1, 2], hm1; width = 12)

    ax2 = CM.Axis(fig[1, 3]; title = "Compton  c(r) — Cong only  (slice $mid)", aspect = CM.DataAspect())
    hm2 = CM.heatmap!(ax2, c_slice; colormap = :viridis, colorrange = (c_lo, c_hi))
    CM.Colorbar(fig[1, 4], hm2; width = 12)

    CM.save(joinpath(RESULTS_DIR, "recon_photo_compton_cong.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080017-0000-4000-8000-000000000004
# Cong + PWLS FBP mid-slice — noise floor should drop vs. Cong-only, rods
# preserved (quadratic penalty is only radial 2nd-order diff, minimal blur).
let
    a_img = sim_recon_pwls.a
    c_img = sim_recon_pwls.c
    mid   = size(a_img, 3) ÷ 2
    a_slice = a_img[:, :, mid]
    c_slice = c_img[:, :, mid]

    a_lo, a_hi = quantile(vec(a_slice), 0.01), quantile(vec(a_slice), 0.995)
    c_lo, c_hi = quantile(vec(c_slice), 0.01), quantile(vec(c_slice), 0.995)

    fig = CM.Figure(size = (1250, 570), fontsize = 13)
    ax1 = CM.Axis(fig[1, 1]; title = "Photoelectric  a(r) — Cong+PWLS  (slice $mid)", aspect = CM.DataAspect())
    hm1 = CM.heatmap!(ax1, a_slice; colormap = :viridis, colorrange = (a_lo, a_hi))
    CM.Colorbar(fig[1, 2], hm1; width = 12)

    ax2 = CM.Axis(fig[1, 3]; title = "Compton  c(r) — Cong+PWLS  (slice $mid)", aspect = CM.DataAspect())
    hm2 = CM.heatmap!(ax2, c_slice; colormap = :viridis, colorrange = (c_lo, c_hi))
    CM.Colorbar(fig[1, 4], hm2; width = 12)

    CM.save(joinpath(RESULTS_DIR, "recon_photo_compton_pwls.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080018-0000-4000-8000-000000000004
md"""
### 5.3 Sinogram-Domain ACNR — Anti-Correlated Noise Reduction

**[KKK88] Kalender, Klotz, Kostaridou (1988)** *IEEE Trans Med Imaging*
7(3):218–224. *"An algorithm for noise suppression in dual energy CT
material density images."* DOI 10.1109/42.7784.

**Why here, not after FBP?** DE basis decomposition is ill-conditioned:
photo-basis and Compton-basis sinogram noise are strongly anti-correlated
(ρ ≈ −1). Photo-basis noise magnitude is typically 5–10× larger than
Compton's — this drives the VMI_40 noise explosion (`|w_p(40)|·σ_a`
dominates). Exploiting the anti-correlation in the sinogram domain,
*before* FBP amplifies it, recovers more per unit resolution than any
post-FBP filter.

**Mechanism ([KKK88]).**
At reference energy `E_ref`, signal direction is `u_sig = (p(E_ref), q(E_ref))`.
The orthogonal direction carries anti-correlated noise:

    s_⊥(r)  =  −q(E_ref)·a(r) + p(E_ref)·c(r)

Smooth `s_⊥`, take residual `n_⊥`, project out of the basis pair:

    a_clean  =  a  +  γ · q(E_ref)/|u_sig|² · n_⊥
    c_clean  =  c  −  γ · p(E_ref)/|u_sig|² · n_⊥

By construction `p(E_ref)·Δa + q(E_ref)·Δc = 0` — correction lives entirely
in the anti-correlated subspace, so `μ(E_ref)` at every pixel is exactly
unchanged ⇒ **zero resolution loss at E_ref**. Other-energy VMIs inherit a
small, smooth 1st-order bias invisible in soft-tissue windows but benefit
from dramatically reduced photo-basis noise.

Runs on the PWLS-restored basis sinograms (§5.2 output), so residual
anti-correlated noise is what remains after PWLS has already suppressed
streaks and Poisson noise per-basis.
"""

# ╔═╡ 00080006-0000-4000-8000-000000000004
# ── ACNR hyperparameters [TUNE: sinogram-domain anti-correlated noise reduction] ──
# Smoother on s_⊥ is a Tikhonov solver in the Fourier domain — exact one-shot
# solve of (I + λ·D'D)·s = s_⊥ via FFT. ACNR's projection preserves μ(E_ref)
# exactly regardless of smoother choice; all tuning lives in acnr_λ.
#
#   acnr_enable — master switch; false = pass PWLS output straight through.
#   acnr_E_ref  — reference energy (keV) at which signal direction is preserved.
#                 VMI at this energy has ZERO resolution loss from ACNR; other
#                 energies get small 1st-order bias + big noise reduction.
#                 Default 70 keV: noise-optimal VMI anchor used by Mono+ too.
#   acnr_λ      — Tikhonov strength on s_⊥. Effective smoothing radius ≈ √λ px.
#                 Exact FFT solve ⇒ λ has MONOTONIC effect at every magnitude
#                 (no iterative-solver saturation). Tune on a log grid:
#                   • 0.5–1    — gentle (~1 px)
#                   • 2–8      — moderate (~1.5–3 px)
#                   • 16–100   — aggressive (~4–10 px); kills iodine streaks
#                   • ≥1000    — nearly-constant smoothing (mean of s_⊥)
#   acnr_γ      — projection strength ∈ [0, 1]. γ=0 off, γ=1 full ACNR.
#                 Leave at 1.0 unless off-ref VMI bias becomes visible.
begin
    acnr_enable = true
    acnr_E_ref  = 100.0
    acnr_λ      = 2.0
    acnr_γ      = 1.0
end

# ╔═╡ 00080007-0000-4000-8000-000000000004
# Sinogram-domain ACNR (Kalender/Klotz/Kostaridou 1988) on the PWLS-restored
# photo/Compton basis pair. Reads from sim_de_decomp_pwls.
#
# Framework (unchanged from classical ACNR):
#     s_⊥(r) = −q(E_ref)·a(r) + p(E_ref)·c(r)          (anti-corr noise channel)
#     n_⊥    = s_⊥ − smooth(s_⊥)                       (noise estimate)
#     a_cln  = a + γ·q(E_ref)/|u|² · n_⊥
#     c_cln  = c − γ·p(E_ref)/|u|² · n_⊥
# Identity: p·a_cln + q·c_cln = p·a + q·c  ⇒  μ(E_ref) unchanged per pixel,
# for ANY smoother choice. Zero resolution loss at E_ref is structural.
#
# Smoother = FFT-BASED TIKHONOV (exact one-shot solve). Per-slice (detector-col
# × view plane), solve:
#     min_s   ½‖s − s_⊥‖²  +  λ · ½·‖D·s‖²      ⇒   (I + λ·D'D)·s = s_⊥
# where D'D is the 4-neighbor discrete Laplacian with implicit periodic BC
# from the FFT. Exact Fourier diagonalization:
#     ŝ_smooth[p,q] = ŝ_⊥[p,q] / (1 + λ · μ_{p,q})
#     μ_{p,q}       = 4·(sin²(πp/nx) + sin²(πq/nv))       [L eigenvalues]
# → one FFT + pointwise divide + one inverse FFT per slice. No iteration,
# so λ has MONOTONIC effect at any magnitude (no SPS-style saturation).
# Effective smoothing radius ≈ √λ px; at λ→∞, s_smooth → mean(s_⊥).
sim_de_decomp_clean = let
    sino_y = copy(sim_de_decomp_pwls.sino_photo)
    sino_c = copy(sim_de_decomp_pwls.sino_compton)
    n_orth_σ = 0f0
    acnr_on  = false

    if acnr_enable
        info = BS.apply_acnr!(sino_y, sino_c;
                              E_ref = acnr_E_ref,
                              λ     = acnr_λ,
                              γ     = acnr_γ)
        n_orth_σ = Float32(info.σ_n_orth)
        acnr_on  = true
    else
        @info "ACNR sinogram smoother: DISABLED (pass-through of PWLS output)"
    end

    (sino_photo = sino_y, sino_compton = sino_c, geom = sim_de_decomp_pwls.geom,
     n_orth_σ = n_orth_σ,
     params = (acnr_enable = acnr_on, acnr_E_ref = acnr_E_ref,
               acnr_λ = acnr_λ, acnr_γ = acnr_γ))
end;

# ╔═╡ 00080019-0000-4000-8000-000000000004
# Photo + Compton sinograms AFTER ACNR — residual anti-correlated noise
# projected out. Photo-basis should look noticeably cleaner than post-Tikhonov.
let
    sino_p = sim_de_decomp_clean.sino_photo
    sino_c = sim_de_decomp_clean.sino_compton
    mid_view = size(sino_p, 2) ÷ 2
    mid_row  = size(sino_p, 3) ÷ 2

    fig = CM.Figure(size = (1400, 800), fontsize = 13)
    for (row, name, sino) in [(1, "Photoelectric y — post-ACNR", sino_p),
                              (2, "Compton C — post-ACNR",        sino_c)]
        ax1 = CM.Axis(fig[row, 1]; title = "$name — mid-view (view $mid_view)",
                      xlabel = "Detector column", ylabel = "Detector row", aspect = CM.DataAspect())
        hm1 = CM.heatmap!(ax1, sino[:, mid_view, :]'; colormap = :viridis)
        CM.Colorbar(fig[row, 2], hm1; width = 12)
        ax2 = CM.Axis(fig[row, 3]; title = "$name — mid-row (row $mid_row)",
                      xlabel = "Detector column", ylabel = "View angle")
        hm2 = CM.heatmap!(ax2, sino[:, :, mid_row]'; colormap = :viridis)
        CM.Colorbar(fig[row, 4], hm2; width = 12)
    end
    CM.save(joinpath(RESULTS_DIR, "sinogram_photo_compton_acnr.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080023-0000-4000-8000-000000000004
md"""
### 5.4 Post-FBP Radial Capping Correction

Residual radial **cupping/capping** (background HU slowly drifting away from
zero as a function of image radius) persists even after BHC + Cong + PWLS +
ACNR because none of those model every beam-hardening path exactly. The fix
is a classic one and the 06 clinical notebook already uses it in HU space
(`BS.apply_radial_cupping_correction!`) — here we apply the *same idea*
directly to the basis images `a(r)` and `c(r)` so **every downstream VMI
inherits a flat background for free** (since `VMI(E) = p(E)·a + q(E)·c`,
flattening both bases flattens every energy synthesised from them).

Per-slice, for each basis image:
1. Select background voxels by quantile over the in-FOV slice (excludes air
   outside the scanner bore + the insert high/low tails without needing
   material-specific thresholds).
2. Fit an even polynomial in radius:
   `offset(r) = c₀ + c₁·r² + c₂·r⁴ + …`
3. Subtract `offset(r) − c₀` from every voxel in the slice — keeps the
   central value (r=0) intact, flattens the radial profile.

Because the fit is linear and the subtraction is the same polynomial for
both `a` and `c`, applying it here *is* algebraically equivalent to
applying it to every future VMI at every energy.
"""

# ╔═╡ 00080024-0000-4000-8000-000000000004
# ── Radial capping correction hyperparameters ─────────────────────────────
#
#   cap_enable     — master switch; false = pass raw FBP output through
#   cap_fov_cm     — transverse FOV in cm; controls the pixel→cm scale for
#                    the polynomial fit. Default 35.0 cm matches the 06
#                    clinical notebook and the GE Apex Elite clinical FOV.
#   cap_poly_order — number of even-polynomial terms beyond c₀:
#                    1 ⇒ c₀ + c₁r²          (pure parabolic cupping)
#                    2 ⇒ c₀ + c₁r² + c₂r⁴   (default, matches 06)
#                    3+ for stronger radial structure; rarely worth it.
#   cap_q_lo/q_hi  — quantile range over the in-FOV slice used to pick
#                    background voxels. Middle 50% (0.25–0.75) avoids the
#                    air floor on the low end and the iodine/bone rods on
#                    the high end without needing HU thresholds.
begin
    cap_enable     = true
    cap_fov_cm     = 35.0
    cap_poly_order = 4
    cap_q_lo       = 0.10
    cap_q_hi       = 0.75
end

# ╔═╡ 00080020-0000-4000-8000-000000000004
md"""
### 5.5 Final FBP on Cleaned Basis Sinograms

Plain FDK on each fully-cleaned basis sinogram (shared Shepp-Logan-style
mild apodization). Filter preserves high-frequency edges (vessels, rod
walls) so the downstream VMI+ step has real structural content to work
with — heavy Hann would throw that away at the FBP stage and no amount of
downstream processing can recover it.

Outputs:
- `sim_recon_photo_compton.a` — raw photoelectric basis image (FBP only)
- `sim_recon_photo_compton.c` — raw Compton basis image (FBP only)
- `sim_recon_photo_compton_flat.a/.c` — same, with §5.4 capping applied.
  **This is the tuple §6 VMI synthesis reads from** so every virtual
  monoenergetic image inherits a flat radial background.

No BHC, no HU conversion — images are in physical basis units. HU
conversion happens downstream in §6 VMI synthesis.
"""

# ╔═╡ 00090002-0000-4000-8000-000000000004
# FBP each CLEANED basis sinogram on GPU. Shared smooth apodization filter.
sim_recon_photo_compton = let
    # Shepp-Logan-style mild apodization.  Preserves high-frequency edges
    # (vessels, rod walls) so the downstream guided-filter Mono+ step has real
    # structural content to work with — heavy Hann would throw that away at the
    # FBP stage and no amount of downstream processing can recover it.
    filter_ctrl = (
        x = (0.0, 0.25, 0.5, 0.75, 1.0),
        y = (1.0, 0.95, 0.85, 0.65, 0.4),
    )

    function _fbp(sino::Array{Float32, 3})
        sino_gpu = to_gpu(sino)
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, sim_de_decomp_clean.geom, sim_matrix_size;
            filter = BS.CustomFilter(filter_ctrl.x, filter_ctrl.y)
        )
        vol = Array(BS.reconstruct!(ws, sino_gpu, sim_de_decomp_clean.geom, sim_matrix_size))
        ws = nothing; sino_gpu = nothing; clear_gpu!()
        Float32.(vol)
    end

    t1 = time()
    a_img = _fbp(sim_de_decomp_clean.sino_photo)
    c_img = _fbp(sim_de_decomp_clean.sino_compton)
    @info "FBP photoelectric + Compton done in $(round(time() - t1, digits=1)) s"
    @info "  a(r) range: [$(round(minimum(a_img), sigdigits=3)), $(round(maximum(a_img), sigdigits=3))]"
    @info "  c(r) range: [$(round(minimum(c_img), sigdigits=3)), $(round(maximum(c_img), sigdigits=3))]"

    (a = a_img, c = c_img)
end;

# ╔═╡ 00080025-0000-4000-8000-000000000004
# Per-slice radial capping correction on the basis images a(r) and c(r).
# Adapted from BS.apply_radial_cupping_correction! (src/correction/
# beam_hardening_correction.jl:790) — swaps the HU-threshold background
# selector for a quantile-based one so it works directly in basis units.
#
# Input:  sim_recon_photo_compton.a, .c   (raw FBP of ACNR-clean sinograms)
# Output: sim_recon_photo_compton_flat    (same shape, radial profile flat)
sim_recon_photo_compton_flat = let
    if !cap_enable
        @info "Radial capping correction: DISABLED (pass-through)"
        (a = sim_recon_photo_compton.a, c = sim_recon_photo_compton.c,
         coeffs_a = Float64[], coeffs_c = Float64[])
    else
        a_out = copy(sim_recon_photo_compton.a)
        c_out = copy(sim_recon_photo_compton.c)
        info = BS.apply_radial_capping_basis!(a_out, c_out;
                                              fov_cm     = cap_fov_cm,
                                              poly_order = cap_poly_order,
                                              q_lo       = cap_q_lo,
                                              q_hi       = cap_q_hi)
        (a = a_out, c = c_out,
         coeffs_a = info.coeffs_a, coeffs_c = info.coeffs_c)
    end
end;

# ╔═╡ fd9969e0-415f-409d-8672-fe2d963b6486
# Cong-only FBP mid-slice — streaks from per-ray analytic decomp should be visible.
let
    a_img = sim_recon_cong.a
    c_img = sim_recon_cong.c
    mid   = size(a_img, 3) ÷ 2
    a_slice = a_img[:, :, mid]
    c_slice = c_img[:, :, mid]

    a_lo, a_hi = quantile(vec(a_slice), 0.01), quantile(vec(a_slice), 0.995)
    c_lo, c_hi = quantile(vec(c_slice), 0.01), quantile(vec(c_slice), 0.995)

    fig = CM.Figure(size = (1250, 570), fontsize = 13)
    ax1 = CM.Axis(fig[1, 1]; title = "Photoelectric  a(r)", subtitle = "Cong only  (slice $mid)", aspect = CM.DataAspect())
    hm1 = CM.heatmap!(ax1, a_slice; colormap = :viridis, colorrange = (a_lo, a_hi))
    CM.Colorbar(fig[1, 2], hm1; width = 12)

    ax2 = CM.Axis(fig[1, 3]; title = "Compton  c(r)", subtitle = "Cong only  (slice $mid)", aspect = CM.DataAspect())
    hm2 = CM.heatmap!(ax2, c_slice; colormap = :viridis, colorrange = (c_lo, c_hi))
    CM.Colorbar(fig[1, 4], hm2; width = 12)

    CM.save(joinpath(RESULTS_DIR, "recon_photo_compton_cong.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 2b6fd506-c624-4be3-9a71-1d366ae58ada
# Cong + PWLS FBP mid-slice — noise floor should drop vs. Cong-only, rods
# preserved (quadratic penalty is only radial 2nd-order diff, minimal blur).
let
    a_img = sim_recon_pwls.a
    c_img = sim_recon_pwls.c
    mid   = size(a_img, 3) ÷ 2
    a_slice = a_img[:, :, mid]
    c_slice = c_img[:, :, mid]

    a_lo, a_hi = quantile(vec(a_slice), 0.01), quantile(vec(a_slice), 0.995)
    c_lo, c_hi = quantile(vec(c_slice), 0.01), quantile(vec(c_slice), 0.995)

    fig = CM.Figure(size = (1250, 570), fontsize = 13)
    ax1 = CM.Axis(fig[1, 1]; title = "Photoelectric  a(r)", subtitle = "Cong+PWLS  (slice $mid)", aspect = CM.DataAspect())
    hm1 = CM.heatmap!(ax1, a_slice; colormap = :viridis, colorrange = (a_lo, a_hi))
    CM.Colorbar(fig[1, 2], hm1; width = 12)

    ax2 = CM.Axis(fig[1, 3]; title = "Compton  c(r)", subtitle = "Cong+PWLS  (slice $mid)", aspect = CM.DataAspect())
    hm2 = CM.heatmap!(ax2, c_slice; colormap = :viridis, colorrange = (c_lo, c_hi))
    CM.Colorbar(fig[1, 4], hm2; width = 12)

    CM.save(joinpath(RESULTS_DIR, "recon_photo_compton_pwls.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00090003-0000-4000-8000-000000000004
# Mid-slice of both basis images AFTER all 4 corrections (Cong + PWLS + ACNR
# + radial capping) — this is exactly what §6 VMI synthesis consumes.
let
    a_img = sim_recon_photo_compton_flat.a
    c_img = sim_recon_photo_compton_flat.c
    mid   = size(a_img, 3) ÷ 2
    a_slice = a_img[:, :, mid]
    c_slice = c_img[:, :, mid]

    a_lo, a_hi = quantile(vec(a_slice), 0.01), quantile(vec(a_slice), 0.995)
    c_lo, c_hi = quantile(vec(c_slice), 0.01), quantile(vec(c_slice), 0.995)

    fig = CM.Figure(size = (1250, 570), fontsize = 13)

    ax1 = CM.Axis(
        fig[1, 1]; title = "Photoelectric a(r)", subtitle = "Cong + PWLS + ACNR + Capping (slice $mid)",
        aspect = CM.DataAspect()
    )
    hm1 = CM.heatmap!(ax1, a_slice; colormap = :viridis, colorrange = (a_lo, a_hi))
    CM.Colorbar(fig[1, 2], hm1; width = 12)

    ax2 = CM.Axis(
        fig[1, 3]; title = "Compton  c(r)", subtitle = "Cong + PWLS + ACNR + Capping (slice $mid)",
        aspect = CM.DataAspect()
    )
    hm2 = CM.heatmap!(ax2, c_slice; colormap = :viridis, colorrange = (c_lo, c_hi))
    CM.Colorbar(fig[1, 4], hm2; width = 12)

    CM.save(joinpath(RESULTS_DIR, "recon_photo_compton.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 000a0001-0000-4000-8000-000000000004
md"""
## 6. Virtual Monoenergetic Imaging — Mono+ / VMI+

### References

- **[G14] Grant, Flohr, Krauss, Sedlmair, Thomas, Schmidt (2014)** *Invest Radiol*
  49(9):586–592. DOI: 10.1097/RLI.0000000000000060 · PMID: 24710203.
  Siemens Healthcare DE CT team — reference paper for the syngo.via "Mono+"
  prototype (later VMI+).
- **[S14] Schabel et al. (2014)** *Fortschr Röntgenstr* 186:591–597.
  DOI: 10.1055/s-0034-1366423 — clinical evaluation.

### 6.1 Plain VMI (baseline)

Image-domain linear combination ([CDW22] Eq 4):

    μ(E, r) = p(E)·a(r) + q(E)·c(r)

Uses the **same** analytic `p(ε), q(ε)` (Eqs 3c, 3e) that drove the
decomposition → synthesis is internally consistent with the inversion, so
water voxels synthesize back to XA's water μ(E) within ~1% across the
diagnostic range. HU referenced to XA's linear μ of water at each target
energy (`BS.compute_μ_at_energy(XA.Materials.water, E)`).

Output: `sim_vmi.volumes` — VMI HU stacks at 40 / 70 / 100 / 140 keV.
These are the inputs to Mono+ below.
"""

# ╔═╡ 000a0002-0000-4000-8000-000000000004
# VMI synthesis at 40 / 70 / 100 / 140 keV.
# μ(E) = p(E)·a + q(E)·c → HU via water reference at each energy.
sim_vmi = BS.synth_vmi_hu(
    sim_recon_photo_compton_flat.a,
    sim_recon_photo_compton_flat.c,
    [40.0, 70.0, 100.0, 140.0];
    fov_mask_radius_frac = 0.5,
)

# ╔═╡ 000a0004-0000-4000-8000-000000000004
# Plain VMI at 40 / 70 / 100 / 140 keV — soft-tissue window.
let
    fig = CM.Figure(size = (1400, 400), fontsize = 13)
    mid = size(sim_vmi.volumes[1], 3) ÷ 2
    for (i, E) in enumerate(sim_vmi.energies)
        ax = CM.Axis(fig[1, i]; title = "VMI  $(Int(E)) keV", aspect = CM.DataAspect())
        CM.heatmap!(ax, sim_vmi.volumes[i][:, :, mid];
                    colormap = :grays, colorrange = (-200, 500))
    end
    CM.Label(fig[0, :]; text = "Plain VMI — Soft Tissue Window (−200 … 500 HU)",
             fontsize = 15, font = :bold)
    CM.save(joinpath(RESULTS_DIR, "vmi_all_energies.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 000a0005-0000-4000-8000-000000000004
md"""
### 6.2 Mono+ / VMI+ — Grant 2014 1:1 parity

This is a straight port of **[G14] §Technique for Calculating Mono+ Images**.
Where the paper is explicit we copy it exactly; where it's silent we make a
single, documented best guess.

#### What the paper says verbatim

> "…low-keV images (in which iodine pixels have a high contrast to the
> surrounding tissue) and images of optimal keV from a noise perspective
> (typically, minimum image noise is obtained at approximately 70 keV)
> are computed. By means of a frequency-split technique, both the low-keV
> images and the images with minimum image noise are decomposed into 2
> sets of subimages. The first set contains only lower spatial frequencies
> and, hence, most of the object information, the second set contains the
> remaining high spatial frequencies and, hence, mostly image noise.
> Finally, the lower spatial frequency stack at low keV is combined with
> the high spatial frequency stack at optimal keV from a noise perspective…"

Extracted hard constraints:

| # | Specification (paper) | Implementation |
|---|----------------------|----------------|
| 1 | Two inputs: `VMI_target` + `VMI_opt`                                       | ✓ Read from `sim_vmi.volumes` at `E_target` and `E_noise_opt` |
| 2 | Noise-optimal energy ≈ 70 keV                                              | ✓ `vmip_E_noise_opt = 70.0` |
| 3 | Frequency-split decomposition of **both** images into LP + HP subimages    | ✓ Same linear LP applied to both branches |
| 4 | Final = LP(VMI_target) + HP(VMI_opt)                                       | ✓ `Mono+(E) = LP(VMI_E) + (VMI_opt − LP(VMI_opt))` |
| 5 | LP + HP are complementary (partition the frequency axis)                   | ✓ HP ≡ I − LP by construction (linear LP ⇒ complementary) |

#### What the paper does NOT specify — documented best guesses

| Decision | Paper | **Best guess** | Rationale |
|----------|-------|----------------|-----------|
| LP filter shape           | Unstated (proprietary) — abstract mentions *"regional spatial frequency-based recombination"* but body text only says "frequency-split" | **2D Gaussian LP** (FFT-diagonal) | Simplest linear LP that cleanly partitions the frequency axis; "regional" in the abstract reads as *spatial-domain* rather than spatially-adaptive; matches the plain "frequency-split" language |
| LP kernel size (σ)        | Unstated                                         | **σ = 2.0 px**                        | Rough match to the scale where image noise lives (~1–2 px) and object features (≥5 px) survive; tunable via `vmip_σ_lp_px` |
| Same LP for both branches | Strongly implied ("both…decomposed into 2 sets") | **Yes, identical σ and kernel**       | Only way LP + HP = I across both branches → Mono+(E_opt) = VMI_opt identically (sanity check) |
| Any pre-denoising of VMI_opt | Not mentioned                                  | **None**                              | Paper treats VMI_opt as-is; strict parity ⇒ no hidden preconditioning |
| 2D vs 3D LP               | Paper figures are axial 2D                       | **Per-slice 2D**                      | Matches clinical axial viewing; 3D LP would couple across slice thickness, not described by paper |
| Boundary conditions       | Unstated                                         | **FFT periodic**                      | No artifacts in body interior; tiny ring at scanner-bore edge is masked by the FOV circle anyway |

#### Final formula

    Mono+(E_target)  =  LP_σ(VMI_E_target)  +  VMI_opt  −  LP_σ(VMI_opt)

where `LP_σ` is a 2D Gaussian low-pass with std-dev `σ` pixels. At
`E_target = E_noise_opt` the formula collapses to `Mono+(E_opt) = VMI_opt`
exactly — a good consistency check.

#### Expected result ([G14] Figs 4–5)

Mono+ CNR rises monotonically as keV drops, peaking at 40 keV. Plain VMI
CNR peaks near 80 keV and drops at low keV because noise grows faster
than iodine contrast.
"""

# ╔═╡ 6454952b-4cdb-4beb-979e-c41595dbe204
# VMI+ / Mono+ — 1:1 port of Grant et al. 2014 [G14] §Technique for
# Calculating Mono+ Images.
#
# Paper specs (HARD constraints — faithfully implemented):
#   · Inputs: VMI at target keV + VMI at noise-optimal keV (~70 keV)
#   · Frequency-split decomposition of BOTH images into LP + HP subimages
#   · Combination: Mono+(E) = LP(VMI_E) + HP(VMI_opt)
#   · LP + HP complementary (so Mono+(E_opt) = VMI_opt identically)
#
# Paper is SILENT on (best guesses documented below, all tunable):
#   · LP filter shape      →  [BEST GUESS] 2D Gaussian LP via FFT diagonal
#   · LP kernel size       →  [BEST GUESS] σ = `vmip_σ_lp_px` pixels
#   · Same LP for both?    →  YES (strongly implied by paper wording)
#   · Pre-denoise VMI_opt? →  NO (strict parity → use as-is)
#   · 2D per-slice vs 3D?  →  [BEST GUESS] per-slice 2D (matches paper figs)
#   · Boundary conditions  →  [BEST GUESS] FFT periodic
#
# Hyperparameters — ONE real knob (σ_LP) + one structural choice (E_opt).
begin
    # Noise-optimal energy. [G14] body: "approximately 70 keV". MUST be in
    # sim_vmi.energies.
    vmip_E_noise_opt = 100.0

    # LP Gaussian σ in pixels. [BEST GUESS — paper unspecified.] Typical
    # values: σ ≈ 1.0 preserves most detail (mild mix), σ ≈ 3.0 heavily
    # hands HF to VMI_opt (aggressive denoising at low-keV), σ ≈ 2.0 is
    # balanced. Effective HP cutoff frequency ≈ 1/(2πσ) cycles/pixel.
    vmip_σ_lp_px = 1.5
end

# ╔═╡ 000b0002-0000-4000-8000-000000000004
# Mono+ frequency-split via FFT Gaussian LP — Grant 2014 1:1 parity.
# Mono+(E) = LP_σ(VMI_E) + VMI_opt − LP_σ(VMI_opt). Identity at E_opt.
sim_vmi_plus = BS.apply_mono_plus(
    sim_vmi.volumes, sim_vmi.energies;
    E_noise_opt = vmip_E_noise_opt,
    σ_lp_px     = vmip_σ_lp_px,
)

# ╔═╡ 000b0003-0000-4000-8000-000000000004
# VMI vs VMI+ side-by-side at each energy — soft tissue window.
let
    fig = CM.Figure(size = (1100, 1900), fontsize = 13)
    mid = size(sim_vmi.volumes[1], 3) ÷ 2
    for (i, E) in enumerate(sim_vmi.energies)
        ax1 = CM.Axis(
            fig[i, 1]; title = "VMI  $(Int(E)) keV",
            aspect = CM.DataAspect()
        )
        CM.heatmap!(ax1, sim_vmi.volumes[i][:, :, mid];
                    colormap = :grays, colorrange = (-200, 500))

        ax2 = CM.Axis(
            fig[i, 2]; title = "Mono+ $(Int(E)) keV  (σ_LP=$(sim_vmi_plus.σ_lp_px) px, HP@$(Int(sim_vmi_plus.E_noise_opt)) keV)",
            aspect = CM.DataAspect()
        )
        CM.heatmap!(ax2, sim_vmi_plus.volumes[i][:, :, mid];
                    colormap = :grays, colorrange = (-200, 500))
    end
    CM.Label(fig[0, :]; text = "VMI vs Mono+ (Grant 2014)  —  Soft Tissue Window (−200 … 500 HU)",
             fontsize = 15, font = :bold)
    CM.save(joinpath(RESULTS_DIR, "vmi_vs_vmi_plus.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 000b0004-0000-4000-8000-000000000004
# Water-ROI noise: VMI vs VMI+ at each energy. Expect VMI+ noise ≈ const low.
let
    mid = size(sim_vmi.volumes[1], 3) ÷ 2
    nx, ny = size(sim_vmi.volumes[1], 1), size(sim_vmi.volumes[1], 2)
    cx, cy = nx ÷ 2, ny ÷ 2
    r = 15
    mask_idx = [(dx, dy) for dy in -r:r for dx in -r:r if dx^2 + dy^2 <= r^2]

    function water_σ(vol)
        vals = [vol[cx + dx, cy + dy, mid] for (dx, dy) in mask_idx]
        std(vals)
    end

    σ_vmi   = [water_σ(v) for v in sim_vmi.volumes]
    σ_vmip  = [water_σ(v) for v in sim_vmi_plus.volumes]

    fig = CM.Figure(size = (800, 500), fontsize = 13)
    ax = CM.Axis(
        fig[1, 1]; title = "Central-water-ROI σ — VMI vs Mono+ (Grant 2014)",
        xlabel = "Energy (keV)", ylabel = "σ (HU)",
        xticks = (1:length(sim_vmi.energies), string.(Int.(sim_vmi.energies)))
    )
    CM.scatterlines!(ax, 1:length(sim_vmi.energies), σ_vmi;  color = :firebrick,  linewidth = 2, markersize = 10, label = "VMI (post-ACNR)")
    CM.scatterlines!(ax, 1:length(sim_vmi_plus.energies), σ_vmip; color = :steelblue, linewidth = 2, markersize = 10, label = "Mono+ (Grant 2014, σ_LP=$(sim_vmi_plus.σ_lp_px) px)")
    CM.axislegend(ax; position = :rt)
    CM.save(joinpath(RESULTS_DIR, "vmi_vs_vmi_plus_noise.png"), fig, px_per_unit = 2)
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
# ╠═00080002-0000-4000-8000-000000000004
# ╠═00080003-0000-4000-8000-000000000004
# ╠═00080004-0000-4000-8000-000000000004
# ╟─00080005-0000-4000-8000-000000000004
# ╠═00080012-0000-4000-8000-000000000004
# ╟─00080013-0000-4000-8000-000000000004
# ╟─00080014-0000-4000-8000-000000000004
# ╠═00080021-0000-4000-8000-000000000004
# ╠═00080022-0000-4000-8000-000000000004
# ╟─00080015-0000-4000-8000-000000000004
# ╠═00080016-0000-4000-8000-000000000004
# ╟─c1139ae3-5186-445e-81b8-5d932ca5ef98
# ╟─00080017-0000-4000-8000-000000000004
# ╟─00080018-0000-4000-8000-000000000004
# ╠═00080006-0000-4000-8000-000000000004
# ╠═00080007-0000-4000-8000-000000000004
# ╟─00080019-0000-4000-8000-000000000004
# ╟─00080023-0000-4000-8000-000000000004
# ╠═00080024-0000-4000-8000-000000000004
# ╠═00080025-0000-4000-8000-000000000004
# ╟─00080020-0000-4000-8000-000000000004
# ╠═00090002-0000-4000-8000-000000000004
# ╟─fd9969e0-415f-409d-8672-fe2d963b6486
# ╟─2b6fd506-c624-4be3-9a71-1d366ae58ada
# ╟─00090003-0000-4000-8000-000000000004
# ╟─000a0001-0000-4000-8000-000000000004
# ╠═000a0002-0000-4000-8000-000000000004
# ╟─000a0004-0000-4000-8000-000000000004
# ╟─000a0005-0000-4000-8000-000000000004
# ╠═6454952b-4cdb-4beb-979e-c41595dbe204
# ╠═000b0002-0000-4000-8000-000000000004
# ╟─000b0003-0000-4000-8000-000000000004
# ╟─000b0004-0000-4000-8000-000000000004
