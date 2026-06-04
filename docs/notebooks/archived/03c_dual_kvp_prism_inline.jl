### A Pluto.jl notebook ###
# v0.19.0

# ╔═╡ 06000001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ 06000001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 06000001-0000-4000-8000-000000000003
using Statistics: mean, std, quantile

# ╔═╡ 06000001-0000-4000-8000-000000000010
md"""
# 03c · Dual-kVp Switching VMI · Cong-Only Baseline (Standalone)

GE Apex Elite GSI rapid-kVp-switching simulation (80 + 140 kVp, Gammex
472 phantom) running the **Cong per-ray polychromatic material
decomposition** with the un-corrected-physics flags disabled — same
pipeline shape as `03_dual_kvp_switching_vmi.jl` but without SF-JSD
sinogram denoising.

```
Simulate 80 kVp  →┐
                   ├─→  Cong polychromatic per-ray decomposition
Simulate 140 kVp →┘                    │
                                       ▼
                    sino_iodine, sino_water  (basis line integrals)
                                       │
                    FBP × 2   (iodine, water basis maps)
                                       │
                    Z-Direction Median Filter
                                       │
                    Monoenergetic VMI Synthesis  (textbook 2-basis)
                                       │
                    Mono+ Post-Processing  (per-keV σ)
                                       │
                    Measured vs Theoretical Per-Rod Regression
                          at 40 / 70 / 100 / 140 keV
```

!!! info "History"
    Earlier revisions of this notebook inlined the full source of
    PrismMaterialDecomposition.jl and ran a Cong → PRISM-regularized-
    denoise hybrid.  The physics audit (`use_fill_factor` and
    `use_optical_crosstalk` are simulated but not yet corrected for
    inside BS) closed the accuracy gap that motivated the PRISM
    denoising step, so this notebook has been simplified back to a
    clean Cong-only pipeline.  See git history (`wip(nb03c): …`) for
    the PRISM-inlined version.
"""

# ╔═╡ 06000001-0000-4000-8000-000000000020
md"""
## Setup
"""

# ╔═╡ 06000001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 06000001-0000-4000-8000-000000000031
import CairoMakie as CM

# ╔═╡ 06000001-0000-4000-8000-000000000040
begin
    GPU_BACKEND = let
        candidates = [
            (:Metal, "dde4c033-4e86-420c-a63e-0dd931031962", :MtlArray),
            (:CUDA, "052768ef-5323-5732-b1bb-66c8b64840ba", :CuArray),
            (:AMDGPU, "21141c5a-9bdb-4563-92ae-f87d6854732e", :ROCArray),
        ]
        detected = (name = "CPU", to_gpu = identity)
        for (pkg, uuid, ctor) in candidates
            pkg_id = Base.PkgId(Base.UUID(uuid), String(pkg))
            Base.locate_package(pkg_id) === nothing && continue
            try
                m = Base.require(pkg_id)
                if Base.invokelatest(getfield(m, :functional))
                    detected = (name = string(pkg), to_gpu = getfield(m, ctor))
                    break
                end
            catch
            end
        end
        detected
    end

    to_gpu(x) = GPU_BACKEND.to_gpu(x)
end

# ╔═╡ 06000001-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 06000002-0000-4000-8000-000000000001
md"""
## 1. `Phantom`: Gammex Model 472
"""

# ╔═╡ 06000002-0000-4000-8000-000000000010
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,
    n_slices = 16,
    fov_cm = 35.0,
    z_cm = 1.0,
);

# ╔═╡ 06000002-0000-4000-8000-000000000020
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 06000003-0000-4000-8000-000000000001
md"""
## 2. `Scanner`: GE Revolution Apex Elite
"""

# ╔═╡ 06000003-0000-4000-8000-000000000010
scanner = BS.Scanner(
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

    electronic_noise = 0,
    detection_gain = 10.0,
);

# ╔═╡ 06000004-0000-4000-8000-000000000001
md"""
## 3. Dual-kVp Protocols (Rapid kVp Switching)

| kVp | Instantaneous mA | Duty cycle | Effective mA |
|-----|------------------|------------|--------------|
| 80  | 407              | 0.65       | 264.55       |
| 140 | 405              | 0.35       | 141.75       |
"""

# ╔═╡ 06000004-0000-4000-8000-000000000010
protocol_low = BS.CTProtocol(
    kVp = 80,
    mA = 407 * 0.65,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 06000004-0000-4000-8000-000000000020
protocol_high = BS.CTProtocol(
    kVp = 140,
    mA = 405 * 0.35,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 06000005-0000-4000-8000-000000000001
md"""
## 4. `SimOptions` and `ReconOptions`
"""

# ╔═╡ 06000005-0000-4000-8000-000000000010
sim_opts = BS.SimOptions(fidelity = :eict, seed = 1234);

# ╔═╡ 06000005-0000-4000-8000-000000000020
recon_opts = let
    slice_thickness_mm = 0.625
    n_recon_slices = round(Int, 5.0 / slice_thickness_mm)
    BS.ReconOptions(
        matrix_size = (512, 512, n_recon_slices),
        fov_cm = 35.0,
        z_cm = 0.5,
    )
end;

# ╔═╡ 06000006-0000-4000-8000-000000000001
md"""
## 5. Forward Project

Run `BS.simulate!` on each kVp protocol.  Same as
`03_dual_kvp_switching_vmi.jl`.
"""

# ╔═╡ 06000006-0000-4000-8000-000000000010
sim_low = let
    @info "Simulating: 80 kVp / $(round(protocol_low.mA, digits = 1)) mA-eff (DE low)…"
    ws = BS.create_eict_workspace(scanner, protocol_low, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, protocol_low, sim_opts)
    result = (sino = Array(ws.sinogram), geom = ws.geom)
    ws = nothing; GC.gc(true)
    result
end;

# ╔═╡ 06000006-0000-4000-8000-000000000020
sim_high = let
    @info "Simulating: 140 kVp / $(round(protocol_high.mA, digits = 1)) mA-eff (DE high)…"
    ws = BS.create_eict_workspace(scanner, protocol_high, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, protocol_high, sim_opts)
    result = (sino = Array(ws.sinogram), geom = ws.geom)
    ws = nothing; GC.gc(true)
    result
end;

# ╔═╡ 06000006-0000-4000-8000-000000000040
let
    n_row = size(sim_low.sino, 2)
    mid_r = n_row ÷ 2 + 1

    slice_lo = permutedims(sim_low.sino[:, mid_r, :], (2, 1))
    slice_hi = permutedims(sim_high.sino[:, mid_r, :], (2, 1))

    all_v = vcat(vec(slice_lo), vec(slice_hi))
    sino_window = (
        Float64(quantile(all_v, 0.01)),
        Float64(quantile(all_v, 0.99)),
    )

    fig = CM.Figure(size = (1180, 580))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "View", ylabel = "Detector Column",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )

    panels = (
        (1, 1, "80 kVp", slice_lo),
        (1, 2, "140 kVp", slice_hi),
    )

    for (r, c, ttl, slice) in panels
        ax = CM.Axis(fig[r, c]; title = ttl, axis_kwargs...)
        CM.heatmap!(ax, slice; colormap = :viridis, colorrange = sino_window)
    end
    CM.Colorbar(
        fig[1, 3]; colormap = :viridis, colorrange = sino_window,
        label = "Log Line Integral", width = 16, labelsize = 22, ticklabelsize = 18
    )
    fig
end

# ╔═╡ 06000008-0000-4000-8000-000000000001
md"""
## 6. Cong Polychromatic Material Decomposition

Per-ray Newton solve on the polychromatic transmission integral with a
fixed `(iodine, water)` basis seeded by `water_basis = (a = 0, c = 1)`.
The basis builder constructs the **per-ray effective spectrum** that
`simulate!` actually applied — tube × flat filter × bowtie × heel ×
η(E) — and normalizes per ray so `Σ_E ŵ[col, row, E] = 1`.  Matches
the spectrum baked into the EICT workspace's `bowtie_air_reference`
so Cong's inversion solves the same polychromatic transmission integral
the forward model used.
"""

# ╔═╡ 06000008-0000-4000-8000-000000000010
# Per-ray effective spectrum (source × bowtie × heel × η) that simulate!
# applied.  `BS.resolve_source_spectrum_full` chains the same `PhysicsConfig`
# the EICT workspace built, so the inversion sees exactly the spectrum the
# forward model used (heel + η are conditionally applied per the SimOptions
# `use_*` flags).
material_basis = let
    iodine_mat = BS.XA.Elements.Iodine
    water_mat = BS.XA.Materials.water

    e_L, ŵ_L = BS.resolve_source_spectrum_full(
        sim_opts, protocol_low;
        scanner = scanner, geom = sim_low.geom, phantom = phantom,
        diagnostic = true, label = "low",
    )
    e_H, ŵ_H = BS.resolve_source_spectrum_full(
        sim_opts, protocol_high;
        scanner = scanner, geom = sim_high.geom, phantom = phantom,
        diagnostic = true, label = "high",
    )

    p_L = Float32[Float32(BS.compute_mass_μ_at_energy(iodine_mat, Float64(E))) for E in e_L]
    q_L = Float32[Float32(BS.compute_mass_μ_at_energy(water_mat, Float64(E))) for E in e_L]
    p_H = Float32[Float32(BS.compute_mass_μ_at_energy(iodine_mat, Float64(E))) for E in e_H]
    q_H = Float32[Float32(BS.compute_mass_μ_at_energy(water_mat, Float64(E))) for E in e_H]

    (
        ŵ_L = ŵ_L, p_L = p_L, q_L = q_L,
        ŵ_H = ŵ_H, p_H = p_H, q_H = q_H,
    )
end;

# ╔═╡ 06000008-0000-4000-8000-000000000040
# Pure Cong polychromatic per-ray decomposition.
#
# Earlier this cell ran Cong → PRISM-regularized-denoise.  Once the
# physics audit pinned the dominant accuracy gap on un-corrected fill_
# factor + optical_crosstalk effects (see SimOptions cell above), raw
# Cong alone tracks the theoretical curves and PRISM denoise adds
# nothing at this operating point.  The PRISM machinery in §6 stays
# inline as dormant code for future noise-handling experiments.
sino_basis = let
    # Use the §5.5-corrected sinograms (fill_factor offset removed + crosstalk
    # deconvolved if enabled in sim_opts).  When both `use_*` flags are false
    # the corrections are no-ops and these are identical to sim_low.sino /
    # sim_high.sino.
    sino_low_gpu = to_gpu(Float32.(sim_low.sino))
    sino_high_gpu = to_gpu(Float32.(sim_high.sino))

    sino_y = similar(sino_low_gpu);  fill!(sino_y, 0.0f0)
    sino_c = similar(sino_low_gpu);  fill!(sino_c, 0.0f0)

    @info "Cong polychromatic decomposition: $(size(sim_low.sino))"
    cong_elapsed = @elapsed begin
        cong_ws = BS.create_cong_workspace(sino_low_gpu, material_basis)
        BS.apply_cong!(
            cong_ws, sino_y, sino_c, sino_low_gpu, sino_high_gpu;
            water_basis = (a = 0.0f0, c = 1.0f0),
        )
    end
    @info "Cong done in $(round(cong_elapsed, digits = 1)) s"

    result = (
        sino_iodine = Array(sino_y),
        sino_water = Array(sino_c),
        geom = sim_low.geom,
    )
    sino_low_gpu = nothing; sino_high_gpu = nothing
    sino_y = nothing; sino_c = nothing; cong_ws = nothing
    GC.gc(true)
    result
end;

# ╔═╡ 06000008-0000-4000-8000-000000000050
let
    n_row = size(sino_basis.sino_iodine, 2)
    mid_r = n_row ÷ 2 + 1

    fig = CM.Figure(size = (1400, 580))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "View", ylabel = "Detector Column",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )

    _qrange(arr) = (
        Float64(quantile(vec(arr), 0.01)),
        Float64(quantile(vec(arr), 0.99)),
    )

    slice_iod = permutedims(sino_basis.sino_iodine[:, mid_r, :], (2, 1))
    slice_wat = permutedims(sino_basis.sino_water[:, mid_r, :], (2, 1))

    panels = (
        (
            1, 1, 2, "PRISM Iodine Basis Sinogram", "g/cm²",
            slice_iod, _qrange(slice_iod),
        ),
        (
            1, 3, 4, "PRISM Water Basis Sinogram", "g/cm²",
            slice_wat, _qrange(slice_wat),
        ),
    )

    for (r, panel_c, cbar_c, ttl, cbar_label, slice, range) in panels
        ax = CM.Axis(fig[r, panel_c]; title = ttl, axis_kwargs...)
        CM.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        CM.Colorbar(
            fig[r, cbar_c]; colormap = :viridis, colorrange = range,
            label = cbar_label, width = 16, labelsize = 22, ticklabelsize = 18
        )
    end
    fig
end

# ╔═╡ 06000009-0000-4000-8000-000000000001
md"""
## 7. FBP: Iodine and Water Basis Maps
"""

# ╔═╡ 06000009-0000-4000-8000-000000000010
basis_volumes = let
    matrix_size = recon_opts.matrix_size
    geom = sino_basis.geom

    function _fbp(sino_cpu)
        sino_gpu = to_gpu(Float32.(sino_cpu))
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, geom, matrix_size; filter = BS.SoftFilter(),
        )
        recon = Array(BS.reconstruct!(ws, sino_gpu, geom))
        ws = nothing; sino_gpu = nothing
        GC.gc(true)
        return Float32.(recon)
    end

    (
        vol_iodine_raw = _fbp(sino_basis.sino_iodine),
        vol_water_raw = _fbp(sino_basis.sino_water),
        geom = geom,
    )
end;

# ╔═╡ 06000009-0000-4000-8000-000000000030
let
    fig = CM.Figure(size = (1180, 580))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    mid = size(basis_volumes.vol_iodine_raw, 3) ÷ 2

    _qrange(arr) = (
        Float64(quantile(vec(arr), 0.01)),
        Float64(quantile(vec(arr), 0.99)),
    )

    slice_iod = basis_volumes.vol_iodine_raw[:, :, mid]
    slice_wat = basis_volumes.vol_water_raw[:, :, mid]

    panels = (
        (1, 1, 2, "Iodine Basis", "g/cm³", slice_iod, _qrange(slice_iod)),
        (1, 3, 4, "Water Basis", "g/cm³", slice_wat, _qrange(slice_wat)),
    )

    for (r, panel_c, cbar_c, ttl, cbar_label, slice, range) in panels
        ax = CM.Axis(
            fig[r, panel_c]; title = ttl,
            aspect = CM.DataAspect(), axis_kwargs...
        )
        CM.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        CM.hidedecorations!(ax)
        CM.Colorbar(
            fig[r, cbar_c]; colormap = :viridis, colorrange = range,
            label = cbar_label, width = 16, labelsize = 22, ticklabelsize = 18
        )
    end
    fig
end

# ╔═╡ 0600000a-0000-4000-8000-000000000001
md"""
## 8. Z-Direction Median Filter
"""

# ╔═╡ 0600000a-0000-4000-8000-000000000005
Z_MEDIAN_ADJACENT = 2;

# ╔═╡ 0600000a-0000-4000-8000-000000000010
basis_z = let
    (
        vol_iodine = BS.apply_median_z(
            basis_volumes.vol_iodine_raw;
            adjacent_slices = Z_MEDIAN_ADJACENT,
        ),
        vol_water = BS.apply_median_z(
            basis_volumes.vol_water_raw;
            adjacent_slices = Z_MEDIAN_ADJACENT,
        ),
        geom = basis_volumes.geom,
    )
end;

# ╔═╡ 0600000b-0000-4000-8000-000000000001
md"""
## 9. VMI Synthesis

Textbook 2-basis linear mix; VMI grid **40, 70, 100, 140 keV** as
requested.
"""

# ╔═╡ 0600000b-0000-4000-8000-000000000012
# Solid-water anchor diagnostic — mirrors nb07's `solid_water_basis` cell.
# Mean (c_water, c_iodine) over a deeply-eroded REGION_SOLID_WATER ROI
# (12-px ≈ 8 mm erosion in the Gammex 472 mid-slice plane).  Used in the
# VMI-synth cell below to log Δ% drift between the SW-anchored μ_water and
# the textbook mono divisor μρ_water(E) — a single-number sanity check
# that the decomposition is recovering pure water at c_water ≈ 1.0.
solid_water_basis = let
    ERODE_PX = 12.0

    mask_2d_raw = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
    sw_bool_raw = (mask_2d_raw .== UInt8(BS.REGION_SOLID_WATER))
    sw_bool = BS.erode_mask_2d(sw_bool_raw; erode_px = ERODE_PX)

    n_raw = count(sw_bool_raw); n_eroded = count(sw_bool)
    n_eroded == 0 && error(
        "solid_water_basis: deep erosion (σ = $(ERODE_PX) px) wiped out the SW " *
            "ROI (raw count = $(n_raw)).  Reduce erode_px or check phantom mask."
    )
    @info "solid_water_basis: SW mid-slice voxel count $(n_raw) → $(n_eroded) " *
        "after $(ERODE_PX)-px erosion"

    sw_idx = findall(sw_bool)
    n_z = size(basis_z.vol_water, 3)
    function _mean(vol)
        s = 0.0; n = 0
        for z in 1:n_z, ci in sw_idx
            s += vol[ci, z]; n += 1
        end
        return s / n
    end

    c_w = Float64(_mean(basis_z.vol_water))
    c_i = Float64(_mean(basis_z.vol_iodine))
    @info "solid_water_basis: ⟨c_water⟩_SW  = $(round(c_w, digits = 4)) g/cm³, " *
        "⟨c_iodine⟩_SW = $(round(c_i, digits = 6)) g/cm³"

    (
        c_water = c_w,
        c_iodine = c_i,
        n_voxels = length(sw_idx) * n_z,
        mask_2d = collect(sw_bool),
    )
end;

# ╔═╡ 0600000b-0000-4000-8000-000000000015
de_vmi_energies = [40.0, 70.0, 100.0, 140.0];

# ╔═╡ 0600000b-0000-4000-8000-000000000020
vmi_HU_by_keV = let
    c_iodine_mg_per_mL = basis_z.vol_iodine .* 1000.0f0

    out = Dict{Float64, Array{Float32, 3}}()
    for E in de_vmi_energies
        # Diagnostic only — the SW-ROI synth-evaluated μ_water vs the
        # textbook mono divisor.  Drift = residual basis-decomp bias.
        μρ_w = BS.compute_mass_μ_at_energy(BS.XA.Materials.water, E)
        μρ_I = BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, E)
        μ_water_anchor = solid_water_basis.c_water * μρ_w +
            solid_water_basis.c_iodine * μρ_I
        Δ_pct = 100.0 * (μ_water_anchor - μρ_w) / μρ_w
        @info "VMI synth @ $(Int(E)) keV: divisor = $(round(μρ_w, digits = 5)) cm⁻¹ " *
            "(mono μρ_water);  SW-ROI anchor = $(round(μ_water_anchor, digits = 5)) " *
            "→ Δ = $(round(Δ_pct, digits = 2))%"

        out[E] = BS.synth_vmi_2basis(
            basis_z.vol_water, c_iodine_mg_per_mL;
            energy_keV = E,
        )
    end
    out
end;

# ╔═╡ 0600000b-0000-4000-8000-000000000040
let
    HU_window = (-200, 500)

    fig = CM.Figure(size = (1180, 1180))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    sample = vmi_HU_by_keV[40.0]
    mid = size(sample, 3) ÷ 2

    for (k, E) in enumerate(de_vmi_energies)
        r = ((k - 1) ÷ 2) + 1
        c = ((k - 1) % 2) + 1
        ax = CM.Axis(
            fig[r, c]; title = "$(Int(E)) keV VMI",
            aspect = CM.DataAspect(), axis_kwargs...,
        )
        CM.heatmap!(
            ax, vmi_HU_by_keV[E][:, :, mid];
            colormap = :grays, colorrange = HU_window,
        )
        CM.hidedecorations!(ax)
    end
    CM.Colorbar(
        fig[1:2, 3];
        colormap = :grays, colorrange = HU_window,
        label = "HU", width = 16, labelsize = 22, ticklabelsize = 18,
    )

    fig
end

# ╔═╡ 0600000c-0000-4000-8000-000000000001
md"""
## 10. VMI Post-Processing (Mono+)
"""

# ╔═╡ 0600000c-0000-4000-8000-000000000005
# σ_vmi_lp_px = Float64[2.0, 0.0, 1.0, 1.0];   # 40, 70, 100, 140 keV
σ_vmi_lp_px = Float64[0.0, 0.0, 0.0, 0.0];   # 40, 70, 100, 140 keV

# ╔═╡ 0600000c-0000-4000-8000-000000000010
vmi_HU_final = let
    volumes = [vmi_HU_by_keV[E] for E in de_vmi_energies]
    ws = BS.create_mono_plus_workspace(
        volumes[1]; n_energies = length(de_vmi_energies),
    )
    BS.apply_mono_plus!(
        ws, volumes, de_vmi_energies;
        E_noise_opt = 70.0,
        σ_lp_px = σ_vmi_lp_px,
        verbose = true,
    )
    out = Dict{Float64, Array{Float32, 3}}()
    for (i, E) in enumerate(de_vmi_energies)
        out[E] = copy(ws.out_vols[i])
    end
    ws = nothing; GC.gc(true)
    out
end;

# ╔═╡ 0600000c-0000-4000-8000-000000000030
let
    HU_window = (-200, 500)

    fig = CM.Figure(size = (1180, 1180))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    sample = vmi_HU_final[40.0]
    mid = size(sample, 3) ÷ 2

    for (k, E) in enumerate(de_vmi_energies)
        r = ((k - 1) ÷ 2) + 1
        c = ((k - 1) % 2) + 1
        ax = CM.Axis(
            fig[r, c]; title = "$(Int(E)) keV VMI",
            subtitle = "Mono+",
            aspect = CM.DataAspect(), axis_kwargs...,
        )
        CM.heatmap!(
            ax, vmi_HU_final[E][:, :, mid];
            colormap = :grays, colorrange = HU_window,
        )
        CM.hidedecorations!(ax)
    end
    CM.Colorbar(
        fig[1:2, 3];
        colormap = :grays, colorrange = HU_window,
        label = "HU", width = 16, labelsize = 22, ticklabelsize = 18,
    )

    fig
end

# ╔═╡ 0600000d-0000-4000-8000-000000000001
md"""
## Results

Per-rod measured vs theoretical HU at 40 / 70 / 100 / 140 keV.
"""

# ╔═╡ 0600000d-0000-4000-8000-000000000010
ROD_LABELS = (
    Ca = (UInt8(10), UInt8(11), UInt8(12), UInt8(13), UInt8(14), UInt8(15), UInt8(16)),
    I = (UInt8(20), UInt8(21), UInt8(22), UInt8(23), UInt8(24), UInt8(25), UInt8(26)),
);

# ╔═╡ 0600000d-0000-4000-8000-000000000020
ROD_NAMES = (
    Ca = ("50 mg/mL", "100 mg/mL", "200 mg/mL", "300 mg/mL", "400 mg/mL", "500 mg/mL", "600 mg/mL"),
    I = ("2.0 mg/mL", "2.5 mg/mL", "5.0 mg/mL", "7.5 mg/mL", "10.0 mg/mL", "15.0 mg/mL", "20.0 mg/mL"),
);

# ╔═╡ 0600000d-0000-4000-8000-000000000030
rod_data = let
    materials = phantom_cpu.materials
    mask_2d = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
    nx, ny = size(mask_2d)
    ROI_RADIUS_PX = 8

    function rod_centroid(label::UInt8)
        idx = findall(==(label), mask_2d)
        isempty(idx) && error("rod_centroid: no voxels with label $label")
        cx = sum(ci -> Float64(ci[1]), idx) / length(idx)
        cy = sum(ci -> Float64(ci[2]), idx) / length(idx)
        return (cx, cy)
    end

    function rod_roi_mask(label::UInt8)
        cx, cy = rod_centroid(label)
        i_lo = max(1, floor(Int, cx - ROI_RADIUS_PX))
        i_hi = min(nx, ceil(Int, cx + ROI_RADIUS_PX))
        j_lo = max(1, floor(Int, cy - ROI_RADIUS_PX))
        j_hi = min(ny, ceil(Int, cy + ROI_RADIUS_PX))
        roi = CartesianIndex{2}[]
        r² = Float64(ROI_RADIUS_PX)^2
        for j in j_lo:j_hi, i in i_lo:i_hi
            ((i - cx)^2 + (j - cy)^2) ≤ r² && push!(roi, CartesianIndex(i, j))
        end
        return roi
    end

    rod_rois = Dict(
        lab => rod_roi_mask(lab)
            for lab in vcat(collect(ROD_LABELS.Ca), collect(ROD_LABELS.I))
    )

    μ_water_E = Dict(
        E => BS.compute_μ_at_energy(BS.XA.Materials.water, E)
            for E in de_vmi_energies
    )

    function theoretical_hu(material, E::Float64)
        mu = BS.compute_μ_at_energy(material, E)
        return 1000.0 * (mu - μ_water_E[E]) / μ_water_E[E]
    end

    function measured_hu(vmi_vol, label::UInt8)
        roi = rod_rois[label]
        s = 0.0; n = 0
        for z in 1:size(vmi_vol, 3), ci in roi
            s += vmi_vol[ci, z]; n += 1
        end
        return s / n
    end

    out = Dict{Symbol, NamedTuple}()
    for group in (:Ca, :I)
        labels = ROD_LABELS[group]
        n_rods = length(labels)
        n_E = length(de_vmi_energies)
        meas = zeros(Float64, n_rods, n_E)
        theo = zeros(Float64, n_rods, n_E)
        for (i, lab) in pairs(labels)
            mat = materials[Int(lab) + 1]
            for (j, E) in pairs(de_vmi_energies)
                meas[i, j] = measured_hu(vmi_HU_final[E], lab)
                theo[i, j] = theoretical_hu(mat, E)
            end
        end
        out[group] = (
            labels = labels, names = ROD_NAMES[group],
            measured = meas, theoretical = theo,
        )
    end
    out
end;

# ╔═╡ 0600000e-0000-4000-8000-000000000001
md"""
### Per-Rod Regression
"""

# ╔═╡ 0600000e-0000-4000-8000-000000000010
let
    fig = CM.Figure(size = (1180, 580))

    cmap_ca = CM.cgrad(:Oranges, 7; categorical = true)
    cmap_i = CM.cgrad(:GnBu, 7; categorical = true)

    panels = (
        (
            group = :Ca, title = "Calcium rods", subtitle = "50–600 mg/mL",
            cmap = cmap_ca, ylim = (0, 4200),
        ),
        (
            group = :I, title = "Iodine rods", subtitle = "2–20 mg/mL",
            cmap = cmap_i, ylim = (0, 1500),
        ),
    )

    for (col, p) in pairs(panels)
        ax = CM.Axis(
            fig[1, col];
            title = p.title, subtitle = p.subtitle,
            xlabel = "VMI energy (keV)", ylabel = "HU",
            xticks = de_vmi_energies,
            titlesize = 32, subtitlesize = 24,
            xlabelsize = 22, ylabelsize = 22,
            xticklabelsize = 18, yticklabelsize = 16,
        )
        CM.ylims!(ax, p.ylim...)

        d = rod_data[p.group]
        rod_lines = Vector{Any}(undef, length(d.names))
        for i in eachindex(d.names)
            color = p.cmap[i]
            CM.scatterlines!(
                ax, de_vmi_energies, vec(d.measured[i, :]);
                color = color, linewidth = 2.5, markersize = 9,
            )
            CM.lines!(
                ax, de_vmi_energies, vec(d.theoretical[i, :]);
                color = color, linewidth = 1.6, linestyle = :dash,
            )
            rod_lines[i] = CM.LineElement(color = color, linewidth = 2.5)
        end

        style_meas = CM.MarkerElement(
            color = :black, marker = :circle, markersize = 9,
            strokecolor = :black, strokewidth = 1,
        )
        style_theo = CM.LineElement(
            color = :black, linewidth = 1.6, linestyle = :dash,
        )
        CM.axislegend(
            ax,
            vcat([style_meas, style_theo], rod_lines),
            vcat(["Measured", "Theoretical"], collect(d.names));
            position = :rt, framevisible = true, labelsize = 18,
            rowgap = 1, padding = (6, 6, 6, 6),
        )
    end

    fig
end

# ╔═╡ 0600000e-0000-4000-8000-000000000020
md"""
### Linear Regression
"""

# ╔═╡ 0600000e-0000-4000-8000-000000000030
let
    fig = CM.Figure(size = (1000, 1200))

    energy_colors = Dict(
        40.0 => CM.RGBf(0.85, 0.27, 0.1),
        70.0 => CM.RGBf(0.95, 0.65, 0.13),
        100.0 => CM.RGBf(0.13, 0.59, 0.85),
        140.0 => CM.RGBf(0.1, 0.27, 0.65),
    )

    function fit_lr(x::Vector{Float64}, y::Vector{Float64})
        x̄ = mean(x); ȳ = mean(y)
        sxx = sum((x .- x̄) .^ 2)
        sxy = sum((x .- x̄) .* (y .- ȳ))
        β = sxy / sxx
        α = ȳ - β * x̄
        ŷ = α .+ β .* x
        ss_res = sum((y .- ŷ) .^ 2)
        ss_tot = sum((y .- ȳ) .^ 2)
        r² = 1 - ss_res / ss_tot
        rmse = sqrt(ss_res / length(y))
        return (slope = β, intercept = α, r² = r², rmse = rmse)
    end

    panels = (
        (:Ca, "Calcium Rods", "50–600 mg/mL"),
        (:I, "Iodine Rods", "2–20 mg/mL"),
    )

    for (row, (group, title, subtitle)) in pairs(panels)
        d = rod_data[group]
        ax = CM.Axis(
            fig[row, 1];
            title = title, subtitle = subtitle,
            xlabel = "Theoretical HU", ylabel = "Measured HU",
            titlesize = 32, subtitlesize = 24,
            xlabelsize = 22, ylabelsize = 22,
            xticklabelsize = 16, yticklabelsize = 16,
        )

        lim_lo = min(0.0, minimum(d.measured), minimum(d.theoretical))
        lim_hi = max(maximum(d.measured), maximum(d.theoretical)) * 1.05
        CM.lines!(
            ax, [lim_lo, lim_hi], [lim_lo, lim_hi];
            color = :black, linestyle = :dash, linewidth = 2,
            label = "Unity (y = x)",
        )

        for (j, E) in pairs(de_vmi_energies)
            x = Vector{Float64}(vec(d.theoretical[:, j]))
            y = Vector{Float64}(vec(d.measured[:, j]))
            color = energy_colors[E]
            CM.scatter!(ax, x, y; color = color, markersize = 11)

            f = fit_lr(x, y)
            xrange = [minimum(x), maximum(x)]
            yrange = f.intercept .+ f.slope .* xrange
            sign_str = f.intercept ≥ 0 ? "+" : "−"
            label = "$(Int(E)) keV: y = $(round(f.slope, digits = 2))·x " *
                "$(sign_str) $(round(abs(f.intercept), digits = 0)) HU   " *
                "R² = $(round(f.r², digits = 3))   " *
                "RMSE = $(round(f.rmse, digits = 1)) HU"
            CM.lines!(
                ax, xrange, yrange;
                color = color, linewidth = 2, label = label,
            )
        end

        CM.axislegend(
            ax; position = :rb, framevisible = true,
            labelsize = 16, padding = (6, 6, 6, 6), rowgap = 1,
        )
    end

    fig
end

# ╔═╡ 0600000e-0000-4000-8000-000000000040
md"""
### Solid Water HU vs VMI Energy

Mean HU over a **deeply-eroded** (12-px ≈ 8 mm) solid-water ROI at
each VMI energy.  Bars share the per-energy color scheme used in the
linear regression panels above so the eye can match across plots.

The expected value at every energy is ≈ 0 HU.  A roughly constant
offset across keVs means the basis decomposition is recovering
physics with a small uniform bias; an energy-dependent drift would
flag a spectral-shape issue upstream.
"""

# ╔═╡ 0600000e-0000-4000-8000-000000000050
let
    ERODE_PX = 12.0

    mask_2d_raw = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
    sw_bool_raw = (mask_2d_raw .== UInt8(BS.REGION_SOLID_WATER))
    sw_bool = BS.erode_mask_2d(sw_bool_raw; erode_px = ERODE_PX)
    sw_idx = findall(sw_bool)

    if isempty(sw_idx)
        error("solid water ROI empty after $(ERODE_PX)-px erosion; reduce erode_px")
    end

    n_z = size(vmi_HU_final[de_vmi_energies[1]], 3)
    function _mean_hu(vol)
        s = 0.0; n = 0
        for z in 1:n_z, ci in sw_idx
            s += vol[ci, z]; n += 1
        end
        return s / n
    end

    sw_hu_per_keV = [_mean_hu(vmi_HU_final[E]) for E in de_vmi_energies]

    # Same energy → color map as the linear regression panel.
    energy_colors = Dict(
        40.0 => CM.RGBf(0.85, 0.27, 0.1),
        70.0 => CM.RGBf(0.95, 0.65, 0.13),
        100.0 => CM.RGBf(0.13, 0.59, 0.85),
        140.0 => CM.RGBf(0.1, 0.27, 0.65),
    )
    bar_colors = [energy_colors[E] for E in de_vmi_energies]

    n_E = length(de_vmi_energies)

    fig = CM.Figure(size = (1000, 500))
    ax = CM.Axis(
        fig[1, 1];
        title = "Solid Water Mean HU",
        subtitle = "Per VMI Energy",
        xlabel = "VMI Energy (keV)", ylabel = "HU",
        xticks = (collect(1:n_E), ["$(Int(E))" for E in de_vmi_energies]),
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    CM.barplot!(
        ax, 1:n_E, sw_hu_per_keV;
        color = bar_colors,
        strokecolor = :black, strokewidth = 1,
    )
    CM.hlines!(ax, [0.0]; color = :black, linewidth = 1, linestyle = :dash)

    for (k, h) in pairs(sw_hu_per_keV)
        CM.text!(
            ax, k, h;
            text = "$(round(h, digits = 1)) HU",
            align = (:center, h ≥ 0 ? :bottom : :top),
            fontsize = 16, offset = (0, h ≥ 0 ? 4 : -4),
        )
    end

    y_max = max(15.0, 1.2 * maximum(abs, sw_hu_per_keV))
    CM.ylims!(ax, -y_max, y_max)

    fig
end

# ╔═╡ Cell order:
# ╟─06000001-0000-4000-8000-000000000010
# ╟─06000001-0000-4000-8000-000000000020
# ╠═06000001-0000-4000-8000-000000000001
# ╠═06000001-0000-4000-8000-000000000002
# ╠═06000001-0000-4000-8000-000000000003
# ╠═06000001-0000-4000-8000-000000000030
# ╠═06000001-0000-4000-8000-000000000031
# ╠═06000001-0000-4000-8000-000000000040
# ╟─06000001-0000-4000-8000-000000000050
# ╟─06000002-0000-4000-8000-000000000001
# ╠═06000002-0000-4000-8000-000000000010
# ╠═06000002-0000-4000-8000-000000000020
# ╟─06000003-0000-4000-8000-000000000001
# ╠═06000003-0000-4000-8000-000000000010
# ╟─06000004-0000-4000-8000-000000000001
# ╠═06000004-0000-4000-8000-000000000010
# ╠═06000004-0000-4000-8000-000000000020
# ╟─06000005-0000-4000-8000-000000000001
# ╠═06000005-0000-4000-8000-000000000010
# ╠═06000005-0000-4000-8000-000000000020
# ╟─06000006-0000-4000-8000-000000000001
# ╠═06000006-0000-4000-8000-000000000010
# ╠═06000006-0000-4000-8000-000000000020
# ╟─06000006-0000-4000-8000-000000000040
# ╟─06000008-0000-4000-8000-000000000001
# ╠═06000008-0000-4000-8000-000000000010
# ╠═06000008-0000-4000-8000-000000000040
# ╟─06000008-0000-4000-8000-000000000050
# ╟─06000009-0000-4000-8000-000000000001
# ╠═06000009-0000-4000-8000-000000000010
# ╟─06000009-0000-4000-8000-000000000030
# ╟─0600000a-0000-4000-8000-000000000001
# ╠═0600000a-0000-4000-8000-000000000005
# ╠═0600000a-0000-4000-8000-000000000010
# ╟─0600000b-0000-4000-8000-000000000001
# ╠═0600000b-0000-4000-8000-000000000012
# ╠═0600000b-0000-4000-8000-000000000015
# ╠═0600000b-0000-4000-8000-000000000020
# ╟─0600000b-0000-4000-8000-000000000040
# ╟─0600000c-0000-4000-8000-000000000001
# ╠═0600000c-0000-4000-8000-000000000005
# ╠═0600000c-0000-4000-8000-000000000010
# ╟─0600000c-0000-4000-8000-000000000030
# ╟─0600000d-0000-4000-8000-000000000001
# ╠═0600000d-0000-4000-8000-000000000010
# ╠═0600000d-0000-4000-8000-000000000020
# ╠═0600000d-0000-4000-8000-000000000030
# ╟─0600000e-0000-4000-8000-000000000001
# ╟─0600000e-0000-4000-8000-000000000010
# ╟─0600000e-0000-4000-8000-000000000020
# ╟─0600000e-0000-4000-8000-000000000030
# ╟─0600000e-0000-4000-8000-000000000040
# ╟─0600000e-0000-4000-8000-000000000050
