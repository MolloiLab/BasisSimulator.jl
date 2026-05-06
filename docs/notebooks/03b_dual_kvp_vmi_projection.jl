### A Pluto.jl notebook ###
# v0.19.0

# ╔═╡ 05000001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 05000001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 05000001-0000-4000-8000-000000000003
using Statistics: mean, std, quantile

# ╔═╡ 05000001-0000-4000-8000-000000000010
md"""
# 03b · Dual-kVp VMI — projection-domain decomposition

Same dual-kVp acquisition as `03_dual_kvp.jl` (GE Apex Elite GSI, 80 +
140 kVp, Gammex 472), but with a **fully projection-domain** VMI
pipeline.  Structurally:

```
sim 80 kVp  →┐
              ├─→ sinogram-domain RSKR-2ch (TODO: passthrough)
sim 140 kVp →┘            │
                          ▼
              bowtie-aware Cong on (low, high)
                  → sino_iodine, sino_water
                          │
              FBP × 3 (iodine, water, low-kVp)
                          │
              z-direction median filter × 3
                          │
              VMI synthesis (old-nb03 style)
                          │
              optional mono+ (TODO: passthrough)
                          │
              measured-vs-theoretical per-rod regression
              at 40 / 70 / 100 / 140 keV
```

The two big differences vs the archived image-domain pipeline:

1. **Decomposition lives in projection space** — Cong's per-ray solver
   consumes log-line-integrals directly, so the basis fit sees the actual
   polychromatic transmission physics (no pre-FBP linearization needed).
2. **Denoising lives in projection space** — sinograms have a strong
   joint structure across kVps (same view angles, geometry, anatomy),
   so a 2-channel RSKR-equivalent operates on `(p_low, p_high)` pairs
   ray-by-ray.  *(Sinogram-domain kernel is TODO — for now §6 is an
   identity passthrough so we can see the whole pipeline end-to-end.)*

Reference: Cong / De Man / Wang 2022 *J X-Ray Sci Technol*; the
material-basis variant (iodine + water instead of photo + Compton) swaps
Cong's mass-attenuation tables for `(μ/ρ)_iodine, (μ/ρ)_water` and seeds
the Brent step with `water_basis = (a = 0, c = 1)`.
"""

# ╔═╡ 05000001-0000-4000-8000-000000000020
md"""
## Setup
"""

# ╔═╡ 05000001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 05000001-0000-4000-8000-000000000031
import CairoMakie as CM

# ╔═╡ 05000001-0000-4000-8000-000000000040
begin
    GPU_BACKEND = let
        candidates = [
            (:Metal,  "dde4c033-4e86-420c-a63e-0dd931031962", :MtlArray),
            (:CUDA,   "052768ef-5323-5732-b1bb-66c8b64840ba", :CuArray),
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

# ╔═╡ 05000001-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 05000002-0000-4000-8000-000000000001
md"""
## 1. `Phantom` — Gammex Model 472
"""

# ╔═╡ 05000002-0000-4000-8000-000000000010
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,
    n_slices = 16,
    fov_cm   = 35.0,
    z_cm     = 1.0,
);

# ╔═╡ 05000002-0000-4000-8000-000000000020
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 05000003-0000-4000-8000-000000000001
md"""
## 2. `Scanner` — GE Revolution Apex Elite
"""

# ╔═╡ 05000003-0000-4000-8000-000000000010
scanner = BS.Scanner(
    source_to_isocenter = 625.6,
    source_to_detector  = 1100.0,

    detector_rows     = 256,
    detector_cols     = 834,
    detector_row_size = 0.625,
    detector_col_size = 0.6,
    detector_shape    = BS.CURVED_DETECTOR,

    focal_spot_width  = 1.0,
    focal_spot_length = 1.0,
    target_angle      = 10.0,

    flat_filter_material  = :aluminum,
    flat_filter_thickness = 2.5,
    bowtie_filter         = :ge_revolution_large,

    detector_material = :lumex,
    detector_depth    = 3.0,
    fill_factor_row   = 0.9,
    fill_factor_col   = 0.9,

    electronic_noise = 0,
    detection_gain   = 10.0,
);

# ╔═╡ 05000004-0000-4000-8000-000000000001
md"""
## 3. Dual-kVp protocols (rapid-switching duty-cycle math)

| kVp | Instantaneous mA | Duty cycle | Effective mA |
|-----|------------------|------------|--------------|
| 80  | 407              | 0.65       | 264.55       |
| 140 | 405              | 0.35       | 141.75       |
"""

# ╔═╡ 05000004-0000-4000-8000-000000000010
protocol_low = BS.CTProtocol(
    kVp = 80,
    mA  = 407 * 0.65,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 05000004-0000-4000-8000-000000000020
protocol_high = BS.CTProtocol(
    kVp = 140,
    mA  = 405 * 0.35,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 05000005-0000-4000-8000-000000000001
md"""
## 4. `SimOptions` and `ReconOptions`
"""

# ╔═╡ 05000005-0000-4000-8000-000000000010
sim_opts = BS.SimOptions(
    fidelity = :eict,
    seed     = 1234,
);

# ╔═╡ 05000005-0000-4000-8000-000000000020
recon_opts = let
    slice_thickness_mm = 0.625
    n_recon_slices = round(Int, 5.0 / slice_thickness_mm)
    BS.ReconOptions(
        algorithm   = :fdk,
        matrix_size = (512, 512, n_recon_slices),
        fov_cm      = 35.0,
        z_cm        = 0.5,
        filter      = :standard,
    )
end;

# ╔═╡ 05000006-0000-4000-8000-000000000001
md"""
## 5. Forward project — both kVps

Same `simulate!` call as `03_dual_kvp.jl` — scatter is built into the
EICT path (per-ray spatial scatter + Compton + Rayleigh) and we keep
the simulator's noisy line-integral sinogram (`ws.sino_noisy_out`).
"""

# ╔═╡ 05000006-0000-4000-8000-000000000010
sim_low = let
    @info "Simulating: 80 kVp / $(round(protocol_low.mA, digits = 1)) mA-eff (DE low)…"
    ws = BS.create_eict_workspace(scanner, protocol_low, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, scanner, protocol_low, sim_opts, recon_opts)
    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom)
    ws = nothing; GC.gc(true)
    result
end;

# ╔═╡ 05000006-0000-4000-8000-000000000020
sim_high = let
    @info "Simulating: 140 kVp / $(round(protocol_high.mA, digits = 1)) mA-eff (DE high)…"
    ws = BS.create_eict_workspace(scanner, protocol_high, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, scanner, protocol_high, sim_opts, recon_opts)
    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom)
    ws = nothing; GC.gc(true)
    result
end;

# ╔═╡ 05000006-0000-4000-8000-000000000030
md"""
### 5b. Visualize raw sinograms — central row, all views
"""

# ╔═╡ 05000006-0000-4000-8000-000000000040
let
    n_row = size(sim_low.sino, 2)
    mid_r = n_row ÷ 2 + 1

    # sino layout is (n_col, n_row, n_view); transpose so heatmap x = view, y = col
    slice_lo  = permutedims(sim_low.sino[:,  mid_r, :], (2, 1))
    slice_hi  = permutedims(sim_high.sino[:, mid_r, :], (2, 1))

    # Dynamic shared range across both kVps — q1/q99 percentile clipping
    all_v = vcat(vec(slice_lo), vec(slice_hi))
    sino_window = (
        Float64(quantile(all_v, 0.01)),
        Float64(quantile(all_v, 0.99)),
    )

    fig = CM.Figure(size = (1100, 500))
    title_kwargs = (titlesize = 24, subtitlesize = 16)

    panels = (
        (1, 1, "80 kVp",  "raw line-integral sinogram", slice_lo),
        (1, 2, "140 kVp", "raw line-integral sinogram", slice_hi),
    )

    hms = nothing
    for (r, c, ttl, sub, slice) in panels
        ax = CM.Axis(
            fig[r, c]; title = ttl, subtitle = sub,
            xlabel = "view", ylabel = "detector col", title_kwargs...,
        )
        hms = CM.heatmap!(ax, slice; colormap = :viridis, colorrange = sino_window)
    end
    CM.Colorbar(fig[1, 3], hms; label = "log line integral", width = 12, labelsize = 14)
    fig
end

# ╔═╡ 05000007-0000-4000-8000-000000000001
md"""
## 6. Sinogram-domain RSKR-equivalent denoising

**TODO** — passthrough.

The image-domain RSKR-2ch (Clark/Badea 2023) joint-bilateral-filters two
co-registered volumes using shared structure as a guide.  The
projection-domain analog operates on `(sino_low, sino_high)` pairs
**ray-by-ray**: same view angles + geometry + anatomy, similar edge
structure across kVps, so the same joint-bilateral logic should drop
per-pixel kVp-uncorrelated quantum noise without smearing edges.

Sketch of what to wire:

1. Estimate per-channel σ via `BS.mad_haar_σ` on the sinogram-shaped array.
2. Run `BS.joint_bf_2ch_gpu!` on `(sino_low, sino_high)` reshaped to
   `(n_col, n_row, n_view)`.  May need to relax the kernel's neighborhood
   assumption — sinogram pixels don't have isotropic spacing.
3. Or — punt on adapting RSKR and try a separable wavelet-domain joint
   denoise (BM4D-style), closer to clinical raw-data denoisers.

For now the cell returns the inputs verbatim, and §6b is the same
visualization as §5b — just spelled differently in the table footer.
"""

# ╔═╡ 05000007-0000-4000-8000-000000000010
sino_denoised = let
    sino_low_f32  = Float32.(sim_low.sino)
    sino_high_f32 = Float32.(sim_high.sino)

    # TODO — joint sinogram-domain RSKR-2ch goes here.  Identity for now.
    (
        low  = sino_low_f32,
        high = sino_high_f32,
        geom = sim_low.geom,
    )
end;

# ╔═╡ 05000007-0000-4000-8000-000000000020
md"""
### 6b. Visualize denoised sinograms (currently identical to §5b)
"""

# ╔═╡ 05000007-0000-4000-8000-000000000030
let
    n_row = size(sino_denoised.low, 2)
    mid_r = n_row ÷ 2 + 1

    slice_lo = permutedims(sino_denoised.low[:,  mid_r, :], (2, 1))
    slice_hi = permutedims(sino_denoised.high[:, mid_r, :], (2, 1))

    all_v = vcat(vec(slice_lo), vec(slice_hi))
    sino_window = (
        Float64(quantile(all_v, 0.01)),
        Float64(quantile(all_v, 0.99)),
    )

    fig = CM.Figure(size = (1100, 500))
    title_kwargs = (titlesize = 24, subtitlesize = 16)

    panels = (
        (1, 1, "80 kVp",  "post-RSKR-passthrough", slice_lo),
        (1, 2, "140 kVp", "post-RSKR-passthrough", slice_hi),
    )

    hms = nothing
    for (r, c, ttl, sub, slice) in panels
        ax = CM.Axis(
            fig[r, c]; title = ttl, subtitle = sub,
            xlabel = "view", ylabel = "detector col", title_kwargs...,
        )
        hms = CM.heatmap!(ax, slice; colormap = :viridis, colorrange = sino_window)
    end
    CM.Colorbar(fig[1, 3], hms; label = "log line integral", width = 12, labelsize = 14)
    fig
end

# ╔═╡ 05000008-0000-4000-8000-000000000001
md"""
## 7. Bowtie-aware Cong (material basis) on the denoised sinograms

Swap Cong's photoelectric/Compton tables for direct mass-attenuation
tables of iodine and water, and seed the Brent step with `water_basis =
(a = 0, c = 1)` — i.e., the low-attenuation side becomes pure water.
The output sinograms now read as **per-ray basis line integrals**:

* `sino_iodine = ∫ c_iodine(r) dr`  — iodine basis density × path
* `sino_water  = ∫ c_water(r)  dr`  — water  basis density × path

The bowtie path uses `resolve_source_spectrum_with_bowtie` — Cong
detects the per-ray 3D `ŵ` automatically (`ndims(ŵ) == 3`) and uses
the right column-row spectrum for each detector position.
"""

# ╔═╡ 05000008-0000-4000-8000-000000000010
material_basis = let
    e_L, ŵ_L = BS.resolve_source_spectrum_with_bowtie(
        sim_opts, protocol_low; scanner = scanner, geom = sim_low.geom,
    )
    e_H, ŵ_H = BS.resolve_source_spectrum_with_bowtie(
        sim_opts, protocol_high; scanner = scanner, geom = sim_high.geom,
    )

    ŵ_L_f32 = Float32.(ŵ_L ./ sum(ŵ_L; dims = ndims(ŵ_L)))
    ŵ_H_f32 = Float32.(ŵ_H ./ sum(ŵ_H; dims = ndims(ŵ_H)))

    iodine_mat = BS.XA.Elements.Iodine     # pure-element iodine basis
    water_mat  = BS.XA.Materials.water     # H₂O compound water basis

    p_L = Float32[Float32(BS.compute_mass_μ_at_energy(iodine_mat, Float64(E))) for E in e_L]
    q_L = Float32[Float32(BS.compute_mass_μ_at_energy(water_mat,  Float64(E))) for E in e_L]
    p_H = Float32[Float32(BS.compute_mass_μ_at_energy(iodine_mat, Float64(E))) for E in e_H]
    q_H = Float32[Float32(BS.compute_mass_μ_at_energy(water_mat,  Float64(E))) for E in e_H]

    (ŵ_L = ŵ_L_f32, p_L = p_L, q_L = q_L,
     ŵ_H = ŵ_H_f32, p_H = p_H, q_H = q_H)
end;

# ╔═╡ 05000008-0000-4000-8000-000000000020
sino_basis = let
    sino_low_gpu  = to_gpu(sino_denoised.low)
    sino_high_gpu = to_gpu(sino_denoised.high)

    sino_y = similar(sino_low_gpu)   # iodine basis line integrals
    sino_c = similar(sino_low_gpu)   # water  basis line integrals
    fill!(sino_y, 0f0); fill!(sino_c, 0f0)

    cong_ws = BS.create_cong_workspace(sino_low_gpu, material_basis)
    BS.apply_cong!(
        cong_ws, sino_y, sino_c, sino_low_gpu, sino_high_gpu;
        water_basis = (a = 0f0, c = 1f0),
    )

    result = (
        sino_iodine = Array(sino_y),
        sino_water  = Array(sino_c),
        sino_low    = sino_denoised.low,    # kept for HU calibration anchor
        geom        = sino_denoised.geom,
    )
    sino_low_gpu = nothing; sino_high_gpu = nothing
    sino_y = nothing; sino_c = nothing; cong_ws = nothing
    GC.gc(true)
    result
end;

# ╔═╡ 05000008-0000-4000-8000-000000000030
md"""
### 7b. Visualize basis sinograms — central row, all views
"""

# ╔═╡ 05000008-0000-4000-8000-000000000040
let
    n_row = size(sino_basis.sino_iodine, 2)
    mid_r = n_row ÷ 2 + 1

    fig = CM.Figure(size = (1100, 500))
    title_kwargs = (titlesize = 24, subtitlesize = 16)

    _qrange(arr) = (
        Float64(quantile(vec(arr), 0.01)),
        Float64(quantile(vec(arr), 0.99)),
    )

    slice_iod = permutedims(sino_basis.sino_iodine[:, mid_r, :], (2, 1))
    slice_wat = permutedims(sino_basis.sino_water[:,  mid_r, :], (2, 1))

    # (panel_col, cbar_col) — colorbar always immediately right of its panel
    panels = (
        (1, 1, 2, "iodine basis sinogram", "∫ c_iodine(r) dr",
            slice_iod, _qrange(slice_iod)),
        (1, 3, 4, "water basis sinogram",  "∫ c_water(r) dr",
            slice_wat, _qrange(slice_wat)),
    )

    for (r, panel_c, cbar_c, ttl, sub, slice, range) in panels
        ax = CM.Axis(
            fig[r, panel_c]; title = ttl, subtitle = sub,
            xlabel = "view", ylabel = "detector col", title_kwargs...,
        )
        hm = CM.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        CM.Colorbar(fig[r, cbar_c], hm; width = 12, labelsize = 14)
    end
    fig
end

# ╔═╡ 05000009-0000-4000-8000-000000000001
md"""
## 8. FBP — iodine, water, and low-kVp

Three FDK passes with `BS.SoftFilter()`.  The iodine + water
reconstructions land in basis-density units (g/cm³ × scaling) directly,
no post-decomposition step.  The low-kVp recon is kept for §10's HU
anchor.
"""

# ╔═╡ 05000009-0000-4000-8000-000000000010
basis_volumes = let
    matrix_size = recon_opts.matrix_size
    geom        = sino_basis.geom

    function _fbp(sino_cpu)
        sino_gpu = to_gpu(Float32.(sino_cpu))
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, geom, matrix_size; filter = BS.SoftFilter(),
        )
        recon = Array(BS.reconstruct!(ws, sino_gpu, geom, matrix_size))
        ws = nothing; sino_gpu = nothing
        GC.gc(true)
        Float32.(recon)
    end

    (
        vol_iodine_raw = _fbp(sino_basis.sino_iodine),
        vol_water_raw  = _fbp(sino_basis.sino_water),
        vol_low_μ_raw  = _fbp(sino_basis.sino_low),
        geom           = geom,
    )
end;

# ╔═╡ 05000009-0000-4000-8000-000000000020
md"""
### 8b. Visualize raw FBP volumes (mid-z slice)
"""

# ╔═╡ 05000009-0000-4000-8000-000000000030
let
    fig = CM.Figure(size = (1500, 500))
    title_kwargs = (titlesize = 24, subtitlesize = 16)

    mid = size(basis_volumes.vol_iodine_raw, 3) ÷ 2

    _qrange(arr) = (
        Float64(quantile(vec(arr), 0.01)),
        Float64(quantile(vec(arr), 0.99)),
    )

    slice_iod  = basis_volumes.vol_iodine_raw[:, :, mid]
    slice_wat  = basis_volumes.vol_water_raw[:,  :, mid]
    slice_lokv = basis_volumes.vol_low_μ_raw[:,  :, mid]

    # (row, panel_col, cbar_col, ...)
    panels = (
        (1, 1, 2, "iodine basis", "FBP raw, g/cm³ × scale", slice_iod,  _qrange(slice_iod)),
        (1, 3, 4, "water basis",  "FBP raw, g/cm³ × scale", slice_wat,  _qrange(slice_wat)),
        (1, 5, 6, "low-kVp μ",    "FBP raw, cm⁻¹",          slice_lokv, _qrange(slice_lokv)),
    )

    for (r, panel_c, cbar_c, ttl, sub, slice, range) in panels
        ax = CM.Axis(
            fig[r, panel_c]; title = ttl, subtitle = sub,
            aspect = CM.DataAspect(), title_kwargs...,
        )
        hm = CM.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        CM.hidedecorations!(ax)
        CM.Colorbar(fig[r, cbar_c], hm; width = 12, labelsize = 14)
    end
    fig
end

# ╔═╡ 0500000a-0000-4000-8000-000000000001
md"""
## 9. z-direction median filter

1D median along z, per `(x, y)` voxel column.  `adjacent_slices = 1`
⇒ 3-slice window (1 above + center + 1 below).  Cheap streak/outlier
suppression that exploits the Gammex 472's z-invariance — zero
in-plane resolution loss.

Bump `Z_MEDIAN_ADJACENT` to widen the window:

| `adjacent_slices` | window size |
|-------------------|-------------|
| `0`               | identity    |
| `1` (default)     | 3 slices    |
| `2`               | 5 slices    |
"""

# ╔═╡ 0500000a-0000-4000-8000-000000000005
Z_MEDIAN_ADJACENT = 3;

# ╔═╡ 0500000a-0000-4000-8000-000000000010
basis_z = let
    (
        vol_iodine = BS.apply_median_z(basis_volumes.vol_iodine_raw;
                                       adjacent_slices = Z_MEDIAN_ADJACENT),
        vol_water  = BS.apply_median_z(basis_volumes.vol_water_raw;
                                       adjacent_slices = Z_MEDIAN_ADJACENT),
        vol_low_μ  = BS.apply_median_z(basis_volumes.vol_low_μ_raw;
                                       adjacent_slices = Z_MEDIAN_ADJACENT),
        geom       = basis_volumes.geom,
    )
end;

# ╔═╡ 0500000a-0000-4000-8000-000000000020
md"""
### 9b. Visualize z-median output (mid-z slice)
"""

# ╔═╡ 0500000a-0000-4000-8000-000000000030
let
    fig = CM.Figure(size = (1500, 500))
    title_kwargs = (titlesize = 24, subtitlesize = 16)

    mid = size(basis_z.vol_iodine, 3) ÷ 2

    _qrange(arr) = (
        Float64(quantile(vec(arr), 0.01)),
        Float64(quantile(vec(arr), 0.99)),
    )

    slice_iod  = basis_z.vol_iodine[:, :, mid]
    slice_wat  = basis_z.vol_water[:,  :, mid]
    slice_lokv = basis_z.vol_low_μ[:,  :, mid]

    panels = (
        (1, 1, 2, "iodine basis", "z-median, g/cm³ × scale", slice_iod,  _qrange(slice_iod)),
        (1, 3, 4, "water basis",  "z-median, g/cm³ × scale", slice_wat,  _qrange(slice_wat)),
        (1, 5, 6, "low-kVp μ",    "z-median, cm⁻¹",          slice_lokv, _qrange(slice_lokv)),
    )

    for (r, panel_c, cbar_c, ttl, sub, slice, range) in panels
        ax = CM.Axis(
            fig[r, panel_c]; title = ttl, subtitle = sub,
            aspect = CM.DataAspect(), title_kwargs...,
        )
        hm = CM.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        CM.hidedecorations!(ax)
        CM.Colorbar(fig[r, cbar_c], hm; width = 12, labelsize = 14)
    end
    fig
end

# ╔═╡ 0500000b-0000-4000-8000-000000000001
md"""
## 10. VMI synthesis — textbook 2-basis monoenergetic

For each VMI energy `E`:

```
μ(E)              = c_water(r) · (μ/ρ)_water(E) + c_iodine(r) · (μ/ρ)_iodine(E)
μ_water_anchor(E) = ⟨c_water⟩_SW · (μ/ρ)_water(E) + ⟨c_iodine⟩_SW · (μ/ρ)_iodine(E)
HU(E)             = 1000 · (μ(E) − μ_water_anchor(E)) / μ_water_anchor(E)
```

The numerator is the canonical 2-basis linear mix from McCollough 2015.
The denominator is **measured from the recon's solid-water ROI** —
synth-evaluated at energy `E` using the mean basis values
`(⟨c_water⟩_SW, ⟨c_iodine⟩_SW)` over the phantom's solid-water region.

Why not `μρ_water(E)` directly? Cong's polychromatic forward model
uses the bowtie-aware source spectrum, but the simulator also applies
detector η(E), pile-up, residual scatter, etc.  Those biases push
`vol_water` in the SW background a few % below 1.0, and the textbook
mono divisor undercalibrates HU at every keV.  Anchoring the divisor
to the basis-decomp's *own* SW reading absorbs that bias — solid water
reads 0 HU exactly at every VMI keV, and rod HU slopes vs theory
recover.  Implicit phantom-size handling: the SW ROI has already seen
phantom hardening / scatter / detector response baked into `basis_z`.

VMI grid: 40, 70, 100, 140 keV.
"""

# ╔═╡ 0500000b-0000-4000-8000-000000000010
solid_water_basis = let
    # Mid-slice SW mask, broadcast across all recon z (Gammex 472 SW
    # background is z-invariant).  Then DEEPLY erode (σ = 12 px ≈ 8 mm
    # at 0.683 mm/px) so we sample only deep-interior SW voxels — well
    # clear of rod edges where partial-volume mixing with iodine / Ca
    # contaminates the basis means.
    ERODE_PX = 12.0

    mask_2d_raw = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
    sw_bool_raw = (mask_2d_raw .== UInt8(BS.REGION_SOLID_WATER))
    sw_bool     = BS.erode_mask_2d(sw_bool_raw; erode_px = ERODE_PX)

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
        s / n
    end

    c_w = Float64(_mean(basis_z.vol_water))
    c_i = Float64(_mean(basis_z.vol_iodine))
    @info "solid_water_basis: ⟨c_water⟩_SW = $(round(c_w, digits = 4)) g/cm³, " *
          "⟨c_iodine⟩_SW = $(round(c_i, digits = 6)) g/cm³"

    (c_water = c_w, c_iodine = c_i, n_voxels = length(sw_idx) * n_z,
     mask_2d = collect(sw_bool))   # for downstream viz
end;

# ╔═╡ 0500000b-0000-4000-8000-000000000015
de_vmi_energies = [40.0, 70.0, 100.0, 140.0];

# ╔═╡ 0500000b-0000-4000-8000-000000000020
vmi_HU_by_keV = let
    iodine_mat = BS.XA.Elements.Iodine
    water_mat  = BS.XA.Materials.water

    out = Dict{Float64, Array{Float32, 3}}()
    for E in de_vmi_energies
        μρ_w = Float32(BS.compute_mass_μ_at_energy(water_mat,  E))
        μρ_I = Float32(BS.compute_mass_μ_at_energy(iodine_mat, E))
        μ_E  = @. basis_z.vol_water * μρ_w + basis_z.vol_iodine * μρ_I

        # Phantom-aware HU divisor — synth at the SW ROI's measured basis means.
        μ_water_anchor = Float32(
            solid_water_basis.c_water  * Float64(μρ_w) +
            solid_water_basis.c_iodine * Float64(μρ_I)
        )

        # Mono μ_water(E) = μρ_water(E) numerically (since ρ_water = 1 g/cm³).
        μ_water_mono = μρ_w
        Δ_pct = 100.0 * (Float64(μ_water_anchor) - Float64(μ_water_mono)) /
                Float64(μ_water_mono)
        @info "VMI synth @ $(Int(E)) keV: μ_water_anchor = " *
              "$(round(Float64(μ_water_anchor), digits = 5)) cm⁻¹  " *
              "(mono μρ_water = $(round(Float64(μ_water_mono), digits = 5)),  " *
              "Δ = $(round(Δ_pct, digits = 2))%)"

        HU_E = @. 1000f0 * (μ_E - μ_water_anchor) / μ_water_anchor
        out[E] = HU_E
    end
    out
end;

# ╔═╡ 0500000b-0000-4000-8000-000000000030
md"""
### 10b. Visualize VMI HU stack (mid-z slice)
"""

# ╔═╡ 0500000b-0000-4000-8000-000000000040
let
    HU_window = (-200, 500)

    fig = CM.Figure(size = (1100, 1000))
    title_kwargs = (titlesize = 24, subtitlesize = 16)

    sample = vmi_HU_by_keV[40.0]
    mid    = size(sample, 3) ÷ 2

    for (k, E) in enumerate(de_vmi_energies)
        r = ((k - 1) ÷ 2) + 1
        c = ((k - 1) % 2) + 1
        ax = CM.Axis(
            fig[r, c]; title = "$(Int(E)) keV",
            subtitle = "VMI · projection-domain Cong",
            aspect = CM.DataAspect(), title_kwargs...,
        )
        CM.heatmap!(
            ax, vmi_HU_by_keV[E][:, :, mid];
            colormap = :grays, colorrange = HU_window,
        )
        CM.hidedecorations!(ax)
    end
    # Explicit colorbar binding — same colormap + colorrange as every panel,
    # no auto-extraction from a plot object.
    CM.Colorbar(
        fig[1:2, 3];
        colormap = :grays, colorrange = HU_window,
        label = "HU", width = 14, labelsize = 18,
    )

    CM.save(
        joinpath(@__DIR__, "..", "assets", "dual_kvp_vmi_projection_grid.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0500000c-0000-4000-8000-000000000001
md"""
## 11. Mono+ post-processing (Grant 2014)

Frequency-split rule from Grant et al. 2014:

```
Mono+(E)     = LP_σ(VMI_E) + VMI_opt − LP_σ(VMI_opt)
Mono+(E_opt) = VMI_opt   (identity at the noise-optimal anchor)
```

Low frequencies come from the noise-optimal anchor `E_opt = 70 keV`;
high frequencies (edges, fine detail) come from the target energy `E`.

`σ_vmi_lp_px` is **per-keV** and pairs element-wise with
`de_vmi_energies` — tweak it to dial each keV's noise/contrast
trade-off independently.  Conventions:

* `σ = 0`  → identity at that energy (no LP, no FFT — `Mono+(E) = VMI_E`).
* `σ > 0` → LP band replaced with the anchor's LP (smoother, less noisy).
* Larger σ → broader anchor-driven low-frequency content; edges blur
  toward the anchor.

Default `[1.0, 0.0, 1.0, 1.0]`: smooth 40 / 100 / 140 toward the 70-keV
anchor (reduces low-keV iodine speckle + high-keV photon-starvation
streaks); leave 70 keV identity by definition.
"""

# ╔═╡ 0500000c-0000-4000-8000-000000000005
# Per-keV Gaussian LP σ in pixels — paired with `de_vmi_energies` order.
# σ = 0  ⇒ identity at that energy (Mono+(E) = VMI_E exactly, no FFT).
# σ > 0  ⇒ that energy's LP band is replaced with the 70-keV anchor's LP.
# Edit these to tune per-energy noise/contrast trade-off.
σ_vmi_lp_px = Float64[1.0, 0.0, 1.0, 1.0];   # ↔ (40, 70, 100, 140) keV

# ╔═╡ 0500000c-0000-4000-8000-000000000010
vmi_HU_final = let
    volumes = [vmi_HU_by_keV[E] for E in de_vmi_energies]

    ws = BS.create_mono_plus_workspace(volumes[1];
                                       n_energies = length(de_vmi_energies))
    BS.apply_mono_plus!(
        ws, volumes, de_vmi_energies;
        E_noise_opt = 70.0,
        σ_lp_px     = σ_vmi_lp_px,
        verbose     = true,
    )

    # ws.out_vols is reused on subsequent apply_mono_plus! calls — copy
    # into a Dict so downstream cells (Results plots) hold their own arrays.
    out = Dict{Float64, Array{Float32, 3}}()
    for (i, E) in enumerate(de_vmi_energies)
        out[E] = copy(ws.out_vols[i])
    end
    ws = nothing; GC.gc(true)
    out
end;

# ╔═╡ 0500000c-0000-4000-8000-000000000020
md"""
### 11b. Visualize mono+ HU stack (mid-z slice)
"""

# ╔═╡ 0500000c-0000-4000-8000-000000000030
let
    HU_window = (-200, 500)

    fig = CM.Figure(size = (1100, 1000))
    title_kwargs = (titlesize = 24, subtitlesize = 16)

    sample = vmi_HU_final[40.0]
    mid    = size(sample, 3) ÷ 2

    for (k, E) in enumerate(de_vmi_energies)
        r = ((k - 1) ÷ 2) + 1
        c = ((k - 1) % 2) + 1
        ax = CM.Axis(
            fig[r, c]; title = "$(Int(E)) keV",
            subtitle = "Mono+ · σ = $(σ_vmi_lp_px[k]) px · anchor 70 keV",
            aspect = CM.DataAspect(), title_kwargs...,
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
        label = "HU", width = 14, labelsize = 18,
    )

    CM.save(
        joinpath(@__DIR__, "..", "assets", "dual_kvp_vmi_projection_monoplus.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0500000d-0000-4000-8000-000000000001
md"""
## Results — measured vs theoretical, per rod

For each rod and each of the canonical four VMI energies (40 / 70 / 100 /
140 keV):

* **Measured HU** = mean over (8-px-radius circular ROI at the rod
  centroid) × (all z slices).  Same 8-px core trick as the archived
  image-domain pipeline — small enough to dodge partial-volume bleed
  at the rod edge.
* **Theoretical HU** = `1000 · (μ_r(E) − μ_water(E)) / μ_water(E)` from
  XrayAttenuation directly, via `BS.compute_μ_at_energy(material_r, E)`
  — pure physics, no fitting, no calibration assumption.

Two panels:

* **Calcium rods** (50 – 600 mg/mL): Compton-dominated at clinical keV,
  smooth roll-off as E increases.
* **Iodine rods** (2 – 20 mg/mL): K-edge at 33.2 keV, so 40 keV catches
  the photoelectric edge and amplifies HU dramatically vs the 70+ keV
  plateau.

Solid line = measured.  Dashed line = theoretical.  Tight overlay means
the projection-domain Cong + FBP + z-median + VMI synth chain is
recovering the underlying physics correctly.
"""

# ╔═╡ 0500000d-0000-4000-8000-000000000002
md"""
### Water ROI sanity check — phantom-aware HU divisor

Left: the deeply-eroded solid-water ROI (12-px erosion ≈ 8 mm) overlaid
in red on the **70 keV Mono+ slice** — that's exactly the voxels feeding
`solid_water_basis.c_water` / `c_iodine`, and therefore the per-keV
`μ_water_anchor(E)` divisor.

Right: mean HU over that ROI vs VMI energy.  By construction, every bar
should sit very close to **0 HU** — solid water is anchored to 0 by the
phantom-aware divisor.  Any residual deviation comes from Mono+'s
LP/HP frequency split shifting the SW baseline at non-anchor energies
(the 70 keV bar is identity by design and should be exactly 0).
"""

# ╔═╡ 0500000d-0000-4000-8000-000000000005
let
    fig = CM.Figure(size = (1180, 540))
    title_kwargs = (titlesize = 26, subtitlesize = 18)

    # ─── Left panel — 70 keV Mono+ slice + eroded SW ROI overlay ───────
    HU_window = (-200, 500)
    mid = size(vmi_HU_final[70.0], 3) ÷ 2
    bg  = vmi_HU_final[70.0][:, :, mid]

    # nb05-style overlay: Float32 array with NaN outside the mask, single
    # built-in colormap symbol + alpha + nan_color = (:white, 0.0).
    overlay = Float32[b ? 1.0f0 : NaN32 for b in solid_water_basis.mask_2d]

    ax1 = CM.Axis(
        fig[1, 1];
        title = "Eroded Water Region on 70 keV Mono+",
        subtitle = "$(solid_water_basis.n_voxels) voxels · 12-px erosion",
        aspect = CM.DataAspect(),
        title_kwargs...,
    )
    CM.heatmap!(ax1, bg; colormap = :grays, colorrange = HU_window)
    CM.heatmap!(
        ax1, overlay;
        colormap = :reds, alpha = 0.5,
        nan_color = (:white, 0.0),
    )
    CM.hidedecorations!(ax1)

    # ─── Right panel — mean SW HU vs VMI energy (bar) ──────────────────
    sw_idx = findall(solid_water_basis.mask_2d)
    n_z = size(vmi_HU_final[70.0], 3)
    function _mean_hu(vol)
        s = 0.0; n = 0
        for z in 1:n_z, ci in sw_idx
            s += vol[ci, z]; n += 1
        end
        s / n
    end
    sw_hu_per_keV = [_mean_hu(vmi_HU_final[E]) for E in de_vmi_energies]

    # cgrad → Vector{RGBAf} for barplot's color kwarg
    n_E = length(de_vmi_energies)
    bar_colors = [CM.cgrad(:plasma, n_E; categorical = true)[i] for i in 1:n_E]

    ax2 = CM.Axis(
        fig[1, 2];
        title = "Water Region Mean HU vs VMI keV",
        subtitle = "anchored to 0 by construction",
        xlabel = "VMI energy (keV)", ylabel = "HU",
        xticks = (collect(1:n_E), ["$(Int(E))" for E in de_vmi_energies]),
        title_kwargs...,
    )
    CM.barplot!(
        ax2, 1:n_E, sw_hu_per_keV;
        color = bar_colors,
        strokecolor = :black, strokewidth = 1,
    )
    CM.hlines!(ax2, [0.0]; color = :black, linewidth = 1, linestyle = :dash)

    # Numeric labels above each bar
    for (k, h) in pairs(sw_hu_per_keV)
        CM.text!(
            ax2, k, h;
            text = "$(round(h, digits = 1)) HU",
            align = (:center, h ≥ 0 ? :bottom : :top),
            fontsize = 14, offset = (0, h ≥ 0 ? 4 : -4),
        )
    end

    # Symmetric y-limits around 0 with sane minimum span
    y_max = max(15.0, 1.2 * maximum(abs, sw_hu_per_keV))
    CM.ylims!(ax2, -y_max, y_max)

    CM.save(
        joinpath(@__DIR__, "..", "assets", "vmi_water_roi_check.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0500000d-0000-4000-8000-000000000010
ROD_LABELS = (
    Ca = (UInt8(10), UInt8(11), UInt8(12), UInt8(13), UInt8(14), UInt8(15), UInt8(16)),
    I  = (UInt8(20), UInt8(21), UInt8(22), UInt8(23), UInt8(24), UInt8(25), UInt8(26)),
);

# ╔═╡ 0500000d-0000-4000-8000-000000000020
ROD_NAMES = (
    Ca = ("50 mg/mL", "100 mg/mL", "200 mg/mL", "300 mg/mL", "400 mg/mL", "500 mg/mL", "600 mg/mL"),
    I  = ("2.0 mg/mL", "2.5 mg/mL", "5.0 mg/mL", "7.5 mg/mL", "10.0 mg/mL", "15.0 mg/mL", "20.0 mg/mL"),
);

# ╔═╡ 0500000d-0000-4000-8000-000000000030
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
        roi
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
        μ = BS.compute_μ_at_energy(material, E)
        return 1000.0 * (μ - μ_water_E[E]) / μ_water_E[E]
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
        n_E    = length(de_vmi_energies)
        meas   = zeros(Float64, n_rods, n_E)
        theo   = zeros(Float64, n_rods, n_E)
        for (i, lab) in pairs(labels)
            mat = materials[Int(lab) + 1]   # mask_value + 1
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

# ╔═╡ 0500000e-0000-4000-8000-000000000001
md"""
### Per-rod regression plot — solid (measured) vs dashed (theoretical)
"""

# ╔═╡ 0500000e-0000-4000-8000-000000000010
let
    fig = CM.Figure(size = (1180, 580))

    # Two panels share the same color scheme: warm for Ca, cool for I, with
    # rod-concentration index → color shade
    cmap_ca = CM.cgrad(:Oranges, 7; categorical = true)
    cmap_i  = CM.cgrad(:GnBu,    7; categorical = true)

    panels = (
        (
            group = :Ca, title = "Calcium rods",
            subtitle = "50–600 mg/mL",
            cmap = cmap_ca, ylim = (0, 4200),
        ),
        (
            group = :I, title = "Iodine rods",
            subtitle = "2–20 mg/mL",
            cmap = cmap_i, ylim = (0, 1500),
        ),
    )

    for (col, p) in pairs(panels)
        ax = CM.Axis(
            fig[1, col];
            title = p.title,
            subtitle = p.subtitle,
            xlabel = "VMI energy (keV)",
            ylabel = "HU",
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

        # Two-section legend: line-style key on top (measured / theoretical)
        # then rod concentrations below.  Black swatches for the style row
        # so the dashed-vs-solid contrast is unmistakable regardless of
        # which rod color the eye lands on first.
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

    CM.save(
        joinpath(@__DIR__, "..", "assets", "vmi_vs_theoretical.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0500000e-0000-4000-8000-000000000020
md"""
### Linear regression — measured HU vs theoretical HU

Same data, scattered as `(theoretical, measured)` per rod-energy pair
with a per-energy least-squares line and the y = x identity.

* Slope ≈ 1, intercept ≈ 0, R² ≈ 1 → pipeline recovers physics.
* Slope ≠ 1 → multiplicative cal mismatch (mass-attn tables, basis
  scaling).
* Intercept ≠ 0 → additive offset (residual cup, water-baseline shift).
* Low R² → non-linear distortion (partial volume, decomp non-linearity).

Ca and I are split because Ca lives at much higher HU and would
otherwise dominate a shared-axis fit.
"""

# ╔═╡ 0500000e-0000-4000-8000-000000000030
let
    fig = CM.Figure(size = (1180, 620))

    # One color per VMI energy (cool→warm sweep)
    energy_colors = Dict(
        40.0  => CM.RGBf(0.85, 0.27, 0.10),
        70.0  => CM.RGBf(0.95, 0.65, 0.13),
        100.0 => CM.RGBf(0.13, 0.59, 0.85),
        140.0 => CM.RGBf(0.10, 0.27, 0.65),
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
        return (slope = β, intercept = α, r² = r²)
    end

    panels = ((:Ca, "Calcium Rods", "50–600 mg/mL"),
              (:I,  "Iodine Rods",  "2–20 mg/mL"))

    for (col, (group, title, subtitle)) in pairs(panels)
        d = rod_data[group]
        ax = CM.Axis(
            fig[1, col];
            title = title,
            subtitle = subtitle,
            xlabel = "Theoretical HU",
            ylabel = "Measured HU",
            aspect = CM.AxisAspect(1),
            titlesize = 32, subtitlesize = 24,
            xlabelsize = 22, ylabelsize = 22,
            xticklabelsize = 16, yticklabelsize = 16,
        )

        # y = x identity reference — drawn FIRST so the per-energy fits
        # paint over it; bumped to a darker dashed line so the "perfect
        # agreement" reference is unmistakable against the per-energy fits.
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
            label = "$(Int(E)) keV: y = $(round(f.slope, digits = 2))·x " *
                "$(f.intercept ≥ 0 ? "+" : "−") $(round(abs(f.intercept), digits = 0))" *
                "   R² = $(round(f.r², digits = 3))"
            CM.lines!(
                ax, xrange, yrange;
                color = color, linewidth = 2, label = label,
            )
        end

        CM.axislegend(
            ax; position = :rb, framevisible = true,
            labelsize = 18, padding = (6, 6, 6, 6), rowgap = 1,
        )
    end

    CM.save(
        joinpath(@__DIR__, "..", "assets", "vmi_regression.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0500000f-0000-4000-8000-000000000001
md"""
## Summary

```
sim 80 + 140 kVp (scatter-corrected line-integral sinograms)
   → sinogram-domain RSKR-2ch          (TODO — passthrough)
   → bowtie-aware Cong (iodine/water)  (✓ live, src/reconstruction/vmi/cong.jl)
   → FBP × 3 (iodine, water, low-kVp)
   → z-direction median filter × 3
   → textbook 2-basis VMI synth (mono-water divisor per keV)
   → Mono+ (Grant 2014, per-keV σ via σ_vmi_lp_px)
   → measured-vs-theoretical per-rod regression at 40 / 70 / 100 / 140 keV
```

When the two TODO blocks land, this notebook should produce the same
HU-quantitative VMIs the old image-domain pipeline did, but with the
denoising and decomposition both happening upstream of FBP — which
should propagate fewer streak / partial-volume artifacts into the
basis maps and a cleaner low-keV mono+.
"""

# ╔═╡ Cell order:
# ╟─05000001-0000-4000-8000-000000000010
# ╟─05000001-0000-4000-8000-000000000020
# ╠═05000001-0000-4000-8000-000000000001
# ╠═05000001-0000-4000-8000-000000000002
# ╠═05000001-0000-4000-8000-000000000003
# ╠═05000001-0000-4000-8000-000000000030
# ╠═05000001-0000-4000-8000-000000000031
# ╠═05000001-0000-4000-8000-000000000040
# ╟─05000001-0000-4000-8000-000000000050
# ╟─05000002-0000-4000-8000-000000000001
# ╠═05000002-0000-4000-8000-000000000010
# ╠═05000002-0000-4000-8000-000000000020
# ╟─05000003-0000-4000-8000-000000000001
# ╠═05000003-0000-4000-8000-000000000010
# ╟─05000004-0000-4000-8000-000000000001
# ╠═05000004-0000-4000-8000-000000000010
# ╠═05000004-0000-4000-8000-000000000020
# ╟─05000005-0000-4000-8000-000000000001
# ╠═05000005-0000-4000-8000-000000000010
# ╠═05000005-0000-4000-8000-000000000020
# ╟─05000006-0000-4000-8000-000000000001
# ╠═05000006-0000-4000-8000-000000000010
# ╠═05000006-0000-4000-8000-000000000020
# ╟─05000006-0000-4000-8000-000000000030
# ╟─05000006-0000-4000-8000-000000000040
# ╟─05000007-0000-4000-8000-000000000001
# ╠═05000007-0000-4000-8000-000000000010
# ╟─05000007-0000-4000-8000-000000000020
# ╟─05000007-0000-4000-8000-000000000030
# ╟─05000008-0000-4000-8000-000000000001
# ╠═05000008-0000-4000-8000-000000000010
# ╠═05000008-0000-4000-8000-000000000020
# ╟─05000008-0000-4000-8000-000000000030
# ╟─05000008-0000-4000-8000-000000000040
# ╟─05000009-0000-4000-8000-000000000001
# ╠═05000009-0000-4000-8000-000000000010
# ╟─05000009-0000-4000-8000-000000000020
# ╟─05000009-0000-4000-8000-000000000030
# ╟─0500000a-0000-4000-8000-000000000001
# ╠═0500000a-0000-4000-8000-000000000005
# ╠═0500000a-0000-4000-8000-000000000010
# ╟─0500000a-0000-4000-8000-000000000020
# ╟─0500000a-0000-4000-8000-000000000030
# ╟─0500000b-0000-4000-8000-000000000001
# ╠═0500000b-0000-4000-8000-000000000010
# ╠═0500000b-0000-4000-8000-000000000015
# ╠═0500000b-0000-4000-8000-000000000020
# ╟─0500000b-0000-4000-8000-000000000030
# ╟─0500000b-0000-4000-8000-000000000040
# ╟─0500000c-0000-4000-8000-000000000001
# ╠═0500000c-0000-4000-8000-000000000005
# ╠═0500000c-0000-4000-8000-000000000010
# ╟─0500000c-0000-4000-8000-000000000020
# ╟─0500000c-0000-4000-8000-000000000030
# ╟─0500000d-0000-4000-8000-000000000001
# ╟─0500000d-0000-4000-8000-000000000002
# ╟─0500000d-0000-4000-8000-000000000005
# ╠═0500000d-0000-4000-8000-000000000010
# ╠═0500000d-0000-4000-8000-000000000020
# ╠═0500000d-0000-4000-8000-000000000030
# ╟─0500000e-0000-4000-8000-000000000001
# ╟─0500000e-0000-4000-8000-000000000010
# ╟─0500000e-0000-4000-8000-000000000020
# ╟─0500000e-0000-4000-8000-000000000030
# ╟─0500000f-0000-4000-8000-000000000001
