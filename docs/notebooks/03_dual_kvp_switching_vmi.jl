### A Pluto.jl notebook ###
# v0.2.3

using Markdown
using InteractiveUtils

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
# Dual-kVp Rapid-Switching Virtual Monoenergetic Imaging

GE Revolution Apex Elite rapid-kVp-switching simulation (80 + 140 kVp,
Gammex 472 phantom) with a **fully projection-domain** VMI pipeline —
every denoising and decomposition step operates on log-line-integrals
before the reconstruction runs.

```
Simulate 80 / 140 kVp
   → Jensen debias + projection-domain SVD-bilateral denoise
   → Projection-domain material decomposition  (Cong; iodine + water)
   → FBP × 2 with per-basis apodization  (soft iodine, sharp water)
   → Kalender-1988 true ACNR on the basis maps
   → Monoenergetic VMI synthesis  (50 / 70 / 100 / 140 keV)
   → Measured-vs-theoretical per-rod verification
```

!!! info "Why Projection Domain?"
    Two structural differences vs an image-domain DECT pipeline:

    1. **Material decomposition before reconstruction.** The per-ray
       Cong univariate solver consumes log-line-integrals directly, so
       the basis fit sees the actual polychromatic transmission physics
       (and the bowtie's per-ray spectral hardening).  No pre-FBP
       linearization, no HU-to-fraction inverse polynomial.
    2. **Image-domain anti-correlated noise reduction (ACNR).** Material
       decomposition stamps anti-correlated noise on the basis maps (the
       VMI-noise "U"); `BS.apply_acnr_kalender!` removes it after FBP with a
       data-adaptive covariance eigen-rotation + edge-aware joint bilateral,
       keeping the structure axis pixel-perfect (no resolution loss).

!!! success "References"
    - Cong, De Man, Wang (2022), *J X-Ray Sci Technol* — projection-
      domain per-ray univariate solver (dual-kVp DECT); material-basis
      variant (iodine + water).
    - Clark, Badea (2023), *Med Phys* — image-domain RSKR (rank-sparse
      bandwidth); the Kalender true ACNR in `BS.apply_acnr_kalender!`
      adapts these moves to the water/iodine basis-map pair.
"""

# ╔═╡ 05000001-0000-4000-8000-000000000020
md"""
## Notebook Setup
"""

# ╔═╡ 05000001-0000-4000-8000-000000000004
import PlutoUI

# ╔═╡ 05000001-0000-4000-8000-000000000030
# ╠═╡ show_logs = false
import BasisSimulator as BS

# ╔═╡ 05000001-0000-4000-8000-000000000031
import CairoMakie as CM

# ╔═╡ 05000001-0000-4000-8000-000000000005
PlutoUI.TableOfContents()

# ╔═╡ 05000001-0000-4000-8000-000000000040
begin
    import GPUSelect
    AT = GPUSelect.Storage() # the backend array type, directly: MtlArray / CuArray / oneArray / ROCArray / Array
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end

# ╔═╡ 05000001-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 05000002-0000-4000-8000-000000000000
md"""
## Scan Setup and Simulation
"""

# ╔═╡ 05000002-0000-4000-8000-000000000001
md"""
### 01. `Phantom()` Struct
**Gammex 472**
"""

# ╔═╡ 05000002-0000-4000-8000-000000000010
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,
    n_slices = 16,
    fov_cm = 35.0,
    z_cm = 1.0,
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
### 02. `Scanner()` Struct
**GE Revolution Apex Elite**
"""

# ╔═╡ 05000003-0000-4000-8000-000000000010
scanner = BS.Scanner(
    source_to_isocenter = 625.6,
    source_to_detector = 1100.0,

    detector_rows = 256,
    detector_cols = 834,
    detector_row_size = 0.625,
    detector_col_size = 0.6,

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

# ╔═╡ 05000004-0000-4000-8000-000000000001
md"""
### 03. `CTProtocol()` Struct
**Dual-kVp Switching**

| kVp | Instantaneous mA | Duty cycle | Effective mA |
|-----|------------------|------------|--------------|
| 80  | 407              | 0.65       | 264.55       |
| 140 | 405              | 0.35       | 141.75       |
"""

# ╔═╡ 05000004-0000-4000-8000-000000000010
protocol_low = BS.CTProtocol(
    # DICOM-faithful GSI per-view exposure (Revolution Apex scan 2026-02-23,
    # DE_80_140kVp_10.07mGyCTDI, tags 0053,1085/1086): the tube runs 407 mA
    # during each 0.508 ms 80 kVp view → 0.207 mAs/view.  The real scanner
    # acquires 492 views/channel interleaved and its GSI chain
    # view-interpolates back to the full 984-view grid before recon; here the
    # 984-view simulation stands in for that interpolated grid, with the
    # per-view photon statistics (what controls starvation) matching the
    # clinical scan exactly.  (The earlier 265 mA-eff had the right dose but
    # 1.54× fewer photons per view → starvation streaks the real GSI does
    # not show.)
    kVp = 80,
    mA = 407.0,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 05000004-0000-4000-8000-000000000020
protocol_high = BS.CTProtocol(
    # 140 kVp channel: 405 mA × 0.508 ms dwell (DICOM tags 0053,1083/1084),
    # same 984-view interpolated grid.
    kVp = 140,
    mA = 405.0,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 05000005-0000-4000-8000-000000000001
md"""
### 04. `SimOptions()` & `ReconOptions()`
"""

# ╔═╡ 05000005-0000-4000-8000-000000000010
sim_opts = BS.SimOptions(
    fidelity = :eict,
    seed = 1234,
    projector = :dd_fast,  # same anti-aliased DD physics, single-pass fused kernels (~47× faster poly)
);

# ╔═╡ 05000005-0000-4000-8000-000000000020
recon_opts = let
    slice_thickness_mm = 0.625
    n_recon_slices = round(Int, 5.0 / slice_thickness_mm)
    BS.ReconOptions(
        matrix_size = (512, 512, n_recon_slices),
        fov_cm = 35.0,
        z_cm = 0.5,
    )
end;

# ╔═╡ 05000006-0000-4000-8000-000000000001
md"""
### 05. Forward Project: `simulate!`

Run `BS.simulate!` on each kVp protocol.  The EICT path bakes in
per-ray spatial scatter + Compton + Rayleigh, and we keep the
simulator's noisy line-integral sinogram (`ws.sinogram`).
"""

# ╔═╡ 05000006-0000-4000-8000-000000000010
sim_low = let
    @info "Simulating: 80 kVp / $(round(protocol_low.mA, digits = 1)) mA-eff (DE low)…"
    ws = BS.create_eict_workspace(scanner, protocol_low, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, protocol_low, sim_opts)
    # Per-ray air-scan counts I0(col,row) — needed by the Jensen debias
    # (bowtie + heel make the EICT flux ray-dependent).
    I0_scalar = BS.compute_detector_I0(ws.geom, protocol_low, sum(ws.weights)) * Float64(ws.η_eff)
    air_ref = ws.bowtie_air_reference === nothing ? ones(Float32, ws.geom.n_cols, ws.geom.n_rows) :
        Array(ws.bowtie_air_reference)
    result = (sino = Array(ws.sinogram), geom = ws.geom,
        I0_ray = Float32.(I0_scalar .* Float64.(air_ref)))
    ws = nothing; GC.gc(true)
    result
end;

# ╔═╡ 05000006-0000-4000-8000-000000000020
sim_high = let
    @info "Simulating: 140 kVp / $(round(protocol_high.mA, digits = 1)) mA-eff (DE high)…"
    ws = BS.create_eict_workspace(scanner, protocol_high, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, protocol_high, sim_opts)
    # Per-ray air-scan counts I0(col,row) — needed by the Jensen debias
    # (bowtie + heel make the EICT flux ray-dependent).
    I0_scalar = BS.compute_detector_I0(ws.geom, protocol_high, sum(ws.weights)) * Float64(ws.η_eff)
    air_ref = ws.bowtie_air_reference === nothing ? ones(Float32, ws.geom.n_cols, ws.geom.n_rows) :
        Array(ws.bowtie_air_reference)
    result = (sino = Array(ws.sinogram), geom = ws.geom,
        I0_ray = Float32.(I0_scalar .* Float64.(air_ref)))
    ws = nothing; GC.gc(true)
    result
end;

# ╔═╡ 05000006-0000-4000-8000-000000000040
let
    n_row = size(sim_low.sino, 2)
    mid_r = n_row ÷ 2 + 1

    # sino layout is (n_col, n_row, n_view); transpose so heatmap x = view, y = col
    slice_lo = permutedims(sim_low.sino[:, mid_r, :], (2, 1))
    slice_hi = permutedims(sim_high.sino[:, mid_r, :], (2, 1))

    # Dynamic shared range across both kVps — q1/q99 percentile clipping
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

    hms = nothing
    for (r, c, ttl, slice) in panels
        ax = CM.Axis(fig[r, c]; title = ttl, axis_kwargs...)
        hms = CM.heatmap!(ax, slice; colormap = :viridis, colorrange = sino_window)
    end
    CM.Colorbar(
        fig[1, 3]; colormap = :viridis, colorrange = sino_window,
        label = "Log Line Integral", width = 16, labelsize = 22, ticklabelsize = 18
    )
    fig
end

# ╔═╡ 05000008-0000-4000-8000-000000000000
md"""
## VMI Pipeline
"""

# ╔═╡ 05000008-0000-4000-8000-000000000001
md"""
### 01. Jensen Debias
**Log-Poisson Bias Correction**

Per ray: `p ← p − 1/(2N)`, `N = I₀(col,row)·e^{−p}` — the first-order bias of
`−log` of Poisson counts.  Deterministic (σ untouched); on photon-starved rays
behind the dense Ca rods (N ≈ 2–50 at 80 kVp) it removes the 1–2.5 % `p`
inflation Cong would amplify into low-keV HU bias.  Runs on the **raw**
sinograms — before any denoising — so `N` is the true detected count.
"""

# ╔═╡ 05000008-0000-4000-8000-000000000005
sino_debiased = let
    # First-order log-Poisson debias: E[−log(N/I0)] = p_true + 1/(2N) for
    # Poisson counts.  Deterministic bias correction (σ untouched); MUST run
    # on the RAW sinograms so N is the true detected count.
    debias(p, I0_ray) = begin
        out = Float32.(p)
        nc, nr, nv = size(out)
        for v in 1:nv, r in 1:nr, c in 1:nc
            N = max(I0_ray[c, r] * exp(-out[c, r, v]), 1.0f0)
            out[c, r, v] -= 1.0f0 / (2.0f0 * N)
        end
        out
    end
    (low  = debias(sim_low.sino,  sim_low.I0_ray),
     high = debias(sim_high.sino, sim_high.I0_ray))
end;

# ╔═╡ 05000008-0000-4000-8000-000000000006
md"""
### 02. SVD Denoise
**Edge-Aware Bilateral (Projection Domain)**

Per-row cross-channel SVD on the debiased `(low, high)` pair: the common
anatomy `U[:,1]` is kept **pixel-perfect** and the residual `U[:,2]` (streaks
+ decorrelated quantum noise) is cleaned with an edge-aware joint bilateral —
zero resolution cost (unlike `apply_sino_svd_denoise`'s Gaussian).  Runs
**after** the debias, so it also removes the noise the debias amplifies in
photon-starved rays.  `APPLY_SVD = false` ⇒ debias-only passthrough.
"""

# ╔═╡ 05000008-0000-4000-8000-0000000000a0
sino_denoised = let
    APPLY_SVD = true
    if APPLY_SVD
        out = BS.apply_sino_svd_denoise_bilateral(
            [sino_debiased.low, sino_debiased.high];
            # ── how FAR it can reach to average. Bigger = smoother but slower;
            #    never averages across an edge. Secondary knob; keep ≈ σ_s.
            bilat_radius  = 3,    # search-window radius (px)
            # ── spatial falloff inside that window. ↑ = stronger flat-region
            #    (streak/noise) smoothing; ↓ = gentler. Pair with radius.
            bilat_sigma_s = 2.0,  # spatial Gaussian σ (px)
            # ── THE important knob (edge sensitivity): how big an intensity
            #    jump counts as a real edge to PROTECT vs. noise to smooth.
            #    ↓ (≈1.0) = guard edges harder, less smoothing, safest for
            #    resolution. ↑ (3–4) = smooth more, risks eating faint detail.
            bilat_range_k = 2.0,
        )
        (low = out[1], high = out[2])
    else
        (low = sino_debiased.low, high = sino_debiased.high)
    end
end;

# ╔═╡ 05000008-0000-4000-8000-000000000007
md"""
### 03. Material Decomposition
**Cong (Projection Domain)**

Per-ray Newton solver on the polychromatic transmission integral (iodine +
water basis, seeded `water_basis = (a = 0, c = 1)`), yielding basis line
integrals `sino_iodine = ∫c_iodine dr` and `sino_water = ∫c_water dr`.

!!! info "Full-Spectrum Awareness"
    The basis builder uses `BS.resolve_source_spectrum_full`, which
    returns a per-ray 3D spectral weight `ŵ[col, row, E]` containing
    EVERY spectrum-shaping factor the simulation applied — bowtie,
    anode heel, and detector efficiency η(E).  The decomposition kernel
    detects the 3D shape automatically and uses the right column–row
    spectrum for each detector position, so the inversion's forward
    model matches the data exactly.
"""

# ╔═╡ 05000008-0000-4000-8000-000000000010
material_basis = let
    # FULL detected spectrum — tube × filters × bowtie × heel × η(E), each
    # gated on sim_opts.use_*.  The decomposition's forward model must match
    # the forward model simulate! actually applied (same doctrine as the
    # knobless water BHC); the old bowtie-only resolver omitted η + heel and
    # left a keV-dependent bias on every ray.
    e_L, ŵ_L = BS.resolve_source_spectrum_full(
        sim_opts, protocol_low; scanner = scanner, geom = sim_low.geom,
    )
    e_H, ŵ_H = BS.resolve_source_spectrum_full(
        sim_opts, protocol_high; scanner = scanner, geom = sim_high.geom,
    )

    ŵ_L_f32 = Float32.(ŵ_L ./ sum(ŵ_L; dims = ndims(ŵ_L)))
    ŵ_H_f32 = Float32.(ŵ_H ./ sum(ŵ_H; dims = ndims(ŵ_H)))

    iodine_mat = BS.XA.Elements.Iodine     # pure-element iodine basis
    water_mat = BS.XA.Materials.water     # H₂O compound water basis

    p_L = Float32[Float32(BS.compute_mass_μ_at_energy(iodine_mat, Float64(E))) for E in e_L]
    q_L = Float32[Float32(BS.compute_mass_μ_at_energy(water_mat, Float64(E))) for E in e_L]
    p_H = Float32[Float32(BS.compute_mass_μ_at_energy(iodine_mat, Float64(E))) for E in e_H]
    q_H = Float32[Float32(BS.compute_mass_μ_at_energy(water_mat, Float64(E))) for E in e_H]

    (
        ŵ_L = ŵ_L_f32, p_L = p_L, q_L = q_L,
        ŵ_H = ŵ_H_f32, p_H = p_H, q_H = q_H,
    )
end;

# ╔═╡ 05000008-0000-4000-8000-000000000020
sino_basis = let
    # Debias + SVD-bilateral denoise already applied upstream (in the
    # `sino_denoised` cell, correct order: debias RAW → denoise).  Just move
    # the finished line-integral sinograms to the GPU for Cong.
    sino_low_gpu = to_gpu(sino_denoised.low)
    sino_high_gpu = to_gpu(sino_denoised.high)

    sino_y = similar(sino_low_gpu)   # iodine basis line integrals
    sino_c = similar(sino_low_gpu)   # water  basis line integrals
    fill!(sino_y, 0.0f0); fill!(sino_c, 0.0f0)

    cong_ws = BS.create_cong_workspace(sino_low_gpu, material_basis)
    BS.apply_cong!(
        cong_ws, sino_y, sino_c, sino_low_gpu, sino_high_gpu;
        water_basis = (a = 0.0f0, c = 1.0f0),
    )

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

# ╔═╡ 05000008-0000-4000-8000-000000000040
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

    # (panel_col, cbar_col) — colorbar always immediately right of its panel.
    panels = (
        (
            1, 1, 2, "Iodine Basis Sinogram", "g/cm²",
            slice_iod, _qrange(slice_iod),
        ),
        (
            1, 3, 4, "Water Basis Sinogram", "g/cm²",
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

# ╔═╡ 05000009-0000-4000-8000-000000000001
md"""
### 04. FBP Basis Maps
**Per-Basis Apodization**

Two FDK passes with `BS.SoftFilter()` — one per basis sinogram.  The
iodine + water reconstructions land in basis-density units (g/cm³)
directly; no post-decomposition step needed.
"""

# ╔═╡ 05000009-0000-4000-8000-000000000010
basis_volumes = let
    matrix_size = recon_opts.matrix_size
    geom = sino_basis.geom

    # PER-BASIS apodization (NOT per-keV).  The FBP filter is applied ONCE to
    # each basis sinogram; every VMI is then synthesized as VMI_E = W + α(E)·I.
    # The photon-starvation STREAKS live in the IODINE channel, so a SOFT
    # iodine apodization crushes the 50 keV streaks (α(50)=54) while barely
    # touching 140 keV (α=5.4, water-dominated).  The WATER channel keeps its
    # sharp filter, so anatomy resolution is preserved at EVERY energy.  The
    # 50-vs-140 selectivity is emergent from α(E), not a per-energy filter.
    #
    # ── TUNE the iodine softness here: lower the mid/high window values ⇒
    #    softer ⇒ less 50-keV streak noise (but softer iodine detail).
    iodine_filter = BS.CustomFilter(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        (1.0, 0.40, 0.12, 0.03, 0.001),
    )
    # Water filter: halfway between SoftFilter and StandardFilter (per-point
    # average) — a touch sharper anatomy than Soft, a touch less high-keV
    # noise than Standard.  Sets the σ_W floor / anatomy resolution.
    water_filter = BS.CustomFilter(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        (1.0, 0.8744, 0.6003, 0.3031, 0.0266),
    )

    function _fbp(sino_cpu, filt)
        sino_gpu = to_gpu(Float32.(sino_cpu))
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, geom, matrix_size; filter = filt,
        )
        recon = Array(BS.reconstruct!(ws, sino_gpu, geom))
        ws = nothing; sino_gpu = nothing
        GC.gc(true)
        return Float32.(recon)
    end

    (
        vol_iodine_raw = _fbp(sino_basis.sino_iodine, iodine_filter),
        vol_water_raw = _fbp(sino_basis.sino_water, water_filter),
        geom = geom,
    )
end;

# ╔═╡ 05000009-0000-4000-8000-000000000030
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

    # (row, panel_col, cbar_col, axis title, colorbar label, slice, range)
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

# ╔═╡ 05000009-0000-4000-8000-000000000040
md"""
### 05. ACNR
**Anti-Correlated Noise Reduction**

Material decomposition stamps anti-correlated noise on the basis maps
(`ρ_basis < 0`) — that anti-correlation *is* the VMI-noise U.  `BS.apply_acnr_kalender!`
(data-adaptive cov-ACNR, `denoising/acnr.jl`) removes it: a closed-form 2×2
covariance eigen-rotation keeps the structure axis **e1** pixel-perfect and
joint-bilateral-denoises only the anti-correlated noise axis **e2** (edge-aware,
so real water/iodine edges survive).  Runs on the FBP basis maps, just
before VMI synthesis.
"""

# ╔═╡ 05000009-0000-4000-8000-000000000050
# Kalender-1988 true ACNR on the FBP basis maps via `BS.apply_acnr_kalender!`.
basis_acnr = let
    APPLY_ACNR = true          # Kalender-1988 true ACNR; false ⇒ passthrough

    W = copy(basis_volumes.vol_water_raw)
    I = copy(basis_volumes.vol_iodine_raw)

    if APPLY_ACNR
        # TRUE ACNR (Kalender 1988): per-pixel local regression between the
        # two maps' high-frequency channels — anti-correlated (noise) content
        # is subtracted exactly, positively-correlated (structure) pixels are
        # clamped to zero correction and stay BIT-untouched.  No smoothing
        # operator touches signal: resolution preservation by construction.
        info = BS.apply_acnr_kalender!(W, I)
        @info "[ACNR · Kalender-1988 true ACNR] ρ_hp(W,I)=$(round(info.ρ_hp, digits = 3)) · σ_hp(W)=$(round(info.σ_hW, sigdigits = 3)) σ_hp(I)=$(round(info.σ_hI, sigdigits = 3)) · anti-correlated HF removed pixelwise, structure clamp-protected"
    else
        @info "[ACNR] OFF (passthrough)"
    end

    (vol_iodine_raw = I, vol_water_raw = W, geom = basis_volumes.geom)
end;

# ╔═╡ 0500000b-0000-4000-8000-000000000001
md"""
### 06. VMI Synthesis

`BS.synth_vmi_2basis(c_water, c_iodine; energy_keV)` evaluates the
textbook 2-basis linear mix (McCollough 2015) at the target keV:

```
μ(E)  = c_water(r) · (μ/ρ)_water(E) + c_iodine(r) · (μ/ρ)_iodine(E)
HU(E) = 1000 · (μ(E) − (μ/ρ)_water(E)) / (μ/ρ)_water(E)
```

The denominator is the **mono-energetic** linear attenuation of pure
water at the target VMI energy from NIST tables.  VMI grid:
50, 70, 100, 140 keV.

!!! note "Solid-Water Diagnostic"
    The `solid_water_basis` cell below measures `⟨c_water⟩` and
    `⟨c_iodine⟩` over a deeply-eroded solid-water ROI.  Its
    synth-evaluated μ_water at each VMI energy is logged next to the
    textbook mono divisor as a Δ% drift.  This is **diagnostic only** —
    after the Cong decomposition + image-domain ACNR, the residual
    bias is small enough that the textbook analytical divisor recovers
    correct HUs directly without needing an empirical anchor.
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
    sw_bool = BS.erode_mask_2d(sw_bool_raw; erode_px = ERODE_PX)

    n_raw = count(sw_bool_raw); n_eroded = count(sw_bool)
    n_eroded == 0 && error(
        "solid_water_basis: deep erosion (σ = $(ERODE_PX) px) wiped out the SW " *
            "ROI (raw count = $(n_raw)).  Reduce erode_px or check phantom mask."
    )
    @info "solid_water_basis: SW mid-slice voxel count $(n_raw) → $(n_eroded) " *
        "after $(ERODE_PX)-px erosion"

    sw_idx = findall(sw_bool)
    n_z = size(basis_acnr.vol_water_raw, 3)
    function _mean(vol)
        s = 0.0; n = 0
        for z in 1:n_z, ci in sw_idx
            s += vol[ci, z]; n += 1
        end
        return s / n
    end

    c_w = Float64(_mean(basis_acnr.vol_water_raw))
    c_i = Float64(_mean(basis_acnr.vol_iodine_raw))
    @info "solid_water_basis: ⟨c_water⟩_SW = $(round(c_w, digits = 4)) g/cm³, " *
        "⟨c_iodine⟩_SW = $(round(c_i, digits = 6)) g/cm³"

    (
        c_water = c_w, c_iodine = c_i, n_voxels = length(sw_idx) * n_z,
        mask_2d = collect(sw_bool),
    )   # for downstream viz
end;

# ╔═╡ 0500000b-0000-4000-8000-000000000015
de_vmi_energies = [50.0, 70.0, 100.0, 140.0];

# ╔═╡ 0500000b-0000-4000-8000-000000000020
vmi_HU_by_keV = let
    # `BS.synth_vmi_2basis` expects c_iodine in mg/mL; our basis maps
    # are in g/cm³ (= g/mL).  Multiply by 1000 to convert.
    c_iodine_mg_per_mL = basis_acnr.vol_iodine_raw .* 1000.0f0

    out = Dict{Float64, Array{Float32, 3}}()
    for E in de_vmi_energies
        # Diagnostic-only — the SW-ROI synth-evaluated μ_water vs the
        # textbook mono divisor.  Drift = residual basis-decomp bias.
        μρ_w = BS.compute_mass_μ_at_energy(BS.XA.Materials.water, E)
        μρ_I = BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, E)
        μ_water_anchor = solid_water_basis.c_water * μρ_w +
            solid_water_basis.c_iodine * μρ_I
        Δ_pct = 100.0 * (μ_water_anchor - μρ_w) / μρ_w
        @info "VMI synth @ $(Int(E)) keV: divisor = $(round(μρ_w, digits = 5)) cm⁻¹ " *
            "(mono μρ_water);  SW-ROI anchor = " *
            "$(round(μ_water_anchor, digits = 5)) → Δ = $(round(Δ_pct, digits = 2))%"

        out[E] = BS.synth_vmi_2basis(
            basis_acnr.vol_water_raw, c_iodine_mg_per_mL;
            energy_keV = E,
        )
    end
    out
end;

# ╔═╡ 0500000b-0000-4000-8000-000000000040
let
    HU_window = (-200, 500)

    fig = CM.Figure(size = (1180, 1180))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    sample = vmi_HU_by_keV[50.0]
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
    # Explicit colorbar binding — same colormap + colorrange as every panel,
    # no auto-extraction from a plot object.
    CM.Colorbar(
        fig[1:2, 3];
        colormap = :grays, colorrange = HU_window,
        label = "HU", width = 16, labelsize = 22, ticklabelsize = 18,
    )

    CM.save(
        joinpath(@__DIR__, "..", "assets", "dual_kvp_vmi_projection_grid.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0500000d-0000-4000-8000-000000000001
md"""
## Results

Per-rod measured vs theoretical HU at the canonical four VMI energies
(50 / 70 / 100 / 140 keV).

!!! info "Methodology"
    - **Measured HU** = mean over an 8-px-radius circular ROI at the
      rod centroid, broadcast across all z slices.  The small core ROI
      avoids partial-volume bleed at the rod edge.
    - **Theoretical HU** = `1000 · (μ_r(E) − μ_water(E)) / μ_water(E)`
      from `BS.compute_μ_at_energy(material_r, E)` — pure physics,
      no fitting, no calibration assumption.

!!! note "What the Plots Show"
    Two panels: **Calcium rods** (50 – 600 mg/mL, Compton-dominated
    smooth roll-off as E increases) and **Iodine rods** (2 – 20 mg/mL,
    K-edge at 33.2 keV so 50 keV still amplifies iodine HU strongly vs
    the 70+ keV plateau).

    Solid line = measured.  Dashed line = theoretical.  Tight overlay
    means the projection-domain pipeline is recovering the underlying
    physics correctly.
"""

# ╔═╡ 0500000d-0000-4000-8000-000000000010
ROD_LABELS = (
    Ca = (UInt8(10), UInt8(11), UInt8(12), UInt8(13), UInt8(14), UInt8(15), UInt8(16)),
    I = (UInt8(20), UInt8(21), UInt8(22), UInt8(23), UInt8(24), UInt8(25), UInt8(26)),
);

# ╔═╡ 0500000d-0000-4000-8000-000000000020
ROD_NAMES = (
    Ca = ("50 mg/mL", "100 mg/mL", "200 mg/mL", "300 mg/mL", "400 mg/mL", "500 mg/mL", "600 mg/mL"),
    I = ("2.0 mg/mL", "2.5 mg/mL", "5.0 mg/mL", "7.5 mg/mL", "10.0 mg/mL", "15.0 mg/mL", "20.0 mg/mL"),
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
        n_E = length(de_vmi_energies)
        meas = zeros(Float64, n_rods, n_E)
        theo = zeros(Float64, n_rods, n_E)
        for (i, lab) in pairs(labels)
            mat = materials[Int(lab) + 1]   # mask_value + 1
            for (j, E) in pairs(de_vmi_energies)
                meas[i, j] = measured_hu(vmi_HU_by_keV[E], lab)
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

# ╔═╡ 0500000d-0000-4000-8000-000000000002
md"""
### Water ROI

Left panel: the deeply-eroded solid-water ROI (12-px erosion ≈ 8 mm)
overlaid in red on the 70 keV VMI slice — exactly the voxels feeding
the `solid_water_basis` diagnostic.  Right panel: mean HU over that
ROI vs VMI energy.

!!! note "How to Read the Bars"
    With the textbook mono divisor `μρ_water(E)`, solid water reads at
    `1000·(⟨c_water⟩_SW − 1)` — a roughly constant offset that
    quantifies the residual basis-decomp bias.

    - All bars cluster near **0 HU** with a small (~few HU) consistent
      offset → pipeline is recovering physics correctly.
    - **Energy-dependent drift** (offset varies systematically across
      keVs) → spectral-shape problem upstream worth investigating.
"""

# ╔═╡ 0500000d-0000-4000-8000-000000000005
let
    fig = CM.Figure(size = (1180, 580))

    # ─── Left panel — 70 keV VMI slice + eroded SW ROI overlay ───────
    HU_window = (-200, 500)
    mid = size(vmi_HU_by_keV[70.0], 3) ÷ 2
    bg = vmi_HU_by_keV[70.0][:, :, mid]

    overlay = Float32[b ? 1.0f0 : NaN32 for b in solid_water_basis.mask_2d]

    ax1 = CM.Axis(
        fig[1, 1];
        title = "Eroded Water Region",
        subtitle = "Overlaid on 70 keV VMI",
        aspect = CM.DataAspect(),
        titlesize = 32, subtitlesize = 24,
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
    n_z = size(vmi_HU_by_keV[70.0], 3)
    function _mean_hu(vol)
        s = 0.0; n = 0
        for z in 1:n_z, ci in sw_idx
            s += vol[ci, z]; n += 1
        end
        return s / n
    end
    sw_hu_per_keV = [_mean_hu(vmi_HU_by_keV[E]) for E in de_vmi_energies]

    # cgrad → Vector{RGBAf} for barplot's color kwarg
    n_E = length(de_vmi_energies)
    bar_colors = [CM.cgrad(:plasma, n_E; categorical = true)[i] for i in 1:n_E]

    ax2 = CM.Axis(
        fig[1, 2];
        title = "Water Region Mean HU",
        subtitle = "Per VMI Energy",
        xlabel = "VMI Energy (keV)", ylabel = "HU",
        xticks = (collect(1:n_E), ["$(Int(E))" for E in de_vmi_energies]),
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
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
            fontsize = 16, offset = (0, h ≥ 0 ? 4 : -4),
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

# ╔═╡ 0500000d-0000-4000-8000-000000000040
md"""
### Water-Region Noise

HU noise (σ) over the **large eroded solid-water region**
(`solid_water_basis.mask_2d`) — the whole solid-water background between the
rods, deeply eroded (σ ≈ 8 mm) to stay clear of every rod edge and
partial-volume halo.  This is the canonical water ROI for **all** noise
measurements in this notebook (not a tiny central circle), so the statistics
sample the full background the way a clinical noise readout would.

Right panel = σ vs VMI energy.  Diagnoses how the textbook
`(c_water, c_iodine) → HU(E)` synth propagates noise through the dual-kVp
pipeline (Cong + image-domain cov-ACNR).  Expectation: σ(50) ≫ σ(70) ≳
σ(140) — monotonic-decreasing, with the natural noise-optimal energy near
70 keV.
"""

# ╔═╡ 0500000d-0000-4000-8000-000000000050
const WATER_NOISE_ROI_RADIUS_PX = 12;   # ≈8.2 mm at 0.683 mm/px (FOV 35 cm / 512)

# ╔═╡ 0500000d-0000-4000-8000-000000000055
# Central circular noise ROI in the solid-water background (image center =
# isocenter = phantom center for the centered Gammex 472).
water_noise_roi = let
    nx_r, ny_r, nz_r = size(basis_acnr.vol_water_raw)
    cx = nx_r ÷ 2 + 1
    cy = ny_r ÷ 2 + 1

    roi_bool = falses(nx_r, ny_r)
    r² = Float64(WATER_NOISE_ROI_RADIUS_PX)^2
    @inbounds for j in 1:ny_r, i in 1:nx_r
        ((i - cx)^2 + (j - cy)^2) ≤ r² && (roi_bool[i, j] = true)
    end

    n_vox = count(roi_bool)
    @info "water_noise_roi: center = ($(cx), $(cy)), radius = $(WATER_NOISE_ROI_RADIUS_PX) px, " *
        "$(n_vox) vx × $(nz_r) z = $(n_vox * nz_r) total"

    (
        center_xy = (Float64(cx), Float64(cy)), mask_2d = roi_bool,
        n_voxels = n_vox, n_total = n_vox * nz_r,
    )
end;

# ╔═╡ 0500000d-0000-4000-8000-000000000060
# Per-keV HU noise (σ) + mean over the LARGE ERODED solid-water region
# (`solid_water_basis.mask_2d`) on the final VMIs — the canonical water
# ROI for ALL noise measurements (not the tiny central circle).
vmi_noise_by_keV = let
    roi_idx = findall(solid_water_basis.mask_2d)
    nz_r = size(vmi_HU_by_keV[70.0], 3)

    out = Dict{Float64, NamedTuple}()
    for E in de_vmi_energies
        vol = vmi_HU_by_keV[E]
        vals = Float64[Float64(vol[ci, z]) for z in 1:nz_r, ci in roi_idx]
        μ = mean(vals); σ = std(vals)
        out[E] = (mean = μ, std = σ, n = length(vals))
        @info "water-region noise @ $(Int(E)) keV: ⟨HU⟩ = $(round(μ, digits = 2)),  σ = $(round(σ, digits = 2)) HU  (n = $(length(vals)))"
    end
    out
end;

# ╔═╡ 0500000d-0000-4000-8000-000000000070
let
    HU_window = (-200, 500)
    mid = size(vmi_HU_by_keV[70.0], 3) ÷ 2
    bg = vmi_HU_by_keV[70.0][:, :, mid]

    overlay = Float32[b ? 1.0f0 : NaN32 for b in solid_water_basis.mask_2d]

    fig = CM.Figure(size = (1180, 580))

    ax1 = CM.Axis(
        fig[1, 1];
        title = "Eroded Water Region",
        subtitle = "Overlaid on 70 keV VMI",
        aspect = CM.DataAspect(),
        titlesize = 32, subtitlesize = 24,
    )
    CM.heatmap!(ax1, bg; colormap = :grays, colorrange = HU_window)
    CM.heatmap!(
        ax1, overlay; colormap = :reds, alpha = 0.5,
        nan_color = (:white, 0.0),
    )
    CM.hidedecorations!(ax1)

    Es = sort(collect(keys(vmi_noise_by_keV)))
    σs = [vmi_noise_by_keV[E].std  for E in Es]
    μs = [vmi_noise_by_keV[E].mean for E in Es]
    n = length(Es)

    ax2 = CM.Axis(
        fig[1, 2];
        title = "Water-Region Noise vs Energy",
        xlabel = "VMI Energy (keV)",
        ylabel = "Noise σ (HU)",
        xticks = (collect(1:n), ["$(Int(E))" for E in Es]),
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    CM.barplot!(
        ax2, 1:n, σs;
        color = :tomato, strokecolor = :black, strokewidth = 1,
    )
    for (i, (σ, μ)) in enumerate(zip(σs, μs))
        CM.text!(
            ax2, i, σ;
            text = "σ=$(round(σ; digits = 1))\n⟨HU⟩=$(round(μ; digits = 1))",
            align = (:center, :bottom),
            fontsize = 16, offset = (0, 8),
        )
    end
    CM.ylims!(ax2, 0, 1.20 * maximum(σs))

    CM.save(
        joinpath(@__DIR__, "..", "assets", "dual_kvp_vmi_water_noise_vs_energy.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0500000e-0000-4000-8000-000000000001
md"""
### Per-Rod Regression
"""

# ╔═╡ 0500000e-0000-4000-8000-000000000010
let
    fig = CM.Figure(size = (1180, 580))

    # Two panels share the same color scheme: warm for Ca, cool for I, with
    # rod-concentration index → color shade
    cmap_ca = CM.cgrad(:Oranges, 7; categorical = true)
    cmap_i = CM.cgrad(:GnBu, 7; categorical = true)

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
### Linear Regression

Same data, scattered as `(theoretical, measured)` per rod-energy pair
with a per-energy least-squares line and the `y = x` identity.  Ca and
I are split into separate panels because Ca lives at much higher HU
and would otherwise dominate a shared-axis fit.

!!! info "How to Read the Fit"
    | Observation                  | Means                                          |
    |------------------------------|------------------------------------------------|
    | Slope ≈ 1, b ≈ 0, R² ≈ 1     | Pipeline recovers physics                      |
    | Slope ≠ 1                    | Multiplicative cal mismatch (mass-attn, basis) |
    | Intercept ≠ 0                | Additive offset (residual cup, water baseline) |
    | Low R²                       | Non-linear distortion (partial volume, decomp) |
"""

# ╔═╡ 0500000e-0000-4000-8000-000000000030
let
    fig = CM.Figure(size = (1000, 1200))

    # One color per VMI energy (cool→warm sweep)
    energy_colors = Dict(
        50.0 => CM.RGBf(0.85, 0.27, 0.1),
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
        rmse = sqrt(ss_res / length(y))   # HU units (same as y)
        return (slope = β, intercept = α, r² = r², rmse = rmse)
    end

    panels = (
        (:Ca, "Calcium Rods", "50–600 mg/mL"),
        (:I, "Iodine Rods", "2–20 mg/mL"),
    )

    # Vertical layout: panels stacked top → bottom (Ca on top, I on bottom).
    for (row, (group, title, subtitle)) in pairs(panels)
        d = rod_data[group]
        ax = CM.Axis(
            fig[row, 1];
            title = title,
            subtitle = subtitle,
            xlabel = "Theoretical HU",
            ylabel = "Measured HU",
            # aspect = CM.AxisAspect(1),
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

    CM.save(
        joinpath(@__DIR__, "..", "assets", "vmi_regression.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 15000001-0000-4000-8000-000000000001
md"""
### Verification

Quantitative PASS/FAIL against first-principles theory — per rod, per VMI
energy.  Theory is mono HU from the same XrayAttenuation data that drove the
simulation.  This is the notebook's contract: the projection-domain direct
solve should put every rod on its theoretical line at every keV.
"""

# ╔═╡ 15000001-0000-4000-8000-000000000002
verification = let
    checks = NamedTuple[]
    addcheck(name, val, lo, hi) = push!(checks,
        (name = name, value = round(val; digits = 1), lo = lo, hi = hi, pass = lo <= val <= hi))

    # solid water: |HU| small at every energy (basis water map → HU ≈ 0)
    sw = solid_water_basis
    sw_worst = maximum(abs(mean(vmi_HU_by_keV[E][sw.mask_2d, :])) for E in de_vmi_energies)
    addcheck("solid water worst |HU| across keV", sw_worst, 0.0, 10.0)

    # CRITICAL clinical requirement: VMI noise decreases MONOTONICALLY with
    # keV (the U-shape = untreated anti-correlated basis noise; cov-ACNR
    # removes it — see the solved U-shape analysis).
    Es = sort(collect(de_vmi_energies))
    σs = [std(vmi_HU_by_keV[E][sw.mask_2d, :]) for E in Es]
    mono_ok = all(σs[i] > σs[i + 1] for i in 1:(length(σs) - 1))
    push!(checks, (name = "noise monotonic ↓ with keV: σ = $(join(round.(σs; digits = 1), " > "))",
        value = mono_ok ? 1.0 : 0.0, lo = 1.0, hi = 1.0, pass = mono_ok))

    # per-rod, per-energy: |Δ| ≤ max(15 HU, 10 %)
    rod_rows = String[]
    n_pass_rod = 0; n_rod = 0
    for group in (:Ca, :I), (i, name) in pairs(rod_data[group].names)
        cells = String[]
        ok_all = true
        for (j, E) in pairs(de_vmi_energies)
            meas = rod_data[group].measured[i, j]
            theo = rod_data[group].theoretical[i, j]
            ok = abs(meas - theo) <= max(15.0, 0.10 * abs(theo))
            ok_all &= ok
            push!(cells, "$(round(meas; digits = 0)) / $(round(theo; digits = 0)) $(ok ? "✅" : "❌")")
        end
        n_rod += 1
        n_pass_rod += ok_all
        push!(rod_rows, "| $(name) | " * join(cells, " | ") * " |")
    end
    addcheck("rods passing at ALL energies", n_pass_rod, n_rod, n_rod)

    n_pass = count(c -> c.pass, checks)
    verdict = n_pass == length(checks) ? "✅ NB03 VERIFICATION: PASS ($(n_pass)/$(length(checks)))" :
                                          "❌ NB03 VERIFICATION: FAIL ($(n_pass)/$(length(checks)))"
    hdr = join(["$(Int(E)) keV" for E in de_vmi_energies], " | ")
    md_str = """
### $(verdict)

| check | value | expected | pass |
|---|---|---|---|
$(join(["| $(c.name) | $(c.value) | [$(c.lo), $(c.hi)] | $(c.pass ? "✅" : "❌") |" for c in checks], "\n"))

**Per-rod measured / theory (VMI volumes), gate |Δ| ≤ max(15 HU, 10 %):**

| rod | $(hdr) |
|---|$(join(["---" for _ in de_vmi_energies], "|"))|
$(join(rod_rows, "\n"))
"""
    Markdown.parse(md_str)
end

# ╔═╡ 0500000f-0000-4000-8000-000000000001
md"""
### Summary

```
Simulate 80 + 140 kVp  (DICOM-faithful GSI technique, per-view flux from
                         the clinical Revolution Apex scan)
   → Jensen debias + projection-domain SVD-bilateral denoise
   → Projection-domain material decomposition  (Cong univariate, iodine + water)
   → FBP × 2 with per-basis apodization  (soft iodine, sharp water)
   → Kalender-1988 true ACNR  (BS.apply_acnr_kalender! — per-pixel HF
                               regression, structure bit-untouched, zero blur)
   → Monoenergetic VMI synthesis  (textbook 2-basis, mono μρ_water divisor)
   → Automated verification  (water HU · monotonic noise-vs-keV · 14-rod NIST regression)
```

The debias, SVD-bilateral denoise, and Cong decomposition all run
**upstream of FBP**, so quantum noise and beam-hardening residuals can't
propagate into the basis maps.  The per-basis apodization softens only the
iodine channel, so low-keV VMI noise drops preferentially (α(E)² weighting)
while water/anatomy resolution stays sharp at every energy.  Kalender true
ACNR then removes the anti-correlated basis noise (the VMI-U) without
resolution loss — yielding HU-quantitative VMIs with low streak content.
"""

# ╔═╡ Cell order:
# ╟─05000001-0000-4000-8000-000000000010
# ╟─05000001-0000-4000-8000-000000000020
# ╠═05000001-0000-4000-8000-000000000001
# ╠═05000001-0000-4000-8000-000000000002
# ╠═05000001-0000-4000-8000-000000000003
# ╠═05000001-0000-4000-8000-000000000004
# ╠═05000001-0000-4000-8000-000000000030
# ╠═05000001-0000-4000-8000-000000000031
# ╠═05000001-0000-4000-8000-000000000005
# ╠═05000001-0000-4000-8000-000000000040
# ╟─05000001-0000-4000-8000-000000000050
# ╟─05000002-0000-4000-8000-000000000000
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
# ╟─05000006-0000-4000-8000-000000000040
# ╟─05000008-0000-4000-8000-000000000000
# ╟─05000008-0000-4000-8000-000000000001
# ╠═05000008-0000-4000-8000-000000000005
# ╟─05000008-0000-4000-8000-000000000006
# ╠═05000008-0000-4000-8000-0000000000a0
# ╟─05000008-0000-4000-8000-000000000007
# ╠═05000008-0000-4000-8000-000000000010
# ╠═05000008-0000-4000-8000-000000000020
# ╟─05000008-0000-4000-8000-000000000040
# ╟─05000009-0000-4000-8000-000000000001
# ╠═05000009-0000-4000-8000-000000000010
# ╟─05000009-0000-4000-8000-000000000030
# ╟─05000009-0000-4000-8000-000000000040
# ╠═05000009-0000-4000-8000-000000000050
# ╟─0500000b-0000-4000-8000-000000000001
# ╠═0500000b-0000-4000-8000-000000000010
# ╠═0500000b-0000-4000-8000-000000000015
# ╠═0500000b-0000-4000-8000-000000000020
# ╟─0500000b-0000-4000-8000-000000000040
# ╟─0500000d-0000-4000-8000-000000000001
# ╠═0500000d-0000-4000-8000-000000000010
# ╠═0500000d-0000-4000-8000-000000000020
# ╠═0500000d-0000-4000-8000-000000000030
# ╟─0500000d-0000-4000-8000-000000000002
# ╟─0500000d-0000-4000-8000-000000000005
# ╟─0500000d-0000-4000-8000-000000000040
# ╠═0500000d-0000-4000-8000-000000000050
# ╠═0500000d-0000-4000-8000-000000000055
# ╠═0500000d-0000-4000-8000-000000000060
# ╟─0500000d-0000-4000-8000-000000000070
# ╟─0500000e-0000-4000-8000-000000000001
# ╟─0500000e-0000-4000-8000-000000000010
# ╟─0500000e-0000-4000-8000-000000000020
# ╟─0500000e-0000-4000-8000-000000000030
# ╟─15000001-0000-4000-8000-000000000001
# ╟─15000001-0000-4000-8000-000000000002
# ╟─0500000f-0000-4000-8000-000000000001
