### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 06000001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 06000001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 06000001-0000-4000-8000-000000000003
using Statistics: mean, std, quantile

# ╔═╡ 06000001-0000-4000-8000-000000000010
md"""
# 04 · PCCT VMI · Projection-Domain Pipeline

Siemens Naeotom Alpha photon-counting CT simulation (140 kVp / 174 mA,
4-threshold acquisition, Gammex 472 phantom) with a **fully
projection-domain** VMI pipeline.  Every denoising and decomposition
step operates on log-line-integrals before the recon ever runs.

```
Simulate 140 kVp PCCT  (4 bins)
                                     │
                    Per-Bin Scatter Correction
                                     │
                    Joint Sinogram Denoising  (4 channels)
                                     │
                    Bin Combine  (1 + 2 → low,  3 + 4 → high)
                                     │
                    Projection-Domain Material Decomposition
                          → sino_iodine, sino_water
                                     │
                    FBP × 2   (iodine, water basis maps)
                                     │
                    Z-Direction Median Filter × 2
                                     │
                    Monoenergetic VMI Synthesis
                          μ(E) = c_water · (μ/ρ)_water(E)
                               + c_iodine · (μ/ρ)_iodine(E)
                                     │
                    Mono+ Post-Processing  (per-keV σ)
                                     │
                    Measured vs Theoretical Per-Rod Regression
                          at 40 / 70 / 100 / 140 keV
```

!!! info "Why Projection Domain?"
    Two structural differences vs an image-domain PCCT pipeline:

    1. **Material decomposition before reconstruction.** The per-ray
       polynomial inverse consumes log-line-integrals directly, so the
       basis fit sees the actual polychromatic transmission physics.
       No pre-FBP linearization, no HU-to-fraction inverse polynomial.
    2. **Joint denoising before reconstruction.** A per-detector-row
       SVD denoiser exploits the strong cross-bin correlation
       (same view angles, geometry, anatomy) and strips quantum noise
       before it can propagate through decomposition + FBP.  The
       4-bin pass extracts more cross-channel redundancy than a
       post-combine 2-channel pass would (3 noise components stripped
       vs 1) and runs *before* the lossy I0-weighted bin combine.

!!! success "References"
    - Alvarez, Macovski (1976), *Phys Med Biol* — polynomial inverse
      for projection-domain material decomposition.
    - Clark, Badea (2023), *Med Phys* — image-domain SVD signal/noise
      split (the inspiration for the projection-domain joint denoiser).
    - Grant et al. (2014) — Mono+ frequency-split rule.
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
## 2. `Scanner`: Siemens Naeotom Alpha (PCCT, 4-threshold)

CdTe direct-conversion detector with native dexels 0.275 × 0.322 mm at
the detector face (2×2 binned in DAS).  Energy thresholds
`T = [20, 35, 55, 70] keV` define 4 bins:

| Bin | Range (keV) |
|-----|-------------|
| 1   | 20 – 35     |
| 2   | 35 – 55     |
| 3   | 55 – 70     |
| 4   | > 70        |
"""

# ╔═╡ 06000003-0000-4000-8000-000000000010
scanner = let
    native_col_mm = 0.275
    native_row_mm = 0.322
    sid = 610.0
    sdd = 1113.0
    magnification = sdd / sid
    bf = 2

    pixel_col_iso = (native_col_mm * bf) / magnification
    pixel_row_iso = (native_row_mm * bf) / magnification
    n_cols = ceil(Int, 360.0 / pixel_col_iso)

    BS.Scanner(
        source_to_isocenter = sid,
        source_to_detector = sdd,

        detector_rows = 144,
        detector_cols = n_cols,
        detector_row_size = pixel_row_iso,
        detector_col_size = pixel_col_iso,
        detector_shape = BS.CURVED_DETECTOR,
        detector_row_offset = 0.0,
        detector_col_offset = pixel_col_iso / 2,

        focal_spot_width = 0.4,
        focal_spot_length = 0.5,
        target_angle = 7.0,

        gantry_rotation_time = 0.5,
        scan_diameter = 360.0,
        gantry_aperture = 820.0,

        flat_filter_material = :aluminum,
        flat_filter_thickness = 3.0,

        detector_material = :cdte,
        detector_depth = 1.6,
        fill_factor_row = 0.95,
        fill_factor_col = 0.95,
        detection_gain = 1.0,
        electronic_noise = 0.0,

        detector_type = :photon_counting,
        n_energy_bins = 4,
        energy_thresholds = [20.0, 35.0, 55.0, 70.0],
        energy_resolution = 10.0,
        charge_sharing_fwhm = 0.08,
        dead_time_ns = 5.0,
        pixel_mode = :standard,

        native_dexel_col_mm = native_col_mm,
        native_dexel_row_mm = native_row_mm,
        binning_factor = bf,
    )
end;

# ╔═╡ 06000004-0000-4000-8000-000000000001
md"""
## 3. `CTProtocol`: 140 kVp / 174 mA / ~10 mGy CTDIvol

Clinical 140 kVp single-energy acquisition.
`additional_filters = [("Ti", 0.9)]` is the Vectron tube's inherent
0.9 mm titanium window on top of the 3 mm Al flat filter.
"""

# ╔═╡ 06000004-0000-4000-8000-000000000010
protocol = BS.CTProtocol(
    kVp = 140,
    mA = 174.0,
    views = 1200,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Ti", 0.9)],
);

# ╔═╡ 06000005-0000-4000-8000-000000000001
md"""
## 4. `SimOptions` and `ReconOptions`

`fidelity = :pcct` switches the simulator into the photon-counting path
(per-bin sinograms + DRM + Compton scatter modeling).
"""

# ╔═╡ 06000005-0000-4000-8000-000000000010
sim_opts = BS.SimOptions(
    fidelity = :pcct,
    seed = 1234,
    pcct_noise_reduction = 0.3,
);

# ╔═╡ 06000005-0000-4000-8000-000000000020
recon_opts = let
    slice_thickness_mm = 0.4
    n_recon_slices = max(1, round(Int, protocol.collimation_mm / slice_thickness_mm))
    BS.ReconOptions(
        algorithm = :fdk,
        matrix_size = (512, 512, n_recon_slices),
        fov_cm = 35.0,
        z_cm = protocol.collimation_mm / 10.0,
        filter = :standard,
    )
end;

# ╔═╡ 06000006-0000-4000-8000-000000000001
md"""
## 5. Forward Project

Run `BS.simulate!` once on the PCCT protocol.  The simulator returns 4
per-bin sinograms + per-bin I0 + the spatial scatter field.  We apply
a model-based per-bin scatter correction (Ohnesorge re-estimation from
the recombined primary, scaled by spectrum × η × DRM bin weights).
"""

# ╔═╡ 06000006-0000-4000-8000-000000000010
sim_bins = let
    @info "Simulating: $(Int(protocol.kVp)) kVp / $(round(protocol.mA, digits = 1)) mA (PCCT 4-bin)…"
    ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    result = BS.simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)

    geom = ws.geom
    bins = [Array(b) for b in result.pcct_sino.bins]
    I0_bins = copy(result.I0_bins)

    let
        I0_total = Float32(sum(I0_bins))
        eps_f = Float32(1.0e-10)

        combined = zeros(Float32, size(bins[1]))
        for (b, bin_sino) in enumerate(bins)
            I0b = Float32(I0_bins[b])
            @. combined += I0b * exp(-bin_sino)
        end
        @. combined = -log(max(combined, eps_f) / I0_total)

        voxel_size_mm = phantom_cpu.voxel_size .* 10.0
        phantom_diam_cm = BS.estimate_phantom_diameter_cm(phantom_cpu.mask, voxel_size_mm)
        scatter_model = BS.geometry_aware_scatter_model(scanner; phantom_diameter_cm = phantom_diam_cm)
        scatter_field = similar(combined)
        BS.estimate_scatter_field!(scatter_field, combined, scatter_model)

        e_full, w_full = BS.resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner = scanner)
        pcct_det = BS._build_pcct_detector(scanner)
        kVp_val = Float64(maximum(e_full))
        R_mat = BS.compute_mc_drm(pcct_det, kVp_val)
        η_vec = BS.quantum_efficiency_vector(pcct_det.material, pcct_det.thickness_mm, e_full)
        ew = BS.compute_scatter_energy_weights(Float64.(e_full))
        scatter_fracs = BS.compute_scatter_bin_weights(
            Float64.(e_full), Float64.(w_full), ew, Float64.(η_vec), R_mat, kVp_val,
        )

        for (b, bin_sino) in enumerate(bins)
            I0b = Float32(I0_bins[b])
            frac = Float32(scatter_fracs[b])
            for idx in eachindex(bin_sino)
                N_measured = I0b * exp(-bin_sino[idx])
                N_scatter = scatter_field[idx] * I0_total * frac
                N_corrected = N_measured - max(N_scatter, Float32(0))
                bin_sino[idx] = -log(max(N_corrected, eps_f) / I0b)
            end
        end
    end

    ws = nothing; result = nothing
    GC.gc(true)
    (bins = bins, I0_bins = I0_bins, geom = geom)
end;

# ╔═╡ 06000006-0000-4000-8000-000000000040
let
    n_row = size(sim_bins.bins[1], 2)
    mid_r = n_row ÷ 2 + 1

    # sino layout is (n_col, n_row, n_view); transpose so heatmap x = view, y = col
    bin_titles = ("Bin 1", "Bin 2", "Bin 3", "Bin 4")
    bin_subs = ("20 – 35 keV", "35 – 55 keV", "55 – 70 keV", "> 70 keV")
    slices = [permutedims(sim_bins.bins[k][:, mid_r, :], (2, 1)) for k in 1:4]

    # Dynamic shared range across all 4 bins — q1/q99 percentile clipping
    all_v = vcat([vec(s) for s in slices]...)
    sino_window = (
        Float64(quantile(all_v, 0.01)),
        Float64(quantile(all_v, 0.99)),
    )

    fig = CM.Figure(size = (1180, 1180))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "View", ylabel = "Detector Column",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )

    for k in 1:4
        r = ((k - 1) ÷ 2) + 1
        c = ((k - 1) % 2) + 1
        ax = CM.Axis(
            fig[r, c]; title = bin_titles[k], subtitle = bin_subs[k],
            axis_kwargs...,
        )
        CM.heatmap!(ax, slices[k]; colormap = :viridis, colorrange = sino_window)
    end
    CM.Colorbar(
        fig[1:2, 3]; colormap = :viridis, colorrange = sino_window,
        label = "Log Line Integral", width = 16, labelsize = 22, ticklabelsize = 18,
    )
    fig
end

# ╔═╡ 06000007-0000-4000-8000-000000000001
md"""
## 6. Joint Sinogram SVD Denoiser

The 4 PCCT bins share the same anatomy (same view angles, geometry,
Radon transform) but differ in spectral content and quantum noise.
`BS.apply_sino_svd_denoise` is N-channel generic; for PCCT we run it
with all four bins at once.

!!! info "Algorithm"
    Per detector row, stack the channels as columns of a 4-wide
    matrix and SVD-decompose:

    - **`U[:, 1]`** carries the common log-attenuation pattern every
      channel sees (signal-dominated).  Left untouched.
    - **`U[:, 2..N]`** carries the spectral residual + most of the
      decorrelated quantum noise.  Smoothed with a separable 2D
      Gaussian in `(col, view)` space.

    The denoised channels come from reconstituting
    `M = U_d · diag(Σ) · V'` and unpacking the columns.  See
    `src/denoising/sino_svd.jl` for details and the RSKR inspiration
    note.

!!! tip "One Knob"
    `SINO_DENOISE_σ_PX` — Gaussian σ in pixels.  σ ≈ 1–2 px keeps
    rod-scale spectral structure intact while killing pixel-scale
    noise.  σ ≤ 0 short-circuits to a passthrough.
"""

# ╔═╡ 06000007-0000-4000-8000-000000000005
SINO_DENOISE_σ_PX = 2.0;   # only knob — Gaussian σ on the SVD residual components

# ╔═╡ 06000007-0000-4000-8000-000000000010
sino_bins_denoised = BS.apply_sino_svd_denoise(
    [Float32.(b) for b in sim_bins.bins];
    σ_px = SINO_DENOISE_σ_PX,
);

# ╔═╡ 06000007-0000-4000-8000-000000000030
let
    n_row = size(sino_bins_denoised[1], 2)
    mid_r = n_row ÷ 2 + 1

    bin_titles = ("Bin 1", "Bin 2", "Bin 3", "Bin 4")
    slices = [permutedims(sino_bins_denoised[k][:, mid_r, :], (2, 1)) for k in 1:4]

    all_v = vcat([vec(s) for s in slices]...)
    sino_window = (
        Float64(quantile(all_v, 0.01)),
        Float64(quantile(all_v, 0.99)),
    )

    fig = CM.Figure(size = (1180, 1180))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "View", ylabel = "Detector Column",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )

    for k in 1:4
        r = ((k - 1) ÷ 2) + 1
        c = ((k - 1) % 2) + 1
        ax = CM.Axis(
            fig[r, c]; title = bin_titles[k],
            subtitle = "After SVD denoiser",
            axis_kwargs...,
        )
        CM.heatmap!(ax, slices[k]; colormap = :viridis, colorrange = sino_window)
    end
    CM.Colorbar(
        fig[1:2, 3]; colormap = :viridis, colorrange = sino_window,
        label = "Log Line Integral", width = 16, labelsize = 22, ticklabelsize = 18,
    )
    fig
end

# ╔═╡ 06000008-0000-4000-8000-000000000001
md"""
## 7. Bin Combine: Denoised 4 → Low / High Pair

I0-weighted Beer recombination on the **denoised** bins:

```
N_grp = Σ_{b ∈ grp} I0[b] · exp(-p[b])
p_grp = -log(N_grp / Σ_{b ∈ grp} I0[b])
```

* **Low**  = bins **1 + 2** (20 – 55 keV)
* **High** = bins **3 + 4** ( > 55 keV)

Each combined sinogram represents a polychromatic measurement at the
I0-weighted average spectrum of its bin group.
"""

# ╔═╡ 06000008-0000-4000-8000-000000000010
sim_lohi = let
    eps_f = Float32(1.0e-10)

    function _combine(bin_indices)
        I0_sum = Float32(sum(Float64.(sim_bins.I0_bins[bin_indices])))
        N = zeros(Float32, size(sino_bins_denoised[1]))
        for b in bin_indices
            I0b = Float32(sim_bins.I0_bins[b])
            @. N += I0b * exp(-sino_bins_denoised[b])
        end
        return @. -log(max(N, eps_f) / I0_sum)
    end

    sino_low = _combine([1, 2])
    sino_high = _combine([3, 4])
    (sino_low = sino_low, sino_high = sino_high, geom = sim_bins.geom)
end;

# ╔═╡ 06000008-0000-4000-8000-000000000030
let
    n_row = size(sim_lohi.sino_low, 2)
    mid_r = n_row ÷ 2 + 1

    slice_lo = permutedims(sim_lohi.sino_low[:, mid_r, :], (2, 1))
    slice_hi = permutedims(sim_lohi.sino_high[:, mid_r, :], (2, 1))

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
        (1, 1, "Low Bins", "1 + 2 · 20 – 55 keV", slice_lo),
        (1, 2, "High Bins", "3 + 4 · > 55 keV", slice_hi),
    )

    for (r, c, ttl, sub, slice) in panels
        ax = CM.Axis(fig[r, c]; title = ttl, subtitle = sub, axis_kwargs...)
        CM.heatmap!(ax, slice; colormap = :viridis, colorrange = sino_window)
    end
    CM.Colorbar(
        fig[1, 3]; colormap = :viridis, colorrange = sino_window,
        label = "Log Line Integral", width = 16, labelsize = 22, ticklabelsize = 18,
    )
    fig
end

# ╔═╡ 06000009-0000-4000-8000-000000000001
md"""
## 8. Projection Domain Material Decomposition

Two-step polynomial decomposition (Alvarez & Macovski 1976):

1. **Calibrate** — build per-bin-group effective spectra from the
   source spectrum × CdTe η × MC-DRM column sum, forward-project a
   synthetic step-wedge across `(t_water, t_iodine)` Chebyshev grids
   through each group's spectrum, and least-squares-fit a 4th-order
   bivariate polynomial inverse `(p_low, p_high) → (t_water, t_iodine)`.
2. **Apply** — evaluate the polynomial per ray on the `(low, high)`
   sinogram pair → `(sino_water, sino_iodine)`.  Threaded CPU loop.

Output sinograms are per-ray basis line integrals:

```
sino_iodine   (g/cm² of iodine along the ray)
sino_water    (cm of pure-water-equivalent path along the ray)
```

!!! info "Why a Polynomial?"
    The polynomial captures **beam hardening** through the polychromatic
    transmission integral — that's the key thing a linear closed-form
    2×2 inversion misses.  On a 33 cm phantom the residual cup from a
    linear inverse becomes the dominant artifact.  The forward-projected
    step-wedge calibration makes the inverse beam-hardening-aware
    by construction.
"""

# ╔═╡ 06000009-0000-4000-8000-000000000005
# Polynomial calibration knobs.
begin
    cmv_low_bins = 1:2          # PCCT bins in the "low" channel
    cmv_high_bins = 3:4          # PCCT bins in the "high" channel
    cmv_poly_order = 4            # bivariate total-order polynomial
    cmv_n_water = 40           # Chebyshev grid size (water axis)
    cmv_n_iodine = 25           # Chebyshev grid size (iodine axis)
    cmv_max_water_cm = 50.0         # step-wedge max water path (cm)
    cmv_max_iodine_g_cm2 = 0.15         # step-wedge max iodine area density (g/cm²)
end

# ╔═╡ 06000009-0000-4000-8000-000000000010
pcct_vmi_calibration = BS.calibrate_pcct_vmi_poly(
    scanner, protocol;
    sim_opts = sim_opts,
    low_bins = cmv_low_bins,
    high_bins = cmv_high_bins,
    order = cmv_poly_order,
    n_water = cmv_n_water,
    n_iodine = cmv_n_iodine,
    max_water_cm = cmv_max_water_cm,
    max_iodine_g_cm2 = cmv_max_iodine_g_cm2,
);

# ╔═╡ 06000009-0000-4000-8000-000000000020
sino_basis = let
    sl = Float32.(sim_lohi.sino_low)
    sh = Float32.(sim_lohi.sino_high)
    sino_water, sino_iodine = BS.apply_pcct_vmi_poly(sl, sh, pcct_vmi_calibration)

    @info "[Polynomial decomp] ⟨∫ρ_W·dr⟩ = $(round(mean(sino_water), sigdigits = 4)) cm   " *
        "⟨∫ρ_I·dr⟩ = $(round(mean(sino_iodine), sigdigits = 4)) g/cm²"

    (
        sino_iodine = sino_iodine,
        sino_water = sino_water,
        geom = sim_lohi.geom,
    )
end;

# ╔═╡ 06000009-0000-4000-8000-000000000040
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
            1, 3, 4, "Water Basis Sinogram", "cm",
            slice_wat, _qrange(slice_wat),
        ),
    )

    for (r, panel_c, cbar_c, ttl, cbar_label, slice, range) in panels
        ax = CM.Axis(fig[r, panel_c]; title = ttl, axis_kwargs...)
        CM.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        CM.Colorbar(
            fig[r, cbar_c]; colormap = :viridis, colorrange = range,
            label = cbar_label, width = 16, labelsize = 22, ticklabelsize = 18,
        )
    end
    fig
end

# ╔═╡ 0600000a-0000-4000-8000-000000000001
md"""
## 9. FBP: Iodine and Water Basis Maps

Two FDK passes with a PCCT-tuned apodization filter (sharper than
`SoftFilter` to recover the higher native PCCT spatial resolution).
The iodine + water reconstructions land in basis-density units (g/cm³)
directly; no post-decomposition step needed.
"""

# ╔═╡ 0600000a-0000-4000-8000-000000000010
basis_volumes = let
    matrix_size = recon_opts.matrix_size
    geom = sino_basis.geom

    fdk_filter = BS.CustomFilter(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        (1.0, 0.75, 0.6, 0.2, 0.001),
    )

    function _fbp(sino_cpu)
        sino_gpu = to_gpu(Float32.(sino_cpu))
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, geom, matrix_size; filter = fdk_filter,
        )
        recon = Array(BS.reconstruct!(ws, sino_gpu, geom, matrix_size))
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

# ╔═╡ 0600000a-0000-4000-8000-000000000030
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
            aspect = CM.DataAspect(), axis_kwargs...,
        )
        CM.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        CM.hidedecorations!(ax)
        CM.Colorbar(
            fig[r, cbar_c]; colormap = :viridis, colorrange = range,
            label = cbar_label, width = 16, labelsize = 22, ticklabelsize = 18,
        )
    end
    fig
end

# ╔═╡ 0600000b-0000-4000-8000-000000000001
md"""
## 10. Z-Direction Median Filter

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

# ╔═╡ 0600000b-0000-4000-8000-000000000005
Z_MEDIAN_ADJACENT = 2;

# ╔═╡ 0600000b-0000-4000-8000-000000000010
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

# ╔═╡ 0600000b-0000-4000-8000-000000000030
let
    fig = CM.Figure(size = (1180, 580))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    mid = size(basis_z.vol_iodine, 3) ÷ 2

    _qrange(arr) = (
        Float64(quantile(vec(arr), 0.01)),
        Float64(quantile(vec(arr), 0.99)),
    )

    slice_iod = basis_z.vol_iodine[:, :, mid]
    slice_wat = basis_z.vol_water[:, :, mid]

    panels = (
        (1, 1, 2, "Iodine Basis", "g/cm³", slice_iod, _qrange(slice_iod)),
        (1, 3, 4, "Water Basis", "g/cm³", slice_wat, _qrange(slice_wat)),
    )

    for (r, panel_c, cbar_c, ttl, cbar_label, slice, range) in panels
        ax = CM.Axis(
            fig[r, panel_c]; title = ttl,
            aspect = CM.DataAspect(), axis_kwargs...,
        )
        CM.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        CM.hidedecorations!(ax)
        CM.Colorbar(
            fig[r, cbar_c]; colormap = :viridis, colorrange = range,
            label = cbar_label, width = 16, labelsize = 22, ticklabelsize = 18,
        )
    end
    fig
end

# ╔═╡ 0600000c-0000-4000-8000-000000000001
md"""
## 11. VMI Synthesis

`BS.synth_vmi_2basis(c_water, c_iodine; energy_keV)` evaluates the
textbook 2-basis linear mix (McCollough 2015) at the target keV:

```
μ(E)  = c_water(r) · (μ/ρ)_water(E) + c_iodine(r) · (μ/ρ)_iodine(E)
HU(E) = 1000 · (μ(E) − (μ/ρ)_water(E)) / (μ/ρ)_water(E)
```

The denominator is the **mono-energetic** linear attenuation of pure
water at the target VMI energy from NIST tables.  VMI grid:
40, 70, 100, 140 keV.

!!! note "Solid-Water Diagnostic"
    The `solid_water_basis` cell below measures `⟨c_water⟩` and
    `⟨c_iodine⟩` over a deeply-eroded solid-water ROI.  Its
    synth-evaluated μ_water at each VMI energy is logged next to the
    textbook mono divisor as a Δ% drift.  This is **diagnostic only** —
    after the SVD sinogram denoiser + polynomial decomposition, the
    residual bias is small enough that the textbook analytical divisor
    recovers correct HUs directly without needing an empirical anchor.
"""

# ╔═╡ 0600000c-0000-4000-8000-000000000010
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
    @info "solid_water_basis: ⟨c_water⟩_SW = $(round(c_w, digits = 4)) g/cm³, " *
        "⟨c_iodine⟩_SW = $(round(c_i, digits = 6)) g/cm³"

    (
        c_water = c_w, c_iodine = c_i, n_voxels = length(sw_idx) * n_z,
        mask_2d = collect(sw_bool),
    )   # for downstream viz
end;

# ╔═╡ 0600000c-0000-4000-8000-000000000015
pcct_vmi_energies = [40.0, 70.0, 100.0, 140.0];

# ╔═╡ 0600000c-0000-4000-8000-000000000020
vmi_HU_by_keV = let
    # `BS.synth_vmi_2basis` expects c_iodine in mg/mL; our basis maps
    # are in g/cm³ (= g/mL).  Multiply by 1000 to convert.
    c_iodine_mg_per_mL = basis_z.vol_iodine .* 1000.0f0

    out = Dict{Float64, Array{Float32, 3}}()
    for E in pcct_vmi_energies
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
            basis_z.vol_water, c_iodine_mg_per_mL;
            energy_keV = E,
        )
    end
    out
end;

# ╔═╡ 0600000c-0000-4000-8000-000000000040
let
    HU_window = (-200, 500)

    fig = CM.Figure(size = (1180, 1180))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    sample = vmi_HU_by_keV[40.0]
    mid = size(sample, 3) ÷ 2

    for (k, E) in enumerate(pcct_vmi_energies)
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

    CM.save(
        joinpath(@__DIR__, "..", "assets", "pcct_vmi_projection_grid.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0600000d-0000-4000-8000-000000000001
md"""
## 12. VMI Post-Processing (Mono+)

Frequency-split rule from Grant et al. 2014:

```
Mono+(E)     = LP_σ(VMI_E) + VMI_opt − LP_σ(VMI_opt)
Mono+(E_opt) = VMI_opt   (identity at the noise-optimal anchor)
```

Low frequencies come from the noise-optimal anchor (`E_opt = 70 keV`);
high frequencies (edges, fine detail) come from the target energy `E`.

!!! tip "Per-keV σ Tuning"
    `σ_vmi_lp_px` pairs element-wise with `pcct_vmi_energies` — one σ
    per VMI energy.

    | σ          | Behavior                                      |
    |------------|-----------------------------------------------|
    | `0`        | Identity (no LP, no FFT — `Mono+(E) = VMI_E`) |
    | `> 0`      | LP band replaced with the anchor's LP         |
    | larger σ   | Broader anchor-driven smoothing; edges blur   |

    Default `[1.0, 0.0, 1.0, 1.0]`: smooth 40 / 100 / 140 toward the
    70-keV anchor (reduces low-keV iodine speckle and high-keV
    photon-starvation streaks); 70 keV stays identity by definition.
"""

# ╔═╡ 0600000d-0000-4000-8000-000000000005
# Per-keV Gaussian LP σ in pixels — paired with `pcct_vmi_energies` order.
# σ = 0  ⇒ identity at that energy (Mono+(E) = VMI_E exactly, no FFT).
# σ > 0  ⇒ that energy's LP band is replaced with the 70-keV anchor's LP.
# Edit these to tune per-energy noise/contrast trade-off.
# (40, 70, 100, 140) keV
σ_vmi_lp_px = Float64[1.0, 0.0, 1.0, 1.0];

# ╔═╡ 0600000d-0000-4000-8000-000000000010
vmi_HU_final = let
    volumes = [vmi_HU_by_keV[E] for E in pcct_vmi_energies]

    ws = BS.create_mono_plus_workspace(
        volumes[1];
        n_energies = length(pcct_vmi_energies),
    )
    BS.apply_mono_plus!(
        ws, volumes, pcct_vmi_energies;
        E_noise_opt = 70.0,
        σ_lp_px = σ_vmi_lp_px,
        verbose = true,
    )

    # ws.out_vols is reused on subsequent apply_mono_plus! calls — copy
    # into a Dict so downstream cells (Results plots) hold their own arrays.
    out = Dict{Float64, Array{Float32, 3}}()
    for (i, E) in enumerate(pcct_vmi_energies)
        out[E] = copy(ws.out_vols[i])
    end
    ws = nothing; GC.gc(true)
    out
end;

# ╔═╡ 0600000d-0000-4000-8000-000000000030
let
    HU_window = (-200, 500)

    fig = CM.Figure(size = (1180, 1180))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    sample = vmi_HU_final[40.0]
    mid = size(sample, 3) ÷ 2

    for (k, E) in enumerate(pcct_vmi_energies)
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

    CM.save(
        joinpath(@__DIR__, "..", "assets", "pcct_vmi_projection_monoplus.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0600000e-0000-4000-8000-000000000001
md"""
## Results

Per-rod measured vs theoretical HU at the canonical four VMI energies
(40 / 70 / 100 / 140 keV).

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
    K-edge at 33.2 keV so 40 keV amplifies iodine HU dramatically vs
    the 70+ keV plateau).

    Solid line = measured.  Dashed line = theoretical.  Tight overlay
    means the projection-domain pipeline is recovering the underlying
    physics correctly.
"""

# ╔═╡ 0600000e-0000-4000-8000-000000000010
ROD_LABELS = (
    Ca = (UInt8(10), UInt8(11), UInt8(12), UInt8(13), UInt8(14), UInt8(15), UInt8(16)),
    I = (UInt8(20), UInt8(21), UInt8(22), UInt8(23), UInt8(24), UInt8(25), UInt8(26)),
);

# ╔═╡ 0600000e-0000-4000-8000-000000000020
ROD_NAMES = (
    Ca = ("50 mg/mL", "100 mg/mL", "200 mg/mL", "300 mg/mL", "400 mg/mL", "500 mg/mL", "600 mg/mL"),
    I = ("2.0 mg/mL", "2.5 mg/mL", "5.0 mg/mL", "7.5 mg/mL", "10.0 mg/mL", "15.0 mg/mL", "20.0 mg/mL"),
);

# ╔═╡ 0600000e-0000-4000-8000-000000000030
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
            for E in pcct_vmi_energies
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
        n_E = length(pcct_vmi_energies)
        meas = zeros(Float64, n_rods, n_E)
        theo = zeros(Float64, n_rods, n_E)
        for (i, lab) in pairs(labels)
            mat = materials[Int(lab) + 1]   # mask_value + 1
            for (j, E) in pairs(pcct_vmi_energies)
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

# ╔═╡ 0600000e-0000-4000-8000-000000000002
md"""
### Water ROI

Left panel: the deeply-eroded solid-water ROI (12-px erosion ≈ 8 mm)
overlaid in red on the 70 keV Mono+ slice — exactly the voxels feeding
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

# ╔═╡ 0600000e-0000-4000-8000-000000000005
let
    fig = CM.Figure(size = (1180, 580))

    # ─── Left panel — 70 keV Mono+ slice + eroded SW ROI overlay ───────
    HU_window = (-200, 500)
    mid = size(vmi_HU_final[70.0], 3) ÷ 2
    bg = vmi_HU_final[70.0][:, :, mid]

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
    n_z = size(vmi_HU_final[70.0], 3)
    function _mean_hu(vol)
        s = 0.0; n = 0
        for z in 1:n_z, ci in sw_idx
            s += vol[ci, z]; n += 1
        end
        return s / n
    end
    sw_hu_per_keV = [_mean_hu(vmi_HU_final[E]) for E in pcct_vmi_energies]

    # cgrad → Vector{RGBAf} for barplot's color kwarg
    n_E = length(pcct_vmi_energies)
    bar_colors = [CM.cgrad(:plasma, n_E; categorical = true)[i] for i in 1:n_E]

    ax2 = CM.Axis(
        fig[1, 2];
        title = "Water Region Mean HU",
        subtitle = "Per VMI Energy",
        xlabel = "VMI Energy (keV)", ylabel = "HU",
        xticks = (collect(1:n_E), ["$(Int(E))" for E in pcct_vmi_energies]),
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
        joinpath(@__DIR__, "..", "assets", "pcct_vmi_water_roi_check.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0600000f-0000-4000-8000-000000000001
md"""
### Per-Rod Regression
"""

# ╔═╡ 0600000f-0000-4000-8000-000000000010
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
            xticks = pcct_vmi_energies,
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
                ax, pcct_vmi_energies, vec(d.measured[i, :]);
                color = color, linewidth = 2.5, markersize = 9,
            )
            CM.lines!(
                ax, pcct_vmi_energies, vec(d.theoretical[i, :]);
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
        joinpath(@__DIR__, "..", "assets", "pcct_vmi_vs_theoretical.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0600000f-0000-4000-8000-000000000020
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

# ╔═╡ 0600000f-0000-4000-8000-000000000030
let
    fig = CM.Figure(size = (1000, 1200))

    # One color per VMI energy (cool→warm sweep)
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

        for (j, E) in pairs(pcct_vmi_energies)
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
        joinpath(@__DIR__, "..", "assets", "pcct_vmi_regression.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 06000010-0000-4000-8000-000000000001
md"""
## Summary

```
Simulate 140 kVp PCCT (4 bins, scatter-injected)
   → Per-Bin Scatter Correction  (model-based)
   → Joint Sinogram SVD Denoiser  (4-channel, BS.apply_sino_svd_denoise)
   → Bin Combine  (1+2 → low,  3+4 → high)
   → Projection-Domain Material Decomposition  (polynomial, Alvarez-Macovski)
   → FBP × 2  (iodine, water basis maps)
   → Z-Direction Median Filter × 2
   → Monoenergetic VMI Synthesis  (textbook 2-basis, mono μρ_water divisor)
   → Mono+ Post-Processing  (per-keV σ via σ_vmi_lp_px)
   → Measured vs Theoretical Per-Rod Regression  at 40 / 70 / 100 / 140 keV
```

The 4-channel SVD denoiser exploits cross-bin correlation *before* the
lossy bin-combine step, then the polynomial material decomposition
handles beam hardening through the polychromatic transmission integral
that a linear closed-form inversion misses on a 33 cm phantom.  The
result is HU-quantitative VMIs with low streak content and clean
low-keV Mono+ output.
"""

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
# ╟─06000005-0000-4000-8000-000000000001
# ╠═06000005-0000-4000-8000-000000000010
# ╠═06000005-0000-4000-8000-000000000020
# ╟─06000006-0000-4000-8000-000000000001
# ╠═06000006-0000-4000-8000-000000000010
# ╟─06000006-0000-4000-8000-000000000040
# ╟─06000007-0000-4000-8000-000000000001
# ╠═06000007-0000-4000-8000-000000000005
# ╠═06000007-0000-4000-8000-000000000010
# ╟─06000007-0000-4000-8000-000000000030
# ╟─06000008-0000-4000-8000-000000000001
# ╠═06000008-0000-4000-8000-000000000010
# ╟─06000008-0000-4000-8000-000000000030
# ╟─06000009-0000-4000-8000-000000000001
# ╠═06000009-0000-4000-8000-000000000005
# ╠═06000009-0000-4000-8000-000000000010
# ╠═06000009-0000-4000-8000-000000000020
# ╟─06000009-0000-4000-8000-000000000040
# ╟─0600000a-0000-4000-8000-000000000001
# ╠═0600000a-0000-4000-8000-000000000010
# ╟─0600000a-0000-4000-8000-000000000030
# ╟─0600000b-0000-4000-8000-000000000001
# ╠═0600000b-0000-4000-8000-000000000005
# ╠═0600000b-0000-4000-8000-000000000010
# ╟─0600000b-0000-4000-8000-000000000030
# ╟─0600000c-0000-4000-8000-000000000001
# ╠═0600000c-0000-4000-8000-000000000010
# ╠═0600000c-0000-4000-8000-000000000015
# ╠═0600000c-0000-4000-8000-000000000020
# ╟─0600000c-0000-4000-8000-000000000040
# ╟─0600000d-0000-4000-8000-000000000001
# ╠═0600000d-0000-4000-8000-000000000005
# ╠═0600000d-0000-4000-8000-000000000010
# ╟─0600000d-0000-4000-8000-000000000030
# ╟─0600000e-0000-4000-8000-000000000001
# ╠═0600000e-0000-4000-8000-000000000010
# ╠═0600000e-0000-4000-8000-000000000020
# ╠═0600000e-0000-4000-8000-000000000030
# ╟─0600000e-0000-4000-8000-000000000002
# ╟─0600000e-0000-4000-8000-000000000005
# ╟─0600000f-0000-4000-8000-000000000001
# ╟─0600000f-0000-4000-8000-000000000010
# ╟─0600000f-0000-4000-8000-000000000020
# ╟─0600000f-0000-4000-8000-000000000030
# ╟─06000010-0000-4000-8000-000000000001
