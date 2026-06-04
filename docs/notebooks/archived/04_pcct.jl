### A Pluto.jl notebook ###
# v0.19.0

# ╔═╡ 04000001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 04000001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 04000001-0000-4000-8000-000000000003
using Statistics: mean, std

# ╔═╡ 04000001-0000-4000-8000-000000000010
md"""
# 04 · PCCT on Gammex 472 — straight FBP per bin + combined low/high

Siemens Naeotom Alpha photon-counting CT, 4-threshold standard-mode
acquisition.  Same setup style as nb03's straight-FBP dual-kVp
notebook — *no* synthesis, no decomposition, no RSKR.

Pipeline:

```
PCCT simulate! (4 bins) → per-bin scatter correction
                       → FBP per bin           → 4 μ → 4 HU volumes
                       → combine (1+2, 3+4)    → 2 μ → 2 HU volumes
                       → 2 visualizations
```

Two visualizations:
1. **All 4 bins** — bin1 (20–35 keV), bin2 (35–55), bin3 (55–70), bin4 (>70)
2. **Combined low / high** — `low = bins 1+2`, `high = bins 3+4`
   via I0-weighted Beer recombination (McCollough 2015 / Yu 2012)

No quantitative comparison vs theory — for the polychromatic-residual
explanation, see the §11 markdown of `03_dual_kvp.jl`.  This notebook
is the "look at the recons" half.
"""

# ╔═╡ 04000001-0000-4000-8000-000000000020
md"""
## Setup
"""

# ╔═╡ 04000001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 04000001-0000-4000-8000-000000000031
import CairoMakie as CM

# ╔═╡ 04000001-0000-4000-8000-000000000040
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

# ╔═╡ 04000001-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 04000002-0000-4000-8000-000000000001
md"""
## 1. `Phantom` — Gammex Model 472
"""

# ╔═╡ 04000002-0000-4000-8000-000000000010
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,
    n_slices = 16,
    fov_cm = 35.0,
    z_cm = 1.0,
);

# ╔═╡ 04000002-0000-4000-8000-000000000020
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 04000003-0000-4000-8000-000000000001
md"""
## 2. `Scanner` — Siemens Naeotom Alpha (PCCT, 4-threshold)

CdTe direct-conversion detector, native dexels 0.275 × 0.322 mm at
the detector face, 2×2 binned in DAS.

Energy thresholds: T = [20, 35, 55, 70] keV → 4 bins.

| Bin | Range (keV) |
|-----|-------------|
| 1   | 20 – 35     |
| 2   | 35 – 55     |
| 3   | 55 – 70     |
| 4   |  > 70       |

`scan_diameter = 360 mm` is a memory-saving approximation (clinical
Naeotom is 500 mm); enough margin for the 330 mm Gammex 472 phantom.
"""

# ╔═╡ 04000003-0000-4000-8000-000000000010
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

# ╔═╡ 04000004-0000-4000-8000-000000000001
md"""
## 3. `CTProtocol` — 140 kVp / 174 mA / ~10 mGy CTDIvol

Matches nb07 Scan 2 (clinical 140 kVp / 174 mA / 10.12 mGy CTDIvol).
`additional_filters = [("Ti", 0.9)]` is the Vectron tube's inherent
0.9 mm Ti window on top of the 3 mm Al flat filter.
"""

# ╔═╡ 04000004-0000-4000-8000-000000000010
protocol = BS.CTProtocol(
    kVp = 140,
    mA = 174.0,
    views = 1200,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Ti", 0.9)],
);

# ╔═╡ 04000005-0000-4000-8000-000000000001
md"""
## 4. `SimOptions` and `ReconOptions`

`fidelity = :pcct` switches the simulator into the photon-counting
path (per-bin sinograms + DRM + Compton scatter modeling).
"""

# ╔═╡ 04000005-0000-4000-8000-000000000010
sim_opts = BS.SimOptions(
    fidelity = :pcct,
    seed = 1234,
    pcct_noise_reduction = 0.3,
);

# ╔═╡ 04000005-0000-4000-8000-000000000020
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

# ╔═╡ 04000006-0000-4000-8000-000000000001
md"""
## 5. PCCT simulation + per-bin scatter correction

`BS.simulate!` returns the 4 per-bin sinograms + per-bin I0 + the
exact spatial scatter field + per-bin scatter weights.  We apply the
known scatter correction per bin (no blind re-estimation needed for
simulated data) and return the corrected per-bin sinograms +
geometry + I0 vector for downstream cells.
"""

# ╔═╡ 04000006-0000-4000-8000-000000000010
sim_bins = let
    @info "Simulating: $(Int(protocol.kVp)) kVp / $(round(protocol.mA, digits = 1)) mA (PCCT 4-bin)…"
    ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    result = BS.simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)

    geom = ws.geom
    bins = [Array(b) for b in result.pcct_sino.bins]   # Float32 each
    I0_bins = copy(result.I0_bins)

    # Per-bin scatter correction — model-based, exact for simulated data.
    let
        I0_total = Float32(sum(I0_bins))
        eps_f = Float32(1.0e-10)

        # Re-estimate scatter from recombined primary (mirrors nb07 clinical flow).
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

        # Per-bin scatter fractions from spectrum × η × DRM.
        e_full, w_full = BS.resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner = scanner)
        pcct_det = BS._build_pcct_detector(scanner)
        kVp_val = Float64(maximum(e_full))
        R_mat = BS.compute_mc_drm(pcct_det, kVp_val)
        η_vec = BS.quantum_efficiency_vector(pcct_det.material, pcct_det.thickness_mm, e_full)
        ew = BS.compute_scatter_energy_weights(Float64.(e_full))
        scatter_fracs = BS.compute_scatter_bin_weights(
            Float64.(e_full), Float64.(w_full), ew, Float64.(η_vec), R_mat, kVp_val,
        )

        # In-place per-bin subtraction.
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

# ╔═╡ 04000007-0000-4000-8000-000000000001
md"""
## 6. FBP per bin → 4 μ-domain volumes

Each per-bin sinogram is FBP-reconstructed independently with a
slightly sharper apodization (the same `(0.0,0.25,0.5,0.75,1.0) →
(1.0,0.75,0.6,0.2,0.001)` curve nb04's clinical verification uses for
PCCT bins).  No BHC, no RSKR, no cupping correction.
"""

# ╔═╡ 04000007-0000-4000-8000-000000000010
bin_μ = let
    matrix_size = recon_opts.matrix_size
    geom = sim_bins.geom

    fdk_filter = BS.CustomFilter(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        (1.0, 0.75, 0.6, 0.2, 0.001),
    )

    function _fbp_to_μ(sino_cpu)
        sino_gpu = to_gpu(Float32.(sino_cpu))
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, geom, matrix_size; filter = fdk_filter,
        )
        recon_μ = Array(BS.reconstruct!(ws, sino_gpu, geom, matrix_size))
        ws = nothing; sino_gpu = nothing
        GC.gc(true)
        return Float32.(recon_μ)
    end

    [_fbp_to_μ(s) for s in sim_bins.bins]
end;

# ╔═╡ 04000008-0000-4000-8000-000000000001
md"""
## 7. Bin combine — 4 → low / high pair via I0-weighted Beer

Standard McCollough-2015 / Yu-2012 recombination:

* **low**  = bins **1 + 2** (20 – 55 keV)
* **high** = bins **3 + 4** ( > 55 keV)

```
N_grp = Σ_{b∈grp} I0[b] · exp(-p[b])
p_grp = -log(N_grp / Σ_{b∈grp} I0[b])
```

Each combined sinogram represents a polychromatic measurement at the
I0-weighted average spectrum of its bin group.
"""

# ╔═╡ 04000008-0000-4000-8000-000000000010
sim_lohi = let
    eps_f = Float32(1.0e-10)

    function _combine(bin_indices)
        I0_sum = Float32(sum(Float64.(sim_bins.I0_bins[bin_indices])))
        N = zeros(Float32, size(sim_bins.bins[1]))
        for b in bin_indices
            I0b = Float32(sim_bins.I0_bins[b])
            @. N += I0b * exp(-sim_bins.bins[b])
        end
        return @. -log(max(N, eps_f) / I0_sum)
    end

    sino_low = _combine([1, 2])
    sino_high = _combine([3, 4])

    (sino_low = sino_low, sino_high = sino_high, geom = sim_bins.geom)
end;

# ╔═╡ 04000009-0000-4000-8000-000000000001
md"""
## 8. FBP per combined sinogram → 2 μ-domain volumes

Same FDK + filter as §6, applied to the combined low / high sinograms.
"""

# ╔═╡ 04000009-0000-4000-8000-000000000010
lohi_μ = let
    matrix_size = recon_opts.matrix_size
    geom = sim_lohi.geom

    fdk_filter = BS.CustomFilter(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        (1.0, 0.75, 0.6, 0.2, 0.001),
    )

    function _fbp_to_μ(sino_cpu)
        sino_gpu = to_gpu(Float32.(sino_cpu))
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, geom, matrix_size; filter = fdk_filter,
        )
        recon_μ = Array(BS.reconstruct!(ws, sino_gpu, geom, matrix_size))
        ws = nothing; sino_gpu = nothing
        GC.gc(true)
        return Float32.(recon_μ)
    end

    (
        vol_low_μ = _fbp_to_μ(sim_lohi.sino_low),
        vol_high_μ = _fbp_to_μ(sim_lohi.sino_high),
        geom = geom,
    )
end;

# ╔═╡ 0400000a-0000-4000-8000-000000000001
md"""
## 9. `μ_water` for combined low / high — measured from FBP center ROI

8-px circular ROI at the phantom center, mean across all z slices.
By construction this makes solid water read at exactly 0 HU after
`to_hounsfield`.  Same trick nb03_dual_kvp uses — the divisor
absorbs each combined volume's polychromatic offset for the water
region.  The 4 bin volumes stay in native μ — no HU divisor needed.
"""

# ╔═╡ 0400000a-0000-4000-8000-000000000010
μ_water_per_vol = let
    nx, ny, nz = size(lohi_μ.vol_low_μ)
    cx, cy = nx / 2 + 0.5, ny / 2 + 0.5
    ROI_R = 8.0
    r² = ROI_R^2

    roi = CartesianIndex{2}[]
    i_lo = max(1, floor(Int, cx - ROI_R)); i_hi = min(nx, ceil(Int, cx + ROI_R))
    j_lo = max(1, floor(Int, cy - ROI_R)); j_hi = min(ny, ceil(Int, cy + ROI_R))
    for j in j_lo:j_hi, i in i_lo:i_hi
        ((i - cx)^2 + (j - cy)^2) ≤ r² && push!(roi, CartesianIndex(i, j))
    end

    function _mean_μ(vol)
        s = 0.0; n = 0
        for z in 1:nz, ci in roi
            s += vol[ci, z]; n += 1
        end
        return s / n
    end

    (
        low = Float64(_mean_μ(lohi_μ.vol_low_μ)),
        high = Float64(_mean_μ(lohi_μ.vol_high_μ)),
    )
end;

# ╔═╡ 0400000b-0000-4000-8000-000000000001
md"""
## 10. HU conversion (combined low / high only)
"""

# ╔═╡ 0400000b-0000-4000-8000-000000000020
lohi_HU = (
    vol_low_HU = Float32.(BS.to_hounsfield(lohi_μ.vol_low_μ; μ_water = μ_water_per_vol.low)),
    vol_high_HU = Float32.(BS.to_hounsfield(lohi_μ.vol_high_μ; μ_water = μ_water_per_vol.high)),
);

# ╔═╡ 0400000c-0000-4000-8000-000000000001
md"""
## Visualization 1 — All 4 PCCT bins (native μ)

Bin1 (20–35 keV) shows the strongest iodine + Ca contrast — soft
photons are most attenuated by high-Z materials.  Bin4 (> 70 keV) is
nearly Compton-only — flat across materials, lowest contrast.
The middle bins (2 + 3) span the transition.
"""

# ╔═╡ 0400000c-0000-4000-8000-000000000010
let
    fig = CM.Figure(size = (1400, 1300))
    title_kwargs = (titlesize = 28, subtitlesize = 20)

    mid = size(bin_μ[1], 3) ÷ 2

    panels = (
        (1, 1, "Bin 1", "20 – 35 keV", bin_μ[1]),
        (1, 2, "Bin 2", "35 – 55 keV", bin_μ[2]),
        (2, 1, "Bin 3", "55 – 70 keV", bin_μ[3]),
        (2, 2, "Bin 4", "> 70 keV", bin_μ[4]),
    )

    μ_window = (0.0, 0.5)

    hms = nothing
    for (r, c, ttl, sub, vol) in panels
        ax = CM.Axis(
            fig[r, c];
            title = ttl, subtitle = sub,
            aspect = CM.DataAspect(),
            title_kwargs...,
        )
        hms = CM.heatmap!(ax, vol[:, :, mid]; colormap = :grays, colorrange = μ_window)
        CM.hidedecorations!(ax)
    end

    CM.Colorbar(fig[1:2, 3], hms; label = "μ (cm⁻¹)", width = 14, labelsize = 20)

    CM.save(
        joinpath(@__DIR__, "..", "assets", "pcct_4bin.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0400000d-0000-4000-8000-000000000001
md"""
## Visualization 2 — Combined low / high pair (HU)

`low = bins 1+2` (20–55 keV) and `high = bins 3+4` ( > 55 keV) — the
canonical input for image-domain DECT decomposition.  The low panel
should look much like a standard 80 kVp clinical recon (high-Z
materials bright); the high panel softens the contrast as the spectrum
shifts into Compton-dominated energies.
"""

# ╔═╡ 0400000d-0000-4000-8000-000000000010
let
    HU_window = (-200, 500)

    fig = CM.Figure(size = (1200, 600))
    title_kwargs = (titlesize = 28, subtitlesize = 20)

    mid = size(lohi_HU.vol_low_HU, 3) ÷ 2

    panels = (
        (1, 1, "20 – 55 keV", "Bins 1 + 2", lohi_HU.vol_low_HU[:, :, mid]),
        (1, 2, "> 55 keV", "Bins 3 + 4", lohi_HU.vol_high_HU[:, :, mid]),
    )

    hms = nothing
    for (r, c, ttl, sub, slice) in panels
        ax = CM.Axis(
            fig[r, c];
            title = ttl, subtitle = sub,
            aspect = CM.DataAspect(),
            title_kwargs...,
        )
        hms = CM.heatmap!(ax, slice; colormap = :grays, colorrange = HU_window)
        CM.hidedecorations!(ax)
    end

    CM.Colorbar(fig[1, 3], hms; label = "HU", width = 14, labelsize = 18)

    CM.save(
        joinpath(@__DIR__, "..", "assets", "pcct_lohi.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0400000e-0000-4000-8000-000000000001
md"""
## Summary

```
PCCT simulate! (4 bins, scatter-injected)
   → per-bin scatter correction (model-based, no blind re-estimation)
   → FBP per bin       → 4 μ              (Visualization 1, native μ)
   → bin combine        → 2 sinograms
   → FBP per combined   → 2 μ → 2 HU      (Visualization 2)
```

No BHC, no RSKR, no cupping correction, no decomposition, no synthesis,
no post-processing — every one of those exists in the archived
`04_pcct_vmi.jl` to push toward the polychromatic-residual ceiling
discussed in `03_dual_kvp.jl`'s §11.

Companion to `03_dual_kvp.jl` — that one is dual-source EICT, this
one is single-source PCCT, both straight-FBP only.  Together they
show what the four "raw" channels look like before any decomposition
or synthesis layer.
"""

# ╔═╡ Cell order:
# ╟─04000001-0000-4000-8000-000000000010
# ╟─04000001-0000-4000-8000-000000000020
# ╠═04000001-0000-4000-8000-000000000001
# ╠═04000001-0000-4000-8000-000000000002
# ╠═04000001-0000-4000-8000-000000000003
# ╠═04000001-0000-4000-8000-000000000030
# ╠═04000001-0000-4000-8000-000000000031
# ╠═04000001-0000-4000-8000-000000000040
# ╟─04000001-0000-4000-8000-000000000050
# ╟─04000002-0000-4000-8000-000000000001
# ╠═04000002-0000-4000-8000-000000000010
# ╠═04000002-0000-4000-8000-000000000020
# ╟─04000003-0000-4000-8000-000000000001
# ╠═04000003-0000-4000-8000-000000000010
# ╟─04000004-0000-4000-8000-000000000001
# ╠═04000004-0000-4000-8000-000000000010
# ╟─04000005-0000-4000-8000-000000000001
# ╠═04000005-0000-4000-8000-000000000010
# ╠═04000005-0000-4000-8000-000000000020
# ╟─04000006-0000-4000-8000-000000000001
# ╠═04000006-0000-4000-8000-000000000010
# ╟─04000007-0000-4000-8000-000000000001
# ╠═04000007-0000-4000-8000-000000000010
# ╟─04000008-0000-4000-8000-000000000001
# ╠═04000008-0000-4000-8000-000000000010
# ╟─04000009-0000-4000-8000-000000000001
# ╠═04000009-0000-4000-8000-000000000010
# ╟─0400000a-0000-4000-8000-000000000001
# ╠═0400000a-0000-4000-8000-000000000010
# ╟─0400000b-0000-4000-8000-000000000001
# ╠═0400000b-0000-4000-8000-000000000020
# ╟─0400000c-0000-4000-8000-000000000001
# ╟─0400000c-0000-4000-8000-000000000010
# ╟─0400000d-0000-4000-8000-000000000001
# ╟─0400000d-0000-4000-8000-000000000010
# ╟─0400000e-0000-4000-8000-000000000001
