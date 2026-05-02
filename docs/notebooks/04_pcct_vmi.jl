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
# 04 · PCCT VMI on Gammex 472 (Siemens Naeotom Alpha)

!!! warning "🚧 IN PROGRESS — not yet matched to nb07"
    This notebook is **work in progress**.  Known gaps vs the clinical
    verification reference (`basis-verification/notebooks/07_siemens_naeotom_alpha_clinical.jl`,
    Scan 2 path):

    - **Memory approximations** in §2 / §3: `scan_diameter = 360 mm`
      (clinical Naeotom is 500 mm) and `views = 600` (clinical is 1200
      single-tube equivalent).  Both are flagged in the relevant cells.
    - **Missing VMI radial cupping correction** (`vmi_cap_*` block in
      nb07, on by default there).  Most likely cause of residual outer-ring
      iodine HU bias in the §17–§18 verification plots.
    - **RSKR `h_param`** uses 2.0 here; nb07 uses 0.5.  Affects noise
      variance, not rod-mean HU.
    - **Mono+ σ at 40 keV** is 2.0 here vs 3.0 in nb07.
    - **No per-VMI HIR** stage.
    - **No Mono+ edge-mask** (eroded phantom mask).
    - **Self-cal ROI** is a hardcoded 8-px radius vs nb07's physical
      ~6 mm radius.

    The scatter correction (now Ohnesorge re-estimation from the combined
    primary) and the per-bin μ_water (now M-matrix bin-effective spectrum
    × η × DRM with measured-phantom hardening) **do** match nb07.  The
    HU regression should improve as the items above get ported.

**One PCCT acquisition (140 kVp / 174 mA / ~10 mGy), four energy bins
combined into a low/high pair, image-domain Ding decomposition, four
virtual monoenergetic images — verified against XrayAttenuation.jl
theory.**

This notebook walks the **photon-counting CT image-domain pipeline**
that the `BasisSimulator.jl` clinical verification suite uses for its
Siemens Naeotom Alpha (CdTe) runs:

```
4-bin PCCT simulation
   → per-bin scatter correction (decoupled, model-based)
   → bin combine (1+2 → low,  3+4 → high)
   → FBP per combined bin (μ-domain)
   → RSKR-2ch joint denoise
   → HU + system noise floor
   → Ding image-domain decomposition  (self-calibrated from rod ROIs)
   → z-median speckle removal on c_iodine
   → VMI synthesis at 40 / 70 / 100 / 140 keV
   → Mono+ post-processing  (FBP-equivalent noise shaping)
   → FOV mask
```

We close with the verification plot of this notebook: **per-rod measured
HU vs theoretical HU** computed directly from XrayAttenuation.jl at each
VMI energy, separated into Ca-rod and I-rod panels so the iodine K-edge
boost reads cleanly.
"""

# ╔═╡ 04000001-0000-4000-8000-000000000020
md"""
## Setup

Same project + GPU detection idiom as notebook 03. The `to_gpu(...)`
one-liner auto-resolves to `MtlArray` / `CuArray` / `ROCArray` /
`Array` based on which backend is installed on the host.
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

Same factory as notebooks 01 / 03.  Each rod is a labeled region of the
mask, so once we have HU volumes back at the end we can read out per-rod
statistics **by mask label** — no polar-coordinate ROI placement needed.

We use **`z_cm = 1.0`, 16 slices** — slightly thicker than the 5 mm
recon slab (§3 collimation).  This gives the FDK enough phantom
material above and below the recon window to suppress cone-beam edge
artifacts that otherwise show up as radial cupping at the rod
positions when the recon volume is too thin.
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
## 2. `Scanner` — Siemens Naeotom Alpha (PCCT)

Standard-mode CdTe direct-conversion detector, native dexels 0.275 ×
0.322 mm at the detector face, **2×2 binned in DAS**.  Forward
projection ray-traces at native resolution and applies detector physics
(charge sharing, pileup, anti-coincidence) before binning down to
displayed pixels.

**Energy thresholds:** 4-threshold clinical configuration (T1 = 20,
T2 = 35, T3 = 55, T4 = 70 keV).  Bins:

| Bin | Range (keV) |
|-----|-------------|
| 1   | 20–35       |
| 2   | 35–55       |
| 3   | 55–70       |
| 4   | > 70        |

!!! warning "APPROX — `scan_diameter = 360 mm` (not the clinical 500 mm)"
    The Naeotom Alpha's clinical scan field is **500 mm** OD.  This
    notebook uses **360 mm** purely as a memory-saving approximation:
    `n_cols` scales linearly with `scan_diameter`, and every sinogram
    buffer (binned + native pre-binning) scales with `n_cols`.  Going
    from 500 → 360 mm trims `n_cols` by ~28% and frees several GB of
    GPU + CPU memory.

    The Gammex 472 phantom is **330 mm** OD, so 360 mm leaves ~15 mm
    of air margin around the phantom — adequate for FBP recon and rod
    HU readout, but *not* representative of the real scanner FOV.
    **Do not** copy these scanner parameters into a clinical-FOV
    notebook without reverting to 500 mm.
"""

# ╔═╡ 04000003-0000-4000-8000-000000000010
scanner = let
    native_col_mm = 0.275                 # at detector face (Konrad 2025)
    native_row_mm = 0.322                 # at detector face (Konrad 2025)
    sid = 610.0                           # source-to-isocenter (mm)
    sdd = 1113.0                          # source-to-detector (mm)
    magnification = sdd / sid             # ~1.824
    bf = 2                                # standard mode: 2×2 binning

    pixel_col_iso = (native_col_mm * bf) / magnification    # ~0.302 mm
    pixel_row_iso = (native_row_mm * bf) / magnification    # ~0.353 mm
    # NOTE: scan_diameter narrowed from clinical 500 mm → 360 mm to fit memory.
    # See the !!! warning above the Scanner cell — Gammex 472 is 33 cm OD so
    # 36 cm covers the phantom with ~1.5 cm air margin.  This is an APPROX
    # for memory; production scans use the full 50 cm.
    n_cols = ceil(Int, 360.0 / pixel_col_iso)               # 36 cm scan diameter (memory approx)

    BS.Scanner(
        source_to_isocenter = sid,
        source_to_detector  = sdd,

        detector_rows       = 144,
        detector_cols       = n_cols,
        detector_row_size   = pixel_row_iso,
        detector_col_size   = pixel_col_iso,
        detector_shape      = BS.CURVED_DETECTOR,
        detector_row_offset = 0.0,
        detector_col_offset = pixel_col_iso / 2,

        focal_spot_width    = 0.4,
        focal_spot_length   = 0.5,
        target_angle        = 7.0,

        gantry_rotation_time = 0.5,
        scan_diameter        = 360.0,           # APPROX — clinical Naeotom is 500 mm; see warning above
        gantry_aperture      = 820.0,

        flat_filter_material  = :aluminum,
        flat_filter_thickness = 3.0,

        detector_material  = :cdte,
        detector_depth     = 1.6,
        fill_factor_row    = 0.95,
        fill_factor_col    = 0.95,
        detection_gain     = 1.0,
        electronic_noise   = 0.0,

        detector_type        = :photon_counting,
        n_energy_bins        = 4,
        energy_thresholds    = [20.0, 35.0, 55.0, 70.0],
        energy_resolution    = 10.0,
        charge_sharing_fwhm  = 0.08,
        dead_time_ns         = 5.0,
        pixel_mode           = :standard,

        native_dexel_col_mm = native_col_mm,
        native_dexel_row_mm = native_row_mm,
        binning_factor      = bf,
    )
end;

# ╔═╡ 04000004-0000-4000-8000-000000000001
md"""
## 3. `CTProtocol` — Scan 2 (140 kVp / 174 mA / ~10 mGy)

The clinical verification notebook (nb07) acquires four scans at varying
dose; **Scan 2** (140 kVp / 174 mA / 10.12 mGy CTDIvol) is the
highest-quality 140 kVp acquisition and matches the dose level used to
populate `BS.SIEMENS_NAEOTOM_ALPHA_140KVP_CAL`.

`additional_filters = [("Ti", 0.9)]` is the Vectron tube's inherent
0.9 mm Ti window (on top of the 3 mm Al flat filter).

Naeotom Alpha is a dual-source system with ~2400 effective views per
rotation; the single-tube equivalent the simulator models is **1200**.
We use **`views = 600`** here as a memory-saving approximation — every
sinogram buffer (binned and native pre-binning) scales linearly with
view count, so halving views frees several GB.  600 single-tube views
is still well above Nyquist for 33 cm rod ROIs at 512² recon (the
displayed columns cap angular resolution before view count does at
this geometry), so rod HUs are unaffected at the magnitudes we care
about.  **For clinical-fidelity reproductions, restore `views = 1200`.**

**Speed-vs-physics tradeoffs** for this notebook:

| Knob | nb04 value | Why |
|------|------------|-----|
| `views`           | 600          | APPROX — clinical Naeotom is 1200 single-tube equivalent; halved here for memory |
| `collimation_mm`  | 5.0          | matches nb07 Scan 2 — wide enough that the FDK reconstructs ~12 slices, the rod ROIs stay well away from the cone-beam z-edges |
| `n_slices` (phantom) | 16        | phantom z slightly bigger than the recon slab so cone-beam rays always see phantom material at the slab edges |
| `n_voxels`        | 512          | preserved — dropping to 256² collapses the 21-px rod radius below ROI tolerance |

PCCT per-ray cost is ~3–5× EICT (native-resolution ray tracing +
per-energy DRM convolution + spatial binning).  This setup is slower
than nb03's two EICT sims, but it gives the FDK enough z-margin that
rod HUs don't get cone-beam-cupped — the *physics* matters more than
matching nb03's exact runtime here.
"""

# ╔═╡ 04000004-0000-4000-8000-000000000010
protocol = BS.CTProtocol(
    kVp = 140,
    mA = 174.0,
    views = 600,                       # APPROX — clinical Naeotom single-tube equiv is 1200 (see warning above)
    rotation_time = 0.5,
    collimation_mm = 5.0,              # matches nb07 Scan 2 — recon slab ~12 slices @ 0.4 mm
    additional_filters = [("Ti", 0.9)],
);

# ╔═╡ 04000005-0000-4000-8000-000000000001
md"""
## 4. `SimOptions` and `ReconOptions`

`fidelity = :pcct` switches the simulator into the photon-counting path
(per-bin sinograms + DRM + Compton scatter modeling).
`pcct_noise_reduction = 0.3` approximates Siemens' DAS-side corrections
(anti-coincidence, gain calibration, pixel interpolation).

FDK reconstruction on a 512² × n grid where **n is derived from the
collimation** so the recon Z-extent matches the simulated z-extent
exactly — preventing the off-by-one slice errors that bit notebook 03.
"""

# ╔═╡ 04000005-0000-4000-8000-000000000010
sim_opts = BS.SimOptions(
    fidelity = :pcct,
    seed = 1234,
    pcct_noise_reduction = 0.3,
);

# ╔═╡ 04000005-0000-4000-8000-000000000020
recon_opts = let
    slice_thickness_mm = 0.4    # native PCCT detector element (standard mode)
    n_recon_slices = max(1, round(Int, protocol.collimation_mm / slice_thickness_mm))
    BS.ReconOptions(
        algorithm    = :fdk,
        matrix_size  = (512, 512, n_recon_slices),
        fov_cm       = 35.0,
        z_cm         = protocol.collimation_mm / 10.0,
        filter       = :standard,
        vmi_basis    = [:water, :iodine],
        vmi_energies = [40.0, 70.0, 100.0, 140.0],
    )
end;

# ╔═╡ 04000007-0000-4000-8000-000000000001
md"""
## 5. PCCT simulation

`BS.simulate!` runs the **full PCCT chain**:

1. native-resolution ray tracing through the phantom
2. per-energy spectral attenuation (`(n_E)` filtered spectrum)
3. detector physics — DRM, charge sharing, pileup, anti-coincidence
4. spatial binning down to `bf × bf` displayed pixels
5. per-energy scatter injection (Ohnesorge spatial × NIST per-bin
   weights)
6. Poisson noise on total counts (after scatter, before logging)

The returned tuple gives us:
- `pcct_sino.bins` — 4 GPU sinograms, one per bin
- `I0_bins`        — air reference per bin (used for HU normalization
  and scatter correction)
- `scatter_field`  — known spatial scatter field (CPU)
- `scatter_bin_weights` — per-bin scatter fraction

Scatter is **injected** by `simulate!` so that Poisson statistics are
computed on the realistic total-count signal, but **correction** is
decoupled to the notebook level — we get the exact known model from the
returned struct, no blind re-estimation needed.
"""

# ╔═╡ 04000008-0000-4000-8000-000000000001
md"""
## 6. Per-bin scatter correction (decoupled, model-based)

`simulate!` returns the exact spatial scatter field and per-bin scatter
fractions it injected.  Subtracting them out is a known-model operation
— no blind re-estimation needed for simulated data.

Per bin `b`:
```
N_measured = I0[b] · exp(-p[b])
N_scatter  = scatter_field · I0_total · scatter_bin_weights[b]
N_corrected = max(N_measured − N_scatter, 0)
p_corrected = -log(N_corrected / I0[b])
```

This is the same procedure used in nb07 — exact, fast, no extra
sinogram passes.
"""

# ╔═╡ 04000009-0000-4000-8000-000000000001
md"""
## 7. Bin combine — 4 bins → low / high pair

Combine the 4 PCCT threshold bins into 2 effective sinograms via
**I0-weighted Beer recombination** (McCollough 2015, Yu 2012):

* **low**  = bins **1 + 2** (20 – 55 keV)
* **high** = bins **3 + 4** (>55 keV)

```
N_grp = Σ_{b∈grp} I0[b] · exp(-p[b])
p_grp = -log( N_grp / Σ_{b∈grp} I0[b] )
```

This is the canonical input for image-domain DECT decomposition: each
combined sinogram represents a polychromatic measurement at the
I0-weighted average spectrum of its bin group.

!!! note "§5 + §6 + §7 are bundled below"
    Pluto retains every named cell value, so keeping `sim` (4 raw bins),
    `sim_bins_corrected` (4 scatter-corrected bins), and `sim_lohi` (2
    combined sinos) as separate cells held ~12 GB of sinograms alive
    simultaneously — OOM territory on 16 GB Macs.  The bundled cell
    below applies scatter correction in-place on the raw bins, combines
    them, then drops the 4 bins before returning.  Only the (low, high,
    geom) triple survives → ~250 MB resident.
"""

# ╔═╡ 04000009-0000-4000-8000-000000000010
sim_lohi = let
    @info "Simulating: 140 kVp / $(round(protocol.mA, digits = 1)) mA (PCCT 4-bin)…"
    ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    result = BS.simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)

    geom    = ws.geom
    bins    = [Array(b) for b in result.pcct_sino.bins]   # Float32 already
    I0_bins = copy(result.I0_bins)

    ws = nothing; result = nothing
    GC.gc(true)

    # ─── §6. Scatter correction — mirrors nb07 `sim_scan2_bins_corrected` ────
    # 1. Recombine bins into combined primary sinogram.
    # 2. Estimate scatter field via Ohnesorge model from combined.
    # 3. Per-bin scatter fractions from spectrum × η × DRM (same model `simulate!` uses).
    # 4. Subtract per-bin scatter (exact inverse of injection in driver.jl).
    #
    # We do NOT use `result.scatter_field` here — the clinical pipeline must
    # re-estimate scatter from the recombined primary, not lift it from the
    # simulator's known-injection state.  Keeping nb04 ↔ nb07 1:1.
    let
        I0_total = Float32(sum(I0_bins))
        eps_f = Float32(1e-10)

        # Step 1: combined primary
        combined = zeros(Float32, size(bins[1]))
        for (b, bin_sino) in enumerate(bins)
            I0b = Float32(I0_bins[b])
            @. combined += I0b * exp(-bin_sino)
        end
        @. combined = -log(max(combined, eps_f) / I0_total)

        # Step 2: Ohnesorge scatter field
        voxel_size_mm = phantom_cpu.voxel_size .* 10.0
        phantom_diam_cm = BS.estimate_phantom_diameter_cm(phantom_cpu.mask, voxel_size_mm)
        scatter_model = BS.geometry_aware_scatter_model(scanner; phantom_diameter_cm = phantom_diam_cm)
        scatter_field = similar(combined)
        BS.estimate_scatter_field!(scatter_field, combined, scatter_model)
        @info "[scatter] field mean=$(round(mean(scatter_field), sigdigits = 3))  max=$(round(maximum(scatter_field), sigdigits = 3))"

        # Step 3: per-bin scatter fractions (spectrum × η × DRM)
        e_full, w_full = BS.resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner = scanner)
        pcct_det = BS._build_pcct_detector(scanner)
        kVp_val  = Float64(maximum(e_full))
        R_mat    = BS.compute_mc_drm(pcct_det, kVp_val)
        η_vec    = BS.quantum_efficiency_vector(pcct_det.material, pcct_det.thickness_mm, e_full)
        ew       = BS.compute_scatter_energy_weights(Float64.(e_full))
        scatter_fracs = BS.compute_scatter_bin_weights(
            Float64.(e_full), Float64.(w_full), ew, Float64.(η_vec), R_mat, kVp_val)
        @info "[scatter] bin fractions: $(round.(scatter_fracs, digits = 3))"

        # Step 4: in-place per-bin subtraction
        for (b, bin_sino) in enumerate(bins)
            I0b  = Float32(I0_bins[b])
            frac = Float32(scatter_fracs[b])
            for idx in eachindex(bin_sino)
                N_measured  = I0b * exp(-bin_sino[idx])
                N_scatter   = scatter_field[idx] * I0_total * frac
                N_corrected = N_measured - max(N_scatter, Float32(0))
                bin_sino[idx] = -log(max(N_corrected, eps_f) / I0b)
            end
        end

        combined = nothing
        scatter_field = nothing
    end
    GC.gc(true)

    # ─── §7. Bin combine — I0-weighted Beer recombination ───
    function _combine(bin_indices)
        I0_sum = Float32(sum(Float64.(I0_bins[bin_indices])))
        N = zeros(Float32, size(bins[1]))
        for b in bin_indices
            I0b = Float32(I0_bins[b])
            @. N += I0b * exp(-bins[b])
        end
        @. -log(max(N, Float32(1e-10)) / I0_sum)
    end

    sino_low  = _combine([1, 2])
    sino_high = _combine([3, 4])

    @info "[bin combine]  low  bins=[1, 2]:  mean p=$(round(mean(sino_low),  digits = 3))"
    @info "[bin combine]  high bins=[3, 4]:  mean p=$(round(mean(sino_high), digits = 3))"

    I0_out = copy(I0_bins)   # tiny (4 floats) — needed by the M-matrix μ_water cell
    bins = nothing
    I0_bins = nothing
    GC.gc(true)

    (sino_low = sino_low, sino_high = sino_high, geom = geom, I0_bins = I0_out)
end;

# ╔═╡ 0400000a-0000-4000-8000-000000000001
md"""
## 8. FBP per combined bin → μ-domain volumes

Each combined sinogram is FBP-reconstructed independently into its own
μ-volume (cm⁻¹).  HU conversion is **deferred** to after RSKR (§10) so
the joint denoiser operates on the raw μ pair — that's where the
iodine/water noise is maximally anti-correlated, which RSKR exploits.

Same Br36-ish (sharper) apodization used by the clinical verification
notebook for VMI bins.
"""

# ╔═╡ 0400000a-0000-4000-8000-000000000010
de_lohi_μ = let
    matrix_size = recon_opts.matrix_size
    geom = sim_lohi.geom

    fdk_filter = BS.CustomFilter(
        (0.0, 0.25, 0.5,  0.75, 1.0),
        (1.0, 0.75, 0.6,  0.2,  0.001),
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

    vol_low_μ  = _fbp_to_μ(sim_lohi.sino_low)
    vol_high_μ = _fbp_to_μ(sim_lohi.sino_high)

    (vol_low_μ = vol_low_μ, vol_high_μ = vol_high_μ, geom = geom)
end;

# ╔═╡ 0400000b-0000-4000-8000-000000000001
md"""
## 10. RSKR-2ch joint denoising (μ-domain)

[`BS.apply_rskr`](@ref) runs a **joint SVD + bilateral filter** on the
`(vol_low_μ, vol_high_μ)` pair (Clark & Badea 2023).  Operating on the
μ-pair *before* HU conversion preserves the anti-correlated noise
structure between iodine-rich (high in low-bin HU) and iodine-poor
(high in high-bin HU) regions.

!!! info "Why μ-domain, not HU-domain"
    HU conversion is a per-bin linear scaling, so it shifts each
    volume's statistics independently and breaks the joint covariance
    that RSKR exploits.  Always denoise *before* HU conversion when
    both volumes share a noise basis.

Hyperparameters match nb03's Gammex DE settings — `n_iter = 2`, `h_param = 2.0`, `radius = 2` — light-touch denoising that knocks down the
basis-pair noise without over-smoothing the iodine contrast.
"""

# ╔═╡ 0400000b-0000-4000-8000-000000000010
de_lohi_rskr = let
    out = BS.apply_rskr(
        [de_lohi_μ.vol_low_μ, de_lohi_μ.vol_high_μ];
        n_iter = 2,
        h_param = 2.0,
        radius = 2,
        γ = 0.5,
        gpu_arr_type = GPU_BACKEND.to_gpu,
        verbose = true,
    )
    (vol_low_μ = out[1], vol_high_μ = out[2], geom = de_lohi_μ.geom)
end;

# ╔═╡ 04000006-0000-4000-8000-000000000001
md"""
## 9. Per-bin `μ_water` — three options + a selector

Three independent estimators for the per-bin water lac.  We compute
all three and let `μ_water_source` pick which one feeds the HU
conversion.  This makes it easy to A/B test how each choice ripples
into the verification regression.

| Option | What it does | Pros | Cons |
|---|---|---|---|
| `:analytic` | Bowtie-aware spectrum × QE × DRM[:, b], phantom-hardened through 33 cm of water | Clean physics, no recon noise | Doesn't see RSKR / cupping / iodine-rod neighborhood effects |
| `:measured` | Mean of 8-px ROI at phantom center on post-RSKR `vol_*_μ` | Anchors water HU = 0 by construction at that voxel | Captures cupping (center is most-hardened) → rods at outer ring read negative |
| `:manual` | User-set `Float64` literal | Lets us dial in a value where the iodine rod HUs land where we expect | Trial-and-error |

After the three values are computed, a comparison table prints all
three so you can see which one is closest to a sensible iodine HU
landing.
"""

# ╔═╡ 04000006-0000-4000-8000-000000000005
function _per_bin_μ_water_analytic(
        bin_group::Vector{Int};
        water_path_cm::Real,
    )
    e, ŵ = BS.resolve_source_spectrum_with_bowtie(
        sim_opts, protocol;
        scanner = scanner, geom = sim_lohi.geom,
        include_bowtie = true,
        label = "bin grp $(bin_group)",
    )
    # Center-ray entrance spectrum — matches `compute_polychromatic_μ_water`'s
    # mid-column convention.  `ŵ` is 3D when bowtie is present, 1D otherwise.
    w_entr = if ndims(ŵ) == 3
        n_col = size(ŵ, 1); n_row = size(ŵ, 2)
        mid_c = n_col ÷ 2 + 1; mid_r = n_row ÷ 2 + 1
        Float64[Float64(ŵ[mid_c, mid_r, k]) for k in eachindex(e)]
    else
        Float64.(ŵ)
    end

    # Per-bin detector response (QE × DRM[:, b]) weighting.
    detector = BS._build_pcct_detector(scanner)
    kVp_val = Float64(maximum(e))
    R_mat = BS.compute_mc_drm(detector, kVp_val)
    η_vec = BS.quantum_efficiency_vector(detector.material, detector.thickness_mm, e)
    n_R = size(R_mat, 1)
    drm_row(E) = clamp(round(Int, (Float64(E) - 1.0) / (kVp_val - 1.0) * (n_R - 1)) + 1, 1, n_R)

    w_bin = zeros(Float64, length(e))
    for b in bin_group, i in eachindex(e)
        w_bin[i] += w_entr[i] * Float64(η_vec[i]) * Float64(R_mat[drm_row(e[i]), b])
    end
    w_bin ./= sum(w_bin)

    # Phantom hardening through `water_path_cm` of water + integrate.
    μ_per_E = Float64[BS.compute_μ_at_energy(BS.XA.Materials.water, Float64(eᵢ)) for eᵢ in e]
    w_hard = w_bin .* exp.(-μ_per_E .* Float64(water_path_cm))
    return sum(w_hard .* μ_per_E) / sum(w_hard)
end

# ╔═╡ 04000006-0000-4000-8000-000000000010
# OPTION 1 — ANALYTIC per-bin μ_water (bin-effective spectrum + 33 cm phantom hardening).
μ_water_per_bin_analytic = let
    body_radius_cm = 16.5      # Gammex 472 body — 33 cm diameter / 2
    water_path = body_radius_cm * 2
    (
        low  = _per_bin_μ_water_analytic([1, 2]; water_path_cm = water_path),
        high = _per_bin_μ_water_analytic([3, 4]; water_path_cm = water_path),
    )
end;

# ╔═╡ 04000006-0000-4000-8000-000000000012
# OPTION 2 — MEASURED μ_water from post-RSKR FBP (8-px ROI at phantom center, mean over slices).
μ_water_per_bin_measured = let
    nx = size(de_lohi_rskr.vol_low_μ, 1)
    ny = size(de_lohi_rskr.vol_low_μ, 2)
    nz = size(de_lohi_rskr.vol_low_μ, 3)
    cx = nx / 2 + 0.5
    cy = ny / 2 + 0.5
    ROI_R = 8

    roi = CartesianIndex{2}[]
    r² = Float64(ROI_R)^2
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
        s / n
    end

    (low = _mean_μ(de_lohi_rskr.vol_low_μ),
     high = _mean_μ(de_lohi_rskr.vol_high_μ))
end;

# ╔═╡ 04000006-0000-4000-8000-000000000014
# OPTION 4 — M-MATRIX per-bin μ_water with phantom hardening.
# Hybrid of nb07 `sim_scan2_M_matrix_s2` (per-bin spectrum × η × DRM, I0-weighted
# across the bin group) and nb04's `:analytic` phantom-hardening, but with the
# water path taken from the **actual phantom diameter** (`estimate_phantom_diameter_cm`
# on the mask — same call the scatter correction uses) rather than a hardcoded
# 33 cm body cylinder.
μ_water_per_bin_m_matrix = let
    e_full, w_full = BS.resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner = scanner)
    pcct_det = BS._build_pcct_detector(scanner)
    kVp_val  = Float64(maximum(e_full))
    R_mat    = BS.compute_mc_drm(pcct_det, kVp_val)
    η_vec    = BS.quantum_efficiency_vector(pcct_det.material, pcct_det.thickness_mm, e_full)
    n_R      = size(R_mat, 1)
    drm_row(E) = clamp(round(Int, (Float64(E) - 1.0) / (kVp_val - 1.0) * (n_R - 1)) + 1, 1, n_R)

    e    = Float64.(e_full)
    μρ_w = Float64[BS.compute_mass_μ_at_energy(BS.XA.Materials.water, Eᵢ) for Eᵢ in e]
    μ_w  = Float64[BS.compute_μ_at_energy(BS.XA.Materials.water,      Eᵢ) for Eᵢ in e]
    I0   = sim_lohi.I0_bins

    # Phantom-hardening path = full chord through the actual phantom (≈ diameter
    # for a center ray on a body cylinder).  Mirrors the scatter correction's
    # phantom-diameter estimation so both stay consistent.
    voxel_size_mm   = phantom_cpu.voxel_size .* 10.0
    phantom_diam_cm = BS.estimate_phantom_diameter_cm(phantom_cpu.mask, voxel_size_mm)
    water_path_cm   = phantom_diam_cm

    function _eff_spectrum(grp::Vector{Int})
        I0_sum = sum(Float64.(I0[grp]))
        wc = zeros(Float64, length(e))
        for b in grp
            wb = [Float64(w_full[i]) * Float64(η_vec[i]) * Float64(R_mat[drm_row(e[i]), b]) for i in eachindex(e)]
            wb .*= exp.(-μ_w .* water_path_cm)         # phantom hardening per energy
            sb = sum(wb)
            sb > 0 || error("_eff_spectrum: bin $b has zero spectral weight")
            wc .+= (Float64(I0[b]) / I0_sum) .* (wb ./ sb)
        end
        wc ./= sum(wc)
        wc
    end

    w_low  = _eff_spectrum([1, 2])
    w_high = _eff_spectrum([3, 4])

    @info "[μ_water m_matrix]  phantom_diam=$(round(phantom_diam_cm, digits = 1)) cm  (used as water-path for hardening)"
    @info "  low  = $(round(sum(w_low  .* μρ_w), sigdigits = 4)) cm⁻¹   (mean E = $(round(sum(w_low  .* e), digits = 1)) keV)"
    @info "  high = $(round(sum(w_high .* μρ_w), sigdigits = 4)) cm⁻¹   (mean E = $(round(sum(w_high .* e), digits = 1)) keV)"

    (low  = sum(w_low  .* μρ_w),
     high = sum(w_high .* μρ_w))
end;

# ╔═╡ 04000006-0000-4000-8000-000000000016
# SELECTOR — pick which of the four feeds HU conversion.
# Set to one of: `:analytic`, `:measured`, `:manual`, `:m_matrix`.
# `:m_matrix` is the canonical nb07 reference — use it unless you need to A/B.
μ_water_source = :m_matrix;

# ╔═╡ 04000006-0000-4000-8000-000000000018
μ_water_per_bin = let
    src = μ_water_source
    if src === :analytic
        μ_water_per_bin_analytic
    elseif src === :measured
        μ_water_per_bin_measured
    elseif src === :manual
        μ_water_per_bin_manual
    elseif src === :m_matrix
        μ_water_per_bin_m_matrix
    else
        error("μ_water_source must be :analytic, :measured, :manual, or :m_matrix (got $(src))")
    end
end;

# ╔═╡ 04000006-0000-4000-8000-000000000020
md"""
### μ_water comparison

| source | low (cm⁻¹) | high (cm⁻¹) |
|---|---|---|
| analytic (bin-spec + 33 cm hardening)              | $(round(μ_water_per_bin_analytic.low, digits = 5)) | $(round(μ_water_per_bin_analytic.high, digits = 5)) |
| measured (FBP center, post-RSKR)                   | $(round(μ_water_per_bin_measured.low, digits = 5)) | $(round(μ_water_per_bin_measured.high, digits = 5)) |
| manual                                             | $(round(μ_water_per_bin_manual.low,   digits = 5)) | $(round(μ_water_per_bin_manual.high,   digits = 5)) |
| m_matrix (bin-effective + measured-phantom hardening) | $(round(μ_water_per_bin_m_matrix.low, digits = 5)) | $(round(μ_water_per_bin_m_matrix.high, digits = 5)) |
| **active** (:$(μ_water_source))                    | **$(round(μ_water_per_bin.low,  digits = 5))** | **$(round(μ_water_per_bin.high, digits = 5))** |
"""

# ╔═╡ 0400000c-0000-4000-8000-000000000001
md"""
## 11. HU conversion + system noise floor

Per-bin `to_hounsfield` against the **per-bin** `μ_water_per_bin` from §9 —
each bin uses its own bin-effective μ_water reference so water reads ≈ 0
HU at *both* low and high before any self-cal absorbs residuals.
Applies the dose-independent DAS Gaussian noise floor afterwards.
"""

# ╔═╡ 0400000c-0000-4000-8000-000000000010
de_lohi_HU = let
    vol_low_HU  = Float32.(BS.to_hounsfield(de_lohi_rskr.vol_low_μ;  μ_water = μ_water_per_bin.low))
    vol_high_HU = Float32.(BS.to_hounsfield(de_lohi_rskr.vol_high_μ; μ_water = μ_water_per_bin.high))
    BS.add_system_noise_floor!(vol_low_HU,  15.0; seed = 1234)
    BS.add_system_noise_floor!(vol_high_HU, 15.0; seed = 5678)
    (vol_low_HU = vol_low_HU, vol_high_HU = vol_high_HU, geom = de_lohi_rskr.geom)
end;

# ╔═╡ 0400000d-0000-4000-8000-000000000001
md"""
## 12. Self-calibration — measure rod HUs, fit Ding coefficients

The Ding image-domain decomposition expresses iodine concentration as
a linear function of the two HU volumes:

```
c_iodine[v] = a₀ + a₁ · HU_low[v] + a₂ · HU_high[v]    [mg/mL]
```

`(a₀, a₁, a₂)` are LSQ-fit from a calibration table of known rods
(water + 7 iodine concentrations).  We **self-calibrate** from this
notebook's own simulated post-RSKR rod HUs — same trick as nb03.  This
matches the simulator's polychromatic HU baseline by construction, so
the per-energy regression slopes (§17) collapse to ≈ 1 across both Ca
and I rods.

Why measure here instead of loading from
`BS.SIEMENS_NAEOTOM_ALPHA_140KVP_CAL`?  That constant was populated
with a per-bin μ_water (one for low, one for high), but this notebook
uses a single full-spectrum analytic μ_water for both bins (matching
nb03's flow).  The HU baselines differ so we re-fit.
"""

# ╔═╡ 0400000d-0000-4000-8000-000000000010
ROD_LABELS = (
    Ca = (UInt8(10), UInt8(11), UInt8(12), UInt8(13), UInt8(14), UInt8(15), UInt8(16)),
    I = (UInt8(20), UInt8(21), UInt8(22), UInt8(23), UInt8(24), UInt8(25), UInt8(26)),
);

# ╔═╡ 0400000d-0000-4000-8000-000000000020
ROD_NAMES = (
    Ca = ("50 mg/mL", "100 mg/mL", "200 mg/mL", "300 mg/mL", "400 mg/mL", "500 mg/mL", "600 mg/mL"),
    I = ("2.0 mg/mL", "2.5 mg/mL", "5.0 mg/mL", "7.5 mg/mL", "10.0 mg/mL", "15.0 mg/mL", "20.0 mg/mL"),
);

# ╔═╡ 0400000d-0000-4000-8000-000000000030
ROD_C_IODINE = (
    Ca = (50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0),
    I = (2.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0),
);

# ╔═╡ 0400000d-0000-4000-8000-000000000040
md"""
The ROI is an **8-pixel-radius circle at each rod's centroid** —
the same trick used by nb03.  Full-mask averages mix in partial-volume
edge voxels that Mono+'s per-energy LP filter blurs differently at each
keV; a small core ROI dodges that.
"""

# ╔═╡ 0400000d-0000-4000-8000-000000000050
rod_rois = let
    mask_2d = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
    ROI_RADIUS_PX = 8

    function rod_centroid(label::UInt8)
        idx = findall(==(label), mask_2d)
        isempty(idx) && error("rod_centroid: no voxels with label $label in mask_2d")
        cx = sum(ci -> Float64(ci[1]), idx) / length(idx)
        cy = sum(ci -> Float64(ci[2]), idx) / length(idx)
        return (cx, cy)
    end

    function rod_roi_mask(label::UInt8)
        cx, cy = rod_centroid(label)
        nx, ny = size(mask_2d)
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

    Dict(
        lab => rod_roi_mask(lab)
            for lab in vcat(collect(ROD_LABELS.Ca), collect(ROD_LABELS.I))
    )
end;

# ╔═╡ 0400000d-0000-4000-8000-000000000060
function _measure_rod_hu(vol::AbstractArray, label::UInt8, roi_dict)
    roi = roi_dict[label]
    n_z = size(vol, 3)
    s = 0.0
    n = 0
    for z in 1:n_z, ci in roi
        s += vol[ci, z]
        n += 1
    end
    return s / n
end

# ╔═╡ 0400000d-0000-4000-8000-000000000070
de_cal = let
    iod_labels = collect(ROD_LABELS.I)
    iod_concs  = collect(ROD_C_IODINE.I)

    # Water anchor: 8-px circular ROI at the phantom geometric center (the
    # Gammex 472 body's solid-water region, label 3, fills the body cylinder
    # with the Ca/I rods inset on inner / outer rings).  The center is well
    # away from any rod so a center ROI samples pure solid water.
    nx = size(de_lohi_HU.vol_low_HU, 1)
    ny = size(de_lohi_HU.vol_low_HU, 2)
    cx_water = nx / 2 + 0.5
    cy_water = ny / 2 + 0.5
    ROI_RADIUS_PX = 8
    water_roi = let
        roi = CartesianIndex{2}[]
        i_lo = max(1, floor(Int, cx_water - ROI_RADIUS_PX))
        i_hi = min(nx, ceil(Int, cx_water + ROI_RADIUS_PX))
        j_lo = max(1, floor(Int, cy_water - ROI_RADIUS_PX))
        j_hi = min(ny, ceil(Int, cy_water + ROI_RADIUS_PX))
        r² = Float64(ROI_RADIUS_PX)^2
        for j in j_lo:j_hi, i in i_lo:i_hi
            ((i - cx_water)^2 + (j - cy_water)^2) ≤ r² && push!(roi, CartesianIndex(i, j))
        end
        roi
    end

    function _mean_at_roi(vol, roi)
        n_z = size(vol, 3)
        s = 0.0; n = 0
        for z in 1:n_z, ci in roi
            s += vol[ci, z]; n += 1
        end
        s / n
    end

    rod_HU_low_water  = _mean_at_roi(de_lohi_HU.vol_low_HU,  water_roi)
    rod_HU_high_water = _mean_at_roi(de_lohi_HU.vol_high_HU, water_roi)

    HU_low_cal  = vcat(rod_HU_low_water,  [_measure_rod_hu(de_lohi_HU.vol_low_HU,  lab, rod_rois) for lab in iod_labels])
    HU_high_cal = vcat(rod_HU_high_water, [_measure_rod_hu(de_lohi_HU.vol_high_HU, lab, rod_rois) for lab in iod_labels])
    c_iod_cal   = vcat(0.0,               iod_concs)

    fit = BS.fit_ding_coeffs(HU_low_cal, HU_high_cal, c_iod_cal)
    rod_names = vcat("Water (center)", collect(ROD_NAMES.I))

    (
        coeffs = fit.coeffs,
        α_iod_low = fit.α_low,
        α_iod_high = fit.α_high,
        rms_c = fit.rms,
        rod_names = rod_names,
        rod_HU_low = HU_low_cal,
        rod_HU_high = HU_high_cal,
        rod_c_iodine = c_iod_cal,
        pred_c = fit.pred_c,
        water_roi = water_roi,
    )
end;

# ╔═╡ 0400000d-0000-4000-8000-000000000080
let
    # Look up water mask voxel count to add a sanity hint to the table
    water_label = UInt8(1)

    rows = [
        "| $(de_cal.rod_names[i]) | $(de_cal.rod_c_iodine[i]) | " *
        "$(round(de_cal.rod_HU_low[i],  digits = 1)) | " *
        "$(round(de_cal.rod_HU_high[i], digits = 1)) | " *
        "$(round(de_cal.pred_c[i],       digits = 2)) |"
            for i in eachindex(de_cal.rod_names)
    ]

    body = """
    **Self-cal rod ROIs** — 8-px core circular ROI at each rod's
    centroid, broadcast across all $(size(de_lohi_HU.vol_low_HU, 3)) recon slices:

    | Rod | c (mg/mL) | HU low (1+2) | HU high (3+4) | pred c |
    |-----|---|---|---|---|
    $(join(rows, "\n"))

    **Ding fit:**

    * a₀ = $(round(de_cal.coeffs[1], digits = 3))
    * a₁ = $(round(de_cal.coeffs[2], sigdigits = 4))
    * a₂ = $(round(de_cal.coeffs[3], sigdigits = 4))
    * α_iodine (low / high) = $(round(de_cal.α_iod_low, digits = 2)) / $(round(de_cal.α_iod_high, digits = 2)) HU per (mg/mL)
    * RMS calibration error = $(round(de_cal.rms_c, digits = 3)) mg/mL
    """
    Markdown.parse(body)
end

# ╔═╡ 0400000e-0000-4000-8000-000000000001
md"""
## 13. Apply Ding decomposition + z-median speckle removal

[`BS.apply_ding_decomp`](@ref) is a per-voxel evaluation of the Ding
equation, returning a `c_iodine` map in mg/mL.  A 3-slice
[`BS.apply_median_z`](@ref) (radius = 1, axial-only) wipes single-voxel
xy impulse noise without any in-plane blurring.
"""

# ╔═╡ 0400000e-0000-4000-8000-000000000010
de_decomp = let
    c_iodine = BS.apply_ding_decomp(
        de_lohi_HU.vol_low_HU, de_lohi_HU.vol_high_HU, de_cal.coeffs
    )
    c_iodine = BS.apply_median_z(c_iodine; radius = 1)
    (
        c_iodine = c_iodine,
        vol_low_HU = de_lohi_HU.vol_low_HU,
        vol_high_HU = de_lohi_HU.vol_high_HU,
        geom = de_lohi_HU.geom,
    )
end;

# ╔═╡ 0400000f-0000-4000-8000-000000000001
md"""
## 14. VMI synthesis at 40 / 70 / 100 / 140 keV

[`BS.synth_vmi_image_domain`](@ref) implements

```
HU_E[v] = HU_low[v] + c_iodine[v] · (αᴱ_phys − α_low_cal)
```

where `αᴱ_phys = (μ/ρ)_iodine(E) / (μ/ρ)_water(E)` is the pure-physics
HU sensitivity at the target energy (computed internally from
XrayAttenuation), and `α_low_cal` is the empirical low-bin sensitivity
returned by `fit_ding_coeffs`.

| keV | What it shows                                            |
|-----|----------------------------------------------------------|
| 40  | Maximum iodine contrast (≈ K-edge boost)                 |
| 70  | ≈ standard 120 kVp HU baseline                           |
| 100 | Beam-hardening robust, low metal artifact                |
| 140 | Quasi-monochromatic high-energy reference                |
"""

# ╔═╡ 0400000f-0000-4000-8000-000000000010
de_vmi_energies = [40.0, 70.0, 100.0, 140.0];

# ╔═╡ 0400000f-0000-4000-8000-000000000020
de_vmi_raw = let
    out = Dict{Float64, Array{Float32, 3}}()
    for E in de_vmi_energies
        out[E] = BS.synth_vmi_image_domain(
            de_decomp.vol_low_HU, de_decomp.c_iodine;
            energy_keV = E,
            α_iod_low_cal = de_cal.α_iod_low,
        )
    end
    out
end;

# ╔═╡ 04000010-0000-4000-8000-000000000001
md"""
## 15. Mono+ post-processing + FOV mask

[`BS.apply_mono_plus!`](@ref) sharpens the contrast at the noise-quiet
anchor energy (70 keV here) and slightly low-passes the others.  Per-energy
σ in pixels (matches nb07's Scan 2 settings):

| keV | σ_lp (px) | What happens                                     |
|-----|-----------|--------------------------------------------------|
| 40  | 2.0       | Noisiest VMI — strongest LP                      |
| 70  | 0.0       | Anchor energy — left untouched                   |
| 100 | 2.0       | Modest LP                                        |
| 140 | 2.0       | Modest LP                                        |

After Mono+, [`BS.apply_fov_mask!`](@ref) zeros out-of-FOV voxels to
−1024 HU (air) so downstream display, ROI stats, and tables don't see
recon-circle ringing.
"""

# ╔═╡ 04000010-0000-4000-8000-000000000010
de_vmi_mono = let
    vols_in = [de_vmi_raw[E] for E in de_vmi_energies]
    σ_vec = Float64[2.0, 0.0, 2.0, 2.0]   # paired with de_vmi_energies

    ws = BS.create_mono_plus_workspace(vols_in[1]; n_energies = length(de_vmi_energies))
    res = BS.apply_mono_plus!(
        ws, vols_in, de_vmi_energies;
        E_noise_opt = 70.0,
        σ_lp_px = σ_vec,
        verbose = true,
    )

    out = Dict{Float64, Array{Float32, 3}}()
    for (i, E) in enumerate(de_vmi_energies)
        v = copy(res.volumes[i])
        BS.apply_fov_mask!(v, de_decomp.geom; sentinel_μ = -1024.0f0)
        out[E] = v
    end

    ws = nothing
    GC.gc(true)
    out
end;

# ╔═╡ 04000011-0000-4000-8000-000000000001
md"""
## 16. The four VMI images

Mid-slice mosaic of all four energies, shared HU window (-200 to 500
HU) so iodine boost vs. soft-tissue baseline is comparable
side-by-side.
"""

# ╔═╡ 04000011-0000-4000-8000-000000000010
let
    fig = CM.Figure(size = (1100, 1080))

    mid = size(de_vmi_mono[de_vmi_energies[1]], 3) ÷ 2
    colorrng = (-200, 500)

    grid_pos = [(1, 1), (1, 2), (2, 1), (2, 2)]
    hms = nothing
    for (i, E) in enumerate(de_vmi_energies)
        r, c = grid_pos[i]
        ax = CM.Axis(
            fig[r, c];
            title = "VMI $(Int(E)) keV",
            subtitle = i == 1 ? "Mono+ post-processed · 140 kVp PCCT · Naeotom Alpha" : "Mono+ post-processed",
            aspect = CM.DataAspect(),
        )
        hm = CM.heatmap!(
            ax, de_vmi_mono[E][:, :, mid];
            colormap = :grays, colorrange = colorrng,
        )
        CM.hidedecorations!(ax)
        if i == 4
            hms = hm
        end
    end

    CM.Colorbar(fig[1:2, 3], hms; label = "HU", width = 14)

    CM.save(
        joinpath(@__DIR__, "..", "assets", "pcct_vmi_4panel.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 04000012-0000-4000-8000-000000000001
md"""
## 17. Per-rod readout: measured HU vs theoretical HU

Same flow as nb03.  For each rod *r* and each VMI energy *E*:

* **Measured HU** = mean of the **8-px core circular ROI** (same ROIs
  used for self-calibration in §12) broadcast across every recon slice.
* **Theoretical HU** = `1000 · (μ_r(E) − μ_water(E)) / μ_water(E)` from
  XrayAttenuation directly via [`BS.compute_μ_at_energy`](@ref).
"""

# ╔═╡ 04000012-0000-4000-8000-000000000010
rod_data = let
    materials = phantom_cpu.materials

    μ_water_E = Dict(
        E => BS.compute_μ_at_energy(BS.XA.Materials.water, E)
            for E in de_vmi_energies
    )

    function theoretical_hu(material, E::Float64)
        μ = BS.compute_μ_at_energy(material, E)
        return 1000.0 * (μ - μ_water_E[E]) / μ_water_E[E]
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
                meas[i, j] = _measure_rod_hu(de_vmi_mono[E], lab, rod_rois)
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

# ╔═╡ 04000013-0000-4000-8000-000000000001
md"""
## 18. The verification plot — measured vs theoretical, per rod

**Solid line** = measured HU at each VMI energy.
**Dashed line** = theoretical HU from XrayAttenuation directly.

The Ca rods (left panel) follow a smooth roll-off as keV increases —
calcium attenuates predominantly through Compton scatter at clinical
energies, so its HU above water decreases monotonically.

The I rods (right panel) show the iodine **K-edge boost at 40 keV** —
iodine's K-edge sits at 33.2 keV, so a 40 keV VMI catches the
photoelectric edge and amplifies iodine HU dramatically (200–500 HU per
mg/mL) compared to the ≈70 keV plateau.  Above the K-edge the rolloff
is steep until 100+ keV where iodine looks soft-tissue-like.
"""

# ╔═╡ 04000013-0000-4000-8000-000000000010
let
    fig = CM.Figure(size = (1180, 580))

    cmap_ca = CM.cgrad(:Oranges, 7; categorical = true)
    cmap_i  = CM.cgrad(:GnBu,    7; categorical = true)

    panels = (
        (
            group = :Ca, title = "Calcium Rods",
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
            titlesize = 32,
            subtitlesize = 24,
            ylabelsize = 22,
            xlabelsize = 22,
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

    CM.save(
        joinpath(@__DIR__, "..", "assets", "pcct_vmi_vs_theoretical.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 04000014-0000-4000-8000-000000000001
md"""
## 19. Linear regression: measured vs theoretical

Same data, different cut.  Every (rod, energy) pair scatters as
**measured HU on y vs theoretical HU on x**, with a per-energy line
fit and the y = x identity overlaid.  Slope ≈ 1, intercept ≈ 0,
R² ≈ 1 → the pipeline recovers physics.
"""

# ╔═╡ 7b51f797-15cc-408d-a7a8-2fdb8df3f5ba
# # OPTION 3 — MANUAL μ_water override.  Edit these two numbers freely; this
# # cell is the only thing that needs reactive re-evaluation when you tweak.
# # Sensible starting values to iterate around:
# #   • spectrum-averaged 45 keV (low bin equiv): ~0.247 cm⁻¹
# #   • spectrum-averaged 85 keV (high bin equiv): ~0.180 cm⁻¹
# μ_water_per_bin_manual = (
#     low  = 0.035,
#     high = 0.115,
# );

# ╔═╡ 04000014-0000-4000-8000-000000000010
let
    fig = CM.Figure(size = (1180, 620))

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

    panels = ((:Ca, "Calcium Rods", "50–600 mg/mL",), (:I, "Iodine Rods", "2–20 mg/mL"))
    for (col, (group, title, subtitle)) in pairs(panels)
        d = rod_data[group]
        ax = CM.Axis(
            fig[1, col];
            title = title,
            subtitle = subtitle,
            xlabel = "Theoretical HU",
            ylabel = "Measured HU",
            aspect = CM.AxisAspect(1),
            titlesize = 32,
            subtitlesize = 24,
            ylabelsize = 22,
            xlabelsize = 22,
        )

        lim_lo = min(0.0, minimum(d.measured), minimum(d.theoretical))
        lim_hi = max(maximum(d.measured), maximum(d.theoretical)) * 1.05
        CM.lines!(
            ax, [lim_lo, lim_hi], [lim_lo, lim_hi];
            color = :black, linestyle = :dash, linewidth = 2,
            label = "Unity (y = x)"
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
        joinpath(@__DIR__, "..", "assets", "pcct_vmi_regression.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 04000015-0000-4000-8000-000000000001
md"""
## 20. Quantitative agreement table

Compact (measured / theoretical / Δ) for every (rod, energy) pair.
Inside the K-edge–insensitive plateau (70–140 keV) we expect single-digit
HU agreement; at 40 keV iodine the K-edge boost amplifies any residual
calibration / spectral mismatch, so a few tens of HU is normal.
"""

# ╔═╡ 04000015-0000-4000-8000-000000000010
let
    function fmt_row(name, m_row, t_row)
        diffs = m_row .- t_row
        cells = ["**$(name)**"]
        for j in eachindex(de_vmi_energies)
            push!(cells, "$(round(m_row[j], digits = 0)) / $(round(t_row[j], digits = 0)) ($(round(diffs[j], digits = 0)))")
        end
        return "| " * join(cells, " | ") * " |"
    end

    header_e = ["measured / theoretical (Δ) at $(Int(E)) keV" for E in de_vmi_energies]
    header = "| Rod | " * join(header_e, " | ") * " |"
    sep = "|" * "---|"^(length(de_vmi_energies) + 1)

    lines_ca = [
        fmt_row(rod_data[:Ca].names[i], rod_data[:Ca].measured[i, :], rod_data[:Ca].theoretical[i, :])
            for i in eachindex(rod_data[:Ca].names)
    ]
    lines_i = [
        fmt_row(rod_data[:I].names[i], rod_data[:I].measured[i, :], rod_data[:I].theoretical[i, :])
            for i in eachindex(rod_data[:I].names)
    ]

    md_str = """
    **Calcium rods (HU)**

    $(header)
    $(sep)
    $(join(lines_ca, "\n"))

    **Iodine rods (HU)**

    $(header)
    $(sep)
    $(join(lines_i, "\n"))
    """
    Markdown.parse(md_str)
end

# ╔═╡ 04000016-0000-4000-8000-000000000001
md"""
## 21. Calibration source-code block

Run this cell to print a `SIEMENS_NAEOTOM_ALPHA_140KVP_CAL`-shaped
constant from this notebook's measured rod HUs (post-RSKR low/high
bins).  Paste the printed block into
`src/reconstruction/vmi/clinical_calibrations.jl` to replace the
existing constant whenever the underlying flow changes (μ_water
formula, scatter model, RSKR hyperparams, etc.).

Compared to nb07's clinical extraction, this BS-side phantom only
models a single solid-water region for the body (no separate
`SW ref 1` / `SW ref 2` / `Water (I)` reference rods).  We emit the
single water-center anchor + the 7 Ca + 7 I rods that the factory
actually places.

The constant feeds [`BS.iodine_calibration_rods`](@ref) and
[`BS.calcium_calibration_rods`](@ref) downstream — keeping it in sync
with this pipeline is what lets future notebooks load the fit instead
of re-extracting it.
"""

# ╔═╡ 04000016-0000-4000-8000-000000000010
let
    rod_concentrations = Dict(
        "Water (O)" => (mat = :water,        mg_per_mL =   0.0, label = UInt8(1)),
        "SW ref 1"  => (mat = :solid_water,  mg_per_mL =   0.0, label = UInt8(2)),
        "SW ref 2"  => (mat = :solid_water,  mg_per_mL =   0.0, label = UInt8(3)),
        "Ca 50"     => (mat = :calcium,      mg_per_mL =  50.0, label = ROD_LABELS.Ca[1]),
        "Ca 100"    => (mat = :calcium,      mg_per_mL = 100.0, label = ROD_LABELS.Ca[2]),
        "Ca 200"    => (mat = :calcium,      mg_per_mL = 200.0, label = ROD_LABELS.Ca[3]),
        "Ca 300"    => (mat = :calcium,      mg_per_mL = 300.0, label = ROD_LABELS.Ca[4]),
        "Ca 400"    => (mat = :calcium,      mg_per_mL = 400.0, label = ROD_LABELS.Ca[5]),
        "Water (I)" => (mat = :water,        mg_per_mL =   0.0, label = UInt8(4)),
        "I 2.0"     => (mat = :iodine,       mg_per_mL =   2.0, label = ROD_LABELS.I[1]),
        "I 2.5"     => (mat = :iodine,       mg_per_mL =   2.5, label = ROD_LABELS.I[2]),
        "I 5.0"     => (mat = :iodine,       mg_per_mL =   5.0, label = ROD_LABELS.I[3]),
        "I 7.5"     => (mat = :iodine,       mg_per_mL =   7.5, label = ROD_LABELS.I[4]),
        "I 10.0"    => (mat = :iodine,       mg_per_mL =  10.0, label = ROD_LABELS.I[5]),
        "I 15.0"    => (mat = :iodine,       mg_per_mL =  15.0, label = ROD_LABELS.I[6]),
        "I 20.0"    => (mat = :iodine,       mg_per_mL =  20.0, label = ROD_LABELS.I[7]),
    )
    rod_order = [
        "Water (O)", "SW ref 1", "SW ref 2",
        "Ca 50", "Ca 100", "Ca 200", "Ca 300", "Ca 400",
        "Water (I)",
        "I 2.0", "I 2.5", "I 5.0", "I 7.5", "I 10.0", "I 15.0", "I 20.0",
    ]

    function _safe_mean(vol, lab)
        try
            return _measure_rod_hu(vol, lab, _build_label_roi(lab))
        catch
            return missing
        end
    end

    # Build per-label ROI lazily — fall back to the existing rod_rois Dict
    # when the label is in our pre-built set; otherwise return missing.
    function _build_label_roi(lab::UInt8)
        haskey(rod_rois, lab) && return rod_rois
        # Solid-water / extra water rods: build on the fly using the same
        # 8-px circular ROI logic.
        mask_2d = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
        idx = findall(==(lab), mask_2d)
        isempty(idx) && return Dict{UInt8, Vector{CartesianIndex{2}}}(lab => CartesianIndex{2}[])
        cx = sum(ci -> Float64(ci[1]), idx) / length(idx)
        cy = sum(ci -> Float64(ci[2]), idx) / length(idx)
        nx, ny = size(mask_2d)
        ROI_R = 8
        i_lo = max(1, floor(Int, cx - ROI_R)); i_hi = min(nx, ceil(Int, cx + ROI_R))
        j_lo = max(1, floor(Int, cy - ROI_R)); j_hi = min(ny, ceil(Int, cy + ROI_R))
        roi = CartesianIndex{2}[]
        r² = Float64(ROI_R)^2
        for j in j_lo:j_hi, i in i_lo:i_hi
            ((i - cx)^2 + (j - cy)^2) ≤ r² && push!(roi, CartesianIndex(i, j))
        end
        Dict{UInt8, Vector{CartesianIndex{2}}}(lab => roi)
    end

    println("# Auto-generated from notebook 04 §21 calibration cell.")
    println("# Siemens Naeotom Alpha — sim post-RSKR low/high-bin rod HUs (Gammex 472).")
    println("# Source: 140 kVp / $(round(protocol.mA, digits = 1)) mA / collimation $(protocol.collimation_mm) mm.")
    println()
    println("const SIEMENS_NAEOTOM_ALPHA_140KVP_CAL = Dict{String, NamedTuple}(")
    for nm in rod_order
        cc  = rod_concentrations[nm]
        h_low  = _safe_mean(de_lohi_HU.vol_low_HU,  cc.label)
        h_high = _safe_mean(de_lohi_HU.vol_high_HU, cc.label)
        if ismissing(h_low) || ismissing(h_high)
            println("    # \"$(nm)\"  — label $(cc.label) not present in mask, skipped")
            continue
        end
        println("    \"$(nm)\" => (material = :$(cc.mat),  mg_per_mL = $(cc.mg_per_mL),  HU_low_bin = $(round(h_low, digits = 1))f0,  HU_high_bin = $(round(h_high, digits = 1))f0),")
    end
    println(")")
end

# ╔═╡ 04000017-0000-4000-8000-000000000001
md"""
## Summary

This notebook walked the full **photon-counting CT image-domain VMI
pipeline** end to end on a simulated Gammex 472 + Siemens Naeotom Alpha
configuration:

- **`BS.simulate!` PCCT path** returns 4 per-bin sinograms + per-bin I0
  + the exact spatial scatter field + per-bin scatter weights.
  Scatter correction at the notebook level subtracts the known model
  exactly — no blind re-estimation.
- **Bin combine (low = 1+2, high = 3+4)** via I0-weighted Beer
  recombination — the canonical input for image-domain DECT decomp
  (McCollough 2015, Yu 2012).
- **Analytic μ_water with phantom hardening** ([`BS.compute_polychromatic_μ_water`](@ref))
  — same pattern as nb01 / nb02 / nb03, computed once per spectrum.
- **RSKR-2ch joint denoising in μ-domain** preserves anti-correlated
  iodine/water noise structure between basis volumes.
- **Self-calibrated Ding decomposition** — rod HUs measured directly on
  this notebook's post-RSKR volumes, fit with [`BS.fit_ding_coeffs`](@ref).
  The §21 calibration extraction cell prints a paste-ready src constant
  whenever the upstream flow changes.
- **Mono+ FBP-equivalent VMI sweep** at 40 / 70 / 100 / 140 keV with
  per-energy LP shaping anchored at 70 keV; FOV mask zeroes
  out-of-circle voxels to −1024 HU.
- **Per-rod verification against XrayAttenuation theory** — measured VMI
  HU vs `BS.compute_μ_at_energy` per rod material, with regression
  slopes that should collapse to ≈ 1 across both Ca and I rods.

This same image-domain pipeline can be repointed at any other PCCT
acquisition (different kVp, different scanner, different bin
configuration) by re-running the self-calibration cell against that
acquisition's rod HUs.
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
# ╟─04000007-0000-4000-8000-000000000001
# ╟─04000008-0000-4000-8000-000000000001
# ╟─04000009-0000-4000-8000-000000000001
# ╠═04000009-0000-4000-8000-000000000010
# ╟─0400000a-0000-4000-8000-000000000001
# ╠═0400000a-0000-4000-8000-000000000010
# ╟─0400000b-0000-4000-8000-000000000001
# ╠═0400000b-0000-4000-8000-000000000010
# ╟─04000006-0000-4000-8000-000000000001
# ╠═04000006-0000-4000-8000-000000000005
# ╠═04000006-0000-4000-8000-000000000010
# ╠═04000006-0000-4000-8000-000000000012
# ╠═04000006-0000-4000-8000-000000000014
# ╠═04000006-0000-4000-8000-000000000016
# ╠═04000006-0000-4000-8000-000000000018
# ╟─04000006-0000-4000-8000-000000000020
# ╟─0400000c-0000-4000-8000-000000000001
# ╠═0400000c-0000-4000-8000-000000000010
# ╟─0400000d-0000-4000-8000-000000000001
# ╠═0400000d-0000-4000-8000-000000000010
# ╠═0400000d-0000-4000-8000-000000000020
# ╠═0400000d-0000-4000-8000-000000000030
# ╟─0400000d-0000-4000-8000-000000000040
# ╠═0400000d-0000-4000-8000-000000000050
# ╠═0400000d-0000-4000-8000-000000000060
# ╠═0400000d-0000-4000-8000-000000000070
# ╟─0400000d-0000-4000-8000-000000000080
# ╟─0400000e-0000-4000-8000-000000000001
# ╠═0400000e-0000-4000-8000-000000000010
# ╟─0400000f-0000-4000-8000-000000000001
# ╠═0400000f-0000-4000-8000-000000000010
# ╠═0400000f-0000-4000-8000-000000000020
# ╟─04000010-0000-4000-8000-000000000001
# ╠═04000010-0000-4000-8000-000000000010
# ╟─04000011-0000-4000-8000-000000000001
# ╟─04000011-0000-4000-8000-000000000010
# ╟─04000012-0000-4000-8000-000000000001
# ╠═04000012-0000-4000-8000-000000000010
# ╟─04000013-0000-4000-8000-000000000001
# ╟─04000013-0000-4000-8000-000000000010
# ╟─04000014-0000-4000-8000-000000000001
# ╠═7b51f797-15cc-408d-a7a8-2fdb8df3f5ba
# ╟─04000014-0000-4000-8000-000000000010
# ╟─04000015-0000-4000-8000-000000000001
# ╟─04000015-0000-4000-8000-000000000010
# ╟─04000016-0000-4000-8000-000000000001
# ╟─04000016-0000-4000-8000-000000000010
# ╟─04000017-0000-4000-8000-000000000001
