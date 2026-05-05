### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

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

!!! warning "🚧 IN PROGRESS — known gaps below"
    This notebook is **work in progress**.  Known gaps versus the
    clinical-fidelity Siemens Naeotom Alpha pipeline:

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

# ╔═╡ 04000001-0000-4000-8000-000000000032
import Optim

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
        scan_diameter = 360.0,           # APPROX — clinical Naeotom is 500 mm; see warning above
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
# Naeotom Alpha runs at either 120 or 140 kVp.  Switch `kVp` here to
# re-derive calibration values for the other tube voltage.  mA is bumped
# proportionally for similar noise budget at 120 kVp (softer spectrum).
protocol = BS.CTProtocol(
    kVp = 140,
    mA = 174.0,                        # matches nb07 Scan 2 (clinical 140 kVp / 174 mA / 10.12 mGy CTDIvol)
    views = 1200,                      # matches nb07 Scan 2 — clinical Naeotom single-tube equivalent
    rotation_time = 0.5,               # matches nb07 Scan 2
    collimation_mm = 5.0,              # matches nb07 Scan 2 dev-mode (clinical 144×0.4mm ≈ 57.6mm)
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
        algorithm = :fdk,
        matrix_size = (512, 512, n_recon_slices),
        fov_cm = 35.0,
        z_cm = protocol.collimation_mm / 10.0,
        filter = :standard,
        vmi_basis = [:water, :iodine],
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
    @info "Simulating: $(Int(protocol.kVp)) kVp / $(round(protocol.mA, digits = 1)) mA (PCCT 4-bin)…"
    ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    result = BS.simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)

    geom = ws.geom
    bins = [Array(b) for b in result.pcct_sino.bins]   # Float32 already
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
        eps_f = Float32(1.0e-10)

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
        kVp_val = Float64(maximum(e_full))
        R_mat = BS.compute_mc_drm(pcct_det, kVp_val)
        η_vec = BS.quantum_efficiency_vector(pcct_det.material, pcct_det.thickness_mm, e_full)
        ew = BS.compute_scatter_energy_weights(Float64.(e_full))
        scatter_fracs = BS.compute_scatter_bin_weights(
            Float64.(e_full), Float64.(w_full), ew, Float64.(η_vec), R_mat, kVp_val
        )
        @info "[scatter] bin fractions: $(round.(scatter_fracs, digits = 3))"

        # Step 4: in-place per-bin subtraction
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
        return @. -log(max(N, Float32(1.0e-10)) / I0_sum)
    end

    sino_low = _combine([1, 2])
    sino_high = _combine([3, 4])

    @info "[bin combine]  low  bins=[1, 2]:  mean p=$(round(mean(sino_low), digits = 3))"
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
de_lohi_μ_raw = let
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

    vol_low_μ = _fbp_to_μ(sim_lohi.sino_low)
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

**Knob below** lets you sweep `h_param` (smoothing strength) without
touching the call site — useful for testing whether under- or
over-smoothing is biasing calcium VMI HU at high keV.  Larger `h_param`
≈ stronger denoise; `h_param = 0` ≈ pass-through.
"""

# ╔═╡ 0400000b-0000-4000-8000-000000000005
rskr_knob = (
    n_iter = 2,
    h_param = 2,
    radius = 3,
    γ = 0.5,
);

# ╔═╡ 0400000b-0000-4000-8000-000000000010
de_lohi_rskr = let
    out = BS.apply_rskr(
        [de_lohi_μ_raw.vol_low_μ, de_lohi_μ_raw.vol_high_μ];
        n_iter = rskr_knob.n_iter,
        h_param = rskr_knob.h_param,
        radius = rskr_knob.radius,
        γ = rskr_knob.γ,
        gpu_arr_type = GPU_BACKEND.to_gpu,
        verbose = true,
    )
    (vol_low_μ = out[1], vol_high_μ = out[2], geom = de_lohi_μ_raw.geom)
end;

# ╔═╡ 0400000b-0000-4000-8000-000000000020
md"""
### RSKR before/after — μ-domain inspection

2×2 grid: rows are **raw FBP** (top) vs **post-RSKR** (bottom), columns
are **low-bin** (1+2) vs **high-bin** (3+4).  Each column shares its
own window so the raw → RSKR change is the only visual delta.

Look for:
* **Calcium rods (inner ring)** flattening in the high bin — RSKR
  collapsing the calcium-direction signal would manifest here as
  reduced rod-vs-water contrast in the high panel.
* **Iodine rods (outer ring)** retaining contrast (iodine's
  anti-correlated noise is what RSKR is built to preserve).
"""

# ╔═╡ 0400000b-0000-4000-8000-000000000030
let
    mid = size(de_lohi_μ_raw.vol_low_μ, 3) ÷ 2

    raw_low = de_lohi_μ_raw.vol_low_μ[:, :, mid]
    raw_high = de_lohi_μ_raw.vol_high_μ[:, :, mid]
    rskr_low = de_lohi_rskr.vol_low_μ[:, :, mid]
    rskr_high = de_lohi_rskr.vol_high_μ[:, :, mid]

    # Fixed physical windows (cm⁻¹) so RSKR knob sweeps stay visually comparable.
    # Low bin (~50 keV eff): water ≈ 0.21, iodine 20 mg/mL ≈ 0.30, calcium 600 ≈ 0.55.
    # High bin (~80 keV eff): water ≈ 0.18, iodine 20 mg/mL ≈ 0.22, calcium 600 ≈ 0.27.
    win_low = (0.15, 0.3)
    win_high = (0.15, 0.3)

    title_kwargs = (titlesize = 28, subtitlesize = 20)

    fig = CM.Figure(size = (1200, 1100))
    CM.Label(
        fig[0, :],
        "RSKR effect (h_param = $(rskr_knob.h_param), n_iter = $(rskr_knob.n_iter))";
        fontsize = 26, font = :bold, padding = (0, 0, 8, 0),
    )

    panels = (
        (1, 1, raw_low, win_low, "Low bin · raw FBP", "μ (cm⁻¹)"),
        (1, 2, raw_high, win_high, "High bin · raw FBP", "μ (cm⁻¹)"),
        (2, 1, rskr_low, win_low, "Low bin · post-RSKR", "μ (cm⁻¹)"),
        (2, 2, rskr_high, win_high, "High bin · post-RSKR", "μ (cm⁻¹)"),
    )

    for (r, c, img, win, ttl, cblbl) in panels
        ax = CM.Axis(
            fig[r, 2c - 1];
            title = ttl, aspect = CM.DataAspect(),
            xticksvisible = false, yticksvisible = false,
            xticklabelsvisible = false, yticklabelsvisible = false,
            title_kwargs...,
        )
        CM.heatmap!(ax, img; colormap = :grays, colorrange = win)
        CM.Colorbar(
            fig[r, 2c]; colormap = :grays, colorrange = win,
            label = cblbl, width = 14, labelsize = 18, ticklabelsize = 12
        )
    end

    fig
end

# ╔═╡ 0400000b-0000-4000-8000-000000000040
md"""
### 10c. Per-bin radial cupping correction (μ-domain) — POST-RSKR

PCCT skips sino BHC (bin-effective μ_water from the M-matrix already
absorbs the bulk polychromatic bias per bin), but the FBP recon still
shows a residual radial cup/halo around the rod cluster.  RSKR (§10)
denoises the μ pair first; **then** [`BS.apply_radial_capping_basis!`](@ref)
flattens the radial profile per slice with an even polynomial in `r`,
fit on **quantile-IQR background voxels** (rods are excluded
automatically — no HU thresholds, no hand-tuned masks).

Runs in **μ-domain** so the cupping-corrected pair (`de_lohi_μ`) drives
every downstream stage (HU → Ding decomp → VMI → Mono+) without any
μ↔HU back-and-forth.  We `deepcopy` the RSKR volumes first so
`de_lohi_rskr` stays available for the §10b before/after figure.
"""

# ╔═╡ 0400000b-0000-4000-8000-000000000045
# Radial cupping correction kwargs — play with these.
#   fov_cm     ⇒ in-plane FOV (cm); pixel scale = fov_cm / nx
#   poly_order ⇒ even-poly terms beyond c₀ (1=parabolic, 2=+r⁴, 3=+r⁶ stiffer)
#   q_lo,q_hi  ⇒ in-FOV quantile range used as the "background" sample
#                (defaults exclude the ~25% brightest + ~25% darkest voxels,
#                which is enough to skip Gammex 472's rods + air ring)
#   enabled    ⇒ flip to `false` to bypass cupping entirely (sets
#                de_lohi_μ ≡ de_lohi_rskr)
cupping_knob = (
    enabled    = true,
    fov_cm     = 35.0,
    poly_order = 2,
    q_lo       = 0.25,
    q_hi       = 0.75,
);

# ╔═╡ 0400000b-0000-4000-8000-000000000050
de_lohi_μ = let
    if !cupping_knob.enabled
        de_lohi_rskr
    else
        vol_low_μ  = deepcopy(de_lohi_rskr.vol_low_μ)
        vol_high_μ = deepcopy(de_lohi_rskr.vol_high_μ)
        BS.apply_radial_capping_basis!(
            vol_low_μ, vol_high_μ;
            fov_cm     = cupping_knob.fov_cm,
            poly_order = cupping_knob.poly_order,
            q_lo       = cupping_knob.q_lo,
            q_hi       = cupping_knob.q_hi,
            verbose    = true,
        )
        (
            vol_low_μ  = vol_low_μ,
            vol_high_μ = vol_high_μ,
            geom       = de_lohi_rskr.geom,
        )
    end
end;

# ╔═╡ 04000006-0000-4000-8000-000000000001
md"""
## 9. Per-bin `μ_water` — measured from the post-cupping FBP

Forget analytic μ_water for now.  Just **measure** it: take an 8-px
circular ROI at the phantom center of the post-RSKR + post-cupping
μ-volume and use the mean as the HU divisor.  By construction this
makes solid water read at exactly 0 HU after §11's `to_hounsfield` step
— independent of any upstream RSKR / cupping residuals.

The center ROI is safely inside the 50 mm Ca inner ring + 105 mm I
outer ring of the Gammex 472, so the sample is pure solid water.
Summing across all z slices (rods are z-invariant cylinders) reduces
noise without diluting signal.
"""

# ╔═╡ 04000006-0000-4000-8000-000000000010
μ_water_per_bin = let
    nx, ny, nz = size(de_lohi_μ.vol_low_μ)
    cx, cy     = nx / 2 + 0.5, ny / 2 + 0.5
    ROI_R      = 8.0
    r²         = ROI_R^2

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
        s / n
    end

    (
        low  = Float64(_mean_μ(de_lohi_μ.vol_low_μ)),
        high = Float64(_mean_μ(de_lohi_μ.vol_high_μ)),
    )
end;

# ╔═╡ 04000006-0000-4000-8000-000000000020
md"""
**Measured μ_water (post-RSKR + post-cupping center ROI):**

* low  bin = $(round(μ_water_per_bin.low,  digits = 5)) cm⁻¹
* high bin = $(round(μ_water_per_bin.high, digits = 5)) cm⁻¹
"""

# ╔═╡ 0400000c-0000-4000-8000-000000000001
md"""
## 11. HU conversion

Per-bin `to_hounsfield` using the measured `μ_water_per_bin` from §9 (so
solid water reads exactly 0 HU by construction at both bins).

We deliberately **do not** call `add_system_noise_floor!` here — that's
the pattern nb01 / nb02 use to model DAS readout noise *after* a
non-iterative FBP, but in this notebook RSKR (§10) has already denoised
the bin pair, so adding a Gaussian floor on top would just defeat
RSKR's job and inject noise into the §12 calibration baseline.  Quantum
+ pileup + charge-sharing + electronic noise at sim time (via PCCT
forward model + scanner.electronic_noise) feeds RSKR with realistic
noise structure; RSKR removes most of it; the result is the cleanest
possible reference for the Ding optimizer to fit against.  Real-world
DAS readout floor is a downstream concern — apply it on the *deployed*
cal at use time, not during cal derivation.
"""

# ╔═╡ 0400000c-0000-4000-8000-000000000010
de_lohi_HU = let
    vol_low_HU  = Float32.(BS.to_hounsfield(de_lohi_μ.vol_low_μ;  μ_water = μ_water_per_bin.low))
    vol_high_HU = Float32.(BS.to_hounsfield(de_lohi_μ.vol_high_μ; μ_water = μ_water_per_bin.high))
    (vol_low_HU = vol_low_HU, vol_high_HU = vol_high_HU, geom = de_lohi_μ.geom)
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

# ╔═╡ 0400000d-0000-4000-8000-000000000063
md"""
## 12. Calibration

Image-domain Ding decomposition + VMI synthesis:
```
c_iodine(voxel) = a₀ + a₁·HU_low(voxel) + a₂·HU_high(voxel)
HU_E(voxel)     = HU_low(voxel) + c_iodine(voxel)·(μρ_I(E)/μρ_water(E) − α_iod_low_cal)
```

Two paths, gated by `optim_knob.enabled` (mirrors nb03 §11):

* **`enabled = true`** — direct optimization: BFGS (`Optim.jl`) with
  closed-form analytical gradient, minimizing per-(rod, energy)
  squared-error loss across all 4 VMI energies × all 14 Gammex 472 rods
  + center-ROI water.  Self-calibrates against this run's actual recon,
  so any upstream tweak (RSKR knobs, cupping, μ_water source) is
  picked up automatically.  `loss_metric = :rmse` works empirically
  better than `:nrmse` for this scanner — Ca rods anchor the fit and
  the iodine fit comes along for the ride.

* **`enabled = false`** — load the pre-optimized const from
  [`BS.pcct_vmi_cal_for(protocol.kVp)`](@ref).  Use this when you don't
  want to re-fit (e.g. running against a clinical scan where rod
  measurements aren't available).
"""

# ╔═╡ 0400000d-0000-4000-8000-000000000067
# Calibration optimizer knobs — flip `enabled` to switch between the
# src cal const and the BFGS optimizer.
#   max_iter, tol     ⇒ BFGS stopping criteria
#   loss_metric       ⇒ :nrmse (relative-error²) | :rmse (absolute-error²)
#   *_weight          ⇒ relative emphasis per rod family in the loss
#   energy_weights    ⇒ paired with `de_vmi_energies` (40 / 70 / 100 / 140 keV)
#   nrmse_floor       ⇒ HU floor on the relative-error denom (only used
#                       when loss_metric = :nrmse)
optim_knob = (
    enabled        = true,
    max_iter       = 200,
    tol            = 1.0e-8,
    loss_metric    = :rmse,                  # :nrmse | :rmse
    iodine_weight  = 1.0,
    calcium_weight = 1.0,
    water_weight   = 1.0,
    energy_weights = (1.0, 1.0, 1.0, 1.0),   # ↔ (40, 70, 100, 140) keV
    nrmse_floor    = 100.0,
);

# ╔═╡ 0400000d-0000-4000-8000-000000000074
de_cal = let
    if !optim_knob.enabled
        # ─── Path A: load pre-optimized const from src ────────────────────
        cal = BS.pcct_vmi_cal_for(protocol.kVp)
        (
            coeffs        = cal.coeffs,
            α_iod_low_cal = cal.α_iod_low_cal,
            μ_water_low   = Float32(μ_water_per_bin.low),
            μ_water_high  = Float32(μ_water_per_bin.high),
            method        = :src_const,
            n_iter        = 0,
            converged     = true,
            loss_metric   = :unknown,
            nrmse_HU      = NaN,
            rmse_HU       = NaN,
            rms_c         = NaN,
        )
    else
        # ─── Path B: BFGS optimizer ───────────────────────────────────────

        # 1. Per-rod 8-px ROI measurements (rods are z-invariant cylinders).
        mask_2d = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
        nx, ny  = size(mask_2d)
        ROI_R   = 8
        r²      = Float64(ROI_R)^2

        function _circular_roi(cx::Real, cy::Real)
            roi = CartesianIndex{2}[]
            i_lo = max(1, floor(Int, cx - ROI_R)); i_hi = min(nx, ceil(Int, cx + ROI_R))
            j_lo = max(1, floor(Int, cy - ROI_R)); j_hi = min(ny, ceil(Int, cy + ROI_R))
            for j in j_lo:j_hi, i in i_lo:i_hi
                ((i - cx)^2 + (j - cy)^2) ≤ r² && push!(roi, CartesianIndex(i, j))
            end
            roi
        end
        function _rod_roi(label::UInt8)
            idx = findall(==(label), mask_2d)
            cx = sum(ci -> Float64(ci[1]), idx) / length(idx)
            cy = sum(ci -> Float64(ci[2]), idx) / length(idx)
            _circular_roi(cx, cy)
        end
        function _mean_HU(vol, roi)
            s = 0.0; n = 0
            for z in 1:size(vol, 3), ci in roi
                s += vol[ci, z]; n += 1
            end
            s / n
        end

        # 2. All 14 Gammex rods + center-ROI water (mirrors nb03).
        rod_specs = (
            ("Water",  UInt8(0),  0.0, :water),
            ("Ca 50",  UInt8(10), 0.0, :calcium),
            ("Ca 100", UInt8(11), 0.0, :calcium),
            ("Ca 200", UInt8(12), 0.0, :calcium),
            ("Ca 300", UInt8(13), 0.0, :calcium),
            ("Ca 400", UInt8(14), 0.0, :calcium),
            ("Ca 500", UInt8(15), 0.0, :calcium),
            ("Ca 600", UInt8(16), 0.0, :calcium),
            ("I 2.0",  UInt8(20), 2.0, :iodine),
            ("I 2.5",  UInt8(21), 2.5, :iodine),
            ("I 5.0",  UInt8(22), 5.0, :iodine),
            ("I 7.5",  UInt8(23), 7.5, :iodine),
            ("I 10.0", UInt8(24), 10.0, :iodine),
            ("I 15.0", UInt8(25), 15.0, :iodine),
            ("I 20.0", UInt8(26), 20.0, :iodine),
        )

        materials = phantom_cpu.materials
        μ_water_E = Dict(
            E => BS.compute_μ_at_energy(BS.XA.Materials.water, E)
                for E in de_vmi_energies
        )

        rod_data = NamedTuple[]
        for (name, lab, c_known, kind) in rod_specs
            roi = name == "Water" ?
                _circular_roi(nx / 2 + 0.5, ny / 2 + 0.5) :
                _rod_roi(lab)
            material = name == "Water" ?
                BS.XA.Materials.water :
                materials[Int(lab) + 1]
            HU_low_r  = _mean_HU(de_lohi_HU.vol_low_HU,  roi)
            HU_high_r = _mean_HU(de_lohi_HU.vol_high_HU, roi)
            HU_theo   = Float64[
                let μ_r = BS.compute_μ_at_energy(material, E)
                    1000.0 * (μ_r - μ_water_E[E]) / μ_water_E[E]
                end
                    for E in de_vmi_energies
            ]
            push!(rod_data, (
                name = name, kind = kind, c_known = c_known,
                HU_low = HU_low_r, HU_high = HU_high_r, HU_theo = HU_theo,
            ))
        end

        # 3. αᴱ_phys per VMI energy.
        α_phys_E = Float64[
            BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, E) /
            BS.compute_mass_μ_at_energy(BS.XA.Materials.water,  E)
                for E in de_vmi_energies
        ]

        N = length(rod_data)
        M = length(de_vmi_energies)

        rod_family_w = [
            r.kind == :iodine  ? Float64(optim_knob.iodine_weight)  :
            r.kind == :calcium ? Float64(optim_knob.calcium_weight) :
                                 Float64(optim_knob.water_weight)
                for r in rod_data
        ]
        e_weights   = collect(Float64.(optim_knob.energy_weights))
        nrmse_floor = Float64(optim_knob.nrmse_floor)
        loss_metric = optim_knob.loss_metric
        loss_metric in (:nrmse, :rmse) ||
            error("optim_knob.loss_metric must be :nrmse or :rmse, got $(loss_metric)")
        function _w(r_idx, e_idx)
            base = rod_family_w[r_idx] * e_weights[e_idx]
            if loss_metric == :rmse
                base
            else
                ht = rod_data[r_idx].HU_theo[e_idx]
                base / max(abs(ht), nrmse_floor)^2
            end
        end

        # 4. Initial guess from BS.fit_ding_coeffs on iodine + water only.
        iod_water_idx = findall(r -> r.kind in (:iodine, :water), rod_data)
        fit_init = BS.fit_ding_coeffs(
            [rod_data[i].HU_low  for i in iod_water_idx],
            [rod_data[i].HU_high for i in iod_water_idx],
            [rod_data[i].c_known for i in iod_water_idx],
        )
        params0 = [Float64.(fit_init.coeffs)..., Float64(fit_init.α_low)]

        # 5. BFGS via Optim.jl with analytical gradient.
        function _resid_and_grad_terms(params, r_idx, e_idx)
            a₀, a₁, a₂, α_low = params
            hl    = rod_data[r_idx].HU_low
            hh    = rod_data[r_idx].HU_high
            ht    = rod_data[r_idx].HU_theo[e_idx]
            γ     = α_phys_E[e_idx] - α_low
            c_iod = a₀ + a₁ * hl + a₂ * hh
            HU_synth = hl + c_iod * γ
            (resid = HU_synth - ht, w = _w(r_idx, e_idx),
             γ = γ, c_iod = c_iod, hl = hl, hh = hh)
        end
        function loss_fn(params)
            total = 0.0
            for r_idx in 1:N, e_idx in 1:M
                t = _resid_and_grad_terms(params, r_idx, e_idx)
                total += t.w * t.resid^2
            end
            total
        end
        function grad_fn!(g, params)
            fill!(g, 0.0)
            for r_idx in 1:N, e_idx in 1:M
                t = _resid_and_grad_terms(params, r_idx, e_idx)
                two_w_r = 2.0 * t.w * t.resid
                g[1] += two_w_r * t.γ
                g[2] += two_w_r * t.γ * t.hl
                g[3] += two_w_r * t.γ * t.hh
                g[4] -= two_w_r * t.c_iod
            end
        end
        opt_options = Optim.Options(
            iterations = optim_knob.max_iter,
            f_abstol   = optim_knob.tol,
            g_abstol   = optim_knob.tol,
            x_abstol   = optim_knob.tol,
        )
        opt_result = Optim.optimize(loss_fn, grad_fn!, params0, Optim.BFGS(), opt_options)
        a₀, a₁, a₂, α_low_cal = Optim.minimizer(opt_result)
        converged = Optim.converged(opt_result)
        n_iter    = Optim.iterations(opt_result)

        # 6. Diagnostics — both metrics for cross-mode comparison.
        sq_rel, sq_abs, n_pairs = 0.0, 0.0, 0
        for r_idx in 1:N, e_idx in 1:M
            hl    = rod_data[r_idx].HU_low
            hh    = rod_data[r_idx].HU_high
            c_iod = a₀ + a₁ * hl + a₂ * hh
            HU_E_synth = hl + c_iod * (α_phys_E[e_idx] - α_low_cal)
            ht         = rod_data[r_idx].HU_theo[e_idx]
            resid      = HU_E_synth - ht
            sq_rel  += (resid / max(abs(ht), nrmse_floor))^2
            sq_abs  += resid^2
            n_pairs += 1
        end
        nrmse_HU = sqrt(sq_rel / n_pairs)
        rmse_HU  = sqrt(sq_abs / n_pairs)

        iod_idx = findall(r -> r.kind == :iodine, rod_data)
        rms_c   = let
            sq = 0.0
            for i in iod_idx
                c_pred = a₀ + a₁ * rod_data[i].HU_low + a₂ * rod_data[i].HU_high
                sq += (c_pred - rod_data[i].c_known)^2
            end
            sqrt(sq / length(iod_idx))
        end

        @info "[Ding optimizer · BFGS · loss=$(loss_metric)] $(converged ? "converged in" : "stopped at") $(n_iter) iter · loss = $(round(Optim.minimum(opt_result), digits = 4)) · nRMSE_HU = $(round(nrmse_HU, digits = 4)) · RMSE_HU = $(round(rmse_HU, digits = 1)) HU · RMS_c (I rods) = $(round(rms_c, digits = 3)) mg/mL"

        (
            coeffs        = Float32[a₀, a₁, a₂],
            α_iod_low_cal = Float32(α_low_cal),
            μ_water_low   = Float32(μ_water_per_bin.low),
            μ_water_high  = Float32(μ_water_per_bin.high),
            method        = :optimized,
            n_iter        = n_iter,
            converged     = converged,
            loss_metric   = loss_metric,
            nrmse_HU      = nrmse_HU,
            rmse_HU       = rmse_HU,
            rms_c         = rms_c,
        )
    end
end;

# ╔═╡ 0400000d-0000-4000-8000-000000000080
let
    optim_block = de_cal.method == :optimized ?
        """
        * **method = :optimized** · loss = `:$(de_cal.loss_metric)` · ($(de_cal.converged ? "converged in" : "stopped at") $(de_cal.n_iter) iter)
        * **nRMSE_HU = $(round(de_cal.nrmse_HU, digits = 4))** · **RMSE_HU = $(round(de_cal.rmse_HU, digits = 1)) HU** · RMS_c (I rods) = $(round(de_cal.rms_c, digits = 3)) mg/mL
        """ :
        "* **method = :src_const** (loaded from `BS.pcct_vmi_cal_for($(Int(protocol.kVp)))`)"

    body = """
    **Active calibration:**

    | Param | Value |
    |---|---|
    | a₀                | $(round(Float64(de_cal.coeffs[1]), digits = 5)) |
    | a₁                | $(round(Float64(de_cal.coeffs[2]), sigdigits = 5)) |
    | a₂                | $(round(Float64(de_cal.coeffs[3]), sigdigits = 5)) |
    | α_iod_low_cal     | $(round(Float64(de_cal.α_iod_low_cal), digits = 4)) |
    | μ_water_low       | $(round(Float64(de_cal.μ_water_low),  digits = 5)) cm⁻¹ |
    | μ_water_high      | $(round(Float64(de_cal.μ_water_high), digits = 5)) cm⁻¹ |

    $(optim_block)

    Synthesis equation:
    `HU_E = HU_low + (a₀ + a₁·HU_low + a₂·HU_high) · (μρ_I(E)/μρ_water(E) − α_iod_low_cal)`
    """
    Markdown.parse(body)
end

# ╔═╡ 0400000e-0000-4000-8000-000000000005
# z-direction denoising kwargs — play with these.
# `radius` ⇒ z-window = (2 · radius + 1) slices, shrunk at z boundaries
# (no padding bias).  Set `radius = 0` to disable z-denoising entirely.
# The Gammex 472 phantom is z-invariant in the rod cores, so larger radii
# are essentially lossless there; aggressive defaults are fine.
z_denoise_kwargs = (
    radius = 3,    # 7-slice window
)

# ╔═╡ 0400000e-0000-4000-8000-000000000010
de_decomp = let
    α_low_f32 = Float32(de_cal.α_iod_low_cal)

    # Raw decomposition outputs (per-voxel Ding + analytic c_water proxy)
    c_iodine_raw = BS.apply_ding_decomp(
        de_lohi_HU.vol_low_HU, de_lohi_HU.vol_high_HU, de_cal.coeffs
    )
    c_water_raw  = @. (de_lohi_HU.vol_low_HU - α_low_f32 * c_iodine_raw) /
                      1000.0f0 + 1.0f0

    # z-direction denoise — both basis maps + the HU pair (the latter is
    # what §13's VMI synth uses as its low-energy baseline; un-denoised
    # HU_low leaks its noise into every output VMI).
    radius = z_denoise_kwargs.radius
    c_iodine    = BS.apply_median_z(c_iodine_raw;          radius = radius)
    c_water     = BS.apply_median_z(c_water_raw;           radius = radius)
    vol_low_HU  = BS.apply_median_z(de_lohi_HU.vol_low_HU;  radius = radius)
    vol_high_HU = BS.apply_median_z(de_lohi_HU.vol_high_HU; radius = radius)

    (
        # Raw — for the §12b before/after figure
        c_iodine_raw    = c_iodine_raw,
        c_water_raw     = c_water_raw,
        vol_low_HU_raw  = de_lohi_HU.vol_low_HU,
        vol_high_HU_raw = de_lohi_HU.vol_high_HU,

        # z-denoised — feeds §13 VMI synth
        c_iodine    = c_iodine,
        c_water     = c_water,
        vol_low_HU  = vol_low_HU,
        vol_high_HU = vol_high_HU,
        geom        = de_lohi_HU.geom,
    )
end;

# ╔═╡ 0400000f-0000-4000-8000-000000000001
md"""
## 13. VMI synthesis at 40 / 70 / 100 / 140 keV

[`BS.synth_vmi_image_domain`](@ref) implements
```
HU_E[v] = HU_low[v] + c_iodine[v] · (μρ_I(E)/μρ_water(E) − α_iod_low_cal)
```
where the calibration coefficients come from `de_cal` (loaded from
`src/reconstruction/vmi/pcct_calibration.jl` per §12).
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
            α_iod_low_cal = de_cal.α_iod_low_cal,
            iodine_material = BS.XA.Elements.Iodine,
        )
    end
    out
end;

# ╔═╡ 04000010-0000-4000-8000-000000000001


# ╔═╡ 04000010-0000-4000-8000-000000000010
de_vmi_mono = let
    vols_in = [de_vmi_raw[E] for E in de_vmi_energies]
    σ_vec = Float64[1.0, 0.0, 1.0, 1.0]   # paired with de_vmi_energies

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


# ╔═╡ 04000011-0000-4000-8000-000000000010
let
    fig = CM.Figure(size = (1200, 1180))

    mid = size(de_vmi_mono[de_vmi_energies[1]], 3) ÷ 2
    colorrng = (-200, 500)
    title_kwargs = (titlesize = 28, subtitlesize = 20)

    grid_pos = [(1, 1), (1, 2), (2, 1), (2, 2)]
    hms = nothing
    for (i, E) in enumerate(de_vmi_energies)
        r, c = grid_pos[i]
        ax = CM.Axis(
            fig[r, c];
            title = "VMI $(Int(E)) keV",
            subtitle = i == 1 ? "Mono+ post-processed · 140 kVp PCCT · Naeotom Alpha" : "Mono+ post-processed",
            aspect = CM.DataAspect(),
            title_kwargs...,
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

    CM.Colorbar(fig[1:2, 3], hms; label = "HU", width = 14, labelsize = 18)

    CM.save(
        joinpath(@__DIR__, "..", "assets", "pcct_vmi_4panel.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 04000012-0000-4000-8000-000000000001


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

# ╔═╡ 04000012-0000-4000-8000-000000000020
# Water readout per VMI keV — 8-px circular ROI at the phantom geometric
# center on each post-Mono+ VMI volume.  Theoretical water HU is 0 by
# definition, so any reading IS the residual.
let
    nx = size(de_vmi_mono[de_vmi_energies[1]], 1)
    ny = size(de_vmi_mono[de_vmi_energies[1]], 2)
    cx = nx / 2 + 0.5
    cy = ny / 2 + 0.5
    R = 8
    water_roi = let
        roi = CartesianIndex{2}[]
        i_lo = max(1, floor(Int, cx - R)); i_hi = min(nx, ceil(Int, cx + R))
        j_lo = max(1, floor(Int, cy - R)); j_hi = min(ny, ceil(Int, cy + R))
        r² = Float64(R)^2
        for j in j_lo:j_hi, i in i_lo:i_hi
            ((i - cx)^2 + (j - cy)^2) ≤ r² && push!(roi, CartesianIndex(i, j))
        end
        roi
    end

    function _mean_at_water_roi(vol)
        n_z = size(vol, 3)
        s = 0.0; n = 0
        for z in 1:n_z, ci in water_roi
            s += vol[ci, z]; n += 1
        end
        return s / n
    end

    rows = String[]
    for E in de_vmi_energies
        hu = _mean_at_water_roi(de_vmi_mono[E])
        push!(rows, "| $(Int(E)) keV | $(round(hu, digits = 2)) HU |")
    end

    body = """
    **Water region — VMI HU per energy** (theoretical = 0 HU at all keV;
    each entry is the residual after the full pipeline: synthesis → Mono+ → FOV mask).

    | Energy | Water HU |
    |---|---|
    $(join(rows, "\n"))
    """
    Markdown.parse(body)
end

# ╔═╡ 04000013-0000-4000-8000-000000000001


# ╔═╡ 04000013-0000-4000-8000-000000000010
let
    fig = CM.Figure(size = (1180, 580))

    cmap_ca = CM.cgrad(:Oranges, 7; categorical = true)
    cmap_i = CM.cgrad(:GnBu, 7; categorical = true)

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

    CM.save(
        joinpath(@__DIR__, "..", "assets", "pcct_vmi_vs_theoretical.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 04000014-0000-4000-8000-000000000001


# ╔═╡ 7b51f797-15cc-408d-a7a8-2fdb8df3f5ba


# ╔═╡ 82f8e413-935f-4823-99cb-23467cfba6eb


# ╔═╡ 04000014-0000-4000-8000-000000000010
let
    fig = CM.Figure(size = (1180, 620))

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
        return (slope = β, intercept = α, r² = r²)
    end

    panels = ((:Ca, "Calcium Rods", "50–600 mg/mL"), (:I, "Iodine Rods", "2–20 mg/mL"))
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
        "Water (O)" => (mat = :water, mg_per_mL = 0.0, label = UInt8(1)),
        "SW ref 1" => (mat = :solid_water, mg_per_mL = 0.0, label = UInt8(2)),
        "SW ref 2" => (mat = :solid_water, mg_per_mL = 0.0, label = UInt8(3)),
        "Ca 50" => (mat = :calcium, mg_per_mL = 50.0, label = ROD_LABELS.Ca[1]),
        "Ca 100" => (mat = :calcium, mg_per_mL = 100.0, label = ROD_LABELS.Ca[2]),
        "Ca 200" => (mat = :calcium, mg_per_mL = 200.0, label = ROD_LABELS.Ca[3]),
        "Ca 300" => (mat = :calcium, mg_per_mL = 300.0, label = ROD_LABELS.Ca[4]),
        "Ca 400" => (mat = :calcium, mg_per_mL = 400.0, label = ROD_LABELS.Ca[5]),
        "Water (I)" => (mat = :water, mg_per_mL = 0.0, label = UInt8(4)),
        "I 2.0" => (mat = :iodine, mg_per_mL = 2.0, label = ROD_LABELS.I[1]),
        "I 2.5" => (mat = :iodine, mg_per_mL = 2.5, label = ROD_LABELS.I[2]),
        "I 5.0" => (mat = :iodine, mg_per_mL = 5.0, label = ROD_LABELS.I[3]),
        "I 7.5" => (mat = :iodine, mg_per_mL = 7.5, label = ROD_LABELS.I[4]),
        "I 10.0" => (mat = :iodine, mg_per_mL = 10.0, label = ROD_LABELS.I[5]),
        "I 15.0" => (mat = :iodine, mg_per_mL = 15.0, label = ROD_LABELS.I[6]),
        "I 20.0" => (mat = :iodine, mg_per_mL = 20.0, label = ROD_LABELS.I[7]),
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
        return Dict{UInt8, Vector{CartesianIndex{2}}}(lab => roi)
    end

    println("# Auto-generated from notebook 04 §21 calibration cell.")
    println("# Siemens Naeotom Alpha — sim post-RSKR low/high-bin rod HUs (Gammex 472).")
    println("# Source: 140 kVp / $(round(protocol.mA, digits = 1)) mA / collimation $(protocol.collimation_mm) mm.")
    println()
    println("const SIEMENS_NAEOTOM_ALPHA_140KVP_CAL = Dict{String, NamedTuple}(")
    for nm in rod_order
        cc = rod_concentrations[nm]
        h_low = _safe_mean(de_lohi_HU.vol_low_HU, cc.label)
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
# ╠═04000001-0000-4000-8000-000000000032
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
# ╠═0400000b-0000-4000-8000-000000000005
# ╠═0400000b-0000-4000-8000-000000000010
# ╟─0400000b-0000-4000-8000-000000000020
# ╟─0400000b-0000-4000-8000-000000000030
# ╟─0400000b-0000-4000-8000-000000000040
# ╠═0400000b-0000-4000-8000-000000000045
# ╠═0400000b-0000-4000-8000-000000000050
# ╟─04000006-0000-4000-8000-000000000001
# ╠═04000006-0000-4000-8000-000000000010
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
# ╟─0400000d-0000-4000-8000-000000000063
# ╠═0400000d-0000-4000-8000-000000000067
# ╠═0400000d-0000-4000-8000-000000000074
# ╟─0400000d-0000-4000-8000-000000000080
# ╠═0400000e-0000-4000-8000-000000000005
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
# ╟─04000012-0000-4000-8000-000000000020
# ╟─04000013-0000-4000-8000-000000000001
# ╟─04000013-0000-4000-8000-000000000010
# ╟─04000014-0000-4000-8000-000000000001
# ╠═7b51f797-15cc-408d-a7a8-2fdb8df3f5ba
# ╠═82f8e413-935f-4823-99cb-23467cfba6eb
# ╟─04000014-0000-4000-8000-000000000010
# ╟─04000015-0000-4000-8000-000000000001
# ╟─04000015-0000-4000-8000-000000000010
# ╟─04000016-0000-4000-8000-000000000001
# ╟─04000016-0000-4000-8000-000000000010
# ╟─04000017-0000-4000-8000-000000000001
