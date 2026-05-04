### A Pluto.jl notebook ###
# v0.19.0

# ╔═╡ 03000001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 03000001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 03000001-0000-4000-8000-000000000003
using Statistics: mean, std

# ╔═╡ 03000001-0000-4000-8000-000000000010
md"""
# 03 · Dual-kVp VMI on Gammex 472

**Two acquisitions, one image-domain decomposition, four virtual monoenergetic
images — verified against XrayAttenuation.jl theory.**

This notebook walks the **image-domain dual-energy pipeline** that the
`BasisSimulator.jl` clinical verification suite uses for its GE Apex Elite
GSI runs:

```
FBP per kVp (μ-domain)
   → RSKR-2ch joint denoise
   → HU + system noise floor
   → Ding 2012 image-domain decomposition  (self-calibrated from pre-RSKR rods)
   → z-median speckle removal on c_iodine
   → VMI synthesis at 40 / 70 / 100 / 140 keV
   → radial cupping correction
   → Mono+ post-processing  (FBP-equivalent noise shaping)
```

We finish with the unique verification plot of this notebook: **per-rod
measured HU vs theoretical HU computed directly from XrayAttenuation.jl** at
each VMI energy, separated into Ca-rod and I-rod panels so the iodine K-edge
boost reads cleanly.
"""

# ╔═╡ 03000001-0000-4000-8000-000000000020
md"""
## Setup

Same project + GPU detection idiom as notebook 01. The `to_gpu(...)` one-liner
auto-resolves to `MtlArray` / `CuArray` / `ROCArray` / `Array` based on which
backend is installed on the host.
"""

# ╔═╡ 03000001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 03000001-0000-4000-8000-000000000031
import CairoMakie as CM

# ╔═╡ 03000001-0000-4000-8000-000000000032
import Optim

# ╔═╡ 03000001-0000-4000-8000-000000000040
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

# ╔═╡ 03000001-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 03000002-0000-4000-8000-000000000001
md"""
## 1. `Phantom` — Gammex Model 472

Same factory as notebook 01.  Each rod is a labeled region of the mask, so
once we have HU volumes back at the end we can read out per-rod statistics
**by mask label** — no polar-coordinate ROI placement needed.
"""

# ╔═╡ 03000002-0000-4000-8000-000000000010
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,
    n_slices = 16,
    fov_cm = 35.0,
    z_cm = 1.0,
);

# ╔═╡ 03000002-0000-4000-8000-000000000020
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 03000003-0000-4000-8000-000000000001
md"""
## 2. `Scanner` — GE Revolution Apex Elite

Identical hardware spec to notebook 01.  Real GE GSI uses rapid kVp switching
on the same physical scanner, so both 80 and 140 kVp share this `Scanner`.
"""

# ╔═╡ 03000003-0000-4000-8000-000000000010
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

# ╔═╡ 03000004-0000-4000-8000-000000000001
md"""
## 3. Dual-kVp `CTProtocol`s

GSI rapid kVp switching alternates 80/140 kVp every other view.  Effective
mA per kVp is the instantaneous tube current scaled by each kVp's duty
cycle, so on a clinical Apex Elite (407 instantaneous mA at 80 kVp,
405 at 140 kVp, 0.5 s rotation):

| kVp | Instantaneous mA | Duty cycle | Effective mA |
|-----|------------------|------------|--------------|
| 80  | 407              | 0.65       | 264.55       |
| 140 | 405              | 0.35       | 141.75       |

Same `additional_filters = [("Al", 4.5)]` as notebook 01 (matches the GE Apex
inherent filtration).
"""

# ╔═╡ 03000004-0000-4000-8000-000000000010
protocol_low = BS.CTProtocol(
    kVp = 80,
    mA = 407 * 0.65,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 03000004-0000-4000-8000-000000000020
protocol_high = BS.CTProtocol(
    kVp = 140,
    mA = 405 * 0.35,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 03000005-0000-4000-8000-000000000001
md"""
## 4. `SimOptions` and `ReconOptions`

`:eict` for energy-integrating CT (same as notebook 01) and FDK reconstruction
on a 512² × 8 grid.  The same options struct is reused for both kVps.
"""

# ╔═╡ 03000005-0000-4000-8000-000000000010
sim_opts = BS.SimOptions(
    fidelity = :eict,
    seed = 1234,
);

# ╔═╡ 03000005-0000-4000-8000-000000000020
recon_opts = let
    slice_thickness_mm = 0.625
    n_recon_slices = round(Int, 5.0 / slice_thickness_mm)
    BS.ReconOptions(
        algorithm = :fdk,
        matrix_size = (512, 512, n_recon_slices),
        fov_cm = 35.0,
        z_cm = 0.5,
        filter = :standard,
    )
end;

# ╔═╡ 03000007-0000-4000-8000-000000000001
md"""
## 5. Forward Project (80 kVp)

Standard `let ... end` workspace pattern from notebook 01: allocate, run,
copy CPU result, drop GPU references, force a real GC.  The Pluto worker only
holds the CPU sinogram between cells.
"""

# ╔═╡ 03000007-0000-4000-8000-000000000010
sim_low = let
    @info "Simulating: 80 kVp / $(round(protocol_low.mA, digits = 1)) mA-eff (DE low)…"
    ws = BS.create_eict_workspace(scanner, protocol_low, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, scanner, protocol_low, sim_opts, recon_opts)

    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom)

    ws = nothing
    GC.gc(true)
    result
end;

# ╔═╡ 03000008-0000-4000-8000-000000000001
md"""
## 6. Forward Project (140 kVp)

Identical pattern; only the protocol is swapped.  Because the previous block
already cleaned up its GPU buffers, this acquisition runs on a fresh device.
"""

# ╔═╡ 03000008-0000-4000-8000-000000000010
sim_high = let
    @info "Simulating: 140 kVp / $(round(protocol_high.mA, digits = 1)) mA-eff (DE high)…"
    ws = BS.create_eict_workspace(scanner, protocol_high, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, scanner, protocol_high, sim_opts, recon_opts)

    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom)

    ws = nothing
    GC.gc(true)
    result
end;

# ╔═╡ 03000009-0000-4000-8000-000000000001
md"""
## 7. Per-kVp sino BHC + FBP → μ-domain volumes

Two stages, per kVp:

1. **Sinogram-domain BHC** ([`BS.calibrate_bhc_two_material`](@ref) →
   [`BS.apply_bhc_two_material`](@ref)) — water+bone polynomial fit
   from the per-kVp source spectrum, applied to each line integral.
   Removes the bulk polychromatic bias before reconstruction.
2. **FDK** with the GE-tuned apodization filter
   `(0.0, 0.25, 0.5, 0.75, 1.0) → (1.0, 0.95, 0.85, 0.65, 0.4)`.

We deliberately stop after sino BHC — no image-domain BHC refinement,
no cupping correction (yet) — to keep the pipeline μ-domain-only between
here and §10's HU conversion.  HU conversion is still **deferred** until
after RSKR (§9) so the joint denoiser operates on the raw μ pair where
iodine and water noise are maximally anti-correlated.

The §11 Ding calibration table (`GE_REVOLUTION_APEX_ELITE_DE_CAL`) was
measured on *uncorrected* polychromatic post-RSKR HU.  With sino BHC
inserted here the rod HU baselines shift slightly, so the cal will be
mildly off until it's recalibrated against this notebook's actual rod
HUs — the §16 regression will tell us how much.
"""

# ╔═╡ 03000009-0000-4000-8000-000000000003
# Sinogram-BHC toggle.  `enabled = true`  ⇒ apply per-column BHC pre-FDK
# and use BHC's mono `μ_water_ref` as the HU divisor in §10 (recon reads
# as approximately monochromatic at ref_E).  `enabled = false` ⇒ pass
# the raw sinogram straight to FDK and use a phantom-hardened
# polychromatic μ_water as the HU divisor (the OLD nb03 behavior).
# Cupping/RSKR/decomp/VMI/Mono+ are unaffected — they consume whatever
# μ/HU volumes come out of the upstream stages.
bhc_knob = (
    enabled = true,
);

# ╔═╡ 03000009-0000-4000-8000-000000000004
md"""
### 7a. Per-kVp BHC calibration (bowtie-aware)

One BHC model per kVp via the high-level
[`BS.calibrate_bhc_two_material(sim_opts, protocol; scanner, geom, …)`](@ref)
entry — auto-resolves the bowtie-hardened source spectrum, collapses to
per-column weights, and fits one polynomial per detector column
(`TwoMaterialBHCPerColumn`).  This captures the GE Revolution large
bowtie's column-dependent spectral shaping exactly; a single global
polynomial would leave residual radial cupping behind that no amount of
downstream correction recovers cleanly.

We always compute both:
* `μ_water_mono` — BHC's calibrated `μ_water_ref` at the column-mean
  spectrum's mean energy.  HU divisor when `bhc_knob.enabled = true`.
* `μ_water_poly` — phantom-hardened polychromatic-effective μ_water from
  [`BS.compute_polychromatic_μ_water`](@ref), with `water_path_cm` pulled
  from the actual phantom mask.  HU divisor when `bhc_knob.enabled = false`.

§8 picks whichever matches the active toggle.
"""

# ╔═╡ 03000009-0000-4000-8000-000000000007
bhc_cal = let
    body_diameter_cm = BS.estimate_phantom_diameter_cm(
        phantom_cpu.mask, phantom_cpu.voxel_size .* 10.0,
    )

    function _calibrate(protocol, geom)
        model = BS.calibrate_bhc_two_material(
            sim_opts, protocol;
            scanner = scanner, geom = geom,
            order   = 5,
            hu_low  = 450.0,
            hu_high = 600.0,
        )
        μ_water_poly = BS.compute_polychromatic_μ_water(
            sim_opts, protocol;
            scanner = scanner, geom = geom,
            water_path_cm = body_diameter_cm,
        )
        (
            model         = model,
            μ_water_mono  = model.μ_water_ref,       # post-BHC HU divisor
            μ_water_poly  = μ_water_poly,            # no-BHC HU divisor
            ref_E_keV     = model.reference_energy_keV,
        )
    end
    Dict(
        80  => _calibrate(protocol_low,  sim_low.geom),
        140 => _calibrate(protocol_high, sim_high.geom),
    )
end;

# ╔═╡ 03000009-0000-4000-8000-000000000008
md"""
**Calibrated:**  *(active divisor highlighted by the toggle)*

* 80 kVp:  ref energy = $(round(bhc_cal[80].ref_E_keV,  digits = 1)) keV  ·  μ_water_mono = $(round(bhc_cal[80].μ_water_mono,  digits = 5))  ·  μ_water_poly = $(round(bhc_cal[80].μ_water_poly,  digits = 5)) cm⁻¹
* 140 kVp: ref energy = $(round(bhc_cal[140].ref_E_keV, digits = 1)) keV  ·  μ_water_mono = $(round(bhc_cal[140].μ_water_mono, digits = 5))  ·  μ_water_poly = $(round(bhc_cal[140].μ_water_poly, digits = 5)) cm⁻¹

**`bhc_knob.enabled = $(bhc_knob.enabled)`** → §10 uses **`μ_water_$(bhc_knob.enabled ? "mono" : "poly")`**.
"""

# ╔═╡ 03000009-0000-4000-8000-000000000010
de_lohi_μ_raw = let
    matrix_size = recon_opts.matrix_size

    function _bhc_fbp_to_μ(sino_cpu, geom, bhc)
        sino_gpu = to_gpu(Float32.(sino_cpu))

        # 1. Sinogram-domain BHC — gated by `bhc_knob.enabled`.
        if bhc_knob.enabled
            sino_bhc = BS.apply_bhc_two_material(
                sino_gpu, bhc.model, geom, matrix_size;
                volume_extent = phantom.extent,
            )
            sino_gpu = to_gpu(sino_bhc)
        end

        # 2. FDK with the GE-tuned apodization
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, geom, matrix_size;
            filter = BS.CustomFilter(
                (0.0, 0.25, 0.5, 0.75, 1.0),
                (1.0, 0.95, 0.85, 0.65, 0.4),
            ),
        )
        recon_μ = Array(BS.reconstruct!(ws, sino_gpu, geom, matrix_size))

        ws = nothing; sino_gpu = nothing
        GC.gc(true)
        return Float32.(recon_μ)
    end

    vol_low_μ  = _bhc_fbp_to_μ(sim_low.sino,  sim_low.geom,  bhc_cal[80])
    vol_high_μ = _bhc_fbp_to_μ(sim_high.sino, sim_high.geom, bhc_cal[140])

    (
        vol_low_μ  = vol_low_μ,
        vol_high_μ = vol_high_μ,
        geom = sim_low.geom,
    )
end;

# ╔═╡ 03000009-0000-4000-8000-000000000025
# RSKR hyperparameters — play with these.  Drives §9's joint denoise (the
# bottom row of the figure below).  Defaults match notebook 04 (Naeotom
# Alpha PCCT VMI) exactly so cross-notebook noise behavior is directly
# comparable.
#   n_iter   ⇒ outer-loop iterations of the joint SVD + bilateral
#   h_param  ⇒ bilateral filter strength (lower = less smoothing)
#   radius   ⇒ neighborhood half-width in voxels
#   γ        ⇒ singular-value soft-thresholding factor
rskr_knob = (
    n_iter  = 2,
    h_param = 2,
    radius  = 3,
    γ       = 0.1,
);

# ╔═╡ 03000010-0000-4000-8000-000000000001
md"""
## 9. RSKR-2ch joint denoising (μ-domain)

[`apply_rskr`](@ref) runs a **joint SVD + bilateral filter** on the
`(vol_low_μ, vol_high_μ)` pair (Clark & Badea 2023).  Operating on the
μ-pair *before* HU conversion preserves the anti-correlated noise structure
between iodine-rich (high in 80 kVp HU) and iodine-poor (high in 140 kVp HU)
regions.  Net effect: the noise drops by 2–3× without smearing the iodine
contrast.

!!! info "Why μ-domain, not HU-domain"
    HU conversion is a per-kVp linear scaling, so it shifts each volume's
    statistics independently and breaks the joint covariance that RSKR
    exploits.  Always denoise *before* HU conversion when both volumes share
    a noise basis.
"""

# ╔═╡ 03000010-0000-4000-8000-000000000010
de_lohi_rskr = let
    # `GPU_BACKEND.to_gpu` is the unparameterized array constructor
    # (`MtlArray` / `CuArray` / `ROCArray` on a GPU host, plain
    # `identity` on CPU).  `apply_rskr` calls `gpu_arr_type(U_vols[i])`
    # on the 3-D U-columns; AcceleratedKernels then dispatches the joint
    # bilateral filter on whatever array type comes back — Metal/CUDA/
    # ROCm kernels on a GPU, threaded CPU loops with `identity`.  Same
    # source, any backend.
    out = BS.apply_rskr(
        [de_lohi_μ_raw.vol_low_μ, de_lohi_μ_raw.vol_high_μ];
        n_iter  = rskr_knob.n_iter,
        h_param = rskr_knob.h_param,
        radius  = rskr_knob.radius,
        γ       = rskr_knob.γ,
        gpu_arr_type = GPU_BACKEND.to_gpu,
        verbose = true,
    )
    (
        vol_low_μ = out[1],
        vol_high_μ = out[2],
        geom = de_lohi_μ_raw.geom,
    )
end;

# ╔═╡ 03000009-0000-4000-8000-000000000015
md"""
### 7c. Per-kVp radial cupping correction (μ-domain) — POST-RSKR

Sino BHC removes the bulk polychromatic bias but leaves a residual
radial cup/halo around the rod cluster.  RSKR (§9) denoises the μ pair
first — operating on noisy data, the joint SVD + bilateral filter is
where it should be — and **then** [`BS.apply_radial_capping_basis!`](@ref)
flattens the radial profile per slice with an even polynomial in `r`,
fit on **quantile-IQR background voxels** (so rods are excluded
automatically — no HU thresholds, no hand-tuned masks).

Runs in **μ-domain** so the cupping-corrected pair (`de_lohi_μ`) drives
every downstream stage (HU → Ding decomp → VMI → Mono+) without any
μ↔HU back-and-forth.  We `deepcopy` the RSKR volumes first so
`de_lohi_rskr` stays available for inspection.
"""

# ╔═╡ 03000009-0000-4000-8000-000000000017
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
    poly_order = 3,
    q_lo       = 0.50,
    q_hi       = 0.80,
);

# ╔═╡ 03000009-0000-4000-8000-000000000020
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

# ╔═╡ 03000006-0000-4000-8000-000000000001
md"""
## 8. Per-kVp `μ_water` — measured from the post-RSKR FBP

Forget analytic μ_water for now.  Just **measure** it: take an 8-px
circular ROI at the phantom center of the post-RSKR μ-volume and use
the mean as the HU divisor.  By construction this makes solid water
read at exactly 0 HU after §10's `to_hounsfield` step — independent of
any upstream BHC / cupping residuals.

The center ROI is safely inside the 50 mm Ca inner ring + 105 mm I
outer ring of the Gammex 472, so the sample is pure solid water.
Summing across all z slices (rods are z-invariant cylinders) reduces
noise without diluting signal.

Both `bhc_cal[kVp].μ_water_mono` (BHC's mono-ref) and
`bhc_cal[kVp].μ_water_poly` (phantom-hardened polychromatic) are still
computed and available for inspection; they're just not what §10 uses
right now.
"""

# ╔═╡ 03000006-0000-4000-8000-000000000010
kVp_μ_water = let
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

    Dict(
        80  => Float64(_mean_μ(de_lohi_μ.vol_low_μ)),
        140 => Float64(_mean_μ(de_lohi_μ.vol_high_μ)),
    )
end;

# ╔═╡ 03000009-0000-4000-8000-000000000030
let
    HU_window = (-200, 500)

    fig = CM.Figure(size = (1100, 1080))
    title_kwargs = (titlesize = 28, subtitlesize = 20)

    mid = size(de_lohi_μ_raw.vol_low_μ, 3) ÷ 2

    # RSKR runs in μ (joint covariance lives there), but display in HU since
    # that's the unit §12's Ding decomp consumes.  Slice-only conversion
    # keeps the cell's residency small.  No system noise floor here — that
    # gets added in §10.  Top row reads `de_lohi_μ_raw` (sino-BHC + FBP,
    # PRE radial cupping); bottom row reads `de_lohi_μ` (post-RSKR + post-
    # cupping — the final μ-volume that §10 converts to HU).  The figure
    # shows the full cumulative transformation BHC+FBP → RSKR → cupping.
    to_HU_slice(vol, kVp) = let μw = kVp_μ_water[kVp]
        @. 1000.0f0 * (vol[:, :, mid] - μw) / μw
    end

    panels = (
        (1, 1, "80 kVp",  "BHC + FBP — raw",     to_HU_slice(de_lohi_μ_raw.vol_low_μ,  80)),
        (1, 2, "140 kVp", "BHC + FBP — raw",     to_HU_slice(de_lohi_μ_raw.vol_high_μ, 140)),
        (2, 1, "80 kVp",  "post-RSKR + cupping", to_HU_slice(de_lohi_μ.vol_low_μ,      80)),
        (2, 2, "140 kVp", "post-RSKR + cupping", to_HU_slice(de_lohi_μ.vol_high_μ,     140)),
    )

    hms = nothing
    for (r, c, ttl, sub, slice) in panels
        ax = CM.Axis(
            fig[r, c];
            title = ttl, subtitle = sub,
            aspect = CM.DataAspect(),
            title_kwargs...,
        )
        hms = CM.heatmap!(
            ax, slice;
            colormap = :grays, colorrange = HU_window,
        )
        CM.hidedecorations!(ax)
    end

    CM.Colorbar(fig[1:2, 3], hms; label = "HU", width = 14, labelsize = 18)

    fig
end

# ╔═╡ 03000006-0000-4000-8000-000000000020
md"""
lac water (80 kVp) = $(round(kVp_μ_water[80],  digits = 5)) cm⁻¹  ||
lac water (140 kVp) = $(round(kVp_μ_water[140], digits = 5)) cm⁻¹
"""

# ╔═╡ 03000011-0000-4000-8000-000000000001
md"""
## 10. HU conversion

Per-kVp `to_hounsfield` using the measured `kVp_μ_water` from §8 (so
solid water reads exactly 0 HU by construction).

We deliberately **do not** call `add_system_noise_floor!` here — that's
the pattern nb01 / nb02 use to model DAS readout noise *after* a
non-iterative FBP, but in this notebook RSKR (§9) has already denoised
the μ pair, so adding a 28 HU Gaussian floor on top would just defeat
RSKR's job and inject noise into the §11 calibration baseline.  Quantum
+ electronic noise at sim time (via `simulate!` and the scanner's
`electronic_noise` field) feeds RSKR with realistic noise structure;
RSKR removes most of it; the result is the cleanest possible reference
for the Ding optimizer to fit against.  Real-world DAS readout floor is
a downstream concern — apply it on the *deployed* cal at use time, not
during cal derivation.
"""

# ╔═╡ 03000011-0000-4000-8000-000000000010
de_lohi_HU = let
    vol_low_HU  = Float32.(BS.to_hounsfield(de_lohi_μ.vol_low_μ;  μ_water = kVp_μ_water[80]))
    vol_high_HU = Float32.(BS.to_hounsfield(de_lohi_μ.vol_high_μ; μ_water = kVp_μ_water[140]))
    (vol_low_HU = vol_low_HU, vol_high_HU = vol_high_HU, geom = de_lohi_μ.geom)
end;

# ╔═╡ 03000012-0000-4000-8000-000000000001
md"""
## 11. Ding calibration from `BS.GE_REVOLUTION_APEX_ELITE_DE_CAL`

The Ding image-domain decomposition expresses iodine concentration as a
linear function of the two HU volumes:

```
c_iodine[v] = a₀ + a₁ · HU_80[v] + a₂ · HU_140[v]    [mg/mL]
```

Two paths, gated by `optim_knob.enabled`:

* **`enabled = false`** — classic LSQ on iodine concentration via
  [`BS.fit_ding_coeffs`](@ref), anchored on the
  [`BS.GE_REVOLUTION_APEX_ELITE_DE_CAL`](@ref) cal-table HUs (frozen
  values measured on the *uncorrected* polychromatic post-RSKR pipeline).
  Coefficients absorb the polychromatic baseline by construction, but the
  table doesn't track upstream pipeline changes (BHC / cupping / etc.).

* **`enabled = true`** — direct optimization that finds `(a₀, a₁, a₂,
  α_iod_low_cal)` minimizing **per-rod relative-error nRMSE between
  measured VMI HU and theoretical** across all 4 VMI energies and all 14
  Gammex 472 rods + water (the same loss nb04's PCCT cal optimizer used).
  Solver is **BFGS** (`Optim.jl`) with closed-form analytical gradient —
  the loss is smooth and bilinear in 4 params, so Newton-step search
  handles the joint optimization in one shot.  Initial guess seeded from
  `BS.fit_ding_coeffs` on iodine + water rods.  Self-calibrates against
  this run's actual recon, so it picks up every upstream change
  automatically.
"""

# ╔═╡ 03000012-0000-4000-8000-000000000005
# # Calibration optimizer knobs — flip `enabled` to switch between the
# # table-fit and the BFGS optimizer.
# #   max_iter, tol     ⇒ BFGS stopping criteria (iterations cap, |Δf|/|Δx|/|∇| tols)
# #   iodine_weight     ⇒ relative emphasis on the 7 iodine rods (2.0–20.0 mg/mL)
# #   calcium_weight    ⇒ relative emphasis on the 7 calcium rods (50–600 mg/mL)
# #   water_weight      ⇒ relative emphasis on the center-ROI water sample
# #                       (set to 0 to remove water from the fit entirely;
# #                       useful when BHC residual leaves water at a non-zero
# #                       baseline that the iodine basis can't absorb)
# #   energy_weights    ⇒ paired one-to-one with `de_vmi_energies`
# #                       (40 / 70 / 100 / 140 keV) — bump the keV you most
# #                       care about, e.g. (2.0, 1.0, 1.0, 0.5) prioritizes
# #                       40 keV iodine contrast and de-emphasizes 140 keV
# #   nrmse_floor       ⇒ HU floor on the relative-error denominator: a (rod,
# #                       energy) loss term is `((HU_synth - HU_theo) /
# #                       max(|HU_theo|, nrmse_floor))²`.  Without a floor,
# #                       low-|HU_theo| rods (water, low-c iodine at high keV)
# #                       blow up the loss and dominate the fit.  100 HU
# #                       matches the d328e79 src comment for nb04's PCCT cal.
# optim_knob = (
#     enabled        = true,
#     max_iter       = 200,
#     tol            = 1.0e-8,
#     iodine_weight  = 1.0,
#     calcium_weight = 1.0,
#     water_weight   = 1.0,
#     energy_weights = (1.0, 1.0, 1.0, 1.0),   # ↔ (40, 70, 100, 140) keV
#     nrmse_floor    = 100.0,
# );

# ╔═╡ 03000012-0000-4000-8000-000000000030
md"""
### 11b. Re-measure rod HUs from this run (cal-table refresh)

The §11 fit is still anchored on `BS.GE_REVOLUTION_APEX_ELITE_DE_CAL`,
which holds rod HUs measured against the **uncorrected** polychromatic
post-RSKR pipeline.  Now that we run sino BHC + bowtie-aware + cupping
+ z-denoise upstream, the rod HU baselines have shifted.

This cell re-measures the iodine + water rods straight off
`de_lohi_HU.vol_*_HU` (post-RSKR + system noise floor — the volumes
that §12's Ding decomp consumes), using the same **8-px-radius core
ROI** pattern the source const documents.  Compare against the const,
then drop the new values into
`src/reconstruction/vmi/clinical_calibrations.jl` so the cal stays
self-consistent with the corrected pipeline.

The Ca rows in the const are clinical FBP DICOM and aren't measured
here — leave them as-is when refreshing.
"""

# ╔═╡ 03000012-0000-4000-8000-000000000040
new_cal_measurements = let
    mask_2d = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
    nx, ny = size(mask_2d)
    ROI_RADIUS_PX = 8
    r² = Float64(ROI_RADIUS_PX)^2

    function _circular_roi(cx::Real, cy::Real)
        i_lo = max(1, floor(Int, cx - ROI_RADIUS_PX))
        i_hi = min(nx, ceil(Int, cx + ROI_RADIUS_PX))
        j_lo = max(1, floor(Int, cy - ROI_RADIUS_PX))
        j_hi = min(ny, ceil(Int, cy + ROI_RADIUS_PX))
        roi = CartesianIndex{2}[]
        for j in j_lo:j_hi, i in i_lo:i_hi
            ((i - cx)^2 + (j - cy)^2) ≤ r² && push!(roi, CartesianIndex(i, j))
        end
        roi
    end

    # Per-label rod ROI (centroid of the mask label, 8-px core).
    function _rod_roi(label::UInt8)
        idx = findall(==(label), mask_2d)
        isempty(idx) && error("rod_roi: no voxels with label $label")
        cx = sum(ci -> Float64(ci[1]), idx) / length(idx)
        cy = sum(ci -> Float64(ci[2]), idx) / length(idx)
        _circular_roi(cx, cy)
    end

    # Mean over (ROI × all z slices) — rods are z-invariant, so summing
    # across z averages the noise without diluting the rod signal.
    function _mean_HU(vol, roi)
        s = 0.0; n = 0
        for z in 1:size(vol, 3), ci in roi
            s += vol[ci, z]; n += 1
        end
        s / n
    end

    # Iodine rods (label 20–26 → 2.0…20.0 mg/mL)
    I_rod_specs = (
        ("I 2.0",  UInt8(20),  2.0),
        ("I 2.5",  UInt8(21),  2.5),
        ("I 5.0",  UInt8(22),  5.0),
        ("I 7.5",  UInt8(23),  7.5),
        ("I 10.0", UInt8(24), 10.0),
        ("I 15.0", UInt8(25), 15.0),
        ("I 20.0", UInt8(26), 20.0),
    )

    iodine_rows = [
        let roi = _rod_roi(lab)
            (
                name      = name,
                material  = :iodine,
                mg_per_mL = mgml,
                HU_80kVp  = Float32(_mean_HU(de_lohi_HU.vol_low_HU,  roi)),
                HU_140kVp = Float32(_mean_HU(de_lohi_HU.vol_high_HU, roi)),
            )
        end
        for (name, lab, mgml) in I_rod_specs
    ]

    # Water — center 8-px ROI (rod-free zone, same trick as §7d).
    water_roi = _circular_roi(nx / 2 + 0.5, ny / 2 + 0.5)
    water_HU_low  = Float32(_mean_HU(de_lohi_HU.vol_low_HU,  water_roi))
    water_HU_high = Float32(_mean_HU(de_lohi_HU.vol_high_HU, water_roi))

    (
        water = (HU_80kVp = water_HU_low, HU_140kVp = water_HU_high),
        iodine = iodine_rows,
    )
end;

# ╔═╡ 03000012-0000-4000-8000-000000000050
let
    # Helper to format a HU value with right-padded width for source code.
    fmt(x) = lpad(string(round(Float64(x), digits = 1)), 7, " ") * "f0"

    w = new_cal_measurements.water

    # Inspection table — old (const) vs new (this run) side-by-side.
    old_rods = BS.iodine_calibration_rods(
        BS.GE_REVOLUTION_APEX_ELITE_DE_CAL;
        hu_low_field = :HU_80kVp, hu_high_field = :HU_140kVp,
    )
    name_to_old = Dict(zip(old_rods.names, zip(old_rods.HU_low, old_rods.HU_high)))

    diff_rows = String[]
    push!(diff_rows,
        "| Water (center ROI) | 0.0 | — | — | $(round(w.HU_80kVp, digits=1)) | $(round(w.HU_140kVp, digits=1)) |",
    )
    for r in new_cal_measurements.iodine
        old_lo, old_hi = get(name_to_old, r.name, (NaN, NaN))
        push!(diff_rows,
            "| $(r.name) | $(r.mg_per_mL) | " *
            "$(isnan(old_lo) ? "—" : round(old_lo, digits=1)) | " *
            "$(isnan(old_hi) ? "—" : round(old_hi, digits=1)) | " *
            "$(round(r.HU_80kVp, digits=1)) | " *
            "$(round(r.HU_140kVp, digits=1)) |",
        )
    end

    # Copy-paste-ready Julia source.
    src_lines = String[]
    push!(src_lines,
        "    \"Water (O)\" => (material = :water,        mg_per_mL =   0.0, HU_80kVp = $(fmt(w.HU_80kVp)), HU_140kVp = $(fmt(w.HU_140kVp))),")
    push!(src_lines,
        "    \"Water (I)\" => (material = :water,        mg_per_mL =   0.0, HU_80kVp = $(fmt(w.HU_80kVp)), HU_140kVp = $(fmt(w.HU_140kVp))),")
    push!(src_lines,
        "    \"SW ref 1\"  => (material = :solid_water,  mg_per_mL =   0.0, HU_80kVp = $(fmt(w.HU_80kVp)), HU_140kVp = $(fmt(w.HU_140kVp))),")
    push!(src_lines,
        "    \"SW ref 2\"  => (material = :solid_water,  mg_per_mL =   0.0, HU_80kVp = $(fmt(w.HU_80kVp)), HU_140kVp = $(fmt(w.HU_140kVp))),")
    for r in new_cal_measurements.iodine
        name_padded = rpad("\"$(r.name)\"", 9, " ")
        mgml_padded = lpad(string(round(r.mg_per_mL, digits=1)), 5, " ")
        push!(src_lines,
            "    $(name_padded)   => (material = :iodine,       mg_per_mL = $(mgml_padded), HU_80kVp = $(fmt(r.HU_80kVp)), HU_140kVp = $(fmt(r.HU_140kVp))),")
    end

    body = """
    **Old (const) vs new (this run) iodine + water HUs:**

    | Rod | mg/mL | old HU 80 | old HU 140 | **new HU 80** | **new HU 140** |
    |-----|---|---|---|---|---|
    $(join(diff_rows, "\n"))

    **Copy-paste into `src/reconstruction/vmi/clinical_calibrations.jl`** — replace the water + iodine rows in `GE_REVOLUTION_APEX_ELITE_DE_CAL` (leave the Ca rows untouched, they're clinical reference values):

    ```julia
    $(join(src_lines, "\n"))
    ```
    """
    Markdown.parse(body)
end

# ╔═╡ 03000013-0000-4000-8000-000000000001
md"""
## 12. Apply Ding decomposition + z-direction denoising

[`BS.apply_ding_decomp`](@ref) is a per-voxel evaluation of the Ding
equation, returning a `c_iodine` map in mg/mL.  We pair it inline with a
water-equivalent density map `c_water` (g/mL — solid water ≈ 1, calcium
rods read higher) so both bases are visible in §12b.

Both basis maps + the post-RSKR HU pair then go through
[`BS.apply_median_z`](@ref), an axial-only running median that exploits
the phantom's z-invariance: Gammex 472 rods are cylinders extruded along
z, so any slice-to-slice variation is pure recon/sim noise.  Knobs live
in the dedicated kwargs cell below — see §12b for the row-by-row visual
(post-RSKR HU → raw decomp → z-denoised decomp).
"""

# ╔═╡ 03000013-0000-4000-8000-000000000005
# z-direction denoising kwargs — play with these.
# `radius` ⇒ z-window = (2 · radius + 1) slices, shrunk at z boundaries
# (no padding bias).  Set `radius = 0` to disable z-denoising entirely.
# The Gammex 472 phantom is z-invariant in the rod cores, so larger radii
# are essentially lossless there; aggressive defaults are fine.
z_denoise_kwargs = (
    radius = 2,    # 7-slice window on the 8-slice z stack
)

# ╔═╡ 03000014-0000-4000-8000-000000000001
md"""
## 13. VMI synthesis at 40 / 70 / 100 / 140 keV

[`BS.synth_vmi_image_domain`](@ref) implements

```
HU_E[v] = HU_low[v] + c_iodine[v] · (αᴱ_phys − α_low_cal)
```

where `αᴱ_phys = (μ/ρ)_iodine(E) / (μ/ρ)_water(E)` is the pure-physics HU
sensitivity at the target energy (computed internally from XrayAttenuation),
and `α_low_cal` is the empirical low-basis sensitivity returned by
`fit_ding_coeffs`.

We pick the standard four:

| keV | What it shows                                      |
|-----|-----------------------------------------------------|
| 40  | Maximum iodine contrast (bremsstrahlung peak ≈ K-edge) |
| 70  | ≈ standard 120 kVp HU values                        |
| 100 | Beam-hardening robust, low metal artifact           |
| 140 | Quasi-monochromatic high-energy reference           |
"""

# ╔═╡ 03000014-0000-4000-8000-000000000010
de_vmi_energies = [40.0, 70.0, 100.0, 140.0];

# ╔═╡ 03000016-0000-4000-8000-000000000001
md"""
## 14. Mono+ post-processing (FBP-equivalent noise shaping)

[`BS.apply_mono_plus!`](@ref) sharpens the contrast at the noise-quiet
anchor energy (70 keV here) and slightly low-passes the others.  Per-energy
σ in pixels:

| keV | σ_lp (px) | What happens                                         |
|-----|-----------|------------------------------------------------------|
| 40  | 1.5       | Noisiest VMI — strongest LP                          |
| 70  | 0.0       | Anchor energy — left untouched                       |
| 100 | 1.0       | Modest LP                                            |
| 140 | 1.0       | Modest LP                                            |

Workspace allocates once for the whole sweep; the call mutates `vols` in
place but copies are returned so each energy is independent.
"""

# ╔═╡ 03000017-0000-4000-8000-000000000001
md"""
## 15. The four VMI images

Mid-slice mosaic of all four energies, shared HU window (-200 to 1000 HU)
so iodine boost vs. soft-tissue baseline is comparable side-by-side.
"""

# ╔═╡ 03000018-0000-4000-8000-000000000001
md"""
## 16. Per-rod readout: measured HU vs theoretical HU

Now the unique plot of this notebook.  The Gammex 472 phantom has 14 rods
of clinical interest — 7 calcium concentrations (50–600 mg/mL) and 7 iodine
concentrations (2.0–20.0 mg/mL) — each occupying its own labeled region of
the phantom mask.

For each rod *r* and each VMI energy *E*:

* **Measured HU** = mean of an **8-pixel-radius circular ROI at the rod's
  centroid** (well inside the 21-px-radius rod core), broadcast across
  every recon slice.  The full-mask average would mix in partial-volume
  edge voxels that Mono+'s per-energy LP filter blurs differently at each
  keV — injecting an *energy-dependent* bias that flips the iodine HU
  ordering between 70 and 100 keV.  A small core ROI dodges that.
* **Theoretical HU** = `1000 · (μ_r(E) − μ_water(E)) / μ_water(E)` from
  XrayAttenuation directly, via [`BS.compute_μ_at_energy`](@ref).

The phantom's `materials` vector already has each rod's `XA.Material`
indexed by `mask_value + 1` — so the theoretical HU is a one-liner per
rod, no fitting, no calibration assumption.
"""

# ╔═╡ 03000018-0000-4000-8000-000000000010
ROD_LABELS = (
    Ca = (UInt8(10), UInt8(11), UInt8(12), UInt8(13), UInt8(14), UInt8(15), UInt8(16)),
    I = (UInt8(20), UInt8(21), UInt8(22), UInt8(23), UInt8(24), UInt8(25), UInt8(26)),
);

# ╔═╡ 03000018-0000-4000-8000-000000000020
ROD_NAMES = (
    Ca = ("50 mg/mL", "100 mg/mL", "200 mg/mL", "300 mg/mL", "400 mg/mL", "500 mg/mL", "600 mg/mL"),
    I = ("2.0 mg/mL", "2.5 mg/mL", "5.0 mg/mL", "7.5 mg/mL", "10.0 mg/mL", "15.0 mg/mL", "20.0 mg/mL"),
);

# ╔═╡ 03000018-0000-4000-8000-000000000030
md"""
We do the per-rod measurement on the **CPU** mask (cheaper than copying the
GPU mask back, and the mask is small).  `phantom_cpu.mask` is the canonical
labeled-region array.
"""

# ╔═╡ 03000019-0000-4000-8000-000000000001
md"""
## 17. The verification plot — measured vs theoretical, per rod

**Solid line** = measured HU at each VMI energy.
**Dashed line** = theoretical HU from XrayAttenuation directly (no
calibration, no recon — pure physics).

The Ca rods (left panel) follow a smooth roll-off as keV increases —
calcium attenuates predominantly through Compton scatter at clinical
energies, so its HU above water decreases monotonically.

The I rods (right panel) show the iodine **K-edge boost at 40 keV** —
iodine's K-edge sits at 33.2 keV, so a 40 keV VMI catches the
photoelectric edge and amplifies iodine HU dramatically (200–500 HU per
mg/mL) compared to the ≈70 keV plateau.  Above the K-edge the rolloff is
steep until 100+ keV where iodine looks soft-tissue-like.

If solid and dashed agree well, the simulator + decomposition + Mono+
pipeline are recovering the underlying physics.
"""

# ╔═╡ 0300001a-0000-4000-8000-000000000001
md"""
## 18. Linear regression: measured vs theoretical

Same data, different cut.  We scatter every (rod, energy) pair as
**measured HU on the y-axis vs theoretical HU on the x-axis**, then fit a
per-energy line and overlay the y = x identity.  Reading the plot:

* Slope close to **1**, intercept close to **0**, R² close to **1** → the
  pipeline recovers physics.
* Slope ≠ 1 → multiplicative calibration mismatch (clinical Ding
  coefficients vs simulator's polychromatic HU baseline).
* Intercept ≠ 0 → additive offset (residual cup, water-baseline shift).
* Low R² → non-linear distortion (partial volume, decomp non-linearity).

The Ca and I panels are split because Ca lives at much higher HU values
than I and would otherwise dominate a shared-axis fit.
"""

# ╔═╡ bc0b9af2-d4d0-44c5-8f93-62dc9ca5d6bd
# Calibration optimizer knobs — flip `enabled` to switch between the
# table-fit and the BFGS optimizer.
#   max_iter, tol     ⇒ BFGS stopping criteria (iterations cap, |Δf|/|Δx|/|∇| tols)
#   iodine_weight     ⇒ relative emphasis on the 7 iodine rods (2.0–20.0 mg/mL)
#   calcium_weight    ⇒ relative emphasis on the 7 calcium rods (50–600 mg/mL)
#   water_weight      ⇒ relative emphasis on the center-ROI water sample
#                       (set to 0 to remove water from the fit entirely;
#                       useful when BHC residual leaves water at a non-zero
#                       baseline that the iodine basis can't absorb)
#   energy_weights    ⇒ paired one-to-one with `de_vmi_energies`
#                       (40 / 70 / 100 / 140 keV) — bump the keV you most
#                       care about, e.g. (2.0, 1.0, 1.0, 0.5) prioritizes
#                       40 keV iodine contrast and de-emphasizes 140 keV
#   nrmse_floor       ⇒ HU floor on the relative-error denominator: a (rod,
#                       energy) loss term is `((HU_synth - HU_theo) /
#                       max(|HU_theo|, nrmse_floor))²`.  Without a floor,
#                       low-|HU_theo| rods (water, low-c iodine at high keV)
#                       blow up the loss and dominate the fit.  100 HU
#                       matches the d328e79 src comment for nb04's PCCT cal.
optim_knob = (
    enabled        = true,
    max_iter       = 200,
    tol            = 1.0e-8,
    loss_metric    = :rmse,                 # :nrmse | :rmse
    iodine_weight  = 1.0,
    calcium_weight = 1.0,
    water_weight   = 3.0,
    energy_weights = (1.0, 1.0, 0.0, 0.0),   # ↔ (40, 70, 100, 140) keV
    nrmse_floor    = 100.0,
);

# ╔═╡ 03000012-0000-4000-8000-000000000010
de_cal = let
    if !optim_knob.enabled
        # ─── Path A: classic table-fit on iodine + water rods ──────────
        rods = BS.iodine_calibration_rods(
            BS.GE_REVOLUTION_APEX_ELITE_DE_CAL;
            hu_low_field = :HU_80kVp,
            hu_high_field = :HU_140kVp,
        )
        fit = BS.fit_ding_coeffs(rods.HU_low, rods.HU_high, rods.mg_per_mL)
        (
            coeffs       = fit.coeffs,
            α_iod_low    = fit.α_low,
            α_iod_high   = fit.α_high,
            rms_c        = fit.rms,
            rod_names    = rods.names,
            rod_HU_low   = rods.HU_low,
            rod_HU_high  = rods.HU_high,
            rod_c_iodine = rods.mg_per_mL,
            method       = :table_fit,
            n_iter       = 0,
            converged    = true,
            loss_metric  = :nrmse,
            nrmse_HU     = NaN,
            rmse_HU      = NaN,
        )
    else
        # ─── Path B: direct VMI-HU-vs-theoretical optimizer ────────────

        # 1. Per-rod 8-px circular ROIs in xy (rods are z-invariant).
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

        # 2. Rod specs — all 14 Gammex rods + 1 center-ROI water sample.
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

        # 3. αᴱ_phys = (μ/ρ)_iodine(E) / (μ/ρ)_water(E) at each VMI energy.
        α_phys_E = Float64[
            BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, E) /
            BS.compute_mass_μ_at_energy(BS.XA.Materials.water,  E)
                for E in de_vmi_energies
        ]

        N = length(rod_data)
        M = length(de_vmi_energies)

        # Per-(rod, energy) weight: rod-family weight × energy weight ×
        # 1/max(|HU_theo|, 1)² (relative-error scaling).
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
                base                                           # absolute squared error
            else
                ht = rod_data[r_idx].HU_theo[e_idx]
                base / max(abs(ht), nrmse_floor)^2             # relative squared error (with floor)
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
        #
        # Loss   L(a₀, a₁, a₂, α_low) = Σ_(r,E) w_rE · (HU_synth_rE − HU_theo_rE)²
        # where  HU_synth = hl + (a₀ + a₁·hl + a₂·hh) · (αᴱ − α_low)
        #
        # The bilinear (a × α_low) coupling that breaks alternating LSQ
        # is just Newton-step territory for BFGS — single-start usually
        # finds the joint optimum directly.
        function _resid_and_grad_terms(params, r_idx, e_idx)
            a₀, a₁, a₂, α_low = params
            hl    = rod_data[r_idx].HU_low
            hh    = rod_data[r_idx].HU_high
            ht    = rod_data[r_idx].HU_theo[e_idx]
            γ     = α_phys_E[e_idx] - α_low
            c_iod = a₀ + a₁ * hl + a₂ * hh
            HU_synth = hl + c_iod * γ
            resid = HU_synth - ht
            w     = _w(r_idx, e_idx)
            (resid = resid, w = w, γ = γ, c_iod = c_iod, hl = hl, hh = hh)
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
                g[1] += two_w_r * t.γ              # ∂L/∂a₀
                g[2] += two_w_r * t.γ * t.hl       # ∂L/∂a₁
                g[3] += two_w_r * t.γ * t.hh       # ∂L/∂a₂
                g[4] -= two_w_r * t.c_iod          # ∂L/∂α_low
            end
        end

        opt_options = Optim.Options(
            iterations = optim_knob.max_iter,
            f_abstol   = optim_knob.tol,
            g_abstol   = optim_knob.tol,
            x_abstol   = optim_knob.tol,
        )
        opt_result = Optim.optimize(
            loss_fn, grad_fn!, params0, Optim.BFGS(), opt_options,
        )

        a₀, a₁, a₂, α_low_cal = Optim.minimizer(opt_result)
        converged = Optim.converged(opt_result)
        n_iter    = Optim.iterations(opt_result)

        # 6. Diagnostics — both metrics computed regardless of which drove
        # the loss, so you can compare across :nrmse / :rmse runs.
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

        iod_idx  = findall(r -> r.kind == :iodine, rod_data)
        rms_c    = let
            sq = 0.0
            for i in iod_idx
                c_pred = a₀ + a₁ * rod_data[i].HU_low + a₂ * rod_data[i].HU_high
                sq += (c_pred - rod_data[i].c_known)^2
            end
            sqrt(sq / length(iod_idx))
        end

        # α_high — slope-through-origin on iodine rods at high kVp (matches
        # fit_ding_coeffs convention; only used for diagnostics, synth uses α_low_cal).
        Σc²    = sum(rod_data[i].c_known^2 for i in iod_idx)
        α_high = sum(rod_data[i].c_known * rod_data[i].HU_high for i in iod_idx) / Σc²

        @info "[Ding optimizer · BFGS · loss=$(loss_metric)] $(converged ? "converged in" : "stopped at") $(n_iter) iter · loss = $(round(Optim.minimum(opt_result), digits = 4)) · nRMSE_HU = $(round(nrmse_HU, digits = 4)) · RMSE_HU = $(round(rmse_HU, digits = 1)) HU · RMS_c (I rods) = $(round(rms_c, digits = 3)) mg/mL"

        (
            coeffs       = (Float32(a₀), Float32(a₁), Float32(a₂)),
            α_iod_low    = Float32(α_low_cal),
            α_iod_high   = Float32(α_high),
            rms_c        = rms_c,
            rod_names    = [r.name    for r in rod_data],
            rod_HU_low   = [r.HU_low  for r in rod_data],
            rod_HU_high  = [r.HU_high for r in rod_data],
            rod_c_iodine = [r.c_known for r in rod_data],
            method       = :optimized,
            n_iter       = n_iter,
            converged    = converged,
            loss_metric  = loss_metric,
            nrmse_HU     = nrmse_HU,
            rmse_HU      = rmse_HU,
        )
    end
end;

# ╔═╡ 03000012-0000-4000-8000-000000000020
let
    rows = [
        "| $(de_cal.rod_names[i]) | $(de_cal.rod_c_iodine[i]) | " *
            "$(round(de_cal.rod_HU_low[i], digits = 1)) | " *
            "$(round(de_cal.rod_HU_high[i], digits = 1)) |"
            for i in eachindex(de_cal.rod_names)
    ]
    src_label = de_cal.method == :optimized ?
        "this run's de_lohi_HU (BHC + cupping + RSKR + noise floor), 8-px core ROI per rod" :
        "BS.GE_REVOLUTION_APEX_ELITE_DE_CAL (frozen const, post-RSKR sim FBP)"

    optim_block = de_cal.method == :optimized ?
        """
        * **method = :optimized** · loss = `:$(de_cal.loss_metric)` · ($(de_cal.converged ? "converged in" : "stopped at") $(de_cal.n_iter) iter)
        * **nRMSE_HU = $(round(de_cal.nrmse_HU, digits = 4))** (relative VMI HU error vs theory, floored at `nrmse_floor`)
        * **RMSE_HU = $(round(de_cal.rmse_HU, digits = 1)) HU** (absolute VMI HU error vs theory)
        """ :
        "* **method = :table_fit**"

    body = """
    **Calibration rods** — $(src_label):

    | Rod | c (mg/mL) | HU @ 80 kVp | HU @ 140 kVp |
    |-----|---|---|---|
    $(join(rows, "\n"))

    **Ding fit:**

    * a₀ = $(round(de_cal.coeffs[1], digits = 3))
    * a₁ = $(round(de_cal.coeffs[2], sigdigits = 4))
    * a₂ = $(round(de_cal.coeffs[3], sigdigits = 4))
    * α_iodine (low / high) = $(round(de_cal.α_iod_low, digits = 2)) / $(round(de_cal.α_iod_high, digits = 2)) HU per (mg/mL)
    * RMS calibration error (iodine rods) = $(round(de_cal.rms_c, digits = 3)) mg/mL
    $(optim_block)
    """
    Markdown.parse(body)
end

# ╔═╡ 03000013-0000-4000-8000-000000000010
de_decomp = let
    α_low_f32 = Float32(de_cal.α_iod_low)

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

# ╔═╡ 03000013-0000-4000-8000-000000000030
let
    HU_window      = (-200, 500)     # HU input panels
    c_iod_window   = (-2.0, 22.0)    # mg/mL — covers 2–20 mg/mL iodine rods
    c_water_window = (0.5, 2.0)      # g/mL — water ≈ 1, calcium rods reach ~2

    fig = CM.Figure(size = (1400, 1700))
    title_kwargs = (titlesize = 28, subtitlesize = 20)

    mid = size(de_decomp.c_iodine, 3) ÷ 2

    function panel!(r, c, slice2d, ttl, sub, cmap, crng, label)
        ax = CM.Axis(fig[r, c]; title = ttl, subtitle = sub,
            aspect = CM.DataAspect(), title_kwargs...)
        hm = CM.heatmap!(ax, slice2d; colormap = cmap, colorrange = crng)
        CM.hidedecorations!(ax)
        CM.Colorbar(fig[r, c, CM.Right()], hm;
            label = label, width = 14, labelsize = 18)
    end

    # Row 1 — post-RSKR HU pair (the inputs to the Ding decomp in §12)
    panel!(1, 1, de_decomp.vol_low_HU_raw[:, :, mid],
        "HU @ 80 kVp", "post-RSKR — low energy",
        :grays, HU_window, "HU")
    panel!(1, 2, de_decomp.vol_high_HU_raw[:, :, mid],
        "HU @ 140 kVp", "post-RSKR — high energy",
        :grays, HU_window, "HU")

    # Row 2 — raw decomposition outputs (BEFORE z-direction denoising)
    panel!(2, 1, de_decomp.c_iodine_raw[:, :, mid],
        "c_iodine", "post image-decomp · pre z-denoise",
        :viridis, c_iod_window, "mg/mL")
    panel!(2, 2, de_decomp.c_water_raw[:, :, mid],
        "c_water", "post image-decomp · pre z-denoise",
        :viridis, c_water_window, "g/mL")

    # Row 3 — z-direction-denoised outputs (this pair feeds §13 VMI synth)
    sub3 = "post z-denoise (radius = $(z_denoise_kwargs.radius))"
    panel!(3, 1, de_decomp.c_iodine[:, :, mid],
        "c_iodine", sub3, :viridis, c_iod_window, "mg/mL")
    panel!(3, 2, de_decomp.c_water[:, :, mid],
        "c_water", sub3, :viridis, c_water_window, "g/mL")

    fig
end

# ╔═╡ 03000014-0000-4000-8000-000000000020
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

# ╔═╡ 03000016-0000-4000-8000-000000000010
de_vmi_mono = let
    vols_in = [de_vmi_raw[E] for E in de_vmi_energies]
    σ_vec = Float64[2.0, 0.0, 2.0, 2.0]   # paired with de_vmi_energies
    # σ_vec = Float64[0.0, 0.0, 0.0, 0.0]   # paired with de_vmi_energies

    ws = BS.create_mono_plus_workspace(vols_in[1]; n_energies = length(de_vmi_energies))
    res = BS.apply_mono_plus!(
        ws, vols_in, de_vmi_energies;
        E_noise_opt = 70.0,
        σ_lp_px = σ_vec,
        verbose = true,
    )

    # FOV mask each VMI to -1024 HU outside the recon circle so out-of-FOV
    # voxels read as air everywhere downstream (display, ROI stats, table).
    # `apply_fov_mask!`'s `sentinel_μ` kwarg is unit-agnostic — pass -1024
    # directly because we're already in HU.
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

# ╔═╡ 03000016-0000-4000-8000-000000000020
let
    # Mean water HU per VMI energy — sanity check on the Mono+ output.
    # 8-px-radius circular ROI at the volume center, safely inside the
    # 50 mm Ca inner ring + 105 mm I outer ring, so the sample is pure
    # solid water.  After the corrections + decomp + Mono+ pipeline,
    # water should read ≈ 0 HU at every VMI energy by construction.
    nx = size(de_vmi_mono[de_vmi_energies[1]], 1)
    ny = size(de_vmi_mono[de_vmi_energies[1]], 2)
    nz = size(de_vmi_mono[de_vmi_energies[1]], 3)
    cx = nx / 2 + 0.5
    cy = ny / 2 + 0.5
    ROI_R = 8.0

    roi = CartesianIndex{2}[]
    r² = ROI_R^2
    i_lo = max(1, floor(Int, cx - ROI_R)); i_hi = min(nx, ceil(Int, cx + ROI_R))
    j_lo = max(1, floor(Int, cy - ROI_R)); j_hi = min(ny, ceil(Int, cy + ROI_R))
    for j in j_lo:j_hi, i in i_lo:i_hi
        ((i - cx)^2 + (j - cy)^2) ≤ r² && push!(roi, CartesianIndex(i, j))
    end

    function _mean_HU(vol)
        s = 0.0; n = 0
        for z in 1:nz, ci in roi
            s += vol[ci, z]; n += 1
        end
        s / n
    end

    energies = de_vmi_energies
    mean_HU  = [_mean_HU(de_vmi_mono[E]) for E in energies]

    fig = CM.Figure(size = (900, 540))
    ax = CM.Axis(
        fig[1, 1];
        title = "Mean water HU per VMI energy",
        subtitle = "Mono+ post-processed · 8-px center ROI · target = 0 HU",
        xlabel = "VMI energy (keV)", ylabel = "Mean HU",
        xticks = (1:length(energies), ["$(Int(E))" for E in energies]),
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )

    bar_colors = [
        CM.RGBf(0.85, 0.32, 0.20),   # 40 keV — warm
        CM.RGBf(0.92, 0.65, 0.20),
        CM.RGBf(0.45, 0.65, 0.85),
        CM.RGBf(0.20, 0.30, 0.65),   # 140 keV — cool
    ]

    CM.barplot!(
        ax, 1:length(energies), Float64.(mean_HU);
        color = bar_colors[1:length(energies)],
        strokecolor = :black, strokewidth = 1,
    )
    CM.hlines!(ax, [0.0]; color = :black, linestyle = :dash, linewidth = 2)

    for (i, hu) in enumerate(mean_HU)
        CM.text!(ax, i, hu;
            text = "$(round(hu, digits = 1)) HU",
            align = (:center, hu < 0 ? :top : :bottom),
            offset = (0, hu < 0 ? -4 : 4),
            fontsize = 16,
        )
    end

    fig
end

# ╔═╡ 03000017-0000-4000-8000-000000000010
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
            subtitle = i == 1 ? "Mono+ post-processed · 80/140 kVp · GE Apex Elite GSI" : "Mono+ post-processed",
            aspect = CM.DataAspect(),
            title_kwargs...,
        )
        hm = CM.heatmap!(
            ax, de_vmi_mono[E][:, :, mid];
            colormap = :grays, colorrange = colorrng
        )
        CM.hidedecorations!(ax)
        if i == 4
            hms = hm
        end
    end

    CM.Colorbar(fig[1:2, 3], hms; label = "HU", width = 14, labelsize = 18)

    CM.save(
        joinpath(@__DIR__, "..", "assets", "vmi_4panel.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 03000018-0000-4000-8000-000000000040
rod_data = let
    materials = phantom_cpu.materials

    # The Gammex 472 rods are z-invariant cylinders, so we work entirely
    # in 2-D using the phantom's mid-slice label pattern (in-plane
    # sampling matches between phantom and recon: both 512 px / 35 cm).
    mask_2d = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]

    # ROI strategy: a fixed 8-pixel-radius circular ROI at each rod's
    # centroid — staying well inside the 21-px-radius rod core (rod
    # radius = 1.4 cm; pixel size = 35 cm / 512 ≈ 0.683 mm, so rod
    # radius in pixels ≈ 20.5).
    #
    # Why not the full mask?  The rod-edge voxels are partial-volume
    # mixtures of iodine and surrounding solid water.  Mono+'s per-energy
    # low-pass filter blurs that mixture differently at each keV, which
    # injects an ENERGY-DEPENDENT bias into a full-mask mean — and
    # specifically inverts the iodine HU ordering between 70 and 100 keV
    # (because the 70 keV anchor is unfiltered while 100 keV is LP'd).
    # A small core ROI dodges all of that — same area for every rod,
    # consistent statistics, no edge mixing.  Verification nb 03 uses
    # the same trick.
    ROI_RADIUS_PX = 8

    # Per-label centroid in pixel coordinates (Float64 mean of indices).
    function rod_centroid(label::UInt8)
        idx = findall(==(label), mask_2d)
        isempty(idx) && error("rod_centroid: no voxels with label $label in mask_2d")
        cx = sum(ci -> Float64(ci[1]), idx) / length(idx)
        cy = sum(ci -> Float64(ci[2]), idx) / length(idx)
        return (cx, cy)
    end

    # Build per-rod 2-D circular ROI mask once (reused across all energies).
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

    rod_rois = Dict(
        lab => rod_roi_mask(lab)
            for lab in vcat(collect(ROD_LABELS.Ca), collect(ROD_LABELS.I))
    )

    # μ_water(E) at every VMI energy — a single Float64 each
    μ_water_E = Dict(
        E => BS.compute_μ_at_energy(BS.XA.Materials.water, E)
            for E in de_vmi_energies
    )

    # Theoretical HU for one rod material at energy E.
    function theoretical_hu(material, E::Float64)
        μ = BS.compute_μ_at_energy(material, E)
        return 1000.0 * (μ - μ_water_E[E]) / μ_water_E[E]
    end

    # Measured HU for one rod label: average over its 8-px circular core
    # ROI broadcast across every recon slice.
    function measured_hu(vmi_vol, label::UInt8)
        roi = rod_rois[label]
        n_z = size(vmi_vol, 3)
        s = 0.0
        n = 0
        for z in 1:n_z, ci in roi
            s += vmi_vol[ci, z]
            n += 1
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
            # `Phantom.materials` is a `Vector{XA.Material}` indexed by
            # `mask_value + 1` (NOT a Dict keyed by UInt8).  See the
            # `Phantom` doc-comment in `src/object/phantom.jl`.
            mat = materials[Int(lab) + 1]
            for (j, E) in pairs(de_vmi_energies)
                meas[i, j] = measured_hu(de_vmi_mono[E], lab)
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

# ╔═╡ 03000019-0000-4000-8000-000000000010
let
    fig = CM.Figure(size = (1180, 580))

    # Two panels share the same color scheme: warm for Ca, cool for I, with
    # rod-concentration index → color shade
    cmap_ca = CM.cgrad(:Oranges, 7; categorical = true)
    cmap_i = CM.cgrad(:GnBu, 7; categorical = true)

    panels = (
        (
            group = :Ca, title = "Calcium rods", subtitle = "50–600 mg/mL · Compton roll-off",
            cmap = cmap_ca, ylim = (0, 4200),
        ),
        (
            group = :I, title = "Iodine rods", subtitle = "2–20 mg/mL · K-edge boost at 40 keV",
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
            # Sample swatch for the rod-concentration legend (solid only —
            # the style legend below captures solid-vs-dashed).
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

# ╔═╡ 0300001a-0000-4000-8000-000000000010
let
    fig = CM.Figure(size = (1180, 620))

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

        # y = x identity reference — drawn FIRST so the per-energy fits
        # paint over it; bumped to a darker dashed line so the "perfect
        # agreement" reference is unmistakable against the per-energy fits.
        lim_lo = min(0.0, minimum(d.measured), minimum(d.theoretical))
        lim_hi = max(maximum(d.measured), maximum(d.theoretical)) * 1.05
        CM.lines!(
            ax, [lim_lo, lim_hi], [lim_lo, lim_hi];
            color = :black, linestyle = :dash, linewidth = 2,
            label = "y = x (unity)"
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
                color = color, linewidth = 2, label = label
            )
        end

        CM.axislegend(
            ax; position = :rb, framevisible = true,
            labelsize = 18, padding = (6, 6, 6, 6), rowgap = 1
        )
    end

    CM.save(
        joinpath(@__DIR__, "..", "assets", "vmi_regression.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 03000020-0000-4000-8000-000000000001
md"""
## 19. Quantitative agreement table

A compact table of (measured − theoretical) for every (rod, energy) pair.
Inside the K-edge–insensitive plateau (70–140 keV) we expect single-digit
HU agreement; at 40 keV iodine the K-edge boost amplifies any residual
calibration / spectral mismatch, so a few tens of HU is normal.
"""

# ╔═╡ 03000020-0000-4000-8000-000000000010
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

# ╔═╡ 03000021-0000-4000-8000-000000000001
md"""
## Summary

This notebook walked the full **image-domain dual-kVp VMI pipeline** end to
end on a simulated Gammex 472 — exactly the same flow used in the
`BasisSimulator.jl` clinical verification suite for GE Apex Elite GSI:

- **Two `let ... end` simulation blocks** — one per kVp, each ending with
  `ws = nothing; GC.gc(true)` so GPU buffers actually come back between
  acquisitions.
- **No BHC on the DE path** — the spectrum stays polychromatic so the
  Ding calibration's coefficients (LSQ-fit on our own pre-RSKR rod HU)
  absorb beam-hardening implicitly.  Per-kVp `μ_water` is still computed
  from each spectrum's mean energy for HU conversion.
- **RSKR-2ch joint denoising in μ-domain** — the joint
  iodine/water statistics drive the SVD + bilateral filter, preserving
  anti-correlated noise structure between basis volumes.
- **Image-domain Ding decomposition — self-calibrated** from pre-RSKR
  rod HU on our own simulated phantom (one solid-water reference + 7
  iodine concentrations).  Replaces the clinical
  `BS.GE_REVOLUTION_APEX_ELITE_DE_CAL` constants, which under-shoot the
  simulator's polychromatic HU baseline.
- **Mono+ FBP-equivalent VMI sweep** at 40 / 70 / 100 / 140 keV, with
  per-energy LP shaping anchored at 70 keV.
- **Per-rod verification against XrayAttenuation theory** — measured VMI HU
  vs `BS.compute_μ_at_energy` per rod material; with self-calibration the
  per-energy regression slopes collapse to ≈ 1 across both Ca and I rods.

The same image-domain pipeline can be repointed at any other dual-kVp
acquisition (e.g. Siemens Naeotom Alpha) by re-running the
self-calibration cell against that scanner's rod HU.
"""

# ╔═╡ Cell order:
# ╟─03000001-0000-4000-8000-000000000010
# ╟─03000001-0000-4000-8000-000000000020
# ╠═03000001-0000-4000-8000-000000000001
# ╠═03000001-0000-4000-8000-000000000002
# ╠═03000001-0000-4000-8000-000000000003
# ╠═03000001-0000-4000-8000-000000000030
# ╠═03000001-0000-4000-8000-000000000031
# ╠═03000001-0000-4000-8000-000000000032
# ╠═03000001-0000-4000-8000-000000000040
# ╟─03000001-0000-4000-8000-000000000050
# ╟─03000002-0000-4000-8000-000000000001
# ╠═03000002-0000-4000-8000-000000000010
# ╠═03000002-0000-4000-8000-000000000020
# ╟─03000003-0000-4000-8000-000000000001
# ╠═03000003-0000-4000-8000-000000000010
# ╟─03000004-0000-4000-8000-000000000001
# ╠═03000004-0000-4000-8000-000000000010
# ╠═03000004-0000-4000-8000-000000000020
# ╟─03000005-0000-4000-8000-000000000001
# ╠═03000005-0000-4000-8000-000000000010
# ╠═03000005-0000-4000-8000-000000000020
# ╟─03000007-0000-4000-8000-000000000001
# ╠═03000007-0000-4000-8000-000000000010
# ╟─03000008-0000-4000-8000-000000000001
# ╠═03000008-0000-4000-8000-000000000010
# ╟─03000009-0000-4000-8000-000000000001
# ╠═03000009-0000-4000-8000-000000000003
# ╟─03000009-0000-4000-8000-000000000004
# ╠═03000009-0000-4000-8000-000000000007
# ╟─03000009-0000-4000-8000-000000000008
# ╠═03000009-0000-4000-8000-000000000010
# ╠═03000009-0000-4000-8000-000000000025
# ╟─03000010-0000-4000-8000-000000000001
# ╠═03000010-0000-4000-8000-000000000010
# ╟─03000009-0000-4000-8000-000000000015
# ╠═03000009-0000-4000-8000-000000000017
# ╠═03000009-0000-4000-8000-000000000020
# ╟─03000009-0000-4000-8000-000000000030
# ╟─03000006-0000-4000-8000-000000000001
# ╠═03000006-0000-4000-8000-000000000010
# ╟─03000006-0000-4000-8000-000000000020
# ╟─03000011-0000-4000-8000-000000000001
# ╠═03000011-0000-4000-8000-000000000010
# ╟─03000012-0000-4000-8000-000000000001
# ╠═03000012-0000-4000-8000-000000000005
# ╠═03000012-0000-4000-8000-000000000010
# ╟─03000012-0000-4000-8000-000000000020
# ╟─03000012-0000-4000-8000-000000000030
# ╠═03000012-0000-4000-8000-000000000040
# ╟─03000012-0000-4000-8000-000000000050
# ╟─03000013-0000-4000-8000-000000000001
# ╠═03000013-0000-4000-8000-000000000005
# ╠═03000013-0000-4000-8000-000000000010
# ╟─03000013-0000-4000-8000-000000000030
# ╟─03000014-0000-4000-8000-000000000001
# ╠═03000014-0000-4000-8000-000000000010
# ╠═03000014-0000-4000-8000-000000000020
# ╟─03000016-0000-4000-8000-000000000001
# ╠═03000016-0000-4000-8000-000000000010
# ╟─03000016-0000-4000-8000-000000000020
# ╟─03000017-0000-4000-8000-000000000001
# ╟─03000017-0000-4000-8000-000000000010
# ╟─03000018-0000-4000-8000-000000000001
# ╠═03000018-0000-4000-8000-000000000010
# ╠═03000018-0000-4000-8000-000000000020
# ╟─03000018-0000-4000-8000-000000000030
# ╠═03000018-0000-4000-8000-000000000040
# ╟─03000019-0000-4000-8000-000000000001
# ╟─03000019-0000-4000-8000-000000000010
# ╟─0300001a-0000-4000-8000-000000000001
# ╠═bc0b9af2-d4d0-44c5-8f93-62dc9ca5d6bd
# ╟─0300001a-0000-4000-8000-000000000010
# ╟─03000020-0000-4000-8000-000000000001
# ╟─03000020-0000-4000-8000-000000000010
# ╟─03000021-0000-4000-8000-000000000001
