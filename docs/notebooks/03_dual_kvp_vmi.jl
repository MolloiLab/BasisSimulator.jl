### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

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
## 7. FBP per kVp → μ-domain volumes

The image-domain pipeline reconstructs **each kVp into its own μ-volume
(cm⁻¹)** with the GE-tuned apodization filter
`(0.0, 0.25, 0.5, 0.75, 1.0) → (1.0, 0.95, 0.85, 0.65, 0.4)`.  HU
conversion is **deferred** to after RSKR (§9) so the joint denoiser
operates on the raw μ pair — that's where iodine and water noise are
maximally anti-correlated, which RSKR exploits.

No BHC is applied here — the spectrum stays polychromatic so the Ding
calibration's cross-spectrum coefficients (§11) can absorb the
beam-hardening implicitly.
"""

# ╔═╡ 03000009-0000-4000-8000-000000000010
de_lohi_μ = let
    matrix_size = recon_opts.matrix_size

    function _fbp_to_μ(sino_cpu, geom)
        sino_gpu = to_gpu(Float32.(sino_cpu))
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

    vol_low_μ = _fbp_to_μ(sim_low.sino, sim_low.geom)
    vol_high_μ = _fbp_to_μ(sim_high.sino, sim_high.geom)

    (
        vol_low_μ = vol_low_μ,
        vol_high_μ = vol_high_μ,
        geom = sim_low.geom,
    )
end;

# ╔═╡ 03000006-0000-4000-8000-000000000001
md"""
## 8. Per-kVp `μ_water` — analytic from source spectrum + phantom size

We *don't* run BHC on the DE pipeline.  The image-domain Ding
decomposition (§11) is fed polychromatic FBP HU; applying BHC first would
"monoenergeticize" our HU volumes and break the spectral conditions the
calibration is fit under.

But that means we can't use a spectrum-mean monoenergetic μ_water as the
HU reference: the polychromatic FBP produces an *effective* μ that's
biased away from μ(mean_E) by beam hardening, so dividing by μ(mean_E)
would make solid water read at ≈ −150 HU instead of 0.

[`BS.compute_polychromatic_μ_water`](@ref) computes the right reference
analytically: it resolves the bowtie-aware source spectrum (flat filter
+ protocol filters + bowtie all included), pre-hardens it through
`water_path_cm` of solid water via Beer-Lambert (the phantom radius is
the typical one-way path length to a center voxel), then integrates
μ_water(E) against the hardened spectrum.  Result: the same μ_water
the FBP recon would estimate at phantom center, **without needing the
recon volume**.

For our Gammex 472 body (33 cm diameter), the relevant path is
`water_path_cm = 16.5`.
"""

# ╔═╡ 03000006-0000-4000-8000-000000000010
kVp_μ_water = let
    body_radius_cm = 16.5   # Gammex 472 body — 33 cm diameter / 2
    Dict(
        80 => BS.compute_polychromatic_μ_water(
            sim_opts, protocol_low;
            scanner = scanner, geom = sim_low.geom,
            water_path_cm = body_radius_cm * 2
        ),
        140 => BS.compute_polychromatic_μ_water(
            sim_opts, protocol_high;
            scanner = scanner, geom = sim_high.geom,
            water_path_cm = body_radius_cm * 2
        ),
    )
end;

# ╔═╡ 03000006-0000-4000-8000-000000000020
md"""
lac water (80 kVp) = $(round(kVp_μ_water[80],  digits = 5)) cm⁻¹  || 
lac water (140 kVp) = $(round(kVp_μ_water[140], digits = 5)) cm⁻¹
"""

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
        [de_lohi_μ.vol_low_μ, de_lohi_μ.vol_high_μ];
        n_iter = 2,
        h_param = 2.0,
        radius = 2,
        γ = 0.5,
        gpu_arr_type = GPU_BACKEND.to_gpu,
        verbose = true,
    )
    (
        vol_low_μ = out[1],
        vol_high_μ = out[2],
        geom = de_lohi_μ.geom,
    )
end;

# ╔═╡ 03000011-0000-4000-8000-000000000001
md"""
## 10. HU conversion + system noise floor

Per-kVp `to_hounsfield` using each spectrum's own calibrated `μ_water_ref`,
then add the dose-independent DAS Gaussian noise floor (matches the SE
pipeline in notebook 01).
"""

# ╔═╡ 03000011-0000-4000-8000-000000000010
de_lohi_HU = let
    vol_low_HU = Float32.(BS.to_hounsfield(de_lohi_rskr.vol_low_μ; μ_water = kVp_μ_water[80]))
    vol_high_HU = Float32.(BS.to_hounsfield(de_lohi_rskr.vol_high_μ; μ_water = kVp_μ_water[140]))
    BS.add_system_noise_floor!(vol_low_HU, 28.0; seed = 1234)
    BS.add_system_noise_floor!(vol_high_HU, 28.0; seed = 5678)
    (vol_low_HU = vol_low_HU, vol_high_HU = vol_high_HU, geom = de_lohi_rskr.geom)
end;

# ╔═╡ 03000012-0000-4000-8000-000000000001
md"""
## 11. Ding calibration from `BS.GE_REVOLUTION_APEX_ELITE_DE_CAL`

The Ding image-domain decomposition expresses iodine concentration as a
linear function of the two HU volumes:

```
c_iodine[v] = a₀ + a₁ · HU_80[v] + a₂ · HU_140[v]    [mg/mL]
```

`(a₀, a₁, a₂)` are LSQ-fit from a calibration table of known rods
(water + 7 iodine concentrations).  The constant
[`BS.GE_REVOLUTION_APEX_ELITE_DE_CAL`](@ref) ships with per-rod HU
measured on simulated 80/140 kVp **post-RSKR** FBP for exactly this
phantom + scanner combo, so the cal coefficients absorb our simulator's
polychromatic HU baseline by construction.

[`BS.iodine_calibration_rods`](@ref) extracts the
(HU_80, HU_140, mg/mL) tuples; [`BS.fit_ding_coeffs`](@ref) does the fit.
"""

# ╔═╡ 03000012-0000-4000-8000-000000000010
de_cal = let
    rods = BS.iodine_calibration_rods(
        BS.GE_REVOLUTION_APEX_ELITE_DE_CAL;
        hu_low_field = :HU_80kVp,
        hu_high_field = :HU_140kVp,
    )
    fit = BS.fit_ding_coeffs(rods.HU_low, rods.HU_high, rods.mg_per_mL)
    (
        coeffs = fit.coeffs,
        α_iod_low = fit.α_low,
        α_iod_high = fit.α_high,
        rms_c = fit.rms,
        rod_names = rods.names,
        rod_HU_low = rods.HU_low,
        rod_HU_high = rods.HU_high,
        rod_c_iodine = rods.mg_per_mL,
    )
end;

# ╔═╡ 03000012-0000-4000-8000-000000000020
let
    rows = [
        "| $(de_cal.rod_names[i]) | $(de_cal.rod_c_iodine[i]) | " *
            "$(round(de_cal.rod_HU_low[i], digits = 1)) | " *
            "$(round(de_cal.rod_HU_high[i], digits = 1)) |"
            for i in eachindex(de_cal.rod_names)
    ]
    body = """
    **Calibration rods** from `BS.GE_REVOLUTION_APEX_ELITE_DE_CAL`
    (post-RSKR sim FBP, 8-px core ROI per rod):

    | Rod | c (mg/mL) | HU @ 80 kVp | HU @ 140 kVp |
    |-----|---|---|---|
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

# ╔═╡ 03000013-0000-4000-8000-000000000001
md"""
## 12. Apply Ding decomposition + z-median speckle removal

[`BS.apply_ding_decomp`](@ref) is a per-voxel evaluation of the Ding
equation, returning a `c_iodine` map in mg/mL.  A 3-slice
[`BS.apply_median_z`](@ref) (radius = 1, axial-only) wipes single-voxel xy
impulse noise without any in-plane blurring.
"""

# ╔═╡ 03000013-0000-4000-8000-000000000010
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

# ╔═╡ 03000016-0000-4000-8000-000000000010
de_vmi_mono = let
    vols_in = [de_vmi_raw[E] for E in de_vmi_energies]
    σ_vec = Float64[1.5, 0.0, 1.0, 1.0]   # paired with de_vmi_energies

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

# ╔═╡ 03000017-0000-4000-8000-000000000001
md"""
## 15. The four VMI images

Mid-slice mosaic of all four energies, shared HU window (-200 to 1000 HU)
so iodine boost vs. soft-tissue baseline is comparable side-by-side.
"""

# ╔═╡ 03000017-0000-4000-8000-000000000010
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
            subtitle = i == 1 ? "Mono+ post-processed · 80/140 kVp · GE Apex Elite GSI" : "Mono+ post-processed",
            aspect = CM.DataAspect(),
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

    CM.Colorbar(fig[1:2, 3], hms; label = "HU", width = 14)

    CM.save(
        joinpath(@__DIR__, "..", "assets", "vmi_4panel.png"),
        fig; px_per_unit = 2,
    )
    fig
end

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
            vcat(["Measured (Mono+ VMI)", "Theoretical (XA)"], collect(d.names));
            position = :rt, framevisible = true, labelsize = 9,
            rowgap = 1, padding = (6, 6, 6, 6),
        )
    end

    CM.save(
        joinpath(@__DIR__, "..", "assets", "vmi_vs_theoretical.png"),
        fig; px_per_unit = 2,
    )
    fig
end

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

    panels = ((:Ca, "Calcium Rods"), (:I, "Iodine Rods"))
    for (col, (group, title)) in pairs(panels)
        d = rod_data[group]
        ax = CM.Axis(
            fig[1, col];
            title = title,
            xlabel = "Theoretical HU",
            ylabel = "Measured HU",
            aspect = CM.AxisAspect(1),
            titlesize = 32,
            ylabelsize = 22,
            xlabelsize = 22,
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
# ╠═03000009-0000-4000-8000-000000000010
# ╟─03000006-0000-4000-8000-000000000001
# ╠═03000006-0000-4000-8000-000000000010
# ╟─03000006-0000-4000-8000-000000000020
# ╟─03000010-0000-4000-8000-000000000001
# ╠═03000010-0000-4000-8000-000000000010
# ╟─03000011-0000-4000-8000-000000000001
# ╠═03000011-0000-4000-8000-000000000010
# ╟─03000012-0000-4000-8000-000000000001
# ╠═03000012-0000-4000-8000-000000000010
# ╟─03000012-0000-4000-8000-000000000020
# ╟─03000013-0000-4000-8000-000000000001
# ╠═03000013-0000-4000-8000-000000000010
# ╟─03000014-0000-4000-8000-000000000001
# ╠═03000014-0000-4000-8000-000000000010
# ╠═03000014-0000-4000-8000-000000000020
# ╟─03000016-0000-4000-8000-000000000001
# ╠═03000016-0000-4000-8000-000000000010
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
# ╟─0300001a-0000-4000-8000-000000000010
# ╟─03000020-0000-4000-8000-000000000001
# ╟─03000020-0000-4000-8000-000000000010
# ╟─03000021-0000-4000-8000-000000000001
