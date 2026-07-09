### A Pluto.jl notebook ###
# v0.2.1

using Markdown
using InteractiveUtils

# ╔═╡ 08010003-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 08010003-0000-4000-8000-000000000005
using Markdown: @md_str, Markdown

# ╔═╡ 08010003-0000-4000-8000-000000000006
using Statistics: mean, std, quantile

# ╔═╡ 08010003-0000-4000-8000-000000000007
using Unitful: @u_str

# ╔═╡ 08010003-0000-4000-8000-000000000008
using Random: MersenneTwister, randn!

# ╔═╡ 08010001-0000-4000-8000-000000000001
md"""
# QRM-Thorax Pure-Material PCCT VMI: Full-Resolution True-Scan Reference

**Intent:** 1-to-1 PCCT counterpart of notebook 07.  Same QRM-Thorax
phantom with four pure-material rods (water · triolein · collagen ·
iodine), same full-fidelity recon grid, same per-rod measured-vs-
theoretical VMI regression — but the acquisition is a single 140 kVp
photon-counting scan with four energy bins, bin-combined to a
`(low, high)` pair and run through the SF-JSD joint sinogram denoiser
before the Cong PCCT-Φ_k decomposition.

| Stage         | Matrix                   | Voxel (mm)             | Extent                        |
|---------------|--------------------------|------------------------|-------------------------------|
| GT phantom    | **1600 × 1100 × 20**     | **0.2 isotropic**      | 320 × 220 × 4 mm              |
| Recon         | 512 × 512 × **3**        | **0.625 isotropic**    | FOV 32 cm × 1.875 mm z        |
| Collimation   | **2.5 mm at iso**        | —                      | ~7 detector rows lit          |
| Scanner       | Siemens Naeotom Alpha    | 0.353 × 0.302 mm pixels | 144 × ~1192 detector          |
| Protocol      | PCCT, 4 bins             | 140 kVp · 174 mA · 1200 views · 0.5 s rot. | clinical PCCT dose |

```
QRM-Thorax mid-slice mask  → relabel rods 9–12 → tile z → BS.Phantom
                                       │
                            Simulate 140 kVp PCCT  (4 bins)
                                       │
                       Per-Bin Pile-up + Scatter Correction
                                       │
                          Bin Combine  (1+2 → low,  3+4 → high)
                                       │
                       Material Decomp → FBP × 2 → z-Median →
                       2-basis VMI →
                                       │
                       Per-Rod Measured-vs-Theoretical Regression
                                       at 40 / 70 / 100 / 140 keV
```

!!! tip "When to reach for this notebook"
    Use 08 when you want **PCCT clinical-realism evidence** — VMI
    accuracy on a body-sized phantom with a body-sized FOV at the
    resolutions a real Siemens Naeotom Alpha would deliver.  For the
    dual-kVp counterpart on the same phantom / rods, see notebook 07.

!!! info "Why pure end-members?"
    `XA.Materials.basis_fat` is ICRU-44 adipose tissue (≈83 % triolein
    + 17 % water + trace electrolytes) — fine as a generic fat
    reference, but it's not a *mathematically pure* lipid.  The new
    `XA.Materials.basis_lipid` (H/C/O at 0.92 g/cm³) and
    `basis_collagen` (H/C/N/O at 1.26 g/cm³) end members were added in
    XrayAttenuation 0.3.0 specifically for clean water/lipid/collagen
    decomposition validation.

!!! success "References"
    - Cong, De Man, Wang (2022), *J X-Ray Sci Technol* — projection-
      domain univariate solver (originally dual-kVp DECT).
    - Black (*in prep.*) — generalization of Cong 2022 to PCCT /
      split-spectrum via an effective spectral response Φ_k(ε) ≥ 0.
    - Grant et al. (2014) — Mono+ frequency-split rule.
"""

# ╔═╡ 08010002-0000-4000-8000-000000000001
md"""
## Notebook Setup
"""

# ╔═╡ 08010003-0000-4000-8000-000000000010
import BasisSimulator as BS

# ╔═╡ 08010003-0000-4000-8000-000000000011
import CairoMakie as CM

# ╔═╡ 08010003-0000-4000-8000-000000000012
import PlutoUI

# ╔═╡ 08010003-0000-4000-8000-000000000013
PlutoUI.TableOfContents()

# ╔═╡ 08010003-0000-4000-8000-000000000040
begin
    import GPUSelect
    AT = GPUSelect.Storage()     # the backend array type, directly: MtlArray / CuArray / ROCArray
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end

# ╔═╡ 08010003-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 08020000-0000-4000-8000-0000000000f1
md"""
## Scan Setup and Simulation

One clinical PCCT acquisition: the QRM-Thorax phantom, the Siemens Naeotom
Alpha photon-counting scanner, the 140 kVp protocol, and the 4-bin forward
projection.
"""

# ╔═╡ 08020001-0000-4000-8000-000000000001
md"""
### 01. `Phantom`: QRM-Thorax with 4 Pure-Material Rods

Identical to notebook 07.  Read the **prepared** QRM-Thorax mask —
already rotated to CT display orientation (spine at the bottom) and
2× downsampled, cached at
`docs/notebooks/data/qrm_thorax/qrm_thorax_1600x1100_rot_uint8.raw`.
Phantom shape after z-tiling: **1600 × 1100 × 20** at **0.2 mm
isotropic** (physical extent 320 × 220 × 4 mm; body envelope ≈ 30 ×
20 cm matches QRM-Thorax-small spec).

Source mask labels:

| Label | Material              |  | Label | Material               |
|-------|-----------------------|--|-------|------------------------|
| 1     | air                   |  | 7     | air rod                |
| 2     | lung                  |  | 8     | heart                  |
| 3     | soft tissue           |  | **9** | rod 1 → water          |
| 4     | bone                  |  | **10**| rod 2 → triolein       |
| 5     | bone marrow           |  | **11**| rod 3 → collagen       |
| 6     | water rod (lung)      |  | **12**| rod 4 → gammex\\_472\\_i5\\_0 |

Rod inserts 9–12 in the source were 4 × Ca-HA concentrations
(50 / 100 / 200 / 400 mg/mL).  We **relabel** them to four pure-
material end members below — same rod geometry, replaced
compositions.
"""

# ╔═╡ 08020001-0000-4000-8000-000000000010
const QRM_CACHE_PATH = joinpath(@__DIR__, "data", "qrm_thorax", "qrm_thorax_1600x1100_rot_uint8.raw");

# ╔═╡ 08020001-0000-4000-8000-000000000011
const QRM_TARGET_NX = 1600;  # full prepared cache, no extra in-plane downsample

# ╔═╡ 08020001-0000-4000-8000-000000000012
const QRM_TARGET_NY = 1100;  # full prepared cache, no extra in-plane downsample

# ╔═╡ 08020001-0000-4000-8000-000000000013
const QRM_TARGET_NZ = 20;    # 20 × 0.2 mm = 4 mm — minimum phantom z that covers all rays for 2.5 mm collimation

# ╔═╡ 08020001-0000-4000-8000-000000000014
const QRM_VOXEL_SIZE_CM = (0.02, 0.02, 0.02);   # (x, y, z) cm — 0.2 mm isotropic ground truth (320 × 220 × 4 mm physical extent)

# ╔═╡ 08020001-0000-4000-8000-000000000015
const QRM_SHARED_DRIVE_DIR = "/Volumes/Molloilab/Shu Nie/water-lipid";

# ╔═╡ 08020001-0000-4000-8000-000000000016
const QRM_SHARED_FULL_PATH = joinpath(QRM_SHARED_DRIVE_DIR, "qrm_thorax_3200x2200_rot_uint8.raw");

# ╔═╡ 08020001-0000-4000-8000-000000000017
const QRM_SHARED_DOWN_PATH = joinpath(QRM_SHARED_DRIVE_DIR, "qrm_thorax_1600x1100_rot_uint8.raw");

# ╔═╡ 08020001-0000-4000-8000-000000000018
md"""
!!! success "Share-drive (lab volume) copies of the QRM-Thorax phantom"
    Both rotated phantom files are mirrored on the lab volume so any
    lab member can pick them up without rerunning prep.

    | resolution                        | path                                                  |
    |-----------------------------------|-------------------------------------------------------|
    | full-resolution (3200 × 2200, ~7 MB) | `$(QRM_SHARED_FULL_PATH)` |
    | 2× downsampled (1600 × 1100, ~1.7 MB) ← used here | `$(QRM_SHARED_DOWN_PATH)` |

    If you don't have the local cache yet:
    ```sh
    cp "$(QRM_SHARED_DOWN_PATH)" "$(QRM_CACHE_PATH)"
    ```
"""

# ╔═╡ 08020001-0000-4000-8000-000000000020
mask_3d_raw = let
    isfile(QRM_CACHE_PATH) || error(
        "QRM-Thorax cache not found at $(QRM_CACHE_PATH).\n" *
            "Either:\n" *
            "  • copy the prepared cache from the lab volume:\n" *
            "      cp \"$(QRM_SHARED_DOWN_PATH)\" \"$(QRM_CACHE_PATH)\"\n" *
            "  • or run the prep notebook once to rebuild the cache from source."
    )
    cache_2d = reshape(read(QRM_CACHE_PATH), QRM_TARGET_NX, QRM_TARGET_NY)
    repeat(cache_2d; outer = (1, 1, QRM_TARGET_NZ))
end;

# ╔═╡ 08020001-0000-4000-8000-000000000022
let
    mid = size(mask_3d_raw, 3) ÷ 2 + 1
    slice = view(mask_3d_raw, :, :, mid)
    body = findall(!=(UInt8(1)), slice)

    if isempty(body)
        md"""
        !!! warning "Phantom probe — empty mask"
            No non-air voxels found in the loaded mid-slice.  The mask may be all air.
        """
    else
        i_lo, i_hi = extrema(c -> c[1], body)
        j_lo, j_hi = extrema(c -> c[2], body)

        px_mm = QRM_VOXEL_SIZE_CM[1] * 10
        body_w_mm = (i_hi - i_lo + 1) * px_mm
        body_h_mm = (j_hi - j_lo + 1) * px_mm
        frame_w_mm = size(slice, 1) * px_mm
        frame_h_mm = size(slice, 2) * px_mm

        md"""
        !!! note "QRM-Thorax body-envelope probe"
            Mid-slice bounding box of non-air voxels (label ≠ 1):

            | metric         | value |
            |----------------|-------|
            | body bbox      | $(round(body_w_mm; digits=1)) × $(round(body_h_mm; digits=1)) mm  ($(round(body_w_mm/10; digits=1)) × $(round(body_h_mm/10; digits=1)) cm) |
            | body fills     | $(round(100 * body_w_mm/frame_w_mm; digits=1)) % × $(round(100 * body_h_mm/frame_h_mm; digits=1)) % of frame |
            | frame          | $(round(frame_w_mm; digits=1)) × $(round(frame_h_mm; digits=1)) mm  ($(size(slice, 1)) × $(size(slice, 2)) voxels @ $(px_mm) mm/voxel) |
            | reference spec | QRM-Thorax small (no fat ring) ≈ 28–30 × 18–22 cm |
        """
    end
end

# ╔═╡ 08020001-0000-4000-8000-000000000025
md"""
**Bore 4 rod inserts** into the heart cavity at cardinal positions
(N / E / S / W) and assign them new labels 9–12 to match
`materials_dict`.

| Direction | Label | Material              |
|-----------|-------|-----------------------|
| North     | 9     | `basis_water`         |
| East      | 10    | `basis_lipid`         |
| South     | 11    | `basis_collagen`      |
| West      | 12    | `gammex_472_i5_0`     |
"""

# ╔═╡ 08020001-0000-4000-8000-000000000026
const ROD_HEART_CENTER_PX = (800, 654);   # ≈ (50%, 59%) of the 1600 × 1100 frame

# ╔═╡ 08020001-0000-4000-8000-000000000027
const ROD_RADIUS_MM = 7.5;                  # mm — each rod radius (physical)

# ╔═╡ 08020001-0000-4000-8000-000000000028
const ROD_OFFSET_MM = 25.0;                  # mm — heart-center → rod-center distance (physical)

# ╔═╡ 08020001-0000-4000-8000-000000000029
mask_3d = let
    out = copy(mask_3d_raw)
    nx, ny, nz = size(out)

    px_mm = QRM_VOXEL_SIZE_CM[1] * 10
    cx_px, cy_px = ROD_HEART_CENTER_PX
    r_px = ROD_RADIUS_MM / px_mm
    o_px = ROD_OFFSET_MM / px_mm

    rod_specs = (
        (0.0, +o_px, UInt8(9)),    # North → basis_water
        (+o_px, 0.0, UInt8(10)),   # East  → basis_lipid
        (0.0, -o_px, UInt8(11)),   # South → basis_collagen
        (-o_px, 0.0, UInt8(12)),   # West  → gammex_472_i5_0
    )

    for (dx, dy, lab) in rod_specs
        rx, ry = cx_px + dx, cy_px + dy
        i_lo = max(1, floor(Int, rx - r_px))
        i_hi = min(nx, ceil(Int, rx + r_px))
        j_lo = max(1, floor(Int, ry - r_px))
        j_hi = min(ny, ceil(Int, ry + r_px))
        @inbounds for j in j_lo:j_hi, i in i_lo:i_hi
            ((i - rx)^2 + (j - ry)^2) <= r_px^2 || continue
            for k in 1:nz
                out[i, j, k] = lab
            end
        end
    end

    out
end;

# ╔═╡ 08020001-0000-4000-8000-000000000040
materials_dict = Dict{Int, BS.XA.Material}(
    # Anatomy
    1 => BS.XA.Materials.air,
    2 => BS.XA.Materials.lung,
    3 => BS.XA.Materials.muscle,
    4 => BS.XA.Materials.corticalbone,
    5 => BS.XA.Materials.marrow_red,
    6 => BS.XA.Materials.water,
    7 => BS.XA.Materials.air,
    8 => BS.XA.Materials.water,
    # Pure-material rod inserts (relabeled from Ca-HA)
    9 => BS.XA.Materials.basis_water,
    10 => BS.XA.Materials.basis_lipid,
    11 => BS.XA.Materials.basis_collagen,
    12 => BS.XA.Materials.gammex_472_i5_0,
);

# ╔═╡ 08020001-0000-4000-8000-000000000050
phantom_cpu = BS.create_phantom_from_mask(
    Array{Int, 3}(mask_3d),
    materials_dict,
    QRM_VOXEL_SIZE_CM,
);

# ╔═╡ 08020001-0000-4000-8000-000000000051
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 08020001-0000-4000-8000-000000000060
QRM_LABEL_NAMES = Dict{Int, String}(
    1 => "1 air",
    2 => "2 lung",
    3 => "3 muscle / soft tissue",
    4 => "4 cortical bone",
    5 => "5 marrow",
    6 => "6 water rod (lung-side)",
    7 => "7 air rod",
    8 => "8 heart (water-filled)",
    9 => "9 → basis_water rod",
    10 => "10 → basis_lipid rod",
    11 => "11 → basis_collagen rod",
    12 => "12 → gammex_472_i5_0 rod",
);

# ╔═╡ 08020001-0000-4000-8000-000000000061
let
    mid = size(phantom_cpu.mask, 3) ÷ 2 + 1
    slice = phantom_cpu.mask[:, :, mid]
    n_lbl = length(unique(slice))

    fig = CM.Figure(size = (1100, 760))
    ax = CM.Axis(
        fig[1, 1];
        title = "QRM-Thorax mid-slice (labeled)",
        subtitle = "$(size(slice, 1)) × $(size(slice, 2)) at $(QRM_VOXEL_SIZE_CM[1] * 10) mm/voxel · $(n_lbl) unique labels",
        aspect = CM.DataAspect(),
        titlesize = 28, subtitlesize = 18,
    )
    hm = CM.heatmap!(ax, Float32.(slice); colormap = :tab20)
    CM.hidedecorations!(ax)
    CM.Colorbar(fig[1, 2], hm; label = "label", width = 14, labelsize = 18)
    fig
end

# ╔═╡ 08020001-0000-4000-8000-000000000062
let
    mid = size(phantom_cpu.mask, 3) ÷ 2 + 1
    slice = phantom_cpu.mask[:, :, mid]
    labels_present = sort(Int.(unique(slice)))

    rows = String[]
    for lab in labels_present
        name = get(QRM_LABEL_NAMES, lab, "(unnamed)")
        push!(rows, "| $(lab) | $(name) |")
    end

    Markdown.parse(
        "**Labels present in the loaded mid-slice:**\n\n" *
            "| label | name |\n|-------|------|\n" *
            join(rows, "\n")
    )
end

# ╔═╡ 08030001-0000-4000-8000-000000000001
md"""
### 02. `Scanner`: Siemens Naeotom Alpha (PCCT, 4-threshold)

CdTe direct-conversion detector with native dexels 0.275 × 0.322 mm at
the detector face (2×2 binned in DAS).  Energy thresholds
`T = [20, 35, 55, 70] keV` define 4 bins:

| Bin | Range (keV) |
|-----|-------------|
| 1   | 20 – 35     |
| 2   | 35 – 55     |
| 3   | 55 – 70     |
| 4   | > 70        |

No bowtie filter on the Naeotom Alpha (uses the Vectron tube's inherent
0.9 mm titanium window stacked on top of the 3 mm Al flat filter).
"""

# ╔═╡ 08030001-0000-4000-8000-000000000010
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
        # No bowtie — the physical Naeotom Alpha config (matches nb04).  The
        # decomposition uses the 1-D applied-W spectrum, so a per-pixel bowtie
        # would need the per-pixel-ŵ path to stay consistent; :none keeps the
        # setup coherent with nb04's certified chain.
        bowtie_filter = :none,

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

# ╔═╡ 08030002-0000-4000-8000-000000000001
md"""
### 03. `CTProtocol`: 140 kVp / 174 mA / 2.5 mm collimation

Clinical 140 kVp single-energy PCCT acquisition.
`additional_filters = [("Ti", 0.9)]` is the Vectron tube's inherent
0.9 mm titanium window on top of the 3 mm Al flat filter.

**Collimation = 2.5 mm at iso** matches notebook 07 so the 3-slice
recon at 0.625 mm fits the usable-z budget of the same QRM-Thorax
phantom.
"""

# ╔═╡ 08030002-0000-4000-8000-000000000010
protocol = BS.CTProtocol(
    kVp = 140,
    mA = 174.0,    # clinical PCCT dose (matches header + canonical nb04); was 5.0
    views = 1200,
    rotation_time = 0.5,
    collimation_mm = 5.0,    # matches nb04 scan approach (recon z-extent unchanged)
    additional_filters = [("Ti", 0.9)],
);

# ╔═╡ 08030003-0000-4000-8000-000000000001
md"""
### 04. `SimOptions` and `ReconOptions`

`fidelity = :pcct` switches the simulator into the photon-counting path
(per-bin sinograms + DRM + Compton scatter modeling + MC pile-up).
"""

# ╔═╡ 08030003-0000-4000-8000-000000000010
sim_opts = BS.SimOptions(
    fidelity = :pcct,
    seed = 1234,
    projector = :dd_fast,  # same anti-aliased DD physics, single-pass fused kernels (~47× faster poly)

    # ─── Inert for PCCT (flag exists but does nothing) ───
    use_fill_factor = false,
    use_detector_efficiency = false,
    use_optical_crosstalk = false,
    use_focal_spot = false,
    use_lag = false,
    use_heel_effect = false,

    # ─── Active for PCCT ───
    use_scatter = false,                  # EICT scatter flag — OFF (PCCT uses use_pcct_scatter)
    use_noise = true,                     # quantum noise inside simulate!()  (src :count, nr below)
    use_pcct_scatter = true,              # ← PCCT scatter injection, inside simulate!()
    use_pcct_scatter_correction = true,   # ← PCCT model-based scatter correction, inside simulate!()
    use_pcct_pileup = true,               # ← PCCT pileup forward, inside simulate!()
    use_pcct_pileup_correction = true,    # ← PCCT pileup correction (inverse S), inside simulate!()
    # DETECTOR-LEVEL CORRECTION SURROGATE — NOT a recon-level (QIR) stand-in
    # (chain is pure FBP; accuracy is independent of this knob).  Stands in
    # for the vendor's detector-side algorithms (anti-coincidence /
    # charge-sharing event reconstruction, count-rate linearization,
    # threshold compensation) whose degradations we Monte-Carlo simulate
    # but whose corrections we do not implement.
    pcct_noise_reduction = 0.7,
)

# ╔═╡ 08030003-0000-4000-8000-000000000020
# Same recon grid as notebook 07: 512 × 512 in-plane at 0.625 mm
# isotropic, 3 slices to fit the usable-z budget for 2.5 mm collimation
# (z_usable ≈ c × (1 − R_body/SID) ≈ 1.9 mm).
recon_opts = BS.ReconOptions(
    matrix_size = (512, 512, 3),
    fov_cm = 32.0,
    z_cm = 0.1875,
);

# ╔═╡ 08030004-0000-4000-8000-000000000001
md"""
### 05. Forward Project (PCCT)

Run `BS.simulate!` once on the PCCT protocol.  The simulator returns
`(pcct_sino, I0_bins, pileup_S)` — the 4 per-bin log-line-integral
sinograms, their matching reference photon counts, and the MC-LUT
pile-up migration matrix `S`.  Pile-up correction is applied directly
on the GPU bins (no-op if `use_pcct_pileup = false`), then a model-
based per-bin scatter correction strips the simulator-injected scatter
field.

Inside `simulate!`:
- Forward projection uses the **MC-LUT detector response matrix**
  (`compute_mc_drm` → `cdte_response_v4.jls`) — captures CdTe transport,
  Fano noise, charge cloud (Dreier 2018), 3×3 charge sharing, and
  threshold comparison in a single Monte-Carlo-derived R(E,b).
- Pulse pileup is the **MC-LUT spectral-migration matrix S**
  (`compute_mc_pileup_matrix`).  Toggle with
  `SimOptions(use_pcct_pileup=…)`; default ON for `:pcct`.
"""

# ╔═╡ 08030004-0000-4000-8000-000000000010
# === SLOW CELL (~6 min) — Full measured sinogram via STANDARD simulate!() ===
# Single src call producing the 4 per-bin log-line-integral sinograms with the
# COMPLETE PCCT physics + corrections, all gated by `sim_opts` flags:
#   forward → scatter inject (use_pcct_scatter) → quantum noise (use_noise,
#   pcct_noise_reduction) → pile-up fwd (use_pcct_pileup) → pile-up correction
#   (use_pcct_pileup_correction) → scatter correction (use_pcct_scatter_correction).
# Consumed directly by the bin-combine below — the former inline forward cells
# (pile-up / scatter fwd + correct) are GONE; this is the src-proper path.
#
sim_raw = let
    @info "simulate!(): $(Int(protocol.kVp)) kVp / $(round(protocol.mA, digits = 1)) mA — full PCCT physics + corrections (nr = $(sim_opts.pcct_noise_reduction))"
    ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    result = BS.simulate!(ws, phantom, protocol, sim_opts)

    bins_raw = [Array(b) for b in result.pcct_sino.bins]   # full physics + corrections (scatter+noise+pileup, per sim_opts)
    I0_bins = copy(result.I0_bins)
    geom = ws.geom
    energies = Float64.(ws.energies)
    weights = copy(ws.weights)
    # The EXACT per-bin detected spectra the forward applied (w·η·DRM with the
    # workspace's MC-LUT η + centre-pixel bowtie fold) — the decomposition
    # basis consumes THESE, so the inversion's forward model is the model
    # simulate! actually applied (matches nb04).
    W_applied = Float64.(Array(ws.W_matrix_gpu))[1:length(ws.energies), :]
    bf = scanner.binning_factor

    ws = nothing; result = nothing
    GC.gc(true)
    (
        bins_raw = bins_raw, I0_bins = I0_bins, W_applied = W_applied,
        geom = geom, energies = energies, weights = weights, bf = bf,
    )
end;

# ╔═╡ 08030004-0000-4000-8000-000000000025
# Resample the phantom labels onto the recon grid via BS's affine
# round-trip — used downstream for ROI construction in recon coords.
phantom_in_recon = BS.resample_to_recon(
    phantom_cpu, sim_raw.geom, recon_opts.matrix_size; method = :nearest,
);

# ╔═╡ 08030004-0000-4000-8000-000000000040
let
    n_row = size(sim_raw.bins_raw[1], 2)
    mid_r = n_row ÷ 2 + 1

    bin_titles = ("Bin 1", "Bin 2", "Bin 3", "Bin 4")
    bin_subs = ("20 – 35 keV", "35 – 55 keV", "55 – 70 keV", "> 70 keV")
    slices = [permutedims(sim_raw.bins_raw[k][:, mid_r, :], (2, 1)) for k in 1:4]

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

# ╔═╡ 08030000-0000-4000-8000-0000000000f1
md"""
## VMI Pipeline

4-bin combine → projection-domain material decomposition → per-basis FBP →
image-domain ACNR → monoenergetic synthesis.  The PCCT counterpart of the
notebook 04 chain.
"""

# ╔═╡ 08030005-0000-4000-8000-000000000001
md"""
### 01. Bin Combine: 4 Bins → Low / High Pair

I₀-weighted Beer recombination of the 4 raw PCCT bins:

```
N_grp = Σ_{b ∈ grp} I0[b] · exp(-p[b])
p_grp = -log(N_grp / Σ_{b ∈ grp} I0[b])
```

* **Low**  = bins **1 + 2 + 3** (20 – 70 keV)
* **High** = bin **4** ( > 70 keV)

Each combined sinogram represents a polychromatic measurement at the
I₀-weighted average spectrum of its bin group — the two-channel
`(low, high)` pair the Cong PCCT-Φ_k decomposition then consumes
directly, exactly as nb04 does — 123|4 partition, physical DAS floor + Jensen
debias, applied-W Cong basis.
"""

# ╔═╡ 08030005-0000-4000-8000-000000000010
sim_lohi = let
    low_bins = [1, 2, 3]     # 123|4 partition — analytic noise optimum (matches nb04)
    high_bins = [4]

    I0_lo = Float32(sum(Float64.(sim_raw.I0_bins[low_bins])))
    I0_hi = Float32(sum(Float64.(sim_raw.I0_bins[high_bins])))

    sz = size(sim_raw.bins_raw[1])
    N_lo = zeros(Float32, sz)
    N_hi = zeros(Float32, sz)
    for b in low_bins
        I0b = Float32(sim_raw.I0_bins[b])
        @. N_lo += I0b * exp(-sim_raw.bins_raw[b])
    end
    for b in high_bins
        I0b = Float32(sim_raw.I0_bins[b])
        @. N_hi += I0b * exp(-sim_raw.bins_raw[b])
    end

    # Physical DAS floor (1 count) + first-order log-Poisson DEBIAS (matches
    # nb04): E[−log(N/I0)] = p_true + s²/(2N), s = 1 − pcct_noise_reduction.
    # Deterministic bias correction (σ untouched).
    s_nr = Float32((1.0 - sim_opts.pcct_noise_reduction)^2)
    N_lo_f = max.(N_lo, 1.0f0)
    N_hi_f = max.(N_hi, 1.0f0)
    sino_low = Float32.(.- log.(N_lo_f ./ I0_lo) .- s_nr ./ (2.0f0 .* N_lo_f))
    sino_high = Float32.(.- log.(N_hi_f ./ I0_hi) .- s_nr ./ (2.0f0 .* N_hi_f))

    (
        sino_low = sino_low, sino_high = sino_high,
        I0_lo = I0_lo, I0_hi = I0_hi,
        geom = sim_raw.geom,
    )
end;

# ╔═╡ 08030005-0000-4000-8000-000000000030
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
        (1, 1, "Low Bins", "20 – 55 keV", slice_lo),
        (1, 2, "High Bins", "> 55 keV", slice_hi),
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

# ╔═╡ 08030007-0000-4000-8000-000000000001
md"""
### 02. Projection-Domain Material Decomposition (PCCT-Φ_k)

Per-ray Cong univariate solver mapped to PCCT via the generalization in
Black (*in prep.*) — re-derives the Cong 2022 framework around an
effective spectral response Φ_k(ε) ≥ 0 so the same algorithm runs on
dual-kVp DECT and PCCT acquisitions without code changes.  The
bin-combine partition (123 → low, 4 → high) is baked into Φ_k by
summing the relevant DRM columns:

```
Φ_low(ε)  = S(ε) · η(ε) · Σ_{b ∈ {1,2}} R(ε, b)
Φ_high(ε) = S(ε) · η(ε) · Σ_{b ∈ {3,4}} R(ε, b)
```

`(p, q)` are the iodine + water mass-attenuation coefficients at the
shared energy grid (matter-based variant, Cong follow-up §2.7) — same
array for both channels since only Φ differs.

Output sinograms are per-ray basis line integrals
`sino_iodine = ∫c_iodine(r)dr` and `sino_water = ∫c_water(r)dr`
(g/cm²) — calibration-free, no forward-projected step-wedge fit.
"""

# ╔═╡ 08030007-0000-4000-8000-000000000005
# Bin-combine partition feeding the two Cong channels.  Must match the
# §6 combine — change here AND there together.
begin
    cong_low_bins = 1:3          # PCCT bins forming the "low"  channel (123|4, matches §6)
    cong_high_bins = 4:4          # PCCT bins forming the "high" channel
end

# ╔═╡ 08030007-0000-4000-8000-000000000010
material_basis = let
    # Per-channel spectra = column sums of the sim's APPLIED W matrix over
    # the bin partition — exact by construction, identical to nb04.  (The
    # forward applies the MC-LUT η + centre-pixel bowtie fold; rebuilding
    # w·η·R with the analytic η here would mismatch it and reintroduce a
    # keV-dependent HU bias.  With no bowtie the per-pixel ŵ collapses to
    # this 1-D spectrum, so this is the exact effective model.)
    e = sim_raw.energies
    ΦL = Float32.(vec(sum(sim_raw.W_applied[:, collect(cong_low_bins)]; dims = 2)))
    ΦH = Float32.(vec(sum(sim_raw.W_applied[:, collect(cong_high_bins)]; dims = 2)))
    ŵ_L_f32 = ΦL ./ sum(ΦL)
    ŵ_H_f32 = ΦH ./ sum(ΦH)

    p = Float32[Float32(BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, Float64(E))) for E in e]
    q = Float32[Float32(BS.compute_mass_μ_at_energy(BS.XA.Materials.water, Float64(E))) for E in e]

    @info "[Cong basis · applied-W] low ⟨E⟩ = $(round(sum(e .* Float64.(ŵ_L_f32)); digits = 1)) keV · " *
        "high ⟨E⟩ = $(round(sum(e .* Float64.(ŵ_H_f32)); digits = 1)) keV"

    (
        ŵ_L = ŵ_L_f32, p_L = p, q_L = q,
        ŵ_H = ŵ_H_f32, p_H = copy(p), q_H = copy(q),
    )
end;

# ╔═╡ 08030007-0000-4000-8000-000000000020
sino_basis = let
    sino_low_gpu = to_gpu(Float32.(sim_lohi.sino_low))
    sino_high_gpu = to_gpu(Float32.(sim_lohi.sino_high))

    sino_y = similar(sino_low_gpu)
    sino_c = similar(sino_low_gpu)
    fill!(sino_y, 0.0f0); fill!(sino_c, 0.0f0)

    @info "Cong polychromatic decomposition: $(size(sim_lohi.sino_low))"
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
        geom = sim_lohi.geom,
    )
    sino_low_gpu = nothing; sino_high_gpu = nothing
    sino_y = nothing; sino_c = nothing; cong_ws = nothing
    GC.gc(true)
    result
end;

# ╔═╡ 08030007-0000-4000-8000-000000000040
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
        (1, 1, 2, "Iodine Basis Sinogram", "g/cm²", slice_iod, _qrange(slice_iod)),
        (1, 3, 4, "Water Basis Sinogram", "g/cm²", slice_wat, _qrange(slice_wat)),
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

# ╔═╡ 08030007-0000-4000-8000-000000000060
# === Basis-sinogram covariance readout (sinogram domain, PRE-FBP) ===
# Measures the iodine↔water correlation in the basis SINOGRAMS that feed FBP — the
# quantity an ACNR-before-FBP step would act on.  Two views:
#   • GLOBAL    — cov over all rays.  With noise OFF this is the deterministic
#                 STRUCTURE correlation Cong bakes in; with noise ON it also carries it.
#   • Δcol HF   — cov of the adjacent-detector-column difference (isolates the
#                 high-frequency / noise-like component — what ACNR actually targets).
# ρ is scale-free (iodine sino tiny, water sino large), so it is the honest correlation.
# ρ < 0 ⇒ anti-correlated basis content ⇒ ACNR has something to remove.  Single-pass
# scalar accumulation — no full-sinogram temporaries (memory budget).
let
    io = sino_basis.sino_iodine
    wa = sino_basis.sino_water
    n = length(io)

    # ── GLOBAL covariance (two scalar passes) ──
    si = 0.0; sw = 0.0
    @inbounds for k in eachindex(io)
        si += io[k]; sw += wa[k]
    end
    mi = si / n; mw = sw / n
    Vi = 0.0; Vw = 0.0; Ciw = 0.0
    @inbounds for k in eachindex(io)
        a = io[k] - mi; b = wa[k] - mw
        Vi += a * a; Vw += b * b; Ciw += a * b
    end
    Vi /= n; Vw /= n; Ciw /= n
    ρ_g = Ciw / sqrt(max(Vi * Vw, 1.0e-30))

    # ── HIGH-FREQ covariance: adjacent-detector-column difference Δcol ──
    nc, nr, nv = size(io)
    sdi = 0.0; sdw = 0.0; ndf = 0
    @inbounds for v in 1:nv, r in 1:nr, c in 2:nc
        sdi += io[c, r, v] - io[c - 1, r, v]
        sdw += wa[c, r, v] - wa[c - 1, r, v]
        ndf += 1
    end
    mdi = sdi / ndf; mdw = sdw / ndf
    Vih = 0.0; Vwh = 0.0; Ciwh = 0.0
    @inbounds for v in 1:nv, r in 1:nr, c in 2:nc
        a = (io[c, r, v] - io[c - 1, r, v]) - mdi
        b = (wa[c, r, v] - wa[c - 1, r, v]) - mdw
        Vih += a * a; Vwh += b * b; Ciwh += a * b
    end
    Vih /= ndf; Vwh /= ndf; Ciwh /= ndf
    ρ_hf = Ciwh / sqrt(max(Vih * Vwh, 1.0e-30))

    @info "[basis-sino cov · GLOBAL]  σ_iod = $(round(sqrt(Vi), sigdigits = 3)) · σ_wat = $(round(sqrt(Vw), sigdigits = 3)) g/cm² · ρ(iod,wat) = $(round(ρ_g, digits = 3))"
    @info "[basis-sino cov · Δcol HF] σ_iod = $(round(sqrt(Vih), sigdigits = 3)) · σ_wat = $(round(sqrt(Vwh), sigdigits = 3)) g/cm² · ρ_hf(iod,wat) = $(round(ρ_hf, digits = 3))  ← ACNR target"
    @info "  ρ<0 ⇒ anti-correlated basis content (ACNR has something to remove).  Noise is OFF → these are the STRUCTURE baseline; turn channel noise ON to read the NOISE correlation."
    (ρ_global = ρ_g, ρ_highfreq = ρ_hf, V_iodine = Vi, V_water = Vw, C_iw = Ciw)
end

# ╔═╡ 08030008-0000-4000-8000-000000000001
md"""
### 03. FBP: Iodine and Water Basis Maps

Two FDK passes — one per basis sinogram.  Output volumes are in
basis-density units (g/cm³).
"""

# ╔═╡ 08030008-0000-4000-8000-000000000010
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

# ╔═╡ 08030008-0000-4000-8000-000000000030
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

# ╔═╡ 08030008-0000-4000-8000-000000000050
md"""
### 04. ACNR: Edge-Aware Anti-Correlated Noise Reduction (Image Domain)

Material decomposition stamps strongly **anti-correlated** noise onto the basis
maps (measured `ρ_basis ≈ −0.92`) — that anti-correlation *is* the VMI-noise U.
ACNR removes it. Now runs via the src-proper **`BS.apply_acnr_kalender!`** (per-pixel regression, zero blur)
(`denoising/acnr.jl`).

**Data-adaptive cov-ACNR.** A closed-form 2×2 eigen-rotation of the joint W–I
covariance *learns* the signal/noise axes instead of assuming them. The
large-variance axis **e1** (correlated structure) is kept **pixel-perfect**; only
the small-variance axis **e2** (the anti-correlated noise that *is* the U) is
denoised. Targeting the *true* noise eigenvector removes more `|C_iw|` per unit
blur than a fixed anchor.

Resolution is preserved two ways: (1) e1 is kept pixel-perfect, and (2) the
denoised e2 axis is smoothed with a **joint bilateral guided by BOTH basis maps**, so
any real water/iodine edge survives — only locally-flat anti-correlated noise is
removed. The resolution check below shows the *removed* component: it must be
**structureless noise** (no rod rings). Runs on the FBP basis maps, **before** the
§9 Kalender ACNR.
"""

# ╔═╡ 08030008-0000-4000-8000-000000000055
# Image-domain cov-ACNR on the FBP basis maps via the src-proper
# `BS.apply_acnr_kalender!` (denoising/acnr.jl) — no knobs, no blur.
basis_acnr = let
    APPLY_ACNR = true        # ON — image-domain edge-aware cov-ACNR

    W = copy(basis_volumes.vol_water_raw)
    I = copy(basis_volumes.vol_iodine_raw)

    if APPLY_ACNR
        info = BS.apply_acnr_kalender!(W, I)
        @info "[ACNR · Kalender-1988 true ACNR] ρ_hp(W,I)=$(round(info.ρ_hp, digits = 3))"
    else
        @info "[ACNR] OFF (passthrough)"
    end

    (vol_iodine = I, vol_water = W, geom = basis_volumes.geom)
end;

# ╔═╡ 08030008-0000-4000-8000-000000000058
# Resolution check — the REMOVED component (after − before) must be structureless
# noise.  If rod rings/edges show up in the right panel, ACNR is sacrificing
let
    mid = size(basis_acnr.vol_iodine, 3) ÷ 2 + 1
    before = basis_volumes.vol_iodine_raw[:, :, mid]
    after = basis_acnr.vol_iodine[:, :, mid]
    removed = after .- before

    rng = (Float64(quantile(vec(before), 0.01)), Float64(quantile(vec(before), 0.99)))
    rmax = max(maximum(abs.(removed)), 1.0f-12)

    fig = CM.Figure(size = (1500, 540))
    ak = (titlesize = 26, subtitlesize = 18, aspect = CM.DataAspect())
    panels = (
        (1, "Iodine basis — before ACNR", "g/cm³", before, :viridis, rng),
        (2, "Iodine basis — after ACNR", "g/cm³", after, :viridis, rng),
        (3, "REMOVED (after − before)", "must be structureless noise", removed, :balance, (-rmax, rmax)),
    )
    for (c, ttl, sub, sl, cm, cr) in panels
        ax = CM.Axis(fig[1, c]; title = ttl, subtitle = sub, ak...)
        CM.heatmap!(ax, sl; colormap = cm)
        CM.hidedecorations!(ax)
    end
    CM.Colorbar(
        fig[1, 4]; colormap = :balance, colorrange = (-rmax, rmax),
        label = "removed (g/cm³)", width = 16, labelsize = 20, ticklabelsize = 16
    )
    fig
end

# ╔═╡ 08030009-0000-4000-8000-000000000030
let
    fig = CM.Figure(size = (1180, 580))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    mid = size(basis_acnr.vol_iodine, 3) ÷ 2

    _qrange(arr) = (
        Float64(quantile(vec(arr), 0.01)),
        Float64(quantile(vec(arr), 0.99)),
    )

    slice_iod = basis_acnr.vol_iodine[:, :, mid]
    slice_wat = basis_acnr.vol_water[:, :, mid]

    panels = (
        (1, 1, 2, "Iodine Basis · Kalender ACNR", "g/cm³", slice_iod, _qrange(slice_iod)),
        (1, 3, 4, "Water Basis · Kalender ACNR", "g/cm³", slice_wat, _qrange(slice_wat)),
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

# ╔═╡ 0803000a-0000-4000-8000-000000000001
md"""
### 05. VMI Synthesis

`BS.synth_vmi_2basis(c_water, c_iodine_mg_per_mL; energy_keV)` evaluates
the textbook 2-basis linear mix (McCollough 2015) at the target keV:

```
μ(E)  = c_water(r) · (μ/ρ)_water(E) + c_iodine(r) · (μ/ρ)_iodine(E)
HU(E) = 1000 · (μ(E) − (μ/ρ)_water(E)) / (μ/ρ)_water(E)
```

VMI grid: 40, 70, 100, 140 keV — matched to notebook 07.  The
`solid_water_basis` cell measures `⟨c_water⟩` and `⟨c_iodine⟩` over the
**dedicated water-rod core ROI** (label 9, `basis_water`) — a diagnostic
used to log Δ% drift between the water-rod-anchored synth μ_water and
the textbook mono divisor.
"""

# ╔═╡ 0803000a-0000-4000-8000-000000000005
solid_water_basis = let
    WATER_ROD_LABEL = UInt8(9)
    ROI_RADIUS_PX = 8

    mask_2d = phantom_in_recon[:, :, size(phantom_in_recon, 3) ÷ 2 + 1]
    nx_r, ny_r, nz_r = size(basis_acnr.vol_water)

    rod_idx = findall(==(WATER_ROD_LABEL), mask_2d)
    isempty(rod_idx) && error(
        "solid_water_basis: no label-$(Int(WATER_ROD_LABEL)) (water rod) voxels in resampled phantom mask."
    )
    cx = sum(ci -> Float64(ci[1]), rod_idx) / length(rod_idx)
    cy = sum(ci -> Float64(ci[2]), rod_idx) / length(rod_idx)

    sw_bool = falses(nx_r, ny_r)
    r² = Float64(ROI_RADIUS_PX)^2
    @inbounds for j in 1:ny_r, i in 1:nx_r
        ((i - cx)^2 + (j - cy)^2) ≤ r² && (sw_bool[i, j] = true)
    end

    n_voxels = count(sw_bool)
    @info "solid_water_basis: water-rod core ROI center = " *
        "($(round(cx, digits = 1)), $(round(cy, digits = 1))) recon voxels, " *
        "$(n_voxels) voxels in core"

    sw_idx = findall(sw_bool)
    function _mean(vol)
        s = 0.0; n = 0
        for z in 1:nz_r, ci in sw_idx
            s += vol[ci, z]; n += 1
        end
        return s / n
    end

    c_w = Float64(_mean(basis_acnr.vol_water))
    c_i = Float64(_mean(basis_acnr.vol_iodine))
    @info "solid_water_basis: ⟨c_water⟩_water-rod = $(round(c_w, digits = 4)) g/cm³, " *
        "⟨c_iodine⟩_water-rod = $(round(c_i, digits = 6)) g/cm³"

    (
        c_water = c_w, c_iodine = c_i, n_voxels = length(sw_idx) * nz_r,
        mask_2d = collect(sw_bool),
    )
end;

# ╔═╡ 0803000a-0000-4000-8000-000000000010
pcct_vmi_energies = [40.0, 70.0, 100.0, 140.0];

# ╔═╡ 0803000a-0000-4000-8000-000000000020
vmi_HU_final = let
    c_iodine_mg_per_mL = basis_acnr.vol_iodine .* 1000.0f0

    out = Dict{Float64, Array{Float32, 3}}()
    for E in pcct_vmi_energies
        μρ_w = BS.compute_mass_μ_at_energy(BS.XA.Materials.water, E)
        μρ_I = BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, E)
        μ_water_anchor = solid_water_basis.c_water * μρ_w +
            solid_water_basis.c_iodine * μρ_I
        Δ_pct = 100.0 * (μ_water_anchor - μρ_w) / μρ_w
        @info "VMI synth @ $(Int(E)) keV: divisor = $(round(μρ_w, digits = 5)) cm⁻¹ " *
            "(mono μρ_water);  SW-ROI anchor = $(round(μ_water_anchor, digits = 5)) " *
            "→ Δ = $(round(Δ_pct, digits = 2))%"

        out[E] = BS.synth_vmi_2basis(
            basis_acnr.vol_water, c_iodine_mg_per_mL; energy_keV = E,
        )
    end
    out
end;

# ╔═╡ 0803000a-0000-4000-8000-000000000040
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
            aspect = CM.DataAspect(), axis_kwargs...,
        )
        CM.heatmap!(
            ax, vmi_HU_final[E][:, :, mid];
            colormap = :grays, colorrange = HU_window
        )
        CM.hidedecorations!(ax)
    end
    CM.Colorbar(
        fig[1:2, 3]; colormap = :grays, colorrange = HU_window,
        label = "HU", width = 16, labelsize = 22, ticklabelsize = 18
    )
    CM.save(
        joinpath(@__DIR__, "..", "assets", "qrm_thorax_pcct_vmi_grid.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0803000c-0000-4000-8000-000000000000
md"""
### Phantom-Recon Alignment Verification

Sanity-check the BS affine round-trip before trusting any ROI built on
`phantom_in_recon`.  Recon HU and the resampled phantom mask side-by-
side on the same recon grid, then overlaid to confirm rod / heart-cavity /
bone-wall edges land cleanly on the corresponding recon edges.
"""

# ╔═╡ 0803000c-0000-4000-8000-00000000000a
let
    z_recon = size(vmi_HU_final[70.0], 3) ÷ 2 + 1
    hu_slice = vmi_HU_final[70.0][:, :, z_recon]

    z_pir = clamp(z_recon, 1, size(phantom_in_recon, 3))
    pir_slice = phantom_in_recon[:, :, z_pir]

    rod_overlay = let
        out = fill(NaN32, size(pir_slice))
        @inbounds for idx in eachindex(pir_slice)
            v = Int(pir_slice[idx])
            (v == 9 || v == 10 || v == 11 || v == 12) && (out[idx] = Float32(v))
        end
        out
    end

    full_overlay = let
        out = Float32.(pir_slice)
        @inbounds for idx in eachindex(out)
            out[idx] == 1.0f0 && (out[idx] = NaN32)
        end
        out
    end

    fig = CM.Figure(size = (1400, 1320))
    hu_kwargs = (colormap = :grays, colorrange = (-200, 500))
    title_kwargs = (titlesize = 28, subtitlesize = 20)

    ax_tl = CM.Axis(
        fig[1, 1];
        title = "70 keV VMI HU recon",
        subtitle = "z = $(z_recon) of $(size(vmi_HU_final[70.0], 3))",
        aspect = CM.DataAspect(),
        title_kwargs...,
    )
    CM.heatmap!(ax_tl, hu_slice; hu_kwargs...)
    CM.hidedecorations!(ax_tl)

    ax_tr = CM.Axis(
        fig[1, 2];
        title = "phantom_in_recon (`:nearest` resample)",
        subtitle = "$(length(unique(pir_slice))) unique labels on recon grid",
        aspect = CM.DataAspect(),
        title_kwargs...,
    )
    CM.heatmap!(ax_tr, Float32.(pir_slice); colormap = :tab20)
    CM.hidedecorations!(ax_tr)

    ax_bl = CM.Axis(
        fig[2, 1];
        title = "HU + rod labels (9–12)",
        subtitle = "α = 0.6 over HU — rod edges should sit on recon rod edges",
        aspect = CM.DataAspect(),
        title_kwargs...,
    )
    CM.heatmap!(ax_bl, hu_slice; hu_kwargs...)
    CM.heatmap!(
        ax_bl, rod_overlay;
        colormap = :tab20, alpha = 0.6, nan_color = (:white, 0.0),
        colorrange = (1, 12),
    )
    CM.hidedecorations!(ax_bl)

    ax_br = CM.Axis(
        fig[2, 2];
        title = "HU + full label mask (every non-air label)",
        subtitle = "α = 0.4 over HU — bone walls / heart cavity edge alignment",
        aspect = CM.DataAspect(),
        title_kwargs...,
    )
    CM.heatmap!(ax_br, hu_slice; hu_kwargs...)
    CM.heatmap!(
        ax_br, full_overlay;
        colormap = :tab20, alpha = 0.4, nan_color = (:white, 0.0),
        colorrange = (1, 12),
    )
    CM.hidedecorations!(ax_br)

    fig
end

# ╔═╡ 0803000c-0000-4000-8000-000000000001
md"""
## Results

Per-rod measured vs theoretical HU at 40 / 70 / 100 / 140 keV.

- **Measured HU** = mean over an 8-px-radius circular ROI at each rod
  centroid, broadcast across all z slices.
- **Theoretical HU** = `1000 · (μ_r(E) − μ_water(E)) / μ_water(E)` from
  `BS.compute_μ_at_energy(material, E)` — pure physics, no fitting.
"""

# ╔═╡ 0803000c-0000-4000-8000-000000000002
md"""
### Water ROI

Water-rod core ROI (label 9 = `basis_water`) overlaid in red on the
70 keV VMI slice.  Right panel: mean HU over that ROI vs VMI energy.
Bars should cluster near 0 HU; consistent ~few-HU offset = residual
basis-decomp bias, energy-dependent drift = upstream spectral problem.
"""

# ╔═╡ 0803000c-0000-4000-8000-000000000003
let
    HU_window = (-200, 500)
    mid = size(vmi_HU_final[70.0], 3) ÷ 2
    bg = vmi_HU_final[70.0][:, :, mid]

    overlay = Float32[b ? 1.0f0 : NaN32 for b in solid_water_basis.mask_2d]

    fig = CM.Figure(size = (1180, 580))

    ax1 = CM.Axis(
        fig[1, 1];
        title = "SW-ROI (water rod core)",
        subtitle = "Overlaid on 70 keV VMI",
        aspect = CM.DataAspect(),
        titlesize = 32, subtitlesize = 24,
    )
    CM.heatmap!(ax1, bg; colormap = :grays, colorrange = HU_window)
    CM.heatmap!(
        ax1, overlay; colormap = :reds, alpha = 0.5,
        nan_color = (:white, 0.0)
    )
    CM.hidedecorations!(ax1)

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

    n_E = length(pcct_vmi_energies)
    bar_colors = [CM.cgrad(:plasma, n_E; categorical = true)[i] for i in 1:n_E]

    ax2 = CM.Axis(
        fig[1, 2];
        title = "Water ROI Mean HU",
        subtitle = "Per VMI Energy",
        xlabel = "VMI Energy (keV)", ylabel = "HU",
        xticks = (collect(1:n_E), ["$(Int(E))" for E in pcct_vmi_energies]),
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    CM.barplot!(
        ax2, 1:n_E, sw_hu_per_keV;
        color = bar_colors, strokecolor = :black, strokewidth = 1
    )
    CM.hlines!(ax2, [0.0]; color = :black, linewidth = 1, linestyle = :dash)

    for (k, h) in pairs(sw_hu_per_keV)
        CM.text!(
            ax2, k, h;
            text = "$(round(h, digits = 1)) HU",
            align = (:center, h ≥ 0 ? :bottom : :top),
            fontsize = 16, offset = (0, h ≥ 0 ? 4 : -4)
        )
    end

    y_max = max(15.0, 1.2 * maximum(abs, sw_hu_per_keV))
    CM.ylims!(ax2, -y_max, y_max)

    CM.save(
        joinpath(@__DIR__, "..", "assets", "qrm_thorax_pcct_water_roi.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0803000d-0000-4000-8000-000000000001
md"""
### Heart-Center Noise ROI

HU noise (σ) inside a circular ROI offset from the **water-rod centroid**
(label 9 — the only label guaranteed present in `phantom_in_recon`
after `:nearest` resample at 0.625 mm).  Default offset
`(dx, dy) = (0, -40)` puts the ROI in the heart cavity between the
water rod and the cavity center — tweak `HEART_NOISE_ROI_OFFSET_PX` to
iterate.

Right panel = σ vs VMI energy.  Diagnoses how the textbook
(c_water, c_iodine) → HU(E) synth propagates noise through the PCCT
pipeline (Cong-Φ_k + SF-JSD).  Expectation: σ(40) ≫ σ(70) ≳ σ(140) —
the natural noise-optimal energy is near 70 keV.
"""

# ╔═╡ 0803000d-0000-4000-8000-000000000004
const HEART_NOISE_ROI_OFFSET_PX = (0, -40);   # (dx, dy) recon vx — default: below water-rod centroid

# ╔═╡ 0803000d-0000-4000-8000-000000000005
const HEART_NOISE_ROI_RADIUS_PX = 12;   # ≈7.5 mm at 0.625 mm/voxel

# ╔═╡ 0803000d-0000-4000-8000-000000000010
heart_noise_roi = let
    WATER_ROD_LABEL = UInt8(9)

    mask_2d = phantom_in_recon[:, :, size(phantom_in_recon, 3) ÷ 2 + 1]
    nx_r, ny_r, nz_r = size(basis_acnr.vol_water)

    rod_idx = findall(==(WATER_ROD_LABEL), mask_2d)
    isempty(rod_idx) && error(
        "heart_noise_roi: no label-$(Int(WATER_ROD_LABEL)) (water rod) voxels in resampled phantom mask."
    )
    rod_cx = sum(ci -> Float64(ci[1]), rod_idx) / length(rod_idx)
    rod_cy = sum(ci -> Float64(ci[2]), rod_idx) / length(rod_idx)

    dx, dy = HEART_NOISE_ROI_OFFSET_PX
    cx = rod_cx + dx
    cy = rod_cy + dy

    roi_bool = falses(nx_r, ny_r)
    r² = Float64(HEART_NOISE_ROI_RADIUS_PX)^2
    @inbounds for j in 1:ny_r, i in 1:nx_r
        ((i - cx)^2 + (j - cy)^2) ≤ r² && (roi_bool[i, j] = true)
    end

    n_vox = count(roi_bool)
    @info "heart_noise_roi: water-rod centroid = ($(round(rod_cx, digits = 1)), $(round(rod_cy, digits = 1))), " *
        "offset = $(HEART_NOISE_ROI_OFFSET_PX), ROI center = ($(round(cx, digits = 1)), $(round(cy, digits = 1))), " *
        "$(n_vox) vx × $(nz_r) z = $(n_vox * nz_r) total"

    (
        rod_center_xy = (rod_cx, rod_cy),
        center_xy = (cx, cy),
        mask_2d = roi_bool,
        n_voxels = n_vox,
        n_total = n_vox * nz_r,
    )
end;

# ╔═╡ 0803000d-0000-4000-8000-000000000015
# === Image-domain basis covariance (heart ROI) — the VMI-noise theorem inputs ===
# The quantity that ACTUALLY governs the U:
#   σ_HU(E)² = 1e6·V_w + α(E)²·V_i + 2e3·α(E)·C_iw ,   min at  α* = −1e3·C_iw/V_i
#   monotonic-DECREASING ⟺ α* ≤ α(140) ⟺ C_iw not too negative.
# Measured over the heart ROI on the ACNR basis maps VMI synth consumes
# (c_iodine in mg/mL).  ρ_basis<0 = anti-correlated basis noise = the U + the ACNR
# target.  Small ROI → tiny vectors, no memory concern.
let
    roi = findall(heart_noise_roi.mask_2d)
    nz = size(basis_acnr.vol_water, 3)
    cw = Float64[Float64(basis_acnr.vol_water[p, z])           for z in 1:nz, p in roi]
    ci = Float64[Float64(basis_acnr.vol_iodine[p, z]) * 1000.0 for z in 1:nz, p in roi]  # mg/mL
    mw = mean(cw); mi = mean(ci)
    Vw = mean((cw .- mw) .^ 2); Vi = mean((ci .- mi) .^ 2)
    Ciw = mean((cw .- mw) .* (ci .- mi))
    ρ_b = Ciw / sqrt(max(Vw * Vi, 1.0e-30))
    αf(E) = Float64(BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, E)) /
        Float64(BS.compute_mass_μ_at_energy(BS.XA.Materials.water, E))
    α_star = -1000.0 * Ciw / max(Vi, 1.0e-30)
    verdict = α_star ≤ αf(140.0) ? "MONOTONIC-decreasing predicted ✓" :
        α_star ≥ αf(40.0) ? "monotonic-INCREASING" :
        "U-shape (min near α=α*)"
    @info "[basis cov · heart ROI, image]  σ_water = $(round(sqrt(Vw), sigdigits = 3)) g/mL · " *
        "σ_iod = $(round(sqrt(Vi), sigdigits = 3)) mg/mL · ρ_basis = $(round(ρ_b, digits = 3)) · " *
        "water floor = $(round(1000 * sqrt(Vw), digits = 1)) HU"
    @info "  α* = $(round(α_star, digits = 1))  vs  α(40)=$(round(αf(40.0), digits = 1)), " *
        "α(70)=$(round(αf(70.0), digits = 1)), α(140)=$(round(αf(140.0), digits = 1))  →  $(verdict)"
    (V_water = Vw, V_iodine = Vi, C_iw = Ciw, ρ_basis = ρ_b, α_star = α_star)
end

# ╔═╡ 0803000d-0000-4000-8000-000000000020
vmi_noise_by_keV = let
    roi_idx = findall(heart_noise_roi.mask_2d)
    nz_r = size(basis_acnr.vol_water, 3)

    out = Dict{Float64, NamedTuple}()
    for E in pcct_vmi_energies
        vol = vmi_HU_final[E]
        vals = Float64[Float64(vol[ci, z]) for z in 1:nz_r, ci in roi_idx]
        μ = mean(vals)
        σ = std(vals)
        out[E] = (mean = μ, std = σ, n = length(vals))
        @info "heart noise @ $(Int(E)) keV: ⟨HU⟩ = $(round(μ, digits = 2)),  σ = $(round(σ, digits = 2)) HU  (n = $(length(vals)))"
    end
    out
end;

# ╔═╡ 0803000d-0000-4000-8000-000000000030
let
    HU_window = (-200, 500)
    mid = size(vmi_HU_final[70.0], 3) ÷ 2
    bg = vmi_HU_final[70.0][:, :, mid]

    overlay = Float32[b ? 1.0f0 : NaN32 for b in heart_noise_roi.mask_2d]

    fig = CM.Figure(size = (1180, 580))

    ax1 = CM.Axis(
        fig[1, 1];
        title = "Heart-Center Noise ROI",
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

    ax2 = CM.Axis(
        fig[1, 2];
        title = "Heart-Center Noise vs Energy",
        xlabel = "VMI Energy (keV)",
        ylabel = "Noise σ (HU)",
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    CM.scatterlines!(
        ax2, Es, σs;
        color = :tomato, markersize = 18, linewidth = 3,
    )
    for (E, σ, μ) in zip(Es, σs, μs)
        CM.text!(
            ax2, E, σ;
            text = "σ=$(round(σ; digits = 1))\n⟨HU⟩=$(round(μ; digits = 1))",
            align = (:center, :bottom),
            fontsize = 16, offset = (0, 8),
        )
    end
    # CM.ylims!(ax2, 0, maximum(σs) * 1.4)

    CM.save(
        joinpath(@__DIR__, "..", "assets", "qrm_thorax_pcct_noise_vs_energy.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0803000c-0000-4000-8000-000000000010
const ROD_LABELS = (UInt8(9), UInt8(10), UInt8(11), UInt8(12));

# ╔═╡ 0803000c-0000-4000-8000-000000000011
const ROD_NAMES = ("water", "lipid", "collagen", "iodine_5");

# ╔═╡ 0803000c-0000-4000-8000-000000000012
const ROD_MATERIALS = (
    BS.XA.Materials.basis_water,
    BS.XA.Materials.basis_lipid,
    BS.XA.Materials.basis_collagen,
    BS.XA.Materials.gammex_472_i5_0,
);

# ╔═╡ 0803000a-0000-4000-8000-000000000008
# Per-rod basis-decomp sanity check.  For each of the 4 rods, measure
# the mean (c_water, c_iodine) inside an 8-px-radius core ROI at the
# rod centroid and compare to the rod material's density.
#
# Expected:
#   rod 9  (basis_water)    → c_water ≈ 1.00 g/cm³, c_iodine ≈ 0
#   rod 10 (basis_lipid)    → c_water ≈ 0.92,        c_iodine ≈ 0
#   rod 11 (basis_collagen) → c_water ≈ 1.26,        c_iodine ≈ 0
#   rod 12 (gammex_472_i5_0)→ c_water ≈ 1.0,         c_iodine ≈ 0.005 (5 mg/mL)
let
    mask_2d = phantom_in_recon[:, :, size(phantom_in_recon, 3) ÷ 2 + 1]
    nx, ny = size(mask_2d)
    ROI_R_PX = 8

    n_z = size(basis_acnr.vol_water, 3)

    function rod_centroid(label::UInt8)
        idx = findall(==(label), mask_2d)
        cx = sum(ci -> Float64(ci[1]), idx) / length(idx)
        cy = sum(ci -> Float64(ci[2]), idx) / length(idx)
        return (cx, cy)
    end

    function _disc_idx(cx, cy)
        idx = CartesianIndex{2}[]
        r² = Float64(ROI_R_PX)^2
        i_lo = max(1, floor(Int, cx - ROI_R_PX))
        i_hi = min(nx, ceil(Int, cx + ROI_R_PX))
        j_lo = max(1, floor(Int, cy - ROI_R_PX))
        j_hi = min(ny, ceil(Int, cy + ROI_R_PX))
        for j in j_lo:j_hi, i in i_lo:i_hi
            ((i - cx)^2 + (j - cy)^2) ≤ r² && push!(idx, CartesianIndex(i, j))
        end
        return idx
    end

    function _mean(vol, roi)
        s = 0.0; n = 0
        for z in 1:n_z, ci in roi
            s += vol[ci, z]; n += 1
        end
        return s / n
    end

    @info "Per-rod basis-decomp diagnostic (8-px core ROI in RECON coords, mean over z):"
    for (lab, name, mat) in zip(ROD_LABELS, ROD_NAMES, ROD_MATERIALS)
        cx, cy = rod_centroid(lab)
        roi = _disc_idx(cx, cy)
        c_w = _mean(basis_acnr.vol_water, roi)
        c_i = _mean(basis_acnr.vol_iodine, roi)
        ρ = round(BS.XA.val(mat.density), digits = 3)
        @info "  rod $(lab) ($(rpad(name, 9))): c_water = $(round(c_w, digits = 4)) g/cm³, " *
            "c_iodine = $(round(c_i, digits = 6)) g/cm³  (truth ρ = $(ρ))"
    end
end

# ╔═╡ 0803000c-0000-4000-8000-000000000020
rod_data = let
    mask_2d = phantom_in_recon[:, :, size(phantom_in_recon, 3) ÷ 2 + 1]
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

    rod_rois = Dict(lab => rod_roi_mask(lab) for lab in ROD_LABELS)

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

    n_rods = length(ROD_LABELS)
    n_E = length(pcct_vmi_energies)
    meas = zeros(Float64, n_rods, n_E)
    theo = zeros(Float64, n_rods, n_E)
    for (i, lab) in enumerate(ROD_LABELS)
        mat = ROD_MATERIALS[i]
        for (j, E) in enumerate(pcct_vmi_energies)
            meas[i, j] = measured_hu(vmi_HU_final[E], lab)
            theo[i, j] = theoretical_hu(mat, E)
        end
    end
    (
        labels = ROD_LABELS, names = ROD_NAMES,
        measured = meas, theoretical = theo,
    )
end;

# ╔═╡ 0803000c-0000-4000-8000-000000000030
md"""
### Per-Rod Regression
"""

# ╔═╡ 0803000c-0000-4000-8000-000000000031
let
    fig = CM.Figure(size = (980, 580))

    rod_colors = [
        CM.RGBf(0.2, 0.6, 0.85),  # water    — blue
        CM.RGBf(0.95, 0.65, 0.13),  # lipid    — orange
        CM.RGBf(0.55, 0.3, 0.65),  # collagen — purple
        CM.RGBf(0.85, 0.27, 0.1),  # iodine   — red
    ]

    ax = CM.Axis(
        fig[1, 1];
        title = "Pure-Material Rods (PCCT)",
        subtitle = "40 / 70 / 100 / 140 keV",
        xlabel = "VMI energy (keV)", ylabel = "HU",
        xticks = pcct_vmi_energies,
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )

    pretty_name = Dict(
        "water" => "Water",
        "lipid" => "Lipid",
        "collagen" => "Collagen",
        "iodine_5" => "Iodine 5 mg/mL",
    )

    rod_lines = Vector{Any}(undef, length(rod_data.names))
    rod_labels_str = Vector{String}(undef, length(rod_data.names))
    for i in eachindex(rod_data.names)
        c = rod_colors[i]
        CM.scatterlines!(
            ax, pcct_vmi_energies, vec(rod_data.measured[i, :]);
            color = c, linewidth = 2.5, markersize = 9
        )
        CM.lines!(
            ax, pcct_vmi_energies, vec(rod_data.theoretical[i, :]);
            color = c, linewidth = 1.6, linestyle = :dash
        )
        rod_lines[i] = CM.LineElement(color = c, linewidth = 2.5)

        meas_i = vec(rod_data.measured[i, :])
        theo_i = vec(rod_data.theoretical[i, :])
        rmse_i = sqrt(mean((meas_i .- theo_i) .^ 2))
        scale_i = maximum(abs.(theo_i))
        nrmse_i = scale_i > 1.0f0 ? rmse_i / scale_i * 100 : nothing

        nm = get(pretty_name, rod_data.names[i], rod_data.names[i])
        rod_labels_str[i] = if nrmse_i === nothing
            "$(nm) (RMSE = $(round(rmse_i, digits = 1)) HU)"
        else
            "$(nm) (RMSE = $(round(rmse_i, digits = 1)) HU, nRMSE = $(round(nrmse_i, digits = 1))%)"
        end
    end

    style_meas = CM.MarkerElement(
        color = :black, marker = :circle, markersize = 9,
        strokecolor = :black, strokewidth = 1
    )
    style_theo = CM.LineElement(
        color = :black, linewidth = 1.6, linestyle = :dash
    )
    CM.axislegend(
        ax,
        vcat([style_meas, style_theo], rod_lines),
        vcat(["Measured", "Theoretical"], rod_labels_str);
        position = :rt, framevisible = true, labelsize = 18,
        rowgap = 1, padding = (6, 6, 6, 6),
    )

    CM.save(
        joinpath(@__DIR__, "..", "assets", "qrm_thorax_pcct_vmi_vs_theoretical.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0803000e-0000-4000-8000-000000000001
md"""
## Summary

```
QRM-Thorax mid-slice mask (1600 × 1100 × 20 phantom @ 0.2 mm iso,
                           rods bored at labels 9–12)
   → Forward-project (140 kVp PCCT, 4 bins, scatter-injected)
   → Per-Bin Pile-up + Scatter Correction
   → Bin Combine  (1+2 → low,  3+4 → high)
   → Projection-Domain Material Decomposition  (Cong PCCT-Φ_k)
   → FBP × 2  (iodine, water basis maps)
   → Monoenergetic VMI Synthesis  (textbook 2-basis, mono μρ_water divisor)
   → Per-rod Measured vs Theoretical Regression
        (water · lipid · collagen · iodine at 40 / 70 / 100 / 140 keV)
```

1:1 parity with notebook 07's QRM-Thorax pure-material pipeline,
swapping the dual-kVp GSI acquisition for a Siemens Naeotom Alpha PCCT
4-bin acquisition.  The 4-bin PCCT measurement is bin-combined into a
two-channel `(low, high)` pair that Cong's PCCT-Φ_k decomposition then
consumes directly — exactly the same downstream pipeline as notebook
07 (Cong → FBP × 2 → Kalender ACNR → VMI → per-rod regression).
Phantom, recon grid, VMI energies, ROI definitions, and per-rod
regression style are identical so the two notebooks are directly
comparable.
"""

# ╔═╡ Cell order:
# ╟─08010001-0000-4000-8000-000000000001
# ╟─08010002-0000-4000-8000-000000000001
# ╠═08010003-0000-4000-8000-000000000001
# ╠═08010003-0000-4000-8000-000000000005
# ╠═08010003-0000-4000-8000-000000000006
# ╠═08010003-0000-4000-8000-000000000007
# ╠═08010003-0000-4000-8000-000000000008
# ╠═08010003-0000-4000-8000-000000000010
# ╠═08010003-0000-4000-8000-000000000011
# ╠═08010003-0000-4000-8000-000000000012
# ╠═08010003-0000-4000-8000-000000000013
# ╠═08010003-0000-4000-8000-000000000040
# ╟─08010003-0000-4000-8000-000000000050
# ╟─08020000-0000-4000-8000-0000000000f1
# ╟─08020001-0000-4000-8000-000000000001
# ╠═08020001-0000-4000-8000-000000000010
# ╠═08020001-0000-4000-8000-000000000011
# ╠═08020001-0000-4000-8000-000000000012
# ╠═08020001-0000-4000-8000-000000000013
# ╠═08020001-0000-4000-8000-000000000014
# ╠═08020001-0000-4000-8000-000000000015
# ╠═08020001-0000-4000-8000-000000000016
# ╠═08020001-0000-4000-8000-000000000017
# ╟─08020001-0000-4000-8000-000000000018
# ╠═08020001-0000-4000-8000-000000000020
# ╟─08020001-0000-4000-8000-000000000022
# ╟─08020001-0000-4000-8000-000000000025
# ╠═08020001-0000-4000-8000-000000000026
# ╠═08020001-0000-4000-8000-000000000027
# ╠═08020001-0000-4000-8000-000000000028
# ╠═08020001-0000-4000-8000-000000000029
# ╠═08020001-0000-4000-8000-000000000040
# ╠═08020001-0000-4000-8000-000000000050
# ╠═08020001-0000-4000-8000-000000000051
# ╠═08020001-0000-4000-8000-000000000060
# ╟─08020001-0000-4000-8000-000000000061
# ╟─08020001-0000-4000-8000-000000000062
# ╟─08030001-0000-4000-8000-000000000001
# ╠═08030001-0000-4000-8000-000000000010
# ╟─08030002-0000-4000-8000-000000000001
# ╠═08030002-0000-4000-8000-000000000010
# ╟─08030003-0000-4000-8000-000000000001
# ╠═08030003-0000-4000-8000-000000000010
# ╠═08030003-0000-4000-8000-000000000020
# ╟─08030004-0000-4000-8000-000000000001
# ╠═08030004-0000-4000-8000-000000000010
# ╠═08030004-0000-4000-8000-000000000025
# ╟─08030004-0000-4000-8000-000000000040
# ╟─08030000-0000-4000-8000-0000000000f1
# ╟─08030005-0000-4000-8000-000000000001
# ╠═08030005-0000-4000-8000-000000000010
# ╟─08030005-0000-4000-8000-000000000030
# ╟─08030007-0000-4000-8000-000000000001
# ╠═08030007-0000-4000-8000-000000000005
# ╠═08030007-0000-4000-8000-000000000010
# ╠═08030007-0000-4000-8000-000000000020
# ╟─08030007-0000-4000-8000-000000000040
# ╠═08030007-0000-4000-8000-000000000060
# ╟─08030008-0000-4000-8000-000000000001
# ╠═08030008-0000-4000-8000-000000000010
# ╟─08030008-0000-4000-8000-000000000030
# ╟─08030008-0000-4000-8000-000000000050
# ╠═08030008-0000-4000-8000-000000000055
# ╟─08030008-0000-4000-8000-000000000058
# ╟─08030009-0000-4000-8000-000000000030
# ╟─0803000a-0000-4000-8000-000000000001
# ╠═0803000a-0000-4000-8000-000000000005
# ╠═0803000a-0000-4000-8000-000000000008
# ╠═0803000a-0000-4000-8000-000000000010
# ╠═0803000a-0000-4000-8000-000000000020
# ╟─0803000a-0000-4000-8000-000000000040
# ╟─0803000c-0000-4000-8000-000000000001
# ╟─0803000c-0000-4000-8000-000000000000
# ╟─0803000c-0000-4000-8000-00000000000a
# ╟─0803000c-0000-4000-8000-000000000002
# ╟─0803000c-0000-4000-8000-000000000003
# ╟─0803000d-0000-4000-8000-000000000001
# ╠═0803000d-0000-4000-8000-000000000004
# ╠═0803000d-0000-4000-8000-000000000005
# ╠═0803000d-0000-4000-8000-000000000010
# ╠═0803000d-0000-4000-8000-000000000015
# ╠═0803000d-0000-4000-8000-000000000020
# ╟─0803000d-0000-4000-8000-000000000030
# ╠═0803000c-0000-4000-8000-000000000010
# ╠═0803000c-0000-4000-8000-000000000011
# ╠═0803000c-0000-4000-8000-000000000012
# ╠═0803000c-0000-4000-8000-000000000020
# ╟─0803000c-0000-4000-8000-000000000030
# ╟─0803000c-0000-4000-8000-000000000031
# ╟─0803000e-0000-4000-8000-000000000001
