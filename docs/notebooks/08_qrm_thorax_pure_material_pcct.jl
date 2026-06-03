### A Pluto.jl notebook ###
# v0.19.0

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

# ╔═╡ 08010003-0000-4000-8000-000000000009
using LinearAlgebra: inv, cond, Diagonal, cholesky, Symmetric, I

# ╔═╡ 08010001-0000-4000-8000-000000000001
md"""
# 08 · QRM-Thorax Pure-Material PCCT VMI · **Full-Resolution True-Scan Reference**

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
                       2-basis VMI → Mono+ →
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
## Setup
"""

# ╔═╡ 08010003-0000-4000-8000-000000000010
import BasisSimulator as BS

# ╔═╡ 08010003-0000-4000-8000-000000000011
import CairoMakie as CM

# ╔═╡ 08010003-0000-4000-8000-000000000040
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

# ╔═╡ 08010003-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 08020001-0000-4000-8000-000000000001
md"""
## 1. `Phantom`: QRM-Thorax with 4 Pure-Material Rods

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
        bowtie_filter = :medium_body,   # SANDBOX: GE medium-body bowtie, turned ON to
                                 # exercise + VALIDATE the per-pixel-bowtie pipeline end to end
                                 # (forward → per-pixel I0 → noise → scatter → combine → Cong Φ).
                                 # The real Naeotom Alpha has NO bowtie — set :none for physical
                                 # fidelity once the machinery is validated here.  :medium_body
                                 # ≈ a 30 cm thorax; :large_body over-attenuates the lateral
                                 # body edges in this 32 cm FOV.  Any GE profile exercises the path.

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
## 3. `CTProtocol`: 140 kVp / 174 mA / 2.5 mm collimation

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
    collimation_mm = 2.5,
    additional_filters = [("Ti", 0.9)],
);

# ╔═╡ 08030003-0000-4000-8000-000000000001
md"""
## 4. `SimOptions` and `ReconOptions`

`fidelity = :pcct` switches the simulator into the photon-counting path
(per-bin sinograms + DRM + Compton scatter modeling + MC pile-up).
"""

# ╔═╡ 08030003-0000-4000-8000-000000000010
sim_opts = BS.SimOptions(
    fidelity = :pcct,
    seed = 1234,

    # ─── Inert for PCCT (flag exists but does nothing) ───
    use_fill_factor = false,
    use_detector_efficiency = false,
    use_optical_crosstalk = false,
    use_focal_spot = false,
    use_lag = false,
    use_heel_effect = false,

    # ─── Active for PCCT ───
    use_scatter = false,
    use_noise = false,
    use_pcct_pileup = false,
    pcct_noise_reduction = 0.0,
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
## 5. Forward Project (PCCT)

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
# === SLOW CELL (~6 min) — Forward-project only ===
# Calls `pcct_forward_project` once with per-pixel bowtie built at PROJECTION
# resolution (native when `bf > 1`, else binned — matches the kernel's
# `bt_sub` shape so `copyto!` doesn't silently misalign data).  Returns the
# raw kernel bins (scalar-I0 normalized) plus everything downstream cells
# need to (a) build a per-pixel I0 reference, (b) tweak the I0 correction
# formula, (c) build Cong's per-pixel ŵ — all without re-running.
#
# WARNING: bypasses `simulate!()` — currently requires
# `use_scatter = use_noise = use_pcct_pileup = false`.
sim_raw = let
    @info "Simulating: $(Int(protocol.kVp)) kVp / $(round(protocol.mA, digits = 1)) mA (PCCT 4-bin, EICT-mirror bowtie)…"
    ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)

    n_E         = length(ws.energies)
    n_bins_loc  = length(ws.I0_bins)
    bf          = scanner.binning_factor
    use_native  = ws.native_geom !== nothing && bf > 1
    proj_geom   = use_native ? ws.native_geom : ws.geom
    proj_shape  = (proj_geom.n_cols, proj_geom.n_rows, proj_geom.n_angles)
    center_col  = proj_shape[1] ÷ 2
    center_row  = proj_shape[2] ÷ 2

    bowtie_filter = BS.resolve_bowtie_filter(scanner.bowtie_filter)
    has_bowtie    = bowtie_filter !== nothing && bowtie_filter.name != "none"

    # Bowtie at PROJECTION resolution (matches kernel bt_sub shape)
    bt_proj = if has_bowtie
        Float32.(BS.compute_bowtie_attenuation_spectral(bowtie_filter, proj_geom, Float64.(ws.energies)))
    else
        ones(Float32, proj_shape[1], proj_shape[2], n_E)
    end

    # Relative bowtie buffer for ws_source_spectral.  W_matrix already has
    # B_center folded in (workspace.jl:296-313); kernel multiplies them →
    # full per-pixel B.  Avoids mutating ws.W_matrix_gpu.
    TILE_K = 16
    n_energies_padded = cld(n_E, TILE_K) * TILE_K
    bt_rel_padded = zeros(Float32, proj_shape[1], proj_shape[2], n_energies_padded)
    for e_idx in 1:n_E
        bt_c = has_bowtie ? max(bt_proj[center_col, center_row, e_idx], 1.0f-10) : 1.0f0
        @inbounds for r in 1:proj_shape[2], c in 1:proj_shape[1]
            bt_rel_padded[c, r, e_idx] = bt_proj[c, r, e_idx] / bt_c
        end
    end
    bowtie_relative_gpu = to_gpu(bt_rel_padded)

    pcct_sino = BS.pcct_forward_project(
        phantom.mask, ws.geom, ws.pcct_detector;
        energies = ws.energies, weights = ws.weights,
        materials = ws.mats,
        ws_bins = ws.bins, ws_μ_volume = ws.μ_volume, ws_sino_buf = ws.sino_buf,
        ws_scratch = ws.scratch,
        ws_thresholds_T = ws.thresholds_T,
        ws_η = ws.η, ws_R = ws.R, ws_R_energies = ws.R_energies,
        ws_I0_bins_norm = ws.I0_bins_norm,
        ws_μ_lut_cpu = ws.μ_lut_cpu, ws_μ_lut_gpu = ws.μ_lut_gpu,
        ws_μ_table = ws.μ_table,
        ws_source_positions = ws.geom_source_positions,
        ws_detector_centers = ws.geom_detector_centers,
        ws_detector_u = ws.geom_detector_u,
        ws_detector_v = ws.geom_detector_v,
        volume_extent = phantom.extent,
        native_geom = ws.native_geom,
        ws_native_bins = ws.native_bins,
        ws_native_sino_buf = ws.native_sino_buf,
        ws_native_source_positions = ws.native_geom_source_positions,
        ws_native_detector_centers = ws.native_geom_detector_centers,
        ws_native_detector_u = ws.native_geom_detector_u,
        ws_native_detector_v = ws.native_geom_detector_v,
        ws_μ_table_gpu = ws.μ_table_gpu,
        ws_W_matrix_gpu = ws.W_matrix_gpu,
        ws_outputs_flat = ws.outputs_flat,
        ws_native_outputs_flat = ws.native_outputs_flat,
        ws_source_spectral = bowtie_relative_gpu,
    )

    bins_kernel    = [Array(b) for b in pcct_sino.bins]
    I0_bins_scalar = copy(ws.I0_bins)
    W_with_center  = Array(ws.W_matrix_gpu)[1:n_E, :]
    energies       = copy(ws.energies)
    weights        = copy(ws.weights)
    geom           = ws.geom

    ws = nothing; pcct_sino = nothing; bowtie_relative_gpu = nothing
    GC.gc(true)
    (bins_kernel = bins_kernel,
     I0_bins_scalar = I0_bins_scalar,
     W_with_center = W_with_center,
     bt_proj = bt_proj,
     energies = energies,
     weights = weights,
     has_bowtie = has_bowtie,
     use_native = use_native,
     bf = bf,
     proj_shape = proj_shape,
     n_E = n_E,
     n_bins = n_bins_loc,
     geom = geom)
end;

# ╔═╡ 08030004-0000-4000-8000-00000000000a
# === FAST CELL — Per-pixel bowtie data at BINNED resolution ===
# Builds the I0 reference the binned `bins_kernel` should be normalized by.
# Operates on cached `sim_raw` outputs so it's free to re-evaluate.
#
#   W_no_bowtie[E, b] = recovered from W_with_center / B_center
#   I0_pp[c, r, b]    = Σ_E W_no_bowtie[E, b] · B_binned[c, r, E]
#
# Tweak this cell (e.g. swap bowtie filter, change bt resolution averaging)
# without re-running sim_raw.
bowtie_data = let
    n_E         = sim_raw.n_E
    n_bins_loc  = sim_raw.n_bins
    has_bowtie  = sim_raw.has_bowtie
    binned_geom = sim_raw.geom

    bowtie_filter = BS.resolve_bowtie_filter(scanner.bowtie_filter)

    bt_binned = if has_bowtie
        Float32.(BS.compute_bowtie_attenuation_spectral(bowtie_filter, binned_geom, Float64.(sim_raw.energies)))
    else
        ones(Float32, binned_geom.n_cols, binned_geom.n_rows, n_E)
    end

    # Recover W_no_bowtie by dividing out the center-pixel value the ws ctor
    # folded into W_matrix.  Use bt_proj (the proj-resolution bowtie that was
    # actually applied) to identify the center value the kernel saw.
    bt_proj  = sim_raw.bt_proj
    cc_p = sim_raw.proj_shape[1] ÷ 2
    cr_p = sim_raw.proj_shape[2] ÷ 2
    W_no_bowtie = similar(sim_raw.W_with_center)
    for e_idx in 1:n_E
        bt_c = has_bowtie ? max(bt_proj[cc_p, cr_p, e_idx], 1.0f-10) : 1.0f0
        for b in 1:n_bins_loc
            W_no_bowtie[e_idx, b] = sim_raw.W_with_center[e_idx, b] / bt_c
        end
    end

    bt_flat = reshape(bt_binned, binned_geom.n_cols * binned_geom.n_rows, n_E)
    I0_per_pixel = Float32.(reshape(bt_flat * W_no_bowtie, binned_geom.n_cols, binned_geom.n_rows, n_bins_loc))

    (bt_binned = bt_binned,
     W_no_bowtie = W_no_bowtie,
     I0_per_pixel = I0_per_pixel)
end;

# ╔═╡ 08030004-0000-4000-8000-00000000000b
# === FAST CELL — Per-pixel I0 correction ===
# Apply the per-pixel I0 normalization to the raw kernel bins.  Edit the
# formula here to experiment with different correction approaches.
#
# Default (air-zeroing):
#   bin_c[c,r,v,b] = bin_k[c,r,v,b] + log(I0_pp[c,r,b] / I0_scalar[b])
#
# At air, bin_k = -log(I0_pp/I0_scalar) (because forward applied per-pixel
# bowtie but kernel divided by scalar I0), so bin_c = 0.
sim_raw_corrected = let
    bins  = [copy(b) for b in sim_raw.bins_kernel]
    I0_pp = bowtie_data.I0_per_pixel
    for b in eachindex(bins)
        bin_arr = bins[b]
        I0_b    = Float32(sim_raw.I0_bins_scalar[b])
        I0_pp_b = view(I0_pp, :, :, b)
        @inbounds for v in 1:size(bin_arr, 3), r in 1:size(bin_arr, 2), c in 1:size(bin_arr, 1)
            bin_arr[c, r, v] += log(I0_pp_b[c, r] / I0_b)
        end
    end
    (bins_raw = bins,
     I0_bins = sim_raw.I0_bins_scalar,
     I0_per_pixel = bowtie_data.I0_per_pixel,
     bt_cpu = bowtie_data.bt_binned,
     pileup_S = nothing,
     geom = sim_raw.geom)
end;

# ╔═╡ 08030004-0000-4000-8000-000000000011
# Air-ray probe — diagnoses whether `I0_bins` matches the I0 baseline that
# `simulate!` actually wrote into the sinograms.  For a TRUE air ray:
#   N_recorded[b] = I0_bins[b]    (no attenuation, no pile-up, no scatter)
#   bin[b] = -log(N/I0_b) = -log(1) = 0
#
# So every bin should hit 0 on air rays.  Three failure modes this catches:
#   - min across all rays ≠ 0 per bin            → I0 baseline ≠ reality
#   - same offset every bin (e.g. all ≈ +0.1)    → global scaling error (benign for Cong)
#   - DIFFERENT offset per bin                   → per-bin I0 mismatch → Cong bias ⚠️
let
    bins_raw = sim_raw_corrected.bins_raw
    I0_bins  = sim_raw_corrected.I0_bins
    n_bins   = length(bins_raw)

    # Step 1: identify air-candidate rays by SUM across bins (air → all-bins small).
    sum_bins = sum(bins_raw)
    air_threshold = quantile(vec(sum_bins), 0.001)
    air_mask = sum_bins .<= air_threshold
    n_air = count(air_mask)
    @info "Air-ray probe — $(n_air) air-candidate pixels (smallest 0.1% by Σ-of-bins, threshold = $(round(air_threshold, digits = 4)))"

    rows = String[]
    push!(rows, "| bin | min(all) | air-ray mean | air-ray std | air-ray q_0.5 | I0_bins[b] |")
    push!(rows, "|-----|----------|--------------|-------------|---------------|------------|")
    for b in 1:n_bins
        bin_arr = bins_raw[b]
        min_b   = minimum(bin_arr)
        air_vs  = bin_arr[air_mask]
        μ_air   = mean(air_vs)
        σ_air   = std(air_vs)
        q_air   = quantile(vec(air_vs), 0.5)
        I0_b    = I0_bins[b]
        push!(rows,
            "| $(b) | $(round(min_b, digits = 5)) | $(round(μ_air, digits = 5)) | $(round(σ_air, digits = 5)) | $(round(q_air, digits = 5)) | $(round(I0_b, sigdigits = 4)) |"
        )
    end

    # Diagnostic verdict
    air_means = [mean(bins_raw[b][air_mask]) for b in 1:n_bins]
    spread = maximum(air_means) - minimum(air_means)
    mean_offset = mean(air_means)
    verdict = if all(abs.(air_means) .< 1.0e-3)
        "✅ baseline is correct — air rays land at 0 to within 1e-3"
    elseif spread < 1.0e-3
        "⚠️ uniform offset $(round(mean_offset, digits = 4)) across all bins — global I0 scaling (likely benign for Cong)"
    else
        "❌ per-bin offsets differ by $(round(spread, digits = 4)) — I0_bins[b] disagrees with forward-model baseline ⇒ Cong bias"
    end

    Markdown.parse(
        "**Air-ray probe**\n\n" *
        join(rows, "\n") * "\n\n" *
        "**Spread across bins:** $(round(spread, digits = 5))  ·  **Mean offset:** $(round(mean_offset, digits = 5))\n\n" *
        "**Verdict:** " * verdict
    )
end

# ╔═╡ 08030004-0000-4000-8000-00000000000e
# === FAST CELL — Forward noise (mirror of apply_pcct_noise! photon_counting.jl:780-853) ===
# Adds Poisson photon-counting noise (Gaussian approx for high counts, exact
# Poisson for low counts) to each bin.  Placed BEFORE pile-up forward to
# match source pipeline order (driver.jl: scatter → noise → pile-up).
#
# Unlike pile-up/scatter, the scalar-I0 trick does NOT preserve noise
# statistics correctly — Poisson variance ∝ N_actual, but using scalar I0
# scales the variance by I0_scalar/I0_pp which is wrong by a factor of
# √(I0_scalar/I0_pp).  So we use the *per-pixel physical* I0:
#     I0_phys_pp[c,r,b] = I0_per_pixel[c,r,b] · (I0_physics / Σ_b I0_bins)
# where I0_physics = `compute_detector_I0(geom, protocol, sum_w)` (actual photon
# flux per detector pixel per view) and Σ_b I0_bins normalizes the UN-normalized
# DRM-weighted I0_per_pixel into a per-bin FRACTION — matching the source
# `_compute_pcct_noise_I0` (photon_counting.jl:890-896) so I0_phys_pp[b] sums to
# I0_physics.  Dividing by a bare 1e6 here leaves the spectrum's raw fluence in
# the count, inflating N by ~total_raw and suppressing σ by ~√total_raw (≈1452×).
#
# Toggle is local — independent of sim_opts.use_noise / sim_opts.seed.
sim_noise_forward = let
    APPLY_NOISE_FORWARD = true      # per-bin Poisson noise ON (normal PCCT path —
                                    # inline mirror of source apply_pcct_noise!).
    NOISE_SEED          = 1234
    NOISE_REDUCTION     = 0.0       # 0 = raw physics; 0.7 ≈ QIR-3 vendor reduction
    # :count = Poisson-in-counts then −log (scanner-faithful, BUT carries the
    #          Jensen/log bias ≈ (1−nr)²/(2N̄) that blows up in photon-starved
    #          low bins → systematic, energy-dependent trend shift).
    # :log   = inject Gaussian noise directly in the line-integral domain with
    #          σ_p = (1−nr)/√N̄.  Same variance, ZERO bias by construction
    #          (drops the +1/(2N̄) term).  Gaussian linearization: correct
    #          variance everywhere, wrong tail shape only when N̄ ≲ 10.
    NOISE_DOMAIN        = :log
    # Noise shaping: per-bin variance = DRM Fano √(Var/mean) (sub-Poisson), plus a
    # TUNABLE inter-bin correlation ρ (the "rho knob").  The correlation is a PURE
    # coupling — the per-bin variance is unchanged (no added magnitude, unlike
    # common-mode) — so it doesn't inflate any channel (best RMSE) and is Cong-safe.
    #
    # By the noise theorem σ_HU(E)² = 1000²V_w + α²V_i + 2000·α·C_iw, with
    #   C_iw ∝ [−(q_Hp_Hσ_L² + q_Lp_Lσ_H²) + ρ·σ_Lσ_H·(q_Hp_L + q_Lp_H)],
    # ρ rotates the noise-vs-keV slope via α* = −1000·C_iw/V_i:
    #   ρ > 0 → C_iw less negative → α* SMALLER → toward MONOTONIC-DECREASING (goal).
    #   ρ < 0 → α* larger → monotonic-INCREASING (the "wrong direction").
    # SWEEP INTERBIN_CORR ∈ [−0.33, 1] (PSD range for uniform 4-bin ρ) to set the slope.
    USE_TAGUCHI_NOISE  = true
    INTERBIN_CORR      = 0.5     # ← the rho knob.  0 = independent (current U).  Sweep up.
    CW_REP_GCM2        = 20.0    # representative water path for the Fano estimate
    COMMON_MODE_SIGMA  = 0.0     # shared added-magnitude term (raises water floor) — keep 0

    bins = [copy(b) for b in sim_raw_corrected.bins_raw]

    if APPLY_NOISE_FORWARD
        I0_physics = BS.compute_detector_I0(sim_raw.geom, protocol, sum(sim_raw.weights))
        # `I0_per_pixel` (= ws.I0_bins, scaled by per-pixel bowtie) is the
        # UN-normalized DRM-weighted count `1e6 · Σ_E w·η·R[·,b]` — it sums over
        # bins to `1e6 · total_raw`, NOT 1e6.  The source noise model
        # (_compute_pcct_noise_I0, photon_counting.jl:890-896) distributes the
        # physical flux I0_physics across bins by the FRACTION raw[b]/total_raw,
        # so the per-pixel physical I0 must be normalized by Σ_b I0_bins.
        # Dividing by 1e6 here (the forward-projection W-matrix baseline) leaves
        # raw[b] un-normalized → counts inflated by total_raw (~2e6) → σ ∝ 1/√N
        # collapses to ~0.  Use Σ_b I0_bins so I0_phys_pp[b] == I0_physics·frac[b].
        I0_bins_sum = Float32(sum(sim_raw.I0_bins_scalar))
        scale = Float32(I0_physics) / I0_bins_sum

        rng = MersenneTwister(NOISE_SEED)
        nr_scale = Float32(1.0 - NOISE_REDUCTION)
        I0_pp = sim_raw_corrected.I0_per_pixel   # (n_col, n_row, n_bins) Float32
        sz = size(bins[1])
        nb = length(bins)

        # ── Taguchi noise-shaping: L = chol(C), C = DRM Fano+correlation matrix ──
        # C[b,b'] = Σ_E λ(E)·Cov_bins[E,b,b'] / √(N̄_b N̄_b')  (diag = per-bin Fano,
        # off-diag = inter-bin correlation), built once at a representative water
        # path.  Drawing δp_b = nr·(L·g)_b/√N̄_b reproduces the sub-Poisson variance
        # and inter-bin coupling.  C = I (naive independent Poisson) when toggle off.
        L_noise = if USE_TAGUCHI_NOISE
            mc_n   = BS.load_mc_response(BS.default_mc_drm_path())
            Eg_n   = mc_n.energies_keV
            idxL   = [findfirst(==(t), mc_n.thresholds_keV) for t in (20, 35, 55, 70)]
            Mdiff  = Float64[1 -1 0 0; 0 1 -1 0; 0 0 1 -1; 0 0 0 1]
            e_n, w_n = BS.resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner = scanner)
            pcd_n  = BS._build_pcct_detector(scanner)
            η_n    = BS.quantum_efficiency_vector(pcd_n.material, pcd_n.thickness_mm, e_n)
            ef_n   = Float64.(e_n)
            μw_n(E) = Float64(BS.compute_mass_μ_at_energy(BS.XA.Materials.water, E))
            Nbar_c = zeros(Float64, 4); Σ_c = zeros(Float64, 4, 4)
            for k in eachindex(ef_n)
                λ  = Float64(w_n[k]) * Float64(η_n[k]) * exp(-μw_n(ef_n[k]) * CW_REP_GCM2)
                iE = clamp(round(Int, ef_n[k] - Float64(Eg_n[1])) + 1, 1, length(Eg_n))
                Rc = mc_n.R_total[iE, idxL]
                Cc = mc_n.Cov_total[iE, idxL, idxL]
                Nbar_c .+= λ .* (Mdiff * Rc)
                Σ_c    .+= λ .* (Mdiff * Cc * Mdiff')
            end
            Cdrm = [Σ_c[b, bp] / sqrt(max(Nbar_c[b] * Nbar_c[bp], 1.0e-30)) for b in 1:4, bp in 1:4]
            fano_var = [max(Cdrm[b, b], 1.0e-6) for b in 1:4]   # per-bin Fano (variance, sub-Poisson)
            ρ = Float64(INTERBIN_CORR)
            # C = Fano on the diagonal + uniform tunable correlation ρ off-diagonal.
            # NO added magnitude — pure coupling.  (ρ=0 → diagonal/independent.)
            C = [b == bp ? fano_var[b] : ρ * sqrt(fano_var[b] * fano_var[bp]) for b in 1:4, bp in 1:4]
            C = (C .+ C') ./ 2 .+ 1.0e-4 * Matrix(I, 4, 4)   # symmetrize + tiny jitter for safe chol
            @info "[noise] per-bin Fano = $(round.(fano_var, digits=3)) (naive=1.0) · inter-bin ρ = $(round(ρ, digits=2)) (ρ>0 → toward monotonic-DECREASING)"
            Float32.(Matrix(cholesky(Symmetric(C)).L))
        else
            Matrix{Float32}(I, 4, 4)
        end

        # Per-view 2D scratch (per the memory budget — no full-sinogram scratch).
        g_b      = [Array{Float32}(undef, sz[1], sz[2]) for _ in 1:nb]   # iid N(0,1) per bin
        Nexp_b   = [Array{Float32}(undef, sz[1], sz[2]) for _ in 1:nb]
        I0pp_b   = [Float32.(view(I0_pp, :, :, b)) .* scale for b in 1:nb]
        g_cm     = Array{Float32}(undef, sz[1], sz[2])                   # shared common-mode field
        cm_σ     = Float32(COMMON_MODE_SIGMA)

        mid_v = sz[3] ÷ 2 + 1
        pre_mid = [copy(view(bins[b], :, :, mid_v)) for b in 1:nb]

        for v in 1:sz[3]
            for b in 1:nb
                randn!(rng, g_b[b])
                bin_v = view(bins[b], :, :, v)
                @. Nexp_b[b] = max(I0pp_b[b] * exp(-bin_v), 0.1f0)
            end
            g1 = g_b[1]; g2 = g_b[2]; g3 = g_b[3]; g4 = g_b[4]
            L = L_noise
            randn!(rng, g_cm)        # shared common-mode draw (same for all 4 bins → +corr)
            # δp_b = nr·(L·g)_b/√N̄_b  +  cm_σ·g_cm  (common-mode added to every bin)
            if NOISE_DOMAIN === :count
                # independent count-domain Poisson (Taguchi shaping not applied here)
                for b in 1:nb
                    bin_v = view(bins[b], :, :, v)
                    @. bin_v = -log(max(Nexp_b[b] + nr_scale * sqrt(Nexp_b[b]) * g_b[b], 1.0f0) / I0pp_b[b]) + cm_σ * g_cm
                end
            else
                bv1 = view(bins[1], :, :, v); bv2 = view(bins[2], :, :, v)
                bv3 = view(bins[3], :, :, v); bv4 = view(bins[4], :, :, v)
                @. bv1 = bv1 + nr_scale * (L[1,1]*g1)                               / sqrt(Nexp_b[1]) + cm_σ * g_cm
                @. bv2 = bv2 + nr_scale * (L[2,1]*g1 + L[2,2]*g2)                   / sqrt(Nexp_b[2]) + cm_σ * g_cm
                @. bv3 = bv3 + nr_scale * (L[3,1]*g1 + L[3,2]*g2 + L[3,3]*g3)       / sqrt(Nexp_b[3]) + cm_σ * g_cm
                @. bv4 = bv4 + nr_scale * (L[4,1]*g1 + L[4,2]*g2 + L[4,3]*g3 + L[4,4]*g4) / sqrt(Nexp_b[4]) + cm_σ * g_cm
            end
        end

        @info "Noise applied [$(NOISE_DOMAIN)-domain, Taguchi=$(USE_TAGUCHI_NOISE)] — I0_phys = $(round(I0_physics, sigdigits = 3)) ph/dexel/view, nr_scale = $(nr_scale)"
        for b in 1:nb
            σ_noise = std(vec(view(bins[b], :, :, mid_v) .- pre_mid[b]))
            @info "  bin $(b): per-pixel log-noise σ (mid-view) = $(round(σ_noise, digits = 4))"
        end
    else
        @info "Noise skipped (APPLY_NOISE_FORWARD = false)"
    end

    (bins_raw = bins,
     I0_bins = sim_raw_corrected.I0_bins,
     I0_per_pixel = sim_raw_corrected.I0_per_pixel,
     bt_cpu = sim_raw_corrected.bt_cpu,
     geom = sim_raw_corrected.geom)
end;

# ╔═╡ 08030004-0000-4000-8000-00000000000c
# === FAST CELL — Forward pile-up (mirror of driver.jl:223-257) ===
# Builds the MC pile-up migration matrix S using the air-rate count rate
# (same recipe as workspace.jl:336-348), then applies forward pile-up to the
# (now-noisy) bins.  Operates inline so the existing `sim_pileup` inverse
# correction below picks it up via `pileup_S`.
#
# Math sanity: bins from `sim_noise_forward` are per-pixel-normalized
# (bin = -log(N / I0_pp)), but we apply pile-up using the scalar `I0_bins`.
# That's fine because S is linear:
#     N_func     = I0_scalar · exp(-bin_pp)  = (I0_scalar/I0_pp) · N_actual
#     r_func     = S · N_func                = (I0_scalar/I0_pp) · S·N_actual
#     bin_new    = -log(r_func / I0_scalar)  = -log(S·N_actual / I0_pp)
# i.e. the I0_scalar factor cancels out and the output is correctly
# per-pixel-normalized.  Same algebra applies to apply_pcct_pileup_correction!.
#
# Toggle: `APPLY_PILEUP_FORWARD` is local to this cell — independent of
# `sim_opts.use_pcct_pileup` since we bypass `simulate!()`.
sim_pileup_forward = let
    APPLY_PILEUP_FORWARD = true

    bins = [copy(b) for b in sim_noise_forward.bins_raw]
    I0_bins = sim_noise_forward.I0_bins
    n_bins_loc = length(bins)

    pileup_S = if APPLY_PILEUP_FORWARD
        # Build MC pile-up matrix (mirror of workspace.jl:336-348)
        pcct_det = BS._build_pcct_detector(scanner)
        I0_air = BS.compute_detector_I0(sim_raw.geom, protocol, sum(sim_raw.weights))
        time_per_view = protocol.rotation_time / protocol.views
        bf = sim_raw.bf
        count_rate_per_dexel = (I0_air / Float64(bf * bf)) / time_per_view
        τ_ns = Float64(pcct_det.dead_time_ns)
        w_norm = Float64.(sim_raw.weights) ./ sum(Float64.(sim_raw.weights))
        @info "Pile-up forward: count rate = $(round(count_rate_per_dexel, sigdigits = 3)) cps/dexel, τ = $(τ_ns) ns"
        S = BS.compute_mc_pileup_matrix(
            pcct_det.energy_thresholds_keV,
            w_norm, Float64.(sim_raw.energies),
            count_rate_per_dexel, τ_ns;
            n_trials = 5000, seed = 42,
        )

        # Forward pile-up (mirror of driver.jl:228-256, 4-bin specialization)
        n_bins_loc == 4 || error("Pile-up forward currently specialized to 4 bins; got $(n_bins_loc)")
        eps_p = Float32(1.0e-10)
        b1, b2, b3, b4 = bins[1], bins[2], bins[3], bins[4]
        I01 = Float32(I0_bins[1]); I02 = Float32(I0_bins[2])
        I03 = Float32(I0_bins[3]); I04 = Float32(I0_bins[4])
        S11 = Float32(S[1, 1])
        S21 = Float32(S[2, 1]); S22 = Float32(S[2, 2])
        S31 = Float32(S[3, 1]); S32 = Float32(S[3, 2]); S33 = Float32(S[3, 3])
        S41 = Float32(S[4, 1]); S42 = Float32(S[4, 2]); S43 = Float32(S[4, 3]); S44 = Float32(S[4, 4])
        @inbounds for idx in eachindex(b1)
            c1 = I01 * exp(-b1[idx])
            c2 = I02 * exp(-b2[idx])
            c3 = I03 * exp(-b3[idx])
            c4 = I04 * exp(-b4[idx])
            r1 = S11 * c1
            r2 = S21 * c1 + S22 * c2
            r3 = S31 * c1 + S32 * c2 + S33 * c3
            r4 = S41 * c1 + S42 * c2 + S43 * c3 + S44 * c4
            b1[idx] = -log(max(r1, eps_p) / I01)
            b2[idx] = -log(max(r2, eps_p) / I02)
            b3[idx] = -log(max(r3, eps_p) / I03)
            b4[idx] = -log(max(r4, eps_p) / I04)
        end
        @info "Pile-up forward applied; S diagonal = $(round.([S[i,i] for i in 1:4], digits = 3))  (column sums = $(round.([sum(S[:,j]) for j in 1:4], digits = 3)))"
        S
    else
        @info "Pile-up forward skipped (APPLY_PILEUP_FORWARD = false)"
        nothing
    end

    (bins_raw = bins,
     I0_bins = sim_noise_forward.I0_bins,
     I0_per_pixel = sim_noise_forward.I0_per_pixel,
     bt_cpu = sim_noise_forward.bt_cpu,
     pileup_S = pileup_S,
     geom = sim_noise_forward.geom)
end;

# ╔═╡ 08030004-0000-4000-8000-000000000012
# Pile-up correction — applies the inverse of `sim_pileup_forward`'s S.
# No-op when `sim_pileup_forward.pileup_S === nothing`.
sim_pileup = let
    bins = [copy(b) for b in sim_pileup_forward.bins_raw]
    if sim_pileup_forward.pileup_S !== nothing
        BS.apply_pcct_pileup_correction!(bins, sim_pileup_forward.I0_bins, sim_pileup_forward.pileup_S)
        @info "Pile-up correction applied"
    else
        @info "Pile-up correction skipped (no pileup_S to invert)"
    end
    (bins = bins,
     I0_bins = sim_pileup_forward.I0_bins,
     I0_per_pixel = sim_pileup_forward.I0_per_pixel,
     bt_cpu = sim_pileup_forward.bt_cpu,
     geom = sim_pileup_forward.geom)
end;

# ╔═╡ 08030004-0000-4000-8000-000000000013
# === FAST CELL — Forward scatter injection (per-pixel I0, mirror of driver.jl:144-177) ===
# Computes the combined primary log-line-integral, runs Ohnesorge spatial
# scatter convolution, computes per-bin scatter fractions via spectrum × η ×
# MC-DRM, and injects scatter into each bin additively in count space.
#
# PER-PIXEL I0 (bowtie-correct).  Unlike pile-up (which is an exact S⁻¹·S
# identity round-trip, so the I0 cancels regardless), scatter forward + the
# correction below do NOT cancel — the correction re-estimates from
# primary+scatter, leaving a realistic residual.  With a bowtie active the
# per-pixel air reference `I0_pp[c,r,b]` varies ~6.8× across columns, so the
# count conversions MUST use it (not a scalar `I0_bins`): otherwise the
# Ohnesorge combined-sinogram is built on a per-pixel-distorted primary and
# the residual becomes per-pixel-wrong (inconsistent with the per-pixel Cong Φ).
#   N_primary[c,r,v,b] = I0_pp[c,r,b] · exp(-bin[c,r,v,b])
#   combined           = -log( Σ_b N_primary / Σ_b I0_pp[c,r,b] )
#   N_scatter          = scatter_field · (Σ_b I0_pp[c,r,b]) · frac[b]
#
# Toggle: `APPLY_SCATTER_FORWARD` is local — does NOT read `sim_opts.use_scatter`
# so that flipping the simopt flag doesn't trigger Pluto to re-run sim_raw.
sim_scatter_forward = let
    APPLY_SCATTER_FORWARD = true

    bins  = [copy(b) for b in sim_pileup.bins]
    I0_pp = sim_pileup.I0_per_pixel               # (n_col, n_row, n_bins) per-pixel air ref

    if APPLY_SCATTER_FORWARD
        eps_f = Float32(1.0e-10)
        n_col, n_row, n_view = size(bins[1])
        n_b = length(bins)

        # Per-pixel total air reference Σ_b I0_pp[c,r,b]  (replaces scalar I0_total)
        I0_total_pp = dropdims(sum(I0_pp; dims = 3); dims = 3)        # (n_col, n_row)

        # Step 1: combined primary sinogram (per-pixel-I0 normalized)
        combined = zeros(Float32, size(bins[1]))
        for b in 1:n_b
            I0_pp_b = reshape(view(I0_pp, :, :, b), n_col, n_row, 1)   # broadcast over views
            bin_b   = bins[b]
            @. combined += I0_pp_b * exp(-bin_b)
        end
        I0_total_pp_3d = reshape(I0_total_pp, n_col, n_row, 1)
        @. combined = -log(max(combined, eps_f) / I0_total_pp_3d)

        # Step 2: Ohnesorge spatial scatter field (on the per-pixel-normalized primary)
        voxel_size_mm = phantom_cpu.voxel_size .* 10.0
        phantom_diam_cm = BS.estimate_phantom_diameter_cm(phantom_cpu.mask, voxel_size_mm)
        scatter_model = BS.geometry_aware_scatter_model(scanner; phantom_diameter_cm = phantom_diam_cm)
        scatter_field = similar(combined)
        BS.estimate_scatter_field!(scatter_field, combined, scatter_model)

        # Step 3: per-bin scatter fractions (spectrum × η × MC-DRM)
        e_full, w_full = BS.resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner = scanner)
        pcct_det = BS._build_pcct_detector(scanner)
        kVp_val = Float64(maximum(e_full))
        R_mat = BS.compute_mc_drm(pcct_det, kVp_val)
        η_vec = BS.quantum_efficiency_vector(pcct_det.material, pcct_det.thickness_mm, e_full)
        ew = BS.compute_scatter_energy_weights(Float64.(e_full))
        scatter_fracs = BS.compute_scatter_bin_weights(
            Float64.(e_full), Float64.(w_full), ew, Float64.(η_vec), R_mat, kVp_val,
        )

        # Step 4: inject scatter into each bin (additive in counts), per-pixel I0
        for b in 1:n_b
            bin_b   = bins[b]
            I0_pp_b = view(I0_pp, :, :, b)                  # (n_col, n_row)
            frac    = Float32(scatter_fracs[b])
            @inbounds for v in 1:n_view, r in 1:n_row, c in 1:n_col
                N_primary = I0_pp_b[c, r] * exp(-bin_b[c, r, v])
                N_scatter = scatter_field[c, r, v] * I0_total_pp[c, r] * frac
                N_total   = N_primary + max(N_scatter, Float32(0))
                bin_b[c, r, v] = -log(max(N_total, eps_f) / I0_pp_b[c, r])
            end
        end
        @info "Scatter forward applied [per-pixel I0] (fracs = $(round.(scatter_fracs, digits = 4)))"
    else
        @info "Scatter forward skipped (APPLY_SCATTER_FORWARD = false)"
    end

    (bins = bins,
     I0_bins = sim_pileup.I0_bins,
     I0_per_pixel = sim_pileup.I0_per_pixel,
     bt_cpu = sim_pileup.bt_cpu,
     geom = sim_pileup.geom)
end;

# ╔═╡ 08030004-0000-4000-8000-000000000014
# Decoupled per-bin scatter correction (model-based Ohnesorge re-estimation,
# scaled by spectrum × η × MC-DRM bin weights).  Operates on the bins from
# `sim_scatter_forward` (which may or may not have scatter injected).
# Re-estimates the scatter field from the current bin data (primary + scatter,
# if injected) — same realistic behavior a clinical scanner exhibits.
#
# Toggle is local (mirrors `APPLY_SCATTER_FORWARD` above by default) so the
# cell doesn't reference `sim_opts` and re-trigger sim_raw on flag flips.
sim_bins = let
    APPLY_SCATTER_CORRECTION = true

    bins  = [copy(b) for b in sim_scatter_forward.bins]
    I0_pp = sim_scatter_forward.I0_per_pixel       # (n_col, n_row, n_bins) per-pixel air ref

    if APPLY_SCATTER_CORRECTION
        eps_f = Float32(1.0e-10)
        n_col, n_row, n_view = size(bins[1])
        n_b = length(bins)

        I0_total_pp = dropdims(sum(I0_pp; dims = 3); dims = 3)        # (n_col, n_row)

        # Re-estimate the scatter field from the CURRENT bins (primary+scatter),
        # per-pixel-I0 normalized — identical recipe to the forward, so the only
        # net effect is the realistic primary-vs-(primary+scatter) re-estimation gap.
        combined = zeros(Float32, size(bins[1]))
        for b in 1:n_b
            I0_pp_b = reshape(view(I0_pp, :, :, b), n_col, n_row, 1)
            bin_b   = bins[b]
            @. combined += I0_pp_b * exp(-bin_b)
        end
        I0_total_pp_3d = reshape(I0_total_pp, n_col, n_row, 1)
        @. combined = -log(max(combined, eps_f) / I0_total_pp_3d)

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

        for b in 1:n_b
            bin_b   = bins[b]
            I0_pp_b = view(I0_pp, :, :, b)
            frac    = Float32(scatter_fracs[b])
            @inbounds for v in 1:n_view, r in 1:n_row, c in 1:n_col
                N_measured  = I0_pp_b[c, r] * exp(-bin_b[c, r, v])
                N_scatter   = scatter_field[c, r, v] * I0_total_pp[c, r] * frac
                N_corrected = N_measured - max(N_scatter, Float32(0))
                bin_b[c, r, v] = -log(max(N_corrected, eps_f) / I0_pp_b[c, r])
            end
        end
        @info "Scatter correction applied [per-pixel I0]"
    else
        @info "Scatter correction skipped (APPLY_SCATTER_CORRECTION = false)"
    end

    (bins = bins, I0_bins = sim_scatter_forward.I0_bins,
     I0_per_pixel = sim_scatter_forward.I0_per_pixel,
     bt_cpu = sim_scatter_forward.bt_cpu,
     geom = sim_scatter_forward.geom)
end;

# ╔═╡ 08030004-0000-4000-8000-000000000025
# Resample the phantom labels onto the recon grid via BS's affine
# round-trip — used downstream for ROI construction in recon coords.
phantom_in_recon = BS.resample_to_recon(
    phantom_cpu, sim_bins.geom, recon_opts.matrix_size; method = :nearest,
);

# ╔═╡ 08030004-0000-4000-8000-000000000040
let
    n_row = size(sim_bins.bins[1], 2)
    mid_r = n_row ÷ 2 + 1

    bin_titles = ("Bin 1", "Bin 2", "Bin 3", "Bin 4")
    bin_subs = ("20 – 35 keV", "35 – 55 keV", "55 – 70 keV", "> 70 keV")
    slices = [permutedims(sim_bins.bins[k][:, mid_r, :], (2, 1)) for k in 1:4]

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

# ╔═╡ 08030004-0000-4000-8000-000000000050
md"""
## 5b. Detector Physics: What the Monte-Carlo DRM Encodes

The four raw bins above are **not** clean incident-energy windows.  Each
count is registered at a *distorted* (usually lower) energy than the photon
that produced it, because the bundled CdTe response matrix
`R(E_inc, threshold)` (`cdte_response_v4.jls`, loaded by
`BS.load_mc_response`) folds in the full photon-counting-detector physics:

1. **Quantum efficiency / transmission rolloff** — finite 1.6 mm CdTe; a
   growing fraction of high-energy photons pass straight through uncounted
   (η falls, and counts ≥20 keV drop *below* 1 by 120–140 keV).
2. **Photoelectric · Compton · Rayleigh transport** — Compton scatter
   deposits only partial energy, the scattered photon escapes or reabsorbs
   elsewhere → a low-energy continuum under every photopeak.
3. **K-fluorescence escape** — photoabsorption in Cd (Kα ≈ 23 keV) or Te
   (Kα ≈ 27, Kβ ≈ 31 keV) leaves the pixel; the photopeak loses E_K and a
   spurious escape count appears ≈23–31 keV lower (or in a neighbour pixel).
4. **Charge sharing** (Stierstorfer/Dreier 2018) — the charge cloud
   diffuses and Coulomb-repels across pixel boundaries (3×3 erf splitting),
   so **one photon is counted as several sub-threshold events**.  Signature:
   counts ≥20 keV *exceed* 1.0 per incident photon (≈1.2 at 60–80 keV), and
   the struck-pixel fraction falls from ≈90 % at 30 keV to ≈67 % at 140 keV.
5. **Fano noise** — sub-Poisson e–h-pair generation sets the intrinsic
   energy-resolution floor (photopeak broadening).
6. **Electronic noise** — per-pixel Gaussian on the charge signal; broadens
   peaks and scatters counts across the lowest threshold.
7. **Incomplete charge collection (CdTe hole tailing)** — poor hole μτ gives
   depth-dependent charge loss → a low-energy tail on the photopeak.
8. **Pulse pile-up** *(flux-dependent, handled separately in the pile-up
   forward path)* — two photons within the dead time sum into one count
   (energy up-shift) or are lost (count loss).
9. **Threshold comparison / binning** — counts are cumulative (≥ threshold);
   differencing adjacent thresholds gives the bins, but after (1)–(8) the
   measured bins no longer map to incident-energy windows.

**Net consequence (the figure below):** high-energy photons leak *down* into
the low bins, so the **low (1+2)** and **high (3+4)** channel spectra
*overlap* far more than their ideal energy windows.  That overlap is exactly
the loss of spectral separation that ill-conditions the Cong decomposition,
inflates the iodine basis noise, and produces the VMI noise U-shape.  The
plan: **correct `R` (spectral unfolding) on the four bins to recover the
ideal energy-windowed counts *before* the 4→2 combine**, restoring
separation so everything downstream (Cong → FBP → VMI) inherits it for free.
The panels here are the *diagnosis*; the correction comes next.
"""

# ╔═╡ 08030004-0000-4000-8000-000000000060
let
    # ── Pull the response matrix + raw MC LUT straight from the bundle ──────
    e, w    = BS.resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner = scanner)
    pcct_det = BS._build_pcct_detector(scanner)
    kVp_val  = Float64(maximum(e))
    D        = BS.compute_mc_drm(pcct_det, kVp_val)          # [200 × 4] P(bin | E_inc)
    η_vec    = BS.quantum_efficiency_vector(pcct_det.material, pcct_det.thickness_mm, e)
    mc       = BS.load_mc_response(BS.default_mc_drm_path())  # raw LUT (count-mult, per-pixel)

    Egrid = collect(range(1.0, kVp_val, length = size(D, 1)))
    ef    = Float64.(e); ηf = Float64.(η_vec)

    bin_colors  = [:dodgerblue, :seagreen, :darkorange, :firebrick]
    bin_labels  = ["Bin 1: 20–35", "Bin 2: 35–55", "Bin 3: 55–70", "Bin 4: >70 keV"]
    bin_windows = [(20.0, 35.0), (35.0, 55.0), (55.0, 70.0), (70.0, kVp_val)]

    # Panel B: two-channel detector RESPONSE (sensitivity), source-spectrum
    # independent so the tungsten characteristic lines don't swamp the axis.
    # This is exactly what the Cong low/high channels integrate against.
    resp_low     = D[:, 1] .+ D[:, 2]
    resp_high    = D[:, 3] .+ D[:, 4]
    ideal_low_E  = [(20.0 <= E < 55.0) ? 1.0 : 0.0 for E in Egrid]
    ideal_high_E = [(E >= 55.0)        ? 1.0 : 0.0 for E in Egrid]

    # Panels C/D: count multiplication + per-pixel charge-sharing leakage
    i20 = findfirst(==(20), mc.thresholds_keV)
    mcE = Float64.(mc.energies_keV)
    counts_per_photon = mc.R_total[:, i20]
    center_frac = Float64[]
    for iE in eachindex(mcE)
        tot = sum(@view mc.R_perpixel[iE, :, :, i20])
        push!(center_frac, tot > 1.0e-3 ? 100.0 * mc.R_perpixel[iE, 2, 2, i20] / tot : NaN)
    end

    fig = CM.Figure(size = (1240, 1060))
    ak  = (titlesize = 28, subtitlesize = 20, xlabelsize = 22, ylabelsize = 22,
           xticklabelsize = 16, yticklabelsize = 16)

    # ── A: bin energy response (ideal window shaded, MC curve leaks across) ──
    axA = CM.Axis(fig[1, 1]; title = "Bin Energy Response",
        subtitle = "P(count in bin | incident E) — ideal = shaded window",
        xlabel = "Incident energy (keV)", ylabel = "Detection probability", ak...)
    for b in 1:4
        lo, hi = bin_windows[b]
        CM.vspan!(axA, lo, hi; color = (bin_colors[b], 0.10))
        CM.lines!(axA, Egrid, D[:, b]; color = bin_colors[b], linewidth = 2.5, label = bin_labels[b])
    end
    CM.axislegend(
        axA;
        position = :rt,
        labelsize = 15,
        framevisible = true,
        framecolor = (:black, 0.15),
        backgroundcolor = (:white, 0.85)
    )

    # ── B: two-channel spectral overlap (the conditioning story) ────────────
    axB = CM.Axis(fig[1, 2]; title = "Two-Channel Spectral Overlap",
        subtitle = "channel sensitivity — MC (solid) leaks past ideal window (dashed)",
        xlabel = "Incident energy (keV)", ylabel = "Channel response  Σ P(bin | E)", ak...)
    CM.lines!(axB, Egrid, resp_low;     color = :dodgerblue, linewidth = 2.5, label = "Low (1+2) — MC")
    CM.lines!(axB, Egrid, resp_high;    color = :firebrick,  linewidth = 2.5, label = "High (3+4) — MC")
    CM.lines!(axB, Egrid, ideal_low_E;  color = :dodgerblue, linewidth = 2.0, linestyle = :dash, label = "Low — ideal")
    CM.lines!(axB, Egrid, ideal_high_E; color = :firebrick,  linewidth = 2.0, linestyle = :dash, label = "High — ideal")
    CM.vlines!(axB, [33.0]; color = :black, linestyle = :dot, linewidth = 1.5)  # iodine K-edge
    CM.axislegend(
        axB;
        position = :rb,
        labelsize = 15,
        framevisible = true,
        framecolor = (:black, 0.15),
        backgroundcolor = (:white, 0.85)
    )

    # ── C: charge-sharing count multiplication + QE rolloff ─────────────────
    axC = CM.Axis(fig[2, 1]; title = "Charge-Sharing Count Multiplication",
        subtitle = "registered counts ≥20 keV per incident photon",
        xlabel = "Incident energy (keV)", ylabel = "Counts / photon  ·  η", ak...)
    CM.lines!(axC, mcE, counts_per_photon; color = :purple, linewidth = 2.5, label = "counts/photon (≥20 keV)")
    CM.lines!(axC, ef, ηf;                 color = :teal,   linewidth = 2.5, label = "quantum efficiency η(E)")
    CM.hlines!(axC, [1.0]; color = :gray, linestyle = :dash, linewidth = 1.5)  # one-count reference
    CM.axislegend(
        axC;
        position = :rb,
        labelsize = 15,
        framevisible = true,
        framecolor = (:black, 0.15),
        backgroundcolor = (:white, 0.85)
    )

    # ── D: charge-sharing spatial leakage (struck-pixel fraction) ───────────
    axD = CM.Axis(fig[2, 2]; title = "Charge-Sharing Spatial Leakage",
        subtitle = "fraction of counts in the struck pixel (rest → neighbours)",
        xlabel = "Incident energy (keV)", ylabel = "Struck-pixel fraction (%)", ak...)
    CM.lines!(axD, mcE, center_frac; color = :crimson, linewidth = 2.5)
    CM.ylims!(axD, 55, 100)

    fig
end

# ╔═╡ 08030005-0000-4000-8000-000000000001
md"""
## 6. Spectral-Distortion Correction & Bin Combine

Before combining, **undo the CdTe charge-sharing distortion** so each bin maps
back to its true-energy window — the projection-domain analogue of the clinical
coincidence-response-matrix correction (Trinci *et al.* 2025).  Charge sharing
and K-escape only move counts **down** in energy, so the bin↔true-window
response is **upper-triangular**:

```
F[b,a] = fraction of true-window-a counts measured in bin b   (b ≤ a; F=0 for b>a)
       = A[b,a] / Σ_{b'≤a} A[b',a],   A[b,a] = Σ_{E ∈ window a} W[E,b]
```

A triangular system is solved **stably by backward substitution** (highest
window first — it has no contamination from above — then subtract its
spill-down from each lower window):

```
N_true[4] =  N_meas[4] / F[4,4]
N_true[a] = (N_meas[a] − Σ_{a'>a} F[a,a']·N_true[a']) / F[a,a],   clamp ≥ 0
```

No matrix inverse, no noise blow-up.  The same backward substitution is applied
to the per-pixel air reference, so air rays stay at 0.  We then combine the
de-migrated windows:

* **Low**  = true windows **1 + 2** (20 – 55 keV)
* **High** = true windows **3 + 4** (> 55 keV)

Because the windows are now de-contaminated, the §7 Cong response `Φ` becomes
the **ideal energy window** `Wtot(E)·1[E ∈ channel]` instead of the smeared
`Σ_b R(E,b)`.
"""

# ╔═╡ 08030005-0000-4000-8000-000000000005
# === Charge-sharing migration F (upper-triangular) ===
# Built from the SAME forward bin spectra W the kernel used (`bowtie_data.W_no_bowtie`),
# so the correction is self-consistent with the forward η/DRM convention.
# F[b,a] = P(a true-window-a count is measured in bin b).  Down-migration only
# (b ≤ a), so F is upper-triangular → backward-substitution inverts it stably.
correction = let
    W   = Float64.(bowtie_data.W_no_bowtie)   # [n_E × n_bins] forward bin spectra
    eN  = Float64.(sim_raw.energies)          # [n_E]
    thr = Float64.(scanner.energy_thresholds) # [20, 35, 55, 70]
    nbn = size(W, 2)
    winid(E) = E < thr[1] ? 0 :
               (E < thr[2] ? 1 : (E < thr[3] ? 2 : (E < thr[4] ? 3 : 4)))

    A = zeros(Float64, nbn, nbn)              # A[b,a] = Σ_{E ∈ true window a} W[E,b]
    @inbounds for i in eachindex(eN)
        a = winid(eN[i]); a == 0 && continue
        for b in 1:nbn
            A[b, a] += W[i, b]
        end
    end
    # Upper-triangular migration: keep only the physical down-migration (b ≤ a),
    # normalize each true-window column so Σ_{b≤a} F[b,a] = 1.
    F = zeros(Float64, nbn, nbn)
    for a in 1:nbn
        col = sum(A[b, a] for b in 1:a)
        for b in 1:a
            F[b, a] = A[b, a] / max(col, 1.0e-30)
        end
    end
    @info "[charge-sharing correction] upper-triangular migration F (backward-substitution per ray)"
    @info "  F diagonal (counts staying in own window) = $(round.([F[b, b] for b in 1:nbn], digits = 3))"

    (F = Float32.(F), A = A, W = W, eN = eN, thr = thr)
end;

# ╔═╡ 08030005-0000-4000-8000-000000000008
# === Correct (charge-sharing backward-substitution) AND combine → low/high ===
# Per ray: N_meas[b] = I0_pp[c,r,b]·exp(-bins[b]).  Backward-substitute the
# upper-triangular F (highest window first) to de-migrate counts into their true
# windows, clamping ≥0 (the starved 20–35 keV window simply clamps to ~0 instead
# of blowing up).  The SAME backward-substitution on the per-pixel air reference
# keeps air rays at 0.  Then low = windows 1+2, high = windows 3+4.
sim_corrected = let
    F     = correction.F                  # Float32 [4×4] upper-triangular
    bins  = sim_bins.bins                 # 4 × (n_col, n_row, n_view) log line integrals
    I0_pp = sim_bins.I0_per_pixel         # (n_col, n_row, 4)
    sz    = size(bins[1])
    eps_f = Float32(1.0e-10)

    # Backward substitution: de-migrate a 4-vector of measured counts → true-window
    # counts (clamp ≥ 0).  F upper-triangular ⇒ solve window 4 → 1.
    @inline function demigrate(n1, n2, n3, n4)
        t4 = n4 / F[4, 4]
        t3 = (n3 - F[3, 4] * t4) / F[3, 3]
        t2 = (n2 - F[2, 3] * t3 - F[2, 4] * t4) / F[2, 2]
        t1 = (n1 - F[1, 2] * t2 - F[1, 3] * t3 - F[1, 4] * t4) / F[1, 1]
        (max(t1, 0.0f0), max(t2, 0.0f0), max(t3, 0.0f0), max(t4, 0.0f0))
    end

    # Per-pixel corrected air reference (low = w1+w2, high = w3+w4)
    I0_lo_pp = Array{Float32}(undef, sz[1], sz[2])
    I0_hi_pp = Array{Float32}(undef, sz[1], sz[2])
    @inbounds for r in 1:sz[2], c in 1:sz[1]
        a1, a2, a3, a4 = demigrate(I0_pp[c, r, 1], I0_pp[c, r, 2], I0_pp[c, r, 3], I0_pp[c, r, 4])
        I0_lo_pp[c, r] = a1 + a2
        I0_hi_pp[c, r] = a3 + a4
    end

    sino_low  = Array{Float32}(undef, sz)
    sino_high = Array{Float32}(undef, sz)
    b1 = bins[1]; b2 = bins[2]; b3 = bins[3]; b4 = bins[4]
    @inbounds for v in 1:sz[3], r in 1:sz[2], c in 1:sz[1]
        n1 = I0_pp[c, r, 1] * exp(-b1[c, r, v])
        n2 = I0_pp[c, r, 2] * exp(-b2[c, r, v])
        n3 = I0_pp[c, r, 3] * exp(-b3[c, r, v])
        n4 = I0_pp[c, r, 4] * exp(-b4[c, r, v])
        t1, t2, t3, t4 = demigrate(n1, n2, n3, n4)
        sino_low[c, r, v]  = -log(max(t1 + t2, eps_f) / max(I0_lo_pp[c, r], eps_f))
        sino_high[c, r, v] = -log(max(t3 + t4, eps_f) / max(I0_hi_pp[c, r], eps_f))
    end

    (sino_low = sino_low, sino_high = sino_high,
     I0_lo_pp = I0_lo_pp, I0_hi_pp = I0_hi_pp,
     bt_cpu = sim_bins.bt_cpu, geom = sim_bins.geom)
end;

# ╔═╡ 08030005-0000-4000-8000-00000000000b
# PRE-correction low/high — naive I0-weighted combine of the RAW 4 bins (1+2→low,
# 3+4→high, NO charge-sharing correction).  Stacked right above the corrected
# channels so you can see exactly what the backward-substitution changed.  Mid-row
# slice, 2D scratch only (memory budget).
let
    eps_f = Float32(1.0e-10)
    bins  = sim_bins.bins
    I0_pp = sim_bins.I0_per_pixel
    ncol  = size(bins[1], 1); nrow = size(bins[1], 2); nview = size(bins[1], 3)
    mid_r = nrow ÷ 2 + 1

    I0lo = Float32[I0_pp[c, mid_r, 1] + I0_pp[c, mid_r, 2] for c in 1:ncol]
    I0hi = Float32[I0_pp[c, mid_r, 3] + I0_pp[c, mid_r, 4] for c in 1:ncol]
    raw_lo = Array{Float32}(undef, ncol, nview)
    raw_hi = Array{Float32}(undef, ncol, nview)
    @inbounds for v in 1:nview, c in 1:ncol
        nlo = I0_pp[c, mid_r, 1] * exp(-bins[1][c, mid_r, v]) + I0_pp[c, mid_r, 2] * exp(-bins[2][c, mid_r, v])
        nhi = I0_pp[c, mid_r, 3] * exp(-bins[3][c, mid_r, v]) + I0_pp[c, mid_r, 4] * exp(-bins[4][c, mid_r, v])
        raw_lo[c, v] = -log(max(nlo, eps_f) / max(I0lo[c], eps_f))
        raw_hi[c, v] = -log(max(nhi, eps_f) / max(I0hi[c], eps_f))
    end
    slice_lo = permutedims(raw_lo, (2, 1)); slice_hi = permutedims(raw_hi, (2, 1))

    all_v = vcat(vec(slice_lo), vec(slice_hi))
    sino_window = (Float64(quantile(all_v, 0.01)), Float64(quantile(all_v, 0.99)))

    fig = CM.Figure(size = (1180, 580))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "View", ylabel = "Detector Column",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )
    panels = (
        (1, 1, "Raw Low (uncorrected)", "bins 1+2 · 20 – 55 keV", slice_lo),
        (1, 2, "Raw High (uncorrected)", "bins 3+4 · > 55 keV", slice_hi),
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

# ╔═╡ 08030005-0000-4000-8000-000000000009
# The two CORRECTED channels (charge-sharing backward-substitution) — true-energy
# windows, spectrally disjoint.  Low [20,55) is dominated by the healthy 35–55 keV
# window so it is clean (no starved-bin noise).  cf. the raw bins directly above.
let
    n_row = size(sim_corrected.sino_low, 2)
    mid_r = n_row ÷ 2 + 1

    slice_lo = permutedims(sim_corrected.sino_low[:, mid_r, :], (2, 1))
    slice_hi = permutedims(sim_corrected.sino_high[:, mid_r, :], (2, 1))

    all_v = vcat(vec(slice_lo), vec(slice_hi))
    sino_window = (Float64(quantile(all_v, 0.01)), Float64(quantile(all_v, 0.99)))

    fig = CM.Figure(size = (1180, 580))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "View", ylabel = "Detector Column",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )
    panels = (
        (1, 1, "Corrected Low", "true 20 – 55 keV", slice_lo),
        (1, 2, "Corrected High", "true > 55 keV", slice_hi),
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

# ╔═╡ 08030005-0000-4000-8000-00000000000a
# Did the correction work? — the §5b channel-response panels, REDONE with the
# corrected channels.  Each energy's measured 4-bin pattern D(E,·) is pushed
# through the same backward-substitution (`correction.F`) used on the data, and
# we plot the resulting per-bin / per-channel sensitivity vs incident energy.
#   Left  — the 4 corrected bins each collapse into their own true window.
#   Right — raw channels (dotted) overlap heavily; corrected (solid) ≈ disjoint
#           ideal windows.  Low-channel >55 keV leakage: 57% → 12%.
let
    pcct_det = BS._build_pcct_detector(scanner)
    kVp_val  = Float64(maximum(sim_raw.energies))
    D        = BS.compute_mc_drm(pcct_det, kVp_val)          # [200 × 4] P(bin | E_inc)
    Egrid    = collect(range(1.0, kVp_val, length = size(D, 1)))
    F        = correction.F                                  # Float32 [4×4] upper-triangular

    # backward-substitution on a measured 4-bin pattern → de-migrated windows
    @inline function demig(n1, n2, n3, n4)
        t4 = n4 / F[4, 4]
        t3 = (n3 - F[3, 4] * t4) / F[3, 3]
        t2 = (n2 - F[2, 3] * t3 - F[2, 4] * t4) / F[2, 2]
        t1 = (n1 - F[1, 2] * t2 - F[1, 3] * t3 - F[1, 4] * t4) / F[1, 1]
        (max(t1, 0.0), max(t2, 0.0), max(t3, 0.0), max(t4, 0.0))
    end
    cb = [zeros(length(Egrid)) for _ in 1:4]                 # corrected per-bin response
    for i in eachindex(Egrid)
        t = demig(D[i, 1], D[i, 2], D[i, 3], D[i, 4])
        for b in 1:4
            cb[b][i] = t[b]
        end
    end
    clo = cb[1] .+ cb[2]; chi = cb[3] .+ cb[4]               # corrected channels
    rlo = D[:, 1] .+ D[:, 2]; rhi = D[:, 3] .+ D[:, 4]       # raw channels

    bin_col = [:dodgerblue, :seagreen, :darkorange, :firebrick]
    bin_lbl = ["Bin 1: 20–35", "Bin 2: 35–55", "Bin 3: 55–70", "Bin 4: >70 keV"]
    bin_win = [(20.0, 35.0), (35.0, 55.0), (55.0, 70.0), (70.0, kVp_val)]

    fig = CM.Figure(size = (1360, 580))
    ak = (titlesize = 27, subtitlesize = 19, xlabelsize = 21, ylabelsize = 21,
          xticklabelsize = 16, yticklabelsize = 16)

    # ── Corrected 4-bin response ──
    axA = CM.Axis(fig[1, 1]; title = "Corrected 4-Bin Response",
        subtitle = "each bin de-migrated to its true window (shaded)",
        xlabel = "Incident energy (keV)", ylabel = "Corrected response", ak...)
    for b in 1:4
        lo, hi = bin_win[b]
        CM.vspan!(axA, lo, hi; color = (bin_col[b], 0.10))
        CM.lines!(axA, Egrid, cb[b]; color = bin_col[b], linewidth = 2.5, label = bin_lbl[b])
    end
    CM.axislegend(
        axA;
        position = :rt,
        labelsize = 14,
        framevisible = true,
        framecolor = (:black, 0.15),
        backgroundcolor = (:white, 0.85)
    )

    # ── Two-channel response: raw vs corrected vs ideal ──
    axB = CM.Axis(fig[1, 2]; title = "Two-Channel Response",
        subtitle = "raw (dotted) overlaps · corrected (solid) ≈ ideal windows",
        xlabel = "Incident energy (keV)", ylabel = "Channel response", ak...)
    CM.vspan!(axB, 20, 55; color = (:dodgerblue, 0.08))
    CM.vspan!(axB, 55, kVp_val; color = (:firebrick, 0.08))
    CM.lines!(axB, Egrid, rlo; color = (:dodgerblue, 0.35), linewidth = 2, linestyle = :dot, label = "Low raw")
    CM.lines!(axB, Egrid, rhi; color = (:firebrick, 0.35), linewidth = 2, linestyle = :dot, label = "High raw")
    CM.lines!(axB, Egrid, clo; color = :dodgerblue, linewidth = 3, label = "Low corrected")
    CM.lines!(axB, Egrid, chi; color = :firebrick, linewidth = 3, label = "High corrected")
    CM.axislegend(
        axB;
        position = :rt,
        labelsize = 14,
        framevisible = true,
        framecolor = (:black, 0.15),
        backgroundcolor = (:white, 0.85)
    )
    fig
end

# ╔═╡ 08030005-0000-4000-8000-000000000010
# MAIN PATH — plain I0-weighted Beer combine of the RAW (uncorrected) bins:
# low = bins 1+2, high = bins 3+4.  Pre-correction (sim_corrected) kept ABOVE as
# a rejected diagnostic only.  Decomposition runs on these raw channels.
sim_lohi = let
    eps_f      = Float32(1.0e-10)
    low_bins   = [1, 2]
    high_bins  = [3, 4]
    I0_pp = sim_bins.I0_per_pixel
    I0_lo_pp = dropdims(sum(view(I0_pp, :, :, low_bins),  dims = 3); dims = 3)
    I0_hi_pp = dropdims(sum(view(I0_pp, :, :, high_bins), dims = 3); dims = 3)
    sz   = size(sim_bins.bins[1])
    N_lo = zeros(Float32, sz)
    N_hi = zeros(Float32, sz)
    for b in low_bins
        bin_b = sim_bins.bins[b]; I0_b_pp = view(I0_pp, :, :, b)
        @inbounds for v in 1:sz[3], r in 1:sz[2], c in 1:sz[1]
            N_lo[c, r, v] += I0_b_pp[c, r] * exp(-bin_b[c, r, v])
        end
    end
    for b in high_bins
        bin_b = sim_bins.bins[b]; I0_b_pp = view(I0_pp, :, :, b)
        @inbounds for v in 1:sz[3], r in 1:sz[2], c in 1:sz[1]
            N_hi[c, r, v] += I0_b_pp[c, r] * exp(-bin_b[c, r, v])
        end
    end
    sino_low  = similar(N_lo); sino_high = similar(N_hi)
    @inbounds for v in 1:sz[3], r in 1:sz[2], c in 1:sz[1]
        sino_low[c, r, v]  = -log(max(N_lo[c, r, v], eps_f) / max(I0_lo_pp[c, r], eps_f))
        sino_high[c, r, v] = -log(max(N_hi[c, r, v], eps_f) / max(I0_hi_pp[c, r], eps_f))
    end
    (sino_low = sino_low, sino_high = sino_high,
     I0_lo_pp = I0_lo_pp, I0_hi_pp = I0_hi_pp,
     bt_cpu = sim_bins.bt_cpu, geom = sim_bins.geom)
end;

# ╔═╡ 08030007-0000-4000-8000-000000000001
md"""
## 7. Projection-Domain Material Decomposition (PCCT-Φ_k)

Per-ray Cong univariate solver mapped to PCCT via the generalization in
Black (*in prep.*) — re-derives the Cong 2022 framework around an
effective spectral response Φ_k(ε) ≥ 0 so the same algorithm runs on
dual-kVp DECT and PCCT acquisitions without code changes.  The
bin-combine partition (1+2 → low, 3+4 → high) is baked into Φ_k by
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
    cong_low_bins  = 1:2          # PCCT bins forming the "low"  channel
    cong_high_bins = 3:4          # PCCT bins forming the "high" channel
end

# ╔═╡ 08030007-0000-4000-8000-000000000010
material_basis = let
    # 1D source-side spectrum (tube × flat × additional × 1/SDD²)
    e, w = BS.resolve_source_spectrum_without_bowtie(
        sim_opts, protocol; scanner = scanner,
    )

    pcct_det = BS._build_pcct_detector(scanner)
    kVp_val  = Float64(maximum(e))
    R_mat    = BS.compute_mc_drm(pcct_det, kVp_val)
    η_vec    = BS.quantum_efficiency_vector(
        pcct_det.material, pcct_det.thickness_mm, e,
    )

    n_R = size(R_mat, 1)
    drm_idx(E) = clamp(round(Int, (Float64(E) - 1.0) / (kVp_val - 1.0) * (n_R - 1)) + 1, 1, n_R)

    # Per-energy 1D weight (no bowtie).  AFTER the §6 spectral correction the
    # channels are TRUE-energy windows, so Φ is the ideal window cut of the
    # total detected response Wtot(E) = w·η·Σ_b R(E,b) — NOT the smeared
    # per-bin ΣR_grp.  This is what restores the spectral contrast Cong needs.
    Φ_L_1d = Float32[
        Float32(w[i] * η_vec[i] * sum(R_mat[drm_idx(e[i]), b] for b in cong_low_bins))
        for i in eachindex(e)
    ]
    Φ_H_1d = Float32[
        Float32(w[i] * η_vec[i] * sum(R_mat[drm_idx(e[i]), b] for b in cong_high_bins))
        for i in eachindex(e)
    ]

    # ─── 3D per-pixel ŵ to match what forward applied ──────────────────────
    # The forward path applies per-pixel bowtie B[c,r,E] (sim_raw EICT-mirror
    # path).  Cong has to invert the same effective spectrum, so we bake the
    # same B[c,r,E] into Φ and normalize per pixel: Σ_E ŵ[c,r,E] = 1 ∀ (c,r).
    bt_cpu = sim_lohi.bt_cpu   # (n_col, n_row, n_E), same one used in sim_raw
    n_col, n_row, n_E = size(bt_cpu)

    Φ_L_pp = zeros(Float32, n_col, n_row, n_E)
    Φ_H_pp = zeros(Float32, n_col, n_row, n_E)
    @inbounds for i in 1:n_E
        φL_E = Φ_L_1d[i]
        φH_E = Φ_H_1d[i]
        for r in 1:n_row, c in 1:n_col
            Φ_L_pp[c, r, i] = φL_E * bt_cpu[c, r, i]
            Φ_H_pp[c, r, i] = φH_E * bt_cpu[c, r, i]
        end
    end
    # Per-pixel normalization (Σ_E ŵ = 1 per (c, r))
    @inbounds for r in 1:n_row, c in 1:n_col
        sL = 0.0f0; sH = 0.0f0
        for i in 1:n_E
            sL += Φ_L_pp[c, r, i]
            sH += Φ_H_pp[c, r, i]
        end
        sL = max(sL, 1.0f-30); sH = max(sH, 1.0f-30)
        for i in 1:n_E
            Φ_L_pp[c, r, i] /= sL
            Φ_H_pp[c, r, i] /= sH
        end
    end

    iodine_mat = BS.XA.Elements.Iodine
    water_mat  = BS.XA.Materials.water

    p = Float32[
        Float32(BS.compute_mass_μ_at_energy(iodine_mat, Float64(E)))
        for E in e
    ]
    q = Float32[
        Float32(BS.compute_mass_μ_at_energy(water_mat, Float64(E)))
        for E in e
    ]

    # Diagnostic: center vs edge mean energy
    mid_c = n_col ÷ 2 + 1
    mid_r = n_row ÷ 2 + 1
    e64 = Float64.(e)
    mean_E_L_c = sum(e64[i] * Φ_L_pp[mid_c, mid_r, i] for i in 1:n_E)
    mean_E_H_c = sum(e64[i] * Φ_H_pp[mid_c, mid_r, i] for i in 1:n_E)
    mean_E_L_e = sum(e64[i] * Φ_L_pp[1,     mid_r, i] for i in 1:n_E)
    mean_E_H_e = sum(e64[i] * Φ_H_pp[1,     mid_r, i] for i in 1:n_E)
    @info "[Cong basis] $(n_E) energies · 3D per-pixel ŵ (EICT-mirror)"
    @info "  low : center ⟨E⟩ = $(round(mean_E_L_c, digits = 1)) keV · edge ⟨E⟩ = $(round(mean_E_L_e, digits = 1)) keV (Δ = $(round(mean_E_L_e - mean_E_L_c, digits = 1)) ← bowtie hardening)"
    @info "  high: center ⟨E⟩ = $(round(mean_E_H_c, digits = 1)) keV · edge ⟨E⟩ = $(round(mean_E_H_e, digits = 1)) keV (Δ = $(round(mean_E_H_e - mean_E_H_c, digits = 1)))"

    (ŵ_L = Φ_L_pp, p_L = p,        q_L = q,
     ŵ_H = Φ_H_pp, p_H = copy(p),  q_H = copy(q))
end;

# ╔═╡ 08030007-0000-4000-8000-000000000015
# ─── INLINED Cong solver (was BS.apply_cong!) — edit freely for the corrected bins ───
# Faithful port of src/reconstruction/vmi/cong.jl::apply_cong!, lifted into the
# notebook so the per-ray kernel can be re-worked for the spectrally-corrected
# channels.  Runs on GPU via BS.AK.foreachindex (same body compiles CPU/Metal/CUDA);
# all root-finds use BS.brent_solve (allocation-free, GPU-safe).
#
# Algorithm (Cong 2022, generalized — Black, in prep.):
#   1. Brent on water-equivalent path L  → anchor c̄ = c_w·L           (LOW channel)
#   2. Newton on the 5th-order Taylor quintic for x = h(y)            (LOW channel)
#   3. Brent on y matching the HIGH channel: T_H_pred(y, c̄+h(y)) = T_H_meas
#   → y = iodine path, C = c̄ + h(y) = water path.
#
# LIKELY TWEAK POINTS for the corrected bins (the corrected LOW window [20,55) is
# soft + iodine-absorbed, so the old water-equiv anchor may be the weak link):
#   • anchor (Step 1) — try a high-channel or linearized-2×2 anchor for c̄.
#   • metric  (Step 3) — exact root-find assumes noiseless m_2; Cong §2.5 suggests
#     a Poisson log-likelihood min for photon-starved bins.
#   • Taylor order / one-step anchor refinement (Cong §2.6, algorithm step 4).
function cong_inline!(
        sino_y, sino_c, sino_low, sino_high,
        ŵ_L, p_L, q_L, ŵ_H, p_H, q_H;
        a_w::Float32 = 0.0f0, c_w::Float32 = 1.0f0,
        anchor_high::Bool = true,    # ← TWEAK: anchor c̄ on the HIGH channel (robust,
                                     #   well-populated).  false = original (LOW anchor).
        newton_max_iter::Int = 12, newton_tol::Float32 = eps(Float32),
        y_max_factor::Float32 = 0.99f0, y_max_cap::Float32 = 1.0f7,
    )
    # Pick the anchor channel (Steps 1+2: water-equiv anchor + quintic h(y)) and the
    # solve channel (Step 3: root-find iodine path y).  The decomposition is symmetric
    # in the two channels — y is iodine and C is water either way — so this only changes
    # WHICH measurement seeds c̄.  The corrected LOW window is soft + K-edge-absorbed, so
    # anchoring on the HIGH channel avoids the water-equiv Brent overflowing on iodine rays.
    ŵ_anc = anchor_high ? ŵ_H : ŵ_L;  p_anc = anchor_high ? p_H : p_L;  q_anc = anchor_high ? q_H : q_L
    ŵ_sol = anchor_high ? ŵ_L : ŵ_H;  p_sol = anchor_high ? p_L : p_H;  q_sol = anchor_high ? q_L : q_H
    per_ray = ndims(ŵ_anc) == 3
    nE_anc = per_ray ? size(ŵ_anc, 3) : length(ŵ_anc)
    nE_sol = per_ray ? size(ŵ_sol, 3) : length(ŵ_sol)
    p_anc_min = Float32(minimum(p_anc))
    nm_iter = newton_max_iter; n_tol = newton_tol
    y_fac = y_max_factor; y_cap = y_max_cap
    n_col = Int(size(sino_low, 1)); n_row = Int(size(sino_low, 2))

    BS.AK.foreachindex(sino_low) do idx
        i0  = idx - 1
        col = (i0 % n_col) + 1
        row = ((i0 ÷ n_col) % n_row) + 1

        p_L_meas = Float32(sino_low[idx])
        p_H_meas = Float32(sino_high[idx])
        if p_L_meas < 1f-6 && p_H_meas < 1f-6
            sino_y[idx] = 0f0; sino_c[idx] = 0f0; return
        end
        p_anc_meas = anchor_high ? p_H_meas : p_L_meas
        p_sol_meas = anchor_high ? p_L_meas : p_H_meas
        T_anc_meas = exp(-p_anc_meas); T_sol_meas = exp(-p_sol_meas)

        # Step 1 — Brent on water-equivalent path L on the ANCHOR channel → c̄.
        water_T = function (L::Float32)
            T = 0f0
            if per_ray
                @inbounds for i in 1:nE_anc
                    T += ŵ_anc[col, row, i] * exp(-(p_anc[i]*a_w + q_anc[i]*c_w) * L)
                end
            else
                @inbounds for i in 1:nE_anc
                    T += ŵ_anc[i] * exp(-(p_anc[i]*a_w + q_anc[i]*c_w) * L)
                end
            end
            T - T_anc_meas
        end
        L_water, ok_L = BS.brent_solve(water_T, 0f0, 60f0)
        if !ok_L
            sino_y[idx] = 0f0; sino_c[idx] = 0f0; return
        end
        c̄ = c_w * L_water

        y_max = min(y_fac * p_anc_meas / max(p_anc_min, eps(Float32)), y_cap)
        if y_max <= 0f0
            sino_y[idx] = 0f0; sino_c[idx] = c̄; return
        end

        # Step 2 — Newton on the Taylor quintic (ANCHOR channel) for x = h(y).
        solve_quintic = function (y::Float32, c̄_::Float32)
            P0=0f0;P1=0f0;P2=0f0;P3=0f0;P4=0f0;P5=0f0
            if per_ray
                @inbounds for i in 1:nE_anc
                    q_i=q_anc[i];q2=q_i*q_i;q3=q2*q_i;q4=q3*q_i;q5=q4*q_i
                    wexp=ŵ_anc[col,row,i]*exp(-p_anc[i]*y - q_i*c̄_)
                    P0+=wexp;P1-=wexp*q_i;P2+=wexp*q2*0.5f0;P3-=wexp*q3*(1f0/6f0);P4+=wexp*q4*(1f0/24f0);P5-=wexp*q5*(1f0/120f0)
                end
            else
                @inbounds for i in 1:nE_anc
                    q_i=q_anc[i];q2=q_i*q_i;q3=q2*q_i;q4=q3*q_i;q5=q4*q_i
                    wexp=ŵ_anc[i]*exp(-p_anc[i]*y - q_i*c̄_)
                    P0+=wexp;P1-=wexp*q_i;P2+=wexp*q2*0.5f0;P3-=wexp*q3*(1f0/6f0);P4+=wexp*q4*(1f0/24f0);P5-=wexp*q5*(1f0/120f0)
                end
            end
            x=0f0
            for _ in 1:nm_iter
                F=(P0-T_anc_meas)+x*(P1+x*(P2+x*(P3+x*(P4+x*P5))))
                dF=P1+x*(2f0*P2+x*(3f0*P3+x*(4f0*P4+x*5f0*P5)))
                abs(dF)<1f-30 && break
                Δ=F/dF; x-=Δ; abs(Δ)<n_tol && break
            end
            x
        end

        # Step 3 — Brent on y matching the SOLVE channel.
        T_sol_pred = function (y::Float32, C::Float32)
            T=0f0
            if per_ray
                @inbounds for i in 1:nE_sol
                    T += ŵ_sol[col,row,i]*exp(-p_sol[i]*y - q_sol[i]*C)
                end
            else
                @inbounds for i in 1:nE_sol
                    T += ŵ_sol[i]*exp(-p_sol[i]*y - q_sol[i]*C)
                end
            end
            T
        end
        G = function (y::Float32)
            x = solve_quintic(y, c̄)
            T_sol_pred(y, c̄ + x) - T_sol_meas
        end
        y_opt, ok_y = BS.brent_solve(G, 0f0, y_max)
        if !ok_y
            sino_y[idx] = a_w*L_water; sino_c[idx] = c_w*L_water; return
        end
        x_final = solve_quintic(y_opt, c̄)
        sino_y[idx] = max(y_opt, 0f0)
        sino_c[idx] = max(c̄ + x_final, 0f0)
    end
    (sino_y, sino_c)
end;

# ╔═╡ 08030007-0000-4000-8000-000000000020
sino_basis = let
    # NORMAL src Cong — BS.create_cong_workspace + BS.apply_cong! (original path).
    # Runs on the raw low/high channels with the distorted R_grp Φ (material_basis).
    # The inlined `cong_inline!` above is kept as a dormant, editable copy but is
    # NOT used here.
    sino_low_gpu  = to_gpu(Float32.(sim_lohi.sino_low))
    sino_high_gpu = to_gpu(Float32.(sim_lohi.sino_high))

    sino_y = similar(sino_low_gpu)
    sino_c = similar(sino_low_gpu)
    fill!(sino_y, 0.0f0); fill!(sino_c, 0.0f0)

    @info "Cong polychromatic decomposition (src BS.apply_cong!): $(size(sim_lohi.sino_low))"
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
        sino_water  = Array(sino_c),
        geom        = sim_lohi.geom,
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

# ╔═╡ 08030008-0000-4000-8000-000000000001
md"""
## 8. FBP: Iodine and Water Basis Maps

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

# ╔═╡ 08030009-0000-4000-8000-000000000001
md"""
## 9. Z-Direction Median Filter

1D median along z, per `(x, y)` voxel column.  `adjacent_slices = n` ⇒
`2n + 1`-slice window.  Cheap streak/outlier suppression that
exploits z-invariance of this tiled phantom.
"""

# ╔═╡ 08030009-0000-4000-8000-000000000005
Z_MEDIAN_ADJACENT = 2;

# ╔═╡ 08030009-0000-4000-8000-000000000010
basis_z = let
    (
        vol_iodine = BS.apply_median_z(
            basis_volumes.vol_iodine_raw; adjacent_slices = Z_MEDIAN_ADJACENT
        ),
        vol_water = BS.apply_median_z(
            basis_volumes.vol_water_raw; adjacent_slices = Z_MEDIAN_ADJACENT
        ),
        geom = basis_volumes.geom,
    )
end;

# ╔═╡ 08030009-0000-4000-8000-000000000030
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
        (1, 1, 2, "Iodine Basis · z-median", "g/cm³", slice_iod, _qrange(slice_iod)),
        (1, 3, 4, "Water Basis · z-median", "g/cm³", slice_wat, _qrange(slice_wat)),
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
## 10. VMI Synthesis

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
    nx_r, ny_r, nz_r = size(basis_z.vol_water)

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

    c_w = Float64(_mean(basis_z.vol_water))
    c_i = Float64(_mean(basis_z.vol_iodine))
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
vmi_HU_by_keV = let
    c_iodine_mg_per_mL = basis_z.vol_iodine .* 1000.0f0

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
            basis_z.vol_water, c_iodine_mg_per_mL; energy_keV = E,
        )
    end
    out
end;

# ╔═╡ 0803000a-0000-4000-8000-000000000040
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
            colormap = :grays, colorrange = HU_window
        )
        CM.hidedecorations!(ax)
    end
    CM.Colorbar(
        fig[1:2, 3]; colormap = :grays, colorrange = HU_window,
        label = "HU", width = 16, labelsize = 22, ticklabelsize = 18
    )
    fig
end

# ╔═╡ 0803000b-0000-4000-8000-000000000001
md"""
## 11. VMI Post-Processing (Mono+)

Frequency-split rule (Grant 2014):

```
Mono+(E)     = LP_σ(VMI_E) + VMI_opt − LP_σ(VMI_opt)
Mono+(E_opt) = VMI_opt   (identity at the noise-optimal anchor)
```

`σ_vmi_lp_px` pairs element-wise with `pcct_vmi_energies` — one σ per
VMI energy.  σ = 0 ⇒ identity (no LP, no FFT).

!!! info "Mono+ is optional — default is pass-through"
    `σ_vmi_lp_px = [0.0, 0.0, 0.0, 0.0]` makes every entry an identity
    (no LP, no FFT), so this section is effectively a no-op and the
    downstream rod regression sees the raw 2-basis VMI volumes from
    §10.  Bump a σ above 0 to opt in per-energy: e.g.
    `Float64[2.0, 0.0, 1.0, 1.0]` smooths 40 / 100 / 140 keV toward
    the 70 keV anchor while keeping 70 keV exact.  Leave it as zeros
    to keep the pipeline transparent — the per-keV plots below will
    be identical to the §11 outputs.
"""

# ╔═╡ 0803000b-0000-4000-8000-000000000005
# Default = all zeros = pass-through (no Mono+ smoothing).  See the
# `!!! info` block above for the opt-in pattern.
σ_vmi_lp_px = Float64[0.0, 0.0, 0.0, 0.0];

# ╔═╡ 0803000b-0000-4000-8000-000000000010
vmi_HU_final = let
    volumes = [vmi_HU_by_keV[E] for E in pcct_vmi_energies]

    ws = BS.create_mono_plus_workspace(
        volumes[1]; n_energies = length(pcct_vmi_energies)
    )
    BS.apply_mono_plus!(
        ws, volumes, pcct_vmi_energies;
        E_noise_opt = 70.0, σ_lp_px = σ_vmi_lp_px, verbose = true,
    )

    out = Dict{Float64, Array{Float32, 3}}()
    for (i, E) in enumerate(pcct_vmi_energies)
        out[E] = copy(ws.out_vols[i])
    end
    ws = nothing; GC.gc(true)
    out
end;

# ╔═╡ 0803000b-0000-4000-8000-000000000040
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
            colormap = :grays, colorrange = HU_window
        )
        CM.hidedecorations!(ax)
    end
    CM.Colorbar(
        fig[1:2, 3]; colormap = :grays, colorrange = HU_window,
        label = "HU", width = 16, labelsize = 22, ticklabelsize = 18
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
        title = "70 keV Mono+ HU recon",
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
70 keV Mono+ slice.  Right panel: mean HU over that ROI vs VMI energy.
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
        subtitle = "Overlaid on 70 keV Mono+",
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
    nx_r, ny_r, nz_r = size(basis_z.vol_water)

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

# ╔═╡ 0803000d-0000-4000-8000-000000000020
vmi_noise_by_keV = let
    roi_idx = findall(heart_noise_roi.mask_2d)
    nz_r = size(basis_z.vol_water, 3)

    # ── Basis-covariance diagnostic (progress meter for the separation work) ──
    # σ_HU(E)² = 1000²V_w + α²V_i + 2·1000·α·C_iw ;  α* = −1000·C_iw/V_i locates
    # the noise minimum.  Monotonic-decreasing ⟺ α* ≤ α(140).  Watch ρ_basis
    # (need it → 0/positive) and α* (need it → ≤5.4).  var/cov via mean.
    cw    = vec(Float64[Float64(basis_z.vol_water[ci, z])           for z in 1:nz_r, ci in roi_idx])
    ci_mg = vec(Float64[Float64(basis_z.vol_iodine[ci, z]) * 1000.0 for z in 1:nz_r, ci in roi_idx])
    mw = mean(cw); mi = mean(ci_mg)
    V_w  = mean((cw .- mw).^2);  V_i = mean((ci_mg .- mi).^2);  C_iw = mean((cw .- mw) .* (ci_mg .- mi))
    ρ_b  = C_iw / sqrt(max(V_w * V_i, 1e-30))
    α(E) = BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, Float64(E)) /
           BS.compute_mass_μ_at_energy(BS.XA.Materials.water,  Float64(E))
    α_star = -1000.0 * C_iw / max(V_i, 1e-30)
    @info "[basis cov · heart ROI]  σ(c_water)=$(round(sqrt(V_w), sigdigits=3)) g/mL · " *
          "σ(c_iod)=$(round(sqrt(V_i), sigdigits=3)) mg/mL · ρ_basis=$(round(ρ_b, digits=3)) · " *
          "water floor=$(round(1000*sqrt(V_w), digits=1)) HU"
    @info "  α* = $(round(α_star, digits=1))  [α(40)=$(round(α(40),digits=1)), α(70)=$(round(α(70),digits=1)), α(140)=$(round(α(140),digits=1))] → " *
          (α_star ≤ α(140) ? "MONOTONIC predicted" : (α_star ≥ α(40) ? "monotonic-increasing" : "U-shape (min where α=α*)"))

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
    CM.ylims!(ax2, 0, maximum(σs) * 1.4)

    fig
end

# ╔═╡ 0803000d-0000-4000-8000-000000000040
# VMI-noise DECOMPOSITION — confirms (in the image domain) where the U comes from.
# From the basis covariance in the heart ROI (σ_w, σ_i, C_iw):
#   σ_HU(E)² = 1000²·V_w  +  α(E)²·V_i  +  2·1000·α(E)·C_iw
#              └ water floor ┘  └ iodine ┘   └── cross term ──┘
# The flat blue line is the water floor 1000·√V_w — the high-keV asymptote.  The
# grey dotted curve is what σ would be WITHOUT the cross term (C_iw=0).  If the
# black "predicted total" tracks the measured points, the theorem holds and the
# high-keV rise is provably the water floor (set by C_iw<0 / poor separation).
let
    roi_idx = findall(heart_noise_roi.mask_2d)
    nz_r = size(basis_z.vol_water, 3)
    cw    = vec(Float64[Float64(basis_z.vol_water[ci, z])           for z in 1:nz_r, ci in roi_idx])
    ci_mg = vec(Float64[Float64(basis_z.vol_iodine[ci, z]) * 1000.0 for z in 1:nz_r, ci in roi_idx])
    mw = mean(cw); mi = mean(ci_mg)
    V_w  = mean((cw .- mw) .^ 2); V_i = mean((ci_mg .- mi) .^ 2); C_iw = mean((cw .- mw) .* (ci_mg .- mi))
    ρ_b  = C_iw / sqrt(max(V_w * V_i, 1.0e-30))
    αf(E) = Float64(BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, E)) /
            Float64(BS.compute_mass_μ_at_energy(BS.XA.Materials.water,  E))
    floor_HU = 1000.0 * sqrt(V_w)
    α_star = -1000.0 * C_iw / max(V_i, 1.0e-30)

    Es = collect(40.0:2.0:140.0)
    σ_total = [sqrt(max(1.0e6 * V_w + αf(E)^2 * V_i + 2.0e3 * αf(E) * C_iw, 0.0)) for E in Es]
    σ_indep = [sqrt(1.0e6 * V_w + αf(E)^2 * V_i) for E in Es]   # cross term = 0
    σ_iod   = [αf(E) * sqrt(V_i) for E in Es]
    E_min   = Es[argmin(σ_total)]

    Em = sort(collect(keys(vmi_noise_by_keV))); σm = [vmi_noise_by_keV[E].std for E in Em]

    fig = CM.Figure(size = (1040, 660))
    ak  = (titlesize = 28, subtitlesize = 18, xlabelsize = 22, ylabelsize = 22,
           xticklabelsize = 16, yticklabelsize = 16)
    ax = CM.Axis(fig[1, 1];
        title = "VMI Noise Decomposition (heart ROI)",
        subtitle = "ρ_basis = $(round(ρ_b, digits=2)) · water floor = $(round(floor_HU, digits=0)) HU · " *
                   "U-min ≈ $(Int(E_min)) keV (α*=$(round(α_star, digits=1)))",
        xlabel = "VMI energy (keV)", ylabel = "Noise σ (HU)", ak...)
    CM.hlines!(ax, [floor_HU]; color = :dodgerblue, linewidth = 2.5, linestyle = :dash,
        label = "water floor 1000·√V_w")
    CM.lines!(ax, Es, σ_iod;   color = :firebrick, linewidth = 2, linestyle = :dash, label = "iodine α·√V_i")
    CM.lines!(ax, Es, σ_indep; color = :gray,      linewidth = 2, linestyle = :dot,  label = "if C_iw = 0 (no cross term)")
    CM.lines!(ax, Es, σ_total; color = :black,     linewidth = 3, label = "predicted total (theorem)")
    CM.scatter!(ax, Em, σm;    color = :tomato,    markersize = 16, label = "measured σ")
    CM.axislegend(ax; position = :rt, labelsize = 14, framevisible = false)
    CM.ylims!(ax, 0, max(maximum(σm), maximum(σ_total)) * 1.15)
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

    n_z = size(basis_z.vol_water, 3)

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
        c_w = _mean(basis_z.vol_water, roi)
        c_i = _mean(basis_z.vol_iodine, roi)
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
   → Z-Direction Median Filter
   → Monoenergetic VMI Synthesis  (textbook 2-basis, mono μρ_water divisor)
   → Mono+ Post-Processing  (per-keV σ via σ_vmi_lp_px)
   → Per-rod Measured vs Theoretical Regression
        (water · lipid · collagen · iodine at 40 / 70 / 100 / 140 keV)
```

1:1 parity with notebook 07's QRM-Thorax pure-material pipeline,
swapping the dual-kVp GSI acquisition for a Siemens Naeotom Alpha PCCT
4-bin acquisition.  The 4-bin PCCT measurement is bin-combined into a
two-channel `(low, high)` pair that Cong's PCCT-Φ_k decomposition then
consumes directly — exactly the same downstream pipeline as notebook
07 (Cong → FBP × 2 → z-median → VMI → Mono+ → per-rod regression).
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
# ╠═08010003-0000-4000-8000-000000000009
# ╠═08010003-0000-4000-8000-000000000010
# ╠═08010003-0000-4000-8000-000000000011
# ╠═08010003-0000-4000-8000-000000000040
# ╟─08010003-0000-4000-8000-000000000050
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
# ╠═08030004-0000-4000-8000-00000000000a
# ╠═08030004-0000-4000-8000-00000000000b
# ╟─08030004-0000-4000-8000-000000000011
# ╠═08030004-0000-4000-8000-00000000000e
# ╠═08030004-0000-4000-8000-00000000000c
# ╠═08030004-0000-4000-8000-000000000012
# ╠═08030004-0000-4000-8000-000000000013
# ╠═08030004-0000-4000-8000-000000000014
# ╠═08030004-0000-4000-8000-000000000025
# ╟─08030004-0000-4000-8000-000000000040
# ╟─08030004-0000-4000-8000-000000000050
# ╟─08030004-0000-4000-8000-000000000060
# ╟─08030005-0000-4000-8000-000000000001
# ╠═08030005-0000-4000-8000-000000000005
# ╠═08030005-0000-4000-8000-000000000008
# ╟─08030005-0000-4000-8000-00000000000b
# ╟─08030005-0000-4000-8000-000000000009
# ╟─08030005-0000-4000-8000-00000000000a
# ╠═08030005-0000-4000-8000-000000000010
# ╟─08030007-0000-4000-8000-000000000001
# ╠═08030007-0000-4000-8000-000000000005
# ╠═08030007-0000-4000-8000-000000000010
# ╠═08030007-0000-4000-8000-000000000015
# ╠═08030007-0000-4000-8000-000000000020
# ╟─08030007-0000-4000-8000-000000000040
# ╟─08030008-0000-4000-8000-000000000001
# ╠═08030008-0000-4000-8000-000000000010
# ╟─08030008-0000-4000-8000-000000000030
# ╟─08030009-0000-4000-8000-000000000001
# ╠═08030009-0000-4000-8000-000000000005
# ╠═08030009-0000-4000-8000-000000000010
# ╟─08030009-0000-4000-8000-000000000030
# ╟─0803000a-0000-4000-8000-000000000001
# ╠═0803000a-0000-4000-8000-000000000005
# ╠═0803000a-0000-4000-8000-000000000008
# ╠═0803000a-0000-4000-8000-000000000010
# ╠═0803000a-0000-4000-8000-000000000020
# ╟─0803000a-0000-4000-8000-000000000040
# ╟─0803000b-0000-4000-8000-000000000001
# ╠═0803000b-0000-4000-8000-000000000005
# ╠═0803000b-0000-4000-8000-000000000010
# ╟─0803000b-0000-4000-8000-000000000040
# ╟─0803000c-0000-4000-8000-000000000000
# ╟─0803000c-0000-4000-8000-00000000000a
# ╟─0803000c-0000-4000-8000-000000000001
# ╟─0803000c-0000-4000-8000-000000000002
# ╟─0803000c-0000-4000-8000-000000000003
# ╟─0803000d-0000-4000-8000-000000000001
# ╠═0803000d-0000-4000-8000-000000000004
# ╠═0803000d-0000-4000-8000-000000000005
# ╠═0803000d-0000-4000-8000-000000000010
# ╠═0803000d-0000-4000-8000-000000000020
# ╟─0803000d-0000-4000-8000-000000000030
# ╟─0803000d-0000-4000-8000-000000000040
# ╠═0803000c-0000-4000-8000-000000000010
# ╠═0803000c-0000-4000-8000-000000000011
# ╠═0803000c-0000-4000-8000-000000000012
# ╠═0803000c-0000-4000-8000-000000000020
# ╟─0803000c-0000-4000-8000-000000000030
# ╟─0803000c-0000-4000-8000-000000000031
# ╟─0803000e-0000-4000-8000-000000000001
