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
    APPLY_NOISE_FORWARD = false     # OFF for the rung-1 channel-noise probe — all noise is
                                    # injected post-combine in `channels_noised` instead.
                                    # Set true to restore the realistic per-bin Poisson path
                                    # (inline mirror of source apply_pcct_noise!).
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

        # Per-view 2D scratch buffers (~700 KB each) — keep allocations off the
        # full sinogram footprint per the project memory budget.
        noise_2d = Array{Float32}(undef, sz[1], sz[2])
        Nexp_2d  = Array{Float32}(undef, sz[1], sz[2])

        # Track per-bin noise magnitude (std of post - pre) by sampling
        # one mid-slice from each bin pre / post.
        mid_v = sz[3] ÷ 2 + 1
        noise_samples = Vector{Tuple{Int, Float64}}()

        for b in eachindex(bins)
            bin_b = bins[b]
            I0_phys_pp_2d = Float32.(view(I0_pp, :, :, b)) .* scale   # (n_col, n_row) physical I0
            pre_mid = copy(view(bin_b, :, :, mid_v))                  # mid-slice before

            for v in 1:sz[3]
                bin_v   = view(bin_b, :, :, v)
                randn!(rng, noise_2d)                                  # N(0,1) per pixel
                @. Nexp_2d = max(I0_phys_pp_2d * exp(-bin_v), 0.1f0)   # expected physical counts
                if NOISE_DOMAIN === :count
                    # Poisson-in-counts → −log  (scanner-faithful, biased)
                    @. noise_2d = max(Nexp_2d + nr_scale * sqrt(Nexp_2d) * noise_2d, 1.0f0)
                    @. bin_v   = -log(noise_2d / I0_phys_pp_2d)
                else
                    # Direct line-integral injection: σ_p = (1−nr)/√N̄, zero-bias
                    @. bin_v = bin_v + nr_scale * noise_2d / sqrt(Nexp_2d)
                end
            end

            post_mid = view(bin_b, :, :, mid_v)
            push!(noise_samples, (b, std(vec(post_mid .- pre_mid))))
        end

        @info "Noise applied [$(NOISE_DOMAIN)-domain] — I0_phys = $(round(I0_physics, sigdigits = 3)) ph/dexel/view, scale = $(round(scale, sigdigits = 3)), nr_scale = $(nr_scale)"
        for (b, σ_noise) in noise_samples
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
    APPLY_PILEUP_FORWARD = false

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
    APPLY_SCATTER_FORWARD = false

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
    APPLY_SCATTER_CORRECTION = false

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

# ╔═╡ 08030005-0000-4000-8000-000000000001
md"""
## 6. Bin Combine: 4 Bins → Low / High Pair

I₀-weighted Beer recombination of the 4 raw PCCT bins:

```
N_grp = Σ_{b ∈ grp} I0[b] · exp(-p[b])
p_grp = -log(N_grp / Σ_{b ∈ grp} I0[b])
```

* **Low**  = bins **1 + 2** (20 – 55 keV)
* **High** = bins **3 + 4** ( > 55 keV)

Each combined sinogram represents a polychromatic measurement at the
I₀-weighted average spectrum of its bin group — the two-channel
`(low, high)` pair the Cong PCCT-Φ_k decomposition then consumes
directly, exactly as notebook 07 does for the dual-kVp `(80, 140)`
pair.
"""

# ╔═╡ 08030005-0000-4000-8000-000000000010
sim_lohi = let
    eps_f      = Float32(1.0e-10)
    low_bins   = [1, 2]
    high_bins  = [3, 4]

    # Combine the (per-bin-noised) bins into low/high channels in the COUNT
    # domain, then back to a log line integral.  `sim_bins.bins` are already
    # -log(N / I0_per_pixel[c,r,b]); recombine with the matching per-pixel I0:
    #   N_grp[c,r,v] = Σ_{b ∈ grp} I0_per_pixel[c,r,b] · exp(-bins[b][c,r,v])
    #   sino_grp     = -log(N_grp / Σ_{b ∈ grp} I0_per_pixel[c,r,b])
    # Air rays land at 0 per pixel (per-pixel-bowtie baseline cancels).  Noise is
    # injected per-bin upstream in sim_noise_forward — this cell only combines.
    I0_pp = sim_bins.I0_per_pixel   # (n_col, n_row, n_bins)
    I0_lo_pp = dropdims(sum(view(I0_pp, :, :, low_bins),  dims = 3); dims = 3)   # (n_col, n_row)
    I0_hi_pp = dropdims(sum(view(I0_pp, :, :, high_bins), dims = 3); dims = 3)

    sz   = size(sim_bins.bins[1])
    N_lo = zeros(Float32, sz)
    N_hi = zeros(Float32, sz)
    for b in low_bins
        bin_b = sim_bins.bins[b]
        I0_b_pp = view(I0_pp, :, :, b)
        @inbounds for v in 1:sz[3], r in 1:sz[2], c in 1:sz[1]
            N_lo[c, r, v] += I0_b_pp[c, r] * exp(-bin_b[c, r, v])
        end
    end
    for b in high_bins
        bin_b = sim_bins.bins[b]
        I0_b_pp = view(I0_pp, :, :, b)
        @inbounds for v in 1:sz[3], r in 1:sz[2], c in 1:sz[1]
            N_hi[c, r, v] += I0_b_pp[c, r] * exp(-bin_b[c, r, v])
        end
    end

    sino_low  = similar(N_lo)
    sino_high = similar(N_hi)
    @inbounds for v in 1:sz[3], r in 1:sz[2], c in 1:sz[1]
        sino_low[c, r, v]  = -log(max(N_lo[c, r, v], eps_f) / max(I0_lo_pp[c, r], eps_f))
        sino_high[c, r, v] = -log(max(N_hi[c, r, v], eps_f) / max(I0_hi_pp[c, r], eps_f))
    end

    (sino_low = sino_low, sino_high = sino_high,
     I0_lo_pp = I0_lo_pp, I0_hi_pp = I0_hi_pp,
     bt_cpu = sim_bins.bt_cpu,
     geom = sim_bins.geom)
end;

# ╔═╡ 08030005-0000-4000-8000-000000000020
# === Channel-domain noise injection — INTER-CHANNEL CORRELATION LEVER ===
# Finding (rung 1): equal, INDEPENDENT, constant-σ white noise on (p_low, p_high)
# STILL produced the U-shape → the U is not an injection-asymmetry artifact, it is
# intrinsic to the Cong → VMI decomposition geometry.  Two symptoms split cleanly:
#   • U-shape LOCATION   ← spectral A-ratios AND inter-channel noise correlation ρ
#                          (independent of conditioning)
#   • overall MAGNITUDE  ← conditioning / spectral separation (RMSE, σ-floor)
#
# Lever (this cell): the basis-noise cross term C_iw sets the VMI-noise minimum
#   σ_HU(E)² = 1e6·V_w + α(E)²·V_i + 2e3·α(E)·C_iw ,  min at  α* = −1e3·C_iw/V_i.
# Monotonic-DECREASING ⟺ α* ≤ α(140) ⟺ C_iw not too negative.  INDEPENDENT channel
# noise → C_iw < 0 → U.  POSITIVE inter-channel correlation ρ raises C_iw toward 0
# → α* drops below α(140) → monotonic.  CHANNEL_CORR is that ρ knob; ρ=1 = common
# mode (same field on both channels).  Sweep ρ: 0 (U) → 0.5 → 1 (expect monotonic).
#
# PAPER CAVEAT: disjoint-energy-bin Poisson counts are physically INDEPENDENT
# (Poisson thinning) → ρ≈0, which is exactly what gives the U.  So a positive ρ is
# the mathematically-indicated lever but still needs a physical story (shared-flux
# common mode, or the real fix lives in the spectral responses / A-ratios).  We
# confirm the lever moves the shape here FIRST, then justify it.
channels_noised = let
    NOISE_MODE   = :simple_white   # ON — rung-1 simple white noise (independent), to read the
                                   # basis-noise correlation Cong produces (sino + image cov cells).
                                   # :none = passthrough.
    N0           = 5000.0          # global count → σ = 1/√N₀ (line-integral domain).  ← visibility knob
    CHANNEL_CORR = 0.0             # ρ ∈ [−1,1] inter-channel noise correlation — THE lever:
                                   #   0  = independent      → C_iw<0 → U-shape (rung-1 result)
                                   #  +1  = common-mode (same field on both) → C_iw→0 → MONOTONIC
                                   #  <0  = anti-correlated   → C_iw more negative → worse
    NOISE_SEED   = 1234

    lo = copy(sim_lohi.sino_low)
    hi = copy(sim_lohi.sino_high)

    if NOISE_MODE === :simple_white
        σ   = Float32(1.0 / sqrt(N0))
        ρ   = Float32(CHANNEL_CORR)
        ρc  = Float32(sqrt(max(1.0f0 - ρ^2, 0.0f0)))  # independent-component weight
        rng = MersenneTwister(NOISE_SEED)
        sz  = size(lo)
        g1  = Array{Float32}(undef, sz[1], sz[2])     # shared component (per-view 2D scratch)
        g2  = Array{Float32}(undef, sz[1], sz[2])     # independent component
        for v in 1:sz[3]
            randn!(rng, g1); randn!(rng, g2)
            # n_L = σ·g1 ;  n_H = σ·(ρ·g1 + √(1−ρ²)·g2)  → Corr(n_L,n_H)=ρ, both var σ²
            @views lo[:, :, v] .+= σ .* g1
            @views hi[:, :, v] .+= σ .* (ρ .* g1 .+ ρc .* g2)
        end
        @info "[channel noise] σ = $(round(σ, sigdigits = 3)) (N₀=$(N0)) · inter-channel ρ = $(round(ρ, digits = 2)) " *
              (ρ ≥ 0.99f0 ? "(common-mode → expect MONOTONIC)" :
               ρ ≤ 0.01f0 ? "(independent → expect U-shape)" : "(partial correlation)")
    else
        @info "[channel noise] passthrough (NOISE_MODE = :none)"
    end

    (sino_low = lo, sino_high = hi,
     I0_lo_pp = sim_lohi.I0_lo_pp, I0_hi_pp = sim_lohi.I0_hi_pp,
     bt_cpu = sim_lohi.bt_cpu, geom = sim_lohi.geom)
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

    # Per-energy 1D weight (no bowtie) — `w · η · ΣR_grp`
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

# ╔═╡ 08030007-0000-4000-8000-000000000020
sino_basis = let
    sino_low_gpu  = to_gpu(Float32.(channels_noised.sino_low))
    sino_high_gpu = to_gpu(Float32.(channels_noised.sino_high))

    sino_y = similar(sino_low_gpu)
    sino_c = similar(sino_low_gpu)
    fill!(sino_y, 0.0f0); fill!(sino_c, 0.0f0)

    @info "Cong polychromatic decomposition: $(size(channels_noised.sino_low))"
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
        geom        = channels_noised.geom,
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

# ╔═╡ 08030007-0000-4000-8000-000000000050
# === Basis-domain noise injection — SLOPE CONTROL (iodine vs water vs correlation) ===
# Finding: basis-domain noise gives PERFECT rod RMSE (no bias — it skips Cong's nonlinear
# propagation), but the σ-vs-E slope came out monotonic-INCREASING — the wrong way.  Why:
#   δHU(E) = 1000·δc_w + 1000·α(E)·δc_i ,  α(E)=(μ/ρ)_i/(μ/ρ)_w  (≈90 @40 keV → ≈15 @140)
#   • water-basis noise  δc_w → FLAT floor (energy-independent)
#   • iodine-basis noise δc_i → ∝ α(E) → DECREASING (this is the no-noise iodine-texture shape)
#   • C_iw < 0 → cancels the low-E noise (α huge there) → leaves the high-E floor → INCREASING
# So monotonic-DECREASING (the goal) needs IODINE-dominant noise with C_iw not too negative.
# The three knobs below dial σ_i, σ_w, and ρ_basis directly; default = iodine-ONLY
# (FRAC_W=0, ρ=0) to test the reverse.  DIAGNOSTIC for the shape recipe — once we know which
# (σ_i, σ_w, ρ) gives the right slope, we ask what channel noise yields that basis covariance
# after Cong.  (channels_noised is :none, so this is the only noise source.)
basis_sino_noised = let
    APPLY      = true
    FRAC_I     = 0.0       # iodine-basis noise: σ_i = FRAC_I · std(iodine sinogram).  ↑ → more DECREASE
    FRAC_W     = 0.0        # water-basis  noise: σ_w = FRAC_W · std(water sinogram).   ↑ → raises FLAT floor
    RHO        = 0.0        # Corr(δc_i, δc_w) ∈ [−1,1].  <0 → low-E cancellation → tips toward INCREASING
    NOISE_SEED = 1234

    io = copy(sino_basis.sino_iodine)
    wa = copy(sino_basis.sino_water)

    if APPLY
        rng = MersenneTwister(NOISE_SEED)
        σ_i = Float32(FRAC_I * std(vec(io)))
        σ_w = Float32(FRAC_W * std(vec(wa)))
        ρ   = Float32(RHO)
        ρc  = Float32(sqrt(max(1.0f0 - ρ^2, 0.0f0)))  # independent-component weight
        sz  = size(io)
        g1  = Array{Float32}(undef, sz[1], sz[2])     # iodine draw / shared component
        g2  = Array{Float32}(undef, sz[1], sz[2])     # independent component for water
        for v in 1:sz[3]
            randn!(rng, g1); randn!(rng, g2)
            # δc_i = σ_i·g1 ;  δc_w = σ_w·(ρ·g1 + √(1−ρ²)·g2)  → Corr(δc_i,δc_w) = ρ
            @views io[:, :, v] .+= σ_i .* g1
            @views wa[:, :, v] .+= σ_w .* (ρ .* g1 .+ ρc .* g2)
        end
        @info "[basis noise] σ_iodine = $(round(σ_i, sigdigits = 3)) g/cm² · σ_water = $(round(σ_w, sigdigits = 3)) g/cm² · ρ_basis = $(round(ρ, digits = 2)) " *
              (FRAC_W ≤ 1.0f-6 ? "(iodine-ONLY → expect MONOTONIC-DECREASING)" : "(mixed)")
    else
        @info "[basis noise] passthrough (APPLY = false)"
    end

    (sino_iodine = io, sino_water = wa, geom = sino_basis.geom)
end;

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
    io = basis_sino_noised.sino_iodine
    wa = basis_sino_noised.sino_water
    n  = length(io)

    # ── GLOBAL covariance (two scalar passes) ──
    si = 0.0; sw = 0.0
    @inbounds for k in eachindex(io); si += io[k]; sw += wa[k]; end
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
## 8. FBP: Iodine and Water Basis Maps

Two FDK passes — one per basis sinogram.  Output volumes are in
basis-density units (g/cm³).
"""

# ╔═╡ 08030008-0000-4000-8000-000000000010
basis_volumes = let
    matrix_size = recon_opts.matrix_size
    geom = basis_sino_noised.geom

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
        vol_iodine_raw = _fbp(basis_sino_noised.sino_iodine),
        vol_water_raw = _fbp(basis_sino_noised.sino_water),
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
## 8b. ACNR — Edge-Aware Anti-Correlated Noise Reduction (image domain)

Material decomposition stamps strongly **anti-correlated** noise onto the basis
maps (measured `ρ_basis ≈ −0.92`) — that anti-correlation *is* the VMI-noise U.
ACNR removes it, and resolution is preserved two independent ways:

1. **Structural guarantee at `E_ref`.** Only the channel *orthogonal* to the
   `E_ref` signal direction `(μρ_w, μρ_i)` is modified, so `μ(E_ref)` is
   **pixel-perfect at every voxel regardless of the smoother** — zero resolution
   loss at the anchor, by construction (`c_a·ΔW + c_b·ΔI ≡ 0`).
2. **Edge-aware everywhere else.** That orthogonal channel is smoothed with a
   **joint bilateral filter guided by BOTH basis maps**, so any real edge (water
   *or* iodine) survives; only locally-flat regions — pure anti-correlated noise —
   are smoothed.

Signal is positively correlated, noise negatively correlated → opposite subspaces,
so ACNR removes noise without touching structure.  The resolution check below shows
the *removed* component: it must be **structureless noise** (no rod rings).  Runs on
the FBP basis maps, **before** the §9 z-median.
"""

# ╔═╡ 08030008-0000-4000-8000-000000000055
# Edge-aware ACNR, inline (image domain).  Touches only the E_ref-orthogonal channel
# (μ(E_ref) preserved pixel-perfect), smoothing it with a joint bilateral guided by
# BOTH basis maps so real water/iodine edges survive.  Two passes (compute s_smooth
# fully, then back-project) so the bilateral reads unmodified neighbours.
basis_acnr = let
    APPLY_ACNR    = true
    ACNR_E_REF    = 70.0     # keV anchor — μ(E_ref) preserved pixel-perfect
    BILAT_RADIUS  = 3        # spatial window radius (px)
    BILAT_SIGMA_S = 2.0      # spatial Gaussian σ (px)
    BILAT_RANGE_K = 2.5      # range σ = K · per-basis noise std (edges > K·σ_noise preserved)
    GAMMA         = 1.0      # strength ∈ [0,1]; 0 = identity

    W = copy(basis_volumes.vol_water_raw)
    I = copy(basis_volumes.vol_iodine_raw)

    if APPLY_ACNR && GAMMA > 0
        nx, ny, nz = size(W)
        ca  = Float32(BS.compute_mass_μ_at_energy(BS.XA.Materials.water,  ACNR_E_REF))
        cb  = Float32(BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, ACNR_E_REF))
        csq = ca * ca + cb * cb
        αa  = Float32(GAMMA * cb / csq)
        αb  = Float32(GAMMA * ca / csq)

        # robust per-basis noise std (adjacent-column difference / √2)
        function _nstd(V)
            s = 0.0; m = 0
            @inbounds for k in 1:nz, j in 1:ny, i in 2:nx
                d = V[i, j, k] - V[i - 1, j, k]; s += d * d; m += 1
            end
            sqrt(s / max(m, 1)) / sqrt(2)
        end
        σW = max(_nstd(W), 1.0f-8); σI = max(_nstd(I), 1.0f-8)
        denW = Float32(2 * (BILAT_RANGE_K * σW)^2)
        denI = Float32(2 * (BILAT_RANGE_K * σI)^2)

        r   = BILAT_RADIUS
        σs2 = Float32(2 * BILAT_SIGMA_S^2)
        sw  = Float32[exp(-(di * di + dj * dj) / σs2) for di in -r:r, dj in -r:r]

        s_perp = Array{Float32}(undef, nx, ny, nz)
        @inbounds @. s_perp = -cb * W + ca * I

        # Pass 1: joint-bilateral smooth of s_perp, guided by (W, I) — read-only on W,I.
        s_smooth = Array{Float32}(undef, nx, ny, nz)
        @inbounds for k in 1:nz
            for j in 1:ny, i in 1:nx
                Wc = W[i, j, k]; Ic = I[i, j, k]
                acc = 0.0f0; wsum = 0.0f0
                for dj in -r:r
                    jj = j + dj; (1 ≤ jj ≤ ny) || continue
                    for di in -r:r
                        ii = i + di; (1 ≤ ii ≤ nx) || continue
                        dW = W[ii, jj, k] - Wc; dI = I[ii, jj, k] - Ic
                        wr = exp(-(dW * dW / denW + dI * dI / denI))
                        w  = sw[di + r + 1, dj + r + 1] * wr
                        acc += w * s_perp[ii, jj, k]; wsum += w
                    end
                end
                s_smooth[i, j, k] = acc / wsum
            end
        end

        # Pass 2: back-project the removed (anti-correlated noise) component, orthogonal
        # to the E_ref signal → μ(E_ref) unchanged at every voxel.
        @inbounds for idx in eachindex(W)
            n = s_perp[idx] - s_smooth[idx]
            W[idx] += αa * n
            I[idx] -= αb * n
        end
        @info "[ACNR · edge-aware] E_ref=$(ACNR_E_REF) keV · γ=$(GAMMA) · σ_noise(W)=$(round(σW, sigdigits = 3)), σ_noise(I)=$(round(σI, sigdigits = 3)) g/cm³ · range_k=$(BILAT_RANGE_K) · μ(E_ref) pixel-perfect preserved"
    else
        @info "[ACNR] OFF (passthrough)"
    end

    (vol_iodine_raw = I, vol_water_raw = W, geom = basis_volumes.geom)
end;

# ╔═╡ 08030008-0000-4000-8000-000000000058
# Resolution check — the REMOVED component (after − before) must be structureless
# noise.  If rod rings/edges show up in the right panel, ACNR is sacrificing
# resolution → lower γ or raise BILAT_RANGE_K.  Iodine basis (most edge-sensitive).
let
    mid     = size(basis_acnr.vol_iodine_raw, 3) ÷ 2 + 1
    before  = basis_volumes.vol_iodine_raw[:, :, mid]
    after   = basis_acnr.vol_iodine_raw[:, :, mid]
    removed = after .- before

    rng  = (Float64(quantile(vec(before), 0.01)), Float64(quantile(vec(before), 0.99)))
    rmax = max(maximum(abs.(removed)), 1.0f-12)

    fig = CM.Figure(size = (1500, 540))
    ak  = (titlesize = 26, subtitlesize = 18, aspect = CM.DataAspect())
    panels = (
        (1, "Iodine basis — before ACNR", "g/cm³",                          before,  :viridis, rng),
        (2, "Iodine basis — after ACNR",  "g/cm³",                          after,   :viridis, rng),
        (3, "REMOVED (after − before)",   "must be structureless noise",    removed, :balance, (-rmax, rmax)),
    )
    for (c, ttl, sub, sl, cm, cr) in panels
        ax = CM.Axis(fig[1, c]; title = ttl, subtitle = sub, ak...)
        CM.heatmap!(ax, sl; colormap = cm, colorrange = cr)
        CM.hidedecorations!(ax)
    end
    CM.Colorbar(fig[1, 4]; colormap = :balance, colorrange = (-rmax, rmax),
        label = "removed (g/cm³)", width = 16, labelsize = 20, ticklabelsize = 16)
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
            basis_acnr.vol_iodine_raw; adjacent_slices = Z_MEDIAN_ADJACENT
        ),
        vol_water = BS.apply_median_z(
            basis_acnr.vol_water_raw; adjacent_slices = Z_MEDIAN_ADJACENT
        ),
        geom = basis_acnr.geom,
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

# ╔═╡ 0803000d-0000-4000-8000-000000000015
# === Image-domain basis covariance (heart ROI) — the VMI-noise theorem inputs ===
# The quantity that ACTUALLY governs the U:
#   σ_HU(E)² = 1e6·V_w + α(E)²·V_i + 2e3·α(E)·C_iw ,   min at  α* = −1e3·C_iw/V_i
#   monotonic-DECREASING ⟺ α* ≤ α(140) ⟺ C_iw not too negative.
# Measured over the heart ROI on the z-medianed basis maps VMI synth consumes
# (c_iodine in mg/mL).  ρ_basis<0 = anti-correlated basis noise = the U + the ACNR
# target.  Small ROI → tiny vectors, no memory concern.
let
    roi = findall(heart_noise_roi.mask_2d)
    nz  = size(basis_z.vol_water, 3)
    cw  = Float64[Float64(basis_z.vol_water[p, z])           for z in 1:nz, p in roi]
    ci  = Float64[Float64(basis_z.vol_iodine[p, z]) * 1000.0 for z in 1:nz, p in roi]  # mg/mL
    mw = mean(cw); mi = mean(ci)
    Vw  = mean((cw .- mw) .^ 2); Vi = mean((ci .- mi) .^ 2)
    Ciw = mean((cw .- mw) .* (ci .- mi))
    ρ_b = Ciw / sqrt(max(Vw * Vi, 1.0e-30))
    αf(E) = Float64(BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, E)) /
            Float64(BS.compute_mass_μ_at_energy(BS.XA.Materials.water,  E))
    α_star = -1000.0 * Ciw / max(Vi, 1.0e-30)
    verdict = α_star ≤ αf(140.0) ? "MONOTONIC-decreasing predicted ✓" :
              α_star ≥ αf(40.0)  ? "monotonic-INCREASING" :
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
    nz_r = size(basis_z.vol_water, 3)

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
# ╟─08030005-0000-4000-8000-000000000001
# ╠═08030005-0000-4000-8000-000000000010
# ╠═08030005-0000-4000-8000-000000000020
# ╟─08030005-0000-4000-8000-000000000030
# ╟─08030007-0000-4000-8000-000000000001
# ╠═08030007-0000-4000-8000-000000000005
# ╠═08030007-0000-4000-8000-000000000010
# ╠═08030007-0000-4000-8000-000000000020
# ╟─08030007-0000-4000-8000-000000000040
# ╠═08030007-0000-4000-8000-000000000050
# ╠═08030007-0000-4000-8000-000000000060
# ╟─08030008-0000-4000-8000-000000000001
# ╠═08030008-0000-4000-8000-000000000010
# ╟─08030008-0000-4000-8000-000000000030
# ╟─08030008-0000-4000-8000-000000000050
# ╠═08030008-0000-4000-8000-000000000055
# ╟─08030008-0000-4000-8000-000000000058
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
