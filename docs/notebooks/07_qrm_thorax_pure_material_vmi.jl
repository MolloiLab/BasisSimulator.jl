### A Pluto.jl notebook ###
# v0.2.1

using Markdown
using InteractiveUtils

# ╔═╡ 07010003-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 07010003-0000-4000-8000-000000000005
using Markdown: @md_str, Markdown

# ╔═╡ 07010003-0000-4000-8000-000000000006
using Statistics: mean, std, quantile

# ╔═╡ 07010003-0000-4000-8000-000000000007
using Unitful: @u_str

# ╔═╡ 07010001-0000-4000-8000-000000000001
md"""
# QRM-Thorax Pure-Material VMI: Full-Resolution True-Scan Reference

**Intent:** the canonical full-fidelity reference for the projection-
domain dual-kVp pipeline.  Every knob here matches a real clinical
acquisition — no resolution compromises in-plane, no fake collimation,
no exaggerated mAs.  The only place we deliberately economize is **z**,
where we trim the phantom and recon to just the slices that are fully
covered by the cone beam (`z_usable = c × (1 − R_body/STI) ≈ 1.9 mm`
for 2.5 mm collimation) — so the truly 0.2 mm isotropic forward
projector still runs in reasonable wall-clock time.

| Stage         | Matrix                   | Voxel (mm)             | Extent                        |
|---------------|--------------------------|------------------------|-------------------------------|
| GT phantom    | **1600 × 1100 × 20**     | **0.2 isotropic**      | 320 × 220 × 4 mm              |
| Recon         | 512 × 512 × **3**        | **0.625 isotropic**    | FOV 32 cm × 1.875 mm z        |
| Collimation   | **2.5 mm at iso**        | —                      | ~7 detector rows lit          |
| Scanner       | GE Revolution Apex Elite | 0.625 × 0.6 mm pixels  | 256 × 834 detector            |
| Protocol      | DE rapid-kVp switching   | 80 + 140 kVp, 984 views, 0.5 s rot. | clinical DECT dose |

GE Apex Elite GSI rapid-kVp-switching simulation (80 + 140 kVp) on a
**QRM-Thorax-Gammex-Heart phantom** with **four pure-material rods**
(water · triolein · collagen · iodine) inserted in the heart cavity.
Same projection-domain VMI pipeline as notebook 03 — only the phantom
and the four rod inserts differ.

```
QRM-Thorax mid-slice mask  → relabel rods 9–12 → tile z → BS.Phantom
                                       │
       ┌───────────────────────────────┴───────────────────────────────┐
       │                                                               │
Simulate 80 kVp →┐                                                     │
                  ├─→ Material Decomp → FBP × 2 → z-Median →           │
Simulate 140 kVp→┘   2-basis VMI → Mono+ →                             │
                                                                       │
                     Per-Rod Measured-vs-Theoretical Regression  ──────┘
                                       at 40 / 70 / 100 / 140 keV
```

!!! tip "When to reach for this notebook"
    Use 07 when you want **clinical-realism evidence** — VMI accuracy
    on a body-sized phantom with a body-sized FOV at the resolutions
    a real GE Revolution Apex Elite would deliver.  For faster
    iteration on the pipeline itself, use notebook 03 (Gammex 472,
    same pipeline, smaller phantom).

!!! info "Why pure end-members?"
    `XA.Materials.basis_fat` is ICRU-44 adipose tissue (≈83 % triolein
    + 17 % water + trace electrolytes) — fine as a generic fat
    reference, but it's not a *mathematically pure* lipid.  The new
    `XA.Materials.basis_lipid` (H/C/O at 0.92 g/cm³) and
    `basis_collagen` (H/C/N/O at 1.26 g/cm³) end members were added in
    XrayAttenuation 0.3.0 specifically for clean water/lipid/collagen
    decomposition validation.
"""

# ╔═╡ 07010002-0000-4000-8000-000000000001
md"""
## Notebook Setup
"""

# ╔═╡ 07010003-0000-4000-8000-000000000010
import BasisSimulator as BS

# ╔═╡ 07010003-0000-4000-8000-000000000011
import CairoMakie as CM

# ╔═╡ 07010003-0000-4000-8000-000000000012
import PlutoUI

# ╔═╡ 07010003-0000-4000-8000-000000000013
PlutoUI.TableOfContents()

# ╔═╡ 07010003-0000-4000-8000-000000000040
begin
    import GPUSelect
    AT = GPUSelect.Storage()     # the backend array type, directly: MtlArray / CuArray / ROCArray
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end

# ╔═╡ 07010003-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 07020000-0000-4000-8000-0000000000f1
md"""
## Scan Setup and Simulation

One clinical GSI acquisition: the QRM-Thorax phantom, the GE Apex Elite
scanner, the dual-kVp switching protocol, and the forward projection that
turns them into 80 + 140 kVp sinograms.
"""

# ╔═╡ 07020001-0000-4000-8000-000000000001
md"""
### 01. `Phantom`: QRM-Thorax with 4 Pure-Material Rods

Read the **prepared** QRM-Thorax mask — already rotated to CT display
orientation (spine at the bottom) and 2× downsampled, cached at
`docs/notebooks/data/qrm_thorax/qrm_thorax_1600x1100_rot_uint8.raw`.
Phantom shape after z-tiling: **1600 × 1100 × 20** at **0.2 mm
isotropic** (physical extent 320 × 220 × 4 mm; body envelope ≈ 30 ×
20 cm matches QRM-Thorax-small spec).  Truly 0.2 mm iso ground
truth — finer than the demagged detector pitch (0.6 / 1.758 ≈ 0.34 mm),
so the forward projector sees a high-res source.  The z extent is
the **minimum** that covers all rays for 2.5 mm collimation (see ## 2).
The probe cell after the mask load prints the actual body bbox.

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

All four rod inserts and every anatomy region use prebuilt
`XA.Materials` directly — no inline material construction.
"""

# ╔═╡ 07020001-0000-4000-8000-000000000010
const QRM_CACHE_PATH = joinpath(@__DIR__, "data", "qrm_thorax", "qrm_thorax_1600x1100_rot_uint8.raw");

# ╔═╡ 07020001-0000-4000-8000-000000000011
const QRM_TARGET_NX = 1600;  # full prepared cache, no extra in-plane downsample

# ╔═╡ 07020001-0000-4000-8000-000000000012
const QRM_TARGET_NY = 1100;  # full prepared cache, no extra in-plane downsample

# ╔═╡ 07020001-0000-4000-8000-000000000013
const QRM_TARGET_NZ = 20;    # 20 × 0.2 mm = 4 mm — minimum phantom z that covers all rays for 2.5 mm collimation (see ## 2 cone-beam math)

# ╔═╡ 07020001-0000-4000-8000-000000000014
const QRM_VOXEL_SIZE_CM = (0.02, 0.02, 0.02);   # (x, y, z) cm — 0.2 mm isotropic ground truth (320 × 220 × 4 mm physical extent)

# ╔═╡ 07020001-0000-4000-8000-000000000015
# Share-drive copies of both prepared phantoms (rotated to CT display
# orientation) — for any lab member who doesn't already have the local
# cache.  `cp` either file into `docs/notebooks/data/qrm_thorax/`
# (relative to this notebook) and the cell below picks it up.
const QRM_SHARED_DRIVE_DIR = "/Volumes/Molloilab/Shu Nie/water-lipid";

# ╔═╡ 07020001-0000-4000-8000-000000000016
const QRM_SHARED_FULL_PATH = joinpath(QRM_SHARED_DRIVE_DIR, "qrm_thorax_3200x2200_rot_uint8.raw");

# ╔═╡ 07020001-0000-4000-8000-000000000017
const QRM_SHARED_DOWN_PATH = joinpath(QRM_SHARED_DRIVE_DIR, "qrm_thorax_1600x1100_rot_uint8.raw");

# ╔═╡ 07020001-0000-4000-8000-000000000018
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

# ╔═╡ 07020001-0000-4000-8000-000000000020
mask_3d_raw = let
    isfile(QRM_CACHE_PATH) || error(
        "QRM-Thorax cache not found at $(QRM_CACHE_PATH).\n" *
            "Either:\n" *
            "  • copy the prepared cache from the lab volume:\n" *
            "      cp \"$(QRM_SHARED_DOWN_PATH)\" \"$(QRM_CACHE_PATH)\"\n" *
            "  • or run the prep notebook once to rebuild the cache from source."
    )
    # Cache is 1600 × 1100 UInt8 at 0.2 mm in-plane.  Use it directly
    # as the ground-truth phantom — truly 0.2 mm isotropic after z-tile.
    cache_2d = reshape(read(QRM_CACHE_PATH), QRM_TARGET_NX, QRM_TARGET_NY)
    repeat(cache_2d; outer = (1, 1, QRM_TARGET_NZ))
end;

# ╔═╡ 07020001-0000-4000-8000-000000000022
let
    # Body bbox = bounding box of non-air voxels (label != 1) on the mid-slice.
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

# ╔═╡ 07020001-0000-4000-8000-000000000025
md"""
**Bore 4 rod inserts** into the heart cavity at cardinal positions
(N / E / S / W) and assign them new labels 9–12 to match
`materials_dict`.  The source mask doesn't carry rod inserts of its
own — we add them here procedurally.

Tune `ROD_HEART_CENTER_PX`, `ROD_RADIUS_MM`, `ROD_OFFSET_MM` to align
the rods with your phantom's actual heart cavity.  Defaults are set
for the 1600 × 1100 (320 × 220 mm) post-rotation phantom at 0.2 mm
in-plane.

Cardinal-direction → rod-label mapping (matches `materials_dict`):

| Direction | Label | Material              |
|-----------|-------|-----------------------|
| North     | 9     | `basis_water`         |
| East      | 10    | `basis_lipid`         |
| South     | 11    | `basis_collagen`      |
| West      | 12    | `gammex_472_i5_0`     |
"""

# ╔═╡ 07020001-0000-4000-8000-000000000026
# Heart-cavity center in voxel (px) coordinates on the loaded mask.
# Eyeball from the visualization and tune.  Invariant under voxel-size
# changes (only the mask shape matters).  Default lands inside the
# upper-half heart cavity of the rotated 1600 × 1100 phantom.
const ROD_HEART_CENTER_PX = (800, 654);   # ≈ (50%, 59%) of the 1600 × 1100 frame

# ╔═╡ 07020001-0000-4000-8000-000000000027
const ROD_RADIUS_MM = 7.5;                  # mm — each rod radius (physical)

# ╔═╡ 07020001-0000-4000-8000-000000000028
const ROD_OFFSET_MM = 25.0;                  # mm — heart-center → rod-center distance (physical)

# ╔═╡ 07020001-0000-4000-8000-000000000029
mask_3d = let
    out = copy(mask_3d_raw)
    nx, ny, nz = size(out)

    px_mm = QRM_VOXEL_SIZE_CM[1] * 10        # in-plane mm/voxel
    cx_px, cy_px = ROD_HEART_CENTER_PX
    r_px = ROD_RADIUS_MM / px_mm
    o_px = ROD_OFFSET_MM / px_mm

    # CairoMakie heatmap puts the y-axis going UP (math convention).
    # So +Δy in voxel coords = north (top of image).  Mapping cardinal
    # direction → (Δi, Δj, new_label):
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

# ╔═╡ 07020001-0000-4000-8000-000000000040
materials_dict = Dict{Int, BS.XA.Material}(
    # Anatomy
    1 => BS.XA.Materials.air,
    2 => BS.XA.Materials.lung,
    3 => BS.XA.Materials.muscle,           # "soft tissue" → ICRU-44 muscle
    4 => BS.XA.Materials.corticalbone,
    5 => BS.XA.Materials.marrow_red,
    6 => BS.XA.Materials.water,            # water rod (lung-side, calibration)
    7 => BS.XA.Materials.air,              # air rod
    8 => BS.XA.Materials.water,            # heart cavity → water (post-fill)
    # Pure-material rod inserts (relabeled from Ca-HA)
    9 => BS.XA.Materials.basis_water,      # rod 1 → water
    10 => BS.XA.Materials.basis_lipid,      # rod 2 → lipid (XA 0.3.0)
    11 => BS.XA.Materials.basis_collagen,   # rod 3 → collagen (XA 0.3.0)
    12 => BS.XA.Materials.gammex_472_i5_0,  # rod 4 → Gammex 472 5 mg/mL iodine insert
);

# ╔═╡ 07020001-0000-4000-8000-000000000050
phantom_cpu = BS.create_phantom_from_mask(
    Array{Int, 3}(mask_3d),
    materials_dict,
    QRM_VOXEL_SIZE_CM,
);

# ╔═╡ 07020001-0000-4000-8000-000000000051
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 07020001-0000-4000-8000-000000000060
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

# ╔═╡ 07020001-0000-4000-8000-000000000061
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

# ╔═╡ 07020001-0000-4000-8000-000000000062
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

# ╔═╡ 07030001-0000-4000-8000-000000000001
md"""
### 02. `Scanner`: GE Revolution Apex Elite
"""

# ╔═╡ 07030001-0000-4000-8000-000000000010
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

# ╔═╡ 07030002-0000-4000-8000-000000000001
md"""
### 03. Dual-kVp Protocols (Rapid kVp Switching)

| kVp | Instantaneous mA | Duty cycle | Effective mA |
|-----|------------------|------------|--------------|
| 80  | 407              | 0.65       | 264.55       |
| 140 | 405              | 0.35       | 141.75       |

**Collimation = 2.5 mm at iso** (clinical thin-slice DE setting).
Chosen so 3 recon slices @ 0.625 mm fit inside the usable-z budget
`z_usable = c × (1 − R_body/STI) ≈ 1.9 mm`.  Only ~7 detector rows
are lit — keeps the forward-projection cost low so we can afford a
truly 0.2 mm iso ground-truth phantom.
"""

# ╔═╡ 07030002-0000-4000-8000-000000000010
protocol_low = BS.CTProtocol(
    kVp = 80,
    mA = 407 * 0.65,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 2.5,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 07030002-0000-4000-8000-000000000020
protocol_high = BS.CTProtocol(
    kVp = 140,
    mA = 405 * 0.35,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 2.5,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 07030003-0000-4000-8000-000000000001
md"""
### 04. `SimOptions` and `ReconOptions`
"""

# ╔═╡ 07030003-0000-4000-8000-000000000010
sim_opts = BS.SimOptions(fidelity = :eict, seed = 1234, projector = :dd_fast);

# ╔═╡ 07030003-0000-4000-8000-000000000020
# Standard CT recon convention: 512 × 512 in-plane at 0.625 mm isotropic
# (GE clinical thin-slice).  The phantom is the high-res forward-projector
# sampling source (1600 × 1100 × 20 at 0.2 mm iso); the recon grid is
# independent (512 × 512 × 3 at 0.625 mm iso, fov 32 cm, 1.875 mm z).
# Rod ROIs are built in physical mm and converted to recon voxel indices —
# the two grids share the isocenter so the conversion is just a centered
# linear transform.  Recon z extent matches the usable-z budget for
# 2.5 mm collimation: z_usable = c × (1 − R_body/STI) ≈ 1.9 mm.
recon_opts = BS.ReconOptions(
    matrix_size = (512, 512, 3),
    fov_cm = 32.0,
    z_cm = 0.1875,
);

# ╔═╡ 07030004-0000-4000-8000-000000000001
md"""
### 05. Forward Project

Run `BS.simulate!` on each kVp protocol.  The EICT path bakes in
per-ray spatial scatter + Compton + Rayleigh, and we keep the
simulator's noisy line-integral sinogram (`ws.sinogram`).
"""

# ╔═╡ 07030004-0000-4000-8000-000000000010
sim_low = let
    @info "Simulating: 80 kVp / $(round(protocol_low.mA, digits = 1)) mA-eff (DE low)…"
    ws = BS.create_eict_workspace(scanner, protocol_low, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, protocol_low, sim_opts)
    # Per-ray air-scan counts I0(col,row) for the Jensen debias.
    I0_scalar = BS.compute_detector_I0(ws.geom, protocol_low, sum(ws.weights)) * Float64(ws.η_eff)
    air_ref = ws.bowtie_air_reference === nothing ? ones(Float32, ws.geom.n_cols, ws.geom.n_rows) :
        Array(ws.bowtie_air_reference)
    result = (sino = Array(ws.sinogram), geom = ws.geom,
        I0_ray = Float32.(I0_scalar .* Float64.(air_ref)))
    ws = nothing; GC.gc(true)
    result
end;

# ╔═╡ 07030004-0000-4000-8000-000000000020
sim_high = let
    @info "Simulating: 140 kVp / $(round(protocol_high.mA, digits = 1)) mA-eff (DE high)…"
    ws = BS.create_eict_workspace(scanner, protocol_high, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, protocol_high, sim_opts)
    # Per-ray air-scan counts I0(col,row) for the Jensen debias.
    I0_scalar = BS.compute_detector_I0(ws.geom, protocol_high, sum(ws.weights)) * Float64(ws.η_eff)
    air_ref = ws.bowtie_air_reference === nothing ? ones(Float32, ws.geom.n_cols, ws.geom.n_rows) :
        Array(ws.bowtie_air_reference)
    result = (sino = Array(ws.sinogram), geom = ws.geom,
        I0_ray = Float32.(I0_scalar .* Float64.(air_ref)))
    ws = nothing; GC.gc(true)
    result
end;

# ╔═╡ 07030004-0000-4000-8000-000000000025
# Resample the phantom labels onto the recon grid via BS's affine
# round-trip (see notebook 05 §10 for the full story on
# `phantom_to_world_affine` / `recon_to_world_affine`).  Result:
# a `recon-shape` UInt8 mask (512 × 512 × 8) where each recon voxel
# carries the phantom label that overlaps with it.  Used downstream
# to build all ROI masks (rod cores, heart-cavity SW disc) directly
# in recon-grid coordinates — no hand-rolled coord conversions.
phantom_in_recon = BS.resample_to_recon(
    phantom_cpu, sim_low.geom, recon_opts.matrix_size; method = :nearest,
);

# ╔═╡ 07030004-0000-4000-8000-000000000030
let
    n_row = size(sim_low.sino, 2)
    mid_r = n_row ÷ 2 + 1

    slice_lo = permutedims(sim_low.sino[:, mid_r, :], (2, 1))
    slice_hi = permutedims(sim_high.sino[:, mid_r, :], (2, 1))

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

    for (c, ttl, slice) in ((1, "80 kVp", slice_lo), (2, "140 kVp", slice_hi))
        ax = CM.Axis(fig[1, c]; title = ttl, axis_kwargs...)
        CM.heatmap!(ax, slice; colormap = :viridis, colorrange = sino_window)
    end
    CM.Colorbar(
        fig[1, 3]; colormap = :viridis, colorrange = sino_window,
        label = "Log Line Integral", width = 16, labelsize = 22, ticklabelsize = 18
    )
    fig
end

# ╔═╡ 07030000-0000-4000-8000-0000000000f1
md"""
## VMI Pipeline

Projection-domain material decomposition → per-basis FBP → Kalender-1988
ACNR → monoenergetic synthesis.  The same certified chain as notebook 03.
"""

# ╔═╡ 07030006-0000-4000-8000-000000000001
md"""
### 01. Projection-Domain Material Decomposition

Per-ray Newton solver on the polychromatic transmission integral with a
fixed `(water, iodine)` basis seeded by `water_basis = (a = 0, c = 1)`.

The basis builder constructs the **per-ray effective spectrum** that
`simulate!` actually applied to the rays — tube × flat filter × bowtie ×
heel × η(E) — and normalizes per ray so `Σ_E ŵ[col, row, E] = 1`.  This
matches the spectrum baked into the EICT workspace's `bowtie_air_reference`,
so Cong's inversion solves the same polychromatic transmission integral
the forward model used.
"""

# ╔═╡ 07030006-0000-4000-8000-000000000010
# Per-ray effective spectrum (source × bowtie × heel × η) that simulate!
# applied.  `BS.resolve_source_spectrum_full` chains the same `PhysicsConfig`
# the EICT workspace built, so the inversion sees exactly the spectrum the
# forward model used (heel + η are conditionally applied per the SimOptions
# `use_*` flags).
material_basis = let
    iodine_mat = BS.XA.Elements.Iodine
    water_mat = BS.XA.Materials.water

    e_L, ŵ_L = BS.resolve_source_spectrum_full(
        sim_opts, protocol_low;
        scanner = scanner, geom = sim_low.geom, phantom = phantom,
        diagnostic = true, label = "low",
    )
    e_H, ŵ_H = BS.resolve_source_spectrum_full(
        sim_opts, protocol_high;
        scanner = scanner, geom = sim_high.geom, phantom = phantom,
        diagnostic = true, label = "high",
    )

    p_L = Float32[Float32(BS.compute_mass_μ_at_energy(iodine_mat, Float64(E))) for E in e_L]
    q_L = Float32[Float32(BS.compute_mass_μ_at_energy(water_mat, Float64(E))) for E in e_L]
    p_H = Float32[Float32(BS.compute_mass_μ_at_energy(iodine_mat, Float64(E))) for E in e_H]
    q_H = Float32[Float32(BS.compute_mass_μ_at_energy(water_mat, Float64(E))) for E in e_H]

    (
        ŵ_L = ŵ_L, p_L = p_L, q_L = q_L,
        ŵ_H = ŵ_H, p_H = p_H, q_H = q_H,
    )
end;

# ╔═╡ 07030006-0000-4000-8000-000000000020
# Pure Cong polychromatic per-ray decomposition.
#
# Earlier this cell ran a Cong → PRISM-regularized-denoise hybrid, but the
# physics audit (see SimOptions notes above) revealed that the perceived
# bias was an un-corrected fill_factor + optical_crosstalk effect, not a
# decomposition deficiency.  With those flags off, raw Cong reaches the
# theoretical curves on its own, so the PRISM denoise step adds nothing
# at the noiseless / corrected-physics operating point.  The PRISM
# machinery (§6.5 / §6.6 cells) is left inline as dormant code in case
# we want to revisit it for noise-handling experiments.
sino_basis = let
    # Use the §5.5-corrected sinograms (fill_factor offset removed + crosstalk
    # deconvolved if enabled in sim_opts).  When both `use_*` flags are false
    # the corrections are no-ops and these are identical to sim_low.sino /
    # sim_high.sino.
    # First-order log-Poisson DEBIAS (deterministic; matches nb03):
    # E[−log(N/I0)] = p_true + 1/(2N), N = I₀(col,row)·e^{−p}.  Removes the
    # low-count log bias behind dense rods before the solve.
    debias(p, I0_ray) = begin
        out = Float32.(p)
        nc, nr, nv = size(out)
        for v in 1:nv, r in 1:nr, c in 1:nc
            N = max(I0_ray[c, r] * exp(-out[c, r, v]), 1.0f0)
            out[c, r, v] -= 1.0f0 / (2.0f0 * N)
        end
        out
    end
    sino_low_gpu = to_gpu(debias(sim_low.sino, sim_low.I0_ray))
    sino_high_gpu = to_gpu(debias(sim_high.sino, sim_high.I0_ray))

    sino_y = similar(sino_low_gpu);  fill!(sino_y, 0.0f0)
    sino_c = similar(sino_low_gpu);  fill!(sino_c, 0.0f0)

    @info "Cong polychromatic decomposition: $(size(sim_low.sino))"
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
        geom = sim_low.geom,
    )
    sino_low_gpu = nothing; sino_high_gpu = nothing
    sino_y = nothing; sino_c = nothing; cong_ws = nothing
    GC.gc(true)
    result
end;

# ╔═╡ 07030006-0000-4000-8000-000000000040
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

# ╔═╡ 07030007-0000-4000-8000-000000000001
md"""
### 02. FBP: Iodine and Water Basis Maps

Two FDK passes with `BS.SoftFilter()` — one per basis sinogram.
Output volumes are in basis-density units (g/cm³).
"""

# ╔═╡ 07030007-0000-4000-8000-000000000010
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

# ╔═╡ 07030007-0000-4000-8000-000000000030
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

# ╔═╡ 07030007-0000-4000-8000-000000000040
md"""
### 03. Kalender-1988 TRUE ACNR

Per-pixel local regression between the two basis maps' high-frequency
channels (`BS.apply_acnr_kalender!`).  Dual-energy basis noise is
ANTI-correlated while structure is positively correlated, so the
anti-correlated (noise) component is subtracted exactly and structure
pixels stay bit-untouched — zero resolution loss by construction.
Identical to nb03.
"""

# ╔═╡ 07030007-0000-4000-8000-000000000050
basis_acnr = let
    W = copy(basis_volumes.vol_water_raw)
    I = copy(basis_volumes.vol_iodine_raw)
    info = BS.apply_acnr_kalender!(W, I)
    @info "[ACNR · Kalender-1988 true ACNR] ρ_hp(W,I)=$(round(info.ρ_hp, digits = 3))"
    (vol_iodine_raw = I, vol_water_raw = W, geom = basis_volumes.geom)
end;

# ╔═╡ 07030008-0000-4000-8000-000000000030
let
    fig = CM.Figure(size = (1180, 580))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    mid = size(basis_acnr.vol_iodine_raw, 3) ÷ 2

    _qrange(arr) = (
        Float64(quantile(vec(arr), 0.01)),
        Float64(quantile(vec(arr), 0.99)),
    )

    slice_iod = basis_acnr.vol_iodine_raw[:, :, mid]
    slice_wat = basis_acnr.vol_water_raw[:, :, mid]

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

# ╔═╡ 07030009-0000-4000-8000-000000000001
md"""
### 04. VMI Synthesis

`BS.synth_vmi_2basis(c_water, c_iodine_mg_per_mL; energy_keV)` evaluates
the textbook 2-basis linear mix (McCollough 2015) at the target keV:

```
μ(E)  = c_water(r) · (μ/ρ)_water(E) + c_iodine(r) · (μ/ρ)_iodine(E)
HU(E) = 1000 · (μ(E) − (μ/ρ)_water(E)) / (μ/ρ)_water(E)
```

VMI grid: 40, 70, 100, 140 keV.  The `solid_water_basis` cell measures
`⟨c_water⟩` and `⟨c_iodine⟩` over the **dedicated water-rod core ROI**
(label 9, `basis_water`) — a diagnostic used to log Δ% drift between
the water-rod-anchored synth μ_water and the textbook mono divisor.
"""

# ╔═╡ 07030009-0000-4000-8000-000000000005
solid_water_basis = let
    # SW reference = the dedicated water rod (label 9, basis_water).
    # We bored it specifically to give a clean pure-water reference;
    # no need to hunt for the heart cavity.  An 8-px-radius core ROI
    # at the water-rod centroid in recon coords gives the SW means.
    WATER_ROD_LABEL = UInt8(9)
    ROI_RADIUS_PX = 8

    mask_2d = phantom_in_recon[:, :, size(phantom_in_recon, 3) ÷ 2 + 1]
    nx_r, ny_r, nz_r = size(basis_acnr.vol_water_raw)

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

    c_w = Float64(_mean(basis_acnr.vol_water_raw))
    c_i = Float64(_mean(basis_acnr.vol_iodine_raw))
    @info "solid_water_basis: ⟨c_water⟩_water-rod = $(round(c_w, digits = 4)) g/cm³, " *
        "⟨c_iodine⟩_water-rod = $(round(c_i, digits = 6)) g/cm³"

    (
        c_water = c_w, c_iodine = c_i, n_voxels = length(sw_idx) * nz_r,
        mask_2d = collect(sw_bool),
    )
end;

# ╔═╡ 07030009-0000-4000-8000-000000000010
de_vmi_energies = [40.0, 70.0, 100.0, 140.0];

# ╔═╡ 07030009-0000-4000-8000-000000000020
vmi_HU_final = let
    c_iodine_mg_per_mL = basis_acnr.vol_iodine_raw .* 1000.0f0

    out = Dict{Float64, Array{Float32, 3}}()
    for E in de_vmi_energies
        μρ_w = BS.compute_mass_μ_at_energy(BS.XA.Materials.water, E)
        μρ_I = BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, E)
        μ_water_anchor = solid_water_basis.c_water * μρ_w +
            solid_water_basis.c_iodine * μρ_I
        Δ_pct = 100.0 * (μ_water_anchor - μρ_w) / μρ_w
        @info "VMI synth @ $(Int(E)) keV: divisor = $(round(μρ_w, digits = 5)) cm⁻¹ " *
            "(mono μρ_water);  SW-ROI anchor = $(round(μ_water_anchor, digits = 5)) " *
            "→ Δ = $(round(Δ_pct, digits = 2))%"

        out[E] = BS.synth_vmi_2basis(
            basis_acnr.vol_water_raw, c_iodine_mg_per_mL; energy_keV = E,
        )
    end
    out
end;

# ╔═╡ 07030009-0000-4000-8000-000000000040
let
    HU_window = (-200, 500)

    fig = CM.Figure(size = (1180, 1180))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    sample = vmi_HU_final[40.0]
    mid = size(sample, 3) ÷ 2

    for (k, E) in enumerate(de_vmi_energies)
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
    fig
end

# ╔═╡ 0703000a-0000-4000-8000-000000000040
let
    HU_window = (-200, 500)

    fig = CM.Figure(size = (1180, 1180))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    sample = vmi_HU_final[40.0]
    mid = size(sample, 3) ÷ 2

    for (k, E) in enumerate(de_vmi_energies)
        r = ((k - 1) ÷ 2) + 1
        c = ((k - 1) % 2) + 1
        ax = CM.Axis(
            fig[r, c]; title = "$(Int(E)) keV VMI",
            subtitle = "VMI",
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

# ╔═╡ 0703000b-0000-4000-8000-000000000000
md"""
### Phantom-Recon Alignment Verification

Sanity-check the BS affine round-trip before trusting any ROI built on
`phantom_in_recon`.  Same pattern as notebook 05's verification mosaic:
recon HU and the resampled phantom mask side-by-side on the same
recon grid, then overlaid to confirm rod / heart-cavity / bone-wall
edges land cleanly on the corresponding recon edges.
"""

# ╔═╡ 0703000b-0000-4000-8000-00000000000a
let
    z_recon = size(vmi_HU_final[70.0], 3) ÷ 2 + 1
    hu_slice = vmi_HU_final[70.0][:, :, z_recon]

    z_pir = clamp(z_recon, 1, size(phantom_in_recon, 3))
    pir_slice = phantom_in_recon[:, :, z_pir]

    # NaN-masked overlay — render non-air & non-rod voxels transparent so the
    # rod inserts (labels 9–12) show on top of the HU recon for direct edge
    # comparison.
    rod_overlay = let
        out = fill(NaN32, size(pir_slice))
        @inbounds for idx in eachindex(pir_slice)
            v = Int(pir_slice[idx])
            (v == 9 || v == 10 || v == 11 || v == 12) && (out[idx] = Float32(v))
        end
        out
    end

    # Full-mask overlay (every label except air = transparent).
    full_overlay = let
        out = Float32.(pir_slice)
        @inbounds for idx in eachindex(out)
            out[idx] == 1.0f0 && (out[idx] = NaN32)   # air → transparent
        end
        out
    end

    fig = CM.Figure(size = (1400, 1320))
    hu_kwargs = (colormap = :grays, colorrange = (-200, 500))
    title_kwargs = (titlesize = 28, subtitlesize = 20)

    # Top-left: raw HU recon.
    ax_tl = CM.Axis(
        fig[1, 1];
        title = "70 keV Mono+ HU recon",
        subtitle = "z = $(z_recon) of $(size(vmi_HU_final[70.0], 3))",
        aspect = CM.DataAspect(),
        title_kwargs...,
    )
    CM.heatmap!(ax_tl, hu_slice; hu_kwargs...)
    CM.hidedecorations!(ax_tl)

    # Top-right: ALL labels — phantom_in_recon multi-label mask.
    ax_tr = CM.Axis(
        fig[1, 2];
        title = "phantom_in_recon (`:nearest` resample)",
        subtitle = "$(length(unique(pir_slice))) unique labels on recon grid",
        aspect = CM.DataAspect(),
        title_kwargs...,
    )
    CM.heatmap!(ax_tr, Float32.(pir_slice); colormap = :tab20)
    CM.hidedecorations!(ax_tr)

    # Bottom-left: HU recon + rod labels (9–12) overlaid.
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

    # Bottom-right: HU + full label mask (every non-air label).
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

# ╔═╡ 0703000b-0000-4000-8000-000000000001
md"""
## Results

Per-rod measured vs theoretical HU at 40 / 70 / 100 / 140 keV.

- **Measured HU** = mean over an 8-px-radius circular ROI at each rod
  centroid, broadcast across all z slices.
- **Theoretical HU** = `1000 · (μ_r(E) − μ_water(E)) / μ_water(E)` from
  `BS.compute_μ_at_energy(material, E)` — pure physics, no fitting.
"""

# ╔═╡ 0703000b-0000-4000-8000-000000000002
md"""
### Water ROI

Water-rod core ROI (label 9 = `basis_water`) overlaid in red on the
70 keV Mono+ slice — the voxels feeding the `solid_water_basis`
diagnostic.  Right panel: mean HU over that ROI vs VMI energy.  Bars
should cluster near 0 HU; consistent ~few-HU offset = residual basis-
decomp bias, energy-dependent drift = upstream spectral problem.
"""

# ╔═╡ 0703000b-0000-4000-8000-000000000003
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
    sw_hu_per_keV = [_mean_hu(vmi_HU_final[E]) for E in de_vmi_energies]

    n_E = length(de_vmi_energies)
    bar_colors = [CM.cgrad(:plasma, n_E; categorical = true)[i] for i in 1:n_E]

    ax2 = CM.Axis(
        fig[1, 2];
        title = "Water ROI Mean HU",
        subtitle = "Per VMI Energy",
        xlabel = "VMI Energy (keV)", ylabel = "HU",
        xticks = (collect(1:n_E), ["$(Int(E))" for E in de_vmi_energies]),
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
        joinpath(
            @__DIR__, "..", "assets",
            "qrm_thorax_vmi_water_roi_check.png"
        ), fig; px_per_unit = 2
    )
    fig
end

# ╔═╡ 0703000d-0000-4000-8000-000000000001
md"""
### Heart-Center Noise ROI

HU noise (σ) inside a circular ROI offset from the **water-rod centroid**
(label 9 — the only label guaranteed present in `phantom_in_recon`
after `:nearest` resample at 0.625 mm).  Default offset
`(dx, dy) = (0, -16)` puts the ROI ~10 mm below the water rod, in the
heart cavity between the rod and the cavity center — tweak
`HEART_NOISE_ROI_OFFSET_PX` to iterate, the left panel below shows
exactly where the ROI lands on the 70 keV Mono+ slice.

Right panel = σ vs VMI energy.  Diagnoses how the textbook
(c_water, c_iodine) → HU(E) synth propagates noise.  Expectation for
80/140 kVp DECT: σ(40) ≫ σ(70) ≳ σ(140) — the natural noise-optimal
energy is near 70 keV.  The streaks visible at 140 keV are *bias*
surfacing once σ drops below the bias amplitude.
"""

# ╔═╡ 0703000d-0000-4000-8000-000000000004
const HEART_NOISE_ROI_OFFSET_PX = (0, -40);   # (dx, dy) recon vx — default: 10 mm below water-rod centroid

# ╔═╡ 0703000d-0000-4000-8000-000000000005
const HEART_NOISE_ROI_RADIUS_PX = 12;   # ≈7.5 mm at 0.625 mm/voxel

# ╔═╡ 0703000d-0000-4000-8000-000000000010
heart_noise_roi = let
    WATER_ROD_LABEL = UInt8(9)   # water rod (basis_water) — bored explicitly, always present

    mask_2d = phantom_in_recon[:, :, size(phantom_in_recon, 3) ÷ 2 + 1]
    nx_r, ny_r, nz_r = size(basis_acnr.vol_water_raw)

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

# ╔═╡ 0703000d-0000-4000-8000-000000000020
vmi_noise_by_keV = let
    roi_idx = findall(heart_noise_roi.mask_2d)
    nz_r = size(basis_acnr.vol_water_raw, 3)

    out = Dict{Float64, NamedTuple}()
    for E in de_vmi_energies
        vol = vmi_HU_final[E]
        vals = Float64[Float64(vol[ci, z]) for z in 1:nz_r, ci in roi_idx]
        μ = mean(vals)
        σ = std(vals)
        out[E] = (mean = μ, std = σ, n = length(vals))
        @info "heart noise @ $(Int(E)) keV: ⟨HU⟩ = $(round(μ, digits = 2)),  σ = $(round(σ, digits = 2)) HU  (n = $(length(vals)))"
    end
    out
end;

# ╔═╡ 0703000d-0000-4000-8000-000000000030
let
    HU_window = (-200, 500)
    mid = size(vmi_HU_final[70.0], 3) ÷ 2
    bg = vmi_HU_final[70.0][:, :, mid]

    overlay = Float32[b ? 1.0f0 : NaN32 for b in heart_noise_roi.mask_2d]

    fig = CM.Figure(size = (1180, 580))

    ax1 = CM.Axis(
        fig[1, 1];
        title = "Heart-Center Noise ROI",
        # subtitle = "Overlaid on 70 keV Mono+ · offset $(HEART_NOISE_ROI_OFFSET_PX) px from water rod",
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
        # subtitle = "$(heart_noise_roi.n_voxels) vx × $(size(basis_acnr.vol_water_raw, 3)) z = $(heart_noise_roi.n_total) samples / point",
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

    CM.save(
        joinpath(
            @__DIR__, "..", "assets",
            "qrm_thorax_vmi_noise_roi.png"
        ), fig; px_per_unit = 2
    )
    fig
end

# ╔═╡ 0703000b-0000-4000-8000-000000000010
const ROD_LABELS = (UInt8(9), UInt8(10), UInt8(11), UInt8(12));

# ╔═╡ 0703000b-0000-4000-8000-000000000011
const ROD_NAMES = ("water", "lipid", "collagen", "iodine_5");

# ╔═╡ 0703000b-0000-4000-8000-000000000012
const ROD_MATERIALS = (
    BS.XA.Materials.basis_water,
    BS.XA.Materials.basis_lipid,
    BS.XA.Materials.basis_collagen,
    BS.XA.Materials.gammex_472_i5_0,
);

# ╔═╡ 07030009-0000-4000-8000-000000000008
# Per-rod basis-decomp sanity check.  For each of the 4 rods, measure
# the mean (c_water, c_iodine) inside an 8-px-radius core ROI at the
# rod centroid and compare to the rod material's density.
#
# Expected:
#   rod 9  (basis_water)    → c_water ≈ 1.00 g/cm³, c_iodine ≈ 0
#   rod 10 (basis_lipid)    → c_water ≈ 0.92,        c_iodine ≈ 0
#   rod 11 (basis_collagen) → c_water ≈ 1.26,        c_iodine ≈ 0
#   rod 12 (gammex_472_i5_0)→ c_water ≈ 1.0,         c_iodine ≈ 0.005 (5 mg/mL)
#
# If c_water for the water rod (label 9) is ~0.5 instead of ~1.0, the
# basis decomp itself has a 2× scaling issue and the SW-ROI Δ_pct = -51%
# is downstream of that.  If rod values look right but the heart-cavity
# SW-ROI still reads c_water ≈ 0.5, the SW disc is sampling material
# that isn't water-equivalent (e.g. air-rod label inside the disc).
let
    mask_2d = phantom_in_recon[:, :, size(phantom_in_recon, 3) ÷ 2 + 1]
    nx, ny = size(mask_2d)
    ROI_R_PX = 8

    n_z = size(basis_acnr.vol_water_raw, 3)

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
        c_w = _mean(basis_acnr.vol_water_raw, roi)
        c_i = _mean(basis_acnr.vol_iodine_raw, roi)
        ρ = round(BS.XA.val(mat.density), digits = 3)
        @info "  rod $(lab) ($(rpad(name, 9))): c_water = $(round(c_w, digits = 4)) g/cm³, " *
            "c_iodine = $(round(c_i, digits = 6)) g/cm³  (truth ρ = $(ρ))"
    end
end

# ╔═╡ 0703000b-0000-4000-8000-000000000020
rod_data = let
    # Operate in RECON-grid coords via `phantom_in_recon` (the phantom
    # labels resampled onto the recon grid by BS's affine round-trip).
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

    n_rods = length(ROD_LABELS)
    n_E = length(de_vmi_energies)
    meas = zeros(Float64, n_rods, n_E)
    theo = zeros(Float64, n_rods, n_E)
    for (i, lab) in enumerate(ROD_LABELS)
        mat = ROD_MATERIALS[i]
        for (j, E) in enumerate(de_vmi_energies)
            meas[i, j] = measured_hu(vmi_HU_final[E], lab)
            theo[i, j] = theoretical_hu(mat, E)
        end
    end
    (
        labels = ROD_LABELS, names = ROD_NAMES,
        measured = meas, theoretical = theo,
    )
end;

# ╔═╡ 0703000b-0000-4000-8000-000000000030
md"""
### Per-Rod Regression
"""

# ╔═╡ 0703000b-0000-4000-8000-000000000031
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
        title = "Pure-Material Rods",
        subtitle = "40 / 70 / 100 / 140 keV",
        xlabel = "VMI energy (keV)", ylabel = "HU",
        xticks = de_vmi_energies,
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
            ax, de_vmi_energies, vec(rod_data.measured[i, :]);
            color = c, linewidth = 2.5, markersize = 9
        )
        CM.lines!(
            ax, de_vmi_energies, vec(rod_data.theoretical[i, :]);
            color = c, linewidth = 1.6, linestyle = :dash
        )
        rod_lines[i] = CM.LineElement(color = c, linewidth = 2.5)

        meas_i = vec(rod_data.measured[i, :])
        theo_i = vec(rod_data.theoretical[i, :])
        rmse_i = sqrt(mean((meas_i .- theo_i) .^ 2))
        # nRMSE = RMSE / max(|theoretical|) × 100% — peak-relative error.
        # Less sensitive to keV points where theoretical magnitude is
        # small (e.g. iodine at 140 keV) than the mean-magnitude form.
        # Skipped for water: water's theoretical HU is 0 by definition
        # (the reference), so the denominator collapses and nRMSE is
        # undefined.
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
        joinpath(
            @__DIR__, "..", "assets",
            "qrm_thorax_vmi_vs_theoretical.png"
        ), fig; px_per_unit = 2
    )
    fig
end

# ╔═╡ 0703000c-0000-4000-8000-000000000001
md"""
## Summary

```
QRM-Thorax mid-slice mask (1600 × 1100 × 20 phantom @ 0.2 mm iso,
                           rods bored at labels 9–12)
   → Forward-project (80 + 140 kVp dual-kVp GSI)
   → Projection-Domain Material Decomposition  (iodine + water basis)
   → FBP × 2  (iodine, water basis maps)
   → Monoenergetic VMI Synthesis  (textbook 2-basis, mono μρ_water divisor)
   → Per-rod Measured vs Theoretical Regression
        (water · lipid · collagen · iodine at 40 / 70 / 100 / 140 keV)
```

1:1 parity with notebook 03's projection-domain dual-kVp pipeline,
swapping the Gammex 472 phantom for the QRM-Thorax phantom with four
pure-material rod inserts (`basis_water`, `basis_lipid`,
`basis_collagen`, `gammex_472_i5_0`).
"""

# ╔═╡ Cell order:
# ╟─07010001-0000-4000-8000-000000000001
# ╟─07010002-0000-4000-8000-000000000001
# ╠═07010003-0000-4000-8000-000000000001
# ╠═07010003-0000-4000-8000-000000000005
# ╠═07010003-0000-4000-8000-000000000006
# ╠═07010003-0000-4000-8000-000000000007
# ╠═07010003-0000-4000-8000-000000000010
# ╠═07010003-0000-4000-8000-000000000011
# ╠═07010003-0000-4000-8000-000000000012
# ╠═07010003-0000-4000-8000-000000000013
# ╠═07010003-0000-4000-8000-000000000040
# ╟─07010003-0000-4000-8000-000000000050
# ╟─07020000-0000-4000-8000-0000000000f1
# ╟─07020001-0000-4000-8000-000000000001
# ╠═07020001-0000-4000-8000-000000000010
# ╠═07020001-0000-4000-8000-000000000011
# ╠═07020001-0000-4000-8000-000000000012
# ╠═07020001-0000-4000-8000-000000000013
# ╠═07020001-0000-4000-8000-000000000014
# ╠═07020001-0000-4000-8000-000000000015
# ╠═07020001-0000-4000-8000-000000000016
# ╠═07020001-0000-4000-8000-000000000017
# ╟─07020001-0000-4000-8000-000000000018
# ╠═07020001-0000-4000-8000-000000000020
# ╟─07020001-0000-4000-8000-000000000022
# ╟─07020001-0000-4000-8000-000000000025
# ╠═07020001-0000-4000-8000-000000000026
# ╠═07020001-0000-4000-8000-000000000027
# ╠═07020001-0000-4000-8000-000000000028
# ╠═07020001-0000-4000-8000-000000000029
# ╠═07020001-0000-4000-8000-000000000040
# ╠═07020001-0000-4000-8000-000000000050
# ╠═07020001-0000-4000-8000-000000000051
# ╠═07020001-0000-4000-8000-000000000060
# ╟─07020001-0000-4000-8000-000000000061
# ╟─07020001-0000-4000-8000-000000000062
# ╟─07030001-0000-4000-8000-000000000001
# ╠═07030001-0000-4000-8000-000000000010
# ╟─07030002-0000-4000-8000-000000000001
# ╠═07030002-0000-4000-8000-000000000010
# ╠═07030002-0000-4000-8000-000000000020
# ╟─07030003-0000-4000-8000-000000000001
# ╠═07030003-0000-4000-8000-000000000010
# ╠═07030003-0000-4000-8000-000000000020
# ╟─07030004-0000-4000-8000-000000000001
# ╠═07030004-0000-4000-8000-000000000010
# ╠═07030004-0000-4000-8000-000000000020
# ╠═07030004-0000-4000-8000-000000000025
# ╟─07030004-0000-4000-8000-000000000030
# ╟─07030000-0000-4000-8000-0000000000f1
# ╟─07030006-0000-4000-8000-000000000001
# ╠═07030006-0000-4000-8000-000000000010
# ╠═07030006-0000-4000-8000-000000000020
# ╟─07030006-0000-4000-8000-000000000040
# ╟─07030007-0000-4000-8000-000000000001
# ╠═07030007-0000-4000-8000-000000000010
# ╟─07030007-0000-4000-8000-000000000030
# ╟─07030007-0000-4000-8000-000000000040
# ╠═07030007-0000-4000-8000-000000000050
# ╟─07030008-0000-4000-8000-000000000030
# ╟─07030009-0000-4000-8000-000000000001
# ╠═07030009-0000-4000-8000-000000000005
# ╠═07030009-0000-4000-8000-000000000008
# ╠═07030009-0000-4000-8000-000000000010
# ╠═07030009-0000-4000-8000-000000000020
# ╟─07030009-0000-4000-8000-000000000040
# ╟─0703000a-0000-4000-8000-000000000040
# ╟─0703000b-0000-4000-8000-000000000001
# ╟─0703000b-0000-4000-8000-000000000000
# ╟─0703000b-0000-4000-8000-00000000000a
# ╟─0703000b-0000-4000-8000-000000000002
# ╟─0703000b-0000-4000-8000-000000000003
# ╟─0703000d-0000-4000-8000-000000000001
# ╠═0703000d-0000-4000-8000-000000000004
# ╠═0703000d-0000-4000-8000-000000000005
# ╠═0703000d-0000-4000-8000-000000000010
# ╠═0703000d-0000-4000-8000-000000000020
# ╟─0703000d-0000-4000-8000-000000000030
# ╠═0703000b-0000-4000-8000-000000000010
# ╠═0703000b-0000-4000-8000-000000000011
# ╠═0703000b-0000-4000-8000-000000000012
# ╠═0703000b-0000-4000-8000-000000000020
# ╟─0703000b-0000-4000-8000-000000000030
# ╟─0703000b-0000-4000-8000-000000000031
# ╟─0703000c-0000-4000-8000-000000000001
