### A Pluto.jl notebook ###
# v0.2.6

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
no exaggerated mAs.  The only place we deliberately economize is **z**:
the saved reconstruction is three slices and the phantom is only deep
enough to cover every acquired ray.  BasisSimulator adds symmetric
detector guard rows for the full reconstruction cylinder, so this small
z grid does not discard terminal cone-beam support.

| Stage         | Matrix                   | Voxel (mm)             | Extent                        |
|---------------|--------------------------|------------------------|-------------------------------|
| GT phantom    | **1600 × 1100 × 20**     | **0.2 isotropic**      | 320 × 220 × 4 mm              |
| Recon         | 512 × 512 × **3**        | **0.625 isotropic**    | FOV 32 cm × 1.875 mm z        |
| Collimation   | **2.5 mm nominal**       | —                      | cone-guard rows added automatically |
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
                  ├─→ generalized Cong → per-basis FBP → ACNR → VMI →  │
Simulate 140 kVp→┘                                                     │
                                                                       │
                     Per-Rod Measured-vs-Theoretical Regression  ──────┘
                                       at 50 / 70 / 100 / 140 keV
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
# Use CairoMakie for faithful build-time rendering. Snapshot can still isolate
# and compile independent browser-safe islands without hoisting this import.
import CairoMakie as Mke

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
the minimum used by this deliberately short reference scan.
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
const QRM_TARGET_NZ = 20;    # 20 × 0.2 mm = 4 mm — short z-invariant reference phantom

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

    # Makie heatmaps put the y-axis going UP (math convention).
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

    fig = Mke.Figure(size = (1100, 760))
    ax = Mke.Axis(
        fig[1, 1];
        title = "QRM-Thorax mid-slice (labeled)",
        subtitle = "$(size(slice, 1)) × $(size(slice, 2)) at $(QRM_VOXEL_SIZE_CM[1] * 10) mm/voxel · $(n_lbl) unique labels",
        aspect = Mke.DataAspect(),
        titlesize = 28, subtitlesize = 18,
    )
    hm = Mke.heatmap!(ax, Float32.(slice); colormap = :tab20)
    Mke.hidedecorations!(ax)
    Mke.Colorbar(fig[1, 2], hm; label = "label", width = 14, labelsize = 18)
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

**Nominal collimation = 2.5 mm at iso** (clinical thin-slice DE setting).
The saved grid is 3 recon slices @ 0.625 mm.  The workspace automatically
adds the symmetric detector rows required to cover those slices across
the full XY FOV, while keeping the forward projection compact enough for
the truly 0.2 mm isotropic ground-truth phantom.
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
# linear transform.  Automatic detector guard rows cover this complete
# reconstruction cylinder at the edge of the XY FOV.
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
    @info "Simulating: 80 kVp / $(round(protocol_low.mA, digits = 1)) mA-eff…"
    ws = BS.create_eict_workspace(
        scanner, protocol_low, sim_opts, recon_opts, phantom,
    )
    try
        BS.simulate!(ws, phantom, protocol_low, sim_opts)
        I0_scalar = BS.compute_detector_I0(
            ws.geom, protocol_low, sum(ws.weights),
        ) * Float64(ws.η_eff)
        air_ref = ws.bowtie_air_reference === nothing ?
            ones(Float32, ws.geom.n_cols, ws.geom.n_rows) :
            Float32.(Array(ws.bowtie_air_reference))
        energies, response = BS.resolve_source_spectrum_full(
            sim_opts, protocol_low; scanner = scanner, geom = ws.geom,
        )
        (
            sino = Float32.(Array(ws.sinogram)),
            geom = ws.geom,
            I0_ray = Float32.(I0_scalar .* Float64.(air_ref)),
            energies = Float32.(energies),
            response = Float32.(response),
        )
    finally
        BS.release_backend!(ws)
    end
end

# ╔═╡ 07030004-0000-4000-8000-000000000020
sim_high = let
    @info "Simulating: 140 kVp / $(round(protocol_high.mA, digits = 1)) mA-eff…"
    ws = BS.create_eict_workspace(
        scanner, protocol_high, sim_opts, recon_opts, phantom,
    )
    try
        BS.simulate!(ws, phantom, protocol_high, sim_opts)
        I0_scalar = BS.compute_detector_I0(
            ws.geom, protocol_high, sum(ws.weights),
        ) * Float64(ws.η_eff)
        air_ref = ws.bowtie_air_reference === nothing ?
            ones(Float32, ws.geom.n_cols, ws.geom.n_rows) :
            Float32.(Array(ws.bowtie_air_reference))
        energies, response = BS.resolve_source_spectrum_full(
            sim_opts, protocol_high; scanner = scanner, geom = ws.geom,
        )
        (
            sino = Float32.(Array(ws.sinogram)),
            geom = ws.geom,
            I0_ray = Float32.(I0_scalar .* Float64.(air_ref)),
            energies = Float32.(energies),
            response = Float32.(response),
        )
    finally
        BS.release_backend!(ws)
    end
end

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

    fig = Mke.Figure(size = (1180, 580))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "View", ylabel = "Detector Column",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )

    for (c, ttl, slice) in ((1, "80 kVp", slice_lo), (2, "140 kVp", slice_hi))
        ax = Mke.Axis(fig[1, c]; title = ttl, axis_kwargs...)
        Mke.heatmap!(ax, slice; colormap = :viridis, colorrange = sino_window)
    end
    Mke.Colorbar(
        fig[1, 3]; colormap = :viridis, colorrange = sino_window,
        label = "Log Line Integral", width = 16, labelsize = 22, ticklabelsize = 18
    )
    fig
end

# ╔═╡ 07030000-0000-4000-8000-0000000000f1
md"""
## VMI Pipeline

Two-channel profiled generalized Cong → validated per-basis FBP →
Kalender ACNR → analytical VMI synthesis. This is the canonical notebook 03
framework applied to the QRM thorax phantom.
"""

# ╔═╡ 07030006-0000-4000-8000-000000000001
md"""
### 01. Projection-Domain Material Decomposition

Both corrected 80/140-kVp channels enter the fully inlined generalized
profiled Cong likelihood at ``K=2``. No effective-energy substitution or
source-level Cong helper is used.

The basis builder constructs the **per-ray effective spectrum** that
`simulate!` actually applied to the rays — tube × flat filter × bowtie ×
heel × η(E) — and normalizes per ray so `Σ_E ŵ[col, row, E] = 1`.  This
matches the spectrum baked into the EICT workspace's `bowtie_air_reference`,
so Cong's inversion solves the same polychromatic transmission integral
the forward model used.
"""

# ╔═╡ 07030006-0000-4000-8000-000000000010
begin
    # Recomputed from the live low-/high-kVp simulations.
    nchannel_controls = (
        iodine_bounds = (-0.10f0, 0.40f0),
        water_bounds = (-2.0f0, 50.0f0),
        outer_iterations = 16,
        inner_iterations = 12,
        max_iodine_step = 0.05f0,
        max_water_step = 5.0f0,
        parameter_tolerance = 5.0f-5,
        fisher_condition_limit = 1.0f8,
        air_gate = 0.0f0,
        tile_views = 8,
    )

    nchannel_slab_counts = let
        channels = (sim_low, sim_high)
        available_rows = size(sim_low.sino, 2)
        selected_rows = 1:available_rows
        row_positions = (
            collect(selected_rows) .- (available_rows + 1) / 2
        ) .* sim_low.geom.pixel_row_size
        cone_scales = sqrt.(1 .+ (row_positions ./ sim_low.geom.SAD).^2)
        channel_data = map(channels) do channel
            I0 = Float64.(channel.I0_ray[:, selected_rows])
            h = Float32.(channel.sino[:, selected_rows, :])
            response = Float64.(channel.response[:, selected_rows, :])
            response ./= max.(sum(response; dims = 3), eps(Float64))
            Φ = response .* reshape(I0, size(I0, 1), size(I0, 2), 1)
            (
                bin = h,
                I0 = Float32.(I0),
                energies = Float32.(channel.energies),
                Φ = Float32.(Φ),
            )
        end
        (
            bins = getproperty.(channel_data, :bin),
            I0 = getproperty.(channel_data, :I0),
            energies = getproperty.(channel_data, :energies),
            Φ = getproperty.(channel_data, :Φ),
            nrows = available_rows,
            selected_rows = selected_rows,
            available_rows = available_rows,
            cone_scales = cone_scales,
            max_cone_relerr = maximum(abs.(cone_scales .- 1)),
        )
    end

    nchannel_basis = let
        E = sort!(unique(vcat(nchannel_slab_counts.energies...)))
        ncol, nrow = size(first(nchannel_slab_counts.Φ))[1:2]
        K = length(nchannel_slab_counts.Φ)
        Φ = zeros(Float32, ncol, nrow, length(E), K)
        lookup = Dict(e => i for (i, e) in enumerate(E))
        for k in 1:K, (source_index, e) in
            enumerate(nchannel_slab_counts.energies[k])
            Φ[:, :, lookup[e], k] .=
                nchannel_slab_counts.Φ[k][:, :, source_index]
        end
        μρ_I = Float32[
            BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, Float64(e))
            for e in E
        ]
        μρ_W = Float32[
            BS.compute_mass_μ_at_energy(BS.XA.Materials.water, Float64(e))
            for e in E
        ]
        I0 = cat(nchannel_slab_counts.I0...; dims = 3)
        I0_from_Φ = dropdims(sum(Float64.(Φ); dims = 3); dims = 3)
        I0_relerr = maximum(
            abs.(I0_from_Φ .- Float64.(I0)) ./
            max.(Float64.(I0), eps(Float64)),
        )
        I0_relerr < 5e-5 || error(
            "Applied response and I0 disagree (max relative error = $(I0_relerr)).",
        )
        Φsum = max.(I0, eps(Float32))
        μI_eff = dropdims(sum(
            Φ .* reshape(μρ_I, 1, 1, length(E), 1); dims = 3,
        ); dims = 3) ./ Φsum
        μW_eff = dropdims(sum(
            Φ .* reshape(μρ_W, 1, 1, length(E), 1); dims = 3,
        ); dims = 3) ./ Φsum
        normal_II = dropdims(sum(abs2, μI_eff; dims = 3); dims = 3)
        normal_IW = dropdims(sum(μI_eff .* μW_eff; dims = 3); dims = 3)
        normal_WW = dropdims(sum(abs2, μW_eff; dims = 3); dims = 3)
        (
            E = E, Φ = Φ, μρ_I = μρ_I, μρ_W = μρ_W,
            I0 = Float32.(I0), μI_eff = μI_eff, μW_eff = μW_eff,
            normal_II = normal_II, normal_IW = normal_IW,
            normal_WW = normal_WW, I0_relerr = I0_relerr,
        )
    end
end

# ╔═╡ 07030006-0000-4000-8000-000000000020
begin
begin
    # Fully inlined generalized Cong implementation and K-channel runner.
    """
        nchannel_forward(A, C, Φ, μI, μW; second=false)

    Exact discrete polychromatic K-channel mean and analytic derivatives.
    `Φ[e,k]` is the absolute air-count contribution, `A` is iodine area
    density, and `C` is water area density, both in g/cm².
    """
    function nchannel_forward(A,C,Φ,μI,μW;second=false)
        nE,K = size(Φ)
        nE == length(μI) == length(μW) ||
            throw(DimensionMismatch("Energy dimensions disagree"))
        λ=zeros(Float64,K)
        dA=zeros(Float64,K)
        dC=zeros(Float64,K)
        dAA=second ? zeros(Float64,K) : nothing
        dAC=second ? zeros(Float64,K) : nothing
        dCC=second ? zeros(Float64,K) : nothing
        @inbounds for k in 1:K, e in 1:nE
            z=Float64(Φ[e,k])*exp(-Float64(μI[e])*A-Float64(μW[e])*C)
            mi,mw=Float64(μI[e]),Float64(μW[e])
            λ[k]+=z; dA[k]-=mi*z; dC[k]-=mw*z
            if second
                dAA[k]+=mi*mi*z; dAC[k]+=mi*mw*z; dCC[k]+=mw*mw*z
            end
        end
        second ? (;λ,dA,dC,dAA,dAC,dCC) : (;λ,dA,dC)
    end

    function nchannel_golden_minimize(f,lo,hi;iterations=80)
        lo == hi && return (x=lo,value=f(lo))
        ϕ=(sqrt(5.0)-1)/2
        x1=hi-ϕ*(hi-lo); x2=lo+ϕ*(hi-lo)
        f1,f2=f(x1),f(x2)
        for _ in 1:iterations
            if f1 ≤ f2
                hi,x2,f2=x2,x1,f1
                x1=hi-ϕ*(hi-lo); f1=f(x1)
            else
                lo,x1,f1=x1,x2,f2
                x2=lo+ϕ*(hi-lo); f2=f(x2)
            end
        end
        candidates=((x1,f1),(x2,f2),(lo,f(lo)),(hi,f(hi)))
        x,value=argmin(last,candidates)
        (;x,value)
    end

    function nchannel_scalar_global(f,bounds;grid_points=129,iterations=80)
        grid=collect(range(bounds...;length=grid_points))
        values=f.(grid)
        basins=Tuple{Float64,Float64,Int}[]
        for i in 2:(length(grid)-1)
            isfinite(values[i]) &&
                values[i] ≤ values[i-1] && values[i] ≤ values[i+1] &&
                push!(basins,(grid[i-1],grid[i+1],i))
        end
        candidates=NamedTuple[
            (x=grid[1],value=values[1],basin=0),
            (x=grid[end],value=values[end],basin=length(grid)),
        ]
        for (lo,hi,i) in basins
            r=nchannel_golden_minimize(f,lo,hi;iterations)
            push!(candidates,(x=r.x,value=r.value,basin=i))
        end
        if isempty(basins)
            i=argmin(values)
            push!(candidates,(x=grid[i],value=values[i],basin=i))
        end
        best=candidates[argmin(getproperty.(candidates,:value))]
        (;best,candidates,grid,values,basin_count=length(basins))
    end

    function nchannel_solve_total_C(
        A,y_total,Φ,μI,μW,water_bounds;
        bisection_iterations=80,
    )
        lo,hi=Float64.(water_bounds)
        residual(C)=sum(nchannel_forward(A,C,Φ,μI,μW).λ)-y_total
        rlo,rhi=residual(lo),residual(hi)
        if !(isfinite(rlo)&&isfinite(rhi)&&rlo≥0&&rhi≤0)
            return (success=false,C=NaN,residual=NaN,bracket=(rlo,rhi))
        end
        for _ in 1:bisection_iterations
            mid=(lo+hi)/2
            residual(mid)>0 ? (lo=mid) : (hi=mid)
        end
        C=(lo+hi)/2
        (success=true,C,residual=residual(C),bracket=(rlo,rhi))
    end

    """
    Exact monotone aggregate-channel Cong-like reference. The inner root is
    guaranteed when bracketed; the complete outer iodine interval is scanned,
    every detected basin is refined, and both endpoints are evaluated.
    """
    function nchannel_cong_constrained_reference(
        y,Φ,μI,μW;
        iodine_bounds=(-0.10,0.40),water_bounds=(-2.0,50.0),
        grid_points=129,bisection_iterations=80,golden_iterations=80,
    )
        K=length(y)
        size(Φ,2)==K || throw(DimensionMismatch("Channel count mismatch"))
        yv=Float64.(y); y_total=sum(yv)
        roots=Dict{Float64,NamedTuple}()
        root(A)=get!(roots,Float64(A)) do
            nchannel_solve_total_C(
                A,y_total,Φ,μI,μW,water_bounds;
                bisection_iterations,
            )
        end
        function objective(A)
            r=root(A)
            r.success || return Inf
            λ=max.(nchannel_forward(A,r.C,Φ,μI,μW).λ,eps(Float64))
            π=λ/sum(λ)
            -sum(yv.*log.(π))
        end
        search=nchannel_scalar_global(
            objective,iodine_bounds;
            grid_points,iterations=golden_iterations,
        )
        A=search.best.x; r=root(A)
        (
            iodine=A,water=r.C,objective=search.best.value,
            root_bracketed=r.success,total_residual=r.residual,
            selected_basin=search.best.basin,
            basin_count=search.basin_count,
            boundary_contact=(
                A≈first(iodine_bounds) || A≈last(iodine_bounds) ||
                r.C≈first(water_bounds) || r.C≈last(water_bounds)
            ),
        )
    end

    nchannel_poisson_quasi_nll(A,C,y,Φ,μI,μW)=let
        λ=max.(nchannel_forward(A,C,Φ,μI,μW).λ,eps(Float64))
        sum(λ-Float64.(y).*log.(λ))
    end

    """
    Slow bounded all-channel profile quasi-likelihood reference. No
    allocation assumptions or local optimizer starting point enter the
    correctness claim: every sampled inner and outer basin plus endpoints is
    evaluated.
    """
    function nchannel_profile_reference(
        y,Φ,μI,μW;
        iodine_bounds=(-0.10,0.40),water_bounds=(-2.0,50.0),
        outer_grid_points=65,inner_grid_points=65,iterations=80,
    )
        inner_cache=Dict{Float64,NamedTuple}()
        function inner(A)
            get!(inner_cache,Float64(A)) do
                f(C)=nchannel_poisson_quasi_nll(A,C,y,Φ,μI,μW)
                s=nchannel_scalar_global(
                    f,water_bounds;grid_points=inner_grid_points,iterations,
                )
                (C=s.best.x,value=s.best.value,basin=s.best.basin,
                 basin_count=s.basin_count)
            end
        end
        outer(A)=inner(A).value
        s=nchannel_scalar_global(
            outer,iodine_bounds;grid_points=outer_grid_points,iterations,
        )
        A=s.best.x; inn=inner(A)
        f=nchannel_forward(A,inn.C,Φ,μI,μW)
        λ=max.(f.λ,eps(Float64)); yv=Float64.(y)
        gA=sum((1 .- yv./λ).*f.dA)
        gC=sum((1 .- yv./λ).*f.dC)
        FAA=sum(f.dA.^2 ./ λ)
        FCC=sum(f.dC.^2 ./ λ)
        (
            iodine=A,water=inn.C,objective=inn.value,
            score_norm=hypot(gA,gC)/sqrt(max(FAA+FCC,eps(Float64))),
            selected_outer_basin=s.best.basin,
            outer_basin_count=s.basin_count,
            selected_inner_basin=inn.basin,
            inner_basin_count=inn.basin_count,
            boundary_contact=(
                A≈first(iodine_bounds)||A≈last(iodine_bounds)||
                inn.C≈first(water_bounds)||inn.C≈last(water_bounds)
            ),
        )
    end

    function nchannel_profile_tile!(
    sino_I, sino_W, fisher_AA, fisher_AC, fisher_CC,
    quality_flag, score_norm, outer_count, inner_count,
    hs::NTuple{K},
    Φ, μρ_I, μρ_W, I0, μI_eff, μW_eff,
    normal_II, normal_IW, normal_WW, controls,
) where {K}
    # Ray-dependent dual-kVp responses use detector-column initializer terms.
    nE = length(μρ_I)
    A_lo, A_hi = controls.iodine_bounds
    C_lo, C_hi = controls.water_bounds
    n_outer, n_inner = controls.outer_iterations, controls.inner_iterations
    A_step, C_step = controls.max_iodine_step, controls.max_water_step
    parameter_tolerance = controls.parameter_tolerance
    fisher_condition_limit = controls.fisher_condition_limit
    air_gate = controls.air_gate

    BS.AK.foreachindex(sino_I) do idx
        ncol=size(sino_I,1)
        nrow=size(sino_I,2)
        col=mod1(idx,ncol)
        row=mod1(cld(idx,ncol),nrow)
        max_abs_h = 0f0
        for k in 1:K
            max_abs_h = max(max_abs_h,abs(hs[k][idx]))
        end
        if max_abs_h < air_gate
            sino_I[idx] = 0f0
            sino_W[idx] = 0f0
            fisher_AA[idx] = 0f0
            fisher_AC[idx] = 0f0
            fisher_CC[idx] = 0f0
            quality_flag[idx] = UInt8(0)
            score_norm[idx] = 0f0
            outer_count[idx] = UInt8(0)
            inner_count[idx] = UInt8(0)
            return
        end

        # K-channel linear initializer; all iterations below are polychromatic.
        rhs_I, rhs_W = 0f0, 0f0
        for k in 1:K
            rhs_I += μI_eff[col,row,k]*hs[k][idx]
            rhs_W += μW_eff[col,row,k]*hs[k][idx]
        end
        nII=normal_II[col,row]
        nIW=normal_IW[col,row]
        nWW=normal_WW[col,row]
        det0_raw = nII*nWW - nIW*nIW
        initializer_valid = isfinite(det0_raw) && det0_raw > 1f-12
        det0 = initializer_valid ? det0_raw : 1f0
        A = initializer_valid ?
            clamp((nWW*rhs_I-nIW*rhs_W)/det0,A_lo,A_hi) :
            clamp(0f0,A_lo,A_hi)
        C = initializer_valid ?
            clamp((nII*rhs_W-nIW*rhs_I)/det0,C_lo,C_hi) :
            clamp(20f0,C_lo,C_hi)

        # Guaranteed monotone aggregate equation, used here only to stabilize
        # the fast solver's initial water value at its current iodine value.
        y_total=0f0
        for k in 1:K
            y_total += max(I0[col,row,k]*exp(-hs[k][idx]),1f-6)
        end
        croot_lo,croot_hi=C_lo,C_hi
        total_lo,total_hi=0f0,0f0
        for k in 1:K, e in 1:nE
            total_lo += Φ[col,row,e,k]*exp(-μρ_I[e]*A-μρ_W[e]*croot_lo)
            total_hi += Φ[col,row,e,k]*exp(-μρ_I[e]*A-μρ_W[e]*croot_hi)
        end
        aggregate_bracketed=total_lo≥y_total && total_hi≤y_total
        attainable_max,attainable_min=0f0,0f0
        for k in 1:K, e in 1:nE
            attainable_max += Φ[col,row,e,k]*exp(
                -μρ_I[e]*A_lo-μρ_W[e]*C_lo,
            )
            attainable_min += Φ[col,row,e,k]*exp(
                -μρ_I[e]*A_hi-μρ_W[e]*C_hi,
            )
        end
        aggregate_feasible =
            attainable_max≥y_total && attainable_min≤y_total
        if aggregate_bracketed
            for _ in 1:28
                mid=(croot_lo+croot_hi)/2f0
                total_mid=0f0
                for k in 1:K, e in 1:nE
                    total_mid += Φ[col,row,e,k]*exp(-μρ_I[e]*A-μρ_W[e]*mid)
                end
                if total_mid>y_total
                    croot_lo=mid
                else
                    croot_hi=mid
                end
            end
            C=(croot_lo+croot_hi)/2f0
        end

        converged = false
        used_outer=0
        used_inner=0
        for outer_iter in 1:n_outer
            used_outer=outer_iter
            # Inner scalar solve: C*(A) = argmin_C L(A,C).
            for _ in 1:n_inner
                used_inner+=1
                gC, FCC = 0f0, 0f0
                for k in 1:K
                    λ, dC = 0f0, 0f0
                    @inbounds for e in 1:nE
                        z = Φ[col,row,e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                        λ += z
                        dC -= μρ_W[e] * z
                    end
                    λ = max(λ, 1f-6)
                    # Corrected counts may be fractional after detector correction.
                    y = max(I0[col,row,k]*exp(-hs[k][idx]),1f-6)
                    gC += (1f0 - y/λ) * dC
                    FCC += dC*dC / λ
                end
                raw_C_step = gC/max(FCC,1f-12)
                C_new = clamp(
                    C-clamp(raw_C_step,-C_step,C_step),C_lo,C_hi,
                )
                C_done = abs(C_new-C) <= parameter_tolerance*(1f0+abs(C))
                C = C_new
                C_done && break
            end

            # Envelope gradient and Fisher Schur-complement profile curvature.
            gA, FAA, FAC, FCC = 0f0, 0f0, 0f0, 0f0
            for k in 1:K
                λ, dA, dC = 0f0, 0f0, 0f0
                @inbounds for e in 1:nE
                    z = Φ[col,row,e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                    λ += z
                    dA -= μρ_I[e] * z
                    dC -= μρ_W[e] * z
                end
                λ = max(λ, 1f-6)
                y = max(I0[col,row,k]*exp(-hs[k][idx]),1f-6)
                gA += (1f0 - y/λ) * dA
                FAA += dA*dA / λ
                FAC += dA*dC / λ
                FCC += dC*dC / λ
            end
            Hprof = max(FAA - FAC*FAC/max(FCC, 1f-12), 1f-12)
            raw_A_step = gA/Hprof
            A_new = clamp(
                A-clamp(raw_A_step,-A_step,A_step),A_lo,A_hi,
            )
            converged = abs(A_new-A) <= parameter_tolerance*(1f0+abs(A))
            A = A_new
            converged && break
        end

        # Re-profile water at the final iodine iterate.
        c_converged = false
        for _ in 1:n_inner
            used_inner+=1
            gC, FCC = 0f0, 0f0
            for k in 1:K
                λ, dC = 0f0, 0f0
                @inbounds for e in 1:nE
                    z = Φ[col,row,e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                    λ += z
                    dC -= μρ_W[e] * z
                end
                λ = max(λ, 1f-6)
                y = max(I0[col,row,k]*exp(-hs[k][idx]),1f-6)
                gC += (1f0 - y/λ) * dC
                FCC += dC*dC / λ
            end
            C_new = clamp(
                C-clamp(gC/max(FCC,1f-12),-C_step,C_step),C_lo,C_hi,
            )
            C_done = abs(C_new-C) <= parameter_tolerance*(1f0+abs(C))
            C = C_new
            if C_done
                c_converged = true
                break
            end
        end
        converged &= c_converged

        # Final score and Fisher conditioning are recorded; they are not silently
        # converted into image regularization.
        gA, gC, FAA, FAC, FCC = 0f0, 0f0, 0f0, 0f0, 0f0
        for k in 1:K
            λ, dA, dC = 0f0, 0f0, 0f0
            @inbounds for e in 1:nE
                z = Φ[col,row,e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                λ += z
                dA -= μρ_I[e]*z
                dC -= μρ_W[e]*z
            end
            λ = max(λ,1f-6)
            y = max(I0[col,row,k]*exp(-hs[k][idx]),1f-6)
            gA += (1f0-y/λ)*dA
            gC += (1f0-y/λ)*dC
            FAA += dA*dA/λ
            FAC += dA*dC/λ
            FCC += dC*dC/λ
        end
        score_norm[idx] = sqrt(gA*gA+gC*gC) /
            sqrt(max(FAA+FCC,1f-12))
        fisher_det = max(FAA*FCC-FAC*FAC,0f0)
        fisher_trace = FAA+FCC
        fisher_disc = sqrt(max(fisher_trace*fisher_trace-4f0*fisher_det,0f0))
        eig_max_raw = max((fisher_trace+fisher_disc)/2f0,1f-12)
        eig_min = max(fisher_det/eig_max_raw,1f-12)
        eig_max = max(eig_max_raw,eig_min)
        ill_conditioned = eig_max/eig_min > fisher_condition_limit

        tol = 2f-4
        hit_A = A <= A_lo + tol || A >= A_hi - tol
        hit_C = C <= C_lo + tol || C >= C_hi - tol
        invalid_model = !(
            isfinite(A)&&isfinite(C)&&isfinite(score_norm[idx])&&
            isfinite(FAA)&&isfinite(FAC)&&isfinite(FCC)
        )
        quality_flag[idx] =
            UInt8(hit_A ? 1 : 0) |
            UInt8(hit_C ? 2 : 0) |
            UInt8(converged ? 0 : 4) |
            UInt8(ill_conditioned || !initializer_valid ? 8 : 0) |
            UInt8(aggregate_feasible ? 0 : 16) |
            UInt8(invalid_model ? 32 : 0)
        outer_count[idx]=UInt8(min(used_outer,255))
        inner_count[idx]=UInt8(min(used_inner,255))
        fisher_AA[idx],fisher_AC[idx],fisher_CC[idx] = FAA,FAC,FCC
        sino_I[idx], sino_W[idx] = A, C
    end
    nothing
    end
end



begin
function run_nchannel_profile(slab_counts,basis,geom)
    shape = size(slab_counts.bins[1])
    sino_I = Array{Float32}(undef,shape)
    sino_W = Array{Float32}(undef,shape)
    flags = Array{UInt8}(undef,shape)
    score_norm = Array{Float32}(undef,shape)
    fisher_AA = Array{Float32}(undef,shape)
    fisher_AC = Array{Float32}(undef,shape)
    fisher_CC = Array{Float32}(undef,shape)
    outer_iterations = Array{UInt8}(undef,shape)
    inner_iterations = Array{UInt8}(undef,shape)
    Φ_gpu = to_gpu(basis.Φ)
    μρ_I_gpu = to_gpu(basis.μρ_I)
    μρ_W_gpu = to_gpu(basis.μρ_W)
    I0_gpu = to_gpu(basis.I0)
    μI_eff_gpu = to_gpu(basis.μI_eff)
    μW_eff_gpu = to_gpu(basis.μW_eff)
    normal_II_gpu = to_gpu(basis.normal_II)
    normal_IW_gpu = to_gpu(basis.normal_IW)
    normal_WW_gpu = to_gpu(basis.normal_WW)
    elapsed = @elapsed for vrange in BS.tile_ranges(
        shape[3],nchannel_controls.tile_views,
    )
        hs = [
            to_gpu(Float32.(slab_counts.bins[k][:,:,vrange]))
            for k in eachindex(slab_counts.bins)
        ]
        I_gpu,W_gpu = similar(hs[1]),similar(hs[1])
        flag_gpu = similar(hs[1],UInt8)
        score_gpu = similar(hs[1],Float32)
        fisher_AA_gpu = similar(hs[1],Float32)
        fisher_AC_gpu = similar(hs[1],Float32)
        fisher_CC_gpu = similar(hs[1],Float32)
        outer_gpu = similar(hs[1],UInt8)
        inner_gpu = similar(hs[1],UInt8)
        nchannel_profile_tile!(
            I_gpu,W_gpu,fisher_AA_gpu,fisher_AC_gpu,fisher_CC_gpu,
            flag_gpu,score_gpu,outer_gpu,inner_gpu,Tuple(hs),
            Φ_gpu,μρ_I_gpu,μρ_W_gpu,I0_gpu,μI_eff_gpu,μW_eff_gpu,
            normal_II_gpu,normal_IW_gpu,normal_WW_gpu,nchannel_controls,
        )
        sino_I[:,:,vrange] .= Array(I_gpu)
        sino_W[:,:,vrange] .= Array(W_gpu)
        flags[:,:,vrange] .= Array(flag_gpu)
        score_norm[:,:,vrange] .= Array(score_gpu)
        fisher_AA[:,:,vrange] .= Array(fisher_AA_gpu)
        fisher_AC[:,:,vrange] .= Array(fisher_AC_gpu)
        fisher_CC[:,:,vrange] .= Array(fisher_CC_gpu)
        outer_iterations[:,:,vrange] .= Array(outer_gpu)
        inner_iterations[:,:,vrange] .= Array(inner_gpu)
    end
    (
        sino_iodine=sino_I,sino_water=sino_W,quality_flag=flags,
        fisher=(AA=fisher_AA,AC=fisher_AC,CC=fisher_CC),
        score_norm,outer_iterations,inner_iterations,
        geom,elapsed_s=elapsed,
    )
end

# Clean unregularized K=2 Cong on every native detector row.
sino_basis=run_nchannel_profile(
    nchannel_slab_counts,nchannel_basis,sim_low.geom,
)
end
end

# ╔═╡ 07030006-0000-4000-8000-000000000040
let
    n_row = size(sino_basis.sino_iodine, 2)
    mid_r = n_row ÷ 2 + 1

    fig = Mke.Figure(size = (1400, 580))
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
        ax = Mke.Axis(fig[r, panel_c]; title = ttl, axis_kwargs...)
        Mke.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        Mke.Colorbar(
            fig[r, cbar_c]; colormap = :viridis, colorrange = range,
            label = cbar_label, width = 16, labelsize = 22, ticklabelsize = 18
        )
    end
    fig
end

# ╔═╡ 07030007-0000-4000-8000-000000000001
md"""
### 02. FBP: Iodine and Water Basis Maps

The validated dual-kVp per-basis kernels from notebook 03 are used: the
original soft iodine apodization and halfway Standard/Soft water apodization.
Output volumes are in basis-density units (g/cm³).
"""

# ╔═╡ 07030007-0000-4000-8000-000000000010
basis_volumes = let
    # Reconstruct both generalized-Cong basis sinograms with the validated kernels.
    matrix_size = recon_opts.matrix_size
    geom = sino_basis.geom
    iodine_filter = BS.CustomFilter(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        (1.0, 0.40, 0.12, 0.03, 0.001),
    )
    water_filter = BS.CustomFilter(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        (1.0, 0.8744, 0.6003, 0.3031, 0.0266),
    )
    function _fbp(sino_cpu, filter)
        sino_gpu = to_gpu(Float32.(sino_cpu))
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, geom, matrix_size; filter = filter,
        )
        try
            Float32.(Array(BS.reconstruct!(ws, sino_gpu, geom)))
        finally
            BS.release_backend!(ws)
        end
    end
    (
        vol_iodine_raw = _fbp(sino_basis.sino_iodine, iodine_filter),
        vol_water_raw = _fbp(sino_basis.sino_water, water_filter),
        geom = geom,
        kernels = (
            water = :StandardSoftBlend,
            iodine = :OriginalDualKvpSoft,
        ),
    )
end

# ╔═╡ 07030007-0000-4000-8000-000000000030
let
    fig = Mke.Figure(size = (1180, 580))
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
        ax = Mke.Axis(
            fig[r, panel_c]; title = ttl,
            aspect = Mke.DataAspect(), axis_kwargs...
        )
        Mke.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        Mke.hidedecorations!(ax)
        Mke.Colorbar(
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
channels uses the canonical five-pass, `beta_max=14` ACNR setting.  Dual-energy basis noise is
ANTI-correlated while structure is positively correlated, so the
anti-correlated (noise) component is subtracted exactly and structure
pixels stay bit-untouched — zero resolution loss by construction.
Identical to nb03.
"""

# ╔═╡ 07030007-0000-4000-8000-000000000050
basis_acnr = let
    W = copy(basis_volumes.vol_water_raw)
    I = copy(basis_volumes.vol_iodine_raw)
    info = BS.apply_acnr_kalender!(
        W, I; hp_sigma_px = 1.5, window = 4, passes = 5, beta_max = 14.0,
    )
    @info "[ACNR · Kalender-1988 true ACNR] ρ_hp(W,I)=$(round(info.ρ_hp, digits = 3))"
    (
        vol_iodine_raw = I, vol_water_raw = W, geom = basis_volumes.geom,
        acnr = (passes = 5, beta_max = 14.0, hp_sigma_px = 1.5, window = 4),
    )
end

# ╔═╡ 07030008-0000-4000-8000-000000000030
let
    fig = Mke.Figure(size = (1180, 580))
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
        ax = Mke.Axis(
            fig[r, panel_c]; title = ttl,
            aspect = Mke.DataAspect(), axis_kwargs...
        )
        Mke.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        Mke.hidedecorations!(ax)
        Mke.Colorbar(
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

VMI grid: 50, 70, 100, 140 keV.  The `solid_water_basis` cell measures
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
de_vmi_energies = Float64[50, 70, 100, 140];

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

    fig = Mke.Figure(size = (1180, 1180))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    sample = vmi_HU_final[50.0]
    mid = size(sample, 3) ÷ 2

    for (k, E) in enumerate(de_vmi_energies)
        r = ((k - 1) ÷ 2) + 1
        c = ((k - 1) % 2) + 1
        ax = Mke.Axis(
            fig[r, c]; title = "$(Int(E)) keV VMI",
            aspect = Mke.DataAspect(), axis_kwargs...,
        )
        Mke.heatmap!(
            ax, vmi_HU_final[E][:, :, mid];
            colormap = :grays, colorrange = HU_window
        )
        Mke.hidedecorations!(ax)
    end
    Mke.Colorbar(
        fig[1:2, 3]; colormap = :grays, colorrange = HU_window,
        label = "HU", width = 16, labelsize = 22, ticklabelsize = 18
    )
    fig
end

# ╔═╡ 0703000a-0000-4000-8000-000000000040
let
    HU_window = (-200, 500)

    fig = Mke.Figure(size = (1180, 1180))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    sample = vmi_HU_final[50.0]
    mid = size(sample, 3) ÷ 2

    for (k, E) in enumerate(de_vmi_energies)
        r = ((k - 1) ÷ 2) + 1
        c = ((k - 1) % 2) + 1
        ax = Mke.Axis(
            fig[r, c]; title = "$(Int(E)) keV VMI",
            subtitle = "VMI",
            aspect = Mke.DataAspect(), axis_kwargs...,
        )
        Mke.heatmap!(
            ax, vmi_HU_final[E][:, :, mid];
            colormap = :grays, colorrange = HU_window
        )
        Mke.hidedecorations!(ax)
    end
    Mke.Colorbar(
        fig[1:2, 3]; colormap = :grays, colorrange = HU_window,
        label = "HU", width = 16, labelsize = 22, ticklabelsize = 18
    )
    fig
end

# ╔═╡ 0703000b-0000-4000-8000-000000000001
md"""
## Results

Per-rod measured vs theoretical HU at 50 / 70 / 100 / 140 keV.

- **Measured HU** = mean over an 8-px-radius circular ROI at each rod
  centroid, broadcast across all z slices.
- **Theoretical HU** = `1000 · (μ_r(E) − μ_water(E)) / μ_water(E)` from
  `BS.compute_μ_at_energy(material, E)` — pure physics, no fitting.
"""

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

    fig = Mke.Figure(size = (1400, 1320))
    hu_kwargs = (colormap = :grays, colorrange = (-200, 500))
    title_kwargs = (titlesize = 28, subtitlesize = 20)

    # Top-left: raw HU recon.
    ax_tl = Mke.Axis(
        fig[1, 1];
        title = "70 keV VMI HU recon",
        subtitle = "z = $(z_recon) of $(size(vmi_HU_final[70.0], 3))",
        aspect = Mke.DataAspect(),
        title_kwargs...,
    )
    Mke.heatmap!(ax_tl, hu_slice; hu_kwargs...)
    Mke.hidedecorations!(ax_tl)

    # Top-right: ALL labels — phantom_in_recon multi-label mask.
    ax_tr = Mke.Axis(
        fig[1, 2];
        title = "phantom_in_recon (`:nearest` resample)",
        subtitle = "$(length(unique(pir_slice))) unique labels on recon grid",
        aspect = Mke.DataAspect(),
        title_kwargs...,
    )
    Mke.heatmap!(ax_tr, Float32.(pir_slice); colormap = :tab20)
    Mke.hidedecorations!(ax_tr)

    # Bottom-left: HU recon + rod labels (9–12) overlaid.
    ax_bl = Mke.Axis(
        fig[2, 1];
        title = "HU + rod labels (9–12)",
        subtitle = "α = 0.6 over HU — rod edges should sit on recon rod edges",
        aspect = Mke.DataAspect(),
        title_kwargs...,
    )
    Mke.heatmap!(ax_bl, hu_slice; hu_kwargs...)
    Mke.heatmap!(
        ax_bl, rod_overlay;
        colormap = :tab20, alpha = 0.6, nan_color = (:white, 0.0),
        colorrange = (1, 12),
    )
    Mke.hidedecorations!(ax_bl)

    # Bottom-right: HU + full label mask (every non-air label).
    ax_br = Mke.Axis(
        fig[2, 2];
        title = "HU + full label mask (every non-air label)",
        subtitle = "α = 0.4 over HU — bone walls / heart cavity edge alignment",
        aspect = Mke.DataAspect(),
        title_kwargs...,
    )
    Mke.heatmap!(ax_br, hu_slice; hu_kwargs...)
    Mke.heatmap!(
        ax_br, full_overlay;
        colormap = :tab20, alpha = 0.4, nan_color = (:white, 0.0),
        colorrange = (1, 12),
    )
    Mke.hidedecorations!(ax_br)

    fig
end

# ╔═╡ 0703000b-0000-4000-8000-000000000002
md"""
### Water ROI

Water-rod core ROI (label 9 = `basis_water`) overlaid in red on the
70 keV VMI slice — the voxels feeding the `solid_water_basis`
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

    fig = Mke.Figure(size = (1180, 580))

    ax1 = Mke.Axis(
        fig[1, 1];
        title = "SW-ROI (water rod core)",
        subtitle = "Overlaid on 70 keV VMI",
        aspect = Mke.DataAspect(),
        titlesize = 32, subtitlesize = 24,
    )
    Mke.heatmap!(ax1, bg; colormap = :grays, colorrange = HU_window)
    Mke.heatmap!(
        ax1, overlay; colormap = :reds, alpha = 0.5,
        nan_color = (:white, 0.0)
    )
    Mke.hidedecorations!(ax1)

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
    bar_colors = [Mke.cgrad(:plasma, n_E; categorical = true)[i] for i in 1:n_E]

    ax2 = Mke.Axis(
        fig[1, 2];
        title = "Water ROI Mean HU",
        subtitle = "Per VMI Energy",
        xlabel = "VMI Energy (keV)", ylabel = "HU",
        xticks = (collect(1:n_E), ["$(Int(E))" for E in de_vmi_energies]),
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    Mke.barplot!(
        ax2, 1:n_E, sw_hu_per_keV;
        color = bar_colors, strokecolor = :black, strokewidth = 1
    )
    Mke.hlines!(ax2, [0.0]; color = :black, linewidth = 1, linestyle = :dash)

    for (k, h) in pairs(sw_hu_per_keV)
        Mke.text!(
            ax2, k, h;
            text = "$(round(h, digits = 1)) HU",
            align = (:center, h ≥ 0 ? :bottom : :top),
            fontsize = 16, offset = (0, h ≥ 0 ? 4 : -4)
        )
    end

    y_max = max(15.0, 1.2 * maximum(abs, sw_hu_per_keV))
    Mke.ylims!(ax2, -y_max, y_max)

    Mke.save(
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
exactly where the ROI lands on the 70 keV VMI slice.

Right panel = σ vs VMI energy.  Diagnoses how the textbook
(c_water, c_iodine) → HU(E) synth propagates noise.  Expectation for
80/140 kVp DECT: σ(50) > σ(70) and the higher-energy behavior is measured
directly from the final common basis pair. The natural noise-optimal
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

    fig = Mke.Figure(size = (1180, 580))

    ax1 = Mke.Axis(
        fig[1, 1];
        title = "Heart-Center Noise ROI",
        # subtitle = "Overlaid on 70 keV VMI · offset $(HEART_NOISE_ROI_OFFSET_PX) px from water rod",
        aspect = Mke.DataAspect(),
        titlesize = 32, subtitlesize = 24,
    )
    Mke.heatmap!(ax1, bg; colormap = :grays, colorrange = HU_window)
    Mke.heatmap!(
        ax1, overlay; colormap = :reds, alpha = 0.5,
        nan_color = (:white, 0.0),
    )
    Mke.hidedecorations!(ax1)

    Es = sort(collect(keys(vmi_noise_by_keV)))
    σs = [vmi_noise_by_keV[E].std  for E in Es]
    μs = [vmi_noise_by_keV[E].mean for E in Es]

    ax2 = Mke.Axis(
        fig[1, 2];
        title = "Heart-Center Noise vs Energy",
        # subtitle = "$(heart_noise_roi.n_voxels) vx × $(size(basis_acnr.vol_water_raw, 3)) z = $(heart_noise_roi.n_total) samples / point",
        xlabel = "VMI Energy (keV)",
        ylabel = "Noise σ (HU)",
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    Mke.scatterlines!(
        ax2, Es, σs;
        color = :tomato, markersize = 18, linewidth = 3,
    )
    for (E, σ, μ) in zip(Es, σs, μs)
        Mke.text!(
            ax2, E, σ;
            text = "σ=$(round(σ; digits = 1))\n⟨HU⟩=$(round(μ; digits = 1))",
            align = (:center, :bottom),
            fontsize = 16, offset = (0, 8),
        )
    end
    Mke.ylims!(ax2, 0, maximum(σs) * 1.4)

    Mke.save(
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
    fig = Mke.Figure(size = (980, 580))

    rod_colors = [
        Mke.RGBf(0.2, 0.6, 0.85),  # water    — blue
        Mke.RGBf(0.95, 0.65, 0.13),  # lipid    — orange
        Mke.RGBf(0.55, 0.3, 0.65),  # collagen — purple
        Mke.RGBf(0.85, 0.27, 0.1),  # iodine   — red
    ]

    ax = Mke.Axis(
        fig[1, 1];
        title = "Pure-Material Rods",
        subtitle = "50 / 70 / 100 / 140 keV",
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
        Mke.scatterlines!(
            ax, de_vmi_energies, vec(rod_data.measured[i, :]);
            color = c, linewidth = 2.5, markersize = 9
        )
        Mke.lines!(
            ax, de_vmi_energies, vec(rod_data.theoretical[i, :]);
            color = c, linewidth = 1.6, linestyle = :dash
        )
        rod_lines[i] = Mke.LineElement(color = c, linewidth = 2.5)

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

    style_meas = Mke.MarkerElement(
        color = :black, marker = :circle, markersize = 9,
        strokecolor = :black, strokewidth = 1
    )
    style_theo = Mke.LineElement(
        color = :black, linewidth = 1.6, linestyle = :dash
    )
    Mke.axislegend(
        ax,
        vcat([style_meas, style_theo], rod_lines),
        vcat(["Measured", "Theoretical"], rod_labels_str);
        position = :rt, framevisible = true, labelsize = 18,
        rowgap = 1, padding = (6, 6, 6, 6),
    )

    Mke.save(
        joinpath(
            @__DIR__, "..", "assets",
            "qrm_thorax_vmi_vs_theoretical.png"
        ), fig; px_per_unit = 2
    )
    fig
end

# ╔═╡ 0703000b-0000-4000-8000-000000000090
verification = let
    sino_finite = all(isfinite, sino_basis.sino_iodine) &&
                  all(isfinite, sino_basis.sino_water)
    vmi_finite = all(
        all(isfinite, image) for image in values(vmi_HU_final)
    )
    checks = [
        (
            name = "both native kVp channels retained",
            value = length(nchannel_slab_counts.bins),
            pass = length(nchannel_slab_counts.bins) == 2,
        ),
        (
            name = "absolute response matches I0",
            value = nchannel_basis.I0_relerr,
            pass = nchannel_basis.I0_relerr < 5e-5,
        ),
        (
            name = "generalized Cong outputs finite",
            value = sino_finite, pass = sino_finite,
        ),
        (
            name = "canonical dual-kVp FBP kernels",
            value = basis_volumes.kernels,
            pass = basis_volumes.kernels == (
                water = :StandardSoftBlend,
                iodine = :OriginalDualKvpSoft,
            ),
        ),
        (
            name = "canonical VMI energies",
            value = de_vmi_energies,
            pass = de_vmi_energies == [50.0, 70.0, 100.0, 140.0],
        ),
        (
            name = "final VMI values finite",
            value = vmi_finite, pass = vmi_finite,
        ),
    ]
    passed = count(check -> check.pass, checks)
    rows = join([
        "| $(check.name) | $(check.value) | $(check.pass ? "✅" : "❌") |"
        for check in checks
    ], "\n")
    Markdown.parse("""
### $(passed == length(checks) ? "✅ Verification: PASS" : "❌ Verification: CHECK")

| check | value | pass |
|---|---:|:---:|
$rows
""")
end

# ╔═╡ 0703000c-0000-4000-8000-000000000001
md"""
## Summary

```
QRM-Thorax mid-slice mask (1600 × 1100 × 20 phantom @ 0.2 mm iso,
                           rods bored at labels 9–12)
   → Forward-project (80 + 140 kVp dual-kVp GSI)
   → Fully inlined K=2 profiled Cong  (iodine + water basis)
   → Validated dual-kVp per-basis FBP kernels
   → Kalender ACNR  (five passes, beta_max=14)
   → Monoenergetic VMI Synthesis  (textbook 2-basis, mono μρ_water divisor)
   → Per-rod Measured vs Theoretical Regression
        (water · lipid · collagen · iodine at 50 / 70 / 100 / 140 keV)
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
# ╟─0703000b-0000-4000-8000-000000000090
# ╟─0703000c-0000-4000-8000-000000000001
