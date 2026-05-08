### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 07010003-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.update()
end

# ╔═╡ 07010003-0000-4000-8000-000000000005
using Markdown: @md_str, Markdown

# ╔═╡ 07010003-0000-4000-8000-000000000006
using Statistics: mean, std, quantile

# ╔═╡ 07010003-0000-4000-8000-000000000007
using Unitful: @u_str

# ╔═╡ 07010001-0000-4000-8000-000000000001
md"""
# 07 · QRM-Thorax Pure-Material VMI · Projection-Domain Pipeline

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
                  ├─→ Joint Sinogram Denoising → Material Decomp →     │
Simulate 140 kVp→┘   FBP × 2 → z-Median → 2-basis VMI → Mono+ →        │
                                                                       │
                     Per-Rod Measured-vs-Theoretical Regression  ──────┘
                                       at 40 / 70 / 100 / 140 keV
```

This notebook is structured in two parts.  **Part 1 (this file, §1)**
builds the phantom: read the rotated downsampled mask cache, define
the four pure-material rod inserts in `XA.Material` style, build the
3D phantom, visualize.  **Part 2 (added later)** runs the dual-kVp
scans and the full projection-domain VMI pipeline.

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
## Setup
"""

# ╔═╡ 07010003-0000-4000-8000-000000000010
import BasisSimulator as BS

# ╔═╡ 07010003-0000-4000-8000-000000000011
import CairoMakie as CM

# ╔═╡ 07010003-0000-4000-8000-000000000040
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

# ╔═╡ 07010003-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 07020001-0000-4000-8000-000000000001
md"""
## 1. `Phantom`: QRM-Thorax with 4 Pure-Material Rods

Read the **prepared** QRM-Thorax mask — already rotated to CT display
orientation (spine at the bottom) and 2× downsampled, cached at
`docs/notebooks/data/qrm_thorax/qrm_thorax_1600x1100_rot_uint8.raw`.
Phantom shape after z-tiling: **1600 × 1100 × 16** at **0.2 mm
in-plane × 0.625 mm in-z** (physical extent 320 × 220 × 10 mm; body
envelope ≈ 30 × 20 cm matches QRM-Thorax-small spec).  The probe
cell after the mask load prints the actual body bbox.

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
const QRM_TARGET_NX = 800;   # 1600 cache / 2× extra NN-downsample

# ╔═╡ 07020001-0000-4000-8000-000000000012
const QRM_TARGET_NY = 550;   # 1100 cache / 2× extra NN-downsample

# ╔═╡ 07020001-0000-4000-8000-000000000013
const QRM_TARGET_NZ = 16;    # matches nb03 Gammex 472 z-budget (16 × 0.625 mm = 10 mm)

# ╔═╡ 07020001-0000-4000-8000-000000000014
const QRM_VOXEL_SIZE_CM = (0.04, 0.04, 0.0625);   # (x, y, z) cm — 0.4 mm in-plane (32 × 22 cm physical extent), 0.625 mm z

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
    # Cache is 1600 × 1100 UInt8.  Read it, then NN-downsample 2×
    # in-plane to 800 × 550 (matches QRM_TARGET_NX / QRM_TARGET_NY) for
    # forward-sim throughput — same physical extent (32 × 22 cm),
    # 4× fewer voxels per slice.
    cache_2d = reshape(read(QRM_CACHE_PATH), 1600, 1100)
    nx_in, ny_in = size(cache_2d)
    out_2d = Matrix{UInt8}(undef, QRM_TARGET_NX, QRM_TARGET_NY)
    @inbounds for j in 1:QRM_TARGET_NY
        sj = clamp(round(Int, ((j - 0.5) * ny_in / QRM_TARGET_NY) + 0.5), 1, ny_in)
        for i in 1:QRM_TARGET_NX
            si = clamp(round(Int, ((i - 0.5) * nx_in / QRM_TARGET_NX) + 0.5), 1, nx_in)
            out_2d[i, j] = cache_2d[si, sj]
        end
    end
    repeat(out_2d; outer = (1, 1, QRM_TARGET_NZ))
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

Tune `ROD_HEART_CENTER_MM`, `ROD_RADIUS_MM`, `ROD_OFFSET_MM` to align
the rods with your phantom's actual heart cavity.  Defaults are set
for the 1600 × 1100 (320 × 220 mm) post-rotation phantom.

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
const ROD_HEART_CENTER_PX = (400, 327);   # ≈ (50%, 59%) of the 800 × 550 frame (halved after 2× downsample)

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
## 2. `Scanner`: GE Revolution Apex Elite
"""

# ╔═╡ 07030001-0000-4000-8000-000000000010
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

# ╔═╡ 07030002-0000-4000-8000-000000000001
md"""
## 3. Dual-kVp Protocols (Rapid kVp Switching)

| kVp | Instantaneous mA | Duty cycle | Effective mA |
|-----|------------------|------------|--------------|
| 80  | 407              | 0.65       | 264.55       |
| 140 | 405              | 0.35       | 141.75       |
"""

# ╔═╡ 07030002-0000-4000-8000-000000000010
protocol_low = BS.CTProtocol(
    kVp = 80,
    mA = 407 * 0.65,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 07030002-0000-4000-8000-000000000020
protocol_high = BS.CTProtocol(
    kVp = 140,
    mA = 405 * 0.35,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 07030003-0000-4000-8000-000000000001
md"""
## 4. `SimOptions` and `ReconOptions`
"""

# ╔═╡ 07030003-0000-4000-8000-000000000010
sim_opts = BS.SimOptions(fidelity = :eict, seed = 1234);

# ╔═╡ 07030003-0000-4000-8000-000000000020
# Standard CT recon convention: 512 × 512 in-plane regardless of phantom
# resolution.  The phantom is the high-res forward-projector sampling
# source (800 × 550 at 0.4 mm); the recon grid is independent (512 × 512
# at 0.683 mm, fov 35 cm).  Rod ROIs are built in physical mm and
# converted to recon voxel indices — the two grids share the isocenter
# so the conversion is just a centered linear transform.
recon_opts = BS.ReconOptions(
    algorithm = :fdk,
    matrix_size = (512, 512, 8),
    fov_cm = 35.0,
    z_cm = 0.5,
    filter = :standard,
);

# ╔═╡ 07030004-0000-4000-8000-000000000001
md"""
## 5. Forward Project

Run `BS.simulate!` on each kVp protocol.  The EICT path bakes in
per-ray spatial scatter + Compton + Rayleigh, and we keep the
simulator's noisy line-integral sinogram (`ws.sinogram`).
"""

# ╔═╡ 07030004-0000-4000-8000-000000000010
sim_low = let
    @info "Simulating: 80 kVp / $(round(protocol_low.mA, digits = 1)) mA-eff (DE low)…"
    ws = BS.create_eict_workspace(scanner, protocol_low, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, protocol_low, sim_opts)
    result = (sino = Array(ws.sinogram), geom = ws.geom)
    ws = nothing; GC.gc(true)
    result
end;

# ╔═╡ 07030004-0000-4000-8000-000000000020
sim_high = let
    @info "Simulating: 140 kVp / $(round(protocol_high.mA, digits = 1)) mA-eff (DE high)…"
    ws = BS.create_eict_workspace(scanner, protocol_high, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, protocol_high, sim_opts)
    result = (sino = Array(ws.sinogram), geom = ws.geom)
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

# ╔═╡ 07030005-0000-4000-8000-000000000001
md"""
## 6. Joint Sinogram SVD Denoiser

Per-row 2-channel SVD on the (sino_low, sino_high) pair: U[:, 1]
keeps common log-attenuation structure, U[:, 2] is the spectral
residual + decorrelated quantum noise smoothed with a separable 2-D
Gaussian σ in (col, view).  Single knob `SINO_DENOISE_σ_PX`.
"""

# ╔═╡ 201bccb5-41ba-4adf-a9df-44ef532ff062
md"""
!!! warning "NOTICE"
    See how the `SINO_DENOISE_σ_PX` is set to `0.0`. This makes the SVD denoiser non-operational.
    It is important to increase the denoising (e.g. the `SINO_DENOISE_σ_PX`) as the phantom iodine density and beam hardening issues
    increase, but for a small phantom with relatively low contrast materials, the SVD denoiser might
    simply decrease the resolution without any noticeable quantitative gains, since beam hardening
    issues are not likely to occur.
"""

# ╔═╡ 07030005-0000-4000-8000-000000000005
SINO_DENOISE_σ_PX = 0.0;

# ╔═╡ 07030005-0000-4000-8000-000000000010
sino_denoised = let
    sino_lo = Float32.(sim_low.sino)
    sino_hi = Float32.(sim_high.sino)
    out = BS.apply_sino_svd_denoise([sino_lo, sino_hi]; σ_px = SINO_DENOISE_σ_PX)
    (low = out[1], high = out[2], geom = sim_low.geom)
end;

# ╔═╡ 07030005-0000-4000-8000-000000000030
let
    n_row = size(sino_denoised.low, 2)
    mid_r = n_row ÷ 2 + 1

    slice_lo = permutedims(sino_denoised.low[:, mid_r, :], (2, 1))
    slice_hi = permutedims(sino_denoised.high[:, mid_r, :], (2, 1))

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

    for (c, ttl, sub, slice) in (
            (1, "80 kVp", "After SVD denoiser", slice_lo),
            (2, "140 kVp", "After SVD denoiser", slice_hi),
        )
        ax = CM.Axis(fig[1, c]; title = ttl, subtitle = sub, axis_kwargs...)
        CM.heatmap!(ax, slice; colormap = :viridis, colorrange = sino_window)
    end
    CM.Colorbar(
        fig[1, 3]; colormap = :viridis, colorrange = sino_window,
        label = "Log Line Integral", width = 16, labelsize = 22, ticklabelsize = 18
    )
    fig
end
# ╔═╡ 07030005-0000-4000-8000-000000000040
md"""
## 6b. Subspace–Frequency Joint Bilateral Denoiser  *(experimental — v2)*

A second projection-domain denoiser built on the same per-row SVD
skeleton as §6, with **a single user knob** (the principal scale `σ₀`).
Every other quantity is fixed by RSKR (Clark, Badea 2023) or
auto-derived from the photon-count map.

### Pipeline

1. **Per-pixel Poisson whitening.**  Subtract a heavy low-pass reference
   `p̄ₖ⁽⁰⁾` (10-px Gaussian, fixed) and scale by `√Nₖ`, with `Nₖ`
   recovered from `I0,k · exp(−pₖ)`.  Whitened residuals `ξₖ` have
   ~unit per-pixel noise variance.
2. **Per-row SVD** of the whitened `(col, view)` matrix
   `M_r = [vec(ξ_lo)  vec(ξ_hi)]`.
3. **Joint bilateral on both subspaces** with rank-sparse bandwidth
   `σ_e = σ₀·√(Σ₁/Σ_e)` (paper-fixed exponent γ=½).  Range scale
   `σ_e^rng = 1.4826·MAD(Λ_e)` (no `h` knob — paper §2.4 absorbs it).
   Locally-averaged range over a fixed 5×5 window.  Stride
   `s ∈ {1, 2, 3}` derived from the noise correlation length measured
   on the input.
4. **Iterate** with `σ₀⁽ᵗ⁾ = 0.7ᵗ · σ₀★`.  Iteration count
   `n_iter ∈ {1, 2}` derived from `min(N)`: 1 if min photon count > 100
   everywhere, else 2.
5. **Inverse-whiten** to recover log-line-integrals.

### The single knob

`SFJBF_σ0` is the only quantity the implementer sets.  Set to `0.0` to
defer to **SURE** — Stein's unbiased risk estimator with Hutchinson MC
divergence and golden-section search on a representative mid-row.  Set
positive to override.  All auto-derived values (corr length, stride,
n_iter, σ₀★) are reported via `@info` per run.

!!! info "Reference paper"
    Black, *Joint Sinogram Denoising via Subspace–Frequency Reduction
    for Two-Channel Spectral CT (v2)*, in prep.
"""

# ╔═╡ 07030005-0000-4000-8000-000000000050
# The only user-facing knob.  0.0 → SURE auto-selects on the mid-row;
# any positive value → use directly.  Every other quantity (γ=½, h=1,
# 5×5 lavg, α=0.7, σ_ref=10 px) is fixed by RSKR / paper §2.4–2.6, and
# stride / n_iter / σ_e^rng are derived from the photon-count map.
SFJBF_σ0 = 0.0

# ╔═╡ 07030005-0000-4000-8000-000000000060
begin
    import LinearAlgebra
    import Statistics
    import Random

    # ─── Paper-fixed constants (§2.4 / §2.6) ─────────────────────────────
    const _SFJBF_α        = 0.7f0    # iteration decay
    const _SFJBF_lavg     = 5        # locally-averaged range window (5×5)
    const _SFJBF_σ_ref_px = 10.0f0   # heavy low-pass reference (Gaussian)
    const _SFJBF_σ_cap    = 12.0f0   # internal compute safety on σ_e (paper §2.9)

    # ─── Per-row 2D separable Gaussian (used for the heavy low-pass reference
    #     and inside the noise-correlation-length helper) ──────────────────
    function _sfjbf_sep_gauss_3d(sino::Array{Float32, 3}, σ_px::Real)
        σ = Float64(σ_px)
        σ ≤ 0 && return copy(sino)
        radius = max(1, ceil(Int, 3σ))
        ks = Float32[exp(-(k^2) / (2σ^2)) for k in -radius:radius]
        ks ./= sum(ks)
        n_col, n_row, n_view = size(sino)
        out = similar(sino)
        Threads.@threads for r in 1:n_row
            slice = Float32.(@view sino[:, r, :])
            out[:, r, :] .= _sfjbf_gauss_2d(slice, ks, radius)
        end
        out
    end

    function _sfjbf_gauss_2d(slice2d::AbstractMatrix{Float32},
                              ks::AbstractVector{Float32}, radius::Int)
        nc, nv = size(slice2d)
        tmp = Matrix{Float32}(undef, nc, nv)
        out = Matrix{Float32}(undef, nc, nv)
        @inbounds for v in 1:nv, c in 1:nc
            s = 0f0; w = 0f0
            for (i, dk) in enumerate(-radius:radius)
                c2 = c + dk
                (1 ≤ c2 ≤ nc) || continue
                s += ks[i] * slice2d[c2, v]; w += ks[i]
            end
            tmp[c, v] = s / w
        end
        @inbounds for v in 1:nv, c in 1:nc
            s = 0f0; w = 0f0
            for (i, dk) in enumerate(-radius:radius)
                v2 = v + dk
                (1 ≤ v2 ≤ nv) || continue
                s += ks[i] * tmp[c, v2]; w += ks[i]
            end
            out[c, v] = s / w
        end
        out
    end

    # ─── Robust MAD scale (paper Eq 14) ──────────────────────────────────
    function _sfjbf_mad_scale(arr2d::AbstractMatrix{Float32})
        med = Statistics.median(arr2d)
        mad = Statistics.median(abs.(arr2d .- med))
        Float32(1.4826 * mad)
    end

    # ─── Joint bilateral pass on a (n_col, n_view) slice.
    #     Range kernel is the product-of-channels form (paper Eq 12) with
    #     locally-averaged squared diff (Eq 13, fixed 5×5 window).  Range
    #     strength h=1 absorbed into σ_e^rng. Stride from auto-rule.        ─
    function _sfjbf_pass!(out::AbstractMatrix{Float32},
                          target::AbstractMatrix{Float32},
                          σ_sp::Float32,
                          Λ1::AbstractMatrix{Float32}, Λ2::AbstractMatrix{Float32},
                          σ1_rng::Float32, σ2_rng::Float32, stride::Int)
        nc, nv = size(target)
        radius = max(1, ceil(Int, 3 * Float64(σ_sp)))
        inv_2σ²_sp = 1f0 / (2f0 * σ_sp * σ_sp + 1f-30)
        inv_2σ²_r1 = 1f0 / (2f0 * σ1_rng * σ1_rng + 1f-30)
        inv_2σ²_r2 = 1f0 / (2f0 * σ2_rng * σ2_rng + 1f-30)
        half_lavg = _SFJBF_lavg ÷ 2

        Threads.@threads for v in 1:nv
            @inbounds for c in 1:nc
                sum_v = 0f0; wtot = 0f0
                for dv in -radius:stride:radius
                    v2 = v + dv
                    (1 ≤ v2 ≤ nv) || continue
                    for dc in -radius:stride:radius
                        c2 = c + dc
                        (1 ≤ c2 ≤ nc) || continue

                        # 5×5 locally-averaged squared diff
                        Δ1² = 0f0; Δ2² = 0f0; cnt = 0
                        for dav in -half_lavg:half_lavg, dac in -half_lavg:half_lavg
                            ca = c + dac;  va = v + dav
                            cb = c2 + dac; vb = v2 + dav
                            (1 ≤ ca ≤ nc && 1 ≤ va ≤ nv) || continue
                            (1 ≤ cb ≤ nc && 1 ≤ vb ≤ nv) || continue
                            d1 = Λ1[cb, vb] - Λ1[ca, va]
                            d2 = Λ2[cb, vb] - Λ2[ca, va]
                            Δ1² += d1 * d1; Δ2² += d2 * d2; cnt += 1
                        end
                        if cnt > 0
                            Δ1² /= cnt; Δ2² /= cnt
                        end

                        spatial_d² = Float32(dc * dc + dv * dv)
                        log_w = -spatial_d² * inv_2σ²_sp -
                                Δ1² * inv_2σ²_r1 -
                                Δ2² * inv_2σ²_r2
                        w = exp(log_w)
                        sum_v += w * target[c2, v2]
                        wtot  += w
                    end
                end
                out[c, v] = sum_v / max(wtot, 1f-30)
            end
        end
        nothing
    end

    # ─── Single-row forward pass of the full operator D (paper Eq 15).
    #     Used by both the SURE optimization and the main loop.            ─
    function _sfjbf_apply_D(M::AbstractMatrix{Float32}, σ0::Float32,
                             n_col::Int, n_view::Int, stride::Int)
        F = LinearAlgebra.svd(M; full = false)
        U, Σ, V = F.U, F.S, F.V

        σ1 = min(σ0,                                 _SFJBF_σ_cap)
        σ2 = min(σ0 * sqrt(Σ[1] / max(Σ[2], 1f-12)), _SFJBF_σ_cap)

        Λ1 = reshape(copy(@view U[:, 1]), n_col, n_view)
        Λ2 = reshape(copy(@view U[:, 2]), n_col, n_view)
        σ1_rng = max(_sfjbf_mad_scale(Λ1), eps(Float32))
        σ2_rng = max(_sfjbf_mad_scale(Λ2), eps(Float32))

        Λ1_d = similar(Λ1); Λ2_d = similar(Λ2)
        _sfjbf_pass!(Λ1_d, Λ1, σ1, Λ1, Λ2, σ1_rng, σ2_rng, stride)
        _sfjbf_pass!(Λ2_d, Λ2, σ2, Λ1, Λ2, σ1_rng, σ2_rng, stride)

        U_d = hcat(vec(Λ1_d), vec(Λ2_d))
        U_d * LinearAlgebra.Diagonal(Σ) * V'
    end

    # ─── Stein's unbiased risk estimator with Hutchinson MC divergence.
    #     Whitened identity-cov coordinates (paper Eq 16+17).              ─
    function _sfjbf_sure(M::AbstractMatrix{Float32}, σ0::Float32,
                          n_col::Int, n_view::Int, stride::Int)
        Md = _sfjbf_apply_D(M, σ0, n_col, n_view, stride)

        rng = Random.MersenneTwister(42)
        b   = randn(rng, Float32, size(M))
        δ   = max(Float32(1f-3) * Float32(Statistics.std(M)), Float32(1f-8))
        Md_p = _sfjbf_apply_D(M .+ δ .* b, σ0, n_col, n_view, stride)
        div_est = sum(b .* (Md_p .- Md)) / δ

        n = length(M)
        Float32(sum((Md .- M) .^ 2)) - Float32(n) + 2f0 * div_est
    end

    # ─── Golden-section search for σ_0★ on a representative row. ─────────
    function _sfjbf_sure_optimize(M::AbstractMatrix{Float32},
                                    n_col::Int, n_view::Int, stride::Int;
                                    σ_lo::Real = 0.5, σ_hi::Real = 5.0,
                                    tol::Real  = 0.15)
        φ  = Float32((sqrt(5) - 1) / 2)
        a, b = Float32(σ_lo), Float32(σ_hi)
        c = b - φ * (b - a)
        d = a + φ * (b - a)
        fc = _sfjbf_sure(M, c, n_col, n_view, stride)
        fd = _sfjbf_sure(M, d, n_col, n_view, stride)
        n_evals = 2
        while abs(b - a) > tol
            if fc < fd
                b = d; d = c; fd = fc
                c = b - φ * (b - a)
                fc = _sfjbf_sure(M, c, n_col, n_view, stride)
            else
                a = c; c = d; fc = fd
                d = a + φ * (b - a)
                fd = _sfjbf_sure(M, d, n_col, n_view, stride)
            end
            n_evals += 1
        end
        σ_star = (a + b) / 2f0
        @info "[SF-jBF SURE] converged: σ₀★ = $(round(σ_star, digits=2)) px after $(n_evals) evals"
        σ_star
    end

    # ─── Noise correlation length (FWHM of autocovariance of high-pass log
    #     residuals on a flat sinogram patch) → stride.                    ─
    function _sfjbf_corr_length(p::Array{Float32, 3})
        n_col, n_row, n_view = size(p)
        c0 = max(1, n_col ÷ 2 - 5); c1 = min(n_col, n_col ÷ 2 + 4)
        mid_r = n_row ÷ 2 + 1
        patch = Float32.(p[c0:c1, mid_r, :])

        σ = 10.0
        radius = ceil(Int, 3σ)
        ks = Float32[exp(-(k^2) / (2σ^2)) for k in -radius:radius]
        ks ./= sum(ks)
        smoothed = similar(patch)
        @inbounds for c_i in axes(patch, 1), v in axes(patch, 2)
            s = 0f0; w = 0f0
            for (i, dk) in enumerate(-radius:radius)
                v2 = v + dk
                (1 ≤ v2 ≤ size(patch, 2)) || continue
                s += ks[i] * patch[c_i, v2]; w += ks[i]
            end
            smoothed[c_i, v] = s / w
        end
        resid = patch .- smoothed

        n_lags = 5
        ac = zeros(Float32, n_lags)
        for lag in 0:(n_lags - 1)
            s = 0.0; n = 0
            for c_i in 1:(size(resid, 1) - lag), v in axes(resid, 2)
                s += Float64(resid[c_i, v]) * Float64(resid[c_i + lag, v])
                n += 1
            end
            ac[lag + 1] = Float32(s / n)
        end

        ac0 = ac[1]
        ac0 ≤ 0 && return 1.0
        for lag in 1:(n_lags - 1)
            r1 = ac[lag] / ac0
            r2 = ac[lag + 1] / ac0
            if r2 ≤ 0.5
                return Float64((lag - 1) + (r1 - 0.5) / max(r1 - r2, 1f-6))
            end
        end
        Float64(n_lags - 1)
    end

    _sfjbf_pick_stride(corr_len::Real) =
        corr_len < 1.5 ? 1 : (corr_len < 2.5 ? 2 : 3)

    _sfjbf_pick_n_iter(min_N::Real) = min_N ≥ 100 ? 1 : 2

    nothing
end

# ╔═╡ 07030005-0000-4000-8000-000000000070
sino_denoised_alt = let
    p_lo = Float32.(sim_low.sino)
    p_hi = Float32.(sim_high.sino)
    n_col, n_row, n_view = size(p_lo)

    # ─── Recover approximate photon counts N_k = I0_k · exp(-p_k).  I0_k
    #     uses the source-flux × pixel-area × time × mA formula.  We
    #     deliberately ignore per-pixel bowtie attenuation and detector
    #     η_eff: the auto-rules below (n_iter, stride, SURE σ_0 in pixel
    #     units) only need order-of-magnitude photon counts.  Inlined
    #     (no closure) so Pluto's reactive analyzer sees every dependency
    #     at the cell's top level. ────────────────────────────────────────
    _, w_spec_lo = BS.resolve_source_spectrum_without_bowtie(
        sim_opts, protocol_low; scanner = scanner,
    )
    _, w_spec_hi = BS.resolve_source_spectrum_without_bowtie(
        sim_opts, protocol_high; scanner = scanner,
    )
    I0_lo = Float32(BS.compute_detector_I0(
        sim_low.geom,  protocol_low,  Float64(sum(w_spec_lo)),
    ))
    I0_hi = Float32(BS.compute_detector_I0(
        sim_high.geom, protocol_high, Float64(sum(w_spec_hi)),
    ))
    N_lo  = I0_lo .* exp.(-p_lo)
    N_hi  = I0_hi .* exp.(-p_hi)
    min_N = Float64(min(minimum(N_lo), minimum(N_hi)))

    # ─── Auto-derive stride and n_iter (paper §2.6 auto-rules) ───────────
    corr_len = max(_sfjbf_corr_length(p_lo), _sfjbf_corr_length(p_hi))
    stride   = _sfjbf_pick_stride(corr_len)
    n_iter   = _sfjbf_pick_n_iter(min_N)

    @info "[SF-jBF auto] I0_lo = $(round(Int, I0_lo)) ph, I0_hi = $(round(Int, I0_hi)) ph, min(N) = $(round(Int, min_N)) ph"
    @info "[SF-jBF auto] noise corr length = $(round(corr_len, digits=2)) px → stride = $(stride)"
    @info "[SF-jBF auto] n_iter = $(n_iter) (rule: 1 if min(N) ≥ 100 everywhere)"

    # ─── Heavy low-pass reference (10-px Gaussian, paper §2.2 fixed) ─────
    p_ref_lo = _sfjbf_sep_gauss_3d(p_lo, _SFJBF_σ_ref_px)
    p_ref_hi = _sfjbf_sep_gauss_3d(p_hi, _SFJBF_σ_ref_px)

    # ─── Whitened residuals ξ_k = √N_k · (p_k − p̄_k⁽⁰⁾) ─────────────────
    w_lo = sqrt.(max.(N_lo, 1f0))
    w_hi = sqrt.(max.(N_hi, 1f0))
    ξ_lo = w_lo .* (p_lo .- p_ref_lo)
    ξ_hi = w_hi .* (p_hi .- p_ref_hi)

    p_lo = nothing; p_hi = nothing
    N_lo = nothing; N_hi = nothing
    GC.gc(true)

    # ─── Auto-derive σ_0★ via SURE on the mid-row, or use user's value. ─
    σ0_user = Float32(SFJBF_σ0)
    σ0_star = if σ0_user > 0
        @info "[SF-jBF] σ_0 from user knob: $(σ0_user) px (SURE skipped)"
        σ0_user
    else
        mid_r = n_row ÷ 2 + 1
        slice_lo = Float32.(@view ξ_lo[:, mid_r, :])
        slice_hi = Float32.(@view ξ_hi[:, mid_r, :])
        M_mid = hcat(vec(slice_lo), vec(slice_hi))
        @info "[SF-jBF SURE] running on mid-row r=$(mid_r), σ ∈ [0.5, 5.0] px..."
        _sfjbf_sure_optimize(M_mid, n_col, n_view, stride)
    end

    # ─── Run the operator with σ_0^(t) = α^t · σ_0★ ──────────────────────
    Σ_ratio_log = Vector{Float32}(undef, n_row)
    σ0 = σ0_star
    for t in 0:(n_iter - 1)
        Threads.@threads for r in 1:n_row
            slice_lo = Float32.(@view ξ_lo[:, r, :])
            slice_hi = Float32.(@view ξ_hi[:, r, :])
            M = hcat(vec(slice_lo), vec(slice_hi))

            F = LinearAlgebra.svd(M; full = false)
            U, Σ, V = F.U, F.S, F.V

            σ1 = min(σ0,                                 _SFJBF_σ_cap)
            σ2 = min(σ0 * sqrt(Σ[1] / max(Σ[2], 1f-12)), _SFJBF_σ_cap)
            Σ_ratio_log[r] = Float32(Σ[1] / max(Σ[2], 1f-12))

            Λ1 = reshape(copy(@view U[:, 1]), n_col, n_view)
            Λ2 = reshape(copy(@view U[:, 2]), n_col, n_view)
            σ1_rng = max(_sfjbf_mad_scale(Λ1), eps(Float32))
            σ2_rng = max(_sfjbf_mad_scale(Λ2), eps(Float32))

            Λ1_d = similar(Λ1); Λ2_d = similar(Λ2)
            _sfjbf_pass!(Λ1_d, Λ1, σ1, Λ1, Λ2, σ1_rng, σ2_rng, stride)
            _sfjbf_pass!(Λ2_d, Λ2, σ2, Λ1, Λ2, σ1_rng, σ2_rng, stride)

            U_d = hcat(vec(Λ1_d), vec(Λ2_d))
            M_d = U_d * LinearAlgebra.Diagonal(Σ) * V'
            ξ_lo[:, r, :] .= reshape(view(M_d, :, 1), n_col, n_view)
            ξ_hi[:, r, :] .= reshape(view(M_d, :, 2), n_col, n_view)
        end
        @info "[SF-jBF iter $(t + 1)/$(n_iter)] σ₀ = $(round(σ0, digits=2)) px, " *
              "median Σ₁/Σ₂ = $(round(Statistics.median(Σ_ratio_log), sigdigits=3)), " *
              "σ_cap = $(_SFJBF_σ_cap) px"
        σ0 *= _SFJBF_α
    end

    # ─── Inverse-whiten: p = ξ/√N + p̄⁽⁰⁾ ────────────────────────────────
    p_lo_d = ξ_lo ./ w_lo .+ p_ref_lo
    p_hi_d = ξ_hi ./ w_hi .+ p_ref_hi

    ξ_lo = nothing; ξ_hi = nothing
    w_lo = nothing; w_hi = nothing
    p_ref_lo = nothing; p_ref_hi = nothing
    GC.gc(true)

    (low = p_lo_d, high = p_hi_d, geom = sim_low.geom)
end;

# ╔═╡ 07030005-0000-4000-8000-000000000080
let
    n_row = size(sino_denoised_alt.low, 2)
    mid_r = n_row ÷ 2 + 1

    slice_lo = permutedims(sino_denoised_alt.low[:, mid_r, :], (2, 1))
    slice_hi = permutedims(sino_denoised_alt.high[:, mid_r, :], (2, 1))

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
        (1, 1, "80 kVp",  "After SF-jBF denoiser", slice_lo),
        (1, 2, "140 kVp", "After SF-jBF denoiser", slice_hi),
    )

    for (r, c, ttl, sub, slice) in panels
        ax = CM.Axis(fig[r, c]; title = ttl, subtitle = sub, axis_kwargs...)
        CM.heatmap!(ax, slice; colormap = :viridis, colorrange = sino_window)
    end
    CM.Colorbar(
        fig[1, 3]; colormap = :viridis, colorrange = sino_window,
        label = "Log Line Integral", width = 16, labelsize = 22, ticklabelsize = 18
    )
    fig
end

# ╔═╡ 07030005-0000-4000-8000-000000000090
md"""
## 6c. Active Denoiser Path

Toggle between the §6 baseline and the §6b SF-jBF denoiser without
touching anything downstream. `sino_denoised_active` is what §7 onward
consumes — flip `DENOISER_PATH`, re-run the rest of the notebook, and
compare per-rod measured-vs-theoretical RMSE in §11.

| `DENOISER_PATH` | Path        | Knobs                                       |
|-----------------|-------------|---------------------------------------------|
| `:svd`          | §6 baseline | `SINO_DENOISE_σ_PX`                         |
| `:sfjbf`        | §6b SF-jBF  | `SFJBF_σ0` (one — `0.0` ⇒ SURE auto-select) |
"""

# ╔═╡ 07030005-0000-4000-8000-000000000095
DENOISER_PATH = :sfjbf

# ╔═╡ 07030005-0000-4000-8000-000000000100
sino_denoised_active = if DENOISER_PATH == :sfjbf
    sino_denoised_alt
elseif DENOISER_PATH == :svd
    sino_denoised
else
    error("Unknown DENOISER_PATH = $(DENOISER_PATH); must be :svd or :sfjbf")
end;


# ╔═╡ 07030006-0000-4000-8000-000000000001
md"""
## 7. Projection Domain Material Decomposition

Per-ray Newton solver on the polychromatic transmission integral
(material-basis variant: iodine + water mass-attenuation tables seeded
with `water_basis = (a = 0, c = 1)`).  Bowtie-aware: the basis builder
uses `BS.resolve_source_spectrum_with_bowtie`, which returns a per-ray
3D spectral weight `ŵ[col, row, E]`.
"""

# ╔═╡ 07030006-0000-4000-8000-000000000010
material_basis = let
    e_L, ŵ_L = BS.resolve_source_spectrum_with_bowtie(
        sim_opts, protocol_low; scanner = scanner, geom = sim_low.geom,
    )
    e_H, ŵ_H = BS.resolve_source_spectrum_with_bowtie(
        sim_opts, protocol_high; scanner = scanner, geom = sim_high.geom,
    )

    ŵ_L_f32 = Float32.(ŵ_L ./ sum(ŵ_L; dims = ndims(ŵ_L)))
    ŵ_H_f32 = Float32.(ŵ_H ./ sum(ŵ_H; dims = ndims(ŵ_H)))

    iodine_mat = BS.XA.Elements.Iodine
    water_mat = BS.XA.Materials.water

    p_L = Float32[Float32(BS.compute_mass_μ_at_energy(iodine_mat, Float64(E))) for E in e_L]
    q_L = Float32[Float32(BS.compute_mass_μ_at_energy(water_mat, Float64(E))) for E in e_L]
    p_H = Float32[Float32(BS.compute_mass_μ_at_energy(iodine_mat, Float64(E))) for E in e_H]
    q_H = Float32[Float32(BS.compute_mass_μ_at_energy(water_mat, Float64(E))) for E in e_H]

    (
        ŵ_L = ŵ_L_f32, p_L = p_L, q_L = q_L,
        ŵ_H = ŵ_H_f32, p_H = p_H, q_H = q_H,
    )
end;

# ╔═╡ 07030006-0000-4000-8000-000000000020
sino_basis = let
    sino_low_gpu = to_gpu(sino_denoised_active.low)
    sino_high_gpu = to_gpu(sino_denoised_active.high)

    sino_y = similar(sino_low_gpu)
    sino_c = similar(sino_low_gpu)
    fill!(sino_y, 0.0f0); fill!(sino_c, 0.0f0)

    cong_ws = BS.create_cong_workspace(sino_low_gpu, material_basis)
    BS.apply_cong!(
        cong_ws, sino_y, sino_c, sino_low_gpu, sino_high_gpu;
        water_basis = (a = 0.0f0, c = 1.0f0),
    )

    result = (
        sino_iodine = Array(sino_y),
        sino_water = Array(sino_c),
        geom = sino_denoised_active.geom,
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
## 8. FBP: Iodine and Water Basis Maps

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

# ╔═╡ 07030008-0000-4000-8000-000000000001
md"""
## 9. Z-Direction Median Filter

1D median along z, per `(x, y)` voxel column.  `adjacent_slices = n` ⇒
`2n + 1`-slice window.  Cheap streak/outlier suppression that
exploits z-invariance of this tiled phantom.
"""

# ╔═╡ 07030008-0000-4000-8000-000000000005
Z_MEDIAN_ADJACENT = 2;

# ╔═╡ 07030008-0000-4000-8000-000000000010
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

# ╔═╡ 07030008-0000-4000-8000-000000000030
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

# ╔═╡ 07030009-0000-4000-8000-000000000001
md"""
## 10. VMI Synthesis

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

# ╔═╡ 07030009-0000-4000-8000-000000000010
de_vmi_energies = [40.0, 70.0, 100.0, 140.0];

# ╔═╡ 07030009-0000-4000-8000-000000000020
vmi_HU_by_keV = let
    c_iodine_mg_per_mL = basis_z.vol_iodine .* 1000.0f0

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
            basis_z.vol_water, c_iodine_mg_per_mL; energy_keV = E,
        )
    end
    out
end;

# ╔═╡ 07030009-0000-4000-8000-000000000040
let
    HU_window = (-200, 500)

    fig = CM.Figure(size = (1180, 1180))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    sample = vmi_HU_by_keV[40.0]
    mid = size(sample, 3) ÷ 2

    for (k, E) in enumerate(de_vmi_energies)
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

# ╔═╡ 0703000a-0000-4000-8000-000000000001
md"""
## 11. VMI Post-Processing (Mono+)

Frequency-split rule (Grant 2014):

```
Mono+(E)     = LP_σ(VMI_E) + VMI_opt − LP_σ(VMI_opt)
Mono+(E_opt) = VMI_opt   (identity at the noise-optimal anchor)
```

`σ_vmi_lp_px` pairs element-wise with `de_vmi_energies` — one σ per
VMI energy.  σ = 0 ⇒ identity (no LP, no FFT).  Default
`[1.0, 0.0, 1.0, 1.0]` smooths 40 / 100 / 140 toward the 70 keV
anchor; 70 stays identity.
"""

# ╔═╡ 0703000a-0000-4000-8000-000000000005
σ_vmi_lp_px = Float64[1.0, 0.0, 1.0, 1.0];

# ╔═╡ 0703000a-0000-4000-8000-000000000010
vmi_HU_final = let
    volumes = [vmi_HU_by_keV[E] for E in de_vmi_energies]

    ws = BS.create_mono_plus_workspace(
        volumes[1]; n_energies = length(de_vmi_energies)
    )
    BS.apply_mono_plus!(
        ws, volumes, de_vmi_energies;
        E_noise_opt = 70.0, σ_lp_px = σ_vmi_lp_px, verbose = true,
    )

    out = Dict{Float64, Array{Float32, 3}}()
    for (i, E) in enumerate(de_vmi_energies)
        out[E] = copy(ws.out_vols[i])
    end
    ws = nothing; GC.gc(true)
    out
end;

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

        rmse_i = sqrt(
            mean(
                (
                    vec(rod_data.measured[i, :]) .-
                        vec(rod_data.theoretical[i, :])
                ) .^ 2
            )
        )
        nm = get(pretty_name, rod_data.names[i], rod_data.names[i])
        rod_labels_str[i] = "$(nm) (RMSE = $(round(rmse_i, digits = 1)) HU)"
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
QRM-Thorax mid-slice mask (1600 × 1100 cache, 800 × 550 working,
                           rods bored at labels 9–12)
   → Forward-project (80 + 140 kVp dual-kVp GSI)
   → Joint Sinogram SVD Denoiser  (2-channel, BS.apply_sino_svd_denoise)
   → Projection-Domain Material Decomposition  (iodine + water basis)
   → FBP × 2  (iodine, water basis maps)
   → Z-Direction Median Filter
   → Monoenergetic VMI Synthesis  (textbook 2-basis, mono μρ_water divisor)
   → Mono+ Post-Processing  (per-keV σ via σ_vmi_lp_px)
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
# ╠═07010003-0000-4000-8000-000000000040
# ╟─07010003-0000-4000-8000-000000000050
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
# ╟─07030005-0000-4000-8000-000000000001
# ╟─201bccb5-41ba-4adf-a9df-44ef532ff062
# ╠═07030005-0000-4000-8000-000000000005
# ╠═07030005-0000-4000-8000-000000000010
# ╟─07030005-0000-4000-8000-000000000030
# ╟─07030005-0000-4000-8000-000000000040
# ╠═07030005-0000-4000-8000-000000000050
# ╠═07030005-0000-4000-8000-000000000060
# ╠═07030005-0000-4000-8000-000000000070
# ╟─07030005-0000-4000-8000-000000000080
# ╟─07030005-0000-4000-8000-000000000090
# ╠═07030005-0000-4000-8000-000000000095
# ╠═07030005-0000-4000-8000-000000000100
# ╟─07030006-0000-4000-8000-000000000001
# ╠═07030006-0000-4000-8000-000000000010
# ╠═07030006-0000-4000-8000-000000000020
# ╟─07030006-0000-4000-8000-000000000040
# ╟─07030007-0000-4000-8000-000000000001
# ╠═07030007-0000-4000-8000-000000000010
# ╟─07030007-0000-4000-8000-000000000030
# ╟─07030008-0000-4000-8000-000000000001
# ╠═07030008-0000-4000-8000-000000000005
# ╠═07030008-0000-4000-8000-000000000010
# ╟─07030008-0000-4000-8000-000000000030
# ╟─07030009-0000-4000-8000-000000000001
# ╠═07030009-0000-4000-8000-000000000005
# ╠═07030009-0000-4000-8000-000000000008
# ╠═07030009-0000-4000-8000-000000000010
# ╠═07030009-0000-4000-8000-000000000020
# ╟─07030009-0000-4000-8000-000000000040
# ╟─0703000a-0000-4000-8000-000000000001
# ╠═0703000a-0000-4000-8000-000000000005
# ╠═0703000a-0000-4000-8000-000000000010
# ╟─0703000a-0000-4000-8000-000000000040
# ╟─0703000b-0000-4000-8000-000000000000
# ╟─0703000b-0000-4000-8000-00000000000a
# ╟─0703000b-0000-4000-8000-000000000001
# ╟─0703000b-0000-4000-8000-000000000002
# ╟─0703000b-0000-4000-8000-000000000003
# ╠═0703000b-0000-4000-8000-000000000010
# ╠═0703000b-0000-4000-8000-000000000011
# ╠═0703000b-0000-4000-8000-000000000012
# ╠═0703000b-0000-4000-8000-000000000020
# ╟─0703000b-0000-4000-8000-000000000030
# ╟─0703000b-0000-4000-8000-000000000031
# ╟─0703000c-0000-4000-8000-000000000001
