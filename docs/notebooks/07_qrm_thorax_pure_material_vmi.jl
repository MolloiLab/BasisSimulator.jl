### A Pluto.jl notebook ###
# v0.2.6

using Markdown
using InteractiveUtils

# ╔═╡ 07010001-0000-4000-8000-000000000001
md"""
# QRM-Thorax Pure-Material VMI: Dual-kVp Reference

A GE Revolution Apex Elite rapid-kVp-switching scan (80 + 140 kVp) of a body-sized
**QRM-Thorax** phantom with **four pure-material rods** (water · lipid · collagen · 5 mg/mL
iodine) in its cardiac insert. The virtual monoenergetic images (VMIs) come from the
package's VMI chain, `BS.vmi_pipeline`, and every rod is checked against a noise-free
acquisition of the same scan and against the HU its material has in theory at 50, 70, 100
and 140 keV. The scanner, the acquisition and the chain are those of the
basis-spectral-denoising paper, as in notebook 03.

| Stage       | Matrix               | Voxel (mm)            | Extent                        |
|-------------|----------------------|-----------------------|-------------------------------|
| Phantom     | **1600 × 1100 × 20** | **0.2 isotropic**     | 320 × 220 × 4 mm              |
| Recon       | 512 × 512 × **3**    | **0.625 isotropic**   | FOV 32 cm × 1.875 mm z        |
| Collimation | **5 mm nominal**     | —                     | cone-guard rows added automatically |
| Scanner     | GE Revolution Apex Elite | 0.625 × 0.6 mm pixels | 256 × 834 detector       |
| Protocol    | rapid kVp switching  | 80 + 140 kVp, duty cycle 0.65 / 0.35, 984 views, 0.5 s rotation | dose-matched to 10.07 mGy CTDIvol |

The only economy is in **z**: three saved slices and a phantom deep enough to cover every ray
those slices use. The workspace adds the detector guard rows the reconstruction cylinder needs.

```
qrm_thorax_slice (analytic 2-D shapes) → bore rods 9–12 into the heart → tile z → BS.Phantom
   → simulate! at 80 kVp and at 140 kVp, each view integrated over its duty-cycle arc
     one projection per exposure → noise-free, measured and calibration draws
   → BS.spectral_basis_from_acquisitions     (per-ray absolute response of each kVp)
   → calibration draw → pair_basis (E*, β), composite_energy
   → BS.vmi_pipeline(; denoiser = SpectralHYPR(), fbp_filter = PairFilter(…), pair_basis, composite_energy)
        projection HYPR → K = 2 decomposition → FDK of the composite and complement
        → ACNR → image HYPR → VMI synthesis
   → per-rod measured vs noise-free vs theoretical HU at 50 / 70 / 100 / 140 keV
```

!!! tip "When to reach for this notebook"
    Use 07 for VMI accuracy on a body-sized phantom and field of view at the resolution a
    GE Revolution Apex Elite delivers. Notebook 03 runs the same chain on the smaller
    Gammex 472 phantom; notebook 08 scans this phantom with a photon-counting CT.

!!! info "Why pure end-members?"
    `XA.Materials.basis_fat` is ICRU-44 adipose tissue (≈ 83 % triolein, 17 % water and
    trace electrolytes), not a pure lipid. `XA.Materials.basis_lipid` (H/C/O at 0.92 g/cm³)
    and `basis_collagen` (H/C/N/O at 1.26 g/cm³) are mathematically pure end members, made
    for clean water/lipid/collagen decomposition checks.
"""

# ╔═╡ 07010002-0000-4000-8000-000000000001
md"""
## Notebook Setup
"""

# ╔═╡ 07010003-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 07010003-0000-4000-8000-000000000005
using Markdown: @md_str, Markdown

# ╔═╡ 07010003-0000-4000-8000-000000000006
using Statistics: mean, std, quantile

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
end;

# ╔═╡ 07010003-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 07020000-0000-4000-8000-0000000000f1
md"""
## Scan Setup and Simulation

One clinical rapid-kVp-switching acquisition: the QRM-Thorax phantom, the GE Apex Elite
scanner, the 80 + 140 kVp protocol, and the forward projection that turns them into two
sinograms.
"""

# ╔═╡ 07020001-0000-4000-8000-000000000001
md"""
### 1. `Phantom`: QRM-Thorax with 4 Pure-Material Rods

The QRM-Thorax phantom is built analytically, from 2-D shapes: a flat-fronted
superellipse body, the two lungs, the cardiac insert and the mediastinum above it, a water
rod and an air rod in the lungs, six ribs and the sternum, and the vertebra with its canal,
laminae and spinous process. Every dimension was measured on the labelled 1600 × 1100
(0.2 mm) mid-slice of the scanned phantom, in CT display orientation (spine at the bottom).
The slice is z-tiled to **1600 × 1100 × 20** at **0.2 mm isotropic** (320 × 220 × 4 mm),
finer than the demagnified detector pitch, so the forward projector samples a
high-resolution object.

| Label | Material         | Label  | Material             |
|:------|:-----------------|:-------|:---------------------|
| 1     | air              | 7      | air rod              |
| 2     | lung             | 8      | (unused)             |
| 3     | soft tissue      | **9**  | rod → water          |
| 4     | cortical bone    | **10** | rod → lipid          |
| 5     | bone marrow      | **11** | rod → collagen       |
| 6     | water rod (lung) | **12** | rod → 5 mg/mL iodine |

As in the reference slice, the cardiac insert is soft tissue (label 3, like the body) with
the four pure-material rods 9–12 bored into it, so label 8 is unused. `qrm_thorax_slice`
draws labels 1–7; the rods are bored below, at the centres `QRM_GEOMETRY.heart_rods` records.
Every material is a prebuilt `XA.Materials` entry.
"""

# ╔═╡ 07020001-0000-4000-8000-000000000010
begin
    # The QRM-Thorax mid-slice as 2-D shapes, measured on the labelled 1600 × 1100 (0.2 mm)
    # reference slice. Millimetres from the slice centre, +x to the image right and +y up:
    # the CT display orientation, sternum at the top and spine at the bottom.
    const QRM_GEOMETRY = (
        # |x/a|^n + |(y - yc)/b|^n ≤ 1, cut flat at the front and back (y range)
        body = (a = 150.2, b = 106.75, n = 2.30, yc = 0.0, y = (-100.2, 101.0)),
        lungs = (a = 127.6, b = 92.2, n = 2.10, yc = 0.4, y = (-76.8, 76.8)),
        # soft tissue between the lungs above the heart; its corners at the lung apex are rounded
        mediastinum = (half_width = 16.8, corner_r = 2.0),
        heart = (center = (0.0, 18.4), r = 55.2),     # the cardiac insert, soft tissue (label 3)
        # the four pure-material rods in the heart, as the reference slice has them (labels 9–12)
        heart_rods = (
            (center = (0.0, 45.8), r = 7.5, label = 9),              # north: water
            (center = (24.9, 20.7), r = 7.5, label = 10),            # east: lipid
            (center = (0.0, -4.3), r = 7.5, label = 11),             # south: collagen
            (center = (-25.1, 20.7), r = 7.5, label = 12),           # west: 5 mg/mL iodine
        ),
        lung_rods = (
            (center = (-91.4, 0.0), r = 7.5, label = 6),             # water rod
            (center = (90.2, 0.0), r = 7.5, label = 7),              # air rod
        ),
        # the spine; `cortex` is the thickness of the cortical-bone shell, mm
        vertebra = (center = (0.0, -53.3), r = 17.4, cortex = 0.6),
        canal = (center = (0.0, -69.6), r = 8.0, cortex = 0.6),      # the spinal canal
        laminae = (corners = ((-35.4, -77.2), (35.4, -77.2), (0.0, -62.3)), cortex = 1.0),
        spinous = (corners = ((-2.6, -76.8), (2.6, -76.8), (0.0, -91.6)), cortex = 0.6),
        ribs = (                                                     # (x, y, long-axis angle in °)
            (-112.4, 64.9, 40.0), (108.9, 67.9, -40.0),
            (-138.1, 1.5, 90.0), (139.0, 6.1, 90.0),
            (-109.2, -65.9, -40.0), (109.2, -67.5, 40.0),
        ),
        rib_axes = (8.2, 3.0),                                       # semi-axes, mm
        sternum = (center = (-2.7, 89.4), axes = (7.6, 3.2)),
        marrow_scale = 0.85, # rib and sternum marrow = their cortical ellipse scaled by this
    )

    """
        qrm_thorax_slice(g = QRM_GEOMETRY; n = (1600, 1100), voxel_mm = 0.2, inserts = ())

    Label image of the QRM-Thorax mid-slice (1 air, 2 lung, 3 soft tissue, including the
    cardiac insert, 4 cortical bone, 5 marrow, 6 water rod, 7 air rod). `inserts` are extra
    discs `(center, r, label)` in mm, painted last: the heart rods of `g.heart_rods`.
    """
    function qrm_thorax_slice(g = QRM_GEOMETRY; n = (1600, 1100), voxel_mm = 0.2, inserts = ())
        superellipse(x, y, s) =
            s.y[1] ≤ y ≤ s.y[2] && abs(x / s.a)^s.n + abs((y - s.yc) / s.b)^s.n ≤ 1
        disc(x, y, c, r) = (x - c[1])^2 + (y - c[2])^2 ≤ r^2
        function ellipse(x, y, c, axes, θ)
            s, co = sincosd(θ)
            u, v = co * (x - c[1]) + s * (y - c[2]), -s * (x - c[1]) + co * (y - c[2])
            return (u / axes[1])^2 + (v / axes[2])^2 ≤ 1
        end
        # inside all three edges, each moved `inset` mm toward the interior
        function triangle(x, y, t, inset = 0.0)
            for k in 1:3
                p, q, r = t[k], t[mod1(k + 1, 3)], t[mod1(k + 2, 3)]
                ex, ey = q[1] - p[1], q[2] - p[2]
                side(px, py) = (ex * (py - p[2]) - ey * (px - p[1])) / hypot(ex, ey)
                side(x, y) * sign(side(r...)) ≥ inset || return false
            end
            return true
        end
        function lung(x, y)
            superellipse(x, y, g.lungs) || return false
            disc(x, y, g.heart.center, g.heart.r) && return false
            m = g.mediastinum
            abs(x) < m.half_width && y > g.heart.center[2] && return false
            # round the apex corner where the lung top meets the mediastinum
            cx, cy = m.half_width + m.corner_r, g.lungs.y[2] - m.corner_r
            abs(x) < cx && y > cy && hypot(abs(x) - cx, y - cy) > m.corner_r && return false
            return true
        end

        bones = (
            ((c = (rx, ry), axes = g.rib_axes, θ = θ) for (rx, ry, θ) in g.ribs)...,
            (c = g.sternum.center, axes = g.sternum.axes, θ = 0.0),
        )
        img = ones(UInt8, n)
        for j in 1:n[2], i in 1:n[1]
            x, y = (i - (n[1] + 1) / 2) * voxel_mm, (j - (n[2] + 1) / 2) * voxel_mm
            superellipse(x, y, g.body) || continue
            label = lung(x, y) ? 2 : 3    # the heart is not lung, so it is soft tissue
            for rod in g.lung_rods
                disc(x, y, rod.center, rod.r) && (label = rod.label)
            end
            for b in bones
                ellipse(x, y, b.c, b.axes, b.θ) &&
                    (label = ellipse(x, y, b.c, b.axes .* g.marrow_scale, b.θ) ? 5 : 4)
            end
            for t in (g.laminae, g.spinous)
                triangle(x, y, t.corners) && (label = triangle(x, y, t.corners, t.cortex) ? 5 : 4)
            end
            v, cn = g.vertebra, g.canal
            disc(x, y, v.center, v.r) && (label = disc(x, y, v.center, v.r - v.cortex) ? 5 : 4)
            disc(x, y, cn.center, cn.r) && (label = disc(x, y, cn.center, cn.r - cn.cortex) ? 3 : 4)
            for (c, r, l) in inserts
                disc(x, y, c, r) && (label = l)
            end
            img[i, j] = label
        end
        return img
    end
end;

# ╔═╡ 07020001-0000-4000-8000-000000000013
const QRM_NZ = 20;    # 20 × 0.2 mm = 4 mm: a short z-invariant phantom

# ╔═╡ 07020001-0000-4000-8000-000000000014
const QRM_VOXEL_SIZE_CM = (0.02, 0.02, 0.02);   # (x, y, z) cm: 0.2 mm isotropic

# ╔═╡ 07020001-0000-4000-8000-000000000025
md"""
**Bore 4 rod inserts** into the heart where the reference slice has them
(`QRM_GEOMETRY.heart_rods`: 15 mm rods, 25 mm north, east, south and west of a point 20.7 mm
above the slice centre), as labels 9–12 (matching `materials_dict`):

| Direction | Label | Material              |
|-----------|-------|-----------------------|
| North     | 9     | `basis_water`         |
| East      | 10    | `basis_lipid`         |
| South     | 11    | `basis_collagen`      |
| West      | 12    | `gammex_472_i5_0`     |
"""

# ╔═╡ 07020001-0000-4000-8000-000000000026
# (center in mm, radius in mm, label) of each heart rod
HEART_RODS = [(rod.center, rod.r, rod.label) for rod in QRM_GEOMETRY.heart_rods];

# ╔═╡ 07020001-0000-4000-8000-000000000029
mask_3d = let
    slice = qrm_thorax_slice(; voxel_mm = QRM_VOXEL_SIZE_CM[1] * 10, inserts = HEART_RODS)
    repeat(slice; outer = (1, 1, QRM_NZ))
end;

# ╔═╡ 07020001-0000-4000-8000-000000000040
materials_dict = Dict{Int, BS.XA.Material}(
    # Anatomy
    1 => BS.XA.Materials.air,
    2 => BS.XA.Materials.lung,
    3 => BS.XA.Materials.muscle,           # soft tissue: ICRU-44 muscle
    4 => BS.XA.Materials.corticalbone,
    5 => BS.XA.Materials.marrow_red,
    6 => BS.XA.Materials.water,            # water rod in the lung
    7 => BS.XA.Materials.air,              # air rod
    8 => BS.XA.Materials.muscle,           # unused: the heart insert is soft tissue (label 3)
    # Pure-material rod inserts
    9 => BS.XA.Materials.basis_water,
    10 => BS.XA.Materials.basis_lipid,
    11 => BS.XA.Materials.basis_collagen,
    12 => BS.XA.Materials.gammex_472_i5_0,  # Gammex 472 5 mg/mL iodine insert
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
    3 => "3 soft tissue",
    4 => "4 cortical bone",
    5 => "5 marrow",
    6 => "6 water rod",
    7 => "7 air rod",
    8 => "8 (unused)",
    9 => "9 water",
    10 => "10 lipid",
    11 => "11 collagen",
    12 => "12 iodine 5 mg/mL",
);

# ╔═╡ 07020001-0000-4000-8000-000000000061
let
    slice = mask_3d[:, :, QRM_NZ ÷ 2 + 1]
    nx, ny = size(slice)
    px_mm = QRM_VOXEL_SIZE_CM[1] * 10
    xs = ((1:nx) .- (nx + 1) / 2) .* px_mm
    ys = ((1:ny) .- (ny + 1) / 2) .* px_mm
    n_lbl = length(unique(slice))

    fig = Mke.Figure(size = (1180, 700))
    ax = Mke.Axis(
        fig[1, 1];
        title = "QRM-Thorax mid-slice (labeled)",
        subtitle = "$(nx) × $(ny) at $(px_mm) mm/voxel · $(n_lbl) unique labels",
        xlabel = "x (mm)", ylabel = "y (mm)", aspect = Mke.DataAspect(),
        titlesize = 28, subtitlesize = 18, xlabelsize = 18, ylabelsize = 18,
    )
    hm = Mke.heatmap!(
        ax, xs, ys, Float32.(slice);
        colormap = Mke.cgrad(:Paired_12; categorical = true), colorrange = (0.5, 12.5),
    )
    Mke.Colorbar(
        fig[1, 2], hm; ticks = (1:12, [QRM_LABEL_NAMES[k] for k in 1:12]),
        width = 14, ticklabelsize = 15,
    )
    fig
end

# ╔═╡ 07020001-0000-4000-8000-000000000062
let
    slice = mask_3d[:, :, QRM_NZ ÷ 2 + 1]
    px_mm = QRM_VOXEL_SIZE_CM[1] * 10
    body = findall(!=(UInt8(1)), slice)
    i_lo, i_hi = extrema(c -> c[1], body)
    j_lo, j_hi = extrema(c -> c[2], body)
    w, h = i_hi - i_lo + 1, j_hi - j_lo + 1
    counts = [count(==(UInt8(k)), slice) for k in 1:12]
    rows = join(
        ["| $(QRM_LABEL_NAMES[k]) | $(counts[k]) | $(round(counts[k] * px_mm^2 / 100, digits = 2)) |"
         for k in 1:12 if counts[k] > 0],
        "\n",
    )
    Markdown.parse("""
    The body measures $(round(w * px_mm / 10, digits = 1)) × $(round(h * px_mm / 10, digits = 1)) cm
    (the QRM-Thorax-small envelope without the fat ring, ≈ 30 × 20 cm).

    | label | voxels in the mid-slice | area (cm²) |
    |-------|------:|------:|
    $(rows)
    """)
end

# ╔═╡ 07030001-0000-4000-8000-000000000001
md"""
### 2. `EICTScanner`: GE Revolution Apex Elite

The basis-spectral-denoising model of the Apex Elite, value for value, as in notebook 03:
the geometry (834 columns × 0.6 mm on an arc with the quarter-detector offset, 256 rows ×
0.625 mm), the Gemstone scintillator (`:lumex`) and the GE large-body bowtie.
"""

# ╔═╡ 07030001-0000-4000-8000-000000000010
scanner = BS.EICTScanner(
    source_to_isocenter = 625.6,
    source_to_detector = 1100.0,

    detector_rows = 256,
    detector_cols = 834,
    detector_row_size = 0.625,
    detector_col_size = 0.6,
    detector_col_offset = 0.25,
    detector_shape = :arc,

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
### 3. Dual-kVp Protocols (Rapid kVp Switching)

Rapid kVp switching alternates the tube between 80 and 140 kVp within every view period, on
one tube and one detector, so both channels see the same rays. Each energy is read for its
**duty cycle** of the view period, 0.65 at 80 kVp and 0.35 at 140 kVp: its tube current is the
instantaneous current (407 and 405 mA on the physical scan) times its duty cycle, and each of
its views integrates over that fraction of the arc the gantry turns per view (`view_arc`).
The acquisition is modelled as two co-registered acquisitions on one geometry, one per kVp,
each with 984 views over a 0.5 s rotation, 5 mm collimation and 4.5 mm of added aluminium.

The simulator's tube output per mA is generic, not GE's, so both currents are scaled by one
factor, `DOSE_SCALE`, that makes the simulated CTDIvol (32 cm body phantom, summed over the
exposures) the 10.07 mGy of the physical GE scan basis-spectral-denoising modelled. The
saved grid is 3 slices of 0.625 mm; the workspace adds the symmetric detector rows those
slices need across the full field of view.
"""

# ╔═╡ 07030002-0000-4000-8000-000000000010
# The two exposures of one rotation; each has its own noise seed, so the two channels carry
# independent quantum noise.
exposures = (
    low = (label = "80 kVp", kVp = 80, mA = 407 * 0.65, view_arc = 0.65, seed = 1234,
           filters = [("Al", 4.5)]),
    high = (label = "140 kVp", kVp = 140, mA = 405 * 0.35, view_arc = 0.35, seed = 4321,
            filters = [("Al", 4.5)]),
);

# ╔═╡ 07030002-0000-4000-8000-000000000011
begin
    "The `CTProtocol` of exposure `e`, its tube current scaled by `mA_scale`."
    exposure_protocol(e; mA_scale = 1.0) = BS.CTProtocol(
        kVp = e.kVp,
        mA = e.mA * mA_scale,
        views = 984,
        rotation_time = 0.5,
        collimation_mm = 5.0,
        additional_filters = e.filters,
    )
    "The CTDIvol (mGy, 32 cm body phantom) of `protocol` on `scanner`."
    ctdi_vol(protocol) = BS.compute_dose(BS.dose_source(scanner, protocol), protocol).ctdi_vol_mGy
end;

# ╔═╡ 07030002-0000-4000-8000-000000000012
# One factor on both tube currents, so the simulated CTDIvol is the physical scan's 10.07 mGy.
DOSE_SCALE = 10.07 / sum(e -> ctdi_vol(exposure_protocol(e)), exposures)

# ╔═╡ 07030002-0000-4000-8000-000000000020
protocols = map(e -> exposure_protocol(e; mA_scale = DOSE_SCALE), exposures);

# ╔═╡ 07030002-0000-4000-8000-000000000021
let
    rows = map(collect(keys(exposures))) do k
        e, p = exposures[k], protocols[k]
        "| $(e.label) | $(round(e.mA / e.view_arc, digits = 1)) | $(e.view_arc) | " *
        "$(round(p.mA, digits = 1)) | $(round(ctdi_vol(p), digits = 2)) |"
    end
    Markdown.parse("""
    `DOSE_SCALE` = $(round(DOSE_SCALE, digits = 4)). The simulated exposures:

    | exposure | physical mA | duty cycle, `view_arc` | simulated mA | CTDIvol (mGy) |
    |---|---:|---:|---:|---:|
    $(join(rows, "\n"))
    """)
end

# ╔═╡ 07030003-0000-4000-8000-000000000001
md"""
### 4. `SimOptions` and `ReconOptions`

The physics switches of the basis-spectral-denoising GE model: every effect of the
energy-integrating path on except optical crosstalk, and the detector efficiency from the
Monte Carlo table of the Lumex scintillator (`detector_efficiency_mode = :mc_lut`).

**View integration.** A detector integrates while the gantry turns, so each view records the
transmitted intensity averaged over the arc it sweeps, `r · view_arc · Δθ` at radius `r`: a
blur of the object, largest far from the isocentre, that the noise, counted once per view,
does not share. `view_samples = 5` samples that arc with five sub-views. Without it the
simulated signal is sharper, relative to its noise, than a physical scanner's.

**Draws.** Each exposure is simulated three times, differing only in the noise: the
noise-free expectation (draw 0, `use_noise = false`), the measured scan (draw 1) and a
calibration scan (draw 9) of its own seeds. The reconstruction grid is 512 × 512 × 3 at
0.625 mm, independent of the 0.2 mm phantom grid; both share the isocenter.
"""

# ╔═╡ 07030003-0000-4000-8000-000000000010
begin
    VIEW_SAMPLES = 5
    ge_opts = BS.SimOptions(
        projector = :dd_fast,
        detector_efficiency_mode = :mc_lut,
        view_samples = VIEW_SAMPLES,
    )
    "The same physics switches with one field replaced."
    with_option(opts, field, value) = BS.SimOptions(;
        (f => getfield(opts, f) for f in fieldnames(BS.SimOptions) if f !== field)...,
        (field => value,)...,
    )
end;

# ╔═╡ 07030003-0000-4000-8000-000000000011
begin
    # Draw 0 is the noise-free expectation; draw r ≥ 1 gives each exposure the seed
    # `seed + 1000 (r − 1)`, so the exposures stay independent within a draw and the draws of
    # each other.
    DRAWS = (reference = 0, measured = 1, calibration = 9)
    realization_seed(e, r) = e.seed + 1000 * (max(r, 1) - 1)
    "The options of draw `r` of exposure `e`: its seed, its noise, its arc."
    draw_opts(e, r) = with_option(with_option(with_option(ge_opts,
        :seed, realization_seed(e, r)), :use_noise, r > 0), :view_arc, e.view_arc)
end;

# ╔═╡ 07030003-0000-4000-8000-000000000020
recon_opts = BS.ReconOptions(
    matrix_size = (512, 512, 3),
    fov_cm = 32.0,
    z_cm = 0.1875,
);

# ╔═╡ 07030004-0000-4000-8000-000000000001
md"""
### 5. Forward Project: `simulate!`

Each exposure is projected once. The noise-free draw runs first with
`keep_projection = true`, which returns everything before the noise; the measured and
calibration draws pass it back (`projection`) and only draw their noise, bit-identically to
simulating each afresh. Each acquisition keeps what the spectral basis needs: its corrected
log sinograms, its per-ray air counts `I0_ray` (the detector's air count × the bowtie air
profile) and its per-ray detected spectrum from `resolve_source_spectrum_full` (source ×
filtration × bowtie × detector response), the same model the forward projector applied.
"""

# ╔═╡ 07030004-0000-4000-8000-000000000010
dual_scans = map(collect(keys(exposures))) do k
    e, protocol = exposures[k], protocols[k]
    @info "Simulating dual-kVp channel: $(e.label)"
    channels = Dict{Int, Array{Float32, 3}}()
    projection = nothing
    model = nothing
    elapsed_s = @elapsed for r in (DRAWS.reference, DRAWS.measured, DRAWS.calibration)
        opts = draw_opts(e, r)
        ws = BS.create_workspace(scanner, protocol, opts, recon_opts, phantom)
        try
            result = BS.simulate!(ws, phantom, protocol, opts; report_dose = false,
                keep_projection = projection === nothing, projection)
            projection === nothing && (projection = result.projection)
            channels[r] = Float32.(Array(ws.sinogram))
            if model === nothing
                air = ws.bowtie_air_reference === nothing ?
                    ones(Float32, ws.geom.n_cols, ws.geom.n_rows) :
                    Float32.(Array(ws.bowtie_air_reference))
                I0 = BS.compute_detector_I0(ws.geom, protocol, sum(ws.weights)) * Float64(ws.η_eff)
                energies, response = BS.resolve_source_spectrum_full(
                    opts, protocol; scanner = scanner, geom = ws.geom,
                )
                model = (
                    geom = ws.geom, I0_ray = Float32.(I0 .* air),
                    energies = Float64.(energies), response = Float32.(response),
                )
            end
        finally
            BS.release_backend!(ws)
        end
    end
    (; label = e.label, channels, elapsed_s, model...)
end;

# ╔═╡ 07030004-0000-4000-8000-000000000011
Markdown.parse("""
The three draws of the two exposures took
$(join(["$(round(s.elapsed_s, digits = 1)) s for $(s.label)" for s in dual_scans], " and ")),
one projection each. Each sinogram is $(join(size(first(dual_scans).channels[DRAWS.measured]), " × "))
(columns × rows × views).
""")

# ╔═╡ 07030004-0000-4000-8000-000000000025
# The phantom labels resampled onto the reconstruction grid (nearest neighbour); every ROI
# below is built from these labels in recon-voxel coordinates.
phantom_in_recon = BS.resample_to_recon(
    phantom_cpu, first(dual_scans).geom, recon_opts.matrix_size; method = :nearest,
);

# ╔═╡ 07030004-0000-4000-8000-000000000030
let
    sinos = [s.channels[DRAWS.measured] for s in dual_scans]
    mid_r = size(first(sinos), 2) ÷ 2 + 1
    slices = [permutedims(s[:, mid_r, :], (2, 1)) for s in sinos]
    sino_window = Tuple(Float64.(quantile(vcat(vec.(slices)...), (0.01, 0.99))))

    fig = Mke.Figure(size = (1180, 580))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "View", ylabel = "Detector Column",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )
    for (c, (s, slice)) in enumerate(zip(dual_scans, slices))
        ax = Mke.Axis(fig[1, c]; title = s.label, subtitle = "central detector row", axis_kwargs...)
        Mke.heatmap!(ax, slice; colormap = :viridis, colorrange = sino_window)
    end
    Mke.Colorbar(
        fig[1, 3]; colormap = :viridis, colorrange = sino_window,
        label = "Log Line Integral", width = 16, labelsize = 22, ticklabelsize = 18,
    )
    fig
end

# ╔═╡ 07030000-0000-4000-8000-0000000000f1
md"""
## VMI Pipeline

### 1. Spectral Basis from the Two Acquisitions

`spectral_basis_from_acquisitions` merges the two kVp energy grids onto their union and
scales each acquisition's per-ray spectrum by its own air counts, so the likelihood sees the
absolute response ``\Phi_k(E)`` of every ray and channel: source, filtration, bowtie and
detector, the model that generated the data. It does not depend on the noise, so every draw
shares it.
"""

# ╔═╡ 07030006-0000-4000-8000-000000000010
basis = BS.spectral_basis_from_acquisitions(acquisitions = [
    (energies = s.energies, response = s.response, I0_ray = s.I0_ray) for s in dual_scans
]);

# ╔═╡ 07030006-0000-4000-8000-000000000011
Markdown.parse("""
The basis holds $(basis.n_channels) channels on a $(length(basis.E))-point energy grid
for $(size(basis.Φ, 1)) × $(size(basis.Φ, 2)) rays, spanning
$(round(Int, minimum(basis.E))) to $(round(Int, maximum(basis.E))) keV; the response sums
to the air counts to within $(round(basis.I0_relerr, sigdigits = 2)) (relative).
""")

# ╔═╡ 07030006-0000-4000-8000-000000000001
md"""
### 2. The Chain's Settings

**The denoiser** is generalized HYPR-LR in both domains, `BS.SpectralHYPR()`: before the
decomposition, a 3 × 3 (column × view) window on the counts of each detector row, pooling
views of one parity so the odd- and even-view halves stay independent, with each channel's
dispersion measured on the air rays; after the reconstruction, the complement of the basis
pair pooled within each slice over the window with the least estimated risk. Neither pools
across detector rows or slices. The decomposition is the K-channel maximum-likelihood
estimator with its published controls; T-LBF (a photon-counting filter) is off; the
view-direction antialias is on; ACNR is on (`vmi_pipeline`'s default).

**A pair of FDK windows.** A VMI is ``A(E)\,M + B(E)\,I_\perp``, with ``M`` the
minimum-noise VMI (the composite) and ``I_\perp`` its complement, whose noise is uncorrelated
with ``M``'s. `BS.PairFilter` gives each its own window; one pair serves every energy. A
window per basis image instead would stop the anti-correlated noise of the water and iodine
images cancelling wherever the two windows differ. The windows are the GE pair
basis-spectral-denoising fitted to the physical Gammex scan, ``W(f) = \exp(-(f/f_c)^p)``.

**Rescaled to this grid.** A window is defined on the frequency axis of the grid it is
applied on, `[0, grid_bandlimit]` of the detector's Nyquist. The pair was fitted on the GE's
clinical grid, 512² over 35 cm; this grid, 512² over 32 cm, has finer pixels and carries
higher frequencies, so the windows are the same functions of physical frequency rescaled onto
its axis, ``\exp(-(f\,r/f_c)^p)`` with ``r`` the ratio of the two grids' bandlimits, as
basis-spectral-denoising's `RECON_BY` does for its XCAT grid.
"""

# ╔═╡ 07030006-0000-4000-8000-000000000012
# the published denoiser: ProjectionHYPR(3 × 3 box, view_stride = 2) and ImageHYPR()
HYPR_CHAIN = BS.SpectralHYPR();

# ╔═╡ 07030006-0000-4000-8000-000000000013
VMI_CHAIN = (method = :nchannel, controls = BS.NChannelControls(), use_tlbf = false, antialias = true);

# ╔═╡ 07030006-0000-4000-8000-000000000014
# The GE pair of basis-spectral-denoising (`RECON.ge`): W(f) = exp(-(f / f_c)^p) at 11 knots.
GE_KERNELS = (composite = (fc = 0.55, p = 1.5), complement = (fc = 1.05, p = 1.0));

# ╔═╡ 07030006-0000-4000-8000-000000000015
# The ratio of this grid's bandlimit to that of the grid the pair was fitted on: the GE's
# clinical series, 512² × 8 slices over 35 cm.
BANDLIMIT_RATIO = let
    fitted_geom = BS.CTGeometry(scanner; n_angles = 984, fov_cm = 35.0, z_cm = 0.5, collimation_mm = 5.0)
    BS.grid_bandlimit(first(dual_scans).geom, recon_opts.matrix_size) /
        BS.grid_bandlimit(fitted_geom, (512, 512, 8))
end

# ╔═╡ 07030006-0000-4000-8000-000000000016
PAIR_FILTER = let knots = Tuple(range(0.0, 1.0, length = 11)), r = BANDLIMIT_RATIO
    window(q) = BS.CustomFilter(knots, Tuple(round(exp(-(x * r / q.fc)^q.p), digits = 5) for x in knots))
    BS.PairFilter(window(GE_KERNELS.composite), window(GE_KERNELS.complement))
end;

# ╔═╡ 07030009-0000-4000-8000-000000000010
de_vmi_energies = [50.0, 70.0, 100.0, 140.0];

# ╔═╡ 07030006-0000-4000-8000-000000000020
md"""
### 3. The Calibration Draw: `pair_basis` and `composite_energy`

The pair the windows act on, ``(E^\star, \beta)``, and the energy of the composite that ACNR
and the image-domain HYPR act on are properties of the scanner and protocol, measured once on
the calibration draw and fixed for every other draw, as basis-spectral-denoising's
`PAIR_BASIS` does: the calibration draw decomposed as measured and after the
projection-domain HYPR alone; `BS.spectral_pair` of the first with a standard soft-tissue
window gives ``E^\star`` and ``\beta`` (`pair_basis`); `BS.spectral_pair` of the second on
that basis gives the minimum-noise energy after the projection-domain HYPR
(`composite_energy`). Near its minimum the VMI noise hardly changes with energy, so the argmin
of one acquisition is itself noise; measured on a draw of its own it cannot tune the chain to
the draw being scored, and the noise-free reference is reconstructed with the same pair.
"""

# ╔═╡ 07030006-0000-4000-8000-000000000021
calibration = let
    geom = first(dual_scans).geom
    matrix_size = recon_opts.matrix_size
    channels = [s.channels[DRAWS.calibration] for s in dual_scans]
    decompose(denoiser) = BS.vmi_pipeline(;
        channels, basis, geom, to_backend = to_gpu, matrix_size, denoiser,
        keep_sinograms = true, use_acnr = false, VMI_CHAIN...,
    ).sinograms
    pair(d; kw...) = BS.spectral_pair(d.water, d.iodine, geom, matrix_size;
        filter = BS.SoftFilter(), antialias = VMI_CHAIN.antialias, to_backend = to_gpu, kw...)
    x = pair(decompose(nothing))
    c = pair(decompose(BS.SpectralHYPR(image = nothing)); basis = (Estar = x.Estar, β = x.β))
    (pair_basis = (Estar = x.Estar, β = x.β), composite_energy = c.Estar)
end

# ╔═╡ 07030007-0000-4000-8000-000000000001
md"""
### 4. `vmi_pipeline`

One call, from the two measured sinograms to the VMI stack, reconstructed on the
notebook's grid with every detector row kept. `keep_sinograms = true` also returns the
decomposed basis sinograms, shown below. The noise-free reference is the noise-free draw
through the same reconstruction, windows and pair, with nothing to denoise (no HYPR, no ACNR).
"""

# ╔═╡ 07030007-0000-4000-8000-000000000010
dual_vmi = BS.vmi_pipeline(;
    channels = [s.channels[DRAWS.measured] for s in dual_scans],
    basis,
    geom = first(dual_scans).geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = Tuple(de_vmi_energies),
    denoiser = HYPR_CHAIN,
    fbp_filter = PAIR_FILTER,
    pair_basis = calibration.pair_basis,
    composite_energy = calibration.composite_energy,
    keep_sinograms = true,
    VMI_CHAIN...,
);

# ╔═╡ 07030007-0000-4000-8000-000000000012
reference_vmi = BS.vmi_pipeline(;
    channels = [s.channels[DRAWS.reference] for s in dual_scans],
    basis,
    geom = first(dual_scans).geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = Tuple(de_vmi_energies),
    fbp_filter = PAIR_FILTER,
    pair_basis = calibration.pair_basis,
    composite_energy = calibration.composite_energy,
    use_acnr = false,
    VMI_CHAIN...,
);

# ╔═╡ 07030007-0000-4000-8000-000000000011
let
    q = dual_vmi.quality
    d = dual_vmi.settings.denoiser
    b = calibration.pair_basis
    pct(x) = round(100x, digits = 3)
    Markdown.parse("""
    The windows were rescaled by r = $(round(BANDLIMIT_RATIO, digits = 3)). The calibration draw fixed
    their pair at E* = $(round(Int, b.Estar)) keV, β = $(round(b.β, sigdigits = 3)), and the composite at
    $(round(Int, calibration.composite_energy)) keV. On the measured draw the decomposition solved
    $(q.n_rays) rays in $(round(dual_vmi.elapsed_s, digits = 1)) s, $(round(q.outer_mean, digits = 1)) outer
    iterations on average; $(pct(q.frac_not_converged))% did not converge and
    $(pct(q.frac_bound_iodine))% / $(pct(q.frac_bound_water))% touched the iodine / water bounds.
    The projection HYPR measured dispersions, the variance-to-mean ratio of the counts on the air rays, of
    $(join(round.(d.dispersion, digits = 2), " and ")) for 80 and 140 kVp. The image HYPR pooled the
    complement over a $(d.image_estimates.window) × $(d.image_estimates.window) window.
    """)
end

# ╔═╡ 07030006-0000-4000-8000-000000000040
let
    mid_r = size(dual_vmi.sinograms.iodine, 2) ÷ 2 + 1
    fig = Mke.Figure(size = (1400, 580))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "View", ylabel = "Detector Column",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )
    panels = (
        ("Iodine Basis Sinogram", dual_vmi.sinograms.iodine),
        ("Water Basis Sinogram", dual_vmi.sinograms.water),
    )
    for (c, (ttl, sino)) in enumerate(panels)
        slice = permutedims(sino[:, mid_r, :], (2, 1))
        range = Tuple(Float64.(quantile(vec(slice), (0.01, 0.99))))
        ax = Mke.Axis(fig[1, 2c - 1]; title = ttl, subtitle = "central detector row", axis_kwargs...)
        Mke.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        Mke.Colorbar(
            fig[1, 2c]; colormap = :viridis, colorrange = range,
            label = "g/cm²", width = 16, labelsize = 22, ticklabelsize = 18,
        )
    end
    fig
end

# ╔═╡ 07030008-0000-4000-8000-000000000001
md"""
### 5. Basis Maps and VMIs

The water and iodine basis pair after ACNR and image HYPR (mid slice), and the VMIs
synthesized from it: ``\mu(E) = c_\mathrm{water}\,(\mu/\rho)_\mathrm{water}(E) +
c_\mathrm{iodine}\,(\mu/\rho)_\mathrm{iodine}(E)``, in HU relative to water at the same
energy. Every VMI comes from the same basis pair.
"""

# ╔═╡ 07030008-0000-4000-8000-000000000030
let
    fig = Mke.Figure(size = (1180, 580))
    mid = size(dual_vmi.images.water, 3) ÷ 2 + 1
    # iodine in mg/mL on a fixed window around the 5 mg/mL rod (a percentile window of the
    # whole slice sits below the rod, which covers under 1 % of it); water on its percentiles
    iodine_mg = 1000 .* dual_vmi.images.iodine[:, :, mid]
    water = dual_vmi.images.water[:, :, mid]
    panels = (
        ("Iodine Basis", iodine_mg, (-2.0, 8.0), "mg/mL"),
        ("Water Basis", water, Tuple(Float64.(quantile(vec(water), (0.01, 0.99)))), "g/mL"),
    )
    for (c, (ttl, slice, range, unit)) in enumerate(panels)
        ax = Mke.Axis(fig[1, 2c - 1]; title = ttl, aspect = Mke.DataAspect(), titlesize = 32)
        Mke.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        Mke.hidedecorations!(ax)
        Mke.Colorbar(
            fig[1, 2c]; colormap = :viridis, colorrange = range,
            label = unit, width = 16, labelsize = 22, ticklabelsize = 18,
        )
    end
    fig
end

# ╔═╡ 07030009-0000-4000-8000-000000000020
# The VMI stacks keyed by energy, (512, 512, 3) HU volumes: measured and noise-free.
begin
    vmi_HU_final = Dict(
        Float64(E) => dual_vmi.vmis[:, :, :, k] for (k, E) in enumerate(dual_vmi.energies)
    )
    vmi_HU_reference = Dict(
        Float64(E) => reference_vmi.vmis[:, :, :, k] for (k, E) in enumerate(reference_vmi.energies)
    )
end;

# ╔═╡ 07030009-0000-4000-8000-000000000040
let
    HU_window = (-200, 500)
    fig = Mke.Figure(size = (1180, 1180))
    mid = size(dual_vmi.vmis, 3) ÷ 2 + 1
    for (k, E) in enumerate(de_vmi_energies)
        ax = Mke.Axis(
            fig[(k - 1) ÷ 2 + 1, (k - 1) % 2 + 1]; title = "$(Int(E)) keV VMI",
            aspect = Mke.DataAspect(), titlesize = 32,
        )
        Mke.heatmap!(ax, vmi_HU_final[E][:, :, mid]; colormap = :grays, colorrange = HU_window)
        Mke.hidedecorations!(ax)
    end
    Mke.Colorbar(
        fig[1:2, 3]; colormap = :grays, colorrange = HU_window,
        label = "HU", width = 16, labelsize = 22, ticklabelsize = 18,
    )
    fig
end

# ╔═╡ 070b0001-0000-4000-8000-000000000001
md"""
## Results

Per-rod measured, noise-free and theoretical HU at 50 / 70 / 100 / 140 keV, the water rod's
mean HU, and the noise in the heart.

- **Measured HU**: the mean over a circular ROI at each rod's centroid, over every slice, of
  the measured VMIs.
- **Noise-free HU**: the same mean on the noise-free reference.
- **Theoretical HU**: ``1000\,(\mu_\mathrm{rod}(E) - \mu_\mathrm{water}(E)) / \mu_\mathrm{water}(E)``
  from `BS.compute_μ_at_energy(material, E)`; pure physics, no fitting.
- **Noise**: the standard deviation of the measured VMI minus the noise-free reference, the
  noise alone.
"""

# ╔═╡ 070b0001-0000-4000-8000-000000000010
const ROD_LABELS = (UInt8(9), UInt8(10), UInt8(11), UInt8(12));

# ╔═╡ 070b0001-0000-4000-8000-000000000011
const ROD_NAMES = ("Water", "Lipid", "Collagen", "Iodine 5 mg/mL");

# ╔═╡ 070b0001-0000-4000-8000-000000000012
const ROD_MATERIALS = (
    BS.XA.Materials.basis_water,
    BS.XA.Materials.basis_lipid,
    BS.XA.Materials.basis_collagen,
    BS.XA.Materials.gammex_472_i5_0,
);

# ╔═╡ 070b0001-0000-4000-8000-000000000013
const ROD_ROI_RADIUS_PX = 8;    # 5 mm at 0.625 mm/voxel, inside the 7.5 mm rods

# ╔═╡ 070b0001-0000-4000-8000-000000000014
const HEART_NOISE_ROI_RADIUS_PX = 12;    # 7.5 mm: fits between the four rods

# ╔═╡ 070b0001-0000-4000-8000-000000000015
begin
    "Every value of `vol` inside the in-plane `roi`, over all slices."
    roi_values(vol, roi) = Float64[vol[c, z] for z in axes(vol, 3) for c in roi]

    # Rod centroids from the phantom labels on the recon grid; the noise ROI sits at the
    # centre of the rod cross, the mean of the four rod centroids.
    rod_rois = let
        mask_2d = phantom_in_recon[:, :, size(phantom_in_recon, 3) ÷ 2 + 1]
        nx, ny = size(mask_2d)
        function centroid(label)
            idx = findall(==(label), mask_2d)
            isempty(idx) && error("no recon voxel carries label $(label)")
            (sum(c -> c[1], idx) / length(idx), sum(c -> c[2], idx) / length(idx))
        end
        disc(c, r) = [CartesianIndex(i, j) for j in 1:ny for i in 1:nx
                      if (i - c[1])^2 + (j - c[2])^2 ≤ r^2]
        centers = Dict(lab => centroid(lab) for lab in ROD_LABELS)
        heart_center = (
            mean(first(centers[l]) for l in ROD_LABELS),
            mean(last(centers[l]) for l in ROD_LABELS),
        )
        (
            centers = centers,
            rods = Dict(lab => disc(centers[lab], ROD_ROI_RADIUS_PX) for lab in ROD_LABELS),
            heart_center = heart_center,
            heart = disc(heart_center, HEART_NOISE_ROI_RADIUS_PX),
        )
    end
end;

# ╔═╡ 070b0001-0000-4000-8000-000000000000
md"""
### Phantom-Recon Alignment Verification

Before trusting any ROI built on `phantom_in_recon`: the 70 keV VMI and the resampled
phantom labels side by side on the same recon grid, then overlaid, so the rod, heart and
bone edges can be checked against the image.
"""

# ╔═╡ 070b0001-0000-4000-8000-00000000000a
let
    z = size(vmi_HU_final[70.0], 3) ÷ 2 + 1
    hu_slice = vmi_HU_final[70.0][:, :, z]
    pir_slice = phantom_in_recon[:, :, clamp(z, 1, size(phantom_in_recon, 3))]
    rod_overlay = Float32[v in ROD_LABELS ? Float32(v) : NaN32 for v in pir_slice]
    # label 0 = outside the phantom grid (the 32 cm FOV is taller than the 22 cm phantom)
    all_labels = Float32[v == 0 ? NaN32 : Float32(v) for v in pir_slice]
    full_overlay = Float32[v ≤ 1 ? NaN32 : Float32(v) for v in pir_slice]
    labels_cmap = Mke.cgrad(:Paired_12; categorical = true)

    fig = Mke.Figure(size = (1400, 1320))
    hu_kwargs = (colormap = :grays, colorrange = (-200, 500))
    title_kwargs = (titlesize = 28, subtitlesize = 20, aspect = Mke.DataAspect())

    ax = Mke.Axis(fig[1, 1]; title = "70 keV VMI", subtitle = "slice $(z) of $(size(vmi_HU_final[70.0], 3))", title_kwargs...)
    Mke.heatmap!(ax, hu_slice; hu_kwargs...)
    Mke.hidedecorations!(ax)

    ax = Mke.Axis(fig[1, 2]; title = "phantom_in_recon (nearest)", subtitle = "$(count(>(0), unique(pir_slice))) labels on the recon grid; white: outside the phantom grid", title_kwargs...)
    Mke.heatmap!(ax, all_labels; colormap = labels_cmap, colorrange = (0.5, 12.5), nan_color = :white)
    Mke.hidedecorations!(ax)

    ax = Mke.Axis(fig[2, 1]; title = "VMI + rod labels (9–12)", subtitle = "rod edges should sit on the image's rod edges", title_kwargs...)
    Mke.heatmap!(ax, hu_slice; hu_kwargs...)
    Mke.heatmap!(ax, rod_overlay; colormap = labels_cmap, colorrange = (0.5, 12.5), alpha = 0.6, nan_color = (:white, 0.0))
    Mke.hidedecorations!(ax)

    ax = Mke.Axis(fig[2, 2]; title = "VMI + every non-air label", subtitle = "bone, heart and lung edges", title_kwargs...)
    Mke.heatmap!(ax, hu_slice; hu_kwargs...)
    Mke.heatmap!(ax, full_overlay; colormap = labels_cmap, colorrange = (0.5, 12.5), alpha = 0.4, nan_color = (:white, 0.0))
    Mke.hidedecorations!(ax)
    fig
end

# ╔═╡ 070b0001-0000-4000-8000-000000000002
md"""
### Water ROI

The water-rod ROI (label 9, `basis_water`) in red on the 70 keV VMI, and its mean HU at every
VMI energy, measured and noise-free. Water is the reference of the HU scale, so the bars
should sit at 0 HU; a constant offset would be a basis-decomposition bias, a trend with
energy a spectral-model error, and a difference between the two bars what the noise and the
denoising add.
"""

# ╔═╡ 070b0001-0000-4000-8000-000000000004
begin
    water_rod_hu = [mean(roi_values(vmi_HU_final[E], rod_rois.rods[UInt8(9)])) for E in de_vmi_energies]
    water_rod_hu_reference = [mean(roi_values(vmi_HU_reference[E], rod_rois.rods[UInt8(9)])) for E in de_vmi_energies]
end;

# ╔═╡ 070b0001-0000-4000-8000-000000000003
let
    HU_window = (-200, 500)
    mid = size(vmi_HU_final[70.0], 3) ÷ 2 + 1
    bg = vmi_HU_final[70.0][:, :, mid]
    overlay = fill(NaN32, size(bg))
    overlay[rod_rois.rods[UInt8(9)]] .= 1.0f0

    fig = Mke.Figure(size = (1180, 580))
    ax1 = Mke.Axis(
        fig[1, 1]; title = "Water-Rod ROI", subtitle = "on the 70 keV VMI",
        aspect = Mke.DataAspect(), titlesize = 32, subtitlesize = 24,
    )
    Mke.heatmap!(ax1, bg; colormap = :grays, colorrange = HU_window)
    Mke.heatmap!(ax1, overlay; colormap = :reds, alpha = 0.5, nan_color = (:white, 0.0))
    Mke.hidedecorations!(ax1)

    n_E = length(de_vmi_energies)
    ax2 = Mke.Axis(
        fig[1, 2]; title = "Water ROI Mean HU", subtitle = "per VMI energy",
        xlabel = "VMI Energy (keV)", ylabel = "HU",
        xticks = (collect(1:n_E), ["$(Int(E))" for E in de_vmi_energies]),
        titlesize = 32, subtitlesize = 24, xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    values = vcat(water_rod_hu, water_rod_hu_reference)
    group = vcat(fill(1, n_E), fill(2, n_E))
    Mke.barplot!(
        ax2, vcat(1:n_E, 1:n_E), values; dodge = group,
        color = [(:tomato, :gray60)[g] for g in group],
        strokecolor = :black, strokewidth = 1,
    )
    Mke.hlines!(ax2, [0.0]; color = :black, linewidth = 1, linestyle = :dash)
    for (k, h) in pairs(water_rod_hu)
        Mke.text!(
            ax2, k - 0.2, h; text = "$(round(h, digits = 1))",
            align = (:center, h ≥ 0 ? :bottom : :top), fontsize = 15, offset = (0, h ≥ 0 ? 4 : -4),
        )
    end
    Mke.axislegend(ax2,
        [Mke.PolyElement(color = :tomato), Mke.PolyElement(color = :gray60)],
        ["Measured", "Noise-free reference"]; position = :lb, labelsize = 16)
    y_max = max(15.0, 1.2 * maximum(abs, values))
    Mke.ylims!(ax2, -y_max, y_max)
    Mke.save(joinpath(@__DIR__, "..", "assets", "qrm_thorax_vmi_water_roi_check.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 070d0001-0000-4000-8000-000000000001
md"""
### Heart-Centre Noise ROI

The noise in a 7.5 mm ROI in the soft-tissue heart, at the centre of the four rods, at every
VMI energy: the standard deviation of the measured VMI about the noise-free reference, how
the chain carries noise from the basis pair into each VMI.
"""

# ╔═╡ 070d0001-0000-4000-8000-000000000020
vmi_noise_by_keV = Dict(
    E => let v = roi_values(vmi_HU_final[E], rod_rois.heart),
             v0 = roi_values(vmi_HU_reference[E], rod_rois.heart)
        (mean = mean(v), std = std(v .- v0), n = length(v))
    end
    for E in de_vmi_energies
);

# ╔═╡ 070d0001-0000-4000-8000-000000000015
# The basis-pair noise covariance in the heart ROI (measured minus noise-free), which sets the
# noise of every VMI:
#   σ_HU(E)² = 10⁶ V_w + α(E)² V_i + 2·10³ α(E) C_iw,  α(E) = (μ/ρ)_I(E) / (μ/ρ)_w(E),
# with water in g/mL and iodine in mg/mL; lowest at α* = -10³ C_iw / V_i.
basis_covariance = let
    w = roi_values(dual_vmi.images.water, rod_rois.heart) .-
        roi_values(reference_vmi.images.water, rod_rois.heart)
    i = (roi_values(dual_vmi.images.iodine, rod_rois.heart) .-
         roi_values(reference_vmi.images.iodine, rod_rois.heart)) .* 1000.0   # mg/mL
    Vw, Vi = mean(abs2, w .- mean(w)), mean(abs2, i .- mean(i))
    Ciw = mean((w .- mean(w)) .* (i .- mean(i)))
    α(E) = BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, E) /
        BS.compute_mass_μ_at_energy(BS.XA.Materials.water, E)
    α_star = -1000.0 * Ciw / Vi
    E_star = let grid = 40.0:1.0:140.0
        grid[argmin(abs.(α.(grid) .- α_star))]
    end
    (σ_water = sqrt(Vw), σ_iodine = sqrt(Vi), ρ = Ciw / sqrt(Vw * Vi), α_star, E_star,
     α_range = (α(140.0), α(40.0)))
end;

# ╔═╡ 070d0001-0000-4000-8000-000000000016
let c = basis_covariance
    inside = c.α_range[1] ≤ c.α_star ≤ c.α_range[2]
    Markdown.parse("""
    In the heart ROI the noise of the water basis is σ = $(round(1000 * c.σ_water, digits = 1)) mg/mL, that
    of the iodine basis σ = $(round(c.σ_iodine, digits = 2)) mg/mL, and their correlation is
    ρ = $(round(c.ρ, digits = 2)). The noise-optimal mixing ratio is α* = $(round(c.α_star, digits = 1)) HU per mg/mL""" *
    (inside ? ", reached at about $(Int(c.E_star)) keV; the chain's composite is at $(round(Int, calibration.composite_energy)) keV." :
        ", outside the 40–140 keV range, so the noise falls monotonically toward $(c.α_star < c.α_range[1] ? "high" : "low") energy."))
end

# ╔═╡ 070d0001-0000-4000-8000-000000000030
let
    HU_window = (-200, 500)
    mid = size(vmi_HU_final[70.0], 3) ÷ 2 + 1
    bg = vmi_HU_final[70.0][:, :, mid]
    overlay = fill(NaN32, size(bg))
    overlay[rod_rois.heart] .= 1.0f0

    fig = Mke.Figure(size = (1180, 580))
    ax1 = Mke.Axis(
        fig[1, 1]; title = "Heart-Centre Noise ROI", subtitle = "on the 70 keV VMI",
        aspect = Mke.DataAspect(), titlesize = 32, subtitlesize = 24,
    )
    Mke.heatmap!(ax1, bg; colormap = :grays, colorrange = HU_window)
    Mke.heatmap!(ax1, overlay; colormap = :reds, alpha = 0.5, nan_color = (:white, 0.0))
    Mke.hidedecorations!(ax1)

    Es = sort(collect(keys(vmi_noise_by_keV)))
    σs = [vmi_noise_by_keV[E].std for E in Es]
    μs = [vmi_noise_by_keV[E].mean for E in Es]
    ax2 = Mke.Axis(
        fig[1, 2]; title = "Heart-Centre Noise vs Energy",
        subtitle = "$(length(rod_rois.heart)) voxels × $(size(vmi_HU_final[70.0], 3)) slices, about the noise-free reference",
        xlabel = "VMI Energy (keV)", ylabel = "Noise σ (HU)", xticks = Es,
        titlesize = 32, subtitlesize = 18, xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    Mke.scatterlines!(ax2, Es, σs; color = :tomato, markersize = 18, linewidth = 3)
    for (E, σ, μ) in zip(Es, σs, μs)
        Mke.text!(
            ax2, E, σ; text = "σ=$(round(σ; digits = 1))\n⟨HU⟩=$(round(μ; digits = 1))",
            align = (:center, :bottom), fontsize = 16, offset = (0, 8),
        )
    end
    Mke.ylims!(ax2, 0, maximum(σs) * 1.4)
    Mke.xlims!(ax2, first(Es) - 12, last(Es) + 12)
    Mke.save(joinpath(@__DIR__, "..", "assets", "qrm_thorax_vmi_noise_roi.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 070b0001-0000-4000-8000-000000000008
md"""
### Basis Values in the Rods

The mean basis pair in each rod ROI. The water rod should read ``c_\mathrm{water} ≈ 1``
g/mL and ``c_\mathrm{iodine} ≈ 0``; the iodine rod carries its 5 mg/mL on a
water-equivalent base. Lipid and collagen are not made of water and iodine, so the chain
represents them as the water/iodine mixture that attenuates the same way (lipid with a
negative iodine value).
"""

# ╔═╡ 070b0001-0000-4000-8000-000000000009
let
    rows = map(zip(ROD_LABELS, ROD_NAMES, ROD_MATERIALS)) do (lab, name, mat)
        cw = mean(roi_values(dual_vmi.images.water, rod_rois.rods[lab]))
        ci = 1000 * mean(roi_values(dual_vmi.images.iodine, rod_rois.rods[lab]))
        "| $(name) | $(round(BS.XA.val(mat.density), digits = 3)) | $(round(cw, digits = 3)) | $(round(ci, digits = 2)) |"
    end
    Markdown.parse("""
    | rod | density (g/cm³) | water (g/mL) | iodine (mg/mL) |
    |-----|------:|------:|------:|
    $(join(rows, "\n"))
    """)
end

# ╔═╡ 070b0001-0000-4000-8000-000000000020
rod_data = let
    μ_water(E) = BS.compute_μ_at_energy(BS.XA.Materials.water, E)
    theoretical_hu(mat, E) = 1000.0 * (BS.compute_μ_at_energy(mat, E) - μ_water(E)) / μ_water(E)
    meas = [mean(roi_values(vmi_HU_final[E], rod_rois.rods[lab])) for lab in ROD_LABELS, E in de_vmi_energies]
    ref = [mean(roi_values(vmi_HU_reference[E], rod_rois.rods[lab])) for lab in ROD_LABELS, E in de_vmi_energies]
    theo = [theoretical_hu(mat, E) for mat in ROD_MATERIALS, E in de_vmi_energies]
    rmse(a) = [sqrt(mean(abs2, a[i, :] .- theo[i, :])) for i in eachindex(ROD_LABELS)]
    (labels = ROD_LABELS, names = ROD_NAMES, measured = meas, reference = ref, theoretical = theo,
     rmse = rmse(meas), rmse_reference = rmse(ref))
end;

# ╔═╡ 070b0001-0000-4000-8000-000000000030
md"""
### Per-Rod Measured vs Theoretical HU

Solid lines show the measured HU, crosses the noise-free reference and dashed lines the
theoretical HU.
"""

# ╔═╡ 070b0001-0000-4000-8000-000000000031
let
    fig = Mke.Figure(size = (1400, 580))
    rod_colors = [
        Mke.RGBf(0.2, 0.6, 0.85),    # water    — blue
        Mke.RGBf(0.95, 0.65, 0.13),  # lipid    — orange
        Mke.RGBf(0.55, 0.3, 0.65),   # collagen — purple
        Mke.RGBf(0.85, 0.27, 0.1),   # iodine   — red
    ]
    ax = Mke.Axis(
        fig[1, 1]; title = "Pure-Material Rods (dual-kVp)", subtitle = "50 / 70 / 100 / 140 keV",
        xlabel = "VMI energy (keV)", ylabel = "HU", xticks = de_vmi_energies,
        titlesize = 32, subtitlesize = 24, xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    rod_lines = Any[]
    rod_labels_str = String[]
    for i in eachindex(rod_data.names)
        c = rod_colors[i]
        Mke.scatterlines!(ax, de_vmi_energies, rod_data.measured[i, :]; color = c, linewidth = 2.5, markersize = 9)
        Mke.scatter!(ax, de_vmi_energies, rod_data.reference[i, :]; color = :black, marker = :xcross, markersize = 10)
        Mke.lines!(ax, de_vmi_energies, rod_data.theoretical[i, :]; color = c, linewidth = 1.6, linestyle = :dash)
        push!(rod_lines, Mke.LineElement(color = c, linewidth = 2.5))
        push!(rod_labels_str, "$(rod_data.names[i]) (RMSE = $(round(rod_data.rmse[i], digits = 1)) HU)")
    end
    Mke.Legend(
        fig[1, 2],
        vcat([Mke.MarkerElement(color = :black, marker = :circle, markersize = 9),
              Mke.MarkerElement(color = :black, marker = :xcross, markersize = 10),
              Mke.LineElement(color = :black, linewidth = 1.6, linestyle = :dash)], rod_lines),
        vcat(["Measured", "Noise-free reference", "Theoretical"], rod_labels_str);
        framevisible = true, labelsize = 18, rowgap = 1, padding = (6, 6, 6, 6),
    )
    Mke.save(joinpath(@__DIR__, "..", "assets", "qrm_thorax_vmi_vs_theoretical.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 070b0001-0000-4000-8000-000000000032
let
    header = "| rod | " * join(["$(Int(E)) keV" for E in de_vmi_energies], " | ") * " | RMSE vs theory | RMSE vs theory, noise-free |"
    rule = "|---|" * repeat("---:|", length(de_vmi_energies) + 2)
    rows = map(eachindex(rod_data.names)) do i
        cells = ["$(round(rod_data.measured[i, j], digits = 1)) / $(round(rod_data.reference[i, j], digits = 1)) / $(round(rod_data.theoretical[i, j], digits = 1))"
                 for j in eachindex(de_vmi_energies)]
        "| $(rod_data.names[i]) | " * join(cells, " | ") *
        " | $(round(rod_data.rmse[i], digits = 1)) | $(round(rod_data.rmse_reference[i], digits = 1)) |"
    end
    Markdown.parse("Measured / noise-free / theoretical HU per rod and energy, in HU:\n\n" *
        join([header, rule, rows...], "\n"))
end

# ╔═╡ 070b0001-0000-4000-8000-000000000090
verification = let
    vmi_finite = all(isfinite, dual_vmi.vmis) && all(isfinite, reference_vmi.vmis)
    checks = [
        (name = "all channels enter the decomposition", value = basis.n_channels,
         pass = basis.n_channels == 2),
        (name = "Σ response matches I0 (max relative error)", value = round(basis.I0_relerr, sigdigits = 2),
         pass = basis.I0_relerr < 5e-5),
        (name = "chain: SpectralHYPR, ACNR, pair of windows",
         value = (dual_vmi.settings.denoiser !== nothing, dual_vmi.settings.acnr !== nothing,
                  dual_vmi.settings.recon.filter isa BS.PairFilter),
         pass = dual_vmi.settings.denoiser !== nothing && dual_vmi.settings.acnr !== nothing &&
                dual_vmi.settings.recon.filter isa BS.PairFilter),
        (name = "measured and reference share the pair",
         value = dual_vmi.settings.pair.basis == reference_vmi.settings.pair.basis,
         pass = dual_vmi.settings.pair.basis == reference_vmi.settings.pair.basis),
        (name = "VMI energies", value = de_vmi_energies, pass = de_vmi_energies == [50.0, 70.0, 100.0, 140.0]),
        (name = "VMI values finite", value = vmi_finite, pass = vmi_finite),
        (name = "water rod within ±10 HU at every energy",
         value = round.(water_rod_hu, digits = 1), pass = all(abs.(water_rod_hu) .≤ 10)),
    ]
    passed = count(check -> check.pass, checks)
    rows = join(["| $(c.name) | $(c.value) | $(c.pass ? "✅" : "❌") |" for c in checks], "\n")
    Markdown.parse("""
    ### $(passed == length(checks) ? "✅ Verification: PASS" : "❌ Verification: CHECK")

    | check | value | pass |
    |---|---:|:---:|
    $rows
    """)
end

# ╔═╡ 070c0001-0000-4000-8000-000000000001
let
    r = rod_data
    fmt(v) = join(round.(v, digits = 1), " / ")
    σ = [vmi_noise_by_keV[E].std for E in de_vmi_energies]
    Markdown.parse("""
    ## Summary

    ```
    qrm_thorax_slice: analytic QRM-Thorax (1600 × 1100 × 20 @ 0.2 mm), rods 9–12 in the heart
       → simulate! at 80 and 140 kVp (GE Apex Elite, duty cycle 0.65 / 0.35, view_samples = 5)
         one projection per exposure → noise-free, measured and calibration draws
       → BS.spectral_basis_from_acquisitions
       → calibration draw → pair_basis, composite_energy
       → BS.vmi_pipeline(; denoiser = SpectralHYPR(), fbp_filter = PAIR_FILTER, pair_basis, composite_energy)
       → per-rod measured vs noise-free vs theoretical HU at 50 / 70 / 100 / 140 keV
    ```

    On this scan, at 50 / 70 / 100 / 140 keV:

    - the water rod reads $(fmt(water_rod_hu)) HU, and $(fmt(water_rod_hu_reference)) HU noise-free (theory: 0 HU);
    - measured vs theoretical HU agree to an RMSE of $(join(["$(n) $(round(e, digits = 1)) HU" for (n, e) in zip(r.names, r.rmse)], ", "));
    - the noise in the heart is σ = $(fmt(σ)) HU.

    Notebook 08 runs the same phantom, rods, grid, energies and ROIs through a photon-counting
    scanner, so the two notebooks compare directly.
    """)
end

# ╔═╡ Cell order:
# ╟─07010001-0000-4000-8000-000000000001
# ╟─07010002-0000-4000-8000-000000000001
# ╠═07010003-0000-4000-8000-000000000001
# ╠═07010003-0000-4000-8000-000000000005
# ╠═07010003-0000-4000-8000-000000000006
# ╠═07010003-0000-4000-8000-000000000010
# ╠═07010003-0000-4000-8000-000000000011
# ╠═07010003-0000-4000-8000-000000000012
# ╠═07010003-0000-4000-8000-000000000013
# ╠═07010003-0000-4000-8000-000000000040
# ╟─07010003-0000-4000-8000-000000000050
# ╟─07020000-0000-4000-8000-0000000000f1
# ╟─07020001-0000-4000-8000-000000000001
# ╠═07020001-0000-4000-8000-000000000010
# ╠═07020001-0000-4000-8000-000000000013
# ╠═07020001-0000-4000-8000-000000000014
# ╟─07020001-0000-4000-8000-000000000025
# ╠═07020001-0000-4000-8000-000000000026
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
# ╠═07030002-0000-4000-8000-000000000011
# ╠═07030002-0000-4000-8000-000000000012
# ╠═07030002-0000-4000-8000-000000000020
# ╟─07030002-0000-4000-8000-000000000021
# ╟─07030003-0000-4000-8000-000000000001
# ╠═07030003-0000-4000-8000-000000000010
# ╠═07030003-0000-4000-8000-000000000011
# ╠═07030003-0000-4000-8000-000000000020
# ╟─07030004-0000-4000-8000-000000000001
# ╠═07030004-0000-4000-8000-000000000010
# ╟─07030004-0000-4000-8000-000000000011
# ╠═07030004-0000-4000-8000-000000000025
# ╟─07030004-0000-4000-8000-000000000030
# ╟─07030000-0000-4000-8000-0000000000f1
# ╠═07030006-0000-4000-8000-000000000010
# ╟─07030006-0000-4000-8000-000000000011
# ╟─07030006-0000-4000-8000-000000000001
# ╠═07030006-0000-4000-8000-000000000012
# ╠═07030006-0000-4000-8000-000000000013
# ╠═07030006-0000-4000-8000-000000000014
# ╠═07030006-0000-4000-8000-000000000015
# ╠═07030006-0000-4000-8000-000000000016
# ╠═07030009-0000-4000-8000-000000000010
# ╟─07030006-0000-4000-8000-000000000020
# ╠═07030006-0000-4000-8000-000000000021
# ╟─07030007-0000-4000-8000-000000000001
# ╠═07030007-0000-4000-8000-000000000010
# ╠═07030007-0000-4000-8000-000000000012
# ╟─07030007-0000-4000-8000-000000000011
# ╟─07030006-0000-4000-8000-000000000040
# ╟─07030008-0000-4000-8000-000000000001
# ╟─07030008-0000-4000-8000-000000000030
# ╠═07030009-0000-4000-8000-000000000020
# ╟─07030009-0000-4000-8000-000000000040
# ╟─070b0001-0000-4000-8000-000000000001
# ╠═070b0001-0000-4000-8000-000000000010
# ╠═070b0001-0000-4000-8000-000000000011
# ╠═070b0001-0000-4000-8000-000000000012
# ╠═070b0001-0000-4000-8000-000000000013
# ╠═070b0001-0000-4000-8000-000000000014
# ╠═070b0001-0000-4000-8000-000000000015
# ╟─070b0001-0000-4000-8000-000000000000
# ╟─070b0001-0000-4000-8000-00000000000a
# ╟─070b0001-0000-4000-8000-000000000002
# ╠═070b0001-0000-4000-8000-000000000004
# ╟─070b0001-0000-4000-8000-000000000003
# ╟─070d0001-0000-4000-8000-000000000001
# ╠═070d0001-0000-4000-8000-000000000020
# ╠═070d0001-0000-4000-8000-000000000015
# ╟─070d0001-0000-4000-8000-000000000016
# ╟─070d0001-0000-4000-8000-000000000030
# ╟─070b0001-0000-4000-8000-000000000008
# ╟─070b0001-0000-4000-8000-000000000009
# ╠═070b0001-0000-4000-8000-000000000020
# ╟─070b0001-0000-4000-8000-000000000030
# ╟─070b0001-0000-4000-8000-000000000031
# ╟─070b0001-0000-4000-8000-000000000032
# ╟─070b0001-0000-4000-8000-000000000090
# ╟─070c0001-0000-4000-8000-000000000001
