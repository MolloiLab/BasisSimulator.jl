### A Pluto.jl notebook ###
# v0.2.6

using Markdown
using InteractiveUtils

# ╔═╡ 08010001-0000-4000-8000-000000000001
md"""
# QRM-Thorax Pure-Material PCCT VMI: Photon-Counting Reference

The photon-counting counterpart of notebook 07: the same analytic **QRM-Thorax** phantom
with four pure-material rods (water · lipid · collagen · 5 mg/mL iodine), the same
reconstruction grid, VMI energies and ROIs, scanned by the Siemens NAEOTOM Alpha model of the
basis-spectral-denoising study at 140 kVp with four energy bins. All four corrected bins enter
the package's VMI chain, `BS.vmi_pipeline`, with that study's denoiser, reconstruction windows
and spectral pair, and every rod is checked against the HU its material has in theory and
against the same chain applied to the noise-free acquisition.

| Stage       | Matrix               | Voxel (mm)             | Extent                        |
|-------------|----------------------|------------------------|-------------------------------|
| Phantom     | **1600 × 1100 × 20** | **0.2 isotropic**      | 320 × 220 × 4 mm              |
| Recon       | 512 × 512 × **3**    | **0.625 isotropic**    | FOV 32 cm × 1.875 mm z        |
| Collimation | **5.0 mm nominal**   | —                      | cone-guard rows added automatically |
| Scanner     | Siemens NAEOTOM Alpha | 0.302 × 0.353 mm pixels (2 × 2 binned) | 144 rows, 50 cm scan field |
| Protocol    | PCCT, 4 bins         | 140 kVp · 1200 views · 0.5 s rotation · 5 sub-views per view | the physical scan's 10.12 mGy CTDIvol |

```
qrm_thorax_slice (analytic 2-D shapes) → bore rods 9–12 into the heart → tile z → BS.Phantom
   → simulate!(…; keep_projection = true)    noise-free draw; keeps the projection
   → simulate!(…; projection)                calibration and measured draws: only the noise is new
        4 corrected bins + per-ray air counts I0[col, row, bin]
   → BS.spectral_basis(ws; I0)               (per-ray absolute response of each bin)
   → calibration draw → spectral_pair        (pair_basis, composite_energy)
   → BS.vmi_pipeline(; denoiser = SpectralHYPR(), fbp_filter = PairFilter(…), pair_basis,
                       composite_energy)     (projection HYPR → K = 4 decomposition → FDK of
                                              composite + complement → ACNR → image HYPR → VMIs)
   → per-rod measured vs theoretical HU at 50 / 70 / 100 / 140 keV, and vs the noise-free draw
```

!!! tip "When to reach for this notebook"
    Use 08 for photon-counting VMI accuracy on a body-sized phantom and field of view at
    the resolution a NAEOTOM Alpha delivers. Notebook 04 runs the same scanner and chain on the
    Gammex 472; notebook 07 scans this phantom with dual-kVp switching.

!!! info "Why pure end-members?"
    `XA.Materials.basis_fat` is ICRU-44 adipose tissue (≈ 83 % triolein, 17 % water and
    trace electrolytes), not a pure lipid. `XA.Materials.basis_lipid` (H/C/O at 0.92 g/cm³)
    and `basis_collagen` (H/C/N/O at 1.26 g/cm³) are mathematically pure end members, made
    for clean water/lipid/collagen decomposition checks.
"""

# ╔═╡ 08010002-0000-4000-8000-000000000001
md"""
## Notebook Setup
"""

# ╔═╡ 08010003-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 08010003-0000-4000-8000-000000000005
using Markdown: @md_str, Markdown

# ╔═╡ 08010003-0000-4000-8000-000000000006
using Statistics: mean, std, quantile

# ╔═╡ 08010003-0000-4000-8000-000000000010
import BasisSimulator as BS

# ╔═╡ 08010003-0000-4000-8000-000000000011
# Use CairoMakie for faithful build-time rendering. Snapshot can still isolate
# and compile independent browser-safe islands without hoisting this import.
import CairoMakie as Mke

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
end;

# ╔═╡ 08010003-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 08020000-0000-4000-8000-0000000000f1
md"""
## Scan Setup and Simulation

One clinical PCCT acquisition: the QRM-Thorax phantom, the Siemens NAEOTOM Alpha
photon-counting scanner, the 140 kVp protocol, and the four-bin forward projection, drawn
three times from one projection.
"""

# ╔═╡ 08020001-0000-4000-8000-000000000001
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

# ╔═╡ 08020001-0000-4000-8000-000000000010
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

# ╔═╡ 08020001-0000-4000-8000-000000000013
const QRM_NZ = 20;    # 20 × 0.2 mm = 4 mm: a short z-invariant phantom

# ╔═╡ 08020001-0000-4000-8000-000000000014
const QRM_VOXEL_SIZE_CM = (0.02, 0.02, 0.02);   # (x, y, z) cm: 0.2 mm isotropic

# ╔═╡ 08020001-0000-4000-8000-000000000025
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

# ╔═╡ 08020001-0000-4000-8000-000000000026
# (center in mm, radius in mm, label) of each heart rod
HEART_RODS = [(rod.center, rod.r, rod.label) for rod in QRM_GEOMETRY.heart_rods];

# ╔═╡ 08020001-0000-4000-8000-000000000029
mask_3d = let
    slice = qrm_thorax_slice(; voxel_mm = QRM_VOXEL_SIZE_CM[1] * 10, inserts = HEART_RODS)
    repeat(slice; outer = (1, 1, QRM_NZ))
end;

# ╔═╡ 08020001-0000-4000-8000-000000000040
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

# ╔═╡ 08020001-0000-4000-8000-000000000061
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

# ╔═╡ 08020001-0000-4000-8000-000000000062
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

# ╔═╡ 08030001-0000-4000-8000-000000000001
md"""
### 2. `PCCTScanner`: Siemens NAEOTOM Alpha (4 thresholds)

The NAEOTOM Alpha model of basis-spectral-denoising (and semmd-bayesian), value for value, as in
notebook 04: a CdTe direct-conversion detector with native dexels 0.275 × 0.322 mm at the
detector face read out 2 × 2 binned, an arc detector with the quarter-detector offset, a 50 cm
scan field behind the large-body bowtie, 10 keV energy resolution, charge sharing and 5 ns dead
time. Pile-up is simulated and corrected, and so is scatter; the counts are exact Poisson. The
thresholds `T = [20, 35, 55, 70] keV` define four bins:

| Bin | Range (keV) |
|-----|-------------|
| 1   | 20 – 35     |
| 2   | 35 – 55     |
| 3   | 55 – 70     |
| 4   | > 70        |

The bowtie makes the air response vary across the fan, so the spectral model is per ray.
"""

# ╔═╡ 08030001-0000-4000-8000-000000000010
scanner = let
    native_col_mm, native_row_mm, binning = 0.275, 0.322, 2
    sid, sdd = 610.0, 1113.0
    col_iso = native_col_mm * binning / (sdd / sid)     # binned pixel at isocentre, mm
    row_iso = native_row_mm * binning / (sdd / sid)
    BS.PCCTScanner(
        source_to_isocenter = sid,
        source_to_detector = sdd,
        detector_rows = 144,
        detector_cols = ceil(Int, 500.0 / col_iso),     # covers the 50 cm scan field
        detector_row_size = row_iso,
        detector_col_size = col_iso,
        detector_shape = :arc,
        detector_col_offset = 0.25,                     # quarter-detector offset
        focal_spot_width = 0.4,
        focal_spot_length = 0.5,
        target_angle = 7.0,
        gantry_rotation_time = 0.5,
        scan_diameter = 500.0,
        gantry_aperture = 820.0,
        flat_filter_material = :aluminum,
        flat_filter_thickness = 3.0,
        bowtie_filter = :large_body,
        detector_material = :cdte,
        detector_depth = 1.6,
        fill_factor_row = 0.95,
        fill_factor_col = 0.95,
        energy_thresholds = [20.0, 35.0, 55.0, 70.0],   # four energy windows, keV
        energy_resolution = 10.0,
        charge_sharing_fwhm = 0.08,
        dead_time_ns = 5.0,
        pixel_mode = :standard,
        native_dexel_col_mm = native_col_mm,
        native_dexel_row_mm = native_row_mm,
        binning_factor = binning,
        pileup_correction = true,      # model-based inverse of the pile-up migration
        scatter_correction = true,     # scatter re-estimated and subtracted per bin
        noise_reduction = 0.0,         # exact Poisson counts
    )
end;

# ╔═╡ 08030002-0000-4000-8000-000000000001
md"""
### 3. `CTProtocol`: 140 kVp, 5.0 mm Collimation, at the Physical Scan's Dose

The protocol of the physical NAEOTOM Alpha Gammex scan: 140 kVp, 0.5 s rotation, 1200 views,
5 mm collimation, 0.9 mm titanium added, 174 mA and a reported CTDIvol of 10.12 mGy. A tube's
output per mA is its own and the simulator's source is generic, so the tube current is scaled by
one factor, `DOSE_SCALE`, that makes the simulated CTDIvol (BasisSimulator's Monte Carlo dose in
the 32 cm body phantom) equal the physical scan's. The reconstruction is the same centred
3-slice grid as notebook 07; the workspace adds the symmetric detector guard rows for full cone
support.
"""

# ╔═╡ 08030002-0000-4000-8000-000000000011
"The NAEOTOM Alpha's 140 kVp axial protocol at tube current `mA`."
alpha_protocol(mA) = BS.CTProtocol(
    kVp = 140,
    mA = mA,
    views = 1200,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Ti", 0.9)],
);

# ╔═╡ 08030002-0000-4000-8000-000000000012
# The tube-current scale that makes the simulated CTDIvol equal the physical scan's
DOSE_SCALE = let physical = (mA = 174.0, ctdi_vol_mGy = 10.12)
    p = alpha_protocol(physical.mA)
    physical.ctdi_vol_mGy / BS.compute_dose(BS.dose_source(scanner, p), p).ctdi_vol_mGy
end

# ╔═╡ 08030002-0000-4000-8000-000000000010
protocol = alpha_protocol(174.0 * DOSE_SCALE);

# ╔═╡ 08030003-0000-4000-8000-000000000001
md"""
### 4. `SimOptions` and `ReconOptions`

A `PCCTScanner` selects the photon-counting path (per-bin sinograms, the Monte Carlo detector
response, Compton scatter, pile-up); the detector-model settings live on the scanner, and
`SimOptions` carries the physics common to both detector families. These are the switches of the
basis-spectral-denoising Alpha model: noise, scatter and the focal-spot blur on; the
scintillator effects (fill factor, optical crosstalk, the scintillator efficiency table, lag)
off, since the CdTe detector's efficiency, fill and charge sharing are in its Monte Carlo
response; the heel effect off.

**View integration.** A detector integrates while the gantry turns, so each view reads the
transmitted intensity averaged over the arc it sweeps: a blur of the object, largest far from
the isocentre, that the noise, counted once per view, does not share. `view_samples = 5` samples
that arc with five sub-views per view (the midpoint rule), and `view_arc = 1.0` because a
photon-counting detector reads for the whole view period.

**Three draws, one projection.** Everything before the noise is the same for every draw, so the
projection (sub-views included) is computed once, on the noise-free draw
(`keep_projection = true`), and the other draws reuse it (`projection`), bit-identically to
simulating them afresh:

| draw | noise | seed | used for |
|---|---|---:|---|
| `reference` | off | — | the noise-free expectation every result is compared with |
| `calibration` | on | 9234 | fixing the spectral pair, never measured |
| `measured` | on | 1234 | every result below |
"""

# ╔═╡ 08030003-0000-4000-8000-000000000010
"The Alpha model's physics switches for one draw."
draw_options(; seed, use_noise) = BS.SimOptions(
    seed = seed,
    use_noise = use_noise,
    projector = :dd_fast,
    use_fill_factor = false,
    use_detector_efficiency = false,
    use_optical_crosstalk = false,
    use_focal_spot = true,
    use_heel_effect = false,
    use_lag = false,
    view_samples = 5,     # sub-views per view: the arc the gantry turns while a view is read
    view_arc = 1.0,       # a photon-counting detector integrates for the whole view period
);

# ╔═╡ 08030003-0000-4000-8000-000000000011
# basis-spectral-denoising's seeds: draw r has seed 1234 + 1000 (r − 1); the calibration draw is draw 9
DRAWS = (
    reference = (seed = 1234, use_noise = false),
    calibration = (seed = 1234 + 1000 * (9 - 1), use_noise = true),
    measured = (seed = 1234, use_noise = true),
);

# ╔═╡ 08030003-0000-4000-8000-000000000020
# The saved recon grid of notebook 07: 512 × 512 at 0.625 mm isotropic, 3 slices.
recon_opts = BS.ReconOptions(
    matrix_size = (512, 512, 3),
    fov_cm = 32.0,
    z_cm = 0.1875,
);

# ╔═╡ 08030004-0000-4000-8000-000000000001
md"""
### 5. Forward Project: `simulate!` and `spectral_basis`

`simulate!` runs the whole photon-counting detector model on the GPU: polychromatic forward
projection of every sub-view through the Monte Carlo detector response, scatter injection,
Poisson counts in every bin, pile-up migration, then pile-up and scatter correction. It returns
the four corrected sinograms ``h_k = -\log(y_k / I_{0,k})`` (`pcct_sino.bins`,
`[col, row, view]`), the air count of every ray in every bin (`I0_bins`, `[col, row, bin]`), the
dose report and, with `keep_projection = true`, the projection the next draw reuses.

`spectral_basis(ws; I0)` is read from the same workspace: the response the simulation
applied, so the decomposition inverts exactly what was simulated and needs no calibration
scan. It checks that the response sums to `I0` in every ray and bin. The model does not depend
on the noise, so one basis serves every draw.
"""

# ╔═╡ 08030004-0000-4000-8000-000000000010
acquisition = let
    projection = nothing
    model = nothing
    channels = Dict{Symbol, Vector{Array{Float32, 3}}}()
    seconds = Dict{Symbol, Float64}()
    for name in (:reference, :calibration, :measured)     # the noise-free draw first: it projects
        opts = draw_options(; DRAWS[name]...)
        t0 = time()
        ws = BS.create_workspace(scanner, protocol, opts, recon_opts, phantom)
        try
            # the dose and the projection do not depend on the noise: both come from the first draw
            sim = BS.simulate!(ws, phantom, protocol, opts; capture_raw_counts = false,
                report_dose = projection === nothing,
                keep_projection = projection === nothing, projection = projection)
            projection === nothing && (projection = sim.projection)
            channels[name] = [Array(b) for b in sim.pcct_sino.bins]   # the four corrected bins
            if model === nothing
                I0 = Float32.(Array(sim.I0_bins))                     # [n_cols, n_rows, n_bins]
                model = (
                    I0 = I0,
                    basis = BS.spectral_basis(ws; I0 = I0),           # ray-resolved spectral model
                    geom = ws.geom,
                    dose = sim.dose,
                )
            end
        finally
            BS.release_backend!(ws)
        end
        seconds[name] = time() - t0
    end
    (; channels, seconds, model...)
end;

# ╔═╡ 08030004-0000-4000-8000-000000000011
let
    a = acquisition
    nc, nr, K = size(a.I0)
    c, r = (nc + 1) ÷ 2, (nr + 1) ÷ 2
    fmt(x) = string(round(x; sigdigits = 3))
    s(name) = string(round(a.seconds[name]; digits = 1), " s")
    Markdown.parse("""
    | quantity | value |
    |---|---|
    | sinogram per bin | $(nc) columns × $(nr) rows × $(size(first(a.channels[:measured]), 3)) views |
    | air counts per ray and view, centre ray (bins 1 to 4) | $(join(round.(Int, a.I0[c, r, :]), " / ")) |
    | spectral basis | $(size(a.basis.Φ, 3)) energies, ray-resolved = $(a.basis.ray_resolved) |
    | largest relative mismatch of Σ Φ and I0 | $(fmt(a.basis.I0_relerr)) |
    | tube current | $(round(protocol.mA; digits = 1)) mA (174 mA × DOSE_SCALE = $(round(DOSE_SCALE; digits = 3))) |
    | CTDIvol (32 cm body phantom) | $(round(a.dose.ctdi_vol_mGy; digits = 2)) mGy |
    | wall time: noise-free draw (projects) / calibration / measured | $(s(:reference)) / $(s(:calibration)) / $(s(:measured)) |
    """)
end

# ╔═╡ 08030004-0000-4000-8000-000000000025
# The phantom labels resampled onto the reconstruction grid (nearest neighbour); every ROI
# below is built from these labels in recon-voxel coordinates.
phantom_in_recon = BS.resample_to_recon(
    phantom_cpu, acquisition.geom, recon_opts.matrix_size; method = :nearest,
);

# ╔═╡ 08030004-0000-4000-8000-000000000040
let
    mid_r = size(first(acquisition.channels[:measured]), 2) ÷ 2 + 1
    bin_subs = ("20 – 35 keV", "35 – 55 keV", "55 – 70 keV", "> 70 keV")
    slices = [permutedims(b[:, mid_r, :], (2, 1)) for b in acquisition.channels[:measured]]
    sino_window = Tuple(Float64.(quantile(vcat(vec.(slices)...), (0.01, 0.99))))

    fig = Mke.Figure(size = (1180, 1180))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "View", ylabel = "Detector Column",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )
    for k in 1:4
        ax = Mke.Axis(
            fig[(k - 1) ÷ 2 + 1, (k - 1) % 2 + 1]; title = "Bin $(k)", subtitle = bin_subs[k],
            axis_kwargs...,
        )
        Mke.heatmap!(ax, slices[k]; colormap = :viridis, colorrange = sino_window)
    end
    Mke.Colorbar(
        fig[1:2, 3]; colormap = :viridis, colorrange = sino_window,
        label = "Log Line Integral", width = 16, labelsize = 22, ticklabelsize = 18,
    )
    fig
end

# ╔═╡ 08030000-0000-4000-8000-0000000000f1
md"""
## VMI Pipeline

### 1. The Chain's Settings

The denoiser is generalized HYPR-LR in both domains, `BS.SpectralHYPR()`, the chain of
basis-spectral-denoising:

- **projection domain**: within each detector row, each ray's split of its total count across
  the four bins is pooled over a 3 × 3 (column × view) window whose view neighbours are two
  views apart (`view_stride = 2`), so the odd and even views stay independent; the count
  dispersion of every bin is measured from the air rays;
- **image domain**: the reconstructed basis pair is rewritten as the minimum-noise VMI (the
  composite), kept as reconstructed, and a complement whose noise is uncorrelated with it,
  pooled within its slice with likelihood weights on the composite over the window width (3 to
  41 pixels) with the least estimated risk. Every noise level comes from the difference of the
  odd- and even-view reconstructions, and no slices are averaged.

The decomposition is the K-channel maximum-likelihood estimator on all four bins with its
published controls; T-LBF is off; the view-direction antialias is on; ACNR acts on the
complement, before the image HYPR.
"""

# ╔═╡ 08030007-0000-4000-8000-000000000012
HYPR_CHAIN = BS.SpectralHYPR();

# ╔═╡ 08030007-0000-4000-8000-000000000013
# The decomposition and reconstruction settings of the chain; ACNR is added on the measured draw
VMI_CHAIN = (method = :nchannel, controls = BS.NChannelControls(), use_tlbf = false, antialias = true);

# ╔═╡ 0803000a-0000-4000-8000-000000000010
pcct_vmi_energies = [50.0, 70.0, 100.0, 140.0];

# ╔═╡ 08030007-0000-4000-8000-000000000020
md"""
### 2. The Reconstruction Windows, Rescaled to This Grid

`BS.PairFilter` gives the composite and its complement each their own FDK window (the water
and iodine images, whose noise is anti-correlated, must be filtered alike, or their noise stops
cancelling in the quiet VMIs). Each window is ``W(f) = \exp(-(f/f_c)^p)`` on the grid-Nyquist
axis, sampled at 11 knots as a `CustomFilter`. basis-spectral-denoising fitted the Alpha's pair to
the physical Gammex scan's MTF and NPS shape on 512² over 35 cm: composite ``f_c = 0.5``,
``p = 2``; complement ``f_c = 0.45``, ``p = 6``. This grid is 512² over 32 cm, whose finer pixels
represent higher frequencies, so each window is kept the same function of physical frequency by
rescaling its argument by the ratio `r` of the two grids' bandlimits (`BS.grid_bandlimit`), as the
study does for its other grids.
"""

# ╔═╡ 08030007-0000-4000-8000-000000000021
ALPHA_WINDOWS = (composite = (fc = 0.5, p = 2.0), complement = (fc = 0.45, p = 6.0));

# ╔═╡ 08030007-0000-4000-8000-000000000022
PAIR_FILTER = let knots = Tuple(range(0.0, 1.0, length = 11))
    # the grid the windows were fitted on: the Alpha's clinical series, 512² over 35 cm
    fitted = BS.CTGeometry(scanner; n_angles = protocol.views, fov_cm = 35.0, z_cm = 0.48,
        collimation_mm = protocol.collimation_mm)
    r = BS.grid_bandlimit(acquisition.geom, recon_opts.matrix_size) /
        BS.grid_bandlimit(fitted, (512, 512, 12))
    window(q) = BS.CustomFilter(knots, Tuple(round(exp(-(x * r / q.fc)^q.p), digits = 5) for x in knots))
    (filter = BS.PairFilter(window(ALPHA_WINDOWS.composite), window(ALPHA_WINDOWS.complement)), r = r)
end;

# ╔═╡ 08030007-0000-4000-8000-000000000023
md"""
### 3. The Spectral Pair, Fixed from the Calibration Draw

Near its minimum the VMI noise hardly changes with energy, so the minimum-noise energy measured
on one acquisition is itself noise; it is a property of the scanner and protocol, measured once
on the calibration draw and fixed for the measured and the noise-free draws:
`pair_basis = (Estar, β)` from the calibration draw decomposed without denoising and
reconstructed with `SoftFilter` (the pair the windows act on), and `composite_energy` from the same
draw after the projection-domain HYPR (the composite ACNR and the image HYPR act on).
"""

# ╔═╡ 08030007-0000-4000-8000-000000000024
PAIR_BASIS = let ch = acquisition.channels[:calibration], I0 = acquisition.I0
    decompose(channels) = BS.vmi_pipeline(; channels, basis = acquisition.basis,
        geom = acquisition.geom, to_backend = to_gpu, matrix_size = recon_opts.matrix_size,
        vmi_energies = Tuple(pcct_vmi_energies), keep_sinograms = true, use_acnr = false,
        VMI_CHAIN...).sinograms
    p = HYPR_CHAIN.projection
    d_none = decompose(ch)
    d_proj = decompose(BS.hypr_lr(ch, I0; kernel = p.kernel, dispersion = BS.estimate_dispersion(ch, I0),
        view_stride = p.view_stride, to_backend = to_gpu))
    sp(d; kw...) = BS.spectral_pair(d.water, d.iodine, acquisition.geom, recon_opts.matrix_size;
        filter = BS.SoftFilter(), antialias = VMI_CHAIN.antialias, to_backend = to_gpu, kw...)
    x = sp(d_none)
    c = sp(d_proj; basis = (Estar = x.Estar, β = x.β))
    (Estar = x.Estar, β = x.β, composite = c.Estar)
end

# ╔═╡ 08030007-0000-4000-8000-000000000001
md"""
### 4. `vmi_pipeline`

One call, from the four corrected bins of the measured draw to the VMI stack, reconstructed on
the notebook's grid with every detector row kept. `keep_sinograms = true` also returns the
decomposed basis sinograms, shown below. The noise-free draw goes through the same call without
the denoiser and ACNR: the reference, reconstructed exactly as its noisy counterpart.
"""

# ╔═╡ 08030007-0000-4000-8000-000000000010
pcct_vmi = BS.vmi_pipeline(;
    channels = acquisition.channels[:measured],
    basis = acquisition.basis,
    geom = acquisition.geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = Tuple(pcct_vmi_energies),
    denoiser = HYPR_CHAIN,
    fbp_filter = PAIR_FILTER.filter,
    pair_basis = (Estar = PAIR_BASIS.Estar, β = PAIR_BASIS.β),
    composite_energy = PAIR_BASIS.composite,
    keep_sinograms = true,
    VMI_CHAIN...,
);

# ╔═╡ 08030007-0000-4000-8000-000000000014
reference_vmi = BS.vmi_pipeline(;
    channels = acquisition.channels[:reference],
    basis = acquisition.basis,
    geom = acquisition.geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = Tuple(pcct_vmi_energies),
    fbp_filter = PAIR_FILTER.filter,
    pair_basis = (Estar = PAIR_BASIS.Estar, β = PAIR_BASIS.β),
    composite_energy = PAIR_BASIS.composite,
    use_acnr = false,
    VMI_CHAIN...,
);

# ╔═╡ 08030007-0000-4000-8000-000000000011
let
    q = pcct_vmi.quality
    d = pcct_vmi.settings.denoiser
    pct(x) = round(100x, digits = 3)
    Markdown.parse("""
    The decomposition solved $(q.n_rays) rays in $(round(pcct_vmi.elapsed_s, digits = 1)) s
    ($(round(q.outer_mean, digits = 1)) outer iterations on average); $(pct(q.frac_not_converged))% did not
    converge and $(pct(q.frac_bound_iodine))% / $(pct(q.frac_bound_water))% touched the iodine / water bounds.
    The projection HYPR measured dispersions (variance / mean of the counts on the air rays) of
    $(join(round.(d.dispersion, digits = 2), " / ")) for bins 1 to 4. From the calibration draw, the windows
    act on the pair at E* = $(round(Int, PAIR_BASIS.Estar)) keV (β = $(round(PAIR_BASIS.β, sigdigits = 3))), and the
    composite is fixed at $(round(Int, PAIR_BASIS.composite)) keV. The windows' bandlimit ratio on this grid is
    r = $(round(PAIR_FILTER.r, digits = 4)), and the image HYPR chose a $(d.image_estimates.window)-pixel window
    for the complement.
    """)
end

# ╔═╡ 08030007-0000-4000-8000-000000000040
let
    mid_r = size(pcct_vmi.sinograms.iodine, 2) ÷ 2 + 1
    fig = Mke.Figure(size = (1400, 580))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "View", ylabel = "Detector Column",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )
    panels = (
        ("Iodine Basis Sinogram", pcct_vmi.sinograms.iodine),
        ("Water Basis Sinogram", pcct_vmi.sinograms.water),
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

# ╔═╡ 08030008-0000-4000-8000-000000000001
md"""
### 3. Basis Maps and VMIs

The water and iodine basis pair after ACNR and the image HYPR (mid slice), and the VMIs
synthesized from it: ``\mu(E) = c_\mathrm{water}\,(\mu/\rho)_\mathrm{water}(E) +
c_\mathrm{iodine}\,(\mu/\rho)_\mathrm{iodine}(E)``, in HU relative to water at the same
energy. Every VMI comes from the same basis pair.
"""

# ╔═╡ 08030008-0000-4000-8000-000000000030
let
    fig = Mke.Figure(size = (1180, 580))
    mid = size(pcct_vmi.images.water, 3) ÷ 2 + 1
    # iodine in mg/mL on a fixed window around the 5 mg/mL rod (a percentile window of the
    # whole slice sits below the rod, which covers under 1 % of it); water on its percentiles
    iodine_mg = 1000 .* pcct_vmi.images.iodine[:, :, mid]
    water = pcct_vmi.images.water[:, :, mid]
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

# ╔═╡ 0803000a-0000-4000-8000-000000000020
# The VMI stack keyed by energy, (512, 512, 3) HU volumes.
vmi_HU_final = Dict(
    Float64(E) => pcct_vmi.vmis[:, :, :, k] for (k, E) in enumerate(pcct_vmi.energies)
);

# ╔═╡ 0803000a-0000-4000-8000-000000000021
# The noise-free reference's VMI stack, keyed alike.
vmi_HU_reference = Dict(
    Float64(E) => reference_vmi.vmis[:, :, :, k] for (k, E) in enumerate(reference_vmi.energies)
);

# ╔═╡ 0803000a-0000-4000-8000-000000000040
let
    HU_window = (-200, 500)
    fig = Mke.Figure(size = (1180, 1180))
    mid = size(pcct_vmi.vmis, 3) ÷ 2 + 1
    for (k, E) in enumerate(pcct_vmi_energies)
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
    Mke.save(joinpath(@__DIR__, "..", "assets", "qrm_thorax_pcct_vmi_grid.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 080b0001-0000-4000-8000-000000000001
md"""
## Results

Per-rod measured vs theoretical HU at 50 / 70 / 100 / 140 keV, the water rod's mean HU, and
the noise in the heart.

- **Measured HU**: the mean over a circular ROI at each rod's centroid, over every slice.
- **Theoretical HU**: ``1000\,(\mu_\mathrm{rod}(E) - \mu_\mathrm{water}(E)) / \mu_\mathrm{water}(E)``
  from `BS.compute_μ_at_energy(material, E)`; pure physics, no fitting.
- **Noise-free HU**: the same ROI on the noise-free reference, the same decomposition and
  reconstruction without noise (and so without the denoiser). Its error against theory is the
  bias of the iodine–water basis model itself; the measured draw's departure from it is what the
  noise and the denoising add.
"""

# ╔═╡ 080b0001-0000-4000-8000-000000000010
const ROD_LABELS = (UInt8(9), UInt8(10), UInt8(11), UInt8(12));

# ╔═╡ 080b0001-0000-4000-8000-000000000011
const ROD_NAMES = ("Water", "Lipid", "Collagen", "Iodine 5 mg/mL");

# ╔═╡ 080b0001-0000-4000-8000-000000000012
const ROD_MATERIALS = (
    BS.XA.Materials.basis_water,
    BS.XA.Materials.basis_lipid,
    BS.XA.Materials.basis_collagen,
    BS.XA.Materials.gammex_472_i5_0,
);

# ╔═╡ 080b0001-0000-4000-8000-000000000013
const ROD_ROI_RADIUS_PX = 8;    # 5 mm at 0.625 mm/voxel, inside the 7.5 mm rods

# ╔═╡ 080b0001-0000-4000-8000-000000000014
const HEART_NOISE_ROI_RADIUS_PX = 12;    # 7.5 mm: fits between the four rods

# ╔═╡ 080b0001-0000-4000-8000-000000000015
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

# ╔═╡ 080b0001-0000-4000-8000-000000000000
md"""
### Phantom-Recon Alignment Verification

Before trusting any ROI built on `phantom_in_recon`: the 70 keV VMI and the resampled
phantom labels side by side on the same recon grid, then overlaid, so the rod, heart and
bone edges can be checked against the image.
"""

# ╔═╡ 080b0001-0000-4000-8000-00000000000a
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

# ╔═╡ 080b0001-0000-4000-8000-000000000002
md"""
### Water ROI

The water-rod ROI (label 9, `basis_water`) in red on the 70 keV VMI, and its mean HU at every
VMI energy, measured (bars) and on the noise-free reference (black diamonds). Water is the
reference of the HU scale, so the bars should sit at 0 HU; a constant offset would be a
basis-decomposition bias, a trend with energy a spectral-model error.
"""

# ╔═╡ 080b0001-0000-4000-8000-000000000004
water_rod_hu = [mean(roi_values(vmi_HU_final[E], rod_rois.rods[UInt8(9)])) for E in pcct_vmi_energies];

# ╔═╡ 080b0001-0000-4000-8000-000000000005
water_rod_hu_reference = [mean(roi_values(vmi_HU_reference[E], rod_rois.rods[UInt8(9)])) for E in pcct_vmi_energies];

# ╔═╡ 080b0001-0000-4000-8000-000000000003
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

    n_E = length(pcct_vmi_energies)
    ax2 = Mke.Axis(
        fig[1, 2]; title = "Water ROI Mean HU", subtitle = "per VMI energy",
        xlabel = "VMI Energy (keV)", ylabel = "HU",
        xticks = (collect(1:n_E), ["$(Int(E))" for E in pcct_vmi_energies]),
        titlesize = 32, subtitlesize = 24, xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    Mke.barplot!(
        ax2, 1:n_E, water_rod_hu;
        color = (:steelblue, 0.8), strokecolor = :black, strokewidth = 1, label = "measured",
    )
    Mke.scatter!(ax2, 1:n_E, water_rod_hu_reference; color = :black, marker = :diamond,
        markersize = 16, label = "noise-free reference")
    Mke.hlines!(ax2, [0.0]; color = :black, linewidth = 1, linestyle = :dash)
    for (k, h) in pairs(water_rod_hu)
        Mke.text!(
            ax2, k, h; text = "$(round(h, digits = 1)) HU",
            align = (:center, h ≥ 0 ? :bottom : :top), fontsize = 16, offset = (0, h ≥ 0 ? 12 : -12),
        )
    end
    Mke.axislegend(ax2; position = :lb, labelsize = 15)
    y_max = max(15.0, 1.2 * maximum(abs, vcat(water_rod_hu, water_rod_hu_reference)))
    Mke.ylims!(ax2, -y_max, y_max)
    Mke.save(joinpath(@__DIR__, "..", "assets", "qrm_thorax_pcct_water_roi.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 080d0001-0000-4000-8000-000000000001
md"""
### Heart-Centre Noise ROI

The standard deviation of HU in a 7.5 mm ROI in the soft-tissue heart, at the centre of the
four rods, at every VMI energy: how the chain carries noise from the basis pair into each VMI.
It is measured on the difference of the measured VMI and the noise-free reference, in which any
structure in the ROI (streaks, cupping) cancels; the labels give the measured VMI's mean HU.
"""

# ╔═╡ 080d0001-0000-4000-8000-000000000020
vmi_noise_by_keV = Dict(
    E => let v = roi_values(vmi_HU_final[E], rod_rois.heart),
             d = roi_values(vmi_HU_final[E] .- vmi_HU_reference[E], rod_rois.heart)
        (mean = mean(v), std = std(v), std_vs_reference = std(d), n = length(v))
    end
    for E in pcct_vmi_energies
);

# ╔═╡ 080d0001-0000-4000-8000-000000000015
# The basis-pair covariance in the heart ROI, which sets the noise of every VMI:
#   σ_HU(E)² = 10⁶ V_w + α(E)² V_i + 2·10³ α(E) C_iw,  α(E) = (μ/ρ)_I(E) / (μ/ρ)_w(E),
# with water in g/mL and iodine in mg/mL; lowest at α* = -10³ C_iw / V_i. Anti-correlated basis noise (ρ < 0) is what ACNR removes.
basis_covariance = let
    w = roi_values(pcct_vmi.images.water, rod_rois.heart)
    i = roi_values(pcct_vmi.images.iodine, rod_rois.heart) .* 1000.0   # mg/mL
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

# ╔═╡ 080d0001-0000-4000-8000-000000000016
let c = basis_covariance
    inside = c.α_range[1] ≤ c.α_star ≤ c.α_range[2]
    gap = maximum(abs(vmi_noise_by_keV[E].std - vmi_noise_by_keV[E].std_vs_reference) for E in pcct_vmi_energies)
    Markdown.parse("""
    The standard deviation of the measured VMI itself differs from that of its difference from the
    noise-free reference by at most $(round(gap, digits = 2)) HU at any energy: the ROI holds noise only.
    """ * "\n" * """
    In the heart ROI the water basis has σ = $(round(1000 * c.σ_water, digits = 1)) mg/mL, the iodine basis
    σ = $(round(c.σ_iodine, digits = 2)) mg/mL, and their correlation is ρ = $(round(c.ρ, digits = 2)).
    The noise-optimal mixing ratio is α* = $(round(c.α_star, digits = 1)) HU per mg/mL""" *
    (inside ? ", reached at about $(Int(c.E_star)) keV." :
        ", outside the 40–140 keV range, so the noise falls monotonically toward $(c.α_star < c.α_range[1] ? "high" : "low") energy."))
end

# ╔═╡ 080d0001-0000-4000-8000-000000000030
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
    σd = [vmi_noise_by_keV[E].std_vs_reference for E in Es]
    μs = [vmi_noise_by_keV[E].mean for E in Es]
    ax2 = Mke.Axis(
        fig[1, 2]; title = "Heart-Centre Noise vs Energy",
        subtitle = "σ of measured − noise-free, $(length(rod_rois.heart)) voxels × $(size(vmi_HU_final[70.0], 3)) slices",
        xlabel = "VMI Energy (keV)", ylabel = "Noise σ (HU)", xticks = Es,
        titlesize = 32, subtitlesize = 20, xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    Mke.scatterlines!(ax2, Es, σd; color = :tomato, markersize = 18, linewidth = 3)
    for (E, σ, μ) in zip(Es, σd, μs)
        Mke.text!(
            ax2, E, σ; text = "σ=$(round(σ; digits = 1))\n⟨HU⟩=$(round(μ; digits = 1))",
            align = (:center, :bottom), fontsize = 16, offset = (0, 8),
        )
    end
    Mke.ylims!(ax2, 0, maximum(σd) * 1.4)
    Mke.xlims!(ax2, first(Es) - 12, last(Es) + 12)
    Mke.save(joinpath(@__DIR__, "..", "assets", "qrm_thorax_pcct_noise_vs_energy.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 080b0001-0000-4000-8000-000000000008
md"""
### Basis Values in the Rods

The mean basis pair in each rod ROI. The water rod should read ``c_\mathrm{water} ≈ 1``
g/mL and ``c_\mathrm{iodine} ≈ 0``; the iodine rod carries its 5 mg/mL on a
water-equivalent base. Lipid and collagen are not made of water and iodine, so the chain
represents them as the water/iodine mixture that attenuates the same way (lipid with a
negative iodine value).
"""

# ╔═╡ 080b0001-0000-4000-8000-000000000009
let
    rows = map(zip(ROD_LABELS, ROD_NAMES, ROD_MATERIALS)) do (lab, name, mat)
        cw = mean(roi_values(pcct_vmi.images.water, rod_rois.rods[lab]))
        ci = 1000 * mean(roi_values(pcct_vmi.images.iodine, rod_rois.rods[lab]))
        "| $(name) | $(round(BS.XA.val(mat.density), digits = 3)) | $(round(cw, digits = 3)) | $(round(ci, digits = 2)) |"
    end
    Markdown.parse("""
    | rod | density (g/cm³) | water (g/mL) | iodine (mg/mL) |
    |-----|------:|------:|------:|
    $(join(rows, "\n"))
    """)
end

# ╔═╡ 080b0001-0000-4000-8000-000000000020
rod_data = let
    μ_water(E) = BS.compute_μ_at_energy(BS.XA.Materials.water, E)
    theoretical_hu(mat, E) = 1000.0 * (BS.compute_μ_at_energy(mat, E) - μ_water(E)) / μ_water(E)
    meas = [mean(roi_values(vmi_HU_final[E], rod_rois.rods[lab])) for lab in ROD_LABELS, E in pcct_vmi_energies]
    noise_free = [mean(roi_values(vmi_HU_reference[E], rod_rois.rods[lab])) for lab in ROD_LABELS, E in pcct_vmi_energies]
    theo = [theoretical_hu(mat, E) for mat in ROD_MATERIALS, E in pcct_vmi_energies]
    rmse = [sqrt(mean(abs2, meas[i, :] .- theo[i, :])) for i in eachindex(ROD_LABELS)]
    rmse_noise_free = [sqrt(mean(abs2, noise_free[i, :] .- theo[i, :])) for i in eachindex(ROD_LABELS)]
    (labels = ROD_LABELS, names = ROD_NAMES, measured = meas, noise_free = noise_free, theoretical = theo,
     rmse = rmse, rmse_noise_free = rmse_noise_free)
end;

# ╔═╡ 080b0001-0000-4000-8000-000000000030
md"""
### Per-Rod Measured vs Theoretical HU

Solid lines with circles: measured; dashed lines: theory; open diamonds: the noise-free reference.
"""

# ╔═╡ 080b0001-0000-4000-8000-000000000031
let
    fig = Mke.Figure(size = (1280, 580))
    rod_colors = [
        Mke.RGBf(0.2, 0.6, 0.85),    # water    — blue
        Mke.RGBf(0.95, 0.65, 0.13),  # lipid    — orange
        Mke.RGBf(0.55, 0.3, 0.65),   # collagen — purple
        Mke.RGBf(0.85, 0.27, 0.1),   # iodine   — red
    ]
    ax = Mke.Axis(
        fig[1, 1]; title = "Pure-Material Rods (PCCT)", subtitle = "50 / 70 / 100 / 140 keV",
        xlabel = "VMI energy (keV)", ylabel = "HU", xticks = pcct_vmi_energies,
        titlesize = 32, subtitlesize = 24, xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    rod_lines = Any[]
    rod_labels_str = String[]
    for i in eachindex(rod_data.names)
        c = rod_colors[i]
        Mke.scatterlines!(ax, pcct_vmi_energies, rod_data.measured[i, :]; color = c, linewidth = 2.5, markersize = 9)
        Mke.lines!(ax, pcct_vmi_energies, rod_data.theoretical[i, :]; color = c, linewidth = 1.6, linestyle = :dash)
        Mke.scatter!(ax, pcct_vmi_energies, rod_data.noise_free[i, :]; color = (:white, 0.0), marker = :diamond,
            markersize = 16, strokecolor = c, strokewidth = 2)
        push!(rod_lines, Mke.LineElement(color = c, linewidth = 2.5))
        push!(rod_labels_str, "$(rod_data.names[i]) (RMSE = $(round(rod_data.rmse[i], digits = 1)) HU)")
    end
    Mke.Legend(
        fig[1, 2],
        vcat([Mke.MarkerElement(color = :black, marker = :circle, markersize = 9),
              Mke.LineElement(color = :black, linewidth = 1.6, linestyle = :dash),
              Mke.MarkerElement(color = (:white, 0.0), marker = :diamond, markersize = 14,
                  strokecolor = :black, strokewidth = 2)], rod_lines),
        vcat(["Measured", "Theoretical", "Noise-free"], rod_labels_str);
        framevisible = true, labelsize = 18, rowgap = 1, padding = (6, 6, 6, 6),
    )
    Mke.save(joinpath(@__DIR__, "..", "assets", "qrm_thorax_pcct_vmi_vs_theoretical.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 080b0001-0000-4000-8000-000000000032
let
    header = "| rod | " * join(["$(Int(E)) keV" for E in pcct_vmi_energies], " | ") * " | RMSE (noise-free) |"
    rule = "|---|" * repeat("---:|", length(pcct_vmi_energies) + 1)
    f(x) = round(x, digits = 1)
    rows = map(eachindex(rod_data.names)) do i
        cells = ["$(f(rod_data.measured[i, j])) ($(f(rod_data.noise_free[i, j]))) / $(f(rod_data.theoretical[i, j]))"
                 for j in eachindex(pcct_vmi_energies)]
        "| $(rod_data.names[i]) | " * join(cells, " | ") * " | $(f(rod_data.rmse[i])) ($(f(rod_data.rmse_noise_free[i]))) |"
    end
    Markdown.parse("Measured (noise-free) / theoretical HU per rod and energy:\n\n" * join([header, rule, rows...], "\n"))
end

# ╔═╡ 080b0001-0000-4000-8000-000000000090
verification = let
    vmi_finite = all(isfinite, pcct_vmi.vmis) && all(isfinite, reference_vmi.vmis)
    checks = [
        (name = "all channels enter the decomposition", value = acquisition.basis.n_channels,
         pass = acquisition.basis.n_channels == 4),
        (name = "Σ response matches I0 (max relative error)", value = round(acquisition.basis.I0_relerr, sigdigits = 2),
         pass = acquisition.basis.I0_relerr < 5e-5),
        (name = "VMI energies", value = pcct_vmi_energies, pass = pcct_vmi_energies == [50.0, 70.0, 100.0, 140.0]),
        (name = "simulated CTDIvol equals the physical scan's 10.12 mGy",
         value = round(acquisition.dose.ctdi_vol_mGy, digits = 2), pass = abs(acquisition.dose.ctdi_vol_mGy - 10.12) < 0.05),
        (name = "VMI values finite (measured and reference)", value = vmi_finite, pass = vmi_finite),
        (name = "water rod within ±10 HU at every energy",
         value = round.(water_rod_hu, digits = 1), pass = all(abs.(water_rod_hu) .≤ 10)),
        (name = "noise-free water rod within ±10 HU at every energy",
         value = round.(water_rod_hu_reference, digits = 1), pass = all(abs.(water_rod_hu_reference) .≤ 10)),
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

# ╔═╡ 080c0001-0000-4000-8000-000000000001
let
    r = rod_data
    fmt(v) = join(round.(v, digits = 1), " / ")
    σ = [vmi_noise_by_keV[E].std_vs_reference for E in pcct_vmi_energies]
    Markdown.parse("""
    ## Summary

    ```
    qrm_thorax_slice: analytic QRM-Thorax (1600 × 1100 × 20 @ 0.2 mm), rods 9–12 in the heart
       → simulate! at 140 kVp, 4 bins, 5 sub-views per view (Siemens NAEOTOM Alpha, 10.12 mGy)
            noise-free, calibration and measured draws from one projection
       → BS.spectral_basis(ws; I0)
       → calibration draw → spectral_pair → pair_basis, composite_energy
       → BS.vmi_pipeline(; denoiser = SpectralHYPR(), fbp_filter = PairFilter(fitted, rescaled),
                           pair_basis, composite_energy)
       → per-rod measured vs theoretical HU at 50 / 70 / 100 / 140 keV, and vs the noise-free draw
    ```

    On this scan, at 50 / 70 / 100 / 140 keV:

    - the water rod reads $(fmt(water_rod_hu)) HU (noise-free: $(fmt(water_rod_hu_reference)) HU; theory: 0 HU);
    - measured vs theoretical HU agree to an RMSE of $(join(["$(n) $(round(e, digits = 1)) HU" for (n, e) in zip(r.names, r.rmse)], ", "));
      on the noise-free reference, $(join(["$(n) $(round(e, digits = 1)) HU" for (n, e) in zip(r.names, r.rmse_noise_free)], ", "));
    - the noise in the heart (measured − noise-free) is σ = $(fmt(σ)) HU.

    Notebook 07 runs the same phantom, rods, grid, energies and ROIs through a dual-kVp
    scanner, so the two notebooks compare directly.
    """)
end

# ╔═╡ Cell order:
# ╟─08010001-0000-4000-8000-000000000001
# ╟─08010002-0000-4000-8000-000000000001
# ╠═08010003-0000-4000-8000-000000000001
# ╠═08010003-0000-4000-8000-000000000005
# ╠═08010003-0000-4000-8000-000000000006
# ╠═08010003-0000-4000-8000-000000000010
# ╠═08010003-0000-4000-8000-000000000011
# ╠═08010003-0000-4000-8000-000000000012
# ╠═08010003-0000-4000-8000-000000000013
# ╠═08010003-0000-4000-8000-000000000040
# ╟─08010003-0000-4000-8000-000000000050
# ╟─08020000-0000-4000-8000-0000000000f1
# ╟─08020001-0000-4000-8000-000000000001
# ╠═08020001-0000-4000-8000-000000000010
# ╠═08020001-0000-4000-8000-000000000013
# ╠═08020001-0000-4000-8000-000000000014
# ╟─08020001-0000-4000-8000-000000000025
# ╠═08020001-0000-4000-8000-000000000026
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
# ╠═08030002-0000-4000-8000-000000000011
# ╠═08030002-0000-4000-8000-000000000012
# ╠═08030002-0000-4000-8000-000000000010
# ╟─08030003-0000-4000-8000-000000000001
# ╠═08030003-0000-4000-8000-000000000010
# ╠═08030003-0000-4000-8000-000000000011
# ╠═08030003-0000-4000-8000-000000000020
# ╟─08030004-0000-4000-8000-000000000001
# ╠═08030004-0000-4000-8000-000000000010
# ╟─08030004-0000-4000-8000-000000000011
# ╠═08030004-0000-4000-8000-000000000025
# ╟─08030004-0000-4000-8000-000000000040
# ╟─08030000-0000-4000-8000-0000000000f1
# ╠═08030007-0000-4000-8000-000000000012
# ╠═08030007-0000-4000-8000-000000000013
# ╠═0803000a-0000-4000-8000-000000000010
# ╟─08030007-0000-4000-8000-000000000020
# ╠═08030007-0000-4000-8000-000000000021
# ╠═08030007-0000-4000-8000-000000000022
# ╟─08030007-0000-4000-8000-000000000023
# ╠═08030007-0000-4000-8000-000000000024
# ╟─08030007-0000-4000-8000-000000000001
# ╠═08030007-0000-4000-8000-000000000010
# ╠═08030007-0000-4000-8000-000000000014
# ╟─08030007-0000-4000-8000-000000000011
# ╟─08030007-0000-4000-8000-000000000040
# ╟─08030008-0000-4000-8000-000000000001
# ╟─08030008-0000-4000-8000-000000000030
# ╠═0803000a-0000-4000-8000-000000000020
# ╠═0803000a-0000-4000-8000-000000000021
# ╟─0803000a-0000-4000-8000-000000000040
# ╟─080b0001-0000-4000-8000-000000000001
# ╠═080b0001-0000-4000-8000-000000000010
# ╠═080b0001-0000-4000-8000-000000000011
# ╠═080b0001-0000-4000-8000-000000000012
# ╠═080b0001-0000-4000-8000-000000000013
# ╠═080b0001-0000-4000-8000-000000000014
# ╠═080b0001-0000-4000-8000-000000000015
# ╟─080b0001-0000-4000-8000-000000000000
# ╟─080b0001-0000-4000-8000-00000000000a
# ╟─080b0001-0000-4000-8000-000000000002
# ╠═080b0001-0000-4000-8000-000000000004
# ╠═080b0001-0000-4000-8000-000000000005
# ╟─080b0001-0000-4000-8000-000000000003
# ╟─080d0001-0000-4000-8000-000000000001
# ╠═080d0001-0000-4000-8000-000000000020
# ╠═080d0001-0000-4000-8000-000000000015
# ╟─080d0001-0000-4000-8000-000000000016
# ╟─080d0001-0000-4000-8000-000000000030
# ╟─080b0001-0000-4000-8000-000000000008
# ╟─080b0001-0000-4000-8000-000000000009
# ╠═080b0001-0000-4000-8000-000000000020
# ╟─080b0001-0000-4000-8000-000000000030
# ╟─080b0001-0000-4000-8000-000000000031
# ╟─080b0001-0000-4000-8000-000000000032
# ╟─080b0001-0000-4000-8000-000000000090
# ╟─080c0001-0000-4000-8000-000000000001
