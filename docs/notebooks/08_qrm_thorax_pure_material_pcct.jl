### A Pluto.jl notebook ###
# v0.2.6

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
photon-counting scan whose four native corrected energy bins enter the
fully inlined generalized Cong likelihood directly. Lee total-likelihood
bilateral filtering begins only after the four-bin decomposition.

| Stage         | Matrix                   | Voxel (mm)             | Extent                        |
|---------------|--------------------------|------------------------|-------------------------------|
| GT phantom    | **1600 × 1100 × 20**     | **0.2 isotropic**      | 320 × 220 × 4 mm              |
| Recon         | 512 × 512 × **3**        | **0.625 isotropic**    | FOV 32 cm × 1.875 mm z        |
| Collimation   | **5.0 mm nominal**       | —                      | cone-guard rows added automatically |
| Scanner       | Siemens Naeotom Alpha    | 0.353 × 0.302 mm pixels | 144 × ~1192 detector          |
| Protocol      | PCCT, 4 bins             | 140 kVp · 174 mA · 1200 views · 0.5 s rot. | clinical PCCT dose |

```
QRM-Thorax mid-slice mask  → relabel rods 9–12 → tile z → BS.Phantom
                                       │
                            Simulate 140 kVp PCCT  (4 bins)
                                       │
                       Per-Bin Pile-up + Scatter Correction
                                       │
                       Four-Bin Profiled Cong Decomposition
                                       │
                       Joint T-LBF → Common FBP → ACNR → VMI
                                       │
                       Per-Rod Measured-vs-Theoretical Regression
                                       at 50 / 70 / 100 / 140 keV
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
    - Lee et al. (2025) — total-likelihood bilateral filtering.
"""

# ╔═╡ 08010002-0000-4000-8000-000000000001
md"""
## Notebook Setup
"""

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
const QRM_CACHE_PATH = joinpath(
    get(ENV, "BASISSIM_QRM_DIR", joinpath(@__DIR__, "data", "qrm_thorax")),
    "qrm_thorax_1600x1100_rot_uint8.raw",
);

# ╔═╡ 08020001-0000-4000-8000-000000000011
const QRM_TARGET_NX = 1600;  # full prepared cache, no extra in-plane downsample

# ╔═╡ 08020001-0000-4000-8000-000000000012
const QRM_TARGET_NY = 1100;  # full prepared cache, no extra in-plane downsample

# ╔═╡ 08020001-0000-4000-8000-000000000013
const QRM_TARGET_NZ = 20;    # 20 × 0.2 mm = 4 mm — short z-invariant reference phantom

# ╔═╡ 08020001-0000-4000-8000-000000000014
const QRM_VOXEL_SIZE_CM = (0.02, 0.02, 0.02);   # (x, y, z) cm — 0.2 mm isotropic ground truth (320 × 220 × 4 mm physical extent)

# ╔═╡ 08020001-0000-4000-8000-000000000015
const QRM_DATA_DIR = dirname(QRM_CACHE_PATH);

# ╔═╡ 08020001-0000-4000-8000-000000000016
const QRM_FULL_PATH = joinpath(QRM_DATA_DIR, "qrm_thorax_3200x2200_rot_uint8.raw");

# ╔═╡ 08020001-0000-4000-8000-000000000017
const QRM_DOWN_PATH = joinpath(QRM_DATA_DIR, "qrm_thorax_1600x1100_rot_uint8.raw");

# ╔═╡ 08020001-0000-4000-8000-000000000018
md"""
!!! info "Prepared QRM-Thorax phantom data"
    Set `BASISSIM_QRM_DIR` to the directory containing either prepared
    phantom file. The downsampled file is used by this notebook.

    | resolution                        | path                                                  |
    |-----------------------------------|-------------------------------------------------------|
    | full-resolution (3200 × 2200, ~7 MB) | `qrm_thorax_3200x2200_rot_uint8.raw` |
    | 2× downsampled (1600 × 1100, ~1.7 MB) ← used here | `qrm_thorax_1600x1100_rot_uint8.raw` |

    If the data is elsewhere, point `BASISSIM_QRM_DIR` at that directory
    before starting Pluto.
"""

# ╔═╡ 08020001-0000-4000-8000-000000000020
mask_3d_raw = let
    isfile(QRM_CACHE_PATH) || error(
        "QRM-Thorax cache is unavailable at " *
            "docs/notebooks/data/qrm_thorax/$(basename(QRM_CACHE_PATH)).\n" *
            "Set BASISSIM_QRM_DIR to the prepared-data directory, or run " *
            "the prep notebook once to rebuild the cache from source."
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
### 03. `CTProtocol`: 140 kVp / 174 mA / 5.0 mm collimation

Clinical 140 kVp single-energy PCCT acquisition.
`additional_filters = [("Ti", 0.9)]` is the Vectron tube's inherent
0.9 mm titanium window on top of the 3 mm Al flat filter.

**Nominal collimation = 5.0 mm at iso**. The saved reconstruction remains
the same centered 3-slice grid as notebook 07; the workspace automatically
adds symmetric detector guard rows for full cone support.
"""

# ╔═╡ 08030002-0000-4000-8000-000000000010
protocol = BS.CTProtocol(
    kVp = 140,
    mA = 174.0,    # clinical PCCT dose (matches header + canonical nb04); was 5.0
    views = 1200,
    rotation_time = 0.5,
    collimation_mm = 5.0,    # nominal width; full-FOV axial cone guards are automatic
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
# Same saved recon grid as notebook 07: 512 × 512 in-plane at 0.625 mm
# isotropic, 3 slices. Automatic detector guards provide full-FOV support.
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
# === SLOW CELL — full PCCT physics + corrections via simulate!() ===
sim_raw = let
    @info "simulate!(): $(Int(protocol.kVp)) kVp / $(round(protocol.mA, digits = 1)) mA — full PCCT physics + corrections"
    ws = BS.create_workspace(
        scanner, protocol, sim_opts, recon_opts, phantom,
    )
    try
        result = BS.simulate!(ws, phantom, protocol, sim_opts)
        bins_raw = [Float32.(Array(b)) for b in result.pcct_sino.bins]
        I0_bins = copy(result.I0_bins)
        geom = ws.geom
        energies = Float64.(ws.energies)
        weights = copy(ws.weights)
        W_applied = Float64.(Array(ws.W_matrix_gpu))[
            1:length(ws.energies), :,
        ]
        bf = scanner.binning_factor
        (
            bins_raw = bins_raw, I0_bins = I0_bins,
            W_applied = W_applied, geom = geom, energies = energies,
            weights = weights, bf = bf,
        )
    finally
        BS.release_backend!(ws)
    end
end

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

    fig = Mke.Figure(size = (1180, 1180))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "View", ylabel = "Detector Column",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )

    for k in 1:4
        r = ((k - 1) ÷ 2) + 1
        c = ((k - 1) % 2) + 1
        ax = Mke.Axis(
            fig[r, c]; title = bin_titles[k], subtitle = bin_subs[k],
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

Four native corrected bins → fully inlined ``K=4`` profiled Cong → Lee
T-LBF → identical common-kernel FBP → Kalender ACNR → analytical VMI.
This is the canonical notebook 04 framework applied to the QRM thorax phantom.
"""

# ╔═╡ 08030005-0000-4000-8000-000000000001
md"""
### 01. Standard Four-Bin Corrected Counts

All four native corrected PCCT bins and all native detector rows remain
separate through decomposition. This matters for the finite 4-mm QRM volume:
the outer rows of the nominal 5-mm acquisition are not repeated measurements
of the same complete object ray, so count-domain row summation would introduce
axial partial-volume bias. No spectral bins or detector rows are merged.
"""

# ╔═╡ 08030005-0000-4000-8000-000000000010
begin
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

    nchannel_basis = let
        E = Float32.(sim_raw.energies)
        Φ = Float32.(sim_raw.W_applied)
        μρ_I = Float32[
            BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, Float64(e))
            for e in E
        ]
        μρ_W = Float32[
            BS.compute_mass_μ_at_energy(BS.XA.Materials.water, Float64(e))
            for e in E
        ]
        I0_from_Φ = vec(sum(Float64.(Φ); dims = 1))
        I0_relerr = maximum(
            abs.(I0_from_Φ .- Float64.(sim_raw.I0_bins)) ./
            max.(Float64.(sim_raw.I0_bins), eps(Float64)),
        )
        I0_relerr < 5e-5 || error(
            "Applied response and I0 disagree (max relative error = $(I0_relerr)).",
        )
        Φsum = vec(sum(Φ; dims = 1))
        μI_eff = Float32[
            sum(view(Φ, :, k) .* μρ_I) / Φsum[k]
            for k in axes(Φ, 2)
        ]
        μW_eff = Float32[
            sum(view(Φ, :, k) .* μρ_W) / Φsum[k]
            for k in axes(Φ, 2)
        ]
        (
            E = E, Φ = Φ, μρ_I = μρ_I, μρ_W = μρ_W,
            I0 = Float32.(sim_raw.I0_bins),
            μI_eff = μI_eff, μW_eff = μW_eff,
            normal_II = sum(abs2, μI_eff),
            normal_IW = sum(μI_eff .* μW_eff),
            normal_WW = sum(abs2, μW_eff),
            I0_relerr = I0_relerr,
        )
    end

    nchannel_slab_counts = let
        available_rows = size(sim_raw.bins_raw[1], 2)
        phantom_thickness_mm = size(phantom_cpu.mask, 3) *
            phantom_cpu.voxel_size[3] * 10
        # Unlike notebook 04's z-invariant cylinder, this finite 4-mm QRM
        # volume does not represent the same ray over the full nominal 5-mm
        # aperture. Preserve every acquired detector row separately rather
        # than averaging partially illuminated edge rows into one projection.
        selected_rows = 1:available_rows
        nrows = length(selected_rows)
        row_positions = (
            collect(selected_rows) .- (available_rows + 1) / 2
        ) .* sim_raw.geom.pixel_row_size
        cone_scales = sqrt.(1 .+ (row_positions ./ sim_raw.geom.SAD).^2)
        bins = [
            Float32.(bin[:, selected_rows, :]) for bin in sim_raw.bins_raw
        ]
        (
            bins = bins, nrows = nrows, selected_rows = selected_rows,
            available_rows = available_rows, cone_scales = cone_scales,
            max_cone_relerr = maximum(abs.(cone_scales .- 1)),
            thickness_mm = nrows * 10 * sim_raw.geom.pixel_row_size,
            phantom_thickness_mm = phantom_thickness_mm,
            count_scale = 1.0f0,
            row_handling = :native_rows,
        )
    end
end

# ╔═╡ 08030005-0000-4000-8000-000000000030
let
    n_row = size(first(nchannel_slab_counts.bins), 2)
    mid_r = n_row ÷ 2 + 1
    slices = [
        permutedims(bin[:, mid_r, :], (2, 1))
        for bin in nchannel_slab_counts.bins
    ]
    all_v = vcat(vec.(slices)...)
    sino_window = (
        Float64(quantile(all_v, 0.01)),
        Float64(quantile(all_v, 0.99)),
    )
    fig = Mke.Figure(size = (1180, 1180))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "View", ylabel = "Detector Column",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )
    for k in eachindex(slices)
        r = (k - 1) ÷ 2 + 1
        c = (k - 1) % 2 + 1
        ax = Mke.Axis(
            fig[r, c]; title = "Native Bin $(k)",
            subtitle = "corrected count-domain row combination",
            axis_kwargs...,
        )
        Mke.heatmap!(
            ax, slices[k]; colormap = :viridis, colorrange = sino_window,
        )
    end
    Mke.Colorbar(
        fig[1:2, 3]; colormap = :viridis, colorrange = sino_window,
        label = "Log Line Integral", width = 16,
        labelsize = 22, ticklabelsize = 18,
    )
    fig
end

# ╔═╡ 08030007-0000-4000-8000-000000000001
md"""
### 02. Four-Bin Profiled Cong Material Decomposition

The same fully inlined generalized profiled likelihood used in notebook 04
runs here at ``K=4``. Every native corrected bin and its absolute applied
MC-detector response enter the solve; no two-bin surrogate is formed.
"""

# ╔═╡ 08030007-0000-4000-8000-000000000010
begin
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
    normal_II::Float32, normal_IW::Float32, normal_WW::Float32, controls,
) where {K}
    nE = length(μρ_I)
    A_lo, A_hi = controls.iodine_bounds
    C_lo, C_hi = controls.water_bounds
    n_outer, n_inner = controls.outer_iterations, controls.inner_iterations
    A_step, C_step = controls.max_iodine_step, controls.max_water_step
    parameter_tolerance = controls.parameter_tolerance
    fisher_condition_limit = controls.fisher_condition_limit
    air_gate = controls.air_gate

    BS.AK.foreachindex(sino_I) do idx
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
            rhs_I += μI_eff[k]*hs[k][idx]
            rhs_W += μW_eff[k]*hs[k][idx]
        end
        det0_raw = normal_II*normal_WW - normal_IW*normal_IW
        initializer_valid = isfinite(det0_raw) && det0_raw > 1f-12
        det0 = initializer_valid ? det0_raw : 1f0
        A = initializer_valid ?
            clamp((normal_WW*rhs_I-normal_IW*rhs_W)/det0,A_lo,A_hi) :
            clamp(0f0,A_lo,A_hi)
        C = initializer_valid ?
            clamp((normal_II*rhs_W-normal_IW*rhs_I)/det0,C_lo,C_hi) :
            clamp(20f0,C_lo,C_hi)

        # Guaranteed monotone aggregate equation, used here only to stabilize
        # the fast solver's initial water value at its current iodine value.
        y_total=0f0
        for k in 1:K
            y_total += max(I0[k]*exp(-hs[k][idx]),1f-6)
        end
        croot_lo,croot_hi=C_lo,C_hi
        total_lo,total_hi=0f0,0f0
        for k in 1:K, e in 1:nE
            total_lo += Φ[e,k]*exp(-μρ_I[e]*A-μρ_W[e]*croot_lo)
            total_hi += Φ[e,k]*exp(-μρ_I[e]*A-μρ_W[e]*croot_hi)
        end
        aggregate_bracketed=total_lo≥y_total && total_hi≤y_total
        attainable_max,attainable_min=0f0,0f0
        for k in 1:K, e in 1:nE
            attainable_max += Φ[e,k]*exp(
                -μρ_I[e]*A_lo-μρ_W[e]*C_lo,
            )
            attainable_min += Φ[e,k]*exp(
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
                    total_mid += Φ[e,k]*exp(-μρ_I[e]*A-μρ_W[e]*mid)
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
                        z = Φ[e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                        λ += z
                        dC -= μρ_W[e] * z
                    end
                    λ = max(λ, 1f-6)
                    # Corrected counts may be fractional after detector correction.
                    y = max(I0[k]*exp(-hs[k][idx]),1f-6)
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
                    z = Φ[e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                    λ += z
                    dA -= μρ_I[e] * z
                    dC -= μρ_W[e] * z
                end
                λ = max(λ, 1f-6)
                y = max(I0[k]*exp(-hs[k][idx]),1f-6)
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
                    z = Φ[e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                    λ += z
                    dC -= μρ_W[e] * z
                end
                λ = max(λ, 1f-6)
                y = max(I0[k]*exp(-hs[k][idx]),1f-6)
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
                z = Φ[e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                λ += z
                dA -= μρ_I[e]*z
                dC -= μρ_W[e]*z
            end
            λ = max(λ,1f-6)
            y = max(I0[k]*exp(-hs[k][idx]),1f-6)
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

# ╔═╡ 08030007-0000-4000-8000-000000000020
sino_basis = let
    shape = size(nchannel_slab_counts.bins[1])
    sino_I = Array{Float32}(undef,shape)
    sino_W = Array{Float32}(undef,shape)
    flags = Array{UInt8}(undef,shape)
    score_norm = Array{Float32}(undef,shape)
    fisher_AA = Array{Float32}(undef,shape)
    fisher_AC = Array{Float32}(undef,shape)
    fisher_CC = Array{Float32}(undef,shape)
    outer_iterations = Array{UInt8}(undef,shape)
    inner_iterations = Array{UInt8}(undef,shape)
    scale = nchannel_slab_counts.count_scale
    Φ_gpu = to_gpu(scale .* nchannel_basis.Φ)
    μρ_I_gpu = to_gpu(nchannel_basis.μρ_I)
    μρ_W_gpu = to_gpu(nchannel_basis.μρ_W)
    I0_gpu = to_gpu(scale .* nchannel_basis.I0)
    μI_eff_gpu = to_gpu(nchannel_basis.μI_eff)
    μW_eff_gpu = to_gpu(nchannel_basis.μW_eff)
    elapsed = @elapsed for vrange in BS.tile_ranges(
        shape[3],nchannel_controls.tile_views,
    )
        hs = [
            to_gpu(Float32.(nchannel_slab_counts.bins[k][:,:,vrange]))
            for k in eachindex(nchannel_slab_counts.bins)
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
            nchannel_basis.normal_II,nchannel_basis.normal_IW,
            nchannel_basis.normal_WW,nchannel_controls,
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
    # Recompute the aggregate feasibility bit on the host from the global
    # attainable count range. This is independent of the initializer's
    # single-A bracket and avoids backend-specific boolean lowering in QC.
    Φ_host=scale.*Float64.(nchannel_basis.Φ)
    attainable_max=sum(nchannel_forward(
        first(nchannel_controls.iodine_bounds),
        first(nchannel_controls.water_bounds),
        Φ_host,nchannel_basis.μρ_I,nchannel_basis.μρ_W,
    ).λ)
    attainable_min=sum(nchannel_forward(
        last(nchannel_controls.iodine_bounds),
        last(nchannel_controls.water_bounds),
        Φ_host,nchannel_basis.μρ_I,nchannel_basis.μρ_W,
    ).λ)
    ytotal=zeros(Float64,shape)
    for k in eachindex(nchannel_slab_counts.bins)
        ytotal .+= scale*nchannel_basis.I0[k].*
            exp.(-Float64.(nchannel_slab_counts.bins[k]))
    end
    feasible=(ytotal.≤attainable_max).&(ytotal.≥attainable_min)
    flags[feasible] .&= 0xef
    flags[.!feasible] .|= 0x10
    (
        sino_iodine=sino_I,sino_water=sino_W,quality_flag=flags,
        fisher=(AA=fisher_AA,AC=fisher_AC,CC=fisher_CC),
        score_norm,outer_iterations,inner_iterations,
        geom=sim_raw.geom,elapsed_s=elapsed,
    )
end

# ╔═╡ 08030007-0000-4000-8000-000000000040
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

# ╔═╡ 08030007-0000-4000-8000-000000000050
md"""
### 03. Lee Total-Likelihood Bilateral Filter

The completed four-bin Cong water/iodine pair is filtered jointly with the
canonical Lee-2025 ``5\times5`` T-LBF: ``\alpha_1=0.9`` and
``\alpha_2=24.636``. The collapsed total likelihood determines only the
shared denoising weights; it is never used to estimate the two materials.
"""

# ╔═╡ 08030007-0000-4000-8000-000000000060
begin
begin
function total_measured_counts(slab_bins,I0,count_scale)
    total=zeros(Float32,size(first(slab_bins)))
    for bin in eachindex(slab_bins)
        @. total+=Float32(count_scale*I0[bin])*exp(-slab_bins[bin])
    end
    total
end

function total_expected_counts(
    sino_iodine,sino_water,Φ_total,μI,μW,
)
    I_gpu=to_gpu(Float32.(sino_iodine))
    W_gpu=to_gpu(Float32.(sino_water))
    Φ_gpu=to_gpu(Float32.(Φ_total))
    μI_gpu=to_gpu(Float32.(μI))
    μW_gpu=to_gpu(Float32.(μW))
    output_gpu=similar(I_gpu)
    try
        BS.AK.foreachindex(output_gpu) do idx
            total=0f0
            @inbounds for energy in eachindex(Φ_gpu)
                total+=Φ_gpu[energy]*exp(
                    -μI_gpu[energy]*I_gpu[idx]-
                    μW_gpu[energy]*W_gpu[idx],
                )
            end
            output_gpu[idx]=max(total,1f-6)
        end
        Array(output_gpu)
    finally
        BS.release_backend!((
            I_gpu,W_gpu,Φ_gpu,μI_gpu,μW_gpu,output_gpu,
        ))
    end
end

function tlbf_filter_pair(
    sino_iodine,sino_water,expected,measured,alpha2;
    alpha1=0.9,
)
    I_gpu=to_gpu(Float32.(sino_iodine))
    W_gpu=to_gpu(Float32.(sino_water))
    expected_gpu=to_gpu(Float32.(expected))
    measured_gpu=to_gpu(Float32.(measured))
    output_I=similar(I_gpu)
    output_W=similar(W_gpu)
    nchannel,nrow,nview=size(sino_iodine)
    a1=Float32(alpha1)
    a2=Float32(alpha2)
    try
        BS.AK.foreachindex(output_I) do idx
            channel=Int32(mod1(idx,nchannel))
            row=Int32(mod1(cld(idx,nchannel),nrow))
            view=Int32(cld(idx,nchannel*nrow))
            center_expected=max(expected_gpu[idx],1f-6)
            Y=measured_gpu[idx]
            center_likelihood=
                -center_expected+Y*log(center_expected)
            weight_sum=0f0
            iodine_sum=0f0
            water_sum=0f0
            for dv in Int32(-2):Int32(2),dc in Int32(-2):Int32(2)
                neighbor_channel=channel+dc
                (
                    neighbor_channel<Int32(1)||
                    neighbor_channel>Int32(nchannel)
                )&&continue
                neighbor_view=mod1(view+dv,Int32(nview))
                neighbor_idx=
                    Int(neighbor_channel)+
                    (Int(row)-1)*nchannel+
                    (Int(neighbor_view)-1)*nchannel*nrow
                spatial=exp(
                    -Float32(dc*dc+dv*dv)/(2f0*a1*a1),
                )
                likelihood_weight=if isinf(a2)
                    1f0
                elseif a2≤0f0
                    dc==0&&dv==0 ? 1f0 : 0f0
                else
                    candidate_expected=max(
                        expected_gpu[neighbor_idx],1f-6,
                    )
                    candidate_likelihood=
                        -candidate_expected+Y*log(candidate_expected)
                    delta=candidate_likelihood-center_likelihood
                    exp(-(delta*delta)/(a2*a2))
                end
                weight=spatial*likelihood_weight
                weight_sum+=weight
                iodine_sum+=weight*I_gpu[neighbor_idx]
                water_sum+=weight*W_gpu[neighbor_idx]
            end
            inverse_weight=1f0/max(weight_sum,eps(Float32))
            output_I[idx]=iodine_sum*inverse_weight
            output_W[idx]=water_sum*inverse_weight
        end
        (
            sino_iodine=Array(output_I),
            sino_water=Array(output_W),
        )
    finally
        BS.release_backend!((
            I_gpu,W_gpu,expected_gpu,measured_gpu,output_I,output_W,
        ))
    end
end

end



qrm_pcct_tlbf = let
    # Fixed Lee-2025 T-LBF configuration selected in notebook 04d.
    alpha1=0.9
    alpha2=24.635648571666497
    count_scale=nchannel_slab_counts.count_scale
    Φ_total=count_scale.*vec(sum(Float64.(nchannel_basis.Φ);dims=2))
    measured=total_measured_counts(
        nchannel_slab_counts.bins,nchannel_basis.I0,count_scale,
    )
    expected=total_expected_counts(
        sino_basis.sino_iodine,
        sino_basis.sino_water,
        Φ_total,nchannel_basis.μρ_I,nchannel_basis.μρ_W,
    )
    sinograms=tlbf_filter_pair(
        sino_basis.sino_iodine,
        sino_basis.sino_water,
        expected,measured,alpha2;alpha1,
    )
    (
        alpha1,alpha2,sinograms,
        implementation=:Lee_2025_joint_total_likelihood,
        shared_material_weights=true,
        window=(5,5),
    )
end
end

# ╔═╡ 08030008-0000-4000-8000-000000000001
md"""
### 04. Common-Kernel FBP Basis Maps

The Lee-filtered water and iodine sinograms use the same `SoftFilter` and
the same deterministic angular anti-alias response. Output volumes are in
basis-density units (g/cm³).
"""

# ╔═╡ 08030008-0000-4000-8000-000000000010
basis_volumes = let
    matrix_size = recon_opts.matrix_size
    geom = sino_basis.geom
    nview = size(qrm_pcct_tlbf.sinograms.sino_water, 3)
    pass_mode = min(nview ÷ 2, ceil(Int, π * matrix_size[1] / 4))
    angular_response = [
        let mode = min(j - 1, nview - (j - 1))
            mode ≤ pass_mode ? 1.0 :
            0.5 * (1 + cos(
                π * (mode - pass_mode) / (nview ÷ 2 - pass_mode),
            ))
        end
        for j in 1:nview
    ]
    function _fbp(sino_native_rows)
        spectrum = BS.FFTW.fft(Float64.(sino_native_rows), 3)
        antialiased = Float32.(real.(BS.FFTW.ifft(
            spectrum .* reshape(angular_response, 1, 1, nview), 3,
        )))
        size(antialiased, 2) == geom.n_rows || error(
            "Native-row sinogram and geometry row counts disagree.",
        )
        sino_gpu = to_gpu(antialiased)
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, geom, matrix_size; filter = BS.SoftFilter(),
        )
        try
            Float32.(Array(BS.reconstruct!(ws, sino_gpu, geom)))
        finally
            BS.release_backend!(ws)
        end
    end
    (
        vol_iodine_raw = _fbp(qrm_pcct_tlbf.sinograms.sino_iodine),
        vol_water_raw = _fbp(qrm_pcct_tlbf.sinograms.sino_water),
        geom = geom, kernel = :SoftFilter,
        angular_response = angular_response, pass_mode = pass_mode,
    )
end

# ╔═╡ 08030008-0000-4000-8000-000000000030
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

# ╔═╡ 08030008-0000-4000-8000-000000000050
md"""
### 05. ACNR

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
        info = BS.apply_acnr_kalender!(
            W, I; hp_sigma_px = 1.5, window = 4,
            passes = 4, beta_max = 20.0,
        )
        @info "[ACNR · Kalender-1988 true ACNR] ρ_hp(W,I)=$(round(info.ρ_hp, digits = 3))"
    else
        @info "[ACNR] OFF (passthrough)"
    end

    (vol_iodine = I, vol_water = W, geom = basis_volumes.geom,
     acnr = (passes = 4, beta_max = 20.0, hp_sigma_px = 1.5, window = 4))
end

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

    fig = Mke.Figure(size = (1500, 540))
    ak = (titlesize = 26, subtitlesize = 18, aspect = Mke.DataAspect())
    panels = (
        (1, "Iodine basis — before ACNR", "g/cm³", before, :viridis, rng),
        (2, "Iodine basis — after ACNR", "g/cm³", after, :viridis, rng),
        (3, "REMOVED (after − before)", "must be structureless noise", removed, :balance, (-rmax, rmax)),
    )
    for (c, ttl, sub, sl, cm, cr) in panels
        ax = Mke.Axis(fig[1, c]; title = ttl, subtitle = sub, ak...)
        Mke.heatmap!(ax, sl; colormap = cm)
        Mke.hidedecorations!(ax)
    end
    Mke.Colorbar(
        fig[1, 4]; colormap = :balance, colorrange = (-rmax, rmax),
        label = "removed (g/cm³)", width = 16, labelsize = 20, ticklabelsize = 16
    )
    fig
end

# ╔═╡ 08030009-0000-4000-8000-000000000030
let
    fig = Mke.Figure(size = (1180, 580))
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

# ╔═╡ 0803000a-0000-4000-8000-000000000001
md"""
### 06. VMI Synthesis

`BS.synth_vmi_2basis(c_water, c_iodine_mg_per_mL; energy_keV)` evaluates
the textbook 2-basis linear mix (McCollough 2015) at the target keV:

```
μ(E)  = c_water(r) · (μ/ρ)_water(E) + c_iodine(r) · (μ/ρ)_iodine(E)
HU(E) = 1000 · (μ(E) − (μ/ρ)_water(E)) / (μ/ρ)_water(E)
```

VMI grid: 50, 70, 100, 140 keV — matched to notebook 07.  The
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
pcct_vmi_energies = [50.0, 70.0, 100.0, 140.0];

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

    fig = Mke.Figure(size = (1180, 1180))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    sample = vmi_HU_final[50.0]
    mid = size(sample, 3) ÷ 2

    for (k, E) in enumerate(pcct_vmi_energies)
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
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "qrm_thorax_pcct_vmi_grid.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0803000c-0000-4000-8000-000000000001
md"""
## Results

Per-rod measured vs theoretical HU at 50 / 70 / 100 / 140 keV.

- **Measured HU** = mean over an 8-px-radius circular ROI at each rod
  centroid, broadcast across all z slices.
- **Theoretical HU** = `1000 · (μ_r(E) − μ_water(E)) / μ_water(E)` from
  `BS.compute_μ_at_energy(material, E)` — pure physics, no fitting.
"""

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

    fig = Mke.Figure(size = (1400, 1320))
    hu_kwargs = (colormap = :grays, colorrange = (-200, 500))
    title_kwargs = (titlesize = 28, subtitlesize = 20)

    ax_tl = Mke.Axis(
        fig[1, 1];
        title = "70 keV VMI HU recon",
        subtitle = "z = $(z_recon) of $(size(vmi_HU_final[70.0], 3))",
        aspect = Mke.DataAspect(),
        title_kwargs...,
    )
    Mke.heatmap!(ax_tl, hu_slice; hu_kwargs...)
    Mke.hidedecorations!(ax_tl)

    ax_tr = Mke.Axis(
        fig[1, 2];
        title = "phantom_in_recon (`:nearest` resample)",
        subtitle = "$(length(unique(pir_slice))) unique labels on recon grid",
        aspect = Mke.DataAspect(),
        title_kwargs...,
    )
    Mke.heatmap!(ax_tr, Float32.(pir_slice); colormap = :tab20)
    Mke.hidedecorations!(ax_tr)

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
    sw_hu_per_keV = [_mean_hu(vmi_HU_final[E]) for E in pcct_vmi_energies]

    n_E = length(pcct_vmi_energies)
    bar_colors = [Mke.cgrad(:plasma, n_E; categorical = true)[i] for i in 1:n_E]

    ax2 = Mke.Axis(
        fig[1, 2];
        title = "Water ROI Mean HU",
        subtitle = "Per VMI Energy",
        xlabel = "VMI Energy (keV)", ylabel = "HU",
        xticks = (collect(1:n_E), ["$(Int(E))" for E in pcct_vmi_energies]),
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
pipeline (four-bin profiled Cong + T-LBF + ACNR). The displayed
50/70/100/140-keV curve is the feasibility result; no separate
energy-dependent VMI filtering is applied.
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
        α_star ≥ αf(50.0) ? "monotonic-INCREASING" :
        "U-shape (min near α=α*)"
    @info "[basis cov · heart ROI, image]  σ_water = $(round(sqrt(Vw), sigdigits = 3)) g/mL · " *
        "σ_iod = $(round(sqrt(Vi), sigdigits = 3)) mg/mL · ρ_basis = $(round(ρ_b, digits = 3)) · " *
        "water floor = $(round(1000 * sqrt(Vw), digits = 1)) HU"
    @info "  α* = $(round(α_star, digits = 1))  vs  α(50)=$(round(αf(50.0), digits = 1)), " *
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

    fig = Mke.Figure(size = (1180, 580))

    ax1 = Mke.Axis(
        fig[1, 1];
        title = "Heart-Center Noise ROI",
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
    # Mke.ylims!(ax2, 0, maximum(σs) * 1.4)

    Mke.save(
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
    fig = Mke.Figure(size = (980, 580))

    rod_colors = [
        Mke.RGBf(0.2, 0.6, 0.85),  # water    — blue
        Mke.RGBf(0.95, 0.65, 0.13),  # lipid    — orange
        Mke.RGBf(0.55, 0.3, 0.65),  # collagen — purple
        Mke.RGBf(0.85, 0.27, 0.1),  # iodine   — red
    ]

    ax = Mke.Axis(
        fig[1, 1];
        title = "Pure-Material Rods (PCCT)",
        subtitle = "50 / 70 / 100 / 140 keV",
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
        Mke.scatterlines!(
            ax, pcct_vmi_energies, vec(rod_data.measured[i, :]);
            color = c, linewidth = 2.5, markersize = 9
        )
        Mke.lines!(
            ax, pcct_vmi_energies, vec(rod_data.theoretical[i, :]);
            color = c, linewidth = 1.6, linestyle = :dash
        )
        rod_lines[i] = Mke.LineElement(color = c, linewidth = 2.5)

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
        joinpath(@__DIR__, "..", "assets", "qrm_thorax_pcct_vmi_vs_theoretical.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0803000d-0000-4000-8000-000000000090
verification = let
    sino_finite = all(isfinite, sino_basis.sino_iodine) &&
                  all(isfinite, sino_basis.sino_water)
    vmi_finite = all(
        all(isfinite, image) for image in values(vmi_HU_final)
    )
    checks = [
        (
            name = "all four native bins retained",
            value = length(nchannel_slab_counts.bins),
            pass = length(nchannel_slab_counts.bins) == 4,
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
            name = "canonical Lee T-LBF",
            value = (
                qrm_pcct_tlbf.alpha1, qrm_pcct_tlbf.alpha2,
                qrm_pcct_tlbf.window,
            ),
            pass = qrm_pcct_tlbf.implementation ==
                   :Lee_2025_joint_total_likelihood &&
                   qrm_pcct_tlbf.shared_material_weights,
        ),
        (
            name = "identical common FBP kernel",
            value = basis_volumes.kernel,
            pass = basis_volumes.kernel == :SoftFilter,
        ),
        (
            name = "water-reference basis calibration",
            value = (
                water = round(solid_water_basis.c_water, digits = 4),
                iodine = round(solid_water_basis.c_iodine, digits = 6),
            ),
            pass = 0.8 ≤ solid_water_basis.c_water ≤ 1.2 &&
                   abs(solid_water_basis.c_iodine) ≤ 0.01,
        ),
        (
            name = "canonical VMI energies",
            value = pcct_vmi_energies,
            pass = pcct_vmi_energies == [50.0, 70.0, 100.0, 140.0],
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

# ╔═╡ 0803000e-0000-4000-8000-000000000001
md"""
## Summary

```
QRM-Thorax mid-slice mask (1600 × 1100 × 20 phantom @ 0.2 mm iso,
                           rods bored at labels 9–12)
   → Forward-project (140 kVp PCCT, 4 bins, scatter-injected)
   → Per-Bin Pile-up + Scatter Correction
   → Fully inlined K=4 profiled Cong  (all native bins retained)
   → Lee 2025 joint T-LBF  (5×5, α₁=0.9, α₂=24.636)
   → Identical SoftFilter FBP  (iodine, water basis maps)
   → Kalender ACNR  (four passes, beta_max=20)
   → Monoenergetic VMI Synthesis  (textbook 2-basis, mono μρ_water divisor)
   → Per-rod Measured vs Theoretical Regression
        (water · lipid · collagen · iodine at 50 / 70 / 100 / 140 keV)
```

1:1 parity with notebook 07's QRM-Thorax pure-material pipeline,
swapping the dual-kVp GSI acquisition for a Siemens Naeotom Alpha PCCT
4-bin acquisition. All four native PCCT bins remain separate through the
generalized Cong solve. The post-decomposition path intentionally follows
notebook 04 (T-LBF → common FBP → Kalender ACNR → VMI), while notebook 07
follows the canonical dual-kVp per-basis-FBP path from notebook 03.
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
# ╠═08030007-0000-4000-8000-000000000010
# ╠═08030007-0000-4000-8000-000000000020
# ╟─08030007-0000-4000-8000-000000000040
# ╟─08030007-0000-4000-8000-000000000050
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
# ╟─0803000d-0000-4000-8000-000000000090
# ╟─0803000e-0000-4000-8000-000000000001
