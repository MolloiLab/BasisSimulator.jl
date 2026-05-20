### A Pluto.jl notebook ###
# v0.19.0

# ╔═╡ 08000001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ 08000001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 08000001-0000-4000-8000-000000000003
using Statistics: mean, std

# ╔═╡ 08000001-0000-4000-8000-000000000004
using Unitful: @u_str, ustrip, uconvert

# ╔═╡ 08000001-0000-4000-8000-000000000010
md"""
# 08 · XCAT Lymphatic Phantom (0.1 mm) · Single-kVp GE Apex Elite

**Forward-project Ayemon's lymphatic XCAT phantom on the GE Revolution
Apex Elite, after upsampling the native 0.75 × 0.75 × 1.5 mm bin to a
0.1 mm isotropic working grid.**

The on-disk bin is a full-body adult XCAT — 922 × 922 × 1178 Float32
voxels at 0.75 × 0.75 × 1.5 mm = **69.15 × 69.15 × 176.7 cm extent**
(body centered with air margins). Organ ID labels are constant across
time; bolus dynamics live entirely in the per-time-point xlsx material
maps (every 5 s out to t0360s).

!!! danger "Whole-phantom upsample to 0.1 mm is not survivable"
    7.5 × 7.5 × 15 = 843 × upsample → 6915 × 6915 × 17670 ≈ 840 G voxels
    ≈ 1.68 TB at UInt16. A **lymphatic-bbox crop before resampling** is
    mandatory — see §3 / §6.

Lymphatic structures of interest:

| Structure       | Organ IDs                           |
|-----------------|-------------------------------------|
| Cisterna Chyli  | 1150, 1151                          |
| Thoracic Duct   | 473, 474, 475, 476, 477, 478, 479, 480 |

Pipeline (mirrors notebook 02's shape, single kVp):

```
Float32 .bin (922×922×1178 @ 0.75/0.75/1.5 mm)
   → round → UInt16 label grid
   → axis reverse to BS convention
   → cisterna chyli bbox (organ IDs 1150, 1151) — anchor
   → Apex-Elite slab: K starts at cisterna bottom, spans 16 cm up
   → lymphatic bbox restricted to slab (organ IDs 473…480, 1150-1151)
   → tight XY crop + XY_MARGIN_MM
   → resample to 0.1 mm isotropic (NN, label-preserving)
material_map_t{TTTT}s.xlsx             ← time-dependent iodine materials
   → BS.Phantom(labels, materials, voxel_size_cm)
GE Apex Elite + 120 kVp / 250 mA, recon 16 cm Z × tight XY FOV
   → BS.simulate! (eict fidelity)
   → BHC (sino + image-domain) → FDK → HU → noise floor → cupping
   → overlay lymphatic ROI on the HU recon
```

!!! info "Archived, exploratory"
    Lives in `archived/` while the loader + material handling stabilizes.
    Once happy, promote to `notebooks/08_*.jl` and render to docs.
"""

# ╔═╡ 08000002-0000-4000-8000-000000000001
md"""
## Setup
"""

# ╔═╡ 08000002-0000-4000-8000-000000000002
import BasisSimulator as BS

# ╔═╡ 08000002-0000-4000-8000-000000000003
import CairoMakie as CM

# ╔═╡ 08000002-0000-4000-8000-000000000004
import XLSX

# ╔═╡ 08000002-0000-4000-8000-000000000005
import JLD2

# ╔═╡ 08000002-0000-4000-8000-000000000010
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

# ╔═╡ 08000002-0000-4000-8000-000000000011
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 08000002-0000-4000-8000-000000000020
# Single destination for every figure this notebook writes — kept inside
# the archived/ tree so the docs build picks them up.
const FIGURES_DIR = let
    d = joinpath(@__DIR__, "figures")
    isdir(d) || mkpath(d)
    d
end

# ╔═╡ 08000003-0000-4000-8000-000000000001
md"""
## 1. Locate the phantom + material maps

The lymphatic XCAT phantom isn't bundled with the repo (organ-labeled
binary + 73 per-timepoint xlsx material maps run to ~5 GB). Point
`BASISSIM_XCAT_LYMPH_DIR` at the local install or accept the default
`/Volumes/Molloilab/Ayemon/XCAT_lymphatic_phantom`.

Expected layout:

```
\$BASISSIM_XCAT_LYMPH_DIR/
├── Lymphatic_Phantom_32bit_real_922_922_1178.bin   ← constant geometry
└── material_map/
    ├── material_map_t0000s.xlsx                     ← baseline (no contrast)
    ├── material_map_t0005s.xlsx
    ├── material_map_t0010s.xlsx
    ├── …                                            (every 5 s)
    └── material_map_t0360s.xlsx
```

The notebook short-circuits cleanly when the directory isn't present so
the static HTML still renders on a fresh checkout.
"""

# ╔═╡ 11e9885b-3c2b-45eb-a5ab-93826cc58d85
const LYMPH_DIR = get(
    ENV, "BASISSIM_XCAT_LYMPH_DIR",
    "/Volumes/Molloilab/Ayemon/XCAT_lymphatic_phantom",
)

# ╔═╡ 08000003-0000-4000-8000-000000000003
# const PHANTOM_PATH = joinpath(
#     LYMPH_DIR,
#     "Lymphatic_Phantom_32bit_real_922_922_1178.bin",
# )

PHANTOM_PATH = "/Users/daleblack/Desktop/Lymphatic_Phantom_32bit_real_922_922_1178.bin"

# ╔═╡ 08000003-0000-4000-8000-000000000004
const MATERIAL_MAP_DIR = joinpath(LYMPH_DIR, "material_map")

# ╔═╡ 08000003-0000-4000-8000-000000000005
const HAS_LYMPH = isfile(PHANTOM_PATH) && isdir(MATERIAL_MAP_DIR)

# ╔═╡ 08000003-0000-4000-8000-000000000006
HAS_LYMPH ? md"""
    **Phantom located:** `$(PHANTOM_PATH)` ($(round(filesize(PHANTOM_PATH) / 1024^3; digits = 2)) GB)

    **Material maps:** `$(MATERIAL_MAP_DIR)` ($(length(filter(endswith(".xlsx"), readdir(MATERIAL_MAP_DIR)))) xlsx files)
    """ : md"""
    !!! warning "Lymphatic phantom not found"
        Looked at `$(PHANTOM_PATH)` / `$(MATERIAL_MAP_DIR)` and one or both
        are missing. Subsequent compute cells will short-circuit to
        `nothing`.  Set `BASISSIM_XCAT_LYMPH_DIR` or mount the Z: share.
    """

# ╔═╡ 08000004-0000-4000-8000-000000000001
md"""
## 2. Load + resample to the canonical 0.1 mm isotropic grid

The bin is **Float32 column-major** at `(922, 922, 1178)`. Labels are
integer IDs encoded as floats (so we `round`), and reach up to 1151 —
**UInt16 labels**, unlike v_male_50's UInt8.

We reverse along axes (2, 3) to match the BasisSimulator convention
(X = lateral right-positive, Y = anterior-positive, Z = inferior→superior).

### Canonical grid: 0.1 mm isotropic (after bbox crop)

The on-disk bin is at **0.75 × 0.75 × 1.5 mm** — that's the native
voxel size in `NATIVE_VOXEL_MM` below. Upsampling the whole phantom to
0.1 mm iso would need ~1.68 TB, so we **do not** upsample here in §2;
this cell only loads + axis-reverses the native grid. The bbox crop
in §3 and the resample-to-0.1 mm step in §6 happen on a small enough
subvolume to be feasible.

`resample_to_voxel_size` is provided here so it's defined alongside the
loader, but it's *called* downstream after the crop.
"""

# ╔═╡ 08000004-0000-4000-8000-000000000002
const PHANTOM_DIMS = (922, 922, 1178)

# ╔═╡ 08000004-0000-4000-8000-000000000003
# Native voxel size of the source bin, in mm — describes the on-disk file.
# The lymphatic XCAT export is anisotropic: 0.75 mm in-plane, 1.5 mm slices.
const NATIVE_VOXEL_MM = (0.75, 0.75, 1.5)

# ╔═╡ 08000004-0000-4000-8000-000000000004
# Canonical grid for every downstream cell.  Don't change.
const TARGET_VOXEL_MM = 0.1

# ╔═╡ 08000004-0000-4000-8000-000000000005
function load_lymph_phantom(
        filepath::AbstractString;
        dims::NTuple{3, Int} = PHANTOM_DIMS,
    )
    cols, rows, slices = dims
    expected = cols * rows * slices * sizeof(Float32)
    actual = filesize(filepath)
    actual == expected ||
        error("Lymphatic phantom size mismatch: expected $(expected) bytes, got $(actual)")

    raw = Vector{Float32}(undef, cols * rows * slices)
    open(filepath, "r") do io
        read!(io, raw)
    end

    labels = Vector{UInt16}(undef, length(raw))
    @inbounds for i in eachindex(raw)
        labels[i] = round(UInt16, raw[i])
    end
    raw = nothing

    phantom = reshape(labels, dims)
    return reverse(phantom; dims = (2, 3))
end

# ╔═╡ 08000004-0000-4000-8000-000000000006
"""
Nearest-neighbor 3D resample of an integer-labeled grid from
`native_voxel_mm` (per-axis) to `target_voxel_mm` (isotropic).

* Preserves physical extent — output dims scale by `native/target` per axis.
* No interpolation across labels (no fractional / spurious IDs).
* Identity when every axis ratio is `1`; **upsamples** when target < native.
"""
function resample_to_voxel_size(
        phantom::AbstractArray{T, 3},
        native_voxel_mm::NTuple{3, Real},
        target_voxel_mm::Real,
    ) where {T}
    sx = native_voxel_mm[1] / target_voxel_mm
    sy = native_voxel_mm[2] / target_voxel_mm
    sz = native_voxel_mm[3] / target_voxel_mm
    (sx ≈ 1 && sy ≈ 1 && sz ≈ 1) && return phantom

    nx, ny, nz = size(phantom)
    mx = max(1, round(Int, nx * sx))
    my = max(1, round(Int, ny * sy))
    mz = max(1, round(Int, nz * sz))
    out = similar(phantom, (mx, my, mz))

    @inbounds for k in 1:mz
        kk = clamp(floor(Int, (k - 0.5) / sz) + 1, 1, nz)
        for j in 1:my
            jj = clamp(floor(Int, (j - 0.5) / sy) + 1, 1, ny)
            for i in 1:mx
                ii = clamp(floor(Int, (i - 0.5) / sx) + 1, 1, nx)
                out[i, j, k] = phantom[ii, jj, kk]
            end
        end
    end
    return out
end

# ╔═╡ 08000004-0000-4000-8000-000000000007
phantom_native = HAS_LYMPH ? load_lymph_phantom(PHANTOM_PATH) : nothing;

# ╔═╡ 08000004-0000-4000-8000-000000000008
phantom_native === nothing ? md"_skipped — see §1_" : md"""
    **Loaded at native (no resample yet):**
    $(size(phantom_native, 1)) × $(size(phantom_native, 2)) × $(size(phantom_native, 3))
    labels (UInt16), $(length(unique(phantom_native))) unique IDs
    at $(NATIVE_VOXEL_MM[1]) × $(NATIVE_VOXEL_MM[2]) × $(NATIVE_VOXEL_MM[3]) mm.
    Crop + resample to 0.1 mm happens in §6.
    """

# ╔═╡ 08000007-0000-4000-8000-000000000001
md"""
## 3. Lymphatic structures — IDs, bboxes, and pre-crop diagnostic preview

Two bboxes at native 0.75 × 0.75 × 1.5 mm resolution:

* **Cisterna chyli alone** — its lowest K anchors the scan slab (the
  "bottom" of the GE Revolution acquisition).
* **All lymphatic structures** (cisterna + thoracic duct) — used in §6
  to set the tight XY crop within the slab.

§3a and §3b below visualize the **full uncropped native phantom** with
the lymphatic ID sets highlighted, and overlay the anticipated §6 crop
bounds on a sagittal slice — so we can sanity-check labels and
orientation before any upsample/crop logic runs.
"""

# ╔═╡ 08000007-0000-4000-8000-000000000002
const CISTERNA_IDS = (1150, 1151)

# ╔═╡ 08000007-0000-4000-8000-000000000003
const THORACIC_DUCT_IDS = (473, 474, 475, 476, 477, 478, 479, 480)

# ╔═╡ 08000007-0000-4000-8000-000000000004
const LYMPHATIC_IDS = (CISTERNA_IDS..., THORACIC_DUCT_IDS...)

# ╔═╡ 08000007-0000-4000-8000-000000000005
function label_bbox(phantom::AbstractArray{T, 3}, ids::Tuple) where {T}
    id_set = Set(T.(ids))
    mask = falses(size(phantom))
    @inbounds for k in axes(phantom, 3), j in axes(phantom, 2), i in axes(phantom, 1)
        mask[i, j, k] = phantom[i, j, k] in id_set
    end
    any(mask) || return nothing
    is = findall(any(mask, dims = (2, 3))[:])
    js = findall(any(mask, dims = (1, 3))[:])
    ks = findall(any(mask, dims = (1, 2))[:])
    return (i = extrema(is), j = extrema(js), k = extrema(ks), n = count(mask))
end

# ╔═╡ 08000007-0000-4000-8000-000000000006
cisterna_bbox = phantom_native === nothing ? nothing :
    label_bbox(phantom_native, CISTERNA_IDS);

# ╔═╡ 08000007-0000-4000-8000-000000000007
lymph_bbox = phantom_native === nothing ? nothing :
    label_bbox(phantom_native, LYMPHATIC_IDS);

# ╔═╡ 08000007-0000-4000-8000-000000000008
(cisterna_bbox === nothing || lymph_bbox === nothing) ?
    md"_no cisterna/duct voxels found (or phantom not loaded)_" : md"""
    **Cisterna chyli** (anchor): K $(cisterna_bbox.k[1])..$(cisterna_bbox.k[2])
    · I $(cisterna_bbox.i[1])..$(cisterna_bbox.i[2])
    · J $(cisterna_bbox.j[1])..$(cisterna_bbox.j[2])

    **Lymphatic ROI** (cisterna + thoracic duct):

    * I: $(lymph_bbox.i[1]) … $(lymph_bbox.i[2])
      ($(round((lymph_bbox.i[2] - lymph_bbox.i[1] + 1) * NATIVE_VOXEL_MM[1] / 10; digits = 1)) cm)
    * J: $(lymph_bbox.j[1]) … $(lymph_bbox.j[2])
      ($(round((lymph_bbox.j[2] - lymph_bbox.j[1] + 1) * NATIVE_VOXEL_MM[2] / 10; digits = 1)) cm)
    * K: $(lymph_bbox.k[1]) … $(lymph_bbox.k[2])
      ($(round((lymph_bbox.k[2] - lymph_bbox.k[1] + 1) * NATIVE_VOXEL_MM[3] / 10; digits = 1)) cm)
    * voxels tagged: $(lymph_bbox.n)
    """

# ╔═╡ 08000007-0000-4000-8000-000000000100
md"""
### 3a. Full native phantom — triplanar with lymphatic highlights

Axial / coronal / sagittal of `phantom_native` at the lymphatic centroid
(in native voxels, **before** any crop or upsample). Cisterna chyli =
magenta, thoracic duct = amber. Non-lymphatic anatomy is crushed into a
muted grey band so the lymph structures pop. Confirms the hardcoded ID
sets actually pick out the structures we think they do.
"""

# ╔═╡ 08000007-0000-4000-8000-000000000101
let
    if phantom_native === nothing || lymph_bbox === nothing
        md"_skipped — see §1 / §3_"
    else
        nx, ny, nz = size(phantom_native)
        i_mid = clamp(round(Int, (lymph_bbox.i[1] + lymph_bbox.i[2]) / 2), 1, nx)
        j_mid = clamp(round(Int, (lymph_bbox.j[1] + lymph_bbox.j[2]) / 2), 1, ny)
        k_mid = clamp(round(Int, (lymph_bbox.k[1] + lymph_bbox.k[2]) / 2), 1, nz)

        is_cisterna = falses(65536)
        is_duct     = falses(65536)
        is_lymph    = falses(65536)
        for id in CISTERNA_IDS;      is_cisterna[id + 1] = true; is_lymph[id + 1] = true end
        for id in THORACIC_DUCT_IDS; is_duct[id + 1]     = true; is_lymph[id + 1] = true end

        function lymph_mask_2d(slice)
            m = falses(size(slice))
            @inbounds for i in eachindex(slice, m)
                m[i] = is_lymph[Int(slice[i]) + 1]
            end
            m
        end

        cisterna_color = CM.RGBAf(1.00, 0.20, 0.55, 1.0)
        duct_color     = CM.RGBAf(1.00, 0.80, 0.15, 1.0)
        palette        = CM.to_colormap(:glasbey_bw_n256)
        n_pal          = length(palette)
        bg             = CM.RGBAf(0.06, 0.07, 0.08, 1.0)

        function colorize(slice)
            out = Array{CM.RGBAf}(undef, size(slice))
            @inbounds for i in eachindex(slice, out)
                lbl = Int(slice[i])
                if lbl == 0
                    out[i] = bg
                elseif is_cisterna[lbl + 1]
                    out[i] = cisterna_color
                elseif is_duct[lbl + 1]
                    out[i] = duct_color
                else
                    c = palette[(hash(UInt(lbl)) % UInt(n_pal)) + 1]
                    g = 0.299f0*c.r + 0.587f0*c.g + 0.114f0*c.b
                    g = 0.22f0 + 0.30f0 * g
                    out[i] = CM.RGBAf(g, g, g, 1.0)
                end
            end
            out
        end

        halo_color    = CM.RGBAf(1.0, 1.0, 1.0, 0.35)
        outline_color = CM.RGBAf(1.0, 0.05, 0.35, 1.0)

        fig = CM.Figure(size = (700, 2400), backgroundcolor = :white, figure_padding = 6)

        function panel!(row, slice, title, subtitle)
            ax = CM.Axis(fig[row, 1];
                title        = title,
                subtitle     = subtitle,
                titlesize    = 18,
                subtitlesize = 12,
                titlealign   = :left,
                aspect       = CM.DataAspect(),
                yreversed    = true,
            )
            CM.image!(ax, colorize(slice); interpolate = false)
            mask = Float32.(lymph_mask_2d(slice))
            if any(>(0), mask)
                CM.contour!(ax, mask; levels = [0.5], color = halo_color,    linewidth = 4)
                CM.contour!(ax, mask; levels = [0.5], color = outline_color, linewidth = 1.2)
            end
            CM.hidedecorations!(ax)
            CM.hidespines!(ax)
            return ax
        end

        panel!(1, phantom_native[:, :, k_mid], "Axial (full native)",    "k = $(k_mid) / $(nz)")
        panel!(2, phantom_native[:, j_mid, :], "Coronal (full native)",  "j = $(j_mid) / $(ny)")
        panel!(3, phantom_native[i_mid, :, :], "Sagittal (full native)", "i = $(i_mid) / $(nx)")

        CM.Legend(fig[4, 1],
            [CM.PolyElement(color = cisterna_color, strokevisible = false),
             CM.PolyElement(color = duct_color,     strokevisible = false)],
            ["Cisterna chyli (IDs $(CISTERNA_IDS[1])-$(CISTERNA_IDS[end]))",
             "Thoracic duct (IDs $(THORACIC_DUCT_IDS[1])-$(THORACIC_DUCT_IDS[end]))"];
            orientation  = :horizontal,
            framevisible = false,
            labelsize    = 13,
            patchsize    = (18, 12),
            tellwidth    = false,
        )

        CM.rowsize!(fig.layout, 1, CM.Aspect(1, ny/nx))
        CM.rowsize!(fig.layout, 2, CM.Aspect(1, nz/nx))
        CM.rowsize!(fig.layout, 3, CM.Aspect(1, nz/ny))
        CM.rowgap!(fig.layout, 4)
        CM.rowgap!(fig.layout, 3, 10)
        CM.resize_to_layout!(fig)

        CM.save(joinpath(FIGURES_DIR, "xcat_lymphatic_full_phantom_triplanar.png"), fig; px_per_unit = 2)
        fig
    end
end

# ╔═╡ 08000007-0000-4000-8000-000000000200
md"""
### 3b. Sagittal + planned §6 crop bounds

Same sagittal slice as §3a, with two red dashed lines marking the K
range that §6's `scan_crop_indices` will keep. In this post-reversal
frame **low k = superior, high k = inferior**, so the slab is anchored
at the cisterna's max-k (anatomical bottom) and extends toward lower k
(superiorly) by `Z_COVERAGE_MM`:

* `k_inferior = cisterna_bbox.k[2]` — slab anchor (cisterna bottom).
* `k_superior = k_inferior − Z_COVERAGE_MM / native_dz` — slab top
  (~16 cm cranial, into the upper chest).

If the duct contour lies between the lines, the slab direction is
right. If the duct is entirely on one side, either the crop direction
or the ID constants are wrong.
"""

# ╔═╡ 08000005-0000-4000-8000-000000000001
md"""
## 4. Pick a time point + load its material map

73 xlsx maps from t0000s (baseline, no contrast) through t0360s in 5 s
steps. Set `TIME_SECONDS` to whichever bolus phase you want; default
60 s is solidly inside the contrast window. To do a dynamic series, wrap
§5–§9 in `for ts in 0:30:360`.
"""

# ╔═╡ 08000005-0000-4000-8000-000000000002
const TIME_SECONDS = 0     # v2 peak — peak iodine bolus in Ayemon's t-series

# ╔═╡ 08000005-0000-4000-8000-000000000003
const MATERIAL_MAP_PATH = joinpath(
    MATERIAL_MAP_DIR,
    "material_map_t" * lpad(TIME_SECONDS, 4, '0') * "s.xlsx",
)

# ╔═╡ 08000005-0000-4000-8000-000000000004
isfile(MATERIAL_MAP_PATH) || !HAS_LYMPH ? md"**Material map:** `$(basename(MATERIAL_MAP_PATH))`" :
    md"!!! danger \"Material map missing\"\n    `$(MATERIAL_MAP_PATH)` does not exist."

# ╔═╡ 08000006-0000-4000-8000-000000000001
md"""
## 5. Build the materials dict

Same xlsx parsing pattern as notebook 02 §3c-d: each row is an organ
with elemental mass fractions + density, derive `ZA_ratio` and the mean
excitation energy `I` from the composition via the textbook Bethe
formulas. Missing labels fall back to water so `simulate!` doesn't error.

!!! info "Ayemon's material map shape"
    The xlsx layout follows the standard XCAT material-spreadsheet shape
    (header row, then one row per organ, columns for each element +
    density + organ ID). If the column order or row count differs from
    v_male_50's `vmale_50_materials_*.xlsx`, adjust `Z_cols` / `data`
    range below.
"""

# ╔═╡ 08000006-0000-4000-8000-000000000002
const _ATOMIC_MASSES = Dict(
    1 => 1.008, 6 => 12.011, 7 => 14.007, 8 => 15.999, 11 => 22.99, 12 => 24.305,
    15 => 30.974, 16 => 32.06, 17 => 35.45, 19 => 39.098, 20 => 40.078, 26 => 55.845, 53 => 126.904,
)

# ╔═╡ 08000006-0000-4000-8000-000000000003
const _I_VALUES_EV = Dict(
    1 => 19.2, 6 => 81.0, 7 => 82.0, 8 => 95.0, 11 => 149.0, 12 => 156.0,
    15 => 173.0, 16 => 180.0, 17 => 174.0, 19 => 190.0, 20 => 191.0, 26 => 286.0, 53 => 491.0,
)

# ╔═╡ 08000006-0000-4000-8000-000000000004
function compute_ZA_ratio(comp::Dict{Int, Float64})
    Z_sum = sum(w * Z / get(_ATOMIC_MASSES, Z, Float64(Z) * 2) for (Z, w) in comp)
    A_sum = sum(values(comp))
    return Z_sum / A_sum
end

# ╔═╡ 08000006-0000-4000-8000-000000000005
function compute_mean_excitation_energy(comp::Dict{Int, Float64})
    log_I_sum = 0.0
    Z_A_sum = 0.0
    for (Z, w) in comp
        A = get(_ATOMIC_MASSES, Z, Float64(Z) * 2)
        I = get(_I_VALUES_EV, Z, 10.0 * Z)
        Z_A = w * Z / A
        log_I_sum += Z_A * log(I)
        Z_A_sum += Z_A
    end
    return exp(log_I_sum / Z_A_sum) * u"eV"
end

# ╔═╡ 08000006-0000-4000-8000-000000000006
"""Load every organ from a lymphatic material xlsx → `Dict{Int, XA.Material}`.

Reads the whole sheet as a matrix, treats row 1 as the header, and
auto-detects which columns are elements / density / organ ID by case-
insensitive name match — robust to layout drift across spreadsheets.
"""
function load_materials_from_xlsx(xlsx_path::AbstractString)
    sheet = XLSX.readxlsx(xlsx_path)[1]
    data = sheet[:]                                          # Matrix{Any}
    nrows, ncols = size(data)
    nrows >= 2 || error("xlsx has no data rows: $(xlsx_path)")

    Z_by_name = Dict(
        "h" => 1, "c" => 6, "n" => 7, "o" => 8, "na" => 11, "mg" => 12,
        "p" => 15, "s" => 16, "cl" => 17, "k" => 19, "ca" => 20, "fe" => 26, "i" => 53,
    )

    norm = v -> lowercase(replace(string(v === missing ? "" : v), r"[\s_]+" => ""))
    col_to_Z = Dict{Int, Int}()
    density_col = 0
    id_col = 0
    name_col = 0
    for c in 1:ncols
        raw = data[1, c]
        # Numeric atomic-number header (Ayemon's lymphatic xlsx + every XCAT
        # material spreadsheet from the same pipeline) — column header is
        # literally `1`, `6`, `7`, … `53` rather than `H`, `C`, `N`, … `I`.
        if raw isa Number && isinteger(raw) && 1 <= Int(raw) <= 92
            col_to_Z[c] = Int(raw)
            continue
        end
        nh = norm(raw)
        if haskey(Z_by_name, nh)
            col_to_Z[c] = Z_by_name[nh]
        elseif nh in ("density", "ρ", "rho", "densitygcm3", "densitygcc")
            density_col = c
        elseif nh in ("id", "organid", "label", "organlabel", "matid", "materialid")
            id_col = c
        elseif nh in ("name", "organ", "material", "tissue")
            name_col = c
        end
    end
    (density_col == 0 || id_col == 0) &&
        error("xlsx layout: could not find density/id columns in header $(data[1, :])")

    out = Dict{Int, BS.XA.Material}()
    for r in 2:nrows
        oid_raw = data[r, id_col]
        ρ_raw = data[r, density_col]
        (oid_raw === missing || ρ_raw === missing) && continue
        (oid_raw isa Number && ρ_raw isa Number) || continue
        oid = Int(round(Float64(oid_raw)))
        ρ = Float64(ρ_raw)
        name = name_col == 0 ? "organ_$(oid)" : String(string(data[r, name_col]))

        comp = Dict{Int, Float64}()
        for (c, Z) in col_to_Z
            v = data[r, c]
            v isa Number && v > 0 && (comp[Z] = Float64(v))
        end
        isempty(comp) && continue

        out[oid] = BS.XA.Material(
            name,
            compute_ZA_ratio(comp),
            compute_mean_excitation_energy(comp),
            ρ * u"g/cm^3",
            comp,
        )
    end
    return out
end

# ╔═╡ 08000006-0000-4000-8000-000000000007
materials = (phantom_native === nothing || !isfile(MATERIAL_MAP_PATH)) ? nothing : let
        base = load_materials_from_xlsx(MATERIAL_MAP_PATH)
        # Ayemon's xlsx skips ID 2 — that's the XCAT whole-body soft-tissue
        # background filler (≈ 79% of all non-air voxels here).  Without
        # this override it falls back to water and the body chord mass is
        # underestimated → HU biased low and BHC calibration skewed.
        base[2] = BS.XA.Materials.softtissue
        for l in unique(phantom_native)
            haskey(base, Int(l)) || (base[Int(l)] = BS.XA.Materials.water)
    end
        base
end;

# ╔═╡ 08000006-0000-4000-8000-000000000008
materials === nothing ? md"_skipped_" : md"""
    **Materials assembled:** $(length(materials)) entries
    (of which $(count(id -> haskey(materials, Int(id)) &&
        materials[Int(id)] !== BS.XA.Materials.water, unique(phantom_native)))
    matched xlsx, rest fall back to water).
    """

# ╔═╡ 08000006-0000-4000-8000-000000000009
md"""
### 5a. Material miss list — which phantom IDs fell back to water?

Re-parses the Organ ID column of the xlsx and diffs against the unique
labels present in the phantom volume, with voxel counts (sorted biggest
first) so it's obvious whether each miss is a meaningful structure or a
stray label.
"""

# ╔═╡ 08000006-0000-4000-8000-000000000010
# let
#     if phantom_native === nothing || !isfile(MATERIAL_MAP_PATH)
#         md"_skipped — see §1 / §3_"
#     else
#         # IDs present in the xlsx (cheap re-parse: just the Organ ID column).
#         sheet = XLSX.readxlsx(MATERIAL_MAP_PATH)[1]
#         xdata = sheet[:]
#         nrows_x, ncols_x = size(xdata)
#         norm = v -> lowercase(replace(string(v === missing ? "" : v), r"[\s_]+" => ""))
#         id_col = findfirst(
#             c -> norm(xdata[1, c]) in (
#                 "id", "organid", "label", "organlabel", "matid", "materialid",
#             ),
#             1:ncols_x,
#         )
#         xlsx_ids = id_col === nothing ? Set{Int}() : Set{Int}(
#             Int(round(Float64(xdata[r, id_col]))) for r in 2:nrows_x
#                 if xdata[r, id_col] isa Number
#         )

#         # Per-ID voxel counts via a 65k-bucket histogram — one pass over
#         # phantom_native (≈ 1 G voxels) is much cheaper than `unique` + a
#         # `count(==(id), ·)` per ID.
#         bin = zeros(Int, 65536)
#         @inbounds for v in phantom_native
#             bin[Int(v) + 1] += 1
#         end
#         total_vox = length(phantom_native)
#         n_present = count(>(0), bin)

#         missing_ids = [id for id in 0:65535 if bin[id + 1] > 0 && id ∉ xlsx_ids]
#         sort!(missing_ids; by = id -> -bin[id + 1])

#         if isempty(missing_ids)
#             md"**All $(n_present) phantom IDs matched xlsx.** ✓"
#         else
#             rows = [
#                 "$(lpad(id, 6))  $(lpad(bin[id + 1], 14))  " *
#                     "$(rpad(round(100 * bin[id + 1] / total_vox; digits = 4), 8))%"
#                     for id in missing_ids
#             ]
#             Markdown.parse(
#                 "**Unmatched phantom IDs:** $(length(missing_ids)) of $(n_present) " *
#                 "(rest fell back to water).\n\n" *
#                 "```\n" *
#                 "ID         voxels          % phantom\n" *
#                 "--------   --------------  ---------\n" *
#                 join(rows, "\n") *
#                 "\n```",
#             )
#         end
#     end
# end

# ╔═╡ 08000008-0000-4000-8000-000000000001
md"""
## 6. Z slab around the cisterna chyli — **keep full XY input**

The split-of-concerns we want:

* **Input phantom** keeps its **full XY extent** so simulated rays
  traverse the realistic body chord (proper attenuation, no air-only
  exit paths through the sides). The only crop on input is the **Z
  slab** anchored at the cisterna chyli's anatomical bottom, spanning
  `Z_COVERAGE_MM = 160 mm` **superiorly** (toward the thoracic duct
  outflow) — that matches Apex Elite's max single-rotation Z coverage
  and is what a real scanner would actually see. In array terms, the
  reversal in `load_lymph_phantom` makes superior = **lower** k for
  this XCAT export (see §3b), so the slab walks from
  `cisterna_bbox.k[2]` toward smaller k.
* **Recon FOV** is set tight in XY in §8 (`recon_opts.fov_cm`) — only
  the small region around the duct gets reconstructed.

So rays go through the *whole patient*; we just don't bother reconning
the parts we don't care about.

Then **resample to 0.1 mm isotropic** via nearest-neighbor.

!!! warning "Memory: 0.1 mm iso with full XY won't fit on a 16 GB M4"
    Full XY upsampled to 0.1 mm is 6915 × 6915 × 1600 ≈ 76 G voxels ≈
    152 GB. The default below keeps the upsample call so the pipeline
    *describes* the right operation, but on this Mac you'll need to
    either (a) leave `TARGET_VOXEL_MM = NATIVE_VOXEL_MM[1]` (no XY
    upsample) and let recon-grid resolution carry the fidelity, or
    (b) bump up to a coarser target (e.g. 0.3 mm) that fits.
"""

# ╔═╡ 08000008-0000-4000-8000-000000000002
const Z_COVERAGE_MM = 160.0

# ╔═╡ 08000007-0000-4000-8000-000000000201
let
    if phantom_native === nothing || cisterna_bbox === nothing || lymph_bbox === nothing
        md"_skipped — see §1 / §3_"
    else
        nx, ny, nz = size(phantom_native)
        i_mid = clamp(round(Int, (lymph_bbox.i[1] + lymph_bbox.i[2]) / 2), 1, nx)

        # Mirrors §6's scan_crop_indices — anchor at cisterna's anatomical
        # bottom (= max k in this post-reversal frame), extend superiorly
        # toward lower k by Z_COVERAGE_MM.
        k_inferior = cisterna_bbox.k[2]
        slab_voxels_z = max(1, round(Int, Z_COVERAGE_MM / NATIVE_VOXEL_MM[3]))
        k_superior = max(1, k_inferior - slab_voxels_z + 1)
        k1, k2 = k_superior, k_inferior

        is_cisterna = falses(65536)
        is_duct     = falses(65536)
        for id in CISTERNA_IDS;      is_cisterna[id + 1] = true end
        for id in THORACIC_DUCT_IDS; is_duct[id + 1]     = true end

        function lymph_mask_2d(slice)
            m = falses(size(slice))
            @inbounds for i in eachindex(slice, m)
                v = Int(slice[i])
                m[i] = is_cisterna[v + 1] | is_duct[v + 1]
            end
            m
        end

        palette        = CM.to_colormap(:glasbey_bw_n256)
        n_pal          = length(palette)
        bg             = CM.RGBAf(0.06, 0.07, 0.08, 1.0)
        cisterna_color = CM.RGBAf(1.00, 0.20, 0.55, 1.0)
        duct_color     = CM.RGBAf(1.00, 0.80, 0.15, 1.0)

        function colorize(slice)
            out = Array{CM.RGBAf}(undef, size(slice))
            @inbounds for i in eachindex(slice, out)
                lbl = Int(slice[i])
                if lbl == 0
                    out[i] = bg
                elseif is_cisterna[lbl + 1]
                    out[i] = cisterna_color
                elseif is_duct[lbl + 1]
                    out[i] = duct_color
                else
                    c = palette[(hash(UInt(lbl)) % UInt(n_pal)) + 1]
                    g = 0.299f0*c.r + 0.587f0*c.g + 0.114f0*c.b
                    g = 0.22f0 + 0.30f0 * g
                    out[i] = CM.RGBAf(g, g, g, 1.0)
                end
            end
            out
        end

        slice = phantom_native[i_mid, :, :]    # (ny, nz)

        fig = CM.Figure(size = (900, 1300), backgroundcolor = :white)
        ax = CM.Axis(fig[1, 1];
            title        = "Sagittal (full native) — planned Z slab in red",
            subtitle     = "i = $(i_mid) / $(nx)   ·   k ∈ [$(k1), $(k2)]   (= $(round((k2 - k1 + 1) * NATIVE_VOXEL_MM[3] / 10; digits = 1)) cm)",
            titlesize    = 18,
            subtitlesize = 12,
            titlealign   = :left,
            aspect       = CM.DataAspect(),
            yreversed    = true,
        )
        CM.image!(ax, colorize(slice); interpolate = false)
        mask = Float32.(lymph_mask_2d(slice))
        if any(>(0), mask)
            CM.contour!(ax, mask; levels = [0.5], color = CM.RGBAf(1, 1, 1, 0.35),    linewidth = 4)
            CM.contour!(ax, mask; levels = [0.5], color = CM.RGBAf(1, 0.05, 0.35, 1), linewidth = 1.2)
        end

        # Slice's 2nd dim (k) maps to y-axis under image! — hlines at y=k1,k2
        # become the K crop bounds as horizontal red dashes.
        CM.hlines!(ax, [k1, k2]; color = :red, linewidth = 2, linestyle = :dash)
        CM.text!(ax, 4, Float32(k1); text = "k_superior = $(k1)  (slab top — $(round(Z_COVERAGE_MM / 10; digits = 1)) cm above cisterna)",
            color = :red, align = (:left, :bottom), fontsize = 12)
        CM.text!(ax, 4, Float32(k2); text = "k_inferior = $(k2)  (cisterna anatomical bottom — slab anchor)",
            color = :red, align = (:left, :top),    fontsize = 12)

        CM.hidedecorations!(ax)
        CM.hidespines!(ax)

        CM.Legend(fig[2, 1],
            [CM.PolyElement(color = cisterna_color, strokevisible = false),
             CM.PolyElement(color = duct_color,     strokevisible = false),
             CM.LineElement(color = :red, linestyle = :dash, linewidth = 2)],
            ["Cisterna chyli", "Thoracic duct", "Planned §6 crop bounds"];
            orientation  = :horizontal,
            framevisible = false,
            labelsize    = 13,
            patchsize    = (24, 12),
            tellwidth    = false,
        )

        CM.save(joinpath(FIGURES_DIR, "xcat_lymphatic_planned_slab.png"), fig; px_per_unit = 2)
        fig
    end
end

# ╔═╡ 08000008-0000-4000-8000-000000000003
"""
Compute the (i, j, k) crop ranges for the Apex-Elite acquisition:

* I / J: **full XY extent** — rays must see the entire body.
* K: anchored at the cisterna chyli's **anatomical bottom**, spanning
  `z_coverage_mm` **superiorly** (toward the thoracic duct outflow at
  the left subclavian/jugular junction).

After `load_lymph_phantom`'s `reverse(dims=3)` the BS-convention Z axis
is *inverted* for this particular XCAT export — **low k = superior,
high k = inferior** (verified empirically against the duct→cisterna
geometry in §3b).  So:

* cisterna's anatomical bottom → `cisterna_bbox.k[2]` (max k)
* slab extends toward **lower** k by `z_coverage_mm / native_dz` voxels
"""
function scan_crop_indices(
        phantom::AbstractArray{T, 3},
        cisterna_bbox,
        z_coverage_mm::Real,
        native_voxel_mm::NTuple{3, Real},
    ) where {T}
    nx, ny, nz = size(phantom)
    k_inferior = cisterna_bbox.k[2]
    slab_voxels_z = max(1, round(Int, z_coverage_mm / native_voxel_mm[3]))
    k_superior = max(1, k_inferior - slab_voxels_z + 1)
    return (
        i_range = 1:nx,
        j_range = 1:ny,
        k_range = k_superior:k_inferior,
    )
end

# ╔═╡ 08000008-0000-4000-8000-000000000004
crop_idx = (phantom_native === nothing || cisterna_bbox === nothing) ? nothing :
    scan_crop_indices(
        phantom_native, cisterna_bbox, Z_COVERAGE_MM, NATIVE_VOXEL_MM,
    );

# ╔═╡ 08000008-0000-4000-8000-000000000005
const RECON_FOV_CM = 10.0     # tight XY recon FOV (§8 uses this)

# ╔═╡ 08000008-0000-4000-8000-000000000006
# phantom_labeled = (phantom_native === nothing || crop_idx === nothing) ? nothing : let
#         cropped = phantom_native[crop_idx.i_range, crop_idx.j_range, crop_idx.k_range]
#         upsampled = resample_to_voxel_size(cropped, NATIVE_VOXEL_MM, TARGET_VOXEL_MM)
#         cropped = nothing
#         GC.gc()
#         upsampled
# end;

# ╔═╡ 08000008-0000-4000-8000-000000000007
(phantom_labeled === nothing || crop_idx === nothing) ?
    md"_skipped — see §1 / §3_" : md"""
    **Z-slabbed native** (full XY, $(NATIVE_VOXEL_MM[1])/$(NATIVE_VOXEL_MM[2])/$(NATIVE_VOXEL_MM[3]) mm):
    I full ($(length(crop_idx.i_range)))
    × J full ($(length(crop_idx.j_range)))
    × K $(first(crop_idx.k_range))..$(last(crop_idx.k_range)) ($(length(crop_idx.k_range)))

    **Upsampled to $(TARGET_VOXEL_MM) mm iso:**
    $(size(phantom_labeled, 1)) × $(size(phantom_labeled, 2)) × $(size(phantom_labeled, 3))
    ≈ $(round(prod(size(phantom_labeled)) * 2 / 1024^3; digits = 2)) GB (UInt16)

    **Physical extent:**
    $(round(size(phantom_labeled, 1) * TARGET_VOXEL_MM / 10; digits = 2)) ×
    $(round(size(phantom_labeled, 2) * TARGET_VOXEL_MM / 10; digits = 2)) ×
    $(round(size(phantom_labeled, 3) * TARGET_VOXEL_MM / 10; digits = 2)) cm
    """

# ╔═╡ 08000008-0000-4000-8000-000000000010
md"""
### 6a. Crop pipeline at a glance

Axial slice through the middle of the cisterna's K range:

1. **Full native phantom (0.75 mm in-plane)** with the **recon FOV**
   (`RECON_FOV_CM = $(RECON_FOV_CM) cm`) drawn in red, centered on the
   phantom's lateral midpoint — that's where isocenter sits and what
   §8's recon will reconstruct. Rays traverse the whole panel; we just
   only output a small square in the middle.
2. **Z-slabbed native (0.75 mm in-plane)** — same data and voxel size,
   only the Apex-Elite Z slab around the cisterna. XY is still full
   width so ray paths through the body are realistic.
3. **Upsampled to $(TARGET_VOXEL_MM) mm isotropic** — same anatomy,
   $(round(NATIVE_VOXEL_MM[1] / TARGET_VOXEL_MM; digits = 1))× more
   voxels per in-plane axis. Edges look the same (NN preserves labels);
   what changed is the discretization for ray-tracing.
"""

# ╔═╡ 08000008-0000-4000-8000-000000000011
let
    if phantom_native === nothing || phantom_labeled === nothing || crop_idx === nothing
        md"_skipped — see §1 / §3_"
    else
        k_native = (first(crop_idx.k_range) + last(crop_idx.k_range)) ÷ 2
        k_crop = k_native - first(crop_idx.k_range) + 1
        scale_z = NATIVE_VOXEL_MM[3] / TARGET_VOXEL_MM
        k_upsamp = clamp(
            round(Int, (k_crop - 0.5) * scale_z + 0.5),
            1, size(phantom_labeled, 3),
        )

        slice_full   = phantom_native[:, :, k_native]
        slice_crop   = phantom_native[crop_idx.i_range, crop_idx.j_range, k_native]
        slice_upsamp = phantom_labeled[:, :, k_upsamp]

        # Recon FOV rectangle (centered on the phantom's lateral midpoint,
        # which is where the cropped phantom's isocenter will sit)
        recon_half_vx_x = (RECON_FOV_CM * 10 / 2) / NATIVE_VOXEL_MM[1]
        recon_half_vx_y = (RECON_FOV_CM * 10 / 2) / NATIVE_VOXEL_MM[2]
        cx = size(phantom_native, 1) / 2
        cy = size(phantom_native, 2) / 2
        fov_pts = CM.Point2f[
            (cx - recon_half_vx_x, cy - recon_half_vx_y),
            (cx + recon_half_vx_x, cy - recon_half_vx_y),
            (cx + recon_half_vx_x, cy + recon_half_vx_y),
            (cx - recon_half_vx_x, cy + recon_half_vx_y),
        ]

        fig = CM.Figure(size = (1700, 650))
        title_kwargs = (titlesize = 24, subtitlesize = 16)

        # 1. Full native + recon FOV
        ax1 = CM.Axis(
            fig[1, 1];
            title = "Full native phantom · K = $(k_native)",
            subtitle = "$(size(phantom_native, 1))×$(size(phantom_native, 2)) @ $(NATIVE_VOXEL_MM[1]) mm in-plane · red = $(RECON_FOV_CM) cm recon FOV",
            aspect = CM.DataAspect(),
            yreversed = true,
            title_kwargs...,
        )
        CM.heatmap!(ax1, Float32.(slice_full); colormap = :tab20)
        CM.poly!(
            ax1, fov_pts;
            color = :transparent, strokecolor = :red, strokewidth = 3,
        )
        CM.hidedecorations!(ax1)

        # 2. Cropped native
        ax2 = CM.Axis(
            fig[1, 2];
            title = "Cropped native · K_crop = $(k_crop)",
            subtitle = "$(length(crop_idx.i_range))×$(length(crop_idx.j_range)) @ $(NATIVE_VOXEL_MM[1]) mm — same voxels, just the bbox",
            aspect = CM.DataAspect(),
            yreversed = true,
            title_kwargs...,
        )
        CM.heatmap!(ax2, Float32.(slice_crop); colormap = :tab20)
        CM.hidedecorations!(ax2)

        # 3. Upsampled to 0.1 mm iso
        ax3 = CM.Axis(
            fig[1, 3];
            title = "Upsampled to $(TARGET_VOXEL_MM) mm iso · K = $(k_upsamp)",
            subtitle = "$(size(phantom_labeled, 1))×$(size(phantom_labeled, 2)) — 7.5× per axis, NN, labels preserved",
            aspect = CM.DataAspect(),
            yreversed = true,
            title_kwargs...,
        )
        CM.heatmap!(ax3, Float32.(slice_upsamp); colormap = :tab20)
        CM.hidedecorations!(ax3)

        CM.save(joinpath(FIGURES_DIR, "xcat_lymphatic_crop_pipeline.png"), fig; px_per_unit = 2)
        fig
    end
end

# ╔═╡ 08000008-0000-4000-8000-000000000020
md"""
### 6b. Where the lymphatics sit in the upsampled FOV

Three orthogonal views of the upsampled (0.1 mm iso) volume, with the
cisterna chyli + thoracic duct mask highlighted in red. Lets you spot-
check that the duct actually runs through the slab from top to bottom
(if it cuts out early you bumped into the end of the phantom's K extent).
"""

# ╔═╡ 08000008-0000-4000-8000-000000000021
let
    if phantom_labeled === nothing || lymph_bbox === nothing || crop_idx === nothing
        md"_skipped — see §1 / §3 / §6_"
    else
        nx, ny, nz = size(phantom_labeled)

        # Analytic centroid (was a full-volume Set-membership scan — ~76 G
        # voxel checks with hashing on the inner loop, multi-minute on a 0.1 mm
        # full-XY upsample).  `lymph_bbox` (§3) is already in native voxel
        # coords; XY is uncropped, K is cropped to `crop_idx.k_range`.  Scale
        # by NATIVE/TARGET, clamp to volume bounds.
        sx = NATIVE_VOXEL_MM[1] / TARGET_VOXEL_MM
        sy = NATIVE_VOXEL_MM[2] / TARGET_VOXEL_MM
        sz = NATIVE_VOXEL_MM[3] / TARGET_VOXEL_MM
        k_lo = first(crop_idx.k_range)
        i_nat = (lymph_bbox.i[1] + lymph_bbox.i[2]) / 2
        j_nat = (lymph_bbox.j[1] + lymph_bbox.j[2]) / 2
        k_nat_slab = ((lymph_bbox.k[1] + lymph_bbox.k[2]) / 2) - k_lo + 1
        i_mid = clamp(round(Int, i_nat * sx), 1, nx)
        j_mid = clamp(round(Int, j_nat * sy), 1, ny)
        k_mid = clamp(round(Int, k_nat_slab * sz), 1, nz)

        # Per-slice lymphatic mask — UInt16 indicator vector instead of a
        # Set so the inner loop is one array load per voxel, not a hash.
        is_lymph = falses(65536)
        for id in LYMPHATIC_IDS
            is_lymph[id + 1] = true
        end
        function lymph_mask_2d(slice)
            m = falses(size(slice))
            @inbounds for i in eachindex(slice, m)
                m[i] = is_lymph[Int(slice[i]) + 1]
            end
            m
        end

        # Label → colour.  `:tab20` only has 20 hues, so Makie's linear
        # interpolation between adjacent integer IDs produces near-identical
        # shades for adjacent labels — useless for a segmentation phantom.
        # Hash each label into a 256-colour glasbey palette so neighbouring
        # IDs are guaranteed visually distinct.  Label 0 (air) → dark grey.
        palette = CM.to_colormap(:glasbey_bw_n256)
        n_pal   = length(palette)
        bg      = CM.RGBAf(0.08, 0.09, 0.10, 1.0)
        function colorize(slice)
            out = Array{CM.RGBAf}(undef, size(slice))
            @inbounds for i in eachindex(slice, out)
                lbl = UInt(slice[i])
                out[i] = lbl == 0 ? bg :
                    let c = palette[(hash(lbl) % UInt(n_pal)) + 1]
                        CM.RGBAf(c.r, c.g, c.b, 1.0)
                    end
            end
            out
        end

        # Lymphatic highlight: translucent yellow fill + thin red contour,
        # so the duct reads against any underlying colour.
        fill_color    = CM.RGBAf(1.0, 0.95, 0.2, 0.55)
        outline_color = CM.RGBAf(0.95, 0.10, 0.10, 1.0)

        fig = CM.Figure(size = (700, 1700), backgroundcolor = :white, figure_padding = 6)

        function panel!(row, slice, title, subtitle)
            ax = CM.Axis(fig[row, 1];
                title        = title,
                subtitle     = subtitle,
                titlesize    = 18,
                subtitlesize = 12,
                titlealign   = :left,
                aspect       = CM.DataAspect(),
                yreversed    = true,
            )
            CM.image!(ax, colorize(slice); interpolate = false)
            mask = Float32.(lymph_mask_2d(slice))
            CM.heatmap!(ax, mask;
                colormap    = [CM.RGBAf(0, 0, 0, 0), fill_color],
                colorrange  = (0, 1),
                interpolate = false,
            )
            CM.contour!(ax, mask;
                levels    = [0.5],
                color     = outline_color,
                linewidth = 1.2,
            )
            CM.hidedecorations!(ax)
            CM.hidespines!(ax)
            return ax
        end

        panel!(1, phantom_labeled[:, :, k_mid], "Axial",    "k = $(k_mid)")
        panel!(2, phantom_labeled[:, j_mid, :], "Coronal",  "j = $(j_mid)")
        panel!(3, phantom_labeled[i_mid, :, :], "Sagittal", "i = $(i_mid)")

        # Slice dims plotted as (x, y):
        #   axial    [:, :, k] → nx × ny  → height/width = ny/nx
        #   coronal  [:, j, :] → nx × nz  → nz/nx
        #   sagittal [i, :, :] → ny × nz  → nz/ny
        CM.rowsize!(fig.layout, 1, CM.Aspect(1, ny/nx))
        CM.rowsize!(fig.layout, 2, CM.Aspect(1, nz/nx))
        CM.rowsize!(fig.layout, 3, CM.Aspect(1, nz/ny))

        CM.rowgap!(fig.layout, 4)
        CM.resize_to_layout!(fig)

        CM.save(joinpath(FIGURES_DIR, "xcat_lymphatic_fov_orthogonal.png"), fig; px_per_unit = 2)
        fig
    end
end

# ╔═╡ 0fe7e0c1-6423-4676-a079-270055afb485
let
    if phantom_labeled === nothing || lymph_bbox === nothing || crop_idx === nothing
        md"_skipped — see §1 / §3 / §6_"
    else
        nx, ny, nz = size(phantom_labeled)

        # Analytic centroid (was a full-volume Set-membership scan — ~76 G
        # voxel checks with hashing on the inner loop, multi-minute on a 0.1 mm
        # full-XY upsample).  `lymph_bbox` (§3) is already in native voxel
        # coords; XY is uncropped, K is cropped to `crop_idx.k_range`.  Scale
        # by NATIVE/TARGET, clamp to volume bounds.
        sx = NATIVE_VOXEL_MM[1] / TARGET_VOXEL_MM
        sy = NATIVE_VOXEL_MM[2] / TARGET_VOXEL_MM
        sz = NATIVE_VOXEL_MM[3] / TARGET_VOXEL_MM
        k_lo = first(crop_idx.k_range)
        i_nat = (lymph_bbox.i[1] + lymph_bbox.i[2]) / 2
        j_nat = (lymph_bbox.j[1] + lymph_bbox.j[2]) / 2
        k_nat_slab = ((lymph_bbox.k[1] + lymph_bbox.k[2]) / 2) - k_lo + 1
        i_mid = clamp(round(Int, i_nat * sx), 1, nx)
        j_mid = clamp(round(Int, j_nat * sy), 1, ny)
        k_mid = clamp(round(Int, k_nat_slab * sz), 1, nz)

        # Per-group lymphatic indicator vectors (UInt16 lookup — one array
        # load per voxel, no hashing in the inner loop).  Splitting cisterna
        # vs. duct lets us colour the reservoir and the channel differently.
        is_cisterna = falses(65536)
        is_duct     = falses(65536)
        is_lymph    = falses(65536)
        for id in CISTERNA_IDS;      is_cisterna[id + 1] = true; is_lymph[id + 1] = true end
        for id in THORACIC_DUCT_IDS; is_duct[id + 1]     = true; is_lymph[id + 1] = true end

        function lymph_mask_2d(slice)
            m = falses(size(slice))
            @inbounds for i in eachindex(slice, m)
                m[i] = is_lymph[Int(slice[i]) + 1]
            end
            m
        end

        # Spotlight palette.  Non-lymph anatomy is compressed into a muted
        # grey band so the lymphatics are the only colour in the frame.
        # Cisterna chyli → hot magenta (the reservoir).
        # Thoracic duct  → amber       (the channel ascending from it).
        cisterna_color = CM.RGBAf(1.00, 0.20, 0.55, 1.0)
        duct_color     = CM.RGBAf(1.00, 0.80, 0.15, 1.0)

        palette = CM.to_colormap(:glasbey_bw_n256)
        n_pal   = length(palette)
        bg      = CM.RGBAf(0.06, 0.07, 0.08, 1.0)

        function colorize(slice)
            out = Array{CM.RGBAf}(undef, size(slice))
            @inbounds for i in eachindex(slice, out)
                lbl = Int(slice[i])
                if lbl == 0
                    out[i] = bg
                elseif is_cisterna[lbl + 1]
                    out[i] = cisterna_color
                elseif is_duct[lbl + 1]
                    out[i] = duct_color
                else
                    c = palette[(hash(UInt(lbl)) % UInt(n_pal)) + 1]
                    g = 0.299f0*c.r + 0.587f0*c.g + 0.114f0*c.b
                    g = 0.22f0 + 0.30f0 * g          # crush context into a muted band
                    out[i] = CM.RGBAf(g, g, g, 1.0)
                end
            end
            out
        end

        halo_color    = CM.RGBAf(1.0, 1.0, 1.0, 0.35)
        outline_color = CM.RGBAf(1.0, 0.05, 0.35, 1.0)

        fig = CM.Figure(size = (700, 1700), backgroundcolor = :white, figure_padding = 6)

        function panel!(row, slice, title, subtitle)
            ax = CM.Axis(fig[row, 1];
                title        = title,
                subtitle     = subtitle,
                titlesize    = 18,
                subtitlesize = 12,
                titlealign   = :left,
                aspect       = CM.DataAspect(),
                yreversed    = true,
            )
            CM.image!(ax, colorize(slice); interpolate = false)

            mask = Float32.(lymph_mask_2d(slice))
            if any(>(0), mask)
                # white halo behind...
                CM.contour!(ax, mask; levels = [0.5],
                    color = halo_color, linewidth = 4)
                # ...crisp magenta line on top
                CM.contour!(ax, mask; levels = [0.5],
                    color = outline_color, linewidth = 1.2)
            end

            CM.hidedecorations!(ax)
            CM.hidespines!(ax)
            return ax
        end

        panel!(1, phantom_labeled[:, :, k_mid], "Axial",    "k = $(k_mid)")
        panel!(2, phantom_labeled[:, j_mid, :], "Coronal",  "j = $(j_mid)")
        panel!(3, phantom_labeled[i_mid, :, :], "Sagittal", "i = $(i_mid)")

        # Self-documenting legend.
        CM.Legend(fig[4, 1],
            [CM.PolyElement(color = cisterna_color, strokevisible = false),
             CM.PolyElement(color = duct_color,     strokevisible = false)],
            ["Cisterna chyli", "Thoracic duct"];
            orientation  = :horizontal,
            framevisible = false,
            labelsize    = 13,
            patchsize    = (18, 12),
            tellwidth    = false,
        )

        # Slice dims plotted as (x, y):
        #   axial    [:, :, k] → nx × ny  → height/width = ny/nx
        #   coronal  [:, j, :] → nx × nz  → nz/nx
        #   sagittal [i, :, :] → ny × nz  → nz/ny
        CM.rowsize!(fig.layout, 1, CM.Aspect(1, ny/nx))
        CM.rowsize!(fig.layout, 2, CM.Aspect(1, nz/nx))
        CM.rowsize!(fig.layout, 3, CM.Aspect(1, nz/ny))

        CM.rowgap!(fig.layout, 4)
        CM.rowgap!(fig.layout, 3, 10)   # a little extra space before the legend
        CM.resize_to_layout!(fig)

        CM.save(joinpath(FIGURES_DIR, "xcat_lymphatic_fov_orthogonal_gray.png"), fig; px_per_unit = 2)
        fig
    end
end

# ╔═╡ 08000009-0000-4000-8000-000000000001
md"""
## 7. Build the `Phantom`

Voxel size is the canonical 0.1 mm isotropic. Origin auto-centers the
cropped subvolume at isocenter.
"""

# ╔═╡ 08000009-0000-4000-8000-000000000002
const VOXEL_SIZE_CM = (
    TARGET_VOXEL_MM / 10,
    TARGET_VOXEL_MM / 10,
    TARGET_VOXEL_MM / 10,
)

# ╔═╡ 08000009-0000-4000-8000-000000000003
# phantom = (phantom_labeled === nothing || materials === nothing) ?
#     nothing :
#     # BS.Phantom(to_gpu(phantom_labeled), materials, VOXEL_SIZE_CM);

# ╔═╡ 08000010-0000-4000-8000-000000000001
md"""
## 8. Scanner, protocol, sim & recon options

GE Revolution Apex Elite (same hardware as notebooks 01/02). Clinical
body CTA: 120 kVp / 250 mA, 1 s rotation, 500 views, 5 mm collimation.

Recon geometry **matches the crop in §6**: `z_cm = Z_COVERAGE_MM / 10`
(16 cm), `fov_cm` set just larger than the cropped XY extent so we
don't waste resolution on air. `matrix_size = (512, 512, 256)` —
the 256-slice axis lines up 1:1 with the Apex Elite detector at
0.625 mm / slice over 16 cm.
"""

# ╔═╡ 08000010-0000-4000-8000-000000000002
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
)

# ╔═╡ 08000010-0000-4000-8000-000000000003
protocol = BS.CTProtocol(
    kVp = 120,
    mA = 250.0,
    views = 500,
    rotation_time = 1.0,
    collimation_mm = 160.0,
    additional_filters = [("Al", 4.5)],
)

# ╔═╡ 08000010-0000-4000-8000-000000000004
sim_opts = BS.SimOptions(fidelity = :eict, seed = 1234)

# ╔═╡ 08000010-0000-4000-8000-000000000005
recon_opts = BS.ReconOptions(
    matrix_size = (512, 512, 256),
    fov_cm = RECON_FOV_CM,                     # tight XY (§6 constant)
    z_cm = Z_COVERAGE_MM / 10,                 # 16 cm matches the Z slab
)

# ╔═╡ 08000011-0000-4000-8000-000000000001
md"""
## 9. Forward project

Standard `create_eict_workspace` → `simulate!` → copy off GPU → drop refs
→ `GC.gc(true)` pattern. At 0.1 mm with a 30 mm margin, the cropped
phantom is a few GB on the GPU; budget several minutes on Metal /
single-digit minutes on a CUDA box.
"""

# ╔═╡ 08000011-0000-4000-8000-000000000002
sim = phantom === nothing ? nothing : let
        @info "Simulating lymphatic XCAT body CTA: 120 kVp / 250 mA, t = $(TIME_SECONDS) s…"
        ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
        BS.simulate!(ws, phantom, protocol, sim_opts)
        result = (sino = Array(ws.sinogram), geom = ws.geom)
        ws = nothing
        GC.gc(true)
        result
end;

# ╔═╡ 08000012-0000-4000-8000-000000000001
md"""
## 10. BHC calibration + FBP recon

Same calibration + FBP pipeline as notebook 02 §7–§8 (one-time BHC
spectrum fit at 120 kVp / 7 mm Al, sino-domain BHC → FDK → image-domain
BHC → HU via the BHC-calibrated `μ_water_ref` → DAS noise floor →
residual cupping). FOV is the tight 15 cm we set in `recon_opts`.
"""

# ╔═╡ 08000012-0000-4000-8000-000000000002
bhc_calibration = sim === nothing ? nothing : let
        prot_for_bhc = BS.CTProtocol(kVp = 120, additional_filters = [("Al", 4.5)])
        model = BS.calibrate_bhc_two_material(
            sim_opts, prot_for_bhc;
            scanner = scanner, geom = sim.geom,
            order = 2,
            hu_low = 450.0,
            hu_high = 600.0,
        )
        (model = model, μ_water = model.μ_water_ref, ref_E_keV = model.reference_energy_keV)
end;

# ╔═╡ 08000012-0000-4000-8000-000000000003
hu_fbp = sim === nothing ? nothing : let
        matrix_size = recon_opts.matrix_size

        sino_gpu = to_gpu(sim.sino)
        sino_bhc = BS.apply_bhc_two_material(
            sino_gpu, bhc_calibration.model, sim.geom, matrix_size,
        )
        sino_gpu = to_gpu(sino_bhc)

        ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, sim.geom, matrix_size)
        recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, sim.geom)

        BS.apply_bhc_image_domain(
            recon_μ, sim.geom, matrix_size, bhc_calibration.μ_water;
            hu_low = 50.0,
            hu_high = 150.0,
            scale_factor = 0.2,
        )

        hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc_calibration.μ_water))

        BS.add_system_noise_floor!(hu, 28.0; seed = 1234)
        BS.apply_radial_cupping_correction!(hu; fov_cm = recon_opts.fov_cm)

        ws_fdk = nothing
        sino_gpu = nothing
        recon_μ = nothing
        GC.gc(true)

        hu
end;

# ╔═╡ f095fd10-c521-484f-8944-d0381b68ca9d
size(hu_fbp)

# ╔═╡ 08000012-0000-4000-8000-000000000010
md"""
### 10b. Save the 3D recon to JLD2

Persist the HU volume plus the acquisition metadata so it can be picked
up by downstream notebooks (registration, ROI analysis, dynamic-series
viewers) without re-running the multi-minute forward + recon pipeline.

Writes to
`docs/notebooks/data/lymphatic_recons/lymphatic_recon_t{TIME}s.jld2`
— one file per `TIME_SECONDS`, so looping over the 30 s subsampled
time series leaves you with a stack of named files.

```julia
# read back later:
using JLD2
data = JLD2.load(path)
hu       = data["hu"]                   # Float32, (512, 512, 256)
t_s      = data["time_seconds"]
voxel_mm = data["recon_voxel_mm"]
```
"""

# ╔═╡ 08000012-0000-4000-8000-000000000011
const RECON_SAVE_DIR = joinpath(@__DIR__, "..", "data", "lymphatic_recons")

# ╔═╡ 08000012-0000-4000-8000-000000000012
recon_save_path = (hu_fbp === nothing || recon_opts === nothing) ? nothing : let
    isdir(RECON_SAVE_DIR) || mkpath(RECON_SAVE_DIR)
    path = joinpath(
        RECON_SAVE_DIR,
        "lymphatic_recon_t" * lpad(TIME_SECONDS, 4, '0') * "s.jld2",
    )
    recon_voxel_mm = (
        recon_opts.fov_cm * 10 / recon_opts.matrix_size[1],
        recon_opts.fov_cm * 10 / recon_opts.matrix_size[2],
        recon_opts.z_cm  * 10 / recon_opts.matrix_size[3],
    )
    JLD2.jldsave(
        path;
        hu               = hu_fbp,
        time_seconds     = TIME_SECONDS,
        recon_matrix     = collect(recon_opts.matrix_size),
        recon_fov_cm     = RECON_FOV_CM,
        recon_z_cm       = recon_opts.z_cm,
        recon_voxel_mm   = collect(recon_voxel_mm),
        z_coverage_mm    = Z_COVERAGE_MM,
        native_voxel_mm  = collect(NATIVE_VOXEL_MM),
        target_voxel_mm  = TARGET_VOXEL_MM,
        kvp              = protocol.kVp,
        mA               = protocol.mA,
        views            = protocol.views,
        bhc_ref_keV      = bhc_calibration === nothing ? NaN : bhc_calibration.ref_E_keV,
        mu_water         = bhc_calibration === nothing ? NaN : bhc_calibration.μ_water,
        lymphatic_ids    = collect(LYMPHATIC_IDS),
        cisterna_ids     = collect(CISTERNA_IDS),
        thoracic_duct_ids = collect(THORACIC_DUCT_IDS),
    )
    path
end;

# ╔═╡ 08000012-0000-4000-8000-000000000013
recon_save_path === nothing ? md"_no recon to save (see §10)_" : md"""
    **Saved 3D HU recon → JLD2**

    * Path: `$(recon_save_path)`
    * Volume: $(size(hu_fbp, 1)) × $(size(hu_fbp, 2)) × $(size(hu_fbp, 3)) Float32
      ($(round(sizeof(hu_fbp) / 1024^2; digits = 1)) MB)
    * Time point: t = $(TIME_SECONDS) s
    """

# ╔═╡ 08000013-0000-4000-8000-000000000001
md"""
## 11. Visualize — phantom labels + HU recon

Three-panel figure on the central lymphatic slice of the cropped 0.1 mm
volume:

1. Phantom labels with the cisterna chyli + thoracic duct masked on top
2. HU recon, soft-tissue window (W = 400, L = 40)
3. HU recon, contrast window (W = 800, L = 200) — bolus dynamics

Center slice is the mid-K of `phantom_labeled` (the cropped grid), so
swapping `CROP_MARGIN_MM` or the source bbox auto-recenters the panel.
"""

# ╔═╡ a0526494-2aaf-4eb3-9b39-a3724a74b33f
z_ind = 20

# ╔═╡ 08000013-0000-4000-8000-000000000002
let
    if phantom_labeled === nothing || hu_fbp === nothing
        md"_skipped — see §1–§10_"
    else
        # Center slice of the cropped 0.1 mm phantom grid
        k_phantom = size(phantom_labeled, 3) ÷ 2

        # Equivalent slice in the recon volume: phantom Z extends VOXEL_SIZE_CM[3] * size cm
        # symmetric about isocenter, recon Z extends recon_opts.z_cm.  Map by world coord.
        phantom_z_cm = VOXEL_SIZE_CM[3] * size(phantom_labeled, 3)
        z_world_cm = (k_phantom - 0.5) * VOXEL_SIZE_CM[3] - phantom_z_cm / 2
        recon_nz = recon_opts.matrix_size[3]
        recon_dz = recon_opts.z_cm / recon_nz
        k_recon = clamp(
            round(Int, z_world_cm / recon_dz + recon_nz / 2 + 0.5),
            1, recon_nz,
        )

        @info k_recon

        lbl_slice = phantom_labeled[:, :, z_ind]
        lymph_mask = falses(size(lbl_slice))
        id_set = Set(UInt16.(LYMPHATIC_IDS))
        @inbounds for j in axes(lbl_slice, 2), i in axes(lbl_slice, 1)
            lymph_mask[i, j] = lbl_slice[i, j] in id_set
        end

        hu_slice = hu_fbp[:, :, z_ind]

        fig = CM.Figure(size = (1500, 600))

        ax1 = CM.Axis(
            fig[1, 1];
            title = "Phantom labels",
            subtitle = "K = $(k_phantom) (phantom) · cisterna+duct overlaid",
            aspect = CM.DataAspect(),
            titlesize = 22, subtitlesize = 16,
        )
        CM.heatmap!(ax1, Float32.(lbl_slice); colormap = :tab20)
        CM.heatmap!(
            ax1, Float32.(lymph_mask);
            colormap = [CM.RGBAf(0, 0, 0, 0), CM.RGBAf(1, 0.2, 0.2, 0.9)],
            colorrange = (0, 1),
        )
        CM.hidedecorations!(ax1)

        ax2 = CM.Axis(
            fig[1, 2];
            title = "HU recon — soft tissue",
            subtitle = "K = $(k_recon) (recon) · W 400 / L 40",
            aspect = CM.DataAspect(),
            titlesize = 22, subtitlesize = 16,
        )
        hm2 = CM.heatmap!(ax2, hu_slice; colormap = :grays, colorrange = (-160, 240))
        CM.hidedecorations!(ax2)
        CM.Colorbar(fig[1, 3], hm2; label = "HU", width = 14, labelsize = 16)

        ax3 = CM.Axis(
            fig[1, 4];
            title = "HU recon — contrast",
            subtitle = "K = $(k_recon) · W 800 / L 200",
            aspect = CM.DataAspect(),
            titlesize = 22, subtitlesize = 16,
        )
        hm3 = CM.heatmap!(ax3, hu_slice; colormap = :grays, colorrange = (-200, 600))
        CM.hidedecorations!(ax3)
        CM.Colorbar(fig[1, 5], hm3; label = "HU", width = 14, labelsize = 16)

        CM.save(joinpath(FIGURES_DIR, "xcat_lymphatic_t$(lpad(TIME_SECONDS, 4, '0'))s.png"), fig; px_per_unit = 2)
        fig
    end
end

# ╔═╡ a7166222-4cb0-4365-aef7-ba1802831cf9
# §10c — strict-NN lymphatic mask aligned to the recon grid.
# Same recon geometry as §10's hu_fbp and every per-t recon written by §10b,
# so this single mask overlays the entire bolus series.  Labels are
# time-invariant, so we only build + save it once.
let
    nx_p, ny_p, nz_p = size(phantom_labeled)
    dx_p, dy_p, dz_p = VOXEL_SIZE_CM

    # Auto-centered phantom origin — matches the `origin = nothing` branch
    # of BS.Phantom that §7 uses.
    pox = -nx_p * dx_p / 2 + dx_p / 2
    poy = -ny_p * dy_p / 2 + dy_p / 2
    poz = -nz_p * dz_p / 2 + dz_p / 2

    Mx, My, Mz = recon_opts.matrix_size
    dxr = recon_opts.fov_cm / Mx
    dyr = recon_opts.fov_cm / My
    dzr = recon_opts.z_cm  / Mz

    # Per-axis NN lookup: which phantom voxel sits at each recon-voxel
    # center.  1-D tables — much cheaper than a 3-arg round inside the
    # inner loop.
    rec_i = Int[round(Int, (-recon_opts.fov_cm / 2 + (i - 0.5) * dxr - pox) / dx_p) + 1 for i in 1:Mx]
    rec_j = Int[round(Int, (-recon_opts.fov_cm / 2 + (j - 0.5) * dyr - poy) / dy_p) + 1 for j in 1:My]
    rec_k = Int[round(Int, (-recon_opts.z_cm  / 2 + (k - 0.5) * dzr - poz) / dz_p) + 1 for k in 1:Mz]

    # 65k-bin indicator vectors beat a Set hash in the inner loop.
    is_cist = falses(65536); for id in CISTERNA_IDS;      is_cist[id + 1] = true end
    is_duct = falses(65536); for id in THORACIC_DUCT_IDS; is_duct[id + 1] = true end

    cisterna_mask = falses(Mx, My, Mz)
    duct_mask     = falses(Mx, My, Mz)
    @inbounds for kr in 1:Mz
        kp = rec_k[kr]
        (1 <= kp <= nz_p) || continue
        for jr in 1:My
            jp = rec_j[jr]
            (1 <= jp <= ny_p) || continue
            for ir in 1:Mx
                ip = rec_i[ir]
                (1 <= ip <= nx_p) || continue
                v = Int(phantom_labeled[ip, jp, kp]) + 1
                cisterna_mask[ir, jr, kr] = is_cist[v]
                duct_mask[ir, jr, kr]     = is_duct[v]
            end
        end
    end
    lymphatic_mask = cisterna_mask .| duct_mask

    isdir(RECON_SAVE_DIR) || mkpath(RECON_SAVE_DIR)
    mask_path = joinpath(RECON_SAVE_DIR, "lymphatic_mask.jld2")
    JLD2.jldsave(
        mask_path;
        lymphatic_mask     = lymphatic_mask,
        cisterna_mask      = cisterna_mask,
        thoracic_duct_mask = duct_mask,
        recon_matrix       = collect(recon_opts.matrix_size),
        recon_fov_cm       = recon_opts.fov_cm,
        recon_z_cm         = recon_opts.z_cm,
        recon_voxel_mm     = [dxr * 10, dyr * 10, dzr * 10],
        phantom_origin_cm  = [pox, poy, poz],
        phantom_voxel_mm   = collect(VOXEL_SIZE_CM .* 10),
        phantom_dims       = collect(size(phantom_labeled)),
        cisterna_ids       = collect(CISTERNA_IDS),
        thoracic_duct_ids  = collect(THORACIC_DUCT_IDS),
        method             = "nearest_neighbor_recon_to_phantom",
    )

    md"""
    **Lymphatic mask saved → `$(mask_path)`**

    * Cisterna chyli:     $(count(cisterna_mask)) recon voxels
    * Thoracic duct:      $(count(duct_mask)) recon voxels
    * Combined lymphatic: $(count(lymphatic_mask)) recon voxels
    * Recon grid: $(Mx)×$(My)×$(Mz) @ ($(round(dxr * 10; digits = 3)), $(round(dyr * 10; digits = 3)), $(round(dzr * 10; digits = 3))) mm
    * Method: strict NN (recon-voxel center → nearest phantom voxel)
    """
end

# ╔═╡ Cell order:
# ╟─08000001-0000-4000-8000-000000000010
# ╠═08000001-0000-4000-8000-000000000001
# ╠═08000001-0000-4000-8000-000000000002
# ╠═08000001-0000-4000-8000-000000000003
# ╠═08000001-0000-4000-8000-000000000004
# ╟─08000002-0000-4000-8000-000000000001
# ╠═08000002-0000-4000-8000-000000000002
# ╠═08000002-0000-4000-8000-000000000003
# ╠═08000002-0000-4000-8000-000000000004
# ╠═08000002-0000-4000-8000-000000000005
# ╠═08000002-0000-4000-8000-000000000010
# ╟─08000002-0000-4000-8000-000000000011
# ╠═08000002-0000-4000-8000-000000000020
# ╟─08000003-0000-4000-8000-000000000001
# ╠═11e9885b-3c2b-45eb-a5ab-93826cc58d85
# ╠═08000003-0000-4000-8000-000000000003
# ╠═08000003-0000-4000-8000-000000000004
# ╠═08000003-0000-4000-8000-000000000005
# ╟─08000003-0000-4000-8000-000000000006
# ╟─08000004-0000-4000-8000-000000000001
# ╠═08000004-0000-4000-8000-000000000002
# ╠═08000004-0000-4000-8000-000000000003
# ╠═08000004-0000-4000-8000-000000000004
# ╠═08000004-0000-4000-8000-000000000005
# ╠═08000004-0000-4000-8000-000000000006
# ╠═08000004-0000-4000-8000-000000000007
# ╟─08000004-0000-4000-8000-000000000008
# ╟─08000007-0000-4000-8000-000000000001
# ╠═08000007-0000-4000-8000-000000000002
# ╠═08000007-0000-4000-8000-000000000003
# ╠═08000007-0000-4000-8000-000000000004
# ╠═08000007-0000-4000-8000-000000000005
# ╠═08000007-0000-4000-8000-000000000006
# ╠═08000007-0000-4000-8000-000000000007
# ╟─08000007-0000-4000-8000-000000000008
# ╟─08000007-0000-4000-8000-000000000100
# ╟─08000007-0000-4000-8000-000000000101
# ╟─08000007-0000-4000-8000-000000000200
# ╟─08000007-0000-4000-8000-000000000201
# ╟─08000005-0000-4000-8000-000000000001
# ╠═08000005-0000-4000-8000-000000000002
# ╠═08000005-0000-4000-8000-000000000003
# ╟─08000005-0000-4000-8000-000000000004
# ╟─08000006-0000-4000-8000-000000000001
# ╠═08000006-0000-4000-8000-000000000002
# ╠═08000006-0000-4000-8000-000000000003
# ╠═08000006-0000-4000-8000-000000000004
# ╠═08000006-0000-4000-8000-000000000005
# ╠═08000006-0000-4000-8000-000000000006
# ╠═08000006-0000-4000-8000-000000000007
# ╟─08000006-0000-4000-8000-000000000008
# ╟─08000006-0000-4000-8000-000000000009
# ╠═08000006-0000-4000-8000-000000000010
# ╟─08000008-0000-4000-8000-000000000001
# ╠═08000008-0000-4000-8000-000000000002
# ╠═08000008-0000-4000-8000-000000000003
# ╠═08000008-0000-4000-8000-000000000004
# ╠═08000008-0000-4000-8000-000000000005
# ╠═08000008-0000-4000-8000-000000000006
# ╟─08000008-0000-4000-8000-000000000007
# ╟─08000008-0000-4000-8000-000000000010
# ╟─08000008-0000-4000-8000-000000000011
# ╟─08000008-0000-4000-8000-000000000020
# ╟─08000008-0000-4000-8000-000000000021
# ╟─0fe7e0c1-6423-4676-a079-270055afb485
# ╟─08000009-0000-4000-8000-000000000001
# ╠═08000009-0000-4000-8000-000000000002
# ╠═08000009-0000-4000-8000-000000000003
# ╟─08000010-0000-4000-8000-000000000001
# ╠═08000010-0000-4000-8000-000000000002
# ╠═08000010-0000-4000-8000-000000000003
# ╠═08000010-0000-4000-8000-000000000004
# ╠═08000010-0000-4000-8000-000000000005
# ╟─08000011-0000-4000-8000-000000000001
# ╠═08000011-0000-4000-8000-000000000002
# ╟─08000012-0000-4000-8000-000000000001
# ╠═08000012-0000-4000-8000-000000000002
# ╠═08000012-0000-4000-8000-000000000003
# ╠═f095fd10-c521-484f-8944-d0381b68ca9d
# ╟─08000012-0000-4000-8000-000000000010
# ╠═08000012-0000-4000-8000-000000000011
# ╠═08000012-0000-4000-8000-000000000012
# ╟─08000012-0000-4000-8000-000000000013
# ╟─08000013-0000-4000-8000-000000000001
# ╠═a0526494-2aaf-4eb3-9b39-a3724a74b33f
# ╠═08000013-0000-4000-8000-000000000002
# ╠═a7166222-4cb0-4365-aef7-ba1802831cf9
