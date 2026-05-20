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
    mandatory — see §5 / §6.

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

# ╔═╡ 08000003-0000-4000-8000-000000000002
const LYMPH_DIR = get(
    ENV, "BASISSIM_XCAT_LYMPH_DIR",
    "/Volumes/Molloilab/Ayemon/XCAT_lymphatic_phantom",
)

# ╔═╡ 08000003-0000-4000-8000-000000000003
const PHANTOM_PATH = joinpath(
    LYMPH_DIR,
    "Lymphatic_Phantom_32bit_real_922_922_1178.bin",
)

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
in §5 and the resample-to-0.1 mm step in §6 happen on a small enough
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

# ╔═╡ 08000005-0000-4000-8000-000000000001
md"""
## 3. Pick a time point + load its material map

73 xlsx maps from t0000s (baseline, no contrast) through t0360s in 5 s
steps. Set `TIME_SECONDS` to whichever bolus phase you want; default
60 s is solidly inside the contrast window. To do a dynamic series, wrap
§4–§9 in `for ts in 0:30:360`.
"""

# ╔═╡ 08000005-0000-4000-8000-000000000002
const TIME_SECONDS = 230     # v2 peak — peak iodine bolus in Ayemon's t-series

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
## 4. Build the materials dict

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
        nh = norm(data[1, c])
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

# ╔═╡ 08000007-0000-4000-8000-000000000001
md"""
## 5. Locate the cisterna chyli (anchor) + lymphatic ROI

Two bboxes at native 0.75 × 0.75 × 1.5 mm resolution:

* **Cisterna chyli alone** — its lowest K anchors the scan slab (the
  "bottom" of the GE Revolution acquisition).
* **All lymphatic structures** (cisterna + thoracic duct) — used in §6
  to set the tight XY crop within the slab.
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

# ╔═╡ 08000008-0000-4000-8000-000000000001
md"""
## 6. Z slab around the cisterna chyli — **keep full XY input**

The split-of-concerns we want:

* **Input phantom** keeps its **full XY extent** so simulated rays
  traverse the realistic body chord (proper attenuation, no air-only
  exit paths through the sides). The only crop on input is the **Z
  slab** anchored at the cisterna chyli, spanning `Z_COVERAGE_MM = 160 mm`
  upward — that matches Apex Elite's max single-rotation Z coverage and
  is what a real scanner would actually see.
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

# ╔═╡ 08000008-0000-4000-8000-000000000003
"""
Compute the (i, j, k) crop ranges for the Apex-Elite acquisition:

* I / J: **full XY extent** — rays must see the entire body.
* K: starts at the cisterna chyli bottom, spans `z_coverage_mm` upward,
  clamped to the phantom's Z extent.
"""
function scan_crop_indices(
        phantom::AbstractArray{T, 3},
        cisterna_bbox,
        z_coverage_mm::Real,
        native_voxel_mm::NTuple{3, Real},
    ) where {T}
    nx, ny, nz = size(phantom)
    k1 = cisterna_bbox.k[1]
    slab_voxels_z = max(1, round(Int, z_coverage_mm / native_voxel_mm[3]))
    k2 = min(nz, k1 + slab_voxels_z - 1)
    return (
        i_range = 1:nx,
        j_range = 1:ny,
        k_range = k1:k2,
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
phantom_labeled = (phantom_native === nothing || crop_idx === nothing) ? nothing : let
        cropped = phantom_native[crop_idx.i_range, crop_idx.j_range, crop_idx.k_range]
        upsampled = resample_to_voxel_size(cropped, NATIVE_VOXEL_MM, TARGET_VOXEL_MM)
        cropped = nothing
        GC.gc()
        upsampled
end;

# ╔═╡ 08000008-0000-4000-8000-000000000007
(phantom_labeled === nothing || crop_idx === nothing) ?
    md"_skipped — see §1 / §5_" : md"""
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
        md"_skipped — see §1 / §5_"
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

        CM.save(
            joinpath(@__DIR__, "..", "..", "assets", "xcat_lymphatic_crop_pipeline.png"),
            fig; px_per_unit = 2,
        )
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
    if phantom_labeled === nothing
        md"_skipped — see §1 / §5_"
    else
        nx, ny, nz = size(phantom_labeled)
        id_set = Set(UInt16.(LYMPHATIC_IDS))

        # Per-slice lymphatic mask, only over the volume we actually display
        function lymph_mask_2d(slice)
            m = falses(size(slice))
            @inbounds for i in eachindex(slice, m)
                m[i] = slice[i] in id_set
            end
            m
        end

        # Pick view planes through the lymphatic centroid
        # Find I/J centroid by accumulating into 1-D projections (cheap)
        i_hits = falses(nx); j_hits = falses(ny); k_hits = falses(nz)
        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            if phantom_labeled[i, j, k] in id_set
                i_hits[i] = true; j_hits[j] = true; k_hits[k] = true
            end
        end
        any(i_hits) || return md"_no lymphatic voxels found in upsampled volume_"

        i_mid = clamp((findfirst(i_hits) + findlast(i_hits)) ÷ 2, 1, nx)
        j_mid = clamp((findfirst(j_hits) + findlast(j_hits)) ÷ 2, 1, ny)
        k_mid = clamp((findfirst(k_hits) + findlast(k_hits)) ÷ 2, 1, nz)

        ax_slice   = phantom_labeled[:, :, k_mid]           # axial (i, j) at K = k_mid
        cor_slice  = phantom_labeled[:, j_mid, :]           # coronal (i, k) at J = j_mid
        sag_slice  = phantom_labeled[i_mid, :, :]           # sagittal (j, k) at I = i_mid

        fig = CM.Figure(size = (1700, 700))
        title_kwargs = (titlesize = 24, subtitlesize = 16)
        lymph_cmap = [CM.RGBAf(0, 0, 0, 0), CM.RGBAf(1, 0.2, 0.2, 0.95)]

        ax1 = CM.Axis(
            fig[1, 1];
            title = "Axial · K = $(k_mid)",
            subtitle = "$(nx)×$(ny) @ $(TARGET_VOXEL_MM) mm — cisterna level",
            aspect = CM.DataAspect(),
            yreversed = true,
            title_kwargs...,
        )
        CM.heatmap!(ax1, Float32.(ax_slice); colormap = :tab20)
        CM.heatmap!(ax1, Float32.(lymph_mask_2d(ax_slice));
            colormap = lymph_cmap, colorrange = (0, 1))
        CM.hidedecorations!(ax1)

        ax2 = CM.Axis(
            fig[1, 2];
            title = "Coronal · J = $(j_mid)",
            subtitle = "$(nx)×$(nz) — duct ascending along K",
            aspect = CM.DataAspect(),
            yreversed = true,
            title_kwargs...,
        )
        CM.heatmap!(ax2, Float32.(cor_slice); colormap = :tab20)
        CM.heatmap!(ax2, Float32.(lymph_mask_2d(cor_slice));
            colormap = lymph_cmap, colorrange = (0, 1))
        CM.hidedecorations!(ax2)

        ax3 = CM.Axis(
            fig[1, 3];
            title = "Sagittal · I = $(i_mid)",
            subtitle = "$(ny)×$(nz) — duct AP profile",
            aspect = CM.DataAspect(),
            yreversed = true,
            title_kwargs...,
        )
        CM.heatmap!(ax3, Float32.(sag_slice); colormap = :tab20)
        CM.heatmap!(ax3, Float32.(lymph_mask_2d(sag_slice));
            colormap = lymph_cmap, colorrange = (0, 1))
        CM.hidedecorations!(ax3)

        CM.save(
            joinpath(@__DIR__, "..", "..", "assets", "xcat_lymphatic_fov_orthogonal.png"),
            fig; px_per_unit = 2,
        )
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
phantom = (phantom_labeled === nothing || materials === nothing) ?
    nothing :
    BS.Phantom(to_gpu(phantom_labeled), materials, VOXEL_SIZE_CM);

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
    collimation_mm = 5.0,
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
# sim = phantom === nothing ? nothing : let
#         @info "Simulating lymphatic XCAT body CTA: 120 kVp / 250 mA, t = $(TIME_SECONDS) s…"
#         ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
#         BS.simulate!(ws, phantom, protocol, sim_opts)
#         result = (sino = Array(ws.sinogram), geom = ws.geom)
#         ws = nothing
#         GC.gc(true)
#         result
# end;

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

        lbl_slice = phantom_labeled[:, :, k_phantom]
        lymph_mask = falses(size(lbl_slice))
        id_set = Set(UInt16.(LYMPHATIC_IDS))
        @inbounds for j in axes(lbl_slice, 2), i in axes(lbl_slice, 1)
            lymph_mask[i, j] = lbl_slice[i, j] in id_set
        end

        hu_slice = hu_fbp[:, :, k_recon]

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

        CM.save(
            joinpath(@__DIR__, "..", "..", "assets", "xcat_lymphatic_t$(lpad(TIME_SECONDS, 4, '0'))s.png"),
            fig; px_per_unit = 2,
        )
        fig
    end
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
# ╟─08000003-0000-4000-8000-000000000001
# ╠═08000003-0000-4000-8000-000000000002
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
# ╟─08000007-0000-4000-8000-000000000001
# ╠═08000007-0000-4000-8000-000000000002
# ╠═08000007-0000-4000-8000-000000000003
# ╠═08000007-0000-4000-8000-000000000004
# ╠═08000007-0000-4000-8000-000000000005
# ╠═08000007-0000-4000-8000-000000000006
# ╠═08000007-0000-4000-8000-000000000007
# ╟─08000007-0000-4000-8000-000000000008
# ╟─08000008-0000-4000-8000-000000000001
# ╠═08000008-0000-4000-8000-000000000002
# ╠═08000008-0000-4000-8000-000000000003
# ╠═08000008-0000-4000-8000-000000000004
# ╠═08000008-0000-4000-8000-000000000005
# ╠═08000008-0000-4000-8000-000000000006
# ╟─08000008-0000-4000-8000-000000000007
# ╟─08000008-0000-4000-8000-000000000010
# ╠═08000008-0000-4000-8000-000000000011
# ╟─08000008-0000-4000-8000-000000000020
# ╠═08000008-0000-4000-8000-000000000021
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
# ╟─08000012-0000-4000-8000-000000000010
# ╠═08000012-0000-4000-8000-000000000011
# ╠═08000012-0000-4000-8000-000000000012
# ╟─08000012-0000-4000-8000-000000000013
# ╟─08000013-0000-4000-8000-000000000001
# ╠═08000013-0000-4000-8000-000000000002
