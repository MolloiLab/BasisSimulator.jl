### A Pluto.jl notebook ###
# v0.20.25

using Markdown
using InteractiveUtils

# ╔═╡ 08b00001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ 08b00001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 08b00001-0000-4000-8000-000000000003
using Statistics: mean, std

# ╔═╡ 08b00001-0000-4000-8000-000000000004
using Unitful: @u_str, ustrip, uconvert

# ╔═╡ 08b00001-0000-4000-8000-000000000010
md"""
# 08b · XCAT Lymphatic Phantom · Bolus Tracking (single-slice TDC)

**Pair to notebook 08.**  Same XCAT phantom, same materials handling,
same GE Apex Elite scanner — but the scan is a **single thin slice**
(5 mm collimation) parked at a fixed Z, and the simulation is **looped
across the bolus time series**.  Output: a time-density curve (TDC) of
HU in the cisterna chyli vs. time.

What's different from nb08:

| nb08 (full-anatomy scan)              | nb08b (bolus tracking)                            |
|---------------------------------------|---------------------------------------------------|
| Z slab: 160 mm around cisterna        | Z slab: 10 mm around cisterna  (tiny phantom)     |
| `collimation_mm = 160.0` (256 rows)   | `collimation_mm = 5.0` (8 rows — bolus slice)     |
| `recon z_cm = 16.0`, 256 slices       | `recon z_cm = 0.5`, 8 slices                      |
| Single `TIME_SECONDS = 230`           | Loop `TIMES_S = [0:30:210; 230]` (9 points)       |
| Output: one HU recon                  | Output: 9 HU stacks + TDC                         |

The `BOLUS_K_SLICE` constant (slice 20 of the upsampled 0.1 mm grid)
sets the Z position the scanner is parked at — the phantom origin is
shifted so this voxel sits exactly at isocenter.

!!! info "Archived, exploratory"
    Sibling of `08_xcat_lymphatic.jl`.  Promote together once both stabilize.
"""

# ╔═╡ 08b00002-0000-4000-8000-000000000001
md"""
## Setup
"""

# ╔═╡ 08b00002-0000-4000-8000-000000000002
import BasisSimulator as BS

# ╔═╡ 08b00002-0000-4000-8000-000000000003
import CairoMakie as CM

# ╔═╡ 08b00002-0000-4000-8000-000000000004
import XLSX

# ╔═╡ 08b00002-0000-4000-8000-000000000005
import JLD2

# ╔═╡ 08b00002-0000-4000-8000-000000000010
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

# ╔═╡ 08b00002-0000-4000-8000-000000000011
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 08b00003-0000-4000-8000-000000000001
md"""
## 1. Locate the phantom + material maps

Same setup as nb08 — `BASISSIM_XCAT_LYMPH_DIR` env var or default mount path.
"""

# ╔═╡ 08b00003-0000-4000-8000-000000000002
const LYMPH_DIR = get(
    ENV, "BASISSIM_XCAT_LYMPH_DIR",
    "/Volumes/Molloilab-1/Ayemon/XCAT_lymphatic_phantom",
)

# ╔═╡ 08b00003-0000-4000-8000-000000000003
const PHANTOM_PATH = joinpath(
    LYMPH_DIR,
    "Lymphatic_Phantom_32bit_real_922_922_1178.bin",
)

# ╔═╡ 08b00003-0000-4000-8000-000000000004
const MATERIAL_MAP_DIR = joinpath(LYMPH_DIR, "material_map")

# ╔═╡ 08b00003-0000-4000-8000-000000000005
const HAS_LYMPH = isfile(PHANTOM_PATH) && isdir(MATERIAL_MAP_DIR)

# ╔═╡ 08b00003-0000-4000-8000-000000000006
HAS_LYMPH ? md"**Phantom located:** `$(PHANTOM_PATH)`" : md"""
    !!! warning "Lymphatic phantom not found"
        Compute cells short-circuit to `nothing` — see nb08 §1 for the
        full mount instructions.
    """

# ╔═╡ 08b00004-0000-4000-8000-000000000001
md"""
## 2. Load + crop a **thin Z slab** around the cisterna

Identical loader to nb08, but the §6-equivalent crop here is intentionally
small in Z — `BOLUS_Z_SLAB_MM = 10 mm` of phantom is plenty for a 5 mm
collimated acquisition + small cone-beam margin.  The phantom upsampled
to 0.1 mm iso is ~100 slices in K, vs. ~1600 in nb08.
"""

# ╔═╡ 08b00004-0000-4000-8000-000000000002
const PHANTOM_DIMS = (922, 922, 1178)

# ╔═╡ 08b00004-0000-4000-8000-000000000003
const NATIVE_VOXEL_MM = (0.75, 0.75, 1.5)

# ╔═╡ 08b00004-0000-4000-8000-000000000004
const TARGET_VOXEL_MM = 0.1

# ╔═╡ 08b00004-0000-4000-8000-000000000005
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

# ╔═╡ 08b00004-0000-4000-8000-000000000006
"""Nearest-neighbor 3D resample (same as nb08)."""
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

# ╔═╡ 08b00004-0000-4000-8000-000000000007
phantom_native = HAS_LYMPH ? load_lymph_phantom(PHANTOM_PATH) : nothing;

# ╔═╡ 08b00004-0000-4000-8000-000000000008
phantom_native === nothing ? md"_skipped_" :
    md"**Native phantom:** $(size(phantom_native, 1))×$(size(phantom_native, 2))×$(size(phantom_native, 3)) UInt16"

# ╔═╡ 08b00005-0000-4000-8000-000000000001
md"""
## 3. Bolus tracking time series

`TIMES_S` defines which xlsx maps get loaded inside the time loop.
Default is every 30 s out to 210 s, plus 230 s (nb08's peak).  Tweak
freely — every value must have a `material_map_t{TTTT}s.xlsx` on disk.
"""

# ╔═╡ 08b00005-0000-4000-8000-000000000002
const TIMES_S = [collect(0:30:210); 230]

# ╔═╡ 08b00005-0000-4000-8000-000000000003
material_map_path(t::Integer) =
    joinpath(MATERIAL_MAP_DIR, "material_map_t" * lpad(t, 4, '0') * "s.xlsx")

# ╔═╡ 08b00005-0000-4000-8000-000000000004
HAS_LYMPH ? let
    missing_t = [t for t in TIMES_S if !isfile(material_map_path(t))]
    isempty(missing_t) ?
        md"**Time points:** $(length(TIMES_S)) — all xlsx files present (t = $(TIMES_S) s)" :
        md"!!! danger \"Missing xlsx files\"\n    These time points have no map: $(missing_t)"
end : md"_skipped_"

# ╔═╡ 08b00006-0000-4000-8000-000000000001
md"""
## 4. Material loaders (identical to nb08 §4)

Same atomic-mass / mean-excitation-energy helpers and the same
`load_materials_from_xlsx` parser.  Inside the time loop we call it once
per `t`, apply the **ID 2 → softtissue** override (the XCAT whole-body
filler — see nb08 §4), and water-fill the rest.
"""

# ╔═╡ 08b00006-0000-4000-8000-000000000002
const _ATOMIC_MASSES = Dict(
    1 => 1.008, 6 => 12.011, 7 => 14.007, 8 => 15.999, 11 => 22.99, 12 => 24.305,
    15 => 30.974, 16 => 32.06, 17 => 35.45, 19 => 39.098, 20 => 40.078, 26 => 55.845, 53 => 126.904,
)

# ╔═╡ 08b00006-0000-4000-8000-000000000003
const _I_VALUES_EV = Dict(
    1 => 19.2, 6 => 81.0, 7 => 82.0, 8 => 95.0, 11 => 149.0, 12 => 156.0,
    15 => 173.0, 16 => 180.0, 17 => 174.0, 19 => 190.0, 20 => 191.0, 26 => 286.0, 53 => 491.0,
)

# ╔═╡ 08b00006-0000-4000-8000-000000000004
function compute_ZA_ratio(comp::Dict{Int, Float64})
    Z_sum = sum(w * Z / get(_ATOMIC_MASSES, Z, Float64(Z) * 2) for (Z, w) in comp)
    A_sum = sum(values(comp))
    return Z_sum / A_sum
end

# ╔═╡ 08b00006-0000-4000-8000-000000000005
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

# ╔═╡ 08b00006-0000-4000-8000-000000000006
function load_materials_from_xlsx(xlsx_path::AbstractString)
    sheet = XLSX.readxlsx(xlsx_path)[1]
    data = sheet[:]
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

# ╔═╡ 08b00006-0000-4000-8000-000000000007
"""
Load materials for a given time `t` (seconds), then apply the same
fix-up nb08 does: override ID 2 with soft tissue (XCAT body filler),
and water-fill any label that's still missing from `present_ids`.
"""
function build_materials_for_time(
        t::Integer,
        present_ids,
    )
    base = load_materials_from_xlsx(material_map_path(t))
    base[2] = BS.XA.Materials.softtissue
    for l in present_ids
        haskey(base, Int(l)) || (base[Int(l)] = BS.XA.Materials.water)
    end
    return base
end

# ╔═╡ 08b00007-0000-4000-8000-000000000001
md"""
## 5. Cisterna chyli bbox (identical to nb08 §5)

Same anchor logic — we use the cisterna's lowest K to position the
bolus slab, and we use the cisterna mask later as the **TDC ROI**.
"""

# ╔═╡ 08b00007-0000-4000-8000-000000000002
const CISTERNA_IDS = (1150, 1151)

# ╔═╡ 08b00007-0000-4000-8000-000000000003
const THORACIC_DUCT_IDS = (473, 474, 475, 476, 477, 478, 479, 480)

# ╔═╡ 08b00007-0000-4000-8000-000000000004
const LYMPHATIC_IDS = (CISTERNA_IDS..., THORACIC_DUCT_IDS...)

# ╔═╡ 08b00007-0000-4000-8000-000000000005
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

# ╔═╡ 08b00007-0000-4000-8000-000000000006
cisterna_bbox = phantom_native === nothing ? nothing :
    label_bbox(phantom_native, CISTERNA_IDS);

# ╔═╡ 08b00007-0000-4000-8000-000000000007
cisterna_bbox === nothing ? md"_no cisterna voxels (or phantom not loaded)_" : md"""
    **Cisterna chyli** (anchor for the bolus slab):
    K $(cisterna_bbox.k[1])..$(cisterna_bbox.k[2])
    · I $(cisterna_bbox.i[1])..$(cisterna_bbox.i[2])
    · J $(cisterna_bbox.j[1])..$(cisterna_bbox.j[2])
    · $(cisterna_bbox.n) voxels
    """

# ╔═╡ 08b00008-0000-4000-8000-000000000001
md"""
## 6. Thin Z slab crop + upsample to 0.1 mm

Same `scan_crop_indices` logic as nb08, but with `BOLUS_Z_SLAB_MM = 10`
instead of 160 — the bolus phantom is just a thin pancake around the
cisterna level.  Full XY kept so ray attenuation through the body chord
stays realistic.

After cropping + upsampling, the phantom is roughly
`(922×7.5, 922×7.5, $(round(Int, 10/0.1)))` =
`(6915, 6915, 100)` voxels at 0.1 mm iso.

!!! warning "GPU memory at 0.1 mm with full XY"
    Same caveat as nb08: 6915² is heavy.  The bolus phantom is ~95×
    thinner in Z than nb08's, so this fits on an M-class Mac;
    on smaller GPUs drop `TARGET_VOXEL_MM` to 0.2 or 0.3.
"""

# ╔═╡ 08b00008-0000-4000-8000-000000000002
const BOLUS_Z_SLAB_MM = 10.0

# ╔═╡ 08b00008-0000-4000-8000-000000000003
function scan_crop_indices(
        phantom::AbstractArray{T, 3},
        cisterna_bbox,
        z_slab_mm::Real,
        native_voxel_mm::NTuple{3, Real},
    ) where {T}
    nx, ny, nz = size(phantom)
    k1 = cisterna_bbox.k[1]
    slab_voxels_z = max(1, round(Int, z_slab_mm / native_voxel_mm[3]))
    k2 = min(nz, k1 + slab_voxels_z - 1)
    return (i_range = 1:nx, j_range = 1:ny, k_range = k1:k2)
end

# ╔═╡ 08b00008-0000-4000-8000-000000000004
crop_idx = (phantom_native === nothing || cisterna_bbox === nothing) ? nothing :
    scan_crop_indices(phantom_native, cisterna_bbox, BOLUS_Z_SLAB_MM, NATIVE_VOXEL_MM);

# ╔═╡ 08b00008-0000-4000-8000-000000000005
const RECON_FOV_CM = 10.0

# ╔═╡ 08b00008-0000-4000-8000-000000000006
phantom_labeled = (phantom_native === nothing || crop_idx === nothing) ? nothing : let
        cropped = phantom_native[crop_idx.i_range, crop_idx.j_range, crop_idx.k_range]
        upsampled = resample_to_voxel_size(cropped, NATIVE_VOXEL_MM, TARGET_VOXEL_MM)
        cropped = nothing
        GC.gc()
        upsampled
end;

# ╔═╡ 08b00008-0000-4000-8000-000000000007
phantom_labeled === nothing ? md"_skipped_" : md"""
    **Cropped + upsampled phantom:**
    $(size(phantom_labeled, 1)) × $(size(phantom_labeled, 2)) × $(size(phantom_labeled, 3))
    @ $(TARGET_VOXEL_MM) mm iso
    ≈ $(round(prod(size(phantom_labeled)) * 2 / 1024^3; digits = 2)) GB (UInt16)
    """

# ╔═╡ 08b00009-0000-4000-8000-000000000001
md"""
## 7. Bolus scan position — shift origin so slice 20 lands at iso

`Phantom(...; origin=…)` lets us slide the phantom along Z so a specific
voxel sits at isocenter.  We point the scanner at `BOLUS_K_SLICE = 20`
of the upsampled volume — that's $(round(20 * 0.1; digits = 1)) mm
above the cisterna bottom anchor, basically dead-center on the cisterna
chyli once it fills with contrast.

* `origin_z = -(BOLUS_K_SLICE - 1) × dz` — places voxel `K = BOLUS_K_SLICE`
  at world z = 0 (= isocenter).
* `origin_x`, `origin_y` are left as the default (auto-centered).
"""

# ╔═╡ 08b00009-0000-4000-8000-000000000002
const BOLUS_K_SLICE = 20

# ╔═╡ 08b00009-0000-4000-8000-000000000003
const VOXEL_SIZE_CM = (
    TARGET_VOXEL_MM / 10,
    TARGET_VOXEL_MM / 10,
    TARGET_VOXEL_MM / 10,
)

# ╔═╡ 08b00009-0000-4000-8000-000000000004
phantom_origin = phantom_labeled === nothing ? nothing : let
    nx, ny, nz = size(phantom_labeled)
    dx, dy, dz = VOXEL_SIZE_CM
    # Default XY centering (matches the nothing-origin branch in BS.Phantom).
    origin_x = -nx * dx / 2 + dx / 2
    origin_y = -ny * dy / 2 + dy / 2
    # Z shift so voxel BOLUS_K_SLICE sits at z = 0.
    origin_z = -(BOLUS_K_SLICE - 1) * dz
    (origin_x, origin_y, origin_z)
end;

# ╔═╡ 08b00009-0000-4000-8000-000000000005
phantom_origin === nothing ? md"_skipped_" : md"""
    **Phantom origin (cm):**
    ($(round(phantom_origin[1]; digits = 3)),
    $(round(phantom_origin[2]; digits = 3)),
    $(round(phantom_origin[3]; digits = 3)))

    Voxel K = $(BOLUS_K_SLICE) sits at world Z = 0 (isocenter).
    """

# ╔═╡ 08b00009-0000-4000-8000-000000000006
# Upload the labels to GPU **once** — we'll rebuild the Phantom wrapper per t
# with new materials, but the integer label volume itself never changes.
phantom_labels_gpu = phantom_labeled === nothing ? nothing : to_gpu(phantom_labeled);

# ╔═╡ 08b00010-0000-4000-8000-000000000001
md"""
## 8. Scanner / protocol / sim & recon opts

Apex Elite hardware identical to nb08.  Protocol is the GE clinical
bolus tracking template: 120 kVp / **lower mA** (typical: ~30 mA for
TDC sensing — we use 30.0 here) and `collimation_mm = 5.0` (single
5 mm "monitoring slice", 8 detector rows active at iso).

Recon volume matches the collimation: `z_cm = 0.5`, `matrix_size[3] = 8`
gives 0.625 mm slice thickness (one detector row pitch).
"""

# ╔═╡ 08b00010-0000-4000-8000-000000000002
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

# ╔═╡ 08b00010-0000-4000-8000-000000000003
protocol = BS.CTProtocol(
    kVp = 120,
    mA = 30.0,                          # bolus-tracking dose (low)
    views = 500,
    rotation_time = 1.0,
    collimation_mm = 5.0,               # single 5 mm monitoring slice
    additional_filters = [("Al", 4.5)],
)

# ╔═╡ 08b00010-0000-4000-8000-000000000004
sim_opts = BS.SimOptions(fidelity = :eict, seed = 1234)

# ╔═╡ 08b00010-0000-4000-8000-000000000005
recon_opts = BS.ReconOptions(
    matrix_size = (512, 512, 8),        # 8 slices × 0.625 mm = 5 mm Z stack
    fov_cm = RECON_FOV_CM,
    z_cm = 0.5,
)

# ╔═╡ 08b00011-0000-4000-8000-000000000001
md"""
## 9. BHC calibration (one-time)

Same two-material BHC fit as nb08 §10.  Calibration depends on the
scanner + protocol spectrum, **not** on the phantom contents — so we
compute it once outside the time loop and reuse the model for every `t`.
"""

# ╔═╡ 08b00011-0000-4000-8000-000000000002
bhc_calibration = phantom_labels_gpu === nothing ? nothing : let
    # We need a geom matching the bolus protocol for calibration —
    # build it directly from scanner + recon_opts so we don't have to
    # run a simulation first.
    geom = BS.CTGeometry(
        scanner;
        n_angles = protocol.views,
        fov_cm = recon_opts.fov_cm,
        z_cm = recon_opts.z_cm,
        collimation_mm = protocol.collimation_mm,
    )
    prot_for_bhc = BS.CTProtocol(kVp = 120, additional_filters = [("Al", 4.5)])
    model = BS.calibrate_bhc_two_material(
        sim_opts, prot_for_bhc;
        scanner = scanner, geom = geom,
        order = 2,
        hu_low = 450.0,
        hu_high = 600.0,
    )
    (model = model, μ_water = model.μ_water_ref, ref_E_keV = model.reference_energy_keV)
end;

# ╔═╡ 08b00011-0000-4000-8000-000000000003
bhc_calibration === nothing ? md"_skipped_" : md"""
    **BHC calibrated:** ref_E = $(round(bhc_calibration.ref_E_keV; digits = 1)) keV,
    μ_water = $(round(bhc_calibration.μ_water; digits = 4)) cm⁻¹
    """

# ╔═╡ 08b00012-0000-4000-8000-000000000001
md"""
## 10. Cisterna ROI in recon coordinates

The TDC needs a per-recon-voxel cisterna mask.  We build it once from
`phantom_labeled` + `phantom_origin`, walking each recon voxel back to
the phantom-voxel that contains its center.

The mask is small enough (recon volume is 512×512×8) that a single
loop over recon voxels is fine — no need for a GPU kernel here.
"""

# ╔═╡ 08b00012-0000-4000-8000-000000000002
"""
For each recon voxel, look up the phantom label at its world coord and
flag it as cisterna if `phantom[label] ∈ CISTERNA_IDS`.

Recon volume is centered at iso (BS convention).  Phantom voxel `[i,j,k]`
has center at `phantom_origin + ((i-1)*dx, (j-1)*dy, (k-1)*dz)`.
"""
function build_cisterna_recon_mask(
        phantom_labeled::AbstractArray{<:Unsigned, 3},
        phantom_origin::NTuple{3, Float64},
        voxel_size_cm::NTuple{3, Float64},
        recon_matrix::NTuple{3, Int},
        recon_fov_cm::Real,
        recon_z_cm::Real,
        cisterna_ids::Tuple,
    )
    Mx, My, Mz = recon_matrix
    dxr = recon_fov_cm / Mx
    dyr = recon_fov_cm / My
    dzr = recon_z_cm / Mz
    dxp, dyp, dzp = voxel_size_cm
    ox, oy, oz = phantom_origin
    nxp, nyp, nzp = size(phantom_labeled)

    is_cist = falses(65536)
    for id in cisterna_ids
        is_cist[id + 1] = true
    end

    mask = falses(Mx, My, Mz)
    @inbounds for kr in 1:Mz
        zw = -recon_z_cm / 2 + (kr - 0.5) * dzr
        kp = round(Int, (zw - oz) / dzp + 1)
        (1 <= kp <= nzp) || continue
        for jr in 1:My
            yw = -recon_fov_cm / 2 + (jr - 0.5) * dyr
            jp = round(Int, (yw - oy) / dyp + 1)
            (1 <= jp <= nyp) || continue
            for ir in 1:Mx
                xw = -recon_fov_cm / 2 + (ir - 0.5) * dxr
                ip = round(Int, (xw - ox) / dxp + 1)
                (1 <= ip <= nxp) || continue
                mask[ir, jr, kr] = is_cist[Int(phantom_labeled[ip, jp, kp]) + 1]
            end
        end
    end
    return mask
end

# ╔═╡ 08b00012-0000-4000-8000-000000000003
cisterna_recon_mask = (phantom_labeled === nothing || phantom_origin === nothing) ? nothing :
    build_cisterna_recon_mask(
        phantom_labeled, phantom_origin, VOXEL_SIZE_CM,
        recon_opts.matrix_size, recon_opts.fov_cm, recon_opts.z_cm,
        CISTERNA_IDS,
    );

# ╔═╡ 08b00012-0000-4000-8000-000000000004
cisterna_recon_mask === nothing ? md"_skipped_" : md"""
    **Cisterna ROI in recon volume:** $(count(cisterna_recon_mask)) voxels
    across $(count(any(cisterna_recon_mask; dims = (1, 2))[:])) of $(recon_opts.matrix_size[3]) recon slices.
    """

# ╔═╡ 08b00013-0000-4000-8000-000000000001
md"""
## 11. Time loop — simulate + recon for every `t` in `TIMES_S`

For each time point:

1. Parse that `t`'s xlsx into materials (with the ID 2 override).
2. Wrap the GPU labels in a new `Phantom` with those materials + the
   bolus origin.
3. `simulate!` → BHC sino correction → FDK → image-domain BHC → HU →
   noise floor → radial cupping.
4. Compute mean HU in the cisterna ROI → push to the TDC.

We drop the workspace + sinogram between iterations to keep GPU memory
flat.  Budget on Metal: ~30-60 s / time point at 0.1 mm iso phantom.
"""

# ╔═╡ 08b00013-0000-4000-8000-000000000002
tdc_results = (phantom_labels_gpu === nothing || bhc_calibration === nothing ||
        cisterna_recon_mask === nothing) ? nothing : let
    present_ids = unique(phantom_labeled)
    matrix_size = recon_opts.matrix_size
    n_cist = count(cisterna_recon_mask)

    times = Int[]
    mean_hu = Float64[]
    hu_stacks = Dict{Int, Array{Float32, 3}}()

    for t in TIMES_S
        @info "Bolus tracking · t = $(t) s"
        materials_t = build_materials_for_time(t, present_ids)
        phantom_t = BS.Phantom(phantom_labels_gpu, materials_t, VOXEL_SIZE_CM; origin = phantom_origin)

        ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_t)
        BS.simulate!(ws, phantom_t, protocol, sim_opts)
        sino_cpu = Array(ws.sinogram)
        geom = ws.geom
        ws = nothing
        GC.gc(true)

        sino_gpu = to_gpu(sino_cpu)
        sino_bhc = BS.apply_bhc_two_material(
            sino_gpu, bhc_calibration.model, geom, matrix_size,
        )
        sino_gpu = to_gpu(sino_bhc)

        ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, geom, matrix_size)
        recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom)

        BS.apply_bhc_image_domain(
            recon_μ, geom, matrix_size, bhc_calibration.μ_water;
            hu_low = 50.0, hu_high = 150.0, scale_factor = 0.2,
        )

        hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc_calibration.μ_water))
        BS.add_system_noise_floor!(hu, 28.0; seed = 1234)
        BS.apply_radial_cupping_correction!(hu; fov_cm = recon_opts.fov_cm)

        m = n_cist > 0 ? mean(hu[cisterna_recon_mask]) : NaN
        push!(times, t)
        push!(mean_hu, m)
        hu_stacks[t] = hu

        ws_fdk = nothing
        sino_gpu = nothing
        recon_μ = nothing
        GC.gc(true)
    end

    (times = times, mean_hu = mean_hu, hu_stacks = hu_stacks)
end;

# ╔═╡ 08b00013-0000-4000-8000-000000000003
tdc_results === nothing ? md"_skipped — see §1 / §7 / §9_" : md"""
    **TDC sampled at $(length(tdc_results.times)) time points.**
    Peak HU (cisterna ROI): $(round(maximum(tdc_results.mean_hu); digits = 1))
    at t = $(tdc_results.times[argmax(tdc_results.mean_hu)]) s.
    """

# ╔═╡ 08b00014-0000-4000-8000-000000000001
md"""
## 12. Visualize — TDC + axial slice at peak

* Left: mean cisterna HU vs. time (clinical-style time-density curve).
* Right: central axial slice of the recon at the peak time point, with
  the cisterna ROI outlined in red.
"""

# ╔═╡ 08b00014-0000-4000-8000-000000000002
let
    if tdc_results === nothing || cisterna_recon_mask === nothing
        md"_skipped — see §11_"
    else
        peak_t = tdc_results.times[argmax(tdc_results.mean_hu)]
        hu_peak = tdc_results.hu_stacks[peak_t]
        k_mid = recon_opts.matrix_size[3] ÷ 2

        fig = CM.Figure(size = (1300, 520))

        ax1 = CM.Axis(
            fig[1, 1];
            title = "Time-density curve · cisterna chyli ROI",
            subtitle = "$(count(cisterna_recon_mask)) recon voxels · " *
                "120 kVp / $(protocol.mA) mA / $(protocol.collimation_mm) mm coll.",
            xlabel = "Time (s)",
            ylabel = "Mean HU",
            titlesize = 22, subtitlesize = 14,
        )
        CM.lines!(ax1, tdc_results.times, tdc_results.mean_hu;
            color = :dodgerblue, linewidth = 3)
        CM.scatter!(ax1, tdc_results.times, tdc_results.mean_hu;
            color = :dodgerblue, markersize = 10)
        CM.vlines!(ax1, [peak_t]; color = (:crimson, 0.4), linestyle = :dash)
        CM.text!(ax1, peak_t, maximum(tdc_results.mean_hu);
            text = "peak t = $(peak_t) s", color = :crimson,
            align = (:left, :bottom), offset = (6, 4))

        ax2 = CM.Axis(
            fig[1, 2];
            title = "Recon @ peak (t = $(peak_t) s) · K = $(k_mid)",
            subtitle = "$(recon_opts.matrix_size[1])×$(recon_opts.matrix_size[2]) " *
                "@ $(round(RECON_FOV_CM * 10 / recon_opts.matrix_size[1]; digits = 3)) mm · W 800 / L 200",
            aspect = CM.DataAspect(),
            titlesize = 22, subtitlesize = 14,
        )
        hm = CM.heatmap!(ax2, hu_peak[:, :, k_mid];
            colormap = :grays, colorrange = (-200, 600))
        CM.heatmap!(
            ax2, Float32.(cisterna_recon_mask[:, :, k_mid]);
            colormap = [CM.RGBAf(0, 0, 0, 0), CM.RGBAf(1, 0.25, 0.25, 0.7)],
            colorrange = (0, 1),
        )
        CM.hidedecorations!(ax2)
        CM.Colorbar(fig[1, 3], hm; label = "HU", width = 14, labelsize = 14)

        fig
    end
end

# ╔═╡ 08b00015-0000-4000-8000-000000000001
md"""
## 13. (Optional) Save the TDC + per-t HU stacks to JLD2

Persisted as `lymphatic_bolus_tdc.jld2` alongside nb08's per-t recons
so downstream notebooks (curve-fitting, registration with the
full-anatomy nb08 recon, etc.) can pick it up without re-running.
"""

# ╔═╡ 08b00015-0000-4000-8000-000000000002
const BOLUS_SAVE_DIR = joinpath(@__DIR__, "..", "data", "lymphatic_recons")

# ╔═╡ 08b00015-0000-4000-8000-000000000003
bolus_save_path = tdc_results === nothing ? nothing : let
    isdir(BOLUS_SAVE_DIR) || mkpath(BOLUS_SAVE_DIR)
    path = joinpath(BOLUS_SAVE_DIR, "lymphatic_bolus_tdc.jld2")
    JLD2.jldsave(
        path;
        times              = tdc_results.times,
        mean_hu            = tdc_results.mean_hu,
        hu_stacks          = tdc_results.hu_stacks,
        bolus_k_slice      = BOLUS_K_SLICE,
        bolus_z_slab_mm    = BOLUS_Z_SLAB_MM,
        collimation_mm     = protocol.collimation_mm,
        recon_matrix       = collect(recon_opts.matrix_size),
        recon_fov_cm       = recon_opts.fov_cm,
        recon_z_cm         = recon_opts.z_cm,
        target_voxel_mm    = TARGET_VOXEL_MM,
        kvp                = protocol.kVp,
        mA                 = protocol.mA,
        cisterna_ids       = collect(CISTERNA_IDS),
        thoracic_duct_ids  = collect(THORACIC_DUCT_IDS),
    )
    path
end;

# ╔═╡ 08b00015-0000-4000-8000-000000000004
bolus_save_path === nothing ? md"_no TDC to save_" : md"""
    **Saved TDC → JLD2**

    * Path: `$(bolus_save_path)`
    * Time points: $(length(tdc_results.times))
    * Mean HU range: $(round(minimum(tdc_results.mean_hu); digits = 1)) … $(round(maximum(tdc_results.mean_hu); digits = 1))
    """

# ╔═╡ Cell order:
# ╟─08b00001-0000-4000-8000-000000000010
# ╠═08b00001-0000-4000-8000-000000000001
# ╠═08b00001-0000-4000-8000-000000000002
# ╠═08b00001-0000-4000-8000-000000000003
# ╠═08b00001-0000-4000-8000-000000000004
# ╟─08b00002-0000-4000-8000-000000000001
# ╠═08b00002-0000-4000-8000-000000000002
# ╠═08b00002-0000-4000-8000-000000000003
# ╠═08b00002-0000-4000-8000-000000000004
# ╠═08b00002-0000-4000-8000-000000000005
# ╠═08b00002-0000-4000-8000-000000000010
# ╟─08b00002-0000-4000-8000-000000000011
# ╟─08b00003-0000-4000-8000-000000000001
# ╠═08b00003-0000-4000-8000-000000000002
# ╠═08b00003-0000-4000-8000-000000000003
# ╠═08b00003-0000-4000-8000-000000000004
# ╠═08b00003-0000-4000-8000-000000000005
# ╟─08b00003-0000-4000-8000-000000000006
# ╟─08b00004-0000-4000-8000-000000000001
# ╠═08b00004-0000-4000-8000-000000000002
# ╠═08b00004-0000-4000-8000-000000000003
# ╠═08b00004-0000-4000-8000-000000000004
# ╠═08b00004-0000-4000-8000-000000000005
# ╠═08b00004-0000-4000-8000-000000000006
# ╠═08b00004-0000-4000-8000-000000000007
# ╟─08b00004-0000-4000-8000-000000000008
# ╟─08b00005-0000-4000-8000-000000000001
# ╠═08b00005-0000-4000-8000-000000000002
# ╠═08b00005-0000-4000-8000-000000000003
# ╟─08b00005-0000-4000-8000-000000000004
# ╟─08b00006-0000-4000-8000-000000000001
# ╠═08b00006-0000-4000-8000-000000000002
# ╠═08b00006-0000-4000-8000-000000000003
# ╠═08b00006-0000-4000-8000-000000000004
# ╠═08b00006-0000-4000-8000-000000000005
# ╠═08b00006-0000-4000-8000-000000000006
# ╠═08b00006-0000-4000-8000-000000000007
# ╟─08b00007-0000-4000-8000-000000000001
# ╠═08b00007-0000-4000-8000-000000000002
# ╠═08b00007-0000-4000-8000-000000000003
# ╠═08b00007-0000-4000-8000-000000000004
# ╠═08b00007-0000-4000-8000-000000000005
# ╠═08b00007-0000-4000-8000-000000000006
# ╟─08b00007-0000-4000-8000-000000000007
# ╟─08b00008-0000-4000-8000-000000000001
# ╠═08b00008-0000-4000-8000-000000000002
# ╠═08b00008-0000-4000-8000-000000000003
# ╠═08b00008-0000-4000-8000-000000000004
# ╠═08b00008-0000-4000-8000-000000000005
# ╠═08b00008-0000-4000-8000-000000000006
# ╟─08b00008-0000-4000-8000-000000000007
# ╟─08b00009-0000-4000-8000-000000000001
# ╠═08b00009-0000-4000-8000-000000000002
# ╠═08b00009-0000-4000-8000-000000000003
# ╠═08b00009-0000-4000-8000-000000000004
# ╟─08b00009-0000-4000-8000-000000000005
# ╠═08b00009-0000-4000-8000-000000000006
# ╟─08b00010-0000-4000-8000-000000000001
# ╠═08b00010-0000-4000-8000-000000000002
# ╠═08b00010-0000-4000-8000-000000000003
# ╠═08b00010-0000-4000-8000-000000000004
# ╠═08b00010-0000-4000-8000-000000000005
# ╟─08b00011-0000-4000-8000-000000000001
# ╠═08b00011-0000-4000-8000-000000000002
# ╟─08b00011-0000-4000-8000-000000000003
# ╟─08b00012-0000-4000-8000-000000000001
# ╠═08b00012-0000-4000-8000-000000000002
# ╠═08b00012-0000-4000-8000-000000000003
# ╟─08b00012-0000-4000-8000-000000000004
# ╟─08b00013-0000-4000-8000-000000000001
# ╠═08b00013-0000-4000-8000-000000000002
# ╟─08b00013-0000-4000-8000-000000000003
# ╟─08b00014-0000-4000-8000-000000000001
# ╠═08b00014-0000-4000-8000-000000000002
# ╟─08b00015-0000-4000-8000-000000000001
# ╠═08b00015-0000-4000-8000-000000000002
# ╠═08b00015-0000-4000-8000-000000000003
# ╟─08b00015-0000-4000-8000-000000000004
