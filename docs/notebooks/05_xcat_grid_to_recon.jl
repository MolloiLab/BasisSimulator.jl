### A Pluto.jl notebook ###
# v0.3.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 05000001-0000-4000-8000-000000000001
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 05000001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 05000001-0000-4000-8000-000000000003
using Statistics: mean, std

# ╔═╡ 05000001-0000-4000-8000-000000000004
using Unitful: @u_str

# ╔═╡ 05000001-0000-4000-8000-000000000010
md"""
# XCAT → CT: Phantom Grids and the Affine Round-Trip

**Scan the heart of an ultra-high-resolution XCAT chest, then lay the ground-truth labels
exactly onto the reconstructed image, axial and helical.**

A simulation has two voxel grids. The **phantom grid** is your ground truth: here an XCAT 3.0
chest at 0.2 mm, read at 0.4 mm. The **reconstruction grid** is what the scanner outputs: a
centred stack of axial slices at the pixel size you choose. Every quantitative use of a
simulation (organ ROI statistics, segmentation scoring, partial-volume analysis) needs the map
between the two. This notebook shows that map and proves that it is exact:

1. The reconstruction grid is **always centred on the isocentre**; `ReconOptions` has no
   off-centre FOV parameter. To scan a sub-region of a large phantom, crop the **phantom** so
   the region sits at the isocentre. Only the cropped block is ray-traced.
2. `BS.phantom_to_world_affine(phantom)` and `BS.recon_to_world_affine(geom, matrix_size)` are the
   two grids' 4 × 4 voxel → world (cm) matrices.
3. `BS.resample_to_recon(phantom, geom, matrix_size; method = :nearest | :linear)` carries the
   ground truth onto the reconstruction grid.

```
XCAT 3.0 chest export (0.2 mm) ─▶ read at 0.4 mm
   ─▶ find the cardiac labels by name ─▶ bounding box + 1 cm
   ─▶ crop the phantom to the box       (the block is centred on the isocentre)
   ─▶ Scan A: axial, 14 cm FOV          ─▶ water BHC ─▶ FDK ─▶ HU
   ─▶ Scan B: helical, 4 cm of z        ─▶ water BHC ─▶ WFBP ─▶ HU
   ─▶ resample_to_recon ─▶ labels on the HU image, organ ROI statistics, exactness audit
```
"""

# ╔═╡ 05000001-0000-4000-8000-000000000020
md"""
## Notebook setup
"""

# ╔═╡ 05000001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 05000001-0000-4000-8000-000000000031
# ╠═╡ show_logs = false
import CairoMakie as Mke

# ╔═╡ 05000001-0000-4000-8000-000000000060
import PlutoUI

# ╔═╡ 05000001-0000-4000-8000-000000000040
begin
    import GPUSelect
    AT = GPUSelect.Storage()   # CuArray / MtlArray / ROCArray / oneArray, or Array on a CPU-only host
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end

# ╔═╡ 05000001-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 05000001-0000-4000-8000-000000000070
PlutoUI.TableOfContents()

# ╔═╡ 05000002-0000-4000-8000-000000000000
md"""
## Load the XCAT phantom
"""

# ╔═╡ 05000002-0000-4000-8000-000000000001
md"""
### 1. Locate the export

The phantom is an **XCAT 3.0 chest export** (adult male, 50th percentile, with coronary plaque):
one 8-bit little-endian activity volume, `*_act_1.raw`, whose dimensions are in its file name,
next to a `Material_Spreadsheets/` folder with one material table per contrast state. The data
is licensed and not in the repository. The notebook looks in `BASISSIM_XCAT_DIR`, or in
`docs/notebooks/data/xcat/` when that variable is unset, and expects exactly one `*act_1.raw`
there. Without it, every compute cell skips and says so.
"""

# ╔═╡ 05000002-0000-4000-8000-000000000010
const XCAT_DIR = get(ENV, "BASISSIM_XCAT_DIR", joinpath(@__DIR__, "data", "xcat"));

# ╔═╡ 05000002-0000-4000-8000-000000000014
const CONTRAST_STATE = "high_contrast";   # the sheets: non_contrast, low_contrast, high_contrast

# ╔═╡ 05000002-0000-4000-8000-000000000011
"""
    find_xcat_export(dir, state) -> NamedTuple or String

The one `*act_1.raw` in `dir`, its dimensions read off its name (`…_1600x1400x867_…`), its
anatomy (`vmale_50`) and the material sheet of contrast `state`, or a String saying what is
missing.
"""
function find_xcat_export(dir, state)
    isdir(dir) || return "no XCAT directory (set `BASISSIM_XCAT_DIR`)"
    raws = filter(f -> endswith(f, "act_1.raw"), readdir(dir))
    length(raws) == 1 || return "expected one *act_1.raw in the XCAT directory, found $(length(raws))"
    name = only(raws)
    m = match(r"(\d{3,4})[x_](\d{3,4})[x_](\d{3,4})", name)
    a = match(r"^(v(?:fe)?male_\d+)", name)
    (m === nothing || a === nothing) && return "cannot read the dimensions or anatomy from $(name)"
    dims = Tuple(parse.(Int, m.captures))
    raw = joinpath(dir, name)
    filesize(raw) == prod(dims) || return "$(name) is not $(join(dims, " × ")) bytes"
    sheet_name = "$(a.captures[1])_materials_heart_$(state).xlsx"
    sheet = joinpath(dir, "Material_Spreadsheets", sheet_name)
    isfile(sheet) || return "no Material_Spreadsheets/$(sheet_name)"
    return (; raw, name, dims, anatomy = a.captures[1], sheet, sheet_name)
end

# ╔═╡ 05000002-0000-4000-8000-000000000012
xcat = find_xcat_export(XCAT_DIR, CONTRAST_STATE);

# ╔═╡ 05000002-0000-4000-8000-000000000015
const HAS_XCAT = xcat isa NamedTuple;

# ╔═╡ 05000002-0000-4000-8000-000000000013
HAS_XCAT ? Markdown.parse("""
    **XCAT export:** `$(xcat.name)` ($(join(xcat.dims, " × ")) voxels,
    $(round(prod(xcat.dims) / 1024^3; digits = 2)) GiB) · materials: `$(xcat.sheet_name)`
    """) : Markdown.parse("""
    !!! warning "XCAT export not found — compute cells skipped"
        $(xcat). Set `BASISSIM_XCAT_DIR` to a folder holding one `*act_1.raw` and its
        `Material_Spreadsheets/`, or place them in `docs/notebooks/data/xcat/`, and re-run.
    """)

# ╔═╡ 05000003-0000-4000-8000-000000000001
md"""
### 2. Read the mask at 0.4 mm

The export is 0.2 mm isotropic, label 0 is air, and the file runs anterior → posterior along its
second axis. Reversing that axis makes `y` increase towards the patient's front, so every image
drawn with `y` up shows the sternum at the top and the patient's left on the viewer's right.
A label-preserving nearest-neighbour step of 2 then reads it at **0.4 mm**: still finer than the
reconstruction pixels below, so the resampling has something to interpolate across.
"""

# ╔═╡ 05000003-0000-4000-8000-000000000010
"""The activity volume, `y` towards anterior (the file runs posterior along axis 2)."""
read_xcat_mask(raw, dims) = reverse(read!(raw, Array{UInt8}(undef, dims...)); dims = 2)

# ╔═╡ 05000003-0000-4000-8000-000000000011
"""Nearest-neighbour 3D downsample by an integer factor: keeps labels intact."""
function downsample_labeled(phantom::AbstractArray{T, 3}, factor::Int) where {T}
    factor == 1 && return phantom
    nx, ny, nz = size(phantom) .÷ factor
    out = similar(phantom, (nx, ny, nz))
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        out[i, j, k] = phantom[(i - 1) * factor + factor ÷ 2 + 1,
                               (j - 1) * factor + factor ÷ 2 + 1,
                               (k - 1) * factor + factor ÷ 2 + 1]
    end
    return out
end

# ╔═╡ 05000003-0000-4000-8000-000000000020
const DOWNSAMPLE_FACTOR = 2;

# ╔═╡ 05000003-0000-4000-8000-000000000021
const VOXEL_SIZE_CM = ntuple(_ -> 0.02 * DOWNSAMPLE_FACTOR, 3);   # 0.2 mm export × factor

# ╔═╡ 05000003-0000-4000-8000-000000000030
phantom_full_uhr = HAS_XCAT ?
    downsample_labeled(read_xcat_mask(xcat.raw, xcat.dims), DOWNSAMPLE_FACTOR) : nothing;

# ╔═╡ 05000003-0000-4000-8000-000000000040
if phantom_full_uhr === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let (nx, ny, nz) = size(phantom_full_uhr)
        Markdown.parse("""
        **Phantom read:** $(nx) × $(ny) × $(nz) voxels of $(round(VOXEL_SIZE_CM[1] * 10; digits = 2)) mm
        ($(round(sizeof(phantom_full_uhr) / 1024^2; digits = 1)) MiB as UInt8),
        $(join(round.((nx, ny, nz) .* VOXEL_SIZE_CM; digits = 1), " × ")) cm,
        $(length(unique(phantom_full_uhr))) labels present.
        """)
    end
end

# ╔═╡ 05000004-0000-4000-8000-000000000001
md"""
### 3. Materials from the XCAT sheet

Each sheet row is one organ: a name, its elemental mass fractions (one column per atomic
number), its density and its label (`Organ ID`). Every row becomes an `XrayAttenuation.Material`
keyed by its label. Labels the sheet does not name stay **air**, which is what `Phantom` assigns
to any label without a material. The `high_contrast` sheet puts iodinated blood in the cardiac
chambers, the aorta and the coronaries.

We need the table before cropping: the organ **names** are how the cardiac labels are found.
"""

# ╔═╡ 05000004-0000-4000-8000-000000000002
import XLSX

# ╔═╡ 05000004-0000-4000-8000-000000000004
const _ATOMIC_MASSES = Dict(
    1 => 1.008, 6 => 12.011, 7 => 14.007, 8 => 15.999, 11 => 22.99, 12 => 24.305,
    15 => 30.974, 16 => 32.06, 17 => 35.45, 19 => 39.098, 20 => 40.078, 26 => 55.845, 53 => 126.904,
);

# ╔═╡ 05000004-0000-4000-8000-000000000005
const _I_VALUES_EV = Dict(
    1 => 19.2, 6 => 81.0, 7 => 82.0, 8 => 95.0, 11 => 149.0, 12 => 156.0,
    15 => 173.0, 16 => 180.0, 17 => 174.0, 19 => 190.0, 20 => 191.0, 26 => 286.0, 53 => 491.0,
);

# ╔═╡ 05000004-0000-4000-8000-000000000006
"""⟨Z/A⟩ of a composition (mass fractions)."""
compute_ZA_ratio(comp::Dict{Int, Float64}) =
    sum(w * Z / _ATOMIC_MASSES[Z] for (Z, w) in comp) / sum(values(comp))

# ╔═╡ 05000004-0000-4000-8000-000000000007
"""Bragg additivity for the mean excitation energy of a composition."""
function compute_mean_excitation_energy(comp::Dict{Int, Float64})
    num = sum(w * Z / _ATOMIC_MASSES[Z] * log(_I_VALUES_EV[Z]) for (Z, w) in comp)
    den = sum(w * Z / _ATOMIC_MASSES[Z] for (Z, w) in comp)
    return exp(num / den) * u"eV"
end

# ╔═╡ 05000004-0000-4000-8000-000000000008
"""
    load_xcat_materials(sheet) -> Dict{Int, XA.Material}

`Name | <Z> … | Density | Organ ID` → one material per label.
"""
function load_xcat_materials(sheet)
    wb = XLSX.readxlsx(sheet)
    data = wb[first(XLSX.sheetnames(wb))][:]
    ncol = size(data, 2)
    (string(data[1, ncol - 1]) == "Density" && string(data[1, ncol]) == "Organ ID") ||
        error("unexpected sheet layout: $(join(string.(data[1, :]), " | "))")
    zcols = [(c, parse(Int, string(data[1, c]))) for c in 2:(ncol - 2)]
    out = Dict{Int, BS.XA.Material}()
    for r in 2:size(data, 1)
        data[r, ncol] isa Number || continue
        comp = Dict{Int, Float64}(Z => Float64(data[r, c]) for (c, Z) in zcols
                                  if data[r, c] isa Number && data[r, c] > 0)
        isapprox(sum(values(comp)), 1; atol = 1.0e-3) || error("mass fractions of $(data[r, 1]) do not sum to 1")
        out[Int(data[r, ncol])] = BS.XA.Material(
            string(data[r, 1]), compute_ZA_ratio(comp), compute_mean_excitation_energy(comp),
            Float64(data[r, ncol - 1]) * u"g/cm^3", comp,
        )
    end
    return out
end

# ╔═╡ 05000004-0000-4000-8000-000000000010
materials_full = HAS_XCAT ? load_xcat_materials(xcat.sheet) : nothing;

# ╔═╡ 05000004-0000-4000-8000-000000000012
const LABEL_RANGE = (0, materials_full === nothing ? 1 : maximum(keys(materials_full)));   # one colour per label ID in every map

# ╔═╡ 05000004-0000-4000-8000-000000000011
if materials_full === nothing || phantom_full_uhr === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let present = sort(Int.(unique(phantom_full_uhr)))
        unnamed = [l for l in present if l != 0 && !haskey(materials_full, l)]
        Markdown.parse("""
        **$(length(materials_full)) materials** in `$(xcat.sheet_name)`; $(length(present)) labels
        present in the volume; labels present but not named by the sheet (read as air):
        $(isempty(unnamed) ? "none" : join(unnamed, ", ")).
        """)
    end
end

# ╔═╡ 05000005-0000-4000-8000-000000000001
md"""
### 4. The cardiac bounding box

The heart is found by **name**: myocardium (`myo…`) and blood pools (`bldpl…`) of the four
chambers, the pericardium, the coronary arteries and veins with their walls, and the coronary
plaque components (`rca…`, `lcx…`, `lad…`). The aorta is left out: it runs the length of the
chest and would stretch the box to the whole export. The bounding box of those labels, padded
by 1 cm on every side, is the region we scan.

!!! info "A heuristic over XCAT's naming"
    For a phantom whose labels are named differently, replace the pattern with an explicit list
    of label IDs.
"""

# ╔═╡ 05000005-0000-4000-8000-000000000010
heart_label_ids = materials_full === nothing ? nothing : let
    pattern = r"^(myo|bldpl)|pericardium|coronary|^(rca|lcx|lad)\d"i
    sort(UInt8[UInt8(id) for (id, mat) in materials_full if occursin(pattern, mat.name)])
end;

# ╔═╡ 05000005-0000-4000-8000-000000000011
heart_label_ids === nothing ? md"" : Markdown.parse(
    "**$(length(heart_label_ids)) cardiac labels:** " *
    join(["$(Int(id)) `$(materials_full[Int(id)].name)`" for id in heart_label_ids], " · "))

# ╔═╡ 05000005-0000-4000-8000-000000000020
heart_bbox = (phantom_full_uhr === nothing || heart_label_ids === nothing) ? nothing : let
    is_heart = falses(256)
    for id in heart_label_ids
        is_heart[Int(id) + 1] = true
    end
    nx, ny, nz = size(phantom_full_uhr)
    lo, hi = [nx + 1, ny + 1, nz + 1], [0, 0, 0]
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        if is_heart[Int(phantom_full_uhr[i, j, k]) + 1]
            lo .= min.(lo, (i, j, k)); hi .= max.(hi, (i, j, k))
        end
    end
    hi[1] == 0 && error("no cardiac voxels in the phantom")
    pad = round.(Int, 1.0 ./ VOXEL_SIZE_CM)                      # 1 cm
    lo .= max.(1, lo .- pad); hi .= min.((nx, ny, nz), hi .+ pad)
    (i_lo = lo[1], i_hi = hi[1], j_lo = lo[2], j_hi = hi[2], k_lo = lo[3], k_hi = hi[3])
end;

# ╔═╡ 05000006-0000-4000-8000-000000000000
md"""
## Scan A: axial, zoomed cardiac FOV
"""

# ╔═╡ 05000006-0000-4000-8000-000000000001
md"""
### 1. Crop the phantom

An index operation on the mask. The cropped block is what `simulate!` sees, so nothing outside it
is ray-traced or held in device memory. It also changes the object: the rest of the chest is no
longer in the beam, so this is a scan of the cardiac block, not of a chest with a small display
field. For a chest scan with a small field, keep the whole phantom and set a small
`ReconOptions(fov_cm = …)` instead.
"""

# ╔═╡ 05000006-0000-4000-8000-000000000010
phantom_cropped = heart_bbox === nothing ? nothing : let b = heart_bbox
    phantom_full_uhr[b.i_lo:b.i_hi, b.j_lo:b.j_hi, b.k_lo:b.k_hi]
end;

# ╔═╡ 05000006-0000-4000-8000-000000000020
phantom_cropped === nothing ? md"" : let
    n_full, n_crop = length(phantom_full_uhr), length(phantom_cropped)
    Markdown.parse("""
    **Crop:** $(join(size(phantom_cropped), " × ")) voxels,
    $(join(round.(size(phantom_cropped) .* VOXEL_SIZE_CM; digits = 1), " × ")) cm ·
    $(round(100 * n_crop / n_full; digits = 1)) % of the $(round(n_full / 1.0e6; digits = 1)) M-voxel phantom
    """)
end

# ╔═╡ 05000007-0000-4000-8000-000000000010
if phantom_cropped === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let b = heart_bbox
        z_full = (b.k_lo + b.k_hi) ÷ 2
        z_crop = z_full - b.k_lo + 1
        vmm = round(VOXEL_SIZE_CM[1] * 10; digits = 2)

        fig = Mke.Figure(size = (1300, 600))
        ax_l = Mke.Axis(fig[1, 1]; title = "Full phantom · z = $(z_full)",
            subtitle = "$(size(phantom_full_uhr, 1)) × $(size(phantom_full_uhr, 2)) @ $(vmm) mm · crop box in red",
            aspect = Mke.DataAspect(), titlesize = 26, subtitlesize = 18)
        Mke.heatmap!(ax_l, Float32.(phantom_full_uhr[:, :, z_full]); colormap = :tab20, colorrange = LABEL_RANGE)
        Mke.poly!(ax_l, Mke.Point2f[(b.i_lo, b.j_lo), (b.i_hi, b.j_lo), (b.i_hi, b.j_hi), (b.i_lo, b.j_hi)];
            color = :transparent, strokecolor = :red, strokewidth = 3)
        Mke.hidedecorations!(ax_l)

        ax_r = Mke.Axis(fig[1, 2]; title = "Cropped cardiac block · z = $(z_crop)",
            subtitle = "$(size(phantom_cropped, 1)) × $(size(phantom_cropped, 2)) @ $(vmm) mm (same voxels, smaller extent)",
            aspect = Mke.DataAspect(), titlesize = 26, subtitlesize = 18)
        Mke.heatmap!(ax_r, Float32.(phantom_cropped[:, :, z_crop]); colormap = :tab20, colorrange = LABEL_RANGE)
        Mke.hidedecorations!(ax_r)
        Mke.Label(fig[2, 1:2], "label maps, colour = label ID · anterior at the top, patient's left on the right";
            fontsize = 16, tellwidth = false)
        Mke.save(joinpath(@__DIR__, "..", "assets", "xcat_grid_crop.png"), fig; px_per_unit = 2)
        fig
    end
end

# ╔═╡ 05000008-0000-4000-8000-000000000001
Markdown.parse("""
### 2. Build the `Phantom` and its world affine

Without an `origin`, `Phantom` centres the block on the isocentre
(`origin = -extent/2 + voxel/2`), which is exactly where the centred reconstruction grid will
look.

Two phantoms are built from the same crop. `phantom_cpu` keeps the XCAT labels, for the
resampling. The simulation phantom is `compact_materials(phantom_cpu)` on the device: the same
attenuation with the labels renumbered densely, because the single-pass `:dd_fast` projector
handles at most 64 materials and the sheet's label IDs run to $(materials_full === nothing ? "—" : maximum(keys(materials_full))).
""")

# ╔═╡ 05000008-0000-4000-8000-000000000011
phantom_cpu = phantom_cropped === nothing ? nothing :
    BS.Phantom(phantom_cropped, materials_full, VOXEL_SIZE_CM);

# ╔═╡ 05000008-0000-4000-8000-000000000010
phantom = phantom_cpu === nothing ? nothing : let c = BS.compact_materials(phantom_cpu)
    BS.Phantom(to_gpu(c.mask), c.materials, c.voxel_size, c.origin, c.extent)
end;

# ╔═╡ 05000009-0000-4000-8000-000000000001
md"""
#### `phantom_to_world_affine`

The 4 × 4 matrix maps a 0-indexed phantom voxel `(i, j, k)` to world coordinates `(x, y, z)` in cm:

```
[ x ]   [ vx  0   0   ox ]   [ i ]
[ y ] = [ 0   vy  0   oy ] · [ j ]
[ z ]   [ 0   0   vz  oz ]   [ k ]
[ 1 ]   [ 0   0   0   1  ]   [ 1 ]
```

`(vx, vy, vz)` is the voxel size and `(ox, oy, oz)` the world position of voxel `(0, 0, 0)`.
"""

# ╔═╡ 05000009-0000-4000-8000-000000000010
if phantom_cpu === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let A = BS.phantom_to_world_affine(phantom_cpu)
        rows = ["| " * join(round.(A[i, :]; digits = 4), " | ") * " |" for i in 1:4]
        centre = phantom_cpu.origin .+ phantom_cpu.extent ./ 2 .- phantom_cpu.voxel_size ./ 2
        Markdown.parse("""
        **`A_phantom = phantom_to_world_affine(phantom_cpu)`** (cm)

        | col 1 | col 2 | col 3 | col 4 |
        |---|---|---|---|
        $(join(rows, "\n"))

        - voxel = $(round.(phantom_cpu.voxel_size .* 10; digits = 2)) mm, extent = $(round.(phantom_cpu.extent; digits = 3)) cm
        - origin (voxel `(0, 0, 0)`) = $(round.(phantom_cpu.origin; digits = 3)) cm
        - centre of the block = $(round.(centre; digits = 6)) cm, the isocentre
        """)
    end
end

# ╔═╡ 0500000a-0000-4000-8000-000000000001
md"""
### 3. Scanner, protocol and a tight reconstruction grid

The GE Revolution Apex Elite of notebook 01: 256 × 0.625 mm rows, a curved Lumex detector and
the large-body bowtie. The grid mapping has nothing to do with the detector type, so any
scanner would do.
"""

# ╔═╡ 0500000a-0000-4000-8000-000000000010
scanner = BS.EICTScanner(
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
    electronic_noise = 3500.0,   # e⁻ rms, added to the counts before the log
    detection_gain = 10.0,
);

# ╔═╡ 0500000a-0000-4000-8000-000000000020
md"""
#### `CTProtocol`: an axial cardiac CTA

120 kVp / 250 mA, a 1 s rotation of 500 views, 5 mm of collimation. `SimOptions` models the
detector integrating each view while the gantry turns (`view_samples = 5`, notebook 01).
"""

# ╔═╡ 0500000a-0000-4000-8000-000000000030
protocol = BS.CTProtocol(
    kVp = 120,
    mA = 250.0,
    views = 500,
    rotation_time = 1.0,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 0500000a-0000-4000-8000-000000000040
sim_opts = BS.SimOptions(seed = 1234, projector = :dd_fast, view_samples = 5);

# ╔═╡ 0500000b-0000-4000-8000-000000000001
md"""
#### `ReconOptions`: a 14 cm field

A **14 cm × 14 cm** field on a 384 × 384 grid (0.365 mm pixels), eight 0.625 mm slices over the
5 mm beam. The field is centred on the isocentre, where the cropped block sits, and it is
deliberately smaller than the block in-plane: the reconstruction grid does not have to contain
the object.
"""

# ╔═╡ 0500000b-0000-4000-8000-000000000010
recon_opts = BS.ReconOptions(
    matrix_size = (384, 384, round(Int, protocol.collimation_mm / 0.625)),
    fov_cm = 14.0,
    z_cm = protocol.collimation_mm / 10,
);

# ╔═╡ 0500000b-0000-4000-8000-000000000020
md"""
#### `recon_to_world_affine`

The reconstruction grid can be inspected before anything is simulated: `CTGeometry` is built
from the scanner, the protocol and the grid, exactly as the workspace builds it. The affine has
the same form as the phantom's, with the reconstruction voxel size (`fov / matrix_size`) and a
centred origin (`-fov/2 + voxel/2`).
"""

# ╔═╡ 0500000b-0000-4000-8000-000000000030
geom_inspect = BS.CTGeometry(
    scanner;
    n_angles = protocol.views,
    fov_cm = recon_opts.fov_cm,
    z_cm = recon_opts.z_cm,
    collimation_mm = protocol.collimation_mm,
);

# ╔═╡ 0500000b-0000-4000-8000-000000000040
let A = BS.recon_to_world_affine(geom_inspect, recon_opts.matrix_size)
    rows = ["| " * join(round.(A[i, :]; digits = 4), " | ") * " |" for i in 1:4]
    n = recon_opts.matrix_size
    fov = geom_inspect.fov
    Markdown.parse("""
    **`A_recon = recon_to_world_affine(geom, matrix_size)`** (cm)

    | col 1 | col 2 | col 3 | col 4 |
    |---|---|---|---|
    $(join(rows, "\n"))

    - matrix = $(join(n, " × ")), voxel = $(join(round.(fov ./ n .* 10; digits = 3), " × ")) mm, FOV = $(join(round.(fov; digits = 3), " × ")) cm
    - origin = $(join(round.(-1 .* fov ./ 2 .+ fov ./ n ./ 2; digits = 4), ", ")) cm: centred on the isocentre
    """)
end

# ╔═╡ 0500000c-0000-4000-8000-000000000010
if phantom_cpu === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let np = size(phantom_cpu.mask), nr = recon_opts.matrix_size, fov = geom_inspect.fov
        vp = round.(phantom_cpu.voxel_size .* 10; digits = 3)
        vr = round.(fov ./ nr .* 10; digits = 3)
        Markdown.parse("""
        The two grids share world coordinates but little else:

        | | phantom (cropped) | reconstruction |
        |---|---|---|
        | voxels | $(join(np, " × ")) | $(join(nr, " × ")) |
        | voxel size (mm) | $(join(vp, " × ")) | $(join(vr, " × ")) |
        | extent (cm) | $(join(round.(phantom_cpu.extent; digits = 2), " × ")) | $(join(round.(fov; digits = 2), " × ")) |
        | origin (cm) | $(join(round.(phantom_cpu.origin; digits = 3), ", ")) | $(join(round.(-1 .* fov ./ 2 .+ fov ./ nr ./ 2; digits = 3), ", ")) |
        """)
    end
end

# ╔═╡ 0500000d-0000-4000-8000-000000000001
md"""
### 4. Simulate and reconstruct

The chain of notebook 01: `create_workspace` → `simulate!`, then the knobless water
beam-hardening correction → FDK → HU with the correction's own μ_water.
"""

# ╔═╡ 0500000d-0000-4000-8000-000000000010
sim = phantom === nothing ? nothing : let
    ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    t = @elapsed result = BS.simulate!(ws, phantom, protocol, sim_opts)
    out = (sino = Array(ws.sinogram), geom = ws.geom, dose = result.dose, t = t)
    ws = nothing
    GC.gc(true)
    out
end;

# ╔═╡ 0500000d-0000-4000-8000-000000000020
bhc = sim === nothing ? nothing : BS.calibrate_bhc_water(sim_opts, protocol; scanner, geom = sim.geom);

# ╔═╡ 0500000d-0000-4000-8000-000000000030
recon_HU = sim === nothing ? nothing : let
    sino = BS.apply_bhc_water(to_gpu(sim.sino), bhc)
    ws = BS.create_fdk_recon_workspace(sino, sim.geom, recon_opts.matrix_size; filter = :standard)
    μ = Array(BS.reconstruct!(ws, sino, sim.geom))
    ws = nothing; sino = nothing; GC.gc(true)
    Float32.(BS.to_hounsfield(μ; μ_water = bhc.μ_water_ref))
end;

# ╔═╡ 0500000d-0000-4000-8000-000000000040
sim === nothing ? md"" : Markdown.parse("""
**Scan A:** $(join(size(sim.sino), " × ")) sinogram (columns × rows × views), simulated in
$(round(sim.t; digits = 1)) s (first call, compilation included) · CTDIvol =
$(round(sim.dose.ctdi_vol_mGy; digits = 2)) mGy · water BHC at
$(round(bhc.reference_energy_keV; digits = 1)) keV
""")

# ╔═╡ 0500000e-0000-4000-8000-000000000001
md"""
### 5. Resample the ground truth

`resample_to_recon` takes each reconstruction voxel's world coordinate from `A_recon`, maps it to
a continuous phantom index through `inv(A_phantom)`, and samples the phantom there:

| `method` | output | use it for |
|---|---|---|
| `:nearest` | `UInt8` labels | ROI extraction, segmentation scoring |
| `:linear` | `Float32` (trilinear) | continuous fields, and the coverage fraction of one binary mask |

Trilinear interpolation of a *multi-label* mask averages label IDs, which means nothing. To get a
partial-volume fraction, resample a 0/1 mask of the structure instead.
"""

# ╔═╡ 0500000e-0000-4000-8000-000000000010
gt_resampled_nn = sim === nothing ? nothing :
    BS.resample_to_recon(phantom_cpu, sim.geom, recon_opts.matrix_size; method = :nearest);

# ╔═╡ 0500000e-0000-4000-8000-000000000012
"""Fraction of each reconstruction voxel covered by the cardiac labels (trilinear on a 0/1 mask)."""
function cardiac_coverage(mask, geom, matrix_size)
    binary = zeros(UInt8, size(mask))
    for id in heart_label_ids
        binary[mask .== id] .= 0x01
    end
    ph = BS.Phantom(binary, Dict(0 => BS.XA.Materials.air, 1 => BS.XA.Materials.water), VOXEL_SIZE_CM)
    return BS.resample_to_recon(ph, geom, matrix_size; method = :linear)
end

# ╔═╡ 0500000e-0000-4000-8000-000000000013
cardiac_coverage_lin = sim === nothing ? nothing :
    cardiac_coverage(phantom_cropped, sim.geom, recon_opts.matrix_size);

# ╔═╡ 0500000e-0000-4000-8000-000000000020
sim === nothing ? md"" : Markdown.parse("""
`gt_resampled_nn` is $(eltype(gt_resampled_nn)) $(join(size(gt_resampled_nn), " × ")), `cardiac_coverage_lin`
is $(eltype(cardiac_coverage_lin)) $(join(size(cardiac_coverage_lin), " × ")), `recon_HU` is
$(eltype(recon_HU)) $(join(size(recon_HU), " × ")): index `(i, j, k)` is the same physical voxel in all three.
""")

# ╔═╡ 0500000f-0000-4000-8000-000000000001
md"""
#### Bring your own interpolator

For a B-spline, a sinc kernel or a learned upsampler, compose the two affines yourself:
`M = inv(A_phantom) * A_recon` takes a 0-indexed reconstruction voxel `(i, j, k, 1)` to a
**continuous** phantom index, which any interpolator (Interpolations.jl,
ImageTransformations.jl, your own kernel) can sample:

```julia
for k in 0:(nz - 1), j in 0:(ny - 1), i in 0:(nx - 1)
    p = M * [i, j, k, 1.0]                      # continuous phantom index
    out[i + 1, j + 1, k + 1] = my_interpolator(phantom.mask, p[1], p[2], p[3])
end
```
"""

# ╔═╡ 0500000f-0000-4000-8000-000000000010
if phantom_cpu === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let M = inv(BS.phantom_to_world_affine(phantom_cpu)) * BS.recon_to_world_affine(geom_inspect, recon_opts.matrix_size)
        rows = ["| " * join(round.(M[i, :]; digits = 4), " | ") * " |" for i in 1:4]
        n = recon_opts.matrix_size
        vc = M * [(n[1] - 1) / 2, (n[2] - 1) / 2, (n[3] - 1) / 2, 1.0]
        pc = (size(phantom_cpu.mask) .- 1) ./ 2
        Markdown.parse("""
        **`M = inv(A_phantom) * A_recon`**

        | col 1 | col 2 | col 3 | col 4 |
        |---|---|---|---|
        $(join(rows, "\n"))

        The centre of the reconstruction grid maps to phantom index $(round.((vc[1], vc[2], vc[3]); digits = 2)),
        the centre of the cropped block ($(round.(pc; digits = 2))).
        """)
    end
end

# ╔═╡ 05000010-0000-4000-8000-000000000001
md"""
#### The overlay

Four views of the central slice, all on the reconstruction grid:

| panel | shows |
|---|---|
| top left | the HU reconstruction |
| top right | every label, resampled with `:nearest` |
| bottom left | the cardiac labels (`:nearest`) over the HU image |
| bottom right | the cardiac coverage fraction (`:linear` on a 0/1 mask) over the HU image |
"""

# ╔═╡ 05000010-0000-4000-8000-000000000010
if recon_HU === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let
        is_cardiac = falses(256)
        for id in heart_label_ids
            is_cardiac[Int(id) + 1] = true
        end
        z = size(recon_HU, 3) ÷ 2 + 1
        hu = recon_HU[:, :, z]
        lab = gt_resampled_nn[:, :, z]
        nn_overlay = [is_cardiac[Int(l) + 1] ? Float32(l) : NaN32 for l in lab]
        cov = cardiac_coverage_lin[:, :, z]
        lin_overlay = [c < 0.05f0 ? NaN32 : Float32(c) for c in cov]

        fig = Mke.Figure(size = (1400, 1320))
        hu_kw = (colormap = :grays, colorrange = (-300, 700))
        tk = (titlesize = 26, subtitlesize = 18, aspect = Mke.DataAspect())

        ax = Mke.Axis(fig[1, 1]; title = "HU reconstruction",
            subtitle = "slice $(z) of $(size(recon_HU, 3)) · FOV $(recon_opts.fov_cm) cm, centred", tk...)
        hm = Mke.heatmap!(ax, hu; hu_kw...); Mke.hidedecorations!(ax)
        Mke.Colorbar(fig[1, 3], hm; label = "HU", width = 14, labelsize = 18)

        ax = Mke.Axis(fig[1, 2]; title = "All labels (:nearest)",
            subtitle = "$(length(unique(gt_resampled_nn))) labels on the reconstruction grid", tk...)
        Mke.heatmap!(ax, Float32.(lab); colormap = :tab20, colorrange = LABEL_RANGE); Mke.hidedecorations!(ax)

        ax = Mke.Axis(fig[2, 1]; title = "HU + cardiac labels", subtitle = ":nearest, α = 0.6", tk...)
        Mke.heatmap!(ax, hu; hu_kw...)
        Mke.heatmap!(ax, nn_overlay; colormap = :tab20, colorrange = LABEL_RANGE, alpha = 0.6, nan_color = (:white, 0.0))
        Mke.hidedecorations!(ax)

        ax = Mke.Axis(fig[2, 2]; title = "HU + cardiac coverage", subtitle = ":linear on a 0/1 mask, shown ≥ 0.05", tk...)
        Mke.heatmap!(ax, hu; hu_kw...)
        hc = Mke.heatmap!(ax, lin_overlay; colormap = :viridis, colorrange = (0, 1), alpha = 0.7,
            nan_color = (:white, 0.0))
        Mke.hidedecorations!(ax)
        Mke.Colorbar(fig[2, 3], hc; label = "cardiac fraction", width = 14, labelsize = 18)
        Mke.save(joinpath(@__DIR__, "..", "assets", "xcat_grid_overlay.png"), fig; px_per_unit = 2)
        fig
    end
end

# ╔═╡ 05000018-0000-4000-8000-000000000001
md"""
#### Organ ROI statistics from the resampled labels

With the labels on the reconstruction grid, an organ ROI is one comparison:
`recon_HU[gt_resampled_nn .== label]`. Below, each label with enough voxels on the central slice is
eroded in-plane by one voxel (to drop the partial-volume rim) and its mean HU is compared with the
monoenergetic HU of its sheet material at the BHC reference energy. Expect soft tissue, fat, lung
and blood within a few tens of HU. Bone reads low, the single-kVp beam-hardening residual that
notebook 01 quantifies, and small or thin structures pick up partial volume from their
neighbours across the 0.625 mm slice.
"""

# ╔═╡ 05000018-0000-4000-8000-000000000002
if recon_HU === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let lab = gt_resampled_nn, k = size(recon_HU, 3) ÷ 2 + 1
        nx, ny = size(lab, 1), size(lab, 2)
        refE, μw = bhc.reference_energy_keV, bhc.μ_water_ref
        # voxels inside the reconstruction circle only
        inside(i, j) = (i - (nx + 1) / 2)^2 + (j - (ny + 1) / 2)^2 <= (min(nx, ny) / 2 - 2)^2
        rows = String[]
        for id in sort(unique(lab[:, :, k]))
            haskey(materials_full, Int(id)) || continue
            vox = [recon_HU[i, j, k] for j in 2:(ny - 1), i in 2:(nx - 1)
                   if inside(i, j) && all(lab[i + di, j + dj, k] == id for di in -1:1, dj in -1:1)]
            length(vox) < 200 && continue
            mat = materials_full[Int(id)]
            theory = 1000 * (BS.compute_μ_at_energy(mat, refE) - μw) / μw
            push!(rows, "| $(Int(id)) | `$(mat.name)` | $(length(vox)) | $(round(mean(vox); digits = 1)) | " *
                        "$(round(std(vox); digits = 1)) | $(round(theory; digits = 1)) | $(round(mean(vox) - theory; digits = 1)) |")
        end
        Markdown.parse("""
        **Scan A, slice $(k): per-label ROI statistics** (theory = monoenergetic HU at $(round(refE; digits = 1)) keV)

        | label | material | voxels | mean HU | σ (HU) | theory HU | mean − theory |
        |--:|:--|--:|--:|--:|--:|--:|
        $(join(rows, "\n"))
        """)
    end
end

# ╔═╡ 05000012-0000-4000-8000-000000000000
md"""
## Scan B: helical, extended z

The same phantom and the same three calls, now with a **taller crop** scanned by a **helical**
acquisition and reconstructed into a stack of 64 axial slices to scroll through. The helical
reconstruction grid has its own affine (a longer z extent than Scan A), and `resample_to_recon`
lands the labels on every slice unchanged.
"""

# ╔═╡ 05000013-0000-4000-8000-000000000001
md"""
### 1. A taller crop

The in-plane box of Scan A, extended 4 cm beyond the cardiac extent at each end in z (where the
phantom allows), so the helix has anatomy to sweep through.
"""

# ╔═╡ 05000013-0000-4000-8000-000000000010
heart_bbox_tall = heart_bbox === nothing ? nothing : let b = heart_bbox
    extra = round(Int, 4.0 / VOXEL_SIZE_CM[3])
    merge(b, (k_lo = max(1, b.k_lo - extra), k_hi = min(size(phantom_full_uhr, 3), b.k_hi + extra)))
end;

# ╔═╡ 05000013-0000-4000-8000-000000000020
phantom_cropped_tall = heart_bbox_tall === nothing ? nothing : let b = heart_bbox_tall
    phantom_full_uhr[b.i_lo:b.i_hi, b.j_lo:b.j_hi, b.k_lo:b.k_hi]
end;

# ╔═╡ 05000013-0000-4000-8000-000000000031
phantom_helical_cpu = phantom_cropped_tall === nothing ? nothing :
    BS.Phantom(phantom_cropped_tall, materials_full, VOXEL_SIZE_CM);

# ╔═╡ 05000013-0000-4000-8000-000000000030
phantom_helical = phantom_helical_cpu === nothing ? nothing : let c = BS.compact_materials(phantom_helical_cpu)
    BS.Phantom(to_gpu(c.mask), c.materials, c.voxel_size, c.origin, c.extent)
end;

# ╔═╡ 05000013-0000-4000-8000-000000000040
recon_opts_helical = BS.ReconOptions(
    matrix_size = (384, 384, 64),   # 64 slices of 0.625 mm
    fov_cm = 14.0,
    z_cm = 4.0,
);

# ╔═╡ 05000012-0000-4000-8000-000000000001
md"""
### 2. The helical acquisition

`pitch` and `n_rotations` make the protocol helical: 10 mm of collimation at pitch 1.0 over 8
rotations is 8 cm of table travel, centred on the isocentre. The geometry becomes a z-ramped
trajectory, `:dd_fast` projects it unchanged (the sub-views of the view integration follow the
helix, each advanced along z by its share of the table feed), and `reconstruct!` recognises the
helical geometry and runs rebinned weighted FBP. The reconstruction grid is still a centred stack of axial
slices, so the affines apply as before.
"""

# ╔═╡ 05000012-0000-4000-8000-000000000005
protocol_helical = BS.CTProtocol(
    kVp = 120, mA = 250.0, views = 500, rotation_time = 1.0,
    collimation_mm = 10.0, additional_filters = [("Al", 4.5)],
    pitch = 1.0, n_rotations = 8,
);

# ╔═╡ 05000012-0000-4000-8000-000000000010
sim_helical = phantom_helical === nothing ? nothing : let
    ws = BS.create_workspace(scanner, protocol_helical, sim_opts, recon_opts_helical, phantom_helical)
    t = @elapsed result = BS.simulate!(ws, phantom_helical, protocol_helical, sim_opts)
    out = (sino = Array(ws.sinogram), geom = ws.geom, dose = result.dose, t = t)
    ws = nothing
    GC.gc(true)
    out
end;

# ╔═╡ 05000012-0000-4000-8000-000000000020
recon_HU_helical = sim_helical === nothing ? nothing : let
    bhc_h = BS.calibrate_bhc_water(sim_opts, protocol_helical; scanner, geom = sim_helical.geom)
    sino = BS.apply_bhc_water(to_gpu(sim_helical.sino), bhc_h)
    ws = BS.create_fdk_recon_workspace(sino, sim_helical.geom, recon_opts_helical.matrix_size; filter = :standard)
    μ = Array(BS.reconstruct!(ws, sino, sim_helical.geom))   # helical geometry → rebinned WFBP
    ws = nothing; sino = nothing; GC.gc(true)
    Float32.(BS.to_hounsfield(μ; μ_water = bhc_h.μ_water_ref))
end;

# ╔═╡ 05000014-0000-4000-8000-000000000010
if sim_helical === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let A = BS.recon_to_world_affine(sim_helical.geom, recon_opts_helical.matrix_size)
        rows = ["| " * join(round.(A[i, :]; digits = 4), " | ") * " |" for i in 1:4]
        d = sim_helical.dose
        Markdown.parse("""
        **Scan B:** $(join(size(sim_helical.sino), " × ")) sinogram, simulated in
        $(round(sim_helical.t; digits = 1)) s · CTDIvol = $(round(d.ctdi_vol_mGy; digits = 2)) mGy,
        DLP = $(round(d.dlp_mGy_cm; digits = 1)) mGy·cm over $(round(d.scan_length_cm; digits = 1)) cm (pitch $(d.pitch))

        **`A_recon` of the helical grid** (cm)

        | col 1 | col 2 | col 3 | col 4 |
        |---|---|---|---|
        $(join(rows, "\n"))

        $(recon_opts_helical.matrix_size[3]) slices over $(recon_opts_helical.z_cm) cm, against Scan A's
        $(recon_opts.matrix_size[3]) over $(recon_opts.z_cm) cm; the same centred origin rule.
        """)
    end
end

# ╔═╡ 05000012-0000-4000-8000-000000000030
gt_helical_nn = sim_helical === nothing ? nothing :
    BS.resample_to_recon(phantom_helical_cpu, sim_helical.geom, recon_opts_helical.matrix_size; method = :nearest);

# ╔═╡ 05000016-0000-4000-8000-000000000010
cardiac_coverage_lin_helical = sim_helical === nothing ? nothing :
    cardiac_coverage(phantom_cropped_tall, sim_helical.geom, recon_opts_helical.matrix_size);

# ╔═╡ 05000015-0000-4000-8000-000000000001
md"""
### 3. Scroll through z

Left: the WFBP reconstruction. Middle: the cardiac labels resampled with `:nearest`, which snap
every reconstruction voxel to the single closest phantom voxel, so boundaries stair-step onto the
coarser grid. Right: the cardiac coverage from `:linear` on the 0/1 mask, a fully 3D
partial-volume fraction that also catches voxels straddling a surface in z.

Trilinear sampling at the voxel centre equals the true covered fraction when the two grids are of
similar resolution, as here (0.4 mm phantom, 0.365 mm × 0.625 mm reconstruction). For a phantom
much finer than the reconstruction, a true volume fraction needs box averaging instead.
"""

# ╔═╡ 05000015-0000-4000-8000-000000000010
@bind z_helical PlutoUI.Slider(1:recon_opts_helical.matrix_size[3]; default = recon_opts_helical.matrix_size[3] ÷ 2, show_value = true)

# ╔═╡ 05000016-0000-4000-8000-000000000020
if sim_helical === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let nz = size(recon_HU_helical, 3)
        z = clamp(z_helical, 1, nz)
        recon = recon_HU_helical[:, :, z]
        lab = gt_helical_nn[:, :, z]
        cov = cardiac_coverage_lin_helical[:, :, z]
        is_cardiac = falses(256)
        for id in heart_label_ids
            is_cardiac[Int(id) + 1] = true
        end
        nx, ny = size(lab)
        inside(i, j) = (i - (nx + 1) / 2)^2 + (j - (ny + 1) / 2)^2 <= (min(nx, ny) / 2)^2
        nn_over = [inside(i, j) && is_cardiac[Int(lab[i, j]) + 1] ? Float32(lab[i, j]) : NaN32 for i in 1:nx, j in 1:ny]
        lin_over = [inside(i, j) && cov[i, j] > 0.01f0 ? Float32(cov[i, j]) : NaN32 for i in 1:nx, j in 1:ny]
        z_mm = 10 * (-recon_opts_helical.z_cm / 2 + (z - 0.5) * recon_opts_helical.z_cm / nz)

        fig = Mke.Figure(size = (1500, 600))
        Mke.Label(fig[0, 1:4], "helical WFBP · slice $(z) of $(nz) · z = $(round(z_mm; digits = 2)) mm";
            fontsize = 24, font = :bold, tellwidth = false)
        hu_kw = (colormap = :grays, colorrange = (-200, 600))
        tk = (titlesize = 21, aspect = Mke.DataAspect())
        ax1 = Mke.Axis(fig[1, 1]; title = "HU reconstruction", tk...)
        hm = Mke.heatmap!(ax1, recon; hu_kw...); Mke.hidedecorations!(ax1)
        ax2 = Mke.Axis(fig[1, 2]; title = ":nearest cardiac labels", tk...)
        Mke.heatmap!(ax2, recon; hu_kw...)
        Mke.heatmap!(ax2, nn_over; colormap = :tab20, colorrange = LABEL_RANGE, alpha = 0.65, nan_color = (:white, 0.0))
        Mke.hidedecorations!(ax2)
        ax3 = Mke.Axis(fig[1, 3]; title = ":linear cardiac coverage", tk...)
        Mke.heatmap!(ax3, recon; hu_kw...)
        hc = Mke.heatmap!(ax3, lin_over; colormap = :viridis, colorrange = (0, 1), alpha = 0.75,
            nan_color = (:white, 0.0))
        Mke.hidedecorations!(ax3)
        Mke.Colorbar(fig[1, 4], hc; label = "cardiac fraction", labelsize = 16, ticklabelsize = 13)
        fig
    end
end

# ╔═╡ 05000017-0000-4000-8000-000000000001
md"""
### 4. The mapping is exact

Three checks, for the axial and the helical geometry:

1. **The affine is the reconstructor's grid.** The FDK and WFBP back-projectors place
   reconstruction voxel `idx` (1-indexed) at world `-fov/2 + (idx − ½)·fov/n`, the rule
   `recon_to_world_affine` encodes. The table measures the largest difference.
2. **The round trip is the identity.** `A⁻¹ · A · v = v` to floating-point precision.
3. **The image is registered to the map.** Checks 1 and 2 certify the coordinate map; this one
   certifies the image. Edges of the reconstruction are correlated with the label boundaries
   over integer in-plane shifts: the best shift should be `(0, 0)`. A half-voxel or rebinning
   offset between the forward projector and the back-projector would show up here and nowhere
   else.

`is_helical(geom)` switches the back-projection algorithm, never the grid, so the helical
mapping is as exact as the axial one. Softness at a boundary is `:nearest` quantisation or the
reconstruction's point-spread function, not the mapping.
"""

# ╔═╡ 05000012-0000-4000-8000-000000000050
if sim === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let
        # affine world vs the back-projector's voxel-centre rule, and A⁻¹·A·v − v
        audit = function (geom, ms)
            A = BS.recon_to_world_affine(geom, ms)
            Ainv = inv(A)
            grid_err = 0.0; rt_err = 0.0
            for ax in 1:3, idx in (1, (ms[ax] + 1) ÷ 2, ms[ax])
                v = zeros(4); v[4] = 1.0; v[ax] = idx - 1
                world_bp = -geom.fov[ax] / 2 + (idx - 0.5) * geom.fov[ax] / ms[ax]
                grid_err = max(grid_err, abs((A * v)[ax] - world_bp))
                rt_err = max(rt_err, abs((Ainv * (A * v))[ax] - (idx - 1)))
            end
            (grid_err * 1.0e4, rt_err)                       # µm, voxels
        end
        rows = ["| axial | $(join(round.(audit(sim.geom, recon_opts.matrix_size); sigdigits = 2), " | ")) |"]
        sim_helical === nothing ||
            push!(rows, "| helical | $(join(round.(audit(sim_helical.geom, recon_opts_helical.matrix_size); sigdigits = 2), " | ")) |")
        Markdown.parse("""
        | geometry | affine vs reconstructor grid (max, µm) | round trip `A⁻¹·A·v − v` (max, voxels) |
        |---|--:|--:|
        $(join(rows, "\n"))
        """)
    end
end

# ╔═╡ 05000099-0000-4000-8000-000000000001
if sim === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let
        # Correlate reconstruction edge strength with the label-boundary map over integer
        # (dx, dy) shifts on the central slice; the best shift is the real in-plane offset.
        # (Integer resolution: a systematic shift below half a voxel also reads (0, 0).)
        reg_offset = function (recon, gt; R = 5)
            mid = size(recon, 3) ÷ 2
            hu = Float32.(recon[:, :, mid]); lab = gt[:, :, mid]
            nx, ny = size(hu)
            gr = zeros(Float32, nx, ny); lb = zeros(Float32, nx, ny)
            for j in 2:(ny - 1), i in 2:(nx - 1)
                gr[i, j] = hypot(hu[i + 1, j] - hu[i - 1, j], hu[i, j + 1] - hu[i, j - 1])
                # symmetric boundary marker (both sides of an edge), so it carries no half-voxel bias
                (lab[i, j] != lab[i + 1, j] || lab[i, j] != lab[i - 1, j] ||
                 lab[i, j] != lab[i, j + 1] || lab[i, j] != lab[i, j - 1]) && (lb[i, j] = 1.0f0)
            end
            scores = Dict((dx, dy) => sum(gr[i, j] * lb[i + dx, j + dy]
                                          for j in (1 + R):(ny - R), i in (1 + R):(nx - R))
                          for dx in -R:R, dy in -R:R)
            best = argmax(scores)
            (best, scores[(0, 0)] / scores[best])
        end
        a = reg_offset(recon_HU, gt_resampled_nn)
        rows = ["| axial | $(a[1]) | $(round(a[2]; digits = 3)) |"]
        if sim_helical !== nothing
            h = reg_offset(recon_HU_helical, gt_helical_nn)
            push!(rows, "| helical | $(h[1]) | $(round(h[2]; digits = 3)) |")
        end
        Markdown.parse("""
        **Registration of the image to the labels** (central slice, shifts in voxels)

        | geometry | best (dx, dy) | score at (0, 0) / best |
        |---|---|--:|
        $(join(rows, "\n"))
        """)
    end
end

# ╔═╡ 05000011-0000-4000-8000-000000000020
md"""
## Summary

- **Crop the phantom, not the reconstruction.** The reconstruction grid is always centred on the
  isocentre; cropping the input brings the region of interest there, and only the cropped block
  is ray-traced.
- **Two affines and one resampler.** `phantom_to_world_affine` and `recon_to_world_affine` map
  each grid's voxels to world centimetres; `resample_to_recon` composes them, with `:nearest`
  for labels and `:linear` for continuous fields or the coverage of a binary mask, and
  `inv(A_phantom) * A_recon` hands the map to any other interpolator.
- **Exact, axial and helical.** The affine matches the back-projectors' voxel rule, the round
  trip is the identity, and the reconstructed edges sit on the label boundaries with no shift.
- **What it buys you.** Organ ROI statistics, segmentation scores and partial-volume fractions
  are one indexing expression on the reconstruction grid, as the per-label table shows.
"""

# ╔═╡ Cell order:
# ╟─05000001-0000-4000-8000-000000000010
# ╟─05000001-0000-4000-8000-000000000020
# ╟─05000001-0000-4000-8000-000000000001
# ╟─05000001-0000-4000-8000-000000000002
# ╟─05000001-0000-4000-8000-000000000003
# ╟─05000001-0000-4000-8000-000000000004
# ╠═05000001-0000-4000-8000-000000000030
# ╟─05000001-0000-4000-8000-000000000031
# ╟─05000001-0000-4000-8000-000000000060
# ╠═05000001-0000-4000-8000-000000000040
# ╟─05000001-0000-4000-8000-000000000050
# ╟─05000001-0000-4000-8000-000000000070
# ╟─05000002-0000-4000-8000-000000000000
# ╟─05000002-0000-4000-8000-000000000001
# ╠═05000002-0000-4000-8000-000000000010
# ╠═05000002-0000-4000-8000-000000000014
# ╟─05000002-0000-4000-8000-000000000011
# ╠═05000002-0000-4000-8000-000000000012
# ╠═05000002-0000-4000-8000-000000000015
# ╟─05000002-0000-4000-8000-000000000013
# ╟─05000003-0000-4000-8000-000000000001
# ╠═05000003-0000-4000-8000-000000000010
# ╟─05000003-0000-4000-8000-000000000011
# ╠═05000003-0000-4000-8000-000000000020
# ╠═05000003-0000-4000-8000-000000000021
# ╠═05000003-0000-4000-8000-000000000030
# ╟─05000003-0000-4000-8000-000000000040
# ╟─05000004-0000-4000-8000-000000000001
# ╟─05000004-0000-4000-8000-000000000002
# ╟─05000004-0000-4000-8000-000000000004
# ╟─05000004-0000-4000-8000-000000000005
# ╟─05000004-0000-4000-8000-000000000006
# ╟─05000004-0000-4000-8000-000000000007
# ╠═05000004-0000-4000-8000-000000000008
# ╠═05000004-0000-4000-8000-000000000010
# ╟─05000004-0000-4000-8000-000000000011
# ╟─05000004-0000-4000-8000-000000000012
# ╟─05000005-0000-4000-8000-000000000001
# ╠═05000005-0000-4000-8000-000000000010
# ╟─05000005-0000-4000-8000-000000000011
# ╠═05000005-0000-4000-8000-000000000020
# ╟─05000006-0000-4000-8000-000000000000
# ╟─05000006-0000-4000-8000-000000000001
# ╠═05000006-0000-4000-8000-000000000010
# ╟─05000006-0000-4000-8000-000000000020
# ╟─05000007-0000-4000-8000-000000000010
# ╟─05000008-0000-4000-8000-000000000001
# ╠═05000008-0000-4000-8000-000000000011
# ╠═05000008-0000-4000-8000-000000000010
# ╟─05000009-0000-4000-8000-000000000001
# ╟─05000009-0000-4000-8000-000000000010
# ╟─0500000a-0000-4000-8000-000000000001
# ╠═0500000a-0000-4000-8000-000000000010
# ╟─0500000a-0000-4000-8000-000000000020
# ╠═0500000a-0000-4000-8000-000000000030
# ╠═0500000a-0000-4000-8000-000000000040
# ╟─0500000b-0000-4000-8000-000000000001
# ╠═0500000b-0000-4000-8000-000000000010
# ╟─0500000b-0000-4000-8000-000000000020
# ╠═0500000b-0000-4000-8000-000000000030
# ╟─0500000b-0000-4000-8000-000000000040
# ╟─0500000c-0000-4000-8000-000000000010
# ╟─0500000d-0000-4000-8000-000000000001
# ╠═0500000d-0000-4000-8000-000000000010
# ╠═0500000d-0000-4000-8000-000000000020
# ╠═0500000d-0000-4000-8000-000000000030
# ╟─0500000d-0000-4000-8000-000000000040
# ╟─0500000e-0000-4000-8000-000000000001
# ╠═0500000e-0000-4000-8000-000000000010
# ╠═0500000e-0000-4000-8000-000000000012
# ╠═0500000e-0000-4000-8000-000000000013
# ╟─0500000e-0000-4000-8000-000000000020
# ╟─0500000f-0000-4000-8000-000000000001
# ╟─0500000f-0000-4000-8000-000000000010
# ╟─05000010-0000-4000-8000-000000000001
# ╟─05000010-0000-4000-8000-000000000010
# ╟─05000018-0000-4000-8000-000000000001
# ╟─05000018-0000-4000-8000-000000000002
# ╟─05000012-0000-4000-8000-000000000000
# ╟─05000013-0000-4000-8000-000000000001
# ╠═05000013-0000-4000-8000-000000000010
# ╠═05000013-0000-4000-8000-000000000020
# ╠═05000013-0000-4000-8000-000000000031
# ╠═05000013-0000-4000-8000-000000000030
# ╠═05000013-0000-4000-8000-000000000040
# ╟─05000012-0000-4000-8000-000000000001
# ╠═05000012-0000-4000-8000-000000000005
# ╠═05000012-0000-4000-8000-000000000010
# ╠═05000012-0000-4000-8000-000000000020
# ╟─05000014-0000-4000-8000-000000000010
# ╠═05000012-0000-4000-8000-000000000030
# ╠═05000016-0000-4000-8000-000000000010
# ╟─05000015-0000-4000-8000-000000000001
# ╟─05000015-0000-4000-8000-000000000010
# ╟─05000016-0000-4000-8000-000000000020
# ╟─05000017-0000-4000-8000-000000000001
# ╟─05000012-0000-4000-8000-000000000050
# ╟─05000099-0000-4000-8000-000000000001
# ╟─05000011-0000-4000-8000-000000000020
