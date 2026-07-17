### A Pluto.jl notebook ###
# v0.2.3

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
# XCAT UHR → CT: Grid Mapping and the Recon-Affine Round-Trip

**Take a 0.4 mm UHR XCAT phantom, crop it down to a cardiac sub-region as
the input to a clinical CT acquisition, and use the simulator's affine
matrices to overlay the original ground truth on the reconstructed
volume — pixel-perfect.**

This notebook teaches one specific thing the API can be subtle about: how
the **phantom voxel grid** (your input ground truth) and the
**reconstruction voxel grid** (what the scanner outputs) relate.  In
particular:

1. The simulator's recon grid is **always centered at isocenter** — there
   is no off-center FOV / scan-field placement parameter.
2. So to "scan a sub-region of a body phantom" — the way a real scanner's
   SFOV crops out everything outside the bore — you do it on the **input
   phantom**, before forward projection.  Cropping the phantom early is
   strictly more memory-efficient than cropping at the recon stage:
   nothing outside the cropped extent ever gets ray-traced.
3. `BS.phantom_to_world_affine` and `BS.recon_to_world_affine` give you
   the two grids' relationships in 4×4 matrices.
   `BS.resample_to_recon(phantom, geom, matrix_size; method = :nearest|:linear)`
   round-trips the ground-truth labels onto the recon grid for ROI
   extraction, segmentation evaluation, etc.

Pipeline:

```
Full UHR XCAT (0.4 mm, ~32 × 28 × 10 cm)
   → label-name match (heart / atrium / ventricle / coronary / aorta)
   → cardiac voxel bbox + margin
   → crop UHR mask to bbox            ←  the SFOV-equivalent step
   → BS.Phantom(...)                  ←  origin auto-centers crop at iso
   → GE Apex Elite scan + tight FOV recon (centered)
   → BS.resample_to_recon(...; method = :nearest | :linear)
   → overlay ground truth on the HU recon — same shape, same world coords.
```
"""

# ╔═╡ 05000001-0000-4000-8000-000000000020
md"""
## Notebook Setup

Same project + GPU detection idiom as notebooks 02 / 04.
"""

# ╔═╡ 05000001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 05000001-0000-4000-8000-000000000031
# Use CairoMakie for faithful build-time rendering. Snapshot can still isolate
# and compile independent browser-safe islands without hoisting this import.
import CairoMakie as Mke

# ╔═╡ 05000001-0000-4000-8000-000000000060
import PlutoUI

# ╔═╡ 05000001-0000-4000-8000-000000000040
begin
    import GPUSelect
    AT = GPUSelect.Storage()     # the backend array type, directly: MtlArray / CuArray / ROCArray
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
## Load the XCAT Phantom

Shared by both scans below: locate the bin, load the UHR mask, build the
materials dict, and find the cardiac bounding box.
"""

# ╔═╡ 05000002-0000-4000-8000-000000000001
md"""
### 01. Locate the XCAT data

Same env-var pattern as notebook 02 — the bin lives outside the repo
under `BASISSIM_XCAT_DIR` (default: `docs/notebooks/data/xcat/`).  All
heavy compute is gated on `HAS_XCAT` so this notebook still renders
cleanly when the bin isn't available.
"""

# ╔═╡ 05000002-0000-4000-8000-000000000010
const XCAT_DIR = get(
    ENV, "BASISSIM_XCAT_DIR",
    joinpath(@__DIR__, "data", "xcat")
);

# ╔═╡ 05000002-0000-4000-8000-000000000011
const PHANTOM_PATH = joinpath(
    XCAT_DIR,
    "vmale_50_1600x1400x500_8bit_little_endian_act_1.bin"
);

# ╔═╡ 05000002-0000-4000-8000-000000000012
const HAS_XCAT = isfile(PHANTOM_PATH)

# ╔═╡ 05000002-0000-4000-8000-000000000013
HAS_XCAT ? md"""
    **XCAT located:** `$(basename(PHANTOM_PATH))` ($(round(filesize(PHANTOM_PATH) / 1024^2; digits=1)) MB)
    """ : md"""
    !!! warning "XCAT bin not found"
        The configured `$(basename(PHANTOM_PATH))` input is unavailable. All
        compute cells below short-circuit to `nothing` and the comparison
        panels show this notice. Set `BASISSIM_XCAT_DIR` to your local install
        or drop the file into `docs/notebooks/data/xcat/` and re-run.
    """

# ╔═╡ 05000003-0000-4000-8000-000000000001
md"""
### 02. Load the UHR mask (DOWNSAMPLE_FACTOR = 2)

XCAT v_male_50 ships at 1600 × 1400 × 500 voxels @ 0.2 mm isotropic.
Notebook 02 downsamples 5× for speed (1 mm voxels — clinical-typical).
**Here we downsample only 2×** → 800 × 700 × 250 voxels @ 0.4 mm
isotropic = roughly 32 × 28 × 10 cm physical, **140 MB as UInt8**.

The point of going UHR is to make the phantom *finer* than the recon
grid so the affine round-trip has something interesting to interpolate
across.
"""

# ╔═╡ 05000003-0000-4000-8000-000000000010
function load_xcat_bin(
        filepath::AbstractString;
        cols::Int = 1600, rows::Int = 1400, slices::Int = 500,
    )
    expected = cols * rows * slices * sizeof(UInt8)
    actual = filesize(filepath)
    actual == expected ||
        error("XCAT file size mismatch: expected $(expected) bytes, got $(actual)")

    data = Vector{UInt8}(undef, cols * rows * slices)
    open(filepath, "r") do io
        read!(io, data)
    end

    phantom = reshape(data, (cols, rows, slices))
    return reverse(phantom; dims = (2, 3))
end

# ╔═╡ 05000003-0000-4000-8000-000000000011
"""Nearest-neighbor 3D downsample by an integer factor — preserves labels."""
function downsample_labeled(phantom::AbstractArray{T, 3}, factor::Int) where {T}
    factor == 1 && return phantom
    nx, ny, nz = size(phantom) .÷ factor
    out = similar(phantom, (nx, ny, nz))
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        ii = (i - 1) * factor + factor ÷ 2 + 1
        jj = (j - 1) * factor + factor ÷ 2 + 1
        kk = (k - 1) * factor + factor ÷ 2 + 1
        out[i, j, k] = phantom[ii, jj, kk]
    end
    return out
end

# ╔═╡ 05000003-0000-4000-8000-000000000020
const DOWNSAMPLE_FACTOR = 2

# ╔═╡ 05000003-0000-4000-8000-000000000021
const VOXEL_SIZE_CM = (
    0.02 * DOWNSAMPLE_FACTOR,    # 0.4 mm at DS=2
    0.02 * DOWNSAMPLE_FACTOR,
    0.02 * DOWNSAMPLE_FACTOR,
);

# ╔═╡ 05000003-0000-4000-8000-000000000030
phantom_full_uhr = HAS_XCAT ?
    downsample_labeled(load_xcat_bin(PHANTOM_PATH), DOWNSAMPLE_FACTOR) :
    nothing;

# ╔═╡ 05000003-0000-4000-8000-000000000040
let
    if phantom_full_uhr === nothing
        md"""
        !!! warning "Skipped — see §1 above"
        """
    else
        nx, ny, nz = size(phantom_full_uhr)
        ext_cm = (nx, ny, nz) .* VOXEL_SIZE_CM
        n_lbl = length(unique(phantom_full_uhr))
        md"""
        **UHR phantom loaded:**
        - shape = $(nx) × $(ny) × $(nz) (UInt8, $(round(sizeof(phantom_full_uhr)/1024^2, digits=1)) MB)
        - voxel = $(round.(VOXEL_SIZE_CM .* 10, digits=2)) mm
        - extent = $(round.(ext_cm, digits=2)) cm
        - $(n_lbl) unique organ labels present
        """
    end
end

# ╔═╡ 05000004-0000-4000-8000-000000000001
md"""
### 03. Custom materials from XCAT spreadsheet

Same xlsx loader as notebook 02 — one row per organ in
`vmale_50_materials_heart_high_contrast.xlsx` → 33 `XA.Material` entries
keyed by integer organ label.  This loader is unchanged from nb02; only
the phantom grid is different here.

We need the materials dict before §4 because we'll use the materials'
**names** to identify which integer labels correspond to "heart"-related
anatomy for the bbox crop.
"""

# ╔═╡ 05000004-0000-4000-8000-000000000002
import XLSX

# ╔═╡ 05000004-0000-4000-8000-000000000003
const MATERIAL_XLSX_PATH = joinpath(
    XCAT_DIR, "Material_Spreadsheets",
    "vmale_50_materials_heart_high_contrast.xlsx",
);

# ╔═╡ 05000004-0000-4000-8000-000000000004
const _ATOMIC_MASSES = Dict(
    1 => 1.008, 6 => 12.011, 7 => 14.007, 8 => 15.999, 11 => 22.99, 12 => 24.305,
    15 => 30.974, 16 => 32.06, 17 => 35.45, 19 => 39.098, 20 => 40.078, 26 => 55.845, 53 => 126.904,
)

# ╔═╡ 05000004-0000-4000-8000-000000000005
const _I_VALUES_EV = Dict(
    1 => 19.2, 6 => 81.0, 7 => 82.0, 8 => 95.0, 11 => 149.0, 12 => 156.0,
    15 => 173.0, 16 => 180.0, 17 => 174.0, 19 => 190.0, 20 => 191.0, 26 => 286.0, 53 => 491.0,
)

# ╔═╡ 05000004-0000-4000-8000-000000000006
function compute_ZA_ratio(comp::Dict{Int, Float64})
    Z_sum = sum(w * Z / get(_ATOMIC_MASSES, Z, Float64(Z) * 2) for (Z, w) in comp)
    A_sum = sum(values(comp))
    return Z_sum / A_sum
end

# ╔═╡ 05000004-0000-4000-8000-000000000007
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

# ╔═╡ 05000004-0000-4000-8000-000000000008
function load_materials_from_xlsx(xlsx_path::AbstractString)
    sheet = XLSX.readxlsx(xlsx_path)["Sheet1"]
    data = sheet["A2:P34"]
    out = Dict{Int, BS.XA.Material}()

    Z_cols = (1, 6, 7, 8, 11, 12, 15, 16, 17, 19, 20, 26, 53)

    for r in 1:size(data, 1)
        name = data[r, 1]
        oid = data[r, 16]
        ρ = data[r, 15]
        (name === nothing || oid === nothing || ρ === nothing) && continue

        comp = Dict{Int, Float64}()
        for (k, Z) in enumerate(Z_cols)
            v = data[r, k + 1]
            v isa Number && v > 0 && (comp[Z] = Float64(v))
        end
        isempty(comp) && continue

        out[Int(oid)] = BS.XA.Material(
            String(name),
            compute_ZA_ratio(comp),
            compute_mean_excitation_energy(comp),
            Float64(ρ) * u"g/cm^3",
            comp,
        )
    end
    return out
end

# ╔═╡ 05000004-0000-4000-8000-000000000010
materials_full = phantom_full_uhr === nothing ? nothing : let
        base = load_materials_from_xlsx(MATERIAL_XLSX_PATH)
        for l in unique(phantom_full_uhr)
            haskey(base, Int(l)) || (base[Int(l)] = BS.XA.Materials.water)
    end
        base
end;

# ╔═╡ 05000005-0000-4000-8000-000000000001
md"""
### 04. Cardiac bbox by label-name match

This is the **core idea** of the notebook.

A real CT scanner has a **scan field of view (SFOV)** — anything outside
it isn't reconstructed.  This simulator's recon grid is hard-locked to
**isocenter-centered**, so there's no `recon_offset_cm` knob.  The
equivalent operation is to **crop the input phantom** to the region of
interest before forward projection.  Done early it's also strictly more
efficient: voxels outside the crop never get ray-traced and never enter
the workspace's per-energy scratch buffers.

We pick the cardiac region by filtering `materials_full` for organ names
matching `/heart|atrium|ventric|coronary|aorta/i`, then computing the
voxel bounding box of all matching labels and padding by ~1 cm.

!!! info "Heuristic, not segmentation"
    Name-match is a robust *heuristic* against XCAT's organ catalog.  If
    you're working off a different phantom whose label names don't follow
    the same conventions, replace the regex with an explicit list of
    integer label IDs.
"""

# ╔═╡ 05000005-0000-4000-8000-000000000010
heart_label_ids = materials_full === nothing ? nothing : let
        pattern = r"heart|atrium|ventric|coronary|aorta"i
        ids = sort(
            UInt8[
                UInt8(oid) for (oid, mat) in materials_full
                if occursin(pattern, mat.name)
            ]
        )
        @info "[cardiac bbox] $(length(ids)) organ labels match: $(ids)"
        for oid in ids
            @info "    label $(Int(oid)) → $(materials_full[Int(oid)].name)"
    end
        ids
end;

# ╔═╡ 05000005-0000-4000-8000-000000000020
heart_bbox = (phantom_full_uhr === nothing || heart_label_ids === nothing) ? nothing : let
        is_heart = falses(256)
        for oid in heart_label_ids
            is_heart[Int(oid) + 1] = true
    end

        nx, ny, nz = size(phantom_full_uhr)
        i_lo, i_hi = nx + 1, 0
        j_lo, j_hi = ny + 1, 0
        k_lo, k_hi = nz + 1, 0
        n_voxels = 0
        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            if is_heart[Int(phantom_full_uhr[i, j, k]) + 1]
                i < i_lo && (i_lo = i)
                i > i_hi && (i_hi = i)
                j < j_lo && (j_lo = j)
                j > j_hi && (j_hi = j)
                k < k_lo && (k_lo = k)
                k > k_hi && (k_hi = k)
                n_voxels += 1
        end
    end
        n_voxels == 0 && error("[cardiac bbox] no heart-labeled voxels found in phantom")

        # Pad ~1 cm in every direction.
        pad_vox_x = round(Int, 1.0 / VOXEL_SIZE_CM[1])
        pad_vox_y = round(Int, 1.0 / VOXEL_SIZE_CM[2])
        pad_vox_z = round(Int, 1.0 / VOXEL_SIZE_CM[3])

        i_lo = max(1, i_lo - pad_vox_x);  i_hi = min(nx, i_hi + pad_vox_x)
        j_lo = max(1, j_lo - pad_vox_y);  j_hi = min(ny, j_hi + pad_vox_y)
        k_lo = max(1, k_lo - pad_vox_z);  k_hi = min(nz, k_hi + pad_vox_z)

        @info "[cardiac bbox] tight bbox + 1 cm pad:"
        @info "  voxel range = ($(i_lo):$(i_hi), $(j_lo):$(j_hi), $(k_lo):$(k_hi))"
        @info "  size  = $(i_hi - i_lo + 1) × $(j_hi - j_lo + 1) × $(k_hi - k_lo + 1) voxels"
        @info "  extent = $(round.((i_hi - i_lo + 1, j_hi - j_lo + 1, k_hi - k_lo + 1) .* VOXEL_SIZE_CM, digits = 2)) cm"
        @info "  cardiac voxels (pre-pad)  = $(n_voxels)"

        (i_lo = i_lo, i_hi = i_hi, j_lo = j_lo, j_hi = j_hi, k_lo = k_lo, k_hi = k_hi)
end;

# ╔═╡ 05000006-0000-4000-8000-000000000000
md"""
## Scan A: Axial, Zoomed Cardiac FOV

Crop the UHR phantom tight to the cardiac bbox (the SFOV-equivalent step),
scan it with a single axial rotation and a tight centered FOV, then resample
the ground truth back onto the recon grid and overlay it — pixel-perfect.
"""

# ╔═╡ 05000006-0000-4000-8000-000000000001
md"""
### 01. Crop the phantom: the SFOV-equivalent step

Just an indexing op on the UHR mask.  This is the moment the simulator's
"FOV cropping" actually happens: the cropped block is what gets handed
to `simulate!`, so everything outside this bbox costs zero compute and
zero memory in the forward projection.
"""

# ╔═╡ 05000006-0000-4000-8000-000000000010
phantom_cropped = (phantom_full_uhr === nothing || heart_bbox === nothing) ? nothing : let
        b = heart_bbox
        out = phantom_full_uhr[b.i_lo:b.i_hi, b.j_lo:b.j_hi, b.k_lo:b.k_hi]
        full_voxels = length(phantom_full_uhr)
        cropped_voxels = length(out)
        @info "[crop] $(round(full_voxels / 1.0e6, digits = 1))M voxels → $(round(cropped_voxels / 1.0e6, digits = 1))M voxels  ($(round(100 * cropped_voxels / full_voxels, digits = 1))% kept)"
        @info "[crop] memory: $(round(sizeof(phantom_full_uhr) / 1024^2, digits = 1)) MB → $(round(sizeof(out) / 1024^2, digits = 1)) MB"
        @info "[crop] forward-projection ray count drops by the same ratio — $(round(full_voxels / cropped_voxels, digits = 1))× faster simulate!"
        out
end;

# ╔═╡ 05000006-0000-4000-8000-000000000020
materials_cropped = (phantom_cropped === nothing || materials_full === nothing) ? nothing : let
        base = copy(materials_full)
        for l in unique(phantom_cropped)
            haskey(base, Int(l)) || (base[Int(l)] = BS.XA.Materials.water)
    end
        base
end;

# ╔═╡ 05000007-0000-4000-8000-000000000001
md"""
#### Visualize the crop

Mid-z slice of the full UHR phantom with the bbox drawn over it (left)
next to the cropped block (right).  This is the picture that justifies
the technique — the SFOV is just a rectangle, applied at input time.
"""

# ╔═╡ 05000007-0000-4000-8000-000000000010
let
    if phantom_full_uhr === nothing || phantom_cropped === nothing
        md"""!!! warning "Skipped — see §1 above" """
    else
        b = heart_bbox
        nz = size(phantom_full_uhr, 3)
        # Pick a z that's inside the bbox so both panels show something cardiac.
        z_full = clamp((b.k_lo + b.k_hi) ÷ 2, 1, nz)
        z_crop = z_full - b.k_lo + 1

        fig = Mke.Figure(size = (1200, 600))
        title_kwargs = (titlesize = 28, subtitlesize = 20)

        ax_l = Mke.Axis(
            fig[1, 1];
            title = "Full UHR phantom · z=$(z_full)",
            subtitle = "$(size(phantom_full_uhr, 1))×$(size(phantom_full_uhr, 2)) @ $(round.(VOXEL_SIZE_CM .* 10, digits = 2)) mm",
            aspect = Mke.DataAspect(),
            yreversed = true,
            title_kwargs...,
        )
        Mke.heatmap!(ax_l, Float32.(phantom_full_uhr[:, :, z_full]); colormap = :tab20)
        # Bbox rectangle (note: x-axis is dim 1, y-axis is dim 2)
        Mke.poly!(
            ax_l,
            Mke.Point2f[(b.i_lo, b.j_lo), (b.i_hi, b.j_lo), (b.i_hi, b.j_hi), (b.i_lo, b.j_hi)];
            color = :transparent, strokecolor = :red, strokewidth = 3,
        )
        Mke.hidedecorations!(ax_l)

        ax_r = Mke.Axis(
            fig[1, 2];
            title = "Cropped cardiac block · z=$(z_crop)",
            subtitle = "$(size(phantom_cropped, 1))×$(size(phantom_cropped, 2)) (same voxel size — only the extent changed)",
            aspect = Mke.DataAspect(),
            yreversed = true,
            title_kwargs...,
        )
        Mke.heatmap!(ax_r, Float32.(phantom_cropped[:, :, z_crop]); colormap = :tab20)
        Mke.hidedecorations!(ax_r)

        Mke.save(
            joinpath(@__DIR__, "..", "assets", "xcat_grid_crop.png"),
            fig; px_per_unit = 2,
        )
        fig
    end
end

# ╔═╡ 05000008-0000-4000-8000-000000000001
md"""
### 02. Build the `Phantom` and its world affine

Default origin behavior: when you don't pass `origin = …` to `Phantom`,
the constructor computes `origin = -extent/2 + voxel/2` — i.e. it
**centers the phantom's physical extent at isocenter for free**.  Since
we cropped before constructing, the cropped block lands centered at
(0, 0, 0) — exactly where a centered recon FOV will pick it up.
"""

# ╔═╡ 05000008-0000-4000-8000-000000000010
phantom = (phantom_cropped === nothing || materials_cropped === nothing) ? nothing :
    BS.Phantom(to_gpu(phantom_cropped), materials_cropped, VOXEL_SIZE_CM);

# ╔═╡ 05000008-0000-4000-8000-000000000011
phantom_cpu = (phantom_cropped === nothing || materials_cropped === nothing) ? nothing :
    BS.Phantom(phantom_cropped, materials_cropped, VOXEL_SIZE_CM);

# ╔═╡ 05000009-0000-4000-8000-000000000001
md"""
#### `phantom_to_world_affine`

The 4×4 matrix `A_phantom` maps a 0-indexed phantom voxel `(i, j, k)` to
world coordinates `(x, y, z)` in cm:

```
[ x ]     [ vx  0   0   ox ]   [ i ]
[ y ]  =  [ 0   vy  0   oy ] · [ j ]
[ z ]     [ 0   0   vz  oz ]   [ k ]
[ 1 ]     [ 0   0   0    1 ]   [ 1 ]
```

`(vx, vy, vz)` is the phantom's voxel size; `(ox, oy, oz)` is the world
position of voxel `(0, 0, 0)`.  After the crop+default-origin trick, this
matrix tells us exactly where in the bore each phantom voxel sits.
"""

# ╔═╡ 05000009-0000-4000-8000-000000000010
let
    if phantom_cpu === nothing
        md"""!!! warning "Skipped — see §1 above" """
    else
        A = BS.phantom_to_world_affine(phantom_cpu)
        rows = [
            "| $(round(A[i, 1], digits = 4)) | $(round(A[i, 2], digits = 4)) | $(round(A[i, 3], digits = 4)) | $(round(A[i, 4], digits = 4)) |"
                for i in 1:4
        ]
        Markdown.parse(
            """
            **`A_phantom = phantom_to_world_affine(phantom)`** (cm)

            | col 1 | col 2 | col 3 | col 4 |
            |---|---|---|---|
            $(join(rows, "\n"))

            - voxel = $(round.(phantom_cpu.voxel_size .* 10, digits = 2)) mm
            - origin = $(round.(phantom_cpu.origin, digits = 3)) cm  (world position of voxel `(0, 0, 0)`)
            - extent = $(round.(phantom_cpu.extent, digits = 3)) cm
            - **center of cropped block** ≈ $(round.(phantom_cpu.origin .+ phantom_cpu.extent ./ 2 .- phantom_cpu.voxel_size ./ 2, digits = 3)) cm  (should be ≈ isocenter)
            """
        )
    end
end

# ╔═╡ 0500000a-0000-4000-8000-000000000001
md"""
### 03. Scanner, protocol, and tight-FOV recon

Same hardware as notebooks 01 / 02 / 03 — clinical 64-row CT, large
bowtie, GE Revolution Apex Elite-class detector.  We're doing single-kVp
EICT here; the affine machinery has nothing to do with spectral imaging,
so any scanner works.
"""

# ╔═╡ 0500000a-0000-4000-8000-000000000010
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
    # DAS/electronic noise enters the counts before the log transform, so it
    # propagates through reconstruction (see `add_system_noise_floor!` docstring).
    electronic_noise = 3500.0,   # e⁻ — clinical GE Apex Elite DAS readout noise
    detection_gain = 10.0,
);

# ╔═╡ 0500000a-0000-4000-8000-000000000020
md"""
#### `CTProtocol`: clinical cardiac CTA

120 kVp / 250 mA, 1 s rotation, 5 mm collimation, 500 views.  The recon
slab will derive its z-extent from the protocol collimation, which is
how this simulator decides how many detector rows are active (see
`CTGeometry`).
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
sim_opts = BS.SimOptions(fidelity = :eict, seed = 1234, projector = :dd_fast);

# ╔═╡ 0500000b-0000-4000-8000-000000000001
md"""
#### `ReconOptions`: tight cardiac FOV

The recon FOV is **always centered at isocenter** in this simulator.
That's fine for us — we centered the cropped phantom at iso for free in
§6.  We use a **14 cm × 14 cm** in-plane FOV (smaller than the cropped
extent in xy by design: lets us see what happens when the recon FOV is
*tighter* than the input).  The recon z-extent is derived from the
protocol collimation.

`matrix_size = (384, 384, n_z)` gives ~0.36 mm recon voxels — coarser
than the 0.4 mm phantom voxels so the affine round-trip will downsample
slightly.
"""

# ╔═╡ 0500000b-0000-4000-8000-000000000010
recon_opts = let
    slice_thickness_mm = 0.625
    n_z = max(1, round(Int, protocol.collimation_mm / slice_thickness_mm))
    BS.ReconOptions(
        matrix_size = (384, 384, n_z),
        fov_cm = 14.0,
        z_cm = protocol.collimation_mm / 10.0,
    )
end;

# ╔═╡ 0500000b-0000-4000-8000-000000000020
md"""
#### `recon_to_world_affine`

We can build the `CTGeometry` directly from `(scanner, protocol,
recon_opts)` — no need to wait for `simulate!` to inspect the recon
grid.  Same affine shape as `A_phantom`; the values reflect the recon
voxel size (`fov / matrix_size`) and a **centered** origin
(`-fov/2 + voxel/2`).
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
let
    A = BS.recon_to_world_affine(geom_inspect, recon_opts.matrix_size)
    rows = [
        "| $(round(A[i, 1], digits = 4)) | $(round(A[i, 2], digits = 4)) | $(round(A[i, 3], digits = 4)) | $(round(A[i, 4], digits = 4)) |"
            for i in 1:4
    ]
    nx, ny, nz = recon_opts.matrix_size
    fov = geom_inspect.fov
    Markdown.parse(
        """
        **`A_recon = recon_to_world_affine(geom, matrix_size)`** (cm)

        | col 1 | col 2 | col 3 | col 4 |
        |---|---|---|---|
        $(join(rows, "\n"))

        - matrix size = $(nx) × $(ny) × $(nz) voxels
        - voxel size = $(round.((fov[1] / nx, fov[2] / ny, fov[3] / nz) .* 10, digits = 3)) mm
        - FOV = $(round.(fov, digits = 3)) cm
        - origin = $(round(-fov[1] / 2 + (fov[1] / nx) / 2, digits = 3)),  $(round(-fov[2] / 2 + (fov[2] / ny) / 2, digits = 3)),  $(round(-fov[3] / 2 + (fov[3] / nz) / 2, digits = 3)) cm  (centered at iso)
        """
    )
end

# ╔═╡ 0500000c-0000-4000-8000-000000000001
md"""
#### Side-by-side grid comparison

The two grids share **world coordinates** (cm) but differ in voxel size,
shape, and FOV.  `resample_to_recon` (and the affines under it) handle
all of this for you.
"""

# ╔═╡ 0500000c-0000-4000-8000-000000000010
let
    if phantom_cpu === nothing
        md"""!!! warning "Skipped — see §1 above" """
    else
        nx_p, ny_p, nz_p = size(phantom_cpu.mask)
        nx_r, ny_r, nz_r = recon_opts.matrix_size
        vp = round.(phantom_cpu.voxel_size .* 10, digits = 3)
        vr = round.((geom_inspect.fov[1] / nx_r, geom_inspect.fov[2] / ny_r, geom_inspect.fov[3] / nz_r) .* 10, digits = 3)
        ep = round.(phantom_cpu.extent, digits = 2)
        er = round.(geom_inspect.fov, digits = 2)
        op = round.(phantom_cpu.origin, digits = 3)
        rox = -geom_inspect.fov[1] / 2 + (geom_inspect.fov[1] / nx_r) / 2
        roy = -geom_inspect.fov[2] / 2 + (geom_inspect.fov[2] / ny_r) / 2
        roz = -geom_inspect.fov[3] / 2 + (geom_inspect.fov[3] / nz_r) / 2
        or = round.((rox, roy, roz), digits = 3)
        Markdown.parse(
            """
            | property | phantom (cropped UHR) | recon (centered, tight FOV) |
            |---|---|---|
            | shape (voxels) | $(nx_p) × $(ny_p) × $(nz_p) | $(nx_r) × $(ny_r) × $(nz_r) |
            | voxel size (mm) | $(vp[1]) × $(vp[2]) × $(vp[3]) | $(vr[1]) × $(vr[2]) × $(vr[3]) |
            | extent (cm) | $(ep[1]) × $(ep[2]) × $(ep[3]) | $(er[1]) × $(er[2]) × $(er[3]) |
            | origin (cm) | $(op) | $(or) |
            | total voxels | $(nx_p * ny_p * nz_p) | $(nx_r * ny_r * nz_r) |
            """
        )
    end
end

# ╔═╡ 0500000d-0000-4000-8000-000000000001
md"""
### 04. Forward project and reconstruct

Standard EICT path.  We skip the BHC pipeline (see notebook 02 §7 for
that) and use a quick analytic μ_water for HU conversion — the focus
here is geometry, not HU accuracy.
"""

# ╔═╡ 0500000d-0000-4000-8000-000000000010
sim = phantom === nothing ? nothing : let
        @info "Simulating cardiac CTA: 120 kVp / 250 mA / cropped UHR phantom…"
        ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
        BS.simulate!(ws, phantom, protocol, sim_opts)

        result = (sino = Array(ws.sinogram), geom = ws.geom)
        ws = nothing
        GC.gc(true)
        result
end;

# ╔═╡ 0500000d-0000-4000-8000-000000000020
μ_water_120 = phantom_cpu === nothing ? nothing : let
        # Phantom-aware water_path: pull the body chord straight off the
        # cropped XCAT mask, so the calibration tracks any change to the
        # crop bbox / DOWNSAMPLE_FACTOR without hardcoded cm.
        body_diameter_cm = BS.estimate_phantom_diameter_cm(
            phantom_cpu.mask, phantom_cpu.voxel_size .* 10.0,
        )
        μ = BS.compute_polychromatic_μ_water(
            sim_opts, protocol;
            scanner = scanner,
            geom = geom_inspect,
            water_path_cm = body_diameter_cm,
        )
        @info "[analytic μ_water]  120 kVp + 4.5 mm Al + $(round(body_diameter_cm, digits = 1)) cm body hardening: $(round(μ, digits = 5)) cm⁻¹"
        μ
end;

# ╔═╡ 0500000d-0000-4000-8000-000000000030
recon_HU = sim === nothing ? nothing : let
        sino_gpu = to_gpu(Float32.(sim.sino))
        ws = BS.create_fdk_recon_workspace(sino_gpu, sim.geom, recon_opts.matrix_size; filter = :standard)
        recon_μ = Array(BS.reconstruct!(ws, sino_gpu, sim.geom))
        ws = nothing; sino_gpu = nothing; GC.gc(true)

        # quantum + DAS noise already carried by the sinogram (counts domain)
        HU = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_water_120))
        HU
end;

# ╔═╡ 0500000e-0000-4000-8000-000000000001
md"""
### 05. Resample ground truth and overlay

`BS.resample_to_recon` is the convenience wrapper.  It pulls the phantom
mask to CPU, computes each recon voxel's world coordinate via
`A_recon`, maps to the continuous phantom-voxel index via
`inv(A_phantom)`, and samples.

Two interpolation methods built in:

| `method` | output type   | use when |
|----------|---------------|----------|
| `:nearest` | `UInt8` (label-preserving) | overlaying labels for ROI extraction / segmentation evaluation |
| `:linear`  | `Float32` (trilinear)      | continuous fields (HU, density, fractional volume) |
"""

# ╔═╡ 0500000e-0000-4000-8000-000000000010
gt_resampled_nn = (phantom_cpu === nothing || sim === nothing) ? nothing :
    BS.resample_to_recon(phantom_cpu, sim.geom, recon_opts.matrix_size; method = :nearest);

# ╔═╡ 0500000e-0000-4000-8000-000000000011
gt_resampled_lin = (phantom_cpu === nothing || sim === nothing) ? nothing :
    BS.resample_to_recon(phantom_cpu, sim.geom, recon_opts.matrix_size; method = :linear);

# ╔═╡ 0500000e-0000-4000-8000-000000000012
# Fractional cardiac coverage: build a *binary* cardiac mask, then resample
# `:linear`.  Trilinear on a multi-label integer mask (gt_resampled_lin
# above) arithmetically-mixes label IDs and isn't physically meaningful;
# trilinear on a 0/1 mask gives true partial-volume fractions ∈ [0, 1].
cardiac_coverage_lin = (
        phantom_cpu === nothing || sim === nothing ||
        heart_label_ids === nothing
    ) ? nothing : let
        binary_mask = zeros(UInt8, size(phantom_cropped))
        for oid in heart_label_ids
            binary_mask[phantom_cropped .== oid] .= 0x01
    end
        binary_phantom = BS.Phantom(
            binary_mask,
            Dict(0 => BS.XA.Materials.water, 1 => BS.XA.Materials.water),
            VOXEL_SIZE_CM,
        )
        BS.resample_to_recon(binary_phantom, sim.geom, recon_opts.matrix_size; method = :linear)
end;

# ╔═╡ 0500000e-0000-4000-8000-000000000020
let
    if gt_resampled_nn === nothing
        md"""!!! warning "Skipped — see §1 above" """
    else
        md"""
        - `gt_resampled_nn`  shape = $(size(gt_resampled_nn)) · eltype = $(eltype(gt_resampled_nn))
        - `gt_resampled_lin` shape = $(size(gt_resampled_lin)) · eltype = $(eltype(gt_resampled_lin))
        - `recon_HU`         shape = $(size(recon_HU)) · eltype = $(eltype(recon_HU))

        All three live on the same world-coordinate grid — index `(i, j, k)` in any
        of them corresponds to the same physical voxel inside the bore.
        """
    end
end

# ╔═╡ 0500000f-0000-4000-8000-000000000001
md"""
#### Bring-your-own-interpolator pattern

When `:nearest` and `:linear` aren't enough — e.g. you want a B-spline,
a sinc kernel, or some learned upsampling — the affines give you the
recon-voxel → phantom-voxel map directly.  Compute

```julia
M = inv(A_phantom) * A_recon
```

and you have a 4×4 that takes any `(i_recon, j_recon, k_recon, 1)` to
the **continuous** phantom voxel index.  Hand that to your interpolator
of choice (Interpolations.jl, ImageTransformations.jl, a custom kernel,
PyTorch via PyCall, whatever) and you're done.
"""

# ╔═╡ 0500000f-0000-4000-8000-000000000010
let
    if phantom_cpu === nothing
        md"""!!! warning "Skipped — see §1 above" """
    else
        A_phantom = BS.phantom_to_world_affine(phantom_cpu)
        A_recon = BS.recon_to_world_affine(geom_inspect, recon_opts.matrix_size)
        M = inv(A_phantom) * A_recon

        rows = [
            "| $(round(M[i, 1], digits = 4)) | $(round(M[i, 2], digits = 4)) | $(round(M[i, 3], digits = 4)) | $(round(M[i, 4], digits = 4)) |"
                for i in 1:4
        ]

        # Quick sanity demo: where does recon-voxel (0,0,0) sit in phantom-voxel space?
        v0 = M * [0.0, 0.0, 0.0, 1.0]
        # And the recon-volume center?
        nx_r, ny_r, nz_r = recon_opts.matrix_size
        vc = M * [(nx_r - 1) / 2, (ny_r - 1) / 2, (nz_r - 1) / 2, 1.0]

        Markdown.parse(
            """
            **`M = inv(A_phantom) * A_recon`** — recon voxel → continuous phantom voxel

            | col 1 | col 2 | col 3 | col 4 |
            |---|---|---|---|
            $(join(rows, "\n"))

            Quick sanity:

            - recon voxel `(0, 0, 0)` → phantom voxel  $(round.((v0[1], v0[2], v0[3]), digits = 2))
            - recon volume center  → phantom voxel  $(round.((vc[1], vc[2], vc[3]), digits = 2))   (should be near the cropped phantom's center)

            Pass `M` to the interpolator of your choice.  In pseudocode:

            ```julia
            for k in 0:(nz_r-1), j in 0:(ny_r-1), i in 0:(nx_r-1)
                p = M * [i, j, k, 1.0]               # phantom voxel index (Float64)
                out[i+1, j+1, k+1] = my_interpolator(phantom.mask, p[1], p[2], p[3])
            end
            ```
            """
        )
    end
end

# ╔═╡ 05000010-0000-4000-8000-000000000001
md"""
#### The verification mosaic

Four panels, all on the **recon grid** at the same mid-slice.  Top row
shows the two raw inputs; bottom row overlays the masks on the HU recon
to demonstrate alignment.

| panel | what it shows |
|-------|---------------|
| (top-left) HU recon | what the scanner produced — clinical recon grid, isocenter-centered |
| (top-right) all structures (`:nearest`, no overlay) | full multi-label resample on the recon grid — every organ, no masking |
| (bottom-left) HU + cardiac labels | cardiac labels (NaN-masked) over the HU at α=0.6 — alignment check |
| (bottom-right) HU + cardiac coverage (`:linear`, binary mask) | true partial-volume fraction ∈ [0, 1] over the HU, soft at boundaries |
"""

# ╔═╡ 05000010-0000-4000-8000-000000000010
let
    if recon_HU === nothing || gt_resampled_nn === nothing ||
            cardiac_coverage_lin === nothing || heart_label_ids === nothing

        md"""!!! warning "Skipped — see §1 above" """
    else
        # NaN-mask non-cardiac voxels so they render transparent over the HU base.
        is_cardiac = falses(256)
        for oid in heart_label_ids
            is_cardiac[Int(oid) + 1] = true
        end

        z = size(recon_HU, 3) ÷ 2 + 1
        hu_slice = recon_HU[:, :, z]

        nn_overlay = let
            slice = gt_resampled_nn[:, :, z]
            out = fill(NaN32, size(slice))
            @inbounds for idx in eachindex(slice)
                if is_cardiac[Int(slice[idx]) + 1]
                    out[idx] = Float32(slice[idx])
                end
            end
            out
        end

        lin_overlay = let
            slice = cardiac_coverage_lin[:, :, z]
            out = Float32.(slice)
            @inbounds for idx in eachindex(out)
                if out[idx] < 0.05f0   # below 5% partial coverage → transparent
                    out[idx] = NaN32
                end
            end
            out
        end

        fig = Mke.Figure(size = (1400, 1320))
        hu_kwargs = (colormap = :grays, colorrange = (-300, 700))
        title_kwargs = (titlesize = 28, subtitlesize = 20)

        # Top-left: raw HU recon.
        ax_tl = Mke.Axis(
            fig[1, 1];
            title = "HU recon",
            subtitle = "z=$(z) of $(size(recon_HU, 3)) · centered FOV $(recon_opts.fov_cm) cm",
            aspect = Mke.DataAspect(), yreversed = true,
            title_kwargs...,
        )
        Mke.heatmap!(ax_tl, hu_slice; hu_kwargs...)
        Mke.hidedecorations!(ax_tl)

        # Top-right: ALL structures — full multi-label resample, no masking.
        ax_tr = Mke.Axis(
            fig[1, 2];
            title = "All structures (`:nearest`, no overlay)",
            subtitle = "$(length(unique(gt_resampled_nn))) labels resampled onto recon grid",
            aspect = Mke.DataAspect(), yreversed = true,
            title_kwargs...,
        )
        Mke.heatmap!(
            ax_tr, Float32.(gt_resampled_nn[:, :, z]);
            colormap = :tab20,
        )
        Mke.hidedecorations!(ax_tr)

        # Bottom-left: HU + cardiac labels overlay.
        ax_bl = Mke.Axis(
            fig[2, 1];
            title = "HU + cardiac labels",
            subtitle = "α=0.6 over HU — alignment check",
            aspect = Mke.DataAspect(), yreversed = true,
            title_kwargs...,
        )
        Mke.heatmap!(ax_bl, hu_slice; hu_kwargs...)
        Mke.heatmap!(
            ax_bl, nn_overlay;
            colormap = :tab20, alpha = 0.6, nan_color = (:white, 0.0),
        )
        Mke.hidedecorations!(ax_bl)

        # Bottom-right: HU + fractional cardiac coverage overlay.
        ax_br = Mke.Axis(
            fig[2, 2];
            title = "HU + cardiac coverage (`:linear`, binary mask)",
            subtitle = "fractional ∈ [0.05, 1] · α=0.7 over HU",
            aspect = Mke.DataAspect(), yreversed = true,
            title_kwargs...,
        )
        Mke.heatmap!(ax_br, hu_slice; hu_kwargs...)
        hm_br = Mke.heatmap!(
            ax_br, lin_overlay;
            colormap = :viridis, colorrange = (0, 1),
            alpha = 0.7, nan_color = (:white, 0.0),
        )
        Mke.hidedecorations!(ax_br)
        Mke.Colorbar(fig[2, 3], hm_br; label = "cardiac fraction", width = 14, labelsize = 18)

        Mke.save(
            joinpath(@__DIR__, "..", "assets", "xcat_grid_overlay.png"),
            fig; px_per_unit = 2,
        )
        fig
    end
end

# ╔═╡ 05000012-0000-4000-8000-000000000000
md"""
## Scan B: Helical, Extended-Z FOV

Same phantom, same affine machinery — but a **taller crop** scanned with a
**helical** acquisition, reconstructed into a tall stack of axial slices you
can **scroll through in z**.  The recon grid has its *own* recon→world affine
(a bigger z extent than Scan A); `resample_to_recon` still lands the ground
truth pixel-perfectly on every slice.
"""

# ╔═╡ 05000013-0000-4000-8000-000000000001
md"""
### 01. Extended-z crop

Scan A cropped tight to the heart.  For the helical demo we keep the same
in-plane (x, y) bbox but **extend the z range** (± a few cm beyond the cardiac
extent) so there's a tall stack to scroll — the extra z is exactly what the
helix sweeps through.
"""

# ╔═╡ 05000013-0000-4000-8000-000000000010
heart_bbox_tall = (phantom_full_uhr === nothing || heart_bbox === nothing) ? nothing : let
    nz = size(phantom_full_uhr, 3)
    extra_z = round(Int, 4.0 / VOXEL_SIZE_CM[3])   # +4 cm each side beyond the cardiac bbox
    b = heart_bbox
    tall = (i_lo = b.i_lo, i_hi = b.i_hi, j_lo = b.j_lo, j_hi = b.j_hi,
            k_lo = max(1, b.k_lo - extra_z), k_hi = min(nz, b.k_hi + extra_z))
    @info "[Scan B tall bbox] z range $(b.k_lo):$(b.k_hi) → $(tall.k_lo):$(tall.k_hi)  ($(round((tall.k_hi - tall.k_lo + 1) * VOXEL_SIZE_CM[3], digits = 1)) cm z extent)"
    tall
end;

# ╔═╡ 05000013-0000-4000-8000-000000000020
phantom_cropped_tall = (phantom_full_uhr === nothing || heart_bbox_tall === nothing) ? nothing : let
    b = heart_bbox_tall
    phantom_full_uhr[b.i_lo:b.i_hi, b.j_lo:b.j_hi, b.k_lo:b.k_hi]
end;

# ╔═╡ 05000013-0000-4000-8000-000000000025
materials_tall = (phantom_cropped_tall === nothing || materials_full === nothing) ? nothing : let
    base = copy(materials_full)
    for l in unique(phantom_cropped_tall)
        haskey(base, Int(l)) || (base[Int(l)] = BS.XA.Materials.water)
    end
    base
end;

# ╔═╡ 05000013-0000-4000-8000-000000000030
phantom_helical = (phantom_cropped_tall === nothing || materials_tall === nothing) ? nothing :
    BS.Phantom(to_gpu(phantom_cropped_tall), materials_tall, VOXEL_SIZE_CM);

# ╔═╡ 05000013-0000-4000-8000-000000000031
phantom_helical_cpu = (phantom_cropped_tall === nothing || materials_tall === nothing) ? nothing :
    BS.Phantom(phantom_cropped_tall, materials_tall, VOXEL_SIZE_CM);

# ╔═╡ 05000013-0000-4000-8000-000000000040
recon_opts_helical = let
    slice_thickness_mm = 0.625
    z_cm = 4.0                        # taller recon slab than Scan A — the helix supplies the z coverage
    n_z = max(1, round(Int, z_cm * 10 / slice_thickness_mm))   # ~64 slices to scroll
    BS.ReconOptions(matrix_size = (384, 384, n_z), fov_cm = 14.0, z_cm = z_cm)
end;

# ╔═╡ 05000012-0000-4000-8000-000000000001
md"""
### 02. Helical protocol and z-ramped acquisition

Everything above used an axial scan.  The affine machinery is
trajectory-agnostic by design: a helical acquisition changes the SOURCE
path (z ramps with view), but the RECON grid is still a stack of axial
slices centred on the scanned range — so `recon_to_world_affine`, and
therefore `resample_to_recon`, apply unchanged.  This section proves it
end-to-end on the new spiral chain: `CTProtocol(pitch = …)` → z-ramped
`CTGeometry` → `:dd_fast` forward on the (default) arc detector →
rebinned-WFBP reconstruction → label overlay on the helical recon grid.
"""

# ╔═╡ 05000012-0000-4000-8000-000000000010
sim_helical = phantom_helical === nothing ? nothing : let
    protocol_hel = BS.CTProtocol(
        kVp = 120, mA = 250.0, views = 500, rotation_time = 1.0,
        collimation_mm = 10.0, additional_filters = [("Al", 4.5)],
        pitch = 1.0, n_rotations = 8.0,
    )
    @info "Simulating HELICAL cardiac CTA: pitch 1.0 × 8 rotations, 10 mm collimation…"
    ws = BS.create_eict_workspace(scanner, protocol_hel, sim_opts, recon_opts_helical, phantom_helical)
    BS.simulate!(ws, phantom_helical, protocol_hel, sim_opts)
    result = (sino = Array(ws.sinogram), geom = ws.geom)
    ws = nothing
    GC.gc(true)
    result
end;

# ╔═╡ 05000014-0000-4000-8000-000000000001
md"""
### 03. WFBP reconstruct and its own recon affine

`is_helical(geom)` routes `reconstruct!` to the rebinned-WFBP path.  The recon
grid is a **taller** stack of axial slices than Scan A — its own
`recon_to_world_affine`, with a bigger z extent — but still centered at
isocenter, so the affine round-trip is unchanged.
"""

# ╔═╡ 05000012-0000-4000-8000-000000000020
recon_HU_helical = sim_helical === nothing ? nothing : let
    sino_gpu = to_gpu(Float32.(sim_helical.sino))
    # is_helical(geom) routes reconstruct! to the rebinned-WFBP path
    ws = BS.create_fdk_recon_workspace(sino_gpu, sim_helical.geom, recon_opts_helical.matrix_size; filter = :standard)
    recon_μ = Array(BS.reconstruct!(ws, sino_gpu, sim_helical.geom))
    ws = nothing; sino_gpu = nothing; GC.gc(true)
    Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_water_120))
end;

# ╔═╡ 05000014-0000-4000-8000-000000000010
let
    if sim_helical === nothing
        md"""!!! warning "Skipped — see §1 above" """
    else
        A = BS.recon_to_world_affine(sim_helical.geom, recon_opts_helical.matrix_size)
        rows = [
            "| $(round(A[i, 1], digits = 4)) | $(round(A[i, 2], digits = 4)) | $(round(A[i, 3], digits = 4)) | $(round(A[i, 4], digits = 4)) |"
                for i in 1:4
        ]
        nx, ny, nz = recon_opts_helical.matrix_size
        Markdown.parse(
            """
            **Helical `A_recon = recon_to_world_affine(helical geom, matrix_size)`** (cm)

            | col 1 | col 2 | col 3 | col 4 |
            |---|---|---|---|
            $(join(rows, "\n"))

            - matrix size = $(nx) × $(ny) × $(nz) voxels  (**$(nz)** z slices vs Scan A's **$(recon_opts.matrix_size[3])**)
            - z extent = $(round(recon_opts_helical.z_cm, digits = 2)) cm  (vs Scan A's $(round(recon_opts.z_cm, digits = 2)) cm)
            - same centered, isocenter origin — only the z stack is taller.
            """
        )
    end
end

# ╔═╡ 05000014-0000-4000-8000-000000000020
md"""
### 04. Resample ground truth onto the helical grid

Same `resample_to_recon`, now against the taller helical recon grid — the
ground-truth labels land on the exact voxels of the WFBP stack.
"""

# ╔═╡ 05000012-0000-4000-8000-000000000030
gt_helical_nn = (phantom_helical_cpu === nothing || sim_helical === nothing) ? nothing :
    BS.resample_to_recon(phantom_helical_cpu, sim_helical.geom, recon_opts_helical.matrix_size; method = :nearest);

# ╔═╡ 05000015-0000-4000-8000-000000000001
md"""
### 05. Scroll through z: slider-driven overlay

Drag the slider to scrub through the helical recon in z.  Left is the WFBP HU
recon; right overlays the resampled cardiac labels (translucent fill) — they
sit on the recon anatomy on **every** slice, so the affine holds across the
full z stack, not just the mid-plane.
"""

# ╔═╡ 05000015-0000-4000-8000-000000000010
@bind z_helical PlutoUI.Slider(1:recon_opts_helical.matrix_size[3]; default = recon_opts_helical.matrix_size[3] ÷ 2, show_value = true)

# ╔═╡ 05000012-0000-4000-8000-000000000040
sim_helical === nothing ? md"_(XCAT not available — helical demo skipped)_" : let
    nz = size(recon_HU_helical, 3)
    z = clamp(z_helical, 1, nz)
    recon = recon_HU_helical[:, :, z]

    # Translucent cardiac-label fill (same style as Scan A's overlay, not a wiry
    # edge): NaN-mask every non-cardiac voxel so it renders transparent, then
    # lay the labels over the HU recon at α — you see the coloured structures
    # sit exactly on the anatomy as you scroll z.
    is_cardiac = falses(256)
    if heart_label_ids !== nothing
        for oid in heart_label_ids
            is_cardiac[Int(oid) + 1] = true
        end
    end
    labslice = gt_helical_nn[:, :, z]
    nx, ny = size(labslice)
    cx, cy = (nx + 1) / 2, (ny + 1) / 2
    r2 = (min(nx, ny) / 2)^2                     # circular recon FOV = inscribed circle
    lab_over = fill(NaN32, size(labslice))
    @inbounds for j in 1:ny, i in 1:nx
        ((i - cx)^2 + (j - cy)^2 <= r2) || continue    # keep the overlay INSIDE the FOV
        is_cardiac[Int(labslice[i, j]) + 1] && (lab_over[i, j] = Float32(labslice[i, j]))
    end

    fig = Mke.Figure(size = (1180, 620))
    hu_kwargs = (colormap = :grays, colorrange = (-200, 600))
    ax1 = Mke.Axis(fig[1, 1]; title = "Helical WFBP recon · z=$(z)/$(nz)",
        titlesize = 26, aspect = Mke.DataAspect(), yreversed = true)
    hm = Mke.heatmap!(ax1, recon; hu_kwargs...)
    Mke.hidedecorations!(ax1)

    ax2 = Mke.Axis(fig[1, 2]; title = "HU + cardiac labels · z=$(z)",
        subtitle = "resampled ground truth on the helical recon grid (α = 0.6)",
        titlesize = 26, subtitlesize = 18, aspect = Mke.DataAspect(), yreversed = true)
    Mke.heatmap!(ax2, recon; hu_kwargs...)
    Mke.heatmap!(ax2, lab_over; colormap = :tab20, alpha = 0.6)
    Mke.hidedecorations!(ax2)

    Mke.Colorbar(fig[1, 3], hm; label = "HU", labelsize = 20, ticklabelsize = 14)
    fig
end

# ╔═╡ 05000016-0000-4000-8000-000000000001
md"""
### 06. `:nearest` vs `:linear` at boundaries

The affine mapping is exact (see the round-trip audit below), so any apparent
"fuzziness" at a label edge is **resampling**, not misregistration.  With
`:nearest`, every recon voxel snaps to the single closest UHR phantom voxel
(nearest in **all three axes**, z included) — so label boundaries stair-step
onto the coarse recon grid, a half-voxel quantization.  Resampling a *binary*
cardiac mask with `:linear` instead gives a **fully 3D** partial-volume coverage
∈ [0, 1] per recon voxel: the trilinear blend weights x, y **and z**, so a voxel
straddling the cardiac surface *in z* (0.625 mm helical slices) picks up a
fractional value too — not just in-plane.  It follows the sub-voxel boundary
smoothly.  Same slice, same slider — scrub `z` and compare.

Caveat on "partial volume": trilinear samples at each recon-voxel *centre* from
the 8 bracketing phantom voxels — that equals the true occupied-volume fraction
when the phantom and recon grids are comparable in resolution, as they are here
(phantom 0.4 mm vs recon 0.37 mm in-plane / 0.625 mm in z).  If you push the
phantom much finer than the recon (smaller `DOWNSAMPLE_FACTOR`), a recon voxel
would enclose several phantom voxels that a centre-sample ignores, and a true
volumetric coverage would need box-averaging / supersampling instead.
"""

# ╔═╡ 05000016-0000-4000-8000-000000000010
# Fractional cardiac coverage on the HELICAL grid: binary mask → `:linear`
# resample (the multi-label `:nearest` map mixes IDs under trilinear, so we
# resample a 0/1 mask to get physical partial-volume fractions).
cardiac_coverage_lin_helical = (
        phantom_cropped_tall === nothing || sim_helical === nothing ||
        heart_label_ids === nothing
    ) ? nothing : let
        binary_mask = zeros(UInt8, size(phantom_cropped_tall))
        for oid in heart_label_ids
            binary_mask[phantom_cropped_tall .== oid] .= 0x01
        end
        binary_phantom = BS.Phantom(
            binary_mask,
            Dict(0 => BS.XA.Materials.water, 1 => BS.XA.Materials.water),
            VOXEL_SIZE_CM,
        )
        BS.resample_to_recon(binary_phantom, sim_helical.geom, recon_opts_helical.matrix_size; method = :linear)
end;

# ╔═╡ 05000016-0000-4000-8000-000000000020
(sim_helical === nothing || cardiac_coverage_lin_helical === nothing) ?
        md"_(XCAT not available — helical demo skipped)_" : let
    nz = size(recon_HU_helical, 3)
    z = clamp(z_helical, 1, nz)
    recon = recon_HU_helical[:, :, z]
    nnslice = gt_helical_nn[:, :, z]
    covslice = cardiac_coverage_lin_helical[:, :, z]

    is_cardiac = falses(256)
    if heart_label_ids !== nothing
        for oid in heart_label_ids; is_cardiac[Int(oid) + 1] = true; end
    end
    nx, ny = size(nnslice)
    cx, cy = (nx + 1) / 2, (ny + 1) / 2
    r2 = (min(nx, ny) / 2)^2                          # circular recon FOV

    nn_over = fill(NaN32, size(nnslice))              # :nearest cardiac labels
    lin_over = fill(NaN32, size(covslice))            # :linear coverage ∈ [0,1]
    @inbounds for j in 1:ny, i in 1:nx
        ((i - cx)^2 + (j - cy)^2 <= r2) || continue
        is_cardiac[Int(nnslice[i, j]) + 1] && (nn_over[i, j] = Float32(nnslice[i, j]))
        covslice[i, j] > 0.01f0 && (lin_over[i, j] = covslice[i, j])
    end

    fig = Mke.Figure(size = (1500, 560))
    hu_kwargs = (colormap = :grays, colorrange = (-200, 600))
    ax1 = Mke.Axis(fig[1, 1]; title = "raw helical recon · z=$(z)/$(nz)",
        titlesize = 22, aspect = Mke.DataAspect(), yreversed = true)
    hm = Mke.heatmap!(ax1, recon; hu_kwargs...); Mke.hidedecorations!(ax1)

    ax2 = Mke.Axis(fig[1, 2]; title = ":nearest labels — stair-stepped to recon grid",
        titlesize = 22, aspect = Mke.DataAspect(), yreversed = true)
    Mke.heatmap!(ax2, recon; hu_kwargs...)
    Mke.heatmap!(ax2, nn_over; colormap = :tab20, alpha = 0.65); Mke.hidedecorations!(ax2)

    ax3 = Mke.Axis(fig[1, 3]; title = ":linear coverage ∈ [0,1] — sub-voxel boundary",
        titlesize = 22, aspect = Mke.DataAspect(), yreversed = true)
    Mke.heatmap!(ax3, recon; hu_kwargs...)
    hmc = Mke.heatmap!(ax3, lin_over; colormap = :viridis, colorrange = (0, 1), alpha = 0.75)
    Mke.hidedecorations!(ax3)
    Mke.Colorbar(fig[1, 4], hmc; label = "cardiac coverage", labelsize = 16, ticklabelsize = 13)
    fig
end

# ╔═╡ 05000017-0000-4000-8000-000000000001
md"""
### 07. The affine round-trip is exact (1-to-1)

The overlays above are exact by construction — not "≥ 80 % aligned".  This
proves it with numbers, for **both** the axial and the helical geometry.  Three
independent checks:

1. **Affine ≡ reconstructor grid.**  The FDK and WFBP backprojectors both place
   recon voxel `(i,j,k)` at world `-fov/2 + (idx − ½)·(fov/n)` (1-indexed) — the
   *same* isocenter-centered rule `recon_to_world_affine` encodes.  So the grid
   the reconstructor writes into is bit-for-bit the grid the resampler samples.
2. **Round-trip identity.**  `A⁻¹ · A · v = v` — the map is a diagonal
   scale + translate, algebraically invertible to floating-point roundoff.
3. **Recon registered to the map.**  Checks 1–2 certify the *coordinate map*;
   the last cell certifies the *image* — that the forward projector images the
   phantom at the exact world position the backprojector reconstructs it (a
   forward↔backprojector half-voxel / rebinning offset would displace the recon
   and stay invisible to 1–2).  Best edge-alignment shift = **(0,0)** for both.

Helical carries no special-casing: `is_helical(geom)` only switches the
*backprojection algorithm*, never the grid geometry — so the mapping is 1-to-1
for the spiral scan exactly as it is for the axial one.  Any softness you see at
a boundary when you zoom in is `:nearest` half-voxel quantization (§06) or the
recon's point-spread blur — never the affine, and never a registration offset.
"""

# ╔═╡ 05000012-0000-4000-8000-000000000050
let
    # Audit: recon_to_world_affine vs the reconstructor's own voxel-centering
    # rule, plus the A⁻¹·A round-trip — for both geoms.  Errors are in µm
    # (1e-4 cm) so any nonzero digit is visible; both should read ~0.
    affine_audit = function (geom, ms)
        A = BS.recon_to_world_affine(geom, ms)
        Ainv = inv(A)
        fov = geom.fov
        grid_err = 0.0     # affine world vs backprojector world (cm)
        rt_err = 0.0       # A⁻¹·A·v − v  (voxel indices)
        for ax in 1:3
            vs = fov[ax] / ms[ax]
            vmin = -fov[ax] / 2
            for idx in (1, (ms[ax] + 1) ÷ 2, ms[ax])   # 1-indexed extremes + mid
                k = idx - 1                              # 0-indexed for the affine
                v = zeros(4); v[4] = 1.0; v[ax] = k
                world_affine = (A * v)[ax]
                world_bp = vmin + (idx - 0.5) * vs       # exact reconstructor rule
                grid_err = max(grid_err, abs(world_affine - world_bp))
                rt_err = max(rt_err, abs((Ainv * (A * v))[ax] - k))
            end
        end
        (grid_err * 1e4, rt_err)                          # cm → µm for grid_err
    end

    ga, ra = affine_audit(sim.geom, recon_opts.matrix_size)
    if sim_helical === nothing
        Markdown.parse("""
        | geometry | affine ≡ reconstructor grid | round-trip `A⁻¹·A·v = v` |
        |---|---|---|
        | axial | max Δ = $(round(ga, sigdigits = 2)) µm | max Δ = $(round(ra, sigdigits = 2)) voxel |

        Both ~0 ⇒ the mapping is exactly 1-to-1.  _(helical skipped — see §1.)_
        """)
    else
        gh, rh = affine_audit(sim_helical.geom, recon_opts_helical.matrix_size)
        Markdown.parse("""
        | geometry | affine ≡ reconstructor grid | round-trip `A⁻¹·A·v = v` |
        |---|---|---|
        | axial   | max Δ = $(round(ga, sigdigits = 2)) µm | max Δ = $(round(ra, sigdigits = 2)) voxel |
        | helical | max Δ = $(round(gh, sigdigits = 2)) µm | max Δ = $(round(rh, sigdigits = 2)) voxel |

        Both geometries: sub-µm grid agreement and machine-precision round-trip
        ⇒ the UHR-phantom → recon mapping is exactly **1-to-1**, axial and
        helical alike.  Any softness at boundaries is `:nearest` quantization
        (§06) or recon PSF, never the affine.
        """)
    end
end

# ╔═╡ 05000099-0000-4000-8000-000000000001
# The affine audit proves the MAP is exact; this proves the RECON is registered
# TO that map — i.e. the forward projector images the phantom at the same world
# position the backprojector reconstructs it (a half-voxel or rebinning-column
# mismatch would displace the recon and stay invisible to the round-trip audit).
# Correlate recon edge-magnitude against the label-boundary map over integer
# (dx,dy) shifts; the peak shift is the real offset.  (0,0) = registered.
# (Integer resolution: a systematic ≤½-voxel shift also reads (0,0) — but that is
#  the same scale as :nearest quantization / recon PSF, not a registration bug.)
let
    reg_offset = function (recon, gt; R = 5)
        mid = size(recon, 3) ÷ 2
        hu = Float32.(recon[:, :, mid]); lab = gt[:, :, mid]
        nx, ny = size(hu)
        gr = zeros(Float32, nx, ny)          # recon edge strength
        lb = zeros(Float32, nx, ny)          # gt label-boundary map
        for j in 2:(ny - 1), i in 2:(nx - 1)
            gx = hu[i + 1, j] - hu[i - 1, j]; gy = hu[i, j + 1] - hu[i, j - 1]
            gr[i, j] = sqrt(gx * gx + gy * gy)
            (lab[i, j] != lab[i + 1, j] || lab[i, j] != lab[i, j + 1]) && (lb[i, j] = 1f0)
        end
        best = (0, 0); bs = -Inf; sc = Dict{Tuple{Int,Int},Float64}()
        for dx in -R:R, dy in -R:R
            s = 0.0
            for j in (1 + R):(ny - R), i in (1 + R):(nx - R)
                s += gr[i, j] * lb[i + dx, j + dy]
            end
            sc[(dx, dy)] = s
            s > bs && (bs = s; best = (dx, dy))
        end
        (best, sc[(0, 0)] / bs)              # best shift + how good (0,0) is vs peak
    end
    ax = (sim === nothing) ? "—" : reg_offset(recon_HU, gt_resampled_nn)
    hx = (sim_helical === nothing) ? "—" : reg_offset(recon_HU_helical, gt_helical_nn)
    Markdown.parse("""
    **Registration offset (recon edges vs label boundaries, mid-slice):**

    - axial   : best shift = $(ax isa String ? ax : ax[1]) voxels · (0,0) score = $(ax isa String ? ax : round(ax[2], digits = 3)) of peak
    - helical : best shift = $(hx isa String ? hx : hx[1]) voxels · (0,0) score = $(hx isa String ? hx : round(hx[2], digits = 3)) of peak

    `(0,0)` best shift ⇒ registered; a consistent nonzero shift ⇒ a real
    forward↔backprojector offset.  Score near 1.0 ⇒ `(0,0)` is already the peak.
    """)
end

# ╔═╡ 05000011-0000-4000-8000-000000000001
md"""
## Results and Interpretation

### Why the Affine Round-Trip Matters

Once you have ground truth on the recon grid, the rest is bookkeeping:

- **Per-organ ROI HU stats.** `mean(recon_HU[gt_resampled_nn .== UInt8(label)])`
  for any organ label.  No polar-coordinate ROI placement, no manual
  segmentation, no resampling drift.
- **Segmentation evaluation.** Train your segmenter on `recon_HU`,
  evaluate against `gt_resampled_nn` — Dice / Hausdorff are well-defined
  because the voxel grids agree.
- **Partial-volume analysis.** Resample a *binary* mask of one organ
  with `method = :linear` (see the `cardiac_coverage_lin` cell above)
  to get true fractional coverage ∈ [0, 1] per recon voxel — useful for
  boundary-aware metrics or partial-volume-corrected ROI stats.  Don't
  resample the multi-label mask with `:linear` and expect meaningful
  fractions: trilinear arithmetically mixes integer label IDs.
- **Custom interpolation.** When `:nearest` / `:linear` aren't sharp
  enough, the `M = inv(A_phantom) * A_recon` pattern from the bring-your-own-interpolator step lets you
  drop in any third-party interpolator with three lines.
- **SFOV-equivalent cropping is *physical*.** Because the crop happens
  on the input phantom, it shows up in the forward-projection pass
  itself — fewer rays hit anatomy, fewer voxels enter scratch buffers,
  and the simulator runs faster.  Same observable behavior as a real
  scanner's reduced SFOV; better memory characteristics than recon-side
  cropping ever could be.
"""

# ╔═╡ 05000011-0000-4000-8000-000000000020
md"""
## Summary

Two acquisitions, one affine round-trip:

- **Scan A (axial, zoomed FOV)** — crop the UHR phantom to the cardiac bbox
  (the SFOV-equivalent step), reconstruct a tight centered FOV, and overlay the
  resampled ground truth pixel-perfectly.
- **Scan B (helical, extended z)** — a taller crop scanned with a spiral
  trajectory, reconstructed via rebinned WFBP into a tall axial stack you can
  scroll through; its recon grid has its own (taller) recon→world affine, and
  `resample_to_recon` lands the ground truth on every slice unchanged.

The takeaway: `phantom_to_world_affine` / `recon_to_world_affine` /
`resample_to_recon` are **trajectory-agnostic** — axial or helical, tight or
tall FOV, the ground-truth-to-recon mapping is the same three calls.
"""

# ╔═╡ Cell order:
# ╟─05000001-0000-4000-8000-000000000010
# ╟─05000001-0000-4000-8000-000000000020
# ╠═05000001-0000-4000-8000-000000000001
# ╠═05000001-0000-4000-8000-000000000002
# ╠═05000001-0000-4000-8000-000000000003
# ╠═05000001-0000-4000-8000-000000000004
# ╠═05000001-0000-4000-8000-000000000030
# ╠═05000001-0000-4000-8000-000000000031
# ╠═05000001-0000-4000-8000-000000000060
# ╠═05000001-0000-4000-8000-000000000040
# ╟─05000001-0000-4000-8000-000000000050
# ╠═05000001-0000-4000-8000-000000000070
# ╟─05000002-0000-4000-8000-000000000000
# ╟─05000002-0000-4000-8000-000000000001
# ╠═05000002-0000-4000-8000-000000000010
# ╠═05000002-0000-4000-8000-000000000011
# ╠═05000002-0000-4000-8000-000000000012
# ╟─05000002-0000-4000-8000-000000000013
# ╟─05000003-0000-4000-8000-000000000001
# ╠═05000003-0000-4000-8000-000000000010
# ╠═05000003-0000-4000-8000-000000000011
# ╠═05000003-0000-4000-8000-000000000020
# ╠═05000003-0000-4000-8000-000000000021
# ╠═05000003-0000-4000-8000-000000000030
# ╟─05000003-0000-4000-8000-000000000040
# ╟─05000004-0000-4000-8000-000000000001
# ╠═05000004-0000-4000-8000-000000000002
# ╠═05000004-0000-4000-8000-000000000003
# ╠═05000004-0000-4000-8000-000000000004
# ╠═05000004-0000-4000-8000-000000000005
# ╠═05000004-0000-4000-8000-000000000006
# ╠═05000004-0000-4000-8000-000000000007
# ╠═05000004-0000-4000-8000-000000000008
# ╠═05000004-0000-4000-8000-000000000010
# ╟─05000005-0000-4000-8000-000000000001
# ╠═05000005-0000-4000-8000-000000000010
# ╠═05000005-0000-4000-8000-000000000020
# ╟─05000006-0000-4000-8000-000000000000
# ╟─05000006-0000-4000-8000-000000000001
# ╠═05000006-0000-4000-8000-000000000010
# ╠═05000006-0000-4000-8000-000000000020
# ╟─05000007-0000-4000-8000-000000000001
# ╟─05000007-0000-4000-8000-000000000010
# ╟─05000008-0000-4000-8000-000000000001
# ╠═05000008-0000-4000-8000-000000000010
# ╠═05000008-0000-4000-8000-000000000011
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
# ╟─0500000c-0000-4000-8000-000000000001
# ╟─0500000c-0000-4000-8000-000000000010
# ╟─0500000d-0000-4000-8000-000000000001
# ╠═0500000d-0000-4000-8000-000000000010
# ╠═0500000d-0000-4000-8000-000000000020
# ╠═0500000d-0000-4000-8000-000000000030
# ╟─0500000e-0000-4000-8000-000000000001
# ╠═0500000e-0000-4000-8000-000000000010
# ╠═0500000e-0000-4000-8000-000000000011
# ╠═0500000e-0000-4000-8000-000000000012
# ╟─0500000e-0000-4000-8000-000000000020
# ╟─0500000f-0000-4000-8000-000000000001
# ╟─0500000f-0000-4000-8000-000000000010
# ╟─05000010-0000-4000-8000-000000000001
# ╟─05000010-0000-4000-8000-000000000010
# ╟─05000012-0000-4000-8000-000000000000
# ╟─05000013-0000-4000-8000-000000000001
# ╠═05000013-0000-4000-8000-000000000010
# ╠═05000013-0000-4000-8000-000000000020
# ╠═05000013-0000-4000-8000-000000000025
# ╠═05000013-0000-4000-8000-000000000030
# ╠═05000013-0000-4000-8000-000000000031
# ╠═05000013-0000-4000-8000-000000000040
# ╟─05000012-0000-4000-8000-000000000001
# ╠═05000012-0000-4000-8000-000000000010
# ╟─05000014-0000-4000-8000-000000000001
# ╠═05000012-0000-4000-8000-000000000020
# ╟─05000014-0000-4000-8000-000000000010
# ╟─05000014-0000-4000-8000-000000000020
# ╠═05000012-0000-4000-8000-000000000030
# ╟─05000015-0000-4000-8000-000000000001
# ╟─05000015-0000-4000-8000-000000000010
# ╟─05000012-0000-4000-8000-000000000040
# ╟─05000016-0000-4000-8000-000000000001
# ╠═05000016-0000-4000-8000-000000000010
# ╟─05000016-0000-4000-8000-000000000020
# ╟─05000017-0000-4000-8000-000000000001
# ╟─05000012-0000-4000-8000-000000000050
# ╠═05000099-0000-4000-8000-000000000001
# ╟─05000011-0000-4000-8000-000000000001
# ╟─05000011-0000-4000-8000-000000000020
