### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

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
# 05 · XCAT UHR → CT Scan: Phantom Grids, Cropping, and the Affine Round-Trip

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
## Setup

Same project + GPU detection idiom as notebooks 02 / 04.
"""

# ╔═╡ 05000001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 05000001-0000-4000-8000-000000000031
import CairoMakie as CM

# ╔═╡ 05000001-0000-4000-8000-000000000040
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

# ╔═╡ 05000001-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 05000002-0000-4000-8000-000000000001
md"""
## 1. Locate the XCAT data

Same env-var pattern as notebook 02 — the bin lives outside the repo
under `BASISSIM_XCAT_DIR` (default: `docs/notebooks/data/xcat/`).  All
heavy compute is gated on `HAS_XCAT` so this notebook still renders
cleanly when the bin isn't available.
"""

# ╔═╡ 05000002-0000-4000-8000-000000000010
const XCAT_DIR = get(
    ENV, "BASISSIM_XCAT_DIR",
    joinpath(@__DIR__, "data", "xcat")
)

# ╔═╡ 05000002-0000-4000-8000-000000000011
const PHANTOM_PATH = joinpath(
    XCAT_DIR,
    "vmale_50_1600x1400x500_8bit_little_endian_act_1.bin"
)

# ╔═╡ 05000002-0000-4000-8000-000000000012
const HAS_XCAT = isfile(PHANTOM_PATH)

# ╔═╡ 05000002-0000-4000-8000-000000000013
HAS_XCAT ? md"""
    **XCAT located:** `$(PHANTOM_PATH)` ($(round(filesize(PHANTOM_PATH) / 1024^2; digits=1)) MB)
    """ : md"""
    !!! warning "XCAT bin not found"
        Looked at $(PHANTOM_PATH) and didn't find it.  All compute cells
        below short-circuit to `nothing` and the comparison panels show
        this notice.  Set `BASISSIM_XCAT_DIR` to your local install or
        drop the file into the default path above and re-run.
    """

# ╔═╡ 05000003-0000-4000-8000-000000000001
md"""
## 2. Load the UHR XCAT (DOWNSAMPLE_FACTOR = 2)

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
## 3. Custom materials from XCAT spreadsheet

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
)

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
## 4. Cardiac bbox by label-name match

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

# ╔═╡ 05000006-0000-4000-8000-000000000001
md"""
## 5. Crop the phantom — the SFOV-equivalent step

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
### 5b. Visualize the crop

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

        fig = CM.Figure(size = (1200, 600))
        title_kwargs = (titlesize = 28, subtitlesize = 20)

        ax_l = CM.Axis(
            fig[1, 1];
            title = "Full UHR phantom · z=$(z_full)",
            subtitle = "$(size(phantom_full_uhr, 1))×$(size(phantom_full_uhr, 2)) @ $(round.(VOXEL_SIZE_CM .* 10, digits = 2)) mm",
            aspect = CM.DataAspect(),
            yreversed = true,
            title_kwargs...,
        )
        CM.heatmap!(ax_l, Float32.(phantom_full_uhr[:, :, z_full]); colormap = :tab20)
        # Bbox rectangle (note: x-axis is dim 1, y-axis is dim 2)
        CM.poly!(
            ax_l,
            CM.Point2f[(b.i_lo, b.j_lo), (b.i_hi, b.j_lo), (b.i_hi, b.j_hi), (b.i_lo, b.j_hi)];
            color = :transparent, strokecolor = :red, strokewidth = 3,
        )
        CM.hidedecorations!(ax_l)

        ax_r = CM.Axis(
            fig[1, 2];
            title = "Cropped cardiac block · z=$(z_crop)",
            subtitle = "$(size(phantom_cropped, 1))×$(size(phantom_cropped, 2)) (same voxel size — only the extent changed)",
            aspect = CM.DataAspect(),
            yreversed = true,
            title_kwargs...,
        )
        CM.heatmap!(ax_r, Float32.(phantom_cropped[:, :, z_crop]); colormap = :tab20)
        CM.hidedecorations!(ax_r)

        CM.save(
            joinpath(@__DIR__, "..", "assets", "xcat_grid_crop.png"),
            fig; px_per_unit = 2,
        )
        fig
    end
end

# ╔═╡ 05000008-0000-4000-8000-000000000001
md"""
## 6. Build the `Phantom` from the cropped mask

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
### 6b. `phantom_to_world_affine`

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
## 7. `Scanner` — GE Apex Elite

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

# ╔═╡ 0500000a-0000-4000-8000-000000000020
md"""
## 8. `CTProtocol` — clinical cardiac CTA

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
sim_opts = BS.SimOptions(fidelity = :eict, seed = 1234);

# ╔═╡ 0500000b-0000-4000-8000-000000000001
md"""
## 9. `ReconOptions` — tight cardiac FOV

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
### 9b. `recon_to_world_affine`

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
## 10. Side-by-side grid comparison

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
## 11. Forward project + reconstruct

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

        HU = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_water_120))
        BS.add_system_noise_floor!(HU, 28.0; seed = 1234)
        HU
end;

# ╔═╡ 0500000e-0000-4000-8000-000000000001
md"""
## 12. Resample ground truth onto the recon grid

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
## 13. Bring-your-own-interpolator pattern

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
## 14. The verification mosaic

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

        fig = CM.Figure(size = (1400, 1320))
        hu_kwargs = (colormap = :grays, colorrange = (-300, 700))
        title_kwargs = (titlesize = 28, subtitlesize = 20)

        # Top-left: raw HU recon.
        ax_tl = CM.Axis(
            fig[1, 1];
            title = "HU recon",
            subtitle = "z=$(z) of $(size(recon_HU, 3)) · centered FOV $(recon_opts.fov_cm) cm",
            aspect = CM.DataAspect(), yreversed = true,
            title_kwargs...,
        )
        CM.heatmap!(ax_tl, hu_slice; hu_kwargs...)
        CM.hidedecorations!(ax_tl)

        # Top-right: ALL structures — full multi-label resample, no masking.
        ax_tr = CM.Axis(
            fig[1, 2];
            title = "All structures (`:nearest`, no overlay)",
            subtitle = "$(length(unique(gt_resampled_nn))) labels resampled onto recon grid",
            aspect = CM.DataAspect(), yreversed = true,
            title_kwargs...,
        )
        CM.heatmap!(
            ax_tr, Float32.(gt_resampled_nn[:, :, z]);
            colormap = :tab20,
        )
        CM.hidedecorations!(ax_tr)

        # Bottom-left: HU + cardiac labels overlay.
        ax_bl = CM.Axis(
            fig[2, 1];
            title = "HU + cardiac labels",
            subtitle = "α=0.6 over HU — alignment check",
            aspect = CM.DataAspect(), yreversed = true,
            title_kwargs...,
        )
        CM.heatmap!(ax_bl, hu_slice; hu_kwargs...)
        CM.heatmap!(
            ax_bl, nn_overlay;
            colormap = :tab20, alpha = 0.6, nan_color = (:white, 0.0),
        )
        CM.hidedecorations!(ax_bl)

        # Bottom-right: HU + fractional cardiac coverage overlay.
        ax_br = CM.Axis(
            fig[2, 2];
            title = "HU + cardiac coverage (`:linear`, binary mask)",
            subtitle = "fractional ∈ [0.05, 1] · α=0.7 over HU",
            aspect = CM.DataAspect(), yreversed = true,
            title_kwargs...,
        )
        CM.heatmap!(ax_br, hu_slice; hu_kwargs...)
        hm_br = CM.heatmap!(
            ax_br, lin_overlay;
            colormap = :viridis, colorrange = (0, 1),
            alpha = 0.7, nan_color = (:white, 0.0),
        )
        CM.hidedecorations!(ax_br)
        CM.Colorbar(fig[2, 3], hm_br; label = "cardiac fraction", width = 14, labelsize = 18)

        CM.save(
            joinpath(@__DIR__, "..", "assets", "xcat_grid_overlay.png"),
            fig; px_per_unit = 2,
        )
        fig
    end
end

# ╔═╡ 05000011-0000-4000-8000-000000000001
md"""
## 15. Why this matters

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
  enough, the `M = inv(A_phantom) * A_recon` pattern from §13 lets you
  drop in any third-party interpolator with three lines.
- **SFOV-equivalent cropping is *physical*.** Because the crop happens
  on the input phantom, it shows up in the forward-projection pass
  itself — fewer rays hit anatomy, fewer voxels enter scratch buffers,
  and the simulator runs faster.  Same observable behavior as a real
  scanner's reduced SFOV; better memory characteristics than recon-side
  cropping ever could be.
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
# ╠═05000001-0000-4000-8000-000000000040
# ╟─05000001-0000-4000-8000-000000000050
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
# ╟─05000011-0000-4000-8000-000000000001
