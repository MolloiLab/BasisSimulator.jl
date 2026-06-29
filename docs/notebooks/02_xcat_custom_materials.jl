### A Pluto.jl notebook ###
# v0.1.0

using Markdown
using InteractiveUtils

# ╔═╡ 02000003-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 02000003-0000-4000-8000-000000000004
using Statistics: std, mean

# ╔═╡ 02000003-0000-4000-8000-000000000005
using Markdown: @md_str

# ╔═╡ 02000003-0000-4000-8000-000000000006
using Unitful: @u_str

# ╔═╡ 02000001-0000-4000-8000-000000000001
md"""
# 02 · XCAT Phantom + Custom Materials

**Scan a digital anatomy with material attenuation you define from scratch.**

Notebook 01 walked the five-struct API on the Gammex 472 calibration phantom.
This notebook swaps the phantom: instead of a labeled cylinder of inserts,
we load a high-resolution **NCAT/XCAT** voxel phantom — a digital adult
torso — and assign each region (lung, blood, muscle, bone, …) its own
[`XrayAttenuation.Material`](https://github.com/MolloiLab/XrayAttenuation.jl)
with explicit elemental composition and density.

We finish with a side-by-side **FBP vs Hybrid IR** reconstruction comparison
on the same scan.

| | |
|---|---|
| **Phantom** | XCAT v_male_50 (1600 × 1400 × 500, 0.2 mm isotropic, downsampled 5× for speed) |
| **Materials** | Mix of `XA.Materials.ncat_*` prebuilt + a custom iodinated-blood material constructed inline |
| **Scanner** | GE Revolution Apex Elite (same as notebook 01) |
| **Recon** | FBP/FDK + Hybrid IR (`:asir`-style PWLS refinement) |
"""

# ╔═╡ 02000002-0000-4000-8000-000000000001
md"""
## Setup

Same shape as notebook 01 — activate `docs/Project.toml`, four discrete
import cells, then probe for a GPU backend.  We add **Unitful** here for
the `u"eV"` and `u"g/cm^3"` units used by the `XA.Material` constructor.
"""

# ╔═╡ 02000003-0000-4000-8000-000000000002
import BasisSimulator as BS

# ╔═╡ 02000003-0000-4000-8000-000000000003
import CairoMakie as CM

# ╔═╡ 05000003-0000-4000-8000-000000000003
import Unitful: ustrip, uconvert

# ╔═╡ 02000004-0000-4000-8000-000000000001
md"""
#### Backend-agnostic device transfer: `to_gpu()`

Same probe-then-load pattern as notebook 01 — checks Metal → CUDA →
AMDGPU via `Base.locate_package`, falls back to CPU `identity`.
"""

# ╔═╡ 02000005-0000-4000-8000-000000000001
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
                    detected = (
                        name = string(pkg),
                        to_gpu = getfield(m, ctor),
                    )
                    break
                end
            catch
            end
        end

        detected
    end

    to_gpu(x) = GPU_BACKEND.to_gpu(x)
end

# ╔═╡ 02000007-0000-4000-8000-000000000001
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 03000001-0000-4000-8000-000000000001
md"""
## 1. Locate the XCAT data

The XCAT phantom is **not bundled with the repo** — voxelized anatomical
phantoms are several hundred MB and have their own licensing terms.  Set
`BASISSIM_XCAT_DIR` to wherever you keep your local install, or drop the
bin file into `docs/notebooks/data/xcat/` (the default path below).

The notebook short-circuits all heavy compute when the file isn't present,
so the static HTML still renders cleanly on a fresh checkout.

!!! info "Where to get a voxelized phantom"
    A reasonable starting point is the open
    [`xcist/phantoms-voxelized`](https://github.com/xcist/phantoms-voxelized)
    repository — XCIST ships several voxelized phantoms (Duke XCAT-derived
    plus simpler shapes) with material lookup tables.  This notebook is
    written against a slightly larger Duke XCAT instance
    (`vmale_50`, 1600 × 1400 × 500 @ 0.2 mm), but the loader below is just
    a `read!` over a column-major `UInt8` array — point it at any voxelized
    phantom with a similar layout and adjust the `cols` / `rows` / `slices`
    kwargs.  Duke's full XCAT program is at
    [cvit.duke.edu/resource/xcat-phantom-program](https://cvit.duke.edu/resource/xcat-phantom-program).

Expected files (default layout, override via env var):

```
\$BASISSIM_XCAT_DIR/
└── vmale_50_1600x1400x500_8bit_little_endian_act_1.bin
```

!!! info "Why isn't the binary tracked in git?"
    Voxelized phantoms come with their own licenses (Duke CVIT terms etc.),
    so each user obtains their own copy.  Once you've rendered the
    notebook locally, the resulting HTML (`docs/notebooks-static/02_*.html`)
    DOES ship with the repo and CI deploys it verbatim — no phantom redistribution.
"""

# ╔═╡ 03000002-0000-4000-8000-000000000001
const XCAT_DIR = get(
    ENV, "BASISSIM_XCAT_DIR",
    joinpath(@__DIR__, "data", "xcat")
)

# ╔═╡ 03000002-0000-4000-8000-000000000002
const PHANTOM_PATH = joinpath(
    XCAT_DIR,
    "vmale_50_1600x1400x500_8bit_little_endian_act_1.bin"
)

# ╔═╡ 03000002-0000-4000-8000-000000000003
const HAS_XCAT = isfile(PHANTOM_PATH)

# ╔═╡ 03000002-0000-4000-8000-000000000004
HAS_XCAT ? md"""
    **XCAT located:** `$(PHANTOM_PATH)` ($(round(filesize(PHANTOM_PATH) / 1024^2; digits=1)) MB)
    """ : md"""
    !!! warning "XCAT bin not found"
        Looked at `$(PHANTOM_PATH)` and didn't find it.  Subsequent compute
        cells will short-circuit to `nothing` and the comparison panels will
        show this notice.  Set `BASISSIM_XCAT_DIR` to your local install or
        drop the file into the default path above and re-run.
    """

# ╔═╡ 04000001-0000-4000-8000-000000000001
md"""
## 2. Load + downsample the labeled mask

XCAT bin layout: column-major (Fortran order) `UInt8` array of shape
`(1600, 1400, 500)` — column → row → slice.  We reverse along axes (2, 3)
to match the BasisSimulator convention (X = lateral right-positive, Y =
anterior-positive, Z = inferior→superior).

A 1.6 GB voxel grid is overkill for a docs example, so we **downsample 5×**
in every axis (320 × 280 × 100, 1 mm voxels) using nearest-neighbor
sampling — preserves label integrity (no interpolated mid-organ voxels).
"""

# ╔═╡ 04000002-0000-4000-8000-000000000001
function load_xcat_bin(
        filepath::AbstractString;
        cols::Int = 1600, rows::Int = 1400, slices::Int = 500
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

# ╔═╡ 04000002-0000-4000-8000-000000000002
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

# ╔═╡ 04000003-0000-4000-8000-000000000001
const DOWNSAMPLE_FACTOR = 5

# ╔═╡ 04000004-0000-4000-8000-000000000001
phantom_labeled = HAS_XCAT ?
    downsample_labeled(load_xcat_bin(PHANTOM_PATH), DOWNSAMPLE_FACTOR) :
    nothing;

# ╔═╡ 04000005-0000-4000-8000-000000000001
let
    if phantom_labeled === nothing
        md"""
        !!! warning "Skipped — see 1 above"
        """
    else
        mid = size(phantom_labeled, 3) ÷ 2
        n_lbl = length(unique(phantom_labeled))
        slice = phantom_labeled[:, :, mid]

        fig = CM.Figure(size = (900, 600))
        ax = CM.Axis(
            fig[1, 1];
            title = "XCAT v_male_50",
            subtitle = "Slice $(mid) / $(size(phantom_labeled, 3))" *
                " · $(size(phantom_labeled, 1))×$(size(phantom_labeled, 2))" *
                " · $(n_lbl) unique labels",
            aspect = CM.DataAspect(),
            titlesize = 28, subtitlesize = 20,
        )
        hm = CM.heatmap!(ax, Float32.(slice); colormap = :tab20)
        CM.hidedecorations!(ax)
        CM.Colorbar(fig[1, 2], hm; label = "label", width = 14, labelsize = 18)

        CM.save(
            joinpath(@__DIR__, "..", "assets", "xcat_phantom.png"),
            fig; px_per_unit = 2
        )
        fig
    end
end

# ╔═╡ 05000001-0000-4000-8000-000000000001
md"""
## 3. Custom materials via `XrayAttenuation`

Every voxel in the labeled mask needs a corresponding
[`XA.Material`](https://github.com/MolloiLab/XrayAttenuation.jl) so the
simulator knows its energy-dependent linear attenuation μ(E).  Two ways
to get one:

1. **Prebuilt** — `XA.Materials.ncat_blood`, `XA.Materials.ncat_muscle`,
   etc.  XrayAttenuation ships with the canonical NCAT/XCAT tissue compositions.
2. **Constructed inline** — `XA.Material(name, ZA_ratio, I, density, composition)`.
   Useful when you want to model *your specific* contrast bolus, calibration
   solution, or non-standard alloy.

The constructor signature:

```julia
XA.Material(
    name::String,                  # human label
    ZA_ratio::Real,                # mean ⟨Z/A⟩ across the composition
    I::Quantity,                   # mean excitation energy, Unitful (e.g. 70u"eV")
    density::Quantity,             # bulk density, Unitful (e.g. 1.06u"g/cm^3")
    composition::Dict{Int,Float64} # element atomic number → mass fraction
)
```

Composition mass fractions must sum to 1. `ZA_ratio` and `I` follow Bethe's
stopping-power convention; for educational use you can compute them from the
composition (see notebook 05's `compute_ZA_ratio` helper) or copy from a
materials reference table.
"""

# ╔═╡ 05000002-0000-4000-8000-000000000001
md"""
### 3a. Pull a prebuilt NCAT material

`XrayAttenuation` ships with the full NCAT/XCAT tissue catalog.  Inspect one:
"""

# ╔═╡ 05000002-0000-4000-8000-000000000002
let m = BS.XA.Materials.ncat_blood
    md"""
    **`XA.Materials.ncat_blood`**

    | Field         | Value                                    |
    |---------------|------------------------------------------|
    | `name`        | $(m.name)                                |
    | `ZA_ratio`    | $(round(m.ZA_ratio; digits=4))           |
    | `I`           | $(m.I)                                   |
    | `density`     | $(m.density)                             |
    | `composition` | $(length(m.composition)) elements (Z = $(sort(collect(keys(m.composition))))) |
    """
end

# ╔═╡ 05000003-0000-4000-8000-000000000001
md"""
### 3b. Construct a custom contrast material

Below: a 5 mg/mL iodinated blood mixture, like a coronary CT-angiography
bolus.  We start from `ncat_blood`'s composition and add elemental iodine.

!!! info "Composition arithmetic"
    Adding `iodine_mg_per_mL` mg of iodine per mL of blood at density
    1.06 g/cm³ gives an iodine **mass fraction** of
    `iodine_mg_per_mL / (1000 · density_g_per_cm3)`.  We rescale the
    other elements proportionally so the new composition still sums to 1.
"""

# ╔═╡ 05000003-0000-4000-8000-000000000002
function build_iodine_blood(iodine_mg_per_mL::Real)
    base = BS.XA.Materials.ncat_blood
    density = base.density                                   # ~1.06 g/cm³
    ρ_g_cm3 = ustrip(uconvert(u"g/cm^3", density))
    f_I = iodine_mg_per_mL / (1000.0 * ρ_g_cm3)          # iodine mass fraction
    scale = 1.0 - f_I

    comp = Dict{Int, Float64}()
    for (Z, frac) in base.composition
        comp[Z] = frac * scale
    end
    comp[53] = get(comp, 53, 0.0) + f_I                       # add iodine

    return BS.XA.Material(
        "Iodine-blood ($(iodine_mg_per_mL) mg/mL)",
        base.ZA_ratio,                                        # ZA shifts ~negligibly at clinical dose
        base.I,                                               # likewise for excitation energy
        density,
        comp,
    )
end

# ╔═╡ 05000003-0000-4000-8000-000000000004
iodine_blood = build_iodine_blood(5.0)

# ╔═╡ 05000004-0000-4000-8000-000000000001
md"""
### 3c. Load the canonical mapping from XCAT's xlsx

XCAT v_male_50 ships with material spreadsheets — one row per organ, with
the elemental mass-fraction columns, density, and the integer label that
appears in the mask.  Three variants in `Material_Spreadsheets/`:
`heart_high_contrast`, `heart_low_contrast`, `heart_non_contrast` —
differing only in the iodine concentration in the blood pool labels.

Reading the xlsx (33 organs × 13 elements) into a `Dict{Int, XA.Material}`
is straightforward: parse each row, build a composition dict, derive
`ZA_ratio` and the mean excitation energy `I` from the composition, then
construct the `XA.Material`.

!!! info "Why ZA and I are computed, not stored"
    The xlsx ships only the elemental mass fractions + density.  ZA_ratio
    and I (mean excitation energy) are derived quantities the simulator
    needs but the spreadsheet doesn't carry — we compute them from the
    composition using the textbook Bethe formulas.
"""

# ╔═╡ 05000004-0000-4000-8000-000000000002
import XLSX

# ╔═╡ 05000004-0000-4000-8000-000000000003
const MATERIAL_XLSX_PATH = joinpath(
    XCAT_DIR, "Material_Spreadsheets",
    "vmale_50_materials_heart_high_contrast.xlsx"
)

# ╔═╡ 20ce81a0-4eed-4b58-9123-efd38b978e54
# Atomic masses + I-values for the elements present in XCAT compositions.
const _ATOMIC_MASSES = Dict(
    1 => 1.008, 6 => 12.011, 7 => 14.007, 8 => 15.999, 11 => 22.99, 12 => 24.305,
    15 => 30.974, 16 => 32.06, 17 => 35.45, 19 => 39.098, 20 => 40.078, 26 => 55.845, 53 => 126.904,
)

# ╔═╡ 05000004-0000-4000-8000-000000000004
const _I_VALUES_EV = Dict(
    1 => 19.2, 6 => 81.0, 7 => 82.0, 8 => 95.0, 11 => 149.0, 12 => 156.0,
    15 => 173.0, 16 => 180.0, 17 => 174.0, 19 => 190.0, 20 => 191.0, 26 => 286.0, 53 => 491.0,
)

# ╔═╡ 05000004-0000-4000-8000-000000000005
"""Bethe ⟨Z/A⟩ from a `composition::Dict{Z=>mass_fraction}`."""
function compute_ZA_ratio(comp::Dict{Int, Float64})
    Z_sum = sum(w * Z / get(_ATOMIC_MASSES, Z, Float64(Z) * 2) for (Z, w) in comp)
    A_sum = sum(values(comp))
    return Z_sum / A_sum
end

# ╔═╡ 05000004-0000-4000-8000-000000000006
"""Bethe mean excitation energy from a composition (returns Unitful eV)."""
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

# ╔═╡ 05000004-0000-4000-8000-000000000007
"""Load every organ from the XCAT material spreadsheet → `Dict{organ_id, XA.Material}`."""
function load_materials_from_xlsx(xlsx_path::AbstractString)
    sheet = XLSX.readxlsx(xlsx_path)["Sheet1"]
    data = sheet["A2:P34"]                                   # 33 organ rows
    out = Dict{Int, BS.XA.Material}()

    # Element atomic numbers in xlsx columns 2..14 (header row 1 confirms order)
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

# ╔═╡ 05000004-0000-4000-8000-000000000008
md"""
### 3d. Assemble the final dict

Load the xlsx, then ensure every label that actually appears in the mask
has a fallback (water) — `simulate!` will error if any label is missing.

!!! info "Override pattern"
    To swap a single label for your own custom material, just reassign in
    the returned dict:
    `materials[19] = iodine_blood` — replaces the LV blood pool with the
    5 mg/mL `iodine_blood` we built in §3b.  Comment that line out and
    you'll get the xlsx's default contrast level instead.
"""

# ╔═╡ 05000004-0000-4000-8000-000000000009
materials = if phantom_labeled === nothing
    nothing
else
    base = load_materials_from_xlsx(MATERIAL_XLSX_PATH)
    for l in unique(phantom_labeled)
        haskey(base, Int(l)) || (base[Int(l)] = BS.XA.Materials.water)
    end
    base
end;

# ╔═╡ 06000001-0000-4000-8000-000000000001
md"""
## 4. Build the `Phantom`

XCAT v_male_50 voxels are 0.2 mm isotropic at full resolution; after
`DOWNSAMPLE_FACTOR = 5` they become 1 mm isotropic.  FOV at 320 × 280 × 100
voxels = 32 × 28 × 10 cm — fits inside the 35 cm scanner FOV with some
margin.
"""

# ╔═╡ 06000002-0000-4000-8000-000000000001
const VOXEL_SIZE_CM = (
    0.02 * DOWNSAMPLE_FACTOR,                # 1 mm
    0.02 * DOWNSAMPLE_FACTOR,
    0.02 * DOWNSAMPLE_FACTOR,
)

# ╔═╡ 06000003-0000-4000-8000-000000000001
phantom = (phantom_labeled === nothing || materials === nothing) ?
    nothing :
    BS.Phantom(to_gpu(phantom_labeled), materials, VOXEL_SIZE_CM);

# ╔═╡ 07000001-0000-4000-8000-000000000001
md"""
## 5. Scanner, protocol, sim & recon options

Same GE Revolution Apex Elite hardware as notebook 01.  The protocol is a
clinical body-CTA acquisition: 120 kVp / 250 mA, 5 mm collimation, 500
views.  Recon: 512 × 512, 35 cm FOV, FBP `:standard` filter.
"""

# ╔═╡ 07000002-0000-4000-8000-000000000001
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

# ╔═╡ 07000003-0000-4000-8000-000000000001
protocol = BS.CTProtocol(
    kVp = 120,
    mA = 250.0,
    views = 500,
    rotation_time = 1.0,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
)

# ╔═╡ 07000004-0000-4000-8000-000000000001
# `projector` picks the forward ray tracer: :dd (default) is distance-driven and
# anti-aliased (robust in severe beam-hardened regions); :siddon is ~3.5-5.5× faster
# on GPU but ALIASES in those regions — use it only when speed outranks accuracy.
# The BHC and Hybrid-IR cells below all read `sim_opts.projector`, so changing it
# HERE updates the whole pipeline consistently — the recon must invert the operator
# that generated the data, or the IR system matrix won't match and convergence suffers.
sim_opts = BS.SimOptions(fidelity = :eict, seed = 1234, projector = :siddon)

# ╔═╡ 07000005-0000-4000-8000-000000000001
recon_opts = BS.ReconOptions(
    matrix_size = (512, 512, 8),
    fov_cm = 35.0,
    z_cm = 0.5,
)

# ╔═╡ 08000001-0000-4000-8000-000000000001
md"""
## 6. Forward project

Same `let ... end` shape as notebook 01 — workspace, `simulate!`, copy off
GPU, drop refs, force `GC.gc(true)`.  The XCAT phantom is ~3× the volume
of the Gammex 472 demo (320×280×100 vs 512×512×16), so this step takes
longer; on a CPU-only fallback it can be **several minutes**.
"""

# ╔═╡ 08000002-0000-4000-8000-000000000001
sim = phantom === nothing ? nothing : let
        @info "Simulating XCAT body CTA: 120 kVp / 250 mA…"
        ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
        BS.simulate!(ws, phantom, protocol, sim_opts)

        result = (sino = Array(ws.sinogram), geom = ws.geom)

        ws = nothing
        GC.gc(true)

        result
end;

# ╔═╡ 09000001-0000-4000-8000-000000000001
md"""
## 7. Postprocessing (The Full Correction Pipeline)

A raw FDK or HIR recon doesn't ship as a clinical image.  Real CT vendors
apply a stack of corrections after the simulator's forward model:

| Stage | Function                              | What it does                                                                             |
|-------|---------------------------------------|------------------------------------------------------------------------------------------|
| 1.    | `calibrate_bhc_two_material`          | Fits a (water + bone) polynomial from the source spectrum once, returns a BHC model      |
| 2.    | `apply_bhc_two_material`              | Sinogram-domain beam-hardening correction — runs *before* FDK, removes the bulk poly bias |
| 3.    | `reconstruct!` (FDK or Hybrid IR)     | Filtered back-projection / iterative reconstruction on the corrected sinogram            |
| 4.    | `apply_bhc_image_domain`              | Image-domain BHC refinement — bone-region smoothstep + scaled error subtraction          |
| 5.    | `to_hounsfield` (with BHC μ_water)    | Convert μ → HU using the BHC model's calibrated reference (not the 70 keV NIST value)    |
| 6.    | `add_system_noise_floor!`             | Add dose-independent DAS Gaussian σ ≈ 28 HU                                              |
| 7.    | `apply_radial_cupping_correction!`    | Residual even-polynomial cup correction on water-like voxels (mostly cosmetic after BHC) |

The same pipeline applies to both FBP and HIR — only the recon workspace
inside the let block differs (`create_fdk_recon_workspace` vs
`create_hir_recon_workspace`). See notebook 01 §9 for the same pattern on
the Gammex 472 phantom.
"""

# ╔═╡ 09000003-0000-4000-8000-000000000001
md"""
#### 7a. Calibrate the BHC model

One-time spectrum fit at 120 kVp + 7 mm Al filtration. Returns a
`TwoMaterialBHC` polynomial model + the calibrated `μ_water_ref` used by
both FBP and HIR pipelines below for `to_hounsfield`.
"""

# ╔═╡ 09000004-0000-4000-8000-000000000001
bhc_calibration = sim === nothing ? nothing : let
        prot_for_bhc = BS.CTProtocol(kVp = 120, additional_filters = [("Al", 4.5)])

        # Bowtie-aware high-level API: auto-resolves the bowtie-hardened spectrum,
        # collapses to per-column weights, and dispatches to TwoMaterialBHCPerColumn.
        model = BS.calibrate_bhc_two_material(
            sim_opts, prot_for_bhc;
            scanner = scanner, geom = sim.geom,
            order = 2,
            hu_low = 450.0,
            hu_high = 600.0,
        )

        (
            model = model,
            μ_water = model.μ_water_ref,
            ref_E_keV = model.reference_energy_keV,
        )
end;

# ╔═╡ 09000005-0000-4000-8000-000000000001
md"""
**Calibrated:**
* ref energy = $(bhc_calibration === nothing ? "—" : round(bhc_calibration.ref_E_keV, digits = 1)) keV,
* lac water = $(bhc_calibration === nothing ? "—" : round(bhc_calibration.μ_water, digits = 5)) cm⁻¹.
"""

# ╔═╡ 09000006-0000-4000-8000-000000000001
md"""
#### 7b. Polychromatic `μ_water` from the XCAT body — *informational*

The BHC's `μ_water_ref` above (the monoenergetic μ_water at the
spectrum-mean energy) is what §8 + §9 use as the HU divisor — that's
the right reference *post-BHC*, where the recon reads as approximately
monochromatic at `ref_E_keV`.

For comparison, here's what the **polychromatic-effective** μ_water
would be — the value an *uncorrected* polychromatic FBP would land on
for solid water at this body chord.  Useful as a sanity check; not
plumbed into `to_hounsfield`.

[`BS.compute_polychromatic_μ_water`](@ref) does the spectrum + Beer-
Lambert hardening analytically; [`BS.estimate_phantom_diameter_cm`](@ref)
reads the body's chord length straight off the **XCAT mask**, so the
calibration scales naturally if you swap phantoms (vmale_50 →
vfemale_50, etc.) without hardcoding any cm.
"""

# ╔═╡ 09000006-0000-4000-8000-000000000010
μ_water_poly_uncorrected = phantom === nothing ? nothing : let
        voxel_size_mm = VOXEL_SIZE_CM .* 10.0
        body_diameter_cm = BS.estimate_phantom_diameter_cm(
            phantom_labeled, voxel_size_mm
        )
        BS.compute_polychromatic_μ_water(
            sim_opts, protocol;
            scanner = scanner,
            geom = sim.geom,
            water_path_cm = body_diameter_cm,
        )
end;

# ╔═╡ 09000006-0000-4000-8000-000000000020
md"""
**Analytic poly μ_water** (XCAT body chord, no-BHC reference for comparison):

* `μ_water_poly_uncorrected = ` $(μ_water_poly_uncorrected === nothing ? "—" : "$(round(μ_water_poly_uncorrected, digits = 5)) cm⁻¹")
* `bhc_calibration.μ_water = ` $(bhc_calibration === nothing ? "—" : "$(round(bhc_calibration.μ_water, digits = 5)) cm⁻¹") *(used downstream)*
"""

# ╔═╡ 09000010-0000-4000-8000-000000000001
md"""
## 8. FBP with the full correction pipeline

Same `let ... end` shape as notebook 01 — sino BHC → FDK → image BHC →
HU → noise floor → cupping, with explicit GPU cleanup at the end.
HU conversion uses `bhc_calibration.μ_water` (the BHC's calibrated
`μ_water_ref` at the spectrum-mean energy), since the post-BHC recon
reads as approximately monochromatic at that reference.
"""

# ╔═╡ 09000002-0000-4000-8000-000000000001
hu_fbp = sim === nothing ? nothing : let
        matrix_size = recon_opts.matrix_size

        # 1. Sinogram-domain BHC (returns a CPU array)
        sino_gpu = to_gpu(sim.sino)
        sino_bhc = BS.apply_bhc_two_material(
            sino_gpu, bhc_calibration.model, sim.geom, matrix_size;
            projector = sim_opts.projector,
        )
        sino_gpu = to_gpu(sino_bhc)

        # 2. FDK on the corrected sinogram
        ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, sim.geom, matrix_size)
        recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, sim.geom)

        # 3. Image-domain BHC refinement (in-place on GPU)
        BS.apply_bhc_image_domain(
            recon_μ, sim.geom, matrix_size, bhc_calibration.μ_water;
            hu_low = 50.0,
            hu_high = 150.0,
            scale_factor = 0.2,
            projector = sim_opts.projector,
        )

        # 4. μ → HU using BHC's calibrated μ_water_ref (Float32 for cupping correction)
        hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc_calibration.μ_water))

        # 5. Dose-independent DAS noise floor
        BS.add_system_noise_floor!(hu, 28.0; seed = 1234)

        # 6. Residual radial cupping correction
        BS.apply_radial_cupping_correction!(hu; fov_cm = 35.0)

        # GPU cleanup
        ws_fdk = nothing
        sino_gpu = nothing
        recon_μ = nothing
        GC.gc(true)

        hu
end;

# ╔═╡ 10000001-0000-4000-8000-000000000001
md"""
## 9. Hybrid IR with the full correction pipeline

Identical pipeline to §8 — sino BHC → recon → image BHC → HU → noise
floor → cupping — with `create_hir_recon_workspace(...; strength = 3)`
swapped in for the FDK workspace.

!!! info "What HIR adds over FBP"
    Hybrid IR is the clinical workhorse: vendors call it ASIR, AIDR3D,
    iDose⁴, SAFIRE depending on brand. All share the same architecture —
    FDK init + iterative PWLS refinement with a Huber prior. It costs
    ~5–15× FBP runtime but returns ~30 % lower pixel σ at matched
    resolution.

!!! warning "Projector consistency (IR only)"
    Hybrid IR's data-fidelity term uses a forward projector `A`, so it
    **must use the same projector that generated the sinogram**. We pass
    `projector = sim_opts.projector` to `create_hir_recon_workspace`, so
    flipping `sim_opts.projector` to `:siddon` (for speed) automatically
    keeps the recon consistent. FBP (§8) is immune — it has no forward `A`.
"""

# ╔═╡ 10000002-0000-4000-8000-000000000001
hu_hir = sim === nothing ? nothing : let
        matrix_size = recon_opts.matrix_size

        # 1. Sinogram-domain BHC
        sino_gpu = to_gpu(sim.sino)
        sino_bhc = BS.apply_bhc_two_material(
            sino_gpu, bhc_calibration.model, sim.geom, matrix_size;
            projector = sim_opts.projector,
        )
        sino_gpu = to_gpu(sino_bhc)

        # 2. Hybrid IR (FBP init + PWLS refinement, strength = 3)
        # projector MUST match the forward sim so the IR system matrix A inverts
        # the operator that generated the data (read from sim_opts, single source).
        ws_hir = BS.create_hir_recon_workspace(
            sino_gpu, sim.geom, matrix_size; strength = 3, projector = sim_opts.projector,
        )
        recon_μ = BS.reconstruct!(ws_hir, sino_gpu, sim.geom)

        # FBP's `reconstruct!` masks voxels outside the inscribed scan FOV
        # (`apply_fov_mask!`); HIR doesn't, so the iterative refinement
        # leaves garbage in the corners.  Apply the same mask for parity.
        BS.apply_fov_mask!(recon_μ, sim.geom)

        # 3. Image-domain BHC refinement
        BS.apply_bhc_image_domain(
            recon_μ, sim.geom, matrix_size, bhc_calibration.μ_water;
            hu_low = 50.0,
            hu_high = 150.0,
            scale_factor = 0.2,
            projector = sim_opts.projector,
        )

        # 4. μ → HU using BHC's calibrated μ_water_ref
        hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc_calibration.μ_water))

        # 5. Noise floor (different seed so noise pattern doesn't match FBP)
        BS.add_system_noise_floor!(hu, 28.0; seed = 5678)

        # 6. Cupping correction
        BS.apply_radial_cupping_correction!(hu; fov_cm = 35.0)

        # GPU cleanup
        ws_hir = nothing
        sino_gpu = nothing
        recon_μ = nothing
        GC.gc(true)

        hu
end;

# ╔═╡ 11000001-0000-4000-8000-000000000001
md"""
## 10. Compare FBP vs Hybrid IR

Both reconstructions go through the identical correction pipeline (sino
BHC + image BHC + noise floor + cupping). Soft-tissue window shows the
iodine-enhanced blood pools (XCAT labels 19–22 = `bldplLV/RV/LA/RA`) and
the texture / noise contrast between the two algorithms.
"""

# ╔═╡ 11000002-0000-4000-8000-000000000001
let
    if hu_fbp === nothing || hu_hir === nothing
        md"""
        !!! warning "Skipped — see 1 for XCAT setup"
        """
    else
        fig = CM.Figure(size = (1200, 660))

        mid = size(hu_fbp, 3) ÷ 2
        img_fbp = hu_fbp[:, :, mid]
        img_hir = hu_hir[:, :, mid]
        colorrng = (-300, 550)   # soft-tissue window: W=400, L=40
        title_kwargs = (titlesize = 28, subtitlesize = 20)

        ax_fbp = CM.Axis(
            fig[1, 1];
            title = "FBP (FDK)",
            subtitle = "120 kVp / 250 mA · :standard filter",
            aspect = CM.DataAspect(),
            title_kwargs...,
        )
        ax_hir = CM.Axis(
            fig[1, 2];
            title = "Hybrid IR",
            subtitle = "120 kVp / 250 mA · strength = 3",
            aspect = CM.DataAspect(),
            title_kwargs...,
        )

        CM.heatmap!(ax_fbp, img_fbp; colormap = :grays, colorrange = colorrng)
        hm = CM.heatmap!(ax_hir, img_hir; colormap = :grays, colorrange = colorrng)

        CM.hidedecorations!(ax_fbp)
        CM.hidedecorations!(ax_hir)
        CM.Colorbar(fig[1, 3], hm; label = "HU", width = 14, labelsize = 18)

        CM.save(
            joinpath(@__DIR__, "..", "assets", "xcat_fbp_vs_hir.png"),
            fig; px_per_unit = 2
        )
        fig
    end
end

# ╔═╡ 11000003-0000-4000-8000-000000000001
let
    if hu_fbp === nothing || hu_hir === nothing
        md""
    else
        # Pixel-σ in a homogeneous region near the body envelope (matters: HIR's
        # whole point is reducing this number while keeping edges sharp).
        n = size(hu_fbp, 1)
        roi = ((n ÷ 2 - 30):(n ÷ 2 + 30), (n ÷ 2 - 30):(n ÷ 2 + 30), :)

        σ_fbp = std(hu_fbp[roi...])
        σ_hir = std(hu_hir[roi...])

        md"""
        **Noise (σ in HU, 60×60 ROI near image center):**

        | Algorithm  | σ (HU)                    |
        |------------|---------------------------|
        | FBP        | $(round(σ_fbp, digits=1)) |
        | Hybrid IR  | $(round(σ_hir, digits=1)) |
        | Reduction  | $(round(100 * (1 - σ_hir / σ_fbp); digits=1))% |
        """
    end
end

# ╔═╡ 12000001-0000-4000-8000-000000000001
md"""
## Summary

This notebook layered three new ideas on top of the five-struct API
walked in notebook 01:

- **External voxel phantoms** — load any labeled `.bin` mask (XCAT, ICRP,
  custom), then hand it to `BS.Phantom(...)` along with a materials Dict.
  No internal preprocessing; whatever labeling you give, you get back.
- **Custom materials via `XrayAttenuation`** — `XA.Materials.ncat_*` for
  canonical tissues, plus `XA.Material(name, ZA, I, density, composition)`
  built inline whenever you need a non-standard mixture (contrast bolus,
  calibration solution, alloy).  Composition is just a `Dict{Int,Float64}`
  of atomic-number → mass-fraction.
- **Hybrid IR** — `BS.create_hir_recon_workspace(sino, geom, matrix; strength)`
  swaps in for the FDK workspace and runs PWLS refinement after the FBP
  init.  Same `reconstruct!` call site, lower pixel noise.

Every other piece — the `let ... end` GPU pattern, μ → HU conversion, the
correction pipeline from §9 of notebook 01 — carries over unchanged.
"""

# ╔═╡ Cell order:
# ╟─02000001-0000-4000-8000-000000000001
# ╟─02000002-0000-4000-8000-000000000001
# ╠═02000003-0000-4000-8000-000000000001
# ╠═02000003-0000-4000-8000-000000000002
# ╠═02000003-0000-4000-8000-000000000003
# ╠═02000003-0000-4000-8000-000000000004
# ╠═02000003-0000-4000-8000-000000000005
# ╠═02000003-0000-4000-8000-000000000006
# ╠═05000003-0000-4000-8000-000000000003
# ╟─02000004-0000-4000-8000-000000000001
# ╠═02000005-0000-4000-8000-000000000001
# ╟─02000007-0000-4000-8000-000000000001
# ╟─03000001-0000-4000-8000-000000000001
# ╠═03000002-0000-4000-8000-000000000001
# ╠═03000002-0000-4000-8000-000000000002
# ╠═03000002-0000-4000-8000-000000000003
# ╟─03000002-0000-4000-8000-000000000004
# ╟─04000001-0000-4000-8000-000000000001
# ╠═04000002-0000-4000-8000-000000000001
# ╠═04000002-0000-4000-8000-000000000002
# ╠═04000003-0000-4000-8000-000000000001
# ╠═04000004-0000-4000-8000-000000000001
# ╟─04000005-0000-4000-8000-000000000001
# ╟─05000001-0000-4000-8000-000000000001
# ╟─05000002-0000-4000-8000-000000000001
# ╟─05000002-0000-4000-8000-000000000002
# ╟─05000003-0000-4000-8000-000000000001
# ╠═05000003-0000-4000-8000-000000000002
# ╠═05000003-0000-4000-8000-000000000004
# ╟─05000004-0000-4000-8000-000000000001
# ╠═05000004-0000-4000-8000-000000000002
# ╠═05000004-0000-4000-8000-000000000003
# ╠═20ce81a0-4eed-4b58-9123-efd38b978e54
# ╠═05000004-0000-4000-8000-000000000004
# ╠═05000004-0000-4000-8000-000000000005
# ╠═05000004-0000-4000-8000-000000000006
# ╠═05000004-0000-4000-8000-000000000007
# ╟─05000004-0000-4000-8000-000000000008
# ╠═05000004-0000-4000-8000-000000000009
# ╟─06000001-0000-4000-8000-000000000001
# ╠═06000002-0000-4000-8000-000000000001
# ╠═06000003-0000-4000-8000-000000000001
# ╟─07000001-0000-4000-8000-000000000001
# ╠═07000002-0000-4000-8000-000000000001
# ╠═07000003-0000-4000-8000-000000000001
# ╠═07000004-0000-4000-8000-000000000001
# ╠═07000005-0000-4000-8000-000000000001
# ╟─08000001-0000-4000-8000-000000000001
# ╠═08000002-0000-4000-8000-000000000001
# ╟─09000001-0000-4000-8000-000000000001
# ╟─09000003-0000-4000-8000-000000000001
# ╠═09000004-0000-4000-8000-000000000001
# ╟─09000005-0000-4000-8000-000000000001
# ╟─09000006-0000-4000-8000-000000000001
# ╠═09000006-0000-4000-8000-000000000010
# ╟─09000006-0000-4000-8000-000000000020
# ╟─09000010-0000-4000-8000-000000000001
# ╠═09000002-0000-4000-8000-000000000001
# ╟─10000001-0000-4000-8000-000000000001
# ╠═10000002-0000-4000-8000-000000000001
# ╟─11000001-0000-4000-8000-000000000001
# ╟─11000002-0000-4000-8000-000000000001
# ╟─11000003-0000-4000-8000-000000000001
# ╟─12000001-0000-4000-8000-000000000001
