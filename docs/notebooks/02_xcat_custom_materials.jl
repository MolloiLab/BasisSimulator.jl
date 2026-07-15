### A Pluto.jl notebook ###
# v0.2.3

using Markdown
using InteractiveUtils

# ╔═╡ 02000003-0000-4000-8000-000000000001
# ╠═╡ show_logs = false
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

# ╔═╡ b0c50f56-f6fc-4589-b699-c25f51d6b247
using PlutoUI: TableOfContents

# ╔═╡ 02000001-0000-4000-8000-000000000001
md"""
# XCAT Phantom + Custom Materials

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
| **Phantom** | XCAT adult-male 50th-percentile chest slab (1385 × 917 × 1, 0.25 mm in-plane, full resolution) |
| **Materials** | Mix of `XA.Materials.ncat_*` prebuilt + a custom iodinated-blood material constructed inline |
| **Scanner** | GE Revolution Apex Elite (same as notebook 01) |
| **Recon** | FBP/FDK + Hybrid IR (`:asir`-style PWLS refinement) |
"""

# ╔═╡ 02000002-0000-4000-8000-000000000001
md"""
## Notebook Setup

Same shape as notebook 01 — activate `docs/Project.toml`, four discrete
import cells, then probe for a GPU backend.  We add **Unitful** here for
the `u"eV"` and `u"g/cm^3"` units used by the `XA.Material` constructor.
"""

# ╔═╡ 02000003-0000-4000-8000-000000000002
# ╠═╡ show_logs = false
import BasisSimulator as BS

# ╔═╡ 02000003-0000-4000-8000-000000000003
# import CairoMakie as Mke
import WasmMakie as Mke

# ╔═╡ 05000003-0000-4000-8000-000000000003
import Unitful: ustrip, uconvert

# ╔═╡ 1ab8176f-2b8e-4948-a332-f326aed838c9
TableOfContents()

# ╔═╡ 02000004-0000-4000-8000-000000000001
md"""
#### Backend-agnostic device transfer: `to_gpu()`

Same probe-then-load pattern as notebook 01 — checks Metal → CUDA →
AMDGPU via `Base.locate_package`, falls back to CPU `identity`.
"""

# ╔═╡ 02000005-0000-4000-8000-000000000001
begin
    import GPUSelect
    AT = GPUSelect.Storage()     # the backend array type, directly: MtlArray / CuArray / ROCArray
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end

# ╔═╡ 02000007-0000-4000-8000-000000000001
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 03000000-0000-4000-8000-000000000000
md"""
## Phantom Construction
"""

# ╔═╡ 03000001-0000-4000-8000-000000000001
md"""
### 01. Download the XCAT phantom

No local files, no manual setup.  `BS.load_xcat_male_chest()` fetches the
voxelized **XCAT adult-male 50th-percentile chest** phantom from the open
[`xcist/phantoms-voxelized`](https://github.com/xcist/phantoms-voxelized)
repository (BSD-3-Clause) into Julia\'s standard artifact store
(`~/.julia/artifacts/`): it downloads once (~65 MB, SHA-256-verified), logs the
Segars XCAT + XCIST citation, and reuses the cached copy on every later call.

!!! info "Using `load_xcat_*`"
    Four phantoms ship — `:female_slab`, `:female_chest`, `:male_slab`,
    `:male_chest` (see `BS.xcat_phantoms()`).  Each `load_xcat_<name>()` returns
    the **labeled pieces**: `mask`, a `materials` dict (`label => XA.Material`),
    `label_names`, and `voxel_size_cm` — which you turn into a `Phantom` in one
    line (§3).  Useful keywords:

    - `downsample = n` — label-preserving majority-vote shrink (physical extent
      preserved).  This notebook uses the **slab** at full 0.25 mm resolution, so
      it passes no `downsample`; reach for it on the full `:male_chest` volume.
      It is lossy — a 6× round-trip agrees with full-res on ~96% of voxels — so
      treat it as a speed knob, not a lossless representation.
    - `materials = Dict("ncat_blood" => …)` — override the tissue → material map
      by XCIST name *before* assembly (see §2).
    - `path = "/my/local/xcat"` — skip the download and read an already-extracted
      copy from disk.
    - `quiet = true` — silence the citation + progress logging.
"""

# ╔═╡ 05000001-0000-4000-8000-000000000001
md"""
### 02. Custom materials via `XrayAttenuation`

The loader assigns every label a sensible default tissue, but the whole point of
a digital phantom is that **you control the material physics**.  Each label maps
to an [`XA.Material`](https://github.com/MolloiLab/XrayAttenuation.jl) that gives
the simulator its energy-dependent μ(E).  Two ways to get one:

1. **Prebuilt** — `XA.Materials.ncat_blood`, `XA.Materials.ncat_muscle`, etc.
   XrayAttenuation ships the canonical NCAT/XCAT tissue compositions.
2. **Constructed inline** — `XA.Material(name, ZA_ratio, I, density, composition)`
   for *your specific* contrast bolus, calibration solution, or alloy.
"""

# ╔═╡ 05000002-0000-4000-8000-000000000001
md"""
#### a. Pull a prebuilt NCAT material

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
#### b. Construct a custom contrast material

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
#### c. The canonical name → material map

The loader ships the canonical NCAT/XCAT mapping as `BS.xcat_default_materials()`
— every XCIST material name → an `XrayAttenuation` material.  This is what
replaces hand-parsing a per-organ spreadsheet: the compositions already live in
XrayAttenuation\'s NCAT catalog.  Copy the dict, edit it by name, and hand it
back to the loader via `materials =` (§d).
"""

# ╔═╡ 05000004-0000-4000-8000-0000000000c1
let m = BS.xcat_default_materials()
    rows = ["| `$(k)` | `$(v.name)` |" for (k, v) in sort(collect(m); by = first)]
    Markdown.parse("**`BS.xcat_default_materials()`**\n\n" *
        "| XCIST name | XrayAttenuation material |\n|---|---|\n" * join(rows, "\n"))
end

# ╔═╡ 05000004-0000-4000-8000-000000000008
md"""
#### d. Override a material

Copy the default map and reassign by XCIST name — here we swap the blood pool for
the 5 mg/mL `iodine_blood` we built in §b, so the heart chambers and great
vessels light up like a contrast-enhanced CTA.  The edited map is passed to the
loader via `materials =` in §3.
"""

# ╔═╡ 05000004-0000-4000-8000-000000000009
materials_map = let m = BS.xcat_default_materials()
    m["ncat_blood"] = iodine_blood        # dope the blood pool with 5 mg/mL iodine contrast
    m
end

# ╔═╡ 06000001-0000-4000-8000-000000000001
md"""
### 03. Build the `Phantom`

`load_xcat_male_slab` returns the labeled pieces; we assemble the `Phantom` from
them.  For a GPU simulation the mask must live on the device — the workspace
picks its compute backend from the mask\'s array type — so we `to_gpu` it first.
The slab is one 5 mm-thick axial section at full 0.25 mm in-plane resolution: real
anatomy at a docs-friendly compute cost, no downsampling needed.
"""

# ╔═╡ 04000004-0000-4000-8000-000000000001
xcat = try
    BS.load_xcat_male_slab(; materials = materials_map, quiet = true)   # full-resolution single slab
catch err
    @warn "XCAT download/load failed — compute cells will skip" err
    nothing
end

# ╔═╡ 06000000-0000-4000-8000-0000000000a1
phantom_labeled = xcat === nothing ? nothing : xcat.mask

# ╔═╡ 06000002-0000-4000-8000-000000000001
VOXEL_SIZE_CM = xcat === nothing ? (0.15, 0.15, 0.15) : xcat.voxel_size_cm

# ╔═╡ 06000003-0000-4000-8000-000000000001
phantom = xcat === nothing ? nothing :
    BS.Phantom(to_gpu(xcat.mask), xcat.materials, xcat.voxel_size_cm)

# ╔═╡ 04000005-0000-4000-8000-000000000001
let
    if phantom_labeled === nothing
        md"""
        !!! warning "Skipped — see 1 above"
        """
    else
        mid = max(1, size(phantom_labeled, 3) ÷ 2)
        n_lbl = length(unique(phantom_labeled))
        slice = phantom_labeled[:, :, mid]

        fig = Mke.Figure(size = (900, 600))
        ax = Mke.Axis(
            fig[1, 1];
            title = "XCAT male chest slab (full resolution)",
            subtitle = "Slice $(mid) / $(size(phantom_labeled, 3))" *
                " · $(size(phantom_labeled, 1))×$(size(phantom_labeled, 2))" *
                " · $(n_lbl) unique labels",
            aspect = Mke.DataAspect(),
            titlesize = 28, subtitlesize = 20,
        )
        hm = Mke.heatmap!(ax, Float32.(slice); colormap = :tab20)
        Mke.hidedecorations!(ax)
        Mke.Colorbar(fig[1, 2], hm; label = "label", width = 14, labelsize = 18)

        Mke.save(
            joinpath(@__DIR__, "..", "assets", "xcat_phantom.png"),
            fig; px_per_unit = 2
        )
        fig
    end
end

# ╔═╡ 07000000-0000-4000-8000-000000000000
md"""
## Scan Setup & Simulation
"""

# ╔═╡ 07000001-0000-4000-8000-000000000001
md"""
### 01. Scanner, protocol, sim & recon options

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
    # DAS/electronic noise, in electrons.  This enters the COUNTS before the log
    # transform (`λ_noisy += σ_e·randn()` in simulate!), which is where a real
    # DAS contributes it — so it propagates through reconstruction and iterative
    # recon can suppress it.  Adding an equivalent floor to the HU volume *after*
    # reconstruction instead would make FBP and HIR look identical by fiat.
    electronic_noise = 3500.0,   # e⁻ — clinical GE Apex Elite DAS readout noise
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
# `projector` picks the forward ray tracer: :dd_fast (default) is distance-driven and
# anti-aliased (robust in severe beam-hardened regions), fusing the whole spectrum into
# one volume walk. :siddon point-samples one voxel per step and ALIASES in those regions —
# use it only when speed outranks accuracy. (:dd is the older, slower reference kernel:
# same physics as :dd_fast, now deprecated.)
# The BHC and Hybrid-IR cells below all read `sim_opts.projector`, so changing it
# HERE updates the whole pipeline consistently — the recon must invert the operator
# that generated the data, or the IR system matrix won't match and convergence suffers.
sim_opts = BS.SimOptions(fidelity = :eict, seed = 1234, projector = :dd_fast)

# ╔═╡ 07000005-0000-4000-8000-000000000001
recon_opts = BS.ReconOptions(
    matrix_size = (512, 512, 8),
    fov_cm = 35.0,
    z_cm = 0.5,
)

# ╔═╡ 08000001-0000-4000-8000-000000000001
md"""
### 02. Forward project

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

# ╔═╡ 09000000-0000-4000-8000-000000000000
md"""
## Reconstruction
"""

# ╔═╡ 09000001-0000-4000-8000-000000000001
md"""
### 01. Postprocessing (The Full Correction Pipeline)

A raw FDK or HIR recon doesn't ship as a clinical image.  Real CT vendors
apply a stack of corrections after the simulator's forward model:

| Stage | Function                              | What it does                                                                             |
|-------|---------------------------------------|------------------------------------------------------------------------------------------|
| 1.    | `calibrate_bhc_water`                 | Precomputes per-column water poly→mono polynomials from the FULL detected spectrum — pure physics, zero tunables |
| 2.    | `apply_bhc_water`                     | Sinogram-domain water BHC — one polynomial pass *before* FDK/HIR                          |
| 3.    | `reconstruct!` (FDK or Hybrid IR)     | Filtered back-projection / iterative reconstruction on the corrected sinogram            |
| 5.    | `to_hounsfield` (with BHC μ_water)    | Convert μ → HU using the BHC model's calibrated reference (not the 70 keV NIST value)    |
| 6.    | `measure_radial_cupping`              | QA metric (never applied): fitted residual cup + DC — both ≈ 0 after a correct BHC        |

!!! warning "Where the DAS noise goes"
    Quantum (Poisson) **and** DAS/electronic noise are both injected by
    `simulate!` in the **counts domain**, before the log transform — electronic
    noise via `scanner.electronic_noise` (electrons). That is where a real DAS
    contributes it, and it means reconstruction sees it and can suppress it.

    Do **not** substitute `add_system_noise_floor!` here. That helper adds white
    Gaussian noise to the HU volume *after* reconstruction, so it is identical in
    FBP and HIR by construction and makes iterative recon look useless — an
    earlier version of this notebook set `electronic_noise = 0` and added a 28 HU
    floor post-recon, which reported HIR at ~1.6 % noise reduction instead of the
    ~26 % it actually delivers. Reserve that helper for effects that genuinely
    survive reconstruction (calibration drift, ring-correction residuals).

The same pipeline applies to both FBP and HIR — only the recon workspace
inside the let block differs (`create_fdk_recon_workspace` vs
`create_hir_recon_workspace`). See notebook 01 §9 for the same pattern on
the Gammex 472 phantom.
"""

# ╔═╡ 09000003-0000-4000-8000-000000000001
md"""
#### a. Calibrate the BHC model

One-time spectrum fit at 120 kVp + 7 mm Al filtration. Returns a
`TwoMaterialBHC` polynomial model + the calibrated `μ_water_ref` used by
both FBP and HIR pipelines below for `to_hounsfield`.
"""

# ╔═╡ 09000004-0000-4000-8000-000000000001
bhc_calibration = sim === nothing ? nothing : let
        # KNOBLESS water BHC — pure physics, zero tunables, zero thresholds:
        # per-column poly→mono polynomials from the FULL detected spectrum
        # (tube × filters × bowtie × heel × η(E) — whatever sim_opts enabled).
        model = BS.calibrate_bhc_water(
            sim_opts, protocol;
            scanner = scanner, geom = sim.geom,
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
#### b. Polychromatic `μ_water` from the XCAT body — *informational*

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
### 02. FBP with the full correction pipeline

Same `let ... end` shape as notebook 01 — detected-spectrum water BHC → FDK
→ HU, with explicit GPU cleanup and non-mutating cupping QA at the end.
HU conversion uses `bhc_calibration.μ_water` (the BHC's calibrated
`μ_water_ref` at the spectrum-mean energy), since the post-BHC recon
reads as approximately monochromatic at that reference.
"""

# ╔═╡ 09000002-0000-4000-8000-000000000001
hu_fbp = sim === nothing ? nothing : let
        matrix_size = recon_opts.matrix_size

        # 1. Sinogram-domain BHC (returns a CPU array)
        sino_gpu = to_gpu(sim.sino)
        sino_bhc = BS.apply_bhc_water(sino_gpu, bhc_calibration.model)
        sino_gpu = to_gpu(sino_bhc)

        # 2. FDK on the corrected sinogram
        ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, sim.geom, matrix_size)
        recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, sim.geom)

    # (image-domain BHC removed — audit: scaled self-subtraction that
    #  deflated dense-material HU; the knobless sinogram water BHC is the
    #  whole correction)

        # 4. μ → HU using BHC's calibrated μ_water_ref
        hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc_calibration.μ_water))

        # 5. DAS/electronic noise is injected in the counts domain by simulate!
        # (scanner.electronic_noise), not bolted onto the HU volume here.

        # 6. Residual radial cupping is measured as QA, never applied
        # (cupping is QA-only now — measure_radial_cupping; never applied)

        # GPU cleanup
        ws_fdk = nothing
        sino_gpu = nothing
        recon_μ = nothing
        GC.gc(true)

        hu
end;

# ╔═╡ 10000001-0000-4000-8000-000000000001
md"""
### 03. Hybrid IR with the full correction pipeline

Identical pipeline to §8 — sino BHC → recon → image BHC → HU → noise
floor → cupping — with `create_hir_recon_workspace(...; strength = 60)`
swapped in for the FDK workspace.

!!! info "What HIR adds over FBP"
    Hybrid IR is the clinical workhorse: vendors call it ASIR, AIDR3D,
    iDose⁴, SAFIRE depending on brand. All share the same architecture —
    FDK init + iterative PWLS refinement with a Huber prior. It costs
    a few FDK passes but returns ~30 % lower pixel σ at matched
    resolution.

!!! tip "One dial: `strength`"
    `strength` is a percentage in 10 % steps, read exactly like the GE
    ASIR-V dial: `0` is pure FBP (the PWLS loop is skipped entirely), `60`
    is the standard clinical setting used here, `100` is maximum noise
    reduction. It moves λ, the Huber edge threshold δ, the relaxation and
    the epoch count together along one tuned trajectory — there is no
    second knob to keep in sync.

!!! warning "Projector consistency (IR only)"
    Hybrid IR's data-fidelity term uses a forward projector `A`, so it
    **must use the same projector that generated the sinogram**. We pass
    `projector = sim_opts.projector` to `create_hir_recon_workspace`, so
    flipping `sim_opts.projector` (e.g. to `:siddon`) automatically
    keeps the recon consistent. FBP (§8) is immune — it has no forward `A`.
"""

# ╔═╡ 10000002-0000-4000-8000-000000000001
hu_hir = sim === nothing ? nothing : let
        matrix_size = recon_opts.matrix_size

        # 1. Sinogram-domain BHC
        sino_gpu = to_gpu(sim.sino)
        sino_bhc = BS.apply_bhc_water(sino_gpu, bhc_calibration.model)
        sino_gpu = to_gpu(sino_bhc)

        # 2. Hybrid IR (FBP init + PWLS refinement, strength = 60 %)
        # projector MUST match the forward sim so the IR system matrix A inverts
        # the operator that generated the data (read from sim_opts, single source).
        ws_hir = BS.create_hir_recon_workspace(
            sino_gpu, sim.geom, matrix_size; strength = 60, projector = sim_opts.projector,
        )
        recon_μ = BS.reconstruct!(ws_hir, sino_gpu, sim.geom)

        # FBP's `reconstruct!` masks voxels outside the inscribed scan FOV
        # (`apply_fov_mask!`); HIR doesn't, so the iterative refinement
        # leaves garbage in the corners.  Apply the same mask for parity.
        BS.apply_fov_mask!(recon_μ, sim.geom)
    
        # 3. μ → HU using BHC's calibrated μ_water_ref
        hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc_calibration.μ_water))

        # GPU cleanup
        ws_hir = nothing
        sino_gpu = nothing
        recon_μ = nothing
        GC.gc(true)

        hu
end;

# ╔═╡ 11000000-0000-4000-8000-000000000000
md"""
## Results
"""

# ╔═╡ 11000001-0000-4000-8000-000000000001
md"""
### Compare FBP vs Hybrid IR

Both reconstructions go through the identical correction pipeline
(water sino-BHC + recon + HU; counts-domain noise; cupping QA). The soft-tissue window shows the
iodine-enhanced blood pools (XCAT labels 19–22 = `bldplLV/RV/LA/RA`) and
the texture / noise contrast between the two algorithms.

!!! warning "Measure noise inside a material, not inside a box"
    A fixed box near the image center spans −794 to +353 HU here — lung
    through iodinated blood — so its σ (≈160 HU) is *anatomy*, not noise, and
    HIR appears to do nothing. The cell below instead uses
    `BS.resample_to_recon` to carry the phantom labels onto the recon grid,
    then erodes the myocardium label so only voxels surrounded by
    myocardium survive. That is the only place where σ means noise.
"""

# ╔═╡ 11000002-0000-4000-8000-000000000001
let
    if hu_fbp === nothing || hu_hir === nothing
        md"""
        !!! warning "Skipped — see 1 for XCAT setup"
        """
    else
        fig = Mke.Figure(size = (1200, 660))

        mid = size(hu_fbp, 3) ÷ 2
        img_fbp = hu_fbp[:, :, mid]
        img_hir = hu_hir[:, :, mid]
        colorrng = (-300, 550)   # soft-tissue window: W=400, L=40
        title_kwargs = (titlesize = 28, subtitlesize = 20)

        ax_fbp = Mke.Axis(
            fig[1, 1];
            title = "FBP (FDK)",
            subtitle = "120 kVp / 250 mA · :standard filter",
            aspect = Mke.DataAspect(),
            title_kwargs...,
        )
        ax_hir = Mke.Axis(
            fig[1, 2];
            title = "Hybrid IR",
            subtitle = "120 kVp / 250 mA · strength = 60 %",
            aspect = Mke.DataAspect(),
            title_kwargs...,
        )

        Mke.heatmap!(ax_fbp, img_fbp; colormap = :grays, colorrange = colorrng)
        hm = Mke.heatmap!(ax_hir, img_hir; colormap = :grays, colorrange = colorrng)

        Mke.hidedecorations!(ax_fbp)
        Mke.hidedecorations!(ax_hir)
        Mke.Colorbar(fig[1, 3], hm; label = "HU", width = 14, labelsize = 18)

        Mke.save(
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
        # Measure σ inside ONE material, not inside a box.  A fixed box near the
        # image center straddles the heart/lung boundary, so its σ is anatomy
        # (≈160 HU) and HIR looks like it does nothing.  Instead, map the phantom
        # labels onto the recon grid and keep only voxels deep inside the
        # myocardium, where the sole remaining variation is noise.
        labels = BS.resample_to_recon(phantom, sim.geom, recon_opts.matrix_size)
        heart = first(k for (k, v) in xcat.label_names if v == "ncat_heart")

        # Erode in-plane by r voxels: a recon voxel is "interior" only if every
        # neighbour within r is the same label.  Boundary voxels are partial
        # volumes of two tissues — their spread is the edge, not the noise.
        r = 2
        nx, ny, nz = size(labels)
        roi = CartesianIndex{3}[]
        @inbounds for k in 1:nz, j in (1 + r):(ny - r), i in (1 + r):(nx - r)
            labels[i, j, k] == heart || continue
            all(labels[i + di, j + dj, k] == heart for dj in -r:r, di in -r:r) &&
                push!(roi, CartesianIndex(i, j, k))
        end

        σ_fbp = std(hu_fbp[roi])
        σ_hir = std(hu_hir[roi])

        md"""
        **Noise (σ in HU, $(length(roi)) voxels deep inside the myocardium,
        mean $(round(Float64(mean(hu_fbp[roi])), digits=1)) HU):**

        | Algorithm  | σ (HU)                    |
        |------------|---------------------------|
        | FBP        | $(round(Float64(σ_fbp), digits=1)) |
        | Hybrid IR  | $(round(Float64(σ_hir), digits=1)) |
        | Reduction  | $(round(100 * (1 - Float64(σ_hir) / Float64(σ_fbp)); digits=1))% |

        Compare against the `strength = 60` target band (25–35 %).  Both quantum
        and DAS/electronic noise enter in the counts domain, so HIR acts on all
        of it — no post-hoc correction needed to read this number.
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
# ╠═b0c50f56-f6fc-4589-b699-c25f51d6b247
# ╠═1ab8176f-2b8e-4948-a332-f326aed838c9
# ╟─02000004-0000-4000-8000-000000000001
# ╠═02000005-0000-4000-8000-000000000001
# ╟─02000007-0000-4000-8000-000000000001
# ╟─03000000-0000-4000-8000-000000000000
# ╟─03000001-0000-4000-8000-000000000001
# ╟─05000001-0000-4000-8000-000000000001
# ╟─05000002-0000-4000-8000-000000000001
# ╟─05000002-0000-4000-8000-000000000002
# ╟─05000003-0000-4000-8000-000000000001
# ╠═05000003-0000-4000-8000-000000000002
# ╠═05000003-0000-4000-8000-000000000004
# ╟─05000004-0000-4000-8000-000000000001
# ╟─05000004-0000-4000-8000-0000000000c1
# ╟─05000004-0000-4000-8000-000000000008
# ╠═05000004-0000-4000-8000-000000000009
# ╟─06000001-0000-4000-8000-000000000001
# ╠═04000004-0000-4000-8000-000000000001
# ╠═06000000-0000-4000-8000-0000000000a1
# ╠═06000002-0000-4000-8000-000000000001
# ╠═06000003-0000-4000-8000-000000000001
# ╟─04000005-0000-4000-8000-000000000001
# ╟─07000000-0000-4000-8000-000000000000
# ╟─07000001-0000-4000-8000-000000000001
# ╠═07000002-0000-4000-8000-000000000001
# ╠═07000003-0000-4000-8000-000000000001
# ╠═07000004-0000-4000-8000-000000000001
# ╠═07000005-0000-4000-8000-000000000001
# ╟─08000001-0000-4000-8000-000000000001
# ╠═08000002-0000-4000-8000-000000000001
# ╟─09000000-0000-4000-8000-000000000000
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
# ╟─11000000-0000-4000-8000-000000000000
# ╟─11000001-0000-4000-8000-000000000001
# ╟─11000002-0000-4000-8000-000000000001
# ╟─11000003-0000-4000-8000-000000000001
# ╟─12000001-0000-4000-8000-000000000001
