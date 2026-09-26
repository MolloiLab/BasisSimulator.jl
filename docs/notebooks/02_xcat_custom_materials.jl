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
| **Phantom** | XCAT adult-male 50th-percentile chest, one axial slab at full resolution (XCIST `phantoms-voxelized`, downloaded as a Julia artifact) |
| **Materials** | `XA.Materials.ncat_*` tissues, plus a custom iodinated-blood material constructed inline |
| **Scanner** | GE Revolution Apex Elite (as in notebook 01) |
| **Reconstruction** | FBP and Hybrid IR at strength 60, both after the water beam-hardening correction |
"""

# ╔═╡ 02000002-0000-4000-8000-000000000001
md"""
## Notebook Setup

As in notebook 01: the shared `docs/` environment, then the device. **Unitful** supplies the
`u"eV"` and `u"g/cm^3"` units the `XA.Material` constructor takes.
"""

# ╔═╡ 02000003-0000-4000-8000-000000000002
# ╠═╡ show_logs = false
import BasisSimulator as BS

# ╔═╡ 02000003-0000-4000-8000-000000000003
# Use CairoMakie for faithful build-time rendering. Snapshot can still isolate
# and compile independent browser-safe islands without hoisting this import.
import CairoMakie as Mke

# ╔═╡ 05000003-0000-4000-8000-000000000003
import Unitful: ustrip, uconvert

# ╔═╡ 1ab8176f-2b8e-4948-a332-f326aed838c9
TableOfContents()

# ╔═╡ 02000004-0000-4000-8000-000000000001
md"""
#### Choosing the device

`GPUSelect.Storage()` returns the array type of the GPU it finds (`CuArray`, `MtlArray`,
`ROCArray`, `oneArray`), or `Array` on a CPU-only host; moving the phantom mask to it decides
where the pipeline runs.
"""

# ╔═╡ 02000005-0000-4000-8000-000000000001
begin
    import GPUSelect
    AT = GPUSelect.Storage()     # the backend array type, directly: MtlArray / CuArray / ROCArray
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end;

# ╔═╡ 02000007-0000-4000-8000-000000000001
Markdown.parse("""
**Backend detected:** $(GPU_BACKEND.name)
""")

# ╔═╡ 03000000-0000-4000-8000-000000000000
md"""
## Phantom Construction
"""

# ╔═╡ 03000001-0000-4000-8000-000000000001
md"""
### 1. Download the XCAT phantom

No local files, no manual setup.  `BS.load_xcat_male_chest()` fetches the
voxelized **XCAT adult-male 50th-percentile chest** phantom from the open
[`xcist/phantoms-voxelized`](https://github.com/xcist/phantoms-voxelized)
repository (BSD-3-Clause) into Julia's content-addressed artifact store: it downloads once,
verifies the file's SHA-256 against the value pinned in the package, logs the Segars XCAT +
XCIST citation, and reuses the cached copy on every later call.

!!! info "Using the XCAT loaders"
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
    - `path = …` — skip the download and read an already-extracted copy from disk.
    - `quiet = true` — silence the citation + progress logging.
"""

# ╔═╡ 05000001-0000-4000-8000-000000000001
md"""
### 2. Custom materials via `XrayAttenuation`

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
    Markdown.parse("""
    **`XA.Materials.ncat_blood`**

    | Field         | Value                                    |
    |---------------|------------------------------------------|
    | `name`        | $(m.name)                                |
    | `ZA_ratio`    | $(round(m.ZA_ratio; digits=4))           |
    | `I`           | $(m.I)                                   |
    | `density`     | $(m.density)                             |
    | `composition` | $(length(m.composition)) elements (Z = $(sort(collect(keys(m.composition))))) |
    """)
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
end;

# ╔═╡ 05000003-0000-4000-8000-000000000004
iodine_blood = build_iodine_blood(5.0);

# ╔═╡ 05000004-0000-4000-8000-000000000001
md"""
#### c. The canonical name → material map

The loader ships the canonical NCAT/XCAT mapping as `BS.xcat_default_materials()`
— every XCIST material name → an `XrayAttenuation` material.  This is what
replaces hand-parsing a per-organ spreadsheet: the compositions already live in
XrayAttenuation's NCAT catalog.  Copy the dict, edit it by name, and hand it
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
end;

# ╔═╡ 06000001-0000-4000-8000-000000000001
md"""
### 3. Build the `Phantom`

`load_xcat_male_slab` returns the labeled pieces; we assemble the `Phantom` from
them.  For a GPU simulation the mask must live on the device — the workspace
picks its compute backend from the mask's array type — so we `to_gpu` it first.
The slab is a single axial section at the phantom's full in-plane resolution: real anatomy at a
docs-friendly compute cost, with no downsampling.
"""

# ╔═╡ 04000004-0000-4000-8000-000000000001
xcat = try
    BS.load_xcat_male_slab(; materials = materials_map, quiet = true)   # full-resolution single slab
catch err
    @warn "XCAT download/load failed; the compute cells will skip" exception = err
    nothing
end;

# ╔═╡ 06000000-0000-4000-8000-0000000000a1
phantom_labeled = xcat === nothing ? nothing : xcat.mask;

# ╔═╡ 06000002-0000-4000-8000-000000000001
VOXEL_SIZE_CM = xcat === nothing ? (0.15, 0.15, 0.15) : xcat.voxel_size_cm;

# ╔═╡ 06000003-0000-4000-8000-000000000001
phantom = xcat === nothing ? nothing :
    BS.Phantom(to_gpu(xcat.mask), xcat.materials, xcat.voxel_size_cm);

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
            subtitle = "$(size(phantom_labeled, 1)) × $(size(phantom_labeled, 2)) × $(size(phantom_labeled, 3))" *
                " voxels of $(join(round.(VOXEL_SIZE_CM .* 10; digits = 3), " × ")) mm" *
                " · $(n_lbl) labels (colour = label ID)",
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
### 1. Scanner, protocol, sim & recon options

The GE Revolution Apex Elite of notebook 01. The protocol is a body CTA: 120 kVp / 250 mA,
5 mm collimation, 500 views in 1 s, each view integrated over the arc the gantry turns through
while the detector reads it (`view_samples = 5`). Reconstruction: 512 × 512 over 35 cm, eight
0.625 mm slices.
"""

# ╔═╡ 07000002-0000-4000-8000-000000000001
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
    # Electronic noise, in electrons, enters the counts before the log, as a real DAS does,
    # so it passes through the reconstruction and iterative reconstruction can act on it.
    electronic_noise = 3500.0,   # e⁻ rms
    detection_gain = 10.0,
);

# ╔═╡ 07000003-0000-4000-8000-000000000001
protocol = BS.CTProtocol(
    kVp = 120,
    mA = 250.0,
    views = 500,
    rotation_time = 1.0,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 07000004-0000-4000-8000-000000000001
# `projector` picks the forward ray tracer: :dd_fast (default) is distance-driven and
# anti-aliased, and walks the volume once for the whole spectrum; :siddon point-samples
# the volume and can alias in strongly beam-hardened regions. The Hybrid-IR cell reads
# `sim_opts.projector`, so its system matrix always matches the operator that made the data.
# `view_samples = 5`: the detector integrates each view while the gantry turns (notebook 01).
sim_opts = BS.SimOptions(seed = 1234, projector = :dd_fast, view_samples = 5);

# ╔═╡ 07000005-0000-4000-8000-000000000001
recon_opts = BS.ReconOptions(
    matrix_size = (512, 512, 8),
    fov_cm = 35.0,
    z_cm = 0.5,
);

# ╔═╡ 08000001-0000-4000-8000-000000000001
md"""
### 2. Forward project

The notebook 01 pattern: `create_workspace`, `simulate!`, copy the sinogram off the device, release the
device buffers. `simulate!` returns the acquisition's dose report.
"""

# ╔═╡ 08000002-0000-4000-8000-000000000001
sim = phantom === nothing ? nothing : let
        @info "Simulating XCAT body CTA: 120 kVp / 250 mA…"
        ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
        out = BS.simulate!(ws, phantom, protocol, sim_opts)

        result = (sino = Array(ws.sinogram), geom = ws.geom, dose = out.dose)

        ws = nothing
        GC.gc(true)

        result
end;

# ╔═╡ 08000003-0000-4000-8000-000000000001
sim === nothing ? md"" : sim.dose

# ╔═╡ 09000000-0000-4000-8000-000000000000
md"""
## Reconstruction
"""

# ╔═╡ 09000001-0000-4000-8000-000000000001
md"""
### 1. The reconstruction chain

The chain of notebook 01, applied to both algorithms:

| step | function | what it does |
|:--|:--|:--|
| 1 | `calibrate_bhc_water` | per detector column, the polynomial mapping polychromatic water line integrals to monochromatic ones, from the full detected spectrum; no tunable parameters |
| 2 | `apply_bhc_water` | applies it to the sinogram, before reconstruction |
| 3 | `reconstruct!` | FBP (`create_fdk_recon_workspace`) or Hybrid IR (`create_hir_recon_workspace`) |
| 4 | `to_hounsfield` | μ → HU with the correction's own water μ at its reference energy |

!!! info "Noise is already in the counts"
    Quantum noise and the electronic noise of `scanner.electronic_noise` are both added by
    `simulate!` to the detector counts, before the log, where a real detector adds them. The
    reconstruction sees all of it, so iterative reconstruction can reduce all of it. Adding
    noise to the HU volume after reconstruction would be identical for FBP and HIR by
    construction.
"""

# ╔═╡ 09000003-0000-4000-8000-000000000001
md"""
#### a. Calibrate the water BHC

One calibration for the 120 kVp beam (2.5 mm flat filter + 4.5 mm Al + bowtie). It returns a
`WaterBHC` with one polynomial per detector column and its reference `μ_water_ref`, used by both
reconstructions below.
"""

# ╔═╡ 09000004-0000-4000-8000-000000000001
bhc_calibration = sim === nothing ? nothing : let
        # Parameter-free water BHC — physics only, no tunables, no thresholds:
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
Markdown.parse("""
**Calibrated:**
* ref energy = $(bhc_calibration === nothing ? "—" : round(bhc_calibration.ref_E_keV, digits = 1)) keV,
* lac water = $(bhc_calibration === nothing ? "—" : round(bhc_calibration.μ_water, digits = 5)) cm⁻¹.
""")

# ╔═╡ 09000010-0000-4000-8000-000000000001
md"""
### 2. FBP

Water BHC → FDK with the `:standard` kernel → HU, releasing the device buffers at the end.
"""

# ╔═╡ 09000002-0000-4000-8000-000000000001
hu_fbp = sim === nothing ? nothing : let
    sino_gpu = BS.apply_bhc_water(to_gpu(sim.sino), bhc_calibration.model)
    ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, sim.geom, recon_opts.matrix_size)
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, sim.geom)
    hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc_calibration.μ_water))
    ws_fdk = nothing; sino_gpu = nothing; recon_μ = nothing
    GC.gc(true)
    hu
end;

# ╔═╡ 10000001-0000-4000-8000-000000000001
md"""
### 3. Hybrid IR

The same chain with `create_hir_recon_workspace(...; strength = 60)` in place of the FDK
workspace.

!!! info "What Hybrid IR does"
    An FBP start, then ordered-subsets penalised weighted least squares with an edge-preserving
    Huber prior: the open-literature (Fessler) counterpart of the vendor hybrid IR algorithms
    (GE ASIR-V, Siemens SAFIRE, Philips iDose⁴, Canon AIDR 3D), whose noise-reduction range its
    strength table targets.

!!! tip "One dial: strength"
    A percentage in steps of 10, read like the GE ASIR-V dial: `0` is pure FBP (the iterative
    loop is skipped), `60` the standard clinical setting used here, `100` the maximum noise
    reduction. It moves the regularisation weight, the Huber threshold, the relaxation and the
    number of epochs together along one calibrated trajectory.

!!! warning "Projector consistency"
    Hybrid IR's data term uses a forward projector, which must be the one that generated the
    sinogram. The cell passes `projector = sim_opts.projector`, so changing the simulation's
    projector keeps the reconstruction consistent. FBP has no forward projector and is unaffected.
"""

# ╔═╡ 10000002-0000-4000-8000-000000000001
hu_hir = sim === nothing ? nothing : let
    sino_gpu = BS.apply_bhc_water(to_gpu(sim.sino), bhc_calibration.model)
    ws_hir = BS.create_hir_recon_workspace(
        sino_gpu, sim.geom, recon_opts.matrix_size; strength = 60, projector = sim_opts.projector,
    )
    recon_μ = BS.reconstruct!(ws_hir, sino_gpu, sim.geom)
    hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc_calibration.μ_water))
    ws_hir = nothing; sino_gpu = nothing; recon_μ = nothing
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

Both reconstructions share the whole chain except the algorithm. The soft-tissue window shows the
iodine-enhanced blood pools (the cardiac chambers) and the difference in noise texture between
the two.

!!! warning "Measure noise inside one material, not inside a box"
    A box near the image centre spans lung to iodinated blood, so its standard deviation is
    anatomy, not noise, and would hide what HIR does. The cell below carries the phantom labels
    onto the reconstruction grid with `BS.resample_to_recon` and erodes the myocardium label, so
    only voxels surrounded by myocardium are measured.
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
        colorrng = (-300, 550)   # window W 850 / L 125: soft tissue and the iodinated blood pools
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
        labels = BS.resample_to_recon(phantom, sim.geom, recon_opts.matrix_size)
        heart = first(k for (k, v) in xcat.label_names if v == "ncat_heart")
        # a voxel counts only if every in-plane neighbour within r is myocardium too:
        # boundary voxels mix two tissues, and their spread is the edge, not the noise
        r = 2
        nx, ny, nz = size(labels)
        roi = [CartesianIndex(i, j, k) for k in 1:nz, j in (1 + r):(ny - r), i in (1 + r):(nx - r)
               if all(labels[i + di, j + dj, k] == heart for dj in -r:r, di in -r:r)]
        σ_fbp = Float64(std(hu_fbp[roi])); σ_hir = Float64(std(hu_hir[roi]))
        band = BS.get_hir_params(60).target_noise_reduction
        Markdown.parse("""
        **Noise inside the myocardium** ($(length(roi)) voxels, all slices)

        | algorithm | mean (HU) | σ (HU) |
        |:--|--:|--:|
        | FBP | $(round(Float64(mean(hu_fbp[roi])); digits = 1)) | $(round(σ_fbp; digits = 1)) |
        | Hybrid IR, strength 60 | $(round(Float64(mean(hu_hir[roi])); digits = 1)) | $(round(σ_hir; digits = 1)) |

        Hybrid IR lowers the noise by **$(round(100 * (1 - σ_hir / σ_fbp); digits = 1)) %** and leaves the mean
        where it was. The package's calibration band for strength 60 is $(band[1])–$(band[2]) %.
        """)
    end
end

# ╔═╡ 12000001-0000-4000-8000-000000000001
md"""
## Summary

Three ideas on top of the five-struct API of notebook 01:

- **Voxel phantoms from an artifact.** `BS.load_xcat_male_slab()` (and the other
  `load_xcat_*` loaders) download a published XCAT phantom once, verify it, and return the
  labeled mask, a material per label and the voxel size;
  `BS.Phantom(to_gpu(mask), materials, voxel_size)` makes it simulatable. Any other labeled mask works the same way.
- **Materials you define.** `XA.Materials.ncat_*` for the standard tissues, or
  `XA.Material(name, ZA, I, density, composition)` for a contrast bolus, a calibration solution
  or an alloy; the composition is a `Dict` of atomic number → mass fraction, and the loader's
  `materials =` keyword swaps it in by XCIST name.
- **Hybrid IR.** `create_hir_recon_workspace(sino, geom, matrix; strength)` in place of the FDK
  workspace: the same `reconstruct!` call, lower noise at the same mean HU.
"""

# ╔═╡ Cell order:
# ╟─02000001-0000-4000-8000-000000000001
# ╟─02000002-0000-4000-8000-000000000001
# ╟─02000003-0000-4000-8000-000000000001
# ╠═02000003-0000-4000-8000-000000000002
# ╟─02000003-0000-4000-8000-000000000003
# ╟─02000003-0000-4000-8000-000000000004
# ╟─02000003-0000-4000-8000-000000000005
# ╟─02000003-0000-4000-8000-000000000006
# ╟─05000003-0000-4000-8000-000000000003
# ╟─b0c50f56-f6fc-4589-b699-c25f51d6b247
# ╟─1ab8176f-2b8e-4948-a332-f326aed838c9
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
# ╠═08000003-0000-4000-8000-000000000001
# ╟─09000000-0000-4000-8000-000000000000
# ╟─09000001-0000-4000-8000-000000000001
# ╟─09000003-0000-4000-8000-000000000001
# ╠═09000004-0000-4000-8000-000000000001
# ╟─09000005-0000-4000-8000-000000000001
# ╟─09000010-0000-4000-8000-000000000001
# ╠═09000002-0000-4000-8000-000000000001
# ╟─10000001-0000-4000-8000-000000000001
# ╠═10000002-0000-4000-8000-000000000001
# ╟─11000000-0000-4000-8000-000000000000
# ╟─11000001-0000-4000-8000-000000000001
# ╟─11000002-0000-4000-8000-000000000001
# ╟─11000003-0000-4000-8000-000000000001
# ╟─12000001-0000-4000-8000-000000000001
