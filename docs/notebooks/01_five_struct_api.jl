### A Pluto.jl notebook ###
# v0.1.0

using Markdown
using InteractiveUtils

# ╔═╡ 01000003-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 01000003-0000-4000-8000-000000000005
using Markdown: @md_str

# ╔═╡ 01000003-0000-4000-8000-000000000004
using Statistics: std

# ╔═╡ 01000001-0000-4000-8000-000000000001
md"""
# 01 · The Five-Struct API

**Build a phantom, configure a scanner, run a simulation, reconstruct an image.**

`BasisSimulator.jl` exposes its entire user surface through five structs:

| Struct          | What it is                                              |
|-----------------|---------------------------------------------------------|
| `Phantom`       | The object being scanned (mask + materials + voxel size)|
| `Scanner`       | The CT hardware (geometry, detector, filtration)        |
| `CTProtocol`    | The acquisition (kVp, mA, views, rotation, collimation) |
| `SimOptions`    | The physics model (which effects to include)            |
| `ReconOptions`  | The output (matrix size, FOV, algorithm, filter)        |

This notebook walks through each one — building a Gammex Model 472 phantom and
scanning it on a model of the **GE Revolution Apex Elite**. We'll finish by
comparing a standard-dose acquisition against a low-dose one.
"""

# ╔═╡ 01000002-0000-4000-8000-000000000001
md"""
## Setup

Activate `docs/Project.toml` (which has `BasisSimulator` from the local source
tree + `CairoMakie`), then bring in our imports — one per cell — and finally
detect a GPU backend.
"""

# ╔═╡ 01000003-0000-4000-8000-000000000002
import BasisSimulator as BS

# ╔═╡ 01000003-0000-4000-8000-000000000003
# ╠═╡ show_logs = false
import CairoMakie as CM

# ╔═╡ 01000004-0000-4000-8000-000000000001
md"""
#### Backend-agnostic device transfer: `to_gpu()`

`BasisSimulator` dispatches kernels on array type via
[AcceleratedKernels.jl](https://github.com/JuliaGPU/AcceleratedKernels.jl), so
the same source runs on Metal, CUDA, ROCm, or CPU. The only thing the user
controls is **which array type** they hand the simulator.

The cell below probes for `Metal → CUDA → AMDGPU` in that order, picks the
first one that's installed *and* functional on the host, and falls back to
plain `Array` (CPU) otherwise. Returned as a one-liner: `to_gpu(arr)`.

!!! info "Probe-then-load, never auto-install"
    `Base.locate_package(pkg_id)` returns the package path if it's already
    installed somewhere on the load path, or `nothing` otherwise — **without
    triggering a download or a load attempt**. Only when a package is already
    present do we call `Base.require(...)` to actually load it. So the same
    notebook runs cleanly on a Mac with Metal globally installed, on an
    NVIDIA box with CUDA installed, on an AMD box with AMDGPU, or on a
    CPU-only CI runner with none of them — no extra dependencies forced into
    the project.
"""

# ╔═╡ 01000005-0000-4000-8000-000000000001
begin
    GPU_BACKEND = let
        # (package name, UUID, array-type name) — UUIDs from the General registry
        candidates = [
            (:Metal, "dde4c033-4e86-420c-a63e-0dd931031962", :MtlArray),
            (:CUDA, "052768ef-5323-5732-b1bb-66c8b64840ba", :CuArray),
            (:AMDGPU, "21141c5a-9bdb-4563-92ae-f87d6854732e", :ROCArray),
        ]

        detected = (name = "CPU", to_gpu = identity)

        for (pkg, uuid, ctor) in candidates
            pkg_id = Base.PkgId(Base.UUID(uuid), String(pkg))
            # Skip if the package isn't installed anywhere on the load path.
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
                # Loaded but errored — try the next candidate.
            end
        end

        detected
    end

    to_gpu(x) = GPU_BACKEND.to_gpu(x)
end

# ╔═╡ 01000007-0000-4000-8000-000000000001
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 02000001-0000-4000-8000-000000000001
md"""
## 1. `Phantom`

A `Phantom` is three pieces of data:

1. **Labeled mask** — a 3D integer array. Each integer is a region label.
2. **Materials dict** — maps each label to an `XrayAttenuation.Material`.
3. **Voxel size** — `(dx, dy, dz)` in cm.

The factory `create_gammex_472(...)` builds the standard Gammex Model 472
multi-energy CT phantom: a 33 cm body of solid water with calcium and iodine
inserts at known concentrations (used as a clinical CT calibration standard).
"""

# ╔═╡ 02000002-0000-4000-8000-000000000001
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,    # in-plane resolution (sets phantom mask size)
    n_slices = 16,     # axial slices in the phantom
    fov_cm = 35.0,   # body diameter region
    z_cm = 1.0,    # axial extent
);

# ╔═╡ 02000003-0000-4000-8000-000000000001
md"""
The mask is $(eltype(phantom_cpu.mask)) of shape $(size(phantom_cpu.mask)),
and there are $(length(phantom_cpu.materials)) materials in the dictionary.

Now we move the mask to the GPU **once** and rebuild the `Phantom` struct
around it. Every subsequent forward projection runs on the device.

!!! info "Phantom is GPU-aware"
    `Phantom` is parameterized on its mask array type. Passing a `MtlArray` /
    `CuArray` / `ROCArray` mask makes the whole forward-projection pipeline
    dispatch to GPU kernels — no other code changes needed.
"""

# ╔═╡ 02000004-0000-4000-8000-000000000001
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 02000005-0000-4000-8000-000000000001
let
    # Per-region (name, color) lookup, keyed on `BS.RegionLabel` UInt8 values
    info = Dict(
        UInt8(0) => ("Background", :gray15),
        UInt8(1) => ("Air", :gray30),
        UInt8(2) => ("Pure Water", :royalblue),
        UInt8(3) => ("Solid Water", :lightskyblue),
        UInt8(10) => ("Ca 50 mg/mL", :wheat),
        UInt8(11) => ("Ca 100 mg/mL", :sandybrown),
        UInt8(12) => ("Ca 200 mg/mL", :orange),
        UInt8(13) => ("Ca 300 mg/mL", :darkorange),
        UInt8(14) => ("Ca 400 mg/mL", :orangered),
        UInt8(15) => ("Ca 500 mg/mL", :red3),
        UInt8(16) => ("Ca 600 mg/mL", :firebrick),
        UInt8(20) => ("I 2.0 mg/mL", :honeydew),
        UInt8(21) => ("I 2.5 mg/mL", :palegreen),
        UInt8(22) => ("I 5.0 mg/mL", :lightgreen),
        UInt8(23) => ("I 7.5 mg/mL", :mediumseagreen),
        UInt8(24) => ("I 10.0 mg/mL", :seagreen),
        UInt8(25) => ("I 15.0 mg/mL", :forestgreen),
        UInt8(26) => ("I 20.0 mg/mL", :darkgreen),
    )

    mid = size(phantom_cpu.mask, 3) ÷ 2
    slice = phantom_cpu.mask[:, :, mid]

    # Remap raw labels → contiguous 1..N for a categorical colormap
    labels = sort(unique(slice))
    n = length(labels)
    lut = Dict(l => i for (i, l) in enumerate(labels))
    mapped = Float32[lut[l] for l in slice]

    colors = [info[l][2] for l in labels]
    names = [info[l][1] for l in labels]
    cmap = CM.cgrad(colors, n; categorical = true)

    fig = CM.Figure(size = (900, 600))
    ax = CM.Axis(
        fig[1, 1];
        title = "Input Phantom",
        subtitle = "Gammex Model 472 (Slice $mid)",
        aspect = CM.DataAspect(),
        titlesize = 28, subtitlesize = 20,
    )
    CM.heatmap!(ax, mapped; colormap = cmap, colorrange = (0.5, n + 0.5))
    CM.hidedecorations!(ax)
    CM.Colorbar(
        fig[1, 2];
        colormap = cmap,
        colorrange = (0.5, n + 0.5),
        ticks = (1:n, names),
        ticklabelsize = 12,
        width = 16,
    )

    CM.save(
        joinpath(@__DIR__, "..", "assets", "gammex_472_phantom.png"),
        fig; px_per_unit = 2
    )
    fig
end

# ╔═╡ 03000001-0000-4000-8000-000000000001
md"""
## 2. `Scanner`

`Scanner` describes the **hardware** — properties that don't change between
acquisitions: source-to-isocenter distance, detector geometry, focal spot,
bowtie filter, scintillator material.

The values below model a **GE Revolution Apex Elite** (256-row, 0.625 mm
isotropic detector pixels, curved Lumex array), verified against vendor
specs and clinical data.
"""

# ╔═╡ 03000002-0000-4000-8000-000000000001
scanner = BS.Scanner(
    # Geometry (mm)
    source_to_isocenter = 625.6,
    source_to_detector = 1100.0,

    # Detector array
    detector_rows = 256,
    detector_cols = 834,
    detector_row_size = 0.625,
    detector_col_size = 0.6,
    detector_shape = BS.CURVED_DETECTOR,

    # Focal spot
    focal_spot_width = 1.0,
    focal_spot_length = 1.0,
    target_angle = 10.0,

    # Filtration
    flat_filter_material = :aluminum,
    flat_filter_thickness = 2.5,
    bowtie_filter = :ge_revolution_large,

    # Scintillator
    detector_material = :lumex,
    detector_depth = 3.0,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,

    # DAS noise model
    electronic_noise = 0,       # e⁻ (set ~3500 for clinical DAS readout noise)
    detection_gain = 10.0,    # e⁻/keV
);

# ╔═╡ 03000003-0000-4000-8000-000000000001
md"""
!!! info "Scanner is hardware, not acquisition"
    Anything that varies per scan (kVp, mA, view count, rotation time)
    belongs in `CTProtocol`, not `Scanner`. Treating them as separate structs
    means you can sweep over protocols without re-specifying the scanner each
    time.

!!! info "Bowtie filter"
    `bowtie_filter = :ge_revolution_large` selects a vendor-specific shaped
    aluminum filter that pre-attenuates the periphery of the beam. This
    flattens dose distribution across the patient and is essential for
    realistic noise modeling. The catalog of available bowties lives in
    `src/source/bowtie_filter.jl`.
"""

# ╔═╡ 04000001-0000-4000-8000-000000000001
md"""
## 3. `CTProtocol`

`CTProtocol` is the **acquisition** struct. We define two protocols — a
**standard-dose** scan at 200 mA and a **low-dose** scan at 50 mA — both at
120 kVp. We'll reconstruct each and compare noise at the end.
"""

# ╔═╡ 04000002-0000-4000-8000-000000000001
protocol_standard = BS.CTProtocol(
    kVp = 120,
    mA = 200.0,
    views = 500,
    rotation_time = 1.0,
    collimation_mm = 5.0,                 # 8 × 0.625 mm rows active
    additional_filters = [("Al", 4.5)],       # ~7 mm total Al (matches GE Apex inherent filtration)
);

# ╔═╡ 04000003-0000-4000-8000-000000000001
protocol_lowdose = BS.CTProtocol(
    kVp = 120,
    mA = 50.0,                # 4× lower current → ~2× more pixel noise
    views = 500,
    rotation_time = 1.0,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 04000004-0000-4000-8000-000000000001
md"""
!!! info "`collimation_mm` slices the detector down"
    Our scanner has 256 rows × 0.625 mm = 160 mm of axial coverage, but a typical clinical
    scan uses the minimum amount of coverage to reduce dose. `collimation_mm = 5.0` activates
    the central 8 rows — keeping the simulation fast while still being clinically representative.
"""

# ╔═╡ 05000001-0000-4000-8000-000000000001
md"""
## 4. `SimOptions`

`SimOptions` is the **fidelity dial**: which physics effects to include in the
forward model.

The `:eict` preset enables a clinically-realistic stack — quantum noise,
beam-hardening, scatter convolution, focal-spot blur, bowtie filtration, heel
effect, and detector efficiency. To study any single effect's contribution,
toggle it off below.
"""

# ╔═╡ 05000002-0000-4000-8000-000000000001
sim_opts = BS.SimOptions(
    fidelity = :eict,
    seed = 1234,

    # Override individual physics toggles by uncommenting:
    # use_scatter           = false,   # disable scatter convolution
    # use_heel_effect       = false,   # disable anode heel effect
    # use_focal_spot        = false,   # disable focal-spot blur
    # use_optical_crosstalk = false,   # disable detector optical crosstalk
    # use_lag               = false,   # disable scintillator afterglow
    # use_noise             = false,   # ideal sinogram only (no Poisson + Gaussian)

    projector = :siddon,   # fast ray tracer (~3.5-5.5×); default :dd is anti-aliased.
                             #  :siddon ALIASES in severe beam-hardened regions — speed over
                             #  accuracy. The BHC cells below read sim_opts.projector to match.
);

# ╔═╡ 05000003-0000-4000-8000-000000000001
md"""
!!! info "Two fidelity presets"
    - **`:eict`** — energy-integrating CT (conventional). Beer-Lambert across
      the full source spectrum, scintillator detection, full physics stack.
    - **`:pcct`** — photon-counting CT. Adds CdTe charge transport,
      K-fluorescence, pulse pileup, and an MC-derived detector response
      matrix. Pair with a photon-counting `Scanner`
      (`detector_type = :photon_counting`).

!!! info "Reproducibility"
    `seed = 1234` controls the Poisson + Gaussian noise RNGs. Same seed +
    same SimOptions = bit-identical sinogram. Pass `seed = nothing` for a
    fresh random draw on every call.
"""

# ╔═╡ 06000001-0000-4000-8000-000000000001
md"""
## 5. `ReconOptions`

`ReconOptions` controls the reconstructed output: matrix size, FOV, algorithm,
and apodization filter. For now we'll do plain FBP (Feldkamp-Davis-Kress).
"""

# ╔═╡ 06000002-0000-4000-8000-000000000001
recon_opts = let
    slice_thickness_mm = 0.625
    n_recon_slices = round(Int, 5.0 / slice_thickness_mm)   # collimation_mm / slice_thickness

    BS.ReconOptions(
        matrix_size = (512, 512, n_recon_slices),
        fov_cm = 35.0,
        z_cm = 0.5,
    )
end

# ╔═╡ 06000003-0000-4000-8000-000000000001
md"""
!!! info "Other reconstruction algorithms"
    `algorithm` accepts `:fdk` (default), plus iterative methods `:sirt`,
    `:cgls`, `:tv_sirt`, `:tv_cgls`, `:asir`, `:mbir`. The Hybrid IR pipeline
    uses `:asir` with a PWLS refinement step.

!!! info "FBP filters"
    `filter` accepts `:ram_lak`, `:shepp_logan`, `:cosine`, `:hamming`,
    `:hann`, `:standard` (vendor-tuned), `:soft`, `:bone`. Each apodizes the
    ramp filter differently, trading sharpness for noise.
"""

# ╔═╡ 07000001-0000-4000-8000-000000000001
md"""
## 6. Forward Project (Standard Dose)

Time to actually scan the phantom. Each "phase" of GPU work — forward
projection, reconstruction — lives inside its own `let ... end` block. The
shape of every cell that touches the device is the same:

```julia
output_cpu = let
    # 1. Allocate GPU workspace + run the kernel
    ws = BS.create_…_workspace(…)
    BS.simulate!(ws, …)              # or BS.reconstruct!(…)

    # 2. Copy results off the device (CPU `Array`)
    result = Array(ws.sinogram)

    # 3. Explicit GPU cleanup
    ws = nothing
    GC.gc(true)

    result
end
```

!!! warning "Why `let` + `GC.gc(true)` matters"
    Each EICT workspace holds spectrum-binned forward buffers, scatter
    kernels, bowtie air references, and a full sinogram on the device —
    easily several hundred MB per protocol. On a 16 GB unified-memory
    M-series Mac, running two protocols back-to-back **without dropping the
    first workspace's GPU buffers will OOM the Pluto worker** (you'll see
    `Malt.TerminatedWorkerException`).

    The `let ... end` scopes every GPU binding to the block; setting
    `ws = nothing` drops the last reference; `GC.gc(true)` forces Julia to
    actually release the device memory (a regular `gc()` won't run a full
    sweep). The outer cell only holds the CPU output — tiny by comparison.

    This is also why we never put intermediate diagnostic GPU arrays in the
    returned NamedTuple — only `Array(...)`-copied CPU data leaves the block.
"""

# ╔═╡ 07000010-0000-4000-8000-000000000001
sim_std = let
    @info "Simulating: 120 kVp / 200 mA (standard dose)…"
    ws = BS.create_eict_workspace(scanner, protocol_standard, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, protocol_standard, sim_opts)

    # Copy off GPU before tearing down the workspace
    result = (sino = Array(ws.sinogram), geom = ws.geom)

    # Drop refs + force a real GC so device memory comes back
    ws = nothing
    GC.gc(true)

    result
end;

# ╔═╡ 07000015-0000-4000-8000-000000000001
let
    sino = sim_std.sino                  # (n_col, n_row, n_view), already −log(I/I₀)
    n_col, n_row, n_view = size(sino)
    mid_row = n_row ÷ 2 + 1

    fig = CM.Figure(size = (1100, 660))
    ax = CM.Axis(
        fig[1, 1];
        title = "Standard-dose sinogram",
        subtitle = "Central detector row · 120 kVp / 200 mA · $n_view views",
        xlabel = "View",
        ylabel = "Detector column",
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )
    hm = CM.heatmap!(
        ax, 1:n_view, 1:n_col, permutedims(sino[:, mid_row, :]);
        colormap = :viridis,
    )
    CM.Colorbar(fig[1, 2], hm; label = "Line integral  −log(I / I₀)", width = 14, labelsize = 18)

    CM.save(
        joinpath(@__DIR__, "..", "assets", "sinogram_standard.png"),
        fig; px_per_unit = 2
    )
    fig
end

# ╔═╡ 07000020-0000-4000-8000-000000000001
md"""
## 7. Reconstruct (Standard Dose: FBP)

Same pattern: allocate an FDK workspace, run `reconstruct!`, copy the volume
off the device, drop GPU references, GC. The output is in linear attenuation
units (cm⁻¹), which we convert to Hounsfield Units using a **polychromatic
effective μ_water** for our 120 kVp spectrum + Gammex 472 body
(33 cm diameter chord through the center voxel).

[`BS.compute_polychromatic_μ_water`](@ref) resolves the bowtie-aware source
spectrum (flat filter + protocol filters + bowtie all included), pre-hardens
it through `water_path_cm` of solid water via Beer-Lambert (mimicking the
hardening a center-of-phantom ray sees), then integrates μ_water(E) against
the hardened spectrum.  The result matches what the FBP recon actually
produces for solid water — so by construction the phantom-center voxel
reads ≈ 0 HU, instead of ≈ −150 HU you'd get with a monoenergetic
`get_reference_μ_water(70.0)`.

Both protocols share the same kVp + filtration, so we compute the
reference once and reuse it across §7 and §8.
"""

# ╔═╡ 07000020-0000-4000-8000-000000000005
μ_water_recon = let
    body_radius_cm = 16.5   # Gammex 472 body
    BS.compute_polychromatic_μ_water(
        sim_opts, protocol_standard;
        scanner = scanner,
        geom = sim_std.geom,
        water_path_cm = body_radius_cm * 2,   # full chord through center voxel
    )
end;

# ╔═╡ 07000021-0000-4000-8000-000000000001
hu_std = let
    sino_gpu = to_gpu(sim_std.sino)
    matrix_size = recon_opts.matrix_size

    ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, sim_std.geom, matrix_size)
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, sim_std.geom)

    # μ → HU, copying off the GPU at the same time. The `Float32.(...)` cast
    # is intentional (not redundant): `BS.μ_to_HU` widens to Float64 due to
    # an internal `1000.0` literal, but `apply_radial_cupping_correction!`
    # (used in section 9) only has a method for `Array{Float32, 3}`.
    hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = μ_water_recon))

    # Drop every GPU ref before exiting
    ws_fdk = nothing
    sino_gpu = nothing
    recon_μ = nothing
    GC.gc(true)

    hu
end;

# ╔═╡ 08000001-0000-4000-8000-000000000001
md"""
## 8. Repeat (Low Dose)

Same two `let ... end` blocks, only `protocol_lowdose` swapped in. Everything
else (scanner, sim_opts, recon_opts, phantom) is reused unchanged — that's
the point of splitting the API into separate structs. And because each
previous block already cleaned up its GPU memory, this protocol gets a fresh
device with no leftover state.
"""

# ╔═╡ 08000010-0000-4000-8000-000000000001
sim_low = let
    @info "Simulating: 120 kVp / 50 mA (low dose)…"
    ws = BS.create_eict_workspace(scanner, protocol_lowdose, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, protocol_lowdose, sim_opts)

    result = (sino = Array(ws.sinogram), geom = ws.geom)

    ws = nothing
    GC.gc(true)

    result
end;

# ╔═╡ 08000011-0000-4000-8000-000000000001
hu_low = let
    sino_gpu = to_gpu(sim_low.sino)
    matrix_size = recon_opts.matrix_size

    ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, sim_low.geom, matrix_size)
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, sim_low.geom)

    # Same polychromatic μ_water as §7 (shared 120 kVp spectrum + Gammex body)
    hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = μ_water_recon))

    ws_fdk = nothing
    sino_gpu = nothing
    recon_μ = nothing
    GC.gc(true)

    hu
end;

# ╔═╡ 09000001-0000-4000-8000-000000000001
md"""
## 9. Postprocessing: The Full Correction Pipeline

A raw FBP recon (`hu_std`, `hu_low` above) is what comes out of the FDK
kernel with no clinical corrections. Real CT vendors apply a stack of
corrections on top — the same stack we'll build here:

| Stage | Function                              | What it does                                                                             |
|-------|---------------------------------------|------------------------------------------------------------------------------------------|
| 1.    | `calibrate_bhc_two_material`          | Fits a (water + bone) polynomial from the source spectrum once, returns a BHC model      |
| 2.    | `apply_bhc_two_material`              | Sinogram-domain beam-hardening correction — runs *before* FBP, removes the bulk poly bias |
| 3.    | `reconstruct!` (FDK)                  | Filtered back-projection on the corrected sinogram                                       |
| 4.    | `apply_bhc_image_domain`              | Image-domain BHC refinement — bone-region smoothstep + scaled error subtraction          |
| 5.    | `to_hounsfield` (with BHC μ_water)    | Convert μ → HU using the BHC model's calibrated reference (not the 70 keV NIST value)    |
| 6.    | `add_system_noise_floor!`             | Add dose-independent DAS Gaussian σ ≈ 28 HU                                              |
| 7.    | `apply_radial_cupping_correction!`    | Residual even-polynomial cup correction on solid water (mostly cosmetic after BHC)       |

!!! info "Why the cupping correction looks subtle"
    Stage 7 only catches what stages 2 and 4 missed — a small residual
    radial bias. After bowtie-aware BHC the residual cup is *supposed to be
    nearly invisible*. If the cupping correction is doing a lot, your BHC
    isn't pulling its weight.
"""

# ╔═╡ 09000005-0000-4000-8000-000000000001
md"""
#### 9a. Calibrate the BHC model

Both protocols use 120 kVp with the same filtration, so a single BHC model
covers both. The calibration is a one-time spectrum fit — input is
`(energies, weights)` from the resolved source spectrum, output is a
`TwoMaterialBHC` polynomial model + the calibrated `μ_water_ref` (which
becomes our reference for `to_hounsfield`).
"""

# ╔═╡ 09000006-0000-4000-8000-000000000001
bhc_calibration = let
    prot_for_bhc = BS.CTProtocol(kVp = 120, additional_filters = [("Al", 4.5)])

    # Bowtie-aware high-level API: auto-resolves the bowtie-hardened spectrum,
    # collapses to per-column weights, and dispatches to TwoMaterialBHCPerColumn.
    model = BS.calibrate_bhc_two_material(
        sim_opts, prot_for_bhc;
        scanner = scanner, geom = sim_std.geom,
        order = 2,
        hu_low = 450.0,   # bone-segmentation lower threshold (HU)
        hu_high = 600.0,   # bone-segmentation upper threshold (HU)
    )

    (
        model = model,
        μ_water = model.μ_water_ref,
        ref_E_keV = model.reference_energy_keV,
    )
end;

# ╔═╡ 09000007-0000-4000-8000-000000000001
md"""
**Calibrated:**

* ref energy = $(round(bhc_calibration.ref_E_keV, digits = 1)) keV

* lac water = $(round(bhc_calibration.μ_water, digits = 5)) cm⁻¹
"""

# ╔═╡ 09000010-0000-4000-8000-000000000001
md"""
#### 9b. Standard-Dose Corrected Pipeline

Same `let ... end` shape as §7, but with the full correction stack inline:
sino BHC → FDK → image BHC → HU → noise floor → cupping. GPU buffers are
dropped at the end exactly like before.
"""

# ╔═╡ 09000011-0000-4000-8000-000000000001
hu_std_corr = let
    matrix_size = recon_opts.matrix_size

    # 1. Sinogram-domain BHC (returns a CPU array)
    sino_gpu = to_gpu(sim_std.sino)
    sino_bhc = BS.apply_bhc_two_material(
        sino_gpu, bhc_calibration.model, sim_std.geom, matrix_size;
        projector = sim_opts.projector,
    )
    sino_gpu = to_gpu(sino_bhc)

    # 2. FDK on the BHC-corrected sinogram
    ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, sim_std.geom, matrix_size)
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, sim_std.geom)

    # 3. Image-domain BHC refinement (in-place on GPU)
    BS.apply_bhc_image_domain(
        recon_μ, sim_std.geom, matrix_size, bhc_calibration.μ_water;
        hu_low = 50.0,
        hu_high = 150.0,
        scale_factor = 0.2,
        projector = sim_opts.projector,
    )

    # 4. μ → HU using BHC's calibrated μ_water_ref (Float32 to feed cupping correction)
    hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc_calibration.μ_water))

    # 5. Dose-independent DAS noise floor
    BS.add_system_noise_floor!(hu, 28.0; seed = 1234)

    # 6. Residual radial cupping correction
    BS.apply_radial_cupping_correction!(hu; fov_cm = 35.0)

    # GPU cleanup
    ws_fdk = nothing; sino_gpu = nothing; recon_μ = nothing
    GC.gc(true)

    hu
end;

# ╔═╡ 09000020-0000-4000-8000-000000000001
md"""
#### 9c. Low-Dose Corrected Pipeline

Identical to 9b — same BHC model, image BHC params, noise floor, cupping
correction — only the input sinogram (`sim_low.sino`) and noise seed differ.
"""

# ╔═╡ 09000021-0000-4000-8000-000000000001
hu_low_corr = let
    matrix_size = recon_opts.matrix_size

    # 1. Sinogram-domain BHC
    sino_gpu = to_gpu(sim_low.sino)
    sino_bhc = BS.apply_bhc_two_material(
        sino_gpu, bhc_calibration.model, sim_low.geom, matrix_size;
        projector = sim_opts.projector,
    )
    sino_gpu = to_gpu(sino_bhc)

    # 2. FDK
    ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, sim_low.geom, matrix_size)
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, sim_low.geom)

    # 3. Image-domain BHC
    BS.apply_bhc_image_domain(
        recon_μ, sim_low.geom, matrix_size, bhc_calibration.μ_water;
        hu_low = 50.0,
        hu_high = 150.0,
        scale_factor = 0.2,
        projector = sim_opts.projector,
    )

    # 4. μ → HU
    hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc_calibration.μ_water))

    # 5. Noise floor (different seed so the noise pattern doesn't match the std-dose scan)
    BS.add_system_noise_floor!(hu, 28.0; seed = 5678)

    # 6. Cupping correction
    BS.apply_radial_cupping_correction!(hu; fov_cm = 35.0)

    # GPU cleanup
    ws_fdk = nothing; sino_gpu = nothing; recon_μ = nothing
    GC.gc(true)

    hu
end;

# ╔═╡ 10000001-0000-4000-8000-000000000001
md"""
## 10. Compare Protocols (Before & After Correction)

The two reconstructions use identical physics, geometry, and reconstruction
settings — only the tube current differs. The lower-dose scan should show
visibly more noise (≈√4 = 2× higher pixel σ) but the same mean HU values
inside each rod. Postprocessing flattens the radial cup and adds the system
noise floor — visible mostly in the rim region.
"""

# ╔═╡ 10000002-0000-4000-8000-000000000001
let
    fig = CM.Figure(size = (1200, 1100))

    mid = size(hu_std, 3) ÷ 2
    colorrng = (-200, 500)
    title_kwargs = (titlesize = 28, subtitlesize = 20)

    ax_std_raw = CM.Axis(
        fig[1, 1];
        title = "Uncorrected",
        subtitle = "Standard dose",
        aspect = CM.DataAspect(),
        title_kwargs...,
    )
    ax_std_corr = CM.Axis(
        fig[1, 2];
        title = "Corrected",
        subtitle = "Standard dose",
        aspect = CM.DataAspect(),
        title_kwargs...,
    )
    ax_low_raw = CM.Axis(
        fig[2, 1];
        title = "Uncorrected",
        subtitle = "Low dose",
        aspect = CM.DataAspect(),
        title_kwargs...,
    )
    ax_low_corr = CM.Axis(
        fig[2, 2];
        title = "Corrected",
        subtitle = "Low dose",
        aspect = CM.DataAspect(),
        title_kwargs...,
    )

    CM.heatmap!(ax_std_raw, hu_std[:, :, mid]; colormap = :grays, colorrange = colorrng)
    CM.heatmap!(ax_std_corr, hu_std_corr[:, :, mid]; colormap = :grays, colorrange = colorrng)
    CM.heatmap!(ax_low_raw, hu_low[:, :, mid]; colormap = :grays, colorrange = colorrng)
    hm = CM.heatmap!(ax_low_corr, hu_low_corr[:, :, mid]; colormap = :grays, colorrange = colorrng)

    for ax in (ax_std_raw, ax_std_corr, ax_low_raw, ax_low_corr)
        CM.hidedecorations!(ax)
    end

    CM.Colorbar(fig[1:2, 3], hm; label = "HU", width = 14, labelsize = 18)

    CM.save(
        joinpath(@__DIR__, "..", "assets", "recon_compare_4panel.png"),
        fig; px_per_unit = 2
    )
    fig
end

# ╔═╡ 10000003-0000-4000-8000-000000000001
let
    # Pixel-σ in a homogeneous solid-water region (rough noise estimate)
    n = size(hu_std, 1)
    roi = ((n ÷ 2 - 30):(n ÷ 2 + 30), (n ÷ 2 - 30):(n ÷ 2 + 30), :)

    σ_std_raw = std(hu_std[roi...])
    σ_std_corr = std(hu_std_corr[roi...])
    σ_low_raw = std(hu_low[roi...])
    σ_low_corr = std(hu_low_corr[roi...])

    md"""
    **Noise (σ in HU, 60×60 ROI at center of solid water):**

    | Protocol      | mA   | σ raw                          | σ corrected                     |
    |---------------|------|--------------------------------|---------------------------------|
    | Standard dose | 200  | $(round(σ_std_raw, digits=1))  | $(round(σ_std_corr, digits=1))  |
    | Low dose      | 50   | $(round(σ_low_raw, digits=1))  | $(round(σ_low_corr, digits=1))  |

    Raw σ ratio low/std = $(round(σ_low_raw  / σ_std_raw,  digits=2))×
    (theory: 2.0× for 4× lower mA).
    """
end

# ╔═╡ 11000001-0000-4000-8000-000000000001
md"""
## Summary

This notebook walked the entire `BasisSimulator.jl` user surface end to end:

- **Five structs** — `Phantom`, `Scanner`, `CTProtocol`, `SimOptions`,
  `ReconOptions` — each owning a clearly bounded slice of the simulation
  contract (object, hardware, acquisition, physics, output).
- **Two `let ... end` blocks per scan** — one for `simulate!`, one for
  `reconstruct!` — each ending with `ws = nothing; GC.gc(true)` so GPU
  buffers actually come back. The Pluto worker stays comfortably under
  memory pressure even with two protocols back-to-back.
- **A clinical-grade postprocessing pipeline** — `calibrate_bhc_two_material`
  → `apply_bhc_two_material` (sinogram BHC) → FDK → `apply_bhc_image_domain`
  → `to_hounsfield` (with the BHC model's own `μ_water_ref`) →
  `add_system_noise_floor!` → `apply_radial_cupping_correction!`.
- **Backend-agnostic GPU dispatch** — `to_gpu()` resolves at startup via
  `Base.locate_package` + `Base.require`, so the same notebook runs on
  Metal, CUDA, ROCm, or pure CPU with no project-level changes.

Every reconstruction in the rest of the docs (PCCT, Hybrid IR, dual-kVp VMI,
…) reuses this same pattern — workspace in a `let`, kernel call, copy CPU
result, `nothing` + `GC.gc(true)`, return.
"""

# ╔═╡ Cell order:
# ╟─01000001-0000-4000-8000-000000000001
# ╟─01000002-0000-4000-8000-000000000001
# ╠═01000003-0000-4000-8000-000000000001
# ╠═01000003-0000-4000-8000-000000000005
# ╠═01000003-0000-4000-8000-000000000002
# ╠═01000003-0000-4000-8000-000000000003
# ╠═01000003-0000-4000-8000-000000000004
# ╟─01000004-0000-4000-8000-000000000001
# ╠═01000005-0000-4000-8000-000000000001
# ╟─01000007-0000-4000-8000-000000000001
# ╟─02000001-0000-4000-8000-000000000001
# ╠═02000002-0000-4000-8000-000000000001
# ╟─02000003-0000-4000-8000-000000000001
# ╠═02000004-0000-4000-8000-000000000001
# ╟─02000005-0000-4000-8000-000000000001
# ╟─03000001-0000-4000-8000-000000000001
# ╠═03000002-0000-4000-8000-000000000001
# ╟─03000003-0000-4000-8000-000000000001
# ╟─04000001-0000-4000-8000-000000000001
# ╠═04000002-0000-4000-8000-000000000001
# ╠═04000003-0000-4000-8000-000000000001
# ╟─04000004-0000-4000-8000-000000000001
# ╟─05000001-0000-4000-8000-000000000001
# ╠═05000002-0000-4000-8000-000000000001
# ╟─05000003-0000-4000-8000-000000000001
# ╟─06000001-0000-4000-8000-000000000001
# ╠═06000002-0000-4000-8000-000000000001
# ╟─06000003-0000-4000-8000-000000000001
# ╟─07000001-0000-4000-8000-000000000001
# ╠═07000010-0000-4000-8000-000000000001
# ╟─07000015-0000-4000-8000-000000000001
# ╟─07000020-0000-4000-8000-000000000001
# ╠═07000020-0000-4000-8000-000000000005
# ╠═07000021-0000-4000-8000-000000000001
# ╟─08000001-0000-4000-8000-000000000001
# ╠═08000010-0000-4000-8000-000000000001
# ╠═08000011-0000-4000-8000-000000000001
# ╟─09000001-0000-4000-8000-000000000001
# ╟─09000005-0000-4000-8000-000000000001
# ╠═09000006-0000-4000-8000-000000000001
# ╟─09000007-0000-4000-8000-000000000001
# ╟─09000010-0000-4000-8000-000000000001
# ╠═09000011-0000-4000-8000-000000000001
# ╟─09000020-0000-4000-8000-000000000001
# ╠═09000021-0000-4000-8000-000000000001
# ╟─10000001-0000-4000-8000-000000000001
# ╟─10000002-0000-4000-8000-000000000001
# ╟─10000003-0000-4000-8000-000000000001
# ╟─11000001-0000-4000-8000-000000000001
