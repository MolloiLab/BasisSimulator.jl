### A Pluto.jl notebook ###
# v0.7.0

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

# ╔═╡ 01000003-0000-4000-8000-000000000001
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 12000001-0000-4000-8000-000000000002
using PlutoUI

# ╔═╡ 01000003-0000-4000-8000-000000000005
using Markdown: @md_str

# ╔═╡ 01000003-0000-4000-8000-000000000004
using Statistics: mean, std

# ╔═╡ 01000001-0000-4000-8000-000000000001
md"""
# The Five-Struct API

**Describe a scan with five structs, simulate it, reconstruct it, and read off its dose.**

Every simulation in `BasisSimulator.jl` is specified by five plain Julia structs, one for each
thing a CT scan needs:

| Struct | What it describes | Built here with |
|:--|:--|:--|
| `Phantom` | the object: a labeled voxel mask, one `XrayAttenuation.Material` per label, the voxel size | `create_gammex_472` |
| `EICTScanner` / `PCCTScanner` | the hardware: geometry, detector, focal spot, filtration, bowtie | `EICTScanner(; ...)` |
| `CTProtocol` | the acquisition: kVp, mA, views, rotation time, collimation, pitch | `CTProtocol(; ...)` |
| `SimOptions` | the physics: which effects the forward model includes, the noise seed, the projector | `SimOptions(; ...)` |
| `ReconOptions` | the output grid: matrix size, in-plane FOV, z extent | `ReconOptions(; ...)` |

Three calls turn them into an image:

```
Phantom       ─┐
Scanner       ─┤
CTProtocol    ─┼─▶ create_workspace ─▶ simulate! ─▶ reconstruct! ─▶ to_hounsfield ─▶ HU
SimOptions    ─┤                           │
ReconOptions  ─┘                           └─▶ result.dose: CTDIvol, DLP
```

This notebook scans a **Gammex 472** calibration phantom on a model of the **GE Revolution Apex
Elite** twice, at 200 mA and at 50 mA. It reports the dose of each scan, reconstructs both with
filtered back-projection (FBP) and with Hybrid IR, and checks the result against
physics-derived expectations: water at 0 HU, noise scaling with dose, and every calcium and
iodine rod against its theoretical HU.
"""

# ╔═╡ 01000002-0000-4000-8000-000000000001
md"""
## Notebook setup

The notebook activates the shared `docs/` environment (which pulls `BasisSimulator` from this
source tree) and loads CairoMakie for the figures.
"""

# ╔═╡ 01000003-0000-4000-8000-000000000002
# ╠═╡ show_logs = false
import BasisSimulator as BS

# ╔═╡ 01000003-0000-4000-8000-000000000003
# ╠═╡ show_logs = false
import CairoMakie as Mke

# ╔═╡ 886270ac-b0c1-4c77-b218-3bb67c8bee20
TableOfContents()

# ╔═╡ 01000004-0000-4000-8000-000000000001
md"""
#### Choosing the device

`BasisSimulator` writes its kernels once, with
[AcceleratedKernels.jl](https://github.com/JuliaGPU/AcceleratedKernels.jl), and runs them on
whatever array type it is handed. `GPUSelect.Storage()` returns the array type of the GPU it
finds (`CuArray`, `MtlArray`, `ROCArray` or `oneArray`), or `Array` when there is none. Moving the
phantom mask to that type is the only thing that decides where the whole pipeline runs.
"""

# ╔═╡ 01000005-0000-4000-8000-000000000001
begin
    import GPUSelect
    AT = GPUSelect.Storage()   # CuArray / MtlArray / ROCArray / oneArray, or Array on a CPU-only host
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end;

# ╔═╡ 01000007-0000-4000-8000-000000000001
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 02000000-0000-4000-8000-000000000000
md"""
## The five structs
"""

# ╔═╡ 02000001-0000-4000-8000-000000000001
md"""
### 1. `Phantom`

A `Phantom` holds three things:

1. a **labeled mask**: a 3D array of unsigned integers, one region label per voxel;
2. a **material per label**: an `XrayAttenuation.Material`, which gives the simulator μ(E) at every
   energy of the spectrum;
3. the **voxel size** `(dx, dy, dz)` in cm. The phantom is centred on the isocentre unless you pass
   an origin.

`create_gammex_472` builds the Gammex Model 472 multi-energy phantom: a 33 cm solid-water body
with seven calcium rods (50–600 mg/mL) on an inner ring and seven iodine rods (2–20 mg/mL) on an
outer ring, each 28 mm across.
"""

# ╔═╡ 02000002-0000-4000-8000-000000000001
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,   # in-plane voxels across the 35 cm field
    n_slices = 16,    # axial slices
    fov_cm = 35.0,
    z_cm = 1.0,
);

# ╔═╡ 02000003-0000-4000-8000-000000000001
Markdown.parse("""
The mask is `$(eltype(phantom_cpu.mask))` of size $(join(size(phantom_cpu.mask), " × ")), with
voxels of $(join(round.(phantom_cpu.voxel_size .* 10; digits = 3), " × ")) mm and
$(length(phantom_cpu.materials)) entries in its material table (entry `L + 1` is the material of
label `L`; labels the phantom doesn't use are air).

`Phantom` is parameterised on its mask's array type, so moving the mask to the device once is
enough: every forward projection of this phantom then runs there.
""")

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
    # Region name + colour for each Gammex label (the `BS.RegionLabel` values)
    info = Dict(
        UInt8(0) => ("Air (outside)", :gray15),
        UInt8(1) => ("Air", :gray30),
        UInt8(2) => ("Pure water", :royalblue),
        UInt8(3) => ("Solid water", :lightskyblue),
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

    # Remap the labels present to 1..N for a categorical colormap
    labels = sort(unique(slice))
    n = length(labels)
    lut = Dict(l => i for (i, l) in enumerate(labels))
    mapped = Float32[lut[l] for l in slice]
    cmap = Mke.cgrad([info[l][2] for l in labels], n; categorical = true)

    fig = Mke.Figure(size = (900, 600))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Input phantom",
        subtitle = "Gammex Model 472 · slice $mid of $(size(phantom_cpu.mask, 3))",
        aspect = Mke.DataAspect(),
        titlesize = 28, subtitlesize = 20,
    )
    Mke.heatmap!(ax, mapped; colormap = cmap, colorrange = (0.5, n + 0.5))
    Mke.hidedecorations!(ax)
    Mke.Colorbar(
        fig[1, 2];
        colormap = cmap, colorrange = (0.5, n + 0.5),
        ticks = (1:n, [info[l][1] for l in labels]),
        ticklabelsize = 12, width = 16,
    )
    Mke.save(joinpath(@__DIR__, "..", "assets", "gammex_472_phantom.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 03000001-0000-4000-8000-000000000001
md"""
### 2. The scanner: `EICTScanner`

The scanner is the **hardware**: everything that stays the same from one scan to the next.
There are two families, one per detector type, and they share a `ScannerGeometry` (source and
detector distances, detector array, focal spot, gantry, flat filter, bowtie):

- `EICTScanner`: an energy-integrating scintillator detector (material, depth, fill factors,
  detection gain, electronic noise);
- `PCCTScanner`: a photon-counting detector (sensor, energy thresholds, energy resolution,
  charge sharing, dead time and pile-up, pixel binning).

Each constructor rejects the other family's keywords, so a photon-counting parameter can't end
up on a scintillator by accident. Geometry fields read the same on both: `scanner.detector_rows`.

The values below model the GE Revolution Apex Elite: 256 rows of 0.625 mm, 834 columns, a curved
(arc) Lumex detector and GE's large-body bowtie. Distances are in mm; the detector pitch is given
at the isocentre.
"""

# ╔═╡ 03000002-0000-4000-8000-000000000001
scanner = BS.EICTScanner(
    # Geometry (mm)
    source_to_isocenter = 625.6,
    source_to_detector = 1100.0,

    # Detector array (pitch at the isocentre, mm)
    detector_rows = 256,
    detector_cols = 834,
    detector_row_size = 0.625,
    detector_col_size = 0.6,

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

    # Data-acquisition electronics: added to the counts before the log, as a real DAS does
    electronic_noise = 3500.0,   # e⁻ rms
    detection_gain = 10.0,       # e⁻ / keV
);

# ╔═╡ 03000003-0000-4000-8000-000000000001
Markdown.parse("""
Scanner: $(scanner.detector_rows) rows × $(scanner.detector_cols) columns, detector shape
`:$(scanner.detector_shape)`, bowtie `:$(scanner.bowtie_filter)`, source–isocentre
$(scanner.source_to_isocenter) mm.

!!! info "Hardware here, acquisition in the protocol"
    Anything that changes from scan to scan (kVp, mA, views, rotation time, collimation, pitch)
    belongs to `CTProtocol`. That split is what lets one scanner run a sweep of protocols, as the
    two scans below do.
""")

# ╔═╡ 04000001-0000-4000-8000-000000000001
md"""
### 3. `CTProtocol`

The **acquisition**. Two protocols share everything except the tube current: a standard-dose scan
at 200 mA and a low-dose scan at 50 mA, both 120 kVp, 500 views in a 1 s rotation, 5 mm of
collimation and 4.5 mm of extra aluminium filtration.
"""

# ╔═╡ 04000002-0000-4000-8000-000000000001
protocol_standard = BS.CTProtocol(
    kVp = 120,
    mA = 200.0,
    views = 500,
    rotation_time = 1.0,
    collimation_mm = 5.0,                 # nominal beam width N·T
    additional_filters = [("Al", 4.5)],   # on top of the scanner's 2.5 mm flat filter
);

# ╔═╡ 04000003-0000-4000-8000-000000000001
protocol_lowdose = BS.CTProtocol(
    kVp = 120,
    mA = 50.0,                            # a quarter of the dose
    views = 500,
    rotation_time = 1.0,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 04000004-0000-4000-8000-000000000001
md"""
!!! info "Collimation selects the detector rows"
    The detector is 256 × 0.625 mm = 160 mm wide; `collimation_mm = 5.0` uses only the central
    rows. The simulator adds symmetric guard rows automatically where the cone beam needs them to
    cover the whole reconstruction cylinder at the edge of the field. Leave `collimation_mm`
    unset to use the full detector; add `pitch` and `n_rotations` for a helical scan
    (notebook 11).
"""

# ╔═╡ 05000001-0000-4000-8000-000000000001
md"""
### 4. `SimOptions`

The **physics** of the forward model. Every effect is on by default except optical crosstalk:
fill factor, energy-dependent detector efficiency, scatter, focal-spot blur, quantum and
electronic noise, detector lag and the anode heel effect. The polychromatic source spectrum,
the bowtie and beam hardening are always part of the model. Turn an effect off to measure its
contribution.

`projector = :dd_fast` (the default) is the anti-aliased distance-driven projector that walks the
volume once for the whole spectrum; `:siddon` is the point-sampled ray tracer, kept for
comparison.

**The detector integrates while the gantry turns.** A clinical detector reads each view for the
whole view period, during which the gantry sweeps the view spacing Δθ = 360° / views. Each
reading is therefore the transmitted intensity averaged over that arc: the object is blurred
along the direction of rotation, more the farther it sits from the isocentre, while the noise,
counted once per view, stays independent from view to view. `view_samples` samples that average
at sub-views spread across the arc (the midpoint rule, in the intensity domain, before scatter,
noise and the detector effects); `view_arc` is the fraction of Δθ the detector integrates over,
1.0 for a detector that reads for the whole view period and the duty cycle of one energy for
rapid kVp switching. The default, `view_samples = 1`, is an instantaneous point view. Every
clinical scanner in these notebooks is modelled with `view_samples = 5`, which costs five forward
projections per view.
"""

# ╔═╡ 05000002-0000-4000-8000-000000000001
sim_opts = BS.SimOptions(
    seed = 1234,            # fixed noise realisation: same seed + same inputs = same sinogram
    projector = :dd_fast,
    view_samples = 5,       # the detector integrates over the arc of each view (5 sub-views)

    # Switch individual effects off, e.g.:
    # use_scatter     = false,
    # use_heel_effect = false,
    # use_focal_spot  = false,
    # use_lag         = false,
    # use_noise       = false,   # the noiseless expected sinogram
);

# ╔═╡ 05000003-0000-4000-8000-000000000001
md"""
!!! info "Detector-specific physics lives on the scanner"
    `SimOptions` carries only the effects both detector families share. Pile-up, its correction,
    photon-counting scatter correction and the count-noise blend are fields of `PCCTScanner`;
    detector lag applies to the scintillator only.
"""

# ╔═╡ 06000001-0000-4000-8000-000000000001
md"""
### 5. `ReconOptions`

The **output grid**: matrix size, in-plane field of view and z extent, centred on the isocentre.
Here the 5 mm beam is reconstructed as eight 0.625 mm slices on a 512 × 512 grid over 35 cm
(0.68 mm pixels). The algorithm and its filter are chosen where they are used, when the
reconstruction workspace is created.
"""

# ╔═╡ 06000002-0000-4000-8000-000000000001
recon_opts = let
    slice_thickness_mm = 0.625
    n_slices = round(Int, protocol_standard.collimation_mm / slice_thickness_mm)
    BS.ReconOptions(
        matrix_size = (512, 512, n_slices),
        fov_cm = 35.0,
        z_cm = protocol_standard.collimation_mm / 10,
    )
end;

# ╔═╡ 07000000-0000-4000-8000-000000000000
md"""
## Simulate
"""

# ╔═╡ 07000001-0000-4000-8000-000000000001
md"""
### Workspace, then `simulate!`

`create_workspace` does all the set-up once: it builds the scan geometry, resolves the
source spectrum through the filters and bowtie, and allocates every device buffer on the
phantom's backend. `simulate!` then runs the forward model into `ws.sinogram`, the
log-transformed line integrals `-log(I/I₀)` of size (columns, rows, views), noise included. The
same constructor serves both detector families: for a `PCCTScanner` it builds the
photon-counting workspace, and `simulate!` returns one sinogram per energy bin.

`simulate!` returns the **dose** of the acquisition it simulated. The CTDI is computed the way
IEC 60601-2-44 defines it: the simulated beam is transported by Monte Carlo through the 32 cm
PMMA body phantom, and the air kerma is integrated over a 100 mm pencil chamber at the centre and
at the periphery. It therefore follows this scanner's spectrum, filtration and bowtie rather than
a lookup table.

Each scan runs inside a `let` block that keeps only CPU copies of what it needs, so the device
buffers are released before the next scan starts.
"""

# ╔═╡ 07000010-0000-4000-8000-000000000001
sim_std = let
    ws = BS.create_workspace(scanner, protocol_standard, sim_opts, recon_opts, phantom)
    result = BS.simulate!(ws, phantom, protocol_standard, sim_opts)

    out = (sino = Array(ws.sinogram), geom = ws.geom, dose = result.dose)
    ws = nothing
    GC.gc(true)   # hand the device memory back now
    out
end;

# ╔═╡ 07000011-0000-4000-8000-000000000001
sim_std.dose

# ╔═╡ 07000015-0000-4000-8000-000000000001
let
    sino = sim_std.sino                  # (n_col, n_row, n_view), −log(I/I₀)
    n_col, n_row, n_view = size(sino)
    mid_row = n_row ÷ 2 + 1

    fig = Mke.Figure(size = (1100, 660))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Standard-dose sinogram",
        subtitle = "central detector row · 120 kVp / 200 mA · $n_view views · $n_col columns",
        xlabel = "view",
        ylabel = "detector column",
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )
    hm = Mke.heatmap!(ax, 1:n_view, 1:n_col, permutedims(sino[:, mid_row, :]); colormap = :viridis)
    Mke.Colorbar(fig[1, 2], hm; label = "line integral  −log(I / I₀)", width = 14, labelsize = 18)
    Mke.save(joinpath(@__DIR__, "..", "assets", "sinogram_standard.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 08000001-0000-4000-8000-000000000001
md"""
### The low-dose scan

The same two calls with `protocol_lowdose`. The phantom, scanner, physics and grid are reused
unchanged, which is the point of keeping them in separate structs.
"""

# ╔═╡ 08000010-0000-4000-8000-000000000001
sim_low = let
    ws = BS.create_workspace(scanner, protocol_lowdose, sim_opts, recon_opts, phantom)
    result = BS.simulate!(ws, phantom, protocol_lowdose, sim_opts)

    out = (sino = Array(ws.sinogram), geom = ws.geom, dose = result.dose)
    ws = nothing
    GC.gc(true)
    out
end;

# ╔═╡ 08000011-0000-4000-8000-000000000001
sim_low.dose

# ╔═╡ 08000012-0000-4000-8000-000000000001
let
    d_s, d_l = sim_std.dose, sim_low.dose
    Markdown.parse("""
    | protocol | mAs / rotation | CTDIvol (mGy) | DLP (mGy·cm) | scan length (cm) |
    |:--|--:|--:|--:|--:|
    | standard dose | $(round(d_s.mAs_per_rotation; digits = 1)) | $(round(d_s.ctdi_vol_mGy; digits = 2)) | $(round(d_s.dlp_mGy_cm; digits = 2)) | $(round(d_s.scan_length_cm; digits = 2)) |
    | low dose | $(round(d_l.mAs_per_rotation; digits = 1)) | $(round(d_l.ctdi_vol_mGy; digits = 2)) | $(round(d_l.dlp_mGy_cm; digits = 2)) | $(round(d_l.scan_length_cm; digits = 2)) |

    The beam is identical, so CTDIvol scales with the tube current alone: the low-dose scan
    delivers $(round(d_l.ctdi_vol_mGy / d_s.ctdi_vol_mGy; digits = 3)) of the standard dose.
    CTDIw of this beam is $(round(d_s.ctdi_w_mGy_per_100mAs; digits = 2)) mGy per 100 mAs. The
    Monte Carlo's statistical uncertainty is ±$(round(100 * d_s.rel_stat_uncertainty; digits = 2)) %.
    Pass `dose_kwargs = (; phantom = :head16)` to `simulate!` for the 16 cm head phantom.
    """)
end

# ╔═╡ 08000020-0000-4000-8000-000000000001
md"""
### Repeated noise draws: `keep_projection` and `projection`

Everything before the noise (the view-integrated forward projection and the physics that
precedes the noise, scatter included) is the same for every noise draw of one phantom, protocol
and workspace. `simulate!(...; keep_projection = true)` returns it as `result.projection`, and
`simulate!(...; projection)` draws a new realization from it, with the seed of the `SimOptions` it
is given, without projecting again. A noise study over many seeds, or a noise-free reference
(`use_noise = false`) that shares the noisy scan's projection, pays for the projection once.
"""

# ╔═╡ 08000021-0000-4000-8000-000000000001
noise_draws = let
    ws = BS.create_workspace(scanner, protocol_standard, sim_opts, recon_opts, phantom)
    # one full simulation that keeps everything before the noise …
    t_full = @elapsed (kept = BS.simulate!(ws, phantom, protocol_standard, sim_opts;
        report_dose = false, keep_projection = true))
    # … and a second noise realization drawn from it with another seed
    opts_2 = BS.SimOptions(; seed = 5678, projector = sim_opts.projector,
        view_samples = sim_opts.view_samples)
    BS.simulate!(ws, phantom, protocol_standard, opts_2; report_dose = false, projection = kept.projection)
    sino_reused = Array(ws.sinogram)
    # the same seed simulated from scratch, for comparison
    BS.simulate!(ws, phantom, protocol_standard, opts_2; report_dose = false)
    identical = Array(ws.sinogram) == sino_reused
    t_reuse = @elapsed BS.simulate!(ws, phantom, protocol_standard, opts_2;
        report_dose = false, projection = kept.projection)
    ws = nothing; kept = nothing
    GC.gc(true)
    (t_full = t_full, t_reuse = t_reuse, identical = identical)
end;

# ╔═╡ 08000022-0000-4000-8000-000000000001
Markdown.parse("""
A full simulation of the standard scan took $(round(noise_draws.t_full; digits = 2)) s; a noise
draw from its kept projection took $(round(noise_draws.t_reuse; digits = 2)) s,
$(round(noise_draws.t_full / noise_draws.t_reuse; digits = 1))× less. The draw from the
projection $(noise_draws.identical ? "is bit-identical to" : "differs from") simulating the same
seed from scratch.
""")

# ╔═╡ 07000000-0000-4000-8000-000000000001
md"""
## Reconstruct
"""

# ╔═╡ 07000020-0000-4000-8000-000000000001
md"""
### Water beam-hardening correction

The simulated sinogram is polychromatic: the spectrum hardens as it crosses the patient, so the
line integrals grow more slowly than the path length, and an uncorrected reconstruction reads
low in the middle of the body (cupping). `calibrate_bhc_water` removes that for water. For every
detector column it resolves the full detected spectrum (tube, filters, bowtie, heel effect and
detector efficiency: exactly the factors `sim_opts` switched on) and fits the polynomial that
maps polychromatic water line integrals to monochromatic ones at the spectrum's mean energy. It
has no tunable parameters, and nothing in it is fitted to the data being corrected.
`apply_bhc_water` applies it to a sinogram.

The reference energy and the water attenuation μ at that energy, which come with the correction, are what `to_hounsfield` needs to
put water at 0 HU. Both protocols share a beam, so one calibration serves both.
"""

# ╔═╡ 09000006-0000-4000-8000-000000000001
bhc = BS.calibrate_bhc_water(sim_opts, protocol_standard; scanner, geom = sim_std.geom);

# ╔═╡ 09000007-0000-4000-8000-000000000001
Markdown.parse("""
**Water BHC:** $(length(bhc.water_bhc_per_col)) per-column polynomials · reference energy
$(round(bhc.reference_energy_keV; digits = 1)) keV · water μ = $(round(bhc.μ_water_ref; digits = 5)) cm⁻¹
""")

# ╔═╡ 09000001-0000-4000-8000-000000000001
md"""
### FBP and Hybrid IR through one `reconstruct!`

`reconstruct!` dispatches on the reconstruction workspace, so the two algorithms differ only in
the workspace you create:

- `create_fdk_recon_workspace`: filtered back-projection (Feldkamp–Davis–Kress for an axial scan,
  weighted FBP for a helical one) with the `:standard` kernel by default;
- `create_hir_recon_workspace(...; strength)`: **Hybrid IR**, an FBP start followed by
  ordered-subsets penalised weighted least squares with an edge-preserving Huber prior.
  `strength` is a percentage in steps of 10, read like the vendor dials (GE ASIR-V, Siemens
  SAFIRE): `0` is plain FBP, `60` the standard clinical setting. HIR's system matrix uses the
  same projector as the simulation (`projector = sim_opts.projector`), so it inverts the
  operator that made the data.

The function below is the whole chain, water BHC → reconstruction → HU, and returns a CPU volume.
"""

# ╔═╡ 09000002-0000-4000-8000-000000000001
"""
    reconstruct_hu(sim, bhc; algorithm = :fbp, strength = 60) -> Array{Float32, 3}

Water BHC, then FBP (`algorithm = :fbp`) or Hybrid IR (`algorithm = :hir`), then HU.
"""
function reconstruct_hu(sim, bhc; algorithm::Symbol = :fbp, strength::Integer = 60)
    sino = BS.apply_bhc_water(to_gpu(sim.sino), bhc)
    ws = if algorithm === :fbp
        BS.create_fdk_recon_workspace(sino, sim.geom, recon_opts.matrix_size)
    elseif algorithm === :hir
        BS.create_hir_recon_workspace(sino, sim.geom, recon_opts.matrix_size;
            strength, projector = sim_opts.projector)
    else
        throw(ArgumentError("algorithm must be :fbp or :hir"))
    end
    μ = BS.reconstruct!(ws, sino, sim.geom)
    hu = Float32.(BS.to_hounsfield(Array(μ); μ_water = bhc.μ_water_ref))
    ws = nothing; sino = nothing; μ = nothing
    GC.gc(true)
    return hu
end;

# ╔═╡ 07000021-0000-4000-8000-000000000001
hu_fbp_std = reconstruct_hu(sim_std, bhc; algorithm = :fbp);

# ╔═╡ 09000011-0000-4000-8000-000000000001
hu_hir_std = reconstruct_hu(sim_std, bhc; algorithm = :hir, strength = 60);

# ╔═╡ 08000013-0000-4000-8000-000000000001
hu_fbp_low = reconstruct_hu(sim_low, bhc; algorithm = :fbp);

# ╔═╡ 09000021-0000-4000-8000-000000000001
hu_hir_low = reconstruct_hu(sim_low, bhc; algorithm = :hir, strength = 60);

# ╔═╡ 10000000-0000-4000-8000-000000000000
md"""
## Results
"""

# ╔═╡ 10000004-0000-4000-8000-000000000001
md"""
### Where the noise is measured

Noise is only meaningful inside one material. The phantom labels are carried onto the
reconstruction grid with `resample_to_recon`, and the solid-water label is eroded so that only
voxels surrounded by water remain: the rods and their partial-volume edges are excluded. All
statistics below use the central slice; the edge slices of a cone-beam reconstruction see
fewer complete rays.
"""

# ╔═╡ 10000005-0000-4000-8000-000000000001
water_roi = let
    labels = BS.resample_to_recon(phantom_cpu, sim_std.geom, recon_opts.matrix_size; method = :nearest)
    nx, ny, nz = size(labels)
    kmid = (nz + 1) ÷ 2
    water = UInt8(BS.REGION_SOLID_WATER)
    r = 3                                     # erosion radius, voxels
    m = falses(nx, ny)
    for j in (1 + r):(ny - r), i in (1 + r):(nx - r)
        m[i, j] = all(labels[i + di, j + dj, kmid] == water for dj in -r:r, di in -r:r)
    end
    (labels = labels, kmid = kmid, mask = m)
end;

# ╔═╡ 10000006-0000-4000-8000-000000000001
noise = let
    k, m = water_roi.kmid, water_roi.mask
    stat(v) = (mean = Float64(mean(v[:, :, k][m])), σ = Float64(std(v[:, :, k][m])))
    (fbp_std = stat(hu_fbp_std), hir_std = stat(hu_hir_std),
     fbp_low = stat(hu_fbp_low), hir_low = stat(hu_hir_low))
end;

# ╔═╡ 10000001-0000-4000-8000-000000000001
md"""
### FBP vs Hybrid IR at two doses

The four reconstructions share physics, geometry and grid. Down the columns, only the tube
current changes; across the rows, only the reconstruction. Quartering the dose should double
the FBP noise (noise ∝ 1/√mAs) and leave every mean HU where it was. Hybrid IR at strength 60
should lower the noise at both doses without moving the means.
"""

# ╔═╡ 10000002-0000-4000-8000-000000000001
let
    k = water_roi.kmid
    colorrng = (-200, 500)
    fig = Mke.Figure(size = (1100, 1000))
    panels = [
        (1, 1, hu_fbp_std, "FBP", "200 mA · σ = $(round(noise.fbp_std.σ; digits = 1)) HU"),
        (1, 2, hu_hir_std, "Hybrid IR (strength 60)", "200 mA · σ = $(round(noise.hir_std.σ; digits = 1)) HU"),
        (2, 1, hu_fbp_low, "FBP", "50 mA · σ = $(round(noise.fbp_low.σ; digits = 1)) HU"),
        (2, 2, hu_hir_low, "Hybrid IR (strength 60)", "50 mA · σ = $(round(noise.hir_low.σ; digits = 1)) HU"),
    ]
    local hm
    for (r, c, vol, title, sub) in panels
        ax = Mke.Axis(fig[r, c]; title, subtitle = sub, aspect = Mke.DataAspect(),
            titlesize = 26, subtitlesize = 19)
        hm = Mke.heatmap!(ax, vol[:, :, k]; colormap = :grays, colorrange = colorrng)
        Mke.hidedecorations!(ax)
    end
    Mke.Colorbar(fig[1:2, 3], hm; label = "HU", width = 14, labelsize = 18)
    Mke.Label(fig[0, 1:2], "Gammex 472, central slice · window $(colorrng[1]) to $(colorrng[2]) HU";
        fontsize = 20, tellwidth = false)
    Mke.save(joinpath(@__DIR__, "..", "assets", "recon_compare_4panel.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 10000003-0000-4000-8000-000000000001
let
    band = BS.get_hir_params(60).target_noise_reduction
    red(a, b) = round(100 * (1 - b.σ / a.σ); digits = 1)
    Markdown.parse("""
    **Water, central slice, $(count(water_roi.mask)) voxels deep inside the solid-water body:**

    | protocol | FBP mean (HU) | FBP σ (HU) | HIR mean (HU) | HIR σ (HU) | HIR noise reduction |
    |:--|--:|--:|--:|--:|--:|
    | 200 mA | $(round(noise.fbp_std.mean; digits = 1)) | $(round(noise.fbp_std.σ; digits = 1)) | $(round(noise.hir_std.mean; digits = 1)) | $(round(noise.hir_std.σ; digits = 1)) | $(red(noise.fbp_std, noise.hir_std)) % |
    | 50 mA | $(round(noise.fbp_low.mean; digits = 1)) | $(round(noise.fbp_low.σ; digits = 1)) | $(round(noise.hir_low.mean; digits = 1)) | $(round(noise.hir_low.σ; digits = 1)) | $(red(noise.fbp_low, noise.hir_low)) % |

    FBP noise ratio, 50 mA / 200 mA = **$(round(noise.fbp_low.σ / noise.fbp_std.σ; digits = 2))×**.
    Quantum noise alone predicts 2×; the fixed electronic noise of the detector weighs more at
    low dose and pushes the ratio a little above that. The package's calibration band for HIR
    at strength 60 is $(band[1])–$(band[2]) % noise reduction.
    """)
end

# ╔═╡ 12000003-0000-4000-8000-000000000001
md"""
### Dose and noise at other tube currents

The two scans anchor a simple rule: CTDIvol is proportional to mAs (same beam, same phantom), and
quantum-limited FBP noise scales as 1/√mAs. The table applies it to this protocol at other tube
currents. The detector's electronic noise makes the real noise somewhat higher than this at low
mA, as the measured 50 mA scan shows.
"""

# ╔═╡ 12000003-0000-4000-8000-000000000002
let
    mA0, ctdi0, σ0 = protocol_standard.mA, sim_std.dose.ctdi_vol_mGy, noise.fbp_std.σ
    rows = ["| $(mA) | $(round(ctdi0 * mA / mA0; digits = 2)) | $(round(σ0 * sqrt(mA0 / mA); digits = 1)) |"
            for mA in (25, 50, 100, 200, 300, 400)]
    Markdown.parse("""
    | tube current (mA) | CTDIvol (mGy) | expected FBP water noise (HU) |
    |--:|--:|--:|
    $(join(rows, "\n"))

    Measured at 50 mA: $(round(noise.fbp_low.σ; digits = 1)) HU.
    """)
end

# ╔═╡ 12000003-0000-4000-8000-000000000003
# this protocol's scalars — tube current (mA), CTDIvol (mGy) and FBP water noise (HU) of the
# standard scan — the inputs of the calculator below, which runs live in the browser
dose_inputs = (Float64(protocol_standard.mA), Float64(sim_std.dose.ctdi_vol_mGy), Float64(noise.fbp_std.σ));

# ╔═╡ 12000003-0000-4000-8000-000000000004
@bind tube_mA PlutoUI.Slider(10:10:500; default = 200, show_value = true)

# ╔═╡ 12000003-0000-4000-8000-000000000005
# tube current, CTDIvol (mGy), expected FBP noise (HU), noise relative to the standard scan
dose_plan = let r = tube_mA / dose_inputs[1]
    (tube_mA, round(dose_inputs[2] * r; digits = 2), round(dose_inputs[3] / sqrt(r); digits = 1),
     round(1 / sqrt(r); digits = 2))
end;

# ╔═╡ 12000003-0000-4000-8000-000000000006
md"""
**At $(dose_plan[1]) mA:** CTDIvol is $(dose_plan[2]) mGy and the expected quantum-limited FBP water noise is $(dose_plan[3]) HU, $(dose_plan[4]) times that of the 200 mA scan.
"""

# ╔═╡ 12000001-0000-4000-8000-000000000001
md"""
### Scroll through the slices

The reconstructed slab is 5 mm thick. The edge slices are reconstructed from more oblique rays
than the central ones, so this is where any z-dependence would show first.
"""

# ╔═╡ 12000001-0000-4000-8000-000000000003
@bind z_slice PlutoUI.Slider(1:size(hu_fbp_std, 3); default = size(hu_fbp_std, 3) ÷ 2, show_value = true)

# ╔═╡ 12000001-0000-4000-8000-000000000004
let
    nz = size(hu_fbp_std, 3)
    fov_z_cm = sim_std.geom.fov[3]
    dz_mm = fov_z_cm * 10 / nz
    z_mm = 10 * (-fov_z_cm / 2 + (z_slice - 0.5) * fov_z_cm / nz)

    fig = Mke.Figure(size = (1100, 560))
    Mke.Label(fig[0, 1:3],
        "slice $(z_slice) of $(nz) · z = $(round(z_mm; digits = 2)) mm · $(round(dz_mm; digits = 3)) mm slices · 200 mA";
        fontsize = 24, font = :bold, tellwidth = false)
    ax1 = Mke.Axis(fig[1, 1]; title = "FBP", titlesize = 22, aspect = Mke.DataAspect())
    hm = Mke.heatmap!(ax1, hu_fbp_std[:, :, z_slice]; colormap = :grays, colorrange = (-200, 500))
    Mke.hidedecorations!(ax1)
    ax2 = Mke.Axis(fig[1, 2]; title = "Hybrid IR (strength 60)", titlesize = 22, aspect = Mke.DataAspect())
    Mke.heatmap!(ax2, hu_hir_std[:, :, z_slice]; colormap = :grays, colorrange = (-200, 500))
    Mke.hidedecorations!(ax2)
    Mke.Colorbar(fig[1, 3], hm; label = "HU", labelsize = 20, ticklabelsize = 14)
    fig
end

# ╔═╡ 12000002-0000-4000-8000-000000000001
md"""
### Verification

Pass/fail against physics-derived expectations. The theory for each rod is its monoenergetic HU
at the BHC reference energy, computed from the same XrayAttenuation data that drove the
simulation. The tolerance is max(15 HU, 15 %): a water-only BHC maps water exactly, but a dense
calcium or iodine rod hardens the beam beyond the water curve. That residual grows with density
(−12.4 % at Ca 600 mg/mL and −14.5 % at I 20 mg/mL in the table below) and a single-energy
scan cannot resolve it; quantitative high-Z imaging is what the dual-energy and photon-counting
VMI notebooks (03, 04) are for. Each rod must also read the same at both doses and in both
reconstructions.
"""

# ╔═╡ 12000002-0000-4000-8000-000000000002
let
    labels, k = water_roi.labels, water_roi.kmid
    refE, μw = bhc.reference_energy_keV, bhc.μ_water_ref
    nx, ny, _ = size(labels)

    # in-plane-eroded mask of one label on the central slice
    function eroded(lab; r = 1)
        m = falses(nx, ny)
        for j in (1 + r):(ny - r), i in (1 + r):(nx - r)
            m[i, j] = all(labels[i + di, j + dj, k] == lab for dj in -r:r, di in -r:r)
        end
        m
    end

    checks = NamedTuple[]
    addcheck(name, val, lo, hi) = push!(checks,
        (name = name, value = round(val; digits = 2), lo = lo, hi = hi, pass = lo <= val <= hi))

    addcheck("water mean, FBP 200 mA (HU)", noise.fbp_std.mean, -6.0, 6.0)
    addcheck("water mean, HIR 200 mA (HU)", noise.hir_std.mean, -6.0, 6.0)
    addcheck("FBP noise ratio 50 / 200 mA", noise.fbp_low.σ / noise.fbp_std.σ, 1.7, 2.4)
    addcheck("HIR noise < FBP noise, 200 mA (ratio)", noise.hir_std.σ / noise.fbp_std.σ, 0.0, 0.95)
    addcheck("CTDIvol ratio 50 / 200 mA", sim_low.dose.ctdi_vol_mGy / sim_std.dose.ctdi_vol_mGy, 0.2499, 0.2501)

    # Radial non-uniformity QA (non-mutating). On a uniform water cylinder both numbers are ≈ 0
    # after the full-spectrum BHC; on the Gammex the radial fit also picks up the hardening
    # around the dense rod rings (about 12 HU), so the gate is wider than a uniform phantom
    # would need.
    cup = BS.measure_radial_cupping(hu_fbp_std; fov_cm = sim_std.geom.fov[1])
    addcheck("radial cupping QA, FBP (HU, worst slice)", cup.cup_hu, 0.0, 15.0)
    addcheck("DC offset QA, FBP (absolute HU, worst slice)", abs(cup.dc_hu), 0.0, 6.0)

    rod_rows = String[]
    n_pass = 0; n_rod = 0
    slc(v) = @view v[:, :, k]
    for lab in 1:(length(phantom_cpu.materials) - 1)   # material entry L + 1 is label L
        lab == Int(BS.REGION_SOLID_WATER) && continue
        m = eroded(UInt8(lab); r = 2)
        count(m) < 20 && continue
        mat = phantom_cpu.materials[lab + 1]
        theory = 1000 * (BS.compute_μ_at_energy(mat, refE) - μw) / μw
        fbp = mean(slc(hu_fbp_std)[m]); hir = mean(slc(hu_hir_std)[m])
        fbp_low = mean(slc(hu_fbp_low)[m])
        tol = max(15.0, 0.15 * abs(theory))
        ok = abs(fbp - theory) <= tol &&
             abs(fbp - fbp_low) <= max(10.0, 0.03 * abs(theory)) &&
             abs(hir - fbp) <= max(10.0, 0.03 * abs(theory))
        n_rod += 1; n_pass += ok
        push!(rod_rows,
            "| $(mat.name) | $(round(Int, theory)) | $(round(Int, fbp)) | " *
            "$(round(fbp - theory; digits = 0)) ($(round(100 * (fbp - theory) / max(abs(theory), 1.0); digits = 1)) %) | " *
            "$(round(Int, fbp_low)) | $(round(Int, hir)) | $(ok ? "✅" : "❌") |")
    end
    addcheck("rods passing (theory, dose- and algorithm-invariant)", n_pass, n_rod, n_rod)

    n_ok = count(c -> c.pass, checks)
    verdict = n_ok == length(checks) ? "✅ VERIFICATION PASS ($(n_ok)/$(length(checks)))" :
                                       "❌ VERIFICATION FAIL ($(n_ok)/$(length(checks)))"
    rows = join(["| $(c.name) | $(c.value) | [$(c.lo), $(c.hi)] | $(c.pass ? "✅" : "❌") |" for c in checks], "\n")
    Markdown.parse("""
    #### $(verdict)

    | check | value | expected | pass |
    |:--|--:|:--|:--|
    $(rows)

    **Per rod, central slice.** Theory is the monoenergetic HU at $(round(refE; digits = 1)) keV.

    | rod | theory | FBP 200 mA | FBP − theory | FBP 50 mA | HIR 200 mA | pass |
    |:--|--:|--:|--:|--:|--:|:--|
    $(join(rod_rows, "\n"))
    """)
end

# ╔═╡ 11000001-0000-4000-8000-000000000001
md"""
## Summary

- **Five structs describe a scan.** `Phantom` (the object), `EICTScanner` / `PCCTScanner` (the
  hardware, sharing a `ScannerGeometry`), `CTProtocol` (the acquisition), `SimOptions` (the
  physics, including the view integration of a turning gantry, `view_samples = 5`) and
  `ReconOptions` (the output grid). Changing the dose meant changing one
  `CTProtocol` and nothing else.
- **Three calls produce an image.** `create_workspace` (for either detector family) sets
  everything up once, `simulate!` runs the forward model and
  returns the dose report, and `reconstruct!` runs FBP or Hybrid IR, depending on the workspace
  it is given.
- **A noise draw can reuse the projection.** `keep_projection = true` keeps everything before
  the noise, and `projection` draws another seed from it, bit-identically to a full simulation.
- **Every scan reports its dose.** CTDIvol and DLP come from a Monte Carlo of the simulated beam
  in the IEC body phantom, so they follow the scanner's spectrum, filtration and bowtie.
- **The standard reconstruction chain** is `calibrate_bhc_water` → `apply_bhc_water` →
  `reconstruct!` → `to_hounsfield` with the correction's own water μ. Quantum and electronic
  noise are already in the simulated counts, and `measure_radial_cupping` is a QA measurement,
  not a correction.
- **Where it runs** is decided by the phantom mask's array type; this page was rendered on an
  NVIDIA RTX PRO 6000 (CUDA), and the same notebook runs on Metal, ROCm, oneAPI or the CPU.

Every other notebook reuses this pattern: build the structs, create a workspace, `simulate!`,
`reconstruct!`.
"""

# ╔═╡ Cell order:
# ╟─01000001-0000-4000-8000-000000000001
# ╟─01000002-0000-4000-8000-000000000001
# ╟─01000003-0000-4000-8000-000000000001
# ╟─12000001-0000-4000-8000-000000000002
# ╟─01000003-0000-4000-8000-000000000005
# ╟─01000003-0000-4000-8000-000000000004
# ╠═01000003-0000-4000-8000-000000000002
# ╟─01000003-0000-4000-8000-000000000003
# ╟─886270ac-b0c1-4c77-b218-3bb67c8bee20
# ╟─01000004-0000-4000-8000-000000000001
# ╠═01000005-0000-4000-8000-000000000001
# ╟─01000007-0000-4000-8000-000000000001
# ╟─02000000-0000-4000-8000-000000000000
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
# ╟─07000000-0000-4000-8000-000000000000
# ╟─07000001-0000-4000-8000-000000000001
# ╠═07000010-0000-4000-8000-000000000001
# ╠═07000011-0000-4000-8000-000000000001
# ╟─07000015-0000-4000-8000-000000000001
# ╟─08000001-0000-4000-8000-000000000001
# ╠═08000010-0000-4000-8000-000000000001
# ╠═08000011-0000-4000-8000-000000000001
# ╟─08000012-0000-4000-8000-000000000001
# ╟─08000020-0000-4000-8000-000000000001
# ╠═08000021-0000-4000-8000-000000000001
# ╟─08000022-0000-4000-8000-000000000001
# ╟─07000000-0000-4000-8000-000000000001
# ╟─07000020-0000-4000-8000-000000000001
# ╠═09000006-0000-4000-8000-000000000001
# ╟─09000007-0000-4000-8000-000000000001
# ╟─09000001-0000-4000-8000-000000000001
# ╠═09000002-0000-4000-8000-000000000001
# ╠═07000021-0000-4000-8000-000000000001
# ╠═09000011-0000-4000-8000-000000000001
# ╠═08000013-0000-4000-8000-000000000001
# ╠═09000021-0000-4000-8000-000000000001
# ╟─10000000-0000-4000-8000-000000000000
# ╟─10000004-0000-4000-8000-000000000001
# ╟─10000005-0000-4000-8000-000000000001
# ╟─10000006-0000-4000-8000-000000000001
# ╟─10000001-0000-4000-8000-000000000001
# ╟─10000002-0000-4000-8000-000000000001
# ╟─10000003-0000-4000-8000-000000000001
# ╟─12000003-0000-4000-8000-000000000001
# ╟─12000003-0000-4000-8000-000000000002
# ╟─12000003-0000-4000-8000-000000000003
# ╠═12000003-0000-4000-8000-000000000004
# ╟─12000003-0000-4000-8000-000000000005
# ╟─12000003-0000-4000-8000-000000000006
# ╟─12000001-0000-4000-8000-000000000001
# ╟─12000001-0000-4000-8000-000000000003
# ╟─12000001-0000-4000-8000-000000000004
# ╟─12000002-0000-4000-8000-000000000001
# ╟─12000002-0000-4000-8000-000000000002
# ╟─11000001-0000-4000-8000-000000000001
