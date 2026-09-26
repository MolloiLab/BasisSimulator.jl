### A Pluto.jl notebook ###
# v0.2.6

using Markdown
using InteractiveUtils

# ╔═╡ 171294a2-26bd-49e2-ac92-9df48ae5444f
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 69358294-97f2-4782-94d7-c29c747c45f4
using Markdown: @md_str, Markdown

# ╔═╡ 9ae27110-5c47-442b-a98e-d137599570f2
using Statistics: mean, std, quantile

# ╔═╡ d3054785-9e00-4094-a491-088ce63be9dc
md"""
# Dual-kVp Rapid-Switching Virtual Monoenergetic Imaging

A GE Revolution Apex Elite rapid-kVp-switching acquisition (80 + 140 kVp) of the
Gammex 472 phantom, turned into virtual monoenergetic images (VMIs) at 50, 70, 100 and
140 keV by one package call, `BS.vmi_pipeline`, and checked rod by rod against a
noise-free acquisition of the same scan and against the attenuation of the phantom's own
materials.

The scanner, the acquisition and the chain are those of the basis-spectral-denoising
paper: its GE model with its physics switches, the two exposures of a rotation with their
duty cycle, view integration, the fitted pair of FDK windows and a calibration draw that
fixes the spectral pair.
"""

# ╔═╡ f2798d62-3509-4cc4-a24f-39ace8bb5a9e
md"""
## Pipeline

```
80 kVp + 140 kVp exposures of one rotation   (duty cycle 0.65 / 0.35, each view integrated over its arc)
  → one projection per exposure, three noise draws from it:
      the noise-free reference, the measured scan and a calibration scan
  → BS.spectral_basis_from_acquisitions       (per-ray absolute response of each kVp)
  → calibration draw → pair_basis (E*, β) and composite_energy
  → BS.vmi_pipeline(; denoiser = SpectralHYPR(), fbp_filter = PairFilter(…), pair_basis, composite_energy)
       projection-domain HYPR-LR on the counts of each detector row
     → K = 2 n-channel Poisson maximum-likelihood decomposition (iodine + water, every ray)
     → FDK of the composite and of its complement, each with its own window
     → ACNR on the complement → image-domain HYPR on the complement
     → VMI synthesis at 50 / 70 / 100 / 140 keV
  → scored against the noise-free reference and against theory
```

Both kVp channels stay separate all the way into the likelihood; nothing is rebinned into a
single "effective" spectrum.
"""

# ╔═╡ 3d515abe-f3d9-4ce5-96c7-bef7da9bf294
md"""
## Notebook Setup
"""

# ╔═╡ 492bb299-678d-4e6f-8c21-1e9178cc2beb
import PlutoUI

# ╔═╡ 9f8d5cd4-147e-4359-95bc-cc096a53f0e7
import BasisSimulator as BS

# ╔═╡ 2ff539c9-a678-403c-b629-8068a332a0e9
# Use CairoMakie for faithful build-time rendering. Snapshot can still isolate
# and compile independent browser-safe islands without hoisting this import.
import CairoMakie as Mke

# ╔═╡ 320e1b29-4ae3-4757-a2cb-d28b0aa3ec2d
PlutoUI.TableOfContents()

# ╔═╡ 86c52e9e-7987-4504-93e6-128017f5e703
begin
    import GPUSelect
    AT = GPUSelect.Storage()     # the backend array type, directly: MtlArray / CuArray / ROCArray
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end

# ╔═╡ 551f84fe-d7b4-48f9-a475-0c63178a6ede
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 59a5079b-a711-4f28-b3d6-665f0d91fb72
md"""
## Scan Setup and Simulation
"""

# ╔═╡ 040e1000-0000-4000-8000-000000000100
md"""
### 1. `Phantom()` Struct

The Gammex 472 multi-energy phantom: a solid-water body with seven calcium
(50–600 mg/mL) and seven iodine (2–20 mg/mL) inserts, 1 cm thick and z-invariant.
"""

# ╔═╡ 939dcda3-9be5-46c8-aaa1-ded273e8cf04
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,
    n_slices = 16,
    fov_cm = 35.0,
    z_cm = 1.0,
);

# ╔═╡ 5248ba55-965a-41f7-845c-99616018b475
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 040e1000-0000-4000-8000-000000000101
md"""
### 2. `EICTScanner()` Struct

The Apex Elite geometry (source–isocenter 625.6 mm, source–detector 1100 mm, 834 columns
× 0.6 mm on an arc with the quarter-detector offset, 256 rows × 0.625 mm), its Gemstone
scintillator (`:lumex`) and the GE large-body bowtie: the basis-spectral-denoising model,
value for value.

Rapid kVp switching alternates the tube between 80 and 140 kVp within every view period, on
one tube and one detector, so the two channels see the same rays. It is modelled as two
co-registered acquisitions on this one geometry, one per kVp, each with its share of the view
period: its **duty cycle**, 0.65 at 80 kVp and 0.35 at 140 kVp. A channel's tube current is
the instantaneous current times its duty cycle, and its views integrate over that fraction of
the arc the gantry turns per view (`view_arc`).
"""

# ╔═╡ 2c157064-8567-450b-bc08-c2606084a77f
scanner = BS.EICTScanner(
    source_to_isocenter = 625.6,
    source_to_detector = 1100.0,
    detector_rows = 256,
    detector_cols = 834,
    detector_row_size = 0.625,
    detector_col_size = 0.6,
    detector_col_offset = 0.25,
    detector_shape = :arc,
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

# ╔═╡ 040e1000-0000-4000-8000-000000000102
md"""
### 3. `CTProtocol()` Struct

One protocol per exposure: 984 views over a 0.5 s rotation, 5 mm collimation, 4.5 mm of
added aluminium on both. The physical scan's currents are 407 mA at 80 kVp and 405 mA at
140 kVp, each multiplied by its duty cycle.

The simulator's tube output per mA is generic, not GE's, so the tube currents alone do not
fix the dose. Both are scaled by one factor, `DOSE_SCALE`, which keeps the split between
the two exposures and makes the simulated CTDIvol (the package's Monte Carlo dose in the
32 cm body phantom, summed over the exposures) equal to the 10.07 mGy the physical scan
reported. Noise is then simulated at the dose the physical scan delivered.
"""

# ╔═╡ 03c00001-0000-4000-8000-000000000001
# The two exposures of one rotation: each energy is read for its duty cycle of the view
# period, the fraction of the view's arc it integrates over and of its tube current. Each has
# its own noise seed: two acquisitions drawn from one seed would carry the same noise pattern,
# correlated between the channels, which a real scan never has.
exposures = (
    low = (label = "80 kVp", kVp = 80, mA = 407 * 0.65, view_arc = 0.65, seed = 1234,
           filters = [("Al", 4.5)]),
    high = (label = "140 kVp", kVp = 140, mA = 405 * 0.35, view_arc = 0.35, seed = 4321,
            filters = [("Al", 4.5)]),
);

# ╔═╡ 03c00001-0000-4000-8000-000000000002
begin
    "The `CTProtocol` of exposure `e`, its tube current scaled by `mA_scale`."
    exposure_protocol(e; mA_scale = 1.0) = BS.CTProtocol(
        kVp = e.kVp,
        mA = e.mA * mA_scale,
        views = 984,
        rotation_time = 0.5,
        collimation_mm = 5.0,
        additional_filters = e.filters,
    )
    "The CTDIvol (mGy, 32 cm body phantom) of `protocol` on `scanner`."
    ctdi_vol(protocol) = BS.compute_dose(BS.dose_source(scanner, protocol), protocol).ctdi_vol_mGy
end;

# ╔═╡ 03c00001-0000-4000-8000-000000000003
# One factor on both tube currents, so the simulated CTDIvol is the physical scan's 10.07 mGy.
DOSE_SCALE = 10.07 / sum(e -> ctdi_vol(exposure_protocol(e)), exposures)

# ╔═╡ 03c00001-0000-4000-8000-000000000004
protocols = map(e -> exposure_protocol(e; mA_scale = DOSE_SCALE), exposures);

# ╔═╡ 03c00001-0000-4000-8000-000000000005
let
    rows = map(collect(keys(exposures))) do k
        e, p = exposures[k], protocols[k]
        "| $(e.label) | $(round(e.mA / e.view_arc, digits = 1)) | $(e.view_arc) | " *
        "$(round(p.mA, digits = 1)) | $(round(ctdi_vol(p), digits = 2)) |"
    end
    Markdown.parse("""
    `DOSE_SCALE` = $(round(DOSE_SCALE, digits = 4)). The simulated exposures:

    | exposure | physical mA | duty cycle, `view_arc` | simulated mA | CTDIvol (mGy) |
    |---|---:|---:|---:|---:|
    $(join(rows, "\n"))
    """)
end

# ╔═╡ 040e1000-0000-4000-8000-000000000103
md"""
### 4. `SimOptions()` & `ReconOptions()`

The physics switches of the basis-spectral-denoising GE model: every effect of the
energy-integrating path on except optical crosstalk, and the detector efficiency from the
Monte Carlo table of the Lumex scintillator (`detector_efficiency_mode = :mc_lut`).

**View integration.** A detector integrates while the gantry turns, so each view records the
transmitted intensity averaged over the arc it sweeps, `r · view_arc · Δθ` at radius `r`: a
blur of the object, largest far from the isocentre, that the noise, counted once per view,
does not share. `view_samples = 5` samples that arc with five sub-views (the midpoint rule),
0.2 mm apart at the edge of a 35 cm field. Without it the simulated signal is sharper,
relative to its noise, than a physical scanner's, and no reconstruction window can match both
the physical resolution and the physical noise texture.

**Draws.** Each exposure is simulated three times, differing only in the noise: the
noise-free expectation (draw 0, `use_noise = false`), the measured scan (draw 1) and a
calibration scan (draw 9) of its own seeds. The reconstruction grid is the clinical series',
8 slices of 0.625 mm over the central 5 mm, 512² over 35 cm.
"""

# ╔═╡ 2d65a0c0-b25d-41ad-9cd3-e7a2d08a2482
begin
    VIEW_SAMPLES = 5
    ge_opts = BS.SimOptions(
        projector = :dd_fast,
        detector_efficiency_mode = :mc_lut,
        view_samples = VIEW_SAMPLES,
    )
    "The same physics switches with one field replaced."
    with_option(opts, field, value) = BS.SimOptions(;
        (f => getfield(opts, f) for f in fieldnames(BS.SimOptions) if f !== field)...,
        (field => value,)...,
    )
end;

# ╔═╡ 03c00001-0000-4000-8000-000000000006
begin
    # Draw 0 is the noise-free expectation; draw r ≥ 1 gives each exposure the seed
    # `seed + 1000 (r − 1)`, so the exposures stay independent within a draw and the draws of
    # each other (basis-spectral-denoising's seeds for the Gammex).
    DRAWS = (reference = 0, measured = 1, calibration = 9)
    realization_seed(e, r) = e.seed + 1000 * (max(r, 1) - 1)
    "The options of draw `r` of exposure `e`: its seed, its noise, its arc."
    draw_opts(e, r) = with_option(with_option(with_option(ge_opts,
        :seed, realization_seed(e, r)), :use_noise, r > 0), :view_arc, e.view_arc)
end;

# ╔═╡ 08cbc6fd-3c7c-432f-99e5-b220f8fe7fde
recon_opts = let
    slice_thickness_mm = 0.625
    n_recon_slices = round(Int, 5.0 / slice_thickness_mm)
    BS.ReconOptions(
        matrix_size = (512, 512, n_recon_slices),
        fov_cm = 35.0,
        z_cm = 0.5,
    )
end;

# ╔═╡ 5ecd97c6-ad47-4558-886d-22ed45eda97d
md"""
### 5. Forward Project: `simulate!`

Each exposure is projected once. The noise-free draw runs first with
`keep_projection = true`, which returns everything before the noise; the measured and
calibration draws pass it back (`projection`) and only draw their noise. That is
bit-identical to simulating each draw afresh, at a fraction of the cost.

Each acquisition keeps what the spectral basis needs: its corrected log sinograms, its
per-ray air counts `I0_ray` (the detector's air count × the bowtie air profile) and its
per-ray detected spectrum from `resolve_source_spectrum_full` (source × filtration × bowtie ×
detector response), the same model the forward projector applied.
"""

# ╔═╡ 4315ef69-aa2f-4ee0-a13b-c65e01fb87ce
dual_scans = map(collect(keys(exposures))) do k
    e, protocol = exposures[k], protocols[k]
    @info "Simulating dual-kVp channel: $(e.label)"
    channels = Dict{Int, Array{Float32, 3}}()
    projection = nothing
    model = nothing
    elapsed_s = @elapsed for r in (DRAWS.reference, DRAWS.measured, DRAWS.calibration)
        opts = draw_opts(e, r)
        ws = BS.create_workspace(scanner, protocol, opts, recon_opts, phantom)
        try
            result = BS.simulate!(ws, phantom, protocol, opts; report_dose = false,
                keep_projection = projection === nothing, projection)
            projection === nothing && (projection = result.projection)
            channels[r] = Float32.(Array(ws.sinogram))
            if model === nothing
                air = ws.bowtie_air_reference === nothing ?
                    ones(Float32, ws.geom.n_cols, ws.geom.n_rows) :
                    Float32.(Array(ws.bowtie_air_reference))
                I0 = BS.compute_detector_I0(ws.geom, protocol, sum(ws.weights)) * Float64(ws.η_eff)
                energies, response = BS.resolve_source_spectrum_full(
                    opts, protocol; scanner = scanner, geom = ws.geom,
                )
                model = (
                    geom = ws.geom, I0_ray = Float32.(I0 .* air),
                    energies = Float64.(energies), response = Float32.(response),
                )
            end
        finally
            BS.release_backend!(ws)
        end
    end
    (; label = e.label, channels, elapsed_s, model...)
end;

# ╔═╡ 03c00001-0000-4000-8000-000000000007
Markdown.parse("""
The three draws of the two exposures took
$(join(["$(round(s.elapsed_s, digits = 1)) s for $(s.label)" for s in dual_scans], " and ")),
one projection each. Each sinogram is $(join(size(first(dual_scans).channels[DRAWS.measured]), " × "))
(columns × rows × views).
""")

# ╔═╡ dc8a8352-5598-4cdd-952f-3d77367850e9
md"""
## VMI Pipeline

### 1. Spectral Basis from the Two Acquisitions

`spectral_basis_from_acquisitions` merges the two kVp energy grids onto their union and
scales each acquisition's per-ray spectrum by its own air counts, so the likelihood sees the
absolute response ``\Phi_k(E)`` of every ray and channel: source, filtration, bowtie and
detector, the model that generated the data. It does not depend on the noise, so every draw
shares it.
"""

# ╔═╡ 4ca28c64-ee96-47c8-b7c3-f0e0c4c99423
basis = BS.spectral_basis_from_acquisitions(acquisitions = [
    (energies = s.energies, response = s.response, I0_ray = s.I0_ray) for s in dual_scans
]);

# ╔═╡ 03a10001-0000-4000-8000-000000000001
let
    col, row = size(basis.Φ, 1) ÷ 2 + 1, size(basis.Φ, 2) ÷ 2 + 1
    fig = Mke.Figure(size = (1180, 520))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Detected Spectra",
        subtitle = "Central ray, Φ_k(E) normalized to its peak",
        xlabel = "Energy (keV)", ylabel = "Relative detected fluence",
        titlesize = 32, subtitlesize = 22, xlabelsize = 22, ylabelsize = 22,
    )
    for (k, color) in zip(1:basis.n_channels, (:royalblue, :crimson))
        φ = Float64.(basis.Φ[col, row, :, k])
        mean_E = sum(basis.E .* φ) / sum(φ)
        Mke.lines!(
            ax, basis.E, φ ./ maximum(φ); color, linewidth = 3,
            label = "$(dual_scans[k].label) (mean $(round(mean_E, digits = 1)) keV)",
        )
    end
    Mke.axislegend(ax; position = :rt, labelsize = 18)
    fig
end

# ╔═╡ 03a10001-0000-4000-8000-000000000002
Markdown.parse("""
The basis holds $(basis.n_channels) channels on a $(length(basis.E))-point energy grid,
$(round(Int, minimum(basis.E))) to $(round(Int, maximum(basis.E))) keV, for
$(size(basis.Φ, 1)) × $(size(basis.Φ, 2)) rays; the response sums to the air counts to
within $(round(basis.I0_relerr, sigdigits = 2)) (relative).
""")

# ╔═╡ 03a10001-0000-4000-8000-000000000003
md"""
### 2. The Chain's Settings

**The denoiser** is generalized HYPR-LR in both domains, `BS.SpectralHYPR()`: before the
decomposition, a 3 × 3 (column × view) window on the counts of each detector row, pooling
views of one parity so the odd- and even-view halves stay independent, with each channel's
dispersion measured on the air rays; after the reconstruction, the complement of the basis
pair pooled within each slice, over the window with the least estimated risk. Neither ever
pools across detector rows or slices. The decomposition is the K-channel estimator with its
published controls; T-LBF (a photon-counting filter) is off; the view-direction antialias is
on; ACNR is on (`vmi_pipeline`'s default).

**A pair of FDK windows.** A VMI is ``A(E)\,M + B(E)\,I_\perp``, with ``M`` the
minimum-noise VMI (the composite, at ``E^\star``) and ``I_\perp`` its complement, the
iodine component whose noise is uncorrelated with ``M``'s. `BS.PairFilter` gives each its own
window: the composite's sets the resolution and noise texture at ``E^\star``, the
complement's how they change away from it, and one pair serves every energy. A window per
basis image instead would stop the strongly anti-correlated noise of the water and iodine
images cancelling wherever the two windows differ, leaving uncancelled noise in every VMI.
Each window is an apodized ramp ``W(f) = \exp(-(f/f_c)^p)`` on the grid-Nyquist axis, the GE
pair fitted by basis-spectral-denoising to the physical Gammex scan's MTF at every energy and
NPS shape, on this scanner's clinical grid, which is this notebook's grid.
"""

# ╔═╡ 03a10001-0000-4000-8000-000000000004
# the published denoiser: ProjectionHYPR(3 × 3 box, view_stride = 2) and ImageHYPR()
HYPR_CHAIN = BS.SpectralHYPR()

# ╔═╡ 03a10001-0000-4000-8000-000000000005
VMI_CHAIN = (method = :nchannel, controls = BS.NChannelControls(), use_tlbf = false, antialias = true);

# ╔═╡ 03a10001-0000-4000-8000-000000000006
# The GE pair of basis-spectral-denoising (`RECON.ge`): W(f) = exp(-(f / f_c)^p) at 11 knots.
GE_KERNELS = (composite = (fc = 0.55, p = 1.5), complement = (fc = 1.05, p = 1.0));

# ╔═╡ 03c00001-0000-4000-8000-000000000008
PAIR_FILTER = let knots = Tuple(range(0.0, 1.0, length = 11))
    window(q) = BS.CustomFilter(knots, Tuple(round(exp(-(x / q.fc)^q.p), digits = 5) for x in knots))
    BS.PairFilter(window(GE_KERNELS.composite), window(GE_KERNELS.complement))
end;

# ╔═╡ 03c00001-0000-4000-8000-000000000009
let
    f = range(0.0, 1.0, length = 201)
    fig = Mke.Figure(size = (900, 480))
    ax = Mke.Axis(
        fig[1, 1]; title = "The GE Pair of FDK Windows",
        xlabel = "Frequency (fraction of the grid's Nyquist)", ylabel = "Window W(f)",
        titlesize = 28, xlabelsize = 20, ylabelsize = 20,
    )
    for (name, w, q, color) in (("composite M", PAIR_FILTER.composite, GE_KERNELS.composite, :royalblue),
                                ("complement I⊥", PAIR_FILTER.complement, GE_KERNELS.complement, :crimson))
        Mke.lines!(ax, f, [BS.frequency_window(w, x) for x in f]; color, linewidth = 3,
            label = "$(name): f_c = $(q.fc), p = $(q.p)")
    end
    Mke.axislegend(ax; position = :rt, labelsize = 18)
    fig
end

# ╔═╡ 03c00001-0000-4000-8000-00000000000a
md"""
### 3. The Calibration Draw: `pair_basis` and `composite_energy`

The pair the windows act on, ``(E^\star, \beta)``, and the energy of the composite that ACNR
and the image-domain HYPR act on are properties of the scanner and protocol. They are
measured once, on the calibration draw, and fixed for every other draw, exactly as
basis-spectral-denoising's `PAIR_BASIS` does:

1. the calibration draw decomposed as measured and after the projection-domain HYPR alone
   (`vmi_pipeline(; keep_sinograms = true)`, with `SpectralHYPR(image = nothing)` for the
   second);
2. `BS.spectral_pair` of the first with a standard soft-tissue window: its minimum-noise
   energy ``E^\star`` and complement ``\beta``, the `pair_basis`;
3. `BS.spectral_pair` of the second on that basis: the minimum-noise energy of the pair after
   the projection-domain HYPR, which moves it, the `composite_energy`.

Near its minimum the VMI noise hardly changes with energy, so the argmin of one acquisition
is itself noise and would move from draw to draw. Measured on a draw of its own, it cannot
tune the chain to the draw being scored, and the noise-free reference, which has no noise to
measure it from, is reconstructed with the same pair as the measured scan.
"""

# ╔═╡ 03c00001-0000-4000-8000-00000000000b
calibration = let
    geom = first(dual_scans).geom
    matrix_size = recon_opts.matrix_size
    channels = [s.channels[DRAWS.calibration] for s in dual_scans]
    decompose(denoiser) = BS.vmi_pipeline(;
        channels, basis, geom, to_backend = to_gpu, matrix_size, denoiser,
        keep_sinograms = true, use_acnr = false, VMI_CHAIN...,
    ).sinograms
    pair(d; kw...) = BS.spectral_pair(d.water, d.iodine, geom, matrix_size;
        filter = BS.SoftFilter(), antialias = VMI_CHAIN.antialias, to_backend = to_gpu, kw...)
    x = pair(decompose(nothing))
    c = pair(decompose(BS.SpectralHYPR(image = nothing)); basis = (Estar = x.Estar, β = x.β))
    (pair_basis = (Estar = x.Estar, β = x.β), composite_energy = c.Estar)
end

# ╔═╡ 040e1000-0000-4000-8000-000000000003
md"""
### 4. `vmi_pipeline`

One call, from the two measured sinograms to the VMI stack, reconstructed on the
notebook's grid (`recon_opts.matrix_size`) with every detector row kept: the published
chain, with the GE's windows and the calibration draw's pair.
"""

# ╔═╡ 040e0002-0000-4000-8000-000000000002
dual_vmi = BS.vmi_pipeline(;
    channels = [s.channels[DRAWS.measured] for s in dual_scans],
    basis,
    geom = first(dual_scans).geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = (50.0, 70.0, 100.0, 140.0),
    denoiser = HYPR_CHAIN,
    fbp_filter = PAIR_FILTER,
    pair_basis = calibration.pair_basis,
    composite_energy = calibration.composite_energy,
    VMI_CHAIN...,
);

# ╔═╡ 03c00001-0000-4000-8000-00000000000c
md"""
The **noise-free reference** is the noise-free draw through the same reconstruction, the
same windows and the same pair, with nothing to denoise: no HYPR and no ACNR. The measured
VMIs are scored against it, so the scores measure what the noise and the denoising do,
separately from what the physics the decomposition does not invert exactly (scatter, and the
blurs of the focal spot, the detector and the view integration) does to both.
"""

# ╔═╡ 03c00001-0000-4000-8000-00000000000d
reference_vmi = BS.vmi_pipeline(;
    channels = [s.channels[DRAWS.reference] for s in dual_scans],
    basis,
    geom = first(dual_scans).geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = (50.0, 70.0, 100.0, 140.0),
    fbp_filter = PAIR_FILTER,
    pair_basis = calibration.pair_basis,
    composite_energy = calibration.composite_energy,
    use_acnr = false,
    VMI_CHAIN...,
);

# ╔═╡ 03a10001-0000-4000-8000-000000000007
let
    q = dual_vmi.quality
    d = dual_vmi.settings.denoiser
    b = calibration.pair_basis
    pct(x) = round(100x, digits = 3)
    Markdown.parse("""
    The calibration draw fixed the windows' pair at E* = $(round(Int, b.Estar)) keV,
    β = $(round(b.β, sigdigits = 3)), and the composite at $(round(Int, calibration.composite_energy)) keV.
    On the measured draw the decomposition solved $(q.n_rays) rays in $(round(dual_vmi.elapsed_s, digits = 1)) s,
    $(round(q.outer_mean, digits = 1)) outer iterations on average; $(pct(q.frac_not_converged))% did not
    converge and $(pct(q.frac_bound_iodine))% / $(pct(q.frac_bound_water))% touched the iodine / water bounds.
    The projection HYPR measured dispersions, the variance-to-mean ratio of the counts on the air rays, of
    $(join(round.(d.dispersion, digits = 2), " and ")) for 80 and 140 kVp. The image HYPR pooled the
    complement over a $(d.image_estimates.window) × $(d.image_estimates.window) window.
    """)
end

# ╔═╡ 040e0002-0000-4000-8000-000000000003
md"""
### 5. Basis Maps and VMIs

The water and iodine basis pair after ACNR and image HYPR (mid slice), and the VMIs
synthesized from it. Every VMI comes from the same basis pair.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000003
let
    fig = Mke.Figure(size = (1180, 580))
    mid = size(dual_vmi.images.water, 3) ÷ 2 + 1
    panels = (
        ("Iodine Basis", "g/cm³", dual_vmi.images.iodine),
        ("Water Basis", "g/cm³", dual_vmi.images.water),
    )
    for (column, (title, label, volume)) in pairs(panels)
        image = volume[:, :, mid]
        range = Tuple(quantile(vec(image), (0.01, 0.99)))
        axis = Mke.Axis(
            fig[1, 2column - 1]; title, aspect = Mke.DataAspect(), titlesize = 32,
        )
        Mke.heatmap!(axis, image; colormap = :viridis, colorrange = range)
        Mke.hidedecorations!(axis)
        Mke.Colorbar(
            fig[1, 2column]; colormap = :viridis, colorrange = range,
            label, width = 16, labelsize = 22,
        )
    end
    fig
end

# ╔═╡ 040e0001-0000-4000-8000-000000000004
let
    fig = Mke.Figure(size = (1180, 1180))
    mid = size(dual_vmi.vmis, 3) ÷ 2 + 1
    for (index, energy) in pairs(dual_vmi.energies)
        row = ((index - 1) ÷ 2) + 1
        column = ((index - 1) % 2) + 1
        axis = Mke.Axis(
            fig[row, column]; title = "$(Int(energy)) keV VMI",
            aspect = Mke.DataAspect(), titlesize = 32,
        )
        Mke.heatmap!(
            axis, dual_vmi.vmis[:, :, mid, index];
            colormap = :grays, colorrange = (-200, 500),
        )
        Mke.hidedecorations!(axis)
    end
    Mke.Colorbar(
        fig[1:2, 3]; colormap = :grays, colorrange = (-200, 500),
        label = "HU", width = 16, labelsize = 22, ticklabelsize = 18,
    )
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "dual_kvp_vmi_projection_grid.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 040e0001-0000-4000-8000-000000000008
md"""
## Results

Per-rod measured, noise-free and theoretical HU at the four VMI energies, the water-region
mean and noise, and the linear regression of measured on theoretical HU.

- **Measured HU**: mean over an 8-px-radius circle at each rod's centroid, over every
  reconstructed slice, of the measured VMIs.
- **Noise-free HU**: the same mean on the noise-free reference.
- **Theoretical HU**: ``1000\,(\mu_r(E) - \mu_w(E))/\mu_w(E)`` of the rod's material from
  `BS.compute_μ_at_energy`, with no fitting.
- **Water region**: the solid-water body eroded by 12 px, away from every insert and edge.
  Its noise σ is the standard deviation of the measured VMI minus the noise-free reference,
  the noise alone, without the phantom's own nonuniformity.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000002
dual_results = let
    mask = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
    water_mask = collect(BS.erode_mask_2d(
        mask .== UInt8(BS.REGION_SOLID_WATER); erode_px = 12.0,
    ))
    rod_labels = (
        Ca = UInt8.((10, 11, 12, 13, 14, 15, 16)),
        I = UInt8.((20, 21, 22, 23, 24, 25, 26)),
    )
    rod_names = (
        Ca = ("50 mg/mL", "100 mg/mL", "200 mg/mL", "300 mg/mL",
              "400 mg/mL", "500 mg/mL", "600 mg/mL"),
        I = ("2.0 mg/mL", "2.5 mg/mL", "5.0 mg/mL", "7.5 mg/mL",
             "10.0 mg/mL", "15.0 mg/mL", "20.0 mg/mL"),
    )
    function rod_roi(label)
        pixels = findall(==(label), mask)
        cx = mean(pixel -> Float64(pixel[1]), pixels)
        cy = mean(pixel -> Float64(pixel[2]), pixels)
        [CartesianIndex(i, j)
         for j in max(1, floor(Int, cy - 8)):min(size(mask, 2), ceil(Int, cy + 8))
         for i in max(1, floor(Int, cx - 8)):min(size(mask, 1), ceil(Int, cx + 8))
         if (i - cx)^2 + (j - cy)^2 ≤ 64]
    end
    energies = Float64.(dual_vmi.energies)
    vmi(stack, index) = view(stack.vmis, :, :, :, index)
    region(volume, pixels) = [volume[p, z] for z in axes(volume, 3) for p in pixels]
    μwater = Dict(E => BS.compute_μ_at_energy(BS.XA.Materials.water, E) for E in energies)
    rods = Dict{Symbol, NamedTuple}()
    for group in (:Ca, :I)
        measured = zeros(length(rod_labels[group]), length(energies))
        reference = similar(measured)
        theoretical = similar(measured)
        for (row, label) in pairs(rod_labels[group])
            roi = rod_roi(label)
            material = phantom_cpu.materials[Int(label) + 1]
            for (column, E) in pairs(energies)
                measured[row, column] = mean(region(vmi(dual_vmi, column), roi))
                reference[row, column] = mean(region(vmi(reference_vmi, column), roi))
                μ = BS.compute_μ_at_energy(material, E)
                theoretical[row, column] = 1000 * (μ - μwater[E]) / μwater[E]
            end
        end
        rods[group] = (names = rod_names[group], measured, reference, theoretical)
    end
    water_pixels = findall(water_mask)
    water(stack, k) = region(vmi(stack, k), water_pixels)
    (
        energies, water_mask, rods,
        water_mean = [mean(water(dual_vmi, k)) for k in eachindex(energies)],
        water_mean_reference = [mean(water(reference_vmi, k)) for k in eachindex(energies)],
        water_noise = [std(water(dual_vmi, k) .- water(reference_vmi, k)) for k in eachindex(energies)],
        finite = all(isfinite, dual_vmi.vmis) && all(isfinite, reference_vmi.vmis),
    )
end;

# ╔═╡ 040e1000-0000-4000-8000-000000000040
md"""
### Water ROI

The eroded solid-water region, overlaid on the 70 keV VMI, and its mean HU at each energy,
measured and noise-free.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000005
let
    overlay = Float32[value ? 1.0f0 : NaN32 for value in dual_results.water_mask]
    index70 = findfirst(==(70.0), dual_results.energies)
    image70 = dual_vmi.vmis[:, :, size(dual_vmi.vmis, 3) ÷ 2 + 1, index70]
    n = length(dual_results.energies)
    fig = Mke.Figure(size = (1180, 580))
    axis = Mke.Axis(
        fig[1, 1]; title = "Eroded Water Region",
        subtitle = "Overlaid on 70 keV VMI",
        aspect = Mke.DataAspect(), titlesize = 32, subtitlesize = 24,
    )
    Mke.heatmap!(axis, image70; colormap = :grays, colorrange = (-200, 500))
    Mke.heatmap!(
        axis, overlay; colormap = :reds, alpha = 0.5,
        nan_color = (:white, 0.0),
    )
    Mke.hidedecorations!(axis)
    mean_axis = Mke.Axis(
        fig[1, 2]; title = "Water Region Mean HU",
        xlabel = "VMI Energy (keV)", ylabel = "HU",
        xticks = (1:n, string.(Int.(dual_results.energies))), titlesize = 32,
    )
    values = vcat(dual_results.water_mean, dual_results.water_mean_reference)
    group = vcat(fill(1, n), fill(2, n))
    Mke.barplot!(mean_axis, vcat(1:n, 1:n), values; dodge = group,
        color = [(:tomato, :gray60)[g] for g in group], strokecolor = :black, strokewidth = 1)
    Mke.hlines!(mean_axis, [0.0]; color = :black, linestyle = :dash)
    for (index, value) in pairs(dual_results.water_mean)
        Mke.text!(
            mean_axis, index - 0.2, value;
            text = "$(round(value, digits = 1))",
            align = (:center, value ≥ 0 ? :bottom : :top),
            offset = (0, value ≥ 0 ? 5 : -5), fontsize = 15,
        )
    end
    Mke.axislegend(mean_axis,
        [Mke.PolyElement(color = :tomato), Mke.PolyElement(color = :gray60)],
        ["Measured", "Noise-free reference"]; position = :lb, labelsize = 16)
    Mke.ylims!(mean_axis, -10, 10)
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "vmi_water_roi_check.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 040e1000-0000-4000-8000-000000000050
md"""
### Water-Region Noise

The standard deviation of the measured VMI about the noise-free reference, over the same
eroded water region, at each energy.
"""

# ╔═╡ 040e1000-0000-4000-8000-000000000051
let
    n = length(dual_results.energies)
    fig = Mke.Figure(size = (900, 580))
    noise_axis = Mke.Axis(
        fig[1, 1]; title = "Water-Region Noise vs Energy",
        xlabel = "VMI Energy (keV)", ylabel = "Noise σ (HU)",
        xticks = (1:n, string.(Int.(dual_results.energies))), titlesize = 32,
    )
    Mke.barplot!(
        noise_axis, 1:n, dual_results.water_noise; color = :tomato,
        strokecolor = :black, strokewidth = 1,
    )
    for index in 1:n
        Mke.text!(
            noise_axis, index, dual_results.water_noise[index];
            text = "σ=$(round(dual_results.water_noise[index], digits = 1))\n" *
                   "⟨HU⟩=$(round(dual_results.water_mean[index], digits = 1))",
            align = (:center, :bottom), offset = (0, 8),
        )
    end
    Mke.ylims!(noise_axis, 0, 1.3maximum(dual_results.water_noise))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "dual_kvp_vmi_water_noise_vs_energy.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 040e1000-0000-4000-8000-000000000060
md"""
### Per-Rod Measured vs Theoretical

Solid lines show measured HU, crosses the noise-free reference and dashed lines theoretical
HU, for the calcium and iodine inserts across the four energies.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000006
let
    fig = Mke.Figure(size = (1180, 580))
    panels = (
        (group = :Ca, title = "Calcium rods", subtitle = "50–600 mg/mL",
         cmap = Mke.cgrad(:Oranges, 7; categorical = true), ylim = (0, 3800)),
        (group = :I, title = "Iodine rods", subtitle = "2–20 mg/mL",
         cmap = Mke.cgrad(:GnBu, 7; categorical = true), ylim = (0, 1500)),
    )
    for (column, panel) in pairs(panels)
        axis = Mke.Axis(
            fig[1, column]; title = panel.title, subtitle = panel.subtitle,
            xlabel = "VMI energy (keV)", ylabel = "HU",
            xticks = dual_results.energies, titlesize = 32, subtitlesize = 24,
        )
        Mke.ylims!(axis, panel.ylim...)
        data = dual_results.rods[panel.group]
        for index in eachindex(data.names)
            color = panel.cmap[index]
            Mke.scatterlines!(
                axis, dual_results.energies, vec(data.measured[index, :]);
                color, linewidth = 2.5, markersize = 9, label = data.names[index],
            )
            Mke.scatter!(
                axis, dual_results.energies, vec(data.reference[index, :]);
                color = :black, marker = :xcross, markersize = 9,
            )
            Mke.lines!(
                axis, dual_results.energies, vec(data.theoretical[index, :]);
                color, linewidth = 1.6, linestyle = :dash,
            )
        end
        Mke.axislegend(axis; position = :rt, labelsize = 16)
    end
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "vmi_vs_theoretical.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 03c00001-0000-4000-8000-00000000000e
let
    r = dual_results
    f(x) = round(x, digits = 1)
    bias(g, k) = r.rods[g].measured[:, k] .- r.rods[g].reference[:, k]
    mae(a, b) = mean(abs.(a .- b))
    rows = map(eachindex(r.energies)) do k
        ca, io = r.rods[:Ca], r.rods[:I]
        allbias = vcat(bias(:Ca, k), bias(:I, k))
        "| $(Int(r.energies[k])) | $(f(r.water_mean[k])) / $(f(r.water_mean_reference[k])) | $(f(r.water_noise[k])) | " *
        "$(f(mae(ca.measured[:, k], ca.theoretical[:, k]))) / $(f(mae(ca.reference[:, k], ca.theoretical[:, k]))) | " *
        "$(f(mae(io.measured[:, k], io.theoretical[:, k]))) / $(f(mae(io.reference[:, k], io.theoretical[:, k]))) | " *
        "$(f(mean(allbias))) | $(f(maximum(abs, allbias))) |"
    end
    Markdown.parse("""
    **Accuracy against the noise-free reference and against theory.** Water: mean HU of the
    measured / noise-free VMI, and the noise σ. Rods: the mean absolute error against theory of
    the measured / noise-free VMI, and the mean and the largest absolute bias of the 14 measured
    rods against the noise-free reference.

    | keV | water HU | water σ (HU) | Ca rods, error vs theory (HU) | I rods, error vs theory (HU) | rod bias vs reference, mean (HU) | rod bias vs reference, max (HU) |
    |---:|---:|---:|---:|---:|---:|---:|
    $(join(rows, "\n"))
    """)
end

# ╔═╡ 03c00001-0000-4000-8000-00000000000f
let
    r = dual_results
    f(x) = round(x, digits = 1)
    gap = maximum(abs, r.water_mean .- r.water_mean_reference)
    mae(g, a) = [mean(abs.(getfield(r.rods[g], a)[:, k] .- r.rods[g].theoretical[:, k])) for k in eachindex(r.energies)]
    worst(a) = maximum(max(x, y) for (x, y) in zip(mae(:Ca, a), mae(:I, a)))
    Markdown.parse("""
    The solid water reads $(join(f.(r.water_mean), " / ")) HU at 50 / 70 / 100 / 140 keV, within
    $(f(gap)) HU of the noise-free reference: an offset the reference shares comes from physics the
    decomposition does not invert exactly (scatter, and the blurs of the focal spot, the detector and
    the view integration, taken before the log), not from the noise or the denoising. The noise falls from σ = $(f(first(r.water_noise))) HU at 50 keV to $(f(last(r.water_noise))) HU at
    140 keV. The largest mean rod error against theory, over both rod groups and every energy, is
    $(f(worst(:measured))) HU measured and $(f(worst(:reference))) HU noise-free.
    """)
end

# ╔═╡ 040e1000-0000-4000-8000-000000000070
md"""
### Linear Regression

Measured rod HU regressed on theoretical HU at each energy. The dashed identity line is
perfect agreement.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000007
let
    colors = Dict(
        50.0 => Mke.RGBf(0.85, 0.27, 0.1),
        70.0 => Mke.RGBf(0.95, 0.65, 0.13),
        100.0 => Mke.RGBf(0.13, 0.59, 0.85),
        140.0 => Mke.RGBf(0.1, 0.27, 0.65),
    )
    fig = Mke.Figure(size = (1000, 1200))
    for (row, group) in enumerate((:Ca, :I))
        data = dual_results.rods[group]
        axis = Mke.Axis(
            fig[row, 1];
            title = group == :Ca ? "Calcium regression" : "Iodine regression",
            xlabel = "Theoretical HU", ylabel = "Measured HU",
            titlesize = 30,
        )
        low = min(0.0, minimum(data.measured), minimum(data.theoretical))
        high = 1.05max(maximum(data.measured), maximum(data.theoretical))
        Mke.lines!(
            axis, [low, high], [low, high]; color = :black,
            linestyle = :dash, linewidth = 2, label = "Unity (y=x)",
        )
        for (column, energy) in pairs(dual_results.energies)
            x = vec(data.theoretical[:, column])
            y = vec(data.measured[:, column])
            slope = sum((x .- mean(x)) .* (y .- mean(y))) / sum(abs2, x .- mean(x))
            intercept = mean(y) - slope * mean(x)
            prediction = intercept .+ slope .* x
            r2 = 1 - sum(abs2, y .- prediction) / sum(abs2, y .- mean(y))
            endpoints = collect(extrema(x))
            color = colors[energy]
            Mke.scatter!(axis, x, y; color, markersize = 11)
            Mke.lines!(
                axis, endpoints, intercept .+ slope .* endpoints;
                color, linewidth = 2,
                label = "$(Int(energy)) keV: slope=$(round(slope, digits = 3)), " *
                        "R²=$(round(r2, digits = 4))",
            )
        end
        Mke.axislegend(axis; position = :rb, labelsize = 16)
    end
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "vmi_regression.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 040e0002-0000-4000-8000-000000000004
verification = let
    checks = NamedTuple[]
    addcheck(name, value, pass) = push!(checks, (; name, value, pass))
    addcheck("both kVp channels in the likelihood", dual_vmi.settings.n_channels,
             dual_vmi.settings.n_channels == 2)
    addcheck("chain: projection + image HYPR, ACNR, pair of windows",
             (dual_vmi.settings.denoiser !== nothing, dual_vmi.settings.acnr !== nothing,
              dual_vmi.settings.recon.filter isa BS.PairFilter),
             dual_vmi.settings.denoiser !== nothing && dual_vmi.settings.acnr !== nothing &&
             dual_vmi.settings.recon.filter isa BS.PairFilter)
    addcheck("measured and reference share the pair",
             dual_vmi.settings.pair.basis == reference_vmi.settings.pair.basis,
             dual_vmi.settings.pair.basis == reference_vmi.settings.pair.basis)
    addcheck("all VMI values finite", dual_results.finite, dual_results.finite)
    water_worst = maximum(abs, dual_results.water_mean)
    addcheck("solid-water worst absolute HU", round(water_worst, digits = 2), water_worst ≤ 10)
    passed = count(check -> check.pass, checks)
    rows = join([
        "| $(check.name) | $(check.value) | $(check.pass ? "✅" : "❌") |"
        for check in checks
    ], "\n")
    Markdown.parse("""
### $(passed == length(checks) ? "✅ Verification: PASS" : "❌ Verification: CHECK")

| check | value | pass |
|---|---:|:---:|
$rows
""")
end

# ╔═╡ 040e0002-0000-4000-8000-000000000006
md"""
### Summary

```
Simulate 80 + 140 kVp (duty cycle 0.65 / 0.35, view_samples = 5, dose-matched to 10.07 mGy)
   one projection per exposure → noise-free, measured and calibration draws
   → BS.spectral_basis_from_acquisitions (absolute per-ray response of each kVp)
   → calibration draw → pair_basis, composite_energy
   → BS.vmi_pipeline(; denoiser = SpectralHYPR(), fbp_filter = PAIR_FILTER, pair_basis, composite_energy)
        projection HYPR → K = 2 n-channel decomposition on every ray
        → FDK of the composite and complement, each with its window
        → ACNR on the complement → image HYPR → VMI synthesis at 50 / 70 / 100 / 140 keV
   → water, noise, per-rod and regression checks against the noise-free reference and theory
```

Every VMI is synthesized from the same water/iodine pair; no energy-dependent filtering is
applied to any VMI.
"""

# ╔═╡ Cell order:
# ╟─d3054785-9e00-4094-a491-088ce63be9dc
# ╟─f2798d62-3509-4cc4-a24f-39ace8bb5a9e
# ╟─3d515abe-f3d9-4ce5-96c7-bef7da9bf294
# ╠═171294a2-26bd-49e2-ac92-9df48ae5444f
# ╠═69358294-97f2-4782-94d7-c29c747c45f4
# ╠═9ae27110-5c47-442b-a98e-d137599570f2
# ╠═492bb299-678d-4e6f-8c21-1e9178cc2beb
# ╠═9f8d5cd4-147e-4359-95bc-cc096a53f0e7
# ╠═2ff539c9-a678-403c-b629-8068a332a0e9
# ╠═320e1b29-4ae3-4757-a2cb-d28b0aa3ec2d
# ╠═86c52e9e-7987-4504-93e6-128017f5e703
# ╟─551f84fe-d7b4-48f9-a475-0c63178a6ede
# ╟─59a5079b-a711-4f28-b3d6-665f0d91fb72
# ╟─040e1000-0000-4000-8000-000000000100
# ╠═939dcda3-9be5-46c8-aaa1-ded273e8cf04
# ╠═5248ba55-965a-41f7-845c-99616018b475
# ╟─040e1000-0000-4000-8000-000000000101
# ╠═2c157064-8567-450b-bc08-c2606084a77f
# ╟─040e1000-0000-4000-8000-000000000102
# ╠═03c00001-0000-4000-8000-000000000001
# ╠═03c00001-0000-4000-8000-000000000002
# ╠═03c00001-0000-4000-8000-000000000003
# ╠═03c00001-0000-4000-8000-000000000004
# ╟─03c00001-0000-4000-8000-000000000005
# ╟─040e1000-0000-4000-8000-000000000103
# ╠═2d65a0c0-b25d-41ad-9cd3-e7a2d08a2482
# ╠═03c00001-0000-4000-8000-000000000006
# ╠═08cbc6fd-3c7c-432f-99e5-b220f8fe7fde
# ╟─5ecd97c6-ad47-4558-886d-22ed45eda97d
# ╠═4315ef69-aa2f-4ee0-a13b-c65e01fb87ce
# ╟─03c00001-0000-4000-8000-000000000007
# ╟─dc8a8352-5598-4cdd-952f-3d77367850e9
# ╠═4ca28c64-ee96-47c8-b7c3-f0e0c4c99423
# ╟─03a10001-0000-4000-8000-000000000001
# ╟─03a10001-0000-4000-8000-000000000002
# ╟─03a10001-0000-4000-8000-000000000003
# ╠═03a10001-0000-4000-8000-000000000004
# ╠═03a10001-0000-4000-8000-000000000005
# ╠═03a10001-0000-4000-8000-000000000006
# ╠═03c00001-0000-4000-8000-000000000008
# ╟─03c00001-0000-4000-8000-000000000009
# ╟─03c00001-0000-4000-8000-00000000000a
# ╠═03c00001-0000-4000-8000-00000000000b
# ╟─040e1000-0000-4000-8000-000000000003
# ╠═040e0002-0000-4000-8000-000000000002
# ╟─03c00001-0000-4000-8000-00000000000c
# ╠═03c00001-0000-4000-8000-00000000000d
# ╟─03a10001-0000-4000-8000-000000000007
# ╟─040e0002-0000-4000-8000-000000000003
# ╟─040e0001-0000-4000-8000-000000000003
# ╟─040e0001-0000-4000-8000-000000000004
# ╟─040e0001-0000-4000-8000-000000000008
# ╠═040e0001-0000-4000-8000-000000000002
# ╟─040e1000-0000-4000-8000-000000000040
# ╟─040e0001-0000-4000-8000-000000000005
# ╟─040e1000-0000-4000-8000-000000000050
# ╟─040e1000-0000-4000-8000-000000000051
# ╟─040e1000-0000-4000-8000-000000000060
# ╟─040e0001-0000-4000-8000-000000000006
# ╟─03c00001-0000-4000-8000-00000000000e
# ╟─03c00001-0000-4000-8000-00000000000f
# ╟─040e1000-0000-4000-8000-000000000070
# ╟─040e0001-0000-4000-8000-000000000007
# ╟─040e0002-0000-4000-8000-000000000004
# ╟─040e0002-0000-4000-8000-000000000006
