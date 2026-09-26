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
using Statistics: mean, std, median

# ╔═╡ d3054785-9e00-4094-a491-088ce63be9dc
md"""
# Photon-Counting CT Virtual Monoenergetic Imaging

A photon-counting CT scan of a Gammex 472 multi-energy phantom, taken all the way to
virtual monoenergetic images (VMIs). The scanner is the Siemens NAEOTOM Alpha model of the
basis-spectral-denoising study: a CdTe detector that sorts every photon into four energy windows
(thresholds 20 / 35 / 55 / 70 keV), a 140 kVp, 0.5 s axial acquisition at the dose of the physical
Gammex scan, and every view integrated over the arc the gantry turns while it is read.

The notebook shows three things:

1. **The acquisition.** Each `simulate!` call returns the four corrected energy-bin sinograms
   and the air count of every ray in every bin, `I0[col, row, bin]`. The bowtie makes that
   air response vary across the fan, so the spectral model is resolved per ray. The phantom is
   projected once; the noise-free reference, a calibration draw and the measured draw all reuse
   that projection.
2. **The VMI chain.** `spectral_basis(ws; I0)` builds the model the simulation applied, and one
   `vmi_pipeline` call turns the four bins into VMIs at 40, 70, 100 and 140 keV, with the
   denoiser, the reconstruction windows and the spectral pair of basis-spectral-denoising.
3. **The check.** Every calcium and iodine rod is compared with its theoretical HU from
   first-principles attenuation, and with the same chain applied to the noise-free draw. The
   solid-water background gives the HU accuracy and the noise at each energy.
"""

# ╔═╡ f2798d62-3509-4cc4-a24f-39ace8bb5a9e
md"""
## Pipeline

```
simulate!(…; keep_projection = true)   noise-free draw; keeps the projection
simulate!(…; projection)               calibration and measured draws: only the noise is new
  →  4 corrected bins h_k = −log(y_k / I0_k)  +  I0[col, row, bin]
  →  spectral_basis(ws; I0)            per-ray absolute response Φ[col, row, E, k]

calibration draw  →  spectral_pair     →  pair_basis (E*, β), composite_energy
measured draw     →  vmi_pipeline(; denoiser = SpectralHYPR(), fbp_filter = PairFilter(…),
                                     pair_basis, composite_energy)
                       projection HYPR-LR on the counts of each detector row
                       K-channel maximum-likelihood decomposition → iodine + water sinograms
                       FDK of the composite and its complement, each with its own window
                       ACNR on the complement
                       image HYPR-LR on the complement
                       VMI synthesis at 40 / 70 / 100 / 140 keV
noise-free draw   →  the same reconstruction without denoising → the reference
```

All four bins enter the decomposition as separate measurements; nothing is merged into
"low" and "high". Every setting is the one the basis-spectral-denoising study settled on for
the NAEOTOM Alpha.
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
## The Acquisition
"""

# ╔═╡ 040e1000-0000-4000-8000-000000000100
md"""
### Phantom

The Gammex 472: a 33 cm solid-water body with seven calcium rods (50 to 600 mg/mL) on an
inner ring and seven iodine rods (2 to 20 mg/mL) on an outer ring, 28 mm each. The label mask
(512 × 512 at 35 cm, 16 slices over 1 cm) goes to the GPU; its material table stays on the host.
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
### Scanner

The NAEOTOM Alpha geometry: 0.275 × 0.322 mm native CdTe dexels read out 2 × 2 binned, an arc
detector with the quarter-detector offset, a 50 cm scan field behind the large-body bowtie,
and a 1.6 mm CdTe sensor with 10 keV energy resolution, charge sharing and 5 ns dead time.
Pile-up is simulated and corrected, and so is scatter; the counts are exact Poisson. These are
the values of the basis-spectral-denoising and semmd-bayesian model. See the
[scanners page](../../scanners/) for where each value comes from.
"""

# ╔═╡ 2c157064-8567-450b-bc08-c2606084a77f
scanner = let
    native_col_mm, native_row_mm, binning = 0.275, 0.322, 2
    sid, sdd = 610.0, 1113.0
    col_iso = native_col_mm * binning / (sdd / sid)     # binned pixel at isocentre, mm
    row_iso = native_row_mm * binning / (sdd / sid)
    BS.PCCTScanner(
        source_to_isocenter = sid,
        source_to_detector = sdd,
        detector_rows = 144,
        detector_cols = ceil(Int, 500.0 / col_iso),     # covers the 50 cm scan field
        detector_row_size = row_iso,
        detector_col_size = col_iso,
        detector_shape = :arc,
        detector_col_offset = 0.25,                     # quarter-detector offset
        focal_spot_width = 0.4,
        focal_spot_length = 0.5,
        target_angle = 7.0,
        gantry_rotation_time = 0.5,
        scan_diameter = 500.0,
        gantry_aperture = 820.0,
        flat_filter_material = :aluminum,
        flat_filter_thickness = 3.0,
        bowtie_filter = :large_body,
        detector_material = :cdte,
        detector_depth = 1.6,
        fill_factor_row = 0.95,
        fill_factor_col = 0.95,
        energy_thresholds = [20.0, 35.0, 55.0, 70.0],   # four energy windows, keV
        energy_resolution = 10.0,
        charge_sharing_fwhm = 0.08,
        dead_time_ns = 5.0,
        pixel_mode = :standard,
        native_dexel_col_mm = native_col_mm,
        native_dexel_row_mm = native_row_mm,
        binning_factor = binning,
        pileup_correction = true,      # model-based inverse of the pile-up migration
        scatter_correction = true,     # scatter re-estimated and subtracted per bin
        noise_reduction = 0.0,         # exact Poisson counts
    )
end

# ╔═╡ 040e1000-0000-4000-8000-000000000102
md"""
### Protocol and Dose

140 kVp, 0.5 s rotation, 1200 views, 5 mm collimation, 0.9 mm titanium added: the physical
Gammex scan, which ran at 174 mA and reported a CTDIvol of 10.12 mGy. A tube's output per mA is
its own and the simulator's source is generic, so the tube current is scaled by one factor,
`DOSE_SCALE`, chosen so that the simulated CTDIvol (BasisSimulator's Monte Carlo dose in the
32 cm body phantom) equals the physical scan's. The noise is then that of the dose the physical
scan delivered.
"""

# ╔═╡ 04b00000-0000-4000-8000-000000000001
"The NAEOTOM Alpha's 140 kVp axial protocol at tube current `mA`."
alpha_protocol(mA) = BS.CTProtocol(
    kVp = 140,
    mA = mA,
    views = 1200,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Ti", 0.9)],
);

# ╔═╡ 04b00000-0000-4000-8000-000000000002
# The tube-current scale that makes the simulated CTDIvol equal the physical scan's
DOSE_SCALE = let physical = (mA = 174.0, ctdi_vol_mGy = 10.12)
    p = alpha_protocol(physical.mA)
    physical.ctdi_vol_mGy / BS.compute_dose(BS.dose_source(scanner, p), p).ctdi_vol_mGy
end

# ╔═╡ f9c0af7a-addd-4249-96fb-b9078765fbd1
protocol = alpha_protocol(174.0 * DOSE_SCALE);

# ╔═╡ 040e1000-0000-4000-8000-000000000103
md"""
### Simulation and Reconstruction Options

The physics switches of the basis-spectral-denoising Alpha model. Quantum noise, Compton
scatter and the focal-spot blur are on. The switches turned off model a scintillator (fill
factor, optical crosstalk, the scintillator efficiency table, lag): the CdTe detector's
efficiency, fill and charge sharing are already in its Monte Carlo response. The heel effect is off.

**View integration.** A detector integrates while the gantry turns, so each view reads the
transmitted intensity averaged over the arc it sweeps: a blur of the object, largest far from
the isocentre, that the noise, counted once per view, does not share. `view_samples = 5`
samples that arc with five sub-views per view (the midpoint rule; at the edge of a 35 cm field
they are 0.2 mm apart, a third of a pixel), and `view_arc = 1.0` because a photon-counting
detector reads for the whole view period. Without it the simulated signal is sharper, relative
to its noise, than a physical scanner's.

**Three draws, one projection.** Everything before the noise (the projection through the
phantom, sub-views included) is the same for every draw, so it is computed once, on the
noise-free draw (`keep_projection = true`), and every other draw reuses it (`projection`) with
only its own noise drawn. This is bit-identical to simulating each draw afresh.

| draw | noise | seed | used for |
|---|---|---:|---|
| `reference` | off | — | the noise-free expectation every result is compared with |
| `calibration` | on | 9234 | fixing the spectral pair, never measured |
| `measured` | on | 1234 | every result below |

The reconstruction grid is the physical scan's clinical series: 512 × 512 over 35 cm, twelve
0.4 mm slices.
"""

# ╔═╡ 2d65a0c0-b25d-41ad-9cd3-e7a2d08a2482
"The Alpha model's physics switches for one draw."
draw_options(; seed, use_noise) = BS.SimOptions(
    seed = seed,
    use_noise = use_noise,
    projector = :dd_fast,
    use_fill_factor = false,
    use_detector_efficiency = false,
    use_optical_crosstalk = false,
    use_focal_spot = true,
    use_heel_effect = false,
    use_lag = false,
    view_samples = 5,     # sub-views per view: the arc the gantry turns while a view is read
    view_arc = 1.0,       # a photon-counting detector integrates for the whole view period
);

# ╔═╡ 04b00000-0000-4000-8000-000000000003
# basis-spectral-denoising's seeds: draw r of the Gammex has seed 1234 + 1000 (r − 1); its
# calibration draw is draw 9
DRAWS = (
    reference = (seed = 1234, use_noise = false),
    calibration = (seed = 1234 + 1000 * (9 - 1), use_noise = true),
    measured = (seed = 1234, use_noise = true),
);

# ╔═╡ 08cbc6fd-3c7c-432f-99e5-b220f8fe7fde
recon_opts = BS.ReconOptions(
    matrix_size = (512, 512, 12),
    fov_cm = 35.0,
    z_cm = 12 * 0.04,                 # twelve 0.4 mm slices
);

# ╔═╡ 5ecd97c6-ad47-4558-886d-22ed45eda97d
md"""
### Simulate: `simulate!` and `spectral_basis`

`simulate!` runs the whole photon-counting detector model on the GPU: polychromatic forward
projection of every sub-view through the Monte Carlo detector response, scatter injection,
Poisson counts in every bin, pile-up migration, then pile-up and scatter correction. It returns

- `pcct_sino.bins`: four corrected sinograms ``h_k = -\log(y_k / I_{0,k})``, `[col, row, view]`;
- `I0_bins`: the air count of every ray in every bin, `[col, row, bin]`;
- `projection`, with `keep_projection = true`: everything before the noise, for the next draw;
- `dose`: the `DoseReport` of the beam.

`spectral_basis(ws; I0)` is read from the same workspace: the response the simulation applied,
passed through each ray's own bowtie transmission, so the decomposition inverts exactly what
was simulated and needs no calibration scan. It checks that the response sums to `I0` in every
ray and bin and refuses a basis that does not. The model does not depend on the noise, so one
basis serves every draw.
"""

# ╔═╡ 4315ef69-aa2f-4ee0-a13b-c65e01fb87ce
acquisition = let
    projection = nothing
    model = nothing
    channels = Dict{Symbol, Vector{Array{Float32, 3}}}()
    seconds = Dict{Symbol, Float64}()
    for name in (:reference, :calibration, :measured)     # the noise-free draw first: it projects
        opts = draw_options(; DRAWS[name]...)
        t0 = time()
        ws = BS.create_workspace(scanner, protocol, opts, recon_opts, phantom)
        try
            # the dose and the projection do not depend on the noise: both come from the first draw
            sim = BS.simulate!(ws, phantom, protocol, opts; capture_raw_counts = false,
                report_dose = projection === nothing,
                keep_projection = projection === nothing, projection = projection)
            projection === nothing && (projection = sim.projection)
            channels[name] = [Array(b) for b in sim.pcct_sino.bins]   # the four corrected bins
            if model === nothing
                I0 = Float32.(Array(sim.I0_bins))                     # [n_cols, n_rows, n_bins]
                model = (
                    I0 = I0,
                    basis = BS.spectral_basis(ws; I0 = I0),           # ray-resolved spectral model
                    geom = ws.geom,
                    dose = sim.dose,
                )
            end
        finally
            BS.release_backend!(ws)
        end
        seconds[name] = time() - t0
    end
    (; channels, seconds, model...)
end;

# ╔═╡ 04a10000-0000-4000-8000-000000000001
let
    a = acquisition
    nc, nr, K = size(a.I0)
    c, r = (nc + 1) ÷ 2, (nr + 1) ÷ 2
    centre = round.(Int, a.I0[c, r, :])
    edge = [a.I0[1, r, k] / a.I0[c, r, k] for k in 1:K]
    fmt(x) = string(round(x; sigdigits = 3))
    s(name) = string(round(a.seconds[name]; digits = 1), " s")
    Markdown.parse("""
    | quantity | value |
    |---|---|
    | sinogram per bin | $(nc) columns × $(nr) rows × $(size(first(a.channels[:measured]), 3)) views |
    | air counts per ray and view, centre ray (bins 1 to 4) | $(join(centre, " / ")) |
    | air counts at the fan edge / centre (bins 1 to 4) | $(join(fmt.(edge), " / ")) |
    | spectral basis | $(size(a.basis.Φ, 3)) energies, ray-resolved = $(a.basis.ray_resolved) |
    | largest relative mismatch of Σ Φ and I0 | $(fmt(a.basis.I0_relerr)) |
    | tube current | $(round(protocol.mA; digits = 1)) mA (174 mA × DOSE_SCALE = $(round(DOSE_SCALE; digits = 3))) |
    | CTDIvol (32 cm body phantom) | $(round(a.dose.ctdi_vol_mGy; digits = 2)) mGy |
    | wall time: noise-free draw (projects) / calibration / measured | $(s(:reference)) / $(s(:calibration)) / $(s(:measured)) |
    """)
end

# ╔═╡ 04a10000-0000-4000-8000-000000000002
md"""
### The Four Energy Windows

Left: what each bin detects on the central ray, the absolute response the decomposition
uses, as a function of the energy the photon arrived with. The windows overlap because the
10 keV energy resolution, charge sharing and K-fluorescence escape let a photon register below
its true energy; the spikes are tungsten's characteristic lines. Right: the air count across the fan. The bowtie cuts
the flux toward the edge and hardens what is left, so the high bins fall off less than the
low ones. One scalar `I0` per bin could not describe this, which is why the basis is
resolved per ray.
"""

# ╔═╡ 04a10000-0000-4000-8000-000000000003
let
    a = acquisition
    nc, nr, K = size(a.I0)
    c, r = (nc + 1) ÷ 2, (nr + 1) ÷ 2
    E = Float64.(a.basis.E)
    dE = median(diff(E))
    colors = Mke.cgrad(:viridis, K; categorical = true)
    t = Int.(scanner.energy_thresholds)
    labels = [k < K ? "bin $(k): $(t[k])–$(t[k + 1]) keV" : "bin $(k): above $(t[k]) keV" for k in 1:K]

    fig = Mke.Figure(size = (1180, 520))
    ax1 = Mke.Axis(fig[1, 1]; title = "What each bin counts", subtitle = "central ray, one view",
        xlabel = "Incident photon energy (keV)", ylabel = "Counts per keV of incident energy",
        titlesize = 24, subtitlesize = 18)
    for k in 1:K
        Mke.lines!(ax1, E, Float64.(a.basis.Φ[c, r, :, k]) ./ dE; color = colors[k], linewidth = 2.5,
            label = labels[k])
    end
    Mke.axislegend(ax1; position = :rt, labelsize = 15)

    γ = ((1:nc) .- (nc + 1) / 2) .* a.geom.pixel_size ./ a.geom.SAD .* (180 / π)
    ax2 = Mke.Axis(fig[1, 2]; title = "Air count across the fan", subtitle = "relative to the central ray",
        xlabel = "Fan angle (degrees)", ylabel = "I0 / I0(centre)", titlesize = 24, subtitlesize = 18)
    for k in 1:K
        Mke.lines!(ax2, γ, Float64.(a.I0[:, r, k]) ./ a.I0[c, r, k]; color = colors[k], linewidth = 2.5,
            label = "bin $(k)")
    end
    Mke.ylims!(ax2, 0, 1.05)
    Mke.axislegend(ax2; position = :cb, labelsize = 15, orientation = :horizontal)
    Mke.save(joinpath(@__DIR__, "..", "assets", "pcct_vmi_energy_windows.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ dc8a8352-5598-4cdd-952f-3d77367850e9
md"""
## The VMI Chain

### The Denoiser

Generalized HYPR-LR in both domains, `BS.SpectralHYPR()`:

- **projection domain** (`ProjectionHYPR`): within each detector row, each ray's split of its
  total count across the four bins is pooled over a 3 × 3 (column × view) window whose view
  neighbours are two views apart (`view_stride = 2`), so the odd and even views stay
  independent. Neighbours are weighted by how likely their total count is under the ray's own,
  and each ray keeps its own total. The count dispersion of every bin is measured from the air rays.
- **image domain** (`ImageHYPR`): the reconstructed basis pair is rewritten as the minimum-noise
  VMI (the composite) and a complement whose noise is uncorrelated with it. The composite is
  kept as reconstructed, so its resolution and noise texture are FDK's; the complement, which
  carries only spectral information, is pooled within its slice with likelihood weights on the
  composite, over the window width (3 to 41 pixels) with the least estimated risk. Every noise
  level comes from the difference of the odd- and even-view reconstructions.

Nothing in the chain averages detector rows or slices. These are `SpectralHYPR()`'s defaults,
the chain of basis-spectral-denoising.
"""

# ╔═╡ 04a20000-0000-4000-8000-000000000001
HYPR_CHAIN = BS.SpectralHYPR()

# ╔═╡ 04a20000-0000-4000-8000-000000000002
VMI_ENERGIES = (40, 70, 100, 140);

# ╔═╡ 04a20000-0000-4000-8000-000000000006
# The decomposition and reconstruction settings of the chain; ACNR is added on the measured draw
VMI_CHAIN = (method = :nchannel, controls = BS.NChannelControls(), use_tlbf = false, antialias = true);

# ╔═╡ 04b00000-0000-4000-8000-000000000010
md"""
### The Reconstruction Windows

The water and iodine images are the wrong pair to filter differently: their noise is strongly
anti-correlated, and a VMI near the minimum-noise energy is quiet only where both were filtered
alike. `BS.PairFilter` therefore filters the uncorrelated pair: the composite, whose window sets
the resolution and noise texture at the minimum-noise energy, and the complement, whose window
sets how that changes away from it. Each window is an apodized ramp,
``W(f) = \exp(-(f/f_c)^p)`` on the grid-Nyquist axis, sampled at 11 knots as a `CustomFilter`.

basis-spectral-denoising fitted the Alpha's pair once, to the physical Gammex scan's MTF at every
energy and its NPS shape (VMI, Br36f, QIR 0, 0.4 mm, 350 mm): composite ``f_c = 0.5``, ``p = 2``;
complement ``f_c = 0.45``, ``p = 6``. A window is defined on the frequency axis of the grid it is
applied on, so on a grid other than the fitted one (512² over 35 cm) its argument is rescaled by
the ratio `r` of the grids' bandlimits (`BS.grid_bandlimit`). This notebook uses the fitted grid, so `r = 1`.
"""

# ╔═╡ 04b00000-0000-4000-8000-000000000011
ALPHA_WINDOWS = (composite = (fc = 0.5, p = 2.0), complement = (fc = 0.45, p = 6.0));

# ╔═╡ 04b00000-0000-4000-8000-000000000012
PAIR_FILTER = let knots = Tuple(range(0.0, 1.0, length = 11))
    # the grid the windows were fitted on: the Alpha's clinical series, 512² over 35 cm
    fitted = BS.CTGeometry(scanner; n_angles = protocol.views, fov_cm = 35.0, z_cm = 0.48,
        collimation_mm = protocol.collimation_mm)
    r = BS.grid_bandlimit(acquisition.geom, recon_opts.matrix_size) /
        BS.grid_bandlimit(fitted, (512, 512, 12))
    window(q) = BS.CustomFilter(knots, Tuple(round(exp(-(x * r / q.fc)^q.p), digits = 5) for x in knots))
    (filter = BS.PairFilter(window(ALPHA_WINDOWS.composite), window(ALPHA_WINDOWS.complement)), r = r)
end;

# ╔═╡ 04b00000-0000-4000-8000-000000000020
md"""
### The Spectral Pair, Fixed from the Calibration Draw

`spectral_pair` measures, from the odd- and even-view halves of a reconstruction, the
minimum-noise energy ``E^*`` and the complement's coefficient ``\beta``. Near its minimum the
VMI noise hardly changes with energy, so the argmin of one acquisition is itself noise and would
move from draw to draw; it is a property of the scanner and protocol, so it is measured once, on
the calibration draw, and fixed for every other draw, the noise-free one included:

- `pair_basis = (Estar, β)`: from the calibration draw decomposed without denoising and
  reconstructed with a standard soft-tissue window (`SoftFilter`); the pair the windows act on;
- `composite_energy`: the minimum-noise energy of the same draw after the projection-domain
  HYPR, which moves it; the composite ACNR and the image HYPR act on.
"""

# ╔═╡ 04b00000-0000-4000-8000-000000000021
PAIR_BASIS = let ch = acquisition.channels[:calibration], I0 = acquisition.I0
    decompose(channels) = BS.vmi_pipeline(; channels, basis = acquisition.basis,
        geom = acquisition.geom, to_backend = to_gpu, matrix_size = recon_opts.matrix_size,
        vmi_energies = VMI_ENERGIES, keep_sinograms = true, use_acnr = false, VMI_CHAIN...).sinograms
    p = HYPR_CHAIN.projection
    d_none = decompose(ch)
    d_proj = decompose(BS.hypr_lr(ch, I0; kernel = p.kernel, dispersion = BS.estimate_dispersion(ch, I0),
        view_stride = p.view_stride, to_backend = to_gpu))
    sp(d; kw...) = BS.spectral_pair(d.water, d.iodine, acquisition.geom, recon_opts.matrix_size;
        filter = BS.SoftFilter(), antialias = VMI_CHAIN.antialias, to_backend = to_gpu, kw...)
    x = sp(d_none)
    c = sp(d_proj; basis = (Estar = x.Estar, β = x.β))
    (Estar = x.Estar, β = x.β, composite = c.Estar)
end

# ╔═╡ 04a20000-0000-4000-8000-000000000003
md"""
### `vmi_pipeline`

One call runs the whole chain on the measured draw. Every detector row is kept and
reconstructed onto the twelve-slice grid; the decomposition is the K-channel maximum-likelihood
estimator (`method = :nchannel`) with its published controls; ACNR (on by default) acts on the
complement only, before the image HYPR. The noise-free draw goes through the same call without
the denoiser and ACNR: the reference, reconstructed exactly as its noisy counterpart.
"""

# ╔═╡ 04a20000-0000-4000-8000-000000000004
vmi = BS.vmi_pipeline(;
    channels = acquisition.channels[:measured],
    basis = acquisition.basis,
    geom = acquisition.geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = VMI_ENERGIES,
    denoiser = HYPR_CHAIN,
    fbp_filter = PAIR_FILTER.filter,
    pair_basis = (Estar = PAIR_BASIS.Estar, β = PAIR_BASIS.β),
    composite_energy = PAIR_BASIS.composite,
    VMI_CHAIN...,
);

# ╔═╡ 04b00000-0000-4000-8000-000000000022
reference = BS.vmi_pipeline(;
    channels = acquisition.channels[:reference],
    basis = acquisition.basis,
    geom = acquisition.geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = VMI_ENERGIES,
    fbp_filter = PAIR_FILTER.filter,
    pair_basis = (Estar = PAIR_BASIS.Estar, β = PAIR_BASIS.β),
    composite_energy = PAIR_BASIS.composite,
    use_acnr = false,
    VMI_CHAIN...,
);

# ╔═╡ 04a20000-0000-4000-8000-000000000005
let
    s = vmi.settings
    q = vmi.quality
    pct(x) = string(round(100x; digits = 3), " %")
    est = s.denoiser.image_estimates
    Markdown.parse("""
    What the chain fixed and measured:

    | quantity | value |
    |---|---|
    | channels decomposed | $(s.n_channels) |
    | count dispersion per bin (variance / mean, from the air rays) | $(join(string.(round.(s.denoiser.dispersion; digits = 3)), " / ")) |
    | pair the windows act on, from the calibration draw: E* and β | $(round(Int, PAIR_BASIS.Estar)) keV, β = $(round(PAIR_BASIS.β; sigdigits = 3)) |
    | composite energy, from the calibration draw after projection HYPR | $(round(Int, PAIR_BASIS.composite)) keV |
    | window bandlimit ratio r (this grid / the fitted grid) | $(round(PAIR_FILTER.r; digits = 4)) |
    | image HYPR window chosen for the complement | $(est.window) pixels |
    | rays hitting an iodine / water bound | $(pct(q.frac_bound_iodine)) / $(pct(q.frac_bound_water)) |
    | rays not converged | $(pct(q.frac_not_converged)) |
    | ACNR | $(s.acnr.passes) passes on the $(s.acnr.on), beta_max = $(s.acnr.beta_max) |
    | VMI stack | $(join(size(vmi.vmis), " × ")) (x, y, slice, energy) |
    """)
end

# ╔═╡ 040e0002-0000-4000-8000-000000000003
md"""
### Basis Images and VMIs

The chain's basis pair: iodine as a mass density (mg/mL) and water as a density (g/mL), the
central slice. Every VMI is synthesized from this one pair with the monoenergetic
attenuation of water and iodine; no energy-dependent filtering is applied.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000003
let
    z = (size(vmi.images.water, 3) + 1) ÷ 2
    fig = Mke.Figure(size = (1180, 560))
    panels = (
        ("Iodine basis", "Iodine (mg/mL)", 1000 .* vmi.images.iodine[:, :, z], (-5.0, 25.0)),
        ("Water basis", "Water (g/mL)", vmi.images.water[:, :, z], (0.0, 1.5)),
    )
    for (column, (title, label, image, range)) in pairs(panels)
        axis = Mke.Axis(fig[1, 2column - 1]; title, aspect = Mke.DataAspect(), titlesize = 28)
        Mke.heatmap!(axis, image; colormap = :viridis, colorrange = range)
        Mke.hidedecorations!(axis)
        Mke.Colorbar(fig[1, 2column]; colormap = :viridis, colorrange = range, label,
            width = 16, labelsize = 20, ticklabelsize = 16)
    end
    fig
end

# ╔═╡ 040e0001-0000-4000-8000-000000000004
let
    z = (size(vmi.vmis, 3) + 1) ÷ 2
    window = (-200, 500)
    fig = Mke.Figure(size = (1180, 1180))
    for (index, energy) in pairs(vmi.energies)
        axis = Mke.Axis(fig[(index - 1) ÷ 2 + 1, (index - 1) % 2 + 1];
            title = "$(energy) keV VMI", aspect = Mke.DataAspect(), titlesize = 32)
        Mke.heatmap!(axis, vmi.vmis[:, :, z, index]; colormap = :grays, colorrange = window)
        Mke.hidedecorations!(axis)
    end
    Mke.Colorbar(fig[1:2, 3]; colormap = :grays, colorrange = window, label = "HU",
        width = 16, labelsize = 22, ticklabelsize = 18)
    Mke.save(joinpath(@__DIR__, "..", "assets", "pcct_vmi_projection_grid.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 040e0001-0000-4000-8000-000000000008
md"""
## Results

The phantom's labels are resampled onto the reconstruction grid (`resample_to_recon`). Each
rod is measured in a disk of 60 % of its radius, and the background in the solid water
eroded 12 pixels (8 mm) away from every edge. Means pool all twelve slices; the noise is the
standard deviation within one slice, averaged (RMS) over the slices. The theoretical HU of
every material is ``1000\,(\mu_E - \mu_{E,\mathrm{water}})/\mu_{E,\mathrm{water}}`` from its
composition. Every quantity is also measured on the noise-free reference, the same
decomposition and reconstruction without noise (and so without the denoiser): its error against
theory is the bias of the basis model itself, and the measured draw's departure from it is what
the noise and the denoising add. The noise of the measured draw is the standard deviation of its
difference from the reference, in which the phantom's structure and any deterministic artefact
cancel.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000002
pcct_results = let
    energies = Float64.(vmi.energies)
    nz = size(vmi.vmis, 3)
    labels = BS.resample_to_recon(phantom_cpu, acquisition.geom, recon_opts.matrix_size)
    water_mask = cat((collect(BS.erode_mask_2d(labels[:, :, z] .== UInt8(BS.REGION_SOLID_WATER);
        erode_px = 12.0)) for z in 1:nz)...; dims = 3)
    theory_hu(material, E) = let μw = BS.compute_μ_at_energy(BS.XA.Materials.water, E)
        1000 * (BS.compute_μ_at_energy(material, E) - μw) / μw
    end
    material(label) = phantom_cpu.materials[Int(label) + 1]

    # a disk of 60 % of the rod's radius about its centroid, in every slice
    function rod_roi(label)
        pixels = findall(==(UInt8(label)), labels[:, :, (nz + 1) ÷ 2])
        cx, cy = mean(p -> p[1], pixels), mean(p -> p[2], pixels)
        radius = 0.6 * sqrt(length(pixels) / π)
        [(i - cx)^2 + (j - cy)^2 <= radius^2 for i in axes(labels, 1), j in axes(labels, 2), _ in 1:nz]
    end
    groups = (   # the region labels of the rods, BS.REGION_CA_50 … and BS.REGION_I_2_0 …
        Ca = (labels = UInt8.(10:16), names = ["50", "100", "200", "300", "400", "500", "600"] .* " mg/mL"),
        I = (labels = UInt8.(20:26), names = ["2.0", "2.5", "5.0", "7.5", "10.0", "15.0", "20.0"] .* " mg/mL"),
    )
    rods = Dict{Symbol, NamedTuple}()
    for (group, g) in pairs(groups)
        rod_labels = g.labels
        measured = zeros(length(rod_labels), length(energies))
        noise_free = similar(measured)
        theoretical = similar(measured)
        for (row, label) in pairs(rod_labels)
            roi = rod_roi(label)
            for (column, E) in pairs(energies)
                measured[row, column] = mean(vmi.vmis[:, :, :, column][roi])
                noise_free[row, column] = mean(reference.vmis[:, :, :, column][roi])
                theoretical[row, column] = theory_hu(material(label), E)
            end
        end
        # measured against theoretical at each energy: least-squares slope, intercept and R²
        fits = map(axes(measured, 2)) do column
            x, y = theoretical[:, column], measured[:, column]
            slope = sum((x .- mean(x)) .* (y .- mean(y))) / sum(abs2, x .- mean(x))
            intercept = mean(y) - slope * mean(x)
            r2 = 1 - sum(abs2, y .- (intercept .+ slope .* x)) / sum(abs2, y .- mean(y))
            (; slope, intercept, r2)
        end
        rods[group] = (names = g.names, measured, noise_free, theoretical, fits)
    end
    # per-slice standard deviation in the water ROI, RMS over the slices
    slice_noise(volume, column) = sqrt(mean(abs2,
        [std(volume[:, :, z, column][water_mask[:, :, z]]) for z in 1:nz]))
    difference = vmi.vmis .- reference.vmis
    (
        energies, labels, water_mask, rods,
        water_mean = [mean(vmi.vmis[:, :, :, c][water_mask]) for c in eachindex(energies)],
        water_reference = [mean(reference.vmis[:, :, :, c][water_mask]) for c in eachindex(energies)],
        water_theory = [theory_hu(material(BS.REGION_SOLID_WATER), E) for E in energies],
        water_noise = [slice_noise(vmi.vmis, c) for c in eachindex(energies)],
        water_noise_vs_reference = [slice_noise(difference, c) for c in eachindex(energies)],
        water_pixels_per_slice = count(water_mask) ÷ nz,
        finite = all(isfinite, vmi.vmis) && all(isfinite, reference.vmis),
    )
end;

# ╔═╡ 040e1000-0000-4000-8000-000000000040
md"""
### Water ROI

The eroded solid-water region (red) on the central slice of the 70 keV VMI, and its mean HU
at each energy, measured (bars) and on the noise-free reference (black diamonds). The phantom's
solid water is modelled with the composition of water, so its theoretical HU is 0 at every
energy (dashed line).
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000005
let
    r = pcct_results
    z = (size(vmi.vmis, 3) + 1) ÷ 2
    overlay = Float32[value ? 1.0f0 : NaN32 for value in r.water_mask[:, :, z]]
    index70 = findfirst(==(70.0), r.energies)
    n = length(r.energies)
    fig = Mke.Figure(size = (1180, 580))
    axis = Mke.Axis(fig[1, 1]; title = "Eroded solid-water ROI", subtitle = "on the 70 keV VMI",
        aspect = Mke.DataAspect(), titlesize = 28, subtitlesize = 20)
    Mke.heatmap!(axis, vmi.vmis[:, :, z, index70]; colormap = :grays, colorrange = (-200, 500))
    Mke.heatmap!(axis, overlay; colormap = :reds, alpha = 0.5, nan_color = (:white, 0.0))
    Mke.hidedecorations!(axis)

    mean_axis = Mke.Axis(fig[1, 2]; title = "Solid-water mean HU", xlabel = "VMI energy (keV)",
        ylabel = "HU", xticks = (1:n, string.(Int.(r.energies))), titlesize = 28)
    Mke.barplot!(mean_axis, 1:n, r.water_mean; color = (:steelblue, 0.8), label = "measured")
    Mke.scatter!(mean_axis, 1:n, r.water_reference; color = :black, marker = :diamond,
        markersize = 16, label = "noise-free reference")
    Mke.hlines!(mean_axis, r.water_theory[1:1]; color = :black, linestyle = :dash)
    for (index, value) in pairs(r.water_mean)
        Mke.text!(mean_axis, index, value; text = "$(round(value; digits = 1)) HU",
            align = (:center, value >= 0 ? :bottom : :top), offset = (0, value >= 0 ? 12 : -12),
            fontsize = 16)
    end
    Mke.ylims!(mean_axis, -15, 15)
    Mke.axislegend(mean_axis; position = :lb, labelsize = 15)
    Mke.save(joinpath(@__DIR__, "..", "assets", "pcct_vmi_water_roi_check.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 040e1000-0000-4000-8000-000000000050
let
    r = pcct_results
    gap = maximum(abs, r.water_noise .- r.water_noise_vs_reference)
    Markdown.parse("""
    ### Noise versus Energy

    The noise in the same solid-water ROI, per 0.4 mm slice: the standard deviation of the
    difference between the measured VMI and the noise-free reference, in which the phantom's
    structure and any deterministic artefact cancel. The standard deviation of the measured VMI
    itself differs from it by at most $(round(gap; digits = 2)) HU at any energy, so the ROI holds
    noise only. The noise is highest at 40 keV, where the synthesis weights the iodine image most
    heavily and so amplifies its noise, and changes little from 70 keV up, where the VMI is mostly
    the composite and the water image.
    """)
end

# ╔═╡ 040e1000-0000-4000-8000-000000000051
let
    r = pcct_results
    n = length(r.energies)
    σ = r.water_noise_vs_reference
    fig = Mke.Figure(size = (880, 560))
    axis = Mke.Axis(fig[1, 1]; title = "Solid-water noise vs VMI energy",
        subtitle = "per-slice σ of measured − noise-free, 0.4 mm slices, 10.12 mGy",
        xlabel = "VMI energy (keV)", ylabel = "Noise σ (HU)",
        xticks = (1:n, string.(Int.(r.energies))), titlesize = 28, subtitlesize = 18)
    Mke.barplot!(axis, 1:n, σ; color = :tomato, strokecolor = :black, strokewidth = 1)
    for index in 1:n
        Mke.text!(axis, index, σ[index]; text = "σ = $(round(σ[index]; digits = 1))",
            align = (:center, :bottom), offset = (0, 6), fontsize = 18)
    end
    Mke.ylims!(axis, 0, 1.25maximum(σ))
    Mke.save(joinpath(@__DIR__, "..", "assets", "pcct_vmi_water_noise_vs_energy.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 040e1000-0000-4000-8000-000000000060
md"""
### Rods against Theory

Solid lines: measured HU of each calcium and iodine rod. Dashed lines: its theoretical HU.

The decomposition represents every voxel as iodine plus water. The solid water and the iodine
rods are close to that basis; calcium is not exactly a combination of iodine and water at every
energy, so its VMIs carry a small, energy-dependent model error on top of the noise. The per-rod
table below separates the two with the noise-free reference.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000006
let
    r = pcct_results
    fig = Mke.Figure(size = (1180, 580))
    panels = (
        (group = :Ca, title = "Calcium rods", subtitle = "50–600 mg/mL",
         cmap = Mke.cgrad(:Oranges, 9; categorical = true)),
        (group = :I, title = "Iodine rods", subtitle = "2–20 mg/mL",
         cmap = Mke.cgrad(:GnBu, 9; categorical = true)),
    )
    for (column, panel) in pairs(panels)
        axis = Mke.Axis(fig[1, column]; title = panel.title, subtitle = panel.subtitle,
            xlabel = "VMI energy (keV)", ylabel = "HU", xticks = r.energies,
            titlesize = 28, subtitlesize = 20)
        data = r.rods[panel.group]
        for index in eachindex(data.names)
            color = panel.cmap[index + 2]
            Mke.scatterlines!(axis, r.energies, data.measured[index, :]; color, linewidth = 2.5,
                markersize = 9, label = data.names[index])
            Mke.lines!(axis, r.energies, data.theoretical[index, :]; color, linewidth = 1.6,
                linestyle = :dash)
        end
        Mke.ylims!(axis; low = 0)
        Mke.axislegend(axis; position = :rt, labelsize = 15)
    end
    Mke.save(joinpath(@__DIR__, "..", "assets", "pcct_vmi_vs_theoretical.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 040e1000-0000-4000-8000-000000000070
md"""
### Linear Regression

Measured rod HU against theoretical HU at each energy, with the least-squares line. The
dashed line is identity.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000007
let
    r = pcct_results
    colors = Mke.cgrad(:plasma, length(r.energies) + 1; categorical = true)
    fig = Mke.Figure(size = (1180, 600))
    for (column, group) in enumerate((:Ca, :I))
        data = r.rods[group]
        axis = Mke.Axis(fig[1, column];
            title = group == :Ca ? "Calcium rods" : "Iodine rods",
            xlabel = "Theoretical HU", ylabel = "Measured HU", titlesize = 28)
        low = min(0.0, minimum(data.measured), minimum(data.theoretical))
        high = 1.05max(maximum(data.measured), maximum(data.theoretical))
        Mke.lines!(axis, [low, high], [low, high]; color = :black, linestyle = :dash, linewidth = 2,
            label = "identity")
        for (index, energy) in pairs(r.energies)
            x = data.theoretical[:, index]
            fit = data.fits[index]
            ends = collect(extrema(x))
            Mke.scatter!(axis, x, data.measured[:, index]; color = colors[index], markersize = 11)
            Mke.lines!(axis, ends, fit.intercept .+ fit.slope .* ends; color = colors[index],
                linewidth = 2,
                label = "$(Int(energy)) keV: slope $(round(fit.slope; digits = 3)), " *
                        "R² $(round(fit.r2; digits = 4))")
        end
        Mke.axislegend(axis; position = :lt, labelsize = 14)
    end
    Mke.save(joinpath(@__DIR__, "..", "assets", "pcct_vmi_regression.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 04a30000-0000-4000-8000-000000000001
let
    r = pcct_results
    relative(g, field) = maximum(abs, (getfield(r.rods[g], field) .- r.rods[g].theoretical) ./ r.rods[g].theoretical)
    pct(x) = round(100x; digits = 1)
    Markdown.parse("""
    ### Per-Rod Error

    Measured minus theoretical HU for every rod at every energy, and in brackets the same for the
    noise-free reference: the bias of the iodine–water basis model itself, which no amount of dose
    would remove. The difference between the two is what the noise and the denoiser add. The largest
    relative error of the measured draw is $(pct(relative(:I, :measured))) % over the iodine rods
    and $(pct(relative(:Ca, :measured))) % over the calcium rods; of the noise-free reference,
    $(pct(relative(:I, :noise_free))) % and $(pct(relative(:Ca, :noise_free))) %.
    """)
end

# ╔═╡ 04a30000-0000-4000-8000-000000000002
let
    r = pcct_results
    header = "| rod | " * join(["$(Int(E)) keV" for E in r.energies], " | ") * " |"
    rule = "|---|" * repeat("---:|", length(r.energies))
    rows = String[]
    for (group, name) in ((:Ca, "Ca"), (:I, "I"))
        data = r.rods[group]
        for (index, rod) in pairs(data.names)
            errors = data.measured[index, :] .- data.theoretical[index, :]
            biases = data.noise_free[index, :] .- data.theoretical[index, :]
            push!(rows, "| $(name) $(rod) | " *
                join(["$(round(e; digits = 1)) ($(round(b; digits = 1)))" for (e, b) in zip(errors, biases)], " | ") * " |")
        end
    end
    Markdown.parse(join([header, rule, rows...], "\n"))
end

# ╔═╡ 040e0002-0000-4000-8000-000000000004
verification = let
    r = pcct_results
    checks = NamedTuple[]
    addcheck(name, value, pass) = push!(checks, (; name, value, pass))
    addcheck("spectral basis reproduces I0 in every ray and bin (max rel. error)",
        round(acquisition.basis.I0_relerr; sigdigits = 2), acquisition.basis.I0_relerr < 5e-5)
    addcheck("simulated CTDIvol equals the physical scan's 10.12 mGy",
        round(acquisition.dose.ctdi_vol_mGy; digits = 2), abs(acquisition.dose.ctdi_vol_mGy - 10.12) < 0.05)
    addcheck("all four bins decomposed", vmi.settings.n_channels, vmi.settings.n_channels == 4)
    addcheck("every VMI value finite (measured and reference)", r.finite, r.finite)
    addcheck("decomposition: fraction of rays not converged", vmi.quality.frac_not_converged,
        vmi.quality.frac_not_converged < 1e-3)
    water_error = maximum(abs, r.water_mean .- r.water_theory)
    addcheck("solid water within 10 HU of theory at every energy (largest error, HU)",
        round(water_error; digits = 1), water_error <= 10)
    reference_error = maximum(abs, r.water_reference .- r.water_theory)
    addcheck("noise-free solid water within 10 HU of theory at every energy (largest error, HU)",
        round(reference_error; digits = 1), reference_error <= 10)
    fits = [f for g in (:Ca, :I) for f in r.rods[g].fits]
    slopes = [f.slope for f in fits]
    addcheck("rod regression slope within 1 ± 0.1 at every energy (min, max)",
        (round(minimum(slopes); digits = 3), round(maximum(slopes); digits = 3)),
        all(s -> abs(s - 1) <= 0.1, slopes))
    r2 = minimum(f.r2 for f in fits)
    addcheck("rod regression R² at least 0.99 at every energy (smallest)", round(r2; digits = 4), r2 >= 0.99)
    passed = count(check -> check.pass, checks)
    rows = join(["| $(c.name) | $(c.value) | $(c.pass ? "pass" : "CHECK") |" for c in checks], "\n")
    Markdown.parse("""
    ### Verification: $(passed) of $(length(checks)) checks pass

    | check | value | result |
    |---|---:|:---:|
    $rows
    """)
end

# ╔═╡ 040e0002-0000-4000-8000-000000000006
let
    r = pcct_results
    fmt(v) = join(round.(v; digits = 1), " / ")
    Markdown.parse("""
    ## Summary

    ```julia
    ws    = BS.create_workspace(scanner, protocol, opts, recon_opts, phantom)
    sim   = BS.simulate!(ws, phantom, protocol, opts; keep_projection = true)  # then (…; projection)
    basis = BS.spectral_basis(ws; I0 = sim.I0_bins)          # ray-resolved, no calibration scan
    vmi   = BS.vmi_pipeline(; channels = sim.pcct_sino.bins, basis, geom = ws.geom,
                              to_backend, matrix_size, vmi_energies = (40, 70, 100, 140),
                              denoiser = BS.SpectralHYPR(), fbp_filter = PAIR_FILTER.filter,
                              pair_basis, composite_energy)
    ```

    Four photon-counting energy windows go into the decomposition as four measurements, and the
    spectral model is the one the simulation applied, ray by ray through the bowtie, so no
    calibration scan is involved. The acquisition integrates every view over five sub-views, and
    the noise-free reference, the calibration draw and the measured draw share one projection.

    At 40 / 70 / 100 / 140 keV the solid water reads $(fmt(r.water_mean)) HU (noise-free:
    $(fmt(r.water_reference)) HU; theory 0 HU), with a noise of $(fmt(r.water_noise_vs_reference)) HU
    per 0.4 mm slice at the physical scan's 10.12 mGy. Measured rod HU track theory at every energy
    (the regression slopes and R² in the verification table). The per-rod table separates each
    rod's error into the basis model's own bias, largest for calcium, which the iodine–water basis
    represents only approximately, and what the noise and the denoiser add to it.
    """)
end

# ╔═╡ Cell order:
# ╟─d3054785-9e00-4094-a491-088ce63be9dc
# ╟─f2798d62-3509-4cc4-a24f-39ace8bb5a9e
# ╟─3d515abe-f3d9-4ce5-96c7-bef7da9bf294
# ╟─171294a2-26bd-49e2-ac92-9df48ae5444f
# ╟─69358294-97f2-4782-94d7-c29c747c45f4
# ╟─9ae27110-5c47-442b-a98e-d137599570f2
# ╟─492bb299-678d-4e6f-8c21-1e9178cc2beb
# ╠═9f8d5cd4-147e-4359-95bc-cc096a53f0e7
# ╟─2ff539c9-a678-403c-b629-8068a332a0e9
# ╟─320e1b29-4ae3-4757-a2cb-d28b0aa3ec2d
# ╠═86c52e9e-7987-4504-93e6-128017f5e703
# ╟─551f84fe-d7b4-48f9-a475-0c63178a6ede
# ╟─59a5079b-a711-4f28-b3d6-665f0d91fb72
# ╟─040e1000-0000-4000-8000-000000000100
# ╠═939dcda3-9be5-46c8-aaa1-ded273e8cf04
# ╠═5248ba55-965a-41f7-845c-99616018b475
# ╟─040e1000-0000-4000-8000-000000000101
# ╠═2c157064-8567-450b-bc08-c2606084a77f
# ╟─040e1000-0000-4000-8000-000000000102
# ╠═04b00000-0000-4000-8000-000000000001
# ╠═04b00000-0000-4000-8000-000000000002
# ╠═f9c0af7a-addd-4249-96fb-b9078765fbd1
# ╟─040e1000-0000-4000-8000-000000000103
# ╠═2d65a0c0-b25d-41ad-9cd3-e7a2d08a2482
# ╠═04b00000-0000-4000-8000-000000000003
# ╠═08cbc6fd-3c7c-432f-99e5-b220f8fe7fde
# ╟─5ecd97c6-ad47-4558-886d-22ed45eda97d
# ╠═4315ef69-aa2f-4ee0-a13b-c65e01fb87ce
# ╟─04a10000-0000-4000-8000-000000000001
# ╟─04a10000-0000-4000-8000-000000000002
# ╟─04a10000-0000-4000-8000-000000000003
# ╟─dc8a8352-5598-4cdd-952f-3d77367850e9
# ╠═04a20000-0000-4000-8000-000000000001
# ╠═04a20000-0000-4000-8000-000000000002
# ╠═04a20000-0000-4000-8000-000000000006
# ╟─04b00000-0000-4000-8000-000000000010
# ╠═04b00000-0000-4000-8000-000000000011
# ╠═04b00000-0000-4000-8000-000000000012
# ╟─04b00000-0000-4000-8000-000000000020
# ╠═04b00000-0000-4000-8000-000000000021
# ╟─04a20000-0000-4000-8000-000000000003
# ╠═04a20000-0000-4000-8000-000000000004
# ╠═04b00000-0000-4000-8000-000000000022
# ╟─04a20000-0000-4000-8000-000000000005
# ╟─040e0002-0000-4000-8000-000000000003
# ╟─040e0001-0000-4000-8000-000000000003
# ╟─040e0001-0000-4000-8000-000000000004
# ╟─040e0001-0000-4000-8000-000000000008
# ╟─040e0001-0000-4000-8000-000000000002
# ╟─040e1000-0000-4000-8000-000000000040
# ╟─040e0001-0000-4000-8000-000000000005
# ╟─040e1000-0000-4000-8000-000000000050
# ╟─040e1000-0000-4000-8000-000000000051
# ╟─040e1000-0000-4000-8000-000000000060
# ╟─040e0001-0000-4000-8000-000000000006
# ╟─040e1000-0000-4000-8000-000000000070
# ╟─040e0001-0000-4000-8000-000000000007
# ╟─04a30000-0000-4000-8000-000000000001
# ╟─04a30000-0000-4000-8000-000000000002
# ╟─040e0002-0000-4000-8000-000000000004
# ╟─040e0002-0000-4000-8000-000000000006
