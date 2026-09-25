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
140 keV by one package call, `BS.vmi_pipeline`, and checked rod by rod against the
attenuation of the phantom's own materials.
"""

# ╔═╡ f2798d62-3509-4cc4-a24f-39ace8bb5a9e
md"""
## Pipeline

```
80 kVp + 140 kVp corrected log sinograms   (two co-registered acquisitions)
  → BS.spectral_basis_from_acquisitions     (per-ray absolute response of each kVp)
  → BS.vmi_pipeline
       projection-domain HYPR-LR on the counts of each detector row
     → K = 2 n-channel Poisson maximum-likelihood decomposition (iodine + water, every ray)
     → image-domain HYPR on the FDK-reconstructed basis pair
     → Kalender ACNR
     → VMI synthesis at 50 / 70 / 100 / 140 keV
```

This is the chain of the basis-vmi and basis-spectral-denoising papers (projection HYPR +
image HYPR + ACNR). Both kVp channels stay separate all the way into the likelihood; nothing
is rebinned into a single "effective" spectrum.
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
### 01. `Phantom()` Struct

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
### 02. `EICTScanner()` Struct

The Apex Elite geometry (source–isocenter 625.6 mm, source–detector 1100 mm, 834 columns
× 0.6 mm, 256 rows × 0.625 mm), its Gemstone scintillator (`:lumex`) and the GE large-body
bowtie. Rapid kVp switching alternates the tube between 80 and 140 kVp from view to view on
one tube and one detector, so the two channels see the same rays. It is modelled here as two
co-registered axial acquisitions on this one geometry, one per kVp, which gives every ray
both channels. It is an idealization: on the scanner each kVp gets only its share of the
views and of the tube output, so each channel here is less noisy than on a clinical scan.
"""

# ╔═╡ 2c157064-8567-450b-bc08-c2606084a77f
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
    electronic_noise = 0,
    detection_gain = 10.0,
)

# ╔═╡ 040e1000-0000-4000-8000-000000000102
md"""
### 03. `CTProtocol()` Struct

One protocol per kVp: 984 views over a 0.5 s rotation, 5 mm collimation, 4.5 mm of added
aluminium on both.
"""

# ╔═╡ f9c0af7a-addd-4249-96fb-b9078765fbd1
protocol_low = BS.CTProtocol(
    kVp = 80,
    mA = 407.0,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 03b00004-0000-4000-8000-000000000020
protocol_high = BS.CTProtocol(
    kVp = 140,
    mA = 405.0,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 040e1000-0000-4000-8000-000000000103
md"""
### 04. `SimOptions()` & `ReconOptions()`

Each kVp gets its own noise seed: two acquisitions drawn from one seed would carry the same
noise pattern, correlated between the channels, which a real scan never has. The
reconstruction grid is 8 slices of 0.625 mm over the central 5 mm, 512² over 35 cm.
"""

# ╔═╡ 2d65a0c0-b25d-41ad-9cd3-e7a2d08a2482
sim_opts = (
    low = BS.SimOptions(seed = 1234, projector = :dd_fast),
    high = BS.SimOptions(seed = 4321, projector = :dd_fast),
)

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
### 05. Forward Project: `simulate!`

One workspace and one `simulate!` per kVp. Each acquisition keeps what the spectral basis
needs: its corrected log sinogram, its per-ray air counts `I0_ray` (the detector's air count
× the bowtie air profile) and its per-ray detected spectrum from
`resolve_source_spectrum_full` (source × filtration × bowtie × detector response), the same
model the forward projector applied.
"""

# ╔═╡ 4315ef69-aa2f-4ee0-a13b-c65e01fb87ce
dual_scans = map((
        (label = "80 kVp", protocol = protocol_low, opts = sim_opts.low),
        (label = "140 kVp", protocol = protocol_high, opts = sim_opts.high),
    )) do (label, protocol, opts)
    @info "Simulating dual-kVp channel: $label"
    ws = BS.create_eict_workspace(scanner, protocol, opts, recon_opts, phantom)
    try
        BS.simulate!(ws, phantom, protocol, opts)
        air = ws.bowtie_air_reference === nothing ?
            ones(Float32, ws.geom.n_cols, ws.geom.n_rows) :
            Float32.(Array(ws.bowtie_air_reference))
        I0 = BS.compute_detector_I0(ws.geom, protocol, sum(ws.weights)) * Float64(ws.η_eff)
        energies, response = BS.resolve_source_spectrum_full(
            opts, protocol; scanner = scanner, geom = ws.geom,
        )
        (
            label, sino = Float32.(Array(ws.sinogram)), geom = ws.geom,
            I0_ray = Float32.(I0 .* air),
            energies = Float64.(energies), response = Float32.(response),
        )
    finally
        BS.release_backend!(ws)
    end
end;

# ╔═╡ dc8a8352-5598-4cdd-952f-3d77367850e9
md"""
## VMI Pipeline

### 01. Spectral Basis from the Two Acquisitions

`spectral_basis_from_acquisitions` merges the two kVp energy grids onto their union and
scales each acquisition's per-ray spectrum by its own air counts, so the likelihood sees the
absolute response ``\Phi_k(E)`` of every ray and channel: source, filtration, bowtie and
detector, the model that generated the data. No calibration scan is involved.
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
The basis holds $(basis.n_channels) channels on a $(length(basis.E))-point energy grid
($(round(Int, minimum(basis.E)))–$(round(Int, maximum(basis.E))) keV) for
$(size(basis.Φ, 1)) × $(size(basis.Φ, 2)) rays; the response sums to the air counts to
within $(round(basis.I0_relerr, sigdigits = 2)) (relative).
""")

# ╔═╡ 03a10001-0000-4000-8000-000000000003
md"""
### 02. The Chain's Settings

The denoiser is generalized HYPR-LR in both domains, `BS.SpectralHYPR`: a 3 × 3
(column × view) window on the counts of each detector row before the decomposition, and on
the reconstructed basis pair a 1 × 1 × 7 composite (across slices only) and a 15 × 15 × 7 complement window. The
decomposition is the K-channel estimator with its published controls; T-LBF (a
photon-counting filter) is off; the view-direction antialias is on.

The FBP kernel is the notebook's GE soft-tissue kernel, a CatSim-style apodization halfway
between the Standard and Soft windows.
"""

# ╔═╡ 03a10001-0000-4000-8000-000000000004
# the published chain (basis-vmi / basis-spectral-denoising): a 3 × 3 (column × view) window on
# the counts, and on the basis pair a 1 × 1 × 7 composite and a 15 × 15 × 7 complement window
HYPR_CHAIN = BS.SpectralHYPR()

# ╔═╡ 03a10001-0000-4000-8000-000000000005
VMI_CHAIN = (method = :nchannel, controls = BS.NChannelControls(), use_tlbf = false, antialias = true);

# ╔═╡ 03a10001-0000-4000-8000-000000000006
# Halfway between CatSim Standard (1, .934, .744, .443, .053) and Soft.
GE_KERNEL = BS.CustomFilter(
    (0.0, 0.25, 0.5, 0.75, 1.0),
    (1.0, 0.8744, 0.6003, 0.3031, 0.0266),
);

# ╔═╡ 040e1000-0000-4000-8000-000000000003
md"""
### 03. `vmi_pipeline`

One call, from the two corrected sinograms to the VMI stack, reconstructed on the
notebook's grid (`recon_opts.matrix_size`) with every detector row kept.
"""

# ╔═╡ 040e0002-0000-4000-8000-000000000002
dual_vmi = BS.vmi_pipeline(;
    channels = [s.sino for s in dual_scans],
    basis,
    geom = first(dual_scans).geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = (50.0, 70.0, 100.0, 140.0),
    fbp_filter = GE_KERNEL,
    denoiser = HYPR_CHAIN,
    use_acnr = true,
    VMI_CHAIN...,
);

# ╔═╡ 03a10001-0000-4000-8000-000000000007
let
    q = dual_vmi.quality
    d = dual_vmi.settings.denoiser
    pct(x) = round(100x, digits = 3)
    Markdown.parse("""
    The decomposition solved $(q.n_rays) rays in $(round(dual_vmi.elapsed_s, digits = 1)) s
    ($(round(q.outer_mean, digits = 1)) outer iterations on average); $(pct(q.frac_not_converged))% did not
    converge and $(pct(q.frac_bound_iodine))% / $(pct(q.frac_bound_water))% touched the iodine / water bounds.
    The projection HYPR measured dispersions (variance / mean of the counts on the air rays) of
    $(join(round.(d.dispersion, digits = 2), " and ")) for 80 and 140 kVp. The image HYPR's minimum-noise
    composite energy is E* = $(round(Int, d.image_estimates.Estar)) keV.
    """)
end

# ╔═╡ 040e0002-0000-4000-8000-000000000003
md"""
### 04. Basis Maps and VMIs

The water and iodine basis pair after image HYPR and ACNR (mid slice), and the VMIs
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

Per-rod measured versus theoretical HU at the four VMI energies, the water-region mean
and noise, and the linear regression of measured on theoretical HU.

- **Measured HU**: mean over an 8-px-radius circle at each rod's centroid, over every
  reconstructed slice.
- **Theoretical HU**: ``1000\,(\mu_r(E) - \mu_w(E))/\mu_w(E)`` of the rod's material from
  `BS.compute_μ_at_energy`, with no fitting.
- **Water region**: the solid-water body eroded by 12 px, away from every insert and edge.
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
    vmi(index) = view(dual_vmi.vmis, :, :, :, index)
    region(volume, pixels) = [volume[p, z] for z in axes(volume, 3) for p in pixels]
    μwater = Dict(E => BS.compute_μ_at_energy(BS.XA.Materials.water, E) for E in energies)
    rods = Dict{Symbol, NamedTuple}()
    for group in (:Ca, :I)
        measured = zeros(length(rod_labels[group]), length(energies))
        theoretical = similar(measured)
        for (row, label) in pairs(rod_labels[group])
            roi = rod_roi(label)
            material = phantom_cpu.materials[Int(label) + 1]
            for (column, E) in pairs(energies)
                measured[row, column] = mean(region(vmi(column), roi))
                μ = BS.compute_μ_at_energy(material, E)
                theoretical[row, column] = 1000 * (μ - μwater[E]) / μwater[E]
            end
        end
        rods[group] = (names = rod_names[group], measured, theoretical)
    end
    water_pixels = findall(water_mask)
    (
        energies, water_mask, rods,
        water_mean = [mean(region(vmi(k), water_pixels)) for k in eachindex(energies)],
        water_noise = [std(region(vmi(k), water_pixels)) for k in eachindex(energies)],
        finite = all(isfinite, dual_vmi.vmis),
    )
end;

# ╔═╡ 040e1000-0000-4000-8000-000000000040
md"""
### Water ROI

The eroded solid-water region, overlaid on the 70 keV VMI, and its mean HU at each energy.
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
    Mke.barplot!(mean_axis, 1:n, dual_results.water_mean;
        color = [Mke.cgrad(:plasma, n; categorical = true)[index] for index in 1:n])
    Mke.hlines!(mean_axis, [0.0]; color = :black, linestyle = :dash)
    for (index, value) in pairs(dual_results.water_mean)
        Mke.text!(
            mean_axis, index, value;
            text = "$(round(value, digits = 2)) HU",
            align = (:center, value ≥ 0 ? :bottom : :top),
            offset = (0, value ≥ 0 ? 5 : -5),
        )
    end
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

The HU standard deviation over the same eroded water region, at each energy.
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

Solid lines show measured HU and dashed lines theoretical HU for the calcium and iodine
inserts across the four energies.
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
    addcheck("chain: projection + image HYPR, ACNR",
             (dual_vmi.settings.denoiser !== nothing, dual_vmi.settings.acnr !== nothing),
             dual_vmi.settings.denoiser !== nothing && dual_vmi.settings.acnr !== nothing)
    addcheck("all VMI values finite", dual_results.finite, dual_results.finite)
    water_worst = maximum(abs, dual_results.water_mean)
    addcheck("solid-water worst absolute HU", round(water_worst, digits = 2), water_worst ≤ 10)
    addcheck(
        "noise decreases from 50 to 140 keV",
        round.(dual_results.water_noise; digits = 2),
        all(diff(dual_results.water_noise) .< 0),
    )
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
Simulate 80 + 140 kVp (two co-registered acquisitions, independent noise)
   → BS.spectral_basis_from_acquisitions (absolute per-ray response of each kVp)
   → BS.vmi_pipeline(; denoiser = HYPR_CHAIN, use_acnr = true)
        projection HYPR → K = 2 n-channel decomposition on every ray
        → image HYPR on the FDK basis pair → Kalender ACNR
        → VMI synthesis at 50 / 70 / 100 / 140 keV
   → water, noise, per-rod and regression checks against theory
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
# ╠═f9c0af7a-addd-4249-96fb-b9078765fbd1
# ╠═03b00004-0000-4000-8000-000000000020
# ╟─040e1000-0000-4000-8000-000000000103
# ╠═2d65a0c0-b25d-41ad-9cd3-e7a2d08a2482
# ╠═08cbc6fd-3c7c-432f-99e5-b220f8fe7fde
# ╟─5ecd97c6-ad47-4558-886d-22ed45eda97d
# ╠═4315ef69-aa2f-4ee0-a13b-c65e01fb87ce
# ╟─dc8a8352-5598-4cdd-952f-3d77367850e9
# ╠═4ca28c64-ee96-47c8-b7c3-f0e0c4c99423
# ╟─03a10001-0000-4000-8000-000000000001
# ╟─03a10001-0000-4000-8000-000000000002
# ╟─03a10001-0000-4000-8000-000000000003
# ╠═03a10001-0000-4000-8000-000000000004
# ╠═03a10001-0000-4000-8000-000000000005
# ╠═03a10001-0000-4000-8000-000000000006
# ╟─040e1000-0000-4000-8000-000000000003
# ╠═040e0002-0000-4000-8000-000000000002
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
# ╟─040e1000-0000-4000-8000-000000000070
# ╟─040e0001-0000-4000-8000-000000000007
# ╟─040e0002-0000-4000-8000-000000000004
# ╟─040e0002-0000-4000-8000-000000000006
