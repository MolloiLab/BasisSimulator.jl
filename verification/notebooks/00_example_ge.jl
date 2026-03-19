### A Pluto.jl notebook ###
# v0.19.0

# ╔═╡ 00010001-0000-4000-8000-000000000000
begin
    using Pkg: Pkg
    Pkg.activate(dirname(@__DIR__))
    # Pkg.resolve()
    # Pkg.instantiate()
    using Revise

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
end

# ╔═╡ 14530566-ecaa-4345-9115-e75981f3837c
using Metal # Choose one or the other

# ╔═╡ 00010003-0000-4000-8000-000000000000
using LinearAlgebra

# ╔═╡ 00010004-0000-4000-8000-000000000000
using FFTW

# ╔═╡ 00010005-0000-4000-8000-000000000000
using Random

# ╔═╡ 35b71532-be75-4fe9-ada8-9d6d4bf21f6e
using Markdown

# ╔═╡ 0b629a9c-722c-47c5-900e-5e5b311a7439
# using CUDA # Choose one or the other

# ╔═╡ 00010002-0000-4000-8000-000000000000
import PlutoUI as UI

# ╔═╡ 00010006-0000-4000-8000-000000000000
import BasisSimulator as BS

# ╔═╡ 00010007-0000-4000-8000-000000000000
import CairoMakie as CM

# ╔═╡ 00010008-0000-4000-8000-000000000000
import Statistics: mean, std

# ╔═╡ 00010009-0000-4000-8000-000000000000
import XrayAttenuation as XA

# ╔═╡ e32e62a8-ecc2-4f64-8edd-bed87b14d2ba
UI.TableOfContents()

# ╔═╡ 00010010-0000-4000-8000-000000000000
const RESULTS_DIR = joinpath(dirname(@__DIR__), "results", "example_ge"); mkpath(RESULTS_DIR)

# ╔═╡ 00010011-0000-4000-8000-000000000000
begin
    """Move array to GPU. Tries CUDA then Metal, falls back to CPU."""
    function to_gpu(x::AbstractArray)
        if @isdefined(CUDA) && CUDA.functional()
            return CUDA.CuArray(x)
        elseif @isdefined(Metal) && Metal.functional()
            return Metal.MtlArray(x)
        else
            @warn "No GPU backend available, using CPU"
            return x
        end
    end

    """Force GPU memory cleanup."""
    function clear_gpu!()
        GC.gc(true)
        if @isdefined(CUDA) && CUDA.functional()
            CUDA.reclaim()
        end
        return nothing
    end
end

# ╔═╡ 00020001-0000-4000-8000-000000000000
md"""
# GE Revolution Apex Elite — Example Simulation

Single 120 kVp / 150 mA scan with 80 mm collimation (~10 mGy CTDIvol equivalent).
Reconstructed with both FBP and HIR (half-iterative reconstruction).

All parameters are verified against clinical GE Revolution Apex Elite data
(see `06_ge_apex_elite_clinical.jl` for full validation).
"""

# ╔═╡ 00030001-0000-4000-8000-000000000000
md"""
## 1. Phantom Setup — Gammex 472 (via `create_phantom_from_mask`)

Uses the generic `create_phantom_from_mask` API that supports arbitrary phantoms
(XCAT, custom segmentations, etc.) with any number of materials.
The mask auto-promotes to UInt16 when >255 labels are present.
"""

# ╔═╡ 00030002-0000-4000-8000-000000000000
begin
    GAMMEX_SOLID_WATER = XA.Materials.gammex_472_solidwater

    # Material mapping: label → XA.Material (generic Dict{Int, ...} API)
    PHANTOM_MATERIALS = Dict{Int, XA.Material}(
        0 => XA.Materials.air,
        1 => GAMMEX_SOLID_WATER,
        2 => XA.Materials.water,
        3 => GAMMEX_SOLID_WATER,
        10 => XA.Materials.gammex_472_ca50_0,
        11 => XA.Materials.gammex_472_ca100_0,
        12 => XA.Materials.gammex_472_ca200_0,
        13 => XA.Materials.gammex_472_ca300_0,
        14 => XA.Materials.gammex_472_ca400_0,
        20 => XA.Materials.gammex_472_i2_0,
        21 => XA.Materials.gammex_472_i2_5,
        22 => XA.Materials.gammex_472_i5_0,
        23 => XA.Materials.gammex_472_i7_5,
        24 => XA.Materials.gammex_472_i10_0,
        25 => XA.Materials.gammex_472_i15_0,
        26 => XA.Materials.gammex_472_i20_0,
    )

    MATERIAL_INFO = Dict(
        0 => (name = "Air", color = :gray15),
        1 => (name = "Solid Water", color = :lightskyblue),
        2 => (name = "Pure Water", color = :royalblue),
        3 => (name = "SW Reference", color = :paleturquoise),
        10 => (name = "Ca 50 mg/mL", color = :wheat),
        11 => (name = "Ca 100 mg/mL", color = :sandybrown),
        12 => (name = "Ca 200 mg/mL", color = :orange),
        13 => (name = "Ca 300 mg/mL", color = :darkorange),
        14 => (name = "Ca 400 mg/mL", color = :orangered),
        20 => (name = "I 2.0 mg/mL", color = :honeydew),
        21 => (name = "I 2.5 mg/mL", color = :palegreen),
        22 => (name = "I 5.0 mg/mL", color = :lightgreen),
        23 => (name = "I 7.5 mg/mL", color = :mediumseagreen),
        24 => (name = "I 10.0 mg/mL", color = :seagreen),
        25 => (name = "I 15.0 mg/mL", color = :forestgreen),
        26 => (name = "I 20.0 mg/mL", color = :darkgreen),
    )
end;

# ╔═╡ 00030003-0000-4000-8000-000000000000
"""Build a raw integer mask for the Gammex 472 geometry (no materials — just labels)."""
function build_gammex_472_mask(;
        n_voxels::Int = 1750,
        n_slices::Int = 250,
        fov_cm::Float64 = 35.0,
        z_cm::Float64 = 5.0,
    )
    dx = fov_cm / n_voxels
    dy = fov_cm / n_voxels
    dz = z_cm / n_slices

    x = range(-fov_cm / 2 + dx / 2, fov_cm / 2 - dx / 2, length = n_voxels)
    y = range(-fov_cm / 2 + dy / 2, fov_cm / 2 - dy / 2, length = n_voxels)

    body_radius = 16.5
    rod_radius = 1.4
    air_gap_water = 0.03
    air_gap_other = 0.01
    rod_radius_sq = rod_radius^2
    outer_ring_R = 10.5
    inner_ring_R = 5.5

    outer_start = pi / 2 - pi / 8
    outer_angles = [outer_start - (i - 1) * pi / 4 for i in 1:8]
    outer_labels = [11, 12, 13, 14, 2, 3, 3, 10]

    inner_start = pi / 2
    inner_angles = [inner_start - (i - 1) * pi / 4 for i in 1:8]
    inner_labels = [21, 22, 23, 24, 25, 26, 2, 20]

    outer_hole_r_sq = [(rod_radius + (outer_labels[i] == 2 ? air_gap_water : air_gap_other))^2 for i in 1:8]
    inner_hole_r_sq = [(rod_radius + (inner_labels[i] == 2 ? air_gap_water : air_gap_other))^2 for i in 1:8]

    outer_cx = [outer_ring_R * cos(a) for a in outer_angles]
    outer_cy = [outer_ring_R * sin(a) for a in outer_angles]
    inner_cx = [inner_ring_R * cos(a) for a in inner_angles]
    inner_cy = [inner_ring_R * sin(a) for a in inner_angles]

    slice = zeros(Int, n_voxels, n_voxels)

    for j in 1:n_voxels, i in 1:n_voxels
        xi, yj = x[i], y[j]

        if xi^2 + yj^2 <= body_radius^2
            slice[i, j] = 1

            for idx in 1:8
                d_sq = (xi - outer_cx[idx])^2 + (yj - outer_cy[idx])^2
                if d_sq <= outer_hole_r_sq[idx]
                    slice[i, j] = d_sq <= rod_radius_sq ? outer_labels[idx] : 0
                    @goto next_voxel
                end
            end

            for idx in 1:8
                d_sq = (xi - inner_cx[idx])^2 + (yj - inner_cy[idx])^2
                if d_sq <= inner_hole_r_sq[idx]
                    slice[i, j] = d_sq <= rod_radius_sq ? inner_labels[idx] : 0
                    break
                end
            end
        end
        @label next_voxel
    end

    slice = reverse(permutedims(rot180(slice)), dims = 2)

    mask = Array{Int, 3}(undef, n_voxels, n_voxels, n_slices)
    @views for k in 1:n_slices
        mask[:, :, k] .= slice
    end

    return mask, (dx, dy, dz), (fov_cm, fov_cm, z_cm)
end

# ╔═╡ 00030004-0000-4000-8000-000000000000
# Phantom: 1750 × 1750 × 300, 0.2 mm isotropic, 35 cm FOV, 6 cm z-extent
# z covers 80 mm collimation at isocenter (~5.5 cm) with margin
begin
    raw_mask, voxel_size, _ = build_gammex_472_mask(
        n_voxels = 1750,
        n_slices = 300,
        fov_cm = 35.0,
        z_cm = 6.0,
    )

    # Generic phantom construction: raw integer mask + materials dict → Phantom
    # Works with any number of labels (auto-promotes UInt8/UInt16 as needed)
    sim_phantom = BS.create_phantom_from_mask(raw_mask, PHANTOM_MATERIALS, voxel_size)

    sim_phantom_gpu = BS.Phantom(
        to_gpu(sim_phantom.mask),
        sim_phantom.materials,
        sim_phantom.voxel_size,
        sim_phantom.origin,
        sim_phantom.extent,
    )
    @info "Phantom: $(size(sim_phantom.mask)) voxels, eltype=$(eltype(sim_phantom.mask)), extent=$(sim_phantom.extent) cm"
end;

# ╔═╡ 00030005-0000-4000-8000-000000000000
# Phantom mid-slice visualization
let
    mid = size(sim_phantom.mask, 3) ÷ 2
    nz = size(sim_phantom.mask, 3)
    slice_data = sim_phantom.mask[:, :, mid]

    unique_labels = sort(unique(slice_data))
    n_labels = length(unique_labels)

    lut = zeros(Float32, 27)
    for (i, l) in enumerate(unique_labels)
        lut[Int(l) + 1] = Float32(i)
    end
    mapped = lut[Int.(slice_data) .+ 1]

    colors = [MATERIAL_INFO[Int(l)].color for l in unique_labels]
    cmap = CM.cgrad(colors, n_labels, categorical = true)
    names = [MATERIAL_INFO[Int(l)].name for l in unique_labels]

    fig = CM.Figure(size = (1000, 850), fontsize = 12)
    ax = CM.Axis(
        fig[1, 1];
        title = "Gammex 472 Phantom — Slice $mid / $nz (0.2 mm voxels)",
        aspect = CM.DataAspect()
    )
    hm = CM.heatmap!(ax, mapped; colormap = cmap, colorrange = (0.5, n_labels + 0.5))
    CM.Colorbar(fig[1, 2], hm; ticks = (1:n_labels, names), ticklabelsize = 11, width = 15)
    CM.save(joinpath(RESULTS_DIR, "phantom.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00040001-0000-4000-8000-000000000000
md"""
## 2. Scanner & X-ray Spectrum
"""

# ╔═╡ 00040002-0000-4000-8000-000000000000
# Additional filtration: 4.5 mm Al (empirically matched to clinical HVL)
additional_filters = [("Al", 4.5)]

# ╔═╡ 00040003-0000-4000-8000-000000000000
# GE Revolution Apex Elite geometry (verified against clinical data)
begin
    sim_electronic_noise = 0       # e⁻ (electronic noise disabled; noise from quantum + floor)
    sim_detection_gain = 10.0      # e⁻/keV

    sim_scanner = BS.Scanner(
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
        electronic_noise = sim_electronic_noise,
        detection_gain = sim_detection_gain,
    )
end

# ╔═╡ 00040004-0000-4000-8000-000000000000
sim_opts = BS.SimOptions(fidelity = :eict, seed = 1234)

# ╔═╡ 00040005-0000-4000-8000-000000000000
# X-ray spectrum: raw tube output vs filtered (flat filter + additional Al)
let
    kVp = 120

    # Raw tube spectrum (IPEM anode model, no filtration)
    e_raw, w_raw = BS.load_spectrum_unfiltered(kVp; anode_angle = 10)

    # After inherent + additional filtration (2.5 mm Al flat + 4.5 mm Al additional = 7.0 mm Al total)
    all_filters = vcat([("Al", 2.5)], additional_filters)  # flat + additional
    _, w_filt = BS.filter_spectrum(e_raw, w_raw; filters = all_filters, sdd_mm = 1100.0)

    # Fully resolved spectrum (includes flat filter, bowtie center, detector response)
    prot = BS.CTProtocol(kVp = kVp, additional_filters = additional_filters)
    e_res, w_res = BS.resolve_spectrum(sim_opts, prot; scanner = sim_scanner)
    E_eff = sum(e_res .* w_res) / sum(w_res)

    fig = CM.Figure(size = (900, 500), fontsize = 13)
    ax = CM.Axis(
        fig[1, 1];
        title = "120 kVp X-ray Spectrum — GE Revolution Apex Elite",
        xlabel = "Energy (keV)", ylabel = "Relative Fluence (a.u.)"
    )

    # Normalize for overlay
    CM.lines!(
        ax, e_raw, w_raw ./ maximum(w_raw);
        color = :gray60, linewidth = 1.5, label = "Raw tube (no filter)"
    )
    CM.lines!(
        ax, e_raw, w_filt ./ maximum(w_filt);
        color = :steelblue, linewidth = 2, label = "After 7.0 mm Al"
    )
    CM.lines!(
        ax, e_res, w_res ./ maximum(w_res);
        color = :darkorange, linewidth = 2, label = "Resolved (+ bowtie + detector)"
    )
    CM.vlines!(
        ax, [E_eff]; color = :red, linestyle = :dash, linewidth = 1,
        label = "E_eff = $(round(E_eff, digits = 1)) keV"
    )

    CM.xlims!(ax, 15, 125)
    CM.ylims!(ax, 0, 1.05)
    CM.axislegend(ax; position = :rt)
    CM.save(joinpath(RESULTS_DIR, "spectrum_120kVp.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00050001-0000-4000-8000-000000000000
md"""
## 3. Acquisition Protocol
"""

# ╔═╡ 00050002-0000-4000-8000-000000000000
begin
    sim_rotation_time = 1.0       # seconds
    # sim_collimation_mm = 80.0     # 128 × 0.625 mm detector rows
    sim_collimation_mm = 10.0     # 128 × 0.625 mm detector rows
    sim_n_views = 984             # standard GE Revolution
end

# ╔═╡ 00050003-0000-4000-8000-000000000000
begin
    sim_recon_xy = 512
    sim_recon_fov_cm = 35.0
    sim_slice_thickness_mm = 0.625
    sim_recon_z_cm = sim_collimation_mm / 10.0
    sim_n_recon_slices = round(Int, sim_collimation_mm / sim_slice_thickness_mm)
    sim_matrix_size = (sim_recon_xy, sim_recon_xy, sim_n_recon_slices)

    sim_recon_geom = BS.ReconOptions(
        algorithm = :fdk,
        matrix_size = sim_matrix_size,
        fov_cm = sim_recon_fov_cm,
        z_cm = sim_recon_z_cm,
    )
    @info "Recon: $(sim_matrix_size), FOV=$(sim_recon_fov_cm) cm, z=$(sim_recon_z_cm) cm"
end

# ╔═╡ 00050004-0000-4000-8000-000000000000
sim_protocol = BS.CTProtocol(
    kVp = 120,
    mA = 150.0,
    views = sim_n_views,
    rotation_time = sim_rotation_time,
    collimation_mm = sim_collimation_mm,
    additional_filters = additional_filters,
)

# ╔═╡ 00060001-0000-4000-8000-000000000000
md"""
## 4. BHC Calibration
"""

# ╔═╡ 00060002-0000-4000-8000-000000000000
# Custom FBP filter (verified against clinical kernel)
custom_filter_control = (
    x = (0.0, 0.25, 0.5, 0.75, 1.0),
    y = (1.0, 0.85, 0.6, 0.15, 0.001),
)

# ╔═╡ 00060003-0000-4000-8000-000000000000
# Noise floor: dose-independent system noise (scatter residuals, electronics, calibration)
sim_noise_floor_hu = 28.0

# ╔═╡ 00060004-0000-4000-8000-000000000000
begin
    bhc_enabled = true
    # Sinogram-domain BHC (water + bone, iterative)
    sino_bhc_hu_low = 450.0
    sino_bhc_hu_high = 600.0
    sino_order = 2
    # Image-domain BHC (residual high-Z correction)
    img_bhc_hu_low = 50.0
    img_bhc_hu_high = 150.0
    img_bhc_scale_factor = 0.2
end

# ╔═╡ 00060005-0000-4000-8000-000000000000
# Calibrate two-material BHC model for 120 kVp
bhc_model, μ_water_ref = let
    prot = BS.CTProtocol(kVp = 120, additional_filters = additional_filters)
    e, w = BS.resolve_spectrum(sim_opts, prot; scanner = sim_scanner)
    ref_E = sum(e .* w) / sum(w)
    model = BS.calibrate_bhc_two_material(
        e, w;
        order = sino_order,
        reference_energy_keV = ref_E,
        hu_low = sino_bhc_hu_low,
        hu_high = sino_bhc_hu_high
    )
    @info "BHC 120 kVp: E_ref=$(round(ref_E, digits = 1)) keV, μ_water=$(round(model.μ_water_ref, digits = 5)) cm⁻¹"
    model, model.μ_water_ref
end

# ╔═╡ 00070001-0000-4000-8000-000000000000
md"""
## 5. Forward Projection (Polychromatic Simulation)
"""

# ╔═╡ 00070002-0000-4000-8000-000000000000
sim_sino = let
    @info "Simulating 120 kVp / 150 mA / $(sim_collimation_mm) mm collimation..."
    ws = BS.create_eict_workspace(
        sim_scanner, sim_protocol, sim_opts, sim_recon_geom, sim_phantom_gpu
    )
    BS.simulate!(ws, sim_phantom_gpu, sim_scanner, sim_protocol, sim_opts, sim_recon_geom)
    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom)
    ws = nothing; clear_gpu!()
    result
end

# ╔═╡ 00070003-0000-4000-8000-000000000000
# Sinogram mid-view visualization
let
    mid_view = size(sim_sino.sino, 2) ÷ 2
    mid_row = size(sim_sino.sino, 3) ÷ 2
    fig = CM.Figure(size = (900, 400), fontsize = 13)

    ax1 = CM.Axis(
        fig[1, 1]; title = "Sinogram — Mid-view (view $mid_view)",
        xlabel = "Detector column", ylabel = "Detector row"
    )
    CM.heatmap!(ax1, sim_sino.sino[:, mid_view, :]'; colormap = :grays)

    ax2 = CM.Axis(
        fig[1, 2]; title = "Sinogram — Mid-row (row $mid_row)",
        xlabel = "Detector column", ylabel = "View angle"
    )
    CM.heatmap!(ax2, sim_sino.sino[:, :, mid_row]'; colormap = :grays)

    CM.save(joinpath(RESULTS_DIR, "sinogram.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080001-0000-4000-8000-000000000000
md"""
## 6. FBP Reconstruction
"""

# ╔═╡ 00080002-0000-4000-8000-000000000000
# FBP: Two-stage BHC → FDK → HU → noise floor → cupping correction
sim_fbp = let
    sino_gpu = to_gpu(sim_sino.sino)
    geom = sim_sino.geom
    recon_size = sim_matrix_size

    # Sinogram-domain BHC (water + bone)
    if bhc_enabled
        sino_corrected = BS.apply_bhc_two_material(
            sino_gpu, bhc_model, geom, recon_size;
            volume_extent = sim_phantom_gpu.extent
        )
        sino_gpu = to_gpu(sino_corrected)
    end

    # FDK with custom filter
    ws_fdk = BS.create_fdk_recon_workspace(
        sino_gpu, geom, recon_size;
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y)
    )
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size)

    # Image-domain BHC (residual high-Z)
    if bhc_enabled
        BS.apply_bhc_image_domain(
            recon_μ, geom, recon_size, μ_water_ref;
            hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
            scale_factor = img_bhc_scale_factor,
            volume_extent = sim_phantom_gpu.extent
        )
    end

    vol = Array(recon_μ)
    ws_fdk = nothing; sino_gpu = nothing; recon_μ = nothing; clear_gpu!()
    vol
end

# ╔═╡ 00080003-0000-4000-8000-000000000000
# HU conversion + post-processing
sim_fbp_hu = let
    recon_hu = Float32.(BS.to_hounsfield(sim_fbp; μ_water = μ_water_ref))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    BS.apply_radial_cupping_correction!(recon_hu; fov_cm = sim_recon_fov_cm)
    recon_hu
end

# ╔═╡ 096ba509-c17b-4d01-97b0-1ce75df58fc0
mid_z = size(sim_fbp_hu, 3) ÷ 2

# ╔═╡ e7d41653-97f9-4f14-b3dc-510492ea11e5
@bind z_fbp UI.Slider(axes(sim_fbp_hu, 3); show_value = true, default = mid_z)

# ╔═╡ 00080004-0000-4000-8000-000000000000
# FBP mid-slice visualization
let
    # slice = reverse(sim_fbp_hu[:, :, z_fbp]', dims = 1)
    slice = sim_fbp_hu[:, :, z_fbp]

    fig = CM.Figure(size = (600, 550), fontsize = 13)
    ax = CM.Axis(
        fig[1, 1];
        title = "FBP — 120 kVp / 150 mA (slice $z_fbp)",
        aspect = CM.DataAspect()
    )
    hm = CM.heatmap!(ax, slice; colormap = :grays, colorrange = (-200, 200))
    CM.Colorbar(fig[1, 2], hm; label = "HU")
    CM.save(joinpath(RESULTS_DIR, "fbp_hu.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00090001-0000-4000-8000-000000000000
md"""
## 7. HIR Reconstruction (Half-Iterative)
"""

# ╔═╡ 00090002-0000-4000-8000-000000000000
begin
    hir_strength = 3
    hir_lambda = 10.0f0
    hir_nepochs = 2
    hir_n_subsets = 12
    hir_huber_delta = 0.06f0
    hir_relaxation = 0.35f0
end

# ╔═╡ 00090003-0000-4000-8000-000000000000
# HIR: Two-stage BHC → HIR → HU → noise floor → cupping correction
sim_hir = let
    sino_gpu = to_gpu(sim_sino.sino)
    geom = sim_sino.geom
    recon_size = sim_matrix_size

    # Sinogram-domain BHC (same as FBP)
    if bhc_enabled
        sino_corrected = BS.apply_bhc_two_material(
            sino_gpu, bhc_model, geom, recon_size;
            volume_extent = sim_phantom_gpu.extent
        )
        sino_gpu = to_gpu(sino_corrected)
    end

    # HIR reconstruction
    ws_hir = BS.create_hir_recon_workspace(
        sino_gpu, geom, recon_size;
        strength = hir_strength,
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y)
    )
    ws_hir.params = BS.HIRParams(
        hir_strength, hir_lambda, 30, hir_nepochs,
        hir_n_subsets, hir_huber_delta, hir_relaxation, (25, 35)
    )
    BS.reconstruct!(ws_hir, sino_gpu, geom, recon_size)
    recon_μ = ws_hir.volume

    # Image-domain BHC (same as FBP)
    if bhc_enabled
        BS.apply_bhc_image_domain(
            recon_μ, geom, recon_size, μ_water_ref;
            hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
            scale_factor = img_bhc_scale_factor,
            volume_extent = sim_phantom_gpu.extent
        )
    end

    vol = Array(recon_μ)
    ws_hir = nothing; sino_gpu = nothing; recon_μ = nothing; clear_gpu!()
    vol
end

# ╔═╡ 00090004-0000-4000-8000-000000000000
# HU conversion + post-processing
sim_hir_hu = let
    recon_hu = Float32.(BS.to_hounsfield(sim_hir; μ_water = μ_water_ref))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    BS.apply_radial_cupping_correction!(recon_hu; fov_cm = sim_recon_fov_cm)
    recon_hu
end

# ╔═╡ 2a0ceae5-6706-4336-9952-b120fc508c60
@bind z_hir UI.Slider(axes(sim_fbp_hu, 3); show_value = true, default = mid_z)

# ╔═╡ 00090005-0000-4000-8000-000000000000
# HIR mid-slice visualization
let
    slice = sim_hir_hu[:, :, z_hir]

    fig = CM.Figure(size = (600, 550), fontsize = 13)
    ax = CM.Axis(
        fig[1, 1];
        title = "HIR — 120 kVp / 150 mA (slice $z_hir)",
        aspect = CM.DataAspect()
    )
    hm = CM.heatmap!(ax, slice; colormap = :grays, colorrange = (-200, 200))
    CM.Colorbar(fig[1, 2], hm; label = "HU")
    CM.save(joinpath(RESULTS_DIR, "hir_hu.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00100001-0000-4000-8000-000000000000
md"""
## 8. FBP vs HIR Comparison
"""

# ╔═╡ 13043885-3fcc-439c-878c-91c663efcd5f
@bind z_combo UI.Slider(axes(sim_fbp_hu, 3); show_value = true, default = mid_z)

# ╔═╡ 00100002-0000-4000-8000-000000000000
# Side-by-side: FBP vs HIR (soft tissue window)
let
    fbp_slice = sim_fbp_hu[:, :, z_combo]
    hir_slice = sim_hir_hu[:, :, z_combo]

    fig = CM.Figure(size = (1100, 500), fontsize = 13)

    ax1 = CM.Axis(fig[1, 1]; title = "FBP (slice $z_combo)", aspect = CM.DataAspect())
    CM.heatmap!(ax1, fbp_slice; colormap = :grays, colorrange = (-200, 200))

    ax2 = CM.Axis(fig[1, 2]; title = "HIR (slice $z_combo)", aspect = CM.DataAspect())
    hm = CM.heatmap!(ax2, hir_slice; colormap = :grays, colorrange = (-200, 200))

    CM.Colorbar(fig[1, 3], hm; label = "HU", width = 12)
    CM.save(joinpath(RESULTS_DIR, "fbp_vs_hir.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00100003-0000-4000-8000-000000000000
# Side-by-side: FBP vs HIR (bone window)
let
    fbp_slice = sim_fbp_hu[:, :, z_combo]
    hir_slice = sim_hir_hu[:, :, z_combo]

    fig = CM.Figure(size = (1100, 500), fontsize = 13)

    ax1 = CM.Axis(fig[1, 1]; title = "FBP — Bone Window", aspect = CM.DataAspect())
    CM.heatmap!(ax1, fbp_slice; colormap = :grays, colorrange = (-500, 1500))

    ax2 = CM.Axis(fig[1, 2]; title = "HIR — Bone Window", aspect = CM.DataAspect())
    hm = CM.heatmap!(ax2, hir_slice; colormap = :grays, colorrange = (-500, 1500))

    CM.Colorbar(fig[1, 3], hm; label = "HU", width = 12)
    CM.save(joinpath(RESULTS_DIR, "fbp_vs_hir_bone.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00100004-0000-4000-8000-000000000000
# Noise comparison: water σ in center ROI
let
    mid_z = size(sim_fbp_hu, 3) ÷ 2 + 1
    cx, cy = size(sim_fbp_hu, 1) ÷ 2, size(sim_fbp_hu, 2) ÷ 2
    r = 15  # pixels

    fbp_vals = [
        sim_fbp_hu[cx + dx, cy + dy, mid_z]
            for dy in -r:r for dx in -r:r if dx^2 + dy^2 <= r^2
    ]
    hir_vals = [
        sim_hir_hu[cx + dx, cy + dy, mid_z]
            for dy in -r:r for dx in -r:r if dx^2 + dy^2 <= r^2
    ]

    fbp_mean = mean(fbp_vals)
    fbp_std = std(fbp_vals)
    hir_mean = mean(hir_vals)
    hir_std = std(hir_vals)

    fig = CM.Figure(size = (500, 400), fontsize = 13)
    ax = CM.Axis(
        fig[1, 1]; title = "Water ROI Noise",
        ylabel = "Water σ (HU)", xticks = ([1, 2], ["FBP", "HIR"])
    )
    CM.barplot!(
        ax, [1], [fbp_std]; width = 0.5, color = :steelblue,
        label = "FBP: μ=$(round(fbp_mean, digits = 1)), σ=$(round(fbp_std, digits = 1))"
    )
    CM.barplot!(
        ax, [2], [hir_std]; width = 0.5, color = :darkorange,
        label = "HIR: μ=$(round(hir_mean, digits = 1)), σ=$(round(hir_std, digits = 1))"
    )
    CM.ylims!(ax, 0, nothing)
    CM.axislegend(ax; position = :rt)
    CM.save(joinpath(RESULTS_DIR, "noise_comparison.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00100005-0000-4000-8000-000000000000
# Line profile through center (horizontal)
let
    mid_z = size(sim_fbp_hu, 3) ÷ 2 + 1
    cy = size(sim_fbp_hu, 2) ÷ 2

    fbp_profile = sim_fbp_hu[:, cy, mid_z]
    hir_profile = sim_hir_hu[:, cy, mid_z]
    x_cm = range(-sim_recon_fov_cm / 2, sim_recon_fov_cm / 2, length = size(sim_fbp_hu, 1))

    fig = CM.Figure(size = (900, 400), fontsize = 13)
    ax = CM.Axis(
        fig[1, 1]; title = "Horizontal Line Profile (mid-slice)",
        xlabel = "Position (cm)", ylabel = "HU"
    )
    CM.lines!(
        ax, collect(x_cm), Float64.(fbp_profile);
        color = :steelblue, linewidth = 1.5, label = "FBP"
    )
    CM.lines!(
        ax, collect(x_cm), Float64.(hir_profile);
        color = :darkorange, linewidth = 1.5, label = "HIR"
    )
    CM.axislegend(ax; position = :rt)
    CM.save(joinpath(RESULTS_DIR, "line_profile.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ Cell order:
# ╠═00010001-0000-4000-8000-000000000000
# ╠═14530566-ecaa-4345-9115-e75981f3837c
# ╠═0b629a9c-722c-47c5-900e-5e5b311a7439
# ╠═00010002-0000-4000-8000-000000000000
# ╠═00010003-0000-4000-8000-000000000000
# ╠═00010004-0000-4000-8000-000000000000
# ╠═00010005-0000-4000-8000-000000000000
# ╠═00010006-0000-4000-8000-000000000000
# ╠═00010007-0000-4000-8000-000000000000
# ╠═00010008-0000-4000-8000-000000000000
# ╠═00010009-0000-4000-8000-000000000000
# ╠═35b71532-be75-4fe9-ada8-9d6d4bf21f6e
# ╠═e32e62a8-ecc2-4f64-8edd-bed87b14d2ba
# ╠═00010010-0000-4000-8000-000000000000
# ╠═00010011-0000-4000-8000-000000000000
# ╠═00020001-0000-4000-8000-000000000000
# ╟─00030001-0000-4000-8000-000000000000
# ╠═00030002-0000-4000-8000-000000000000
# ╠═00030003-0000-4000-8000-000000000000
# ╠═00030004-0000-4000-8000-000000000000
# ╟─00030005-0000-4000-8000-000000000000
# ╟─00040001-0000-4000-8000-000000000000
# ╠═00040002-0000-4000-8000-000000000000
# ╠═00040003-0000-4000-8000-000000000000
# ╠═00040004-0000-4000-8000-000000000000
# ╟─00040005-0000-4000-8000-000000000000
# ╟─00050001-0000-4000-8000-000000000000
# ╠═00050002-0000-4000-8000-000000000000
# ╠═00050003-0000-4000-8000-000000000000
# ╠═00050004-0000-4000-8000-000000000000
# ╟─00060001-0000-4000-8000-000000000000
# ╠═00060002-0000-4000-8000-000000000000
# ╠═00060003-0000-4000-8000-000000000000
# ╠═00060004-0000-4000-8000-000000000000
# ╠═00060005-0000-4000-8000-000000000000
# ╟─00070001-0000-4000-8000-000000000000
# ╠═00070002-0000-4000-8000-000000000000
# ╟─00070003-0000-4000-8000-000000000000
# ╟─00080001-0000-4000-8000-000000000000
# ╠═00080002-0000-4000-8000-000000000000
# ╠═00080003-0000-4000-8000-000000000000
# ╠═096ba509-c17b-4d01-97b0-1ce75df58fc0
# ╟─e7d41653-97f9-4f14-b3dc-510492ea11e5
# ╟─00080004-0000-4000-8000-000000000000
# ╟─00090001-0000-4000-8000-000000000000
# ╠═00090002-0000-4000-8000-000000000000
# ╠═00090003-0000-4000-8000-000000000000
# ╠═00090004-0000-4000-8000-000000000000
# ╟─2a0ceae5-6706-4336-9952-b120fc508c60
# ╟─00090005-0000-4000-8000-000000000000
# ╟─00100001-0000-4000-8000-000000000000
# ╟─13043885-3fcc-439c-878c-91c663efcd5f
# ╟─00100002-0000-4000-8000-000000000000
# ╟─00100003-0000-4000-8000-000000000000
# ╟─00100004-0000-4000-8000-000000000000
# ╟─00100005-0000-4000-8000-000000000000
