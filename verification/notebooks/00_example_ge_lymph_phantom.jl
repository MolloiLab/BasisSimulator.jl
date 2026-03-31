### A Pluto.jl notebook ###
# v0.19.0

# ╔═╡ 00010001-0000-4000-8000-000000000003
begin
    using Pkg: Pkg
    Pkg.activate(dirname(@__DIR__))
    using Revise
end

# ╔═╡ 14530566-ecaa-4345-9115-e75981f3837f
using Metal # Choose one or the other

# ╔═╡ 00010003-0000-4000-8000-000000000003
using LinearAlgebra

# ╔═╡ 00010004-0000-4000-8000-000000000003
using FFTW

# ╔═╡ 00010005-0000-4000-8000-000000000003
using Random

# ╔═╡ 00010013-0000-4000-8000-000000000003
import Unitful
using Unitful: @u_str, ustrip

# ╔═╡ 00010014-0000-4000-8000-000000000003
using XLSX

# ╔═╡ 35b71532-be75-4fe9-ada8-9d6d4bf21f8e
using Markdown

# ╔═╡ 0b629a9c-722c-47c5-900e-5e5b311a744a
# using CUDA # Choose one or the other

# ╔═╡ 00010006-0000-4000-8000-000000000003
import BasisSimulator as BS

# ╔═╡ 00010007-0000-4000-8000-000000000003
import CairoMakie as CM

# ╔═╡ 00010008-0000-4000-8000-000000000003
import Statistics: mean, std

# ╔═╡ 00010009-0000-4000-8000-000000000003
import XrayAttenuation as XA

# ╔═╡ 00010010-0000-4000-8000-000000000003
const RESULTS_DIR = joinpath(dirname(@__DIR__), "results", "example_ge_lymph"); mkpath(RESULTS_DIR)

# ╔═╡ 00010011-0000-4000-8000-000000000003
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

    # ── Phantom loading helpers ──

    """Load XCAT phantom from raw binary file (Float32 labels)."""
    function load_phantom_bin(filepath::String;
            cols=922, rows=922, slices=1178, dtype::Type=Float32,
            reverse_dims::Tuple=(2,3))
        expected = cols * rows * slices * sizeof(dtype)
        actual = filesize(filepath)
        @assert actual == expected "File size mismatch: expected $expected, got $actual"
        data = Vector{dtype}(undef, cols * rows * slices)
        open(filepath, "r") do io; read!(io, data); end
        phantom = reshape(data, (cols, rows, slices))
        for d in reverse_dims
            phantom = reverse(phantom, dims=d)
        end
        phantom
    end

    """Downsample phantom by integer factor using nearest-neighbor (preserves labels)."""
    function downsample_phantom(phantom::AbstractArray{T,3}, factor::Int) where T
        factor == 1 && return phantom
        new_size = size(phantom) .÷ factor
        result = similar(phantom, new_size)
        for k in 1:new_size[3], j in 1:new_size[2], i in 1:new_size[1]
            oi = (i-1)*factor + factor÷2 + 1
            oj = (j-1)*factor + factor÷2 + 1
            ok = (k-1)*factor + factor÷2 + 1
            result[i,j,k] = phantom[oi,oj,ok]
        end
        result
    end

    # ── Material helpers ──

    const _ATOMIC_MASSES = Dict(
        1=>1.008, 6=>12.011, 7=>14.007, 8=>15.999, 11=>22.990, 12=>24.305,
        15=>30.974, 16=>32.06, 17=>35.45, 19=>39.098, 20=>40.078, 26=>55.845, 53=>126.904)
    const _I_VALUES = Dict(
        1=>19.2, 6=>81.0, 7=>82.0, 8=>95.0, 11=>149.0, 12=>156.0,
        15=>173.0, 16=>180.0, 17=>174.0, 19=>190.0, 20=>191.0, 26=>286.0, 53=>491.0)

    function _compute_ZA(comp::Dict{Int,Float64})
        z_sum = sum(w * Z / get(_ATOMIC_MASSES, Z, Float64(Z)*2) for (Z,w) in comp)
        z_sum / sum(values(comp))
    end

    function _compute_I(comp::Dict{Int,Float64})
        log_I = sum(w * (Z / get(_ATOMIC_MASSES, Z, Float64(Z)*2)) * log(get(_I_VALUES, Z, 10.0*Z)) for (Z,w) in comp)
        za = sum(w * (Z / get(_ATOMIC_MASSES, Z, Float64(Z)*2)) for (Z,w) in comp)
        exp(log_I / za) * u"eV"
    end

    """Load materials from XCAT xlsx spreadsheet (organ_id → XA.Material)."""
    function load_materials_from_xlsx(xlsx_path::String; strip_iodine::Bool=false)
        xf = XLSX.readxlsx(xlsx_path)
        sheet = xf["Sheet1"]
        data = sheet["A2:P110"]
        materials = Dict{Int, XA.Material}()
        _sf(val) = (val === nothing ? 0.0 : val isa Number ? Float64(val) : (r = tryparse(Float64, strip(string(val))); r === nothing ? 0.0 : r))
        _si(val) = (val === nothing ? nothing : val isa Integer ? Int(val) : tryparse(Int, strip(string(val))))
        for i in 1:size(data, 1)
            try
                raw_name = data[i, 1]
                (raw_name === nothing || isempty(strip(string(raw_name)))) && continue
                name = string(raw_name)
                organ_id = _si(data[i, 16])
                organ_id === nothing && continue
                density = _sf(data[i, 15]) * u"g/cm^3"
                comp = Dict{Int,Float64}(
                    1=>_sf(data[i,2]), 6=>_sf(data[i,3]), 7=>_sf(data[i,4]), 8=>_sf(data[i,5]),
                    11=>_sf(data[i,6]), 12=>_sf(data[i,7]), 15=>_sf(data[i,8]), 16=>_sf(data[i,9]),
                    17=>_sf(data[i,10]), 19=>_sf(data[i,11]), 20=>_sf(data[i,12]), 26=>_sf(data[i,13]),
                    53=>_sf(data[i,14]))
                if strip_iodine
                    iodine_frac = get(comp, 53, 0.0)
                    if iodine_frac > 0
                        delete!(comp, 53)
                        # Renormalize remaining fractions
                        total = sum(values(comp))
                        if total > 0
                            for Z in keys(comp); comp[Z] /= total; end
                        end
                    end
                end
                filter!(p -> p.second > 0, comp)
                isempty(comp) && continue
                materials[organ_id] = XA.Material(name, _compute_ZA(comp), _compute_I(comp), density, comp)
            catch e
                @warn "Failed to parse xlsx row $i" exception=e
            end
        end
        materials
    end
end

# ╔═╡ 00020001-0000-4000-8000-000000000003
md"""
# GE Revolution Apex Elite — Lymph Phantom (XCAT)

Two scans of the same XCAT lymphatic phantom:
1. **Non-contrast** — baseline (materials with iodine stripped)
2. **Contrast-enhanced** — iodine at t=90 s

Both at 120 kVp / 150 mA with 160 mm collimation.
Reconstructed with FBP.

**Phantom:** XCAT lymphatic phantom (922×922×1178, Float32 labels).
Voxel size: 0.075×0.075×0.15 cm (0.75×0.75×1.5 mm).
Materials from xlsx spreadsheet with elemental compositions.
"""

# ╔═╡ 00030001-0000-4000-8000-000000000003
md"""
## 1. Phantom Setup

> **Data:** Copy files from `/Volumes/Molloilab/Ayemon/XCAT_lymphatic_phantom/`
> into `verification/data/lymph_flow/`:
> - `phantom_iodine_t0090s.bin` (3.7 GB)
> - `material_map_t0090s.xlsx` (11 KB)
"""

# ╔═╡ 00030002-0000-4000-8000-000000000003
begin
    PHANTOM_DIR = joinpath(dirname(@__DIR__), "data", "lymph_flow")
    PHANTOM_BIN_PATH = joinpath(PHANTOM_DIR, "phantom_iodine_t0090s.bin")
    MATERIAL_XLSX_PATH = joinpath(PHANTOM_DIR, "material_map_t0090s.xlsx")

    @assert isfile(PHANTOM_BIN_PATH) "Phantom bin not found: $PHANTOM_BIN_PATH"
    @assert isfile(MATERIAL_XLSX_PATH) "Material xlsx not found: $MATERIAL_XLSX_PATH"
    "Paths verified"
end

# ╔═╡ 00030003-0000-4000-8000-000000000003
# Load full phantom and downsample
begin
    # Resolution control — change factor to trade speed vs quality
    DOWNSAMPLE_FACTOR = 4  # 2→ 461×461×589 (4× faster)

    phantom_labeled_raw = load_phantom_bin(PHANTOM_BIN_PATH);

    # Crop z to lymphatic region (cisterna chyli / thoracic duct)
    # Full phantom is 1178 slices ≈ 176.7 cm (full body)
    z_start = 356  # cisterna chyli region
    z_end = 535    # thoracic duct / upper abdomen (180 slices, divisible by 1,2,4,6)
    cropped_raw = phantom_labeled_raw[:, :, z_start:z_end]
    phantom_labeled_raw = nothing  # free 3.7 GB

    phantom_labeled = downsample_phantom(cropped_raw, DOWNSAMPLE_FACTOR)
    cropped_raw = nothing

    @info "Phantom: $(size(phantom_labeled)), downsample=$(DOWNSAMPLE_FACTOR)×"
end;

# ╔═╡ 00030004-0000-4000-8000-000000000003
# Load materials: contrast (with iodine) and non-contrast (iodine stripped)
begin
    materials_contrast = load_materials_from_xlsx(MATERIAL_XLSX_PATH; strip_iodine=false)
    materials_noncontrast = load_materials_from_xlsx(MATERIAL_XLSX_PATH; strip_iodine=true)

    unique_labels = sort(Int.(unique(phantom_labeled)))
    missing_contrast = setdiff(unique_labels, keys(materials_contrast))
    @info "Materials: $(length(materials_contrast)) loaded, $(length(missing_contrast)) missing labels"
end;

# ╔═╡ 00030005-0000-4000-8000-000000000003
# Build GPU phantoms
begin
    # Voxel size scales with downsample factor
    base_voxel_cm = (0.075, 0.075, 0.15)
    voxel_size_cm = base_voxel_cm .* DOWNSAMPLE_FACTOR
    lymph_fov_x = size(phantom_labeled, 1) * voxel_size_cm[1]
    lymph_fov_y = size(phantom_labeled, 2) * voxel_size_cm[2]
    lymph_fov_z = size(phantom_labeled, 3) * voxel_size_cm[3]

    # Use max label to determine mask type (UInt8 if ≤255, else UInt16)
    max_label = maximum(unique_labels)
    mask_type = max_label <= 255 ? UInt8 : UInt16
    phantom_mask_gpu = to_gpu(mask_type.(phantom_labeled))

    phantom_noncontrast = BS.Phantom(phantom_mask_gpu, materials_noncontrast, voxel_size_cm)
    phantom_contrast = BS.Phantom(phantom_mask_gpu, materials_contrast, voxel_size_cm)

    @info "Phantom FOV: $(round(lymph_fov_x, digits=1))×$(round(lymph_fov_y, digits=1))×$(round(lymph_fov_z, digits=1)) cm, mask=$(mask_type)"
end;

# ╔═╡ 00030006-0000-4000-8000-000000000003
# Phantom mid-slice visualization
let
    mid = size(phantom_labeled, 3) ÷ 2
    fig = CM.Figure(size = (800, 700), fontsize = 12)
    ax = CM.Axis(fig[1, 1];
        title = "XCAT Lymph Phantom — Slice $mid / $(size(phantom_labeled, 3))",
        aspect = CM.DataAspect())
    CM.heatmap!(ax, Float32.(phantom_labeled[:, :, mid]); colormap = :glasbey_hv_n256)
    CM.save(joinpath(RESULTS_DIR, "phantom.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00040001-0000-4000-8000-000000000003
md"""
## 2. Scanner & X-ray Spectrum
"""

# ╔═╡ 00040002-0000-4000-8000-000000000003
additional_filters = [("Al", 4.5)]

# ╔═╡ 00040003-0000-4000-8000-000000000003
begin
    sim_electronic_noise = 0
    sim_detection_gain = 10.0
    lymph_scan_fov_cm = max(lymph_fov_x, lymph_fov_y)  # match phantom lateral extent
    lymph_mag = 1100.0 / 625.6
    lymph_det_cols = ceil(Int, lymph_scan_fov_cm * 10.0 * lymph_mag / 0.6)

    sim_scanner = BS.Scanner(
        source_to_isocenter = 625.6,
        source_to_detector = 1100.0,
        detector_rows = 256,
        detector_cols = lymph_det_cols,
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

# ╔═╡ 00040004-0000-4000-8000-000000000003
sim_opts = BS.SimOptions(fidelity = :eict, seed = 1234)

# ╔═╡ 00040005-0000-4000-8000-000000000003
# X-ray spectrum visualization
let
    kVp = 120
    e_raw, w_raw = BS.load_spectrum_unfiltered(kVp; anode_angle = 10)
    all_filters = vcat([("Al", 2.5)], additional_filters)
    _, w_filt = BS.filter_spectrum(e_raw, w_raw; filters = all_filters, sdd_mm = 1100.0)
    prot = BS.CTProtocol(kVp = kVp, additional_filters = additional_filters)
    e_res, w_res = BS.resolve_spectrum(sim_opts, prot; scanner = sim_scanner)
    E_eff = sum(e_res .* w_res) / sum(w_res)

    fig = CM.Figure(size = (900, 500), fontsize = 13)
    ax = CM.Axis(fig[1, 1]; title = "120 kVp X-ray Spectrum (Lymph)", xlabel = "Energy (keV)", ylabel = "Relative Fluence")
    CM.lines!(ax, e_raw, w_raw ./ maximum(w_raw); color = :gray60, linewidth = 1.5, label = "Raw tube")
    CM.lines!(ax, e_raw, w_filt ./ maximum(w_filt); color = :steelblue, linewidth = 2, label = "After 7.0 mm Al")
    CM.lines!(ax, e_res, w_res ./ maximum(w_res); color = :darkorange, linewidth = 2, label = "Resolved")
    CM.vlines!(ax, [E_eff]; color = :red, linestyle = :dash, linewidth = 1, label = "E_eff=$(round(E_eff, digits=1)) keV")
    CM.xlims!(ax, 15, 125); CM.ylims!(ax, 0, 1.05)
    CM.axislegend(ax; position = :rt)
    CM.save(joinpath(RESULTS_DIR, "spectrum_120kVp.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00050001-0000-4000-8000-000000000003
md"""
## 3. Acquisition Protocol
"""

# ╔═╡ 00050002-0000-4000-8000-000000000003
begin
    sim_rotation_time = 1.0
    sim_collimation_mm = 160.0  # 256 × 0.625 mm = full detector coverage
    sim_n_views = 984
end

# ╔═╡ 00050003-0000-4000-8000-000000000003
begin
    # Match recon to phantom resolution
    sim_recon_xy = min(512, minimum(size(phantom_labeled)[1:2]))
    sim_recon_fov_cm = lymph_scan_fov_cm
    sim_slice_thickness_mm = voxel_size_cm[3] * 10.0  # match phantom z voxel
    sim_recon_z_cm = lymph_fov_z
    sim_n_recon_slices = round(Int, sim_collimation_mm / sim_slice_thickness_mm)
    sim_matrix_size = (sim_recon_xy, sim_recon_xy, sim_n_recon_slices)

    sim_recon_geom = BS.ReconOptions(
        algorithm = :fdk,
        matrix_size = sim_matrix_size,
        fov_cm = sim_recon_fov_cm,
        z_cm = sim_recon_z_cm,
    )
    @info "Recon: $(sim_matrix_size), FOV=$(round(sim_recon_fov_cm, digits=1)) cm, z=$(round(sim_recon_z_cm, digits=1)) cm, slice=$(sim_slice_thickness_mm) mm"
end

# ╔═╡ 00050004-0000-4000-8000-000000000003
sim_protocol = BS.CTProtocol(
    kVp = 120, mA = 150.0, views = sim_n_views,
    rotation_time = sim_rotation_time,
    collimation_mm = sim_collimation_mm,
    additional_filters = additional_filters,
)

# ╔═╡ 00060001-0000-4000-8000-000000000003
md"""
## 4. BHC Calibration
"""

# ╔═╡ 00060002-0000-4000-8000-000000000003
custom_filter_control = (x = (0.0, 0.25, 0.5, 0.75, 1.0), y = (1.0, 0.85, 0.6, 0.15, 0.001))

# ╔═╡ 00060003-0000-4000-8000-000000000003
sim_noise_floor_hu = 28.0

# ╔═╡ 00060004-0000-4000-8000-000000000003
begin
    bhc_enabled = true
    sino_bhc_hu_low = 450.0; sino_bhc_hu_high = 600.0; sino_order = 2
    img_bhc_hu_low = 50.0; img_bhc_hu_high = 150.0; img_bhc_scale_factor = 0.2
end

# ╔═╡ 00060005-0000-4000-8000-000000000003
bhc_model, μ_water_ref = let
    prot = BS.CTProtocol(kVp = 120, additional_filters = additional_filters)
    e, w = BS.resolve_spectrum(sim_opts, prot; scanner = sim_scanner)
    ref_E = sum(e .* w) / sum(w)
    model = BS.calibrate_bhc_two_material(e, w; order = sino_order, reference_energy_keV = ref_E, hu_low = sino_bhc_hu_low, hu_high = sino_bhc_hu_high)
    @info "BHC 120 kVp: E_ref=$(round(ref_E, digits=1)) keV, μ_water=$(round(model.μ_water_ref, digits=5)) cm⁻¹"
    model, model.μ_water_ref
end

# ╔═╡ 00070001-0000-4000-8000-000000000003
md"""
## 5. Forward Projection & Reconstruction

Two scans:
1. **Non-contrast** — baseline (iodine stripped from materials)
2. **Contrast** — iodine at t=90 s (from xlsx)
"""

# ╔═╡ 00070002-0000-4000-8000-000000000003
# Shared reconstruction pipeline
function _recon_pipeline!(sino_data, geom, phantom_extent)
    recon_size = sim_matrix_size
    sino_gpu = to_gpu(sino_data)

    if bhc_enabled
        BS.apply_bhc!(sino_gpu, bhc_model.water_bhc)
    end

    ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, geom, recon_size;
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y))
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size)

    if bhc_enabled
        BS.apply_bhc_image_domain(recon_μ, geom, recon_size, μ_water_ref;
            hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
            scale_factor = img_bhc_scale_factor, volume_extent = phantom_extent)
    end

    vol = Array(recon_μ)
    ws_fdk = nothing; sino_gpu = nothing; recon_μ = nothing; clear_gpu!()

    recon_hu = Float32.(BS.to_hounsfield(vol; μ_water = μ_water_ref))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    nx, ny, nz = size(recon_hu)
    cx, cy = (nx + 1) / 2, (ny + 1) / 2
    r_fov = nx / 2
    for k in 1:nz, j in 1:ny, i in 1:nx
        if (i - cx)^2 + (j - cy)^2 > r_fov^2
            recon_hu[i, j, k] = -2048f0
        end
    end
    recon_hu
end;

# ╔═╡ 00070003-0000-4000-8000-000000000003
# Scan 1: Non-contrast
sim_noncontrast_hu = let
    @info "Scan 1: Non-contrast lymph..."
    ws = BS.create_eict_workspace(sim_scanner, sim_protocol, sim_opts, sim_recon_geom, phantom_noncontrast)
    BS.simulate!(ws, phantom_noncontrast, sim_scanner, sim_protocol, sim_opts, sim_recon_geom)
    sino = Array(ws.sino_noisy_out); geom = ws.geom
    ws = nothing; clear_gpu!()
    _recon_pipeline!(sino, geom, phantom_noncontrast.extent)
end;

# ╔═╡ 00070004-0000-4000-8000-000000000003
# Scan 2: Contrast-enhanced (t = 90 s)
sim_contrast_hu = let
    @info "Scan 2: Contrast lymph (t=90s)..."
    ws = BS.create_eict_workspace(sim_scanner, sim_protocol, sim_opts, sim_recon_geom, phantom_contrast)
    BS.simulate!(ws, phantom_contrast, sim_scanner, sim_protocol, sim_opts, sim_recon_geom)
    sino = Array(ws.sino_noisy_out); geom = ws.geom
    ws = nothing; clear_gpu!()
    _recon_pipeline!(sino, geom, phantom_contrast.extent)
end;

# ╔═╡ 00080001-0000-4000-8000-000000000003
md"""
## 6. Visualization
"""

# ╔═╡ 096ba509-c17b-4d01-97b0-1ce75df58cd0
mid_z = size(sim_noncontrast_hu, 3) ÷ 2

# ╔═╡ e7d41653-97f9-4f14-b3dc-510492ea11dd
z_view = 20  # tweak this to view different slices

# ╔═╡ 00080002-0000-4000-8000-000000000003
# Non-contrast vs Contrast: soft tissue window
let
    nc_slice = sim_noncontrast_hu[:, :, z_view]
    ce_slice = sim_contrast_hu[:, :, z_view]

    fig = CM.Figure(size = (1100, 500), fontsize = 13)
    ax1 = CM.Axis(fig[1, 1]; title = "Non-contrast (slice $z_view)", aspect = CM.DataAspect())
    CM.heatmap!(ax1, nc_slice; colormap = :grays, colorrange = (-200, 500))
    ax2 = CM.Axis(fig[1, 2]; title = "Contrast t=90s (slice $z_view)", aspect = CM.DataAspect())
    hm = CM.heatmap!(ax2, ce_slice; colormap = :grays, colorrange = (-200, 500))
    CM.Colorbar(fig[1, 3], hm; label = "HU", width = 12)
    CM.save(joinpath(RESULTS_DIR, "noncontrast_vs_contrast.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080003-0000-4000-8000-000000000003
# Subtraction image: Contrast - Non-contrast
let
    diff = sim_contrast_hu[:, :, z_view] .- sim_noncontrast_hu[:, :, z_view]
    diff[sim_noncontrast_hu[:, :, z_view] .< -1500] .= 0

    fig = CM.Figure(size = (600, 550), fontsize = 13)
    ax = CM.Axis(fig[1, 1]; title = "Subtraction: Contrast − Non-contrast (slice $z_view)", aspect = CM.DataAspect())
    hm = CM.heatmap!(ax, diff; colormap = :hot, colorrange = (100, 300))
    CM.Colorbar(fig[1, 2], hm; label = "ΔHU")
    CM.save(joinpath(RESULTS_DIR, "subtraction.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080004-0000-4000-8000-000000000003
# Bone window comparison
let
    nc_slice = sim_noncontrast_hu[:, :, z_view]
    ce_slice = sim_contrast_hu[:, :, z_view]

    fig = CM.Figure(size = (1100, 500), fontsize = 13)
    ax1 = CM.Axis(fig[1, 1]; title = "Non-contrast — Bone Window", aspect = CM.DataAspect())
    CM.heatmap!(ax1, nc_slice; colormap = :grays, colorrange = (-500, 1500))
    ax2 = CM.Axis(fig[1, 2]; title = "Contrast — Bone Window", aspect = CM.DataAspect())
    hm = CM.heatmap!(ax2, ce_slice; colormap = :grays, colorrange = (-500, 1500))
    CM.Colorbar(fig[1, 3], hm; label = "HU", width = 12)
    CM.save(joinpath(RESULTS_DIR, "bone_window.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080005-0000-4000-8000-000000000003
# Noise comparison: center ROI
let
    mz = size(sim_noncontrast_hu, 3) ÷ 2 + 1
    cx, cy = size(sim_noncontrast_hu, 1) ÷ 2, size(sim_noncontrast_hu, 2) ÷ 2
    r = 15

    nc_vals = [sim_noncontrast_hu[cx+dx, cy+dy, mz] for dy in -r:r for dx in -r:r if dx^2+dy^2 <= r^2]
    ce_vals = [sim_contrast_hu[cx+dx, cy+dy, mz] for dy in -r:r for dx in -r:r if dx^2+dy^2 <= r^2]

    fig = CM.Figure(size = (500, 400), fontsize = 13)
    ax = CM.Axis(fig[1, 1]; title = "Center ROI Noise", ylabel = "σ (HU)", xticks = ([1, 2], ["Non-contrast", "Contrast"]))
    CM.barplot!(ax, [1], [std(nc_vals)]; width = 0.5, color = :steelblue,
        label = "NC: μ=$(round(mean(nc_vals), digits=1)), σ=$(round(std(nc_vals), digits=1))")
    CM.barplot!(ax, [2], [std(ce_vals)]; width = 0.5, color = :darkorange,
        label = "CE: μ=$(round(mean(ce_vals), digits=1)), σ=$(round(std(ce_vals), digits=1))")
    CM.ylims!(ax, 0, nothing)
    CM.axislegend(ax; position = :rt)
    CM.save(joinpath(RESULTS_DIR, "noise_comparison.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080006-0000-4000-8000-000000000003
# Line profile through center
let
    mz = size(sim_noncontrast_hu, 3) ÷ 2 + 1
    cy = size(sim_noncontrast_hu, 2) ÷ 2
    x_cm = range(-sim_recon_fov_cm/2, sim_recon_fov_cm/2, length=size(sim_noncontrast_hu, 1))

    fig = CM.Figure(size = (900, 400), fontsize = 13)
    ax = CM.Axis(fig[1, 1]; title = "Horizontal Line Profile (mid-slice)", xlabel = "Position (cm)", ylabel = "HU")
    CM.lines!(ax, collect(x_cm), Float64.(sim_noncontrast_hu[:, cy, mz]); color = :steelblue, linewidth = 1.5, label = "Non-contrast")
    CM.lines!(ax, collect(x_cm), Float64.(sim_contrast_hu[:, cy, mz]); color = :darkorange, linewidth = 1.5, label = "Contrast (t=90s)")
    CM.axislegend(ax; position = :rt)
    CM.save(joinpath(RESULTS_DIR, "line_profile.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ Cell order:
# ╠═00010001-0000-4000-8000-000000000003
# ╠═14530566-ecaa-4345-9115-e75981f3837f
# ╠═0b629a9c-722c-47c5-900e-5e5b311a744a
# ╠═00010003-0000-4000-8000-000000000003
# ╠═00010004-0000-4000-8000-000000000003
# ╠═00010005-0000-4000-8000-000000000003
# ╠═00010006-0000-4000-8000-000000000003
# ╠═00010007-0000-4000-8000-000000000003
# ╠═00010008-0000-4000-8000-000000000003
# ╠═00010009-0000-4000-8000-000000000003
# ╠═00010013-0000-4000-8000-000000000003
# ╠═00010014-0000-4000-8000-000000000003
# ╠═35b71532-be75-4fe9-ada8-9d6d4bf21f8e
# ╠═00010010-0000-4000-8000-000000000003
# ╠═00010011-0000-4000-8000-000000000003
# ╟─00020001-0000-4000-8000-000000000003
# ╟─00030001-0000-4000-8000-000000000003
# ╠═00030002-0000-4000-8000-000000000003
# ╠═00030003-0000-4000-8000-000000000003
# ╠═00030004-0000-4000-8000-000000000003
# ╠═00030005-0000-4000-8000-000000000003
# ╟─00030006-0000-4000-8000-000000000003
# ╟─00040001-0000-4000-8000-000000000003
# ╠═00040002-0000-4000-8000-000000000003
# ╠═00040003-0000-4000-8000-000000000003
# ╠═00040004-0000-4000-8000-000000000003
# ╟─00040005-0000-4000-8000-000000000003
# ╟─00050001-0000-4000-8000-000000000003
# ╠═00050002-0000-4000-8000-000000000003
# ╠═00050003-0000-4000-8000-000000000003
# ╠═00050004-0000-4000-8000-000000000003
# ╟─00060001-0000-4000-8000-000000000003
# ╠═00060002-0000-4000-8000-000000000003
# ╠═00060003-0000-4000-8000-000000000003
# ╠═00060004-0000-4000-8000-000000000003
# ╠═00060005-0000-4000-8000-000000000003
# ╟─00070001-0000-4000-8000-000000000003
# ╠═00070002-0000-4000-8000-000000000003
# ╠═00070003-0000-4000-8000-000000000003
# ╠═00070004-0000-4000-8000-000000000003
# ╟─00080001-0000-4000-8000-000000000003
# ╠═096ba509-c17b-4d01-97b0-1ce75df58cd0
# ╠═e7d41653-97f9-4f14-b3dc-510492ea11dd
# ╟─00080002-0000-4000-8000-000000000003
# ╟─00080003-0000-4000-8000-000000000003
# ╟─00080004-0000-4000-8000-000000000003
# ╟─00080005-0000-4000-8000-000000000003
# ╟─00080006-0000-4000-8000-000000000003
