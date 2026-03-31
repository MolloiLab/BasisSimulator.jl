### A Pluto.jl notebook ###
# v0.19.0

# ╔═╡ 00010001-0000-4000-8000-000000000002
begin
    using Pkg: Pkg
    Pkg.activate(dirname(@__DIR__))
    using Revise
end

# ╔═╡ 14530566-ecaa-4345-9115-e75981f3837e
using Metal # Choose one or the other

# ╔═╡ 00010003-0000-4000-8000-000000000002
using LinearAlgebra

# ╔═╡ 00010004-0000-4000-8000-000000000002
using FFTW

# ╔═╡ 00010005-0000-4000-8000-000000000002
using Random

# ╔═╡ 00010012-0000-4000-8000-000000000002
using MAT

# ╔═╡ 35b71532-be75-4fe9-ada8-9d6d4bf21f7e
using Markdown

# ╔═╡ 0b629a9c-722c-47c5-900e-5e5b311a743a
# using CUDA # Choose one or the other

# ╔═╡ 00010006-0000-4000-8000-000000000002
import BasisSimulator as BS

# ╔═╡ 00010007-0000-4000-8000-000000000002
import CairoMakie as CM

# ╔═╡ 00010008-0000-4000-8000-000000000002
import Statistics: mean, std

# ╔═╡ 00010009-0000-4000-8000-000000000002
import XrayAttenuation as XA

# ╔═╡ 00010013-0000-4000-8000-000000000002
import Unitful

# ╔═╡ 00010010-0000-4000-8000-000000000002
const RESULTS_DIR = joinpath(dirname(@__DIR__), "results", "example_ge_brain"); mkpath(RESULTS_DIR)

# ╔═╡ 00010011-0000-4000-8000-000000000002
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

    # ── Inline helpers (from phantom-loading branch, not on main) ──

    """Load XCAT voxelize table: tab-separated `name\\tID` per line → Dict{Int,String}."""
    function load_structure_map(path::String)::Dict{Int, String}
        result = Dict{Int, String}()
        open(path, "r") do io
            for line in eachline(io)
                parts = split(strip(line), '\t')
                length(parts) == 2 || continue
                name = strip(parts[1])
                id = tryparse(Int, strip(parts[2]))
                (id === nothing || isempty(name)) && continue
                result[id] = name
            end
        end
        result
    end

    const _ATOMIC_MASSES = Dict(
        1=>1.008, 6=>12.011, 7=>14.007, 8=>15.999, 11=>22.990, 12=>24.305,
        15=>30.974, 16=>32.06, 17=>35.45, 19=>39.098, 20=>40.078, 26=>55.845, 53=>126.904)
    const _I_VALUES = Dict(
        1=>19.2, 6=>81.0, 7=>82.0, 8=>95.0, 11=>149.0, 12=>156.0,
        15=>173.0, 16=>180.0, 17=>174.0, 19=>190.0, 20=>191.0, 26=>286.0, 53=>491.0)

    """Create iodine-doped material from base tissue + concentration (mg/mL)."""
    function iodine_contrast_material(base_mat::XA.Material, conc_mg_per_mL::Float64;
            density_g_per_mL::Union{Nothing,Float64}=nothing)::XA.Material
        conc_mg_per_mL < 1e-6 && return base_mat
        rho = density_g_per_mL !== nothing ? density_g_per_mL :
              Unitful.ustrip(Unitful.u"g/cm^3", base_mat.density)
        f_I = min((conc_mg_per_mL / 1000.0) / rho, 1.0)
        f_s = 1.0 - f_I
        base_comp = Dict{Int,Float64}(base_mat.composition)
        new_comp = Dict{Int,Float64}(Z => w * f_s for (Z, w) in base_comp)
        new_comp[53] = get(new_comp, 53, 0.0) + f_I
        new_density = rho + conc_mg_per_mL / 1000.0
        log_I_num = sum(new_comp[Z] * (Z / get(_ATOMIC_MASSES, Z, Float64(Z)*2)) * log(get(_I_VALUES, Z, 10.0*Z)) for Z in keys(new_comp))
        log_I_den = sum(new_comp[Z] * (Z / get(_ATOMIC_MASSES, Z, Float64(Z)*2)) for Z in keys(new_comp))
        I_mean = exp(log_I_num / log_I_den) * Unitful.u"eV"
        ZA = sum(w * Z / get(_ATOMIC_MASSES, Z, Float64(Z)*2) for (Z, w) in new_comp)
        XA.Material("$(base_mat.name)_$(round(Int, conc_mg_per_mL))mgI", ZA, I_mean, new_density * Unitful.u"g/cm^3", new_comp)
    end

    # Brain tissue materials (Woodard & White 1986 / ICRU-44)
    function _zoa(comp); sum(w*Z/get(_ATOMIC_MASSES,Z,Float64(Z)*2) for (Z,w) in comp)/sum(values(comp)); end
    function _mex(comp); exp(sum(w*(Z/get(_ATOMIC_MASSES,Z,Float64(Z)*2))*log(get(_I_VALUES,Z,10.0*Z)) for (Z,w) in comp) / sum(w*(Z/get(_ATOMIC_MASSES,Z,Float64(Z)*2)) for (Z,w) in comp))*Unitful.u"eV"; end
    const _gm_comp = Dict{Int,Float64}(1=>0.1070,6=>0.0955,7=>0.0180,8=>0.7620,11=>0.0020,15=>0.0035,16=>0.0020,17=>0.0030,19=>0.0070)
    const _wm_comp = Dict{Int,Float64}(1=>0.1060,6=>0.1980,7=>0.0130,8=>0.6703,11=>0.0020,15=>0.0040,16=>0.0017,17=>0.0020,19=>0.0030)
    const _LOCAL_GM = XA.Material("Gray Matter", _zoa(_gm_comp), _mex(_gm_comp), 1.04*Unitful.u"g/cm^3", _gm_comp)
    const _LOCAL_WM = XA.Material("White Matter", _zoa(_wm_comp), _mex(_wm_comp), 1.04*Unitful.u"g/cm^3", _wm_comp)

    const _BRAIN_MATERIALS = Dict{Symbol, XA.Material}(
        :air => XA.Materials.air,
        :water => XA.Materials.water,
        :bone => XA.Materials.corticalbone,
        :blood => XA.Materials.blood,
        :muscle => XA.Materials.muscle,
        :soft_tissue => XA.Materials.softtissue,
        :csf => XA.Materials.cerebrospinal_fluid,
        :gray_matter => _LOCAL_GM,
        :white_matter => _LOCAL_WM,
    )

    """Look up a material by symbol (brain-specific registry)."""
    function local_get_material(sym::Symbol)::XA.Material
        haskey(_BRAIN_MATERIALS, sym) && return _BRAIN_MATERIALS[sym]
        hasproperty(XA.Materials, sym) && return getproperty(XA.Materials, sym)
        error("Material :$sym not found")
    end

    """Apply iodine contrast to P2 segments matching a tissue prefix at a given time index."""
    function update_structures!(
            P1_stamped::Array{Int,3}, structure_map::Dict{Int,String},
            prefix::String, P2_crop::Array{Int,3},
            base_sym::Symbol, base_mat::XA.Material,
            info::Dict{String,Any}, iodine::Matrix{Float64}, t_idx::Int;
            index_map::Union{Nothing, Dict{Int, Vector{CartesianIndex{3}}}}=nothing)::Dict{Int, XA.Material}
        entries = filter(kv -> startswith(kv[2], prefix), structure_map)
        sorted = sort(collect(entries), by=kv->kv[1])
        ids = [kv[1] for kv in sorted]
        names = [kv[2] for kv in sorted]
        for id in ids
            idxs = index_map !== nothing ? get(index_map, id, CartesianIndex{3}[]) : findall(==(id), P2_crop)
            isempty(idxs) && continue
            P1_stamped[idxs] .= id
        end
        seg_mats = Dict{Int, XA.Material}()
        density = Unitful.ustrip(Unitful.u"g/cm^3", base_mat.density)
        is_vessel = (base_sym == :blood)
        n_rows = size(iodine, 1)
        for (id, raw_name) in zip(ids, names)
            m = match(r"^\d(\d{3})_", raw_name)
            m === nothing && continue
            row = parse(Int, m.captures[1])
            (row < 1 || row > n_rows) && continue
            row > length(info["volume"]) && continue
            vol_mL = is_vessel ? info["volume"][row] / 1000.0 : info["volume"][row]
            vol_mL <= 0 && continue
            conc = iodine[row, t_idx] / vol_mL
            seg_mats[id] = iodine_contrast_material(base_mat, conc; density_g_per_mL=density)
        end
        seg_mats
    end
end

# ╔═╡ 00020001-0000-4000-8000-000000000002
md"""
# GE Revolution Apex Elite — Brain Phantom (XCAT)

Two scans of the same brain phantom:
1. **Non-contrast** — baseline brain CT
2. **Contrast-enhanced** — with iodine at peak arterial enhancement (~15 s)

Both at 120 kVp / 150 mA with 140 mm collimation.
Reconstructed with both FBP and HIR (half-iterative reconstruction).

**Phantom:** XCAT P1/P2 brain phantom (400×400×400, 1 mm voxels).
Materials: skull bone, gray matter, white matter, CSF, blood, muscle, air.
Contrast: iodine mass data from Divel et al. (2021), *Med. Phys.*
"""

# ╔═╡ 00030001-0000-4000-8000-000000000002
md"""
## 1. Phantom Setup — XCAT Brain (P1/P2 raw files)

Loads the XCAT P1 material phantom and P2 segment phantom.
P1 provides material labels; P2 provides anatomical segment IDs
for applying per-structure iodine contrast.

> **Data:** Copy files from the group share drive into
> `verification/data/brain_perfusion/`:
> - `P1_brain_all_2020_RAW_400_400_400.raw` (122 MB)
> - `P1_voxelize_table.txt`
> - `P2_brain_all_2020_RAW_400_400_400.raw` (122 MB)
> - `P2_vozelize_table.txt`
> - `structure_info.mat`
> - `iodine_mass_data.mat` (501 MB)
"""

# ╔═╡ 00030002-0000-4000-8000-000000000002
begin
    PHANTOM_DIR = joinpath(dirname(@__DIR__), "data", "brain_perfusion")
    P1_RAW_PATH = joinpath(PHANTOM_DIR, "P1_brain_all_2020_RAW_400_400_400.raw")
    P1_TABLE_PATH = joinpath(PHANTOM_DIR, "P1_voxelize_table.txt")
    P2_RAW_PATH = joinpath(PHANTOM_DIR, "P2_brain_all_2020_RAW_400_400_400.raw")
    P2_TABLE_PATH = joinpath(PHANTOM_DIR, "P2_vozelize_table.txt")
    STRUCT_INFO_PATH = joinpath(PHANTOM_DIR, "structure_info.mat")
    IODINE_DATA_PATH = joinpath(PHANTOM_DIR, "iodine_mass_data.mat")

    @assert isfile(P1_RAW_PATH) "P1 raw not found: $P1_RAW_PATH"
    @assert isfile(P2_RAW_PATH) "P2 raw not found: $P2_RAW_PATH"
    @assert isfile(IODINE_DATA_PATH) "iodine_mass_data.mat not found: $IODINE_DATA_PATH"

    MATERIAL_MAP = Dict{Int, Symbol}(
        0  => :air,       1  => :muscle,    2  => :air,
        3  => :bone,      5  => :soft_tissue, 10 => :soft_tissue,
        13 => :bone,      14 => :soft_tissue,
        17 => :csf,       18 => :gray_matter, 19 => :white_matter,
        21 => :blood,     22 => :blood,
    )

    MATERIAL_INFO = Dict(
        0  => (name = "Air", color = :gray15),
        1  => (name = "Muscle", color = :salmon),
        2  => (name = "Airway", color = :gray30),
        3  => (name = "Spine Bone", color = :wheat),
        5  => (name = "Soft Tissue", color = :lightskyblue),
        10 => (name = "Interior Fill", color = :paleturquoise),
        13 => (name = "Skull Bone", color = :sandybrown),
        14 => (name = "Disk", color = :thistle),
        17 => (name = "CSF", color = :royalblue),
        18 => (name = "Gray Matter", color = :lightgreen),
        19 => (name = "White Matter", color = :mediumseagreen),
        21 => (name = "Arteries", color = :red),
        22 => (name = "Veins", color = :darkblue),
    )
end;

# ╔═╡ 00030003-0000-4000-8000-000000000002
# Load P1 (materials) and P2 (segments)
begin
    P1_raw = let
        buf = Array{UInt16}(undef, 400, 400, 400)
        open(P1_RAW_PATH, "r") do io; read!(io, buf); end
        arr = Int.(buf)
        arr = reverse(arr, dims=2)
        # Note: interior zero-island relabeling skipped here —
        # the P2 segment stamping in _build_phantom() fills interior
        # voxels with proper segment IDs, and remaining isolated zeros
        # (air pockets) are physically correct for the simulation.
        arr
    end

    P2_raw = let
        buf = Array{UInt16}(undef, 400, 400, 400)
        open(P2_RAW_PATH, "r") do io; read!(io, buf); end
        arr = Int.(buf)
        reverse(arr, dims=2)
    end

    P1_structure_map = load_structure_map(P1_TABLE_PATH)
    P2_structure_map = load_structure_map(P2_TABLE_PATH)

    # Base material list
    unique_ids = sort(unique(vec(P1_raw)))
    material_map_init = Dict{Int, Symbol}(
        id => MATERIAL_MAP[id] for id in unique_ids if haskey(MATERIAL_MAP, id)
    )
    material_list_init = Dict{Symbol, XA.Material}(
        sym => local_get_material(sym) for sym in unique(collect(values(material_map_init)))
    )

    @info "Brain phantom: $(size(P1_raw)), $(length(unique_ids)) unique P1 labels"
end;

# ╔═╡ 00030004-0000-4000-8000-000000000002
# Crop to brain region along z (auto-detect tissue extent, clamp to 140 mm)
begin
    brain_vox_cm = 0.1  # 1 mm voxels
    brain_collimation_mm = 140.0

    _z_any = vec(any(P1_raw .!= 0, dims=(1, 2)))
    _z_first = findfirst(_z_any)
    _z_last = findlast(_z_any)
    _z_tissue = max(1, _z_first - 4) : min(size(P1_raw, 3), _z_last + 4)

    max_z_slices = round(Int, brain_collimation_mm / (brain_vox_cm * 10.0))
    if length(_z_tissue) > max_z_slices
        _z_mid = (_z_tissue[1] + _z_tissue[end]) ÷ 2
        _z_half = max_z_slices ÷ 2
        BRAIN_Z_CROP = max(1, _z_mid - _z_half + 1) : min(size(P1_raw, 3), _z_mid + _z_half)
    else
        BRAIN_Z_CROP = _z_tissue
    end

    P1_cropped = P1_raw[:, :, BRAIN_Z_CROP]
    P2_cropped = P2_raw[:, :, BRAIN_Z_CROP]
    brain_z_cm = length(BRAIN_Z_CROP) * brain_vox_cm

    @info "Z crop: slices $(BRAIN_Z_CROP) ($(length(BRAIN_Z_CROP)) slices, $(round(brain_z_cm, digits=1)) cm)"
end;

# ╔═╡ 00030005-0000-4000-8000-000000000002
# Load iodine contrast data + structure info
begin
    si = matread(STRUCT_INFO_PATH)
    function _parse_info(d)
        names = vec(String.(d["name"]))
        vols = vec(Float64.(d["volume"]))
        Dict{String, Any}("name" => names, "volume" => vols)
    end
    artery_info = _parse_info(si["artery_info"])
    vein_info = _parse_info(si["vein_info"])
    gm_info = _parse_info(si["gm_info"])
    wm_info = _parse_info(si["wm_info"])

    iodine_data = matread(IODINE_DATA_PATH)
    iodine_artery = Float64.(iodine_data["mass_arteries"])
    iodine_vein = Float64.(iodine_data["mass_vein"])
    iodine_gm = Float64.(iodine_data["mass_gm"])
    iodine_wm = Float64.(iodine_data["mass_wm"])

    @info "Iodine data: arteries=$(size(iodine_artery,1)) segs, $(size(iodine_artery,2)) time points"
end;

# ╔═╡ 00030006-0000-4000-8000-000000000002
# Build non-contrast phantom (Scan 1) and contrast phantom (Scan 2)
begin
    # Pre-build segment -> voxel index map
    segment_index_map = Dict{Int, Vector{CartesianIndex{3}}}()
    for (id, name) in P2_structure_map
        first(name) in ('2', '3', '4', '5') || continue
        idxs = findall(==(id), P2_cropped)
        isempty(idxs) || (segment_index_map[id] = idxs)
    end

    # Stamp segment IDs into P1
    P1_stamped = copy(P1_cropped)
    for (id, idxs) in segment_index_map
        P1_stamped[idxs] .= id
    end

    mask_gpu = to_gpu(UInt16.(P1_stamped))
    dx, dy, dz = brain_vox_cm, brain_vox_cm, brain_vox_cm
    nx, ny, nz = size(P1_stamped)

    # Map P2 segment prefix → base tissue symbol
    _PREFIX_TO_BASE = Dict("5" => :blood, "4" => :blood, "3" => :white_matter, "2" => :gray_matter)

    function _build_phantom(; contrast_time_idx=nothing)
        materials_dict = Dict{Int, XA.Material}(
            id => material_list_init[sym] for (id, sym) in material_map_init
        )

        # Always map ALL stamped P2 segments to their base tissue material
        # (prevents black holes from unmapped segment IDs)
        for (id, name) in P2_structure_map
            id ∈ keys(segment_index_map) || continue
            c = string(first(name))
            c ∈ keys(_PREFIX_TO_BASE) || continue
            materials_dict[id] = material_list_init[_PREFIX_TO_BASE[c]]
        end

        if contrast_time_idx !== nothing
            # Override with iodine-doped materials at specified time index
            for (prefix, info, iodine, base_sym) in [
                ("5", artery_info, iodine_artery, :blood),
                ("4", vein_info, iodine_vein, :blood),
                ("3", wm_info, iodine_wm, :white_matter),
                ("2", gm_info, iodine_gm, :gray_matter),
            ]
                seg_mats = update_structures!(
                    P1_stamped, P2_structure_map, prefix, P2_cropped,
                    base_sym, material_list_init[base_sym], info, iodine, contrast_time_idx;
                    index_map = segment_index_map
                )
                merge!(materials_dict, seg_mats)
            end
        end

        BS.Phantom(mask_gpu, BS.build_materials_vector(materials_dict),
            (dx, dy, dz),
            (-dx*nx/2 + dx/2, -dy*ny/2 + dy/2, -dz*nz/2 + dz/2),
            (dx*nx, dy*ny, dz*nz))
    end

    # Scan 1: Non-contrast
    phantom_noncontrast = _build_phantom()

    # Scan 2: Contrast at peak arterial enhancement (~15 s = index 15001)
    contrast_time_s = 15  # seconds — peak arterial enhancement
    contrast_time_idx = contrast_time_s * 1000 + 1
    phantom_contrast = _build_phantom(; contrast_time_idx)

    @info "Phantoms built: non-contrast + contrast at t=$(contrast_time_s) s"
end;

# ╔═╡ 00030007-0000-4000-8000-000000000002
# Phantom mid-slice visualization
let
    mid = size(P1_stamped, 3) ÷ 2
    slice_data = P1_stamped[:, :, mid]

    unique_labels = sort(unique(slice_data))
    n_labels = min(length(unique_labels), 20)  # cap for colorbar

    fig = CM.Figure(size = (800, 700), fontsize = 12)
    ax = CM.Axis(fig[1, 1];
        title = "XCAT Brain Phantom — Slice $mid / $(size(P1_stamped, 3))",
        aspect = CM.DataAspect())
    CM.heatmap!(ax, Float32.(slice_data); colormap = :tab20)
    CM.save(joinpath(RESULTS_DIR, "phantom.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00040001-0000-4000-8000-000000000002
md"""
## 2. Scanner & X-ray Spectrum
"""

# ╔═╡ 00040002-0000-4000-8000-000000000002
additional_filters = [("Al", 4.5)]

# ╔═╡ 00040003-0000-4000-8000-000000000002
begin
    sim_electronic_noise = 0
    sim_detection_gain = 10.0
    brain_fov_cm = 25.0
    brain_mag = 1100.0 / 625.6
    brain_det_cols = ceil(Int, brain_fov_cm * 10.0 * brain_mag / 0.6)

    sim_scanner = BS.Scanner(
        source_to_isocenter = 625.6,
        source_to_detector = 1100.0,
        detector_rows = 256,
        detector_cols = brain_det_cols,
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

# ╔═╡ 00040004-0000-4000-8000-000000000002
sim_opts = BS.SimOptions(fidelity = :eict, seed = 1234)

# ╔═╡ 00040005-0000-4000-8000-000000000002
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
    ax = CM.Axis(fig[1, 1]; title = "120 kVp X-ray Spectrum (Brain)", xlabel = "Energy (keV)", ylabel = "Relative Fluence")
    CM.lines!(ax, e_raw, w_raw ./ maximum(w_raw); color = :gray60, linewidth = 1.5, label = "Raw tube")
    CM.lines!(ax, e_raw, w_filt ./ maximum(w_filt); color = :steelblue, linewidth = 2, label = "After 7.0 mm Al")
    CM.lines!(ax, e_res, w_res ./ maximum(w_res); color = :darkorange, linewidth = 2, label = "Resolved")
    CM.vlines!(ax, [E_eff]; color = :red, linestyle = :dash, linewidth = 1, label = "E_eff=$(round(E_eff, digits=1)) keV")
    CM.xlims!(ax, 15, 125); CM.ylims!(ax, 0, 1.05)
    CM.axislegend(ax; position = :rt)
    CM.save(joinpath(RESULTS_DIR, "spectrum_120kVp.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00050001-0000-4000-8000-000000000002
md"""
## 3. Acquisition Protocol
"""

# ╔═╡ 00050002-0000-4000-8000-000000000002
begin
    sim_rotation_time = 1.0
    sim_collimation_mm = brain_collimation_mm  # 140 mm
    sim_n_views = 984
end

# ╔═╡ 00050003-0000-4000-8000-000000000002
begin
    # Match recon resolution to phantom input (1 mm isotropic)
    sim_recon_xy = 256
    sim_recon_fov_cm = brain_fov_cm
    sim_slice_thickness_mm = 1.0
    sim_recon_z_cm = brain_z_cm
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

# ╔═╡ 00050004-0000-4000-8000-000000000002
sim_protocol = BS.CTProtocol(
    kVp = 120, mA = 150.0, views = sim_n_views,
    rotation_time = sim_rotation_time,
    collimation_mm = sim_collimation_mm,
    additional_filters = additional_filters,
)

# ╔═╡ 00060001-0000-4000-8000-000000000002
md"""
## 4. BHC Calibration
"""

# ╔═╡ 00060002-0000-4000-8000-000000000002
custom_filter_control = (x = (0.0, 0.25, 0.5, 0.75, 1.0), y = (1.0, 0.85, 0.6, 0.15, 0.001))

# ╔═╡ 00060003-0000-4000-8000-000000000002
sim_noise_floor_hu = 28.0

# ╔═╡ 00060004-0000-4000-8000-000000000002
begin
    bhc_enabled = true
    sino_bhc_hu_low = 450.0; sino_bhc_hu_high = 600.0; sino_order = 2
    img_bhc_hu_low = 50.0; img_bhc_hu_high = 150.0; img_bhc_scale_factor = 0.2
end

# ╔═╡ 00060005-0000-4000-8000-000000000002
bhc_model, μ_water_ref = let
    prot = BS.CTProtocol(kVp = 120, additional_filters = additional_filters)
    e, w = BS.resolve_spectrum(sim_opts, prot; scanner = sim_scanner)
    ref_E = sum(e .* w) / sum(w)
    model = BS.calibrate_bhc_two_material(e, w; order = sino_order, reference_energy_keV = ref_E, hu_low = sino_bhc_hu_low, hu_high = sino_bhc_hu_high)
    @info "BHC 120 kVp: E_ref=$(round(ref_E, digits=1)) keV, μ_water=$(round(model.μ_water_ref, digits=5)) cm⁻¹"
    model, model.μ_water_ref
end

# ╔═╡ 00070001-0000-4000-8000-000000000002
md"""
## 5. Forward Projection & Reconstruction
"""

# ╔═╡ 00070002-0000-4000-8000-000000000002
# Shared reconstruction pipeline
function _recon_pipeline!(sino_data, geom, phantom_extent)
    recon_size = sim_matrix_size
    sino_gpu = to_gpu(sino_data)

    # Water-only polynomial BHC
    if bhc_enabled
        BS.apply_bhc!(sino_gpu, bhc_model.water_bhc)
    end

    # FDK
    ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, geom, recon_size;
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y))
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size)

    # Image-domain BHC
    if bhc_enabled
        BS.apply_bhc_image_domain(recon_μ, geom, recon_size, μ_water_ref;
            hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
            scale_factor = img_bhc_scale_factor, volume_extent = phantom_extent)
    end

    vol = Array(recon_μ)
    ws_fdk = nothing; sino_gpu = nothing; recon_μ = nothing; clear_gpu!()

    # HU + noise floor + FOV mask
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

# ╔═╡ 00070003-0000-4000-8000-000000000002
# Scan 1: Non-contrast
sim_noncontrast_hu = let
    @info "Scan 1: Non-contrast brain..."
    ws = BS.create_eict_workspace(sim_scanner, sim_protocol, sim_opts, sim_recon_geom, phantom_noncontrast)
    BS.simulate!(ws, phantom_noncontrast, sim_scanner, sim_protocol, sim_opts, sim_recon_geom)
    sino = Array(ws.sino_noisy_out)
    geom = ws.geom
    ws = nothing; clear_gpu!()
    _recon_pipeline!(sino, geom, phantom_noncontrast.extent)
end;

# ╔═╡ 00070004-0000-4000-8000-000000000002
# Scan 2: Contrast-enhanced (t = 15 s)
sim_contrast_hu = let
    @info "Scan 2: Contrast brain (t=$(contrast_time_s) s)..."
    ws = BS.create_eict_workspace(sim_scanner, sim_protocol, sim_opts, sim_recon_geom, phantom_contrast)
    BS.simulate!(ws, phantom_contrast, sim_scanner, sim_protocol, sim_opts, sim_recon_geom)
    sino = Array(ws.sino_noisy_out)
    geom = ws.geom
    ws = nothing; clear_gpu!()
    _recon_pipeline!(sino, geom, phantom_contrast.extent)
end;

# ╔═╡ 00080001-0000-4000-8000-000000000002
md"""
## 6. Visualization
"""

# ╔═╡ 096ba509-c17b-4d01-97b0-1ce75df58cc0
mid_z = size(sim_noncontrast_hu, 3) ÷ 2

# ╔═╡ e7d41653-97f9-4f14-b3dc-510492ea11cc
z_view = 50  # tweak this to view different slices

# ╔═╡ 00080002-0000-4000-8000-000000000002
# Non-contrast vs Contrast: soft tissue window
let
    nc_slice = sim_noncontrast_hu[:, :, z_view]
    ce_slice = sim_contrast_hu[:, :, z_view]

    fig = CM.Figure(size = (1100, 500), fontsize = 13)
    ax1 = CM.Axis(fig[1, 1]; title = "Non-contrast (slice $z_view)", aspect = CM.DataAspect())
    CM.heatmap!(ax1, nc_slice; colormap = :grays, colorrange = (-200, 500))
    ax2 = CM.Axis(fig[1, 2]; title = "Contrast t=$(contrast_time_s)s (slice $z_view)", aspect = CM.DataAspect())
    hm = CM.heatmap!(ax2, ce_slice; colormap = :grays, colorrange = (-200, 500))
    CM.Colorbar(fig[1, 3], hm; label = "HU", width = 12)
    CM.save(joinpath(RESULTS_DIR, "noncontrast_vs_contrast.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080004-0000-4000-8000-000000000002
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

# ╔═╡ 00080003-0000-4000-8000-000000000002
# Subtraction image: Contrast - Non-contrast (enhancing regions)
let
    diff = sim_contrast_hu[:, :, z_view] .- sim_noncontrast_hu[:, :, z_view]
    # Mask out-of-FOV regions
    diff[sim_noncontrast_hu[:, :, z_view] .< -1500] .= 0

    fig = CM.Figure(size = (600, 550), fontsize = 13)
    ax = CM.Axis(fig[1, 1]; title = "Subtraction: Contrast − Non-contrast (slice $z_view)", aspect = CM.DataAspect())
    hm = CM.heatmap!(ax, diff; colormap = :hot, colorrange = (100, 200))
    CM.Colorbar(fig[1, 2], hm; label = "ΔHU")
    CM.save(joinpath(RESULTS_DIR, "subtraction.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080005-0000-4000-8000-000000000002
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

# ╔═╡ 00080006-0000-4000-8000-000000000002
# Line profile through center
let
    mz = size(sim_noncontrast_hu, 3) ÷ 2 + 1
    cy = size(sim_noncontrast_hu, 2) ÷ 2
    x_cm = range(-sim_recon_fov_cm/2, sim_recon_fov_cm/2, length=size(sim_noncontrast_hu, 1))

    fig = CM.Figure(size = (900, 400), fontsize = 13)
    ax = CM.Axis(fig[1, 1]; title = "Horizontal Line Profile (mid-slice)", xlabel = "Position (cm)", ylabel = "HU")
    CM.lines!(ax, collect(x_cm), Float64.(sim_noncontrast_hu[:, cy, mz]); color = :steelblue, linewidth = 1.5, label = "Non-contrast")
    CM.lines!(ax, collect(x_cm), Float64.(sim_contrast_hu[:, cy, mz]); color = :darkorange, linewidth = 1.5, label = "Contrast (t=$(contrast_time_s)s)")
    CM.axislegend(ax; position = :rt)
    CM.save(joinpath(RESULTS_DIR, "line_profile.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ Cell order:
# ╠═00010001-0000-4000-8000-000000000002
# ╠═14530566-ecaa-4345-9115-e75981f3837e
# ╠═0b629a9c-722c-47c5-900e-5e5b311a743a
# ╠═00010003-0000-4000-8000-000000000002
# ╠═00010004-0000-4000-8000-000000000002
# ╠═00010005-0000-4000-8000-000000000002
# ╠═00010006-0000-4000-8000-000000000002
# ╠═00010007-0000-4000-8000-000000000002
# ╠═00010008-0000-4000-8000-000000000002
# ╠═00010009-0000-4000-8000-000000000002
# ╠═00010013-0000-4000-8000-000000000002
# ╠═00010012-0000-4000-8000-000000000002
# ╠═35b71532-be75-4fe9-ada8-9d6d4bf21f7e
# ╠═00010010-0000-4000-8000-000000000002
# ╠═00010011-0000-4000-8000-000000000002
# ╟─00020001-0000-4000-8000-000000000002
# ╟─00030001-0000-4000-8000-000000000002
# ╠═00030002-0000-4000-8000-000000000002
# ╠═00030003-0000-4000-8000-000000000002
# ╠═00030004-0000-4000-8000-000000000002
# ╠═00030005-0000-4000-8000-000000000002
# ╠═00030006-0000-4000-8000-000000000002
# ╟─00030007-0000-4000-8000-000000000002
# ╟─00040001-0000-4000-8000-000000000002
# ╠═00040002-0000-4000-8000-000000000002
# ╠═00040003-0000-4000-8000-000000000002
# ╠═00040004-0000-4000-8000-000000000002
# ╟─00040005-0000-4000-8000-000000000002
# ╟─00050001-0000-4000-8000-000000000002
# ╠═00050002-0000-4000-8000-000000000002
# ╠═00050003-0000-4000-8000-000000000002
# ╠═00050004-0000-4000-8000-000000000002
# ╟─00060001-0000-4000-8000-000000000002
# ╠═00060002-0000-4000-8000-000000000002
# ╠═00060003-0000-4000-8000-000000000002
# ╠═00060004-0000-4000-8000-000000000002
# ╠═00060005-0000-4000-8000-000000000002
# ╟─00070001-0000-4000-8000-000000000002
# ╠═00070002-0000-4000-8000-000000000002
# ╠═00070003-0000-4000-8000-000000000002
# ╠═00070004-0000-4000-8000-000000000002
# ╟─00080001-0000-4000-8000-000000000002
# ╠═096ba509-c17b-4d01-97b0-1ce75df58cc0
# ╠═e7d41653-97f9-4f14-b3dc-510492ea11cc
# ╟─00080002-0000-4000-8000-000000000002
# ╟─00080004-0000-4000-8000-000000000002
# ╟─00080003-0000-4000-8000-000000000002
# ╟─00080005-0000-4000-8000-000000000002
# ╟─00080006-0000-4000-8000-000000000002
