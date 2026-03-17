#!/usr/bin/env julia
#
# test_brain_fix_t20s.jl — Test fixed brain perfusion pipeline (single t=20s timepoint)
#
# Fixes applied:
#   1. NO sinogram-domain two-material BHC (use water polynomial only)
#   2. NO cupping correction
#   3. Keep: water BHC polynomial, image-domain BHC, custom filter, noise floor
#
# Usage:  julia --project=verification verification/scripts/test_brain_fix_t20s.jl
#

println("="^60)
println("  Brain Perfusion Fix Test — t=20s")
println("  Fixes: water-only BHC + image-domain BHC, no cupping")
println("="^60)

# ─── Setup ──────────────────────────────────────────────────────────────────

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Metal
using MAT
using Statistics: mean, std
using Printf
import BasisSimulator as BS
import XrayAttenuation as XA
import CairoMakie as CM

function to_gpu(x::AbstractArray)
    if @isdefined(Metal) && Metal.functional()
        return Metal.MtlArray(x)
    elseif @isdefined(CUDA) && CUDA.functional()
        return CUDA.CuArray(x)
    else
        return x
    end
end

function clear_gpu!()
    GC.gc(true)
    if @isdefined(CUDA) && CUDA.functional(); CUDA.reclaim(); end
    nothing
end

# ─── Paths ──────────────────────────────────────────────────────────────────

PHANTOM_DIR    = joinpath(dirname(@__DIR__), "data", "brain_perfusion")
P1_RAW_PATH    = joinpath(PHANTOM_DIR, "P1_brain_all_2020_RAW_400_400_400.raw")
P1_TABLE_PATH  = joinpath(PHANTOM_DIR, "P1_voxelize_table.txt")
P2_RAW_PATH    = joinpath(PHANTOM_DIR, "P2_brain_all_2020_RAW_400_400_400.raw")
P2_TABLE_PATH  = joinpath(PHANTOM_DIR, "P2_vozelize_table.txt")
STRUCT_INFO_PATH  = joinpath(PHANTOM_DIR, "structure_info.mat")
IODINE_DATA_PATH  = joinpath(PHANTOM_DIR, "iodine_mass_data.mat")
RESULTS_DIR       = joinpath(dirname(@__DIR__), "results", "brain_perfusion")
RAW_DIR           = joinpath(RESULTS_DIR, "raw")
DEBUG_DIR         = joinpath(RESULTS_DIR, "debug")
mkpath(RAW_DIR)
mkpath(DEBUG_DIR)

for p in [P1_RAW_PATH, P2_RAW_PATH, P1_TABLE_PATH, P2_TABLE_PATH, STRUCT_INFO_PATH, IODINE_DATA_PATH]
    @assert isfile(p) "Missing: $p"
end

# ─── Load phantom data ──────────────────────────────────────────────────────

println("\n▶ Loading phantom data...")

P1_raw_file = let
    buf = Array{UInt16}(undef, 400, 400, 400)
    open(P1_RAW_PATH, "r") do io; read!(io, buf); end
    arr = Int.(buf)
    arr = reverse(arr, dims=2)
    BS.relabel_zero_islands_2d!(arr; newlabel=10)
    arr
end

P2_raw_file = let
    buf = Array{UInt16}(undef, 400, 400, 400)
    open(P2_RAW_PATH, "r") do io; read!(io, buf); end
    arr = Int.(buf)
    reverse(arr, dims=2)
end

(P1_structure_map, P2_structure_map) = (
    BS.load_structure_map(P1_TABLE_PATH),
    BS.load_structure_map(P2_TABLE_PATH),
)

println("  ✓ P1/P2 loaded ($(size(P1_raw_file)))")

# ─── Materials ──────────────────────────────────────────────────────────────

MATERIAL_MAP_BASE = Dict{Int, Symbol}(
    0 => :air, 1 => :muscle, 2 => :air, 3 => :bone,
    5 => :soft_tissue, 10 => :soft_tissue, 13 => :bone,
    14 => :soft_tissue, 17 => :csf, 18 => :gray_matter,
    19 => :white_matter, 21 => :blood, 22 => :blood,
)

(material_map_init, material_list_init) = let
    unique_ids = sort(unique(vec(P1_raw_file)))
    material_map = Dict{Int, Symbol}(k => v for (k, v) in MATERIAL_MAP_BASE if k in unique_ids)
    material_list = Dict{Symbol, XA.Material}(
        sym => BS.get_material(sym) for sym in unique(collect(values(material_map)))
    )
    (material_map, material_list)
end

# ─── Load MAT data ──────────────────────────────────────────────────────────

println("▶ Loading MAT files...")

(artery_info, vein_info, gm_info, wm_info) = let
    si = matread(STRUCT_INFO_PATH)
    function parse_info(d)
        names = vec(String.(d["name"]))
        vols = vec(Float64.(d["volume"]))
        Dict{String, Any}("name" => names, "volume" => vols)
    end
    (parse_info(si["artery_info"]), parse_info(si["vein_info"]),
     parse_info(si["gm_info"]),     parse_info(si["wm_info"]))
end

(iodine_artery, iodine_vein, iodine_gm, iodine_wm) = let
    d = matread(IODINE_DATA_PATH)
    (Float64.(d["mass_arteries"]), Float64.(d["mass_vein"]),
     Float64.(d["mass_gm"]),       Float64.(d["mass_wm"]))
end

println("  ✓ MAT files loaded")

# ─── Scanner & protocol ────────────────────────────────────────────────────

additional_filters = [("Al", 4.5)]

brain_extent_mm = 400 * 0.1 * 10.0
brain_eict_mag  = 1100.0 / 625.6
brain_det_cols  = ceil(Int, brain_extent_mm * brain_eict_mag / 1.0)
brain_det_rows  = 256
brain_recon_fov = 40.0
brain_vox_cm    = 0.1

brain_max_collimation_mm = brain_det_rows * 0.625
brain_max_z_slices = floor(Int, brain_max_collimation_mm / (brain_vox_cm * 10.0))

_z_any   = vec(any(P1_raw_file .!= 0, dims=(1,2)))
_z_first = findfirst(_z_any)
_z_last  = findlast(_z_any)
_z_tissue = max(1, _z_first - 4) : min(size(P1_raw_file, 3), _z_last + 4)

if length(_z_tissue) > brain_max_z_slices
    _z_mid  = (_z_tissue[1] + _z_tissue[end]) ÷ 2
    _z_half = brain_max_z_slices ÷ 2
    BRAIN_Z_CROP = max(1, _z_mid - _z_half + 1) : min(size(P1_raw_file, 3), _z_mid + _z_half)
else
    BRAIN_Z_CROP = _z_tissue
end

brain_z_cm     = length(BRAIN_Z_CROP) * brain_vox_cm
brain_recon_xy = 512
brain_n_slices = round(Int, brain_z_cm / brain_vox_cm)

println("  Z-crop: $(BRAIN_Z_CROP) → $(brain_n_slices) slices, z=$(brain_z_cm) cm")

scanner_brain = BS.Scanner(
    source_to_isocenter = 625.6, source_to_detector = 1100.0,
    detector_rows = brain_det_rows, detector_cols = brain_det_cols,
    detector_row_size = 0.625, detector_col_size = 1.0,
    detector_shape = BS.CURVED_DETECTOR,
    focal_spot_width = 1.0, focal_spot_length = 1.0,
    target_angle = 7.0,
    flat_filter_material = :aluminum, flat_filter_thickness = 2.5,
    bowtie_filter = :ge_revolution_large,
    detector_material = :lumex, detector_depth = 3.0,
    fill_factor_row = 0.9, fill_factor_col = 0.9,
    electronic_noise = 0, detection_gain = 10.0,
)

protocol_brain = BS.CTProtocol(
    kVp = 120, mA = 300.0, views = 984, rotation_time = 0.5,
    collimation_mm = brain_z_cm * 10.0,
    additional_filters = additional_filters,
)

sim_opts_brain = BS.SimOptions(fidelity = :high, seed = 1234)

recon_opts_brain = BS.ReconOptions(
    algorithm = :fdk,
    matrix_size = (brain_recon_xy, brain_recon_xy, brain_n_slices),
    fov_cm = brain_recon_fov, z_cm = brain_z_cm,
)

# ─── BHC calibration ───────────────────────────────────────────────────────

custom_filter_control = (
    x = (0.0, 0.25, 0.5, 0.75, 1.0),
    y = (1.0, 0.85, 0.60, 0.15, 0.001),
)

sim_noise_floor_hu = 28.0
img_bhc_hu_low = 50.0
img_bhc_hu_high = 150.0
img_bhc_scale_factor = 0.2

# Calibrate BHC (we still use calibrate_bhc_two_material to get μ_water_ref,
# but only use the .water_bhc polynomial for sinogram correction)
bhc_model, μ_water_brain = let
    e, w = BS.resolve_spectrum(sim_opts_brain, protocol_brain; scanner = scanner_brain)
    ref_E = sum(e .* w) / sum(w)
    model = BS.calibrate_bhc_two_material(e, w;
        order = 2, reference_energy_keV = ref_E,
        hu_low = 450.0, hu_high = 600.0)
    @info "BHC 120 kVp: E_ref=$(round(ref_E, digits=1)) keV, μ_water=$(round(model.μ_water_ref, digits=5)) cm⁻¹"
    model, model.μ_water_ref
end

# ─── Build phantom for t=20s ───────────────────────────────────────────────

println("\n▶ Building phantom for t=20s...")

t_s = 20
t_contrast = t_s * 1000 + 1  # = 20001

recon_size = (brain_recon_xy, brain_recon_xy, brain_n_slices)
P1_stamped = copy(P1_raw_file)[:, :, BRAIN_Z_CROP]
P2_raw_crop = P2_raw_file[:, :, BRAIN_Z_CROP]

segment_index_map = Dict{Int, Vector{CartesianIndex{3}}}()
for (id, name) in P2_structure_map
    first(name) in ('2', '3', '4', '5') || continue
    idxs = findall(==(id), P2_raw_crop)
    isempty(idxs) || (segment_index_map[id] = idxs)
end

for (id, idxs) in segment_index_map
    P1_stamped[idxs] .= id
end

mask_gpu = to_gpu(UInt16.(P1_stamped))

function build_phantom(t_idx)
    materials_dict = Dict{Int, XA.Material}(
        id => material_list_init[sym] for (id, sym) in material_map_init
    )
    for (prefix, info, iodine, base_sym) in [
        ("5", artery_info, iodine_artery, :blood),
        ("4", vein_info,   iodine_vein,   :blood),
        ("3", wm_info,     iodine_wm,     :white_matter),
        ("2", gm_info,     iodine_gm,     :gray_matter),
    ]
        seg_mats = BS.update_structures!(
            P1_stamped, P2_structure_map, prefix, P2_raw_crop,
            base_sym, material_list_init[base_sym], info, iodine, t_idx;
            index_map = segment_index_map)
        merge!(materials_dict, seg_mats)
    end
    _prefix_to_base = Dict("5" => :blood, "4" => :blood, "3" => :white_matter, "2" => :gray_matter)
    for (id, name) in P2_structure_map
        id ∈ keys(materials_dict) && continue
        id ∈ keys(segment_index_map) || continue
        c = string(first(name))
        c ∈ keys(_prefix_to_base) || continue
        materials_dict[id] = material_list_init[_prefix_to_base[c]]
    end
    let (nx, ny, nz) = size(P1_stamped)
        dx, dy, dz = 0.1, 0.1, 0.1
        BS.Phantom(mask_gpu, BS.build_materials_vector(materials_dict),
            (dx, dy, dz),
            (-dx*nx/2 + dx/2, -dy*ny/2 + dy/2, -dz*nz/2 + dz/2),
            (dx*nx, dy*ny, dz*nz))
    end
end

phantom_t = build_phantom(t_contrast)
println("  ✓ Phantom built: $(size(P1_stamped)), extent=$(phantom_t.extent)")

# ─── Create workspace & warmup ─────────────────────────────────────────────

println("\n▶ Creating workspace & warmup...")

ws = BS.create_eict_workspace(
    scanner_brain, protocol_brain,
    sim_opts_brain, recon_opts_brain, phantom_t)
geom = ws.geom

BS.simulate!(ws, phantom_t, scanner_brain, protocol_brain, sim_opts_brain, recon_opts_brain)
println("  ✓ Warmup complete")

# ─── Simulate t=20s with FIXED pipeline ────────────────────────────────────

println("\n▶ Simulating t=$(t_s)s (FIXED pipeline)...")
println("  FIX 1: Water-only polynomial BHC (no two-material sinogram BHC)")
println("  FIX 2: No cupping correction")
println("  KEEP:  Image-domain BHC, custom filter, noise floor")

fdk_hu_result = nothing

@time begin
    # Forward projection (full physics)
    BS.simulate!(ws, phantom_t, scanner_brain, protocol_brain, sim_opts_brain, recon_opts_brain)

    sino_gpu = to_gpu(ws.sino_noisy_out)

    # ══════════════════════════════════════════════════════════════════
    # FIX 1: Water-only polynomial BHC (replaces apply_bhc_two_material)
    # ══════════════════════════════════════════════════════════════════
    BS.apply_bhc!(sino_gpu, bhc_model.water_bhc)

    # FDK reconstruction with custom filter
    ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, geom, recon_size;
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y))
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size)

    # Image-domain BHC (residual bone correction — robust to cone-beam)
    BS.apply_bhc_image_domain(recon_μ, geom, recon_size, μ_water_brain;
        hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
        scale_factor = img_bhc_scale_factor,
        volume_extent = phantom_t.extent)

    vol = Array(recon_μ)

    # HU conversion
    recon_hu = Float32.(BS.to_hounsfield(vol; μ_water = μ_water_brain))

    # Noise floor (dose-independent system noise — keep this)
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)

    # ══════════════════════════════════════════════════════════════════
    # FIX 2: NO cupping correction (not appropriate for brain with BHC)
    # ══════════════════════════════════════════════════════════════════
    # BS.apply_radial_cupping_correction!(recon_hu; fov_cm = brain_recon_fov)  # REMOVED

    global fdk_hu_result = reverse(recon_hu, dims=3)

    ws_fdk = nothing; sino_gpu = nothing; recon_μ = nothing; vol = nothing
end

println("  ✓ Simulation complete")

# ─── Save raw output ────────────────────────────────────────────────────────

println("\n▶ Saving output...")

vol_out = fdk_hu_result
nx, ny, nz = size(vol_out)
vol_ij = vol_out[:, end:-1:1, :]  # flip Y for ImageJ

fname = "brain_fdk_FIXED_t$(t_s)s_$(nx)x$(ny)x$(nz).raw"
open(joinpath(RAW_DIR, fname), "w") do io
    write(io, vec(vol_ij))
end
println("  ✓ RAW: $fname")

# ─── Quick diagnostics ──────────────────────────────────────────────────────

println("\n", "="^60)
println("  DIAGNOSTIC COMPARISON")
println("="^60)

v = vec(vol_out)
@printf("  Min        : %12.2f HU\n", minimum(v))
@printf("  Max        : %12.2f HU\n", maximum(v))
@printf("  Mean       : %12.2f HU\n", mean(v))
@printf("  Median     : %12.2f HU\n", median(v))

# Center brain ROI
roi = vol_out[200:300, 200:300, nz÷2]
@printf("\n  Center brain ROI (200:300 × 200:300, mid-z):\n")
@printf("    Mean     : %8.2f HU  (expected: 30-45 GM, 20-30 WM)\n", mean(roi))
@printf("    Std      : %8.2f HU\n", std(roi))
@printf("    [min,max]: [%.1f, %.1f]\n", minimum(roi), maximum(roi))

# Air region check
air_roi = vol_out[10:50, 10:50, nz÷2]
@printf("\n  Air ROI (10:50 × 10:50, mid-z):\n")
@printf("    Mean     : %8.2f HU  (expected: ~-1000)\n", mean(air_roi))

# Extreme voxel count
@printf("\n  Extreme voxels:\n")
@printf("    > 1000 HU  : %d  (%.2f%%)\n", count(x -> x > 1000, v), 100 * count(x -> x > 1000, v) / length(v))
@printf("    > 10000 HU : %d\n", count(x -> x > 10000, v))
@printf("    < -1100 HU : %d\n", count(x -> x < -1100, v))

# ─── Save comparison PNGs ───────────────────────────────────────────────────

println("\n▶ Saving comparison PNGs...")

function save_png(path, slice; W=80, L=40, title="")
    lo, hi = L - W/2, L + W/2
    img = clamp.((slice .- lo) ./ (hi - lo), 0f0, 1f0)
    f = CM.Figure(size=(600, 600))
    ax = CM.Axis(f[1, 1]; title, aspect=CM.DataAspect(), yreversed=true)
    CM.heatmap!(ax, img'; colormap=:grays, colorrange=(0, 1))
    CM.hidedecorations!(ax)
    CM.save(path, f, px_per_unit=2)
    println("  Saved: $(basename(path))")
end

mz = nz ÷ 2 + 1
sl = vol_out[:, :, mz]

save_png(joinpath(DEBUG_DIR, "FIXED_mid_brain_window.png"), sl;
    W=80, L=40, title="FIXED t=$(t_s)s — Brain (W=80 L=40)")
save_png(joinpath(DEBUG_DIR, "FIXED_mid_soft_tissue.png"), sl;
    W=400, L=40, title="FIXED t=$(t_s)s — Soft tissue (W=400 L=40)")
save_png(joinpath(DEBUG_DIR, "FIXED_mid_wide_window.png"), sl;
    W=3000, L=0, title="FIXED t=$(t_s)s — Wide (W=3000 L=0)")

# Top/bottom slices (cone-beam artifact check)
save_png(joinpath(DEBUG_DIR, "FIXED_slice10_wide.png"), vol_out[:, :, 10];
    W=3000, L=0, title="FIXED Slice 10 (near top, W=3000)")
save_png(joinpath(DEBUG_DIR, "FIXED_slice150_wide.png"), vol_out[:, :, min(150, nz)];
    W=3000, L=0, title="FIXED Slice 150 (near bottom, W=3000)")

# Load CatSim reference for comparison
ref_path = joinpath(dirname(@__DIR__), "data", "brain_perfusion", "reference", "t23001",
    "contrast_brain_phantom_t23001_512x512x160.raw")
if isfile(ref_path) && nz == 160
    vol_ref = Array{Float32}(undef, 512, 512, 160)
    open(ref_path, "r") do io; read!(io, vol_ref); end
    ref_roi = vol_ref[200:300, 200:300, 80]
    @printf("\n  CatSim reference center ROI: mean=%.2f HU\n", mean(ref_roi))

    save_png(joinpath(DEBUG_DIR, "REF_mid_brain_window.png"), vol_ref[:, :, 81];
        W=80, L=40, title="CatSim ref — Brain (W=80 L=40)")
end

# ─── Load broken output for side-by-side ────────────────────────────────────

broken_path = joinpath(RAW_DIR, "brain_fdk_t0s_512x512x$(nz).raw")
if isfile(broken_path)
    vol_broken = Array{Float32}(undef, 512, 512, nz)
    open(broken_path, "r") do io; read!(io, vol_broken); end
    broken_roi = vol_broken[200:300, 200:300, nz÷2]
    @printf("\n  Broken (old) center ROI: mean=%.2f HU\n", mean(broken_roi))
end

println("\n", "="^60)
println("  Output saved to: $RAW_DIR")
println("  PNGs saved to:   $DEBUG_DIR")
println("  Done.")
println("="^60)
