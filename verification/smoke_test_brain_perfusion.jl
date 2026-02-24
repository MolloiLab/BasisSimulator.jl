"""
End-to-end smoke test for notebook 07_brain_perfusion.jl

Tests the full pipeline without Pluto/PlutoUI:
  - Data loading (P1/P2 XCAT raw files, iodine MAT files)
  - update_structures! iodine contrast
  - create_eict_workspace + simulate! (GPU if Metal available, else CPU)
  - FDK + HIR reconstruction
  - ImageJ RAW export

Uses only 2 time points (t=0, t=10 s) and z-crops to a thin slab for speed.
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using BasisSimulator
import BasisSimulator as BS
import MAT: matread
import Statistics: mean
import XrayAttenuation as XA

# Load GPU backend(s) upfront on the main thread.
# Metal.__init__ and CUDA.__init__ make driver calls that require the main thread.
# gpu_array_type() only inspects Base.loaded_modules — it never calls Base.require.
try; using Metal; catch; end   # no-op on non-Apple or when Metal unavailable
try; using CUDA;  catch; end   # no-op when CUDA unavailable
const GPU_ARRAY_TYPE = BS.gpu_array_type()
const USE_GPU = GPU_ARRAY_TYPE !== Array
println("GPU backend: ", USE_GPU ? string(GPU_ARRAY_TYPE) : "CPU")

# ── Paths ────────────────────────────────────────────────────────────────────
const PHANTOM_DIR   = joinpath(@__DIR__, "data", "brain_perfusion")
const P1_RAW_PATH   = joinpath(PHANTOM_DIR, "P1_brain_all_2020_RAW_400_400_400.raw")
const P1_TABLE_PATH = joinpath(PHANTOM_DIR, "P1_voxelize_table.txt")
const P2_RAW_PATH   = joinpath(PHANTOM_DIR, "P2_brain_all_2020_RAW_400_400_400.raw")
const P2_TABLE_PATH = joinpath(PHANTOM_DIR, "P2_vozelize_table.txt")
const STRUCT_INFO_PATH = joinpath(PHANTOM_DIR, "structure_info.mat")
const IODINE_DATA_PATH = joinpath(PHANTOM_DIR, "iodine_mass_data.mat")
const FIGURES_DIR   = joinpath(@__DIR__, "figures", "brain_perfusion")
const RAW_DIR       = joinpath(FIGURES_DIR, "raw")
mkpath(RAW_DIR)

for p in (P1_RAW_PATH, P1_TABLE_PATH, P2_RAW_PATH, P2_TABLE_PATH, STRUCT_INFO_PATH, IODINE_DATA_PATH)
    @assert isfile(p) "Missing: $p"
end
println("✓ All input paths verified")

# ── Load P1 / P2 ─────────────────────────────────────────────────────────────
println("\nLoading P1 raw file…")
P1_raw_file = let
    buf = Array{UInt16}(undef, 400, 400, 400)
    open(P1_RAW_PATH, "r") do io; read!(io, buf); end
    arr = Int.(buf)
    arr = reverse(arr, dims=2)
    BS.relabel_zero_islands_2d!(arr; newlabel=10)
    arr
end
println("  P1 size: ", size(P1_raw_file), "  unique labels: ", length(unique(vec(P1_raw_file))))

println("Loading P2 raw file…")
P2_raw_file = let
    buf = Array{UInt16}(undef, 400, 400, 400)
    open(P2_RAW_PATH, "r") do io; read!(io, buf); end
    arr = Int.(buf)
    reverse(arr, dims=2)
end
println("  P2 size: ", size(P2_raw_file))

(P1_structure_map, P2_structure_map) = (BS.load_structure_map(P1_TABLE_PATH), BS.load_structure_map(P2_TABLE_PATH))
println("  P1 structure map: ", length(P1_structure_map), " entries")
println("  P2 structure map: ", length(P2_structure_map), " entries")

# ── Materials ────────────────────────────────────────────────────────────────
const MATERIAL_MAP_BASE = Dict{Int, Symbol}(
    0  => :air, 1 => :muscle, 2 => :air, 3 => :bone,
    5  => :soft_tissue, 10 => :soft_tissue, 13 => :bone,
    14 => :soft_tissue, 17 => :csf, 18 => :gray_matter,
    19 => :white_matter, 21 => :blood, 22 => :blood,
)
(material_map_init, material_list_init) = let
    unique_ids   = sort(unique(vec(P1_raw_file)))
    material_map = Dict{Int, Symbol}(k => v for (k, v) in MATERIAL_MAP_BASE if k in unique_ids)
    material_list = Dict{Symbol, XA.Material}(
        sym => BS.get_material(sym) for sym in unique(collect(values(material_map)))
    )
    (material_map, material_list)
end
println("✓ Base materials: ", sort(collect(keys(material_list_init)), by=string))

# ── Iodine contrast data ─────────────────────────────────────────────────────
println("\nLoading iodine contrast data…")
(artery_info, vein_info, gm_info, wm_info) = let
    si = matread(STRUCT_INFO_PATH)
    function parse_info(d)
        names = vec(String.(d["name"]))
        vols  = vec(Float64.(d["volume"]))
        Dict{String, Any}("name" => names, "volume" => vols)
    end
    (parse_info(si["artery_info"]), parse_info(si["vein_info"]),
     parse_info(si["gm_info"]),    parse_info(si["wm_info"]))
end
(iodine_artery, iodine_vein, iodine_gm, iodine_wm) = let
    d = matread(IODINE_DATA_PATH)
    (Float64.(d["mass_arteries"]), Float64.(d["mass_vein"]),
     Float64.(d["mass_gm"]),       Float64.(d["mass_wm"]))
end
println("  Artery segments: ", size(iodine_artery, 1),
        "  time points: ", size(iodine_artery, 2))

# ── Scanner / protocol (identical to notebook) ───────────────────────────────
brain_eict_mag = 1100.0 / 625.6
brain_det_cols = ceil(Int, 400 * 0.1 * 10.0 * brain_eict_mag / 1.0)
brain_det_rows = 4
brain_recon_fov = 40.0
brain_vox_cm    = 0.1

# Thin z-crop for speed: 10 slices around the centre
_z_any     = vec(any(P1_raw_file .!= 0, dims=(1,2)))
_z_mid     = div(findfirst(_z_any) + findlast(_z_any), 2)
BRAIN_Z_CROP = max(1, _z_mid - 1) : min(400, _z_mid + 2)   # 4 slices
brain_z_cm   = length(BRAIN_Z_CROP) * brain_vox_cm
brain_recon_xy = 64
brain_n_slices = length(BRAIN_Z_CROP)

println("\nZ-crop: ", BRAIN_Z_CROP, "  (", brain_n_slices, " slices, ", brain_z_cm, " cm)")

scanner_brain = BS.Scanner(
    source_to_isocenter   = 625.6,
    source_to_detector    = 1100.0,
    detector_rows         = brain_det_rows,
    detector_cols         = brain_det_cols,
    detector_row_size     = 0.625,
    detector_col_size     = 1.0,
    detector_shape        = BS.CURVED_DETECTOR,
    focal_spot_width      = 1.0,
    focal_spot_length     = 1.0,
    target_angle          = 7.0,
    flat_filter_material  = :aluminum,
    flat_filter_thickness = 2.5,
    detector_material     = :gos,
    detector_depth        = 3.0,
    fill_factor_row       = 0.9,
    fill_factor_col       = 0.9,
    detection_gain        = 1.0,
)
protocol_brain  = BS.CTProtocol(kVp=120.0, mA=300.0, views=32, rotation_time=0.5)
sim_opts_brain  = BS.SimOptions(fidelity=:low, seed=42)
recon_opts_brain = BS.ReconOptions(
    algorithm   = :fdk,
    matrix_size = (brain_recon_xy, brain_recon_xy, brain_n_slices),
    fov_cm      = brain_recon_fov,
    z_cm        = brain_z_cm,
)

# ── μ_water calibration ───────────────────────────────────────────────────────
μ_water_brain = let
    energies_raw, weights_raw = BS.load_spectrum(120)
    energies, weights = BS.downsample_spectrum(energies_raw, weights_raw, 30)
    filter_t_cm  = scanner_brain.flat_filter_thickness / 10.0
    filter_trans = [exp(-BS.get_bowtie_mu("Al", Float64(e)) * filter_t_cm) for e in energies]
    filtered_weights = weights .* filter_trans
    E_eff = sum(energies .* filtered_weights) / sum(filtered_weights)
    BS.compute_μ_at_energy(XA.Materials.water, E_eff)
end
println("μ_water (120 kVp, NIST): ", round(μ_water_brain, sigdigits=4), " cm⁻¹")

# ── Pre-compute time points ───────────────────────────────────────────────────
# Use only 2 time points for the smoke test (fast)
const CONTRAST_TIME_S   = [0, 10]
const CONTRAST_INDICES  = CONTRAST_TIME_S .* 1000 .+ 1

all_fdk_hu = Dict{Int, Array{Float32, 3}}()
all_hir_hu = Dict{Int, Array{Float32, 3}}()

println("\nBuilding segment index map…")
P1_stamped  = copy(P1_raw_file)[:, :, BRAIN_Z_CROP]
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
println("  Segment map: ", length(segment_index_map), " entries")

function build_phantom(t_contrast)
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
            base_sym, material_list_init[base_sym], info, iodine, t_contrast;
            index_map = segment_index_map
        )
        merge!(materials_dict, seg_mats)
    end
    # Some P2 segments are stamped into the mask but skipped by update_structures! when
    # their row index is out of bounds in the iodine table (e.g. segments 5398, 5399).
    # Fill them with their base material (blood/gm/wm) so Phantom's materials vector
    # covers all mask labels without gaps.
    prefix_to_base = Dict("5" => :blood, "4" => :blood, "3" => :white_matter, "2" => :gray_matter)
    for (id, name) in P2_structure_map
        id ∈ keys(materials_dict) && continue
        id ∈ keys(segment_index_map) || continue
        c = string(first(name))
        c ∈ keys(prefix_to_base) || continue
        materials_dict[id] = material_list_init[prefix_to_base[c]]
    end
    mask = GPU_ARRAY_TYPE(UInt16.(P1_stamped))
    BS.Phantom(mask, materials_dict, (0.1, 0.1, 0.1))
end

println("\nBuilding t=0 phantom + workspace…")
phantom_t_0 = build_phantom(CONTRAST_INDICES[1])
@time ws = BS.create_eict_workspace(
    scanner_brain, protocol_brain,
    sim_opts_brain, recon_opts_brain, phantom_t_0
)

geom      = ws.geom
recon_size = (brain_recon_xy, brain_recon_xy, brain_n_slices)

println("JIT warmup simulate!…")
@time BS.simulate!(ws, phantom_t_0, scanner_brain, protocol_brain, sim_opts_brain, recon_opts_brain)

sino_gpu = ws.sinogram
ws_fdk   = BS.create_fdk_recon_workspace(sino_gpu, geom, recon_size)
ws_hir   = BS.create_hir_recon_workspace(sino_gpu, geom, recon_size; strength=2)

println("\nRunning $(length(CONTRAST_INDICES)) time points…")
for (i, t_contrast) in enumerate(CONTRAST_INDICES)
    println("▶ t=$(CONTRAST_TIME_S[i]) s")
    phantom_t = i == 1 ? phantom_t_0 : build_phantom(t_contrast)

    @time begin
        BS.simulate!(ws, phantom_t, scanner_brain, protocol_brain, sim_opts_brain, recon_opts_brain)
        sino = ws.sinogram

        all_fdk_hu[i] = reverse(BS.to_hounsfield(
            Array(BS.reconstruct!(ws_fdk, sino, geom, recon_size));
            μ_water = μ_water_brain
        ), dims=3)

        all_hir_hu[i] = reverse(BS.to_hounsfield(
            Array(BS.reconstruct!(ws_hir, sino, geom, recon_size));
            μ_water = μ_water_brain
        ), dims=3)
    end
    println("   ✓ done")
end
ws_fdk = nothing; ws_hir = nothing; ws = nothing
GC.gc(true)

# ── Validation + RAW export ────────────────────────────────────────────────────
let all_ok = true
    println("\n=== Validation ===")
    for (i, t_s) in enumerate(CONTRAST_TIME_S)
        fdk = all_fdk_hu[i]; hir = all_hir_hu[i]
        fdk_ok = all(isfinite, fdk) && minimum(fdk) > -5000 && maximum(fdk) < 20000
        hir_ok = all(isfinite, hir) && minimum(hir) > -5000 && maximum(hir) < 20000
        check_fdk = fdk_ok ? "\u2713" : "\u2717"; println("  t=$(t_s)s  FDK: min=$(round(minimum(fdk),digits=1)) max=$(round(maximum(fdk),digits=1))  $check_fdk")
        check_hir = hir_ok ? "\u2713" : "\u2717"; println("  t=$(t_s)s  HIR: min=$(round(minimum(hir),digits=1)) max=$(round(maximum(hir),digits=1))  $check_hir")
        all_ok &= fdk_ok & hir_ok
    end

    # Check iodine contrast: t=10s mean HU should exceed t=0s
    mean_hu_t0  = mean(all_fdk_hu[1])
    mean_hu_t10 = mean(all_fdk_hu[2])
    contrast_ok = mean_hu_t10 > mean_hu_t0
    check_contrast = contrast_ok ? "\u2713" : "\u2717"; println("  Iodine contrast: mean HU t=0s=$(round(mean_hu_t0,digits=2))  t=10s=$(round(mean_hu_t10,digits=2))  $check_contrast")
    all_ok &= contrast_ok

    # RAW export
    println("\nExporting ImageJ RAW files…")
    for (i, t_s) in enumerate(CONTRAST_TIME_S)
        for (name, vols) in [("fdk", all_fdk_hu), ("hir", all_hir_hu)]
            vol  = vols[i]
            nx, ny, nz = size(vol)
            # No permutation: Julia column-major matches ImageJ raw import (open as width=nx height=ny nSlices=nz)
            fname  = "brain_$(name)_t$(t_s)s_$(nx)x$(ny)x$(nz).raw"
            fpath  = joinpath(RAW_DIR, fname)
            open(fpath, "w") do io; write(io, vec(vol)); end
            raw_ok = isfile(fpath) && filesize(fpath) == nx * ny * nz * 4
            check_raw = raw_ok ? "\u2713" : "\u2717"; println("  $fname  ($(filesize(fpath)) bytes)  $check_raw")
            all_ok &= raw_ok
        end
    end
    println("RAW files saved to: $RAW_DIR")
    println()
    if all_ok
        println("✓ ALL CHECKS PASSED — notebook 07 pipeline is functional")
    else
        println("✗ SOME CHECKS FAILED — see above")
        exit(1)
    end
end