#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════════════════════
# Ghost Artifact Diagnostic Script
#
# Usage: julia --project=verification ralph_loop_diagnose/diag.jl
#
# Configurable via the CONFIG section below. Loads XCAT phantom, runs
# simulate!() + reconstruct!(), saves reconstruction and sinogram PNGs.
# ═══════════════════════════════════════════════════════════════════════════════

using Printf

# ─── CONFIG (modify these for each test) ─────────────────────────────────────
PHYSICS_ENABLED     = true     # false = bare Siddon + FDK only (overrides all USE_* flags)

# Individual physics toggles (only used when PHYSICS_ENABLED = true)
USE_FILL_FACTOR        = true
USE_FLAT_FILTER        = true
USE_BOWTIE_FILTER      = true
USE_DETECTOR_EFFICIENCY = true
USE_CROSSTALK          = true
USE_OPTICAL_CROSSTALK  = true
USE_FOCAL_SPOT         = true
USE_LAG                = true
USE_HEEL_EFFECT        = true
USE_BHC                = true

# Detector geometry
DETECTOR_COL_SIZE  = 1.0     # mm
DETECTOR_ROW_SIZE  = 0.625   # mm
DETECTOR_COLS      = nothing  # nothing = auto-compute from phantom extent
DETECTOR_ROWS      = 64

# Scan parameters
VIEWS              = 1600
KVP                = 120.0
MA                 = 700.0

# Reconstruction
RECON_XY           = 512
RECON_SLICES       = 64
RECON_FOV_CM       = 35.0

# Phantom
DOWNSAMPLE_FACTOR  = 2

# Output
TEST_NAME          = "all_physics"
OUTPUT_DIR         = joinpath(@__DIR__, "outputs")

# HU window for display
HU_WINDOW          = (-300, 400)

# ─── END CONFIG ──────────────────────────────────────────────────────────────

mkpath(OUTPUT_DIR)

println("=" ^ 70)
println("GHOST ARTIFACT DIAGNOSTIC — TEST: $TEST_NAME")
println("=" ^ 70)
println("Physics enabled: $PHYSICS_ENABLED")
if PHYSICS_ENABLED
    println("  fill_factor=$USE_FILL_FACTOR, flat_filter=$USE_FLAT_FILTER")
    println("  bowtie=$USE_BOWTIE_FILTER, det_eff=$USE_DETECTOR_EFFICIENCY")
    println("  crosstalk=$USE_CROSSTALK, optical=$USE_OPTICAL_CROSSTALK")
    println("  focal_spot=$USE_FOCAL_SPOT, lag=$USE_LAG")
    println("  heel=$USE_HEEL_EFFECT, bhc=$USE_BHC")
end
println("Detector: $(DETECTOR_COL_SIZE)mm cols, $(DETECTOR_ROW_SIZE)mm rows")
println("Views: $VIEWS, kVp: $KVP, mA: $MA")
println("Recon: $(RECON_XY)×$(RECON_XY)×$(RECON_SLICES) at $(RECON_FOV_CM)cm FOV")
println()

# ─── SETUP ────────────────────────────────────────────────────────────────────
println("Loading packages...")
t0 = time()

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "verification"))

using Revise
import BasisSimulator as BS
import CairoMakie as CM
import Metal
import XLSX
import XrayAttenuation as XA
using Unitful: @u_str, ustrip
using Statistics: mean, std

println("  Packages loaded in $(@sprintf("%.1f", time() - t0))s")

# ─── LOAD XCAT PHANTOM ───────────────────────────────────────────────────────
println("Loading XCAT phantom...")
t1 = time()

ROOT_DIR = joinpath(@__DIR__, "..", "verification")
PHANTOM_PATH = joinpath(ROOT_DIR, "data/xcat/vmale_50_1600x1400x500_8bit_little_endian_act_1.bin")
MATERIAL_XLSX_PATH = joinpath(ROOT_DIR, "data/xcat/Material_Spreadsheets/vmale_50_materials_heart_high_contrast.xlsx")

function load_phantom_bin(filepath; cols=1600, rows=1400, slices=500, dtype=UInt8)
    expected_size = cols * rows * slices * sizeof(dtype)
    actual_size = filesize(filepath)
    @assert actual_size == expected_size "File size mismatch: expected $expected_size, got $actual_size"
    data = Vector{dtype}(undef, cols * rows * slices)
    open(filepath, "r") do io
        read!(io, data)
    end
    phantom = reshape(data, (cols, rows, slices))
    return reverse(phantom, dims=(2, 3))
end

function downsample_phantom(phantom::AbstractArray{T, 3}, factor::Int) where T
    factor == 1 && return phantom
    new_size = size(phantom) .÷ factor
    result = similar(phantom, new_size)
    for k in 1:new_size[3], j in 1:new_size[2], i in 1:new_size[1]
        oi = (i - 1) * factor + factor ÷ 2 + 1
        oj = (j - 1) * factor + factor ÷ 2 + 1
        ok = (k - 1) * factor + factor ÷ 2 + 1
        result[i, j, k] = phantom[oi, oj, ok]
    end
    return result
end

phantom_raw = load_phantom_bin(PHANTOM_PATH)
phantom_labeled = downsample_phantom(phantom_raw, DOWNSAMPLE_FACTOR)
phantom_raw = nothing  # free memory

println("  Phantom size: $(size(phantom_labeled))")
println("  Unique labels: $(length(unique(phantom_labeled)))")

# ─── LOAD MATERIALS ──────────────────────────────────────────────────────────
function compute_ZA_ratio(composition::Dict{Int, Float64})
    atomic_masses = Dict(
        1=>1.008, 6=>12.011, 7=>14.007, 8=>15.999, 11=>22.990, 12=>24.305,
        15=>30.974, 16=>32.06, 17=>35.45, 19=>39.098, 20=>40.078, 26=>55.845, 53=>126.904
    )
    Z_sum = 0.0; A_sum = 0.0
    for (Z, mass_frac) in composition
        A = get(atomic_masses, Z, Float64(Z)*2)
        Z_sum += mass_frac * Z / A
        A_sum += mass_frac
    end
    return Z_sum / A_sum
end

function compute_mean_excitation_energy(composition::Dict{Int, Float64})
    I_values = Dict(
        1=>19.2, 6=>81.0, 7=>82.0, 8=>95.0, 11=>149.0, 12=>156.0,
        15=>173.0, 16=>180.0, 17=>174.0, 19=>190.0, 20=>191.0, 26=>286.0, 53=>491.0
    )
    atomic_masses = Dict(
        1=>1.008, 6=>12.011, 7=>14.007, 8=>15.999, 11=>22.990, 12=>24.305,
        15=>30.974, 16=>32.06, 17=>35.45, 19=>39.098, 20=>40.078, 26=>55.845, 53=>126.904
    )
    log_I_sum = 0.0; Z_A_sum = 0.0
    for (Z, mass_frac) in composition
        A = get(atomic_masses, Z, Float64(Z)*2)
        I = get(I_values, Z, 10.0 * Z)
        Z_A = mass_frac * Z / A
        log_I_sum += Z_A * log(I)
        Z_A_sum += Z_A
    end
    return exp(log_I_sum / Z_A_sum) * u"eV"
end

function load_materials_from_xlsx(xlsx_path)
    xf = XLSX.readxlsx(xlsx_path)
    sheet = xf["Sheet1"]
    data = sheet["A2:P34"]
    materials = Dict{Int, XA.Material}()
    for i in 1:size(data, 1)
        try
            name = String(data[i, 1])
            organ_id = Int(data[i, 16])
            density = Float64(data[i, 15]) * u"g/cm^3"
            comp = Dict{Int, Float64}()
            comp[1] = Float64(data[i, 2])   # H
            comp[6] = Float64(data[i, 3])   # C
            comp[7] = Float64(data[i, 4])   # N
            comp[8] = Float64(data[i, 5])   # O
            comp[11] = Float64(data[i, 6])  # Na
            comp[12] = Float64(data[i, 7])  # Mg
            comp[15] = Float64(data[i, 8])  # P
            comp[16] = Float64(data[i, 9])  # S
            comp[17] = Float64(data[i, 10]) # Cl
            comp[19] = Float64(data[i, 11]) # K
            comp[20] = Float64(data[i, 12]) # Ca
            comp[26] = Float64(data[i, 13]) # Fe
            comp[53] = Float64(data[i, 14]) # I
            filter!(p -> p.second > 0, comp)
            ZA = compute_ZA_ratio(comp)
            I = compute_mean_excitation_energy(comp)
            mat = XA.Material(name, ZA, I, density, comp)
            materials[organ_id] = mat
        catch e
            @warn "Failed to parse row $i" exception=(e, catch_backtrace())
        end
    end
    return materials
end

materials_dict = load_materials_from_xlsx(MATERIAL_XLSX_PATH)
println("  Materials loaded: $(length(materials_dict))")
println("  Loaded in $(@sprintf("%.1f", time() - t1))s")

# ─── VOXEL SIZE ───────────────────────────────────────────────────────────────
base_voxel_cm = (0.03, 0.03, 0.1)
voxel_size_cm = base_voxel_cm .* DOWNSAMPLE_FACTOR

phantom_extent_mm = max(
    size(phantom_labeled, 1) * voxel_size_cm[1],
    size(phantom_labeled, 2) * voxel_size_cm[2]
) * 10.0

println("  Voxel size: $(voxel_size_cm) cm")
println("  Phantom extent: $(round(phantom_extent_mm/10, digits=1)) cm")

# ─── GPU PHANTOM ──────────────────────────────────────────────────────────────
println("Creating GPU phantom...")
phantom_mask_gpu = Metal.MtlArray(phantom_labeled)
phantom_gpu = BS.Phantom(phantom_mask_gpu, materials_dict, voxel_size_cm)
phantom_labeled = nothing  # free CPU copy
GC.gc(true)

# ─── SCANNER ──────────────────────────────────────────────────────────────────
SID = 625.6   # mm
SDD = 1100.0  # mm
mag = SDD / SID

det_cols = if DETECTOR_COLS === nothing
    ceil(Int, phantom_extent_mm * mag / DETECTOR_COL_SIZE)
else
    DETECTOR_COLS
end

println("Scanner: SID=$(SID)mm, SDD=$(SDD)mm, mag=$(round(mag, digits=3))")
println("Detector: $(det_cols) cols × $(DETECTOR_ROWS) rows")
println("  col_size=$(DETECTOR_COL_SIZE)mm, row_size=$(DETECTOR_ROW_SIZE)mm")

scanner = BS.Scanner(
    source_to_isocenter = SID,
    source_to_detector = SDD,
    detector_rows = DETECTOR_ROWS,
    detector_cols = det_cols,
    detector_row_size = DETECTOR_ROW_SIZE,
    detector_col_size = DETECTOR_COL_SIZE,
    detector_shape = BS.CURVED_DETECTOR,
    focal_spot_width = 1.0,
    focal_spot_length = 1.0,
    target_angle = 7.0,
    flat_filter_material = :aluminum,
    flat_filter_thickness = 2.5,
    detector_material = :gos,
    detector_depth = 3.0,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,
    detection_gain = 1.0,
)

# ─── PROTOCOL ─────────────────────────────────────────────────────────────────
protocol = BS.CTProtocol(
    kVp = KVP,
    mA = MA,
    views = VIEWS,
    rotation_time = 0.5
)

# ─── SIM OPTIONS ──────────────────────────────────────────────────────────────
if PHYSICS_ENABLED
    sim_opts = BS.SimOptions(
        fidelity = :high,
        use_fill_factor = USE_FILL_FACTOR,
        use_flat_filter = USE_FLAT_FILTER,
        use_bowtie_filter = USE_BOWTIE_FILTER,
        use_detector_efficiency = USE_DETECTOR_EFFICIENCY,
        use_scatter = false,
        use_scatter_correction = false,
        use_crosstalk = USE_CROSSTALK,
        use_optical_crosstalk = USE_OPTICAL_CROSSTALK,
        use_focal_spot = USE_FOCAL_SPOT,
        use_noise = false,   # ALWAYS off for diagnosis
        use_lag = USE_LAG,
        use_heel_effect = USE_HEEL_EFFECT,
        use_das = false,
        use_bhc = USE_BHC,
        use_pcct_corrections = false,
        pcct_noise_reduction = 0.0,
        n_energy_bins = 100,
        seed = 42
    )
else
    sim_opts = BS.SimOptions(
        fidelity = :high,
        use_fill_factor = false,
        use_flat_filter = false,
        use_bowtie_filter = false,
        use_detector_efficiency = false,
        use_scatter = false,
        use_scatter_correction = false,
        use_crosstalk = false,
        use_optical_crosstalk = false,
        use_focal_spot = false,
        use_noise = false,
        use_lag = false,
        use_heel_effect = false,
        use_das = false,
        use_bhc = false,
        use_pcct_corrections = false,
        pcct_noise_reduction = 0.0,
        n_energy_bins = 100,
        seed = 42
    )
end

# ─── RECON OPTIONS ────────────────────────────────────────────────────────────
recon_size = (RECON_XY, RECON_XY, RECON_SLICES)
recon_opts = BS.ReconOptions(
    algorithm = :fdk,
    matrix_size = recon_size,
    fov_cm = RECON_FOV_CM
)

# ─── WATER CALIBRATION ───────────────────────────────────────────────────────
println("\n--- Water Phantom Calibration ---")
t_water = time()

μ_water = let
    water_size = (400, 400, 16)
    water_voxel_cm = (0.05, 0.05, 0.1)
    water_mask = zeros(UInt8, water_size...)
    cx_w, cy_w = water_size[1] ÷ 2, water_size[2] ÷ 2
    r_w = 200  # 10cm radius
    for i in 1:water_size[1], j in 1:water_size[2]
        if (i - cx_w)^2 + (j - cy_w)^2 <= r_w^2
            water_mask[i, j, :] .= UInt8(1)
        end
    end

    air_material = XA.Material(
        "Air", 0.499, 85.7u"eV", 0.001205u"g/cm^3",
        Dict(7 => 0.7553, 8 => 0.2318, 18 => 0.0129)
    )
    water_materials = Dict(0 => air_material, 1 => XA.Materials.water)
    water_mask_gpu = Metal.MtlArray(water_mask)
    water_phantom_gpu = BS.Phantom(water_mask_gpu, water_materials, water_voxel_cm)

    water_recon_opts = BS.ReconOptions(
        algorithm = :fdk,
        matrix_size = (256, 256, 8),
        fov_cm = 25.0
    )

    ws = BS.create_eict_workspace(scanner, protocol, sim_opts, water_recon_opts, water_phantom_gpu)
    BS.simulate!(ws, water_phantom_gpu, scanner, protocol, sim_opts, water_recon_opts)
    ws_fdk = BS.create_fdk_recon_workspace(
        ws.sino_noisy_out, ws.geom, water_recon_opts.matrix_size;
        filter=BS.HannFilter()
    )
    vol = Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, water_recon_opts.matrix_size))

    # Extract center ROI
    nx, ny, nz = size(vol)
    cx, cy = nx ÷ 2, ny ÷ 2
    r = nx ÷ 10
    z_start = max(1, nz ÷ 4)
    z_end = min(nz, 3 * nz ÷ 4)
    vals = Float64[]
    for k in z_start:z_end, j in (cy - r):(cy + r), i in (cx - r):(cx + r)
        if (i - cx)^2 + (j - cy)^2 <= r^2
            push!(vals, vol[i, j, k])
        end
    end
    result = mean(vals)

    ws = nothing; ws_fdk = nothing; vol = nothing
    water_mask_gpu = nothing; water_phantom_gpu = nothing
    GC.gc(true)

    result
end

println("  μ_water = $(round(μ_water, sigdigits=4)) cm⁻¹")
println("  Water calibration in $(@sprintf("%.1f", time() - t_water))s")

# ─── SIMULATE ─────────────────────────────────────────────────────────────────
println("\n--- Simulation ---")
t_sim = time()

ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_gpu)
@time BS.simulate!(ws, phantom_gpu, scanner, protocol, sim_opts, recon_opts)

sino = ws.sino_noisy_out
geom = ws.geom

println("  Sinogram size: $(size(sino))")
println("  Simulation in $(@sprintf("%.1f", time() - t_sim))s")

# ─── RECONSTRUCT ──────────────────────────────────────────────────────────────
println("\n--- Reconstruction ---")
t_recon = time()

ws_fdk = BS.create_fdk_recon_workspace(sino, geom, recon_size; filter=BS.HannFilter())
recon = Array(BS.reconstruct!(ws_fdk, sino, geom, recon_size))

# Convert to HU
recon_hu = BS.to_hounsfield(recon; μ_water=μ_water)

println("  Recon size: $(size(recon))")
println("  Recon range: [$(round(minimum(recon_hu), digits=1)), $(round(maximum(recon_hu), digits=1))] HU")
println("  Reconstruction in $(@sprintf("%.1f", time() - t_recon))s")

# ─── SAVE RECONSTRUCTION PNG ─────────────────────────────────────────────────
println("\n--- Saving outputs ---")

slice_idx = RECON_SLICES ÷ 2

# Reconstruction image
f = CM.Figure(size=(600, 600))
ax = CM.Axis(f[1, 1],
    title="$TEST_NAME (slice $slice_idx)",
    aspect=CM.DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false
)
CM.heatmap!(ax, recon_hu[:, :, slice_idx], colormap=:grays, colorrange=HU_WINDOW)
CM.Colorbar(f[1, 2], colormap=:grays, colorrange=HU_WINDOW, label="HU")
recon_path = joinpath(OUTPUT_DIR, "diag_$(TEST_NAME).png")
CM.save(recon_path, f)
println("  Saved: $recon_path")

# Sinogram — middle row, all angles
sino_cpu = Array(sino)
f2 = CM.Figure(size=(800, 400))
ax2 = CM.Axis(f2[1, 1],
    title="Sinogram — $TEST_NAME (row $(size(sino_cpu, 2) ÷ 2))",
    xlabel="Detector column", ylabel="View angle"
)
CM.heatmap!(ax2, sino_cpu[:, size(sino_cpu, 2) ÷ 2, :]')
sino_path = joinpath(OUTPUT_DIR, "sino_$(TEST_NAME).png")
CM.save(sino_path, f2)
println("  Saved: $sino_path")

# ─── CLEANUP ──────────────────────────────────────────────────────────────────
ws = nothing; ws_fdk = nothing; sino = nothing; sino_cpu = nothing
GC.gc(true)

println("\n" * "=" ^ 70)
println("DONE — TEST: $TEST_NAME")
println("Total time: $(@sprintf("%.1f", time() - t0))s")
println("=" ^ 70)
