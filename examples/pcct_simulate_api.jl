# =============================================================================
# PCCT via simulate() API — The 4-Struct Photon-Counting CT Pipeline
# =============================================================================
#
# Demonstrates the complete PCCT workflow using the high-level simulate() API.
# The user never touches PhotonCountingDetector directly — PCCT is detected
# automatically via Scanner's detector_type field.
#
# KEY WORKFLOW:
# 1. create_naeotom_alpha() → Scanner (PCCT auto-detected)
# 2. CTProtocol            → Acquisition settings (single kVp)
# 3. SimOptions            → Physics fidelity
# 4. ReconOptions          → Reconstruction + VMI + material basis
# 5. simulate()            → Routes to PCCT pipeline automatically
#
# PCCT OUTPUT (SimulationResult fields):
# - result.pcct_sinogram      → EnergyResolvedSinogram (4 energy bins)
# - result.material_maps      → MaterialMap (2-material decomposition, unified with dual-kVp)
# - result.pcct_vmi_volumes   → Dict{Float64, Array} (VMI at each energy)
# - result.reconstruction     → Standard FDK reconstruction
#
# =============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BasisSimulator
using Statistics

# =============================================================================
# Configuration
# =============================================================================

const N_VOXELS = 32
const N_SLICES = 8
const FOV_CM = 20.0
const Z_CM = 4.0
const RECON_SIZE = (N_VOXELS, N_VOXELS, N_SLICES)

println("=" ^ 70)
println("PCCT SIMULATION via simulate() API")
println("=" ^ 70)

# =============================================================================
# Step 1: Create PCCT Scanner
# =============================================================================

println("\n--- Step 1: Create PCCT Scanner ---")

# create_naeotom_alpha() returns a Scanner with detector_type=:photon_counting
scanner = create_naeotom_alpha(mode=:standard)

println("Scanner created: NAEOTOM Alpha (standard mode)")
println("  detector_type:     $(scanner.detector_type)")
println("  n_energy_bins:     $(scanner.n_energy_bins)")
println("  energy_thresholds: $(scanner.energy_thresholds) keV")
println("  is_pcct:           $(is_pcct(scanner))")

# =============================================================================
# Step 2: Create Phantom
# =============================================================================

println("\n--- Step 2: Create Phantom ---")

phantom = create_gammex_472(n_voxels=N_VOXELS, n_slices=N_SLICES, fov_cm=FOV_CM, z_cm=Z_CM)
println("Phantom: Gammex 472 ($(size(phantom.mask)))")

# =============================================================================
# Step 3: Define Acquisition Protocol
# =============================================================================

println("\n--- Step 3: Define Protocol ---")

protocol = CTProtocol(
    kVp = 120.0,
    mA = 300.0,
    views = 180
)
println("Protocol: $(protocol.kVp) kVp, $(protocol.mA) mA, $(protocol.views) views")

# =============================================================================
# Step 4: Configure Simulation Options
# =============================================================================

println("\n--- Step 4: SimOptions ---")

sim_opts = SimOptions(fidelity=:medium, seed=42)
println("SimOptions: fidelity=$(sim_opts.fidelity), noise=$(sim_opts.use_noise)")

# =============================================================================
# Step 5: Configure Reconstruction + VMI
# =============================================================================

println("\n--- Step 5: ReconOptions ---")

recon_opts = ReconOptions(
    algorithm = :fdk,
    matrix_size = RECON_SIZE,
    fov_cm = FOV_CM,
    vmi_basis = [:water, :iodine, :calcium],  # 3-material PCCT decomposition
    vmi_energies = [40.0, 50.0, 70.0, 100.0]  # VMI at these energies
)
println("ReconOptions:")
println("  algorithm:    $(recon_opts.algorithm)")
println("  matrix_size:  $(recon_opts.matrix_size)")
println("  vmi_basis:    $(recon_opts.vmi_basis)  (3-material!)")
println("  vmi_energies: $(recon_opts.vmi_energies) keV")

# =============================================================================
# Step 6: Run Simulation (Auto-Routes to PCCT)
# =============================================================================

println("\n--- Step 6: simulate() ---")
println("Calling simulate()... (auto-detects PCCT from Scanner)")

result = simulate(phantom, scanner, protocol, sim_opts, recon_opts)

println("\nSimulation complete!")
println("  Type: $(typeof(result))")

# =============================================================================
# Step 7: Inspect PCCT Results
# =============================================================================

println("\n--- Step 7: PCCT Results ---")

# 7a. Energy-resolved sinogram
pcct_sino = result.pcct_sinogram
println("\nPCCT Sinogram (EnergyResolvedSinogram):")
println("  Number of bins: $(length(pcct_sino.bins))")
println("  Thresholds: $(pcct_sino.thresholds_keV) keV")
println("  Sino shape:  $(size(pcct_sino.bins[1]))")
for (i, bin) in enumerate(pcct_sino.bins)
    lower = pcct_sino.thresholds_keV[i]
    upper = i < length(pcct_sino.thresholds_keV) ? pcct_sino.thresholds_keV[i+1] : 120.0
    println("    Bin $i ($lower-$upper keV): mean=$(round(mean(bin), digits=4))")
end

# 7b. Material decomposition maps (unified MaterialMap)
mat_map = result.material_maps
println("\nPCCT Material Maps (2-material, unified with dual-kVp):")
println("  Material 1: $(mat_map.material1_name)")
println("  Material 2: $(mat_map.material2_name)")
println("  Domain: $(mat_map.domain)")
m1 = Array(mat_map.material1)
m2 = Array(mat_map.material2)
println("    $(mat_map.material1_name): range=[$(round(minimum(m1), digits=4)), $(round(maximum(m1), digits=4))]")
println("    $(mat_map.material2_name): range=[$(round(minimum(m2), digits=4)), $(round(maximum(m2), digits=4))]")

# 7c. VMI volumes
println("\nPCCT VMI Volumes:")
for E in sort(collect(keys(result.pcct_vmi_volumes)))
    vol = result.pcct_vmi_volumes[E]
    println("  $(E) keV: size=$(size(vol)), range=[$(round(minimum(vol), digits=4)), $(round(maximum(vol), digits=4))]")
end

# 7d. Standard reconstruction
recon = result.reconstruction
println("\nStandard Reconstruction:")
println("  Size: $(size(recon))")
println("  Range: [$(round(minimum(recon), digits=4)), $(round(maximum(recon), digits=4))]")

# =============================================================================
# Step 8: HU Calibration and Verification
# =============================================================================

println("\n--- Step 8: HU Verification ---")

water_mask = phantom.mask .== UInt8(REGION_SOLID_WATER)
n_water = sum(water_mask)
println("Water voxels: $n_water")

if n_water > 0
    # Calibrate each VMI energy
    println("\nVMI HU Calibration:")
    for E in sort(collect(keys(result.pcct_vmi_volumes)))
        vol = result.pcct_vmi_volumes[E]
        μ_water = mean(vol[water_mask])
        if μ_water > 1e-6
            hu = 1000.0 * (mean(vol[water_mask]) - μ_water) / μ_water
            hu_std = 1000.0 * std(vol[water_mask]) / μ_water
            println("  $(E) keV: Water HU = $(round(hu, digits=1)) ± $(round(hu_std, digits=1))")
        else
            println("  $(E) keV: μ_water too small ($(round(μ_water, digits=6)))")
        end
    end
end

# =============================================================================
# Step 9: Compare with Non-PCCT Scanner
# =============================================================================

println("\n--- Step 9: Comparison with EID Scanner ---")

# Regular energy-integrating scanner (same geometry but no PCCT)
eid_scanner = Scanner(
    source_to_isocenter = 595.0,
    source_to_detector = 1085.5,
    detector_rows = 32,
    detector_cols = 128,
    detector_row_size = 1.0,
    detector_col_size = 1.0
)

println("EID Scanner: is_pcct=$(is_pcct(eid_scanner))")

eid_result = simulate(phantom, eid_scanner, protocol, sim_opts,
    ReconOptions(algorithm=:fdk, matrix_size=RECON_SIZE, fov_cm=FOV_CM))

println("EID result:")
println("  pcct_sinogram:      $(eid_result.pcct_sinogram)")
println("  pcct_material_maps: $(eid_result.pcct_material_maps)")
println("  pcct_vmi_volumes:   $(length(eid_result.pcct_vmi_volumes)) entries")
println("  (EID scanner correctly routes to standard axial pipeline)")

# =============================================================================
# Summary
# =============================================================================

println("\n" * "=" ^ 70)
println("SUMMARY: PCCT via simulate() API")
println("=" ^ 70)
println("""

The simulate() API automatically detects PCCT scanners and runs:
1. Polychromatic forward projection → energy-resolved sinograms
2. Per-bin Poisson noise (no electronic noise — PCCT advantage)
3. N-material decomposition (3-material with 4 bins)
4. VMI synthesis from material maps
5. Standard FDK reconstruction from combined sinogram

Key differences from dual-energy:
- Single kVp acquisition (no switching needed)
- 4 energy bins (vs 2 for DECT)
- 3-material decomposition (vs 2 for DECT)
- Material-based VMI (not bin-weighted)
- No electronic noise contribution

User-facing API: The user only calls:
    scanner = create_naeotom_alpha()
    result = simulate(phantom, scanner, protocol, sim_opts, recon_opts)

Everything else is handled internally.
""")
println("=" ^ 70)
