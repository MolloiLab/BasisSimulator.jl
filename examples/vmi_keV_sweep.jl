# =============================================================================
# Virtual Monoenergetic Imaging (VMI) keV Sweep Example
# =============================================================================
#
# This example demonstrates:
# 1. Dual-energy forward projection with GE GSI protocol
# 2. Material decomposition (water/iodine basis)
# 3. VMI reconstruction at multiple energies (40-140 keV)
# 4. Visualization of energy-dependent contrast behavior
#
# Key concepts:
# - Iodine: Maximum enhancement at ~40-50 keV (above K-edge at 33.2 keV)
# - Calcium: Monotonic decrease with energy (K-edge at 4 keV, below diagnostic range)
# - Water: HU = 0 at all energies (by definition)
# - 70 keV: Approximately equivalent to 120 kVp single-energy CT
#
# =============================================================================

using BasisSimulator
using Statistics

# Check for GPU
HAS_GPU = false
try
    using Metal
    if Metal.functional()
        HAS_GPU = true
        println("Using Metal GPU acceleration")
    end
catch
end

# Helper to move data to appropriate backend
to_device(x) = HAS_GPU ? MtlArray(x) : x
to_cpu(x) = HAS_GPU ? Array(x) : x

# =============================================================================
# Setup: Phantom and Scanner
# =============================================================================

println("=" ^ 60)
println("VMI keV Sweep Demonstration")
println("=" ^ 60)
println()

# Create Gammex 472 phantom (contains both calcium and iodine inserts)
println("Creating phantom...")
phantom = create_gammex_472(n_voxels=128, n_slices=16, fov_cm=35.0, z_cm=4.0)
println("  Phantom size: $(size(phantom.mask))")

# Create scanner geometry
println("Creating scanner geometry...")
spec = GERevolutionApex()
geom = create_aquilion_one(n_angles=180, n_rows=16, n_cols=256, fov_cm=35.0, z_cm=4.0)
println("  Angles: $(geom.n_angles)")
println("  Detector: $(geom.n_cols) × $(geom.n_rows)")

# Get phantom materials
materials = get_region_materials()

# =============================================================================
# Dual-Energy Forward Projection
# =============================================================================

println()
println("-" ^ 60)
println("Dual-Energy Forward Projection (GE GSI Protocol)")
println("-" ^ 60)

# Create GSI protocol (80/140 kVp rapid switching)
protocol = default_gsi_protocol(
    low_mA = 400.0,   # 80 kVp tube current
    high_mA = 400.0   # 140 kVp tube current
)
println("  Protocol: $(protocol.low_kvp)/$(protocol.high_kvp) kVp")
println("  Current: $(protocol.low_mA)/$(protocol.high_mA) mA")

# Forward projection
println("  Running dual-energy forward projection...")
mask_device = to_device(phantom.mask)
de_sino = forward_project_dual_energy(
    mask_device, geom, protocol;
    materials = materials,
    scanner = spec
)

println("  Low kVp sinogram: size=$(size(de_sino.low)), mean=$(round(mean(to_cpu(de_sino.low)), digits=3))")
println("  High kVp sinogram: size=$(size(de_sino.high)), mean=$(round(mean(to_cpu(de_sino.high)), digits=3))")

# =============================================================================
# Material Decomposition
# =============================================================================

println()
println("-" ^ 60)
println("Material Decomposition (Water/Iodine Basis)")
println("-" ^ 60)

mat_map = decompose_materials(de_sino; basis=(:water, :iodine))
println("  Basis: water/iodine")
println("  Water map: mean=$(round(mean(to_cpu(mat_map.material1)), digits=3))")
println("  Iodine map: mean=$(round(mean(to_cpu(mat_map.material2)), digits=3))")

# =============================================================================
# VMI Reconstruction at Multiple Energies
# =============================================================================

println()
println("-" ^ 60)
println("VMI Reconstruction at Multiple Energies")
println("-" ^ 60)

# Define energies for keV sweep
energies = [40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 120.0, 140.0]
recon_size = (128, 128, 16)

# Storage for results
vmi_images = Dict{Float64, Array{Float32,3}}()

println()
println("Reconstructing VMI at each energy:")
println("-" ^ 50)

for E in energies
    print("  $E keV: ")

    # Reconstruct VMI using FDK
    vmi_hu = reconstruct_vmi(mat_map, E, geom, recon_size;
                             method=:fdk, to_hu=true)
    vmi_images[E] = vmi_hu

    # Report statistics
    center_slice = vmi_hu[:, :, 8]  # Middle slice
    println("mean=$(round(mean(center_slice), digits=1)) HU, " *
            "std=$(round(std(center_slice), digits=1)) HU, " *
            "range=[$(round(minimum(center_slice), digits=0)), $(round(maximum(center_slice), digits=0))]")
end

# =============================================================================
# Analyze Energy-Dependent Contrast
# =============================================================================

println()
println("-" ^ 60)
println("Energy-Dependent Contrast Analysis")
println("-" ^ 60)

# Define ROI masks for different materials
mid_z = 8
center = div(128, 2)

# Water ROI (center region, solid water)
water_mask = phantom.mask[:, :, mid_z] .== UInt8(REGION_SOLID_WATER)

# Calcium ROI (Ca_200 insert)
ca_mask = phantom.mask[:, :, mid_z] .== UInt8(REGION_CA_200)

# Iodine ROI (I_10_0 insert)
i_mask = phantom.mask[:, :, mid_z] .== UInt8(REGION_I_10_0)

# Calculate HU at each energy for each material
println()
println("Material HU vs Energy:")
println("-" ^ 60)
println("Energy (keV) | Water HU | Ca-200 HU | I-10.0 HU")
println("-" ^ 60)

water_hu = Float64[]
ca_hu = Float64[]
iodine_hu = Float64[]

for E in energies
    img = vmi_images[E][:, :, mid_z]

    w_hu = sum(water_mask) > 0 ? mean(img[water_mask]) : NaN
    c_hu = sum(ca_mask) > 0 ? mean(img[ca_mask]) : NaN
    i_hu = sum(i_mask) > 0 ? mean(img[i_mask]) : NaN

    push!(water_hu, w_hu)
    push!(ca_hu, c_hu)
    push!(iodine_hu, i_hu)

    println("   $(lpad(Int(E), 3))       | $(lpad(round(Int, w_hu), 6))   | $(lpad(round(Int, c_hu), 7))   | $(lpad(round(Int, i_hu), 7))")
end

println("-" ^ 60)

# =============================================================================
# Verify Physics Behavior
# =============================================================================

println()
println("-" ^ 60)
println("Physics Validation")
println("-" ^ 60)

# Water should be approximately 0 HU at all energies
water_deviation = maximum(abs.(water_hu))
println("Water HU stability: max deviation = $(round(water_deviation, digits=1)) HU")
if water_deviation < 50
    println("  ✓ PASS: Water approximately 0 HU at all energies")
else
    println("  ✗ FAIL: Water HU varies too much with energy")
end

# Iodine should have highest HU at low keV (above K-edge)
if !any(isnan.(iodine_hu))
    iodine_enhancement = iodine_hu[1] / iodine_hu[4]  # 40 keV vs 70 keV
    println("Iodine enhancement (40 keV / 70 keV): $(round(iodine_enhancement, digits=2))×")
    if iodine_enhancement > 1.2
        println("  ✓ PASS: Iodine shows K-edge enhancement at low keV")
    else
        println("  ✗ Unexpected: Iodine enhancement lower than expected")
    end
end

# Calcium should decrease monotonically with energy
if !any(isnan.(ca_hu))
    ca_monotonic = all(diff(ca_hu) .< 0)
    println("Calcium trend: $(ca_monotonic ? "monotonically decreasing" : "not monotonic")")
    if ca_monotonic
        println("  ✓ PASS: Calcium HU decreases with energy (K-edge below diagnostic range)")
    else
        println("  ⚠ Note: Calcium trend may be affected by reconstruction artifacts")
    end
end

# =============================================================================
# Clinical Energy Selection Guide
# =============================================================================

println()
println("-" ^ 60)
println("Clinical VMI Energy Selection Guide")
println("-" ^ 60)
println()
println("Energy (keV) | Relative Noise | Use Case")
println("-" ^ 55)
println("   40-50     | High (2-3×)    | Maximum iodine contrast, salvage low-dose")
println("   50-60     | Moderate (1.5×)| Vascular imaging, lesion detection")
println("   65-75     | Baseline (1×)  | Standard imaging (≈120 kVp equivalent)")
println("   80-100    | Low (0.9×)     | Reduced beam hardening, liver imaging")
println("  100-140    | Lowest (0.8×)  | Metal artifact reduction")
println("-" ^ 55)
println()
println("Recommendation: Start with 70 keV for standard imaging,")
println("then adjust based on diagnostic task.")

println()
println("=" ^ 60)
println("VMI keV Sweep Complete")
println("=" ^ 60)
