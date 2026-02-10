using BasisSimulator
println("✓ Package loaded successfully")

# Quick test: create workspace and verify it works
println("\nCreating test objects...")
scanner = Scanner()
protocol = CTProtocol(kVp=120, mA=200, views=16)
sim_opts = SimOptions(fidelity=:high)
recon_opts = ReconOptions(fov_cm=25.0)

# Small phantom
mask = zeros(UInt8, 16, 16, 1)
mask[5:12, 5:12, 1] .= 1
phantom = Phantom(mask, nothing, (0.1, 0.1, 0.5), (0.0, 0.0, 0.0), (1.6, 1.6, 0.5))

println("Creating EICT workspace...")
ws = create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
println("✓ Workspace created successfully")
println("  Type: ", typeof(ws))
println("  physics_output: ", size(ws.physics_output))
println("  lag_intensity: ", size(ws.lag_intensity))
println("  has scatter_kernel: ", ws.scatter_kernel !== nothing)
println("  has scatter_correct_kernel: ", ws.scatter_correct_kernel !== nothing)
println("  has crosstalk_kernel: ", ws.crosstalk_kernel !== nothing)
println("  has optical_crosstalk_kernel: ", ws.optical_crosstalk_kernel !== nothing)
println("  has flat_filter_proj: ", ws.flat_filter_projection !== nothing)
println("  has bowtie_proj: ", ws.bowtie_projection !== nothing)
println("  has lag_coeffs: ", ws.lag_coeffs !== nothing)
println("  has focal_spot_kernel: ", ws.focal_spot_kernel !== nothing)

# Test simulate!
println("\nRunning simulate!...")
result = simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)
println("✓ simulate! completed")
println("  sino_ideal range: ", extrema(result.sino_ideal))

# Test parity with allocating path
println("\nRunning allocating simulate...")
result_alloc = simulate(phantom, scanner, protocol, sim_opts, recon_opts)
sino_alloc = result_alloc.sinogram_ideal

max_diff = maximum(abs.(result.sino_ideal .- sino_alloc))
println("  Max abs diff (parity): ", max_diff)
if max_diff < 0.01
    println("✓ PARITY: PASS")
else
    println("✗ PARITY: FAIL")
end

# Test allocations
println("\nMeasuring allocations (warmup)...")
simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)
println("Measuring allocations (real)...")
allocs = @allocated simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)
println("  @allocated: ", allocs, " bytes")
if allocs < 5000
    println("✓ ALLOCATION GATE: PASS (< 5000 bytes)")
else
    println("✗ ALLOCATION GATE: FAIL (need < 5000 bytes)")
end

println("\nDone!")
