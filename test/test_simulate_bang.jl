using BasisSimulator

println("Module loaded OK")

# Create test setup
phantom = create_gammex_472(n_voxels=32, n_slices=4, fov_cm=20.0, z_cm=2.0)
pcct_scanner = create_naeotom_alpha(mode=:standard)
protocol = CTProtocol(kVp=120.0, mA=300.0, views=36)
sim_opts = SimOptions(fidelity=:low, seed=42)
recon_opts = ReconOptions(
    algorithm=:fdk,
    matrix_size=(16, 16, 4),
    fov_cm=20.0,
    vmi_basis=[:water, :iodine],
    vmi_energies=[40.0, 70.0]
)

# Run simulate() — now delegates to simulate!() internally
println("Running simulate() [now uses simulate!() internally]...")
result1 = simulate(phantom, pcct_scanner, protocol, sim_opts, recon_opts)
println("simulate() OK")

# Run again — same seed should produce identical results
println("Running simulate() [call 2, same seed]...")
result2 = simulate(phantom, pcct_scanner, protocol, sim_opts, recon_opts)
println("simulate() call 2 OK")

# Parity check
println("\n=== Parity: call1 vs call2 ===")
all_pass = true

for i in 1:length(result1.pcct_sinogram.bins)
    b1 = Array(result1.pcct_sinogram.bins[i])
    b2 = Array(result2.pcct_sinogram.bins[i])
    max_err = maximum(abs.(b1 .- b2))
    status = max_err == 0.0 ? "PASS" : "FAIL"
    if max_err != 0.0
        global all_pass = false
    end
    println("  $status  pcct_bin$i  max_abs_err=$max_err")
end

i1 = Array(result1.sinogram_ideal)
i2 = Array(result2.sinogram_ideal)
max_err = maximum(abs.(i1 .- i2))
println("  ", max_err == 0.0 ? "PASS" : "FAIL", "  ideal_combined  max_abs_err=$max_err")
if max_err != 0.0; all_pass = false; end

n1 = Array(result1.sinogram_noisy)
n2 = Array(result2.sinogram_noisy)
max_err = maximum(abs.(n1 .- n2))
println("  ", max_err == 0.0 ? "PASS" : "FAIL", "  noisy_combined  max_abs_err=$max_err")
if max_err != 0.0; all_pass = false; end

if result1.pcct_material_maps !== nothing
    for i in 1:length(result1.pcct_material_maps.materials)
        m1 = result1.pcct_material_maps.materials[i]
        m2 = result2.pcct_material_maps.materials[i]
        local max_err = maximum(abs.(m1 .- m2))
        local status = max_err == 0.0 ? "PASS" : "FAIL"
        if max_err != 0.0; global all_pass = false; end
        name = result1.pcct_material_maps.material_names[i]
        println("  $status  material_$name  max_abs_err=$max_err")
    end
end

println(all_pass ? "\n=== ALL PARITY CHECKS PASS ===" : "\n=== SOME PARITY CHECKS FAILED ===")

# Explicit simulate!() call
println("\nRunning explicit simulate!(ws, ...)...")
ws = create_workspace(pcct_scanner, protocol, sim_opts, recon_opts, phantom)
result3 = simulate!(ws, phantom, pcct_scanner, protocol, sim_opts, recon_opts)
println("simulate!() OK")

# Check simulate!() result matches simulate()
b1 = Array(result1.pcct_sinogram.bins[1])
b3 = Array(result3.pcct_sinogram.bins[1])
max_err = maximum(abs.(b1 .- b3))
println("simulate!() vs simulate() pcct_bin1: ", max_err == 0.0 ? "PASS" : "FAIL", "  max_abs_err=$max_err")
