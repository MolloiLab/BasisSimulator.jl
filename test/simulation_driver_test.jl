
using Test
using BasisSimulator

@testset "High-Level Simulation Driver" begin
    # 1. Setup minimal inputs
    scanner = Scanner() # Default generic scanner
    
    # Use low resolution for speed
    # New API: Use mA directly
    protocol = CTProtocol(views=36, mA=100.0, rotation_time=1.0) 
    
    # Create a small water cylinder mask (64x64x16)
    # Region 2 = Water, Region 0 = Air (Background)
    mask = zeros(UInt8, 64, 64, 16)
    # Simple circle in center
    for i in 1:64, j in 1:64
        if (i-32)^2 + (j-32)^2 < 20^2
            mask[i, j, :] .= 2 # Region 2 is Water in get_region_materials()
        end
    end
    phantom = MockPhantom(mask)
    
    # 2. Configure Options
    # Use :ideal fidelity to avoid expensive physics (scatter etc) during test
    sim_opts = SimOptions(fidelity=:ideal, use_noise=true)
    
    recon_opts = ReconOptions(
        matrix_size=(64, 64, 16),
        fov_cm=35.0
    )
    
    # 3. Run Simulation
    println("Running high-level simulation test...")
    result = simulate(phantom, scanner, protocol, sim_opts, recon_opts)
    
    # 4. Verify Outputs
    @test result isa SimulationResult
    @test size(result.sinogram_ideal) == (900, 64, 36) # Scanner default cols/rows
    @test size(result.reconstruction) == (64, 64, 16)
    
    # Check that noise was applied (ideal != noisy)
    # Since we set use_noise=true, they should be different
    @test result.sinogram_ideal != result.sinogram_noisy
    
    # Check basic reconstruction value (Water should be > 0)
    center_val = result.reconstruction[32, 32, 8]
    println("Center reconstruction value: $center_val")
    @test center_val > 0.01 # Should be around 0.2 for water at 60keV (cm^-1)
    
    println("High-level driver test passed!")
end
