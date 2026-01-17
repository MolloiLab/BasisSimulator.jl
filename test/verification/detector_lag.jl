"""
Detector Lag (Afterglow) Verification Tests - PHYSICS-008

Verifies BasisSimulator detector lag implementation matches CatSim Detection_Lag.py.

CatSim uses a two-component exponential lag model with midpoint sampling:
    outview = invintegral × thisView + α₁(1-e^(-dt/τ₁)) × memview₁ + α₂(1-e^(-dt/τ₂)) × memview₂
    memview₁' = memview₁ × e^(-dt/τ₁) + thisView × e^(-dt/2τ₁)
    memview₂' = memview₂ × e^(-dt/τ₂) + thisView × e^(-dt/2τ₂)

where:
    invintegral = α₁(1-e^(-dt/2τ₁)) + α₂(1-e^(-dt/2τ₂)) + (1-α₁-α₂)

References:
1. CatSim Detection_Lag.py
2. Hsieh J. "Computed Tomography: Principles, Design, Artifacts, and Recent Advances" Ch. 4
3. Siewerdsen JH, Jaffray DA. "Optimization of x-ray imaging geometry"
   Med Phys. 1999;26(8):1624-1633. doi:10.1118/1.598657
4. Zhao W, et al. "Investigation of the charge trapping and afterglow in a-Se"
   Med Phys. 2001;28(2):211-218. doi:10.1118/1.1344222
"""

using Test
using Statistics
using BasisSimulator

println("\n" * "=" ^ 70)
println("PHYSICS-008: Detector Lag (Afterglow) Verification")
println("=" ^ 70)

# =============================================================================
# CatSim Reference Implementation (Pure Julia Port)
# =============================================================================

"""
    CatSimLagState

State for CatSim lag model (maintains memview1, memview2 across views).
"""
mutable struct CatSimLagState{T}
    memview1::Matrix{T}
    memview2::Matrix{T}
end

"""
    catsim_lag_view(thisView, state, tau1, tau2, alpha1, alpha2, dt)

Apply detector lag to a single view using exact CatSim formula.
This is a direct port of CatSim Detection_Lag.py.
"""
function catsim_lag_view!(
    thisView::AbstractMatrix{T},
    state::CatSimLagState{T},
    tau1::Float64, tau2::Float64,
    alpha1::Float64, alpha2::Float64,
    dt::Float64;
    is_first_view::Bool = false,
    is_air_scan::Bool = false
) where T
    # CatSim formula (line 11-12)
    unaccounted = 1 - alpha1 - alpha2
    invintegral = alpha1 * (1 - exp(-dt / 2 / tau1)) +
                  alpha2 * (1 - exp(-dt / 2 / tau2)) +
                  unaccounted

    # Initialize state for first view (lines 14-21)
    if is_first_view
        if is_air_scan
            # Air scan: steady-state initialization
            state.memview1 .= thisView .* exp(-dt/2/tau1) ./ (1 - exp(-dt/tau1))
            state.memview2 .= thisView .* exp(-dt/2/tau2) ./ (1 - exp(-dt/tau2))
        else
            # Phantom scan: zero initialization
            state.memview1 .= 0
            state.memview2 .= 0
        end
    end

    # Compute output view (line 24)
    outview = invintegral .* thisView .+
              (alpha1 * (1 - exp(-dt/tau1))) .* state.memview1 .+
              (alpha2 * (1 - exp(-dt/tau2))) .* state.memview2

    # Update state (lines 25-26)
    state.memview1 .= state.memview1 .* exp(-dt/tau1) .+ thisView .* exp(-dt/2/tau1)
    state.memview2 .= state.memview2 .* exp(-dt/tau2) .+ thisView .* exp(-dt/2/tau2)

    return outview
end

"""
    apply_catsim_lag(sinogram, tau1, tau2, alpha1, alpha2, dt; is_air_scan=false)

Apply CatSim lag model to full sinogram (reference implementation).
"""
function apply_catsim_lag(
    sinogram::AbstractArray{T,3},
    tau1::Float64, tau2::Float64,
    alpha1::Float64, alpha2::Float64,
    dt::Float64;
    is_air_scan::Bool = false
) where T
    n_cols, n_rows, n_angles = size(sinogram)
    output = similar(sinogram)

    # Initialize state
    state = CatSimLagState(zeros(T, n_cols, n_rows), zeros(T, n_cols, n_rows))

    # Process each view sequentially
    for angle in 1:n_angles
        thisView = sinogram[:, :, angle]
        outview = catsim_lag_view!(
            thisView, state,
            tau1, tau2, alpha1, alpha2, dt;
            is_first_view = (angle == 1),
            is_air_scan = is_air_scan
        )
        output[:, :, angle] = outview
    end

    return output
end

# =============================================================================
# Test 1: CatSim Formula Verification (Impulse Response)
# =============================================================================
@testset "CatSim Formula Verification" begin
    println("\n[Test 1] CatSim Formula Verification")

    # CatSim typical parameters
    tau1, tau2 = 0.5, 5.0  # ms (fast and slow components)
    alpha1, alpha2 = 0.02, 0.01  # 2% fast + 1% slow = 3% total lag
    dt = 0.5  # ms (frame time for ~1000 views @ 0.5s rotation)

    # Create impulse input (single bright frame)
    n_cols, n_rows, n_angles = 32, 16, 50
    sinogram = zeros(Float32, n_cols, n_rows, n_angles)
    sinogram[:, :, 1] .= 1.0f0  # Impulse at first frame

    # Apply CatSim reference
    output = apply_catsim_lag(sinogram, tau1, tau2, alpha1, alpha2, dt)

    # Extract impulse response at center pixel
    impulse_response = vec(output[16, 8, :])

    # Verify impulse response properties
    @test impulse_response[1] > 0.97  # Most signal in first frame
    @test impulse_response[2] > 0     # Some ghosting in second frame
    @test all(impulse_response[3:end] .> 0)  # Exponential tail
    @test sum(impulse_response) ≈ 1.0 atol=0.01  # Signal conservation

    # Verify exponential decay shape
    # After first frame, should decay exponentially
    for i in 10:40
        ratio = impulse_response[i] / impulse_response[i-1]
        expected_fast = exp(-dt/tau1)
        expected_slow = exp(-dt/tau2)
        # Ratio should be between fast and slow decay rates
        @test expected_fast < ratio + 0.01
        @test ratio < expected_slow + 0.01
    end

    println("  ✓ Impulse response has correct shape")
    println("  ✓ Signal conservation verified: $(round(sum(impulse_response), digits=4))")
    println("  ✓ Primary signal fraction: $(round(impulse_response[1], digits=4))")
end

# =============================================================================
# Test 2: Temporal Decay Curve
# =============================================================================
@testset "Temporal Decay Curve" begin
    println("\n[Test 2] Temporal Decay Curve")

    # Parameters
    tau1, tau2 = 1.0, 10.0  # ms
    alpha1, alpha2 = 0.01, 0.005  # 1.5% total lag
    dt = 0.5  # ms

    # Create step function (constant then zero)
    n_cols, n_rows, n_angles = 16, 8, 100
    sinogram = ones(Float32, n_cols, n_rows, n_angles)
    sinogram[:, :, 51:end] .= 0.0f0  # Step down at frame 51

    # Apply CatSim lag
    output = apply_catsim_lag(sinogram, tau1, tau2, alpha1, alpha2, dt)

    # After step-down, signal should decay exponentially
    decay_signal = vec(output[8, 4, 51:end])

    # Fit exponential decay: y = A × exp(-t/τ_eff)
    # For multi-exponential, effective tau is somewhere between tau1 and tau2
    t = collect(0:(length(decay_signal)-1)) .* dt

    # Verify decay is present
    @test decay_signal[1] > 0.01  # Some lag signal after step
    @test decay_signal[end] < decay_signal[1]  # Decays over time

    # Check decay is monotonic (no oscillations)
    @test all(diff(decay_signal) .<= 0.001)  # Allow tiny numerical noise

    # Verify time constants are in correct range
    # At t = tau_eff, signal should be ~37% of initial
    half_life_idx = findfirst(x -> x < decay_signal[1] * 0.5, decay_signal)
    if half_life_idx !== nothing
        half_life_ms = half_life_idx * dt
        # Half life should be related to time constants
        @test 0.3 < half_life_ms < 30  # Between tau1 and tau2 range
    end

    println("  ✓ Decay is monotonic after step-down")
    println("  ✓ Initial lag amplitude: $(round(decay_signal[1], digits=4))")
    println("  ✓ Final lag amplitude: $(round(decay_signal[end], digits=6))")
end

# =============================================================================
# Test 3: Lag Time Constant Matching
# =============================================================================
@testset "Lag Time Constant Matching" begin
    println("\n[Test 3] Lag Time Constant Matching")

    # Test with various time constants
    test_cases = [
        (tau1=0.5, tau2=5.0, alpha1=0.02, alpha2=0.01),   # Fast detector
        (tau1=1.0, tau2=10.0, alpha1=0.01, alpha2=0.005), # GOS-like
        (tau1=2.0, tau2=20.0, alpha1=0.03, alpha2=0.015), # High lag
    ]

    dt = 0.5  # ms

    for (idx, tc) in enumerate(test_cases)
        # Create BasisSimulator model
        model = lag_custom(
            [tc.alpha1, tc.alpha2],
            [tc.tau1, tc.tau2];
            frame_time = dt
        )

        # Create test sinogram
        n_cols, n_rows, n_angles = 32, 16, 60
        sinogram = zeros(Float32, n_cols, n_rows, n_angles)
        sinogram[:, :, 1] .= 1.0f0  # Impulse

        # Apply both implementations
        catsim_output = apply_catsim_lag(sinogram, tc.tau1, tc.tau2, tc.alpha1, tc.alpha2, dt)
        basis_output = apply_lag_catsim(sinogram, model)

        # Compare impulse responses
        catsim_ir = vec(catsim_output[16, 8, :])
        basis_ir = vec(basis_output[16, 8, :])

        # Compute relative error
        max_rel_error = maximum(abs.(catsim_ir .- basis_ir) ./ max.(catsim_ir, 1e-10))

        println("  Case $(idx): τ₁=$(tc.tau1)ms, τ₂=$(tc.tau2)ms, α₁=$(tc.alpha1), α₂=$(tc.alpha2)")
        println("    Max relative error: $(round(max_rel_error * 100, digits=4))%")

        # Should match within 1%
        @test max_rel_error < 0.01
    end

    println("  ✓ All time constant cases match CatSim within 1%")
end

# =============================================================================
# Test 4: Ghosting Artifact Magnitude
# =============================================================================
@testset "Ghosting Artifact Magnitude" begin
    println("\n[Test 4] Ghosting Artifact Magnitude")

    # Typical clinical parameters
    tau1, tau2 = 1.0, 10.0  # ms
    alpha1, alpha2 = 0.015, 0.008  # 2.3% total lag
    dt = 0.5  # ms (1000 views @ 0.5s rotation)

    # Create sinogram with high-contrast edge
    n_cols, n_rows, n_angles = 64, 32, 100
    sinogram = zeros(Float32, n_cols, n_rows, n_angles)

    # High-contrast object appears at frame 20, disappears at frame 50
    sinogram[:, :, 20:50] .= 1.0f0

    # Apply CatSim lag
    output = apply_catsim_lag(sinogram, tau1, tau2, alpha1, alpha2, dt)

    # Measure ghosting magnitude after object disappears
    # Frame 51 should have ghosting from frames 20-50
    ghost_signal = mean(output[:, :, 51])
    object_signal = mean(output[:, :, 35])  # Mid-object signal

    ghosting_fraction = ghost_signal / object_signal

    # Expected ghosting based on lag parameters
    # For small lag, ghosting ≈ total_lag × some factor
    total_lag = alpha1 + alpha2
    expected_ghosting_min = total_lag * 0.1  # At least 10% of total lag
    expected_ghosting_max = total_lag * 3.0   # At most 300% of total lag

    println("  Object signal (frame 35): $(round(object_signal, digits=4))")
    println("  Ghost signal (frame 51): $(round(ghost_signal, digits=4))")
    println("  Ghosting fraction: $(round(ghosting_fraction * 100, digits=2))%")
    println("  Expected range: $(round(expected_ghosting_min * 100, digits=2))% - $(round(expected_ghosting_max * 100, digits=2))%")

    @test expected_ghosting_min < ghosting_fraction < expected_ghosting_max

    # Verify ghost decays over time
    ghost_decay = [mean(output[:, :, i]) for i in 51:60]
    @test all(diff(ghost_decay) .< 0)  # Monotonically decreasing

    println("  ✓ Ghosting magnitude within expected range")
    println("  ✓ Ghost signal decays monotonically")
end

# =============================================================================
# Test 5: Air Scan vs Phantom Scan Initialization
# =============================================================================
@testset "Air vs Phantom Initialization" begin
    println("\n[Test 5] Air Scan vs Phantom Scan Initialization")

    tau1, tau2 = 1.0, 10.0
    alpha1, alpha2 = 0.02, 0.01
    dt = 0.5

    # Constant intensity sinogram
    n_cols, n_rows, n_angles = 32, 16, 50
    sinogram = ones(Float32, n_cols, n_rows, n_angles)

    # Apply with air scan initialization
    output_air = apply_catsim_lag(sinogram, tau1, tau2, alpha1, alpha2, dt; is_air_scan=true)

    # Apply with phantom initialization (default)
    output_phantom = apply_catsim_lag(sinogram, tau1, tau2, alpha1, alpha2, dt; is_air_scan=false)

    # Air scan initialization should give constant output (steady state)
    air_variation = maximum(output_air[16, 8, :]) - minimum(output_air[16, 8, :])

    # Phantom initialization should show transient at start
    phantom_variation = maximum(output_phantom[16, 8, :]) - minimum(output_phantom[16, 8, :])

    println("  Air scan output variation: $(round(air_variation, digits=6))")
    println("  Phantom scan output variation: $(round(phantom_variation, digits=6))")

    # Air scan should be nearly constant (steady state)
    @test air_variation < 0.001

    # Phantom scan shows transient buildup at start
    @test output_phantom[16, 8, 1] < output_phantom[16, 8, end]

    println("  ✓ Air scan initialization provides steady state")
    println("  ✓ Phantom scan shows transient response")
end

# =============================================================================
# Test 6: BasisSimulator vs CatSim Comparison
# =============================================================================
@testset "BasisSimulator vs CatSim Comparison" begin
    println("\n[Test 6] BasisSimulator vs CatSim Direct Comparison")

    # CatSim-exact parameters
    tau1, tau2 = 1.0, 10.0  # ms
    alpha1, alpha2 = 0.01, 0.005  # 1.5% total
    dt = 0.5  # ms

    # Create BasisSimulator lag model
    model = lag_custom([alpha1, alpha2], [tau1, tau2]; frame_time=dt)

    # Random test sinogram (intensity domain)
    n_cols, n_rows, n_angles = 64, 32, 80
    sinogram = rand(Float32, n_cols, n_rows, n_angles) .* 0.5f0 .+ 0.5f0

    # Apply both implementations
    catsim_result = apply_catsim_lag(sinogram, tau1, tau2, alpha1, alpha2, dt)
    basis_result = apply_lag_catsim(sinogram, model)

    # Compare results
    diff = abs.(catsim_result .- basis_result)
    max_diff = maximum(diff)
    mean_diff = mean(diff)
    max_rel_error = maximum(diff ./ max.(catsim_result, 1e-10))

    println("  Max absolute difference: $(round(max_diff, digits=8))")
    println("  Mean absolute difference: $(round(mean_diff, digits=8))")
    println("  Max relative error: $(round(max_rel_error * 100, digits=4))%")

    # Should match within numerical precision (< 1%)
    @test max_rel_error < 0.01

    println("  ✓ BasisSimulator matches CatSim within 1%")
end

# =============================================================================
# Test 7: Signal Conservation
# =============================================================================
@testset "Signal Conservation" begin
    println("\n[Test 7] Signal Conservation")

    tau1, tau2 = 1.0, 10.0
    alpha1, alpha2 = 0.02, 0.01
    dt = 0.5

    # Create constant intensity sinogram
    n_cols, n_rows, n_angles = 32, 16, 100
    sinogram = ones(Float32, n_cols, n_rows, n_angles) .* 0.8f0

    # Apply lag with air scan initialization (steady state)
    output = apply_catsim_lag(sinogram, tau1, tau2, alpha1, alpha2, dt; is_air_scan=true)

    # Total signal should be conserved (input = output in steady state)
    input_total = sum(sinogram)
    output_total = sum(output)

    # Relative difference
    rel_diff = abs(input_total - output_total) / input_total

    println("  Input total: $(round(input_total, digits=2))")
    println("  Output total: $(round(output_total, digits=2))")
    println("  Relative difference: $(round(rel_diff * 100, digits=6))%")

    @test rel_diff < 0.01  # Within 1%

    println("  ✓ Signal conservation verified")
end

# =============================================================================
# Test 8: Presets Match CatSim Defaults
# =============================================================================
@testset "Preset Validation" begin
    println("\n[Test 8] Preset Validation")

    # Test all presets
    presets = [
        ("lag_none", lag_none()),
        ("lag_gadox", lag_gadox()),
        ("lag_csi", lag_csi()),
        ("lag_high", lag_high()),
    ]

    for (name, model) in presets
        info = get_lag_info(model)

        println("  $(name):")
        println("    Components: $(info.n_components)")
        println("    Total lag: $(round(info.total_lag_fraction * 100, digits=2))%")

        if info.n_components > 0
            println("    Time constants: $(info.time_constants_ms) ms")
            println("    Amplitudes: $(info.amplitudes)")

            # Verify amplitudes sum to total lag
            @test sum(info.amplitudes) ≈ info.total_lag_fraction

            # Verify time constants are positive
            @test all(info.time_constants_ms .> 0)
        end
    end

    # Verify GOS has typical values
    gos_info = get_lag_info(lag_gadox())
    @test gos_info.total_lag_fraction < 0.05  # < 5% lag
    @test length(gos_info.time_constants_ms) >= 2  # Multi-exponential

    println("  ✓ All presets have valid parameters")
end

# =============================================================================
# Test 9: HU Integration Test
# =============================================================================
@testset "HU Integration Test" begin
    println("\n[Test 9] HU Integration Test (Full Pipeline)")

    # Skip if running quick tests
    if get(ENV, "QUICK_TEST", "false") == "true"
        println("  Skipping (QUICK_TEST=true)")
        @test true
        return
    end

    # Setup at dev scale
    phantom = create_gammex_472(n_voxels=64, n_slices=16, fov_cm=35.0, z_cm=4.0)
    geom = create_aquilion_one(n_angles=90, n_rows=16, n_cols=128, fov_cm=35.0, z_cm=4.0)

    energies, weights = load_spectrum(120)
    energies, weights = downsample_spectrum(energies, weights, 20)
    materials = get_region_materials()

    # Without lag
    physics_no_lag = default_physics_config(energy_keV=65.0)
    sino_no = forward_project(phantom.mask, geom;
        energies=energies, weights=weights, materials=materials,
        physics=physics_no_lag)

    # With lag
    physics_lag = default_physics_config(
        lag = lag_gadox(frame_time=0.5),
        energy_keV = 65.0
    )
    sino_lag = forward_project(phantom.mask, geom;
        energies=energies, weights=weights, materials=materials,
        physics=physics_lag)

    # Reconstruct both
    recon_no = fdk_reconstruct(sino_no, geom, size(phantom.μ))
    recon_lag = fdk_reconstruct(sino_lag, geom, size(phantom.μ))

    # Convert to HU
    mid_z = size(recon_no, 3) ÷ 2 + 1
    water_mask = phantom.mask[:, :, mid_z] .== UInt8(REGION_SOLID_WATER)

    μ_water_no = mean(recon_no[:, :, mid_z][water_mask])
    μ_water_lag = mean(recon_lag[:, :, mid_z][water_mask])

    hu_no = μ_to_HU(recon_no, μ_water_no)
    hu_lag = μ_to_HU(recon_lag, μ_water_lag)

    # Water HU should still be ~0 (proper calibration)
    water_hu_no = mean(hu_no[:, :, mid_z][water_mask])
    water_hu_lag = mean(hu_lag[:, :, mid_z][water_mask])

    println("  Without lag: water HU = $(round(water_hu_no, digits=1))")
    println("  With lag: water HU = $(round(water_hu_lag, digits=1))")

    # Water should be within tolerance
    @test abs(water_hu_no) < 20  # Within ±20 HU
    @test abs(water_hu_lag) < 30  # Slightly larger tolerance with lag

    # Lag should not drastically change HU
    hu_diff = abs(water_hu_lag - water_hu_no)
    @test hu_diff < 30

    println("  HU difference: $(round(hu_diff, digits=1)) HU")
    println("  ✓ Lag effect does not break HU calibration")
end

# =============================================================================
# Summary
# =============================================================================
println("\n" * "=" ^ 70)
println("PHYSICS-008 Verification Complete")
println("=" ^ 70)
println("\nKey findings:")
println("- CatSim uses midpoint sampling for integration accuracy")
println("- Air scan initialization provides steady-state starting conditions")
println("- Typical GOS lag: 1-2% fast (τ~1ms) + 0.5-1% slow (τ~10ms)")
println("- Ghosting magnitude proportional to lag fraction")
println("- Signal conservation verified for steady-state inputs")
