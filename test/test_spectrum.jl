"""
Test suite for Physics/Spectrum.jl

Tests X-ray spectrum generation with known physics properties.
"""

using Test
using BasisSimulator
using Statistics

@testset "Spectrum Generation" begin

    @testset "Basic 120 kVp Spectrum" begin
        spec = generate_spectrum(kVp=120.0, mAs=200.0)

        # Test 1: Energy range
        @test minimum(spec.energies) >= 1.0
        @test maximum(spec.energies) <= 120.0
        @test all(diff(spec.energies) .> 0)  # Monotonic increasing

        # Test 2: Photon counts
        @test all(spec.photons .>= 0)  # Non-negative
        @test sum(spec.photons) > 0    # Non-zero total

        # Test 3: Metadata
        @test spec.kVp == 120.0
        @test spec.mAs == 200.0
        @test spec.anode_material == :tungsten

        # Test 4: Mean energy in expected range
        E_mean = mean_energy(spec)
        @test 50.0 < E_mean < 80.0  # Typical for 120 kVp
    end

    @testset "Characteristic Lines (Tungsten)" begin
        spec = generate_spectrum(kVp=120.0, mAs=200.0)

        # Test 1: K-α₁ peak at 59.32 keV
        kalpha1_idx = argmin(abs.(spec.energies .- 59.32))
        kalpha1_energy = spec.energies[kalpha1_idx]
        @test abs(kalpha1_energy - 59.32) < 1.0  # Within 1 keV

        # Test 2: Peak should be prominent
        kalpha1_photons = spec.photons[kalpha1_idx]
        mean_photons = mean(spec.photons)
        @test kalpha1_photons > 1.5 * mean_photons  # At least 50% above mean

        # Test 3: K-β peak at 67.24 keV
        kbeta_idx = argmin(abs.(spec.energies .- 67.24))
        kbeta_photons = spec.photons[kbeta_idx]
        @test kbeta_photons > mean_photons  # Should be visible
    end

    @testset "No Characteristic Lines Below K-Edge" begin
        spec = generate_spectrum(kVp=60.0, mAs=100.0)  # Below 69.5 keV K-edge

        # Spectrum should be smooth (no sharp peaks)
        # Check that no point is more than 2x local neighborhood average
        for i in 3:(length(spec.photons)-2)
            local_mean = mean(spec.photons[(i-2):(i+2)])
            @test spec.photons[i] < 2.5 * local_mean
        end
    end

    @testset "mAs Scaling" begin
        spec1 = generate_spectrum(kVp=120.0, mAs=100.0)
        spec2 = generate_spectrum(kVp=120.0, mAs=200.0)

        # Test 1: Fluence should double
        fluence1 = total_fluence(spec1)
        fluence2 = total_fluence(spec2)
        ratio = fluence2 / fluence1
        @test isapprox(ratio, 2.0, rtol=0.01)  # Within 1%

        # Test 2: Spectral shape should be identical (just scaled)
        normalized1 = spec1.photons ./ sum(spec1.photons)
        normalized2 = spec2.photons ./ sum(spec2.photons)
        @test isapprox(normalized1, normalized2, rtol=0.001)
    end

    @testset "kVp Effect on Mean Energy" begin
        spec80 = generate_spectrum(kVp=80.0, mAs=200.0)
        spec120 = generate_spectrum(kVp=120.0, mAs=200.0)
        spec140 = generate_spectrum(kVp=140.0, mAs=200.0)

        E80 = mean_energy(spec80)
        E120 = mean_energy(spec120)
        E140 = mean_energy(spec140)

        # Test: Mean energy should increase with kVp
        @test E80 < E120 < E140

        # Test: Mean energy should be roughly 0.5-0.7 × kVp
        @test 0.4 * 80.0 < E80 < 0.8 * 80.0
        @test 0.4 * 120.0 < E120 < 0.8 * 120.0
        @test 0.4 * 140.0 < E140 < 0.8 * 140.0
    end

    @testset "Filtration Effect" begin
        spec_light = generate_spectrum(kVp=120.0, mAs=200.0,
                                      filtration_al_mm=2.0, filtration_cu_mm=0.0)
        spec_heavy = generate_spectrum(kVp=120.0, mAs=200.0,
                                       filtration_al_mm=6.0, filtration_cu_mm=0.3)

        # Test 1: Heavy filtration reduces low-energy photons more
        low_energy_idx = findfirst(e -> e > 30.0, spec_light.energies)
        ratio_low = spec_heavy.photons[low_energy_idx] / spec_light.photons[low_energy_idx]

        high_energy_idx = findfirst(e -> e > 80.0, spec_light.energies)
        ratio_high = spec_heavy.photons[high_energy_idx] / spec_light.photons[high_energy_idx]

        @test ratio_low < ratio_high  # Low energies attenuated more

        # Test 2: Mean energy increases with filtration
        E_light = mean_energy(spec_light)
        E_heavy = mean_energy(spec_heavy)
        @test E_heavy > E_light
    end

    @testset "Anode Angle Effect" begin
        spec_7deg = generate_spectrum(kVp=120.0, mAs=200.0, anode_angle_deg=7.0)
        spec_15deg = generate_spectrum(kVp=120.0, mAs=200.0, anode_angle_deg=15.0)

        # Larger anode angle → less heel effect → higher intensity
        # (simplified model - in reality depends on field position)
        # Just test that changing angle produces different spectrum
        @test !isapprox(total_fluence(spec_7deg), total_fluence(spec_15deg), rtol=0.01)
    end

    @testset "Validation Function" begin
        # Valid spectrum
        spec_valid = generate_spectrum(kVp=120.0, mAs=200.0)
        @test validate_spectrum(spec_valid) == true

        # Invalid: negative photons (manual construction)
        spec_invalid = XRaySpectrum(
            collect(1.0:1.0:120.0),
            vcat(ones(119), [-1.0]),  # Negative value
            120.0, 200.0, :tungsten, 7.0,
            (Al_mm=4.0, Cu_mm=0.1)
        )
        @test_throws ErrorException validate_spectrum(spec_invalid)

        # Invalid: energy exceeds kVp
        @test_throws ErrorException XRaySpectrum(
            collect(1.0:1.0:130.0),  # Exceeds kVp
            ones(130),
            120.0, 200.0, :tungsten, 7.0,
            (Al_mm=4.0, Cu_mm=0.1)
        )
    end

    @testset "Edge Cases" begin
        # Minimum kVp
        spec_min = generate_spectrum(kVp=40.0, mAs=50.0)
        @test maximum(spec_min.energies) <= 40.0
        @test total_fluence(spec_min) > 0

        # Maximum kVp
        spec_max = generate_spectrum(kVp=150.0, mAs=500.0)
        @test maximum(spec_max.energies) <= 150.0
        @test total_fluence(spec_max) > 0

        # High mAs
        spec_high_mas = generate_spectrum(kVp=120.0, mAs=1000.0)
        @test total_fluence(spec_high_mas) > 0

        # Test that all are valid
        @test validate_spectrum(spec_min)
        @test validate_spectrum(spec_max)
        @test validate_spectrum(spec_high_mas)
    end

    @testset "Utility Functions" begin
        spec = generate_spectrum(kVp=120.0, mAs=200.0)

        # mean_energy
        E_mean = mean_energy(spec)
        @test E_mean > 0
        @test E_mean < spec.kVp

        # total_fluence
        fluence = total_fluence(spec)
        @test fluence > 0
        @test fluence == sum(spec.photons)
    end

    @testset "Reproducibility" begin
        spec1 = generate_spectrum(kVp=120.0, mAs=200.0)
        spec2 = generate_spectrum(kVp=120.0, mAs=200.0)

        # Same inputs should give identical outputs
        @test spec1.energies == spec2.energies
        @test spec1.photons == spec2.photons
        @test spec1.kVp == spec2.kVp
        @test spec1.mAs == spec2.mAs
    end

    @testset "Physical Realism" begin
        spec = generate_spectrum(kVp=120.0, mAs=200.0)

        # Test 1: Spectrum should be mostly smooth (except K-lines)
        # Compute second derivative (curvature)
        d2 = diff(diff(spec.photons))
        # Allow some sharp features (K-lines) but most should be smooth
        sharp_points = sum(abs.(d2) .> 10 * median(abs.(d2)))
        @test sharp_points < 0.1 * length(d2)  # <10% sharp points

        # Test 2: Spectrum should decrease toward kVp (fewer high-energy photons)
        upper_third = spec.photons[end-div(length(spec.photons), 3):end]
        lower_third = spec.photons[1:div(length(spec.photons), 3)]
        @test mean(upper_third) < mean(lower_third)
    end
end

@testset "Spectrum Struct Construction" begin
    energies = collect(1.0:1.0:120.0)
    photons = rand(120) .* 1e6

    @testset "Valid Construction" begin
        spec = XRaySpectrum(
            energies, photons, 120.0, 200.0, :tungsten, 7.0,
            (Al_mm=4.0, Cu_mm=0.1)
        )
        @test spec isa XRaySpectrum
    end

    @testset "Invalid: Length Mismatch" begin
        @test_throws ErrorException XRaySpectrum(
            energies, photons[1:100],  # Wrong length
            120.0, 200.0, :tungsten, 7.0,
            (Al_mm=4.0, Cu_mm=0.1)
        )
    end

    @testset "Invalid: Negative kVp" begin
        @test_throws ErrorException XRaySpectrum(
            energies, photons, -120.0, 200.0, :tungsten, 7.0,
            (Al_mm=4.0, Cu_mm=0.1)
        )
    end

    @testset "Invalid: Negative mAs" begin
        @test_throws ErrorException XRaySpectrum(
            energies, photons, 120.0, -200.0, :tungsten, 7.0,
            (Al_mm=4.0, Cu_mm=0.1)
        )
    end
end

# ==============================================================================
# Performance Benchmarks (not run in standard test suite)
# ==============================================================================

function benchmark_spectrum_generation()
    println("\n=== Spectrum Generation Benchmarks ===")

    using BenchmarkTools

    println("120 kVp standard protocol:")
    @btime generate_spectrum(kVp=120.0, mAs=200.0)

    println("\n80 kVp low-dose protocol:")
    @btime generate_spectrum(kVp=80.0, mAs=50.0)

    println("\n140 kVp high-energy protocol:")
    @btime generate_spectrum(kVp=140.0, mAs=400.0)
end

# Uncomment to run benchmarks:
# benchmark_spectrum_generation()
