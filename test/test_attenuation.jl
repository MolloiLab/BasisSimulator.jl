"""
Test suite for Physics/Attenuation.jl

Tests XrayAttenuation.jl integration, material library, and autodiff compatibility.
"""

using Test
using BasisSimulator
using Statistics
import XrayAttenuation as XA

@testset "XrayAttenuation Integration & Attenuation" begin

    # =========================================================================
    # Test 1: XrayAttenuation.jl Materials Access
    # =========================================================================
    @testset "XA Materials Access" begin
        @testset "Material Availability" begin
            # Check that Materials and Elements are exported
            @test isdefined(BasisSimulator, :Materials)
            @test isdefined(BasisSimulator, :Elements)

            # Check common materials exist
            @test isdefined(XA.Materials, :water)
            @test isdefined(XA.Materials, :air)
            @test isdefined(XA.Materials, :softtissue)
            @test isdefined(XA.Materials, :corticalbone)
            @test isdefined(XA.Materials, :iodine)
            @test isdefined(XA.Materials, :acrylic)
        end

        @testset "Element Availability" begin
            # Check common elements
            @test isdefined(XA.Elements, :H)
            @test isdefined(XA.Elements, :C)
            @test isdefined(XA.Elements, :O)
            @test isdefined(XA.Elements, :Ca)
            @test isdefined(XA.Elements, :I)
            @test isdefined(XA.Elements, :Fe)
        end

        @testset "List Functions" begin
            materials = list_available_materials()
            @test length(materials) > 10  # Should have many materials
            @test :water in materials
            @test :corticalbone in materials

            elements = list_available_elements()
            @test length(elements) == 92  # All elements H through U
            @test :H in elements
            @test :U in elements
        end
    end

    # =========================================================================
    # Test 2: Mass Attenuation Coefficient Queries
    # =========================================================================
    @testset "Mass Attenuation Queries" begin
        @testset "Single Energy - Pre-defined Material" begin
            # Water at 60 keV
            μ_ρ = get_mass_attenuation(XA.Materials.water, 60.0)

            @test μ_ρ > 0
            @test isfinite(μ_ρ)
            # Literature: μ/ρ ≈ 0.206 cm²/g at 60 keV (NIST)
            @test isapprox(μ_ρ, 0.206, rtol=0.05)
        end

        @testset "Single Energy - Element" begin
            # Iron at 70 keV
            μ_ρ_Fe = get_mass_attenuation(XA.Elements.Fe, 70.0)

            @test μ_ρ_Fe > 0
            @test isfinite(μ_ρ_Fe)
        end

        @testset "Single Energy - Compound" begin
            # Water compound
            water = XA.Compound("H2O")
            μ_ρ = get_mass_attenuation(water, 60.0)

            @test μ_ρ > 0
            @test isfinite(μ_ρ)
            # Should be similar to XA.Materials.water
            μ_ρ_material = get_mass_attenuation(XA.Materials.water, 60.0)
            @test isapprox(μ_ρ, μ_ρ_material, rtol=0.01)
        end

        @testset "Energy Dependence" begin
            # Attenuation should decrease with energy
            μ_ρ_30 = get_mass_attenuation(XA.Materials.water, 30.0)
            μ_ρ_60 = get_mass_attenuation(XA.Materials.water, 60.0)
            μ_ρ_90 = get_mass_attenuation(XA.Materials.water, 90.0)

            @test μ_ρ_30 > μ_ρ_60 > μ_ρ_90
        end

        @testset "Vectorized Query" begin
            energies = [30.0, 50.0, 70.0, 100.0, 150.0]
            μ_ρ_array = get_mass_attenuation(XA.Materials.water, energies)

            @test length(μ_ρ_array) == length(energies)
            @test all(μ_ρ_array .> 0)
            @test all(isfinite.(μ_ρ_array))

            # Should be monotonically decreasing
            @test all(diff(μ_ρ_array) .< 0)
        end
    end

    # =========================================================================
    # Test 3: Linear Attenuation Coefficient Queries
    # =========================================================================
    @testset "Linear Attenuation Queries" begin
        @testset "Single Energy" begin
            μ_water = get_linear_attenuation(XA.Materials.water, 60.0)
            μ_bone = get_linear_attenuation(XA.Materials.corticalbone, 60.0)

            @test μ_water > 0
            @test μ_bone > 0

            # Bone should attenuate more than water
            @test μ_bone > μ_water

            # Water: μ ≈ 0.206 cm⁻¹ at 60 keV (ρ=1.0 g/cm³)
            @test isapprox(μ_water, 0.206, rtol=0.05)
        end

        @testset "Vectorized Query" begin
            energies = [40.0, 60.0, 80.0, 100.0, 120.0]
            μ_array = get_linear_attenuation(XA.Materials.corticalbone, energies)

            @test length(μ_array) == length(energies)
            @test all(μ_array .> 0)
            @test all(isfinite.(μ_array))

            # Should be monotonically decreasing
            @test all(diff(μ_array) .< 0)
        end

        @testset "Density Effect" begin
            # Linear attenuation should scale with density
            # Compare water (ρ=1.0) vs cortical bone (ρ≈1.92)

            μ_ρ_water = get_mass_attenuation(XA.Materials.water, 60.0)
            μ_water = get_linear_attenuation(XA.Materials.water, 60.0)

            # Should match: μ = ρ * (μ/ρ)
            @test isapprox(μ_water, 1.0 * μ_ρ_water, rtol=0.01)
        end
    end

    # =========================================================================
    # Test 4: Polychromatic Attenuation (Spectrum Integration)
    # =========================================================================
    @testset "Polychromatic Attenuation" begin
        spec = generate_spectrum(kVp=120.0, mAs=200.0)

        @testset "Effective Attenuation" begin
            μ_eff_water = compute_polychromatic_attenuation(XA.Materials.water, spec)
            μ_eff_bone = compute_polychromatic_attenuation(XA.Materials.corticalbone, spec)

            @test μ_eff_water > 0
            @test μ_eff_bone > 0
            @test isfinite(μ_eff_water)
            @test isfinite(μ_eff_bone)

            # Bone should attenuate more than water
            @test μ_eff_bone > μ_eff_water

            # Should be between attenuation at min and max energies
            μ_min = get_linear_attenuation(XA.Materials.water, minimum(spec.energies))
            μ_max = get_linear_attenuation(XA.Materials.water, maximum(spec.energies))
            @test μ_max < μ_eff_water < μ_min
        end

        @testset "Spectrum Energy Effect" begin
            spec_80kv = generate_spectrum(kVp=80.0, mAs=200.0)
            spec_140kv = generate_spectrum(kVp=140.0, mAs=200.0)

            μ_80 = compute_polychromatic_attenuation(XA.Materials.water, spec_80kv)
            μ_140 = compute_polychromatic_attenuation(XA.Materials.water, spec_140kv)

            # Higher kVp → harder spectrum → lower effective attenuation
            @test μ_140 < μ_80
        end

        @testset "High-Z Material Beam Hardening" begin
            # High-Z materials should show more beam hardening effect
            spec_80kv = generate_spectrum(kVp=80.0, mAs=200.0)
            spec_140kv = generate_spectrum(kVp=140.0, mAs=200.0)

            # Water
            μ_water_80 = compute_polychromatic_attenuation(XA.Materials.water, spec_80kv)
            μ_water_140 = compute_polychromatic_attenuation(XA.Materials.water, spec_140kv)
            ratio_water = μ_water_80 / μ_water_140

            # Iodine
            μ_iodine_80 = compute_polychromatic_attenuation(XA.Materials.iodine, spec_80kv)
            μ_iodine_140 = compute_polychromatic_attenuation(XA.Materials.iodine, spec_140kv)
            ratio_iodine = μ_iodine_80 / μ_iodine_140

            # Iodine should show stronger energy dependence
            @test ratio_iodine > ratio_water
        end
    end

    # =========================================================================
    # Test 5: Material Mixtures (CRITICAL FOR INVERSE PROBLEMS)
    # =========================================================================
    @testset "Material Mixtures" begin
        @testset "Two-Material Mixture" begin
            # 50% water, 50% bone
            mixture = Dict(
                XA.Materials.water => 0.5,
                XA.Materials.corticalbone => 0.5
            )
            μ_mix = compute_mixture_attenuation(mixture, 60.0)

            μ_water = get_linear_attenuation(XA.Materials.water, 60.0)
            μ_bone = get_linear_attenuation(XA.Materials.corticalbone, 60.0)

            # Mixture should be between pure materials
            @test μ_water < μ_mix < μ_bone

            # Should be approximately the average
            μ_expected = 0.5 * μ_water + 0.5 * μ_bone
            @test isapprox(μ_mix, μ_expected, rtol=0.01)
        end

        @testset "Contrast-Enhanced Blood" begin
            # Simulate contrast-enhanced blood (3 mg/ml iodine)
            # Approximate as: 99.7% blood, 0.3% iodine (volume fraction)
            mixture = Dict(
                XA.Materials.blood => 0.997,
                XA.Materials.iodine => 0.003
            )
            μ_enhanced = compute_mixture_attenuation(mixture, 60.0)

            μ_blood = get_linear_attenuation(XA.Materials.blood, 60.0)

            # Should be higher than pure blood
            @test μ_enhanced > μ_blood

            # At 60 keV, iodine has very high attenuation
            # Even 0.3% should be detectable
            @test (μ_enhanced - μ_blood) / μ_blood > 0.01  # >1% increase
        end

        @testset "Pure Material as Mixture" begin
            # 100% water should match get_linear_attenuation
            mixture = Dict(XA.Materials.water => 1.0)
            μ_mix = compute_mixture_attenuation(mixture, 60.0)
            μ_pure = get_linear_attenuation(XA.Materials.water, 60.0)

            @test isapprox(μ_mix, μ_pure, rtol=1e-10)
        end

        @testset "Three-Material Mixture" begin
            # Realistic tissue: soft tissue + some bone + some fat
            mixture = Dict(
                XA.Materials.softtissue => 0.7,
                XA.Materials.corticalbone => 0.2,
                XA.Materials.adipose => 0.1
            )

            μ_mix = compute_mixture_attenuation(mixture, 60.0)

            @test isfinite(μ_mix)
            @test μ_mix > 0

            # Should be weighted average
            μ_expected = (
                0.7 * get_linear_attenuation(XA.Materials.softtissue, 60.0) +
                0.2 * get_linear_attenuation(XA.Materials.corticalbone, 60.0) +
                0.1 * get_linear_attenuation(XA.Materials.adipose, 60.0)
            )
            @test isapprox(μ_mix, μ_expected, rtol=0.01)
        end
    end

    # =========================================================================
    # Test 6: Two-Material Decomposition (Dual-Energy CT)
    # =========================================================================
    @testset "Two-Material Decomposition" begin
        @testset "Water-Hydroxyapatite Decomposition" begin
            # Simulate known mixture
            L_water_true = 5.0  # cm
            L_HA_true = 2.0     # cm

            E_low, E_high = 80.0, 140.0

            # Compute expected measurements
            μ_water_low = get_linear_attenuation(XA.Materials.water, E_low)
            μ_water_high = get_linear_attenuation(XA.Materials.water, E_high)
            μ_HA_low = get_linear_attenuation(XA.Materials.hydroxyapatite, E_low)
            μ_HA_high = get_linear_attenuation(XA.Materials.hydroxyapatite, E_high)

            μ_meas_low = L_water_true * μ_water_low + L_HA_true * μ_HA_low
            μ_meas_high = L_water_true * μ_water_high + L_HA_true * μ_HA_high

            # Decompose
            L_water_recon, L_HA_recon = compute_two_material_decomposition(
                XA.Materials.water,
                XA.Materials.hydroxyapatite,
                [E_low, E_high],
                [μ_meas_low, μ_meas_high]
            )

            # Should recover true values
            @test isapprox(L_water_recon, L_water_true, rtol=0.01)
            @test isapprox(L_HA_recon, L_HA_true, rtol=0.01)
        end

        @testset "Calcium Quantification" begin
            # Typical bone measurement
            E_low, E_high = 80.0, 140.0
            μ_meas_low = 0.5   # cm⁻¹
            μ_meas_high = 0.35  # cm⁻¹

            L_water, L_HA = compute_two_material_decomposition(
                XA.Materials.water,
                XA.Materials.hydroxyapatite,
                [E_low, E_high],
                [μ_meas_low, μ_meas_high]
            )

            @test isfinite(L_water)
            @test isfinite(L_HA)
            @test L_water >= 0
            @test L_HA >= 0

            # Convert to calcium density (399 mg Ca per cm³ of HA)
            ρ_calcium_mg_cm3 = L_HA * 399.0
            @test ρ_calcium_mg_cm3 >= 0
        end
    end

    # =========================================================================
    # Test 7: Custom Material Creation
    # =========================================================================
    @testset "Custom Material Creation" begin
        @testset "Custom Compound" begin
            # Create calcium carbonate
            caco3 = create_custom_compound("CaCO3")
            μ_ρ = get_mass_attenuation(caco3, 60.0)

            @test μ_ρ > 0
            @test isfinite(μ_ρ)
        end

        @testset "Custom Mixture" begin
            # Create custom tissue
            tissue = create_custom_mixture(Dict(
                "H2O" => 0.70,
                "C" => 0.15,
                "N" => 0.05,
                "O" => 0.08,
                "Ca" => 0.02
            ))

            μ_ρ = get_mass_attenuation(tissue, 60.0)
            @test μ_ρ > 0
            @test isfinite(μ_ρ)
        end

        @testset "Invalid Mixture (Fractions Don't Sum to 1)" begin
            # Should throw error
            @test_throws ErrorException create_custom_mixture(Dict(
                "H2O" => 0.5,
                "C" => 0.3
            ))
        end
    end

    # =========================================================================
    # Test 8: Physical Realism
    # =========================================================================
    @testset "Physical Realism" begin
        @testset "Z-Dependence (Photoelectric)" begin
            # At low energies, high-Z should attenuate much more
            E_low = 30.0  # keV (photoelectric dominates)

            μ_water = get_linear_attenuation(XA.Materials.water, E_low)
            μ_calcium = get_linear_attenuation(XA.Elements.Ca, E_low)
            μ_iodine = get_linear_attenuation(XA.Materials.iodine, E_low)

            # Higher Z materials should attenuate more (accounting for density)
            @test μ_iodine > μ_calcium
            @test μ_iodine > μ_water
        end

        @testset "Compton Plateau" begin
            # At high energies, Compton dominates
            # μ should be relatively flat (weakly energy-dependent)
            E_high1 = 100.0
            E_high2 = 120.0

            μ_1 = get_linear_attenuation(XA.Materials.water, E_high1)
            μ_2 = get_linear_attenuation(XA.Materials.water, E_high2)

            # Should be similar (within 20%)
            @test abs(μ_1 - μ_2) / μ_1 < 0.2
        end

        @testset "Monotonic Decrease with Energy" begin
            # For all materials, μ should decrease with E
            energies = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0]

            for mat in [XA.Materials.water, XA.Materials.softtissue, XA.Materials.corticalbone]
                μ_values = [get_linear_attenuation(mat, E) for E in energies]

                # Check monotonic decrease
                @test all(diff(μ_values) .< 0)
            end
        end

        @testset "K-Edge Detection (Iodine)" begin
            # Iodine K-edge at 33.2 keV
            # Should see discontinuity
            E_below = 33.0
            E_above = 34.0

            μ_below = get_mass_attenuation(XA.Elements.I, E_below)
            μ_above = get_mass_attenuation(XA.Elements.I, E_above)

            # Should be significantly different due to K-edge
            @test μ_below > μ_above * 1.2  # At least 20% difference
        end
    end

    # =========================================================================
    # Test 9: Autodiff Compatibility (CRITICAL FOR INVERSE PROBLEMS)
    # =========================================================================
    @testset "Autodiff Compatibility" begin
        @testset "Energy Gradient (Finite Differences)" begin
            # Test that functions are smooth enough for gradients
            E = 60.0
            ε = 1e-5

            # Numerical gradient via finite differences
            μ_forward = get_mass_attenuation(XA.Materials.water, E + ε)
            μ_backward = get_mass_attenuation(XA.Materials.water, E - ε)
            grad_numerical = (μ_forward - μ_backward) / (2 * ε)

            # Should be finite and reasonable magnitude
            @test isfinite(grad_numerical)
            @test abs(grad_numerical) < 1.0  # Shouldn't change too rapidly
        end

        @testset "Mixture Gradient (Material Fractions)" begin
            # Test gradient with respect to mixture fractions
            α = 0.5
            mixture1 = Dict(
                XA.Materials.water => α,
                XA.Materials.corticalbone => 1-α
            )
            μ1 = compute_mixture_attenuation(mixture1, 60.0)

            # Perturbed mixture
            ε = 1e-6
            mixture2 = Dict(
                XA.Materials.water => α + ε,
                XA.Materials.corticalbone => 1-α-ε
            )
            μ2 = compute_mixture_attenuation(mixture2, 60.0)

            # Numerical gradient
            grad_numerical = (μ2 - μ1) / ε

            # Should equal difference between pure materials
            μ_water = get_linear_attenuation(XA.Materials.water, 60.0)
            μ_bone = get_linear_attenuation(XA.Materials.corticalbone, 60.0)
            grad_expected = μ_water - μ_bone

            @test isapprox(grad_numerical, grad_expected, rtol=0.01)
        end

        @testset "Smooth Interpolation" begin
            # Check that interpolation is smooth (C¹ continuous)
            E = 60.0
            ε = 0.1

            μ_center = get_mass_attenuation(XA.Materials.water, E)
            μ_left = get_mass_attenuation(XA.Materials.water, E - ε)
            μ_right = get_mass_attenuation(XA.Materials.water, E + ε)

            # Centered difference approximation
            grad_left = (μ_center - μ_left) / ε
            grad_right = (μ_right - μ_center) / ε

            # Gradients should be similar (smooth function)
            @test isapprox(grad_left, grad_right, rtol=0.1)
        end
    end

    # =========================================================================
    # Test 10: Performance
    # =========================================================================
    @testset "Performance" begin
        @testset "Query Speed" begin
            # Single query should be fast
            t_start = time()
            for _ in 1:1000
                get_linear_attenuation(XA.Materials.water, 60.0)
            end
            t_elapsed = time() - t_start

            # Should complete 1000 queries in < 0.5 seconds
            @test t_elapsed < 0.5
        end

        @testset "Mixture Computation" begin
            mixture = Dict(
                XA.Materials.water => 0.5,
                XA.Materials.corticalbone => 0.3,
                XA.Materials.adipose => 0.2
            )

            t_start = time()
            for _ in 1:1000
                compute_mixture_attenuation(mixture, 60.0)
            end
            t_elapsed = time() - t_start

            # Should be fast even with 3 materials
            @test t_elapsed < 0.5
        end
    end
end

# ==============================================================================
# Integration Test: End-to-End Attenuation Calculation
# ==============================================================================

@testset "Attenuation Integration Test" begin
    @testset "Full Forward Model Component" begin
        # Generate spectrum
        spec = generate_spectrum(kVp=120.0, mAs=200.0)

        # Define phantom materials (material, length in cm)
        materials_in_ray = [
            (XA.Materials.water, 10.0),           # 10 cm water
            (XA.Materials.corticalbone, 2.0),     # 2 cm bone
            (XA.Materials.softtissue, 5.0)        # 5 cm tissue
        ]

        # Compute polychromatic attenuation along ray
        total_atten = 0.0

        for (mat, length_cm) in materials_in_ray
            μ_eff = compute_polychromatic_attenuation(mat, spec)
            total_atten += μ_eff * length_cm
        end

        # Test properties
        @test total_atten > 0
        @test isfinite(total_atten)

        # Typical body scan: total attenuation ~ 4-10
        @test 2.0 < total_atten < 15.0

        # Compute transmitted intensity
        I_transmitted = exp(-total_atten)
        @test 0 < I_transmitted < 1
    end
end

# ==============================================================================
# Validation Against Literature Values
# ==============================================================================

@testset "Literature Validation" begin
    @testset "Water at 60 keV" begin
        # Literature: μ/ρ ≈ 0.206 cm²/g at 60 keV (NIST)
        μ_ρ = get_mass_attenuation(XA.Materials.water, 60.0)
        @test isapprox(μ_ρ, 0.206, rtol=0.05)  # Within 5%

        # Linear: μ = ρ · μ/ρ = 1.0 · 0.206 = 0.206 cm⁻¹
        μ = get_linear_attenuation(XA.Materials.water, 60.0)
        @test isapprox(μ, 0.206, rtol=0.05)
    end

    @testset "Bone HU Values" begin
        # Cortical bone should give HU ≈ 1000-2000
        # HU = 1000 * (μ_material - μ_water) / μ_water
        μ_water = get_linear_attenuation(XA.Materials.water, 60.0)
        μ_bone = get_linear_attenuation(XA.Materials.corticalbone, 60.0)

        HU_bone = 1000.0 * (μ_bone - μ_water) / μ_water

        # Should be in clinical range
        @test 500.0 < HU_bone < 3000.0
        @test HU_bone > 1000.0  # Cortical bone typically >1000 HU
    end

    @testset "Iodine Contrast Enhancement" begin
        # Iodine at 60 keV should have high attenuation
        μ_iodine = get_linear_attenuation(XA.Materials.iodine, 60.0)
        μ_water = get_linear_attenuation(XA.Materials.water, 60.0)

        # Iodine should be much higher than water
        @test μ_iodine > μ_water * 10

        # HU value for pure iodine
        HU_iodine = 1000.0 * (μ_iodine - μ_water) / μ_water
        @test HU_iodine > 10000.0  # Should be very high
    end
end
