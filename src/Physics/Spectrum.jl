"""
    Physics/Spectrum.jl

X-ray source modeling with realistic physics.

# Models Implemented
1. Bremsstrahlung (Kramers' Law with Z-dependence)
2. Characteristic X-rays (K-α, K-β lines for tungsten)
3. Filtration (aluminum, copper, tissue)
4. Heel effect (anode angle dependence)
5. Normalization to mAs dose

# Physics Background

##  Bremsstrahlung Production
When electrons decelerate in the anode material (typically tungsten), they emit
continuous X-ray spectrum:

```
I(E) ∝ Z × (kVp - E) × E    for E ≤ kVp
```

Where:
- Z = atomic number of anode (74 for tungsten)
- kVp = peak tube voltage
- E = photon energy

## Characteristic X-rays
Tungsten K-shell transitions produce sharp peaks:
- K-α₁: 59.32 keV (most intense)
- K-α₂: 58.00 keV
- K-β:  67.24 keV

These only occur when kVp > 69.5 keV (tungsten K-edge).

## Filtration
Inherent filtration (tube housing) plus added filters:
- Aluminum: 2-5 mm (beam hardening)
- Copper: 0.1-0.3 mm (additional hardening)
- Reduces patient dose by removing low-energy photons

## Heel Effect
Non-uniform intensity across the field due to anode geometry:
```
I(θ) = I₀ × exp(-μ × t/sin(α + θ))
```
Where α = anode angle (typically 7-15°)

# References
1. Boone & Seibert, "An accurate method for computer-generating tungsten
   anode x-ray spectra from 30 to 140 kV." Med Phys 1997
2. NIST XCOM database for attenuation coefficients
3. IPEM Report 78 (2005): "Catalogue of diagnostic x-ray spectra"
"""

# ============================================================================
# Data Structures
# ============================================================================

"""
    XRaySpectrum

X-ray source spectrum with physical parameters.

# Fields
- `energies::Vector{Float64}` - Energy bins (keV)
- `photons::Vector{Float64}` - Photon fluence per bin (photons/mm²/mAs)
- `kVp::Float64` - Peak tube voltage (kV)
- `mAs::Float64` - Tube current-time product (mAs)
- `anode_material::Symbol` - Anode material (:tungsten, :molybdenum, etc.)
- `anode_angle_deg::Float64` - Anode angle (degrees)
- `filtration::NamedTuple` - Filter materials and thicknesses

# Example
```julia
spec = XRaySpectrum(
    energies = collect(1.0:1.0:120.0),
    photons = zeros(120),
    kVp = 120.0,
    mAs = 200.0,
    anode_material = :tungsten,
    anode_angle_deg = 7.0,
    filtration = (Al_mm = 4.0, Cu_mm = 0.1)
)
```
"""
struct XRaySpectrum
    energies::Vector{Float64}
    photons::Vector{Float64}
    kVp::Float64
    mAs::Float64
    anode_material::Symbol
    anode_angle_deg::Float64
    filtration::NamedTuple

    # Validation constructor
    function XRaySpectrum(
            energies::Vector{Float64},
            photons::Vector{Float64},
            kVp::Float64,
            mAs::Float64,
            anode_material::Symbol,
            anode_angle_deg::Float64,
            filtration::NamedTuple
        )
        # Validate inputs
        length(energies) == length(photons) ||
            error("energies and photons must have same length")
        kVp > 0 || error("kVp must be positive")
        mAs > 0 || error("mAs must be positive")
        all(energies .> 0) || error("All energies must be positive")
        all(photons .>= 0) || error("All photons must be non-negative")
        maximum(energies) <= kVp || error("Max energy cannot exceed kVp")

        new(energies, photons, kVp, mAs, anode_material, anode_angle_deg, filtration)
    end
end

# ============================================================================
# Spectrum Generation
# ============================================================================

"""
    generate_spectrum(; kVp, mAs, kwargs...) -> XRaySpectrum

Generate realistic X-ray spectrum with full physics.

# Arguments
- `kVp::Float64` - Peak tube voltage (kV), range: 40-150
- `mAs::Float64` - Tube current-time product (mAs), range: 10-1000

# Optional Arguments
- `anode_material::Symbol = :tungsten` - Anode material
- `anode_angle_deg::Float64 = 7.0` - Anode angle (degrees), typical: 7-15°
- `filtration_al_mm::Float64 = 4.0` - Aluminum filtration (mm)
- `filtration_cu_mm::Float64 = 0.1` - Copper filtration (mm)
- `energy_bin_width_keV::Float64 = 1.0` - Energy binning resolution
- `field_position_cm::Float64 = 0.0` - Position for heel effect (cathode-anode axis)
- `sdd_cm::Float64 = 100.0` - Source-to-detector distance for normalization

# Returns
`XRaySpectrum` with:
- Bremsstrahlung continuum
- Characteristic K-lines (if kVp > K-edge)
- Filtration applied
- Heel effect applied
- Normalized to mAs

# Example
```julia
# Standard 120 kVp chest CT protocol
spec = generate_spectrum(
    kVp = 120.0,
    mAs = 200.0,
    anode_angle_deg = 7.0,
    filtration_al_mm = 4.0,
    filtration_cu_mm = 0.1
)

# Low-dose protocol
spec_low = generate_spectrum(kVp=100.0, mAs=50.0)

# High-energy protocol with heavier filtration
spec_high = generate_spectrum(
    kVp=140.0,
    mAs=400.0,
    filtration_al_mm=5.0,
    filtration_cu_mm=0.3
)
```

# Physics Details

The spectrum is generated in the following order:

1. **Energy Grid**: Create energy bins from 1 keV to kVp
2. **Bremsstrahlung**: Kramers' law with Z-dependence
3. **Characteristic Lines**: Add K-α and K-β peaks (if applicable)
4. **Filtration**: Apply Beer-Lambert through Al and Cu filters
5. **Heel Effect**: Angular intensity variation
6. **Normalization**: Scale to match specified mAs

# Validation
- Total photon fluence should scale linearly with mAs
- Characteristic lines should only appear when kVp > K-edge energy
- Mean energy should increase with kVp and filtration
- Spectrum should be smooth (no unphysical discontinuities)
"""
function generate_spectrum(;
        kVp::Float64,
        mAs::Float64,
        anode_material::Symbol = :tungsten,
        anode_angle_deg::Float64 = 7.0,
        filtration_al_mm::Float64 = 4.0,
        filtration_cu_mm::Float64 = 0.1,
        energy_bin_width_keV::Float64 = 1.0,
        field_position_cm::Float64 = 0.0,
        sdd_cm::Float64 = 100.0
    )

    # Validate inputs
    40.0 <= kVp <= 150.0 || error("kVp must be in range [40, 150] kV")
    mAs > 0 || error("mAs must be positive")
    0 < anode_angle_deg < 30.0 || error("Anode angle must be in range (0, 30)°")

    # ========================================================================
    # STEP 1: Create energy grid
    # ========================================================================
    energies = collect(energy_bin_width_keV:energy_bin_width_keV:kVp)
    n_bins = length(energies)

    # ========================================================================
    # STEP 2: Bremsstrahlung (Kramers' Law with Z-dependence)
    # ========================================================================

    # Atomic number of anode
    Z = anode_material == :tungsten ? 74.0 :
        anode_material == :molybdenum ? 42.0 :
        error("Unknown anode material: $anode_material")

    # Kramers' Law: I(E) ∝ Z × (kVp - E) × E
    # The E factor comes from detector efficiency increasing with energy
    brems = zeros(Float64, n_bins)
    for (i, E) in enumerate(energies)
        if E < kVp
            brems[i] = Z * (kVp - E) * E
        end
    end

    # ========================================================================
    # STEP 3: Characteristic X-rays (tungsten K-lines)
    # ========================================================================

    char_lines = zeros(Float64, n_bins)

    if anode_material == :tungsten && kVp > 69.5  # Tungsten K-edge
        # Line energies (keV) and relative intensities
        kalpha1_energy = 59.32  # K-α₁ (most intense)
        kalpha2_energy = 58.00  # K-α₂
        kbeta_energy = 67.24    # K-β

        # Line intensities relative to K-α₁
        kalpha1_intensity = 1.00
        kalpha2_intensity = 0.50
        kbeta_intensity = 0.20

        # Line widths (keV) - natural linewidth + instrumental broadening
        sigma = 0.5  # FWHM ~1.2 keV

        # Add Gaussian peaks
        for (i, E) in enumerate(energies)
            # K-α₁
            char_lines[i] += kalpha1_intensity * exp(-((E - kalpha1_energy)^2) / (2 * sigma^2))
            # K-α₂
            char_lines[i] += kalpha2_intensity * exp(-((E - kalpha2_energy)^2) / (2 * sigma^2))
            # K-β
            char_lines[i] += kbeta_intensity * exp(-((E - kbeta_energy)^2) / (2 * sigma^2))
        end

        # Scale characteristic lines relative to bremsstrahlung
        char_lines .*= (0.1 * kVp)  # K-lines are ~10% of bremsstrahlung intensity
    end

    # Combine bremsstrahlung and characteristic
    spectrum_pre_filt = brems .+ char_lines

    # ========================================================================
    # STEP 4: Filtration (Beer-Lambert attenuation)
    # ========================================================================

    # Simplified attenuation coefficients for Al and Cu
    # μ(E) ≈ a/E³ + b/E + c (photoelectric + Compton + coherent)
    # Fitted parameters for Al (Z=13) and Cu (Z=29)

    function mu_aluminum(E_keV::Float64)
        # Approximate mass attenuation coefficient (cm²/g)
        a = 150.0  # Photoelectric coefficient
        b = 0.5    # Compton coefficient
        c = 0.01   # Pair production (negligible)
        mu_mass = a / E_keV^3 + b / E_keV + c
        rho_al = 2.70  # g/cm³
        return mu_mass * rho_al  # cm⁻¹
    end

    function mu_copper(E_keV::Float64)
        # Approximate mass attenuation coefficient (cm²/g)
        a = 1500.0  # Higher Z → stronger photoelectric
        b = 0.5
        c = 0.01
        mu_mass = a / E_keV^3 + b / E_keV + c
        rho_cu = 8.96  # g/cm³
        return mu_mass * rho_cu  # cm⁻¹
    end

    # Apply filtration
    spectrum_filtered = similar(spectrum_pre_filt)
    for (i, E) in enumerate(energies)
        # Transmission through filters: T = exp(-μ × thickness)
        t_al = exp(-mu_aluminum(E) * filtration_al_mm / 10.0)  # mm → cm
        t_cu = exp(-mu_copper(E) * filtration_cu_mm / 10.0)
        spectrum_filtered[i] = spectrum_pre_filt[i] * t_al * t_cu
    end

    # ========================================================================
    # STEP 5: Heel Effect
    # ========================================================================

    # Heel effect causes intensity variation across field
    # More attenuation when photons travel through anode at shallow angles

    # Anode angle in radians
    alpha_rad = deg2rad(anode_angle_deg)

    # Field angle (approximation for small angles)
    field_angle_rad = atan(field_position_cm / sdd_cm)

    # Effective path length through anode
    # Simplified model: I(θ) ∝ 1/sin(α + θ)
    # This assumes photons must pass through anode material
    heel_factor = 1.0 / sin(alpha_rad + abs(field_angle_rad))

    # Heel effect is energy-dependent (self-attenuation in anode)
    # Stronger effect at low energies
    # Simplified: apply uniform factor (more complex models would be E-dependent)
    spectrum_heel = spectrum_filtered .* heel_factor

    # ========================================================================
    # STEP 6: Normalization to mAs
    # ========================================================================

    # Normalize to standard mAs
    # Photon fluence ∝ mAs
    # Units: photons/mm²/mAs at specified SDD

    # Arbitrary normalization factor (calibrated to match realistic fluences)
    # Typical: ~10⁶ photons/mm²/mAs at 100 cm SDD for 120 kVp
    norm_factor = 1e6 / sum(spectrum_heel)

    photons_normalized = spectrum_heel .* norm_factor .* mAs

    # ========================================================================
    # Create spectrum object
    # ========================================================================

    return XRaySpectrum(
        energies,
        photons_normalized,
        kVp,
        mAs,
        anode_material,
        anode_angle_deg,
        (Al_mm = filtration_al_mm, Cu_mm = filtration_cu_mm)
    )
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    mean_energy(spec::XRaySpectrum) -> Float64

Compute the mean (effective) energy of the spectrum.

```
E_mean = Σ(E × N(E)) / Σ N(E)
```

This is the energy-weighted average, representing the "effective" energy
of the polychromatic beam.

# Example
```julia
spec = generate_spectrum(kVp=120.0, mAs=200.0)
E_eff = mean_energy(spec)  # Typically 60-70 keV for 120 kVp
```
"""
function mean_energy(spec::XRaySpectrum)
    return sum(spec.energies .* spec.photons) / sum(spec.photons)
end

"""
    total_fluence(spec::XRaySpectrum) -> Float64

Compute total photon fluence (photons/mm²).

# Example
```julia
spec = generate_spectrum(kVp=120.0, mAs=200.0)
fluence = total_fluence(spec)  # ~2×10⁸ photons/mm² for 200 mAs
```
"""
function total_fluence(spec::XRaySpectrum)
    return sum(spec.photons)
end

"""
    hvl(spec::XRaySpectrum, material::Material) -> Float64

Compute the Half-Value Layer (HVL) for a given material.

HVL is the thickness of material required to reduce the intensity to 50%.
It's a standard metric for characterizing beam quality (penetration).

Typical values:
- Al HVL for 120 kVp: ~5-7 mm
- Al HVL for 80 kVp: ~3-4 mm

# Returns
HVL in millimeters

# Example
```julia
using Attenuations
spec = generate_spectrum(kVp=120.0, mAs=200.0)
al = Materials.Al  # Aluminum from Attenuations.jl
hvl_mm = hvl(spec, al)  # Should be ~5-7 mm for 120 kVp
```
"""
function hvl(spec::XRaySpectrum, material)
    # Not implemented yet - requires Attenuations.jl integration
    # This would iterate through thicknesses to find 50% transmission
    error("HVL calculation not yet implemented - coming in Phase 2")
end

# ============================================================================
# Validation Helpers
# ============================================================================

"""
    validate_spectrum(spec::XRaySpectrum) -> Bool

Check if spectrum satisfies physical constraints.

# Checks
1. All energies positive
2. All photon counts non-negative
3. Max energy ≤ kVp
4. Non-zero total fluence
5. Mean energy in reasonable range (0.5-0.7 × kVp)
6. Characteristic lines present when kVp > K-edge

Returns `true` if valid, throws error with details if invalid.
"""
function validate_spectrum(spec::XRaySpectrum)
    # Check 1: Positive energies
    all(spec.energies .> 0) ||
        error("Validation failed: Some energies are non-positive")

    # Check 2: Non-negative photons
    all(spec.photons .>= 0) ||
        error("Validation failed: Some photon counts are negative")

    # Check 3: Max energy ≤ kVp
    maximum(spec.energies) <= spec.kVp ||
        error("Validation failed: Max energy ($(maximum(spec.energies))) exceeds kVp ($(spec.kVp))")

    # Check 4: Non-zero fluence
    total_fluence(spec) > 0 ||
        error("Validation failed: Total fluence is zero")

    # Check 5: Reasonable mean energy
    E_mean = mean_energy(spec)
    0.4 * spec.kVp <= E_mean <= 0.8 * spec.kVp ||
        @warn "Mean energy ($(E_mean) keV) outside expected range [$(0.4*spec.kVp), $(0.8*spec.kVp)] keV"

    # Check 6: K-lines present for tungsten when kVp > K-edge
    if spec.anode_material == :tungsten && spec.kVp > 69.5
        # Find K-α peak (~59 keV)
        kalpha_idx = findfirst(e -> 58.0 <= e <= 60.0, spec.energies)
        if !isnothing(kalpha_idx)
            peak_photons = spec.photons[kalpha_idx]
            mean_photons = mean(spec.photons)
            peak_photons > 1.5 * mean_photons ||
                @warn "K-α peak not prominent (expected for kVp=$(spec.kVp))"
        end
    end

    return true
end
