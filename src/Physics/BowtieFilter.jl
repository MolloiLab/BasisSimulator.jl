"""
    Physics/BowtieFilter.jl

Bowtie filter modeling for CT dose modulation.

The bowtie filter is a shaped aluminum filter that varies in thickness across
the fan angle, reducing peripheral dose while maintaining central signal quality.

# Physical Purpose

1. **Dose Reduction**: Thicker filter at periphery attenuates more photons
2. **Uniform Detector Signal**: Compensates for varying path lengths through patient
3. **Improved Image Quality**: Reduces dynamic range requirements

# Typical Geometry

```
         Thick edges
              ↓
         ┌────┬────┐
    X-ray│ ╲  │  ╱ │ → Detector
   Source│  ╲ │ ╱  │
         └────┴────┘
              ↑
         Thin center
```

# References

**Bowtie Filter Design:**
- Hsieh, J. (2009). Computed Tomography: Principles, Design, Artifacts, Reduction (2nd ed.).
  SPIE Press, Section 2.4 - Beam shaping filters.
- AAPM TG-111 (2010). "Comprehensive methodology for the evaluation of radiation dose in CT."
  Med. Phys., 37(1), 113-122.

**Dose Optimization:**
- Bushberg, J. T., et al. (2012). The Essential Physics of Medical Imaging (3rd ed.).
  Lippincott Williams & Wilkins, Chapter 13.
"""

import XrayAttenuation as XA
using Unitful: keV, ustrip

# ==============================================================================
# Bowtie Filter Thickness Profile
# ==============================================================================

"""
    bowtie_thickness_profile(
        fan_angles::Vector{Float64};
        max_thickness_cm::Float64 = 3.0,
        center_thickness_cm::Float64 = 0.5,
        profile::Symbol = :parabolic
    )::Vector{Float64}

Compute bowtie filter thickness as a function of fan angle.

# Arguments

- `fan_angles::Vector{Float64}` - Fan angles in degrees (typically -25° to +25°)
- `max_thickness_cm::Float64 = 3.0` - Maximum thickness at edges (cm)
- `center_thickness_cm::Float64 = 0.5` - Minimum thickness at center (cm)
- `profile::Symbol = :parabolic` - Thickness profile shape
  - `:parabolic` - Smooth parabolic (most common)
  - `:linear` - Linear taper
  - `:exponential` - Exponential taper

# Returns

- `thickness::Vector{Float64}` - Filter thickness in cm for each fan angle

# Profile Shapes

**Parabolic** (default):
```
t(θ) = t_center + (t_max - t_center) × (θ / θ_max)²
```

**Linear**:
```
t(θ) = t_center + (t_max - t_center) × |θ / θ_max|
```

**Exponential**:
```
t(θ) = t_center × exp(k × |θ|)
```

# Example

```julia
# Canon Aquilion ONE fan angle range: ±24.75°
fan_angles = collect(range(-24.75, 24.75, length=800))

# Compute bowtie thickness
thickness = bowtie_thickness_profile(fan_angles, max_thickness_cm=3.0)

# Plot (requires Plots.jl)
using Plots
plot(fan_angles, thickness, xlabel="Fan Angle (deg)", ylabel="Thickness (cm)")
```

# Typical Values

**Body Bowtie Filter:**
- Center: 0.5 cm Al
- Edge: 3.0-4.0 cm Al
- Profile: Parabolic

**Head Bowtie Filter:**
- Center: 0.3 cm Al
- Edge: 2.0 cm Al
- Profile: Parabolic

# References

- Hsieh (2009) Section 2.4 - Beam shaping filters
- Mail et al. (2009) Med Phys - Bowtie filter design
"""
function bowtie_thickness_profile(
        fan_angles::Vector{Float64};
        max_thickness_cm::Float64 = 3.0,
        center_thickness_cm::Float64 = 0.5,
        profile::Symbol = :parabolic
    )::Vector{Float64}

    @assert max_thickness_cm >= center_thickness_cm "Max thickness must be >= center thickness"

    n_angles = length(fan_angles)
    thickness = zeros(Float64, n_angles)

    # Normalize angles to [-1, 1] range
    θ_max = maximum(abs.(fan_angles))
    @assert θ_max > 0 "Fan angles must have non-zero range"

    for (i, θ) in enumerate(fan_angles)
        # Normalized position: 0 at center, ±1 at edges
        u = θ / θ_max

        # Compute thickness based on profile type
        if profile == :parabolic
            # Parabolic: t(u) = t_center + (t_max - t_center) × u²
            thickness[i] = center_thickness_cm + (max_thickness_cm - center_thickness_cm) * u^2

        elseif profile == :linear
            # Linear: t(u) = t_center + (t_max - t_center) × |u|
            thickness[i] = center_thickness_cm + (max_thickness_cm - center_thickness_cm) * abs(u)

        elseif profile == :exponential
            # Exponential: t(u) = t_center × exp(k × |u|)
            # Choose k such that t(±1) = t_max
            k = log(max_thickness_cm / center_thickness_cm)
            thickness[i] = center_thickness_cm * exp(k * abs(u))

        else
            error("Unknown bowtie profile: $profile. Use :parabolic, :linear, or :exponential")
        end
    end

    return thickness
end

# ==============================================================================
# Bowtie Filter Attenuation
# ==============================================================================

"""
    apply_bowtie_filter(
        energies::Vector{Float64},
        photons::Vector{Float64},
        detector_col::Int,
        n_cols::Int;
        max_thickness_cm::Float64 = 3.0,
        center_thickness_cm::Float64 = 0.5,
        material::XA.Material = XA.Materials.aluminum
    )::Vector{Float64}

Apply bowtie filter attenuation to X-ray spectrum.

# Algorithm

1. **Determine fan angle** for detector column:
   ```
   θ_fan = atan((col - n_cols/2) × pixel_width / SDD)
   ```

2. **Lookup filter thickness** at this angle:
   ```
   t = bowtie_thickness_profile(θ_fan)
   ```

3. **Attenuate spectrum**:
   ```
   N'(E) = N₀(E) × exp(-μ_Al(E) × t)
   ```

# Arguments

- `energies::Vector{Float64}` - Energy bins (keV)
- `photons::Vector{Float64}` - Photon fluence per energy bin
- `detector_col::Int` - Detector column index (1-based)
- `n_cols::Int` - Total number of detector columns
- `max_thickness_cm::Float64 = 3.0` - Edge thickness (cm Al)
- `center_thickness_cm::Float64 = 0.5` - Center thickness (cm Al)
- `material::XA.Material = XA.Materials.aluminum` - Filter material

# Returns

- `filtered_photons::Vector{Float64}` - Attenuated photon fluence

# Example

```julia
# Generate 120 kVp spectrum
spectrum = generate_spectrum(kVp=120.0, mAs=200.0)

# Apply bowtie filter for column 400 (center) of 800 total
filtered = apply_bowtie_filter(
    spectrum.energies,
    spectrum.photons,
    400,  # center column
    800,  # total columns
    max_thickness_cm=3.0
)

# Verify: center column has minimal attenuation
@assert sum(filtered) > 0.9 * sum(spectrum.photons)  # < 10% loss at center

# Edge column (col=1 or 800) should have more attenuation
filtered_edge = apply_bowtie_filter(
    spectrum.energies,
    spectrum.photons,
    1,  # edge column
    800,
    max_thickness_cm=3.0
)
@assert sum(filtered_edge) < 0.5 * sum(spectrum.photons)  # > 50% loss at edge
```

# Integration with Forward Model

```julia
for col in 1:geometry.n_cols
    # Apply bowtie filter to source spectrum for this column
    filtered_photons = apply_bowtie_filter(
        spectrum.energies,
        spectrum.photons,
        col,
        geometry.n_cols,
        max_thickness_cm=3.0
    )

    # Use filtered spectrum for forward projection
    # ... (ray tracing and attenuation)
end
```

# References

- Hsieh (2009) - Beam shaping filters
- AAPM TG-111 (2010) - CT dose evaluation
"""
function apply_bowtie_filter(
        energies::Vector{Float64},
        photons::Vector{Float64},
        detector_col::Int,
        n_cols::Int;
        max_thickness_cm::Float64 = 3.0,
        center_thickness_cm::Float64 = 0.5,
        material::XA.Material = XA.Materials.aluminum
    )::Vector{Float64}

    @assert length(energies) == length(photons) "Energy and photon arrays must have same length"
    @assert detector_col >= 1 && detector_col <= n_cols "Detector column out of range"

    # Convert column index to normalized position [-1, 1]
    # Center column: u = 0
    # Edge columns: u = ±1
    u = 2.0 * (detector_col - (n_cols + 1) / 2.0) / n_cols

    # Compute filter thickness at this position
    # Use parabolic profile (most common)
    thickness_cm = center_thickness_cm + (max_thickness_cm - center_thickness_cm) * u^2

    # Apply Beer-Lambert law for each energy
    filtered_photons = similar(photons)

    for (i, E_keV) in enumerate(energies)
        # Get aluminum attenuation coefficient at this energy
        E = E_keV * keV
        μ = XA.linear_attenuation_coeff(material, E)
        μ_cm = ustrip(μ)  # Convert to cm^-1

        # Exponential attenuation
        attenuation_factor = exp(-μ_cm * thickness_cm)

        # Apply to photon fluence
        filtered_photons[i] = photons[i] * attenuation_factor
    end

    return filtered_photons
end

# ==============================================================================
# Exports
# ==============================================================================

export bowtie_thickness_profile
export apply_bowtie_filter
