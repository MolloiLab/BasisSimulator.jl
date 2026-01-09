"""
    Physics/Noise.jl

Noise modeling for CT systems.

Includes:
- Poisson (quantum) noise
- Electronic (Gaussian) noise
- 1/f noise (low-frequency drift)
- Noise Power Spectrum (NPS) calculation

# References

**Noise Theory:**
- Barrett, H. H., & Myers, K. J. (2004). Foundations of Image Science.
  Wiley-Interscience.
- Wagner, R. F., et al. (1999). Med. Phys., 26(11), 2237-2242.
  "Application of information theory to the assessment of computed tomography"

**Poisson Statistics:**
- Whiting, B. R. (2002). Med. Phys., 29(11), 2404-2411.
  "Signal statistics in x-ray computed tomography"

**Detector Noise:**
- Gang, G. J., et al. (2014). Med. Phys., 41(10), 101906.
  "Analysis of Fourier-domain task-based detectability index in tomosynthesis and CT"
- Siewerdsen, J. H., & Jaffray, D. A. (2000). Med. Phys., 27(8), 1813-1831.
  "Cone-beam CT with a flat-panel imager: Magnitude and effects of x-ray scatter"
"""

using Distributions
using Random
using Statistics

# ==============================================================================
# Poisson (Quantum) Noise
# ==============================================================================

"""
    apply_poisson_noise(
        signal::Array{Float64, 3};
        dose_factor::Float64 = 1.0,
        seed::Union{Int,Nothing} = nothing
    )::Array{Float64, 3}

Add Poisson (quantum) noise to detector signal.

Poisson noise arises from the statistical variation in photon counting and is
the dominant noise source in CT imaging at diagnostic dose levels.

# Algorithm

For each detector pixel with expected photon count N:

1. **Convert energy → photon counts**:
   ```
   N = E_total / E_mean
   ```
   where E_total is the integrated energy, E_mean ≈ 60 keV (typical)

2. **Sample from Poisson distribution**:
   ```
   N_measured ~ Poisson(N × dose_factor)
   ```

3. **Convert back to energy**:
   ```
   E_measured = N_measured × E_mean
   ```

# Arguments

- `signal::Array{Float64, 3}` - Detector signal (energy integrated) [rows, cols, angles]
- `dose_factor::Float64 = 1.0` - Dose multiplier (1.0 = nominal, 0.5 = half dose, etc.)
- `seed::Union{Int,Nothing} = nothing` - Random seed for reproducibility

# Returns

- `noisy_signal::Array{Float64, 3}` - Signal with Poisson noise added

# Example

```julia
# After detector response
signal = simulate_ct_scan(phantom, geometry, spectrum)

# Add nominal dose Poisson noise
noisy_signal = apply_poisson_noise(signal, dose_factor=1.0)

# Simulate low-dose scan (25% dose)
low_dose_signal = apply_poisson_noise(signal, dose_factor=0.25)
```

# Dose Scaling

The **dose factor** scales the expected photon count:

- `dose_factor = 2.0` → Double dose → σ decreases by √2
- `dose_factor = 0.5` → Half dose → σ increases by √2
- `dose_factor = 0.25` → Quarter dose → σ doubles

**Relationship**: σ_noise ∝ 1/√dose

# Physical Interpretation

For a detector pixel measuring N photons:

- **Standard deviation**: σ = √N (Poisson statistics)
- **Relative noise**: σ/N = 1/√N (SNR ∝ √N)
- **After dose scaling**: N' = N × dose_factor → σ' = √(N × dose_factor)

# Validation

Check that noise follows expected statistics:
```julia
# High-dose limit: should approach Gaussian
# Low-dose limit: discrete photon counting effects
```

# References

- Whiting (2002) Med Phys - Signal statistics in CT
- Tward & Siewerdsen (2008) Med Phys - Cascaded systems analysis
"""
function apply_poisson_noise(
        signal::Array{Float64, 3};
        dose_factor::Float64 = 1.0,
        seed::Union{Int,Nothing} = nothing
    )::Array{Float64, 3}

    @assert dose_factor > 0.0 "Dose factor must be positive"

    # Set random seed if provided
    if seed !== nothing
        Random.seed!(seed)
    end

    # Mean photon energy (keV) - typical for diagnostic CT
    # At 120 kVp with filtration, effective energy ≈ 60 keV
    E_mean = 60.0

    # Convert signal (energy) to expected photon counts
    # Signal is energy-weighted: E_total = Σ N(E) × E
    # Approximate as: E_total ≈ N_photons × E_mean
    photon_counts = signal ./ E_mean

    # Scale by dose factor
    photon_counts .*= dose_factor

    # Sample from Poisson distribution
    noisy_photon_counts = similar(photon_counts)

    for idx in eachindex(photon_counts)
        λ = max(photon_counts[idx], 0.0)  # Poisson parameter (must be non-negative)

        if λ > 0
            # For large λ, Poisson → Gaussian (faster sampling)
            if λ > 100
                # Gaussian approximation: N ~ Normal(λ, √λ)
                noisy_photon_counts[idx] = max(0.0, λ + √λ * randn())
            else
                # True Poisson sampling for small λ
                noisy_photon_counts[idx] = Float64(rand(Poisson(λ)))
            end
        else
            noisy_photon_counts[idx] = 0.0
        end
    end

    # Convert back to energy
    noisy_signal = noisy_photon_counts .* E_mean

    return noisy_signal
end

# ==============================================================================
# Electronic (Gaussian) Noise
# ==============================================================================

"""
    add_electronic_noise(
        signal::Array{Float64, 3};
        sigma::Float64 = 1000.0,
        seed::Union{Int,Nothing} = nothing
    )::Array{Float64, 3}

Add Gaussian electronic noise to detector signal.

Electronic noise arises from detector readout electronics and is independent
of signal level (additive noise).

# Algorithm

For each detector pixel:
```
signal_noisy = signal + σ × N(0, 1)
```

where N(0, 1) is a standard normal random variable.

# Arguments

- `signal::Array{Float64, 3}` - Detector signal [rows, cols, angles]
- `sigma::Float64 = 1000.0` - Electronic noise standard deviation (energy units)
- `seed::Union{Int,Nothing} = nothing` - Random seed

# Returns

- `noisy_signal::Array{Float64, 3}` - Signal with electronic noise added

# Typical Values

**Canon Aquilion ONE** (estimated):
- High quality mode: σ ≈ 500 (energy units)
- Standard mode: σ ≈ 1000
- Fast mode: σ ≈ 2000

**Relative to Quantum Noise:**
- At high dose: quantum noise >> electronic noise (negligible)
- At low dose: electronic noise becomes significant
- Crossover: ~10% of nominal dose

# Example

```julia
# Add combined quantum + electronic noise
signal_poisson = apply_poisson_noise(signal, dose_factor=1.0)
signal_total = add_electronic_noise(signal_poisson, sigma=1000.0)
```

# References

- Siewerdsen & Jaffray (2000) Med Phys - Flat-panel imaging
- Tward & Siewerdsen (2008) Med Phys - Cascaded systems analysis
"""
function add_electronic_noise(
        signal::Array{Float64, 3};
        sigma::Float64 = 1000.0,
        seed::Union{Int,Nothing} = nothing
    )::Array{Float64, 3}

    @assert sigma >= 0.0 "Sigma must be non-negative"

    # Set random seed if provided
    if seed !== nothing
        Random.seed!(seed)
    end

    # Add Gaussian noise
    noise = sigma .* randn(size(signal)...)
    noisy_signal = signal .+ noise

    return noisy_signal
end

# ==============================================================================
# 1/f Noise (Low-Frequency Drift) - Placeholder
# ==============================================================================

"""
    add_1_over_f_noise(signal::Array{Float64, 3})::Array{Float64, 3}

Add 1/f (pink) noise to simulate low-frequency drift.

**Status**: Placeholder for future implementation.

1/f noise is characterized by power spectral density:
```
S(f) ∝ 1/f^α    (α ≈ 1)
```

This causes slow drifts in detector response over time/projection angle.

# References

- Press, W. H. (1978). Comm. Mod. Phys. C, 7(4), 103-119.
  "Flicker noises in astronomy and elsewhere"
"""
function add_1_over_f_noise(signal::Array{Float64, 3})::Array{Float64, 3}
    error("1/f noise not yet implemented - coming in Phase 3")
    # Will model slow detector drift across projection angles
end

# ==============================================================================
# Noise Power Spectrum (NPS) - Placeholder
# ==============================================================================

"""
    compute_nps(image::Matrix{Float64})::Matrix{Float64}

Compute Noise Power Spectrum (NPS) for image quality assessment.

**Status**: Placeholder for future implementation.

NPS quantifies the frequency content of noise:
```
NPS(f) = |FFT(noise)|² / N
```

# References

- Samei et al. (2006) Med Phys - Performance evaluation of CT systems
- ICRU Report 87 (2012) - Radiation dose and image quality assessment in CT
"""
function compute_nps(image::Matrix{Float64})::Matrix{Float64}
    error("NPS computation not yet implemented - coming in Phase 3")
    # Will be used for image quality metrics
end

# ==============================================================================
# Exports
# ==============================================================================

export apply_poisson_noise
export add_electronic_noise
export add_1_over_f_noise  # Placeholder
export compute_nps  # Placeholder
