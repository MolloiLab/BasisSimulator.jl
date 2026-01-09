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
    add_1_over_f_noise(
        signal::Array{Float64, 3};
        alpha::Float64 = 1.0,
        amplitude::Float64 = 1000.0,
        seed::Union{Int,Nothing} = nothing
    )::Array{Float64, 3}

Add 1/f (pink) noise to simulate low-frequency drift across projection angles.

1/f noise is characterized by power spectral density:
```
S(f) ∝ 1/f^α    (α ≈ 1)
```

This causes slow drifts in detector response over time/projection angle, modeling:
- Thermal drift in detector electronics
- X-ray tube output variations
- Mechanical vibrations

# Algorithm

1. Generate white noise in frequency domain
2. Apply 1/f^α filter:
   ```
   H(f) = 1 / f^α    for f > 0
   H(0) = 0          (no DC component)
   ```
3. Transform back to spatial domain
4. Add to each detector pixel as function of projection angle

# Arguments

- `signal::Array{Float64, 3}` - Detector signal [rows, cols, angles]
- `alpha::Float64 = 1.0` - Power law exponent (α=1 is pink noise)
  - α = 0: white noise
  - α = 1: pink noise (1/f)
  - α = 2: brown noise (1/f²)
- `amplitude::Float64 = 1000.0` - Noise amplitude scaling
- `seed::Union{Int,Nothing} = nothing` - Random seed

# Returns

- `noisy_signal::Array{Float64, 3}` - Signal with 1/f noise added

# Example

```julia
# Add pink noise (α=1) to detector signal
noisy = add_1_over_f_noise(signal, alpha=1.0, amplitude=500.0)

# Brown noise (slower drift, α=2)
noisy = add_1_over_f_noise(signal, alpha=2.0, amplitude=200.0)
```

# Physical Interpretation

**1/f noise** appears as slow variations across projection angles:
- Low frequencies (f → 0): High power → slow drift
- High frequencies: Low power → fast variations suppressed

**Typical values**:
- Amplitude: 100-1000 (energy units, ~0.1-1% of signal)
- α: 0.8-1.2 (α=1 is canonical pink noise)

# References

- Press, W. H. (1978). Comm. Mod. Phys. C, 7(4), 103-119.
  "Flicker noises in astronomy and elsewhere"
- Timmer, J., & Koenig, M. (1995). A&A, 300, 707.
  "On generating power law noise"
"""
function add_1_over_f_noise(
        signal::Array{Float64, 3};
        alpha::Float64 = 1.0,
        amplitude::Float64 = 1000.0,
        seed::Union{Int,Nothing} = nothing
    )::Array{Float64, 3}

    @assert alpha >= 0.0 "Alpha must be non-negative"
    @assert amplitude >= 0.0 "Amplitude must be non-negative"

    # Set random seed if provided
    if seed !== nothing
        Random.seed!(seed)
    end

    n_rows, n_cols, n_angles = size(signal)

    # Generate 1/f noise for each detector pixel
    # Noise varies across projection angles
    noisy_signal = copy(signal)

    for row in 1:n_rows
        for col in 1:n_cols
            # Generate 1/f noise sequence across angles
            noise_1d = generate_1_over_f_sequence(n_angles, alpha)

            # Scale to desired amplitude
            noise_1d .*= amplitude

            # Add to signal at this detector pixel
            for angle_idx in 1:n_angles
                noisy_signal[row, col, angle_idx] += noise_1d[angle_idx]
            end
        end
    end

    return noisy_signal
end

"""
    generate_1_over_f_sequence(n::Int, alpha::Float64)::Vector{Float64}

Generate 1D sequence with 1/f^α power spectral density.

Uses the Timmer & Koenig (1995) algorithm:
1. Create white noise in frequency domain
2. Apply 1/f^α filter
3. Inverse FFT to get time series

# Arguments

- `n::Int` - Length of sequence
- `alpha::Float64` - Power law exponent

# Returns

- `Vector{Float64}` - Noise sequence with 1/f^α spectrum
"""
function generate_1_over_f_sequence(n::Int, alpha::Float64)::Vector{Float64}
    # Create frequency array
    freqs = fftfreq(n)

    # Generate white noise in frequency domain
    # Complex Gaussian (independent real and imaginary parts)
    noise_fft = Complex{Float64}.(randn(n), randn(n))

    # Apply 1/f^α filter
    for i in 1:n
        f = abs(freqs[i])

        if f == 0.0
            # No DC component (zero mean)
            noise_fft[i] = 0.0
        else
            # Scale by 1/f^α
            noise_fft[i] /= f^(alpha / 2)
        end
    end

    # Inverse FFT to get time series
    noise_time = real.(ifft(noise_fft))

    # Normalize to unit variance
    noise_time ./= std(noise_time)

    return noise_time
end

# ==============================================================================
# Noise Power Spectrum (NPS) - Placeholder
# ==============================================================================

"""
    compute_nps(
        image::Matrix{Float64};
        roi_size::Int = 64,
        n_rois::Int = 100,
        detrend::Bool = true
    )::Matrix{Float64}

Compute 2D Noise Power Spectrum (NPS) for image quality assessment.

NPS quantifies the frequency content of noise:
```
NPS(fx, fy) = (Δx × Δy) / (Nx × Ny) × |FFT(noise)|²
```

Averaged over multiple regions of interest (ROIs) for statistical reliability.

# Algorithm

1. Extract multiple ROIs from image (typically 64×64 pixels)
2. Detrend each ROI (remove linear trends)
3. Compute 2D FFT of each ROI
4. Calculate power spectrum: |FFT|²
5. Average over all ROIs
6. Normalize by ROI size and pixel spacing

# Arguments

- `image::Matrix{Float64}` - 2D image (typically reconstructed CT slice)
- `roi_size::Int = 64` - Size of square ROI (pixels)
- `n_rois::Int = 100` - Number of ROIs to average
- `detrend::Bool = true` - Remove linear trends from ROIs

# Returns

- `nps::Matrix{Float64}` - 2D noise power spectrum [roi_size × roi_size]
  - Units: [HU² × mm²] if image is in HU
  - Center corresponds to DC (zero frequency)
  - Use fftshift to center for visualization

# Example

```julia
# Compute NPS from reconstructed slice
slice = volume[:, :, div(end, 2)]  # Central slice
nps = compute_nps(slice, roi_size=64, n_rois=100)

# Radial profile for 1D visualization
nps_radial = radial_profile(fftshift(nps))

# Plot (requires Plots.jl)
using Plots
heatmap(log10.(fftshift(nps)), title="NPS (log scale)")
```

# Physical Interpretation

**NPS shape** indicates noise characteristics:
- Flat spectrum: White noise (Poisson-dominated)
- 1/f² fall-off: Correlated noise (detector blur, reconstruction filter)
- Peaks: Structured noise (aliasing, gridding artifacts)

**Integrated NPS** → noise variance:
```
σ² = ∫∫ NPS(fx, fy) dfx dfy
```

# Quality Metrics

**Noise Equivalent Quanta (NEQ)**:
```
NEQ = (SNR)² / NPS
```

**Detectability Index**:
```
d' = ∫∫ (MTF(f) × Contrast)² / NPS(f) df
```

# References

- Samei et al. (2006) Med Phys 33(10):3683-3693
  "Performance evaluation of computed tomography systems"
- ICRU Report 87 (2012)
  "Radiation dose and image quality assessment in CT"
- Richard et al. (2012) Med Phys 39(4):2091-2106
  "Towards task-based assessment of CT performance"
"""
function compute_nps(
        image::Matrix{Float64};
        roi_size::Int = 64,
        n_rois::Int = 100,
        detrend::Bool = true
    )::Matrix{Float64}

    @assert roi_size > 0 "ROI size must be positive"
    @assert n_rois > 0 "Number of ROIs must be positive"

    m, n = size(image)
    @assert m >= roi_size && n >= roi_size "Image must be larger than ROI size"

    # Initialize NPS accumulator
    nps_sum = zeros(Float64, roi_size, roi_size)
    n_valid_rois = 0

    # Extract ROIs and compute NPS
    for _ in 1:n_rois
        # Random ROI location
        row_start = rand(1:(m - roi_size + 1))
        col_start = rand(1:(n - roi_size + 1))

        # Extract ROI
        roi = image[row_start:(row_start + roi_size - 1),
                    col_start:(col_start + roi_size - 1)]

        # Detrend if requested (remove linear trends)
        if detrend
            roi = detrend_2d(roi)
        end

        # Compute 2D FFT
        roi_fft = fft(roi)

        # Power spectrum
        power = abs2.(roi_fft)

        # Accumulate
        nps_sum .+= power
        n_valid_rois += 1
    end

    # Average over ROIs
    nps = nps_sum ./ n_valid_rois

    # Normalize by ROI size
    # Standard NPS normalization: (Δx × Δy) / (Nx × Ny)
    # Assuming unit pixel spacing (can be scaled by voxel size if needed)
    nps ./= (roi_size * roi_size)

    return nps
end

"""
    detrend_2d(image::Matrix{Float64})::Matrix{Float64}

Remove linear trend from 2D image.

Fits a plane: z = a + bx + cy and subtracts it.

This removes low-frequency trends that can bias NPS computation.
"""
function detrend_2d(image::Matrix{Float64})::Matrix{Float64}
    m, n = size(image)

    # Create coordinate grids
    x = repeat(1:n, 1, m)'
    y = repeat(1:m, 1, n)

    # Flatten
    x_flat = vec(x)
    y_flat = vec(y)
    z_flat = vec(image)

    # Fit plane: z = a + b*x + c*y
    # Using least squares: [1 x y] * [a; b; c] = z
    A = hcat(ones(length(x_flat)), x_flat, y_flat)
    coeffs = A \ z_flat  # [a, b, c]

    # Compute fitted plane
    plane = coeffs[1] .+ coeffs[2] .* x .+ coeffs[3] .* y

    # Subtract trend
    detrended = image .- plane

    return detrended
end

# ==============================================================================
# Exports
# ==============================================================================

export apply_poisson_noise
export add_electronic_noise
export add_1_over_f_noise
export compute_nps
