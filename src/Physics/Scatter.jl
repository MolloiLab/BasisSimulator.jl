"""
    Physics/Scatter.jl

Compton and Rayleigh scatter modeling for CT systems.

Includes:
- Klein-Nishina differential cross section
- Convolution-based scatter estimation
- Scatter-to-Primary Ratio (SPR) calculation

# References

**Compton Scattering:**
- Klein, O., & Nishina, Y. (1929). Z. Physik, 52(11-12), 853-868.
- Hubbell, J. H., et al. (1975). J. Phys. Chem. Ref. Data, 4(3), 471-538.

**CT Scatter Modeling:**
- Siewerdsen, J. H., et al. (2006). Med. Phys., 33(11), 3957-3975.
  "A simple, direct method for x-ray scatter estimation and correction"
- Ohnesorge, B., et al. (1999). Med. Phys., 26(12), 2563-2568.
  "Efficient object scatter correction algorithm for third and fourth generation CT scanners"

**Monte Carlo Validation:**
- Elbakri, I. A., & Fessler, J. A. (2002). Phys. Med. Biol., 47(16), 2773.
  "Statistical image reconstruction for polyenergetic X-ray computed tomography"
"""

using FFTW
using Statistics

# ==============================================================================
# Scatter Estimation (Convolution Method)
# ==============================================================================

"""
    estimate_scatter_convolution(
        primary::Array{Float64, 3};
        spr::Float64 = 0.15,
        kernel_sigma::Float64 = 30.0
    )::Array{Float64, 3}

Estimate scatter using convolution with Gaussian kernel.

This is a fast approximation based on the observation that scatter has a
smooth, low-frequency spatial distribution relative to primary signal.

# Algorithm

1. Assume Scatter-to-Primary Ratio (SPR):
   ```
   S / P ≈ constant (typically 0.10-0.20 for body CT)
   ```

2. Model scatter as smoothed version of primary:
   ```
   S(u,v) = SPR × [P(u,v) ⊗ G_σ(u,v)]
   ```
   where G_σ is a Gaussian kernel with standard deviation σ.

# Arguments

- `primary::Array{Float64, 3}` - Primary (non-scattered) signal [rows, cols, angles]
- `spr::Float64 = 0.15` - Scatter-to-Primary Ratio (default: 0.15 or 15%)
- `kernel_sigma::Float64 = 30.0` - Gaussian kernel std dev (pixels, default: 30)

# Returns

- `scatter::Array{Float64, 3}` - Estimated scatter distribution (same size as primary)

# Example

```julia
# After forward projection (primary signal)
primary = simulate_ct_scan(phantom, geometry, spectrum)

# Estimate scatter
scatter = estimate_scatter_convolution(primary, spr=0.15, kernel_sigma=30.0)

# Total signal = primary + scatter
total_signal = primary .+ scatter

# Apply -log transform for reconstruction
sinogram = -log.(total_signal ./ I0)
```

# Physical Justification

**SPR Values** (body CT at 120 kVp):
- Thin patients (~20 cm): SPR ≈ 0.10 (10%)
- Average patients (~30 cm): SPR ≈ 0.15 (15%)
- Large patients (~40 cm): SPR ≈ 0.25 (25%)

**Kernel Size** (σ in pixels):
- Small FOV (head): σ ≈ 20 pixels
- Medium FOV (body): σ ≈ 30 pixels
- Large FOV (pelvis): σ ≈ 40 pixels

# Limitations

- Assumes SPR is constant across field of view
- Does not account for energy-dependent scatter
- Simplified vs. full Monte Carlo simulation
- Best for quick estimates, not high-accuracy applications

# References

- Siewerdsen et al. (2006) Med Phys - Convolution-based scatter correction
- Zhu et al. (2009) Med Phys - Scatter correction for cone-beam CT
"""
function estimate_scatter_convolution(
        primary::Array{Float64, 3};
        spr::Float64 = 0.15,
        kernel_sigma::Float64 = 30.0
    )::Array{Float64, 3}

    n_rows, n_cols, n_angles = size(primary)

    # Validate inputs
    @assert spr >= 0.0 && spr <= 1.0 "SPR must be in [0, 1]"
    @assert kernel_sigma > 0.0 "Kernel sigma must be positive"

    # Create 2D Gaussian kernel
    # Kernel size: ±3σ captures 99.7% of Gaussian
    kernel_radius = ceil(Int, 3 * kernel_sigma)
    kernel_size = 2 * kernel_radius + 1

    # Coordinate grid
    x = collect(-kernel_radius:kernel_radius)
    y = collect(-kernel_radius:kernel_radius)

    # 2D Gaussian kernel
    kernel = zeros(Float64, kernel_size, kernel_size)
    for (i, yi) in enumerate(y)
        for (j, xi) in enumerate(x)
            r_sq = xi^2 + yi^2
            kernel[i, j] = exp(-r_sq / (2 * kernel_sigma^2))
        end
    end

    # Normalize kernel
    kernel ./= sum(kernel)

    # Estimate scatter for each projection angle
    scatter = zeros(Float64, n_rows, n_cols, n_angles)

    for angle_idx in 1:n_angles
        # Get primary for this angle
        primary_proj = primary[:, :, angle_idx]

        # Convolve with Gaussian kernel (smooth the primary)
        # Use FFT convolution for speed (O(n log n) vs O(n²))
        scatter_proj = imfilter_fft(primary_proj, kernel)

        # Scale by SPR
        scatter[:, :, angle_idx] = spr .* scatter_proj
    end

    return scatter
end

"""
    imfilter_fft(image::Matrix{Float64}, kernel::Matrix{Float64})::Matrix{Float64}

Fast 2D convolution using FFT.

This is much faster than direct convolution for large kernels.
Complexity: O(n log n) vs O(n²m²) for direct convolution.
"""
function imfilter_fft(image::Matrix{Float64}, kernel::Matrix{Float64})::Matrix{Float64}
    # Pad to avoid circular convolution artifacts
    m, n = size(image)
    km, kn = size(kernel)

    # Output size (same as input)
    out_m, out_n = m, n

    # Padded size (must be at least m+km-1, n+kn-1)
    pad_m = nextpow(2, m + km - 1)
    pad_n = nextpow(2, n + kn - 1)

    # Zero-pad image
    image_pad = zeros(Float64, pad_m, pad_n)
    image_pad[1:m, 1:n] .= image

    # Zero-pad kernel (centered)
    kernel_pad = zeros(Float64, pad_m, pad_n)
    k_start_m = div(pad_m - km, 2) + 1
    k_start_n = div(pad_n - kn, 2) + 1
    kernel_pad[k_start_m:(k_start_m+km-1), k_start_n:(k_start_n+kn-1)] .= kernel

    # FFT convolution
    image_fft = fft(image_pad)
    kernel_fft = fft(kernel_pad)
    result_fft = image_fft .* kernel_fft
    result_pad = real.(ifft(result_fft))

    # Extract valid region (same size as input)
    offset_m = div(km, 2)
    offset_n = div(kn, 2)
    result = result_pad[(offset_m+1):(offset_m+out_m), (offset_n+1):(offset_n+out_n)]

    return result
end

# ==============================================================================
# Klein-Nishina Cross Section (Future)
# ==============================================================================

"""
    klein_nishina_cross_section(E_keV::Float64, θ::Float64)::Float64

Klein-Nishina differential cross section for Compton scattering.

**Status**: Placeholder for future Monte Carlo scatter simulation.

# Formula

```
dσ/dΩ = (r_e² / 2) × (E' / E)² × (E/E' + E'/E - sin²θ)
```

where:
- r_e = classical electron radius (2.818e-15 m)
- E = incident photon energy
- E' = scattered photon energy = E / (1 + (E/m_e c²)(1 - cos θ))
- θ = scattering angle

# References

Klein, O., & Nishina, Y. (1929). Z. Physik, 52(11-12), 853-868.
"""
function klein_nishina_cross_section(E_keV::Float64, θ::Float64)::Float64
    error("Klein-Nishina cross section not yet implemented - coming in Phase 3")
    # Will be used for Monte Carlo scatter simulation
end

# ==============================================================================
# Exports
# ==============================================================================

export estimate_scatter_convolution
export klein_nishina_cross_section  # Placeholder for future
