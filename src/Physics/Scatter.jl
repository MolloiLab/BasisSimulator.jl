"""
# Scatter Physics Module

Implements X-ray scatter models for CT simulation.

## References
- Klein & Nishina (1929) Z. Phys. 52:853-868
- Siewerdsen et al. (2006) Med Phys 33(11):3967-3976
"""

using FFTW
using Statistics

function klein_nishina_cross_section(E_keV::Float64, theta_rad::Float64)::Float64
    m_e_c2 = 511.0
    alpha = E_keV / m_e_c2
    E_prime = E_keV / (1.0 + alpha * (1.0 - cos(theta_rad)))
    ratio = E_prime / E_keV
    return ratio^2 * (ratio + 1.0/ratio - sin(theta_rad)^2)
end

function create_scatter_kernel(
        kernel_size::Int,
        pixel_width_cm::Float64,
        E_mean_keV::Float64;
        SPR::Float64 = 0.1
    )::Matrix{Float64}
    
    sigma_cm = 2.0 * sqrt(60.0 / E_mean_keV)
    sigma_pixels = sigma_cm / pixel_width_cm
    center = (kernel_size + 1) / 2
    kernel = zeros(Float64, kernel_size, kernel_size)
    
    for i in 1:kernel_size
        for j in 1:kernel_size
            r_pixels = sqrt((i - center)^2 + (j - center)^2)
            kernel[i, j] = exp(-r_pixels^2 / (2 * sigma_pixels^2))
        end
    end
    
    kernel ./= sum(kernel)
    kernel .*= SPR
    return kernel
end

function apply_scatter(
        primary_signal::Matrix{Float64},
        pixel_width_cm::Float64,
        E_mean_keV::Float64;
        SPR::Float64 = 0.1,
        kernel_size::Int = 51
    )::Matrix{Float64}
    
    kernel = create_scatter_kernel(kernel_size, pixel_width_cm, E_mean_keV, SPR=SPR)
    n_rows, n_cols = size(primary_signal)
    kernel_padded = zeros(Float64, n_rows, n_cols)
    
    k_half = div(kernel_size, 2)
    r_start = div(n_rows, 2) - k_half + 1
    c_start = div(n_cols, 2) - k_half + 1
    kernel_padded[r_start:(r_start+kernel_size-1), c_start:(c_start+kernel_size-1)] .= kernel
    
    primary_fft = fft(primary_signal)
    kernel_fft = fft(kernel_padded)
    scatter_fft = primary_fft .* kernel_fft
    scatter_signal = real(ifft(scatter_fft))
    
    return primary_signal .+ scatter_signal
end

function estimate_SPR(object_thickness_cm::Float64, kVp::Float64, field_size_cm::Float64)::Float64
    SPR_base = 0.02
    thickness_factor = object_thickness_cm / 10.0
    field_factor = field_size_cm / 20.0
    energy_factor = sqrt(100.0 / kVp)
    SPR = SPR_base * thickness_factor * field_factor * energy_factor
    return clamp(SPR, 0.0, 2.0)
end

export klein_nishina_cross_section, create_scatter_kernel, apply_scatter, estimate_SPR
