"""
# Detector Physics Module

Implements detector response modeling:
- Quantum Detection Efficiency (QDE)
- Modulation Transfer Function (MTF)
- Point Spread Function (PSF)

## References
- Fujita et al. (1992) IEEE TMI 11(1):34-39
- Samei et al. (1998) Med Phys 25(1):102-113
"""

using ImageFiltering
import XrayAttenuation as XA

function compute_detector_efficiency(
        energies::Vector{Float64},
        scintillator_material::XA.Material,
        thickness_mm::Float64
    )::Vector{Float64}
    
    QDE = zeros(Float64, length(energies))
    thickness_cm = thickness_mm / 10.0
    
    for (i, E) in enumerate(energies)
        mu = get_linear_attenuation(scintillator_material, E)
        QDE[i] = 1.0 - exp(-mu * thickness_cm)
    end
    
    return QDE
end

function create_detector_psf(
        size::Int,
        pixel_width_cm::Float64;
        fwhm_pixels::Float64 = 1.5
    )::Matrix{Float64}
    
    sigma = fwhm_pixels / 2.355
    center = (size + 1) / 2
    psf = zeros(Float64, size, size)
    
    for i in 1:size
        for j in 1:size
            r = sqrt((i - center)^2 + (j - center)^2)
            psf[i, j] = exp(-r^2 / (2 * sigma^2))
        end
    end
    
    psf ./= sum(psf)
    return psf
end

function apply_detector_blur(
        projection::Matrix{Float64},
        pixel_width_cm::Float64;
        fwhm_pixels::Float64 = 1.5
    )::Matrix{Float64}
    
    psf_size = min(15, div(size(projection, 1), 4))
    if psf_size % 2 == 0
        psf_size += 1
    end
    
    psf = create_detector_psf(psf_size, pixel_width_cm, fwhm_pixels=fwhm_pixels)
    kernel = centered(psf)
    
    return imfilter(projection, kernel, "symmetric")
end

export compute_detector_efficiency, create_detector_psf, apply_detector_blur
