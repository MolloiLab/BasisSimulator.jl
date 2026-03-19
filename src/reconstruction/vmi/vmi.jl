# =============================================================================
# spectral/vmi.jl — Unified VMI Pipeline
#
# Shared functions for both dual-kVp and PCCT virtual monoenergetic imaging.
# PCCT combines bins into low/high → same decomposition as dual-kVp → same VMI.
# Only water/iodine/calcium basis materials.
# =============================================================================

import AcceleratedKernels as AK
import XrayAttenuation as XA

"""
    get_basis_mu(material::Symbol, energy_keV::Float64) -> Float64

Get linear attenuation coefficient for a basis material at given energy.

Supports three basis materials:
- `:water` — pure water (NIST XCOM via XrayAttenuation.jl)
- `:iodine` — dilute iodine solution (5 mg/mL in water)
- `:calcium` — calcium-equivalent material (200 mg/cc)

# Arguments
- `material::Symbol`: Basis material
- `energy_keV::Float64`: Energy in keV

# Returns
Linear attenuation coefficient (cm⁻¹).
"""
function get_basis_mu(material::Symbol, energy_keV::Float64)
    if material == :water
        return compute_μ_at_energy(XA.Materials.water, energy_keV)
    elseif material == :iodine
        return get_iodine_solution_attenuation(energy_keV; conc_mg_ml=5.0)
    elseif material == :calcium
        return get_calcium_material_attenuation(energy_keV; density_mg_cc=200.0)
    else
        error("Unknown basis material: $material. Use :water, :iodine, or :calcium")
    end
end

"""
    compute_decomposition_matrix(basis::Tuple{Symbol,Symbol}, E_low::Float64, E_high::Float64; T::Type=Float64) -> NTuple{4}

Compute the 2×2 inverse decomposition matrix for two basis materials at two energies.

Returns (inv_a11, inv_a12, inv_a21, inv_a22) such that:
    material1 = inv_a11 * p_low + inv_a12 * p_high
    material2 = inv_a21 * p_low + inv_a22 * p_high

# Arguments
- `basis`: Tuple of two basis material symbols (e.g., `(:water, :iodine)`)
- `E_low`: Effective energy of low-energy measurement (keV)
- `E_high`: Effective energy of high-energy measurement (keV)

# Keyword Arguments
- `T::Type=Float64`: Output element type
"""
function compute_decomposition_matrix(basis::Tuple{Symbol,Symbol}, E_low::Float64, E_high::Float64; T::Type=Float64)
    m1, m2 = basis
    μ1_low  = get_basis_mu(m1, E_low)
    μ1_high = get_basis_mu(m1, E_high)
    μ2_low  = get_basis_mu(m2, E_low)
    μ2_high = get_basis_mu(m2, E_high)

    det_A = μ1_low * μ2_high - μ2_low * μ1_high
    if abs(det_A) < 1e-10
        error("Singular decomposition matrix — basis materials too similar at E_low=$E_low, E_high=$E_high")
    end

    inv_a11 = T(μ2_high / det_A)
    inv_a12 = T(-μ2_low / det_A)
    inv_a21 = T(-μ1_high / det_A)
    inv_a22 = T(μ1_low / det_A)

    return (inv_a11, inv_a12, inv_a21, inv_a22)
end

"""
    spectral_decompose!(mat1, mat2, low, high, inv_a11, inv_a12, inv_a21, inv_a22)

GPU-compatible in-place 2-material decomposition.

Applies: mat1[i] = inv_a11 * low[i] + inv_a12 * high[i]
         mat2[i] = inv_a21 * low[i] + inv_a22 * high[i]

All arrays must be the same size and on the same device.
"""
function spectral_decompose!(mat1, mat2, low, high, inv_a11, inv_a12, inv_a21, inv_a22)
    let inv_a11=inv_a11, inv_a12=inv_a12, inv_a21=inv_a21, inv_a22=inv_a22,
        mat1=mat1, mat2=mat2, low=low, high=high
        AK.foreachindex(mat1) do idx
            p_low = low[idx]
            p_high = high[idx]
            mat1[idx] = inv_a11 * p_low + inv_a12 * p_high
            mat2[idx] = inv_a21 * p_low + inv_a22 * p_high
        end
    end
    return nothing
end

"""
    spectral_vmi!(output, mat1, mat2, μ1, μ2)

GPU-compatible in-place VMI synthesis from two material maps.

Applies: output[i] = mat1[i] * μ1 + mat2[i] * μ2

# Arguments
- `output`: Pre-allocated output array
- `mat1, mat2`: Material density maps
- `μ1, μ2`: Attenuation coefficients at target VMI energy
"""
function spectral_vmi!(output, mat1, mat2, μ1, μ2)
    let output=output, mat1=mat1, mat2=mat2, μ1=μ1, μ2=μ2
        AK.foreachindex(output) do idx
            output[idx] = mat1[idx] * μ1 + mat2[idx] * μ2
        end
    end
    return output
end

"""
    combine_pcct_bins!(low_out, high_out, bins; split_bin::Int=2)

GPU-compatible combination of PCCT energy bins into low/high sinograms.

Bins 1:split_bin → low_out (averaged), bins (split_bin+1):end → high_out (averaged).
Operates in line-integral domain.

Specialised for NAEOTOM-style 4-bin detectors (split_bin=2): fuses sum+divide
into a single kernel per output, avoiding extra kernel launches.

# Arguments
- `low_out`: Pre-allocated output for low-energy combined sinogram
- `high_out`: Pre-allocated output for high-energy combined sinogram
- `bins::Vector`: Vector of sinogram arrays (one per energy bin)

# Keyword Arguments
- `split_bin::Int=2`: Last bin index included in low-energy group
"""
function combine_pcct_bins!(low_out, high_out, bins; split_bin::Int=2)
    T = eltype(low_out)
    n_bins = length(bins)
    n_low = split_bin
    n_high = n_bins - split_bin

    # Specialised 2+2 path (NAEOTOM 4-bin): 2 kernel launches total
    if n_low == 2 && n_high == 2
        b1, b2, b3, b4 = bins[1], bins[2], bins[3], bins[4]
        half = T(0.5)
        let lo=low_out, hi=high_out, b1=b1, b2=b2, b3=b3, b4=b4, h=half
            AK.foreachindex(lo) do idx
                lo[idx] = (b1[idx] + b2[idx]) * h
            end
        end
        let lo=low_out, hi=high_out, b1=b1, b2=b2, b3=b3, b4=b4, h=half
            AK.foreachindex(hi) do idx
                hi[idx] = (b3[idx] + b4[idx]) * h
            end
        end
        return nothing
    end

    # General path: accumulate then divide (N+2 kernel launches)
    inv_n_low = T(1) / T(n_low)
    fill!(low_out, zero(T))
    for b in 1:split_bin
        let bin_b = bins[b], lo = low_out
            AK.foreachindex(lo) do idx
                lo[idx] += bin_b[idx]
            end
        end
    end
    # Fold division into final pass
    let lo = low_out, s = inv_n_low
        AK.foreachindex(lo) do idx
            lo[idx] *= s
        end
    end

    if n_high > 0
        inv_n_high = T(1) / T(n_high)
        fill!(high_out, zero(T))
        for b in (split_bin+1):n_bins
            let bin_b = bins[b], hi = high_out
                AK.foreachindex(hi) do idx
                    hi[idx] += bin_b[idx]
                end
            end
        end
        let hi = high_out, s = inv_n_high
            AK.foreachindex(hi) do idx
                hi[idx] *= s
            end
        end
    end

    return nothing
end

"""
    vmi_to_hu(vmi_image::AbstractArray{T}, energy_keV::Float64; μ_water=nothing, output=nothing) -> AbstractArray

Convert Virtual Monoenergetic Image from attenuation to Hounsfield Units.

HU = 1000 × (μ - μ_water) / μ_water

GPU-native: single fused kernel via AK.foreachindex, zero temporaries.

# Arguments
- `vmi_image`: VMI reconstruction (attenuation values)
- `energy_keV`: VMI energy in keV

# Keyword Arguments
- `μ_water=nothing`: Water attenuation for calibration. If nothing, uses NIST value.
- `output=nothing`: Pre-allocated output array. If nothing, allocates a new array.
"""
function vmi_to_hu(vmi_image::AbstractArray{T}, energy_keV::Float64; μ_water=nothing, output=nothing) where T
    μw = if μ_water === nothing
        T(get_basis_mu(:water, energy_keV))
    else
        T(μ_water)
    end
    inv_μw = T(1000) / μw

    result = output === nothing ? similar(vmi_image) : output
    let src=vmi_image, dst=result, μw=μw, s=inv_μw
        AK.foreachindex(dst) do idx
            dst[idx] = (src[idx] - μw) * s
        end
    end
    return result
end

# =============================================================================
# Image-Domain VMI (Wu et al. 2009)
#
# Reference: Wu X, Langan DA, Xu D, Benson TM, Pack JD, Schmitz AM, Tkaczyk JE.
# "Monochromatic CT Image Representation via Fast Switching Dual kVp."
# Proc. SPIE 7258, Medical Imaging 2009, 725845. doi: 10.1117/12.811698
#
# Pipeline:
#   1. Projection-domain decomposition → material sinograms (ρ₁, ρ₂)
#   2. FBP reconstruct each → density images m₁(x,y), m₂(x,y)
#   3. Image-domain VMI: Im_mono = m₁ + g(E)×m₂  where g(E) = μ₂(E)/μ_water(E)
#   4. HU = (Im_mono − 1) × 1000
#   5. Optional noise compensation (Eq. 13)
# =============================================================================

"""
    compute_vmi_g(energy_keV, basis_material2) -> Float64

Compute VMI weighting factor g(E) = μ₂(E) / μ_water(E) (Wu et al. 2009).

In image-domain VMI with water as basis material 1:

    Im_mono(x,y,E) = m₁(x,y) + g(E) × m₂(x,y)

# Arguments
- `energy_keV::Float64`: Target VMI energy in keV
- `basis_material2::Symbol`: Second basis material (e.g., `:calcium`, `:iodine`)
"""
function compute_vmi_g(energy_keV::Float64, basis_material2::Symbol)
    μ2 = get_basis_mu(basis_material2, energy_keV)
    μw = get_basis_mu(:water, energy_keV)
    return μ2 / μw
end

"""
    image_domain_vmi!(output, m1, m2, g_E)

Image-domain VMI synthesis (Wu et al. 2009, Eq. 4).

    Im_mono[i] = m₁[i] + g(E) × m₂[i]

Requires water as first basis material. `m₁`, `m₂` are reconstructed
material density images (water density ≈ 1.0, second material density ≈ 0 for water).

GPU-compatible via AcceleratedKernels.
"""
function image_domain_vmi!(output, m1, m2, g_E)
    let output=output, m1=m1, m2=m2, g_E=g_E
        AK.foreachindex(output) do idx
            output[idx] = m1[idx] + g_E * m2[idx]
        end
    end
    return output
end

"""
    image_domain_vmi_to_hu!(output, im_mono)

Convert image-domain VMI to Hounsfield Units (Wu et al. 2009):

    HU = (Im_mono − 1) × 1000

For water (m₁=1, m₂=0): Im_mono = 1 → HU = 0.

GPU-compatible via AcceleratedKernels.
"""
function image_domain_vmi_to_hu!(output, im_mono)
    T = eltype(output)
    let output=output, src=im_mono, k=T(1000), one_val=one(T)
        AK.foreachindex(output) do idx
            output[idx] = (src[idx] - one_val) * k
        end
    end
    return output
end

"""
    noise_compensated_vmi!(output, m1, m2, m2_noise, g_E, g_min)

Noise-compensated VMI synthesis (Wu et al. 2009, Eq. 13):

    Im_mono[i] = m₁[i] + g(E)×m₂[i] + (g_min − g(E))×Δ_m₂[i]

At E_optimal (g_E == g_min), the correction vanishes → standard Eq. 4.
At other energies, it suppresses the energy-dependent noise amplified by
material decomposition.

# Arguments
- `output`: Pre-allocated output array
- `m1, m2`: Material density images (water, second material)
- `m2_noise`: High-frequency noise map extracted from m₂ (Δ_m₂)
- `g_E`: VMI weighting factor g(E) at target energy
- `g_min`: VMI weighting factor at minimum-noise energy

GPU-compatible via AcceleratedKernels.
"""
function noise_compensated_vmi!(output, m1, m2, m2_noise, g_E, g_min)
    let output=output, m1=m1, m2=m2, m2n=m2_noise, g=g_E, gm=g_min
        AK.foreachindex(output) do idx
            output[idx] = m1[idx] + g * m2[idx] + (gm - g) * m2n[idx]
        end
    end
    return output
end

"""
    extract_noise_map(m2::AbstractArray{T,3}; sigma=2.0) -> Array{T,3}

Extract high-frequency noise map Δ_m₂ from material density image m₂.

    Δ_m₂ = m₂ − gaussian_lowpass(m₂)

Used for noise compensation in VMI (Wu et al. 2009, Eq. 13).
Separable 2D Gaussian blur per slice (CPU), σ in pixels.
"""
function extract_noise_map(m2::AbstractArray{T,3}; sigma::Float64=2.0) where T
    m2_cpu = Array(m2)
    nx, ny, nz = size(m2_cpu)

    # 1D Gaussian kernel (truncated at 3σ)
    r = ceil(Int, 3 * sigma)
    kernel = T[exp(-T(i)^2 / (2 * T(sigma)^2)) for i in -r:r]
    kernel ./= sum(kernel)
    klen = length(kernel)

    smooth = similar(m2_cpu)
    temp = similar(m2_cpu)

    # Separable 2D Gaussian: x-pass then y-pass, per-slice
    @inbounds for z in 1:nz
        for y in 1:ny, x in 1:nx
            s = zero(T)
            for k in 1:klen
                ix = clamp(x + k - r - 1, 1, nx)
                s += m2_cpu[ix, y, z] * kernel[k]
            end
            temp[x, y, z] = s
        end
        for y in 1:ny, x in 1:nx
            s = zero(T)
            for k in 1:klen
                iy = clamp(y + k - r - 1, 1, ny)
                s += temp[x, iy, z] * kernel[k]
            end
            smooth[x, y, z] = s
        end
    end

    return m2_cpu .- smooth
end

# =============================================================================
# Exports
# =============================================================================

export get_basis_mu, compute_decomposition_matrix
export spectral_decompose!, spectral_vmi!, combine_pcct_bins!
export vmi_to_hu
export compute_vmi_g, image_domain_vmi!, image_domain_vmi_to_hu!
export noise_compensated_vmi!, extract_noise_map
