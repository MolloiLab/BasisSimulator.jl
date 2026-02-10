"""
    Forward/DetectorNoise.jl

Detector response and noise modeling for CT simulation.

# Physics Background

CT imaging is fundamentally limited by quantum noise arising from the statistical
nature of X-ray photon detection. The number of photons detected follows Poisson
statistics, leading to signal-dependent noise characteristics.

## Quantum Noise Model

The detected photon count N follows a Poisson distribution:

    N ~ Poisson(λ)

where λ = I₀ × exp(-∫μ dl) is the expected photon count:
- I₀: incident photon fluence (photons/detector element)
- μ: linear attenuation coefficient
- dl: path length through object

For Poisson distributions, the variance equals the mean: Var(N) = λ

In projection domain (after log transform), the noise standard deviation is:

    σ_p = 1/√λ = 1/√(I₀ × exp(-p))

where p = ∫μ dl is the line integral (projection value).

## Gaussian Approximation

For λ > 100 (typical in diagnostic CT: λ ~ 10⁴-10⁶), the Poisson distribution
is well-approximated by a Gaussian:

    N ≈ λ + √λ × Z    where Z ~ N(0,1)

This approximation is standard practice in CT simulation due to:
1. High photon counts in clinical imaging (> 10⁴ per detector element)
2. Central limit theorem applicability
3. Computational efficiency (no rejection sampling)
4. Identical first two moments: E[N] = λ, Var[N] = λ

## CatSim Compatibility

CatSim uses exact Poisson sampling via the PTRD algorithm (Poisson Transformed
Rejection with Squeeze, Hormann 1992) implemented in clib_build/src/rndpoi.c.
For λ ≥ 10, CatSim also uses normal approximation internally.

Both implementations produce statistically equivalent results for CT-relevant
photon counts (λ > 100), with:
- Identical mean (E[N] = λ)
- Identical variance (Var[N] = λ)
- White noise power spectrum (NPS)
- Spatially uncorrelated noise

## Noise Power Spectrum (NPS)

Quantum noise in CT projections produces white noise (flat NPS) in the
projection domain. After filtered backprojection reconstruction, the
NPS becomes frequency-weighted due to the ramp filter.

## Electronic Noise

In addition to quantum noise, detectors exhibit electronic noise from:
- Readout electronics (additive Gaussian)
- Dark current (additive)
- Gain variations (multiplicative)

Electronic noise is typically 5-20 counts σ, becoming significant at low
dose levels where quantum noise is comparable.

# GPU Compatibility

All noise functions are GPU-native via AcceleratedKernels.jl:
- ✅ Metal (Apple Silicon)
- ✅ CUDA (NVIDIA)
- ✅ ROCm (AMD)
- ✅ CPU fallback

Random numbers are pre-generated on CPU and transferred to GPU for
reproducibility and deterministic behavior.

# References

1. Macovski A. "Medical Imaging Systems." Prentice Hall, 1983.
   Chapter 5: Noise in Medical Imaging

2. Hsieh J. "Computed Tomography: Principles, Design, Artifacts, and
   Recent Advances." 3rd ed. SPIE Press, 2015.
   Chapter 7: Noise, Artifacts, and Quality Assurance

3. Hormann W. "The transformed rejection method for generating Poisson
   random variables." Insurance: Mathematics and Economics, 1993;12:39-45.
   doi:10.1016/0167-6687(93)90997-4

4. Barrett HH, Myers KJ. "Foundations of Image Science." Wiley, 2004.
   Chapter 11: Quantum Noise in Imaging

5. CatSim/XCIST implementation: pyfiles/Detection_EI.py, clib_build/src/rndpoi.c
   https://github.com/xcist/main
"""

import AcceleratedKernels as AK
using Random

"""
    DetectorModel

Parameters for detector response simulation.
"""
struct DetectorModel
    # Detector MTF (blur) - FWHM in pixels
    blur_fwhm::Float64

    # Photon count at detector (affects quantum noise)
    # Higher = less noise, more dose
    I0::Float64

    # Electronic noise standard deviation (in detector units)
    electronic_noise_std::Float64

    # Random seed (for reproducibility)
    seed::Union{Nothing, Int}
end

"""
    default_detector_model(; blur_fwhm=1.5, I0=1e5, electronic_noise_std=10.0, seed=nothing)

Create detector model with default parameters.

# Parameters
- `blur_fwhm`: Detector PSF FWHM in pixels (typical: 1.0-2.0)
- `I0`: Incident photon count (typical: 1e4-1e6)
- `electronic_noise_std`: Electronic noise σ (typical: 5-20)
- `seed`: Random seed for reproducibility
"""
function default_detector_model(;
    blur_fwhm::Float64=1.5,
    I0::Float64=1e5,
    electronic_noise_std::Float64=10.0,
    seed::Union{Nothing, Int}=nothing
)
    return DetectorModel(blur_fwhm, I0, electronic_noise_std, seed)
end

# =============================================================================
# Clinical mA/mAs Support
# =============================================================================
#
# These functions enable clinical dose-based noise specification by converting
# tube current (mA) to photon count (I0) using scanner geometry parameters.
#
# Physics:
#   I0 = flux_density × mA × t_proj × (1/SDD)² × A_det × η_det
#
# Where:
#   - flux_density: ~2×10⁶ photons/mA/mm²/s at 1m (CatSim/XCIST standard)
#   - t_proj: Exposure time per projection (rotation_time / n_views)
#   - SDD: Source-to-detector distance
#   - A_det: Detector element area at detector
#   - η_det: Detector quantum efficiency
#
# Reference: XCIST/CatSim (De Man et al., Phys Med Biol, 2022)
# =============================================================================

"""
    CATSIM_FLUX_DENSITY

Reference photon flux density from CatSim/XCIST at 1 meter distance.
Units: photons / (mA × mm² × s)

This value incorporates:
- X-ray tube electron-to-photon conversion efficiency (~1%)
- Geometric solid angle factors
- Spectral averaging

Reference: XCIST PMC10151073, typical diagnostic CT tube output.
"""
const CATSIM_FLUX_DENSITY = 2.0e6  # photons/mA/mm²/s at 1m

"""
    mA_to_I0(mA; SDD_mm, det_area_mm2, rotation_time_s=0.5, n_views=984, η_det=0.85) -> Float64

Convert clinical tube current (mA) to photon count per detector element per projection.

This function enables clinical dose specification by computing the expected photon
count I0 from tube current and scanner geometry, matching CatSim/XCIST conventions.

# Physics Model

The photon count per detector element is:

    I0 = Φ × mA × t_proj × (d_ref/SDD)² × A_det × η_det

Where:
- Φ = 2×10⁶ photons/mA/mm²/s (CatSim flux density at 1 meter)
- t_proj = rotation_time / n_views (exposure per projection)
- d_ref = 1000 mm (reference distance for flux density)
- SDD = source-to-detector distance
- A_det = detector element area at detector plane
- η_det = detector quantum efficiency

# Arguments
- `mA::Float64`: Tube current in milliamperes (typical: 100-740)

# Keyword Arguments
- `SDD_mm::Float64`: Source-to-detector distance in mm (required)
- `det_area_mm2::Float64`: Detector element area at detector plane in mm² (required)
- `rotation_time_s::Float64=0.5`: Gantry rotation time in seconds
- `n_views::Int=984`: Number of projection views per rotation
- `η_det::Float64=0.85`: Detector quantum efficiency (0-1)

# Returns
- `I0::Float64`: Photon count per detector element per projection

# Clinical Reference Values (GE Revolution Apex at 120 kVp, 0.5s rotation, 984 views)

| mA  | I0 (approx)  | Clinical Use |
|-----|--------------|--------------|
| 100 | ~145,000     | Ultra-low dose screening |
| 200 | ~290,000     | Low dose chest |
| 400 | ~580,000     | Standard body |
| 600 | ~870,000     | High quality |
| 740 | ~1,070,000   | Maximum at 120 kVp |

# Noise Relationship

Image noise scales as: σ ∝ 1/√(mA) ∝ 1/√I0

Doubling mA reduces noise by factor of √2 ≈ 1.41

# Example

```julia
# GE Revolution Apex geometry
SDD_mm = 1097.0  # Source-to-detector distance
det_area_mm2 = 1.84 * 1.10  # ~2.0 mm² at detector plane

# Standard chest protocol: 400 mA, 0.5s rotation
I0 = mA_to_I0(400.0; SDD_mm=SDD_mm, det_area_mm2=det_area_mm2)
# → ~580,000 photons/detector/projection

# Low dose: 100 mA
I0_low = mA_to_I0(100.0; SDD_mm=SDD_mm, det_area_mm2=det_area_mm2)
# → ~145,000 photons/detector/projection
```

# References

1. De Man B, et al. "XCIST—an open access x-ray/CT simulation toolkit."
   Phys Med Biol. 2022;67(18). PMC10151073

2. Hsieh J. "Computed Tomography: Principles, Design, Artifacts, and
   Recent Advances." 3rd ed. SPIE Press, 2015. Chapter 7.

See also: [`clinical_detector_model`](@ref), [`DetectorModel`](@ref)
"""
function mA_to_I0(mA::Float64;
                  SDD_mm::Float64,
                  det_area_mm2::Float64,
                  rotation_time_s::Float64=0.5,
                  n_views::Int=984,
                  η_det::Float64=0.85)

    # Validate inputs
    mA > 0 || throw(ArgumentError("mA must be positive (got $mA)"))
    SDD_mm > 0 || throw(ArgumentError("SDD_mm must be positive (got $SDD_mm)"))
    det_area_mm2 > 0 || throw(ArgumentError("det_area_mm2 must be positive (got $det_area_mm2)"))
    rotation_time_s > 0 || throw(ArgumentError("rotation_time_s must be positive (got $rotation_time_s)"))
    n_views > 0 || throw(ArgumentError("n_views must be positive (got $n_views)"))
    0 < η_det <= 1 || throw(ArgumentError("η_det must be in (0, 1] (got $η_det)"))

    # Time per projection (seconds)
    t_proj = rotation_time_s / n_views

    # Distance scaling: flux density at 1m reference
    # Inverse square law from reference distance (1000 mm)
    d_ref = 1000.0  # Reference distance in mm
    distance_factor = (d_ref / SDD_mm)^2

    # Calculate I0: photons per detector element per projection
    I0 = CATSIM_FLUX_DENSITY * mA * t_proj * distance_factor * det_area_mm2 * η_det

    return I0
end

"""
    mA_to_I0(mA, spec::AbstractScannerSpec; rotation_time_s=0.5, n_views=984, η_det=0.85) -> Float64

Convert mA to I0 using scanner specification for geometry parameters.

This convenience method extracts SDD and detector element area from
the scanner specification.

# Arguments
- `mA::Float64`: Tube current in milliamperes
- `spec::AbstractScannerSpec`: Scanner specification (e.g., GERevolutionApex())

# Keyword Arguments
- `rotation_time_s::Float64=0.5`: Gantry rotation time in seconds
- `n_views::Int=984`: Number of projection views per rotation
- `η_det::Float64=0.85`: Detector quantum efficiency

# Example

```julia
spec = GERevolutionApex()

# 400 mA standard protocol
I0 = mA_to_I0(400.0, spec; rotation_time_s=0.5)

# Create detector model with clinical mA
detector = clinical_detector_model(mA=400.0, scanner=spec)
```

See also: [`clinical_detector_model`](@ref)
"""
function mA_to_I0(mA::Float64, spec;
                  rotation_time_s::Float64=0.5,
                  n_views::Int=984,
                  η_det::Float64=0.85)

    # Extract geometry from scanner spec
    geom_spec = geometry(spec)
    det_spec = detector(spec)

    SDD_mm = geom_spec.sdd_mm[]

    # Compute detector element area at detector plane (not at isocenter)
    # The spec stores sizes at isocenter, need to scale by magnification
    magnification = SDD_mm / geom_spec.sid_mm[]
    col_size_at_det = det_spec.col_size_mm[] * magnification
    row_size_at_det = det_spec.row_size_mm[] * magnification
    det_area_mm2 = col_size_at_det * row_size_at_det

    return mA_to_I0(mA;
                    SDD_mm=SDD_mm,
                    det_area_mm2=det_area_mm2,
                    rotation_time_s=rotation_time_s,
                    n_views=n_views,
                    η_det=η_det)
end

"""
    clinical_detector_model(; mA, scanner, rotation_time_s=0.5, n_views=984, kwargs...) -> DetectorModel

Create a detector model with clinical mA specification.

This is the recommended way to specify noise levels in BasisSimulator when
you want to match clinical CT protocols. The mA setting is converted to
photon count (I0) using scanner geometry and CatSim-compatible physics.

# Arguments

All arguments are keyword-only.

- `mA::Float64`: Tube current in milliamperes (required)
- `scanner::AbstractScannerSpec`: Scanner specification (required)
- `rotation_time_s::Float64=0.5`: Gantry rotation time in seconds
- `n_views::Int=984`: Number of projection views per rotation
- `η_det::Float64=0.85`: Detector quantum efficiency
- `blur_fwhm::Float64=1.5`: Detector PSF FWHM in pixels
- `electronic_noise_std::Float64=10.0`: Electronic noise standard deviation
- `seed::Union{Nothing,Int}=nothing`: Random seed for reproducibility

# Returns

`DetectorModel` with I0 computed from mA and scanner geometry.

# Example

```julia
using BasisSimulator

spec = GERevolutionApex()

# Standard chest protocol: 400 mA, 0.5s rotation
detector_standard = clinical_detector_model(
    mA = 400.0,
    scanner = spec,
    rotation_time_s = 0.5
)
println("Standard I0: \$(detector_standard.I0)")  # ~580,000

# Low dose screening: 100 mA
detector_lowdose = clinical_detector_model(
    mA = 100.0,
    scanner = spec,
    rotation_time_s = 0.5
)
println("Low dose I0: \$(detector_lowdose.I0)")  # ~145,000

# Use in physics config
physics = default_physics_config(
    noise = detector_standard,
    fill_factor = fill_factor_standard(),
    flat_filter = flat_filter_al(3.0)
)
```

# Clinical Protocol Reference (GE Revolution Apex at 120 kVp)

| Protocol | mA | Rotation | mAs | Use Case |
|----------|-----|----------|-----|----------|
| Ultra-low dose | 50-100 | 0.5s | 25-50 | Lung screening |
| Low dose | 100-200 | 0.5s | 50-100 | Chest |
| Standard | 300-450 | 0.5s | 150-225 | Abdomen |
| High quality | 500-740 | 0.5s | 250-370 | Contrast studies |

See also: [`mA_to_I0`](@ref), [`default_detector_model`](@ref)
"""
function clinical_detector_model(;
    mA::Float64,
    scanner,
    rotation_time_s::Float64=0.5,
    n_views::Int=984,
    η_det::Float64=0.85,
    blur_fwhm::Float64=1.5,
    electronic_noise_std::Float64=10.0,
    seed::Union{Nothing, Int}=nothing
)
    # Compute I0 from mA and scanner geometry
    I0 = mA_to_I0(mA, scanner;
                  rotation_time_s=rotation_time_s,
                  n_views=n_views,
                  η_det=η_det)

    return DetectorModel(blur_fwhm, I0, electronic_noise_std, seed)
end

# Maximum kernel size for detector blur
const MAX_DETECTOR_BLUR_KERNEL_SIZE = 15

"""
    apply_detector_blur!(sinogram, model::DetectorModel) -> sinogram

Apply detector blur (PSF convolution) to sinogram (in-place, GPU-native).

Uses spatial domain convolution for GPU compatibility.
"""
function apply_detector_blur!(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    if model.blur_fwhm <= 0.0
        return sinogram
    end

    n_cols = size(sinogram, 1)
    n_rows = size(sinogram, 2)

    # Create 2D Gaussian blur kernel on CPU
    sigma = model.blur_fwhm / (2 * sqrt(2 * log(2)))
    extent = min(MAX_DETECTOR_BLUR_KERNEL_SIZE ÷ 2, max(1, ceil(Int, 3 * sigma)))
    kernel_size = 2 * extent + 1
    half_k = extent

    kernel_cpu = zeros(T, kernel_size, kernel_size)
    center = extent + 1

    for dy in -extent:extent
        for dx in -extent:extent
            kernel_cpu[center + dx, center + dy] = exp(-(dx^2 + dy^2) / (2 * sigma^2))
        end
    end
    kernel_cpu ./= sum(kernel_cpu)

    # Transfer kernel to GPU
    kernel = similar(sinogram, kernel_size, kernel_size)
    copyto!(kernel, kernel_cpu)

    # Output buffer
    output = similar(sinogram)

    # GPU-native spatial convolution
    AK.foreachindex(sinogram) do idx
        ci = CartesianIndices(sinogram)[idx]
        col, row, angle = Tuple(ci)

        # Apply kernel
        acc = zero(T)
        for dj in -half_k:half_k
            for di in -half_k:half_k
                src_col = clamp(col + di, 1, n_cols)
                src_row = clamp(row + dj, 1, n_rows)

                ki = di + half_k + 1
                kj = dj + half_k + 1

                acc += sinogram[src_col, src_row, angle] * kernel[ki, kj]
            end
        end

        output[idx] = acc
    end

    copyto!(sinogram, output)

    return sinogram
end

# Convenience wrapper that allocates
function apply_detector_blur(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    result = copy(sinogram)
    return apply_detector_blur!(result, model)
end

"""
    add_quantum_noise!(sinogram, model::DetectorModel) -> sinogram

Add Poisson quantum noise to sinogram projections (in-place, GPU-native).

# Algorithm

Implements Gaussian approximation of Poisson noise, valid for high photon counts:

1. Convert projection to intensity: λ = I₀ × exp(-p)
2. Apply Gaussian approximation: λ_noisy = λ + √λ × Z, where Z ~ N(0,1)
3. Clamp to positive values: λ_noisy = max(λ_noisy, 1)
4. Convert back to projection: p_noisy = -log(λ_noisy / I₀)

# Mathematical Formulation

For Poisson distribution with mean λ:
- E[N] = λ
- Var[N] = λ
- σ_N = √λ

The Gaussian approximation N ≈ λ + √λ × Z preserves:
- Mean: E[N] = λ
- Variance: Var[N] = (√λ)² = λ

In projection domain, noise standard deviation is:

    σ_p = |d p / d λ| × σ_λ = (1/λ) × √λ = 1/√λ

# Arguments

- `sinogram::AbstractArray{T,3}`: Projection data (cols × rows × angles) in
  attenuation units (line integrals, dimensionless)
- `model::DetectorModel`: Detector model containing I₀ (photon fluence) and seed

# Returns

- The modified sinogram with quantum noise added (same array, mutated)

# GPU Compatibility

- ✅ Metal (Apple Silicon)
- ✅ CUDA (NVIDIA)
- ✅ ROCm (AMD)
- ✅ CPU fallback

Random numbers are pre-generated on CPU and transferred to GPU.

# CatSim Compatibility

CatSim uses exact Poisson sampling via PTRD algorithm for all λ values, but
internally uses normal approximation for λ ≥ 10 (Detection_EI.py, rndpoi.c).

For CT-relevant photon counts (λ > 10³), both methods are statistically
indistinguishable with identical:
- Mean preservation (bias < 0.01%)
- Variance scaling (σ² = λ within < 1%)
- White noise power spectrum (flat NPS)
- Spatial uncorrelation (autocorr[lag>0] ≈ 0)

# Example

```julia
# Create projection data
sinogram = zeros(Float32, 512, 64, 360)  # Uniform field

# Add quantum noise with 10⁵ photons/detector
model = default_detector_model(I0=1e5, seed=42)
add_quantum_noise!(sinogram, model)

# Noise standard deviation in projection domain
# σ_p = 1/√(I₀ × exp(-p)) ≈ 0.00316 for p=0, I₀=10⁵
```

# See Also

- [`add_electronic_noise!`](@ref): Add Gaussian electronic noise
- [`apply_detector_model!`](@ref): Apply full detector model (blur + noise)
- [`default_detector_model`](@ref): Create detector model with default parameters
"""
function add_quantum_noise!(sinogram::AbstractArray{T,3}, model::DetectorModel;
                            ws_rand_cpu::Union{Nothing, Vector{T}}=nothing,
                            ws_rand_gpu=nothing,
                            ws_rng::Union{Nothing, MersenneTwister}=nothing) where T
    rng = if ws_rng !== nothing
        ws_rng
    elseif isnothing(model.seed)
        Random.default_rng()
    else
        MersenneTwister(model.seed)
    end

    n_elements = length(sinogram)
    I0 = T(model.I0)

    # Pre-generate Gaussian random numbers on CPU
    rand_cpu = ws_rand_cpu !== nothing ? ws_rand_cpu : Vector{T}(undef, n_elements)
    randn!(rng, rand_cpu)

    # Transfer to GPU (same type as sinogram)
    rand_gpu = ws_rand_gpu !== nothing ? ws_rand_gpu : similar(sinogram, n_elements)
    copyto!(rand_gpu, rand_cpu)

    # GPU-native noise application
    # Poisson noise with Gaussian approximation:
    # λ = I0 * exp(-sinogram), noisy_λ = λ + sqrt(λ) * randn
    # noisy_sinogram = -log(noisy_λ / I0)
    AK.foreachindex(sinogram) do idx
        # Convert to intensity (λ)
        λ = I0 * exp(-sinogram[idx])

        # Apply Gaussian approximation of Poisson noise
        # σ = sqrt(λ) for Poisson distribution
        λ_noisy = λ + sqrt(max(λ, T(1))) * rand_gpu[idx]

        # Clamp to positive
        λ_noisy = max(λ_noisy, T(1))

        # Convert back to attenuation
        sinogram[idx] = -log(λ_noisy / I0)
    end

    return sinogram
end

"""
    add_electronic_noise!(sinogram, model::DetectorModel) -> sinogram

Add Gaussian electronic noise to sinogram (in-place, GPU-native).

Electronic noise is additive in detector signal space, approximated
as additive noise in attenuation space scaled by I0.
"""
function add_electronic_noise!(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    if model.electronic_noise_std <= 0.0
        return sinogram
    end

    rng = isnothing(model.seed) ? Random.default_rng() : MersenneTwister(model.seed + 1)

    n_elements = length(sinogram)
    noise_scale = T(model.electronic_noise_std / model.I0)

    # Pre-generate Gaussian random numbers on CPU
    rand_cpu = randn(rng, T, n_elements)

    # Transfer to GPU (same type as sinogram)
    rand_gpu = similar(sinogram, n_elements)
    copyto!(rand_gpu, rand_cpu)

    # GPU-native noise application
    AK.foreachindex(sinogram) do idx
        sinogram[idx] += noise_scale * rand_gpu[idx]
    end

    return sinogram
end

# Convenience wrappers that allocate (for backward compatibility during transition)
function add_quantum_noise(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    result = copy(sinogram)
    return add_quantum_noise!(result, model)
end

function add_electronic_noise(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    result = copy(sinogram)
    return add_electronic_noise!(result, model)
end

"""
    apply_detector_model!(sinogram, model::DetectorModel) -> sinogram

Apply full detector model in-place: blur + quantum noise + electronic noise (GPU-native).

All effects are GPU-native using AcceleratedKernels.jl.
"""
function apply_detector_model!(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    # Apply in order: blur -> quantum noise -> electronic noise
    # All operations are now GPU-native
    apply_detector_blur!(sinogram, model)
    add_quantum_noise!(sinogram, model)
    add_electronic_noise!(sinogram, model)
    return sinogram
end

# Convenience wrapper that allocates
function apply_detector_model(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    result = copy(sinogram)
    return apply_detector_model!(result, model)
end

# =============================================================================
# Exports
# =============================================================================

export DetectorModel, default_detector_model
export CATSIM_FLUX_DENSITY, mA_to_I0, clinical_detector_model
export apply_detector_blur!, apply_detector_blur
export add_quantum_noise!, add_electronic_noise!
export add_quantum_noise, add_electronic_noise
export apply_detector_model!, apply_detector_model
export sim_detect, compute_detector_I0

"""
    compute_detector_I0(geom::CTGeometry, protocol::CTProtocol) -> Float64

Compute physics-based I₀ (photons per pixel per view) from scanner geometry
and acquisition protocol.

# Formula
    I₀ = flux_density × mA × (rotation_time / views) × pixel_area_mm² × (1000/SDD_mm)²

where pixel_area is at the detector plane (magnified from isocenter).

# Arguments
- `geom`: Scanner geometry (SDD, SAD, pixel_size in cm)
- `protocol`: CT protocol (flux_density, mA, rotation_time, views)
"""
function compute_detector_I0(geom::CTGeometry, protocol::CTProtocol)
    # Convert cm (CTGeometry) to mm (physics standard)
    SDD_mm = geom.SDD * 10.0
    SAD_mm = geom.SAD * 10.0

    # Pixel size at detector plane (magnified from isocenter)
    magnification = SDD_mm / SAD_mm
    pixel_col_det_mm = (geom.pixel_size * 10.0) * magnification
    pixel_row_det_mm = (geom.pixel_row_size * 10.0) * magnification
    pixel_area_mm2 = pixel_col_det_mm * pixel_row_det_mm

    # Time per view
    time_per_view = protocol.rotation_time / protocol.views

    # Inverse square law from reference distance (1000 mm)
    dist_factor = (1000.0 / SDD_mm)^2

    # I₀ = Flux × mA × TimePerView × Area × DistFactor
    return protocol.flux_density * protocol.mA * time_per_view * pixel_area_mm2 * dist_factor
end

"""
    sim_detect(sinogram, geom::CTGeometry, protocol::CTProtocol)

Apply physics-based photon statistics to a clean sinogram.

This function bridges the gap between geometric ray tracing and physical
data acquisition by converting attenuation line integrals into photon counts,
applying Poisson statistics, and converting back to attenuation.

# Algorithm
1. Calculates incident photon count (I0) from protocol and geometry.
   I0 = Flux × (mAs/views) × PixelArea × (1000/SDD)²
2. Applies Poisson noise (quantum noise) to the sinogram.

# Arguments
- `sinogram`: Input attenuation sinogram (clean). Not modified.
- `geom`: Scanner geometry (provides SDD, SAD, pixel size)
- `protocol`: CT protocol (provides mAs, kVp, flux)

# Returns
- A new sinogram array with quantum noise applied.
"""
function sim_detect(
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    protocol::CTProtocol
) where T
    # 1. Compute physics-based I0 using shared helper
    I0 = compute_detector_I0(geom, protocol)

    # 2. Apply Noise
    # Create a temporary detector model for noise application
    # We use 0.0 for other params as we only want quantum noise here
    model = DetectorModel(0.0, I0, 0.0, nothing)

    # Create copy and apply noise
    noisy_sino = copy(sinogram)
    add_quantum_noise!(noisy_sino, model)

    return noisy_sino
end
