# =============================================================================
# Calibration Pipeline
# =============================================================================
#
# Implements clinical CT calibration workflow:
# 1. Air scan simulation (I₀ reference)
# 2. Offset scan (dark current)
# 3. Gain/offset correction
# 4. Log transformation with proper normalization
#
# Reference: CatSim/XCIST signal processing chain
#
# =============================================================================

import AcceleratedKernels as AK

export simulate_air_scan, simulate_air_scan!
export simulate_offset_scan
export apply_calibration!, apply_calibration
export apply_log_transform!, apply_log_transform
export calibrate_sinogram!, calibrate_sinogram

# =============================================================================
# Air Scan Simulation
# =============================================================================

"""
    simulate_air_scan(geom; spectrum=nothing, physics=nothing)

Simulate an air scan (no phantom) to get I₀ reference for each detector pixel.

This represents the detector signal when X-rays pass through air only (no attenuation).
Used for gain correction: `normalized = raw / air`

# Arguments
- `geom`: CTGeometry with scanner parameters

# Keyword Arguments
- `spectrum`: Optional (energies, weights) tuple for polychromatic. If nothing, returns 1.0
- `physics`: Optional PhysicsConfig for bowtie filter, flat filter effects

# Returns
- Air scan array [n_cols, n_rows, n_angles] with transmitted intensity (not log-transformed)

# Example
```julia
geom = create_aquilion_one(n_angles=180, n_rows=32, n_cols=256)
air = simulate_air_scan(geom)

# With spectrum and physics
energies, weights = load_spectrum(120)
physics = default_physics_config(bowtie_filter=default_bowtie_filter())
air = simulate_air_scan(geom; spectrum=(energies, weights), physics=physics)
```
"""
function simulate_air_scan(
    geom::CTGeometry;
    spectrum::Union{Nothing, Tuple{Vector, Vector}} = nothing,
    physics::Union{Nothing, PhysicsConfig} = nothing
)
    T = Float32
    air = ones(T, geom.n_cols, geom.n_rows, geom.n_angles)
    simulate_air_scan!(air, geom; spectrum=spectrum, physics=physics)
    return air
end

"""
    simulate_air_scan!(air, geom; spectrum=nothing, physics=nothing)

In-place version of simulate_air_scan. See `simulate_air_scan` for details.
"""
function simulate_air_scan!(
    air::AbstractArray{T, 3},
    geom::CTGeometry;
    spectrum::Union{Nothing, Tuple{Vector, Vector}} = nothing,
    physics::Union{Nothing, PhysicsConfig} = nothing
) where T <: AbstractFloat

    # Start with unit intensity
    fill!(air, one(T))

    # If spectrum provided, compute spectral weighting
    if spectrum !== nothing
        energies, weights = spectrum
        weights_norm = T.(weights ./ sum(weights))

        # Air has negligible attenuation, so intensity ≈ Σ wₑ ≈ 1.0
        # But we keep per-energy weights for consistency
        # Note: In real systems, detector efficiency varies with energy
    end

    # Apply physics effects that affect air scan (bowtie, flat filter, DQE)
    if physics !== nothing
        # Bowtie filter affects air scan (position-dependent attenuation)
        if physics.bowtie_filter !== nothing
            apply_bowtie_filter!(air, physics.bowtie_filter, geom)
        end

        # Flat filter affects air scan
        if physics.flat_filter !== nothing
            apply_flat_filter!(air, physics.flat_filter, geom)
        end

        # Detector efficiency affects air scan
        if physics.detector_efficiency !== nothing
            apply_detector_efficiency!(air, physics.detector_efficiency, geom,
                physics.energy_keV)
        end
    end

    return air
end

# =============================================================================
# Offset Scan Simulation
# =============================================================================

"""
    simulate_offset_scan(geom; electronic_noise_sigma=0.0)

Simulate an offset scan (X-ray tube OFF) to measure dark current.

This represents the detector signal with no X-ray illumination - just electronic
noise and dark current. Used for offset correction: `corrected = raw - offset`

# Arguments
- `geom`: CTGeometry with scanner parameters

# Keyword Arguments
- `electronic_noise_sigma`: Standard deviation of electronic noise (default: 0.0)

# Returns
- Offset array [n_cols, n_rows, n_angles] with dark current values

# Example
```julia
geom = create_aquilion_one(n_angles=180, n_rows=32, n_cols=256)
offset = simulate_offset_scan(geom; electronic_noise_sigma=10.0)
```
"""
function simulate_offset_scan(
    geom::CTGeometry;
    electronic_noise_sigma::Real = 0.0
)
    T = Float32
    offset = zeros(T, geom.n_cols, geom.n_rows, geom.n_angles)

    if electronic_noise_sigma > 0
        # Add Gaussian electronic noise
        noise = randn(T, size(offset)) .* T(electronic_noise_sigma)
        offset .+= noise
    end

    return offset
end

# =============================================================================
# Calibration Correction
# =============================================================================

"""
    apply_calibration!(raw, air, offset)

Apply gain and offset correction to raw detector signal.

Implements the standard calibration formula:
    normalized = (raw - offset) / (air - offset)

# Arguments
- `raw`: Raw detector signal [n_cols, n_rows, n_angles] (modified in place)
- `air`: Air scan reference (same size as raw, or [n_cols, n_rows] for single reference)
- `offset`: Offset scan (same size as raw, or [n_cols, n_rows], or scalar)

# Returns
- Modified raw array with normalized values in [0, 1] range (ideally)

# Example
```julia
raw = forward_project(phantom, geom; output_mode=:intensity)
air = simulate_air_scan(geom)
offset = simulate_offset_scan(geom)
apply_calibration!(raw, air, offset)
```
"""
function apply_calibration!(
    raw::AbstractArray{T, 3},
    air::AbstractArray{T},
    offset::Union{AbstractArray{T}, Real}
) where T <: AbstractFloat

    n_cols, n_rows, n_angles = size(raw)
    eps = T(1e-10)  # Prevent division by zero

    # Handle different offset dimensions
    if offset isa Real
        offset_val = T(offset)

        if ndims(air) == 3
            # Full 3D air scan
            AK.foreachindex(raw) do idx
                air_val = max(air[idx] - offset_val, eps)
                raw[idx] = (raw[idx] - offset_val) / air_val
            end
        else
            # 2D air scan (same for all angles)
            AK.foreachindex(raw) do idx
                ci = CartesianIndices(raw)[idx]
                col, row, angle = Tuple(ci)
                air_val = max(air[col, row] - offset_val, eps)
                raw[idx] = (raw[idx] - offset_val) / air_val
            end
        end
    elseif ndims(offset) == 2
        # 2D offset
        if ndims(air) == 3
            AK.foreachindex(raw) do idx
                ci = CartesianIndices(raw)[idx]
                col, row, angle = Tuple(ci)
                off = offset[col, row]
                air_val = max(air[idx] - off, eps)
                raw[idx] = (raw[idx] - off) / air_val
            end
        else
            AK.foreachindex(raw) do idx
                ci = CartesianIndices(raw)[idx]
                col, row, angle = Tuple(ci)
                off = offset[col, row]
                air_val = max(air[col, row] - off, eps)
                raw[idx] = (raw[idx] - off) / air_val
            end
        end
    else
        # Full 3D offset
        if ndims(air) == 3
            AK.foreachindex(raw) do idx
                air_val = max(air[idx] - offset[idx], eps)
                raw[idx] = (raw[idx] - offset[idx]) / air_val
            end
        else
            AK.foreachindex(raw) do idx
                ci = CartesianIndices(raw)[idx]
                col, row, angle = Tuple(ci)
                off = offset[idx]
                air_val = max(air[col, row] - off, eps)
                raw[idx] = (raw[idx] - off) / air_val
            end
        end
    end

    return raw
end

"""
    apply_calibration(raw, air, offset)

Non-mutating version of apply_calibration!. See `apply_calibration!` for details.
"""
function apply_calibration(
    raw::AbstractArray{T, 3},
    air::AbstractArray{T},
    offset::Union{AbstractArray{T}, Real}
) where T <: AbstractFloat
    result = similar(raw)
    copyto!(result, raw)
    return apply_calibration!(result, air, offset)
end

# =============================================================================
# Log Transform
# =============================================================================

"""
    apply_log_transform!(normalized)

Apply negative log transform to convert normalized intensity to line integrals.

Implements: sinogram = -log(normalized)

# Arguments
- `normalized`: Normalized detector signal in (0, 1] range (modified in place)

# Returns
- Modified array with line integral values (sinogram)

# Notes
- Values ≤ 0 are clamped to eps to avoid log(0)
- Output is in line integral space (attenuation × path length)
"""
function apply_log_transform!(
    normalized::AbstractArray{T, 3}
) where T <: AbstractFloat

    eps = T(1e-10)

    AK.foreachindex(normalized) do idx
        normalized[idx] = -log(max(normalized[idx], eps))
    end

    return normalized
end

"""
    apply_log_transform(normalized)

Non-mutating version of apply_log_transform!.
"""
function apply_log_transform(
    normalized::AbstractArray{T, 3}
) where T <: AbstractFloat
    result = similar(normalized)
    copyto!(result, normalized)
    return apply_log_transform!(result)
end

# =============================================================================
# Combined Calibration Pipeline
# =============================================================================

"""
    calibrate_sinogram!(raw, air, offset)

Apply full calibration pipeline: offset correction, gain normalization, and log transform.

This is the standard CT preprocessing pipeline:
1. Offset correction: raw - offset
2. Gain normalization: (raw - offset) / (air - offset)
3. Log transform: -log(normalized)

# Arguments
- `raw`: Raw detector signal [n_cols, n_rows, n_angles] (modified in place)
- `air`: Air scan reference
- `offset`: Offset scan (dark current)

# Returns
- Calibrated sinogram (line integrals)

# Example
```julia
# Full calibration pipeline
raw = forward_project(phantom, geom; output_mode=:intensity)
air = simulate_air_scan(geom)
offset = simulate_offset_scan(geom)
sinogram = calibrate_sinogram!(raw, air, offset)

# Then reconstruct
recon = fdk_reconstruct(sinogram, geom, volume_size)
```
"""
function calibrate_sinogram!(
    raw::AbstractArray{T, 3},
    air::AbstractArray{T},
    offset::Union{AbstractArray{T}, Real}
) where T <: AbstractFloat

    apply_calibration!(raw, air, offset)
    apply_log_transform!(raw)

    return raw
end

"""
    calibrate_sinogram(raw, air, offset)

Non-mutating version of calibrate_sinogram!.
"""
function calibrate_sinogram(
    raw::AbstractArray{T, 3},
    air::AbstractArray{T},
    offset::Union{AbstractArray{T}, Real}
) where T <: AbstractFloat
    result = similar(raw)
    copyto!(result, raw)
    return calibrate_sinogram!(result, air, offset)
end

# =============================================================================
# Raw Intensity Forward Projection Mode
# =============================================================================

"""
    forward_project_intensity(volume_or_mask, geom; kwargs...)

Forward project returning raw transmitted intensity (NOT log-transformed).

This is the same as `forward_project` but returns I (intensity) instead of -log(I/I₀).
Use this when you want to apply your own calibration pipeline.

# Arguments
Same as `forward_project`

# Returns
- Transmitted intensity array [n_cols, n_rows, n_angles] in (0, 1] range

# Example
```julia
# Get raw intensity
intensity = forward_project_intensity(phantom.mask, geom;
    energies=energies, weights=weights, materials=materials)

# Apply custom calibration
air = simulate_air_scan(geom)
apply_calibration!(intensity, air, 0.0)
apply_log_transform!(intensity)  # Now it's a sinogram
```
"""
function forward_project_intensity(
    volume_or_mask::AbstractArray,
    geom::CTGeometry;
    energy::Union{Nothing, Real} = nothing,
    energies::Union{Nothing, Vector} = nothing,
    weights::Union{Nothing, Vector} = nothing,
    materials::Union{Nothing, Vector} = nothing,
    physics::Union{Nothing, PhysicsConfig} = nothing
)
    T = eltype(volume_or_mask) <: AbstractFloat ? eltype(volume_or_mask) : Float32

    if eltype(volume_or_mask) <: AbstractFloat
        # Direct volume input - compute exp(-line_integral)
        sinogram = forward_project(volume_or_mask, geom)

        # Convert line integral to intensity: I = exp(-sinogram)
        intensity = similar(sinogram)
        AK.foreachindex(sinogram) do idx
            intensity[idx] = exp(-sinogram[idx])
        end

        # Apply physics effects if specified (in intensity domain)
        if physics !== nothing
            # Note: Some physics effects work in intensity domain, some in sinogram domain
            # For now, convert to sinogram, apply effects, convert back
            # This is a simplification - proper implementation would handle each effect appropriately
        end

        return intensity

    elseif eltype(volume_or_mask) == UInt8
        # Mask input - polychromatic returns intensity directly before log
        return _forward_project_intensity_poly(volume_or_mask, geom,
            energies, weights, materials, physics)
    else
        error("volume_or_mask must be Float32/Float64 (μ volume) or UInt8 (material mask)")
    end
end

"""Internal: Polychromatic forward projection returning intensity"""
function _forward_project_intensity_poly(
    mask::AbstractArray{UInt8, 3},
    geom::CTGeometry,
    energies::Vector,
    weights::Vector,
    materials::Vector,
    physics::Union{Nothing, PhysicsConfig}
)
    T = Float32
    n_energies = length(energies)

    # Normalize weights
    weights_norm = T.(weights ./ sum(weights))

    # Allocate arrays
    μ_volume = similar(mask, T, size(mask))
    sino_mono = similar(mask, T, geom.n_cols, geom.n_rows, geom.n_angles)
    I_transmitted = similar(sino_mono)
    fill!(I_transmitted, zero(T))

    # Loop over energies (Beer-Lambert)
    for e_idx in 1:n_energies
        # Create μ volume for this energy
        create_μ_volume!(μ_volume, mask, materials, energies[e_idx])

        # Forward project at this energy
        fill!(sino_mono, zero(T))
        siddon_forward_project!(sino_mono, μ_volume, geom)

        # Accumulate: I += w × exp(-line_integral)
        w = weights_norm[e_idx]
        AK.foreachindex(I_transmitted) do idx
            I_transmitted[idx] += w * exp(-sino_mono[idx])
        end
    end

    # Apply physics effects in intensity domain if specified
    if physics !== nothing
        # Bowtie and flat filter affect intensity
        if physics.bowtie_filter !== nothing
            apply_bowtie_filter!(I_transmitted, physics.bowtie_filter, geom)
        end
        if physics.flat_filter !== nothing
            apply_flat_filter!(I_transmitted, physics.flat_filter, geom)
        end
    end

    return I_transmitted
end
