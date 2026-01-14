# =============================================================================
# FDK Reconstruction (Feldkamp-Davis-Kress Algorithm)
# =============================================================================
#
# Complete FDK cone-beam CT reconstruction:
#   1. Cosine weighting (in Filtering.jl)
#   2. Ramp filtering (in Filtering.jl)
#   3. Weighted backprojection (in Backprojection.jl)
#
# Reference:
#   - Feldkamp, Davis, Kress (1984) "Practical cone-beam algorithm"
#   - TIGRE: CERN/TIGRE
#
# =============================================================================

export fdk_reconstruct

"""
    fdk_reconstruct(sinogram, geom, volume_size; filter=RampFilter(), cutoff=1.0)

Full FDK cone-beam CT reconstruction.

# Arguments
- `sinogram`: Raw sinogram [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters
- `volume_size`: (nx, ny, nz) output volume dimensions
- `filter`: Filter type (RampFilter, SheppLoganFilter, CosineFilter, HammingFilter, HannFilter)
- `cutoff`: Frequency cutoff as fraction of Nyquist (0-1)

# Returns
Reconstructed volume [nx, ny, nz]

# Example
```julia
# Basic reconstruction
recon = fdk_reconstruct(sinogram, geom, (128, 128, 64))

# With Shepp-Logan filter and 80% cutoff
recon = fdk_reconstruct(sinogram, geom, (128, 128, 64);
                        filter=SheppLoganFilter(), cutoff=0.8)
```

# Pipeline
1. Copy sinogram (to avoid modifying input)
2. Apply cosine weighting for cone-beam geometry
3. Apply ramp filter in Fourier domain
4. Weighted backprojection with FDK distance weights
"""
function fdk_reconstruct(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int};
    filter::FilterType = RampFilter(),
    cutoff::Float64 = 1.0
) where T <: AbstractFloat

    # Step 1: Filter sinogram (includes cosine weighting)
    # Note: FFTW requires CPU arrays, so we convert if needed
    sinogram_cpu = Array(sinogram)  # No-op if already CPU array
    filtered_cpu = filter_sinogram(sinogram_cpu, geom; filter=filter, cutoff=cutoff)

    # Step 2: Transfer filtered sinogram back to same device as input
    filtered = similar(sinogram)
    copyto!(filtered, filtered_cpu)

    # Step 3: Backproject
    volume = backproject(filtered, geom, volume_size)

    return volume
end

"""
    fdk_reconstruct(sinogram, geom, volume_size, fov; filter=RampFilter(), cutoff=1.0)

FDK reconstruction with explicit FOV specification.

Use this when you want the reconstruction FOV to differ from the geometry FOV.

# Arguments
- `sinogram`: Raw sinogram [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters
- `volume_size`: (nx, ny, nz) output volume dimensions
- `fov`: (fx, fy, fz) reconstruction field of view in cm
- `filter`: Filter type
- `cutoff`: Frequency cutoff

# Returns
Reconstructed volume [nx, ny, nz]
"""
function fdk_reconstruct(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int},
    fov::NTuple{3, Float64};
    filter::FilterType = RampFilter(),
    cutoff::Float64 = 1.0
) where T <: AbstractFloat

    # Create a copy of geometry with modified FOV
    geom_fov = CTGeometry(
        geom.SAD, geom.SDD,
        geom.n_angles, geom.n_rows, geom.n_cols, geom.pixel_size,
        geom.angles, geom.source_positions, geom.detector_centers,
        geom.detector_u, geom.detector_v,
        fov  # Use specified FOV
    )

    return fdk_reconstruct(sinogram, geom_fov, volume_size; filter=filter, cutoff=cutoff)
end
