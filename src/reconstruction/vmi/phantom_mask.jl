# =============================================================================
# Phantom-mask helpers for image-domain post-processing — e.g. masking the bright
# phantom-air ring that Mono+ frequency splitting leaves at a high-contrast
# phantom boundary:
#
#     eroded = BS.erode_mask_3d(mask; erode_px = 8.0)
#     BS.apply_mono_plus!(ws, vols, energies; ..., phantom_mask = eroded)
# =============================================================================

# =============================================================================
# Mask erosion — FFT-Gaussian smooth + tight threshold
# =============================================================================

"""
    erode_mask_2d(mask2d::AbstractMatrix{Bool}; erode_px::Real) -> BitMatrix

"Soft" erosion of a 2D Bool mask by approximately `erode_px` voxels.
Implementation: FFT-Gaussian-blur the float-cast mask with σ = `erode_px`
pixels, then threshold at 0.999 — interior voxels (whose neighborhood is
fully inside the mask) survive; edge voxels (whose neighborhood is
partially outside) get rejected.

`erode_px ≤ 0` returns a copy of the input mask (no-op).
"""
function erode_mask_2d(mask2d::AbstractMatrix{Bool}; erode_px::Real)
    erode_px > 0 || return copy(mask2d)
    nx, ny = size(mask2d)
    σ = Float64(erode_px); σ² = σ^2
    fx = [min(i - 1, nx - (i - 1)) / nx for i in 1:nx]
    fy = [min(j - 1, ny - (j - 1)) / ny for j in 1:ny]
    kernel = [exp(-2π^2 * σ² * (fx[i]^2 + fy[j]^2)) for i in 1:nx, j in 1:ny]
    blurred = real.(FFTW.ifft(FFTW.fft(Float64.(mask2d)) .* kernel))
    blurred .≥ 0.999
end

"""
    erode_mask_3d(mask3d::AbstractArray{Bool, 3}; erode_px::Real,
                  broadcast_in_z::Bool = true) -> BitArray{3}

Per-slice 2D erosion broadcast across all z slices.

`broadcast_in_z = true` (default): erode the FIRST slice only and
replicate (correct for z-invariant masks). Faster.

`broadcast_in_z = false`: erode each slice independently.  Use when the
mask varies in z.
"""
function erode_mask_3d(
        mask3d::AbstractArray{Bool, 3};
        erode_px::Real,
        broadcast_in_z::Bool = true,
    )
    nx, ny, nz = size(mask3d)
    out = falses(nx, ny, nz)
    if broadcast_in_z
        slice = erode_mask_2d(view(mask3d, :, :, 1); erode_px = erode_px)
        for k in 1:nz
            out[:, :, k] .= slice
        end
    else
        for k in 1:nz
            out[:, :, k] .= erode_mask_2d(view(mask3d, :, :, k); erode_px = erode_px)
        end
    end
    out
end


export erode_mask_2d,
       erode_mask_3d
