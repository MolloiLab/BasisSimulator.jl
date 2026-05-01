"""
Phantom-mask helpers for image-domain post-processing — primarily used
to mask the bright phantom-air ring artifact left by Mono+ frequency
splitting at the high-contrast phantom boundary.

# Public API
- [`resample_phantom_mask_to_recon`](@ref) — nearest-neighbor sample
  the simulation phantom mask into recon space (2D in-plane, broadcast
  in z for z-invariant phantoms).
- [`erode_mask_2d`](@ref), [`erode_mask_3d`](@ref) — FFT-Gaussian +
  tight-threshold "soft" erosion in voxels.

# Typical usage
```julia
base   = BS.resample_phantom_mask_to_recon(
            phantom.mask, phantom.voxel_size, recon_size, recon_fov_cm)
eroded = BS.erode_mask_3d(base; erode_px = 8.0)

BS.apply_mono_plus!(ws, vols, energies; ..., phantom_mask = eroded)
```
"""

# =============================================================================
# Phantom mask resampling
# =============================================================================

"""
    resample_phantom_mask_to_recon(
        phantom_mask_cpu::AbstractArray{<:Integer, 3},
        phantom_voxel_size::NTuple{3, <:Real},
        recon_size::NTuple{3, Int},
        recon_fov_cm::Real;
        phantom_origin_cm::Union{Nothing, NTuple{2, Real}} = nothing,
        recon_origin_cm::Union{Nothing,  NTuple{2, Real}} = nothing,
        broadcast_in_z::Bool = true,
    ) -> BitArray{3}

Resample a `phantom_mask_cpu` (CPU integer-labeled material mask)
into recon-space binary mask: any non-zero voxel = phantom material,
zero = air.

`broadcast_in_z = true` (default) builds the 2D mask from the central
phantom z-slice and replicates it across all `recon_size[3]` slices —
correct for z-invariant phantoms (Gammex 472 rods).  Set to `false` to
sample per-slice (slower; only needed for z-varying phantoms).

Both phantom and recon are assumed centered at `(0, 0)` in plane unless
explicit origins are passed.
"""
function resample_phantom_mask_to_recon(
        phantom_mask_cpu::AbstractArray{<:Integer, 3},
        phantom_voxel_size::NTuple{3, <:Real},
        recon_size::NTuple{3, Int},
        recon_fov_cm::Real;
        phantom_origin_cm::Union{Nothing, NTuple{2, Real}} = nothing,
        recon_origin_cm::Union{Nothing,  NTuple{2, Real}} = nothing,
        broadcast_in_z::Bool = true,
    )
    nx, ny, nz = recon_size
    pnx, pny, pnz = size(phantom_mask_cpu)
    rx = Float64(recon_fov_cm) / nx                     # cm/pixel (recon)
    px = Float64(phantom_voxel_size[1])                 # cm/pixel (phantom)
    py = Float64(phantom_voxel_size[2])

    cx_p = phantom_origin_cm === nothing ? pnx / 2 + 0.5 :
           pnx / 2 + 0.5 - Float64(phantom_origin_cm[1]) / px
    cy_p = phantom_origin_cm === nothing ? pny / 2 + 0.5 :
           pny / 2 + 0.5 - Float64(phantom_origin_cm[2]) / py
    cx_r = recon_origin_cm   === nothing ? nx  / 2 + 0.5 :
           nx  / 2 + 0.5 - Float64(recon_origin_cm[1]) / rx
    cy_r = recon_origin_cm   === nothing ? ny  / 2 + 0.5 :
           ny  / 2 + 0.5 - Float64(recon_origin_cm[2]) / rx

    function _sample_2d(z_p_idx::Int)
        slice2d = view(phantom_mask_cpu, :, :, z_p_idx)
        out2d = falses(nx, ny)
        for j in 1:ny, i in 1:nx
            x_cm = (i - cx_r) * rx
            y_cm = (j - cy_r) * rx
            ip = round(Int, cx_p + x_cm / px)
            jp = round(Int, cy_p + y_cm / py)
            if 1 ≤ ip ≤ pnx && 1 ≤ jp ≤ pny
                out2d[i, j] = slice2d[ip, jp] > 0
            end
        end
        out2d
    end

    mask3d = falses(nx, ny, nz)
    if broadcast_in_z
        z_mid_p = pnz ÷ 2 + 1
        slice = _sample_2d(z_mid_p)
        for k in 1:nz
            mask3d[:, :, k] .= slice
        end
    else
        # Map each recon z to the corresponding phantom z.
        pz_per_rz = pnz / nz
        for k in 1:nz
            zp = clamp(round(Int, (k - 0.5) * pz_per_rz + 0.5), 1, pnz)
            mask3d[:, :, k] .= _sample_2d(zp)
        end
    end
    mask3d
end


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
replicate (correct for z-invariant masks built via
[`resample_phantom_mask_to_recon`](@ref) with `broadcast_in_z = true`).
Faster.

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


export resample_phantom_mask_to_recon,
       erode_mask_2d,
       erode_mask_3d
