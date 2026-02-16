"""
    Geometry/Affine.jl

Affine transforms between phantom, reconstruction, and world coordinate systems.

Both phantom and recon grids are centered at isocenter (0,0,0). The mapping is
a pure scale+translate (diagonal affine, no rotation).

Used for resampling ground truth phantom labels onto the reconstruction grid
for ROI analysis and segmentation evaluation.
"""

# =============================================================================
# Affine Transform Construction
# =============================================================================

"""
    phantom_to_world_affine(phantom::Phantom) -> Matrix{Float64}

4×4 affine matrix mapping 0-indexed phantom voxel `(i,j,k)` to world `(x,y,z)` in cm.

The transform encodes:
- Scale: `phantom.voxel_size` (cm per voxel)
- Translation: `phantom.origin` (center of first voxel in world coordinates)

# Example
```julia
A = phantom_to_world_affine(phantom)
# World coordinate of voxel (0,0,0):
world = A * [0, 0, 0, 1]  # == [origin_x, origin_y, origin_z, 1]
```
"""
function phantom_to_world_affine(phantom::Phantom)
    sx, sy, sz = phantom.voxel_size
    tx, ty, tz = phantom.origin

    return Float64[
        sx  0   0   tx
        0   sy  0   ty
        0   0   sz  tz
        0   0   0   1
    ]
end

"""
    recon_to_world_affine(geom::CTGeometry, matrix_size) -> Matrix{Float64}

4×4 affine matrix mapping 0-indexed reconstruction voxel `(i,j,k)` to world `(x,y,z)` in cm.

`matrix_size` is `(nx, ny, nz)` — the reconstruction volume dimensions.

The reconstruction grid is centered at isocenter with voxel size `geom.fov ./ matrix_size`.

# Example
```julia
A = recon_to_world_affine(geom, (512, 512, 64))
# World coordinate of first voxel:
world = A * [0, 0, 0, 1]  # == [-fov_x/2 + vx/2, -fov_y/2 + vy/2, -fov_z/2 + vz/2, 1]
```
"""
function recon_to_world_affine(geom::CTGeometry, matrix_size)
    nx, ny, nz = matrix_size
    fov_x, fov_y, fov_z = geom.fov

    sx = fov_x / nx
    sy = fov_y / ny
    sz = fov_z / nz

    tx = -fov_x / 2 + sx / 2
    ty = -fov_y / 2 + sy / 2
    tz = -fov_z / 2 + sz / 2

    return Float64[
        sx  0   0   tx
        0   sy  0   ty
        0   0   sz  tz
        0   0   0   1
    ]
end

# =============================================================================
# Resampling
# =============================================================================

"""
    resample_to_recon(phantom, geom::CTGeometry, matrix_size; method=:nearest) -> Array{UInt8,3}

Resample phantom labels onto the reconstruction grid.

For each reconstruction voxel, computes its world coordinate, maps to
the continuous phantom voxel index, and samples `phantom.mask`.

Returns a `UInt8` array of size `matrix_size` with resampled phantom labels.
CPU-only (one-time operation).

# Arguments
- `phantom::Phantom`: Source phantom (mask can be on GPU — will be pulled to CPU)
- `geom::CTGeometry`: Reconstruction geometry
- `matrix_size`: `(nx, ny, nz)` reconstruction volume dimensions

# Keyword Arguments
- `method::Symbol = :nearest`: Interpolation method
  - `:nearest` — nearest-neighbor (preserves label integrity, default)
  - `:linear` — trilinear interpolation (returns `Float32` array; useful for continuous volumes)

# Example
```julia
ground_truth = resample_to_recon(phantom, geom, (350, 350, 128))
# ground_truth is UInt8, same shape as reconstruction volume
# Overlay: heatmap(ground_truth[:,:,64]) vs heatmap(recon[:,:,64])
```
"""
function resample_to_recon(phantom::Phantom, geom::CTGeometry, matrix_size;
                           method::Symbol = :nearest)

    # Pull mask to CPU if on GPU
    mask_cpu = Array(phantom.mask)
    pnx, pny, pnz = size(mask_cpu)

    nx, ny, nz = matrix_size

    # Precompute affine parameters (avoid matrix multiply per voxel)
    # Recon voxel (i,j,k) → world: world = recon_origin + (i,j,k) .* recon_voxel
    fov_x, fov_y, fov_z = geom.fov
    rvx = fov_x / nx
    rvy = fov_y / ny
    rvz = fov_z / nz
    rox = -fov_x / 2 + rvx / 2
    roy = -fov_y / 2 + rvy / 2
    roz = -fov_z / 2 + rvz / 2

    # World → phantom voxel: pi = (world - phantom_origin) / phantom_voxel_size
    pvx, pvy, pvz = phantom.voxel_size
    pox, poy, poz = phantom.origin
    inv_pvx = 1.0 / pvx
    inv_pvy = 1.0 / pvy
    inv_pvz = 1.0 / pvz

    if method == :nearest
        out = zeros(UInt8, nx, ny, nz)

        @inbounds for k in 0:(nz-1)
            wz = roz + k * rvz
            pzi = (wz - poz) * inv_pvz
            pk = round(Int, pzi) + 1  # 0-indexed → 1-indexed
            (pk < 1 || pk > pnz) && continue

            for j in 0:(ny-1)
                wy = roy + j * rvy
                pyi = (wy - poy) * inv_pvy
                pj = round(Int, pyi) + 1
                (pj < 1 || pj > pny) && continue

                for i in 0:(nx-1)
                    wx = rox + i * rvx
                    pxi = (wx - pox) * inv_pvx
                    pi_idx = round(Int, pxi) + 1
                    (pi_idx < 1 || pi_idx > pnx) && continue

                    out[i+1, j+1, k+1] = mask_cpu[pi_idx, pj, pk]
                end
            end
        end

        return out

    elseif method == :linear
        out = zeros(Float32, nx, ny, nz)

        @inbounds for k in 0:(nz-1)
            wz = roz + k * rvz
            pzi = (wz - poz) * inv_pvz
            kz0 = floor(Int, pzi)
            kz1 = kz0 + 1
            fz = Float32(pzi - kz0)
            kz0 += 1; kz1 += 1  # 0-indexed → 1-indexed
            (kz1 < 1 || kz0 > pnz) && continue

            for j in 0:(ny-1)
                wy = roy + j * rvy
                pyi = (wy - poy) * inv_pvy
                jy0 = floor(Int, pyi)
                jy1 = jy0 + 1
                fy = Float32(pyi - jy0)
                jy0 += 1; jy1 += 1
                (jy1 < 1 || jy0 > pny) && continue

                for i in 0:(nx-1)
                    wx = rox + i * rvx
                    pxi = (wx - pox) * inv_pvx
                    ix0 = floor(Int, pxi)
                    ix1 = ix0 + 1
                    fx = Float32(pxi - ix0)
                    ix0 += 1; ix1 += 1
                    (ix1 < 1 || ix0 > pnx) && continue

                    # Clamp to valid range
                    ix0c = clamp(ix0, 1, pnx)
                    ix1c = clamp(ix1, 1, pnx)
                    jy0c = clamp(jy0, 1, pny)
                    jy1c = clamp(jy1, 1, pny)
                    kz0c = clamp(kz0, 1, pnz)
                    kz1c = clamp(kz1, 1, pnz)

                    # Trilinear interpolation
                    c000 = Float32(mask_cpu[ix0c, jy0c, kz0c])
                    c100 = Float32(mask_cpu[ix1c, jy0c, kz0c])
                    c010 = Float32(mask_cpu[ix0c, jy1c, kz0c])
                    c110 = Float32(mask_cpu[ix1c, jy1c, kz0c])
                    c001 = Float32(mask_cpu[ix0c, jy0c, kz1c])
                    c101 = Float32(mask_cpu[ix1c, jy0c, kz1c])
                    c011 = Float32(mask_cpu[ix0c, jy1c, kz1c])
                    c111 = Float32(mask_cpu[ix1c, jy1c, kz1c])

                    c00 = c000 * (1 - fx) + c100 * fx
                    c01 = c001 * (1 - fx) + c101 * fx
                    c10 = c010 * (1 - fx) + c110 * fx
                    c11 = c011 * (1 - fx) + c111 * fx

                    c0 = c00 * (1 - fy) + c10 * fy
                    c1 = c01 * (1 - fy) + c11 * fy

                    out[i+1, j+1, k+1] = c0 * (1 - fz) + c1 * fz
                end
            end
        end

        return out
    else
        error("Unknown interpolation method :$method. Use :nearest or :linear.")
    end
end

# =============================================================================
# Exports
# =============================================================================

export phantom_to_world_affine, recon_to_world_affine, resample_to_recon
