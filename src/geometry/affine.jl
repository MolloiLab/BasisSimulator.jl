"""
    src/geometry/affine.jl

Affine transforms between phantom, reconstruction, and world coordinate
systems.  Both phantom and recon grids are centered at isocenter `(0,0,0)`;
the mapping is a pure scale+translate (diagonal affine, no rotation).

Used to resample ground-truth phantom labels onto the reconstruction grid
for ROI analysis and segmentation evaluation — see nb05's `:nearest` /
`:linear` round-trip and nb07's pure-material-VMI ROI overlay.

BasisSim-original — standard image-resampling math (trilinear interpolation
of a centered scale+translate map), not a port of any upstream package.
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
function resample_to_recon(
        phantom::Phantom, geom::CTGeometry, matrix_size;
        method::Symbol = :nearest
    )

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
        out = zeros(eltype(mask_cpu), nx, ny, nz)

        @inbounds for k in 0:(nz - 1)
            wz = roz + k * rvz
            pzi = (wz - poz) * inv_pvz
            pk = round(Int, pzi) + 1  # 0-indexed → 1-indexed
            (pk < 1 || pk > pnz) && continue

            for j in 0:(ny - 1)
                wy = roy + j * rvy
                pyi = (wy - poy) * inv_pvy
                pj = round(Int, pyi) + 1
                (pj < 1 || pj > pny) && continue

                for i in 0:(nx - 1)
                    wx = rox + i * rvx
                    pxi = (wx - pox) * inv_pvx
                    pi_idx = round(Int, pxi) + 1
                    (pi_idx < 1 || pi_idx > pnx) && continue

                    out[i + 1, j + 1, k + 1] = mask_cpu[pi_idx, pj, pk]
                end
            end
        end

        return out

    elseif method == :linear
        out = zeros(Float32, nx, ny, nz)

        @inbounds for k in 0:(nz - 1)
            wz = roz + k * rvz
            pzi = (wz - poz) * inv_pvz
            kz0 = floor(Int, pzi)
            kz1 = kz0 + 1
            fz = Float32(pzi - kz0)
            kz0 += 1; kz1 += 1  # 0-indexed → 1-indexed
            (kz1 < 1 || kz0 > pnz) && continue

            for j in 0:(ny - 1)
                wy = roy + j * rvy
                pyi = (wy - poy) * inv_pvy
                jy0 = floor(Int, pyi)
                jy1 = jy0 + 1
                fy = Float32(pyi - jy0)
                jy0 += 1; jy1 += 1
                (jy1 < 1 || jy0 > pny) && continue

                for i in 0:(nx - 1)
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

                    out[i + 1, j + 1, k + 1] = c0 * (1 - fz) + c1 * fz
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

# =============================================================================
# Box-average a continuous field onto the reconstruction grid
# =============================================================================

"""
    resample_field_to_recon(field, voxel_size, origin, geom, matrix_size; outside = 0) -> Array{Float32, 3}

The exact box average of a continuous field — a material fraction, a density — onto the
reconstruction grid: every output voxel is the mean of the field over that voxel's own footprint,
from the axis-aligned overlap of the two grids. This is what a 0.2 mm truth has to become on
0.625 mm slices for its partial volume to be the partial volume a reconstruction sees; point
sampling ([`resample_to_recon`](@ref) with `:linear`) reads the field at the voxel centre and
misses it.

`field` is `(nx, ny, nz)` on a grid with `voxel_size` and `origin` in cm, in the
[`Phantom`](@ref) convention (`origin` is the centre of voxel `(1, 1, 1)`); `geom` and
`matrix_size` define the reconstruction grid as [`recon_to_world_affine`](@ref) does. Where an
output voxel reaches beyond the field's grid the missing part is `outside` — `1` for an air
fraction, `0` for everything else — so a set of fractions that sums to one still does.

The overlap is separable, so it is three matrix products, not a voxel loop: a 1850 × 1350 × 75
field onto 512 × 512 × 24 takes well under a second.
"""
function resample_field_to_recon(
        field::AbstractArray{<:Real, 3}, voxel_size, origin, geom::CTGeometry, matrix_size;
        outside::Real = 0.0,
    )
    length(voxel_size) == 3 && length(origin) == 3 || throw(ArgumentError(
        "voxel_size and origin are 3-vectors in cm"))
    nx, ny, nz = matrix_size
    fov = geom.fov
    dst = ntuple(3) do d
        (n = matrix_size[d], step = fov[d] / matrix_size[d],
            first = -fov[d] / 2 + fov[d] / matrix_size[d] / 2)
    end
    W = ntuple(3) do d
        _overlap_weights(
            size(field, d), Float64(voxel_size[d]), Float64(origin[d]),
            dst[d].n, dst[d].step, dst[d].first,
        )
    end
    # separable contraction, one axis at a time, in Float64
    f = Float64.(field)
    f = reshape(W[1] * reshape(f, size(f, 1), :), nx, size(f, 2), size(f, 3))
    f = permutedims(reshape(W[2] * reshape(permutedims(f, (2, 1, 3)), size(f, 2), :), ny, nx, size(f, 3)), (2, 1, 3))
    f = permutedims(reshape(W[3] * reshape(permutedims(f, (3, 1, 2)), size(f, 3), :), nz, nx, ny), (2, 3, 1))
    if outside != 0
        # the fraction of each output voxel that lay outside the field's grid
        cover = ntuple(d -> vec(sum(W[d]; dims = 2)), 3)
        c = [cover[1][i] * cover[2][j] * cover[3][k] for i in 1:nx, j in 1:ny, k in 1:nz]
        f .+= Float64(outside) .* (1 .- c)
    end
    return Float32.(f)
end

# Rows: destination voxels; columns: source voxels. Entry = overlap length / destination pitch,
# so a fully covered destination row sums to 1.
function _overlap_weights(
        n_src::Integer, d_src::Float64, first_src::Float64,
        n_dst::Integer, d_dst::Float64, first_dst::Float64,
    )
    W = zeros(Float64, n_dst, n_src)
    for j in 1:n_dst
        lo_j = first_dst + (j - 1) * d_dst - d_dst / 2
        hi_j = lo_j + d_dst
        # the source voxels that can overlap
        i_lo = max(1, floor(Int, (lo_j - (first_src - d_src / 2)) / d_src) + 1)
        i_hi = min(n_src, ceil(Int, (hi_j - (first_src - d_src / 2)) / d_src))
        for i in i_lo:i_hi
            lo_i = first_src + (i - 1) * d_src - d_src / 2
            hi_i = lo_i + d_src
            overlap = min(hi_i, hi_j) - max(lo_i, lo_j)
            # grids that share an edge meet there in exact arithmetic; rounding can leave a
            # sliver of order eps, which is not a real overlap and must not leak a weight
            overlap > 1.0e-9 * d_dst && (W[j, i] = overlap / d_dst)
        end
    end
    return W
end

export resample_field_to_recon
