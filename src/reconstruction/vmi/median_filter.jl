"""
Z-direction median filter — edge-preserving impulse-noise (speckle)
removal that exploits z-axis correlation in z-invariant phantoms.

For phantoms whose cross-section is constant along z (e.g. Gammex 472
rods extruded in z), a single-voxel xy speckle does NOT repeat in
adjacent z slices.  A 1D median over `[k - adjacent_slices, k +
adjacent_slices]` at each `(x, y)` replaces it with the median of its
z-neighbors — leaving xy resolution **completely untouched**.

Compared to a 2D in-plane median:
- 2D: removes single-pixel xy speckle but slightly blurs sub-rod edges.
- z-only: zero in-plane resolution loss, perfect for z-invariant inserts.

Use this on the iodine concentration map (or any basis map) directly
after the LSQ-amplified material decomp.
"""

# =============================================================================
# Public API
# =============================================================================

"""
    apply_median_z!(out::AbstractArray{Float32, 3},
                    src::AbstractArray{Float32, 3};
                    adjacent_slices::Int = 1)
        -> out

In-place 1D median along z, per `(x, y)`.  `adjacent_slices` is the
number of slices included **on each side** of the center slice:

| `adjacent_slices` | window size | effect                          |
|-------------------|-------------|---------------------------------|
| `0`               | 1 (center)  | identity — no filtering         |
| `1`               | 3           | 1 above + center + 1 below      |
| `2`               | 5           | 2 above + center + 2 below      |
| `n`               | `2n + 1`    | `n` above + center + `n` below  |

At slice boundaries the window shrinks to the available extent (no
padding bias).
"""
function apply_median_z!(
        out::AbstractArray{Float32, 3},
        src::AbstractArray{Float32, 3};
        adjacent_slices::Int = 1,
    )
    size(out) == size(src) || error("apply_median_z!: out shape $(size(out)) ≠ src shape $(size(src))")
    adjacent_slices ≥ 0 || error("apply_median_z!: adjacent_slices = $(adjacent_slices) must be ≥ 0")
    adjacent_slices == 0 && (copyto!(out, src); return out)
    nx, ny, nz = size(src)
    n_max = 2 * adjacent_slices + 1
    Threads.@threads for k in 1:nz
        klo = max(1, k - adjacent_slices); khi = min(nz, k + adjacent_slices)
        n = khi - klo + 1
        buf = Vector{Float32}(undef, n_max)
        @inbounds for j in 1:ny, i in 1:nx
            @inbounds for (m, kk) in enumerate(klo:khi)
                buf[m] = src[i, j, kk]
            end
            sort!(view(buf, 1:n))
            out[i, j, k] = buf[(n + 1) ÷ 2]
        end
    end
    out
end

"""
    apply_median_z(src::AbstractArray{Float32, 3}; adjacent_slices::Int = 1)
        -> Array{Float32, 3}

Allocating wrapper around [`apply_median_z!`](@ref).  Returns a freshly
allocated denoised volume (same shape + element type as `src`).

`adjacent_slices = 0` returns a plain copy of `src` (no filtering).
"""
function apply_median_z(
        src::AbstractArray{Float32, 3};
        adjacent_slices::Int = 1,
    )
    adjacent_slices ≥ 0 || error("apply_median_z: adjacent_slices = $(adjacent_slices) must be ≥ 0")
    adjacent_slices == 0 && return copy(src)
    out = similar(src)
    apply_median_z!(out, src; adjacent_slices = adjacent_slices)
    out
end

export apply_median_z, apply_median_z!
