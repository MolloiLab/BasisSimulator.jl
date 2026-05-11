"""
N-channel projection-domain SVD joint denoiser.

Operates on any set of co-registered sinogram channels — dual-kVp pairs
(N = 2), PCCT bin sets (N = 4), arbitrary N — and exploits the fact
that all channels share the same anatomy (Radon transform of the same
object), differing only in spectrum-dependent attenuation amplitude
and (cross-channel-decorrelated) quantum noise.

# Algorithm (per detector row, threaded across rows)

For each of the N input sinograms `channels[k]`, slice the row:
`slice_k = channels[k][:, r, :]  # (n_col, n_view)`

Stack as columns of a matrix:
`M = [vec(slice_1)  vec(slice_2)  …  vec(slice_N)]`

Decompose:
`M = U · Σ · V'`

* **U[:, 1]**       — common log-attenuation structure (all channels
  share the same anatomy + edges).  Signal-dominated.  **Untouched.**
* **U[:, 2..N]**    — spectral residuals (where the channels *differ*)
  + the bulk of decorrelated quantum noise.  **Smoothed** with a
  separable 2D Gaussian in `(col, view)` of the same σ.

Reconstitute `M_d = U_d · diag(Σ) · V'` and split column-wise back
into denoised channels.

# Why this works

The N noisy sinograms see the same anatomy but with independent Poisson
noise.  SVD's first component picks the optimal weighted combination of
the channels — N noisy estimates of the same edge → ~√N gain in U[:, 1].
The remaining N-1 components are linear combinations orthogonal to the
common structure: they carry per-channel spectral signatures (signal
that's spatially smooth at the rod scale) plus all the cross-channel-
decorrelated noise (signal that's spatially impulse).  A small
2D Gaussian (σ ≈ 1–2 px) suppresses the impulse noise while leaving
rod-scale spectral content untouched.

# Memory

SVD is done **per detector row**.  Each row's matrix is
`(n_col · n_view) × N`; SVD on a tall-skinny `(M × N)` matrix is fast
and the memory budget per worker thread is ~`(n_col · n_view · N · 8)`
bytes, independent of total sinogram size.  Ideal for the 16 GB Mac
budget — never holds a global `(n_pixels × N)` SVD matrix.

# Notes

* `σ_px ≤ 0` short-circuits to a passthrough (returns copies).
* Number of channels must be ≥ 2 (no SVD content for N = 1).
* All channels must share the same `(n_col, n_row, n_view)` shape and
  `Float32` element type.

# Inspiration

The signal/noise SVD-component split is the same starting move as
**RSKR** (Clark, Badea 2023 — "Rank-Sparse Kernel Regression"), but
applied to log-line-integral sinograms before reconstruction instead
of to image-domain basis-pair / PCCT-bin volumes after.  The
projection-domain version is intentionally stripped down — no
bilateral filter, no rank-sparse h-scaling, no iteration — to keep one
knob and respect the per-row independence of the sinogram domain.
For the image-domain RSKR see `apply_rskr` in
`reconstruction/vmi/rskr.jl`.
"""

# =============================================================================
#  Public API
# =============================================================================

"""
    apply_sino_svd_denoise!(out, channels; σ_px = 1.5) -> out

In-place N-channel projection-domain SVD joint denoiser.  Writes into
`out[k]` (vector of pre-allocated `Array{Float32, 3}`); `channels[k]` is
the corresponding raw sinogram input.  See the module-level docstring
for the full algorithm description and inspiration.

# Arguments
- `out::Vector{<:AbstractArray{Float32, 3}}`     : pre-allocated output
  arrays, one per channel.  Same shape as the input channels.
- `channels::Vector{<:AbstractArray{Float32, 3}}`: input sinograms
  `(n_col, n_row, n_view)`.  Length 2 (dual-kVp), 3, 4 (PCCT 4-bin), …

# Keyword arguments
- `σ_px::Real = 1.5` : Gaussian σ in pixels for the smoothing of
  `U[:, 2..N]` in `(col, view)`.  `σ_px ≤ 0` ⇒ passthrough copy.

# Returns
`out` — same vector that was passed in.
"""
function apply_sino_svd_denoise!(
        out::AbstractVector{<:AbstractArray{Float32, 3}},
        channels::AbstractVector{<:AbstractArray{Float32, 3}};
        σ_px::Real = 1.5,
    )
    n_ch = length(channels)
    n_ch ≥ 2 || error("apply_sino_svd_denoise!: requires ≥ 2 channels (got $(n_ch)).")
    length(out) == n_ch ||
        error("apply_sino_svd_denoise!: length(out) = $(length(out)) ≠ length(channels) = $(n_ch).")
    sz = size(channels[1])
    for (k, c) in enumerate(channels)
        size(c) == sz || error("apply_sino_svd_denoise!: channels[$(k)] shape $(size(c)) ≠ channels[1] shape $(sz).")
    end
    for (k, o) in enumerate(out)
        size(o) == sz || error("apply_sino_svd_denoise!: out[$(k)] shape $(size(o)) ≠ channels[1] shape $(sz).")
    end

    n_col, n_row, n_view = sz
    σ = Float64(σ_px)

    if σ ≤ 0
        for k in 1:n_ch
            copyto!(out[k], channels[k])
        end
        return out
    end

    # Pre-build separable 1D Gaussian kernel (truncate at 3σ).
    radius = max(1, ceil(Int, 3σ))
    ks = Float32[exp(-(k^2) / (2σ^2)) for k in -radius:radius]
    ks ./= sum(ks)

    Threads.@threads for r in 1:n_row
        # Per-row 2D slices (n_col, n_view) for each channel.
        slices = [Float32.(@view channels[b][:, r, :]) for b in 1:n_ch]
        M = hcat([vec(s) for s in slices]...)

        # Per-row SVD — matrix is (n_col·n_view) × N (tall-skinny, fast).
        F = LinearAlgebra.svd(M; full = false)
        U = F.U; Σ = F.S; V = F.V

        # Smooth U[:, 2..N] in (col, view) space; keep U[:, 1] untouched.
        U_d = similar(U)
        U_d[:, 1] .= U[:, 1]
        for k in 2:n_ch
            U_d[:, k] .= vec(_separable_gauss_2d(reshape(copy(U[:, k]), n_col, n_view), ks, radius))
        end

        # Reconstitute denoised channels: M_d = U_d · diag(Σ) · V'.
        M_d = U_d * LinearAlgebra.Diagonal(Σ) * V'
        for b in 1:n_ch
            out[b][:, r, :] .= reshape(view(M_d, :, b), n_col, n_view)
        end
    end

    @info "apply_sino_svd_denoise!: $(n_ch)-channel per-row SVD + 2D Gaussian " *
        "(σ = $(σ) px, radius = $(radius)) on residual components 2–$(n_ch)"
    return out
end

"""
    apply_sino_svd_denoise(channels; σ_px = 1.5) -> Vector{Array{Float32, 3}}

Allocating wrapper around `apply_sino_svd_denoise!`.  Returns a freshly
allocated vector of denoised `Array{Float32, 3}` (one per channel),
same shape as the inputs.

# Example — dual-kVp (2 channels)
```julia
out = BS.apply_sino_svd_denoise([sino_low, sino_high]; σ_px = 1.5)
sino_low_d, sino_high_d = out[1], out[2]
```

# Example — PCCT 4-bin (4 channels)
```julia
out = BS.apply_sino_svd_denoise(sim_bins.bins; σ_px = 1.5)
# out[1..4] are the denoised bins; combine to low/high downstream.
```
"""
function apply_sino_svd_denoise(
        channels::AbstractVector{<:AbstractArray{Float32, 3}};
        σ_px::Real = 1.5,
    )
    out = [Array{Float32}(undef, size(channels[1])) for _ in eachindex(channels)]
    return apply_sino_svd_denoise!(out, channels; σ_px = σ_px)
end


# =============================================================================
#  Internal — separable 1D Gaussian on a 2D slice (col-pass then view-pass)
# =============================================================================

function _separable_gauss_2d(
        slice2d::AbstractMatrix{Float32},
        ks::AbstractVector{Float32},
        radius::Int
    )
    nc, nv = size(slice2d)
    tmp = Matrix{Float32}(undef, nc, nv)
    out = Matrix{Float32}(undef, nc, nv)
    @inbounds for v in 1:nv, c in 1:nc
        s = 0.0f0; w = 0.0f0
        for (i, dk) in enumerate(-radius:radius)
            c2 = c + dk
            (1 <= c2 <= nc) || continue
            s += ks[i] * slice2d[c2, v]; w += ks[i]
        end
        tmp[c, v] = s / w
    end
    @inbounds for v in 1:nv, c in 1:nc
        s = 0.0f0; w = 0.0f0
        for (i, dk) in enumerate(-radius:radius)
            v2 = v + dk
            (1 <= v2 <= nv) || continue
            s += ks[i] * tmp[c, v2]; w += ks[i]
        end
        out[c, v] = s / w
    end
    return out
end


export apply_sino_svd_denoise, apply_sino_svd_denoise!
