# Generalized HYPR-LR for spectral CT, in the projection and the image domain.
#
# HYPR-LR (Leng et al. 2011) estimates each energy image as a low-noise composite times the
# locally pooled ratio of the image to the composite. Written as a local-likelihood estimate it
# has four parts — the composite, the quantity pooled (its complement), the weights and the local
# model — and here each follows from the measurement:
#
#   - projection domain: the composite is each ray's total count T_i, the complement its split
#     p_ik across the K channels. For independent Poisson channels the likelihood factorizes
#     exactly into Poisson(T) × Multinomial(split | T), so the totals are ancillary for the split
#     and weights computed from them, exp(-(T_i - T_j)² / 2(T_i + T_j)), do not bias it. The split
#     is fitted locally linear in (column, view) within each detector row — rows are never mixed —
#     and each ray keeps its own total: ŷ_ik = T_i p̂_ik. Its job is the O(1/N) bias of the
#     per-ray decomposition, which only the counts can reach.
#   - image domain: on the reconstructed basis pair (a, c), the composite is the minimum-noise VMI
#     M at E* = argmin gᵀΣg, Σ the pair's noise covariance measured from the odd/even half-view
#     reconstructions; the complement I⊥ = a − βM is the iodine component whose noise is
#     uncorrelated with M. M, the conventional image of the acquisition, is kept as reconstructed;
#     I⊥, which carries only spectral information, is pooled within its slice with Gaussian
#     likelihood weights on M at M's local noise, over a window selected by an unbiased estimate of
#     its risk; and the pair is recombined. Neither instance ever pools across detector rows or
#     slices: on an object that does not change along z that would be a thicker slice.
#
# Nothing in either instance is a smoothing parameter: the weights' scales are measured from the
# acquisition, and the image-domain window is selected from it. The window extents and spatial
# profiles can also be specified the way an FBP apodization is — `HYPRKernel(window, profile)` with
# `BoxProfile`, `TriangleProfile` or `CustomProfile(control_x, control_y)`, the counterpart of
# `CustomFilter`.
#
# The odd and even views of an acquisition stay independent through the whole chain — the
# projection-domain instance pools views of one parity — so the image-domain instance can measure
# the noise of what it receives from their difference.
#
# The kernels are `AK.foreachindex` bodies, so the same code runs on CPU, CUDA, Metal, ROCm and
# oneAPI arrays. Ported from MolloiLab/basis-spectral-denoising (`notebooks/denoising_results.jl`).

# =============================================================================
# Window profiles and kernels
# =============================================================================

"""
Spatial profile of a HYPR-LR window: the weight a neighbour receives as a function of its
distance from the centre, as a fraction `x ∈ [0, 1]` of one step past the window's half-width
(so the outermost neighbours keep a nonzero weight). Separable along each window axis.
"""
abstract type HYPRProfile end

"""Uniform weights — Leng's HYPR-LR."""
struct BoxProfile <: HYPRProfile end

"""Weights falling linearly to zero one step past the window's edge."""
struct TriangleProfile <: HYPRProfile end

"""
    CustomProfile(control_x, control_y)

Piecewise-linear window profile through control points, exactly as [`CustomFilter`](@ref)
specifies an FBP apodization over frequency: `control_x` runs from 0 (the centre) to 1 (one step
past the window's edge), `control_y` is the weight there.

# Example
```julia
p = CustomProfile((0.0, 0.5, 1.0), (1.0, 0.8, 0.2))
k = HYPRKernel((5, 5), p)
```
"""
struct CustomProfile{N} <: HYPRProfile
    control_x::NTuple{N, Float64}
    control_y::NTuple{N, Float64}
    function CustomProfile{N}(control_x::NTuple{N, Float64}, control_y::NTuple{N, Float64}) where {N}
        N >= 1 || throw(ArgumentError("a profile needs at least one control point"))
        issorted(control_x) || throw(ArgumentError("control_x must be increasing, got $(control_x)"))
        all(>=(0), control_y) || throw(ArgumentError("control_y must be non-negative, got $(control_y)"))
        new{N}(control_x, control_y)
    end
end
CustomProfile(control_x::NTuple{N, Real}, control_y::NTuple{N, Real}) where {N} =
    CustomProfile{N}(Float64.(control_x), Float64.(control_y))

control_points(::BoxProfile) = ((0.0, 1.0), (1.0, 1.0))
control_points(::TriangleProfile) = ((0.0, 1.0), (1.0, 0.0))
control_points(p::CustomProfile) = (p.control_x, p.control_y)

"""
    profile_weight(profile, x) -> Float64

The profile's weight at fractional distance `x ∈ [0, 1]`.
"""
function profile_weight(p::HYPRProfile, x::Real)
    xs, ys = control_points(p)
    x <= xs[1] && return ys[1]
    for i in 1:(length(xs) - 1)
        x <= xs[i + 1] && return ys[i] + (x - xs[i]) / (xs[i + 1] - xs[i]) * (ys[i + 1] - ys[i])
    end
    return ys[end]
end

"""
    profile_weights(profile, n) -> Vector{Float64}

The 1-D weights of a width-`n` (odd) window.
"""
function profile_weights(p::HYPRProfile, n::Integer)
    h = n ÷ 2
    return [profile_weight(p, abs(d) / (h + 1)) for d in -h:h]
end

"""
    HYPRKernel(window, profile = BoxProfile(); guided = true, linear = true)

A generalized HYPR-LR kernel: the window extent in odd widths — `(columns, views)` within a
detector row in the projection domain, `(x, y, 1)` within a slice in the image domain; its spatial profile ([`HYPRProfile`](@ref)),
separable along each axis; whether each weight is also multiplied by the likelihood that the
neighbour's composite shares the centre's (`guided`); and whether the complement is fitted locally
linear (`linear`) or locally constant — HYPR-LR's pooled ratio. The local linear fit is available
in the projection domain; image-domain kernels are locally constant.

`HYPRKernel((7, 7); guided = false, linear = false)` is Leng's HYPR-LR in the counts.
"""
struct HYPRKernel{D, P <: HYPRProfile}
    window::NTuple{D, Int}
    profile::P
    guided::Bool
    linear::Bool
end
function HYPRKernel(window::NTuple{D, Integer}, profile::HYPRProfile = BoxProfile();
        guided::Bool = true, linear::Bool = D == 2) where {D}
    D in (2, 3) || throw(ArgumentError("a window has 2 (projection) or 3 (image) axes, got $(D)"))
    all(w -> w >= 1 && isodd(w), window) ||
        throw(ArgumentError("window widths must be odd and positive, got $(window)"))
    D == 3 && linear &&
        throw(ArgumentError("image-domain kernels are locally constant; pass linear = false"))
    return HYPRKernel{D, typeof(profile)}(Int.(window), profile, guided, linear)
end

function Base.show(io::IO, k::HYPRKernel)
    print(io, "HYPRKernel(", join(k.window, " × "))
    k.profile isa BoxProfile || print(io, ", ", k.profile)
    k.guided || print(io, ", unguided")
    k.linear && print(io, ", local linear")
    print(io, ")")
end


# =============================================================================
# The two instances and the chain's denoiser
# =============================================================================

"""
    ProjectionHYPR(; kernel = HYPRKernel((3, 3)), dispersion = :measured, view_stride = 2)

The projection-domain instance: count-domain generalized HYPR-LR within each detector row
([`hypr_lr`](@ref)). `dispersion` is `:measured` — read off the acquisition's own air rays by
[`estimate_dispersion`](@ref) — or a vector of `K` variance-to-mean ratios. `view_stride = 2` pools
views of one parity (the kernel's view neighbours are two views apart), so that the odd and even
halves of the acquisition stay independent: the image-domain instance measures the noise of the
chain from their difference, which pooling adjacent views would correlate and understate.
"""
struct ProjectionHYPR{Kr <: HYPRKernel{2}, Dp}
    kernel::Kr
    dispersion::Dp
    view_stride::Int
end
function ProjectionHYPR(; kernel::HYPRKernel{2} = HYPRKernel((3, 3)), dispersion = :measured,
        view_stride::Integer = 2)
    dispersion === :measured || dispersion isa AbstractVector{<:Real} ||
        throw(ArgumentError("dispersion must be :measured or a vector, got $(dispersion)"))
    view_stride in (1, 2) || throw(ArgumentError("view_stride must be 1 or 2, got $(view_stride)"))
    return ProjectionHYPR(kernel, dispersion, Int(view_stride))
end

"""
    ImageHYPR(; candidates = (3, 5, 7, 11, 15, 21, 31, 41), noise_window = 15)

The image-domain instance on the reconstructed basis pair ([`image_hypr`](@ref)). The composite, the
minimum-noise VMI, is kept as reconstructed; the complement is pooled within its slice — adjacent
slices are never averaged, which on an object that does not change along z would be a thicker slice —
over the width among `candidates` with the least estimated full-data risk. `noise_window` is the width
of the neighbourhood over which the composite's local noise, the scale of every weight, is measured
from the half-view difference (225 samples at 15: about 5 % precision).
"""
struct ImageHYPR{N}
    candidates::NTuple{N, Int}
    noise_window::Int
end
function ImageHYPR(; candidates = (3, 5, 7, 11, 15, 21, 31, 41), noise_window::Integer = 15)
    all(w -> w >= 1 && isodd(w), candidates) ||
        throw(ArgumentError("candidate widths must be odd and positive, got $(candidates)"))
    noise_window >= 3 && isodd(noise_window) ||
        throw(ArgumentError("noise_window must be odd and at least 3, got $(noise_window)"))
    return ImageHYPR(Tuple(Int.(candidates)), Int(noise_window))
end

"""
    SpectralHYPR(; projection = ProjectionHYPR(), image = ImageHYPR())

The spectral chain's denoiser for [`vmi_pipeline`](@ref): the projection-domain instance on the
counts before the decomposition, and the image-domain instance on the basis pair after FDK. Pass
`nothing` for either to run the other alone.

The published configuration is `vmi_pipeline(; …, denoiser = SpectralHYPR())`.
"""
struct SpectralHYPR{P <: Union{Nothing, ProjectionHYPR}, I <: Union{Nothing, ImageHYPR}}
    projection::P
    image::I
end
SpectralHYPR(; projection = ProjectionHYPR(), image = ImageHYPR()) = SpectralHYPR(projection, image)

# =============================================================================
# Projection domain
# =============================================================================

function _air_slab(I0, k, n_col, n_row, dispersion)
    air = size(I0, 1) == 1 && size(I0, 2) == 1 ?
        fill(Float32(I0[1, 1, k]), n_col, n_row) : Float32.(view(I0, :, :, k))
    return reshape(air ./ Float32(dispersion[k]), n_col, n_row, 1)
end

"""
    poisson_counts(channels, I0, dispersion) -> (N, T)

Poisson-equivalent counts `N (n_col, n_row, n_view, K)` — `I0_k e^{-h_k} / D_k` — and their totals
`T (n_col, n_row, n_view)`, in Float32.
"""
function poisson_counts(channels::AbstractVector, I0::AbstractArray{<:Real, 3}, dispersion)
    length(channels) == size(I0, 3) == length(dispersion) || throw(DimensionMismatch(
        "$(length(channels)) channels, $(size(I0, 3)) air channels, $(length(dispersion)) dispersions"))
    n_col, n_row, n_view = size(first(channels))
    K = length(channels)
    N = Array{Float32, 4}(undef, n_col, n_row, n_view, K)
    for k in 1:K
        N[:, :, :, k] .= _air_slab(I0, k, n_col, n_row, dispersion) .* exp.(-Float32.(channels[k]))
    end
    return N, dropdims(sum(N; dims = 4); dims = 4)
end

"""
    kernel_sums(N, T, kernel; power = 1, moments = false, to_backend = identity) -> S

The `kernel`-weighted sums, within each detector row, about each ray — every weight raised to
`power` (`power = 2` gives the sums the effective count needs). With `moments = false`,
`S (…, K + 1)` holds each channel's counts and, last, the totals. With `moments = true` — what the
local linear fit needs — `S (…, 6 + 3K)` holds the totals times `1, Δc, Δv, Δc², ΔcΔv, Δv²` and
then each channel's counts times `1, Δc, Δv`, with `Δc, Δv` the column and view offsets from the
ray. Columns are clamped at the detector edges; views wrap around a full rotation.
`view_stride = 2` takes the view neighbours two views apart, within one parity.
"""
function kernel_sums(N::AbstractArray{Float32, 4}, T::AbstractArray{Float32, 3}, kernel::HYPRKernel{2};
        power::Integer = 1, moments::Bool = false, view_stride::Integer = 1, to_backend = identity)
    power in (1, 2) || throw(ArgumentError("power must be 1 or 2, got $(power)"))
    nc, nr, nv, K = size(N)
    a = kernel.window[1] ÷ 2
    vs = Int(view_stride)
    b = min(kernel.window[2] ÷ 2, (nv - 1) ÷ (2vs))
    # views wrap around a full rotation; with a stride of 2 only an even number of views keeps
    # each parity closed under the wrap, otherwise views are clamped at the ends
    wrap = vs == 1 || iseven(nv)
    wc = to_backend(Float32.(profile_weights(kernel.profile, 2a + 1)))
    wv = to_backend(Float32.(profile_weights(kernel.profile, 2b + 1)))
    Nd = to_backend(N)
    Td = to_backend(T)
    S = to_backend(zeros(Float32, nc, nr, nv, moments ? 6 + 3K : K + 1))
    guided = kernel.guided
    squared = power == 2
    try
        AK.foreachindex(S) do idx
            i = (idx - 1) % nc + 1
            r = ((idx - 1) ÷ nc) % nr + 1
            j = ((idx - 1) ÷ (nc * nr)) % nv + 1
            s = (idx - 1) ÷ (nc * nr * nv) + 1
            Ti = Td[i, r, j]
            acc = 0.0f0
            for dj in -b:b
                jv = j + vs * dj
                (wrap || 1 <= jv <= nv) || continue
                jj = mod(jv - 1, nv) + 1
                for ii in max(1, i - a):min(nc, i + a)
                    Tj = Td[ii, r, jj]
                    w = wc[ii - i + a + 1] * wv[dj + b + 1]
                    if guided
                        d = Ti - Tj
                        w *= exp(-d * d / (2.0f0 * max(Ti + Tj, 1.0f0)))
                    end
                    squared && (w *= w)
                    if !moments
                        acc += w * (s <= K ? Nd[ii, r, jj, s] : Tj)
                    else
                        x = Float32(ii - i)
                        y = Float32(vs * dj)
                        if s <= 6
                            m = s == 1 ? 1.0f0 : s == 2 ? x : s == 3 ? y : s == 4 ? x * x : s == 5 ? x * y : y * y
                            acc += w * Tj * m
                        else
                            q = (s - 7) % 3
                            acc += w * Nd[ii, r, jj, (s - 7) ÷ 3 + 1] * (q == 0 ? 1.0f0 : q == 1 ? x : y)
                        end
                    end
                end
            end
            S[idx] = acc
        end
        return Array(S)
    finally
        release_backend!((S, Nd, Td, wc, wv); collect = false)
    end
end

"""
    pooled_split(N, T, kernel; view_stride = 1, to_backend = identity) -> p (n_col, n_row, n_view, K)

Every ray's pooled split under `kernel`: the kernel-weighted pooled ratio (locally constant —
HYPR-LR), or the intercept of the weighted least-squares fit of the neighbours' splits `y_jk / T_j`
on `(1, Δc, Δv)` with weights `w_ij T_j` (locally linear, which removes the bias an asymmetric
window would otherwise put into the pooled ratio). Fractions are floored at 1e-6 and renormalised to sum to one.
"""
function pooled_split(N::AbstractArray{Float32, 4}, T::AbstractArray{Float32, 3}, kernel::HYPRKernel{2};
        view_stride::Integer = 1, to_backend = identity)
    nc, nr, nv, K = size(N)
    n = nc * nr * nv
    S = kernel_sums(N, T, kernel; moments = kernel.linear, view_stride = view_stride, to_backend = to_backend)
    p = Array{Float32, 4}(undef, nc, nr, nv, K)
    Threads.@threads for idx in 1:n
        if kernel.linear
            m00 = Float64(S[idx]); m10 = Float64(S[idx + n]); m01 = Float64(S[idx + 2n])
            m20 = Float64(S[idx + 3n]); m11 = Float64(S[idx + 4n]); m02 = Float64(S[idx + 5n])
            # Cramer's rule on the 3 × 3 moment system; its intercept is the pooled split
            c00 = m20 * m02 - m11^2
            F = m00 * c00 - m10 * (m10 * m02 - m11 * m01) + m01 * (m10 * m11 - m20 * m01)
            ok = abs(F) > 1.0e-9 * abs(m00 * m20 * m02) && m00 > 0
            tot = 0.0
            for k in 1:K
                base = (6 + 3(k - 1)) * n
                b0 = Float64(S[idx + base]); b1 = Float64(S[idx + base + n]); b2 = Float64(S[idx + base + 2n])
                v = ok ? (b0 * c00 - m10 * (b1 * m02 - m11 * b2) + m01 * (b1 * m11 - m20 * b2)) / F :
                    b0 / max(m00, 1.0e-30)
                v = max(v, 1.0e-6)
                p[idx + (k - 1) * n] = Float32(v)
                tot += v
            end
            for k in 1:K
                p[idx + (k - 1) * n] = Float32(p[idx + (k - 1) * n] / tot)
            end
        else
            den = Float64(S[idx + K * n])
            for k in 1:K
                num = Float64(S[idx + (k - 1) * n])
                p[idx + (k - 1) * n] = Float32(max(num, 1.0e-6 * den) / max(den, 1.0e-30))
            end
        end
    end
    return p
end

"""
    hypr_lr(channels, I0; kernel = HYPRKernel((3, 3)), dispersion = ones(K), view_stride = 1,
            to_backend = identity) -> Vector{Array{Float32,3}}

Count-domain generalized HYPR-LR within each detector row: each ray keeps its own total count,
and its split across the `K` channels is the `kernel`'s pooled split of its neighbourhood
([`pooled_split`](@ref)), `ŷ_ik = T_i p̂_ik`. Returns the `K` denoised log-transmission sinograms.
Rows are never mixed.

- `channels`: `K` corrected log-transmission sinograms `(n_col, n_row, n_view)`.
- `I0`: `(nc, nr, K)` air counts; `nc`, `nr` may be 1 when every ray shares one value.
- `dispersion`: each channel's variance-to-mean ratio ([`estimate_dispersion`](@ref)); 1 for
  photon counting.
"""
function hypr_lr(channels::AbstractVector, I0::AbstractArray{<:Real, 3};
        kernel::HYPRKernel{2} = HYPRKernel((3, 3)), dispersion = ones(length(channels)),
        view_stride::Integer = 1, to_backend = identity)
    N, T = poisson_counts(channels, I0, dispersion)
    p = pooled_split(N, T, kernel; view_stride = view_stride, to_backend = to_backend)
    fl = 1.0f-6   # a count that the log can take, far below any count a detector reports
    n_col, n_row = size(T, 1), size(T, 2)
    return map(eachindex(channels)) do k
        air = _air_slab(I0, k, n_col, n_row, dispersion)
        Float32.(-log.(max.(view(p, :, :, :, k) .* T, fl) ./ air))
    end
end

"""
    estimate_dispersion(channels, I0; max_attenuation = 0.05) -> Vector{Float64}

`D_k`, the variance-to-mean ratio of each channel's photon-equivalent counts, read off the
acquisition's air rays: the (column, row) positions whose attenuation, averaged over the rotation, is
below `max_attenuation` (95 % transmission; a centimetre of water attenuates by 0.2), which miss the
object in every view. At such a ray the expected signal is the same, or changes only slowly, from view
to view, while its noise is independent from view to view, so half the mean square difference of
successive views is the noise variance, free of the ray's residual attenuation (the air of a phantom's
volume, a bowtie edge) and of any structure that changes slowly with the view. Each air ray gives one
estimate; `D_k` is their median, which a few rays grazing the object cannot move. 1.0, with a
warning, when the acquisition has no air ray; a photon-counting acquisition should return values
near one.
"""
function estimate_dispersion(channels::AbstractVector, I0::AbstractArray{<:Real, 3};
        max_attenuation::Real = 0.05)
    n_col, n_row, n_view = size(first(channels))
    n_view >= 3 || throw(ArgumentError("the dispersion needs at least 3 views, got $(n_view)"))
    constant = size(I0, 1) == 1 && size(I0, 2) == 1
    return map(eachindex(channels)) do k
        D = Float64[]
        for r in 1:n_row, i in 1:n_col
            h = Float64.(view(channels[k], i, r, :))
            mean(h) < max_attenuation || continue
            N = Float64(constant ? I0[1, 1, k] : I0[i, r, k]) .* exp.(-h)
            m = mean(N)
            v = sum(abs2, diff(N)) / (2 * (n_view - 1))
            m > 0 && v > 0 && push!(D, v / m)
        end
        if isempty(D)
            @warn "estimate_dispersion: channel $(k) has no air ray (no ray below $(max_attenuation) attenuation); using 1.0"
            return 1.0
        end
        median(D)
    end
end

# =============================================================================
# Image domain
# =============================================================================

"""
    guided_pool(X, G, σ, kernel; to_backend = identity) -> Array{Float32,3}

The local estimate of `X` at every voxel over `kernel`'s in-plane window `(x, y, 1)`, within the
voxel's slice: each neighbour weighted by the kernel's spatial profile and, if `kernel.guided`, by
the Gaussian likelihood that its guide value equals the voxel's, `exp(-(G_i - G_j)² / 2(σ_i² + σ_j²))`,
with `σ` the guide's noise, a map the size of `X`. Slices are never pooled: on an object that does not
change along z that would be a thicker slice.
"""
function guided_pool(X::AbstractArray{<:Real, 3}, G::AbstractArray{<:Real, 3}, σ::AbstractArray{<:Real, 3},
        kernel::HYPRKernel{3}; to_backend = identity)
    size(X) == size(G) == size(σ) ||
        throw(DimensionMismatch("X $(size(X)), guide $(size(G)) and noise map $(size(σ)) differ"))
    kernel.window[3] == 1 || throw(ArgumentError(
        "image-domain windows are (x, y, 1): pooling adjacent slices makes a thicker slice, got $(kernel.window)"))
    nx, ny, nz = size(X)
    a, b = kernel.window[1] ÷ 2, kernel.window[2] ÷ 2
    wx = to_backend(Float32.(profile_weights(kernel.profile, 2a + 1)))
    wy = to_backend(Float32.(profile_weights(kernel.profile, 2b + 1)))
    Xd = to_backend(Float32.(X))
    Gd = to_backend(Float32.(G))
    σd = to_backend(Float32.(σ))
    out = to_backend(zeros(Float32, nx, ny, nz))
    guided = kernel.guided
    try
        AK.foreachindex(out) do idx
            i = (idx - 1) % nx + 1
            j = ((idx - 1) ÷ nx) % ny + 1
            z = (idx - 1) ÷ (nx * ny) + 1
            Gi = Gd[i, j, z]
            si = σd[i, j, z]
            s = 0.0f0
            sw = 0.0f0
            for jj in max(1, j - b):min(ny, j + b), ii in max(1, i - a):min(nx, i + a)
                w = wx[ii - i + a + 1] * wy[jj - j + b + 1]
                if guided
                    d = Gi - Gd[ii, jj, z]
                    sj = σd[ii, jj, z]
                    w *= exp(-d * d / (2.0f0 * max(si * si + sj * sj, 1.0f-30)))
                end
                s += w * Xd[ii, jj, z]
                sw += w
            end
            out[idx] = sw > 0 ? s / sw : Xd[idx]
        end
        return Array(out)
    finally
        release_backend!((Xd, Gd, σd, out, wx, wy); collect = false)
    end
end

"""
    local_noise(d, width) -> Array{Float64,3}

The local noise of an image from its half-view difference `d` (half the odd-view minus the
even-view reconstruction, whose noise is the full reconstruction's): the root mean square of `d`
over the `width × width` neighbourhood of every voxel, within its slice. FDK noise is not
stationary — it is highest where the rays are most attenuated — so the weights of the image-domain
instance take it voxel by voxel; one value per slice would make the noisiest voxels look like
structure to the weights, and leave them unpooled.
"""
function local_noise(d::AbstractArray{<:Real, 3}, width::Integer)
    isodd(width) || throw(ArgumentError("width must be odd, got $(width)"))
    nx, ny, nz = size(d)
    h = width ÷ 2
    out = Array{Float64}(undef, nx, ny, nz)
    for z in 1:nz
        c = zeros(nx + 1, ny + 1)
        for j in 1:ny, i in 1:nx
            c[i + 1, j + 1] = Float64(d[i, j, z])^2 + c[i, j + 1] + c[i + 1, j] - c[i, j]
        end
        for j in 1:ny, i in 1:nx
            i0, i1, j0, j1 = max(1, i - h), min(nx, i + h), max(1, j - h), min(ny, j + h)
            s = c[i1 + 1, j1 + 1] - c[i0, j1 + 1] - c[i1 + 1, j0] + c[i0, j0]
            out[i, j, z] = sqrt(max(s, 0.0) / ((i1 - i0 + 1) * (j1 - j0 + 1)))
        end
    end
    return out
end

"""
    PairFilter(composite, complement)
    PairFilter(; composite, complement)

The FDK windows of a spectral reconstruction, on the pair of images whose noise is uncorrelated:
the composite `M`, the minimum-noise VMI, and its complement `I⊥` ([`spectral_pair`](@ref)). The
water and iodine images are the wrong pair to filter differently: their noise is strongly
anti-correlated, a VMI near the minimum-noise energy is quiet because that noise cancels, and it
cancels only at the frequencies where both were filtered alike — two windows there leave a shelf
of uncancelled noise in every VMI. On the uncorrelated pair the windows may differ freely: the
composite's sets the resolution and noise texture at `E*`, the complement's how the resolution
changes away from it, and every VMI of the acquisition comes from the same two reconstructions.
"""
struct PairFilter{A, B}
    composite::A
    complement::B
end
PairFilter(; composite, complement) = PairFilter(composite, complement)

_composite_filter(f) = f isa PairFilter ? f.composite : f
_complement_filter(f) = f isa PairFilter ? f.complement : f

"""
    view_subset(geom, idx) -> CTGeometry

The geometry of the projection views `idx` of an acquisition.
"""
function view_subset(g::CTGeometry, idx::AbstractVector{<:Integer})
    sub(M) = size(M, 1) == g.n_angles ? M[idx, :] : M[:, idx]
    return CTGeometry(g.SAD, g.SDD, length(idx), g.n_rows, g.n_cols, g.pixel_size, g.pixel_row_size,
        g.angles[idx], sub(g.source_positions), sub(g.detector_centers), sub(g.detector_u),
        sub(g.detector_v), g.fov, g.pitch, g.table_feed, g.detector_shape, g.column_offset)
end

_reconstruction_circle(nx, ny) = [hypot(i - (nx + 1) / 2, j - (ny + 1) / 2) < 0.45nx for i in 1:nx, j in 1:ny]

"""
    spectral_pair(sino_water, sino_iodine, geom, matrix_size; filter = SoftFilter(),
                  basis = nothing, composite_energy = nothing, antialias = true,
                  n_rows = geom.n_rows, to_backend = identity, energies = 40:140) -> NamedTuple

FDK of a decomposed basis pair (g/cm², as [`vmi_pipeline`](@ref) returns them) and of its odd- and
even-view halves, whose difference measures the noise of everything downstream:

- the pair's noise covariance `Σ` from half the halves' difference inside the reconstruction
  circle, both basis images reconstructed with the composite's window;
- the minimum-noise energy `E* = argmin gᵀΣg` over `energies`, `g = (μ_I, μ_W) / μ_W`,
  `f = (μ_I(E*), μ_W(E*))`, the composite `M = f₁ a + f₂ c` and the complement `I⊥ = a − βM`,
  `β = (Σf)₁ / fᵀΣf`, the iodine component whose noise is uncorrelated with `M`;
- with a [`PairFilter`](@ref), `M` reconstructed from its own sinogram `f₁ p_a + f₂ p_c` with the
  composite's window and `I⊥` from `p_a − β(f₁ p_a + f₂ p_c)` with the complement's, and the pair
  recovered, `a = I⊥ + βM`, `c = (M − f₁ a) / f₂`; with one window, FDK of each basis image;
- then the minimum-noise composite of the reconstructed pair itself, `E*`, `f` and `β` from the
  noise of its halves: what ACNR and the image-domain instance act on.

`composite_energy` fixes the composite's energy instead of measuring it; `β` is still measured from
the pair's halves at that energy. Where the noise of the VMIs hardly changes with energy near its
minimum, the measured argmin is itself noise and moves between acquisitions of one scanner and
protocol; its energy is a property of the scanner and protocol, measured once.

`basis = (Estar = …, β = …)` fixes the pair the windows act on instead of measuring it: a
noise-free acquisition has no noise to measure it from, and is reconstructed exactly as its
noisy counterpart. The composite of the result is still measured from the result (for a noise-free
acquisition, from its aliasing alone, which nothing downstream uses).

Returns `(water, iodine, halves, Estar, f, β, Σ, basis)`: the reconstructed pair, the `(water,
iodine)` pairs of the odd and even views reconstructed alike, the minimum-noise composite of the
pair and its noise covariance `Σ`, and `basis`, the `(Estar, β)` the windows acted on.
"""
function spectral_pair(sino_water::AbstractArray{<:Real, 3}, sino_iodine::AbstractArray{<:Real, 3},
        geom::CTGeometry, matrix_size; filter = SoftFilter(), basis = nothing,
        composite_energy = nothing, antialias::Bool = true, n_rows::Integer = geom.n_rows,
        to_backend = identity, energies = 40.0:1.0:140.0)
    size(sino_water) == size(sino_iodine) ||
        throw(DimensionMismatch("water $(size(sino_water)) and iodine $(size(sino_iodine)) differ"))
    nv = size(sino_water, 3)
    nv >= 4 || throw(ArgumentError("the half-view noise estimate needs at least 4 views, got $(nv)"))
    rec(sino, g, w) = reconstruct_basis_slice(sino, g, matrix_size; to_backend = to_backend,
        filter = w, n_rows = n_rows, antialias = antialias)
    wM, wP = _composite_filter(filter), _complement_filter(filter)
    halves_idx = (1:2:nv, 2:2:nv)
    geoms = map(h -> view_subset(geom, collect(h)), halves_idx)
    sw_h = [sino_water[:, :, h] for h in halves_idx]
    si_h = [sino_iodine[:, :, h] for h in halves_idx]
    # both basis images of each half with the composite's window: the noise covariance at E*
    Wh = [rec(sw_h[h], geoms[h], wM) for h in 1:2]
    Ih = [rec(si_h[h], geoms[h], wM) for h in 1:2]
    nx, ny, nz = size(Wh[1])
    inside = repeat(_reconstruction_circle(nx, ny), 1, 1, nz)
    dI = vec(((Ih[1] .- Ih[2]) ./ 2)[inside])
    dW = vec(((Wh[1] .- Wh[2]) ./ 2)[inside])
    Σ = [var(dI) cov(dI, dW); cov(dI, dW) var(dW)]
    μI(E) = compute_mass_μ_at_energy(XA.Elements.Iodine, Float64(E))
    μW(E) = compute_mass_μ_at_energy(XA.Materials.water, Float64(E))
    Estar, β = if basis === nothing
        E = energies[argmin([let g = [μI(E), μW(E)] ./ μW(E); g' * Σ * g end for E in energies])]
        fE = [μI(E), μW(E)]
        (Float64(E), (Σ * fE)[1] / (fE' * Σ * fE))
    else
        (Float64(basis.Estar), Float64(basis.β))
    end
    f = [μI(Estar), μW(Estar)]
    function pair(sw, si, g, Wc, Ic)
        if filter isa PairFilter
            pM = f[1] .* si .+ f[2] .* sw
            M = Wc === nothing ? rec(pM, g, wM) : f[1] .* Ic .+ f[2] .* Wc
            P = rec(si .- β .* pM, g, wP)
            iodine = P .+ β .* M
            return (water = Float32.((M .- f[1] .* iodine) ./ f[2]), iodine = Float32.(iodine))
        end
        return Wc === nothing ? (water = rec(sw, g, wM), iodine = rec(si, g, wM)) :
            (water = Float32.(Wc), iodine = Float32.(Ic))
    end
    full = pair(sino_water, sino_iodine, geom, nothing, nothing)
    halves = [pair(sw_h[h], si_h[h], geoms[h], Wh[h], Ih[h]) for h in 1:2]
    # the composite of the pair as reconstructed: its own minimum-noise energy, from its halves
    dI = vec(((halves[1].iodine .- halves[2].iodine) ./ 2)[inside])
    dW = vec(((halves[1].water .- halves[2].water) ./ 2)[inside])
    Σp = [var(dI) cov(dI, dW); cov(dI, dW) var(dW)]
    Ep = composite_energy !== nothing ? Float64(composite_energy) :
        Float64(energies[argmin([let g = [μI(E), μW(E)] ./ μW(E); g' * Σp * g end for E in energies])])
    fp = [μI(Ep), μW(Ep)]
    return (water = full.water, iodine = full.iodine, halves = halves, Estar = Ep, f = fp,
        β = (Σp * fp)[1] / (fp' * Σp * fp), Σ = Σp, basis = (Estar = Estar, β = β))
end

"""
    acnr_complement!(pair; hp_sigma_px = 1.5, window = 4, passes = 4, beta_max = 20) -> pair

Anti-correlated noise reduction (Kalender 1988, [`apply_acnr_kalender!`](@ref)) acting only on what
it is for: the anti-correlated part of the pair. The Kalender correction is applied to the pair and
the composite then restored — `a = I⊥′ + βM`, `c = (M − f₁ a) / f₂`, with `I⊥′` the complement of
the corrected pair and `M` the composite before it — so the minimum-noise image, which carries no
anti-correlated noise, is unchanged by construction; left to itself the per-pixel regression also
moves the composite, and adds noise at the energies where the VMIs are quietest. Applied alike to
the odd- and even-view halves of `pair` ([`spectral_pair`](@ref)), so that the noise downstream
steps measure is the noise of what they receive.
"""
function acnr_complement!(pair; hp_sigma_px::Real = 1.5, window::Integer = 4, passes::Integer = 4,
        beta_max::Real = 20.0)
    f, β = pair.f, pair.β
    for p in (pair, pair.halves...)
        M = f[1] .* p.iodine .+ f[2] .* p.water
        W, I = copy(p.water), copy(p.iodine)
        apply_acnr_kalender!(W, I; hp_sigma_px = hp_sigma_px, window = window, passes = passes,
            beta_max = beta_max)
        P = I .- β .* (f[1] .* I .+ f[2] .* W)
        p.iodine .= Float32.(P .+ β .* M)
        p.water .= Float32.((M .- f[1] .* p.iodine) ./ f[2])
    end
    return pair
end

"""
    image_hypr(pair; image = ImageHYPR(), to_backend = identity) -> NamedTuple

The image-domain instance on a reconstructed pair ([`spectral_pair`](@ref), after ACNR when the chain
uses it), slice by slice — adjacent slices are never pooled:

- the composite `M = f₁ a + f₂ c` at the pair's `E*`, kept as reconstructed; its local noise `σ_M`
  from the half-view difference ([`local_noise`](@ref));
- the complement `I⊥ = a − βM` with the pair's `β`, measured by [`spectral_pair`](@ref) before any
  step that changes the pair, so that its noise is uncorrelated with `M`'s (ACNR's correction,
  [`acnr_complement!`](@ref), leaves the complement correlated with the composite; re-measuring `β`
  after it would fold part of the composite, its edges included, into the pooled complement);
- `I⊥` pooled with weights from `M`, `exp(-(M_i − M_j)² / 2(σ_i² + σ_j²))`: the complement carries
  only spectral information, which changes where the material does, and there the composite does
  too. Its window is the candidate width with the least estimated full-data risk: the weights come
  from `M`, whose noise is uncorrelated with `I⊥`'s, so the pool is a linear smoother `S` of `I⊥`,
  and with `I⊥₁, I⊥₂` the halves' complements and `δ = (I⊥₁ − I⊥₂) / 2` — the full reconstruction's
  noise — `E‖S I⊥₁ − I⊥₂‖² − E‖S δ‖² = ‖(S − 1) I⊥‖² + tr(S C Sᵀ) + const`, bias² plus variance of
  the full-data estimate (`C` the complement's noise covariance), summed inside the reconstruction
  circle. The halves must be independent, which the projection-domain instance keeps
  ([`ProjectionHYPR`](@ref)'s `view_stride`);
- the pair recombined, `a = Î⊥ + βM`, `c = (M − f₁ a) / f₂`.

Returns `(water, iodine, Estar, β, window, risks, σM)`, `window` the complement's width and `risks`
each candidate's estimated risk relative to the smallest's.
"""
function image_hypr(pair::NamedTuple; image::ImageHYPR = ImageHYPR(), to_backend = identity)
    f, β = pair.f, pair.β
    Mof(p) = f[1] .* Float64.(p.iodine) .+ f[2] .* Float64.(p.water)
    M = Mof(pair)
    dM = (Mof(pair.halves[1]) .- Mof(pair.halves[2])) ./ 2
    nx, ny, nz = size(M)
    inside = repeat(_reconstruction_circle(nx, ny), 1, 1, nz)
    σM = local_noise(dM, image.noise_window)
    P = Float64.(pair.iodine) .- β .* M
    Ph = [Float64.(h.iodine) .- β .* Mof(h) for h in pair.halves]
    δ = (Ph[1] .- Ph[2]) ./ 2
    kernel_of(w) = HYPRKernel((w, w, 1); linear = false)
    r = map(image.candidates) do w
        k = kernel_of(w)
        e = guided_pool(Ph[1], M, σM, k; to_backend = to_backend) .- Ph[2]
        q = guided_pool(δ, M, σM, k; to_backend = to_backend)
        sum(abs2, e[inside]) - sum(abs2, q[inside])
    end
    window = image.candidates[argmin(r)]
    Pp = Float64.(guided_pool(P, M, σM, kernel_of(window); to_backend = to_backend))
    iodine = Float32.(Pp .+ β .* M)
    water = Float32.((M .- f[1] .* iodine) ./ f[2])
    return (water = water, iodine = iodine, Estar = pair.Estar, β = β, window = window,
        risks = collect(zip(image.candidates, r ./ first(r))), σM = σM)
end
export HYPRProfile, BoxProfile, TriangleProfile, CustomProfile, HYPRKernel
export ProjectionHYPR, ImageHYPR, SpectralHYPR, PairFilter
export hypr_lr, estimate_dispersion, guided_pool, local_noise, spectral_pair, acnr_complement!, image_hypr
