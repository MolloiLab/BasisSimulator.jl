# Total-likelihood bilateral filter (T-LBF) for a projection-domain basis pair.
#
# Lee et al. (2025): each neighbouring (iodine, water) pair is weighted by how well it explains
# the CENTRE ray's summed measured counts under the Poisson model, and one shared, normalised
# weight is applied jointly to both materials so the pair stays coherent. The collapsed total is
# used only to score neighbours; it never re-enters the material estimate.
#
# It needs Poisson counts, so it is a photon-counting tool: an energy-integrating signal is not a
# count and the likelihood weight below is not defined for it.
#
# The kernels are `AK.foreachindex` bodies, so the same code runs on CPU, CUDA, Metal, ROCm and
# oneAPI arrays. Ported from `docs/notebooks/04_pcct_vmi.jl` (`tlbf_filter_pair`) in the form
# hardened in MolloiLab/basis-vmi.

_ray_constant(a) = size(a, 1) == 1 && size(a, 2) == 1

"""
    total_measured_counts(channels, I0) -> Array{Float32,3}

Summed corrected counts across channels, `Σ_k I0_k · exp(-h_k)`: the quantity the
total-likelihood filter scores neighbours against.

- `channels`: `K` corrected log-transmission sinograms `(n_col, n_row, n_view)`.
- `I0`: `(nc, nr, K)` absolute air counts; `nc`, `nr` may be 1 when every ray shares one value.
"""
function total_measured_counts(channels::AbstractVector, I0::AbstractArray{<:Real, 3})
    length(channels) == size(I0, 3) || throw(
        DimensionMismatch("channels ($(length(channels))) ≠ I0 channels ($(size(I0, 3)))")
    )
    n_col, n_row = size(first(channels))[1:2]
    total = zeros(Float32, size(first(channels)))
    for k in eachindex(channels)
        I0_k = _ray_constant(I0) ? fill(Float64(I0[1, 1, k]), n_col, n_row) : Float64.(I0[:, :, k])
        # reshape once, outside the broadcast: under `@.` it would be applied per element
        I0_slab = reshape(I0_k, n_col, n_row, 1)
        h = channels[k]
        @. total += Float32(I0_slab * exp(-h))
    end
    return total
end

"""
    total_expected_counts(sino_iodine, sino_water, Φ, μρ_I, μρ_W; to_backend = identity)

Summed expected counts `Σ_k λ_k(A, C)` under the forward model the decomposition inverted,
evaluated at the estimated basis pair. `Φ` is the absolute response `(nc, nr, nE, K)`; it is
summed over channels here. The result is floored at `1e-6`.
"""
function total_expected_counts(
        sino_iodine::AbstractArray{<:Real, 3}, sino_water::AbstractArray{<:Real, 3},
        Φ::AbstractArray{<:Real, 4}, μρ_I::AbstractVector, μρ_W::AbstractVector;
        to_backend = identity,
    )
    Φ_total = dropdims(sum(Float64.(Φ); dims = 4); dims = 4)
    I_dev = to_backend(Float32.(sino_iodine))
    W_dev = to_backend(Float32.(sino_water))
    Φ_dev = to_backend(Float32.(Φ_total))
    μI_dev = to_backend(Float32.(μρ_I))
    μW_dev = to_backend(Float32.(μρ_W))
    out_dev = similar(I_dev)
    n_energy = length(μρ_I)
    try
        AK.foreachindex(out_dev) do idx
            n_col = size(out_dev, 1)
            n_row = size(out_dev, 2)
            col = mod1(idx, n_col)
            row = mod1(cld(idx, n_col), n_row)
            cΦ = size(Φ_dev, 1) == 1 ? 1 : col
            rΦ = size(Φ_dev, 2) == 1 ? 1 : row
            total = 0.0f0
            @inbounds for e in 1:n_energy
                ϕ = Φ_dev[cΦ, rΦ, e]
                ϕ > 0.0f0 || continue       # 0 · exp(overflow) would be NaN at a few keV
                total += ϕ * exp(-μI_dev[e] * I_dev[idx] - μW_dev[e] * W_dev[idx])
            end
            out_dev[idx] = max(total, 1.0f-6)
        end
        return Array(out_dev)
    finally
        release_backend!((I_dev, W_dev, Φ_dev, μI_dev, μW_dev, out_dev); collect = false)
    end
end

"""
    tlbf_denoise(sino_iodine, sino_water, expected, measured;
                 alpha1 = 0.9, alpha2 = 24.635648571666497, radius = 2,
                 to_backend = identity) -> (; sino_iodine, sino_water)

Lee total-likelihood bilateral filter on a projection-domain basis pair.

For centre ray `i` with measured total counts `Y_i` and Poisson log-likelihood
`ℓ(x) = -x + Y_i log x`, a neighbour `j` in the `(2·radius+1)²` column × view window gets

    w_ij = exp(-(Δc² + Δv²) / (2·alpha1²)) · exp(-(ℓ(x_j) - ℓ(x_i))² / alpha2²)

with `x` the expected total counts (see [`total_expected_counts`](@ref)). The normalised weights
are applied to iodine and water alike. Views wrap circularly; detector columns do not.

`alpha2 = Inf` keeps the spatial weights only; `alpha2 ≤ 0` is the identity. Both are the
ablation endpoints. The defaults are the fixed Lee-2025 configuration selected in notebook 04.

The neighbourhood is (column, view) and never crosses detector rows, so a sinogram with any number
of rows is filtered row by row, each in its own plane — on a single row (what
[`reduce_detector_rows`](@ref) leaves) that is exactly the published filter, and on many it is that
filter applied to each.
"""
function tlbf_denoise(
        sino_iodine::AbstractArray{<:Real, 3}, sino_water::AbstractArray{<:Real, 3},
        expected::AbstractArray{<:Real, 3}, measured::AbstractArray{<:Real, 3};
        alpha1::Real = 0.9, alpha2::Real = 24.635648571666497, radius::Integer = 2,
        to_backend = identity,
    )
    n_col, n_row, n_view = size(sino_iodine)
    size(sino_water) == size(sino_iodine) == size(expected) == size(measured) ||
        throw(DimensionMismatch("tlbf_denoise inputs must share one shape"))

    I_dev = to_backend(Float32.(sino_iodine))
    W_dev = to_backend(Float32.(sino_water))
    expected_dev = to_backend(Float32.(expected))
    measured_dev = to_backend(Float32.(measured))
    out_I = similar(I_dev)
    out_W = similar(W_dev)
    a1 = Float32(alpha1)
    a2 = Float32(alpha2)
    r = Int32(radius)
    try
        AK.foreachindex(out_I) do idx
            col = Int32(mod1(idx, n_col))
            row_view = Int32(cld(idx, n_col))        # rows vary fastest after columns
            row = mod1(row_view, Int32(n_row))
            view = cld(row_view, Int32(n_row))
            centre_expected = max(expected_dev[idx], 1.0f-6)
            Y = measured_dev[idx]
            centre_likelihood = -centre_expected + Y * log(centre_expected)
            weight_sum, iodine_sum, water_sum = 0.0f0, 0.0f0, 0.0f0
            for dv in (-r):r, dc in (-r):r
                neighbour_col = col + dc
                (neighbour_col < Int32(1) || neighbour_col > Int32(n_col)) && continue
                neighbour_view = mod1(view + dv, Int32(n_view))
                neighbour_idx = Int(neighbour_col) + (Int(row) - 1) * n_col +
                    (Int(neighbour_view) - 1) * n_col * n_row
                spatial = exp(-Float32(dc * dc + dv * dv) / (2.0f0 * a1 * a1))
                likelihood_weight = if isinf(a2)
                    1.0f0
                elseif a2 <= 0.0f0
                    (dc == 0 && dv == 0) ? 1.0f0 : 0.0f0
                else
                    candidate = max(expected_dev[neighbour_idx], 1.0f-6)
                    delta = (-candidate + Y * log(candidate)) - centre_likelihood
                    exp(-(delta * delta) / (a2 * a2))
                end
                weight = spatial * likelihood_weight
                weight_sum += weight
                iodine_sum += weight * I_dev[neighbour_idx]
                water_sum += weight * W_dev[neighbour_idx]
            end
            inverse = 1.0f0 / max(weight_sum, eps(Float32))
            out_I[idx] = iodine_sum * inverse
            out_W[idx] = water_sum * inverse
        end
        return (sino_iodine = Array(out_I), sino_water = Array(out_W))
    finally
        release_backend!((I_dev, W_dev, expected_dev, measured_dev, out_I, out_W); collect = false)
    end
end

export total_measured_counts, total_expected_counts, tlbf_denoise
