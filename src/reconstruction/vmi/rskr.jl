"""
Rank-Sparse Kernel Regression (RSKR) — joint multi-channel image-domain
denoiser for spectral CT basis pairs and PCCT bin volumes.

Reference: Clark, Badea (2023) — joint SVD + bilateral filter on the
basis-pair (or bin-set) preserves anti-correlated water↔iodine noise
structure that per-channel Gaussian/bilateral filters destroy.

Pipeline (per iteration):
  1. SVD of the (n_voxels × N) channel matrix → U (n_vox × N), Σ, V₀.
  2. Per-SV adaptive σ via [`mad_haar_σ`](@ref) + rank-sparse h scaling
     `h_e = h₀ · (Σ₁ / Σ_e)^γ` — smaller SVs get more smoothing.
  3. Joint bilateral filter on U columns using product-of-channels range
     kernel.
  4. Reconstitute V_denoised = U_denoised · diag(Σ) · V₀'.

# Public API
- [`apply_rskr`](@ref) — driver; auto-dispatches on `length(vols) ∈ {2, 4}`.

Backend: the joint bilateral filter is written against
`AcceleratedKernels.foreachindex`, so it dispatches on the array type
of the U columns — Metal/CUDA/ROCm when `gpu_arr_type` is the matching
device constructor, threaded CPU when `gpu_arr_type = identity` (the
default).  `vols` are CPU arrays on input/output regardless.

All hyperparameters are explicit kwargs with defaults — see
[`apply_rskr`](@ref) docstring for tuning guidance.
"""

# =============================================================================
# Adaptive σ — MAD of 2D Haar HH high-pass (paper Eq 7)
# =============================================================================

"""
    mad_haar_σ(vol::AbstractArray{Float32, 3}) -> Float32

Robust noise σ estimate from the median absolute deviation of the 2D Haar
HH (diagonal high-pass) of the mid-z slice.  Scaled by 1.4826 to match
Gaussian σ.  Used by RSKR per-SV adaptive σ.
"""
function mad_haar_σ(vol::AbstractArray{Float32, 3})
    n_col, n_row, n_z = size(vol)
    mid_z = max(1, n_z ÷ 2 + 1)
    slice = @view vol[:, :, mid_z]
    n_pairs = (n_col - 1) * (n_row - 1)
    absvec  = Vector{Float32}(undef, n_pairs)
    idx = 0
    @inbounds for j in 1:n_row-1, i in 1:n_col-1
        v = slice[i+1, j+1] - slice[i+1, j] - slice[i, j+1] + slice[i, j]
        idx += 1
        absvec[idx] = abs(v)
    end
    Float32(1.4826) * Float32(median(absvec))
end


# =============================================================================
# Joint bilateral filter — 2 channel
# =============================================================================

"""
    joint_bf_2ch_gpu!(out1, out2, in1, in2, σ1, σ2, h, radius)

In-place 2-channel joint bilateral filter on a Metal/CUDA-resident pair.
Spherical neighborhood of `radius` voxels; product-of-channels range kernel
with per-channel scale `(h·σ_c)`.

This is an internal helper for [`apply_rskr`](@ref).  Inputs/outputs must
all be on the same backend (Metal/CUDA) and have matching shapes.
"""
function joint_bf_2ch_gpu!(
        out1::AbstractArray{Float32, 3}, out2::AbstractArray{Float32, 3},
        in1::AbstractArray{Float32, 3},  in2::AbstractArray{Float32, 3},
        σ1::Float32, σ2::Float32, h::Float32, radius::Int,
    )
    n_col, n_row, n_z = size(in1)
    inv_2hσ²_1 = 1f0 / (2f0 * (h * σ1) * (h * σ1) + 1f-30)
    inv_2hσ²_2 = 1f0 / (2f0 * (h * σ2) * (h * σ2) + 1f-30)
    radius² = Int32(radius * radius)
    r_i32   = Int32(radius)

    AK.foreachindex(in1) do idx
        i0 = idx - 1
        ci = (i0 % n_col) + 1
        ri = ((i0 ÷ n_col) % n_row) + 1
        zi = ((i0 ÷ (n_col * n_row)) % n_z) + 1

        v1 = in1[ci, ri, zi]; v2 = in2[ci, ri, zi]
        sum_1 = 0f0; sum_2 = 0f0; wtot = 0f0

        for dz in -r_i32:r_i32
            zi2 = zi + dz
            (zi2 < 1 || zi2 > n_z) && continue
            dz² = Int32(dz) * Int32(dz)
            for dr in -r_i32:r_i32
                ri2 = ri + dr
                (ri2 < 1 || ri2 > n_row) && continue
                dr² = Int32(dr) * Int32(dr)
                drz² = dz² + dr²
                drz² > radius² && continue
                for dc in -r_i32:r_i32
                    ci2 = ci + dc
                    (ci2 < 1 || ci2 > n_col) && continue
                    dist² = drz² + Int32(dc) * Int32(dc)
                    dist² > radius² && continue

                    nv1 = in1[ci2, ri2, zi2]; nv2 = in2[ci2, ri2, zi2]
                    d1 = nv1 - v1; d2 = nv2 - v2
                    log_w = -d1 * d1 * inv_2hσ²_1 - d2 * d2 * inv_2hσ²_2
                    w = exp(log_w)
                    sum_1 += w * nv1
                    sum_2 += w * nv2
                    wtot  += w
                end
            end
        end

        inv_w = 1f0 / max(wtot, 1f-30)
        out1[ci, ri, zi] = sum_1 * inv_w
        out2[ci, ri, zi] = sum_2 * inv_w
    end
    nothing
end


# =============================================================================
# Joint bilateral filter — 4 channel
# =============================================================================

"""
    joint_bf_4ch_gpu!(out1, out2, out3, out4, in1, in2, in3, in4,
                      σ1, σ2, σ3, σ4, h, radius)

In-place 4-channel joint bilateral filter (PCCT 4-bin).  Same structure
as [`joint_bf_2ch_gpu!`](@ref) with four channels.
"""
function joint_bf_4ch_gpu!(
        out1::AbstractArray{Float32, 3}, out2::AbstractArray{Float32, 3},
        out3::AbstractArray{Float32, 3}, out4::AbstractArray{Float32, 3},
        in1::AbstractArray{Float32, 3},  in2::AbstractArray{Float32, 3},
        in3::AbstractArray{Float32, 3},  in4::AbstractArray{Float32, 3},
        σ1::Float32, σ2::Float32, σ3::Float32, σ4::Float32,
        h::Float32, radius::Int,
    )
    n_col, n_row, n_z = size(in1)
    inv_2hσ²_1 = 1f0 / (2f0 * (h * σ1) * (h * σ1) + 1f-30)
    inv_2hσ²_2 = 1f0 / (2f0 * (h * σ2) * (h * σ2) + 1f-30)
    inv_2hσ²_3 = 1f0 / (2f0 * (h * σ3) * (h * σ3) + 1f-30)
    inv_2hσ²_4 = 1f0 / (2f0 * (h * σ4) * (h * σ4) + 1f-30)
    radius² = Int32(radius * radius)
    r_i32   = Int32(radius)

    AK.foreachindex(in1) do idx
        i0 = idx - 1
        ci = (i0 % n_col) + 1
        ri = ((i0 ÷ n_col) % n_row) + 1
        zi = ((i0 ÷ (n_col * n_row)) % n_z) + 1

        v1 = in1[ci, ri, zi]; v2 = in2[ci, ri, zi]
        v3 = in3[ci, ri, zi]; v4 = in4[ci, ri, zi]
        sum_1 = 0f0; sum_2 = 0f0; sum_3 = 0f0; sum_4 = 0f0
        wtot  = 0f0

        for dz in -r_i32:r_i32
            zi2 = zi + dz
            (zi2 < 1 || zi2 > n_z) && continue
            dz² = Int32(dz) * Int32(dz)
            for dr in -r_i32:r_i32
                ri2 = ri + dr
                (ri2 < 1 || ri2 > n_row) && continue
                dr² = Int32(dr) * Int32(dr)
                drz² = dz² + dr²
                drz² > radius² && continue
                for dc in -r_i32:r_i32
                    ci2 = ci + dc
                    (ci2 < 1 || ci2 > n_col) && continue
                    dist² = drz² + Int32(dc) * Int32(dc)
                    dist² > radius² && continue

                    nv1 = in1[ci2, ri2, zi2]
                    nv2 = in2[ci2, ri2, zi2]
                    nv3 = in3[ci2, ri2, zi2]
                    nv4 = in4[ci2, ri2, zi2]

                    d1 = nv1 - v1; d2 = nv2 - v2
                    d3 = nv3 - v3; d4 = nv4 - v4
                    log_w = -d1 * d1 * inv_2hσ²_1 -
                             d2 * d2 * inv_2hσ²_2 -
                             d3 * d3 * inv_2hσ²_3 -
                             d4 * d4 * inv_2hσ²_4
                    w = exp(log_w)
                    sum_1 += w * nv1
                    sum_2 += w * nv2
                    sum_3 += w * nv3
                    sum_4 += w * nv4
                    wtot  += w
                end
            end
        end

        inv_w = 1f0 / max(wtot, 1f-30)
        out1[ci, ri, zi] = sum_1 * inv_w
        out2[ci, ri, zi] = sum_2 * inv_w
        out3[ci, ri, zi] = sum_3 * inv_w
        out4[ci, ri, zi] = sum_4 * inv_w
    end
    nothing
end


# =============================================================================
# Public driver — apply_rskr
# =============================================================================

"""
    apply_rskr(vols::Vector{Array{Float32, 3}};
               n_iter::Int    = 4,
               h_param::Real  = 1.0,
               radius::Int    = 6,
               γ::Real        = 0.5,
               gpu_arr_type   = identity,
               verbose::Bool  = true)
        -> Vector{Array{Float32, 3}}

Joint multi-channel RSKR denoising of a basis pair (`length(vols) == 2`)
or PCCT 4-bin set (`length(vols) == 4`).

# Arguments
- `vols`: vector of equally-shaped Float32 volumes (CPU).  Length must be
  2 or 4.

# Keyword arguments
- `n_iter`       : number of SVD-and-jBF iterations.  Each iteration
                   re-balances noise across channels (2 is plenty for
                   typical basis pairs; 4 for noisier 4-bin PCCT).
- `h_param`      : range-kernel scale `h₀`.  `1.0` is the paper default;
                   `0.5–2.0` typical.  Lower → less smoothing.
- `radius`       : spatial neighborhood half-radius (voxels).  Spherical
                   mask of `radius²` distance²; 2–6 typical.  Larger →
                   more averaging per voxel + slower.
- `γ`            : rank-sparse h-scaling exponent.  `0.5` default;
                   higher → smaller SVs get *more* smoothing.
- `gpu_arr_type` : array constructor for the BF backend.  Default
                   `identity` (CPU — `AK.foreachindex` falls back to
                   threaded loops over `Array{Float32, 3}`).  Pass
                   `MtlArray` / `CuArray` / `ROCArray` to dispatch the
                   kernel onto the corresponding device.
- `verbose`      : per-iteration timing + σ/h/Σ logging.

# Returns
A `Vector{Array{Float32, 3}}` of denoised volumes (CPU), same length and
shapes as `vols`.

# Example — DE basis pair
```julia
out = BS.apply_rskr([vol_low, vol_high]; n_iter = 2, h_param = 2.0, radius = 2)
```
"""
function apply_rskr(
        vols::Vector{Array{Float32, 3}};
        n_iter::Int       = 4,
        h_param::Real     = 1.0,
        radius::Int       = 6,
        γ::Real           = 0.5,
        gpu_arr_type      = identity,
        verbose::Bool     = true,
    )
    nch = length(vols)
    nch in (2, 4) || error("apply_rskr: requires exactly 2 or 4 channel volumes (got $nch)")
    sz = size(vols[1])
    for v in vols
        size(v) == sz || error("apply_rskr: all volumes must share shape $(sz)")
    end

    n_vox = prod(sz)
    V_mat = Matrix{Float32}(undef, n_vox, nch)
    for b in 1:nch
        V_mat[:, b] .= vec(vols[b])
    end
    h_f = Float32(h_param)
    γ_f = Float32(γ)

    for iter in 1:n_iter
        t0 = time()
        F = svd(V_mat; full = false)
        U = F.U; Σ = F.S; V₀ = F.V

        σ_per_sv = Vector{Float32}(undef, nch)
        h_per_sv = Vector{Float32}(undef, nch)
        U_vols   = [reshape(U[:, e], sz) for e in 1:nch]
        for e in 1:nch
            σ_per_sv[e] = mad_haar_σ(U_vols[e])
            h_per_sv[e] = h_f * (Float32(Σ[1]) / max(Float32(Σ[e]), 1f-12)) ^ γ_f
        end

        # Upload U columns onto whichever array backend `gpu_arr_type`
        # selects (`identity` keeps them on CPU; `MtlArray`/`CuArray`/
        # `ROCArray` ship them to the device).  AK.foreachindex inside
        # `joint_bf_*ch_gpu!` dispatches accordingly — same source, any
        # backend.

        if nch == 2
            in1 = gpu_arr_type(U_vols[1]); in2 = gpu_arr_type(U_vols[2])
            out1 = similar(in1);             out2 = similar(in2)
            σ_eff_1 = h_per_sv[1] * σ_per_sv[1]
            σ_eff_2 = h_per_sv[2] * σ_per_sv[2]
            joint_bf_2ch_gpu!(out1, out2, in1, in2, σ_eff_1, σ_eff_2, 1f0, radius)
            U_denoised = hcat(vec(Array(out1)), vec(Array(out2)))
            in1 = nothing; in2 = nothing; out1 = nothing; out2 = nothing
        else  # nch == 4
            in1 = gpu_arr_type(U_vols[1]); in2 = gpu_arr_type(U_vols[2])
            in3 = gpu_arr_type(U_vols[3]); in4 = gpu_arr_type(U_vols[4])
            out1 = similar(in1); out2 = similar(in2)
            out3 = similar(in3); out4 = similar(in4)
            σ_eff = [h_per_sv[e] * σ_per_sv[e] for e in 1:4]
            joint_bf_4ch_gpu!(out1, out2, out3, out4,
                              in1,  in2,  in3,  in4,
                              σ_eff[1], σ_eff[2], σ_eff[3], σ_eff[4],
                              1f0, radius)
            U_denoised = hcat(vec(Array(out1)), vec(Array(out2)),
                              vec(Array(out3)), vec(Array(out4)))
            in1 = nothing; in2 = nothing; in3 = nothing; in4 = nothing
            out1 = nothing; out2 = nothing; out3 = nothing; out4 = nothing
        end

        V_mat = U_denoised * Diagonal(Σ) * V₀'

        if verbose
            dt = time() - t0
            @info "[apply_rskr iter $(iter)/$(n_iter)] $(round(dt, digits=2))s   σ_per_SV = $(round.(σ_per_sv, sigdigits=3))   h_per_SV = $(round.(h_per_sv, sigdigits=3))   Σ = $(round.(Σ, sigdigits=3))"
        end
    end

    [reshape(V_mat[:, b], sz) for b in 1:nch]
end


export apply_rskr, mad_haar_σ, joint_bf_2ch_gpu!, joint_bf_4ch_gpu!
