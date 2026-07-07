# =============================================================================
# Helical WFBP — rebinned weighted filtered backprojection
# =============================================================================
#
# The production helical FBP family (Stierstorfer et al., "Weighted FBP — a
# simple approximate 3D FBP algorithm for multislice spiral CT with good dose
# usage for arbitrary pitch", Phys Med Biol 49:2209, 2004; shipped by Siemens,
# open reference implementation: UCLA FreeCT_wFBP, Hoffman et al. Med Phys
# 43:1411, 2016).  Three steps:
#
#   1. REBIN each detector row from fan (β, u) to parallel (θ, t):
#          θ = β + γ,   t = R·sin γ,   γ = atan(u_world / SDD)
#      Rays become parallel in-plane (still diverging in z — "wedge" geometry).
#   2. Ramp-FILTER the rebinned rows (plain parallel ramp — no fan cosine
#      weighting, kernel spacing Δt).
#   3. Voxel-driven BACKPROJECTION with the smooth detector-row aperture
#      weight W_Q and per-half-turn-family ΣW normalisation:
#
#          f(x) = Δθ · Σ_{θ̃∈[0,π)} [ Σ_k W_Q(q̂_k)·p_f(θ̃+kπ, t̂_k) ] / [ Σ_k W_Q(q̂_k) ]
#
#      In PARALLEL geometry the conjugate at θ+π, −t̂ is the SAME in-plane
#      line with the SAME filtering direction, so the family normalisation is
#      exactly the redundancy partition — the fan-angle conjugate wobble
#      (β* = β+π+2γ) that breaks native-cone voxel weighting is rebinned away.
#      No 1/L² distance weight in the wedge backprojection (parallel in-plane).
#
# Why not native-cone voxel-weighted FDK: post-filter per-voxel redundancy
# weights in fan geometry couple to the ramp filter and imprint feed-periodic
# banding off-axis (measured ~10% on a 30 cm phantom at pitch 1).  Rebinning
# is one extra resampling kernel and removes the artifact class — which is
# exactly why the clinical algorithms rebin.
# =============================================================================

import AcceleratedKernels as AK

"""
    _wfbp_rebin(sinogram, geom) -> (rebinned, Δt)

Row-wise fan→parallel rebinning.  Output grid: same array shape as the input
sinogram; column i ↦ t = (i − t_center)·Δt with Δt = pixel_size
(the column pitch at isocentre); view j ↦ parallel angle θ_j = angles[j]
(the unwrapped helix angle grid).  Views sampled beyond the acquired angular
range contribute zero (helix end transition).
"""
function _wfbp_rebin(
        sinogram::AbstractArray{T, 3},
        geom::CTGeometry,
    ) where {T <: AbstractFloat}
    reb = similar(sinogram)
    return _wfbp_rebin!(reb, sinogram, geom)
end

"""
    _wfbp_rebin!(reb, sinogram, geom) -> (reb, Δt)

In-place form of [`_wfbp_rebin`](@ref): writes into a caller-provided buffer
(`reb` must be sinogram-shaped and distinct from `sinogram`).
"""
function _wfbp_rebin!(
        reb::AbstractArray{T, 3},
        sinogram::AbstractArray{T, 3},
        geom::CTGeometry,
    ) where {T <: AbstractFloat}

    n_cols = Int32(size(sinogram, 1))
    n_rows = Int32(size(sinogram, 2))
    n_views = Int32(size(sinogram, 3))

    R = T(geom.SAD)
    SDD = T(geom.SDD)
    Δβ = T(geom.angles[2] - geom.angles[1])
    # geom.pixel_size is the column pitch AT ISOCENTRE, and t = R·sin(atan(u_iso/R))
    # ≈ u_iso to <0.3% over clinical fans — so the natural parallel-ray spacing
    # IS the iso column pitch.  (Using ps·R/SDD here would halve the t-grid
    # extent and truncate wide objects → interior-truncation DC bias.)
    Δt = T(geom.pixel_size)
    t_center = (T(n_cols) + one(T)) / T(2)
    col_center = (T(n_cols) + one(T)) / T(2)
    pixel_mag = T(geom.pixel_size) * (SDD / R)

    fill!(reb, zero(T))

    let sino = sinogram, reb = reb, R = R, SDD = SDD, Δβ = Δβ, Δt = Δt,
            tc = t_center, cc = col_center, pm = pixel_mag,
            nc = n_cols, nr = n_rows, nv = n_views

        AK.foreachindex(reb) do idx
            idx_0 = Int32(idx - 1)
            ti = (idx_0 % nc) + Int32(1)
            idx_0 = idx_0 ÷ nc
            row = (idx_0 % nr) + Int32(1)
            j = (idx_0 ÷ nr) + Int32(1)

            t = (T(ti) - tc) * Δt
            s = t / R
            if s > T(-0.999) && s < T(0.999)
                γ = asin(s)
                θ = (T(j) - one(T)) * Δβ          # unwrapped parallel angle
                β = θ - γ                          # unwrapped fan view angle
                jf = β / Δβ + one(T)               # fractional view index
                u_world = SDD * tan(γ)             # detector column offset (world)
                col_f = u_world / pm + cc

                if jf >= one(T) && jf <= T(nv) && col_f >= one(T) && col_f <= T(nc)
                    j_lo = unsafe_trunc(Int32, jf)
                    j_hi = min(j_lo + Int32(1), nv)
                    c_lo = unsafe_trunc(Int32, col_f)
                    c_hi = min(c_lo + Int32(1), nc)
                    wj = jf - T(j_lo)
                    wc = col_f - T(c_lo)
                    reb[idx] =
                        (one(T) - wj) * ((one(T) - wc) * sino[c_lo, row, j_lo] + wc * sino[c_hi, row, j_lo]) +
                        wj * ((one(T) - wc) * sino[c_lo, row, j_hi] + wc * sino[c_hi, row, j_hi])
                end
            end
        end
    end
    return reb, Δt
end

"""
    _wfbp_backproject!(volume, reb, geom, Δt; helical_q=0.7) -> volume

Wedge (parallel-in-plane) voxel-driven backprojection with Stierstorfer
aperture weighting and per-half-turn-family normalisation.  See the file
header for the formula.  `reb` must be the ramp-filtered rebinned sinogram.
"""
function _wfbp_backproject!(
        volume::AbstractArray{T, 3},
        reb::AbstractArray{T, 3},
        geom::CTGeometry,
        Δt::T;
        helical_q::Real = 0.7,
    ) where {T <: AbstractFloat}

    nx = Int32(size(volume, 1))
    ny = Int32(size(volume, 2))
    nz = Int32(size(volume, 3))
    n_cols = Int32(size(reb, 1))
    n_rows = Int32(size(reb, 2))
    n_views = Int32(size(reb, 3))

    vol_min_x = T(-geom.fov[1] / 2)
    vol_min_y = T(-geom.fov[2] / 2)
    vol_min_z = T(-geom.fov[3] / 2)
    vsx = T(geom.fov[1]) / T(nx)
    vsy = T(geom.fov[2]) / T(ny)
    vsz = T(geom.fov[3]) / T(nz)

    R = T(geom.SAD)
    SDD = T(geom.SDD)
    prm = T(geom.pixel_row_size) * (SDD / R)
    Δβ = T(geom.angles[2] - geom.angles[1])
    n_half = Int32(max(1, round(Int, π / Δβ)))
    feed = T(geom.table_feed)
    z_start = T(geom.source_positions[3, 1])     # source z at β = 0
    t_center = (T(n_cols) + one(T)) / T(2)
    row_center = (T(n_rows) + one(T)) / T(2)
    half_rows = T(n_rows) / T(2)
    q_plat = T(helical_q)
    twoπ = T(2π)
    half = T(0.5)

    let reb = reb, volume = volume

        AK.foreachindex(volume) do idx
            idx_0 = Int32(idx - 1)
            ix = (idx_0 % nx) + Int32(1)
            idx_0 = idx_0 ÷ nx
            iy = (idx_0 % ny) + Int32(1)
            iz = (idx_0 ÷ ny) + Int32(1)

            x = vol_min_x + (T(ix) - half) * vsx
            y = vol_min_y + (T(iy) - half) * vsy
            z = vol_min_z + (T(iz) - half) * vsz

            acc = zero(T)
            fam = Int32(1)
            while fam <= n_half
                sumW = zero(T)
                sumWP = zero(T)

                j = fam
                while j <= n_views
                    θ = (T(j) - one(T)) * Δβ
                    sinθ = sin(θ)
                    cosθ = cos(θ)
                    t̂ = x * cosθ - y * sinθ            # ray impact parameter
                    s = t̂ / R
                    if s > T(-0.999) && s < T(0.999)
                        γ = asin(s)
                        β = θ - γ
                        # source z when this parallel ray was measured
                        z_s = z_start + feed * β / twoπ
                        cosγ = cos(γ)
                        l̂ = x * sinθ + y * cosθ         # in-plane depth along the ray
                        denom = l̂ + R * cosγ            # source→voxel in-plane distance
                        if denom > T(1e-3)
                            # wedge row coordinate: z magnified source→detector
                            v = (z - z_s) * (SDD / cosγ) / denom / prm
                            q̂ = v / half_rows
                            Wq = _wq_aperture(q̂, q_plat)
                            if Wq > zero(T)
                                t_f = t̂ / Δt + t_center
                                row_f = v + row_center
                                if t_f >= one(T) && t_f <= T(n_cols) &&
                                   row_f >= one(T) && row_f <= T(n_rows)
                                    t_lo = unsafe_trunc(Int32, t_f)
                                    t_hi = min(t_lo + Int32(1), n_cols)
                                    r_lo = unsafe_trunc(Int32, row_f)
                                    r_hi = min(r_lo + Int32(1), n_rows)
                                    wt = t_f - T(t_lo)
                                    wr = row_f - T(r_lo)
                                    val = (one(T) - wt) * ((one(T) - wr) * reb[t_lo, r_lo, j] + wr * reb[t_lo, r_hi, j]) +
                                          wt * ((one(T) - wr) * reb[t_hi, r_lo, j] + wr * reb[t_hi, r_hi, j])
                                    sumW += Wq
                                    sumWP += Wq * val
                                end
                            end
                        end
                    end
                    j += n_half
                end

                if sumW > T(1e-8)
                    acc += sumWP / sumW
                end
                fam += Int32(1)
            end

            volume[idx] = acc * Δβ
        end
    end
    return volume
end

"""
    wfbp_helical_reconstruct(sinogram, geom, volume_size;
                             filter=StandardFilter(), cutoff=1.0,
                             helical_q=0.7) -> volume

Full helical WFBP chain: fan→parallel rebinning, parallel ramp filtering,
aperture-weighted wedge backprojection.  Called automatically by
[`fdk_reconstruct`](@ref) when `is_helical(geom)`.
"""
function wfbp_helical_reconstruct(
        sinogram::AbstractArray{T, 3},
        geom::CTGeometry,
        volume_size::NTuple{3, Int};
        filter::FilterType = StandardFilter(),
        cutoff::Float64 = 1.0,
        helical_q::Real = 0.7,
    ) where {T <: AbstractFloat}

    reb, Δt = _wfbp_rebin(sinogram, geom)
    filter_sinogram!(reb, geom; filter = filter, cutoff = cutoff,
        apply_cosine = false, ray_spacing = Δt)
    volume = similar(sinogram, T, volume_size...)
    fill!(volume, zero(T))
    _wfbp_backproject!(volume, reb, geom, Δt; helical_q = helical_q)
    return volume
end
