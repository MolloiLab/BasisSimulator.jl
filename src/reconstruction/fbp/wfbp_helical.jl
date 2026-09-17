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
#
# MEASURED BEHAVIOUR AND KNOWN LIMITS (water cylinder r = 10 cm, 32 rows x
# 0.625 mm, arc detector, noise and scatter off, against the axial FDK of the
# same object as reference):
#
#   * Uniformity: centre -0.83 HU, peripheral ring -1.97 HU, slice-to-slice
#     sd 0.000 HU, and no feed-periodic z modulation (0.016 HU at the rotation
#     period, 0.004 HU at half of it) - i.e. no windmill in a uniform object.
#   * Slice sensitivity profile from a thin disk: FWHM 0.79-0.85 mm at any
#     radius out to 20 cm (the axial FDK gives 0.58-0.66 mm; a helical scan
#     interpolates in z, so a broader profile is the algorithm, not a defect).
#   * Sharp z edges (a 1500 HU puck): |error| up to 60-100 HU in the
#     neighbouring water, comparable to the axial FDK of the same object
#     (73 HU) and worst at pitch 1.
#   * `coverage` (below) is the honest end-of-helix and data-insufficiency
#     answer: at pitch 2 with 32 rows more than half the core voxels are not
#     reconstructable, and the unflagged volume reads -49 HU mean in them.
#
# NOT modelled: a cos(cone angle) weight in the aperture normalisation. Its
# absence is invisible at clinical collimations (16 rows: 0.04 HU; 32: 0.01;
# 64: 0.10) and grows with the cone (128 rows / 8 cm: 0.57 HU; 256 rows /
# 16 cm: 2.44 HU, tracking 1/cos(max cone angle) - 1 = 1.09 %). Treat a
# collimation beyond about 8 cm as uncalibrated in this path.
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
    arc_det = is_arc(geom)
    dγ_arc = T(geom.pixel_size / geom.SAD)

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
                col_f = if arc_det
                    γ / dγ_arc + cc                # equiangular: column IS the angle
                else
                    SDD * tan(γ) / pm + cc         # flat: planar offset
                end

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
        coverage::Union{Nothing, AbstractArray{T, 3}} = nothing,
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
    # The conjugate families are the view grid strided by a half turn, so a half turn has to
    # be a whole number of views. With an odd number of views per rotation it is not, every
    # family drifts, and the reconstruction picks up a global offset (measured on a water
    # cylinder: 360 views/rotation -0.01 HU, 361 -2.71 HU, 359 +2.85 HU, 181 +5.69 HU).
    if abs(π / Δβ - round(π / Δβ)) > 1.0e-6
        @warn "Helical WFBP: $(round(2π / Δβ; digits = 3)) views per rotation is not an even " *
            "integer, so a half turn is not a whole number of views and the conjugate-view " *
            "families are misaligned; expect a global offset of a few HU. Use an even " *
            "number of views per rotation." maxlog = 1
    end
    # Row coordinate of a voxel at height dz above the source plane, at in-plane
    # source-to-voxel distance `denom`. An arc (cylindrical, third-generation) detector has
    # every row at the same IN-PLANE distance SDD from the source, so dz scales by SDD/denom;
    # a flat panel sits at PERPENDICULAR distance SDD, which adds 1/cos(fan angle). The axial
    # backprojector already branches this way (`v_arc` in
    # reconstruction/core/backprojection.jl); this path used the flat mapping for both, which
    # stretched the off-centre slice profile of an arc scanner (thin-disk FWHM at 20 cm:
    # 0.95-1.01 mm against 0.78-0.81 mm with the correct mapping, and asymmetric in x).
    use_arc = is_arc(geom)
    feed = T(geom.table_feed)
    z_start = T(geom.source_positions[3, 1])     # source z at β = 0
    t_center = (T(n_cols) + one(T)) / T(2)
    row_center = (T(n_rows) + one(T)) / T(2)
    half_rows = T(n_rows) / T(2)
    q_plat = T(helical_q)
    twoπ = T(2π)
    half = T(0.5)
    # `coverage` is optional, but the kernel needs a concrete array to write to either way.
    cover = coverage === nothing ? similar(volume, T, 1, 1, 1) : coverage
    want_coverage = coverage !== nothing
    inv_n_half = one(T) / T(n_half)

    let reb = reb, volume = volume, cover = cover

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
            n_seen = Int32(0)
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
                            v = use_arc ? (z - z_s) * SDD / denom / prm :
                                (z - z_s) * (SDD / cosγ) / denom / prm
                            q̂ = v / half_rows
                            Wq = _wq_aperture(q̂, q_plat)
                            if Wq > zero(T)
                                t_f = t̂ / Δt + t_center
                                row_f = v + row_center
                                if t_f >= T(0.5) && t_f <= T(n_cols) + T(0.5) &&
                                   row_f >= T(0.5) && row_f <= T(n_rows) + T(0.5)
                                    wt = t_f - floor(t_f)
                                    wr = row_f - floor(row_f)
                                    t_lo = clamp(unsafe_trunc(Int32, floor(t_f)), Int32(1), n_cols)
                                    t_hi = clamp(t_lo + Int32(1), Int32(1), n_cols)
                                    r_lo = clamp(unsafe_trunc(Int32, floor(row_f)), Int32(1), n_rows)
                                    r_hi = clamp(r_lo + Int32(1), Int32(1), n_rows)
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
                    n_seen += Int32(1)
                end
                fam += Int32(1)
            end

            volume[idx] = acc * Δβ
            if want_coverage
                cover[idx] = T(n_seen) * inv_n_half
            end
        end
    end
    return volume
end

"""
    wfbp_helical_reconstruct(sinogram, geom, volume_size;
                             filter=StandardFilter(), cutoff=1.0, helical_q=0.7,
                             coverage=nothing, mask_fov=true) -> volume

Full helical WFBP chain: fan→parallel rebinning, parallel ramp filtering,
aperture-weighted wedge backprojection.  Called automatically by
[`fdk_reconstruct`](@ref) when `is_helical(geom)`.

`helical_q` is the width of the flat top of the detector-row aperture weight, as a fraction of
the half-collimation: 1 uses the whole row range with no roll-off, smaller values taper the rows
nearest the edge of the cone.  Lower `q` costs dose efficiency and a little noise (measured on a
water cylinder: in-ROI σ 3.03 HU at q = 0.3, 2.77 at 0.7, 2.63 at 1.0) and buys tolerance to the
cone.

**`coverage` is how you tell a reconstructed voxel from an unreconstructed one.** Pass an array
shaped like the volume and it is filled with the fraction of half-turn conjugate families that
found data for that voxel.  1 means fully sampled; anything less means the helix ended, or the
pitch outran the cone, and the voxel is an extrapolation — it is NOT flagged in the volume
itself, which simply reads whatever the surviving families gave (at pitch 2 with 32 rows that is
a mean of −49 HU over more than half the core, with no other sign of trouble).  Roughly one
collimation width at each end of the helix never reaches coverage 1.

`mask_fov` sets the voxels outside the reconstruction circle to the same sentinel the axial FDK
uses, so the two paths return the same convention, and zeroes their `coverage` with them.
"""
function wfbp_helical_reconstruct(
        sinogram::AbstractArray{T, 3},
        geom::CTGeometry,
        volume_size::NTuple{3, Int};
        filter::FilterType = StandardFilter(),
        cutoff::Float64 = 1.0,
        helical_q::Real = 0.7,
        coverage::Union{Nothing, AbstractArray{T, 3}} = nothing,
        mask_fov::Bool = true,
    ) where {T <: AbstractFloat}

    if coverage !== nothing
        size(coverage) == volume_size || throw(
            DimensionMismatch("coverage $(size(coverage)) must match the volume $(volume_size)")
        )
        # Caught here rather than as a kernel-compilation failure deep in the backprojection.
        eltype(coverage) === T || throw(
            ArgumentError(
                "coverage is $(eltype(coverage)); the sinogram is $(T), and they share a kernel"
            )
        )
        typeof(similar(coverage, T, 1)).name === typeof(similar(sinogram, T, 1)).name || throw(
            ArgumentError("coverage lives on a different backend from the sinogram")
        )
    end
    reb, Δt = _wfbp_rebin(sinogram, geom)
    filter_sinogram!(reb, geom; filter = filter, cutoff = cutoff,
        apply_cosine = false, ray_spacing = Δt)
    volume = similar(sinogram, T, volume_size...)
    fill!(volume, zero(T))
    _wfbp_backproject!(volume, reb, geom, Δt; helical_q = helical_q, coverage = coverage)
    if mask_fov
        apply_fov_mask!(volume, geom)
        # A voxel the mask discards is not reconstructed either, so it gets no coverage: a
        # caller filtering on `coverage .== 1` must not keep the sentinel ring.
        coverage === nothing || apply_fov_mask!(coverage, geom; sentinel_μ = zero(T))
    end
    return volume
end
