# =============================================================================
# Voxel-Driven Backprojection (TIGRE-style, AcceleratedKernels.jl)
# =============================================================================
#
# Direct port of TIGRE's voxel backprojection algorithm using AcceleratedKernels.jl
# for backend-agnostic GPU/CPU execution.
#
# Two modes:
#   - weighted=true (default): FDK-weighted backprojection for filtered backprojection
#   - weighted=false: Matched/unweighted backprojection for iterative algorithms (SIRT, CGLS)
#
# Reference:
#   - TIGRE: CERN/TIGRE/Common/CUDA/voxel_backprojection.cu (FDK weighted)
#   - TIGRE: CERN/TIGRE/Common/CUDA/voxel_backprojection_parallel.cu (matched)
#   - Feldkamp, Davis, Kress (1984) for FDK weights
#
# =============================================================================

import AcceleratedKernels as AK

export backproject!, backproject

# =============================================================================
# Single Voxel Backprojection (inlined into the main loop)
# =============================================================================

"""
    backproject_voxel(...)

Backproject all angles onto a single voxel using TIGRE's algorithm.
Port of TIGRE's voxel backprojection logic from voxel_backprojection.cu

Returns accumulated weighted backprojection value.

Note: Uses Int32 for dimensions to ensure GPU compatibility.
"""
@inline function backproject_voxel(
    sinogram::AbstractArray{T, 3},
    voxel_x::T, voxel_y::T, voxel_z::T,
    source_positions::AbstractArray{T, 2},
    detector_centers::AbstractArray{T, 2},
    detector_u::AbstractArray{T, 2},
    detector_v::AbstractArray{T, 2},
    n_cols::Int32, n_rows::Int32, n_angles::Int32,
    col_center::T, row_center::T,
    pixel_mag::T, pixel_row_mag::T, SAD::T, SAD_sq::T, pi_over_angles::T,
    arc_det::Bool, dγ::T
) where T

    acc = zero(T)
    w_full = zero(T)
    w_acc = zero(T)

    # Loop over all angles
    for angle in Int32(1):n_angles
        # Source position
        src_x = source_positions[1, angle]
        src_y = source_positions[2, angle]
        src_z = source_positions[3, angle]

        # Detector center and orientation
        dcx = detector_centers[1, angle]
        dcy = detector_centers[2, angle]
        dcz = detector_centers[3, angle]

        dux = detector_u[1, angle]
        duy = detector_u[2, angle]
        duz = detector_u[3, angle]

        dvx = detector_v[1, angle]
        dvy = detector_v[2, angle]
        dvz = detector_v[3, angle]

        # Vector from source to voxel
        sv_x = voxel_x - src_x
        sv_y = voxel_y - src_y
        sv_z = voxel_z - src_z

        # Vector from source to detector center
        sd_x = dcx - src_x
        sd_y = dcy - src_y
        sd_z = dcz - src_z

        # Distance squared from source to detector center
        sd_len_sq = sd_x^2 + sd_y^2 + sd_z^2

        # Parameter t where ray from source through voxel intersects detector plane
        sv_dot_sd = sv_x * sd_x + sv_y * sd_y + sv_z * sd_z

        # Avoid division by zero
        if abs(sv_dot_sd) < T(1e-10)
            continue
        end

        t = sd_len_sq / sv_dot_sd

        # Projected point on detector plane
        proj_x = src_x + t * sv_x
        proj_y = src_y + t * sv_y
        proj_z = src_z + t * sv_z

        # Vector from detector center to projected point
        dp_x = proj_x - dcx
        dp_y = proj_y - dcy
        dp_z = proj_z - dcz

        # Detector coordinates (col_f, row_f).  :arc → fan angle of the voxel
        # ray on the equiangular cylinder + z at in-plane distance SDD;
        # :flat → perpendicular projection onto the planar panel.
        col_f, row_f = if arc_det
            sd_len = sqrt(sd_x * sd_x + sd_y * sd_y)          # = SDD (in-plane)
            a_c = (sv_x * sd_x + sv_y * sd_y) / sd_len        # along central ray
            a_u = sv_x * dux + sv_y * duy                     # along detector û
            γv = atan(a_u, a_c)
            L_in = sqrt(sv_x * sv_x + sv_y * sv_y)
            v_arc = sv_z * (sd_len / L_in) / pixel_row_mag
            (γv / dγ + col_center, v_arc + row_center)
        else
            u = (dp_x * dux + dp_y * duy + dp_z * duz) / pixel_mag
            v = (dp_x * dvx + dp_y * dvy + dp_z * dvz) / pixel_row_mag
            (u + col_center, v + row_center)
        end

        # FDK distance weight is pure geometry (no detector dependence) — track
        # the full-scan weight sum so partial-coverage voxels renormalize
        # consistently (see return).
        dist_sq_g = arc_det ? (sv_x^2 + sv_y^2) : (sv_x^2 + sv_y^2 + sv_z^2)
        weight_g = SAD_sq / dist_sq_g
        w_full += weight_g

        # Check if within detector bounds
        if col_f >= T(0.5) && col_f <= T(n_cols) + T(0.5) && row_f >= T(0.5) && row_f <= T(n_rows) + T(0.5)
            # Bilinear interpolation indices
            col_lo = unsafe_trunc(Int32, col_f)
            col_hi = col_lo + Int32(1)
            row_lo = unsafe_trunc(Int32, row_f)
            row_hi = row_lo + Int32(1)

            # Interpolation weights
            w_col = col_f - T(col_lo)
            w_row = row_f - T(row_lo)

            # Clamp indices to valid range
            col_lo = clamp(col_lo, Int32(1), n_cols)
            col_hi = clamp(col_hi, Int32(1), n_cols)
            row_lo = clamp(row_lo, Int32(1), n_rows)
            row_hi = clamp(row_hi, Int32(1), n_rows)

            # Bilinear interpolation
            val = (one(T) - w_col) * (one(T) - w_row) * sinogram[col_lo, row_lo, angle] +
                  w_col * (one(T) - w_row) * sinogram[col_hi, row_lo, angle] +
                  (one(T) - w_col) * w_row * sinogram[col_lo, row_hi, angle] +
                  w_col * w_row * sinogram[col_hi, row_hi, angle]

            acc += val * weight_g
            w_acc += weight_g
        end
    end

    # Per-voxel redundancy normalization in WEIGHT space (same principle as
    # WFBP's sumW and IR's 1/(Bᵀ1)): Σ_full(val·w) ≈ Σ_acc(val·w)·(Σ_full w /
    # Σ_acc w).  Interior voxels: Σ_full == Σ_acc → plain FDK unchanged.
    # z-edge rim voxels keep ≥ half-scan coverage and reconstruct at ~correct
    # HU from far-side views — narrow collimation → clean edges; peak/wide
    # collimation → genuine cone inconsistency stays visible (clinical
    # behaviour).
    return w_acc > zero(T) ? acc * (w_full / w_acc) * pi_over_angles : zero(T)
end

"""
    _wq_aperture(q̂, Q) -> T

Stierstorfer cos² detector-row aperture weight (WFBP, Phys Med Biol 49:2209,
2004).  `q̂ ∈ [-1, 1]` is the voxel's projected detector-row coordinate
normalised to the outer row edges; `Q` is the flat-plateau fraction
(canonical 0.7; LEAP uses an equivalent C¹ quadratic with the same default).

    W_Q(q̂) = 1                                |q̂| < Q
           = cos²( (π/2)·(|q̂|−Q)/(1−Q) )      Q ≤ |q̂| < 1
           = 0                                |q̂| ≥ 1
"""
@inline function _wq_aperture(q̂::T, Q::T) where {T}
    aq = abs(q̂)
    if aq < Q
        return one(T)
    elseif aq < one(T)
        c = cos(T(π) / T(2) * (aq - Q) / (one(T) - Q))
        return c * c
    else
        return zero(T)
    end
end

"""
    backproject_voxel_helical(...)

Aperture-weighted helical FDK backprojection for a single voxel — native
cone-beam (non-rebinned) WFBP-family weighting with ANALYTIC conjugate-ray
normalisation (the LEAP `backprojectors_VD.cu` scheme; Stierstorfer 2004,
Tang 2006 lineage):

    f(x) = Δθ · Σ_views  [ W_Q(q̂) / N(x, view) ] · (SAD²/L²) · p_filt

    N(x, view) = Σ_k W_Q(q̂ − k·δq)  +  Σ_k W_Q(q̂* − k·δq*)

where q̂ is the voxel's projected detector-row coordinate at this view, q̂* is
its row coordinate at the CONJUGATE ray's view (β* = β + π − 2γ, the
fan-angle-corrected redundant partner — pairing β with β+π at the same voxel
is wrong off-axis and imprints feed-periodic banding), and the k-sums run
over ±full turns of the helix (row shift δq per turn), restricted to source
positions that exist on the scan.  The per-line partition of unity
(w + w* = 1) makes the in-plane fan redundancy exact — no Parker weights —
while the aperture taper W_Q handles the helical z-redundancy smoothly.
"""
@inline function backproject_voxel_helical(
    sinogram::AbstractArray{T, 3},
    voxel_x::T, voxel_y::T, voxel_z::T,
    source_positions::AbstractArray{T, 2},
    detector_centers::AbstractArray{T, 2},
    detector_u::AbstractArray{T, 2},
    detector_v::AbstractArray{T, 2},
    n_cols::Int32, n_rows::Int32, n_angles::Int32,
    col_center::T, row_center::T,
    pixel_mag::T, pixel_row_mag::T, SAD::T, SAD_sq::T, SDD::T,
    delta_theta::T, q_plateau::T, feed::T, n_turns_k::Int32,
    src_z_lo::T, src_z_hi::T,
    arc_det::Bool, dγ::T
) where T

    half_rows = T(n_rows) / T(2)
    det_dist = SDD - SAD              # detector-centre distance from isocentre
    acc = zero(T)

    for angle in Int32(1):n_angles
        # ── identical per-view projection math to backproject_voxel ──
        src_x = source_positions[1, angle]
        src_y = source_positions[2, angle]
        src_z = source_positions[3, angle]

        dcx = detector_centers[1, angle]
        dcy = detector_centers[2, angle]
        dcz = detector_centers[3, angle]

        dux = detector_u[1, angle]
        duy = detector_u[2, angle]
        duz = detector_u[3, angle]

        dvx = detector_v[1, angle]
        dvy = detector_v[2, angle]
        dvz = detector_v[3, angle]

        sv_x = voxel_x - src_x
        sv_y = voxel_y - src_y
        sv_z = voxel_z - src_z

        sd_x = dcx - src_x
        sd_y = dcy - src_y
        sd_z = dcz - src_z

        sd_len_sq = sd_x^2 + sd_y^2 + sd_z^2
        sv_dot_sd = sv_x * sd_x + sv_y * sd_y + sv_z * sd_z

        if abs(sv_dot_sd) < T(1e-10)
            continue
        end

        t = sd_len_sq / sv_dot_sd

        proj_x = src_x + t * sv_x
        proj_y = src_y + t * sv_y
        proj_z = src_z + t * sv_z

        dp_x = proj_x - dcx
        dp_y = proj_y - dcy
        dp_z = proj_z - dcz

        col_f, row_f = if arc_det
            sd_len = sqrt(sd_x * sd_x + sd_y * sd_y)
            a_c = (sv_x * sd_x + sv_y * sd_y) / sd_len
            a_u = sv_x * dux + sv_y * duy
            γv = atan(a_u, a_c)
            L_in = sqrt(sv_x * sv_x + sv_y * sv_y)
            (γv / dγ + col_center, sv_z * (sd_len / L_in) / pixel_row_mag + row_center)
        else
            u = (dp_x * dux + dp_y * duy + dp_z * duz) / pixel_mag
            v = (dp_x * dvx + dp_y * dvy + dp_z * dvz) / pixel_row_mag
            (u + col_center, v + row_center)
        end

        if !(col_f >= T(0.5) && col_f <= T(n_cols) + T(0.5) && row_f >= T(0.5) && row_f <= T(n_rows) + T(0.5))
            continue
        end

        q̂ = (row_f - row_center) / half_rows
        Wq = _wq_aperture(q̂, q_plateau)
        if Wq <= zero(T)
            continue
        end

        # ── analytic redundancy normalisation ────────────────────────────
        # view angle β from the source position (s = (−R sinβ, −R cosβ, z)):
        sinβ = -src_x / SAD
        cosβ = -src_y / SAD
        # fan angle of THIS RAY from its impact parameter ℓ (signed
        # perpendicular distance from the isocentre to the source→voxel line);
        # under this package's rotation convention the redundant partner view
        # is β* = β + π + 2γ (verified numerically against brute force).
        dlen2d = sqrt(sv_x * sv_x + sv_y * sv_y)
        ℓ = (src_x * sv_y - src_y * sv_x) / max(dlen2d, T(1e-6))
        γ = asin(clamp(ℓ / SAD, -one(T), one(T)))

        # conjugate ray: view β* = β + π + 2γ, source z advanced along the helix
        Δβc = T(π) + T(2) * γ
        βc = atan(sinβ, cosβ) + Δβc          # atan(sin, cos) = β
        zc = src_z + feed * Δβc / T(2π)
        sinβc = sin(βc)
        cosβc = cos(βc)
        # parametric scale of the conjugate projection (flat detector ⟂ ray):
        #   sd* = (SDD sinβ*, SDD cosβ*, 0);  sv*·sd* = SDD(x sinβ* + y cosβ* + R)
        denomc = voxel_x * sinβc + voxel_y * cosβc + SAD
        # row-coordinate slopes per unit z (v = t·(z_v − z_src)/pixel_row_mag)
        m = t / (pixel_row_mag * half_rows)
        q̂c = abs(denomc) > T(1e-6) ?
             (SDD / denomc) * (voxel_z - zc) / (pixel_row_mag * half_rows) : T(2)
        mc = abs(denomc) > T(1e-6) ? SDD / (denomc * pixel_row_mag * half_rows) : zero(T)

        # partition-of-unity normalisation over all helix copies that exist
        norm = zero(T)
        for k in (-n_turns_k):n_turns_k
            zk = src_z + T(k) * feed
            if zk >= src_z_lo - T(1e-6) && zk <= src_z_hi + T(1e-6)
                norm += _wq_aperture(q̂ - T(k) * m * feed, q_plateau)
            end
            zkc = zc + T(k) * feed
            if zkc >= src_z_lo - T(1e-6) && zkc <= src_z_hi + T(1e-6)
                norm += _wq_aperture(q̂c - T(k) * mc * feed, q_plateau)
            end
        end
        norm = max(norm, Wq)          # guard: the k=0 main term is always present

        # ── bilinear sample + FDK distance weight ─────────────────────────
        col_lo = unsafe_trunc(Int32, col_f)
        col_hi = col_lo + Int32(1)
        row_lo = unsafe_trunc(Int32, row_f)
        row_hi = row_lo + Int32(1)

        w_col = col_f - T(col_lo)
        w_row = row_f - T(row_lo)

        col_lo = clamp(col_lo, Int32(1), n_cols)
        col_hi = clamp(col_hi, Int32(1), n_cols)
        row_lo = clamp(row_lo, Int32(1), n_rows)
        row_hi = clamp(row_hi, Int32(1), n_rows)

        val = (one(T) - w_col) * (one(T) - w_row) * sinogram[col_lo, row_lo, angle] +
              w_col * (one(T) - w_row) * sinogram[col_hi, row_lo, angle] +
              (one(T) - w_col) * w_row * sinogram[col_lo, row_hi, angle] +
              w_col * w_row * sinogram[col_hi, row_hi, angle]

        dist_sq = arc_det ? (sv_x^2 + sv_y^2) : (sv_x^2 + sv_y^2 + sv_z^2)
        w_fdk = SAD_sq / dist_sq

        acc += (Wq / norm) * w_fdk * val
    end

    return acc * delta_theta
end

"""
    backproject_voxel_matched(...)

Matched (unweighted) backprojection for a single voxel - for iterative algorithms.

This is the transpose of the forward projection operator (A'), without FDK weighting.
Used by SIRT, CGLS, and other iterative reconstruction methods.

The key difference from `backproject_voxel` is that NO distance weighting is applied.
This ensures the backprojection is the matched adjoint of the Siddon forward projection.
"""
@inline function backproject_voxel_matched(
    sinogram::AbstractArray{T, 3},
    voxel_x::T, voxel_y::T, voxel_z::T,
    source_positions::AbstractArray{T, 2},
    detector_centers::AbstractArray{T, 2},
    detector_u::AbstractArray{T, 2},
    detector_v::AbstractArray{T, 2},
    n_cols::Int32, n_rows::Int32, n_angles::Int32,
    col_center::T, row_center::T,
    pixel_mag::T, pixel_row_mag::T,
    arc_det::Bool, dγ::T
) where T

    acc = zero(T)

    # Loop over all angles
    for angle in Int32(1):n_angles
        # Source position
        src_x = source_positions[1, angle]
        src_y = source_positions[2, angle]
        src_z = source_positions[3, angle]

        # Detector center and orientation
        dcx = detector_centers[1, angle]
        dcy = detector_centers[2, angle]
        dcz = detector_centers[3, angle]

        dux = detector_u[1, angle]
        duy = detector_u[2, angle]
        duz = detector_u[3, angle]

        dvx = detector_v[1, angle]
        dvy = detector_v[2, angle]
        dvz = detector_v[3, angle]

        # Vector from source to voxel
        sv_x = voxel_x - src_x
        sv_y = voxel_y - src_y
        sv_z = voxel_z - src_z

        # Vector from source to detector center
        sd_x = dcx - src_x
        sd_y = dcy - src_y
        sd_z = dcz - src_z

        # Distance squared from source to detector center
        sd_len_sq = sd_x^2 + sd_y^2 + sd_z^2

        # Parameter t where ray from source through voxel intersects detector plane
        sv_dot_sd = sv_x * sd_x + sv_y * sd_y + sv_z * sd_z

        # Avoid division by zero
        if abs(sv_dot_sd) < T(1e-10)
            continue
        end

        t = sd_len_sq / sv_dot_sd

        # Projected point on detector plane
        proj_x = src_x + t * sv_x
        proj_y = src_y + t * sv_y
        proj_z = src_z + t * sv_z

        # Vector from detector center to projected point
        dp_x = proj_x - dcx
        dp_y = proj_y - dcy
        dp_z = proj_z - dcz

        # Detector coordinates (col_f, row_f).  :arc → fan angle of the voxel
        # ray on the equiangular cylinder + z at in-plane distance SDD;
        # :flat → perpendicular projection onto the planar panel.
        col_f, row_f = if arc_det
            sd_len = sqrt(sd_x * sd_x + sd_y * sd_y)          # = SDD (in-plane)
            a_c = (sv_x * sd_x + sv_y * sd_y) / sd_len        # along central ray
            a_u = sv_x * dux + sv_y * duy                     # along detector û
            γv = atan(a_u, a_c)
            L_in = sqrt(sv_x * sv_x + sv_y * sv_y)
            v_arc = sv_z * (sd_len / L_in) / pixel_row_mag
            (γv / dγ + col_center, v_arc + row_center)
        else
            u = (dp_x * dux + dp_y * duy + dp_z * duz) / pixel_mag
            v = (dp_x * dvx + dp_y * dvy + dp_z * dvz) / pixel_row_mag
            (u + col_center, v + row_center)
        end

        # Check if within detector bounds
        if col_f >= T(0.5) && col_f <= T(n_cols) + T(0.5) && row_f >= T(0.5) && row_f <= T(n_rows) + T(0.5)
            # Bilinear interpolation indices
            col_lo = unsafe_trunc(Int32, col_f)
            col_hi = col_lo + Int32(1)
            row_lo = unsafe_trunc(Int32, row_f)
            row_hi = row_lo + Int32(1)

            # Interpolation weights
            w_col = col_f - T(col_lo)
            w_row = row_f - T(row_lo)

            # Clamp indices to valid range
            col_lo = clamp(col_lo, Int32(1), n_cols)
            col_hi = clamp(col_hi, Int32(1), n_cols)
            row_lo = clamp(row_lo, Int32(1), n_rows)
            row_hi = clamp(row_hi, Int32(1), n_rows)

            # Bilinear interpolation
            val = (one(T) - w_col) * (one(T) - w_row) * sinogram[col_lo, row_lo, angle] +
                  w_col * (one(T) - w_row) * sinogram[col_hi, row_lo, angle] +
                  (one(T) - w_col) * w_row * sinogram[col_lo, row_hi, angle] +
                  w_col * w_row * sinogram[col_hi, row_hi, angle]

            # NO distance weighting - this is the matched adjoint of forward projection
            acc += val
        end
    end

    return acc
end

# =============================================================================
# High-Level Interface using AcceleratedKernels.jl
# =============================================================================

"""
    backproject!(volume, sinogram, geom; weighted=true)

In-place backprojection using AcceleratedKernels.jl.

Automatically runs on GPU (Metal/CUDA/ROCm) or CPU based on array type.

# Arguments
- `volume`: Output volume [nx, ny, nz] (modified in place)
- `sinogram`: Filtered sinogram [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters

# Keyword Arguments
- `weighted`: If true (default), apply FDK distance weighting for filtered backprojection.
              If false, use matched/unweighted backprojection for iterative algorithms (SIRT, CGLS).

# Returns
The modified volume array

# Notes
- Use `weighted=true` for FDK reconstruction (after ramp filtering)
- Use `weighted=false` for iterative methods like SIRT and CGLS

The weighted version applies the FDK distance weight: w = SAD² / dist²
The unweighted version is the matched adjoint of Siddon forward projection.
"""
function backproject!(
    volume::AbstractArray{T, 3},
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry;
    weighted::Bool = true,
    helical_q::Real = 0.7,
    ws_source_positions = nothing,
    ws_detector_centers = nothing,
    ws_detector_u = nothing,
    ws_detector_v = nothing
) where T <: AbstractFloat

    # Get dimensions as Int32 for GPU compatibility
    nx = Int32(size(volume, 1))
    ny = Int32(size(volume, 2))
    nz = Int32(size(volume, 3))
    n_cols = Int32(size(sinogram, 1))
    n_rows = Int32(size(sinogram, 2))
    n_angles = Int32(size(sinogram, 3))

    # Volume parameters (typed constants)
    vol_min_x = T(-geom.fov[1] / 2)
    vol_min_y = T(-geom.fov[2] / 2)
    vol_min_z = T(-geom.fov[3] / 2)

    voxel_size_x = T(geom.fov[1]) / T(nx)
    voxel_size_y = T(geom.fov[2]) / T(ny)
    voxel_size_z = T(geom.fov[3]) / T(nz)

    magnification = T(geom.SDD / geom.SAD)
    pixel_size = T(geom.pixel_size)
    pixel_row_size = T(geom.pixel_row_size)
    SAD = T(geom.SAD)
    SAD_sq = SAD * SAD
    pixel_mag = pixel_size * magnification
    pixel_row_mag = pixel_row_size * magnification
    arc_det = is_arc(geom)
    dγ = T(geom.pixel_size / geom.SAD)

    # Pre-compute constants
    col_center = (T(n_cols) + one(T)) / T(2)
    row_center = (T(n_rows) + one(T)) / T(2)
    pi_over_angles = T(π) / T(n_angles)

    # Use pre-allocated geometry arrays if provided (zero-alloc path), else allocate
    source_positions = if ws_source_positions !== nothing
        ws_source_positions
    else
        _sp = similar(sinogram, T, size(geom.source_positions)...)
        copyto!(_sp, T.(geom.source_positions))
        _sp
    end
    detector_centers = if ws_detector_centers !== nothing
        ws_detector_centers
    else
        _dc = similar(sinogram, T, size(geom.detector_centers)...)
        copyto!(_dc, T.(geom.detector_centers))
        _dc
    end
    detector_u = if ws_detector_u !== nothing
        ws_detector_u
    else
        _du = similar(sinogram, T, size(geom.detector_u)...)
        copyto!(_du, T.(geom.detector_u))
        _du
    end
    detector_v = if ws_detector_v !== nothing
        ws_detector_v
    else
        _dv = similar(sinogram, T, size(geom.detector_v)...)
        copyto!(_dv, T.(geom.detector_v))
        _dv
    end

    # Pre-compute voxel offset for centering
    half = T(0.5)

    # Use AcceleratedKernels.jl to parallelize over all voxels
    if weighted && is_helical(geom)
        # Aperture-weighted helical FDK (WFBP-family; Stierstorfer 2004/LEAP).
        # View loop regrouped into half-turn conjugate families with per-family
        # ΣW normalisation — handles helical AND in-plane fan redundancy.
        Δθ = length(geom.angles) > 1 ? geom.angles[2] - geom.angles[1] : 2π
        delta_theta = T(Δθ)
        q_plateau = T(helical_q)
        feed = T(geom.table_feed)
        SDD_T = T(geom.SDD)
        pitch_eff = geom.pitch > 0 ? geom.pitch : 1.0
        n_turns_k = Int32(ceil(Int, 1 / pitch_eff) + 1)
        src_z_lo = T(minimum(@view geom.source_positions[3, :]))
        src_z_hi = T(maximum(@view geom.source_positions[3, :]))

        AK.foreachindex(volume) do idx
            idx_0 = Int32(idx - 1)
            ix = (idx_0 % nx) + Int32(1)
            idx_0 = idx_0 ÷ nx
            iy = (idx_0 % ny) + Int32(1)
            iz = (idx_0 ÷ ny) + Int32(1)

            voxel_x = vol_min_x + (T(ix) - half) * voxel_size_x
            voxel_y = vol_min_y + (T(iy) - half) * voxel_size_y
            voxel_z = vol_min_z + (T(iz) - half) * voxel_size_z

            volume[idx] = backproject_voxel_helical(
                sinogram,
                voxel_x, voxel_y, voxel_z,
                source_positions, detector_centers,
                detector_u, detector_v,
                n_cols, n_rows, n_angles,
                col_center, row_center,
                pixel_mag, pixel_row_mag, SAD, SAD_sq, SDD_T,
                delta_theta, q_plateau, feed, n_turns_k,
                src_z_lo, src_z_hi,
                arc_det, dγ
            )
        end
    elseif weighted
        # FDK-weighted backprojection
        AK.foreachindex(volume) do idx
            # Convert linear index to (ix, iy, iz) using integer arithmetic
            idx_0 = Int32(idx - 1)
            ix = (idx_0 % nx) + Int32(1)
            idx_0 = idx_0 ÷ nx
            iy = (idx_0 % ny) + Int32(1)
            iz = (idx_0 ÷ ny) + Int32(1)

            # Voxel center in world coordinates
            voxel_x = vol_min_x + (T(ix) - half) * voxel_size_x
            voxel_y = vol_min_y + (T(iy) - half) * voxel_size_y
            voxel_z = vol_min_z + (T(iz) - half) * voxel_size_z

            # Backproject all angles for this voxel (FDK weighted)
            volume[idx] = backproject_voxel(
                sinogram,
                voxel_x, voxel_y, voxel_z,
                source_positions, detector_centers,
                detector_u, detector_v,
                n_cols, n_rows, n_angles,
                col_center, row_center,
                pixel_mag, pixel_row_mag, SAD, SAD_sq, pi_over_angles,
                arc_det, dγ
            )
        end
    else
        # Matched/unweighted backprojection for iterative algorithms
        AK.foreachindex(volume) do idx
            # Convert linear index to (ix, iy, iz) using integer arithmetic
            idx_0 = Int32(idx - 1)
            ix = (idx_0 % nx) + Int32(1)
            idx_0 = idx_0 ÷ nx
            iy = (idx_0 % ny) + Int32(1)
            iz = (idx_0 ÷ ny) + Int32(1)

            # Voxel center in world coordinates
            voxel_x = vol_min_x + (T(ix) - half) * voxel_size_x
            voxel_y = vol_min_y + (T(iy) - half) * voxel_size_y
            voxel_z = vol_min_z + (T(iz) - half) * voxel_size_z

            # Backproject all angles for this voxel (unweighted/matched)
            volume[idx] = backproject_voxel_matched(
                sinogram,
                voxel_x, voxel_y, voxel_z,
                source_positions, detector_centers,
                detector_u, detector_v,
                n_cols, n_rows, n_angles,
                col_center, row_center,
                pixel_mag, pixel_row_mag,
                arc_det, dγ
            )
        end
    end

    return volume
end

"""
    backproject(sinogram, geom, volume_size; weighted=true)

Allocating version of backprojection.

# Arguments
- `sinogram`: Filtered sinogram [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters
- `volume_size`: (nx, ny, nz) output volume dimensions

# Keyword Arguments
- `weighted`: If true (default), apply FDK distance weighting for filtered backprojection.
              If false, use matched/unweighted backprojection for iterative algorithms.

# Returns
New volume array [nx, ny, nz]
"""
function backproject(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int};
    weighted::Bool = true
) where T <: AbstractFloat

    volume = similar(sinogram, T, volume_size...)
    fill!(volume, zero(T))

    return backproject!(volume, sinogram, geom; weighted=weighted)
end
