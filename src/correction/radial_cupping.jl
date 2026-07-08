# =============================================================================
# Radial Cupping / Capping Correction (Post-Reconstruction)
# =============================================================================
#
# Corrects residual low-frequency radial non-uniformity in an HU image after
# all sinogram-domain and image-domain BHC stages.  Fits a low-order even
# polynomial in radius to the water-like background (excluding air and high-
# contrast inserts), then subtracts the fitted offset so the background
# becomes flat at `target_hu`.
#
# Handles both cupping (center-dark) and capping (center-bright).  This is a
# BasisSim-original cosmetic post-recon fix, not a CatSim/TIGRE port.
# =============================================================================

export apply_radial_cupping_correction!
export measure_radial_cupping


"""
    measure_radial_cupping(hu_vol; fov_cm=35.0, half_window=90.0, poly_order=2) -> NamedTuple

QA-ONLY, non-mutating radial-cup measurement.  Fits an even polynomial in r
to water-like voxels and returns `(cup_hu, dc_hu, coeffs)` per volume
(worst slice).  Robust to noise: the voxel window is centred on the MEDIAN
of a first pass (a fixed asymmetric window truncates noise asymmetrically
and biases the DC — measured +8 HU injection on σ≈83 HU data, which is why
[`apply_radial_cupping_correction!`](@ref) is deprecated as a correction).

Doctrine: after a correct full-spectrum BHC both numbers should be ≈ 0.
A large value means the upstream chain is wrong — fix it there.
"""
function measure_radial_cupping(
        hu_vol::AbstractArray{<:Real, 3};
        fov_cm::Float64 = 35.0,
        half_window::Float64 = 90.0,
        poly_order::Int = 2,
    )
    nx, ny, nz = size(hu_vol)
    pixel_cm = fov_cm / nx
    cx = (nx + 1) / 2.0
    cy = (ny + 1) / 2.0
    worst_cup = 0.0
    worst_dc = 0.0
    worst_coeffs = zeros(poly_order + 1)
    for iz in 1:nz
        slice = @view hu_vol[:, :, iz]
        # pass 1: median of loosely water-like voxels → symmetric window centre
        rough = [Float64(v) for v in slice if -300 <= v <= 300]
        length(rough) < 100 && continue
        centre = sort!(rough)[(length(rough) + 1) ÷ 2]
        radii = Float64[]; vals = Float64[]
        for j in 1:ny, i in 1:nx
            v = Float64(slice[i, j])
            if abs(v - centre) <= half_window
                push!(radii, sqrt(((i - cx) * pixel_cm)^2 + ((j - cy) * pixel_cm)^2))
                push!(vals, v)
            end
        end
        length(radii) < 100 && continue
        A = zeros(length(radii), poly_order + 1)
        for (k, r) in enumerate(radii), p in 0:poly_order
            A[k, p + 1] = r^(2p)
        end
        coeffs = A \ vals
        r_max = maximum(radii)
        cup = abs(sum(coeffs[p + 1] * r_max^(2p) for p in 1:poly_order))
        if cup > worst_cup
            worst_cup = cup
            worst_dc = coeffs[1]
            worst_coeffs = coeffs
        end
    end
    return (cup_hu = worst_cup, dc_hu = worst_dc, coeffs = worst_coeffs)
end

"""
    apply_radial_cupping_correction!(hu_vol; fov_cm=35.0, hu_lo=-100.0, hu_hi=80.0, poly_order=2, target_hu=0.0)

Post-reconstruction radial correction for residual cupping/capping artifacts.

Fits a low-order **even** polynomial in radius `c₀ + c₁ r² + c₂ r⁴ + …` to the
water-like background voxels (those with HU in `[hu_lo, hu_hi]`), then
subtracts the fitted offset so the background becomes flat at `target_hu`.

Operates slice-by-slice; modifies `hu_vol` in place.

# Arguments
- `hu_vol::Array{Float32, 3}`: reconstructed HU volume (modified in place).

# Keyword Arguments
- `fov_cm::Float64=35.0`: field of view in cm (used to convert pixel index to
  radius).
- `hu_lo::Float64=-100.0`: lower HU threshold for water-like voxels.
- `hu_hi::Float64=80.0`: upper HU threshold for water-like voxels.
- `poly_order::Int=2`: number of even polynomial terms (fits
  `c₀ + c₁ r² + … + c_poly_order · r^(2 · poly_order)`).
- `target_hu::Float64=0.0`: target HU for the water background after correction.

# Returns
The modified `hu_vol` array.
"""
function apply_radial_cupping_correction!(
        hu_vol::Array{Float32, 3};
        fov_cm::Float64 = 35.0,
        hu_lo::Float64 = -100.0,
        hu_hi::Float64 = 80.0,
        poly_order::Int = 2,
        target_hu::Float64 = 0.0,
    )
    Base.depwarn(
        "apply_radial_cupping_correction! is DEPRECATED as a correction: on noisy data its " *
        "asymmetric fit window truncates noise asymmetrically and INJECTS a DC bias " *
        "(measured +8 HU at σ≈83 HU) while re-anchoring away real calibration errors. " *
        "Use measure_radial_cupping (non-mutating QA) and fix the upstream chain instead.",
        :apply_radial_cupping_correction!,
    )
    nx, ny, nz = size(hu_vol)
    pixel_cm = fov_cm / nx
    cx = (nx + 1) / 2.0
    cy = (ny + 1) / 2.0

    for iz in 1:nz
        slice = @view hu_vol[:, :, iz]

        # Collect (radius, HU) samples from water-like voxels.
        radii = Float64[]
        vals = Float64[]
        for j in 1:ny, i in 1:nx
            v = slice[i, j]
            if hu_lo <= v <= hu_hi
                r = sqrt(((i - cx) * pixel_cm)^2 + ((j - cy) * pixel_cm)^2)
                push!(radii, r)
                push!(vals, Float64(v))
            end
        end

        length(radii) < 10 && continue

        # Fit even polynomial.
        n_coeffs = poly_order + 1
        A = zeros(length(radii), n_coeffs)
        for (k, r) in enumerate(radii)
            for p in 0:poly_order
                A[k, p + 1] = r^(2p)
            end
        end
        coeffs = A \ vals

        # QA doctrine: after a correct (full-detected-spectrum) BHC the fitted
        # cup should be ≈ 0.  If this correction is doing real work, the BHC
        # upstream is under-correcting — surface that loudly instead of
        # silently absorbing it.
        r_max = maximum(radii)
        cup_mag = abs(sum(coeffs[p + 1] * r_max^(2p) for p in 1:poly_order))  # r-dependent part
        dc_mag = abs(coeffs[1] - target_hu)
        if cup_mag > 5.0 || dc_mag > 5.0
            @warn "apply_radial_cupping_correction!: fitted residual cup = " *
                  "$(round(cup_mag; digits=1)) HU, DC offset = $(round(dc_mag; digits=1)) HU " *
                  "(slice $iz).  Values > ~5 HU mean the upstream BHC is under-correcting — " *
                  "fix the BHC calibration rather than relying on this crutch." maxlog = 3

        end

        # Subtract fitted profile, shifting to target_hu.
        for j in 1:ny, i in 1:nx
            r = sqrt(((i - cx) * pixel_cm)^2 + ((j - cy) * pixel_cm)^2)
            fit_val = sum(coeffs[p + 1] * r^(2p) for p in 0:poly_order)
            slice[i, j] -= Float32(fit_val - target_hu)
        end
    end

    return hu_vol
end
