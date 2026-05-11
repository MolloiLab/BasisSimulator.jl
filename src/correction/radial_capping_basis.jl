"""
Per-basis radial capping correction.

Adapts `apply_radial_cupping_correction!` (HU-space) to operate directly
on the photoelectric / Compton basis images.  Uses a **quantile-based**
background selector instead of HU thresholds, so it works in basis units.

Because VMI at every energy is a fixed linear combination
`VMI(E) = p(E)·a + q(E)·c`, flattening the radial profile of `a` and `c`
here propagates automatically to every downstream synthesised VMI and
Mono+ image — one upstream correction, free everywhere else.

Per-slice, for each basis image:
  1. Select background voxels by quantile over the in-FOV slice.
  2. Fit an even polynomial in radius:
     `offset(r) = c₀ + c₁·r² + c₂·r⁴ + …`
  3. Subtract `offset(r) − c₀` from every voxel in the slice (keeps the
     central `r=0` value intact, flattens the radial curvature).
"""

"""
    apply_radial_capping_basis!(a, c;
                                fov_cm=35.0,
                                poly_order=2,
                                q_lo=0.25,
                                q_hi=0.75,
                                verbose=true)

Per-slice radial even-polynomial capping correction on both basis images.
Mutates `a` and `c` in place.

# Keyword arguments
- `fov_cm`     : transverse FOV in cm (pixel → cm scale)
- `poly_order` : even polynomial terms beyond c₀ (1=parabolic, 2=matches 06, 3+=stronger)
- `q_lo`, `q_hi` : quantile range over in-FOV voxels used as background

Returns a NamedTuple with per-basis mean c₁ coefficients and implied
FOV-edge drops for diagnostics.
"""
function apply_radial_capping_basis!(
        a::AbstractArray{Float32, 3},
        c::AbstractArray{Float32, 3};
        fov_cm::Real = 35.0,
        poly_order::Integer = 2,
        q_lo::Real = 0.25,
        q_hi::Real = 0.75,
        verbose::Bool = true,
    )
    nx, ny, nz = size(a)
    pixel_cm = Float64(fov_cm) / nx
    cx = (nx + 1) / 2
    cy = (ny + 1) / 2
    r_fov_sq = (nx / 2 - 2)^2
    poly_order_i = Int(poly_order)
    q_lo_f = Float64(q_lo)
    q_hi_f = Float64(q_hi)

    # In-place per-basis correction.  Returns the (poly_order+1, nz) matrix
    # of fitted polynomial coefficients per slice for diagnostics.
    correct_basis! = function (vol::AbstractArray{Float32, 3})
        coeffs_all = zeros(Float64, poly_order_i + 1, nz)
        for iz in 1:nz
            slice = @view vol[:, :, iz]

            in_fov = Float64[]
            for j in 1:ny, i in 1:nx
                if (i - cx)^2 + (j - cy)^2 <= r_fov_sq
                    push!(in_fov, Float64(slice[i, j]))
                end
            end
            isempty(in_fov) && continue
            lo = quantile(in_fov, q_lo_f)
            hi = quantile(in_fov, q_hi_f)

            radii = Float64[]
            vals = Float64[]
            for j in 1:ny, i in 1:nx
                if (i - cx)^2 + (j - cy)^2 <= r_fov_sq
                    v = Float64(slice[i, j])
                    if lo <= v <= hi
                        r_cm = sqrt(((i - cx) * pixel_cm)^2 + ((j - cy) * pixel_cm)^2)
                        push!(radii, r_cm)
                        push!(vals, v)
                    end
                end
            end
            length(radii) < 10 && continue

            n_coeffs = poly_order_i + 1
            A = zeros(length(radii), n_coeffs)
            for (k, r_cm) in enumerate(radii), p in 0:poly_order_i
                A[k, p + 1] = r_cm^(2p)
            end
            coeffs = A \ vals
            coeffs_all[:, iz] .= coeffs

            # Subtract offset − c₀ ⇒ keep DC (r=0), flatten curvature.
            target = coeffs[1]
            for j in 1:ny, i in 1:nx
                r_cm = sqrt(((i - cx) * pixel_cm)^2 + ((j - cy) * pixel_cm)^2)
                offset = sum(coeffs[p + 1] * r_cm^(2p) for p in 0:poly_order_i)
                slice[i, j] -= Float32(offset - target)
            end
        end
        return coeffs_all
    end

    t0 = time()
    coeffs_a = correct_basis!(a)
    coeffs_c = correct_basis!(c)
    dt = time() - t0

    r_max_cm = (nx / 2) * pixel_cm
    mean_c1_a = poly_order_i >= 1 ? mean(coeffs_a[2, :]) : 0.0
    mean_c1_c = poly_order_i >= 1 ? mean(coeffs_c[2, :]) : 0.0
    drop_a = mean_c1_a * r_max_cm^2
    drop_c = mean_c1_c * r_max_cm^2

    if verbose
        @info "Radial capping correction: fov=$(fov_cm) cm, poly_order=$(poly_order_i), q=[$(q_lo_f), $(q_hi_f)], $(round(dt, digits = 2)) s"
        @info "  a(r): mean c₁=$(round(mean_c1_a; sigdigits = 3))/cm²   → edge drop ≈ $(round(drop_a; sigdigits = 3))"
        @info "  c(r): mean c₁=$(round(mean_c1_c; sigdigits = 3))/cm²   → edge drop ≈ $(round(drop_c; sigdigits = 3))"
    end

    return (
        coeffs_a = coeffs_a,
        coeffs_c = coeffs_c,
        mean_c1_a = mean_c1_a,
        mean_c1_c = mean_c1_c,
        edge_drop_a = drop_a,
        edge_drop_c = drop_c,
        fov_cm = Float64(fov_cm),
        poly_order = poly_order_i,
        q_lo = q_lo_f,
        q_hi = q_hi_f,
    )
end

export apply_radial_capping_basis!
