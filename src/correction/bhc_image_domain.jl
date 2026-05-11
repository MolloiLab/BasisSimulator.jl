# =============================================================================
# Image-Domain Beam Hardening Correction (So et al. 2009)
# =============================================================================
#
# Post-reconstruction BHC refinement that subtracts a scaled forward+back-
# projected error image from a water-BHC-corrected μ-domain image.  Used
# downstream of `apply_bhc_two_material` to mop up residual high-attenuation
# artifacts (bone, contrast) that the sinogram-domain stage leaves behind.
#
# ## Algorithm (5 steps)
#
# 1. Start with a water-BHC-corrected reconstructed image (μ domain).
# 2. Segment high-attenuation voxels via a linear weighting factor on
#    `[hu_low, hu_high]` → weighted "excess μ above water" image.
# 3. Forward-project the weighted high-atten image → ξ (error sinogram).
# 4. FDK-reconstruct ξ → error image.
# 5. Subtract `scale_factor × error_image` from the original.
#
# ## Reference
#
# So A, Hsieh J, Li JY, Lee TY. "Beam hardening correction in CT myocardial
# perfusion measurement." Phys Med Biol. 2009;54(10):3031-3050.
# doi:10.1088/0031-9155/54/10/004
#
# The `scale_factor` knob empirically absorbs all polynomial coefficient
# differences (α_i - β_i from equation 8 in the paper) into a single tunable.
# =============================================================================

import AcceleratedKernels as AK

export apply_bhc_image_domain

"""
    apply_bhc_image_domain(recon_μ, geom, matrix_size, μ_water_ref;
                           hu_low=70.0, hu_high=150.0, scale_factor=0.5)

Apply image-domain beam hardening correction (So et al. 2009).

# Arguments
- `recon_μ::AbstractArray{T,3}`: reconstructed image in μ (cm⁻¹), already
  corrected with water-only or two-material sinogram BHC. GPU array.
- `geom`: `CTGeometry` used for reconstruction.
- `matrix_size::Tuple{Int,Int,Int}`: reconstruction matrix `(nx, ny, nz)`.
- `μ_water_ref::Real`: water linear attenuation at the reference energy.

# Keyword Arguments
- `hu_low::Real=70.0`: lower HU threshold; voxels ≤ this are 100% soft tissue (WF=0).
- `hu_high::Real=150.0`: upper HU threshold; voxels ≥ this are 100% high-atten (WF=1).
- `scale_factor::Real=0.5`: scale of error-image subtraction.
  - `0.0` = no correction (identity)
  - `1.0` = full subtraction
  - Typical 0.2–0.7; start at 0.5.

# Returns
The modified `recon_μ` array (same object, mutated).

# Note on geometry
Both halves of the internal forward+back-projection round-trip use `geom`'s
recon FOV. Passing a separate `volume_extent` (the physical extent of the
original phantom) would be conceptually wrong here: by the time this function
runs, the phantom is gone — we're working with a reconstructed image that
already lives in `geom`'s recon grid. An earlier signature accepted such a
kwarg and produced bone-shaped halo + diagonal Moiré artifacts whenever the
two extents differed; it has been removed.

# Reference
So A, Hsieh J, Li JY, Lee TY. Phys Med Biol. 2009;54(10):3031-3050.
"""
function apply_bhc_image_domain(
        recon_μ::AbstractArray{T, 3},
        geom,
        matrix_size::Tuple{Int, Int, Int},
        μ_water_ref::Real;
        hu_low::Real = 70.0,
        hu_high::Real = 150.0,
        scale_factor::Real = 0.5,
    ) where {T <: AbstractFloat}

    μ_w_ref = T(μ_water_ref)
    hu_low_T = T(hu_low)
    hu_high_T = T(hu_high)

    # Step 1: segment high-attenuation voxels → weighted μ image (GPU).
    # Linear interpolation weighting factor per So et al. 2009 §2.2.
    high_atten_μ = similar(recon_μ)
    let recon_μ = recon_μ, high_atten_μ = high_atten_μ,
            μ_w_ref = μ_w_ref, hu_low_T = hu_low_T, hu_high_T = hu_high_T
        AK.foreachindex(recon_μ) do idx
            μ_val = recon_μ[idx]
            hu_val = T(1000) * (μ_val - μ_w_ref) / μ_w_ref

            wf = if hu_val <= hu_low_T
                zero(T)
            elseif hu_val >= hu_high_T
                one(T)
            else
                (hu_val - hu_low_T) / (hu_high_T - hu_low_T)
            end

            # Paper works in HU space (water=0); in μ space we subtract the
            # water baseline so we only capture excess attenuation above water.
            high_atten_μ[idx] = wf * (μ_val - μ_w_ref)
        end
    end

    # Step 2: forward-project the high-attenuation image → ξ (error sinogram).
    # Use geom's recon FOV (no volume_extent override) so this matches Step 3's
    # FDK back-projection grid exactly — same physical box on both halves of
    # the round-trip.
    ξ_gpu = siddon_forward_project(high_atten_μ, geom)

    # Step 3: FDK-reconstruct the error sinogram → error image.
    error_image = fdk_reconstruct(ξ_gpu, geom, matrix_size)

    # Step 4: subtract scaled error image from original.
    sf = T(scale_factor)
    let recon_μ = recon_μ, error_image = error_image, sf = sf
        AK.foreachindex(recon_μ) do idx
            recon_μ[idx] = recon_μ[idx] - sf * error_image[idx]
        end
    end

    return recon_μ
end
