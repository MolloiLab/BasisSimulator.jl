# =============================================================================
# Hybrid Iterative Reconstruction (TRUE Hybrid IR)
# =============================================================================
#
# Vendor-general implementation based on clinical research (Geyer et al. 2015,
# Willemink & Noël 2019, SAFIRE clinical studies).
#
# TRUE Hybrid IR = FDK initialization + PWLS refinement with Huber regularization
#
# This is NOT simple blending (which is FALSE HIR):
#   FALSE: x = (1-α) * FDK + α * smooth(FDK)
#   TRUE:  x = PWLS_refine(FDK_init, sinogram, weights, regularization)
#
# Clinical Vendor Mapping:
#   - GE ASIR-V: strength 1-5 ≈ ASIR-V 20-100%
#   - Siemens SAFIRE: strength 1-5 = SAFIRE S1-S5 (direct)
#   - Philips iDose4: strength 1-5 ≈ iDose4 levels 1-5
#   - Canon AIDR 3D: strength 2/3/4 ≈ Mild/Standard/Strong
#
# =============================================================================

export hybrid_ir_reconstruct, get_hir_params, HIRParams

# =============================================================================
# Hybrid IR Parameters (Research-Based)
# =============================================================================

"""
    HIRParams

Parameters for Hybrid IR reconstruction at a given strength level.

Based on SAFIRE clinical validation data (Ghetti et al., PMC5714520).

# Fields
- `strength`: Strength level (1-5)
- `lambda`: Regularization strength
- `niter`: Number of PWLS iterations (legacy full-data mode when n_subsets=0)
- `nepochs`: Number of OS epochs (used when n_subsets > 0)
- `n_subsets`: Number of ordered subsets (0 = legacy full-data mode)
- `huber_delta`: Huber penalty edge threshold
- `relaxation`: SIRT relaxation parameter (< 1.0 for stability)
- `target_noise_reduction`: Expected noise reduction percentage
"""
struct HIRParams
    strength::Int
    lambda::Float32
    niter::Int
    nepochs::Int
    n_subsets::Int
    huber_delta::Float32
    relaxation::Float32
    target_noise_reduction::Tuple{Int, Int}  # (min%, max%)
end

"""
    get_hir_params(strength::Int) -> HIRParams

Get physics-based parameters for a given strength level (1-5).

Parameters are derived from SAFIRE clinical studies (PMC5714520, PMC4401802):
- Higher strength → more noise reduction, more regularization
- Lower strength → less noise reduction, preserves more texture

# Strength Level Summary

| Strength | Noise Red. | Performance | Clinical Use |
|----------|------------|-------------|--------------|
| 1 | 8-15% | ~1.5x FDK | Preserve texture (lung nodules) |
| 2 | 15-25% | ~2x FDK | Light smoothing |
| 3 | 25-35% | ~3x FDK | Standard clinical (recommended) |
| 4 | 30-42% | ~5x FDK | Strong smoothing |
| 5 | 35-50% | ~8x FDK | Maximum noise reduction |

# Example
```julia
params = get_hir_params(3)
# HIRParams(3, 4.0f0, 15, 0.06f0, 0.3f0, (30, 42))
```
"""
function get_hir_params(strength::Int)
    1 ≤ strength ≤ 5 || error("Strength must be 1-5, got $strength")

    # Parameters tuned via sensitivity analysis (v29.0 HIR-DISCOVER/HIR-FIX-WEIGHTS).
    # Previous values had Huber δ ~70x too small and λ ~200x too weak.
    # Key insight: V_inv normalization (~0.03) and stat_weights (~0.14 mean)
    # suppress the effective regularization, so λ must be O(1-10).
    # Relaxation < 1.0 is needed for stability at higher λ.
    #
    # Ordered Subsets (OS-PWLS): n_subsets=12 with nepochs epochs.
    # One OS epoch with M subsets ≈ M full iterations of convergence.
    # niter is preserved for legacy (n_subsets=0) backward compatibility.
    #                    strength, lambda, niter, nepochs, n_subsets, huber_delta, relaxation, target_noise_reduction
    params = Dict(
        1 => HIRParams(1, 1.0f0,   8, 1, 12, 0.08f0, 0.5f0,  (8, 15)),
        2 => HIRParams(2, 2.0f0,  15, 2, 12, 0.07f0, 0.4f0,  (15, 25)),
        3 => HIRParams(3, 4.0f0,  30, 2, 12, 0.06f0, 0.35f0, (25, 35)),
        4 => HIRParams(4, 6.0f0,  60, 3, 12, 0.05f0, 0.3f0,  (30, 42)),
        5 => HIRParams(5, 6.0f0, 100, 4, 12, 0.05f0, 0.3f0,  (35, 50)),
    )

    return params[strength]
end

# =============================================================================
# Hybrid IR Reconstruction (Main API)
# =============================================================================

"""
    hybrid_ir_reconstruct(sinogram, geometry, volume_size; strength=3, ...)

TRUE Hybrid IR reconstruction using PWLS with FDK initialization.

This implements vendor-general Hybrid IR as described in clinical literature:
1. FDK reconstruction provides fast, high-quality initialization
2. PWLS with statistical weights refines the image iteratively
3. Huber regularization preserves edges while reducing noise
4. NO final blending — returns the iteratively refined result directly

# Arguments
- `sinogram`: Measured sinogram (log-transformed) [n_cols, n_rows, n_angles]
- `geometry`: CTGeometry with scanner parameters
- `volume_size`: (nx, ny, nz) output volume dimensions

# Keyword Arguments
- `strength`: Noise reduction level 1-5 (default: 3, standard clinical)
  - 1: Minimal (~10% noise reduction, preserves FBP texture)
  - 2: Light (~20% noise reduction)
  - 3: Standard clinical (~30% noise reduction, recommended)
  - 4: Strong (~37% noise reduction)
  - 5: Maximum (~38% noise reduction, limited by SIRT ceiling)
- `filter`: FDK filter type (RampFilter(), etc.) (default: RampFilter())
- `verbose`: Print progress (default: false)

# Returns
Reconstructed volume [nx, ny, nz] — the PWLS-refined result (NOT a blend)

# Example
```julia
# Standard clinical reconstruction
recon = hybrid_ir_reconstruct(sinogram, geom, (256, 256, 128); strength=3)

# Maximum noise reduction
recon = hybrid_ir_reconstruct(sinogram, geom, (256, 256, 128); strength=5)

# Minimal processing (preserve texture for lung imaging)
recon = hybrid_ir_reconstruct(sinogram, geom, (256, 256, 128); strength=1)
```

# Clinical Guidelines (similar to ASIR-V/SAFIRE)
- **Level 1**: Use when FBP texture is critical (e.g., lung nodules, emphysema)
- **Level 2-3**: General clinical imaging, good balance of noise/texture
- **Level 4**: Higher dose reduction needed, some texture loss acceptable
- **Level 5**: Maximum dose reduction, may affect diagnostic texture

# Vendor Equivalents
- GE ASIR-V: strength 1-5 ≈ ASIR-V 20%, 40%, 60%, 80%, 100%
- Siemens SAFIRE: strength 1-5 = SAFIRE S1-S5
- Philips iDose4: strength 1-5 ≈ iDose4 levels 1-5
- Canon AIDR 3D: strength 2/3/4 ≈ Mild/Standard/Strong

See also: [`get_hir_params`](@ref), [`pwls_reconstruct`](@ref), [`fdk_reconstruct`](@ref)
"""
function hybrid_ir_reconstruct(
    sinogram::AbstractArray{T, 3},
    geometry::CTGeometry,
    volume_size::NTuple{3, Int};
    strength::Int = 3,
    filter::FilterType = StandardFilter(),
    verbose::Bool = false
) where T <: AbstractFloat

    # 1. Get research-based parameters for this strength level
    params = get_hir_params(strength)

    verbose && println("Hybrid IR reconstruction (Strength $strength)...")
    verbose && println("  Parameters: λ=$(params.lambda), n_subsets=$(params.n_subsets), nepochs=$(params.nepochs), δ=$(params.huber_delta), relax=$(params.relaxation)")
    verbose && println("  Expected noise reduction: $(params.target_noise_reduction[1])-$(params.target_noise_reduction[2])%")

    # 2. FDK initialization (fast warm start)
    verbose && println("  Step 1: FDK initialization...")
    x = fdk_reconstruct(sinogram, geometry, volume_size; filter)

    # 3. PWLS refinement with Huber regularization
    # This is TRUE HIR — iterative refinement using measured data
    verbose && println("  Step 2: PWLS refinement ($(params.niter) iterations)...")
    pwls_reconstruct!(x, sinogram, geometry;
        niter = params.niter,
        lambda = params.lambda,
        penalty = HuberPenalty(params.huber_delta),
        relaxation = params.relaxation,
        update_weights = true,
        verbose = false  # pwls has its own verbosity
    )

    verbose && println("  Done.")

    # 4. Return refined result (NO blending!)
    return x
end

# =============================================================================
# Convenience Aliases
# =============================================================================

# Backwards compatibility with old API
# These functions call hybrid_ir_reconstruct with appropriate mappings

"""
    hir_reconstruct(sinogram, geometry, volume_size; strength=3, ...)

Alias for [`hybrid_ir_reconstruct`](@ref).
"""
const hir_reconstruct = hybrid_ir_reconstruct
