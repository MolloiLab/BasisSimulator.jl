# =============================================================================
# Projector selection — Distance-Driven (default) vs Siddon
# =============================================================================
#
# The `dd_*` and `siddon_*` forward projectors share byte-identical signatures
# (the DD port preserved the Siddon API exactly), so a single run-level
# `projector::Symbol` knob can pick between them at every forward-projection
# site without any other code change.
#
#   :dd_fast — DEFAULT.  Distance-driven (DD3), anti-aliased footprint integration.
#              Single-pass per-material path-length fused kernels: the full spectrum
#              is produced in ONE volume walk instead of the K=16 tiled per-energy
#              re-walks of the dd.jl kernels (same footprint/overlap weights; results
#              agree to float ordering).  Measured 47x faster on the 234-bin
#              polychromatic forward path (M4 Metal).  Mono projection and the
#              transpose use the dd.jl kernels directly.  Supports ≤ 64 materials;
#              warns and falls back to the dd.jl tiled kernels above that.
#   :siddon  — Siddon exact ray tracing retained for comparison and
#              compatibility. It point-samples one voxel per step, can ALIAS
#              in severe beam-hardened regions, and is slower than :dd_fast
#              for full polychromatic/spectral simulations.
#
# Consistency contract: the forward simulation, the iterative-recon system
# matrix (A·x and W = 1/(A·1)), and the BHC correction all read the SAME
# `projector` symbol (same `:dd_fast` default), so the model is self-consistent —
# the recon inverts the operator that generated the data.  The voxel-driven
# back-projector (`backproject!`) has no DD/Siddon variant and is unchanged.
#
# These helpers branch on a runtime Symbol (one cheap comparison) rather than
# `Val` dispatch: they are called per-energy / per-tile / per-subset, never in
# the per-voxel hot loop, so the branch cost is negligible and call sites stay
# readable.
# =============================================================================

"""
    _validate_projector(p::Symbol) -> Symbol

Throw an `ArgumentError` unless `p` is `:dd_fast` or `:siddon`; return `p`
unchanged.  Call at every public entry point that accepts a projector so an
invalid symbol fails loudly instead of silently falling back to distance-driven
in the shims below.  (`:dd`, the per-energy distance-driven option, was removed
in 0.15.0; `:dd_fast` is the same model.)
"""
function _validate_projector(p::Symbol)
    (p === :dd_fast || p === :siddon) ||
        throw(ArgumentError("projector must be :dd_fast or :siddon, got :$p"))
    return p
end

# In-place monochromatic forward projection.  :dd_fast has no mono variant —
# mono has no energy loop, so it is the dd.jl kernel.
@inline _project_mono!(proj::Symbol, args...; kw...) =
    proj === :siddon ? siddon_forward_project!(args...; kw...) :
                       dd_forward_project!(args...; kw...)

# HIR subset projections use a row-tiled arc kernel.  It is algebraically
# identical to DD and falls back to the generic path for flat/Siddon geometry.
@inline _project_mono_hir!(proj::Symbol, output, volume, geom; kw...) =
    proj === :siddon ? siddon_forward_project!(output, volume, geom; kw...) :
    is_arc(geom) ? _dd_forward_project_arc_rowtile4!(output, volume, geom; kw...) :
                   dd_forward_project!(output, volume, geom; kw...)

# Allocating monochromatic forward projection.
@inline _project_mono(proj::Symbol, args...; kw...) =
    proj === :siddon ? siddon_forward_project(args...; kw...) :
                       dd_forward_project(args...; kw...)

# In-place algebraic transpose of the distance-driven mono operator (dd_transpose.jl).
# Siddon retains the legacy voxel-driven approximation until its own exact
# transpose is implemented.
@inline _backproject_mono!(proj::Symbol, volume, sinogram, geom; kw...) =
    proj === :siddon ? backproject!(volume, sinogram, geom; weighted = false, kw...) :
                       dd_backproject!(volume, sinogram, geom; kw...)

# Fused polychromatic (energy-integrating EI).  dd_fast_fused_* fall back to the
# dd.jl tiled kernels above 64 materials.
@inline _project_fused_poly!(proj::Symbol, args...; kw...) =
    proj === :siddon ? siddon_fused_poly_project!(args...; kw...) :
                       dd_fast_fused_poly_project!(args...; kw...)

# Fused spectral (photon-counting PCCT, tiled).
@inline _project_fused_spectral!(proj::Symbol, args...; kw...) =
    proj === :siddon ? siddon_fused_spectral_project!(args...; kw...) :
                       dd_fast_fused_spectral_project!(args...; kw...)
