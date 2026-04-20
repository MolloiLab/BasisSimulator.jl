"""
VMI synthesis from photo/compton basis images.

Image-domain linear combination ([Cong 2022 Eq 4]):

    μ(E, r) = p(E)·a(r) + q(E)·c(r)

Uses the same analytic `p(ε), q(ε)` that drove the decomposition → the
synthesis is internally consistent with the inversion, so water voxels
synthesise back to XrayAttenuation's water `μ(E)` within ~1% across the
diagnostic range.  HU referenced to XA's linear `μ` of water at each
target energy.
"""

"""
    synth_vmi_hu(a, c, energies;
                 μ_water_fn = E -> compute_μ_at_energy(XA.Materials.water, E),
                 fov_mask_radius_frac = 0.5)

Synthesise Virtual Monoenergetic HU volumes at each target energy.

# Arguments
- `a`, `c`    : photoelectric / Compton basis images (Float32, 3D)
- `energies`  : Vector of target energies in keV

# Keyword arguments
- `μ_water_fn` : function `E → μ_water(E)` in cm⁻¹ (default: XA water)
- `fov_mask_radius_frac` : radial fraction of the transverse matrix
  inside which voxels are kept; voxels outside set to -1000 HU.  Pass
  `nothing` to disable masking.

Returns:
- `energies` : echoed input
- `volumes`  : Vector{Array{Float32,3}} aligned to `energies`
"""
function synth_vmi_hu(
        a::AbstractArray{Float32, 3},
        c::AbstractArray{Float32, 3},
        energies::AbstractVector;
        μ_water_fn = (E -> compute_μ_at_energy(XA.Materials.water, Float64(E))),
        fov_mask_radius_frac = 0.5,
        verbose::Bool = true,
    )
    nx, ny, nz = size(a)
    cx, cy = (nx + 1) / 2, (ny + 1) / 2
    mask = fov_mask_radius_frac !== nothing
    r_fov_sq = mask ? (Float64(fov_mask_radius_frac) * nx)^2 : -1.0

    volumes = Vector{Array{Float32, 3}}(undef, length(energies))

    for (k, E_keV) in enumerate(energies)
        E_f   = Float64(E_keV)
        p_E   = Float32(p_photoelectric(E_f))
        q_E   = Float32(q_compton(E_f))
        μ_vol = @. p_E * a + q_E * c                   # cm⁻¹
        μ_w   = Float64(μ_water_fn(E_f))
        hu    = Float32.(to_hounsfield(μ_vol; μ_water = μ_w))
        if mask
            @inbounds for k2 in 1:nz, j in 1:ny, i in 1:nx
                if (i - cx)^2 + (j - cy)^2 > r_fov_sq
                    hu[i, j, k2] = -1000f0
                end
            end
        end
        volumes[k] = hu
    end

    if verbose
        @info "VMI synthesis at $(Int.(Float64.(energies))) keV — done"
        for (E, vol) in zip(energies, volumes)
            if size(vol, 1) >= 384 && size(vol, 2) >= 384
                mid = size(vol, 3) ÷ 2
                inner = vol[128:384, 128:384, mid]
                @info "  $(Int(Float64(E))) keV: body-ROI HU range [$(round(quantile(vec(inner), 0.01); digits = 1)), $(round(quantile(vec(inner), 0.99); digits = 1))]"
            end
        end
    end

    (energies = collect(energies), volumes = volumes)
end

export synth_vmi_hu
