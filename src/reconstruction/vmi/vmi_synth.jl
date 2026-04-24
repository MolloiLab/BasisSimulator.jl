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

"""
    synth_vmi_sino_domain(sino_a_by_E, sino_b_by_E, energies;
                          μρ_a_by_E, μρ_b_by_E,
                          fbp_workspace_builder, fbp_recon!, geom, matrix_size,
                          μ_water_fn = (E -> compute_μ_at_energy(XA.Materials.water, Float64(E))),
                          fov_mask_radius_frac = 0.5,
                          verbose = true) -> (energies, volumes)

Sinogram-domain VMI synthesis (matches the pattern in notebooks 06/07).
At each target keV, a linear combine happens in the **sinogram** domain
before a single FBP:

    sino(E) = μρ_a(E) · sino_a(E) + μρ_b(E) · sino_b(E)   [cm⁻¹·cm]
    μ(E)   = FBP( sino(E), geom, matrix_size )             [cm⁻¹]
    HU(E)  = 1000 · (μ(E) − μ_water(E)) / μ_water(E)

This is the correct choice when the decomposition is basis-independent
(water/iodine) and the material sinograms may vary per energy (e.g. ACNR
produces a separate `(sino_a, sino_b)` per energy).  When you already
have reconstructed basis volumes, use `synth_vmi_hu` (image-domain,
fixed basis).

# Arguments
- `sino_a_by_E` : `Dict{Float64, AbstractArray{Float32,3}}` — basis-a
  sinograms keyed by target energy (g/cm²).  `Float64(E)` keys.
- `sino_b_by_E` : same for basis b.
- `energies`    : target keVs (will be converted to `Float64`).

# Keyword arguments
- `μρ_a_by_E`, `μρ_b_by_E` : vectors aligned with `energies` — mass
  attenuations `μρ_a(E_k)`, `μρ_b(E_k)` in cm²/g.
- `fbp_workspace_builder`  : `(sino_gpu, geom, matrix_size) -> ws`.
  Returns an FBP workspace (e.g. from `BS.create_fdk_recon_workspace`).
- `fbp_recon!`             : `(ws, sino_gpu, geom, matrix_size) -> μ_vol`.
  Runs the FBP (e.g. `BS.reconstruct!`).
- `geom`, `matrix_size`    : forwarded to the workspace builder + FBP call.
- `μ_water_fn`             : `E → μ_water(E)` in cm⁻¹ (default XA water).
- `fov_mask_radius_frac`   : radial FOV keep fraction (`nothing` = no mask).

Returns `(energies, volumes::Vector{Array{Float32,3}})`.

The workspace builder/FBP callback lets this function stay backend
agnostic — the caller picks `Array` / `MtlArray` / `CuArray` staging.
"""
function synth_vmi_sino_domain(
        sino_a_by_E::AbstractDict,
        sino_b_by_E::AbstractDict,
        energies::AbstractVector;
        μρ_a_by_E::AbstractVector,
        μρ_b_by_E::AbstractVector,
        fbp_workspace_builder::Function,
        fbp_recon!::Function,
        geom,
        matrix_size,
        μ_water_fn = (E -> compute_μ_at_energy(XA.Materials.water, Float64(E))),
        fov_mask_radius_frac = 0.5,
        verbose::Bool = true,
    )
    E_list = Float64.(energies)
    length(E_list) == length(μρ_a_by_E) == length(μρ_b_by_E) ||
        error("synth_vmi_sino_domain: energies / μρ_a_by_E / μρ_b_by_E must match length.")

    # Pick a template sinogram to size the FBP workspace (any key works;
    # they must share shape).
    sino_template = first(values(sino_a_by_E))
    ws = fbp_workspace_builder(sino_template, geom, matrix_size)

    nx_img = Int(matrix_size[1])
    ny_img = Int(matrix_size[2])
    nz_img = Int(matrix_size[3])
    cx, cy = (nx_img + 1) / 2, (ny_img + 1) / 2
    mask   = fov_mask_radius_frac !== nothing
    r_fov_sq = mask ? (Float64(fov_mask_radius_frac) * nx_img)^2 : -1.0

    volumes = Vector{Array{Float32, 3}}(undef, length(E_list))

    t0 = time()
    for (k, E) in enumerate(E_list)
        μρ_a = Float32(μρ_a_by_E[k])
        μρ_b = Float32(μρ_b_by_E[k])

        sino_a_E = sino_a_by_E[E]
        sino_b_E = sino_b_by_E[E]
        sino_E   = @. μρ_a * sino_a_E + μρ_b * sino_b_E   # cm⁻¹·cm

        μ_vol = fbp_recon!(ws, sino_E, geom, matrix_size)  # cm⁻¹
        μ_w_E = Float64(μ_water_fn(E))
        hu = Float32.(to_hounsfield(Array(μ_vol); μ_water = μ_w_E))

        if mask
            @inbounds for kz in 1:nz_img, j in 1:ny_img, i in 1:nx_img
                if (i - cx)^2 + (j - cy)^2 > r_fov_sq
                    hu[i, j, kz] = -1000f0
                end
            end
        end
        volumes[k] = hu
    end
    dt = time() - t0

    verbose && @info "VMI sino-domain synthesis: $(Int.(E_list)) keV — $(length(E_list)) energies in $(round(dt, digits=1)) s"

    (energies = E_list, volumes = volumes)
end

export synth_vmi_sino_domain
