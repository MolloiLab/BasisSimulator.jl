"""
PCCT per-bin basis + bin-count helpers for the VMI material-decomposition
pipeline.

Every stage of the PCCT VMI pipeline — CMV polynomial calibration, RWLS-GN,
PWLS-L₂ — builds the same thing from the scanner+protocol: a per-bin-group
effective spectrum (source × QE × DRM column sum), then uses it plus the
material mass-attenuation tables.  These helpers factor out that boilerplate
so callers only pick the bin grouping and hand everything else to the library.
"""

"""
    pcct_effective_spectrum(scanner, protocol;
                            sim_opts,
                            bin_groups::AbstractVector{<:AbstractVector{<:Integer}})
        -> (e::Vector{Float64}, ŵ_bins::Vector{Vector{Float32}})

Build per-bin-group effective-spectrum weights for a PCCT detector:

    ŵ_k(E)  =  w_src(E) · η(E) · Σ_{b ∈ bin_groups[k]} DRM[E, b]

All groups share the same energy grid `e`.  The returned `ŵ_bins[k]` are
Float32 vectors (length `n_E`) aligned with `e`; the downstream library
(`apply_rwls_pcct!`, `apply_pwls!`, etc.) handles any per-bin renormalization
internally.

# Keyword arguments
- `sim_opts`   : `BS.SimOptions`; only its flags for the resolved spectrum matter.
- `bin_groups` : list of PCCT bin index groups.  Indices are 1-based into the
  detector's DRM columns.  For Siemens Alpha: `[[1, 2], [3], [4]]` is the
  classic 3-bin RWLS grouping; `[[1, 2], [3, 4]]` is the 2-bin CMV / PWLS split.
"""
function pcct_effective_spectrum(
        scanner,
        protocol;
        sim_opts,
        bin_groups::AbstractVector{<:AbstractVector{<:Integer}},
    )
    e_full, w_full = resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner = scanner)
    pcct_det = _build_pcct_detector(scanner)
    kVp      = Float64(maximum(e_full))
    R_mat    = compute_mc_drm(pcct_det, kVp)
    η_vec    = quantum_efficiency_vector(pcct_det.material, pcct_det.thickness_mm, e_full)
    n_R      = size(R_mat, 1)
    drm_row(E) = clamp(round(Int, (Float64(E) - 1.0) / (kVp - 1.0) * (n_R - 1)) + 1, 1, n_R)

    e = Float64.(e_full)
    n_E = length(e)
    ŵ_bins = Vector{Vector{Float32}}(undef, length(bin_groups))
    for (k, grp) in enumerate(bin_groups)
        wk = [Float64(w_full[i]) * Float64(η_vec[i]) *
              sum(R_mat[drm_row(e[i]), b] for b in grp)
              for i in 1:n_E]
        total = sum(wk)
        total > 0 ||
            error("pcct_effective_spectrum: bin_groups[$k] = $(grp) has zero spectral weight.")
        ŵ_bins[k] = Float32.(wk ./ total)     # normalize to Σ_k ŵ = 1
    end
    (e = e, ŵ_bins = ŵ_bins)
end

"""
    pcct_rwls_basis(scanner, protocol;
                    sim_opts,
                    bin_groups,
                    water_material = XA.Materials.water,
                    iodine_material = XA.Elements.Iodine)
        -> NamedTuple (ŵ_bins, p, q)

One-shot basis builder for `apply_rwls_pcct!`.  Wraps
`pcct_effective_spectrum` and adds the iodine / water mass-attenuation
tables `p(E)`, `q(E)` (both Float32, length `n_E`), yielding a basis
NamedTuple ready to pass as the `basis = ...` kwarg.
"""
function pcct_rwls_basis(
        scanner,
        protocol;
        sim_opts,
        bin_groups,
        water_material  = XA.Materials.water,
        iodine_material = XA.Elements.Iodine,
    )
    e, ŵ_bins = pcct_effective_spectrum(scanner, protocol;
                                        sim_opts = sim_opts, bin_groups = bin_groups)
    p = Float32[Float32(compute_mass_μ_at_energy(iodine_material, E)) for E in e]
    q = Float32[Float32(compute_mass_μ_at_energy(water_material,  E)) for E in e]
    (ŵ_bins = ŵ_bins, p = p, q = q)
end

"""
    combine_pcct_bin_counts(bins, I0_bins, bin_groups; T = Float32)
        -> (counts::Vector{Array{T, 3}}, I0::Vector{Float64})

Reduce a 4-bin (or N-bin) PCCT sinogram stack + per-bin I0 scalars into the
grouped counts + grouped I0 vectors that `apply_rwls_pcct!` expects:

    counts[k][px] = Σ_{b ∈ bin_groups[k]} I0_bins[b] · exp(-bins[b][px])
    I0[k]         = Σ_{b ∈ bin_groups[k]} I0_bins[b]

Scatter-correct the bins *before* this call; this helper just sums the
count-domain contributions per group.  Runs on CPU; stage to GPU afterwards
if needed (`MtlArray`, `CuArray`).
"""
function combine_pcct_bin_counts(
        bins::AbstractVector,
        I0_bins::AbstractVector{<:Real},
        bin_groups::AbstractVector{<:AbstractVector{<:Integer}};
        T::Type = Float32,
    )
    length(bins) == length(I0_bins) ||
        error("combine_pcct_bin_counts: length(bins) = $(length(bins)) ≠ length(I0_bins) = $(length(I0_bins)).")
    shape = size(first(bins))
    n_groups = length(bin_groups)
    counts = Vector{Array{T, 3}}(undef, n_groups)
    I0     = Vector{Float64}(undef, n_groups)
    for (k, grp) in enumerate(bin_groups)
        buf = zeros(T, shape)
        for b in grp
            I0b = T(I0_bins[b])
            @. buf += I0b * exp(-bins[b])
        end
        counts[k] = buf
        I0[k]     = Float64(sum(I0_bins[b] for b in grp))
    end
    (counts = counts, I0 = I0)
end

"""
    pcct_pwls_basis(scanner, protocol;
                    sim_opts,
                    low_bins = 1:2, high_bins = 3:4,
                    water_material = XA.Materials.water,
                    iodine_material = XA.Elements.Iodine)
        -> NamedTuple (ŵ_bins, p, q)

Convenience wrapper around `pcct_rwls_basis` for the 2-bin PWLS case.  Same
returned structure — `basis.ŵ_bins` has exactly two entries (low / high) that
the 2×2 matrix-curvature solver consumes.
"""
function pcct_pwls_basis(
        scanner,
        protocol;
        sim_opts,
        low_bins  = 1:2,
        high_bins = 3:4,
        water_material  = XA.Materials.water,
        iodine_material = XA.Elements.Iodine,
    )
    pcct_rwls_basis(scanner, protocol;
                    sim_opts = sim_opts,
                    bin_groups = [collect(low_bins), collect(high_bins)],
                    water_material  = water_material,
                    iodine_material = iodine_material)
end

export pcct_effective_spectrum, pcct_rwls_basis, pcct_pwls_basis, combine_pcct_bin_counts
