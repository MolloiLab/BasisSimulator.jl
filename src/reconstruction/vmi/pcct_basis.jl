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
    combine_pcct_bin_counts!(out_bins, raw_bins, I0_bins, bin_groups;
                              chunk_size = nothing) -> I0::Vector{Float64}

Streaming, in-place combine.  Writes into the **pre-allocated** `out_bins`
on whatever backend they live on (CPU `Array`, Metal `MtlArray`, …):

    out_bins[k][px] = Σ_{b ∈ bin_groups[k]} I0_bins[b] · exp(-raw_bins[b][px])

The caller pre-allocates `out_bins[k]` with `similar(template, shape...)`,
where `template` chooses the backend (e.g. an `MtlArray` for GPU).  This
is the **memory-economical path** — there is no `combo.counts` CPU
intermediate as in the legacy allocating wrapper, and `raw_bins` are
streamed chunk-by-chunk along the view axis when `raw_bins[b]` lives on
a different backend than `out_bins[k]`.

# Why this matters for unified-memory Macs
The legacy `combine_pcct_bin_counts` allocated a fresh CPU `combo.counts`
of size `n_groups × n_col × n_row × n_view × 4 B`, and the caller then
copied it to GPU with `MtlArray(c)` — *both* copies live simultaneously
in unified memory.  At PCCT scales (3 × ~1 GB) this is 6 GB of
double-allocation.  The streaming path here uses a single chunk-sized
staging buffer (~100 MB at `chunk_size = 100`) plus the GPU-resident
`out_bins`, eliminating the duplicate.

# Arguments
- `out_bins`     : `Vector` of `length(bin_groups)` pre-allocated arrays,
  matching `size(raw_bins[1])` and the desired output backend.
- `raw_bins`     : `Vector` of raw per-bin sinograms (any backend).
- `I0_bins`      : per-bin I0 scalars.
- `bin_groups`   : list of bin-index groups.

# Keyword arguments
- `chunk_size`   : tile size along dim 3 for cross-backend staging.
  `nothing` ⇒ stage in one shot (only safe when raw and out are on the
  same backend, or when raw bins are small).  Pass an explicit chunk
  (e.g. 100) for big sinograms with raw on CPU + out on GPU.

Returns the per-group `I0` aggregate vector.
"""
function combine_pcct_bin_counts!(
        out_bins::AbstractVector,
        raw_bins::AbstractVector,
        I0_bins::AbstractVector{<:Real},
        bin_groups::AbstractVector{<:AbstractVector{<:Integer}};
        chunk_size::Union{Nothing, Integer} = nothing,
    )
    length(raw_bins) == length(I0_bins) ||
        error("combine_pcct_bin_counts!: length(raw_bins) = $(length(raw_bins)) ≠ length(I0_bins) = $(length(I0_bins)).")
    length(out_bins) == length(bin_groups) ||
        error("combine_pcct_bin_counts!: length(out_bins) = $(length(out_bins)) ≠ length(bin_groups) = $(length(bin_groups)).")
    shape = size(first(raw_bins))
    for (k, ob) in enumerate(out_bins)
        size(ob) == shape ||
            error("combine_pcct_bin_counts!: out_bins[$k] shape $(size(ob)) ≠ raw bin shape $(shape).")
    end

    n_view = shape[3]
    chunk  = chunk_size === nothing ? n_view : Int(chunk_size)

    # Allocate one staging buffer on the OUTPUT backend, sized to a chunk.
    # Used only when raw_bins[b]'s backend ≠ out_bins[k]'s backend; on same
    # backend, AK.foreachindex over the raw bin directly is zero-copy.
    template = first(out_bins)
    same_backend = typeof(first(raw_bins)).name === typeof(template).name
    staging = same_backend ? nothing :
              similar(template, eltype(template), (shape[1], shape[2], chunk))

    for (k, grp) in enumerate(bin_groups)
        out_k = out_bins[k]
        fill!(out_k, zero(eltype(out_k)))
        for b in grp
            I0b   = eltype(out_k)(I0_bins[b])
            raw_b = raw_bins[b]
            if same_backend
                # Single fused kernel — out_k += I0b * exp(-raw_b).
                @. out_k = out_k + I0b * exp(-raw_b)
            else
                # Stream raw_b's chunks into the staging buffer, accumulate.
                # Materialize each chunk via INDEXING (fresh contiguous Array)
                # rather than `view(raw_b, ...)`.  `copyto!(::MtlArray, ::Array)`
                # has a fast memcpy specialization; `copyto!(::MtlArray,
                # ::SubArray{<:Any,<:Any,Array})` does not, and would fall back
                # to scalar GPU setindex (forbidden by GPUArraysCore).
                for tile_range in tile_ranges(n_view, chunk)
                    kk = length(tile_range)
                    sv = kk == chunk ? staging : view(staging, :, :, 1:kk)
                    copyto!(sv, raw_b[:, :, tile_range])
                    out_view = view(out_k, :, :, tile_range)
                    @. out_view = out_view + I0b * exp(-sv)
                end
            end
        end
    end
    staging = nothing
    [Float64(sum(I0_bins[b] for b in grp)) for grp in bin_groups]
end

"""
    combine_pcct_bin_counts(bins, I0_bins, bin_groups; T = Float32)
        -> (counts::Vector{Array{T, 3}}, I0::Vector{Float64})

Allocating CPU wrapper around `combine_pcct_bin_counts!` — kept for
notebooks that don't care about peak memory.  Output `counts` are CPU
`Array`s; stage to GPU afterwards if needed.  For PCCT-scale sinos on
unified-memory Macs, **prefer the in-place form** to avoid the 3+ GB
CPU+GPU duplicate.
"""
function combine_pcct_bin_counts(
        bins::AbstractVector,
        I0_bins::AbstractVector{<:Real},
        bin_groups::AbstractVector{<:AbstractVector{<:Integer}};
        T::Type = Float32,
    )
    shape = size(first(bins))
    n_groups = length(bin_groups)
    counts = [zeros(T, shape) for _ in 1:n_groups]
    I0 = combine_pcct_bin_counts!(counts, bins, I0_bins, bin_groups)
    (counts = counts, I0 = I0)
end

export pcct_effective_spectrum, pcct_rwls_basis,
       combine_pcct_bin_counts, combine_pcct_bin_counts!
