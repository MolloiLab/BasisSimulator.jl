# =============================================================================
# Gantry rotation during a view (view integration)
# =============================================================================
#
# A detector integrates for the whole period of a view while the gantry turns, so each reading is
# the transmitted intensity averaged over the arc the gantry sweeps in that time. At radius r from
# the isocentre the object is smeared over an arc of length r · view_arc · Δθ, a blur of the signal
# alone: the quanta are counted once per view, so the noise of adjacent views stays independent.
# Point views (the default, `view_samples = 1`) omit it. The average is sampled by the midpoint rule
# at `view_samples` sub-views and taken in the intensity domain, where the detector integrates,
# before scatter, noise and every detector effect that follows (CatSim samples it the same way).

"""
    view_sample_offsets(geom, sim_opts) -> Vector{Float64}

The angular offsets (rad) of the sub-views that sample a view's integration arc: the midpoints of
`sim_opts.view_samples` equal parts of the arc `sim_opts.view_arc × Δθ` centred on the view, with
`Δθ` the geometry's view spacing. `[0.0]` for point views.
"""
function view_sample_offsets(geom::CTGeometry, sim_opts::SimOptions)
    m = sim_opts.view_samples
    m == 1 && return [0.0]
    geom.n_angles >= 2 || throw(ArgumentError("view integration needs at least two views"))
    arc = sim_opts.view_arc * (geom.angles[2] - geom.angles[1])
    return [arc * ((s - 0.5) / m - 0.5) for s in 1:m]
end

# Copy the view geometry of `g` into the workspace's backend arrays.
function _load_view_geometry!(sp, dc, du, dv, g::CTGeometry)
    T = eltype(sp)
    copyto!(sp, T.(g.source_positions)); copyto!(dc, T.(g.detector_centers))
    copyto!(du, T.(g.detector_u)); copyto!(dv, T.(g.detector_v))
    return nothing
end

# The path caches for sub-view `s` of `n`: `paths` itself for point views, element `s` of a vector.
function _view_paths(paths, s::Integer, n::Integer)
    paths === nothing && return nothing
    if paths isa AbstractVector
        length(paths) == n || throw(ArgumentError(
            "paths holds $(length(paths)) caches for $(n) sub-views; walk one per offset of view_sample_offsets"))
        return paths[s]
    end
    n == 1 || throw(ArgumentError(
        "view_samples = $(n) needs one path cache per sub-view: pass material_paths(ws, phantom; angle_offset = δ) for each δ of view_sample_offsets"))
    return paths
end

"""
    _integrate_views!(project!, geom, sim_opts, load!) -> outputs

The one pathway by which every detector family integrates a view over its arc. `project!(s, g)`
forward-projects sub-view `s` through geometry `g` and returns its outputs (log line integrals —
the energy-integrated signal, or each energy bin's counts, both linear in the intensity);
`load!(g)` loads a geometry into the workspace's backend arrays. Point views project once through
`geom`. Otherwise every sub-view of [`view_sample_offsets`](@ref) is projected, the transmissions
averaged, and the outputs of the last projection overwritten with `−log` of the mean; the
workspace geometry is restored even if a projection throws.
"""
function _integrate_views!(project!, geom::CTGeometry, sim_opts::SimOptions, load!)
    offsets = view_sample_offsets(geom, sim_opts)
    length(offsets) == 1 && return project!(1, geom)
    acc = nothing
    out = nothing
    try
        for (s, δ) in enumerate(offsets)
            g = rotate_geometry(geom, δ)
            load!(g)
            out = project!(s, g)
            acc === nothing && (acc = [fill!(similar(o), zero(eltype(o))) for o in out])
            for (a, o) in zip(acc, out)
                a .+= exp.(.-o)
            end
        end
    finally
        load!(geom)
    end
    n = eltype(first(acc))(length(offsets))
    for (a, o) in zip(acc, out)
        o .= .-log.(a ./ n)
        release_backend!(a; collect = false)
    end
    return out
end

export view_sample_offsets
