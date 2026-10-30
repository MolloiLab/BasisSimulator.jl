# =============================================================================
# Running many simulations at once
# =============================================================================
# Geant4 allows one run manager per process, and its own worker threads do not speed
# BasisMonteCarlo up (every step calls back into Julia, and those calls run one at a time). So
# parallelism is by process: each worker process owns its own Geant4 engine, and a sweep hands
# each worker whole runs (one detector at one energy). The seed of a run depends only on
# (seed, energy), so the results are identical however the runs are spread over workers.
# =============================================================================

const WORKER_POOL = Int[]

# About a third of the machine, so a sweep never crowds out other work on the node
default_worker_count() = max(1, Sys.CPU_THREADS ÷ 3)

"""
    start_workers(n = default_worker_count()) -> Vector{Int}

Make sure `n` worker processes are running for parallel sweeps, each with the current Julia
project and `BasisMonteCarlo` loaded, starting only as many new ones as are missing (so running
it again, as a notebook cell does, changes nothing). Every later
[`simulate_monoenergetic`](@ref) over several energies and every [`simulate_sweep`](@ref) then
spreads its runs over them. Each worker starts its own Geant4 the first time it runs (a few
seconds); after that a sweep runs about `n` times faster.

The default takes about a third of the machine's cores (at least 1), leaving the rest for
whatever else runs on it. [`stop_workers`](@ref) stops them.
"""
function start_workers(n::Integer = default_worker_count())
    filter!(in(Distributed.workers()), WORKER_POOL)    # forget any that have died
    missing_count = n - length(WORKER_POOL)
    if missing_count > 0
        project = Base.active_project()
        pids = Distributed.addprocs(missing_count; exeflags = "--project=$(project)")
        Distributed.remotecall_eval(Main, pids, :(using BasisMonteCarlo))
        append!(WORKER_POOL, pids)
    end
    return copy(WORKER_POOL)
end

"""
    stop_workers()

Stop the worker processes started by [`start_workers`](@ref).
"""
function stop_workers()
    isempty(WORKER_POOL) && return nothing
    Distributed.rmprocs(WORKER_POOL)
    empty!(WORKER_POOL)
    return nothing
end

"""
    workers() -> Vector{Int}

The worker processes [`start_workers`](@ref) started (empty when sweeps run in this process).
"""
workers() = copy(WORKER_POOL)

"""
    simulate_sweep(jobs; n = 100_000, beam = PencilBeam(), physics = PhysicsSettings(), seed = 1,
                   readout = nothing) -> Vector

Run every job in `jobs`, in order, and return one result per job. A job is
`(detector, energy_keV)`, `(detector, spectrum)` with a [`Spectrum`](@ref), or either with a
third element, its own seed (to split one large run into independent chunks), a fourth, its
own beam (otherwise `beam`), and a fifth, its own number of photons (otherwise `n`). With workers
started by [`start_workers`](@ref), the jobs are spread over them; otherwise they run one after
another in this process. The results are the same either way.

Each result is the [`MonoenergeticRun`](@ref), or, given a readout, the run reduced where it
ran, so only the small result travels back: a [`DexelReadout`](@ref) gives a
[`DexelResponse`](@ref), an [`ArcReadout`](@ref) an [`ArcCounts`](@ref).

Grouping the jobs by detector keeps Geant4 from rebuilding a geometry it already has; a sweep
whose detector changes with energy (a thickness rule, say) rebuilds it for every run, which
costs a fraction of a second each (a whole scanner's detector, a few seconds).
"""
function simulate_sweep(
        jobs::AbstractVector;
        n::Integer = 100_000,
        beam::Beam = PencilBeam(),
        physics::PhysicsSettings = PhysicsSettings(),
        seed::Integer = 1,
        readout = nothing,
    )
    function run_one(job)
        detector, source = job[1], job[2]
        job_seed = length(job) >= 3 ? job[3] : seed
        job_beam = length(job) >= 4 ? job[4] : beam
        job_n = length(job) >= 5 ? job[5] : n
        record_deposits = needs_deposits(readout)
        record_vertices = needs_vertices(readout)
        run = if source isa Spectrum
            simulate_spectrum(
                detector,
                source;
                n = job_n,
                beam = job_beam,
                physics,
                seed = job_seed,
                record_deposits,
                record_vertices,
            )
        else
            simulate_monoenergetic(
                detector,
                Float64(source);
                n = job_n,
                beam = job_beam,
                physics,
                seed = job_seed,
                record_deposits,
                record_vertices,
            )
        end
        return reduce_run(run, readout)
    end

    isempty(WORKER_POOL) && return [run_one(job) for job in jobs]

    pool = Distributed.WorkerPool(WORKER_POOL)
    return Distributed.pmap(run_one, pool, collect(jobs))
end
