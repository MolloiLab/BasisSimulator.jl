# Memory probes, deterministic GPU release and tile helpers for the spectral workspaces.
#
# `Sys.free_memory()` is the right probe on the CPU and on unified-memory devices (Apple
# Silicon Metal, integrated GPUs); it over-reports on discrete GPUs, whose VRAM it cannot see —
# there, pass `mem_budget_GB = ...` where a workspace accepts it.

const _OOM_MAX_RETRIES       = 4
const _PCCT_DEVICE_SAFETY_FACTOR = 0.5

"""
    estimate_pcct_workspace_bytes(sino_shape, volume_shape, n_bins, binning_factor;
                                  T = Float32)

Conservative byte estimate for the persistent PCCT workspace buffers. The GPU
estimate includes binned and native-resolution bins/scratch/output storage;
the host estimate includes the two full sinogram noise staging buffers.
Small spectral tables and geometry vectors are covered by a 16 MiB allowance.
"""
function estimate_pcct_workspace_bytes(
        sino_shape::NTuple{3, <:Integer},
        volume_shape::NTuple{3, <:Integer},
        n_bins::Integer,
        binning_factor::Integer;
        T::Type = Float32,
    )
    n_bins > 0 || error("n_bins must be positive")
    binning_factor > 0 || error("binning_factor must be positive")
    bytes = sizeof(T)
    binned = prod(Int.(sino_shape)) * bytes
    volume = prod(Int.(volume_shape)) * bytes
    # bins K + sino/scratch/combined/tube scratch 4 + flat outputs K
    gpu = (2Int(n_bins) + 4) * binned + volume
    if binning_factor > 1
        native_shape = (
            Int(sino_shape[1]) * Int(binning_factor),
            Int(sino_shape[2]) * Int(binning_factor),
            Int(sino_shape[3]),
        )
        native = prod(native_shape) * bytes
        # native bins K + native sino 1 + native flat outputs K
        gpu += (2Int(n_bins) + 1) * native
    end
    gpu += 16 * 2^20
    host = 2 * binned + 16 * 2^20
    (gpu_bytes = gpu, host_bytes = host, total_bytes = gpu + host)
end

"""
    check_pcct_workspace_budget(template, estimate)

Reject a PCCT workspace allocation before it begins when the backend reports
a recommended device working set and the estimated workspace would consume
more than half of the currently unallocated portion, so a second retained
workspace is refused instead of exhausting device memory.
"""
function check_pcct_workspace_budget(template, estimate)
    snap = backend_memory_snapshot(template)
    working = snap.device_working_set_bytes
    allocated = snap.device_allocated_bytes
    if !ismissing(working) && !ismissing(allocated)
        available = max(Int(working) - Int(allocated), 0)
        allowed = floor(Int, _PCCT_DEVICE_SAFETY_FACTOR * available)
        estimate.gpu_bytes <= allowed || error(
            "PCCT workspace requires approximately " *
            Base.format_bytes(estimate.gpu_bytes) *
            " of GPU/unified memory, exceeding the safe allocation budget " *
            Base.format_bytes(allowed) * ". Release the existing PCCT/FDK " *
            "workspace (BasisSimulator.release_backend!) before creating " *
            "another, or reduce views/detector size/binning.",
        )
    end
    (; estimate..., snapshot = snap)
end

"""
    backend_memory_snapshot(template = nothing) -> NamedTuple

Return host memory and, when the array backend exposes it, device allocation
information for long-running notebook audits. No GPU package is imported.
"""
function backend_memory_snapshot(template = nothing)
    host = (
        host_free_bytes = Int(Sys.free_memory()),
        host_total_bytes = Int(Sys.total_memory()),
    )
    template === nothing && return merge(host, (
        backend = :unknown,
        device_allocated_bytes = missing,
        device_working_set_bytes = missing,
    ))
    mod = parentmodule(typeof(template))
    backend = Symbol(nameof(mod))
    if isdefined(mod, :device)
        devfun = getfield(mod, :device)
        if applicable(devfun, template)
            dev = devfun(template)
            allocated = hasproperty(dev, :currentAllocatedSize) ?
                Int(getproperty(dev, :currentAllocatedSize)) : missing
            working = hasproperty(dev, :recommendedMaxWorkingSetSize) ?
                Int(getproperty(dev, :recommendedMaxWorkingSetSize)) : missing
            return merge(host, (
                backend,
                device_allocated_bytes = allocated,
                device_working_set_bytes = working,
            ))
        end
    end
    merge(host, (
        backend,
        device_allocated_bytes = missing,
        device_working_set_bytes = missing,
    ))
end

@inline function _release_backend_array!(a::AbstractArray)
    a isa Array && return false
    mod = parentmodule(typeof(a))
    isdefined(mod, :unsafe_free!) || return false
    free! = getfield(mod, :unsafe_free!)
    applicable(free!, a) || return false
    free!(a)
    true
end

function _release_backend_arrays!(x, seen::IdSet{Any})
    x === nothing && return 0
    if x isa Array
        x in seen && return 0
        push!(seen, x)
        isbitstype(eltype(x)) && return 0
        return sum(v -> _release_backend_arrays!(v, seen), x; init = 0)
    elseif x isa AbstractArray
        x in seen && return 0
        push!(seen, x)
        return _release_backend_array!(x) ? 1 : 0
    elseif x isa Tuple || x isa NamedTuple
        return sum(v -> _release_backend_arrays!(v, seen), x; init = 0)
    end
    T = typeof(x)
    ismutabletype(T) || return 0
    x in seen && return 0
    push!(seen, x)
    sum(fieldnames(T); init = 0) do name
        _release_backend_arrays!(getfield(x, name), seen)
    end
end

"""
    release_backend!(object; collect = true) -> Int

Deterministically release GPU arrays reachable from a workspace after all
required values have been copied to the CPU. The object must not be used
afterward. CPU arrays are untouched. Returns the number of released handles.
"""
function release_backend!(object; collect::Bool = true)
    released = _release_backend_arrays!(object, IdSet{Any}())
    collect && GC.gc(true)
    released
end

# ─────────────────────────────────────────────────────────────────────
"""
    tile_ranges(n_view, tile_size) -> generator of UnitRange{Int}

Iterate the tile boundaries that cover `1:n_view` with steps of `tile_size`.
The last range is shorter when `n_view` isn't a multiple of `tile_size`.
"""
@inline tile_ranges(n_view::Integer, tile_size::Integer) =
    (k₁:min(k₁ + Int(tile_size) - 1, Int(n_view)) for k₁ in 1:Int(tile_size):Int(n_view))

# ─────────────────────────────────────────────────────────────────────
"""
    with_oom_retry(allocate, label, tile_size; max_retries = 4) -> ws

Call `allocate(tile_size)`.  On `OutOfMemoryError` / `OutOfGPUMemoryError`,
halve `tile_size` and retry, up to `max_retries` times.  Loud `@warn` per
retry, hard `error` if even `tile_size = 1` fails.
"""
function with_oom_retry(allocate::Function, label::AbstractString,
                         tile_size::Int; max_retries::Int = _OOM_MAX_RETRIES)
    cur = tile_size
    for attempt in 0:max_retries
        try
            return allocate(cur)
        catch e
            _is_oom(e) || rethrow()
            if cur <= 1
                error("$label: out of memory at tile_size=1 for this sinogram shape. " *
                      "Pass `mem_budget_GB = ...` to override the probe, or downsize the input.")
            end
            # Free the failed attempt's partial allocations before retrying —
            # otherwise the next allocation sees the dead arrays still pinning
            # GPU memory until a future GC, defeating the retry.
            GC.gc(true)
            new_tile = max(1, cur ÷ 2)
            @warn "$label OOM at tile_size=$cur; halving to $new_tile and retrying" attempt
            cur = new_tile
        end
    end
    error("$label: exhausted $max_retries OOM retries (final tile_size = $cur).")
end

# Backend-agnostic OOM detection — match by exception name string so we
# don't have to import Metal or CUDA in the main module.
_is_oom(e::Exception) = let n = string(typeof(e))
    occursin("OutOfGPUMemoryError", n) || occursin("OutOfMemoryError", n)
end
