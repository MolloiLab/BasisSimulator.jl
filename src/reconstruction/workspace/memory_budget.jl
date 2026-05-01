"""
Memory-budget probe + tile-loop helpers shared by the VMI workspaces
(`RwlsWorkspace`, `PwlsWorkspace`, `CongWorkspace`, `MonoPlusWorkspace`).

Free utility functions only — no traits, no abstract types.  Concrete
workspaces consume these in their `create_*_workspace(...)` factories and
in their `apply_*!(...)` tile loops.  Once we have four data points we'll
extract a proper abstract contract (post-VMI follow-up).

# Backend-agnostic by design
The probe queries `Sys.free_memory()`, which is correct on:
  • CPU (`Array`)
  • Apple Silicon Metal (unified memory — CPU and GPU share the pool)
  • Integrated GPUs

For **discrete GPUs** (CUDA / ROCm / Intel Arc), the probe over-reports —
discrete VRAM is not visible to `Sys.free_memory`.  Discrete-GPU users
must pass `mem_budget_GB = ...` explicitly to override.  This caveat is
documented per-workspace.

# Failure semantics
`suggest_tile_size` returns the largest tile that fits given a per-view
byte cost and budget × safety factor (0.6).  Allocation can still fail —
workspace constructors wrap the alloc in `with_oom_retry` which halves
the tile on `OutOfMemoryError` / `OutOfGPUMemoryError` up to 4 times.
"""

const _DEFAULT_SAFETY_FACTOR = 0.6
const _OOM_MAX_RETRIES       = 4

# ─────────────────────────────────────────────────────────────────────
"""
    suggest_tile_size(per_view_bytes, n_view; mem_budget_GB = nothing) -> Int

Largest tile size (clamped to `[1, n_view]`) such that
`per_view_bytes × tile_size ≤ budget`, where `budget` is either the user's
`mem_budget_GB · 2³⁰` override, or `Sys.free_memory() · 0.6`.

Hint, not a guarantee — the workspace constructor wraps allocation in
`with_oom_retry` because `Sys.free_memory()` and Metal's
`recommendedMaxWorkingSetSize` are both hints, not contracts.
"""
function suggest_tile_size(per_view_bytes::Integer,
                            n_view::Integer;
                            mem_budget_GB::Union{Nothing, Real} = nothing) :: Int
    per_view_bytes > 0 || error("suggest_tile_size: per_view_bytes must be > 0.")
    n_view > 0         || error("suggest_tile_size: n_view must be > 0.")
    avail = if mem_budget_GB === nothing
        floor(Int, Float64(Sys.free_memory()) * _DEFAULT_SAFETY_FACTOR)
    else
        floor(Int, Float64(mem_budget_GB) * 2^30)
    end
    avail > 0 || error("suggest_tile_size: zero available memory; pass `mem_budget_GB` to override.")
    clamp(avail ÷ Int(per_view_bytes), 1, Int(n_view))
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
