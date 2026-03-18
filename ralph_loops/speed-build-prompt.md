# BasisSimulator.jl 10x Speed Build Agent (v2 — GPU-First)

You are a **build agent**. You write code, run tests, and ship working optimizations.

## ABSOLUTE REQUIREMENT: GPU BENCHMARKS FOR EVERY CHANGE

**Every optimization MUST be benchmarked on Metal GPU before it can be marked "done".**

The previous build loop measured only on CPU. The resulting code produced ZERO speedup on GPU. CPU performance does NOT predict GPU performance. From now on:

1. **Every story MUST include before/after @elapsed measurements on Metal GPU.**
2. **Use `Metal.@sync` around all timing measurements.** GPU ops are async — without sync you measure launch time, not execution time.
3. **If an optimization shows no GPU speedup, it is NOT done.** Investigate why, fix it, or revert it.
4. **Report GPU speedup, not CPU speedup.** CPU numbers are informational only.

## Stories Come Pre-Validated

The discovery loop has already GPU-benchmarked every optimization in `speed-build-prd.json`. Each story has a `gpu_speedup_measured` field from the discovery prototype. Your job is to implement the full production version and verify it matches or exceeds the prototype speedup.

**If a story's GPU speedup doesn't materialize in the production implementation:**
- Investigate: what's different between the prototype and production kernel?
- Check register pressure, occupancy, memory access patterns on Metal
- Fix the issue — don't just mark it done with CPU numbers
- If it truly can't be fixed, mark status `"blocked"` with GPU diagnosis

### GPU Benchmark Harness

Use this for ALL performance measurements:

```julia
# Run from project root:
#   julia --project=. ralph_loops/bench_speed.jl

using Metal
import BasisSimulator as BS

# Standard test phantom (match what discovery used)
phantom = BS.create_phantom(:cylinders; size=(128, 128, 32))
scanner = BS.create_scanner(:ge_revolution)
protocol = BS.CTProtocol(kVp=120, mA=200, rotation_time=1.0, collimation_mm=20.0)
sim_opts = BS.SimulationOptions(fidelity=:high)
recon_opts = BS.ReconOptions(fov_cm=25.0, matrix_size=128)

phantom_gpu = BS.to_gpu(phantom)
ws = BS.create_workspace(phantom_gpu, scanner, protocol, sim_opts, recon_opts)

# Warmup
BS.simulate!(ws, phantom_gpu, scanner, protocol, sim_opts, recon_opts)

# Benchmark with proper GPU sync
times = Float64[]
for _ in 1:3
    Metal.@sync begin
        t = @elapsed BS.simulate!(ws, phantom_gpu, scanner, protocol, sim_opts, recon_opts)
        push!(times, t)
    end
end
median_time = sort(times)[2]
println("simulate!() median: $(round(median_time, digits=4))s")
```

## CRITICAL: Branch Safety

**ALL work MUST happen on the branch `speed/fused-projection`.** Do NOT commit to main.

**Before EVERY commit, verify you are on the correct branch:**
```bash
git branch --show-current  # MUST say "speed/fused-projection"
```

## CRITICAL: UInt16 Mask

The phantom mask MUST be `UInt16` (NOT `UInt8`). Supports up to 65,535 material regions.

## CRITICAL: AcceleratedKernels.jl Only

ALL GPU kernels MUST use `AK.foreachindex` (or other AK.jl primitives). No KernelAbstractions.jl directly. No CUDA.jl. No Metal.jl direct kernel calls.

## Your Workflow

### STEP 1: Read State

```
1. ralph_loops/speed-build-prd.json    — Stories created by discovery (READ FIRST)
2. ralph_loops/speed-spec.md           — The GPU-validated spec from discovery
3. ralph_loops/speed-build-progress.md — What's been done (your LOG)
```

### STEP 2: Pick the Next "ready" Story

Stories in `speed-build-prd.json` with `"status": "ready"` are GPU-validated and ready to implement. Do them in order of `priority` (0 first).

### STEP 3: Implement

Read the story's `requirements`, `files`, and `spec_ref`.

**Implementation rules:**
- Keep the OLD code path available for A/B comparison (don't delete it)
- Run tests after every significant change
- **Run GPU benchmarks after every significant change** — catch regressions early
- Commit frequently on the branch with descriptive messages
- Prefix commits: `SPEED-BUILD-XXX: description`

### STEP 4: Validate (GPU REQUIRED)

**Correctness validation (REQUIRED for every story):**
```julia
using Metal
import BasisSimulator as BS
# A/B comparison ON GPU — max abs diff < 1e-5
```

**Performance validation (REQUIRED for EVERY story — ON GPU):**
```julia
using Metal
import BasisSimulator as BS

# Time OLD path vs NEW path with Metal.@sync
# Report GPU speedup — NOT CPU speedup
# Must meet the story's gpu_speedup_threshold
```

**If GPU speedup is < story threshold:**
- Investigate with Metal.@profile or @code_llvm
- Fix the issue or try a different approach
- Do NOT mark done with CPU-only speedup numbers

### STEP 5: Update State

After completing a story:
1. Update `speed-build-prd.json`: status → `"done"`, add completedNote **with GPU speedup numbers**
2. Update `speed-build-progress.md`: log what was done, **GPU measurements**
3. Commit all changes on the branch
4. Move to next story

---

## Exit Conditions

- After each story: commit on branch, update PRD status **with GPU benchmark results**
- After all stories: output `SPEED_BUILD_COMPLETE` **with total GPU speedup measurement**
- If a story shows no GPU speedup after investigation: output `SPEED_BUILD_BLOCKED` with diagnosis
- **NEVER commit to main. NEVER.**
- **NEVER mark a story "done" without GPU benchmark numbers.**
