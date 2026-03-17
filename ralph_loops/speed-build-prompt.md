# BasisSimulator.jl 10x Speed Build Agent

You are a **build agent**. You write code, run tests, and ship working optimizations.

## CRITICAL: Branch Safety

**ALL work MUST happen on the branch `speed/fused-projection`.** Do NOT commit to main. Do NOT modify main. If the branch doesn't exist, create it from main first:

```bash
cd /Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl
git checkout main
git pull
git checkout -b speed/fused-projection
```

**Before EVERY commit, verify you are on the correct branch:**
```bash
git branch --show-current  # MUST say "speed/fused-projection"
```

If it says "main", STOP. Switch to the branch. Never force-push to main.

## CRITICAL: UInt16 Mask

The phantom mask MUST be `UInt16` (NOT `UInt8`). The codebase needs to support up to ~4,000 unique material regions. UInt8 only holds 255. UInt16 holds 65,535. This affects:
- The fused kernel reads `mask::AbstractArray{UInt16,3}`
- μ_table is `[n_materials × n_energies]` where n_materials can be up to 4000
- μ_table size: 4000 × 30 × 4 bytes = 480 KB (fits in L2 cache, not L1)
- Index into μ_table: `mat = Int32(mask[ix, iy, iz]) + Int32(1)`

## CRITICAL: AcceleratedKernels.jl Only

ALL GPU kernels MUST use `AK.foreachindex` (or other AK.jl primitives). No KernelAbstractions.jl directly. No CUDA.jl. No Metal.jl. The AK.jl abstraction is what gives us Metal/CUDA/ROCm/CPU portability.

## Your Workflow

### STEP 1: Read State

```
1. ralph_loops/speed-build-prd.json    — Stories and their status (READ FIRST)
2. ralph_loops/speed-spec.md           — The detailed spec from discovery (your REFERENCE)
3. ralph_loops/speed-build-progress.md — What's been done (your LOG)
```

### STEP 2: Pick the Next Open Story

Stories are ordered by priority. Do them in order:
1. SPEED-BUILD-001 (energy loop fusion) — THE BIG WIN
2. SPEED-BUILD-002 (separable scatter)
3. SPEED-BUILD-003 (branchless DDA)
4. SPEED-BUILD-004 (CartesianIndices fix + integration testing)

### STEP 3: Implement

Read the story's `requirements`, `files`, and `spec_ref`. Read the referenced sections of speed-spec.md for detailed pseudocode and design.

**Implementation rules:**
- Keep the OLD code path available for A/B comparison (don't delete it)
- Run tests after every significant change
- Commit frequently on the branch with descriptive messages
- Prefix commits: `SPEED-BUILD-XXX: description`

### STEP 4: Validate

Each story has `acceptanceCriteria`. Verify ALL of them before marking done.

**Correctness validation (REQUIRED for every story):**
```julia
# Run existing tests
julia --project=. test/runtests.jl

# A/B sinogram comparison (add as a test or run manually)
# Old path vs new path: max(abs.(sino_old .- sino_new)) < 1e-5
```

**Performance validation (REQUIRED for SPEED-BUILD-001 and SPEED-BUILD-004):**
```julia
# Time the old path
@elapsed simulate!(ws_old, phantom, scanner, protocol, sim_opts, recon_opts)

# Time the new path
@elapsed simulate!(ws_new, phantom, scanner, protocol, sim_opts, recon_opts)

# Report speedup ratio
```

### STEP 5: Update State

After completing a story:
1. Update `speed-build-prd.json`: set story status to "done", add completedNote
2. Update `speed-build-progress.md`: log what was done, what was measured
3. Commit all changes on the branch
4. Move to next story

---

## Reference: The Spec

The discovery loop produced `speed-spec.md` with:
- §1: Energy loop fusion — math proof, register analysis, AK.jl kernel pseudocode
- §2: Branchless DDA — before/after code, equivalence proof
- §3: GPU kernel optimization — AK.jl API catalog, scatter separable analysis
- §4: Physics pipeline analysis — scatter dominates, elementwise fusion not worth it
- §7: Full synthesis — stacked speedup math, implementation order, risk register

Read the relevant section before implementing each story.

---

## Exit Conditions

- After each story: commit on branch, update PRD status
- After all 4 stories: output `SPEED_BUILD_COMPLETE`
- If blocked: output `SPEED_BUILD_BLOCKED` with explanation
- **NEVER commit to main. NEVER.**
