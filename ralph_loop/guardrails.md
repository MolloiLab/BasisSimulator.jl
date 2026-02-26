# Guardrails — Two-Material BHC Implementation

## Absolute Rules

1. **Checkpoint early and often** — Commit progress.md after EVERY meaningful step. If you get killed by timeout, your work must survive.

2. **Don't break existing code** — The water-only BHC functions (`calibrate_bhc`, `apply_bhc!`, `apply_bhc`, etc.) must remain untouched. Add new code, don't modify existing.

3. **Track status properly** — Set story to `"in_progress"` when starting, `"done"` only after acceptance criteria are met.

4. **Document everything in progress.md** — Each iteration appends: what was attempted, what was done, what's next.

5. **Use git properly**:
   - Commit after EVERY meaningful step with `[STORY-ID]` tag
   - Don't commit broken code
   - Working directory: `/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl`

---

## Code Rules

6. **Add code to end of file** — New BHC functions go at the END of `src/correction/beam_hardening_correction.jl`, after all existing code.

7. **Exports go near the top** — Add new exports alongside existing export statements.

8. **Bone material is `corticalbone`** — XrayAttenuation.jl uses `XA.Materials.corticalbone` (NO underscore). This WILL cause an error if you use `cortical_bone`.

9. **Float64 for spectral computations** — The inner loop computing `Σ wₑ exp(-μ_w(E)Lw - μ_b(E)Lb)` must use Float64, not Float32. Float32 loses precision in the exponential sum.

10. **CPU-only for the correction loop** — The per-ray correction loop uses `Threads.@threads`. Don't try to GPU-ify this. The forward projection and FDK already use GPU.

11. **Array() to move GPU→CPU** — After `fdk_reconstruct` and `siddon_forward_project`, call `Array()` to move results to CPU before indexing.

12. **volume_extent matters** — When calling `siddon_forward_project` for the bone image, pass `volume_extent` to match the phantom extent, NOT the reconstruction FOV.

13. **GPU closure rule** — If you write any `AK.foreachindex` closures that capture conditionally-assigned variables, wrap the closure in a `let` block. See CLAUDE.md for details.

---

## Test Script Rules

14. **Activate project first** — Script must start with `Pkg.activate(joinpath(@__DIR__, "..", ".."))` to use the BasisSimulator.jl project.

15. **mkpath before save** — Create the output directory before saving: `mkpath(joinpath(@__DIR__, "..", "results"))`.

16. **High-res output** — Use `save(path, fig, px_per_unit=2)` for publication-quality PNG.

17. **Use CairoMakie, not GLMakie** — CairoMakie works headlessly. GLMakie requires a display.

18. **Transpose for image display** — Use `hu[:, :, slice]'` (transposed) for correct image orientation in heatmap.

---

## Exit Conditions

- **RALPH_COMPLETE**: All 3 stories done, PNG verified
- **RALPH_BLOCKED**: Cannot proceed (explain why)

---

## Mistake Log

(Updated as mistakes are discovered)
