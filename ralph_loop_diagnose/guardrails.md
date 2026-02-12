# Guardrails — Ghost Artifact Diagnosis

## Absolute Rules

1. **DIAGNOSIS ONLY** — Do NOT modify any source code in `src/`. Do NOT modify notebooks in `verification/`. You are here to FIND the bug, not fix it.

2. **Checkpoint early and often** — Commit progress.md after EVERY test run. If you get killed by timeout, your work must survive.

3. **One test per iteration** — Each GPU simulation takes 3-10 minutes. Run ONE test, document the result, checkpoint. Don't try to run 5 tests in one iteration.

4. **Track status properly** — Set story to `"in_progress"` when starting, `"done"` only after acceptance criteria are met.

5. **Document EVERY finding in progress.md** — Each test result must be recorded with: test name, what was tested, ARTIFACTS PRESENT or ABSENT, observations.

6. **Use git properly**:
   - Commit after EVERY test with `[DIAG-STORY-ID]` tag
   - Working directory: `/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl`
   - Only add ralph_loop_diagnose/ files: `git add ralph_loop_diagnose/`

---

## Diagnostic Script Rules

7. **Keep noise OFF** — `use_noise = false` in ALL tests. Noise masks the artifact.

8. **Save BOTH reconstruction and sinogram** — For every test, save the reconstruction image AND a sinogram view. Both are needed for diagnosis.

9. **Use consistent window/level** — All reconstructions displayed at (-300, 400) HU or equivalent. All sinograms with the same colorrange per series.

10. **Use the SAME slice for all comparisons** — slice_idx = 32 (or whatever shows the ribs).

11. **Name outputs systematically** — `diag_<test_name>.png` and `sino_<test_name>.png` in `ralph_loop_diagnose/outputs/`.

---

## Code Reading Rules

12. **Read the full context in prd.json** — The `context` section has critical information about pixel_size vs pixel_row_size, potential causes ranked by likelihood, and the exact notebook settings. This saves you from re-discovering known information.

13. **When auditing code, note EXACT line numbers** — "siddon.jl uses pixel_size on line 445" is useful. "siddon.jl uses pixel_size somewhere" is not.

14. **Compare FP and BP side by side** — The forward projection (siddon.jl) and backprojection (backprojection.jl) MUST be geometrically consistent. Any difference in how they compute detector positions is a potential root cause.

---

## Scientific Method

15. **Change ONE variable per test** — Don't change detector size AND disable physics in the same run. Isolate variables.

16. **Record negative results** — "Disabling X did NOT remove artifacts" is valuable information. Document it.

17. **If a test is inconclusive, say so** — Don't claim certainty you don't have. "Artifacts may be slightly reduced but still present" is honest.

18. **Trust the images over theory** — If the image shows artifacts when theory says it shouldn't, trust the image.

---

## Exit Conditions

- **RALPH_COMPLETE**: DIAGNOSIS.md written with root cause identified and evidence documented
- **RALPH_BLOCKED**: Cannot proceed (explain why — e.g., Metal GPU unavailable, XCAT binary missing)

---

## Mistake Log

1. **Phantom is CLEAN** — The XCAT labeled phantom has NO aliasing. The artifact is introduced by the simulation pipeline. Do not blame the phantom.
2. **More dexels = worse** — This rules out spatial undersampling as the cause. Think about what gets WORSE with finer sampling.
3. **Don't skip the water calibration** — Without proper μ_water, HU values will be off, making it harder to see subtle artifacts. At minimum, use a hardcoded μ_water ≈ 0.20 cm⁻¹ for 120 kVp.
4. **GPU memory** — MtlArray allocations accumulate. Use `let` blocks and `GC.gc(true)` to free GPU memory between tests.
5. **JIT time** — First Julia run is slow (~2 min just for compilation). Subsequent runs in the same session are faster. Consider running multiple tests in one script invocation.
