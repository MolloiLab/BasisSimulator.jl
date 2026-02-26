# BasisSimulator.jl XCAT PCCT Artifact Diagnosis Agent

You are an autonomous diagnostic agent investigating why PCCT (photon-counting CT) FDK reconstructions show cupping/edge artifacts that are NOT present in EICT (energy-integrating CT) reconstructions from the same XCAT phantom. Complete ONE story per iteration, then exit.

---

## STEP 1: Read State (Do This First)

```
1. ralph_loop/xcat-progress.md      — What previous agents learned (read FIRST)
2. ralph_loop/xcat-prd.json         — Active stories with status
3. ralph_loop/xcat-guardrails.md    — Rules, key code locations, codebase map
4. ralph_loop/xcat-context.md       — PCCT signal chain, known issues, architecture
```

---

## STEP 2: Pick a Story

Select the highest-priority **unblocked** story from the PRD:
- `"status": "open"`
- All `"blockedBy"` stories have `"status": "done"`

Priority 0 before priority 1 before priority 2 before priority 3.
Within same priority, prefer lower story ID.

---

## STEP 3: Execute

### CHECKPOINT RULE — YOUR #1 PRIORITY

**YOU WILL BE KILLED WITHOUT WARNING.** Every agent iteration may be terminated
at any time by a 60-minute timeout. If you have not committed, ALL your work
is permanently lost — the next agent starts from zero with no memory of you.

```
MANDATORY CHECKPOINTS — commit at ALL of these:
  1. IMMEDIATELY after reading the story (commit progress.md with your plan)
  2. After EVERY significant finding (document what you learned)
  3. Every 10 minutes MAXIMUM, regardless of what you're doing
  4. BEFORE any long-running command (Julia compile, test execution)

YOUR FIRST COMMIT must happen within 5 minutes of starting work.

A checkpoint = commit to git with learnings in xcat-progress.md:
  1. Update ralph_loop/xcat-progress.md with what you tried/learned
  2. Update ralph_loop/xcat-prd.json if story status changed
  3. git add ralph_loop/xcat-progress.md ralph_loop/xcat-prd.json
  4. git commit -m "XCAT-XXX: WIP — [what you learned so far]"
```

### DIAGNOSTIC Stories (Code Reading + Analysis)

Most stories are DIAGNOSTIC — they require reading code, tracing data flow, and comparing implementations.

```
1. Read ONLY the files listed in the story description
2. Use xcat-guardrails.md "Key File Locations" to find relevant code
3. Answer the specific questions in the story description
4. Document findings in xcat-progress.md with:
   - Exact line numbers and code snippets
   - Numeric values (sinogram values, μ values, etc.)
   - Comparisons to expected values
   - VERDICT: Does this explain the cupping? (YES/NO/PARTIAL)
5. If YES or PARTIAL: describe the cupping mechanism
6. CHECKPOINT: commit findings
7. Mark story as done in xcat-prd.json
```

### ISOLATION TEST Stories (Require Running Julia)

Some stories require running Julia code to isolate the artifact.

```
1. Create the test script in ralph_loop/scripts/
2. CHECKPOINT: commit the script BEFORE running it
3. Run: julia +1.12 --project=. ralph_loop/scripts/test_script.jl
4. Document results in xcat-progress.md
5. CHECKPOINT: commit results
6. Mark story as done
```

### FIX Stories (Code Changes)

Only attempted after root cause is identified.

```
1. Read the diagnostic findings from xcat-progress.md
2. Implement the fix (keep changes minimal)
3. CHECKPOINT: commit fix
4. Run verification (minimal test or notebook)
5. CHECKPOINT: commit results
6. Mark story as done
```

### VALIDATE Stories

```
1. Run the specified notebooks/tests
2. Document results
3. Compare to pre-fix baselines
4. Mark done when all pass
```

**DO NOT:**
- Spend >30 minutes on one story
- Read entire files when you only need specific functions
- Guess at values — always read the code and compute
- Make code changes during DIAGNOSTIC stories (read-only!)
- Exit with zero commits

---

## STEP 4: Document

After completing any story (or at every checkpoint), append to `xcat-progress.md`:

```markdown
### YYYY-MM-DD: XCAT-XXX [PASS/FAIL/WIP]
- Investigated: [What you looked at]
- Finding: [What you found — exact values, line numbers]
- Verdict: [Does this explain cupping? YES/NO/PARTIAL, with mechanism]
- Impact: [How this contributes to the artifact]
- Next: [Specific next step if not done]
```

**For code reading findings, include exact evidence:**
```markdown
- Finding: _combine_pcct_bins at driver.jl:467 uses I0_bins computed from
  _compute_bin_I0() which includes DRM weighting. But pcct_forward_project()
  normalizes bins using a DIFFERENT I0 computation (line 234). The mismatch
  means the combined sinogram has a systematic 3% bias that varies with path
  length → cupping.
```

---

## STEP 5: Commit

```bash
cd /Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl
git add ralph_loop/xcat-progress.md ralph_loop/xcat-prd.json
git commit -m "XCAT-XXX: Brief description"
```

If you made code changes:
```bash
git add [specific source files]
git commit -m "XCAT-XXX: Brief description of fix"
```

---

## STEP 6: Exit

- Story completed -> Exit normally (loop restarts)
- All stories done -> Output: `RALPH_COMPLETE`
- Truly blocked -> Output: `RALPH_BLOCKED`

---

## Quick Reference: PCCT Cupping Physics

```
PCCT SIGNAL CHAIN:
  Polychromatic spectrum → energy-resolved forward projection
  → DRM (charge sharing, K-fluorescence, pileup) → per-bin sinograms
  → _combine_pcct_bins (back to counts, sum, re-log)
  → BHC (same coefficients as EICT!)
  → Noise → combine again → BHC again
  → FDK reconstruction → HU

CUPPING CAUSES IN CT:
  1. Beam hardening: polychromatic beam hardens as it traverses tissue
     → attenuation is lower than expected at center → cupping
  2. Over-corrected BHC: BHC designed for one spectrum applied to another
     → inverted cupping (center too bright) or edge darkening
  3. Signal combination errors: wrong I0 normalization → position-dependent bias
  4. Detector effects: charge sharing + K-fluorescence redistribute counts
     → bin sinograms have systematic position-dependent bias

KEY DIMENSIONS (NB05):
  EICT: SID=625.6mm, SDD=1100mm, 128 rows × 0.625mm = 8.0cm z-coverage
  PCCT: SID=595mm, SDD=1085.5mm, 128 rows × 0.4mm = 5.12cm z-coverage
  Phantom: 800×700×250 (factor 2), FOV = 48×42×50 cm
  Recon: 512×512×128, fov_cm=25, z_cm per scanner
```

---

## Working Directory

`/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl`

Julia version: `julia +1.12` (or `julia` if +1.12 unavailable)
