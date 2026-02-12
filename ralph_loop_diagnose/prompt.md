# BasisSimulator — Ghost Artifact Diagnosis

You are an autonomous diagnostic agent investigating ghost-like aliasing artifacts in CT reconstruction. Your goal is to DIAGNOSE the root cause through systematic testing. **Do NOT fix any code — only diagnose.**

## STEP 0: Check Environment

```bash
ls /Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl/src/BasisSimulator.jl
ls /Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl/verification/notebooks/05_xcat_full.jl
```

---

## STEP 1: Read State (Do This First — EVERY iteration)

```
1. ralph_loop_diagnose/prd.json       — Stories with status and priorities (READ THE FULL context SECTION)
2. ralph_loop_diagnose/progress.md    — What was done previously
3. ralph_loop_diagnose/guardrails.md  — Rules you MUST follow
```

**CRITICAL**: The `prd.json` `context` section contains detailed analysis of the problem, the current notebook settings, ranked potential causes, and a critical code observation about pixel_size vs pixel_row_size. READ IT THOROUGHLY before starting any work.

---

## STEP 2: Pick a Story

Select the highest-priority **unblocked** story:
- `"status": "open"` (or `"in_progress"` — finish it first)
- All `"blockers"` stories have `"status": "done"`

Priority order: lower number first. Set chosen story to `"in_progress"` in prd.json immediately.

---

## STEP 3: Execute

### CHECKPOINT RULE (CRITICAL)

**Commit early and often.** Every agent iteration may be killed at any time.

```
CHECKPOINT after:
  - Every test run completed
  - Every significant finding
  - Every 10 minutes of work
  - BEFORE running long commands

A checkpoint = update progress.md + commit:
  1. Append findings to ralph_loop_diagnose/progress.md
  2. Update ralph_loop_diagnose/prd.json if status changed
  3. git add ralph_loop_diagnose/
  4. git commit -m "[DIAG-STORY-ID]: [what was done]"
```

### Running the Diagnostic Script

After SETUP-DIAG creates `diag.jl`, run tests like:

```bash
cd /Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl
julia --project=verification ralph_loop_diagnose/diag.jl
```

This will take 3-10 minutes per run (GPU CT simulation + JIT compilation).

### Creating the Diagnostic Script (SETUP-DIAG)

The script should:

1. **Load XCAT phantom** — reuse the loading code from NB05 (lines 105-196):
   ```julia
   # Load binary, downsample by factor 2, load materials from XLSX
   ```

2. **Create GPU phantom** — `BS.Phantom(Metal.MtlArray(phantom_labeled), materials_dict, voxel_size_cm)`

3. **Create scanner, protocol, options** — mirror NB05 settings but make physics toggleable:
   ```julia
   sim_opts = BS.SimOptions(
       fidelity = :high,
       use_fill_factor = USE_FILL_FACTOR,
       use_flat_filter = USE_FLAT_FILTER,
       use_bowtie_filter = USE_BOWTIE_FILTER,
       # ... all toggleable
       use_noise = false,  # ALWAYS off for diagnosis
       seed = 42
   )
   ```

4. **Simulate + reconstruct** — use the workspace pattern:
   ```julia
   ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_gpu)
   BS.simulate!(ws, phantom_gpu, scanner, protocol, sim_opts, recon_opts)
   ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=BS.HannFilter())
   recon = Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size))
   ```

5. **Save outputs** — reconstruction and sinogram PNGs:
   ```julia
   # Reconstruction
   f = CM.Figure(size=(600, 600))
   ax = CM.Axis(f[1,1], title="$TEST_NAME", aspect=CM.DataAspect())
   CM.heatmap!(ax, recon_hu[:,:,32], colormap=:grays, colorrange=(-300, 400))
   CM.save("ralph_loop_diagnose/outputs/diag_$(TEST_NAME).png", f)

   # Sinogram (middle row, all angles)
   sino_cpu = Array(ws.sino_noisy_out)
   f2 = CM.Figure(size=(800, 400))
   ax2 = CM.Axis(f2[1,1], title="Sinogram — $TEST_NAME")
   CM.heatmap!(ax2, sino_cpu[:, size(sino_cpu,2)÷2, :])
   CM.save("ralph_loop_diagnose/outputs/sino_$(TEST_NAME).png", f2)
   ```

### Water Calibration

For HU conversion, run a quick water phantom calibration first (same as NB05 lines 744-835):
```julia
# Quick water calibration
μ_water = let
    water_mask = zeros(UInt8, 400, 400, 16)
    # ... (20cm water cylinder)
    # simulate + reconstruct + extract center ROI mean
end
```

Or if that's too slow, just use the raw μ values without HU conversion — the ghost artifact is visible in raw attenuation too.

### For CODE-AUDIT Story

This is a pure code-reading task. Read these files carefully and compare:

```
src/projection/siddon.jl                    — lines 440-530 (pixel position computation)
src/reconstruction/core/backprojection.jl   — lines 316-393 (voxel → pixel mapping)
src/reconstruction/core/filtering.jl        — cosine weighting computation
src/geometry/scanner.jl                     — CTGeometry constructor (pixel_size computation)
src/api/driver.jl                           — simulate!() FOV setup
src/api/workspace.jl                        — workspace creation, geometry propagation
```

Document each comparison with exact line numbers.

---

## STEP 4: Document

After every checkpoint, append to `ralph_loop_diagnose/progress.md`:

```markdown
### [STORY-ID] — Iteration N (YYYY-MM-DD)
- **Test:** [What was tested]
- **Result:** [ARTIFACTS PRESENT / ARTIFACTS ABSENT]
- **Observation:** [What was observed]
- **Image:** [Filename of output]
- **Next:** [What to do next]
```

---

## STEP 5: Commit

Working directory: `/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl`

```bash
git add ralph_loop_diagnose/
git commit -m "[DIAG-STORY-ID]: [description]"
```

---

## STEP 6: Exit

- Story completed → Exit normally
- All stories done → Output: `RALPH_COMPLETE`
- Blocked → Output: `RALPH_BLOCKED: [reason]`

---

## Key Directories

- **BasisSimulator.jl**: `/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl/`
- **Diagnostic outputs**: `ralph_loop_diagnose/outputs/`
- **Source code**: `src/` (projection, reconstruction, detector, geometry, api)
- **Notebook reference**: `verification/notebooks/05_xcat_full.jl`

## XCAT Data Location

The XCAT phantom binary is at:
```
verification/data/xcat/vmale_50_1600x1400x500_8bit_little_endian_act_1.bin
```
(symlink to BasisSimulatorVerification)

Material spreadsheet:
```
verification/data/xcat/Material_Spreadsheets/vmale_50_materials_heart_high_contrast.xlsx
```

## Important: What NOT to Do

1. **Do NOT modify any source code** in `src/` — diagnosis only
2. **Do NOT modify the notebook** in `verification/notebooks/`
3. **Do NOT run the full Pluto notebook** — too slow, use the standalone script
4. **Do NOT try to fix the bug** — only identify it
5. **Do NOT skip documenting results** — every test must be recorded in progress.md
