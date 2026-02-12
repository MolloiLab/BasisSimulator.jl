# BasisSimulator — Verification Suite Migration

You are an autonomous agent migrating the BasisSimulatorVerification project into BasisSimulator.jl/verification/ so the verification suite ships with the package.

## STEP 0: Check Environment

```bash
ls /Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl/src/BasisSimulator.jl
ls /Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulatorVerification/notebooks/
```

---

## STEP 1: Read State (Do This First — EVERY iteration)

```
1. ralph_loop/prd.json       — Stories with status and priorities
2. ralph_loop/progress.md    — What was done previously
3. ralph_loop/guardrails.md  — Rules you MUST follow
```

For notebook migration stories, also read:
```
4. The SOURCE notebook: BasisSimulatorVerification/notebooks/[filename]
5. The TARGET location: BasisSimulator.jl/verification/notebooks/ (check if file exists already)
```

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
  - Every file copied/created
  - Every path adjustment verified
  - Every 10 minutes of work
  - BEFORE running long commands

A checkpoint = update progress.md + commit:
  1. Append findings to ralph_loop/progress.md
  2. Update ralph_loop/prd.json if status changed
  3. git add ralph_loop/ verification/
  4. git commit -m "[STORY-ID]: [what was done]"
```

### CREATE-SCAFFOLD Story

```
1. Create directory structure:
   verification/
   verification/notebooks/
   verification/data/gammex_basic/
   verification/data/xcat/
   verification/data/xcat/Material_Spreadsheets/
   verification/figures/

2. Create verification/Project.toml:
   - Copy from BasisSimulatorVerification/Project.toml
   - Change BasisSimulator source to: path = "../"
   - Keep all other deps and compat entries

3. Create verification/CondaPkg.toml:
   - Copy from BasisSimulatorVerification/CondaPkg.toml

4. Create verification/.gitignore:
   .CondaPkg/
   Manifest.toml
   .DS_Store
   figures/
   catsim_work/
   *.air
   *.offset
   *.scan

5. Update BasisSimulator.jl/.gitignore to add verification/ exclusions

6. CHECKPOINT
```

### MIGRATE-PHANTOM-* Stories

```
1. Copy files from source to target
2. Verify file sizes match
3. For XCAT: check git lfs, set up tracking or download script
4. CHECKPOINT
```

### MIGRATE-NB* Stories

```
1. Copy notebook file from BasisSimulatorVerification/notebooks/ to verification/notebooks/
2. Read the notebook and search for ALL path references:
   - "phantom_export/"  → "data/gammex_basic/"
   - "phantom_export_xcat/" → "data/xcat/"
   - "phantom_export_multimaterial/" → remove or update
   - Any hardcoded absolute paths → make relative
3. Update paths using @__DIR__-relative construction
4. Verify no old BasisSimulatorVerification references remain
5. CHECKPOINT
```

### SETUP-XCAT-SYMLINK Story

```
1. Check if XCAT binary source exists:
   ls -la /Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulatorVerification/notebooks/phantom_export_xcat/vmale_50_1600x1400x500_8bit_little_endian_act_1.bin

2. Create symlink:
   ln -sf /Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulatorVerification/notebooks/phantom_export_xcat/vmale_50_1600x1400x500_8bit_little_endian_act_1.bin \
     /Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl/verification/data/xcat/vmale_50_1600x1400x500_8bit_little_endian_act_1.bin

3. Verify: ls -la verification/data/xcat/vmale_50*.bin
4. CHECKPOINT
```

### ADD-NB05-SAVES Story

```
1. Read verification/notebooks/05_xcat_full.jl
2. Search for all figure creation patterns (CM.Figure, fig = ...)
3. Add CM.save(joinpath(FIGURES_DIR, "nb05_XXX.png"), fig) after each figure
4. Preserve ALL cell UUIDs — edit content only, never touch cell ID lines
5. Expected figure names:
   - nb05_scanner_comparison.png
   - nb05_fdk_vs_hir.png
   - nb05_detail_comparison.png
   - nb05_vmi_comparison.png
   - nb05_noise_stats.png
   - nb05_noise_reduction.png
6. CHECKPOINT
```

### RUN-NB* Stories (Running Notebooks Headlessly)

**How to run a Pluto notebook headlessly:**

```bash
cd /Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl
julia --project=verification -e '
  using Pluto
  s = Pluto.ServerSession()
  nb = Pluto.SessionActions.open(s, "verification/notebooks/NOTEBOOK_FILENAME.jl"; run_async=false)
  # Check for errors
  errored = [c for (id, c) in nb.cells_dict if c.errored]
  if !isempty(errored)
      for c in errored
          println("ERRORED: ", c.code[1:min(100, length(c.code))])
          println("  ", c.output.body)
      end
      error("$(length(errored)) cells errored")
  end
  Pluto.SessionActions.shutdown(s, nb)
  println("NOTEBOOK PASSED")
'
```

Replace `NOTEBOOK_FILENAME.jl` with the actual filename:
- `01_single_kvp_verification.jl`
- `02_multi_dose_and_iterative_reconstruction.jl`
- `03_dual_kvp_vmi_verification.jl`
- `04_pcct_demonstration.jl`
- `05_xcat_full.jl`

**After running, verify figures:**

```bash
ls -la verification/figures/nbNN_*.png
# Check all files exist and are >1KB
find verification/figures -name 'nbNN_*.png' -size -1k
# (should return nothing — all files should be >=1KB)
```

**If notebook errors:**
1. Read the error output carefully
2. Open the notebook file and find the erroring cell
3. Fix the issue (path, import, API change, etc.)
4. Re-run the notebook
5. Do NOT mark the story done until 0 errored cells AND all figures verified

**For NB01 (CatSim):**
Before running, ensure Python deps are set up:
```bash
cd /Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl
julia --project=verification -e 'using CondaPkg; CondaPkg.resolve()'
```

### VERIFY-ALL-FIGURES Story

```
1. Count all figure PNGs:
   ls verification/figures/nb*.png | wc -l
   # Expected: 43 total

2. Check by notebook:
   ls verification/figures/nb01_*.png | wc -l  # Expected: 6
   ls verification/figures/nb02_*.png | wc -l  # Expected: 7
   ls verification/figures/nb03_*.png | wc -l  # Expected: 8
   ls verification/figures/nb04_*.png | wc -l  # Expected: 16
   ls verification/figures/nb05_*.png | wc -l  # Expected: 6

3. Check none are too small:
   find verification/figures -name 'nb*.png' -size -1k
   # Must return empty

4. If all pass → mark done → output RALPH_COMPLETE
5. If any missing → identify which RUN-NB* story needs re-running
```

---

## STEP 4: Document

After every checkpoint, append to `ralph_loop/progress.md`:

```markdown
### [STORY-ID] — Iteration N (YYYY-MM-DD)
- **Done:** [What was accomplished]
- **Files:** [Files created/modified]
- **Next:** [What to do next if not done]
```

---

## STEP 5: Commit

Working directory: `/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl`

```bash
git add ralph_loop/ verification/
git commit -m "[STORY-ID]: [description]"
```

---

## STEP 6: Exit

- Story completed → Exit normally
- All stories done → Output: `RALPH_COMPLETE`
- Blocked → Output: `RALPH_BLOCKED: [reason]`

---

## Key Directories

- **Source**: `/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulatorVerification/`
- **Target**: `/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl/verification/`
- **BasisSimulator.jl**: `/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl/`

## Source Notebook Locations

```
BasisSimulatorVerification/notebooks/01_single_kvp_verification.jl
BasisSimulatorVerification/notebooks/02_multi_dose_and_iterative_reconstruction.jl
BasisSimulatorVerification/notebooks/03_dual_kvp_vmi_verification.jl
BasisSimulatorVerification/notebooks/04_pcct_demonstration.jl
BasisSimulatorVerification/notebooks/05_xcat_full.jl
```

## Phantom Data Locations

```
BasisSimulatorVerification/notebooks/phantom_export/          → verification/data/gammex_basic/
BasisSimulatorVerification/notebooks/phantom_export_xcat/     → verification/data/xcat/
```
