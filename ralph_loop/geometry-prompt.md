# BasisSimulator.jl Z/XY Geometry Scaling Audit Agent

You are an autonomous audit agent performing an exhaustive review of z-direction and xy-direction geometry scaling in BasisSimulator.jl. Complete ONE story per iteration, then exit.

---

## STEP 1: Read State (Do This First)

```
1. ralph_loop/geometry-progress.md      — What previous agents learned (read FIRST)
2. ralph_loop/geometry-prd.json         — Active stories with status
3. ralph_loop/geometry-guardrails.md    — Rules, key conventions, codebase map
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
is permanently lost.

```
MANDATORY CHECKPOINTS — commit at ALL of these:
  1. IMMEDIATELY after reading the story (commit progress.md with your plan)
  2. After EVERY file audited (document what you found)
  3. Every 10 minutes MAXIMUM
  4. BEFORE any long-running command

YOUR FIRST COMMIT must happen within 5 minutes of starting work.

A checkpoint = commit to git:
  1. Update ralph_loop/geometry-progress.md with findings
  2. Update ralph_loop/geometry-prd.json if story status changed
  3. git add ralph_loop/geometry-progress.md ralph_loop/geometry-prd.json
  4. git commit -m "GEO-XXX: WIP — [what you found]"
```

### AUDIT Stories (Most Common)

```
1. Run the grep command specified in testCommand
2. For EACH occurrence found:
   a. Read the surrounding code (5-10 lines of context)
   b. Determine if it's used for z-direction (rows/v) or xy-direction (cols/u)
   c. Verdict: CORRECT or WRONG (with explanation)
3. Document findings in geometry-progress.md as a TABLE:
   | File:Line | Variable | Direction | Verdict | Notes |
4. CHECKPOINT after each file
5. Mark story done when all files checked
```

### FIX Stories

```
1. Read audit findings from geometry-progress.md
2. For each WRONG finding, make the minimal fix
3. CHECKPOINT after each fix
4. Mark done when all fixes applied
```

**DO NOT:**
- Spend >30 minutes on one story
- Skip any grep results — check EVERY occurrence
- Assume correctness without reading the code
- Make code changes during AUDIT stories
- Exit with zero commits

---

## STEP 4: Document

Append to `geometry-progress.md`:

```markdown
### YYYY-MM-DD: GEO-XXX [PASS/FAIL/WIP]
- Files checked: [list]
- Issues found: [count]
- Details:

| File:Line | Variable | Direction | Verdict | Notes |
|-----------|----------|-----------|---------|-------|
| siddon.jl:518 | pixel_size | xy (col) | CORRECT | u_offset calculation |
| scatter.jl:45 | pixel_size | z (row) | **WRONG** | should be pixel_row_size |
```

---

## STEP 5: Commit & Exit

```bash
cd /Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl
git add ralph_loop/geometry-progress.md ralph_loop/geometry-prd.json
git commit -m "GEO-XXX: Brief description"
```

If code changes: `git add [specific files]`

- Story done -> Exit normally
- All stories done -> Output: `RALPH_COMPLETE`
- Blocked -> Output: `RALPH_BLOCKED`

---

## Quick Reference: Geometry Conventions

```
COORDINATE SYSTEM:
  x,y = in-plane (transaxial) — columns/u-direction
  z   = longitudinal (superior-inferior) — rows/v-direction

SCANNER STRUCT (mm at isocenter):
  detector_col_size → xy-direction pixel pitch
  detector_row_size → z-direction pixel pitch
  detector_cols     → number of columns (xy)
  detector_rows     → number of rows (z)

CTGEOMETRY (cm):
  pixel_size     = detector_col_size / 10  → xy at isocenter
  pixel_row_size = detector_row_size / 10  → z at isocenter
  fov = (fov_x, fov_y, fov_z)

SIDDON (uses geom fields):
  u_offset = col × pixel_size × magnification      → xy
  v_offset = row × pixel_row_size × magnification   → z
  volume_fov → phantom physical extent (overrides geom.fov for bounds)

RECONSTRUCTION:
  Forward proj (simulate): volume_fov = phantom.fov
  Backward proj (recon):   geom.fov = reconstruction FOV
  Iterative (SIRT/CGLS):   geom.fov = reconstruction FOV (NO volume_fov!)

EICT (GE Revolution):
  detector_row_size = 0.625 mm, detector_col_size = 0.6 mm (non-square!)
  z_cm = n_slices × 0.625 / 10

PCCT (Siemens NAEOTOM):
  detector_row_size = 0.4 mm, detector_col_size = 0.4 mm (square)
  z_cm = n_slices × 0.4 / 10
```

---

## Working Directory

`/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl`
