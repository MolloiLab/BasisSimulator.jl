# Guardrails — Verification Suite Migration

## Absolute Rules

1. **Checkpoint early and often** — Commit progress.md after EVERY meaningful step. If you get killed by timeout, your work must survive.

2. **One notebook at a time** — Copy, adjust paths, verify, checkpoint. Don't try to migrate all notebooks in one pass.

3. **Preserve notebook cell IDs** — Pluto notebooks have UUID cell identifiers. Do NOT regenerate or modify cell IDs. Copy files byte-for-byte, then only edit path strings.

4. **Track status properly** — Set story to `"in_progress"` when starting, `"done"` only after acceptance criteria are met.

5. **Document everything in progress.md** — Each iteration appends: what was attempted, what was done, what's next.

6. **Use git properly**:
   - Commit after EVERY migration step with `[STORY-ID]` tag
   - Don't commit broken notebooks
   - Working directory: `/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl`

---

## Path Rules

7. **BasisSimulator dependency**: verification/Project.toml must use `path = "../"` to reference the parent package. Never use an absolute path.

8. **Notebook Pkg.activate**: `Pkg.activate(dirname(@__DIR__))` resolves from `verification/notebooks/` → `verification/`. This is CORRECT. Do NOT change it.

9. **Phantom data paths**: Use `@__DIR__` relative paths, NOT absolute paths:
   - From notebooks/: `joinpath(@__DIR__, "..", "data", "gammex_basic", "filename")`
   - Or: `joinpath(dirname(@__DIR__), "data", "gammex_basic", "filename")`

10. **Figure save paths**: `joinpath(dirname(@__DIR__), "figures")` — already correct, do NOT change.

11. **CatSim work directory**: Should be `joinpath(@__DIR__, "catsim_work")` or similar relative path from notebooks/. Gitignored.

---

## Data Rules

12. **Small files (< 10 MB) → commit directly**: gammex_basic phantom data, XLSX spreadsheets.

13. **Large files (> 50 MB) → Git LFS or download script**: XCAT binary (1 GB). Check if Git LFS is available with `git lfs install`. If not available, create a .gitignore entry + download instructions.

14. **Generated files → gitignore**: figures/, catsim_work/, .CondaPkg/, Manifest.toml.

15. **Never commit secrets or credentials**: No API keys, no personal paths in committed files.

---

## Notebook Editing Rules

16. **Search-and-replace paths carefully**: Some phantom path strings may appear in multiple places (e.g., `phantom_export/` referenced in code AND markdown cells). Search ALL occurrences.

17. **Test path resolution mentally**: When changing a path, mentally trace what `@__DIR__` resolves to from the notebook's location, then verify the path exists.

18. **Preserve the import-as pattern**: All notebooks use `import BasisSimulator as BS`, `import CairoMakie as CM`, etc. Do NOT change import patterns.

19. **Preserve the let-block + GC pattern**: Notebooks use `let ... end` blocks with `GC.gc(true)` for memory management. Do NOT remove these.

---

## Git LFS Rules (if used)

20. **Track binary patterns, not specific files**: Use `git lfs track 'verification/data/xcat/*.bin'` in .gitattributes.

21. **Commit .gitattributes BEFORE adding LFS files**: The tracking must be in place before git add.

22. **Verify LFS is working**: After commit, `git lfs ls-files` should show the tracked files.

---

## Notebook Execution Rules

23. **Run ONE notebook per iteration** — Each notebook takes 10-30 minutes with JIT compilation and GPU CT simulations. Don't try to run multiple notebooks in one iteration.

24. **If a notebook errors, debug and fix** — Don't just mark the story done. Read the error output, find the failing cell, fix the issue, and re-run. A story is only done when the notebook runs with 0 errored cells.

25. **Figure verification is mandatory** — After running a notebook, check that ALL expected PNG files exist in `verification/figures/` and are >1KB. A 0-byte or missing figure means the notebook didn't fully succeed.

26. **For NB01 (CatSim)** — Ensure CondaPkg.toml is set up and gecatsim installs via `CondaPkg.resolve()`. The Python environment must be ready before running the notebook. If gecatsim fails to install, check the pip URL in CondaPkg.toml.

27. **For NB05 (XCAT)** — The XCAT binary symlink must exist before running. Verify with `ls -la verification/data/xcat/vmale_50*.bin`. If the symlink is broken or missing, run SETUP-XCAT-SYMLINK first.

28. **Order: NB02 → NB03 → NB04 → NB01 → NB05** — Self-contained notebooks first (no external deps), then CatSim (Python), then XCAT (large data). This maximizes early progress.

29. **Pluto headless execution** — Use the `Pluto.ServerSession()` + `Pluto.SessionActions.open()` pattern from prompt.md. Do NOT try to use `Pluto.run()` (that opens a browser).

30. **Don't modify notebook cell UUIDs** — When fixing errors in notebooks, only edit cell content, never touch the cell ID comment lines (`# ╔═╡ UUID`).

---

## Exit Conditions

- **RALPH_COMPLETE**: All 5 notebooks run to completion, all expected figures verified (43 PNGs, all >1KB)
- **RALPH_BLOCKED**: Cannot proceed (explain why — e.g., XCAT binary missing, gecatsim won't install)

---

## Mistake Log

1. **Don't modify cell UUIDs** — Pluto will break if cell IDs change unexpectedly
2. **@__DIR__ is notebook location** — NOT the project root. From notebooks/, @__DIR__ = verification/notebooks/
3. **dirname(@__DIR__) = verification/** — This is where Project.toml lives
4. **CatSim auto-downloads** — gecatsim installs via CondaPkg on first Pkg.instantiate(). Don't manually install it.
5. **phantom_export/ vs phantom_export_xcat/** — These are DIFFERENT datasets. Don't confuse them.
6. **XCAT notebook is 05 (renamed from 06_xcat_full.jl)** — The file in BasisSimulatorVerification is `05_xcat_full.jl` (was renumbered).
