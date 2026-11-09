#!/usr/bin/env python3
"""Format the Julia code of every Pluto notebook cell with Fatou, leaving Pluto's own text alone.

Fatou formats `.jl` files whole, but a Pluto notebook is also Pluto's: its header (the `@bind`
mock), the `# ╔═╡` cell markers, the package cells and the Cell order block are rewritten by Pluto
on every save, so formatting them would be undone and fail the check. `fatou.toml` excludes the
notebook folders from Fatou's own walk; this script formats each cell's code alone (through
`fatou format -`, with the repository's `fatou.toml`) and puts it back between the markers.

    python3 docs/fatou_notebooks.py           # check: print the cells that would change, exit 1 if any
    python3 docs/fatou_notebooks.py --write   # format them in place
    python3 docs/fatou_notebooks.py --write path/to/notebook.jl ...

Disabled cells (`#=╠═╡ … ╠═╡ =#`) and Pluto's package cells are left as they are. Lint the notebooks
with `fatou lint <files>` (named files are linted even though `fatou.toml` excludes the folders).
"""
import difflib
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GLOBS = ["docs/notebooks/*.jl", "lib/BasisMonteCarlo/docs/notebooks/*.jl"]
MARKER = re.compile(r"^# ╔═╡ ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$")
PACKAGE_CELLS = {"00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"}


def fatou(code):
    """The formatted code, or None when Fatou cannot format it (left as it is, and reported)."""
    r = subprocess.run(["fatou", "format", "-"], input=code, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        return None
    return r.stdout


def format_notebook(text):
    """(new text, [(cell id, old code, new code)] for every cell that changed, failures)."""
    lines = text.split("\n")
    out, changed, failures = [], [], 0
    i = 0
    while i < len(lines):
        m = MARKER.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        cell = m.group(1)
        out.append(lines[i])
        i += 1
        meta = []
        while i < len(lines) and lines[i].startswith("# ╠═╡ "):
            meta.append(lines[i])
            i += 1
        out.extend(meta)
        j = i
        while j < len(lines) and not lines[j].startswith("# ╔═╡ "):
            j += 1
        body = lines[i:j]
        # the code, then the blank line Pluto writes before the next marker
        trailing = 0
        while body and body[-1] == "":
            body.pop()
            trailing += 1
        code = "\n".join(body)
        skip = cell in PACKAGE_CELLS or any("disabled = true" in x for x in meta) or code.startswith("#=╠═╡")
        new = code
        if not skip and code.strip():
            formatted = fatou(code + "\n")
            if formatted is None:
                failures += 1
                print(f"  cell {cell}: Fatou could not format it (left as it is)", file=sys.stderr)
            else:
                new = formatted.rstrip("\n")
                if new != code:
                    changed.append((cell, code, new))
        out.extend(new.split("\n") if new else [])
        out.extend([""] * trailing)
        i = j
    return "\n".join(out), changed, failures


def main(argv):
    write = "--write" in argv
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        paths = sorted(p for g in DEFAULT_GLOBS for p in glob.glob(os.path.join(ROOT, g)))
    n_changed = n_failed = 0
    for path in paths:
        text = open(path, encoding="utf-8").read()
        new, changed, failures = format_notebook(text)
        n_failed += failures
        if not changed:
            continue
        n_changed += 1
        rel = os.path.relpath(path, ROOT)
        if write:
            open(path, "w", encoding="utf-8").write(new)
            print(f"formatted {len(changed)} cell(s) of {rel}")
        else:
            print(f"would reformat {len(changed)} cell(s) of {rel}")
            for cell, old, fmt in changed:
                sys.stdout.writelines(
                    difflib.unified_diff(
                        (old + "\n").splitlines(True), (fmt + "\n").splitlines(True), f"{rel} {cell}", f"{rel} {cell} (fatou)"
                    )
                )
    verb = "formatted" if write else "would reformat"
    print(f"{n_changed} of {len(paths)} notebook(s) {verb}" + (f"; {n_failed} cell(s) Fatou could not format" if n_failed else ""))
    return 1 if (n_failed or (n_changed and not write)) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
