# `docs/notebooks/archived/`

Notebooks parked here are **kept in the repo** but **not published** to
the docs site or rendered to HTML.

## Why

Both `docs/extract_all.jl` (Pluto → static HTML) and `docs/app.jl`
(docs listing) discover notebooks via:

```julia
[
    splitext(f)[1]
    for f in readdir(joinpath(@__DIR__, "notebooks"))
    if endswith(f, ".jl") && !endswith(f, ".sessions.toml")
]
```

`readdir` returns subdirectory entries by name only — `"archived"`
doesn't end in `.jl`, so the filter drops it.  Anything *inside* this
folder is therefore invisible to both the HTML build and the docs
listing without any `BASISSIM_SKIP_NOTEBOOKS` configuration.

## When to use

Drop a notebook here when:

- It's a useful reference for an old approach we may want to consult
  again (e.g. before a pipeline rewrite).
- We've replaced it with a newer flow but don't want to lose the
  history beyond `git log`.
- It's a one-off exploration we don't want shown alongside the
  curated example set.

To resurrect: move the `.jl` back up to `docs/notebooks/`.
