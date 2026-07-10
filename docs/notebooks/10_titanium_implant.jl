### A Pluto.jl notebook ###
# v0.1.0

using Markdown
using InteractiveUtils

# ╔═╡ 10000001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 10000001-0000-4000-8000-000000000002
import BasisSimulator as BS

# ╔═╡ 10000001-0000-4000-8000-000000000003
# ╠═╡ show_logs = false
import CairoMakie as CM

# ╔═╡ 10000001-0000-4000-8000-000000000007
import PlutoUI

# ╔═╡ 10000001-0000-4000-8000-000000000004
using Statistics: mean, std

# ╔═╡ 10000001-0000-4000-8000-000000000005
using Unitful: @u_str

# ╔═╡ 10000001-0000-4000-8000-000000000006
begin
    import GPUSelect
    AT = GPUSelect.Storage()     # the backend array type, directly: MtlArray / CuArray / ROCArray
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end

# ╔═╡ 10000002-0000-4000-8000-000000000001
md"""
# 10 · Titanium Implant · Metal Artifacts

**A user-defined metal material, and the artifacts it produces.**

Metal artifacts in CT come from two physical mechanisms that
`BasisSimulator.jl` models by construction:

1. **Beam hardening** — the polychromatic forward model accumulates
   ``I = \sum_E w_E\, e^{-L_E}`` before the negative-log transform, so a
   high-``Z``, high-density object hardens the transmitted spectrum and the
   log line integrals stop being proportional to path length. Reconstructed:
   dark bands, especially *between* two metal objects.
2. **Photon starvation** — behind thick metal only a handful of photons reach
   the detector, so Poisson noise (plus electronic noise) dominates those
   rays. Reconstructed: noisy streaks along the metal-crossing directions.

No metal ships in the built-in material registry — and that is the point of
this notebook: **any implant material can be registered by the user** as an
elemental composition and density resolved against NIST XCOM, the same
mechanism used for the built-in tissues and Gammex rods. Note that
`BasisSimulator` currently ships **no metal-artifact-reduction (MAR)
algorithm**; the bundled beam-hardening correction is a water-based
correction.

**Backend detected:** $(GPU_BACKEND.name)

```
Define titanium → build water + implant phantom → simulate with :dd_fast
   → uncorrected FDK → soft-tissue / wide-window comparison
   → quantitative dark-band, water, and peak-HU checks
```
"""

# ╔═╡ 10000002-0000-4000-8000-000000000002
md"""
## Notebook Setup

This notebook uses the shared docs environment and the same backend-neutral
`GPUSelect.Storage()` pattern as notebooks 01–04. The simulation is intentionally
left uncorrected: the goal is to expose the metal artifact mechanisms, not to
present a MAR algorithm.
"""

# ╔═╡ 10000002-0000-4000-8000-000000000003
PlutoUI.TableOfContents()

# ╔═╡ 10000003-0000-4000-8000-000000000001
md"""
## 1. Titanium as a custom `XA.Material`

Titanium is exposed by `XrayAttenuation.jl` as an `Element`, not a prebuilt
`Material`, so we construct the material ourselves: pure Ti (``Z = 22``,
mass fraction 1.0) at 4.54 g/cm³. `ZA_ratio` and the mean excitation energy
`I` come from the element data (0.45948, 233 eV).
"""

# ╔═╡ 10000003-0000-4000-8000-000000000002
titanium = BS.XA.Material(
    "Titanium (implant)",
    0.45948,            # ⟨Z/A⟩ for Ti
    233.0u"eV",         # mean excitation energy
    4.54u"g/cm^3",      # bulk density
    Dict(22 => 1.0),    # elemental composition: pure Ti
)

# ╔═╡ 10000004-0000-4000-8000-000000000001
md"""
## 2. Phantom: water cylinder + two titanium rods

A 26 cm water cylinder with two 1.5 cm titanium rods, 4 cm either side of
center — two rods so the classic *between-rod dark band* shows up. The rods
give a maximum line integral of ≈ 10, comfortably inside the simulator's
dynamic range, and leave only a few detected photons on the rays crossing
both rods (starvation streaks).
"""

# ╔═╡ 10000004-0000-4000-8000-000000000002
phantom = let
    n = 128
    nz = 8
    fov = 30.0                      # cm
    vox = fov / n
    mask = zeros(UInt8, n, n, nz)
    c = (n + 1) / 2
    r_water = 13.0 / vox            # 26 cm diameter water cylinder
    r_ti = 0.75 / vox               # 1.5 cm diameter rods
    rod1 = (c - 4.0 / vox, c)
    rod2 = (c + 4.0 / vox, c)
    for k in 1:nz, j in 1:n, i in 1:n
        d2 = (i - c)^2 + (j - c)^2
        if d2 <= r_water^2
            mask[i, j, k] = 1
            if (i - rod1[1])^2 + (j - rod1[2])^2 <= r_ti^2 ||
               (i - rod2[1])^2 + (j - rod2[2])^2 <= r_ti^2
                mask[i, j, k] = 2
            end
        end
    end
    BS.Phantom(
        to_gpu(UInt8.(mask)),
        Dict(1 => BS.XA.Materials.water, 2 => titanium),
        (vox, vox, vox),
    )
end;

# ╔═╡ 10000005-0000-4000-8000-000000000001
md"""
## 3. Scan and reconstruct

Standard 120 kVp / 200 mA axial acquisition on the generic scanner, FDK
reconstruction, shown **uncorrected** so the artifacts are what the physics
produced.
"""

# ╔═╡ 10000005-0000-4000-8000-000000000002
scanner = BS.Scanner(
    source_to_isocenter = 540.0,
    source_to_detector = 950.0,
    detector_rows = 16,
    detector_cols = 512,
    detector_row_size = 1.0,
    detector_col_size = 1.0,
    detector_material = :lumex,
    detector_depth = 3.0,
);

# ╔═╡ 10000005-0000-4000-8000-000000000003
protocol = BS.CTProtocol(kVp = 120.0, mA = 200.0, views = 360, rotation_time = 0.5);

# ╔═╡ 10000005-0000-4000-8000-000000000004
sim_opts = BS.SimOptions(fidelity = :eict, seed = 42, projector = :dd_fast);

# ╔═╡ 10000005-0000-4000-8000-000000000005
recon_opts = BS.ReconOptions(matrix_size = (256, 256, 8), fov_cm = 30.0);

# ╔═╡ 10000005-0000-4000-8000-000000000006
hu = let
    ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, protocol, sim_opts)

    sino = ws.sinogram
    ws_fdk = BS.create_fdk_recon_workspace(sino, ws.geom, recon_opts.matrix_size)
    recon_μ = BS.reconstruct!(ws_fdk, sino, ws.geom)

    μ_water = BS.compute_polychromatic_μ_water(
        sim_opts, protocol;
        scanner = scanner, geom = ws.geom, water_path_cm = 26.0,
    )
    result = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = μ_water))

    ws = nothing
    ws_fdk = nothing
    recon_μ = nothing
    GC.gc(true)

    result
end;

# ╔═╡ 10000006-0000-4000-8000-000000000001
md"""
## Results

### 01. The artifacts

Left: soft-tissue window — the between-rod **beam-hardening dark band** and
the radiating **photon-starvation streaks**. Right: wide window — the rods
themselves (titanium reconstructs at several thousand HU).
"""

# ╔═╡ 10000006-0000-4000-8000-000000000002
fig_artifacts = let
    sl = hu[:, :, 4]
    fig = CM.Figure(size = (900, 460))
    ax1 = CM.Axis(fig[1, 1], title = "Soft-tissue window [-500, 500] HU", aspect = 1)
    CM.heatmap!(ax1, sl; colorrange = (-500, 500), colormap = :grays)
    CM.hidedecorations!(ax1)
    ax2 = CM.Axis(fig[1, 2], title = "Wide window [-1000, 4000] HU", aspect = 1)
    CM.heatmap!(ax2, sl; colorrange = (-1000, 4000), colormap = :grays)
    CM.hidedecorations!(ax2)
    CM.save(joinpath(@__DIR__, "..", "assets", "titanium_artifacts.png"), fig)
    fig
end

# ╔═╡ 10000006-0000-4000-8000-000000000003
let
    sl = hu[:, :, 4]
    cn = size(sl, 1) ÷ 2
    band = mean(sl[(cn - 8):(cn + 8), cn])          # between the rods
    water_roi = mean(sl[(cn - 8):(cn + 8), cn - 70]) # water away from the rods
    peak = maximum(sl)
    md"""
    | Measurement | HU |
    |---|---|
    | Between-rod dark band (mean) | $(round(Int, band)) |
    | Water ROI away from rods (mean) | $(round(Int, water_roi)) |
    | Titanium rod (peak) | $(round(Int, peak)) |

    The band between the rods drops **hundreds of HU below water** — the
    classic beam-hardening signature — while starvation streaks radiate
    along the rod-crossing directions.
    """
end

# ╔═╡ 10000007-0000-4000-8000-000000000001
md"""
## Verification and Scope

- The bundled sinogram/image-domain beam-hardening correction is a **water
  polynomial**; it is not designed for metal and will not remove these
  bands. A dedicated MAR method (e.g. sinogram inpainting over the metal
  trace) is a natural extension on the shared reconstruction stack, but no
  MAR algorithm ships with this release.
- Any other implant material works the same way: stainless steel, CoCr, or
  amalgam are one `XA.Material(...)` constructor call each, with NIST XCOM
  providing the cross-sections.
- On CPU this notebook takes a few minutes (the polychromatic spectral path
  is GPU-oriented); with Metal/CUDA/ROCm available it runs in seconds.
"""

# ╔═╡ 10000008-0000-4000-8000-000000000001
md"""
## Summary

- A user-defined `XA.Material` enters the same attenuation pipeline as every
  built-in tissue.
- `:dd_fast` remains the default projector and retains the anti-aliased DD
  footprint in the severe beam-hardening region around titanium.
- The uncorrected reconstruction intentionally demonstrates the between-rod
  dark band and photon-starvation streaks.
- The notebook makes no MAR claim; its quantitative table is the regression
  target to preserve when a future MAR method is added.
"""

# ╔═╡ Cell order:
# ╟─10000002-0000-4000-8000-000000000001
# ╟─10000002-0000-4000-8000-000000000002
# ╠═10000001-0000-4000-8000-000000000001
# ╠═10000001-0000-4000-8000-000000000002
# ╠═10000001-0000-4000-8000-000000000003
# ╠═10000001-0000-4000-8000-000000000007
# ╠═10000001-0000-4000-8000-000000000004
# ╠═10000001-0000-4000-8000-000000000005
# ╠═10000001-0000-4000-8000-000000000006
# ╠═10000002-0000-4000-8000-000000000003
# ╟─10000003-0000-4000-8000-000000000001
# ╠═10000003-0000-4000-8000-000000000002
# ╟─10000004-0000-4000-8000-000000000001
# ╠═10000004-0000-4000-8000-000000000002
# ╟─10000005-0000-4000-8000-000000000001
# ╠═10000005-0000-4000-8000-000000000002
# ╠═10000005-0000-4000-8000-000000000003
# ╠═10000005-0000-4000-8000-000000000004
# ╠═10000005-0000-4000-8000-000000000005
# ╠═10000005-0000-4000-8000-000000000006
# ╟─10000006-0000-4000-8000-000000000001
# ╠═10000006-0000-4000-8000-000000000002
# ╟─10000006-0000-4000-8000-000000000003
# ╟─10000007-0000-4000-8000-000000000001
# ╟─10000008-0000-4000-8000-000000000001
