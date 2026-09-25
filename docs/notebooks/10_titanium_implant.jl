### A Pluto.jl notebook ###
# v0.1.0

using Markdown
using InteractiveUtils

# ╔═╡ 10000001-0000-4000-8000-000000000001
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 10000001-0000-4000-8000-000000000002
import BasisSimulator as BS

# ╔═╡ 10000001-0000-4000-8000-000000000003
# ╠═╡ show_logs = false
# Use CairoMakie for faithful build-time rendering. Snapshot can still isolate
# and compile independent browser-safe islands without hoisting this import.
import CairoMakie as Mke

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
end;

# ╔═╡ 10000002-0000-4000-8000-000000000001
md"""
# 10 · Titanium Implant · Metal Artifacts

**A user-defined metal, and the artifacts it produces in the standard reconstruction chain.**

Metal artifacts come from two physical mechanisms that `BasisSimulator.jl` models by
construction:

1. **Beam hardening.** The polychromatic forward model sums ``I = \sum_E w_E\, e^{-L_E}``
   before the log, so a dense high-``Z`` object hardens the transmitted spectrum and the line
   integrals stop growing in proportion to the path length. Reconstructed: dark bands, most of
   all *between* two metal objects.
2. **Photon starvation.** Behind thick metal few photons reach the detector, so quantum and
   electronic noise dominate those rays. Reconstructed: noisy streaks along the directions that
   cross the metal.

No metal ships in the built-in material table, which is the point of this notebook: **any
implant material can be defined by the user** as an elemental composition and a density, and
NIST XCOM (through XrayAttenuation.jl) supplies its cross-sections. The reconstruction uses the
package's standard chain, a water beam-hardening correction and FDK. The package ships no
metal-artifact-reduction (MAR) algorithm, and the water correction is not designed for metal;
the notebook measures how much artifact is left.

```
define titanium → water cylinder with two Ti rods, and the same cylinder without them
   → simulate both (:dd_fast) → water BHC → FDK → HU
   → images, a profile through the rods, and the artifact measured against the metal-free scan
```
"""

# ╔═╡ 10000002-0000-4000-8000-000000000002
Markdown.parse("""
## Notebook Setup

The shared docs environment and the backend-neutral `GPUSelect.Storage()` device choice of
notebook 01.

**Backend detected:** $(GPU_BACKEND.name)
""")

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
);

# ╔═╡ 10000004-0000-4000-8000-000000000001
md"""
## 2. Phantom: water cylinder + two titanium rods

A 26 cm water cylinder with two 1.5 cm titanium rods, 4 cm either side of the centre (two rods,
so that the classic *between-rod dark band* appears). A second phantom is the same cylinder with
the rods made of water: scanned identically, it is the artifact-free reference every
measurement below is compared with.
"""

# ╔═╡ 10000004-0000-4000-8000-000000000002
phantom_mask = let
    n = 128
    nz = 8
    vox = 30.0 / n                  # 30 cm field, cm per voxel
    mask = zeros(UInt8, n, n, nz)
    c = (n + 1) / 2
    r_water = 13.0 / vox            # 26 cm diameter water cylinder
    r_ti = 0.75 / vox               # 1.5 cm diameter rods
    rods = ((c - 4.0 / vox, c), (c + 4.0 / vox, c))
    for k in 1:nz, j in 1:n, i in 1:n
        (i - c)^2 + (j - c)^2 <= r_water^2 || continue
        mask[i, j, k] = any((i - x)^2 + (j - y)^2 <= r_ti^2 for (x, y) in rods) ? 2 : 1
    end
    (mask = mask, vox = vox)
end;

# ╔═╡ 10000004-0000-4000-8000-000000000003
phantom = BS.Phantom(
    to_gpu(phantom_mask.mask),
    Dict(0 => BS.XA.Materials.air, 1 => BS.XA.Materials.water, 2 => titanium),
    ntuple(_ -> phantom_mask.vox, 3),
);

# ╔═╡ 10000004-0000-4000-8000-000000000004
phantom_reference = BS.Phantom(       # the same geometry, rods made of water
    to_gpu(phantom_mask.mask),
    Dict(0 => BS.XA.Materials.air, 1 => BS.XA.Materials.water, 2 => BS.XA.Materials.water),
    ntuple(_ -> phantom_mask.vox, 3),
);

# ╔═╡ 10000005-0000-4000-8000-000000000001
md"""
## 3. Scan and reconstruct

A 120 kVp / 200 mA axial acquisition (360 views, 0.5 s) on a generic 16-row scanner with the
default large-body bowtie, then the standard chain: `calibrate_bhc_water` → `apply_bhc_water` →
FDK → HU. The function runs it for either phantom and also returns the largest line integral in
the sinogram.
"""

# ╔═╡ 10000005-0000-4000-8000-000000000002
scanner = BS.EICTScanner(
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
sim_opts = BS.SimOptions(seed = 42, projector = :dd_fast);

# ╔═╡ 10000005-0000-4000-8000-000000000005
recon_opts = BS.ReconOptions(matrix_size = (256, 256, 8), fov_cm = 30.0);

# ╔═╡ 10000005-0000-4000-8000-000000000006
"""
    scan_hu(phantom) -> (; hu, max_line_integral)

`create_eict_workspace` → `simulate!` → water BHC → FDK → HU, returning a CPU volume.
"""
function scan_hu(phantom)
    ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, protocol, sim_opts; report_dose = false)
    bhc = BS.calibrate_bhc_water(sim_opts, protocol; scanner, geom = ws.geom)
    sino = BS.apply_bhc_water(ws.sinogram, bhc)
    ws_fdk = BS.create_fdk_recon_workspace(sino, ws.geom, recon_opts.matrix_size)
    μ = BS.reconstruct!(ws_fdk, sino, ws.geom)
    out = (hu = Float32.(BS.to_hounsfield(Array(μ); μ_water = bhc.μ_water_ref)),
           max_line_integral = Float64(maximum(ws.sinogram)))
    ws = nothing; ws_fdk = nothing; sino = nothing; μ = nothing
    GC.gc(true)
    return out
end;

# ╔═╡ 10000005-0000-4000-8000-000000000007
scan_ti = scan_hu(phantom);

# ╔═╡ 10000005-0000-4000-8000-000000000008
scan_ref = scan_hu(phantom_reference);

# ╔═╡ 10000006-0000-4000-8000-000000000001
md"""
## Results

### 1. The artifacts

Top: the metal-free reference and the titanium scan in a soft-tissue window, and the titanium
scan in a wide window that shows the rods themselves. Bottom: the difference titanium −
reference, which is the artifact alone (the anatomy cancels), and the HU profile along the line
through both rods.
"""

# ╔═╡ 10000006-0000-4000-8000-000000000002
fig_artifacts = let
    k = size(scan_ti.hu, 3) ÷ 2
    ti, ref = scan_ti.hu[:, :, k], scan_ref.hu[:, :, k]
    n = size(ti, 1)
    x_cm = ((1:n) .- (n + 1) / 2) .* (recon_opts.fov_cm / n)
    row = n ÷ 2

    fig = Mke.Figure(size = (1350, 900))
    tk = (aspect = Mke.DataAspect(), titlesize = 20)
    ax = Mke.Axis(fig[1, 1]; title = "Reference (no metal) · [-200, 200] HU", tk...)
    Mke.heatmap!(ax, ref; colorrange = (-200, 200), colormap = :grays); Mke.hidedecorations!(ax)
    ax = Mke.Axis(fig[1, 2]; title = "Titanium rods · [-200, 200] HU", tk...)
    hm = Mke.heatmap!(ax, ti; colorrange = (-200, 200), colormap = :grays); Mke.hidedecorations!(ax)
    Mke.Colorbar(fig[1, 3], hm; label = "HU")
    ax = Mke.Axis(fig[1, 4]; title = "Titanium rods · [-1000, 10000] HU", tk...)
    hw = Mke.heatmap!(ax, ti; colorrange = (-1000, 10000), colormap = :grays); Mke.hidedecorations!(ax)
    Mke.Colorbar(fig[1, 5], hw; label = "HU")

    ax = Mke.Axis(fig[2, 1:2]; title = "Artifact: titanium − reference · [-300, 300] HU", tk...)
    hd = Mke.heatmap!(ax, ti .- ref; colorrange = (-300, 300), colormap = :RdBu); Mke.hidedecorations!(ax)
    Mke.hlines!(ax, [row]; color = (:black, 0.4), linestyle = :dash)
    Mke.Colorbar(fig[2, 3], hd; label = "ΔHU")
    ax = Mke.Axis(fig[2, 4:5]; title = "Profile through both rods", xlabel = "x (cm)", ylabel = "HU",
        titlesize = 20, xlabelsize = 18, ylabelsize = 18)
    Mke.lines!(ax, x_cm, ref[:, row]; label = "reference", linewidth = 2)
    Mke.lines!(ax, x_cm, ti[:, row]; label = "titanium", linewidth = 2)
    Mke.ylims!(ax, -400, 1000)
    Mke.axislegend(ax; position = :lt)
    Mke.save(joinpath(@__DIR__, "..", "assets", "titanium_artifacts.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 10000006-0000-4000-8000-000000000003
let
    k = size(scan_ti.hu, 3) ÷ 2
    ti, ref = scan_ti.hu[:, :, k], scan_ref.hu[:, :, k]
    cn = size(ti, 1) ÷ 2
    band(v) = mean(v[(cn - 8):(cn + 8), (cn - 2):(cn + 2)])        # between the rods
    away(v) = mean(v[(cn - 8):(cn + 8), (cn - 72):(cn - 64)])      # water, 8 cm below the rod axis
    ring(v) = std(v[(cn - 40):(cn + 40), (cn + 30):(cn + 40)])     # water band 3–4 cm above the axis
    Markdown.parse("""
    | measurement | reference (no metal) | titanium | titanium − reference |
    |:--|--:|--:|--:|
    | between the rods, mean HU | $(round(band(ref); digits = 1)) | $(round(band(ti); digits = 1)) | $(round(band(ti) - band(ref); digits = 1)) |
    | water far from the rods, mean HU | $(round(away(ref); digits = 1)) | $(round(away(ti); digits = 1)) | $(round(away(ti) - away(ref); digits = 1)) |
    | streak region beside the rods, σ (HU) | $(round(ring(ref); digits = 1)) | $(round(ring(ti); digits = 1)) | |
    | peak HU (rods) | | $(round(Int, maximum(ti))) | |
    | largest line integral in the sinogram | $(round(scan_ref.max_line_integral; digits = 2)) | $(round(scan_ti.max_line_integral; digits = 2)) | |

    The water BHC holds the metal-free cylinder at water ($(round(away(ref); digits = 1)) HU far
    from the rods). With titanium present, the region between the rods drops by
    $(round(band(ref) - band(ti); digits = 0)) HU: the beam-hardening dark band that a water
    correction cannot remove. The σ beside the rods rises from $(round(ring(ref); digits = 1)) to
    $(round(ring(ti); digits = 1)) HU, the streaks. A line integral of
    $(round(scan_ti.max_line_integral; digits = 1)) means the rays through both rods keep about
    $(round(100 * exp(-scan_ti.max_line_integral); sigdigits = 2)) % of their unattenuated signal.
    """)
end

# ╔═╡ 10000007-0000-4000-8000-000000000001
md"""
## Scope

- The water beam-hardening correction is a per-column **water** polynomial. It is exact for
  water and leaves the metal artifacts, as measured above. A MAR method (for example sinogram
  inpainting over the metal trace) would sit on the same reconstruction stack; none ships with
  the package.
- Any other implant works the same way: stainless steel, CoCr or amalgam are one
  `XA.Material(...)` call each, with NIST XCOM supplying the cross-sections.
"""

# ╔═╡ 10000008-0000-4000-8000-000000000001
md"""
## Summary

- A user-defined `XA.Material` enters the same attenuation pipeline as every built-in tissue.
- Scanning the same geometry with and without the metal isolates the artifact: the difference
  image and the table show the between-rod dark band and the streaks.
- `:dd_fast`, the default projector, keeps its anti-aliased distance-driven footprint in the
  severe beam hardening around titanium.
- No MAR claim is made; the table is the baseline a future MAR method would be measured against.
"""

# ╔═╡ Cell order:
# ╟─10000002-0000-4000-8000-000000000001
# ╟─10000002-0000-4000-8000-000000000002
# ╟─10000001-0000-4000-8000-000000000001
# ╠═10000001-0000-4000-8000-000000000002
# ╟─10000001-0000-4000-8000-000000000003
# ╟─10000001-0000-4000-8000-000000000007
# ╟─10000001-0000-4000-8000-000000000004
# ╟─10000001-0000-4000-8000-000000000005
# ╠═10000001-0000-4000-8000-000000000006
# ╟─10000002-0000-4000-8000-000000000003
# ╟─10000003-0000-4000-8000-000000000001
# ╠═10000003-0000-4000-8000-000000000002
# ╟─10000004-0000-4000-8000-000000000001
# ╠═10000004-0000-4000-8000-000000000002
# ╠═10000004-0000-4000-8000-000000000003
# ╠═10000004-0000-4000-8000-000000000004
# ╟─10000005-0000-4000-8000-000000000001
# ╠═10000005-0000-4000-8000-000000000002
# ╠═10000005-0000-4000-8000-000000000003
# ╠═10000005-0000-4000-8000-000000000004
# ╠═10000005-0000-4000-8000-000000000005
# ╠═10000005-0000-4000-8000-000000000006
# ╠═10000005-0000-4000-8000-000000000007
# ╠═10000005-0000-4000-8000-000000000008
# ╟─10000006-0000-4000-8000-000000000001
# ╟─10000006-0000-4000-8000-000000000002
# ╟─10000006-0000-4000-8000-000000000003
# ╟─10000007-0000-4000-8000-000000000001
# ╟─10000008-0000-4000-8000-000000000001
