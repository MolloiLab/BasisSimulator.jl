### A Pluto.jl notebook ###
# v0.2.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 11000001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 11000001-0000-4000-8000-000000000004
using Statistics: mean, std

# ╔═╡ 11000001-0000-4000-8000-000000000005
using PlutoUI

# ╔═╡ 11000001-0000-4000-8000-000000000002
import BasisSimulator as BS

# ╔═╡ 11000001-0000-4000-8000-000000000003
# ╠═╡ show_logs = false
import CairoMakie as CM

# ╔═╡ 11000001-0000-4000-8000-000000000006
begin
    import GPUSelect
    AT = GPUSelect.Storage()     # the backend array type, directly: MtlArray / CuArray / ROCArray
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end

# ╔═╡ 11000002-0000-4000-8000-000000000001
md"""
# 11 · Helical Scanning · Narrow Collimation, Long Coverage

**One new kwarg — `pitch` — turns any protocol into a spiral scan.**

Helical (spiral) CT solves a geometry problem no detector can: covering a
long ``z`` range with a **narrow** beam. Before spiral CT, long coverage
meant **step-and-shoot**: scan a slab, move the table, scan the next slab.
Each station is a clean axial scan — but every station has cone-beam edges,
and the stitched volume carries a seam at every station boundary. A helical
scan instead sweeps the collimator *past* every slice continuously: each
``z`` position is, at some moment of the spiral, at the centre of the beam.

This notebook covers a **32 cm** z-slab both ways at matched exposure, on a
wide-cone 256 × 0.625 mm (16 cm) volume scanner:

|                | collimation | rotations | table          |
|:---------------|:-----------|:----------|:---------------|
| volume axial   | 160 mm (full detector) | 2 × axial | steps 16 cm between stations |
| **helical**    | **20 mm (1/8th!)**     | 16 turns  | glides at pitch 1.0 |

The forward projection is the anti-aliased `:dd_fast` projector — helical
costs it nothing (the projectors consume per-view source/detector arrays; a
helix is just a z-ramp in those arrays). Helical reconstruction is
**rebinned WFBP** (Stierstorfer *et al.* 2004 — the production spiral
algorithm family), dispatched automatically whenever the geometry is
helical. Both pipelines get the full clinical correction stack from
notebook 01: sinogram BHC → recon → image BHC → HU → noise floor → cupping.

**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 11000003-0000-4000-8000-000000000001
md"""
## 1. A phantom that changes along ``z`` — with no strong hardeners

Two design rules. First, a uniform cylinder would look identical at every
slice — useless for judging long-``z`` fidelity — so every slice must be
different. Second, **no high-Z inserts**: beam hardening is its own topic
(notebook 10); here it would only confound the geometry comparison.

- **30 cm water body**, 36 cm long (coarse 2.3 mm in-plane / 2 mm z voxels);
- a **lung-density rod** (≈ −700 HU, radiologically *soft*) that **winds
  helically** around the body axis — one turn per 12 cm of ``z``, so its
  angular position tags every slice;
- an **adipose cone** (≈ −90 HU) on the axis, radius tapering 4 cm → 0
  across the slab — its diameter tags every slice.
"""

# ╔═╡ 11000003-0000-4000-8000-000000000002
phantom_data = let
    n = 128                          # in-plane grid
    nz = 180                         # 2 mm z-voxels → 36 cm slab
    fov = 30.0                       # cm, in-plane phantom extent
    zext = 36.0                      # cm, phantom z extent
    vox = fov / n
    voxz = zext / nz
    mask = zeros(UInt8, n, n, nz)
    c = (n + 1) / 2
    r_body = 12.0 / vox              # 24 cm water body
    for k in 1:nz, j in 1:n, i in 1:n
        if (i - c)^2 + (j - c)^2 <= r_body^2
            mask[i, j, k] = 1
        end
    end
    # helically winding lung-density rod: r = 7 cm, one turn per 12 cm of z
    for k in 1:nz
        θ = 2π * (k * voxz) / 12.0
        cx = c + (7.0 / vox) * cos(θ)
        cy = c + (7.0 / vox) * sin(θ)
        r = 1.0 / vox                # 2 cm diameter rod
        for j in 1:n, i in 1:n
            if (i - cx)^2 + (j - cy)^2 <= r^2 && mask[i, j, k] == 1
                mask[i, j, k] = 2
            end
        end
    end
    # central adipose cone: radius tapers 4 cm → 0 across the slab
    for k in 1:nz
        r = (4.0 * (1 - (k - 1) / (nz - 1))) / vox
        for j in 1:n, i in 1:n
            if (i - c)^2 + (j - c)^2 <= r^2 && mask[i, j, k] == 1
                mask[i, j, k] = 3
            end
        end
    end
    (mask = mask, vox = vox, voxz = voxz, n = n, nz = nz, zext = zext)
end;

# ╔═╡ 11000003-0000-4000-8000-000000000003
phantom_materials = Dict(
    1 => BS.XA.Materials.water,
    2 => BS.XA.Materials.lung,
    3 => BS.XA.Materials.adipose,
);

# ╔═╡ 11000003-0000-4000-8000-000000000004
phantom = BS.Phantom(
    to_gpu(phantom_data.mask),
    phantom_materials,
    (phantom_data.vox, phantom_data.vox, phantom_data.voxz),
);

# ╔═╡ 11000004-0000-4000-8000-000000000001
md"""
## 2. The helical scan: `pitch` is the whole API

IEC pitch = table feed per rotation ÷ active collimation. With **20 mm**
collimation (32 of the scanner's 64 × 0.625 mm rows), pitch 1.0, and 16
rotations, the table travels

```math
\text{travel} = \text{pitch} \times \text{collimation} \times n_\text{rot}
             = 1.0 \times 20\,\text{mm} \times 16 = 32\,\text{cm}.
```

Everything else — `Phantom`, `Scanner`, `SimOptions`, `ReconOptions`,
`simulate!`, the corrections, the recon call — is untouched from an axial
workflow.
"""

# ╔═╡ 11000004-0000-4000-8000-000000000002
scanner = BS.Scanner(
    source_to_isocenter = 541.0,
    source_to_detector = 949.0,
    detector_rows = 256,             # 256 × 0.625 mm = 16 cm wide-cone volume scanner
    detector_cols = 512,
    detector_row_size = 0.625,
    detector_col_size = 1.0,
);

# ╔═╡ 11000004-0000-4000-8000-000000000003
begin
    protocol_helical = BS.CTProtocol(
        kVp = 120.0, mA = 200.0, views = 360,
        rotation_time = 0.5,
        collimation_mm = 20.0,       # NARROW: half the physical detector
        pitch = 1.0,                 # ← the one helical kwarg
        n_rotations = 16,            # 16 × 2 cm = 32 cm table travel
    )
    protocol_axial = BS.CTProtocol(
        kVp = 120.0, mA = 200.0, views = 360,
        rotation_time = 0.5,
        collimation_mm = 160.0,      # WIDE: the full 16 cm detector, volume mode
    )
    # use_heel_effect = false: the anode heel is a per-ROW spectral gradient.
    # Axial scans absorb it (each slice keeps its rows); a helical scan sweeps
    # the rows past every voxel, which needs the per-row water calibration real
    # scanners apply.  Until BasisSimulator ships per-row BHC, disable it here
    # so the comparison isolates GEOMETRY (same choice as notebook 09).
    sim_opts = BS.SimOptions(fidelity = :eict, seed = 42, projector = :dd_fast,
        use_heel_effect = false)
    recon_opts = BS.ReconOptions(matrix_size = (160, 160, 150), fov_cm = 30.0, z_cm = 30.0)
    recon_opts_station = BS.ReconOptions(matrix_size = (160, 160, 80), fov_cm = 30.0, z_cm = 16.0)
end;

# ╔═╡ 11000005-0000-4000-8000-000000000001
md"""
## 3. Corrections — the notebook-01 clinical stack

One BHC model per protocol (the bowtie-hardened spectrum depends on the
collimation), then the standard chain per reconstruction: sinogram-domain
BHC → recon → image-domain BHC → HU (using the BHC-calibrated
``\mu_\text{water}``) → DAS noise floor → residual radial cupping
correction.
"""

# ╔═╡ 11000005-0000-4000-8000-000000000002
"""
    corrected_recon(sino_gpu, geom, matrix_size, bhc) -> Array{Float32,3} (HU)

Notebook-01 correction chain around a single reconstruction (helical
geometries dispatch to WFBP inside `reconstruct!` automatically).  Noise
floor + cupping are applied by the caller (once per final volume).
"""
function corrected_recon(sino_gpu, geom, matrix_size, bhc)
    sino_bhc = BS.apply_bhc_two_material(sino_gpu, bhc.model, geom, matrix_size)
    sino_g = to_gpu(Float32.(sino_bhc))
    ws_fdk = BS.create_fdk_recon_workspace(sino_g, geom, matrix_size)
    recon_μ = BS.reconstruct!(ws_fdk, sino_g, geom)
    # NOTE: no image-domain BHC — audit found it deflates dense-material HU
    # (a scaled self-subtraction, not So et al.); the sinogram-domain BHC
    # (calibrated on the full detected spectrum) is the whole correction.
    return Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc.μ_water))
end;

# ╔═╡ 11000006-0000-4000-8000-000000000001
md"""
## 4. Run both acquisitions

**Helical**: one `simulate!`, one reconstruction.  **Volume axial
(step-and-shoot)**: two independent 16 cm wide-cone stations back to back —
for each, the table (here: the phantom window) moves so the station is
centred at the isocentre — then the two slabs are stitched at ``z = 0``.
"""

# ╔═╡ 11000006-0000-4000-8000-000000000002
helical_result = let
    ws = BS.create_eict_workspace(scanner, protocol_helical, sim_opts, recon_opts, phantom)
    t = @elapsed BS.simulate!(ws, phantom, protocol_helical, sim_opts)
    bhc = let
        model = BS.calibrate_bhc_two_material(
            sim_opts, protocol_helical;
            scanner = scanner, geom = ws.geom,
            order = 2, hu_low = 450.0, hu_high = 600.0,
        )
        (model = model, μ_water = model.μ_water_ref)
    end
    t += @elapsed (hu = corrected_recon(ws.sinogram, ws.geom, recon_opts.matrix_size, bhc))
    ws = nothing
    GC.gc()
    BS.add_system_noise_floor!(hu, 28.0; seed = 1234)
    BS.apply_radial_cupping_correction!(hu; fov_cm = recon_opts.fov_cm)
    (hu = hu, t = t)
end;

# ╔═╡ 11000006-0000-4000-8000-000000000003
sns_result = let
    station_zs = [-8.0, 8.0]                    # 2 stations × 16 cm, back to back
    n_slab = 80                                 # 16 cm at 2 mm recon slices
    hu = zeros(Float32, 160, 160, 150)
    t_total = 0.0
    bhc_ax = nothing
    for (s, z0) in enumerate(station_zs)
        # move the "table": phantom window (±10 cm) centred on this station
        k0 = round(Int, 90 + z0 / phantom_data.voxz)
        window = (k0 - 49):(k0 + 50)            # 100 slices = 20 cm (covers the cone)
        ph_st = BS.Phantom(
            to_gpu(phantom_data.mask[:, :, window]),
            phantom_materials,
            (phantom_data.vox, phantom_data.vox, phantom_data.voxz),
        )
        ws = BS.create_eict_workspace(scanner, protocol_axial, sim_opts, recon_opts_station, ph_st)
        t_total += @elapsed BS.simulate!(ws, ph_st, protocol_axial, sim_opts)
        if bhc_ax === nothing
            model = BS.calibrate_bhc_two_material(
                sim_opts, protocol_axial;
                scanner = scanner, geom = ws.geom,
                order = 2, hu_low = 450.0, hu_high = 600.0,
            )
            bhc_ax = (model = model, μ_water = model.μ_water_ref)
        end
        t_total += @elapsed (hu_st = corrected_recon(
            ws.sinogram, ws.geom, recon_opts_station.matrix_size, bhc_ax))
        ws = nothing
        GC.gc()
        # station slab → global z index: station s covers z0 ± 8 cm.  The two
        # stations overhang the ±15 cm recon volume by 1 cm (they cover ±16 —
        # real volume scans over-range too): clip.
        k_lo = round(Int, (z0 - 8.0 + 15.0) / 0.2) + 1
        valid = max(k_lo, 1):min(k_lo + n_slab - 1, 150)
        hu[:, :, valid] .= hu_st[:, :, (first(valid) - k_lo + 1):(last(valid) - k_lo + 1)]
    end
    BS.add_system_noise_floor!(hu, 28.0; seed = 1234)
    BS.apply_radial_cupping_correction!(hu; fov_cm = recon_opts.fov_cm)
    (hu = hu, t = t_total)
end;

# ╔═╡ 11000007-0000-4000-8000-000000000001
md"""
## 5. Slide through ``z``: phantom truth vs both scans

Top row: the phantom ground truth as a labelled categorical map. Bottom
row: **helical** (20 mm collimation) and **volume axial** (the full 16 cm
detector, 8× wider). Watch the rod rotate and the cone shrink — and watch
what the wide cone does away from each station centre, with the seam at
``z = 0``.
"""

# ╔═╡ 11000007-0000-4000-8000-000000000002
@bind z_idx PlutoUI.Slider(1:150; default = 75, show_value = true)

# ╔═╡ 11000007-0000-4000-8000-000000000003
let
    z_cm = -15.0 + (z_idx - 0.5) * (30.0 / 150)
    kp = clamp(round(Int, (z_cm + 18.0) / phantom_data.voxz + 0.5), 1, phantom_data.nz)

    fig = CM.Figure(size = (1100, 1150))

    # Super-title.  tellwidth = false so the spanning label cannot stretch the
    # columns (Makie FAQ: elements with tellwidth = true resize their column).
    CM.Label(fig[0, 1:3], "z = $(round(z_cm; digits = 1)) cm  (slice $(z_idx)/150)";
        fontsize = 32, font = :bold, tellwidth = false)

    # ── Row 1: phantom ground truth, CATEGORICAL colours + labels ──
    cat_colors = CM.cgrad([:gray10, :steelblue, :darkorange, :seagreen]; categorical = true)
    ax1 = CM.Axis(fig[1, 1:2]; title = "Phantom ground truth", titlesize = 24,
        aspect = CM.DataAspect())
    hm1 = CM.heatmap!(ax1, Float32.(phantom_data.mask[:, :, kp]);
        colormap = cat_colors, colorrange = (-0.5, 3.5))
    CM.hidedecorations!(ax1)
    CM.Colorbar(fig[1, 3], hm1;
        ticks = (0:3, ["air", "water body", "lung rod", "adipose cone"]),
        ticklabelsize = 18, tellheight = false)

    # ── Row 2: the two reconstructions (HU grayscale) ──
    ax2 = CM.Axis(fig[2, 1]; title = "Helical · 20 mm collim, pitch 1.0", titlesize = 24,
        aspect = CM.DataAspect())
    hm2 = CM.heatmap!(ax2, helical_result.hu[:, :, z_idx]; colormap = :grays, colorrange = (-500, 100))
    CM.hidedecorations!(ax2)

    ax3 = CM.Axis(fig[2, 2]; title = "Volume axial · 2 × 16 cm", titlesize = 24,
        aspect = CM.DataAspect())
    CM.heatmap!(ax3, sns_result.hu[:, :, z_idx]; colormap = :grays, colorrange = (-500, 100))
    CM.hidedecorations!(ax3)

    CM.Colorbar(fig[2, 3], hm2; label = "HU", labelsize = 22, ticklabelsize = 16)
    fig
end

# ╔═╡ 11000008-0000-4000-8000-000000000001
md"""
## 6. The money figures: water flatness and the coronal view

Mean HU in a fixed water ROI (clear of rod and cone), slice by slice — the
helical scan should hold water flat across the full 30 cm, while the
step-and-shoot volume shows its station structure. The coronal reformats
show the same thing spatially.
"""

# ╔═╡ 11000008-0000-4000-8000-000000000002
z_profile_fig = let
    xr, yr = 25:36, 73:88            # 9.4 cm left of centre: clear of rod ring & cone
    zs = [-15.0 + (k - 0.5) * 0.2 for k in 1:150]
    prof_hel = [mean(helical_result.hu[xr, yr, k]) for k in 1:150]
    prof_sns = [mean(sns_result.hu[xr, yr, k]) for k in 1:150]

    fig = CM.Figure(size = (1400, 600))
    ax = CM.Axis(fig[1, 1];
        title = "Water HU vs z", titlesize = 32,
        subtitle = "helical (20 mm collim) vs wide-cone volume axial (160 mm collim), matched exposure",
        subtitlesize = 24,
        xlabel = "z (cm)", ylabel = "mean HU (water ROI)",
        xlabelsize = 22, ylabelsize = 22, xticklabelsize = 18, yticklabelsize = 18)
    CM.lines!(ax, zs, prof_hel; linewidth = 4, label = "helical · 20 mm, pitch 1.0 × 16 rot")
    CM.lines!(ax, zs, prof_sns; linewidth = 4, label = "volume axial · 2 × 16 cm stations")
    CM.hlines!(ax, [0.0]; color = :gray, linestyle = :dash, linewidth = 2)
    CM.vlines!(ax, [0.0]; color = (:orange, 0.5), linestyle = :dot, linewidth = 3)
    CM.ylims!(ax, -100, 100)
    CM.axislegend(ax; labelsize = 18, position = :cb)
    fig
end

# ╔═╡ 11000008-0000-4000-8000-000000000003
coronal_fig = let
    yc = 80
    fig = CM.Figure(size = (1400, 640))
    ax1 = CM.Axis(fig[1, 1]; title = "Helical — coronal", titlesize = 24)
    hm = CM.heatmap!(ax1, helical_result.hu[:, yc, :]; colormap = :grays, colorrange = (-500, 100))
    CM.hidedecorations!(ax1)
    ax2 = CM.Axis(fig[1, 2]; title = "Volume axial (2 × 16 cm) — coronal", titlesize = 24)
    CM.heatmap!(ax2, sns_result.hu[:, yc, :]; colormap = :grays, colorrange = (-500, 100))
    CM.hidedecorations!(ax2)
    CM.Colorbar(fig[1, 3], hm; label = "HU", labelsize = 22, ticklabelsize = 16)
    CM.save(joinpath(@__DIR__, "..", "assets", "helical_vs_stepshoot_coronal.png"), fig)
    fig
end

# ╔═╡ 11000009-0000-4000-8000-000000000001
md"""
## Notes and scope

- **`:dd_fast` needed zero changes for helical.** The projectors consume
  per-view geometry arrays (the same representation as ASTRA's `cone_vec`
  and CatSim's internal trajectory); the helix lives entirely in those
  arrays.
- **Helical reconstruction is rebinned WFBP** (Stierstorfer *et al.*, Phys
  Med Biol 49:2209, 2004): row-wise fan→parallel rebinning, parallel ramp
  filtering, and aperture-weighted (cos², plateau ``Q = 0.7``) wedge
  backprojection with per-half-turn redundancy normalisation.  This is the
  production spiral-CT algorithm family (Siemens WFBP; UCLA FreeCT), and it
  is dispatched automatically by `reconstruct!`/`fdk_reconstruct` whenever
  `is_helical(geom)`.
- **Both scans got identical corrections** (notebook 01 stack) and matched
  exposure (16 × 20 mm vs 2 × 160 mm beam-time product, same mA).
- Hybrid IR works on helical unchanged — matched forward/backprojection are
  geometry-general, subsets are angular-interleaved, and the HIR FDK
  initialisation routes through WFBP.
- **Honest limits**: WFBP-family recon is clinically credible up to ~128
  detector rows and pitch ≲ 1.5.  Windmill artifacts around sharp
  ``z``-edges at high pitch are physics (longitudinal Nyquist), not a bug.
- Wall-clock on this machine: helical (5760 views, full EICT physics +
  corrections) ≈ $(round(helical_result.t; digits = 1)) s; volume axial
  (2 stations × 360 views) ≈ $(round(sns_result.t; digits = 1)) s.
"""

# ╔═╡ Cell order:
# ╟─11000002-0000-4000-8000-000000000001
# ╠═11000001-0000-4000-8000-000000000001
# ╠═11000001-0000-4000-8000-000000000004
# ╠═11000001-0000-4000-8000-000000000005
# ╠═11000001-0000-4000-8000-000000000002
# ╠═11000001-0000-4000-8000-000000000003
# ╠═11000001-0000-4000-8000-000000000006
# ╟─11000003-0000-4000-8000-000000000001
# ╠═11000003-0000-4000-8000-000000000002
# ╠═11000003-0000-4000-8000-000000000003
# ╠═11000003-0000-4000-8000-000000000004
# ╟─11000004-0000-4000-8000-000000000001
# ╠═11000004-0000-4000-8000-000000000002
# ╠═11000004-0000-4000-8000-000000000003
# ╟─11000005-0000-4000-8000-000000000001
# ╠═11000005-0000-4000-8000-000000000002
# ╟─11000006-0000-4000-8000-000000000001
# ╠═11000006-0000-4000-8000-000000000002
# ╠═11000006-0000-4000-8000-000000000003
# ╟─11000007-0000-4000-8000-000000000001
# ╟─11000007-0000-4000-8000-000000000002
# ╟─11000007-0000-4000-8000-000000000003
# ╟─11000008-0000-4000-8000-000000000001
# ╟─11000008-0000-4000-8000-000000000002
# ╟─11000008-0000-4000-8000-000000000003
# ╟─11000009-0000-4000-8000-000000000001
