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
# ╠═╡ show_logs = false
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
# Use CairoMakie for faithful build-time rendering. Snapshot can still isolate
# and compile independent browser-safe islands without hoisting this import.
import CairoMakie as Mke

# ╔═╡ 11000001-0000-4000-8000-000000000006
begin
    import GPUSelect
    AT = GPUSelect.Storage()     # the backend array type, directly: MtlArray / CuArray / ROCArray
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end

# ╔═╡ 11000002-0000-4000-8000-000000000001
Markdown.parse("""
# 11 · Helical Scanning · Narrow Collimation, Long Coverage

**One new kwarg — `pitch` — turns any protocol into a spiral scan.**

Helical (spiral) CT solves a geometry problem no detector can: covering a
long ``z`` range with a **narrow** beam. Before spiral CT, long coverage
meant **step-and-shoot**: scan a slab, move the table, scan the next slab.
Each station is a clean axial scan — but every station has cone-beam edges,
and the stitched volume can introduce a discontinuity at a station boundary. A helical
scan instead sweeps the collimator *past* every slice continuously: each
``z`` position is, at some moment of the spiral, at the centre of the beam.

This notebook covers a **30 cm** reconstructed z-slab both ways, on a wide-cone
256 × 0.625 mm (16 cm) volume scanner, with the tube current chosen so that the two scans
deliver the same total beam-width × current × time:

|                | collimation | rotations | table          |
|:---------------|:-----------|:----------|:---------------|
| volume axial   | 160 mm (full detector) | 3 × axial | steps 10 cm between stations |
| **helical**    | **20 mm (1/8th!)**     | 16 turns  | glides at pitch 1.0 |

The forward projection is the anti-aliased `:dd_fast` projector — helical
costs it nothing (the projectors consume per-view source/detector arrays; a
helix is just a z-ramp in those arrays). Both scans integrate each view over the arc the gantry
turns while the detector reads it (`view_samples = 5`, notebook 01); on the helix each sub-view
is also advanced along ``z`` by its share of the table feed. Helical reconstruction is
**rebinned WFBP** (Stierstorfer *et al.* 2004 — the production spiral
algorithm family), dispatched automatically whenever the geometry is
helical. Both pipelines use the reconstruction chain of notebook 01:
detected-spectrum water BHC → reconstruction → HU, with quantum and electronic noise already in
the simulated counts. Each `simulate!` also reports its dose, so the two acquisitions can be
compared in CTDIvol and DLP as well as in image quality.

**Backend detected:** $(GPU_BACKEND.name)
""")

# ╔═╡ 11000002-0000-4000-8000-000000000002
md"""
## Notebook Setup

The shared docs environment, the device from `GPUSelect`, and a table of contents. Nothing is
simulated until section 4.
"""

# ╔═╡ 11000002-0000-4000-8000-000000000003
PlutoUI.TableOfContents()

# ╔═╡ 11000003-0000-4000-8000-000000000001
md"""
## 1. A phantom that changes along ``z`` — with no strong hardeners

Two design rules. First, a uniform cylinder would look identical at every
slice — useless for judging long-``z`` fidelity — so every slice must be
different. Second, **no high-Z inserts**: beam hardening is its own topic
(notebook 10); here it would only confound the geometry comparison.

- a **24 cm water body**, 36 cm long (coarse 2.3 mm in-plane / 2 mm z voxels);
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
collimation (32 of the scanner's 256 × 0.625 mm rows), pitch 1.0, and 16
rotations, the table travels

```math
\text{travel} = \text{pitch} \times \text{collimation} \times n_\text{rot}
             = 1.0 \times 20\,\text{mm} \times 16 = 32\,\text{cm}.
```

Everything else (`Phantom`, the scanner, `SimOptions`, `ReconOptions`, `simulate!`, the
corrections, the reconstruction call) is the same as in an axial workflow.
"""

# ╔═╡ 11000004-0000-4000-8000-000000000002
scanner = BS.EICTScanner(
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
        collimation_mm = 20.0,       # NARROW: one eighth of the physical detector
        pitch = 1.0,                 # ← the one helical kwarg
        n_rotations = 16,            # 16 × 2 cm = 32 cm table travel
    )
    protocol_axial = BS.CTProtocol(
        # Three 160 mm stations match 16 × 20 mm helical rotations when
        # weighted by tube current: 3 × 160 × 133⅓ = 16 × 20 × 200.
        kVp = 120.0, mA = 400.0 / 3.0, views = 360,
        rotation_time = 0.5,
        collimation_mm = 160.0,      # WIDE: the full 16 cm detector, volume mode
    )
    # use_heel_effect = false: the anode heel is a per-ROW spectral gradient.
    # An axial slice keeps its rows, but a helical scan sweeps every row past
    # every voxel, which a real scanner handles with a per-row water
    # calibration. `calibrate_bhc_water` is per column, so the heel is switched
    # off here and the comparison isolates the geometry.
    # view_samples = 5: each view integrates over the arc the gantry turns while the detector
    # reads it (notebook 01); on a helix every sub-view also advances by its share of the feed.
    sim_opts = BS.SimOptions(seed = 42, projector = :dd_fast, view_samples = 5,
        use_heel_effect = false)
    recon_opts = BS.ReconOptions(matrix_size = (160, 160, 150), fov_cm = 30.0, z_cm = 30.0)
    # A 16 cm *physical* detector cannot reconstruct a full 16 cm axial
    # cylinder at 30 cm transverse FOV: cone magnification requires 356 rows.
    # Ten centimetres requires 222 guarded rows and fits the 256-row scanner.
    recon_opts_station = BS.ReconOptions(matrix_size = (160, 160, 50), fov_cm = 30.0, z_cm = 10.0)
end;

# ╔═╡ 11000005-0000-4000-8000-000000000001
md"""
## 3. The reconstruction chain

One BHC model per protocol (calibrated on that acquisition's geometry), then per
reconstruction: sinogram-domain water BHC → reconstruction → HU with the BHC's own
``\mu_\text{water}``. `reconstruct!` recognises a helical geometry and runs rebinned WFBP; an
axial one runs FDK.
"""

# ╔═╡ 11000005-0000-4000-8000-000000000002
"""
    corrected_recon(sino_gpu, geom, matrix_size, bhc) -> Array{Float32,3} (HU)

Water BHC → reconstruction → HU for one sinogram. A helical `geom` dispatches to WFBP inside
`reconstruct!`.
"""
function corrected_recon(sino_gpu, geom, matrix_size, bhc)
    sino = BS.apply_bhc_water(sino_gpu, bhc)
    ws_fdk = BS.create_fdk_recon_workspace(sino, geom, matrix_size)
    recon_μ = BS.reconstruct!(ws_fdk, sino, geom)
    return Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc.μ_water_ref))
end;

# ╔═╡ 11000006-0000-4000-8000-000000000001
md"""
## 4. Run both acquisitions

**Helical**: one `simulate!`, one reconstruction. **Volume axial (step-and-shoot)**: three
10 cm reconstruction stations. For each, the table (here, the phantom window) moves so that the
station is centred on the isocentre; the three slabs are then stitched at ``z = ±5`` cm. The
16 cm beams of neighbouring stations overlap by 6 cm, so each station's dose is reported with a
10 cm table increment.
"""

# ╔═╡ 11000006-0000-4000-8000-000000000002
helical_result = let
    ws = BS.create_workspace(scanner, protocol_helical, sim_opts, recon_opts, phantom)
    t = @elapsed (sim = BS.simulate!(ws, phantom, protocol_helical, sim_opts))
    bhc = BS.calibrate_bhc_water(sim_opts, protocol_helical; scanner, geom = ws.geom)
    t += @elapsed (hu = corrected_recon(ws.sinogram, ws.geom, recon_opts.matrix_size, bhc))
    n_views = ws.geom.n_angles
    ws = nothing
    GC.gc()
    (hu = hu, t = t, dose = sim.dose, n_views = n_views)
end;

# ╔═╡ 11000006-0000-4000-8000-000000000003
sns_result = let
    station_zs = [-10.0, 0.0, 10.0]             # 3 stations × 10 cm, back to back
    n_slab = 50                                 # 10 cm at 2 mm recon slices
    hu = zeros(Float32, 160, 160, 150)
    t_total = 0.0
    bhc_ax = nothing
    doses = BS.DoseReport[]
    for (s, z0) in enumerate(station_zs)
        # move the "table": phantom window (±10 cm) centred on this station
        k0 = round(Int, 90 + z0 / phantom_data.voxz)
        requested = (k0 - 49):(k0 + 50)         # 100 slices = 20 cm (covers the cone)
        available = max(first(requested), 1):min(last(requested), phantom_data.nz)
        # End stations extend beyond the finite source phantom. Keep their
        # isocentres fixed and pad the missing exterior with material 0 (air),
        # rather than clamping the window and shifting anatomy toward isocentre.
        station_mask = zeros(eltype(phantom_data.mask), size(phantom_data.mask, 1),
            size(phantom_data.mask, 2), length(requested))
        destination = ((first(available) - first(requested) + 1):
            (last(available) - first(requested) + 1))
        station_mask[:, :, destination] .= phantom_data.mask[:, :, available]
        ph_st = BS.Phantom(
            to_gpu(station_mask),
            phantom_materials,
            (phantom_data.vox, phantom_data.vox, phantom_data.voxz),
        )
        ws = BS.create_workspace(scanner, protocol_axial, sim_opts, recon_opts_station, ph_st)
        t_total += @elapsed (sim = BS.simulate!(ws, ph_st, protocol_axial, sim_opts;
            dose_kwargs = (; table_increment_mm = 100.0)))
        push!(doses, sim.dose)
        if bhc_ax === nothing
            bhc_ax = BS.calibrate_bhc_water(sim_opts, protocol_axial; scanner, geom = ws.geom)
        end
        t_total += @elapsed (hu_st = corrected_recon(
            ws.sinogram, ws.geom, recon_opts_station.matrix_size, bhc_ax))
        ws = nothing
        GC.gc()
        # station slab → global z index: each station covers z0 ± 5 cm.
        k_lo = round(Int, (z0 - 5.0 + 15.0) / 0.2) + 1
        valid = max(k_lo, 1):min(k_lo + n_slab - 1, 150)
        hu[:, :, valid] .= hu_st[:, :, (first(valid) - k_lo + 1):(last(valid) - k_lo + 1)]
    end
    (hu = hu, t = t_total, doses = doses)
end;

# ╔═╡ 11000006-0000-4000-8000-000000000004
let
    h = helical_result.dose
    a = sns_result.doses
    dlp_ax = sum(d.dlp_mGy_cm for d in a)
    Markdown.parse("""
    **Dose of each acquisition** (CTDI from a Monte Carlo of the simulated beam in the 32 cm body phantom)

    | acquisition | N·T (mm) | mAs / rotation | rotations | CTDIvol (mGy) | DLP (mGy·cm) |
    |:--|--:|--:|--:|--:|--:|
    | helical, pitch $(h.pitch) | $(h.nominal_collimation_mm) | $(round(h.mAs_per_rotation; digits = 1)) | $(Int(h.n_rotations)) | $(round(h.ctdi_vol_mGy; digits = 2)) | $(round(h.dlp_mGy_cm; digits = 1)) |
    | volume axial, 3 stations, 10 cm apart | $(a[1].nominal_collimation_mm) | $(round(a[1].mAs_per_rotation; digits = 1)) | 3 × 1 | $(round(a[1].ctdi_vol_mGy; digits = 2)) | $(round(dlp_ax; digits = 1)) |

    The two DLPs are $(round(h.dlp_mGy_cm; digits = 1)) and $(round(dlp_ax; digits = 1)) mGy·cm:
    matching beam width × current × time matches the total exposure. CTDIvol differs because that
    exposure is spread over different lengths: $(round(h.scan_length_cm; digits = 1)) cm of helical
    travel against three $(round(a[1].table_increment_mm / 10; digits = 1)) cm table increments for the
    axial stations, whose overlapping $(a[1].nominal_collimation_mm) mm beams$(a[1].wide_beam_reference_mm === nothing ? "" : " are dosed with the IEC wide-beam rule (reference $(a[1].wide_beam_reference_mm) mm)").
    """)
end

# ╔═╡ 1100000b-0000-4000-8000-000000000001
md"""
### Plan a helical scan

The dose above is the pitch-1.0 scan. At a fixed tube current and rotation time, CTDIvol scales as
1 / pitch, and the table feed per rotation is pitch × collimation, which sets how many rotations
(and seconds) a range takes. Move the slider: the numbers below are computed live in your browser
from this scan's measured CTDIvol and its 20 mm collimation.
"""

# ╔═╡ 1100000b-0000-4000-8000-000000000002
# this scan's scalars — CTDIvol scaled to pitch 1, the collimation (mm), the rotation time (s) —
# the inputs of the calculator below, which runs live in the browser
helical_inputs = (
    Float64(helical_result.dose.ctdi_vol_mGy * helical_result.dose.pitch),
    Float64(helical_result.dose.nominal_collimation_mm),
    Float64(protocol_helical.rotation_time),
);

# ╔═╡ 1100000b-0000-4000-8000-000000000003
@bind pitch_x100 PlutoUI.Slider(50:5:150; default = 100, show_value = true)

# ╔═╡ 1100000b-0000-4000-8000-000000000004
# pitch, table feed (mm), CTDIvol (mGy), rotations and seconds for a 30 cm range
helical_plan = let p = pitch_x100 / 100
    feed = helical_inputs[2] * p
    rotations = 300.0 / feed
    (p, round(feed; digits = 1), round(helical_inputs[1] / p; digits = 2),
     round(rotations; digits = 1), round(rotations * helical_inputs[3]; digits = 1))
end;

# ╔═╡ 1100000b-0000-4000-8000-000000000005
md"""
**Pitch $(helical_plan[1]):** the table moves $(helical_plan[2]) mm per rotation, CTDIvol is $(helical_plan[3]) mGy, and a 30 cm range takes $(helical_plan[4]) rotations, $(helical_plan[5]) s.
"""

# ╔═╡ 11000007-0000-4000-8000-000000000001
md"""
## 5. Slide through ``z``: phantom truth vs both scans

Top row: the phantom ground truth as a labelled categorical map. Bottom
row: **helical** (20 mm collimation) and **volume axial** (the full 16 cm
detector, 8× wider). Watch the rod rotate and the cone shrink — and watch
what the wide cone does away from each station centre. The axial station
boundaries are marked at ``z = ±5`` cm for inspection.
"""

# ╔═╡ 11000007-0000-4000-8000-000000000002
@bind z_idx PlutoUI.Slider(1:150; default = 75, show_value = true)

# ╔═╡ 11000007-0000-4000-8000-000000000003
let
    z_cm = -15.0 + (z_idx - 0.5) * (30.0 / 150)
    kp = clamp(round(Int, (z_cm + 18.0) / phantom_data.voxz + 0.5), 1, phantom_data.nz)

    fig = Mke.Figure(size = (1100, 1150))

    # Super-title.  tellwidth = false so the spanning label cannot stretch the
    # columns (Makie FAQ: elements with tellwidth = true resize their column).
    Mke.Label(fig[0, 1:3], "z = $(round(z_cm; digits = 1)) cm  (slice $(z_idx)/150)";
        fontsize = 32, font = :bold, tellwidth = false)

    # ── Row 1: phantom ground truth, CATEGORICAL colours + labels ──
    cat_colors = Mke.cgrad([:gray10, :steelblue, :darkorange, :seagreen]; categorical = true)
    ax1 = Mke.Axis(fig[1, 1:2]; title = "Phantom ground truth", titlesize = 24,
        aspect = Mke.DataAspect())
    hm1 = Mke.heatmap!(ax1, Float32.(phantom_data.mask[:, :, kp]);
        colormap = cat_colors, colorrange = (-0.5, 3.5))
    Mke.hidedecorations!(ax1)
    Mke.Colorbar(fig[1, 3], hm1;
        ticks = (0:3, ["air", "water body", "lung rod", "adipose cone"]),
        ticklabelsize = 18, tellheight = false)

    # ── Row 2: the two reconstructions (HU grayscale) ──
    ax2 = Mke.Axis(fig[2, 1]; title = "Helical · 20 mm collim, pitch 1.0", titlesize = 24,
        aspect = Mke.DataAspect())
    hm2 = Mke.heatmap!(ax2, helical_result.hu[:, :, z_idx]; colormap = :grays, colorrange = (-500, 100))
    Mke.hidedecorations!(ax2)

    ax3 = Mke.Axis(fig[2, 2]; title = "Volume axial · 3 × 10 cm", titlesize = 24,
        aspect = Mke.DataAspect())
    Mke.heatmap!(ax3, sns_result.hu[:, :, z_idx]; colormap = :grays, colorrange = (-500, 100))
    Mke.hidedecorations!(ax3)

    Mke.Colorbar(fig[2, 3], hm2; label = "HU", labelsize = 22, ticklabelsize = 16)
    fig
end

# ╔═╡ 11000008-0000-4000-8000-000000000001
md"""
## Results

### 1. Water flatness and the coronal view

Mean HU in a fixed water ROI (clear of the rod and the cone), slice by slice, with the axial
station boundaries marked, then coronal reformats of both volumes. The table under the plot
reduces the profiles to numbers.
"""

# ╔═╡ 11000008-0000-4000-8000-000000000002
z_profile_fig = let
    xr, yr = 25:36, 73:88            # 9.4 cm left of centre: clear of rod ring & cone
    zs = [-15.0 + (k - 0.5) * 0.2 for k in 1:150]
    prof_hel = [mean(helical_result.hu[xr, yr, k]) for k in 1:150]
    prof_sns = [mean(sns_result.hu[xr, yr, k]) for k in 1:150]

    fig = Mke.Figure(size = (1400, 600))
    ax = Mke.Axis(fig[1, 1];
        title = "Water HU vs z", titlesize = 32,
        subtitle = "helical (20 mm collim) vs volume axial (160 mm collim), matched beam-width–current product",
        subtitlesize = 24,
        xlabel = "z (cm)", ylabel = "mean HU (water ROI)",
        xlabelsize = 22, ylabelsize = 22, xticklabelsize = 18, yticklabelsize = 18)
    Mke.lines!(ax, zs, prof_hel; linewidth = 4, label = "helical · 20 mm, pitch 1.0 × 16 rot")
    Mke.lines!(ax, zs, prof_sns; linewidth = 4, label = "volume axial · 3 × 10 cm stations")
    Mke.hlines!(ax, [0.0]; color = :gray, linestyle = :dash, linewidth = 2)
    Mke.vlines!(ax, [-5.0, 5.0]; color = (:orange, 0.5), linestyle = :dot, linewidth = 3)
    Mke.ylims!(ax, -100, 100)
    Mke.axislegend(ax; labelsize = 18, position = :cb)
    fig
end

# ╔═╡ 11000008-0000-4000-8000-000000000004
let
    xr, yr = 25:36, 73:88                        # the water ROI of the profile plot
    zs = [-15.0 + (k - 0.5) * 0.2 for k in 1:150]
    prof(v) = [mean(v[xr, yr, k]) for k in 1:150]
    noise(v) = mean(std(v[xr, yr, k]) for k in 1:150)
    step_at(p, z) = (k = searchsortedfirst(zs, z); p[k] - p[k - 1])
    rows = String[]
    for (name, r) in (("helical", helical_result), ("volume axial", sns_result))
        p = prof(r.hu)
        push!(rows, "| $(name) | $(round(mean(p); digits = 1)) | $(round(minimum(p); digits = 1)) … $(round(maximum(p); digits = 1)) | " *
                    "$(round(std(p); digits = 1)) | $(round(step_at(p, -5.0); digits = 1)) / $(round(step_at(p, 5.0); digits = 1)) | $(round(noise(r.hu); digits = 1)) |")
    end
    Markdown.parse("""
    | acquisition | mean water HU over z | range over z | σ over z | step at z = −5 / +5 cm | pixel noise σ (HU) |
    |:--|--:|--:|--:|--:|--:|
    $(join(rows, "\n"))

    "Step" is the change in the ROI mean between the two slices either side of a station boundary;
    "pixel noise" is the within-ROI standard deviation averaged over the slices.
    """)
end

# ╔═╡ 11000008-0000-4000-8000-000000000003
coronal_fig = let
    yc = 80
    fig = Mke.Figure(size = (1400, 640))
    ax1 = Mke.Axis(fig[1, 1]; title = "Helical — coronal", titlesize = 24)
    hm = Mke.heatmap!(ax1, helical_result.hu[:, yc, :]; colormap = :grays, colorrange = (-500, 100))
    Mke.hidedecorations!(ax1)
    ax2 = Mke.Axis(fig[1, 2]; title = "Volume axial (3 × 10 cm) — coronal", titlesize = 24)
    Mke.heatmap!(ax2, sns_result.hu[:, yc, :]; colormap = :grays, colorrange = (-500, 100))
    Mke.hidedecorations!(ax2)
    Mke.Colorbar(fig[1, 3], hm; label = "HU", labelsize = 22, ticklabelsize = 16)
    Mke.save(joinpath(@__DIR__, "..", "assets", "helical_vs_stepshoot_coronal.png"), fig)
    fig
end

# ╔═╡ 11000009-0000-4000-8000-000000000001
Markdown.parse("""
## Verification and Scope

- **`:dd_fast` needed zero changes for helical.** The projectors consume
  per-view geometry arrays (the same representation as ASTRA's `cone_vec`
  and CatSim's internal trajectory); the helix lives entirely in those
  arrays, and so do the sub-views of the view integration, each rotated and advanced along
  ``z`` with the table.
- **Helical reconstruction is rebinned WFBP** (Stierstorfer *et al.*, Phys
  Med Biol 49:2209, 2004): row-wise fan→parallel rebinning, parallel ramp
  filtering, and aperture-weighted (cos², plateau ``Q = 0.7``) wedge
  backprojection with per-half-turn redundancy normalisation.  This is the
  production spiral-CT algorithm family (Siemens WFBP; UCLA FreeCT), and it
  is dispatched automatically by `reconstruct!`/`fdk_reconstruct` whenever
  `is_helical(geom)`.
- **Both scans got identical corrections.** Exposure is matched by beam width × current:
  16 × 20 mm × 200 mA versus 3 × 160 mm × 133⅓ mA at equal rotation time. That is an
  acquisition-integral match, not equal CTDIvol or equal noise per slice; the dose table in
  section 4 gives both scans' CTDIvol and DLP.
- Hybrid IR works on helical unchanged — matched forward/backprojection are
  geometry-general, subsets are angular-interleaved, and the HIR FDK
  initialisation routes through WFBP.
- Windmill artifacts around sharp ``z`` edges at high pitch are physics (longitudinal
  sampling), not a defect of the reconstruction.
- Wall-clock (this page was rendered on an NVIDIA RTX PRO 6000, CUDA): helical
  ($(helical_result.n_views) views, simulation + corrections + WFBP)
  $(round(helical_result.t; digits = 1)) s; volume axial (3 stations × $(protocol_axial.views) views)
  $(round(sns_result.t; digits = 1)) s. Both include first-call compilation.
""")

# ╔═╡ 1100000a-0000-4000-8000-000000000001
md"""
## Summary

- `pitch` and `n_rotations` are the only additions needed to turn the axial
  acquisition geometry into a centered helical trajectory.
- `:dd_fast` consumes the per-view geometry arrays unchanged and remains the
  default forward projector.
- Helical geometries dispatch automatically to rebinned WFBP; axial stations
  retain the ordinary FDK path.
- The slice slider, the water-HU profile and the coronal reformats inspect longitudinal coverage,
  station boundaries and the z-dependent anatomy; the dose table shows what each acquisition
  costs.
"""

# ╔═╡ Cell order:
# ╟─11000002-0000-4000-8000-000000000001
# ╟─11000002-0000-4000-8000-000000000002
# ╠═11000001-0000-4000-8000-000000000001
# ╠═11000001-0000-4000-8000-000000000004
# ╠═11000001-0000-4000-8000-000000000005
# ╠═11000001-0000-4000-8000-000000000002
# ╠═11000001-0000-4000-8000-000000000003
# ╠═11000001-0000-4000-8000-000000000006
# ╠═11000002-0000-4000-8000-000000000003
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
# ╟─11000006-0000-4000-8000-000000000004
# ╟─1100000b-0000-4000-8000-000000000001
# ╟─1100000b-0000-4000-8000-000000000002
# ╠═1100000b-0000-4000-8000-000000000003
# ╟─1100000b-0000-4000-8000-000000000004
# ╟─1100000b-0000-4000-8000-000000000005
# ╟─11000007-0000-4000-8000-000000000001
# ╟─11000007-0000-4000-8000-000000000002
# ╟─11000007-0000-4000-8000-000000000003
# ╟─11000008-0000-4000-8000-000000000001
# ╟─11000008-0000-4000-8000-000000000002
# ╟─11000008-0000-4000-8000-000000000004
# ╟─11000008-0000-4000-8000-000000000003
# ╟─11000009-0000-4000-8000-000000000001
# ╟─1100000a-0000-4000-8000-000000000001
