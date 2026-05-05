### A Pluto.jl notebook ###
# v0.19.0

# ╔═╡ 030b0001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 030b0001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 030b0001-0000-4000-8000-000000000003
using Statistics: mean, std

# ╔═╡ 030b0001-0000-4000-8000-000000000010
md"""
# 03b · Dual-keV Monoenergetic on Gammex 472

**The physical ceiling for what `nb03`'s polychromatic dual-kVp + VMI
pipeline is trying to approximate** — and a demonstration that with
monoenergetic input the entire post-FBP processing chain collapses to
**just `to_hounsfield`**.

Mirrors `03_dual_kvp_vmi.jl` for setup (phantom, scanner, protocol flux
budget, sim opts, recon opts) but the source spectra are
**monoenergetic** at 70 keV (low) and 150 keV (high) instead of
polychromatic 80 / 140 kVp.

```
mono FBP per energy (μ-domain)
   → HU = 1000·(μ − μ_water_NIST(E)) / μ_water_NIST(E)
   → 2-panel display + per-rod regression vs XrayAttenuation theory
```

Because the inputs are intrinsically monoenergetic:

| nb03 stage                                 | nb03b |
|--------------------------------------------|-------|
| Sino-domain BHC                            | dropped — no beam hardening on mono input |
| RSKR-2ch joint denoise                     | dropped — no anti-correlated spectral noise to exploit |
| Radial cupping correction                  | dropped — no spectral cup |
| Measured μ_water from center ROI           | dropped — `BS.compute_μ_at_energy(water, E)` is exact |
| 2-basis Ding decomp + VMI synth + Mono+    | dropped — recon @ E_mono IS the E_mono image |

What's left: forward-project, FBP, divide by NIST μ_water at the chosen
energy.  That's the entire pipeline.

Source-spectrum injection happens via
[`BS.create_eict_workspace(...; spectrum_override = ([E_mono], [1.0]))`](@ref).
"""

# ╔═╡ 030b0001-0000-4000-8000-000000000020
md"""
## Setup
"""

# ╔═╡ 030b0001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 030b0001-0000-4000-8000-000000000031
import CairoMakie as CM

# ╔═╡ 030b0001-0000-4000-8000-000000000040
begin
    GPU_BACKEND = let
        candidates = [
            (:Metal,  "dde4c033-4e86-420c-a63e-0dd931031962", :MtlArray),
            (:CUDA,   "052768ef-5323-5732-b1bb-66c8b64840ba", :CuArray),
            (:AMDGPU, "21141c5a-9bdb-4563-92ae-f87d6854732e", :ROCArray),
        ]
        detected = (name = "CPU", to_gpu = identity)
        for (pkg, uuid, ctor) in candidates
            pkg_id = Base.PkgId(Base.UUID(uuid), String(pkg))
            Base.locate_package(pkg_id) === nothing && continue
            try
                m = Base.require(pkg_id)
                if Base.invokelatest(getfield(m, :functional))
                    detected = (name = string(pkg), to_gpu = getfield(m, ctor))
                    break
                end
            catch
            end
        end
        detected
    end

    to_gpu(x) = GPU_BACKEND.to_gpu(x)
end

# ╔═╡ 030b0001-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 030b0002-0000-4000-8000-000000000001
md"""
## 1. `Phantom` — Gammex Model 472

Identical to `nb03 §1`.
"""

# ╔═╡ 030b0002-0000-4000-8000-000000000010
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,
    n_slices = 16,
    fov_cm = 35.0,
    z_cm = 1.0,
);

# ╔═╡ 030b0002-0000-4000-8000-000000000020
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 030b0003-0000-4000-8000-000000000001
md"""
## 2. `Scanner` — GE Revolution Apex Elite

Identical to `nb03 §2`.  Bowtie + heel + detector response still
evaluate at the override's mono energy, so the non-spectral physics
stays faithful — there's just no photon-energy variation across the
spectrum to make BHC / Ding decomp / VMI relevant.
"""

# ╔═╡ 030b0003-0000-4000-8000-000000000010
scanner = BS.Scanner(
    source_to_isocenter = 625.6,
    source_to_detector  = 1100.0,

    detector_rows     = 256,
    detector_cols     = 834,
    detector_row_size = 0.625,
    detector_col_size = 0.6,
    detector_shape    = BS.CURVED_DETECTOR,

    focal_spot_width  = 1.0,
    focal_spot_length = 1.0,
    target_angle      = 10.0,

    flat_filter_material  = :aluminum,
    flat_filter_thickness = 2.5,
    bowtie_filter         = :ge_revolution_large,

    detector_material = :lumex,
    detector_depth    = 3.0,
    fill_factor_row   = 0.9,
    fill_factor_col   = 0.9,

    electronic_noise = 0,
    detection_gain   = 10.0,
);

# ╔═╡ 030b0004-0000-4000-8000-000000000001
md"""
## 3. Mono energies + dual-acquisition `CTProtocol`s

Two true monoenergetic beams: **70 keV (low)** and **150 keV (high)**.
The `kVp` field is just a flux-calibration proxy now — it drives
`compute_detector_I0` so total photon flux matches nb03's, but the
spectrum itself comes entirely from `spectrum_override` (§5).

Same `mA × duty_cycle` effective tube current as nb03 (rapid-kVp
switching duty-cycle math) — even though there's no actual
kVp-switching here, keeping the flux matched to nb03 makes the
noise/SNR comparison meaningful.
"""

# ╔═╡ 030b0004-0000-4000-8000-000000000010
de_mono_energies = (low = 70.0, high = 150.0);

# ╔═╡ 030b0004-0000-4000-8000-000000000020
protocol_low = BS.CTProtocol(
    kVp = 80,                    # flux-calibration proxy only
    mA  = 407 * 0.65,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 030b0004-0000-4000-8000-000000000030
protocol_high = BS.CTProtocol(
    kVp = 140,                   # flux-calibration proxy only
    mA  = 405 * 0.35,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 030b0005-0000-4000-8000-000000000001
md"""
## 4. `SimOptions` and `ReconOptions`

Identical to nb03 §4.
"""

# ╔═╡ 030b0005-0000-4000-8000-000000000010
sim_opts = BS.SimOptions(
    fidelity = :eict,
    seed     = 1234,
);

# ╔═╡ 030b0005-0000-4000-8000-000000000020
recon_opts = let
    slice_thickness_mm = 0.625
    n_recon_slices = round(Int, 5.0 / slice_thickness_mm)
    BS.ReconOptions(
        algorithm   = :fdk,
        matrix_size = (512, 512, n_recon_slices),
        fov_cm      = 35.0,
        z_cm        = 0.5,
        filter      = :standard,
    )
end;

# ╔═╡ 030b0006-0000-4000-8000-000000000001
md"""
## 5. Forward project — 70 keV mono

`spectrum_override = ([70.0], [1.0])` bypasses the IPEM polychromatic
lookup and injects a delta-function spectrum.  Bowtie + heel + detector
response still get evaluated at the override's energy grid, so the
non-spectral physics (per-pixel attenuation, scatter, noise) all run
faithfully.
"""

# ╔═╡ 030b0006-0000-4000-8000-000000000010
sim_low = let
    @info "Simulating: 70 keV mono / $(round(protocol_low.mA, digits = 1)) mA-eff (DE low)…"
    ws = BS.create_eict_workspace(
        scanner, protocol_low, sim_opts, recon_opts, phantom;
        spectrum_override = ([de_mono_energies.low], [1.0]),
    )
    BS.simulate!(ws, phantom, scanner, protocol_low, sim_opts, recon_opts)

    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom)

    ws = nothing
    GC.gc(true)
    result
end;

# ╔═╡ 030b0007-0000-4000-8000-000000000001
md"""
## 6. Forward project — 150 keV mono

Same pattern; only the override and the protocol change.
"""

# ╔═╡ 030b0007-0000-4000-8000-000000000010
sim_high = let
    @info "Simulating: 150 keV mono / $(round(protocol_high.mA, digits = 1)) mA-eff (DE high)…"
    ws = BS.create_eict_workspace(
        scanner, protocol_high, sim_opts, recon_opts, phantom;
        spectrum_override = ([de_mono_energies.high], [1.0]),
    )
    BS.simulate!(ws, phantom, scanner, protocol_high, sim_opts, recon_opts)

    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom)

    ws = nothing
    GC.gc(true)
    result
end;

# ╔═╡ 030b0008-0000-4000-8000-000000000001
md"""
## 7. FBP per energy → μ-domain pair

Direct FDK with the GE-tuned apodization filter.  No BHC, no RSKR, no
cupping correction — none of those are needed when the input is
already monoenergetic.
"""

# ╔═╡ 030b0008-0000-4000-8000-000000000010
de_lohi_μ = let
    matrix_size = recon_opts.matrix_size

    function _fbp_to_μ(sino_cpu, geom)
        sino_gpu = to_gpu(Float32.(sino_cpu))
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, geom, matrix_size;
            filter = BS.CustomFilter(
                (0.0, 0.25, 0.5, 0.75, 1.0),
                (1.0, 0.95, 0.85, 0.65, 0.4),
            ),
        )
        recon_μ = Array(BS.reconstruct!(ws, sino_gpu, geom, matrix_size))
        ws = nothing; sino_gpu = nothing
        GC.gc(true)
        return Float32.(recon_μ)
    end

    vol_low_μ  = _fbp_to_μ(sim_low.sino,  sim_low.geom)
    vol_high_μ = _fbp_to_μ(sim_high.sino, sim_high.geom)

    (vol_low_μ = vol_low_μ, vol_high_μ = vol_high_μ, geom = sim_low.geom)
end;

# ╔═╡ 030b0009-0000-4000-8000-000000000001
md"""
## 8. NIST `μ_water` lookup — exact at the mono energy

For a truly monoenergetic beam, μ_water is just whatever
`XrayAttenuation` returns at that energy — no center-ROI measurement,
no calibration tricks.  This is the single biggest simplification the
mono pipeline gets vs nb03's polychromatic flow.
"""

# ╔═╡ 030b0009-0000-4000-8000-000000000010
keV_μ_water = Dict(
    de_mono_energies.low  => BS.compute_μ_at_energy(BS.XA.Materials.water, de_mono_energies.low),
    de_mono_energies.high => BS.compute_μ_at_energy(BS.XA.Materials.water, de_mono_energies.high),
);

# ╔═╡ 030b0009-0000-4000-8000-000000000020
md"""
**μ_water** (NIST analytical):

* $(de_mono_energies.low) keV: $(round(keV_μ_water[de_mono_energies.low], digits = 5)) cm⁻¹
* $(de_mono_energies.high) keV: $(round(keV_μ_water[de_mono_energies.high], digits = 5)) cm⁻¹
"""

# ╔═╡ 030b000a-0000-4000-8000-000000000001
md"""
## 9. HU conversion

`BS.to_hounsfield` per energy with the NIST μ_water from §8.  This is
the entire post-FBP pipeline.
"""

# ╔═╡ 030b000a-0000-4000-8000-000000000010
de_lohi_HU = let
    vol_low_HU  = Float32.(BS.to_hounsfield(de_lohi_μ.vol_low_μ;  μ_water = keV_μ_water[de_mono_energies.low]))
    vol_high_HU = Float32.(BS.to_hounsfield(de_lohi_μ.vol_high_μ; μ_water = keV_μ_water[de_mono_energies.high]))
    (vol_low_HU = vol_low_HU, vol_high_HU = vol_high_HU, geom = de_lohi_μ.geom)
end;

# ╔═╡ 030b000b-0000-4000-8000-000000000001
md"""
## 10. 2-panel display — HU at 70 keV vs HU at 150 keV

These ARE the monoenergetic outputs by construction — no Ding decomp,
no synthesis pass, no Mono+ noise shaping.  The two recons already
satisfy `μ(E) = c_water · μρ_water(E) + c_iodine · 1e-3 · μρ_I(E)`
exactly at the chosen E because there's only one E.
"""

# ╔═╡ 030b000b-0000-4000-8000-000000000010
let
    HU_window = (-200, 500)

    fig = CM.Figure(size = (1200, 600))
    title_kwargs = (titlesize = 28, subtitlesize = 20)

    mid = size(de_lohi_HU.vol_low_HU, 3) ÷ 2

    panels = (
        (1, 1, "HU @ $(Int(de_mono_energies.low)) keV",
                "monoenergetic input · FBP only",
                de_lohi_HU.vol_low_HU[:, :, mid]),
        (1, 2, "HU @ $(Int(de_mono_energies.high)) keV",
                "monoenergetic input · FBP only",
                de_lohi_HU.vol_high_HU[:, :, mid]),
    )

    hms = nothing
    for (r, c, ttl, sub, slice) in panels
        ax = CM.Axis(
            fig[r, c];
            title = ttl, subtitle = sub,
            aspect = CM.DataAspect(),
            title_kwargs...,
        )
        hms = CM.heatmap!(ax, slice; colormap = :grays, colorrange = HU_window)
        CM.hidedecorations!(ax)
    end

    CM.Colorbar(fig[1, 3], hms; label = "HU", width = 14, labelsize = 18)

    CM.save(
        joinpath(@__DIR__, "..", "assets", "monoe_2panel.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 030b000c-0000-4000-8000-000000000001
md"""
## 11. Mean water HU per energy — sanity check

Center-ROI water HU should land near 0 at both energies — `to_hounsfield`
divided by the exact NIST μ_water, so any deviation is residual noise
(no bias from polychromatic averaging).
"""

# ╔═╡ 030b000c-0000-4000-8000-000000000010
let
    nx, ny, nz = size(de_lohi_HU.vol_low_HU)
    cx, cy = nx / 2 + 0.5, ny / 2 + 0.5
    ROI_R = 8.0

    roi = CartesianIndex{2}[]
    r² = ROI_R^2
    i_lo = max(1, floor(Int, cx - ROI_R)); i_hi = min(nx, ceil(Int, cx + ROI_R))
    j_lo = max(1, floor(Int, cy - ROI_R)); j_hi = min(ny, ceil(Int, cy + ROI_R))
    for j in j_lo:j_hi, i in i_lo:i_hi
        ((i - cx)^2 + (j - cy)^2) ≤ r² && push!(roi, CartesianIndex(i, j))
    end

    function _mean_HU(vol)
        s = 0.0; n = 0
        for z in 1:nz, ci in roi
            s += vol[ci, z]; n += 1
        end
        s / n
    end

    energies = (de_mono_energies.low, de_mono_energies.high)
    mean_HU  = [_mean_HU(de_lohi_HU.vol_low_HU), _mean_HU(de_lohi_HU.vol_high_HU)]

    fig = CM.Figure(size = (700, 480))
    ax = CM.Axis(
        fig[1, 1];
        title = "Mean water HU per energy",
        subtitle = "8-px center ROI · target = 0 HU",
        xlabel = "Beam energy (keV)", ylabel = "Mean HU",
        xticks = (1:length(energies), ["$(Int(E))" for E in energies]),
        titlesize = 30, subtitlesize = 22,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )

    bar_colors = [
        CM.RGBf(0.92, 0.65, 0.20),   # 70  keV — warm
        CM.RGBf(0.20, 0.30, 0.65),   # 150 keV — cool
    ]

    CM.barplot!(
        ax, 1:length(energies), Float64.(mean_HU);
        color = bar_colors,
        strokecolor = :black, strokewidth = 1,
    )
    CM.hlines!(ax, [0.0]; color = :black, linestyle = :dash, linewidth = 2)

    for (i, hu) in enumerate(mean_HU)
        CM.text!(ax, i, hu;
            text = "$(round(hu, digits = 2)) HU",
            align = (:center, hu < 0 ? :top : :bottom),
            offset = (0, hu < 0 ? -4 : 4),
            fontsize = 16,
        )
    end

    fig
end

# ╔═╡ 030b000d-0000-4000-8000-000000000001
md"""
## 12. Per-rod measured vs theoretical HU at 70 + 150 keV

Per-rod measured HU (8-px core ROI × all z slices) versus theoretical
HU computed from `XrayAttenuation`'s monoenergetic μ(E).

**Solid line** = measured.  **Dashed line** = `1000·(μ_rod(E) − μ_water(E)) / μ_water(E)`
straight from XrayAttenuation — pure NIST physics, no calibration.
"""

# ╔═╡ 030b000d-0000-4000-8000-000000000010
const ROD_LABELS = (
    Ca = (UInt8(10), UInt8(11), UInt8(12), UInt8(13), UInt8(14), UInt8(15), UInt8(16)),
    I  = (UInt8(20), UInt8(21), UInt8(22), UInt8(23), UInt8(24), UInt8(25), UInt8(26)),
);

# ╔═╡ 030b000d-0000-4000-8000-000000000020
const ROD_NAMES = (
    Ca = ("50 mg/mL", "100 mg/mL", "200 mg/mL", "300 mg/mL", "400 mg/mL", "500 mg/mL", "600 mg/mL"),
    I  = ("2.0 mg/mL", "2.5 mg/mL", "5.0 mg/mL", "7.5 mg/mL", "10.0 mg/mL", "15.0 mg/mL", "20.0 mg/mL"),
);

# ╔═╡ 030b000d-0000-4000-8000-000000000030
rod_data = let
    materials = phantom_cpu.materials
    mask_2d = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
    ROI_RADIUS_PX = 8

    function rod_centroid(label::UInt8)
        idx = findall(==(label), mask_2d)
        isempty(idx) && error("rod_centroid: no voxels with label $label")
        cx = sum(ci -> Float64(ci[1]), idx) / length(idx)
        cy = sum(ci -> Float64(ci[2]), idx) / length(idx)
        return (cx, cy)
    end

    function rod_roi_mask(label::UInt8)
        cx, cy = rod_centroid(label)
        nx, ny = size(mask_2d)
        i_lo = max(1, floor(Int, cx - ROI_RADIUS_PX))
        i_hi = min(nx, ceil(Int, cx + ROI_RADIUS_PX))
        j_lo = max(1, floor(Int, cy - ROI_RADIUS_PX))
        j_hi = min(ny, ceil(Int, cy + ROI_RADIUS_PX))
        roi = CartesianIndex{2}[]
        r² = Float64(ROI_RADIUS_PX)^2
        for j in j_lo:j_hi, i in i_lo:i_hi
            ((i - cx)^2 + (j - cy)^2) ≤ r² && push!(roi, CartesianIndex(i, j))
        end
        return roi
    end

    rod_rois = Dict(
        lab => rod_roi_mask(lab)
            for lab in vcat(collect(ROD_LABELS.Ca), collect(ROD_LABELS.I))
    )

    energies = (de_mono_energies.low, de_mono_energies.high)
    μ_water_E = Dict(
        E => BS.compute_μ_at_energy(BS.XA.Materials.water, E)
            for E in energies
    )

    function theoretical_hu(material, E::Float64)
        μ = BS.compute_μ_at_energy(material, E)
        return 1000.0 * (μ - μ_water_E[E]) / μ_water_E[E]
    end

    function measured_hu(vol, label::UInt8)
        roi = rod_rois[label]
        n_z = size(vol, 3)
        s = 0.0; n = 0
        for z in 1:n_z, ci in roi
            s += vol[ci, z]; n += 1
        end
        return s / n
    end

    energy_vols = Dict(
        de_mono_energies.low  => de_lohi_HU.vol_low_HU,
        de_mono_energies.high => de_lohi_HU.vol_high_HU,
    )

    out = Dict{Symbol, NamedTuple}()
    for group in (:Ca, :I)
        labels = ROD_LABELS[group]
        n_rods = length(labels)
        n_E = length(energies)
        meas = zeros(Float64, n_rods, n_E)
        theo = zeros(Float64, n_rods, n_E)
        for (i, lab) in pairs(labels)
            mat = materials[Int(lab) + 1]
            for (j, E) in pairs(energies)
                meas[i, j] = measured_hu(energy_vols[E], lab)
                theo[i, j] = theoretical_hu(mat, E)
            end
        end
        out[group] = (
            labels = labels, names = ROD_NAMES[group],
            measured = meas, theoretical = theo,
        )
    end
    out
end;

# ╔═╡ 030b000d-0000-4000-8000-000000000040
let
    fig = CM.Figure(size = (1180, 580))

    cmap_ca = CM.cgrad(:Oranges, 7; categorical = true)
    cmap_i  = CM.cgrad(:GnBu,    7; categorical = true)
    energies = [de_mono_energies.low, de_mono_energies.high]

    panels = (
        (group = :Ca, title = "Calcium rods",
            subtitle = "50–600 mg/mL · Compton roll-off",
            cmap = cmap_ca, ylim = (0, 4200)),
        (group = :I, title = "Iodine rods",
            subtitle = "2–20 mg/mL · 70 vs 150 keV",
            cmap = cmap_i, ylim = (0, 1500)),
    )

    for (col, p) in pairs(panels)
        ax = CM.Axis(
            fig[1, col];
            title = p.title,
            subtitle = p.subtitle,
            xlabel = "Beam energy (keV)",
            ylabel = "HU",
            xticks = energies,
            titlesize = 32, subtitlesize = 24,
            xlabelsize = 22, ylabelsize = 22,
            xticklabelsize = 18, yticklabelsize = 16,
        )
        CM.ylims!(ax, p.ylim...)

        d = rod_data[p.group]
        rod_lines = Vector{Any}(undef, length(d.names))
        for i in eachindex(d.names)
            color = p.cmap[i]
            CM.scatterlines!(
                ax, energies, vec(d.measured[i, :]);
                color = color, linewidth = 2.5, markersize = 9,
            )
            CM.lines!(
                ax, energies, vec(d.theoretical[i, :]);
                color = color, linewidth = 1.6, linestyle = :dash,
            )
            rod_lines[i] = CM.LineElement(color = color, linewidth = 2.5)
        end

        style_meas = CM.MarkerElement(
            color = :black, marker = :circle, markersize = 9,
            strokecolor = :black, strokewidth = 1,
        )
        style_theo = CM.LineElement(
            color = :black, linewidth = 1.6, linestyle = :dash,
        )
        CM.axislegend(
            ax,
            vcat([style_meas, style_theo], rod_lines),
            vcat(["Measured", "Theoretical"], collect(d.names));
            position = :rt, framevisible = true, labelsize = 18,
            rowgap = 1, padding = (6, 6, 6, 6),
        )
    end

    CM.save(
        joinpath(@__DIR__, "..", "assets", "monoe_vs_theoretical.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 030b000e-0000-4000-8000-000000000001
md"""
## 13. Linear regression — measured vs theoretical (per energy)

Every (rod, energy) pair scattered with measured HU on the y-axis vs
theoretical HU on the x-axis, fit per-energy and overlaid against the
y = x identity.  In a clean mono pipeline both per-energy regression
slopes should sit on or very close to **1.00 with R² ≈ 1.0** — no
calibration tricks needed because there's no spectral compromise to
make.
"""

# ╔═╡ 030b000e-0000-4000-8000-000000000010
let
    fig = CM.Figure(size = (1180, 620))

    energy_colors = Dict(
        de_mono_energies.low  => CM.RGBf(0.95, 0.65, 0.13),
        de_mono_energies.high => CM.RGBf(0.10, 0.27, 0.65),
    )
    energies = [de_mono_energies.low, de_mono_energies.high]

    function fit_lr(x::Vector{Float64}, y::Vector{Float64})
        x̄ = mean(x); ȳ = mean(y)
        sxx = sum((x .- x̄) .^ 2)
        sxy = sum((x .- x̄) .* (y .- ȳ))
        β = sxy / sxx
        α = ȳ - β * x̄
        ŷ = α .+ β .* x
        ss_res = sum((y .- ŷ) .^ 2)
        ss_tot = sum((y .- ȳ) .^ 2)
        r² = 1 - ss_res / ss_tot
        return (slope = β, intercept = α, r² = r²)
    end

    panels = ((:Ca, "Calcium rods", "50–600 mg/mL"), (:I, "Iodine rods", "2–20 mg/mL"))
    for (col, (group, title, subtitle)) in pairs(panels)
        d = rod_data[group]
        ax = CM.Axis(
            fig[1, col];
            title = title, subtitle = subtitle,
            xlabel = "Theoretical HU", ylabel = "Measured HU",
            aspect = CM.AxisAspect(1),
            titlesize = 32, subtitlesize = 24,
            xlabelsize = 22, ylabelsize = 22,
            xticklabelsize = 16, yticklabelsize = 16,
        )

        lim_lo = min(0.0, minimum(d.measured), minimum(d.theoretical))
        lim_hi = max(maximum(d.measured), maximum(d.theoretical)) * 1.05
        CM.lines!(
            ax, [lim_lo, lim_hi], [lim_lo, lim_hi];
            color = :black, linestyle = :dash, linewidth = 2,
            label = "y = x (unity)",
        )

        for (j, E) in pairs(energies)
            x = Vector{Float64}(vec(d.theoretical[:, j]))
            y = Vector{Float64}(vec(d.measured[:, j]))
            color = energy_colors[E]
            CM.scatter!(ax, x, y; color = color, markersize = 11)

            f = fit_lr(x, y)
            xrange = [minimum(x), maximum(x)]
            yrange = f.intercept .+ f.slope .* xrange
            label = "$(Int(E)) keV: y = $(round(f.slope, digits = 2))·x " *
                "$(f.intercept ≥ 0 ? "+" : "−") $(round(abs(f.intercept), digits = 0))" *
                "   R² = $(round(f.r², digits = 3))"
            CM.lines!(
                ax, xrange, yrange;
                color = color, linewidth = 2, label = label,
            )
        end

        CM.axislegend(
            ax; position = :rb, framevisible = true,
            labelsize = 18, padding = (6, 6, 6, 6), rowgap = 1,
        )
    end

    CM.save(
        joinpath(@__DIR__, "..", "assets", "monoe_regression.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 030b000f-0000-4000-8000-000000000001
md"""
## 14. Quantitative agreement table

Compact `(measured / theoretical / Δ)` for every (rod, energy) pair.
At true mono input we expect single-digit HU agreement everywhere —
no K-edge boost to amplify residual mismatches because the K-edge
boost is *itself* a polychromatic averaging artifact.
"""

# ╔═╡ 030b000f-0000-4000-8000-000000000010
let
    energies = [de_mono_energies.low, de_mono_energies.high]

    function fmt_row(name, m_row, t_row)
        diffs = m_row .- t_row
        cells = ["**$(name)**"]
        for j in eachindex(energies)
            push!(cells, "$(round(m_row[j], digits = 0)) / $(round(t_row[j], digits = 0)) ($(round(diffs[j], digits = 0)))")
        end
        return "| " * join(cells, " | ") * " |"
    end

    header_e = ["measured / theoretical (Δ) at $(Int(E)) keV" for E in energies]
    header = "| Rod | " * join(header_e, " | ") * " |"
    sep = "|" * "---|"^(length(energies) + 1)

    lines_ca = [
        fmt_row(rod_data[:Ca].names[i], rod_data[:Ca].measured[i, :], rod_data[:Ca].theoretical[i, :])
            for i in eachindex(rod_data[:Ca].names)
    ]
    lines_i = [
        fmt_row(rod_data[:I].names[i], rod_data[:I].measured[i, :], rod_data[:I].theoretical[i, :])
            for i in eachindex(rod_data[:I].names)
    ]

    md_str = """
    **Calcium rods (HU)**

    $(header)
    $(sep)
    $(join(lines_ca, "\n"))

    **Iodine rods (HU)**

    $(header)
    $(sep)
    $(join(lines_i, "\n"))
    """
    Markdown.parse(md_str)
end

# ╔═╡ 030b0010-0000-4000-8000-000000000001
md"""
## Summary

The whole notebook is `simulate! → FBP → to_hounsfield`.  No BHC, no
RSKR, no cupping, no 2-basis decomp, no VMI synthesis, no Mono+ —
every one of those exists in nb03 to fight a problem that
**doesn't exist** when the source spectrum is monoenergetic.

The only structural change vs nb03 is

```julia
spectrum_override = ([E_mono], [1.0])
```

passed into `BS.create_eict_workspace(...)`, which bypasses the IPEM
polychromatic lookup.  Bowtie + heel + detector response still
evaluate at the override's energy grid, so per-pixel attenuation /
scatter / noise physics still runs faithfully.

The §13 regression should land at slope ≈ 1.0 / R² ≈ 1.0 on **both**
the Ca and I panels at **both** 70 and 150 keV.  That's the physical
ceiling that nb03's polychromatic 2-basis pipeline is trying to
recover via Ding-2020 RQ + algebraic c_water + per-energy synth — any
gap between nb03's regression and this notebook's regression IS the
cost of polychromatic input.
"""

# ╔═╡ Cell order:
# ╟─030b0001-0000-4000-8000-000000000010
# ╟─030b0001-0000-4000-8000-000000000020
# ╠═030b0001-0000-4000-8000-000000000001
# ╠═030b0001-0000-4000-8000-000000000002
# ╠═030b0001-0000-4000-8000-000000000003
# ╠═030b0001-0000-4000-8000-000000000030
# ╠═030b0001-0000-4000-8000-000000000031
# ╠═030b0001-0000-4000-8000-000000000040
# ╟─030b0001-0000-4000-8000-000000000050
# ╟─030b0002-0000-4000-8000-000000000001
# ╠═030b0002-0000-4000-8000-000000000010
# ╠═030b0002-0000-4000-8000-000000000020
# ╟─030b0003-0000-4000-8000-000000000001
# ╠═030b0003-0000-4000-8000-000000000010
# ╟─030b0004-0000-4000-8000-000000000001
# ╠═030b0004-0000-4000-8000-000000000010
# ╠═030b0004-0000-4000-8000-000000000020
# ╠═030b0004-0000-4000-8000-000000000030
# ╟─030b0005-0000-4000-8000-000000000001
# ╠═030b0005-0000-4000-8000-000000000010
# ╠═030b0005-0000-4000-8000-000000000020
# ╟─030b0006-0000-4000-8000-000000000001
# ╠═030b0006-0000-4000-8000-000000000010
# ╟─030b0007-0000-4000-8000-000000000001
# ╠═030b0007-0000-4000-8000-000000000010
# ╟─030b0008-0000-4000-8000-000000000001
# ╠═030b0008-0000-4000-8000-000000000010
# ╟─030b0009-0000-4000-8000-000000000001
# ╠═030b0009-0000-4000-8000-000000000010
# ╟─030b0009-0000-4000-8000-000000000020
# ╟─030b000a-0000-4000-8000-000000000001
# ╠═030b000a-0000-4000-8000-000000000010
# ╟─030b000b-0000-4000-8000-000000000001
# ╠═030b000b-0000-4000-8000-000000000010
# ╟─030b000c-0000-4000-8000-000000000001
# ╠═030b000c-0000-4000-8000-000000000010
# ╟─030b000d-0000-4000-8000-000000000001
# ╠═030b000d-0000-4000-8000-000000000010
# ╠═030b000d-0000-4000-8000-000000000020
# ╠═030b000d-0000-4000-8000-000000000030
# ╠═030b000d-0000-4000-8000-000000000040
# ╟─030b000e-0000-4000-8000-000000000001
# ╠═030b000e-0000-4000-8000-000000000010
# ╟─030b000f-0000-4000-8000-000000000001
# ╟─030b000f-0000-4000-8000-000000000010
# ╟─030b0010-0000-4000-8000-000000000001
