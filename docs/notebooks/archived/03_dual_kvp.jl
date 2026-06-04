### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 04000001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 04000001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 04000001-0000-4000-8000-000000000003
using Statistics: mean, std

# ╔═╡ 04000001-0000-4000-8000-000000000010
md"""
# 03 · Dual-kVp on Gammex 472 — straight FBP, no VMI

GE Apex Elite GSI rapid-kVp-switching simulation: same hardware spec
+ phantom + per-kVp duty-cycle math as the (archived) `03_dual_kvp_vmi.jl`,
but **no VMI synthesis pass** — we just FBP each acquisition with
`BS.SoftFilter()` and compare per-rod measured HU to a
**polychromatic-effective-energy** theoretical HU.

Pipeline:

```
sim 80 kVp + sim 140 kVp → 2× FDK FBP (SoftFilter) → 2× HU volumes
                                                      ↓
                                  per-rod measured HU vs theory @ E_eff
```

The "effective energy" per kVp is the spectrum-weighted mean of the
**bowtie-aware, phantom-hardened** source spectrum (`Σ E·ŵ·ĥ /
Σ ŵ·ĥ`), where `ĥ(E) = exp(−μ_water(E)·D/2)` is the Beer-Lambert
hardening through half the phantom diameter — the standard "single
transit" approximation for a centrally-positioned ROI.  We then
evaluate `XrayAttenuation`'s `compute_μ_at_energy` for each rod
material at that single E_eff, and compute theoretical HU exactly
the way nb03's old VMI verification did at each VMI keV, just
without the per-keV sweep.
"""

# ╔═╡ 04000001-0000-4000-8000-000000000020
md"""
## Setup
"""

# ╔═╡ 04000001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 04000001-0000-4000-8000-000000000031
import CairoMakie as CM

# ╔═╡ 04000001-0000-4000-8000-000000000040
begin
    GPU_BACKEND = let
        candidates = [
            (:Metal, "dde4c033-4e86-420c-a63e-0dd931031962", :MtlArray),
            (:CUDA, "052768ef-5323-5732-b1bb-66c8b64840ba", :CuArray),
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

# ╔═╡ 04000001-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 04000002-0000-4000-8000-000000000001
md"""
## 1. `Phantom` — Gammex Model 472
"""

# ╔═╡ 04000002-0000-4000-8000-000000000010
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,
    n_slices = 16,
    fov_cm = 35.0,
    z_cm = 1.0,
);

# ╔═╡ 04000002-0000-4000-8000-000000000020
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 04000003-0000-4000-8000-000000000001
md"""
## 2. `Scanner` — GE Revolution Apex Elite
"""

# ╔═╡ 04000003-0000-4000-8000-000000000010
scanner = BS.Scanner(
    source_to_isocenter = 625.6,
    source_to_detector = 1100.0,

    detector_rows = 256,
    detector_cols = 834,
    detector_row_size = 0.625,
    detector_col_size = 0.6,
    detector_shape = BS.CURVED_DETECTOR,

    focal_spot_width = 1.0,
    focal_spot_length = 1.0,
    target_angle = 10.0,

    flat_filter_material = :aluminum,
    flat_filter_thickness = 2.5,
    bowtie_filter = :ge_revolution_large,

    detector_material = :lumex,
    detector_depth = 3.0,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,

    electronic_noise = 0,
    detection_gain = 10.0,
);

# ╔═╡ 04000004-0000-4000-8000-000000000001
md"""
## 3. Dual-kVp protocols (rapid-switching duty-cycle math)

| kVp | Instantaneous mA | Duty cycle | Effective mA |
|-----|------------------|------------|--------------|
| 80  | 407              | 0.65       | 264.55       |
| 140 | 405              | 0.35       | 141.75       |
"""

# ╔═╡ 04000004-0000-4000-8000-000000000010
protocol_low = BS.CTProtocol(
    kVp = 80,
    mA = 407 * 0.65,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 04000004-0000-4000-8000-000000000020
protocol_high = BS.CTProtocol(
    kVp = 140,
    mA = 405 * 0.35,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 04000005-0000-4000-8000-000000000001
md"""
## 4. `SimOptions` and `ReconOptions`
"""

# ╔═╡ 04000005-0000-4000-8000-000000000010
sim_opts = BS.SimOptions(
    fidelity = :eict,
    seed = 1234,
);

# ╔═╡ 04000005-0000-4000-8000-000000000020
recon_opts = let
    slice_thickness_mm = 0.625
    n_recon_slices = round(Int, 5.0 / slice_thickness_mm)
    BS.ReconOptions(
        algorithm = :fdk,
        matrix_size = (512, 512, n_recon_slices),
        fov_cm = 35.0,
        z_cm = 0.5,
        filter = :standard,
    )
end;

# ╔═╡ 04000006-0000-4000-8000-000000000001
md"""
## 5. Forward project — 80 kVp
"""

# ╔═╡ 04000006-0000-4000-8000-000000000010
sim_low = let
    @info "Simulating: 80 kVp / $(round(protocol_low.mA, digits = 1)) mA-eff (DE low)…"
    ws = BS.create_eict_workspace(scanner, protocol_low, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, scanner, protocol_low, sim_opts, recon_opts)
    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom)
    ws = nothing; GC.gc(true)
    result
end;

# ╔═╡ 04000007-0000-4000-8000-000000000001
md"""
## 6. Forward project — 140 kVp
"""

# ╔═╡ 04000007-0000-4000-8000-000000000010
sim_high = let
    @info "Simulating: 140 kVp / $(round(protocol_high.mA, digits = 1)) mA-eff (DE high)…"
    ws = BS.create_eict_workspace(scanner, protocol_high, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, scanner, protocol_high, sim_opts, recon_opts)
    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom)
    ws = nothing; GC.gc(true)
    result
end;

# ╔═╡ 04000008-0000-4000-8000-000000000001
md"""
## 7. FBP per kVp with `BS.SoftFilter()`

Direct FDK on each polychromatic sinogram — **no BHC**, no RSKR, no
cupping correction.  The recon naturally delivers
**polychromatic-effective μ** at every voxel (the spectrum-weighted
average attenuation through that material), and §8 / §11 will compare
against a polychromatic-effective theoretical that uses the same
definition — so HU_meas and HU_theo share the same normalization and
slopes should land at ≈ 1.

(BHC was tried at this slot — it linearizes water nicely but only
partially handles Ca/I, leaving a slope ≈ 0.85 residual against any
single-energy theoretical.  Without BHC, the recon = polychromatic-
effective μ exactly; matching that on the theoretical side closes the
loop cleanly.)
"""

# ╔═╡ 04000008-0000-4000-8000-000000000010
de_lohi_μ = let
    matrix_size = recon_opts.matrix_size

    function _fbp_to_μ(sino_cpu, geom)
        sino_gpu = to_gpu(Float32.(sino_cpu))
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, geom, matrix_size; filter = BS.SoftFilter(),
        )
        recon_μ = Array(BS.reconstruct!(ws, sino_gpu, geom, matrix_size))
        ws = nothing; sino_gpu = nothing
        GC.gc(true)
        return Float32.(recon_μ)
    end

    vol_low_μ = _fbp_to_μ(sim_low.sino, sim_low.geom)
    vol_high_μ = _fbp_to_μ(sim_high.sino, sim_high.geom)

    (vol_low_μ = vol_low_μ, vol_high_μ = vol_high_μ, geom = sim_low.geom)
end;

# ╔═╡ 04000009-0000-4000-8000-000000000001
md"""
## 8. Per-kVp `μ_water` — measured from the FBP center ROI

8-px circular ROI at the phantom center, mean across all z slices.
By construction this makes solid water read at exactly 0 HU after §9's
`to_hounsfield` step.  No BHC / cupping in this notebook, so the
divisor absorbs all the polychromatic offset for the water region.
"""

# ╔═╡ 04000009-0000-4000-8000-000000000010
kVp_μ_water = let
    nx, ny, nz = size(de_lohi_μ.vol_low_μ)
    cx, cy = nx / 2 + 0.5, ny / 2 + 0.5
    ROI_R = 8.0
    r² = ROI_R^2

    roi = CartesianIndex{2}[]
    i_lo = max(1, floor(Int, cx - ROI_R)); i_hi = min(nx, ceil(Int, cx + ROI_R))
    j_lo = max(1, floor(Int, cy - ROI_R)); j_hi = min(ny, ceil(Int, cy + ROI_R))
    for j in j_lo:j_hi, i in i_lo:i_hi
        ((i - cx)^2 + (j - cy)^2) ≤ r² && push!(roi, CartesianIndex(i, j))
    end

    function _mean_μ(vol)
        s = 0.0; n = 0
        for z in 1:nz, ci in roi
            s += vol[ci, z]; n += 1
        end
        return s / n
    end

    Dict(
        80 => Float64(_mean_μ(de_lohi_μ.vol_low_μ)),
        140 => Float64(_mean_μ(de_lohi_μ.vol_high_μ)),
    )
end;

# ╔═╡ 0400000a-0000-4000-8000-000000000001
md"""
## 9. HU conversion

`BS.to_hounsfield` per kVp using the measured `kVp_μ_water` from §8.
"""

# ╔═╡ 0400000a-0000-4000-8000-000000000010
de_lohi_HU = let
    vol_low_HU = Float32.(BS.to_hounsfield(de_lohi_μ.vol_low_μ; μ_water = kVp_μ_water[80]))
    vol_high_HU = Float32.(BS.to_hounsfield(de_lohi_μ.vol_high_μ; μ_water = kVp_μ_water[140]))
    (vol_low_HU = vol_low_HU, vol_high_HU = vol_high_HU, geom = de_lohi_μ.geom)
end;

# ╔═╡ 0400000b-0000-4000-8000-000000000001
md"""
## 10. Per-kVp **effective energy** — spectrum-weighted mean, phantom-hardened

For each kVp, fetch the bowtie-aware, additional-filter-attenuated
source spectrum, harden it through `D/2` of solid water (single-transit
Beer-Lambert), and take the photon-fluence-weighted mean energy:

```
E_eff(kVp) = Σ E · ŵ(E) · exp(−μ_water(E)·D/2)
           ─────────────────────────────────────
              Σ ŵ(E) · exp(−μ_water(E)·D/2)
```

`D` is the actual phantom diameter from
`BS.estimate_phantom_diameter_cm(...)`.  This is just a **single-anchor
proxy** for the polychromatic spectrum — useful for "what mono energy
would give similar HU values?" intuition.  The §11 / §V2 regression
uses NIST `μ(E_eff)` for theoretical and shows where direct-FBP at kVp
deviates from a perfect-mono comparison.
"""

# ╔═╡ 0400000b-0000-4000-8000-000000000010
kVp_E_eff = let
    body_diameter_cm = BS.estimate_phantom_diameter_cm(
        phantom_cpu.mask, phantom_cpu.voxel_size .* 10.0,
    )
    D_half = body_diameter_cm / 2

    function _eff_E(protocol, geom)
        e, w = BS.resolve_source_spectrum_with_bowtie(
            sim_opts, protocol; scanner = scanner, geom = geom,
        )
        e_f = Float64.(e)
        w_1d = if ndims(w) == 1
            Float64.(w)
        else
            mid_c = size(w, 1) ÷ 2 + 1
            mid_r = size(w, 2) ÷ 2 + 1
            Float64.(@view w[mid_c, mid_r, :])
        end
        μ = [BS.compute_μ_at_energy(BS.XA.Materials.water, ei) for ei in e_f]
        w_hard = w_1d .* exp.(-μ .* D_half)
        return sum(e_f .* w_hard) / sum(w_hard)
    end

    Dict(
        80 => _eff_E(protocol_low, sim_low.geom),
        140 => _eff_E(protocol_high, sim_high.geom),
    )
end;

# ╔═╡ 0400000b-0000-4000-8000-000000000020
md"""
**Effective energies:**

* 80 kVp:  $(round(kVp_E_eff[80],  digits = 2)) keV
* 140 kVp: $(round(kVp_E_eff[140], digits = 2)) keV
"""

# ╔═╡ 0400000c-0000-4000-8000-000000000001
md"""
## 11. Per-rod measured vs single-energy NIST theoretical @ E_eff

Per-rod measured HU is an 8-px core-ROI mean across all z slices.  Per-rod
theoretical HU is single-anchor NIST physics evaluated at `E_eff(kVp)`:

```
HU_theo(rod, kVp) = 1000·(μ_rod(E_eff) − μ_water(E_eff)) / μ_water(E_eff)
```

with `μ` from `XrayAttenuation.compute_μ_at_energy`.  No basis decomp,
no synthesis — direct comparison to NIST.

### Why the §V2 slopes will land < 1 (and that's expected)

Direct FBP at a polychromatic kVp **cannot** match a single-energy
NIST anchor exactly — the recon physically integrates over the whole
spectrum, while `μ(E_eff)` is just one mono evaluation.  Three layered
effects pull the regression slopes below unity:

1. **Polychromatic averaging vs single-E anchor.**  The recon's
   effective μ at any voxel is `Σ ŵ(E)·μ(E) / Σ ŵ(E)` (over the
   detector-weighted, phantom-hardened spectrum), not `μ(E_eff)`.
   For materials whose μ(E) shape differs from water (high-Z, K-edge)
   the two diverge by ~10 %.

2. **Rod self-hardening.**  A 2.8 cm chord through a Ca 600 mg/mL or
   I 20 mg/mL rod hardens the spectrum *further* before the central
   voxel — the recon extracts a μ at an even higher effective energy
   than the phantom-water hardening alone predicts.  This effect
   scales with concentration, so high-c rods drag the slope down
   while low-c rods stay near unity.

3. **Detector response + scatter.**  The detector's energy-dependent
   QE η(E) re-weights the spectrum, and DAS scatter subtraction
   leaves residuals that bias the recon.  Neither shows up in the
   single-E `μ(E_eff)` anchor.

To recover slope ≈ 1 you need either a calibrated VMI synthesis pass
(which is what the archived `03_dual_kvp_vmi.jl` does — Ding c_iodine
fit + per-energy synth + Mono+ all conspire to absorb the polychromatic
residuals), or a much more sophisticated theoretical that simulates
the full forward model.  Both are out of scope for this notebook.

The slope-< 1 *is* the polychromatic residual, and it's the
**motivation** for needing VMI / decomposition pipelines on
dual-kVp data.
"""

# ╔═╡ 0400000c-0000-4000-8000-000000000005
const ROD_LABELS = (
    Ca = (UInt8(10), UInt8(11), UInt8(12), UInt8(13), UInt8(14), UInt8(15), UInt8(16)),
    I = (UInt8(20), UInt8(21), UInt8(22), UInt8(23), UInt8(24), UInt8(25), UInt8(26)),
);

# ╔═╡ 0400000c-0000-4000-8000-000000000010
const ROD_NAMES = (
    Ca = ("50 mg/mL", "100 mg/mL", "200 mg/mL", "300 mg/mL", "400 mg/mL", "500 mg/mL", "600 mg/mL"),
    I = ("2.0 mg/mL", "2.5 mg/mL", "5.0 mg/mL", "7.5 mg/mL", "10.0 mg/mL", "15.0 mg/mL", "20.0 mg/mL"),
);

# ╔═╡ 0400000c-0000-4000-8000-000000000020
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

    kVps = (80, 140)
    μ_water_E = Dict(
        kVp => BS.compute_μ_at_energy(BS.XA.Materials.water, kVp_E_eff[kVp])
            for kVp in kVps
    )

    function theoretical_hu(material, kVp::Int)
        # Single-anchor "as if the source were monoenergetic at E_eff"
        # theoretical HU — pure NIST μ at E_eff(kVp).
        E = kVp_E_eff[kVp]
        μ = BS.compute_μ_at_energy(material, E)
        return 1000.0 * (μ - μ_water_E[kVp]) / μ_water_E[kVp]
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

    kVp_vols = Dict(
        80 => de_lohi_HU.vol_low_HU,
        140 => de_lohi_HU.vol_high_HU,
    )

    out = Dict{Symbol, NamedTuple}()
    for group in (:Ca, :I)
        labels = ROD_LABELS[group]
        n_rods = length(labels)
        n_kVp = length(kVps)
        meas = zeros(Float64, n_rods, n_kVp)
        theo = zeros(Float64, n_rods, n_kVp)
        for (i, lab) in pairs(labels)
            mat = materials[Int(lab) + 1]
            for (j, kVp) in pairs(kVps)
                meas[i, j] = measured_hu(kVp_vols[kVp], lab)
                theo[i, j] = theoretical_hu(mat, kVp)
            end
        end
        out[group] = (
            labels = labels, names = ROD_NAMES[group],
            measured = meas, theoretical = theo,
        )
    end
    out
end;

# ╔═╡ 0400000d-0000-4000-8000-000000000001
md"""
## Visualization 1 — 2-panel FBP HU display
"""

# ╔═╡ 0400000d-0000-4000-8000-000000000010
let
    HU_window = (-200, 500)

    fig = CM.Figure(size = (1200, 600))
    title_kwargs = (titlesize = 28, subtitlesize = 20)

    mid = size(de_lohi_HU.vol_low_HU, 3) ÷ 2

    panels = (
        (
            1, 1, "HU @ 80 kVp",
            "$(round(kVp_E_eff[80], digits = 1)) keV effective · SoftFilter FBP",
            de_lohi_HU.vol_low_HU[:, :, mid],
        ),
        (
            1, 2, "HU @ 140 kVp",
            "$(round(kVp_E_eff[140], digits = 1)) keV effective · SoftFilter FBP",
            de_lohi_HU.vol_high_HU[:, :, mid],
        ),
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
        joinpath(@__DIR__, "..", "assets", "dual_kvp_2panel.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0400000e-0000-4000-8000-000000000001
md"""
## Visualization 2 — Linear regression: measured vs theoretical @ E_eff

Per-(rod, kVp) measured HU on the y-axis vs theoretical HU at the
kVp's effective energy on the x-axis, fit per-kVp and overlaid against
y = x.  Slope ≈ 1 / R² ≈ 1 ⇒ the FBP recon matches NIST physics
evaluated at the spectrum-equivalent monoenergetic anchor.  Slope > 1
on iodine ⇒ residual K-edge / spectral effects that the single-energy
anchor doesn't capture (the same effect that motivates VMI synthesis
in the archived `03_dual_kvp_vmi.jl`).
"""

# ╔═╡ 0400000e-0000-4000-8000-000000000010
let
    fig = CM.Figure(size = (1180, 620))

    kVps = [80, 140]
    kvp_colors = Dict(
        80 => CM.RGBf(0.95, 0.65, 0.13),   # warm
        140 => CM.RGBf(0.1, 0.27, 0.65),   # cool
    )

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
            xlabel = "Theoretical HU (@ E_eff)", ylabel = "Measured HU",
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

        for (j, kVp) in pairs(kVps)
            x = Vector{Float64}(vec(d.theoretical[:, j]))
            y = Vector{Float64}(vec(d.measured[:, j]))
            color = kvp_colors[kVp]
            CM.scatter!(ax, x, y; color = color, markersize = 11)

            f = fit_lr(x, y)
            xrange = [minimum(x), maximum(x)]
            yrange = f.intercept .+ f.slope .* xrange
            E_eff = round(kVp_E_eff[kVp], digits = 1)
            label = "$(kVp) kVp ($(E_eff) keV): y = $(round(f.slope, digits = 2))·x " *
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
        joinpath(@__DIR__, "..", "assets", "dual_kvp_regression.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0400000f-0000-4000-8000-000000000001
md"""
## Summary

The whole notebook is

```
simulate! per kVp → FDK with SoftFilter → to_hounsfield (measured μ_water)
                                          ↓
                      compare per-rod measured HU vs theory @ E_eff
```

with `E_eff(kVp)` the photon-fluence-weighted mean energy of the
bowtie + flat + additional-filtered + phantom-half-hardened source
spectrum at that kVp.

No BHC, no RSKR, no cupping correction, no Ding decomposition, no VMI
synthesis, no Mono+ post-processing — every one of those exists in the
archived `03_dual_kvp_vmi.jl` to fight a problem this notebook simply
**accepts**: residual polychromatic offset gets folded into the HU
divisor (water reads 0 by construction), and per-rod K-edge / spectral
mismatches show up as deviation from y = x in the §V2 regression.

For an actual VMI pipeline (40 / 70 / 100 / 140 keV per-energy synth
+ Ding c_iodine fit + Mono+ noise shaping), see the archived versions
in `docs/notebooks/archived/`.
"""

# ╔═╡ Cell order:
# ╟─04000001-0000-4000-8000-000000000010
# ╟─04000001-0000-4000-8000-000000000020
# ╠═04000001-0000-4000-8000-000000000001
# ╠═04000001-0000-4000-8000-000000000002
# ╠═04000001-0000-4000-8000-000000000003
# ╠═04000001-0000-4000-8000-000000000030
# ╠═04000001-0000-4000-8000-000000000031
# ╠═04000001-0000-4000-8000-000000000040
# ╟─04000001-0000-4000-8000-000000000050
# ╟─04000002-0000-4000-8000-000000000001
# ╠═04000002-0000-4000-8000-000000000010
# ╠═04000002-0000-4000-8000-000000000020
# ╟─04000003-0000-4000-8000-000000000001
# ╠═04000003-0000-4000-8000-000000000010
# ╟─04000004-0000-4000-8000-000000000001
# ╠═04000004-0000-4000-8000-000000000010
# ╠═04000004-0000-4000-8000-000000000020
# ╟─04000005-0000-4000-8000-000000000001
# ╠═04000005-0000-4000-8000-000000000010
# ╠═04000005-0000-4000-8000-000000000020
# ╟─04000006-0000-4000-8000-000000000001
# ╠═04000006-0000-4000-8000-000000000010
# ╟─04000007-0000-4000-8000-000000000001
# ╠═04000007-0000-4000-8000-000000000010
# ╟─04000008-0000-4000-8000-000000000001
# ╠═04000008-0000-4000-8000-000000000010
# ╟─04000009-0000-4000-8000-000000000001
# ╠═04000009-0000-4000-8000-000000000010
# ╟─0400000a-0000-4000-8000-000000000001
# ╠═0400000a-0000-4000-8000-000000000010
# ╟─0400000b-0000-4000-8000-000000000001
# ╠═0400000b-0000-4000-8000-000000000010
# ╟─0400000b-0000-4000-8000-000000000020
# ╟─0400000c-0000-4000-8000-000000000001
# ╠═0400000c-0000-4000-8000-000000000005
# ╠═0400000c-0000-4000-8000-000000000010
# ╠═0400000c-0000-4000-8000-000000000020
# ╟─0400000d-0000-4000-8000-000000000001
# ╠═0400000d-0000-4000-8000-000000000010
# ╟─0400000e-0000-4000-8000-000000000001
# ╠═0400000e-0000-4000-8000-000000000010
# ╟─0400000f-0000-4000-8000-000000000001
