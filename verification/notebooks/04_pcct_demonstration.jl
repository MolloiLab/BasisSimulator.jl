### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ a0000005-0001-4000-8000-000000000001
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()

	using Revise
end

# ╔═╡ b0000004-0001-0001-0001-000000000090
using Unitful: @u_str

# ╔═╡ a0000005-0006-4000-8000-000000000006
using Metal

# ╔═╡ a0000005-0007-4000-8000-000000000007
md"""
# Notebook 04: Photon-Counting CT (PCCT) Verification

**BasisSimulator.jl — Part 4: PCCT Physics Verification on Gammex 472**

*Same NAEOTOM Alpha scanner as notebook 06 (XCAT), applied to Gammex 472 phantom for quantitative validation*

---

This notebook verifies the PCCT pipeline using the **same scanner/protocol/SimOptions as notebook 06** but on the Gammex 472 phantom for quantitative analysis:

1. **v24.0 Detector Physics Visualizations** (pure CPU — no GPU memory)
   - Koch-Mehrin charge cloud transport (σ ≈ 12-14 μm)
   - K-fluorescence with full Table 1 (5 K-lines/element, Te→Cd cascade)
   - Hecht CCE with small-pixel weighting potential (w/L ≈ 0.17)
   - Yang 2025 seminonparalyzable pileup (VMR sub-Poisson at high flux)
   - Unified DRM (physics-based FWHM = 3.55 keV at 60 keV)
2. **PCCT simulation** — NAEOTOM Alpha, 140 kVp, 4 energy bins, 984 views
3. **Water calibration** → empirical HU conversion
4. **FDK + Hybrid IR** reconstruction comparison
5. **VMI sweep** at 40, 70, 100, 140 keV with energy-dependent contrast
6. **K-edge imaging** — iodine-specific contrast from bin subtraction
"""

# ╔═╡ a0000005-0008-4000-8000-000000000008
md"## 1. Setup"

# ╔═╡ a0000005-0002-4000-8000-000000000002
# ╠═╡ show_logs = false
import PlutoUI as UI

# ╔═╡ a0000005-0003-4000-8000-000000000003
import BasisSimulator as BS

# ╔═╡ a0000005-0004-4000-8000-000000000004
# ╠═╡ show_logs = false
import CairoMakie as CM

# ╔═╡ a0000005-0005-4000-8000-000000000005
import Statistics: mean, std

# ╔═╡ b0000004-0001-0001-0001-000000000091
import XrayAttenuation as XA

# ╔═╡ a0000005-0009-4000-8000-000000000009
begin
	FIGURES_DIR = joinpath(dirname(@__DIR__), "figures")
	mkpath(FIGURES_DIR)
end

# ╔═╡ a0000005-0010-4000-8000-000000000010
md"""
## 2. NAEOTOM Alpha Scanner

Identical to notebook 06 (XCAT) `scanner_pcct_standard`. Fixed detector array for Gammex 472 FOV.
"""

# ╔═╡ a0000005-0010-4000-8000-000000000099
SIM_CONFIG = (
	imageSize = 512,
	sliceCount = 32,
	sliceThickness = 1.25,      # mm
	fov_mm = 350.0,

	# NAEOTOM Alpha geometry — VERIFIED (Konrad 2025, FDA K201501)
	sid = 595.0,
	sdd = 1085.5,
	detectorColCount = ceil(Int, 500.0 / 0.4),  # 1250 cols for 50cm FOV at isocenter
	detectorRowCount = 64,      # subset for speed
	detectorColSize = 0.4,      # mm at isocenter (2×2 binned from 0.2mm native)
	detectorRowSize = 0.4,      # mm at isocenter

	viewsPerRotation = 984,
	rotationTime = 0.25,        # fast gantry
)

# ╔═╡ c9636083-f2fe-451c-ae38-077eb7cd19e7
# NAEOTOM Alpha — matches notebook 06 scanner_pcct_standard exactly
naeotom = BS.Scanner(
	# GEOMETRY — VERIFIED (Konrad 2025, FDA K201501)
	source_to_isocenter = SIM_CONFIG.sid,
	source_to_detector = SIM_CONFIG.sdd,

	# DETECTOR ARRAY (standard mode: 2×2 binned from 0.2mm native)
	detector_rows = SIM_CONFIG.detectorRowCount,
	detector_cols = SIM_CONFIG.detectorColCount,
	detector_row_size = SIM_CONFIG.detectorRowSize,

	# detector_col_size is the element pitch at isocenter (mm).
	detector_col_size = SIM_CONFIG.detectorColSize,

	detector_shape = BS.CURVED_DETECTOR,
	detector_row_offset = 0.0,
	detector_col_offset = 0.2, # quarter-detector offset

	# X-RAY SOURCE
	focal_spot_width = 0.4, # mm
	focal_spot_length = 0.5, # mm
	target_angle = 7.0, # degrees

	# GANTRY
	gantry_rotation_time = SIM_CONFIG.rotationTime,
	max_scan_fov = 500.0,
	gantry_aperture = 820.0,

	# FILTRATION
	flat_filter_material = :aluminum,
	flat_filter_thickness = 2.5, # mm

	# DETECTOR PHYSICS — CdTe direct-conversion (v24.0 physics)
	detector_material = :cdte,
	detector_depth = 1.6, # mm — VERIFIED (Konrad 2025)
	fill_factor_row = 0.95,
	fill_factor_col = 0.95,
	detection_gain = 1.0, # direct conversion (no scintillator gain)
	electronic_noise = 0.0, # PCCT: thresholds eliminate electronic noise

	# PCCT-SPECIFIC FIELDS — all actively used in pcct_forward_project()
	detector_type = :photon_counting,
	n_energy_bins = 4,
	energy_thresholds = [20.0, 35.0, 55.0, 70.0], # keV — clinical NAEOTOM thresholds
	energy_resolution = 10.0, # keV FWHM — superseded by unified DRM at fidelity=:pcct
	charge_sharing_fwhm = 0.08, # mm — superseded by Koch-Mehrin ODE at fidelity=:pcct
	dead_time_ns = 5.0, # ns — used in Yang 2025 pileup model
	pixel_mode = :standard # :standard (0.4mm), :uhr (0.2mm), :macro (0.8mm)
)

# ╔═╡ a0000005-0011-4000-8000-000000000011
md"""
### Scanner Specification
- **Detector material**: CdTe ($(naeotom.detector_material))
- **Crystal thickness**: $(naeotom.detector_depth) mm
- **Energy bins**: $(naeotom.n_energy_bins)
- **Thresholds**: $(naeotom.energy_thresholds) keV
- **w/L ratio**: $(round(BS.pixel_to_thickness_ratio(BS.NAEOTOM_ALPHA), digits=3)) (small-pixel regime)
- **Electronic noise**: $(BS.NAEOTOM_ALPHA.electronic_noise_keV) keV RMS
- **Bias voltage**: $(BS.NAEOTOM_ALPHA.bias_voltage_V) V
- **SID**: $(naeotom.source_to_isocenter) mm | **SDD**: $(naeotom.source_to_detector) mm
- **Detector**: $(naeotom.detector_rows) × $(naeotom.detector_cols) at $(naeotom.detector_col_size) mm pitch
"""

# ╔═╡ a0000005-0012-4000-8000-000000000012
let
	# Quantum efficiency η(E) for CdTe, CZT, and Si
	E_range = collect(10.0:0.25:140.0)
	η_cdte = [BS.quantum_efficiency(BS.CDTE_MATERIAL, 1.6, E) for E in E_range]
	η_czt = [BS.quantum_efficiency(BS.CZT_MATERIAL, 2.0, E) for E in E_range]
	η_si = [BS.quantum_efficiency(BS.SI_MATERIAL, 1.6, E) for E in E_range]
	η_si_thick = [BS.quantum_efficiency(BS.SI_MATERIAL, 30.0, E) for E in E_range]

	fig = CM.Figure(size=(1100, 600), fontsize=14)
	ax = CM.Axis(fig[1,1], xlabel="Photon Energy (keV)", ylabel="Quantum Efficiency η(E)",
		title="Detector Quantum Efficiency: CdTe vs Si")
	CM.lines!(ax, E_range, η_cdte, linewidth=3, label="CdTe 1.6mm (NAEOTOM)")
	CM.lines!(ax, E_range, η_czt, linewidth=2.5, linestyle=:dash, label="CZT 2.0mm")
	CM.lines!(ax, E_range, η_si, linewidth=2.5, label="Si 1.6mm")
	CM.lines!(ax, E_range, η_si_thick, linewidth=2.5, linestyle=:dash, label="Si 30mm (deep-Si)")
	CM.vlines!(ax, [26.7, 31.8], color=:gray, linestyle=:dot, linewidth=1.5)
	CM.text!(ax, 27.0, 0.5, text="Cd K", fontsize=12, color=:gray)
	CM.text!(ax, 32.0, 0.5, text="Te K", fontsize=12, color=:gray)
	CM.axislegend(ax, position=:rb)
	CM.ylims!(ax, 0.0, 1.05)

	CM.save(joinpath(FIGURES_DIR, "nb04_naeotom_qe_curve.png"), fig, px_per_unit=3)
	fig
end

# ╔═╡ a0000005-0040-4000-8000-000000000040
md"""
## 3. Charge Cloud Transport (Koch-Mehrin 2020)

The charge cloud grows during electron drift from interaction site to anode via thermal diffusion and electrostatic repulsion. Solved via RK4 integration of Koch-Mehrin 2020 (NIM-A 976:164241).
"""

# ╔═╡ a0000005-0041-4000-8000-000000000041
let
	E_range = collect(10.0:0.25:140.0)
	σ_naeotom = [BS.mean_charge_cloud_sigma_mm(E, BS.NAEOTOM_ALPHA) for E in E_range]
	σ_hexitec = [BS.mean_charge_cloud_sigma_mm(E, BS.HEXITEC) for E in E_range]
	σ_initial = [BS.initial_cloud_sigma_um(E) / 1000.0 for E in E_range]

	fig = CM.Figure(size=(1100, 600), fontsize=14)
	ax = CM.Axis(fig[1,1], xlabel="Photon Energy (keV)", ylabel="Charge Cloud σ (mm)",
		title="Depth-Averaged Charge Cloud Size vs Energy")
	CM.lines!(ax, E_range, σ_naeotom, linewidth=3, color=:blue, label="NAEOTOM (1.6mm, 800V)")
	CM.lines!(ax, E_range, σ_hexitec, linewidth=2.5, color=:red, linestyle=:dash, label="HEXITEC (1.0mm, 500V)")
	CM.lines!(ax, E_range, σ_initial, linewidth=2, color=:gray, linestyle=:dot, label="Initial cloud only")
	CM.vlines!(ax, [26.7, 31.8], color=:gray, linestyle=:dot, linewidth=1.5)
	CM.axislegend(ax, position=:lt)
	CM.save(joinpath(FIGURES_DIR, "nb04_charge_cloud_sigma.png"), fig, px_per_unit=3)
	fig
end

# ╔═╡ a0000005-0042-4000-8000-000000000042
let
	z_range = collect(0.0:0.002:1.0)
	σ_nae_60 = [BS.charge_cloud_sigma_mm(60.0, z, BS.NAEOTOM_ALPHA) for z in z_range]
	σ_nae_30 = [BS.charge_cloud_sigma_mm(30.0, z, BS.NAEOTOM_ALPHA) for z in z_range]
	σ_hex_60 = [BS.charge_cloud_sigma_mm(60.0, z, BS.HEXITEC) for z in z_range]
	σ_hex_30 = [BS.charge_cloud_sigma_mm(30.0, z, BS.HEXITEC) for z in z_range]

	fig = CM.Figure(size=(1100, 600), fontsize=14)
	ax = CM.Axis(fig[1,1], xlabel="Fractional Depth z (0=cathode, 1=anode)",
		ylabel="Charge Cloud σ (mm)", title="Charge Cloud Size vs Interaction Depth")
	CM.lines!(ax, z_range, σ_nae_60, linewidth=3, color=:blue, label="NAEOTOM 60 keV")
	CM.lines!(ax, z_range, σ_nae_30, linewidth=2.5, color=:blue, linestyle=:dash, label="NAEOTOM 30 keV")
	CM.lines!(ax, z_range, σ_hex_60, linewidth=2.5, color=:red, label="HEXITEC 60 keV")
	CM.lines!(ax, z_range, σ_hex_30, linewidth=2, color=:red, linestyle=:dash, label="HEXITEC 30 keV")
	CM.axislegend(ax, position=:rt)
	CM.save(joinpath(FIGURES_DIR, "nb04_charge_cloud_depth.png"), fig, px_per_unit=3)
	fig
end

# ╔═╡ a0000005-0043-4000-8000-000000000043
let
	E_range = collect(10.0:0.5:140.0)
	fluor_model = BS.compute_cdte_fluorescence_model(
		BS.NAEOTOM_ALPHA.pixel_pitch_mm, BS.NAEOTOM_ALPHA.thickness_mm)

	p_cloud = Float64[]; p_fluor = Float64[]; p_total = Float64[]
	for E in E_range
		σ = BS.mean_charge_cloud_sigma_mm(E, BS.NAEOTOM_ALPHA)
		pc = BS.charge_sharing_probability(σ, BS.NAEOTOM_ALPHA.pixel_pitch_mm)
		pf = BS.fluorescence_sharing_boost(E, fluor_model)
		push!(p_cloud, pc); push!(p_fluor, pf); push!(p_total, min(pc + pf, 1.0))
	end

	fig = CM.Figure(size=(1100, 600), fontsize=14)
	ax = CM.Axis(fig[1,1], xlabel="Photon Energy (keV)", ylabel="Charge Sharing Probability",
		title="Charge Sharing: Cloud Transport + K-Fluorescence")
	CM.band!(ax, E_range, zeros(length(E_range)), p_cloud, color=(:blue, 0.15), label="Cloud sharing")
	CM.band!(ax, E_range, p_cloud, p_total, color=(:orange, 0.2), label="Fluorescence boost")
	CM.lines!(ax, E_range, p_total, linewidth=3, color=:black, label="Total sharing")
	CM.vlines!(ax, [26.7, 31.8], color=:gray, linestyle=:dot, linewidth=1.5)
	CM.axislegend(ax, position=:rt)
	CM.ylims!(ax, 0.0, nothing)
	CM.save(joinpath(FIGURES_DIR, "nb04_charge_sharing.png"), fig, px_per_unit=3)
	fig
end

# ╔═╡ a0000005-0050-4000-8000-000000000050
md"""
## 4. K-Fluorescence (Koch-Mehrin 2020, Table 1)

5 K-lines per element, Te→Cd cascade, analytical escape fractions.
"""

# ╔═╡ a0000005-0051-4000-8000-000000000051
let
	fluor_model = BS.compute_cdte_fluorescence_model(
		BS.NAEOTOM_ALPHA.pixel_pitch_mm, BS.NAEOTOM_ALPHA.thickness_mm)
	cd = BS.CD_FLUORESCENCE; te = BS.TE_FLUORESCENCE

	lines = String[]
	push!(lines, "### Cadmium K-Shell Transitions")
	push!(lines, "K-edge: $(cd.k_edge_keV) keV | ω_K = $(cd.k_yield)\n")
	push!(lines, "| Transition | Energy (keV) | Relative Rate | Escape Prob |")
	push!(lines, "|------------|-------------|---------------|-------------|")
	for (i, t) in enumerate(cd.k_transitions)
		p_esc = i <= length(fluor_model.cd_escape_probs) ? round(fluor_model.cd_escape_probs[i], digits=4) : "-"
		push!(lines, "| $(t.label) | $(t.energy_keV) | $(round(t.relative_rate, digits=4)) | $(p_esc) |")
	end
	push!(lines, "\n### Tellurium K-Shell Transitions")
	push!(lines, "K-edge: $(te.k_edge_keV) keV | ω_K = $(te.k_yield)\n")
	push!(lines, "| Transition | Energy (keV) | Relative Rate | Escape Prob |")
	push!(lines, "|------------|-------------|---------------|-------------|")
	for (i, t) in enumerate(te.k_transitions)
		p_esc = i <= length(fluor_model.te_escape_probs) ? round(fluor_model.te_escape_probs[i], digits=4) : "-"
		push!(lines, "| $(t.label) | $(t.energy_keV) | $(round(t.relative_rate, digits=4)) | $(p_esc) |")
	end
	push!(lines, "\nTe→Cd cascade: $(round(fluor_model.te_to_cd_cascade_prob, digits=4))")
	Markdown.parse(join(lines, "\n"))
end

# ╔═╡ a0000005-0060-4000-8000-000000000060
md"""
## 5. Charge Collection Efficiency (Hecht + Small-Pixel Weighting)

NAEOTOM w/L ≈ 0.17 → deep in the small-pixel regime. Barrett 1995 weighting potential.
"""

# ╔═╡ a0000005-0061-4000-8000-000000000061
let
	z = collect(0.0:0.002:1.0)
	wL_values = [0.17, 0.25, 1.0, 5.0]
	labels = ["w/L=0.17 (NAEOTOM)", "w/L=0.25", "w/L=1.0", "w/L=5.0 (planar)"]
	colors = [:blue, :red, :green, :purple]

	fig = CM.Figure(size=(1100, 600), fontsize=14)
	ax = CM.Axis(fig[1,1], xlabel="Normalized Depth z", ylabel="Weighting Potential ψ(z)",
		title="Small-Pixel Weighting Potential (Barrett 1995)")
	for (i, wL) in enumerate(wL_values)
		ψ = [BS.small_pixel_weighting_potential(zi, wL) for zi in z]
		CM.lines!(ax, z, ψ, linewidth=2.5, color=colors[i], label=labels[i])
	end
	CM.lines!(ax, z, z, linewidth=1.5, color=:gray, linestyle=:dash, label="Linear (planar)")
	CM.axislegend(ax, position=:lt)
	CM.save(joinpath(FIGURES_DIR, "nb04_weighting_potential.png"), fig, px_per_unit=3)
	fig
end

# ╔═╡ a0000005-0062-4000-8000-000000000062
let
	z_range = collect(0.005:0.002:0.995)
	L_cm = BS.NAEOTOM_ALPHA.thickness_mm / 10.0
	V = BS.NAEOTOM_ALPHA.bias_voltage_V
	μeτe = BS.CDTE_TRANSPORT.mu_e_tau_e_cm2_per_V
	μhτh = BS.CDTE_TRANSPORT.mu_h_tau_h_cm2_per_V
	wL_naeotom = BS.pixel_to_thickness_ratio(BS.NAEOTOM_ALPHA)

	cce_naeotom = [BS.hecht_cce_weighted(z * L_cm, L_cm, V, μeτe, μhτh, wL_naeotom) for z in z_range]
	cce_planar = [BS.hecht_cce_weighted(z * L_cm, L_cm, V, μeτe, μhτh, 5.0) for z in z_range]

	fig = CM.Figure(size=(1100, 600), fontsize=14)
	ax = CM.Axis(fig[1,1], xlabel="Fractional Depth z", ylabel="Charge Collection Efficiency",
		title="CCE vs Depth: Small-Pixel vs Planar")
	CM.lines!(ax, z_range, cce_naeotom, linewidth=3, color=:blue,
		label="NAEOTOM (w/L=$(round(wL_naeotom, digits=2)))")
	CM.lines!(ax, z_range, cce_planar, linewidth=2.5, color=:red, linestyle=:dash, label="Planar (w/L=5.0)")
	CM.hlines!(ax, [1.0], color=:gray, linestyle=:dot, linewidth=1.5)
	CM.axislegend(ax, position=:rb)
	CM.ylims!(ax, 0.0, 1.05)
	CM.save(joinpath(FIGURES_DIR, "nb04_cce_vs_depth.png"), fig, px_per_unit=3)
	fig
end

# ╔═╡ a0000005-0070-4000-8000-000000000070
md"""
## 6. Pileup & Variance-to-Mean Ratio (Yang 2025)

Sub-Poisson statistics (VMR < 1) — a key PCCT signature confirmed by Konrad 2025.
"""

# ╔═╡ a0000005-0071-4000-8000-000000000071
let
	τ_s = 10.0e-9
	true_rates = collect(range(0.0, 2e8, length=1000))
	rec_nonpar = [BS.recorded_count_rate(a, τ_s; model=:nonparalyzable) for a in true_rates]
	rec_par = [BS.recorded_count_rate(a, τ_s; model=:paralyzable) for a in true_rates]
	rec_semi = [BS.recorded_count_rate(a, τ_s; model=:seminonparalyzable) for a in true_rates]

	fig = CM.Figure(size=(1100, 600), fontsize=14)
	ax = CM.Axis(fig[1,1], xlabel="True Count Rate (Mcps)", ylabel="Recorded Count Rate (Mcps)",
		title="Dead-Time Models (τ = 10 ns)")
	CM.lines!(ax, true_rates./1e6, rec_nonpar./1e6, linewidth=2.5, color=:blue, label="Nonparalyzable")
	CM.lines!(ax, true_rates./1e6, rec_par./1e6, linewidth=2.5, color=:red, label="Paralyzable")
	CM.lines!(ax, true_rates./1e6, rec_semi./1e6, linewidth=3, color=:green, label="Seminonparalyzable (Yang 2025)")
	CM.lines!(ax, true_rates./1e6, true_rates./1e6, linewidth=1.5, color=:gray, linestyle=:dash, label="Ideal")
	CM.axislegend(ax, position=:rb)
	CM.save(joinpath(FIGURES_DIR, "nb04_count_rate_curves.png"), fig, px_per_unit=3)
	fig
end

# ╔═╡ a0000005-0072-4000-8000-000000000072
let
	aτ_range = collect(range(0.001, 2.0, length=500))
	vmr_nonpar = [BS.pileup_vmr_prediction(at; model=:nonparalyzable) for at in aτ_range]
	vmr_semi = [BS.pileup_vmr_prediction(at; model=:seminonparalyzable) for at in aτ_range]
	vmr_binned = [BS.binned_pixel_vmr(at, 4, 0.12) for at in aτ_range]

	fig = CM.Figure(size=(1100, 600), fontsize=14)
	ax = CM.Axis(fig[1,1], xlabel="aτ (count rate × dead time)", ylabel="Variance-to-Mean Ratio",
		title="Sub-Poisson Statistics Under Pileup")
	CM.lines!(ax, aτ_range, vmr_nonpar, linewidth=2.5, color=:blue, label="Nonparalyzable")
	CM.lines!(ax, aτ_range, vmr_semi, linewidth=3, color=:green, label="Seminonparalyzable")
	CM.lines!(ax, aτ_range, vmr_binned, linewidth=2.5, color=:orange, linestyle=:dash,
		label="2×2 binned (p_share=0.12)")
	CM.hlines!(ax, [1.0], color=:gray, linestyle=:dot, linewidth=1.5, label="Poisson (VMR=1)")
	CM.axislegend(ax, position=:rt)
	CM.ylims!(ax, 0.0, 1.5)
	CM.save(joinpath(FIGURES_DIR, "nb04_vmr_vs_flux.png"), fig, px_per_unit=3)
	fig
end

# ╔═╡ a0000005-0080-4000-8000-000000000080
md"""
## 7. Unified Detector Response Matrix (DRM)

Combines Fano noise + electronic noise + K-fluorescence escape + Hecht CCE into D[E, bin].
"""

# ╔═╡ a0000005-0081-4000-8000-000000000081
begin
	pcct_detector = BS.naeotom_detector_standard()
	D_matrix = BS.compute_unified_drm(pcct_detector, 140.0; n_energy_points=500)
	E_drm = BS.drm_energy_grid(140.0; n_energy_points=500)
end

# ╔═╡ a0000005-0082-4000-8000-000000000082
let
	fig = CM.Figure(size=(1400, 550), fontsize=14)

	ax1 = CM.Axis(fig[1,1], xlabel="Energy Bin", ylabel="Photon Energy (keV)",
		title="Unified DRM: D[E, bin]", xticks=1:4)
	CM.heatmap!(ax1, 1:4, E_drm, D_matrix', colormap=:inferno)

	ax2 = CM.Axis(fig[1,2], xlabel="Photon Energy (keV)", ylabel="Response (a.u.)",
		title="Per-Bin Response with 140 kVp Spectrum")
	spec_energies, spec_weights = BS.load_spectrum(140)
	S_interp = [begin
		idx = searchsortedfirst(spec_energies, E)
		idx = clamp(idx, 1, length(spec_weights))
		spec_weights[idx]
	end for E in E_drm]
	η_interp = [BS.quantum_efficiency(BS.CDTE_MATERIAL, 1.6, E) for E in E_drm]
	S_norm = S_interp .* η_interp; S_norm ./= maximum(S_norm)

	CM.band!(ax2, E_drm, zeros(length(E_drm)), S_norm .* 0.8, color=(:gray, 0.2))
	colors = [:blue, :green, :orange, :red]
	bin_labels_drm = ["Bin 1 (20-35)", "Bin 2 (35-55)", "Bin 3 (55-70)", "Bin 4 (70+)"]
	for b in 1:4
		CM.lines!(ax2, E_drm, D_matrix[:, b] .* S_norm, linewidth=2.5, color=colors[b], label=bin_labels_drm[b])
	end
	CM.vlines!(ax2, [20.0, 35.0, 55.0, 70.0], color=:gray, linestyle=:dot, linewidth=1.5)
	CM.axislegend(ax2, position=:rt, labelsize=11)
	CM.save(joinpath(FIGURES_DIR, "nb04_unified_drm.png"), fig, px_per_unit=3)
	fig
end

# ╔═╡ a0000005-0083-4000-8000-000000000083
let
	E_range = collect(10.0:0.25:140.0)
	σ_physics = [BS.physics_energy_resolution_keV(E) for E in E_range]
	fwhm_physics = σ_physics .* 2.355

	fig = CM.Figure(size=(1100, 600), fontsize=14)
	ax = CM.Axis(fig[1,1], xlabel="Photon Energy (keV)", ylabel="FWHM (keV)",
		title="Energy Resolution: Physics-Based (v24.0) vs Ad-Hoc")
	CM.lines!(ax, E_range, fwhm_physics, linewidth=3, color=:blue, label="v24.0: Fano + electronic")
	CM.lines!(ax, E_range, fill(10.0, length(E_range)), linewidth=2.5, color=:red,
		linestyle=:dash, label="Old: fixed 10 keV FWHM")
	idx_60keV = searchsortedfirst(E_range, 60.0)
	CM.text!(ax, 70.0, fwhm_physics[idx_60keV] + 0.5, text="3.55 keV at 60 keV", fontsize=13, color=:blue)
	CM.axislegend(ax, position=:lt)
	CM.ylims!(ax, 0.0, 12.0)
	CM.save(joinpath(FIGURES_DIR, "nb04_energy_resolution.png"), fig, px_per_unit=3)
	fig
end

# ╔═╡ a0000005-0013-4000-8000-000000000013
md"## 8. Phantom & Protocol Setup"

# ╔═╡ d1d7870b-d0bd-4010-847e-5af71edb7fc1
begin
	phantom_cpu = BS.create_gammex_472(
		n_voxels=SIM_CONFIG.imageSize,
		n_slices=SIM_CONFIG.sliceCount,
		fov_cm=SIM_CONFIG.fov_mm / 10.0,
		z_cm=(SIM_CONFIG.sliceCount * SIM_CONFIG.sliceThickness) / 10.0
	)

	phantom_gpu = BS.Phantom(
		MtlArray(phantom_cpu.mask),
		phantom_cpu.materials,
		phantom_cpu.voxel_size,
		phantom_cpu.origin,
		phantom_cpu.fov
	)
end

# ╔═╡ a0000005-0014-4000-8000-000000000014
let
	fig = CM.Figure(size=(800, 800), fontsize=14)
	ax = CM.Axis(fig[1,1], title="Gammex 472 (Central Slice)", aspect=CM.DataAspect())
	CM.heatmap!(ax, Array(phantom_gpu.mask[:, :, size(phantom_gpu.mask, 3) ÷ 2]), colormap=:viridis)
	CM.hidedecorations!(ax)
	fig
end

# ╔═╡ 17efb817-98c6-4c96-86df-5fb74d7bf44e
# Matches notebook 06: single 140 kVp, 984 views
protocol_pcct = BS.CTProtocol(
	kVp = 140.0,
	mA = 300.0,
	views = SIM_CONFIG.viewsPerRotation,
	rotation_time = SIM_CONFIG.rotationTime
)

# ╔═╡ bb6f40e3-ef82-4c0c-80a8-121a6019ecec
# Matches notebook 06: fidelity=:high, pcct_noise_reduction=0.60
sim_opts_pcct = BS.SimOptions(
	fidelity = :high,
	pcct_noise_reduction = 0.60,
	seed = 42
)

# ╔═╡ e3e97465-801a-4b4a-a177-c67b7dd99da9
recon_opts_pcct = BS.ReconOptions(
	algorithm = :fdk,
	matrix_size = (SIM_CONFIG.imageSize, SIM_CONFIG.imageSize, SIM_CONFIG.sliceCount),
	fov_cm = SIM_CONFIG.fov_mm / 10.0,
	filter = :standard,
	vmi_energies = [40.0, 70.0, 100.0, 140.0],
	vmi_basis = [:water, :iodine, :calcium]
)

# ╔═╡ a0000005-0090-4000-8000-000000000090
md"## 9. Water Phantom Calibration"

# ╔═╡ a0000005-0091-4000-8000-000000000091
# ╠═╡ show_logs = false
# Empirical μ_water for HU conversion — matches notebook 06 pattern
μ_water_pcct = let
	water_size = (512, 512, 16)
	water_voxel_cm = (0.03, 0.03, 0.1)
	water_mask = zeros(UInt8, water_size...)
	cx_w, cy_w = water_size[1] ÷ 2, water_size[2] ÷ 2
	r_w = 200  # ~6cm radius
	for i in 1:water_size[1], j in 1:water_size[2]
		if (i - cx_w)^2 + (j - cy_w)^2 <= r_w^2
			water_mask[i, j, :] .= UInt8(1)
		end
	end
	air_mat = XA.Material("Air", 0.499, 85.7u"eV", 0.001205u"g/cm^3",
		Dict(7 => 0.7553, 8 => 0.2318, 18 => 0.0129))
	water_materials = Dict(0 => air_mat, 1 => XA.Materials.water)
	water_phantom = BS.Phantom(MtlArray(water_mask), water_materials, water_voxel_cm)

	water_recon = BS.ReconOptions(algorithm=:fdk, matrix_size=(256, 256, 16), fov_cm=15.0)

	ws = BS.create_workspace(naeotom, protocol_pcct, sim_opts_pcct, water_recon, water_phantom)
	BS.simulate!(ws, water_phantom, naeotom, protocol_pcct, sim_opts_pcct, water_recon)
	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, water_recon.matrix_size)
	vol = Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, water_recon.matrix_size))

	# Extract central ROI
	nx, ny, nz = size(vol)
	cx, cy = nx ÷ 2, ny ÷ 2
	r = nx ÷ 10
	vals = Float64[]
	for k in (nz÷4):(3*nz÷4), j in (cy-r):(cy+r), i in (cx-r):(cx+r)
		if (i - cx)^2 + (j - cy)^2 <= r^2
			push!(vals, vol[i, j, k])
		end
	end
	result = mean(vals)

	ws_fdk = nothing; ws = nothing; water_phantom = nothing; GC.gc(true)
	result
end

# ╔═╡ a0000005-0092-4000-8000-000000000092
md"**μ\_water (PCCT 140 kVp):** $(round(μ_water_pcct, sigdigits=4)) cm⁻¹ (expected ~0.19-0.21)"

# ╔═╡ a0000005-0015-4000-8000-000000000015
md"## 10. PCCT Simulation"

# ╔═╡ 3f00dc43-62b6-4f38-8086-30506cbd1e80
# ╠═╡ show_logs = false
# Matches notebook 06: create_workspace → simulate! → reconstruct!
(pcct_fdk_hu, pcct_hir_hu, pcct_vmi_volumes, pcct_bin_recons) = let
	recon_size = recon_opts_pcct.matrix_size

	# --- Simulate ---
	ws = BS.create_workspace(naeotom, protocol_pcct, sim_opts_pcct, recon_opts_pcct, phantom_gpu)
	result = BS.simulate!(ws, phantom_gpu, naeotom, protocol_pcct, sim_opts_pcct, recon_opts_pcct)

	geom = ws.geom
	combined_sino = ws.combined
	vmi_sino_buf = ws.vmi_sino
	mat_map = result.mat_map
	pcct_sino = result.pcct_sino

	# --- FDK → CPU HU ---
	ws_fdk = BS.create_fdk_recon_workspace(combined_sino, geom, recon_size)
	fdk_hu = BS.to_hounsfield(
		Array(BS.reconstruct!(ws_fdk, combined_sino, geom, recon_size));
		μ_water=μ_water_pcct
	)
	ws_fdk = nothing; GC.gc(true)

	# --- HIR (strength 3) → CPU HU ---
	ws_hir = BS.create_hir_recon_workspace(combined_sino, geom, recon_size; strength=3)
	hir_hu = BS.to_hounsfield(
		Array(BS.reconstruct!(ws_hir, combined_sino, geom, recon_size));
		μ_water=μ_water_pcct
	)
	ws_hir = nothing; GC.gc(true)

	# --- VMI volumes (one at a time, GC between each) ---
	vmi_dict = Dict{Float64, Array{Float32, 3}}()
	if mat_map !== nothing
		for E in [40.0, 70.0, 100.0, 140.0]
			vmi_sino = BS.virtual_monoenergetic(mat_map, E; ws_output=vmi_sino_buf)
			ws_fdk_vmi = BS.create_fdk_recon_workspace(vmi_sino, geom, recon_size)
			vmi_dict[E] = Array(BS.reconstruct!(ws_fdk_vmi, vmi_sino, geom, recon_size))
			ws_fdk_vmi = nothing; GC.gc(true)
		end
	end

	# --- Per-bin reconstructions ---
	bin_vols = Vector{Array{Float32, 3}}()
	for b in 1:4
		vol = Array(BS.fdk_reconstruct(pcct_sino.bins[b], geom, recon_size))
		push!(bin_vols, vol)
		GC.gc(true)
	end

	# --- Cleanup ---
	ws = nothing; result = nothing; combined_sino = nothing
	vmi_sino_buf = nothing; mat_map = nothing; pcct_sino = nothing
	GC.gc(true)

	(fdk_hu, hir_hu, vmi_dict, bin_vols)
end

# ╔═╡ a0000005-0100-4000-8000-000000000100
md"## 11. FDK vs Hybrid IR Comparison"

# ╔═╡ a0000005-0101-4000-8000-000000000101
let
	mid_z = SIM_CONFIG.sliceCount ÷ 2
	window = (-200, 500)

	fig = CM.Figure(size=(1400, 600), fontsize=14)
	ax1 = CM.Axis(fig[1, 1], title="FDK", aspect=CM.DataAspect())
	hm = CM.heatmap!(ax1, pcct_fdk_hu[:, :, mid_z], colormap=:grays, colorrange=window)
	CM.hidedecorations!(ax1)

	ax2 = CM.Axis(fig[1, 2], title="Hybrid IR (Strength 3)", aspect=CM.DataAspect())
	CM.heatmap!(ax2, pcct_hir_hu[:, :, mid_z], colormap=:grays, colorrange=window)
	CM.hidedecorations!(ax2)

	CM.Colorbar(fig[1, 3], hm, label="HU")
	CM.Label(fig[0, :], text="PCCT NAEOTOM Alpha: FDK vs HIR-3 (140 kVp)", fontsize=16, font=:bold)

	CM.save(joinpath(FIGURES_DIR, "nb04_fdk_vs_hir.png"), fig, px_per_unit=3)
	fig
end

# ╔═╡ a0000005-0102-4000-8000-000000000102
# Noise comparison
let
	cx, cy, cz = SIM_CONFIG.imageSize ÷ 2, SIM_CONFIG.imageSize ÷ 2, SIM_CONFIG.sliceCount ÷ 2
	r = 40

	fdk_roi = [pcct_fdk_hu[i, j, cz] for i in (cx-r):(cx+r) for j in (cy-r):(cy+r)
		if (i-cx)^2 + (j-cy)^2 <= r^2]
	hir_roi = [pcct_hir_hu[i, j, cz] for i in (cx-r):(cx+r) for j in (cy-r):(cy+r)
		if (i-cx)^2 + (j-cy)^2 <= r^2]

	noise_fdk = std(fdk_roi)
	noise_hir = std(hir_roi)
	reduction = 100.0 * (1.0 - noise_hir / noise_fdk)

	fig = CM.Figure(size=(700, 550), fontsize=14)
	ax = CM.Axis(fig[1,1], title="Water ROI Noise", ylabel="σ (HU)", xticks=(1:2, ["FDK", "HIR-3"]))
	CM.barplot!(ax, [1, 2], [noise_fdk, noise_hir], color=[:coral, :steelblue])
	CM.text!(ax, 2, noise_hir + 0.5, text="-$(round(Int, reduction))%",
		align=(:center, :bottom), fontsize=14)
	CM.ylims!(ax, 0, noise_fdk * 1.2)

	CM.save(joinpath(FIGURES_DIR, "nb04_noise_comparison.png"), fig, px_per_unit=3)
	fig
end

# ╔═╡ a0000005-0110-4000-8000-000000000110
md"## 12. Per-Bin Reconstructions"

# ╔═╡ a0000005-0111-4000-8000-000000000111
let
	bin_labels = ["Bin 1: 20-35 keV", "Bin 2: 35-55 keV", "Bin 3: 55-70 keV", "Bin 4: 70+ keV"]
	mid_z = SIM_CONFIG.sliceCount ÷ 2

	fig = CM.Figure(size=(1600, 450), fontsize=14)
	for (i, label) in enumerate(bin_labels)
		ax = CM.Axis(fig[1, i], title=label, titlesize=13, aspect=CM.DataAspect())
		CM.heatmap!(ax, pcct_bin_recons[i][:, :, mid_z], colormap=:grays)
		CM.hidedecorations!(ax)
	end
	CM.save(joinpath(FIGURES_DIR, "nb04_4bin_reconstructions.png"), fig, px_per_unit=3)
	fig
end

# ╔═╡ a0000005-0120-4000-8000-000000000120
md"## 13. VMI Sweep"

# ╔═╡ a0000005-0121-4000-8000-000000000121
let
	vmi_E = [40.0, 70.0, 100.0, 140.0]
	mid_z = SIM_CONFIG.sliceCount ÷ 2

	fig = CM.Figure(size=(1600, 450), fontsize=14)
	for (idx, E) in enumerate(vmi_E)
		ax = CM.Axis(fig[1, idx], title="VMI $(Int(E)) keV", titlesize=13, aspect=CM.DataAspect())
		if haskey(pcct_vmi_volumes, E)
			CM.heatmap!(ax, pcct_vmi_volumes[E][:, :, mid_z], colormap=:grays)
		end
		CM.hidedecorations!(ax)
	end
	CM.Label(fig[0, :], text="PCCT VMI Energy Sweep (Gammex 472)", fontsize=16, font=:bold)
	CM.save(joinpath(FIGURES_DIR, "nb04_vmi_montage.png"), fig, px_per_unit=3)
	fig
end

# ╔═╡ a0000005-0122-4000-8000-000000000122
let
	# ROI HU vs VMI energy using mask-based insert locations
	slice = SIM_CONFIG.sliceCount ÷ 2
	mask_cpu = Array(phantom_gpu.mask)
	mask_slice = mask_cpu[:, :, slice]

	roi_defs = [
		("Solid Water", UInt8(3)),
		("Ca 200 mg/cc", UInt8(12)),
		("Iodine 10 mg/mL", UInt8(24)),
	]

	fig = CM.Figure(size=(1100, 600), fontsize=14)
	ax = CM.Axis(fig[1,1], xlabel="VMI Energy (keV)", ylabel="HU",
		title="VMI Energy-Dependent Contrast (Mask-Based ROIs)")

	vmi_E = sort(collect(keys(pcct_vmi_volumes)))
	colors = [:blue, :red, :green]
	for (ci, (name, region_val)) in enumerate(roi_defs)
		roi_mask = mask_slice .== region_val
		!any(roi_mask) && continue

		hu_values = Float64[]
		for E in vmi_E
			vol_slice = pcct_vmi_volumes[E][:, :, slice]
			μ_roi = mean(vol_slice[roi_mask])
			μ_water_E = BS.get_water_attenuation_vmi(E)
			hu = 1000.0 * (μ_roi - μ_water_E) / μ_water_E
			push!(hu_values, hu)
		end
		CM.lines!(ax, vmi_E, hu_values, linewidth=3, color=colors[ci], label=name)
		CM.scatter!(ax, vmi_E, hu_values, color=colors[ci], markersize=8)
	end
	CM.hlines!(ax, [0.0], color=:gray, linestyle=:dash, linewidth=1.5)
	CM.axislegend(ax, position=:rt)
	CM.save(joinpath(FIGURES_DIR, "nb04_vmi_energy_curves.png"), fig, px_per_unit=3)
	fig
end

# ╔═╡ a0000005-0028-4000-8000-000000000028
md"## 14. K-Edge Imaging"

# ╔═╡ a0000005-0029-4000-8000-000000000029
let
	mid_z = SIM_CONFIG.sliceCount ÷ 2
	vol_below = pcct_bin_recons[1]  # 20-35 keV
	vol_above = pcct_bin_recons[2]  # 35-55 keV
	kedge_map = vol_above .- vol_below

	fig = CM.Figure(size=(1400, 500), fontsize=14)
	ax1 = CM.Axis(fig[1,1], title="Below I K-edge\n(20-35 keV)", aspect=CM.DataAspect())
	CM.heatmap!(ax1, vol_below[:, :, mid_z], colormap=:grays)
	CM.hidedecorations!(ax1)

	ax2 = CM.Axis(fig[1,2], title="Above I K-edge\n(35-55 keV)", aspect=CM.DataAspect())
	CM.heatmap!(ax2, vol_above[:, :, mid_z], colormap=:grays)
	CM.hidedecorations!(ax2)

	ax3 = CM.Axis(fig[1,3], title="K-edge Enhancement\n(Above - Below)", aspect=CM.DataAspect())
	CM.heatmap!(ax3, kedge_map[:, :, mid_z], colormap=:jet)
	CM.hidedecorations!(ax3)

	CM.save(joinpath(FIGURES_DIR, "nb04_kedge_imaging.png"), fig, px_per_unit=3)
	fig
end

# ╔═╡ a0000005-0030-4000-8000-000000000030
md"## 15. Summary"

# ╔═╡ a0000005-0031-4000-8000-000000000031
# md"""
# ### PCCT Verification Summary

# This notebook uses the **same scanner/protocol/SimOptions as notebook 06** (XCAT):
# - **Scanner**: NAEOTOM Alpha (SID=595, SDD=1085.5, CdTe 0.4mm dexels, 4 energy bins)
# - **Protocol**: 140 kVp, 300 mA, 984 views
# - **SimOptions**: `fidelity=:high`, `pcct_noise_reduction=0.60`
# - **Phantom**: Gammex 472 ($(SIM_CONFIG.imageSize)×$(SIM_CONFIG.imageSize)×$(SIM_CONFIG.sliceCount))

# ### Workspace API Pattern (matches notebook 06)
# ```julia
# ws     = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
# result = BS.simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)
# # Combined sinogram: ws.combined
# # Energy bins: result.pcct_sino.bins
# # Material maps: result.mat_map
# # VMI: BS.virtual_monoenergetic(result.mat_map, E; ws_output=ws.vmi_sino)
# ```

# ### Physics Models Verified
# | Model | Reference | Key Result |
# |-------|-----------|------------|
# | Charge cloud | Koch-Mehrin 2020 | σ ≈ 12-14 μm (depth-averaged) |
# | K-fluorescence | Koch-Mehrin Table 1 | 5 K-lines/element, Te→Cd cascade |
# | CCE | Hecht + Barrett 1995 | w/L ≈ 0.17 (small-pixel regime) |
# | Pileup | Yang 2025 | Seminonparalyzable, VMR sub-Poisson |
# | DRM | Unified v24.0 | FWHM = 3.55 keV at 60 keV |
# | Energy resolution | Fano + electronic | Replaces ad-hoc 10 keV FWHM |
# """

# ╔═╡ a0000005-0032-4000-8000-000000000032
md"### Figures saved to `$(FIGURES_DIR)`"

# ╔═╡ Cell order:
# ╟─a0000005-0007-4000-8000-000000000007
# ╟─a0000005-0008-4000-8000-000000000008
# ╠═a0000005-0001-4000-8000-000000000001
# ╠═a0000005-0002-4000-8000-000000000002
# ╠═a0000005-0003-4000-8000-000000000003
# ╠═a0000005-0004-4000-8000-000000000004
# ╠═a0000005-0005-4000-8000-000000000005
# ╠═b0000004-0001-0001-0001-000000000091
# ╠═b0000004-0001-0001-0001-000000000090
# ╠═a0000005-0006-4000-8000-000000000006
# ╠═a0000005-0009-4000-8000-000000000009
# ╟─a0000005-0010-4000-8000-000000000010
# ╠═a0000005-0010-4000-8000-000000000099
# ╠═c9636083-f2fe-451c-ae38-077eb7cd19e7
# ╟─a0000005-0011-4000-8000-000000000011
# ╟─a0000005-0012-4000-8000-000000000012
# ╟─a0000005-0040-4000-8000-000000000040
# ╟─a0000005-0041-4000-8000-000000000041
# ╟─a0000005-0042-4000-8000-000000000042
# ╟─a0000005-0043-4000-8000-000000000043
# ╟─a0000005-0050-4000-8000-000000000050
# ╟─a0000005-0051-4000-8000-000000000051
# ╟─a0000005-0060-4000-8000-000000000060
# ╟─a0000005-0061-4000-8000-000000000061
# ╟─a0000005-0062-4000-8000-000000000062
# ╟─a0000005-0070-4000-8000-000000000070
# ╟─a0000005-0071-4000-8000-000000000071
# ╟─a0000005-0072-4000-8000-000000000072
# ╟─a0000005-0080-4000-8000-000000000080
# ╠═a0000005-0081-4000-8000-000000000081
# ╟─a0000005-0082-4000-8000-000000000082
# ╟─a0000005-0083-4000-8000-000000000083
# ╟─a0000005-0013-4000-8000-000000000013
# ╠═d1d7870b-d0bd-4010-847e-5af71edb7fc1
# ╟─a0000005-0014-4000-8000-000000000014
# ╠═17efb817-98c6-4c96-86df-5fb74d7bf44e
# ╠═bb6f40e3-ef82-4c0c-80a8-121a6019ecec
# ╠═e3e97465-801a-4b4a-a177-c67b7dd99da9
# ╟─a0000005-0090-4000-8000-000000000090
# ╠═a0000005-0091-4000-8000-000000000091
# ╟─a0000005-0092-4000-8000-000000000092
# ╟─a0000005-0015-4000-8000-000000000015
# ╠═3f00dc43-62b6-4f38-8086-30506cbd1e80
# ╟─a0000005-0100-4000-8000-000000000100
# ╟─a0000005-0101-4000-8000-000000000101
# ╟─a0000005-0102-4000-8000-000000000102
# ╟─a0000005-0110-4000-8000-000000000110
# ╟─a0000005-0111-4000-8000-000000000111
# ╟─a0000005-0120-4000-8000-000000000120
# ╟─a0000005-0121-4000-8000-000000000121
# ╟─a0000005-0122-4000-8000-000000000122
# ╟─a0000005-0028-4000-8000-000000000028
# ╟─a0000005-0029-4000-8000-000000000029
# ╟─a0000005-0030-4000-8000-000000000030
# ╟─a0000005-0031-4000-8000-000000000031
# ╟─a0000005-0032-4000-8000-000000000032
