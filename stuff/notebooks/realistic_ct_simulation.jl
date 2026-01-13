### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 00000001-0000-0000-0000-000000000001
begin
	using Pkg
	Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ 00000001-0000-0000-0000-000000000002
begin
	using BasisSimulator
	using Reactant
	using Statistics
	using CairoMakie
end

# ╔═╡ 00000001-0000-0000-0000-000000000003
md"""
# Realistic CT Simulation with BasisSimulator.jl

This notebook demonstrates physically realistic CT simulation using:

1. **High-resolution phantom** - Physical objects at higher resolution than CT voxels
2. **Clinical scanner configuration** - GE Revolution Apex Elite (FDA 510(k): K213715)
3. **Multiple acquisition protocols** - Different kVp/mAs combinations
4. **Full physics pipeline** - Polychromatic X-rays, filters, scatter, noise
5. **Multiple reconstruction methods** - FDK (analytical), SIRT, CGLS (iterative)
6. **Reactant compilation** - XLA compilation for performance
"""

# ╔═╡ 00000001-0000-0000-0000-000000000004
md"""
## 1. Configuration
"""

# ╔═╡ 00000001-0000-0000-0000-000000000005
PHANTOM_SIZE = 128

# ╔═╡ 00000001-0000-0000-0000-000000000006
RECON_SIZE = 64

# ╔═╡ 00000001-0000-0000-0000-000000000007
RECON_SLICES = 8

# ╔═╡ 00000001-0000-0000-0000-000000000008
N_ANGLES = 90

# ╔═╡ 00000001-0000-0000-0000-000000000009
N_DETECTOR_ROWS = 8

# ╔═╡ 00000001-0000-0000-0000-000000000010
N_DETECTOR_COLS = 128

# ╔═╡ 00000001-0000-0000-0000-000000000011
SIRT_ITERATIONS = 10

# ╔═╡ 00000001-0000-0000-0000-000000000012
CGLS_ITERATIONS = 5

# ╔═╡ 00000001-0000-0000-0000-000000000013
md"""
## 2. High-Resolution Phantom
"""

# ╔═╡ 00000001-0000-0000-0000-000000000014
phantom = create_gammex_472(n_voxels=PHANTOM_SIZE, z_cm=2.0)

# ╔═╡ 00000001-0000-0000-0000-000000000015
μ_water_ref = get_reference_μ_water(60.0)

# ╔═╡ 00000001-0000-0000-0000-000000000016
md"""
Phantom size: $(size(phantom.μ))
"""

# ╔═╡ 00000001-0000-0000-0000-000000000017
let
	mid_slice = size(phantom.μ, 3) ÷ 2
	fig = Figure(size=(700, 250))

	ax1 = Axis(fig[1, 1], aspect=DataAspect(), title="Regions")
	heatmap!(ax1, Float64.(phantom.mask[:, :, mid_slice])'; colormap=:tab20)
	hidedecorations!(ax1)

	ax2 = Axis(fig[1, 2], aspect=DataAspect(), title="Attenuation")
	hm2 = heatmap!(ax2, phantom.μ[:, :, mid_slice]'; colormap=:viridis)
	Colorbar(fig[1, 3], hm2, label="μ (cm⁻¹)")
	hidedecorations!(ax2)

	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000018
md"""
## 3. Scanner Configuration
"""

# ╔═╡ 00000001-0000-0000-0000-000000000019
scanner_spec = GERevolutionApexElite()

# ╔═╡ 00000001-0000-0000-0000-000000000020
geom = create_geometry(scanner_spec;
	n_angles=N_ANGLES,
	n_rows=N_DETECTOR_ROWS,
	n_cols=N_DETECTOR_COLS,
	fov_cm=phantom.fov[1]
)

# ╔═╡ 00000001-0000-0000-0000-000000000021
md"""
SAD: $(geom.SAD) cm, SDD: $(geom.SDD) cm, Detector: $(geom.n_cols)×$(geom.n_rows)
"""

# ╔═╡ 00000001-0000-0000-0000-000000000022
md"""
## 4. Acquisition Protocols
"""

# ╔═╡ 00000001-0000-0000-0000-000000000023
protocols = [
	(name="Low", kvp=80, I0=1e5),
	(name="Standard", kvp=120, I0=2e5),
	(name="High", kvp=140, I0=4e5),
]

# ╔═╡ 00000001-0000-0000-0000-000000000024
md"""
## 5. Geometry Pre-computation
"""

# ╔═╡ 00000001-0000-0000-0000-000000000025
output_size = (RECON_SIZE, RECON_SIZE, RECON_SLICES)

# ╔═╡ 00000001-0000-0000-0000-000000000026
output_fov = phantom.fov

# ╔═╡ 00000001-0000-0000-0000-000000000027
output_voxel_size = output_fov ./ output_size

# ╔═╡ 00000001-0000-0000-0000-000000000028
proj_geom = precompute_projection_geometry(
	geom, phantom.fov, phantom.voxel_size, size(phantom.μ)
)

# ╔═╡ 00000001-0000-0000-0000-000000000029
bp_geom = precompute_backprojection_geometry(geom, output_size, output_fov)

# ╔═╡ 00000001-0000-0000-0000-000000000030
proj_geom_recon = precompute_projection_geometry(
	geom, output_fov, output_voxel_size, output_size
)

# ╔═╡ 00000001-0000-0000-0000-000000000031
md"""
## 6. Forward Projection
"""

# ╔═╡ 00000001-0000-0000-0000-000000000032
phantom_ra = Reactant.to_rarray(Float32.(phantom.μ))

# ╔═╡ 00000001-0000-0000-0000-000000000033
compiled_project = @compile project_volume(phantom_ra, proj_geom)

# ╔═╡ 00000001-0000-0000-0000-000000000034
sino_mono = Array(compiled_project(phantom_ra, proj_geom))

# ╔═╡ 00000001-0000-0000-0000-000000000035
md"""
Sinogram: $(size(sino_mono))
"""

# ╔═╡ 00000001-0000-0000-0000-000000000036
let
	fig = Figure(size=(400, 200))
	ax = Axis(fig[1, 1], xlabel="Column", ylabel="Angle", title="Sinogram")
	heatmap!(ax, sino_mono[:, N_DETECTOR_ROWS÷2, :]'; colormap=:hot)
	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000037
md"""
## 7. Physical Effects
"""

# ╔═╡ 00000001-0000-0000-0000-000000000038
flat_filter = flat_filter_al_cu(2.5, 0.1)

# ╔═╡ 00000001-0000-0000-0000-000000000039
bowtie = bowtie_filter_medium_body()

# ╔═╡ 00000001-0000-0000-0000-000000000040
scatter_model = default_scatter_model(scale_factor=1.0)

# ╔═╡ 00000001-0000-0000-0000-000000000041
sinograms = let
	result = Dict{String, Array{Float32,3}}()
	for p in protocols
		proj = create_polychromatic_projector(phantom, geom, p.kvp; n_bins=10)
		sino = forward_project_polychromatic(phantom, proj)
		sino = apply_flat_filter(sino, flat_filter, geom)
		sino = apply_bowtie_filter(sino, bowtie, geom)
		sino = add_scatter(sino, scatter_model)
		det = default_detector_model(blur_fwhm=0.0, I0=p.I0, electronic_noise_std=20.0, seed=42)
		sino = add_quantum_noise(sino, det)
		result[p.name] = Float32.(sino)
	end
	result
end

# ╔═╡ 00000001-0000-0000-0000-000000000042
md"""
## 8. FDK Reconstruction
"""

# ╔═╡ 00000001-0000-0000-0000-000000000043
test_sino_ra = Reactant.to_rarray(sinograms[protocols[1].name])

# ╔═╡ 00000001-0000-0000-0000-000000000044
compiled_fdk = @compile fdk_reconstruct_xla(test_sino_ra, geom, bp_geom)

# ╔═╡ 00000001-0000-0000-0000-000000000045
recon_fdk = let
	result = Dict{String, Array{Float32,3}}()
	for p in protocols
		sino_ra = Reactant.to_rarray(sinograms[p.name])
		result[p.name] = Array(compiled_fdk(sino_ra, geom, bp_geom))
	end
	result
end

# ╔═╡ 00000001-0000-0000-0000-000000000046
let
	mid = RECON_SLICES ÷ 2
	fig = Figure(size=(800, 200))
	for (i, p) in enumerate(protocols)
		ax = Axis(fig[1, i], aspect=DataAspect(), title="FDK: $(p.name)")
		hm = heatmap!(ax, μ_to_HU(recon_fdk[p.name][:,:,mid], μ_water_ref)';
			colormap=:grays, colorrange=(-200, 500))
		hidedecorations!(ax)
		i == length(protocols) && Colorbar(fig[1, 4], hm, label="HU")
	end
	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000047
md"""
## 9. SIRT Reconstruction
"""

# ╔═╡ 00000001-0000-0000-0000-000000000048
recon_sirt = let
	result = Dict{String, Array{Float32,3}}()
	for p in protocols
		r = sirt_reconstruct(sinograms[p.name], proj_geom_recon, bp_geom;
			n_iterations=SIRT_ITERATIONS, verbose=false)
		result[p.name] = r.volume
	end
	result
end

# ╔═╡ 00000001-0000-0000-0000-000000000049
let
	mid = RECON_SLICES ÷ 2
	fig = Figure(size=(800, 200))
	for (i, p) in enumerate(protocols)
		ax = Axis(fig[1, i], aspect=DataAspect(), title="SIRT: $(p.name)")
		hm = heatmap!(ax, μ_to_HU(recon_sirt[p.name][:,:,mid], μ_water_ref)';
			colormap=:grays, colorrange=(-200, 500))
		hidedecorations!(ax)
		i == length(protocols) && Colorbar(fig[1, 4], hm, label="HU")
	end
	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000050
md"""
## 10. CGLS Reconstruction
"""

# ╔═╡ 00000001-0000-0000-0000-000000000051
recon_cgls = let
	result = Dict{String, Array{Float32,3}}()
	for p in protocols
		r = cgls_reconstruct(sinograms[p.name], proj_geom_recon, bp_geom;
			n_iterations=CGLS_ITERATIONS, verbose=false)
		result[p.name] = r.volume
	end
	result
end

# ╔═╡ 00000001-0000-0000-0000-000000000052
let
	mid = RECON_SLICES ÷ 2
	fig = Figure(size=(800, 200))
	for (i, p) in enumerate(protocols)
		ax = Axis(fig[1, i], aspect=DataAspect(), title="CGLS: $(p.name)")
		hm = heatmap!(ax, μ_to_HU(recon_cgls[p.name][:,:,mid], μ_water_ref)';
			colormap=:grays, colorrange=(-200, 500))
		hidedecorations!(ax)
		i == length(protocols) && Colorbar(fig[1, 4], hm, label="HU")
	end
	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000053
md"""
## 11. Comparison Grid
"""

# ╔═╡ 00000001-0000-0000-0000-000000000054
let
	mid = RECON_SLICES ÷ 2
	fig = Figure(size=(800, 700))
	methods = ["FDK", "SIRT", "CGLS"]
	recons = [recon_fdk, recon_sirt, recon_cgls]

	for (j, m) in enumerate(methods)
		for (i, p) in enumerate(protocols)
			ax = Axis(fig[j, i], aspect=DataAspect())
			hm = heatmap!(ax, μ_to_HU(recons[j][p.name][:,:,mid], μ_water_ref)';
				colormap=:grays, colorrange=(-200, 500))
			hidedecorations!(ax)
			j == 1 && (ax.title = "$(p.kvp) kVp")
			i == 1 && (ax.ylabel = m)
		end
	end

	Colorbar(fig[2, 4], limits=(-200, 500), colormap=:grays, label="HU")
	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000055
md"""
## 12. Quantitative Analysis
"""

# ╔═╡ 00000001-0000-0000-0000-000000000056
let
	c = RECON_SIZE ÷ 2
	r = RECON_SIZE ÷ 8
	s = RECON_SLICES ÷ 2

	results = []
	for p in protocols
		for (m, rd) in [("FDK", recon_fdk), ("SIRT", recon_sirt), ("CGLS", recon_cgls)]
			roi = rd[p.name][c-r:c+r, c-r:c+r, s]
			roi_HU = μ_to_HU(roi, μ_water_ref)
			push!(results, (p=p.name, kvp=p.kvp, m=m,
				mean=round(mean(roi_HU), digits=1),
				std=round(std(roi_HU), digits=1)))
		end
	end

	md"""
	| Protocol | kVp | Method | Mean HU | Noise |
	|----------|-----|--------|---------|-------|
	$(join(["| $(r.p) | $(r.kvp) | $(r.m) | $(r.mean) | $(r.std) |" for r in results], "\n"))
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000057
md"""
## Summary

- **Noise decreases with mAs** (higher I₀)
- **Beam hardening varies with kVp**
- **Iterative methods** can reduce noise vs FDK
- **CGLS** converges faster than SIRT

---
*Generated with BasisSimulator.jl*
"""

# ╔═╡ Cell order:
# ╟─00000001-0000-0000-0000-000000000001
# ╟─00000001-0000-0000-0000-000000000002
# ╟─00000001-0000-0000-0000-000000000003
# ╟─00000001-0000-0000-0000-000000000004
# ╠═00000001-0000-0000-0000-000000000005
# ╠═00000001-0000-0000-0000-000000000006
# ╠═00000001-0000-0000-0000-000000000007
# ╠═00000001-0000-0000-0000-000000000008
# ╠═00000001-0000-0000-0000-000000000009
# ╠═00000001-0000-0000-0000-000000000010
# ╠═00000001-0000-0000-0000-000000000011
# ╠═00000001-0000-0000-0000-000000000012
# ╟─00000001-0000-0000-0000-000000000013
# ╠═00000001-0000-0000-0000-000000000014
# ╠═00000001-0000-0000-0000-000000000015
# ╟─00000001-0000-0000-0000-000000000016
# ╠═00000001-0000-0000-0000-000000000017
# ╟─00000001-0000-0000-0000-000000000018
# ╠═00000001-0000-0000-0000-000000000019
# ╠═00000001-0000-0000-0000-000000000020
# ╟─00000001-0000-0000-0000-000000000021
# ╟─00000001-0000-0000-0000-000000000022
# ╠═00000001-0000-0000-0000-000000000023
# ╟─00000001-0000-0000-0000-000000000024
# ╠═00000001-0000-0000-0000-000000000025
# ╠═00000001-0000-0000-0000-000000000026
# ╠═00000001-0000-0000-0000-000000000027
# ╠═00000001-0000-0000-0000-000000000028
# ╠═00000001-0000-0000-0000-000000000029
# ╠═00000001-0000-0000-0000-000000000030
# ╟─00000001-0000-0000-0000-000000000031
# ╠═00000001-0000-0000-0000-000000000032
# ╠═00000001-0000-0000-0000-000000000033
# ╠═00000001-0000-0000-0000-000000000034
# ╟─00000001-0000-0000-0000-000000000035
# ╠═00000001-0000-0000-0000-000000000036
# ╟─00000001-0000-0000-0000-000000000037
# ╠═00000001-0000-0000-0000-000000000038
# ╠═00000001-0000-0000-0000-000000000039
# ╠═00000001-0000-0000-0000-000000000040
# ╠═00000001-0000-0000-0000-000000000041
# ╟─00000001-0000-0000-0000-000000000042
# ╠═00000001-0000-0000-0000-000000000043
# ╠═00000001-0000-0000-0000-000000000044
# ╠═00000001-0000-0000-0000-000000000045
# ╠═00000001-0000-0000-0000-000000000046
# ╟─00000001-0000-0000-0000-000000000047
# ╠═00000001-0000-0000-0000-000000000048
# ╠═00000001-0000-0000-0000-000000000049
# ╟─00000001-0000-0000-0000-000000000050
# ╠═00000001-0000-0000-0000-000000000051
# ╠═00000001-0000-0000-0000-000000000052
# ╟─00000001-0000-0000-0000-000000000053
# ╠═00000001-0000-0000-0000-000000000054
# ╟─00000001-0000-0000-0000-000000000055
# ╠═00000001-0000-0000-0000-000000000056
# ╟─00000001-0000-0000-0000-000000000057
