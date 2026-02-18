### A Pluto.jl notebook ###
# v0.20.21

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

# ╔═╡ 00000000-0000-0000-0000-000000000003
# ╠═╡ show_logs = false
begin
    using CairoMakie
    using Statistics
    using LinearAlgebra
    import BasisSimulator as BS
    import XrayAttenuation as XA
    import BasisSimulator.SemanticClassification as SC
end

# ╔═╡ 00000000-0000-0000-0000-000000000005
# ╠═╡ show_logs = false
begin
    # Phantom parameters
    global phantom_dims = (64, 64, 32)
    global phantom_voxel_size = (0.2, 0.2, 0.2)
end

# ╔═╡ 00000000-0000-0000-0000-000000000007
# ╠═╡ show_logs = false
begin
    # Perfusion parameters
    global time_points = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    global arterial_conc = [0.0, 0.5, 2.0, 5.0, 4.0, 2.0, 1.0, 0.5, 0.0]
    global venous_conc = [0.0, 0.0, 0.5, 2.0, 4.0, 5.0, 3.0, 1.0, 0.0]
    global energy_keV = 60.0
end

# ╔═╡ 00000000-0000-0000-0000-000000000009
# ╠═╡ show_logs = false
begin
    function create_test_phantom(dims, voxel_size)
        dtype = UInt8
        labeled = zeros(dtype, dims)
        
        cx, cy, cz = dims .÷ 2
        r_brain = minimum(dims) ÷ 8
        r_head = minimum(dims) ÷ 3
        
        for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
            dx, dy, dz = Float64(i - cx), Float64(j - cy), Float64(k - cz)
            r = sqrt(dx^2 + dy^2 + dz^2)
            
            if r <= r_brain
                labeled[i, j, k] = 2
            elseif r <= r_head
                labeled[i, j, k] = 1
            else
                labeled[i, j, k] = 0
            end
        end
        
        materials_dict = Dict{Int, XA.Material}()
        materials_dict[0] = XA.Materials.air
        materials_dict[1] = BS.get_material(:soft_tissue)
        materials_dict[2] = BS.get_material(:brain)
        
        phantom = BS.Phantom(labeled, materials_dict, voxel_size)
        return phantom
    end
    
    global phantom = create_test_phantom(phantom_dims, phantom_voxel_size)
    "Phantom created: $(size(phantom.mask))"
end

# ╔═╡ 00000000-0000-0000-0000-000000000011
# ╠═╡ show_logs = false
begin
    function run_perfusion_simulation(phantom, time_points, arterial_conc, venous_conc, energy_keV)
        results = Dict()
        
        for (t_idx, t) in enumerate(time_points)
            art_c = arterial_conc[t_idx]
            ven_c = venous_conc[t_idx]
            
            materials_contrast = Dict{Int, XA.Material}()
            
            for (id, mat) in zip(keys(phantom.materials), phantom.materials)
                if id == 2
                    if art_c > 0
                        materials_contrast[id] = BS.create_iodine_blood_mixture(mat, art_c)
                    else
                        materials_contrast[id] = mat
                    end
                else
                    materials_contrast[id] = mat
                end
            end
            
            contrast_phantom = BS.Phantom(phantom.mask, materials_contrast, phantom.voxel_size)
            mu = BS.compute_μ(contrast_phantom, energy_keV)
            
            results[t] = Dict(
                "time" => t,
                "arterial_conc" => art_c,
                "venous_conc" => ven_c,
                "mu" => mu,
                "phantom" => contrast_phantom
            )
        end
        
        return results
    end
    
    global perfusion_results = run_perfusion_simulation(phantom, time_points, arterial_conc, venous_conc, energy_keV)
    "Simulation complete: $(length(time_points)) time points"
end

# ╔═╡ 00000000-0000-0000-0000-000000000012
# ╠═╡ show_logs = false
begin
    global mu_water = BS.calculate_mixture_attenuation(XA.Materials.water, energy_keV)
    global dims = size(phantom.mask)
    global cz = dims[3] ÷ 2
    
    global mu_min = minimum(perfusion_results[t]["mu"] for t in time_points)
    global mu_max = maximum(perfusion_results[t]["mu"] for t in time_points)
end

# ╔═╡ 00000000-0000-0000-0000-000000000014
# ╠═╡ show_logs = false
begin
    "Time slider - select time point"
end

# ╔═╡ 00000000-0000-0000-0000-000000000015
# ╠═╡ show_logs = false
@bind selected_time Slider(1:length(time_points); default=5, show_value=true)

# ╔═╡ 00000000-0000-0000-0000-000000000016
# ╠═╡ show_logs = false
begin
    "Slice position slider"
end

# ╔═╡ 00000000-0000-0000-0000-000000000017
# ╠═╡ show_logs = false
@bind selected_slice Slider(1:dims[3]; default=cz, show_value=true)

# ╔═╡ 00000000-0000-0000-0000-000000000018
# ╠═╡ show_logs = false
begin
    t = time_points[selected_time]
    z = selected_slice
    
    mu_slice = perfusion_results[t]["mu"][:, :, z]
    hu_slice = 1000 .* (mu_slice .- mu_water) ./ mu_water
    
    fig = Figure(resolution=(600, 500))
    ax = Axis(fig[1, 1], title="HU Map: t=$(round(t, digits=1))s, z=$z")
    hm = heatmap!(ax, hu_slice; colormap=:viridis, colorrange=(-100, 100))
    Colorbar(fig[1, 2], hm)
    ax.xlabel = "X (voxel)"
    ax.ylabel = "Y (voxel)"
    fig
end

# ╔═╡ 00000000-0000-0000-0000-000000000019
# ╠═╡ show_logs = false
begin
    t = time_points[selected_time]
    mu_data = vec(perfusion_results[t]["mu"])
    
    fig = Figure(resolution=(500, 300))
    ax = Axis(fig[1, 1], title="Attenuation Distribution (t=$(round(t, digits=1))s)",
              xlabel="mu (1/cm)", ylabel="Count")
    hist!(ax, mu_data, bins=30)
    fig
end

# ╔═╡ 00000000-0000-0000-0000-000000000021
# ╠═╡ show_logs = false
begin
    cx, cy = dims[1]÷2, dims[2]÷2
    
    region_points = Dict(
        "Center" => (cx, cy, cz),
        "Left (+20)" => (cx-20, cy, cz),
        "Right (+20)" => (cx+20, cy, cz),
    )
    
    fig = Figure(resolution=(600, 400))
    ax = Axis(fig[1, 1], title="Time-Attenuation Curves", xlabel="Time (s)", ylabel="HU")
    
    colors = [:red, :blue, :green]
    
    for (idx, (name, (i, j, k))) in enumerate(region_points)
        hu_values = [1000 * (perfusion_results[t]["mu"][i, j, k] - mu_water) / mu_water for t in time_points]
        lines!(ax, time_points, hu_values; label=name, color=colors[idx], linewidth=2)
    end
    
    axislegend(ax)
    fig
end

# ╔═╡ 00000000-0000-0000-0000-000000000023
# ╠═╡ show_logs = false
begin
    selected_times = [time_points[1], time_points[length(time_points)÷2+1], time_points[end]]
    
    fig = Figure(resolution=(900, 300))
    
    for (idx, t) in enumerate(selected_times)
        ax = Axis(fig[1, idx], title="t=$(round(t, digits=1))s")
        
        mu_slice = perfusion_results[t]["mu"][:, :, cz]
        hu_slice = 1000 .* (mu_slice .- mu_water) ./ mu_water
        
        hm = heatmap!(ax, hu_slice; colormap=:viridis, colorrange=(-100, 100))
        
        if idx == 3
            Colorbar(fig[1, 4], hm)
        end
    end
    
    fig
end

# ╔═╡ 00000000-0000-0000-0000-000000000025
# ╠═╡ show_logs = false
begin
    fig = Figure(resolution=(1000, 700))
    
    ax1 = Axis(fig[1, 1], title="Mean Attenuation")
    mu_mean = mean(perfusion_results[t]["mu"] for t in time_points)
    hm1 = heatmap!(ax1, mu_mean[:, :, cz]; colormap=:viridis)
    Colorbar(fig[1, 2], hm1)
    
    ax2 = Axis(fig[1, 3], title="Max Change from Baseline")
    mu_baseline = perfusion_results[time_points[1]]["mu"]
    mu_change = maximum(perfusion_results[t]["mu"] .- mu_baseline for t in time_points)
    hm2 = heatmap!(ax2, mu_change[:, :, cz]; colormap=:magma)
    Colorbar(fig[1, 4], hm2)
    
    ax3 = Axis(fig[2, 1:2], title="TAC at Center", xlabel="Time (s)", ylabel="HU")
    hu_center = [1000 * (perfusion_results[t]["mu"][cx, cy, cz] - mu_water) / mu_water for t in time_points]
    lines!(ax3, time_points, hu_center; color=:red, linewidth=2)
    
    ax4 = Axis(fig[2, 3], title="Distribution of mu Changes")
    hist!(ax4, vec(mu_change), bins=30)
    
    ax5 = Axis(fig[2, 4], title="Material Composition")
    mask_data = vec(phantom.mask)
    unique_ids = unique(mask_data)
    counts = [count(==(id), mask_data) for id in unique_ids]
    pie!(ax5, counts, labels=["Air", "Soft Tissue", "Brain"][1:length(unique_ids)])
    
    fig
end

# ╔═╡ 00000000-0000-0000-0000-000000000027
# ╠═╡ show_logs = false
begin
    const MATERIAL_COLORS = Dict(
        0 => RGBf(0.0, 0.0, 0.0),
        1 => RGBf(0.8, 0.5, 0.5),
        2 => RGBf(0.9, 0.8, 0.7),
    )
    
    slice_data = phantom.mask[:, :, cz]
    dims_slice = size(slice_data)
    
    color_image = Matrix{RGBf}(undef, dims_slice[1], dims_slice[2])
    for j in 1:dims_slice[2], i in 1:dims_slice[1]
        color_image[i, j] = get(MATERIAL_COLORS, slice_data[i, j], RGBf(0.5, 0.5, 0.5))
    end
    
    fig = Figure(resolution=(400, 350))
    ax = Axis(fig[1, 1], title="Phantom Material Map (z=$cz)")
    image!(ax, color_image, aspect=:equal)
    ax.xlabel = "X (voxel)"
    ax.ylabel = "Y (voxel)"
    fig
end

# ╔═╡ 00000000-0000-0000-0000-000000000028
# ╠═╡ show_logs = false
begin
    "## Summary"
    Markdown.md"""
    This notebook demonstrates brain perfusion CT simulation with interactive visualization.
    
    **Features:**
    - Phantom generation with brain structures
    - Perfusion simulation over time
    - Interactive time/slice sliders
    - Time-attenuation curves
    - Summary dashboard
    """
end

# ╔═╡ cell_hash = "00000000-0000-0000-0000-000000000001"
# ╠═╡ skip_as_script = false
end
