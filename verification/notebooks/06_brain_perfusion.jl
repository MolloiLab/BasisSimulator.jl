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

# ╔═╡ 00000001-0000-0000-0000-000000000001
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()
end

# ╔═╡ 00000001-0000-0000-0000-000000000002
# ╠═╡ show_logs = false
begin
    using Revise
end

# ╔═╡ 00000001-0000-0000-0000-000000000003
# ╠═╡ show_logs = false
begin
    using CairoMakie
    using Statistics
    using LinearAlgebra
end

# ╔═╡ 00000001-0000-0000-0000-000000000004
# ╠═╡ show_logs = false
begin
    "## Brain Perfusion CT Simulation"
end

# ╔═╡ 00000001-0000-0000-0000-000000000005
# ╠═╡ show_logs = false
begin
    "### 1. Configuration Parameters"
end

# ╔═╡ 00000001-0000-0000-0000-000000000006
# ╠═╡ show_logs = false
begin
    global phantom_dims = (64, 64, 32)
    global phantom_voxel_size = 0.2
end

# ╔═╡ 00000001-0000-0000-0000-000000000007
# ╠═╡ show_logs = false
begin
    global time_points = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    global arterial_conc = [0.0, 0.5, 2.0, 5.0, 4.0, 2.0, 1.0, 0.5, 0.0]
    global energy_keV = 60.0
end

# ╔═╡ 00000001-0000-0000-0000-000000000008
# ╠═╡ show_logs = false
begin
    "### 2. Generate Test Phantom"
end

# ╔═╡ 00000001-0000-0000-0000-000000000009
# ╠═╡ show_logs = false
begin
    function create_test_phantom(dims)
        labeled = zeros(UInt8, dims)
        
        cx, cy, cz = dims .÷ 2
        r_brain = minimum(dims) ÷ 8
        r_head = minimum(dims) ÷ 3
        
        for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
            dx, dy, dz = Float64(i - cx), Float64(j - cy), Float64(k - cz)
            r = sqrt(dx^2 + dy^2 + dz^2)
            
            if r <= r_brain
                labeled[i, j, k] = 2  # Brain
            elseif r <= r_head
                labeled[i, j, k] = 1  # Soft tissue
            else
                labeled[i, j, k] = 0  # Air
            end
        end
        
        return labeled
    end
    
    global phantom_mask = create_test_phantom(phantom_dims)
    "Phantom created: $(size(phantom_mask))"
end

# ╔═╡ 00000001-0000-0000-0000-000000000010
# ╠═╡ show_logs = false
begin
    "### 3. Perfusion Simulation"
end

# ╔═╡ 00000001-0000-0000-0000-000000000011
# ╠═╡ show_logs = false
begin
    # Simulate attenuation values for demo
    # Base attenuation values (1/cm): air=0, tissue=0.2, brain=0.22
    # Iodine contrast enhances brain tissue
    
    global mu_water = 0.2
    
    function simulate_perfusion(phantom_mask, time_points, arterial_conc)
        results = Dict()
        
        for (t_idx, t) in enumerate(time_points)
            conc = arterial_conc[t_idx]
            
            # Simulate attenuation map
            mu = Float64.(phantom_mask) .* 0.2
            
            # Add contrast enhancement to brain region (ID=2)
            # Iodine increases attenuation proportionally
            contrast_factor = conc * 0.05  # ~5% enhancement per mg/g
            brain_mask = (phantom_mask .== 2)
            mu = mu .+ brain_mask .* contrast_factor
            
            results[t] = Dict(
                "time" => t,
                "concentration" => conc,
                "mu" => mu
            )
        end
        
        return results
    end
    
    global perfusion_results = simulate_perfusion(phantom_mask, time_points, arterial_conc)
    "Simulation complete: $(length(time_points)) time points"
end

# ╔═╡ 00000001-0000-0000-0000-000000000012
# ╠═╡ show_logs = false
begin
    global dims = size(phantom_mask)
    global cz = dims[3] ÷ 2
end

# ╔═╡ 00000001-0000-0000-0000-000000000013
# ╠═╡ show_logs = false
begin
    "### 4. Interactive Controls"
end

# ╔═╡ 00000001-0000-0000-0000-000000000014
# ╠═╡ show_logs = false
begin
    "Select time point:"
end

# ╔═╡ 00000001-0000-0000-0000-000000000015
# ╠═╡ show_logs = false
@bind selected_time Slider(1:length(time_points); default=5, show_value=true)

# ╔═╡ 00000001-0000-0000-0000-000000000016
# ╠═╡ show_logs = false
begin
    "Select slice position:"
end

# ╔═╡ 00000001-0000-0000-0000-000000000017
# ╠═╡ show_logs = false
@bind selected_slice Slider(1:dims[3]; default=cz, show_value=true)

# ╔═╡ 00000001-0000-0000-0000-000000000018
# ╠═╡ show_logs = false
begin
    "### 5. Phantom Slice View"
end

# ╔═╡ 00000001-0000-0000-0000-000000000019
# ╠═╡ show_logs = false
begin
    t = time_points[selected_time]
    z = selected_slice
    
    mu_slice = perfusion_results[t]["mu"][:, :, z]
    hu_slice = 1000 .* (mu_slice .- mu_water) ./ mu_water
    
    fig = Figure(resolution=(600, 500))
    ax = Axis(fig[1, 1], title="HU Map: t=$(round(t, digits=1))s, z=$z")
    hm = heatmap!(ax, hu_slice; colormap=:viridis, colorrange=(-50, 200))
    Colorbar(fig[1, 2], hm)
    ax.xlabel = "X (voxel)"
    ax.ylabel = "Y (voxel)"
    fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000020
# ╠═╡ show_logs = false
begin
    "### 6. Attenuation Distribution"
end

# ╔═╡ 00000001-0000-0000-0000-000000000021
# ╠═╡ show_logs = false
begin
    t = time_points[selected_time]
    mu_data = vec(perfusion_results[t]["mu"])
    
    fig = Figure(resolution=(500, 300))
    ax = Axis(fig[1, 1], title="Attenuation Distribution (t=$(round(t, digits=1))s)",
              xlabel="μ (1/cm)", ylabel="Count")
    hist!(ax, mu_data, bins=30)
    fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000022
# ╠═╡ show_logs = false
begin
    "### 7. Time-Attenuation Curves"
end

# ╔═╡ 00000001-0000-0000-0000-000000000023
# ╠═╡ show_logs = false
begin
    cx, cy = dims[1]÷2, dims[2]÷2
    
    region_points = Dict(
        "Center" => (cx, cy, cz),
        "Left (+15)" => (cx-15, cy, cz),
        "Right (+15)" => (cx+15, cy, cz),
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

# ╔═╡ 00000001-0000-0000-0000-000000000024
# ╠═╡ show_logs = false
begin
    "### 8. Multi-Timepoint Comparison"
end

# ╔═╡ 00000001-0000-0000-0000-000000000025
# ╠═╡ show_logs = false
begin
    selected_times = [time_points[1], time_points[length(time_points)÷2+1], time_points[end]]
    
    fig = Figure(resolution=(900, 300))
    
    for (idx, t) in enumerate(selected_times)
        ax = Axis(fig[1, idx], title="t=$(round(t, digits=1))s")
        
        mu_slice = perfusion_results[t]["mu"][:, :, cz]
        hu_slice = 1000 .* (mu_slice .- mu_water) ./ mu_water
        
        hm = heatmap!(ax, hu_slice; colormap=:viridis, colorrange=(-50, 200))
        
        if idx == 3
            Colorbar(fig[1, 4], hm)
        end
    end
    
    fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000026
# ╠═╡ show_logs = false
begin
    "### 9. Summary Dashboard"
end

# ╔═╡ 00000001-0000-0000-0000-000000000027
# ╠═╡ show_logs = false
begin
    fig = Figure(resolution=(1000, 700))
    
    # Mean attenuation
    ax1 = Axis(fig[1, 1], title="Mean Attenuation")
    mu_mean = mean(perfusion_results[t]["mu"] for t in time_points)
    hm1 = heatmap!(ax1, mu_mean[:, :, cz]; colormap=:viridis)
    Colorbar(fig[1, 2], hm1)
    
    # Max change from baseline
    ax2 = Axis(fig[1, 3], title="Max Change from Baseline")
    mu_baseline = perfusion_results[time_points[1]]["mu"]
    mu_change = maximum(perfusion_results[t]["mu"] .- mu_baseline for t in time_points)
    hm2 = heatmap!(ax2, mu_change[:, :, cz]; colormap=:magma)
    Colorbar(fig[1, 4], hm2)
    
    # TAC at center
    ax3 = Axis(fig[2, 1:2], title="TAC at Center", xlabel="Time (s)", ylabel="HU")
    hu_center = [1000 * (perfusion_results[t]["mu"][cx, cy, cz] - mu_water) / mu_water for t in time_points]
    lines!(ax3, time_points, hu_center; color=:red, linewidth=2)
    
    # Histogram
    ax4 = Axis(fig[2, 3], title="Distribution of Changes")
    hist!(ax4, vec(mu_change), bins=30)
    
    # Material composition
    ax5 = Axis(fig[2, 4], title="Material Composition")
    mask_data = vec(phantom_mask)
    unique_ids = unique(mask_data)
    counts = [count(==(id), mask_data) for id in unique_ids]
    pie!(ax5, counts, labels=["Air", "Soft Tissue", "Brain"][1:length(unique_ids)])
    
    fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000028
# ╠═╡ show_logs = false
begin
    "### 10. Material Map"
end

# ╔═╡ 00000001-0000-0000-0000-000000000029
# ╠═╡ show_logs = false
begin
    MATERIAL_COLORS = Dict(
        0 => RGBf(0.0, 0.0, 0.0),
        1 => RGBf(0.8, 0.5, 0.5),
        2 => RGBf(0.9, 0.8, 0.7),
    )
    
    slice_data = phantom_mask[:, :, cz]
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

# ╔═╡ 00000001-0000-0000-0000-000000000030
# ╠═╡ show_logs = false
begin
    "## Summary"
end

# ╔═╡ 00000001-0000-0000-0000-000000000031
# ╠═╡ show_logs = false
begin
    Markdown.md"""
    This notebook demonstrates brain perfusion CT simulation with interactive visualization.
    
    **Features:**
    - Phantom generation with brain structures (air, soft tissue, brain)
    - Perfusion simulation with contrast enhancement over time
    - Interactive time/slice sliders
    - Time-attenuation curves
    - Summary dashboard
    
    **Note:** This simplified version uses simulated attenuation data. 
    For full simulation with BasisSimulator, run the script from command line.
    """
end

# ╔═╡ cell_hash = "00000001-0000-0000-0000-000000000001"
# ╠═╡ skip_as_script = false
end
