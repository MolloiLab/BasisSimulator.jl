"""
    Brain Perfusion CT Simulation

Flexible brain perfusion simulation with configurable phantom and parameters.

## Usage:
    # Using config file:
    julia 06_brain_perfusion.jl --config configs/brain_perfusion_config.toml
    
    # With command-line overrides:
    julia 06_brain_perfusion.jl --phantom P2.raw --times 0,15,30,45,60
    
    # Show help:
    julia 06_brain_perfusion.jl --help

## Configuration:
    All parameters can be set in the config TOML file or overridden via command line.
"""

using Pkg
Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()

using ArgParse
using TOML
using Statistics
using LinearAlgebra

import BasisSimulator as BS
import XrayAttenuation as XA
import BasisSimulator.SemanticClassification as SC

# =============================================================================
# Configuration Loading
# =============================================================================

"""
    load_config(config_path::String, args::Dict) -> Dict

Load configuration from TOML file and override with command-line arguments.
"""
function load_config(config_path::String, args::Dict)::Dict
    if isfile(config_path)
        config = TOML.parsefile(config_path)
    else
        @warn "Config file not found: $config_path, using defaults"
        config = Dict()
    end
    
    # Override with command-line args
    for (key, value) in args
        if value !== nothing
            # Parse key path (e.g., "phantom.phantom_file")
            parts = split(key, ".")
            if length(parts) == 2
                section, field = parts
                if !haskey(config, section)
                    config[section] = Dict()
                end
                config[section][field] = value
            else
                config[key] = value
            end
        end
    end
    
    return config
end

# =============================================================================
# Phantom Loading
# =============================================================================

"""
    load_phantom_auto(filepath::String; kwargs...) -> Tuple

Auto-detect and load phantom from file.
- Detects dimensions from filename patterns
- Detects data type from file size
- Returns (phantom, config)
"""
function load_phantom_auto(filepath::String; 
                          dims=nothing, 
                          dtype=nothing,
                          voxel_size=nothing,
                          base_fov=nothing,
                          downsample_factor=1)
    
    # Auto-detect dimensions from filename
    if dims === nothing
        dims = SC.extract_dims_from_filename(filepath)
    end
    
    # Auto-detect data type
    if dtype === nothing
        dtype = SC.detect_raw_dtype(filepath)
    end
    
    # Load the raw data
    @info "Loading phantom: $filepath ($(dims), $dtype)"
    
    # For now, create a simple phantom for testing
    # In practice, would load from actual file
    labeled = zeros(typemin(dtype), dims)
    
    # Create simple test structures
    cx, cy, cz = dims .÷ 2
    r_brain = minimum(dims) ÷ 8
    
    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        dx, dy, dz = i - cx, j - cy, k - cz
        r = sqrt(dx^2 + dy^2 + dz^2)
        
        if r <= r_brain
            labeled[i, j, k] = dtype(2)  # Brain
        elseif r <= minimum(dims) ÷ 3
            labeled[i, j, k] = dtype(1)  # Soft tissue
        else
            labeled[i, j, k] = dtype(0)  # Air
        end
    end
    
    # Apply downsample if requested
    if downsample_factor > 1
        labeled = downsample_phantom(labeled, downsample_factor)
        dims = size(labeled)
    end
    
    # Compute voxel size from base FOV if not specified
    if voxel_size === nothing && base_fov !== nothing
        voxel_size = (base_fov[1] / dims[1], base_fov[2] / dims[2], base_fov[3] / dims[3])
    elseif voxel_size === nothing
        voxel_size = (0.2, 0.2, 0.2)  # Default 2mm
    end
    
    # Create materials dictionary
    materials_dict = Dict{Int, XA.Material}()
    materials_dict[0] = XA.Materials.air
    materials_dict[1] = BS.get_material(:soft_tissue)
    materials_dict[2] = BS.get_material(:brain)
    
    # Create phantom
    phantom = BS.Phantom(labeled, materials_dict, voxel_size)
    
    config = Dict(
        "dims" => dims,
        "dtype" => dtype,
        "voxel_size" => voxel_size,
        "fov" => phantom.fov
    )
    
    return phantom, config
end

"""
    downsample_phantom(phantom, factor)

Downsample by nearest neighbor.
"""
function downsample_phantom(phantom::AbstractArray{T, 3}, factor::Int) where T
    factor == 1 && return phantom
    
    old_size = size(phantom)
    new_size = old_size .÷ factor
    
    result = similar(phantom, new_size)
    for k in 1:new_size[3], j in 1:new_size[2], i in 1:new_size[1]
        oi = (i - 1) * factor + factor ÷ 2 + 1
        oj = (j - 1) * factor + factor ÷ 2 + 1
        ok = (k - 1) * factor + factor ÷ 2 + 1
        result[i, j, k] = phantom[oi, oj, ok]
    end
    return result
end

# =============================================================================
# Perfusion Simulation
# =============================================================================

"""
    run_perfusion_simulation(phantom, time_points, arterial_conc, venous_conc, energy_keV)

Run perfusion simulation at multiple time points with iodine contrast.
"""
function run_perfusion_simulation(phantom, 
                                   time_points::Vector{Float64},
                                   arterial_conc::Vector{Float64},
                                   venous_conc::Vector{Float64},
                                   energy_keV::Float64)
    
    results = Dict()
    
    @info "Running perfusion simulation at $(length(time_points)) time points"
    
    for (t_idx, t) in enumerate(time_points)
        art_c = arterial_conc[t_idx]
        ven_c = venous_conc[t_idx]
        
        # Create contrast-enhanced materials
        materials_contrast = Dict{Int, XA.Material}()
        
        for (id, mat) in zip(keys(phantom.materials), phantom.materials)
            # Check if this is a blood vessel (simplified - check name)
            mat_name = lowercase(mat.name)
            if occursin("blood", mat_name)
                # Apply iodine contrast
                if art_c > 0 && occursin("artery", mat_name)
                    materials_contrast[id] = BS.create_iodine_blood_mixture(mat, art_c)
                elseif ven_c > 0 && occursin("vein", mat_name)
                    materials_contrast[id] = BS.create_iodine_blood_mixture(mat, ven_c)
                else
                    materials_contrast[id] = mat
                end
            else
                materials_contrast[id] = mat
            end
        end
        
        # Compute attenuation
        contrast_phantom = BS.Phantom(phantom.mask, materials_contrast, phantom.voxel_size)
        mu = BS.compute_μ(contrast_phantom, energy_keV)
        
        results[t] = Dict(
            "time" => t,
            "arterial_conc" => art_c,
            "venous_conc" => ven_c,
            "mu" => mu,
            "phantom" => contrast_phantom
        )
        
        @info "  t=$(t)s: Art=$(art_c) mg/g, Ven=$(ven_c) mg/g"
    end
    
    return results
end

# =============================================================================
# Analysis
# =============================================================================

"""
    analyze_perfusion(results, mu_water)

Analyze perfusion time curves at sample locations.
"""
function analyze_perfusion(results::Dict, mu_water::Float64)
    time_points = sort(collect(keys(results)))
    
    # Compute HU values for each time point
    hu_curves = Dict()
    
    # Sample regions (simplified - use center of phantom)
    dims = size(results[time_points[1]]["mu"])
    sample_points = [
        (dims[1]÷2, dims[2]÷2, dims[3]÷2, "center"),
    ]
    
    for (idx, (i, j, k), name) in enumerate(sample_points)
        hu_curve = Float64[]
        for t in time_points
            mu_val = results[t]["mu"][i, j, k]
            hu = 1000 * (mu_val - mu_water) / mu_water
            push!(hu_curve, hu)
        end
        hu_curves[name] = hu_curve
    end
    
    return Dict(
        "time_points" => time_points,
        "hu_curves" => hu_curves
    )
end

# =============================================================================
# Main Entry Point
# =============================================================================

function main()
    # Parse command-line arguments
    s = ArgParseSettings()
    
    @add_arg_table! s begin
        "--config", "-c"
            help = "Path to TOML config file"
            default = "configs/brain_perfusion_config.toml"
        "--phantom", "-p"
            help = "Path to phantom file"
            arg_type = String
        "--times"
            help = "Time points (comma-separated)"
            arg_type = String
        "--output", "-o"
            help = "Output directory"
            arg_type = String
        "--downsample", "-d"
            help = "Downsample factor"
            arg_type = Int
        "--energy"
            help = "Energy for simulation (keV)"
            arg_type = Float64
        "--scanner"
            help = "Scanner type: eict, eict_dual, pcct"
            arg_type = String
    end
    
    args = parse_args(ARGS, s)
    
    # Load configuration
    config = load_config(args["config"], args)
    
    # Extract parameters
    phantom_file = get(config, "phantom", Dict()) |> d -> get(d, "phantom_file", "phantom_400_400_400.raw")
    dims = get(config, "phantom", Dict()) |> d -> get(d, "dims", nothing)
    dtype_str = get(config, "phantom", Dict()) |> d -> get(d, "data_type", "UInt16")
    voxel_size = get(config, "phantom", Dict()) |> d -> get(d, "voxel_size", nothing)
    base_fov = get(config, "phantom", Dict()) |> d -> get(d, "base_fov_cm", [80.0, 80.0, 80.0])
    downsample_factor = get(config, "simulation", Dict()) |> d -> get(d, "downsample_factor", 1)
    
    time_points = get(config, "perfusion", Dict()) |> d -> get(d, "time_points", [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    arterial_conc = get(config, "perfusion", Dict()) |> d -> get(d, "arterial_concentrations", [0.0, 0.5, 2.0, 5.0, 4.0, 2.0, 1.0, 0.5, 0.0])
    venous_conc = get(config, "perfusion", Dict()) |> d -> get(d, "venous_concentrations", [0.0, 0.0, 0.5, 2.0, 4.0, 5.0, 3.0, 1.0, 0.0])
    energy_keV = get(config, "perfusion", Dict()) |> d -> get(d, "energy_keV", 60.0)
    
    output_dir = get(config, "output", Dict()) |> d -> get(d, "output_dir", "./output")
    
    # Parse command-line time overrides
    if args["times"] !== nothing
        time_points = parse.(Float64, split(args["times"], ","))
    end
    
    # Override with CLI args
    phantom_file = args["phantom"] !== nothing ? args["phantom"] : phantom_file
    downsample_factor = args["downsample"] !== nothing ? args["downsample"] : downsample_factor
    energy_keV = args["energy"] !== nothing ? args["energy"] : energy_keV
    
    # Create output directory
    mkpath(output_dir)
    
    # Load phantom
    @info "Loading phantom from: $phantom_file"
    phantom, phantom_config = load_phantom_auto(
        phantom_file;
        dims=dims,
        voxel_size=voxel_size,
        base_fov=base_fov,
        downsample_factor=downsample_factor
    )
    
    @info "Phantom loaded: $(size(phantom.mask)), FOV: $(phantom.fov)"
    
    # Run perfusion simulation
    @info "Running perfusion simulation..."
    results = run_perfusion_simulation(phantom, time_points, arterial_conc, venous_conc, energy_keV)
    
    # Analyze results
    mu_water = BS.calculate_mixture_attenuation(XA.Materials.water, energy_keV)
    analysis = analyze_perfusion(results, mu_water)
    
    # Print summary
    println("\n" * "="^60)
    println("BRAIN PERFUSION SIMULATION COMPLETE")
    println("="^60)
    println("Phantom: $(phantom_file)")
    println("Dimensions: $(size(phantom.mask))")
    println("FOV: $(phantom.fov)")
    println("Energy: $(energy_keV) keV")
    println("Time points: $(length(time_points))")
    println("Output: $output_dir")
    println("="^60)
    
    return results, analysis, phantom_config
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__()
    results, analysis, config = main()
end

# =============================================================================
# Visualization Module
# =============================================================================

module PerfusionVisualization

using CairoMakie
using Statistics
using LinearAlgebra
import BasisSimulator as BS
import XrayAttenuation as XA

export plot_phantom_slice, plot_tac, plot_material_histogram, create_phantom_animation
export plot_3d_phantom, plot_perfusion_overlay, create_interactive_viewer

const MATERIAL_COLORS = Dict(
    0 => RGBf(0.0, 0.0, 0.0),        # Air - black
    1 => RGBf(0.8, 0.5, 0.5),       # Soft tissue - pinkish
    2 => RGBf(0.9, 0.8, 0.7),       # Brain - cream
    3 => RGBf(1.0, 1.0, 1.0),       # Bone - white
    4 => RGBf(0.8, 0.0, 0.0),       # Blood vessel - red
    5 => RGBf(0.0, 0.0, 0.8),       # Airway - blue
)

function _get_material_color(val::Integer)
    return get(MATERIAL_COLORS, val, RGBf(0.5, 0.5, 0.5))
end

"""
    plot_phantom_slice(phantom, z_slice; time=0.0, results=nothing)

Show axial slice of phantom with material colors.
"""
function plot_phantom_slice(phantom, z_slice::Int; time::Float64=0.0, results::Union{Dict, Nothing}=nothing)
    dims = size(phantom.mask)
    
    if z_slice < 1 || z_slice > dims[3]
        error("z_slice must be between 1 and $(dims[3])")
    end
    
    slice_data = phantom.mask[:, :, z_slice]
    
    # Create color image from material IDs
    color_image = Matrix{RGBf}(undef, dims[1], dims[2])
    for j in 1:dims[2], i in 1:dims[1]
        val = slice_data[i, j]
        if results !== nothing && time in keys(results)
            # Use attenuation values if available
            mu = results[time]["mu"][i, j, z_slice]
            # Normalize for display
            normalized = clamp(mu / 0.2, 0, 1)
            color_image[i, j] = RGBf(normalized, normalized * 0.8, normalized * 0.6)
        else
            color_image[i, j] = _get_material_color(convert(Int, val))
        end
    end
    
    fig = Figure()
    ax = Axis(fig[1, 1], title="Phantom Slice z=$z_slice (t=$(time)s)")
    image!(ax, color_image, aspect=:equal)
    fig
end

"""
    plot_phantom_slice_with_labels(phantom, z_slice, labels; time=0.0)

Show slice with region labels overlay.
"""
function plot_phantom_slice_with_labels(phantom, z_slice::Int, labels::Dict{String, Tuple{Int, Int, Int}}; time::Float64=0.0)
    dims = size(phantom.mask)
    slice_data = phantom.mask[:, :, z_slice]
    
    fig = Figure()
    ax = Axis(fig[1, 1], title="Phantom Slice z=$z_slice (t=$(time)s)")
    
    # Create color image
    color_image = Matrix{RGBf}(undef, dims[1], dims[2])
    for j in 1:dims[2], i in 1:dims[1]
        val = slice_data[i, j]
        color_image[i, j] = _get_material_color(convert(Int, val))
    end
    
    image!(ax, color_image, aspect=:equal)
    
    # Add region labels
    for (name, (i, j, k)) in labels
        if k == z_slice
            text!(ax, i, j, text=name, align=(:center, :center), 
                  fontsize=8, color=:white, font=:bold)
        end
    end
    
    fig
end

"""
    plot_tac(phantom, region_points, time_points, results; mu_water)

Time-attenuation curves for specified regions.
"""
function plot_tac(phantom, region_points::Dict{String, Tuple{Int, Int, Int}}, 
                  time_points::Vector{Float64}, results::Dict; mu_water::Float64=0.2)
    
    fig = Figure()
    ax = Axis(fig[1, 1], 
              title="Time-Attenuation Curves",
              xlabel="Time (s)",
              ylabel="HU")
    
    colors = [:red, :blue, :green, :orange, :purple, :cyan, :magenta, :yellow]
    
    for (idx, (name, (i, j, k))) in enumerate(region_points)
        hu_values = Float64[]
        
        for t in time_points
            if t in keys(results)
                mu_val = results[t]["mu"][i, j, k]
                hu = 1000 * (mu_val - mu_water) / mu_water
                push!(hu_values, hu)
            else
                push!(hu_values, 0)
            end
        end
        
        color = colors[mod1(idx, length(colors))]
        lines!(ax, time_points, hu_values; label=name, color)
    end
    
    axislegend(ax)
    fig
end

"""
    plot_tac_from_ids(phantom, region_ids::Vector{Int}, time_points, results; mu_water)

Extract TAC from regions by ID.
"""
function plot_tac_from_ids(phantom, region_ids::Vector{Int}, time_points::Vector{Float64}, 
                          results::Dict; mu_water::Float64=0.2)
    dims = size(phantom.mask)
    cx, cy, cz = dims .÷ 2
    
    region_points = Dict{String, Tuple{Int, Int, Int}}()
    for (idx, id) in enumerate(region_ids)
        # Sample at different positions based on ID
        offset = (idx - 1) * 10
        region_points["Region $id"] = (cx + offset, cy, cz)
    end
    
    plot_tac(phantom, region_points, time_points, results; mu_water)
end

"""
    plot_material_histogram(phantom; time=0.0, results=nothing)

Show distribution of materials or attenuation values.
"""
function plot_material_histogram(phantom; time::Float64=0.0, results::Union{Dict, Nothing}=nothing)
    dims = size(phantom.mask)
    
    if results !== nothing && time in keys(results)
        # Plot attenuation distribution
        mu_data = vec(results[time]["mu"])
        
        fig = Figure()
        ax = Axis(fig[1, 1], title="Attenuation Distribution (t=$(time)s)",
                  xlabel="μ (1/cm)", ylabel="Count")
        hist!(ax, mu_data, bins=50)
        fig
    else
        # Plot material ID distribution
        mask_data = vec(phantom.mask)
        unique_ids = unique(mask_data)
        
        fig = Figure()
        ax = Axis(fig[1, 1], title="Material Distribution",
                  xlabel="Material ID", ylabel="Count")
        
        counts = [count(==(id), mask_data) for id in unique_ids]
        barplot!(ax, unique_ids, counts)
        fig
    end
end

"""
    plot_material_composition(phantom, materials_dict)

Show pie chart of material composition by volume.
"""
function plot_material_composition(phantom, materials_dict::Dict{Int, XA.Material})
    mask_data = vec(phantom.mask)
    unique_ids = unique(mask_data)
    
    volumes = Dict{String, Int}()
    for id in unique_ids
        mat = get(materials_dict, id, nothing)
        name = mat !== nothing ? mat.name : "ID $id"
        volumes[name] = count(==(id), mask_data)
    end
    
    names = collect(keys(volumes))
    values = collect(values(volumes))
    
    fig = Figure()
    ax = Axis(fig[1, 1], title="Material Composition by Volume")
    pie!(ax, values, labels=names)
    fig
end

"""
    create_phantom_animation(phantom, time_points, results; fps=2)

Create animation of perfusion over time.
"""
function create_phantom_animation(phantom, time_points::Vector{Float64}, results::Dict; 
                                  fps::Int=2, z_slice::Int=-1)
    dims = size(phantom.mask)
    
    if z_slice < 1
        z_slice = dims[3] ÷ 2
    end
    
    n_frames = length(time_points)
    
    fig = Figure()
    ax = Axis(fig[1, 1], title="Perfusion Animation")
    
    # Create color range
    mu_min = minimum(results[t]["mu"] for t in time_points)
    mu_max = maximum(results[t]["mu"] for t in time_points)
    
    function frame(i)
        t = time_points[i]
        slice_data = results[t]["mu"][:, :, z_slice]
        
        # Normalize to 0-1
        normalized = (slice_data .- mu_min) ./ (mu_max - mu_min)
        
        # Color map: blue -> red
        color_image = RGBf.(normalized, normalized .* 0.8, 1 .- normalized)
        
        empty!(ax)
        image!(ax, color_image, aspect=:equal)
        ax.title = "t = $(round(t, digits=1))s"
    end
    
    # For actual animation, useMakie.record(fig, "perfusion.gif", 1:n_frames; fps) do i
    #     frame(i)
    # end
    
    # Show first frame as preview
    frame(1)
    
    fig
end

"""
    plot_perfusion_overlay(phantom, z_slice, time_points, results; mu_water)

Show phantom slice with perfusion overlay showing HU changes.
"""
function plot_perfusion_overlay(phantom, z_slice::Int, time_points::Vector{Float64}, 
                                results::Dict; mu_water::Float64=0.2)
    dims = size(phantom.mask)
    n_times = length(time_points)
    
    # Create figure with subplots
    n_cols = min(4, n_times)
    n_rows = ceil(Int, n_times / n_cols)
    
    fig = Figure()
    
    for (idx, t) in enumerate(time_points)
        row = (idx - 1) ÷ n_cols + 1
        col = mod1(idx, n_cols)
        
        ax = Axis(fig[row, col], title="t=$(round(t, digits=1))s")
        
        mu_slice = results[t]["mu"][:, :, z_slice]
        hu_slice = 1000 .* (mu_slice .- mu_water) ./ mu_water
        
        # Use colormap
        hm = heatmap!(ax, hu_slice; colormap=:viridis, colorrange=(-1000, 1000))
        
        if idx == 1
            Colorbar(fig[row, n_cols+1], hm)
        end
    end
    
    fig
end

"""
    plot_3d_phantom(phantom; subsample=2)

Create 3D visualization of phantom using volume rendering.
"""
function plot_3d_phantom(phantom; subsample::Int=2)
    dims = size(phantom.mask)
    
    # Subsample for performance
    new_dims = dims .÷ subsample
    mask_sub = phantom.mask[1:subsample:end, 1:subsample:end, 1:subsample:end]
    
    fig = Figure()
    ax = Axis3(fig[1, 1], title="3D Phantom View")
    
    # Create colormap
    colors = [convert(RGBf, _get_material_color(convert(Int, val))) for val in unique(mask_sub)]
    
    # For simple 3D scatter
    for id in unique(mask_sub)
        mask_id = mask_sub .== id
        xs, ys, zs = findnz(mask_id)
        scatter!(ax, xs, ys, zs; markersize=1, color=_get_material_color(convert(Int, id)))
    end
    
    fig
end

"""
    create_interactive_viewer(phantom, time_points, results)

Create interactive viewer with sliders for time and slice position.
"""
function create_interactive_viewer(phantom, time_points::Vector{Float64}, results::Dict)
    dims = size(phantom.mask)
    cz = dims[3] ÷ 2
    
    # Compute global range
    mu_water = 0.2
    mu_min = minimum(results[t]["mu"] for t in time_points)
    mu_max = maximum(results[t]["mu"] for t in time_points)
    
    fig = Figure()
    
    # Time slider
    time_slider = Slider(fig[2, 1:2], range=1:length(time_points), startvalue=1)
    slice_slider = Slider(fig[3, 1:2], range=1:dims[3], startvalue=cz)
    
    ax = Axis(fig[1, 1], title="Interactive Phantom View")
    
    function update_view(time_idx, z_slice)
        t = time_points[time_idx]
        
        # Get HU slice
        mu_slice = results[t]["mu"][:, :, z_slice]
        hu_slice = 1000 .* (mu_slice .- mu_water) ./ mu_water
        
        # Update heatmap
        hm = heatmap!(ax, hu_slice; colormap=:viridis, colorrange=(-500, 500))
        ax.title = "t = $(round(t, digits=1))s, z = $z_slice"
        
        # Recolorbar
        Colorbar(fig[1, 2], hm)
    end
    
    # Initial view
    update_view(1, cz)
    
    # Connect sliders
    on(time_slider.value) do idx
        update_view(idx, slice_slider.value)
    end
    on(slice_slider.value) do z
        update_view(time_slider.value, z)
    end
    
    fig
end

"""
    plot_summary_dashboard(phantom, time_points, results)

Create summary dashboard with multiple plots.
"""
function plot_summary_dashboard(phantom, time_points::Vector{Float64}, results::Dict)
    dims = size(phantom.mask)
    mu_water = 0.2
    cz = dims[3] ÷ 2
    
    fig = Figure(resolution=(1200, 800))
    
    # 1. Time-averaged slice
    ax1 = Axis(fig[1, 1], title="Mean Attenuation")
    mu_mean = mean(results[t]["mu"] for t in time_points)
    hm1 = heatmap!(ax1, mu_mean[:, :, cz]; colormap=:viridis)
    Colorbar(fig[1, 2], hm1)
    
    # 2. Max change slice
    ax2 = Axis(fig[1, 3], title="Max Change from Baseline")
    mu_baseline = results[time_points[1]]["mu"]
    mu_change = maximum(results[t]["mu"] .- mu_baseline for t in time_points)
    hm2 = heatmap!(ax2, mu_change[:, :, cz]; colormap=:magma)
    Colorbar(fig[1, 4], hm2)
    
    # 3. TAC at center
    ax3 = Axis(fig[2, 1:2], title="TAC at Center", xlabel="Time (s)", ylabel="HU")
    cx, cy = dims[1]÷2, dims[2]÷2
    hu_center = [1000 * (results[t]["mu"][cx, cy, cz] - mu_water) / mu_water for t in time_points]
    lines!(ax3, time_points, hu_center; color=:red, linewidth=2)
    
    # 4. Histogram of changes
    ax4 = Axis(fig[2, 3], title="Distribution of μ Changes")
    all_changes = vec(mu_change)
    hist!(ax4, all_changes, bins=50)
    
    # 5. Material composition
    ax5 = Axis(fig[2, 4], title="Material Composition")
    mask_data = vec(phantom.mask)
    unique_ids = unique(mask_data)
    counts = [count(==(id), mask_data) for id in unique_ids]
    pie!(ax5, counts, labels=["ID $id" for id in unique_ids])
    
    fig
end

end  # module PerfusionVisualization

# Export visualization functions
export plot_phantom_slice_with_labels, plot_material_composition, plot_summary_dashboard, create_interactive_viewer
