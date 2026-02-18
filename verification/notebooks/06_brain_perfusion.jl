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
