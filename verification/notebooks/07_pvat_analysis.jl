"""
    PVAT Analysis with Multi-Material Mixtures

Flexible PVAT analysis with configurable materials and mixtures.

## Usage:
    julia 07_pvat_analysis.jl --config configs/pvat_analysis_config.toml
    
    # With overrides:
    julia 07_pvat_analysis.jl --phantom my_phantom.raw --downsample 2

## Features:
- Auto-detect phantom dimensions from filename
- Configurable material mappings
- Dynamic material mixtures (e.g., bone marrow, iodine contrast)
- ROI-based analysis
- HU histogram computation
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

# =============================================================================
# Configuration
# =============================================================================

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
    load_pvat_phantom(filepath; dims, dtype, voxel_size, downsample)

Load PVAT phantom with auto-detection.
"""
function load_pvat_phantom(filepath::String; 
                           dims=nothing, 
                           dtype=UInt8,
                           voxel_size=(0.1, 0.1, 0.2),
                           downsample_factor=1)
    
    # Auto-detect dimensions from filename
    if dims === nothing
        dims = try
            BS.SemanticClassification.extract_dims_from_filename(filepath)
        catch
            @warn "Could not auto-detect dimensions, using default"
            (1600, 1400, 500)
        end
    end
    
    @info "Loading PVAT phantom: $filepath ($(dims), $dtype)"
    
    # Create sample phantom (in practice would load from file)
    labeled = zeros(dtype, dims)
    
    # Create simple anatomical structures
    cx, cy, cz = dims .÷ 2
    
    # Body outline (ellipsoid)
    rx, ry, rz = dims[1]÷2 - 20, dims[2]÷2 - 20, dims[3]÷2 - 10
    
    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        dx, dy, dz = (i - cx)/rx, (j - cy)/ry, (k - cz)/rz
        r = sqrt(dx^2 + dy^2 + dz^2)
        
        if r <= 0.9
            # Inside body
            labeled[i, j, k] = dtype(1)  # Soft tissue
        else
            labeled[i, j, k] = dtype(0)  # Air
        end
    end
    
    # Add bone-like structures (spine)
    for k in 1:dims[3]
        for j in (cy-10):(cy+10)
            for i in (cx-5):(cx+5)
                labeled[i, j, k] = dtype(2)  # Bone
            end
        end
    end
    
    # Downsample if requested
    if downsample_factor > 1
        labeled = downsample_phantom(labeled, downsample_factor)
        dims = size(labeled)
        voxel_size = voxel_size .* downsample_factor
    end
    
    return labeled, dims, voxel_size
end

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
# Material Management
# =============================================================================

"""
    create_materials_dict(phantom; mapping_strategy, custom_mappings)

Create materials dictionary for phantom.
"""
function create_materials_dict(phantom; 
                              mapping_strategy="auto",
                              custom_mappings=Dict())
    
    unique_ids = unique(phantom)
    materials_dict = Dict{Int, XA.Material}()
    
    # Default material registry
    material_registry = Dict(
        0 => XA.Materials.air,
        1 => BS.get_material(:soft_tissue),  # Water-equivalent
        2 => BS.get_material(:bone),
        3 => BS.get_material(:muscle),
        4 => BS.get_material(:brain),
        5 => BS.get_material(:blood),
    )
    
    for id in unique_ids
        if haskey(custom_mappings, id)
            # Use custom mapping
            mat_symbol = custom_mappings[id]
            if isa(mat_symbol, Symbol)
                materials_dict[id] = BS.get_material(mat_symbol)
            else
                materials_dict[id] = mat_symbol
            end
        elseif haskey(material_registry, Int(id))
            materials_dict[Int(id)] = material_registry[Int(id)]
        else
            # Default to soft tissue
            materials_dict[Int(id)] = BS.get_material(:soft_tissue)
        end
    end
    
    return materials_dict
end

"""
    apply_material_mixtures(phantom, materials_dict, mixture_config)

Apply dynamic material mixtures.
"""
function apply_material_mixtures(phantom, 
                                materials_dict, 
                                mixture_config::Dict)
    
    if !get(mixture_config, "enabled", false)
        return materials_dict
    end
    
    # Enable iodine contrast if configured
    if get(mixture_config, "enable_iodine_contrast", false)
        iodine_conc = get(mixture_config, "iodine_concentration_mg_g", 5.0)
        
        for (id, mat) in materials_dict
            mat_name = lowercase(mat.name)
            if occursin("blood", mat_name)
                # Apply iodine contrast
                materials_dict[id] = BS.create_iodine_blood_mixture(mat, iodine_conc)
            end
        end
    end
    
    # Apply custom mixtures if specified
    custom = get(mixture_config, "custom", Dict())
    for (id, mixture_spec) in custom
        # mixture_spec format: ["material1", ratio1, "material2", ratio2, ...]
        materials = []
        fractions = []
        for i in 1:2:length(mixture_spec)-1
            mat_name = mixture_spec[i]
            frac = mixture_spec[i+1]
            push!(materials, BS.get_material(Symbol(mat_name)))
            push!(fractions, frac)
        end
        
        if length(materials) > 0
            mixed = BS.create_mixture(materials, fractions)
            materials_dict[id] = mixed
        end
    end
    
    return materials_dict
end

# =============================================================================
# Analysis
# =============================================================================

"""
    analyze_phantom(phantom, materials_dict, voxel_size, energy_keV, analysis_config)

Perform ROI and histogram analysis on phantom.
"""
function analyze_phantom(phantom, 
                        materials_dict,
                        voxel_size,
                        energy_keV::Float64,
                        analysis_config::Dict)
    
    # Compute attenuation
    ph = BS.Phantom(phantom, materials_dict, voxel_size)
    mu = BS.compute_μ(ph, energy_keV)
    
    # Water attenuation for HU conversion
    mu_water = BS.calculate_mixture_attenuation(XA.Materials.water, energy_keV)
    
    # Convert to HU
    hu = 1000 .* (mu .- mu_water) ./ mu_water
    
    results = Dict()
    
    # ROI analysis
    if get(analysis_config, "compute_roi_stats", true)
        center = get(analysis_config, "roi_center", [0.5, 0.5, 0.5])
        radius = get(analysis_config, "roi_radius", 20)
        
        dims = size(hu)
        cx = floor(Int, center[1] * dims[1])
        cy = floor(Int, center[2] * dims[2])
        cz = floor(Int, center[3] * dims[3])
        
        # Extract ROI
        roi_hu = Float64[]
        for k in max(1, cz-radius):min(dims[3], cz+radius)
            for j in max(1, cy-radius):min(dims[2], cy+radius)
                for i in max(1, cx-radius):min(dims[1], cx+radius)
                    if (i-cx)^2 + (j-cy)^2 + (k-cz)^2 <= radius^2
                        push!(roi_hu, hu[i, j, k])
                    end
                end
            end
        end
        
        results["roi"] = Dict(
            "mean" => mean(roi_hu),
            "std" => std(roi_hu),
            "min" => minimum(roi_hu),
            "max" => maximum(roi_hu),
            "n_pixels" => length(roi_hu)
        )
    end
    
    # Histogram analysis
    if get(analysis_config, "compute_histograms", true)
        n_bins = get(analysis_config, "histogram_bins", 100)
        hu_range = get(analysis_config, "histogram_range", [-1000, 1000])
        
        hist_counts = fit(Histogram, vec(hu), closed=:left).weights
        hist_edges = range(hu_range[1], hu_range[2], length=n_bins+1)
        
        results["histogram"] = Dict(
            "counts" => hist_counts,
            "edges" => collect(hist_edges)
        )
    end
    
    # Per-material statistics
    results["materials"] = Dict()
    for (id, mat) in materials_dict
        mask = (phantom .== id)
        if sum(mask) > 0
            mat_hu = hu[mask]
            results["materials"][Int(id)] = Dict(
                "name" => mat.name,
                "n_voxels" => sum(mask),
                "mean_hu" => mean(mat_hu),
                "std_hu" => std(mat_hu)
            )
        end
    end
    
    return results, hu
end

# =============================================================================
# Main
# =============================================================================

function main()
    # Parse arguments
    s = ArgParseSettings()
    
    @add_arg_table! s begin
        "--config", "-c"
            help = "Config file path"
            default = "configs/pvat_analysis_config.toml"
        "--phantom", "-p"
            help = "Phantom file path"
            arg_type = String
        "--downsample", "-d"
            help = "Downsample factor"
            arg_type = Int
        "--output", "-o"
            help = "Output directory"
            arg_type = String
        "--energy"
            help = "Energy (keV)"
            arg_type = Float64
    end
    
    args = parse_args(ARGS, s)
    
    # Load config
    config = load_config(args["config"], args)
    
    # Extract parameters
    phantom_file = get(config, "phantom", Dict()) |> d -> get(d, "phantom_file", "vmale50.raw")
    dims = get(config, "phantom", Dict()) |> d -> get(d, "dims", nothing)
    dtype_str = get(config, "phantom", Dict()) |> d -> get(d, "data_type", "UInt8")
    dtype = dtype_str == "UInt16" ? UInt16 : UInt8
    voxel_size = get(config, "phantom", Dict()) |> d -> get(d, "voxel_size", [0.1, 0.1, 0.2])
    downsample = get(config, "phantom", Dict()) |> d -> get(d, "downsample_factor", 4)
    
    # Material config
    mapping_strategy = get(config, "materials", Dict()) |> d -> get(d, "mapping_strategy", "auto")
    mixtures = get(config, "mixtures", Dict())
    
    # Analysis config
    analysis = get(config, "analysis", Dict())
    
    # Simulation config
    energy_keV = get(config, "perfusion", Dict()) |> d -> get(d, "energy_keV", 60.0)
    energy_keV = args["energy"] !== nothing ? args["energy"] : energy_keV
    
    # Output config
    output_dir = get(config, "output", Dict()) |> d -> get(d, "output_dir", "./pvat_output")
    phantom_file = args["phantom"] !== nothing ? args["phantom"] : phantom_file
    downsample = args["downsample"] !== nothing ? args["downsample"] : downsample
    
    # Create output directory
    mkpath(output_dir)
    
    # Load phantom
    @info "Loading phantom: $phantom_file"
    labeled, dims, vs = load_pvat_phantom(
        phantom_file;
        dims=dims,
        dtype=dtype,
        voxel_size=tuple(voxel_size...),
        downsample_factor=downsample
    )
    
    @info "Phantom: $(size(labeled)), voxel_size: $vs"
    
    # Create materials
    @info "Creating materials dictionary..."
    materials_dict = create_materials_dict(labeled; mapping_strategy=mapping_strategy)
    
    # Apply mixtures
    @info "Applying material mixtures..."
    materials_dict = apply_material_mixtures(labeled, materials_dict, mixtures)
    
    # Analyze
    @info "Analyzing phantom at $(energy_keV) keV..."
    results, hu = analyze_phantom(labeled, materials_dict, vs, energy_keV, analysis)
    
    # Print summary
    println("\n" * "="^60)
    println("PVAT ANALYSIS COMPLETE")
    println("="^60)
    println("Phantom: $phantom_file")
    println("Dimensions: $(size(labeled))")
    println("Voxel size: $vs")
    println("Energy: $energy_keV keV")
    
    if haskey(results, "roi")
        roi = results["roi"]
        println("\nROI Statistics:")
        println("  Mean HU: $(round(roi["mean"], digits=1))")
        println("  Std HU: $(round(roi["std"], digits=1))")
        println("  Range: [$(round(roi["min"], digits=1)), $(round(roi["max"], digits=1))]")
    end
    
    println("\nMaterial Statistics:")
    for (id, mat_stats) in results["materials"]
        println("  ID $id ($(mat_stats["name"])): $(round(mat_stats["mean_hu"], digits=1)) ± $(round(mat_stats["std_hu"], digits=1)) HU")
    end
    
    println("\nOutput: $output_dir")
    println("="^60)
    
    return results, labeled, materials_dict
end

# Run if executed
if abspath(PROGRAM_FILE) == @__FILE__()
    results, labeled, materials = main()
end
