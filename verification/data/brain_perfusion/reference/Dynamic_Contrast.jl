""" 
Dynamic Brain Phantom
"""
using MAT
using FileIO
using CSV
using DataFrames
using Random
using Statistics
using Interpolations
using LinearAlgebra
using DSP
using Printf
using JSON3

# ------------------------------------------------------------
# Function to Load Materials
# ------------------------------------------------------------

const ATOMIC_SYMBOLS = Dict(
    1=>"H", 2=>"He", 3=>"Li", 4=>"Be", 5=>"B", 6=>"C", 7=>"N", 8=>"O",
    9=>"F", 10=>"Ne", 11=>"Na", 12=>"Mg", 13=>"Al", 14=>"Si", 15=>"P",
    16=>"S", 17=>"Cl", 18=>"Ar", 19=>"K", 20=>"Ca", 21=>"Sc", 22=>"Ti",
    23=>"V", 24=>"Cr", 25=>"Mn", 26=>"Fe", 27=>"Co", 28=>"Ni", 29=>"Cu",
    30=>"Zn", 35=>"Br", 53=>"I"
)

function load_material_file(filepath::AbstractString)
	open(filepath, "r") do io
		lines = collect(eachline(io))

		# Initialize arrays
	   	material_name = ""
   		num_elements = 0
   		density = 0.0
    	composition = []
		
		for (i, line) in enumerate(lines)
			line = strip(line)
				
			# extract material name
			if occursin("# Material name:", line)
				material_name = strip(replace(lines[i+1], "#" => ""))
			
			# Extract number of elements
			elseif occursin("# Number of elements:", line)
				num_elements = strip(lines[i+1])
				num_elements = parse(Int, num_elements)

			# Extract density in g/cm^3
			elseif occursin("# Density:", line)
				density = strip(lines[i+1])
				density = parse(Float64, density)
            
            elseif occursin("Atomic number", line)
				# Initialize array
            	composition = zeros(Float64, num_elements, 2)
            	# Read the next num_elements lines
            	for j in 1:num_elements
            		parts = split(strip(lines[i+j]))
            		composition[j, 1] = parse(Int, parts[1])     # atomic number
               		composition[j, 2] = parse(Float64, parts[2]) # mass fraction
            	end
			end
				
		end

        return Dict(
            "material_name" => material_name,
            "density" => density,
            "n_elements" => num_elements === missing ? length(composition) : num_elements,
            "composition" => composition
        )
    end
end

# ------------------------------------------------------------
# Save Material Composition to text file (CatSim format) => change to new simulation format
# ------------------------------------------------------------

function save_material_file(mat::Dict{String,Any}, filename::AbstractString)
    open(filename, "w") do io
	    println(io, "# Material name:")
	    println(io, "# ", mat["material_name"])
	    println(io)
	    println(io, "# Number of elements:")
	    println(io, mat["n_elements"])
	    println(io)
	    println(io, "# Density:")
	    println(io, @sprintf("%.10f", mat["density"]))
	    println(io)
	    println(io, "# Atomic number(s) and corresponding mass fraction(s):")
	    for row in eachrow(mat["composition"])
	        Z, frac = row
	        println(io, @sprintf("%-4.0f %-12.10f     #", Z, frac), ATOMIC_SYMBOLS[Z])
	    end
	end
end

function bbox_3d(mask)::Union{Nothing, NTuple{3, UnitRange{Int}}}
    @assert ndims(mask) == 3 "bbox_3d expects a 3D array"
    nz = mask .!= 0
    any(nz) || return nothing

    rows_nonzero = vec(any(nz, dims=(2,3)))  # reduce over cols & slices
    cols_nonzero = vec(any(nz, dims=(1,3)))  # reduce over rows & slices
    slc_nonzero  = vec(any(nz, dims=(1,2)))  # reduce over rows & cols

    i1 = findfirst(rows_nonzero); isnothing(i1) && return nothing
    i2 = findlast(rows_nonzero)
    j1 = findfirst(cols_nonzero); isnothing(j1) && return nothing
    j2 = findlast(cols_nonzero)
    k1 = findfirst(slc_nonzero);  isnothing(k1) && return nothing
    k2 = findlast(slc_nonzero)

    return (i1:i2, j1:j2, k1:k2)
end

# ------------------------------------------------------------
# Save JSON
# ------------------------------------------------------------

function save_json_pretty(path::AbstractString, manifest::Dict)
    # Start building string manually
    open(path, "w") do io
        write(io, "{\n")
        n = length(manifest)
        keys_list = collect(keys(manifest))
        for (i, k) in enumerate(keys_list)
            v = manifest[k]
            write(io, "    \"$(k)\": ")
            if isa(v, AbstractVector)
                write(io, "[\n")
                for (j, elem) in enumerate(v)
                    if isa(elem, String)
                        write(io, "        \"$(elem)\"")
                    else
                        write(io, "        $(elem)")
                    end
                    write(io, j < length(v) ? ",\n" : "\n")
                end
                write(io, "    ]")
            else
                write(io, "$(v)")
            end
            write(io, i < n ? ",\n" : "\n")
        end
        write(io, "}\n")
    end
end

# Trim numbers
function trim_numbers(structure_map::Dict{Int64, String})

	new_map = Dict{Int, String}()
	
    for (k, v) in structure_map
        # Remove leading digits + underscore only if present
        trimmed = replace(v, r"^\d+_" => "")
        new_map[k] = trimmed
    end

	return new_map
end

# ------------------------------------------------------------
# Relabel zero islands
# ------------------------------------------------------------

# --------- 2D per-slice (ignore 3D tunnels) ----------
function relabel_zero_islands_2d!(L::AbstractArray{<:Integer,3}; newlabel::Int=10)
    nx, ny, nz = size(L)
    neigh = ((-1,0),(1,0),(0,-1),(0,1))
    @inbounds for k in 1:nz
        ext = falses(nx, ny)
        q = Tuple{Int,Int}[]
        S = @view L[:,:,k]

        # border zeros in this slice
        for i in (1,nx), j in 1:ny
            if S[i,j] == 0 && !ext[i,j]; ext[i,j]=true; push!(q,(i,j)); end
        end
        for i in 1:nx, j in (1,ny)
            if S[i,j] == 0 && !ext[i,j]; ext[i,j]=true; push!(q,(i,j)); end
        end

        # BFS
        while !isempty(q)
            (i,j) = pop!(q)
            for (di,dj) in neigh
                ii, jj = i+di, j+dj
                if 1 <= ii <= nx && 1 <= jj <= ny
                    if S[ii,jj] == 0 && !ext[ii,jj]
                        ext[ii,jj] = true
                        push!(q,(ii,jj))
                    end
                end
            end
        end

        S[(S .== 0) .& .!ext] .= newlabel
    end
    return L
end

# ------------------------------------------------------------
# Function to generate catsim phantom format (change for simulation)
# ------------------------------------------------------------
function generate_catsim_phantom(phantom_shift::Array{Int,3},
                                       phantom_materials::Vector{String},
                                       material_map::Dict{Int,String},
                                       material_list::Dict{String,Any},
                                       voxel_size::Tuple{Float64,Float64,Float64},
                                       output_base_dir::AbstractString,
										file_name::AbstractString,
										zrange::UnitRange{Int};)

	raw_dir = joinpath(output_base_dir, "raw_masks")
	mkpath(raw_dir)
	
    mat_dir = joinpath(output_base_dir, "materials")
    mkpath(mat_dir)

    #rows, cols, slices = size(phantom_shift)
	vol = @view phantom_shift[:, :, zrange]
	cols, rows, slices = size(vol)


    # JSON manifest
    manifest = Dict(
        "n_materials" => 0,
        "mat_name" => String[],
        "volumefractionmap_filename" => String[],
        "volumefractionmap_datatype" => String[],
        "cols" => Int[],
        "rows" => Int[],
        "slices" => Int[],
        "x_size" => Float64[],
        "y_size" => Float64[],
        "z_size" => Float64[],
        "x_offset" => Float64[],
        "y_offset" => Float64[],
        "z_offset" => Float64[]
    )

    for mat_name in phantom_materials
        # Find intensity corresponding to this material
        #intensity = first(k for (k,v) in material_map if v == mat_name)

        # Find material properties
        mat = material_list[mat_name]
		ids = Set([k for (k,v) in material_map if v == mat_name])
		
        # Create mask
        mask = Array{Int8}(undef, cols, rows, slices)
		@inbounds for i in 1:cols, j in 1:rows, k in 1:slices
        	mask[i,j,k] = (vol[i,j,k] in ids) ? Int8(1) : Int8(0)
    	end

		# Calculate offsets
		bb = bbox_3d(mask)                           # returns (I,J,K) or nothing
   		bb === nothing && return nothing
    	I, J, K = bb

		x_off = cols/2 - (first(I) - 0.5)
    	y_off = rows/2 - (rows - last(J) - 0.5)
    	z_off = slices/2 - first(K)  + 0.5


		# Calculate number of cols, rows, slices for each material mask
		sub = @view mask[I, J, K]                    # shape: (nx_crop, ny_crop, nz_crop)
   		nx = size(sub, 1)
    	ny = size(sub, 2)
    	nz = size(sub, 3)
		
        # Save RAW
        raw_fname = mat_name
 		raw_path = joinpath(raw_dir, raw_fname)
        #save_raw_mask(mask, raw_path)
		open(raw_path, "w") do io
        	write(io, sub)
		end

        # Save material file
        mat_fname = mat_name
        mat_path = joinpath(mat_dir, mat_fname)
        save_material_file(mat, mat_path)

        # Update JSON manifest
        push!(manifest["mat_name"], mat_name)
        push!(manifest["volumefractionmap_filename"], "raw_masks/" * raw_fname)
        push!(manifest["volumefractionmap_datatype"], "int8")
        push!(manifest["cols"], nx)#cols)
        push!(manifest["rows"], ny)#rows)
        push!(manifest["slices"], nz)#slices)
        push!(manifest["x_size"], voxel_size[1])
        push!(manifest["y_size"], voxel_size[2])
        push!(manifest["z_size"], voxel_size[3])
        push!(manifest["x_offset"], x_off)
        push!(manifest["y_offset"], y_off)
        push!(manifest["z_offset"], z_off)
    end

    manifest["n_materials"] = length(phantom_materials)

    # Save JSON manifest
    json_path = joinpath(output_base_dir, string(file_name, ".json"))
    #open(json_path, "w") do io
    #    JSON.print(io, manifest)#; indent=4)
    #end
	save_json_pretty(json_path, manifest)

    println("Generated CatSim phantom in $output_base_dir")
end

# ------------------------------------------------------------
# Import All Data
# ------------------------------------------------------------
data_dir = joinpath(@__DIR__, "data")
material_dir = joinpath(@__DIR__, "materials")

# Load in P1 Phantom Array
p1_x, p1_y, p1_z = 400, 400, 400  # adjust to your phantom dimensions
P1_raw_file = Array{UInt16}(undef, p1_x, p1_y, p1_z) # for 16 bit

open(joinpath(data_dir, "P1_brain_all_2020_RAW_400_400_400.raw")) do io # put directory to RAW file
   	read!(io, P1_raw_file)
end
P1_raw_file = Int.(P1_raw_file)

relabel_zero_islands_2d!(P1_raw_file, newlabel=10)

# Load in P2 Phantom Array
p2_x, p2_y, p2_z = 400, 400, 400  # adjust to your phantom dimensions
P2_raw_file = Array{UInt16}(undef, p2_x, p2_y, p2_z) # for 16 bit

open(joinpath(data_dir,"P2_brain_all_2020_RAW_400_400_400.raw")) do io # put directory to RAW file
   	read!(io, P2_raw_file)
end
P2_raw_file = Int.(P2_raw_file)

# Load in structure map
P1_table = CSV.read(joinpath(data_dir,"P1_voxelize_table.txt"), DataFrame; delim='\t', header=false)
	
rename!(P1_table, [:Structure, :Value])

P1_structure_map = Dict(P1_table.Value .=> P1_table.Structure)

# do same for P2
P2_table = CSV.read(joinpath(data_dir,"P2_vozelize_table.txt"), DataFrame; delim='\t', header=false)

rename!(P2_table, [:Structure, :Value])

P2_structure_map = Dict(P2_table.Value .=> P2_table.Structure)

# ------------------------------------------------------------
# Set up Phantom Materials and Material Map
# ------------------------------------------------------------
const material_map = Dict(
	0 => "empty",
	1 => "ncat_muscle",
	2 => "airway",
	3 => "ncat_dry_spine",
	5 => "ncat_cartilage",
	10 => "tissue_soft_icru-44",
	13 => "ncat_skull",
	17 => "csf",
	18 => "gray_matter",
	19 => "white_matter",
	21 => "artery_blood",
	22 => "vein_blood"
)

# Initialize dictionary to hold all materials
matfile = matread(joinpath(data_dir,"updated_materialsXCAT.mat"))
materialsXCAT = matfile["updated_materialsXCAT"]  # should be an Array{Any,1} of strings
	
n_materials = length(materialsXCAT)
material_list = Dict{String, Any}()
phantom_materials = String[]
unique_ids = sort(unique(P1_raw_file))

for i in 0:n_materials
	# Save only the materials that are in the phantom
	if i in unique_ids
		push!(phantom_materials, material_map[i])
	else
		continue
	end
end

# Import the elemental compositions that correspond to those materials
# Change directory to where your materials are located
for i in phantom_materials
		materials = joinpath(material_dir, "$i")
				
		material_list[i] = load_material_file(materials)
end	

# ------------------------------------------------------------
# Load in Structure Info and Iodine Mass Arrays
# ------------------------------------------------------------
structure_info = matread(joinpath(dir_data, "structure_info.mat"))

const info_tables = Dict(
    artery_info => structure_info["artery_info"]
    vein_info => structure_info["vein_info"]
    gm_info => structure_info["gm_info"]
    wm_info => structure_info["wm_info"]
    )

iodine_mass_data = matread(joinpath(dir_data,"iodine_mass_data.mat"))

const iodine_matrices = Dict(
    iodine_artery => iodine_mass_data["mass_arteries"] # (in mg / mL) calculated from artery_volume * conc_artery 
    iodine_vein => iodine_mass_data["mass_vein"] # (in mg / mL) calculated from vein_volume * conc_vein 
    iodine_gm => iodine_mass_data["mass_gm"] # (in mg / g) calculated from gm_volume * conc_gm_mg_per_g
    iodine_wm => iodine_mass_data["mass_wm"] # (in mg / g) calculated from wm_volume * conc_wm_mg_per_g
    )
# ------------------------------------------------------------
# Main function that updates iodine in structures
# ------------------------------------------------------------
function update_structures!(
	new_phantom_shift::Array{Int64,3}, # Phantom with shifted values
	structure_map::Dict{Int64, String}, # Ex) P2_structure_map
	tissue_prefix::String, # "5" for arteries, "4" for veins, "3" for white matter, "2" for gray matter
	raw_file::Array{Int64,3}, # Ex) P2_raw_file
	material_map::Dict{Int64, String}, # Generated from baseline phantom
	phantom_material::Vector{String}, # Generated from baseline phantom
	material_list::Dict{String, Any}, # Generated from baseline phantom
	info_table::Dict{String, Any}, # Ex) artery_info, where the names/indices/structure volumes are
	iodine_matrix::Matrix{Float64}, # iodine_artery, etc. What the mass of iodine is at specific timepoint
	t_contrast::Int64, # The time point where to find how much mass of iodine to add
	material::String # Replace with whatever you material "gray_matter", "white_matter", "vein_blood" for whichever material
)
	# --- 1. Extract tissue segment intensities and names ---
    entries = filter(kv -> startswith(kv[2], tissue_prefix), structure_map)
    ids = collect(keys(entries))
    names = [replace(v, r"^\d{4}_" => "") for v in values(entries)]

	# Sort the gm and wm names in order since we will be using index values instead of names for gm/wm
	if material == "gray_matter" || material == "white_matter"
    	# sort both ids and names by ascending id
    	sorted_pairs = sort(collect(zip(ids, names)), by = x -> x[1])
   		ids = [p[1] for p in sorted_pairs]
    	names = [p[2] for p in sorted_pairs]
	end

	# --- 2. Update phantom array ---
    for i in 1:length(ids)
        idxs = findall(==(ids[i]), raw_file)
        isempty(idxs) && continue
        new_phantom_shift[idxs] .= ids[i]
    end

	# --- 3. Update material map ---
    new_map = Dict(filter(kv -> kv[2] != material, material_map))
    for (id,name) in zip(ids, names)
        new_map[id] = name
    end
    unique_vals = unique(vec(new_phantom_shift))
    new_map = Dict(k => v for (k,v) in new_map if k in unique_vals)

	
    # --- 4. Update phantom materials ---
	new_phantom_materials = copy(phantom_materials)

	if material != "gray_matter" && material != "white_matter"
    	# only remove if it's an artery/vein being replaced
    	filter!(x -> x != material, new_phantom_materials)
	end
    #new_phantom_materials = filter(x -> x != material, phantom_materials)
    append!(new_phantom_materials, names)
    names_in_phantom = Set(new_map[k] for k in unique_vals if haskey(new_map,k))
    new_phantom_materials = filter(name -> name in names_in_phantom, 						new_phantom_materials)

	# --- 5. Update material list ---
    new_list = Dict(filter(kv -> kv[1] != material, material_list))

    for (i, name) in enumerate(names)
        new_list[name] = deepcopy(material_list[material])

		# Since Sarah's code leaves mass of iodine in gm and wm in mg, we have to dela with that here and convert it to grams
		if material == "gray_matter" || material == "white_matter"
			segment_volume = info_table["volume"][i]
			density = new_list[name]["density"]
			segment_mass = density * segment_volume
			mass_I = iodine_matrix[i, t_contrast] / 1000
		else
        	row = findfirst(==(name), info_table["name"])
        	row === nothing && continue  
        	segment_volume = info_table["volume"][row]
			density = new_list[name]["density"]
        	segment_mass = density * segment_volume
        	mass_I = iodine_matrix[row[2], t_contrast]
		end
		
        f_I = mass_I / (mass_I + segment_mass)
        f_s = 1 - f_I

        comp = new_list[name]["composition"]
        if any(comp[:,1] .== 53)
            comp[comp[:,1] .== 53, 2] .= f_I
        else
            comp = vcat(comp, [53 f_I])
        end
        for r in 1:size(comp,1)
            comp[r,1] != 53 && (comp[r,2] *= f_s)
        end

        new_list[name]["composition"] = comp
        new_list[name]["density"] = (segment_mass + mass_I) / segment_volume
        new_list[name]["n_elements"] = size(comp,1)
        new_list[name]["material_name"] = name
    end

    return new_phantom_shift, new_map, new_phantom_materials, new_list
end

# ------------------------------------------------------------
# High-level wrapper around update_structures! (from your code)
# ------------------------------------------------------------
function update_phantom_for_contrast!(
    new_phantom_shift::Array{Int64,3},
    structure_map::Dict{Int64,String},
    raw_file::Array{Int64,3},
    material_map::Dict{Int64,String},
    phantom_materials::Vector{String},
    material_list::Dict{String,Any},
    info_tables::Dict{String,Any},
    iodine_matrices::Dict{String,Matrix{Float64}},
    t_contrast::Int64
)
    # Tissue configuration
    tissue_specs = [
        ("artery_blood", "5"),
        ("vein_blood", "4"),
        ("white_matter", "3"),
        ("gray_matter", "2")
    ]

    new_phantom_shift = copy(new_phantom_shift)
    new_material_map = deepcopy(material_map)
    new_phantom_materials = copy(phantom_materials)
    new_material_list = deepcopy(material_list)

    for (material_name, prefix) in tissue_specs
        phantom_shift, map_tmp, materials_tmp, list_tmp = update_structures!(
            new_phantom_shift,
            structure_map,
            prefix,
            raw_file,
            new_material_map,
            new_phantom_materials,
            new_material_list,
            info_tables[material_name],
            iodine_matrices[material_name],
            t_contrast,
            material_name
        )

        new_phantom_shift .= phantom_shift
        new_material_map = merge(new_material_map, map_tmp)
        new_phantom_materials = union(new_phantom_materials, materials_tmp)
        merge!(new_material_list, list_tmp)
    end

    return new_phantom_shift, new_material_map, new_phantom_materials, new_material_list
end

# ------------------------------------------------------------
# Call Function
# ------------------------------------------------------------
const t_contrast = 10001

# Example:

# new_phantom_shift, new_material_map, new_phantom_materials, new_material_list =
#    	update_phantom_for_contrast!(
#        	P1_raw_file,
#        	P2_structure_map,
#        	P2_raw_file,
#        	material_map,
#        	phantom_materials,
#        	material_list,
#        	info_tables,
#        	iodine_matrices,
#        	t_contrast)


export P1_raw_file, P2_raw_file, P2_structure_map, material_map, phantom_materials
export material_list, info_tables, iodine_matricies
export t_contrast
export update_phantom_for_contrast!
export update_structures!
