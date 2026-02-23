"""
    xcat_io.jl

XCAT phantom I/O helpers for brain perfusion CT simulation.

Provides `update_structures!` to stamp XCAT segment IDs from a reference subject
into a target phantom array, and populate per-segment iodine-doped materials.
"""

# =============================================================================
# update_structures!
# =============================================================================

"""
    update_structures!(new_phantom_shift, structure_map, tissue_prefix,
                       raw_file, base_sym, base_mat, info_table,
                       iodine_matrix, t_contrast) -> Dict{Int, XA.Material}

Stamp XCAT segment IDs from a reference phantom (`raw_file`) into `new_phantom_shift`,
and build per-segment iodine-doped materials for each segment whose name starts with
`tissue_prefix`.

# Arguments
- `new_phantom_shift::Array{Int,3}`: Target array (modified in-place); typically a
  copy of another subject's raw phantom array.
- `structure_map::Dict{Int,String}`: ID → name mapping from `load_structure_map`.
- `tissue_prefix::String`: Only segments whose name *starts with* this string are
  processed (e.g. `"5"`, `"4"`, `"3"`, `"2"` for XCAT brain prefix codes).
- `raw_file::Array{Int,3}`: Reference subject's raw phantom (label source).
- `base_sym::Symbol`: Key into `MATERIALS_REGISTRY` for the base tissue
  (e.g. `:gray_matter`, `:white_matter`, `:blood`).
- `base_mat::XA.Material`: Pre-resolved material for `base_sym`.
- `info_table::Dict{String,Any}`: Must have `"name"::Vector{String}` and
  `"volume"::Vector{Float64}` (cm³ for `:gray_matter`/`:white_matter`,
  mm³ otherwise — automatic unit conversion applied).
- `iodine_matrix::Matrix{Float64}`: rows = segments (matching `info_table["name"]`),
  cols = time points; values in **mg**.
- `t_contrast::Int`: Column index (1-based) into `iodine_matrix` for the desired
  contrast time point.

# Returns
`Dict{Int, XA.Material}` mapping each stamped segment ID to its iodine-doped material.
IDs with no corresponding row in `info_table` are silently skipped.

# Example
```julia
gm_mats = update_structures!(
    P1_stamped, P2_structure_map, "5", P2_raw_file,
    :gray_matter, material_list[:gray_matter],
    info, iodine_matrix, t_contrast
)
```
"""
function update_structures!(
    new_phantom_shift::Array{Int,3},
    structure_map::Dict{Int,String},
    tissue_prefix::String,
    raw_file::Array{Int,3},
    base_sym::Symbol,
    base_mat::XA.Material,
    info_table::Dict{String,Any},
    iodine_matrix::Matrix{Float64},
    t_contrast::Int,
)::Dict{Int, XA.Material}

    # --- 1. Filter segments by prefix ---
    entries = filter(kv -> startswith(kv[2], tissue_prefix), structure_map)
    ids     = collect(keys(entries))
    names   = [replace(v, r"^\d{4}_" => "") for v in values(entries)]

    # --- 2. Stamp IDs from reference raw_file into target new_phantom_shift ---
    for id in ids
        idxs = findall(==(id), raw_file)
        isempty(idxs) && continue
        new_phantom_shift[idxs] .= id
    end

    # --- 3. Build iodine-doped material per segment ---
    seg_materials = Dict{Int, XA.Material}()
    density = ustrip(u"g/cm^3", base_mat.density)

    for (id, name) in zip(ids, names)
        row = findfirst(==(name), info_table["name"])
        row === nothing && continue

        # Volume units depend on tissue type
        if base_sym == :gray_matter || base_sym == :white_matter
            segment_volume = info_table["volume"][row]           # already cm³
        else
            segment_volume = info_table["volume"][row] / 1000.0  # mm³ → cm³
        end

        segment_mass = density * segment_volume
        mass_I       = iodine_matrix[row, t_contrast] / 1000.0  # mg → g

        seg_materials[id] = make_iodine_doped_material(
            name, base_mat, segment_volume, segment_mass, mass_I
        )
    end

    return seg_materials
end

export update_structures!
