"""
    SemanticClassification.jl

Intelligent semantic classification system for anatomical structures in CT phantom simulation.

# Structure ID Patterns (from XCAT voxelize tables):
- ID 0: Air/Empty (tendons, oral_cavity, etc.)
- ID 1: Soft tissue/muscle (musc*, eyes, tongue, parotid)
- ID 2: Airway (sinus, throat)
- ID 3: Bone (C3-C5 vertebrae)
- ID 5: Head
- ID 10: Soft tissue
- ID 13: Bone (skull)
- ID 14: Vertebral disks
- ID 17: CSF (ventricles)
- ID 18: Gray matter
- ID 19: White matter
- ID 21: Arteries (prefix 5xxx)
- ID 22: Veins (prefix 4xxx)

# Categories:
- :air_cavity: Hollow organs that should be air (esophagus, trachea, stomach)
- :soft_tissue: General soft tissue, muscle
- :bone: Skeletal structures
- :brain: Brain tissues (GM, WM)
- :blood_vessel: Arteries and veins (customizable for perfusion)
- :csf: Cerebrospinal fluid
- :airway: Respiratory passages
- :eye: Eye structures
- :cartilage: Cartilage
"""

module SemanticClassification

export 
    StructureCategory,
    SemanticMapping,
    PerfusionData,
    IodineMapping,
    XCATPhantom,
    classify_structure,
    get_category_material,
    is_customizable,
    get_vascular_path,
    classify_all_structures,
    load_perfusion_data,
    create_iodine_mapping,
    get_iodine_mass,
    load_voxelize_table,
    load_and_classify_phantom,
    generate_julia_mapping,
    load_xcat_raw,
    load_xcat_phantom,
    load_xcat_with_perfusion,
    load_phantom,
    apply_iodine_contrast,
    export_classification_summary,
    create_basis_phantom_components,
    is_mat_available,
    extract_dims_from_filename,
    detect_raw_dtype,
    get_category_color,
    generate_organ_colors,
    convert_organ_ids,
    create_compatible_phantom,
    HOLLOW_ORGANS,
    ARTERY_PATTERNS,
    VEIN_PATTERNS,
    BONE_PATTERNS,
    MUSCLE_PATTERNS,
    BRAIN_PATTERNS,
    SPECIAL_MATERIALS

const _MAT_AVAILABLE = Ref{Bool}(false)

function __init__()
    try
        @eval using MAT
        _MAT_AVAILABLE[] = true
    catch
    end
end

function is_mat_available()
    return _MAT_AVAILABLE[]
end

# =============================================================================
# Category Definitions
# =============================================================================

@enum StructureCategory begin
    CAT_AIR_CAVITY      # Hollow organs (esophagus, trachea, stomach lumen)
    CAT_SOFT_TISSUE    # Muscle, general soft tissue
    CAT_BONE           # Bone, cartilage
    CAT_BRAIN          # Gray/white matter
    CAT_BLOOD_VESSEL   # Arteries or veins (customizable)
    CAT_CSF            # Cerebrospinal fluid
    CAT_AIRWAY         # Sinus, throat
    CAT_EYE            # Eye structures
    CAT_CARTILAGE      # Cartilage
    CAT_UNKNOWN        # Unclassified
end

"""
    StructureCategory enum as symbols
"""
const CATEGORY_SYMBOLS = Dict(
    CAT_AIR_CAVITY => :air_cavity,
    CAT_SOFT_TISSUE => :soft_tissue,
    CAT_BONE => :bone,
    CAT_BRAIN => :brain,
    CAT_BLOOD_VESSEL => :blood_vessel,
    CAT_CSF => :csf,
    CAT_AIRWAY => :airway,
    CAT_EYE => :eye,
    CAT_CARTILAGE => :cartilage,
    CAT_UNKNOWN => :unknown
)

# =============================================================================
# Pattern Matching Rules
# =============================================================================

"""
Hollow organs that should be classified as air cavities (NOT muscle/tissue).
These structures refer to the hollow space inside, not the muscular wall.
"""
const HOLLOW_ORGANS = Set([
    "esophagus",
    "trachea",
    "stomach",
    "intestine",
    "colon",
    "rectum",
    "jejunum",
    "ileum",
    "duodenum",
    "gallbladder",
    "urinary_bladder",
    "urethra",
    "larynx",
    "pharynx",
    "nasopharynx",
    "oropharynx",
    "bronchi",
    "bronchiole",
    "alveoli",
    "lungs",
    "pleural",
    "pericardium",
    "heart",
    "sinus",
    "throat",
    "oral_cavity",
])

"""
    PerfusionData

Container for iodine contrast perfusion data from Sarah's code.
This data maps structure segments to iodine mass over time.
"""
struct PerfusionData
    mass_arteries::Matrix{Float64}    # (n_segments, n_timepoints) in mg
    mass_veins::Matrix{Float64}       # (n_segments, n_timepoints) in mg
    mass_gm::Matrix{Float64}          # (n_segments, n_timepoints) in mg/g tissue
    mass_wm::Matrix{Float64}          # (n_segments, n_timepoints) in mg/g tissue
    artery_names::Vector{String}
    vein_names::Vector{String}
    gm_names::Vector{String}
    wm_names::Vector{String}
    timepoints::Vector{Int}           # Time in milliseconds
end

"""
    load_perfusion_data(mat_path::String) -> PerfusionData

Load perfusion contrast data from Sarah's iodine_mass_data.mat file.
Requires MAT.jl to be installed.
"""
function load_perfusion_data(mat_path::String)
    if !_MAT_AVAILABLE[]
        error("MAT.jl is required for loading perfusion data. Install with: Pkg.add(\"MAT\")")
    end
    
    mat = MAT.matread(mat_path)
    
    mass_arteries = mat["mass_arteries"]
    mass_veins = mat["mass_vein"]
    mass_gm = mat["mass_gm"]
    mass_wm = mat["mass_wm"]
    
    n_timepoints = size(mass_arteries, 2)
    timepoints = Vector{Int}(0:1:(n_timepoints-1))
    
    n_arteries = size(mass_arteries, 1)
    n_veins = size(mass_veins, 1)
    n_gm = size(mass_gm, 1)
    n_wm = size(mass_wm, 1)
    
    artery_names = ["artery_$i" for i in 1:n_arteries]
    vein_names = ["vein_$i" for i in 1:n_veins]
    gm_names = ["gm_$i" for i in 1:n_gm]
    wm_names = ["wm_$i" for i in 1:n_wm]
    
    info_path = replace(mat_path, "iodine_mass_data.mat" => "structure_info.mat")
    if isfile(info_path)
        try
            info_mat = MAT.matread(info_path)
            if haskey(info_mat, "artery_names")
                artery_names = vec(info_mat["artery_names"])
            end
            if haskey(info_mat, "vein_names")
                vein_names = vec(info_mat["vein_names"])
            end
            if haskey(info_mat, "gm_names")
                gm_names = vec(info_mat["gm_names"])
            end
            if haskey(info_mat, "wm_names")
                wm_names = vec(info_mat["wm_names"])
            end
        catch e
            @warn "Could not load structure_info.mat: $e"
        end
    end
    
    return PerfusionData(
        mass_arteries, mass_veins, mass_gm, mass_wm,
        artery_names, vein_names, gm_names, wm_names,
        timepoints
    )
end

"""
    get_iodine_mass(p::PerfusionData, tissue_type::Symbol, segment_idx::Int, time_ms::Int) -> Float64

Get iodine mass (mg) for a specific tissue type, segment, and time point.
"""
function get_iodine_mass(p::PerfusionData, tissue_type::Symbol, segment_idx::Int, time_ms::Int)
    t_idx = time_ms + 1  # 1-indexed
    
    if tissue_type == :artery
        return p.mass_arteries[segment_idx, t_idx]
    elseif tissue_type == :vein
        return p.mass_veins[segment_idx, t_idx]
    elseif tissue_type == :gm
        return p.mass_gm[segment_idx, t_idx]
    elseif tissue_type == :wm
        return p.mass_wm[segment_idx, t_idx]
    else
        error("Unknown tissue type: $tissue_type")
    end
end

"""
    IodineMapping

Mapping from phantom structure IDs to perfusion data indices.
This is needed because P1/P2 use different ID schemes than Sarah's data.
"""
struct IodineMapping
    # P1 ID -> (tissue_type, segment_index)
    p1_iodine_map::Dict{Int, Tuple{Symbol, Int}}
    
    # P2 ID -> (tissue_type, segment_index)  
    p2_iodine_map::Dict{Int, Tuple{Symbol, Int}}
    
    # Tissue type counts
    n_arteries::Int
    n_veins::Int
    n_gm::Int
    n_wm::Int
end

"""
    create_iodine_mapping(perfusion_data::PerfusionData, p1_table_path::String, p2_table_path::String) -> IodineMapping

Create mapping from phantom IDs to perfusion data indices.

# P1 ID scheme (from P1_voxelize_table.txt):
- ID 21: All arteries
- ID 22: All veins
- ID 18: Gray matter
- ID 19: White matter

# P2 ID scheme (from P2_vozelize_table.txt):
- IDs 703-1102: Artery segments
- IDs 468-702: Vein segments
- IDs 252-329: GM segments
- IDs 338-454: WM segments
"""
function create_iodine_mapping(perfusion_data::PerfusionData, p1_table_path::String, p2_table_path::String)
    p1_map = Dict{Int, Tuple{Symbol, Int}}()
    p2_map = Dict{Int, Tuple{Symbol, Int}}()
    
    p1_map[21] = (:artery, 1)
    p1_map[22] = (:vein, 1)
    p1_map[18] = (:gm, 1)
    p1_map[19] = (:wm, 1)
    
    p2_sm = load_voxelize_table(p2_table_path)
    
    for (name, p2_id) in p2_sm
        m = match(r"^(\d+)_", name)
        if m !== nothing
            num_id = parse(Int, m.captures[1])
            
            if num_id >= 5001 && num_id <= 5400
                sarah_idx = num_id - 5000
                if sarah_idx >= 1 && sarah_idx <= length(perfusion_data.artery_names)
                    p2_map[p2_id] = (:artery, sarah_idx)
                end
            elseif num_id >= 4006 && num_id <= 4240
                sarah_idx = num_id - 4005
                if sarah_idx >= 1 && sarah_idx <= length(perfusion_data.vein_names)
                    p2_map[p2_id] = (:vein, sarah_idx)
                end
            elseif num_id >= 2001 && num_id < 3000
                sarah_idx = num_id - 2000
                if sarah_idx >= 1 && sarah_idx <= length(perfusion_data.gm_names)
                    p2_map[p2_id] = (:gm, sarah_idx)
                end
            elseif num_id >= 3001 && num_id < 4000
                sarah_idx = num_id - 3000
                if sarah_idx >= 1 && sarah_idx <= length(perfusion_data.wm_names)
                    p2_map[p2_id] = (:wm, sarah_idx)
                end
            end
        end
        
        if p2_id >= 703 && p2_id <= 1102
            sarah_idx = p2_id - 702
            if sarah_idx >= 1 && sarah_idx <= length(perfusion_data.artery_names)
                p2_map[p2_id] = (:artery, sarah_idx)
            end
        elseif p2_id >= 468 && p2_id <= 702
            sarah_idx = p2_id - 467
            if sarah_idx >= 1 && sarah_idx <= length(perfusion_data.vein_names)
                p2_map[p2_id] = (:vein, sarah_idx)
            end
        elseif p2_id >= 252 && p2_id <= 329
            sarah_idx = p2_id - 251
            if sarah_idx >= 1 && sarah_idx <= length(perfusion_data.gm_names)
                p2_map[p2_id] = (:gm, sarah_idx)
            end
        elseif p2_id >= 338 && p2_id <= 454
            sarah_idx = p2_id - 337
            if sarah_idx >= 1 && sarah_idx <= length(perfusion_data.wm_names)
                p2_map[p2_id] = (:wm, sarah_idx)
            end
        end
    end
    
    return IodineMapping(
        p1_map,
        p2_map,
        length(perfusion_data.artery_names),
        length(perfusion_data.vein_names),
        length(perfusion_data.gm_names),
        length(perfusion_data.wm_names)
    )
end

"""
Patterns indicating arterial structures (customizable for perfusion)
"""
const ARTERY_PATTERNS = [
    r"^4\d{3}_.*_artery",
    r"_artery$",
    r"_arterial$",
    r"internal_carotid",
    r"external_carotid",
    r"middle_cerebral",
    r"anterior_cerebral",
    r"posterior_cerebral",
    r"vertebral",
    r"basilar",
    r"ophthalmic",
    r"internal_jugular",
    r"mca_",
    r"aca_",
    r"pca_",
    r"mca\d",
    r"aca\d",
    r"pca\d",
]

"""
Patterns indicating venous structures (customizable for perfusion)
"""
const VEIN_PATTERNS = [
    r"^4\d{3}_.*_vein",
    r"_vein$",
    r"_venous$",
    r"jugular",
    r"sigmoid_sinus",
    r"transverse_sinus",
    r"superior_sagittal_sinus",
    r"straight_sinus",
    r"internal_cerebral_vein",
    r"basal_vein",
    r"sepal_vein",
    r"central_vein",
    r"precentral_vein",
    r"trolard",
]

"""
Patterns indicating bone structures
"""
const BONE_PATTERNS = [
    r"bone$",
    r"skull",
    r"spine",
    r"vertebra",
    r"C[1-7]",
    r"T[1-9]",
    r"T1[0-2]",
    r"L[1-5]",
    r"atlas",
    r"axis",
    r"rib",
    r"sternum",
    r"pelvis",
    r"mandible",
    r"maxilla",
    r"hyoid",
    r"nasal",
    r"frontal",
    r"parietal",
    r"temporal",
    r"occipital",
    r"sphenoid",
    r"ethmoid",
    r"vomer",
    r"zygomatic",
    r"cartilage",
    r"disk\d",
    r"femur",
    r"humerus",
    r"tibia",
    r"fibula",
    r"ulna",
    r"radius",
    r"clavicle",
    r"scapula",
    r"patella",
]

"""
Patterns indicating muscle/soft tissue
"""
const MUSCLE_PATTERNS = [
    r"^musc\d+",
    r"tongue",
    r"parotid",
    r"tonsil",
    r"pituitary",
    r"lens",
    r"sclera",
    r"cornea",
    r"retina",
    r"optic_nerve",
]

"""
Patterns indicating brain tissue
"""
const BRAIN_PATTERNS = [
    r"_gm_",
    r"_wm_",
    r"gray_matter",
    r"white_matter",
    r"brain",
    r"lobe",
    r"ventricle",
    r"cerebral_aqueduct",
]

"""
Map IDs to XCAT material names (for BasisSimulator compatibility)
"""
const ID_TO_MATERIAL = Dict{Int, String}(
    0 => "air",
    1 => "ncat_muscle",
    2 => "airway",
    3 => "ncat_dry_spine",
    5 => "head",
    10 => "tissue_soft_icru-44",
    13 => "ncat_skull",
    14 => "cartilage",
    17 => "csf",
    18 => "gray_matter",
    19 => "white_matter",
    21 => "artery_blood",
    22 => "vein_blood",
)

"""
Map IDs to XCAT material names (for BasisSimulator compatibility)
"""
const ID_TO_CATEGORY = Dict{Int, StructureCategory}(
    0 => CAT_UNKNOWN,  # ID 0 is background/air, not a specific organ
    1 => CAT_SOFT_TISSUE,
    2 => CAT_AIRWAY,
    3 => CAT_BONE,
    5 => CAT_SOFT_TISSUE,
    10 => CAT_SOFT_TISSUE,
    13 => CAT_BONE,
    14 => CAT_CARTILAGE,
    17 => CAT_CSF,
    18 => CAT_BRAIN,
    19 => CAT_BRAIN,
    21 => CAT_BLOOD_VESSEL,
    22 => CAT_BLOOD_VESSEL,
)

# =============================================================================
# Special Material Overrides
# =============================================================================

"""
Special structure names that need explicit material assignment
"""
const SPECIAL_MATERIALS = Dict{String, Symbol}(
    "esophagus" => :air,
    "trachea" => :air,
    "stomach" => :air,
    "sinus" => :air,
    "throat" => :air,
    "bronchi" => :air,
    "oral_cavity" => :air,
    "eye" => :soft_tissue,
    "l_eye" => :soft_tissue,
    "r_eye" => :soft_tissue,
    "lens" => :soft_tissue,
    "sclera" => :soft_tissue,
    "cornea" => :soft_tissue,
    "retina" => :soft_tissue,
    "optic_nerve" => :soft_tissue,
    "brain" => :brain,
    "gray_matter" => :brain,
    "white_matter" => :brain,
)

"""
Structures that are customizable for perfusion simulation
"""
const CUSTOMIZABLE_PATTERNS = [
    r"_artery$",
    r"_vein$",
    r"internal_carotid",
    r"middle_cerebral",
    r"anterior_cerebral",
    r"posterior_cerebral",
    r"vertebral",
    r"basilar",
    r"jugular",
]

# =============================================================================
# Classification Functions
# =============================================================================

"""
    classify_structure(name::String, id::Int) -> StructureCategory

Classify a structure by name and ID using semantic understanding.

# Convenience method - classifies by name only (uses ID=0 as default)
"""
function classify_structure(name::String)
    return classify_structure(name, 0)
end

"""
Classify a structure by name and ID using semantic understanding.
"""
function classify_structure(name::String, id::Int)
    name_lower = lowercase(name)
    
    for hollow in HOLLOW_ORGANS
        if occursin(name_lower, hollow)
            return CAT_AIR_CAVITY
        end
    end
    
    # Only use ID lookup for known categories, not CAT_UNKNOWN
    if haskey(ID_TO_CATEGORY, id)
        cat = ID_TO_CATEGORY[id]
        if cat != CAT_UNKNOWN
            if id == 21 || id == 22
                if matches_any_pattern(name_lower, ARTERY_PATTERNS)
                    return CAT_BLOOD_VESSEL
                elseif matches_any_pattern(name_lower, VEIN_PATTERNS)
                    return CAT_BLOOD_VESSEL
                end
                return CAT_BLOOD_VESSEL
            end
            return cat
        end
        # If ID maps to CAT_UNKNOWN, fall through to pattern matching
    end
    
    if id >= 703 && id <= 1102
        return CAT_BLOOD_VESSEL
    end
    if id >= 468 && id <= 702
        return CAT_BLOOD_VESSEL
    end
    if id >= 200 && id <= 329
        if occursin(r"_gm_", name_lower) || occursin(r"gm_", name_lower)
            return CAT_BRAIN
        end
    end
    if id >= 330 && id <= 470
        if occursin(r"_wm_", name_lower) || occursin(r"wm_", name_lower)
            return CAT_BRAIN
        end
    end
    
    m = match(r"^(\d+)_", name)
    if m !== nothing
        num_prefix = parse(Int, m.captures[1])
        
        if num_prefix >= 5000
            return CAT_BLOOD_VESSEL
        elseif num_prefix >= 4000 && num_prefix < 5000
            return CAT_BLOOD_VESSEL
        elseif num_prefix >= 2000 && num_prefix < 3000
            return CAT_BRAIN
        elseif num_prefix >= 3000 && num_prefix < 4000
            return CAT_BRAIN
        end
    end
    
    if matches_any_pattern(name_lower, BONE_PATTERNS)
        return CAT_BONE
    elseif matches_any_pattern(name_lower, BRAIN_PATTERNS)
        return CAT_BRAIN
    elseif matches_any_pattern(name_lower, MUSCLE_PATTERNS)
        return CAT_SOFT_TISSUE
    elseif matches_any_pattern(name_lower, ARTERY_PATTERNS) || matches_any_pattern(name_lower, VEIN_PATTERNS)
        return CAT_BLOOD_VESSEL
    end
    
    return CAT_UNKNOWN
end

function matches_any_pattern(text::String, patterns::Vector{Regex})
    for pattern in patterns
        if occursin(pattern, text)
            return true
        end
    end
    return false
end

"""
    get_category_material(category::StructureCategory) -> Symbol

Get the recommended material symbol for a category.
"""
function get_category_material(category::StructureCategory)
    return get(CATEGORY_MATERIALS, category, :soft_tissue)
end

const CATEGORY_MATERIALS = Dict{StructureCategory, Symbol}(
    CAT_AIR_CAVITY => :air,
    CAT_SOFT_TISSUE => :soft_tissue,
    CAT_BONE => :bone,
    CAT_BRAIN => :brain,
    CAT_BLOOD_VESSEL => :blood,
    CAT_CSF => :csf,
    CAT_AIRWAY => :air,
    CAT_EYE => :soft_tissue,
    CAT_CARTILAGE => :cartilage,
    CAT_UNKNOWN => :soft_tissue,
)

"""
    is_customizable(name::String, category::StructureCategory) -> Bool

Check if a structure can be customized for perfusion simulation.
"""
function is_customizable(name::String, category::StructureCategory)
    category == CAT_BLOOD_VESSEL && return true
    name_lower = lowercase(name)
    return matches_any_pattern(name_lower, CUSTOMIZABLE_PATTERNS)
end

"""
    classify_all_structures(structure_map::Dict{String, Int}) -> Dict{Int, NamedTuple}
"""
function classify_all_structures(structure_map::Dict{String, Int})
    results = Dict{Int, NamedTuple}()
    
    for id in unique(values(structure_map))
        names_for_id = [name for (name, nid) in structure_map if nid == id]
        isempty(names_for_id) && continue
        name = first(names_for_id)
        
        category = classify_structure(name, id)
        material = get_category_material(category)
        customizable = is_customizable(name, category)
        
        results[id] = (
            names=names_for_id,
            category=CATEGORY_SYMBOLS[category],
            material=material,
            customizable=customizable,
            base_id=id,
        )
    end
    
    return results
end

"""
    get_vascular_path(name::String, id::Int) -> Symbol

Identify the vascular path for a blood vessel.
"""
function get_vascular_path(name::String, id::Int)
    name_lower = lowercase(name)
    
    if id >= 468 && id <= 702
        if occursin(r"jugular", name_lower)
            return :jugular
        elseif occursin(r"sigmoid_sinus", name_lower)
            return :sigmoid_sinus
        elseif occursin(r"transverse_sinus", name_lower)
            return :lateral_sinus
        elseif occursin(r"sagittal_sinus", name_lower)
            return :deep_venous
        elseif occursin(r"straight_sinus", name_lower)
            return :deep_venous
        elseif occursin(r"internal_cerebral_vein", name_lower)
            return :deep_venous
        elseif occursin(r"basal_vein", name_lower)
            return :deep_venous
        elseif occursin(r"septal_vein", name_lower)
            return :deep_venous
        elseif occursin(r"central_vein", name_lower)
            return :central_veins
        elseif occursin(r"precentral_vein", name_lower)
            return :precentral_veins
        elseif occursin(r"superficial", name_lower)
            return :superficial_veins
        elseif occursin(r"inferior_cerebral", name_lower)
            return :inferior_cerebral_veins
        elseif occursin(r"_vein", name_lower)
            return :jugular
        end
        return :vein
    end
    
    if occursin(r"internal_carotid", name_lower)
        return :internal_carotid
    elseif occursin(r"middle_cerebral|mca", name_lower)
        return :mca
    elseif occursin(r"anterior_cerebral|aca", name_lower)
        return :aca
    elseif occursin(r"posterior_cerebral|pca", name_lower)
        return :pca
    elseif occursin(r"vertebral|basilar", name_lower)
        return :vertebral_basilar
    elseif occursin(r"ophthalmic", name_lower)
        return :ophthalmic
    end
    
    if occursin(r"jugular", name_lower)
        return :jugular
    elseif occursin(r"sagittal_sinus", name_lower)
        return :deep_venous
    elseif occursin(r"transverse_sinus", name_lower)
        return :lateral_sinus
    elseif occursin(r"straight_sinus", name_lower)
        return :deep_venous
    end
    
    return :other
end

# =============================================================================
# Export mapping for BasisSimulator
# =============================================================================

"""
    create_material_mapping(structure_map::Dict{String, Int}) -> Dict{Int, Symbol}
"""
function create_material_mapping(structure_map::Dict{String, Int})
    mapping = Dict{Int, Symbol}()
    
    for id in unique(values(structure_map))
        names_for_id = [name for (name, nid) in structure_map if nid == id]
        isempty(names_for_id) && continue
        name = first(names_for_id)
        
        category = classify_structure(name, id)
        mapping[id] = get_category_material(category)
    end
    
    return mapping
end

"""
    create_customizable_sets(structure_map::Dict{String, Int}) -> Dict{Symbol, Set{Int}}
"""
function create_customizable_sets(structure_map::Dict{String, Int})
    arteries = Set{Int}()
    veins = Set{Int}()
    
    for id in unique(values(structure_map))
        names_for_id = [name for (name, nid) in structure_map if nid == id]
        isempty(names_for_id) && continue
        name = first(names_for_id)
        
        category = classify_structure(name, id)
        if is_customizable(name, category)
            path = get_vascular_path(name, id)
            if path in (:mca, :aca, :pca, :internal_carotid, :vertebral_basilar, :ophthalmic, :other)
                push!(arteries, id)
            else
                push!(veins, id)
            end
        end
    end
    
    return Dict(:arteries => arteries, :veins => veins)
end

"""
    load_voxelize_table(filepath::String) -> Dict{String, Int}

Load a voxelize table (format: name\\tID) and return as Dict.
"""
function load_voxelize_table(filepath::String)
    structure_map = Dict{String, Int}()
    
    open(filepath, "r") do io
        for line in eachline(io)
            line = strip(line)
            isempty(line) && continue
            
            parts = split(line, '\t')
            length(parts) >= 2 || continue
            
            name = strip(parts[1])
            id_str = strip(parts[2])
            
            id = tryparse(Int, id_str)
            id === nothing && continue
            
            structure_map[name] = id
        end
    end
    
    return structure_map
end

"""
    load_and_classify_phantom(voxelize_table_path::String) -> Dict
"""
function load_and_classify_phantom(voxelize_table_path::String)
    structure_map = load_voxelize_table(voxelize_table_path)
    
    unique_ids = unique(values(structure_map))
    
    classifications = Dict{Int, NamedTuple}()
    material_mapping = Dict{Int, Symbol}()
    customizable_ids = Set{Int}()
    arteries = Set{Int}()
    veins = Set{Int}()
    
    for id in unique_ids
        names_for_id = [name for (name, nid) in structure_map if nid == id]
        isempty(names_for_id) && continue
        
        name = first(names_for_id)
        
        category = classify_structure(name, id)
        material = get_category_material(category)
        customizable = is_customizable(name, category)
        
        vascular_path = :other
        if category == CAT_BLOOD_VESSEL
            vein_count = 0
            artery_count = 0
            for n in names_for_id
                path = get_vascular_path(n, id)
                if path in (:jugular, :deep_venous, :lateral_sinus)
                    vein_count += 1
                elseif path in (:mca, :aca, :pca, :internal_carotid, :vertebral_basilar, :ophthalmic)
                    artery_count += 1
                end
            end
            if vein_count > artery_count
                vascular_path = :vein
            elseif artery_count > vein_count
                vascular_path = :artery
            end
        end
        
        classifications[id] = (;
            names=names_for_id,
            category=CATEGORY_SYMBOLS[category],
            material=material,
            customizable=customizable,
            vascular_path=vascular_path
        )
        
        material_mapping[id] = material
        
        if customizable
            push!(customizable_ids, id)
            if vascular_path == :vein
                push!(veins, id)
            else
                push!(arteries, id)
            end
        end
    end
    
    return (;
        structure_map=structure_map,
        classifications=classifications,
        material_mapping=material_mapping,
        customizable_ids=customizable_ids,
        arteries=arteries,
        veins=veins
    )
end

"""
    generate_julia_mapping(voxelize_table_path::String, output_path::String)
"""
function generate_julia_mapping(voxelize_table_path::String, output_path::String)
    result = load_and_classify_phantom(voxelize_table_path)
    
    open(output_path, "w") do io
        println(io, "# Auto-generated semantic classification mapping")
        println(io, "# Generated from: $voxelize_table_path")
        println(io)
        
        println(io, "# Material mapping: ID => Symbol")
        println(io, "const MATERIAL_MAPPING = Dict{Int, Symbol}(")
        for id in sort(collect(keys(result.material_mapping)))
            material = result.material_mapping[id]
            println(io, "    $id => :$material,")
        end
        println(io, ")")
        println(io)
        
        println(io, "# Customizable structure IDs (for perfusion simulation)")
        println(io, "const CUSTOMIZABLE_IDS = Set{Int}([")
        for id in sort(collect(result.customizable_ids))
            println(io, "    $id,")
        end
        println(io, "])")
        println(io)
        
        println(io, "# Artery IDs")
        println(io, "const ARTERY_IDS = Set{Int}([")
        for id in sort(collect(result.arteries))
            println(io, "    $id,")
        end
        println(io, "])")
        println(io)
        
        println(io, "# Vein IDs")
        println(io, "const VEIN_IDS = Set{Int}([")
        for id in sort(collect(result.veins))
            println(io, "    $id,")
        end
        println(io, "])")
    end
    
    println("Generated mapping saved to: $output_path")
end

# =============================================================================
# XCAT Phantom Loading
# =============================================================================

"""
    load_xcat_raw(filepath::String; dims=(400,400,400), dtype=UInt16) -> Array

Load XCAT phantom from RAW file.

# Arguments
- `filepath::String`: Path to RAW file
- `dims::Tuple{Int,Int,Int}`: Dimensions (cols, rows, slices) - auto-detected from filename if not provided
- `dtype::Type`: Data type (default UInt16)

# Auto-detection
If dims not provided, extracts from filename pattern: *_RAW_XXX_YYY_ZZZ.raw
Example: P1_brain_all_2020_RAW_400_400_400.raw -> (400, 400, 400)
"""
function load_xcat_raw(filepath::String; dims::Union{Tuple{Int,Int,Int}, Nothing}=nothing, dtype::Type=UInt16)
    # Auto-detect dimensions from filename if not provided
    if dims === nothing
        dims = extract_dims_from_filename(filepath)
    end
    
    cols, rows, slices = dims
    expected_size = cols * rows * slices * sizeof(dtype)
    
    if !isfile(filepath)
        error("File not found: $filepath")
    end
    
    actual_size = filesize(filepath)
    if actual_size != expected_size
        @warn "File size mismatch: expected $expected_size bytes, got $actual_size"
    end
    
    data = Vector{dtype}(undef, cols * rows * slices)
    open(filepath, "r") do io
        read!(io, data)
    end
    
    return reshape(data, (cols, rows, slices))
end

"""
    extract_dims_from_filename(filepath::String) -> Tuple{Int,Int,Int}

Extract dimensions from RAW file filename.
Pattern: *_RAW_XXX_YYY_ZZZ.raw or *_XXX_YYY_ZZZ.raw
Examples:
- P1_brain_all_2020_RAW_400_400_400.raw -> (400, 400, 400)
- P2_400_400_400.raw -> (400, 400, 400)
- xcat_512_512_256.raw -> (512, 512, 256)
"""
function extract_dims_from_filename(filepath::String)::Tuple{Int,Int,Int}
    filename = basename(filepath)
    
    # Try pattern: RAW_XXX_YYY_ZZZ
    m = match(r"RAW_(\d+)_(\d+)_(\d+)\.raw$", lowercase(filename))
    if m !== nothing
        return (parse(Int, m[1]), parse(Int, m[2]), parse(Int, m[3]))
    end
    
    # Try pattern: XXX_YYY_ZZZ.raw (at end of filename)
    m = match(r"_(\d+)_(\d+)_(\d+)\.raw$", filename)
    if m !== nothing
        return (parse(Int, m[1]), parse(Int, m[2]), parse(Int, m[3]))
    end
    
    # Try to get from file size if UInt16 (400*400*400 * 2 = 320000 bytes)
    file_size = filesize(filepath)
    for dims in [(512,512,512), (400,400,400), (256,256,256), (512,512,256), (400,400,200)]
        expected = prod(dims) * 2  # UInt16 = 2 bytes
        if file_size == expected
            @warn "Auto-detected dimensions $dims from file size. Verify this is correct."
            return dims
        end
    end
    
    error("Could not extract dimensions from filename: $filename. Please provide dims explicitly.")
end

"""
    detect_raw_dtype(filepath::String) -> Type

Detect the data type of a RAW file from its size.
"""
function detect_raw_dtype(filepath::String)::Type
    file_size = filesize(filepath)
    
    # Try common dimensions and data types
    for dims in [(400,400,400), (512,512,512), (256,256,256), (512,512,256)]
        # UInt16
        if file_size == prod(dims) * 2
            return UInt16
        end
        # UInt8
        if file_size == prod(dims) * 1
            return UInt8
        end
        # UInt32
        if file_size == prod(dims) * 4
            return UInt32
        end
    end
    
    # Default to UInt16
    @warn "Could not detect dtype, defaulting to UInt16"
    return UInt16
end

"""
    XCATPhantom
"""
struct XCATPhantom
    mask::Array{UInt16,3}
    voxel_size::NTuple{3,Float64}
    phantom_type::Symbol
    material_mapping::Dict{Int, Symbol}
    customizable_ids::Set{Int}
    artery_ids::Set{Int}
    vein_ids::Set{Int}
    iodine_mapping::Union{IodineMapping, Missing}
    perfusion_data::Union{PerfusionData, Missing}
end

"""
    load_xcat_phantom(raw_path, table_path; phantom_type, voxel_size_cm, dims, dtype)

Load and classify XCAT phantom from RAW file and voxelize table.

# Arguments
- `raw_path::String`: Path to RAW file
- `table_path::String`: Path to voxelize table
- `phantom_type::Symbol`: :P1 or :P2
- `voxel_size_cm::Tuple`: Voxel dimensions in cm (auto-computed from dims if not provided)
- `dims::Union{Tuple,Nothing}`: Dimensions - auto-detected from filename if not provided
- `dtype::Type`: Data type - auto-detected from file size if not provided
"""
function load_xcat_phantom(
    raw_path::String,
    table_path::String;
    phantom_type::Symbol=:P1,
    voxel_size_cm::Union{Tuple{Float64,Float64,Float64}, Nothing}=nothing,
    dims::Union{Tuple{Int,Int,Int}, Nothing}=nothing,
    dtype::Type=UInt16
)
    mask = load_xcat_raw(raw_path; dims=dims, dtype=dtype)
    result = load_and_classify_phantom(table_path)
    
    # Auto-compute voxel_size if not provided
    actual_voxel_size = if voxel_size_cm !== nothing
        voxel_size_cm
    else
        # Default: assume 40cm FOV for 400x400 (0.1 cm = 1mm voxels)
        nx, ny, nz = size(mask)
        # Assume isotropic FOV of 40cm if unknown
        (0.1, 0.1, 0.1)
    end
    
    return XCATPhantom(
        mask,
        actual_voxel_size,
        phantom_type,
        result.material_mapping,
        result.customizable_ids,
        result.arteries,
        result.veins,
        missing,
        missing
    )
end

"""
    load_xcat_with_perfusion(p1_raw_path, p1_table_path, p2_raw_path, p2_table_path, perfusion_data_path; voxel_size_cm)

Load P1 and P2 XCAT phantoms with perfusion data integration.

# Arguments
- `p1_raw_path::String`: Path to P1 RAW file
- `p1_table_path::String`: Path to P1 voxelize table
- `p2_raw_path::String`: Path to P2 RAW file
- `p2_table_path::String`: Path to P2 voxelize table
- `perfusion_data_path::String`: Path to iodine_mass_data.mat
- `voxel_size_cm::Union{Tuple,Nothing}`: Voxel dimensions - auto-detected if not provided
"""
function load_xcat_with_perfusion(
    p1_raw_path::String,
    p1_table_path::String,
    p2_raw_path::String,
    p2_table_path::String,
    perfusion_data_path::String;
    voxel_size_cm::Union{Tuple{Float64,Float64,Float64}, Nothing}=nothing
)
    # Auto-detect dimensions from filenames
    p1_dims = extract_dims_from_filename(p1_raw_path)
    p2_dims = extract_dims_from_filename(p2_raw_path)
    
    p1_mask = load_xcat_raw(p1_raw_path; dims=p1_dims)
    p2_mask = load_xcat_raw(p2_raw_path; dims=p2_dims)
    
    p1_result = load_and_classify_phantom(p1_table_path)
    p2_result = load_and_classify_phantom(p2_table_path)
    
    if !_MAT_AVAILABLE[]
        error("MAT.jl is required for loading perfusion data. Install with: Pkg.add(\"MAT\")")
    end
    
    perfusion_data = load_perfusion_data(perfusion_data_path)
    iodine_mapping = create_iodine_mapping(perfusion_data, p1_table_path, p2_table_path)
    
    # Auto-compute voxel_size if not provided
    actual_voxel_size = if voxel_size_cm !== nothing
        voxel_size_cm
    else
        (0.05, 0.05, 0.05)  # Default: 0.5mm voxels (5cm / 400)
    end
    
    p1_phantom = XCATPhantom(
        p1_mask,
        actual_voxel_size,
        :P1,
        p1_result.material_mapping,
        p1_result.customizable_ids,
        p1_result.arteries,
        p1_result.veins,
        iodine_mapping,
        perfusion_data
    )
    
    p2_phantom = XCATPhantom(
        p2_mask,
        voxel_size_cm,
        :P2,
        p2_result.material_mapping,
        p2_result.customizable_ids,
        p2_result.arteries,
        p2_result.veins,
        iodine_mapping,
        perfusion_data
    )
    
    return p1_phantom, p2_phantom, perfusion_data, iodine_mapping
end

"""
    apply_iodine_contrast(xcat::XCATPhantom, time_ms::Int; scale_factor=1.0)
"""
function apply_iodine_contrast(xcat::XCATPhantom, time_ms::Int; scale_factor=1.0)
    result = Dict{Int, Float64}()
    
    if xcat.perfusion_data === missing || xcat.iodine_mapping === missing
        @warn "No perfusion data available"
        return result
    end
    
    p = xcat.perfusion_data
    imap = xcat.iodine_mapping
    t_idx = min(time_ms + 1, length(p.timepoints))
    
    if xcat.phantom_type == :P1
        for id in xcat.artery_ids
            if haskey(imap.p1_iodine_map, id)
                tissue_type, seg_idx = imap.p1_iodine_map[id]
                if tissue_type == :artery
                    result[id] = p.mass_arteries[seg_idx, t_idx] * scale_factor
                end
            end
        end
        for id in xcat.vein_ids
            if haskey(imap.p1_iodine_map, id)
                tissue_type, seg_idx = imap.p1_iodine_map[id]
                if tissue_type == :vein
                    result[id] = p.mass_veins[seg_idx, t_idx] * scale_factor
                end
            end
        end
    else
        for id in xcat.artery_ids
            if haskey(imap.p2_iodine_map, id)
                tissue_type, seg_idx = imap.p2_iodine_map[id]
                if tissue_type == :artery && seg_idx <= size(p.mass_arteries, 1)
                    result[id] = p.mass_arteries[seg_idx, t_idx] * scale_factor
                end
            end
        end
        for id in xcat.vein_ids
            if haskey(imap.p2_iodine_map, id)
                tissue_type, seg_idx = imap.p2_iodine_map[id]
                if tissue_type == :vein && seg_idx <= size(p.mass_veins, 1)
                    result[id] = p.mass_veins[seg_idx, t_idx] * scale_factor
                end
            end
        end
    end
    
    return result
end

"""
    export_classification_summary(xcat::XCATPhantom, output_path::String)
"""
function export_classification_summary(xcat::XCATPhantom, output_path::String)
    open(output_path, "w") do io
        println(io, "XCAT Phantom Classification Summary")
        println(io, "=" ^ 50)
        println(io, "Phantom Type: $(xcat.phantom_type)")
        println(io, "Dimensions: $(size(xcat.mask))")
        println(io, "Voxel Size: $(xcat.voxel_size) cm")
        println(io)
        
        println(io, "Material Mapping:")
        println(io, "-" ^ 30)
        for (id, mat) in sort(collect(xcat.material_mapping))
            println(io, "  ID $id => $mat")
        end
        println(io)
        
        println(io, "Customizable Structures: $(length(xcat.customizable_ids))")
        println(io, "  Arteries: $(length(xcat.artery_ids))")
        println(io, "  Veins: $(length(xcat.vein_ids))")
        
        if xcat.perfusion_data !== missing
            println(io)
            println(io, "Perfusion Data Loaded:")
            println(io, "  Arteries: $(length(xcat.perfusion_data.artery_names)) segments")
            println(io, "  Veins: $(length(xcat.perfusion_data.vein_names)) segments")
            println(io, "  GM: $(length(xcat.perfusion_data.gm_names)) segments")
            println(io, "  WM: $(length(xcat.perfusion_data.wm_names)) segments")
            println(io, "  Timepoints: $(length(xcat.perfusion_data.timepoints))")
        end
    end
    
    println("Classification summary saved to: $output_path")
end

"""
    create_basis_phantom_components(xcat::XCATPhantom) -> NamedTuple
"""
function create_basis_phantom_components(xcat::XCATPhantom)
    mask_uint8 = UInt8.(xcat.mask)
    
    int_mapping = Dict{Int, Symbol}()
    for (k, v) in xcat.material_mapping
        int_mapping[Int(k)] = v
    end
    
    return (
        mask=mask_uint8,
        material_mapping=int_mapping,
        voxel_size=xcat.voxel_size
    )
end

# =============================================================================
# Unified Phantom Loading (Backward Compatibility)
# =============================================================================

"""
    load_phantom(filepath::String; dims=nothing, dtype=nothing, voxel_size_cm=nothing) -> Array

Unified phantom loading that handles both PVAT (UInt8) and XCAT (UInt16) formats.

# Arguments
- `filepath::String`: Path to RAW file
- `dims::Union{Tuple,Nothing}`: Dimensions - auto-detected if nothing
- `dtype::Union{Type,Nothing}`: Data type - auto-detected if nothing  
- `voxel_size_cm::Union{Tuple,Nothing}`: Voxel size - computed from dims if nothing

# Auto-detection
- PVAT: 1600×1400×500, UInt8 (detected from file size or use defaults)
- XCAT P1/P2: 400×400×400, UInt16 (from filename pattern)

# Example
```julia
# Load PVAT phantom (existing workflow)
pvat = load_phantom("vmale50.raw"; dims=(1600,1400,500), dtype=UInt8)

# Load XCAT phantom (auto-detected)
xcat = load_phantom("P2_400_400_400.raw")
```
"""
function load_phantom(filepath::String; 
                     dims::Union{Tuple{Int,Int,Int}, Nothing}=nothing,
                     dtype::Union{Type, Nothing}=nothing,
                     voxel_size_cm::Union{Nothing, NTuple{3, Float64}}=nothing)
    # Auto-detect dimensions from filename if not provided
    if dims === nothing
        dims = extract_dims_from_filename(filepath)
    end
    
    # Auto-detect data type if not provided
    if dtype === nothing
        dtype = detect_raw_dtype(filepath)
    end
    
    return load_xcat_raw(filepath; dims=dims, dtype=dtype)
end

# =============================================================================
# Color Mapping from Semantic Categories
# =============================================================================

"""
    get_category_color(category::StructureCategory) -> NTuple{3, Float64}

Get RGB color for a semantic category (normalized 0-1).
"""
function get_category_color(category::StructureCategory)::NTuple{3, Float64}
    return get(CATEGORY_COLORS, category, (0.5, 0.5, 0.5))
end

const CATEGORY_COLORS = Dict{StructureCategory, NTuple{3, Float64}}(
    CAT_AIR_CAVITY => (0.9, 0.95, 1.0),    # Light gray-blue (air)
    CAT_SOFT_TISSUE => (0.8, 0.4, 0.4),    # Red-brown (muscle)
    CAT_BONE => (0.9, 0.9, 0.8),           # Off-white (bone)
    CAT_BRAIN => (0.9, 0.6, 0.6),          # Pinkish (brain)
    CAT_BLOOD_VESSEL => (1.0, 0.0, 0.0),    # Red (blood)
    CAT_CSF => (0.7, 0.9, 1.0),            # Light cyan (CSF)
    CAT_AIRWAY => (1.0, 1.0, 0.8),         # Yellow (airway)
    CAT_EYE => (0.6, 0.8, 1.0),            # Light blue (eye)
    CAT_CARTILAGE => (1.0, 0.7, 0.8),      # Coral/pink (cartilage)
    CAT_UNKNOWN => (0.3, 0.3, 0.3),        # Gray (unknown)
)

"""
    generate_organ_colors(material_mapping::Dict{Int, Symbol}) -> Dict{Int, NTuple{3, Float64}}

Generate organ colors from material mapping (compatible with existing PVAT workflow).

# Example
```julia
# Generate colors for organ IDs
colors = generate_organ_colors(Dict(0 => :air, 1 => :soft_tissue, 8 => :bone))
# Returns: Dict(0 => (0.9, 0.95, 1.0), 1 => (0.8, 0.4, 0.4), 8 => (0.9, 0.9, 0.8))
```
"""
function generate_organ_colors(material_mapping::Dict{Int, Symbol})::Dict{Int, NTuple{3, Float64}}
    colors = Dict{Int, NTuple{3, Float64}}()
    
    material_to_color = Dict{Symbol, NTuple{3, Float64}}(
        :air => (0.9, 0.95, 1.0),
        :soft_tissue => (0.8, 0.4, 0.4),
        :bone => (0.9, 0.9, 0.8),
        :brain => (0.9, 0.6, 0.6),
        :blood => (1.0, 0.0, 0.0),
        :csf => (0.7, 0.9, 1.0),
        :airway => (1.0, 1.0, 0.8),
        :cartilage => (1.0, 0.7, 0.8),
    )
    
    for (id, material) in material_mapping
        colors[id] = get(material_to_color, material, (0.5, 0.5, 0.5))
    end
    
    return colors
end

# =============================================================================
# Organ ID Adapters
# =============================================================================

"""
    PVAT_ORGAN_IDS

Mapping from PVAT organ IDs (0-32) to semantic categories.
From PVAT_main_script_Shu.jl organ table.
"""
const PVAT_ORGAN_IDS = Dict{Int, Symbol}(
    0  => :air,
    1  => :soft_tissue,       # fat / background tissue
    2  => :cartilage,
    3  => :bone,              # sternum
    4  => :soft_tissue,       # muscle / bone marrow
    5  => :bone,              # ribs
    6  => :soft_tissue,       # lung parenchyma
    7  => :air_cavity,        # trachea_bronchi
    8  => :bone,              # cortical bone
    9  => :bone,              # spine
    10 => :soft_tissue,       # liver
    11 => :air_cavity,        # airway
    12 => :air_cavity,        # stomach
    13 => :air_cavity,        # esophagus
    14 => :soft_tissue,       # spleen
    15 => :soft_tissue,       # left ventricle myocardium
    16 => :soft_tissue,       # right ventricle myocardium
    17 => :soft_tissue,       # left atrium myocardium
    18 => :soft_tissue,       # right atrium myocardium
    19 => :blood,             # left ventricle chamber (blood pool)
    20 => :blood,             # right ventricle chamber (blood pool)
    21 => :blood,             # left atria chamber (blood pool)
    22 => :blood,             # right atria chamber (blood pool)
    23 => :soft_tissue,       # skin
    24 => :blood,             # vein
    25 => :blood,             # artery
    26 => :blood,             # coronary artery
    27 => :blood,             # coronary vein
    28 => :blood,             # aorta
    29 => :soft_tissue,       # pericardium
    30 => :soft_tissue,       # coronary artery vessel wall
    31 => :soft_tissue,       # lipid plaque
    32 => :bone,              # calcification plaque
)

"""
    convert_organ_ids(pvat_ids::Dict{Int, Symbol}) -> Dict{Int, Symbol}

Convert PVAT organ IDs to semantic material symbols.

# Example
```julia
# Convert PVAT format
materials = convert_organ_ids(PVAT_ORGAN_IDS)
# Returns: Dict(0 => :air, 1 => :soft_tissue, ...)
```
"""
function convert_organ_ids(pvat_ids::Dict{Int, Symbol})::Dict{Int, Symbol}
    result = Dict{Int, Symbol}()
    for (id, material) in pvat_ids
        result[id] = material
    end
    return result
end

"""
    create_compatible_phantom(
        array_3d::Array{T,3};
        organ_mapping::Union{Dict{Int, Symbol}, Nothing}=nothing,
        voxel_size_cm::Union{NTuple{3, Float64}, Nothing}=nothing
    ) -> XCATPhantom

Create an XCATPhantom from any 3D array with optional organ mapping.

# Arguments
- `array_3d::Array{T,3}`: 3D array of organ IDs
- `organ_mapping::Union{Dict{Int, Symbol}, Nothing}`: ID → material symbol mapping
- `voxel_size_cm::Union{NTuple{3, Float64}, Nothing}`: Voxel dimensions

# Example
```julia
# Create from PVAT data
pvat_array = load_phantom("vmale50.raw"; dims=(1600,1400,500), dtype=UInt8)
phantom = create_compatible_phantom(pvat_array; organ_mapping=PVAT_ORGAN_IDS)
```
"""
function create_compatible_phantom(
    array_3d::Array{T,3};
    organ_mapping::Union{Dict{Int, Symbol}, Nothing}=nothing,
    voxel_size_cm::Union{NTuple{3, Float64}, Nothing}=nothing
)::XCATPhantom where T
    # Use provided mapping or try to infer from unique values
    if organ_mapping === nothing
        organ_mapping = infer_organ_mapping(array_3d)
    end
    
    # Compute voxel size if not provided
    if voxel_size_cm === nothing
        # Try to infer from array size (assuming 40cm FOV)
        nx, ny, nz = size(array_3d)
        voxel_size_cm = (40.0 / nx, 40.0 / ny, 40.0 / nz)
    end
    
    # Create material mapping
    material_mapping = Dict{Int, Symbol}()
    for (id, material) in organ_mapping
        material_mapping[Int(id)] = material
    end
    
    # Determine phantom type based on ID range
    unique_ids = unique(array_3d)
    max_id = maximum(unique_ids)
    phantom_type = max_id > 100 ? :PVAT_detailed : :PVAT_simple
    
    # Find customizable IDs (blood vessels)
    customizable_ids = Set{Int}()
    artery_ids = Set{Int}()
    vein_ids = Set{Int}()
    
    for (id, material) in organ_mapping
        if material == :blood || material == :artery || material == :vein
            push!(customizable_ids, Int(id))
            if material == :vein
                push!(vein_ids, Int(id))
            else
                push!(artery_ids, Int(id))
            end
        end
    end
    
    return XCATPhantom(
        array_3d,
        voxel_size_cm,
        phantom_type,
        material_mapping,
        customizable_ids,
        artery_ids,
        vein_ids,
        missing,
        missing
    )
end

"""
    infer_organ_mapping(array_3d::Array{T,3}) -> Dict{Int, Symbol}

Infer organ mapping from array values (simple heuristic).
"""
function infer_organ_mapping(array_3d::Array{T,3})::Dict{Int, Symbol} where T
    unique_ids = unique(array_3d)
    mapping = Dict{Int, Symbol}()
    
    for id in unique_ids
        if id == 0
            mapping[Int(id)] = :air
        elseif id <= 9
            mapping[Int(id)] = :bone
        elseif id <= 18
            mapping[Int(id)] = :soft_tissue
        elseif id <= 22
            mapping[Int(id)] = :blood
        elseif id <= 28
            mapping[Int(id)] = :blood  # vessels
        else
            mapping[Int(id)] = :soft_tissue
        end
    end
    
    return mapping
end

end # module
