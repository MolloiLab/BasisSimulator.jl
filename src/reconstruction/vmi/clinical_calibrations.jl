"""
Clinical rod-HU calibration constants for image-domain Ding-style
material decomposition.

Each scanner exports a `Dict{String, NamedTuple}` keyed by Gammex 472 rod
name (e.g. `"I 5.0"`, `"Ca 200"`, `"Water (O)"`).  Per entry:
- `material`     : `:water`, `:solid_water`, `:calcium`, or `:iodine`
- `mg_per_mL`    : nominal concentration (0 for water/solid water,
                   calcium concs in mg/cm³ HA-equivalent, iodine in mg/mL)
- one or more measured-HU fields, named after the scan/bin they came from
  (e.g. `HU_80kVp`, `HU_140kVp` for DE; `HU_low_bin`, `HU_high_bin` for PCCT).

Available constants:
- [`GE_REVOLUTION_APEX_ELITE_DE_CAL`](@ref) — DE 80 + 140 kVp sim post-RSKR (iodine + water; calcium rows still clinical DICOM)
- [`SIEMENS_NAEOTOM_ALPHA_140KVP_CAL`](@ref) — PCCT sim post-RSKR (140 kVp)
- [`SIEMENS_NAEOTOM_ALPHA_120KVP_CAL`](@ref) — PCCT sim post-RSKR (120 kVp)

Two helpers select rod subsets and unpack the (HU_low, HU_high)
columns for a Ding LSQ fit:
- [`iodine_calibration_rods`](@ref) — water-O + 7 iodine rods
- [`calcium_calibration_rods`](@ref) — water-O + 5 calcium rods

Each helper takes the calibration dict + the field-symbol pair
`hu_low_field`, `hu_high_field` so the caller can pick which two scans
(or bins, for PCCT) form the (low, high) decomposition basis.

Generated from the in-notebook calibration cells:
- nb06: `verification/notebooks/06_ge_apex_elite_clinical.jl` cell `07040010-…`
- nb07: `verification/notebooks/07_siemens_naeotom_alpha_clinical.jl` cell `08070100-…`
"""

# =============================================================================
# GE Revolution Apex Elite — DE 80 kVp + 140 kVp clinical FBP
# =============================================================================

"""
    GE_REVOLUTION_APEX_ELITE_DE_CAL :: Dict{String, NamedTuple}

Per-rod HU on the simulated 80 kVp + 140 kVp DE FBP (Gammex 472 phantom)
after **RSKR-2ch joint denoising** at the μ-domain stage.  Same flow the
image-domain Ding decomp consumes downstream, so the cal is fit on the
exact spectral conditions the synthesis sees (matches the convention
used by `SIEMENS_NAEOTOM_ALPHA_*` constants).

Calcium rod values come from real clinical GE Apex Elite GSI DICOMs at
~10 mGy CTDI; iodine + water rod values were re-derived from the
simulated post-RSKR FBP HU on a 512²×16 Gammex 472 — the cal then
matches the simulator's polychromatic HU baseline by construction
(slopes ≈ 1 in measured-vs-theoretical regression at every VMI energy).

Fields per rod: `material`, `mg_per_mL`, `HU_80kVp`, `HU_140kVp`.
"""
const GE_REVOLUTION_APEX_ELITE_DE_CAL = Dict{String, NamedTuple}(
    # Water + solid water — sim post-RSKR (phantom-center 30-px ROI)
    "Water (O)" => (material = :water,        mg_per_mL =   0.0, HU_80kVp =    9.5f0, HU_140kVp =   33.7f0),
    "Water (I)" => (material = :water,        mg_per_mL =   0.0, HU_80kVp =    9.5f0, HU_140kVp =   33.7f0),
    "SW ref 1"  => (material = :solid_water,  mg_per_mL =   0.0, HU_80kVp =    9.5f0, HU_140kVp =   33.7f0),
    "SW ref 2"  => (material = :solid_water,  mg_per_mL =   0.0, HU_80kVp =    9.5f0, HU_140kVp =   33.7f0),
    # Calcium rods — clinical FBP DICOM (preserved from prior version)
    "Ca 50"     => (material = :calcium,      mg_per_mL =  50.0, HU_80kVp =  213.7f0, HU_140kVp =  181.6f0),
    "Ca 100"    => (material = :calcium,      mg_per_mL = 100.0, HU_80kVp =  390.4f0, HU_140kVp =  302.4f0),
    "Ca 200"    => (material = :calcium,      mg_per_mL = 200.0, HU_80kVp =  764.6f0, HU_140kVp =  539.4f0),
    "Ca 300"    => (material = :calcium,      mg_per_mL = 300.0, HU_80kVp = 1132.4f0, HU_140kVp =  779.0f0),
    "Ca 400"    => (material = :calcium,      mg_per_mL = 400.0, HU_80kVp = 1459.4f0, HU_140kVp =  995.6f0),
    # Iodine rods — sim post-RSKR (8-px core ROI per rod)
    "I 2.0"     => (material = :iodine,       mg_per_mL =   2.0, HU_80kVp =    4.8f0, HU_140kVp =    4.4f0),
    "I 2.5"     => (material = :iodine,       mg_per_mL =   2.5, HU_80kVp =   25.0f0, HU_140kVp =   13.7f0),
    "I 5.0"     => (material = :iodine,       mg_per_mL =   5.0, HU_80kVp =  107.6f0, HU_140kVp =   56.3f0),
    "I 7.5"     => (material = :iodine,       mg_per_mL =   7.5, HU_80kVp =  180.0f0, HU_140kVp =   94.9f0),
    "I 10.0"    => (material = :iodine,       mg_per_mL =  10.0, HU_80kVp =  264.1f0, HU_140kVp =  135.4f0),
    "I 15.0"    => (material = :iodine,       mg_per_mL =  15.0, HU_80kVp =  426.6f0, HU_140kVp =  212.7f0),
    "I 20.0"    => (material = :iodine,       mg_per_mL =  20.0, HU_80kVp =  587.6f0, HU_140kVp =  300.2f0),
)


# =============================================================================
# Siemens Naeotom Alpha — PCCT sim post-RSKR low/high-bin (Gammex 472)
# =============================================================================
#
# These calibrations come from the **simulated** post-RSKR-2ch low/high-bin
# volumes (NOT clinical scans), since the image-domain Ding decomp consumes
# the same simulated bin-recon basis pair downstream.  Bin grouping is
# fixed at low = bins 1+2, high = bins 3+4.  HU conversion uses the
# bin-effective μ_water from the M-matrix construction at the source kVp.
#
# Fields per rod: `material`, `mg_per_mL`, `HU_low_bin`, `HU_high_bin`.

"""
    SIEMENS_NAEOTOM_ALPHA_140KVP_CAL :: Dict{String, NamedTuple}

Per-rod HU on the simulated 140 kVp / 174 mA / 10.12 mGy CTDI Naeotom Alpha
PCCT scan (Gammex 472 phantom) after bin combine (1+2 / 3+4) → FBP →
RSKR-2ch joint denoising.  Source: `sim_scan2_combined_bin_volumes_ring_s2`.

Fields per rod: `material`, `mg_per_mL`, `HU_low_bin`, `HU_high_bin`.
"""
const SIEMENS_NAEOTOM_ALPHA_140KVP_CAL = Dict{String, NamedTuple}(
    "Water (O)" => (material = :water,        mg_per_mL =   0.0, HU_low_bin =  -46.3f0, HU_high_bin =    0.7f0),
    "SW ref 1"  => (material = :solid_water,  mg_per_mL =   0.0, HU_low_bin =  -46.6f0, HU_high_bin =   -0.6f0),
    "SW ref 2"  => (material = :solid_water,  mg_per_mL =   0.0, HU_low_bin =  -48.0f0, HU_high_bin =   -3.1f0),
    "Ca 50"     => (material = :calcium,      mg_per_mL =  50.0, HU_low_bin =  112.5f0, HU_high_bin =  152.6f0),
    "Ca 100"    => (material = :calcium,      mg_per_mL = 100.0, HU_low_bin =  240.6f0, HU_high_bin =  261.1f0),
    "Ca 200"    => (material = :calcium,      mg_per_mL = 200.0, HU_low_bin =  515.4f0, HU_high_bin =  493.8f0),
    "Ca 300"    => (material = :calcium,      mg_per_mL = 300.0, HU_low_bin =  764.7f0, HU_high_bin =  709.5f0),
    "Ca 400"    => (material = :calcium,      mg_per_mL = 400.0, HU_low_bin = 1011.6f0, HU_high_bin =  928.8f0),
    "Water (I)" => (material = :water,        mg_per_mL =   0.0, HU_low_bin =  -46.6f0, HU_high_bin =    2.3f0),
    "I 2.0"     => (material = :iodine,       mg_per_mL =   2.0, HU_low_bin =    0.6f0, HU_high_bin =   39.6f0),
    "I 2.5"     => (material = :iodine,       mg_per_mL =   2.5, HU_low_bin =   15.8f0, HU_high_bin =   45.6f0),
    "I 5.0"     => (material = :iodine,       mg_per_mL =   5.0, HU_low_bin =   67.9f0, HU_high_bin =   89.1f0),
    "I 7.5"     => (material = :iodine,       mg_per_mL =   7.5, HU_low_bin =  124.6f0, HU_high_bin =  130.9f0),
    "I 10.0"    => (material = :iodine,       mg_per_mL =  10.0, HU_low_bin =  174.5f0, HU_high_bin =  162.4f0),
    "I 15.0"    => (material = :iodine,       mg_per_mL =  15.0, HU_low_bin =  290.4f0, HU_high_bin =  247.8f0),
    "I 20.0"    => (material = :iodine,       mg_per_mL =  20.0, HU_low_bin =  412.1f0, HU_high_bin =  338.1f0),
)

"""
    SIEMENS_NAEOTOM_ALPHA_120KVP_CAL :: Dict{String, NamedTuple}

Per-rod HU on the simulated 120 kVp / 253 mA / 10.15 mGy CTDI Naeotom Alpha
PCCT scan (Gammex 472 phantom) after bin combine (1+2 / 3+4) → FBP →
RSKR-2ch joint denoising.  Source: inline build from `sim_scan4.bins`.

Fields per rod: `material`, `mg_per_mL`, `HU_low_bin`, `HU_high_bin`.
"""
const SIEMENS_NAEOTOM_ALPHA_120KVP_CAL = Dict{String, NamedTuple}(
    "Water (O)" => (material = :water,        mg_per_mL =   0.0, HU_low_bin =  -45.1f0, HU_high_bin =    9.1f0),
    "SW ref 1"  => (material = :solid_water,  mg_per_mL =   0.0, HU_low_bin =  -47.9f0, HU_high_bin =    7.8f0),
    "SW ref 2"  => (material = :solid_water,  mg_per_mL =   0.0, HU_low_bin =  -46.2f0, HU_high_bin =    8.5f0),
    "Ca 50"     => (material = :calcium,      mg_per_mL =  50.0, HU_low_bin =  118.0f0, HU_high_bin =  167.4f0),
    "Ca 100"    => (material = :calcium,      mg_per_mL = 100.0, HU_low_bin =  253.1f0, HU_high_bin =  282.2f0),
    "Ca 200"    => (material = :calcium,      mg_per_mL = 200.0, HU_low_bin =  536.9f0, HU_high_bin =  528.4f0),
    "Ca 300"    => (material = :calcium,      mg_per_mL = 300.0, HU_low_bin =  799.3f0, HU_high_bin =  765.9f0),
    "Ca 400"    => (material = :calcium,      mg_per_mL = 400.0, HU_low_bin = 1063.2f0, HU_high_bin =  996.2f0),
    "Water (I)" => (material = :water,        mg_per_mL =   0.0, HU_low_bin =  -46.4f0, HU_high_bin =   15.2f0),
    "I 2.0"     => (material = :iodine,       mg_per_mL =   2.0, HU_low_bin =    5.2f0, HU_high_bin =   51.1f0),
    "I 2.5"     => (material = :iodine,       mg_per_mL =   2.5, HU_low_bin =   16.5f0, HU_high_bin =   61.2f0),
    "I 5.0"     => (material = :iodine,       mg_per_mL =   5.0, HU_low_bin =   77.5f0, HU_high_bin =  105.6f0),
    "I 7.5"     => (material = :iodine,       mg_per_mL =   7.5, HU_low_bin =  139.7f0, HU_high_bin =  154.5f0),
    "I 10.0"    => (material = :iodine,       mg_per_mL =  10.0, HU_low_bin =  191.7f0, HU_high_bin =  193.5f0),
    "I 15.0"    => (material = :iodine,       mg_per_mL =  15.0, HU_low_bin =  315.0f0, HU_high_bin =  284.2f0),
    "I 20.0"    => (material = :iodine,       mg_per_mL =  20.0, HU_low_bin =  450.1f0, HU_high_bin =  391.7f0),
)


# =============================================================================
# Helpers — subset selection + (HU_low, HU_high) unpack for Ding LSQ
# =============================================================================

"""
    iodine_calibration_rods(cal;
                            hu_low_field::Symbol,
                            hu_high_field::Symbol,
                            include_water::Bool = true)
        -> NamedTuple{(:names, :HU_low, :HU_high, :mg_per_mL)}

Return the iodine-calibration rod subset (water-O anchor + 7 iodine rods
by default) for image-domain Ding fits.  The caller specifies which two
fields of each rod's NamedTuple form the (low, high) pair via
`hu_low_field` / `hu_high_field`.

# Example — DE GE Revolution Apex Elite
```julia
rods = iodine_calibration_rods(GE_REVOLUTION_APEX_ELITE_DE_CAL;
    hu_low_field = :HU_80kVp, hu_high_field = :HU_140kVp)
A = hcat(ones(length(rods.HU_low)), rods.HU_low, rods.HU_high)
coeffs_iod = A \\ rods.mg_per_mL
```

# Example — PCCT Siemens Naeotom Alpha
```julia
rods = iodine_calibration_rods(SIEMENS_NAEOTOM_ALPHA_140KVP_CAL;
    hu_low_field = :HU_low_bin, hu_high_field = :HU_high_bin)
```

# Returns
NamedTuple of 1:1-aligned vectors:
- `names::Vector{String}`
- `HU_low::Vector{Float64}`
- `HU_high::Vector{Float64}`
- `mg_per_mL::Vector{Float64}`
"""
function iodine_calibration_rods(
        cal::Dict{String, <:NamedTuple};
        hu_low_field::Symbol,
        hu_high_field::Symbol,
        include_water::Bool = true,
    )
    iod_names = [
        "I 2.0", "I 2.5", "I 5.0", "I 7.5", "I 10.0", "I 15.0", "I 20.0",
    ]
    names = include_water ? vcat(["Water (O)"], iod_names) : iod_names
    _subset_rods(cal, names, hu_low_field, hu_high_field)
end

"""
    calcium_calibration_rods(cal;
                             hu_low_field::Symbol,
                             hu_high_field::Symbol,
                             include_water::Bool = true)
        -> NamedTuple{(:names, :HU_low, :HU_high, :mg_per_mL)}

Return the calcium-calibration rod subset (water-O anchor + 5 calcium rods
by default).  Same layout as `iodine_calibration_rods`; for fitting the
calcium image-domain decomp coefficients.
"""
function calcium_calibration_rods(
        cal::Dict{String, <:NamedTuple};
        hu_low_field::Symbol,
        hu_high_field::Symbol,
        include_water::Bool = true,
    )
    ca_names = ["Ca 50", "Ca 100", "Ca 200", "Ca 300", "Ca 400"]
    names = include_water ? vcat(["Water (O)"], ca_names) : ca_names
    _subset_rods(cal, names, hu_low_field, hu_high_field)
end

function _subset_rods(
        cal::Dict{String, <:NamedTuple},
        names::Vector{String},
        hu_low_field::Symbol,
        hu_high_field::Symbol,
    )
    HU_low    = Float64[Float64(getfield(cal[n], hu_low_field))  for n in names]
    HU_high   = Float64[Float64(getfield(cal[n], hu_high_field)) for n in names]
    mg_per_mL = Float64[Float64(cal[n].mg_per_mL) for n in names]
    (names = names, HU_low = HU_low, HU_high = HU_high, mg_per_mL = mg_per_mL)
end

export GE_REVOLUTION_APEX_ELITE_DE_CAL,
       SIEMENS_NAEOTOM_ALPHA_140KVP_CAL,
       SIEMENS_NAEOTOM_ALPHA_120KVP_CAL,
       iodine_calibration_rods,
       calcium_calibration_rods


# =============================================================================
# Image-domain DE VMI optimized hyperparameters — GE Revolution Apex Elite GSI
# =============================================================================
#
# Each const encodes the four parameters of the iodine-basis Ding image-
# domain decomposition + VMI synthesis:
#
#     c_iodine[v]  = a₀ + a₁·HU_low[v] + a₂·HU_high[v]                  (mg/mL)
#     HU_E[v]      = HU_low[v] + c_iodine[v] · (μρ_I(E)/μρ_water(E) − α_iod_low_cal)
#
# Values were derived in `docs/notebooks/03_dual_kvp_vmi.jl` §11 by BFGS
# minimization (Optim.jl, analytical gradient) of per-rod relative-error
# nRMSE across all 4 VMI energies (40 / 70 / 100 / 140 keV) and all 14
# Gammex 472 rods (water + 7 I + 7 Ca), with material/energy weights all
# 1.0 and `nrmse_floor = 100 HU`.  Corresponding nb04 PCCT cal in
# `src/reconstruction/vmi/pcct_calibration.jl`.
#
# Pipeline scope per const:
#   - Scanner    : GE Revolution Apex Elite (large bowtie / Lumex detector)
#   - kVp pair   : rapid-switching GSI (low / high listed below)
#   - Body size  : ~33 cm water-path hardening (Gammex 472 body cylinder)
#   - Recon      : sino-BHC (per-column, bowtie-aware) → FBP → RSKR-2ch
#                  (n_iter=2, h_param=2, radius=3, γ=0.5) → radial cupping
#                  → measured center-ROI μ_water → to_hounsfield + 28 HU
#                  noise floor → Ding decomp + 7-slice z-median
#   - Synthesis  : iodine basis (XA.Elements.Iodine vs XA.Materials.water)
#
# For other scanners / kVp pairs / body sizes, re-derive in nb03 §11.

"""
    GE_REVOLUTION_APEX_ELITE_80_140KVP_DE_VMI_CAL :: NamedTuple

Image-domain DE VMI calibration for GE Revolution Apex Elite GSI at the
80 / 140 kVp rapid-switching pair.

Source: nb03 protocol matching the clinical Apex Elite GSI acquisition
(80 kVp / 264.55 mA-eff, 140 kVp / 141.75 mA-eff, 984 views, 0.5 s
rotation, 5 mm collimation, 2.5 mm Al flat filter + 4.5 mm Al additional).
"""
const GE_REVOLUTION_APEX_ELITE_80_140KVP_DE_VMI_CAL = (
    coeffs        = Float32[-0.015, 0.04588, -0.05301],
    α_iod_low_cal = 45.09f0,
)

"""
    de_vmi_cal_for(scanner::Symbol, low_kVp::Real, high_kVp::Real)
        -> NamedTuple

Lookup helper for image-domain DE VMI calibrations.  Throws if no cal is
registered for the requested `(scanner, low_kVp, high_kVp)` combo —
re-derive in `docs/notebooks/03_dual_kvp_vmi.jl` §11 with
`optim_knob.enabled = true`.
"""
function de_vmi_cal_for(scanner::Symbol, low_kVp::Real, high_kVp::Real)
    klo = round(Int, low_kVp)
    khi = round(Int, high_kVp)
    if scanner == :ge_revolution_apex_elite && klo == 80 && khi == 140
        return GE_REVOLUTION_APEX_ELITE_80_140KVP_DE_VMI_CAL
    else
        error("de_vmi_cal_for: no calibration available for scanner=$(scanner), " *
              "($(klo), $(khi)) kVp pair.  Re-derive in nb03 §11 (see docstring above).")
    end
end

export GE_REVOLUTION_APEX_ELITE_80_140KVP_DE_VMI_CAL, de_vmi_cal_for
