"""
    Scanners/Siemens.jl

Siemens Healthineers CT scanner configurations.

# Supported Scanners
- Siemens NAEOTOM Alpha (FDA 510(k): K201501, K220814, K243523)

# References
- FDA 510(k) K201501: https://www.accessdata.fda.gov/cdrh_docs/pdf20/K201501.pdf
- FDA 510(k) K220814: https://www.accessdata.fda.gov/cdrh_docs/pdf22/K220814.pdf
- FDA 510(k) K243523: https://www.accessdata.fda.gov/cdrh_docs/pdf24/K243523.pdf
- PMC10321251: https://pmc.ncbi.nlm.nih.gov/articles/PMC10321251/
- PMC9125732: https://pmc.ncbi.nlm.nih.gov/articles/PMC9125732/
"""

# =============================================================================
# Siemens NAEOTOM Alpha - Photon Counting CT
# =============================================================================

# URL constants for source citations
const NAEOTOM_FDA_K201501 = "https://www.accessdata.fda.gov/cdrh_docs/pdf20/K201501.pdf"
const NAEOTOM_FDA_K220814 = "https://www.accessdata.fda.gov/cdrh_docs/pdf22/K220814.pdf"
const NAEOTOM_FDA_K243523 = "https://www.accessdata.fda.gov/cdrh_docs/pdf24/K243523.pdf"
const NAEOTOM_PMC10321251 = "https://pmc.ncbi.nlm.nih.gov/articles/PMC10321251/"
const NAEOTOM_PMC9125732 = "https://pmc.ncbi.nlm.nih.gov/articles/PMC9125732/"
const NAEOTOM_RADIOLOGY_2022 = "https://pubs.rsna.org/doi/full/10.1148/radiol.212579"
const NAEOTOM_BROCHURE = "https://cdn0.scrvt.com/39b415fb07de4d9656c7b516d8e2d907/d7552511faabc710/99a5c3244e83/CT_NAEOTOM-Alpha_Brochure_USA_2021.pdf"

"""
    NAEOTOMMode

Enumeration of NAEOTOM Alpha acquisition modes.
"""
@enum NAEOTOMMode begin
    NAEOTOM_STANDARD     # Standard mode: 144 rows × 0.4 mm = 57.6 mm z-coverage
    NAEOTOM_UHR          # Ultra-High Resolution: 120 rows × 0.2 mm = 24 mm z-coverage
    NAEOTOM_QUANTUM      # QuantumPlus spectral mode (same geometry as standard)
end

"""
    SiemensNAEOTOMAlpha <: AbstractScannerSpec

Siemens NAEOTOM Alpha photon-counting CT scanner specification.

This is the world's first FDA-cleared clinical photon-counting CT system,
featuring a cadmium telluride (CdTe) detector with energy-resolved imaging.

# FDA 510(k)
- Primary: K201501 (September 2021)
- Updates: K220814 (Alpha.Pro/Peak), K243523 (Alpha.Prime)

# Key Features
- CdTe photon-counting detector (first-ever clinical PCD-CT)
- 4 energy thresholds: 20, 35, 55, 70 keV
- Dual-source configuration (Peak/Pro models)
- Ultra-high resolution (UHR) mode: 0.2 mm slices
- Native spectral imaging (no dual-kVp required)

# Modes
- **Standard (Quantum)**: 144 rows × 0.4 mm = 57.6 mm z-coverage
- **UHR**: 120 rows × 0.2 mm = 24 mm z-coverage, 0.11 mm in-plane resolution
- **QuantumPlus**: Spectral mode with 4 energy bins

# Usage
```julia
spec = SiemensNAEOTOMAlpha()           # Standard mode
spec_uhr = SiemensNAEOTOMAlpha(:uhr)   # UHR mode

print_scanner_info(spec)

# Create geometry
geom = create_geometry(spec; n_angles=984, n_rows=64)

# Create with protocol
protocol = NAEOTOMHeadAxial()
geom = create_geometry(spec, protocol)
```

# References
All parameters sourced from FDA 510(k) summaries, DukeSim validation papers,
and peer-reviewed publications. See field citations for specific sources.
"""
struct SiemensNAEOTOMAlpha <: AbstractScannerSpec
    detector_spec::DetectorSpecification
    tube_spec::TubeSpecification
    geometry_spec::GeometrySpecification
    acquisition_spec::AcquisitionSpecification
    mode::NAEOTOMMode
    energy_thresholds_keV::Vector{Float64}
end

"""
    SiemensNAEOTOMAlpha(mode::Symbol=:standard)

Construct Siemens NAEOTOM Alpha scanner specification.

# Arguments
- `mode::Symbol`: Operating mode - `:standard`, `:uhr`, or `:quantum_plus`

# Returns
`SiemensNAEOTOMAlpha` scanner specification configured for the selected mode.
"""
function SiemensNAEOTOMAlpha(mode::Symbol=:standard)
    # Map symbol to enum
    _mode = if mode == :standard
        NAEOTOM_STANDARD
    elseif mode == :uhr
        NAEOTOM_UHR
    elseif mode == :quantum_plus || mode == :spectral
        NAEOTOM_QUANTUM
    else
        error("Unknown NAEOTOM mode: $mode. Use :standard, :uhr, or :quantum_plus")
    end

    return SiemensNAEOTOMAlpha(_mode)
end

"""
    SiemensNAEOTOMAlpha(mode::NAEOTOMMode)

Construct Siemens NAEOTOM Alpha scanner specification with specified mode.
"""
function SiemensNAEOTOMAlpha(mode::NAEOTOMMode)
    # =========================================================================
    # DETECTOR SPECIFICATION
    # =========================================================================
    # CdTe photon-counting detector
    # Source: FDA K201501, PMC10321251, DukeSim validation papers

    # Mode-dependent parameters (derived from Konrad 2025 native dexel sizes)
    # Native dexel: 0.275 mm col × 0.322 mm row at detector face
    # Magnification: 1085.5/595.0 = 1.824
    if mode == NAEOTOM_UHR
        n_rows = 120
        row_size_mm = 0.176   # native_row / magnification (unbinned)
        z_coverage_mm = n_rows * row_size_mm  # ~21.2 mm
        col_size_mm = 0.151   # native_col / magnification (unbinned)
    else
        # Standard and QuantumPlus modes (2×2 binned)
        n_rows = 144
        row_size_mm = 0.353   # native_row × 2 / magnification
        z_coverage_mm = n_rows * row_size_mm  # ~50.8 mm
        col_size_mm = 0.302   # native_col × 2 / magnification
    end

    # Detector columns: computed to cover 50 cm FOV
    # Geometry: SID = 600 mm, SDD = 1072 mm, magnification = 1072/600 ≈ 1.787
    # At detector: FOV × magnification / col_size
    # For standard: 500 × 1.787 / 0.302 ≈ 2960 columns
    # For UHR: 500 × 1.787 / 0.151 ≈ 5920 columns
    n_cols = mode == NAEOTOM_UHR ? 5920 : 2960

    detector_spec = DetectorSpecification(
        # Material: CdTe (cadmium telluride) photon-counting
        SourceCitation(CDTE;
            source=:fda_510k,
            url=NAEOTOM_FDA_K201501,
            note="CdTe direct-conversion photon-counting detector"),

        # Number of rows (mode-dependent)
        SourceCitation(n_rows;
            source=:fda_510k,
            url=NAEOTOM_FDA_K201501,
            note=mode == NAEOTOM_UHR ? "UHR: 120 rows" : "Standard: 144 rows"),

        # Number of columns (estimated)
        SourceCitation(n_cols;
            source=:derived,
            url=NAEOTOM_PMC9125732,
            note="Estimated from 50 cm FOV and pixel size"),

        # Row size (mode-dependent)
        SourceCitation(row_size_mm;
            source=:fda_510k,
            url=NAEOTOM_FDA_K201501,
            note=mode == NAEOTOM_UHR ? "UHR: 0.2 mm" : "Standard: 0.4 mm"),

        # Column size (mode-dependent)
        SourceCitation(col_size_mm;
            source=:publication,
            url=NAEOTOM_PMC10321251,
            note=mode == NAEOTOM_UHR ? "UHR unbinned: 0.151 mm" : "Standard 2×2 binned: 0.302 mm"),

        # Detector depth: 1.6 mm CdTe
        SourceCitation(1.6;
            source=:publication,
            url=NAEOTOM_PMC10321251,
            note="CdTe sensor thickness"),

        # Photon-counting detector type
        SourceCitation(PHOTON_COUNTING;
            source=:fda_510k,
            url=NAEOTOM_FDA_K201501,
            note="First clinical photon-counting detector CT"),

        # Z-coverage (mode-dependent)
        SourceCitation(z_coverage_mm;
            source=:fda_510k,
            url=NAEOTOM_FDA_K201501,
            note=mode == NAEOTOM_UHR ? "UHR: 24 mm" : "Standard: 57.6 mm"),

        # Fill factors: ~0.95 for direct conversion detectors
        SourceCitation(0.95;
            source=:estimate,
            url="",
            note="Direct conversion detectors have high fill factor"),

        SourceCitation(0.95;
            source=:estimate,
            url="",
            note="Direct conversion detectors have high fill factor")
    )

    # =========================================================================
    # TUBE SPECIFICATION
    # =========================================================================
    # Vectron X-ray tube
    # Source: Siemens technical documentation

    tube_spec = TubeSpecification(
        # Model name
        SourceCitation("Vectron";
            source=:manufacturer,
            url=NAEOTOM_BROCHURE,
            note="Siemens Vectron X-ray tube"),

        # Max power: ~120 kW (estimated)
        SourceCitation(120.0;
            source=:estimate,
            url=NAEOTOM_BROCHURE,
            note="High-power tube for PCCT, exact value proprietary"),

        # Target angle: 7 degrees (typical)
        SourceCitation(7.0;
            source=:estimate,
            url="",
            note="Typical tungsten target angle"),

        # Small focal spot: 0.4 × 0.5 mm (micro focal spot for UHR)
        SourceCitation((0.4, 0.5);
            source=:fda_510k,
            url=NAEOTOM_FDA_K201501,
            note="Micro focal spot for UHR imaging"),

        # Large focal spot: 1.0 × 1.0 mm (estimated)
        SourceCitation((1.0, 1.0);
            source=:estimate,
            url="",
            note="Standard focal spot, exact value proprietary"),

        # kVp options: 70, 90, 120, 140 (NOT 80/100 like traditional CT)
        SourceCitation([70, 90, 120, 140];
            source=:publication,
            url=NAEOTOM_PMC10321251,
            note="NAEOTOM uses 70/90/120/140 kVp, not 80/100"),

        # Max mA: 1300 mA
        SourceCitation(1300;
            source=:manufacturer,
            url=NAEOTOM_BROCHURE,
            note="Generator power reserve up to 1300 mA"),

        # Flying focal spot: yes
        SourceCitation(true;
            source=:manufacturer,
            url=NAEOTOM_BROCHURE,
            note="Flying focal spot supported")
    )

    # =========================================================================
    # GEOMETRY SPECIFICATION
    # =========================================================================
    # Source: DukeSim validation papers (PMC9125732)
    # Note: Exact SID/SDD values not in public FDA docs, estimated from simulations

    geometry_spec = GeometrySpecification(
        # SID: ~600 mm (from DukeSim)
        SourceCitation(600.0;
            source=:simulation,
            url=NAEOTOM_PMC9125732,
            note="Source-to-isocenter distance from DukeSim geometry match"),

        # SDD: ~1072 mm (from DukeSim)
        SourceCitation(1072.0;
            source=:simulation,
            url=NAEOTOM_PMC9125732,
            note="Source-to-detector distance from DukeSim geometry match"),

        # Gantry aperture: 820 mm (82 cm bore)
        SourceCitation(820.0;
            source=:manufacturer,
            url=NAEOTOM_BROCHURE,
            note="Largest bore in class: 82 cm"),

        # Max SFOV: 500 mm (primary) or 360 mm (secondary for cardiac)
        SourceCitation(500.0;
            source=:publication,
            url=NAEOTOM_RADIOLOGY_2022,
            note="Primary FOV: 50 cm, Secondary: 36 cm"),

        # Detector curve radius: equal to SDD for third-gen geometry
        SourceCitation(1072.0;
            source=:derived,
            url="",
            note="Curved detector array, radius = SDD")
    )

    # =========================================================================
    # ACQUISITION SPECIFICATION
    # =========================================================================
    # Source: FDA filings and technical specs

    acquisition_spec = AcquisitionSpecification(
        # Min rotation time: 0.25 seconds
        SourceCitation(0.25;
            source=:fda_510k,
            url=NAEOTOM_FDA_K201501,
            note="Fastest rotation time"),

        # Max rotation time: 1.0 seconds
        SourceCitation(1.0;
            source=:fda_510k,
            url=NAEOTOM_FDA_K201501,
            note="Slowest rotation time"),

        # Rotation time options
        SourceCitation([0.25, 0.33, 0.5, 0.75, 1.0];
            source=:fda_510k,
            url=NAEOTOM_FDA_K201501,
            note="Available rotation times"),

        # Max views per rotation: estimated ~2300
        SourceCitation(2304;
            source=:estimate,
            url="",
            note="Estimated from temporal resolution")
    )

    # Energy thresholds: 20, 35, 55, 70 keV (fixed in clinical mode)
    energy_thresholds = [20.0, 35.0, 55.0, 70.0]

    return SiemensNAEOTOMAlpha(
        detector_spec, tube_spec, geometry_spec, acquisition_spec,
        mode, energy_thresholds
    )
end

# =============================================================================
# Implement AbstractScannerSpec interface
# =============================================================================

manufacturer(::SiemensNAEOTOMAlpha) = SIEMENS_HEALTHINEERS
model_name(spec::SiemensNAEOTOMAlpha) = "NAEOTOM Alpha ($(spec.mode))"
fda_510k(::SiemensNAEOTOMAlpha) = "K201501"
detector(spec::SiemensNAEOTOMAlpha) = spec.detector_spec
tube(spec::SiemensNAEOTOMAlpha) = spec.tube_spec
geometry(spec::SiemensNAEOTOMAlpha) = spec.geometry_spec
acquisition(spec::SiemensNAEOTOMAlpha) = spec.acquisition_spec

# =============================================================================
# NAEOTOM-specific accessors
# =============================================================================

"""
    get_energy_thresholds(spec::SiemensNAEOTOMAlpha) -> Vector{Float64}

Get the energy thresholds in keV for the NAEOTOM Alpha.
"""
get_energy_thresholds(spec::SiemensNAEOTOMAlpha) = spec.energy_thresholds_keV

"""
    get_mode(spec::SiemensNAEOTOMAlpha) -> NAEOTOMMode

Get the operating mode of the NAEOTOM Alpha scanner.
"""
get_mode(spec::SiemensNAEOTOMAlpha) = spec.mode

"""
    is_photon_counting(spec::SiemensNAEOTOMAlpha) -> Bool

Check if the scanner is photon-counting (always true for NAEOTOM Alpha).
"""
is_photon_counting(::SiemensNAEOTOMAlpha) = true

"""
    is_uhr_mode(spec::SiemensNAEOTOMAlpha) -> Bool

Check if the scanner is in UHR (ultra-high resolution) mode.
"""
is_uhr_mode(spec::SiemensNAEOTOMAlpha) = spec.mode == NAEOTOM_UHR

"""
    is_spectral_mode(spec::SiemensNAEOTOMAlpha) -> Bool

Check if the scanner is in spectral (QuantumPlus) mode.
"""
is_spectral_mode(spec::SiemensNAEOTOMAlpha) = spec.mode == NAEOTOM_QUANTUM

"""
    get_pcct_detector(spec::SiemensNAEOTOMAlpha) -> PhotonCountingDetector

Get a PhotonCountingDetector configured for NAEOTOM Alpha.

This creates a detector specification compatible with the PCCT forward projection
pipeline, with correct energy thresholds and detector physics parameters.
"""
function get_pcct_detector(spec::SiemensNAEOTOMAlpha)
    scanner = create_naeotom_alpha(mode = spec.mode == NAEOTOM_UHR ? :uhr : :standard)
    return _build_pcct_detector(scanner)
end

# =============================================================================
# Siemens NAEOTOM Alpha - Protocol Presets
# =============================================================================

"""
    NAEOTOMHeadAxial(; dose_level=:standard)

Standard head axial protocol for Siemens NAEOTOM Alpha.

# Protocol Details
- 120 kVp
- 1.0 s rotation
- Full 360° coverage
- 0.4 mm slice thickness
"""
function NAEOTOMHeadAxial(; dose_level::Symbol=:standard)
    ma = dose_level == :low ? 100 :
         dose_level == :standard ? 200 :
         dose_level == :high ? 350 : 200

    return AxialProtocol(
        120,        # kVp
        ma,         # mA
        1.0,        # rotation time
        984,        # angles
        0.4         # slice thickness
    )
end

# =============================================================================
# Convenience Alias
# =============================================================================

"""
    NAEOTOMAlpha(mode::Symbol=:standard) -> SiemensNAEOTOMAlpha

Convenience constructor for Siemens NAEOTOM Alpha scanner preset.

# Arguments
- `mode::Symbol`: Operating mode - `:standard`, `:uhr`, or `:quantum_plus`

# Example
```julia
using BasisSimulator

# Standard mode
spec = NAEOTOMAlpha()

# UHR mode for maximum resolution
spec_uhr = NAEOTOMAlpha(:uhr)

# Spectral mode for VMI/material decomposition
spec_spectral = NAEOTOMAlpha(:quantum_plus)

# Create geometry for simulation
geom = create_geometry(spec; n_angles=984, n_rows=64)

# Get PCCT detector for energy-resolved simulation
pcct_detector = get_pcct_detector(spec)
```

See also: [`SiemensNAEOTOMAlpha`](@ref), [`get_pcct_detector`](@ref)
"""
function NAEOTOMAlpha(mode::Symbol=:standard)
    return SiemensNAEOTOMAlpha(mode)
end

# =============================================================================
# Extended print_scanner_info for PCCT-specific details
# =============================================================================

"""
    print_naeotom_info(spec::SiemensNAEOTOMAlpha)

Print detailed NAEOTOM Alpha specification including PCCT-specific parameters.
"""
function print_naeotom_info(spec::SiemensNAEOTOMAlpha)
    # Call base print function
    print_scanner_info(spec)

    # Add PCCT-specific info
    println()
    println("PHOTON-COUNTING SPECIFIC")
    println("-" ^ 40)
    println("  Mode:              $(spec.mode)")
    println("  Energy Thresholds: $(spec.energy_thresholds_keV) keV")
    println("  Number of Bins:    $(length(spec.energy_thresholds_keV))")
    println("  Detector Material: CdTe (1.6 mm thick)")

    if spec.mode == NAEOTOM_UHR
        println("  Resolution:        0.11 mm in-plane (UHR)")
        println("  Focal Spot:        Micro (0.4 × 0.5 mm)")
    else
        println("  Resolution:        0.24 mm in-plane (Standard)")
        println("  Focal Spot:        Standard or Micro")
    end

    if spec.mode == NAEOTOM_QUANTUM
        println("  Spectral:          4-bin energy-resolved imaging")
        println("  VMI Range:         40-190 keV")
    end

    println("=" ^ 80)
end

# =============================================================================
# Exports
# =============================================================================

export NAEOTOMMode, NAEOTOM_STANDARD, NAEOTOM_UHR, NAEOTOM_QUANTUM
export SiemensNAEOTOMAlpha, NAEOTOMAlpha
export get_energy_thresholds, get_mode, is_photon_counting, is_uhr_mode, is_spectral_mode
export get_pcct_detector, print_naeotom_info
export NAEOTOMHeadAxial
