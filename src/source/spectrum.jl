"""
    Physics/Spectrum.jl

Load pre-computed X-ray spectra from XCAT/XCIST .dat files.

No spectrum generation - just load validated spectra from files.
"""

using DelimitedFiles

# Path to spectrum data files
const SPECTRUM_DIR = joinpath(@__DIR__, "..", "spectrum")

"""
    load_spectrum(kVp::Int; target_angle::Float64=7.0, source::Symbol=:xspect)

Load a pre-computed X-ray spectrum from .dat file.

# Arguments
- `kVp::Int`: Peak voltage (70, 80, 90, 100, 110, 120, 130, 140)
- `target_angle::Float64`: Anode target angle (7.0 or 10.0 degrees)
- `source::Symbol`: Spectrum source (:xspect or :xcist)

# Returns
- `energies::Vector{Float64}`: Energy bin centers (keV)
- `weights::Vector{Float64}`: Photon fluence per energy bin

# File format
- Line 1: Number of energy bins
- Lines 2+: energy_keV, photon_fluence (CSV)

# Units
- xspect: photons/mA/cm²/s at 1m
- xcist: photons/mA/mm²/s at 1m

# Example
```julia
energies, weights = load_spectrum(120)
energies, weights = load_spectrum(120; target_angle=10.0)
```
"""
function load_spectrum(kVp::Int; spectrum_dir::AbstractString=SPECTRUM_DIR, target_angle::Float64=7.0, source::Symbol=:xspect)
    # Validate inputs
    valid_kvp_xspect = [70, 80, 90, 100, 110, 120, 130, 140]
    valid_kvp_xcist = [80, 100, 120, 140]
    valid_angles = [7.0, 10.0]

    if source == :xspect
        kVp in valid_kvp_xspect || error("kVp must be one of $valid_kvp_xspect for xspect source")
        target_angle in valid_angles || error("target_angle must be one of $valid_angles")
        filename = "tungsten_tar$(target_angle)_$(kVp)_filt.dat"
    elseif source == :xcist
        kVp in valid_kvp_xcist || error("kVp must be one of $valid_kvp_xcist for xcist source")
        filename = "xcist_kVp$(kVp)_tar7_bin1.dat"
    else
        error("source must be :xspect or :xcist")
    end

    filepath = joinpath(spectrum_dir, filename)
    isfile(filepath) || error("Spectrum file not found: $filepath")

    # Read file, filtering out empty lines
    all_lines = readlines(filepath)
    lines = filter(l -> !isempty(strip(l)), all_lines)

    n_bins = parse(Int, lines[1])

    energies = Vector{Float64}(undef, n_bins)
    weights = Vector{Float64}(undef, n_bins)

    for i in 1:n_bins
        line = lines[i+1]
        # Handle lines with comma (CSV) or just single value at end of file
        if occursin(',', line)
            parts = split(line, ',')
            energies[i] = parse(Float64, parts[1])
            weights[i] = parse(Float64, parts[2])
        else
            # Single value line (should not happen for valid data lines)
            break
        end
    end

    return energies, weights
end


"""
    spectrum_mean_energy(energies, weights)

Compute fluence-weighted mean energy of spectrum.

# Returns
Mean energy in keV.
"""
function spectrum_mean_energy(energies::Vector{Float64}, weights::Vector{Float64})
    total_weight = sum(weights)
    total_weight > 0 || error("Spectrum has zero total fluence")
    return sum(energies .* weights) / total_weight
end

"""
    downsample_spectrum(energies, weights, n_bins::Int)

Downsample spectrum to fewer energy bins by averaging.

This is important for memory efficiency in polychromatic projection.
240 bins → 24 bins reduces memory by 10x with minimal accuracy loss.

# Arguments
- `energies`: Original energy values (keV)
- `weights`: Original photon fluence weights
- `n_bins`: Target number of bins (default 24)

# Returns
- `new_energies`: Downsampled energy bin centers
- `new_weights`: Downsampled weights (summed within each bin)

# Example
```julia
energies, weights = load_spectrum(120)  # 240 bins
energies_ds, weights_ds = downsample_spectrum(energies, weights, 24)
```
"""
function downsample_spectrum(
    energies::Vector{Float64},
    weights::Vector{Float64},
    n_bins::Int=24
)
    n_original = length(energies)
    n_bins = min(n_bins, n_original)

    if n_bins >= n_original
        return copy(energies), copy(weights)
    end

    # Compute bin boundaries
    bin_size = n_original / n_bins

    new_energies = Vector{Float64}(undef, n_bins)
    new_weights = Vector{Float64}(undef, n_bins)

    for i in 1:n_bins
        start_idx = round(Int, (i - 1) * bin_size) + 1
        end_idx = round(Int, i * bin_size)
        end_idx = min(end_idx, n_original)

        # Energy: fluence-weighted average
        bin_weights = weights[start_idx:end_idx]
        bin_energies = energies[start_idx:end_idx]
        total_weight = sum(bin_weights)

        if total_weight > 0
            new_energies[i] = sum(bin_energies .* bin_weights) / total_weight
        else
            new_energies[i] = mean(bin_energies)
        end

        # Weight: sum (total fluence in bin)
        new_weights[i] = total_weight
    end

    return new_energies, new_weights
end

# =============================================================================
# Unfiltered Spectrum Loading (Anode .TXT format)
# =============================================================================

# Path to unfiltered spectrum data (Anode8, Anode10 subdirectories)
const UNFILTERED_SPECTRUM_DIR = joinpath(@__DIR__, "..", "spectrum")

"""
    load_spectrum_unfiltered(kVp::Int; anode_angle::Int=8) -> (energies, flux)

Load an unfiltered X-ray spectrum from Anode .TXT files.

These are raw tube output spectra without any filtration, measured at 750 mm
from the focal spot. The user should apply additional filters via
`filter_spectrum()` to model the clinical beam.

# File Format
- Line 1: Header string (e.g., "120K08D0W2*")
- Lines 2+: `energy_keV  flux` whitespace-separated
- Energy in 0.5 keV steps (0.5, 1.0, 1.5, ..., kVp)

# Arguments
- `kVp::Int`: Peak voltage (80, 100, 120, or 140)
- `anode_angle::Int`: Anode target angle in degrees (8 or 10)

# Returns
- `energies::Vector{Float64}`: Energy bin centers (keV)
- `flux::Vector{Float64}`: Photon flux (photons/mA/s/mm² at 750mm)

# Example
```julia
energies, flux = load_spectrum_unfiltered(120; anode_angle=8)
# → 240 bins from 0.5 to 120 keV
```
"""
function load_spectrum_unfiltered(kVp::Int; anode_angle::Int=8)
    valid_kvp = [80, 100, 120, 140]
    valid_angles = [8, 10]

    kVp in valid_kvp || error("kVp must be one of $valid_kvp for unfiltered spectra (got $kVp)")
    anode_angle in valid_angles || error("anode_angle must be one of $valid_angles (got $anode_angle)")

    # Build path: src/spectrum/Anode{angle}/{kVp}.TXT
    folder = "Anode$(anode_angle)"
    filename = "$(kVp).TXT"
    filepath = joinpath(UNFILTERED_SPECTRUM_DIR, folder, filename)
    isfile(filepath) || error("Unfiltered spectrum file not found: $filepath")

    # Read file
    all_lines = readlines(filepath)
    lines = filter(l -> !isempty(strip(l)), all_lines)

    # Skip header line (e.g., "120K08D0W2*")
    data_lines = lines[2:end]

    n_bins = length(data_lines)
    energies = Vector{Float64}(undef, n_bins)
    flux = Vector{Float64}(undef, n_bins)

    for i in 1:n_bins
        parts = split(strip(data_lines[i]))
        energies[i] = parse(Float64, parts[1])
        flux[i] = parse(Float64, parts[2])
    end

    return energies, flux
end

"""
    filter_spectrum(energies, flux; filters, sdd_mm=750.0) -> (energies, filtered_flux, total_flux)

Apply material filtration and distance scaling to an unfiltered spectrum.

Implements the Beer-Lambert law for N filter materials:

    T(E) = exp(-Σᵢ μᵢ(E) × tᵢ_mm × 0.1)

where μᵢ(E) is the linear attenuation coefficient (cm⁻¹) and tᵢ is
filter thickness in mm (converted to cm via ×0.1).

Also applies inverse-square-law scaling from the spectrum reference
distance (750 mm) to the scanner SDD.

# Arguments
- `energies::Vector{Float64}`: Energy bin centers (keV)
- `flux::Vector{Float64}`: Unfiltered photon flux (photons/mA/s/mm² at 750mm)

# Keyword Arguments
- `filters::Vector{Tuple{String,Float64}}`: Filter materials and thicknesses in mm.
  Example: `[("Al", 2.5), ("Cu", 0.1), ("Sn", 0.2)]`
- `sdd_mm::Float64`: Source-to-detector distance in mm (default: 750.0 = no scaling)

# Returns
- `energies::Vector{Float64}`: Same energy bins (unchanged)
- `filtered_flux::Vector{Float64}`: Filtered flux at SDD (photons/mA/s/mm²)
- `total_flux::Float64`: Integrated flux Σ flux(E)×ΔE (photons/mA/s/mm² at SDD)

# Example
```julia
# Load unfiltered 120 kVp spectrum
e, f = load_spectrum_unfiltered(120; anode_angle=8)

# Apply 2.5mm Al + 0.1mm Cu filter, scale to SDD=1097mm
e, f_filt, total = filter_spectrum(e, f;
    filters=[("Al", 2.5), ("Cu", 0.1)],
    sdd_mm=1097.0)

# Verify beam hardening: filtered mean energy > unfiltered
@assert spectrum_mean_energy(e, f_filt) > spectrum_mean_energy(e, f)
```
"""
function filter_spectrum(
    energies::Vector{Float64},
    flux::Vector{Float64};
    filters::Vector{Tuple{String,Float64}}=Tuple{String,Float64}[],
    sdd_mm::Float64=750.0
)
    n = length(energies)
    filtered_flux = copy(flux)

    # 1. Apply inverse-square-law distance scaling (750mm → SDD)
    if sdd_mm != 750.0 && sdd_mm > 0.0
        dist_factor = (750.0 / sdd_mm)^2
        filtered_flux .*= dist_factor
    end

    # 2. Apply Beer-Lambert attenuation for each filter material
    for (material, thickness_mm) in filters
        thickness_cm = thickness_mm * 0.1  # mm → cm
        for i in 1:n
            E = energies[i]
            if E > 0.0 && filtered_flux[i] > 0.0
                μ = get_filter_mu(material, E)
                filtered_flux[i] *= exp(-μ * thickness_cm)
            end
        end
    end

    # 3. Compute total integrated flux (for I₀ calculation)
    # ΔE = energy bin width (0.5 keV for Anode spectra)
    ΔE = length(energies) > 1 ? energies[2] - energies[1] : 1.0
    total_flux = sum(filtered_flux) * ΔE

    return energies, filtered_flux, total_flux
end

# Exports
export load_spectrum, load_spectrum_unfiltered, spectrum_mean_energy, downsample_spectrum
export filter_spectrum

