"""
    Physics/Spectrum.jl

Load pre-computed X-ray spectra from XCAT/XCIST .dat files.

No spectrum generation - just load validated spectra from files.
"""

using DelimitedFiles

# Path to spectrum data files
const SPECTRUM_DIR = joinpath(@__DIR__, "..", "..", "spectrum")

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
function load_spectrum(kVp::Int; target_angle::Float64=7.0, source::Symbol=:xspect)
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

    filepath = joinpath(SPECTRUM_DIR, filename)
    isfile(filepath) || error("Spectrum file not found: $filepath")

    # Read file, filtering out empty lines
    all_lines = readlines(filepath)
    lines = filter(l -> !isempty(strip(l)), all_lines)

    n_bins = parse(Int, lines[1])

    energies = Vector{Float64}(undef, n_bins)
    weights = Vector{Float64}(undef, n_bins)

    for i in 1:n_bins
        line = lines[i + 1]
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
    available_spectra()

List all available pre-computed spectra.

# Returns
Dictionary with :xspect and :xcist keys, each containing available kVp values.
"""
function available_spectra()
    return Dict(
        :xspect => Dict(
            :kVp => [70, 80, 90, 100, 110, 120, 130, 140],
            :target_angles => [7.0, 10.0]
        ),
        :xcist => Dict(
            :kVp => [80, 100, 120, 140],
            :target_angles => [7.0]
        )
    )
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

# Exports
export load_spectrum, available_spectra, spectrum_mean_energy
