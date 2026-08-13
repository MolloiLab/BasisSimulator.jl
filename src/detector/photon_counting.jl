"""
    Forward/PhotonCounting.jl

Photon-counting CT (PCCT) detector simulation for BasisSimulator.jl.

# Physics Background

Photon-counting detectors (PCDs) directly convert individual X-ray photons into
electrical signals using semiconductor materials (CdTe, CZT, or Si). Unlike
energy-integrating detectors that sum all absorbed energy, PCDs:

1. **Count individual photons** - each photon registered as a discrete event
2. **Measure photon energy** - pulse height proportional to deposited energy
3. **Apply energy thresholds** - classify photons into energy bins

## Detector Response Model

All detector physics (charge sharing, fluorescence escape, pulse pileup,
anti-coincidence) are handled by the Monte Carlo detector response matrix (DRM)
computed in `mc_response.jl` and MC pileup in `mc_pileup.jl`. The DRM R(E,b)
maps incident photon energy E to probability of registration in bin b,
incorporating all physical effects in a single matrix.

# GPU Compatibility

All functions are GPU-native via AcceleratedKernels.jl:
- Metal (Apple Silicon)
- CUDA (NVIDIA)
- ROCm (AMD)
- CPU fallback

# References

1. Willemink MJ, Persson M. "Photon-counting CT: Technical Principles and
   Clinical Prospects." Radiology. 2018;289(2):293-312. doi:10.1148/radiol.2018172656

2. Taguchi K, Iwanczyk JS. "Vision 20/20: Single photon counting x-ray detectors
   in medical imaging." Med Phys. 2013;40(10):100901. doi:10.1118/1.4820371

3. FDA 510(k) K201501 - Siemens NAEOTOM Alpha

See also: [`compute_mc_drm`](@ref), [`mc_pileup_response`](@ref)
"""

import AcceleratedKernels as AK
using Random

# =============================================================================
# Photon Counting Detector Types
# =============================================================================

"""
    DetectorMaterialPCCT

Semiconductor materials used in photon-counting detectors.
"""
@enum DetectorMaterialPCCT begin
    CDTE_MATERIAL    # Cadmium Telluride (Siemens NAEOTOM)
    CZT_MATERIAL     # Cadmium Zinc Telluride
    SI_MATERIAL      # Silicon (lower Z, limited stopping power)
end

"""
    PhotonCountingDetector{T<:AbstractFloat}

Photon-counting detector specification for PCCT simulation.

# Fields

## Detector Material Properties
- `material::DetectorMaterialPCCT`: Semiconductor material (CdTe, CZT, Si)
- `thickness_mm::T`: Sensor thickness in mm (typical: 1.5-3.0 mm)
- `pixel_size_mm::Tuple{T,T}`: Native dexel size at detector face (row, col) in mm

## Energy Binning
- `energy_thresholds_keV::Vector{T}`: Energy threshold values in keV
- `energy_resolution_keV::T`: Energy resolution FWHM at 60 keV (typical: 8-12 keV)

## Charge Sharing
- `charge_sharing_fwhm_mm::T`: Charge cloud FWHM in mm (typical: 0.05-0.15 mm)
- `enable_charge_sharing::Bool`: Whether to simulate charge sharing

## Pulse Pile-up
- `dead_time_ns::T`: Detector dead time in nanoseconds (typical: 20-100 ns)
- `enable_pile_up::Bool`: Whether to simulate pulse pile-up

## Anti-Coincidence
- `enable_anti_coincidence::Bool`: Whether to apply anti-coincidence logic
- `coincidence_window_ns::T`: Time window for coincidence detection (typical: 20-50 ns)

## Noise
- `electronic_noise_keV::T`: Electronic noise RMS in keV (typical: 1-2 keV)

## Control
- `seed::Union{Nothing, Int}`: Random seed for reproducibility

# Example

```julia
# NAEOTOM Alpha-like detector
detector = PhotonCountingDetector(
    material = CDTE_MATERIAL,
    thickness_mm = 1.6,
    pixel_size_mm = (0.302, 0.302),
    energy_thresholds_keV = [20.0, 35.0, 55.0, 70.0],
    charge_sharing_fwhm_mm = 0.08,
    dead_time_ns = 5.0
)
```

See also: [`_build_pcct_detector`](@ref), [`create_naeotom_alpha`](@ref)
"""
struct PhotonCountingDetector{T<:AbstractFloat}
    # Material properties
    material::DetectorMaterialPCCT
    thickness_mm::T
    pixel_size_mm::Tuple{T,T}

    # Energy binning
    energy_thresholds_keV::Vector{T}
    energy_resolution_keV::T

    # Charge sharing
    charge_sharing_fwhm_mm::T
    enable_charge_sharing::Bool

    # Pulse pile-up
    dead_time_ns::T
    enable_pile_up::Bool

    # Anti-coincidence
    enable_anti_coincidence::Bool
    coincidence_window_ns::T

    # Electronic noise
    electronic_noise_keV::T

    # Spatial binning
    binning_factor::Int         # 1=unbinned, 2=2×2 standard

    # Control
    seed::Union{Nothing, Int}
end

"""
    PhotonCountingDetector(; kwargs...)

Construct a PhotonCountingDetector with keyword arguments.

# Keyword Arguments
- `material::DetectorMaterialPCCT=CDTE_MATERIAL`: Detector material
- `thickness_mm::Float64=1.6`: Sensor thickness
- `pixel_size_mm::Tuple{Float64,Float64}=(0.302, 0.302)`: Pixel size (row, col)
- `energy_thresholds_keV::Vector{Float64}=[20.0, 35.0, 55.0, 70.0]`: Energy thresholds
- `energy_resolution_keV::Float64=10.0`: Energy resolution FWHM at 60 keV (informational — see note)
- `charge_sharing_fwhm_mm::Float64=0.08`: Charge cloud FWHM (informational — see note)
- `enable_charge_sharing::Bool=true`: Enable charge sharing simulation
- `dead_time_ns::Float64=5.0`: Detector dead time
- `enable_pile_up::Bool=true`: Enable pile-up simulation
- `enable_anti_coincidence::Bool=true`: Enable anti-coincidence correction
- `coincidence_window_ns::Float64=30.0`: Coincidence time window
- `electronic_noise_keV::Float64=1.5`: Electronic noise RMS (informational — see note)
- `seed::Union{Nothing,Int}=nothing`: Random seed

!!! note "Informational spectral-response fields"
    `energy_resolution_keV`, `charge_sharing_fwhm_mm`, and
    `electronic_noise_keV` describe the detector but do not drive the
    simulated spectral response. The effective energy resolution, charge
    sharing, K-escape, and electronic-noise smearing are all encoded in the
    Monte-Carlo-derived detector response matrix (`cdte_response_v4.jls`,
    see `mc_response.jl`), which is the single source of detector physics
    on the forward path. These fields are used for display and for the
    qualitative K-edge sensitivity rating in `pcct_spectral.jl`.
"""
function PhotonCountingDetector(;
    material::DetectorMaterialPCCT=CDTE_MATERIAL,
    thickness_mm::Float64=1.6,
    pixel_size_mm::Tuple{Float64,Float64}=(0.302, 0.302),
    energy_thresholds_keV::Vector{Float64}=[20.0, 35.0, 55.0, 70.0],
    energy_resolution_keV::Float64=10.0,
    charge_sharing_fwhm_mm::Float64=0.08,
    enable_charge_sharing::Bool=true,
    dead_time_ns::Float64=5.0,
    enable_pile_up::Bool=true,
    enable_anti_coincidence::Bool=true,
    coincidence_window_ns::Float64=30.0,
    electronic_noise_keV::Float64=1.5,
    binning_factor::Int=1,
    seed::Union{Nothing, Int}=nothing
)
    return PhotonCountingDetector{Float64}(
        material, thickness_mm, pixel_size_mm,
        energy_thresholds_keV, energy_resolution_keV,
        charge_sharing_fwhm_mm, enable_charge_sharing,
        dead_time_ns, enable_pile_up,
        enable_anti_coincidence, coincidence_window_ns,
        electronic_noise_keV, binning_factor, seed
    )
end

# =============================================================================
# Energy-Resolved Sinogram Container
# =============================================================================

"""
    EnergyResolvedSinogram{T,A}

Container for energy-resolved sinogram data from photon-counting CT.

Each energy bin contains a full sinogram, allowing spectral analysis and
reconstruction at specific energies or with spectral weighting.

# Fields
- `bins::Vector{A}`: Vector of sinograms, one per energy bin
- `thresholds_keV::Vector{T}`: Energy thresholds defining bins (N+1 values for N bins)
- `n_cols::Int`: Number of detector columns
- `n_rows::Int`: Number of detector rows
- `n_angles::Int`: Number of projection angles

# Energy Bins
For thresholds [T₁, T₂, ..., Tₙ]:
- Bin 1: T₁ ≤ E < T₂
- Bin 2: T₂ ≤ E < T₃
- ...
- Bin N: Tₙ ≤ E < ∞ (or max tube kVp)

# Example

```julia
# After PCCT forward projection
pcct_sino = pcct_forward_project(volume, geom, detector, spectrum)

# Access individual energy bins
bin1 = pcct_sino.bins[1]  # Lowest energy bin
bin4 = pcct_sino.bins[4]  # Highest energy bin

# Total counts (sum of all bins)
total = sum(pcct_sino.bins)
```

See also: [`pcct_forward_project`](@ref), [`PhotonCountingDetector`](@ref)
"""
struct EnergyResolvedSinogram{T<:AbstractFloat, A<:AbstractArray{T,3}}
    bins::Vector{A}
    thresholds_keV::Vector{T}
    n_cols::Int
    n_rows::Int
    n_angles::Int
end

function EnergyResolvedSinogram(bins::Vector{A}, thresholds_keV::Vector{T}) where {T<:AbstractFloat, A<:AbstractArray{T,3}}
    @assert length(bins) == length(thresholds_keV) "Number of bins must match number of thresholds"
    @assert length(bins) > 0 "Must have at least one bin"
    n_cols, n_rows, n_angles = size(bins[1])
    return EnergyResolvedSinogram{T,A}(bins, thresholds_keV, n_cols, n_rows, n_angles)
end

Base.size(es::EnergyResolvedSinogram) = (es.n_cols, es.n_rows, es.n_angles)
Base.eltype(::EnergyResolvedSinogram{T}) where T = T
n_energy_bins(es::EnergyResolvedSinogram) = length(es.bins)

# =============================================================================
# Spatial Binning (native dexel → binned pixel)
# =============================================================================

"""
    spatial_bin!(output, input, factor)

Sum `factor × factor` native dexels into one binned pixel (GPU-native).

Operates on count-domain data (before log transform). The sum of independent
Poisson counts is Poisson-distributed with rate = sum of rates, so spatial
binning before Poisson noise is physically correct.

# Arguments
- `output::AbstractArray{T,3}`: Binned sinogram `(n_cols÷factor, n_rows÷factor, n_angles)`
- `input::AbstractArray{T,3}`: Native-res sinogram `(n_cols, n_rows, n_angles)`
- `factor::Int`: Binning factor (2 for 2×2)
"""
function spatial_bin!(output::AbstractArray{T,3}, input::AbstractArray{T,3}, factor::Int) where T
    out_cols, out_rows, n_angles = size(output)
    f = Int32(factor)

    let inp = input, out = output, oc = Int32(out_cols), or_ = Int32(out_rows), bf = f
        AK.foreachindex(out) do idx
            idx_0 = Int32(idx - 1)
            col = (idx_0 % oc) + Int32(1)
            idx_0 = idx_0 ÷ oc
            row = (idx_0 % or_) + Int32(1)
            angle = (idx_0 ÷ or_) + Int32(1)

            # Sum the factor×factor block from input
            acc = zero(T)
            col0 = (Int32(col) - Int32(1)) * bf
            row0 = (Int32(row) - Int32(1)) * bf
            for dr in Int32(1):bf
                for dc in Int32(1):bf
                    acc += inp[col0 + dc, row0 + dr, angle]
                end
            end
            out[idx] = acc
        end
    end
    return output
end

# =============================================================================
# Main PCCT Forward Projection API
# =============================================================================

"""
    pcct_forward_project(mask, geom, detector; energies, weights, materials, kwargs...) -> EnergyResolvedSinogram

Perform polychromatic photon-counting CT forward projection with MC DRM spectral response.

This is the main PCCT forward projection API. It:
1. Computes energy-dependent μ-volumes from mask + materials (per energy)
2. Forward projects at each energy using Siddon ray-tracing
3. Applies quantum efficiency η(E) weighting (material-dependent)
4. Applies spectral response matrix R(E,b) from MC DRM for realistic energy binning
5. Applies spatial binning if native dexel resolution is used
6. Returns energy-resolved sinograms in line-integral domain

All detector physics (charge sharing, fluorescence escape, pileup, anti-coincidence)
are encoded in the MC DRM R matrix. No separate analytical detector effects are applied.

# Arguments
- `mask::AbstractArray{UInt8,3}`: Material mask (region indices)
- `geom::CTGeometry`: CT geometry
- `detector::PhotonCountingDetector`: PCCT detector specification

# Keyword Arguments
- `energies::AbstractVector`: Spectral energies in keV
- `weights::AbstractVector`: Spectral weights (normalized photon fluence)
- `materials::Vector`: Material vector (from get_region_materials())
- `I0::Real=1e6`: Reference photon count per detector element

# Returns
`EnergyResolvedSinogram` with one sinogram per energy bin.

# Physics

The per-bin photon count is:
    N_b = Σ_E R(E,b) × S(E) × η(E) × exp(-∫μ(x,E)dl)

where:
- R(E,b) = MC DRM spectral response matrix entry (probability E registers in bin b)
- S(E) = normalized tube spectrum weight
- η(E) = quantum detection efficiency of detector crystal
- ∫μ(x,E)dl = line integral of energy-dependent attenuation

# Example

```julia
detector = _build_pcct_detector(create_naeotom_alpha())
energies, weights = load_spectrum(120)
materials = get_region_materials()

pcct_sino = pcct_forward_project(
    phantom.mask, geom, detector;
    energies=energies, weights=weights,
    materials=materials
)

# Access energy bins
low_E_sino = pcct_sino.bins[1]   # 20-35 keV
high_E_sino = pcct_sino.bins[4]  # 70+ keV
```

See also: [`PhotonCountingDetector`](@ref), [`EnergyResolvedSinogram`](@ref)
"""
function pcct_forward_project(
    mask::AbstractArray{<:Unsigned,3},
    geom,
    detector::PhotonCountingDetector;
    energies::AbstractVector,
    weights::AbstractVector,
    materials::Vector = get_region_materials(),
    I0::Real = 1e6,
    # Workspace buffers (optional — allocate internally if not provided)
    ws_bins = nothing,
    ws_μ_volume = nothing,
    ws_sino_buf = nothing,
    ws_scratch = nothing,
    ws_thresholds_T = nothing,
    # Pre-computed spectral data (optional — compute internally if not provided)
    ws_η = nothing,           # quantum efficiency vector (n_energies)
    ws_R = nothing,           # spectral response matrix (n_energies × n_bins)
    ws_R_energies = nothing,  # energy grid for R
    ws_I0_bins_norm = nothing, # per-bin I0 for normalization
    # Pre-allocated μ lookup buffers for create_μ_volume!
    ws_μ_lut_cpu = nothing,   # Vector{T}(n_regions) CPU
    ws_μ_lut_gpu = nothing,   # similar(mask, T, n_regions) GPU-side
    ws_μ_table = nothing,     # Matrix{T}(n_regions, n_energies) pre-computed μ table
    # Pre-allocated geometry arrays for siddon_forward_project!
    ws_source_positions = nothing,
    ws_detector_centers = nothing,
    ws_detector_u = nothing,
    ws_detector_v = nothing,
    # Override volume bounds for phantom physical extent
    volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
    # Native-resolution forward projection (spatial binning)
    native_geom = nothing,            # CTGeometry at native dexel resolution (nothing = no binning)
    ws_native_bins = nothing,         # per-bin sinograms at native res
    ws_native_sino_buf = nothing,     # Siddon scratch at native res
    ws_native_scratch = nothing,      # scratch at native res
    ws_native_μ_volume = nothing,     # μ volume (same as binned — phantom hasn't changed)
    ws_native_source_positions = nothing,
    ws_native_detector_centers = nothing,
    ws_native_detector_u = nothing,
    ws_native_detector_v = nothing,
    # Tiled spectral projection buffers (optional — enables fused PCCT path)
    ws_μ_table_gpu = nothing,         # GPU [n_regions, n_energies_padded]
    ws_W_matrix_gpu = nothing,        # GPU [n_energies_padded, n_bins]
    ws_outputs_flat = nothing,        # GPU [n_elements * n_bins]
    ws_native_outputs_flat = nothing, # GPU [native_n_elements * n_bins] (for bf>1)
    ws_source_spectral = nothing,    # GPU [n_cols, n_rows, n_energies_padded] heel × bowtie
    # Ray tracer: :dd_fast (default/fastest general path), :dd (DEPRECATED), or :siddon (compatibility, aliases).
    projector::Symbol = :dd_fast,
    # Ignored kwargs for backward compat with callers that still pass them
    kwargs...
)
    T = Float32  # Use Float32 for GPU efficiency

    n_cols = geom.n_cols
    n_rows = geom.n_rows
    n_angles = geom.n_angles
    n_energies = length(energies)
    n_bins = length(detector.energy_thresholds_keV)
    thresholds = detector.energy_thresholds_keV
    kVp = maximum(energies)
    bf = detector.binning_factor

    # Determine whether to use native-resolution path
    use_native = native_geom !== nothing && bf > 1

    # Geometry for forward projection: native if binning, else standard
    proj_geom = use_native ? native_geom : geom

    # Pre-compute quantum efficiency for all energies (CPU, scalar values)
    η = ws_η !== nothing ? ws_η : quantum_efficiency_vector(detector.material, detector.thickness_mm, energies)

    # MC-LUT detector response matrix R(E, bin).  Either passed in via
    # `ws_R` (workspace pre-compute) or built here from `compute_mc_drm`.
    R          = ws_R          !== nothing ? ws_R          : compute_mc_drm(detector, kVp)
    R_energies = ws_R_energies !== nothing ? ws_R_energies : collect(range(1.0, Float64(kVp), length = size(R, 1)))

    # --- Allocate projection-resolution buffers (native or binned) ---
    proj_shape = (proj_geom.n_cols, proj_geom.n_rows, proj_geom.n_angles)
    binned_shape = (n_cols, n_rows, n_angles)

    # Binned-res per-bin sinograms (output)
    bins = if ws_bins !== nothing
        ws_bins
    else
        [similar(mask, T, binned_shape) for _ in 1:n_bins]
    end
    for bin in bins
        fill!(bin, zero(T))
    end

    # =========================================================================
    # TILED SPECTRAL PATH: fused DDA with multi-output (K=16 per tile)
    # Requires: μ_table_gpu + W_matrix_gpu + outputs_flat + spectral response
    # =========================================================================
    if ws_μ_table_gpu !== nothing && ws_W_matrix_gpu !== nothing && ws_outputs_flat !== nothing
        @info "PCCT TILED PATH: K=16, n_energies=$n_energies, n_bins=$n_bins, mask=$(size(mask))" maxlog=1

        TILE_K = 16
        n_energies_padded = size(ws_μ_table_gpu, 2)
        n_tiles = n_energies_padded ÷ TILE_K

        # Select geometry arrays (native or binned)
        _ws_src = use_native ? ws_native_source_positions : ws_source_positions
        _ws_det = use_native ? ws_native_detector_centers : ws_detector_centers
        _ws_u = use_native ? ws_native_detector_u : ws_detector_u
        _ws_v = use_native ? ws_native_detector_v : ws_detector_v

        # Select outputs_flat buffer (native or binned)
        _outputs_flat = use_native ? ws_native_outputs_flat : ws_outputs_flat
        _proj_geom = use_native ? native_geom : geom
        _proj_n_elements = prod(proj_shape)

        # Zero-initialize outputs_flat
        fill!(_outputs_flat, zero(T))

        # Pilot array for AK.foreachindex (sinogram-shaped, provides iteration count)
        _pilot = if use_native
            ws_native_sino_buf !== nothing ? ws_native_sino_buf : similar(mask, T, proj_shape)
        else
            ws_sino_buf !== nothing ? ws_sino_buf : similar(mask, T, binned_shape)
        end

        # Per-tile subset copy + kernel call (same approach as EICT tiled path)
        n_regions = size(ws_μ_table_gpu, 1)
        has_src = ws_source_spectral !== nothing

        # ---------------------------------------------------------------------
        # SINGLE-PASS PATH (:dd_fast, ≤64 materials): per-material path-length
        # DD kernel handles the FULL padded energy range in one volume walk —
        # identical DD footprint/weights, no per-tile re-walk, no subset copies.
        # ---------------------------------------------------------------------
        if projector === :dd_fast && n_regions <= _PLEN_MAX_MATERIALS
            @info "PCCT SINGLE-PASS PATH (:dd_fast path-length): n_energies=$n_energies, n_bins=$n_bins" maxlog=1
            dd_fast_fused_spectral_project!(
                _pilot, _outputs_flat, Int32(n_bins), mask, _proj_geom,
                ws_μ_table_gpu, ws_W_matrix_gpu, Val(n_energies_padded), Int32(1);
                volume_extent=volume_extent,
                ws_source_positions=_ws_src, ws_detector_centers=_ws_det,
                ws_detector_u=_ws_u, ws_detector_v=_ws_v,
                ws_bowtie_spectral=(has_src ? ws_source_spectral : nothing))
            @goto tiles_done
        end

        projector === :dd_fast && _warn_dd_fast_fallback(n_regions)
        μ_sub = similar(mask, T, n_regions, TILE_K)
        W_sub = similar(mask, T, TILE_K, n_bins)
        bt_sub = has_src ? similar(mask, T, size(_pilot, 1), size(_pilot, 2), TILE_K) : nothing

        for tile_idx in 1:n_tiles
            ts = (tile_idx - 1) * TILE_K + 1
            te = ts + TILE_K - 1

            copyto!(μ_sub, @view ws_μ_table_gpu[:, ts:te])
            copyto!(W_sub, @view ws_W_matrix_gpu[ts:te, :])

            if has_src
                copyto!(bt_sub, @view ws_source_spectral[:, :, ts:te])
                _project_fused_spectral!(
                    projector,
                    _pilot, _outputs_flat, Int32(n_bins), mask, _proj_geom,
                    μ_sub, W_sub, Val(TILE_K), Int32(1);
                    volume_extent=volume_extent,
                    ws_source_positions=_ws_src, ws_detector_centers=_ws_det,
                    ws_detector_u=_ws_u, ws_detector_v=_ws_v,
                    ws_bowtie_spectral=bt_sub)
            else
                _project_fused_spectral!(
                    projector,
                    _pilot, _outputs_flat, Int32(n_bins), mask, _proj_geom,
                    μ_sub, W_sub, Val(TILE_K), Int32(1);
                    volume_extent=volume_extent,
                    ws_source_positions=_ws_src, ws_detector_centers=_ws_det,
                    ws_detector_u=_ws_u, ws_detector_v=_ws_v)
            end
        end

        @label tiles_done

        # Unpack outputs_flat → bins (or native_bins if spatial binning)
        if use_native
            # Native-res: unpack to native_bins, then spatial_bin → bins
            proj_bins = if ws_native_bins !== nothing
                ws_native_bins
            else
                [similar(mask, T, proj_shape) for _ in 1:n_bins]
            end
            for b in 1:n_bins
                # Copy from flat buffer to 3D bin array
                offset = (b - 1) * _proj_n_elements
                let pbin = proj_bins[b], oflat = _outputs_flat, off = offset
                    AK.foreachindex(pbin) do idx
                        pbin[idx] = oflat[idx + off]
                    end
                end
            end
            for b in 1:n_bins
                spatial_bin!(bins[b], proj_bins[b], bf)
            end
        else
            # Direct: unpack flat buffer → bins
            n_elements = prod(binned_shape)
            for b in 1:n_bins
                offset = (b - 1) * n_elements
                let ba = bins[b], oflat = _outputs_flat, off = offset
                    AK.foreachindex(ba) do idx
                        ba[idx] = oflat[idx + off]
                    end
                end
            end
        end

        # Convert from photon counts to line-integral domain: sino = -log(N / I₀_bin)
        eps_val = T(1e-10)
        I0_scale = use_native ? Float64(bf * bf) : 1.0
        I0_bins_norm = if ws_I0_bins_norm !== nothing
            ws_I0_bins_norm
        else
            [_compute_bin_I0(detector, energies, weights, η, thresholds, b,
                              Float64(kVp), Float64(I0); R=R) for b in 1:n_bins]
        end
        for b in 1:n_bins
            let I0_bin_T = T(I0_bins_norm[b] * I0_scale), ba = bins[b], eps = eps_val
                AK.foreachindex(ba) do idx
                    ba[idx] = -log(max(ba[idx], eps) / I0_bin_T)
                end
            end
        end

        # Thresholds
        thresh_T = if ws_thresholds_T !== nothing
            for i in eachindex(ws_thresholds_T)
                ws_thresholds_T[i] = T(detector.energy_thresholds_keV[i])
            end
            ws_thresholds_T
        else
            T.(detector.energy_thresholds_keV)
        end
        return EnergyResolvedSinogram(bins, thresh_T)
    end

    # =========================================================================
    # SEQUENTIAL FALLBACK: per-energy ray-tracing (original path)
    # Used when tiled buffers are not provided (standalone calls without workspace)
    # =========================================================================
    @info "PCCT SEQUENTIAL PATH: n_energies=$n_energies, n_bins=$n_bins" maxlog=1

    # Native-res per-bin count sinograms (for spatial binning)
    proj_bins = if use_native
        nb = ws_native_bins !== nothing ? ws_native_bins : [similar(mask, T, proj_shape) for _ in 1:n_bins]
        for bin in nb; fill!(bin, zero(T)); end
        nb
    else
        nothing
    end

    # The bins we accumulate into during the energy loop
    accum_bins = use_native ? proj_bins : bins

    # Temporary μ-volume (reused across energies, same device as mask)
    μ_volume = if use_native && ws_native_μ_volume !== nothing
        ws_native_μ_volume
    elseif ws_μ_volume !== nothing
        ws_μ_volume
    else
        similar(mask, T, size(mask))
    end

    # Temporary sinogram buffer at projection resolution
    sino_buf = if use_native
        ws_native_sino_buf !== nothing ? ws_native_sino_buf : similar(mask, T, proj_shape)
    else
        ws_sino_buf !== nothing ? ws_sino_buf : similar(mask, T, binned_shape)
    end

    # Geometry arrays for Siddon (native or binned)
    _ws_src = use_native ? ws_native_source_positions : ws_source_positions
    _ws_det = use_native ? ws_native_detector_centers : ws_detector_centers
    _ws_u = use_native ? ws_native_detector_u : ws_detector_u
    _ws_v = use_native ? ws_native_detector_v : ws_detector_v

    # Per-energy ray-tracing with spectral weighting
    for (e_idx, E) in enumerate(energies)
        E_float = Float64(E)

        # Skip energies below lowest threshold (not detected)
        if E_float < thresholds[1]
            continue
        end

        # Skip negligible spectrum weights
        w = Float64(weights[e_idx])
        if w < 1e-12
            continue
        end

        # Create energy-dependent μ-volume from mask + materials
        create_μ_volume!(μ_volume, mask, materials, E_float;
                         ws_μ_lut_cpu=ws_μ_lut_cpu, ws_μ_lut_gpu=ws_μ_lut_gpu,
                         ws_μ_table=ws_μ_table, energy_idx=e_idx)

        # Forward project at this energy (native or binned resolution)
        fill!(sino_buf, zero(T))
        _project_mono!(projector, sino_buf, μ_volume, proj_geom;
            ws_source_positions=_ws_src,
            ws_detector_centers=_ws_det,
            ws_detector_u=_ws_u,
            ws_detector_v=_ws_v,
            volume_extent=volume_extent)

        # Weight by spectrum and quantum efficiency
        η_E = T(η[e_idx])
        w_T = T(w)
        I0_T = T(I0)

        # Spectrum × QE × MC-LUT DRM column for each bin: wt = I0 · w(E) · η(E) · R(E, b).
        # The MC DRM is the single source of detector physics (charge sharing,
        # fluorescence escape, Compton escape, Fano broadening) — ideal-bin
        # fallback is intentionally absent.
        n_R = size(R, 1)
        r_idx = clamp(round(Int, (E_float - 1.0) / (Float64(kVp) - 1.0) * (n_R - 1)) + 1, 1, n_R)

        for b in 1:n_bins
            R_val = T(R[r_idx, b])
            if R_val < T(1e-10)
                continue
            end
            let wt = I0_T * w_T * η_E * R_val, ba = accum_bins[b]
                AK.foreachindex(sino_buf) do idx
                    ba[idx] += wt * exp(-sino_buf[idx])
                end
            end
        end
    end

    # --- Spatial binning (native → binned) if applicable ---
    if use_native
        for b in 1:n_bins
            spatial_bin!(bins[b], proj_bins[b], bf)
        end
    end

    # Convert from photon counts to line-integral domain: sino = -log(N / I₀_bin)
    # When binning, I0 scales by bf² (sum of bf² dexels)
    eps_val = T(1e-10)
    I0_scale = use_native ? Float64(bf * bf) : 1.0
    I0_bins_norm = if ws_I0_bins_norm !== nothing
        ws_I0_bins_norm
    else
        [_compute_bin_I0(detector, energies, weights, η, thresholds, b,
                          Float64(kVp), Float64(I0); R=R) for b in 1:n_bins]
    end
    for b in 1:n_bins
        let I0_bin_T = T(I0_bins_norm[b] * I0_scale), ba = bins[b], eps = eps_val
            AK.foreachindex(ba) do idx
                ba[idx] = -log(max(ba[idx], eps) / I0_bin_T)
            end
        end
    end

    # Use workspace thresholds if provided, otherwise allocate
    thresh_T = if ws_thresholds_T !== nothing
        for i in eachindex(ws_thresholds_T)
            ws_thresholds_T[i] = T(detector.energy_thresholds_keV[i])
        end
        ws_thresholds_T
    else
        T.(detector.energy_thresholds_keV)
    end
    return EnergyResolvedSinogram(bins, thresh_T)
end

"""
    _compute_bin_I0(detector, energies, weights, η, thresholds, bin_idx, kVp, I0; R) -> Float64

Reference photon count I₀ for one energy bin (the unattenuated count expected
in `bin_idx` — the denominator for the `-log(N / I0)` step in
`pcct_forward_project`).

`R` is the MC-LUT detector response matrix (`compute_mc_drm`).  Mapping
each spectrum energy onto its R-row and summing `I0 · w(E) · η(E) · R(E, bin_idx)`
keeps this consistent with the forward-projection accumulator and guarantees
the `-log(N / I0)` normalization produces well-defined line integrals.
"""
function _compute_bin_I0(detector, energies, weights, η, thresholds, bin_idx, kVp, I0; R)
    n_energies = length(energies)
    n_R = size(R, 1)
    I0_bin = 0.0
    for e_idx in 1:n_energies
        E_float = Float64(energies[e_idx])
        w = Float64(weights[e_idx])
        if w < 1e-12
            continue
        end
        # Map spectrum energy to DRM grid index
        r_idx = clamp(round(Int, (E_float - 1.0) / (kVp - 1.0) * (n_R - 1)) + 1, 1, n_R)
        R_val = R[r_idx, bin_idx]
        I0_bin += I0 * w * η[e_idx] * R_val
    end
    return max(I0_bin, 1.0)  # Avoid division by zero
end

# =============================================================================
# PCCT Noise Model (PCCT-NOISE-DECOMP)
# =============================================================================

"""
    apply_pcct_noise!(sino::EnergyResolvedSinogram, I0_bins;
                      seed=nothing, ws_noise_staging=nothing, ws_rng=nothing,
                      noise_reduction=0.0) -> EnergyResolvedSinogram

Draw exact per-bin Poisson count realizations in-place.

Incident photons are Poisson distributed and the MC-LUT detector response
(quantum efficiency + spectral migration) acts as independent per-photon
thinning, so the recorded counts in each bin are exactly Poisson with the
DRM-shaped expected counts as their mean.  Charge-sharing double counting
across bins is the one residual effect outside this model.  NO electronic
noise — eliminated by the counting thresholds, the fundamental PCCT
advantage.

`I0_bins` must be the SAME per-bin air calibration the sinograms are
normalized against (`ws.I0_bins`), so a downstream `I0 · exp(-h)` recovers
the sampled integer counts exactly.

For each bin `b` and ray:
1. expected counts `λ = I0_bins[b] · exp(-h)`;
2. `N ~ Poisson(λ)` — exact integer sampling at every λ (`_poisson_sample`);
3. optional `noise_reduction` blend `λ + (1 - nr)·(N - λ)`; any nonzero value
   leaves the strict Poisson count model (vendor-denoising surrogate only);
4. the measured count is written verbatim to `raw_out[b]` when provided
   (TRUE ZEROS preserved), then floored at 1 ONLY for the log-domain bins —
   a zero count has no finite log line integral — and stored back as
   `h = -log(max(N, 1) / I0_bins[b])`.

The simulator imposes no zero-count policy on the counts product: any
epsilon/interpolation convention for zeros is a downstream (user) choice.

# Arguments
- `sino::EnergyResolvedSinogram`: per-bin log line integrals, mutated in place.
- `I0_bins::AbstractVector`: per-bin air-calibration counts (one per bin).

# Keyword Arguments
- `seed::Union{Nothing,Int}`: RNG seed for reproducibility.
- `ws_noise_staging`: optional pre-allocated sinogram-shaped CPU buffer.
- `ws_rng`: optional pre-allocated RNG (re-seeded each call).
- `noise_reduction::Float64`: 0.0 = exact Poisson counts (default).
- `raw_out`: optional vector of per-bin arrays (same backend/shape as the
  bins) that receives the measured counts BEFORE the log-domain floor.
  With `noise_reduction = 0` these are the exact integer draws.
"""
function apply_pcct_noise!(
    sino::EnergyResolvedSinogram{T,A},
    I0_bins::AbstractVector;
    seed::Union{Nothing,Int} = nothing,
    ws_noise_staging = nothing,
    ws_rng = nothing,
    noise_reduction::Float64 = 0.0,
    raw_out = nothing,
) where {T, A}
    length(sino.bins) == length(I0_bins) ||
        throw(DimensionMismatch("one I0 value is required per PCCT bin"))
    raw_out === nothing || length(raw_out) == length(sino.bins) ||
        throw(DimensionMismatch("one raw_out array is required per PCCT bin"))

    rng = if ws_rng !== nothing
        Random.seed!(ws_rng, isnothing(seed) ? 0 : seed)
        ws_rng
    else
        isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    end

    cpu_buf = ws_noise_staging !== nothing ? ws_noise_staging : Array(sino.bins[1])
    nr_scale = 1.0 - noise_reduction

    for (b, bin) in enumerate(sino.bins)
        I0_bin = Float64(I0_bins[b])

        # Bulk GPU→CPU transfer (reuses pre-allocated buffer)
        copyto!(cpu_buf, bin)

        # Pass 1: measured counts (pre-floor; exact integers when nr = 0)
        @inbounds for idx in eachindex(cpu_buf)
            λ = I0_bin * exp(-Float64(cpu_buf[idx]))
            N = Float64(_poisson_sample(rng, λ))
            if nr_scale != 1.0
                N = λ + nr_scale * (N - λ)
            end
            cpu_buf[idx] = T(N)
        end
        raw_out === nothing || copyto!(raw_out[b], cpu_buf)

        # Pass 2: floor at 1 for the log domain only, convert back in place
        @inbounds for idx in eachindex(cpu_buf)
            cpu_buf[idx] = T(-log(Float64(max(cpu_buf[idx], one(T))) / I0_bin))
        end

        # Bulk CPU→GPU transfer
        copyto!(bin, cpu_buf)
    end

    return sino
end

# log(k!) for k = 0…20 exactly; Stirling–de Moivre series beyond (rel. error
# < 1e-10 at k = 21, decreasing with k — far below rejection-test resolution).
const _LOG_FACTORIAL_TABLE = Float64[sum(log(j) for j in 1:k; init = 0.0) for k in 0:20]

function _log_factorial(k::Int)
    k <= 20 && return @inbounds _LOG_FACTORIAL_TABLE[k + 1]
    invk = 1.0 / k
    return (k + 0.5) * log(k) - k + 0.9189385332046728 +
        invk * (1.0 / 12.0 - invk * invk * (1.0 / 360.0 - invk * invk / 1260.0))
end

"""
    _poisson_sample(rng, λ) -> Int

Exact Poisson sample at any λ: sequential inversion below λ = 10, Hörmann's
PTRS transformed rejection above (W. Hörmann, "The transformed rejection
method for generating Poisson random variables", Insurance: Mathematics and
Economics 12, 39–45, 1993).  No Gaussian approximation at any count level.
"""
function _poisson_sample(rng, λ::Float64)
    if λ < 1e-10
        return 0
    elseif λ < 10.0
        # Inversion by sequential search (Knuth): exact, O(λ).
        L = exp(-λ)
        k = 0
        p = 1.0
        while true
            p *= rand(rng)
            p < L && return k
            k += 1
        end
    else
        # PTRS: O(1) exact transformed-rejection sampling for λ ≥ 10.
        b = 0.931 + 2.53 * sqrt(λ)
        a = -0.059 + 0.02483 * b
        inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
        v_r = 0.9277 - 3.6224 / (b - 2.0)
        logλ = log(λ)
        while true
            U = rand(rng) - 0.5
            V = rand(rng)
            us = 0.5 - abs(U)
            k = floor(Int, (2.0 * a / us + b) * U + λ + 0.43)
            if us >= 0.07 && V <= v_r
                return k
            end
            if k < 0 || (us < 0.013 && V > us)
                continue
            end
            if log(V) + log(inv_alpha) - log(a / (us * us) + b) <=
                    k * logλ - λ - _log_factorial(k)
                return k
            end
        end
    end
end

# Note: synthesize_vmi and get_material_attenuation_pcct are defined in PCCTSpectral.jl
# (requires PCCTMaterialMap which is defined there)

# =============================================================================
# Utility Functions
# =============================================================================

"""
    get_pcct_detector_info(detector::PhotonCountingDetector) -> NamedTuple

Get diagnostic information about PCCT detector configuration.
"""
function get_pcct_detector_info(detector::PhotonCountingDetector{T}) where T
    n_bins = length(detector.energy_thresholds_keV)

    return (
        material = detector.material,
        thickness_mm = detector.thickness_mm,
        pixel_size_mm = detector.pixel_size_mm,
        n_energy_bins = n_bins,
        thresholds_keV = detector.energy_thresholds_keV,
        energy_resolution_keV = detector.energy_resolution_keV,
        charge_sharing_enabled = detector.enable_charge_sharing,
        charge_sharing_fwhm_mm = detector.charge_sharing_fwhm_mm,
        pile_up_enabled = detector.enable_pile_up,
        dead_time_ns = detector.dead_time_ns,
        anti_coincidence_enabled = detector.enable_anti_coincidence,
        electronic_noise_keV = detector.electronic_noise_keV
    )
end

"""
    print_pcct_detector_info(detector::PhotonCountingDetector)

Print formatted PCCT detector specification.
"""
function print_pcct_detector_info(detector::PhotonCountingDetector)
    info = get_pcct_detector_info(detector)

    println("=" ^ 60)
    println("PHOTON-COUNTING DETECTOR SPECIFICATION")
    println("=" ^ 60)
    println("Material:           $(info.material)")
    println("Thickness:          $(info.thickness_mm) mm")
    println("Pixel Size:         $(info.pixel_size_mm[1]) × $(info.pixel_size_mm[2]) mm")
    println()
    println("ENERGY BINNING")
    println("-" ^ 40)
    println("Number of bins:     $(info.n_energy_bins)")
    println("Thresholds (keV):   $(info.thresholds_keV)")
    println("Energy resolution:  $(info.energy_resolution_keV) keV FWHM")
    println()
    println("DETECTOR EFFECTS (via MC DRM)")
    println("-" ^ 40)
    println("Charge sharing:     $(info.charge_sharing_enabled ? "ON" : "OFF") (FWHM=$(info.charge_sharing_fwhm_mm) mm)")
    println("Pulse pile-up:      $(info.pile_up_enabled ? "ON" : "OFF") (τ=$(info.dead_time_ns) ns)")
    println("Anti-coincidence:   $(info.anti_coincidence_enabled ? "ON" : "OFF")")
    println("Electronic noise:   $(info.electronic_noise_keV) keV RMS")
    println("=" ^ 60)
end

# =============================================================================
# Material-Dependent Detector Physics (PCCT-MATERIAL-MODEL)
# =============================================================================

# Material-agnostic architecture: ALL physics functions dispatch on
# DetectorMaterialPCCT enum. CZT and Si work by adding entries to
# get_detector_material_properties() — no other code changes needed.

"""
    get_detector_material_properties(material::DetectorMaterialPCCT) -> NamedTuple

Master lookup for ALL material-dependent PCCT detector parameters.

This is the SINGLE source of truth for material constants. All physics functions
dispatch through this function to ensure material-agnostic operation.

# Returns NamedTuple with fields:
- `elements`: Element symbols
- `atomic_numbers`: Z values for each element
- `mass_fractions`: Mass fraction of each element in compound
- `density_g_cm3`: Crystal density
- `k_edges_keV`: K-edge energies for each element
- `k_alpha_keV`: K-α fluorescence line energies
- `k_yields`: Fluorescence yields ω_K for each element
- `k_fluorescence_range_mm`: Mean free path of fluorescence X-rays in crystal
- `mu_e_tau_e`: Electron mobility-lifetime product (cm²/V)
- `mu_h_tau_h`: Hole mobility-lifetime product (cm²/V)
- `bias_voltage_V`: Typical operating bias voltage
- `pair_creation_energy_eV`: Energy per electron-hole pair
- `fano_factor`: Fano factor for intrinsic energy resolution
- `charge_cloud_sigma_mm`: Charge cloud size at typical bias
- `energy_resolution_fwhm_keV`: Measured system FWHM at 60 keV

# References
- CdTe: Taguchi & Iwanczyk, Med Phys 2013; del Risco Norrlid et al, NIMPA 2017
- CZT: Pennicard et al, JINST 2014; Veale et al, NIMPA 2014
- Si: Persson et al, Phys Med Biol 2014; Fredenberg et al, NIMPA 2010

# Example
```julia
props = get_detector_material_properties(CDTE_MATERIAL)
props.density_g_cm3  # 5.85
props.k_edges_keV    # [26.7, 31.8]
```
"""
function get_detector_material_properties(material::DetectorMaterialPCCT)
    if material == CDTE_MATERIAL
        # Cadmium Telluride — primary PCCT detector material (Siemens NAEOTOM Alpha)
        # Atomic weights: Cd = 112.411, Te = 127.60 → total = 240.011
        return (
            elements = [:Cd, :Te],
            atomic_numbers = [48, 52],
            mass_fractions = [112.411 / 240.011, 127.60 / 240.011],  # [0.4683, 0.5317]
            density_g_cm3 = 5.85,

            # K-edges and fluorescence (NIST XCOM, Koch-Mehrin 2020 Table 1)
            k_edges_keV = [26.711, 31.814],       # Cd K-edge, Te K-edge (NIST)
            k_alpha_keV = [23.2, 27.5],           # Cd K-α, Te K-α fluorescence
            k_yields = [0.84, 0.88],              # Fluorescence yields ω_K (Bambynek 1972)
            k_fluorescence_range_mm = [0.10, 0.06], # Koch-Mehrin 2020: Cd Kα ~100μm, Te Kα ~60μm

            # Charge transport (Koch-Mehrin 2020 + Owens 2012)
            mu_e_tau_e = 3.3e-3,                  # cm²/V — μ_e=1100×τ_e=3μs (Koch-Mehrin 2020)
            mu_h_tau_h = 2.0e-4,                  # cm²/V — μ_h=100×τ_h=2μs (Koch-Mehrin 2020)
            bias_voltage_V = 800.0,               # Typical operating voltage

            # Intrinsic energy resolution (Koch-Mehrin 2020, Redus 2009)
            pair_creation_energy_eV = 4.3,        # eV per e-h pair (Koch-Mehrin 2020)
            fano_factor = 0.1,                    # Intrinsic variance reduction (Redus 2009)
            charge_cloud_sigma_mm = 0.013,        # Physics-based: ~13μm from ODE (Koch-Mehrin 2020)

            # System-level measured resolution (includes electronics)
            energy_resolution_fwhm_keV = 10.0     # Typical for NAEOTOM at 60 keV
        )
    elseif material == CZT_MATERIAL
        # Cadmium Zinc Telluride (Cd₀.₉Zn₀.₁Te)
        # Zn substitutes ~10% of Cd → slightly different properties
        # Atomic weights: Cd=112.411, Zn=65.38, Te=127.60
        # CZT formula: Cd₀.₉Zn₀.₁Te → effective mass per formula unit
        # = 0.9×112.411 + 0.1×65.38 + 127.60 = 234.307
        eff_mass = 0.9 * 112.411 + 0.1 * 65.38 + 127.60  # 234.307
        return (
            elements = [:Cd, :Zn, :Te],
            atomic_numbers = [48, 30, 52],
            mass_fractions = [0.9 * 112.411 / eff_mass, 0.1 * 65.38 / eff_mass, 127.60 / eff_mass],
            density_g_cm3 = 5.78,                 # Slightly lower than CdTe

            # K-edges and fluorescence
            k_edges_keV = [26.7, 9.66, 31.8],    # Cd, Zn, Te K-edges
            k_alpha_keV = [23.2, 8.63, 27.4],    # Cd, Zn, Te K-α lines
            k_yields = [0.84, 0.47, 0.87],       # ω_K values (Zn lower yield)
            k_fluorescence_range_mm = [0.12, 0.50, 0.10], # Zn K-α has long range (low-E)

            # Charge transport (Pennicard et al 2014)
            mu_e_tau_e = 1.0e-2,                  # cm²/V (better than CdTe)
            mu_h_tau_h = 5.0e-5,                  # cm²/V (similar hole trapping)
            bias_voltage_V = 600.0,               # Lower voltage needed

            # Intrinsic energy resolution
            pair_creation_energy_eV = 4.64,       # eV per e-h pair (Owens 2004)
            fano_factor = 0.089,                  # Better than CdTe (Pennicard 2014)
            charge_cloud_sigma_mm = 0.045,        # Slightly larger cloud

            energy_resolution_fwhm_keV = 8.0      # CZT typically better resolution
        )
    elseif material == SI_MATERIAL
        # Silicon — excellent charge transport but very low stopping power at CT energies
        # Atomic weight: Si = 28.085
        return (
            elements = [:Si],
            atomic_numbers = [14],
            mass_fractions = [1.0],
            density_g_cm3 = 2.33,

            # K-edge (irrelevant for CT — 1.84 keV is far below useful range)
            k_edges_keV = [1.84],
            k_alpha_keV = [1.74],                 # Si K-α (irrelevant at CT energies)
            k_yields = [0.05],                    # Very low fluorescence yield for low-Z
            k_fluorescence_range_mm = [0.001],    # Absorbed immediately

            # Charge transport (excellent! — Persson 2014)
            mu_e_tau_e = 1.0,                     # cm²/V (near-perfect collection)
            mu_h_tau_h = 0.5,                     # cm²/V (also excellent)
            bias_voltage_V = 200.0,               # Low voltage sufficient

            # Intrinsic energy resolution (best of all three)
            pair_creation_energy_eV = 3.62,       # eV per e-h pair (lowest)
            fano_factor = 0.115,                  # Similar to CdTe
            charge_cloud_sigma_mm = 0.015,        # Very small cloud (light material)

            energy_resolution_fwhm_keV = 2.5      # Excellent energy resolution
        )
    else
        error("Unknown detector material: $material")
    end
end

"""
    get_detector_material_attenuation(material::DetectorMaterialPCCT, energy_keV::Real) -> Float64

Get linear attenuation coefficient μ (cm⁻¹) for PCCT detector material at given energy.

Uses the tabulated data in DetectorEfficiency.jl (SCINTILLATOR_MU_DATA) with log-linear
interpolation. This provides NIST XCOM-calibrated values with proper K-edge handling.

# Arguments
- `material::DetectorMaterialPCCT`: Detector material enum
- `energy_keV::Real`: Photon energy in keV

# Returns
- `Float64`: Linear attenuation coefficient in cm⁻¹

# Example
```julia
μ = get_detector_material_attenuation(CDTE_MATERIAL, 60.0)  # ~33 cm⁻¹
μ = get_detector_material_attenuation(SI_MATERIAL, 60.0)    # ~0.46 cm⁻¹
```
"""
function get_detector_material_attenuation(material::DetectorMaterialPCCT, energy_keV::Real)
    mat_name = if material == CDTE_MATERIAL
        "CdTe"
    elseif material == CZT_MATERIAL
        "CZT"
    elseif material == SI_MATERIAL
        "Si"
    else
        error("Unknown detector material: $material")
    end
    return get_scintillator_mu(mat_name, Float64(energy_keV))
end

"""
    quantum_efficiency(material::DetectorMaterialPCCT, thickness_mm::Real, energy_keV::Real) -> Float64

Compute quantum detection efficiency η(E) for a PCCT detector crystal.

The probability that a photon of energy E is absorbed in the detector:
    η(E) = 1 - exp(-μ_det(E) × d)

where μ_det(E) is the linear attenuation of the detector material and d is thickness.

# Arguments
- `material::DetectorMaterialPCCT`: Detector crystal material
- `thickness_mm::Real`: Crystal thickness in mm
- `energy_keV::Real`: Photon energy in keV

# Returns
- `Float64`: Detection efficiency in [0, 1]

# Physics
- CdTe 1.6mm: η ≈ 0.99 at 60 keV (nearly opaque)
- CZT 2.0mm: η ≈ 0.99 at 60 keV (similar to CdTe)
- Si 1.6mm: η ≈ 0.05 at 60 keV (mostly transparent!)
- Si needs 30+ mm thickness for >90% efficiency at CT energies

# Example
```julia
η_cdte = quantum_efficiency(CDTE_MATERIAL, 1.6, 60.0)  # ≈ 0.995
η_si = quantum_efficiency(SI_MATERIAL, 1.6, 60.0)      # ≈ 0.071
```
"""
function quantum_efficiency(material::DetectorMaterialPCCT, thickness_mm::Real, energy_keV::Real)
    μ = get_detector_material_attenuation(material, energy_keV)  # cm⁻¹
    thickness_cm = Float64(thickness_mm) / 10.0
    return 1.0 - exp(-μ * thickness_cm)
end

"""
    quantum_efficiency_vector(material::DetectorMaterialPCCT, thickness_mm::Real,
                              energies::AbstractVector) -> Vector{Float64}

Compute quantum efficiency η(E) for a vector of energies (batch operation).

More efficient than calling `quantum_efficiency` in a loop for spectrum processing.
"""
function quantum_efficiency_vector(material::DetectorMaterialPCCT, thickness_mm::Real,
                                    energies::AbstractVector)
    return [quantum_efficiency(material, thickness_mm, E) for E in energies]
end

# =============================================================================
# Exports
# =============================================================================

export DetectorMaterialPCCT, CDTE_MATERIAL, CZT_MATERIAL, SI_MATERIAL
export PhotonCountingDetector
export EnergyResolvedSinogram, n_energy_bins
export pcct_forward_project
export get_pcct_detector_info, print_pcct_detector_info
export get_detector_material_properties, get_detector_material_attenuation
export quantum_efficiency, quantum_efficiency_vector
export spatial_bin!
export apply_pcct_noise!
