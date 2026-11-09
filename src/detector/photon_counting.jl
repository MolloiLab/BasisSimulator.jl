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

See also: [`compute_mc_drm`](@ref), [`compute_mc_pileup_matrix`](@ref)
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
    CDTE_MATERIAL # Cadmium Telluride (Siemens NAEOTOM)
    CZT_MATERIAL  # Cadmium Zinc Telluride
    SI_MATERIAL   # Silicon (lower Z, limited stopping power)
end

for (label, what, syms) in (
        (
            :CDTE_MATERIAL,
            "cadmium telluride (CdTe, ρ = 5.85 g/cm³; the Siemens NAEOTOM Alpha sensor)",
            ":CdTe",
        ),
        (:CZT_MATERIAL, "cadmium zinc telluride (Cd₀.₉Zn₀.₁Te, ρ ≈ 5.78 g/cm³)", ":CZT"),
        (
            :SI_MATERIAL,
            "silicon (ρ = 2.33 g/cm³; low Z, so limited stopping power at CT energies)",
            ":Si",
        ),
    )
    local doc = """
        $(label)::DetectorMaterialPCCT

    Photon-counting sensor material: $(what). A [`PCCTScanner`](@ref) selects it with
    `detector_material = $(syms)`. [`get_detector_material_properties`](@ref) returns its
    physical constants, and [`quantum_efficiency`](@ref) gives its absorption.
    """
    @eval @doc $doc $label
end

"""
    PhotonCountingDetector{T<:AbstractFloat}

Photon-counting detector specification for PCCT simulation: the sensor, its thresholds, its dead
time and its binning. Everything else about its response (charge sharing, K-escape, electronic
noise, the anti-scatter grid) is the response table's (`load_mc_response`).

# Fields
- `material::DetectorMaterialPCCT`: semiconductor (CdTe, CZT, Si)
- `thickness_mm::T`: sensor thickness, mm
- `pixel_size_mm::Tuple{T,T}`: a native dexel's pitch at the detector face (row, col), mm
- `energy_thresholds_keV::Vector{T}`: thresholds, keV
- `dead_time_ns::T`: each native dexel's comparator dead time, ns
- `binning_factor::Int`: native dexels summed per pixel side (1 = unbinned, 2 = 2 × 2)

Built from a [`PCCTScanner`](@ref) by the internal `_build_pcct_detector(scanner)`.
"""
struct PhotonCountingDetector{T <: AbstractFloat}
    material::DetectorMaterialPCCT
    thickness_mm::T
    pixel_size_mm::Tuple{T, T}
    energy_thresholds_keV::Vector{T}
    dead_time_ns::T
    binning_factor::Int
end

"""
    PhotonCountingDetector(; material = CDTE_MATERIAL, thickness_mm = 1.6,
                           pixel_size_mm = (0.34, 0.364), energy_thresholds_keV = [20.0, 35.0, 55.0, 70.0],
                           dead_time_ns = 27.0, binning_factor = 1)

Construct a [`PhotonCountingDetector`](@ref) from keywords.
"""
function PhotonCountingDetector(;
    material::DetectorMaterialPCCT = CDTE_MATERIAL,
    thickness_mm::Float64 = 1.6,
    pixel_size_mm::Tuple{Float64, Float64} = (0.34, 0.364),
    energy_thresholds_keV::Vector{Float64} = [20.0, 35.0, 55.0, 70.0],
    dead_time_ns::Float64 = 27.0,
    binning_factor::Int = 1,
)
    return PhotonCountingDetector{Float64}(
        material,
        thickness_mm,
        pixel_size_mm,
        energy_thresholds_keV,
        dead_time_ns,
        binning_factor,
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
- `thresholds_keV::Vector{T}`: Lower energy threshold of each bin (N values for N bins)
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
bin1 = pcct_sino.bins[1] # Lowest energy bin
bin4 = pcct_sino.bins[4] # Highest energy bin

# Total counts (sum of all bins)
total = sum(pcct_sino.bins)
```

See also: [`pcct_forward_project`](@ref), [`PhotonCountingDetector`](@ref)
"""
struct EnergyResolvedSinogram{T <: AbstractFloat, A <: AbstractArray{T, 3}}
    bins::Vector{A}
    thresholds_keV::Vector{T}
    n_cols::Int
    n_rows::Int
    n_angles::Int
end

function EnergyResolvedSinogram(bins::Vector{A}, thresholds_keV::Vector{T}) where {T <:
    AbstractFloat, A <: AbstractArray{T, 3}}
    @assert length(bins) ==
        length(thresholds_keV) "Number of bins must match number of thresholds"
    @assert length(bins) > 0 "Must have at least one bin"
    n_cols, n_rows, n_angles = size(bins[1])
    return EnergyResolvedSinogram{T, A}(bins, thresholds_keV, n_cols, n_rows, n_angles)
end

Base.size(es::EnergyResolvedSinogram) = (es.n_cols, es.n_rows, es.n_angles)
Base.eltype(::EnergyResolvedSinogram{T}) where {T} = T
"""
    n_energy_bins(sino::EnergyResolvedSinogram) -> Int

Number of energy bins in `sino`, `length(sino.bins)`.
"""
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
function spatial_bin!(
    output::AbstractArray{T, 3},
    input::AbstractArray{T, 3},
    factor::Int,
) where {T}
    out_cols, out_rows, _ = size(output)
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
    _normalize_bins_per_ray!(bins, I0, …; bf)

`bins[b][col, row, view] = -log(N / I0[col, row, b])`: every bin becomes the log-transmission
against its own ray's air response, so an air ray reads 0 in every column whatever the bowtie
does across the fan.  `I0` is the workspace's `[n_cols, n_rows, n_bins]` table; without a
workspace a flat one is built from `_compute_bin_I0` (scaled by `bf²` when the counts were
binned from native dexels).
"""
function _normalize_bins_per_ray!(
    bins,
    I0,
    detector,
    energies,
    weights,
    thresholds,
    kVp,
    I0_phys,
    R,
    bf,
)
    T = eltype(bins[1])
    n_bins = length(bins)
    nc = Int32(size(bins[1], 1))
    nr = Int32(size(bins[1], 2))
    ncnr = nc * nr
    I0_use = if I0 !== nothing
        I0
    else
        flat = similar(bins[1], T, Int(nc), Int(nr), n_bins)
        for b in 1:n_bins
            fill!(
                view(flat, :, :, b),
                T(
                    _compute_bin_I0(
                        detector,
                        energies,
                        weights,
                        thresholds,
                        b,
                        Float64(kVp),
                        Float64(I0_phys);
                        R = R,
                    ) *
                        bf *
                        bf,
                ),
            )
        end
        flat
    end
    eps_val = T(1.0e-10)
    for b in 1:n_bins
        let ba = bins[b], i0 = I0_use, off = Int32(b - 1) * ncnr, eps = eps_val, m = ncnr
            AK.foreachindex(ba) do idx
                ray = (Int32(idx - 1) % m) + Int32(1)
                ba[idx] = -log(max(ba[idx], eps) / i0[ray + off])
            end
        end
    end
    return bins
end

"""
    pcct_forward_project(mask, geom, detector; energies, weights, materials, kwargs...) -> EnergyResolvedSinogram

Perform polychromatic photon-counting CT forward projection with MC DRM spectral response.

This is the main PCCT forward projection API. It:
1. Computes energy-dependent μ-volumes from mask + materials (per energy)
2. Forward projects at each energy using Siddon ray-tracing
3. Applies spectral response matrix R(E,b) from MC DRM (counts per incident photon, absorption
   included) for realistic energy binning
4. Applies spatial binning if native dexel resolution is used
5. Returns energy-resolved sinograms in line-integral domain

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
    N_b = Σ_E R(E,b) × S(E) × exp(-∫μ(x,E)dl)

where:
- R(E,b) = MC DRM spectral response matrix entry (counts in bin b per incident photon of E)
- S(E) = normalized tube spectrum weight
- ∫μ(x,E)dl = line integral of energy-dependent attenuation

# Example

```julia
detector = BasisSimulator._build_pcct_detector(PCCTScanner(energy_thresholds = [
    20.0,
    35.0,
    55.0,
    70.0,
]))
energies, weights = load_spectrum(120)
materials = get_region_materials()

pcct_sino = pcct_forward_project(
    phantom.mask,
    geom,
    detector;
    energies = energies,
    weights = weights,
    materials = materials,
)

# Access energy bins
low_E_sino = pcct_sino.bins[1]  # 20-35 keV
high_E_sino = pcct_sino.bins[4] # 70+ keV
```

See also: [`PhotonCountingDetector`](@ref), [`EnergyResolvedSinogram`](@ref)
"""
function pcct_forward_project(mask::AbstractArray{
    <:Unsigned,
    3,
}, geom, detector::PhotonCountingDetector;
    energies::AbstractVector,
    weights::AbstractVector,
    materials::Vector = get_region_materials(),
    I0::Real = 1.0e6,
    # Workspace buffers (optional — allocate internally if not provided)
    ws_bins = nothing,
    ws_μ_volume = nothing,
    ws_sino_buf = nothing,
    ws_scratch = nothing,
    ws_thresholds_T = nothing,
    # Pre-computed spectral data (optional — compute internally if not provided)
    ws_R = nothing, # spectral response matrix (n_energies × n_bins)
    # per-ray air response [n_cols, n_rows, n_bins] (binned resolution) for normalization
    ws_I0 = nothing,
    # Pre-allocated μ lookup buffers for create_μ_volume!
    ws_μ_lut_cpu = nothing, # Vector{T}(n_regions) CPU
    ws_μ_lut_gpu = nothing, # similar(mask, T, n_regions) GPU-side
    ws_μ_table = nothing, # Matrix{T}(n_regions, n_energies) pre-computed μ table
    # Pre-allocated geometry arrays for siddon_forward_project!
    ws_source_positions = nothing,
    ws_detector_centers = nothing,
    ws_detector_u = nothing,
    ws_detector_v = nothing,
    # Override volume bounds for phantom physical extent
    volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
    # Native-resolution forward projection (spatial binning)
    # CTGeometry at native dexel resolution (nothing = no binning)
    native_geom = nothing,
    ws_native_bins = nothing, # per-bin sinograms at native res
    ws_native_sino_buf = nothing, # Siddon scratch at native res
    ws_native_scratch = nothing, # scratch at native res
    # μ volume (same as binned — phantom hasn't changed)
    ws_native_μ_volume = nothing,
    ws_native_source_positions = nothing,
    ws_native_detector_centers = nothing,
    ws_native_detector_u = nothing,
    ws_native_detector_v = nothing,
    # Tiled spectral projection buffers (optional — enables fused PCCT path)
    ws_μ_table_gpu = nothing, # GPU [n_regions, n_energies_padded]
    ws_W_matrix_gpu = nothing, # GPU [n_energies_padded, n_bins]
    ws_outputs_flat = nothing, # GPU [n_elements * n_bins]
    ws_native_outputs_flat = nothing, # GPU [native_n_elements * n_bins] (for bf>1)
    # GPU [n_cols, n_rows, n_energies_padded] heel × bowtie
    ws_source_spectral = nothing,
    # Ray tracer: :dd_fast (default/fastest general path) or :siddon (compatibility, aliases).
    projector::Symbol = :dd_fast,
    # Ignored kwargs for backward compat with callers that still pass them
    kwargs...
)
    _validate_projector(projector)
    T = Float32 # Use Float32 for GPU efficiency

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

    # MC-LUT detector response matrix R(E, bin).  Either passed in via
    # `ws_R` (workspace pre-compute) or built here from `compute_mc_drm`.
    R = ws_R !== nothing ? ws_R : compute_mc_drm(detector, kVp)

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
    if ws_μ_table_gpu !== nothing &&
            ws_W_matrix_gpu !== nothing &&
            ws_outputs_flat !== nothing
        @info "PCCT TILED PATH: K=16, n_energies=$n_energies, n_bins=$n_bins, mask=$(size(mask))" maxlog = 1

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
            ws_native_sino_buf !== nothing ?
                ws_native_sino_buf :
                similar(mask, T, proj_shape)
        else
            ws_sino_buf !== nothing ? ws_sino_buf : similar(mask, T, binned_shape)
        end

        # Per-tile subset copy + kernel call (same approach as EICT tiled path)
        n_regions = size(ws_μ_table_gpu, 1)
        has_src = ws_source_spectral !== nothing

        # ---------------------------------------------------------------------
        # SINGLE-PASS PATH (:dd_fast, any number of materials): the path-length
        # DD kernel handles the FULL padded energy range in one volume walk —
        # identical DD footprint/weights, no per-tile re-walk, no subset copies.
        # ---------------------------------------------------------------------
        if projector === :dd_fast
            @info "PCCT SINGLE-PASS PATH (:dd_fast path-length): n_energies=$n_energies, n_bins=$n_bins" maxlog = 1
            dd_fast_fused_spectral_project!(
                _pilot,
                _outputs_flat,
                Int32(n_bins),
                mask,
                _proj_geom,
                ws_μ_table_gpu,
                ws_W_matrix_gpu,
                Val(n_energies_padded),
                Int32(1);
                volume_extent = volume_extent,
                ws_source_positions = _ws_src,
                ws_detector_centers = _ws_det,
                ws_detector_u = _ws_u,
                ws_detector_v = _ws_v,
                ws_bowtie_spectral = (has_src ? ws_source_spectral : nothing),
            )
            @goto tiles_done
        end

        # :siddon holds one register per energy: K = 16 energies per volume walk
        μ_sub = similar(mask, T, n_regions, TILE_K)
        W_sub = similar(mask, T, TILE_K, n_bins)
        bt_sub = has_src ?
            similar(mask, T, size(_pilot, 1), size(_pilot, 2), TILE_K) :
            nothing

        for tile_idx in 1:n_tiles
            ts = (tile_idx - 1) * TILE_K + 1
            te = ts + TILE_K - 1

            copyto!(μ_sub, @view ws_μ_table_gpu[:, ts:te])
            copyto!(W_sub, @view ws_W_matrix_gpu[ts:te, :])

            if has_src
                copyto!(bt_sub, @view ws_source_spectral[:, :, ts:te])
                _project_fused_spectral!(
                    projector,
                    _pilot,
                    _outputs_flat,
                    Int32(n_bins),
                    mask,
                    _proj_geom,
                    μ_sub,
                    W_sub,
                    Val(TILE_K),
                    Int32(1);
                    volume_extent = volume_extent,
                    ws_source_positions = _ws_src,
                    ws_detector_centers = _ws_det,
                    ws_detector_u = _ws_u,
                    ws_detector_v = _ws_v,
                    ws_bowtie_spectral = bt_sub,
                )
            else
                _project_fused_spectral!(
                    projector,
                    _pilot,
                    _outputs_flat,
                    Int32(n_bins),
                    mask,
                    _proj_geom,
                    μ_sub,
                    W_sub,
                    Val(TILE_K),
                    Int32(1);
                    volume_extent = volume_extent,
                    ws_source_positions = _ws_src,
                    ws_detector_centers = _ws_det,
                    ws_detector_u = _ws_u,
                    ws_detector_v = _ws_v,
                )
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

        # Convert from photon counts to line-integral domain against each ray's own air
        # response: sino = -log(N / I₀[col, row, b])
        _normalize_bins_per_ray!(
            bins,
            ws_I0,
            detector,
            energies,
            weights,
            thresholds,
            kVp,
            I0,
            R,
            use_native ? bf : 1,
        )

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
    @info "PCCT SEQUENTIAL PATH: n_energies=$n_energies, n_bins=$n_bins" maxlog = 1

    # Native-res per-bin count sinograms (for spatial binning)
    proj_bins = if use_native
        nb = ws_native_bins !== nothing ?
            ws_native_bins :
            [similar(mask, T, proj_shape) for _ in 1:n_bins]
        for bin in nb
            fill!(bin, zero(T))
        end
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
        if w < 1.0e-12
            continue
        end

        # Create energy-dependent μ-volume from mask + materials
        create_μ_volume!(
            μ_volume,
            mask,
            materials,
            E_float;
            ws_μ_lut_cpu = ws_μ_lut_cpu,
            ws_μ_lut_gpu = ws_μ_lut_gpu,
            ws_μ_table = ws_μ_table,
            energy_idx = e_idx,
        )

        # Forward project at this energy (native or binned resolution)
        fill!(sino_buf, zero(T))
        _project_mono!(
            projector,
            sino_buf,
            μ_volume,
            proj_geom;
            ws_source_positions = _ws_src,
            ws_detector_centers = _ws_det,
            ws_detector_u = _ws_u,
            ws_detector_v = _ws_v,
            volume_extent = volume_extent,
        )

        # Weight by spectrum and quantum efficiency
        w_T = T(w)
        I0_T = T(I0)

        # Spectrum × MC-LUT DRM column for each bin: wt = I0 · w(E) · R(E, b).
        # The MC DRM is the single source of detector physics (charge sharing,
        # fluorescence escape, Compton escape, Fano broadening) — ideal-bin
        # fallback is intentionally absent.
        n_R = size(R, 1)
        r_idx = clamp(
            round(Int, (E_float - 1.0) / (Float64(kVp) - 1.0) * (n_R - 1)) + 1,
            1,
            n_R,
        )

        for b in 1:n_bins
            R_val = T(R[r_idx, b])
            if R_val < T(1.0e-10)
                continue
            end
            if ws_source_spectral === nothing
                let wt = I0_T * w_T * R_val, ba = accum_bins[b]
                    AK.foreachindex(sino_buf) do idx
                        ba[idx] += wt * exp(-sino_buf[idx])
                    end
                end
            else
                # the source transmission at this ray and energy (bowtie × heel), as the fused path
                let wt = I0_T * w_T * R_val,
                    ba = accum_bins[b],
                    bt = ws_source_spectral,
                    m = Int32(size(sino_buf, 1) * size(sino_buf, 2)),
                    off = Int32(e_idx - 1) * Int32(size(sino_buf, 1) * size(sino_buf, 2))
                    AK.foreachindex(sino_buf) do idx
                        ray = (Int32(idx - 1) % m) + Int32(1)
                        ba[idx] += wt * bt[ray + off] * exp(-sino_buf[idx])
                    end
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

    # Convert from photon counts to line-integral domain against each ray's own air response
    _normalize_bins_per_ray!(
        bins,
        ws_I0,
        detector,
        energies,
        weights,
        thresholds,
        kVp,
        I0,
        R,
        use_native ? bf : 1,
    )

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
    _compute_bin_I0(detector, energies, weights, thresholds, bin_idx, kVp, I0; R) -> Float64

Reference photon count I₀ for one energy bin (the unattenuated count expected
in `bin_idx` — the denominator for the `-log(N / I0)` step in
`pcct_forward_project`).

`R` is the MC-LUT detector response matrix (`compute_mc_drm`).  Mapping
each spectrum energy onto its R-row and summing `I0 · w(E) · R(E, bin_idx)` (R per incident photon: absorption included)
keeps this consistent with the forward-projection accumulator and guarantees
the `-log(N / I0)` normalization produces well-defined line integrals.
"""
function _compute_bin_I0(detector, energies, weights, thresholds, bin_idx, kVp, I0; R)
    n_energies = length(energies)
    n_R = size(R, 1)
    I0_bin = 0.0
    for e_idx in 1:n_energies
        E_float = Float64(energies[e_idx])
        w = Float64(weights[e_idx])
        if w < 1.0e-12
            continue
        end
        # Map spectrum energy to DRM grid index
        r_idx = clamp(round(Int, (E_float - 1.0) / (kVp - 1.0) * (n_R - 1)) + 1, 1, n_R)
        R_val = R[r_idx, bin_idx]
        I0_bin += I0 * w * R_val
    end
    return max(I0_bin, 1.0) # Avoid division by zero
end

# =============================================================================
# PCCT Noise Model (PCCT-NOISE-DECOMP)
# =============================================================================

# log(k!) for k = 0…20 exactly (a tuple, so device code reads it too); Stirling–de Moivre series
# beyond (rel. error < 1e-10 at k = 21, decreasing with k — far below rejection-test resolution).
const _LOG_FACTORIAL_TABLE = Tuple(
    Float64[sum(log(j) for j in 1:k; init = 0.0) for k in 0:20]
)

function _log_factorial(k::Int)
    k <= 20 && return @inbounds _LOG_FACTORIAL_TABLE[k + 1]
    invk = 1.0 / k
    return (k + 0.5) * log(k) -
        k +
        0.9189385332046728 +
        invk * (1.0 / 12.0 - invk * invk * (1.0 / 360.0 - invk * invk / 1260.0))
end

# log P(k; λ) = k log λ − λ − log k!. Past the table it is Stirling's series written in
# t = (λ − k) / k, k (log1p(t) − t) − ½ log k − ½ log 2π − …, so the large terms cancel exactly
# and Float32 keeps it to about |λ − k|·eps.
@inline function _log_poisson_pmf(k::T, λ::T) where {T <: AbstractFloat}
    k <= 20 && return k * log(λ) -
        λ -
        T(@inbounds _LOG_FACTORIAL_TABLE[unsafe_trunc(Int, k) + 1])
    invk = inv(k)
    t = (λ - k) * invk
    return k * (log1p(t) - t) -
        log(k) / 2 -
        T(0.9189385332046728) -
        invk * (T(1 / 12) - invk * invk * (T(1 / 360) - invk * invk / 1260))
end

# A counter-based stream of uniform variates for device code: its `n`-th variate is a hash of
# `(key, n)`, so every draw is a pure function of its key and no state is shared between threads.
struct _DrawStream
    key::UInt64
    n::UInt64
end

@inline function _mix64(x::UInt64)
    # splitmix64's finaliser
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    return x ⊻ (x >> 31)
end

# Mix `x` into the hash `h`. `x` is spread over all 64 bits first (an odd multiplier), so small
# seeds and small classes never collide (`seed ⊻ class` would: seed 15's class x is seed 16's y).
@inline _combine(h::UInt64, x) = _mix64(
    h ⊻ ((x % UInt64) * 0xd1b54a32d192ed03 + 0x9e3779b97f4a7c15),
)

# The stream of one draw: a seed, a class, and the dexel `(c, r)` of view `v`.
@inline _DrawStream(seed::UInt64, class, c, r, v) = _DrawStream(
    _combine(_combine(_combine(_combine(seed, class), v), r), c),
    UInt64(0),
)

@inline _uniform(rng::AbstractRNG, ::Type{T}) where {T} = (rand(rng, T), rng)
@inline function _uniform(s::_DrawStream, ::Type{Float32})
    x = _mix64(s.key + s.n * 0x9e3779b97f4a7c15)
    # (0, 1)
    return ((Float32(x >> 40) + 0.5f0) * 5.9604645f-8, _DrawStream(s.key, s.n + 1))
end
@inline function _uniform(s::_DrawStream, ::Type{Float64})
    x = _mix64(s.key + s.n * 0x9e3779b97f4a7c15)
    return ((Float64(x >> 11) + 0.5) * 1.1102230246251565e-16, _DrawStream(s.key, s.n + 1))
end

"""
    _poisson_sample(rng, λ) -> Int

Exact Poisson sample at any λ: sequential inversion below λ = 10, Hörmann's
PTRS transformed rejection above (W. Hörmann, "The transformed rejection
method for generating Poisson random variables", Insurance: Mathematics and
Economics 12, 39–45, 1993).  No Gaussian approximation at any count level.
"""
_poisson_sample(rng, λ::Real) = Int(first(_poisson_draw(rng, Float64(λ))))

# The sampler, in λ's precision (Float32 on a device): the count, as a number of that type, and
# the stream past the variates it used. `rng` is an `AbstractRNG` or a `_DrawStream`.
@inline function _poisson_draw(rng, λ::T) where {T <: AbstractFloat}
    if λ < T(1.0e-10)
        return (zero(T), rng)
    elseif λ < 10
        # Inversion by sequential search (Knuth): exact, O(λ).
        L = exp(-λ)
        k = zero(T)
        p = one(T)
        while true
            u, rng = _uniform(rng, T)
            p *= u
            p < L && return (k, rng)
            k += one(T)
        end
    end
    # PTRS: O(1) exact transformed-rejection sampling for λ ≥ 10.
    b = T(0.931) + T(2.53) * sqrt(λ)
    a = T(-0.059) + T(0.02483) * b
    inv_alpha = T(1.1239) + T(1.1328) / (b - T(3.4))
    v_r = T(0.9277) - T(3.6224) / (b - 2)
    while true
        u, rng = _uniform(rng, T)
        V, rng = _uniform(rng, T)
        U = u - T(0.5)
        us = T(0.5) - abs(U)
        k = floor((2 * a / us + b) * U + λ + T(0.43))
        us >= T(0.07) && V <= v_r && return (k, rng)
        (k < 0 || (us < T(0.013) && V > us)) && continue
        log(V) + log(inv_alpha) - log(a / (us * us) + b) <=
            _log_poisson_pmf(k, λ) && return (k, rng)
    end
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
props.density_g_cm3 # 5.85
props.k_edges_keV   # [26.7, 31.8]
```
"""
function get_detector_material_properties(material::DetectorMaterialPCCT)
    if material == CDTE_MATERIAL
        # Cadmium Telluride — primary PCCT detector material (Siemens NAEOTOM Alpha)
        # Atomic weights: Cd = 112.411, Te = 127.60 → total = 240.011
        return (
            elements = [:Cd, :Te],
            atomic_numbers = [48, 52],
            mass_fractions = [112.411 / 240.011, 127.6 / 240.011], # [0.4683, 0.5317]
            density_g_cm3 = 5.85,
            # K-edges and fluorescence (NIST XCOM, Koch-Mehrin 2020 Table 1)
            k_edges_keV = [26.711, 31.814], # Cd K-edge, Te K-edge (NIST)
            k_alpha_keV = [23.2, 27.5],     # Cd K-α, Te K-α fluorescence
            k_yields = [0.84, 0.88],        # Fluorescence yields ω_K (Bambynek 1972)
            # Koch-Mehrin 2020: Cd Kα ~100μm, Te Kα ~60μm
            k_fluorescence_range_mm = [0.1, 0.06],
            # Charge transport (Koch-Mehrin 2020 + Owens 2012)
            # cm²/V — μ_e=1100×τ_e=3μs (Koch-Mehrin 2020)
            mu_e_tau_e = 3.3e-3,
            # cm²/V — μ_h=100×τ_h=2μs (Koch-Mehrin 2020)
            mu_h_tau_h = 2.0e-4,
            bias_voltage_V = 800.0, # Typical operating voltage
            # Intrinsic energy resolution (Koch-Mehrin 2020, Redus 2009)
            pair_creation_energy_eV = 4.3, # eV per e-h pair (Koch-Mehrin 2020)
            # Intrinsic variance reduction (Redus 2009)
            fano_factor = 0.1,
            # Physics-based: ~13μm from ODE (Koch-Mehrin 2020)
            charge_cloud_sigma_mm = 0.013,
            # System-level measured resolution (includes electronics)
            energy_resolution_fwhm_keV = 10.0, # Typical for NAEOTOM at 60 keV
        )
    elseif material == CZT_MATERIAL
        # Cadmium Zinc Telluride (Cd₀.₉Zn₀.₁Te)
        # Zn substitutes ~10% of Cd → slightly different properties
        # Atomic weights: Cd=112.411, Zn=65.38, Te=127.60
        # CZT formula: Cd₀.₉Zn₀.₁Te → effective mass per formula unit
        # = 0.9×112.411 + 0.1×65.38 + 127.60 = 234.307
        eff_mass = 0.9 * 112.411 + 0.1 * 65.38 + 127.6 # 234.307
        return (
            elements = [:Cd, :Zn, :Te],
            atomic_numbers = [48, 30, 52],
            mass_fractions = [
                0.9 * 112.411 / eff_mass,
                0.1 * 65.38 / eff_mass,
                127.6 / eff_mass,
            ],
            density_g_cm3 = 5.78, # Slightly lower than CdTe
            # K-edges and fluorescence
            k_edges_keV = [26.7, 9.66, 31.8],           # Cd, Zn, Te K-edges
            k_alpha_keV = [23.2, 8.63, 27.4],           # Cd, Zn, Te K-α lines
            k_yields = [0.84, 0.47, 0.87],              # ω_K values (Zn lower yield)
            k_fluorescence_range_mm = [0.12, 0.5, 0.1], # Zn K-α has long range (low-E)
            # Charge transport (Pennicard et al 2014)
            mu_e_tau_e = 1.0e-2,    # cm²/V (better than CdTe)
            mu_h_tau_h = 5.0e-5,    # cm²/V (similar hole trapping)
            bias_voltage_V = 600.0, # Lower voltage needed
            # Intrinsic energy resolution
            pair_creation_energy_eV = 4.64,   # eV per e-h pair (Owens 2004)
            fano_factor = 0.089,              # Better than CdTe (Pennicard 2014)
            charge_cloud_sigma_mm = 0.045,    # Slightly larger cloud
            energy_resolution_fwhm_keV = 8.0, # CZT typically better resolution
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
            k_alpha_keV = [1.74],              # Si K-α (irrelevant at CT energies)
            k_yields = [0.05],                 # Very low fluorescence yield for low-Z
            k_fluorescence_range_mm = [0.001], # Absorbed immediately
            # Charge transport (excellent! — Persson 2014)
            mu_e_tau_e = 1.0,       # cm²/V (near-perfect collection)
            mu_h_tau_h = 0.5,       # cm²/V (also excellent)
            bias_voltage_V = 200.0, # Low voltage sufficient
            # Intrinsic energy resolution (best of all three)
            pair_creation_energy_eV = 3.62,   # eV per e-h pair (lowest)
            fano_factor = 0.115,              # Similar to CdTe
            charge_cloud_sigma_mm = 0.015,    # Very small cloud (light material)
            energy_resolution_fwhm_keV = 2.5, # Excellent energy resolution
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
μ = get_detector_material_attenuation(CDTE_MATERIAL, 60.0) # ~33 cm⁻¹
μ = get_detector_material_attenuation(SI_MATERIAL, 60.0)   # ~0.46 cm⁻¹
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
η_cdte = quantum_efficiency(CDTE_MATERIAL, 1.6, 60.0) # ≈ 0.995
η_si = quantum_efficiency(SI_MATERIAL, 1.6, 60.0)     # ≈ 0.071
```
"""
function quantum_efficiency(
    material::DetectorMaterialPCCT,
    thickness_mm::Real,
    energy_keV::Real,
)
    μ = get_detector_material_attenuation(material, energy_keV) # cm⁻¹
    thickness_cm = Float64(thickness_mm) / 10.0
    return 1.0 - exp(-μ * thickness_cm)
end

"""
    quantum_efficiency_vector(material::DetectorMaterialPCCT, thickness_mm::Real,
                              energies::AbstractVector) -> Vector{Float64}

Compute quantum efficiency η(E) for a vector of energies (batch operation).

More efficient than calling `quantum_efficiency` in a loop for spectrum processing.
"""
function quantum_efficiency_vector(
    material::DetectorMaterialPCCT,
    thickness_mm::Real,
    energies::AbstractVector,
)
    return [quantum_efficiency(material, thickness_mm, E) for E in energies]
end

# =============================================================================
# Exports
# =============================================================================

export DetectorMaterialPCCT, CDTE_MATERIAL, CZT_MATERIAL, SI_MATERIAL
export PhotonCountingDetector
export EnergyResolvedSinogram, n_energy_bins
export pcct_forward_project
export get_detector_material_properties, get_detector_material_attenuation
export quantum_efficiency, quantum_efficiency_vector
export spatial_bin!
