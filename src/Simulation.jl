"""
    Simulation.jl

Forward CT simulation: complete imaging chain from phantom to sinogram.

# Overview

This module orchestrates the complete CT forward model:
1. **X-ray Source** - Polychromatic spectrum generation
2. **Phantom** - 3D material and density distribution
3. **Ray Tracing** - Geometric path calculation through phantom
4. **Attenuation** - Energy-dependent photon absorption
5. **Detector** - Energy integration and readout
6. **Sinogram** - Projection data (input to reconstruction)

# Physics Pipeline

```
X-ray Spectrum → [Phantom] → Polychromatic Attenuation → Detector → Sinogram
     N₀(E)          μ(E,x)        I(E) = N₀(E)·exp(-∫μ dx)      I_total    [r,c,θ]
```

**Key Equation - Polychromatic Beer-Lambert Law:**
```
I_total = Σₑ N₀(E) · exp(-∫ μ(E,x) dx) · E · η(E)
```

Where:
- N₀(E) = incident photon fluence at energy E
- μ(E,x) = linear attenuation coefficient along ray path
- E = photon energy
- η(E) = detector quantum efficiency

# Design Philosophy

**Pure Functional for Reactant/Enzyme:**
- All functions are **pure** (no side effects)
- No closures or dynamic dispatch in hot loops
- Pre-computed geometries (avoid sin/cos at runtime)
- Statically-typed operations

**Parallel-First:**
- Threads.@threads over projection angles
- Each angle is independent (embarrassingly parallel)
- GPU-ready via Reactant compilation

# Usage Example

```julia
using BasisSimulator
import XrayAttenuation as XA

# 1. Create phantom
phantom = create_gammex_472(resolution_mm=2.0, z_coverage_mm=40.0)

# 2. Define scanner
protocol = ScanProtocol(kVp=120.0, mAs=200.0, scan_fov_mm=400.0, num_projections=360)
geometry = create_aquilion_one(protocol=protocol)

# 3. Generate spectrum
spectrum = generate_spectrum(kVp=120.0, mAs=200.0)

# 4. Run simulation
sinogram = simulate_ct_scan(
    phantom = phantom,
    geometry = geometry,
    spectrum = spectrum
)

# 5. Reconstruct
volume = reconstruct_fdk(sinogram, geometry)
```

# Performance Targets

| Operation | Target | Status |
|-----------|--------|--------|
| Forward projection (360 views, 320×800) | < 10 s | Testing |
| Reactant compilation | < 30 s | Testing |
| Memory (512³ phantom) | < 16 GB | Achieved |

# References

**Forward Modeling:**
- De Man, B., & Basu, S. (2004). "Distance-driven projection and backprojection."
  IEEE Trans. Med. Imaging, 23(9), 1176-1185.
- Joseph, P. M. (1982). "An improved algorithm for reprojecting rays."
  IEEE Trans. Med. Imaging, 1(3), 192-196.

**Polychromatic Imaging:**
- Alvarez, R. E., & Macovski, A. (1976). "Energy-selective reconstructions."
  Physics in Medicine & Biology, 21(5), 733.

# Author

Dale Black, MolloiLab
Created: January 2026
"""

import XrayAttenuation as XA

# ==============================================================================
# Main Simulation Function
# ==============================================================================

"""
    simulate_ct_scan(;
        phantom::PhantomData,
        geometry::CTGeometry,
        spectrum::XRaySpectrum,
        verbose::Bool = true
    )

Run complete forward CT simulation: phantom → sinogram.

# Arguments

- `phantom::PhantomData` - Digital phantom with materials and densities
- `geometry::CTGeometry` - Scanner geometry (pre-computed trajectories)
- `spectrum::XRaySpectrum` - X-ray source spectrum
- `verbose::Bool` - Print progress information (default: true)

# Returns

- `sinogram::Array{Float64, 3}` - Projection data [n_rows, n_cols, n_angles]

# Physics Model

For each detector pixel (r, c) at projection angle θ:

1. **Ray Definition**: Source → Detector pixel
2. **Ray Tracing**: Compute path lengths through each material
3. **Polychromatic Attenuation**:
   ```
   I_pixel = Σₑ N₀(E) · exp(-Σₘ μₘ(E)·Lₘ) · E
   ```
   where:
   - N₀(E) = source photon fluence at energy E
   - μₘ(E) = attenuation coefficient for material m at energy E
   - Lₘ = path length through material m

4. **Energy Integration**: Convert to detector signal

# Example

```julia
# Basic simulation
sinogram = simulate_ct_scan(
    phantom = phantom,
    geometry = geo,
    spectrum = spec
)

# Check output
@assert size(sinogram) == (geo.n_rows, geo.n_cols, length(geo.angles))
@assert all(isfinite, sinogram)
```

# Performance Notes

- Parallelized over projection angles (Threads.@threads)
- Memory-efficient: processes one angle at a time
- Reactant-compatible for GPU acceleration

# Implementation Details

**Sub-pixel Sampling:**
For improved accuracy, each detector pixel is sampled at 2×2 sub-locations
to reduce aliasing artifacts. Disable for faster approximate simulations.

**Material Library:**
Attenuation coefficients are queried from XrayAttenuation.jl on-the-fly.
For high-performance simulations, consider pre-computing a material
attenuation matrix (similar to ct_simulator_final.jl).
"""
function simulate_ct_scan(;
        phantom::PhantomData,
        geometry::CTGeometry,
        spectrum::XRaySpectrum,
        verbose::Bool = true
    )

    if verbose
        @info """
        🚀 CT Forward Simulation
        ========================
        Phantom: $(phantom.name)
        Matrix: $(phantom.grid.nx) × $(phantom.grid.ny) × $(phantom.grid.nz)
        Scanner: Canon Aquilion ONE
        Detector: $(geometry.n_rows) × $(geometry.n_cols)
        Projections: $(length(geometry.angles))
        Spectrum: $(spectrum.kVp) kVp, $(spectrum.mAs) mAs
        Energies: $(minimum(spectrum.energies)) - $(maximum(spectrum.energies)) keV
        ========================
        """
    end

    # =========================================================================
    # 1. Prepare GridMeta for Ray Tracing
    # =========================================================================

    grid_meta = GridMeta(
        nx = phantom.grid.nx,
        ny = phantom.grid.ny,
        nz = phantom.grid.nz,
        fov_xy = phantom.grid.fov_xy_cm,
        fov_z = phantom.grid.fov_z_cm
    )

    # =========================================================================
    # 2. Build Material ID Lookup Table
    # =========================================================================

    # Map material IDs to indices for attenuation lookup
    unique_mat_ids = sort(unique(phantom.material_ids))
    n_materials = length(unique_mat_ids)

    id_to_idx = Dict{UInt8, Int}()
    idx_to_id = Dict{Int, UInt8}()
    for (idx, mat_id) in enumerate(unique_mat_ids)
        id_to_idx[mat_id] = idx
        idx_to_id[idx] = mat_id
    end

    # Create lookup table for ray tracer (maps phantom ID → material index)
    id_lut = zeros(Int, 256)
    for (mat_id, idx) in id_to_idx
        id_lut[Int(mat_id) + 1] = idx  # +1 for 1-based indexing
    end

    if verbose
        @info "📊 Material library: $n_materials unique materials"
    end

    # =========================================================================
    # 3. Pre-compute Attenuation Matrix [n_materials × n_energies]
    # =========================================================================

    if verbose
        @info "🔬 Pre-computing attenuation coefficients..."
    end

    n_energies = length(spectrum.energies)
    μ_matrix = zeros(Float64, n_materials, n_energies)

    for (idx, mat_id) in idx_to_id
        mat_symbol = phantom.id_to_material[mat_id]

        # Skip air (ID=0)
        if mat_symbol == :air
            continue
        end

        # Map material symbol to XA material
        # Uses custom materials (Gammex inserts) if defined, otherwise XA.Materials
        xa_material = try
            get_material(mat_symbol)
        catch e
            # Fallback: use water if material not found
            @warn "Material $mat_symbol not found in custom materials or XA.Materials, using water as fallback"
            @warn "Error: $e"
            XA.Materials.water
        end

        # Compute attenuation for all energies
        # NOTE: Use MASS attenuation because ray tracer returns density-weighted paths
        # path_lengths [g/cm²] × μ_mass [cm²/g] = dimensionless attenuation
        for (e_idx, E) in enumerate(spectrum.energies)
            μ_matrix[idx, e_idx] = get_mass_attenuation(xa_material, E)
        end
    end

    if verbose
        @info "✅ Attenuation matrix ready: $(n_materials) × $(n_energies)"
    end

    # =========================================================================
    # 4. Compute Detector Efficiency
    # =========================================================================

    if verbose
        @info "🔬 Computing detector quantum efficiency..."
    end

    # Canon Aquilion ONE uses CsI scintillator, ~0.5mm thick
    detector_material = XA.Materials.csi
    detector_thickness_mm = 0.5

    detector_efficiency = compute_detector_efficiency(
        spectrum.energies,
        detector_material,
        detector_thickness_mm
    )

    if verbose
        @info "✅ Detector efficiency: $(round(mean(detector_efficiency), digits=3)) (mean)"
    end

    # =========================================================================
    # 5. Allocate Sinogram
    # =========================================================================

    sinogram = zeros(Float64, geometry.n_rows, geometry.n_cols, length(geometry.angles))

    # =========================================================================
    # 6. Forward Projection Loop (Polychromatic Ray Tracing)
    # =========================================================================

    if verbose
        @info "📷 Acquiring projections..."
    end

    # Sub-pixel sampling for anti-aliasing
    n_sub = 2  # 2×2 subpixel sampling
    sub_width = 1.0 / n_sub
    sub_offsets = (collect(1:n_sub) .* sub_width) .- (sub_width/2 + 0.5)

    # Parallelize over projection angles
    Threads.@threads for angle_idx in 1:length(geometry.angles)

        # Get geometry for this projection
        src_pos = geometry.source_positions[:, angle_idx]
        det_center = geometry.det_centers[:, angle_idx]
        u_vec = geometry.det_u_vecs[:, angle_idx]
        v_vec = geometry.det_v_vecs[:, angle_idx]

        # Loop over detector pixels
        for row in 1:geometry.n_rows
            for col in 1:geometry.n_cols

                pixel_energy_sum = 0.0

                # Sub-pixel sampling (2×2 grid)
                for sub_row in 1:n_sub
                    for sub_col in 1:n_sub

                        # Detector pixel position (with sub-pixel offset)
                        u_offset = (col - geometry.n_cols/2 - 0.5 + sub_offsets[sub_col]) * geometry.pixel_width_cm
                        v_offset = (row - geometry.n_rows/2 - 0.5 + sub_offsets[sub_row]) * geometry.pixel_height_cm

                        detector_pos = det_center .+ (u_offset .* u_vec) .+ (v_offset .* v_vec)

                        # =====================================================
                        # Ray Trace Through Phantom
                        # =====================================================

                        path_lengths = trace_ray_material_paths(
                            grid_meta,
                            phantom.material_ids,
                            phantom.densities,
                            id_lut,
                            n_materials,
                            src_pos[1], src_pos[2], src_pos[3],
                            detector_pos[1], detector_pos[2], detector_pos[3]
                        )

                        # =====================================================
                        # Polychromatic Attenuation
                        # =====================================================

                        # Compute attenuation for each energy bin
                        transmitted_energy = 0.0

                        for e_idx in 1:n_energies
                            E = spectrum.energies[e_idx]
                            N0 = spectrum.photons[e_idx]

                            # Total attenuation at this energy
                            total_atten = 0.0
                            for m_idx in 1:n_materials
                                if path_lengths[m_idx] > 0
                                    total_atten += μ_matrix[m_idx, e_idx] * path_lengths[m_idx]
                                end
                            end

                            # Beer-Lambert law: I(E) = I₀(E) · exp(-μ·L)
                            transmission = exp(-total_atten)

                            # Energy-weighted transmission with detector efficiency
                            # I(E) = N₀(E) × exp(-μL) × E × η(E)
                            # This matches the old working implementation
                            transmitted_energy += N0 * transmission * E * detector_efficiency[e_idx]
                        end

                        pixel_energy_sum += transmitted_energy
                    end
                end

                # Average over sub-pixels
                avg_energy = pixel_energy_sum / (n_sub * n_sub)

                # Store in sinogram
                sinogram[row, col, angle_idx] = avg_energy
            end
        end

        # Progress reporting
        if verbose && angle_idx % 60 == 0
            pct = round(100 * angle_idx / length(geometry.angles), digits=1)
            @info "  Progress: $(angle_idx)/$(length(geometry.angles)) ($(pct)%)"
        end
    end

    if verbose
        @info "✅ Projection acquisition complete!"
        @info "📊 Sinogram stats:"
        @info "  Min: $(minimum(sinogram))"
        @info "  Max: $(maximum(sinogram))"
        @info "  Mean: $(mean(sinogram))"
    end

    return sinogram
end

# ==============================================================================
# Utility Functions
# ==============================================================================

"""
    convert_to_attenuation(sinogram::Array{Float64, 3}, I0::Float64)::Array{Float64, 3}

Convert transmitted energy to line integrals (attenuation).

# Formula

```
p = -log(I / I₀)
```

where:
- I = transmitted intensity
- I₀ = incident intensity (air scan)
- p = line integral (Radon transform)

# Arguments

- `sinogram` - Transmitted energy values
- `I0` - Reference air scan intensity

# Returns

- Line integrals suitable for reconstruction
"""
function convert_to_attenuation(sinogram::Array{Float64, 3}, I0::Float64)::Array{Float64, 3}
    # Avoid division by zero
    I0_safe = max(I0, 1e-10)

    # -log(I/I0) = log(I0) - log(I)
    return @. -log(max(sinogram, 1e-10) / I0_safe)
end

"""
    estimate_air_scan(spectrum::XRaySpectrum)::Float64

Estimate incident intensity (I₀) assuming no phantom.

This is the energy-weighted photon fluence that would reach the detector,
including detector quantum efficiency.

# Arguments
- `spectrum::XRaySpectrum` - X-ray source spectrum

# Optional Arguments
- `detector_material` - Scintillator material (default: CsI)
- `detector_thickness_mm` - Detector thickness (default: 0.5 mm)

# Returns
- `Float64` - Reference intensity for air scan
"""
function estimate_air_scan(
        spectrum::XRaySpectrum;
        detector_material::XA.Material = XA.Materials.csi,
        detector_thickness_mm::Float64 = 0.5
    )::Float64

    # Compute detector efficiency
    efficiency = compute_detector_efficiency(
        spectrum.energies,
        detector_material,
        detector_thickness_mm
    )

    # Sum of energy-weighted photon fluence with detector response
    # I₀ = Σ [N₀(E) × E × η(E)]
    return sum(spectrum.photons .* spectrum.energies .* efficiency)
end

# ==============================================================================
# Exports
# ==============================================================================

export simulate_ct_scan
export convert_to_attenuation, estimate_air_scan
