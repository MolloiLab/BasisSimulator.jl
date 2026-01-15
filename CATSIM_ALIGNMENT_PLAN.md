# Plan: Aligning BasisSimulator.jl with CatSim/XCIST Physics

## Status: ✅ PHASE 1 COMPLETE (2026-01-15)

Phase 1 changes have been implemented:
- Air scan without noise (CatSim-exact)
- Low signal correction (replace negatives with smoothed neighbors)
- Full signal chain integrated into `forward_project()` as kwargs

## Executive Summary

After analyzing the CatSim/XCIST repository (`/Users/daleblack/Documents/dev/main`), I've identified several key differences in how we handle the CT signal chain. This plan outlines the changes needed to match CatSim's physics accurately.

---

## Key Findings from CatSim

### 1. Air Scan Acquisition (CatSim)
**Location**: `gecatsim/pyfiles/Prep.py`

CatSim acquires air scans with **NO noise**:
```python
# Save current noise settings
savedQuantumNoise = cfg.sim.enableQuantumNoise
savedEletronicNoise = cfg.sim.eNoise

# Disable noise for air scan
cfg.sim.enableQuantumNoise = 0
cfg.sim.eNoise = 0

# Acquire air scan...

# Restore noise settings
cfg.sim.enableQuantumNoise = savedQuantumNoise
cfg.sim.eNoise = savedEletronicNoise
```

**Rationale**: Air scans in real CT are averaged over many acquisitions, effectively removing noise. CatSim simulates this by disabling noise entirely.

### 2. Three-Scan Calibration (CatSim)
**Location**: `gecatsim/pyfiles/Prep.py`

CatSim uses THREE scans for calibration:
1. **Phantom scan**: Object scan with all noise
2. **Air scan**: Reference scan (no noise)
3. **Offset scan**: Dark current/electronic offset (no X-rays, no noise)

**Calibration formula**:
```python
prep = (phantomScan - offsetScan) / (airscan - offsetScan)
```

**Our current approach**: We only use phantom and air scans:
```julia
normalized = intensity ./ air_scan
```

### 3. DAS Noise Model (CatSim)
**Location**: `gecatsim/pyfiles/Detection.py`

CatSim uses **electron-based** noise units:
```python
eNoise = 3500           # Electronic noise in electrons
eResponse = 17.0        # Gain: electrons per keV detected
```

The noise is added in **electron space**, then converted to signal:
```python
signal_electrons = detected_photons * energy * eResponse
noise_electrons = gaussian(0, eNoise)
total_electrons = signal_electrons + noise_electrons
output_signal = total_electrons / (some_reference)
```

**Our current approach**: We add noise in normalized intensity space with arbitrary sigma (0.001), which doesn't map to physical units.

### 4. Low Signal Correction (CatSim)
**Location**: `gecatsim/pyfiles/Prep.py`

CatSim handles negative values from noise with sophisticated correction:
```python
# Replace negative values with convolved (smoothed) neighbor values
negIdx = (prep <= 0)
if negIdx.any():
    prep_convolved = convolve2d(prep, kernel, 'same')
    prep[negIdx] = prep_convolved[negIdx]
```

Then uses a C library function `negative_log` that handles edge cases.

**Our current approach**: Simple clamping:
```julia
sinogram = clamp.(sinogram, -0.1, 10.0)
```

### 5. Quantum Noise (CatSim)
**Location**: `gecatsim/reconstruction/pyfiles/randpf.c`

CatSim uses Poisson sampling for quantum noise via a C library. For high counts, it approximates with Gaussian.

**Our approach**: Our physics pipeline already does Poisson-based quantum noise, which is correct.

### 6. Prep Clamping (CatSim)
**Location**: `gecatsim/pyfiles/Prep.py`

Optional maximum prep value:
```python
if hasattr(cfg.protocol, 'maxPrep') and cfg.protocol.maxPrep:
    prep = np.minimum(prep, cfg.protocol.maxPrep)
```

---

## Required Changes to BasisSimulator.jl

### Change 1: Air Scan Without Noise

**Current behavior**: We apply DAS noise to both phantom and air scans, then try to divide them out.

**New behavior**: Air scan should have NO noise applied (simulating averaged reference).

**Files to modify**:
- `src/signal_chain/das_model.jl` - Add flag for noise-free mode
- Example scripts - Update air scan simulation

**Implementation**:
```julia
# Option A: Add flag to DAS model
apply_das_model!(data, das; add_noise=true)

# Option B: Separate air scan function
simulate_air_scan(size, das, heel, geom)  # Returns noise-free air reference
```

### Change 2: Offset Scan Support

**Current behavior**: No offset scan concept.

**New behavior**: Support dark current / electronic offset.

**Implementation**:
```julia
struct OffsetScan
    value::Float32  # Constant offset (or could be per-detector)
end

# Calibration becomes:
prep = (phantom - offset) ./ (air - offset)
```

**Files to modify**:
- New file: `src/signal_chain/offset_scan.jl`
- `src/signal_chain/calibration.jl` - Add calibration function

### Change 3: Physical DAS Noise Units

**Current behavior**:
```julia
das = default_das_model(electronic_noise_sigma = 0.001)  # Arbitrary units
```

**New behavior**:
```julia
das = default_das_model(
    electronic_noise_electrons = 3500.0,  # σ in electrons
    electron_gain = 17.0,                  # electrons per keV
    reference_flux = 1e6,                  # Reference photon flux for normalization
)
```

**Noise calculation**:
```julia
# Convert intensity to electron counts
signal_electrons = intensity * reference_flux * mean_energy * electron_gain
# Add noise in electron space
noise_electrons = randn() * electronic_noise_electrons
total_electrons = signal_electrons + noise_electrons
# Convert back to normalized intensity
output = total_electrons / (reference_flux * mean_energy * electron_gain)
```

**Files to modify**:
- `src/signal_chain/das_model.jl`
- `src/signal_chain/types.jl` - Update DASModel struct

### Change 4: Low Signal Correction

**Current behavior**: Simple clamping `clamp(sinogram, -0.1, 10.0)`

**New behavior**: Replace negative/zero values with smoothed neighbors before log transform.

**Implementation**:
```julia
function low_signal_correction!(prep::AbstractArray)
    # Find problematic values
    bad_idx = prep .<= 0

    if any(bad_idx)
        # 2D convolution for smoothing
        kernel = [1 2 1; 2 4 2; 1 2 1] ./ 16.0f0
        prep_smoothed = imfilter(prep, kernel)

        # Replace bad values with smoothed neighbors
        prep[bad_idx] .= prep_smoothed[bad_idx]
    end

    return prep
end
```

**Files to modify**:
- New file: `src/signal_chain/low_signal_correction.jl`
- Or add to existing calibration code

### Change 5: Unified Calibration Function

Create a high-level calibration function that encapsulates the CatSim workflow:

```julia
function calibrate_sinogram(
    phantom_intensity::AbstractArray,
    air_intensity::AbstractArray,
    offset::Union{AbstractArray, Real} = 0.0f0;
    low_signal_correction::Bool = true,
    max_prep::Union{Nothing, Real} = nothing
)
    # Apply offset correction
    prep = (phantom_intensity .- offset) ./ max.(air_intensity .- offset, eps(Float32))

    # Low signal correction
    if low_signal_correction
        low_signal_correction!(prep)
    end

    # Log transform
    sinogram = -log.(max.(prep, eps(Float32)))

    # Optional clamping
    if max_prep !== nothing
        sinogram = min.(sinogram, max_prep)
    end

    return sinogram
end
```

---

## Implementation Priority

### Phase 1: Critical (Highest Impact) ✅ COMPLETE
1. **Air scan without noise** ✅ - Most impactful fix for HU accuracy
2. **Low signal correction** ✅ - Prevents numerical issues from noise

### Phase 2: Important (Better Physics)
3. **Physical DAS noise units** - More realistic noise model (electrons vs arbitrary)
4. **Unified calibration function** ✅ - Clean API (integrated into forward_project)

### Phase 3: Complete (Full CatSim Parity)
5. **Offset scan support** - Dark current modeling (structure exists, not exposed)
6. **Configurable max prep** ✅ - Optional sinogram clamping (max_prep kwarg)

---

## Example: Updated Signal Chain

After implementing these changes, the full CatSim-style signal chain would be:

```julia
# 1. Forward project with physics (quantum noise applied here)
sinogram_raw = forward_project(phantom, geom;
    energies=energies, weights=weights,
    materials=materials, physics=physics_config)

# 2. Convert to intensity domain
intensity_phantom = exp.(-sinogram_raw)

# 3. Simulate air scan (NO noise, simulating averaged reference)
intensity_air = simulate_air_scan(size(intensity_phantom), heel, das, geom)

# 4. Apply heel effect to phantom (air already has it)
apply_heel_effect!(intensity_phantom, heel, geom)

# 5. Apply DAS model to phantom only (physical units)
apply_das_model!(intensity_phantom, das; seed=42)
# Note: Air scan already has gain applied, but NO noise

# 6. Calibrate with low signal correction
sinogram_calibrated = calibrate_sinogram(
    intensity_phantom, intensity_air;
    offset = 0.0f0,  # Optional dark current
    low_signal_correction = true
)

# 7. Beam hardening correction
apply_bhc!(sinogram_calibrated, bhc)

# 8. Reconstruct
recon = fdk_reconstruct(sinogram_calibrated, geom, recon_size)
```

---

## Testing Strategy

1. **Unit tests**: Each new function in isolation
2. **Integration test**: Full signal chain produces correct HU values
3. **Validation targets**:
   - Solid water: 0 ± 20 HU
   - Ca-100: ~100 HU
   - Ca-200: ~200 HU
   - Noise std in water: clinically reasonable (< 50 HU)

---

## Questions to Resolve

1. **Offset scan value**: What is a typical dark current value? CatSim uses 0 in examples, but the structure supports it.

2. **Reference flux**: What reference photon flux should we use for DAS noise normalization? This affects the noise-to-signal ratio.

3. **Kernel for low signal correction**: CatSim uses a simple averaging kernel. Should we use the same or something more sophisticated?

---

## Summary

The main issue with our current implementation is that we apply DAS noise to BOTH phantom and air scans, which doesn't cancel out during calibration and leads to excessive noise in the final sinogram. CatSim's approach of noise-free air scans is more physically accurate (real air scans are averaged) and produces stable calibration.

The secondary issue is our noise units being arbitrary rather than physical (electrons), which makes it hard to set realistic noise levels.

Implementing these changes will bring BasisSimulator.jl into alignment with CatSim's physics and produce clinically realistic CT simulations.
