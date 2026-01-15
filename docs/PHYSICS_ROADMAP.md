# Physics Realism Roadmap - GPU-Compatible Implementation

## Current Status

BasisSimulator.jl has **10 complete physics effect modules**, all currently CPU-only:

| Module | Physics Effect | Status | GPU Ready? |
|--------|---------------|--------|------------|
| `Scatter.jl` | Compton/Rayleigh scatter | Complete | No (FFTW) |
| `DetectorNoise.jl` | Quantum + Electronic noise | Complete | No (Random.jl) |
| `DetectorEfficiency.jl` | Scintillator absorption/DQE | Complete | Easy |
| `BowtieFilter.jl` | Angle-dependent pre-filtration | Complete | Easy |
| `FlatFilter.jl` | Uniform pre-filtration | Complete | Easy |
| `FocalSpot.jl` | Geometric blur from finite spot | Complete | No (FFTW) |
| `Crosstalk.jl` | Detector pixel coupling | Complete | No (FFTW) |
| `DetectorLag.jl` | Afterglow/temporal persistence | Complete | Hard (sequential) |
| `FillFactor.jl` | Pixel dead area | Complete | Trivial |
| `FlyingFocalSpot.jl` | Focal spot deflection | Complete | N/A (geometry) |

---

## CatSim/XCIST Comparison

Based on the [XCIST paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10151073/) and [CatSim documentation](https://github.com/xcist/documentation/wiki):

### CatSim Simulation Pipeline

```
1. Spectrum Generation (energy bins + focal spot weighting)
2. Pre-filter Attenuation (flat filter + bowtie)
3. Per-View Loop:
   a. Scale flux by tube current (mA)
   b. Ray-trace through phantom → path lengths
   c. Energy-dependent attenuation (Beer-Lambert)
   d. Scatter convolution (optional)
   e. Detector response:
      - Apply DQE (energy & position dependent)
      - Poisson noise on photon counts
      - Energy integration (sum over bins)
      - Electronic noise (Gaussian)
      - Crosstalk convolution
      - Lag/afterglow (temporal IIR)
   f. Signal conversion (keV → electrons)
4. Gantry/table motion → next view
```

### BasisSimulator vs CatSim Feature Parity

| Feature | CatSim | BasisSimulator | Notes |
|---------|--------|----------------|-------|
| Polychromatic source | ✓ | ✓ | `Polychromatic.jl` |
| Spectrum modeling | ✓ | ✓ | `Spectrum.jl` |
| Focal spot blur | ✓ (sampling) | ✓ (convolution) | Different approaches |
| Bowtie filter | ✓ | ✓ | CatSim-compatible format |
| Flat filter | ✓ | ✓ | Multi-material |
| Scatter (convolution) | ✓ | ✓ | Ohnesorge model |
| Detector efficiency/DQE | ✓ | ✓ | NIST XCOM data |
| Quantum noise (Poisson) | ✓ | ✓ | Knuth algorithm |
| Electronic noise | ✓ | ✓ | Gaussian additive |
| Detector crosstalk | ✓ | ✓ | X-ray + optical models |
| Detector lag | ✓ | ✓ | Multi-exponential |
| Fill factor | ✓ | ✓ | Row/column separable |
| Flying focal spot | ✓ | ✓ | 2/4 position patterns |
| Monte Carlo scatter | ✓ | ✗ | Not implemented |
| Heel effect | ✓ | ✗ | Not implemented |

**Key finding: BasisSimulator has feature parity with CatSim for all major physics effects.**

---

## GPU Implementation Strategy

### Tier 1: Easy (Element-wise Operations)

These can be directly converted to GPU using `AK.foreachindex`:

#### 1. Fill Factor
```julia
# Current (CPU)
sinogram .-= log(fill_factor)

# GPU version
AK.foreachindex(sinogram) do idx
    sinogram[idx] -= log(T(fill_factor))
end
```

#### 2. Flat Filter
```julia
# Per-pixel attenuation is pre-computed, then element-wise multiply
AK.foreachindex(intensity) do idx
    intensity[idx] *= flat_attenuation[idx]
end
```

#### 3. Bowtie Filter
Same as flat filter - pre-compute 2D attenuation map, then element-wise.

#### 4. Detector Efficiency
Pre-compute efficiency map (energy-dependent if spectral), then:
```julia
AK.foreachindex(intensity) do idx
    intensity[idx] *= efficiency[idx]
end
```

**Estimated effort: 1-2 days total**

---

### Tier 2: Medium (GPU Random Number Generation)

#### 5. Electronic Noise (Gaussian)
Use GPU-compatible RNG from AcceleratedKernels or Metal.jl:
```julia
# Need to generate Gaussian random numbers on GPU
# Options:
# a) Use Metal.jl's rand() + Box-Muller transform
# b) Pre-generate noise on CPU, transfer to GPU
# c) Use a GPU-compatible RNG library
```

#### 6. Quantum Noise (Poisson)
More challenging - Poisson sampling on GPU:
```julia
# For high counts (>30): Gaussian approximation
# σ = √(count), sample N(count, σ)

# For low counts: Need GPU Poisson sampler
# Options:
# a) Inverse CDF method (expensive)
# b) Gaussian approximation for all (fast but less accurate)
# c) Hybrid: Gaussian for high, pre-tabulated for low
```

**Recommendation**: Start with Gaussian approximation (valid for typical CT counts >1000), add exact Poisson later.

**Estimated effort: 2-3 days**

---

### Tier 3: Hard (Convolution Operations)

Current implementation uses FFTW (CPU-only). Options:

#### Option A: GPU FFT via CUDA.jl/Metal.jl
- CUDA has cuFFT (excellent)
- Metal has limited FFT support (MPS FFT)
- Would need backend-specific code paths

#### Option B: Spatial Domain Convolution (Kernel-based)
- Works on any GPU backend via AK.jl
- Slower for large kernels, but kernels are typically small (3x3 to 15x15)
- Already proven with `Filtering.jl` ramp filter

**Recommendation**: Use spatial domain convolution for GPU compatibility.

#### 7. Crosstalk (3x3 kernel)
```julia
# Kernel is only 3x3 - spatial convolution is efficient
AK.foreachindex(output) do idx
    # Linear index to (col, row, angle)
    # Accumulate 3x3 neighborhood weighted sum
    acc = zero(T)
    for di in -1:1, dj in -1:1
        weight = kernel[di+2, dj+2]
        src_idx = clamp(...)
        acc += input[src_idx] * weight
    end
    output[idx] = acc
end
```

#### 8. Focal Spot Blur (variable kernel)
- Kernel size depends on magnification (typically 5x5 to 11x11)
- Same spatial convolution approach as crosstalk

#### 9. Scatter (larger kernel)
- Scatter kernel can be larger (~31x31 typical)
- May need to use separable approximation or truncated kernel
- Alternative: Apply scatter in projection domain with smaller effective kernel

**Estimated effort: 3-5 days**

---

### Tier 4: Very Hard (Sequential/Temporal)

#### 10. Detector Lag
The current IIR formulation requires sequential processing:
```
state[n] = decay × state[n-1] + amp × input[n]
```

**Options:**
a) **Keep on CPU**: Transfer data for lag computation, then back to GPU
b) **Parallel approximation**: Process each pixel's temporal sequence in parallel
c) **Matrix formulation**: Express as sparse matrix multiply (GPU-friendly)

**Recommendation**: Start with option (a) - keep lag on CPU. It's a minor performance hit since it's O(n_pixels) per view, not O(n_pixels × n_angles).

**Estimated effort: 2-3 days**

---

## Proposed Implementation Order

**Strategy: Replace CPU code entirely, test after each phase.**

### Phase 1: Element-wise Effects
1. `FillFactor.jl` - **Replace** with GPU version (trivial, warm-up)
2. `FlatFilter.jl` - **Replace** with GPU version
3. `BowtieFilter.jl` - **Replace** with GPU version
4. `DetectorEfficiency.jl` - **Replace** with GPU version
5. **TEST**: Run demo with all 4 effects on Metal GPU, verify HU accuracy

### Phase 2: Noise Effects
6. `DetectorNoise.jl` - **Replace** electronic noise with GPU version
7. `DetectorNoise.jl` - **Replace** quantum noise with GPU version (Gaussian approx)
8. **TEST**: Run noisy simulation on Metal GPU, verify noise statistics

### Phase 3: Convolution Effects
9. `Crosstalk.jl` - **Replace** with spatial domain GPU convolution
10. `FocalSpot.jl` - **Replace** with spatial domain GPU convolution
11. **TEST**: Run blur/crosstalk simulation on Metal GPU, verify MTF

### Phase 4: Complex Effects
12. `Scatter.jl` - **Replace** with spatial domain GPU
13. `DetectorLag.jl` - **Replace** (may need hybrid CPU/GPU approach)
14. **TEST**: Full physics pipeline on Metal GPU, compare to baseline

### Phase 5: Integration
15. Create unified `apply_physics_effects!()` pipeline
16. Update demo script with full physics
17. Final benchmarks and documentation

---

## Architecture Design

### Unified Physics Pipeline

```julia
"""
Apply all physics effects to intensity data (GPU-native).

Order of operations (matches CatSim):
1. Flat filter attenuation
2. Bowtie filter attenuation
3. Detector efficiency (DQE)
4. Fill factor
5. Quantum noise (Poisson)
6. Electronic noise (Gaussian)
7. Crosstalk convolution
8. Lag (per-view, may require CPU)
9. Focal spot blur
10. Scatter (optional, typically in projection domain)
"""
function apply_physics_effects!(
    intensity::AbstractArray{T,3},
    geom::CTGeometry;
    flat_filter::Union{FlatFilter,Nothing} = nothing,
    bowtie::Union{BowtieFilter,Nothing} = nothing,
    detector_efficiency::Union{DetectorEfficiency,Nothing} = nothing,
    fill_factor::Union{FillFactorModel,Nothing} = nothing,
    noise::Union{DetectorModel,Nothing} = nothing,
    crosstalk::Union{CrosstalkModel,Nothing} = nothing,
    lag::Union{LagModel,Nothing} = nothing,
    focal_spot::Union{FocalSpot,Nothing} = nothing,
    scatter::Union{ScatterModel,Nothing} = nothing,
    rng_seed::Int = 42
) where T <: AbstractFloat
    # Apply effects in physics-correct order
    ...
end
```

### GPU Array Handling

```julia
# All physics functions should:
# 1. Accept any AbstractArray (CPU or GPU)
# 2. Use AK.foreachindex for element-wise operations
# 3. Return same array type as input
# 4. Pre-compute any lookup tables on CPU, transfer once

function apply_bowtie_filter!(
    intensity::AbstractArray{T,3},
    bowtie::BowtieFilter,
    geom::CTGeometry
) where T <: AbstractFloat
    # Pre-compute attenuation map on CPU
    atten_cpu = compute_bowtie_attenuation(bowtie, geom, T)

    # Transfer to GPU (same type as intensity)
    atten = similar(intensity, size(atten_cpu)...)
    copyto!(atten, atten_cpu)

    # Apply element-wise on GPU
    AK.foreachindex(intensity) do idx
        intensity[idx] *= atten[idx]
    end

    return intensity
end
```

---

## Performance Expectations

| Effect | CPU Time | GPU Time (est.) | Speedup |
|--------|----------|-----------------|---------|
| Fill Factor | <1ms | <0.1ms | 10x |
| Flat Filter | ~5ms | <0.5ms | 10x |
| Bowtie Filter | ~10ms | <1ms | 10x |
| Detector Efficiency | ~10ms | <1ms | 10x |
| Electronic Noise | ~50ms | ~5ms | 10x |
| Quantum Noise | ~100ms | ~10ms | 10x |
| Crosstalk (3x3) | ~50ms | ~5ms | 10x |
| Focal Spot Blur | ~100ms | ~20ms | 5x |
| Scatter | ~200ms | ~50ms | 4x |
| Lag | ~20ms | ~20ms (CPU) | 1x |

**Total physics pipeline: ~500ms CPU → ~100ms GPU (5x speedup)**

---

## References

1. [XCIST Paper (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10151073/)
2. [CatSim Documentation Wiki](https://github.com/xcist/documentation/wiki)
3. [CatSim Source Code](https://github.com/xcist/main)
4. Ohnesorge et al., "Efficient correction for CT image artifacts caused by objects extending outside the scan field of view" (1999) - Scatter model
5. Feldkamp, Davis, Kress (1984) - FDK reconstruction

---

Last Updated: 2026-01-14
