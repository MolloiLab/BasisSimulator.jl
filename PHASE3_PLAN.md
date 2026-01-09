# Phase 3 Implementation Plan

**Date**: January 8, 2026
**Goal**: Implement advanced reconstruction algorithms and complete noise models
**Timeline**: ~3-4 hours

---

## Priority Order

### 1. SIRT (Simultaneous Iterative Reconstruction Technique) - HIGH PRIORITY
**Complexity**: Medium
**Time**: 1-1.5 hours

**Algorithm**:
```
x^(k+1) = x^(k) + λ × C × A^T × R × (b - A × x^(k))
```

**Components needed**:
- Forward projection operator A (ray tracing through volume)
- Back projection operator A^T (backproject sinogram to volume)
- Row/column normalization (R, C matrices)
- Iterative update with relaxation parameter

**Testing**:
- Simple water cylinder phantom
- Compare with FDK reconstruction
- Convergence monitoring
- Full pipeline test

---

### 2. MLEM (Maximum Likelihood Expectation Maximization) - MEDIUM PRIORITY
**Complexity**: Medium-High
**Time**: 1-1.5 hours

**Algorithm**:
```
x^(k+1)_j = (x^(k)_j / Σ_i A_ij) × Σ_i A_ij × (b_i / [A × x^(k)]_i)
```

**Components needed**:
- Same projection operators as SIRT
- Multiplicative update rule
- Non-negativity constraint (automatic)
- Epsilon for numerical stability

**Testing**:
- Low-dose reconstruction test
- Poisson noise validation
- Compare with SIRT and FDK
- Full pipeline test

---

### 3. TV (Total Variation) Regularization - LOWER PRIORITY
**Complexity**: High
**Time**: 1.5-2 hours

**Algorithm**:
```
E(x) = ||A×x - b||² + λ × TV(x)
TV(x) = Σ |∇x|
```

**Components needed**:
- Projection operators
- Gradient computation (finite differences)
- TV norm computation
- Optimization solver (gradient descent or ADMM)

**Testing**:
- Sparse-angle reconstruction (60 projections)
- Edge preservation validation
- Parameter sweep (λ)
- Full pipeline test

**Decision**: If time-limited, implement placeholder with detailed algorithm description

---

### 4. 1/f Noise Model - QUICK WIN
**Complexity**: Low
**Time**: 30 minutes

**Algorithm**:
```
S(f) ∝ 1/f^α    (α ≈ 1)
```

**Components needed**:
- FFT-based noise generation
- Power spectral density shaping
- Apply to detector signal over projection angles

**Testing**:
- Visual inspection of noise spectrum
- Power spectral density validation
- Integration with Poisson + electronic noise

---

### 5. NPS (Noise Power Spectrum) Computation - UTILITY
**Complexity**: Low
**Time**: 30 minutes

**Algorithm**:
```
NPS(f) = |FFT(noise)|² / N
```

**Components needed**:
- 2D FFT of image regions
- Averaging over multiple ROIs
- Radial profile computation

**Testing**:
- Known noise statistics validation
- Compare Poisson vs electronic vs combined
- Use in image quality metrics

---

## Implementation Strategy

### Phase 3A: Iterative Reconstruction (SIRT + MLEM)
1. ✅ Create system matrix infrastructure
2. ✅ Implement forward projection (A × x)
3. ✅ Implement back projection (A^T × y)
4. ✅ Implement SIRT algorithm
5. ✅ Test SIRT with full pipeline
6. ✅ Implement MLEM algorithm
7. ✅ Test MLEM with full pipeline

### Phase 3B: TV Regularization (if time permits)
1. Implement gradient operators
2. Implement TV norm
3. Implement optimization loop
4. Test with sparse-angle data

### Phase 3C: Advanced Noise Models
1. ✅ Implement 1/f noise generation
2. ✅ Implement NPS computation
3. ✅ Test with full pipeline

---

## Testing Philosophy

**For each algorithm**:
1. **Unit test**: Simple phantom, known solution
2. **Integration test**: Full pipeline with Phase 2 physics
3. **Comparison test**: SIRT vs MLEM vs FDK vs TV
4. **Performance test**: Timing and memory usage

**Validation metrics**:
- RMSE vs ground truth
- SSIM (structural similarity)
- CNR (contrast-to-noise ratio)
- Iteration convergence plots

---

## Deferred to Phase 4

### GECATSIM Cross-Validation
**Scope**: Comprehensive comparison with NIH reference simulator
**Components**:
- Python interop setup (PythonCall.jl)
- GECATSIM installation and configuration
- Matching phantom definitions
- Multi-kVp validation
- Statistical comparison (RMSE, SSIM, HU accuracy)

**Timeline**: 2-3 hours (separate phase)

### Klein-Nishina Monte Carlo Scatter
**Scope**: Physics-accurate scatter simulation
**Complexity**: Very High
**Timeline**: 4-6 hours (maybe future work)
**Reason for deferral**: Convolution-based scatter is sufficient for current needs

---

## Success Criteria for Phase 3

- ✅ SIRT working and validated
- ✅ MLEM working and validated
- ⚠️ TV implemented (at least placeholder with algorithm description)
- ✅ 1/f noise working
- ✅ NPS computation working
- ✅ All algorithms tested with full pipeline
- ✅ Comprehensive validation report
- ✅ Performance benchmarks

---

## Files to Create/Modify

**Modify**:
- `src/Reconstruction/Iterative.jl` - Implement SIRT, MLEM, TV
- `src/Physics/Noise.jl` - Implement 1/f noise and NPS

**Create**:
- `test/test_iterative_reconstruction.jl` - SIRT/MLEM/TV validation
- `test/test_advanced_noise.jl` - 1/f noise and NPS validation
- `test/PHASE3_VALIDATION.md` - Results report

---

## Next Steps

1. Start with SIRT implementation (highest priority)
2. Create test suite alongside implementation
3. Validate each algorithm before moving to next
4. Create comprehensive Phase 3 report
5. Commit and prepare for Phase 4 (GECATSIM)
