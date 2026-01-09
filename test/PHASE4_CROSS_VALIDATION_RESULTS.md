# Phase 4: GECATSIM Cross-Validation Results

**Date**: January 8, 2026
**Status**: Initial cross-validation complete - significant discrepancies identified

---

## Summary

Initial cross-validation between BasisSimulator and GECATSIM shows **poor quantitative agreement**, despite both simulators running successfully. This indicates fundamental differences in signal conventions, phantom geometries, or physics models that require investigation.

---

## Test Configuration

### Common Parameters
- **Phantom**: Water cylinder (~100mm diameter, 20mm height)
- **kVp**: 120.0
- **mAs**: 200.0
- **Projections**: 360 (full rotation)
- **Scanner**: Canon Aquilion ONE geometry
  - SAD: 600 mm
  - SDD: 1000 mm

### GECATSIM Configuration
- **Phantom**: W20.ppm (scaled 0.5)
- **Detector**: 320 × 800 pixels
- **Output format**: Raw float32 sinogram
- **Output file**: test_output.scan (352 MB)
- **Physics**: 20 energy bins, no scatter/noise

### BasisSimulator Configuration
- **Phantom**: `create_water_cylinder(diameter_mm=100.0, height_mm=100.0, resolution_mm=2.0)`
- **Detector**: 320 × 400 pixels (auto-generated from Canon geometry)
- **Spectrum**: `generate_spectrum(kVp=120.0, mAs=200.0)`
- **Output**: Detector signal array (Float64)

---

## Quantitative Results

### Signal Statistics

| Metric | GECATSIM | BasisSimulator | Ratio |
|--------|----------|----------------|-------|
| **Shape** | 320 × 800 × 360 | 320 × 400 × 360 | — |
| **Min** | 2.11e7 | 9.33e8 | 44× |
| **Max** | 1.97e8 | 8.00e9 | 41× |
| **Mean** | 1.58e8 | 5.97e9 | 38× |

**Observation**: BasisSimulator produces signals ~40× larger than GECATSIM.

### Comparison Metrics (Normalized [0,1])

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **RMSE** | 0.5546 | < 0.1 | ❌ Failed |
| **MAE** | 0.3514 | — | ❌ High |
| **Max Diff** | 1.0 | — | ❌ Saturated |
| **Correlation** | 0.06 | > 0.9 | ❌ Failed |
| **Profile Correlation** | -0.9082 | > 0.9 | ❌ **NEGATIVE!** |
| **Profile RMSE** | 0.7872 | < 0.1 | ❌ Failed |

---

## Critical Issues Identified

### 1. Negative Profile Correlation ⚠️

**Finding**: Central profile shows correlation of **-0.9082** (strong negative correlation).

**Implication**: The signals are **inverted** relative to each other. This suggests:
- One simulator outputs **transmission** (I/I₀)
- The other outputs **attenuation** (-log(I/I₀))
- Or one outputs **photon counts** while the other outputs **intensity**

**Evidence**:
```
Profile correlation: -0.9082  ← Should be positive!
```

**Action Required**: Investigate signal conventions in both simulators.

---

### 2. Signal Magnitude Mismatch (40× Difference)

**Finding**: BasisSimulator signals are 38-44× larger than GECATSIM.

**Possible Causes**:
1. **Unit mismatch**: BasisSimulator in photons, GECATSIM in intensity?
2. **mAs normalization**: BasisSimulator might not be normalizing by exposure time
3. **Detector area**: Different pixel sizes or collection efficiencies
4. **Spectrum normalization**: Total fluence differs by 40×

**Investigation Needed**:
- Check BasisSimulator `simulate_ct_scan` output units
- Verify GECATSIM `.scan` file format specification
- Compare air scan values (I₀) from both simulators

---

### 3. Detector Dimensions Mismatch

**GECATSIM**: 320 × 800 pixels
**BasisSimulator**: 320 × 400 pixels

**Observation**: BasisSimulator generates half the detector columns.

**Possible Causes**:
- `create_aquilion_one()` may have wrong default detector geometry
- GECATSIM uses full detector array
- Canon Aquilion ONE spec ambiguity (320×640 vs 320×800?)

**Action**: Verify Canon Aquilion ONE official specifications and fix `ScannerGeometry.jl:294-309`.

---

### 4. Phantom Geometry Uncertainty

**GECATSIM**: Uses `W20.ppm` scaled 0.5
- Official GECATSIM phantom with validated geometry
- Actual dimensions unknown (need to inspect W20.ppm format)

**BasisSimulator**: Custom water cylinder
- 100mm diameter
- 100mm height (vs 20mm in test name)
- 2mm voxel resolution

**Action**:
1. Extract exact W20.ppm geometry specifications
2. Match BasisSimulator phantom dimensions precisely
3. Verify voxel grid alignment

---

## Hypotheses for Discrepancies

### Hypothesis 1: Signal Convention Mismatch (HIGH PROBABILITY)
**Evidence**: Negative correlation
**Fix**: Identify which simulator outputs transmission vs attenuation, apply appropriate conversion

### Hypothesis 2: Spectrum Normalization (HIGH PROBABILITY)
**Evidence**: 40× magnitude difference
**Fix**: Normalize BasisSimulator spectrum to match GECATSIM total fluence

### Hypothesis 3: Detector QDE Modeling (MEDIUM PROBABILITY)
**Evidence**: Magnitude mismatch
**Fix**: GECATSIM may include quantum detection efficiency that BasisSimulator lacks

### Hypothesis 4: Phantom Geometry Mismatch (MEDIUM PROBABILITY)
**Evidence**: Different detector dimensions, unknown W20.ppm specs
**Fix**: Exactly replicate GECATSIM phantom in BasisSimulator

### Hypothesis 5: Polychromatic Spectrum Differences (LOW PROBABILITY)
**Evidence**: Both use 120 kVp
**Fix**: Compare spectrum files directly (tungsten_tar7.0_120_filt.dat vs BasisSimulator)

---

## Next Steps

### Immediate Actions (Priority 1)

1. **Investigate Signal Conventions**
   - Read GECATSIM documentation for `.scan` file format
   - Check if GECATSIM outputs transmission (I) or attenuation (-log(I/I₀))
   - Print BasisSimulator `detector_signal` meaning (photons? intensity?)
   - Try inversion: `detector_signal_inverted = -log(detector_signal / I0)`

2. **Match Detector Dimensions**
   - Verify Canon Aquilion ONE official specs (320×640 or 320×800?)
   - Fix `create_aquilion_one()` to match GECATSIM (320×800)
   - Re-run cross-validation

3. **Extract W20.ppm Geometry**
   - Locate W20.ppm file in GECATSIM installation
   - Parse phantom specifications (diameter, height, material density)
   - Replicate exactly in BasisSimulator

### Secondary Actions (Priority 2)

4. **Compare Spectra**
   - Extract GECATSIM spectrum file: `tungsten_tar7.0_120_filt.dat`
   - Compare energy bins, fluence, filtration
   - Normalize BasisSimulator spectrum to match

5. **Verify Air Scan (I₀)**
   - Run both simulators with empty phantom (air only)
   - Compare air scan intensities
   - Calculate magnitude ratio

6. **Incremental Validation**
   - Start with monochromatic (single energy)
   - Then polychromatic (full spectrum)
   - Add complexity incrementally

### Analysis Tools (Priority 3)

7. **Create Diagnostic Plots**
   - Sinogram slices (row, column, angle views)
   - Profile plots with both simulators overlaid
   - Difference maps (absolute and relative)
   - Histogram comparisons

8. **Unit Tests**
   - Water attenuation at 60 keV (monochromatic)
   - Ray tracing path length conservation
   - Spectrum total fluence

---

## Validation Criteria (Revised)

### Phase 4a: Signal Convention Alignment
- [ ] Identify signal types (transmission vs attenuation)
- [ ] Apply appropriate conversions
- [ ] Achieve **same sign** correlation (positive)
- [ ] Target: Profile correlation > 0.5 (preliminary)

### Phase 4b: Magnitude Normalization
- [ ] Match detector dimensions (320×800)
- [ ] Normalize spectra to same total fluence
- [ ] Target: Signal ratio within 2× (not 40×)

### Phase 4c: Geometry Matching
- [ ] Replicate W20.ppm phantom exactly
- [ ] Verify voxel grid alignment
- [ ] Target: Correlation > 0.8

### Phase 4d: Quantitative Agreement
- [ ] RMSE (normalized) < 0.1
- [ ] Correlation > 0.9
- [ ] HU values within ±10 HU

---

## Lessons Learned

1. **Signal Convention Documentation Critical**: Need clear specification of what `simulate_ct_scan()` outputs (photons? intensity? transmission?)

2. **Geometry Validation Before Cross-Validation**: Should have verified exact phantom match first

3. **Incremental Comparison Strategy**: Should start with monochromatic, simple geometry before full polychromatic complex phantom

4. **Unit Tests Essential**: Need tests for individual components (spectrum fluence, attenuation calculation, ray tracing) before integration tests

---

## References

**GECATSIM Documentation**:
- Configuration file format: `test/cfg_gecatsim/*.cfg`
- Phantom format: Water cylinder phantom definition in `.ppm` files
- Output format: Raw binary `.scan` files (float32)

**BasisSimulator Implementation**:
- Forward simulation: `src/Simulation.jl:simulate_ct_scan()`
- Spectrum generation: `src/Physics/Spectrum.jl:generate_spectrum()`
- Scanner geometry: `src/Geometry/ScannerGeometry.jl:create_aquilion_one()`
- Ray tracing: `src/Geometry/RayTracing.jl:raytrace_phantom()`

**Test Scripts**:
- Cross-validation: `test/test_cross_validation_simple.jl`
- GECATSIM runner: `test/test_gecatsim_simple_run.jl`
- Full pipeline: `test/test_full_pipeline_water.jl`

---

## Status: INVESTIGATION REQUIRED

Current cross-validation shows fundamental mismatches that prevent meaningful comparison. Further investigation into signal conventions, detector geometry, and phantom specifications required before proceeding to advanced validation.

**Recommendation**: Pause advanced physics implementation (Phase 2-3) until baseline cross-validation shows reasonable agreement (correlation > 0.8).

---

**Document Version**: 1.0
**Last Updated**: January 8, 2026
**Next Review**: After signal convention investigation
