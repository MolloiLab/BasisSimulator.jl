# GECATSIM vs BasisSimulator Feature Comparison

**Date**: January 8, 2026
**Purpose**: Document GECATSIM features to identify gaps in BasisSimulator for realistic 3D cone-beam CT simulation

---

## Executive Summary

GECATSIM is a comprehensive CT simulator with extensive physics models. This document identifies features BasisSimulator currently lacks that would be valuable for realistic 3D cone-beam CT simulation.

**Key Findings**:
- GECATSIM has significantly more detector physics models
- Focal spot modeling is more sophisticated
- Multiple phantom types supported
- Extensive noise and artifact models
- More complete scanner geometry options

---

## 1. Detector Physics

### GECATSIM Has

```python
# Detector sensor
scanner.detectorMaterial = "Lumex"
scanner.detectorDepth = 3.0  # mm
scanner.detectionGain = 15.0  # electrons / keV
scanner.detectorColFillFraction = 0.9
scanner.detectorRowFillFraction = 0.9
scanner.eNoise = 5000.0  # electronic noise (electrons)

# Crosstalk
physics.col_crosstalk = 0.01
physics.row_crosstalk = 0.01
physics.col_crosstalk_opt = 0.01  # optical crosstalk
physics.row_crosstalk_opt = 0.01

# Lag (temporal effects)
physics.lag_taus = [1., 10.]  # time constants
physics.lag_alphas = [0.9, 0.1]  # weights
```

### BasisSimulator Has

- Basic polychromatic detection (energy-weighted)
- Electronic noise (Gaussian)
- Poisson (quantum) noise
- 1/f noise (low-frequency drift)

### BasisSimulator Missing

| Feature | Impact | Priority | Effort |
|---------|--------|----------|--------|
| **Crosstalk** | Blur, resolution loss | MEDIUM | 2-3 hours |
| **Detector lag** | Temporal artifacts | LOW | 3-4 hours |
| **Fill fraction** | Quantum efficiency | LOW | 1 hour |
| **Energy-dependent QDE** | Spectral realism | MEDIUM | 2-3 hours |
| **Material-specific response** | Realism | LOW | 2 hours |

**Recommendation**: Implement crosstalk and energy-dependent QDE for Phase 5. Defer lag and material response unless needed for specific applications.

---

## 2. X-ray Source Physics

### GECATSIM Has

```python
# Focal spot
scanner.focalspotCallback = "SetFocalspot"
scanner.focalspotShape = "Uniform"  # or Gaussian, etc.
scanner.targetAngle = 7.0  # degrees
scanner.focalspotWidth = 1.0  # mm
scanner.focalspotLength = 1.0  # mm

# Wobbling and offset
protocol.wobbleDistance = 0.0  # mm
protocol.focalspotOffset = [0, 0, 0]  # mm

# Tube modulation
protocol.dutyRatio = 1.0  # pulsed tubes
protocol.mA = 200  # tube current
```

### BasisSimulator Has

- Kramers' Law spectrum generation
- K-alpha/K-beta characteristic lines
- Filtration (Al, Cu)
- Heel effect modeling
- mAs scaling

### BasisSimulator Missing

| Feature | Impact | Priority | Effort |
|---------|--------|----------|--------|
| **Focal spot size** | Resolution modeling | MEDIUM | 2-3 hours |
| **Focal spot shape** | Blur kernel | LOW | 2-3 hours |
| **Wobbling** | Helical scan artifacts | LOW | 1-2 hours |
| **Pulsed tube** | Dose modulation | LOW | 1 hour |
| **mA modulation** | Tube current control | LOW | 1 hour |

**Recommendation**: Implement focal spot size/shape for Phase 5 if resolution modeling becomes important. Defer wobbling and pulsed tube to specialized applications.

---

## 3. Scanner Geometry

### GECATSIM Has

```python
# Basic geometry
scanner.sid = 540.0  # mm (source-iso distance)
scanner.sdd = 950.0  # mm (source-detector distance)

# Detector layout
scanner.detectorColsPerMod = 1
scanner.detectorRowsPerMod = 16
scanner.detectorColOffset = 0.25  # fractional offset
scanner.detectorRowOffset = 0.0

# Gantry
protocol.tiltAngle = 0  # degrees
protocol.rotationDirection = 1  # CW/CCW

# Helical scanning
protocol.tableSpeed = 0  # mm/sec
protocol.startZ = 0  # mm
```

### BasisSimulator Has

- SAD, SDD (Canon Aquilion ONE: 600mm, 1000mm)
- Detector rows/cols (320x800)
- Pixel width/height
- Rotation angles
- Pre-computed source/detector trajectories

### BasisSimulator Missing

| Feature | Impact | Priority | Effort |
|---------|--------|----------|--------|
| **Gantry tilt** | Specialized scans | LOW | 1-2 hours |
| **Helical scanning** | Clinical protocols | HIGH | 6-8 hours |
| **Table translation** | Dynamic scanning | MEDIUM | 2-3 hours |
| **Modular detector** | Realism | LOW | 2 hours |
| **Detector offsets** | Calibration | LOW | 1 hour |

**Recommendation**: Implement helical scanning for Phase 6-7 if clinical realism needed. Defer gantry tilt and modular detector to specialized applications.

---

## 4. Phantom Types

### GECATSIM Has

```python
phantom.callback = "Phantom_Voxelized"
# OR: Phantom_Analytic, Phantom_Hybrid, Phantom_Polygonal,
#     Phantom_NURB, Phantom_Dynamic, Phantom_XCAT
```

**Phantom types**:
1. **Voxelized**: Regular grid of materials/densities
2. **Analytical**: Mathematical shapes (cylinders, ellipsoids)
3. **Hybrid**: Mix of analytical and voxelized
4. **Polygonal**: Surface meshes
5. **NURB**: Smooth surface representations
6. **Dynamic**: Time-varying (cardiac, respiratory)
7. **XCAT**: Realistic human anatomy with motion

### BasisSimulator Has

- **Voxelized phantoms** (regular grid)
- Gammex 472 calibration phantom
- Water cylinders
- Material IDs + densities

### BasisSimulator Missing

| Feature | Impact | Priority | Effort |
|---------|--------|----------|--------|
| **Analytical phantoms** | Speed, memory | MEDIUM | 3-4 hours |
| **Dynamic phantoms** | Motion artifacts | MEDIUM | 8-10 hours |
| **XCAT integration** | Realism | LOW | External tool |
| **Polygonal meshes** | Complex shapes | LOW | 4-6 hours |

**Recommendation**: Implement analytical phantoms (cylinders, ellipsoids) for Phase 5-6 for speed/memory benefits. Defer dynamic phantoms unless motion modeling becomes a priority.

---

## 5. Physics Models

### GECATSIM Has

```python
# Sampling
physics.energyCount = 20
physics.colSampleCount = 2  # detector sub-pixels
physics.rowSampleCount = 2
physics.srcXSampleCount = 2  # focal spot sub-pixels
physics.srcYSampleCount = 2
physics.viewSampleCount = 2  # temporal sub-sampling

# Noise
physics.enableQuantumNoise = 1
physics.enableElectronicNoise = 1

# Scatter
physics.scatterCallback = ""  # can plug in scatter model

# Recalculation flags
physics.recalcDet = 0  # 0: no recalc, 1: every view, 2: every subview
physics.recalcSrc = 0
physics.recalcGantry = 2
physics.recalcSpec = 0
```

### BasisSimulator Has

- **Polychromatic spectrum** (20+ energy bins)
- **Sub-pixel sampling** (2x2 detector)
- **Poisson noise** (quantum)
- **Electronic noise** (Gaussian)
- **1/f noise** (low-frequency drift)
- **Attenuation** (NIST XCOM via XrayAttenuation.jl)

### BasisSimulator Missing

| Feature | Impact | Priority | Effort |
|---------|--------|----------|--------|
| **Focal spot sampling** | Blur realism | MEDIUM | 2-3 hours |
| **Temporal sampling** | Motion blur | LOW | 2-3 hours |
| **Scatter models** | Accuracy | HIGH | 8-12 hours |
| **Recalc optimization** | Speed | MEDIUM | 4-6 hours |
| **Callback architecture** | Flexibility | LOW | 6-8 hours |

**Recommendation**: Implement scatter models (Phase 2 plan) as highest priority. Add focal spot sampling in Phase 5. Defer callback architecture to major refactor.

---

## 6. Scan Types

### GECATSIM Has

```python
protocol.scanTypes = [1, 1, 1, 1]
# [air scan, offset scan, phantom scan, prep]
```

**Scan types**:
1. **Air scan**: No phantom (calibration)
2. **Offset scan**: Dark current measurement
3. **Phantom scan**: Actual CT scan
4. **Prep**: Final preprocessing

### BasisSimulator Has

- **Phantom scan**: Forward projection through phantom
- **Air scan**: Can simulate with empty phantom

### BasisSimulator Missing

| Feature | Impact | Priority | Effort |
|---------|--------|----------|--------|
| **Offset scan** | Calibration realism | LOW | 1 hour |
| **Scan workflow** | Clinical realism | LOW | 2 hours |
| **Multi-scan protocol** | Automation | LOW | 2-3 hours |

**Recommendation**: Defer unless building full calibration pipeline. Not critical for physics validation.

---

## 7. Reconstruction Options

### GECATSIM Has

```python
# Multiple reconstruction algorithms available
# - FDK (cone-beam)
# - 2D filtered backprojection
# - Helical reconstruction
# - Iterative methods
```

### BasisSimulator Has

- **FDK** (Feldkamp-Davis-Kress)
- Filters: Ram-Lak, Shepp-Logan, Hann, Cosine
- Parker weighting (short-scan)
- Hounsfield Unit conversion

### BasisSimulator Missing (from Phase 3)

| Feature | Impact | Priority | Effort |
|---------|--------|----------|--------|
| **SIRT** | Iterative recon | MEDIUM | 4-6 hours* |
| **MLEM** | Statistical recon | MEDIUM | 4-6 hours* |
| **TV regularization** | Noise reduction | MEDIUM | 6-8 hours* |
| **Helical recon** | Clinical scans | LOW | 6-8 hours |

*Requires volume-based forward projector infrastructure (Phase 3 decision)

**Recommendation**: Defer until forward projector implemented (Phase 4+). FDK sufficient for current validation needs.

---

## 8. File I/O and Data Formats

### GECATSIM Has

```python
# Raw data I/O
xc.rawread(filename, dims, dtype)
xc.rawwrite(filename, data)

# Configuration
ct = xc.CatSim("phantom.cfg", "scanner.cfg", "protocol.cfg")
ct.load_cfg("physics.cfg", "recon.cfg")

# Results
ct.resultsName = "test"  # base filename
# Generates: test.prep, test.raw, etc.
```

### BasisSimulator Has

- Julia native arrays
- No specific file format (users handle I/O)
- HDF5 potential (not implemented)

### BasisSimulator Missing

| Feature | Impact | Priority | Effort |
|---------|--------|----------|--------|
| **Standard file format** | Interoperability | LOW | 2-3 hours |
| **Config file system** | Reproducibility | MEDIUM | 4-6 hours |
| **Result management** | Usability | LOW | 2-3 hours |

**Recommendation**: Consider HDF5 or JLD2 for Phase 6-7 if data sharing becomes important. Config system could improve reproducibility.

---

## 9. Advanced Features

### GECATSIM Has

**Detector models**:
- Energy-integrating (EI)
- Photon-counting (PCCT)
- Multiple detector materials (Lumex, CdTe, etc.)

**Projector types**:
- C-based voxelized projector (fast)
- Analytical projector
- Hybrid projector

**Spectrum handling**:
- Pre-tabulated spectra with filtration
- Bowtie filter modeling
- Heel effect

**Materials database**:
- 200+ materials with NIST data
- Custom material support

### BasisSimulator Has

- **Energy-integrating** (polychromatic)
- **Pure-Julia ray tracer** (Amanatides-Woo)
- **Spectrum generation** (Kramers + K-lines)
- **XrayAttenuation.jl** (NIST XCOM)
- **Custom materials** (via XrayAttenuation.jl)

### BasisSimulator Missing

| Feature | Impact | Priority | Effort |
|---------|--------|----------|--------|
| **PCCT support** | Future tech | LOW | 12-16 hours |
| **C/C++ projector** | Speed (10-100x) | MEDIUM | 16-20 hours |
| **Bowtie filter** | Clinical realism | MEDIUM | 3-4 hours |
| **Materials database** | Convenience | LOW | 4-6 hours |

**Recommendation**: Consider bowtie filter for Phase 5. C/C++ projector deferred in favor of Reactant/XLA compilation (existing plan). PCCT for Phase 7+ if specialty scanning needed.

---

## 10. Summary: Priority Gaps

### HIGH Priority (Implement Soon)

1. **Scatter models** (Phase 2 plan)
   - Klein-Nishina differential cross section
   - Single-scatter or convolution approximation
   - Effort: 8-12 hours
   - Impact: Accuracy for quantitative imaging

2. **Helical scanning** (Phase 6)
   - Table translation
   - Helical interpolation
   - Effort: 6-8 hours
   - Impact: Clinical protocol compatibility

### MEDIUM Priority (Consider for Phase 5-6)

3. **Focal spot modeling**
   - Size and shape
   - Blur kernel
   - Effort: 2-3 hours
   - Impact: Resolution realism

4. **Detector crosstalk**
   - Column and row crosstalk
   - Effort: 2-3 hours
   - Impact: Spatial resolution accuracy

5. **Energy-dependent QDE**
   - Quantum detection efficiency vs energy
   - Effort: 2-3 hours
   - Impact: Spectral accuracy

6. **Analytical phantoms**
   - Cylinders, ellipsoids
   - Effort: 3-4 hours
   - Impact: Speed and memory efficiency

7. **Bowtie filter**
   - Dose modulation
   - Effort: 3-4 hours
   - Impact: Clinical realism

### LOW Priority (Defer or Skip)

8. Detector lag
9. Gantry tilt
10. Pulsed tubes
11. PCCT support
12. Dynamic phantoms (unless motion modeling becomes priority)
13. Configuration file system
14. Standard file formats

---

## 11. BasisSimulator Strengths

### What BasisSimulator Does Better

1. **Pure Julia**
   - No compilation required
   - Easy to install and modify
   - Type-stable and fast

2. **Automatic Differentiation Ready**
   - Enzyme.jl compatible (unique!)
   - Reactant.jl for XLA compilation
   - Enables gradient-based optimization

3. **Modular Architecture**
   - Clean separation of concerns
   - Easy to extend
   - Well-documented

4. **Modern Dependencies**
   - XrayAttenuation.jl (NIST data)
   - FFTW.jl (fast reconstruction)
   - CairoMakie.jl (visualization)

5. **Comprehensive Testing**
   - 790 physics validation tests
   - NIST cross-validation
   - Full pipeline integration tests

### Unique Capabilities

- **Fully differentiable** (no other CT simulator has this!)
- **GPU/TPU ready** (via Reactant.jl)
- **Inverse problems focused**
  - Material decomposition
  - Dose optimization
  - Geometry calibration

---

## 12. Recommendations

### Short Term (Phase 4b-5)

1. Complete GECATSIM cross-validation
2. Implement scatter models (Phase 2 original plan)
3. Add bowtie filter support
4. Consider focal spot and crosstalk models

### Medium Term (Phase 6-7)

1. Helical scanning support
2. Analytical phantom support
3. Configuration file system for reproducibility
4. Performance optimization (Reactant compilation)

### Long Term (Phase 8+)

1. Dynamic phantom support (if motion modeling needed)
2. PCCT support (if specialty imaging needed)
3. C/C++ projector (if Reactant insufficient)
4. Advanced scatter models (Monte Carlo)

### Philosophy

**BasisSimulator should focus on**:
- Differentiability (unique advantage)
- Core physics accuracy (validation against NIST, GECATSIM)
- Inverse problems (material decomp, dose optimization)
- Ease of use and modification (Julia ecosystem)

**BasisSimulator should not try to**:
- Match every GECATSIM feature
- Support every clinical protocol
- Replicate commercial scanner behavior exactly

**BasisSimulator is for**:
- Research and development
- Algorithm prototyping
- Inverse problem solving
- Education and demonstration

**BasisSimulator is not for**:
- Clinical workflow simulation
- Regulatory compliance
- Commercial scanner emulation
- Real-time operation

---

## 13. Conclusions

GECATSIM is a mature, comprehensive CT simulator with extensive physics models. BasisSimulator has a focused scope: differentiable CT simulation for inverse problems.

**Key gaps**:
- Scatter modeling (high priority)
- Helical scanning (medium priority)
- Detector physics (crosstalk, QDE)
- Focal spot modeling

**Unique advantages**:
- Full differentiability
- Modern Julia ecosystem
- Inverse problem focus
- Automatic differentiation

**Next steps**:
1. Complete GECATSIM validation (Phase 4b)
2. Implement scatter models (Phase 2 continuation)
3. Selectively add detector/source physics (Phase 5)
4. Maintain focus on differentiability and inverse problems

---

**Document Version**: 1.0
**Last Updated**: January 8, 2026
**Next Review**: After Phase 4b completion
