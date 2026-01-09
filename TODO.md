# BasisSimulator.jl - Development TODO

## High Priority

### 1. Define Gammex 472 Calibration Phantom Materials

**Status**: BLOCKING visual validation

**Description**: Create custom XrayAttenuation.jl material definitions for Gammex 472 phantom inserts to enable meaningful contrast in validation simulations.

**Required Materials**:

**Calcium Inserts** (hydroxyapatite in water-equivalent resin):
- Ca_50: 50 mg/mL calcium
- Ca_100: 100 mg/mL calcium
- Ca_200: 200 mg/mL calcium
- Ca_300: 300 mg/mL calcium
- Ca_400: 400 mg/mL calcium
- Ca_500: 500 mg/mL calcium
- Ca_600: 600 mg/mL calcium

**Iodine Inserts** (iodine in water):
- I_2_0: 2.0 mg/mL iodine
- I_2_5: 2.5 mg/mL iodine
- I_5_0: 5.0 mg/mL iodine
- I_7_5: 7.5 mg/mL iodine
- I_10_0: 10.0 mg/mL iodine
- I_15_0: 15.0 mg/mL iodine
- I_20_0: 20.0 mg/mL iodine

**Implementation Approach**:

Option 1: Create local material definitions in `src/Geometry/Phantoms.jl`
```julia
# Calcium insert: Ca5(PO4)3(OH) in water-equivalent resin
const Ca_50_composition = Dict(
    "H" => ...,
    "C" => ...,
    "N" => ...,
    "O" => ...,
    "Ca" => ...
)
Ca_50 = XA.Compound(Ca_50_composition, density_g_cm3=...)
```

Option 2: Contribute to XrayAttenuation.jl with new materials
- Fork XrayAttenuation.jl
- Add Gammex materials to XA.Materials module
- Submit PR upstream
- More sustainable long-term solution

**References**:
- Gammex 472 specification sheet (concentrations and compositions)
- Hydroxyapatite formula: Ca₁₀(PO₄)₆(OH)₂
- NIST density data for calibration materials

**Blocked Tasks**:
- Full visual validation (currently no sinogram contrast)
- Gammex phantom HU calibration tests
- Material decomposition validation
- GECATSIM comparison with calibration phantom

**Next Steps**:
1. Look up Gammex 472 specification sheet for exact compositions
2. Calculate elemental mass fractions for each insert
3. Determine densities (may need to measure or use manufacturer specs)
4. Implement XA.Compound definitions
5. Validate against known HU values from literature
6. Re-run visual_validation.jl to verify contrast

**Estimated Effort**: 2-4 hours
**Priority**: HIGH (blocks validation workflow)

---

### 2. Implement GECATSIM Python Interop

**Status**: Placeholder implementation exists

**Description**: Complete the GECATSIM integration in `test/visual_validation.jl` and `test/test_gecatsim_validation.jl` to enable cross-simulator comparison.

**Current State**:
- `run_gecatsim()` returns `(nothing, nothing)` with warning
- Infrastructure for comparison ready
- Optional dependencies configured in Project.toml

**Required Work**:
1. Map BasisSimulator phantom → GECATSIM cfg.phantom
2. Map BasisSimulator geometry → GECATSIM cfg.scanner
3. Call GECATSIM forward projection
4. Call GECATSIM FDK reconstruction
5. Return sinogram and reconstruction arrays

**Python API Research Needed**:
- GECATSIM phantom definition syntax
- Scanner geometry configuration
- Running simulation and extracting results
- Array format conversion (Python → Julia)

**References**:
- GECATSIM documentation: https://github.com/xcist/main
- De Man et al. (2022) GECATSIM paper
- Example GECATSIM scripts in their repo

**Estimated Effort**: 4-6 hours
**Priority**: HIGH (required for publication validation)

---

## Medium Priority

### 3. Add Custom Materials Support

Create `src/Physics/Materials.jl` module for project-specific material definitions separate from XrayAttenuation.jl.

**Benefits**:
- Cleaner separation from upstream XA.jl
- Easier to maintain custom Gammex materials
- Can include tissue-mimicking materials for XCAT phantom

**Implementation**:
```julia
module Materials
    import XrayAttenuation as XA

    # Re-export standard materials
    using XrayAttenuation: Materials as StandardMaterials

    # Define custom materials
    const Ca_50 = XA.Compound(...)
    const Ca_100 = XA.Compound(...)
    # ... etc

    export Ca_50, Ca_100, ...
end
```

**Estimated Effort**: 1-2 hours
**Priority**: MEDIUM (nice-to-have organization)

---

### 4. Expand Physics Tests for Gammex Materials

Once materials are defined, add comprehensive tests:
- HU values match literature/manufacturer specs
- Attenuation increases monotonically with concentration
- Energy dependence (kVp = 80, 100, 120, 140)
- Material separability (can we distinguish Ca vs I?)

**Estimated Effort**: 2-3 hours
**Priority**: MEDIUM (after materials defined)

---

## Low Priority

### 5. Automated Visual Validation Metrics

Add automated pass/fail checks to `test/visual_validation.jl`:
- SSIM > 0.95 for reconstructions
- RMSE < 5% for HU values
- Could integrate into CI/CD

**Estimated Effort**: 2 hours
**Priority**: LOW (nice-to-have for CI)

---

### 6. Visual Validation Animation

Generate MP4 or GIF showing slice-by-slice comparison through phantom volume.

**Estimated Effort**: 1-2 hours
**Priority**: LOW (publication supplementary material)

---

## Completed

- Repository cleanup and organization
- Professional README.md and CLAUDE.md documentation
- Visual validation infrastructure (test/visual_validation.jl)
- Known limitations documented
- Git commits with clear messages

---

## Notes

This TODO list tracks technical debt and pending implementation work. For long-term roadmap and research objectives, see `CLAUDE.md` Development Roadmap section.

Last Updated: 2026-01-08
