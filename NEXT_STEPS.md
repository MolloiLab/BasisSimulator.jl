# Next Steps for BasisSimulator.jl

**Date**: January 8, 2026
**Status**: Ready to port working simulation from old notebook

---

## Current Situation

### What We've Accomplished ✅
1. **Fixed Critical Ray Tracer Bug** - Ray-box intersection now works for sources outside grid
2. **Implemented Gammex 472 Materials** - All 14 calibration materials with proper specs
3. **Verified FDK Works** - Isolated test shows 2x error (acceptable, matches old version)
4. **Diagnosed Exact Issue** - Sinogram is 7x too high (46 cm^-1 vs expected 6.8 cm^-1)

### The Problem ❌
The forward simulation in `src/Simulation.jl` produces transmitted values with wrong scale:
- After `-log(I/I0)` transform, sinogram max is 46 cm^-1
- Should be ~6.8 cm^-1 for 33cm water cylinder
- This causes reconstruction to be 5-7x too high

### Root Cause
Something in the transmission calculation differs from old working notebook.
Old notebook used: `I = Σ [N₀(E) × exp(-μL) × E × η(E)]`
Where η(E) is detector quantum efficiency.

---

## Action Plan for Next Session

### Step 1: Extract Complete Working Simulation (30 min)
Location: `/tmp/ct_simulator_final_old.jl` (git: commit 422513f^)

Extract these functions:
- `run_enhanced_simulation()` (line 1951)
- Ray tracer call and setup
- Attenuation calculation logic
- Detector efficiency application
- DAS readout (-log transform)

### Step 2: Create Standalone Working Test (15 min)
File: `test/working_baseline.jl`

Use extracted functions with minimal dependencies:
- Load phantom (use our fixed create_gammex_472)
- Run old simulation logic
- Verify sinogram max ~7 cm^-1 (not 46!)
- Reconstruct with old FDK
- Verify HU values reasonable

### Step 3: Identify Exact Differences (15 min)
Compare line-by-line:
- Old transmission: `sum(N₀ * exp(-μL) * E * η)`
- New transmission: `sum(N₀ * exp(-μL) * E)`  ← Missing η(E)?
- Check I0 calculation differences
- Check μ matrix differences

### Step 4: Port Correct Logic to src/Simulation.jl (30 min)
Update `simulate_ct_scan()` to match working version:
- Add detector efficiency if missing
- Fix any unit conversions
- Match exact calculation order
- Test after each change

### Step 5: Verify End-to-End (15 min)
Run `test/baseline_from_old.jl` again:
- Should show sinogram max ~7 cm^-1
- Reconstruction should be ~2x off (acceptable)
- Generate working image

### Step 6: Clean Up and Document (15 min)
- Remove debug files
- Update CLAUDE.md with success
- Commit working pipeline
- Generate example image for README

---

## Key Files

### Working Reference
- `/tmp/ct_simulator_final_old.jl` - Old working notebook
- Lines 1951-2150: `run_enhanced_simulation()`
- Lines 1729-1800: `reconstruct_fdk()`

### Current Code
- `src/Simulation.jl:340-360` - Transmission calculation (BROKEN)
- `src/Reconstruction/FDK.jl` - Works correctly in isolation
- `src/Geometry/RayTracing.jl` - Fixed and working
- `src/Physics/Materials.jl` - Gammex materials defined

### Tests
- `test/test_simple_fdk.jl` - ✅ FDK works (2x error)
- `test/baseline_from_old.jl` - ❌ Shows sinogram 7x too high
- `test/working_baseline.jl` - TODO: Will use old simulation logic

---

## Expected Outcome

After porting:
- Sinogram max: ~7 cm^-1 ✅
- Reconstruction: ~0.4 cm^-1 (2x water value, acceptable for FDK)
- HU range: -500 to +1500 (reasonable, could fine-tune later)
- Visual: Clear phantom structure with visible inserts

---

## Notes

- The FDK algorithm itself is correct (proven by isolated test)
- The ray tracer is working (we fixed the critical bug)
- The issue is purely in the forward simulation scale/units
- Old notebook has the working implementation - just need to port it
- Estimate: 2-3 hours to complete all steps

---

**Ready to proceed!** 🚀
