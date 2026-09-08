# BasisSimulator.jl v0.14.0 — Architectural Map of the Imaging Pipeline

Root: `/Users/daleblack/Documents/dev/MolloiLab/BasisSimulator.jl`. All references are `file:line` relative to that root.

Purpose: foundation for a re-implementation as a pure-functional, Reactant.jl-compiled, Enzyme-differentiable pipeline, using the current code as a numerical oracle.

---

## 0. Seven corrections to the brief, up front

These change what a re-implementation targets.

1. **`:joseph` does not exist.** `_validate_projector` accepts only `:dd`, `:dd_fast`, `:siddon` at `src/projection/select_projector.jl:44-46`. Every "joseph" hit in the repo is a bibliography citation to Joseph & Spital 1978 at `src/correction/bhc_sinogram.jl:50` and `src/projection/polychromatic.jl:110`.
2. **The published n-channel VMI estimator is notebook-only.** `grep -rn "nchannel" src/` returns zero hits. `src`'s `apply_cong!` is a hard-wired 2-channel estimator and reproduces only notebook 09. The numbers in notebooks 03, 04, 07, 08, and 12 come from `nchannel_profile_tile!`, defined inside `docs/notebooks/04_pcct_vmi.jl:566` and mirrored per notebook.
3. **`src/detector/physics_pipeline.jl` contains no orchestration.** It is only the `PhysicsConfig` struct at `:37` and its keyword constructor at `:57`. Effect ordering lives in the two `simulate!` methods at `src/api/driver.jl:117` and `:423`, plus `_apply_physics_no_noise!` at `src/projection/polychromatic.jl:271`.
4. **FBP filtering is a spatial-domain convolution, not an FFT ramp.** The hot kernel is an explicit tap loop at `src/reconstruction/core/filtering.jl:514-537`. The only FFT runs once at workspace construction to window the kernel, at `:266` and `:281`. No FFT plan is created, cached, or stored in any workspace.
5. **Units are cm throughout**, despite `mm` in many docstrings. Verified at `src/geometry/scanner.jl:413` (`pixel_row_size # cm`), `:419` (`fov # cm`), `src/object/phantom.jl:94`, `src/object/attenuation.jl:34` (`ustrip(u"cm^-1", …)`). Stale `mm` docstrings at `src/projection/siddon.jl:81`, `:162`, `src/projection/polychromatic.jl:81`, `src/reconstruction/fbp/fdk.jl:72-79`.
6. **No atomics anywhere in `src/`.** `grep -rn "Atomix|@atomic"` returns zero. Every kernel is an output-stationary gather. Design intent stated at `src/projection/dd.jl:34-42` and `src/projection/dd_transpose.jl:279-280`.
7. **Only 6 of 12 notebooks have a coded PASS/FAIL cell.** Notebooks 01, 03, 04, 07, 08, 12. The rest state numeric targets in prose only.

---

## 1. Module layout

`src/BasisSimulator.jl` is 343 lines of `include` statements in dependency order. 61 source files, roughly 20k lines.

| Directory | Files (lines) | Owns |
|---|---|---|
| `src/api/` | `options.jl` 249, `workspace.jl` 1134, `driver.jl` 1640 | The five-struct API, all four workspaces, `simulate!`, `reconstruct!`, spectrum resolution |
| `src/geometry/` | `scanner.jl` 817, `affine.jl` 243 | `Scanner`, `CTGeometry`, `is_helical`, `is_arc`, phantom/world/recon affines, `resample_to_recon` |
| `src/object/` | `phantom.jl` 607, `materials.jl` 143, `attenuation.jl` 247 | `Phantom`, `create_gammex_472`, region labels, `compute_μ_at_energy`, `to_hounsfield` |
| `src/phantoms/` | `xcat_artifacts.jl` 397 | XCAT download + XCIST voxelized parsing |
| `src/projection/` | `siddon.jl` 1437, `dd.jl` 777, `dd_fast.jl` 446, `dd_transpose.jl` 510, `select_projector.jl` 91, `polychromatic.jl` 659 | All forward projectors, the exact DD adjoint, the polychromatic driver |
| `src/source/` | `spectrum.jl` 348, `bowtie_filter.jl` 411, `heel_effect.jl` 329, `focal_spot.jl` 557, `protocol.jl` 275 | `CTProtocol`, spectrum loading + filtering, bowtie, heel, focal spot |
| `src/detector/` | `photon_counting.jl` 1226, `detector_efficiency.jl` 621, `scatter.jl` 789, `pcct/mc_response.jl` 420, `pcct/mc_pileup.jl` 413, `optical_crosstalk.jl` 126, `detector_lag.jl` 166, `fill_factor.jl` 114, `physics_pipeline.jl` 87 | PCCT forward projection + binning, noise, MC DRM + pileup, all detector effects |
| `src/correction/` | `bhc_sinogram.jl` 796, `calibration.jl` , `bhc_image_domain.jl` 140, `radial_cupping.jl` 173, `radial_capping_basis.jl` 139, `pcct_pileup_correction.jl` 126 | Water BHC, air calibration, deprecated cupping + image BHC |
| `src/reconstruction/core/` | `backprojection.jl` 725, `filtering.jl` 573 | Voxel-driven backprojection, ramp kernel + windows |
| `src/reconstruction/fbp/` | `fdk.jl` 490, `wfbp_helical.jl` 267 | FDK, helical rebinned WFBP |
| `src/reconstruction/hybrid_ir/` | `hybrid_ir.jl` 159 | `HIRParams` lookup table only; the loop is in the driver |
| `src/reconstruction/ir/` | `utils.jl` 297 | Huber penalty, projection + image weights, ordered subsets |
| `src/reconstruction/vmi/` | 13 files; `mono_plus.jl` 590, `rwls.jl` 497, `pwls.jl` 424, `cong.jl` 368 largest | Cong, CMV, PWLS, RWLS, GPU Brent, VMI synth, Mono+, clinical calibrations |
| `src/reconstruction/workspace/` | `memory_budget.jl` 256 | Tiling, OOM retry, backend memory reflection |
| `src/denoising/` | `sino_sfjsd.jl` 546, `acnr.jl` 410, `sino_svd.jl` 395, `rskr.jl` 342, `median_z.jl` 87 | All denoisers |
| `src/spectral/` | `pcct_spectral.jl` 648 | K-edge, effective Z, multi-material decomposition |

### Top-level exports

There is no single export block. 118 `export` statements are scattered across files. `src/BasisSimulator.jl:48` re-exports `XA` (XrayAttenuation), which notebooks access exclusively as `BS.XA` — never `import XrayAttenuation`. `src/BasisSimulator.jl:250-251` exports the memory-budget helpers.

Export sites by file (the complete list):

```
src/BasisSimulator.jl:48                 XA
src/BasisSimulator.jl:250-251            backend_memory_snapshot, release_backend!,
                                         estimate_pcct_workspace_bytes, check_pcct_workspace_budget
src/api/options.jl:7                     SimOptions, ReconOptions
src/api/driver.jl:7-15                   simulate!, reconstruct!, add_system_noise_floor!,
                                         compute_detector_I0, build_physics_config,
                                         resolve_source_spectrum_without_bowtie,
                                         resolve_source_spectrum_with_bowtie,
                                         resolve_source_spectrum_full, apply_bowtie_to_spectrum
src/api/workspace.jl:11-14               PCCTWorkspace, create_workspace, EICTWorkspace,
                                         create_eict_workspace, FDKReconWorkspace,
                                         create_fdk_recon_workspace, HIRReconWorkspace,
                                         create_hir_recon_workspace
src/api/workspace.jl:883                 calibrate_pcct_poly_bhc
src/geometry/scanner.jl:465,476,496,814,817   is_helical, is_arc, required_axial_detector_rows,
                                              Scanner, CTGeometry
src/geometry/affine.jl:243               phantom_to_world_affine, recon_to_world_affine,
                                         resample_to_recon
src/object/phantom.jl:602-607            RegionLabel, REGION_* (15 constants), Phantom, compute_μ,
                                         create_gammex_472, create_phantom_from_mask,
                                         compact_materials
src/object/materials.jl:142-143          get_material, get_region_materials
src/object/attenuation.jl:245-247        compute_μ_at_energy, compute_mass_μ_at_energy, μ_to_HU,
                                         HU_to_μ, get_reference_μ_water, to_hounsfield, get_basis_mu
src/phantoms/xcat_artifacts.jl:392       xcat_phantoms, xcat_default_materials, xcat_citation, …
src/projection/siddon.jl:120-121         siddon_forward_project!, siddon_forward_project,
                                         siddon_fused_poly_project!, siddon_fused_spectral_project!
src/projection/dd.jl:67-68               dd_forward_project!, dd_forward_project,
                                         dd_fused_poly_project!, dd_fused_spectral_project!
src/projection/dd_fast.jl:33             dd_fast_fused_poly_project!, dd_fast_fused_spectral_project!
src/projection/dd_transpose.jl:5         dd_backproject!
src/source/spectrum.jl                   (no export; loaded via driver)
src/source/bowtie_filter.jl:405-411      BowtieFilter, bowtie_filter_head, bowtie_filter_none,
                                         load_catsim_bowtie, load_builtin_bowtie,
                                         resolve_bowtie_filter, interpolate_thickness,
                                         compute_bowtie_attenuation_spectral, get_bowtie_mu
src/source/heel_effect.jl:32-34          HeelEffect, default_heel_effect, heel_effect_none,
                                         apply_heel_effect!, apply_heel_effect
src/detector/scatter.jl:678-687          ScatterModel, create_scatter_kernel_spatial,
                                         geometry_aware_scatter_model, estimate_phantom_diameter_cm,
                                         estimate_scatter_field!, compute_scatter_energy_weights,
                                         inject_scatter!, inject_scatter_bins!,
                                         compute_scatter_bin_weights
src/detector/detector_efficiency.jl:614-620  DetectorEfficiency, DetectorEfficiencyMode,
                                             BEER_LAMBERT, MC_LUT, detector_efficiency_gemstone,
                                             detector_efficiency_ufc, get_scintillator_mu,
                                             get_gemstone_mc_efficiency, get_ufc_mc_efficiency,
                                             compute_eid_efficiency_vector,
                                             GEMSTONE_MC_EFFICIENCY_LUT, UFC_MC_EFFICIENCY_LUT,
                                             UFC_FLASH_MC_EFFICIENCY_LUT
src/detector/photon_counting.jl:1218-1226   DetectorMaterialPCCT, CDTE/CZT/SI_MATERIAL,
                                            PhotonCountingDetector, EnergyResolvedSinogram,
                                            n_energy_bins, pcct_forward_project,
                                            get_pcct_detector_info, quantum_efficiency,
                                            quantum_efficiency_vector, spatial_bin!,
                                            apply_pcct_noise!
src/detector/pcct/mc_response.jl:381,418-420  compute_mc_count_moments, MCResponseData,
                                              load_mc_response, mc_cumulative_to_bins,
                                              compute_mc_drm, mc_drm_summary, default_mc_drm_path
src/detector/pcct/mc_pileup.jl:413       PileupResult, simulate_pulse_train, compute_mc_pileup_matrix
src/detector/optical_crosstalk.jl:123-126   OpticalCrosstalkModel, optical_crosstalk_typical,
                                            create_optical_crosstalk_kernel, apply_optical_crosstalk!
src/detector/detector_lag.jl:163-166     LagModel, lag_gadox, compute_lag_coefficients, apply_lag!
src/detector/fill_factor.jl:111-114      FillFactorModel, fill_factor_standard,
                                         effective_fill_factor, apply_fill_factor!
src/detector/physics_pipeline.jl:85-86   PhysicsConfig, default_physics_config
src/correction/bhc_sinogram.jl:71-75     BHCPolynomial, BeamHardeningCorrection,
                                         TwoMaterialBHCPerColumn, calibrate_bhc, apply_bhc!,
                                         calibrate_bhc_water, apply_bhc_water,
                                         calibrate_bhc_two_material, apply_bhc_two_material,
                                         bhc_spectrum_per_column, compute_polychromatic_μ_water
src/correction/bhc_image_domain.jl:31    apply_bhc_image_domain           [DEPRECATED]
src/correction/radial_cupping.jl:15-16   apply_radial_cupping_correction! [DEPRECATED],
                                         measure_radial_cupping
src/correction/radial_capping_basis.jl:139   apply_radial_capping_basis!
src/correction/pcct_pileup_correction.jl:126 apply_pcct_pileup_correction!
src/reconstruction/core/backprojection.jl:21 backproject!, backproject
src/reconstruction/core/filtering.jl:19-22,341  filter_sinogram!, filter_sinogram, FilterType,
                                                RampFilter, SheppLoganFilter, CosineFilter,
                                                HammingFilter, HannFilter, StandardFilter,
                                                SoftFilter, BoneFilter, CustomFilter,
                                                create_spatial_kernel, filter_from_symbol
src/reconstruction/fbp/fdk.jl:125        fdk_reconstruct, apply_fov_mask!
src/reconstruction/fbp/wfbp_helical.jl   (NO export — module-internal)
src/reconstruction/ir/utils.jl:14-17     PenaltyType, HuberPenalty, compute_huber_penalty,
                                         compute_huber_gradient!, compute_projection_weights,
                                         compute_image_weights, create_ordered_subsets,
                                         create_subset_geometry, extract_subset_sinogram
src/reconstruction/hybrid_ir/hybrid_ir.jl:31  get_hir_params, HIRParams
src/reconstruction/vmi/basis.jl:107      p_photoelectric, q_compton, water_basis_constants, …
src/reconstruction/vmi/roots_kernels.jl:197   brent_solve
src/reconstruction/vmi/cmv.jl:109        apply_cmv!, apply_cmv
src/reconstruction/vmi/pcct_basis.jl:211 pcct_effective_spectrum, pcct_rwls_basis, …
src/reconstruction/vmi/vmi_synth.jl:82,189    synth_vmi_hu, synth_vmi_sino_domain
src/reconstruction/vmi/image_domain_decomp.jl:225,390  fit_ding_coeffs, …, eval_cal, apply_cal!,
                                                       apply_cal, synth_vmi_2basis, decomp_2basis
src/reconstruction/vmi/phantom_mask.jl:169    resample_phantom_mask_to_recon, erode_mask_2d,
                                              erode_mask_3d
src/reconstruction/vmi/clinical_calibrations.jl:229,323,396  GE_REVOLUTION_* cal tables,
                                                             de_vmi_cal_for, de_vmi_cal_2basis_for
src/denoising/acnr.jl:285-286,410        apply_acnr!, apply_acnr, apply_image_acnr!,
                                         apply_image_acnr, apply_acnr_kalender!
src/denoising/sino_svd.jl:394-395        apply_sino_svd_denoise(!), apply_sino_svd_denoise_bilateral(!)
src/denoising/sino_sfjsd.jl:546          apply_sino_sfjsd_denoise
src/denoising/median_z.jl:87             apply_median_z, apply_median_z!
src/denoising/rskr.jl:342                apply_rskr, mad_haar_σ, joint_bf_2ch_gpu!, joint_bf_4ch_gpu!
```

Note `src/projection/select_projector.jl` and `src/projection/polychromatic.jl` export nothing (explicit comment at `polychromatic.jl:124-127`); they are internal dispatch shims.

---

## 2. The five-struct public API

### 2.1 `Phantom` — `src/object/phantom.jl:136`

```julia
struct Phantom{T<:Unsigned, M<:AbstractArray{T,3}, Mat}
    mask::M                        # GPU or CPU, UInt8 label volume (nx,ny,nz)
    materials::Mat                 # Vector{XA.Material} or Dict{Int,XA.Material}
    voxel_size::NTuple{3,Float64}  # cm, host
    origin::NTuple{3,Float64}      # cm, host
    extent::NTuple{3,Float64}      # cm, host
end
```

`mask` is the **only device array** in the whole five-struct API. Two constructor arities are live in notebooks: 3-positional `(mask, materials, voxel_size)` (nb 02, 05, 10, 11) and 5-positional (nb 01, 03, 04, 06, 07, 08, 09, 12). Both must survive. `materials` is sometimes indexed `materials[Int(label)+1]` (nb01 `:961`, nb03 `:1105`, nb09 `:1180`) and sometimes a `Dict{Int, XA.Material}` (nb07 `:322-337`).

### 2.2 `Scanner{T}` — `src/geometry/scanner.jl:123`

32 fields, **all host scalars** except `energy_thresholds::Vector{T}`. Distances in **mm** (this struct is the exception to the cm rule).

| Group | Fields |
|---|---|
| Geometry | `source_to_isocenter`, `source_to_detector` |
| Detector array | `detector_rows::Int`, `detector_cols::Int`, `detector_row_size`, `detector_col_size`, `detector_row_offset`, `detector_col_offset` |
| Source | `focal_spot_width`, `focal_spot_length`, `target_angle` |
| Gantry | `gantry_rotation_time`, `scan_diameter`, `gantry_aperture` |
| Filters | `flat_filter_material::Symbol`, `flat_filter_thickness`, `bowtie_filter::Symbol` |
| Detection | `detector_material::Symbol`, `detector_depth`, `fill_factor_row`, `fill_factor_col`, `detection_gain`, `electronic_noise` |
| PCCT | `detector_type::Symbol`, `n_energy_bins::Int`, `energy_thresholds::Vector{T}`, `energy_resolution`, `charge_sharing_fwhm`, `dead_time_ns`, `pixel_mode::Symbol` |
| Native dexel | `native_dexel_col_mm`, `native_dexel_row_mm`, `binning_factor::Int` |
| Shape | `detector_shape::Symbol` (`:arc` default, `:flat` for C-arm) |

Constructor kwargs + defaults documented at `:188-211`.

### 2.3 `CTProtocol` — `src/source/protocol.jl:43`

9 host scalars:

```julia
struct CTProtocol
    mA::Float64
    kVp::Float64
    views::Int
    rotation_time::Float64
    n_rotations::Float64
    collimation_mm::Union{Float64,Nothing}
    anode_angle::Int                                    # 8 or 10 (IPEM tables)
    additional_filters::Vector{Tuple{String,Float64}}   # (material, thickness_mm)
    pitch::Union{Float64,Nothing}                       # nothing = axial
end
```

`pitch` is IEC 60601-2-44: table feed per rotation ÷ total active collimation. Table travel = `pitch × collimation × n_rotations`.

### 2.4 `CTGeometry` — `src/geometry/scanner.jl:406`

**Entirely `Float64` host data. No device arrays.**

```julia
struct CTGeometry
    SAD::Float64                       # cm
    SDD::Float64                       # cm
    n_angles::Int
    n_rows::Int
    n_cols::Int
    pixel_size::Float64                # cm at isocenter
    pixel_row_size::Float64            # cm at isocenter
    angles::Vector{Float64}            # radians
    source_positions::Matrix{Float64}  # [3, n_angles]
    detector_centers::Matrix{Float64}  # [3, n_angles]
    detector_u::Matrix{Float64}        # [3, n_angles]
    detector_v::Matrix{Float64}        # [3, n_angles]
    fov::NTuple{3,Float64}             # cm
    pitch::Float64
    table_feed::Float64                # cm per rotation; 0.0 = axial
    detector_shape::Symbol             # :arc | :flat
end
```

Docstring at `:382-383`: positions are precomputed at construction **specifically to enable Reactant/XLA compilation with no runtime trigonometry**. Two backward-compatible positional constructors at `:428` (14-field, defaults axial+flat) and `:445` (16-field, defaults flat).

Dispatch predicates: `is_helical(geom) = geom.table_feed != 0.0` (`:463`), `is_arc(geom) = geom.detector_shape === :arc` (`:474`).

Coordinate system (`:401-405`): X left-right, Y anterior-posterior with source starting at `-SAD` on the Y axis, Z inferior-superior, rotation about Z counter-clockwise viewed from above.

### 2.5 `SimOptions` — `src/api/options.jl:66`

16 host scalar fields. `fidelity` is consumed in the constructor for preset lookup and **not stored**.

| Group | Fields |
|---|---|
| Physics (7) | `use_fill_factor`, `use_detector_efficiency`, `use_scatter`, `use_optical_crosstalk`, `use_focal_spot`, `use_noise`, `use_lag` |
| Signal chain | `use_heel_effect` |
| PCCT (5) | `use_pcct_pileup`, `use_pcct_pileup_correction`, `use_pcct_scatter`, `use_pcct_scatter_correction`, `pcct_noise_reduction::Float64` |
| General (3) | `seed::Union{Int,Nothing}`, `detector_efficiency_mode::Symbol`, `projector::Symbol` |

Presets at `:152-172`:

| Effect | `:eict` | `:pcct` |
|---|---|---|
| fill_factor, detector_efficiency, scatter, noise, heel_effect | true | true (scatter → `use_pcct_scatter`) |
| optical_crosstalk | false | false |
| focal_spot | true | **false** |
| lag | true | **false** (direct-conversion has no afterglow) |
| pcct_pileup | false | **true** |
| pcct_pileup_correction, pcct_scatter_correction | false | false |

Defaults: `seed = 42` (`:129`), `projector = :dd_fast` (`:131`), `detector_efficiency_mode = :auto` (`:130`), `pcct_noise_reduction = 0.0` clamped to `[0,1]` at `:197`.

**Validation doctrine** recorded at `:42-48`: any nonzero `pcct_noise_reduction` leaves the strict Poisson count model, so HU-accuracy claims must hold at 0.0.

### 2.6 `ReconOptions` — `src/api/options.jl:221`

```julia
struct ReconOptions
    matrix_size::NTuple{3,Int}
    fov_cm::Float64
    z_cm::Union{Float64,Nothing}   # nothing = auto from detector coverage
end
```

Defaults `(512,512,64)`, `35.0`, `nothing` (`:246-248`).

### 2.7 Where the precomputed tables live

**None of the five structs carries a table.** Every table is built inside a workspace constructor or on demand.

| Table | Shape / eltype | Built at | Cached? |
|---|---|---|---|
| Source spectrum | `Vector{Float64}` pair; 234 bins @120 kVp, 274 @140 kVp, 0.5 keV grid | `src/source/spectrum.jl:263` `load_spectrum_unfiltered`, reads `src/spectrum/Anode{8,10}/{kVp}.TXT` | **No.** Re-read every call. Data-dependent length (zero-flux tails dropped at `:284-289`) |
| Legacy spectrum loader | 240 bins @120 kVp | `src/source/spectrum.jl:76` `load_spectrum`, reads `src/spectrum/tungsten_tar{angle}_{kVp}_filt.dat` | No. Still called from `src/spectral/pcct_spectral.jl:82` |
| Filtered spectrum | `Vector{Float64}` | `src/source/spectrum.jl:316` `filter_spectrum`; one Unitful XCOM call per (material, energy); inverse-square `(750/sdd_mm)^2` applied only when `sdd_mm != 750.0` (float-equality branch at `:334`) | No |
| Bowtie thickness | `Matrix{Float64}` 888 rows × 4 materials `{Al, graphite, Cu, Ti}`, angles ∈ ±0.4796 rad, thickness cm | `src/source/bowtie_filter.jl:249` reads `src/bowtie/{large,medium,small}.txt` | **No.** Re-read at `driver.jl:846`, `:1002`, `workspace.jl:317`, `:680` |
| Bowtie transmission `B` | `(n_col, n_row, n_E)` `Float64`, dimensionless ∈ (0,1] | `src/source/bowtie_filter.jl:370` `compute_bowtie_attenuation_spectral` | Into `ws.bowtie_spectral` as `Float32` |
| Heel spectral | `(n_col, n_row, n_E)` `Float64`, **row-constant** (3× redundant) | `src/source/heel_effect.jl:271` `compute_heel_spectral` | Multiplied into `ws.bowtie_spectral` at `workspace.jl:685-689` |
| Per-ray spectrum `ŵ` | `(n_col, n_row, n_E)` `Float32`, per-ray normalized `Σ_E ŵ = 1` | `src/api/driver.jl:1022` (`resolve_source_spectrum_full`) | **No**, recomputed per call |
| EICT scintillator MC LUTs | 3 `const` NamedTuples, 140 `Float64` each | `src/detector/detector_efficiency.jl:156` (Gemstone), `:215` (UFC Force), `:291` (UFC Flash) | **Compile-time literals. Pure.** The cleanest artifact in the codebase |
| Scintillator μ data | `Dict{String,Tuple{Vector,Vector}}`, 9–42 points each | `src/detector/detector_efficiency.jl:50` | **`const` Dict mutated at module load** — alias keys inserted at `:95-99`, `:117-119`, `:123-124` |
| CdTe MC DRM | `Matrix{Float64}` `(200, n_bins)` (default), from a 7,551,360-byte `.jls` blob | `src/detector/pcct/mc_response.jl:167` `compute_mc_drm`; file `src/detector/pcct/cdte_response_v4.jls` loaded by `open(deserialize, …)` at `:89` | Into `ws.R`. **The loader itself is uncached** and called at `:171` and `:291` |
| MC response raw | `energies_keV` 140, `thresholds_keV` `[20,25,30,35,55,60,70,75,90]`, `R_total (140,9)`, `R_perpixel (140,3,3,9)`, `Var_total (140,9)`, `Cov_total (140,9,9)`, `Cov_pixpair (140,9,9,9,9)` | `MCResponseData` at `mc_response.jl:54-62` | No |
| Pileup matrix `S` | `Matrix{Float64}` `(n_bins, n_bins)`, lower triangular, column sums ≤ 1 | `src/detector/pcct/mc_pileup.jl:325`, 5000 MC trials, seed 42, at `workspace.jl:360-376` | `ws.pileup_S` |
| PCCT `W` matrix | `(n_energies_padded, n_bins)` `Float32` on GPU; `W[e,b] = I0_anchor·w[e]·η[e]·R[r_idx(e),b]·bt_center` | `src/api/workspace.jl:298-327` | `ws.W_matrix_gpu` |
| μ table | host `(n_regions, n_energies)` `Float32`; GPU `(n_regions, n_E_pad)` zero-padded to a multiple of 16 | `workspace.jl:221-226`/`:290-292` (PCCT), `:651-656`/`:660-663` (EICT) | `ws.μ_table`, `ws.μ_table_gpu` |
| `wη` vector | `(n_E_pad,)` `Float32` = `weights_norm .* η_vec`, zero-padded | `workspace.jl:673-676` | `ws.wη_gpu` |
| EICT air reference | `(n_col, n_row)` `Float32`; `air[c,r] = Σ_E w_norm[E]·T_source[c,r,E]·η[E]` | `workspace.jl:716-727` | `ws.bowtie_air_reference` (only when bowtie or heel present) |
| PCCT per-bin I0 | `Vector{Float64}` length `n_bins` = column sums of `W` | `workspace.jl:336-337` | `ws.I0_bins` **and** `ws.I0_bins_norm` — two distinct objects holding identical values |
| Log-factorial table | `Vector{Float64}` 21 entries (k=0..20) | `src/detector/photon_counting.jl:877` | Built at load |
| K-edge energies | `Dict{Symbol,Float64}` 5 entries | `src/spectral/pcct_spectral.jl:310` | `const` |
| Effect kernels | scatter 1D/2D, optical crosstalk 3×3, focal spot, lag coeffs (`n_frames = min(20, n_angles)`) | `workspace.jl:579-636`, `:384-396` | On the workspace |
| Noise constants | `ws.η_eff = Σ w_norm·η`; `ws.σ_e_photon = electronic_noise/(mean_E_keV·detection_gain)` | `workspace.jl:747-749` | On the workspace |

**Energy → DRM-row index map** appears 6× and must be reproduced bit-exactly:

```julia
r_idx = clamp(round(Int, (E - 1.0)/(kVp - 1.0) * (n_R - 1)) + 1, 1, n_R)
```

at `src/detector/photon_counting.jl:691`, `:766`, `src/detector/scatter.jl:421`, `src/api/workspace.jl:305`, `src/detector/pcct/mc_response.jl:355`, `:403`. It is a `round(Int)` — piecewise-constant in E, zero gradient with respect to the energy grid.

**MC pileup reproducibility defect.** `simulate_pulse_train` threads an explicit RNG (`mc_pileup.jl:107`), but the per-trial photon count comes from `Poisson_approx` at `:129`, which takes no RNG and pulls from the global stream at `:259` and `:267`. So `compute_mc_pileup_matrix(seed=42)` is **not** reproducible across sessions. `Poisson_approx` also switches to a Gaussian approximation at λ ≥ 30 (`:265-268`), unlike the exact `_poisson_sample` used for the actual noise.

---

## 3. Data flow

### 3.1 EICT simulation — `simulate!(ws::EICTWorkspace, phantom, protocol, sim_opts)` at `src/api/driver.jl:423`

Returns `nothing`. Mutates `ws.sinogram`, shape `(n_col, n_row, n_view)` `Float32`.

| Step | Lines | Operation |
|---|---|---|
| 1 | `:437` | `fill!(ws.sinogram, zero(T))` |
| 2 | `:438-455` | `_forward_project_poly!` (four internal paths — see 3.3) |
| 3 | `:463-474` | `_apply_physics_no_noise!` — fill factor, optical crosstalk, focal spot, lag |
| 4 | `:485-497` | Scatter field estimate + injection |
| 5 | `:511-512` | `I0_T = T(compute_detector_I0(geom, protocol, sum(ws.weights))) * ws.η_eff` |
| 6 | `:522-578` | Fused Gaussian noise + scatter subtraction + `-log(λ/I0)` |
| 7 | `:586-590` | `sino[idx] = exp(-clamp(sino[idx], T(-1), T(15)))` |
| 8 | `:593-611` | Air scan: `fill!(air, 1)`, multiply by the 2-D reference broadcast across views, divide |
| 9 | `:614` | `low_signal_correction_gpu!` (clamps ≤0 to 1e-10, `src/correction/calibration.jl:37-49`) |
| 10 | `:617-621` | `sino[idx] = -log(max(sino[idx], T(1e-10)))` |
| 11 | `:632-642` | Fill-factor offset cancellation `sino[idx] += T(log(ff_eff))` |

Steps 6–10 form a `-log` → `exp` → divide → `-log` round trip in `Float32`. That is a real precision sink a functional rewrite should fuse, **but the oracle must reproduce it**.

Note `ws.air_scan` is a full 3-D sinogram-sized buffer (`workspace.jl:444`) even though the reference is 2-D — pure waste in a functional rewrite.

BHC is deliberately **decoupled** and applied at notebook level (`:644`).

### 3.2 PCCT simulation — `simulate!(ws::PCCTWorkspace, …; capture_raw_counts=true)` at `src/api/driver.jl:117`

Returns a NamedTuple `(pcct_sino, I0_bins, pileup_S[, raw_counts])`.

| Step | Lines | Operation | Gate |
|---|---|---|---|
| 1 | `:133-163` | `pcct_forward_project` → per-bin `-log(N/I0_bins_norm)` | always |
| 2 | `:175-183` | `apply_focal_spot_blur!` per bin, **on log values, at binned resolution** | `config.focal_spot !== nothing` (preset OFF) |
| 3a | `:200-214` | Combine bins → counts → `-log(·/I0_total)` into `ws.combined` | scatter gate |
| 3b | `:216-221` | `estimate_scatter_field!(ws.tube_physics_scratch, combined, config.scatter)` | ″ |
| 3c | `:224-228` | `compute_scatter_energy_weights` + `compute_scatter_bin_weights` (CPU `Float64`) | ″ |
| 3d | `:231` | `inject_scatter_bins!` | `config.scatter !== nothing && use_pcct_scatter` |
| 4 | `:243-245` | Allocate `raw_from_noise` iff `capture_raw_counts && use_noise && !pileup_active` | — |
| 5 | `:246-255` | `apply_pcct_noise!` | `use_noise` |
| 6 | `:284-318` | Fused 4-bin pileup: counts → `r = S·c` (unrolled lower-triangular) → `-log(r/I0_truth)` | `use_pcct_pileup && pileup_S !== nothing` |
| 7 | `:326-335` | `raw_counts` capture | `capture_raw_counts` |
| 8 | `:342-344` | `apply_pcct_pileup_correction!` | `+ use_pcct_pileup_correction` (preset FALSE) |
| 9 | `:352-381` | Scatter correction: re-combine, re-estimate, `inject_scatter_bins!(…; subtract=true)` | `+ use_pcct_scatter_correction` (preset FALSE) |
| 10 | `:397-402` | Return | — |

Step 6 hard-errors unless `n_bins == 4` at `:287`.

**Truth-basis normalization** at `:312-315` is load-bearing: the post-pileup bin is `-log(recorded / I0_truth)`, so the identity `I0_b · exp(-bin) = recorded count` holds and every downstream count-domain stage stays valid. The rationale for rejecting recorded-basis normalization is at `:279-283` (it broke nb04's scatter subtraction and produced bin-2-only streaks).

Bin combine, BHC, HU calibration, and reconstruction are all **decoupled** (`:383-385`).

### 3.3 Forward projection

`_forward_project_poly!` at `src/projection/polychromatic.jl:423` has **four mutually exclusive paths**, selected in this order:

| Path | Condition | Lines | Behavior |
|---|---|---|---|
| **A** FUSED NTuple | `fused=true && ws_μ_table_gpu !== nothing` | `:465-491` | Default `fused=false` at `:456` — a 234-bin NTuple spills registers, 3.5× slower |
| **B** SINGLE-PASS | `ws_μ_table_gpu && ws_wη_gpu && projector===:dd_fast && n_mat ≤ 64` | `:497-519` | **The production default.** One `dd_fast_fused_poly_project!` call |
| **C** TILED K=16 | otherwise with tables | `:521-582` | 15 tiles for 234 bins; `copyto!` subsets per tile at `:547-551`; `-log`→`exp` round trip at `:569`, final `-log` at `:577` |
| **D** UNFUSED | no `ws_μ_table_gpu` | `:584-658` | One `create_μ_volume!` + one mono projection per energy; host scalar `weights_norm[e]`, `ws_η[e]` |

Math in all four paths:

```
p(col,row,view) = -log( Σ_E ŵ(E) · η(E) · B(col,row,E) · heel(col,row,E) · exp(-L_E) )
```

In paths A/B/C the spectrum weight is pre-fused as `wη = weights_norm .* η_vec` on the host (`workspace.jl:674`); in D `w` and `η_e` stay separate host scalars (`:623-624`). Bowtie is **always** a multiplicative per-(col,row,energy) transmission inside the exponentiated sum, never applied post-log.

#### Projector inventory

Sinograms `(n_col, n_row, n_view)`, volumes `(nx, ny, nz)`, generic in `T<:AbstractFloat`, `Float32` in practice. All geometry quantities are `Float64` on the host and narrowed to `T` at the kernel boundary.

| Projector | Entry | Structure |
|---|---|---|
| Siddon mono | `siddon_forward_project!` `src/projection/siddon.jl:457` | Ray-driven DDA, one thread per sinogram cell, `while t_current < t_exit && iter < nx+ny+nz+10` |
| Siddon poly | `siddon_fused_poly_project!` `siddon.jl:788` | `N_E` register accumulators, one DDA pass |
| Siddon spectral | `siddon_fused_spectral_project!` `siddon.jl:1174` | `K` energies × `n_bins`, accumulates `+=` into `outputs_flat` |
| DD mono | `dd_forward_project!` `src/projection/dd.jl:398` | Distance-driven footprint, 3-deep `while` nest, two inner loops data-dependent |
| DD arc row-tile | `_dd_forward_project_arc_rowtile4!` `dd.jl:450` | HIR-only arc fast path, 4 rows per thread; falls back when `!is_arc` |
| DD poly / spectral | `dd_fused_poly_project!` `dd.jl:540`, `dd_fused_spectral_project!` `dd.jl:662` | Legacy per-energy tiled kernels; the `:dd` reference |
| **DD fast poly** | `dd_fast_fused_poly_project!` `src/projection/dd_fast.jl:310` | Per-material path lengths, ONE volume walk |
| **DD fast spectral** | `dd_fast_fused_spectral_project!` `dd_fast.jl:381` | Same walk, `K` energies × `n_bins` |
| DD adjoint | `dd_backproject!` `src/projection/dd_transpose.jl:282` | Exact algebraic transpose, voxel gather |

`dd_fast` signatures are **byte-identical** to the `dd_fused_*` ones — that is the drop-in contract.

Dispatch shims (no exports) in `src/projection/select_projector.jl`:

| Shim | Line | Routes to |
|---|---|---|
| `_validate_projector` | `:44` | Throws unless `:dd`/`:dd_fast`/`:siddon`; `@warn maxlog=1` deprecation for `:dd` at `:47-52` |
| `_project_mono!` | `:58` | `siddon_forward_project!` \| `dd_forward_project!` |
| `_project_mono_hir!` | `:64` | siddon \| `_dd_forward_project_arc_rowtile4!` (if arc) \| `dd_forward_project!` |
| `_project_mono` | `:70` | allocating variants |
| `_backproject_mono!` | `:77` | `backproject!(…; weighted=false)` for `:siddon` (**an approximation, NOT an adjoint** — `:75-76`) \| `dd_backproject!` |
| `_project_fused_poly!` | `:82` | siddon \| `dd_fast_*` \| `dd_*` |
| `_project_fused_spectral!` | `:88` | same |

#### The `:dd_fast` algorithm, precisely

One thread owns **one detector cell** `(col, row, angle)`, decoded from the linear index at `dd_fast.jl:113-117`. It:

1. Loads its view's 9 geometry scalars at `:119-121` — `sp[1:3,angle]`, `dc[1:3,angle]`, `du[1,angle]`, `du[2,angle]`, `dv[3,angle]`. Note **only `dv[3]`**, i.e. detector `v = ẑ` is assumed (`dd.jl:57-58`).
2. Calls `_dd_cell_setup` (`dd.jl:194`) at `:123-126` to get the iso-plane footprint `[dXlo,dXhi] × [dZlo,dZhi]`, the obliquity normalization `norm`, and the traversal axis choice.
3. Initializes `plens = ntuple(_ -> zero(T), Val(M))` at `:128`, where **`M = size(μ_table_gpu, 1) = n_materials`**, not `N_E`.
4. Walks the volume ONCE at `:129-161`, and for each overlapped voxel:
   ```julia
   mat   = Int32(mask[ixv, iyv, ip]) + Int32(1)      # :151
   plens = _plen_accum(plens, mat, ox * oz * norm)   # :152
   ```
5. Converts to energies and reduces **in the same kernel body** at `:163-175`.

`_plen_accum` (`dd_fast.jl:56-59`) expands to an M-wide **branchless select chain**:

```julia
tuple(ifelse(mat == Int32(1), plens[1] + w, plens[1]),
      ifelse(mat == Int32(2), plens[2] + w, plens[2]), …)
```

Per-voxel cost is `O(M)` selects regardless of how many materials the ray actually hits — 64 `ifelse`s at the limit. In exchange, **the energy dimension leaves the inner loop entirely**.

Exact math per cell:

```
P_m     = Σ_{v : mat(v)=m}  ox_v · oz_v · norm                 # dd_fast.jl:152
L_e     = Σ_{m=1}^{M} P_m · μ_tbl[m, e]                        # _plen_line_integral, :62-68
I_total = Σ_{e=1}^{n_E} wη[e] · bt[col,row,e] · exp(-L_e)      # :166-174
sino[idx] = -log(max(I_total, T(1e-10)))                       # :175
```

This is a **reassociation** of the legacy `L_e = Σ_v w_v · μ[m_v, e]` (documented `dd_fast.jl:7-19`, `:39-51`). Footprint, bounds, and overlap weights are bit-for-bit the same code as `dd.jl`; only the summation order of a linear sum changes. Reported agreement with `:dd` is mean relative **5e-7** (`dd_fast.jl:22-23`).

Spectral variant at `:258-276`:

```
out[cell, b] += Σ_{e=ts}^{ts+K-1} W[e,b] · min(exp(-L_e), 1e30) · bt[col,row,e]
```

`_plen_line_integral` is re-evaluated **inside** the bin loop at `:265`, so with `K = n_E_padded` and `n_b = 4` the M-wide dot product runs `4·n_E` times per cell. That is the intentional registers-vs-FLOPs trade. The `1e30` clamp is applied **only** on the bowtie branch, matching `siddon.jl:1135,1137`.

In practice `:dd_fast` spectral is called **once** with `Val(n_energies_padded)` and `tile_start = 1` (`src/detector/photon_counting.jl:505-513`), then jumps past the tile loop via `@goto tiles_done` at `:512`.

**The ≤64-material limit** is `const _PLEN_MAX_MATERIALS = 64` at `dd_fast.jl:71`. It is a **register-budget** bound, not a correctness bound: the `NTuple{M,T}` must stay in registers (`:52-53`, `:70`, `:16-19`). Checked at four sites: `dd_fast.jl:325` (poly), `:399` (spectral), `polychromatic.jl:506` (host pre-check), `photon_counting.jl:502`. Above it, `_warn_dd_fast_fallback` fires (`:73-81`, `@warn maxlog=1`) and the call re-dispatches **verbatim** to `dd_fused_poly_project!` at `:327-334` / `dd_fused_spectral_project!` at `:401-409`. Test at `test/projection.jl:371` asserts the warning fires at 65 materials. `compact_materials` removes inactive table entries.

#### The DD adjoint

`dd_backproject!` at `dd_transpose.jl:282` **is a verified exact adjoint** of `dd_forward_project!` (mono only). Mechanism at `:276-280`: each voxel gathers the detector cells its footprint overlaps and reuses the **identical** `_dd_col_setup` / `_dd_row_setup` / `_dd_overlap` coefficient `ox · oz · norm` from the forward operator. Forward writes `norm · Σ ox·oz·vol` (`dd.jl:263,273`); transpose writes `Σ sino·ox·oz·norm` (`dd_transpose.jl:495`).

Verified in `test/projection.jl:31-95` for both `:flat` and `:arc`:

- Dot-product identity `⟨Ax,y⟩ == ⟨x,Aᵀy⟩` at `rtol = atol = 2.0e-11` (`:56-58`)
- **Brute-force matrix oracle**: every column of `A` formed explicitly, `A' * vec(y)` compared to the gather at `rtol = 2.0e-12` (`:78-93`)
- Determinism: repeated calls bit-identical (`:59-61`)
- The `active_z` tile-4 path reproduces the untiled result exactly (`:66-71`)

Two caveats that make it not the adjoint of `A` proper:
1. `circular_support = true` zeros voxels outside `(min(fov_x,fov_y)/2)^2` (`:88`, `:110-116`, `:316`, `:343-346`) — that is `(A∘M)ᵀ`.
2. The candidate-cell bounds at `:467-470`, `:159-160`, `_dd_arc_row_bounds` `:21-24` are **conservative-plus-one-safety-cell heuristics**, acknowledged at `:460-466`. The brute-force oracle is what pins this down.

Call sites: `select_projector.jl:79`; via that shim `src/api/driver.jl:1582`, `:1590` (HIR inner loop) and `src/reconstruction/ir/utils.jl:232`, `:234` (`V = 1/(Aᵀ·1)`).

**Siddon has no exact transpose.** `_backproject_mono!(:siddon, …)` routes to the legacy voxel-driven `backproject!` with `weighted=false`, explicitly labelled an approximation at `select_projector.jl:75-76`. `select_projector.jl:19-22` also records that Siddon "can ALIAS in severe beam-hardened regions".

**Siddon air-ray special case, poly only** (`siddon.jl:948-961`): when the ray misses the volume Siddon does **not** write 0 — it evaluates the full Beer-Lambert sum at `L = 0` so tiled accumulation recovers the true air intensity.

### 3.4 Detector response

**EICT** — no bins, no bin edges, no DRM. `η(E)` enters the spectral sum only, never as a separate sinogram-domain step (comment at `polychromatic.jl:317-319`). `compute_eid_efficiency_vector` (`detector_efficiency.jl:596`) routes by `model.mode == MC_LUT` **and** a material string-tuple membership test, falling through to Beer-Lambert `1 - exp(-μ(E)·d)` at `:606-607`.

**PCCT** — bin edges are the thresholds, half-open, top bin unbounded. Defaults `[20.0, 35.0, 55.0, 70.0]` (`photon_counting.jl:179`). For thresholds `[T₁…Tₙ]`: bin k = `[Tₖ, Tₖ₊₁)`, bin n = `[Tₙ, ∞)`.

```
N_b = Σ_E I0 · w(E) · η(E) · R(E,b) · exp(-∫μ dl)
```

Tiled path precomputes the product into `W_matrix_gpu[e,b]` (`workspace.jl:298-309`). Sequential path is an explicit loop at `photon_counting.jl:653-704` with three data-dependent `continue`s (`E < thresholds[1]`, `w < 1e-12`, `R_val < 1e-10`).

`η(E) = quantum_efficiency_vector(material, thickness_mm, energies) = 1 - exp(-μ_det(E)·d_cm)` (`:1195-1199`).

`spatial_bin!(output, input, factor)` at `:280` sums `factor × factor` native dexels **in count domain** before the log; output `(n_cols÷f, n_rows÷f, n_angles)`.

**PCCT bowtie is center-pixel only.** The driver never passes `ws_source_spectral` to `pcct_forward_project` (`driver.jl:133-163` has no such kwarg), so the per-(col,row) bowtie map is *not* applied on the PCCT forward path; only the scalar energy-hardening factor `bt_center` folded into `W` at `workspace.jl:322-327`.

**The 4→2 combine** lives at `src/reconstruction/vmi/pcct_basis.jl:130`:

```julia
combine_pcct_bin_counts!(out_bins, raw_bins, I0_bins, bin_groups; chunk_size)
```

Per group: `fill!(out_k, 0)` then `out_k += I0b·exp(-raw_b)` over `b ∈ grp` at `:160-165` — combine **in count domain**. Groups typically `[[1,2],[3,4]]` (see `clinical_calibrations.jl:92`, `:120`). Returns `[Float64(sum(I0_bins[b] for b in grp)) …]` at `:184`. There is a backend-mismatch streaming branch with a staging buffer at `:150-155`, `:172-180`.

**`debias` is NOT in `src/`.** `grep -rni "debias\|jensen"` over `src/` returns nothing. The Jensen / log-Poisson debias lives only in notebooks: `docs/notebooks/09_siemens_force_ufc_dual_source_vmi.jl:793-804` implements `out -= 1/(2·max(I0·exp(-out), 1))`. Notebook 12 at `:1009-1011` explicitly states the count-domain quasi-likelihood estimator uses **no** log debias. If the re-implementation needs a per-bin combined-unbiased debias it must be written fresh.

### 3.5 Noise injection

**PCCT — `apply_pcct_noise!` at `src/detector/photon_counting.jl:822`.** The hardest thing to differentiate in the codebase.

```julia
apply_pcct_noise!(sino::EnergyResolvedSinogram{T,A}, I0_bins::AbstractVector;
                  seed::Union{Nothing,Int} = nothing,
                  ws_noise_staging = nothing, ws_rng = nothing,
                  noise_reduction::Float64 = 0.0, raw_out = nothing) where {T,A}
```

- **CPU `MersenneTwister` only.** No GPU RNG anywhere in the package. Reseeded per call at `:836-841`. `ws.rng = MersenneTwister(0)` created at `workspace.jl:215`.
- **Full GPU → CPU → GPU round trip per bin** at `:850` and `:869`, staging through a plain `Array{T,3}` (`workspace.jl:45`).
- **Exact integer Poisson at all λ.** `_poisson_sample` at `:894`: λ < 1e-10 → 0; λ < 10 → Knuth sequential inversion with an unbounded `while true` (`:898-906`); λ ≥ 10 → Hörmann PTRS transformed rejection with a `while true` rejection loop (`:908-930`). **No Gaussian approximation anywhere.**
- Loop math, all in `Float64` (`:846-870`):
  ```
  Pass 1 (:853-860):  λ = I0_bins[b] * exp(-Float64(cpu_buf[idx]))
                      N = Float64(_poisson_sample(rng, λ))
                      if nr_scale != 1.0:  N = λ + nr_scale*(N - λ)
                      cpu_buf[idx] = T(N)
  raw capture (:861): raw_out === nothing || copyto!(raw_out[b], cpu_buf)
  Pass 2 (:864-866):  cpu_buf[idx] = T(-log(Float64(max(cpu_buf[idx], one(T))) / I0_bin))
  ```
- **`pcct_noise_reduction` blend**: with `nr_scale = 1 - noise_reduction` (`:844`), `N' = λ + (1−nr)·(N − λ)`. So `E[N'] = λ` (unbiased) and `Var[N'] = (1−nr)²·λ` — a variance shrink toward the Poisson mean, a vendor-denoising surrogate. Any `nr > 0` produces non-integer values and is no longer Poisson.
- **`raw_counts` semantics** (`driver.jl:103-115`, `:244-245`, `:326-335`): noise ON + pileup OFF gives the **verbatim integer draws before the floor at 1, with true zeros preserved**. Pileup ON or noise OFF gives `_capture_pcct_raw_counts` (`driver.jl:63`) which reconstructs `I0_b·exp(-bin)` — fractional recorded counts, or continuous expected counts λ.

**EICT — inline at `src/api/driver.jl:522-578`.** There is no `apply_eict_noise!`.

- CPU `MersenneTwister` (`ws.rng`, `workspace.jl:743`), reseeded **only if `sim_opts.seed !== nothing`** at `:527-529`.
- `randn!(ws.rng, ws.noise_rand_cpu)` into a host `Vector{T}` of length `n_elements`, then `copyto!` to GPU at `:530-531`. Electronic noise uses a **second** draw from the *same* RNG at `:536-537` — quantum and electronic streams are sequentially coupled.
- **Gaussian, not Poisson.** Two branches on `σ_e_photon > 0` at `:535`:
  ```
  λ_total   = I0v * exp(-sino[idx])
  λ_noisy   = λ_total + sqrt(max(λ_total, one(T))) * rg[idx]
  λ_noisy  += σ_e * eg[idx]
  λ_primary = do_sc ? max(λ_noisy - I0v*sf[idx]*sw, one(T)) : max(λ_noisy, one(T))
  sino[idx] = -log(λ_primary / I0v)
  ```
  All in `T = Float32` on device.
- `sf_kernel = has_scatter ? scatter_field_gpu : ws.physics_output` at `:520` is a **type-stability hack for Metal**: the dead branch still needs a real GPU buffer, because capturing `sf::Nothing` tripped GPUCompiler's `box_int64` path (comment `:513-519`).
- Post-recon `add_system_noise_floor!(vol, sigma_hu; seed)` at `:734` uses `MersenneTwister(seed + 7919)`. Documented as forbidden for PCCT (`:694-701`) and forbidden for FBP-vs-IR comparisons (`:675-681`).

### 3.6 Air calibration and log

**EICT**, `driver.jl:580-642`, all `Float32` on device, `eps = T(1e-10)` at `:583`. See the table in 3.1, steps 7–11. The `I/I0` division is at `:606-611`; the `-log` at `:617-621`.

**PCCT**, inside `pcct_forward_project` at `photon_counting.jl:592-598` (tiled) and `:723-729` (sequential):

```julia
ba[idx] = -log(max(ba[idx], T(1e-10)) / T(I0_bins_norm[b] * I0_scale))
```

with `I0_scale = use_native ? bf*bf : 1.0`. Because `I0_bins_norm` is the column sums of the forward kernel's own `W` matrix, `p_air ≡ 0` by construction (`workspace.jl:331-335`). The fallback `_compute_bin_I0` at `photon_counting.jl:755` disagrees by 13–26 % (documented at `workspace.jl:204-208`).

**Five different zero-floors coexist in one pipeline** and are numerically load-bearing:

| Floor | Sites |
|---|---|
| `1e-10` | `photon_counting.jl:715`, `scatter.jl:330`, `optical_crosstalk.jl:111`, `detector_lag.jl:151`, `calibration.jl:41`, `driver.jl:197`, `:289`, `pcct_pileup_correction.jl:96`, `polychromatic.jl:577`, `:654` |
| `1` count | `photon_counting.jl:865`, `scatter.jl:364`, `driver.jl:547`, `:549`, `:562`, `:574` |
| `clamp(sino, -1, 15)` | `driver.jl:588` |
| `min(proj, 20)` | `scatter.jl:328`, `:741`, `:774` |
| `clamp(exp_term, -700, 700)` | `heel_effect.jl:182`, `:212`, `:313` |
| `1e-12` weight skip | `photon_counting.jl:663`, `:763`, `workspace.jl:302`, `scatter.jl:418` |
| `1e-30` guard | `heel_effect.jl:318`, `driver.jl:495`, `:862` |
| `max(I0_bin, 1.0)` | `photon_counting.jl:770` |

### 3.7 Beam-hardening correction

**Calibration** — `calibrate_bhc_water(sim_opts, protocol; scanner, geom, order=5, max_path_cm=50.0, n_points=100, reference_energy_keV=nothing)` at `src/correction/bhc_sinogram.jl:583`. A **second method** at `:632` takes `(energies, w_per_column; reference_energy_keV)` and is what notebooks 09 and 12 use.

Pipeline:
1. `resolve_source_spectrum_full` → `(e, ŵ)`, the **full detected** spectrum: tube × filters × bowtie × heel × η (`:593-595`).
2. `bhc_spectrum_per_column(e, ŵ)` at `:482` takes the **center detector row** `mid_r = n_row÷2 + 1` and returns `Array{Float64,2}` of shape `[n_E, n_col]` (`:490-498`). **Row-direction bowtie variation is discarded by design** (`:477-479`) — this is the known heel-effect gap.
3. `ref_E` = fluence-weighted mean energy of the column-averaged spectrum when not given (`:601-606`).
4. `for c in 1:n_col` (`:610-616`), each calling `calibrate_bhc(e, w_col[:,c]; …)`:
   - `generate_water_calibration_curve` at `:184`: `paths = range(0, 50.0, length=100)`, `measured[i] = -log(max(Σ w_norm·exp(-μ·d), 1e-10))`, `true_values[i] = μ_water_ref · d`.
   - `fit_polynomial(x, y, order)` at `:218`: dense Vandermonde `V[i,j+1] = x[i]^j`, solved by **normal equations `(V'*V) \ (V'*y)`** at `:224` — `Float64`, CPU, LAPACK, 100 × 6.
5. `μ_water_ref = ustrip(u"cm^-1", XA.linear_attenuation_coeff(water, ref_E*u"keV"))` (`:617-618`).

For a 920-column scanner that is 920 independent 100×6 least-squares solves at calibration time, purely a function of `(e, ŵ)` — no data touched. It *is* a closed-form linear solve, so an Enzyme-friendly rewrite (QR/lstsq on the Vandermonde) is straightforward.

Default order is **5** via `calibrate_bhc_water`, **3** via bare `calibrate_bhc` (`:243`) — a real inconsistency.

**Application** — `apply_bhc!(sinogram, polys_per_col; ws_coeffs_gpu)` at `:296`. Validates `length(polys_per_col) == size(sinogram,1)` and that all orders match (else `error()`). Uploads to a `(order+1, n_col)` device matrix, then one kernel at `:328-340`:

```julia
col = ((idx - 1) % n_col) + 1        # relies on column-major (n_col, n_row, n_view)
p_corrected = coeffs[1, col]
p_power = one(T)
for i in 1:order
    p_power *= p
    p_corrected += coeffs[i+1, col] * p_power
end
sinogram[idx] = p_corrected
```

**Not Horner** despite the comment at `:266-268`. `Float32`. `order` is a runtime `Int` — needs `Val`-lifting for Reactant.

`apply_bhc_water(sinogram_raw, bhc)` at `:663` copies then delegates.

**Deprecated**: `calibrate_bhc_two_material` (`:380`, `:422`, `depwarn` at `:392`, `:433`), `apply_bhc_two_material` (`:707`, `depwarn` at `:714`) — the latter does an FDK round trip *inside* the correction and returns `Array(sino_out)`, forcing a GPU→CPU copy at `:795`. `apply_bhc_image_domain` (`bhc_image_domain.jl:73`, `depwarn` at `:84`) mutates `recon_μ` in place **despite having no `!` in the name**.

### 3.8 FBP and helical WFBP

`reconstruct!(ws::FDKReconWorkspace, sinogram, geom)` at `src/api/driver.jl:1236`. Four steps: copy → filter → backproject → FOV mask. **Filter identity, kernel length, `cutoff`, and volume size are all frozen at workspace creation** (`:1208-1211`). There are no runtime overrides on this hot path.

**Backprojection** — `src/reconstruction/core/backprojection.jl:640-664` (weighted branch):

```julia
AK.foreachindex(volume) do idx
    idx_0 = Int32(idx - 1)
    ix = (idx_0 % nx) + Int32(1);  idx_0 = idx_0 ÷ nx
    iy = (idx_0 % ny) + Int32(1);  iz = (idx_0 ÷ ny) + Int32(1)
    voxel_x = vol_min_x + (T(ix) - half) * voxel_size_x
    …
    volume[idx] = backproject_voxel(sinogram, voxel_x, voxel_y, voxel_z, …)
end
```

**Voxel-driven gather.** One thread per voxel, `volume[idx] = <scalar>` — a pure write, not an accumulate, so the preceding `fill!` is redundant. Three structurally identical `AK.foreachindex(volume)` blocks: helical `:614-637`, FDK-weighted `:640-664`, matched `:667-691`.

Interpolation is **bilinear on the detector `(col,row)`, nearest in view** (`:135-155`). There is no angular interpolation. Weights are formed **before** clamping, so the half-pixel skirt degenerates to clamp-to-edge replication. Bounds guard `col_f ∈ [0.5, n_cols+0.5]`, `row_f ∈ [0.5, n_rows+0.5]` — deliberately no row extrapolation (`:129-132`).

Geometry comes from four precomputed `(3, n_angles)` device matrices; the per-view ray math is computed **inline** at `:74-121` with separate `:arc` (`:110-116`) and `:flat` (`:118-120`) branches. `arc_det::Bool` and `dγ = pixel_size/SAD` are hoisted host constants (`:558-559`).

FDK weight is `SAD²/L²` at `:126-127`, pure geometry, no `p` dependence; final scale `π/n_angles` at `:564`. Note `w_acc` is accumulated but used only as a nonzero test — partial-coverage voxels are **not** renormalized, contradicting the comment at `:123-125`.

Index arithmetic is `Int32` throughout for Metal. Accumulators are `zero(T)` — **no `Float64` promotion anywhere**.

`backproject_voxel_matched` (`:375`, `weighted=false`) is the same geometry and bilinear code with `acc += val` — no distance weight, no `π/N`. Used only via `_backproject_mono!(:siddon, …)`.

**Filtering** — `filter_sinogram!` at `src/reconstruction/core/filtering.jl:461`. The hot kernel at `:514-537` is an explicit spatial tap loop with implicit zero-padding via an in-kernel bounds branch:

```julia
AK.foreachindex(sinogram) do idx
    acc = zero(T)
    for k in Int32(1):kernel_size
        src_col = col + (k - kernel_half - Int32(1))
        if src_col >= Int32(1) && src_col <= n_cols
            acc += sinogram[src_col, row, angle] * kernel[k]
        end
    end
    filtered[idx] = acc
end
copyto!(sinogram, filtered)
```

Kernel length `min(max(ceil(2·n_col·cutoff), 64) + parity, 2·n_col - 1)` at `:488-489`, forced odd. The comment at `:483-487` records that the old `n_col` clamp caused +8 HU capping. Cost is `O(n_col · n_row · n_view · 2·n_col)` — **the most expensive non-backprojection kernel**.

Ramp kernel at `create_spatial_kernel` `:120-155`: `h[0] = 1/(4Δ)`, `h[even≠0] = 0`, `h[odd] = -1/(π²k²Δ)`.

Two windowing mechanisms:
- **Spatial multiply**: `RampFilter` no-op `:158`, `SheppLoganFilter` `sinc(k/n)` `:163`, `CosineFilter` `:178`, `HammingFilter` `:190`, `HannFilter` `:201`.
- **Frequency-domain FFT round trip**: `_apply_catsim_freq_window!` at `:254-290` — manual fftshift, `fft` at `:266`, per-bin window with symmetric folding, `ifft` at `:281`. **Unplanned, uncached, CPU `Vector{Complex{T}}` only, construction-time only.** The apodization interpolation at `:228-243` is **piecewise LINEAR** despite the name, admitted at `:236-237`.

Clinical control points (`control_x = (0.0,0.25,0.5,0.75,1.0)` for all):

| Filter | `control_y` | Line |
|---|---|---|
| `StandardFilter` | `(1.0, 0.9338, 0.7441, 0.4425, 0.0531)` | `:295` |
| `SoftFilter` | `(1.0, 0.815, 0.4564, 0.1636, 0.0)` | `:302` |
| `BoneFilter` | `(1.0, 1.0485, 1.17, 1.2202, 0.9201)` | `:309` |
| `CustomFilter{N}` | user tuple pair | `:93-96`, `:313-315` |

**Per-basis apodization is a notebook-level `CustomFilter` choice, not a src feature.** The soft-iodine and sharp-water kernels are constructed at `docs/notebooks/12_siemens_flash_ufc.jl:1507-1514`:

```julia
nchannel_iodine_filter = BS.CustomFilter((0.0,0.25,0.5,0.75,1.0), (1.0, 0.40, 0.12, 0.03, 0.001))
nchannel_water_filter  = BS.CustomFilter((0.0,0.25,0.5,0.75,1.0), (1.0, 0.8744, 0.6003, 0.3031, 0.0266))
```

Same pair at nb03 `:1362` and nb07 `:1273`. The symbolic labels `:OriginalDualKvpSoft` / `:StandardSoftBlend` (nb12 `:1541`) are **display-only tags**, not dispatchable.

Cosine weighting at `cosine_weight!` `:385-432`, gated by `apply_cosine`, mutates the caller's array in place. Equiangular fan correction `(γ/sinγ)²` at `:371-383`, applied only when `is_arc(geom) && ray_spacing === nothing` (`:498-500`).

**There is no Parker weighting, no short-scan weighting, no redundancy code** in `filtering.jl` or `fdk.jl`. Fan redundancy is handled implicitly by `π/N` (circular) and the per-family `ΣW` partition (WFBP, `wfbp_helical.jl:230-232`).

**Helical dispatch** at three sites:
1. `src/reconstruction/fbp/fdk.jl:338-341` — allocating API → `wfbp_helical_reconstruct`
2. `src/api/driver.jl:1245-1263` — `reconstruct!(::FDKReconWorkspace)`, reuses `ws.filtered` as the rebin target
3. `src/api/driver.jl:1417-1428` — HIR FDK-init step

Path differences:

| Step | Circular FDK | Helical WFBP |
|---|---|---|
| Rebin | — | `_wfbp_rebin!` fan→parallel, `wfbp_helical.jl:59-124` |
| Cosine weight | yes | **no** (`apply_cosine=false`) |
| Equiangular `(γ/sinγ)²` | applied if `:arc` | **not** applied |
| Backprojection | `backproject_voxel`, `SAD²/L²`, `×π/N` | `_wfbp_backproject!`, **no `1/L²`**, `×Δβ` |
| Redundancy | full-scan `π/N` | per-half-turn-family `ΣWP/ΣW` |
| FOV mask | `fdk.jl:351` | **`wfbp_helical_reconstruct` does NOT mask**; the workspace path does at `driver.jl:1256` |

That last row is an **oracle discrepancy between the two helical entry points** — pick one deliberately.

The WFBP backprojection kernel at `wfbp_helical.jl:171-237` is the helical oracle:

```julia
fam = Int32(1)
while fam <= n_half                    # n_half = round(π/Δβ)
    sumW = zero(T); sumWP = zero(T)
    j = fam
    while j <= n_views                 # stride n_half → conjugate family
        θ = (T(j)-1)*Δβ;  t̂ = x*cosθ - y*sinθ;  s = t̂/R
        if s > -0.999 && s < 0.999
            γ = asin(s);  β = θ - γ
            z_s = z_start + feed*β/2π                    # helical z-ramp, :199
            denom = (x*sinθ + y*cosθ) + R*cos(γ)
            if denom > T(1e-3)
                v = (z - z_s)*(SDD/cosγ)/denom/prm       # wedge row coord, :205
                Wq = _wq_aperture(v/half_rows, q_plat)
                if Wq > 0
                    … bilinear (t,row) at fixed view j …
                    sumW += Wq;  sumWP += Wq * val
                end
            end
        end
        j += n_half
    end
    if sumW > T(1e-8);  acc += sumWP / sumW;  end          # family normalization
    fam += Int32(1)
end
volume[idx] = acc * Δβ
```

The Stierstorfer cos² aperture family, `_wq_aperture(q̂, Q)` at `backprojection.jl:177-187`:

```
W_Q(q̂) = 1                            |q̂| < Q
       = cos²((π/2)(|q̂|−Q)/(1−Q))     Q ≤ |q̂| < 1
       = 0                            |q̂| ≥ 1
```

`Q = 0.7` is the default at three sites (`backprojection.jl:527`, `wfbp_helical.jl:138`, `:257`), and the two `driver.jl` helical call sites do **not** pass it, so it is always 0.7.

**`backproject_voxel_helical` at `backprojection.jl:210` is unreachable from the pipeline.** Every production caller short-circuits to WFBP first. Its analytic normalization at `:295-334` (conjugate view `β* = β + π + 2γ`, `n_turns_k = ceil(1/pitch_eff)+1`) is kept for reference/oracle-diffing only.

### 3.9 HIR / OS-PWLS

`hybrid_ir.jl` is a **parameter table only** — the loop is `reconstruct!(::HIRReconWorkspace, …)` at `src/api/driver.jl:1404-1640`, exactly as `hybrid_ir.jl:6-7` states.

Parameter table at `hybrid_ir.jl:83-91`, all `Float32`:

```
#  %    λ       nepochs  δ        relax    band
(  0,  0.0f0,   0,       0.08f0,  0.5f0,   (0,0))
( 20,  1.0f0,   1,       0.08f0,  0.5f0,   (8,15))
( 40,  2.0f0,   2,       0.07f0,  0.4f0,   (15,25))
( 60,  4.0f0,   2,       0.06f0,  0.35f0,  (25,35))
( 80,  6.0f0,   3,       0.05f0,  0.3f0,   (30,42))
(100,  8.0f0,   4,       0.045f0, 0.28f0,  (35,50))
```

`_HIR_N_SUBSETS = 12` at every strength (`:66`). Odd decades linearly interpolated (`:144-158`). `get_hir_params` throws `ArgumentError` for anything not in `0:10:100`.

**Iteration structure — fixed trip count, no convergence check.** `for epoch in 1:nepochs` × `for (s, angle_indices) in enumerate(ws.subsets)` (12 subsets) at `driver.jl:1502-1503`. There is **no residual test, no tolerance kwarg, no early exit**. The only data-dependent exit is the `nepochs == 0` short-circuit at `:1479-1483`. Total inner iterations ∈ `{0, 12, 24, 36, 48}`.

Update at `:1596-1619`:

```julia
reg_views_scale = T(length(geom.angles)) / T(1000)
AK.foreachindex(vol, backend) do idx
    if x*x + y*y > radius_sq
        vol[idx] = zero(T)                                  # circular support, every sub-iter
    else
        data_update = λ_relax * vinv[idx] * ss * corr[idx]   # ss = 12
        reg_update  = λ * rvs * vinv[idx] * rg[idx]
        vol[idx] += data_update - reg_update
    end
end
```

with `λ = T(params.lambda) * (ws.projector === :siddon ? one(T) : T(0.1))` at `:1467` — a **10× projector-dependent rescale**.

**`n_views` invariance**: the data term is already view-count-invariant because `V_inv ≈ 1/n_views` cancels `subset_scale = 12`; the regularizer is multiplied by `n_angles/1000` so "strength" is protocol-independent, anchored at 1000-view protocols (`:1590-1595`, "Audit A1").

Weights:
- `W_proj = 1/(A·1)` — `compute_projection_weights(work_geom, work_size, T; projector, like=sinogram, circular_support=true)` at `workspace.jl:1073-1076`, eps `1e-8` (`ir/utils.jl:204`)
- `V_inv = 1/(Aᵀ·1)` — `compute_image_weights(…; active_z=1:work_size[3], circular_support=true)` at `workspace.jl:1077-1080`, eps `1e-8` (`ir/utils.jl:237`)
- Statistical `w = air_ref·exp(−clamp(y,0,10)) + 1e-6` at `:1497-1519`. The `clamp(y, 0, 10)` is "audit A4" — it prevents negative-`y` air rays from getting `exp(+|y|)`
- Folded once as `data_weights = W_proj ⊙ stat_w` at `:1525-1530`

**The system matrix is the SAME forward projector as the simulation**: `_project_mono_hir!(ws.projector, …)` at `:1552-1559` → `siddon_forward_project!` / `_dd_forward_project_arc_rowtile4!` / `dd_forward_project!`. Adjoint via `_backproject_mono!` at `:1580-1595` → `dd_backproject!` for DD. **So HIR inverts the DD operator, not the FDK voxel backprojector** — a different operator from the FBP path.

Huber penalty at `ir/utils.jl:44-60`, `:121-169`: 6-connected face-neighbour stencil, forward differences for the 3 upper neighbours and backward for the 3 lower, with hard boundary guards. `_huber_deriv(t,δ) = |t| ≤ δ ? t : δ·sign(t)` — `sign` is a kink, `abs` is a kink. Gradient recomputed **per sub-iteration** for DD (`:1510`) but **once per epoch** for Siddon (`:1504`), preserving legacy cadence.

Gather-indexed residual at `:1565-1571` (no staging copy):

```julia
a = aidx[k]                                   # Int32 device gather: slot → global view
residual = sino[col, row, a] - ax[idx]
ax[idx] = dw[col, row, a] * residual
```

Axial halo: `_hir_axial_support` at `workspace.jl:1094-1129` expands `nz` by the cone magnification `(SAD+r)/(SAD−r)`; `_hir_seed_work!` (`driver.jl:1293`) replicates terminal slices, `_hir_extract_output!` (`:1311`) crops back. Skipped for helical, `nz==1`, or `nepochs==0`.

Buffers mutated per sub-iteration: `ws.reg_grad`, `ws.subset_Ax_buf`, `ws.correction`, `ws.work_volume` — 4, plus two `fill!` per sub-iteration.

### 3.10 The VMI chain

**The chain named in the brief does not exist verbatim in any notebook.** Two eras coexist.

#### Era 1 — `src`-based, notebook 09 only

`apply_cong!` is called from exactly one non-archived notebook, at `docs/notebooks/09_siemens_force_ufc_dual_source_vmi.jl:811`.

| # | Stage | Entry point | Notes |
|---|---|---|---|
| 1 | Debias | notebook-local closure at nb09 `:794-802` | `out -= 1/(2·max(I0·exp(-out), 1))`. **Not in src** |
| 2 | *(no sinogram denoise)* | nb09 §8 header `:753-756` explicitly says "running on the **raw noisy sinograms**" | — |
| 3 | Basis build | `resolve_source_spectrum_full` → `(e, ŵ)`; notebook packs `p = μρ_iodine`, `q = μρ_water` at `:781-789` | material-direct |
| 4 | Cong | `create_cong_workspace` `cong.jl:98`, `apply_cong!` `cong.jl:154` | mutates `sino_y`, `sino_c` |
| 5 | FBP | `create_fdk_recon_workspace(…; filter = SoftFilter())` for **both** bases at nb09 `:880-883` | |
| 6 | ACNR | `apply_acnr_kalender!(W, I)` with **all defaults** at nb09 `:901` | |
| 7 | VMI synth | `synth_vmi_2basis` at nb09 `:1013` | |

#### Era 2 — the production chain, notebook-only

nb03 (K=2), nb04/08 (K=4), nb07 (K=2), nb12 (K=2):

1. Corrected per-kVp / per-bin channels — **no debias, no SVD, no SF-JSD, no combine**; every native detector row retained (nb03 `:714-748`).
2. `build_nchannel_basis` → per-ray absolute response `Φ[n_col, n_row, n_E, K]` `Float32` (nb03 `:754-804`); asserts `I0_relerr < 5e-5` with a hard `error()`.
3. `nchannel_profile_tile!` — the K-channel profiled Poisson quasi-likelihood, an `AK.foreachindex` kernel, tiled over views with `BS.tile_ranges(shape[3], tile_views)` (tile size 8).
4. Per-basis FBP with the two `CustomFilter` kernels (nb03 `:929-944`).
5. `apply_acnr_kalender!` — per-notebook hyperparameters (see 6.2).
6. `synth_vmi_2basis(images.water, iodine .* 1000f0; energy_keV=E)` at 50/70/100/140 keV.

nb04/nb08 additionally insert a **Lee-2025 T-LBF** joint sinogram filter between decomposition and FBP (nb04 `:1073-1138`, 5×5 window, `alpha1=0.9`, `alpha2=24.635648571666497`) and use a **common** `SoftFilter()` FBP preceded by a notebook-local angular-response FFT apodization (nb04 `:987-1017`).

The complete list of notebook-only n-channel symbols (nb04 line numbers):

| Symbol | Line | Role |
|---|---|---|
| `nchannel_basis` | `:321-360` | builds `Φ[e,k]`, `μρ_I`, `μρ_W`, `I0`, `μI_eff/μW_eff`, normal-equation scalars |
| `nchannel_controls` | `:363-377` | `iodine_bounds=(-0.10,0.40)`, `water_bounds=(-2.0,50.0)`, `outer_iterations=16`, `inner_iterations=12`, `max_iodine_step=0.05`, `max_water_step=5.0`, `parameter_tolerance=5e-5`, `fisher_condition_limit=1e8`, `air_gate=0.0`, `tile_views=8` |
| `nchannel_forward` | `:387-406` | exact polychromatic λ + analytic 1st/2nd derivatives, `Float64` |
| `nchannel_golden_minimize` | `:408-425` | golden section, fixed 80 iterations |
| `nchannel_scalar_global` | `:427-450` | 129-point grid → basin detect → golden refine |
| `nchannel_solve_total_C` | `:452-469` | fixed-80 bisection on the monotone aggregate |
| `nchannel_cong_constrained_reference` | `:475-511` | exact monotone aggregate-channel oracle |
| `nchannel_poisson_quasi_nll` | `:514-517` | `Σ(λ − y·log λ)` |
| `nchannel_profile_reference` | `:525-561` | slow bounded profile-likelihood oracle |
| **`nchannel_profile_tile!`** | `:566-…` | **the production estimator**; 28-step aggregate bisection at `:657`, `for outer_iter in 1:16` at `:678` with inner `for _ in 1:12` at `:681`; writes `sino_I, sino_W, fisher_AA/AC/CC, quality_flag::UInt8, score_norm, outer_count, inner_count` |

**Bottom line: there is no `src` function that reproduces the numbers in nb03/04/07/08/12.** The oracle for the published estimator must be extracted from the notebook cells.

#### Cong math — `src/reconstruction/vmi/cong.jl:195-363`

Per ray with `T_L_meas = exp(-p_L_meas)`, `T_H_meas = exp(-p_H_meas)`:

1. **Air gate** at `:206-210`: both `|p| < 5e-3` → both outputs 0, early return.
2. **Outer Brent on water-equivalent path `L`** at `:215-239`, solving
   `Σ_i ŵ_L[i]·exp(-(p_L[i]·a_w + q_L[i]·c_w)·L) = T_L_meas`.
   Bracket `[-1, L_hi]` with **adaptive** `L_hi = max(60, 1.5·p_L_meas / max(μ_w_min, 1e-4))` at `:233`. This adaptive bracket is the historical VMI bug fix. `μ_w_min` is computed on the **host** via `Array(p_L)`/`Array(q_L)` at `:184`. Then `c̄ = c_w · L_water`.
3. **y upper bound**: `y_max = min(0.99·p_L_meas / max(p_L_min, eps(Float32)), 1e7)` at `:242`.
4. **Inner Newton on the Eq-8 quintic** at `:250-291`: accumulate `P0..P5` with Taylor coefficients `1, −q, q²/2, −q³/6, q⁴/24, −q⁵/120`. Fixed `for _ in 1:12` at `:281` with two data-dependent `break`s at `:285` (`|dF| < 1e-30`) and `:288` (`|Δ| < eps(Float32)`).
5. **Outer Brent on y** at `:294-349`, `G(y) = T_H_pred(y, c̄ + solve_quintic(y, c̄)) − T_H_meas`. Three branches on `G0 = G(0)`:
   - `G0 == 0` → `(0, true)`
   - `G0 < 0` → geometric doubling from `y_hi = min(0.01, y_max)`: `while isfinite(G_hi) && G_hi < 0 && y_hi < y_max && n_exp < 24` at `:329`, **inside the GPU kernel**
   - `G0 > 0` (noise excursion) → try `G(-0.1)`; if negative, `brent_solve(G, -0.1, 0)`
6. **Writeback** at `:350-362`: on failure, water-only `(a_w·L, c_w·L)`. Otherwise `clamp(y_opt, -5, 1e4)` and `clamp(c̄ + x_final, -5, 1e4)` — **deliberately not rectified to ≥0** (comment `:356-360`; rectification bias was what `noise_reduction` was masking).

Signature:

```julia
apply_cong!(ws::CongWorkspace,
            sino_y::AbstractArray{Float32,3}, sino_c::AbstractArray{Float32,3},
            sino_low::AbstractArray{Float32,3}, sino_high::AbstractArray{Float32,3};
            water_basis,
            newton_max_iter::Int = 12, newton_tol::Real = eps(Float32),
            y_max_factor::Real = 0.99, y_max_cap::Real = 1f7)
```

`CongWorkspace` at `cong.jl:70-76` has **six fields and no scratch buffers**:

```julia
struct CongWorkspace{T, AŴ_L, AŴ_H, A1}
    ŵ_L::AŴ_L;  p_L::A1;  q_L::A1
    ŵ_H::AŴ_H;  p_H::A1;  q_H::A1
end
```

`ŵ_*` is 1-D `[n_E]` (centered) or 3-D `[n_col, n_row, n_E]` (per-ray bowtie). All `Float32`, enforced at `:102-108`. No tiling (`:30-31`).

**The material-direct convention** (the live one): `p := μρ_iodine(E)` cm²/g, `q := μρ_water(E)` cm²/g, `water_basis = (a = 0.0f0, c = 1.0f0)`. With `a_w = 0, c_w = 1` the step-1 exponent collapses to `μρ_water·L`, so `L` is water areal density g/cm², `sino_y` is **iodine g/cm²**, `sino_c` is **water g/cm²**.

The photo/Compton pairing at `basis.jl:90-105` emits a `Base.depwarn` at `:91`: the LSQ fit fails on the iodine K-edge.

**`brent_solve`** at `roots_kernels.jl:96-103`:

```julia
brent_solve(f, a::T, b::T; xabstol::T = eps(T)*oneunit(T), xreltol::T = eps(T),
            maxiters::Int = 100) where {T<:AbstractFloat}  ->  (root, converged::Bool)
```

`for _ in 1:100` at `:127` with **three data-dependent exits**: `iszero(fs)` at `:162`, NaN/Inf `fs` at `:165-167`, and bit-exact bracket collapse `nextfloat(min(a,b)) == max(a,b)` at `:187-190`. **The actual termination is bit-level, not tolerance-based.** Inverse-quadratic when `fa != fc && fb != fc`, else secant; 5-clause `force_bisect` predicate. Bisection midpoint is a **bit-reinterpret** (`reinterpret` to `UInt`, `>> 1`, reinterpret back) at `:35-59`.

Parity oracle is `test/archived/vmi_brent_parity.jl` against `Roots.find_zero(f, (a,b), Roots.Brent())`. Note `roots_kernels.jl:25` references `test/vmi/test_brent_parity.jl`, which **does not exist**.

#### ACNR

```julia
apply_acnr_kalender!(W::AbstractArray{T,3}, I::AbstractArray{T,3};
                     hp_sigma_px::Real = 1.5, window::Int = 4,
                     beta_max::Real = 8.0, passes::Int = 2) where {T<:AbstractFloat}
```

at `src/denoising/acnr.jl:319-326`. **The docstring at `:293` says `window=2`; the actual default is 4.**

**Fully serial CPU.** Forces `Array(W)`, `Array(I)` at `:372`. Multi-pass is Julia-level **recursion** at `:333-340`.

High-pass `G(vol)` at `:344-370` is a **separable direct convolution**, radius `r = max(2, ceil(3σ))`, two fully serial triple loops with `clamp` replicate BC. Despite the comment "CPU FFT fine", **there is no FFT**.

Global variance-limited bounds at `:381-384`:
```
λ_I = sqrt(Σ hI² / max(Σ hW², 1e-30));  βmaxI = min(beta_max, λ_I)
λ_W = sqrt(Σ hW² / max(Σ hI², 1e-30));  βmaxW = min(beta_max, λ_W)
```

Per-pixel regression at `:385-403`: `(2w+1)² = 81` taps at `w=4`, **in-plane only, no z extent**; accumulate `sWW, sII, sWI`; then
```
βI = clamp(sWI / max(sWW,1e-20), -βmaxI, 0)     # one-sided clamp
βW = clamp(sWI / max(sII,1e-20), -βmaxW, 0)
outI = Ic - βI·hW;   outW = Wc - βW·hI
```
Simultaneous (reads pristine `Wc/Ic/hW/hI`), so the update is symmetric. `copyto!` back at `:404`. Returns `(ρ_hp, σ_hW, σ_hI)`.

**Eight volume-sized allocations per pass.**

The key numerical distinction versus the older variants: `apply_acnr!` (`:58`) and `apply_image_acnr!` (`:180`) apply a linear or edge-aware **operator to the signal** (FFT Gaussian/Tikhonov, or joint bilateral). Kalender applies a per-pixel scalar β as a **pure subtraction**, so no operator multiplies the signal and resolution is preserved by construction.

#### The denoisers are not wired into any notebook

`apply_sino_svd_denoise`, `apply_sino_sfjsd_denoise`, `apply_median_z`, `apply_rskr`, `apply_pwls!`, `apply_rwls!`, `apply_cmv!`, `apply_mono_plus*` appear **only** in `test/denoising.jl:52-378`. The comment at `src/BasisSimulator.jl:294-303` claiming SF-JSD and median-z are "live in the current notebooks: used by nb03/04/07" is **stale**. The only denoiser called from a notebook is `apply_acnr_kalender!`.

Nonetheless, for completeness:

| Denoiser | Entry | Structure |
|---|---|---|
| `apply_sino_svd_denoise!` | `sino_svd.jl:95` | `Threads.@threads for r in 1:n_row`; per row `M = (n_col·n_view) × N`, **CPU LAPACK `svd`** at `:133`; `U[:,1]` untouched, `U[:,2..N]` separable-Gaussian smoothed; `M_d = U_d·Diagonal(Σ)·V'` |
| `apply_sino_svd_denoise_bilateral!` | `sino_svd.jl:263` | same, residual components through `_joint_bilateral_2d` with per-row MAD scales (2 `median` calls each) |
| `apply_sino_sfjsd_denoise` | `sino_sfjsd.jl:433` | non-mutating API but mutates internal `ξ` in place; `GC.gc(true)` at `:486`; per-row SVD at `:511`; SURE-optimized σ₀ with a `while abs(b-a) > tol` golden section at `:281`, each iteration = 2 SVDs + 4 bilateral passes; `MersenneTwister(42)` at `:250`; `stride` and `n_iter` both derived from data |
| `apply_median_z!` | `median_z.jl:42` | `Threads.@threads for k in 1:nz`; `sort!(view(buf,1:n))` **per voxel** at `:60` |
| `apply_rskr` | `rskr.jl:259` | `nch ∈ (2,4)`; **SVD of the entire volume set** `(n_vox × nch)` once per iteration at `:285`; `mad_haar_σ` uses `median` on the mid-z slice; `joint_bf_{2,4}ch_gpu!` are `AK.foreachindex` kernels |

#### PWLS / RWLS

| | PWLS | RWLS |
|---|---|---|
| Entry | `apply_pwls!` `pwls.jl:196` | `apply_rwls!` `rwls.jl:354` |
| Channels | exactly 2, errors otherwise `:212` | exactly 3, errors otherwise `:373` |
| Workspace | `PwlsWorkspace` `:58-64`, 5 fields, 4 buffers at `(n_col,n_row,tile_size)` | `RwlsWorkspace` `:71-78`, **15 fields**, 9 tiled sinogram buffers + 5 `n_E` vectors |
| Loop nesting | `for iter` OUTER, `for tile_range` INNER (`:299`, `:305`) | `for tile_range` OUTER, `for iter` INNER (`:431`, `:447`) |
| Convergence | **none**; runs exactly `n_iter`. Only a `@warn` if cost increased (`:402-404`) | outer GN has **none**; CG has one (`rr_new < tol_sq` at `:212`) |
| Per iteration | 4 Laplacian launches + 1 fused kernel + 3 device→host reductions per tile | 1 fused step kernel + a `@.` block with **transient buffer aliasing** (`cg_b` as `det_H`, `cg_r` as `δ_Wg`, `cg_p` as `δ_Ig`, `:461-469`) + optional CG prior |
| Projection | `max(x, 0f0)` non-negativity at `:393-394` | `max(x, 0f0)` at `:467-468` |
| Host prep | `Array(ŵ_raw)` + serial per-ray normalization `:244-258` | `Float64.(basis.ŵ_bins[b])` + `Array(p)`/`Array(q)` at `:410-416` |

`_rwls_cg_solve!` at `rwls.jl:185`: warm start `copyto!(x,b)`, `for k in 1:max_iter` (20) with a data-dependent break; each iteration is 1 stencil kernel (**periodic BC** on dims 1,2, `:154-181`) + 2 global `sum` reductions + 3 broadcast updates.

#### VMI synth, Mono+, phantom mask

| Function | Location | Math |
|---|---|---|
| `synth_vmi_2basis!(HU_E, c_water, c_iodine; energy_keV, …)` | `image_domain_decomp.jl:328-343` | `α_E = μρ_I(E)/μρ_w(E)`; `HU_E = 1000·(c_water − 1) + c_iodine·α_E`. `c_water` g/mL, `c_iodine` **mg/mL** — this is why notebooks multiply the iodine basis by `1000f0` |
| `synth_vmi_hu(a, c, energies; …)` | `vmi_synth.jl:36-43` | `μ = p_photoelectric(E)·a + q_compton(E)·c` → `to_hounsfield`. **Photo/Compton only — deprecated pairing** |
| `synth_vmi_sino_domain(…)` | `vmi_synth.jl:128-141` | per energy `sino_E = μρ_a·sino_a + μρ_b·sino_b`, one FBP each, shared workspace mutated across energies |
| `apply_mono_plus!(ws, volumes, energies; E_noise_opt=70.0, σ_lp_px=2.0, …)` | `mono_plus.jl:160-168` | `Mono+(E) = LP_σ(VMI_E) + VMI_opt − LP_σ(VMI_opt)`; per-z-slice `Float64` FFTW round trip at `:214`; returns **references into the workspace** |
| `apply_mono_plus_regression!(…; σ_lp_px=1.5, window=4, beta_max=4.0)` | `mono_plus.jl:351-361` | `LP_σ(VMI_E) + β(x)·HP_σ(VMI_opt)`, `β = clamp(Σ hE·hOpt / Σ hOpt², 0, β_max)`; `Threads.SpinLock` for β accumulation at `:433` |
| `erode_mask_2d(mask2d; erode_px)` | `phantom_mask.jl:123-132` | FFT-Gaussian blur of the `Float64` mask then `blurred .≥ 0.999` |
| `resample_phantom_mask_to_recon(…)` | `phantom_mask.jl:50-105` | nearest-neighbour `round(Int,…)` in a serial loop |

`eval_cal` (`image_domain_decomp.jl:249`) and `apply_cal!` (`:271`) both emit `Base.depwarn` — constants stale, rational-quadratic denominators have poles inside the clinical HU domain (audit B1/B6).

---

## 4. Workspace and mutation inventory

### 4.1 The eight `create_*_workspace` constructors

| Constructor | Location | Fields | Notes |
|---|---|---|---|
| `create_workspace` (PCCT) | `src/api/workspace.jl:141`, struct `:32-109` | 40+ | Includes `rng::MersenneTwister`, `pileup_S`, `R`, `R_energies`, `W_matrix_gpu`, `μ_table_gpu`, `outputs_flat`, native-resolution mirror buffers (`native_bins`, `native_geom`, …), `noise_staging::Array{T,3}` on the **host** |
| `create_eict_workspace` | `:525`, struct `:436-492` | 33 | Includes `rng`, four noise staging buffers (2 host `Vector{T}` + 2 device), `bowtie_spectral`, `bowtie_air_reference`, `wη_gpu`, `η_eff`, `σ_e_photon`. `spectrum_override` kwarg at `:501` |
| `create_fdk_recon_workspace` | `:807`, struct `:783-797` | 8 | `volume`, `filtered`, `conv_scratch`, `filter_kernel`, 4 geometry matrices. **Filter, kernel size, and volume size frozen at construction.** The FFT windowing runs once here |
| `create_hir_recon_workspace` | `:1006`, struct `:942-1004` | 22 | Runs **one full forward projection and one full backprojection at construction** (for `W_proj` and `V_inv`). `work_volume` **aliases** `volume` when there is no halo (`:1030`). 12 subsets × 4 geometry matrices + 12 `Int32` index vectors |
| `create_cong_workspace` | `vmi/cong.jl:98` | 6 | Immutable struct, no scratch, no tiling |
| `create_pwls_workspace` | `vmi/pwls.jl:84` | 5 | 4 buffers at `(n_col, n_row, tile_size)` |
| `create_rwls_workspace` | `vmi/rwls.jl:106` | 15 | 9 tiled sinogram buffers + 5 `n_E` vectors |
| `create_mono_plus_workspace` | `vmi/mono_plus.jl:84` | 5 | Includes a **growing `Dict{Float64,Matrix{Float64}}` kernel cache**. Does not tile (`with_oom_retry(…, 1)`) |

Approximate HIR hot-path footprint: 5 sinogram-sized + 5 volume-sized arrays, plus 4 + 48 small geometry matrices.

### 4.2 Every `!` function and its write target

**Sinogram in place**

| Function | Location | Writes |
|---|---|---|
| `simulate!` (EICT) | `driver.jl:423` | `ws.sinogram` + every scratch buffer |
| `simulate!` (PCCT) | `driver.jl:117` | `pcct_sino.bins`, `ws.combined`, `ws.tube_physics_scratch`, `ws.scratch`, `ws.noise_staging`, `ws.rng` |
| `_forward_project_poly!` | `polychromatic.jl:423` | `sinogram`, `I_transmitted`, `sino_mono`, `μ_volume`, μ LUTs |
| `create_μ_volume!` | `polychromatic.jl:217` | `μ_volume`, `μ_at_energy_cpu`, `μ_at_energy` |
| `_apply_physics_no_noise!` | `polychromatic.jl:271` | `sinogram` + all effect scratch |
| `apply_fill_factor!` | `fill_factor.jl:88` | `sinogram` (`+= -log(ff)`), auto-cancelled at `driver.jl:632` |
| `apply_optical_crosstalk!` | `optical_crosstalk.jl:67` | `ws_output` then `copyto!(sinogram, output)` |
| `apply_focal_spot_blur!` | `focal_spot.jl:397` | `ws_output` then `copyto!(sinogram, output)` |
| `apply_lag!` | `detector_lag.jl:101` | `ws_intensity`, `ws_output`, then `copyto!` |
| `estimate_scatter_field!` | `scatter.jl:712` | `output` **and** `ws_scatter_temp`; `sinogram` read-only |
| `inject_scatter!` | `scatter.jl:317` | `sinogram` |
| `inject_scatter_bins!` | `scatter.jl:352` | each `bins[b]` |
| `_convolve_separable_h!` / `_v!` | `scatter.jl:154` / `:186` | `output` |
| `apply_pcct_noise!` | `photon_counting.jl:822` | `sino.bins[b]`, `cpu_buf`, `raw_out[b]`, `ws_rng` state |
| `spatial_bin!` | `photon_counting.jl:280` | `output` |
| `apply_pcct_pileup_correction!` | `pcct_pileup_correction.jl:84` | `bins[1..4]`, **hard-coded 4 bins** |
| `low_signal_correction_gpu!` | `calibration.jl:37` | `prep` |
| `apply_bhc!` | `bhc_sinogram.jl:280`, `:296` | `sinogram`, `ws_coeffs_gpu` |
| `filter_sinogram!` | `filtering.jl:461` | `sinogram` **twice** (cosine weight + `copyto!`), plus `conv_scratch` |
| `cosine_weight!` | `filtering.jl:385` | `sinogram` (`*=`) |
| `apply_cong!` | `cong.jl:154` | `sino_y`, `sino_c` at 5 write sites |
| `apply_cmv!` | `cmv.jl:36` | `sino_water`, `sino_iodine` |
| `apply_pwls!` | `pwls.jl:196` | `sino_iodine`, `sino_water`, 4 ws buffers, `cost_history` |
| `apply_rwls!` | `rwls.jl:354` | `sino_iodine`, `sino_water`, all 9 ws sino buffers, 5 vectors |
| `apply_pcct_vmi_poly!` | `pcct_calibration.jl:164` | `sino_water`, `sino_iodine` under `Threads.@threads` |
| `combine_pcct_bin_counts!` | `pcct_basis.jl:130` | `out_bins[k]`, staging buffer |

**Volume in place**

| Function | Location | Writes |
|---|---|---|
| `backproject!` | `backprojection.jl:522` | `volume`, full overwrite |
| `_wfbp_backproject!` | `wfbp_helical.jl:133` | `volume`, full overwrite |
| `_wfbp_rebin!` | `wfbp_helical.jl:59` | `reb` |
| `apply_fov_mask!` | `fdk.jl:463` | `volume` outside the radius |
| `reconstruct!` (FDK) | `driver.jl:1236` | `ws.filtered`, `ws.conv_scratch`, `ws.volume` |
| `reconstruct!` (HIR) | `driver.jl:1404` | 8 buffers: `filtered`, `conv_scratch`, `volume`, `work_volume`, `data_weights`, `correction`, `reg_grad`, `subset_Ax_buf` |
| `compute_huber_gradient!` | `ir/utils.jl:121` | `grad`, full overwrite |
| `compute_image_weights` | `ir/utils.jl:219` | `voxel_sums` |
| `apply_acnr_kalender!` | `acnr.jl:319` | `W`, `I` via `copyto!` |
| `apply_acnr!` / `apply_image_acnr!` | `acnr.jl:58` / `:180` | `sino_a`,`sino_b` / `W`,`I` |
| `apply_mono_plus!` and variants | `mono_plus.jl:160`, `:351`, `:567` | `ws.out_vols`, `ws.lp_opt`, `ws.hp_opt`, `ws.lp_buf`, `ws.kernel_cache` |
| `apply_radial_capping_basis!` | `radial_capping_basis.jl:40` | `a`, `c` slices |
| `apply_radial_cupping_correction!` | `radial_cupping.jl:102` | `hu_vol`. **DEPRECATED** |
| `apply_bhc_image_domain` | `bhc_image_domain.jl:73` | `recon_μ` **despite no `!` in the name**. DEPRECATED |
| `add_system_noise_floor!` | `driver.jl:734` | `vol` |
| `joint_bf_{2,4}ch_gpu!` | `rskr.jl:71` / `:137` | `out1..out4` |
| `apply_median_z!` | `median_z.jl:42` | `out`, `buf` |
| `apply_sino_svd_denoise!` and bilateral | `sino_svd.jl:95` / `:263` | `out[b][:, r, :]` |
| `_sfjsd_pass!` | `sino_sfjsd.jl:154` | `out` |
| `release_backend!` | `memory_budget.jl:178` | frees every reachable device array — **destroys the object** |

### 4.3 State carried between calls

Every workspace field. Critically:

- `ws.rng` internal state advances unless reseeded. EICT reseeds only when `sim_opts.seed !== nothing`.
- The FDK filter kernel, kernel size, `cutoff`, and volume size are frozen at construction — re-create the workspace to change them.
- `ws.pileup_S`, `ws.R`, `ws.W_matrix_gpu`, `ws.μ_table*`, `ws.bowtie_spectral`, `ws.bowtie_air_reference` are computed once.
- `HIRReconWorkspace.work_volume === volume` when there is no axial halo — guarded with `===` at `driver.jl:1294`, `:1312`.
- `MonoPlusWorkspace.kernel_cache` is a `Dict` that grows lazily.
- Both `PCCTWorkspace` and `EICTWorkspace` are `mutable struct` but no field is reassigned after construction.

---

## 5. Numerics

### 5.1 Precision by stage

| Stage | Precision |
|---|---|
| `CTGeometry`, all fields | `Float64` host |
| Geometry → device | `T.(geom.X)` narrowing, `T = Float32` in practice. **Allocates a host `Matrix{T}` temporary per call** on the non-workspace path (`backprojection.jl:571,578,585,592`; `dd.jl:384`; `siddon.jl:509…1247`) |
| Spectrum loading + filtering | `Float64` host |
| Bowtie `B`, heel, `η` | `Float64` host, narrowed to `Float32` when stored |
| μ table | `compute_μ_at_energy` returns `Float64`, stored as `T` |
| All projector kernels | `T`, generic `where {T<:AbstractFloat}`. `Float32` in practice. **No `Float64` accumulators anywhere** |
| Filtering, backprojection | `T` throughout; `acc`/`w_acc` are `zero(T)` |
| WFBP `Δβ`, `n_half`, `z_start`, `q̂` | `T` |
| EICT noise | `Float32` on device |
| PCCT noise | **`Float64` inside `_poisson_sample`**, cast back to `T` |
| BHC fitting | `Float64` host, LAPACK `\` on normal equations |
| `HIRParams` (`lambda`, `huber_delta`, `relaxation`), `HuberPenalty.delta` | **Hard `Float32`, not parameterized** |
| Radial cupping / capping | `Float64` host end to end, `A \ vals` least squares, cast back to `Float32` on write |
| PWLS tile cost accumulators | `Float64` |
| Mono+ / ACNR image-ACNR FFT | `Float64` promotion per slice |
| Scatter bin weights, heel spectral | `Float64` host |

### 5.2 RNG

Three CPU `MersenneTwister` instances and two global-RNG leaks. **There is no GPU RNG anywhere.**

| Site | Seeding |
|---|---|
| `ws.rng` PCCT | `workspace.jl:215` `MersenneTwister(0)`, reseeded per call at `photon_counting.jl:837` |
| `ws.rng` EICT | `workspace.jl:743` `MersenneTwister(0)`, reseeded at `driver.jl:528` **only if `seed !== nothing`** |
| `add_system_noise_floor!` | `driver.jl:736` `MersenneTwister(seed + 7919)` or `Random.default_rng()` |
| `compute_mc_pileup_matrix` | `mc_pileup.jl:336` seed 42, but see the leak below |
| `Poisson_approx` | `mc_pileup.jl:259`, `:267` — **global RNG leak**, breaks pileup reproducibility |
| `_sfjsd_sure` | `sino_sfjsd.jl:250` `MersenneTwister(42)` hardcoded |
| Fallbacks to `Random.default_rng()` | `photon_counting.jl:840`, `mc_pileup.jl:107`, `driver.jl:736` |

There is no RNG anywhere under `src/projection/`, `src/reconstruction/`, or `src/correction/`.

### 5.3 Gather versus scatter

**Every kernel in the package is a gather. Zero atomics, zero scatter-adds.**

`outputs_flat[...] += ...` at `siddon.jl:1425`, `:1430`; `dd.jl:767`, `:771`; `dd_fast.jl:274` is race-free because `idx` is unique per thread and the bin offsets `(b-1)*n_elem` are disjoint.

`AK` usage is exclusively `AK.foreachindex` plus `AK.mapreduce` at `ir/utils.jl:101`. There is no `AK.map`, no `AK.reduce`, no `AK.sort` in the projection or reconstruction subsystems. Backprojection calls `AK.foreachindex(volume)` **without** an explicit backend; `ir/utils.jl` passes `AK.get_backend(x)` explicitly.

Complete `AK.foreachindex` inventory in projection:

| Site | Domain | One thread computes |
|---|---|---|
| `dd.jl:429` | sinogram | one cell (mono DD) |
| `dd.jl:479` | sino (threads with `slot >= ntiles` early-`return` at `:485`) | 4 rows (arc tile) |
| `dd.jl:586` | sinogram | one cell, `N_E` energies |
| `dd.jl:713` | pilot | one cell, `K` energies × `n_b` bins |
| `dd_fast.jl:112` | sinogram | one cell, `M` path lengths → all `n_E` |
| `dd_fast.jl:207` | pilot | one cell, `M` path lengths → `K` × `n_b` |
| `dd_transpose.jl:98` | `view(volume,:,:,z_first:…)` | 4 voxels (arc tile) |
| `dd_transpose.jl:336` | `active_volume` view | one voxel |
| `polychromatic.jl:232`, `:259` | mask | one voxel μ lookup |
| `polychromatic.jl:568,576,633,644,653` | sinogram / `I_trans` | accumulate + final `-log` |
| `siddon.jl:535`, `:880`, `:1269` | sinogram / pilot | one ray |

`@inbounds` at 16 sites, all inside `@generated` NTuple helpers or `dd_fast` cell loops: `dd_fast.jl:63,65,168,170,269,271`; `siddon.jl:718,733,735,752,754,1101,1117,1119,1135,1137`. **There is no `@inbounds` in any reconstruction file.**

`unsafe_trunc(Int32, …)` at 18 sites in projection (`dd.jl:100-101`; `dd_transpose.jl:22-23,159-160,467-470`; `siddon.jl:256-258,970-972,1352-1354`) and 5 in reconstruction (`backprojection.jl:136-139,337-340,463-466`; `wfbp_helical.jl:111-113,215-217`). UB on NaN or overflow; the only guards are preceding `clamp` calls or bounds branches that happen to be false for NaN.

### 5.4 Tolerance constants and clamps

| Value | Purpose | Sites |
|---|---|---|
| `T(1e-10)` | log floor `-log(max(I, 1e-10))` | `dd.jl:643`; `dd_fast.jl:175`; `siddon.jl:959`, `:1059`; `polychromatic.jl:575`, `:652`; and the 10 detector-side sites listed in 3.6 |
| `T(1e-10)` | Siddon ray de-singularization **and** path-length significance | `siddon.jl:213-216` + `:316`; `:923-926` + `:1027`; `:1312-1315` + `:1397` |
| `T(1e-10)` | backprojection `abs(sv_dot_sd)` degeneracy → `continue` | `backprojection.jl:90`, `:258`, `:427` |
| `T(1e-12)` | DD footprint degeneracy (`detXstep`/`detZstep`) | `dd.jl:165`, `:186` |
| `T(1e-12)` | arc/flat projection degeneracy → `(Inf, Inf)` | `dd_transpose.jl:42`, `:53`; `rho_min` floor `:165`, `:422` |
| `T(1e-4)` | in-plane voxel anisotropy warning | `dd.jl:522` |
| `T(1e30)` | `min(exp(-L), 1e30)` on the **bowtie branch only**, prevents `Inf × 0 = NaN` in Float32 | `siddon.jl:1135`, `:1137`; `dd_fast.jl:269` |
| `T(1e-8)` | reciprocal guards in `W_proj`/`V_inv` | `ir/utils.jl:204`, `:237`; `wfbp_helical.jl:230` |
| `T(1e-6)` | statistical-weight floor; helix z-range slack | `driver.jl:1500`, `:1513`; `backprojection.jl:304-330` |
| `T(1e-3)` | WFBP in-plane denominator guard | `wfbp_helical.jl:203` |
| `±0.999` | `asin` domain guard on `t/R` | `wfbp_helical.jl:98`, `:195` |
| `π*0.999` | `(γ/sinγ)²` far-tap guard | `filtering.jl:378` |
| `1e-20`, `1e-30` | ACNR / PWLS / RWLS determinant and variance floors | `acnr.jl:381-399`; `pwls.jl:389`; `rwls.jl` |

Clamps: `clamp(col/row, 1, n)` bilinear bounds; `clamp(ℓ/SAD, -1, 1)` before `asin`; `clamp(f_norm, 0, 1)` in apodization; `clamp(y, 0, 10)` before `exp(-y)` in HIR ("audit A4"); `clamp(sino, -1, 15)` in EICT calibration; `clamp(avail ÷ per_view_bytes, 1, n_view)` in tiling; `clamp(y_opt, -5, 1e4)` in Cong.

**No `Base.eps()` is used anywhere in projection.** `siddon.jl:213` shadows the name with a local `T(1e-10)` constant. `eps(Float32)` *is* used as the Cong Newton tolerance and the Brent default.

---

## 6. Notebook contract

### 6.1 The API surface that must not change

Consolidated across all 12 notebooks. Numbers indicate call sites.

**Five structs** — `Phantom`, `Scanner`, `CTProtocol`, `SimOptions`, `ReconOptions`. All 12.

| Group | Symbol | Notebooks |
|---|---|---|
| Phantom | `create_gammex_472(; n_voxels, n_slices, fov_cm, z_cm)` | 01,03,04,06,09,12 |
| | `create_phantom_from_mask(labeled, materials::Dict, voxel_size_cm)` | 07,08 |
| | `load_xcat_male_slab(; materials, quiet)` | 02 |
| | `xcat_default_materials()` | 02 |
| | `REGION_SOLID_WATER` | 03,04,09,12 |
| Geometry | `CTGeometry(scanner; n_angles, fov_cm, z_cm, collimation_mm)` | 05,06 |
| | `phantom_to_world_affine`, `recon_to_world_affine` | 05 |
| | `resample_to_recon(phantom, geom, matrix_size; method=:nearest\|:linear)` | 01,02,05,07,08 |
| | `estimate_phantom_diameter_cm(mask, voxel_size_mm)` | 02,05 |
| Workspaces | `create_eict_workspace` | all but 04,08 |
| | `create_workspace` (PCCT) | 04,08 |
| | `create_fdk_recon_workspace(sino, geom, matrix; filter)` | **all 12** |
| | `create_hir_recon_workspace(sino, geom, matrix; strength, projector)` | 02 |
| | `create_cong_workspace(sino_template, basis)` | 09 |
| | `release_backend!(ws)` | 03,04,07,08,12 |
| | `backend_memory_snapshot(arr)` | 04 |
| Forward/recon | `simulate!(ws, phantom, protocol, sim_opts[; capture_raw_counts])` | **all 12** |
| | `reconstruct!(ws_fdk, sino, geom)` | **all 12** |
| Spectrum | `resolve_source_spectrum_full(sim_opts, protocol; scanner, geom)` | 03,07,09,12 |
| | `resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner)` | 09,12 |
| | `compute_detector_I0(geom, protocol, sum(ws.weights))` | 03,07,09,12 |
| | `get_ufc_mc_efficiency`, `get_gemstone_mc_efficiency` | 09,12 |
| | `get_ufc_flash_mc_efficiency` | 12 |
| Corrections | `calibrate_bhc_water(sim_opts, protocol; scanner, geom)` | 01,02,06,11 |
| | `calibrate_bhc_water(energies, w_per_column; reference_energy_keV)` **2nd method** | 09,12 |
| | `apply_bhc_water(sino, model)` | 01,02,06,09,11,12 |
| | `bhc_spectrum_per_column(e, ŵ)` | 09,12 |
| | `compute_polychromatic_μ_water(sim_opts, protocol; scanner, geom, water_path_cm)` | 01,02,05,10 |
| | `measure_radial_cupping(hu_vol; fov_cm)` | 01 |
| | `apply_fov_mask!(recon_μ, geom)` | 02 |
| | `apply_acnr_kalender!(W, I; hp_sigma_px, window, passes, beta_max)` | 03,04,07,08,09,12 |
| | `apply_cong!(ws, sino_y, sino_c, sino_low, sino_high; water_basis)` | 09 |
| VMI | `synth_vmi_2basis(c_water, c_iodine_mg_mL; energy_keV)` | 03,04,07,08,09,12 |
| | `erode_mask_2d(mask2d; erode_px)` | 03,04,09,12 |
| | `tile_ranges(n_view, tile_size)` | 03,04,07,08,12 |
| Attenuation | `to_hounsfield(vol; μ_water)` | 01,02,05,06,09,10,11,12 |
| | `compute_μ_at_energy(material, E_keV)` | 01,03,04,07,08,09,12 |
| | `compute_mass_μ_at_energy(material, E_keV)` | 03,04,07,08,09,12 |
| Filters | `CustomFilter(x_knots::NTuple, y_knots::NTuple)` | 03,07,12 |
| | `SoftFilter()` | 04,08,09 |
| | `filter = :standard` (Symbol form) | 05,06 |
| Modules | `BS.XA`, `BS.AK`, `BS.FFTW` | most |

**Workspace fields read directly by notebooks** (part of the contract): `ws.sinogram`, `ws.geom`, `ws.weights`, `ws.η_eff`, `ws.energies`, `ws.bowtie_air_reference`, `ws.bins`, `ws.W_matrix_gpu`, `ws.volume`; `geom.fov`, `geom.n_cols`, `geom.n_rows`; PCCT return `result.pcct_sino.bins`, `result.I0_bins`, `result.pileup_S`; BHC model `model.μ_water_ref`, `model.reference_energy_keV`.

Four contract subtleties:

1. **Two `Phantom` arities** and two `materials` container types.
2. **Two `calibrate_bhc_water` methods.** Notebooks 09 and 12 depend on the second.
3. **The `filter` kwarg is polymorphic**: omitted, `Symbol`, `SoftFilter()`, or `CustomFilter(tuple, tuple)`.
4. **Mutation is load-bearing.** `simulate!`, `reconstruct!`, `apply_acnr_kalender!`, `apply_cong!`, `apply_fov_mask!` are called at roughly 90 sites. A pure-functional rewrite must keep these names as thin wrappers.

### 6.2 The verification cells

**Notebook 01** — `docs/notebooks/01_five_struct_api.jl:896-1010`, 5 checks, reports "✅ NB01 VERIFICATION: PASS (5/5)".

Setup: `resample_to_recon(phantom_cpu, sim_std.geom, recon_opts.matrix_size; method=:nearest)`, central slice `kmid`, 4-neighbour in-plane erosion, water label auto-detected as the most-common non-zero label.

| # | Metric | Gate | Line |
|---|---|---|---|
| 1 | Water mean HU, corrected, central slice | `[-6.0, 6.0]` HU | `:934` |
| 2 | Noise ratio `std(hu_low)/std(hu_std)` | `[1.7, 2.3]` (4× mA drop ⇒ ~2× σ) | `:936` |
| 3 | `measure_radial_cupping(...).cup_hu`, worst slice | `[0.0, 12.0]` HU | `:945` |
| 4 | `abs(cupqa.dc_hu)`, worst slice | `[0.0, 6.0]` HU | `:946` |
| 5 | All rods pass | see below | `:987` |

Per-rod gate at `:970-978`, central slice, eroded ROI, `count(m) ≥ 20`:
- theory `= 1000·(μ_mat(refE) − μ_w)/μ_w` with `refE = bhc_calibration.ref_E_keV`, `μ_w = model.μ_water_ref`
- `ok_theory = abs(meas − theory) ≤ max(15.0, 0.15·abs(theory))` — **±15 HU or ±15 %, whichever is larger**
- `ok_dose = abs(meas − meas_low) ≤ max(10.0, 0.03·abs(theory))` — **dose invariance ±10 HU or ±3 %**

Also reported ungated: full-body versus central-`r<5 cm` water z-profiles at `:975-977` (central-ROI falloff = bug; peripheral-only falloff = real cone divergence), and the uncorrected recon as an alignment sentinel.

**Notebook 03** — `:1320-1352`, 5 checks:

| # | Check | Gate |
|---|---|---|
| 1 | Both native kVp channels retained | `length(bins) == 2` |
| 2 | All final values finite | `all(isfinite, dual_final.vmis)` |
| 3 | Solid-water worst absolute HU | `maximum(abs, water_mean) ≤ 10` HU |
| 4 | Noise decreases 50→140 keV | `all(diff(water_noise) .< 0)` — **hard monotone, no tolerance** |
| 5 | Per-basis FBP kernels | `(water = :StandardSoftBlend, iodine = :OriginalDualKvpSoft)` |

Water ROI = `erode_mask_2d(mask .== UInt8(REGION_SOLID_WATER); erode_px = 12.0)`. VMI energies `[50,70,100,140]` keV. Per-rod measured-vs-theory (Ca 50–600 mg/mL labels 10–16; I 2.0–20 mg/mL labels 20–26; 16-px-radius disc ROIs) is plotted but **not gated** here.

**Notebook 04** — `:1537-1562`, 5 checks: 4 bins retained, `pcct_results.finite`, water `|HU| ≤ 10`, monotone noise, kernel `:SoftFilter`. ACNR `passes=4, beta_max=20.0, hp_sigma_px=1.5, window=4`.

**Notebook 07** — `:2141-2192`, 6 checks: 2 channels, `nchannel_basis.I0_relerr < 5e-5`, Cong outputs finite, canonical kernels, canonical energies `[50,70,100,140]`, final VMI finite. **No HU-accuracy gate** — the per-rod regression is plotted only.

**Notebook 08** — `:2372-2437`, 8 checks: 4 bins, `I0_relerr < 5e-5`, finite, Lee T-LBF identity (`implementation == :Lee_2025_joint_total_likelihood && shared_material_weights`), common kernel `:SoftFilter`, **water-reference basis calibration `0.8 ≤ c_water ≤ 1.2` and `abs(c_iodine) ≤ 0.01`**, canonical energies, finite.

**Notebook 12** — `:2151-2251`, **21 gates in 7 families**, `@info "🎉 VERIFICATION PASS n/21"`. The strongest oracle in the repo.

| Family | Count | Gate |
|---|---|---|
| 1. Regular dual-power water accuracy: tube A, tube B, combined | 3 | `abs(mean) ≤ 5.0` HU each |
| 2. Dual-power √2 noise reduction | 1 | `0.62 ≤ σ_comb/σ_single ≤ 0.80` (ideal 0.707) |
| 3. DE poly water accuracy: 100 kVp, Sn140, mixed | 3 | `abs(mean) ≤ 5.0` HU each |
| 4. VMI water accuracy @ 50/70/100/140 keV | 4 | `abs(mean) ≤ 10.0` HU each |
| 5. VMI noise monotonic decreasing | 1 | `all(diff(σs) .< 0)` — hard gate |
| 6. Per-basis FBP kernels | 1 | exact tuple match |
| 7. Per-rod regression, `:Ca`/`:I` × 4 keV | 8 | `0.85 ≤ slope ≤ 1.15` **and** `R² ≥ 0.99` |

**Gate 2 doubles as a determinism test**: it fails if the two `SimOptions` share a seed. nb12 uses `seed=1234` for tube A at `:389` and `seed=4321` for tube B at `:400`. Notebook 01 states the seed contract at `:373`.

**Ungated notebooks:**

| NB | What it measures | Target |
|---|---|---|
| 02 | FBP vs Hybrid IR σ in an eroded myocardium ROI, `:690-737` | 25–35 % noise reduction at `strength = 60` |
| 05 | Affine ≡ reconstructor grid, round-trip `A⁻¹Av = v`, recon registration, `:1349-1470` | max Δ ~0 µm, machine precision, best edge-alignment shift `(0,0)` for both axial and helical |
| 06 | CatSim vs BasisSim CPU vs GPU 3-panel mosaic + timing, `:860-1000` | qualitative agreement, HU window `(-200,600)`, `use_noise=false` |
| 09 | Per-rod regression table + water-ROI mean HU per keV, `:1140-1260`, `:1543-1578` | prose only. **nb12 is its gated successor** |
| 10 | Dark-band mean HU, water-ROI mean HU, titanium peak HU, `:225-241` | explicitly "the regression target to preserve when a future MAR method is added" (`:270`) |
| 11 | Water-ROI mean HU vs z for helical and axial, `:399-420` | "both water curves nearly flat" at the `z = ±5 cm` station boundaries |

**If the regression budget is limited, run 01, 05, and 12 first.** Notebook 01 is the best single-energy theory oracle, 05 is the only geometry/affine oracle, 12 has the most coded gates.

### 6.3 Per-notebook five-struct construction

| NB | Scanner highlights | Protocol | SimOptions | ReconOptions |
|---|---|---|---|---|
| 01 | GE Revolution Apex Elite: SID 625.6, SDD 1100, 256×834, 0.625/0.6 mm, `:lumex` 3.0 mm, bowtie `:ge_revolution_large`, e-noise 3500 | 120 kVp, 200 & 50 mA, 500 views, 1.0 s, 5 mm coll, Al 4.5 | `:eict`, seed 1234, `:dd_fast` | `(512,512,8)`, 35 cm, 0.5 cm |
| 02 | same as 01 | 120 kVp, 250 mA, 500 views | `:eict`, seed 1234 | `(512,512,8)`, 35, 0.5 |
| 03 | 01 hardware, **e-noise 0** | 80 kVp/407 mA + 140 kVp/405 mA, 984 views, 0.5 s | `:eict`, seed 1234 | `(512,512,8)`, 35, 0.5 |
| 04 | Naeotom-class PCCT: SID 610, SDD 1113, 144 rows, `:cdte` 1.6 mm, 4 bins `[20,35,55,70]`, `charge_sharing_fwhm 0.08`, `dead_time_ns 5.0`, `binning_factor 2` | 140 kVp, 174 mA, 1200 views, Ti 0.9 | `:pcct`, seed 1234, `nr = 0.0`, scatter+pileup on **with** corrections | `(512,512,coll/0.4)`, 35 |
| 05 | 01 hardware | 120 kVp, 250 mA, 500 views; helical `pitch=1.0, n_rotations=8` | `:eict`, seed 1234 | `(384,384,8)` 14 cm; helical `(384,384,64)` 14 cm, 4 cm |
| 06 | 01 hardware, e-noise 0 | 120 kVp, 200 mA, 500 views, 4 mm coll | `:eict`, seed 1234, **`use_noise=false`** | `(256,256,6)`, 35, 0.4 |
| 07 | 01 hardware, e-noise 0 | 80/140 kVp split-mA, 984 views, 2.5 mm coll | `:eict`, seed 1234 | `(512,512,3)`, 32, 0.1875 |
| 08 | 04 PCCT but **bowtie `:none`** | 140 kVp, 174 mA, 1200 views, Ti 0.9 | `:pcct`, **`pcct_noise_reduction = 0.7`** | `(512,512,3)`, 32, 0.1875 |
| 09 | SOMATOM Force: SID 595, SDD 1085.6, 96×920, `:ufc` 1.4 mm, bowtie `:large_body`, e-noise 0 | 100 kVp/380 mA (Ti 0.9) + 140 kVp/190 mA (Ti 0.9 + Sn 0.6), 1160 views | `:eict`, seed 1234, **`use_heel_effect=false`** | `(512,512,5)`, 35, 0.30 |
| 10 | minimal: SID 540, SDD 950, 16×512, `:lumex` | 120 kVp, 200 mA, 360 views, no filters | `:eict`, seed 42 | `(256,256,8)`, 30, **no `z_cm`** |
| 11 | SID 541, SDD 949, 256×512 | helical 120 kVp/200 mA, 20 mm coll, pitch 1.0, 16 rot; axial 400/3 mA, 160 mm coll | `:eict`, seed 42, `use_heel_effect=false` | `(160,160,150)` 30 cm/30 cm; `(160,160,50)` 30/10 |
| 12 | SOMATOM Definition Flash: SID 595, SDD 1085.6, 64×736, `:ufc_flash` 1.0 mm, flat filter 8.4 mm, e-noise 1500 | 120/420, 100/460, 140/356+Sn 0.4, 1152 views | **two `SimOptions`, seeds 1234 and 4321** | `(512,512,5)`, 35, 0.30 |

`phantom_b` in notebooks 09 and 12 is the same arrays with `origin[3] − 0.088 cm` for the tube-B detector z-offset.

---

## 7. Trace-hostility summary for the Reactant/Enzyme port

Ranked by how much work each will cost.

1. **`apply_pcct_noise!`** (`photon_counting.jl:822`). Integer Poisson with unbounded rejection loops, `max(N,1)` floor, host round trip, `Float64` inner math. Needs a reparameterized or relaxed sampler plus threading the count-domain identity `I0_b·exp(-bin) = recorded`.
2. **The 3-deep data-dependent `while` nest** in five DD kernels (`dd.jl:604-634` ≡ `:731-761` ≡ `dd_fast.jl:130-160` ≡ `:225-255` ≡ `dd.jl:241-271`), and the Siddon DDA `while t < t_exit && iter < max_iter` with `break`. Also `dd_transpose.jl:351-505`, the deepest nest, with four corner-enumeration loops.
3. **`vertical = abs(sy) >= abs(sx)`** at `dd.jl:142`, `dd_transpose.jl:126`, `:363` — a runtime-value-dependent **index permutation** that selects which array axis `it` and `il` index. `arc_det::Bool` similarly picks between two entirely different detector-boundary formulas.
4. **Runtime `Val`**, forcing a fresh specialization per call-site value: `Val(size(μ_table_gpu,1))` at `dd_fast.jl:362`, `:438`; `Val(n_energies)` at `polychromatic.jl:481`, `:510`; `Val(n_energies_padded)` at `photon_counting.jl:507`. Plus runtime-`Int` loop bounds needing `Val`-lifting: BHC `order` (`bhc_sinogram.jl:333`), lag `n_frames` (`detector_lag.jl:142`), focal-spot and scatter `half_k`.
5. **Uncached file IO** on the calibration path: spectrum, bowtie, and the 7.5 MB MC DRM are re-read on every call. **No memoization anywhere** — the only "cache" is field storage on the mutable workspaces.
6. **CPU LAPACK SVD** at five sites: per detector row in `sino_svd.jl:133`, `:299` and `sino_sfjsd.jl:219`, `:511`; over the **entire volume set** in `rskr.jl:285`.
7. **`Sys.free_memory()`-derived tile sizes** at `memory_budget.jl:207` make kernel-visible array shapes environment-dependent and non-reproducible. Kernels see `SubArray{T,3}` with a runtime-varying third extent, and the tail tile is smaller. `_is_oom` matches exception **type names as strings** at `:254`.
8. **`apply_bowtie_to_spectrum` is ndims-polymorphic**, returning a `Vector{Float32}` or a 3-D `Array{Float32,3}` depending on config (`driver.jl:841-845` vs `:874`). Downstream dispatches on `ndims(ŵ)` at `bhc_sinogram.jl:482-501` and `cong.jl:175`.
9. **`PhysicsConfig` fields are all `Union{Nothing,…}`** (`physics_pipeline.jl:38-46`), so every effect is a `!== nothing` branch.
10. **Early `continue` / `break` / `return` inside per-thread loops**: `dd.jl:485`; `dd_transpose.jl:115`, `:345`; `siddon.jl:308`, `:960`, `:1022`, `:1391`; `backprojection.jl:90`, `:258`, `:286`, `:292`, `:427`. All must become `select` or masked accumulation.
11. **Non-differentiable points**: `sign` in `_huber_deriv` (`ir/utils.jl:58`), `abs`, `_wq_aperture` piecewise boundaries (`backprojection.jl:179-186`), every `clamp` saturation, all bilinear bounds branches, `max(norm, Wq)` (`backprojection.jl:334`), `sumW > 1e-8` (`wfbp_helical.jl:230`), the `> eps` reciprocal switches (`ir/utils.jl:207`, `:240`), and `round(Int,…)` in the energy→DRM-row map at six sites.
12. **`similar(…)` allocations inside "in-place" entry points**: the `nothing`-bowtie dummy `similar(μ_table_gpu, T, 1, 1, 1)` at `dd.jl:577`, `:703`; `dd_fast.jl:359`, `:435`; `siddon.jl:867`, `:1255`. Rationale at `siddon.jl:864-866`: capturing `nothing` makes the closure compile a `getindex(::Nothing,…)` branch and emit invalid Metal IR. A pure rewrite must keep some equivalent.
13. **Fully host-serial, not traceable at all**: `measure_radial_cupping`, `apply_radial_cupping_correction!`, `apply_radial_capping_basis!` (`push!`-grown vectors, `quantile`, dense `A \ vals` with a data-dependent row count), `apply_acnr_kalender!`, every denoiser, `estimate_phantom_diameter_cm` (forces `Array(mask)`), `pcct_material_decomposition` (`pcct_spectral.jl:271-285`).
14. **`CartesianIndices(volume)[idx]` inside a kernel** at `fdk.jl:477` — every other kernel uses manual `%`/`÷`.
15. **Aliasing**: `ws.work_volume === ws.volume` when there is no HIR halo (`workspace.jl:1030`).
16. **Mixed Int32/Int64 index arithmetic inside a kernel** at `polychromatic.jl:637` — `col`/`row` are `Int32` but `nc` and `bt_off` are host `Int64`, promoting to 64-bit index math. Every other kernel keeps this in `Int32`.
17. **`@warn`/`@info`/`println` inside numeric code** at ~15 sites, and `Base.depwarn` on live-ish entry points (`bhc_sinogram.jl:392`, `:433`, `:714`; `pcct_spectral.jl:202`; `select_projector.jl:47`).
18. **Mutation of a `const` global at load time** — `detector_efficiency.jl:95-99`, `:117-124`.
19. **The `-log` → `exp` → `-log` round trip** in the tiled polychromatic loop (`polychromatic.jl:569`, `:577`) and in EICT calibration steps 6–10. A precision hazard, but the oracle depends on it.
20. **Dead or divergent oracle branches to exclude deliberately**: `backproject_voxel_helical` (`backprojection.jl:210`) is unreachable; `apply_heel_effect!` (`heel_effect.jl:149`) is never called by `simulate!`; `wfbp_helical_reconstruct` omits the FOV mask the workspace path applies; `generate_focal_spot_samples` (`focal_spot.jl:498`) is dead and its `:bimodal` branch produces an empty sample set.

### Cheapest to port, already pure

`_dd_overlap` (`dd.jl:80`), `_dd_bounds` (`:93`), `_dd_col_setup` (`:110`), `_dd_row_setup` (`:170`), `_dd_cell_setup` (`:194`), `_dd_arc_row_bounds` (`dd_transpose.jl:9`), `_dd_project_point` (`:27`), `_wq_aperture` (`backprojection.jl:177`), `_huber`/`_huber_deriv` (`ir/utils.jl:44`, `:53`), `_catsim_apodization_window` (`filtering.jl:228`), `get_hir_params` (`hybrid_ir.jl:126`), and all eight `@generated` NTuple helpers (`siddon.jl:717,728,747,1098,1113,1130`; `dd_fast.jl:56,62`). These are value-in / value-out and already allocation-free.

### Best oracles already in-tree

| Oracle | Location | Tolerance |
|---|---|---|
| DD exact-adjoint dot product | `test/projection.jl:56-58` | `rtol = atol = 2e-11` |
| DD brute-force matrix adjoint | `test/projection.jl:78-93` | `rtol = 2e-12` |
| DD determinism | `test/projection.jl:59-61` | bit-identical |
| `dd_fast` vs `dd` agreement | `test/projection.jl:300,313,335,359` | mean rel 5e-7 |
| `_project_mono` equivalence | `test/projection.jl:438-457` | — |
| `create_μ_volume!` Float32/Float64 | `test/projection.jl:380-417` | — |
| GPU Brent vs `Roots.jl` | `test/archived/vmi_brent_parity.jl:9` | `4·eps·max(1,\|root\|)`. **Not in `runtests.jl`** |
| Notebook 01 per-rod theory + dose invariance | nb01 `:970-978` | ±15 HU/15 %, ±10 HU/3 % |
| Notebook 05 affine round trip | nb05 `:1391-1465` | machine precision, shift `(0,0)` |
| Notebook 12 full gate suite | nb12 `:2172-2247` | 21 gates including slope/R² |

---

## 8. Files to read first for the refactor

| Purpose | Path |
|---|---|
| Everything orchestrates from here | `src/api/driver.jl:117` (PCCT) and `:423` (EICT) |
| Every buffer and table is allocated here | `src/api/workspace.jl` |
| The production forward kernel | `src/projection/dd_fast.jl:93-186` |
| The shared DD geometry primitive, reused verbatim by five kernels | `src/projection/dd.jl:194` |
| The verified exact adjoint plus its oracle test | `src/projection/dd_transpose.jl:282`, `test/projection.jl:31-95` |
| The four-path polychromatic driver | `src/projection/polychromatic.jl:423` |
| The backprojection kernel | `src/reconstruction/core/backprojection.jl:37-162` |
| The helical oracle | `src/reconstruction/fbp/wfbp_helical.jl:171-237` |
| The HIR loop | `src/api/driver.jl:1404-1640` |
| The GPU root-finder | `src/reconstruction/vmi/roots_kernels.jl:96` |
| The Cong per-ray solve | `src/reconstruction/vmi/cong.jl:195-363` |
| The published estimator, which is **not** in src | `docs/notebooks/04_pcct_vmi.jl:566` |
| The strongest numerical gate suite | `docs/notebooks/12_siemens_flash_ufc.jl:2151-2251` |
