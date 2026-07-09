"""
    BasisSimulator.jl

CT simulation with backend-agnostic GPU/CPU execution via AcceleratedKernels.jl.

Core algorithms ported from TIGRE (CERN/TIGRE).
Works on: Metal (Apple), CUDA (NVIDIA), ROCm (AMD), or CPU.

# Quick Start
```julia
using BasisSimulator

# Create phantom and geometry
phantom = create_gammex_472(n_voxels=128, fov_cm=35.0, z_cm=4.0)
geom = create_aquilion_one(n_angles=180, n_rows=32, n_cols=256, fov_cm=35.0)

# Get μ at desired energy (v20.0-pivot: compute on demand)
μ = compute_μ(phantom, 60.0)  # 60 keV

# Forward projection
sinogram = siddon_forward_project(μ, geom)

# FDK reconstruction
recon = fdk_reconstruct(sinogram, geom, size(phantom.mask))
```
"""
module BasisSimulator

using LinearAlgebra
using Statistics
using Random
using DelimitedFiles
using Serialization
using Unitful
using FFTW
import AcceleratedKernels as AK
import XrayAttenuation as XA

# XCAT phantom download/load (artifact store + XCIST voxelized parsing)
import Artifacts
import Downloads
import JSON
import SHA
import p7zip_jll

# Re-export for convenience
export XA

# =============================================================================
# Source (X-ray source + beam shaping)
# =============================================================================

# Spectrum loading (from .dat files)
include("source/spectrum.jl")

# =============================================================================
# Object (scanned object: phantoms, materials, attenuation)
# =============================================================================

# Materials (Gammex 472)
include("object/materials.jl")

# Attenuation coefficient computation
include("object/attenuation.jl")


# =============================================================================
# Geometry
# =============================================================================

# Phantom generation with semantic masks
include("object/phantom.jl")

# CT scanner geometry
include("geometry/scanner.jl")

# Affine transforms: phantom ↔ world ↔ recon coordinate mapping
include("geometry/affine.jl")

# XCAT voxelized phantom download + load (XCIST phantoms-voxelized, BSD-3)
include("phantoms/xcat_artifacts.jl")


# =============================================================================
# Projection (ray tracing)
# =============================================================================

# Siddon forward projection (exact ray-voxel intersections)
# Reference: CERN/TIGRE/Common/CUDA/Siddon_projection.cu
include("projection/siddon.jl")

# Distance-driven forward projection (CatSim/XCIST DD3, gather form)
# Reference: De Man & Basu 2004; gecatsim DD3Proj_roi_notrans_mm.cpp
# Drop-in replacement for Siddon (same signatures, cm units, Σμ·l output).
include("projection/dd.jl")

# :dd_fast — same DD3 model, single-pass per-material path-length fused kernels
# (results ≡ :dd to float ordering; full spectrum in ONE volume walk).
include("projection/dd_fast.jl")

# Projector selection (:dd_fast default / :dd DEPRECATED / :siddon) — shared by the
# forward sim, the IR system matrix, and the BHC correction so the model stays
# consistent.
include("projection/select_projector.jl")

# =============================================================================
# Detector (ALL detector effects + noise)
# =============================================================================

# Scatter simulation (spatial convolution)
include("detector/scatter.jl")

# Protocol definitions
include("source/protocol.jl")


# Detector absorption efficiency and DQE
include("detector/detector_efficiency.jl")

# Bowtie filter modeling
include("source/bowtie_filter.jl")


# Finite focal spot modeling
include("source/focal_spot.jl")

# Detector optical crosstalk modeling
include("detector/optical_crosstalk.jl")

# Detector lag (afterglow) modeling
include("detector/detector_lag.jl")

# Detector fill factor modeling
include("detector/fill_factor.jl")


# =============================================================================
# CatSim-style Signal Processing (types without PhysicsConfig dependency)
# Must be loaded before PhysicsPipeline which references these types
# =============================================================================

# Heel effect (anode self-attenuation)
include("source/heel_effect.jl")

# Beam hardening correction (water-based polynomial, two-material per-column)
include("correction/bhc_sinogram.jl")
# Image-domain BHC refinement (So et al. 2009)
include("correction/bhc_image_domain.jl")
# Post-recon residual radial cupping/capping correction (HU domain)
include("correction/radial_cupping.jl")
# Per-basis radial capping correction (basis domain — sibling of radial_cupping)
include("correction/radial_capping_basis.jl")

# PCCT pulse-pileup correction (inverse of the MC pile-up applied by simulate!)
include("correction/pcct_pileup_correction.jl")

# =============================================================================
# Unified Physics Pipeline
# Depends on all physics effects above including CatSim-style
# =============================================================================

include("detector/physics_pipeline.jl")

# =============================================================================
# Correction (post-detection corrections)
# =============================================================================

# Calibration pipeline: air scan, offset, gain correction, log transform
include("correction/calibration.jl")


# =============================================================================
# Projection - Unified API (includes physics)
# =============================================================================

# Beer-Lambert spectral physics: I = Σ wₑ × exp(-∫μₑ dl)
# Unified API for mono/poly projection + optional physics effects
include("projection/polychromatic.jl")

# =============================================================================
# Reconstruction
# =============================================================================

# --- Core Components ---
# Voxel-driven backprojection
# Reference: CERN/TIGRE/Common/CUDA/voxel_backprojection.cu
include("reconstruction/core/backprojection.jl")

# FDK filtering (ramp filter, cosine weighting)
include("reconstruction/core/filtering.jl")

# --- FBP (Filtered Back-Projection) ---
# FDK reconstruction (filter + backproject)
include("reconstruction/fbp/fdk.jl")

# Helical WFBP (rebinned weighted FBP — Stierstorfer 2004 family); dispatched
# from fdk_reconstruct when is_helical(geom).
include("reconstruction/fbp/wfbp_helical.jl")

# --- Iterative Reconstruction Utilities ---
# Huber penalty, projection/image weights (shared by HIR)
include("reconstruction/ir/utils.jl")

# --- Hybrid IR ---
# Vendor-general Hybrid IR with PWLS refinement
# Reference: Geyer et al. 2015, Willemink & Noel 2019, SAFIRE clinical studies
include("reconstruction/hybrid_ir/hybrid_ir.jl")

# =============================================================================
# Photon-Counting CT (Spectral Imaging)
# =============================================================================

# Photon-counting detector model: energy binning, forward projection
include("detector/photon_counting.jl")

# MC-based detector response matrix (replaces analytical DRM)
include("detector/pcct/mc_response.jl")

# MC-based pulse pileup model (replaces analytical Taguchi model)
include("detector/pcct/mc_pileup.jl")

# PCCT spectral imaging: K-edge, effective Z, multi-material decomposition
include("spectral/pcct_spectral.jl")

# =============================================================================
# API (top-level orchestration)
# =============================================================================

include("api/options.jl")
include("api/workspace.jl")
include("api/driver.jl")

# =============================================================================
# Dual-Energy VMI Pipeline
# Photo/Compton basis decomp → PWLS restoration → ACNR → capping → VMI → Mono+
#
# Each stage is a `BS.apply_*!` entry point with every hyperparameter as a
# kwarg.  See `verification/notebooks/00_example_ge_gammex_phantom_dual.jl`
# for a reference wiring across §5-§6.
# =============================================================================

# Memory-budget probe + tile-loop helpers for VMI workspaces.
# Free utility functions (suggest_tile_size, tile_ranges, with_oom_retry)
# consumed by RwlsWorkspace / PwlsWorkspace / CongWorkspace / MonoPlusWorkspace.
include("reconstruction/workspace/memory_budget.jl")

# Photoelectric + Compton physical basis tables (Cong 2022 Eqs 3a-3e, 4)
include("reconstruction/vmi/basis.jl")

# GPU-safe 1:1 port of Roots.jl Brent + helper kernels (consumed by Cong).
# Parity against Roots.jl verified in test/vmi/test_brent_parity.jl.
include("reconstruction/vmi/roots_kernels.jl")

# Linear 2×2 DE (Constant Material Value) decomposition — cheap baseline
include("reconstruction/vmi/cmv.jl")

# Per-ray Cong 2022 analytic dual-energy decomposition
include("reconstruction/vmi/cong.jl")

# Ducros 2017 RWLS-GN DE decomp (count-domain GN + FFT proximal prior)
include("reconstruction/vmi/rwls.jl")

# Noh/Fessler 2009 + Long/Fessler 2014 PWLS-L₂ 2×2-curvature SQS
include("reconstruction/vmi/pwls.jl")

# PCCT per-bin basis + count-combine helpers (shared by CMV/RWLS/PWLS).
include("reconstruction/vmi/pcct_basis.jl")

# PCCT polynomial calibration: (p_low, p_high) → (t_water, t_iodine)
# Alvarez/Macovski 1976 sinogram-domain decomposition.
include("reconstruction/vmi/pcct_calibration.jl")

# VMI synthesis — μ(E) = p(E)·a + q(E)·c → HU
include("reconstruction/vmi/vmi_synth.jl")

# Mono+ frequency-split — Grant 2014 1:1 parity (FFT Gaussian LP).
# Optional `phantom_mask` kwarg masks the phantom-air ring artifact.
include("reconstruction/vmi/mono_plus.jl")

# =============================================================================
# Denoising — sinogram-domain and image-domain noise reduction
# =============================================================================
#
# Live in the current notebooks:
#   - sino_sfjsd.jl : two-channel projection-domain SF-JSD (Black, in prep.)
#                     used by nb03/04/07
#   - median_z.jl   : z-direction median filter (BasisSim-original)
#                     used by nb03/04/07
#
# Kept as advertised tools (not in current notebooks, but exported + tested):
#   - sino_svd.jl : N-channel projection-domain SVD joint denoiser
#                   (predecessor of SF-JSD)
#   - rskr.jl     : Rank-Sparse Kernel Regression — joint multi-channel
#                   image-domain denoiser (Clark/Badea 2023).  2-/4-ch.
#   - acnr.jl     : Anti-Correlated Noise Reduction sinogram-domain
#                   (Kalender/Klotz/Kostaridou 1988)

# Z-direction median filter — edge-preserving impulse-noise removal that
# exploits z-axis correlation in z-invariant phantoms (Gammex 472).
include("denoising/median_z.jl")

# RSKR (Rank-Sparse Kernel Regression) — joint multi-channel basis-pair
# denoiser (Clark/Badea 2023).  2-channel + 4-channel variants.
include("denoising/rskr.jl")

# Kalender/Klotz/Kostaridou 1988 ACNR (FFT Tikhonov smoother on s_⊥)
include("denoising/acnr.jl")

# N-channel projection-domain SVD joint denoiser — per-row SVD + 2D
# Gaussian on U[:, 2..N] residual components.  Works on any set of
# co-registered sinogram channels (dual-kVp pairs, PCCT bin sets, …).
# Inspired by RSKR's signal/noise SV split (see comment in
# `denoising/sino_svd.jl` for the comparison).
include("denoising/sino_svd.jl")

# Two-channel projection-domain Subspace–Frequency Joint Sinogram
# Denoiser (SF-JSD).  Per-row Poisson-whitened SVD + rank-sparse joint
# bilateral filter with locally-averaged range, MAD-derived per-component
# range scale, stride-based resampling, and SURE-tuned principal scale.
# Single user knob (σ₀); everything else fixed by RSKR / paper or
# auto-derived from the photon-count map.  Black (in prep.).
include("denoising/sino_sfjsd.jl")

# Phantom-mask helpers — recon-space resample + FFT-Gaussian erosion.
# Used by Mono+ phantom_mask kwarg + edge-mask post-processing.
include("reconstruction/vmi/phantom_mask.jl")

# Image-domain Ding 2012 DE decomp — fit (a₀, a₁, a₂) coeffs from rod
# HUs, apply to (HU_low, HU_high) → c_iodine, synth per-energy VMI.
include("reconstruction/vmi/image_domain_decomp.jl")

# Clinical rod-HU calibration constants (image-domain Ding fits).  Per-
# scanner Dicts of measured rod HUs at relevant kVp / VMI energies, plus
# iodine_calibration_rods / calcium_calibration_rods helpers.
include("reconstruction/vmi/clinical_calibrations.jl")

end # module
