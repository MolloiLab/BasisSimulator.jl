#!/usr/bin/env python3
"""
CatSim Runner for BasisSimulator Verification Pipeline

Runs CatSim with configurable phantom and scanner settings,
outputs sinogram and reconstruction for comparison with BasisSimulator.

Usage:
    python run_catsim.py --phantom water --scale dev --output results/
    python run_catsim.py --phantom gammex --scale integration --output results/

Output Files:
    - {output}/catsim_sinogram.npy: Raw sinogram [n_views, n_rows, n_cols]
    - {output}/catsim_recon.npy: Reconstruction [n_slices, n_rows, n_cols]
    - {output}/catsim_config.json: Configuration used for the run
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

# Add CatSim to path
CATSIM_PATH = Path(__file__).parent.parent.parent / "main"
sys.path.insert(0, str(CATSIM_PATH))

try:
    import gecatsim as xc
    import gecatsim.reconstruction.pyfiles.recon as recon
    CATSIM_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: CatSim not available: {e}")
    CATSIM_AVAILABLE = False

# =============================================================================
# Scale Configurations (match BasisSimulator)
# =============================================================================

SCALE_CONFIGS = {
    "dev": {
        "phantom_n_voxels": 64,
        "phantom_n_slices": 8,
        "n_views": 90,
        "n_rows": 16,
        "n_cols": 128,
        "recon_size": 64,
        "fov_cm": 35.0,
        "z_cm": 4.0,
    },
    "integration": {
        "phantom_n_voxels": 128,
        "phantom_n_slices": 16,
        "n_views": 180,
        "n_rows": 32,
        "n_cols": 256,
        "recon_size": 128,
        "fov_cm": 35.0,
        "z_cm": 4.0,
    },
    "verification": {
        "phantom_n_voxels": 256,
        "phantom_n_slices": 32,
        "n_views": 360,
        "n_rows": 64,
        "n_cols": 512,
        "recon_size": 256,
        "fov_cm": 35.0,
        "z_cm": 4.0,
    },
    "publication": {
        "phantom_n_voxels": 512,
        "phantom_n_slices": 64,
        "n_views": 900,
        "n_rows": 64,
        "n_cols": 736,
        "recon_size": 512,
        "fov_cm": 35.0,
        "z_cm": 4.0,
    },
}

# =============================================================================
# Phantom Definitions
# =============================================================================

def create_water_phantom_config(cfg, scale_config):
    """Create a simple water cylinder phantom configuration."""

    # Create a JSON phantom file for water cylinder
    phantom_data = {
        "voxels": {
            "x": scale_config["phantom_n_voxels"],
            "y": scale_config["phantom_n_voxels"],
            "z": scale_config["phantom_n_slices"]
        },
        "spacing": {
            "x": scale_config["fov_cm"] * 10 / scale_config["phantom_n_voxels"],  # mm
            "y": scale_config["fov_cm"] * 10 / scale_config["phantom_n_voxels"],
            "z": scale_config["z_cm"] * 10 / scale_config["phantom_n_slices"]
        },
        "objects": [
            {
                "type": "cylinder",
                "material": "water",
                "center": [0, 0, 0],
                "radius": 100.0,  # 200mm diameter = 100mm radius
                "height": scale_config["z_cm"] * 10  # mm
            }
        ]
    }

    return phantom_data


def create_gammex_phantom_config(cfg, scale_config):
    """Create Gammex 472 phantom configuration with inserts."""

    # Gammex 472 dimensions (mm)
    body_radius = 165.0  # 330mm diameter
    rod_radius = 14.0    # 28mm diameter
    inner_ring_radius = 50.0   # Calcium inserts
    outer_ring_radius = 105.0  # Iodine inserts

    objects = []

    # Body cylinder (solid water)
    objects.append({
        "type": "cylinder",
        "material": "water",
        "center": [0, 0, 0],
        "radius": body_radius,
        "height": scale_config["z_cm"] * 10
    })

    # Calcium inserts (7 at inner ring)
    # NOTE: CatSim material names may differ - using water as placeholder
    # Actual material files need to be created
    n_inserts = 7
    for i in range(n_inserts):
        angle = 2 * np.pi * i / n_inserts
        cx = inner_ring_radius * np.cos(angle)
        cy = inner_ring_radius * np.sin(angle)
        objects.append({
            "type": "cylinder",
            "material": "calcium_100",  # Placeholder
            "center": [cx, cy, 0],
            "radius": rod_radius,
            "height": scale_config["z_cm"] * 10
        })

    # Iodine inserts (7 at outer ring, offset by half spacing)
    for i in range(n_inserts):
        angle = 2 * np.pi * i / n_inserts + np.pi / n_inserts
        cx = outer_ring_radius * np.cos(angle)
        cy = outer_ring_radius * np.sin(angle)
        objects.append({
            "type": "cylinder",
            "material": "iodine_5",  # Placeholder
            "center": [cx, cy, 0],
            "radius": rod_radius,
            "height": scale_config["z_cm"] * 10
        })

    phantom_data = {
        "voxels": {
            "x": scale_config["phantom_n_voxels"],
            "y": scale_config["phantom_n_voxels"],
            "z": scale_config["phantom_n_slices"]
        },
        "spacing": {
            "x": scale_config["fov_cm"] * 10 / scale_config["phantom_n_voxels"],
            "y": scale_config["fov_cm"] * 10 / scale_config["phantom_n_voxels"],
            "z": scale_config["z_cm"] * 10 / scale_config["phantom_n_slices"]
        },
        "objects": objects
    }

    return phantom_data


# =============================================================================
# CatSim Runner
# =============================================================================

def run_catsim_simulation(phantom_type: str, scale: str, output_dir: str, kvp: int = 120):
    """
    Run CatSim simulation and save results.

    Args:
        phantom_type: "water" or "gammex"
        scale: "dev", "integration", "verification", or "publication"
        output_dir: Directory to save output files
        kvp: Tube voltage (default 120)

    Returns:
        dict with paths to output files and configuration
    """
    if not CATSIM_AVAILABLE:
        raise RuntimeError("CatSim is not available. Check installation.")

    scale_config = SCALE_CONFIGS[scale]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"CatSim Simulation: {phantom_type} phantom at {scale} scale")
    print(f"  Phantom: {scale_config['phantom_n_voxels']}^3")
    print(f"  Views: {scale_config['n_views']}")
    print(f"  Detector: {scale_config['n_cols']}x{scale_config['n_rows']}")
    print()

    # Use CatSim example configs as base
    # For water phantom, use the analytic projector and W20.ppm phantom
    # Reference: test_WaterPhantom.py in CatSim tests
    cfg_path = CATSIM_PATH / "gecatsim" / "examples" / "vct_examples"

    # Initialize CatSim with configs optimized for water phantom verification
    ct = xc.CatSim(
        str(cfg_path / "Phantom_Sample_Analytic"),
        str(cfg_path / "Physics_Sample"),
        str(cfg_path / "Protocol_Sample_axial"),
        str(cfg_path / "Recon_Sample_2d"),
        str(cfg_path / "Scanner_Sample_generic")
    )

    # Use water phantom file (W20 = 20cm water cylinder)
    ct.phantom.filename = 'W20.ppm'

    # Configure scanner geometry to match BasisSimulator
    ct.scanner.sid = 540.0  # Source-to-isocenter (mm) - matches generic scanner
    ct.scanner.sdd = 950.0  # Source-to-detector (mm)
    ct.scanner.detectorRowsPerMod = scale_config["n_rows"]
    ct.scanner.detectorRowCount = scale_config["n_rows"]
    ct.scanner.detectorColCount = scale_config["n_cols"]
    ct.scanner.detectorColSize = 1.0  # mm
    ct.scanner.detectorRowSize = 1.0  # mm

    # Configure protocol
    ct.protocol.viewsPerRotation = scale_config["n_views"]
    ct.protocol.viewCount = scale_config["n_views"]
    ct.protocol.stopViewId = scale_config["n_views"] - 1
    ct.protocol.mA = 200
    ct.protocol.rotationTime = 1.0

    # Configure spectrum (120 kVp default)
    # CatSim uses spectrum files in gecatsim/spectrum/
    if kvp == 120:
        ct.protocol.spectrumFilename = "tungsten_tar7.0_120_filt.dat"
    elif kvp == 80:
        ct.protocol.spectrumFilename = "tungsten_tar7.0_80_filt.dat"
    elif kvp == 100:
        ct.protocol.spectrumFilename = "tungsten_tar7.0_100_filt.dat"
    elif kvp == 140:
        ct.protocol.spectrumFilename = "tungsten_tar7.0_140_filt.dat"

    # Configure reconstruction
    ct.recon.fov = scale_config["fov_cm"] * 10  # mm
    ct.recon.sliceCount = scale_config["phantom_n_slices"]
    ct.recon.sliceThickness = scale_config["z_cm"] * 10 / scale_config["phantom_n_slices"]
    ct.recon.imageSize = scale_config["recon_size"]

    # Configure physics for proper HU output
    # These settings enable proper beam hardening correction
    # Reference: test_WaterPhantom.py in CatSim tests
    ct.physics.callback_post_log = 'Prep_BHC_Accurate'
    ct.physics.EffectiveMu = 0.2  # Effective water mu at ~120 kVp
    ct.physics.BHC_poly_order = 5
    ct.physics.BHC_max_length_mm = 500  # Max path length for BHC
    ct.physics.BHC_length_step_mm = 10

    # Configure reconstruction for HU output
    # Reference: cfg/GE_Revolution_Apex/Recon_GE_Revolution_Apex.cfg
    # Water attenuation at 120 kVp effective energy (~65 keV)
    ct.recon.unit = 'HU'        # Output in Hounsfield Units
    ct.recon.mu = 0.02          # Water attenuation coefficient (1/mm) at ~65 keV
    ct.recon.huOffset = -1000   # Standard HU offset (air = -1000)

    # Energy settings for polychromatic simulation
    ct.physics.energyCount = 24
    ct.physics.monochromatic = -1  # Polychromatic mode
    ct.physics.colSampleCount = 2
    ct.physics.rowSampleCount = 2
    ct.physics.srcXSampleCount = 2
    ct.physics.srcYSampleCount = 2
    ct.physics.viewSampleCount = 1

    # Set output name
    results_name = str(output_path / "catsim_temp")
    ct.resultsName = results_name

    # Run simulation
    print("Running CatSim simulation...")
    try:
        ct.run_all()
    except Exception as e:
        print(f"CatSim simulation failed: {e}")
        raise

    # Run reconstruction
    print("Running CatSim reconstruction...")
    ct.do_Recon = 1
    try:
        recon.recon(ct)
    except Exception as e:
        print(f"CatSim reconstruction failed: {e}")
        raise

    # Read results
    print("Reading results...")

    # Read sinogram (prep file)
    prep_file = f"{results_name}.prep"
    if os.path.exists(prep_file):
        sinogram = xc.rawread(
            prep_file,
            [ct.protocol.viewCount, ct.scanner.detectorRowCount, ct.scanner.detectorColCount],
            'float'
        )
        sinogram_path = output_path / "catsim_sinogram.npy"
        np.save(sinogram_path, sinogram)
        print(f"  Sinogram: {sinogram.shape} saved to {sinogram_path}")
    else:
        sinogram = None
        sinogram_path = None
        print(f"  WARNING: Sinogram file not found: {prep_file}")

    # Read reconstruction
    img_file = f"{results_name}_{ct.recon.imageSize}x{ct.recon.imageSize}x{ct.recon.sliceCount}.raw"
    if os.path.exists(img_file):
        recon_img = xc.rawread(
            img_file,
            [ct.recon.sliceCount, ct.recon.imageSize, ct.recon.imageSize],
            'float'
        )
        recon_path = output_path / "catsim_recon.npy"
        np.save(recon_path, recon_img)
        print(f"  Reconstruction: {recon_img.shape} saved to {recon_path}")
    else:
        recon_img = None
        recon_path = None
        print(f"  WARNING: Reconstruction file not found: {img_file}")

    # Save configuration
    config = {
        "phantom_type": phantom_type,
        "scale": scale,
        "kvp": kvp,
        "scale_config": scale_config,
        "scanner": {
            "sid": ct.scanner.sid,
            "sdd": ct.scanner.sdd,
            "n_rows": ct.scanner.detectorRowCount,
            "n_cols": ct.scanner.detectorColCount,
            "pixel_size_mm": ct.scanner.detectorColSize,
        },
        "protocol": {
            "views": ct.protocol.viewCount,
            "mA": ct.protocol.mA,
            "spectrum": ct.protocol.spectrumFilename,
        },
        "recon": {
            "fov_mm": ct.recon.fov,
            "image_size": ct.recon.imageSize,
            "slice_count": ct.recon.sliceCount,
        }
    }

    config_path = output_path / "catsim_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Config: saved to {config_path}")

    # Clean up temporary files
    for ext in ['.prep', '.air', '.offset', '.raw']:
        temp_file = f"{results_name}{ext}"
        if os.path.exists(temp_file):
            os.remove(temp_file)
    # Also clean reconstruction file
    if os.path.exists(img_file):
        os.remove(img_file)

    return {
        "sinogram_path": str(sinogram_path) if sinogram_path else None,
        "recon_path": str(recon_path) if recon_path else None,
        "config_path": str(config_path),
        "config": config,
        "sinogram_shape": sinogram.shape if sinogram is not None else None,
        "recon_shape": recon_img.shape if recon_img is not None else None,
    }


def run_catsim_water_phantom(scale: str, output_dir: str, kvp: int = 120):
    """Convenience function for water phantom simulation."""
    return run_catsim_simulation("water", scale, output_dir, kvp)


def run_catsim_gammex_phantom(scale: str, output_dir: str, kvp: int = 120):
    """Convenience function for Gammex 472 phantom simulation."""
    return run_catsim_simulation("gammex", scale, output_dir, kvp)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run CatSim simulation for BasisSimulator verification"
    )
    parser.add_argument(
        "--phantom",
        choices=["water", "gammex"],
        default="water",
        help="Phantom type (default: water)"
    )
    parser.add_argument(
        "--scale",
        choices=["dev", "integration", "verification", "publication"],
        default="dev",
        help="Simulation scale (default: dev)"
    )
    parser.add_argument(
        "--kvp",
        type=int,
        choices=[80, 100, 120, 140],
        default=120,
        help="Tube voltage in kVp (default: 120)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./catsim_results",
        help="Output directory (default: ./catsim_results)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Just check if CatSim is available"
    )

    args = parser.parse_args()

    if args.check:
        if CATSIM_AVAILABLE:
            print("CatSim is available and ready to use.")
            return 0
        else:
            print("CatSim is NOT available. Check installation.")
            return 1

    if not CATSIM_AVAILABLE:
        print("ERROR: CatSim is not available. Cannot run simulation.")
        return 1

    try:
        results = run_catsim_simulation(
            phantom_type=args.phantom,
            scale=args.scale,
            output_dir=args.output,
            kvp=args.kvp
        )
        print("\nSimulation complete!")
        print(f"  Sinogram: {results['sinogram_path']}")
        print(f"  Reconstruction: {results['recon_path']}")
        print(f"  Config: {results['config_path']}")
        return 0
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
