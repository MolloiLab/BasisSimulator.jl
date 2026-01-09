#=
Visual Validation: BasisSimulator.jl vs GECATSIM

This script generates a comprehensive visual comparison between BasisSimulator.jl
and GECATSIM (NIH reference CT simulator) for qualitative validation.

Output: A single PNG file with 8-panel comparison showing:
  - Sinogram comparisons (central slice)
  - Reconstruction comparisons (axial slice)
  - Difference maps
  - Quantitative metrics (HU profiles, scatter plots)

Usage:
  julia --project=. test/visual_validation.jl

Requirements (optional extras):
  - CairoMakie.jl (visualization)
  - PythonCall.jl (GECATSIM integration)
  - CondaPkg.jl (Python environment)

Output Location:
  test/outputs/visual_comparison.png

Known Limitations:
  - Currently uses water cylinder phantom only (simple validation)
  - Gammex 472 materials (Ca_50, I_2_0, etc.) not yet defined in XrayAttenuation.jl
  - Full GECATSIM integration pending (placeholder implementation)
  - Falls back to BasisSimulator-only visualization if GECATSIM not available
=#

using BasisSimulator
import XrayAttenuation as XA

# Check for optional dependencies
try
    using CairoMakie
    using Statistics
    MAKIE_AVAILABLE = true
catch e
    @warn "CairoMakie not available. Install with: using Pkg; Pkg.add(\"CairoMakie\")"
    MAKIE_AVAILABLE = false
    exit(1)
end

# Check for GECATSIM
GECATSIM_AVAILABLE = false
try
    using PythonCall
    gecatsim = pyimport("gecatsim")
    GECATSIM_AVAILABLE = true
    @info "GECATSIM detected - will generate comparative figures"
catch e
    @warn """
    GECATSIM not available. Install with:
        pip install gecatsim

    Will generate BasisSimulator-only validation figures.
    """
end

# =============================================================================
# Configuration
# =============================================================================

const OUTPUT_DIR = joinpath(@__DIR__, "outputs")
const OUTPUT_FILE = joinpath(OUTPUT_DIR, "visual_comparison.png")

# Simulation parameters
const PHANTOM_PARAMS = (
    diameter_mm = 200.0,
    height_mm = 40.0,
    resolution_mm = 2.0
)

const SCAN_PARAMS = (
    kVp = 120.0,
    mAs = 200.0,
    scan_fov_mm = 250.0,
    num_projections = 360
)

# =============================================================================
# Helper Functions
# =============================================================================

"""
    run_basis_simulator() -> (sinogram, reconstruction, phantom, geometry)

Run BasisSimulator.jl forward simulation and reconstruction.
"""
function run_basis_simulator()
    @info "Running BasisSimulator.jl simulation..."

    # Create phantom
    phantom = create_water_cylinder(
        diameter_mm = PHANTOM_PARAMS.diameter_mm,
        height_mm = PHANTOM_PARAMS.height_mm,
        resolution_mm = PHANTOM_PARAMS.resolution_mm
    )

    # Define scanner
    protocol = ScanProtocol(
        kVp = SCAN_PARAMS.kVp,
        mAs = SCAN_PARAMS.mAs,
        scan_fov_mm = SCAN_PARAMS.scan_fov_mm,
        num_projections = SCAN_PARAMS.num_projections
    )
    geometry = create_aquilion_one(protocol=protocol)

    # Generate spectrum
    spectrum = generate_spectrum(
        kVp = SCAN_PARAMS.kVp,
        mAs = SCAN_PARAMS.mAs
    )

    # Run forward simulation
    sinogram = simulate_ct_scan(
        phantom = phantom,
        geometry = geometry,
        spectrum = spectrum,
        verbose = true
    )

    # Reconstruct
    recon_fov_cm = SCAN_PARAMS.scan_fov_mm / 10.0
    recon_size = 256
    recon_coords = range(-recon_fov_cm/2, recon_fov_cm/2, length=recon_size)
    z_coords = range(-phantom.grid.fov_z_cm/2, phantom.grid.fov_z_cm/2, length=64)

    reconstruction = reconstruct_fdk(
        sinogram,
        geometry.SAD_cm,
        geometry.SDD_cm,
        geometry.pixel_width_cm,
        geometry.pixel_height_cm,
        rad2deg.(geometry.angles),
        collect(recon_coords),
        collect(recon_coords),
        collect(z_coords)
    )

    # Convert to HU
    μ_water = get_linear_attenuation(XA.Materials.water, 60.0)
    reconstruction_hu = convert_to_hounsfield_units(reconstruction, μ_water)

    @info "BasisSimulator.jl simulation complete"
    @info "  Sinogram: $(size(sinogram))"
    @info "  Reconstruction: $(size(reconstruction_hu))"
    @info "  HU range: $(minimum(reconstruction_hu)) to $(maximum(reconstruction_hu))"

    return sinogram, reconstruction_hu, phantom, geometry
end

"""
    run_gecatsim() -> (sinogram, reconstruction)

Run GECATSIM simulation with equivalent parameters.
Returns nothing if GECATSIM not available.
"""
function run_gecatsim()
    if !GECATSIM_AVAILABLE
        return nothing, nothing
    end

    @info "Running GECATSIM simulation..."

    # TODO: Implement GECATSIM Python interop
    # This is a placeholder - actual implementation requires:
    # 1. Configure GECATSIM phantom (equivalent to water cylinder)
    # 2. Configure scanner geometry (Aquilion ONE equivalent)
    # 3. Run forward projection
    # 4. Reconstruct with FDK

    @warn "GECATSIM integration not yet implemented - using placeholder"

    # Placeholder: return nothing to indicate not available
    return nothing, nothing
end

"""
    create_comparison_figure(sino_basis, recon_basis, sino_gecat, recon_gecat)

Create comprehensive 8-panel comparison figure.
"""
function create_comparison_figure(
    sino_basis::Array{Float64, 3},
    recon_basis::Array{Float64, 3},
    sino_gecat::Union{Nothing, Array{Float64, 3}},
    recon_gecat::Union{Nothing, Array{Float64, 3}}
)
    @info "Generating comparison figure..."

    # Figure setup
    fig = Figure(resolution=(1600, 1200), fontsize=12)

    # Get central slices
    sino_slice_basis = sino_basis[:, :, div(size(sino_basis, 3), 2)]
    recon_slice_basis = recon_basis[:, :, div(size(recon_basis, 3), 2)]

    has_gecatsim = !isnothing(sino_gecat) && !isnothing(recon_gecat)

    if has_gecatsim
        # Full comparison mode (BasisSimulator vs GECATSIM)
        sino_slice_gecat = sino_gecat[:, :, div(size(sino_gecat, 3), 2)]
        recon_slice_gecat = recon_gecat[:, :, div(size(recon_gecat, 3), 2)]

        # Row 1: Sinogram comparison
        ax1 = Axis(fig[1, 1], title="BasisSimulator - Sinogram", aspect=DataAspect())
        heatmap!(ax1, sino_slice_basis', colormap=:grays)

        ax2 = Axis(fig[1, 2], title="GECATSIM - Sinogram", aspect=DataAspect())
        heatmap!(ax2, sino_slice_gecat', colormap=:grays)

        ax3 = Axis(fig[1, 3], title="Difference (BasisSim - GECATSIM)", aspect=DataAspect())
        diff_sino = sino_slice_basis .- sino_slice_gecat
        heatmap!(ax3, diff_sino', colormap=:RdBu, colorrange=(-maximum(abs.(diff_sino)), maximum(abs.(diff_sino))))

        # Row 2: Reconstruction comparison
        ax4 = Axis(fig[2, 1], title="BasisSimulator - Reconstruction (HU)", aspect=DataAspect())
        heatmap!(ax4, recon_slice_basis', colormap=:grays, colorrange=(-1000, 1000))

        ax5 = Axis(fig[2, 2], title="GECATSIM - Reconstruction (HU)", aspect=DataAspect())
        heatmap!(ax5, recon_slice_gecat', colormap=:grays, colorrange=(-1000, 1000))

        ax6 = Axis(fig[2, 3], title="Difference (BasisSim - GECATSIM)", aspect=DataAspect())
        diff_recon = recon_slice_basis .- recon_slice_gecat
        heatmap!(ax6, diff_recon', colormap=:RdBu, colorrange=(-100, 100))

        # Row 3: Quantitative comparisons
        # HU profile through center
        ax7 = Axis(fig[3, 1:2],
                   title="HU Profile (Horizontal Line Through Center)",
                   xlabel="Position (pixels)",
                   ylabel="HU")
        center_row = div(size(recon_slice_basis, 1), 2)
        profile_basis = recon_slice_basis[center_row, :]
        profile_gecat = recon_slice_gecat[center_row, :]
        lines!(ax7, 1:length(profile_basis), profile_basis, label="BasisSimulator", linewidth=2)
        lines!(ax7, 1:length(profile_gecat), profile_gecat, label="GECATSIM", linewidth=2, linestyle=:dash)
        axislegend(ax7, position=:rt)
        ylims!(ax7, -1200, 200)

        # Scatter plot
        ax8 = Axis(fig[3, 3],
                   title="Pixel-by-Pixel Comparison",
                   xlabel="GECATSIM HU",
                   ylabel="BasisSimulator HU")
        # Downsample for visualization
        step = 5
        scatter!(ax8, recon_slice_gecat[1:step:end, 1:step:end][:],
                      recon_slice_basis[1:step:end, 1:step:end][:],
                      markersize=2, alpha=0.3)
        # Identity line
        lims = (-1000, 100)
        lines!(ax8, lims, lims, color=:red, linestyle=:dash, linewidth=2, label="y=x")
        xlims!(ax8, lims)
        ylims!(ax8, lims)

        # Compute metrics
        rmse = sqrt(mean((recon_slice_basis .- recon_slice_gecat).^2))
        mae = mean(abs.(recon_slice_basis .- recon_slice_gecat))

        Label(fig[4, :],
              text="RMSE: $(round(rmse, digits=2)) HU  |  MAE: $(round(mae, digits=2)) HU",
              fontsize=14, font=:bold, halign=:center)

    else
        # BasisSimulator-only mode (no GECATSIM available)
        @info "Generating BasisSimulator-only validation figures"

        # Row 1: Sinogram views
        ax1 = Axis(fig[1, 1], title="Sinogram - Central Slice", aspect=DataAspect())
        heatmap!(ax1, sino_slice_basis', colormap=:grays)

        ax2 = Axis(fig[1, 2], title="Sinogram - Row Profile",
                   xlabel="Detector Column", ylabel="Intensity")
        center_row_sino = div(size(sino_slice_basis, 1), 2)
        lines!(ax2, sino_slice_basis[center_row_sino, :])

        # Row 2: Reconstruction views
        ax3 = Axis(fig[2, 1], title="Reconstruction - Axial Slice (HU)", aspect=DataAspect())
        hm = heatmap!(ax3, recon_slice_basis', colormap=:grays, colorrange=(-1000, 1000))
        Colorbar(fig[2, 2], hm, label="HU")

        # Row 3: Quantitative analysis
        ax4 = Axis(fig[3, 1:2],
                   title="HU Profile (Horizontal Line Through Center)",
                   xlabel="Position (pixels)",
                   ylabel="HU")
        center_row = div(size(recon_slice_basis, 1), 2)
        profile = recon_slice_basis[center_row, :]
        lines!(ax4, 1:length(profile), profile, linewidth=2)
        ylims!(ax4, -1200, 200)
        hlines!(ax4, [0.0], color=:red, linestyle=:dash, linewidth=1, label="Water (0 HU)")
        axislegend(ax4, position=:rt)

        # HU histogram
        ax5 = Axis(fig[3, 3],
                   title="HU Histogram",
                   xlabel="HU",
                   ylabel="Frequency")
        hist!(ax5, recon_slice_basis[:], bins=50, color=(:blue, 0.5))
        vlines!(ax5, [0.0], color=:red, linestyle=:dash, linewidth=2, label="Water")
        axislegend(ax5, position=:rt)

        # Statistics
        mean_hu = mean(recon_slice_basis)
        std_hu = std(recon_slice_basis)
        Label(fig[4, :],
              text="Water Cylinder: Mean HU = $(round(mean_hu, digits=1)) ± $(round(std_hu, digits=1))",
              fontsize=14, font=:bold, halign=:center)
    end

    # Overall title
    supertitle = has_gecatsim ?
        "BasisSimulator.jl vs GECATSIM Validation" :
        "BasisSimulator.jl Validation (Water Cylinder Phantom)"
    Label(fig[0, :], text=supertitle, fontsize=18, font=:bold, halign=:center)

    return fig
end

# =============================================================================
# Main Execution
# =============================================================================

function main()
    @info "="^70
    @info "Visual Validation: BasisSimulator.jl"
    @info "="^70

    # Create output directory
    mkpath(OUTPUT_DIR)
    @info "Output directory: $OUTPUT_DIR"

    # Run simulations
    sino_basis, recon_basis, phantom, geometry = run_basis_simulator()
    sino_gecat, recon_gecat = run_gecatsim()

    # Generate comparison figure
    fig = create_comparison_figure(sino_basis, recon_basis, sino_gecat, recon_gecat)

    # Save figure
    @info "Saving figure to: $OUTPUT_FILE"
    save(OUTPUT_FILE, fig, px_per_unit=2)  # High-resolution output

    @info "="^70
    @info "Visual validation complete!"
    @info "Output: $OUTPUT_FILE"
    @info "="^70

    # Display success message
    if GECATSIM_AVAILABLE && !isnothing(sino_gecat)
        @info """
        ✓ Generated BasisSimulator vs GECATSIM comparison
        ✓ Check sinogram and reconstruction agreement
        ✓ Examine difference maps for systematic errors
        ✓ Review HU profiles and scatter plot for correlation
        """
    else
        @info """
        ✓ Generated BasisSimulator validation figures
        ✓ Check HU values (water should be near 0 HU)
        ✓ Examine noise characteristics
        ✓ Install GECATSIM for comparative validation:
            pip install gecatsim
        """
    end
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
