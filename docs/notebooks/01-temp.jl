### A Pluto.jl notebook ###
# v0.2.6

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ d0f8865c-104f-4ac9-88c9-e02dba318f50
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ b606834e-a070-4d27-a4ce-adb64be1c3fa
begin
    import BasisSimulator as BS
    import CairoMakie as CM
    import GPUSelect
    import PlutoUI
    using Statistics: mean, median, std
    using Printf: @sprintf
end

# ╔═╡ 5ef3b53c-43ef-4c10-a2a5-c30339e228dc
md"""
# 01-temp — Full-scale PCCT polychromatic FBP edge-slice check

This is the direct PCCT counterpart to notebook 01:

```text
Gammex 472 → full four-bin PCCT simulation → I₀-weighted poly combination → FDK
```

It deliberately excludes Cong decomposition, ACNR, Mono+, BHC, HIR, and all
post-reconstruction noise injection. The purpose is to inspect whether the
base PCCT polychromatic FBP develops terminal-slice radial streaks.
"""

# ╔═╡ 4976d9d7-d69c-4875-b93f-156bea4390f7
begin
    AT = GPUSelect.Storage()
    to_gpu(x) = AT(x)
    GPU_BACKEND = string(nameof(AT))
end

# ╔═╡ 4cb3cc2f-88f5-43ad-8b46-50c8d43b091f
md"""**Backend:** $(GPU_BACKEND)"""

# ╔═╡ b971e288-8085-4b5e-808a-a242d93243e1
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,
    n_slices = 16,
    fov_cm = 35.0,
    z_cm = 1.0,
);

# ╔═╡ d1550c3a-602b-4768-b4a5-75be3d9cf904
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 70490132-a100-4c84-9e13-aa81e4246ffc
scanner = let
    native_col_mm = 0.275
    native_row_mm = 0.322
    sid = 610.0
    sdd = 1113.0
    magnification = sdd / sid
    bf = 2
    pixel_col_iso = native_col_mm * bf / magnification
    pixel_row_iso = native_row_mm * bf / magnification

    BS.Scanner(
        source_to_isocenter = sid,
        source_to_detector = sdd,
        detector_rows = 144,
        detector_cols = ceil(Int, 360.0 / pixel_col_iso),
        detector_row_size = pixel_row_iso,
        detector_col_size = pixel_col_iso,
        detector_row_offset = 0.0,
        detector_col_offset = pixel_col_iso / 2,
        focal_spot_width = 0.4,
        focal_spot_length = 0.5,
        target_angle = 7.0,
        gantry_rotation_time = 0.5,
        scan_diameter = 360.0,
        gantry_aperture = 820.0,
        flat_filter_material = :aluminum,
        flat_filter_thickness = 3.0,
        detector_material = :cdte,
        detector_depth = 1.6,
        fill_factor_row = 0.95,
        fill_factor_col = 0.95,
        detection_gain = 1.0,
        electronic_noise = 0.0,
        detector_type = :photon_counting,
        n_energy_bins = 4,
        energy_thresholds = [20.0, 35.0, 55.0, 70.0],
        energy_resolution = 10.0,
        charge_sharing_fwhm = 0.08,
        dead_time_ns = 5.0,
        pixel_mode = :standard,
        native_dexel_col_mm = native_col_mm,
        native_dexel_row_mm = native_row_mm,
        binning_factor = bf,
    )
end;

# ╔═╡ fc1db85b-3385-413f-b3e9-25979861bee9
begin
    recon_extent_mm = 5.0
    protocol = BS.CTProtocol(
        kVp = 140,
        mA = 174.0,
        views = 1200,
        rotation_time = 0.5,
        collimation_mm = recon_extent_mm,
        additional_filters = [("Ti", 0.9)],
    )
end;

# ╔═╡ 0514fdeb-0adc-4eca-863f-1fa2ee76f754
sim_opts = BS.SimOptions(
    fidelity = :pcct,
    seed = 1234,
    projector = :dd_fast,
    use_scatter = false,
    use_noise = true,
    use_pcct_scatter = true,
    use_pcct_scatter_correction = true,
    use_pcct_pileup = true,
    use_pcct_pileup_correction = true,
    pcct_noise_reduction = 0.7,
);

# ╔═╡ 41138011-ac7b-4030-bb7f-d9c0a540b069
recon_opts = let
    slice_thickness_mm = 0.4
    n_slices = max(1, round(Int, recon_extent_mm / slice_thickness_mm))
    BS.ReconOptions(
        matrix_size = (512, 512, n_slices),
        fov_cm = 35.0,
        z_cm = recon_extent_mm / 10,
    )
end;

# ╔═╡ f384a1f7-1471-4542-9d2f-982913de015c
md"""
## Full PCCT forward simulation

This is the expensive cell. It returns CPU copies of the four corrected PCCT
bin sinograms and the geometry, then releases the GPU workspace.
"""

# ╔═╡ ef3a1763-9c60-400a-9fb0-ddd1276461b1
pcct_forward = let
    @info "Running full-scale PCCT forward" GPU_BACKEND
    ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    result = BS.simulate!(ws, phantom, protocol, sim_opts)
    output = (
        bins = [Array(b) for b in result.pcct_sino.bins],
        I0_bins = Float32.(result.I0_bins),
        geom = ws.geom,
        energies = Float64.(ws.energies),
        detected_weights = Float64.(Array(ws.W_matrix_gpu))[1:length(ws.energies), :],
        poly_bhc = BS.calibrate_pcct_poly_bhc(ws; max_path_cm = 40.0),
    )
    ws = nothing
    result = nothing
    GC.gc(true)
    output
end;

# ╔═╡ 8129a8c9-00ce-4fb8-a7f9-199a3a7ac2fc
geometry_report = let
    g = pcct_forward.geom
    (
        detector_columns = g.n_cols,
        active_detector_rows = g.n_rows,
        detector_row_pitch_mm = g.pixel_row_size * 10,
        acquired_row_support_mm = g.n_rows * g.pixel_row_size * 10,
        requested_recon_extent_mm = g.fov[3] * 10,
        minimum_cone_support_mm = BS.required_axial_detector_rows(
            scanner; fov_cm = recon_opts.fov_cm, z_cm = recon_opts.z_cm,
        ) * g.pixel_row_size * 10,
        reconstructed_slices = recon_opts.matrix_size[3],
        reconstructed_dz_mm = g.fov[3] * 10 / recon_opts.matrix_size[3],
    )
end

# ╔═╡ c6368c68-40e2-4113-bfab-9cb2cd68c22d
md"""
## Direct polychromatic bin combination

For bin line integral ``p_b=-\log(N_b/I_{0,b})``, recover counts, sum all four
bins, and return the combined line integral:

```math
p_{poly}=-\log\left(\frac{\sum_b I_{0,b}e^{-p_b}}{\sum_b I_{0,b}}\right).
```
"""

# ╔═╡ 453b47bb-7907-4669-baf7-e280db0831ce
sino_poly = let
    bins = pcct_forward.bins
    I0 = pcct_forward.I0_bins
    I0_total = sum(I0)
    counts = zeros(Float32, size(bins[1]))
    for b in eachindex(bins)
        @. counts += I0[b] * exp(-bins[b])
    end
    @. -log(max(counts, 1.0f-10) / I0_total)
end;

# ╔═╡ 8eac0ae0-5250-4e41-bbd2-5dd4e606f3d9
sinogram_report = (
    size = size(sino_poly),
    extrema = extrema(sino_poly),
    nonfinite = count(!isfinite, sino_poly),
);

# ╔═╡ 29f9fa1e-4fac-4bd3-87b5-7607e3aa3f89
bhc_calibration = let
    model = pcct_forward.poly_bhc
    (model = model, μ_water = model.μ_water_ref,
     reference_energy_keV = model.reference_energy_keV)
end;

# ╔═╡ b66b6120-d7d0-4210-b3a3-9595ba910b8e
hu_poly_fbp = let
    sino_gpu = to_gpu(sino_poly)
    sino_bhc = BS.apply_bhc_water(sino_gpu, bhc_calibration.model)
    ws = BS.create_fdk_recon_workspace(
        sino_bhc,
        pcct_forward.geom,
        recon_opts.matrix_size;
        filter = BS.StandardFilter(),
    )
    recon_μ = BS.reconstruct!(ws, sino_bhc, pcct_forward.geom)
    hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc_calibration.μ_water))
    ws = nothing
    sino_gpu = nothing
    sino_bhc = nothing
    recon_μ = nothing
    GC.gc(true)
    hu
end;

# ╔═╡ 31f07a99-305d-453e-8b22-a848830fe43c
water_metrics = let
    labels = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2 + 1]

    function eroded_label_mask(label::UInt8)
        m = labels .== label
        # One-pixel 8-connected erosion removes insert/body partial-volume edges.
        e = copy(m)
        for di in -1:1, dj in -1:1
            (di == 0 && dj == 0) && continue
            e .&= circshift(m, (di, dj))
        end
        e
    end

    function per_slice(mask)
        count(mask) == 0 && return nothing
        [
            (
                mean_hu = mean(@view hu_poly_fbp[:, :, k][mask]),
                std_hu = std(@view hu_poly_fbp[:, :, k][mask]),
                median_hu = median(@view hu_poly_fbp[:, :, k][mask]),
                n = count(mask),
            ) for k in axes(hu_poly_fbp, 3)
        ]
    end

    pure_water_mask = eroded_label_mask(UInt8(2))
    solid_water_mask = eroded_label_mask(UInt8(3))
    (
        pure_water = per_slice(pure_water_mask),
        solid_water = per_slice(solid_water_mask),
        μ_water_reference_cm_inv = bhc_calibration.μ_water,
    )
end

# ╔═╡ b4972cfc-176f-49ff-b98c-2908f28bd0f6
slice_metrics = let
    nz = size(hu_poly_fbp, 3)
    center_roi = 210:302
    sd = [std(@view hu_poly_fbp[center_roi, center_roi, k]) for k in 1:nz]
    (
        slice = collect(1:nz),
        central_roi_sd_hu = sd,
        minimum_hu = [minimum(@view hu_poly_fbp[:, :, k]) for k in 1:nz],
        maximum_hu = [maximum(@view hu_poly_fbp[:, :, k]) for k in 1:nz],
        nonfinite = [count(!isfinite, @view hu_poly_fbp[:, :, k]) for k in 1:nz],
        edge_to_middle_sd = max(sd[1], sd[end]) / sd[(nz + 1) ÷ 2],
    )
end

# ╔═╡ e73e72a0-7e95-4b92-b506-2adb7876f98e
md"""
## Slide through the reconstruction

The HU window stays fixed so a terminal-slice failure cannot hide behind
auto-ranging.
"""

# ╔═╡ d76914f5-7ff3-4f73-a391-c04f14a77588
@bind z_slice PlutoUI.Slider(
    1:size(hu_poly_fbp, 3);
    default = size(hu_poly_fbp, 3),
    show_value = true,
)

# ╔═╡ 1eb4f0c0-a4f2-43bd-8298-97d7149ca474
let
    nz = size(hu_poly_fbp, 3)
    dz_mm = pcct_forward.geom.fov[3] * 10 / nz
    z_mm = -pcct_forward.geom.fov[3] * 5 + (z_slice - 0.5) * dz_mm
    fig = CM.Figure(size = (760, 650))
    ax = CM.Axis(
        fig[1, 1];
        title = "PCCT poly FBP — slice $z_slice / $nz",
        subtitle = "z = $(@sprintf("%.2f", z_mm)) mm · fixed window −200 to 500 HU",
        aspect = CM.DataAspect(),
    )
    hm = CM.heatmap!(ax, hu_poly_fbp[:, :, z_slice]; colormap = :grays, colorrange = (-300, 500))
    CM.hidedecorations!(ax)
    CM.Colorbar(fig[1, 2], hm)
    fig
end

# ╔═╡ Cell order:
# ╟─5ef3b53c-43ef-4c10-a2a5-c30339e228dc
# ╠═d0f8865c-104f-4ac9-88c9-e02dba318f50
# ╠═b606834e-a070-4d27-a4ce-adb64be1c3fa
# ╠═4976d9d7-d69c-4875-b93f-156bea4390f7
# ╟─4cb3cc2f-88f5-43ad-8b46-50c8d43b091f
# ╠═b971e288-8085-4b5e-808a-a242d93243e1
# ╠═d1550c3a-602b-4768-b4a5-75be3d9cf904
# ╠═70490132-a100-4c84-9e13-aa81e4246ffc
# ╠═fc1db85b-3385-413f-b3e9-25979861bee9
# ╠═0514fdeb-0adc-4eca-863f-1fa2ee76f754
# ╠═41138011-ac7b-4030-bb7f-d9c0a540b069
# ╟─f384a1f7-1471-4542-9d2f-982913de015c
# ╠═ef3a1763-9c60-400a-9fb0-ddd1276461b1
# ╠═8129a8c9-00ce-4fb8-a7f9-199a3a7ac2fc
# ╟─c6368c68-40e2-4113-bfab-9cb2cd68c22d
# ╠═453b47bb-7907-4669-baf7-e280db0831ce
# ╠═8eac0ae0-5250-4e41-bbd2-5dd4e606f3d9
# ╠═29f9fa1e-4fac-4bd3-87b5-7607e3aa3f89
# ╠═b66b6120-d7d0-4210-b3a3-9595ba910b8e
# ╠═31f07a99-305d-453e-8b22-a848830fe43c
# ╠═b4972cfc-176f-49ff-b98c-2908f28bd0f6
# ╟─e73e72a0-7e95-4b92-b506-2adb7876f98e
# ╟─d76914f5-7ff3-4f73-a391-c04f14a77588
# ╟─1eb4f0c0-a4f2-43bd-8298-97d7149ca474
