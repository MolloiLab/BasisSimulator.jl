# Visual parity gallery at a CT-realistic configuration: the compiled five-struct pipelines vs the
# legacy Metal pipeline on the Gammex 472 (GE Revolution arc, 984 views).
#   NV=256 VIEWS=984 RECON=256 julia --project=envs/reactant -t 2 --heap-size-hint=6G design/reactant/probes/gallery_parity.jl
# Produces gallery_eict.png (legacy scan, compiled twin, difference, per-rod HU) and gallery_vmi.png
# (compiled dual-kVp VMI stack with per-rod HU vs theory at each keV).
using Reactant, BasisSimulator, Statistics, Printf, CairoMakie
const BS = BasisSimulator; const BSF = BS.Functional
envi(k, d) = parse(Int, get(ENV, k, string(d)))
NV = envi("NV", 256); VIEWS = envi("VIEWS", 984); RECON = envi("RECON", 256); OUT = get(ENV, "OUT", @__DIR__)
T0 = time(); say(m) = (println(@sprintf("[%7.1f s] ", time() - T0), m); flush(stdout))
scanner = BS.EICTScanner(source_to_isocenter = 625.6, source_to_detector = 1100.0, detector_rows = 256, detector_cols = 834,
    detector_row_size = 0.625, detector_col_size = 0.6, detector_shape = :arc, focal_spot_width = 1.0, focal_spot_length = 1.0,
    target_angle = 10.0, flat_filter_material = :aluminum, flat_filter_thickness = 2.5, bowtie_filter = :ge_revolution_large,
    detector_material = :lumex, detector_depth = 3.0, fill_factor_row = 0.9, fill_factor_col = 0.9, electronic_noise = 0, detection_gain = 10.0)
proto(kvp, mA) = BS.CTProtocol(kVp = kvp, mA = mA, views = VIEWS, rotation_time = 1.0, collimation_mm = 5.0, additional_filters = [("Al", 4.5)])
opts = BS.SimOptions(use_noise = false, use_scatter = false, use_focal_spot = false, use_optical_crosstalk = false, use_lag = false, seed = 1234)
recon_opts = BS.ReconOptions(matrix_size = (RECON, RECON, 2), fov_cm = 35.0, z_cm = 0.5)
phantom = BS.compact_materials(BS.create_gammex_472(n_voxels = NV, n_slices = 2, fov_cm = 45.0, z_cm = 1.0))
say("phantom $(size(phantom.mask)), $(length(phantom.materials)) materials")
function legacy_scan(protocol)
    ws = BS.create_eict_workspace(scanner, protocol, opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, protocol, opts)
    model = BS.calibrate_bhc_water(opts, protocol; scanner, geom = ws.geom)
    sino_bhc = BS.apply_bhc_water(ws.sinogram, model)
    ws_fdk = BS.create_fdk_recon_workspace(sino_bhc, ws.geom, recon_opts.matrix_size)
    Float32.(BS.to_hounsfield(copy(BS.reconstruct!(ws_fdk, sino_bhc, ws.geom)); μ_water = model.μ_water_ref))
end
t = @elapsed hu_leg = legacy_scan(proto(120, 200.0)); say(@sprintf("legacy 120 kVp scan %.0f s", t))
pipe = BSF.pipeline(phantom, scanner, proto(120, 200.0), opts, recon_opts; batch_budget_mb = 2048)
fr = BSF.onehot_fractions(phantom.mask, pipe.n_mat); fr_r = Reactant.to_rarray(fr)
t = @elapsed fwd = @compile sync = true BSF.forward(fr_r, pipe); say(@sprintf("compiled EICT forward %.0f s (batching %s)", t, string(pipe.batching)))
hu_twin = Array(fwd(fr_r, pipe)); t = @elapsed fwd(fr_r, pipe); say(@sprintf("compiled forward run %.2f s", t))
k = 1; m = [hypot(i - (RECON + 1) / 2, j - (RECON + 1) / 2) < 0.47RECON for i in 1:RECON, j in 1:RECON]
d = hu_twin[:, :, k] .- hu_leg[:, :, k]
say(@sprintf("twin vs legacy: rms %.3f HU, max |Δ| %.2f HU inside the FOV", sqrt(mean(d[m] .^ 2)), maximum(abs.(d[m]))))
# per-rod HU (legacy vs twin) on the phantom labels resampled to the recon grid
function label_on_recon(mask)
    nr = RECON; out = zeros(UInt8, nr, nr)
    for j in 1:nr, i in 1:nr
        x = -17.5 + (i - 0.5) * 35 / nr; y = -17.5 + (j - 0.5) * 35 / nr
        ip = round(Int, (x + 22.5) / (45 / NV) + 0.5); jp = round(Int, (y + 22.5) / (45 / NV) + 0.5)
        (1 <= ip <= NV && 1 <= jp <= NV) && (out[i, j] = mask[ip, jp, 1])
    end
    out
end
lab = label_on_recon(phantom.mask)
rows = String[]
for (i, mat) in enumerate(phantom.materials)
    sel = lab .== UInt8(i - 1); count(sel) < 20 && continue
    push!(rows, @sprintf("%-36s legacy %8.1f  twin %8.1f  Δ %6.2f HU", first(mat.name, 36), mean(hu_leg[:, :, k][sel]), mean(hu_twin[:, :, k][sel]), mean(hu_twin[:, :, k][sel]) - mean(hu_leg[:, :, k][sel])))
end
say("per-material HU:\n" * join(rows, "\n"))
fig = Figure(fontsize = 18)
Label(fig[0, :], @sprintf("Gammex 472 — GE Revolution 120 kVp, %d views, phantom %d², recon %d²: legacy Metal vs compiled twin", VIEWS, NV, RECON), fontsize = 24, font = :bold)
for (c, (img, ttl, cm, cr)) in enumerate(((hu_leg[:, :, k], "legacy scan (HU)", :grays, (-200, 800)), (hu_twin[:, :, k], "compiled twin (HU)", :grays, (-200, 800)),
        (d, @sprintf("twin − legacy (rms %.2f HU)", sqrt(mean(d[m] .^ 2))), :RdBu, (-5, 5))))
    ax = Axis(fig[1, c]; title = ttl, width = 420, height = 420); hidedecorations!(ax); heatmap!(ax, img; colormap = cm, colorrange = cr)
end
Colorbar(fig[1, 4]; colormap = :grays, limits = (-200, 800), label = "HU", width = 12); Colorbar(fig[1, 5]; colormap = :RdBu, limits = (-5, 5), label = "ΔHU", width = 12)
resize_to_layout!(fig); save(joinpath(OUT, "gallery_eict.png"), fig); say("saved gallery_eict.png")
# VMI: compiled dual-kVp chain
vp = BSF.vmi_pipeline(phantom, scanner, [proto(80, 400.0), proto(140, 150.0)], opts, recon_opts; energies = [50.0, 70.0, 100.0, 140.0], batch_budget_mb = 2048)
vmi_fwd(f, p) = BSF.vmi_forward(f, p).vmis
t = @elapsed cv = @compile sync = true vmi_fwd(fr_r, vp); say(@sprintf("compiled VMI forward %.0f s", t))
vmis = Array(cv(fr_r, vp)); t = @elapsed cv(fr_r, vp); say(@sprintf("compiled VMI run %.2f s", t))
μw(e) = BS.compute_μ_at_energy(BS.XA.Materials.water, e)
theory(mat, e) = 1000 * (BS.compute_μ_at_energy(mat, e) / μw(e) - 1)
fig = Figure(fontsize = 18)
Label(fig[0, :], "Compiled dual-kVp n-channel VMI (80/140 kVp) — per-rod HU vs theory", fontsize = 24, font = :bold)
for (c, e) in enumerate(vp.energies)
    ax = Axis(fig[1, c]; title = @sprintf("VMI %.0f keV", e), width = 340, height = 340); hidedecorations!(ax); heatmap!(ax, vmis[:, :, k, c]; colormap = :grays, colorrange = (-200, 800))
end
Colorbar(fig[1, length(vp.energies) + 1]; colormap = :grays, limits = (-200, 800), label = "HU", width = 12)
ax = Axis(fig[2, 1:length(vp.energies)]; xlabel = "theory HU", ylabel = "measured HU", height = 300, title = "per-rod HU, all energies (dashed = identity)")
for (c, e) in enumerate(vp.energies)
    th = Float64[]; me = Float64[]
    for (i, mat) in enumerate(phantom.materials)
        sel = lab .== UInt8(i - 1); count(sel) < 20 && continue
        push!(th, theory(mat, e)); push!(me, mean(vmis[:, :, k, c][sel]))
    end
    scatter!(ax, th, me; label = @sprintf("%.0f keV", e), markersize = 10)
end
lines!(ax, [-100, 1500], [-100, 1500]; linestyle = :dash, color = :black); axislegend(ax; position = :lt)
resize_to_layout!(fig); save(joinpath(OUT, "gallery_vmi.png"), fig); say("saved gallery_vmi.png")
say("GALLERY_DONE")
