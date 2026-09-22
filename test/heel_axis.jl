# The heel effect runs along the anode axis, which in a CT gantry is z: the detector rows.
using Test
using Statistics
import BasisSimulator as BS

@testset "heel effect varies along the rows, not the fan" begin
    heel = BS.default_heel_effect(anode_angle_deg = 7.0, effective_thickness_mm = 0.01)
    E = [40.0, 70.0, 100.0]
    # a wide fan (48 cm) and a wide cone (160 mm collimation): 256 rows × 0.625 mm
    wide = BS.EICTScanner(source_to_isocenter = 625.6, source_to_detector = 1100.0, detector_rows = 256, detector_cols = 400,
        detector_row_size = 0.625, detector_col_size = 1.2, detector_shape = :arc)
    g = BS.CTGeometry(wide; n_angles = 4, fov_cm = 40.0, collimation_mm = 160.0)
    tr = BS.compute_heel_spectral(heel, g, E)
    # flat along the fan
    @test maximum(abs, tr[:, 1, 2] .- tr[1, 1, 2]) < 1e-12
    @test maximum(abs, tr[:, end, 2] .- tr[1, end, 2]) < 1e-12
    # monotonic along the rows, falling towards the anode (+z) side
    profile = tr[1, :, 2]
    @test all(diff(profile) .<= 0)
    @test profile[end] < profile[1]
    # tens of percent across a 160 mm cone; the central row is the reference (1)
    @test isapprox(profile[128], 1.0; atol = 0.02) || isapprox(profile[129], 1.0; atol = 0.02)
    @test 0.02 < 1 - profile[end] / profile[1] < 0.6
    # a few percent across a 15 mm collimation
    narrow = BS.CTGeometry(wide; n_angles = 4, fov_cm = 40.0, collimation_mm = 15.0)
    trn = BS.compute_heel_spectral(heel, narrow, E)
    p = trn[1, :, 2]
    @test 1 - p[end] / p[1] < 0.05
    # harder spectrum on the anode side: the low-energy transmission falls more than the high
    @test (1 - tr[1, end, 1]) > (1 - tr[1, end, 3])
end
