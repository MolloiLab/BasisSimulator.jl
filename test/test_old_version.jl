"""
Test using the EXACT working code from the old ct_simulator_final.jl notebook
This bypasses the new module structure entirely to verify basic functionality
"""

using FFTW
using Statistics
import XrayAttenuation as XA

# Copy the EXACT working FDK from old notebook
function old_reconstruct_fdk(projections, SAD_cm, SDD_cm, pixel_width_cm, pixel_height_cm,
                              angles_deg, n_rows, n_cols,
                              recon_x, recon_y, recon_z)
    vol = zeros(length(recon_x), length(recon_y), length(recon_z))
    filtered_proj = zeros(size(projections))

    n_pad = nextpow(2, n_cols * 2)
    ramp_scale = 1.0 / pixel_width_cm
    ramp = abs.(fftshift(fftfreq(n_pad))) .* ramp_scale

    # 1. Filter
    for k in 1:length(angles_deg)
        for r in 1:n_rows
            row_data = projections[r, :, k]

            v_pos = (r - n_rows/2 - 0.5) * pixel_height_cm
            u_coords = ((1:n_cols) .- n_cols/2 .- 0.5) .* pixel_width_cm
            weights = SDD_cm ./ sqrt.(SDD_cm^2 .+ u_coords.^2 .+ v_pos^2)

            padded = zeros(n_pad)
            padded[1:n_cols] = row_data .* weights
            filtered = real(ifft(fft(padded) .* fftshift(ramp)))
            filtered_proj[r, :, k] = filtered[1:n_cols]
        end
    end

    # 2. Backproject
    mid_row = n_rows/2 + 0.5
    mid_col = n_cols/2 + 0.5
    rads = deg2rad.(angles_deg)
    sins, coss = sin.(rads), cos.(rads)

    for k_slice in 1:length(recon_z)
        z_world = recon_z[k_slice]
        slice_buffer = zeros(length(recon_x), length(recon_y))

        for k_angle in 1:length(angles_deg)
            sin_a, cos_a = sins[k_angle], coss[k_angle]
            for j in 1:length(recon_y), i in 1:length(recon_x)
                px, py = recon_x[i], recon_y[j]

                x_r = px * cos_a + py * sin_a
                y_r = -px * sin_a + py * cos_a
                dist = SAD_cm + y_r

                if dist > 0.1
                    mag = SDD_cm / dist
                    col_idx = (x_r * mag) / pixel_width_cm + mid_col
                    row_idx = (z_world * mag) / pixel_height_cm + mid_row

                    c_fl = floor(Int, col_idx)
                    r_fl = floor(Int, row_idx)

                    if c_fl >= 1 && c_fl < n_cols && r_fl >= 1 && r_fl < n_rows
                        dc, dr = col_idx - c_fl, row_idx - r_fl
                        val = (1-dc)*(1-dr)*filtered_proj[r_fl, c_fl, k_angle] +
                              (dc)*(1-dr)*filtered_proj[r_fl, c_fl+1, k_angle] +
                              (1-dc)*(dr)*filtered_proj[r_fl+1, c_fl, k_angle] +
                              (dc)*(dr)*filtered_proj[r_fl+1, c_fl+1, k_angle]

                        slice_buffer[i, j] += val * (SAD_cm / dist)^2
                    end
                end
            end
        end
        vol[:, :, k_slice] = slice_buffer
    end
    return vol .* (2 * π / length(angles_deg))
end

println("="^70)
println("Testing with OLD WORKING CODE")
println("="^70)

# Create simple test: uniform cylinder with known μ
println("\n1. Creating analytical sinogram for uniform water cylinder...")
SAD_cm = 60.0
SDD_cm = 100.0
pixel_width_cm = 0.1
pixel_height_cm = 0.1
n_rows = 64
n_cols = 128
n_angles = 180
angles_deg = collect(range(0.0, 360.0, length=n_angles+1)[1:end-1])

# Create sinogram: line integrals through 20cm diameter water cylinder
μ_water = 0.2  # cm^-1
cylinder_radius_cm = 10.0
projections = zeros(n_rows, n_cols, n_angles)

for (a_idx, angle) in enumerate(angles_deg)
    for col in 1:n_cols
        u = (col - n_cols/2 - 0.5) * pixel_width_cm
        if abs(u) < cylinder_radius_cm
            chord_length = 2 * sqrt(cylinder_radius_cm^2 - u^2)
            projections[:, col, a_idx] .= μ_water * chord_length
        end
    end
end

println("  Sinogram range: $(minimum(projections)) to $(maximum(projections)) cm^-1")
println("  Expected max: $(μ_water * 2 * cylinder_radius_cm) cm^-1")

# Reconstruct
println("\n2. Reconstructing with OLD FDK implementation...")
recon_x = collect(range(-15.0, 15.0, length=128))
recon_y = collect(range(-15.0, 15.0, length=128))
recon_z = [0.0]

volume = old_reconstruct_fdk(projections, SAD_cm, SDD_cm, pixel_width_cm, pixel_height_cm,
                              angles_deg, n_rows, n_cols,
                              recon_x, recon_y, recon_z)

println("  Reconstruction range: $(minimum(volume)) to $(maximum(volume)) cm^-1")
println("  Center value: $(volume[64, 64, 1]) cm^-1")
println("  Target value: $μ_water cm^-1")
println("  Error: $(round(100*(volume[64, 64, 1] - μ_water)/μ_water, digits=1))%")

# Check if it's reasonable
if abs(volume[64, 64, 1] - μ_water) / μ_water < 0.2  # Within 20%
    println("\n✅ OLD CODE WORKS! Reconstruction is accurate.")
else
    println("\n❌ OLD CODE ALSO BROKEN! Error: $(round(100*(volume[64, 64, 1] - μ_water)/μ_water, digits=1))%")
end

println("="^70)
