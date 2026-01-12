# test_visualization.jl - Heatmap generation for visual inspection
# These outputs are tracked by git for qualitative human review

"""
    save_heatmap(filename, data; colormap=:gray, vmin=nothing, vmax=nothing)

Save 2D array as PPM heatmap. No external dependencies.
"""
function save_heatmap(filename, data; colormap=:gray, vmin=nothing, vmax=nothing)
    filename = endswith(filename, ".ppm") ? filename : filename * ".ppm"
    mkpath(dirname(filename))

    data_f = Float64.(data)
    v_min = isnothing(vmin) ? minimum(data_f) : Float64(vmin)
    v_max = isnothing(vmax) ? maximum(data_f) : Float64(vmax)
    normalized = v_max ≈ v_min ? zeros(size(data_f)) : clamp.((data_f .- v_min) ./ (v_max - v_min), 0.0, 1.0)

    cmap = colormap == :hot ? hot_color : (colormap == :viridis ? viridis_color : gray_color)
    ny, nx = size(normalized)

    open(filename, "w") do io
        println(io, "P6\n$nx $ny\n255")
        for y in 1:ny, x in 1:nx
            r, g, b = cmap(normalized[y, x])
            write(io, UInt8.(round.((r, g, b) .* 255))...)
        end
    end
    filename
end

gray_color(t) = (t, t, t)
hot_color(t) = (clamp(3t, 0, 1), clamp(3t - 1, 0, 1), clamp(3t - 2, 0, 1))
viridis_color(t) = (clamp(0.267 + t * (0.329 + t * (1.45 - t * 1.65)), 0, 1),
                    clamp(0.004 + t * (1.08 - t * 0.15), 0, 1),
                    clamp(0.329 + t * (0.42 - t * 0.85), 0, 1))

"""Save montage of slices from 3D volume."""
function save_slice_montage(filename, volume; slices=nothing, colormap=:gray, vmin=nothing, vmax=nothing)
    nz = size(volume, 3)
    indices = isnothing(slices) ? round.(Int, range(1, nz, length=min(9, nz))) : slices
    n = length(indices)
    cols, rows = ceil(Int, sqrt(n)), ceil(Int, n / ceil(Int, sqrt(n)))
    ny, nx = size(volume, 1), size(volume, 2)
    montage = zeros(eltype(volume), rows * ny, cols * nx)
    for (i, idx) in enumerate(indices)
        r, c = (i - 1) ÷ cols, (i - 1) % cols
        montage[r*ny+1:(r+1)*ny, c*nx+1:(c+1)*nx] = volume[:, :, idx]
    end
    save_heatmap(filename, montage; colormap=colormap, vmin=vmin, vmax=vmax)
end

"""Save 2D sinogram view (angle vs detector column)."""
function save_sinogram_view(filename, sinogram; row=1, colormap=:hot)
    save_heatmap(filename, sinogram[:, clamp(row, 1, size(sinogram, 2)), :]'; colormap=colormap)
end
