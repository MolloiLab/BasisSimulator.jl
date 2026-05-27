### A Pluto.jl notebook ###
# v0.19.0

# ╔═╡ 06000001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
    # Add the PRISM deps that BasisSimulator's Project.toml doesn't already
    # pull in. Idempotent: `Pkg.add` is a no-op if a compatible version is
    # already installed.
    let
        needed = [
            "SparseArrays",       # stdlib but must be added explicitly to the env
            "LinearAlgebra",      # stdlib
            "SciMLOperators",
            "LinearSolve",
            "Krylov",
            "GeometryBasics",
        ]
        installed = keys(Pkg.project().dependencies)
        missing_pkgs = filter(n -> !(n in installed), needed)
        isempty(missing_pkgs) || Pkg.add(missing_pkgs)
    end
end

# ╔═╡ 06000001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 06000001-0000-4000-8000-000000000003
using Statistics: mean, std, var, quantile

# ╔═╡ 06000001-0000-4000-8000-000000000004
using LinearAlgebra: I, transpose, dot, norm

# ╔═╡ 06000001-0000-4000-8000-000000000005
using SparseArrays: sparse, spdiagm, blockdiag, SparseMatrixCSC

# ╔═╡ 06000001-0000-4000-8000-000000000006
using SciMLOperators: MatrixOperator, FunctionOperator, IdentityOperator, cache_operator

# ╔═╡ 06000001-0000-4000-8000-000000000007
using LinearSolve: LinearProblem, init, solve!, KrylovJL_CG

# ╔═╡ 06000001-0000-4000-8000-000000000008
using GeometryBasics: HyperRectangle, Rect

# ╔═╡ 06000001-0000-4000-8000-000000000010
md"""
# 03c · Dual-kVp Switching VMI · PRISM Inlined (Standalone)

GE Apex Elite GSI rapid-kVp-switching simulation (80 + 140 kVp, Gammex
472 phantom) where the **projection-domain material decomposition step
is a direct, inline 1-to-1 port of PrismMaterialDecomposition.jl** —
no `using PrismMaterialDecomposition`; every PRISM struct / matrix /
solver call is pasted into the cells below.

```
Simulate 80 kVp  →┐
                   ├─→  PRISM Regularized Linear PWLS (per detector row)
Simulate 140 kVp →┘                    │
                                       ▼
                    sino_iodine, sino_water  (basis line integrals)
                                       │
                    FBP × 2   (iodine, water basis maps)
                                       │
                    Z-Direction Median Filter
                                       │
                    Monoenergetic VMI Synthesis  (textbook 2-basis)
                                       │
                    Mono+ Post-Processing  (per-keV σ)
                                       │
                    Measured vs Theoretical Per-Rod Regression
                          at 40 / 70 / 100 / 140 keV
```

!!! info "What this notebook is for"
    A clean, standalone test of PRISM's regularized linear PWLS
    formulation `(AᵀV⁻¹A + λ∇R) x = AᵀV⁻¹L` directly on our dual-kVp
    sinograms — bypassing both the SF-JSD denoiser and the Cong
    decomposition that `03_dual_kvp_switching_vmi.jl` uses.

    The CPU code path of PRISM is pasted verbatim (function names
    preserved); GPU operators are omitted.  We solve through PRISM's
    real `LinearSolve.jl` / `KrylovJL_CG` stack — no hand-rolled CG.

!!! info "Sinogram-as-2D mapping"
    PRISM was written for 2D dual-layer detector images (`nrows, ncols`).
    We feed it our sinogram one detector row at a time — each slab
    `sino[:, r, :]` is a 2D image of shape `(n_col, n_view)`.  The
    5-point Laplacian then smooths across both adjacent detector
    columns *and* adjacent projection views.

!!! warning "Known approximation"
    PRISM's linear `A` uses scalar effective μ values per channel; the
    true forward model is polychromatic.  Residual beam-hardening bias
    in basis maps is *expected* — quantifying that gap vs the Cong +
    SF-JSD pipeline is exactly the point of this notebook.

!!! success "Source"
    Upstream:
    <https://github.com/fdekerme/PrismMaterialDecomposition.jl> ·
    function names mirrored 1-to-1.
"""

# ╔═╡ 06000001-0000-4000-8000-000000000020
md"""
## Setup
"""

# ╔═╡ 06000001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 06000001-0000-4000-8000-000000000031
import CairoMakie as CM

# ╔═╡ 06000001-0000-4000-8000-000000000040
begin
    GPU_BACKEND = let
        candidates = [
            (:Metal, "dde4c033-4e86-420c-a63e-0dd931031962", :MtlArray),
            (:CUDA, "052768ef-5323-5732-b1bb-66c8b64840ba", :CuArray),
            (:AMDGPU, "21141c5a-9bdb-4563-92ae-f87d6854732e", :ROCArray),
        ]
        detected = (name = "CPU", to_gpu = identity)
        for (pkg, uuid, ctor) in candidates
            pkg_id = Base.PkgId(Base.UUID(uuid), String(pkg))
            Base.locate_package(pkg_id) === nothing && continue
            try
                m = Base.require(pkg_id)
                if Base.invokelatest(getfield(m, :functional))
                    detected = (name = string(pkg), to_gpu = getfield(m, ctor))
                    break
                end
            catch
            end
        end
        detected
    end

    to_gpu(x) = GPU_BACKEND.to_gpu(x)
end

# ╔═╡ 06000001-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)

PRISM solve runs on **CPU** (sparse matrices + Krylov CG); only the FBP
recon uses the GPU backend, matching `03_dual_kvp_switching_vmi.jl`.
"""

# ╔═╡ 06000002-0000-4000-8000-000000000001
md"""
## 1. `Phantom`: Gammex Model 472
"""

# ╔═╡ 06000002-0000-4000-8000-000000000010
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,
    n_slices = 16,
    fov_cm = 35.0,
    z_cm = 1.0,
);

# ╔═╡ 06000002-0000-4000-8000-000000000020
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 06000003-0000-4000-8000-000000000001
md"""
## 2. `Scanner`: GE Revolution Apex Elite
"""

# ╔═╡ 06000003-0000-4000-8000-000000000010
scanner = BS.Scanner(
    source_to_isocenter = 625.6,
    source_to_detector = 1100.0,

    detector_rows = 256,
    detector_cols = 834,
    detector_row_size = 0.625,
    detector_col_size = 0.6,
    detector_shape = BS.CURVED_DETECTOR,

    focal_spot_width = 1.0,
    focal_spot_length = 1.0,
    target_angle = 10.0,

    flat_filter_material = :aluminum,
    flat_filter_thickness = 2.5,
    bowtie_filter = :ge_revolution_large,

    detector_material = :lumex,
    detector_depth = 3.0,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,

    electronic_noise = 0,
    detection_gain = 10.0,
);

# ╔═╡ 06000004-0000-4000-8000-000000000001
md"""
## 3. Dual-kVp Protocols (Rapid kVp Switching)

| kVp | Instantaneous mA | Duty cycle | Effective mA |
|-----|------------------|------------|--------------|
| 80  | 407              | 0.65       | 264.55       |
| 140 | 405              | 0.35       | 141.75       |
"""

# ╔═╡ 06000004-0000-4000-8000-000000000010
protocol_low = BS.CTProtocol(
    kVp = 80,
    mA = 407 * 0.65,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 06000004-0000-4000-8000-000000000020
protocol_high = BS.CTProtocol(
    kVp = 140,
    mA = 405 * 0.35,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 06000005-0000-4000-8000-000000000001
md"""
## 4. `SimOptions` and `ReconOptions`
"""

# ╔═╡ 06000005-0000-4000-8000-000000000010
sim_opts = BS.SimOptions(
    fidelity = :eict,
    seed = 1234,
);

# ╔═╡ 06000005-0000-4000-8000-000000000020
recon_opts = let
    slice_thickness_mm = 0.625
    n_recon_slices = round(Int, 5.0 / slice_thickness_mm)
    BS.ReconOptions(
        matrix_size = (512, 512, n_recon_slices),
        fov_cm = 35.0,
        z_cm = 0.5,
    )
end;

# ╔═╡ 06000006-0000-4000-8000-000000000001
md"""
## 5. Forward Project

Run `BS.simulate!` on each kVp protocol.  Same as
`03_dual_kvp_switching_vmi.jl`.
"""

# ╔═╡ 06000006-0000-4000-8000-000000000010
sim_low = let
    @info "Simulating: 80 kVp / $(round(protocol_low.mA, digits = 1)) mA-eff (DE low)…"
    ws = BS.create_eict_workspace(scanner, protocol_low, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, protocol_low, sim_opts)
    result = (sino = Array(ws.sinogram), geom = ws.geom)
    ws = nothing; GC.gc(true)
    result
end;

# ╔═╡ 06000006-0000-4000-8000-000000000020
sim_high = let
    @info "Simulating: 140 kVp / $(round(protocol_high.mA, digits = 1)) mA-eff (DE high)…"
    ws = BS.create_eict_workspace(scanner, protocol_high, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, protocol_high, sim_opts)
    result = (sino = Array(ws.sinogram), geom = ws.geom)
    ws = nothing; GC.gc(true)
    result
end;

# ╔═╡ 06000006-0000-4000-8000-000000000040
let
    n_row = size(sim_low.sino, 2)
    mid_r = n_row ÷ 2 + 1

    slice_lo = permutedims(sim_low.sino[:, mid_r, :], (2, 1))
    slice_hi = permutedims(sim_high.sino[:, mid_r, :], (2, 1))

    all_v = vcat(vec(slice_lo), vec(slice_hi))
    sino_window = (
        Float64(quantile(all_v, 0.01)),
        Float64(quantile(all_v, 0.99)),
    )

    fig = CM.Figure(size = (1180, 580))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "View", ylabel = "Detector Column",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )

    panels = (
        (1, 1, "80 kVp", slice_lo),
        (1, 2, "140 kVp", slice_hi),
    )

    for (r, c, ttl, slice) in panels
        ax = CM.Axis(fig[r, c]; title = ttl, axis_kwargs...)
        CM.heatmap!(ax, slice; colormap = :viridis, colorrange = sino_window)
    end
    CM.Colorbar(
        fig[1, 3]; colormap = :viridis, colorrange = sino_window,
        label = "Log Line Integral", width = 16, labelsize = 22, ticklabelsize = 18
    )
    fig
end

# ╔═╡ 06000007-0000-4000-8000-000000000001
md"""
## 6. PRISM — Inlined Source (CPU, Quadratic Regularizer)

Everything between here and §7 is a verbatim paste of the relevant
files from PrismMaterialDecomposition.jl, restricted to the CPU /
quadratic-Laplacian / `KrylovJL_CG` solve path.  Function and struct
names are preserved 1-to-1 with the upstream repo so the API in §7
calls look exactly like the upstream tests / examples.
"""

# ╔═╡ 06000007-0000-4000-8000-000000000010
md"""
### 6.1 `types.jl`
"""

# ╔═╡ 06000007-0000-4000-8000-000000000011
struct DLI{T<:AbstractArray}
    top::T
    bottom::T
end

# ╔═╡ 06000007-0000-4000-8000-000000000012
struct μ
    name::String
    low::Float64
    high::Float64
end

# ╔═╡ 06000007-0000-4000-8000-000000000013
struct MI{T<:AbstractArray}
    μ₁::μ
    μ₂::μ
    mat1::T
    mat2::T
end

# ╔═╡ 06000007-0000-4000-8000-000000000014
struct Regularization
    name::String
    constructor::Function
    params::Tuple
end

# ╔═╡ 06000007-0000-4000-8000-000000000020
md"""
### 6.2 `blend_functions.jl` (change-of-basis + per-pixel direct inverse)

PRISM's initial guess for the regularized solve is the algebraic
per-pixel pseudo-inverse — same as Cong with effective μ values.
"""

# ╔═╡ 06000007-0000-4000-8000-000000000021
function change_of_basis(v, μ₁::μ, μ₂::μ)
    M = [μ₁.low μ₂.low; μ₁.high μ₂.high]
    return M \ v
end

# ╔═╡ 06000007-0000-4000-8000-000000000022
function material_decomposition(dli_images::DLI, μ₁::μ, μ₂::μ)
    top_flat    = reshape(dli_images.top,    1, :)
    bottom_flat = reshape(dli_images.bottom, 1, :)
    decomposed  = change_of_basis(vcat(top_flat, bottom_flat), μ₁, μ₂)
    decomposed_mat1 = reshape(decomposed[1, :], size(dli_images.top))
    decomposed_mat2 = reshape(decomposed[2, :], size(dli_images.bottom))
    return MI(μ₁, μ₂, decomposed_mat1, decomposed_mat2)
end

# ╔═╡ 06000007-0000-4000-8000-000000000030
md"""
### 6.3 `A_contructor.jl` (CPU sparse-matrix form only)

Sparse 2N × 2N block-mixing matrix
`A = [μ₁_L·I  μ₂_L·I ; μ₁_H·I  μ₂_H·I]` and its transpose.
"""

# ╔═╡ 06000007-0000-4000-8000-000000000031
function A_mat_cpu(N::Integer, M::Matrix{Float64})::MatrixOperator
    @assert N > 0 "N must be positive"
    @assert size(M) == (2, 2) "Mixing matrix M must be 2×2"
    Id_N = sparse(I, N, N)
    A11 = M[1, 1] * Id_N
    A12 = M[1, 2] * Id_N
    A21 = M[2, 1] * Id_N
    A22 = M[2, 2] * Id_N
    A = [A11 A12; A21 A22]
    return MatrixOperator(A)
end

# ╔═╡ 06000007-0000-4000-8000-000000000032
function At_mat_cpu(N::Integer, M::Matrix{Float64})::MatrixOperator
    @assert N > 0 "N must be positive"
    @assert size(M) == (2, 2) "Mixing matrix M must be 2×2"
    Mᵀ = transpose(M)
    Id_N = sparse(I, N, N)
    A11 = Mᵀ[1, 1] * Id_N
    A12 = Mᵀ[1, 2] * Id_N
    A21 = Mᵀ[2, 1] * Id_N
    A22 = Mᵀ[2, 2] * Id_N
    Aᵀ = [A11 A12; A21 A22]
    return MatrixOperator(Aᵀ)
end

# ╔═╡ 06000007-0000-4000-8000-000000000040
md"""
### 6.4 `V_constructor.jl` (CPU diagonal V⁻¹ + Rect-based extract_pixels)

PRISM samples a `Rect` background patch to estimate per-channel
variance, normalises the two scalars to sum to 1, then assembles
`V⁻¹ = diag([1/var_top_n · 1ₙ ; 1/var_bot_n · 1ₙ])`.
"""

# ╔═╡ 06000007-0000-4000-8000-000000000041
function extract_pixels(image::AbstractMatrix, rect::Rect)
    xmin = rect.origin[1]
    ymin = rect.origin[2]
    xmax = xmin + rect.widths[1]
    ymax = ymin + rect.widths[2]
    if xmin < 1 || ymin < 1 || xmax > size(image, 1) || ymax > size(image, 2)
        error("Rectangle is out of bounds")
    end
    return image[xmin:xmax, ymin:ymax]
end

# ╔═╡ 06000007-0000-4000-8000-000000000042
function _noise_variance_diag_inv(dli_images::DLI, background_mask)
    noise_bottom = extract_pixels(dli_images.bottom, background_mask)
    noise_top    = extract_pixels(dli_images.top,    background_mask)

    var_noise_bottom = var(noise_bottom)
    var_noise_top    = var(noise_top)

    var_noise_bottom_norm = var_noise_bottom / (var_noise_bottom + var_noise_top)
    var_noise_top_norm    = var_noise_top    / (var_noise_bottom + var_noise_top)

    var_noise_bottom_vec = fill(1 / var_noise_bottom_norm, length(dli_images.bottom))
    var_noise_top_vec    = fill(1 / var_noise_top_norm,    length(dli_images.top))

    return vcat(var_noise_top_vec, var_noise_bottom_vec)
end

# ╔═╡ 06000007-0000-4000-8000-000000000043
function Vinv_mat_cpu(dli_images::DLI, background_mask)::MatrixOperator
    V_inv_diag = _noise_variance_diag_inv(dli_images, background_mask)
    V⁻¹ = spdiagm(V_inv_diag)
    return MatrixOperator(V⁻¹)
end

# ╔═╡ 06000007-0000-4000-8000-000000000050
md"""
### 6.5 `regularization_matrix.jl` (quadratic 5-point Laplacian)
"""

# ╔═╡ 06000007-0000-4000-8000-000000000051
function generate_quadratic_regularization_matrix(img)
    nrows, ncols = size(img)
    N = nrows * ncols
    return spdiagm(
        N, N,
        -nrows => fill(-1.0, N - nrows),
         nrows => fill(-1.0, N - nrows),
             0 => fill( 4.0, N),
            -1 => fill(-1.0, N - 1),
             1 => fill(-1.0, N - 1),
    )
end

# ╔═╡ 06000007-0000-4000-8000-000000000060
md"""
### 6.6 `regularization_constructors.jl` (matrix-form ∇R for quadratic)

PRISM expects `∇R` to act on the stacked 2N vector `[m₁; m₂]`.  The 5-
point Laplacian `L_N` is N × N, so we hand it the block-diagonal
`blockdiag(L_N, L_N)` — independent smoothing per material (no cross-
material coupling — matches `feedback_pwls_no_cross_basis_prior`).

The constructor signature `(params, prototype)` is the one PRISM's
problem builder expects so we can plug straight into
`Regularization(...)`.
"""

# ╔═╡ 06000007-0000-4000-8000-000000000061
function ∇R_quad_mat(params, prototype::AbstractArray)::MatrixOperator
    # NB: matches the unpacking pattern of `∇R_similarity_mat` in PRISM's
    # `regularization_constructors.jl`, where `reg.params` is a 1-tuple
    # `(matrix,)`.  The upstream `∇R_quad_mat` destructures with
    # `_, _, _, _, matrix = params` which leaves `matrix` as the 1-tuple
    # itself and then errors inside `MatrixOperator(::Tuple)`.
    _, _, _, _, (quadratic_regularization_matrix,) = params
    return MatrixOperator(quadratic_regularization_matrix)
end

# ╔═╡ 06000007-0000-4000-8000-000000000070
md"""
### 6.7 `utils.jl` (LinearSolve callback)
"""

# ╔═╡ 06000007-0000-4000-8000-000000000071
function callback(state, cost)
    println("Current cost: ", cost)
    return false
end

# ╔═╡ 06000007-0000-4000-8000-000000000080
md"""
### 6.8 `linear_solve.jl` (CPU problem container + solver)
"""

# ╔═╡ 06000007-0000-4000-8000-000000000081
mutable struct RegularizedDecompositionProblem{Tλ,Tmat,Tvec}
    dli_images::DLI
    μ₁::μ
    μ₂::μ
    λ::Tλ
    background_mask::HyperRectangle{2,Int}

    nrows::Int
    ncols::Int
    N::Int

    M::Tmat
    L::Tvec
    initial_guess::Tvec
    prototype::Tvec
    regularization_name::String

    A::Union{FunctionOperator,MatrixOperator}
    Aᵀ::Union{FunctionOperator,MatrixOperator}
    V⁻¹::Union{FunctionOperator,MatrixOperator}
    ∇R::Union{FunctionOperator,MatrixOperator}
end

# ╔═╡ 06000007-0000-4000-8000-000000000082
function RegularizedDecompositionProblemCPU(
    dli_images::DLI,
    μ₁::μ,
    μ₂::μ,
    λ::Float64,
    background_mask::HyperRectangle{2,Int},
    reg::Regularization,
)
    nrows, ncols = size(dli_images.top)
    N = nrows * ncols

    M = [μ₁.low μ₂.low; μ₁.high μ₂.high]
    L = vcat(vec(dli_images.top), vec(dli_images.bottom))

    initial_guess_result = material_decomposition(dli_images, μ₁, μ₂)
    initial_guess_vec = vcat(
        vec(initial_guess_result.mat1),
        vec(initial_guess_result.mat2),
    )

    prototype = zeros(Float64, 2N)
    A   = A_mat_cpu(N, M)
    Aᵀ  = At_mat_cpu(N, M)
    V⁻¹ = Vinv_mat_cpu(dli_images, background_mask)
    ∇R  = reg.constructor(
        (N, nrows, ncols, LinearIndices((nrows, ncols)), reg.params),
        prototype,
    )

    return RegularizedDecompositionProblem{Float64,Matrix{Float64},Vector{Float64}}(
        dli_images,
        μ₁, μ₂,
        λ,
        background_mask,
        nrows, ncols, N,
        M, L,
        initial_guess_vec, prototype,
        reg.name,
        A, Aᵀ, V⁻¹, ∇R,
    )
end

# ╔═╡ 06000007-0000-4000-8000-000000000083
function RegularizedDecomposition(
    prob::RegularizedDecompositionProblem;
    solver = KrylovJL_CG(),
    reltol = 1e-6,
    verbose::Bool = false,
)
    A   = prob.A
    Aᵀ  = prob.Aᵀ
    V⁻¹ = prob.V⁻¹
    ∇R  = prob.∇R
    λ   = prob.λ

    system_matrix = cache_operator(Aᵀ * V⁻¹ * A + λ * ∇R, prob.L)
    L2 = Aᵀ * V⁻¹ * prob.L

    linear_problem = LinearProblem{true}(system_matrix, L2; u0 = prob.initial_guess)
    linear_solve = init(
        linear_problem, solver;
        reltol = reltol, verbose = verbose,
    )
    solve!(linear_solve)

    u = linear_solve.u
    mat1 = reshape(u[1:prob.N],         (prob.nrows, prob.ncols)) |> Matrix{Float64}
    mat2 = reshape(u[prob.N+1:end],     (prob.nrows, prob.ncols)) |> Matrix{Float64}

    return MI(prob.μ₁, prob.μ₂, mat1, mat2)
end

# ╔═╡ 06000008-0000-4000-8000-000000000001
md"""
## 7. Hybrid: Cong Polychromatic Decomp + PRISM Regularization

### 7.1 Why this isn't pure-PRISM anymore

The previous version of §7 ran PRISM's **linear** PWLS directly on the
dual-kVp log-line-integral sinograms.  That collapsed the iodine K-edge
into a single scalar `μ_iod_L`, which forced the inversion to
over-attribute iodine in rod regions and under-attribute water →
iodine rods went deeply negative at high keV.  That bias is intrinsic
to PRISM's linear forward model and can't be fixed inside the
linear-A framework.

The pragmatic fix that keeps PRISM's §6 machinery intact:

```
Step 1.  Cong per-ray polychromatic decomposition
         → noisy but unbiased basis line integrals
                            │
                            ▼
Step 2.  PRISM regularized solve with M = I
         → quadratic-Laplacian-denoised basis line integrals
```

Step 2 uses the **same** `RegularizedDecompositionProblemCPU` /
`RegularizedDecomposition` we inlined in §6 — we just feed it Cong's
output as the "measurement" and set the mixing matrix to identity,
which collapses the data term `‖A·x − L‖²_V⁻¹` to a Tikhonov-style
proximal `‖x − x_Cong‖²_V⁻¹`.

What you gain over linear-PRISM:
- **Polychromatic accuracy** — Cong handles the K-edge and the bowtie
  per ray, so iodine basis line integrals come out unbiased.
- **PRISM-style spatial smoothing** — the quadratic Laplacian still
  couples adjacent `(col, view)` pixels per detector row.
- **Per-channel V⁻¹ weighting** — PRISM still re-weights iodine vs
  water by their respective basis-sinogram noise levels.

### 7.2 Material basis (per-energy spectrum, Cong-style)
"""

# ╔═╡ 06000008-0000-4000-8000-000000000010
material_basis = let
    e_L, ŵ_L = BS.resolve_source_spectrum_with_bowtie(
        sim_opts, protocol_low; scanner = scanner, geom = sim_low.geom,
    )
    e_H, ŵ_H = BS.resolve_source_spectrum_with_bowtie(
        sim_opts, protocol_high; scanner = scanner, geom = sim_high.geom,
    )

    # Per-pixel normalised spectrum (3D, Cong-format).
    ŵ_L_f32 = Float32.(ŵ_L ./ sum(ŵ_L; dims = ndims(ŵ_L)))
    ŵ_H_f32 = Float32.(ŵ_H ./ sum(ŵ_H; dims = ndims(ŵ_H)))

    iodine_mat = BS.XA.Elements.Iodine
    water_mat  = BS.XA.Materials.water

    p_L = Float32[Float32(BS.compute_mass_μ_at_energy(iodine_mat, Float64(E))) for E in e_L]
    q_L = Float32[Float32(BS.compute_mass_μ_at_energy(water_mat,  Float64(E))) for E in e_L]
    p_H = Float32[Float32(BS.compute_mass_μ_at_energy(iodine_mat, Float64(E))) for E in e_H]
    q_H = Float32[Float32(BS.compute_mass_μ_at_energy(water_mat,  Float64(E))) for E in e_H]

    (
        ŵ_L = ŵ_L_f32, p_L = p_L, q_L = q_L,
        ŵ_H = ŵ_H_f32, p_H = p_H, q_H = q_H,
    )
end;

# ╔═╡ 06000008-0000-4000-8000-000000000020
md"""
### 7.3 Background `Rect` + λ choice

PRISM samples a rectangular background region of the (basis) sinogram
to estimate per-channel noise variance.  We take the first 20 detector
columns × all views — well off the phantom, so the basis values there
are near zero and their variance captures Cong's output noise.

`λ` is the single regularizer knob.  Now operating on **basis line
integrals** (not log-line-integrals), so the natural scale is smaller —
default `1e-3`.  Increase for more spatial smoothing, decrease to stay
closer to Cong's per-ray output.
"""

# ╔═╡ 06000008-0000-4000-8000-000000000021
PRISM_λ = 1.0e-3;

# ╔═╡ 06000008-0000-4000-8000-000000000022
PRISM_BG_RECT = let
    n_col, n_row, n_view = size(sim_low.sino)
    # Rect((xmin, ymin), (width, height)) on the (n_col × n_view) slab —
    # x runs along detector columns, y runs along views.
    Rect(1, 1, 19, n_view - 1)
end;

# ╔═╡ 06000008-0000-4000-8000-000000000030
md"""
### 7.4 Cong polychromatic decomposition + PRISM denoising

**Step 1**: Run Cong on the full 3D sinograms.  This is BS's
`apply_cong!` — per-ray Brent + Newton on the polychromatic
transmission integral.  Output: `cong_iodine`, `cong_water` basis line
integrals (noisy but unbiased).

**Step 2**: For each detector row, fold Cong's `(cong_iodine,
cong_water)` slab into a PRISM `DLI` and call
`RegularizedDecompositionProblemCPU` with **identity mixing matrix**
`M = [1 0; 0 1]`.  PRISM's data term collapses to `‖x − x_Cong‖²_V⁻¹`,
and CG converges in tens of iterations to the Tikhonov-denoised
basis maps.

!!! tip "Subsetting for fast iteration"
    Narrow `PRISM_ROW_RANGE` to a sub-range to iterate λ quickly;
    uncomputed rows stay zero.
"""

# ╔═╡ 06000008-0000-4000-8000-000000000031
PRISM_ROW_RANGE = let
    n_row = size(sim_low.sino, 2)
    # The simulator already sizes the detector to match the recon z-slab
    # (recon_opts.z_cm = 0.5 cm here), so n_row is small (≈8 rows).  Just
    # solve every detector row — narrow band logic isn't useful at this
    # z-extent.  Override to a sub-range only if you want a smoke test.
    1:n_row
end

# ╔═╡ 06000008-0000-4000-8000-000000000040
sino_basis = let
    sino_low_cpu  = sim_low.sino
    sino_high_cpu = sim_high.sino

    # ─── Step 1: Cong polychromatic per-ray decomposition ────────────────
    sino_low_gpu  = to_gpu(Float32.(sino_low_cpu))
    sino_high_gpu = to_gpu(Float32.(sino_high_cpu))
    sino_y = similar(sino_low_gpu);  fill!(sino_y, 0.0f0)
    sino_c = similar(sino_low_gpu);  fill!(sino_c, 0.0f0)

    @info "Cong polychromatic decomposition: $(size(sino_low_cpu))"
    cong_elapsed = @elapsed begin
        cong_ws = BS.create_cong_workspace(sino_low_gpu, material_basis)
        BS.apply_cong!(
            cong_ws, sino_y, sino_c, sino_low_gpu, sino_high_gpu;
            water_basis = (a = 0.0f0, c = 1.0f0),
        )
    end
    @info "Cong done in $(round(cong_elapsed, digits = 1)) s"

    cong_iodine = Array(sino_y)   # (n_col, n_row, n_view), g/cm²
    cong_water  = Array(sino_c)
    sino_low_gpu = nothing; sino_high_gpu = nothing
    sino_y = nothing; sino_c = nothing; cong_ws = nothing
    GC.gc(true)

    # ─── Step 2: PRISM regularized denoise (per detector row) ────────────
    n_col, n_row, n_view = size(cong_iodine)
    sino_iodine = zeros(Float32, n_col, n_row, n_view)
    sino_water  = zeros(Float32, n_col, n_row, n_view)

    # Identity mixing — PRISM's `A` becomes the 2N × 2N identity, so the
    # data term `‖A·x − L‖²_V⁻¹` collapses to `‖x − L‖²_V⁻¹`.
    μ_id_1 = μ("BasisOne", 1.0, 0.0)   # M[:,1] = [1, 0]
    μ_id_2 = μ("BasisTwo", 0.0, 1.0)   # M[:,2] = [0, 1]
    λ      = PRISM_λ

    # Build per-slab Laplacian once (slab shape is identical for every row)
    proto_top = zeros(Float64, n_col, n_view)
    L_N       = generate_quadratic_regularization_matrix(proto_top)
    ∇R_blk    = blockdiag(L_N, L_N)
    reg_quad  = Regularization("quadratic", ∇R_quad_mat, (∇R_blk,))

    n_rows_to_do = length(PRISM_ROW_RANGE)
    @info "PRISM denoising: $(n_rows_to_do) detector row(s)"
    den_elapsed = @elapsed begin
        for (k, r) in enumerate(PRISM_ROW_RANGE)
            slab_iod = Float64.(@view cong_iodine[:, r, :])
            slab_wat = Float64.(@view cong_water[:,  r, :])
            # DLI.top = cong_iodine slab, DLI.bottom = cong_water slab
            dli = DLI(slab_iod, slab_wat)

            prob = RegularizedDecompositionProblemCPU(
                dli, μ_id_1, μ_id_2, λ, PRISM_BG_RECT, reg_quad,
            )
            mi = RegularizedDecomposition(prob; reltol = 1e-6, verbose = false)

            sino_iodine[:, r, :] .= Float32.(mi.mat1)
            sino_water[:,  r, :] .= Float32.(mi.mat2)

            (k == 1 || k == n_rows_to_do) && @info "  row $(r)  done"
        end
    end
    @info "Denoising done in $(round(den_elapsed, digits = 1)) s"

    (sino_iodine = sino_iodine, sino_water = sino_water, geom = sim_low.geom)
end;

# ╔═╡ 06000008-0000-4000-8000-000000000050
let
    n_row = size(sino_basis.sino_iodine, 2)
    mid_r = n_row ÷ 2 + 1

    fig = CM.Figure(size = (1400, 580))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "View", ylabel = "Detector Column",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )

    _qrange(arr) = (
        Float64(quantile(vec(arr), 0.01)),
        Float64(quantile(vec(arr), 0.99)),
    )

    slice_iod = permutedims(sino_basis.sino_iodine[:, mid_r, :], (2, 1))
    slice_wat = permutedims(sino_basis.sino_water[:,  mid_r, :], (2, 1))

    panels = (
        (1, 1, 2, "PRISM Iodine Basis Sinogram", "g/cm²",
            slice_iod, _qrange(slice_iod)),
        (1, 3, 4, "PRISM Water Basis Sinogram",  "g/cm²",
            slice_wat, _qrange(slice_wat)),
    )

    for (r, panel_c, cbar_c, ttl, cbar_label, slice, range) in panels
        ax = CM.Axis(fig[r, panel_c]; title = ttl, axis_kwargs...)
        CM.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        CM.Colorbar(
            fig[r, cbar_c]; colormap = :viridis, colorrange = range,
            label = cbar_label, width = 16, labelsize = 22, ticklabelsize = 18
        )
    end
    fig
end

# ╔═╡ 06000009-0000-4000-8000-000000000001
md"""
## 8. FBP: Iodine and Water Basis Maps
"""

# ╔═╡ 06000009-0000-4000-8000-000000000010
basis_volumes = let
    matrix_size = recon_opts.matrix_size
    geom = sino_basis.geom

    function _fbp(sino_cpu)
        sino_gpu = to_gpu(Float32.(sino_cpu))
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, geom, matrix_size; filter = BS.SoftFilter(),
        )
        recon = Array(BS.reconstruct!(ws, sino_gpu, geom))
        ws = nothing; sino_gpu = nothing
        GC.gc(true)
        return Float32.(recon)
    end

    (
        vol_iodine_raw = _fbp(sino_basis.sino_iodine),
        vol_water_raw  = _fbp(sino_basis.sino_water),
        geom = geom,
    )
end;

# ╔═╡ 06000009-0000-4000-8000-000000000030
let
    fig = CM.Figure(size = (1180, 580))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    mid = size(basis_volumes.vol_iodine_raw, 3) ÷ 2

    _qrange(arr) = (
        Float64(quantile(vec(arr), 0.01)),
        Float64(quantile(vec(arr), 0.99)),
    )

    slice_iod = basis_volumes.vol_iodine_raw[:, :, mid]
    slice_wat = basis_volumes.vol_water_raw[:,  :, mid]

    panels = (
        (1, 1, 2, "Iodine Basis", "g/cm³", slice_iod, _qrange(slice_iod)),
        (1, 3, 4, "Water Basis",  "g/cm³", slice_wat, _qrange(slice_wat)),
    )

    for (r, panel_c, cbar_c, ttl, cbar_label, slice, range) in panels
        ax = CM.Axis(
            fig[r, panel_c]; title = ttl,
            aspect = CM.DataAspect(), axis_kwargs...
        )
        CM.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        CM.hidedecorations!(ax)
        CM.Colorbar(
            fig[r, cbar_c]; colormap = :viridis, colorrange = range,
            label = cbar_label, width = 16, labelsize = 22, ticklabelsize = 18
        )
    end
    fig
end

# ╔═╡ 0600000a-0000-4000-8000-000000000001
md"""
## 9. Z-Direction Median Filter
"""

# ╔═╡ 0600000a-0000-4000-8000-000000000005
Z_MEDIAN_ADJACENT = 2;

# ╔═╡ 0600000a-0000-4000-8000-000000000010
basis_z = let
    (
        vol_iodine = BS.apply_median_z(
            basis_volumes.vol_iodine_raw;
            adjacent_slices = Z_MEDIAN_ADJACENT,
        ),
        vol_water = BS.apply_median_z(
            basis_volumes.vol_water_raw;
            adjacent_slices = Z_MEDIAN_ADJACENT,
        ),
        geom = basis_volumes.geom,
    )
end;

# ╔═╡ 0600000b-0000-4000-8000-000000000001
md"""
## 10. VMI Synthesis

Textbook 2-basis linear mix; VMI grid **40, 70, 100, 140 keV** as
requested.
"""

# ╔═╡ 0600000b-0000-4000-8000-000000000015
de_vmi_energies = [40.0, 70.0, 100.0, 140.0];

# ╔═╡ 0600000b-0000-4000-8000-000000000020
vmi_HU_by_keV = let
    c_iodine_mg_per_mL = basis_z.vol_iodine .* 1000.0f0
    out = Dict{Float64, Array{Float32, 3}}()
    for E in de_vmi_energies
        out[E] = BS.synth_vmi_2basis(
            basis_z.vol_water, c_iodine_mg_per_mL;
            energy_keV = E,
        )
    end
    out
end;

# ╔═╡ 0600000b-0000-4000-8000-000000000040
let
    HU_window = (-200, 500)

    fig = CM.Figure(size = (1180, 1180))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    sample = vmi_HU_by_keV[40.0]
    mid = size(sample, 3) ÷ 2

    for (k, E) in enumerate(de_vmi_energies)
        r = ((k - 1) ÷ 2) + 1
        c = ((k - 1) % 2) + 1
        ax = CM.Axis(
            fig[r, c]; title = "$(Int(E)) keV VMI",
            aspect = CM.DataAspect(), axis_kwargs...,
        )
        CM.heatmap!(
            ax, vmi_HU_by_keV[E][:, :, mid];
            colormap = :grays, colorrange = HU_window,
        )
        CM.hidedecorations!(ax)
    end
    CM.Colorbar(
        fig[1:2, 3];
        colormap = :grays, colorrange = HU_window,
        label = "HU", width = 16, labelsize = 22, ticklabelsize = 18,
    )

    fig
end

# ╔═╡ 0600000c-0000-4000-8000-000000000001
md"""
## 11. VMI Post-Processing (Mono+)
"""

# ╔═╡ 0600000c-0000-4000-8000-000000000005
σ_vmi_lp_px = Float64[2.0, 0.0, 1.0, 1.0];   # 40, 70, 100, 140 keV

# ╔═╡ 0600000c-0000-4000-8000-000000000010
vmi_HU_final = let
    volumes = [vmi_HU_by_keV[E] for E in de_vmi_energies]
    ws = BS.create_mono_plus_workspace(
        volumes[1]; n_energies = length(de_vmi_energies),
    )
    BS.apply_mono_plus!(
        ws, volumes, de_vmi_energies;
        E_noise_opt = 70.0,
        σ_lp_px = σ_vmi_lp_px,
        verbose = true,
    )
    out = Dict{Float64, Array{Float32, 3}}()
    for (i, E) in enumerate(de_vmi_energies)
        out[E] = copy(ws.out_vols[i])
    end
    ws = nothing; GC.gc(true)
    out
end;

# ╔═╡ 0600000c-0000-4000-8000-000000000030
let
    HU_window = (-200, 500)

    fig = CM.Figure(size = (1180, 1180))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    sample = vmi_HU_final[40.0]
    mid = size(sample, 3) ÷ 2

    for (k, E) in enumerate(de_vmi_energies)
        r = ((k - 1) ÷ 2) + 1
        c = ((k - 1) % 2) + 1
        ax = CM.Axis(
            fig[r, c]; title = "$(Int(E)) keV VMI",
            subtitle = "Mono+",
            aspect = CM.DataAspect(), axis_kwargs...,
        )
        CM.heatmap!(
            ax, vmi_HU_final[E][:, :, mid];
            colormap = :grays, colorrange = HU_window,
        )
        CM.hidedecorations!(ax)
    end
    CM.Colorbar(
        fig[1:2, 3];
        colormap = :grays, colorrange = HU_window,
        label = "HU", width = 16, labelsize = 22, ticklabelsize = 18,
    )

    fig
end

# ╔═╡ 0600000d-0000-4000-8000-000000000001
md"""
## Results

Per-rod measured vs theoretical HU at 40 / 70 / 100 / 140 keV.
"""

# ╔═╡ 0600000d-0000-4000-8000-000000000010
ROD_LABELS = (
    Ca = (UInt8(10), UInt8(11), UInt8(12), UInt8(13), UInt8(14), UInt8(15), UInt8(16)),
    I  = (UInt8(20), UInt8(21), UInt8(22), UInt8(23), UInt8(24), UInt8(25), UInt8(26)),
);

# ╔═╡ 0600000d-0000-4000-8000-000000000020
ROD_NAMES = (
    Ca = ("50 mg/mL", "100 mg/mL", "200 mg/mL", "300 mg/mL", "400 mg/mL", "500 mg/mL", "600 mg/mL"),
    I  = ("2.0 mg/mL", "2.5 mg/mL", "5.0 mg/mL", "7.5 mg/mL", "10.0 mg/mL", "15.0 mg/mL", "20.0 mg/mL"),
);

# ╔═╡ 0600000d-0000-4000-8000-000000000030
rod_data = let
    materials = phantom_cpu.materials
    mask_2d = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
    nx, ny = size(mask_2d)
    ROI_RADIUS_PX = 8

    function rod_centroid(label::UInt8)
        idx = findall(==(label), mask_2d)
        isempty(idx) && error("rod_centroid: no voxels with label $label")
        cx = sum(ci -> Float64(ci[1]), idx) / length(idx)
        cy = sum(ci -> Float64(ci[2]), idx) / length(idx)
        return (cx, cy)
    end

    function rod_roi_mask(label::UInt8)
        cx, cy = rod_centroid(label)
        i_lo = max(1, floor(Int, cx - ROI_RADIUS_PX))
        i_hi = min(nx, ceil(Int, cx + ROI_RADIUS_PX))
        j_lo = max(1, floor(Int, cy - ROI_RADIUS_PX))
        j_hi = min(ny, ceil(Int, cy + ROI_RADIUS_PX))
        roi = CartesianIndex{2}[]
        r² = Float64(ROI_RADIUS_PX)^2
        for j in j_lo:j_hi, i in i_lo:i_hi
            ((i - cx)^2 + (j - cy)^2) ≤ r² && push!(roi, CartesianIndex(i, j))
        end
        return roi
    end

    rod_rois = Dict(
        lab => rod_roi_mask(lab)
            for lab in vcat(collect(ROD_LABELS.Ca), collect(ROD_LABELS.I))
    )

    μ_water_E = Dict(
        E => BS.compute_μ_at_energy(BS.XA.Materials.water, E)
            for E in de_vmi_energies
    )

    function theoretical_hu(material, E::Float64)
        mu = BS.compute_μ_at_energy(material, E)
        return 1000.0 * (mu - μ_water_E[E]) / μ_water_E[E]
    end

    function measured_hu(vmi_vol, label::UInt8)
        roi = rod_rois[label]
        s = 0.0; n = 0
        for z in 1:size(vmi_vol, 3), ci in roi
            s += vmi_vol[ci, z]; n += 1
        end
        return s / n
    end

    out = Dict{Symbol, NamedTuple}()
    for group in (:Ca, :I)
        labels = ROD_LABELS[group]
        n_rods = length(labels)
        n_E    = length(de_vmi_energies)
        meas   = zeros(Float64, n_rods, n_E)
        theo   = zeros(Float64, n_rods, n_E)
        for (i, lab) in pairs(labels)
            mat = materials[Int(lab) + 1]
            for (j, E) in pairs(de_vmi_energies)
                meas[i, j] = measured_hu(vmi_HU_final[E], lab)
                theo[i, j] = theoretical_hu(mat, E)
            end
        end
        out[group] = (
            labels = labels, names = ROD_NAMES[group],
            measured = meas, theoretical = theo,
        )
    end
    out
end;

# ╔═╡ 0600000e-0000-4000-8000-000000000001
md"""
### Per-Rod Regression
"""

# ╔═╡ 0600000e-0000-4000-8000-000000000010
let
    fig = CM.Figure(size = (1180, 580))

    cmap_ca = CM.cgrad(:Oranges, 7; categorical = true)
    cmap_i  = CM.cgrad(:GnBu,    7; categorical = true)

    panels = (
        (group = :Ca, title = "Calcium rods", subtitle = "50–600 mg/mL",
            cmap = cmap_ca, ylim = (0, 4200)),
        (group = :I,  title = "Iodine rods",  subtitle = "2–20 mg/mL",
            cmap = cmap_i,  ylim = (0, 1500)),
    )

    for (col, p) in pairs(panels)
        ax = CM.Axis(
            fig[1, col];
            title = p.title, subtitle = p.subtitle,
            xlabel = "VMI energy (keV)", ylabel = "HU",
            xticks = de_vmi_energies,
            titlesize = 32, subtitlesize = 24,
            xlabelsize = 22, ylabelsize = 22,
            xticklabelsize = 18, yticklabelsize = 16,
        )
        CM.ylims!(ax, p.ylim...)

        d = rod_data[p.group]
        rod_lines = Vector{Any}(undef, length(d.names))
        for i in eachindex(d.names)
            color = p.cmap[i]
            CM.scatterlines!(
                ax, de_vmi_energies, vec(d.measured[i, :]);
                color = color, linewidth = 2.5, markersize = 9,
            )
            CM.lines!(
                ax, de_vmi_energies, vec(d.theoretical[i, :]);
                color = color, linewidth = 1.6, linestyle = :dash,
            )
            rod_lines[i] = CM.LineElement(color = color, linewidth = 2.5)
        end

        style_meas = CM.MarkerElement(
            color = :black, marker = :circle, markersize = 9,
            strokecolor = :black, strokewidth = 1,
        )
        style_theo = CM.LineElement(
            color = :black, linewidth = 1.6, linestyle = :dash,
        )
        CM.axislegend(
            ax,
            vcat([style_meas, style_theo], rod_lines),
            vcat(["Measured", "Theoretical"], collect(d.names));
            position = :rt, framevisible = true, labelsize = 18,
            rowgap = 1, padding = (6, 6, 6, 6),
        )
    end

    fig
end

# ╔═╡ 0600000e-0000-4000-8000-000000000020
md"""
### Linear Regression
"""

# ╔═╡ 0600000e-0000-4000-8000-000000000030
let
    fig = CM.Figure(size = (1000, 1200))

    energy_colors = Dict(
         40.0 => CM.RGBf(0.85, 0.27, 0.10),
         70.0 => CM.RGBf(0.95, 0.65, 0.13),
        100.0 => CM.RGBf(0.13, 0.59, 0.85),
        140.0 => CM.RGBf(0.10, 0.27, 0.65),
    )

    function fit_lr(x::Vector{Float64}, y::Vector{Float64})
        x̄ = mean(x); ȳ = mean(y)
        sxx = sum((x .- x̄) .^ 2)
        sxy = sum((x .- x̄) .* (y .- ȳ))
        β = sxy / sxx
        α = ȳ - β * x̄
        ŷ = α .+ β .* x
        ss_res = sum((y .- ŷ) .^ 2)
        ss_tot = sum((y .- ȳ) .^ 2)
        r²   = 1 - ss_res / ss_tot
        rmse = sqrt(ss_res / length(y))
        return (slope = β, intercept = α, r² = r², rmse = rmse)
    end

    panels = (
        (:Ca, "Calcium Rods", "50–600 mg/mL"),
        (:I,  "Iodine Rods",  "2–20 mg/mL"),
    )

    for (row, (group, title, subtitle)) in pairs(panels)
        d = rod_data[group]
        ax = CM.Axis(
            fig[row, 1];
            title = title, subtitle = subtitle,
            xlabel = "Theoretical HU", ylabel = "Measured HU",
            titlesize = 32, subtitlesize = 24,
            xlabelsize = 22, ylabelsize = 22,
            xticklabelsize = 16, yticklabelsize = 16,
        )

        lim_lo = min(0.0, minimum(d.measured), minimum(d.theoretical))
        lim_hi = max(maximum(d.measured), maximum(d.theoretical)) * 1.05
        CM.lines!(
            ax, [lim_lo, lim_hi], [lim_lo, lim_hi];
            color = :black, linestyle = :dash, linewidth = 2,
            label = "Unity (y = x)",
        )

        for (j, E) in pairs(de_vmi_energies)
            x = Vector{Float64}(vec(d.theoretical[:, j]))
            y = Vector{Float64}(vec(d.measured[:, j]))
            color = energy_colors[E]
            CM.scatter!(ax, x, y; color = color, markersize = 11)

            f = fit_lr(x, y)
            xrange = [minimum(x), maximum(x)]
            yrange = f.intercept .+ f.slope .* xrange
            sign_str = f.intercept ≥ 0 ? "+" : "−"
            label = "$(Int(E)) keV: y = $(round(f.slope, digits = 2))·x " *
                "$(sign_str) $(round(abs(f.intercept), digits = 0)) HU   " *
                "R² = $(round(f.r², digits = 3))   " *
                "RMSE = $(round(f.rmse, digits = 1)) HU"
            CM.lines!(
                ax, xrange, yrange;
                color = color, linewidth = 2, label = label,
            )
        end

        CM.axislegend(
            ax; position = :rb, framevisible = true,
            labelsize = 16, padding = (6, 6, 6, 6), rowgap = 1,
        )
    end

    fig
end

# ╔═╡ 0600000e-0000-4000-8000-000000000040
md"""
### Solid Water HU vs VMI Energy

Mean HU over a **deeply-eroded** (12-px ≈ 8 mm) solid-water ROI at
each VMI energy.  Bars share the per-energy color scheme used in the
linear regression panels above so the eye can match across plots.

The expected value at every energy is ≈ 0 HU.  A roughly constant
offset across keVs means the basis decomposition is recovering
physics with a small uniform bias; an energy-dependent drift would
flag a spectral-shape issue upstream.
"""

# ╔═╡ 0600000e-0000-4000-8000-000000000050
let
    ERODE_PX = 12.0

    mask_2d_raw = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
    sw_bool_raw = (mask_2d_raw .== UInt8(BS.REGION_SOLID_WATER))
    sw_bool     = BS.erode_mask_2d(sw_bool_raw; erode_px = ERODE_PX)
    sw_idx      = findall(sw_bool)

    if isempty(sw_idx)
        error("solid water ROI empty after $(ERODE_PX)-px erosion; reduce erode_px")
    end

    n_z = size(vmi_HU_final[de_vmi_energies[1]], 3)
    function _mean_hu(vol)
        s = 0.0; n = 0
        for z in 1:n_z, ci in sw_idx
            s += vol[ci, z]; n += 1
        end
        return s / n
    end

    sw_hu_per_keV = [_mean_hu(vmi_HU_final[E]) for E in de_vmi_energies]

    # Same energy → color map as the linear regression panel.
    energy_colors = Dict(
         40.0 => CM.RGBf(0.85, 0.27, 0.10),
         70.0 => CM.RGBf(0.95, 0.65, 0.13),
        100.0 => CM.RGBf(0.13, 0.59, 0.85),
        140.0 => CM.RGBf(0.10, 0.27, 0.65),
    )
    bar_colors = [energy_colors[E] for E in de_vmi_energies]

    n_E = length(de_vmi_energies)

    fig = CM.Figure(size = (1000, 500))
    ax = CM.Axis(
        fig[1, 1];
        title = "Solid Water Mean HU",
        subtitle = "Per VMI Energy",
        xlabel = "VMI Energy (keV)", ylabel = "HU",
        xticks = (collect(1:n_E), ["$(Int(E))" for E in de_vmi_energies]),
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    CM.barplot!(
        ax, 1:n_E, sw_hu_per_keV;
        color = bar_colors,
        strokecolor = :black, strokewidth = 1,
    )
    CM.hlines!(ax, [0.0]; color = :black, linewidth = 1, linestyle = :dash)

    for (k, h) in pairs(sw_hu_per_keV)
        CM.text!(
            ax, k, h;
            text = "$(round(h, digits = 1)) HU",
            align = (:center, h ≥ 0 ? :bottom : :top),
            fontsize = 16, offset = (0, h ≥ 0 ? 4 : -4),
        )
    end

    y_max = max(15.0, 1.2 * maximum(abs, sw_hu_per_keV))
    CM.ylims!(ax, -y_max, y_max)

    fig
end

# ╔═╡ 0600000f-0000-4000-8000-000000000001
md"""
## Summary

```
Simulate 80 + 140 kVp  (scatter-corrected log-line-integral sinograms)
   → PRISM Inlined PWLS  (per detector row, AᵀV⁻¹A + λ∇R, KrylovJL_CG)
   → FBP × 2  (iodine, water basis maps)
   → Z-Direction Median Filter
   → Monoenergetic VMI Synthesis  (textbook 2-basis)
   → Mono+ Post-Processing
   → Measured vs Theoretical Per-Rod Regression  at 40 / 70 / 100 / 140 keV
```

Single solver knob: `PRISM_λ` ($(PRISM_λ)).  Effective μ values come
from the known polychromatic spectra (zero ROI-tuning).

!!! info "Where to go next"
    1. Sweep `PRISM_λ` over `[1e-4, 1e-3, 1e-2, 1e-1]` and watch the Ca/I
       rod regression slope and RMSE.
    2. Replace the global per-channel `V⁻¹` with per-pixel
       `1/(I₀·exp(-p))` Poisson weights from `BS.compute_detector_I0`.
    3. Replace effective-μ `A` with a polychromatic forward model and
       outer Gauss-Newton — the path discussed in the planning thread.
"""

# ╔═╡ Cell order:
# ╟─06000001-0000-4000-8000-000000000010
# ╟─06000001-0000-4000-8000-000000000020
# ╠═06000001-0000-4000-8000-000000000001
# ╠═06000001-0000-4000-8000-000000000002
# ╠═06000001-0000-4000-8000-000000000003
# ╠═06000001-0000-4000-8000-000000000004
# ╠═06000001-0000-4000-8000-000000000005
# ╠═06000001-0000-4000-8000-000000000006
# ╠═06000001-0000-4000-8000-000000000007
# ╠═06000001-0000-4000-8000-000000000008
# ╠═06000001-0000-4000-8000-000000000030
# ╠═06000001-0000-4000-8000-000000000031
# ╠═06000001-0000-4000-8000-000000000040
# ╟─06000001-0000-4000-8000-000000000050
# ╟─06000002-0000-4000-8000-000000000001
# ╠═06000002-0000-4000-8000-000000000010
# ╠═06000002-0000-4000-8000-000000000020
# ╟─06000003-0000-4000-8000-000000000001
# ╠═06000003-0000-4000-8000-000000000010
# ╟─06000004-0000-4000-8000-000000000001
# ╠═06000004-0000-4000-8000-000000000010
# ╠═06000004-0000-4000-8000-000000000020
# ╟─06000005-0000-4000-8000-000000000001
# ╠═06000005-0000-4000-8000-000000000010
# ╠═06000005-0000-4000-8000-000000000020
# ╟─06000006-0000-4000-8000-000000000001
# ╠═06000006-0000-4000-8000-000000000010
# ╠═06000006-0000-4000-8000-000000000020
# ╟─06000006-0000-4000-8000-000000000040
# ╟─06000007-0000-4000-8000-000000000001
# ╟─06000007-0000-4000-8000-000000000010
# ╠═06000007-0000-4000-8000-000000000011
# ╠═06000007-0000-4000-8000-000000000012
# ╠═06000007-0000-4000-8000-000000000013
# ╠═06000007-0000-4000-8000-000000000014
# ╟─06000007-0000-4000-8000-000000000020
# ╠═06000007-0000-4000-8000-000000000021
# ╠═06000007-0000-4000-8000-000000000022
# ╟─06000007-0000-4000-8000-000000000030
# ╠═06000007-0000-4000-8000-000000000031
# ╠═06000007-0000-4000-8000-000000000032
# ╟─06000007-0000-4000-8000-000000000040
# ╠═06000007-0000-4000-8000-000000000041
# ╠═06000007-0000-4000-8000-000000000042
# ╠═06000007-0000-4000-8000-000000000043
# ╟─06000007-0000-4000-8000-000000000050
# ╠═06000007-0000-4000-8000-000000000051
# ╟─06000007-0000-4000-8000-000000000060
# ╠═06000007-0000-4000-8000-000000000061
# ╟─06000007-0000-4000-8000-000000000070
# ╠═06000007-0000-4000-8000-000000000071
# ╟─06000007-0000-4000-8000-000000000080
# ╠═06000007-0000-4000-8000-000000000081
# ╠═06000007-0000-4000-8000-000000000082
# ╠═06000007-0000-4000-8000-000000000083
# ╟─06000008-0000-4000-8000-000000000001
# ╠═06000008-0000-4000-8000-000000000010
# ╟─06000008-0000-4000-8000-000000000020
# ╠═06000008-0000-4000-8000-000000000021
# ╠═06000008-0000-4000-8000-000000000022
# ╟─06000008-0000-4000-8000-000000000030
# ╠═06000008-0000-4000-8000-000000000031
# ╠═06000008-0000-4000-8000-000000000040
# ╟─06000008-0000-4000-8000-000000000050
# ╟─06000009-0000-4000-8000-000000000001
# ╠═06000009-0000-4000-8000-000000000010
# ╟─06000009-0000-4000-8000-000000000030
# ╟─0600000a-0000-4000-8000-000000000001
# ╠═0600000a-0000-4000-8000-000000000005
# ╠═0600000a-0000-4000-8000-000000000010
# ╟─0600000b-0000-4000-8000-000000000001
# ╠═0600000b-0000-4000-8000-000000000015
# ╠═0600000b-0000-4000-8000-000000000020
# ╟─0600000b-0000-4000-8000-000000000040
# ╟─0600000c-0000-4000-8000-000000000001
# ╠═0600000c-0000-4000-8000-000000000005
# ╠═0600000c-0000-4000-8000-000000000010
# ╟─0600000c-0000-4000-8000-000000000030
# ╟─0600000d-0000-4000-8000-000000000001
# ╠═0600000d-0000-4000-8000-000000000010
# ╠═0600000d-0000-4000-8000-000000000020
# ╠═0600000d-0000-4000-8000-000000000030
# ╟─0600000e-0000-4000-8000-000000000001
# ╟─0600000e-0000-4000-8000-000000000010
# ╟─0600000e-0000-4000-8000-000000000020
# ╟─0600000e-0000-4000-8000-000000000030
# ╟─0600000e-0000-4000-8000-000000000040
# ╟─0600000e-0000-4000-8000-000000000050
# ╟─0600000f-0000-4000-8000-000000000001
