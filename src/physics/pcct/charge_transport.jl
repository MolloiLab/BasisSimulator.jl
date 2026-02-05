# =============================================================================
# Charge Cloud Transport Model for CdTe PCCT Detectors
# =============================================================================
#
# Physics-based charge cloud transport using drift-diffusion with electrostatic
# repulsion (Koch-Mehrin 2020, NIM-A 976:164241, Section 2.3, Equations 9-14).
#
# Replaces the ad-hoc fixed Gaussian blur (FWHM=0.08mm, σ=34μm) with a
# depth-dependent, energy-dependent charge cloud model yielding σ≈13μm.
#
# The charge cloud grows during electron drift from interaction site to anode
# via two mechanisms:
#   1. Thermal diffusion (Fick's law): σ_d² grows linearly with time
#   2. Electrostatic repulsion: charge carriers repel, growth ∝ 1/σ
#
# The final cloud size also includes the initial cloud from photoelectron
# excitation (Blevis & Levinson 2005 via Koch-Mehrin 2020).
#
# DESIGN: Pre-compute σ(E, z) lookup table on CPU, transfer to GPU.
# The LUT is computed once per detector configuration and reused.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Core ODE Solver: Charge Cloud Growth During Drift
# ─────────────────────────────────────────────────────────────────────────────

"""
    charge_cloud_sigma_mm(E_keV, z_fraction, geom::PCCTDetectorGeometry;
                           transport::CdTeTransport=CDTE_TRANSPORT,
                           n_steps::Int=1000) -> Float64

Compute the charge cloud standard deviation at the anode for a photon of
energy `E_keV` absorbed at fractional depth `z_fraction` from the cathode.

Solves the drift-diffusion ODE (Koch-Mehrin 2020, Eq. 14):

    dσ²/dt = 2D + μₑ·N·q / (12π^(3/2) · ε · σ(t))

where:
- D = μₑ·kB·T/q  (Einstein diffusion coefficient)
- N = E/(ω·q)     (number of e-h pairs, converted from eV)
- ε = εr·ε₀       (CdTe permittivity in F/m)
- σ(t)            (current cloud size, in meters for SI consistency)

The ODE is solved using RK4 with `n_steps` time steps.

# Arguments
- `E_keV`: Photon energy in keV
- `z_fraction`: Fractional depth from cathode (0=cathode, 1=anode)
- `geom`: Detector geometry (thickness, bias voltage)
- `transport`: CdTe transport properties
- `n_steps`: Number of RK4 integration steps

# Returns
- Total charge cloud σ in mm (including initial cloud size)

# References
- Koch-Mehrin 2020, Eq. 9-14 (charge cloud transport)
- Blevis & Levinson 2005 (initial cloud polynomial)
"""
function charge_cloud_sigma_mm(
    E_keV::Real,
    z_fraction::Real,
    geom::PCCTDetectorGeometry;
    transport::CdTeTransport=CDTE_TRANSPORT,
    n_steps::Int=1000
)
    E = Float64(E_keV)
    zf = clamp(Float64(z_fraction), 0.0, 1.0)

    # Detector parameters
    L_mm = geom.thickness_mm
    L_m = L_mm * 1e-3                      # sensor thickness [m]
    V = geom.effective_voltage_V            # effective bias voltage [V]

    # Drift distance: from interaction point to anode
    # z_fraction=0 → cathode (top), z_fraction=1 → anode (bottom)
    # Electrons drift toward anode, so drift distance = L - z = L(1 - z_fraction)
    d_m = L_m * (1.0 - zf)

    # If absorbed right at anode, no drift → only initial cloud
    if d_m < 1e-9
        σ_i_um = initial_cloud_sigma_um(E)
        return σ_i_um * 1e-3  # μm → mm
    end

    # Number of electron-hole pairs
    N = num_electron_hole_pairs(E; transport=transport)

    # Initial charge cloud size (Blevis & Levinson 2005 polynomial)
    σ_i_um = initial_cloud_sigma_um(E)
    σ_i_m = σ_i_um * 1e-6                  # μm → m

    # Electron mobility in SI: cm²/V·s → m²/V·s
    μ_e_SI = transport.electron_mobility_cm2_per_Vs * 1e-4  # m²/V·s

    # Diffusion coefficient (Koch-Mehrin 2020, Eq. 11)
    # D = μₑ · kB · T / q  [m²/s]
    D_SI = μ_e_SI * BOLTZMANN_J_PER_K * transport.temperature_K / ELEMENTARY_CHARGE_C

    # Drift time (Koch-Mehrin 2020, Eq. 12)
    # t_d = d · L / (μₑ · V)  [s]
    # Note: Koch-Mehrin uses drift_length × sensor_thickness / (μₑ × V)
    # This comes from: t = d / v_drift, where v_drift = μₑ × E_field = μₑ × V/L
    # So t = d × L / (μₑ × V)
    t_d = d_m * L_m / (μ_e_SI * V)

    # Permittivity in SI (F/m)
    ε = transport.permittivity_F_per_m

    # Repulsion prefactor (Koch-Mehrin 2020, Eq. 14)
    # Repulsion term: μₑ · N · q / (12π^(3/2) · ε · σ(t))
    # In SI: [m²/V·s] · [1] · [C] / ([F/m] · [m]) = [m²/V·s · C · m / (C/V · m)] = [m²/s]
    # This is dimensionally correct: [m²/V·s · C / (F/m · m)] = [m²/V·s · C·m / (As/V · m²)] = [m²/s]
    repulsion_prefactor = μ_e_SI * N * ELEMENTARY_CHARGE_C / (12.0 * π^1.5 * ε)

    # Solve ODE: dσ²/dt = 2D + repulsion_prefactor / σ(t)
    # Using RK4 integration
    dt = t_d / n_steps
    σ_sq = σ_i_m^2                          # initial condition: σ² = σ_i²

    # RK4 integration of dσ²/dt = f(σ²)
    for _ in 1:n_steps
        σ_current = sqrt(max(σ_sq, σ_i_m^2))

        # f(σ) = 2D + repulsion_prefactor / σ
        f1 = 2.0 * D_SI + repulsion_prefactor / σ_current

        σ_sq_mid1 = σ_sq + 0.5 * dt * f1
        σ_mid1 = sqrt(max(σ_sq_mid1, σ_i_m^2))
        f2 = 2.0 * D_SI + repulsion_prefactor / σ_mid1

        σ_sq_mid2 = σ_sq + 0.5 * dt * f2
        σ_mid2 = sqrt(max(σ_sq_mid2, σ_i_m^2))
        f3 = 2.0 * D_SI + repulsion_prefactor / σ_mid2

        σ_sq_end = σ_sq + dt * f3
        σ_end = sqrt(max(σ_sq_end, σ_i_m^2))
        f4 = 2.0 * D_SI + repulsion_prefactor / σ_end

        σ_sq += dt * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0
        σ_sq = max(σ_sq, σ_i_m^2)  # σ² cannot decrease below initial
    end

    # Final drift cloud size [m]
    σ_d_m = sqrt(σ_sq)

    # Total cloud: quadrature sum of drift and initial (Koch-Mehrin 2020, Eq. 9)
    # σ_total = sqrt(σ_d² + σ_i²)
    σ_total_m = sqrt(σ_d_m^2 + σ_i_m^2)

    return σ_total_m * 1e3  # m → mm
end

# ─────────────────────────────────────────────────────────────────────────────
# Pre-computed Lookup Table for GPU Efficiency
# ─────────────────────────────────────────────────────────────────────────────

"""
    ChargeCloudLUT{T}

Pre-computed lookup table mapping (energy, depth) → charge cloud σ.

# Fields
- `energies_keV`: Energy grid points [keV]
- `depth_fractions`: Depth grid points (0=cathode, 1=anode)
- `sigma_mm`: 2D array σ[energy_idx, depth_idx] in mm
- `geom`: Detector geometry used for computation
"""
struct ChargeCloudLUT{T<:AbstractFloat}
    energies_keV::Vector{T}
    depth_fractions::Vector{T}
    sigma_mm::Matrix{T}
    geom::PCCTDetectorGeometry
end

"""
    compute_charge_cloud_lut(geom::PCCTDetectorGeometry;
                              transport::CdTeTransport=CDTE_TRANSPORT,
                              E_min::Real=1.0, E_max::Real=150.0, n_energies::Int=150,
                              n_depths::Int=50) -> ChargeCloudLUT

Pre-compute charge cloud σ lookup table for all (energy, depth) combinations.

This table is computed once on CPU and can be transferred to GPU for kernel use.

# Arguments
- `geom`: Detector geometry
- `transport`: CdTe transport properties
- `E_min`, `E_max`: Energy range in keV
- `n_energies`: Number of energy grid points
- `n_depths`: Number of depth grid points

# Returns
- `ChargeCloudLUT` with pre-computed σ values
"""
function compute_charge_cloud_lut(
    geom::PCCTDetectorGeometry;
    transport::CdTeTransport=CDTE_TRANSPORT,
    E_min::Real=1.0,
    E_max::Real=150.0,
    n_energies::Int=150,
    n_depths::Int=50
)
    energies = collect(range(Float64(E_min), Float64(E_max), length=n_energies))
    depths = collect(range(0.0, 1.0, length=n_depths))
    sigma = zeros(Float64, n_energies, n_depths)

    for j in 1:n_depths
        for i in 1:n_energies
            sigma[i, j] = charge_cloud_sigma_mm(energies[i], depths[j], geom;
                                                  transport=transport)
        end
    end

    return ChargeCloudLUT{Float64}(energies, depths, sigma, geom)
end

"""
    lookup_sigma_mm(lut::ChargeCloudLUT, E_keV::Real, z_fraction::Real) -> Float64

Bilinear interpolation of charge cloud σ from pre-computed LUT.

# Arguments
- `lut`: Pre-computed lookup table
- `E_keV`: Photon energy in keV
- `z_fraction`: Fractional depth (0=cathode, 1=anode)

# Returns
- Charge cloud σ in mm
"""
function lookup_sigma_mm(lut::ChargeCloudLUT{T}, E_keV::Real, z_fraction::Real) where T
    E = clamp(T(E_keV), lut.energies_keV[1], lut.energies_keV[end])
    z = clamp(T(z_fraction), T(0), T(1))

    # Find energy index (linear spacing)
    n_E = length(lut.energies_keV)
    dE = (lut.energies_keV[end] - lut.energies_keV[1]) / (n_E - 1)
    fi = (E - lut.energies_keV[1]) / dE + 1.0
    i0 = clamp(floor(Int, fi), 1, n_E - 1)
    i1 = i0 + 1
    wi = fi - i0

    # Find depth index (linear spacing)
    n_z = length(lut.depth_fractions)
    dz = 1.0 / (n_z - 1)
    fj = z / dz + 1.0
    j0 = clamp(floor(Int, fj), 1, n_z - 1)
    j1 = j0 + 1
    wj = fj - j0

    # Bilinear interpolation
    σ = (1 - wi) * (1 - wj) * lut.sigma_mm[i0, j0] +
        wi       * (1 - wj) * lut.sigma_mm[i1, j0] +
        (1 - wi) * wj       * lut.sigma_mm[i0, j1] +
        wi       * wj       * lut.sigma_mm[i1, j1]

    return Float64(σ)
end

# ─────────────────────────────────────────────────────────────────────────────
# Depth-Averaged σ for Polychromatic Beams
# ─────────────────────────────────────────────────────────────────────────────

"""
    mean_charge_cloud_sigma_mm(E_keV::Real, geom::PCCTDetectorGeometry;
                                transport::CdTeTransport=CDTE_TRANSPORT,
                                n_depths::Int=20) -> Float64

Compute depth-averaged charge cloud σ for a monoenergetic beam, weighted
by Beer-Lambert absorption probability.

The absorption probability at depth z is:
    p(z) ∝ μ(E) · exp(-μ(E) · z)

For CdTe at diagnostic energies, most absorption occurs near the cathode
(entrance surface), so interactions have longer drift distances and thus
larger charge clouds than a uniform-depth average would suggest.

# Arguments
- `E_keV`: Photon energy in keV
- `geom`: Detector geometry
- `transport`: CdTe transport properties
- `n_depths`: Number of depth integration points

# Returns
- Depth-averaged σ in mm
"""
function mean_charge_cloud_sigma_mm(
    E_keV::Real,
    geom::PCCTDetectorGeometry;
    transport::CdTeTransport=CDTE_TRANSPORT,
    n_depths::Int=20
)
    L_mm = geom.thickness_mm

    # CdTe linear attenuation coefficient at this energy
    # Approximate: μ_CdTe ≈ ρ × μ/ρ (mass attenuation)
    # For CdTe at 60 keV: μ ≈ ~3 cm⁻¹ (30 mm⁻¹ is too high, ~3/mm)
    # Use a simple exponential model: most photons absorbed near entrance
    # μ_CdTe from NIST at key energies:
    #   30 keV: ~30 cm⁻¹, 60 keV: ~6 cm⁻¹, 100 keV: ~2.5 cm⁻¹
    # Approximate: μ ∝ E^(-2.8) for photoelectric-dominated range
    E = max(Float64(E_keV), 1.0)
    # Rough CdTe μ approximation (cm⁻¹), calibrated to μ≈6 cm⁻¹ at 60 keV
    μ_approx_cm = 6.0 * (60.0 / E)^2.8  # cm⁻¹
    μ_approx_mm = μ_approx_cm / 10.0     # mm⁻¹

    σ_weighted = 0.0
    weight_sum = 0.0

    for i in 1:n_depths
        z_mm = L_mm * (i - 0.5) / n_depths
        z_frac = z_mm / L_mm

        # Beer-Lambert absorption weight: μ·exp(-μ·z)
        w = μ_approx_mm * exp(-μ_approx_mm * z_mm)

        σ = charge_cloud_sigma_mm(E, z_frac, geom; transport=transport, n_steps=200)
        σ_weighted += w * σ
        weight_sum += w
    end

    return weight_sum > 0 ? σ_weighted / weight_sum : charge_cloud_sigma_mm(E, 0.5, geom; transport=transport)
end

# ─────────────────────────────────────────────────────────────────────────────
# Charge Sharing Probability from Cloud Size
# ─────────────────────────────────────────────────────────────────────────────

"""
    charge_sharing_probability(σ_mm::Real, pixel_pitch_mm::Tuple{<:Real,<:Real}) -> Float64

Compute the probability that a charge cloud extends beyond the pixel boundary.

For a 2D Gaussian charge cloud centered at a random position within a pixel,
the probability that charge crosses a boundary depends on σ/pixel_size.

The probability is computed as the fraction of the pixel area where the cloud
center would result in >5% of charge crossing any boundary (3σ criterion):
    P_share ≈ 1 - ((w - 2·k·σ) / w)²    for k·σ < w/2
where w is pixel pitch and k≈2 (for ~5% threshold).

For a noise threshold of 3 keV and ω=4.3 eV:
    N_threshold = 3000/4.3 ≈ 698 e-h pairs
At 60 keV: N = 13953, so 5% = 698 pairs — matches noise threshold!

# Arguments
- `σ_mm`: Charge cloud standard deviation in mm
- `pixel_pitch_mm`: Pixel pitch (row, col) in mm

# Returns
- Probability of charge sharing (0 to 1)
"""
function charge_sharing_probability(σ_mm::Real, pixel_pitch_mm::Tuple{<:Real,<:Real})
    σ = Float64(σ_mm)
    w_row = Float64(pixel_pitch_mm[1])
    w_col = Float64(pixel_pitch_mm[2])

    if σ ≤ 0.0
        return 0.0
    end

    # Use k=2σ as the effective charge cloud radius (enclosing ~95% of charge)
    # A photon absorbed within kσ of any boundary will share charge
    k = 2.0

    # Fraction of pixel area within kσ of the boundary
    # Inner safe region: (w - 2kσ) × (w - 2kσ)
    inner_row = max(w_row - 2.0 * k * σ, 0.0)
    inner_col = max(w_col - 2.0 * k * σ, 0.0)

    p_no_share = (inner_row / w_row) * (inner_col / w_col)
    return 1.0 - p_no_share
end
