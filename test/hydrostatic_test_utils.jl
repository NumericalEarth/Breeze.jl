# Helpers shared by the tests of `HydrostaticDynamics`. Not a test: removed from the test suite in
# `runtests.jl`.

using Breeze
using Oceananigans
using Oceananigans.Advection: VectorInvariant

# The dry hydrostatic reference core is a Float64 oracle: the tolerances below are statements about
# double-precision rounding.
const FT = Float64

#####
##### Helpers
#####

# Constants of the one-dimensional reference model that produced the vertical-mode speeds below
const g = 9.80665
const Rᵈ = 287.0
const cᵖᵈ = 1004.5
const κ = Rᵈ / cᵖᵈ
const pˢᵗ = 1e5

reference_model_constants() =
    ThermodynamicConstants(FT; gravitational_acceleration = g,
                               dry_air_molar_mass = 8.314462618 / Rᵈ,
                               dry_air_heat_capacity = cᵖᵈ)

# Interfaces of η = p / pˢᵗ at pˢ = pˢᵗ, from the model top down: uniform in a variable that blends
# log-pressure, pressure, and a surface-concentrated term
function eta_interfaces(N; top_pressure, w_log=0.4, w_lin=0.4, w_sfc=0.2, δ=0.1)
    ηᵗ = top_pressure / pˢᵗ
    e(η) = exp((η - 1) / δ)
    s(η) = w_log * log(η / ηᵗ) / log(1 / ηᵗ) + w_lin * (η - ηᵗ) / (1 - ηᵗ) +
           w_sfc * (e(η) - e(ηᵗ)) / (1 - e(ηᵗ))
    η = zeros(N + 1)
    for j in 0:N
        target = j / N
        lo, hi = ηᵗ, 1.0
        for _ in 1:200
            mid = (lo + hi) / 2
            s(mid) < target ? (lo = mid) : (hi = mid)
        end
        η[j+1] = (lo + hi) / 2
    end
    η[1] = ηᵗ
    η[end] = 1.0
    return η
end

# Hybrid coefficients, ordered from the ground to the model top: pure pressure above ηᶜ
function hybrid_coefficients(N; top_pressure=200.0, ηᶜ=0.08, r=1.5)
    η = eta_interfaces(N; top_pressure)
    B = [ηʲ > ηᶜ ? ((ηʲ - ηᶜ) / (1 - ηᶜ))^r : 0.0 for ηʲ in η]
    B[end] = 1.0
    A = pˢᵗ .* (η .- B)
    A[end] = 0.0
    return reverse(A), reverse(B)
end

# Temperature of the standard atmosphere as a function of pressure
standard_temperature(p) = max(288.15 * (p / pˢᵗ)^(Rᵈ * 0.0065 / g), 216.65)

function hydrostatic_model(grid; tracers=(), kw...)
    return AtmosphereModel(grid; momentum_advection=VectorInvariant(), tracers, kw...)
end

function total(field)
    integral = Field(Integral(field))
    compute!(integral)
    return first(Array(interior(integral)))
end

max_abs(field) = maximum(abs, Array(interior(field)))

# Surface pressure of a resting isentropic atmosphere of potential temperature θ₀ over the
# surface geopotential Φˢ: Φˢ + cᵖᵈ θ₀ Πˢ is uniform
isentropic_surface_pressure(Φˢ, θ₀) = pˢᵗ * (1 - Φˢ / (cᵖᵈ * θ₀))^(1 / κ)

#####
##### Vertical modes of the resting standard atmosphere on 30 hybrid levels
#####

# The three largest eigenvalues of the vertical-structure problem of this discretization (the
# campaign's one-dimensional reference model, `vertical_modes.csv`, rows N = 30, state = standard)
const reference_mode_speeds = (306.337367, 176.474257, 107.565423) # m s⁻¹

# Power at frequency ω of the Hann-windowed time series in the columns of `signals`
function periodogram(ω, times, signals, window)
    phase = window .* cis.(-ω .* times)
    return sum(abs2, transpose(signals) * phase)
end

# Frequency of the largest periodogram peak in (ω⁻, ω⁺), by a scan and a golden-section refinement
function peak_frequency(ω⁻, ω⁺, times, signals, window; scan=400)
    ωs = range(ω⁻, ω⁺, length=scan)
    powers = [periodogram(ω, times, signals, window) for ω in ωs]
    n = argmax(powers)
    interior_peak = 1 < n < scan
    a, b = ωs[max(n - 1, 1)], ωs[min(n + 1, scan)]
    ϕ = (sqrt(5) - 1) / 2
    for _ in 1:60
        c, d = b - ϕ * (b - a), a + ϕ * (b - a)
        if periodogram(c, times, signals, window) > periodogram(d, times, signals, window)
            b = d
        else
            a = c
        end
    end
    return (a + b) / 2, interior_peak, maximum(powers) / minimum(powers)
end

#####
##### Split-explicit configurations
#####

split_explicit(tableau, inner_solver; kw...) =
    (timestepper = :GravityWaveMultirateInfinitesimalStep,
     timestepper_kwargs = (; tableau, inner_solver, kw...))

configuration_name(tableau, inner_solver) = string(tableau, " + ", summary(inner_solver))

#####
##### The baroclinic wave test of Jablonowski and Williamson (2006), between walls
#####
#
# The steady state of Jablonowski and Williamson (2006): a zonal jet of 35 m s⁻¹ in each
# hemisphere in thermal-wind balance with its temperature field, over the surface geopotential
# that balances the jet at the ground, under a uniform surface pressure of 1e5 Pa. It is an exact
# steady solution of the hydrostatic equations on the sphere. The perturbed case adds the
# standard localized perturbation of the zonal wind, 1 m s⁻¹ at (20°E, 40°N) with a Gaussian
# profile of radius a / 10 in great-circle distance, at every level.
#
# The case is frozen here: a 2° latitude-longitude grid between walls at 70°S and 70°N, along
# which the unperturbed flow is zonal, on `Nz` sigma levels uniform in σ with the model top at
# 50 hPa, without dissipation of any kind. The walls replace the poles, so this is not the
# published test and its results are not comparable with the published ones: its reference is
# the unsplit integration of the same discrete equations at a small time step.

const sphere_radius = 6.371229e6    # m
const rotation_rate = 7.29212e-5    # s⁻¹
const jet_speed = 35.0              # m s⁻¹
const jet_level = 0.252             # η₀
const tropopause_level = 0.2        # ηᵗ
const surface_temperature = 288.0   # K
const lapse_rate = 0.005            # K m⁻¹
const stratospheric_temperature_difference = 4.8e5 # K
const perturbation_speed = 1.0      # m s⁻¹
const perturbation_longitude = 20.0 # degrees
const perturbation_latitude = 40.0  # degrees
const perturbation_radius = sphere_radius / 10

vertical_argument(η) = (η - jet_level) * π / 2

jet(φ, η) = jet_speed * cos(vertical_argument(η))^(3/2) * sind(2φ)^2

mean_temperature(η) = surface_temperature * η^(Rᵈ * lapse_rate / g) +
                      (η < tropopause_level ? stratospheric_temperature_difference * (tropopause_level - η)^5 : 0.0)

latitude_structure_1(φ) = -2 * sind(φ)^6 * (cosd(φ)^2 + 1/3) + 10/63
latitude_structure_2(φ) = 8/5 * cosd(φ)^3 * (sind(φ)^2 + 2/3) - π/4

function balanced_temperature(φ, η)
    ηᵛ = vertical_argument(η)
    return mean_temperature(η) + 3/4 * η * π * jet_speed / Rᵈ * sin(ηᵛ) * sqrt(cos(ηᵛ)) *
           (latitude_structure_1(φ) * 2 * jet_speed * cos(ηᵛ)^(3/2) + latitude_structure_2(φ) * sphere_radius * rotation_rate)
end

function balanced_surface_geopotential(λ, φ)
    c = cos(vertical_argument(1.0))^(3/2)
    return jet_speed * c * (latitude_structure_1(φ) * jet_speed * c + latitude_structure_2(φ) * sphere_radius * rotation_rate)
end

# The perturbation of the zonal wind: a Gaussian of the great-circle distance to its center
function jet_perturbation(λ, φ)
    cosine = sind(perturbation_latitude) * sind(φ) +
             cosd(perturbation_latitude) * cosd(φ) * cosd(λ - perturbation_longitude)
    r = sphere_radius * acos(clamp(cosine, -1, 1))
    return perturbation_speed * exp(-(r / perturbation_radius)^2)
end

function baroclinic_wave_model(; Nz=10, perturbed=true, kw...)
    grid = LatitudeLongitudeGrid(default_arch, FT; size=(180, 70, Nz), longitude=(0, 360), latitude=(-70, 70),
                                 z=(0, 1), radius=sphere_radius)

    dynamics = HydrostaticDynamics(surface_pressure = pˢᵗ,
                                   standard_pressure = pˢᵗ,
                                   top_pressure = 5000,
                                   surface_geopotential = balanced_surface_geopotential)

    model = hydrostatic_model(grid; dynamics, coriolis = HydrostaticSphericalCoriolis(FT; rotation_rate),
                              thermodynamic_constants = reference_model_constants(), kw...)

    # η = p / pˢ on each level, uniform in the horizontal because the surface pressure is
    p = Array(interior(dynamics_pressure(model.dynamics)))
    η = p[1, 1, :] ./ pˢᵗ
    λ = Array(λnodes(grid, Face()))
    φ = Array(φnodes(grid, Center()))
    Nx, Ny, Nz = size(grid)

    θ = [balanced_temperature(φ[j], η[k]) / η[k]^κ for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    u = [jet(φ[j], η[k]) + perturbed * jet_perturbation(λ[i], φ[j]) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    set!(model; θ, u)

    return model
end

balanced_jet_model(; kw...) = baroclinic_wave_model(; perturbed=false, kw...)

zonal_departure(u) = maximum(abs, u .- sum(u, dims=1) ./ size(u, 1))

# Area weights of the cell centers of a latitude-longitude grid, normalized to sum to one
function area_weights(grid)
    φ = Array(φnodes(grid, Center()))
    Nx = size(grid, 1)
    weights = [cosd(φ[j]) for i in 1:Nx, j in eachindex(φ)]
    return weights ./ sum(weights)
end

# Area-weighted root-mean-square of a two-dimensional array at the cell centers
weighted_rms(field, weights) = sqrt(sum(weights .* field .^ 2))
