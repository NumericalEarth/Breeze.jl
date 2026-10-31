# Shared setup for the tier-C (2D x–z) thermodynamic-formulation validation suite.
#
# Formulation labels used by every case:
#
#   C-theta  CompressibleDynamics(SplitExplicitTimeDiscretization()), LiquidIcePotentialTemperatureFormulation()
#   C-E0     CompressibleDynamics(...), TotalEnergyFormulation(reference_temperature = 0)
#   C-E273   CompressibleDynamics(...), TotalEnergyFormulation(reference_temperature = 273.16)
#   C-eta    CompressibleDynamics(...), MoistEntropyFormulation()
#   A-theta  AnelasticDynamics(ReferenceState(...)), LiquidIcePotentialTemperatureFormulation()
#   A-s      AnelasticDynamics(ReferenceState(...)), :StaticEnergy
#
# The compressible core time-steps with AcousticRungeKutta3 (the default for the split-explicit
# discretization); the anelastic core with its default SSP RK3.

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: on_architecture
using Oceananigans.Utils: launch!, @kernel, @index
using Oceananigans.TurbulenceClosures: ScalarDiffusivity
using Breeze.AtmosphereModels: grid_moisture_fractions, specific_prognostic_moisture, total_density, dynamics_pressure
using Breeze.Thermodynamics: absolute_zero_latent_heat, vapor_gas_constant, dry_air_gas_constant
using CUDA
using Printf
using Serialization
using Statistics

const ALL_FORMULATIONS = ("C-theta", "C-E0", "C-E273", "C-eta", "A-theta", "A-s")

is_compressible(name) = startswith(name, "C-")

architecture() = (CUDA.functional() && get(ENV, "CPU", "0") != "1") ? GPU() : CPU()

function thermodynamic_formulation(name)
    name in ("C-theta", "A-theta") && return LiquidIcePotentialTemperatureFormulation()
    name == "C-E0"   && return TotalEnergyFormulation(reference_temperature = 0)
    name == "C-E273" && return TotalEnergyFormulation(reference_temperature = 273.16)
    name == "C-eta"  && return MoistEntropyFormulation()
    name == "A-s"    && return :StaticEnergy
    error("unknown formulation $name")
end

"""
    build_model(name, grid; constants, base_pressure, θ_reference, qᵛ_reference=nothing, kw...)

An `AtmosphereModel` with the dynamics and thermodynamic formulation that `name` labels.
`θ_reference` (number or function of z) is the reference potential temperature of either core.
"""
function build_model(name, grid; constants, base_pressure, θ_reference, qᵛ_reference = nothing, kw...)
    if is_compressible(name)
        moist = qᵛ_reference === nothing ? (;) : (; reference_vapor_mass_fraction = qᵛ_reference)
        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure,
                                        reference_potential_temperature = θ_reference, moist...)
    else
        moist = qᵛ_reference === nothing ? (;) : (; vapor_mass_fraction = qᵛ_reference)
        reference_state = ReferenceState(grid, constants; base_pressure, potential_temperature = θ_reference, moist...)
        dynamics = AnelasticDynamics(reference_state)
    end
    return AtmosphereModel(grid; dynamics, formulation = thermodynamic_formulation(name),
                           thermodynamic_constants = constants, kw...)
end

# Reference pressure and density on the model grid (host arrays, length Nz)
reference_column(model, field) = vec(Float64.(Array(interior(getproperty(model.dynamics.reference_state, field))))[1:size(model.grid, 3)])

#####
##### Host snapshots and integrals (Float64, cell-centered)
#####

@kernel function _moisture_fractions!(qv, ql, qi, grid, microphysics, ρ, qᵛᵉ, μ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ[i, j, k], qᵛᵉ[i, j, k], μ)
        qv[i, j, k] = q.vapor
        ql[i, j, k] = q.liquid
        qi[i, j, k] = q.ice
    end
end

host(f) = Float64.(Array(interior(f)))
squeeze_y(a) = a[:, 1, :]

# Mean of the squares of the two faces bounding each center (x faces), Bounded or Periodic in x
function center_square_x(u, Nx)
    if size(u, 1) == Nx + 1
        return (u[1:Nx, :] .^ 2 .+ u[2:Nx+1, :] .^ 2) ./ 2
    else
        return (u .^ 2 .+ circshift(u, -1) .^ 2) ./ 2
    end
end
center_square_z(w, Nz) = (w[:, 1:Nz] .^ 2 .+ w[:, 2:Nz+1] .^ 2) ./ 2
center_average_z(w, Nz) = (w[:, 1:Nz] .+ w[:, 2:Nz+1]) ./ 2

"""
    snapshot(model)

Host Float64 x–z arrays (Nx × Nz) of the state: T, p, ρ (total; the reference density on the
anelastic core), qv, ql, qi, the cell-centered K, and the face velocities u (x faces) and w (z faces).
"""
function snapshot(model)
    grid = model.grid
    Nx, Ny, Nz = size(grid)
    qv, ql, qi = CenterField(grid), CenterField(grid), CenterField(grid)
    ρ = total_density(model.dynamics)
    launch!(architecture_of(model), grid, :xyz, _moisture_fractions!, qv, ql, qi, grid, model.microphysics,
            ρ, specific_prognostic_moisture(model), model.microphysical_fields)
    T = squeeze_y(host(model.temperature))
    z1 = zeros(size(T))
    p = squeeze_y(host(dynamics_pressure(model.dynamics)) .+ zeros(Nx, 1, Nz))
    ρh = squeeze_y(host(ρ) .+ zeros(Nx, 1, Nz))
    u = squeeze_y(host(model.velocities.u))
    w = squeeze_y(host(model.velocities.w))
    K = (center_square_x(u, Nx) .+ center_square_z(w, Nz)) ./ 2
    return (; T, p, ρ = ρh, qv = squeeze_y(host(qv)) .+ z1, ql = squeeze_y(host(ql)) .+ z1,
              qi = squeeze_y(host(qi)) .+ z1, K, u, w, wc = center_average_z(w, Nz))
end

architecture_of(model) = Oceananigans.Architectures.architecture(model.grid)

struct Constants64
    Rd :: Float64; Rv :: Float64; cpd :: Float64; cpv :: Float64; cl :: Float64; ci :: Float64
    g :: Float64; Lv0 :: Float64; Li0 :: Float64; Ttr :: Float64; ptr :: Float64; ηv0 :: Float64; ηi0 :: Float64
    pst :: Float64
end

function Constants64(constants)
    Rd = Float64(dry_air_gas_constant(constants))
    Rv = Float64(vapor_gas_constant(constants))
    Ttr = Float64(constants.triple_point_temperature)
    ℒˡᵗʳ = Float64(Breeze.Thermodynamics.liquid_latent_heat(Ttr, constants))
    ℒⁱᵗʳ = Float64(Breeze.Thermodynamics.ice_latent_heat(Ttr, constants))
    return Constants64(Rd, Rv, Float64(constants.dry_air.heat_capacity), Float64(constants.vapor.heat_capacity),
                       Float64(constants.liquid.heat_capacity), Float64(constants.ice.heat_capacity),
                       Float64(constants.gravitational_acceleration),
                       Float64(absolute_zero_latent_heat(constants, constants.liquid)),
                       Float64(absolute_zero_latent_heat(constants, constants.ice)),
                       Ttr, Float64(constants.triple_point_pressure), ℒˡᵗʳ / Ttr, (ℒⁱᵗʳ - ℒˡᵗʳ) / Ttr, 1e5)
end

xlogy_safe(x, y) = x > 0 ? x * log(y) : 0.0

# Moist entropy per unit total mass (liquid water at the triple point; Breeze's constant heat capacities)
function specific_entropy(T, ρ, qv, ql, qi, c::Constants64)
    qd = 1 - qv - ql - qi
    lnT = log(T / c.Ttr)
    return qd * (c.cpd * lnT - c.Rd * log(ρ * qd * c.Rd * T / c.ptr)) +
           qv * (c.cpv * lnT + c.ηv0) - c.Rv * xlogy_safe(qv, max(ρ * qv * c.Rv * T, 1e-300) / c.ptr) +
           ql * c.cl * lnT + qi * (c.ci * lnT - c.ηi0)
end

"""
    budgets(s, z, Δx, Δz, c, compressible)

Per unit horizontal area (domain mean column integrals): total energy (J/m², liquid water at 0 K as
the origin; compressible: internal + kinetic + potential; anelastic: enthalpy + kinetic + potential on
the reference density), moist entropy (J/K/m²), total mass and total water (kg/m²).
"""
function budgets(s, z, Δx, Δz, c::Constants64, compressible)
    Nx, Nz = size(s.T)
    Lx = sum(Δx)
    E = 0.0; S = 0.0; M = 0.0; W = 0.0; KE = 0.0
    for k in 1:Nz, i in 1:Nx
        ρ, T, qv, ql, qi = s.ρ[i, k], s.T[i, k], s.qv[i, k], s.ql[i, k], s.qi[i, k]
        qd = 1 - qv - ql - qi
        m = ρ * Δx[i] * Δz[k] / Lx
        c_mix = compressible ? qd * (c.cpd - c.Rd) + qv * (c.cpv - c.Rv) + ql * c.cl + qi * c.ci :
                               qd * c.cpd + qv * c.cpv + ql * c.cl + qi * c.ci
        E += m * (c_mix * T + c.Lv0 * qv + (c.Lv0 - c.Li0) * qi + s.K[i, k] + c.g * z[k])
        KE += m * s.K[i, k]
        S += m * specific_entropy(T, ρ, qv, ql, qi, c)
        M += m
        W += m * (qv + ql + qi)
    end
    return (; E, S, M, W, KE)
end

dry_potential_temperature(s, c::Constants64) = s.T .* (c.pst ./ s.p) .^ (c.Rd / c.cpd)

# Linear interpolation of the columns of a (Nx × Nz) array to height zᵢ
function at_height(a, z, zᵢ)
    k = clamp(searchsortedlast(z, zᵢ), 1, length(z) - 1)
    f = (zᵢ - z[k]) / (z[k+1] - z[k])
    return (1 - f) .* a[:, k] .+ f .* a[:, k+1]
end

# A field from a host array on the grid's x–z centers
function center_field_from(grid, a)
    f = CenterField(grid)
    set!(f, reshape(a, size(a, 1), 1, size(a, 2)))
    return f
end

# RESULTS_TAG (e.g. "_prefix") keeps runs against another Breeze clone apart
results_directory(case) = (d = joinpath(@__DIR__, "results", case * get(ENV, "RESULTS_TAG", "")); mkpath(d); d)

parse_list(key, default) = split(get(ENV, key, default), ",")

wall_seconds(t₀) = 1e-9 * (time_ns() - t₀)
