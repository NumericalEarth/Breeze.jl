# Helpers shared by the tests of moist `HydrostaticDynamics`. Not a test: removed from the test
# suite in `runtests.jl`.

using Breeze
using Oceananigans
using Oceananigans.Advection: VectorInvariant
using CloudMicrophysics: CloudMicrophysics

using Breeze.Thermodynamics: MoistureMassFractions, mixture_heat_capacity, condensate_latent_energy

# Column states of the moist derivation: one row per layer, the lowest first. `r_e` (vapor and
# cloud liquid) and `r_r` (rain) are mixing ratios per unit dry mass, `theta` the liquid-ice
# potential temperature with the dry Exner exponent.
function read_column(path)
    lines = filter(line -> !startswith(line, "#"), readlines(path))
    names = Symbol.(split(first(lines), ","))
    rows = [parse.(Float64, split(line, ",")) for line in lines[2:end]]
    column = NamedTuple{Tuple(names)}(Tuple([row[n] for row in rows] for n in eachindex(names)))
    header = Dict{String, Float64}()
    for line in filter(line -> startswith(line, "#") && occursin(" = ", line), readlines(path))
        key, value = split(strip(line[2:end]), " = "; limit=2)
        number = tryparse(Float64, first(split(value, " ;")))
        isnothing(number) || (header[key] = number)
    end
    return column, header
end

const BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

# The budgets below are measured from the diagnosed temperature, so the saturation adjustment is
# solved to rounding: with the default tolerance of 1e-4 K the temperature, and with it the
# measured enthalpy, is only that accurate, although the enthalpy the model carries is exact.
warm_rain_microphysics(FT=Float64; kw...) =
    OneMomentCloudMicrophysics(FT; cloud_formation = SaturationAdjustment(FT; equilibrium = WarmPhaseEquilibrium(),
                                                                          solver = SecantSolver(FT; abstol=1e-11, maxiter=50)),
                               kw...)

# A single column of moist `HydrostaticDynamics` with the hybrid coefficients of a column state
function moist_column_model(column; Φˢ=0, microphysics=warm_rain_microphysics(), timestepper=:SSPRungeKutta3)
    Nz = length(column.k)
    grid = RectilinearGrid(CPU(), Float64; size=Nz, x=0, y=0, z=(0, 1), topology=(Flat, Flat, Bounded))
    A = vcat(column.A_lower, column.A_upper[end])
    B = vcat(column.B_lower, column.B_upper[end])
    dynamics = HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B, surface_geopotential=Φˢ)
    return AtmosphereModel(grid; dynamics, microphysics, timestepper,
                           formulation = moist_formulation(),
                           momentum_advection = VectorInvariant())
end

# The surface pressure and the mass fractions of the total-mass coordinate for a column state of
# the derivation, given by its dry-mass surface pressure πˢ and its mixing ratios per unit dry
# mass: the surface pressure is the weight of the dry air and of all the water, pˢ = πˢ + Σ rᵗ Δπ,
# and the mass fractions are q = r / (1 + rᵗ). The layer masses of the two coordinates agree in
# the column sum only, so the state is the same column up to the distribution of its mass.
function total_column_state(model, πˢ, rᵉ, rʳ)
    A = Array(interior(model.dynamics.pressure_coefficient))[:]
    B = Array(interior(model.dynamics.surface_pressure_coefficient))[:]
    Nz = size(model.grid, 3)
    rᵗ = rᵉ .+ rʳ
    Δπ = [(A[k] - A[k+1]) + (B[k] - B[k+1]) * πˢ for k in 1:Nz]
    return (pˢ = πˢ + sum(rᵗ .* Δπ), qᵉ = rᵉ ./ (1 .+ rᵗ), qʳ = rʳ ./ (1 .+ rᵗ))
end

# Set the prognostic state of a column from its total surface pressure, its potential temperature
# and its mass fractions, which are what the model carries: no pressure enters, so nothing is iterated.
function set_total_column_state!(model, pˢ, θ, qᵉ, qʳ)
    set!(model.dynamics.surface_pressure, pˢ)
    set!(model)
    ρ = Array(interior(model.dynamics.density))
    shape(a) = reshape(a, 1, 1, size(model.grid, 3))
    set!(model; ρθ = ρ .* shape(θ), ρqᵉ = ρ .* shape(qᵉ), ρqʳ = ρ .* shape(qʳ))
    return nothing
end

# Set the prognostic state of a column from the dry-mass surface pressure and the mixing ratios
# per unit dry mass of a column state of the derivation (`total_column_state`)
function set_column_state!(model, πˢ, θ, rᵉ, rʳ)
    state = total_column_state(model, πˢ, rᵉ, rʳ)
    set_total_column_state!(model, state.pˢ, θ, state.qᵉ, state.qʳ)
    return nothing
end

moist_formulation() = LiquidIcePotentialTemperatureFormulation(exner_exponent=DryExponent())

# The column diagnostics below read fields on the host: each field's interior is copied once to an
# Array and the vertical spacings are taken from a CPU copy of the grid, so that they also work on a
# GPU without scalar indexing of device arrays. On the CPU the values are those of indexing the
# fields directly (same values, same summation order).
host(field) = Array(interior(field))
host_grid(grid) = Oceananigans.Architectures.on_architecture(CPU(), grid)
# A phase that the scheme does not have is a `ZeroField`: zeros of the shape of `like`
host_or_zero(field, like) = field isa Oceananigans.Fields.ZeroField ? zero(like) : host(field)

# Enthalpy of the column plus the geopotential of its mass at the ground, per unit area in
# J m⁻²: Σ Δp (cᵖᵐ T - ℒˡ₀ qˡ - ℒⁱ₀ qⁱ) / g + Φˢ pˢ / g, from the diagnosed temperature and composition
# rather than from the potential temperature that the model carries.
function column_energy(model)
    grid = host_grid(model.grid)
    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    g = constants.gravitational_acceleration
    ρᵗ = host(dynamics.moisture.total_density)
    water = Breeze.HydrostaticEquations.hydrostatic_water_fields(model.microphysics, model.microphysical_fields)
    qᵛ = host(water.qᵛ)
    qˡ = host_or_zero(water.qˡ, qᵛ)
    qⁱ = host_or_zero(water.qⁱ, qᵛ)
    T = host(model.temperature)
    Φˢ = host(dynamics.surface_geopotential)
    pˢ = host(dynamics.moisture.total_surface_pressure)
    Nx, Ny, Nz = size(grid)
    energy = zeros(Nx, Ny)

    for i in 1:Nx, j in 1:Ny
        for k in 1:Nz
            q = MoistureMassFractions(qᵛ[i, j, k], qˡ[i, j, k], qⁱ[i, j, k])
            H = mixture_heat_capacity(q, constants) * T[i, j, k] - condensate_latent_energy(q, constants)
            energy[i, j] += ρᵗ[i, j, k] * Oceananigans.Operators.Δzᶜᶜᶜ(i, j, k, grid) * H
        end
        energy[i, j] += Φˢ[i, j, 1] * pˢ[i, j, 1] / g
    end

    return energy
end

# Water of the column per unit area, in kg m⁻²
function column_water(model)
    grid = host_grid(model.grid)
    Nx, Ny, Nz = size(grid)
    # Every prognostic water density: the moisture density and those of the scheme
    ρq = map(host, Breeze.HydrostaticEquations.water_densities(model))
    return [sum(sum(ρqˣ[i, j, k] for ρqˣ in ρq) * Oceananigans.Operators.Δzᶜᶜᶜ(i, j, k, grid) for k in 1:Nz)
            for i in 1:Nx, j in 1:Ny]
end

# The dry mass of a column is the total mass of the coordinate less its water, a difference of
# sums of Nz layers: it is kept to the rounding of those sums, relative to the column mass.
dry_mass_rounding(model) = 4 * size(model.grid, 3) * eps(eltype(model.grid))

# Dry mass of the column per unit area, in kg m⁻²: the total mass less the water
function column_dry_mass(model)
    grid = host_grid(model.grid)
    Nx, Ny, Nz = size(grid)
    ρ = host(model.dynamics.density)
    water = map(host, Breeze.HydrostaticEquations.water_densities(model))
    return [sum((ρ[i, j, k] - sum(ρq[i, j, k] for ρq in water)) * Oceananigans.Operators.Δzᶜᶜᶜ(i, j, k, grid) for k in 1:Nz)
            for i in 1:Nx, j in 1:Ny]
end

# A slice of moist or dry `HydrostaticDynamics`, periodic in x, in pure sigma coordinates
function slice_model(; size=(16, 10), extent=1.6e6, Φˢ=0, top_pressure=2.5e4, microphysics=warm_rain_microphysics(),
                       tracers=(), timestepper=:SSPRungeKutta3, kw...)
    grid = RectilinearGrid(CPU(), Float64; size, x=(0, extent), z=(0, 1), topology=(Periodic, Flat, Bounded))
    dynamics = HydrostaticDynamics(; surface_geopotential=Φˢ, top_pressure)
    return AtmosphereModel(grid; dynamics, microphysics, tracers, timestepper,
                           formulation = moist_formulation(),
                           momentum_advection = VectorInvariant(), kw...)
end

max_abs(field) = maximum(abs, Array(interior(field)))
