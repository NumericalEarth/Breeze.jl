# Common forward LES configuration for forcing comparisons at ENA.
# This file only defines constructors; including it never launches a forecast.
using Breeze
using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: znode, znodes
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ
using Breeze.Thermodynamics: LiquidIcePotentialTemperatureState, MoistureMassFractions
using Breeze.Advection: ScaledWENOWeights

include("ena_data.jl")

struct ENADrivers{P, S, F, R, H, I}
    profiles :: P
    surface :: S
    fields :: F
    radiative_heating :: R
    prescribed_radiation :: H
    initialized :: I
end

ena_update_prescribed_radiation!(field, ::Nothing, time) = nothing
ena_update_prescribed_radiation!(field, profile::TimeHeightProfile, time) = set!(field, profile, time)

@inline function ena_initial_profile(z, heights, values)
    k, fraction = Breeze.Forcings.profile_bracket(heights, z)
    @inbounds return (1 - fraction) * values[k] + fraction * values[k+1]
end

function (drivers::ENADrivers)(simulation)
    drivers.initialized[] || error("Load an ENACase with set!(simulation, case) before running.")
    time = simulation.model.clock.time
    for name in keys(drivers.fields)
        profile = drivers.profiles[name]
        set!(drivers.fields[name], profile, time)
    end
    ena_update_prescribed_radiation!(drivers.radiative_heating, drivers.prescribed_radiation, time)
    return nothing
end

@inline function ena_temperature_forcing(i, j, k, grid, clock, fields, parameters)
    z = znode(i, j, k, grid, Center(), Center(), Center())
    @inbounds pressure = parameters.pressure[i, j, k]
    # SAM tls is a temperature tendency, not a potential-temperature tendency.
    return parameters.profile(z, clock.time) / (pressure / 100000) ^ parameters.κ
end

@inline ena_surface_flux(x, y, time, profile) = profile(0, time)
@inline ena_discrete_surface_flux(i, j, grid, clock, fields, profile) = profile(0, clock.time)

@inline function ena_surface_stress(i, j, grid, clock, fields, parameters)
    # Interpolate the cross component onto the forced momentum face. Named
    # accesses remain type-stable with P3's large field tuple on Metal.
    # TODO Oceananigans: specialize continuous-BC dependency indices at compile
    # time; runtime NamedTuple indexing can retain get_nth_field_checked here.
    @inbounds begin
        u = ifelse(parameters.component == 1, fields.u[i, j, 1], ℑxyᶜᶠᵃ(i, j, 1, grid, fields.u))
        v = ifelse(parameters.component == 1, ℑxyᶠᶜᵃ(i, j, 1, grid, fields.v), fields.v[i, j, 1])
    end
    speed = max(sqrt(u*u + v*v), 1f-6)
    velocity = ifelse(parameters.component == 1, u, v)
    return -parameters.density * parameters.stress(0, clock.time) * velocity / speed
end

function ena_profile_buffers(profiles, arch, FT, times)
    return map(profiles) do profile
        TimeHeightProfile(FT.(profile.heights), FT.(times),
                          zeros(FT, length(profile.heights), length(times)); architecture=arch)
    end
end

# The initial sounding is deliberately shared between runs: set!(simulation, case)
# changes only external forcing arrays. The default radiation is a CPU mean
# column coupled to the LES. A factory taking (grid, constants, surface_temperature_profile)
# can instead supply native radiation. A TimeHeightProfile prescribes volumetric
# radiative heating in W/m³, held independent of the evolving cloud state.
# radiation=nothing disables radiation for short smoke tests.
function ena_simulation(arch=CPU(); size=(64, 64, 75), extent=(6400, 6400, 3000),
                        Δt=1, stop_time=9*3600, radiation=:mean_column,
                        directory=download_covert_ena())
    FT = Oceananigans.defaults.FloatType
    grid = RectilinearGrid(arch, FT; size, x=(0, extent[1]), y=(0, extent[2]), z=(0, extent[3]), halo=(5, 5, 5),
                           topology=(Periodic, Periodic, Bounded))
    data = covert_ena_case(directory)
    sounding = sam_sounding(joinpath(directory, "snd"))
    constants = ThermodynamicConstants(FT)
    heights = Tuple(FT.(sounding.height))
    temperatures = Tuple(FT.(sounding.θ))
    reference_temperature(z) = ena_initial_profile(z, heights, temperatures)
    reference_state = ReferenceState(grid, constants;
                                     base_pressure=sounding.surface_pressure,
                                     potential_temperature=reference_temperature)
    dynamics = AnelasticDynamics(reference_state)
    times = sort!(unique(vcat(collect(0:3600:stop_time), [stop_time])))
    profiles = ena_profile_buffers(data.profiles, arch, FT, times)
    surface = ena_profile_buffers(data.surface, arch, FT, times)
    fields = (u=Field{Nothing, Nothing, Center}(grid),
              v=Field{Nothing, Nothing, Center}(grid),
              subsidence=Field{Nothing, Nothing, Face}(grid))
    radiative_heating = Field{Nothing, Nothing, Center}(grid)
    prescribed_radiation = if radiation isa TimeHeightProfile
        radiation_times = FT.(Array(radiation.times))
        first(radiation_times) ≤ 0 && last(radiation_times) ≥ stop_time ||
            throw(ArgumentError("Prescribed radiation must cover the forecast interval."))
        TimeHeightProfile(FT.(Array(radiation.heights)), radiation_times,
                          FT.(Array(radiation.values)); architecture=arch)
    else
        nothing
    end
    subsidence = SubsidenceForcing(fields.subsidence)
    geostrophic = geostrophic_forcings(fields.u, fields.v)
    κ = FT(dry_air_gas_constant(constants) / constants.dry_air.heat_capacity)
    thermal = Forcing(ena_temperature_forcing; discrete_form=true,
                       parameters=(profile=profiles.temperature_tendency, pressure=reference_state.pressure, κ))
    # P3 stores vapor and condensates separately; subsidence transports all
    # specific mass, number, and rime-volume fields with the same velocity.
    forcing = (θ=(subsidence, thermal), qᵗ=(subsidence, Forcing(profiles.moisture_tendency)),
               qᶜˡ=subsidence, qʳ=subsidence, nʳ=subsidence,
               qⁱ=subsidence, nⁱ=subsidence, qᶠ=subsidence, bᶠ=subsidence, qʷⁱ=subsidence,
               u=(subsidence, geostrophic.u), v=(subsidence, geostrophic.v),
               ρE=Forcing(radiative_heating),
               w=Relaxation(rate=FT(1/30), mask=GaussianMask{:z}(center=extent[3], width=500)))

    density = FT(surface_density(reference_state))
    stress_parameters(component) = (; density, component, stress=surface.kinematic_stress)
    stress(component) = FluxBoundaryCondition(ena_surface_stress; discrete_form=true,
                                              parameters=stress_parameters(component))
    vapor_flux = TimeHeightProfile(FT.(data.surface.latent_heat_flux.heights),
                                   FT.(times),
                                   zeros(FT, 2, length(times)); architecture=arch)
    # Keep the converted latent-heat flux in the same storage as the callback's
    # surface array; set!(simulation, case) performs the W/m² → kg/m²/s conversion.
    surface = merge(surface, (; vapor_flux))
    drivers = ENADrivers(profiles, surface, fields, radiative_heating, prescribed_radiation, Ref(false))
    boundary_conditions = (
        ρE=FieldBoundaryConditions(bottom=FluxBoundaryCondition(ena_discrete_surface_flux; discrete_form=true, parameters=surface.sensible_heat_flux)),
        ρqᵗ=FieldBoundaryConditions(bottom=FluxBoundaryCondition(ena_surface_flux; parameters=vapor_flux)),
        ρu=FieldBoundaryConditions(bottom=stress(1)),
        ρv=FieldBoundaryConditions(bottom=stress(2)))
    microphysics = P3Microphysics(FT; thermodynamic_constants=constants)
    weno = WENO(order=5)
    positive = WENO(order=5, bounds=(0, 1))
    positive_number = WENO(order=5, bounds=(FT(0), FT(Inf)), weight_computation=ScaledWENOWeights)
    scalar_advection = (ρθ=weno, ρqᵛ=positive, ρqᶜˡ=positive, ρqʳ=positive,
                        ρnʳ=positive_number, ρqⁱ=positive, ρnⁱ=positive_number,
                        ρqᶠ=positive, ρbᶠ=positive_number, ρqʷⁱ=positive)
    column_radiation = radiation === :mean_column
    radiation_model = isnothing(radiation) || column_radiation || radiation isa TimeHeightProfile ?
                      nothing : radiation(grid, constants, surface.temperature)
    model = Base.inferencebarrier(AtmosphereModel(grid; dynamics, microphysics, forcing, boundary_conditions,
                            radiation=radiation_model, coriolis=FPlane(latitude=39.09),
                            closure=SmagorinskyLilly(), momentum_advection=weno, scalar_advection))
    initial_profiles = map((sounding.θ, sounding.q, sounding.u, sounding.v)) do values
        TimeHeightProfile(FT.(sounding.height), FT[0, stop_time], FT.(hcat(values, values)))
    end
    θ, q, u, v = initial_profiles
    Lx, Ly, Lz = FT.(extent)
    perturbation(x, y, z) = FT(0.01) * sinpi(2x/Lx) * cospi(2y/Ly) * ifelse(z < 1000, sinpi(z/1000), 0)
    initial_θ(x, y, z) = θ(z, 0) + perturbation(x, y, z)
    # Partition the common initial total water into vapor and cloud liquid at
    # fixed θ and pressure. P3 subsequently evolves that cloud prognostically.
    height = vec(Array(znodes(grid, Center(), Center(), Center(); with_halos=false)))
    pressure = vec(Array(interior(reference_state.pressure)))
    pressure_profile = TimeHeightProfile(height, FT[0, stop_time], hcat(pressure, pressure))
    adjustment = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    standard_pressure = reference_state.standard_pressure
    function initial_cloud(x, y, z)
        state = LiquidIcePotentialTemperatureState(initial_θ(x, y, z), MoistureMassFractions(q(z, 0)),
                                                   standard_pressure, pressure_profile(z, 0))
        adjusted = Breeze.Microphysics.adjust_thermodynamic_state(state, adjustment, constants)
        return adjusted.moisture_mass_fractions.liquid
    end
    set!(model; θ=initial_θ, qᵗ=(x,y,z) -> q(z, 0), qᶜˡ=initial_cloud,
         u=(x,y,z) -> u(z, 0), v=(x,y,z) -> v(z, 0))
    simulation = Simulation(model; Δt, stop_time)
    simulation.callbacks[:ena_drivers] = Callback(drivers, IterationInterval(1))
    Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)
    column_radiation && add_ena_radiation!(simulation; directory)
    return Base.inferencebarrier(simulation)
end

function Oceananigans.set!(simulation::Simulation, data::ENACase)
    data.date == Date(2017, 7, 18) || throw(ArgumentError("All runs must use 18 July 2017."))
    simulation.model.clock.iteration == 0 || throw(ArgumentError("Construct a fresh simulation before changing cases."))
    drivers = simulation.callbacks[:ena_drivers].func
    # Validate every input before changing any buffer.
    for (buffers, inputs) in ((drivers.profiles, data.profiles), (drivers.surface, data.surface))
        for name in keys(inputs)
            profile = inputs[name]
            first(profile.times) ≤ 0 && last(profile.times) ≥ simulation.stop_time ||
                throw(ArgumentError("$name does not cover the requested forecast interval."))
            haskey(buffers, name) || throw(ArgumentError("Unknown ENA forcing $name."))
        end
    end
    keys(data.profiles) == keys(drivers.profiles) || throw(ArgumentError("Incomplete atmospheric forcing."))
    keys(data.surface) == (:temperature, :sensible_heat_flux, :latent_heat_flux, :kinematic_stress) ||
        throw(ArgumentError("Incomplete surface forcing."))
    for (buffers, inputs) in ((drivers.profiles, data.profiles), (drivers.surface, data.surface))
        for name in keys(inputs)
            buffer = buffers[name]
            values = [inputs[name](z, t) for z in Array(buffer.heights), t in Array(buffer.times)]
            copyto!(buffer.values, eltype(buffer.values).(values))
        end
    end
    # The reference latent heat follows Breeze's thermodynamic constants.
    latent_heat = simulation.model.thermodynamic_constants.liquid.reference_latent_heat
    drivers.surface.vapor_flux.values .= drivers.surface.latent_heat_flux.values ./ latent_heat
    drivers.initialized[] = true
    drivers(simulation)
    if haskey(simulation.callbacks, :ena_radiation)
        simulation.callbacks[:ena_radiation].func(simulation)
    end
    return simulation
end

include("ena_radiation.jl")
