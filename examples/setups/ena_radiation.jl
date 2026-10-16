# A CPU RRTMGP column supplies horizontally uniform radiative heating to the LES.
# This approximation retains feedback from the evolving mean cloud profile but
# omits horizontal cloud variability in the radiation calculation.
using NCDatasets
using RRTMGP
using Oceananigans.Grids: znodes

struct ENAMeanRadiation{M, F, H, T, Q}
    model :: M
    means :: F
    heating :: H
    initial_temperature :: T
    initial_moisture :: Q
end

function (radiation::ENAMeanRadiation)(simulation)
    drivers = simulation.callbacks[:ena_drivers].func
    model = radiation.model
    time = simulation.model.clock.time
    model.clock.time = time
    for field in radiation.means
        compute!(field)
    end
    temperature = copy(radiation.initial_temperature)
    moisture = copy(radiation.initial_moisture)
    count = simulation.model.grid.Nz
    temperature[1:count] .= vec(Array(interior(radiation.means.θ)))
    moisture[1:count] .= vec(Array(interior(radiation.means.q)))
    surface = drivers.surface.temperature
    # Read the surface table on the host; never scalar-index a Metal array.
    surface_host = TimeHeightProfile(Array(surface.heights), Array(surface.times), Array(surface.values))
    set!(model.radiation.surface_radiation.surface_temperature, surface_host(0, time))
    set!(model; θ=reshape(temperature, 1, 1, :), qᵗ=reshape(moisture, 1, 1, :))
    heating = Array(interior(model.radiation.flux_divergence))[:, :, 1:count]
    set!(radiation.heating, heating)
    return nothing
end

function add_ena_radiation!(simulation; directory=download_covert_ena(), interval=60)
    model = simulation.model
    FT = eltype(model.grid)
    sounding = sam_sounding(joinpath(directory, "snd"))
    faces = vec(Array(znodes(model.grid, Center(), Center(), Face(); with_halos=false)))
    # Continue the sounding above the LES lid so that its radiative boundary is
    # not incorrectly treated as the top of the atmosphere.
    top = min(25000, last(sounding.height))
    upper_faces = range(last(faces), top; length=41)[2:end]
    radiation_faces = vcat(faces, upper_faces)
    grid = RectilinearGrid(CPU(), FT; size=length(radiation_faces)-1,
                           x=-28.03, y=39.09, z=radiation_faces,
                           topology=(Flat, Flat, Bounded))
    constants = model.thermodynamic_constants
    heights = Tuple(FT.(sounding.height))
    temperatures = Tuple(FT.(sounding.θ))
    reference_temperature(z) = ena_initial_profile(z, heights, temperatures)
    reference_state = ReferenceState(grid, constants; base_pressure=sounding.surface_pressure,
                                     potential_temperature=reference_temperature)
    surface_temperature = Field{Nothing, Nothing, Nothing}(grid)
    set!(surface_temperature, FT(294.937))
    radiation = RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                       surface_temperature, surface_albedo=0.06,
                                       surface_emissivity=0.98,
                                       solar_position=ApparentSolarPosition(coordinate=(-28.03, 39.09),
                                                                          epoch=DateTime(2017, 7, 18, 6)),
                                       liquid_effective_radius=ConstantRadiusParticles(10e-6))
    column = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), radiation,
                             microphysics=SaturationAdjustment(equilibrium=WarmPhaseEquilibrium()))
    height = vec(Array(znodes(grid, Center(), Center(), Center(); with_halos=false)))
    initial_temperature = FT.(linear_profile.(Ref(sounding.height), Ref(sounding.θ), height))
    initial_moisture = FT.(linear_profile.(Ref(sounding.height), Ref(sounding.q), height))
    θ = liquid_ice_potential_temperature(model)
    # P3's primary moisture prognostic is vapor; add each independent condensate
    # mass once. Rime mass is already included in qⁱ and must not be added again.
    μ = model.microphysical_fields
    q = μ.qᵛ + μ.qᶜˡ + μ.qʳ + μ.qⁱ + μ.qʷⁱ
    means = (θ=Field(Average(θ; dims=(1, 2))), q=Field(Average(q; dims=(1, 2))))
    heating = simulation.callbacks[:ena_drivers].func.radiative_heating
    callback = ENAMeanRadiation(column, means, heating, initial_temperature, initial_moisture)
    simulation.callbacks[:ena_radiation] = Callback(callback, TimeInterval(interval))
    simulation.callbacks[:ena_drivers].func.initialized[] && callback(simulation)
    return simulation
end
