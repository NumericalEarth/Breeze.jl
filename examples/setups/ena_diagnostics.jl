using Statistics
using Printf
using Oceananigans.Grids: znodes
using Oceananigans.AbstractOperations: @at

struct ENADiagnostics{L, P, W, T, R}
    liquid_water_path :: L
    profiles :: P
    turbulence :: W
    times :: T
    records :: R
    directory :: String
end

function (diagnostics::ENADiagnostics)(simulation)
    time = simulation.model.clock.time
    compute!(diagnostics.liquid_water_path)
    liquid_water_path = only(Array(interior(diagnostics.liquid_water_path)))
    push!(diagnostics.times, Float64(time))
    push!(diagnostics.records, Float64(liquid_water_path))
    open(joinpath(diagnostics.directory, "timeseries.csv"), "a") do io
        println(io, time, ',', liquid_water_path)
    end
    for field in diagnostics.profiles
        compute!(field)
    end
    height = vec(Array(znodes(simulation.model.grid, Center(), Center(), Center(); with_halos=false)))
    profiles = map(field -> vec(Array(interior(field))), diagnostics.profiles)
    open(joinpath(diagnostics.directory, "profiles.csv"), "a") do io
        # Output already-reduced profiles, rather than iterating over LES cells.
        writedlm(io, hcat(fill(time, length(height)), height, profiles...), ',')
    end
    for field in diagnostics.turbulence
        compute!(field)
    end
    turbulence = map(field -> vec(Array(interior(field))), diagnostics.turbulence)
    open(joinpath(diagnostics.directory, "turbulence_profiles.csv"), "a") do io
        writedlm(io, hcat(fill(time, length(height)), height, turbulence...), ',')
    end
    heating = simulation.callbacks[:ena_drivers].func.radiative_heating
    open(joinpath(diagnostics.directory, "radiative_heating.csv"), "a") do io
        writedlm(io, hcat(fill(time, length(height)), height, vec(Array(interior(heating)))), ',')
    end
    return nothing
end

function add_ena_diagnostics!(simulation, directory; interval=60)
    mkpath(directory)
    write(joinpath(directory, "timeseries.csv"), "time_seconds,liquid_water_path_kg_m2\n")
    write(joinpath(directory, "profiles.csv"), "time_seconds,height_m,temperature_K,vapor_kg_kg,u_m_s,v_m_s\n")
    write(joinpath(directory, "turbulence_profiles.csv"),
          "time_seconds,height_m,mean_w_m_s,w_variance_m2_s2,w_temperature_covariance_K_m_s,w_vapor_covariance_m_s\n")
    write(joinpath(directory, "radiative_heating.csv"), "time_seconds,height_m,heating_W_m3\n")
    model = simulation.model
    μ = model.microphysical_fields
    liquid = μ.qᶜˡ + μ.qʳ + μ.qʷⁱ
    density = total_density(model.dynamics)
    column_water = Field(Integral(density * liquid; dims=3))
    path = Field(Average(column_water; dims=(1, 2)))
    variables = (T=model.temperature, q=specific_humidity(model), u=model.velocities.u, v=model.velocities.v)
    profiles = map(field -> Field(Average(field; dims=(1, 2))), variables)
    w = @at (Center, Center, Center) model.velocities.w
    mean_w = Field(Average(w; dims=(1, 2)))
    turbulence = (w=mean_w,
                  variance=Field(Average((w - mean_w)^2; dims=(1, 2))),
                  heat_flux=Field(Average((w - mean_w) * (model.temperature - profiles.T); dims=(1, 2))),
                  vapor_flux=Field(Average((w - mean_w) * (specific_humidity(model) - profiles.q); dims=(1, 2))))
    diagnostics = ENADiagnostics(path, profiles, turbulence, Float64[], Float64[], directory)
    simulation.callbacks[:ena_diagnostics] = Callback(diagnostics, TimeInterval(interval))
    return diagnostics
end
