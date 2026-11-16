# The prognostic state after ten steps of a moist thermal bubble, for comparing branches bitwise.
#
#   julia --project=<env> validation/mis_memory/ten_step_state.jl write <file>     # save the states
#   julia --project=<env> validation/mis_memory/ten_step_state.jl compare <file>   # compare with saved ones
#
# Cases: dry, one-moment and P3 microphysics, Float32 and Float64, for AcousticRungeKutta3 and every
# coefficient set of MultirateInfinitesimalStep, with pre-loop and post-loop scalar tendencies,
# stochastic rounding, and a vertically-implicit closure.

using Breeze
using Oceananigans
using Oceananigans: prognostic_fields
using CloudMicrophysics: CloudMicrophysics
using Serialization

const CMExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
const mode, file = ARGS[1], ARGS[2]

function microphysics_scheme(case, FT)
    case === :dry && return nothing
    case === :P3 && return PredictedParticlePropertiesMicrophysics(FT)
    categories = CMExt.one_moment_cloud_microphysics_categories(FT)
    cloud_formation = NonEquilibriumCloudFormation(nothing, categories.parameters.cloud.ice)
    return CMExt.OneMomentCloudMicrophysics(FT; cloud_formation, categories)
end

function ten_step_state(FT, case, timestepper, timestepper_kwargs; closure = nothing, steps = 10)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(CPU(); size = (12, 8, 12), halo = (5, 5, 5),
                           x = (-6e3, 6e3), y = (-4e3, 4e3), z = (0, 6e3),
                           topology = (Periodic, Periodic, Bounded))
    θ₀ = 300
    p₀ = 1e5
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                    base_pressure = p₀, standard_pressure = p₀,
                                    reference_potential_temperature = θ₀)
    model = AtmosphereModel(grid; dynamics, microphysics = microphysics_scheme(case, FT), closure,
                            advection = WENO(order=5), timestepper, timestepper_kwargs)
    constants = model.thermodynamic_constants
    r(x, y, z) = min(1, sqrt((x / 2e3)^2 + (y / 2e3)^2 + ((z - 2e3) / 2e3)^2))
    θᵢ(x, y, z) = θ₀ + 2 * cos(π * r(x, y, z) / 2)^2
    ρᵢ(x, y, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, p₀, constants)
    qᵛᵢ(x, y, z) = 0.004 + 0.012 * cos(π * r(x, y, z) / 2)^2
    qᶜˡᵢ(x, y, z) = 0.002 * cos(π * r(x, y, z) / 2)^2
    uᵢ(x, y, z) = 5 + z / 1e3
    if case === :dry
        set!(model; ρ = ρᵢ, θ = θᵢ, qᵗ = qᵛᵢ, u = uᵢ)
    else
        set!(model; ρ = ρᵢ, θ = θᵢ, qᵛ = qᵛᵢ, qᶜˡ = qᶜˡᵢ, u = uᵢ)
    end

    for _ in 1:steps
        time_step!(model, 4)
    end

    return NamedTuple{keys(prognostic_fields(model))}(map(f -> Array(parent(f)), values(prognostic_fields(model))))
end

mis(coefficients; kwargs...) = (:MultirateInfinitesimalStep, (; coefficients, kwargs...))

configurations = []
for FT in (Float32, Float64), case in (:dry, :OneMoment, :P3)
    push!(configurations, (FT, case, :AcousticRungeKutta3, NamedTuple(), nothing))
    FT === Float32 && push!(configurations, (FT, case, :AcousticRungeKutta3, (; post_loop_scalar_tendencies = true), nothing))
    for coefficients in (:SSPMIS43, :MIS4a, :TVDMISB, :WickerSkamarock3)
        push!(configurations, (FT, case, mis(coefficients)..., nothing))
    end
    if FT === Float32
        push!(configurations, (FT, case, mis(:SSPMIS43; post_loop_scalar_tendencies = false)..., nothing))
        push!(configurations, (FT, case, mis(:MIS4a; post_loop_scalar_tendencies = false)..., nothing))
        push!(configurations, (FT, case, mis(:SSPMIS43; rounding = StochasticRounding(seed = 3))..., nothing))
        push!(configurations, (FT, case, mis(:TVDMISB; rounding = StochasticRounding(seed = 3))..., nothing))
        push!(configurations, (FT, case, mis(:SSPMIS43; relinearize_stages = true)..., nothing))
        if case !== :P3
            implicit_closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), FT; ν = 5, κ = 5)
            push!(configurations, (FT, case, mis(:SSPMIS43)..., implicit_closure))
            push!(configurations, (FT, case, mis(:MIS4a)..., implicit_closure))
        end
    end
end

label(c) = string(c[1], " ", c[2], " ", c[3], " ", c[4], isnothing(c[5]) ? "" : " implicit closure")

states = Dict{String, Any}()
for configuration in configurations
    FT, case, timestepper, timestepper_kwargs, closure = configuration
    state = try
        ten_step_state(FT, case, timestepper, timestepper_kwargs; closure)
    catch err
        @warn "failed: $(label(configuration))" exception = err
        nothing
    end
    states[label(configuration)] = state
    isnothing(state) || println(label(configuration), ": finite = ", all(a -> all(isfinite, a), state),
                                ", max|ρw| = ", maximum(abs, state.ρw))
end

function compare(states, reference)
    identical = 0
    for name in sort(collect(keys(states)))
        state, expected = states[name], get(reference, name, nothing)
        if isnothing(state) || isnothing(expected)
            println("MISSING    ", name)
            continue
        end
        differing = [k for k in keys(state) if !isequal(reinterpret(UInt8, state[k]), reinterpret(UInt8, expected[k]))]
        if isempty(differing)
            identical += 1
            println("identical  ", name)
        else
            worst = maximum(maximum(abs, Float64.(state[k]) .- Float64.(expected[k])) for k in differing)
            println("DIFFERENT  ", name, ": ", differing, ", max |difference| = ", worst)
        end
    end
    println("bitwise identical: ", identical, " of ", length(states))
    return nothing
end

if mode == "write"
    serialize(file, states)
    println("wrote ", length(states), " states to ", file)
else
    compare(states, deserialize(file))
end
