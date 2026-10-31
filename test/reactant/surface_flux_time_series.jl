include(joinpath(dirname(@__DIR__), "setup.jl"))

#####
##### Reactant: per-column surface flux time series and total-water relaxation in a traced window
#####
#
# A column ensemble whose columns carry their own archived surface fluxes, as `FieldTimeSeries` of
# horizontal fields under the interface keys `ρE` and `ρqᵗ`, and whose total water is relaxed by a
# `Relaxation` under `qᵗ`. With `times` a range the record lookup is arithmetic inside the kernels, so
# a traced window, which reads the series at a traced clock, reproduces the eager one.

using Breeze
using Breeze.Microphysics: DCMIP2016KesslerMicrophysics
using Breeze.Thermodynamics: TetensFormula
using Breeze.TimeSteppers: prepare_first_time_step!
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: ColumnEnsembleSize
using Reactant
using Reactant: @trace
using Test
using CUDA

Reactant.set_default_backend(default_arch isa GPU ? "gpu" : "cpu")

const series_times = 0:600:3600

function flux_series(grid, values)
    fts = FieldTimeSeries{Center, Center, Nothing}(grid, series_times)
    for (n, t) in enumerate(series_times)
        set!(fts[n], reshape(values.(1:size(grid, 1), t), :, 1))
    end
    return fts
end

function flux_series_columns(arch)
    grid = RectilinearGrid(arch; size=ColumnEnsembleSize(Nz=8, ensemble=(3, 1), Hz=3),
                           z=(0, 1600), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants(saturation_vapor_pressure=TetensFormula())
    reference_state = ReferenceState(grid, constants; base_pressure=101500, potential_temperature=299)

    sensible(i, t) = 30 + 20i + 10 * sin(2π * t / 3600)         # W m⁻²
    evaporation(i, t) = (100 + 30i + 15 * cos(2π * t / 3600)) / 2.5e6 # kg m⁻² s⁻¹
    boundary_conditions = (; ρE=FieldBoundaryConditions(bottom=FluxBoundaryCondition(flux_series(grid, sensible))),
                             ρqᵗ=FieldBoundaryConditions(bottom=FluxBoundaryCondition(flux_series(grid, evaporation))))

    forcing = (; qᵗ=Relaxation(rate=1/1800, target=0.012))
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), thermodynamic_constants=constants,
                            microphysics=DCMIP2016KesslerMicrophysics(maximum_sedimentation_substeps=4),
                            advection=nothing, boundary_conditions, forcing)
    set!(model; θ = z -> 299 + 3e-3 * z, qᵛ = z -> 0.014 * exp(-z / 2000),
                qᶜˡ = z -> 5e-4 * exp(-((z - 800) / 200)^2))
    return model
end

function flux_series_window!(model, Δt, steps)
    prepare_first_time_step!(model, Δt)
    @trace mincut=true checkpointing=false track_numbers=false for _ in 1:steps
        time_step!(model, Δt)
    end
    return nothing
end

column_state(model) = map(f -> Array(interior(f)), (model.temperature, model.moisture_density,
                                                    model.microphysical_fields.ρqᶜˡ))

@testset "Surface flux time series and total-water relaxation under Reactant" begin
    Δt, steps = 75.0, 40 # 3000 s, across five records
    eager = flux_series_columns(CPU())
    for _ in 1:steps
        time_step!(eager, Δt)
    end

    model = flux_series_columns(ReactantState())
    @test model.moisture_density.boundary_conditions.bottom.condition.times isa AbstractRange
    Reactant.@jit flux_series_window!(model, Δt, steps)

    for (traced, expected) in zip(column_state(model), column_state(eager))
        @test traced ≈ expected rtol=1e-9
    end

    # Each column received its own fluxes
    ρqᵛ₁ = column_state(eager)[2][:, 1, 1]
    @test ρqᵛ₁[1] < ρqᵛ₁[2] < ρqᵛ₁[3]
end
