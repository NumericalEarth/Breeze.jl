using Breeze
using Breeze: ReferenceState, AnelasticDynamics, RelaxationForcing
using Oceananigans: Oceananigans, FieldTimeSeries, prognostic_fields
using Oceananigans.Fields: interior
using Oceananigans.Grids: znodes, Center
using Test

#####
##### Stub construction
#####

@testset "RelaxationForcing stub construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 8), x=(0, 100), y=(0, 100), z=(0, 3000))
    fts = FieldTimeSeries{Center, Center, Center}(grid, [FT(0), FT(3600)])

    nudging = RelaxationForcing(fts; time_scale=FT(3600))

    @test nudging.reference === fts
    @test nudging.time_scale == FT(3600)
    @test nudging.z_bottom == 1500
    @test isnothing(nudging.target)
    @test isnothing(nudging.clock)
    @test isnothing(nudging.density)
    @test isnothing(nudging.current_field)
    @test isnothing(nudging.reference_column)
end

#####
##### Profile mode materialization (1D FTS)
#####

@testset "RelaxationForcing profile mode materialization [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 3000))

    # 1D column FTS → profile mode automatically
    column_grid = RectilinearGrid(default_arch; size=(1, 1, 8), x=(0, 100), y=(0, 100), z=(0, 3000))
    fts = FieldTimeSeries{Center, Center, Center}(column_grid, [FT(0), FT(3600)])

    nudging = RelaxationForcing(fts; time_scale=FT(3600))
    model = AtmosphereModel(grid; forcing=(; ρθ=nudging))

    mat = model.forcing.ρθ
    @test mat isa RelaxationForcing
    @test !isnothing(mat.target)
    @test !isnothing(mat.current_field)
    @test mat.reference_column == (1, 1)   # profile mode
    @test mat.time_scale == FT(3600)
    @test mat.z_bottom == 1500
end

#####
##### 3D mode materialization
#####

@testset "RelaxationForcing 3D mode materialization [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 3000))
    fts = FieldTimeSeries{Center, Center, Center}(grid, [FT(0), FT(3600)])

    nudging = RelaxationForcing(fts; time_scale=FT(3600))
    model = AtmosphereModel(grid; forcing=(; ρθ=nudging))

    mat = model.forcing.ρθ
    @test mat isa RelaxationForcing
    @test isnothing(mat.reference_column)  # 3D mode
    @test !isnothing(mat.target)
    @test !isnothing(mat.clock)
end

#####
##### Analytical tendency test: profile mode
#####

@testset "RelaxationForcing profile tendency [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    Nz = 8
    grid = RectilinearGrid(default_arch; size=(4, 4, Nz), x=(0, 100), y=(0, 100), z=(0, 3000))
    reference_state = ReferenceState(grid)
    dynamics = AnelasticDynamics(reference_state)

    θ_ref = FT(300)
    θ_init = FT(310)
    τ = FT(3600)

    # Reference FTS: constant θ_ref at all levels and times
    column_grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 100), y=(0, 100), z=(0, 3000))
    fts = FieldTimeSeries{Center, Center, Center}(column_grid, [FT(0), FT(7200)])
    for n in 1:2
        parent(fts[n]) .= θ_ref
    end

    nudging = RelaxationForcing(fts; time_scale=τ, z_bottom=FT(0))
    model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature,
                            forcing=(; ρθ=nudging))

    θ₀ = model.dynamics.reference_state.potential_temperature
    set!(model, θ=θ_init)

    ρᵣ = interior(model.dynamics.reference_state.density) |> Array

    Gρθ_before = interior(model.timestepper.Gⁿ.ρθ) |> Array

    Δt = FT(1)
    time_step!(model, Δt)

    Gρθ = interior(model.timestepper.Gⁿ.ρθ) |> Array

    # Expected tendency: -ρ * (θ_init - θ_ref) / τ at every cell above z_bottom
    z = znodes(grid, Center())
    for k in 1:Nz
        expected = -ρᵣ[1, 1, k] * (θ_init - θ_ref) / τ
        @test Gρθ[1, 1, k] ≈ expected rtol=1e-3
    end
end

#####
##### z_bottom cutoff: no nudging below z_bottom
#####

@testset "RelaxationForcing z_bottom cutoff [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    Nz = 8
    z_bottom = FT(1500)
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 10), y=(0, 10), z=(0, 3000))
    reference_state = ReferenceState(grid)
    dynamics = AnelasticDynamics(reference_state)

    fts = FieldTimeSeries{Center, Center, Center}(grid, [FT(0), FT(7200)])
    for n in 1:2
        parent(fts[n]) .= FT(300)
    end

    nudging = RelaxationForcing(fts; time_scale=FT(3600), z_bottom=z_bottom)
    model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature,
                            forcing=(; ρθ=nudging))

    set!(model, θ=FT(310))
    Δt = FT(1)
    time_step!(model, Δt)

    Gρθ = interior(model.timestepper.Gⁿ.ρθ) |> Array
    z = znodes(grid, Center())

    for k in 1:Nz
        if z[k] < z_bottom
            @test Gρθ[1, 1, k] ≈ 0 atol=1e-10
        else
            @test Gρθ[1, 1, k] < 0   # negative: nudging θ=310 toward θ=300
        end
    end
end
