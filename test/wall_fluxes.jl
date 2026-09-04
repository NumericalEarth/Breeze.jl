include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.BoundaryConditions: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux, FilteredSurfaceVelocities
using Breeze.Thermodynamics: surface_density, saturation_specific_humidity, PlanarLiquidSurface,
                             potential_temperature_from_temperature
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.BoundaryConditions: getbc
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.Models: boundary_condition_args
using Test

#####
##### Bulk fluxes on the six walls of a closed box
#####
##### Every wall carries drag on its two tangential momentum components, a sensible heat
##### flux, and a vapor flux. The wall temperatures and humidities differ from wall to
##### wall so that the sign and the near-wall state of each flux are unambiguous.
#####

const sides = (:west, :east, :south, :north, :bottom, :top)

# Fluxes point along the positive coordinate direction: a flux out of the domain is
# negative on a left wall and positive on a right wall.
outward_sign(side) = side ∈ (:west, :south, :bottom) ? -1 : 1

# Oceananigans evaluates a boundary condition with the two indices tangential to it.
# Return those indices and the near-wall cell for a probe at (i, j, k).
tangential_indices(side, i, j, k) = side ∈ (:bottom, :top) ? (i, j) : side ∈ (:west, :east) ? (j, k) : (i, k)
near_wall_cell(side, i, j, k, Nx, Ny, Nz) = (side === :west   ? (1, j, k) :
                                             side === :east   ? (Nx, j, k) :
                                             side === :south  ? (i, 1, k) :
                                             side === :north  ? (i, Ny, k) :
                                             side === :bottom ? (i, j, 1) : (i, j, Nz))

wall_temperatures(FT) = (west=FT(295), east=FT(285), south=FT(287), north=FT(288), bottom=FT(299), top=FT(280))
wall_humidities(FT)   = (west=FT(0.78), east=FT(0.78), south=FT(0.78), north=FT(0.78), bottom=FT(1), top=FT(1))

function closed_box_model(FT; formulation=:LiquidIcePotentialTemperature, coefficient=FT(2e-3), Nx=4, Ny=4, Nz=4)
    grid = RectilinearGrid(default_arch, FT; size=(Nx, Ny, Nz), x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Bounded, Bounded, Bounded))

    T₀ = wall_temperatures(FT)
    ℋ₀ = wall_humidities(FT)
    drag(side) = BulkDrag(; coefficient, surface_temperature=T₀[side])
    heat(side) = BulkSensibleHeatFlux(; coefficient, surface_temperature=T₀[side])
    vapor(side) = BulkVaporFlux(; coefficient, surface_temperature=T₀[side], surface_relative_humidity=ℋ₀[side])

    # Drag acts on the two momentum components tangential to each wall
    ρu_bcs = FieldBoundaryConditions(south=drag(:south), north=drag(:north), bottom=drag(:bottom), top=drag(:top))
    ρv_bcs = FieldBoundaryConditions(west=drag(:west), east=drag(:east), bottom=drag(:bottom), top=drag(:top))
    ρw_bcs = FieldBoundaryConditions(west=drag(:west), east=drag(:east), south=drag(:south), north=drag(:north))
    heat_bcs = FieldBoundaryConditions(; (side => heat(side) for side in sides)...)
    ρqᵛ_bcs = FieldBoundaryConditions(; (side => vapor(side) for side in sides)...)

    thermodynamic_name = formulation === :StaticEnergy ? :ρs : :ρθ
    names = (:ρu, :ρv, :ρw, thermodynamic_name, :ρqᵛ)
    boundary_conditions = NamedTuple{names}((ρu_bcs, ρv_bcs, ρw_bcs, heat_bcs, ρqᵛ_bcs))

    return AtmosphereModel(grid; formulation, boundary_conditions, advection=WENO(order=3))
end

momentum_field(model, side, name) = getproperty(model.momentum, name)
wall_bc(field, side) = getproperty(field.boundary_conditions, side)

# Evaluate a materialized boundary condition the way Oceananigans does, at the probe (i, j, k)
function evaluate_bc(model, field, side, i, j, k)
    bc = wall_bc(field, side)
    a, b = tangential_indices(side, i, j, k)
    clock, fields = boundary_condition_args(model)
    return @allowscalar getbc(bc, a, b, model.grid, clock, fields)
end

@testset "Bulk fluxes on the six walls [$(FT)]" for FT in test_float_types()
    old_FT = Oceananigans.defaults.FloatType
    Oceananigans.defaults.FloatType = FT

    T₀ = wall_temperatures(FT)
    ℋ₀ = wall_humidities(FT)
    C = FT(2e-3)
    θᵢ, ℋᵢ = FT(290), FT(0.5)
    u, v, w = FT(1), FT(2), FT(3)

    # Tangential wind speed of a uniform flow on each wall
    tangential_speed = (west=√(v^2 + w^2), east=√(v^2 + w^2), south=√(u^2 + w^2), north=√(u^2 + w^2),
                        bottom=√(u^2 + v^2), top=√(u^2 + v^2))

    @testset "Drag, heat, and vapor fluxes match the bulk formulas on every wall" begin
        model = closed_box_model(FT; coefficient=C)
        grid = model.grid
        Nx, Ny, Nz = size(grid)
        constants = model.thermodynamic_constants
        set!(model; θ=θᵢ, ℋ=ℋᵢ, u, v, w, enforce_mass_conservation=false)

        probe = (2, 3, 2)
        clock, fields = boundary_condition_args(model)

        for side in sides
            i, j, k = near_wall_cell(side, probe..., Nx, Ny, Nz)
            Ũ = tangential_speed[side]
            sign = outward_sign(side)

            # Momentum: drag on the two tangential components, with the sign that removes momentum
            tangential_momentum = side ∈ (:bottom, :top) ? ((:ρu, u), (:ρv, v)) :
                                  side ∈ (:west, :east)  ? ((:ρv, v), (:ρw, w)) : ((:ρu, u), (:ρw, w))
            for (name, uₜ) in tangential_momentum
                field = getproperty(model.momentum, name)
                bc = wall_bc(field, side)
                ρ₀ = surface_density(bc.condition.surface_pressure, T₀[side], constants)
                expected = sign * ρ₀ * C * Ũ * uₜ
                @test evaluate_bc(model, field, side, probe...) ≈ expected rtol=10 * eps(FT)
            end

            # Sensible heat: into the domain when the wall is warmer than the air
            bc = wall_bc(prognostic_fields(model).ρθ, side)
            p₀ = bc.condition.surface_pressure
            pˢᵗ = bc.condition.standard_pressure
            ρ₀ = surface_density(p₀, T₀[side], constants)
            θ₀ = potential_temperature_from_temperature(T₀[side], p₀, pˢᵗ, constants)
            θ = @allowscalar fields.θ[i, j, k]
            expected = sign * ρ₀ * C * Ũ * (θ - θ₀)
            @test evaluate_bc(model, prognostic_fields(model).ρθ, side, probe...) ≈ expected rtol=100 * eps(FT)

            # Vapor: into the domain when the wall is moister than the air
            bc = wall_bc(prognostic_fields(model).ρqᵛ, side)
            ρ₀ = surface_density(bc.condition.surface_pressure, T₀[side], constants)
            qᵛ₀ = ℋ₀[side] * saturation_specific_humidity(T₀[side], ρ₀, constants, PlanarLiquidSurface())
            qᵛ = @allowscalar fields.qᵛ[i, j, k]
            expected = sign * ρ₀ * C * Ũ * (qᵛ - qᵛ₀)
            @test evaluate_bc(model, prognostic_fields(model).ρqᵛ, side, probe...) ≈ expected rtol=100 * eps(FT)
        end
    end

    @testset "Static energy flux includes the potential energy of the wall" begin
        model = closed_box_model(FT; formulation=:StaticEnergy, coefficient=C)
        grid = model.grid
        Nx, Ny, Nz = size(grid)
        constants = model.thermodynamic_constants
        set!(model; θ=θᵢ, ℋ=ℋᵢ, u, v, w, enforce_mass_conservation=false)
        clock, fields = boundary_condition_args(model)
        g = constants.gravitational_acceleration
        cᵖᵈ = constants.dry_air.heat_capacity
        cᵖᵛ = constants.vapor.heat_capacity

        probe = (2, 3, 2)
        wall_height = (west=nothing, east=nothing, south=nothing, north=nothing, bottom=FT(0), top=FT(1))
        for side in sides
            i, j, k = near_wall_cell(side, probe..., Nx, Ny, Nz)
            z₀ = isnothing(wall_height[side]) ? (@allowscalar znode(i, j, k, grid, Center(), Center(), Center())) : wall_height[side]
            bc = wall_bc(prognostic_fields(model).ρs, side)
            ρ₀ = surface_density(bc.condition.surface_pressure, T₀[side], constants)
            qᵛ = @allowscalar fields.qᵛ[i, j, k]
            cᵖᵐ = (1 - qᵛ) * cᵖᵈ + qᵛ * cᵖᵛ
            s₀ = cᵖᵐ * T₀[side] + g * z₀
            s = @allowscalar fields.s[i, j, k]
            expected = outward_sign(side) * ρ₀ * C * tangential_speed[side] * (s - s₀)
            @test evaluate_bc(model, prognostic_fields(model).ρs, side, probe...) ≈ expected rtol=100 * eps(FT)
        end
    end

    @testset "Wall fluxes heat and moisten the near-wall cells with the right sign" begin
        model = closed_box_model(FT; coefficient=C)
        Nx, Ny, Nz = size(model.grid)
        # At rest the drag vanishes, so give the walls a gust to move heat and vapor. The gust
        # and the time step are large enough that the change in a near-wall cell is well above
        # the roundoff of the cell's ρθ in single precision.
        gusty(side) = BulkSensibleHeatFlux(; coefficient=C, gustiness=FT(1), surface_temperature=T₀[side])
        gusty_vapor(side) = BulkVaporFlux(; coefficient=C, gustiness=FT(1), surface_temperature=T₀[side],
                                          surface_relative_humidity=ℋ₀[side])
        boundary_conditions = (; ρθ = FieldBoundaryConditions(; (side => gusty(side) for side in sides)...),
                                 ρqᵛ = FieldBoundaryConditions(; (side => gusty_vapor(side) for side in sides)...))
        model = AtmosphereModel(model.grid; boundary_conditions, advection=WENO(order=3))
        set!(model; θ=θᵢ, ℋ=FT(0.3))
        ρθ⁰ = Array(interior(prognostic_fields(model).ρθ))
        ρqᵛ⁰ = Array(interior(prognostic_fields(model).ρqᵛ))
        time_step!(model, FT(0.05))
        Δρθ = Array(interior(prognostic_fields(model).ρθ)) .- ρθ⁰
        Δρqᵛ = Array(interior(prognostic_fields(model).ρqᵛ)) .- ρqᵛ⁰

        # The air is at 290 K; walls warmer than that heat the near-wall cell, colder ones cool it
        warmer(side) = T₀[side] > θᵢ
        near(A, side) = (side === :west  ? A[1, 2:Ny-1, 2:Nz-1] : side === :east  ? A[Nx, 2:Ny-1, 2:Nz-1] :
                         side === :south ? A[2:Nx-1, 1, 2:Nz-1] : side === :north ? A[2:Nx-1, Ny, 2:Nz-1] :
                         side === :bottom ? A[2:Nx-1, 2:Ny-1, 1] : A[2:Nx-1, 2:Ny-1, Nz])
        for side in sides
            @test all(warmer(side) ? near(Δρθ, side) .> 0 : near(Δρθ, side) .< 0)
        end
        # Cells away from every wall change far less than the near-wall cells: only the pressure
        # projection of the near-wall buoyancy reaches them within one step
        interior_change = maximum(abs, Δρθ[2:Nx-1, 2:Ny-1, 2:Nz-1])
        wall_change = minimum(abs, Δρθ[2:Nx-1, 2:Ny-1, 1])
        @test interior_change < 1e-3 * wall_change

        # Air at 30 % relative humidity and 290 K holds less vapor than the air in contact with any of
        # the walls (saturated at 280 and 299 K, 78 % at 285–295 K), so every wall moistens it
        for side in sides
            @test all(near(Δρqᵛ, side) .> 0)
        end
    end

    @testset "Wall state given as a function of the wall coordinates" begin
        grid = RectilinearGrid(default_arch, FT; size=(4, 4, 4), x=(0, 1), y=(0, 1), z=(0, 1),
                               topology=(Bounded, Bounded, Bounded))
        T_west(y, z) = 290 + 4y + 2z
        T_bottom(x, y) = 290 + 3x + y
        ρθ_bcs = FieldBoundaryConditions(west=BulkSensibleHeatFlux(; coefficient=C, gustiness=FT(1), surface_temperature=T_west),
                                         bottom=BulkSensibleHeatFlux(; coefficient=C, gustiness=FT(1), surface_temperature=T_bottom))
        model = AtmosphereModel(grid; boundary_conditions=(; ρθ=ρθ_bcs), advection=WENO(order=3))
        set!(model; θ=θᵢ, ℋ=ℋᵢ)
        clock, fields = boundary_condition_args(model)
        constants = model.thermodynamic_constants

        for (side, T_wall, i, j, k) in ((:west, T_west, 1, 3, 2), (:bottom, T_bottom, 2, 3, 1))
            bc = wall_bc(prognostic_fields(model).ρθ, side)
            x, y, z = @allowscalar (xnode(i, j, k, grid, Center(), Center(), Center()),
                                    ynode(i, j, k, grid, Center(), Center(), Center()),
                                    znode(i, j, k, grid, Center(), Center(), Center()))
            T₀ⁱʲᵏ = side === :west ? T_wall(y, z) : T_wall(x, y)
            p₀ = bc.condition.surface_pressure
            ρ₀ = surface_density(p₀, T₀ⁱʲᵏ, constants)
            θ₀ = potential_temperature_from_temperature(T₀ⁱʲᵏ, p₀, bc.condition.standard_pressure, constants)
            θ = @allowscalar fields.θ[i, j, k]
            expected = outward_sign(side) * ρ₀ * C * FT(1) * (θ - θ₀)   # at rest: Ũ = gustiness
            @test evaluate_bc(model, prognostic_fields(model).ρθ, side, i, j, k) ≈ expected rtol=100 * eps(FT)
        end
    end

    @testset "Invalid wall configurations are rejected" begin
        grid = RectilinearGrid(default_arch, FT; size=(4, 4, 4), x=(0, 1), y=(0, 1), z=(0, 1),
                               topology=(Bounded, Bounded, Bounded))
        # Drag on the momentum component normal to the wall
        ρu_bcs = FieldBoundaryConditions(west=BulkDrag(; coefficient=C))
        @test_throws ArgumentError AtmosphereModel(grid; boundary_conditions=(; ρu=ρu_bcs))
        ρw_bcs = FieldBoundaryConditions(top=BulkDrag(; coefficient=C))
        @test_throws ArgumentError AtmosphereModel(grid; boundary_conditions=(; ρw=ρw_bcs))
        # Temporal filtering of the surface state is a bottom-wall feature
        fv = FilteredSurfaceVelocities(grid; filter_timescale=FT(10))
        ρv_bcs = FieldBoundaryConditions(top=BulkDrag(; coefficient=C, filtered_velocities=fv))
        @test_throws ArgumentError AtmosphereModel(grid; boundary_conditions=(; ρv=ρv_bcs))
    end

    Oceananigans.defaults.FloatType = old_FT
end
