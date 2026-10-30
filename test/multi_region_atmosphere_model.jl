include(joinpath(@__DIR__, "setup.jl"))

#####
##### AtmosphereModel with CompressibleDynamics on a partitioned MultiRegionGrid. The regions exchange
##### halos by a plain copy and carry the metrics of the global grid, so the model must reproduce the
##### single-grid model, up to the rounding of kernels compiled for the regions.
#####

using Breeze
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.Advection: cell_advection_timescale
using Oceananigans.MultiRegion: MultiRegionGrid, XPartition, YPartition
using Oceananigans.Units
using Oceananigans.Utils: getregion, regions
using Test

sphere(arch, FT) = LatitudeLongitudeGrid(arch, FT; size=(32, 16, 8), halo=(5, 5, 5), longitude=(0, 360),
                                         latitude=(-60, 60), z=(0, 10kilometers), topology=(Periodic, Bounded, Bounded))

# TODO: `reconstruct_global_field` drops the last face of a `Face` field in a partitioned Bounded
# direction; fix it in Oceananigans and use it here.
partition_dimension(::XPartition) = 1
partition_dimension(::YPartition) = 2
global_interior(field) = interior(field)
global_interior(field::Field{<:Any, <:Any, <:Any, <:Any, <:MultiRegionGrid}) =
    cat((interior(getregion(field, r)) for r in regions(field.grid))...; dims=partition_dimension(field.grid.partition))

# Every prognostic field of the two models agrees to `rtol`, which is bit for bit at zero
function test_prognostic_fields_agree(single, multi; rtol=0)
    for (name, field) in pairs(prognostic_fields(single))
        @test global_interior(field) ≈ global_interior(prognostic_fields(multi)[name]) rtol=rtol
    end
end

θᵢ(λ, φ, z) = 300 + 0.5 * cosd(3λ) * sind(2φ) * exp(-z / 2000)
uᵢ(λ, φ, z) = 10 * cosd(φ) + 0.1 * sind(2λ) * cosd(φ)

# Dry, explicit SSP RK3, with forcings keyed on specific variables, which Breeze wraps with the model density
function dry_explicit_model(grid)
    dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature=300)
    forcing = (; θ = Forcing((λ, φ, z, t) -> 1e-3 * exp(-z / 1000)),
                 u = Forcing((λ, φ, z, t, u) -> -1e-3 * u, field_dependencies=:u))
    model = AtmosphereModel(grid; dynamics, forcing, coriolis=SphericalCoriolis(), advection=WENO())
    set!(model; θ=θᵢ, u=uᵢ, ρ=model.dynamics.reference_state.density)
    return model
end

# Moist, split-explicit acoustic RK3 with saturation adjustment, the TKE closure and the Held–Suarez forcing
function moist_acoustic_model(grid)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps=4);
                                    reference_potential_temperature = z -> 300 + 3e-3 * z)
    model = AtmosphereModel(grid; dynamics, coriolis=SphericalCoriolis(), advection=WENO(),
                            microphysics=SaturationAdjustment(equilibrium=WarmPhaseEquilibrium()),
                            closure=TKEBasedTurbulenceClosure(), forcing=held_suarez_forcings())
    set!(model; θ = (λ, φ, z) -> θᵢ(λ, φ, z) + 3e-3 * z,
                qᵗ = (λ, φ, z) -> 0.02 * exp(-z / 3000) * (1 + 0.1 * cosd(2λ)),
                u = uᵢ, ρe = 1e-2, ρ = model.dynamics.reference_state.density)
    return model
end

# Open boundaries in a Bounded partitioned direction: the last region holds the east face
function open_boundary_model(grid)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=300)
    ρu_bcs = FieldBoundaryConditions(west=NormalFlowBoundaryCondition(6), east=NormalFlowBoundaryCondition(6))
    model = AtmosphereModel(grid; dynamics, advection=WENO(), boundary_conditions=(; ρu=ρu_bcs))
    set!(model; θ = (x, y, z) -> 300 + exp(-((x - 4e3)^2 + (z - 2e3)^2) / 1e6), u=5, ρ=HydrostaticallyBalancedDensity())
    return model
end

# TODO: on Julia 1.11 for x86_64, LLVM's instruction scheduler takes about 40 minutes to compile the moist
# multi-region model below (the constructor of its `AtmosphereModel` alone takes minutes) and overflows the
# stack of the CI test worker. Julia 1.11 on aarch64 and Julia ≥ 1.12 compile it in a minute or two.
compiles_moist_multi_region_model() = VERSION ≥ v"1.12" || Sys.ARCH !== :x86_64

function step_both!(single, multi, Δt, N)
    for _ in 1:N
        time_step!(single, Δt)
        time_step!(multi, Δt)
    end
    return nothing
end

@testset "Multi-region AtmosphereModel [$(FT)]" for FT in test_float_types()
    previous_float_type = Oceananigans.defaults.FloatType
    Oceananigans.defaults.FloatType = FT

    try
        grid = sphere(default_arch, FT)
        xpartitioned_grid = MultiRegionGrid(grid; partition=XPartition(2))
        rounding = 1e5 * eps(FT)

        @testset "Reference column" begin
            single = dry_explicit_model(grid)
            multi = dry_explicit_model(xpartitioned_grid)
            @test size(multi.dynamics.reference_state.pressure) == (1, 1, size(grid, 3))

            # The reset takes horizontal means across the regions, which are those of the single grid
            θ̄ᵢ(λ, φ, z) = 300 + 3e-3 * z + 2 * sind(φ) * cosd(λ)
            set!(single; θ=θ̄ᵢ, ρ=1.1, compute_reference_state=true)
            set!(multi; θ=θ̄ᵢ, ρ=1.1, compute_reference_state=true)
            for name in (:pressure, :density, :exner_function)
                @test interior(getproperty(multi.dynamics.reference_state, name)) ≈
                      interior(getproperty(single.dynamics.reference_state, name))
            end

            # The anelastic pressure solve is global and refused up front
            @test_throws ArgumentError AtmosphereModel(xpartitioned_grid)
        end

        @testset "Dry explicit SSPRK3 vs single grid [$(P)]" for P in (XPartition, YPartition)
            single = dry_explicit_model(grid)
            multi = dry_explicit_model(MultiRegionGrid(grid; partition=P(2)))
            step_both!(single, multi, 2, 3)
            test_prognostic_fields_agree(single, multi)
            @test global_interior(single.velocities.u) == global_interior(multi.velocities.u)
            @test maximum(abs, single.momentum.ρw) > 0
        end

        compiles_moist_multi_region_model() &&
        @testset "Moist acoustic RK3 with TKE closure and Held–Suarez forcing vs single grid" begin
            single = moist_acoustic_model(grid)
            multi = moist_acoustic_model(xpartitioned_grid)
            test_prognostic_fields_agree(single, multi)
            @test cell_advection_timescale(multi) == cell_advection_timescale(single)
            for Diagnostic in (PotentialTemperature, StaticEnergy, RelativeHumidity, SaturationSpecificHumidity, DewpointTemperature)
                @test global_interior(Field(Diagnostic(single))) == global_interior(Field(Diagnostic(multi)))
            end

            # The moisture and TKE tendencies of a region differ from the single grid by a few ulps, which the
            # acoustic substeps amplify in the momentum
            step_both!(single, multi, 20, 3)
            test_prognostic_fields_agree(single, multi; rtol=rounding)
            @test maximum(single.microphysical_fields.qˡ) > 0
        end

        @testset "Open boundaries in a partitioned Bounded direction" begin
            box = RectilinearGrid(default_arch, FT; size=(16, 16, 8), halo=(5, 5, 5), x=(0, 8kilometers),
                                  y=(0, 8kilometers), z=(0, 8kilometers), topology=(Bounded, Periodic, Bounded))
            single = open_boundary_model(box)
            multi = open_boundary_model(MultiRegionGrid(box; partition=XPartition(2)))
            step_both!(single, multi, 2, 2)
            test_prognostic_fields_agree(single, multi; rtol=rounding)
            ρu = global_interior(multi.momentum.ρu)
            @test all(==(6), view(ρu, 1, :, :)) && all(==(6), view(ρu, size(box, 1) + 1, :, :))
        end
    finally
        Oceananigans.defaults.FloatType = previous_float_type
    end
end
