include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Dates
using Oceananigans
using Oceananigans.Fields: ConstantField
using Oceananigans.Units
using Test

# Trigger RRTMGP + netCDF lookup table loading
using ClimaComms
using NCDatasets
using RRTMGP

# The surface emissivity and albedos may be scalars or 2D fields. RRTMGP stores them band by band and
# column by column, in arrays only Breeze writes, so a field-valued property that never gets
# transferred leaves the solver reading its own uninitialized allocation: a silently wrong albedo at
# best, NaN fluxes at worst. These tests check the transfer for all three optics types, at
# construction and again after a solve (so a property that evolves is picked up).

# Compare against the column ordering the extension actually uses, rather than a copy of it.
const rrtmgp_column_index = Base.get_extension(Breeze, :BreezeRRTMGPExt).rrtmgp_column_index

# All-sky and clear-sky keep both RTE solvers inside one `RRTMGPSolver`, which publishes accessors;
# gray keeps a `NoScatLWRTE` and a `NoScatSWRTE` side by side, which do not.
function rrtmgp_surface_arrays(radiation)
    solver = radiation.longwave_solver

    isnothing(radiation.shortwave_solver) &&
        return (Array(RRTMGP.surface_emissivity(solver)),
                Array(RRTMGP.direct_sw_surface_albedo(solver)),
                Array(RRTMGP.diffuse_sw_surface_albedo(solver)))

    return (Array(solver.bcs.sfc_emis),
            Array(radiation.shortwave_solver.bcs.sfc_alb_direct),
            Array(radiation.shortwave_solver.bcs.sfc_alb_diffuse))
end

# What RRTMGP's (nband, ncolumn) array should hold: every band of a column carrying that column's
# field value. Comparing whole arrays means a failure reports the mismatch, not just `false`.
function expected_surface_array(field, grid, Nband)
    Nx, Ny, _ = size(grid)
    values = Array(interior(field))
    expected = similar(values, Nband, Nx * Ny)

    for j in 1:Ny, i in 1:Nx
        expected[:, rrtmgp_column_index(i, j, Nx)] .= values[i, j, 1]
    end

    return expected
end

@testset "Field-valued surface radiative properties [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    Nx = Ny = 4
    Nz = 8

    grid = RectilinearGrid(default_arch; size = (Nx, Ny, Nz),
                           x = (0, 1kilometers), y = (0, 1kilometers), z = (0, 10kilometers),
                           topology = (Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure = 101325,
                                     potential_temperature = 300)
    dynamics = AnelasticDynamics(reference_state)

    # Horizontally varying properties, so a per-column transfer is distinguishable from a
    # domain-uniform fill.
    α = Field{Center, Center, Nothing}(grid)
    ε = Field{Center, Center, Nothing}(grid)
    set!(α, [0.10 + 0.50 * (i - 1) / Nx for i in 1:Nx, j in 1:Ny, k in 1:1])
    set!(ε, [0.90 + 0.09 * (j - 1) / Ny for i in 1:Nx, j in 1:Ny, k in 1:1])

    solar_position = ApparentSolarPosition(coordinate = (0, 45), epoch = DateTime(2024, 6, 21, 12))

    # Gray optics solves the shortwave without scattering, so `NoScatSWRTE` never reflects at the
    # surface and the albedo arrays, though transferred, do not influence the gray fluxes. The gray
    # albedo assertions below therefore cover the transfer only; the emissivity assertions bear on
    # the gray longwave, which does use them.
    optics = (GrayOptics(), ClearSkyOptics(), AllSkyOptics())

    for optic in optics
        @testset "$(summary(optic))" begin
            radiation = RadiativeTransferModel(grid, optic, constants;
                                               solar_position,
                                               surface_temperature = 300,
                                               surface_emissivity = ε,
                                               surface_albedo = α)

            ε₀, αᵈ₀, αˢ₀ = rrtmgp_surface_arrays(radiation)
            @test ε₀ ≈ expected_surface_array(ε, grid, size(ε₀, 1))      # emissivity at construction
            @test αᵈ₀ ≈ expected_surface_array(α, grid, size(αᵈ₀, 1))    # direct albedo at construction
            @test αˢ₀ ≈ expected_surface_array(α, grid, size(αˢ₀, 1))    # diffuse albedo at construction

            model = AtmosphereModel(grid; dynamics, radiation,
                                    clock = Clock(time = DateTime(2024, 6, 21, 12)),
                                    microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()),
                                    formulation = :LiquidIcePotentialTemperature)

            set!(model; θ = (x, y, z) -> 300 + 0.005 * z / 1000,
                        qᵗ = (x, y, z) -> 0.010 * exp(-z / 3000))

            # A property that changes between solves must be re-read, not frozen at construction.
            set!(α, (x, y) -> 0.7 - 0.5 * x / 1kilometers)

            Breeze.AtmosphereModels._update_radiation!(radiation, model)

            ε₀, αᵈ₀, αˢ₀ = rrtmgp_surface_arrays(radiation)
            @test ε₀ ≈ expected_surface_array(ε, grid, size(ε₀, 1))      # emissivity after a solve
            @test αᵈ₀ ≈ expected_surface_array(α, grid, size(αᵈ₀, 1))    # updated direct albedo
            @test αˢ₀ ≈ expected_surface_array(α, grid, size(αˢ₀, 1))    # updated diffuse albedo

            @test all(isfinite, Array(interior(radiation.flux_divergence)))
            @test all(isfinite, Array(interior(radiation.upwelling_longwave_flux)))
            @test all(isfinite, Array(interior(radiation.downwelling_shortwave_flux)))
        end
    end

    # Scalars are the common case and must land in every band of every column too.
    @testset "Scalar properties" begin
        radiation = RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                           solar_position,
                                           surface_temperature = 300,
                                           surface_emissivity = 0.97,
                                           surface_albedo = 0.23)

        ε₀, αᵈ₀, αˢ₀ = rrtmgp_surface_arrays(radiation)
        @test all(ε₀ .≈ FT(0.97))
        @test all(αᵈ₀ .≈ FT(0.23))
        @test all(αˢ₀ .≈ FT(0.23))
    end

    # The direct and diffuse albedos may be given separately and may differ. Each has to reach its
    # own RRTMGP array, rather than both picking up whichever field was materialized last.
    @testset "Distinct direct and diffuse albedos" begin
        αᵈ = Field{Center, Center, Nothing}(grid)
        αˢ = Field{Center, Center, Nothing}(grid)
        set!(αᵈ, (x, y) -> 0.15 + 0.40 * x / 1kilometers)
        set!(αˢ, (x, y) -> 0.60 - 0.40 * y / 1kilometers)

        radiation = RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                           solar_position,
                                           surface_temperature = 300,
                                           surface_emissivity = 0.98,
                                           direct_surface_albedo = αᵈ,
                                           diffuse_surface_albedo = αˢ)

        ε₀, αᵈ₀, αˢ₀ = rrtmgp_surface_arrays(radiation)
        @test all(ε₀ .≈ FT(0.98))
        @test αᵈ₀ ≈ expected_surface_array(αᵈ, grid, size(αᵈ₀, 1))
        @test αˢ₀ ≈ expected_surface_array(αˢ, grid, size(αˢ₀, 1))
    end

    # Emissivity and albedo are fractions. A scalar outside [0, 1] is a user error — an albedo in
    # percent, say — and is rejected at construction rather than integrated forward.
    @testset "Out-of-range scalars" begin
        @test_throws ArgumentError RadiativeTransferModel(grid, GrayOptics(), constants;
                                                          solar_position,
                                                          surface_temperature = 300,
                                                          surface_albedo = 30)

        @test_throws ArgumentError RadiativeTransferModel(grid, GrayOptics(), constants;
                                                          solar_position,
                                                          surface_temperature = 300,
                                                          surface_emissivity = 1.2,
                                                          surface_albedo = 0.2)

        # A `ConstantField` is a scalar in a field's clothing, so it is held to the same range as
        # the number it wraps rather than slipping through as "some field".
        @test_throws ArgumentError RadiativeTransferModel(grid, GrayOptics(), constants;
                                                          solar_position,
                                                          surface_temperature = 300,
                                                          surface_albedo = ConstantField(30))

        # A `Field` still passes: its contents may be rewritten before the next solve, so a check
        # here would say nothing about what the solver eventually reads.
        αⁱⁿ = Field{Center, Center, Nothing}(grid)
        set!(αⁱⁿ, 0.3)
        @test RadiativeTransferModel(grid, GrayOptics(), constants;
                                     solar_position,
                                     surface_temperature = 300,
                                     surface_albedo = αⁱⁿ) isa RadiativeTransferModel
    end
end
