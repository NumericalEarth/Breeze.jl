include(joinpath(@__DIR__, "setup.jl"))

#####
##### All-sky ecCKD radiation: cloud scattering tables on the ecCKD g points
#####
##### A 500 m liquid cloud of 0.5 g kg⁻¹ at 1–1.5 km, prescribed exactly through a prognostic
##### cloud liquid that is held (zero condensation rate), is solved with the Mie droplet and Baum
##### ice scattering tables mapped onto the `climate_32x32` g points. The cloud must shade the
##### surface and warm it in the longwave, the all-sky optics must reproduce the clear-sky
##### fluxes bit for bit when there is no condensate, and the droplet size must matter.
#####

using Breeze
using Breeze.AtmosphereModels: _update_radiation!, total_density
using CloudMicrophysics
using Dates: DateTime
using NCDatasets
using NumericalRadiation: NumericalRadiation, SpectralCloudOptics, EcCKDGasOpticsModel
using Oceananigans
using Oceananigans.Units
using Test

const CloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .CloudMicrophysicsExt: OneMomentCloudMicrophysics

const NumericalRadiationExt = Base.get_extension(Breeze, :BreezeNumericalRadiationExt)
using .NumericalRadiationExt: EcCKDRadiativeTransferModel, CLOUD_SCATTERING_FILES

# 20 layers of 500 m: the cloud fills cell 3, from 1 to 1.5 km
const Nz = 20
const TOP = 10kilometers
const CLOUD_BOTTOM = 1kilometer
const CLOUD_TOP = 1.5kilometers
const CLOUD_CELL = 3
const CLOUD_LIQUID = 0.5e-3   # kg kg⁻¹
const S₀ = 1361
const μ₀ = 0.5

column_grid(FT) = RectilinearGrid(default_arch, FT; size = Nz, x = 0.0, y = 45.0, z = (0, TOP),
                                  topology = (Flat, Flat, Bounded))

function radiation(grid, optics; liquid_radius = 10e-6, ice_radius = 30e-6, kw...)
    return RadiativeTransferModel(grid, optics, ThermodynamicConstants();
                                  surface_temperature = 300, surface_emissivity = 0.98, surface_albedo = 0.1,
                                  solar_constant = S₀, solar_position = FixedCosineZenith(μ₀),
                                  liquid_effective_radius = ConstantRadiusParticles(liquid_radius),
                                  ice_effective_radius = ConstantRadiusParticles(ice_radius), kw...)
end

all_sky_radiation(grid; kw...) = radiation(grid, EcCKDOptics(clouds = CloudScatteringTables()); kw...)
clear_sky_radiation(grid; kw...) = radiation(grid, EcCKDOptics(); kw...)

# Cloud liquid is a prognostic that is held (zero condensation rate), so `set!` prescribes it exactly
held_liquid_microphysics(FT) =
    OneMomentCloudMicrophysics(FT; cloud_formation = NonEquilibriumCloudFormation(ConstantRateCondensateFormation(zero(FT))))

# A warm, moist column with the cloud in its cell (or clear, with `cloud_liquid = 0`)
function column_model(grid, radiation; microphysics = held_liquid_microphysics(eltype(grid)), cloud_liquid = CLOUD_LIQUID)
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure = 101325, potential_temperature = 300)
    dynamics = AnelasticDynamics(reference_state)
    clock = Clock(time = DateTime(2024, 6, 21, 12, 0, 0))
    model = AtmosphereModel(grid; clock, dynamics, microphysics, formulation = :LiquidIcePotentialTemperature, radiation)
    θ(z) = 300 + 5e-3 * z
    qᵗ(z) = 0.010 * exp(-z / 2500)
    if isnothing(microphysics)
        set!(model; θ, qᵗ)
    else
        qᶜˡ(z) = ifelse(CLOUD_BOTTOM < z < CLOUD_TOP, cloud_liquid, 0)
        set!(model; θ, qᵗ, qᶜˡ)
    end
    return model
end

# The four fluxes on the faces of the single column, as host vectors
function column_fluxes(radiation)
    ℐ_lw_up = Array(interior(radiation.upwelling_longwave_flux))[1, 1, :]
    ℐ_lw_dn = Array(interior(radiation.downwelling_longwave_flux))[1, 1, :]
    ℐ_sw_up = Array(interior(radiation.upwelling_shortwave_flux))[1, 1, :]
    ℐ_sw_dn = Array(interior(radiation.downwelling_shortwave_flux))[1, 1, :]
    return (; ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn)
end

@testset "All-sky ecCKD RadiativeTransferModel" begin

    @testset "Construction and types [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        grid = column_grid(FT)
        all_sky = all_sky_radiation(grid)

        @test all_sky isa EcCKDRadiativeTransferModel
        for solver in (all_sky.longwave_solver, all_sky.shortwave_solver)
            cloud = solver.cloud
            @test cloud.liquid isa SpectralCloudOptics{FT}
            @test cloud.ice isa SpectralCloudOptics{FT}
            for phase in (cloud.liquid, cloud.ice)
                @test size(phase.mass_extinction_coefficient) == (length(solver.weights), 1)
                @test all(>(0), Array(phase.mass_extinction_coefficient))
                @test all(x -> 0 ≤ x ≤ 1, Array(phase.single_scattering_albedo))
                @test all(x -> -1 ≤ x ≤ 1, Array(phase.asymmetry_factor))
            end
            @test Array(cloud.liquid.effective_radius) == [FT(10e-6)]
            @test Array(cloud.ice.effective_radius) == [FT(30e-6)]
            @test all(isconcretetype, fieldtypes(typeof(solver)))
        end

        # Liquid droplets scatter sunlight more conservatively than they absorb the longwave
        @test minimum(Array(all_sky.shortwave_solver.cloud.liquid.single_scattering_albedo)) >
              minimum(Array(all_sky.longwave_solver.cloud.liquid.single_scattering_albedo))

        @test occursin("32 longwave and 32 shortwave g-points, all sky", sprint(show, all_sky))

        # The selectors name files in the ecRad data: a path to the same file gives the same optics
        paths = CloudScatteringTables(liquid = NumericalRadiation.ecrad_data_file(CLOUD_SCATTERING_FILES.mie_droplet),
                                      ice = NumericalRadiation.ecrad_data_file(CLOUD_SCATTERING_FILES.baum_general_habit_mixture))
        by_path = radiation(grid, EcCKDOptics(clouds = paths))
        for region in (:longwave_solver, :shortwave_solver), phase in (:liquid, :ice)
            a = getproperty(getproperty(all_sky, region).cloud, phase)
            b = getproperty(getproperty(by_path, region).cloud, phase)
            @test Array(a.mass_extinction_coefficient) == Array(b.mass_extinction_coefficient)
            @test Array(a.single_scattering_albedo) == Array(b.single_scattering_albedo)
            @test Array(a.asymmetry_factor) == Array(b.asymmetry_factor)
        end

        # A radius off the table is held at its edge node rather than extrapolated
        droplet_table = NumericalRadiation.read_cloud_scattering_table(paths.liquid)
        tiny = radiation(grid, EcCKDOptics(clouds = CloudScatteringTables()); liquid_radius = 1e-7)
        edge = radiation(grid, EcCKDOptics(clouds = CloudScatteringTables()); liquid_radius = first(droplet_table.effective_radius))
        @test Array(tiny.shortwave_solver.cloud.liquid.mass_extinction_coefficient) ==
              Array(edge.shortwave_solver.cloud.liquid.mass_extinction_coefficient)

        @testset "Argument errors" begin
            # An unknown scattering table selector
            err = try
                radiation(grid, EcCKDOptics(clouds = CloudScatteringTables(liquid = :nonexistent)))
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("nonexistent", err.msg)

            # A preloaded gas model carries no definition files to map the tables onto
            gas_model = NumericalRadiation.read_reference_ecckd_gas_optics(:climate_32x32; names = NumericalRadiationExt.ECCKD_GAS_NAMES)
            @test_throws ArgumentError radiation(grid, EcCKDOptics(gas_model; clouds = CloudScatteringTables()))

            # Nor does a gray one
            gray = EcCKDGasOpticsModel(names = (:composite,), longwave_absorption = [1e-3;;], shortwave_absorption = [1e-4;;])
            @test_throws ArgumentError radiation(grid, EcCKDOptics(gray; clouds = CloudScatteringTables()); column_extension = nothing)
        end
    end

    @testset "Clear sky is all sky without condensate [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        grid = column_grid(FT)

        # No microphysics: the water paths are zero and the cloud optics add exactly nothing
        all_sky = all_sky_radiation(grid)
        clear_sky = clear_sky_radiation(grid)
        column_model(grid, all_sky; microphysics = nothing)
        column_model(grid, clear_sky; microphysics = nothing)

        @test all(iszero, Array(all_sky.atmospheric_state.liquid_water_path))
        @test all(iszero, Array(all_sky.atmospheric_state.ice_water_path))

        all_sky_fluxes = column_fluxes(all_sky)
        clear_sky_fluxes = column_fluxes(clear_sky)
        for name in keys(all_sky_fluxes)
            @test all_sky_fluxes[name] == clear_sky_fluxes[name]
        end
        @test Array(interior(all_sky.flux_divergence)) == Array(interior(clear_sky.flux_divergence))

        # And so with the cloud microphysics but no cloud
        column_model(grid, all_sky; cloud_liquid = 0)
        column_model(grid, clear_sky; cloud_liquid = 0)
        all_sky_fluxes = column_fluxes(all_sky)
        clear_sky_fluxes = column_fluxes(clear_sky)
        for name in keys(all_sky_fluxes)
            @test all_sky_fluxes[name] == clear_sky_fluxes[name]
        end
    end

    @testset "A 500 m cloud of 0.5 g kg⁻¹ at 1–1.5 km [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        grid = column_grid(FT)

        all_sky = all_sky_radiation(grid)
        clear_sky = clear_sky_radiation(grid)
        model = column_model(grid, all_sky)
        column_model(grid, clear_sky)   # same state; the clear-sky optics ignore the condensate

        # The staged cloud: one layer with the prescribed liquid water path, no ice
        columns = all_sky.atmospheric_state
        N = size(columns.liquid_water_path, 2)
        lwp = Array(columns.liquid_water_path)[1, :]
        ρ = Array(interior(total_density(model.dynamics)))[1, 1, CLOUD_CELL]
        Δz = TOP / Nz
        @test count(>(0), lwp) == 1
        @test lwp[N + 1 - CLOUD_CELL] ≈ ρ * CLOUD_LIQUID * Δz rtol = (FT == Float64 ? 1e-12 : 1e-5)
        @test all(iszero, Array(columns.ice_water_path))

        cloudy = column_fluxes(all_sky)
        clear = column_fluxes(clear_sky)

        @test all(isfinite, cloudy.ℐ_lw_up) && all(isfinite, cloudy.ℐ_lw_dn)
        @test all(isfinite, cloudy.ℐ_sw_up) && all(isfinite, cloudy.ℐ_sw_dn)

        # The cloud shades the surface: at least 30 % less sunlight reaches it, and every
        # face below the cloud base sees the shade
        surface_shortwave_down(f) = -f.ℐ_sw_dn[1]
        @info "Cloud effect ($FT): surface SW↓ $(surface_shortwave_down(clear)) → $(surface_shortwave_down(cloudy)) W m⁻²; " *
              "surface LW↓ $(-clear.ℐ_lw_dn[1]) → $(-cloudy.ℐ_lw_dn[1]) W m⁻²; " *
              "top SW↑ $(clear.ℐ_sw_up[Nz+1]) → $(cloudy.ℐ_sw_up[Nz+1]) W m⁻²; " *
              "OLR $(clear.ℐ_lw_up[Nz+1]) → $(cloudy.ℐ_lw_up[Nz+1]) W m⁻²"
        @test surface_shortwave_down(cloudy) ≤ 0.7 * surface_shortwave_down(clear)
        @test all(-cloudy.ℐ_sw_dn[1:CLOUD_CELL] .≤ 0.7 .* -clear.ℐ_sw_dn[1:CLOUD_CELL])

        # and reflects sunlight back to space
        @test cloudy.ℐ_sw_up[Nz+1] > clear.ℐ_sw_up[Nz+1] + 50

        # The cloud emits toward the surface as a warm black body: at least 30 W m⁻² more
        # downwelling longwave, and the cloud base is close to black
        @test -cloudy.ℐ_lw_dn[1] ≥ -clear.ℐ_lw_dn[1] + 30
        σ = all_sky.longwave_solver.gas_model.stefan_boltzmann
        T_base = Array(interior(model.temperature))[1, 1, CLOUD_CELL]
        @test -cloudy.ℐ_lw_dn[CLOUD_CELL] > 0.9 * σ * T_base^4

        # The cloud top is colder than the surface, so less longwave leaves the column
        @test cloudy.ℐ_lw_up[Nz+1] < clear.ℐ_lw_up[Nz+1]

        # The cloud cools at its top in the longwave: net longwave heating of the cloud cell is negative
        # and stronger than in clear sky
        ℐ_lw_net(f) = f.ℐ_lw_up .+ f.ℐ_lw_dn
        cloud_longwave_heating(f) = -(ℐ_lw_net(f)[CLOUD_CELL+1] - ℐ_lw_net(f)[CLOUD_CELL]) / Δz
        @test cloud_longwave_heating(cloudy) < cloud_longwave_heating(clear) < 0

        # Column energy closure survives the cloud
        ℐ_net = cloudy.ℐ_lw_up .+ cloudy.ℐ_lw_dn .+ cloudy.ℐ_sw_up .+ cloudy.ℐ_sw_dn
        column_heating = Δz * sum(Array(interior(all_sky.flux_divergence)))
        @test column_heating ≈ ℐ_net[1] - ℐ_net[Nz+1] rtol = (FT == Float64 ? 1e-10 : 1e-5)

        # The top-of-grid shortwave irradiance is unchanged
        @test -cloudy.ℐ_sw_dn[Nz+1] ≈ -clear.ℐ_sw_dn[Nz+1] rtol = 1e-2
    end

    @testset "Effective radius [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        grid = column_grid(FT)

        small = all_sky_radiation(grid; liquid_radius = 5e-6)
        large = all_sky_radiation(grid; liquid_radius = 20e-6)
        column_model(grid, small)
        column_model(grid, large)
        small_fluxes = column_fluxes(small)
        large_fluxes = column_fluxes(large)

        # Smaller droplets at the same water path make a thicker cloud: more reflection, less
        # transmission, by more than 5 %
        @info "Effective radius ($FT): top SW↑ $(small_fluxes.ℐ_sw_up[Nz+1]) (5 μm) vs $(large_fluxes.ℐ_sw_up[Nz+1]) W m⁻² (20 μm)"
        @test small_fluxes.ℐ_sw_up[Nz+1] > 1.05 * large_fluxes.ℐ_sw_up[Nz+1]
        @test -small_fluxes.ℐ_sw_dn[1] < -large_fluxes.ℐ_sw_dn[1]

        # The ice radius does not matter without ice
        ice₁ = all_sky_radiation(grid; ice_radius = 30e-6)
        ice₂ = all_sky_radiation(grid; ice_radius = 60e-6)
        column_model(grid, ice₁)
        column_model(grid, ice₂)
        f₁ = column_fluxes(ice₁)
        f₂ = column_fluxes(ice₂)
        for name in keys(f₁)
            @test f₁[name] == f₂[name]
        end
    end

    @testset "Allocation scaling" begin
        # As for clear sky: the cloudy update allocates the fixed overhead of its kernel
        # launches, not per column
        Oceananigans.defaults.FloatType = Float64
        constants = ThermodynamicConstants()
        allocations = map((1, 64)) do Nx
            grid = RectilinearGrid(default_arch; size = (Nx, 1, Nz), x = (0, Nx), y = (0, 1), z = (0, TOP),
                                   topology = (Periodic, Periodic, Bounded))
            all_sky = all_sky_radiation(grid)
            reference_state = ReferenceState(grid, constants; base_pressure = 101325, potential_temperature = 300)
            model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state),
                                    microphysics = held_liquid_microphysics(Float64),
                                    formulation = :LiquidIcePotentialTemperature, radiation = all_sky)
            set!(model; θ = (x, y, z) -> 300 + 5e-3 * z, qᵗ = (x, y, z) -> 0.010 * exp(-z / 2500),
                        qᶜˡ = (x, y, z) -> ifelse(CLOUD_BOTTOM < z < CLOUD_TOP, CLOUD_LIQUID, 0))
            _update_radiation!(all_sky, model)
            return @allocated _update_radiation!(all_sky, model)
        end
        @test abs(allocations[2] - allocations[1]) < 1024
    end
end

Oceananigans.defaults.FloatType = Float64
