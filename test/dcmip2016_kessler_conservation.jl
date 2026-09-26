include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Test
using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using Breeze.AtmosphereModels: microphysics_model_update!, surface_precipitation_flux, standard_pressure
using Breeze.Microphysics: DCMIP2016KesslerMicrophysics, phase_change_temperature_slope
using Breeze.Thermodynamics:
    LiquidIcePotentialTemperatureState,
    StaticEnergyState,
    MoistureMassFractions,
    MoistureMixingRatio,
    PlanarLiquidSurface,
    TetensFormula,
    saturation_specific_humidity,
    temperature,
    with_moisture

#####
##### Helpers
#####
# Every test below is a single anelastic column with the DCMIP2016 Kessler scheme, stepped by
# one operator-split Kessler update (`microphysics_model_update!`), and checks a budget or an
# invariant against the model's own prognostic fields. The budgets are formed with the
# finite-volume cell thicknesses, which is what the prognostic partial densities are defined on.

kessler_constants(FT) = ThermodynamicConstants(FT; saturation_vapor_pressure = TetensFormula(FT))

# Vertical grids: uniform; the NumericalEarth hindcast-like junction (uniform 50 m to 1 km, then
# 10 % stretching); and a geometric stretching from the surface up.
uniform_faces(FT) = collect(FT, 0:50:4000)

function junction_faces(FT; Δz = 50, uniform_top = 1000, top = 4000, stretching = 1.1)
    zf = collect(0.0:Δz:uniform_top)
    Δ = Float64(Δz)
    while zf[end] < top - 1e-6
        Δ *= stretching
        push!(zf, min(zf[end] + Δ, top))
    end
    return collect(FT, zf)
end

function geometric_faces(FT; Δz₁ = 30, top = 4000, stretching = 1.08)
    zf = [0.0]
    Δ = Float64(Δz₁)
    while zf[end] < top - 1e-6
        push!(zf, min(zf[end] + Δ, top))
        Δ *= stretching
    end
    return collect(FT, zf)
end

test_grids(FT) = (("uniform", uniform_faces(FT)),
                  ("junction", junction_faces(FT)),
                  ("geometric", geometric_faces(FT)))

column_grid(FT, faces) = RectilinearGrid(default_arch, FT; size = length(faces) - 1, z = faces,
                                         topology = (Flat, Flat, Bounded))

column(f) = vec(Array(interior(f)))
cell_centers(faces) = (faces[1:end-1] .+ faces[2:end]) ./ 2

function column_model(FT, faces, microphysics; θ = 300)
    grid = column_grid(FT, faces)
    constants = kessler_constants(FT)
    model = AtmosphereModel(grid; microphysics, thermodynamic_constants = constants)
    set!(model; θ = FT(θ))
    update_state!(model)
    return model
end

# Set the water partial densities from specific values on the reference density. The
# temperature is diagnosed from the scheme's diagnostic mass-fraction fields, which the same
# `update_state!` pass refreshes only after reading them, so two passes are needed for the
# diagnosed temperature to reflect freshly set cloud and rain.
function set_water!(model; qᵛ = nothing, qᶜˡ = nothing, qʳ = nothing)
    Nz = model.grid.Nz
    ρ = column(model.dynamics.reference_state.density)
    isnothing(qᵛ)  || set!(model.moisture_density, reshape(ρ .* qᵛ, 1, 1, Nz))
    isnothing(qᶜˡ) || set!(model.microphysical_fields.ρqᶜˡ, reshape(ρ .* qᶜˡ, 1, 1, Nz))
    isnothing(qʳ)  || set!(model.microphysical_fields.ρqʳ, reshape(ρ .* qʳ, 1, 1, Nz))
    update_state!(model)
    update_state!(model)
    return nothing
end

function kessler_step!(model, Δt)
    model.clock.last_Δt = Δt
    microphysics_model_update!(model.microphysics, model)
    return nothing
end

water_densities(model) = (column(model.moisture_density),
                          column(model.microphysical_fields.ρqᶜˡ),
                          column(model.microphysical_fields.ρqʳ))

rain_inventory(model, Δz) = sum(column(model.microphysical_fields.ρqʳ) .* Δz)
water_inventory(model, Δz) = sum(sum(water_densities(model)) .* Δz)
surface_flux(model) = first(Array(interior(compute!(surface_precipitation_flux(model)))))

saturation_profile(model) = begin
    ρ = column(model.dynamics.reference_state.density)
    T = column(model.temperature)
    constants = model.thermodynamic_constants
    [saturation_specific_humidity(T[k], ρ[k], constants, PlanarLiquidSurface()) for k in eachindex(T)]
end

# Budget tolerance relative to the column inventory: round-off accumulated over cells,
# sedimentation substeps and steps.
budget_rtol(FT) = FT === Float64 ? 1e-11 : 1e-4

rain_profile(z) = 0.002 * exp(-((z - 1500) / 600)^2)

# Sedimentation only: dry column, no cloud, rain evaporation switched off
sedimentation_only_microphysics(FT) =
    DCMIP2016KesslerMicrophysics(FT; evaporation_ventilation_coefficient_1 = 0,
                                     evaporation_ventilation_coefficient_2 = 0)

#####
##### Temperature response to condensation at fixed invariant
#####

@testset "Kessler phase-change temperature slope [$FT]" for FT in all_float_types()
    constants = kessler_constants(FT)
    δr = FT === Float64 ? FT(1e-7) : FT(1e-3)
    rtol = FT === Float64 ? 1e-6 : 5e-3

    # Finite difference of the state's own T(partition) along vapor → liquid at fixed total
    # mixing ratio, compared with the analytic slope.
    function finite_difference_slope(𝒰, rᵛ, rˡ)
        q⁺ = MoistureMassFractions(MoistureMixingRatio(rᵛ - δr, rˡ + δr))
        q⁻ = MoistureMassFractions(MoistureMixingRatio(rᵛ + δr, rˡ - δr))
        return (temperature(with_moisture(𝒰, q⁺), constants) - temperature(with_moisture(𝒰, q⁻), constants)) / (2δr)
    end

    for (rᵛ, rˡ) in ((FT(0.012), FT(0.001)), (FT(0.004), FT(0)), (FT(0.02), FT(0.003)))
        q = MoistureMassFractions(MoistureMixingRatio(rᵛ, rˡ))
        for p in (FT(101325), FT(80000), FT(50000))
            𝒰 = LiquidIcePotentialTemperatureState(FT(300), q, FT(1e5), p)
            @test phase_change_temperature_slope(𝒰, constants) ≈ finite_difference_slope(𝒰, rᵛ, rˡ) rtol=rtol
            # The slope is close to, but not, the DCMIP2016 Fortran value ℒ/cᵖᵈ (the moist heat
            # capacity and the composition dependence of the Exner function are percent-level)
            ℒ_over_cᵖᵈ = constants.liquid.reference_latent_heat / constants.dry_air.heat_capacity
            @test 0 < abs(phase_change_temperature_slope(𝒰, constants) / ℒ_over_cᵖᵈ - 1) < 0.1
        end
        𝒰ˢ = StaticEnergyState(FT(3.1e5), q, FT(1000), FT(90000))
        @test phase_change_temperature_slope(𝒰ˢ, constants) ≈ finite_difference_slope(𝒰ˢ, rᵛ, rˡ) rtol=rtol
    end
end

#####
##### Rain sedimentation: column budget, geometry, positivity
#####

@testset "Kessler rain sedimentation budget [$FT, $name]" for FT in all_float_types(), (name, faces) in test_grids(FT)
    Δz = diff(faces)
    microphysics = sedimentation_only_microphysics(FT)
    model = column_model(FT, faces, microphysics)
    z = cell_centers(faces)
    set_water!(model; qᵛ = zeros(FT, length(z)), qᶜˡ = zeros(FT, length(z)), qʳ = FT.(rain_profile.(z)))

    M₀ = rain_inventory(model, Δz)
    W₀ = water_inventory(model, Δz)
    Δt = FT(20)
    removed = zero(FT)
    for step in 1:10
        kessler_step!(model, Δt)
        F = surface_flux(model)
        @test F > 0
        removed += F * Δt

        # The rain leaving the column through the bottom face is exactly what the column lost
        @test rain_inventory(model, Δz) + removed ≈ M₀ rtol=budget_rtol(FT)
        @test water_inventory(model, Δz) + removed ≈ W₀ rtol=budget_rtol(FT)

        # No negative rain, no vapor or cloud created by sedimentation
        ρqᵛ, ρqᶜˡ, ρqʳ = water_densities(model)
        @test all(ρqʳ .≥ 0)
        @test all(ρqᵛ .== 0)
        @test all(ρqᶜˡ .== 0)
    end
    @test removed > 0.05 * M₀ # the test is not vacuous: a substantial fraction rained out
end

@testset "Kessler sedimentation leaves vapor untouched [$FT]" for FT in all_float_types()
    faces = junction_faces(FT)
    microphysics = sedimentation_only_microphysics(FT)
    model = column_model(FT, faces, microphysics)
    z = cell_centers(faces)
    q⁺ = saturation_profile(model)
    set_water!(model; qᵛ = FT(0.5) .* q⁺, qᶜˡ = zeros(FT, length(z)), qʳ = FT.(rain_profile.(z)))

    ρqᵛ₀ = column(model.moisture_density)
    ρqʳ₀ = column(model.microphysical_fields.ρqʳ)
    kessler_step!(model, FT(20))
    ρqᵛ₁ = column(model.moisture_density)
    ρqʳ₁ = column(model.microphysical_fields.ρqʳ)

    # Rain fell (so the mixing ratios were rescaled by the changing dry-air density) ...
    @test maximum(abs.(ρqʳ₁ .- ρqʳ₀) ./ maximum(ρqʳ₀)) > 1e-3
    # ... but the vapor partial density is the prognostic and is not touched by sedimentation
    @test ρqᵛ₁ ≈ ρqᵛ₀ rtol=(FT === Float64 ? 1e-14 : 1e-6)
end

@testset "Kessler top cell loses rain only through its bottom face [$FT]" for FT in all_float_types()
    faces = collect(FT, 0:100:2000)
    Δz = diff(faces)
    microphysics = sedimentation_only_microphysics(FT)
    model = column_model(FT, faces, microphysics)
    Nz = model.grid.Nz
    qʳ = zeros(FT, Nz)
    qʳ[Nz] = FT(0.001)
    set_water!(model; qᵛ = zeros(FT, Nz), qᶜˡ = zeros(FT, Nz), qʳ)

    before = column(model.microphysical_fields.ρqʳ) .* Δz
    kessler_step!(model, FT(5))
    after = column(model.microphysical_fields.ρqʳ) .* Δz

    lost = before[Nz] - after[Nz]
    gained = after[Nz-1] - before[Nz-1]
    @test lost > 0
    @test lost ≈ gained rtol=budget_rtol(FT)
    @test all(after[1:Nz-2] .== 0)
    @test surface_flux(model) == 0

    # A large step is subcycled so that the top cell, too, is never emptied beyond the CFL
    # fraction of its thickness per substep: no clipping, so the budget still closes.
    set_water!(model; qʳ)
    before = column(model.microphysical_fields.ρqʳ) .* Δz
    kessler_step!(model, FT(200))
    after = column(model.microphysical_fields.ρqʳ) .* Δz
    @test all(after .≥ 0)
    @test sum(after) + surface_flux(model) * FT(200) ≈ sum(before) rtol=budget_rtol(FT)
end

#####
##### All processes: prognostic water budget closes to the surface flux
#####

@testset "Kessler water budget with all processes [$FT, $name]" for FT in all_float_types(), (name, faces) in test_grids(FT)
    Δz = diff(faces)
    microphysics = DCMIP2016KesslerMicrophysics(FT)
    model = column_model(FT, faces, microphysics)
    z = cell_centers(faces)
    q⁺ = saturation_profile(model)

    # Supersaturated cloudy layer over a dry, raining lower column: condensation, cloud
    # evaporation, autoconversion, accretion, rain evaporation and sedimentation all active.
    qᵛ  = [FT(800 < z[k] < 2000 ? 1.02 : 0.6) * q⁺[k] for k in eachindex(z)]
    qᶜˡ = [FT(1000 < z[k] < 1800 ? 0.0015 : 0) for k in eachindex(z)]
    qʳ  = FT.(rain_profile.(z))
    set_water!(model; qᵛ, qᶜˡ, qʳ)

    W₀ = water_inventory(model, Δz)
    ρqᶜˡ₀ = column(model.microphysical_fields.ρqᶜˡ)
    Δt = FT(20)
    removed = zero(FT)
    for step in 1:5
        kessler_step!(model, Δt)
        removed += surface_flux(model) * Δt
        @test water_inventory(model, Δz) + removed ≈ W₀ rtol=budget_rtol(FT)
        ρqᵛ, ρqᶜˡ, ρqʳ = water_densities(model)
        @test all(ρqᵛ .≥ 0)
        @test all(ρqᶜˡ .≥ 0)
        @test all(ρqʳ .≥ 0)
    end
    @test removed > 0
    @test column(model.microphysical_fields.ρqᶜˡ) != ρqᶜˡ₀ # phase change and conversion happened
end

#####
##### Phase change conserves the prognostic invariant θˡⁱ
#####

@testset "Kessler phase change conserves θˡⁱ [$FT]" for FT in all_float_types()
    # Sedimentation off (zero terminal velocity) isolates the phase changes. The column spans
    # ~1000–470 hPa, so every case is exercised across pressures.
    microphysics = DCMIP2016KesslerMicrophysics(FT; terminal_velocity_coefficient = 0)
    faces = collect(FT, 0:250:6000)
    θ_rtol = FT === Float64 ? 1e-12 : 1e-5

    cases = (("condensation",      1.05, 0,    0),
             ("cloud evaporation", 0.9,  5e-4, 0),
             ("rain evaporation",  0.7,  0,    1e-3),
             ("dry",               0.5,  0,    0))

    for (case, saturation_ratio, qᶜˡ₀, qʳ₀) in cases
        model = column_model(FT, faces, microphysics)
        Nz = model.grid.Nz
        q⁺ = saturation_profile(model)
        set_water!(model; qᵛ = FT(saturation_ratio) .* q⁺, qᶜˡ = fill(FT(qᶜˡ₀), Nz), qʳ = fill(FT(qʳ₀), Nz))
        p = column(model.dynamics.reference_state.pressure)
        @test minimum(p) < 50000 < 80000 < maximum(p)

        θ₀ = column(model.formulation.potential_temperature)
        ρqᵛ₀, ρqᶜˡ₀, ρqʳ₀ = water_densities(model)
        T₀ = column(model.temperature)
        kessler_step!(model, FT(20))
        θ₁ = column(model.formulation.potential_temperature)
        ρqᵛ₁, ρqᶜˡ₁, ρqʳ₁ = water_densities(model)
        T₁ = column(model.temperature)

        @testset "$case" begin
            # The invariant is untouched by every phase change (and the microphysics also
            # conserves water exactly, cell by cell, since nothing moves)
            @test θ₁ ≈ θ₀ rtol=θ_rtol
            @test ρqᵛ₁ .+ ρqᶜˡ₁ .+ ρqʳ₁ ≈ ρqᵛ₀ .+ ρqᶜˡ₀ .+ ρqʳ₀ rtol=budget_rtol(FT)
            @test surface_flux(model) == 0

            if case == "dry"
                @test ρqᵛ₁ ≈ ρqᵛ₀ rtol=(FT === Float64 ? 1e-15 : 1e-6)
                @test T₁ ≈ T₀ rtol=(FT === Float64 ? 1e-14 : 1e-6)
            else
                Δρqˡ = (ρqᶜˡ₁ .+ ρqʳ₁) .- (ρqᶜˡ₀ .+ ρqʳ₀)
                @test maximum(abs.(Δρqˡ)) > 1e-6 # a phase change happened in every case
                # Latent heating is implied by the invariant: condensation warms, evaporation cools
                warmed = T₁ .- T₀
                @test all(sign.(warmed[abs.(Δρqˡ) .> 1e-9]) .== sign.(Δρqˡ[abs.(Δρqˡ) .> 1e-9]))
            end

            if case == "condensation"
                # One Newton step of the adjustment, linearized about the invariant, removes the
                # bulk of the supersaturation (measured against the model's refreshed temperature)
                ρ = column(model.dynamics.reference_state.density)
                q⁺₁ = [saturation_specific_humidity(T₁[k], ρ[k], model.thermodynamic_constants, PlanarLiquidSurface()) for k in 1:Nz]
                qᵛ₁ = ρqᵛ₁ ./ ρ
                qᵛ₀ = ρqᵛ₀ ./ ρ
                @test maximum(abs.(qᵛ₁ .- q⁺₁) ./ q⁺₁) < 0.1 * maximum(abs.(qᵛ₀ .- q⁺) ./ q⁺)
            end
        end
    end
end

#####
##### Negative inputs: clipped on entry, a documented source, not covered by the budgets above
#####

@testset "Kessler clips negative inputs on entry [$FT]" for FT in all_float_types()
    # Advection can hand the kernel negative partial densities. The kernel clips them to zero
    # before doing anything else (as the DCMIP2016 Fortran does), which *creates* exactly the
    # clipped mass. The budget tests above therefore hold for non-negative states only; this
    # test pins the size of the source so it is not mistaken for closure.
    faces = junction_faces(FT)
    Δz = diff(faces)
    microphysics = sedimentation_only_microphysics(FT)
    model = column_model(FT, faces, microphysics)
    z = cell_centers(faces)
    Nz = length(z)
    # A dry, raining column keeps the inventory small, so the created mass stands out from the
    # round-off of the inventory difference at either precision
    set_water!(model; qᵛ = zeros(FT, Nz), qᶜˡ = zeros(FT, Nz), qʳ = FT.(rain_profile.(z)))

    # Negative vapor in one cell and negative rain in another, written straight into the
    # prognostics (bypassing any negative-moisture correction of `update_state!`)
    ρ = column(model.dynamics.reference_state.density)
    ρqᵛ = column(model.moisture_density)
    ρqʳ = column(model.microphysical_fields.ρqʳ)
    ρqᵛ[5] = -FT(2e-4) * ρ[5]
    ρqʳ[Nz-3] = -FT(4e-5) * ρ[Nz-3]
    set!(model.moisture_density, reshape(ρqᵛ, 1, 1, Nz))
    set!(model.microphysical_fields.ρqʳ, reshape(ρqʳ, 1, 1, Nz))
    clipped = -(ρqᵛ[5] * Δz[5] + ρqʳ[Nz-3] * Δz[Nz-3])

    # The injected signal must stand well clear of the round-off floor of the inventory sums,
    # eps(FT) × W₀ × Nz, so that the bound below (relative to the created mass) is meaningful
    W₀ = water_inventory(model, Δz)
    @test clipped / (eps(FT) * W₀ * Nz) ≥ 1000

    Δt = FT(20)
    kessler_step!(model, Δt)
    W₁ = water_inventory(model, Δz)
    ρqᵛ₁, ρqᶜˡ₁, ρqʳ₁ = water_densities(model)

    @test all(ρqᵛ₁ .≥ 0)
    @test all(ρqʳ₁ .≥ 0)
    @test ρqᵛ₁[5] == 0
    # The water created is exactly the clipped mass, nothing more
    @test W₁ + surface_flux(model) * Δt - W₀ ≈ clipped rtol=budget_rtol(FT)
end

#####
##### Unsupported formulation fails clearly
#####

@testset "Kessler rejects the static energy formulation" begin
    FT = Float64
    faces = collect(FT, 0:500:2000)
    grid = column_grid(FT, faces)
    model = AtmosphereModel(grid; formulation = :StaticEnergy,
                            microphysics = DCMIP2016KesslerMicrophysics(FT),
                            thermodynamic_constants = kessler_constants(FT))
    model.clock.last_Δt = FT(10)
    @test_throws ArgumentError microphysics_model_update!(model.microphysics, model)
end
