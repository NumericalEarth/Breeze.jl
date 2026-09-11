include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze: DCMIP2016KesslerMicrophysics, TetensFormula
using Breeze.AtmosphereModels: compute_forcing!
using Breeze.TurbulenceClosures: static_stabilityᶜᶜᶠ, nonprecipitating_water_mixing_ratioᶜᶜᶜ
using Oceananigans: Oceananigans, Flat, Bounded, Center, Face, Field, set!, interior, UpwindBiased
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.TurbulenceClosures: buoyancy_force, buoyancy_tracers
using Test

@testset "Kessler condensate subsidence and rain stability [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    rtol = max(1e-10, 100eps(FT))
    grid = RectilinearGrid(default_arch; size = 12, z = (0, 1200), topology = (Flat, Flat, Bounded))
    constants = ThermodynamicConstants(FT; saturation_vapor_pressure = TetensFormula(FT))
    microphysics = DCMIP2016KesslerMicrophysics(FT)
    subsidence = SubsidenceForcing(z -> FT(-0.01); advection = UpwindBiased(order=1))
    model = AtmosphereModel(grid; microphysics, thermodynamic_constants = constants, advection = nothing,
                            forcing = (qᶜˡ = subsidence, qʳ = subsidence))
    fields = Oceananigans.fields(model)
    density = model.dynamics.reference_state.density
    # Constant mass fractions must have no subsidence tendency despite varying density.
    set!(model; θ = 290, qᵛ = 0.002, qᶜˡ = 0.001, qʳ = 0.002)
    for name in (:ρqᶜˡ, :ρqʳ)
        forcing = model.forcing[name]
        compute_forcing!(forcing)
        tendency = Field(KernelFunctionOperation{Center, Center, Center}(forcing, grid, model.clock, fields))
        @test maximum(abs, Array(interior(tendency))) < max(1e-15, 10eps(FT) * 1e-6)
    end
    # A known linear mass-fraction gradient gives -ρ w ∂z q, once per density factor.
    slope = FT(1e-7)
    set!(model; qᶜˡ = z -> FT(0.001) + slope * z, qʳ = z -> FT(0.002) + 2slope * z)
    for (name, gradient) in ((:ρqᶜˡ, slope), (:ρqʳ, 2slope))
        forcing = model.forcing[name]
        compute_forcing!(forcing)
        tendency = Field(KernelFunctionOperation{Center, Center, Center}(forcing, grid, model.clock, fields))
        normalized = Array(interior(tendency)) ./ Array(interior(density))
        @test all(isapprox.(normalized[:, :, 2:11], 0.01gradient; rtol))
    end

    # A deliberately rainy, subsaturated state: rain exceeds the saturation deficit,
    # but cannot supply cloud condensate to an equilibrating displacement.
    set!(model; T = 290, qᵛ = 0.002, qᶜˡ = 0, qʳ = 0.02)
    buoyancy, tracers = buoyancy_force(model), buoyancy_tracers(model)
    stability(s) = Array(interior(Field(KernelFunctionOperation{Center, Center, Face}(
                                      static_stabilityᶜᶜᶠ, grid, s, buoyancy, tracers))))
    @test stability(MoistStaticStability())[:, :, 2:12] == stability(DryStaticStability())[:, :, 2:12]
    ratio = Field(KernelFunctionOperation{Center, Center, Center}(
                  nonprecipitating_water_mixing_ratioᶜᶜᶜ, grid, buoyancy, tracers.T, tracers.qᵛ))
    @test all(isapprox.(Array(interior(ratio)), 0.002 / (1 - 0.002 - 0.02); rtol))
end
