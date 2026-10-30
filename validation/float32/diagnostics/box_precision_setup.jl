# Droplet box of test/lagrangian_microphysics.jl with precision options
quasirandom(n, k=1) = mod(n * sqrt(2 + k), 1)
function box(FT; thermodynamic_float_type=FT, self_interaction=DiffusiveSelfInteraction(), velocity=true)
    L = FT(0.02); Nd = 200
    grid = RectilinearGrid(GPU(), FT; size=(6, 6, 6), x=(0, L), y=(0, L), z=(0, L), topology=(Periodic, Periodic, Periodic))
    dynamics = BoussinesqDynamics(grid; pressure=90000, temperature=281)
    n = 1:Nd
    droplets = (x = L .* quasirandom.(n, 1), y = L .* quasirandom.(n, 2), z = L .* quasirandom.(n, 3),
                r = fill(5e-6, Nd), rᵈ = fill(5e-8, Nd), κ = fill(0.6, Nd))
    microphysics = LagrangianMicrophysics(FT; droplets, self_interaction)
    model = AtmosphereModel(grid; dynamics, microphysics, thermodynamic_float_type)
    constants = model.thermodynamic_constants
    pᵛ = 1.005 * saturation_vapor_pressure(281.0, ThermodynamicConstants(Float64), PlanarLiquidSurface())
    ε = 287.0 / 461.5
    ε = dry_air_gas_constant(ThermodynamicConstants(Float64)) / vapor_gas_constant(ThermodynamicConstants(Float64))
    qᵛ₀ = ε * pᵛ / (90000 - (1 - ε) * pᵛ)
    if velocity
        set!(model; T=281, qᵗ=qᵛ₀, u=(x, y, z) -> sin(2π * x / L) / 20, v=(x, y, z) -> -cos(2π * y / L) / 20)
    else
        set!(model; T=281, qᵗ=qᵛ₀)
    end
    return model
end

