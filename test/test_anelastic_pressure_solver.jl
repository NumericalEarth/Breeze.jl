include(joinpath(@__DIR__, "runtests_setup.jl"))

using Test
using Breeze
using Breeze.AtmosphereModels: AtmosphereModel, AnelasticFormulation, solve_for_anelastic_pressure!
using Breeze.Thermodynamics: AtmosphereThermodynamics, ReferenceStateConstants
using Oceananigans
using Oceananigans.Grids: RectilinearGrid, Center, znodes
using Oceananigans.Fields: fill_halo_regions!, interior
using Oceananigans.Models.NonhydrostaticModels: NonhydrostaticModel, solve_for_pressure!

@testset "Matches NonhydrostaticModel when rho_ref == 1" begin
    FT = Float64
    grid = RectilinearGrid(FT; size=(4, 4, 4), x=(0, 1), y=(0, 1), z=(0, 1))
    thermo = AtmosphereThermodynamics(FT)
    constants = ReferenceStateConstants(FT; base_pressure=101325.0, potential_temperature=288.0)

    formulation = AnelasticFormulation(grid, constants, thermo)
    set!(formulation.reference_density, FT(1))
    fill_halo_regions!(formulation.reference_density)

    atmos = AtmosphereModel(grid; thermodynamics=thermo, formulation=formulation, tracers=())
    nonhydro = NonhydrostaticModel(; grid, tracers=())

    rho_u = atmos.momentum.ρu
    rho_v = atmos.momentum.ρv
    rho_w = atmos.momentum.ρw
    u = nonhydro.velocities.u
    v = nonhydro.velocities.v
    w = nonhydro.velocities.w

    f1(x, y, z) = sinpi(x) * cospi(z)
    f2(x, y, z) = cospi(y) * sinpi(z)
    f3(x, y, z) = sinpi(x) * sinpi(y)

    set!(rho_u, f1); set!(u, f1)
    set!(rho_v, f2); set!(v, f2)
    set!(rho_w, f3); set!(w, f3)

    fill_halo_regions!(rho_u); fill_halo_regions!(rho_v); fill_halo_regions!(rho_w)
    fill_halo_regions!(u); fill_halo_regions!(v); fill_halo_regions!(w)

    delta_t = FT(0.1)
    solve_for_anelastic_pressure!(atmos.nonhydrostatic_pressure, atmos.pressure_solver, atmos.momentum, delta_t)
    solve_for_pressure!(nonhydro.pressures.pNHS, nonhydro.pressure_solver, delta_t, nonhydro.velocities)

    p_anelastic = interior(atmos.nonhydrostatic_pressure)
    p_nonhydro = interior(nonhydro.pressures.pNHS)

    difference = maximum(abs.(p_anelastic .- p_nonhydro ./ delta_t))
    @test difference < 1e-11
end

@testset "Recovers analytic 1D solution with variable rho_ref" begin
    FT = Float64
    grid = RectilinearGrid(FT; size=(1, 1, 48), x=(0, 1), y=(0, 1), z=(0, 1))
    thermo = AtmosphereThermodynamics(FT)
    constants = ReferenceStateConstants(FT; base_pressure=101325.0, potential_temperature=288.0)

    formulation = AnelasticFormulation(grid, constants, thermo)
    rho_ref(z) = 1 + 0.5 * cospi(z)
    set!(formulation.reference_density, z -> rho_ref(z))
    fill_halo_regions!(formulation.reference_density)

    model = AtmosphereModel(grid; thermodynamics=thermo, formulation=formulation, tracers=())

    set!(model.momentum.ρu, FT(0))
    set!(model.momentum.ρv, FT(0))

    phi(z) = cospi(z)
    dphidz(z) = -pi * sinpi(z)
    set!(model.momentum.ρw, (x, y, z) -> rho_ref(z) * dphidz(z))
    fill_halo_regions!(model.momentum.ρw)

    delta_t = FT(1)
    solve_for_anelastic_pressure!(model.nonhydrostatic_pressure, model.pressure_solver, model.momentum, delta_t)

    zs = collect(znodes(grid, Center(), Center(), Center()))
    phi_exact = phi.(zs)
    phi_numeric = vec(interior(model.nonhydrostatic_pressure))

    error = maximum(abs.(phi_numeric .- phi_exact))
    @test error < 5e-4
end
