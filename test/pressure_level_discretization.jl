using Breeze
using Oceananigans
using Oceananigans.Architectures: on_architecture
using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ, Δzᶠᶜᶜ, Δzᶜᶠᶜ, Δz⁻¹ᶜᶜᶜ
using Oceananigans.Grids: znode
using Test
using Breeze.HydrostaticEquations: mass_flux_x, mass_flux_y, cell_mass, XMomentumDensity, YMomentumDensity,
                                   x_velocity_cell_mass, x_face_mass, layer_mass_per_area
using Breeze.Advection: tracer_mass_flux_x, tracer_mass_flux_y
using Oceananigans.Operators: Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ, Azᶜᶜᶜ
using Oceananigans.Advection: transport_x, transport_y
using Oceananigans.BoundaryConditions: fill_halo_regions!

using Breeze.PressureLevelDiscretization: PressureLevelVerticalDiscretization, PressureLevelGrid,
                                          layer_pressure_thickness, interface_pressure, layer_mass

# Hybrid coefficients of Nz layers with a top at 200 Pa: pure pressure above η = 0.2, terrain following below
function hybrid_coefficients(Nz; top_pressure = 200.0, p₀ = 1e5)
    η = range(1, top_pressure / p₀, length = Nz + 1)
    B = [max(0, (e - 0.2) / 0.8) for e in η]
    A = [e * p₀ - b * p₀ for (e, b) in zip(η, B)]
    A[1] = 0; B[1] = 1; B[end] = 0; A[end] = top_pressure
    return A, B
end

@testset "PressureLevelVerticalDiscretization [$FT]" for FT in (Float64, Float32)
    Nz = 8
    A, B = hybrid_coefficients(Nz)
    z = PressureLevelVerticalDiscretization(A, B)
    grid = RectilinearGrid(CPU(), FT; size = (4, 3, Nz), halo = (3, 3, 3), x = (0, 1e5), y = (0, 1e5), z,
                           topology = (Periodic, Periodic, Bounded))

    @test grid isa PressureLevelGrid

    # A surface pressure that varies in x and y; a density that varies in z
    pˢ = grid.z.surface_pressure
    for j in axes(pˢ, 2), i in axes(pˢ, 1)
        pˢ[i, j, 1] = 9.6e4 + 500 * i - 300 * j
    end
    ρ = grid.z.density
    for k in axes(ρ, 3), j in axes(ρ, 2), i in axes(ρ, 1)
        ρ[i, j, k] = 1.2 * exp(-(k - 0.5) / 6) + 0.01 * i
    end

    g = grid.z.gravitational_acceleration
    top = A[end]

    @testset "layer mass Δp / g" begin
        for j in 1:3, i in 1:4
            @test all(layer_mass(i, j, k, grid) == layer_pressure_thickness(i, j, k, grid) / g for k in 1:Nz)
            # g Σ Δp / g telescopes to pˢ - p_top up to the rounding of Nz additions
            column_weight = g * sum(layer_mass(i, j, k, grid) for k in 1:Nz)
            @test isapprox(column_weight, pˢ[i, j, 1] - top; rtol = 4Nz * eps(FT))
        end
    end

    @testset "Σ Δp = pˢ - p_top and Δp > 0" begin
        for j in 1:3, i in 1:4
            Δp = [layer_pressure_thickness(i, j, k, grid) for k in 1:Nz]
            @test all(>(0), Δp)
            @test sum(Δp) ≈ pˢ[i, j, 1] - top rtol = 10eps(FT)
            @test interface_pressure(i, j, 1, grid) == pˢ[i, j, 1]
            @test interface_pressure(i, j, Nz + 1, grid) ≈ top
            # Telescoping: each thickness is the difference of the two face pressures
            for k in 1:Nz
                @test Δp[k] ≈ interface_pressure(i, j, k, grid) - interface_pressure(i, j, k + 1, grid) rtol = 10eps(FT)
            end
        end
    end

    @testset "Δz = Δp / (g ρ) > 0, also in the halos" begin
        for j in 0:4, i in 0:5, k in 0:Nz+1
            @test Δzᶜᶜᶜ(i, j, k, grid) > 0
        end
        for j in 1:3, i in 1:4, k in 1:Nz
            @test Δzᶜᶜᶜ(i, j, k, grid) ≈ layer_pressure_thickness(i, j, k, grid) / (g * ρ[i, j, k]) rtol = 10eps(FT)
            @test ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid) ≈ layer_pressure_thickness(i, j, k, grid) / g rtol = 10eps(FT)
            @test Δz⁻¹ᶜᶜᶜ(i, j, k, grid) ≈ 1 / Δzᶜᶜᶜ(i, j, k, grid)
            @test Δzᶠᶜᶜ(i, j, k, grid) ≈ (Δzᶜᶜᶜ(i - 1, j, k, grid) + Δzᶜᶜᶜ(i, j, k, grid)) / 2
            @test Δzᶜᶠᶜ(i, j, k, grid) ≈ (Δzᶜᶜᶜ(i, j - 1, k, grid) + Δzᶜᶜᶜ(i, j, k, grid)) / 2
        end
        for j in 1:3, i in 1:4, k in 2:Nz
            @test Δzᶜᶜᶠ(i, j, k, grid) ≈ (Δzᶜᶜᶜ(i, j, k - 1, grid) + Δzᶜᶜᶜ(i, j, k, grid)) / 2
        end
    end

    @testset "Heights" begin
        for j in 1:3, i in 1:4
            @test znode(i, j, 1, grid, Center(), Center(), Face()) == 0
            ztop = znode(i, j, Nz + 1, grid, Center(), Center(), Face())
            @test ztop ≈ sum(Δzᶜᶜᶜ(i, j, k, grid) for k in 1:Nz) rtol = 10eps(FT)
            zc = [znode(i, j, k, grid, Center(), Center(), Center()) for k in 1:Nz]
            @test issorted(zc)
        end
    end

    @testset "The metrics follow the surface pressure" begin
        i, j = 2, 2
        before = [layer_pressure_thickness(i, j, k, grid) for k in 1:Nz]
        depth = Δzᶜᶜᶜ(i, j, 1, grid)
        pˢ[i, j, 1] -= 1000
        after = [layer_pressure_thickness(i, j, k, grid) for k in 1:Nz]
        @test sum(before) - sum(after) ≈ 1000 rtol = 100eps(FT)
        @test Δzᶜᶜᶜ(i, j, 1, grid) < depth
        @test after[end] ≈ before[end]   # the pure-pressure layers do not move
    end

    @testset "Rebuilding the grid keeps the coefficients and the state" begin
        cpu_grid = on_architecture(CPU(), grid)
        @test cpu_grid isa PressureLevelGrid
        for j in 1:3, i in 1:4, k in 1:Nz
            @test Δzᶜᶜᶜ(i, j, k, cpu_grid) == Δzᶜᶜᶜ(i, j, k, grid)
        end
    end

    # Every flux of the dynamics through a horizontal face is the one mass flux of continuity,
    # written in Δp / g (step 2.2). These are identities of the same floating-point expression.
    @testset "Mass-consistent operators" begin
        ρ = grid.z.density
        u = XFaceField(grid); v = YFaceField(grid); c = CenterField(grid)
        set!(u, (x, y, z) -> 3 + sin(2π * x / 1e5))
        set!(v, (x, y, z) -> -2 + cos(2π * y / 1e5))
        set!(c, 1)
        fill_halo_regions!((u, v, c))
        ρu = XMomentumDensity(ρ, u); ρv = YMomentumDensity(ρ, v)

        for scheme in (Centered(order = 2), UpwindBiased(order = 1)), j in 1:3, i in 1:4, k in 1:Nz
            Fx = mass_flux_x(i, j, k, grid, ρ, u)
            Fy = mass_flux_y(i, j, k, grid, ρ, v)
            @test Ax_qᶠᶜᶜ(i, j, k, grid, ρu) == Fx
            @test Ay_qᶜᶠᶜ(i, j, k, grid, ρv) == Fy
            @test transport_x(i, j, k, grid, scheme, ρ, u) == Fx
            @test transport_y(i, j, k, grid, scheme, ρ, v) == Fy
            # A uniform c = 1 is reconstructed exactly by these schemes
            @test tracer_mass_flux_x(i, j, k, grid, ρ, scheme, u, c) == Fx
            @test tracer_mass_flux_y(i, j, k, grid, ρ, scheme, v, c) == Fy
        end

        for j in 1:3, i in 1:4, k in 1:Nz
            @test cell_mass(i, j, k, grid, ρ) == Azᶜᶜᶜ(i, j, k, grid) * layer_mass(i, j, k, grid)
            @test layer_mass_per_area(i, j, k, grid, ρ) == layer_mass(i, j, k, grid)
            @test x_velocity_cell_mass(i, j, k, grid, ρ) == x_face_mass(i, j, k, grid, ρ)
        end
    end

    @testset "Construction errors" begin
        @test_throws ArgumentError PressureLevelVerticalDiscretization(A[2:end], B[2:end])
        @test_throws ArgumentError PressureLevelVerticalDiscretization(A, B[1:end-1])
        bad_B = copy(B); bad_B[end] = 0.1
        @test_throws ArgumentError PressureLevelVerticalDiscretization(A, bad_B)
        @test_throws ArgumentError RectilinearGrid(CPU(), FT; size = (4, 3, Nz + 1), x = (0, 1), y = (0, 1), z,
                                                   topology = (Periodic, Periodic, Bounded))
    end
end
