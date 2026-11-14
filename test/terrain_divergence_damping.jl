include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.CompressibleEquations: apply_divergence_damping!, _apply_direct_divergence_damping!, dρθ′
using Oceananigans
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ
using Oceananigans.Utils: launch!
using Test

@testset "Terrain divergence damping" begin
    FT = Oceananigans.defaults.FloatType

    @testset "$(typeof(formulation))" for formulation in
        (LinearDecay(), TwoLevelDecay(large_scale_height=4, small_scale_height=2))

        Nx, Ny, Nz = 8, 8, 6
        z = TerrainFollowingVerticalDiscretization(FT.(0:Nz); formulation)
        grid = RectilinearGrid(default_arch, FT; size=(Nx, Ny, Nz), halo=(3, 3, 3),
                               x=(0, Nx), y=(0, Ny), z, topology=(Periodic, Periodic, Bounded))
        materialize_terrain!(grid, (x, y) -> (sinpi(2x / Nx) + sinpi(2y / Ny)) / 4)

        substepper = AcousticSubstepper(grid, SplitExplicitTimeDiscretization())
        ρu′, ρv′ = substepper.momentum_perturbation.u, substepper.momentum_perturbation.v
        ρθ′ = substepper.density_potential_temperature_perturbation
        ρθ′ˢ⁻ = substepper.previous_density_potential_temperature_perturbation
        θᴸ = substepper.linearization_potential_temperature
        δ = substepper.density_predictor
        fill!(parent(θᴸ), 1)

        damping = substepper.damping
        α = FT(damping.coefficient)
        tolerance = 100 * eps(FT)

        @testset "Physical horizontal gradient = $a" for a in (0, 1)
            # The damping argument is z + a*x + 2a*y. Its constant-height gradients
            # are a and 2a, although its coordinate-surface gradients include the slope.
            set!(ρθ′, (x, y, z) -> 3z + a * x + 2a * y)
            set!(ρθ′ˢ⁻, (x, y, z) -> 2z)
            set!(δ, (x, y, z) -> z + a * x + 2a * y)
            fill_halo_regions!((ρθ′, ρθ′ˢ⁻, δ))

            # Preserve the established discrete gradients for both formulations. TwoLevelDecay
            # uses Jacobian-based vertical spacings, so its nonlinear coordinate map leaves a
            # truncation error even for a field linear in physical height.
            thermal_gradient_x = Field(KernelFunctionOperation{Face, Center, Center}(∂xᶠᶜᶜ, grid, dρθ′, ρθ′, ρθ′ˢ⁻))
            thermal_gradient_y = Field(KernelFunctionOperation{Center, Face, Center}(∂yᶜᶠᶜ, grid, dρθ′, ρθ′, ρθ′ˢ⁻))
            direct_gradient_x = Field(KernelFunctionOperation{Face, Center, Center}(∂xᶠᶜᶜ, grid, δ))
            direct_gradient_y = Field(KernelFunctionOperation{Center, Face, Center}(∂yᶜᶠᶜ, grid, δ))

            # z is periodic horizontally, so also check the periodic seams for a = 0.
            # Exclude the vertical boundary cells, whose halos impose zero normal gradients.
            indices = (a == 0 ? (1:Nx) : (2:Nx), a == 0 ? (1:Ny) : (2:Ny), 2:Nz-1)
            set!(ρu′, 0)
            set!(ρv′, 0)
            apply_divergence_damping!(damping, substepper, grid, one(FT), nothing)
            @test Array(interior(ρu′)) ≈ -α .* Array(interior(thermal_gradient_x))
            @test Array(interior(ρv′)) ≈ -α .* Array(interior(thermal_gradient_y))
            if formulation isa LinearDecay
                @test maximum(abs, Array(interior(ρu′))[indices...] .+ α * a) < tolerance
                @test maximum(abs, Array(interior(ρv′))[indices...] .+ 2α * a) < tolerance
            end

            set!(ρu′, 0)
            set!(ρv′, 0)
            launch!(default_arch, grid, :xyz, _apply_direct_divergence_damping!,
                    ρu′, ρv′, grid, δ, θᴸ, α)
            @test Array(interior(ρu′)) ≈ α .* Array(interior(direct_gradient_x))
            @test Array(interior(ρv′)) ≈ α .* Array(interior(direct_gradient_y))
            if formulation isa LinearDecay
                @test maximum(abs, Array(interior(ρu′))[indices...] .- α * a) < tolerance
                @test maximum(abs, Array(interior(ρv′))[indices...] .- 2α * a) < tolerance
            end
        end

        @testset "Direct damping fills the divergence halos" begin
            set!(ρu′, (x, y, z) -> z * sinpi(2x / Nx))
            set!(ρv′, (x, y, z) -> z * cospi(2y / Ny))
            fill_halo_regions!((ρu′, ρv′))
            initial_u = Array(interior(ρu′))
            initial_v = Array(interior(ρv′))
            fill!(parent(δ), NaN)

            apply_divergence_damping!(DirectDivergenceDamping(coefficient=α), substepper, grid, one(FT), nothing)
            @test all(isfinite, Array(interior(ρu′)))
            @test all(isfinite, Array(interior(ρv′)))

            # Build the reference after a full halo fill, independently of the substep policy.
            fill_halo_regions!(δ)
            gradient_x = Field(KernelFunctionOperation{Face, Center, Center}(∂xᶠᶜᶜ, grid, δ))
            gradient_y = Field(KernelFunctionOperation{Center, Face, Center}(∂yᶜᶠᶜ, grid, δ))
            @test Array(interior(ρu′)) ≈ initial_u .+ α .* Array(interior(gradient_x))
            @test Array(interior(ρv′)) ≈ initial_v .+ α .* Array(interior(gradient_y))
        end
    end
end
