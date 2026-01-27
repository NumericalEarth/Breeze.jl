#####
##### Oceananigans differentiation tests (without Breeze)
##### Tests Reactant/Enzyme autodiff with HydrostaticFreeSurfaceModel
#####

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface
using Reactant
using Enzyme
using Statistics: mean
using Test
using CUDA

Reactant.set_default_backend("cpu")

@testset "Oceananigans HydrostaticFreeSurfaceModel differentiation - Bounded topology" begin
    grid = RectilinearGrid(ReactantState(); size=(40, 40, 1), extent=(1000, 1000, 100),
                           halo=(3, 3, 3), topology=(Bounded, Bounded, Bounded))

    free_surface = ExplicitFreeSurface()
    model = HydrostaticFreeSurfaceModel(grid, free_surface=free_surface, tracers=:c)
    dmodel = Enzyme.make_zero(model)

    c_init = CenterField(grid)
    set!(c_init, (x, y, z) -> 0.01 * x + 0.01 * y)
    dc_init = CenterField(grid)
    set!(dc_init, 0.0)

    function loss(model, c_init, Δt, nsteps)
        set!(model, c=c_init)
        @trace mincut=true checkpointing=false track_numbers=false for i in 1:nsteps
            time_step!(model, Δt)
        end
        return mean(interior(model.tracers.c).^2)
    end

    function grad_loss(model, dmodel, c_init, dc_init, Δt, nsteps)
        parent(dc_init) .= 0
        _, loss_value = Enzyme.autodiff(
            Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
            loss, Enzyme.Active,
            Enzyme.Duplicated(model, dmodel),
            Enzyme.Duplicated(c_init, dc_init),
            Enzyme.Const(Δt),
            Enzyme.Const(nsteps))
        return dc_init, loss_value
    end

    Δt = 0.01
    nsteps = 10

    @testset "Compilation succeeds" begin
        compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss(
            model, dmodel, c_init, dc_init, Δt, nsteps)
        @test compiled !== nothing
    end

    @testset "Gradient computation" begin
        compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss(
            model, dmodel, c_init, dc_init, Δt, nsteps)
        dc, loss_val = compiled(model, dmodel, c_init, dc_init, Δt, nsteps)

        @test loss_val > 0
        @test !isnan(loss_val)
        @test maximum(abs, interior(dc)) > 0  # Gradient should be non-zero
        @test !any(isnan, interior(dc))
    end
end
