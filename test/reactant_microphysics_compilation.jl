#####
##### Reactant compilation + autodiff tests — CompressibleDynamics + CloudMicrophysics
#####
#
# Forward raise and reverse-mode AD through CompressibleDynamics AtmosphereModels
# with different CloudMicrophysics schemes on a RectilinearGrid with
# (Periodic, Bounded, Bounded) topology and Centered advection.
#
# Tests:
#   1. OneMomentCloudMicrophysics — mixed-phase non-equilibrium (MPNE1M)
#   2. TwoMomentCloudMicrophysics — Seifert-Beheng 2006 warm-phase NE + aerosol activation

using Breeze
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudIce
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean
using Test
using CUDA

using Breeze.Microphysics: NonEquilibriumCloudFormation

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics, TwoMomentCloudMicrophysics

#####
##### Loss and gradient helpers
#####

function loss_1m(model, θ_init, Δt, Nsteps)
    set!(model; θ=θ_init, ρ=1.0, qᵗ=0.02, qᶜˡ=1e-3, qᶜⁱ=1e-5, qʳ=1e-4)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return mean(interior(model.temperature) .^ 2)
end

function grad_loss_1m(model, dmodel, θ_init, dθ_init, Δt, Nsteps)
    parent(dθ_init) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_1m, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(Δt),
        Enzyme.Const(Nsteps))
    return dθ_init, loss_value
end

function loss_2m(model, θ_init, Δt, Nsteps)
    set!(model; θ=θ_init, ρ=1.0, qᵗ=0.02, qᶜˡ=1e-3, nᶜˡ=100e6, qʳ=1e-4, nʳ=1e4)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return mean(interior(model.temperature) .^ 2)
end

function grad_loss_2m(model, dmodel, θ_init, dθ_init, Δt, Nsteps)
    parent(dθ_init) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_2m, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(Δt),
        Enzyme.Const(Nsteps))
    return dθ_init, loss_value
end

#####
##### 1M mixed-phase non-equilibrium (MPNE1M)
#####

@testset "Reactant forward raise — CompressibleDynamics + 1M mixed-phase NE microphysics" begin
    topo = (Periodic, Bounded, Bounded)

    @info "Building grid: RectilinearGrid 4×4×4, topology=$topo"
    @time "Constructing grid" grid = RectilinearGrid(ReactantState();
        size=(4, 4, 4), extent=(1e3, 1e3, 1e3), topology=topo)

    @info "Setting up microphysics: OneMomentCloudMicrophysics (mixed-phase NE)"
    microphysics = OneMomentCloudMicrophysics(Float64;
        cloud_formation = NonEquilibriumCloudFormation(nothing, CloudIce(Float64)))

    @info "Building AtmosphereModel: CompressibleDynamics + Centered(order=2)"
    @time "Constructing model" model = AtmosphereModel(grid;
        dynamics    = CompressibleDynamics(),
        advection   = Centered(order=2),
        microphysics)

    @testset "Build" begin
        @test model isa AtmosphereModel
        @test model.dynamics isa CompressibleDynamics
        @test haskey(model.microphysical_fields, :ρqᶜˡ)
        @test haskey(model.microphysical_fields, :ρqᶜⁱ)
        @test haskey(model.microphysical_fields, :ρqʳ)
        @test haskey(model.microphysical_fields, :ρqˢ)
    end

    Δt = 0.01
    Ns = 1

    @info "Constructing θ_init field"
    @time "Constructing θ_init" begin
        θ_init = CenterField(grid)
        set!(θ_init, (x, y, z) -> 300.0)
    end

    @testset "Raise forward" begin
        @info "Compiling loss_1m (raise=true, raise_first=true, sync=true)..."
        @time "Compiling loss_1m" compiled_loss = Reactant.@compile optimize=true raise=true raise_first=true sync=true loss_1m(
            model, θ_init, Δt, Ns)

        @info "Running compiled loss_1m..."
        @time "Running compiled loss_1m" loss_val = compiled_loss(model, θ_init, Δt, Ns)

        @info "Result: loss = $loss_val"

        @test loss_val > 0
        @test isfinite(loss_val)
    end

    @testset "Raise backward (autodiff)" begin
        @info "Creating shadow model (make_zero)..."
        @time "Creating shadow model" dmodel = Enzyme.make_zero(model)

        @info "Creating shadow θ field..."
        @time "Creating shadow θ" begin
            dθ_init = CenterField(grid)
            set!(dθ_init, 0.0)
        end

        @info "Compiling grad_loss_1m (raise=true, raise_first=true, sync=true)..."
        @time "Compiling grad_loss_1m" compiled_grad = Reactant.@compile raise_first=true raise=true sync=true grad_loss_1m(
            model, dmodel, θ_init, dθ_init, Δt, Ns)

        @info "Running compiled grad_loss_1m..."
        @time "Running compiled grad_loss_1m" dθ, loss_val = compiled_grad(
            model, dmodel, θ_init, dθ_init, Δt, Ns)

        @info "Result: loss = $loss_val, max|∇θ| = $(maximum(abs, interior(dθ)))"

        @test loss_val > 0
        @test isfinite(loss_val)
        @test maximum(abs, interior(dθ)) > 0
        @test !any(isnan, interior(dθ))
    end
end

#####
##### 2M warm-phase NE (Seifert-Beheng 2006 + aerosol activation)
#####

@testset "Reactant forward raise — CompressibleDynamics + 2M warm-rain microphysics (PBB)" begin
    topo = (Periodic, Bounded, Bounded)

    @info "Building grid: RectilinearGrid 4×4×4, topology=$topo"
    @time "Constructing grid" grid = RectilinearGrid(ReactantState();
        size=(4, 4, 4), extent=(1e3, 1e3, 1e3), topology=topo)

    @info "Setting up microphysics: TwoMomentCloudMicrophysics (SB2006 warm-phase NE + aerosol activation)"
    microphysics = TwoMomentCloudMicrophysics(Float64)

    @info "Building AtmosphereModel: CompressibleDynamics + Centered(order=2)"
    @time "Constructing model" model = AtmosphereModel(grid;
        dynamics    = CompressibleDynamics(),
        advection   = Centered(order=2),
        microphysics)

    @testset "Build" begin
        @test model isa AtmosphereModel
        @test model.dynamics isa CompressibleDynamics
        @test haskey(model.microphysical_fields, :ρqᶜˡ)
        @test haskey(model.microphysical_fields, :ρnᶜˡ)
        @test haskey(model.microphysical_fields, :ρqʳ)
        @test haskey(model.microphysical_fields, :ρnʳ)
        @test haskey(model.microphysical_fields, :ρnᵃ)
    end

    Δt = 0.01
    Ns = 1

    @info "Constructing θ_init field"
    @time "Constructing θ_init" begin
        θ_init = CenterField(grid)
        set!(θ_init, (x, y, z) -> 300.0)
    end

    @testset "Raise forward" begin
        @info "Compiling loss_2m (raise=true, raise_first=true, sync=true)..."
        @time "Compiling loss_2m" compiled_loss = Reactant.@compile optimize=true raise=true raise_first=true sync=true loss_2m(
            model, θ_init, Δt, Ns)

        @info "Running compiled loss_2m..."
        @time "Running compiled loss_2m" loss_val = compiled_loss(model, θ_init, Δt, Ns)

        @info "Result: loss = $loss_val"

        @test loss_val > 0
        @test isfinite(loss_val)
    end

    @testset "Raise backward (autodiff)" begin
        @info "Creating shadow model (make_zero)..."
        @time "Creating shadow model" dmodel = Enzyme.make_zero(model)

        @info "Creating shadow θ field..."
        @time "Creating shadow θ" begin
            dθ_init = CenterField(grid)
            set!(dθ_init, 0.0)
        end

        @info "Compiling grad_loss_2m (raise=true, raise_first=true, sync=true)..."
        @time "Compiling grad_loss_2m" compiled_grad = Reactant.@compile raise_first=true raise=true sync=true grad_loss_2m(
            model, dmodel, θ_init, dθ_init, Δt, Ns)

        @info "Running compiled grad_loss_2m..."
        @time "Running compiled grad_loss_2m" dθ, loss_val = compiled_grad(
            model, dmodel, θ_init, dθ_init, Δt, Ns)

        @info "Result: loss = $loss_val, max|∇θ| = $(maximum(abs, interior(dθ)))"

        @test loss_val > 0
        @test isfinite(loss_val)
        @test maximum(abs, interior(dθ)) > 0
        @test !any(isnan, interior(dθ))
    end
end
