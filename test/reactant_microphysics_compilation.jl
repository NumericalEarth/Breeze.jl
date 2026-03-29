#####
##### Reactant forward-raise test — CompressibleDynamics + 1M mixed-phase NE microphysics (PBB)
#####
#
# Raises (forward only) a loss function through a CompressibleDynamics
# AtmosphereModel with OneMomentCloudMicrophysics (mixed-phase non-equilibrium
# with ice) on a RectilinearGrid with (Periodic, Bounded, Bounded) topology
# and Centered advection.

using Breeze
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudLiquid, CloudIce
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using Reactant
using Reactant: @trace
using Statistics: mean
using Test
using CUDA

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

if default_arch isa GPU
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

#####
##### Forward loss (all state mutation happens inside the compiled segment)
#####

function loss(model, θ_init, Δt, Nsteps)
    set!(model; θ=θ_init, ρ=1.0, qᵗ=0.01)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return mean(interior(model.temperature) .^ 2)
end

#####
##### Test
#####

@testset "Reactant forward raise — CompressibleDynamics + 1M NE microphysics (PBB)" begin
    topo = (Periodic, Bounded, Bounded)

    @info "Building grid: RectilinearGrid 4×4×4, topology=$topo"
    @time "Constructing grid" grid = RectilinearGrid(ReactantState();
        size=(4, 4, 4), extent=(1e3, 1e3, 1e3), topology=topo)

    @info "Setting up microphysics: OneMomentCloudMicrophysics (mixed-phase NE, liquid + ice)"
    cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(Float64), CloudIce(Float64))
    microphysics = OneMomentCloudMicrophysics(Float64; cloud_formation)

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
    end

    Δt = 0.01
    Ns = 1

    @info "Constructing θ_init field"
    @time "Constructing θ_init" begin
        θ_init = CenterField(grid)
        set!(θ_init, (x, y, z) -> 300.0)
    end

    @testset "Raise forward" begin
        @info "Compiling loss (raise=true, raise_first=true, sync=true)..."
        @time "Compiling loss" compiled_loss = Reactant.@compile debug=true optimize=true raise=true raise_first=true sync=true loss(
            model, θ_init, Δt, Ns)

        @info "Running compiled loss..."
        @time "Running compiled loss" loss_val = compiled_loss(model, θ_init, Δt, Ns)

        @info "Result: loss = $loss_val"

        @test loss_val > 0
        @test isfinite(loss_val)
    end
end
