#=
Investigation: Microphysics Differentiability
Status: EXPLORATORY (as of 2026-02-25)
Purpose: Test Reactant/Enzyme AD through each Breeze microphysics scheme
Related: Built-in (SA, Kessler, BulkMicrophysics) + CloudMicrophysics extension (0M, 1M, 2M)
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using Breeze
using Breeze.Microphysics: DCMIP2016KesslerMicrophysics, SaturationAdjustment, WarmPhaseEquilibrium
using Breeze.Thermodynamics: TetensFormula, ThermodynamicConstants
using Reactant
using Enzyme
using Statistics: mean
using Test
using CUDA
using Pkg

Reactant.set_default_backend("cpu")

println("Package versions:")
for pkg in ["Oceananigans", "Breeze", "Reactant", "Enzyme"]
    v = Pkg.dependencies()[Base.UUID(Pkg.project().dependencies[pkg])].version
    println("  $pkg: v$v")
end

# Enable MLIR dumping for debugging
mlir_dump_dir = joinpath(@__DIR__, "mlir_dump")
mkpath(mlir_dump_dir)
# Uncomment to enable MLIR dumps:
# Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
# Reactant.MLIR.IR.DUMP_MLIR_DIR[] = mlir_dump_dir

#####
##### Shared setup
#####

Nx, Ny = 4, 4
Nz = 4
Lx, Ly = 1000.0, 1000.0
Lz = 1000.0
Δt = 0.01
nsteps = 4  # perfect square for checkpointing

function make_model(arch; microphysics=nothing, moisture=false, thermodynamic_constants=nothing)
    grid = RectilinearGrid(arch;
        size = (Nx, Ny, Nz),
        extent = (Lx, Ly, Lz),
        halo = (3, 3, 3),
        topology = (Periodic, Bounded, Bounded))

    kwargs = (; dynamics = CompressibleDynamics())
    if microphysics !== nothing
        kwargs = (; kwargs..., microphysics)
    end
    if thermodynamic_constants !== nothing
        kwargs = (; kwargs..., thermodynamic_constants)
    end
    model = AtmosphereModel(grid; kwargs...)

    if moisture
        set!(model, θ = 300.0, ρ = 1.0, qᵗ = 0.015)
    else
        set!(model, θ = 300.0, ρ = 1.0)
    end

    return model
end

function loss(model, θ_init, Δt, nsteps)
    set!(model, θ = θ_init, ρ = 1.0)
    @trace mincut=true checkpointing=true track_numbers=false for i in 1:nsteps
        time_step!(model, Δt)
    end
    return mean(interior(model.temperature) .^ 2)
end

function grad_loss(model, dmodel, θ_init, dθ_init, Δt, nsteps)
    parent(dθ_init) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return dθ_init, loss_value
end

function run_forward_test(model)
    grid = model.grid
    θ_init = CenterField(grid)
    set!(θ_init, (x, y) -> 300.0 + 0.01 * x + 0.01 * y)

    @info "  Compiling forward loss..."
    @time "  Compiling forward loss" compiled_loss = Reactant.@compile raise_first=true raise=true sync=true loss(
        model, θ_init, Δt, nsteps)

    @info "  Running compiled forward loss..."
    @time "  Running forward loss" result = compiled_loss(model, θ_init, Δt, nsteps)

    @info "  Forward result: loss = $result"
    @test !isnan(result)
    @test result > 0
    return result
end

function run_gradient_test(model)
    grid = model.grid
    @time "  Creating fields" begin
        θ_init = CenterField(grid)
        set!(θ_init, (x, y) -> 300.0 + 0.01 * x + 0.01 * y)
        dθ_init = CenterField(grid)
        set!(dθ_init, 0.0)
    end

    @info "  Creating shadow model (Enzyme.make_zero)..."
    @time "  Creating shadow model" dmodel = Enzyme.make_zero(model)

    @info "  Compiling grad_loss..."
    @time "  Compiling grad_loss" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss(
        model, dmodel, θ_init, dθ_init, Δt, nsteps)

    @info "  Running compiled grad_loss..."
    @time "  Running grad_loss" dθ, loss_val = compiled(model, dmodel, θ_init, dθ_init, Δt, nsteps)

    grad_max = maximum(abs, interior(dθ))
    grad_has_nan = any(isnan, interior(dθ))
    @info "  Gradient result: loss = $loss_val, max|∇θ| = $grad_max, has NaN = $grad_has_nan"

    @test loss_val > 0
    @test !isnan(loss_val)
    @test grad_max > 0
    @test !grad_has_nan
    return dθ, loss_val
end

#####
##### 1. No microphysics (dry baseline)
#####

# @info "=" ^ 60
# @info "Test 1/9: Dry baseline (no microphysics)"
# @info "=" ^ 60

# @testset "Dry baseline (no microphysics)" begin
#     @time "Constructing dry model" model = make_model(ReactantState())

#     @testset "Forward" begin
#         run_forward_test(model)
#     end

#     @testset "Gradient" begin
#         run_gradient_test(model)
#     end
# end

#####
##### 2. SaturationAdjustment — Warm phase
#####

# @info "=" ^ 60
# @info "Test 2/9: SaturationAdjustment (WarmPhaseEquilibrium)"
# @info "=" ^ 60

# @testset "SaturationAdjustment (WarmPhaseEquilibrium)" begin
#     microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
#     @time "Constructing SA warm-phase model" model = make_model(ReactantState(); microphysics, moisture=true)

#     @testset "Forward" begin
#         run_forward_test(model)
#     end

#     @testset "Gradient" begin
#         run_gradient_test(model)
#     end
# end

#####
##### 3. SaturationAdjustment — Mixed phase
#####

# @info "=" ^ 60
# @info "Test 3/9: SaturationAdjustment (MixedPhaseEquilibrium)"
# @info "=" ^ 60

# @testset "SaturationAdjustment (MixedPhaseEquilibrium)" begin
#     microphysics = SaturationAdjustment(equilibrium = MixedPhaseEquilibrium())
#     @time "Constructing SA mixed-phase model" model = make_model(ReactantState(); microphysics, moisture=true)

#     @testset "Forward" begin
#         run_forward_test(model)
#     end

#     @testset "Gradient" begin
#         run_gradient_test(model)
#     end
# end

#####
##### 4. Non-precipitating BulkMicrophysics (SA cloud formation, no categories)
#####

# @info "=" ^ 60
# @info "Test 4/9: BulkMicrophysics (non-precipitating)"
# @info "=" ^ 60

# @testset "BulkMicrophysics (non-precipitating)" begin
#     microphysics = BulkMicrophysics()
#     @time "Constructing non-precipitating BulkMicrophysics model" model = make_model(ReactantState(); microphysics, moisture=true)

#     @testset "Forward" begin
#         run_forward_test(model)
#     end

#     @testset "Gradient" begin
#         run_gradient_test(model)
#     end
# end

#####
##### 5. DCMIP2016 Kessler warm-rain microphysics
#####

# @info "=" ^ 60
# @info "Test 5/9: DCMIP2016KesslerMicrophysics"
# @info "=" ^ 60

# @testset "DCMIP2016KesslerMicrophysics" begin
#     microphysics = DCMIP2016KesslerMicrophysics()
#     kessler_constants = ThermodynamicConstants(saturation_vapor_pressure=TetensFormula())
#     @time "Constructing Kessler model" model = make_model(ReactantState();
#         microphysics, moisture=true, thermodynamic_constants=kessler_constants)

#     @testset "Forward" begin
#         run_forward_test(model)
#     end

#     @testset "Gradient" begin
#         run_gradient_test(model)
#     end
# end

#####
##### CloudMicrophysics extension schemes (require CloudMicrophysics.jl)
#####

@info "Loading CloudMicrophysics extension..."
using CloudMicrophysics
BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: ZeroMomentCloudMicrophysics,
                                   OneMomentCloudMicrophysics,
                                   TwoMomentCloudMicrophysics
@info "CloudMicrophysics extension loaded"

#####
##### 6. Zero-moment CloudMicrophysics (instant precipitation removal)
#####

@info "=" ^ 60
@info "Test 6/9: ZeroMomentCloudMicrophysics (0M)"
@info "=" ^ 60

@testset "ZeroMomentCloudMicrophysics (0M)" begin
    microphysics = ZeroMomentCloudMicrophysics()
    @time "Constructing 0M model" model = make_model(ReactantState(); microphysics, moisture=true)

    @testset "Forward" begin
        run_forward_test(model)
    end

    @testset "Gradient" begin
        run_gradient_test(model)
    end
end

#####
##### 7. One-moment CloudMicrophysics — Non-equilibrium (warm phase)
#####

@info "=" ^ 60
@info "Test 7/9: OneMomentCloudMicrophysics (1M, non-equilibrium)"
@info "=" ^ 60

@testset "OneMomentCloudMicrophysics (1M, non-equilibrium)" begin
    microphysics = OneMomentCloudMicrophysics()
    @time "Constructing 1M NE model" model = make_model(ReactantState(); microphysics, moisture=true)

    @testset "Forward" begin
        run_forward_test(model)
    end

    @testset "Gradient" begin
        run_gradient_test(model)
    end
end

#####
##### 8. One-moment CloudMicrophysics — Saturation adjustment (warm phase)
#####

@info "=" ^ 60
@info "Test 8/9: OneMomentCloudMicrophysics (1M, saturation adjustment)"
@info "=" ^ 60

@testset "OneMomentCloudMicrophysics (1M, saturation adjustment)" begin
    cloud_formation = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
    microphysics = OneMomentCloudMicrophysics(; cloud_formation)
    @time "Constructing 1M SA model" model = make_model(ReactantState(); microphysics, moisture=true)

    @testset "Forward" begin
        run_forward_test(model)
    end

    @testset "Gradient" begin
        run_gradient_test(model)
    end
end

#####
##### 9. Two-moment CloudMicrophysics (Seifert-Beheng 2006, warm phase)
#####

@info "=" ^ 60
@info "Test 9/9: TwoMomentCloudMicrophysics (2M)"
@info "=" ^ 60

@testset "TwoMomentCloudMicrophysics (2M)" begin
    microphysics = TwoMomentCloudMicrophysics()
    @time "Constructing 2M model" model = make_model(ReactantState(); microphysics, moisture=true)

    @testset "Forward" begin
        run_forward_test(model)
    end

    @testset "Gradient" begin
        run_gradient_test(model)
    end
end
