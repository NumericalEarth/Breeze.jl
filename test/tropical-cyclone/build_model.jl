using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Enzyme
using Statistics: mean
using Test
using CloudMicrophysics
using CUDA

# Toggle components here while debugging differentiability.
const TC_COMPONENTS = (
    coriolis = false,
    microphysics = false,
    boundary_conditions = false,
    advection = false,
)

const TC_GRID = (
    Nx = 6,
    Ny = 6,
    Nz = 8,
    halo = (5, 5, 5),
    topology = (Periodic, Periodic, Bounded),
    extent = (48_000, 48_000, 20_500),
)

const DUNION2011_MOIST_TROPICAL_MT_CSV = """
Pressure_hPa,GPH_m,Temperature_C,Dewpoint_C,RH_percent,Mixing_ratio_g_kg,Theta_K,Theta_e_K,Wind_speed_m_s,Wind_direction_deg
50,20726,-63.0,-73.8,25.5,0.04,494.5,495.4,12.7,88
100,16590,-74.5,-81.3,33.8,0.01,383.5,384.4,4.1,67
150,14203,-67.2,-74.7,35.3,0.01,354.2,354.5,3.6,330
200,12418,-54.3,-63.2,35.9,0.04,346.6,346.8,3.6,309
250,10949,-42.3,-52.4,37.0,0.14,343.0,343.6,2.1,305
300,9690,-32.3,-43.5,38.7,0.34,339.8,341.1,0.9,309
400,7596,-17.1,-28.7,44.4,1.12,332.7,336.8,0.8,100
500,5887,-6.6,-16.9,51.8,2.41,325.0,333.3,1.8,111
600,4437,1.6,-7.1,57.7,4.11,317.9,331.4,2.7,113
700,3178,8.9,2.5,65.7,6.74,312.3,333.6,3.6,112
850,1541,17.6,13.8,79.5,11.96,304.6,340.5,4.7,109
925,810,21.9,19.0,84.2,15.27,301.7,346.9,4.9,108
1000,124,26.5,23.3,83.3,18.50,299.6,354.0,3.0,98
1014.8,0,26.8,23.7,83.3,18.65,298.8,353.5,1.8,97
"""

function load_dunion_surface_state()
    lines = split(strip(DUNION2011_MOIST_TROPICAL_MT_CSV), '\n')
    pressure_hpa = Float64[]
    theta_k = Float64[]

    # Skip the CSV header and only parse the columns we need.
    for line in lines[2:end]
        cols = split(line, ',')
        push!(pressure_hpa, parse(Float64, cols[1]))
        push!(theta_k, parse(Float64, cols[7]))
    end

    p_data = reverse(100 .* pressure_hpa)
    θ_data = reverse(theta_k)
    return (surface_pressure = p_data[1], surface_potential_temperature = θ_data[1])
end

function maybe_microphysics(enabled)
    enabled || return nothing

    ext = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
    ext === nothing && error("BreezeCloudMicrophysicsExt is not available.")

    cloud_formation = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
    return ext.OneMomentCloudMicrophysics(; cloud_formation)
end

function maybe_advection(enabled)
    enabled || return (nothing, nothing)

    weno = WENO(order = 9)
    bounded_weno = WENO(order = 9, bounds = (0, 1))
    momentum_advection = weno
    scalar_advection = (
        ρθ = weno,
        ρqᵗ = bounded_weno,
        ρqᶜˡ = bounded_weno,
        ρqʳ = bounded_weno,
    )
    return momentum_advection, scalar_advection
end

function maybe_boundary_conditions(enabled, FT)
    enabled || return nothing

    Cᴰ = FT(1.229e-3)
    Cᵀ = FT(1.094e-3)
    Cᵛ = FT(1.133e-3)
    T₀ = FT(300)

    ρe = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient = Cᵀ,
                                                               surface_temperature = T₀))
    ρqᵗ = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient = Cᵛ,
                                                         surface_temperature = T₀))
    ρu = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ))
    ρv = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ))
    return (ρe = ρe, ρqᵗ = ρqᵗ, ρu = ρu, ρv = ρv)
end

function build_model(;
    arch,
    Nx,
    Ny,
    Nz,
    halo,
    topology,
    extent,
    components,
    float_type = Float32,
)
    Oceananigans.defaults.FloatType = float_type

    grid = RectilinearGrid(arch;
                           size = (Nx, Ny, Nz),
                           extent = extent,
                           halo = halo,
                           topology = topology)

    sounding = load_dunion_surface_state()
    FT = eltype(grid)
    dynamics = CompressibleDynamics(; standard_pressure = FT(1e5),
                                      surface_pressure = FT(sounding.surface_pressure),
                                      reference_potential_temperature = FT(sounding.surface_potential_temperature))

    kwargs = (; dynamics)

    if components.coriolis
        kwargs = merge(kwargs, (; coriolis = FPlane(f = 5e-5)))
    end

    microphysics = maybe_microphysics(components.microphysics)
    microphysics !== nothing && (kwargs = merge(kwargs, (; microphysics)))

    momentum_advection, scalar_advection = maybe_advection(components.advection)
    momentum_advection !== nothing && (kwargs = merge(kwargs, (; momentum_advection)))
    scalar_advection !== nothing && (kwargs = merge(kwargs, (; scalar_advection)))

    boundary_conditions = maybe_boundary_conditions(components.boundary_conditions, FT)
    boundary_conditions !== nothing && (kwargs = merge(kwargs, (; boundary_conditions)))

    model = AtmosphereModel(grid; kwargs...)
    return grid, model
end

function run_timesteps!(model, Δt, Nt)
    @trace track_numbers=false for _ in 1:Nt
        time_step!(model, Δt)
    end
    return nothing
end

@testset "Tropical cyclone model blueprint" begin
    @time "Building tropical cyclone model" begin
        global grid, model
        grid, model = build_model(;
                                  arch = ReactantState(),
                                  Nx = TC_GRID.Nx,
                                  Ny = TC_GRID.Ny,
                                  Nz = TC_GRID.Nz,
                                  halo = TC_GRID.halo,
                                  topology = TC_GRID.topology,
                                  extent = TC_GRID.extent,
                                  components = TC_COMPONENTS)
    end

    @testset "(a) Construction" begin
        @test model isa AtmosphereModel
        @test model.grid === grid
        @test model.grid.architecture isa ReactantState
        @test model.dynamics isa CompressibleDynamics
    end

    @testset "(c) Compiled time_step! (raise=true)" begin
        FT = eltype(grid)
        Δt = FT(0.25)
        Nt = 4
        @time "Phase (c): compile run_timesteps!" compiled_run! = Reactant.@compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
        @time "Phase (c): run compiled forward pass" compiled_run!(model, Δt, Nt)
        @test model.clock.iteration == Nt
    end

    @testset "(d) Enzyme reverse-mode gradient" begin
        dmodel = Enzyme.make_zero(model)
        FT = eltype(grid)

        θ_init = CenterField(grid)
        set!(θ_init, (x, y, z) -> FT(300) + FT(1e-4) * x)
        dθ_init = CenterField(grid)
        set!(dθ_init, FT(0))

        function loss(model, θ_init, Δt, nsteps)
            set!(model, θ = θ_init, ρ = FT(1))
            @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
                time_step!(model, Δt)
            end
            return mean(interior(model.temperature).^2)
        end

        function grad_loss(model, dmodel, θ_init, dθ_init, Δt, nsteps)
            parent(dθ_init) .= 0
            _, loss_val = Enzyme.autodiff(
                Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                loss, Enzyme.Active,
                Enzyme.Duplicated(model, dmodel),
                Enzyme.Duplicated(θ_init, dθ_init),
                Enzyme.Const(Δt),
                Enzyme.Const(nsteps),
            )
            return dθ_init, loss_val
        end

        Δt = FT(0.25)
        nsteps = 4
        @time "Phase (d): compile grad_loss" compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
            model, dmodel, θ_init, dθ_init, Δt, nsteps)
        @test compiled_grad !== nothing

        @time "Phase (d): run compiled gradient" dθ, loss_val = compiled_grad(model, dmodel, θ_init, dθ_init, Δt, nsteps)
        @test loss_val > 0
        @test isfinite(loss_val)
        @test maximum(abs, interior(dθ)) > 0
        @test !any(isnan, interior(dθ))
    end
end
