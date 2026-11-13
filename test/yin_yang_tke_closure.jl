include(joinpath(@__DIR__, "setup.jl"))

#####
##### TKEBasedTurbulenceClosure on a YinYangGrid: the halos of the diffusivities, interpolated from the
##### other panel, stay nonnegative next to a sharp jump in the TKE, and a sheared, heated state steps
##### stably, with the panel edges like the interior and like a latitude-longitude twin.
#####

using Breeze
using Breeze.TurbulenceClosures: TKE_NAME
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.MultiRegion: YinYangGrid
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Units
using Oceananigans.Utils: getregion
using Test

function tke_model(grid)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 1e5,
                                    reference_potential_temperature = 300)
    boundary_conditions = (; ρθ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(0.3)))
    model = AtmosphereModel(grid; dynamics, boundary_conditions, closure = TKEBasedTurbulenceClosure(),
                            advection = WENO(order=5))
    θᵢ(λ, φ, z) = 300 + 3e-3 * max(z - 1000, 0) + 0.01 * sind(3λ) * cosd(φ)
    uᵢ(λ, φ, z) = 10 * cosd(φ) * z / 3000
    set!(model; θ = θᵢ, u = uᵢ, ρ = HydrostaticallyBalancedDensity())
    return model
end

panel_minimum(field, r) = minimum(Array(parent(getregion(field, r))))

@testset "TKEBasedTurbulenceClosure on a YinYangGrid" begin
    grid = YinYangGrid(default_arch; size=(48, 16, 16), z=(0, 3kilometers), halo=(3, 3, 3))
    model = tke_model(grid)

    @testset "Nonnegative diffusivity halos next to a jump in the TKE" begin
        # Turbulent bands, 5 m²/s², separated by quiescent air: the 4 × 4 Lagrange interpolation that
        # fills the panel halos overshoots below zero next to each jump
        set!(model.tracers[TKE_NAME], (λ, φ, z) -> ifelse(mod(floor(λ / 10), 2) == 0, 6, 0))
        update_state!(model)

        Kᵘ = model.closure_fields.Kᵘ
        for K in (Kᵘ, model.closure_fields.Kᶜ, model.closure_fields.Kᵉ), r in 1:2
            @test panel_minimum(K, r) ≥ 0
        end

        # The interpolation alone, without the clipping of the closure fields, does overshoot below
        # zero, so the test above is not vacuous
        fill_halo_regions!(Kᵘ)
        @test min(panel_minimum(Kᵘ, 1), panel_minimum(Kᵘ, 2)) < 0
    end

    @testset "Sheared, heated state steps stably; edges match the interior and lat-lon" begin
        # TEMPORARY (#1103 CI stack overflow): print the overflowing stack, then rethrow
        stage = "construct Yin-Yang model"
        model = twin = nothing
        try
            model = tke_model(grid)
            stage = "construct lat-lon twin"
            twin = tke_model(LatitudeLongitudeGrid(default_arch; size=(96, 48, 16), longitude=(0, 360),
                                                   latitude=(-80, 80), z=(0, 3kilometers), halo=(3, 3, 3)))
            for n in 1:10
                stage = "Yin-Yang time step $n"
                time_step!(model, 60)
                stage = "lat-lon time step $n"
                time_step!(twin, 60)
            end
        catch e
            e isa StackOverflowError || rethrow()
            bt = stacktrace(catch_backtrace())
            println(stderr, "StackOverflowError during: ", stage, "; backtrace depth = ", length(bt))
            println(stderr, "--- innermost frames ---")
            foreach(f -> println(stderr, f), bt[1:min(end, 40)])
            println(stderr, "--- outermost frames ---")
            foreach(f -> println(stderr, f), bt[max(1, end - 39):end])
            # Which method is being compiled when the stack runs out: trace compilation in a fresh
            # process, stepping in tasks with shrinking stacks
            probe = joinpath(@__DIR__, "yin_yang_overflow_probe.jl")
            for stack_size in (2^23, 2^22, 2^21)
                log = IOBuffer()
                cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --trace-compile=stderr $probe $stack_size`
                run(pipeline(ignorestatus(cmd); stdout=devnull, stderr=log))
                lines = split(String(take!(log)), '\n')
                probe_lines = filter(l -> startswith(l, "PROBE"), lines)
                println(stderr, "=== probe, stack ", stack_size, ": ", join(probe_lines, " | "))
                compiled = filter(l -> startswith(l, "precompile("), lines)
                foreach(l -> println(stderr, first(l, 400)), compiled[max(1, end - 14):end])
            end
            rethrow()
        end

        ρe = model.tracers[TKE_NAME]
        for r in 1:2
            @test all(isfinite, Array(parent(getregion(ρe, r))))
            @test all(isfinite, Array(parent(getregion(model.velocities.u, r))))
        end
        @test minimum(ρe) ≥ -1e-6 * maximum(ρe)
        @test maximum(ρe) ≈ maximum(twin.tracers[TKE_NAME]) rtol=0.05

        # The surface TKE of the edge columns is that of the interior: the forcing is zonally uniform
        for r in 1:2
            e = Array(interior(getregion(ρe, r), :, :, 1))
            Nx, Ny = size(e)
            edges = vcat(e[1, :], e[Nx, :], e[:, 1], e[:, Ny])
            @test maximum(edges) ≤ 1.1 * maximum(e[4:Nx-3, 4:Ny-3])
        end
    end
end
