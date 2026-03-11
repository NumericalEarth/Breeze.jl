# Self-contained Julia-GPU-only forward pass for NCU profiling (baroclinic wave).
# No Reactant loaded — avoids CUPTI conflicts and OOM from child processes.
#
# Usage:
#   ncu --set basic --replay-mode application --target-processes all \
#       --kernel-name-base demangled --log-file report.log --export report \
#       julia --project=. examples/forward_profiling/baroclinic_wave_ncu_profile.jl
#
# Optional env vars:
#   BREEZE_NCU_GRID_IDX   — run only grid at this 1-based index (default: all)
#   BREEZE_NCU_NSTEPS     — number of time steps (default: 4)

using Breeze
using Oceananigans
using Oceananigans.Units
using CUDA
using Statistics: mean
using Printf

function main()
    @info "CUDA runtime configuration" runtime = CUDA.runtime_version() local_toolkit = CUDA.local_toolkit

    float_type = Float32
    Oceananigans.defaults.FloatType = float_type

    H = float_type(30kilometers)
    halo = (5, 5, 5)
    longitude = (float_type(0), float_type(360))
    latitude = (float_type(-85), float_type(85))
    grid_sizes = [(90, 43, 30), (180, 85, 30), (360, 170, 30)]
    nsteps = parse(Int, get(ENV, "BREEZE_NCU_NSTEPS", "4"))

    ncu_grid_idx = tryparse(Int, get(ENV, "BREEZE_NCU_GRID_IDX", ""))
    if ncu_grid_idx !== nothing
        if !(1 <= ncu_grid_idx <= length(grid_sizes))
            error("Invalid BREEZE_NCU_GRID_IDX=$(ncu_grid_idx), expected 1:$(length(grid_sizes)).")
        end
        grid_sizes = [grid_sizes[ncu_grid_idx]]
        @info "Running single grid for ncu" grid_sizes
    end

    for (Nλ, Nφ, Nz) in grid_sizes
        GC.gc()
        @info @sprintf("Grid resolution: %d × %d × %d", Nλ, Nφ, Nz)

        grid = LatitudeLongitudeGrid(GPU();
            size = (Nλ, Nφ, Nz),
            halo = halo,
            longitude = longitude,
            latitude = latitude,
            z = (float_type(0), H),
        )

        FT = eltype(grid)

        constants = ThermodynamicConstants()
        g = FT(constants.gravitational_acceleration)
        p₀ = FT(100000)
        θ₀ = FT(300)
        N² = FT(1e-4)

        θᵇ(z) = θ₀ * exp(N² * z / g)

        coriolis = HydrostaticSphericalCoriolis()
        Ω = FT(coriolis.rotation_rate)
        Rplanet = FT(Oceananigans.defaults.planet_radius)
        Δθ_ep = FT(60)
        z_T = FT(15000)
        τ_bal = Rplanet * θ₀ * Ω / (g * Δθ_ep)

        λ_c = FT(90)
        φ_c = FT(45)
        σ = FT(10)
        Δθ = FT(1)

        function uᵢ(λ, φ, z)
            vertical_scale = ifelse(z ≤ z_T, z / FT(2) * (FT(2) - z / z_T), z_T / FT(2))
            return (vertical_scale / τ_bal) * cosd(φ)
        end

        function θᵢ(λ, φ, z)
            θ_merid = -Δθ_ep * sind(φ) * max(FT(0), FT(1) - z / z_T)
            r² = (λ - λ_c)^2 + (φ - φ_c)^2
            θ_pert = Δθ * exp(-r² / (FT(2) * σ^2)) * sin(FT(π) * z / H)
            return θᵇ(z) + θ_merid + θ_pert
        end

        Rᵈ = FT(dry_air_gas_constant(constants))
        cᵖ = FT(constants.dry_air.heat_capacity)
        κ = Rᵈ / cᵖ
        cᵥ_over_Rᵈ = (cᵖ - Rᵈ) / Rᵈ

        function ρᵢ(λ, φ, z)
            nz = max(1, round(Int, z / FT(100)))
            dz = z / nz
            Π = FT(1)
            for n in 1:nz
                zn = (n - FT(1) / FT(2)) * dz
                θn = θᵢ(λ, φ, zn)
                Π -= κ * g / (Rᵈ * θn) * dz
            end
            θ = θᵢ(λ, φ, z)
            return p₀ * Π^cᵥ_over_Rᵈ / (Rᵈ * θ)
        end

        dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                        surface_pressure = p₀,
                                        reference_potential_temperature = θᵇ)
        model = AtmosphereModel(grid; dynamics, coriolis)

        ρ_initial = CenterField(grid); set!(ρ_initial, ρᵢ)
        θ_initial = CenterField(grid); set!(θ_initial, θᵢ)

        set!(model; ρ = ρ_initial, u = uᵢ, θ = θ_initial)
        u_initial = deepcopy(model.velocities.u)

        γ = cᵖ / (cᵖ - Rᵈ)
        cₛ = sqrt(γ * Rᵈ * θ₀)
        Δz = H / Nz
        Δt = FT(0.4 * Δz / cₛ)
        @info @sprintf("Δt = %.6f s, running %d steps", Δt, nsteps)

        function forward_loss!(model, u_init, θ_init, ρ_init, Δt, nsteps)
            set!(model; ρ = ρ_init, u = u_init, θ = θ_init)
            for _ in 1:nsteps
                time_step!(model, Δt)
            end
            v = model.velocities.v
            return mean(interior(v) .^ 2)
        end

        # Warmup
        forward_loss!(model, u_initial, θ_initial, ρ_initial, Δt, nsteps)
        CUDA.synchronize()

        # Profiled run
        loss = forward_loss!(model, u_initial, θ_initial, ρ_initial, Δt, nsteps)
        CUDA.synchronize()

        @info @sprintf("Forward loss J = %.6e", loss)
    end
end

main()
