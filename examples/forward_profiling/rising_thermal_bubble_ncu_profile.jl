# Self-contained Julia-GPU-only forward pass for NCU profiling.
# No Reactant loaded — avoids CUPTI conflicts and OOM from child processes.
#
# Usage:
#   ncu --set basic --replay-mode application --target-processes all \
#       --kernel-name-base demangled --log-file report.log --export report \
#       julia --project=. examples/forward_profiling/rising_thermal_bubble_ncu_profile.jl
#
# Optional env vars:
#   BREEZE_NCU_GRID_IDX   — run only grid at this 1-based index (default: all)
#   BREEZE_NCU_NSTEPS     — number of time steps (default: 4)

using Breeze
using Oceananigans
using CUDA
using Statistics: mean
using Printf

function main()
    @info "CUDA runtime configuration" runtime = CUDA.runtime_version() local_toolkit = CUDA.local_toolkit

    float_type = Float32
    Oceananigans.defaults.FloatType = float_type

    θ_background           = float_type(300.0)
    perturbation_amplitude = float_type(2.0)
    perturbation_radius    = float_type(1000.0)
    bubble_center_x        = float_type(5000.0)
    bubble_center_y        = float_type(5000.0)
    bubble_center_z        = float_type(2000.0)
    latitude               = float_type(45.0)

    domain_x, domain_y, domain_z = float_type(10000.0), float_type(10000.0), float_type(10000.0)
    topology = (Periodic, Bounded, Bounded)

    grid_sizes = [(128, 128, 128), (256, 256, 128), (512, 512, 128), (1024, 1024, 128)]
    loss_z_threshold = float_type(5000.0)
    nsteps = parse(Int, get(ENV, "BREEZE_NCU_NSTEPS", "4"))

    ncu_grid_idx = tryparse(Int, get(ENV, "BREEZE_NCU_GRID_IDX", ""))
    if ncu_grid_idx !== nothing
        if !(1 <= ncu_grid_idx <= length(grid_sizes))
            error("Invalid BREEZE_NCU_GRID_IDX=$(ncu_grid_idx), expected 1:$(length(grid_sizes)).")
        end
        grid_sizes = [grid_sizes[ncu_grid_idx]]
        @info "Running single grid for ncu" grid_sizes
    end

    function forward_loss!(model, θ_initial, Δt, nsteps, loss_z_threshold, domain_z)
        FT = eltype(model.grid)
        set!(model; θ = θ_initial, ρ = FT(1))
        for _ in 1:nsteps
            time_step!(model, Δt)
        end
        θ_evolved = interior(model.temperature)
        k_start = ceil(Int, loss_z_threshold / domain_z * size(model.grid, 3)) + 1
        upper_θ = @view θ_evolved[:, :, k_start:end]
        return mean(upper_θ .^ 2)
    end

    for (Nx, Ny, Nz) in grid_sizes
        GC.gc()
        @info @sprintf("Grid resolution: %d × %d × %d", Nx, Ny, Nz)

        grid = RectilinearGrid(GPU();
            size = (Nx, Ny, Nz),
            x = (0, domain_x), y = (0, domain_y), z = (0, domain_z),
            topology = topology,
        )

        FT = eltype(grid)
        advection = WENO(order = 5)
        coriolis = FPlane(; latitude)
        model = AtmosphereModel(grid; advection, coriolis, dynamics = CompressibleDynamics())

        thermo = model.thermodynamic_constants
        Rᵈ = thermo.molar_gas_constant / thermo.dry_air.molar_mass
        cᵖᵈ = thermo.dry_air.heat_capacity
        γ = cᵖᵈ / (cᵖᵈ - Rᵈ)
        sound_speed = sqrt(γ * Rᵈ * float_type(θ_background))
        Δt = FT(0.4 * domain_x / Nx / sound_speed)

        θ_perturbation(x, y, z) = FT(perturbation_amplitude) * exp(
            -((x - bubble_center_x)^2 + (y - bubble_center_y)^2 + (z - bubble_center_z)^2)
            / perturbation_radius^2
        )
        θ_initial_fn(x, y, z) = FT(θ_background) + θ_perturbation(x, y, z)
        θ_initial = CenterField(grid)
        set!(θ_initial, θ_initial_fn)

        @info @sprintf("Δt = %.6f s, running %d steps", Δt, nsteps)

        # Warmup
        forward_loss!(model, θ_initial, Δt, nsteps, loss_z_threshold, domain_z)
        CUDA.synchronize()

        # Profiled run
        loss = forward_loss!(model, θ_initial, Δt, nsteps, loss_z_threshold, domain_z)
        CUDA.synchronize()

        @info @sprintf("Forward loss J = %.6e", loss)
    end
end

main()
