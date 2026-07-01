# # Resolution convergence of the Cartesian power method
#
# Does the wavenumber-9 growth rate depend on the grid? We run the power
# method on six grids — four horizontal refinements from a coarse 400 km
# mesh up to a 50 km mesh, plus two vertical-only refinements at the
# 100 km and 50 km horizontal resolutions — and check that ``\sigma``
# converges. At the finest resolution we also plot the eigenmode to
# confirm it retains the correct baroclinic structure (westward tilt with
# height).

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans: prognostic_fields
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Models: boundary_condition_args
using Oceananigans.TimeSteppers: update_state!, reset!
using Printf
using CairoMakie
using CUDA

# ## URJ15 parameters
#
# Identical to `baroclinic_wave_cartesian.jl`.

Oceananigans.defaults.FloatType = Float32
Oceananigans.defaults.gravitational_acceleration = 9.80616

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287.0)

g   = constants.gravitational_acceleration
Rᵈ  = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ

a  = 6.371229e6
Ω  = 7.29212e-5
p₀ = 1e5
T₀ = 288
Γ  = 0.005
b  = 2
u₀ = 35
ΔT = 4.8e5
ηₜ = 0.2
κ_T = Rᵈ * Γ / g

φ₀ = π / 4
f₀ = 2Ω * sin(φ₀)

Lx = 40_000kilometers
Ly = 6_000kilometers
Lz = 30kilometers

# ## Balanced state
#
# The URJ15 analytic profiles, converted from ``\eta`` to height
# coordinates via Newton iteration.

α_exp = g / (Rᵈ * Γ)
η_mean(z) = (1 - Γ * z / T₀)^α_exp

function T_bar(η)
    T = T₀ * η^κ_T
    return ifelse(η < ηₜ, T + ΔT * (ηₜ - η)^5, T)
end

function urj15_u(y, η)
    s = log(η)
    return -u₀ * sin(π * y / Ly)^2 * s * exp(-(s / b)^2)
end

function T_prime(y, η)
    s = log(η)
    G = (1 - 2 * s^2 / b^2) * exp(-(s / b)^2)
    I = y / 2 - Ly / (4π) * sin(2π * y / Ly) - Ly / 4
    return -(f₀ * u₀ / Rᵈ) * I * G
end

T_full(y, η) = T_bar(η) + T_prime(y, η)

function Φ_bar(η)
    Φ = (g * T₀ / Γ) * (1 - η^κ_T)
    if η < ηₜ
        Φ -= Rᵈ * ΔT * (ηₜ^5 * log(η / ηₜ)
                         - 5 * ηₜ^4 * (η - ηₜ)
                         + 5 * ηₜ^3 * (η^2 - ηₜ^2)
                         - 10/3 * ηₜ^2 * (η^3 - ηₜ^3)
                         + 5/4 * ηₜ * (η^4 - ηₜ^4)
                         - 1/5 * (η^5 - ηₜ^5))
    end
    return Φ
end

function Φ_prime(y, η)
    s = log(η)
    return f₀ * u₀ * (y / 2 - Ly / (4π) * sin(2π * y / Ly) - Ly / 4) * s * exp(-(s / b)^2)
end

Φ_total(y, η) = Φ_bar(η) + Φ_prime(y, η)

function η_at(y, z)
    target = g * z
    η = η_mean(z)
    for _ in 1:10
        Φ = Φ_total(y, η)
        T = T_full(y, η)
        dΦdη = -Rᵈ * T / η
        η = η - (Φ - target) / dΦdη
        η = clamp(η, 1e-8, 1.0)
    end
    return η
end

virtual_temperature(y, z) = T_full(y, η_at(y, z))
pressure(y, z) = p₀ * η_at(y, z)
density(y, z) = pressure(y, z) / (Rᵈ * virtual_temperature(y, z))
potential_temperature_bg(y, z) = virtual_temperature(y, z) * (p₀ / pressure(y, z))^κ
balanced_u(y, z) = urj15_u(y, η_at(y, z))

# ## Power method driver
#
# Build a model at the requested resolution, seed a wavenumber-9
# perturbation in ``v``, and iterate until ``\sigma`` converges.
# Returns the grid spacing, converged growth rate, iteration count,
# and the final model (so the caller can inspect the eigenmode).

v_ref = 1.0
m = 9
Δτ = 3days
max_iterations = 40
convergence_threshold = 0.001

function run_power_method(Nx, Ny, Nz_run)
    @info @sprintf("═══ Resolution: %d × %d × %d (Δx ≈ %.0f km) ═══", Nx, Ny, Nz_run, Lx/Nx/1e3)

    grid = RectilinearGrid(GPU();
                           size = (Nx, Ny, Nz_run),
                           halo = (5, 5, 5),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = (0, Lz),
                           topology = (Periodic, Bounded, Bounded))

    coriolis = FPlane(f=f₀)

    T₀ᵣ = 250
    θᵣ(z) = T₀ᵣ * exp(g * z / (cᵖᵈ * T₀ᵣ))

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                    surface_pressure = p₀,
                                    reference_potential_temperature = θᵣ)

    model = AtmosphereModel(grid; dynamics, coriolis,
                            thermodynamic_constants = constants,
                            advection = WENO())

    initial_u(x, y, z) = balanced_u(y, z)
    initial_θ(x, y, z) = potential_temperature_bg(y, z)
    initial_ρ(x, y, z) = density(y, z)

    set!(model; θ=initial_θ, u=initial_u, ρ=initial_ρ)

    background = map(f -> copy(parent(f)), prognostic_fields(model))

    v_pert(x, y, z) = begin
        zₚ = 15000
        taper = ifelse(z < zₚ, 1 - 3 * (z / zₚ)^2 + 2 * (z / zₚ)^3, zero(z))
        Lₚ = 1000kilometers
        v_ref * sin(2π * m * x / Lx) * exp(-(y - Ly / 2)^2 / Lₚ^2) * taper
    end

    set!(model; v=v_pert)

    fill_halo_regions!(prognostic_fields(model), boundary_condition_args(model)..., async=true)
    update_state!(model, compute_tendencies=false)

    simulation = Simulation(model; Δt=10minutes, stop_time=Δτ)
    conjure_time_step_wizard!(simulation; cfl=1.4, max_Δt=10minutes)
    Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

    σ_history = Float64[]

    for n in 1:max_iterations
        run!(simulation)

        v_sfc_max = maximum(abs, view(model.velocities.v, :, :, 1))
        σ = log(v_sfc_max / v_ref) / Δτ
        push!(σ_history, σ)

        @info @sprintf("  iter %3d | σ = %.4f day⁻¹ | sfc max|v| = %.4e m/s", n, σ * 86400, v_sfc_max)

        scale = convert(eltype(model.grid), v_ref / v_sfc_max)
        for (f, bg) in zip(prognostic_fields(model), background)
            parent(f) .= bg .+ scale .* (parent(f) .- bg)
        end

        fill_halo_regions!(prognostic_fields(model), boundary_condition_args(model)..., async=true)

        converged = n ≥ 2 && abs(σ_history[end] - σ_history[end-1]) / abs(σ_history[end]) < convergence_threshold

        if converged
            update_state!(model, compute_tendencies=false)
            @info @sprintf("  Converged after %d iterations (σ = %.4f day⁻¹)", n, σ * 86400)
            return Lx / Nx, σ * 86400, n, model, σ_history .* 86400
        end

        reset!(model.clock)
        update_state!(model, compute_tendencies=false)
        simulation.stop_time = Δτ
    end

    σ_final = σ_history[end] * 86400
    @info @sprintf("  Did NOT converge after %d iterations (last σ = %.4f day⁻¹)", max_iterations, σ_final)
    return Lx / Nx, σ_final, max_iterations, model, σ_history .* 86400
end

# ## Run at six resolutions
#
# The first four configs double the horizontal grid points in each
# direction; the last two hold horizontal resolution fixed and instead
# refine ``N_z`` from 30 to 128. Vertical resolution matters most for
# this problem: the jump from ``N_z = 15`` to ``N_z = 30`` is where
# ``\sigma`` locks in, and the last two configs check whether it shifts
# again at even higher vertical resolution.

resolution_configs = [
    (100,  15,  8),   ## Δx ≈ 400 km
    (200,  30,  15),  ## Δx ≈ 200 km
    (400,  60,  30),  ## Δx ≈ 100 km (1°)
    (800,  120, 30),  ## Δx ≈  50 km (0.5°)
    (400,  60,  128), ## Δx ≈ 100 km, high vertical resolution
    (800,  120, 128), ## Δx ≈  50 km (0.5°), high vertical resolution
]

Δx_list = Float64[]
σ_list = Float64[]
niter_list = Int[]
σ_histories = Vector{Float64}[]
last_model = nothing

for (nx, ny, nz) in resolution_configs
    global last_model
    Δx, σ_val, niter, model, σ_hist = run_power_method(nx, ny, nz)
    push!(Δx_list, Δx)
    push!(σ_list, σ_val)
    push!(niter_list, niter)
    push!(σ_histories, σ_hist)
    last_model = model
end

@info "Resolution convergence results:"
for i in eachindex(Δx_list)
    @info @sprintf("  Δx = %6.0f km | σ = %.4f day⁻¹ | %d iterations",
                   Δx_list[i]/1e3, σ_list[i], niter_list[i])
end

# ## Convergence plot
#
# The iteration history for each resolution is overlaid on a single figure,
# with a summary table on the right. Each line shows ``\sigma`` vs.
# power iteration — the resolved grids converge rapidly and agree.

Δx_km = Δx_list ./ 1e3
colors = Makie.wong_colors()

fig = Figure(size=(1100, 500))

ax = Axis(fig[1, 1];
          xlabel = "Power iteration",
          ylabel = "Growth rate σ (day⁻¹)",
          title = "Power method convergence at each resolution (m = 9)")

for (i, σ_hist) in enumerate(σ_histories)
    nx, ny, nz = resolution_configs[i]
    label = @sprintf("%d×%d×%d", nx, ny, nz)
    lines!(ax, 1:length(σ_hist), σ_hist;
           linewidth=2, color=colors[i], label=label)
    scatter!(ax, 1:length(σ_hist), σ_hist;
             markersize=5, color=colors[i])
end

axislegend(ax; position=:rb)

## Right panel: summary table
table_ax = Axis(fig[1, 2]; title="Convergence summary")
hidedecorations!(table_ax)
hidespines!(table_ax)

header = "  Nx × Ny × Nz      Δx (km)   σ (day⁻¹)  iters"
rows = [header, "  " * "─"^48]
for i in eachindex(Δx_list)
    nx, ny, nz = resolution_configs[i]
    row = @sprintf("  %3d × %3d × %3d   %6.0f     %.4f     %2d",
                   nx, ny, nz, Δx_km[i], σ_list[i], niter_list[i])
    push!(rows, row)
end
table_text = join(rows, "\n")
text!(table_ax, 0.05, 0.5; text=table_text, fontsize=14,
      font=:regular, align=(:left, :center), space=:relative)
xlims!(table_ax, 0, 1)
ylims!(table_ax, 0, 1)

save("power_method_resolution_convergence.png", fig)
nothing #hide

# ![](power_method_resolution_convergence.png)

# ## Eigenmode at the finest resolution
#
# The converged eigenmode from the finest grid (``\Delta x = 50`` km).
# The surface plan view shows the wavenumber-9 pattern in ``v`` and
# ``\delta\theta``, and the longitude–height cross section at the jet
# core confirms the **westward tilt with height** characteristic of
# a growing baroclinic wave.

model = last_model
Nx_fine = resolution_configs[end][1]
Ny_fine = resolution_configs[end][2]
Nz_fine = resolution_configs[end][3]

v = model.velocities.v
θ_field = Field(PotentialTemperature(model))
compute!(θ_field)

## θ perturbation: subtract the analytic background
θ_perturbation = Field{Center, Center, Center}(model.grid)
set!(θ_perturbation, (x, y, z) -> potential_temperature_bg(y, z))
parent(θ_perturbation) .= parent(θ_field) .- parent(θ_perturbation)

# ### Surface plan view

v_sfc = view(v, :, :, 1)
δθ_sfc = view(θ_perturbation, :, :, 1)

vlim = maximum(abs, v_sfc)
δθlim = maximum(abs, δθ_sfc)

fig2 = Figure(size=(1200, 500))

ax1 = Axis(fig2[1, 1]; title="v eigenmode (surface)",
           xlabel="x (m)", ylabel="y (m)")
hm1 = heatmap!(ax1, v_sfc; colormap=:balance, colorrange=(-vlim, vlim))
Colorbar(fig2[1, 2], hm1; label="v (m/s)")

ax2 = Axis(fig2[1, 3]; title="δθ eigenmode (surface)",
           xlabel="x (m)", ylabel="y (m)")
hm2 = heatmap!(ax2, δθ_sfc; colormap=:balance, colorrange=(-δθlim, δθlim))
Colorbar(fig2[1, 4], hm2; label="δθ (K)")

save("power_method_convergence_eigenmode_surface.png", fig2)
nothing #hide

# ![](power_method_convergence_eigenmode_surface.png)

# ### Longitude–height cross section at the jet core

j_jet = Ny_fine ÷ 2

v_xz = view(v, :, j_jet, :)
δθ_xz = view(θ_perturbation, :, j_jet, :)

vlim_xz = maximum(abs, v_xz)
δθlim_xz = maximum(abs, δθ_xz)

fig3 = Figure(size=(1200, 500))

ax3 = Axis(fig3[1, 1]; title="v eigenmode (x–z at jet core)",
           xlabel="x (m)", ylabel="z (m)")
hm3 = heatmap!(ax3, v_xz; colormap=:balance, colorrange=(-vlim_xz, vlim_xz))
Colorbar(fig3[1, 2], hm3; label="v (m/s)")

ax4 = Axis(fig3[1, 3]; title="δθ eigenmode (x–z at jet core)",
           xlabel="x (m)", ylabel="z (m)")
hm4 = heatmap!(ax4, δθ_xz; colormap=:balance, colorrange=(-δθlim_xz, δθlim_xz))
Colorbar(fig3[1, 4], hm4; label="δθ (K)")

save("power_method_convergence_eigenmode_xz.png", fig3)
nothing #hide

# ![](power_method_convergence_eigenmode_xz.png)
