# # Power method with Fourier projection
#
# Same as `power_method_baroclinic.jl` but after each iteration we
# **project perturbations onto a target wavenumber band** via FFT along
# longitude. This isolates the baroclinic eigenmode from the zonally
# symmetric (m = 0) mode and other wavenumber contamination, letting the
# power method converge on the full symmetric DCMIP2016 jet without any
# hemispheric filtering.
#
# The target wavenumber `m_target` (default 9) can be changed to study
# other wavenumbers, or extended to a range of wavenumbers by editing the
# `mask` below.

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans: prognostic_fields
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: update_state!, reset!
using Printf
using CairoMakie
using CUDA
using FFTW

# ## DCMIP2016 parameters

Oceananigans.defaults.FloatType = Float32
Oceananigans.defaults.gravitational_acceleration = 9.80616
Oceananigans.defaults.planet_radius = 6371220
Oceananigans.defaults.planet_rotation_rate = 7.29212e-5

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287)

g   = constants.gravitational_acceleration
Rᵈ  = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ
p₀  = 1e5
a   = Oceananigans.defaults.planet_radius
Ω   = Oceananigans.defaults.planet_rotation_rate

# ## Grid

Nλ = 360; Nφ = 150; Nz = 128

H = 30kilometers

grid = LatitudeLongitudeGrid(GPU();
                             size = (Nλ, Nφ, Nz),
                             halo = (5, 5, 5),
                             longitude = (0, 360),
                             latitude = (-75, 75),
                             z = (0, H))

# ## Analytic initial conditions

Tᴱ = 310
Tᴾ = 240
Tᴹ = (Tᴱ + Tᴾ) / 2
Γ  = 0.005
K  = 3
b  = 2

function τ_and_integrals(z)
    Hˢ = Rᵈ * Tᴹ / g
    η  = z / (b * Hˢ)
    e  = exp(-η^2)

    A = (Tᴹ - Tᴾ) / (Tᴹ * Tᴾ)
    C = (K + 2) * (Tᴱ - Tᴾ) / (2 * Tᴱ * Tᴾ)

    τ₁  = A * (1 - 2η^2) * e + exp(Γ * z / Tᴹ) / Tᴹ
    ∫τ₁ = A * z * e + (exp(Γ * z / Tᴹ) - 1) / Γ

    τ₂  = C * (1 - 2η^2) * e
    ∫τ₂ = C * z * e

    return τ₁, τ₂, ∫τ₁, ∫τ₂
end

F(φ)  = cosd(φ)^K - K / (K + 2) * cosd(φ)^(K + 2)
dF(φ) = cosd(φ)^(K - 1) - cosd(φ)^(K + 1)

function virtual_temperature(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z)
    return 1 / (τ₁ - τ₂ * F(φ))
end

function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-g / Rᵈ * (∫τ₁ - ∫τ₂ * F(φ)))
end

density(λ, φ, z) = pressure(λ, φ, z) / (Rᵈ * virtual_temperature(λ, φ, z))

potential_temperature(λ, φ, z) = virtual_temperature(λ, φ, z) * (p₀ / pressure(λ, φ, z))^κ

function zonal_velocity_balanced(λ, φ, z)
    _, _, _, ∫τ₂ = τ_and_integrals(z)
    Tᵥ = virtual_temperature(λ, φ, z)
    U = g / a * K * ∫τ₂ * dF(φ) * Tᵥ
    rcosφ  = a * cosd(φ)
    Ωrcosφ = Ω * rcosφ
    return -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ * U)
end

# ## Model

coriolis = SphericalCoriolis(rotation_rate=Ω)

T₀ᵣ = 250
θᵣ(z) = T₀ᵣ * exp(g * z / (cᵖᵈ * T₀ᵣ))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure = p₀,
                                reference_potential_temperature = θᵣ)

model = AtmosphereModel(grid; dynamics, coriolis,
                        thermodynamic_constants = constants,
                        advection = WENO())

# ## Balanced state + background snapshot

set!(model; θ=potential_temperature, u=zonal_velocity_balanced, ρ=density)

background = map(f -> copy(parent(f)), prognostic_fields(model))

# ## Initial perturbation — wavenumber 9

v_ref = 1.2  ## m/s — Park et al. reference amplitude

function v_perturbation(λ, φ, z)
    zₚ = 15000
    taper = ifelse(z < zₚ, 1 - 3 * (z / zₚ)^2 + 2 * (z / zₚ)^3, zero(z))
    return v_ref * exp(-(φ - 40)^2 / 225) * sind(9λ) * taper
end

set!(model; v=v_perturbation)

# ## Simulation

Δτ = 3days

simulation = Simulation(model; Δt=12minutes, stop_time=Δτ)
conjure_time_step_wizard!(simulation; cfl=1.4, max_Δt=12minutes)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

function power_method_progress(sim)
    v = sim.model.velocities.v
    v_sfc = maximum(abs, view(v, :, :, 1))
    @info @sprintf("  step %5d | t = %s | Δt = %s | sfc max|v| = %.4f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), v_sfc)
    return nothing
end

add_callback!(simulation, power_method_progress, IterationInterval(100))

# ## Fourier projection setup
#
# After each power iteration we project all perturbation fields onto
# wavenumber `m_target` by zeroing all other Fourier coefficients.
# Edit `m_target` or the `mask` to retain a different wavenumber or
# a range of wavenumbers.

m_target = 9  ## wavenumber to retain

halo_λ = grid.Hx
halo_φ = grid.Hy
halo_z = grid.Hz

iλ = halo_λ .+ (1:Nλ)
iφ = halo_φ .+ (1:Nφ)
iz = halo_z .+ (1:Nz)

## Build the frequency mask — retain only ±m_target
mask = zeros(Bool, Nλ)
mask[m_target + 1] = true       ## positive frequency m_target
mask[Nλ - m_target + 1] = true  ## negative frequency -m_target
mask_3d = reshape(mask, Nλ, 1, 1)

# ## Power iteration with Fourier projection

max_iterations = 80
convergence_threshold = 0.001  ## relative change in σ
σ_history = Float64[]

for n in 1:max_iterations
    run!(simulation)

    ## Fourier-project all perturbation fields onto wavenumber m_target
    for (f, bg) in zip(prognostic_fields(model), background)
        pf = parent(f)
        perturbation = Array(pf[iλ, iφ, iz] .- bg[iλ, iφ, iz])
        F̂ = fft(perturbation, 1)
        F̂ .= F̂ .* mask_3d
        filtered = real(ifft(F̂, 1))
        pf[iλ, iφ, iz] .= bg[iλ, iφ, iz] .+ CuArray(Float32.(filtered))
    end

    ## Measure max|v| of the m=9 component at the lowest model level (k=1)
    v_sfc_max = maximum(abs, view(model.velocities.v, :, :, 1))

    ## Growth rate
    σ = log(v_sfc_max / v_ref) / Δτ
    push!(σ_history, σ)

    @info @sprintf("Power iteration %3d | σ = %.4f day⁻¹ | sfc max|v| = %.4e m/s",
                   n, σ * 86400, v_sfc_max)

    ## Rescale all prognostic perturbations to reference amplitude
    scale = v_ref / v_sfc_max
    for (f, bg) in zip(prognostic_fields(model), background)
        parent(f) .= bg .+ scale .* (parent(f) .- bg)
    end

    ## Fill halos after all modifications
    for f in prognostic_fields(model)
        fill_halo_regions!(f)
    end

    ## Check convergence
    converged = n ≥ 2 && abs(σ_history[end] - σ_history[end-1]) / abs(σ_history[end]) < convergence_threshold

    if converged
        update_state!(model, compute_tendencies=false)
        @info @sprintf("Converged after %d iterations (σ = %.4f day⁻¹)", n, σ * 86400)
        break
    end

    ## Reset for next iteration
    reset!(model.clock)
    update_state!(model, compute_tendencies=false)
    simulation.stop_time = Δτ
end

# ## Visualization

# ### Convergence

fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1];
          xlabel = "Power iteration",
          ylabel = "Growth rate σ (day⁻¹)",
          title = "Power method convergence (m=$m_target projection)")

σ_per_day = σ_history .* 86400
lines!(ax, 1:length(σ_per_day), σ_per_day; linewidth=2, color=:dodgerblue)
scatter!(ax, 1:length(σ_per_day), σ_per_day; markersize=6, color=:dodgerblue)
hlines!(ax, [0.46]; linestyle=:dash, color=:gray60, label="Park et al. (2013)")
axislegend(ax; position=:rb)

save("power_method_projection_convergence.png", fig)
nothing

# ### Eigenmode structure

v = model.velocities.v
θ_field = Field(PotentialTemperature(model))
compute!(θ_field)

θ_perturbation = Field{Center, Center, Center}(model.grid)
set!(θ_perturbation, potential_temperature)
parent(θ_perturbation) .= parent(θ_field) .- parent(θ_perturbation)

v_sfc = view(v, :, :, 1)
δθ_sfc = view(θ_perturbation, :, :, 1)

vlim = maximum(abs, v_sfc)
δθlim = maximum(abs, δθ_sfc)

fig2 = Figure(size=(1200, 500))

ax1 = Axis(fig2[1, 1]; title="v eigenmode (surface)",
           xlabel="Longitude", ylabel="Latitude")
hm1 = heatmap!(ax1, v_sfc; colormap=:balance, colorrange=(-vlim, vlim))
Colorbar(fig2[1, 2], hm1; label="v (m/s)")

ax2 = Axis(fig2[1, 3]; title="δθ eigenmode (surface)",
           xlabel="Longitude", ylabel="Latitude")
hm2 = heatmap!(ax2, δθ_sfc; colormap=:balance, colorrange=(-δθlim, δθlim))
Colorbar(fig2[1, 4], hm2; label="δθ (K)")

save("power_method_projection_eigenmode.png", fig2)
nothing

# ### Longitude–height cross section at the jet core

j_jet = Nφ ÷ 2 + 45  ## ≈ 45° N, near the jet maximum

v_xz = view(v, :, j_jet, :)
δθ_xz = view(θ_perturbation, :, j_jet, :)

vlim_xz = maximum(abs, v_xz)
δθlim_xz = maximum(abs, δθ_xz)

fig3 = Figure(size=(1200, 500))

ax3 = Axis(fig3[1, 1]; title="v eigenmode (λ–z at jet core)",
           xlabel="Longitude", ylabel="z (m)")
hm3 = heatmap!(ax3, v_xz; colormap=:balance, colorrange=(-vlim_xz, vlim_xz))
Colorbar(fig3[1, 2], hm3; label="v (m/s)")

ax4 = Axis(fig3[1, 3]; title="δθ eigenmode (λ–z at jet core)",
           xlabel="Longitude", ylabel="z (m)")
hm4 = heatmap!(ax4, δθ_xz; colormap=:balance, colorrange=(-δθlim_xz, δθlim_xz))
Colorbar(fig3[1, 4], hm4; label="δθ (K)")

save("power_method_projection_eigenmode_xz.png", fig3)
nothing
