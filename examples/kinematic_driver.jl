# # Kinematic driver: cloud formation in an idealized updraft
#
# In atmospheric modeling, we sometimes want to isolate microphysics and thermodynamics
# from dynamics. **Kinematic models** prescribe the velocity field rather than solving
# momentum equations, letting us focus purely on tracer transport and phase changes.
#
# This example demonstrates Breeze's [`PrescribedDynamics`](@ref) formulation by simulating
# cloud formation in an idealized updraft. A uniform vertical velocity lifts moist air
# through a realistic temperature profile. As air rises and cools, water vapor
# condenses to form clouds — a fundamental process driving all precipitation on Earth.
#
# ## Physical setup
#
# We simulate a 1D column representing a rising air parcel with:
# - A realistic potential temperature profile (troposphere + stratosphere)
# - Uniform upward velocity `W₀ = 2 m/s` (a gentle cumulus updraft)
# - Moist boundary layer air entering from below
#
# The **divergence correction** option compensates for the non-zero mass flux divergence
# ``𝛁·(ρ𝐔)`` that arises when velocity doesn't vary with the reference density profile.
# This is essential for physically consistent tracer advection in kinematic models.

using Breeze
using CairoMakie
using Printf

# ## Grid and reference state
#
# We construct a 20 km tall column extending from the surface through the tropopause
# into the lower stratosphere. The reference state establishes the background
# pressure and density profile based on a hydrostatically-balanced atmosphere.

Nz = 100
Lz = 20000 # 20 km domain height
grid = RectilinearGrid(CPU(); size=Nz, x=0, y=0, z=(0, Lz),
                       topology=(Flat, Flat, Bounded))

constants = ThermodynamicConstants()
θ₀ = 300   # Surface potential temperature (K)
p₀ = 1e5   # Surface pressure (Pa)
reference_state = ReferenceState(grid, constants;
                                 surface_pressure=p₀,
                                 potential_temperature=θ₀)

# ## Prescribing dynamics with divergence correction
#
# The key feature of kinematic models is [`PrescribedDynamics`](@ref), which fixes
# the density and pressure fields from a reference state. We enable
# `divergence_correction=true` because our constant vertical velocity doesn't
# satisfy the anelastic continuity constraint ``𝛁·(ρ𝐔) = 0``.
#
# Without this correction, the tracer equation would see spurious sources/sinks
# from the non-zero velocity divergence. The correction adds a term ``c 𝛁·(ρ𝐔)``
# that compensates for the prescribed velocity field's divergence.

W₀ = 2 # Vertical velocity (m/s) — a gentle updraft
dynamics = PrescribedDynamics(reference_state; divergence_correction=true)

# ## Boundary conditions
#
# The key boundary condition is at the surface: we prescribe incoming moist air
# with constant potential temperature and specific humidity. This represents
# the boundary layer air being lifted into the updraft.

ρ₀ = surface_density(reference_state)

# Surface boundary conditions for tracers
qᵗ₀ = 0.018 # Incoming specific humidity (18 g/kg) — typical tropical boundary layer
ρθ_bcs = FieldBoundaryConditions(bottom=ValueBoundaryCondition(ρ₀ * θ₀))
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ValueBoundaryCondition(ρ₀ * qᵗ₀))
w_bcs = FieldBoundaryConditions(bottom=OpenBoundaryCondition(W₀), top=OpenBoundaryCondition(W₀))

# ## Microphysics: warm-phase saturation adjustment
#
# We use [`SaturationAdjustment`](@ref) with [`WarmPhaseEquilibrium`](@ref), which
# instantaneously partitions total water between vapor and liquid based on
# saturation. When air becomes supersaturated, excess vapor condenses to cloud
# liquid, releasing latent heat. This captures the essence of cloud formation
# without explicit condensation timescales.

microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())

# ## Building the atmosphere model
#
# We assemble all components into an [`AtmosphereModel`](@ref). The combination
# of `PrescribedDynamics` with microphysics creates a powerful tool for
# understanding cloud processes in isolation from dynamics.

model = AtmosphereModel(grid; dynamics, microphysics,
                        advection = WENO(order=5),
                        boundary_conditions = (ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, w=w_bcs),
                        thermodynamic_constants = constants)

# ## Initial conditions
#
# We initialize with a realistic tropospheric potential temperature profile
# that increases with height (stable stratification). Above the tropopause
# at 12 km, we switch to a stratospheric profile. The initial moisture
# decreases with height, typical of a real atmosphere.

zᵗʳ = 12000 # Tropopause height (m)
θᵗʳ = 343   # Potential temperature at tropopause (K)
Tᵗʳ = 213   # Temperature at tropopause (K)
g = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity

function θ_initial(z)
    θ_troposphere = θ₀ + (θᵗʳ - θ₀) * (z / zᵗʳ)^(5/4)
    θ_stratosphere = θᵗʳ * exp(g / (cᵖᵈ * Tᵗʳ) * (z - zᵗʳ))
    return ifelse(z <= zᵗʳ, θ_troposphere, θ_stratosphere)
end

# Moisture profile: high in the boundary layer, decreasing with height
function qᵗ_initial(z)
    z_scale = 3000 # Scale height for moisture (m)
    return qᵗ₀ * exp(-z / z_scale)
end

set!(model; θ=θ_initial, qᵗ=qᵗ_initial, w=W₀)

# ## Running the simulation
#
# We run for 60 minutes, enough time for air parcels to rise several kilometers
# and for a quasi-steady cloud layer to develop.

simulation = Simulation(model; Δt=1, stop_time=60*60, verbose=false)

θ = model.formulation.potential_temperature
qˡ = model.microphysical_fields.qˡ
qᵛ = model.microphysical_fields.qᵛ

times = Float64[]
θ_data, qˡ_data, qᵛ_data = [], [], []

function record_profiles(sim)
    push!(times, time(sim))
    push!(θ_data, Array(interior(θ, 1, 1, :)))
    push!(qˡ_data, Array(interior(qˡ, 1, 1, :)))
    push!(qᵛ_data, Array(interior(qᵛ, 1, 1, :)))
end

add_callback!(simulation, record_profiles, TimeInterval(10*60))

function progress(sim)
    qˡ_max = maximum(qˡ)
    θ_surf = θ[1, 1, 1]
    @info @sprintf("t = %s, θ_surface = %.1f K, max(qˡ) = %.2e kg/kg",
                   prettytime(sim), θ_surf, qˡ_max)
end

add_callback!(simulation, progress, TimeInterval(10*60))

@info "Running kinematic updraft simulation with cloud microphysics..."
run!(simulation)

# ## Visualization
#
# The results reveal the physics of adiabatic cloud formation. The left panel
# shows how potential temperature evolves — influenced by latent heat release
# where clouds form. The center panel shows cloud liquid mixing ratio,
# clearly revealing the cloud base and cloud layer. The right panel shows
# water vapor, which decreases sharply above the cloud base where condensation occurs.

z_km = znodes(grid, Center()) ./ 1000
fig = Figure(size=(1000, 450))

ax_θ = Axis(fig[1, 1]; xlabel="θ (K)", ylabel="z (km)",
            title="Potential temperature")
ax_qˡ = Axis(fig[1, 2]; xlabel="qˡ (g/kg)", ylabel="z (km)",
             title="Cloud liquid", yticklabelsvisible=false)
ax_qᵛ = Axis(fig[1, 3]; xlabel="qᵛ (g/kg)", ylabel="z (km)",
             title="Water vapor", yticklabelsvisible=false)

colors = cgrad(:viridis, length(times), categorical=true)

for (n, t) in enumerate(times)
    t_min = Int(t / 60)
    lines!(ax_θ, θ_data[n], z_km; color=colors[n], linewidth=2, label="t = $t_min min")
    lines!(ax_qˡ, qˡ_data[n] .* 1000, z_km; color=colors[n], linewidth=2)
    lines!(ax_qᵛ, qᵛ_data[n] .* 1000, z_km; color=colors[n], linewidth=2)
end

# Add tropopause marker
for ax in [ax_θ, ax_qˡ, ax_qᵛ]
    hlines!(ax, [zᵗʳ/1000]; color=:gray, linestyle=:dash, linewidth=1.5, label="Tropopause")
end

Legend(fig[1, 4], ax_θ; framevisible=false)

Label(fig[0, :], "Kinematic updraft (W₀ = $W₀ m/s) with warm-phase saturation adjustment";
      fontsize=18, tellwidth=false)

save("kinematic_driver.png", fig)
fig

# ![](kinematic_driver.png)

# ## Discussion
#
# The kinematic driver framework enables focused study of cloud microphysics by
# decoupling them from dynamical feedbacks. Key observations from this simulation:
#
# 1. **Cloud base formation**: Moist boundary layer air rises and cools adiabatically.
#    When it reaches its Lifting Condensation Level (LCL), condensation begins
#    and cloud liquid appears. The sharp transition in qˡ marks the cloud base.
#
# 2. **Moisture partitioning**: Above the cloud base, total water is partitioned
#    between vapor (at saturation) and liquid (the excess). Water vapor decreases
#    with height because saturation vapor pressure decreases with temperature.
#
# 3. **Potential temperature**: Initially, θ increases with height. As the simulation
#    progresses, latent heat release from condensation modifies the temperature
#    profile within the cloud layer.
#
# 4. **Divergence correction**: Without `divergence_correction=true`, the constant
#    velocity field would create spurious tracer sources because ``𝛁·(ρ𝐔) ≠ 0``.
#    The correction adds a compensating term to the tracer equations.
#
# This setup is analogous to classic parcel theory experiments in cloud physics,
# but resolved on a grid. It's particularly useful for:
# - Testing and validating microphysics schemes in isolation
# - Understanding sensitivities to initial moisture and temperature
# - Pedagogical demonstrations of cloud formation physics
