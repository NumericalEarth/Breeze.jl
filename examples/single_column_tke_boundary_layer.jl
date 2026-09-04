# # Single-column boundary layers with a prognostic-TKE closure
#
# This example runs three canonical atmospheric boundary layers — stable, neutral and convective
# — in a single column, each configured as a published intercomparison case. A single column
# resolves no turbulence, so every turbulent flux is carried by [`TKEBasedTurbulenceClosure`](@ref):
# a vertical eddy-diffusivity closure with a prognostic equation for turbulent kinetic energy,
# described in [Turbulence closures](@ref). The three regimes exercise different terms of its
# turbulent kinetic energy budget: shear production ``P`` alone in the neutral case, ``P`` against
# a stabilizing buoyancy flux ``B`` in the stable case, and ``B`` alone in the convective one.
#
# The example demonstrates
#
#   * How to set up a single-column `AtmosphereModel` on a `(Flat, Flat, Bounded)` grid.
#   * How to drive a boundary layer with a geostrophic wind, a bulk surface layer, and a surface
#     heat flux or a cooling surface.
#   * How to diagnose the closure's mixing length, diffusivities and fluxes from the model state.

using Breeze
using Oceananigans.Units
using CairoMakie

# The mixing length is not stored by the closure, so we diagnose it below by evaluating the
# closure's own kernel function, which takes the buoyancy and its tracers as arguments.

using Breeze.TurbulenceClosures: mixing_lengthᶜᶜᶠ
using Oceananigans.TurbulenceClosures: buoyancy_tracers, buoyancy_force

# ## Three published cases
#
# Each column follows the specification of an intercomparison or reference study, so that its
# profiles can be judged against published large-eddy simulations.
#
# The stable case is GABLS1 ([Beare et al. 2006](@cite Beare2006)): a 400 m column at
# ``f = 1.39 × 10⁻⁴`` s⁻¹ under an 8 m s⁻¹ geostrophic wind, with ``θ = 265`` K below 100 m and a
# 0.01 K m⁻¹ gradient above, over a dry surface with a 0.1 m roughness length that cools from 265 K
# at 0.25 K h⁻¹ for 9 hours. The turbulent kinetic energy starts from the profile the case
# prescribes, ``0.4 (1 - z / 250)³`` m² s⁻² below 250 m. The large-eddy simulations of the
# intercomparison produce a 150–200 m deep layer beneath a 9–9.5 m s⁻¹ low-level jet.
#
# The neutral case is the conventionally neutral boundary layer of
# [Shin, Yang and Howland (2025)](@cite ShinYangHowland2025), after
# [Liu, Gadde and Stevens (2021)](@cite LiuGaddeStevens2021): a 2 km column at
# ``f = 1.37 × 10⁻⁴`` s⁻¹ under a 12 m s⁻¹ geostrophic wind, stratified at 3 K km⁻¹ from the ground,
# with zero surface heat flux and neutral log-law drag over a 0.1 m roughness length. The
# specification spins up for 20 hours and then averages over one inertial period ``2π / f``; we run
# to the end of that period and plot the instantaneous state.
#
# The convective case is the dry convective boundary layer of
# [Han and Bretherton (2019)](@cite HanBretherton2019): ``θ = 288`` K + 3 K km⁻¹ ``z`` with no
# mean wind, heated for 8 hours by a surface buoyancy flux of ``8 × 10⁻³`` m² s⁻³, which is the
# kinematic heat flux ``B₀ θ₀ / g = 0.235`` K m s⁻¹, at 50 m resolution.

cases = (
    stable = (; Lz = 400, Δz₁ = 6.25, θ₀ = 265, Γ = 0.01, zᵢ = 100, f = 1.39e-4, uᵍ = 8,
                surface_temperature = 265, cooling_rate = 0.25 / hour,
                initial_tke = z -> 0.4 * max(0, 1 - z / 250)^3, stop_time = 9hours),

    neutral = (; Lz = 2000, Δz₁ = 25, stretching = 4, θ₀ = 300, Γ = 0.003, f = 1.37e-4, uᵍ = 12,
                 stop_time = 20hours + 2π / 1.37e-4),

    convective = (; Lz = 4000, Δz₁ = 50, θ₀ = 288, Γ = 0.003, surface_heat_flux = 8e-3 * 288 / 9.81,
                    stop_time = 8hours),
)

# ## Surface layer
#
# The surface exchanges momentum and heat with the first cell through bulk formulae whose transfer
# coefficients are built from a roughness length ``ℓʳ`` and the von Kármán constant ``κ``. The
# closure itself carries no von Kármán constant: its neutral logarithmic layer has the value
# ``(Cᵘ³ / Cᴰ)^{1/4}`` implied by its stability functions, 0.40 by default.
#
# The neutral case prescribes the neutral log-law drag coefficient referenced to the first cell
# center, ``Cᵈ = [κ / \ln(z₁ / ℓʳ)]²``, which the column builder below computes from its grid. The
# stable case prescribes Monin–Obukhov similarity instead, which Breeze supplies through a
# `PolynomialCoefficient`: its polynomial ``(a₀ + a₁ U + a₂ / U) × 10⁻³`` is the neutral 10 m
# transfer coefficient, so ``a₀ = 10³ [κ / \ln(10 / ℓʳ)]²`` with the other two zero is the neutral
# log law, and a `FittedStabilityFunction` corrects it away from neutral. A `moisture_availability`
# of zero declares a dry surface, whose humidity is that of the air above it rather than the
# saturation humidity at the surface temperature. Over a saturated surface at 265 K that humidity
# would add 0.3 K of spurious virtual warming, comparable to the whole surface-layer temperature
# deficit.

κ = 0.4  # von Kármán constant
ℓʳ = 0.1 # m, roughness length
Cᵈ = (κ / log(10 / ℓʳ))^2 # the neutral log-law drag coefficient, referenced to 10 m

## The polynomial is evaluated in Large and Yeager's units of 10⁻³, so its constant term is 10³ Cᵈ
monin_obukhov_coefficient = PolynomialCoefficient(polynomial = 1e3 * Cᵈ .* (1, 0, 0),
                                                  roughness_length = ℓʳ,
                                                  stability_function = FittedStabilityFunction(ℓʳ),
                                                  moisture_availability = 0)

# ## Building a column
#
# Everything that differs between the cases is a keyword argument of one function. The vertical
# grid ramps linearly from a spacing `Δz₁` at the surface to `stretching * Δz₁` at the top. The
# initial potential temperature is `θ₀` up to the inversion height `zᵢ` and increases at the lapse
# rate `Γ` above it. A `surface_temperature` selects the Monin–Obukhov surface layer; otherwise the
# drag is the neutral log law and the heat flux is the prescribed kinematic `surface_heat_flux`.
# The stable case cools its surface at a fixed rate, which a callback applies to the surface
# temperature field every iteration.
#
# The closure adds the tracer `ρe`, the density-weighted turbulent kinetic energy, to the model,
# so a specific initial profile is weighted by the reference density after it is set. Vertical
# diffusion and the sinks of turbulent kinetic energy are treated implicitly, so every column
# takes one-minute time steps.

function boundary_layer_simulation(; Lz, Δz₁, θ₀, Γ, stop_time, stretching = 1, zᵢ = 0, f = 0, uᵍ = 0,
                                     surface_temperature = nothing, cooling_rate = 0,
                                     surface_heat_flux = 0, initial_tke = 0)

    z = PiecewiseStretchedDiscretization(z = [0, Lz], Δz = [Δz₁, stretching * Δz₁])
    grid = RectilinearGrid(size = length(z) - 1; z, topology = (Flat, Flat, Bounded))

    reference_state = ReferenceState(grid, surface_pressure = 1e5, potential_temperature = θ₀)
    dynamics = AnelasticDynamics(reference_state)

    if isnothing(surface_temperature)
        T₀ = nothing
        z₁ = first(znodes(grid, Center()))
        coefficient = (κ / log(z₁ / ℓʳ))^2
        ρθ_bc = FluxBoundaryCondition(surface_density(reference_state) * surface_heat_flux)
    else
        T₀ = Field{Center, Center, Nothing}(grid)
        set!(T₀, surface_temperature)
        coefficient = monin_obukhov_coefficient
        ρθ_bc = BulkSensibleHeatFlux(; coefficient, surface_temperature = T₀)
    end

    ρu_bc = BulkDrag(; coefficient, surface_temperature = T₀)
    boundary_conditions = (ρu = FieldBoundaryConditions(bottom = ρu_bc),
                           ρv = FieldBoundaryConditions(bottom = ρu_bc),
                           ρθ = FieldBoundaryConditions(bottom = ρθ_bc))

    model = AtmosphereModel(grid; dynamics, boundary_conditions,
                            closure = TKEBasedTurbulenceClosure(),
                            coriolis = FPlane(; f),
                            forcing = geostrophic_forcings(uᵍ, 0))

    θᵢ(z) = θ₀ + Γ * max(0, z - zᵢ)
    set!(model, θ = θᵢ, u = uᵍ, ρe = initial_tke)
    set!(model.tracers.ρe, reference_state.density * model.tracers.ρe)

    simulation = Simulation(model, Δt = 1minute; stop_time)
    cool!(sim) = set!(T₀, surface_temperature - cooling_rate * time(sim))
    iszero(cooling_rate) || add_callback!(simulation, cool!)

    return simulation
end

simulations = map(case -> boundary_layer_simulation(; case...), cases)

for simulation in simulations
    run!(simulation)
end

# ## Boundary-layer depth
#
# The three layers span very different depths, a few hundred meters to a few kilometers, so the
# profiles are compared against ``z / hᵇˡ``, with the depth ``hᵇˡ`` diagnosed the way each case's
# literature defines it. GABLS1 and the neutral intercomparison take the height at which the stress
# falls to 5% of its maximum, divided by 0.95. The stress is a vector, ``|τ| = Kᵘ |∂_z 𝐮|``, so it
# does not vanish at the low-level jet, where the wind speed peaks but the wind is still turning
# with height. The convective case has no wind, so its depth is the height of the inversion, where
# ``∂_z θ`` is largest.

function stress_depth(model)
    u, v, w = model.velocities
    τ = Field(model.closure_fields.Kᵘ * sqrt(∂z(u)^2 + ∂z(v)^2))
    τₖ = interior(τ, 1, 1, :)
    kᵖ = argmax(τₖ)
    k = findnext(τ -> τ < 0.05 * τₖ[kᵖ], τₖ, kᵖ)
    return znodes(τ)[k] / 0.95
end

function inversion_depth(model)
    ∂zθ = Field(∂z(model.formulation.potential_temperature))
    return znodes(∂zθ)[argmax(interior(∂zθ, 1, 1, :))]
end

depths = (stable = stress_depth, neutral = stress_depth, convective = inversion_depth)

# ## Visualization
#
# The top row shows what the model carries forward in time; the bottom row shows what the closure
# makes of it, in the order it is built: the mixing length, the diffusivity formed from it, and the
# heat flux they produce. The diffusivity spans two orders of magnitude between the stable and
# convective cases, so each is scaled by its own maximum, which the legend records, and the heat
# flux is scaled by its value at the first interior face. The dashed line is the mixed-layer
# reference for the convective case only: a flux decreasing linearly from the surface to an
# entrainment ratio of about 0.2 at the inversion. The neutral case has no surface heat flux to
# normalize by, and so no flux curve.

set_theme!(fontsize = 14, linewidth = 2.5)
colors = (stable = :dodgerblue, neutral = :black, convective = :orangered)

fig = Figure(size = (1100, 800))
ax_θ = Axis(fig[1, 1]; xlabel = "θ - θ(z=0) (K)", ylabel = "z / hᵇˡ")
ax_U = Axis(fig[1, 2]; xlabel = "Wind speed (m s⁻¹)")
ax_e = Axis(fig[1, 3]; xlabel = "TKE (m² s⁻²)")
ax_ℓ = Axis(fig[2, 1]; xlabel = "Mixing length ℓ (m)", ylabel = "z / hᵇˡ")
ax_K = Axis(fig[2, 2]; xlabel = "Kᶜ / max(Kᶜ)")
ax_J = Axis(fig[2, 3]; xlabel = "w′θ′ / (w′θ′)₀")

for ax in (ax_θ, ax_U, ax_e, ax_ℓ, ax_K, ax_J)
    ylims!(ax, 0, 1.5)
end
for ax in (ax_U, ax_e, ax_K, ax_J)
    hideydecorations!(ax, grid = false)
end

xlims!(ax_J, -0.25, 1.3)
lines!(ax_J, [1, -0.2], [0, 1]; color = :gray50, linestyle = :dash)
vlines!(ax_J, [0]; color = :gray80, linewidth = 1)

for (name, simulation) in pairs(simulations)
    model = simulation.model
    u, v, w = model.velocities
    θ = model.formulation.potential_temperature
    Kᶜ = model.closure_fields.Kᶜ
    hᵇˡ = depths[name](model)
    color = colors[name]

    U = Field(sqrt(u^2 + v^2))
    e = Field(model.tracers.ρe / model.dynamics.reference_state.density)
    ℓ = Field(KernelFunctionOperation{Center, Center, Face}(mixing_lengthᶜᶜᶠ, model.grid, model.closure,
                                                              e, buoyancy_tracers(model), buoyancy_force(model)))
    Jᶿ = Field(- Kᶜ * ∂z(θ))

    label = "$name: hᵇˡ = $(round(Int, hᵇˡ)) m, max Kᶜ = $(round(Int, maximum(Kᶜ))) m² s⁻¹"
    lines!(ax_θ, Field(θ - θ[1, 1, 1]), znodes(θ) ./ hᵇˡ; color)
    lines!(ax_U, U, znodes(U) ./ hᵇˡ; color)
    lines!(ax_e, e, znodes(e) ./ hᵇˡ; color, label)
    lines!(ax_ℓ, ℓ, znodes(ℓ) ./ hᵇˡ; color)
    lines!(ax_K, Field(Kᶜ / maximum(Kᶜ)), znodes(Kᶜ) ./ hᵇˡ; color)
    name == :neutral || lines!(ax_J, Field(Jᶿ / Jᶿ[1, 1, 2]), znodes(Jᶿ) ./ hᵇˡ; color)
end

axislegend(ax_e, position = :rt, framevisible = false)

save("single_column_tke_boundary_layer.png", fig) #src
fig
