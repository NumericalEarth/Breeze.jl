# # Single-column boundary layers with a prognostic-TKE closure
#
# Three different canonical boundary layers in a single column — stable, neutral and convective
# — differences driven by the sign of the surface heat flux. Eddys are not resolved, highlighting
# the behavior of our [`TKEBasedTurbulenceClosure`](@ref).
#
# The closure is described in [Turbulence closures](@ref). The three regimes below exercise
# different terms of its turbulent kinetic energy budget: shear production ``P`` alone in the
# neutral case, ``P`` against a stabilizing buoyancy flux ``B`` in the stable case, and ``B`` alone
# in the convective one.

using Breeze
using Oceananigans
using Oceananigans.Units
using AtmosphericProfilesLibrary
using CairoMakie

## `BulkDrag` is exported by both Oceananigans and Breeze, so the bulk surface-layer names are
## imported explicitly rather than taken from `using Breeze`.
using Breeze.BoundaryConditions: BulkDrag, BulkSensibleHeatFlux, PolynomialCoefficient,
                                 FittedStabilityFunction

# ## Surface layer definition
#
# Breeze's bulk surface-layer scheme forms the surface virtual potential temperature from the
# *saturation* humidity at the surface temperature — correct over ocean, wrong over land. At the
# 265 K of the stable case below, that is ``0.608 q⁺ T₀ = 0.33`` K of spurious virtual warming,
# which is comparable to the entire surface-layer temperature deficit and is one-signed toward
# instability.
#
# The `surface` field of a `PolynomialCoefficient` is the trait slot for exactly this choice: it
# selects the saturation curve, and Breeze ships `PlanarLiquidSurface` and `PlanarIceSurface`. We
# add a third `DrySurface` member that has no saturation humidity at all.

struct DrySurface end
Breeze.AtmosphereModels.Diagnostics.saturation_total_specific_moisture(T, p, constants, ::DrySurface) = zero(T)

# ## Simulation constructor
#
# Everything that varies between runs is a keyword argument, so the same generator serves the
# regime comparison, the coefficient comparison, and the grid-sensitivity sweep. The vertical grid
# is built from a first spacing and a stretching ratio: `Δz` ramps linearly from `Δz₁` at the
# surface to `stretching * Δz₁` at the model top.
#
# `surface_heat_flux` is kinematic, in K m s⁻¹. Passing `geostrophic_wind = nothing` drops the
# Coriolis force and the pressure-gradient forcing together for free convection.

function single_column_simulation(; closure = TKEBasedTurbulenceClosure(),
                                    Δz₁ = 20,
                                    stretching = 4,
                                    Lz = 2000,
                                    latitude = 45,
                                    coriolis_parameter = nothing,
                                    geostrophic_wind = 10,
                                    friction_velocity = 0.3,
                                    surface_heat_flux = 0,
                                    surface_temperature = nothing,
                                    cooling_rate = 0,
                                    roughness_length = nothing,
                                    potential_temperature = 300,
                                    lapse_rate = 0.003,
                                    inversion_height = 0,
                                    initial_tke = nothing,
                                    Δt = 20,
                                    stop_time = 8hours)

    z = PiecewiseStretchedDiscretization(z = [0, Lz], Δz = [Δz₁, stretching * Δz₁])
    grid = RectilinearGrid(size = length(z) - 1; z, topology = (Flat, Flat, Bounded))

    ## Reference state: a dry, neutral adiabatic atmosphere
    θ₀ = potential_temperature
    p₀ = 1e5
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure = p₀,
                                     potential_temperature = θ₀)
    dynamics = AnelasticDynamics(reference_state)

    q₀ = zero(Breeze.Thermodynamics.MoistureMassFractions{Float64})
    ρ₀ = Breeze.Thermodynamics.density(θ₀, p₀, q₀, constants)
    cᵖ = constants.dry_air.heat_capacity

    ## The drag laws below use the closure's own von Kármán constant rather than a repeated
    ## literal, so that the surface layer they impose stays consistent with the log layer the
    ## mixing length produces. It is read back from the geometric branch, and is `nothing` for a
    ## mixing length that has none — harmless, since those configurations prescribe `u★` directly.
    κ = Breeze.TurbulenceClosures.von_karman_constant(closure.mixing_length)

    ## The surface is configured in two ways for momentum and two for heat:
    ##
    ##   * momentum — a prescribed `friction_velocity` writes the drag directly, leaving the column
    ##     one fewer moving part; a `roughness_length` instead lets u★ emerge from a bulk drag law.
    ##   * heat — a prescribed kinematic `surface_heat_flux`, or a `surface_temperature` for the
    ##     near-surface air to be differenced against.
    ##
    ## The two are not quite separable, because the bulk drag law itself takes one of two forms.
    ## Given a `surface_temperature` it is Breeze's Monin-Obukhov scheme, which maps a bulk
    ## Richardson number to ``ζ = z/L`` after Li et al. (2010) and integrates the Högström (1996)
    ## and Beljaars-Holtslag (1991) stability functions for unstable and stable conditions
    ## respectively. Without one it falls back to the neutral log law ``Cᵈ = [κ/\ln(z₁/ℓʳ)]²``,
    ## since the correction needs a surface value that a prescribed flux does not supply. Bulk drag
    ## therefore requires either a `surface_temperature` or a zero `surface_heat_flux`.
    ##
    ## A `(Flat, Flat, Bounded)` column has no horizontal coordinates, so a bottom boundary
    ## condition is a function of time and its field dependencies alone. The closure reads the
    ## surface stress, however it is set, to floor the near-surface TKE.
    if !isnothing(surface_temperature)
        surface_temperature_field = Field{Center, Center, Nothing}(grid)
        set!(surface_temperature_field, surface_temperature)
    else
        surface_temperature_field = nothing
    end

    if isnothing(roughness_length)
        @inline ρu_drag(t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρu / max(sqrt(ρu^2 + ρv^2), 1e-6)
        @inline ρv_drag(t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρv / max(sqrt(ρu^2 + ρv^2), 1e-6)

        drag_parameters = (; ρ₀, u★ = friction_velocity)
        ρu_bc = FluxBoundaryCondition(ρu_drag, field_dependencies=(:ρu, :ρv), parameters=drag_parameters)
        ρv_bc = FluxBoundaryCondition(ρv_drag, field_dependencies=(:ρu, :ρv), parameters=drag_parameters)

    elseif isnothing(surface_temperature)
        ## Neutral log-law drag referenced to the first cell center, as the conventional neutral
        ## boundary layer (CNBL) intercomparisons specify it. Skipping the stability correction is
        ## only right where the surface layer is genuinely neutral, so this path is available only
        ## for a zero surface heat flux.
        iszero(surface_heat_flux) || throw(ArgumentError(
            "bulk drag without a `surface_temperature` uses the neutral log law, which does not " *
            "apply to the non-neutral surface layer implied by `surface_heat_flux = " *
            "$surface_heat_flux`. Either prescribe a `surface_temperature`, so that the drag " *
            "carries a stability correction, or drop `roughness_length` and prescribe " *
            "`friction_velocity` instead."))

        z₁ = first(znodes(grid, Center()))
        Cᵈ = (κ / log(z₁ / roughness_length))^2
        ρu_bc = BulkDrag(coefficient = Cᵈ)
        ρv_bc = BulkDrag(coefficient = Cᵈ)

    else
        ## `PolynomialCoefficient` parameterizes the neutral 10 m transfer coefficient as
        ## ``(a₀ + a₁ U + a₂ / U) × 10⁻³``. Setting ``a₀ = 10³ [κ / \ln(10/ℓʳ)]²`` with the other two
        ## coefficients zero makes it the neutral log law, and the stability function corrects it
        ## away from neutral — which is what GABLS1 prescribes.
        ℓʳ = roughness_length
        a₀ = 1e3 * (κ / log(10 / ℓʳ))^2
        coefficient() = PolynomialCoefficient(Float64; polynomial = (a₀, 0.0, 0.0),
                                              roughness_length = ℓʳ,
                                              stability_function = FittedStabilityFunction(ℓʳ),
                                              surface = DrySurface())

        ρu_bc = BulkDrag(coefficient = coefficient(), surface_temperature = surface_temperature_field)
        ρv_bc = BulkDrag(coefficient = coefficient(), surface_temperature = surface_temperature_field)
    end

    if isnothing(surface_temperature)
        ## A positive bottom flux warms the first cell, so a positive `surface_heat_flux` is heating.
        ρe_bc = FluxBoundaryCondition(ρ₀ * cᵖ * surface_heat_flux)
    else
        ℓʳ = roughness_length
        a₀ = 1e3 * (κ / log(10 / ℓʳ))^2
        ρe_bc = BulkSensibleHeatFlux(coefficient = PolynomialCoefficient(Float64;
                                                       polynomial = (a₀, 0.0, 0.0),
                                                       roughness_length = ℓʳ,
                                                       stability_function = FittedStabilityFunction(ℓʳ),
                                                       surface = DrySurface()),
                                     surface_temperature = surface_temperature_field)
    end

    boundary_conditions = (ρu = FieldBoundaryConditions(bottom=ρu_bc),
                           ρv = FieldBoundaryConditions(bottom=ρv_bc),
                           ρe = FieldBoundaryConditions(bottom=ρe_bc))

    ## A geostrophic wind along x, balanced by Coriolis — dropped entirely for free convection.
    ## The intercomparison cases quote `f` itself rather than a latitude, so both are accepted.
    coriolis = if isnothing(geostrophic_wind)
        nothing
    elseif isnothing(coriolis_parameter)
        FPlane(; latitude)
    else
        FPlane(f = coriolis_parameter)
    end
    forcing = isnothing(geostrophic_wind) ? NamedTuple() : geostrophic_forcings(geostrophic_wind, 0)

    model = AtmosphereModel(grid; dynamics, closure, coriolis, forcing, boundary_conditions,
                            advection = nothing)

    ## A mixed layer capped by a stratified free atmosphere
    θᵢ(z) = θ₀ + lapse_rate * max(0, z - inversion_height)

    if isnothing(geostrophic_wind)
        set!(model; θ = θᵢ)
    else
        set!(model; θ = θᵢ, ρu = reference_state.density * geostrophic_wind)
    end

    ## `:ρtke` holds ρe, so a specific initial TKE profile has to be weighted by the reference
    ## density after it is set.
    if !isnothing(initial_tke)
        set!(model.tracers.ρtke, z -> max(initial_tke(z), 1e-6))
        parent(model.tracers.ρtke) .*= parent(reference_state.density)
    end

    simulation = Simulation(model; Δt, stop_time)

    ## GABLS1 cools its surface at a fixed rate rather than holding it at a fixed temperature
    if cooling_rate != 0
        T₀ = surface_temperature
        cool!(sim) = (surface_temperature_field[1, 1, 1] = T₀ - cooling_rate * time(sim) / 3600; nothing)
        add_callback!(simulation, cool!, IterationInterval(1))
    end

    return simulation
end

# ## Model evaluation in three regimes
#
# All three are published benchmarks, run to their own specifications. The CNBL spec averages its
# diagnostics over the final inertial period, while the other profiles below are instantaneous.

regimes = [
    ## Beare et al. (2006), GABLS1: a 400 m domain at f = 1.39 × 10⁻⁴ s⁻¹ under an 8 m s⁻¹
    ## geostrophic wind, θ = 265 K below 100 m over a +0.01 K m⁻¹ capping gradient, and a surface
    ## cooling from 265 K at 0.25 K h⁻¹ for 9 h over a dry ℓʳ = 0.1 m surface. The LES ensemble
    ## gives a 150-200 m boundary layer under a 9-9.5 m s⁻¹ super-geostrophic jet, with
    ## u★ ≈ 0.24-0.28 m s⁻¹ and a surface sensible heat flux of -12.5 to -19.6 W m⁻².
    "stable"     => (; Lz = 400,  Δz₁ = 6.25, stretching = 1, coriolis_parameter = 1.39e-4,
                       geostrophic_wind = 8, potential_temperature = 265,
                       surface_temperature = 265, cooling_rate = 0.25, roughness_length = 0.1,
                       lapse_rate = 0.01, inversion_height = 100,
                       initial_tke = AtmosphericProfilesLibrary.GABLS_tke(Float64),
                       Δt = 5, stop_time = 9hours),
    ## The conventionally neutral boundary layer of Shin, Yang & Howland (2025), after Liu, Gadde &
    ## Stevens (2021): a 2 km domain at f = 1.37 × 10⁻⁴ s⁻¹ (70°N) under a 12 m s⁻¹ geostrophic
    ## wind, θ = 300 K + (3 K km⁻¹)z stratified from the ground, zero surface heat flux, and
    ## neutral log-law drag over ℓʳ = 0.1 m. The spec spins up for 20 h and then averages over one
    ## inertial period 2π/f ≈ 12.74 h, so the integration runs to ≈ 32.7 h.
    "neutral"    => (; Lz = 2000, Δz₁ = 25, stretching = 4, coriolis_parameter = 1.37e-4,
                       geostrophic_wind = 12, surface_heat_flux = 0, roughness_length = 0.1,
                       potential_temperature = 300, lapse_rate = 0.003,
                       Δt = 20, stop_time = 20hours + 2π / 1.37e-4),
    ## Han & Bretherton (2019, WAF 34, 869-886) §3a: θ = 288 K + (3 K km⁻¹)z stratified from the
    ## ground, a surface buoyancy flux of 8 × 10⁻³ m² s⁻³ — kinematically w'θ'₀ = B₀θ₀/g =
    ## 0.235 K m s⁻¹ — no mean wind, 8 h, Δz = 50 m.
    "convective" => (; Lz = 4000, Δz₁ = 50, stretching = 1, potential_temperature = 288,
                       geostrophic_wind = nothing, friction_velocity = 0,
                       surface_heat_flux = 8e-3 * 288 / 9.81,
                       lapse_rate = 0.003, Δt = 5, stop_time = 8hours),
]

simulations = [name => single_column_simulation(; settings...) for (name, settings) in regimes]

for (name, simulation) in simulations
    run!(simulation)
end

## Postprocessing
#
# Each regime leaves a different signature, but they span very different depths — 400 m to 4 km —
# so the profiles are plotted against ``z/zᵢ``. We diagnose the boundary-layer depth here, with each
# regime getting the definition its own literature uses — there is no single one that works everywhere.
# The shear-driven cases use the stress threshold of the GABLS1 and CNBL intercomparisons; the
# convective case has no wind at all, so stress is undefined there and the inversion height is used
# instead.

"""Stress-based depth: the 5% level of the peak stress, rescaled by 0.95 (GABLS1/CNBL convention)."""
function stress_depth(model)
    Nz = size(model.grid, 3)
    u = vec(Array(view(model.velocities.u, 1, 1, :)))
    v = vec(Array(view(model.velocities.v, 1, 1, :)))
    Kᵘ = vec(Array(view(model.closure_fields.Kᵘ, 1, 1, :)))
    zc = Array(znodes(model.formulation.potential_temperature))

    ## The stress is a vector: τ = Kᵘ |∂z 𝐔| at faces, where Kᵘ lives. Using the gradient of the
    ## wind *speed* instead would vanish at the low-level jet, where |𝐔| peaks — but the stress
    ## does not vanish there, because the wind is still turning with height.
    ∂zᶠ(a) = [k == 1 ? 0.0 : (a[k] - a[k-1]) / (zc[k] - zc[k-1]) for k in 1:Nz]
    τ = sqrt.((Kᵘ[1:Nz] .* ∂zᶠ(u)) .^ 2 .+ (Kᵘ[1:Nz] .* ∂zᶠ(v)) .^ 2)

    ## Interpolate the crossing rather than snapping to a cell centre. At GABLS1's Δz = 6.25 m the
    ## quantisation is ~6.6 m, comparable to the differences between closure configurations, so a
    ## snapped depth reports them as identical. Searching down from the peak also keeps a secondary
    ## stress maximum from ending the search early.
    τs, kᵖ = findmax(τ)
    threshold = 0.05τs
    for k in kᵖ:Nz-1
        if τ[k] ≥ threshold > τ[k+1]
            f = (τ[k] - threshold) / (τ[k] - τ[k+1])
            return (zc[k] + f * (zc[k+1] - zc[k])) / 0.95
        end
    end
    return last(zc)
end

"""Inversion height: the level of maximum ``∂_z θ`` (convective convention)."""
function inversion_depth(model)
    θ = vec(Array(view(model.formulation.potential_temperature, 1, 1, :)))
    z = Array(znodes(model.formulation.potential_temperature))
    N = length(θ)
    ∂zθ = [(θ[min(k+1, N)] - θ[max(k-1, 1)]) / (z[min(k+1, N)] - z[max(k-1, 1)]) for k in 1:N]
    return z[argmax(∂zθ)]
end

depths = ("stable" => stress_depth, "neutral" => stress_depth, "convective" => inversion_depth)

"""Kinematic heat flux ``-Kᶜ ∂_z θ``, at faces where ``Kᶜ`` lives."""
function heat_flux(model)
    θ = vec(Array(view(model.formulation.potential_temperature, 1, 1, :)))
    Kᶜ = vec(Array(view(model.closure_fields.Kᶜ, 1, 1, :)))
    z = Array(znodes(model.formulation.potential_temperature))
    N = length(θ)
    ∂zθᶠ = [k == 1 ? 0.0 : (θ[k] - θ[k-1]) / (z[k] - z[k-1]) for k in 1:N]
    return -Kᶜ[1:N] .* ∂zθᶠ
end

## Make plots

set_theme!(fontsize = 14, linewidth = 2.5)
colors = (:dodgerblue, :black, :orangered)

fig = Figure(size = (1100, 800))

## Top row: the prognostic state, what the model carries forward in time.
ax_θ = Axis(fig[1, 1]; xlabel = "θ - θ(z=0) (K)", ylabel = "z / zᵢ")
ax_U = Axis(fig[1, 2]; xlabel = "Wind speed (m s⁻¹)")
ax_e = Axis(fig[1, 3]; xlabel = "TKE (m² s⁻²)")

## Bottom row: what the closure makes of it, in the order it is built — the length scale, the
## diffusivity formed from it, and the flux they produce. `Kᶜ` spans two orders of magnitude between
## the stable and convective regimes, so it is scaled by its own maximum in each: the height axis
## already normalizes by `zᵢ`, and this makes the horizontal axis a shape comparison to match. A
## linear axis keeps the collapse at `zᵢ` looking like the cliff it is, which a logarithmic one
## would smooth into a gentle slide. The magnitudes that scaling divides out are annotated on the panel.
ax_ℓ = Axis(fig[2, 1]; xlabel = "Mixing length ℓ (m)", ylabel = "z / zᵢ")
ax_K = Axis(fig[2, 2]; xlabel = "Kᶜ / max(Kᶜ)")
ax_J = Axis(fig[2, 3]; xlabel = "w′θ′ / (w′θ′)₀")

for ax in (ax_θ, ax_U, ax_e, ax_K, ax_ℓ, ax_J)
    ylims!(ax, 0, 1.5)
end
xlims!(ax_J, -0.25, 1.3)
xlims!(ax_K, -0.03, 1.08)
[hideydecorations!(ax, grid = false) for ax in (ax_U, ax_e, ax_K, ax_J)]

## Reference for the *convective* case only: the mixed-layer flux is near-linear from 1 at the
## surface to -A at zᵢ, with the entrainment ratio A ≈ 0.17 (Soares et al. 2004) to 0.2. The stable
## case has no such result and is not judged against this line.
lines!(ax_J, [1, -0.2], [0, 1]; color = :gray50, linestyle = :dash)
vlines!(ax_J, [0]; color = :gray80, linewidth = 1)

legend_labels = String[]
diffusivity_labels = String[]

for ((name, simulation), (_, settings), (_, depth), color) in zip(simulations, regimes, depths, colors)
    model = simulation.model
    u, v, w = model.velocities
    zᵢ = depth(model)
    push!(legend_labels, "$name (zᵢ = $(round(Int, zᵢ)) m)")

    ## The profiles are plotted against a rescaled coordinate, so values and heights are taken
    ## separately rather than handing Makie the `Field`.
    θ = model.formulation.potential_temperature
    U = Field(sqrt(u^2 + v^2))
    compute!(U)

    ## Each regime starts from a different θ₀, so plot the departure from the surface value
    θᵥ = vec(Array(view(θ, 1, 1, :)))
    lines!(ax_θ, θᵥ .- θᵥ[1], Array(znodes(θ)) ./ zᵢ; color)
    lines!(ax_U, vec(Array(view(U, 1, 1, :))), Array(znodes(U)) ./ zᵢ; color)
    lines!(ax_e, vec(Array(view(model.closure_fields.e, 1, 1, :))),
           Array(znodes(model.closure_fields.e)) ./ zᵢ; color)
    Kᶜ = vec(Array(view(model.closure_fields.Kᶜ, 1, 1, :)))
    zᴷ = Array(znodes(model.closure_fields.Kᶜ))
    kᵐᵃˣ = argmax(Kᶜ)
    push!(diffusivity_labels,
          "Kᶜ(z = $(round(zᴷ[kᵐᵃˣ] / zᵢ, digits = 2)) zᵢ) = $(round(Kᶜ[kᵐᵃˣ], digits = 1)) m² s⁻¹")
    lines!(ax_K, Kᶜ ./ Kᶜ[kᵐᵃˣ], zᴷ ./ zᵢ; color)
    lines!(ax_ℓ, vec(Array(view(model.closure_fields.ℓ, 1, 1, :))),
           Array(znodes(model.closure_fields.ℓ)) ./ zᵢ; color)
    ## The surface flux is prescribed in the convective case but emergent in the stable one, where
    ## the bulk scheme sets it, so both are normalized by the closure's own flux at the lowest
    ## interior face rather than by a setting. The neutral case is defined as having no surface heat
    ## flux, so it gets no curve here — the test is on the specification rather than on the
    ## diagnosed flux, whose residual 6 × 10⁻⁵ K m s⁻¹ of entrainment would otherwise be normalized
    ## by itself and plotted as a spurious profile of order one.
    thermally_driven = get(settings, :surface_heat_flux, 0) != 0 || haskey(settings, :surface_temperature)
    if thermally_driven
        J = heat_flux(model)
        lines!(ax_J, J ./ J[2], Array(znodes(θ)) ./ zᵢ; color)
    end
end

## Built explicitly from the regime list rather than from plotted labels, so that every regime is
## listed with its diagnosed depth even though the flux panel carries only two curves.
axislegend(ax_e, [LineElement(color = c) for c in colors], legend_labels;
           position = :rt, framevisible = false)

## Above zᵢ every scaled diffusivity has collapsed to the axis, so the top of that panel is free for
## the magnitude and the height the scaling removed. Colors match the legend above.
for (n, (label, color)) in enumerate(zip(diffusivity_labels, colors))
    text!(ax_K, 0.10, 1.43 - 0.10 * (n - 1); text = label, color, fontsize = 12,
          align = (:left, :center))
end

save("single_column_tke_boundary_layer.png", fig) #src
fig
