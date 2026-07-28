# # Single-column boundary layers with a prognostic-TKE closure
#
# Three boundary layers in a single column — stable, neutral and convective — driven by nothing but
# the sign of the surface heat flux. Nothing is resolved, so every eddy is parameterized, which
# makes this the cleanest place to see what [`TKEBasedTurbulenceClosure`](@ref) actually does.
#
# The closure carries one prognostic equation for the subgrid turbulent kinetic energy ``e``,
#
# ```math
# ∂e/∂t = P + B - ε + \text{transport}, \qquad P = ν S², \qquad B = -K N²,
# ```
#
# and closes the eddy viscosity on it, ``ν = Cᴷ ℓ \sqrt{e}``, with ``ℓ`` blended harmonically from a
# distance-to-the-surface branch, a turbulence branch and a buoyancy branch. The three regimes
# below exercise different terms: shear production ``P`` alone in the neutral case, ``P`` against a
# stabilizing ``B`` in the stable case, and ``B`` alone in the convective one.

using Breeze
using Oceananigans
using Oceananigans.Units
using CairoMakie

# ## The column, as a function
#
# Everything that varies between runs is a keyword argument, so the same generator serves the
# regime comparison, the coefficient comparison, and the grid-sensitivity sweep. The vertical grid
# is built from a first spacing and a stretching ratio: `Δz` ramps linearly from `Δz₁` at the
# surface to `stretching * Δz₁` at the model top.
#
# `surface_heat_flux` is kinematic, in K m s⁻¹. Passing `geostrophic_wind = nothing` drops the
# Coriolis force and the pressure-gradient forcing together, which is what free convection wants.

function single_column_simulation(; closure = TKEBasedTurbulenceClosure(),
                                    Δz₁ = 20,
                                    stretching = 4,
                                    Lz = 2000,
                                    latitude = 45,
                                    geostrophic_wind = 10,
                                    friction_velocity = 0.3,
                                    surface_heat_flux = 0,
                                    potential_temperature = 300,
                                    lapse_rate = 0.003,
                                    inversion_height = 0,
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

    ## Surface drag with a prescribed friction velocity. A surface-layer scheme would set u★ from
    ## the roughness and the surface fluxes; here we impose it so the column has one fewer moving
    ## part. The closure reads this same stress to floor the near-surface TKE.
    ##
    ## A `(Flat, Flat, Bounded)` column has no horizontal coordinates, so a bottom boundary
    ## condition is a function of time and its field dependencies alone.
    @inline ρu_drag(t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρu / max(sqrt(ρu^2 + ρv^2), 1e-6)
    @inline ρv_drag(t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρv / max(sqrt(ρu^2 + ρv^2), 1e-6)

    drag_parameters = (; ρ₀, u★ = friction_velocity)
    ρu_bc = FluxBoundaryCondition(ρu_drag, field_dependencies=(:ρu, :ρv), parameters=drag_parameters)
    ρv_bc = FluxBoundaryCondition(ρv_drag, field_dependencies=(:ρu, :ρv), parameters=drag_parameters)

    ## A positive bottom flux warms the first cell, so a positive `surface_heat_flux` is heating.
    ρe_bc = FluxBoundaryCondition(ρ₀ * cᵖ * surface_heat_flux)

    boundary_conditions = (ρu = FieldBoundaryConditions(bottom=ρu_bc),
                           ρv = FieldBoundaryConditions(bottom=ρv_bc),
                           ρe = FieldBoundaryConditions(bottom=ρe_bc))

    ## A geostrophic wind along x, balanced by Coriolis — dropped entirely for free convection
    coriolis = isnothing(geostrophic_wind) ? nothing : FPlane(; latitude)
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

    return Simulation(model; Δt, stop_time)
end

# ## Three regimes
#
# These are illustrative configurations, not reproductions of published cases. The stable one
# borrows GABLS1's geometry — a 400 m domain, an 8 m s⁻¹ geostrophic wind at 73°N, a cooled
# surface — but prescribes the surface flux directly instead of running a bulk surface-layer scheme
# against a prescribed surface temperature. The neutral one is a conventionally neutral boundary
# layer (geostrophic wind, Coriolis, no surface heat flux, capping lapse rate) but is *not* the
# Shin et al. CNBL, which uses a 12 m s⁻¹ wind at 70°N, a bulk drag law, and a 33 h integration.
# The convective one *is* a published case: Han & Bretherton (2019) §3a, whose LES gives a mixed
# layer near 294.5-295 K capped at 2.7-2.8 km after 8 h.
#
# Faithful versions of all three, with the reference comparisons that make them validation rather
# than illustration, live outside the docs — the example is here to show the closure's behaviour
# across regimes, and the runs are kept short enough to rebuild on every docs build.

regimes = [
    ## The stable flux is the middle of the GABLS1 LES ensemble's surface sensible heat flux,
    ## -12.5 to -19.6 W m⁻², converted with ρ(265 K, 10⁵ Pa) = 1.315 kg m⁻³: -0.0095 to -0.0148,
    ## so -0.012 K m s⁻¹.
    "stable"     => (; Lz = 400,  Δz₁ = 6.25, stretching = 1, latitude = 73,
                       geostrophic_wind = 8, surface_heat_flux = -0.012,
                       lapse_rate = 0.01, inversion_height = 100, Δt = 5, stop_time = 9hours),
    "neutral"    => (; Lz = 2000, Δz₁ = 20, stretching = 4,
                       geostrophic_wind = 10, surface_heat_flux = 0,
                       lapse_rate = 0.003, stop_time = 8hours),
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

# Each regime leaves a different signature, but they span very different depths — 400 m to 4 km —
# so the profiles are plotted against ``z/zᵢ``.
#
# The closure deliberately carries no ``zᵢ``: its turbulence length scale is a ``q``-weighted
# centroid, not a boundary-layer depth, which keeps a tunable coefficient from absorbing regime
# error. So the depth is diagnosed here, and each regime gets the definition its own literature
# uses — there is no single one that works everywhere. The shear-driven cases use the stress
# threshold of the GABLS1 and CNBL intercomparisons; the convective case has no wind at all, so
# stress is undefined there and the inversion height is used instead.

"""Stress-based depth: the 5% level of the peak stress, rescaled by 0.95 (GABLS1/CNBL convention)."""
function stress_depth(model)
    Nz = size(model.grid, 3)
    U = Field(sqrt(model.velocities.u^2 + model.velocities.v^2))
    compute!(U)
    Uᵥ = vec(Array(view(U, 1, 1, :)))
    ν = vec(Array(view(model.closure_fields.νₑ, 1, 1, :)))
    zc = Array(znodes(model.formulation.potential_temperature))

    ## τ = νₑ |∂z U| at faces, where νₑ lives
    ∂zUᶠ = [k == 1 ? 0.0 : (Uᵥ[k] - Uᵥ[k-1]) / (zc[k] - zc[k-1]) for k in 1:Nz]
    τ = ν[1:Nz] .* abs.(∂zUᶠ)
    τs = maximum(τ)

    k = findfirst(k -> τ[k] < 0.05τs, 2:Nz)
    return isnothing(k) ? last(zc) : zc[k+1] / 0.95
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

set_theme!(fontsize = 14, linewidth = 2.5)

fig = Figure(size = (1200, 450))

ax_θ = Axis(fig[1, 1]; xlabel = "θ - θ(z=0) (K)", ylabel = "z / zᵢ")
ax_U = Axis(fig[1, 2]; xlabel = "Wind speed (m s⁻¹)")
ax_e = Axis(fig[1, 3]; xlabel = "TKE (m² s⁻²)")
ax_ℓ = Axis(fig[1, 4]; xlabel = "Mixing length (m)")

for ax in (ax_θ, ax_U, ax_e, ax_ℓ)
    ylims!(ax, 0, 1.4)
end
[hideydecorations!(ax, grid = false) for ax in (ax_U, ax_e, ax_ℓ)]

colors = (:dodgerblue, :black, :orangered)

for ((name, simulation), (_, depth), color) in zip(simulations, depths, colors)
    model = simulation.model
    u, v, w = model.velocities
    zᵢ = depth(model)

    ## The profiles are plotted against a rescaled coordinate, so values and heights are taken
    ## separately rather than handing Makie the `Field`.
    θ = model.formulation.potential_temperature
    U = Field(sqrt(u^2 + v^2))
    compute!(U)

    ## Each regime starts from a different θ₀, so plot the departure from the surface value
    θᵥ = vec(Array(view(θ, 1, 1, :)))
    lines!(ax_θ, θᵥ .- θᵥ[1], Array(znodes(θ)) ./ zᵢ; color, label = "$name (zᵢ = $(round(Int, zᵢ)) m)")
    lines!(ax_U, vec(Array(view(U, 1, 1, :))), Array(znodes(U)) ./ zᵢ; color)
    lines!(ax_e, vec(Array(view(model.closure_fields.e, 1, 1, :))),
           Array(znodes(model.closure_fields.e)) ./ zᵢ; color)
    lines!(ax_ℓ, vec(Array(view(model.closure_fields.ℓ, 1, 1, :))),
           Array(znodes(model.closure_fields.ℓ)) ./ zᵢ; color)
end

axislegend(ax_θ; position = :rb, framevisible = false)

fig

# The convective profile is the one to look at sceptically. Its interior is *not* well mixed: an
# eddy-diffusivity closure transports heat down the gradient, so carrying a positive heat flux
# upward requires ``∂_z θ < 0``, and the mixed layer keeps a residual superadiabatic lapse. For the
# same reason it barely entrains — ``K`` collapses where the inversion is stable — so the boundary
# layer also grows too slowly.
#
# Neither is a defect in the implementation. Han & Bretherton ran exactly this ablation, a TKE
# closure with the mass-flux term removed, and report it shows "a lack of a well-mixed CBL feature
# (i.e., unstable profile throughout the whole CBL) as well as an underprediction of the CBL growth
# compared to LES" (their Fig. 3a). Both are what a mass-flux branch is for.

save("single_column_tke_boundary_layer.png", fig) #src
nothing #hide
