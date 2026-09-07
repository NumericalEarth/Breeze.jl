# # Single-column boundary layers with a prognostic-TKE closure
#
# A single column resolves no turbulence, so every turbulent flux in it is carried by the
# closure — here [`TKEBasedTurbulenceClosure`](@ref), a vertical eddy-diffusivity closure with a
# prognostic equation for turbulent kinetic energy, described in [Turbulence closures](@ref).
# This example puts it through three acts of increasing realism:
#
# 1. Three canonical dry boundary layers — stable, neutral and convective — each configured as a
#    published intercomparison case, run with both of the closure's coefficient sets: the
#    Mellor–Yamada constants of Nakanishi and Niino and the Richardson-number-dependent stability
#    functions of CATKE.
# 2. Two cloud-topped boundary layers from the library of GCM-forced large-eddy simulations of
#    [Shen et al. (2022)](@cite Shen2022), a stratocumulus deck off California and a trade-cumulus
#    column over the open Pacific, driven in single-column mode by the forcing the LES saw and
#    compared with the LES in detail.
# 3. The whole CNRM-CM6-1 slice of that library — 21 sites along the Pacific transect, four months —
#    as an ensemble of single columns, summarized by the error of each member against its LES.
#
# The example demonstrates
#
#   * How to set up a single-column `AtmosphereModel` on a `(Flat, Flat, Bounded)` grid.
#   * How to drive a boundary layer with a geostrophic wind, a bulk surface layer, and a surface
#     heat flux or a cooling surface; and how to drive one with the large-scale forcing, radiative
#     heating and surface fluxes of a large-eddy simulation.
#   * How to diagnose the closure's mixing length, diffusivities, fluxes and TKE budget from the
#     model state, and how to choose between its coefficient sets and static stabilities.

using Breeze
using Oceananigans.Units
using CairoMakie
using Statistics

# The mixing length is not stored by the closure, so we diagnose it below by evaluating the
# closure's own kernel function, which takes the specific TKE and the stored static stability.

using Breeze.TurbulenceClosures: mixing_lengthᶜᶜᶠ

# ## Two coefficient sets
#
# The closure's diffusivities are ``K = S ℓ \sqrt{e}`` with one primary mixing length
# ``ℓ = \min(Cˢ z, \sqrt{e} / N)`` and a stability function ``S`` for each of momentum, scalars and
# TKE, plus one for the dissipation. The default [`ConstantStabilityFunctions`](@ref) are the
# Mellor–Yamada constants of Nakanishi and Niino, which put the neutral log layer at ``κ = 0.40``
# with ``e / u_\star² = 4.2``. [`RiDependentStabilityFunctions`](@ref) are CATKE's: each function
# takes one value in unstable air, another at neutral, and ramps to a third as the Richardson
# number grows, so that stratification lengthens the dissipation length and raises the Prandtl
# number. `catke_parameters()` supplies them together with CATKE's wall coefficient. Everything
# below is run with both.

closures = (NN09 = TKEBasedTurbulenceClosure(),
            CATKE = TKEBasedTurbulenceClosure(; catke_parameters()...))

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
# ``Cˢ (Cᵘ³ / Cᴰ)^{1/4}`` implied by its mixing length and stability functions, 0.40 by default.
#
# The neutral case prescribes the neutral log-law drag coefficient referenced to the first cell
# center, ``Cᵈ = [κ / \ln(z₁ / ℓʳ)]²``, which the column builder below computes from its grid. The
# stable case prescribes Monin–Obukhov similarity instead, which Breeze supplies through a
# `PolynomialCoefficient`: its polynomial ``a₀ + a₁ U + a₂ / U`` is the neutral 10 m transfer
# coefficient, so ``a₀ = Cᵈ = [κ / \ln(10 / ℓʳ)]²`` with the other two zero is the neutral
# log law, and a `FittedStabilityFunction` corrects it away from neutral. A `moisture_availability`
# of zero declares a dry surface, whose humidity is that of the air above it rather than the
# saturation humidity at the surface temperature. Over a saturated surface at 265 K that humidity
# would add 0.3 K of spurious virtual warming, comparable to the whole surface-layer temperature
# deficit.

κ = 0.4  # von Kármán constant
ℓʳ = 0.1 # m, roughness length
Cᵈ = (κ / log(10 / ℓʳ))^2 # the neutral log-law drag coefficient, referenced to 10 m

monin_obukhov_coefficient = PolynomialCoefficient(polynomial = Cᵈ .* (1, 0, 0),
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

function boundary_layer_simulation(; closure, Lz, Δz₁, θ₀, Γ, stop_time, stretching = 1, zᵢ = 0, f = 0, uᵍ = 0,
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

    model = AtmosphereModel(grid; dynamics, boundary_conditions, closure,
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

simulations = map(closures) do closure
    map(case -> boundary_layer_simulation(; closure, case...), cases)
end

for closure_simulations in simulations, simulation in closure_simulations
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

# ## The dry cases
#
# The top row shows what the model carries forward in time; the bottom row shows what the closure
# makes of it, in the order it is built: the mixing length, the diffusivity formed from it, and the
# heat flux they produce. Solid lines are the Nakanishi–Niino constants, dashed lines CATKE's
# functions, both against ``z / hᵇˡ`` with the depth of the default closure's layer. The
# diffusivity spans two orders of magnitude between the stable and convective cases, so each is
# scaled by the default closure's maximum, which the legend records, and the heat flux by its
# value at the first interior face. The gray dashed line is the mixed-layer reference for the
# convective case only: a flux decreasing linearly from the surface to an entrainment ratio of
# about 0.2 at the inversion. The neutral case has no surface heat flux to normalize by, and so no
# flux curve.
#
# CATKE's functions mix the stable layer less and let it grow less deep, since its Prandtl number
# rises with the Richardson number, and they carry less turbulent kinetic energy everywhere:
# ``e / u_\star²`` is a third of the constants' in the neutral surface layer, and the mixing length
# is shorter by the ratio of the wall coefficients. Without CATKE's convective length scale, the
# convective layer is mixed by ``Cˢ z`` alone.

set_theme!(fontsize = 14, linewidth = 2.5)
colors = (stable = :dodgerblue, neutral = :black, convective = :orangered)
linestyles = (NN09 = :solid, CATKE = :dash)

function specific_tke(model)
    return Field(model.tracers.ρe / model.dynamics.reference_state.density)
end

function mixing_length(model)
    e = specific_tke(model)
    return Field(KernelFunctionOperation{Center, Center, Face}(mixing_lengthᶜᶜᶠ, model.grid, model.closure,
                                                                 e, model.closure_fields.N²))
end

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

for name in keys(cases)
    reference_model = simulations.NN09[name].model
    hᵇˡ = depths[name](reference_model)
    Kᶜ_max = maximum(reference_model.closure_fields.Kᶜ)
    color = colors[name]

    for (set, closure_simulations) in pairs(simulations)
        model = closure_simulations[name].model
        u, v, w = model.velocities
        θ = model.formulation.potential_temperature
        Kᶜ = model.closure_fields.Kᶜ
        linestyle = linestyles[set]

        U = Field(sqrt(u^2 + v^2))
        e = specific_tke(model)
        ℓ = mixing_length(model)
        Jᶿ = Field(- Kᶜ * ∂z(θ))

        label = "$name: hᵇˡ = $(round(Int, hᵇˡ)) m, max Kᶜ = $(round(Int, Kᶜ_max)) m² s⁻¹"
        lines!(ax_θ, Field(θ - θ[1, 1, 1]), znodes(θ) ./ hᵇˡ; color, linestyle)
        lines!(ax_U, U, znodes(U) ./ hᵇˡ; color, linestyle)
        if set == :NN09
            lines!(ax_e, e, znodes(e) ./ hᵇˡ; color, linestyle, label)
        else
            lines!(ax_e, e, znodes(e) ./ hᵇˡ; color, linestyle)
        end
        lines!(ax_ℓ, ℓ, znodes(ℓ) ./ hᵇˡ; color, linestyle)
        lines!(ax_K, Field(Kᶜ / Kᶜ_max), znodes(Kᶜ) ./ hᵇˡ; color, linestyle)
        name == :neutral || lines!(ax_J, Field(Jᶿ / Jᶿ[1, 1, 2]), znodes(Jᶿ) ./ hᵇˡ; color, linestyle)
    end
end

axislegend(ax_e, position = :rt, framevisible = false)
axislegend(ax_ℓ, [LineElement(linestyle = :solid), LineElement(linestyle = :dash)], ["Nakanishi–Niino constants", "CATKE functions"],
           position = :rb, framevisible = false)

save("single_column_tke_boundary_layer.png", fig) #src
fig

# ## A library of cloud-topped boundary layers
#
# Dry boundary layers exercise the closure's mechanics; clouds are where the choices in it bite.
# [Shen et al. (2022)](@cite Shen2022) ran large-eddy simulations of the boundary layer at 22
# points along the GPCI transect across the Pacific — off Peru, through the deep tropics, and from
# the California stratocumulus deck out into the trade cumulus — each forced by a global climate
# model's large-scale state for four months of the year and two climates, and published the
# library (CC0, [doi:10.22002/D1.20052](https://doi.org/10.22002/D1.20052)). The CNRM-CM6-1 slice of
# it, reduced to what a single column needs (see `validation/cloud_les_library`), is the
# `cloud_les_library` artifact: for each member the large-scale forcing the LES saw, its radiative
# heating, its surface fluxes, its initial and time-mean profiles, and the fluxes and TKE budget it
# produced.

using NCDatasets
using Pkg.Artifacts: ensure_artifact_installed, artifact_hash, artifact_path

artifacts_toml = joinpath(pkgdir(Breeze), "Artifacts.toml")
ensure_artifact_installed("cloud_les_library", artifacts_toml)
library = artifact_path(artifact_hash("cloud_les_library", artifacts_toml))

member(site, month) = joinpath(library, "cfsite$(lpad(site, 2, '0'))_CNRM-CM6-1_amip_$(month).nc")

# ## Driving a column the way the LES was driven
#
# The LES protocol is reproduced term by term, with the LES's own output standing in for
# everything the LES computed for itself, so that the closure is the only unknown in the column:
#
# * large-scale subsidence, horizontal advection and the GCM's vertical eddy advection ("fluctuation")
#   tendencies of temperature and moisture, which are time-invariant, as forcings;
# * the LES's radiative heating, with its diurnal cycle, as an hourly time series;
# * the LES's surface sensible and latent heat fluxes and momentum fluxes, prescribed;
# * relaxation of the winds toward the LES mean on 6 hours everywhere, and of temperature and
#   moisture on 24 hours above the boundary layer, ramping in between 3.0 and 3.5 km — as in the
#   LES — plus a 10-minute sponge in the top 200 m, the inflow boundary of a subsiding column,
#   without which the top cell is unstable to the centered subsidence term.
#
# Temperature tendencies are supplied as static-energy forcings (`s`), which the model converts to
# its potential temperature. Microphysics is warm-phase saturation adjustment; the LES also had
# drizzle, which is one reason its stratocumulus is thinner than a non-precipitating column's.

function interpolated(zs, vs)
    return function (z)
        z ≤ zs[1] && return vs[1]
        z ≥ zs[end] && return vs[end]
        k = searchsortedlast(zs, z)
        w = (z - zs[k]) / (zs[k+1] - zs[k])
        return (1 - w) * vs[k] + w * vs[k+1]
    end
end

# The diagnostics of the column are averaged over the LES's own analysis window, the final two
# days, by a callback that accumulates them every ten minutes.

mutable struct TimeAverages{D, S}
    diagnostics :: D
    sums :: S
    count :: Int
    start :: Float64
end

TimeAverages(diagnostics, start) =
    TimeAverages(diagnostics, map(f -> zeros(length(interior(f, 1, 1, :))), diagnostics), 0, start)

function (averages::TimeAverages)(simulation)
    time(simulation) < averages.start && return nothing
    for (sum, field) in zip(averages.sums, averages.diagnostics)
        compute!(field)
        sum .+= interior(field, 1, 1, :)
    end
    averages.count += 1
    return nothing
end

time_means(averages) = map(sum -> sum ./ averages.count, averages.sums)

using Breeze.TurbulenceClosures: shear_productionᶜᶜᶠ, buoyancy_productionᶜᶜᶠ, dissipationᶜᶜᶜ

function les_driven_column(path; closure, Δt = 1minute)
    ds = NCDataset(path)
    z = ds["z"][:]
    t = ds["time"][:]
    Nz = length(z)
    Lz = z[end] + (z[2] - z[1]) / 2
    grid = RectilinearGrid(size = Nz, z = (0, Lz), topology = (Flat, Flat, Bounded))

    constants = ThermodynamicConstants()
    g = constants.gravitational_acceleration
    cᵖᵈ = constants.dry_air.heat_capacity

    ## The LES reference pressure extrapolated to the surface, and the initial mixed-layer θ
    p₀ = ds["p0"][1] + ds["rho0"][1] * g * z[1]
    θ₀ = ds["thetali_mean_initial"][1]
    reference_state = ReferenceState(grid; surface_pressure = p₀, potential_temperature = θ₀)
    dynamics = AnelasticDynamics(reference_state)
    ρ₀ = surface_density(reference_state)
    microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())

    profile(name) = interpolated(z, ds[name][:])

    ## Large-scale forcing
    wˢ = Field{Nothing, Nothing, Face}(grid)
    set!(wˢ, profile("ls_subsidence"))
    subsidence = SubsidenceForcing(wˢ)

    dTdt = Field{Nothing, Nothing, Center}(grid)
    set!(dTdt, interpolated(z, cᵖᵈ .* (ds["dtdt_hadv"][:] .+ ds["dtdt_fluc"][:])))
    dqdt = Field{Nothing, Nothing, Center}(grid)
    set!(dqdt, interpolated(z, ds["dqtdt_hadv"][:] .+ ds["dqtdt_fluc"][:]))

    ## The LES's radiative heating, hourly, padded to the end of the run
    t_end = ds.attrib["target_window_end"]
    times = vcat(t, t_end)
    dTdt_rad = FieldTimeSeries{Center, Center, Center}(grid, times)
    heating = ds["dtdt_rad_hourly"][:, :]
    for n in eachindex(times)
        set!(dTdt_rad[n], reshape(cᵖᵈ .* heating[:, min(n, length(t))], 1, 1, Nz))
    end

    ## Relaxation toward the LES means: winds on 6 h, thermodynamics on 24 h above 3 km, and the
    ## 10-minute sponge in the top 200 m
    uₙ, vₙ, θₙ, qₙ = profile("u_mean_nudge"), profile("v_mean_nudge"), profile("thetali_mean_nudge"), profile("qt_mean_nudge")
    τˢ = 10minutes
    ramp(z) = z < 3000 ? 0.0 : z > 3500 ? 1.0 : (1 - cos(π * (z - 3000) / 500)) / 2
    sponge(z) = max(0, (z - (Lz - 200)) / 200)
    wind_mask(z) = τˢ / 6hours + sponge(z)
    thermodynamic_mask(z) = ramp(z) * τˢ / 24hours + sponge(z)
    relax_u = Relaxation(rate = 1 / τˢ, mask = wind_mask, target = (z, t) -> uₙ(z))
    relax_v = Relaxation(rate = 1 / τˢ, mask = wind_mask, target = (z, t) -> vₙ(z))
    relax_θ = Relaxation(rate = 1 / τˢ, mask = thermodynamic_mask, target = (z, t) -> θₙ(z))
    relax_q = Relaxation(rate = 1 / τˢ, mask = thermodynamic_mask, target = (z, t) -> qₙ(z))

    forcing = (u = (subsidence, relax_u),
               v = (subsidence, relax_v),
               θ = (subsidence, relax_θ),
               qᵉ = (subsidence, Forcing(dqdt), relax_q),
               s = (Forcing(dTdt), Forcing(dTdt_rad)))

    ## Surface fluxes prescribed from the LES time series
    series(name) = interpolated(t, ds[name][:])
    shf, lhf, uw, vw = series("shf_surface_mean"), series("lhf_surface_mean"), series("uw_surface_mean"), series("vw_surface_mean")
    ℒᵛ = Breeze.Thermodynamics.liquid_latent_heat(ds["surface_temperature"][1], constants)
    prescribed(flux) = FluxBoundaryCondition((i, j, grid, clock, fields) -> flux(clock.time), discrete_form = true)
    boundary_conditions = (ρs = FieldBoundaryConditions(bottom = prescribed(shf)),
                           ρqᵉ = FieldBoundaryConditions(bottom = prescribed(t -> lhf(t) / ℒᵛ)),
                           ρu = FieldBoundaryConditions(bottom = prescribed(t -> ρ₀ * uw(t))),
                           ρv = FieldBoundaryConditions(bottom = prescribed(t -> ρ₀ * vw(t))))

    model = AtmosphereModel(grid; dynamics, microphysics, closure, forcing, boundary_conditions, advection = nothing)

    set!(model; θ = profile("thetali_mean_initial"), qᵗ = profile("qt_mean_initial"),
                u = profile("u_mean_initial"), v = profile("v_mean_initial"), ρe = 1e-3)
    set!(model.tracers.ρe, reference_state.density * model.tracers.ρe)

    simulation = Simulation(model; Δt, stop_time = t_end, verbose = false)

    ## Diagnostics: the state, the closure's fluxes, and the terms of its TKE budget
    Kᵘ, Kᶜ, N² = model.closure_fields.Kᵘ, model.closure_fields.Kᶜ, model.closure_fields.N²
    u, v, w = model.velocities
    qᵗ = model.microphysical_fields.qᵉ
    e = Field(model.tracers.ρe / reference_state.density)
    diagnostics = (θˡ = model.formulation.potential_temperature,
                   qᵗ = qᵗ,
                   qˡ = model.microphysical_fields.qˡ,
                   u = u,
                   e = e,
                   Kᶜ = Kᶜ,
                   w′qᵗ′ = Field(- Kᶜ * ∂z(qᵗ)),
                   P = Field(KernelFunctionOperation{Center, Center, Face}(shear_productionᶜᶜᶠ, grid, Kᵘ, u, v)),
                   B = Field(KernelFunctionOperation{Center, Center, Face}(buoyancy_productionᶜᶜᶠ, grid, Kᶜ, N²)),
                   ε = Field(KernelFunctionOperation{Center, Center, Center}(dissipationᶜᶜᶜ, grid, closure, e, model.velocities, N²)))

    averages = TimeAverages(diagnostics, ds.attrib["target_window_start"])
    add_callback!(simulation, averages, IterationInterval(10))

    close(ds)
    return simulation, averages
end

# The LES targets are the time means over the same window; the error of a column is the root mean
# square of its difference from the LES below 3 km, above which both are relaxed to the same state.

function les_means(path)
    ds = NCDataset(path)
    means = (z = ds["z"][:],
             θˡ = ds["thetali_mean"][:],
             qᵗ = ds["qt_mean"][:],
             qˡ = ds["ql_mean"][:],
             u = ds["u_mean"][:],
             e = ds["tke_mean"][:],
             w′qᵗ′ = ds["qt_flux_z_mean"][:] .+ ds["qt_sgs_flux_z_mean"][:],
             P = ds["tke_prod_S_mean"][:],
             B = ds["tke_prod_B_mean"][:],
             ## The LES budget's residual is its dissipation: the shear and buoyancy production,
             ## the transport and pressure terms, and what the SGS scheme diffuses
             ε = -(ds["tke_prod_S_mean"][:] .+ ds["tke_prod_B_mean"][:] .+ ds["tke_prod_T_mean"][:] .+
                   ds["tke_prod_P_mean"][:] .+ ds["tke_prod_A_mean"][:] .+ ds["tke_prod_D_mean"][:]),
             cloud_fraction = mean(ds["cloud_fraction"][ds["time"][:] .≥ ds.attrib["target_window_start"]]),
             surface_temperature = mean(ds["surface_temperature"][:]))
    close(ds)
    return means
end

function rmse(column, les, z; below = 3000)
    k = z .≤ below
    return sqrt(mean((column[k] .- les[k]).^2))
end

# ## Four configurations
#
# Two coefficient sets, and two static stabilities: the dry buoyancy gradient
# ([`DryStaticStability`](@ref)), which treats a saturated parcel like a dry one, and the saturated
# buoyancy frequency ([`MoistStaticStability`](@ref), the default), in which a rising parcel
# condenses and its latent heating offsets part of the stratification.

configurations = (
    NN09 = TKEBasedTurbulenceClosure(static_stability = DryStaticStability()),
    CATKE = TKEBasedTurbulenceClosure(; catke_parameters()..., static_stability = DryStaticStability()),
    NN09_moist = TKEBasedTurbulenceClosure(),
    CATKE_moist = TKEBasedTurbulenceClosure(; catke_parameters()...),
)

configuration_colors = (NN09 = :dodgerblue, CATKE = :orangered, NN09_moist = :dodgerblue, CATKE_moist = :orangered)
configuration_linestyles = (NN09 = :solid, CATKE = :solid, NN09_moist = :dash, CATKE_moist = :dash)
configuration_labels = (NN09 = "Nakanishi–Niino, dry N²", CATKE = "CATKE, dry N²",
                        NN09_moist = "Nakanishi–Niino, moist N²", CATKE_moist = "CATKE, moist N²")

# ## Two members in detail
#
# Site 17 in July is the stratocumulus deck off California: cloud fraction 0.96, a 460–1190 m cloud,
# 4 W m⁻² of sensible and 63 W m⁻² of latent heat flux over 291 K water. Site 22 in July, six
# hundred kilometers further out over 297.5 K water, is trade cumulus: cloud fraction 0.2, bases near
# 500 m and tops at 2.3 km. Between them lies the transition the library was built to span.

details = (stratocumulus = member(17, "07"), cumulus = member(22, "07"))

detail_columns = map(details) do path
    map(configurations) do closure
        simulation, averages = les_driven_column(path; closure)
        run!(simulation)
        (; simulation, means = time_means(averages))
    end
end

detail_les = map(les_means, details)

# The top row is the state — liquid-water potential temperature, total water, cloud liquid — and the
# bottom row the turbulence: the turbulent kinetic energy, the total-water flux, and the TKE budget of
# shear production ``P``, buoyancy flux ``B`` and dissipation ``-ε`` against the LES's, whose
# dissipation is the residual of its budget. Solid lines are the dry static stability, dashed the
# moist; blue is Nakanishi–Niino, red CATKE; black is the LES. The flux and budget axes are scaled
# to the LES, and the budget is shown for the moist-``N²`` configurations; the dry ones run off the
# axes, for reasons the figures make plain.

function detail_figure(name, columns, les)
    fig = Figure(size = (1200, 800))
    grid = columns.NN09.simulation.model.grid
    zc = znodes(grid, Center())
    zf = znodes(grid, Face())
    ax_θ = Axis(fig[1, 1]; xlabel = "θˡ (K)", ylabel = "z (m)", title = string(name))
    ax_q = Axis(fig[1, 2]; xlabel = "qᵗ (g kg⁻¹)")
    ax_l = Axis(fig[1, 3]; xlabel = "qˡ (g kg⁻¹)")
    ax_e = Axis(fig[2, 1]; xlabel = "TKE (m² s⁻²)", ylabel = "z (m)")
    ax_F = Axis(fig[2, 2]; xlabel = "w′qᵗ′ (g kg⁻¹ m s⁻¹)")
    ax_b = Axis(fig[2, 3]; xlabel = "TKE budget (10⁻⁴ m² s⁻³)")

    ztop = 1.6 * max(les.z[argmax(diff(les.θˡ))], 800)
    for ax in (ax_θ, ax_q, ax_l, ax_e, ax_F, ax_b)
        ylims!(ax, 0, ztop)
    end
    for ax in (ax_q, ax_l, ax_F, ax_b)
        hideydecorations!(ax, grid = false)
    end

    ## Axes scaled to the LES below the top of the plot
    below = les.z .≤ ztop
    xlims!(ax_θ, minimum(les.θˡ[below]) - 1, maximum(les.θˡ[below]) + 1)
    xlims!(ax_F, -0.5e3 * maximum(abs, les.w′qᵗ′[below]), 3e3 * maximum(abs, les.w′qᵗ′[below]))
    budget_scale = 3e4 * maximum(abs, vcat(les.P[below], les.B[below], les.ε[below]))
    xlims!(ax_b, -budget_scale, budget_scale)

    les_kw = (color = :black, linewidth = 3.5)
    lines!(ax_θ, les.θˡ, les.z; les_kw..., label = "LES")
    lines!(ax_q, 1e3 .* les.qᵗ, les.z; les_kw...)
    lines!(ax_l, 1e3 .* les.qˡ, les.z; les_kw...)
    lines!(ax_e, les.e, les.z; les_kw...)
    lines!(ax_F, 1e3 .* les.w′qᵗ′, les.z; les_kw...)
    lines!(ax_b, 1e4 .* les.P, les.z; les_kw..., linestyle = :solid)
    lines!(ax_b, 1e4 .* les.B, les.z; les_kw..., linestyle = :dash)
    lines!(ax_b, -1e4 .* les.ε, les.z; les_kw..., linestyle = :dot)
    vlines!(ax_b, [0]; color = :gray80, linewidth = 1)

    for (set, column) in pairs(columns)
        m = column.means
        color = configuration_colors[set]
        linestyle = configuration_linestyles[set]
        lines!(ax_θ, m.θˡ, zc; color, linestyle, label = configuration_labels[set])
        lines!(ax_q, 1e3 .* m.qᵗ, zc; color, linestyle)
        lines!(ax_l, 1e3 .* m.qˡ, zc; color, linestyle)
        lines!(ax_e, m.e, zc; color, linestyle)
        lines!(ax_F, 1e3 .* m.w′qᵗ′, zf; color, linestyle)
        if linestyle == :dash
            lines!(ax_b, 1e4 .* m.P, zf; color, linestyle = :solid)
            lines!(ax_b, 1e4 .* m.B, zf; color, linestyle = :dash)
            lines!(ax_b, -1e4 .* m.ε, zc; color, linestyle = :dot)
        end
    end

    axislegend(ax_θ, position = :lt, framevisible = false)
    axislegend(ax_b, [LineElement(linestyle = :solid), LineElement(linestyle = :dash), LineElement(linestyle = :dot)],
               ["P", "B", "−ε"], position = :rt, framevisible = false)
    return fig
end

fig = detail_figure(:stratocumulus, detail_columns.stratocumulus, detail_les.stratocumulus)
save("single_column_tke_stratocumulus.png", fig) #src
fig

# With the dry static stability, both coefficient sets stop the boundary layer about a hundred meters
# short of the LES inversion. The cloud layer of a stratocumulus deck is stably stratified in ``θᵨ`` —
# condensation warms it toward the moist adiabat — so the stratification length shuts the mixing off
# just where the LES's turbulence is strongest, at cloud top, where radiative cooling drives it. The
# closure carries no turbulent kinetic energy there, the layer never reaches the inversion, and the
# cloud is a trace. With the moist static stability the saturated layer is nearly neutral to the
# displacements the closure represents: the layer deepens to the LES inversion and turbulent kinetic
# energy peaks at cloud top as it does in the LES — CATKE's functions overshoot the peak by a factor
# of two — but the layer now entrains too much dry air across the inversion, its upper half ends up
# half a gram per kilogram drier than the LES, subsaturated, and the cloud is gone. The LES also
# drizzles, which removes water from its cloud and cannot be why the column is drier still. The
# total-water flux tells the same story from the other side: the dry configurations' flux is several
# times the LES's and flickers between adjacent levels in the upper half of the layer — a grid-scale
# staircase that a stability-function closure builds wherever ``N²`` is set by small differences in
# condensation — while the moist configurations' flux is smooth through the layer and of the LES's
# size, and flickers only in the entrainment zone at its top.

fig = detail_figure(:cumulus, detail_columns.cumulus, detail_les.cumulus)
save("single_column_tke_cumulus.png", fig) #src
fig

# The trade-cumulus column separates the two static stabilities completely. With the dry one the layer
# is capped at 1.6–1.7 km, half a kilometer below the LES's cloud tops; moisture piles up beneath the
# cap until the whole layer saturates into a stratocumulus with a gram per kilogram of liquid that the
# LES does not have, and the fluxes in it flicker at the grid scale. With the moist one the saturated
# layer is near-neutral and ``θˡ`` and ``qᵗ`` follow the LES up through the cloud layer to the trade
# inversion — overshooting it by two or three hundred meters, and with two to three times the LES's
# turbulent kinetic energy in the cloud layer, since a local closure can only represent the cumulus
# layer's transport as diffusion down the mean gradients. Its fluxes and buoyancy production still
# flicker from one interface to the next through the cloud layer: a layer that hovers at saturation
# switches between the two branches of ``N²`` from level to level, and only the mean state is smooth.
# The coefficient sets are a second-order distinction in both columns.

# ## The ensemble
#
# Every member of the `amip` climate — 21 sites and four months, 83 columns — is run with the four
# configurations, and each is scored by the root-mean-square error of its time-mean ``θˡ`` and
# ``qᵗ`` below 3 km against its LES. The sites are laid out as in Shen et al. (2022): 2–4 lie off
# Peru, 11–15 in the deep tropics near the ITCZ, 17–18 under the California stratocumulus, and
# 19–23 across the northeast Pacific trades. Site 16 is not in the library and site 15 has no
# January member.

sites = [2:15; 17:23]
months = ["01", "04", "07", "10"]

errors = Dict(name => (θˡ = fill(NaN, length(sites), length(months)),
                       qᵗ = fill(NaN, length(sites), length(months))) for name in keys(configurations))

for (i, site) in enumerate(sites), (j, month) in enumerate(months)
    path = member(site, month)
    isfile(path) || continue
    les = les_means(path)
    for (name, closure) in pairs(configurations)
        simulation, averages = les_driven_column(path; closure)
        run!(simulation)
        means = time_means(averages)
        z = znodes(simulation.model.grid, Center())
        errors[name].θˡ[i, j] = rmse(means.θˡ, les.θˡ, z)
        errors[name].qᵗ[i, j] = 1e3 * rmse(means.qᵗ, les.qᵗ, z)
    end
end

# The heatmaps put the sites along the transect on the horizontal axis and the months on the
# vertical, one panel per configuration, with a common color scale for each variable so that the
# panels can be read against each other.

function error_heatmaps(variable, label)
    fig = Figure(size = (1300, 620))
    layout = (NN09 = (1, 1), CATKE = (1, 2), NN09_moist = (2, 1), CATKE_moist = (2, 2))
    colorrange = (0, maximum(filter(isfinite, vcat((vec(errors[name][variable]) for name in keys(configurations))...))))
    month_names = ["January", "April", "July", "October"]
    for (name, (row, col)) in pairs(layout)
        ax = Axis(fig[row, col]; title = configuration_labels[name],
                  xticks = (1:length(sites), string.(sites)), yticks = (1:length(months), month_names),
                  xlabel = row == 2 ? "cfSite" : "")
        heatmap!(ax, 1:length(sites), 1:length(months), errors[name][variable]; colormap = :viridis, colorrange)
        row == 2 || hidexdecorations!(ax, grid = false)
        col == 1 || hideydecorations!(ax, grid = false)
    end
    Colorbar(fig[:, 3]; colormap = :viridis, colorrange, label)
    return fig
end

fig = error_heatmaps(:θˡ, "RMSE of θˡ below 3 km (K)")
save("single_column_tke_ensemble_theta.png", fig) #src
fig

# And the same for total water:

fig = error_heatmaps(:qᵗ, "RMSE of qᵗ below 3 km (g kg⁻¹)")
save("single_column_tke_ensemble_moisture.png", fig) #src
fig

# The pattern of the two columns holds along the whole transect. With the dry static stability the
# error grows from the coasts into the tropics and the trades, where the boundary layer is deep and
# cumulus-topped, and CATKE's functions reduce it a little. With the moist static stability the error
# drops by half across the ensemble with either coefficient set, and most of its dependence on regime
# disappears — except in the deepest tropical columns in April (sites 14 and 15), where the LES cloud
# layer reaches 3 km and a local closure that treats a saturated layer as neutral mixes the whole
# column; there the dry static stability, whatever its faults, errs less. Summarized across the
# ensemble, the median errors of the four configurations are

for (name, closure) in pairs(configurations)
    θ_error = median(filter(isfinite, errors[name].θˡ))
    q_error = median(filter(isfinite, errors[name].qᵗ))
    println(rpad(configuration_labels[name], 28), " median RMSE: θˡ ", round(θ_error, digits = 2), " K,  qᵗ ", round(q_error, digits = 2), " g kg⁻¹")
end
