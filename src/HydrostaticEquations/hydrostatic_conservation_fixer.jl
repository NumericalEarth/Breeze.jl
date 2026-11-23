#####
##### Conservation of the mass and of the scalar contents of HydrostaticDynamics on a Yin-Yang grid
#####
##### Each panel of a Yin-Yang grid steps all of its cells with its own fluxes, so the panels disagree
##### on the flux through the curve that divides the sphere between them. With w the fraction of a
##### cell that its panel owns, the owned content ∫ w ρc dV of a density changes at the rate -E, with
##### E = ∫ w ∇ʰ·F dV the seam budget of its horizontal flux F: the fluxes through the faces between
##### two owned cells cancel, and the vertical fluxes vanish at the ground and at the model top.
#####
##### The fixer returns E over the whole sphere. It is a global correction: it conserves the owned
##### integrals and nothing else, and it is off unless it is added to a simulation.
#####
#####   * Mass. The surface pressure gains r (pˢ - pᵗ), in proportion to the mass of the column,
#####     with r = Eᵐ / M the seam budget of the layer mass over the mass of the atmosphere.
#####
#####   * Scalars. The mass that a layer gains, ∂ρ/∂pˢ r (pˢ - pᵗ), carries the specific value c of
#####     the layer, and what is left of the seam budget of ρc is returned in proportion to the
#####     part of ρc that has the sign of most of its content:
#####
#####         δ(ρc) = c ∂ρ/∂pˢ r (pˢ - pᵗ) + rᶜ max(0, s ρc) ,    rᶜ = (Eᶜ - r Iᶜ) / Cˢ ,
#####
#####     with Iᶜ = ∫ w c ∂ρ/∂pˢ (pˢ - pᵗ) dV, C⁺ = ∫ w max(0, ρc) dV, C⁻ = ∫ w max(0, -ρc) dV,
#####     s = +1 and Cˢ = C⁺ if C⁺ ≥ C⁻, and s = -1 and Cˢ = C⁻ otherwise. A uniform c has
#####     Eᶜ = c Eᵐ and Iᶜ = c M, so rᶜ = 0 and it stays uniform on every layer of the hybrid
#####     coordinate.
#####
#####     The weight is defined for a scalar of either sign. A non-negative scalar is scaled by
#####     1 + rᶜ, which keeps it non-negative and its zeros zero, and its undershoots, if a
#####     transport scheme left any, are not amplified; a non-positive scalar is scaled by
#####     1 - rᶜ, likewise. Of a scalar that takes both signs only the part of the dominant sign
#####     is scaled, and no value changes sign while |rᶜ| < 1. Only a scalar that vanishes in
#####     every owned cell has no weight (C⁺ = C⁻ = 0): it has no content to restore, and it is
#####     left as it is. The choice of the part switches where C⁺ = C⁻; either choice conserves
#####     the integral. The seam budget of the potential temperature includes that of the
#####     hyperdiffusion, when there is one.
#####
##### The same correction serves the tendencies of the full state (kg s⁻¹) and the increment of a
##### stage of a split-explicit time stepper, whose substeps move mass with the velocity
##### perturbation (kg): only the fields that the integrals are taken over differ.
#####

struct HydrostaticConservationFixer{T, S}
    tendency_budgets :: T # seam budgets of the tendencies of the full state (see `seam_budgets`)
    stage_budgets :: S    # seam budgets of the substeps of a stage of a split-explicit time stepper, or nothing
end

Base.summary(fixer::HydrostaticConservationFixer) =
    string("HydrostaticConservationFixer of pˢ, ", join(keys(fixer.tendency_budgets.scalars), ", ", " and "))
Base.summary(fixer::HydrostaticConservationFixer{Nothing}) = "HydrostaticConservationFixer of the convex scalar stages"

Base.show(io::IO, fixer::HydrostaticConservationFixer) = print(io, summary(fixer))

# Oceananigans' `add_conservation_fixer!` builds the fixer of a Yin-Yang grid through `YinYangConservationFixer`
MultiRegion.YinYangConservationFixer(model::HydrostaticModel) = HydrostaticConservationFixer(model)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The callback that conserves the dry mass and the content of every prognostic scalar of `model`,
an `AtmosphereModel` with [`HydrostaticDynamics`](@ref) on a `YinYangGrid`, which
`add_conservation_fixer!` adds to a simulation at the `TracerTendencyCallsite`.

The panels of a Yin-Yang grid disagree on the fluxes through the curve that divides the sphere
between them, so the integrals over the sphere of the mass, of the potential temperature density
and of the tracers are not conserved. The fixer returns what the seam takes to the tendency of the
surface pressure of every column, in proportion to the mass of the column, and to the tendency of
every scalar in proportion to the part of it that has the sign of most of its content, so that
a uniform mixing ratio stays uniform and a scalar of either sign, or of both, keeps the sign of
every value. With
`:GravityWaveMultirateInfinitesimalStep` it also returns the mass and the scalar contents that
the gravity-wave substeps of each stage move through the seam.

The correction is global: the integrals over the sphere are conserved to rounding error, and the
fluxes of the two panels across the seam are left as they are. Without the fixer, which is the
default, the integrals drift. The velocities are not corrected.

```jldoctest
julia> using Breeze, Oceananigans

julia> using Oceananigans.MultiRegion: YinYangGrid

julia> grid = YinYangGrid(size=(48, 16, 4), z=(0, 1), halo=(3, 3, 3));

julia> model = AtmosphereModel(grid; dynamics=HydrostaticDynamics(), momentum_advection=VectorInvariant(), tracers=:c);

julia> Breeze.HydrostaticEquations.HydrostaticConservationFixer(model)
HydrostaticConservationFixer of pˢ, ρθ, ρqᵛ and c
```
"""
function HydrostaticConservationFixer(model::HydrostaticModel)
    dynamics = model.dynamics
    specific = specific_prognostic_scalar_fields(model)
    flux_divergences = NamedTuple{keys(specific)}(map(name -> total_flux_divergence(model, name), keys(specific)))

    tendency_budgets = seam_budgets(model, dynamics.density, dynamics.surface_pressure, model.velocities,
                                    specific, flux_divergences)

    return HydrostaticConservationFixer(tendency_budgets, stage_seam_budgets(model.timestepper, model))
end

# The horizontal flux divergence of a scalar in the tendencies: that of advection, and for the
# potential temperature also that of the hyperdiffusion, whose fluxes the panels disagree on at
# the seam like any others
function total_flux_divergence(model, name)
    advective = flux_divergence(model, name)
    hyperdiffusion = model.dynamics.hyperdiffusion
    name == thermodynamic_density_name(model.formulation) || return advective
    return total_flux_divergence(advective, hyperdiffusion, model)
end

total_flux_divergence(advective, ::Nothing, model) = advective

total_flux_divergence(advective, hyperdiffusion, model) =
    advective + KernelFunctionOperation{Center, Center, Center}(hyperdiffusion_flux_divergence, model.grid, hyperdiffusion,
                                                                model.dynamics.density,
                                                                model.formulation.potential_temperature)

# The seam budgets of the substeps of a time stepper that has them
stage_seam_budgets(timestepper, model) = nothing

#####
##### Seam budgets
#####

# Horizontal divergence of the mass flux of layer `k` per unit volume of the grid
@inline layer_mass_flux_divergence(i, j, k, grid, ρ, u, v) =
    V⁻¹ᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, mass_flux_x, ρ, u) + δyᵃᶜᵃ(i, j, k, grid, mass_flux_y, ρ, v))

# The density of a scalar that the mass (pˢ - pᵗ) / g of its column, spread over the layers as the
# surface pressure spreads it, would carry: c ∂ρ/∂pˢ (pˢ - pᵗ)
@inline mass_coupled_density(i, j, k, grid, c, pˢ, B, pᵗ, g) =
    @inbounds c[i, j, k] * layer_mass_derivative(i, j, k, grid, B, g) * (pˢ[i, j, 1] - pᵗ)

# The parts of the density of a scalar of either sign, and the weight of its correction: the part
# whose sign is `s` (+1 or -1)
@inline positive_density(i, j, k, grid, ρ, c) = @inbounds max(0, ρ[i, j, k] * c[i, j, k])
@inline negative_density(i, j, k, grid, ρ, c) = @inbounds max(0, - ρ[i, j, k] * c[i, j, k])
@inline signed_density(i, j, k, grid, ρ, c, s) = @inbounds max(0, s * (ρ[i, j, k] * c[i, j, k]))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the integrals over the owned cells of a Yin-Yang grid from which the seam budget of the
layer mass `ρ` of the surface pressure `pˢ`, carried by the horizontal velocities `u` and `v` of
`velocities`, is returned, with the budgets of the scalars whose `specific` values and horizontal
`flux_divergences` are given by name:

- `mass.budget`: ``Eᵐ = ∫ w ∇ʰ·(ρ 𝐮) dV`` and `mass.content`: ``M = ∫ w ρ dV``;
- for every scalar, `budget`: ``Eᶜ = ∫ w ∇ʰ·F dV``, `coupled`: ``Iᶜ = ∫ w c ∂ρ/∂pˢ (pˢ - pᵗ) dV``
  `positive`: ``C⁺ = ∫ w \\max(0, ρ c) dV`` and `negative`: ``C⁻ = ∫ w \\max(0, -ρ c) dV``.
"""
function seam_budgets(model, ρ, pˢ, velocities, specific, flux_divergences)
    grid = model.grid
    dynamics = model.dynamics
    B = dynamics.surface_pressure_coefficient
    pᵗ = dynamics.top_pressure
    g = model.thermodynamic_constants.gravitational_acceleration
    center_operation(f, args...) = KernelFunctionOperation{Center, Center, Center}(f, grid, args...)

    mass = (budget = owned_integral(center_operation(layer_mass_flux_divergence, ρ, velocities.u, velocities.v)),
            content = owned_integral(ρ))

    scalars = map(specific, flux_divergences) do c, flux_divergence
        (budget = owned_integral(flux_divergence),
         coupled = owned_integral(center_operation(mass_coupled_density, c, pˢ, B, pᵗ, g)),
         positive = owned_integral(center_operation(positive_density, ρ, c)),
         negative = owned_integral(center_operation(negative_density, ρ, c)))
    end

    return (; ρ, pˢ, specific, mass, scalars)
end

#####
##### The correction
#####

@kernel function _return_mass_budget!(Gpˢ, pˢ, pᵗ, r)
    i, j = @index(Global, NTuple)
    @inbounds Gpˢ[i, j, 1] += r * (pˢ[i, j, 1] - pᵗ)
end

@kernel function _return_scalar_budget!(Gρc, grid, c, ρ, pˢ, B, pᵗ, g, r, rᶜ, s)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρc[i, j, k] += r * mass_coupled_density(i, j, k, grid, c, pˢ, B, pᵗ, g) +
                              rᶜ * signed_density(i, j, k, grid, ρ, c, s)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Add to the surface pressure and to the scalars of every `NamedTuple` of `targets` the seam
budgets of `budgets` (see [`seam_budgets`](@ref)): to tendencies when the budgets are those of
the fluxes, and to increments when they are those of the fluxes integrated over a stage.
"""
function return_seam_budgets!(targets, budgets, model)
    grid = model.grid
    arch = grid.architecture
    FT = eltype(grid)
    dynamics = model.dynamics
    B = dynamics.surface_pressure_coefficient
    pᵗ = dynamics.top_pressure
    g = model.thermodynamic_constants.gravitational_acceleration
    (; ρ, pˢ, specific, mass, scalars) = budgets

    foreach(compute!, mass)
    foreach(integrals -> foreach(compute!, integrals), scalars)

    r = owned_integral_value(mass.budget) / owned_integral_value(mass.content)

    for target in targets
        @apply_regionally launch!(arch, grid, :xy, _return_mass_budget!, target.pˢ, pˢ, pᵗ, convert(FT, r))
    end

    for name in keys(scalars)
        (; budget, coupled, positive, negative) = scalars[name]
        C⁺ = owned_integral_value(positive)
        C⁻ = owned_integral_value(negative)
        s = ifelse(C⁺ ≥ C⁻, 1, -1)
        Cˢ = max(C⁺, C⁻)
        E = owned_integral_value(budget) - r * owned_integral_value(coupled)
        rᶜ = ifelse(Cˢ > 0, E / Cˢ, zero(Cˢ))

        for target in targets
            @apply_regionally launch!(arch, grid, :xyz, _return_scalar_budget!, target[name], grid,
                                      specific[name], ρ, pˢ, B, pᵗ, g, convert(FT, r), convert(FT, rᶜ), s)
        end
    end

    return nothing
end

# At the `TracerTendencyCallsite` of every stage the tendencies about to be integrated have just
# been computed from the current state, and the tracers are specific
function (fixer::HydrostaticConservationFixer)(model)
    return_seam_budgets!(tuple(model.timestepper.Gⁿ), fixer.tendency_budgets, model)
    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return to the fields of every `NamedTuple` of `targets` what the substeps of the stage that a
split-explicit time stepper has just completed moved through the seam of a Yin-Yang grid. Called
by the time stepper for the function of each of its tendency callbacks: only a
[`HydrostaticConservationFixer`](@ref) acts.
"""
correct_completed_stage!(callback_function, model, targets) = nothing

function correct_completed_stage!(fixer::HydrostaticConservationFixer, model, targets)
    return_seam_budgets!(targets, fixer.stage_budgets, model)
    return nothing
end

#####
##### Seam budgets of the gravity-wave substeps
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the [`seam_budgets`](@ref) of a stage of the gravity-wave substeps of `substepper`. The
substeps move the layer mass, the potential temperature and the other scalars of the state at
the start of the time step, whose surface pressure is `pˢ⁰`, with the mass fluxes of the
velocity perturbation, which are linear in it. Their budget over a stage is therefore that of
the flux of the integrated velocity perturbation, once the time stepper has taken from it the
part that the full tendencies carry.
"""
function gravity_wave_seam_budgets(substepper::GravityWaveSubstepper, pˢ⁰, model)
    grid = model.grid
    ρᴸ = substepper.linearization.ρ
    (; u, v) = substepper.integrated_velocities
    θ_name = thermodynamic_density_name(model.formulation)
    specific = merge(NamedTuple{(θ_name,)}((substepper.linearization.θ,)), substepper.specific_scalars)

    flux_divergences = map(specific) do cᴸ
        KernelFunctionOperation{Center, Center, Center}(horizontal_div_ρUc, grid, substepper.advection, ρᴸ, (; u, v), cᴸ)
    end

    return seam_budgets(model, ρᴸ, pˢ⁰, (; u, v), specific, flux_divergences)
end

#####
##### Seam budgets of the convex scalar stages
#####
##### A convex stage carries the layer mass and every scalar of `specific_scalars` with one mass flux, that of
##### the stage velocities 𝐮̂ (`compute_stage_transport_velocities!`), from the state the stage starts from:
#####
#####     Mⁱ = (1 - w) M⁰ + w M^{i-1} - Δtˢ ∇·F ,
#####
##### so what the seam takes from the owned content over the stage is Δtˢ E, with E = ∫ w ∇ʰ·F dV evaluated with
##### the operators of the stage itself (the layer-mass flux of 𝐮̂, and `div_ρUc` with the advection of the
##### scalar and its mixing ratio at the start of the stage). These budgets must be taken before the stage
##### changes the scalars, so they are recorded when the stage velocities are ready (`record_stage_seam_budgets!`)
##### and returned once the stage is complete. Returning Δtˢ E at every stage restores the owned content to
##### the convex combination of the owned contents, so that the integrals over the sphere are conserved.
#####
##### On a pressure-level grid the layer density is diagnosed from the surface pressure by `update_state!`,
##### which carries every scalar density at fixed mixing ratio: the mass returned through pˢ brings the scalar
##### mass c ΔB δpˢ / g with it, and only the rest, Δtˢ Eᶜ - r Iᶜ, is returned to the scalar itself, in
##### proportion to the part of it with the sign of most of its content. The potential temperature is not a
##### convex-stage scalar (the substeps advance it) and is not corrected here.

mutable struct ConvexStageSeamBudgets{I, R}
    integrals :: I               # owned integrals, built once: the stage budgets (kg s⁻¹) and the contents of the state
    recorded :: R                # Δtˢ Eᵐ and Δtˢ Eᶜ of the stage being completed (kg), with its weight
    last_correction :: R         # what the last stage returned: r, and rᶜ per scalar (for diagnostics)
end

# The content of a layer per unit volume of the grid, (Δp / g) / (Az Δz) times its own area: owned_integral of it is
# the owned mass of the layers at the current surface pressure (exact, whatever density the grid still holds)
@inline layer_mass_density(i, j, k, grid) = layer_mass(i, j, k, grid) / Δzᶜᶜᶜ(i, j, k, grid)
@inline signed_scalar_content(i, j, k, grid, r, s) = @inbounds max(0, s * r[i, j, k]) * layer_mass_density(i, j, k, grid)
@inline scalar_mass_coupled(i, j, k, grid, r, B, pˢ, pᵗ, g) =
    @inbounds r[i, j, k] * (B[i, j, k] - B[i, j, k+1]) * (pˢ[i, j, 1] - pᵗ) / (g * Δzᶜᶜᶜ(i, j, k, grid))

function convex_stage_seam_budgets(timestepper, model)
    grid = model.grid
    substepper = timestepper.substepper
    ρ = model.dynamics.density
    ∫𝐮′ = substepper.integrated_velocities
    𝐮̂ = coordinate_transport_velocities(grid, ∫𝐮′.u, ∫𝐮′.v, ∫𝐮′.w, ∫𝐮′.ρw)
    B = model.dynamics.surface_pressure_coefficient
    pˢ = model.dynamics.surface_pressure
    pᵗ = model.dynamics.top_pressure
    g = model.thermodynamic_constants.gravitational_acceleration
    prognostic = prognostic_fields(model)
    names = keys(substepper.specific_scalars)
    operation(f, args...) = KernelFunctionOperation{Center, Center, Center}(f, grid, args...)

    mass = (budget = owned_integral(operation(layer_mass_flux_divergence, ρ, ∫𝐮′.u, ∫𝐮′.v)),
            content = owned_integral(operation(layer_mass_density)))

    scalars = NamedTuple{names}(map(names) do name
        r = LayerMassFraction(prognostic[name], ρ)
        (budget = owned_integral(operation(AtmosphereModels.div_ρUc, model.advection[name], ρ, 𝐮̂, r)),
         coupled = owned_integral(operation(scalar_mass_coupled, r, B, pˢ, pᵗ, g)),
         positive = owned_integral(operation(signed_scalar_content, r, 1)),
         negative = owned_integral(operation(signed_scalar_content, r, -1)))
    end)

    zeros_of(names) = NamedTuple{names}(map(_ -> 0.0, names))
    recorded = (mass = 0.0, scalars = zeros_of(names))
    return ConvexStageSeamBudgets((; mass, scalars), recorded, recorded)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Record the seam budgets of a convex scalar stage of length `Δtˢ`: called by the time stepper once the stage
velocities are ready and before the stage changes the scalars. Only a [`HydrostaticConservationFixer`](@ref)
whose stage budgets are those of the convex scalar stages acts.
"""
record_stage_seam_budgets!(callback_function, model, Δtˢ) = nothing
record_stage_seam_budgets!(fixer::HydrostaticConservationFixer, model, Δtˢ) =
    record_stage_seam_budgets!(fixer.stage_budgets, model, Δtˢ)

function record_stage_seam_budgets!(budgets::ConvexStageSeamBudgets, model, Δtˢ)
    (; mass, scalars) = budgets.integrals
    compute!(mass.budget)
    for integrals in scalars
        compute!(integrals.budget)
    end
    budgets.recorded = (mass = Δtˢ * owned_integral_value(mass.budget),
                        scalars = map(integrals -> Δtˢ * owned_integral_value(integrals.budget), scalars))
    return nothing
end

@kernel function _return_convex_scalar_budget!(ρc, ρ, rᶜ, s)
    i, j, k = @index(Global, NTuple)
    @inbounds ρc[i, j, k] += rᶜ * max(0, s * ρc[i, j, k])
end

function return_convex_stage_seam_budgets!(targets, budgets::ConvexStageSeamBudgets, model)
    grid = model.grid
    arch = grid.architecture
    FT = eltype(grid)
    dynamics = model.dynamics
    pˢ = dynamics.surface_pressure
    pᵗ = dynamics.top_pressure
    (; mass, scalars) = budgets.integrals
    recorded = budgets.recorded
    prognostic = prognostic_fields(model)

    compute!(mass.content)
    r = recorded.mass / owned_integral_value(mass.content)

    for target in targets
        @apply_regionally launch!(arch, grid, :xy, _return_mass_budget!, target.pˢ, pˢ, pᵗ, convert(FT, r))
    end

    rᶜs = map(keys(scalars)) do name
        integrals = scalars[name]
        foreach(compute!, (integrals.coupled, integrals.positive, integrals.negative))
        C⁺ = owned_integral_value(integrals.positive)
        C⁻ = owned_integral_value(integrals.negative)
        s = ifelse(C⁺ ≥ C⁻, 1, -1)
        Cˢ = max(C⁺, C⁻)
        E = recorded.scalars[name] - r * owned_integral_value(integrals.coupled)
        rᶜ = ifelse(Cˢ > 0, E / Cˢ, zero(Cˢ))
        @apply_regionally launch!(arch, grid, :xyz, _return_convex_scalar_budget!, prognostic[name],
                                  dynamics.density, convert(FT, rᶜ), s)
        rᶜ
    end

    budgets.last_correction = (mass = r, scalars = NamedTuple{keys(scalars)}(rᶜs))
    return nothing
end

correct_completed_stage!(fixer::HydrostaticConservationFixer{<:Any, <:ConvexStageSeamBudgets}, model, targets) =
    return_convex_stage_seam_budgets!(targets, fixer.stage_budgets, model)

# With convex scalar stages the mass and the convex-stage scalars are corrected stage by stage, with the
# budgets of the stage mass flux; the tendencies carry no transport of those scalars to correct
(fixer::HydrostaticConservationFixer{<:Any, <:ConvexStageSeamBudgets})(model) = nothing
