#####
##### Multi-region grids: see the Multi-Region Grids section of .claude/rules/kernel-rules.md
#####

using Breeze.Utils: @regional
using Oceananigans: prognostic_fields
using Oceananigans.AbstractOperations: Integral
using Oceananigans.Fields: compute!
using Oceananigans.MultiRegion: MultiRegion, ImmersedMultiRegionGrid, MultiRegionGrids
using Oceananigans.Operators: div_xyᶜᶜᶜ
using Oceananigans.Utils: Utils, @apply_regionally, regions
using GPUArraysCore: @allowscalar

const MultiRegionAtmosphereModel = AtmosphereModel{<:Any, <:Any, <:Any, <:Any, <:MultiRegionGrids}

@inline Utils.isregional(::MultiRegionAtmosphereModel) = true
@inline Utils.regions(model::MultiRegionAtmosphereModel) = regions(model.grid)

validate_grid(::ImmersedMultiRegionGrid) =
    throw(ArgumentError("AtmosphereModel does not support an ImmersedBoundaryGrid built on a MultiRegionGrid."))

@regional AtmosphereModel
@regional AtmosphereModelBuoyancy
@regional SlowDynamics
@regional SedimentingCondensate
@regional OutflowEnabledVelocity
@regional DensityWeightedImplicitOperator

# A connected multi-region grid rotates the horizontal pair `(u, v)` across regions, so the pair is filled
# together. `Vararg{Any, N}` makes the compiler specialize on the forwarded boundary-condition arguments.
function fill_horizontal_velocity_halos!(𝐯::NamedTuple, args::Vararg{Any, N}; kwargs...) where N
    u, v, w = 𝐯
    fill_halo_regions!((u, v), args...; kwargs...)
    fill_halo_regions!(w, args...; kwargs...)
    return nothing
end

# Dynamics without prognostic momentum carry an empty momentum tuple
fill_horizontal_velocity_halos!(::NamedTuple{()}, args...; kwargs...) = nothing

function fill_prognostic_halo_regions!(model::AtmosphereModel, args::Vararg{Any, N}; kwargs...) where N
    momentum = model.momentum
    fill_horizontal_velocity_halos!(momentum, args...; kwargs...)
    scalars = Base.structdiff(prognostic_fields(model), momentum)
    fill_halo_regions!(scalars, args...; kwargs...)
    return nothing
end

# TODO: a multi-region field has no single parent array. Oceananigans should support `copyto!` between
# multi-region fields, halos included.
copy_parent!(dst, src) = @apply_regionally copyto!(parent(dst), parent(src))

#####
##### The conservation of mass on a Yin-Yang grid
#####

# Oceananigans' `add_conservation_fixer!` builds the fixer of a Yin-Yang grid through `YinYangConservationFixer`
MultiRegion.YinYangConservationFixer(model::AtmosphereModel) = ConservationFixer(model)

struct ConservationFixer{D, E, C, A, I}
    densities :: D            # name => the density ρc of each prognostic scalar at the callsite, the tracers as total density × mass fraction
    seam_budgets :: E         # name => ∫ ∇ʰ·F dV, the seam budget E of the horizontal fluxes F of the tendency of ρc
    positive_contents :: C    # name => ∫ max(0, ρc) dV, the content C⁺
    acoustic_budgets :: A     # (; entry, mean) seam budgets of the acoustic loop (see `acoustic_divergences`), `nothing` without one
    corrected_iteration :: I  # the iteration whose third stage the acoustic correction last completed
end

# Over the panels of a Yin-Yang grid, `Integral` weights every cell by the fraction of it that its panel owns
owned_integral(f) = Field(Integral(f))
owned_integral_value(∫f) = @allowscalar ∫f[1, 1, 1]

Base.summary(fixer::ConservationFixer) = string("ConservationFixer of ", join(keys(fixer.densities), ", ", " and "))
Base.show(io::IO, fixer::ConservationFixer) = print(io, summary(fixer))

"""
$(TYPEDSIGNATURES)

The callback that conserves the dry mass and the content of every other density of `model` on a
`YinYangGrid`, which `add_conservation_fixer!` adds to a simulation at the `TracerTendencyCallsite`.
Each panel steps all of its cells with its own fluxes, so the panels disagree on the flux through the
curve that divides the sphere between them, and the owned content of every density changes at the rate
`-E`, where `E = Σ w V ∇ʰ·F` is the seam budget of the horizontal fluxes `F` of its tendency. The fixer
returns `E` to the tendency of every density of every stage, in proportion to the positive part of the
density, which keeps it non-negative and keeps a uniform mixing ratio uniform. Momentum is not corrected.
The acoustic loop of `AcousticRungeKutta3` moves the dry density and the thermodynamic density with the
momentum perturbation of its substeps, whose seam budget is known once the stage is complete; after the
third stage, whose state is kept, the fixer returns that mass to the two densities.

```jldoctest
julia> using Breeze, Oceananigans

julia> using Oceananigans.MultiRegion: YinYangGrid

julia> grid = YinYangGrid(size=(48, 16, 4), z=(0, 10e3), halo=(3, 3, 3));

julia> model = AtmosphereModel(grid; dynamics=CompressibleDynamics(SplitExplicitTimeDiscretization()), tracers=:c);

julia> Breeze.AtmosphereModels.ConservationFixer(model)
ConservationFixer of ρᵈ, ρθ, ρqᵛ and c
```
"""
function ConservationFixer(model::AtmosphereModel)
    grid = model.grid
    names = (:ρᵈ, closure_scalar_names(model)...)
    densities = NamedTuple{names}(map(name -> density(model, name), names))
    seam_budgets = NamedTuple{names}(map(name -> owned_integral(flux_divergence(model, name)), names))
    positive_parts = map(ρc -> KernelFunctionOperation{Center, Center, Center}(positive_part, grid, ρc), densities)
    positive_contents = map(owned_integral, positive_parts)
    acoustic_budgets = acoustic_seam_budgets(acoustic_divergences(model))
    return ConservationFixer(densities, seam_budgets, positive_contents, acoustic_budgets, Ref(0))
end

acoustic_seam_budgets(::Nothing) = nothing
acoustic_seam_budgets(divergences) = map(fluxes -> map(owned_integral, fluxes), divergences)

@inline positive_part(i, j, k, grid, ρc) = @inbounds max(0, ρc[i, j, k])

# At the `TracerTendencyCallsite` the tracers are mass fractions, between the two conversions of `update_state!`
function density(model, name)
    ρc = prognostic_fields(model)[name]
    return name ∈ keys(model.tracers) ? total_density(model.dynamics) * ρc : ρc
end

#####
##### The flux divergences: Σ w V ∇ʰ·F over the owned cells is Σ F Δw over the faces where the owned weight w jumps,
##### because the fluxes through the faces between two owned cells cancel. The fluxes are those of the tendencies:
##### the dry density is carried by the momentum, the thermodynamic density by the dry density at its own
##### transport velocities, and every other scalar by the total density at the transport velocities of the model
##### (its microphysical velocity, if any, is vertical).
#####

function flux_divergence(model, name)
    grid = model.grid
    name === :ρᵈ && return KernelFunctionOperation{Center, Center, Center}(div_xyᶜᶜᶜ, grid, model.momentum.ρu, model.momentum.ρv)

    if name === thermodynamic_density_name(model.formulation)
        ρ = dynamics_density(model.dynamics)
        U = thermodynamic_transport_velocities(model)
    else
        ρ = total_density(model.dynamics)
        U = transport_velocities(model)
    end

    buoyancy = AtmosphereModelBuoyancy(model.dynamics, model.formulation, model.thermodynamic_constants,
                                       model.microphysics, model.microphysical_fields)
    args = (field_advection_scheme(model.advection, name), ρ, U, specific_scalar_fields(model)[name],
            model.closure, model.closure_fields, closure_scalar_index(model, name), model.clock, fields(model), buoyancy)
    return KernelFunctionOperation{Center, Center, Center}(horizontal_flux_divergence, grid, args...)
end

@inline horizontal_flux_divergence(i, j, k, grid, advection, ρ, U, c, closure, closure_fields, id, clock, model_fields, buoyancy) =
    horizontal_div_ρUc(i, j, k, grid, advection, ρ, U, c) +
    ∇ʰ_dot_Jᶜ(i, j, k, grid, ρ, closure, closure_fields, id, c, clock, model_fields, buoyancy)

# The flux divergences of the acoustic loop, for dynamics that have one
acoustic_divergences(model) = nothing

#####
##### The corrections
#####

# A density without a positive part is left alone
removal_rate(∫f, C⁺) = ifelse(C⁺ > 0, owned_integral_value(∫f) / C⁺, zero(C⁺))

# The tendency of ρc gains back the seam budget E, in proportion to the positive part of ρc
@kernel function _correct_tendency!(G, ρc, r)
    i, j, k = @index(Global, NTuple)
    @inbounds G[i, j, k] += r * max(0, ρc[i, j, k])
end

# The state after the third stage gains back the mass Δt E that the acoustic loop moved through the seam, where
# E = mean − entry is the budget of the flux of the mean momentum perturbation
@kernel function _correct_density!(ρc, a)
    i, j, k = @index(Global, NTuple)
    @inbounds ρc[i, j, k] += a * max(0, ρc[i, j, k])
end

# At the `TracerTendencyCallsite` of every stage, the tendencies about to be integrated have just been computed
# from the current state, so the fixer evaluates their fluxes from the same state and corrects them. The state
# of the acoustic loop is corrected afterwards, from the budget of the uncorrected state, and the entry
# divergences of its third stage are recorded last, from the state that stage starts from.
function (fixer::ConservationFixer)(model)
    grid = model.grid
    arch = architecture(grid)
    foreach(compute!, fixer.seam_budgets)
    foreach(compute!, fixer.positive_contents)
    contents = map(owned_integral_value, fixer.positive_contents)
    Gⁿ = model.timestepper.Gⁿ

    for name in keys(fixer.densities)
        r = removal_rate(fixer.seam_budgets[name], contents[name])
        @apply_regionally launch!(arch, grid, :xyz, _correct_tendency!, Gⁿ[name], fixer.densities[name], r)
    end

    correct_acoustic_state!(fixer, model, contents, fixer.acoustic_budgets)
    return nothing
end

correct_acoustic_state!(fixer, model, contents, ::Nothing) = nothing

# Every stage of `AcousticRungeKutta3` restarts from Uⁿ, so only the third's state is kept and corrected, once
correcting(fixer, clock) = clock.stage == 1 && fixer.corrected_iteration[] != clock.iteration

function correct_acoustic_state!(fixer, model, contents, budgets)
    grid = model.grid
    arch = architecture(grid)
    clock = model.clock

    # The dry density and the thermodynamic density, the densities the acoustic loop moves
    names = (:ρᵈ, thermodynamic_density_name(model.formulation))

    if correcting(fixer, clock)
        fixer.corrected_iteration[] = clock.iteration
        foreach(compute!, budgets.mean)
        Δt = convert(eltype(grid), clock.last_Δt)
        for (n, name) in enumerate(names)
            C⁺ = contents[name]
            a = Δt * (removal_rate(budgets.mean[n], C⁺) - removal_rate(budgets.entry[n], C⁺))
            @apply_regionally launch!(arch, grid, :xyz, _correct_density!, fixer.densities[name], a)
            # The rest of the step reads these densities across the region boundaries
            fill_halo_regions!(fixer.densities[name])
        end
    end

    clock.stage == 3 && foreach(compute!, budgets.entry)

    return nothing
end
