module YinYangConservation

using KernelAbstractions: @kernel, @index

using Oceananigans: Center, TracerTendencyCallsite, fields, prognostic_fields
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: architecture
using Oceananigans.MultiRegion: MultiRegion, YinYangConservationFixer, YinYangGridOfSomeKind, owned_weights
using Oceananigans.Operators: Vᶜᶜᶜ, div_xyᶜᶜᶜ
using Oceananigans.Simulations: Simulations, Simulation, Callback, add_callback!
using Oceananigans.Utils: MultiRegionObject, launch!, @apply_regionally

using Breeze.AtmosphereModels: AtmosphereModel, AtmosphereModelBuoyancy, closure_scalar_names, closure_scalar_index,
                               dynamics_density, total_density, thermodynamic_density, thermodynamic_density_name,
                               specific_scalar_fields, transport_velocities, thermodynamic_transport_velocities,
                               field_advection_scheme
using Breeze.Advection: horizontal_div_ρUc
using Breeze.CompressibleEquations: horizontal_theta_flux_divergence
using Breeze.TimeSteppers: CompressibleAcousticModel
using Breeze.TurbulenceClosures: horizontal_∇_dot_Jᶜ

"""
    YinYangConservationFixer(model::AtmosphereModel)

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

julia> YinYangConservationFixer(model)
YinYangConservationFixer of ρᵈ, ρθ, ρqᵛ and c
```
"""
MultiRegion.YinYangConservationFixer(model::AtmosphereModel) = MassFixer(model)

struct MassFixer{D, F, P, W, B, R, S, A, I}
    densities :: D            # name => the density ρc of each prognostic scalar at the callsite, the tracers as total density × mass fraction
    divergences :: F          # name => ∇ʰ·F, the horizontal flux divergence of the tendency of ρc
    positive_parts :: P       # name => max(0, ρc)
    weights :: W              # the owned weights w of the cells
    budget :: B               # (column, quantity, density, panel): Σₖ w V over the column of the four quantities below
    panels :: R               # the panels of `budget`
    sums :: S                 # (1, quantity, density, 1), the sums of `budget` over its columns and panels
    acoustic_divergences :: A # (; entry, mean) flux divergences of the acoustic loop (see `acoustic_divergences`), `nothing` without one
    corrected_iteration :: I  # the iteration whose third stage the acoustic correction last completed
end

# The quantities of the budget
const seam_budget = 1      # ∇ʰ·F of the tendency, whose sum is the seam budget E
const positive_content = 2 # max(0, ρc), whose sum is C⁺
const acoustic_entry = 3   # the flux divergence of the acoustic loop at the momentum its stage starts from
const acoustic_mean = 4    # the flux divergence of the acoustic loop at the momentum averaged over its stage

Base.summary(fixer::MassFixer) = string("YinYangConservationFixer of ", join(keys(fixer.densities), ", ", " and "))
Base.show(io::IO, fixer::MassFixer) = print(io, summary(fixer))

# The fixed name makes a second call replace the first fixer, so that the budget is returned once
Simulations.add_conservation_fixer!(simulation::Simulation{<:AtmosphereModel}, ::YinYangGridOfSomeKind) =
    add_callback!(simulation, Callback(YinYangConservationFixer(simulation.model); callsite=TracerTendencyCallsite());
                  name=:conservation_fixer)

function MassFixer(model::AtmosphereModel)
    grid = model.grid
    arch = architecture(grid)
    names = (:ρᵈ, closure_scalar_names(model)...)
    densities = NamedTuple{names}(map(name -> density(model, name), names))
    divergences = NamedTuple{names}(map(name -> flux_divergence(model, name), names))
    positive_parts = map(ρc -> KernelFunctionOperation{Center, Center, Center}(positive_part, grid, ρc), densities)
    Nx, Ny, _ = size(grid)
    budget = zeros(arch, eltype(grid), Nx * Ny, 4, length(names), 2)
    panels = MultiRegionObject(Tuple(view(budget, :, :, :, r) for r in 1:2))
    sums = zeros(arch, eltype(grid), 1, 4, length(names), 1)
    return MassFixer(densities, divergences, positive_parts, owned_weights(grid), budget, panels, sums,
                     acoustic_divergences(model), Ref(0))
end

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
    horizontal_∇_dot_Jᶜ(i, j, k, grid, ρ, closure, closure_fields, id, c, clock, model_fields, buoyancy)

# The acoustic loop moves the dry density with the momentum perturbation (ρu)′ of its substeps and the
# thermodynamic density with θᴸ (ρu)′, where θᴸ is θ at the entry of the stage. Its transport velocity
# ⟨u⟩ = (ρuᴸ + ⟨(ρu)′⟩) / ℑ(ρᴸ) averages the perturbation over the stage, so the seam budget of the
# loop's fluxes is that of the flux of the mean momentum ⟨u⟩ ℑ(ρᴸ), with ρᴸ = ρᵈ − ρ′ the density the
# stage started from, less that of the flux of the momentum ρuᴸ at the entry of the stage.
acoustic_divergences(model) = nothing

function acoustic_divergences(model::CompressibleAcousticModel)
    substepper = model.timestepper.substepper
    ρᵈ = dynamics_density(model.dynamics)
    ρθ = thermodynamic_density(model.formulation)
    ρᴸ = ρᵈ - substepper.density_perturbation
    mean_ρu = substepper.time_averaged_velocities.u * ρᴸ
    mean_ρv = substepper.time_averaged_velocities.v * ρᴸ
    entry = acoustic_flux_divergences(model.grid, ρθ / ρᵈ, model.momentum.ρu, model.momentum.ρv)
    mean = acoustic_flux_divergences(model.grid, substepper.linearization_potential_temperature, mean_ρu, mean_ρv)
    return (; entry, mean)
end

acoustic_flux_divergences(grid, θ, ρu, ρv) =
    (KernelFunctionOperation{Center, Center, Center}(div_xyᶜᶜᶜ, grid, ρu, ρv),
     KernelFunctionOperation{Center, Center, Center}(horizontal_theta_flux_divergence, grid, θ, ρu, ρv))

#####
##### The budget and the corrections
#####

# Σₖ w V f over the column (i, j), into row i + (j - 1) Nx of quantity q of density d
@kernel function _owned_column_integral!(budget, q, d, grid, weights, f)
    i, j = @index(Global, NTuple)
    n = i + (j - 1) * size(grid, 1)
    @inbounds w = weights[i, j, 1]
    integral = zero(w)
    for k in 1:size(grid, 3)
        @inbounds integral += w * Vᶜᶜᶜ(i, j, k, grid) * f[i, j, k]
    end
    @inbounds budget[n, q, d] = integral
end

integrate_columns!(panels, q, d, grid, weights, f) =
    launch!(architecture(grid), grid, :xy, _owned_column_integral!, panels, q, d, grid, weights, f)

# The sums are on the device, read at [1, q, d, 1] on every point. A density without a positive part is left alone.
@inline removal_rate(sums, q, d) = @inbounds ifelse(sums[1, positive_content, d, 1] > 0,
                                                    sums[1, q, d, 1] / sums[1, positive_content, d, 1],
                                                    zero(eltype(sums)))

# The tendency of ρc gains back the seam budget E, in proportion to the positive part of ρc
@kernel function _correct_tendency!(G, ρc, sums, d)
    i, j, k = @index(Global, NTuple)
    r = removal_rate(sums, seam_budget, d)
    @inbounds G[i, j, k] += r * max(0, ρc[i, j, k])
end

# The state after the third stage gains back the mass Δt E that the acoustic loop moved through the seam, where
# E = mean − entry is the budget of the flux of the mean momentum perturbation
@kernel function _correct_density!(ρc, sums, d, Δt)
    i, j, k = @index(Global, NTuple)
    a = Δt * (removal_rate(sums, acoustic_mean, d) - removal_rate(sums, acoustic_entry, d))
    @inbounds ρc[i, j, k] += a * max(0, ρc[i, j, k])
end

# At the `TracerTendencyCallsite` of every stage, the tendencies about to be integrated have just been computed
# from the current state, so the fixer evaluates their fluxes from the same state and corrects them. The state
# of the acoustic loop is corrected afterwards, from the budget of the uncorrected state, and the entry
# divergences of its third stage are recorded last, from the state that stage starts from.
function (fixer::MassFixer)(model)
    grid = model.grid
    arch = architecture(grid)
    for (d, name) in enumerate(keys(fixer.densities))
        @apply_regionally integrate_columns!(fixer.panels, seam_budget, d, grid, fixer.weights, fixer.divergences[name])
        @apply_regionally integrate_columns!(fixer.panels, positive_content, d, grid, fixer.weights, fixer.positive_parts[name])
    end
    prepare_acoustic_correction!(fixer, model, fixer.acoustic_divergences)
    sum!(fixer.sums, fixer.budget)

    Gⁿ = model.timestepper.Gⁿ
    for (d, name) in enumerate(keys(fixer.densities))
        @apply_regionally launch!(arch, grid, :xyz, _correct_tendency!, Gⁿ[name], fixer.densities[name], fixer.sums, d)
    end

    correct_acoustic_state!(fixer, model, fixer.acoustic_divergences)
    return nothing
end

prepare_acoustic_correction!(fixer, model, ::Nothing) = nothing
correct_acoustic_state!(fixer, model, ::Nothing) = nothing

# Every stage of `AcousticRungeKutta3` restarts from Uⁿ, so only the third's state is kept and corrected, once
correcting(fixer, clock) = clock.stage == 1 && fixer.corrected_iteration[] != clock.iteration

# The dry density and the thermodynamic density, the densities the acoustic loop moves
acoustic_density_indices(fixer, model) =
    (1, findfirst(==(thermodynamic_density_name(model.formulation)), keys(fixer.densities)))

function prepare_acoustic_correction!(fixer, model, divergences)
    correcting(fixer, model.clock) || return nothing
    for (d, f) in zip(acoustic_density_indices(fixer, model), divergences.mean)
        @apply_regionally integrate_columns!(fixer.panels, acoustic_mean, d, model.grid, fixer.weights, f)
    end
    return nothing
end

function correct_acoustic_state!(fixer, model, divergences)
    grid = model.grid
    arch = architecture(grid)
    clock = model.clock
    indices = acoustic_density_indices(fixer, model)

    if correcting(fixer, clock)
        fixer.corrected_iteration[] = clock.iteration
        Δt = convert(eltype(grid), clock.last_Δt)
        for d in indices
            @apply_regionally launch!(arch, grid, :xyz, _correct_density!, fixer.densities[d], fixer.sums, d, Δt)
        end
    end

    if clock.stage == 3
        for (d, f) in zip(indices, divergences.entry)
            @apply_regionally integrate_columns!(fixer.panels, acoustic_entry, d, grid, fixer.weights, f)
        end
    end

    return nothing
end

end # module
