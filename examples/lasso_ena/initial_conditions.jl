#####
##### Sounding → reference state and initial condition.
#####
##### The SAM `snd` file gives potential temperature (interpreted, as in SAM, as the
##### liquid-water potential temperature of the initial state, which has no condensate
##### until the first saturation adjustment), total water, and winds. Following
##### `setdata.f90`, levels are converted to heights hydrostatically and interpolated
##### linearly to the grid; the ENA cases have no levels above the sounding top so no
##### standard-atmosphere extension is needed.
#####

using Random: AbstractRNG, MersenneTwister
using Oceananigans.Grids: znodes, Center
using Breeze: ThermodynamicConstants, SaturationAdjustment, WarmPhaseEquilibrium
using Breeze.Thermodynamics: LiquidIcePotentialTemperatureState, MoistureMassFractions, temperature
using Breeze.Microphysics: adjust_thermodynamic_state

"""
    SoundingProfiles

Piecewise-linear interpolants (in height) of a `snd` record: `θ`, `qᵗ`, `u`, `v`, plus the
surface pressure. Levels below the surface are kept in the record but excluded from the
interpolation only when `exclude_subsurface_levels=true`. The fidelity default is `false`:
SAM's `setdata.f90` interpolates through the below-surface levels, so they bracket the
lowest LES cells.
"""
struct SoundingProfiles{FT}
    z :: Vector{FT}
    θ :: Vector{FT}
    qᵗ :: Vector{FT}          # the file's moisture (mixing ratio or mass fraction, see `convert_moisture`)
    u :: Vector{FT}
    v :: Vector{FT}
    surface_pressure :: FT
    convert_moisture :: Bool  # true: `qᵗ` is a dry mixing ratio converted after interpolation
end

"""
    SoundingProfiles(sounding; exclude_subsurface_levels=false, moisture_basis=:mixing_ratio)

`moisture_basis = :mixing_ratio` (SAM's convention: kg per kg dry air) interpolates the
file's `q` to the requested height, as `setdata.f90` does, and converts the result to a
Breeze mass fraction `q/(1 + q)`; `:mass_fraction` passes it through.
"""
function SoundingProfiles(sounding::SAMSounding; exclude_subsurface_levels=false, moisture_basis=:mixing_ratio)
    z = record_heights(sounding)
    keep = exclude_subsurface_levels ? findall(≥(0), z) : eachindex(z)
    moisture_basis in (:mixing_ratio, :mass_fraction) ||
        throw(ArgumentError("moisture_basis must be :mixing_ratio or :mass_fraction"))
    return SoundingProfiles(z[keep], sounding.θ[keep], sounding.q[keep],
                            sounding.u[keep], sounding.v[keep], sounding.surface_pressure,
                            moisture_basis === :mixing_ratio)
end

function (p::SoundingProfiles)(name::Symbol, z)
    value = interpolate_profile(p.z, getproperty(p, name), z)
    return name === :qᵗ && p.convert_moisture ? mass_fraction_from_mixing_ratio(value) : value
end

#####
##### Saturation partition of (θˡ, qᵗ, p) into (T, qᵛ, qᶜˡ) with Breeze's own thermodynamics
#####

"""
    saturation_partition(θˡ, qᵗ, p; constants=ThermodynamicConstants(Float64), standard_pressure=1e5)

Given the liquid-water potential temperature `θˡ`, total water `qᵗ` and pressure `p`,
return `(T, qᵛ, qᶜˡ)` in warm-phase saturation equilibrium, computed with Breeze's
`SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())` secant solver acting on a
`LiquidIcePotentialTemperatureState` — the same thermodynamics (Clausius-Clapeyron,
mixture heat capacity, Exner function) that the one-moment control run uses internally,
so the P3 and one-moment initial states are partitioned identically. This mirrors what
SAM's `micro_init`/`satadj_liquid` does before the first step; the two codes' saturation
formulae differ at the sub-percent level.
"""
function saturation_partition(θˡ, qᵗ, p; constants=ThermodynamicConstants(Float64), standard_pressure=1e5)
    FT = Float64
    q₀ = MoistureMassFractions(FT(qᵗ))
    𝒰₀ = LiquidIcePotentialTemperatureState(FT(θˡ), q₀, FT(standard_pressure), FT(p))
    adjustment = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    𝒰₁ = adjust_thermodynamic_state(𝒰₀, adjustment, constants)
    T = temperature(𝒰₁, constants)
    q₁ = 𝒰₁.moisture_mass_fractions
    return (T, q₁.vapor, q₁.liquid)
end

"""
    InitialPerturbation(; amplitude_T=0.1, amplitude_q=0.025e-3, depth=600, seed=1234, sam_perturb_type=5)

Uniform random perturbations of SAM's `setperturb.f90`. `sam_perturb_type = 5` (the
LASSO-ENA namelist choice) applies ±`amplitude_T` in temperature and ±`amplitude_q` in vapor
below `depth` (±0.1 K, ±0.025 g/kg below 600 m). `sam_perturb_type = 0` (SAM's default when a
namelist omits `perturb_type`) applies ±0.02 (6 − k) K in the five lowest levels and no
moisture perturbation; see [`perturbation_amplitudes`](@ref).
"""
Base.@kwdef struct InitialPerturbation
    amplitude_T :: Float64 = 0.1
    amplitude_q :: Float64 = 0.025e-3
    depth :: Float64 = 600
    seed :: Int = 1234
    sam_perturb_type :: Int = 5
end

"""
    perturbation_amplitudes(perturbation, z_centers)

Per-level amplitudes `(δT, δq)` [K, kg/kg] of the `setperturb.f90` case selected by
`perturbation.sam_perturb_type`: case 5 applies `amplitude_T`/`amplitude_q` wherever
`z ≤ depth`; case 0 applies `0.02 (6 - k)` K for `k ≤ 5` and no moisture perturbation.
"""
function perturbation_amplitudes(perturbation::InitialPerturbation, z_centers)
    n = length(z_centers)
    if perturbation.sam_perturb_type == 5
        below = [z ≤ perturbation.depth for z in z_centers]
        return (perturbation.amplitude_T .* below, perturbation.amplitude_q .* below)
    elseif perturbation.sam_perturb_type == 0
        return ([k ≤ 5 ? 0.02 * (6 - k) : 0.0 for k in 1:n], zeros(n))
    else
        throw(ArgumentError("sam_perturb_type must be 0 or 5 (the setperturb.f90 cases implemented), got $(perturbation.sam_perturb_type)"))
    end
end

"""
    initial_state_columns(profiles::SoundingProfiles, z_centers, reference_pressure)

Compute the column profiles (T, qᵛ, qᶜˡ, θˡ, qᵗ, u, v) at the cell centers `z_centers`
given the reference pressure at those heights.
"""
function initial_state_columns(profiles::SoundingProfiles, z_centers, reference_pressure;
                               constants=ThermodynamicConstants(Float64), standard_pressure=1e5)
    n = length(z_centers)
    T = zeros(n); qᵛ = zeros(n); qᶜˡ = zeros(n); θˡ = zeros(n); qᵗ = zeros(n); u = zeros(n); v = zeros(n)
    Tᵈ = zeros(n) # condensate-free temperature (all water as vapor), as SAM's HUJI-SBM micro_init
    for (k, z) in enumerate(z_centers)
        θˡ[k] = profiles(:θ, z)
        qᵗ[k] = profiles(:qᵗ, z)
        u[k] = profiles(:u, z)
        v[k] = profiles(:v, z)
        T[k], qᵛ[k], qᶜˡ[k] = saturation_partition(θˡ[k], qᵗ[k], reference_pressure[k]; constants)
        𝒰 = LiquidIcePotentialTemperatureState(θˡ[k], MoistureMassFractions(qᵗ[k]), standard_pressure, reference_pressure[k])
        Tᵈ[k] = temperature(𝒰, constants)
    end
    return (; z=collect(z_centers), T, qᵛ, qᶜˡ, θˡ, qᵗ, u, v, T_condensate_free=Tᵈ)
end

"""
    perturbation_array(Nx, Ny, z_centers, perturbation::InitialPerturbation)

One deterministic array of uniform random numbers in [-1, 1] for every cell of a level
with a nonzero amplitude (see [`perturbation_amplitudes`](@ref); zero elsewhere), drawn
from `MersenneTwister(perturbation.seed)` in a fixed (i, j, k) order on the host. The *same* array multiplies both the temperature and
the vapor perturbation, as in `setperturb.f90` (one `rrr` per cell), and it is copied to
the device by `set!`, so CPU and GPU runs start from identical states.
"""
function perturbation_array(Nx, Ny, z_centers, perturbation::InitialPerturbation)
    rng = MersenneTwister(perturbation.seed)
    δT, δq = perturbation_amplitudes(perturbation, z_centers)
    active = [(δT[k] != 0) | (δq[k] != 0) for k in eachindex(z_centers)]
    ϵ = zeros(Nx, Ny, length(z_centers))
    for k in eachindex(z_centers), j in 1:Ny, i in 1:Nx
        r = 2 * rand(rng) - 1     # one draw per cell in every level, as setperturb.f90
        ϵ[i, j, k] = active[k] ? r : 0.0
    end
    return ϵ
end

"""
    set_sounding_initial_state!(model, columns; perturbation, moisture_basis,
                                initialization=:condensate_free, frame_velocity=(0, 0),
                                initial_droplet_number=nothing)

Set the model's initial state from the sounding `columns` of [`initial_state_columns`](@ref)
with the `setperturb.f90` perturbation `perturbation` ([`perturbation_array`](@ref), the same
random number multiplying the temperature and vapor perturbations of a cell). Winds are
relative to `frame_velocity` (SAM's translating frame).

With `moisture_basis = :mixing_ratio` the vapor perturbation is applied to the dry mixing
ratio `r = q / (1 - q - qᶜ)`, as SAM perturbs its dry-basis vapor, and converted back.

For P3, `initialization` selects
- `:condensate_free` (SAM HUJI-SBM `micro_init`): no condensate, all water as vapor, cloud
  forms through the scheme's own activation and condensation in the first steps;
- `:equilibrium`: the warm-phase saturation partition (the one-moment control's first
  adjustment), with `initial_droplet_number` [m⁻³] in cloudy cells when the scheme carries
  an aerosol reservoir.
Other microphysics schemes are initialized from (θˡ, qᵗ).
"""
function set_sounding_initial_state!(model, columns; perturbation, moisture_basis,
                                     initialization = :condensate_free, frame_velocity = (0, 0),
                                     initial_droplet_number = nothing)
    initialization ∈ (:condensate_free, :equilibrium) ||
        throw(ArgumentError("initialization must be :condensate_free or :equilibrium, got $initialization"))
    FT = eltype(model.grid)
    Nx, Ny, Nz = size(model.grid)
    z = columns.z
    ϵ = perturbation_array(Nx, Ny, z, perturbation)
    column(values) = reshape(values, 1, 1, Nz)
    δT_levels, δq_levels = perturbation_amplitudes(perturbation, z)
    δT = column(δT_levels)
    δq = column(δq_levels)
    perturbed_moisture(q, ϵ, δq, qᶜ=0.0) = moisture_basis === :mixing_ratio ?
        (r = q / (1 - q - qᶜ) + δq * ϵ; max(0, r * (1 - qᶜ) / (1 + r))) : max(0, q + δq * ϵ)

    uᶠ, vᶠ = FT.(frame_velocity)
    u₀ = repeat(column(columns.u .- uᶠ), Nx, Ny, 1)
    v₀ = repeat(column(columns.v .- vᶠ), Nx, Ny, 1)

    if is_p3(model.microphysics) && initialization === :condensate_free
        T₀ = column(columns.T_condensate_free) .+ δT .* ϵ
        qᵛ₀ = perturbed_moisture.(column(columns.qᵗ), ϵ, δq)
        set!(model; T=T₀, qᵛ=qᵛ₀, u=u₀, v=v₀)
    elseif is_p3(model.microphysics)
        T₀ = column(columns.T) .+ δT .* ϵ
        qᵛ₀ = perturbed_moisture.(column(columns.qᵛ), ϵ, δq, column(columns.qᶜˡ))
        qᶜˡ₀ = repeat(column(columns.qᶜˡ), Nx, Ny, 1)
        if isnothing(model.microphysics.aerosol)
            set!(model; T=T₀, qᵛ=qᵛ₀, qᶜˡ=qᶜˡ₀, u=u₀, v=v₀)
        else
            isnothing(initial_droplet_number) &&
                throw(ArgumentError("initialization = :equilibrium with aerosol activation needs `initial_droplet_number` [m⁻³]"))
            ρᵣ = Array(interior(model.dynamics.reference_state.density, 1, 1, :))
            nᶜˡ₀ = repeat(column(initial_droplet_number ./ ρᵣ .* (columns.qᶜˡ .> 0)), Nx, Ny, 1)
            set!(model; T=T₀, qᵛ=qᵛ₀, qᶜˡ=qᶜˡ₀, nᶜˡ=nᶜˡ₀, u=u₀, v=v₀)
        end
    else
        Π = columns.T ./ columns.θˡ
        θ₀ = column(columns.θˡ) .+ δT .* ϵ ./ column(Π)
        qᵗ₀ = perturbed_moisture.(column(columns.qᵗ), ϵ, δq)
        set!(model; θ=θ₀, qᵗ=qᵗ₀, u=u₀, v=v₀)
    end
    return model
end
