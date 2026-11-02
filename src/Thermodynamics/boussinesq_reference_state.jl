using Oceananigans.Fields: ConstantField

#####
##### Boussinesq reference state
#####

"""
    BoussinesqReferenceState

A reference state whose pressure, density, temperature, and moisture mass fractions are
uniform in space. Expanding about a uniform reference state yields the Boussinesq approximation,
which is accurate for domains much shallower than the atmospheric scale height (about 8 km),
such as direct numerical simulations of cloud turbulence or laboratory convection.

The reference density is fixed. If the reference frame moves vertically, the reference pressure
changes in time hydrostatically, and the reference temperature and moisture mass fractions follow
the domain mean.
"""
struct BoussinesqReferenceState{FT, SP, SD, ST, P, D, T, QV, QL, QI, F}
    base_pressure :: FT # pᵣ at the start of the simulation
    surface_pressure :: SP
    surface_density :: SD
    surface_temperature :: ST
    potential_temperature :: FT # θᵣ at the start of the simulation
    standard_pressure :: FT
    pressure :: P
    density :: D
    temperature :: T
    vapor_mass_fraction :: QV
    liquid_mass_fraction :: QL
    ice_mass_fraction :: QI
    frame :: F
end

"""
    MovingFrame

The state of a vertically moving reference frame of a [`BoussinesqReferenceState`](@ref).

The reference `pressure` evolves hydrostatically according to ``∂_t pᵣ = - ρ̄ g W``,
where ``W`` is the `vertical_velocity` of the frame and ``ρ̄ = pᵣ / (R^m T̄)``
is the domain-mean density. The `temperature` and moisture mass fractions are the domain means,
and `time` is the time at which `pressure` was last advanced.

The frame state is always stored in `Float64`, also on single-precision grids: the per-step change
of the reference pressure, ``ρ̄ g W Δt / pᵣ ∼ 10⁻⁷`` for ``W = 2`` m s⁻¹ and ``Δt = 1`` ms, is
comparable to the `Float32` machine epsilon and would be rounded away.
"""
mutable struct MovingFrame{FT}
    vertical_velocity :: FT
    gravitational_acceleration :: FT
    time :: FT
    pressure :: FT
    temperature :: FT
    vapor_mass_fraction :: FT
    liquid_mass_fraction :: FT
    ice_mass_fraction :: FT
end

"""
    MovingFrameField{N}

A spatially uniform field whose value is the property `N` of a [`MovingFrame`](@ref).
Converted to a `ConstantField` with the current value of the property when passed to a GPU kernel.
"""
struct MovingFrameField{N, FT, F}
    frame :: F
end

MovingFrameField{N}(FT, frame::F) where {N, F} = MovingFrameField{N, FT, F}(frame)

# The Float64 frame state is converted to the float type FT of the grid
@inline Base.getindex(field::MovingFrameField{N, FT}, i...) where {N, FT} = convert(FT, getfield(field.frame, N))
Adapt.adapt_structure(to, field::MovingFrameField{N, FT}) where {N, FT} = ConstantField(convert(FT, getfield(field.frame, N)))

reference_value(field::ConstantField) = field.constant
reference_value(field::MovingFrameField{N}) where N = getfield(field.frame, N)

Adapt.adapt_structure(to, ref::BoussinesqReferenceState) =
    BoussinesqReferenceState(adapt(to, ref.base_pressure),
                             adapt(to, ref.surface_pressure),
                             adapt(to, ref.surface_density),
                             adapt(to, ref.surface_temperature),
                             adapt(to, ref.potential_temperature),
                             adapt(to, ref.standard_pressure),
                             adapt(to, ref.pressure),
                             adapt(to, ref.density),
                             adapt(to, ref.temperature),
                             adapt(to, ref.vapor_mass_fraction),
                             adapt(to, ref.liquid_mass_fraction),
                             adapt(to, ref.ice_mass_fraction),
                             nothing)

Base.eltype(::BoussinesqReferenceState{FT}) where FT = FT

"""
$(TYPEDSIGNATURES)

Return a [`BoussinesqReferenceState`](@ref) on `grid` with uniform `pressure`, `temperature`,
and moisture mass fractions. The reference density follows from the ideal gas law,
``ρᵣ = pᵣ / (Rᵐ Tᵣ)``, where ``Rᵐ`` is the mixture gas constant of the reference moisture.

Keyword arguments
=================
- `pressure`: Reference pressure ``pᵣ``. By default, 101325.
- `temperature`: Reference temperature ``Tᵣ``. By default, 288.
- `standard_pressure`: Reference pressure for potential temperature (``pˢᵗ``). By default, 1e5.
- `vapor_mass_fraction`: Reference vapor mass fraction ``qᵛᵣ``. By default, 0.
- `liquid_mass_fraction`: Reference liquid mass fraction ``qˡᵣ``. By default, 0.
- `ice_mass_fraction`: Reference ice mass fraction ``qⁱᵣ``. By default, 0.
- `frame_vertical_velocity`: Vertical velocity ``W`` of the reference frame, positive upwards.
  By default, `nothing`, which fixes the reference state in time.
  Otherwise, the reference pressure evolves hydrostatically as the frame moves,
  ``∂_t pᵣ = - ρ̄ g W``, so that the domain cools when it rises and warms when it sinks.
  The reference temperature and moisture mass fractions follow the domain mean,
  so that buoyancy is measured relative to the domain mean.
  The reference density, which is used by the dynamics, is fixed.
  A moving frame requires the liquid-ice potential temperature formulation.

Examples
========

```jldoctest
julia> using Breeze

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Periodic));

julia> reference_state = BoussinesqReferenceState(grid, temperature=290)
BoussinesqReferenceState{Float64}(pᵣ=101325.0, Tᵣ=290.0, ρᵣ=1.2174)

julia> rising_reference_state = BoussinesqReferenceState(grid, temperature=290, frame_vertical_velocity=2)
BoussinesqReferenceState{Float64}(pᵣ=101325.0, Tᵣ=290.0, ρᵣ=1.2174, frame_vertical_velocity=2.0)
```
"""
function BoussinesqReferenceState(grid, constants=ThermodynamicConstants(eltype(grid));
                                  pressure = 101325,
                                  temperature = 288,
                                  standard_pressure = 1e5,
                                  vapor_mass_fraction = 0,
                                  liquid_mass_fraction = 0,
                                  ice_mass_fraction = 0,
                                  frame_vertical_velocity = nothing)

    FT = eltype(grid)
    pᵣ = convert(FT, pressure)
    Tᵣ = convert(FT, temperature)
    pˢᵗ = convert(FT, standard_pressure)
    qᵣ = MoistureMassFractions(convert(FT, vapor_mass_fraction),
                               convert(FT, liquid_mass_fraction),
                               convert(FT, ice_mass_fraction))

    Rᵐ = mixture_gas_constant(qᵣ, constants)
    ρᵣ = pᵣ / (Rᵐ * Tᵣ)

    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    θᵣ = Tᵣ * (pˢᵗ / pᵣ)^(Rᵈ / cᵖᵈ)

    pˢ = surface_state_field(grid, pᵣ)
    ρˢ = surface_state_field(grid, ρᵣ)
    Tˢ = surface_state_field(grid, Tᵣ)

    frame = moving_frame(frame_vertical_velocity, constants, pressure, temperature,
                         vapor_mass_fraction, liquid_mass_fraction, ice_mass_fraction)
    p, T, qᵛ, qˡ, qⁱ = reference_fields(frame, pᵣ, Tᵣ, qᵣ)

    return BoussinesqReferenceState(pᵣ, pˢ, ρˢ, Tˢ, θᵣ, pˢᵗ, p, ConstantField(ρᵣ), T, qᵛ, qˡ, qⁱ, frame)
end

moving_frame(::Nothing, constants, p, T, qᵛ, qˡ, qⁱ) = nothing

# The time is set when the frame is first advanced
moving_frame(W, constants, p, T, qᵛ, qˡ, qⁱ) =
    MovingFrame(Float64(W), Float64(constants.gravitational_acceleration), NaN,
                Float64(p), Float64(T), Float64(qᵛ), Float64(qˡ), Float64(qⁱ))

reference_fields(::Nothing, pᵣ, Tᵣ, qᵣ) =
    ConstantField(pᵣ), ConstantField(Tᵣ), ConstantField(qᵣ.vapor), ConstantField(qᵣ.liquid), ConstantField(qᵣ.ice)

reference_fields(frame::MovingFrame, pᵣ::FT, Tᵣ, qᵣ) where FT =
    MovingFrameField{:pressure}(FT, frame), MovingFrameField{:temperature}(FT, frame),
    MovingFrameField{:vapor_mass_fraction}(FT, frame), MovingFrameField{:liquid_mass_fraction}(FT, frame),
    MovingFrameField{:ice_mass_fraction}(FT, frame)

frame_summary(::Nothing) = ""
frame_summary(frame::MovingFrame) = string(", frame_vertical_velocity=", prettysummary(frame.vertical_velocity))

function Base.summary(ref::BoussinesqReferenceState)
    FT = eltype(ref)
    return string("BoussinesqReferenceState{$FT}(pᵣ=", prettysummary(reference_value(ref.pressure)),
                  ", Tᵣ=", prettysummary(reference_value(ref.temperature)),
                  ", ρᵣ=", prettysummary(reference_value(ref.density)),
                  frame_summary(ref.frame), ")")
end

Base.show(io::IO, ref::BoussinesqReferenceState) = print(io, summary(ref))

surface_density(ref::BoussinesqReferenceState) = reference_value(ref.density)

#####
##### Moving frames
#####

"""
$(TYPEDSIGNATURES)

Advance the reference pressure of a moving `frame` from `frame.time` to `time`, assuming that the domain-mean
temperature and moisture are constant over the interval, and that the mean density is ``ρ̄ = pᵣ / (R^m T̄)``.
"""
function advance_frame!(frame::MovingFrame, time, constants)
    if isnan(frame.time)
        frame.time = time
        return nothing
    end

    q̄ = MoistureMassFractions(frame.vapor_mass_fraction, frame.liquid_mass_fraction, frame.ice_mass_fraction)
    Rᵐ = mixture_gas_constant(q̄, constants)
    T̄ = frame.temperature
    g = frame.gravitational_acceleration
    W = frame.vertical_velocity
    Δt = time - frame.time

    # ∂t pᵣ = - ρ̄ g W = - pᵣ g W / (Rᵐ T̄)
    frame.pressure *= exp(- g * W * Δt / (Rᵐ * T̄))
    frame.time = time

    return nothing
end

advance_frame!(::Nothing, time, constants) = nothing

"""
$(TYPEDSIGNATURES)

Set the time of a moving `frame` to `time` without changing its pressure, so that the current
pressure becomes the pressure at `time`.
"""
function synchronize_frame!(frame::MovingFrame, time)
    frame.time = time
    return nothing
end

synchronize_frame!(::Nothing, time) = nothing

Oceananigans.prognostic_state(frame::MovingFrame) = (time = frame.time,
                                                     pressure = frame.pressure,
                                                     temperature = frame.temperature,
                                                     vapor_mass_fraction = frame.vapor_mass_fraction,
                                                     liquid_mass_fraction = frame.liquid_mass_fraction,
                                                     ice_mass_fraction = frame.ice_mass_fraction)

function Oceananigans.restore_prognostic_state!(frame::MovingFrame, from)
    frame.time = from.time
    frame.pressure = from.pressure
    frame.temperature = from.temperature
    frame.vapor_mass_fraction = from.vapor_mass_fraction
    frame.liquid_mass_fraction = from.liquid_mass_fraction
    frame.ice_mass_fraction = from.ice_mass_fraction
    return frame
end

Oceananigans.restore_prognostic_state!(::MovingFrame, ::Nothing) = nothing

#####
##### A Boussinesq reference state is not hydrostatic, and cannot be recomputed as one
#####

function throw_boussinesq_reference_error(name)
    throw(ArgumentError("$name is not supported for a BoussinesqReferenceState, " *
                        "which is uniform in space rather than hydrostatically balanced."))
end

compute_hydrostatic_reference!(::BoussinesqReferenceState, constants) =
    throw_boussinesq_reference_error("compute_hydrostatic_reference!")

compute_reference_state!(::BoussinesqReferenceState, T̄, q̄ᵛ, q̄ˡ, q̄ⁱ, constants) =
    throw_boussinesq_reference_error("compute_reference_state!")

compute_reference_state!(::BoussinesqReferenceState, T̄, q̄ᵗ, constants) =
    throw_boussinesq_reference_error("compute_reference_state!")
