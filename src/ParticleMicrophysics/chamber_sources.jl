#####
##### Particle sources and sinks for cloud-chamber "core region" simulations
#####
##### A triply-periodic box represents the core of a cloud chamber (Chen et al. 2025, doi:10.1029/2024MS004562,
##### NCAR-DNS): aerosol particles are injected at a prescribed rate at random positions, and particles are
##### removed when their *virtual* position, which follows the particle's trajectory in the coordinates of the
##### physical chamber while the computational position stays periodic, leaves the chamber. The approximation
##### represents the chamber walls as a sink at the rate at which particles would reach them by advection and settling.
#####
##### Scheduling: the sources act once per time step, at the end of `microphysics_model_update!`, after the particles
##### were advanced through the stages of the time stepper and have condensed, and before the liquid mass fraction
##### is diagnosed and the state is refreshed. Events (injection batches, removal resamplings) are due at fixed times
##### `start_time + n × interval`; a step may complete several events, each of which is then processed in turn,
##### and a time step longer than an interval is rejected for the rule that is defined per interval.
#####

#####
##### A small deterministic random-number generator (SplitMix64), so that injections and removals are
##### reproducible for a given seed without a dependency on `Random`
#####

mutable struct SplitMix64
    state :: UInt64
end

SplitMix64(seed::Integer) = SplitMix64(UInt64(seed) * 0x9e3779b97f4a7c15 + 0x7f4a7c15f39cc060)

function next_uint64!(rng::SplitMix64)
    rng.state += 0x9e3779b97f4a7c15
    z = rng.state
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

# A uniform deviate in [0, 1)
uniform_deviate!(rng::SplitMix64) = (next_uint64!(rng) >> 11) * 2.0^-53

#####
##### Injection
#####

"""
    AerosolInjection(FT = Float64; rate, dry_radius, hygroscopicity, interval,
                     multiplicity = 1, wet_radius_factor = cbrt(2), start_time = 0, seed = 1)

Inject aerosol particles into the domain at the prescribed number source `rate` (m⁻³ s⁻¹ of physical
air volume), in batches at the times `start_time + n × interval`, at uniformly random positions. Each particle
is dry aerosol of `dry_radius` and `hygroscopicity` κ, inserted as haze with wet radius `wet_radius_factor × dry_radius`
(the reference inserts a wet volume of twice the dry volume) and zero terminal velocity, and represents
`multiplicity` physical particles. The batch size is `rate × V × interval / multiplicity` computational particles,
carried fractionally between batches so that the long-run physical rate is exact (the reference's
`chamber_inject`: `count = n_inj V Δt_inj / multi_inj`).

The injected haze water appears without a vapor sink, as in the reference; it is recorded in the
[`SourceSinkBookkeeping`](@ref) of the microphysics.
"""
struct AerosolInjection{FT, R}
                 rate :: FT
           dry_radius :: FT
       hygroscopicity :: FT
             interval :: FT
         multiplicity :: FT
    wet_radius_factor :: FT
           start_time :: FT
                  rng :: R
end

function AerosolInjection(FT = Float64; rate, dry_radius, hygroscopicity, interval, multiplicity = 1,
                          wet_radius_factor = cbrt(2), start_time = 0, seed = 1)
    rate > 0 || throw(ArgumentError("the injection rate must be positive"))
    interval > 0 || throw(ArgumentError("the injection interval must be positive"))
    multiplicity > 0 || throw(ArgumentError("the injection multiplicity must be positive"))
    return AerosolInjection{FT, SplitMix64}(convert(FT, rate), convert(FT, dry_radius), convert(FT, hygroscopicity),
                                            convert(FT, interval), convert(FT, multiplicity), convert(FT, wet_radius_factor),
                                            convert(FT, start_time), SplitMix64(seed))
end

Base.summary(s::AerosolInjection) = string("AerosolInjection(rate=", prettysummary(s.rate), " m⁻³ s⁻¹, rᵈ=",
                                           prettysummary(s.dry_radius), ", κ=", prettysummary(s.hygroscopicity),
                                           ", interval=", prettysummary(s.interval), ", ξ=", prettysummary(s.multiplicity), ")")

#####
##### Removal at the walls of a virtual chamber
#####

"""
    IntegratedVirtualPosition()

Removal rule of Chen et al. (2025): each particle carries a virtual position in the physical chamber,
initialized at random, that follows the particle's trajectory, and the particle is removed when the
virtual position leaves the chamber. The virtual position is advanced every time step by the
particle's displacement in the periodic domain, unwrapped, so that it follows exactly the trajectory
integrated by the time stepper (air velocity plus settling, through all stages). Particles that left are
removed every time step; the `interval` of [`VirtualChamberRemoval`](@ref) is not used.
"""
struct IntegratedVirtualPosition end

"""
    ResampledVirtualPosition()

Removal rule of the warm-phase-dns code (PRECISION-GM/warm-phase-dns 4322f1b, `chamber_remove`): at the times
`n × interval`, each particle is placed at a new uniformly random virtual position in the chamber, and is
removed if that position, projected with the particle velocity (air velocity of its cell less its terminal
velocity) over the interval, leaves the chamber. This is a memoryless approximation of
[`IntegratedVirtualPosition`](@ref). The time step must not exceed the interval.
"""
struct ResampledVirtualPosition end

"""
    VirtualChamberRemoval(FT = Float64; size = (2, 2, 1), interval = 0.1,
                          rule = IntegratedVirtualPosition(), seed = 2)

Remove particles whose virtual position leaves a chamber of `size` (m), see [`IntegratedVirtualPosition`](@ref)
and [`ResampledVirtualPosition`](@ref). `interval` (s) is the resampling interval of the resampled rule.
Removed particles vanish with their liquid water, without returning vapor or heat to the air, as in the
reference; the removed number and liquid mass are recorded in the [`SourceSinkBookkeeping`](@ref) of the microphysics.
"""
struct VirtualChamberRemoval{FT, U, R}
        size :: NTuple{3, FT}
    interval :: FT
        rule :: U
         rng :: R
end

function VirtualChamberRemoval(FT = Float64; size = (2, 2, 1), interval = 0.1, rule = IntegratedVirtualPosition(), seed = 2)
    all(>(0), size) || throw(ArgumentError("the chamber size must be positive"))
    interval > 0 || throw(ArgumentError("the removal interval must be positive"))
    return VirtualChamberRemoval(convert.(FT, Tuple(size)), convert(FT, interval), rule, SplitMix64(seed))
end

Base.summary(s::VirtualChamberRemoval) = string("VirtualChamberRemoval(size=", s.size, ", interval=", prettysummary(s.interval),
                                                ", ", nameof(typeof(s.rule)), ")")

"""
    SourceSinkBookkeeping

Counts and masses exchanged by [`AerosolInjection`](@ref) and [`VirtualChamberRemoval`](@ref) since the start
of the simulation: `injected` and `removed` computational particles, `injected_physical` and `removed_physical`
(multiplicity-weighted counts), `injected_liquid_mass` (the haze water of injected particles, kg) and
`removed_liquid_mass` (the liquid water of removed particles, kg; both multiplicity-weighted), the times of the
next injection and removal events, and the fractional batch carried to the next injection.

The counts and masses accumulate in `Float64` whatever the precision of the model, and the mass of each removed
droplet is evaluated in `Float64` from its stored radii: the accumulated masses are then exact to the precision of
the stored state, and a water budget closes to the rounding of the model's own vapor updates (see the tests).
"""
mutable struct SourceSinkBookkeeping{FT}
          next_injection :: FT
            next_removal :: FT
         injection_carry :: FT
                injected :: Int
                 removed :: Int
       injected_physical :: Float64
        removed_physical :: Float64
    injected_liquid_mass :: Float64
     removed_liquid_mass :: Float64
end

first_event(::Nothing, FT) = convert(FT, Inf)
first_event(s::AerosolInjection, FT) = convert(FT, s.start_time + s.interval)
first_event(s::VirtualChamberRemoval, FT) = s.interval

SourceSinkBookkeeping(FT, injection, removal) =
    SourceSinkBookkeeping{FT}(first_event(injection, FT), first_event(removal, FT), 0, 0, 0, 0, 0, 0, 0)

"""
    ChamberSources(; injection = nothing, removal = nothing)

The particle sources and sinks of [`LagrangianMicrophysics`](@ref): an [`AerosolInjection`](@ref), a
[`VirtualChamberRemoval`](@ref), or both, with their [`SourceSinkBookkeeping`](@ref).
"""
struct ChamberSources{I, R, B, W}
    injection :: I
      removal :: R
  bookkeeping :: B
    workspace :: W
end

ChamberSources(; injection = nothing, removal = nothing) = ChamberSources(injection, removal, nothing, nothing)

function Base.summary(s::ChamberSources)
    parts = String[]
    isnothing(s.injection) || push!(parts, summary(s.injection))
    isnothing(s.removal) || push!(parts, summary(s.removal))
    return join(parts, ", ")
end

# The workspace holds the removal flags (which Oceananigans' compaction reads), the flags accumulated over the
# events of one step, and a host buffer for new positions
function materialize_sources(s::ChamberSources, properties, FT)
    bookkeeping = SourceSinkBookkeeping(FT, s.injection, s.removal)
    workspace = (removal = ParticleRemoval(properties), flagged = similar(properties.x, Bool, 0), host = Vector{FT}(undef, 0))
    return ChamberSources(s.injection, s.removal, bookkeeping, workspace)
end

materialize_sources(::Nothing, properties, FT) = nothing
materialize_sources(s::ChamberSources{<:Any, <:Any, <:SourceSinkBookkeeping}, properties, FT) = s

# Lagrangian particles with the removal workspace of the sources, so that `remove_absorbed_particles!` can compact them
with_removal(particles::LagrangianParticles, ::Nothing) = particles
with_removal(particles::LagrangianParticles, s::ChamberSources) =
    LagrangianParticles(particles.properties, particles.boundary_conditions, particles.tracked_fields, particles.dynamics,
                        particles.parameters, particles.sorting, s.workspace.removal, particles.tendencies)

# A virtual position congruent with the computational position x (measured from the domain origin) modulo the
# domain length L, uniformly distributed over the images of x that lie in a chamber of length Lᶜ
function congruent_virtual_position(rng, x, L, Lᶜ)
    nimages = max(1, floor(Int, (Lᶜ - mod(x, L)) / L)) # images x + kL with 0 ≤ k < nimages lie in [0, Lᶜ)
    k = min(floor(Int, uniform_deviate!(rng) * nimages), nimages - 1)
    return mod(x, L) + k * L
end

# Initial virtual positions of the droplets: uniformly random in the chamber, congruent with their computational positions
function initial_virtual_positions(::Nothing, initial, grid, FT)
    Nd = length(initial.x)
    return (xᵛ = zeros(FT, Nd), yᵛ = zeros(FT, Nd), zᵛ = zeros(FT, Nd))
end

function initial_virtual_positions(s::ChamberSources, initial, grid, FT)
    isnothing(s.removal) && return initial_virtual_positions(nothing, initial, grid, FT)
    rng = s.removal.rng
    Lᶜ = s.removal.size
    L = (grid.Lx, grid.Ly, grid.Lz)
    origin = map(d -> domain_origin(grid, d), (1, 2, 3))
    virtual = map((1, 2, 3), (initial.x, initial.y, initial.z)) do d, ξ
        [convert(FT, congruent_virtual_position(rng, ξ[p] - origin[d], L[d], Lᶜ[d])) for p in eachindex(ξ)]
    end
    return (xᵛ = virtual[1], yᵛ = virtual[2], zᵛ = virtual[3])
end

#####
##### Kernels
#####

# The velocity of droplet p relative to the grid: the air velocity of its cell less its terminal velocity
@inline function droplet_velocity(droplets, p, grid, velocities)
    @inbounds X = (droplets.x[p], droplets.y[p], droplets.z[p])
    i, j, k = droplet_cell(X, grid)
    @inbounds 𝕎 = droplets.𝕎[p]
    return (ℑxᶜᵃᵃ(i, j, k, grid, velocities.u), ℑyᵃᶜᵃ(i, j, k, grid, velocities.v), ℑzᵃᵃᶜ(i, j, k, grid, velocities.w) - 𝕎)
end

@inline outside((x, y, z), L) = (x < 0) | (x ≥ L[1]) | (y < 0) | (y ≥ L[2]) | (z < 0) | (z ≥ L[3])

# The displacement of the computational position ξ (measured from the domain origin) since the virtual position ξᵛ
# was last updated, unwrapped across the periodic boundaries: the two are congruent modulo L, so the displacement is
# the representative of ξ - ξᵛ in (-L/2, L/2]. Exact as long as a particle moves less than half the domain per step.
@inline function unwrapped_displacement(ξ, ξᵛ, L)
    δ = ξ - ξᵛ
    return δ - L * round(δ / L)
end

# Integrated rule: advance the virtual positions by the unwrapped displacement of the step and flag the particles that left
@kernel function _advance_virtual_positions!(flags, droplets, origin, L, Lᶜ)
    p = @index(Global)
    @inbounds begin
        xᵛ = droplets.xᵛ[p] + unwrapped_displacement(droplets.x[p] - origin[1], droplets.xᵛ[p], L[1])
        yᵛ = droplets.yᵛ[p] + unwrapped_displacement(droplets.y[p] - origin[2], droplets.yᵛ[p], L[2])
        zᵛ = droplets.zᵛ[p] + unwrapped_displacement(droplets.z[p] - origin[3], droplets.zᵛ[p], L[3])
        droplets.xᵛ[p] = xᵛ
        droplets.yᵛ[p] = yᵛ
        droplets.zᵛ[p] = zᵛ
        flags[p] = outside((xᵛ, yᵛ, zᵛ), Lᶜ)
    end
end

# Resampled rule: the virtual position was redrawn (on the host) and is projected over the removal interval
@kernel function _project_virtual_positions!(flags, droplets, grid, interval, velocities, Lᶜ)
    p = @index(Global)
    u, v, w = droplet_velocity(droplets, p, grid, velocities)
    @inbounds begin
        X = (droplets.xᵛ[p] + u * interval, droplets.yᵛ[p] + v * interval, droplets.zᵛ[p] + w * interval)
        flags[p] |= outside(X, Lᶜ)
    end
end

#####
##### Applying the sources once per time step
#####

apply_chamber_sources!(::Nothing, μ, model, Δt) = nothing

function apply_chamber_sources!(s::ChamberSources, μ, model, Δt)
    remove_particles!(s.removal, s, μ, model, Δt)
    inject_particles!(s.injection, s, μ, model, Δt)
    return nothing
end

# Events are due when the clock has reached their time, within a relative tolerance of the clock's precision
@inline event_due(time, event, FT) = time ≥ event * (1 - 4 * eps(FT))

remove_particles!(::Nothing, s, μ, model, Δt) = nothing

function remove_particles!(removal::VirtualChamberRemoval, s::ChamberSources, μ, model, Δt)
    droplets = μ.droplets
    properties = droplets.properties
    Np = length(properties)
    grid = model.grid
    FT = eltype(grid)
    book = s.bookkeeping
    time = model.clock.time

    flags = s.workspace.flagged
    resize!(flags, Np)
    fill!(flags, false)
    flag_particles!(removal.rule, flags, properties, grid, model.velocities, removal, book, time, Δt, s.workspace)
    Np == 0 && return nothing

    # Bookkeeping of what leaves, then compaction (Oceananigans keeps the particle order and compacts the tendencies)
    nremoved = count(flags)
    nremoved == 0 && return nothing
    constants = model.thermodynamic_constants
    book.removed += nremoved
    book.removed_physical += mapreduce((f, ξ) -> f * Float64(ξ), +, flags, properties.ξ; init = 0.0)
    book.removed_liquid_mass += mapreduce((f, r, rᵈ, ξ) -> f * Float64(ξ) * droplet_water_mass(Float64(r), Float64(rᵈ), constants), +,
                                          flags, properties.r, properties.rᵈ, properties.ξ; init = 0.0)
    absorbed = s.workspace.removal.absorbed
    resize!(absorbed, Np)
    copyto!(absorbed, flags)
    remove_absorbed_particles!(droplets, grid)
    return nothing
end

function flag_particles!(::IntegratedVirtualPosition, flags, properties, grid, velocities, removal, book, time, Δt, workspace)
    Np = length(properties)
    Np == 0 && return nothing
    origin = map(d -> domain_origin(grid, d), (1, 2, 3))
    L = (grid.Lx, grid.Ly, grid.Lz)
    launch!(architecture(grid), grid, KernelParameters(1:Np), _advance_virtual_positions!,
            flags, properties, origin, L, removal.size)
    return nothing
end

function flag_particles!(::ResampledVirtualPosition, flags, properties, grid, velocities, removal, book, time, Δt, workspace)
    FT = eltype(grid)
    Δt ≤ removal.interval * (1 + 4 * eps(FT)) ||
        throw(ArgumentError("the time step $Δt exceeds the removal interval $(removal.interval) of ResampledVirtualPosition"))
    Np = length(properties)
    host = workspace.host
    Lᶜ = removal.size
    rng = removal.rng
    while event_due(time, book.next_removal, FT)
        book.next_removal += removal.interval
        Np == 0 && continue
        resize!(host, Np)
        for (coordinate, Lᵈ) in zip((properties.xᵛ, properties.yᵛ, properties.zᵛ), Lᶜ)
            for p in 1:Np
                @inbounds host[p] = convert(FT, uniform_deviate!(rng) * Lᵈ)
            end
            copyto!(coordinate, host)
        end
        launch!(architecture(grid), grid, KernelParameters(1:Np), _project_virtual_positions!,
                flags, properties, grid, removal.interval, velocities, Lᶜ)
    end
    return nothing
end

inject_particles!(::Nothing, s, μ, model, Δt) = nothing

function inject_particles!(injection::AerosolInjection, s::ChamberSources, μ, model, Δt)
    FT = eltype(model.grid)
    book = s.bookkeeping
    time = model.clock.time
    while event_due(time, book.next_injection, FT)
        book.next_injection += injection.interval
        inject_batch!(injection, s, μ, model)
    end
    return nothing
end

function inject_batch!(injection::AerosolInjection, s::ChamberSources, μ, model)
    grid = model.grid
    FT = eltype(grid)
    book = s.bookkeeping
    Nx, Ny, Nz = size(grid)
    V = Float64(Vᶜᶜᶜ(1, 1, 1, grid)) * Nx * Ny * Nz
    batch = Float64(injection.rate) * V * Float64(injection.interval) / Float64(injection.multiplicity) + Float64(book.injection_carry)
    n = floor(Int, batch)
    book.injection_carry = convert(FT, batch - n)
    n == 0 && return nothing

    properties = μ.droplets.properties
    Np = length(properties)
    rng = injection.rng
    rᵈ = injection.dry_radius
    r = injection.wet_radius_factor * rᵈ
    origin = map(d -> domain_origin(grid, d), (1, 2, 3))
    L = (grid.Lx, grid.Ly, grid.Lz)
    Lᶜ = isnothing(s.removal) ? L : s.removal.size

    ξ = map((1, 2, 3)) do d
        [uniform_deviate!(rng) * L[d] for _ in 1:n]
    end
    virtual = map((1, 2, 3)) do d
        [convert(FT, congruent_virtual_position(rng, ξ[d][p], L[d], Lᶜ[d])) for p in 1:n]
    end
    new = (x = [convert(FT, origin[1] + ξ[1][p]) for p in 1:n],
           y = [convert(FT, origin[2] + ξ[2][p]) for p in 1:n],
           z = [convert(FT, origin[3] + ξ[3][p]) for p in 1:n],
           r = fill(convert(FT, r), n), rᵈ = fill(rᵈ, n), κ = fill(injection.hygroscopicity, n), 𝕎 = zeros(FT, n),
           ξ = fill(injection.multiplicity, n), xᵛ = virtual[1], yᵛ = virtual[2], zᵛ = virtual[3])

    resize!(properties, Np + n)
    for name in propertynames(new)
        copyto!(getproperty(properties, name), Np + 1, getproperty(new, name), 1, n)
    end

    book.injected += n
    book.injected_physical += n * Float64(injection.multiplicity)
    book.injected_liquid_mass += n * Float64(injection.multiplicity) * droplet_water_mass(Float64(r), Float64(rᵈ), model.thermodynamic_constants)
    return nothing
end

# Origin of the regular grid in direction d
domain_origin(grid, d) = first((xnodes, ynodes, znodes)[d](grid, Face()))

validate_sources(::Nothing, grid) = nothing

function validate_sources(::ChamberSources, grid)
    all(T -> T === Periodic, topology(grid)) ||
        throw(ArgumentError("ChamberSources require a triply periodic grid, but the grid's topology is $(topology(grid))."))
    return nothing
end
