#####
##### Lagrangian microphysics: cloud droplets as Lagrangian particles
#####
##### Each droplet is a particle with a wet radius, a dry (solute) radius, and a hygroscopicity.
##### Droplets move with the air and fall at their terminal velocity, and they are advanced with the
##### time-stepping scheme of the model, stage by stage.
#####
##### All liquid water lives on the droplets: the thermodynamic state of the air on the grid contains
##### only vapor, so the potential temperature is θ = T / Π, and droplets carry their liquid
##### (and its enthalpy) with them without heating or cooling the air they leave behind.
##### The liquid-ice potential temperature would instead carry a latent-heat deficit at every droplet,
##### which is advected by the grid while the droplets move as particles; with about one droplet per
##### cell, that mismatch creates temperature noise of order ℒ qˡ / cᵖ.
#####
##### Once per time step, after the dynamics, `microphysics_model_update!` couples droplets and air:
##### each droplet grows or evaporates at the supersaturation of its cell, the condensed mass is moved
##### from vapor to the droplets, and the latent heat it releases warms the cell at constant pressure.
##### Droplets interact with the grid through the cell that contains them. Because a droplet samples the
##### cell into which it deposits its own sink, its growth is corrected for its own contribution to that cell
##### (`DiffusiveSelfInteraction`).
#####
##### TODO: add the liquid loading of the droplets to the buoyancy.
#####

"""
    Droplet{FT}

A cloud droplet with position `(x, y, z)`, wet radius `r`, dry (solute) radius `rᵈ`,
hygroscopicity `κ`, and terminal velocity `𝕎` (positive downwards).
"""
struct Droplet{FT}
     x :: FT
     y :: FT
     z :: FT
     r :: FT
    rᵈ :: FT
     κ :: FT
     𝕎 :: FT
end

"""
    IceParticle{FT}

An ice particle with position `(x, y, z)`, equivalent-volume radius `r` (zero once the particle has
sublimated completely), and terminal velocity `𝕎` (positive downwards).
"""
struct IceParticle{FT}
    x :: FT
    y :: FT
    z :: FT
    r :: FT
    𝕎 :: FT
end

"""
    DiffusiveSelfInteraction()

Correct the condensational growth of each droplet for its own sink in the cell it samples.

A droplet grows at the supersaturation of the cell into which it deposits its condensed vapor and
latent heat, so the cell value includes the droplet's own vapor deficit and warming. The growth law,
however, already accounts for the droplet's own vapor and temperature fields from its surface to
infinity, and needs the ambient supersaturation without the droplet. Sampling the cell counts part of
the droplet's own deficit twice. The error is about 𝒮 r / (0.31 Δ) for a cell of width Δ, so it grows
as the grid is refined.

The correction assumes that the droplet's own perturbation of its cell is quasi-steady and set by the
molecular diffusion of vapor and heat on the grid. The cell value of a point sink ṁ is then δρᵛ = - gᵛ ṁ / κᵛ
for vapor and δT = gᵀ ℒˡ ṁ / (ρ cᵖᵈ κᵀ) for temperature, where κᵛ and κᵀ are the diffusivities
of the closure for `ρqᵛ` and `ρθ`. Without time discretization, gᵛ = gᵀ = g, the cell value of the periodic
discrete Green's function of the grid's second-order Laplacian (g ≈ 0.2527 / Δ for cubic cells). Solving for
the ambient supersaturation replaces the growth resistance from r to infinity by the resistance from r to
the effective radius 1 / (4π g) ≈ 0.31 Δ of the cell.

Because the droplet deposits its sink at the end of a time step and samples its cell after the next
step's diffusion, the cell value is the steady state of the discrete map, Σₖ Pₖ / (1 - Pₖ) over the
Laplacian's modes k, where Pₖ is the time stepper's amplification factor for diffusion over one step
Δt. This is g - κΔt / (2V) + …, for a cell of volume V, and is evaluated for the current Δt from
a histogram of the Laplacian's eigenvalues that is computed once.

The correction requires a regular `RectilinearGrid` and a `ScalarDiffusivity` closure with constant
diffusivities for `ρθ` and `ρqᵛ`. Otherwise it is not applied. It neglects transport of the
droplet's own perturbation away from its cell by the flow and by settling, which shortens its
lifetime when the grid-scale strain rate or the settling speed exceed the cell's diffusive rate
κ / Δ² (or κ / Δ).
"""
struct DiffusiveSelfInteraction{G, E, F, A}
     greens_function :: G # cell value g of the Green's function of -∇² (1/m)
         eigenvalues :: E # mean eigenvalue of -∇² in each bin of a histogram of the nonzero modes
      mode_fractions :: F # fraction of all modes in each bin
       amplification :: A # amplification factor of the time stepper for diffusion
end

DiffusiveSelfInteraction() = DiffusiveSelfInteraction(nothing, nothing, nothing, nothing)

# Only the parameters computed by `self_interaction_parameters` enter kernels
Adapt.adapt_structure(to, ::DiffusiveSelfInteraction) = nothing

"""
    LagrangianMicrophysics(FT = Oceananigans.defaults.FloatType;
                           physics = DropletPhysicsParameters(FT),
                           self_interaction = DiffusiveSelfInteraction(),
                           sorting = nothing,
                           droplets = nothing,
                           ice_physics = IcePhysicsParameters(FT),
                           ice = nothing)

Return Lagrangian microphysics in which every cloud droplet is a Lagrangian particle.

`physics` is a [`DropletPhysicsParameters`](@ref) that determines condensational growth
and the terminal velocity of droplets. `self_interaction` corrects droplet growth for each droplet's own
sink in the cell it samples ([`DiffusiveSelfInteraction`](@ref)), or is `nothing` for no correction.
`droplets` is a `NamedTuple` of vectors with the initial
positions `x`, `y`, `z`, wet radii `r`, dry radii `rᵈ`, and hygroscopicities `κ` of the droplets,
or `nothing` for no droplets. `sorting` is an algorithm for sorting the droplets by grid cell once per
time step (`CountingSort()` or `ComparisonSort()` from Oceananigans), which keeps the memory accesses of
droplets in the same cell together, or `nothing` (the default) to keep the droplets in their initial order.
The droplets are allocated on the architecture of the grid
when the model is built, and can be accessed with `model.microphysics.droplets`.

`ice` is a `NamedTuple` of vectors with the initial positions `x`, `y`, `z` and equivalent-volume
radii `r` of ice particles, or `nothing` (the default) for no ice. Ice particles grow by vapor deposition
and shrink by sublimation according to `ice_physics` ([`IcePhysicsParameters`](@ref)), exchange vapor
and the latent heat of sublimation with the cell that contains them, and fall at their terminal velocity.
They are accessed with `model.microphysics.ice`.
"""
struct LagrangianMicrophysics{P, S, O, D, IP, I}
             physics :: P
    self_interaction :: S
             sorting :: O
            droplets :: D
         ice_physics :: IP
                 ice :: I
end

LagrangianMicrophysics(FT = Oceananigans.defaults.FloatType; physics = DropletPhysicsParameters(FT),
                       self_interaction = DiffusiveSelfInteraction(), sorting = nothing, droplets = nothing,
                       ice_physics = IcePhysicsParameters(FT), ice = nothing) =
    LagrangianMicrophysics(physics, self_interaction, sorting, droplets, ice_physics, ice)

const IceLagrangianMicrophysics = LagrangianMicrophysics{<:Any, <:Any, <:Any, <:Any, <:Any, <:Union{NamedTuple, LagrangianParticles}}

Base.summary(::LagrangianMicrophysics) = "LagrangianMicrophysics"

function Base.show(io::IO, μ::LagrangianMicrophysics)
    print(io, summary(μ), '\n',
              "├── self_interaction: ", self_interaction_summary(μ.self_interaction), '\n',
              "├── droplets: ", droplet_summary(μ.droplets), '\n',
              "└── ice: ", ice_summary(μ.ice))
end

ice_summary(::Nothing) = "none"
ice_summary(ice::LagrangianParticles) = string(length(ice), " ice particles")
ice_summary(ice::NamedTuple) = string(length(ice.x), " ice particles (not materialized)")

self_interaction_summary(::Nothing) = "nothing"
self_interaction_summary(::DiffusiveSelfInteraction{Nothing}) = "DiffusiveSelfInteraction()"
self_interaction_summary(s::DiffusiveSelfInteraction) =
    string("DiffusiveSelfInteraction(g=", prettysummary(s.greens_function), ", ", nameof(typeof(s.amplification)), ")")

droplet_summary(::Nothing) = "none"
droplet_summary(droplets::LagrangianParticles) = string(length(droplets), " droplets")
droplet_summary(droplets::NamedTuple) = string(length(droplets.x), " droplets (not materialized)")

# The droplets are not used inside kernels, where only the physics parameters are needed
Adapt.adapt_structure(to, μ::LagrangianMicrophysics) =
    LagrangianMicrophysics(adapt(to, μ.physics), adapt(to, μ.self_interaction), nothing, nothing,
                           adapt(to, μ.ice_physics), nothing)

# Droplets are moved to the architecture of the grid when the model is built
Architectures.on_architecture(arch, μ::LagrangianMicrophysics) = μ

#####
##### Materialization
#####

function AtmosphereModels.materialize_microphysics(μ::LagrangianMicrophysics, grid, timestepper)
    FT = eltype(grid)
    arch = architecture(grid)
    initial = initial_droplets(μ.droplets, FT)
    Nd = length(initial.x)

    components = (x = initial.x, y = initial.y, z = initial.z,
                  r = initial.r, rᵈ = initial.rᵈ, κ = initial.κ, 𝕎 = zeros(FT, Nd))

    components = map(c -> on_architecture(arch, convert(Vector{FT}, c)), components)
    properties = StructArray{Droplet{FT}}(components)
    droplets = LagrangianParticles(properties; sorting = μ.sorting)
    droplets = materialize_lagrangian_particles(droplets, timestepper)

    self_interaction = materialize_self_interaction(μ.self_interaction, grid, timestepper)
    ice = materialize_ice(μ.ice, grid, timestepper, μ.sorting)

    return LagrangianMicrophysics(μ.physics, self_interaction, μ.sorting, droplets, μ.ice_physics, ice)
end

AtmosphereModels.materialize_microphysics(μ::LagrangianMicrophysics{<:Any, <:Any, <:Any, <:LagrangianParticles}, grid, timestepper) = μ

materialize_ice(::Nothing, grid, timestepper, sorting) = nothing

function materialize_ice(ice::NamedTuple, grid, timestepper, sorting)
    FT = eltype(grid)
    arch = architecture(grid)
    Ni = length(ice.x)
    components = (x = ice.x, y = ice.y, z = ice.z, r = ice.r, 𝕎 = zeros(FT, Ni))
    components = map(c -> on_architecture(arch, convert(Vector{FT}, c)), components)
    particles = LagrangianParticles(StructArray{IceParticle{FT}}(components); sorting)
    return materialize_lagrangian_particles(particles, timestepper)
end

materialize_self_interaction(::Nothing, grid, timestepper) = nothing
materialize_self_interaction(s::DiffusiveSelfInteraction{<:Number}, grid, timestepper) = s

function materialize_self_interaction(::DiffusiveSelfInteraction{Nothing}, grid, timestepper)
    spectrum = laplacian_spectrum(grid)
    isnothing(spectrum) && return nothing
    return DiffusiveSelfInteraction(spectrum.greens_function, spectrum.eigenvalues, spectrum.mode_fractions,
                                    diffusion_amplification(timestepper))
end

#####
##### Diffusion over one time step
#####

struct RungeKutta3Amplification end
struct ExponentialAmplification end

# Any three-stage, third-order Runge-Kutta method advances dφ/dt = -κλφ by P = 1 - z + z²/2 - z³/6, z = κλΔt
diffusion_amplification(::Union{SSPRungeKutta3, AcousticRungeKutta3}) = RungeKutta3Amplification()
diffusion_amplification(timestepper) = ExponentialAmplification()

amplification_factor(::RungeKutta3Amplification, z) = 1 - z + z^2 / 2 - z^3 / 6
amplification_factor(::ExponentialAmplification, z) = exp(-z)

# h(z) = P / (1 - P) - 1 / z + 1 / 2, which is z / 12 + O(z²) for both amplification factors
function discrete_time_remainder(amplification, z)
    z < 1e-3 && return z / 12
    P = amplification_factor(amplification, z)
    return P / (1 - P) - 1 / z + 1 / 2
end

"""
$(TYPEDSIGNATURES)

Return the cell value of the droplet's own quasi-steady perturbation per unit sink, sampled one time step
after deposition, for diffusion over `τ = κ Δt`: the sum over the Laplacian's nonzero modes of
τ Pₖ / (1 - Pₖ) / V_domain, which is g + (τ / V) Σ fₖ (h(λₖ τ) - 1/2) with V the cell volume.
"""
function effective_greens_function(s::DiffusiveSelfInteraction, τ, V)
    remainder = zero(τ)
    for (λ, f) in zip(s.eigenvalues, s.mode_fractions)
        remainder += f * (discrete_time_remainder(s.amplification, λ * τ) - 1 / 2)
    end
    return s.greens_function + τ / V * remainder
end

#####
##### The discrete Green's function of the second-order Laplacian at the source cell
#####

# The eigenvalues of -∂² discretized with second-order differences on N cells of width Δ
lattice_eigenvalues(::Type{Periodic}, N, Δ) = [(2 - 2 * cospi(2m / N)) / Δ^2 for m in 0:N-1]
lattice_eigenvalues(::Type{Bounded}, N, Δ) = [(2 - 2 * cospi(m / N)) / Δ^2 for m in 0:N-1]

regular_spacing(Δ::Number) = Δ
regular_spacing(Δ) = nothing

"""
$(TYPEDSIGNATURES)

Return the cell value g of the discrete Green's function of `-∇²` (second-order differences)
on the regular `grid`, for a unit point source spread over one cell, excluding the domain mean:
the solution of `-∇² φ = δ / V` in the source cell is `φ = g`, with `V` the cell volume. For a large
cubic grid with spacing `Δ`, `g → 0.2527 / Δ` (Watson's integral). For `Bounded` directions the
Neumann eigenvalues are used, which gives the average of the cell value over the grid.
Returns `nothing` for grids that are not regular rectilinear grids without `Flat` directions.
"""
function periodic_lattice_greens_function(grid)
    spectrum = laplacian_spectrum(grid)
    return isnothing(spectrum) ? nothing : spectrum.greens_function
end

laplacian_spectrum(grid; bins = 1024) = nothing

# The Green's function g and a histogram of the nonzero eigenvalues of -∇² on a regular grid
function laplacian_spectrum(grid::RectilinearGrid; bins = 1024)
    TX, TY, TZ = topology(grid)
    Δx, Δy, Δz = regular_spacing(grid.Δxᶜᵃᵃ), regular_spacing(grid.Δyᵃᶜᵃ), regular_spacing(grid.z.Δᵃᵃᶜ)
    regular = !any(isnothing, (Δx, Δy, Δz)) && !any(T -> T === Flat, (TX, TY, TZ))
    regular || return nothing

    Nx, Ny, Nz = size(grid)
    λx = lattice_eigenvalues(TX, Nx, Float64(Δx))
    λy = lattice_eigenvalues(TY, Ny, Float64(Δy))
    λz = lattice_eigenvalues(TZ, Nz, Float64(Δz))
    λmax = maximum(λx) + maximum(λy) + maximum(λz)

    sum_of_inverses = 0.0
    counts = zeros(bins)
    sums = zeros(bins)
    for λᶻ in λz, λʸ in λy, λˣ in λx
        λ = λˣ + λʸ + λᶻ
        if λ > 0
            sum_of_inverses += 1 / λ
            b = clamp(ceil(Int, bins * λ / λmax), 1, bins)
            counts[b] += 1
            sums[b] += λ
        end
    end

    N = Nx * Ny * Nz
    occupied = counts .> 0
    greens_function = sum_of_inverses / (N * Float64(Δx * Δy * Δz))
    return (; greens_function, eigenvalues = sums[occupied] ./ counts[occupied], mode_fractions = counts[occupied] ./ N)
end

# The molecular diffusivities of vapor and heat of a closure, or `nothing` if they are not constant
molecular_diffusivities(closure) = nothing

function molecular_diffusivities(closure::ScalarDiffusivity{<:Any, ThreeDimensionalFormulation})
    κᵀ = get(closure.κ, :ρθ, nothing)
    κᵛ = get(closure.κ, :ρqᵛ, nothing)
    return (κᵀ isa Number && κᵛ isa Number) ? (; κᵀ, κᵛ) : nothing
end

# The parameters of the self-interaction correction passed to the condensation kernel
self_interaction_parameters(::Nothing, closure, grid, Δt) = nothing

function self_interaction_parameters(s::DiffusiveSelfInteraction, closure, grid, Δt)
    κ = molecular_diffusivities(closure)
    isnothing(κ) && return nothing
    FT = eltype(grid)
    V = Float64(Vᶜᶜᶜ(1, 1, 1, grid))
    gᵛ = effective_greens_function(s, Float64(κ.κᵛ * Δt), V)
    gᵀ = effective_greens_function(s, Float64(κ.κᵀ * Δt), V)
    return (vapor_greens_function = convert(FT, gᵛ), thermal_greens_function = convert(FT, gᵀ),
            vapor_diffusivity = convert(FT, κ.κᵛ), thermal_diffusivity = convert(FT, κ.κᵀ))
end

initial_droplets(::Nothing, FT) = (x = FT[], y = FT[], z = FT[], r = FT[], rᵈ = FT[], κ = FT[])
initial_droplets(droplets::NamedTuple, FT) = droplets

AtmosphereModels.microphysics_particles(μ::LagrangianMicrophysics) = μ.droplets
AtmosphereModels.microphysics_particles(μ::IceLagrangianMicrophysics) = (μ.droplets, μ.ice)

# Droplets fall relative to the air at their terminal velocity
@inline LagrangianParticleTracking.particle_w_velocity(droplets::AbstractVector{<:Droplet}, p, w) =
    @inbounds w - droplets.𝕎[p]

@inline LagrangianParticleTracking.particle_w_velocity(ice::AbstractVector{<:IceParticle}, p, w) =
    @inbounds w - ice.𝕎[p]

#####
##### Microphysics interface
#####

AtmosphereModels.prognostic_field_names(::LagrangianMicrophysics) = tuple()
AtmosphereModels.moisture_prognostic_name(::LagrangianMicrophysics) = :ρqᵛ

"""
$(TYPEDSIGNATURES)

Return the fields of `LagrangianMicrophysics`: the vapor mass fraction `qᵛ`,
the liquid mass fraction `qˡ` of the droplets in each cell (a diagnostic that does not
enter the thermodynamic state of the air), and the mass density `C` of water condensed
in each cell during the most recent time step.
"""
AtmosphereModels.materialize_microphysical_fields(::LagrangianMicrophysics, grid, boundary_conditions) =
    (qᵛ = CenterField(grid), qˡ = CenterField(grid), C = CenterField(grid))

# With ice: the ice mass fraction `qⁱ` of the ice particles in each cell and the mass density `D`
# of vapor deposited in each cell during the most recent time step
AtmosphereModels.materialize_microphysical_fields(::IceLagrangianMicrophysics, grid, boundary_conditions) =
    (qᵛ = CenterField(grid), qˡ = CenterField(grid), C = CenterField(grid),
     qⁱ = CenterField(grid), D = CenterField(grid))

@inline AtmosphereModels.microphysical_tendency(::LagrangianMicrophysics, name, ρ, ℳ, 𝒰, constants) = zero(ρ)
@inline AtmosphereModels.maybe_adjust_thermodynamic_state(𝒰, ::LagrangianMicrophysics, qᵛ, constants) = 𝒰

# The liquid water lives on the droplets, so the air on the grid contains only vapor
@inline AtmosphereModels.grid_moisture_fractions(i, j, k, grid, ::LagrangianMicrophysics, ρ, qᵛ, μ) =
    MoistureMassFractions(qᵛ)

@inline function AtmosphereModels.update_microphysical_fields!(μ, i, j, k, grid, ::LagrangianMicrophysics, ρ, 𝒰, constants)
    @inbounds μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor
    return nothing
end

AtmosphereModels.liquid_mass_fraction(::LagrangianMicrophysics, model) = model.microphysical_fields.qˡ
AtmosphereModels.ice_mass_fraction(::LagrangianMicrophysics, model) = nothing
AtmosphereModels.ice_mass_fraction(::IceLagrangianMicrophysics, model) = model.microphysical_fields.qⁱ

#####
##### Coupling droplets and air
#####

# The indices of the cell that contains the position (x, y, z)
@inline function droplet_cell((x, y, z), grid)
    X = LagrangianParticleTracking.flattened_node((x, y, z), grid)
    fi = FractionalIndices(X, grid, Face(), Face(), Face())
    i, _, _ = interpolator(fi.i)
    j, _, _ = interpolator(fi.j)
    k, _, _ = interpolator(fi.k)
    Nx, Ny, Nz = size(grid)
    return clamp(i, 1, Nx), clamp(j, 1, Ny), clamp(k, 1, Nz)
end

# Atomically add `value` to the parent array of a field at the interior indices (i, j, k)
@inline function atomic_add!(data, (Hx, Hy, Hz), i, j, k, value)
    @atomic data[i + Hx, j + Hy, k + Hz] += value
    return nothing
end

@kernel function _deposit_droplet_liquid!(qˡ, halo, droplets, grid, ρ, constants)
    p = @index(Global)
    @inbounds begin
        X = (droplets.x[p], droplets.y[p], droplets.z[p])
        r = droplets.r[p]
        rᵈ = droplets.rᵈ[p]
    end
    i, j, k = droplet_cell(X, grid)
    m = droplet_water_mass(r, rᵈ, constants)
    @inbounds ρᵢ = ρ[i, j, k]
    atomic_add!(qˡ, halo, i, j, k, m / (ρᵢ * Vᶜᶜᶜ(i, j, k, grid)))
end

@kernel function _condense_droplets!(C, halo, droplets, grid, Δt, T, ρ, ρqᵛ, p, constants, physics, self_interaction)
    n = @index(Global)
    @inbounds begin
        X = (droplets.x[n], droplets.y[n], droplets.z[n])
        r = droplets.r[n]
        rᵈ = droplets.rᵈ[n]
        κ = droplets.κ[n]
        𝕎 = droplets.𝕎[n]
    end

    i, j, k = droplet_cell(X, grid)

    @inbounds begin
        Tᵢ = T[i, j, k]
        ρᵢ = ρ[i, j, k]
        pᵢ = p[i, j, k]
        q = MoistureMassFractions(ρqᵛ[i, j, k] / ρᵢ)
    end

    # The vapor pressure follows from the pressure, pᵛ = qᵛ Rᵛ p / Rᵐ, rather than from the reference
    # density, which is constant in a Boussinesq frame that rises and expands.
    ρᵗ = pᵢ / (mixture_gas_constant(q, constants) * Tᵢ)
    𝒮 = supersaturation(Tᵢ, ρᵗ, q, constants, PlanarLiquidSurface())
    fᵛ = ventilation_factor(r, 𝕎, Tᵢ, pᵢ, constants, physics)
    step = condensational_growth_step(r, rᵈ, κ, 𝒮, fᵛ, Tᵢ, pᵢ, Δt, constants, physics, self_interaction)
    r⁺ = step.radius

    @inbounds begin
        droplets.r[n] = r⁺
        droplets.𝕎[n] = droplet_terminal_velocity(physics.terminal_velocity, r⁺, Tᵢ, pᵢ, constants, physics)
    end

    atomic_add!(C, halo, i, j, k, step.condensed_mass / Vᶜᶜᶜ(i, j, k, grid))
end

@kernel function _deposit_ice_mass!(qⁱ, halo, ice, grid, ρ, physics)
    p = @index(Global)
    @inbounds X = (ice.x[p], ice.y[p], ice.z[p])
    @inbounds r = ice.r[p]
    i, j, k = droplet_cell(X, grid)
    m = ice_particle_mass(r, physics)
    @inbounds ρᵢ = ρ[i, j, k]
    atomic_add!(qⁱ, halo, i, j, k, m / (ρᵢ * Vᶜᶜᶜ(i, j, k, grid)))
end

@kernel function _deposit_vapor_on_ice!(D, halo, ice, grid, Δt, T, ρ, ρqᵛ, p, constants, physics, self_interaction)
    n = @index(Global)
    @inbounds begin
        X = (ice.x[n], ice.y[n], ice.z[n])
        r = ice.r[n]
        𝕎 = ice.𝕎[n]
    end

    i, j, k = droplet_cell(X, grid)

    @inbounds begin
        Tᵢ = T[i, j, k]
        ρᵢ = ρ[i, j, k]
        pᵢ = p[i, j, k]
        q = MoistureMassFractions(ρqᵛ[i, j, k] / ρᵢ)
    end

    ρᵗ = pᵢ / (mixture_gas_constant(q, constants) * Tᵢ)
    𝒮ⁱ = supersaturation(Tᵢ, ρᵗ, q, constants, PlanarIceSurface())
    fᵛ = ventilation_factor(physics.capacitance_ratio * r, 𝕎, Tᵢ, pᵢ, constants, physics)
    step = depositional_growth_step(r, 𝒮ⁱ, fᵛ, Tᵢ, pᵢ, Δt, constants, physics, self_interaction)
    r⁺ = step.radius

    @inbounds begin
        ice.r[n] = r⁺
        ice.𝕎[n] = ice_terminal_velocity(r⁺, Tᵢ, pᵢ, constants, physics)
    end

    atomic_add!(D, halo, i, j, k, step.deposited_mass / Vᶜᶜᶜ(i, j, k, grid))
end

@inline deposited_vapor(::Nothing, i, j, k) = 0
@inline deposited_vapor(D, i, j, k) = @inbounds D[i, j, k]

# Remove the condensed and deposited mass densities C and D from the vapor, and warm the air at constant
# pressure by the latent heat released, cᵖᵐ ΔT = (ℒˡ C + ℒⁱ D) / ρ.
@kernel function _apply_condensation!(ρθ, ρqᵛ, C, D, T, ρ, p, pˢᵗ, constants)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Cᵢ = C[i, j, k]
        Dᵢ = deposited_vapor(D, i, j, k)
        ρᵢ = ρ[i, j, k]
        Tᵢ = T[i, j, k]
        ρqᵛ⁺ = ρqᵛ[i, j, k] - Cᵢ - Dᵢ
        q⁺ = MoistureMassFractions(ρqᵛ⁺ / ρᵢ)
        ℒˡ = liquid_latent_heat(Tᵢ, constants)
        ℒⁱ = ice_latent_heat(Tᵢ, constants)
        cᵖᵐ = mixture_heat_capacity(q⁺, constants)
        T⁺ = Tᵢ + (ℒˡ * Cᵢ + ℒⁱ * Dᵢ) / (ρᵢ * cᵖᵐ)
        θ⁺ = potential_temperature_from_temperature(T⁺, p[i, j, k], pˢᵗ, constants, q⁺)
        ρqᵛ[i, j, k] = ρqᵛ⁺
        ρθ[i, j, k] = ρᵢ * θ⁺
    end
end

validate_formulation(::LiquidIcePotentialTemperatureFormulation) = nothing

validate_formulation(formulation) =
    throw(ArgumentError("LagrangianMicrophysics requires the liquid-ice potential temperature formulation, " *
                        "but the model uses $(summary(formulation))."))

# The update always ends with `update_state!`, so time-steppers need not compute tendencies before it
AtmosphereModels.microphysics_model_update_refreshes_state(::LagrangianMicrophysics) = true

function AtmosphereModels.microphysics_model_update!(μ::LagrangianMicrophysics, model)
    Δt = model.clock.last_Δt
    (isnan(Δt) || isinf(Δt) || Δt ≤ 0) && return update_state!(model)
    validate_formulation(model.formulation)

    grid = model.grid
    arch = architecture(grid)
    droplets = μ.droplets.properties
    Nd = length(droplets)
    constants = model.thermodynamic_constants

    ρ = dynamics_density(model.dynamics)
    p = dynamics_pressure(model.dynamics)
    pˢᵗ = standard_pressure(model.dynamics)
    ρθ = model.formulation.potential_temperature_density
    ρqᵛ = model.moisture_density
    T = model.temperature
    qˡ = model.microphysical_fields.qˡ
    C = model.microphysical_fields.C
    halo = halo_size(grid)

    # Condensation at the supersaturation of each droplet's cell, without the droplet's own sink
    fill!(parent(C), 0)
    kernel_Δt = kernel_time_step(arch, grid, Δt)
    self_interaction = self_interaction_parameters(μ.self_interaction, model.closure, grid, Δt)
    Nd > 0 && launch!(arch, grid, KernelParameters(1:Nd), _condense_droplets!,
                      parent(C), halo, droplets, grid, kernel_Δt, T, ρ, ρqᵛ, p, constants, μ.physics, self_interaction)
    D = deposit_vapor_on_ice!(μ, model, kernel_Δt, self_interaction)
    launch!(arch, grid, :xyz, _apply_condensation!, ρθ, ρqᵛ, C, D, T, ρ, p, pˢᵗ, constants)

    # Diagnose the liquid mass fraction of the droplets in each cell
    fill!(parent(qˡ), 0)
    Nd > 0 && launch!(arch, grid, KernelParameters(1:Nd), _deposit_droplet_liquid!,
                      parent(qˡ), halo, droplets, grid, ρ, constants)

    diagnose_ice_mass_fraction!(μ, model)

    # Refill halos and recompute diagnostics so the post-update state is consistent for the next step
    update_state!(model)

    return nothing
end

#####
##### Ice deposition and sublimation
#####

deposit_vapor_on_ice!(μ::LagrangianMicrophysics, model, Δt, self_interaction) = nothing
diagnose_ice_mass_fraction!(μ::LagrangianMicrophysics, model) = nothing

# Deposition at the supersaturation over ice of each particle's cell, from the same state as the
# condensation on droplets, so that both see the vapor and temperature before either is applied.
# Returns the deposited mass density `D`.
function deposit_vapor_on_ice!(μ::IceLagrangianMicrophysics, model, Δt, self_interaction)
    grid = model.grid
    arch = architecture(grid)
    ice = μ.ice.properties
    D = model.microphysical_fields.D
    fill!(parent(D), 0)
    Ni = length(ice)
    Ni > 0 && launch!(arch, grid, KernelParameters(1:Ni), _deposit_vapor_on_ice!,
                      parent(D), halo_size(grid), ice, grid, Δt, model.temperature,
                      dynamics_density(model.dynamics), model.moisture_density,
                      dynamics_pressure(model.dynamics), model.thermodynamic_constants, μ.ice_physics,
                      ice_self_interaction(self_interaction))
    return D
end

# The self-interaction parameters of the grid apply to ice with the capacitance in place of the radius
ice_self_interaction(self_interaction) = self_interaction

function diagnose_ice_mass_fraction!(μ::IceLagrangianMicrophysics, model)
    grid = model.grid
    ice = μ.ice.properties
    qⁱ = model.microphysical_fields.qⁱ
    fill!(parent(qⁱ), 0)
    Ni = length(ice)
    Ni > 0 && launch!(architecture(grid), grid, KernelParameters(1:Ni), _deposit_ice_mass!,
                      parent(qⁱ), halo_size(grid), ice, grid, dynamics_density(model.dynamics), μ.ice_physics)
    return nothing
end
