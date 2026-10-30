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
    DiffusiveSelfInteraction(; grid_translation = true)

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

A droplet that moves through the grid, with the air or by settling, samples less of its own perturbation:
its sink is deposited in the cell that contains it and is then advected and diffused by the model, while the
droplet's position in the cell it later samples is shifted by its displacement relative to the grid. For a
droplet that moves by c_d = |U_d| Δt / Δ_d cells per time step in direction d, the cell value g is multiplied by
Π_d r_d(c_d), where r_d is computed from the Fourier modes of the grid's diffusion operator, the modified
wavenumber of the advection scheme of `ρqᵛ` and `ρθ`, and the droplet's sub-cell position, and tabulated
whenever κΔt / Δ² changes by more than 5%. The product falls to about 1/2 when |U| Δ / κ ≫ 1, so that a
correction for droplets at rest overcorrects droplets carried through the grid by turbulence.
With `grid_translation = false`, droplets are corrected as if they were at rest relative to the grid.

The correction requires a regular `RectilinearGrid` and a `ScalarDiffusivity` closure with constant
diffusivities for `ρθ` and `ρqᵛ`. Otherwise it is not applied.
"""
struct DiffusiveSelfInteraction{G, E, F, A, T}
     greens_function :: G # cell value g of the Green's function of -∇² (1/m)
         eigenvalues :: E # mean eigenvalue of -∇² in each bin of a histogram of the nonzero modes
      mode_fractions :: F # fraction of all modes in each bin
       amplification :: A # amplification factor of the time stepper for diffusion
         translation :: T # tables of the reduction for droplets that move through the grid
end

DiffusiveSelfInteraction(; grid_translation = true) =
    DiffusiveSelfInteraction(nothing, nothing, nothing, nothing, grid_translation)

# Only the parameters computed by `self_interaction_parameters` enter kernels
Adapt.adapt_structure(to, ::DiffusiveSelfInteraction) = nothing

"""
    LagrangianMicrophysics(FT = Oceananigans.defaults.FloatType;
                           physics = DropletPhysicsParameters(FT),
                           self_interaction = DiffusiveSelfInteraction(),
                           droplets = nothing)

Return Lagrangian microphysics in which every cloud droplet is a Lagrangian particle.

`physics` is a [`DropletPhysicsParameters`](@ref) that determines condensational growth
and the terminal velocity of droplets. `self_interaction` corrects droplet growth for each droplet's own
sink in the cell it samples ([`DiffusiveSelfInteraction`](@ref)), or is `nothing` for no correction.
`droplets` is a `NamedTuple` of vectors with the initial
positions `x`, `y`, `z`, wet radii `r`, dry radii `rᵈ`, and hygroscopicities `κ` of the droplets,
or `nothing` for no droplets. The droplets are allocated on the architecture of the grid
when the model is built, and can be accessed with `model.microphysics.droplets`.
"""
struct LagrangianMicrophysics{P, S, D}
             physics :: P
    self_interaction :: S
            droplets :: D
end

LagrangianMicrophysics(FT = Oceananigans.defaults.FloatType; physics = DropletPhysicsParameters(FT),
                       self_interaction = DiffusiveSelfInteraction(), droplets = nothing) =
    LagrangianMicrophysics(physics, self_interaction, droplets)

Base.summary(::LagrangianMicrophysics) = "LagrangianMicrophysics"

function Base.show(io::IO, μ::LagrangianMicrophysics)
    print(io, summary(μ), '\n',
              "├── self_interaction: ", self_interaction_summary(μ.self_interaction), '\n',
              "└── droplets: ", droplet_summary(μ.droplets))
end

self_interaction_summary(::Nothing) = "nothing"
self_interaction_summary(::DiffusiveSelfInteraction{Nothing}) = "DiffusiveSelfInteraction()"
self_interaction_summary(s::DiffusiveSelfInteraction) =
    string("DiffusiveSelfInteraction(g=", prettysummary(s.greens_function), ", ", nameof(typeof(s.amplification)), ")")

droplet_summary(::Nothing) = "none"
droplet_summary(droplets::LagrangianParticles) = string(length(droplets), " droplets")
droplet_summary(droplets::NamedTuple) = string(length(droplets.x), " droplets (not materialized)")

# The droplets are not used inside kernels, where only the physics parameters are needed
Adapt.adapt_structure(to, μ::LagrangianMicrophysics) =
    LagrangianMicrophysics(adapt(to, μ.physics), adapt(to, μ.self_interaction), nothing)

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
    droplets = LagrangianParticles(properties)
    droplets = materialize_lagrangian_particles(droplets, timestepper)

    self_interaction = materialize_self_interaction(μ.self_interaction, grid, timestepper)

    return LagrangianMicrophysics(μ.physics, self_interaction, droplets)
end

AtmosphereModels.materialize_microphysics(μ::LagrangianMicrophysics{<:Any, <:Any, <:LagrangianParticles}, grid, timestepper) = μ

materialize_self_interaction(::Nothing, grid, timestepper) = nothing
materialize_self_interaction(s::DiffusiveSelfInteraction{<:Number}, grid, timestepper) = s

function materialize_self_interaction(s::DiffusiveSelfInteraction{Nothing}, grid, timestepper)
    spectrum = laplacian_spectrum(grid)
    isnothing(spectrum) && return nothing
    translation = s.translation ? GridTranslationTable(grid) : nothing
    return DiffusiveSelfInteraction(spectrum.greens_function, spectrum.eigenvalues, spectrum.mode_fractions,
                                    diffusion_amplification(timestepper), translation)
end

#####
##### Droplets that move through the grid
#####

# A droplet deposits its sink in the cell that contains it at the end of every time step, and the deposit is then
# advected and diffused by the model. The droplet samples the cell that contains it after the following steps.
# Each deposit is a sum of the grid's Fourier modes; over n steps a mode is advected and diffused by the
# time stepper, while the droplet is displaced by nc cells (c = |U| Δt / Δ). Averaged over the droplet's position
# in its cell at the time of the deposit, the sampled cell is offset from the deposit cell by o = ⌊nc⌋ cells with
# probability 1 - f, and by o + 1 with probability f, where f = nc - o. In one direction, with diffusion number
# z = κ Δt / Δ², the sampled fraction of a deposit of age n is
#
#     S(n, c) = (1/N) Σₖ exp(-z λₖ n) [(1 - f) cos(k o - k̃ c n) + f cos(k (o + 1) - k̃ c n)],
#
# where λₖ = 2 - 2 cos k and k̃ is the modified wavenumber of the advection scheme. This treats the amplification
# factor of the time stepper as exp(-z λ n - i k̃ c n), which separates over directions. The cell value of the
# droplet's own perturbation, relative to that of a droplet at rest, is close to the product over directions
#
#     r_d(c) = Σₙ [S_d(n, c) Π_{e ≠ d} S_e(n, 0) - 1 / N] / Σₙ [Π_e S_e(n, 0) - 1 / N],
#
# within 0.03 of the full three-dimensional sum in tests with uniform flow (validation/ncar_dns, D11).

# The modified wavenumber k̃Δ of the advective derivative of `scheme` for the wavenumber kΔ. Centered schemes of
# order 2, 4 and 6 have no dissipation; other schemes are treated as exact (k̃ = k).
# TODO: include the dissipation of upwind-biased schemes (a complex k̃).
modified_wavenumber(scheme, k) = k
modified_wavenumber(::Centered{1}, k) = sin(k)
modified_wavenumber(::Centered{2}, k) = (8 * sin(k) - sin(2k)) / 6
modified_wavenumber(::Centered{3}, k) = (45 * sin(k) - 9 * sin(2k) + sin(3k)) / 30

# The sampled fraction S(n, c) of a deposit of age n (in time steps) on a periodic lattice of N cells.
# The terms of k and -k are equal.
function sampled_deposit_fraction(N, z, c, n, scheme)
    m = c * n
    o = floor(m)
    f = m - o
    S = 0.0
    for j in 0:N÷2
        k = 2π * j / N
        k̃ = modified_wavenumber(scheme, k)
        multiplicity = (j == 0 || 2j == N) ? 1 : 2
        S += multiplicity * exp(-z * (2 - 2cos(k)) * n) * ((1 - f) * cos(k * o - k̃ * m) + f * cos(k * (o + 1) - k̃ * m))
    end
    return S / N
end

# Ages (in time steps) and weights of the sums over n: every step up to n₀, then Q logarithmically spaced intervals
# up to a diffusive time of 30 cells², each sampled at P points
function deposit_ages(z; n₀ = 32, Q = 24, P = 3)
    ages = collect(1.0:n₀)
    weights = ones(n₀)
    edges = exp.(range(log(n₀ + 0.5), log(n₀ + 0.5 + 30 / z), length = Q + 1))
    for q in 1:Q
        a, b = edges[q], edges[q + 1]
        for p in 1:P
            push!(ages, a + (p - 1/2) * (b - a) / P)
            push!(weights, (b - a) / P)
        end
    end
    return ages, weights
end

# Courant numbers c = |U| Δt / Δ of the table, quadratically spaced up to 1
const translation_courant_numbers = Tuple(range(0, 1, length = 33) .^ 2)
const translation_table_size = length(translation_courant_numbers)

# Tables of the factors r_d(c) by which the cell value of a droplet's own perturbation is reduced when the droplet
# moves by c cells per time step in direction d, for vapor and for heat, at the diffusion numbers for which they
# were last computed
mutable struct GridTranslationTable{T, H}
                  table :: T # factors r_d(c) on the architecture of the grid, (c, d, vapor or heat)
                   host :: H # the same, on the CPU
    diffusion_numbers :: NTuple{2, Float64} # κᵛ Δt / Δx² and κᵀ Δt / Δx² of the table
end

function GridTranslationTable(grid)
    FT = eltype(grid)
    host = ones(FT, translation_table_size, 3, 2)
    table = on_architecture(architecture(grid), host)
    return GridTranslationTable(table, host, (NaN, NaN))
end

# Lattices of at most 32 cells in each direction suffice: the droplet's own perturbation is local
surrogate_lattice_size(N) = min(N, 32)

function compute_translation_factors!(factors, κ, Δt, spacings, sizes, scheme)
    z = ntuple(d -> κ * Δt / spacings[d]^2, 3)
    N = map(surrogate_lattice_size, sizes)
    ages, weights = deposit_ages(minimum(z))
    S⁰ = [sampled_deposit_fraction(N[d], z[d], 0.0, n, scheme) for n in ages, d in 1:3]
    mean_mode = 1 / prod(N)
    at_rest = sum(weights[a] * (S⁰[a, 1] * S⁰[a, 2] * S⁰[a, 3] - mean_mode) for a in eachindex(ages))
    for d in 1:3, (i, c) in enumerate(translation_courant_numbers)
        moving = zero(at_rest)
        for a in eachindex(ages)
            others = prod(S⁰[a, e] for e in 1:3 if e != d)
            moving += weights[a] * (sampled_deposit_fraction(N[d], z[d], c, ages[a], scheme) * others - mean_mode)
        end
        factors[i, d] = moving / at_rest
    end
    return factors
end

# Recompute the tables when the diffusion numbers change by more than 5%
function update_translation_table!(t::GridTranslationTable, κ, Δt, grid, schemes)
    spacings = (Float64(grid.Δxᶜᵃᵃ), Float64(grid.Δyᵃᶜᵃ), Float64(grid.z.Δᵃᵃᶜ))
    z = (κ.κᵛ * Δt / spacings[1]^2, κ.κᵀ * Δt / spacings[1]^2)
    current = all(abs(z[i] - t.diffusion_numbers[i]) ≤ 0.05 * z[i] for i in 1:2)
    current && return nothing
    compute_translation_factors!(view(t.host, :, :, 1), κ.κᵛ, Δt, spacings, size(grid), schemes.vapor)
    compute_translation_factors!(view(t.host, :, :, 2), κ.κᵀ, Δt, spacings, size(grid), schemes.heat)
    copyto!(t.table, t.host)
    t.diffusion_numbers = z
    return nothing
end

update_translation_table!(::Nothing, κ, Δt, grid, schemes) = nothing

# The factor r(c) interpolated from the table
@inline function translation_factor(table, c, d, field)
    ξ = sqrt(min(c, 1)) * (translation_table_size - 1)
    i = min(floor(Int, ξ), translation_table_size - 2)
    w = ξ - i
    @inbounds return (1 - w) * table[i + 1, d, field] + w * table[i + 2, d, field]
end

# The self-interaction parameters of a droplet that moves at velocity U relative to the grid
@inline moving_self_interaction(s, U) = s # without grid translation

@inline function moving_self_interaction(s::NamedTuple{(:vapor_greens_function, :thermal_greens_function,
                                                        :vapor_diffusivity, :thermal_diffusivity,
                                                        :translation, :courant_scales)}, U)
    c₁ = abs(U[1]) * s.courant_scales[1]
    c₂ = abs(U[2]) * s.courant_scales[2]
    c₃ = abs(U[3]) * s.courant_scales[3]
    t = s.translation
    rᵛ = translation_factor(t, c₁, 1, 1) * translation_factor(t, c₂, 2, 1) * translation_factor(t, c₃, 3, 1)
    rᵀ = translation_factor(t, c₁, 1, 2) * translation_factor(t, c₂, 2, 2) * translation_factor(t, c₃, 3, 2)
    return (vapor_greens_function = rᵛ * s.vapor_greens_function, thermal_greens_function = rᵀ * s.thermal_greens_function,
            vapor_diffusivity = s.vapor_diffusivity, thermal_diffusivity = s.thermal_diffusivity)
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
    parameters = (vapor_greens_function = convert(FT, gᵛ), thermal_greens_function = convert(FT, gᵀ),
                  vapor_diffusivity = convert(FT, κ.κᵛ), thermal_diffusivity = convert(FT, κ.κᵀ))
    return with_translation(parameters, s.translation, grid, Δt)
end

with_translation(parameters, ::Nothing, grid, Δt) = parameters

function with_translation(parameters, translation, grid, Δt)
    FT = eltype(grid)
    courant_scales = convert.(FT, (Δt / grid.Δxᶜᵃᵃ, Δt / grid.Δyᵃᶜᵃ, Δt / grid.z.Δᵃᵃᶜ))
    return merge(parameters, (translation = translation.table, courant_scales))
end

update_self_interaction!(self_interaction, closure, grid, Δt, advection) = nothing

function update_self_interaction!(s::DiffusiveSelfInteraction, closure, grid, Δt, advection)
    κ = molecular_diffusivities(closure)
    isnothing(κ) && return nothing
    schemes = (vapor = advection.ρqᵛ, heat = advection.ρθ)
    return update_translation_table!(s.translation, κ, Float64(Δt), grid, schemes)
end

initial_droplets(::Nothing, FT) = (x = FT[], y = FT[], z = FT[], r = FT[], rᵈ = FT[], κ = FT[])
initial_droplets(droplets::NamedTuple, FT) = droplets

AtmosphereModels.microphysics_particles(μ::LagrangianMicrophysics) = μ.droplets

# Droplets fall relative to the air at their terminal velocity
@inline LagrangianParticleTracking.particle_w_velocity(droplets::AbstractVector{<:Droplet}, p, w) =
    @inbounds w - droplets.𝕎[p]

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

@kernel function _condense_droplets!(C, halo, droplets, grid, Δt, T, ρ, ρqᵛ, p, velocities, constants, physics,
                                     self_interaction)
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

    # The droplet's velocity relative to the grid: the air velocity in its cell, less its terminal velocity
    U = (ℑxᶜᵃᵃ(i, j, k, grid, velocities.u), ℑyᵃᶜᵃ(i, j, k, grid, velocities.v), ℑzᵃᵃᶜ(i, j, k, grid, velocities.w) - 𝕎)
    droplet_self_interaction = moving_self_interaction(self_interaction, U)
    step = condensational_growth_step(r, rᵈ, κ, 𝒮, fᵛ, Tᵢ, pᵢ, Δt, constants, physics, droplet_self_interaction)
    r⁺ = step.radius

    @inbounds begin
        droplets.r[n] = r⁺
        droplets.𝕎[n] = droplet_terminal_velocity(physics.terminal_velocity, r⁺, Tᵢ, pᵢ, constants, physics)
    end

    atomic_add!(C, halo, i, j, k, step.condensed_mass / Vᶜᶜᶜ(i, j, k, grid))
end

# Remove the condensed mass density C from the vapor, and warm the air at constant pressure
# by the latent heat released, cᵖᵐ ΔT = ℒˡ C / ρ.
@kernel function _apply_condensation!(ρθ, ρqᵛ, C, T, ρ, p, pˢᵗ, constants)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Cᵢ = C[i, j, k]
        ρᵢ = ρ[i, j, k]
        Tᵢ = T[i, j, k]
        ρqᵛ⁺ = ρqᵛ[i, j, k] - Cᵢ
        q⁺ = MoistureMassFractions(ρqᵛ⁺ / ρᵢ)
        ℒˡ = liquid_latent_heat(Tᵢ, constants)
        cᵖᵐ = mixture_heat_capacity(q⁺, constants)
        T⁺ = Tᵢ + ℒˡ * Cᵢ / (ρᵢ * cᵖᵐ)
        θ⁺ = potential_temperature_from_temperature(T⁺, p[i, j, k], pˢᵗ, constants, q⁺)
        ρqᵛ[i, j, k] = ρqᵛ⁺
        ρθ[i, j, k] = ρᵢ * θ⁺
    end
end

validate_formulation(::LiquidIcePotentialTemperatureFormulation) = nothing

validate_formulation(formulation) =
    throw(ArgumentError("LagrangianMicrophysics requires the liquid-ice potential temperature formulation, " *
                        "but the model uses $(summary(formulation))."))

function AtmosphereModels.microphysics_model_update!(μ::LagrangianMicrophysics, model)
    Δt = model.clock.last_Δt
    (isnan(Δt) || isinf(Δt) || Δt ≤ 0) && return nothing
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
    update_self_interaction!(μ.self_interaction, model.closure, grid, Δt, model.advection)
    self_interaction = self_interaction_parameters(μ.self_interaction, model.closure, grid, Δt)
    Nd > 0 && launch!(arch, grid, KernelParameters(1:Nd), _condense_droplets!,
                      parent(C), halo, droplets, grid, kernel_Δt, T, ρ, ρqᵛ, p, model.velocities, constants, μ.physics,
                      self_interaction)
    launch!(arch, grid, :xyz, _apply_condensation!, ρθ, ρqᵛ, C, T, ρ, p, pˢᵗ, constants)

    # Diagnose the liquid mass fraction of the droplets in each cell
    fill!(parent(qˡ), 0)
    Nd > 0 && launch!(arch, grid, KernelParameters(1:Nd), _deposit_droplet_liquid!,
                      parent(qˡ), halo, droplets, grid, ρ, constants)

    # Refill halos and recompute diagnostics so the post-update state is consistent for the next step
    update_state!(model)

    return nothing
end
