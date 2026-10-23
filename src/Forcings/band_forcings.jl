using ..AtmosphereModels: AtmosphereModels
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.Grids: Periodic, RectilinearGrid, xnodes, ynodes, znodes
using Oceananigans.Utils: launch!, prettysummary
using KernelAbstractions: @kernel, @index
using Adapt: Adapt

#####
##### Forcing of homogeneous turbulence in a band of low wavenumbers
#####
##### A band forcing acts on the Fourier modes of the velocity whose wavevectors satisfy
##### k_min ≤ |k| ≤ k_max in a triply-periodic domain. Because the band contains few modes,
##### the Fourier coefficients of the velocity are computed by partial discrete Fourier
##### transforms (sums over the grid, one direction at a time), and the forcing is evaluated
##### in physical space as a sum over the forced modes, without three-dimensional FFTs.
#####
##### The forced modes are stored once per conjugate pair, with wavevectors in the half-space
##### nx > 0, or nx = 0 and ny > 0, or nx = ny = 0 and nz > 0. A real field f with Fourier
##### coefficients f̂ₘ in the band is then
#####
#####     f(x) = 2 Re Σₘ f̂ₘ exp(i kₘ ⋅ x),
#####
##### where the sum runs over the half-space modes.
#####

#####
##### Wavenumber band and its tables
#####

"""
$(TYPEDSIGNATURES)

Return the integer wavevectors ``n = (nx, ny, nz)``, one for each conjugate pair, whose
physical wavenumbers ``k = 2π (nx / Lx, ny / Ly, nz / Lz)`` satisfy
`minimum_wavenumber ≤ |k| ≤ maximum_wavenumber` and ``|k| > 0``.
"""
function band_wavevectors(Lx, Ly, Lz, minimum_wavenumber, maximum_wavenumber)
    # A relative tolerance keeps the band from depending on the rounding of the domain size,
    # such as Lx = 2π in single precision.
    δ = sqrt(eps(Float32))
    k_min = minimum_wavenumber * (1 - δ)
    k_max = maximum_wavenumber * (1 + δ)

    Kx = floor(Int, k_max * Lx / 2π)
    Ky = floor(Int, k_max * Ly / 2π)
    Kz = floor(Int, k_max * Lz / 2π)

    wavevectors = NTuple{3, Int}[]

    for nx in 0:Kx, ny in -Ky:Ky, nz in -Kz:Kz
        in_half_space = (nx > 0) | ((nx == 0) & (ny > 0)) | ((nx == 0) & (ny == 0) & (nz > 0))
        k = sqrt((2π * nx / Lx)^2 + (2π * ny / Ly)^2 + (2π * nz / Lz)^2)
        in_band = (k ≥ k_min) & (k ≤ k_max)
        in_half_space & in_band && push!(wavevectors, (nx, ny, nz))
    end

    return wavevectors, (Kx, Ky, Kz)
end

# Phases exp(i 2π n x / L) for n = n₀, n₀ + 1, …, n₀ + Q - 1, tabulated at the `nodes` of one direction.
function phase_table(arch, FT, nodes, L, wavenumber_range)
    table = [exp(im * 2π * n * x / L) for x in nodes, n in wavenumber_range]
    return on_architecture(arch, Complex{FT}.(table))
end

# The modified wavenumber k̃ = (2 / Δ) sin(k Δ / 2) of the second-order difference between
# faces and centers, which is the wavenumber "seen" by the discrete divergence.
modified_wavenumber(n, L, N) = 2 * N / L * sin(π * n / N)

struct ForcedWavenumberBand{FT, W, I, K, P}
    minimum_wavenumber :: FT
    maximum_wavenumber :: FT
    wavevectors :: W           # host vector of integer wavevectors
    maximum_wavevector :: NTuple{3, Int}
    mode_indices :: I          # M × 3 indices into the phase tables
    modified_wavenumbers :: K  # M × 3 modified wavenumbers
    phases :: P                # phase tables at faces and centers in each direction
end

Adapt.adapt_structure(to, band::ForcedWavenumberBand) =
    ForcedWavenumberBand(band.minimum_wavenumber,
                         band.maximum_wavenumber,
                         nothing,
                         band.maximum_wavevector,
                         Adapt.adapt(to, band.mode_indices),
                         Adapt.adapt(to, band.modified_wavenumbers),
                         Adapt.adapt(to, band.phases))

const TriplyPeriodicGrid = RectilinearGrid{<:Any, Periodic, Periodic, Periodic}

validate_band_forcing_grid(grid) =
    throw(ArgumentError("Band forcings require a triply-periodic RectilinearGrid, got $(summary(grid))."))

function validate_band_forcing_grid(grid::TriplyPeriodicGrid)
    regular = (grid.Δxᶠᵃᵃ isa Number) & (grid.Δyᵃᶠᵃ isa Number) & (grid.z.Δᵃᵃᶠ isa Number)
    regular || throw(ArgumentError("Band forcings require a grid with uniform spacing in every direction."))
    return nothing
end

default_maximum_wavenumber(grid) = 3π / max(grid.Lx, grid.Ly, grid.Lz)

function ForcedWavenumberBand(grid; minimum_wavenumber, maximum_wavenumber)
    validate_band_forcing_grid(grid)

    FT = eltype(grid)
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)
    Lx, Ly, Lz = grid.Lx, grid.Ly, grid.Lz

    wavevectors, (Kx, Ky, Kz) = band_wavevectors(Lx, Ly, Lz, minimum_wavenumber, maximum_wavenumber)

    if isempty(wavevectors)
        throw(ArgumentError("No Fourier modes satisfy $minimum_wavenumber ≤ |k| ≤ $maximum_wavenumber " *
                            "in a domain of size ($Lx, $Ly, $Lz)."))
    end

    if (2Kx ≥ Nx) | (2Ky ≥ Ny) | (2Kz ≥ Nz)
        throw(ArgumentError("The forced band extends to wavevector ($Kx, $Ky, $Kz), which is not resolved " *
                            "by a grid of size ($Nx, $Ny, $Nz)."))
    end

    # Indices into the phase tables, which hold nx = 0:Kx, ny = -Ky:Ky, and nz = -Kz:Kz.
    mode_indices = hcat([n[1] + 1 for n in wavevectors],
                        [n[2] + Ky + 1 for n in wavevectors],
                        [n[3] + Kz + 1 for n in wavevectors])

    modified_wavenumbers = hcat([modified_wavenumber(n[1], Lx, Nx) for n in wavevectors],
                                [modified_wavenumber(n[2], Ly, Ny) for n in wavevectors],
                                [modified_wavenumber(n[3], Lz, Nz) for n in wavevectors])

    phases = (xᶠ = phase_table(arch, FT, xnodes(grid, Face()),   Lx, 0:Kx),
              xᶜ = phase_table(arch, FT, xnodes(grid, Center()), Lx, 0:Kx),
              yᶠ = phase_table(arch, FT, ynodes(grid, Face()),   Ly, -Ky:Ky),
              yᶜ = phase_table(arch, FT, ynodes(grid, Center()), Ly, -Ky:Ky),
              zᶠ = phase_table(arch, FT, znodes(grid, Face()),   Lz, -Kz:Kz),
              zᶜ = phase_table(arch, FT, znodes(grid, Center()), Lz, -Kz:Kz))

    return ForcedWavenumberBand(convert(FT, minimum_wavenumber),
                                convert(FT, maximum_wavenumber),
                                wavevectors,
                                (Kx, Ky, Kz),
                                on_architecture(arch, mode_indices),
                                on_architecture(arch, FT.(modified_wavenumbers)),
                                phases)
end

number_of_modes(band::ForcedWavenumberBand) = length(band.wavevectors)

# Phase tables at the location of velocity component `c` = 1, 2, 3 (u, v, w) on the C-grid.
component_phases(band, c) = component_phases(band.phases, Val(c))
component_phases(phases, ::Val{1}) = (phases.xᶠ, phases.yᶜ, phases.zᶜ)
component_phases(phases, ::Val{2}) = (phases.xᶜ, phases.yᶠ, phases.zᶜ)
component_phases(phases, ::Val{3}) = (phases.xᶜ, phases.yᶜ, phases.zᶠ)

band_summary(band) = string(number_of_modes(band), " modes with ",
                            prettysummary(band.minimum_wavenumber), " ≤ |k| ≤ ",
                            prettysummary(band.maximum_wavenumber))

#####
##### Transverse projection with the modified wavenumber
#####

# Remove the part of the complex 3-vector `a` along the real vector `k`, so that a Fourier mode
# with amplitude P(k) a is discretely divergence-free on the C-grid when `k` is the modified wavenumber.
@inline function transverse_projection(a, k)
    k² = k[1]^2 + k[2]^2 + k[3]^2
    k_dot_a = k[1] * a[1] + k[2] * a[2] + k[3] * a[3]
    return (a[1] - k[1] * k_dot_a / k²,
            a[2] - k[2] * k_dot_a / k²,
            a[3] - k[3] * k_dot_a / k²)
end

#####
##### Kernel-callable component of a band forcing
#####

"""
The forcing of velocity component `component` (1, 2, 3 for `u`, `v`, `w`) by a band forcing `scheme`.
Its kernel function evaluates

```math
F(x, y, z) = 2 \\, \\mathrm{Re} \\sum_m A_m \\exp(i \\boldsymbol{k}_m ⋅ \\boldsymbol{x})
           = 2 \\, \\mathrm{Re} \\sum_{n_x} \\exp(i k_x x) \\, G(y, z, n_x) ,
```

at the location of the velocity component, where ``A_m`` are the forcing amplitudes of the
modes in the band and ``G(y, z, n_x) = \\sum_m A_m \\exp(i k_y y + i k_z z)``, summed over the modes
with ``x``-wavenumber ``n_x``, is computed by `scheme` before every Runge-Kutta stage. Because the
forced modes have only a few ``x``-wavenumbers, the forcing costs a few operations per grid point.
"""
struct BandForcingComponent{S, X, G}
    scheme :: S
    component :: Int
    x_phases :: X
    yz_amplitudes :: G
end

Adapt.adapt_structure(to, forcing::BandForcingComponent) =
    BandForcingComponent(nothing,
                         forcing.component,
                         Adapt.adapt(to, forcing.x_phases),
                         Adapt.adapt(to, forcing.yz_amplitudes))

@inline function (forcing::BandForcingComponent)(i, j, k, grid, clock, fields)
    X = forcing.x_phases
    G = forcing.yz_amplitudes
    c = forcing.component

    F = zero(real(eltype(G)))

    for q in 1:size(G, 3)
        @inbounds F += real(X[i, q] * G[j, k, q, c])
    end

    return 2F
end

# G[j, k, nx, c] = Σₘ Aₘ exp(i ky yⱼ + i kz zₖ) over the modes m with x-wavenumber nx
@kernel function _compute_yz_amplitudes!(G, A, q, Y, Z, c)
    j, k, qx = @index(Global, NTuple)
    s = zero(eltype(G))
    for m in 1:size(A, 1)
        @inbounds s += ifelse(q[m, 1] == qx, A[m, c] * Y[j, q[m, 2]] * Z[k, q[m, 3]], zero(s))
    end
    @inbounds G[j, k, qx, c] = s
end

yz_amplitudes_array(grid, band) =
    on_architecture(architecture(grid), zeros(Complex{eltype(grid)}, size(grid, 2), size(grid, 3),
                                              band.maximum_wavevector[1] + 1, 3))

function compute_yz_amplitudes!(scheme)
    grid = scheme.grid
    band = scheme.band
    G = scheme.yz_amplitudes
    for c in 1:3
        X, Y, Z = component_phases(band, c)
        launch!(architecture(grid), grid, size(G)[1:3], _compute_yz_amplitudes!,
                G, scheme.amplitudes, band.mode_indices, Y, Z, c)
    end
    return nothing
end

velocity_component_index(name) = velocity_component_index(Val(name))
velocity_component_index(::Val{:u}) = 1
velocity_component_index(::Val{:v}) = 2
velocity_component_index(::Val{:w}) = 3

function velocity_component_index(::Val{name}) where name
    msg = string("Band forcings act on velocity and must be supplied under the velocity names ",
                 "`u`, `v`, and `w`, got `", name, "`. Breeze applies the density factor ρ ",
                 "automatically, so that the forcing of ρu is ρ times the forcing of u.")
    throw(ArgumentError(msg))
end

const velocity_component_names = (:u, :v, :w)

function materialize_band_forcing_component(scheme, name)
    c = velocity_component_index(name)
    register_forced_component!(scheme, c)
    X, Y, Z = component_phases(scheme.band, c)
    return BandForcingComponent(scheme, c, X, scheme.yz_amplitudes)
end

AtmosphereModels.compute_forcing!(forcing::BandForcingComponent, model) =
    compute_band_forcing!(forcing.scheme, forcing.component, model)

register_forced_component!(scheme, c) = nothing

Base.summary(forcing::BandForcingComponent) =
    string("BandForcingComponent(", velocity_component_names[forcing.component], ") of ", summary(forcing.scheme))

Base.show(io::IO, forcing::BandForcingComponent) = print(io, summary(forcing))

#####
##### Band Fourier coefficients of the velocity by partial transforms
#####

@inline add_conjugate_phases(s::NTuple{Q}, φ, Z, k) where Q = ntuple(q -> @inbounds(s[q] + φ * conj(Z[k, q])), Val(Q))

# Σₖ φ[i, j, k] exp(-i kz zₖ) for each of the Q values of nz, reading the column of φ once
@kernel function _sum_over_z!(Sᶻ, φ, Z, Nz, ::Val{Q}) where Q
    i, j = @index(Global, NTuple)
    s = ntuple(q -> zero(eltype(Sᶻ)), Val(Q))
    for k in 1:Nz
        s = @inbounds add_conjugate_phases(s, φ[i, j, k], Z, k)
    end
    for q in 1:Q
        @inbounds Sᶻ[i, j, q] = s[q]
    end
end

# Σⱼ Sᶻ[i, j, nz] exp(-i ky yⱼ) for each (ny, nz)
@kernel function _sum_over_y!(Sʸᶻ, Sᶻ, Y, Ny)
    i, qy, qz = @index(Global, NTuple)
    s = zero(eltype(Sʸᶻ))
    for j in 1:Ny
        @inbounds s += Sᶻ[i, j, qz] * conj(Y[j, qy])
    end
    @inbounds Sʸᶻ[i, qy, qz] = s
end

# Σᵢ Sʸᶻ[i, ny, nz] exp(-i kx xᵢ) / N for each (nx, ny, nz)
@kernel function _sum_over_x!(û, Sʸᶻ, X, Nx, N, c)
    qx, qy, qz = @index(Global, NTuple)
    s = zero(eltype(û))
    for i in 1:Nx
        @inbounds s += Sʸᶻ[i, qy, qz] * conj(X[i, qx])
    end
    @inbounds û[qx, qy, qz, c] = s / N
end

struct PartialTransformWorkspace{C, Z, YZ}
    coefficients :: C  # Fourier coefficients of (u, v, w) for nx = 0:Kx, ny = -Ky:Ky, nz = -Kz:Kz
    z_sums :: Z        # Nx × Ny × (2Kz + 1)
    yz_sums :: YZ      # Nx × (2Ky + 1) × (2Kz + 1)
end

function PartialTransformWorkspace(grid, band)
    FT = eltype(grid)
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)
    Kx, Ky, Kz = band.maximum_wavevector
    Qx, Qy, Qz = Kx + 1, 2Ky + 1, 2Kz + 1
    coefficients = on_architecture(arch, zeros(Complex{FT}, Qx, Qy, Qz, 3))
    z_sums = on_architecture(arch, zeros(Complex{FT}, Nx, Ny, Qz))
    yz_sums = on_architecture(arch, zeros(Complex{FT}, Nx, Qy, Qz))
    return PartialTransformWorkspace(coefficients, z_sums, yz_sums)
end

"""
$(TYPEDSIGNATURES)

Compute the Fourier coefficients of velocity component `c` (1, 2, 3 for `u`, `v`, `w`), stored in `φ`,
for every wavevector in the box that contains the `band`, by summing over ``z``, then ``y``, then ``x``.
"""
function compute_band_coefficients!(workspace, band, grid, c, φ)
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)
    Kx, Ky, Kz = band.maximum_wavevector
    Qx, Qy, Qz = Kx + 1, 2Ky + 1, 2Kz + 1
    X, Y, Z = component_phases(band, c)

    launch!(arch, grid, (Nx, Ny), _sum_over_z!, workspace.z_sums, φ, Z, Nz, Val(Qz))
    launch!(arch, grid, (Nx, Qy, Qz), _sum_over_y!, workspace.yz_sums, workspace.z_sums, Y, Ny)
    launch!(arch, grid, (Qx, Qy, Qz), _sum_over_x!, workspace.coefficients, workspace.yz_sums, X, Nx, Nx * Ny * Nz, c)

    return nothing
end

#####
##### ConstantPowerBandForcing
#####

mutable struct ForcedComponents
    forced :: Int   # bit c is set if velocity component c is forced
    updated :: Int  # bit c is set if the coefficients of component c are current
end

component_bit(c) = 1 << (c - 1)

struct ConstantPowerBandForcing{FT, G, B, W, A, YZ}
    energy_injection_rate :: FT
    grid :: G
    band :: B
    workspace :: W
    amplitudes :: A
    yz_amplitudes :: YZ
    components :: ForcedComponents
end

"""
$(TYPEDSIGNATURES)

Return a deterministic forcing of homogeneous turbulence on the triply-periodic `grid`
that injects kinetic energy at the constant rate `energy_injection_rate` ``ε`` by
amplifying the Fourier modes of the velocity in the band
`minimum_wavenumber` ``≤ |k| ≤`` `maximum_wavenumber`, without changing their phases.

The forcing of the velocity is

```math
\\boldsymbol{F} = \\frac{ε}{2 E_f} \\, \\boldsymbol{u}_f , \\qquad E_f = \\frac{1}{2} ⟨|\\boldsymbol{u}_f|^2⟩ ,
```

where ``\\boldsymbol{u}_f`` is the part of the velocity in the forced band, projected onto the
plane perpendicular to the modified wavenumber of each mode so that ``\\boldsymbol{F}`` is
discretely divergence-free, and ``⟨⋅⟩`` is the domain average. The power input
``⟨\\boldsymbol{u} ⋅ \\boldsymbol{F}⟩`` is then exactly ``ε``.
In continuous time this is the forcing of the NCAR parcel-DNS code of
[Chen et al. (2016)](@cite Chen2016), which, after every time step, rescales the amplitudes of
the forced modes so that their energy grows by ``ε`` times the time step. Breeze computes
``\\boldsymbol{u}_f`` and ``E_f`` at every Runge-Kutta stage, so the power input is ``ε`` at
every stage, the forcing work per time step is ``ε Δt``, and the forcing does not depend on the time step.
The fluid must not be at rest: the forcing vanishes when ``E_f = 0``.

The default `maximum_wavenumber` is ``1.5 × 2π / L``, where ``L`` is the longest side of the domain:
in a cubic domain, the 9 pairs of modes with ``|n|² = 1`` or ``2``.

Supply the forcing under the velocity names `u`, `v`, and `w`;
Breeze multiplies it by the reference density to force the momentum.
A band forcing holds state that belongs to one model, so construct one for each model.

Example
=======

```jldoctest
using Breeze

grid = RectilinearGrid(size=(16, 16, 16), extent=(2π, 2π, 2π), topology=(Periodic, Periodic, Periodic))
forcing = ConstantPowerBandForcing(grid; energy_injection_rate=0.1)
model = AtmosphereModel(grid; dynamics=BoussinesqDynamics(grid), forcing=(u=forcing, v=forcing, w=forcing))
forcing

# output
ConstantPowerBandForcing(ε=0.1) with 9 modes with 0.0 ≤ |k| ≤ 1.5
```
"""
function ConstantPowerBandForcing(grid;
                                  energy_injection_rate,
                                  minimum_wavenumber = 0,
                                  maximum_wavenumber = default_maximum_wavenumber(grid))

    FT = eltype(grid)
    band = ForcedWavenumberBand(grid; minimum_wavenumber, maximum_wavenumber)
    workspace = PartialTransformWorkspace(grid, band)
    amplitudes = on_architecture(architecture(grid), zeros(Complex{FT}, number_of_modes(band), 3))

    return ConstantPowerBandForcing(convert(FT, energy_injection_rate), grid, band, workspace,
                                    amplitudes, yz_amplitudes_array(grid, band), ForcedComponents(0, 0))
end

Base.summary(forcing::ConstantPowerBandForcing) =
    string("ConstantPowerBandForcing(ε=", prettysummary(forcing.energy_injection_rate), ")")

Base.show(io::IO, forcing::ConstantPowerBandForcing) =
    print(io, summary(forcing), " with ", band_summary(forcing.band))

AtmosphereModels.materialize_atmosphere_model_forcing(forcing::ConstantPowerBandForcing, field, name,
                                                      model_field_names, context::NamedTuple) =
    materialize_band_forcing_component(forcing, name)

function register_forced_component!(forcing::ConstantPowerBandForcing, c)
    forcing.components.forced |= component_bit(c)
    return nothing
end

# Each forced component computes the Fourier coefficients of its own velocity component.
# Once the coefficients of every forced component are current, the amplitudes are computed
# from all of them, because the projection and the band energy couple the components.
function compute_band_forcing!(forcing::ConstantPowerBandForcing, c, model)
    velocity = model.velocities[c]
    compute_band_coefficients!(forcing.workspace, forcing.band, forcing.grid, c, velocity)

    components = forcing.components
    components.updated |= component_bit(c)

    if components.updated == components.forced
        compute_constant_power_amplitudes!(forcing)
        components.updated = 0
    end

    return nothing
end

function compute_constant_power_amplitudes!(forcing::ConstantPowerBandForcing)
    grid = forcing.grid
    band = forcing.band
    launch!(architecture(grid), grid, (1,), _compute_constant_power_amplitudes!,
            forcing.amplitudes, forcing.workspace.coefficients, band.mode_indices,
            band.modified_wavenumbers, forcing.energy_injection_rate)
    compute_yz_amplitudes!(forcing)
    return nothing
end

# A single work item: the band contains few modes.
@kernel function _compute_constant_power_amplitudes!(A, û, q, k̃, ε)
    _ = @index(Global)
    E = zero(real(eltype(A)))

    for m in 1:size(A, 1)
        @inbounds begin
            n = (q[m, 1], q[m, 2], q[m, 3])
            ûₘ = (û[n[1], n[2], n[3], 1], û[n[1], n[2], n[3], 2], û[n[1], n[2], n[3], 3])
            Pûₘ = transverse_projection(ûₘ, (k̃[m, 1], k̃[m, 2], k̃[m, 3]))
            A[m, 1] = Pûₘ[1]
            A[m, 2] = Pûₘ[2]
            A[m, 3] = Pûₘ[3]
        end
        E += abs2(Pûₘ[1]) + abs2(Pûₘ[2]) + abs2(Pûₘ[3])
    end

    # E = Σₘ |P ûₘ|² over half-space modes is the band kinetic energy ½ ⟨|u_f|²⟩
    α = ifelse(E > 0, ε / 2E, zero(E))

    for m in 1:size(A, 1)
        @inbounds begin
            A[m, 1] *= α
            A[m, 2] *= α
            A[m, 3] *= α
        end
    end
end

#####
##### StochasticBandForcing
#####

# A counter-based random number generator: the SplitMix64 mixing function of the seed and a counter,
# so that the sequence is reproducible and independent of the order in which amplitudes are drawn.
@inline function splitmix64(x::UInt64)
    z = x + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

unit_interval(x::UInt64) = ((x >> 11) + 0.5) * exp2(-53)

# A complex number whose real and imaginary parts are independent standard normal variates (Box-Muller).
function complex_normal(seed::UInt64, counter::UInt64)
    a = splitmix64(splitmix64(seed) + counter)
    b = splitmix64(a)
    r = sqrt(-2 * log(unit_interval(a)))
    θ = 2π * unit_interval(b)
    return complex(r * cos(θ), r * sin(θ))
end

mutable struct OrnsteinUhlenbeckState{T}
    amplitudes :: Matrix{ComplexF64}  # M × 3, before projection
    time :: T
    iteration :: Int
    initialized :: Bool
    counter :: UInt64
end

struct StochasticBandForcing{FT, G, B, A, YZ, S}
    energy_injection_rate :: FT
    correlation_time :: FT
    grid :: G
    band :: B
    amplitudes :: A
    yz_amplitudes :: YZ
    state :: S
    seed :: UInt64
end

"""
$(TYPEDSIGNATURES)

Return a stochastic forcing of homogeneous turbulence on the triply-periodic `grid`, following
[Eswaran and Pope (1988)](@cite EswaranPope1988). The forcing of the velocity is

```math
\\boldsymbol{F}(\\boldsymbol{x}, t) = 2 \\, \\mathrm{Re} \\sum_m P(k_m) \\, \\boldsymbol{b}_m(t) \\exp(i k_m ⋅ \\boldsymbol{x}) ,
```

where the sum runs over one of each pair of modes in the band
`minimum_wavenumber` ``≤ |k| ≤`` `maximum_wavenumber`, ``P(k_m)`` projects onto the plane
perpendicular to the modified wavenumber of each mode so that the forcing is discretely
divergence-free, and the components of ``\\boldsymbol{b}_m`` are independent complex
Ornstein-Uhlenbeck processes with correlation time ``T_f`` (`correlation_time`).
Their variance is chosen so that ``⟨|\\boldsymbol{F}|²⟩ \\, T_f = ε`` (`energy_injection_rate`),
which is the power input when ``T_f`` is short compared with the time scale of the forced
eddies; [Eswaran and Pope (1988)](@cite EswaranPope1988) give the correction for longer ``T_f``.

The amplitudes are advanced once per time step, on the host, with the exact Ornstein-Uhlenbeck
transition over the elapsed time, and are held fixed over the Runge-Kutta stages of the step.
The random numbers come from a counter-based generator seeded with `seed`, so a simulation is reproducible.

The default `maximum_wavenumber` is ``1.5 × 2π / L``, where ``L`` is the longest side of the domain.
Supply the forcing under the velocity names `u`, `v`, and `w`.
A band forcing holds state that belongs to one model, so construct one for each model.
The amplitudes are not checkpointed.

Example
=======

```jldoctest
using Breeze

grid = RectilinearGrid(size=(16, 16, 16), extent=(2π, 2π, 2π), topology=(Periodic, Periodic, Periodic))
forcing = StochasticBandForcing(grid; energy_injection_rate=0.1, correlation_time=0.5, seed=1)

# output
StochasticBandForcing(ε=0.1, T_f=0.5) with 9 modes with 0.0 ≤ |k| ≤ 1.5
```
"""
function StochasticBandForcing(grid;
                               energy_injection_rate,
                               correlation_time,
                               minimum_wavenumber = 0,
                               maximum_wavenumber = default_maximum_wavenumber(grid),
                               seed = 0)

    FT = eltype(grid)
    band = ForcedWavenumberBand(grid; minimum_wavenumber, maximum_wavenumber)
    M = number_of_modes(band)
    amplitudes = on_architecture(architecture(grid), zeros(Complex{FT}, M, 3))
    state = OrnsteinUhlenbeckState(zeros(ComplexF64, M, 3), 0.0, 0, false, zero(UInt64))

    return StochasticBandForcing(convert(FT, energy_injection_rate), convert(FT, correlation_time),
                                 grid, band, amplitudes, yz_amplitudes_array(grid, band), state,
                                 convert(UInt64, seed))
end

Base.summary(forcing::StochasticBandForcing) =
    string("StochasticBandForcing(ε=", prettysummary(forcing.energy_injection_rate),
           ", T_f=", prettysummary(forcing.correlation_time), ")")

Base.show(io::IO, forcing::StochasticBandForcing) =
    print(io, summary(forcing), " with ", band_summary(forcing.band))

AtmosphereModels.materialize_atmosphere_model_forcing(forcing::StochasticBandForcing, field, name,
                                                      model_field_names, context::NamedTuple) =
    materialize_band_forcing_component(forcing, name)

"""
$(TYPEDSIGNATURES)

Return the standard deviation of each real component of the Ornstein-Uhlenbeck amplitudes of `forcing`.
The projected amplitude of each of the ``M`` half-space modes has two complex components,
so ``⟨|\\boldsymbol{F}|²⟩ = 8 M σ²``, and ``σ² = ε / (8 M T_f)``.
"""
function amplitude_standard_deviation(forcing::StochasticBandForcing)
    M = number_of_modes(forcing.band)
    ε = Float64(forcing.energy_injection_rate)
    T = Float64(forcing.correlation_time)
    return sqrt(ε / (8 * M * T))
end

# The amplitudes are independent of the velocity and are advanced once per time step,
# when the first component asks for them at a new iteration.
function compute_band_forcing!(forcing::StochasticBandForcing, c, model)
    clock = model.clock
    state = forcing.state
    new_step = !state.initialized | (clock.iteration != state.iteration)
    new_step && advance_amplitudes!(forcing, clock.time, clock.iteration)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Advance the Ornstein-Uhlenbeck amplitudes of `forcing` to `time` with the exact transition
``b ← b \\, e^{-δt/T_f} + σ \\sqrt{1 - e^{-2δt/T_f}} \\, ξ``, where ``δt`` is the time elapsed since the
last update and ``ξ`` is a complex standard normal variate, then project and copy the amplitudes to the device.
The first call draws the amplitudes from the stationary distribution.
"""
function advance_amplitudes!(forcing::StochasticBandForcing, time, iteration)
    state = forcing.state
    σ = amplitude_standard_deviation(forcing)

    if state.initialized
        δt = Float64(time - state.time)
        decay = exp(-δt / forcing.correlation_time)
        diffusion = σ * sqrt(1 - decay^2)
    else
        decay = 0.0
        diffusion = σ
    end

    b = state.amplitudes

    for c in 1:3, m in axes(b, 1)
        state.counter += 1
        b[m, c] = decay * b[m, c] + diffusion * complex_normal(forcing.seed, state.counter)
    end

    state.time = time
    state.iteration = iteration
    state.initialized = true

    copyto!(forcing.amplitudes, projected_amplitudes(forcing))
    compute_yz_amplitudes!(forcing)

    return nothing
end

function projected_amplitudes(forcing::StochasticBandForcing)
    FT = eltype(forcing.grid)
    b = forcing.state.amplitudes
    k̃ = Array(forcing.band.modified_wavenumbers)
    A = similar(b, Complex{FT})

    for m in axes(b, 1)
        Pb = transverse_projection((b[m, 1], b[m, 2], b[m, 3]), (k̃[m, 1], k̃[m, 2], k̃[m, 3]))
        A[m, 1], A[m, 2], A[m, 3] = Pb
    end

    return A
end
