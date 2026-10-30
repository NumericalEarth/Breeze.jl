#####
##### Compressible vector-invariant momentum advection
#####
##### Advects the momentum density ρ𝐮 of the compressible `AtmosphereModel` in the vector-invariant form
#####
#####   ∂t(ρ𝐮) = 𝐔 × 𝛇 − ρ ∇K − 𝐮 ∇·𝐔,    𝐔 = ρ𝐮,  K = ½|𝐮|²,  𝛇 = ∇ × 𝐮,
#####
##### which is the flux form −∇·(𝐔 ⊗ 𝐮) rewritten with the mass flux 𝐔 that the continuity
##### equation evolves ρ with. The planetary vorticity stays in the model's Coriolis term.
#####
##### By default the vertical direction is split off as in hydrostatic ocean models: horizontal
##### momentum takes the horizontal vector-invariant form with the horizontal mass-flux
##### divergence, plus flux-form vertical advection; vertical momentum is advected in flux form
##### horizontally and in gradient/divergence form vertically. The unsplit three-dimensional form
##### is available through `vertical_formulation = UnsplitVerticalAdvection()`.
#####
##### Every discrete term is built so that the kinetic energy weighted by the face densities of
##### the model's velocity diagnosis, Σ V ℑρ u²/2, is exactly conserved by the centered scheme
##### on any orthogonal grid (stretched spacings included), and so that an upwind-biased
##### reconstruction of the relative vorticity still does no work (its pairing is symmetric).
##### See the derivation note shipped with the scheme for the proofs.
#####

using Oceananigans.Advection:
    AbstractAdvectionScheme,
    AbstractCenteredAdvectionScheme,
    AbstractUpwindBiasedAdvectionScheme,
    Centered,
    EnergyConserving,
    FunctionStencil,
    DefaultStencil,
    VelocityStencil,
    bias,
    divergence_smoothness,
    materialize_advection,
    u_smoothness,
    v_smoothness,
    weno_order,
    _biased_interpolate_xᶠᵃᵃ,
    _biased_interpolate_xᶜᵃᵃ,
    _biased_interpolate_yᵃᶠᵃ,
    _biased_interpolate_yᵃᶜᵃ,
    _biased_interpolate_zᵃᵃᶠ,
    _symmetric_interpolate_xᶠᵃᵃ,
    _symmetric_interpolate_xᶜᵃᵃ,
    _symmetric_interpolate_yᵃᶠᵃ,
    _symmetric_interpolate_yᵃᶜᵃ,
    _symmetric_interpolate_zᵃᵃᶠ

using Oceananigans.Architectures: Architectures, on_architecture
using Oceananigans.Grids: Grids, AbstractGrid, AbstractHorizontallyCurvilinearGrid, required_halo_size_x
using Oceananigans.TimeSteppers: TimeSteppers, ExplicitTimeDiscretization

using Oceananigans.Operators:
    Axᶠᶜᶜ, Axᶠᶜᶠ, Ayᶜᶠᶜ, Ayᶜᶠᶠ, Azᶜᶜᶠ, Azᶠᶜᶠ, Azᶜᶠᶠ,
    Vᶠᶜᶜ, Vᶜᶠᶜ, Vᶜᶜᶠ, Vᶜᶜᶜ,
    V⁻¹ᶠᶜᶜ, V⁻¹ᶜᶠᶜ, V⁻¹ᶜᶜᶠ, Az⁻¹ᶠᶠᶜ,
    Δxᶜᶜᶠ, Δyᶜᶜᶠ,
    Δx_qᶠᶜᶜ, Δx_qᶜᶠᶜ, Δx_qᶜᶜᶠ, Δy_qᶠᶜᶜ, Δy_qᶜᶠᶜ, Δy_qᶜᶜᶠ, Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ, Az_qᶜᶜᶠ,
    δxᶜᶜᶜ, δyᶜᶜᶜ, δzᶜᶜᶜ, δxᶠᶠᶜ, δyᶠᶠᶜ, δxᶠᵃᵃ, δyᵃᶠᵃ, δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, δzᵃᵃᶠ,
    ℑxᶜᵃᵃ, ℑxᶠᵃᵃ, ℑyᵃᶜᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ, ℑxyᶠᶠᵃ, ℑxzᶠᵃᶠ, ℑyzᵃᶠᶠ,
    ∂xᶠᶜᶠ, ∂yᶜᶠᶠ, ∂zᶠᶜᶠ, ∂zᶜᶠᶠ,
    ζ₃ᶠᶠᶜ, div_xyᶜᶜᶜ, divᶜᶜᶜ

using Oceananigans: Oceananigans
using DocStringExtensions: TYPEDEF

using ..AtmosphereModels: dynamics_density

#####
##### Vertical formulations
#####

"""
$(TYPEDEF)

Hydrostatic-style split of [`CompressibleVectorInvariant`](@ref): horizontal momentum is advected
by the horizontal vector-invariant form with the horizontal mass-flux divergence, plus flux-form
vertical advection, and vertical momentum is advected in flux form.
"""
struct SplitVerticalAdvection end

"""
$(TYPEDEF)

Unsplit three-dimensional vector-invariant form of [`CompressibleVectorInvariant`](@ref): all
three momentum components take the Lamb, kinetic energy gradient, and full three-dimensional
mass-flux divergence terms. Supported with centered (`EnergyConserving`) sub-schemes only.
"""
struct UnsplitVerticalAdvection end

#####
##### Vorticity pairings
#####

"""
$(TYPEDEF)

Pairing of a reconstructed vorticity in which each (u, v) pair takes the vorticity at the corner it
shares, corrected by the deviations of the reconstructions from their centered counterparts,
`ζ_q + wʸ ℑʸ(Rʸζ − Cʸζ) + wˣ ℑˣ(Rˣζ − Cˣζ)`, each weighted by the share `wˣ = u²/(u² + v²)` of the
corner's transport along the reconstruction. Its linearization about a sheared flow is that of the
corner vorticity, so the pairing is free of the Hollingsworth instability of the averaged vorticity,
and an upwind-biased reconstruction enters only through its odd (dissipative) and nonlinear parts,
multiplied by the transport that biases it.
"""
struct CornerVorticityPairing end

"""
$(TYPEDEF)

Pairing of a reconstructed vorticity in which each (u, v) pair takes the vorticity at the corner it
shares plus the deviations of the reconstructions from their centered counterparts at its own two
points, each weighted by the share of the corner's transport along the reconstruction,
`ζ_q + wʸ_q (Rʸζ − Cʸζ)_u + wˣ_q (Rˣζ − Cˣζ)_v`, `wˣ = u²/(u² + v²)`. Symmetric, hence energy-neutral
for any reconstruction; its linearization about a sheared flow is that of the corner vorticity, so
it is free of the Hollingsworth instability; and, unlike `CornerVorticityPairing`, the deviations are
not averaged back to the corner, so the upwind dissipation acts on grid-scale (2Δ) vorticity, where
a two-point average would annihilate it. The default.
"""
struct UpwindVorticityPairing end

"""
$(TYPEDEF)

Pairing of a reconstructed vorticity in which each (u, v) pair takes the mean of the vorticity
reconstructed at the u point along y and at the v point along x, as `EnergyConservingUpwinding`
does for `VectorInvariant`. Energy-neutral, but the averaging of the vorticity across the pair
leaves a non-cancellation with the kinetic energy gradient in the linearization about a sheared
flow (Hollingsworth et al. 1983); kept for reference and analysis.
"""
struct AveragedVorticityPairing end

#####
##### Gradient-form upwinding of the kinetic energy gradient
#####

"""
$(TYPEDEF)

Upwinding of the kinetic energy gradient that keeps it a discrete gradient: the kinetic energy of
each cell is reconstructed from the face kinetic energies with the upwind-biased `scheme`, biased
by the cell's own velocity, and the momentum equations carry `ρ ∇K` of that cell value. Unlike the
upwinded differences of `OnlySelfUpwinding`, the result has no discrete curl, so it cannot inject
vorticity at interfaces where the velocities are interpolated (the Yin-Yang overlap).
"""
struct GradientFormUpwinding{S}
    scheme :: S
end

Base.summary(g::GradientFormUpwinding) = string("GradientFormUpwinding(", summary(g.scheme), ")")
Adapt.adapt_structure(to, g::GradientFormUpwinding) = GradientFormUpwinding(Adapt.adapt(to, g.scheme))
Architectures.on_architecture(to, g::GradientFormUpwinding) = GradientFormUpwinding(on_architecture(to, g.scheme))
Oceananigans.Advection.materialize_advection(g::GradientFormUpwinding, grid) = GradientFormUpwinding(materialize_advection(g.scheme, grid))

#####
##### The scheme
#####

struct CompressibleVectorInvariant{N, FT, Z, ZS, V, K, D, VF, P, C, CS} <: AbstractAdvectionScheme{N, FT, ExplicitTimeDiscretization}
    vorticity_scheme               :: Z
    vorticity_stencil              :: ZS
    vertical_advection_scheme      :: V
    kinetic_energy_gradient_scheme :: K
    divergence_scheme              :: D
    vertical_formulation           :: VF
    vorticity_pairing              :: P
    centered_vorticity_scheme      :: C  # the centered counterpart of `vorticity_scheme` the pairings subtract
    core_scheme                    :: CS # the centered interpolation P of the non-dissipative core (and its transpose)

    function CompressibleVectorInvariant{N, FT}(vorticity_scheme::Z,
                                                vorticity_stencil::ZS,
                                                vertical_advection_scheme::V,
                                                kinetic_energy_gradient_scheme::K,
                                                divergence_scheme::D,
                                                vertical_formulation::VF,
                                                vorticity_pairing::P,
                                                centered_vorticity_scheme::C,
                                                core_scheme::CS) where {N, FT, Z, ZS, V, K, D, VF, P, C, CS}

        return new{N, FT, Z, ZS, V, K, D, VF, P, C, CS}(vorticity_scheme,
                                                        vorticity_stencil,
                                                        vertical_advection_scheme,
                                                        kinetic_energy_gradient_scheme,
                                                        divergence_scheme,
                                                        vertical_formulation,
                                                        vorticity_pairing,
                                                        centered_vorticity_scheme,
                                                        core_scheme)
    end
end

const CVI = CompressibleVectorInvariant

# Sub-scheme selection: `EnergyConserving` is the centered, exactly energy-conserving form of each term;
# any other advection scheme reconstructs the term with that scheme (upwind-biased or centered of higher order).
const CenteredVorticity = CVI{<:Any, <:Any, <:EnergyConserving}
const CenteredKineticEnergy = CVI{<:Any, <:Any, <:Any, <:Any, <:Any, <:EnergyConserving}
const ReconstructedKineticEnergy = CVI{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractAdvectionScheme}
const GradientUpwindedKineticEnergy = CVI{<:Any, <:Any, <:Any, <:Any, <:Any, <:GradientFormUpwinding}
const CenteredDivergence = CVI{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:EnergyConserving}
const ReconstructedDivergence = CVI{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractAdvectionScheme}
const SplitCVI = CVI{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:SplitVerticalAdvection}
const UnsplitCVI = CVI{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:UnsplitVerticalAdvection}
const CornerPairedVorticity = CVI{<:Any, <:Any, <:AbstractAdvectionScheme, <:Any, <:Any, <:Any, <:Any, <:Any, <:CornerVorticityPairing}
const UpwindPairedVorticity = CVI{<:Any, <:Any, <:AbstractAdvectionScheme, <:Any, <:Any, <:Any, <:Any, <:Any, <:UpwindVorticityPairing}
const AveragedPairedVorticity = CVI{<:Any, <:Any, <:AbstractAdvectionScheme, <:Any, <:Any, <:Any, <:Any, <:Any, <:AveragedVorticityPairing}

# How far beyond its own stencil a reconstructed vorticity is read by the pairing: the upwind pairing
# averages the partners' deviations with two-point interpolations (one point), the corner and averaged
# pairings interpolate them with the core scheme (its buffer).
pairing_reach(::UpwindVorticityPairing, core_scheme) = 1
pairing_reach(pairing, core_scheme) = reconstruction_buffer(core_scheme)
pairing_reach(::Nothing, core_scheme) = 1

# The centered scheme whose stencil is the symmetric part of the linearized reconstruction
centered_counterpart(::EnergyConserving, FT) = nothing
centered_counterpart(::AbstractAdvectionScheme{B}, FT) where B = Centered(FT; order = 2B)

reconstruction_buffer(::EnergyConserving) = 0
reconstruction_buffer(::AbstractAdvectionScheme{B}) where B = B
reconstruction_buffer(g::GradientFormUpwinding) = reconstruction_buffer(g.scheme)

"""
$(TYPEDSIGNATURES)

Return a vector-invariant momentum advection scheme for the compressible `AtmosphereModel`, which
advects the momentum density `ρ𝐮` with the mass flux `𝐔 = ρ𝐮` of the continuity equation,

```
∂t(ρ𝐮) = 𝐔 × 𝛇 − ρ ∇K − 𝐮 ∇·𝐔,    K = ½|𝐮|²,  𝛇 = ∇ × 𝐮.
```

The planetary vorticity is not part of the scheme: it stays in the model's Coriolis term.
Velocities are the model's own diagnosis `u = ρu / ℑˣρ`, and the discrete kinetic energy
`Σ V ℑρ u² / 2` weighted by the same face densities is exactly conserved by the centered scheme
on any orthogonal grid. The relative vorticity may be reconstructed with an upwind-biased scheme
without doing work on the flow: the reconstructed vorticity enters the `u` and `v` equations
through a symmetric pairing, so upwinding dissipates enstrophy but not energy.

Keyword arguments
=================

- `vorticity_scheme`: Reconstruction of the relative vorticity at the velocity points. Default:
  `EnergyConserving()`, which uses the vorticity at its own (corner) points. Any other advection
  scheme (e.g. `WENO(order=9)`) reconstructs it along the direction of transport, paired symmetrically.

- `vorticity_stencil`: Smoothness stencil of a `WENO` vorticity reconstruction. Default: `VelocityStencil()`.

- `vertical_advection_scheme`: Reconstruction of the vertical flux of horizontal momentum and of the
  horizontal fluxes of vertical momentum. Default: `EnergyConserving()` (centered).

- `kinetic_energy_gradient_scheme`: `EnergyConserving()` (default, the exact discrete gradient of the
  cell kinetic energy), an upwind scheme (reconstruction of the self-transported differences
  `δx(u²/2)`, as `WENOVectorInvariant`), or `GradientFormUpwinding(scheme)` (the gradient of a cell
  kinetic energy reconstructed with `scheme`, which stays curl-free).

- `divergence_scheme`: Reconstruction of the self-transported part of the mass-flux divergence
  (`δx(ρu)` in the `u` equation, and so on). Default: `EnergyConserving()`.

- `vertical_formulation`: `SplitVerticalAdvection()` (default) or `UnsplitVerticalAdvection()`.

- `core_order`: Order (2, 4, …) of the centered interpolation `P` that carries the self-transported
  part of the Lamb term: in the `v` equation the `∂x v` part of the vorticity is interpolated along
  `x` with `P` (and the `Δy u` transport along `y` with `ℑ`), in the `u` equation the `∂y u` part
  along `y` with `P`; the parts that cancel the kinetic energy gradient keep the two-point
  interpolation that the gradient has, so the pairing stays symmetric (energy-exact) and the
  linearization about a sheared flow unchanged (no Hollingsworth instability). Default: 4.

- `vorticity_pairing`: How a reconstructed vorticity is shared between the `u` and `v` equations:
  `UpwindVorticityPairing()` (default), which is free of the Hollingsworth instability and dissipates
  grid-scale vorticity; `CornerVorticityPairing()`, also Hollingsworth-free but with no dissipation of
  2Δ vorticity; or `AveragedVorticityPairing()`, the pairing of Oceananigans' `EnergyConservingUpwinding`,
  which is Hollingsworth-unstable. All three conserve energy exactly.

Example
=======

```jldoctest
julia> using Breeze

julia> CompressibleVectorInvariant()
CompressibleVectorInvariant (SplitVerticalAdvection)
├── vorticity_scheme: EnergyConserving
├── vertical_advection_scheme: EnergyConserving
├── kinetic_energy_gradient_scheme: EnergyConserving
├── divergence_scheme: EnergyConserving
└── core_scheme: Centered(order=4)
```
"""
function CompressibleVectorInvariant(FT::DataType = Oceananigans.defaults.FloatType;
                                     vorticity_scheme = EnergyConserving(FT),
                                     vorticity_stencil = VelocityStencil(),
                                     vertical_advection_scheme = EnergyConserving(FT),
                                     kinetic_energy_gradient_scheme = EnergyConserving(FT),
                                     divergence_scheme = EnergyConserving(FT),
                                     vertical_formulation = SplitVerticalAdvection(),
                                     vorticity_pairing = UpwindVorticityPairing(),
                                     core_order = 4)

    core_scheme = Centered(FT; order = core_order)

    if vertical_formulation isa UnsplitVerticalAdvection
        all(s -> s isa EnergyConserving, (vorticity_scheme, vertical_advection_scheme,
                                          kinetic_energy_gradient_scheme, divergence_scheme)) ||
            throw(ArgumentError("UnsplitVerticalAdvection supports only EnergyConserving sub-schemes."))
    end

    # The centered vorticity has nothing to pair but its corner value
    vorticity_pairing = vorticity_scheme isa EnergyConserving ? nothing : vorticity_pairing

    N = max(1,
            reconstruction_buffer(vorticity_scheme) + pairing_reach(vorticity_pairing, core_scheme),
            reconstruction_buffer(kinetic_energy_gradient_scheme) + 1,
            reconstruction_buffer(divergence_scheme) + 1,
            reconstruction_buffer(core_scheme) + 1,
            reconstruction_buffer(vertical_advection_scheme))

    return CompressibleVectorInvariant{N, FT}(vorticity_scheme,
                                              vorticity_stencil,
                                              vertical_advection_scheme,
                                              kinetic_energy_gradient_scheme,
                                              divergence_scheme,
                                              vertical_formulation,
                                              vorticity_pairing,
                                              centered_counterpart(vorticity_scheme, FT),
                                              core_scheme)
end

nothing_to_default(user_value; default) = isnothing(user_value) ? default : user_value

"""
$(TYPEDSIGNATURES)

Return a [`CompressibleVectorInvariant`](@ref) scheme whose relative vorticity, vertical fluxes
and mass-flux divergence are reconstructed with weighted essentially non-oscillatory (`WENO`)
schemes of the given orders, while the kinetic energy gradient stays the exact discrete gradient
(`EnergyConserving()`): an upwinded kinetic energy gradient is not curl-free and injects vorticity
wherever the velocities are interpolated rather than integrated, as in the overlap of a Yin-Yang
grid, and the self-upwinded vorticity, vertical fluxes and divergence already damp the grid scale.
`order` sets every order at once; the specific orders override it. Pass `order = nothing` (the
default) to use the defaults 9 for the vorticity and 5 for the other terms, and
`kinetic_energy_gradient_order` to upwind the kinetic energy gradient too. Remaining keyword
arguments are passed to `WENO`.

The grid halo must hold the stencils: `(vorticity_order + 1) ÷ 2 + 1` points horizontally and
`(kinetic_energy_gradient_order + 1) ÷ 2 + 1` points vertically with the defaults, that is a halo
of `(6, 6, 4)`.

Example
=======

```jldoctest
julia> using Breeze

julia> CompressibleWENOVectorInvariant()
CompressibleWENOVectorInvariant{6, Float64}(vorticity_order=9, vertical_order=5)
├── vorticity_scheme: WENO{5, Float64, Nothing}(order=9)
├── vorticity_stencil: Oceananigans.Advection.VelocityStencil
├── vorticity_pairing: UpwindVorticityPairing
├── vertical_advection_scheme: WENO{3, Float64, Nothing}(order=5)
├── kinetic_energy_gradient_scheme: EnergyConserving
├── divergence_scheme: WENO{3, Float64, Nothing}(order=5)
└── core_scheme: Centered(order=4)
```
"""
function CompressibleWENOVectorInvariant(FT::DataType = Oceananigans.defaults.FloatType;
                                         order = nothing,
                                         vorticity_order = nothing,
                                         vertical_order = nothing,
                                         divergence_order = nothing,
                                         kinetic_energy_gradient_order = nothing,
                                         vorticity_stencil = VelocityStencil(),
                                         vorticity_pairing = UpwindVorticityPairing(),
                                         core_order = 4,
                                         minimum_buffer_upwind_order = 1,
                                         weno_kw...)

    default_vorticity_order = isnothing(order) ? 9 : order
    default_order = isnothing(order) ? 5 : order

    vorticity_order  = nothing_to_default(vorticity_order; default=default_vorticity_order)
    vertical_order   = nothing_to_default(vertical_order; default=default_order)
    divergence_order = nothing_to_default(divergence_order; default=default_order)

    vorticity_scheme               = WENO(FT; order=vorticity_order, minimum_buffer_upwind_order, weno_kw...)
    vertical_advection_scheme      = WENO(FT; order=vertical_order, minimum_buffer_upwind_order, weno_kw...)
    divergence_scheme              = WENO(FT; order=divergence_order, minimum_buffer_upwind_order, weno_kw...)
    kinetic_energy_gradient_scheme = isnothing(kinetic_energy_gradient_order) ? EnergyConserving(FT) :
                                     WENO(FT; order=kinetic_energy_gradient_order, minimum_buffer_upwind_order, weno_kw...)

    return CompressibleVectorInvariant(FT; vorticity_scheme, vorticity_stencil, vertical_advection_scheme,
                                       kinetic_energy_gradient_scheme, divergence_scheme, vorticity_pairing, core_order)
end

const CompressibleWENOVectorInvariantScheme{N} = CVI{<:Any, <:Any, <:WENO{N}} where N

TimeSteppers.time_discretization(::CVI) = ExplicitTimeDiscretization()

# The vorticity needs one halo point beyond the stencil that reconstructs it; the kinetic energy
# gradient and divergence reconstruct differences of the prognostic fields, which also reach one
# point further than the stencil.
@inline Grids.required_halo_size_x(scheme::CVI{N}) where N = N
@inline Grids.required_halo_size_y(scheme::CVI{N}) where N = N
@inline Grids.required_halo_size_z(scheme::CVI) = max(1,
                                                      reconstruction_buffer(scheme.vertical_advection_scheme),
                                                      reconstruction_buffer(scheme.kinetic_energy_gradient_scheme) + 1,
                                                      reconstruction_buffer(scheme.divergence_scheme) + 1)

Adapt.adapt_structure(to, scheme::CVI{N, FT}) where {N, FT} =
    CompressibleVectorInvariant{N, FT}(Adapt.adapt(to, scheme.vorticity_scheme),
                                       Adapt.adapt(to, scheme.vorticity_stencil),
                                       Adapt.adapt(to, scheme.vertical_advection_scheme),
                                       Adapt.adapt(to, scheme.kinetic_energy_gradient_scheme),
                                       Adapt.adapt(to, scheme.divergence_scheme),
                                       scheme.vertical_formulation,
                                       scheme.vorticity_pairing,
                                       scheme.centered_vorticity_scheme,
                                       scheme.core_scheme)

Architectures.on_architecture(to, scheme::CVI{N, FT}) where {N, FT} =
    CompressibleVectorInvariant{N, FT}(on_architecture(to, scheme.vorticity_scheme),
                                       on_architecture(to, scheme.vorticity_stencil),
                                       on_architecture(to, scheme.vertical_advection_scheme),
                                       on_architecture(to, scheme.kinetic_energy_gradient_scheme),
                                       on_architecture(to, scheme.divergence_scheme),
                                       scheme.vertical_formulation,
                                       scheme.vorticity_pairing,
                                       scheme.centered_vorticity_scheme,
                                       scheme.core_scheme)

# The sub-schemes may defer settings (such as WENO weight computation) until the grid is known.
Oceananigans.Advection.materialize_advection(scheme::CVI{N, FT}, grid) where {N, FT} =
    CompressibleVectorInvariant{N, FT}(materialize_advection(scheme.vorticity_scheme, grid),
                                       scheme.vorticity_stencil,
                                       materialize_advection(scheme.vertical_advection_scheme, grid),
                                       materialize_advection(scheme.kinetic_energy_gradient_scheme, grid),
                                       materialize_advection(scheme.divergence_scheme, grid),
                                       scheme.vertical_formulation,
                                       scheme.vorticity_pairing,
                                       scheme.centered_vorticity_scheme,
                                       scheme.core_scheme)

# The scheme has no per-direction order to adapt to a small grid.
Oceananigans.Advection.adapt_advection_order(scheme::CVI, grid::AbstractGrid) = scheme

Base.summary(::CVI{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, VF}) where VF =
    string("CompressibleVectorInvariant (", nameof(VF), ")")

function Base.summary(scheme::CompressibleWENOVectorInvariantScheme{N}) where N
    FT = eltype(scheme.vorticity_scheme)
    H = required_halo_size_x(scheme)
    return string("CompressibleWENOVectorInvariant{$H, $FT}(vorticity_order=", weno_order(scheme.vorticity_scheme),
                  ", vertical_order=", weno_order(scheme.vertical_advection_scheme), ")")
end

function Base.show(io::IO, scheme::CVI)
    print(io, summary(scheme), '\n')
    print(io, "├── vorticity_scheme: ", summary(scheme.vorticity_scheme), '\n')
    if scheme.vorticity_scheme isa WENO
        print(io, "├── vorticity_stencil: ", summary(scheme.vorticity_stencil), '\n')
    end
    if !(scheme.vorticity_scheme isa EnergyConserving)
        print(io, "├── vorticity_pairing: ", nameof(typeof(scheme.vorticity_pairing)), '\n')
    end
    print(io, "├── vertical_advection_scheme: ", summary(scheme.vertical_advection_scheme), '\n')
    print(io, "├── kinetic_energy_gradient_scheme: ", summary(scheme.kinetic_energy_gradient_scheme), '\n')
    print(io, "├── divergence_scheme: ", summary(scheme.divergence_scheme), '\n')
    print(io, "└── core_scheme: ", summary(scheme.core_scheme))
end

#####
##### Building blocks
#####
##### Locations follow the C-grid: ρ at ccc; ρu, u at fcc; ρv, v at cfc; ρw, w at ccf.
##### Face densities are those of the model's velocity diagnosis, ρᵘ = ℑˣρ, so ρᵘ u = ρu exactly.
##### Mass fluxes through the cell faces are U = Ax ρu, V = Ay ρv, W = Az ρw, as in continuity.
#####

# Volume-weighted kinetic energies of the faces, 2 V_face K_face
@inline Vu²ᶠᶜᶜ(i, j, k, grid, u) = @inbounds Vᶠᶜᶜ(i, j, k, grid) * u[i, j, k]^2
@inline Vv²ᶜᶠᶜ(i, j, k, grid, v) = @inbounds Vᶜᶠᶜ(i, j, k, grid) * v[i, j, k]^2
@inline Vw²ᶜᶜᶠ(i, j, k, grid, w) = @inbounds Vᶜᶜᶠ(i, j, k, grid) * w[i, j, k]^2

# The centered interpolation P of the core (`Centered(order=core_order)`) and its transpose: ᶠᵃᵃ from
# ᶜᵃᵃ and back with the same symmetric coefficients, so that Σ a P(b) = Σ b Pᵀ(a) exactly away from walls.
@inline Pxᶠᵃᵃ(i, j, k, grid, scheme, f, args...) = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.core_scheme, f, args...)
@inline Pxᶜᵃᵃ(i, j, k, grid, scheme, f, args...) = _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.core_scheme, f, args...)
@inline Pyᵃᶠᵃ(i, j, k, grid, scheme, f, args...) = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.core_scheme, f, args...)
@inline Pyᵃᶜᵃ(i, j, k, grid, scheme, f, args...) = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.core_scheme, f, args...)

# Kinetic energy of the cell: the volume-weighted average of the kinetic energies of its faces, the
# weighting under which the face-density-weighted kinetic energy Σ V ℑρ u²/2 responds to the
# continuity equation as Σ V ρₜ K. On a uniform grid this is ½(ℑˣu² + ℑʸv² + ℑᶻw²).
# (The kinetic energy must keep the two-point interpolation of the velocity diagnosis: the divergence
# term u ℑˣ(∇·ρ𝐮) cancels u ℑˣ(∂tρ) in the velocity equation only with ℑˣ, and the energy identity then
# pins K to ℑˣ(V u²)/2V. A higher-order K would need a higher-order divergence weighting, whose mismatch
# with continuity couples through the Coriolis force into a grid-scale instability.)
@inline u_kinetic_energyᶜᶜᶜ(i, j, k, grid, scheme, u, v) = ℑxᶜᵃᵃ(i, j, k, grid, Vu²ᶠᶜᶜ, u) / (2 * Vᶜᶜᶜ(i, j, k, grid))
@inline v_kinetic_energyᶜᶜᶜ(i, j, k, grid, scheme, u, v) = ℑyᵃᶜᵃ(i, j, k, grid, Vv²ᶜᶠᶜ, v) / (2 * Vᶜᶜᶜ(i, j, k, grid))
@inline w_kinetic_energyᶜᶜᶜ(i, j, k, grid, w) = ℑzᵃᵃᶜ(i, j, k, grid, Vw²ᶜᶜᶠ, w) / (2 * Vᶜᶜᶜ(i, j, k, grid))

@inline horizontal_kinetic_energyᶜᶜᶜ(i, j, k, grid, scheme, u, v) =
    u_kinetic_energyᶜᶜᶜ(i, j, k, grid, scheme, u, v) + v_kinetic_energyᶜᶜᶜ(i, j, k, grid, scheme, u, v)

# The kinetic energy whose gradient the horizontal momentum equations carry
@inline kinetic_energyᶜᶜᶜ(i, j, k, grid, ::SplitVerticalAdvection, scheme, u, v, w) =
    horizontal_kinetic_energyᶜᶜᶜ(i, j, k, grid, scheme, u, v)

@inline kinetic_energyᶜᶜᶜ(i, j, k, grid, ::UnsplitVerticalAdvection, scheme, u, v, w) =
    horizontal_kinetic_energyᶜᶜᶜ(i, j, k, grid, scheme, u, v) + w_kinetic_energyᶜᶜᶜ(i, j, k, grid, w)

# Mass-flux divergences per unit volume, formed from the same face fluxes as continuity, and the
# smoothness of the horizontal divergence that biases their reconstruction
@inline mass_flux_divergence_smoothness(i, j, k, grid, ρu, ρv, ρw) = divergence_smoothness(i, j, k, grid, ρu, ρv)

@inline x_mass_flux_divergenceᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw) = δxᶜᶜᶜ(i, j, k, grid, Ax_qᶠᶜᶜ, ρu) / Vᶜᶜᶜ(i, j, k, grid)
@inline y_mass_flux_divergenceᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw) = δyᶜᶜᶜ(i, j, k, grid, Ay_qᶜᶠᶜ, ρv) / Vᶜᶜᶜ(i, j, k, grid)
@inline z_mass_flux_divergenceᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw) = δzᶜᶜᶜ(i, j, k, grid, Az_qᶜᶜᶠ, ρw) / Vᶜᶜᶜ(i, j, k, grid)

@inline mass_flux_divergenceᶜᶜᶜ(i, j, k, grid, ::SplitVerticalAdvection, ρu, ρv, ρw) = div_xyᶜᶜᶜ(i, j, k, grid, ρu, ρv)
@inline mass_flux_divergenceᶜᶜᶜ(i, j, k, grid, ::UnsplitVerticalAdvection, ρu, ρv, ρw) = divᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw)

# The parts of the divergence that are not self-transported in each equation
@inline x_cross_divergenceᶜᶜᶜ(i, j, k, grid, ::SplitVerticalAdvection, ρu, ρv, ρw) = y_mass_flux_divergenceᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw)
@inline y_cross_divergenceᶜᶜᶜ(i, j, k, grid, ::SplitVerticalAdvection, ρu, ρv, ρw) = x_mass_flux_divergenceᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw)
@inline z_cross_divergenceᶜᶜᶜ(i, j, k, grid, ::SplitVerticalAdvection, ρu, ρv, ρw) = zero(grid)

# Velocity-sign proxies for the transport direction of the vorticity at the velocity points
@inline ℑxy_Δx_vᶠᶜᶜ(i, j, k, grid, v) = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, v)
@inline ℑxy_Δy_uᶜᶠᶜ(i, j, k, grid, u) = ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, u)

# The density of a vorticity pairing: the density at the corner the paired u and v points share
@inline ρζᶠᶠᶜ(i, j, k, grid, ρ, u, v) = ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)

#####
##### Horizontal Lamb term, V ζ in the u equation and −U ζ in the v equation
#####
##### Each u point is paired with the four v points around it (and vice versa) through the corner
##### they share, with the pair coefficient ¼ Ax Δx ρ_corner (ζ) — symmetric in the pair since
##### Axᶠᶜᶜ Δxᶜᶠᶜ = Ayᶜᶠᶜ Δyᶠᶜᶜ — so that Σ V u Lᵘ + Σ V v Lᵛ = 0 exactly. The centered form uses the
##### corner vorticity itself; the reconstructed form gives the pair the mean of the vorticity
##### reconstructed at the u point along y (biased by v) and at the v point along x (biased by u).
#####

# The corner vorticity split into its two parts, ζ = ζᵛ − ζᵘ with ζᵛ = δx(Δy v)/Az and ζᵘ = δy(Δx u)/Az.
# In the v equation ζᵛ carries the self-advection −U ∂x v and is interpolated along x with the core
# scheme P, while ζᵘ, which cancels the kinetic energy gradient −∂y K, keeps the two-point ℑˣ that K
# has; in the u equation the roles swap. The transposes on the other side of each pair keep the
# pairing symmetric: the pair (u, v) through corner q weighs ζᵛ_q by [P_x]_{q,v}[ℑʸ]_{q,u} and ζᵘ_q by
# [ℑˣ]_{q,v}[P_y]_{q,u} on both sides.
@inline ζᵛᶠᶠᶜ(i, j, k, grid, u, v) = δxᶠᶠᶜ(i, j, k, grid, Δy_qᶜᶠᶜ, v) * Az⁻¹ᶠᶠᶜ(i, j, k, grid)
@inline ζᵘᶠᶠᶜ(i, j, k, grid, u, v) = δyᶠᶠᶜ(i, j, k, grid, Δx_qᶠᶜᶜ, u) * Az⁻¹ᶠᶠᶜ(i, j, k, grid)

# The corner correction a pairing adds to the corner vorticity (none for the upwind pairing, whose
# corrections sit at the velocity points)
@inline corner_correctionᶠᶠᶜ(i, j, k, grid, scheme, u, v) = zero(grid)

@inline ρζᵛ_Px_Δx_vᶠᶠᶜ(i, j, k, grid, scheme, ρ, u, v) =
    ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * ζᵛᶠᶠᶜ(i, j, k, grid, u, v) * Pxᶠᵃᵃ(i, j, k, grid, scheme, Δx_qᶜᶠᶜ, v)

@inline ρζᵘ_ℑx_Δx_vᶠᶠᶜ(i, j, k, grid, scheme, ρ, u, v) =
    ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * (ζᵘᶠᶠᶜ(i, j, k, grid, u, v) - corner_correctionᶠᶠᶜ(i, j, k, grid, scheme, u, v)) *
    ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v)

@inline ρζᵘ_Py_Δy_uᶠᶠᶜ(i, j, k, grid, scheme, ρ, u, v) =
    ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * (ζᵘᶠᶠᶜ(i, j, k, grid, u, v) - corner_correctionᶠᶠᶜ(i, j, k, grid, scheme, u, v)) *
    Pyᵃᶠᵃ(i, j, k, grid, scheme, Δy_qᶠᶜᶜ, u)

@inline ρζᵛ_ℑy_Δy_uᶠᶠᶜ(i, j, k, grid, scheme, ρ, u, v) =
    ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * ζᵛᶠᶠᶜ(i, j, k, grid, u, v) * ℑyᵃᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u)

@inline x_lamb_term(i, j, k, grid, scheme::CVI, ρ, u, v) =
    + Axᶠᶜᶜ(i, j, k, grid) * V⁻¹ᶠᶜᶜ(i, j, k, grid) * (ℑyᵃᶜᵃ(i, j, k, grid, ρζᵛ_Px_Δx_vᶠᶠᶜ, scheme, ρ, u, v) -
                                                    Pyᵃᶜᵃ(i, j, k, grid, scheme, ρζᵘ_ℑx_Δx_vᶠᶠᶜ, scheme, ρ, u, v))

@inline y_lamb_term(i, j, k, grid, scheme::CVI, ρ, u, v) =
    - Ayᶜᶠᶜ(i, j, k, grid) * V⁻¹ᶜᶠᶜ(i, j, k, grid) * (Pxᶜᵃᵃ(i, j, k, grid, scheme, ρζᵛ_ℑy_Δy_uᶠᶠᶜ, scheme, ρ, u, v) -
                                                    ℑxᶜᵃᵃ(i, j, k, grid, ρζᵘ_Py_Δy_uᶠᶠᶜ, scheme, ρ, u, v))

@inline function reconstructed_vorticityᶠᶜᶜ(i, j, k, grid, scheme, u, v)
    v̂ = ℑxy_Δx_vᶠᶜᶜ(i, j, k, grid, v)
    return _biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.vorticity_scheme, bias(v̂), ζ₃ᶠᶠᶜ, scheme.vorticity_stencil, u, v)
end

@inline function reconstructed_vorticityᶜᶠᶜ(i, j, k, grid, scheme, u, v)
    û = ℑxy_Δy_uᶜᶠᶜ(i, j, k, grid, u)
    return _biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.vorticity_scheme, bias(û), ζ₃ᶠᶠᶜ, scheme.vorticity_stencil, u, v)
end

@inline ρ_ℑx_Δx_vᶠᶠᶜ(i, j, k, grid, scheme, ρ, v) = ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * Pxᶠᵃᵃ(i, j, k, grid, scheme, Δx_qᶜᶠᶜ, v)
@inline ρ_ℑy_Δy_uᶠᶠᶜ(i, j, k, grid, scheme, ρ, u) = ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * Pyᵃᶠᵃ(i, j, k, grid, scheme, Δy_qᶠᶜᶜ, u)

@inline Δx_v_ζᶜᶠᶜ(i, j, k, grid, scheme, u, v) = Δx_qᶜᶠᶜ(i, j, k, grid, v) * reconstructed_vorticityᶜᶠᶜ(i, j, k, grid, scheme, u, v)
@inline Δy_u_ζᶠᶜᶜ(i, j, k, grid, scheme, u, v) = Δy_qᶠᶜᶜ(i, j, k, grid, u) * reconstructed_vorticityᶠᶜᶜ(i, j, k, grid, scheme, u, v)

@inline ρ_ℑx_Δx_v_ζᶠᶠᶜ(i, j, k, grid, scheme, ρ, u, v) = ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * Pxᶠᵃᵃ(i, j, k, grid, scheme, Δx_v_ζᶜᶠᶜ, scheme, u, v)
@inline ρ_ℑy_Δy_u_ζᶠᶠᶜ(i, j, k, grid, scheme, ρ, u, v) = ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * Pyᵃᶠᵃ(i, j, k, grid, scheme, Δy_u_ζᶠᶜᶜ, scheme, u, v)

# Corner pairing: the corner vorticity plus the deviations of the reconstructions at the velocity points
# from their centered counterparts, averaged back to the corner. The deviations carry the odd
# (dissipative) and nonlinear parts of an upwind-biased reconstruction, and vanish for a centered one.
# Each deviation is weighted by the share of the corner's transport in its direction, u²/(u² + v²) for
# the reconstruction along x biased by u, so that a pair multiplies each deviation by (the magnitude of)
# the velocity that biased it, as the unpaired upwinding does, while the pairing stays symmetric.
@inline function x_deviation_weightᶠᶠᶜ(i, j, k, grid, u, v)
    û² = ℑyᵃᶠᵃ(i, j, k, grid, u)^2
    v̂² = ℑxᶠᵃᵃ(i, j, k, grid, v)^2
    s = û² + v̂²
    return ifelse(s > 0, û² / s, oftype(s, 0.5))
end

@inline y_deviation_weightᶠᶠᶜ(i, j, k, grid, u, v) = 1 - x_deviation_weightᶠᶠᶜ(i, j, k, grid, u, v)

@inline y_vorticity_deviationᶠᶜᶜ(i, j, k, grid, scheme, u, v) =
    reconstructed_vorticityᶠᶜᶜ(i, j, k, grid, scheme, u, v) -
    _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.centered_vorticity_scheme, ζ₃ᶠᶠᶜ, u, v)

@inline x_vorticity_deviationᶜᶠᶜ(i, j, k, grid, scheme, u, v) =
    reconstructed_vorticityᶜᶠᶜ(i, j, k, grid, scheme, u, v) -
    _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.centered_vorticity_scheme, ζ₃ᶠᶠᶜ, u, v)

@inline corner_correctionᶠᶠᶜ(i, j, k, grid, scheme::CornerPairedVorticity, u, v) =
    - y_deviation_weightᶠᶠᶜ(i, j, k, grid, u, v) * ℑyᵃᶠᵃ(i, j, k, grid, y_vorticity_deviationᶠᶜᶜ, scheme, u, v) -
      x_deviation_weightᶠᶠᶜ(i, j, k, grid, u, v) * ℑxᶠᵃᵃ(i, j, k, grid, x_vorticity_deviationᶜᶠᶜ, scheme, u, v)

# Upwind pairing: each pair takes the corner vorticity plus the transport-weighted deviations at its own two
# points. Through the corner q the u point sees + Ax/V ℑʸ(ρ_q ζ_q ℑˣ(Δx v)) [corner part, as the centered scheme]
# + Ax/V devʸ(u) ℑʸ(ρ_q wʸ_q ℑˣ(Δx v)) [its own deviation, times the transport it pairs with]
# + Ax/V ℑʸ(ρ_q wˣ_q ℑˣ(Δx v devˣ(v))) [the deviations of its v partners], and the v point the mirror image.
# (The deviation terms are dissipative corrections: they keep the two-point interpolations, which cost a
# quarter of the reconstructions of the core interpolation and are their own transposes.)
@inline ρwʸ_ℑx_Δx_vᶠᶠᶜ(i, j, k, grid, scheme, ρ, u, v) =
    ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * y_deviation_weightᶠᶠᶜ(i, j, k, grid, u, v) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v)

@inline ρwˣ_ℑy_Δy_uᶠᶠᶜ(i, j, k, grid, scheme, ρ, u, v) =
    ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * x_deviation_weightᶠᶠᶜ(i, j, k, grid, u, v) * ℑyᵃᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u)

@inline Δx_v_devᶜᶠᶜ(i, j, k, grid, scheme, u, v) = Δx_qᶜᶠᶜ(i, j, k, grid, v) * x_vorticity_deviationᶜᶠᶜ(i, j, k, grid, scheme, u, v)
@inline Δy_u_devᶠᶜᶜ(i, j, k, grid, scheme, u, v) = Δy_qᶠᶜᶜ(i, j, k, grid, u) * y_vorticity_deviationᶠᶜᶜ(i, j, k, grid, scheme, u, v)

@inline ρwˣ_ℑx_Δx_v_devᶠᶠᶜ(i, j, k, grid, scheme, ρ, u, v) =
    ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * x_deviation_weightᶠᶠᶜ(i, j, k, grid, u, v) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_v_devᶜᶠᶜ, scheme, u, v)

@inline ρwʸ_ℑy_Δy_u_devᶠᶠᶜ(i, j, k, grid, scheme, ρ, u, v) =
    ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * y_deviation_weightᶠᶠᶜ(i, j, k, grid, u, v) * ℑyᵃᶠᵃ(i, j, k, grid, Δy_u_devᶠᶜᶜ, scheme, u, v)

@inline function x_lamb_term(i, j, k, grid, scheme::UpwindPairedVorticity, ρ, u, v)
    corner = ℑyᵃᶜᵃ(i, j, k, grid, ρζᵛ_Px_Δx_vᶠᶠᶜ, scheme, ρ, u, v) - Pyᵃᶜᵃ(i, j, k, grid, scheme, ρζᵘ_ℑx_Δx_vᶠᶠᶜ, scheme, ρ, u, v)
    own = y_vorticity_deviationᶠᶜᶜ(i, j, k, grid, scheme, u, v) * ℑyᵃᶜᵃ(i, j, k, grid, ρwʸ_ℑx_Δx_vᶠᶠᶜ, scheme, ρ, u, v)
    partners = ℑyᵃᶜᵃ(i, j, k, grid, ρwˣ_ℑx_Δx_v_devᶠᶠᶜ, scheme, ρ, u, v)
    return + Axᶠᶜᶜ(i, j, k, grid) * V⁻¹ᶠᶜᶜ(i, j, k, grid) * (corner + own + partners)
end

@inline function y_lamb_term(i, j, k, grid, scheme::UpwindPairedVorticity, ρ, u, v)
    corner = Pxᶜᵃᵃ(i, j, k, grid, scheme, ρζᵛ_ℑy_Δy_uᶠᶠᶜ, scheme, ρ, u, v) - ℑxᶜᵃᵃ(i, j, k, grid, ρζᵘ_Py_Δy_uᶠᶠᶜ, scheme, ρ, u, v)
    own = x_vorticity_deviationᶜᶠᶜ(i, j, k, grid, scheme, u, v) * ℑxᶜᵃᵃ(i, j, k, grid, ρwˣ_ℑy_Δy_uᶠᶠᶜ, scheme, ρ, u, v)
    partners = ℑxᶜᵃᵃ(i, j, k, grid, ρwʸ_ℑy_Δy_u_devᶠᶠᶜ, scheme, ρ, u, v)
    return - Ayᶜᶠᶜ(i, j, k, grid) * V⁻¹ᶜᶠᶜ(i, j, k, grid) * (corner + own + partners)
end

# Averaged pairing: each pair takes ½(ζᵘ + ζᵛ) of the reconstructions at its two points
@inline function x_lamb_term(i, j, k, grid, scheme::AveragedPairedVorticity, ρ, u, v)
    ζᵘ = reconstructed_vorticityᶠᶜᶜ(i, j, k, grid, scheme, u, v)
    ρv_ζᵘ = ζᵘ * Pyᵃᶜᵃ(i, j, k, grid, scheme, ρ_ℑx_Δx_vᶠᶠᶜ, scheme, ρ, v)
    ρv_ζᵛ = Pyᵃᶜᵃ(i, j, k, grid, scheme, ρ_ℑx_Δx_v_ζᶠᶠᶜ, scheme, ρ, u, v)
    return + Axᶠᶜᶜ(i, j, k, grid) * V⁻¹ᶠᶜᶜ(i, j, k, grid) * (ρv_ζᵘ + ρv_ζᵛ) / 2
end

@inline function y_lamb_term(i, j, k, grid, scheme::AveragedPairedVorticity, ρ, u, v)
    ζᵛ = reconstructed_vorticityᶜᶠᶜ(i, j, k, grid, scheme, u, v)
    ρu_ζᵛ = ζᵛ * Pxᶜᵃᵃ(i, j, k, grid, scheme, ρ_ℑy_Δy_uᶠᶠᶜ, scheme, ρ, u)
    ρu_ζᵘ = Pxᶜᵃᵃ(i, j, k, grid, scheme, ρ_ℑy_Δy_u_ζᶠᶠᶜ, scheme, ρ, u, v)
    return - Ayᶜᶠᶜ(i, j, k, grid) * V⁻¹ᶜᶠᶜ(i, j, k, grid) * (ρu_ζᵛ + ρu_ζᵘ) / 2
end

#####
##### Kinetic energy gradient, ρ ∇K, with the face density of the momentum
#####
##### The gradient is Ax/V δx K rather than δx K / Δx so that Σ V u ρᵘ (Ax/V) δxK = Σ U δxK with the
##### face mass flux U, on grids whose areas are not products of spacings (LatitudeLongitudeGrid).
#####

@inline x_kinetic_energy_gradient(i, j, k, grid, scheme::CenteredKineticEnergy, vf, ρ, u, v, w) =
    ℑxᶠᵃᵃ(i, j, k, grid, ρ) * Axᶠᶜᶜ(i, j, k, grid) * V⁻¹ᶠᶜᶜ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, kinetic_energyᶜᶜᶜ, vf, scheme, u, v, w)

@inline y_kinetic_energy_gradient(i, j, k, grid, scheme::CenteredKineticEnergy, vf, ρ, u, v, w) =
    ℑyᵃᶠᵃ(i, j, k, grid, ρ) * Ayᶜᶠᶜ(i, j, k, grid) * V⁻¹ᶜᶠᶜ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, kinetic_energyᶜᶜᶜ, vf, scheme, u, v, w)

# Self-transported parts, reconstructed along the direction of transport with the bias of the
# transporting velocity, as the differences δx(V u²/2)/V of the face kinetic energies, and the
# cross parts, differenced exactly.
@inline δx_u_kinetic_energyᶜᶜᶜ(i, j, k, grid, u, v) = δxᶜᶜᶜ(i, j, k, grid, Vu²ᶠᶜᶜ, u) / (2 * Vᶜᶜᶜ(i, j, k, grid))
@inline δy_v_kinetic_energyᶜᶜᶜ(i, j, k, grid, u, v) = δyᶜᶜᶜ(i, j, k, grid, Vv²ᶜᶠᶜ, v) / (2 * Vᶜᶜᶜ(i, j, k, grid))

@inline function x_kinetic_energy_gradient(i, j, k, grid, scheme::ReconstructedKineticEnergy, vf, ρ, u, v, w)
    @inbounds û = u[i, j, k]
    δKᵘ = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.kinetic_energy_gradient_scheme, bias(û),
                                   δx_u_kinetic_energyᶜᶜᶜ, FunctionStencil(u_smoothness), u, v)
    δKᵛ = δxᶠᵃᵃ(i, j, k, grid, v_kinetic_energyᶜᶜᶜ, scheme, u, v)
    return ℑxᶠᵃᵃ(i, j, k, grid, ρ) * Axᶠᶜᶜ(i, j, k, grid) * V⁻¹ᶠᶜᶜ(i, j, k, grid) * (δKᵘ + δKᵛ)
end

@inline function y_kinetic_energy_gradient(i, j, k, grid, scheme::ReconstructedKineticEnergy, vf, ρ, u, v, w)
    @inbounds v̂ = v[i, j, k]
    δKᵛ = _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.kinetic_energy_gradient_scheme, bias(v̂),
                                   δy_v_kinetic_energyᶜᶜᶜ, FunctionStencil(v_smoothness), u, v)
    δKᵘ = δyᵃᶠᵃ(i, j, k, grid, u_kinetic_energyᶜᶜᶜ, scheme, u, v)
    return ℑyᵃᶠᵃ(i, j, k, grid, ρ) * Ayᶜᶠᶜ(i, j, k, grid) * V⁻¹ᶜᶠᶜ(i, j, k, grid) * (δKᵛ + δKᵘ)
end

# Gradient form: the cell kinetic energy reconstructed from the face kinetic energies with the
# upwind-biased scheme, biased by the cell velocity, then differenced exactly
@inline function upwinded_kinetic_energyᶜᶜᶜ(i, j, k, grid, scheme, vf, u, v, w)
    g = scheme.kinetic_energy_gradient_scheme.scheme
    û = ℑxᶜᵃᵃ(i, j, k, grid, u)
    v̂ = ℑyᵃᶜᵃ(i, j, k, grid, v)
    Kᵘ = _biased_interpolate_xᶜᵃᵃ(i, j, k, grid, g, bias(û), Vu²ᶠᶜᶜ, DefaultStencil(), u)
    Kᵛ = _biased_interpolate_yᵃᶜᵃ(i, j, k, grid, g, bias(v̂), Vv²ᶜᶠᶜ, DefaultStencil(), v)
    return (Kᵘ + Kᵛ) / (2 * Vᶜᶜᶜ(i, j, k, grid)) + vertical_kinetic_energyᶜᶜᶜ(i, j, k, grid, vf, w)
end

@inline vertical_kinetic_energyᶜᶜᶜ(i, j, k, grid, ::SplitVerticalAdvection, w) = zero(grid)
@inline vertical_kinetic_energyᶜᶜᶜ(i, j, k, grid, ::UnsplitVerticalAdvection, w) = w_kinetic_energyᶜᶜᶜ(i, j, k, grid, w)

@inline x_kinetic_energy_gradient(i, j, k, grid, scheme::GradientUpwindedKineticEnergy, vf, ρ, u, v, w) =
    ℑxᶠᵃᵃ(i, j, k, grid, ρ) * Axᶠᶜᶜ(i, j, k, grid) * V⁻¹ᶠᶜᶜ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, upwinded_kinetic_energyᶜᶜᶜ, scheme, vf, u, v, w)

@inline y_kinetic_energy_gradient(i, j, k, grid, scheme::GradientUpwindedKineticEnergy, vf, ρ, u, v, w) =
    ℑyᵃᶠᵃ(i, j, k, grid, ρ) * Ayᶜᶠᶜ(i, j, k, grid) * V⁻¹ᶜᶠᶜ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, upwinded_kinetic_energyᶜᶜᶜ, scheme, vf, u, v, w)

@inline z_kinetic_energy_gradient(i, j, k, grid, scheme::GradientUpwindedKineticEnergy, vf, ρ, u, v, w) =
    z_kinetic_energy_gradient(i, j, k, grid, scheme, vf, scheme.vertical_formulation, ρ, u, v, w)

# the w equation keeps its centered vertical kinetic energy gradient under the gradient form
@inline z_kinetic_energy_gradient(i, j, k, grid, scheme::GradientUpwindedKineticEnergy, vf, ::SplitVerticalAdvection, ρ, u, v, w) =
    ℑzᵃᵃᶠ(i, j, k, grid, ρ) * Azᶜᶜᶠ(i, j, k, grid) * V⁻¹ᶜᶜᶠ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, z_kinetic_energyᶜᶜᶜ, vf, scheme, u, v, w)

#####
##### Mass-flux divergence term, u ∇·𝐔
#####

@inline x_divergence_term(i, j, k, grid, scheme::CenteredDivergence, vf, u, ρu, ρv, ρw) =
    @inbounds u[i, j, k] * ℑxᶠᵃᵃ(i, j, k, grid, mass_flux_divergenceᶜᶜᶜ, vf, ρu, ρv, ρw)

@inline y_divergence_term(i, j, k, grid, scheme::CenteredDivergence, vf, v, ρu, ρv, ρw) =
    @inbounds v[i, j, k] * ℑyᵃᶠᵃ(i, j, k, grid, mass_flux_divergenceᶜᶜᶜ, vf, ρu, ρv, ρw)

# Self-transported part reconstructed along the direction of transport, cross part interpolated exactly
# (reconstruction is only supported with the split form, whose cross part is the other horizontal divergence)
@inline function x_divergence_term(i, j, k, grid, scheme::ReconstructedDivergence, vf, u, ρu, ρv, ρw)
    @inbounds û = u[i, j, k]
    δU = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.divergence_scheme, bias(û),
                                  x_mass_flux_divergenceᶜᶜᶜ, FunctionStencil(mass_flux_divergence_smoothness), ρu, ρv, ρw)
    δV = ℑxᶠᵃᵃ(i, j, k, grid, x_cross_divergenceᶜᶜᶜ, vf, ρu, ρv, ρw)
    return û * (δU + δV)
end

@inline function y_divergence_term(i, j, k, grid, scheme::ReconstructedDivergence, vf, v, ρu, ρv, ρw)
    @inbounds v̂ = v[i, j, k]
    δV = _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.divergence_scheme, bias(v̂),
                                  y_mass_flux_divergenceᶜᶜᶜ, FunctionStencil(mass_flux_divergence_smoothness), ρu, ρv, ρw)
    δU = ℑyᵃᶠᵃ(i, j, k, grid, y_cross_divergenceᶜᶜᶜ, vf, ρu, ρv, ρw)
    return v̂ * (δV + δU)
end

# In the split form the vertical momentum carries only the vertical divergence, δz W / V
@inline function z_divergence_term(i, j, k, grid, scheme::ReconstructedDivergence, vf, w, ρu, ρv, ρw)
    @inbounds ŵ = w[i, j, k]
    δW = _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme.divergence_scheme, bias(ŵ),
                                  z_mass_flux_divergenceᶜᶜᶜ, DefaultStencil(), ρu, ρv, ρw)
    return ŵ * δW
end

#####
##### Vertical advection of horizontal momentum (split form): flux form with the mass flux
##### Wᵘ = Azᶠᶜᶠ ℑˣ(ρw), the vertical mass flux of the u column. Interpolating ρw and multiplying
##### by the column's own area (rather than interpolating the flux Az ρw) is what pairs this flux
##### exactly with the volume-weighted kinetic energy under the continuity equation.
#####

@inline Az_ℑx_ρwᶠᶜᶠ(i, j, k, grid, scheme, ρw) = Azᶠᶜᶠ(i, j, k, grid) * ℑxᶠᵃᵃ(i, j, k, grid, ρw)
@inline Az_ℑy_ρwᶜᶠᶠ(i, j, k, grid, scheme, ρw) = Azᶜᶠᶠ(i, j, k, grid) * ℑyᵃᶠᵃ(i, j, k, grid, ρw)

# Reconstruction of the advected velocity at the flux face: centered for `EnergyConserving`, with
# the given scheme otherwise (upwind-biased by the sign of the transporting mass flux)
@inline reconstruct_zᵃᵃᶠ(i, j, k, grid, ::EnergyConserving, W, c) = ℑzᵃᵃᶠ(i, j, k, grid, c)
@inline reconstruct_zᵃᵃᶠ(i, j, k, grid, scheme::AbstractCenteredAdvectionScheme, W, c) = _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c)
@inline reconstruct_zᵃᵃᶠ(i, j, k, grid, scheme::AbstractUpwindBiasedAdvectionScheme, W, c) = _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, bias(W), c)

@inline reconstruct_xᶠᵃᵃ(i, j, k, grid, ::EnergyConserving, U, c) = ℑxᶠᵃᵃ(i, j, k, grid, c)
@inline reconstruct_xᶠᵃᵃ(i, j, k, grid, scheme::AbstractCenteredAdvectionScheme, U, c) = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c)
@inline reconstruct_xᶠᵃᵃ(i, j, k, grid, scheme::AbstractUpwindBiasedAdvectionScheme, U, c) = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, bias(U), c)

@inline reconstruct_yᵃᶠᵃ(i, j, k, grid, ::EnergyConserving, V, c) = ℑyᵃᶠᵃ(i, j, k, grid, c)
@inline reconstruct_yᵃᶠᵃ(i, j, k, grid, scheme::AbstractCenteredAdvectionScheme, V, c) = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c)
@inline reconstruct_yᵃᶠᵃ(i, j, k, grid, scheme::AbstractUpwindBiasedAdvectionScheme, V, c) = _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, bias(V), c)

@inline function vertical_momentum_flux_Wu(i, j, k, grid, scheme, ρw, u)
    Wᵘ = Az_ℑx_ρwᶠᶜᶠ(i, j, k, grid, scheme, ρw)
    return Wᵘ * reconstruct_zᵃᵃᶠ(i, j, k, grid, scheme.vertical_advection_scheme, Wᵘ, u)
end

@inline function vertical_momentum_flux_Wv(i, j, k, grid, scheme, ρw, v)
    Wᵛ = Az_ℑy_ρwᶜᶠᶠ(i, j, k, grid, scheme, ρw)
    return Wᵛ * reconstruct_zᵃᵃᶠ(i, j, k, grid, scheme.vertical_advection_scheme, Wᵛ, v)
end

@inline x_vertical_advection(i, j, k, grid, scheme::SplitCVI, ρw, u) =
    V⁻¹ᶠᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, vertical_momentum_flux_Wu, scheme, ρw, u)

@inline y_vertical_advection(i, j, k, grid, scheme::SplitCVI, ρw, v) =
    V⁻¹ᶜᶠᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, vertical_momentum_flux_Wv, scheme, ρw, v)

@inline x_vertical_advection(i, j, k, grid, ::UnsplitCVI, ρw, u) = zero(grid)
@inline y_vertical_advection(i, j, k, grid, ::UnsplitCVI, ρw, v) = zero(grid)

#####
##### Vertical momentum (split form): flux form horizontally with the mass fluxes Axᶠᶜᶠ ℑᶻ(ρu) and
##### Ayᶜᶠᶠ ℑᶻ(ρv) of the w level, and the gradient / divergence form vertically, with the vertical
##### kinetic energy K_w = ℑᶻ(V w²)/(2V) and the vertical mass-flux divergence δz W / V, which is
##### the vertical flux form on a uniform grid and stays exactly energy-consistent when Δz varies.
#####

@inline Ax_ℑz_ρuᶠᶜᶠ(i, j, k, grid, ρu) = Axᶠᶜᶠ(i, j, k, grid) * ℑzᵃᵃᶠ(i, j, k, grid, ρu)
@inline Ay_ℑz_ρvᶜᶠᶠ(i, j, k, grid, ρv) = Ayᶜᶠᶠ(i, j, k, grid) * ℑzᵃᵃᶠ(i, j, k, grid, ρv)

@inline function horizontal_momentum_flux_Uw(i, j, k, grid, scheme, ρu, w)
    Uʷ = Ax_ℑz_ρuᶠᶜᶠ(i, j, k, grid, ρu)
    return Uʷ * reconstruct_xᶠᵃᵃ(i, j, k, grid, scheme, Uʷ, w)
end

@inline function horizontal_momentum_flux_Vw(i, j, k, grid, scheme, ρv, w)
    Vʷ = Ay_ℑz_ρvᶜᶠᶠ(i, j, k, grid, ρv)
    return Vʷ * reconstruct_yᵃᶠᵃ(i, j, k, grid, scheme, Vʷ, w)
end

@inline z_horizontal_advection(i, j, k, grid, scheme::SplitCVI, ρu, ρv, w) =
    V⁻¹ᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, horizontal_momentum_flux_Uw, scheme.vertical_advection_scheme, ρu, w) +
                              δyᵃᶜᵃ(i, j, k, grid, horizontal_momentum_flux_Vw, scheme.vertical_advection_scheme, ρv, w))

@inline z_horizontal_advection(i, j, k, grid, ::UnsplitCVI, ρu, ρv, w) = zero(grid)

# The kinetic energy whose vertical gradient the vertical momentum equation carries
@inline z_kinetic_energyᶜᶜᶜ(i, j, k, grid, ::SplitVerticalAdvection, scheme, u, v, w) = w_kinetic_energyᶜᶜᶜ(i, j, k, grid, w)
@inline z_kinetic_energyᶜᶜᶜ(i, j, k, grid, vf::UnsplitVerticalAdvection, scheme, u, v, w) = kinetic_energyᶜᶜᶜ(i, j, k, grid, vf, scheme, u, v, w)

@inline z_kinetic_energy_gradient(i, j, k, grid, scheme::CenteredKineticEnergy, vf, ρ, u, v, w) =
    ℑzᵃᵃᶠ(i, j, k, grid, ρ) * Azᶜᶜᶠ(i, j, k, grid) * V⁻¹ᶜᶜᶠ(i, j, k, grid) *
    δzᵃᵃᶠ(i, j, k, grid, z_kinetic_energyᶜᶜᶜ, vf, scheme, u, v, w)

@inline δz_w_kinetic_energyᶜᶜᶜ(i, j, k, grid, w) = δzᶜᶜᶜ(i, j, k, grid, Vw²ᶜᶜᶠ, w) / (2 * Vᶜᶜᶜ(i, j, k, grid))
@inline w_smoothness(i, j, k, grid, w) = ℑzᵃᵃᶜ(i, j, k, grid, w)

@inline function z_kinetic_energy_gradient(i, j, k, grid, scheme::ReconstructedKineticEnergy, vf, ρ, u, v, w)
    @inbounds ŵ = w[i, j, k]
    δKʷ = _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme.kinetic_energy_gradient_scheme, bias(ŵ),
                                   δz_w_kinetic_energyᶜᶜᶜ, FunctionStencil(w_smoothness), w)
    return ℑzᵃᵃᶠ(i, j, k, grid, ρ) * Azᶜᶜᶠ(i, j, k, grid) * V⁻¹ᶜᶜᶠ(i, j, k, grid) * δKʷ
end

# The vertical mass-flux divergence the split vertical momentum equation carries
@inline z_mass_flux_divergenceᶜᶜᶜ(i, j, k, grid, ::SplitVerticalAdvection, ρu, ρv, ρw) = z_mass_flux_divergenceᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw)
@inline z_mass_flux_divergenceᶜᶜᶜ(i, j, k, grid, ::UnsplitVerticalAdvection, ρu, ρv, ρw) = divᶜᶜᶜ(i, j, k, grid, ρu, ρv, ρw)

@inline z_divergence_term(i, j, k, grid, ::CenteredDivergence, vf, w, ρu, ρv, ρw) =
    @inbounds w[i, j, k] * ℑzᵃᵃᶠ(i, j, k, grid, z_mass_flux_divergenceᶜᶜᶜ, vf, ρu, ρv, ρw)

#####
##### Unsplit form: vertical Lamb pairings, 𝐔 × 𝛇 with ζˣ = ∂y w − ∂z v and ζʸ = ∂z u − ∂x w,
##### with the pair coefficients ¼ Ax Δx ρ ζʸ for (u, w) and ¼ Ay Δy ρ ζˣ for (v, w), the same on both sides.
#####

@inline ζ₂ᶠᶜᶠ(i, j, k, grid, u, w) = ∂zᶠᶜᶠ(i, j, k, grid, u) - ∂xᶠᶜᶠ(i, j, k, grid, w)
@inline ζ₁ᶜᶠᶠ(i, j, k, grid, v, w) = ∂yᶜᶠᶠ(i, j, k, grid, w) - ∂zᶜᶠᶠ(i, j, k, grid, v)

@inline ρζ₂ᶠᶜᶠ(i, j, k, grid, ρ, u, w) = ℑxzᶠᵃᶠ(i, j, k, grid, ρ) * ζ₂ᶠᶜᶠ(i, j, k, grid, u, w)
@inline ρζ₁ᶜᶠᶠ(i, j, k, grid, ρ, v, w) = ℑyzᵃᶠᶠ(i, j, k, grid, ρ) * ζ₁ᶜᶠᶠ(i, j, k, grid, v, w)

@inline ρζ₂_ℑx_Δx_wᶠᶜᶠ(i, j, k, grid, ρ, u, w) = ρζ₂ᶠᶜᶠ(i, j, k, grid, ρ, u, w) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶜᶠ, w)
@inline ρζ₂_ℑz_Ax_uᶠᶜᶠ(i, j, k, grid, ρ, u, w) = ρζ₂ᶠᶜᶠ(i, j, k, grid, ρ, u, w) * ℑzᵃᵃᶠ(i, j, k, grid, Ax_qᶠᶜᶜ, u)
@inline ρζ₁_ℑy_Δy_wᶜᶠᶠ(i, j, k, grid, ρ, v, w) = ρζ₁ᶜᶠᶠ(i, j, k, grid, ρ, v, w) * ℑyᵃᶠᵃ(i, j, k, grid, Δy_qᶜᶜᶠ, w)
@inline ρζ₁_ℑz_Ay_vᶜᶠᶠ(i, j, k, grid, ρ, v, w) = ρζ₁ᶜᶠᶠ(i, j, k, grid, ρ, v, w) * ℑzᵃᵃᶠ(i, j, k, grid, Ay_qᶜᶠᶜ, v)

# − W ζʸ in the u equation
@inline x_vertical_lamb_term(i, j, k, grid, ::UnsplitCVI, ρ, u, w) =
    - Axᶠᶜᶜ(i, j, k, grid) * V⁻¹ᶠᶜᶜ(i, j, k, grid) * ℑzᵃᵃᶜ(i, j, k, grid, ρζ₂_ℑx_Δx_wᶠᶜᶠ, ρ, u, w)

# + W ζˣ in the v equation
@inline y_vertical_lamb_term(i, j, k, grid, ::UnsplitCVI, ρ, v, w) =
    + Ayᶜᶠᶜ(i, j, k, grid) * V⁻¹ᶜᶠᶜ(i, j, k, grid) * ℑzᵃᵃᶜ(i, j, k, grid, ρζ₁_ℑy_Δy_wᶜᶠᶠ, ρ, v, w)

# + U ζʸ − V ζˣ in the w equation
@inline z_lamb_term(i, j, k, grid, ::UnsplitCVI, ρ, u, v, w) =
    V⁻¹ᶜᶜᶠ(i, j, k, grid) * (Δxᶜᶜᶠ(i, j, k, grid) * ℑxᶜᵃᵃ(i, j, k, grid, ρζ₂_ℑz_Ax_uᶠᶜᶠ, ρ, u, w) -
                              Δyᶜᶜᶠ(i, j, k, grid) * ℑyᵃᶜᵃ(i, j, k, grid, ρζ₁_ℑz_Ay_vᶜᶠᶠ, ρ, v, w))

@inline x_vertical_lamb_term(i, j, k, grid, ::SplitCVI, ρ, u, w) = zero(grid)
@inline y_vertical_lamb_term(i, j, k, grid, ::SplitCVI, ρ, v, w) = zero(grid)
@inline z_lamb_term(i, j, k, grid, ::SplitCVI, ρ, u, v, w) = zero(grid)

#####
##### Curvature of the vertical direction on the sphere (deep-atmosphere w coupling, cf. MITgcm 2.105–2.107):
##### −uw/r in the u equation, −vw/r in the v equation, +(u² + v²)/r in the w equation, weighted by the cell
##### density and paired through the cell so that Σ V u Mᵘ + Σ V v Mᵛ + Σ V w Mʷ = 0 exactly. The horizontal
##### (tan φ) curvature is carried by the vector-invariant form itself.
#####

const HCG = AbstractHorizontallyCurvilinearGrid

@inline ρuwVᶜᶜᶜ(i, j, k, grid, ρ, u, w)   = @inbounds ρ[i, j, k] * ℑxᶜᵃᵃ(i, j, k, grid, u) * ℑzᵃᵃᶜ(i, j, k, grid, w) * Vᶜᶜᶜ(i, j, k, grid)
@inline ρvwVᶜᶜᶜ(i, j, k, grid, ρ, v, w)   = @inbounds ρ[i, j, k] * ℑyᵃᶜᵃ(i, j, k, grid, v) * ℑzᵃᵃᶜ(i, j, k, grid, w) * Vᶜᶜᶜ(i, j, k, grid)
@inline ρu²v²Vᶜᶜᶜ(i, j, k, grid, ρ, u, v) = @inbounds ρ[i, j, k] * (ℑxᶜᵃᵃ(i, j, k, grid, u)^2 + ℑyᵃᶜᵃ(i, j, k, grid, v)^2) * Vᶜᶜᶜ(i, j, k, grid)

@inline x_curvature_term(i, j, k, grid, ρ, u, v, w) = zero(grid)
@inline y_curvature_term(i, j, k, grid, ρ, u, v, w) = zero(grid)
@inline z_curvature_term(i, j, k, grid, ρ, u, v, w) = zero(grid)

@inline x_curvature_term(i, j, k, grid::HCG, ρ, u, v, w) = + V⁻¹ᶠᶜᶜ(i, j, k, grid) / grid.radius * ℑxᶠᵃᵃ(i, j, k, grid, ρuwVᶜᶜᶜ, ρ, u, w)
@inline y_curvature_term(i, j, k, grid::HCG, ρ, u, v, w) = + V⁻¹ᶜᶠᶜ(i, j, k, grid) / grid.radius * ℑyᵃᶠᵃ(i, j, k, grid, ρvwVᶜᶜᶜ, ρ, v, w)
@inline z_curvature_term(i, j, k, grid::HCG, ρ, u, v, w) = - V⁻¹ᶜᶜᶠ(i, j, k, grid) / grid.radius * ℑzᵃᵃᶠ(i, j, k, grid, ρu²v²Vᶜᶜᶜ, ρ, u, v)

#####
##### Momentum flux divergences: the negative of the advective tendencies, as the model subtracts them
#####

@inline function AtmosphereModels.x_momentum_flux_divergence(i, j, k, grid, scheme::CVI, momentum, velocities, dynamics)
    ρ = dynamics_density(dynamics)
    u, v, w = velocities
    ρu, ρv, ρw = momentum
    vf = scheme.vertical_formulation
    return ( - x_lamb_term(i, j, k, grid, scheme, ρ, u, v)
             - x_vertical_lamb_term(i, j, k, grid, scheme, ρ, u, w)
             + x_kinetic_energy_gradient(i, j, k, grid, scheme, vf, ρ, u, v, w)
             + x_divergence_term(i, j, k, grid, scheme, vf, u, ρu, ρv, ρw)
             + x_vertical_advection(i, j, k, grid, scheme, ρw, u)
             + x_curvature_term(i, j, k, grid, ρ, u, v, w))
end

@inline function AtmosphereModels.y_momentum_flux_divergence(i, j, k, grid, scheme::CVI, momentum, velocities, dynamics)
    ρ = dynamics_density(dynamics)
    u, v, w = velocities
    ρu, ρv, ρw = momentum
    vf = scheme.vertical_formulation
    return ( - y_lamb_term(i, j, k, grid, scheme, ρ, u, v)
             - y_vertical_lamb_term(i, j, k, grid, scheme, ρ, v, w)
             + y_kinetic_energy_gradient(i, j, k, grid, scheme, vf, ρ, u, v, w)
             + y_divergence_term(i, j, k, grid, scheme, vf, v, ρu, ρv, ρw)
             + y_vertical_advection(i, j, k, grid, scheme, ρw, v)
             + y_curvature_term(i, j, k, grid, ρ, u, v, w))
end

@inline function AtmosphereModels.z_momentum_flux_divergence(i, j, k, grid, scheme::CVI, momentum, velocities, dynamics)
    ρ = dynamics_density(dynamics)
    u, v, w = velocities
    ρu, ρv, ρw = momentum
    vf = scheme.vertical_formulation
    return ( - z_lamb_term(i, j, k, grid, scheme, ρ, u, v, w)
             + z_kinetic_energy_gradient(i, j, k, grid, scheme, vf, ρ, u, v, w)
             + z_divergence_term(i, j, k, grid, scheme, vf, w, ρu, ρv, ρw)
             + z_horizontal_advection(i, j, k, grid, scheme, ρu, ρv, w)
             + z_curvature_term(i, j, k, grid, ρ, u, v, w))
end
