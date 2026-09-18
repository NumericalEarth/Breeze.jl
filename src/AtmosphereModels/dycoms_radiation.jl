#####
##### Idealized longwave radiation for nocturnal marine stratocumulus (DYCOMS-II)
#####

using Oceananigans: Field
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: Center, Face, znode
using Oceananigans.Utils: IterationInterval

"""
$(TYPEDEF)

Idealized longwave radiation for nocturnal marine stratocumulus following
[Stevens (2005)](@cite Stevens2005).

The net upward radiative flux ``ℐ`` is parameterized from the liquid water lying above
and below each height, plus a term representing free-tropospheric cooling by large-scale
divergence above the inversion,

```math
ℐ(z) = ℐ₀ \\, \\mathrm{e}^{-τ^{ℓw}(z, H)} + ℐ₁ \\, \\mathrm{e}^{-τ^{ℓw}(0, z)}
     + ρᵢ \\, cᵖ D \\left [ \\frac{(z - zᵢ)^{4/3}}{4} + zᵢ (z - zᵢ)^{1/3} \\right ] ,
```

where the longwave optical thickness of the intervening liquid water is

```math
τ^{ℓw}(a, b) = κˡ ∫_a^b ρ \\, qˡ \\, \\mathrm{d} z ,
```

``H`` is the top of the column, ``zᵢ`` the inversion height and ``ρᵢ`` the density there.
The first two terms respond to the liquid water the simulation actually produces: ``ℐ₀``
sets the cloud-top cooling and ``ℐ₁`` the cloud-base warming. The third acts only above
the inversion, where its cooling is constructed to balance the warming by the prescribed
subsidence ``w = -D z``, and so to *maintain* the ``(z - zᵢ)^{1/3}`` structure of
``θ^{ℓi}`` that the initial condition prescribes.

The inversion is diagnosed in each column as the level whose total moisture ``qᵗ`` lies
closest to `inversion_moisture`, resolving ties upward so that a well-mixed layer puts
``zᵢ`` at its top rather than at the surface.

What the thermodynamic equation receives is the flux divergence ``-∂_z ℐ``, held in
`flux_divergence` as a volumetric energy source in W m⁻³.

$(TYPEDFIELDS)
"""
struct DYCOMSRadiation{FT, F, C, I, S}
    "Cloud-top cooling amplitude ``ℐ₀`` (W m⁻²)"
    cloud_top_cooling :: FT
    "Cloud-base warming amplitude ``ℐ₁`` (W m⁻²)"
    cloud_base_warming :: FT
    "Mass absorption coefficient of liquid water ``κˡ`` (m² kg⁻¹)"
    absorption_coefficient :: FT
    "Large-scale horizontal divergence ``D`` (s⁻¹)"
    divergence :: FT
    "Isobaric heat capacity ``cᵖ`` in the free-tropospheric term (J kg⁻¹ K⁻¹)"
    heat_capacity :: FT
    "Total moisture ``qᵗ`` that locates the inversion height ``zᵢ`` (kg kg⁻¹)"
    inversion_moisture :: FT
    "Net upward radiative flux ``ℐ`` (W m⁻²) at cell interfaces"
    net_upward_flux :: F
    "Radiative flux divergence ``-∂_z ℐ`` (W m⁻³) at cell centers"
    flux_divergence :: C
    "Diagnosed inversion height ``zᵢ`` (m) in each column"
    inversion_height :: I
    "Schedule on which the radiative fluxes are recomputed"
    schedule :: S
end

"""
$(TYPEDSIGNATURES)

Return a [`DYCOMSRadiation`](@ref) on `grid`.

The defaults are the parameters recommended for DYCOMS-II RF01 by
[Stevens (2005)](@cite Stevens2005), Table 3: ``ℐ₀ = 70`` W m⁻², ``ℐ₁ = 22`` W m⁻²,
``κˡ = 85`` m² kg⁻¹ and ``D = 3.75 × 10^{-6}`` s⁻¹, with the inversion located by the
``qᵗ = 8`` g kg⁻¹ contour.

```jldoctest
using Breeze

grid = RectilinearGrid(size=(4, 4, 8), x=(0, 1), y=(0, 1), z=(0, 1500))
radiation = DYCOMSRadiation(grid)

# output
DYCOMSRadiation:
├── cloud top cooling ℐ₀: 70.0 W m⁻²
├── cloud base warming ℐ₁: 22.0 W m⁻²
├── absorption coefficient κˡ: 85.0 m² kg⁻¹
├── divergence D: 3.75e-6 s⁻¹
└── inversion moisture qᵗ: 0.008 kg kg⁻¹
```
"""
function DYCOMSRadiation(grid;
                         cloud_top_cooling = 70,
                         cloud_base_warming = 22,
                         absorption_coefficient = 85,
                         divergence = 3.75e-6,
                         heat_capacity = 1015,
                         inversion_moisture = 8e-3,
                         schedule = IterationInterval(1))

    FT = eltype(grid)
    net_upward_flux = Field{Center, Center, Face}(grid)
    flux_divergence = CenterField(grid)
    inversion_height = Field{Center, Center, Nothing}(grid)

    return DYCOMSRadiation(convert(FT, cloud_top_cooling),
                           convert(FT, cloud_base_warming),
                           convert(FT, absorption_coefficient),
                           convert(FT, divergence),
                           convert(FT, heat_capacity),
                           convert(FT, inversion_moisture),
                           net_upward_flux,
                           flux_divergence,
                           inversion_height,
                           schedule)
end

function Base.show(io::IO, radiation::DYCOMSRadiation)
    print(io, "DYCOMSRadiation:", '\n',
              "├── cloud top cooling ℐ₀: ", prettysummary(radiation.cloud_top_cooling), " W m⁻²", '\n',
              "├── cloud base warming ℐ₁: ", prettysummary(radiation.cloud_base_warming), " W m⁻²", '\n',
              "├── absorption coefficient κˡ: ", prettysummary(radiation.absorption_coefficient), " m² kg⁻¹", '\n',
              "├── divergence D: ", prettysummary(radiation.divergence), " s⁻¹", '\n',
              "└── inversion moisture qᵗ: ", prettysummary(radiation.inversion_moisture), " kg kg⁻¹")
end

"""
$(TYPEDSIGNATURES)

Return the net upward radiative flux ``ℐ`` at height `z`, given the optical thickness
`τ₀ᶻ` between the surface and `z`, the optical thickness `τ₀ᴴ` of the whole column, the
inversion height `zᵢ`, and the density `ρᵢ` there.
"""
@inline function dycoms_radiative_flux(z, τ₀ᶻ, τ₀ᴴ, zᵢ, ρᵢ, ℐ₀, ℐ₁, cᵖ, D)
    # The optical thickness above z is τˡʷ(z, H) = τˡʷ(0, H) - τˡʷ(0, z)
    cloud_top = ℐ₀ * exp(τ₀ᶻ - τ₀ᴴ)
    cloud_base = ℐ₁ * exp(-τ₀ᶻ)

    # `max` confines the free-tropospheric term to z > zᵢ, and keeps the cube root real.
    # The bracket (z - zᵢ)^{4/3}/4 + zᵢ (z - zᵢ)^{1/3} factors as (z - zᵢ)^{1/3} (ζ/4 + zᵢ),
    # which needs only one cube root.
    ζ = max(0, z - zᵢ)
    free_troposphere = ρᵢ * cᵖ * D * cbrt(ζ) * (ζ / 4 + zᵢ)

    return cloud_top + cloud_base + free_troposphere
end

@kernel function _compute_dycoms_radiation!(flux_divergence, net_upward_flux, inversion_height, grid,
                                           total_density, qˡ, qᵛ, ℐ₀, ℐ₁, κˡ, cᵖ, D, qᵗᵢ)

    i, j = @index(Global, NTuple)

    FT = eltype(grid)
    Nz = size(grid, 3)

    @inbounds begin
        # First pass up the column: the optical thickness of the whole column, and the
        # inversion height, taken as the level whose qᵗ lies closest to qᵗᵢ.
        τ₀ᴴ = zero(FT)
        zᵢ = zero(FT)
        ρᵢ = zero(FT)
        δqᵗ = convert(FT, Inf)

        for k in 1:Nz
            ρ = total_density[i, j, k]
            Δz = Δzᶜᶜᶜ(i, j, k, grid)
            τ₀ᴴ += κˡ * ρ * qˡ[i, j, k] * Δz

            qᵗ = qᵛ[i, j, k] + qˡ[i, j, k]
            δ = abs(qᵗ - qᵗᵢ)

            # `≤` rather than `<` so that ties resolve to the *highest* tied level. The
            # initial state is uniform below the inversion, so every level in the mixed
            # layer is exactly equidistant from qᵗᵢ; keeping the first would put zᵢ at the
            # surface. This matches the reference implementations, which resolve ties
            # upward and so place zᵢ at the top of the mixed layer.
            closer = δ <= δqᵗ
            δqᵗ = ifelse(closer, δ, δqᵗ)
            zᵢ = ifelse(closer, znode(i, j, k, grid, Center(), Center(), Center()), zᵢ)
            ρᵢ = ifelse(closer, ρ, ρᵢ)
        end

        inversion_height[i, j, 1] = zᵢ

        # Second pass: the flux at each interface, accumulating the optical thickness
        # τˡʷ(0, z) from the surface as we climb.
        τ₀ᶻ = zero(FT)

        for k in 1:Nz
            z = znode(i, j, k, grid, Center(), Center(), Face())
            net_upward_flux[i, j, k] = dycoms_radiative_flux(z, τ₀ᶻ, τ₀ᴴ, zᵢ, ρᵢ, ℐ₀, ℐ₁, cᵖ, D)
            τ₀ᶻ += κˡ * total_density[i, j, k] * qˡ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
        end

        zᴴ = znode(i, j, Nz + 1, grid, Center(), Center(), Face())
        net_upward_flux[i, j, Nz + 1] = dycoms_radiative_flux(zᴴ, τ₀ᶻ, τ₀ᴴ, zᵢ, ρᵢ, ℐ₀, ℐ₁, cᵖ, D)

        # Radiative heating is minus the divergence of the net upward flux.
        for k in 1:Nz
            Δℐ = net_upward_flux[i, j, k + 1] - net_upward_flux[i, j, k]
            flux_divergence[i, j, k] = - Δℐ / Δzᶜᶜᶜ(i, j, k, grid)
        end
    end
end

function _update_radiation!(radiation::DYCOMSRadiation, model)
    grid = model.grid
    arch = architecture(grid)

    qˡ = model.microphysical_fields.qˡ
    qᵛ = model.microphysical_fields.qᵛ

    launch!(arch, grid, :xy,
            _compute_dycoms_radiation!,
            radiation.flux_divergence,
            radiation.net_upward_flux,
            radiation.inversion_height,
            grid,
            total_density(model.dynamics),
            qˡ, qᵛ,
            radiation.cloud_top_cooling,
            radiation.cloud_base_warming,
            radiation.absorption_coefficient,
            radiation.heat_capacity,
            radiation.divergence,
            radiation.inversion_moisture)

    return nothing
end
