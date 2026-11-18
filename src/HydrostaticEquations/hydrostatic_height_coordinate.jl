#####
##### The heights of the hydrostatic column as the vertical metric of a grid
#####
#
# The grid of `HydrostaticDynamics` labels coordinate surfaces: its vertical coordinate `r` is not
# a height, and the prognostic densities are masses per unit area and unit of `r`. With
# σ = Δz / Δr the stretching of a layer, in meters per unit of `r`,
#
#   ρ = Δp / (g Δr) = σ Δp / (g Δz) ,
#
# so the prognostic variables are the σ-weighted densities per cubic meter: the quantities that a
# z* coordinate evolves (Oceananigans' `MutableVerticalDiscretization`). The dynamics is written
# in the reference coordinate `r` and is not changed here. What this file adds is the other half
# of such a coordinate: a second grid, with every horizontal array of the model's grid, whose
# vertical spacings and nodes are the heights of the hydrostatic column. Kernel functions written
# for a height coordinate (turbulence closures, bulk formulas, sedimentation) are given that grid.
#
# The heights are a diagnostic of the state, computed with the geopotential in every
# `update_state!`, from the density potential temperature and the Exner functions that the
# pressure-gradient force reads. Nothing of them is history, so nothing is checkpointed.
#
# TODO (Oceananigans): `MutableVerticalDiscretization` holds a two-dimensional stretching σ(i, j)
# and the bookkeeping of z* (ηⁿ, σᶜᶜ⁻, ∂t_σ). The stretching of a pressure layer depends on the
# layer, so it does not fit. A vertical discretization with three-dimensional spacings and node
# heights that the model updates, with the operators `Δz`, `Δz⁻¹`, `znode` and
# `height_above_bottom` keyed on it for every underlying grid, belongs in Oceananigans: it would
# hold this coordinate, the `PressureLevelVerticalDiscretization` of NumericalEarth (heights from
# a prescribed geopotential) and terrain-following height coordinates. Also missing there: a
# function that returns a grid with another vertical discretization (`with_vertical_discretization`
# below rebuilds the grids from the positional signatures of their constructors, and breaks when
# those change), areas and volumes in the reference coordinate, and `znodes` of a grid whose
# heights differ from column to column.

"""
$(TYPEDEF)

Vertical discretization of a grid whose heights are those of the hydrostatic column of
`HydrostaticDynamics`: the depth of a layer is `ΔΦ / g` across it, from the hydrostatic integral
with the density potential temperature.

The reference coordinate (`rnode`, `Δr`) is the vertical coordinate of the model's grid, a label
of the coordinate surfaces. The heights are diagnosed in every `update_state!` and are valid for
the state it diagnosed.
"""
struct HydrostaticPressureVerticalDiscretization{C, D, E, F, DC, DF, G, GS, FT} <: AbstractVerticalCoordinate
    cᵃᵃᶠ :: C                         # reference coordinate of the z-faces (a label)
    cᵃᵃᶜ :: D                         # reference coordinate of the cell centers
    Δᵃᵃᶠ :: E                         # reference spacing at z-faces
    Δᵃᵃᶜ :: F                         # reference spacing at cell centers
    Δ⁻¹ᵃᵃᶠ :: E                       # reciprocal of the reference spacing at z-faces
    Δ⁻¹ᵃᵃᶜ :: F                       # reciprocal of the reference spacing at cell centers
    layer_depth :: DC                 # depth of the layers, at cell centers, m
    interface_spacing :: DF           # height between two layers, at z-faces, m
    interface_height :: DF            # height of the z-faces above the ground, m
    geopotential :: G                 # Φ of the layers, at cell centers, m² s⁻²
    surface_geopotential :: GS        # Φˢ of the ground (two-dimensional), m² s⁻²
    gravitational_acceleration :: FT  # g, m s⁻²
end

Adapt.adapt_structure(to, z::HydrostaticPressureVerticalDiscretization) =
    HydrostaticPressureVerticalDiscretization(adapt(to, z.cᵃᵃᶠ),
                                              adapt(to, z.cᵃᵃᶜ),
                                              adapt(to, z.Δᵃᵃᶠ),
                                              adapt(to, z.Δᵃᵃᶜ),
                                              adapt(to, z.Δ⁻¹ᵃᵃᶠ),
                                              adapt(to, z.Δ⁻¹ᵃᵃᶜ),
                                              adapt(to, z.layer_depth),
                                              adapt(to, z.interface_spacing),
                                              adapt(to, z.interface_height),
                                              adapt(to, z.geopotential),
                                              adapt(to, z.surface_geopotential),
                                              z.gravitational_acceleration)

Base.summary(::HydrostaticPressureVerticalDiscretization) = "HydrostaticPressureVerticalDiscretization"

Base.show(io::IO, z::HydrostaticPressureVerticalDiscretization) =
    print(io, "HydrostaticPressureVerticalDiscretization: heights of the hydrostatic column, ",
              "g = ", prettysummary(z.gravitational_acceleration), " m s⁻²")

const HPVD = HydrostaticPressureVerticalDiscretization

const HydrostaticPressureRectilinearGrid = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:HPVD}
const HydrostaticPressureLatitudeLongitudeGrid = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:HPVD}
const HydrostaticPressureOrthogonalSphericalShellGrid = OrthogonalSphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:HPVD}

"""
    HydrostaticPressureGrid

A grid whose vertical discretization is a [`HydrostaticPressureVerticalDiscretization`](@ref):
its `Δz` and `znode` are heights of the hydrostatic column, in meters.
"""
const HydrostaticPressureGrid = Union{HydrostaticPressureRectilinearGrid,
                                      HydrostaticPressureLatitudeLongitudeGrid,
                                      HydrostaticPressureOrthogonalSphericalShellGrid}

#####
##### Spacings and nodes
#####

# Layer depth and height between layers at the horizontal location of the cell centers; the
# averages below bring them to the velocity points.
@inline layer_depthᶜᶜᶜ(i, j, k, grid) = @inbounds grid.z.layer_depth[i, j, k]
@inline interface_spacingᶜᶜᶠ(i, j, k, grid) = @inbounds grid.z.interface_spacing[i, j, k]

for G in (:HydrostaticPressureRectilinearGrid,
          :HydrostaticPressureLatitudeLongitudeGrid,
          :HydrostaticPressureOrthogonalSphericalShellGrid)

    @eval begin
        @inline Operators.Δzᵃᵃᶜ(i, j, k, grid::$G) = layer_depthᶜᶜᶜ(i, j, k, grid)
        @inline Operators.Δzᵃᵃᶠ(i, j, k, grid::$G) = interface_spacingᶜᶜᶠ(i, j, k, grid)

        @inline Operators.Δzᶜᶜᶜ(i, j, k, grid::$G) = layer_depthᶜᶜᶜ(i, j, k, grid)
        @inline Operators.Δzᶜᶜᶠ(i, j, k, grid::$G) = interface_spacingᶜᶜᶠ(i, j, k, grid)
        @inline Operators.Δzᶠᶜᶜ(i, j, k, grid::$G) = ℑxᶠᵃᵃ(i, j, k, grid, grid.z.layer_depth)
        @inline Operators.Δzᶜᶠᶜ(i, j, k, grid::$G) = ℑyᵃᶠᵃ(i, j, k, grid, grid.z.layer_depth)
        @inline Operators.Δzᶠᶜᶠ(i, j, k, grid::$G) = ℑxᶠᵃᵃ(i, j, k, grid, grid.z.interface_spacing)
        @inline Operators.Δzᶜᶠᶠ(i, j, k, grid::$G) = ℑyᵃᶠᵃ(i, j, k, grid, grid.z.interface_spacing)
        @inline Operators.Δzᶠᶠᶜ(i, j, k, grid::$G) = ℑxyᶠᶠᵃ(i, j, k, grid, grid.z.layer_depth)
        @inline Operators.Δzᶠᶠᶠ(i, j, k, grid::$G) = ℑxyᶠᶠᵃ(i, j, k, grid, grid.z.interface_spacing)

        @inline Operators.Δz⁻¹ᶜᶜᶜ(i, j, k, grid::$G) = 1 / Operators.Δzᶜᶜᶜ(i, j, k, grid)
        @inline Operators.Δz⁻¹ᶜᶜᶠ(i, j, k, grid::$G) = 1 / Operators.Δzᶜᶜᶠ(i, j, k, grid)
        @inline Operators.Δz⁻¹ᶠᶜᶜ(i, j, k, grid::$G) = 1 / Operators.Δzᶠᶜᶜ(i, j, k, grid)
        @inline Operators.Δz⁻¹ᶜᶠᶜ(i, j, k, grid::$G) = 1 / Operators.Δzᶜᶠᶜ(i, j, k, grid)
        @inline Operators.Δz⁻¹ᶠᶜᶠ(i, j, k, grid::$G) = 1 / Operators.Δzᶠᶜᶠ(i, j, k, grid)
        @inline Operators.Δz⁻¹ᶜᶠᶠ(i, j, k, grid::$G) = 1 / Operators.Δzᶜᶠᶠ(i, j, k, grid)
        @inline Operators.Δz⁻¹ᶠᶠᶜ(i, j, k, grid::$G) = 1 / Operators.Δzᶠᶠᶜ(i, j, k, grid)
        @inline Operators.Δz⁻¹ᶠᶠᶠ(i, j, k, grid::$G) = 1 / Operators.Δzᶠᶠᶠ(i, j, k, grid)

        # Heights above the geoid, of the column of the cell centers whatever the horizontal
        # location: Φ / g of a layer (the height the pressure-gradient force uses), and
        # Φˢ / g plus the height above the ground of a coordinate surface.
        @inline Grids.znode(i, j, k, grid::$G, ℓx, ℓy, ::Center) =
            @inbounds grid.z.geopotential[i, j, k] / grid.z.gravitational_acceleration

        @inline Grids.znode(i, j, k, grid::$G, ℓx, ℓy, ::Face) =
            @inbounds grid.z.surface_geopotential[i, j, 1] / grid.z.gravitational_acceleration +
                      grid.z.interface_height[i, j, k]

        # As for a height coordinate (Oceananigans.TurbulenceClosures), without subtracting the
        # height of the ground from the height of the node: the height of a face is not less than
        # the depth of the layer below it, nor the height of a center less than half its layer.
        @inline function OceanTurbulenceClosures.height_above_bottomᶜᶜᶠ(i, j, k, grid::$G)
            h = @inbounds grid.z.interface_height[i, j, k]
            return max(Operators.Δzᶜᶜᶜ(i, j, k-1, grid), h)
        end

        @inline function OceanTurbulenceClosures.height_above_bottomᶜᶜᶜ(i, j, k, grid::$G)
            h = @inbounds (grid.z.interface_height[i, j, k] + grid.z.interface_height[i, j, k+1]) / 2
            return max(Operators.Δzᶜᶜᶜ(i, j, k, grid) / 2, h)
        end
    end
end

#####
##### A grid with another vertical discretization
#####

grid_size_type(grid) = last(typeof(grid).parameters)

fields_with_vertical_discretization(grid, z) =
    map(name -> name === :z ? z : getfield(grid, name), fieldnames(typeof(grid)))

function with_vertical_discretization(grid::RectilinearGrid, z)
    TX, TY, TZ = topology(grid)
    SZ = grid_size_type(grid)
    return RectilinearGrid{TX, TY, TZ, SZ}(fields_with_vertical_discretization(grid, z)...)
end

function with_vertical_discretization(grid::LatitudeLongitudeGrid, z)
    TX, TY, TZ = topology(grid)
    SZ = grid_size_type(grid)
    return LatitudeLongitudeGrid{TX, TY, TZ, SZ}(fields_with_vertical_discretization(grid, z)...)
end

function with_vertical_discretization(grid::OrthogonalSphericalShellGrid, z)
    FT = eltype(grid)
    TX, TY, TZ = topology(grid)
    SZ = grid_size_type(grid)
    return OrthogonalSphericalShellGrid{FT, TX, TY, TZ, SZ}(fields_with_vertical_discretization(grid, z)...)
end

with_vertical_discretization(grid, z) =
    throw(ArgumentError("The physical height grid of HydrostaticDynamics is not supported on $(summary(grid))."))

# The grid of one region, with the data of the fields of that region
function hydrostatic_pressure_grid(grid, Δzᶜ, Δzᶠ, zᶠ, Φ, Φˢ, g)
    r = grid.z
    z = HydrostaticPressureVerticalDiscretization(r.cᵃᵃᶠ, r.cᵃᵃᶜ, r.Δᵃᵃᶠ, r.Δᵃᵃᶜ, r.Δ⁻¹ᵃᵃᶠ, r.Δ⁻¹ᵃᵃᶜ,
                                                  Δzᶜ.data, Δzᶠ.data, zᶠ.data, Φ.data, Φˢ.data,
                                                  convert(eltype(grid), g))
    return with_vertical_discretization(grid, z)
end

function hydrostatic_pressure_grid(grid::MultiRegion.MultiRegionGrid, Δzᶜ, Δzᶠ, zᶠ, Φ, Φˢ, g)
    region_grids = construct_regionally(hydrostatic_pressure_grid, grid, Δzᶜ, Δzᶠ, zᶠ, Φ, Φˢ, g)
    FT = eltype(grid)
    TX, TY, TZ = topology(grid)
    CZ = typeof(getregion(region_grids, 1).z)
    return MultiRegion.MultiRegionGrid{FT, TX, TY, TZ, CZ}(grid.architecture, grid.partition, grid.connectivity,
                                                           region_grids)
end

#####
##### The height coordinate of a model
#####

"""
$(TYPEDEF)

The heights of the hydrostatic column of `HydrostaticDynamics` and the grid that has them for
its vertical metric, a [`HydrostaticPressureGrid`](@ref): `grid` shares every horizontal array
with the grid of the model, and fields of the model are indexed alike on both.
"""
struct HydrostaticHeightCoordinate{G, C, F}
    grid :: G               # the grid of the model with heights for its vertical metric
    layer_depth :: C        # depth of the layers, at cell centers, m
    interface_spacing :: F  # height between two layers, at z-faces, m
    interface_height :: F   # height of the z-faces above the ground, m
end

# Kernels are given the grid, never the coordinate through the dynamics
Adapt.adapt_structure(to, ::HydrostaticHeightCoordinate) = nothing

@regional HydrostaticHeightCoordinate

Base.summary(::HydrostaticHeightCoordinate) = "HydrostaticHeightCoordinate"
Base.show(io::IO, coordinate::HydrostaticHeightCoordinate) =
    print(io, "HydrostaticHeightCoordinate on ", summary(coordinate.grid))

materialize_height_coordinate(enabled::Bool, grid, Φ, Φˢ, constants) =
    enabled ? build_height_coordinate(grid, Φ, Φˢ, constants) : nothing

function build_height_coordinate(grid, Φ, Φˢ, constants)
    grid.z isa StaticVerticalDiscretization ||
        throw(ArgumentError("The physical height grid of HydrostaticDynamics needs a grid with a static " *
                            "vertical discretization, got $(summary(grid))."))

    Δzᶜ = CenterField(grid)
    Δzᶠ = ZFaceField(grid)
    zᶠ = ZFaceField(grid)
    height_grid = hydrostatic_pressure_grid(grid, Δzᶜ, Δzᶠ, zᶠ, Φ, Φˢ, constants.gravitational_acceleration)

    return HydrostaticHeightCoordinate(height_grid, Δzᶜ, Δzᶠ, zᶠ)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the grid of `model` with the heights of its hydrostatic column for a vertical metric, a
[`HydrostaticPressureGrid`](@ref), or throw an `ArgumentError` if the `HydrostaticDynamics` of
the model was not built with `physical_height_grid = true`.

The heights are those of the last `update_state!`.
"""
function physical_height_grid(model::HydrostaticModel)
    has_physical_height_grid(model.dynamics) ||
        throw(ArgumentError("This model has no physical height grid: build its dynamics with " *
                            "HydrostaticDynamics(physical_height_grid=true)."))
    return model.dynamics.height_coordinate.grid
end

has_physical_height_grid(dynamics::HydrostaticDynamics) = !isnothing(dynamics.height_coordinate)

# The grid whose `Δzᶜᶜᶜ` is the depth of the layers in meters, if the model has one; otherwise
# the grid of the model, on which `layer_depth_in_meters` evaluates the hydrostatic integral.
height_grid_or(grid, ::Nothing) = grid
height_grid_or(grid, coordinate::HydrostaticHeightCoordinate) = coordinate.grid

# Depth of layer `k` in meters: read from a grid of heights, and otherwise the hydrostatic
# integral across the layer. The two are the same number wherever the heights are current.
@inline layer_depth_in_meters(i, j, k, grid, θᵨ, Πᶠ, cᵖᵈ, g) = physical_layer_depth(i, j, k, θᵨ, Πᶠ, cᵖᵈ, g)
@inline layer_depth_in_meters(i, j, k, grid::HydrostaticPressureGrid, θᵨ, Πᶠ, cᵖᵈ, g) = Δzᶜᶜᶜ(i, j, k, grid)

#####
##### Diagnosing the heights
#####

# Depth of the layers, height between them and height of the coordinate surfaces above the
# ground, from the hydrostatic integral with the density potential temperature. The spacings of
# the first and last face, through which nothing is exchanged, are half a layer.
@kernel function _compute_hydrostatic_heights!(Δzᶜ, Δzᶠ, zᶠ, grid, θᵨ, Π, Πᶠ, cᵖᵈ, g)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        zᶠ[i, j, 1] = zero(grid)

        for k in 1:Nz
            Δzᶜ[i, j, k] = physical_layer_depth(i, j, k, θᵨ, Πᶠ, cᵖᵈ, g)
            zᶠ[i, j, k+1] = zᶠ[i, j, k] + Δzᶜ[i, j, k]
        end

        Δzᶠ[i, j, 1] = Δzᶜ[i, j, 1] / 2

        for k in 2:Nz
            Δzᶠ[i, j, k] = height_between_layers(i, j, k, grid, θᵨ, Π, Πᶠ, cᵖᵈ, g)
        end

        Δzᶠ[i, j, Nz+1] = Δzᶜ[i, j, Nz] / 2
    end
end

compute_hydrostatic_heights!(::Nothing, grid, dynamics, θᵨ, constants) = nothing

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Diagnose the heights of the hydrostatic column from the density potential temperature `θᵨ` (the
potential temperature of dry air) and the Exner functions of `dynamics`, and fill their halos.
Called where the geopotential is diagnosed, in every `update_state!`.

On a multi-region grid the halos are interpolated from the other regions, like those of the
geopotential: they are not diagnosed again from the interpolated state.
"""
function compute_hydrostatic_heights!(coordinate::HydrostaticHeightCoordinate, grid, dynamics, θᵨ, constants)
    arch = grid.architecture
    cᵖᵈ = constants.dry_air.heat_capacity
    g = constants.gravitational_acceleration

    @apply_regionally launch!(arch, grid, :xy, _compute_hydrostatic_heights!,
                              coordinate.layer_depth, coordinate.interface_spacing, coordinate.interface_height,
                              grid, θᵨ, dynamics.exner_function, dynamics.interface_exner_function, cᵖᵈ, g)

    fill_halo_regions!(coordinate.layer_depth)
    fill_halo_regions!(coordinate.interface_spacing)
    fill_halo_regions!(coordinate.interface_height)

    return nothing
end
