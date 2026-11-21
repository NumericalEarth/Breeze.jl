#####
##### PressureLevelVerticalDiscretization — a grid-owned hybrid sigma-pressure vertical coordinate.
#####
##### The face pressures are p = A + B pˢ. The grid stores the reference coordinate r ∈ (0, 1)
##### (uniform, one unit per column), the one-dimensional hybrid coefficients A and B at the faces,
##### and the two-dimensional surface pressure pˢ of the current and of the previous stage. The
##### physical depth of a layer is Δz = Δp / (g ρ) with the cell density ρ, a three-dimensional
##### array the dynamics diagnoses. Nothing else is stored: every metric is derived on the fly
##### (see pressure_level_metrics.jl). The data arrays are raw arrays, as the free-surface
##### displacement `ηⁿ` of Oceananigans' `MutableVerticalDiscretization`, so that the dynamics
##### can build its `pˢ` and `ρ` fields on them without a field inside the grid.
#####

struct PressureLevelVerticalDiscretization{C, D, E, F, AB, PS, R, FT} <: AbstractVerticalCoordinate
    cᵃᵃᶠ :: C                     # face-centered reference coordinate r
    cᵃᵃᶜ :: D                     # cell-centered reference coordinate r
    Δᵃᵃᶠ :: E                     # face-centered reference spacing Δr
    Δᵃᵃᶜ :: F                     # cell-centered reference spacing Δr
    Δ⁻¹ᵃᵃᶠ :: E                   # reciprocal of the face-centered reference spacing
    Δ⁻¹ᵃᵃᶜ :: F                   # reciprocal of the cell-centered reference spacing
    pressure_coefficient :: AB    # A at the z-faces (Pa), with halo faces
    surface_pressure_coefficient :: AB  # B at the z-faces, with halo faces
    surface_pressure :: PS        # pˢ of the current stage (Center, Center, Nothing)
    previous_surface_pressure :: PS     # pˢ at the start of the stage (Center, Center, Nothing)
    density :: R                  # cell density ρ (kg m⁻³), diagnosed by the dynamics
    gravitational_acceleration :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Skeleton constructor of a hybrid sigma-pressure vertical discretization with the hybrid
coefficients `A` (Pa) and `B` (dimensionless) at the `Nz + 1` z-faces, from the ground (`A[1] = 0`,
`B[1] = 1`) to the model top (`B[end] = 0`, a surface of constant pressure `A[end]`). The face
pressures of a column with surface pressure `pˢ` are `A + B pˢ`.

The surface pressure and the cell density are allocated when the grid is built and owned by the
dynamics afterwards; until the dynamics sets them, the surface pressure is
`reference_surface_pressure` and the density that of an isothermal atmosphere at
`reference_temperature` with the gas constant `reference_gas_constant`, which only serve to give
the grid finite physical depths before a model exists.
"""
function PressureLevelVerticalDiscretization(A, B;
                                             gravitational_acceleration = 9.81,
                                             reference_surface_pressure = 1e5,
                                             reference_temperature = 250,
                                             reference_gas_constant = 287)
    length(A) == length(B) || throw(ArgumentError("A and B must have the same length (the number of z-faces)"))
    length(A) ≥ 2 || throw(ArgumentError("A and B need at least two z-faces"))
    (A[1] == 0 && B[1] == 1) || throw(ArgumentError("the lowest z-face must be the ground: A[1] = 0 and B[1] = 1"))
    B[end] == 0 || throw(ArgumentError("the highest z-face must be a surface of constant pressure: B[end] = 0"))
    A[end] > 0 || throw(ArgumentError("the model top must have a positive pressure A[end]"))

    pˢ = reference_surface_pressure
    Δp = [(A[k] - A[k+1]) + (B[k] - B[k+1]) * pˢ for k in 1:length(A)-1]
    all(>(0), Δp) || throw(ArgumentError("every layer must have a positive pressure thickness at pˢ = $pˢ, got Δp = $Δp"))

    Nz = length(A) - 1
    r_faces = collect(range(0, 1, length = Nz + 1))
    reference = (; surface_pressure = reference_surface_pressure, temperature = reference_temperature,
                   gas_constant = reference_gas_constant)

    return PressureLevelVerticalDiscretization(r_faces, nothing, nothing, nothing, nothing, nothing,
                                               collect(Float64, A), collect(Float64, B),
                                               nothing, nothing, reference, gravitational_acceleration)
end

const PLVD = PressureLevelVerticalDiscretization

"""
    PressureLevelGrid

Any `AbstractUnderlyingGrid` whose vertical coordinate is a
[`PressureLevelVerticalDiscretization`](@ref); `z` is `Bounded`.
"""
const PressureLevelGrid = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Bounded, <:PLVD}

Grids.coordinate_summary(::Bounded, z::PLVD, name) = "hybrid sigma-pressure"

function Grids.validate_dimension_specification(T, ξ::PLVD, dir, N, FT)
    length(ξ.pressure_coefficient) == N + 1 ||
        throw(ArgumentError("PressureLevelVerticalDiscretization has $(length(ξ.pressure_coefficient) - 1) layers, the grid $N"))
    cᶠ = Grids.validate_dimension_specification(T, ξ.cᵃᵃᶠ, dir, N, FT)
    return PressureLevelVerticalDiscretization(cᶠ, ξ.cᵃᵃᶜ, ξ.Δᵃᵃᶠ, ξ.Δᵃᵃᶜ, ξ.Δ⁻¹ᵃᵃᶠ, ξ.Δ⁻¹ᵃᵃᶜ,
                                               ξ.pressure_coefficient, ξ.surface_pressure_coefficient,
                                               ξ.surface_pressure, ξ.previous_surface_pressure, ξ.density,
                                               ξ.gravitational_acceleration)
end

function map_coordinate(f, z::PLVD)
    return PressureLevelVerticalDiscretization(f(z.cᵃᵃᶠ), f(z.cᵃᵃᶜ), f(z.Δᵃᵃᶠ), f(z.Δᵃᵃᶜ), f(z.Δ⁻¹ᵃᵃᶠ), f(z.Δ⁻¹ᵃᵃᶜ),
                                               f(z.pressure_coefficient), f(z.surface_pressure_coefficient),
                                               f(z.surface_pressure), f(z.previous_surface_pressure), f(z.density),
                                               z.gravitational_acceleration)
end

Adapt.adapt_structure(to, z::PLVD) = map_coordinate(x -> Adapt.adapt(to, x), z)
Oceananigans.Architectures.on_architecture(arch, z::PLVD) = map_coordinate(x -> on_architecture(arch, x), z)

# Face coefficients with halo faces: the layers of the halos repeat the thicknesses ΔA and ΔB of the
# interior layers they mirror, so that every halo layer has the pressure thickness of its image
function coefficients_with_halos(FT, c, Hz)
    Nz = length(c) - 1
    Δ = [c[k] - c[k+1] for k in 1:Nz]                      # thickness of layer k (positive downward ordering)
    mirror(k) = k < 1 ? Δ[min(Nz, 1 - k)] : k > Nz ? Δ[max(1, 2Nz + 1 - k)] : Δ[k]
    faces = zeros(FT, Nz + 1 + 2Hz)
    offset = Hz                                            # faces[k + offset] is face k
    for k in 1:Nz+1
        faces[k + offset] = c[k]
    end
    for k in 0:-1:1-Hz                                     # below the ground: c[k] = c[k+1] + Δ(k)
        faces[k + offset] = faces[k + 1 + offset] + mirror(k)
    end
    for k in Nz+2:Nz+1+Hz                                  # above the top: c[k] = c[k-1] - Δ(k-1)
        faces[k + offset] = faces[k - 1 + offset] - mirror(k - 1)
    end
    return OffsetArray(faces, -offset)
end

function Grids.generate_coordinate(FT, topo, size, halo, coordinate::PLVD, coordinate_name, dim::Int, arch)
    dim == 3 || throw(ArgumentError("PressureLevelVerticalDiscretization is only supported in the z (third) dimension"))
    coordinate_name == :z || throw(ArgumentError("PressureLevelVerticalDiscretization is only supported for the z-coordinate"))

    Nz = size[3]; Hz = halo[3]
    Lz, rᵃᵃᶠ, rᵃᵃᶜ, Δrᵃᵃᶠ, Δrᵃᵃᶜ = Grids.generate_coordinate(FT, topo[3](), Nz, Hz, coordinate.cᵃᵃᶠ, :r, arch)
    Δr⁻¹ᵃᵃᶠ, Δr⁻¹ᵃᵃᶜ = map((Δrᵃᵃᶠ, Δrᵃᵃᶜ)) do Δ
        on_architecture(arch, Grids.reciprocal_metric(on_architecture(CPU(), Δ)))
    end

    A = coefficients_with_halos(FT, on_architecture(CPU(), coordinate.pressure_coefficient), Hz)
    B = coefficients_with_halos(FT, on_architecture(CPU(), coordinate.surface_pressure_coefficient), Hz)
    A = on_architecture(arch, A)
    B = on_architecture(arch, B)

    args = (topo, size, halo)
    pˢ = new_data(FT, arch, (Center, Center, Nothing), args...)
    pˢ⁻ = new_data(FT, arch, (Center, Center, Nothing), args...)
    ρ = new_data(FT, arch, (Center, Center, Center), args...)
    initialize_state!(pˢ, pˢ⁻, ρ, coordinate, A, B, FT)

    g = convert(FT, coordinate.gravitational_acceleration)
    z = PressureLevelVerticalDiscretization(rᵃᵃᶠ, rᵃᵃᶜ, Δrᵃᵃᶠ, Δrᵃᵃᶜ, Δr⁻¹ᵃᵃᶠ, Δr⁻¹ᵃᵃᶜ, A, B, pˢ, pˢ⁻, ρ, g)
    return Lz, z
end

# A skeleton carries the reference state (a NamedTuple) in place of the surface pressure; a grid rebuilt
# from another grid (`cpu_face_constructor_z`) carries that grid's arrays, which are copied
function initialize_state!(pˢ, pˢ⁻, ρ, coordinate::PLVD{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing, <:NamedTuple}, A, B, FT)
    reference = coordinate.density
    fill!(pˢ, reference.surface_pressure)
    fill!(pˢ⁻, reference.surface_pressure)
    Ac = on_architecture(CPU(), A); Bc = on_architecture(CPU(), B)
    ρc = on_architecture(CPU(), ρ)
    for k in axes(ρc, 3)
        k₊ = clamp(k, firstindex(Ac), lastindex(Ac) - 1)
        p = (Ac[k₊] + Ac[k₊ + 1]) / 2 + (Bc[k₊] + Bc[k₊ + 1]) / 2 * reference.surface_pressure
        ρc[:, :, k] .= max(p, Ac[lastindex(Ac)] / 2) / (reference.gas_constant * reference.temperature)
    end
    copyto!(ρ, on_architecture(Oceananigans.Architectures.architecture(ρ), ρc))
    return nothing
end

function initialize_state!(pˢ, pˢ⁻, ρ, coordinate::PLVD, A, B, FT)
    copyto!(parent(pˢ), on_architecture(Oceananigans.Architectures.architecture(pˢ), convert(Array{FT}, Array(parent(coordinate.surface_pressure)))))
    copyto!(parent(pˢ⁻), on_architecture(Oceananigans.Architectures.architecture(pˢ⁻), convert(Array{FT}, Array(parent(coordinate.previous_surface_pressure)))))
    copyto!(parent(ρ), on_architecture(Oceananigans.Architectures.architecture(ρ), convert(Array{FT}, Array(parent(coordinate.density)))))
    return nothing
end

# A grid rebuilt from this one (`on_architecture(CPU(), grid)`, which `set!` uses) keeps the hybrid
# coefficients, the surface pressures and the density, as the terrain-following grid keeps its terrain
function Grids.cpu_face_constructor_z(grid::PressureLevelGrid)
    z = grid.z
    Nz = Base.size(grid, 3)
    interior_faces(c) = [on_architecture(CPU(), c)[k] for k in 1:Nz+1]
    return PressureLevelVerticalDiscretization(collect(range(0, 1, length = Nz + 1)), nothing, nothing, nothing, nothing, nothing,
                                               interior_faces(z.pressure_coefficient), interior_faces(z.surface_pressure_coefficient),
                                               on_architecture(CPU(), z.surface_pressure), on_architecture(CPU(), z.previous_surface_pressure),
                                               on_architecture(CPU(), z.density), z.gravitational_acceleration)
end
