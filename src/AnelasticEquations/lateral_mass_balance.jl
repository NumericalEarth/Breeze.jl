#####
##### Lateral mass balance for a laterally-bounded anelastic domain
#####
##### `FourierTridiagonalPoissonSolver` transforms a `Bounded` direction with a DCT, which imposes
##### homogeneous Neumann conditions on the pressure. The resulting discrete Poisson problem is
##### solvable only if the net mass flux through the lateral boundaries vanishes:
#####
#####     ∮ ρᵣ u⋅n̂ dA = 0.
#####
##### The solver zeroes the mean of the *solution* for uniqueness but never touches the right-hand
##### side, so a nonzero net flux is not rejected — it is spread through the domain as spurious
##### divergence. Prescribed lateral boundaries (a nested child driven by a parent) have no reason to
##### balance to round-off: an ERA5 large-scale divergence of 1e-5 s⁻¹ over a 6 km × 6 km × 2 km column
##### is a net ~7 × 10⁵ m³ s⁻¹.
#####
##### The imbalance is removed by correcting the boundary data — a uniform offset subtracted from the
##### outward normal momentum on every lateral face that is not an impenetrable wall — after the halo
##### fill that applied the boundary conditions. An Oceananigans `Callback` cannot interleave there, so
##### `AtmosphereModels` calls this directly at each fill site.
#####
##### It must run after EVERY momentum halo fill, not once per Runge-Kutta stage. A fill rewrites the
##### boundary faces from the prescribed data, so it silently reverts an earlier correction. Correcting
##### only inside `compute_pressure_correction!` — and letting `compute_velocities!` refill afterwards —
##### leaves the interior pressure-corrected for balanced walls while the walls carry their raw values,
##### so every boundary face ends the step with a flux discontinuity of exactly `offset`. A corner cell
##### has two prescribed normal faces and so carries twice that. Measured on a 16³ domain with
##### `west = 2`, `east = 1` over twenty steps, correcting at the first site alone leaves the edge cells
##### at `offset/Δx` and the corners at twice that — a corner/edge ratio of 2.000 — and max|w| worse than
##### applying no correction at all, 1.83 against 0.689 m s⁻¹. Correcting at both sites brings the net
##### flux to zero, max|∇⋅ρᵣu| to 6.0e−18 and max|w| to 0.0476 m s⁻¹. The correction is a pure function
##### of the freshly-filled boundary data, so reapplying it is a fixed point rather than a cumulative
##### subtraction: repeated application moves a wall by at most a few ulp of the data.

const LaterallyBoundedGrid = AbstractGrid{<:Any, Bounded, Bounded}

# An impenetrable wall carries no flux and must stay closed: only the other lateral faces absorb
# the imbalance. TODO: replace this file with Oceananigans' `enforce_net_zero_transport!` once it
# is public, masks immersed boundary faces, and lets imposed normal flow join the correction.
const ImpenetrableBC = BoundaryCondition{<:NormalFlow{Nothing}, Nothing}
open_face(bc) = true
open_face(::ImpenetrableBC) = false

# Which of the four lateral faces absorb the imbalance, as (west, east, south, north)
function open_lateral_faces(ρu, ρv)
    ubcs = ρu.boundary_conditions
    vbcs = ρv.boundary_conditions
    return open_face(ubcs.west), open_face(ubcs.east), open_face(vbcs.south), open_face(vbcs.north)
end

"""
$(TYPEDSIGNATURES)

Remove the net mass flux through the lateral boundaries, so the anelastic pressure Poisson problem is
solvable.

A no-op unless both horizontal directions are `Bounded`. A `Periodic` direction carries no net flux, and
the transform's mean mode handles it.

A `Flat` horizontal is a known gap rather than a case that needs no correction. A two-dimensional x–z or
y–z domain — one horizontal `Bounded`, the other `Flat` — has the same solvability condition over the two
faces it does have, and is left uncorrected here.
"""
AtmosphereModels.enforce_lateral_mass_balance!(::AnelasticDynamics, momentum, grid) = nothing

function AtmosphereModels.enforce_lateral_mass_balance!(::AnelasticDynamics, momentum,
                                                        grid::LaterallyBoundedGrid)
    ρu, ρv = momentum.ρu, momentum.ρv
    Nx, Ny, Nz = size(grid)

    open_faces = open_lateral_faces(ρu, ρv)
    imbalance, zonal_area, meridional_area = lateral_mass_imbalance(ρu, ρv, grid)
    west, east, south, north = open_faces
    area = (west + east) * zonal_area + (south + north) * meridional_area
    area > 0 || return nothing
    offset = imbalance / area

    arch = architecture(grid)
    launch!(arch, grid, KernelParameters(1:Ny, 1:Nz), _correct_zonal_faces!, ρu, Nx, west * offset, east * offset)
    launch!(arch, grid, KernelParameters(1:Nx, 1:Nz), _correct_meridional_faces!, ρv, Ny, south * offset, north * offset)

    return nothing
end

# Slice on the DEVICE. `Array(interior(ρu))[1, :, :]` would materialize the whole interior on the host
# first and throw all but one plane away — 215.7 MB per field at production size, against 1.4 MB for the
# four plane views, and there are six calls per step.
function lateral_boundary_planes(ρu, ρv, grid)
    Nx, Ny, _ = size(grid)

    west  = view(interior(ρu), 1, :, :)
    east  = view(interior(ρu), Nx + 1, :, :)
    south = view(interior(ρv), :, 1, :)
    north = view(interior(ρv), :, Ny + 1, :)

    return west, east, south, north
end

# Spacings must be read off a HOST copy of the grid. `xspacing(i, grid, Center())` on a GPU grid with a
# stretched direction is a scalar read of a device array — a kernel launch and a synchronization per
# face. `on_architecture` returns the grid itself when it already lives on the host, so this is free on
# a CPU grid; on a GPU grid it copies the coordinates once, which is the largest single piece of the
# call below.
#
# `xspacing`/`yspacing`/`zspacing` (singular) are the eager `Oceananigans.Operators` accessors that the
# lazy `xspacings`/`yspacings`/`zspacings` operations wrap, so the arithmetic is identical. The eager
# spelling is used because constructing one of those operations allocates a small struct that embeds the
# grid by value, so the lazy spelling adds one allocation per lookup on a GPU grid. On a CPU grid neither
# allocates.
host_grid(grid) = on_architecture(CPU(), grid)

"""
$(TYPEDSIGNATURES)

Net outward mass flux through the four lateral faces, and the area of one zonal (west or east)
and one meridional (south or north) face.

This method weights every boundary face by its own area, so it is exact for a grid stretched in any
direction. It pays for that with the four plane copies and a host loop over all of them: `2(Nx + Ny)Nz`
values is only 1.4 MB at production size, but the copies are strided device views and cost far more than
that byte count suggests: on a GPU they dominate the call, and the `XYRegularRG` method below reaches the
same answer without them, with identical spacings and host loop.

No anelastic model reaches this method. `FourierTridiagonalPoissonSolver` requires regularly spaced
transform directions, so a laterally-bounded anelastic grid is always an `XYRegularRG` or an
`ImmersedBoundaryGrid` over one, and both take the method below. It is kept, and tested, because
`lateral_mass_imbalance` is usable on its own.
"""
function lateral_mass_imbalance(ρu, ρv, grid)
    Nx, Ny, Nz = size(grid)
    FT = eltype(grid)

    cpu_grid = host_grid(grid)
    device_west, device_east, device_south, device_north = lateral_boundary_planes(ρu, ρv, grid)

    # `interior` builds its view indices at run time, so inference gives up and hands back `Any`, which
    # would make every element access below a dynamic dispatch. The planes carry the grid's float type
    # by construction, so assert it once here rather than 2(Nx + Ny)Nz times implicitly.
    west  = Array(device_west)::Matrix{FT}
    east  = Array(device_east)::Matrix{FT}
    south = Array(device_south)::Matrix{FT}
    north = Array(device_north)::Matrix{FT}

    flux = zero(FT)
    zonal_area = zero(FT)
    meridional_area = zero(FT)

    for k in 1:Nz
        Δzᵏ = zspacing(k, cpu_grid, Center())
        for j in 1:Ny
            ΔA = yspacing(j, cpu_grid, Center()) * Δzᵏ
            flux += (east[j, k] - west[j, k]) * ΔA
            zonal_area += ΔA
        end
        for i in 1:Nx
            ΔA = xspacing(i, cpu_grid, Center()) * Δzᵏ
            flux += (north[i, k] - south[i, k]) * ΔA
            meridional_area += ΔA
        end
    end

    return flux, zonal_area, meridional_area
end

"""
$(TYPEDSIGNATURES)

Uniform horizontal spacing lets `Δy` factor out of the sum over `j` (and `Δx` out of the sum over `i`),
so the wall-parallel reduction happens ON THE DEVICE and only a length-`Nz` column per wall pair crosses
to the host for the vertical (possibly stretched) weighting. Three transfers leave the device per call,
not two: the two reduction results at 596 bytes each, and the `on_architecture` copy of the `z`
coordinates above, at most 2552 bytes.

Every anelastic model on a laterally-bounded grid lands here, terrain included.

`sum(east .- west, dims = 1)` forms the difference before reducing it, so one dense `(Ny, Nz)` temporary
is allocated per wall pair — 0.72 MB per call at production size in Float32. Reducing each wall on its
own would avoid that, at a different summation order; it has not been measured.
"""
function lateral_mass_imbalance(ρu, ρv, grid::XYRegularRG)
    Nx, Ny, Nz = size(grid)
    FT = eltype(grid)

    cpu_grid = host_grid(grid)
    west, east, south, north = lateral_boundary_planes(ρu, ρv, grid)

    Δx = xspacing(1, cpu_grid, Center())
    Δy = yspacing(1, cpu_grid, Center())

    # Sum over the wall-parallel index on the device; only 2Nz values come back. The assertion sees past
    # `interior`, which inference cannot resolve.
    zonal      = Array(sum(east .- west, dims = 1))::Matrix{FT}
    meridional = Array(sum(north .- south, dims = 1))::Matrix{FT}

    flux = zero(FT)
    zonal_area = zero(FT)
    meridional_area = zero(FT)

    for k in 1:Nz
        Δzᵏ = zspacing(k, cpu_grid, Center())
        flux += (Δy * zonal[1, k] + Δx * meridional[1, k]) * Δzᵏ
        zonal_area += Ny * Δy * Δzᵏ
        meridional_area += Nx * Δx * Δzᵏ
    end

    return flux, zonal_area, meridional_area
end

# An `ImmersedBoundaryGrid` over a laterally-bounded grid is laterally bounded, but it is not an
# `XYRegularRG`, so without this it would take the general host loop above — a nested child over terrain
# is exactly that grid. The immersed boundary does not enter the solvability condition:
# `pressure_solver_grid` drops it, so every cell is active in the solve and the right-hand side
# telescopes to the full unmasked boundary flux however the interior is masked. Forwarding also skips
# copying `bottom_height` and `bottom_cell_index` to the host, which `on_architecture` would do on every
# call — 94 kB at production size on a CPU grid, and a device-to-host transfer of two Nx×Ny fields on a
# GPU grid.
lateral_mass_imbalance(ρu, ρv, grid::ImmersedBoundaryGrid) =
    lateral_mass_imbalance(ρu, ρv, grid.underlying_grid)

# Subtract the offset from the OUTWARD normal component on each face, so every face sheds the same
# momentum per unit area and the four contributions sum to the imbalance exactly. The correction is
# uniform in momentum, not in velocity: the velocity increment therefore scales as 1/ρᵣ(z), which is
# what keeps the flux integral algebraically exact.

@kernel function _correct_zonal_faces!(ρu, Nx, west_offset, east_offset)
    j, k = @index(Global, NTuple)
    @inbounds ρu[1, j, k] += west_offset
    @inbounds ρu[Nx + 1, j, k] -= east_offset
end

@kernel function _correct_meridional_faces!(ρv, Ny, south_offset, north_offset)
    i, k = @index(Global, NTuple)
    @inbounds ρv[i, 1, k] += south_offset
    @inbounds ρv[i, Ny + 1, k] -= north_offset
end
