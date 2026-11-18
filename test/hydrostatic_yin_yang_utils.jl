# Helpers shared by the tests of `HydrostaticDynamics` on a `YinYangGrid`. Not a test: removed from
# the test suite in `runtests.jl`.

using Breeze
using Breeze.HydrostaticEquations: HydrostaticConservationFixer, VerletSubsteps, hydrostatic_state_check
using Oceananigans
using Oceananigans: TracerTendencyCallsite
using Oceananigans.Architectures: on_architecture
using Oceananigans.MultiRegion: YinYangGrid, owned_weights
using Oceananigans.Operators: Azᶜᶜᶜ
using Oceananigans.Utils: getregion
using Printf

include(joinpath(@__DIR__, "hydrostatic_test_utils.jl"))

#####
##### Grids, time steppers and time steps
#####
#
# Two resolutions. The Yin-Yang panels have Nφ cells across their 90° core, so their cells are
# 90° / Nφ wide: 4.09° for Nφ = 22 and 2° for Nφ = 45. The latitude-longitude grid they are
# compared with has cells of 4° and 2°, between walls at 70°S and 70°N that replace the poles.
#
# The time step of SSP RK3 resolves the fastest gravity wave on both grids. With the Lamb speed
# c ≈ 340 m s⁻¹ and the largest wavenumber 2 √(Δx⁻² + Δy⁻²) of the C-grid, the stability limit
# √3 / ω of SSP RK3 is about 600 s on the 4.09° panels (Δx ≥ 280 km at their edges), 300 s on the
# 2° panels, 366 s on the 4° latitude-longitude grid (Δx = 152 km at the walls) and 183 s on the
# 2° one. The time steps of the two split-explicit candidates are those of DECISIONS D043.

const resolutions = ((name = "4°", Nφ = 22, latitude_longitude_size = (90, 35), Δt = 300),
                     (name = "2°", Nφ = 45, latitude_longitude_size = (180, 70), Δt = 150))

const schemes = ((name = "SSP RK3", kw = (;), Δt = nothing),
                 (name = "WS-RK3 + Verlet", kw = split_explicit(:WickerSkamarock3, VerletSubsteps()), Δt = 1200),
                 (name = "MIS3C + Verlet", kw = split_explicit(:MIS3C, VerletSubsteps()), Δt = 600))

time_step(scheme, resolution) = isnothing(scheme.Δt) ? resolution.Δt : scheme.Δt

const Nz = 10

yin_yang_grid(resolution; Nz=Nz) =
    YinYangGrid(default_arch, FT; size=(3 * resolution.Nφ, resolution.Nφ, Nz), z=(0, 1), halo=(4, 4, 3),
                radius=sphere_radius)

latitude_longitude_grid(resolution; Nz=Nz) =
    LatitudeLongitudeGrid(default_arch, FT; size=(resolution.latitude_longitude_size..., Nz),
                          longitude=(-180, 180), latitude=(-70, 70), z=(0, 1), radius=sphere_radius)

grid_name(grid) = grid isa YinYangGrid ? "Yin-Yang" : "latitude-longitude"

#####
##### Results
#####

# Every measurement is logged, and appended to the file named by the environment variable
# BREEZE_HYDROSTATIC_YIN_YANG_RESULTS when it is set, as tab-separated (test, case, quantity, value).
function record(test, case, quantity, value)
    line = @sprintf("%s\t%s\t%s\t%.6e", test, case, quantity, value)
    @info "RESULT\t$line"
    path = get(ENV, "BREEZE_HYDROSTATIC_YIN_YANG_RESULTS", "")
    if !isempty(path)
        open(io -> println(io, line), path, "a")
    end
    return value
end

# The directory of the results file, where the tests also leave the profiles and the field
# that figures are drawn from; `nothing` when no results are kept
function results_directory()
    path = get(ENV, "BREEZE_HYDROSTATIC_YIN_YANG_RESULTS", "")
    return isempty(path) ? nothing : dirname(path)
end

function record(test, case, measurements::NamedTuple)
    for (quantity, value) in pairs(measurements)
        record(test, case, string(quantity), value)
    end
    return measurements
end

#####
##### The cells of the sphere: panel by panel, with the area that each panel owns
#####

regions_of(grid) = grid isa YinYangGrid ? (1, 2) : (1,)
region(field, r) = field.grid isa YinYangGrid ? getregion(field, r) : field
region_grid(grid, r) = grid isa YinYangGrid ? getregion(grid, r) : grid

# The interior of `field` on every panel, cropped to the cells of the panel: a velocity
# component on a wall has one more face than there are cells
function panels(field)
    Nx, Ny, _ = size(field.grid)
    return map(r -> Array(interior(region(field, r)))[1:Nx, 1:Ny, :], regions_of(field.grid))
end

function horizontal_field(grid, f)
    field = Field{Center, Center, Nothing}(grid)
    set!(field, f)
    return field
end

cell_areas(grid) = map(regions_of(grid)) do r
    panel = on_architecture(CPU(), region_grid(grid, r))
    Nx, Ny, _ = size(panel)
    [Azᶜᶜᶜ(i, j, 1, panel) for i in 1:Nx, j in 1:Ny]
end

owned_fractions(grid) = grid isa YinYangGrid ? map(w -> w[:, :, 1], panels(owned_weights(grid))) :
                                               (ones(size(grid, 1), size(grid, 2)),)

# The seam band: the owned cells within `width` cells, in either index, of a cell that its panel
# does not own entirely, which is where the curve that divides the sphere between the panels
# runs. A latitude-longitude grid has no seam.
function seam_band(w; width=2)
    Nx, Ny = size(w)
    band = falses(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        is_owned(w[i, j]) || continue
        neighbors = view(w, max(1, i - width):min(Nx, i + width), max(1, j - width):min(Ny, j + width))
        band[i, j] = !all(entirely_owned, neighbors)
    end
    return band
end

# The owned fractions are closed-form integrals, exact to rounding error: a cell that its panel
# owns entirely has a fraction within a few eps of one
is_owned(w) = w > 1e-9
entirely_owned(w) = w > 1 - 1e-9

"""
    sphere_cells(grid)

The cell centers of `grid` panel by panel: longitude `λ` and latitude `φ` in degrees, the `area`
that the panel owns of each cell, whether the cell is `owned`, and whether it lies in the seam
`band` or in the `interior`, the owned cells outside the band.
"""
function sphere_cells(grid)
    λ = map(a -> a[:, :, 1], panels(horizontal_field(grid, (λ, φ) -> λ)))
    φ = map(a -> a[:, :, 1], panels(horizontal_field(grid, (λ, φ) -> φ)))
    w = owned_fractions(grid)
    area = map((w, Az) -> w .* Az, w, cell_areas(grid))
    owned = map(w -> is_owned.(w), w)
    band = map(seam_band, w)
    interior = map((owned, band) -> owned .& .!band, owned, band)
    return (; λ, φ, area, owned, band, interior)
end

# Area-weighted root-mean-square over the cells of `mask` (and over the levels) and the maximum
# over the owned cells of |values|, panel by panel arrays of size (Nx, Ny, Nz) or (Nx, Ny, 1)
function masked_rms(values, area, mask)
    numerator = sum(sum(area[r] .* mask[r] .* values[r] .^ 2) for r in eachindex(values))
    denominator = sum(sum(area[r] .* mask[r]) * size(values[r], 3) for r in eachindex(values))
    return sqrt(numerator / denominator)
end

owned_maximum(values, owned) = maximum(maximum(abs, values[r] .* owned[r]) for r in eachindex(values))

# The error statistics of the panel arrays `error`: the largest value, the root-mean-square over
# the sphere, over the seam band and over the interior, and the ratio of the last two
function seam_statistics(error, cells)
    band = any(any, cells.band) ? masked_rms(error, cells.area, cells.band) : NaN
    interior = masked_rms(error, cells.area, cells.interior)
    return (maximum = owned_maximum(error, cells.owned),
            rms = masked_rms(error, cells.area, cells.owned),
            band = band,
            interior = interior,
            ratio = band / interior)
end

#####
##### Geometry of the sphere
#####

unit_vector(λ, φ) = (cosd(φ) * cosd(λ), cosd(φ) * sind(λ), sind(φ))

# Great-circle angle between two points, in degrees
function great_circle_angle(λ₁, φ₁, λ₂, φ₂)
    a, b = unit_vector(λ₁, φ₁), unit_vector(λ₂, φ₂)
    cross = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
    return atand(sqrt(sum(abs2, cross)), sum(a .* b))
end

# Rotate the point (λ, φ) by `angle` (radians) about the axis tilted by α from the pole towards
# longitude 180°, n = (-sin α, 0, cos α) (Rodrigues' formula)
function rotate_about_tilted_axis(λ, φ, α, angle)
    n = (-sin(α), 0.0, cos(α))
    r = unit_vector(λ, φ)
    n_cross_r = (n[2] * r[3] - n[3] * r[2], n[3] * r[1] - n[1] * r[3], n[1] * r[2] - n[2] * r[1])
    n_dot_r = sum(n .* r)
    x, y, z = r .* cos(angle) .+ n_cross_r .* sin(angle) .+ n .* (n_dot_r * (1 - cos(angle)))
    return atand(y, x), asind(clamp(z, -1, 1))
end

#####
##### Resting isentropic atmosphere, flat or over a mountain across the seam
#####

# A 2 km mountain centered on the southern edge of the core of the Yin panel, at (0°, 45°S):
# the curve that divides the sphere between the panels runs through its summit. It is the
# southern edge of the Yin panel and the western edge of the Yang panel that meet there, and
# those are the edges where the pressure-gradient force reads the halos: the velocities on the
# first faces of a panel (west and south) are prognostic and difference the scalars across the
# edge, while those on its last faces (east and north) are halo points, filled from the other
# panel. (When the tolerances of the resting state were written the mountain stood at 45°N,
# where the northern edge of Yin meets the eastern edge of Yang and no pressure gradient reads
# a halo. It was moved south, to the stricter place, after the first runs; the tolerances were
# not changed.)
const mountain_longitude = 0
const mountain_latitude = -45
seam_mountain(λ, φ) = g * 2000 * exp(-((λ - mountain_longitude) / 30)^2 - ((φ - mountain_latitude) / 15)^2)

function resting_isentropic_model(grid, terrain; θ₀=300, kw...)
    A, B = hybrid_coefficients(size(grid, 3))
    surface_geopotential = terrain === :mountain ? seam_mountain : 0
    dynamics = HydrostaticDynamics(; surface_geopotential, pressure_coefficient=A, surface_pressure_coefficient=B)
    model = hydrostatic_model(grid; dynamics, coriolis=HydrostaticSphericalCoriolis(FT; rotation_rate),
                              thermodynamic_constants=reference_model_constants(), kw...)

    if terrain === :mountain
        set!(model; pˢ = (λ, φ) -> isentropic_surface_pressure(seam_mountain(λ, φ), θ₀), θ=θ₀)
    else
        set!(model; pˢ=pˢᵗ, θ=θ₀)
    end

    return model
end

#####
##### Solid-body rotation of an isothermal atmosphere
#####
#
# The flow rotates rigidly at the angular speed ω = U / a about an axis tilted by α from the pole
# towards longitude 180°, n = (-sin α, 0, cos α), on a planet that rotates at Ω about its pole
# (Williamson et al. 1992, test case 2, with a planet at rest when the axis is tilted):
#
#     u = ω a (cos φ cos α + sin φ cos λ sin α) ,    v = - ω a sin λ sin α .
#
# With sin φ′ = n ⋅ r̂ the latitude from the tilted axis, an isothermal atmosphere at T₀ is in
# cyclostrophic (and, for α = 0, geostrophic) balance with this flow on every surface of constant
# pressure when its surface pressure over flat ground is
#
#     pˢ = p₀ exp[- (ω² / 2 + Ω ω) a² sin² φ′ / (Rᵈ T₀)] ,
#
# with p₀ the surface pressure on the equator of the flow: an exact steady solution of the
# hydrostatic equations. α ≠ 0 requires Ω = 0, because the
# Coriolis parameter of the model is not tilted with the flow.

function solid_body_rotation(; α, Ω, U=40.0, T₀=300.0, p₀=1e5)
    (α == 0 || Ω == 0) || throw(ArgumentError("A tilted solid-body rotation is steady only on a planet at rest."))
    return (; α, Ω, ω = U / sphere_radius, U, T₀, p₀)
end

sine_of_tilted_latitude(s, λ, φ) = cos(s.α) * sind(φ) - sin(s.α) * cosd(φ) * cosd(λ)

solid_body_surface_pressure(s, λ, φ) =
    s.p₀ * exp(- (s.ω^2 / 2 + s.Ω * s.ω) * sphere_radius^2 * sine_of_tilted_latitude(s, λ, φ)^2 / (Rᵈ * s.T₀))

solid_body_u(s, λ, φ) = s.ω * sphere_radius * (cosd(φ) * cos(s.α) + sind(φ) * cosd(λ) * sin(s.α))
solid_body_v(s, λ, φ) = - s.ω * sphere_radius * sind(λ) * sin(s.α)

# A tracer blob: a Gaussian of the great-circle angle to its center, uniform in the vertical,
# which the solid-body rotation carries around its axis without changing its shape
const blob_radius = 20.0 # degrees

blob(λ, φ, center) = exp(-(great_circle_angle(λ, φ, center...) / blob_radius)^2)

blob_center(s, center, t) = rotate_about_tilted_axis(center..., s.α, s.ω * t)

"""
    solid_body_model(grid, s; blob_center, kw...)

The balanced solid-body rotation `s` on `grid`, isothermal layer by layer, with two tracers: `c`,
a uniform mixing ratio of 3 advected by the second-order centered scheme, and `d`, a blob
centered at `blob_center` advected by fifth-order WENO.
"""
function solid_body_model(grid, s; blob_center, kw...)
    A, B = hybrid_coefficients(size(grid, 3))
    dynamics = HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B)
    coriolis = s.Ω == 0 ? nothing : HydrostaticSphericalCoriolis(FT; rotation_rate=s.Ω)

    model = hydrostatic_model(grid; dynamics, coriolis, tracers=(:c, :d),
                              scalar_advection=(; d=WENO(FT; order=5)),
                              thermodynamic_constants=reference_model_constants(), kw...)

    set!(model; pˢ = (λ, φ) -> solid_body_surface_pressure(s, λ, φ))

    # Isothermal at the layer pressure, at which T = θ Π
    p = dynamics_pressure(model.dynamics)
    set!(model; θ = s.T₀ * (pˢᵗ / p)^κ,
                u = (λ, φ, z) -> solid_body_u(s, λ, φ),
                v = (λ, φ, z) -> solid_body_v(s, λ, φ))

    ρ = continuity_density(model.dynamics)
    set!(model; c = 3 * ρ)
    set!(model; d = ρ * horizontal_field(grid, (λ, φ) -> blob(λ, φ, blob_center)))

    return model
end

# The owned integrals of the layer mass, of the potential temperature density and of the two tracers
function contents(model)
    ρ = continuity_density(model.dynamics)
    ρθ = model.formulation.potential_temperature_density
    return (mass = total(ρ), Θ = total(ρθ), c = total(model.tracers.c), d = total(model.tracers.d))
end

# The mixing ratio of a tracer, whose prognostic variable is a density between time steps
mixing_ratio(model, name) = map((ρc, ρ) -> ρc ./ ρ, panels(model.tracers[name]), panels(continuity_density(model.dynamics)))

speed_error(u, v, u⁰, v⁰) = map((u, v, u⁰, v⁰) -> sqrt.((u .- u⁰) .^ 2 .+ (v .- v⁰) .^ 2), u, v, u⁰, v⁰)

conservation_fixer_callbacks(model) = [Callback(HydrostaticConservationFixer(model); callsite=TracerTendencyCallsite())]
