using Oceananigans.Architectures: CPU

#####
##### Virtual temperature
#####

"""
$(TYPEDSIGNATURES)

Return the virtual temperature ``Tᵛ = Rᵐ T / Rᵈ`` of moist air at temperature `T` with moisture
mass fractions `q`: the temperature at which dry air has the density of the moist air at the same
pressure, ``ρ = p / (Rᵐ T) = p / (Rᵈ Tᵛ)``. Condensate counts as mass that carries no gas constant,
``Tᵛ = T (1 + (Rᵛ / Rᵈ - 1) qᵛ - qˡ - qⁱ)``, so this is the *density* temperature of air that
carries cloud or precipitation.

```jldoctest
using Breeze
using Breeze.Thermodynamics: virtual_temperature, MoistureMassFractions

constants = ThermodynamicConstants()
virtual_temperature(300.0, MoistureMassFractions(0.01, 0.001, 0.0), constants)

# output
301.5243130724396
```
"""
@inline virtual_temperature(T, q, constants) = mixture_gas_constant(q, constants) * T / dry_air_gas_constant(constants)

#####
##### Hydrostatic soundings
#####

"""
$(TYPEDSIGNATURES)

A tabulated atmospheric column: the virtual temperature `virtual_temperature` at `heights`,
anchored by the pressure `bottom_pressure` at the lowest height. The virtual temperature is
linear in ``z`` between heights and constant beyond the first and last; a height may be repeated
to represent a jump, the value above taking effect at the height itself. The pressure follows from
hydrostatic balance, ``∂_z \\log p = -g / (Rᵈ Tᵛ)``, integrated exactly for this profile (see
[`sounding_pressure`](@ref)). Built from an observed or modeled column — for example
``Tᵛ = p / (Rᵈ ρ)`` from archived pressure and density, or [`virtual_temperature`](@ref) from
temperature and moisture — it reproduces that column's own hydrostatic pressure.

A `HydrostaticSounding` (or an `(Nx, Ny)` array of them, one per column of a column ensemble) is
passed to [`ReferenceState`](@ref) as `sounding`.

```jldoctest
using Breeze

sounding = HydrostaticSounding([0, 1000, 2000], [300, 293.5, 287]; bottom_pressure=101325)

# output
HydrostaticSounding with 3 levels in 0.0 ≤ z ≤ 2000.0, Tᵛ ∈ [287.0, 300.0], p(0.0) = 101325.0
```
"""
struct HydrostaticSounding{Z, T, P}
    heights :: Z
    virtual_temperature :: T
    bottom_pressure :: P

    function HydrostaticSounding(heights, virtual_temperature; bottom_pressure)
        length(heights) == length(virtual_temperature) ≥ 1 ||
            throw(ArgumentError("`heights` and `virtual_temperature` must have the same, nonzero length; \
                                 got $(length(heights)) and $(length(virtual_temperature))."))
        issorted(heights) || throw(ArgumentError("`heights` must be nondecreasing."))
        all(T -> isfinite(T) && T > 0, virtual_temperature) ||
            throw(ArgumentError("`virtual_temperature` must be finite and positive."))
        isfinite(bottom_pressure) && bottom_pressure > 0 ||
            throw(ArgumentError("`bottom_pressure` must be finite and positive, got $bottom_pressure."))
        z = float.(collect(heights))
        Tᵛ = float.(collect(virtual_temperature))
        p = float(bottom_pressure)
        return new{typeof(z), typeof(Tᵛ), typeof(p)}(z, Tᵛ, p)
    end
end

"""
$(TYPEDSIGNATURES)

A `HydrostaticSounding` from a column's `pressure` and `density` at `heights`, with virtual
temperature ``Tᵛ = p / (Rᵈ ρ)`` and the first pressure as its anchor. The column must be hydrostatic
under the gravitational acceleration of `constants`: the sounding's pressure (see
[`sounding_pressure`](@ref)) must reproduce `pressure` at every height to within `rtol` times the
column's pressure range, or an `ArgumentError` is thrown. This rejects, for example, an archived
reference state that was integrated with another gravitational acceleration or gas constant.

```jldoctest
using Breeze
using Breeze.Thermodynamics: sounding_virtual_temperature

constants = ThermodynamicConstants()
Rᵈ = dry_air_gas_constant(constants)
g = constants.gravitational_acceleration
z = 0:100:3000
p = @. 1e5 * exp(-g * z / (Rᵈ * 280))
ρ = @. p / (Rᵈ * 280)
sounding = HydrostaticSounding(z, p, ρ, constants)
sounding_virtual_temperature(sounding, 1550) ≈ 280

# output
true
```

With a density that is 1% too large for its pressure the column is not hydrostatic:

```jldoctest
using Breeze

constants = ThermodynamicConstants()
Rᵈ = dry_air_gas_constant(constants)
g = constants.gravitational_acceleration
z = 0:100:3000
p = @. 1e5 * exp(-g * z / (Rᵈ * 280))
try
    HydrostaticSounding(z, p, 1.01 .* p ./ (Rᵈ * 280), constants)
catch err
    err isa ArgumentError
end

# output
true
```
"""
function HydrostaticSounding(heights, pressure, density, constants; rtol = 1e-4)
    length(pressure) == length(density) == length(heights) ||
        throw(ArgumentError("`heights`, `pressure` and `density` must have the same length."))
    Rᵈ = dry_air_gas_constant(constants)
    sounding = HydrostaticSounding(heights, pressure ./ (Rᵈ .* density); bottom_pressure = first(pressure))
    residual = maximum(abs(sounding_pressure(sounding, z, constants) - p) for (z, p) in zip(heights, pressure))
    pressure_range = maximum(pressure) - minimum(pressure)
    residual ≤ rtol * pressure_range ||
        throw(ArgumentError("The column is not hydrostatic under g = $(constants.gravitational_acceleration): \
                             its pressure departs from the hydrostatic pressure of its density by $residual Pa, \
                             more than rtol = $rtol times its pressure range ($pressure_range Pa)."))
    return sounding
end

function Base.summary(sounding::HydrostaticSounding)
    z = sounding.heights
    Tᵛ = sounding.virtual_temperature
    return string("HydrostaticSounding with ", length(z), " levels in ", z[1], " ≤ z ≤ ", z[end],
                  ", Tᵛ ∈ [", minimum(Tᵛ), ", ", maximum(Tᵛ), "], p(", z[1], ") = ", sounding.bottom_pressure)
end

Base.show(io::IO, sounding::HydrostaticSounding) = print(io, summary(sounding))

"""
$(TYPEDSIGNATURES)

The virtual temperature of `sounding` at height `z`: linear between its heights, constant beyond
them, and the value above a jump at the height of the jump.
"""
function sounding_virtual_temperature(sounding::HydrostaticSounding, z)
    zᵀ = sounding.heights
    Tᵛ = sounding.virtual_temperature
    N = length(zᵀ)
    n = searchsortedlast(zᵀ, z) # the last height at or below z
    n == 0 && return Tᵛ[1]
    n == N && return Tᵛ[N]
    w = (z - zᵀ[n]) / (zᵀ[n+1] - zᵀ[n])
    return Tᵛ[n] + w * (Tᵛ[n+1] - Tᵛ[n])
end

# ∫ dz / Tᵛ over [z₁, z₂] where Tᵛ is linear from T₁ to T₂: (z₂ - z₁) log(T₂ / T₁) / (T₂ - T₁),
# written with log1p so that it stays accurate as T₂ → T₁.
function linear_inverse_temperature_integral(z₁, z₂, T₁, T₂)
    x = (T₂ - T₁) / T₁
    return (z₂ - z₁) / T₁ * ifelse(x == 0, one(x), log1p(x) / x)
end

# ∫ dz / Tᵛ from the lowest height of `sounding` to z (negative below it)
function inverse_virtual_temperature_integral(sounding::HydrostaticSounding, z)
    zᵀ = sounding.heights
    Tᵛ = sounding.virtual_temperature
    N = length(zᵀ)
    z ≤ zᵀ[1] && return (z - zᵀ[1]) / Tᵛ[1]

    integral = zero(z / Tᵛ[1])
    for n in 1:N-1
        zᵀ[n] < z || break
        if zᵀ[n+1] > zᵀ[n]
            ζ = min(z, zᵀ[n+1])
            Tᶻ = Tᵛ[n] + (ζ - zᵀ[n]) / (zᵀ[n+1] - zᵀ[n]) * (Tᵛ[n+1] - Tᵛ[n])
            integral += linear_inverse_temperature_integral(zᵀ[n], ζ, Tᵛ[n], Tᶻ)
        end
    end

    z > zᵀ[N] && (integral += (z - zᵀ[N]) / Tᵛ[N])
    return integral
end

"""
$(TYPEDSIGNATURES)

The hydrostatic pressure of `sounding` at height `z`,

```math
p(z) = p₁ \\exp \\left( - \\frac{g}{Rᵈ} ∫_{z₁}^{z} \\frac{dz'}{Tᵛ(z')} \\right) ,
```

where ``p₁`` is the `bottom_pressure` at the lowest height ``z₁`` and ``Tᵛ`` the sounding's
piecewise-linear virtual temperature (constant below ``z₁`` and above its top). The integral is
evaluated exactly, so the pressure does not depend on any grid. `g` and `Rᵈ` are those of
`constants`.

```jldoctest
using Breeze
using Breeze.Thermodynamics: sounding_pressure

constants = ThermodynamicConstants()
isothermal = HydrostaticSounding([0, 10000], [250, 250]; bottom_pressure=1e5)
Rᵈ = dry_air_gas_constant(constants)
g = constants.gravitational_acceleration
sounding_pressure(isothermal, 5000, constants) ≈ 1e5 * exp(-g * 5000 / (Rᵈ * 250))

# output
true
```
"""
function sounding_pressure(sounding::HydrostaticSounding, z, constants)
    g = constants.gravitational_acceleration
    Rᵈ = dry_air_gas_constant(constants)
    return sounding.bottom_pressure * exp(-g / Rᵈ * inverse_virtual_temperature_integral(sounding, z))
end

#####
##### Reference states from soundings
#####

validate_sounding_reference(::Nothing, name) = nothing
validate_sounding_reference(value, name) =
    throw(ArgumentError("`$name` cannot be combined with `sounding`: the sounding determines the reference \
                         pressure, density and temperature, and carries the moisture in its virtual temperature."))

# The bottom-face state and the profiles at the cell centers of one column, at heights zˢ and zᶜ
function sounding_column(sounding::HydrostaticSounding, zˢ, zᶜ, constants)
    Rᵈ = dry_air_gas_constant(constants)
    pˢ = sounding_pressure(sounding, zˢ, constants)
    Tˢ = sounding_virtual_temperature(sounding, zˢ)
    pᵣ = [sounding_pressure(sounding, z, constants) for z in zᶜ]
    Tᵣ = [sounding_virtual_temperature(sounding, z) for z in zᶜ]
    return (; pˢ, Tˢ, ρˢ = pˢ / (Rᵈ * Tˢ), pᵣ, Tᵣ, ρᵣ = pᵣ ./ (Rᵈ .* Tᵣ))
end

sounding_at(sounding::HydrostaticSounding, i, j) = sounding
sounding_at(soundings::AbstractArray, i, j) = soundings[i, j]

"""
$(TYPEDSIGNATURES)

Construct a `ReferenceState` whose pressure, temperature and density are those of `sounding`:
``pᵣ`` is [`sounding_pressure`](@ref) at the cell centers, ``Tᵣ`` the sounding's virtual
temperature there, and ``ρᵣ = pᵣ / (Rᵈ Tᵣ)``; the bottom-face anchors are the same at the bottom
face, and `base_pressure` is the sounding's pressure at ``z = 0``. `sounding` is one
`HydrostaticSounding` shared by every column, or an `(Nx, Ny)` array of them, one per column.

The reference is dry with the virtual temperature of the sounding's air, so its density and its
pressure are those of the moist air; `potential_temperature` metadata is the potential temperature
of that virtual temperature. The profiles are evaluated on the CPU at construction and then
transferred to the grid's architecture.
"""
function sounding_reference_state(grid, constants, sounding, pˢᵗ, discrete_hydrostatic_balance)
    discrete_hydrostatic_balance &&
        throw(ArgumentError("`discrete_hydrostatic_balance` is not supported with a `sounding` reference."))

    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)
    sounding isa AbstractArray && size(sounding) != (Nx, Ny) &&
        throw(ArgumentError("A `sounding` array must have size ($Nx, $Ny), one per column, got $(size(sounding))."))

    # A sounding shared by every column of an ordinary grid gives reduced column profiles, as for
    # the other homogeneous references; per-column soundings need full fields.
    LX, LY = sounding isa AbstractArray ? (Center, Center) : reference_horizontal_location(grid)
    loc = (LX(), LY(), Center())
    Nxᶠ = LX === Nothing ? 1 : Nx
    Nyᶠ = LY === Nothing ? 1 : Ny

    cpu_grid = on_architecture(CPU(), grid)
    c = Center()
    columns = [let s = sounding_at(sounding, i, j)
                   zˢ = znode(i, j, 1, cpu_grid, c, c, Face())
                   zᶜ = [znode(i, j, k, cpu_grid, c, c, c) for k in 1:Nz]
                   sounding_column(s, zˢ, zᶜ, constants)
               end for i in 1:Nxᶠ, j in 1:Nyᶠ]

    surface(name) = surface_state_field(grid, FT[getproperty(columns[min(i, Nxᶠ), min(j, Nyᶠ)], name) for i in 1:Nx, j in 1:Ny, _ in 1:1])
    pˢ = surface(:pˢ)
    ρˢ = surface(:ρˢ)
    Tˢ = surface(:Tˢ)

    profile(name) = FT[getproperty(columns[i, j], name)[k] for i in 1:Nxᶠ, j in 1:Nyᶠ, k in 1:Nz]

    # As for the other references: one profile shared by every column takes its bottom-face anchors as
    # boundary values, while per-column profiles (as `profile_reference_state`) keep default boundary conditions.
    reference_field(surface) = sounding isa AbstractArray ? CenterField(grid) :
        Field{LX, LY, Center}(grid, boundary_conditions=FieldBoundaryConditions(grid, loc,
                              bottom=ValueBoundaryCondition(surface_boundary_value(surface))))

    ρᵣ = reference_field(ρˢ)
    set!(ρᵣ, profile(:ρᵣ))
    fill_halo_regions!(ρᵣ)

    pᵣ = reference_field(pˢ)
    set!(pᵣ, profile(:pᵣ))
    fill_halo_regions!(pᵣ)

    Tᵣ = Field{LX, LY, Center}(grid)
    set!(Tᵣ, profile(:Tᵣ))
    fill_halo_regions!(Tᵣ)

    # Scalar metadata: the datum and the potential temperature of the first column at z = 0
    s₁₁ = sounding_at(sounding, 1, 1)
    p₀ = convert(FT, sounding_pressure(s₁₁, 0, constants))
    κ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    θ₀ = convert(FT, sounding_virtual_temperature(s₁₁, 0) * (pˢᵗ / p₀)^κ)
    Z = reference_moisture_field(nothing, grid)

    return ReferenceState(p₀, pˢ, ρˢ, Tˢ, θ₀, pˢᵗ, pᵣ, ρᵣ, Tᵣ, Z, Z, Z)
end
