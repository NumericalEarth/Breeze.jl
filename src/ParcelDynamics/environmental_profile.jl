#####
##### EnvironmentalProfile: prescribed atmospheric sounding
#####

"""
$(TYPEDEF)

Represents a prescribed atmospheric sounding that provides environmental
conditions as functions of height.

The environmental profile specifies the thermodynamic and kinematic state
of the atmosphere through which a parcel rises. Temperature, pressure, and
density are computed from the sounding, while velocity functions describe
the background wind field that advects the parcel.

# Fields
$(TYPEDFIELDS)

# Example

```julia
using Breeze.ParcelDynamics

# Simple linear temperature profile (9.8 K/km lapse rate)
T_profile(z) = 288.15 - 0.0098 * z

# Hydrostatic pressure from ideal gas
p₀ = 101325.0  # Pa
function p_profile(z)
    T = T_profile(z)
    # Simplified: p = p₀ * exp(-g*z/(R*T_mean))
    return p₀ * exp(-9.81 * z / (287.0 * 250.0))
end

# Density from ideal gas law
ρ_profile(z) = p_profile(z) / (287.0 * T_profile(z))

profile = EnvironmentalProfile(
    temperature = T_profile,
    pressure = p_profile,
    density = ρ_profile,
    specific_humidity = z -> 0.01 * exp(-z/2500),  # Exponential decay
    u = z -> 5.0,   # 5 m/s background wind
    v = z -> 0.0,
    w = z -> 0.5    # 0.5 m/s updraft
)
```
"""
struct EnvironmentalProfile{T, P, R, Q, U, V, W}
    "Temperature as a function of height: `T(z)` [K]"
    temperature :: T

    "Pressure as a function of height: `p(z)` [Pa]"
    pressure :: P

    "Density as a function of height: `ρ(z)` [kg/m³]"
    density :: R

    "Specific humidity (total water) as a function of height: `qᵗ(z)` [kg/kg]"
    specific_humidity :: Q

    "Zonal velocity as a function of height: `u(z)` [m/s]"
    u :: U

    "Meridional velocity as a function of height: `v(z)` [m/s]"
    v :: V

    "Vertical velocity as a function of height: `w(z)` [m/s]"
    w :: W
end

"""
$(TYPEDSIGNATURES)

Construct an `EnvironmentalProfile` from functions of height.

# Keyword Arguments
- `temperature`: Function `T(z)` returning temperature in K
- `pressure`: Function `p(z)` returning pressure in Pa
- `density`: Function `ρ(z)` returning density in kg/m³
- `specific_humidity`: Function `qᵗ(z)` returning total water mixing ratio (default: dry)
- `u, v, w`: Functions returning velocity components in m/s (default: calm conditions)
"""
function EnvironmentalProfile(;
    temperature,
    pressure,
    density,
    specific_humidity = z -> 0.0,
    u = z -> 0.0,
    v = z -> 0.0,
    w = z -> 0.0)

    return EnvironmentalProfile(temperature, pressure, density, specific_humidity, u, v, w)
end

# Evaluation methods
@inline (profile::EnvironmentalProfile)(::Val{:T}, z) = profile.temperature(z)
@inline (profile::EnvironmentalProfile)(::Val{:p}, z) = profile.pressure(z)
@inline (profile::EnvironmentalProfile)(::Val{:ρ}, z) = profile.density(z)
@inline (profile::EnvironmentalProfile)(::Val{:qᵗ}, z) = profile.specific_humidity(z)
@inline (profile::EnvironmentalProfile)(::Val{:u}, z) = profile.u(z)
@inline (profile::EnvironmentalProfile)(::Val{:v}, z) = profile.v(z)
@inline (profile::EnvironmentalProfile)(::Val{:w}, z) = profile.w(z)

# Convenience accessors
@inline environmental_temperature(profile, z) = profile(Val(:T), z)
@inline environmental_pressure(profile, z) = profile(Val(:p), z)
@inline environmental_density(profile, z) = profile(Val(:ρ), z)
@inline environmental_specific_humidity(profile, z) = profile(Val(:qᵗ), z)
@inline environmental_velocity(profile, z) = (profile(Val(:u), z), profile(Val(:v), z), profile(Val(:w), z))
