#####
##### Per-droplet physics for Lagrangian (particle-based) cloud microphysics
#####
##### Every function below acts on scalars, is `@inline`, type-stable and allocation-free,
##### so it can be called from inside GPU kernels that loop over droplets.
#####
##### Notation (see the "Lagrangian droplet notation" section of the notation appendix):
#####   r   wet droplet radius          rᵈ  dry (solute) radius        κ   hygroscopicity
#####   z   dry-to-wet volume ratio (rᵈ/r)³                            aʷ  water activity
#####   A   curvature (Kelvin) coefficient 2σ/(Rᵛ ρᴸ T)                 𝒮ᵉ  equilibrium supersaturation
#####   Fᵛ  vapor-diffusion term         Fᵀ  heat-conduction term       G   growth coefficient 1/(Fᵛ + Fᵀ)
#####   ψᵛ  gas-kinetic vapor factor     ψᵀ  gas-kinetic thermal factor fᵛ  ventilation factor
#####   Φ   r dr/dt                      τᵖ  Stokes response time       fᴰ  drag correction
#####

using Breeze.Solvers: NewtonSolver, bracketed_newton_solve
using Breeze.Thermodynamics: PlanarLiquidSurface, dry_air_gas_constant, liquid_latent_heat,
                             saturation_vapor_pressure, vapor_gas_constant

using Oceananigans: Oceananigans
using Oceananigans.Utils: prettysummary

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF

#####
##### Air transport properties
#####

"""
$(TYPEDEF)

Diffusivity of water vapor in air following the power law

```math
Dᵛ = D₀ (T / T₀)^n (p₀ / p) ,
```

with the default coefficients of [Pruppacher and Klett (2010)](@cite pruppacher2010microphysics)
(their equation 13.3): ``D₀ = 2.11 × 10⁻⁵`` m² s⁻¹, ``T₀ = 273.15`` K,
``p₀ = 101325`` Pa and ``n = 1.94``.
"""
struct PowerLawVaporDiffusivity{FT}
    reference_diffusivity :: FT
    reference_temperature :: FT
    reference_pressure :: FT
    temperature_exponent :: FT
end

"""
$(TYPEDSIGNATURES)

Return a [`PowerLawVaporDiffusivity`](@ref).

```jldoctest
using Breeze

PowerLawVaporDiffusivity()

# output
PowerLawVaporDiffusivity{Float64}(D₀=2.11e-5, T₀=273.15, p₀=101325.0, n=1.94)
```
"""
function PowerLawVaporDiffusivity(FT = Oceananigans.defaults.FloatType;
                                  reference_diffusivity = 2.11e-5,
                                  reference_temperature = 273.15,
                                  reference_pressure = 101325,
                                  temperature_exponent = 1.94)

    return PowerLawVaporDiffusivity{FT}(convert(FT, reference_diffusivity),
                                        convert(FT, reference_temperature),
                                        convert(FT, reference_pressure),
                                        convert(FT, temperature_exponent))
end

"""
$(TYPEDEF)

Thermal conductivity of air varying linearly with temperature,

```math
Kᵀ = K₀ + K₁ (T - T₀) ,
```

with the default coefficients of [Pruppacher and Klett (2010)](@cite pruppacher2010microphysics)
(their equation 13.18a converted to SI units): ``K₀ = 2.38 × 10⁻²`` W m⁻¹ K⁻¹,
``K₁ = 7.11 × 10⁻⁵`` W m⁻¹ K⁻² and ``T₀ = 273.15`` K.
"""
struct LinearThermalConductivity{FT}
    reference_conductivity :: FT
    temperature_coefficient :: FT
    reference_temperature :: FT
end

"""
$(TYPEDSIGNATURES)

Return a [`LinearThermalConductivity`](@ref).

```jldoctest
using Breeze

LinearThermalConductivity()

# output
LinearThermalConductivity{Float64}(K₀=0.0238, K₁=7.11e-5, T₀=273.15)
```
"""
function LinearThermalConductivity(FT = Oceananigans.defaults.FloatType;
                                   reference_conductivity = 2.38e-2,
                                   temperature_coefficient = 7.11e-5,
                                   reference_temperature = 273.15)

    return LinearThermalConductivity{FT}(convert(FT, reference_conductivity),
                                         convert(FT, temperature_coefficient),
                                         convert(FT, reference_temperature))
end

"""
$(TYPEDEF)

Dynamic viscosity of air following Sutherland's law,

```math
μ = μ₀ (T / T₀)^{3/2} (T₀ + Tˢ) / (T + Tˢ) ,
```

with ``μ₀ = 1.716 × 10⁻⁵`` Pa s, ``T₀ = 273.15`` K and the Sutherland temperature ``Tˢ = 110.4`` K.
"""
struct SutherlandViscosity{FT}
    reference_viscosity :: FT
    reference_temperature :: FT
    sutherland_temperature :: FT
end

"""
$(TYPEDSIGNATURES)

Return a [`SutherlandViscosity`](@ref).

```jldoctest
using Breeze

SutherlandViscosity()

# output
SutherlandViscosity{Float64}(μ₀=1.716e-5, T₀=273.15, Tˢ=110.4)
```
"""
function SutherlandViscosity(FT = Oceananigans.defaults.FloatType;
                             reference_viscosity = 1.716e-5,
                             reference_temperature = 273.15,
                             sutherland_temperature = 110.4)

    return SutherlandViscosity{FT}(convert(FT, reference_viscosity),
                                   convert(FT, reference_temperature),
                                   convert(FT, sutherland_temperature))
end

"""
$(TYPEDEF)

Dynamic viscosity of air varying linearly with temperature,

```math
μ = μ₀ + μ₁ (T - T₀) ,
```

with the default coefficients of [Pruppacher and Klett (2010)](@cite pruppacher2010microphysics)
for ``T ≥ 0`` °C in SI units: ``μ₀ = 1.718 × 10⁻⁵`` Pa s, ``μ₁ = 4.9 × 10⁻⁸`` Pa s K⁻¹
and ``T₀ = 273.15`` K.
"""
struct LinearViscosity{FT}
    reference_viscosity :: FT
    temperature_coefficient :: FT
    reference_temperature :: FT
end

"""
$(TYPEDSIGNATURES)

Return a [`LinearViscosity`](@ref).

```jldoctest
using Breeze

LinearViscosity()

# output
LinearViscosity{Float64}(μ₀=1.718e-5, μ₁=4.9e-8, T₀=273.15)
```
"""
function LinearViscosity(FT = Oceananigans.defaults.FloatType;
                         reference_viscosity = 1.718e-5,
                         temperature_coefficient = 4.9e-8,
                         reference_temperature = 273.15)

    return LinearViscosity{FT}(convert(FT, reference_viscosity),
                               convert(FT, temperature_coefficient),
                               convert(FT, reference_temperature))
end

"""
$(TYPEDSIGNATURES)

Return the diffusivity of water vapor in air ``Dᵛ`` (m² s⁻¹) at temperature `T` and pressure `p`
according to `model`.
"""
@inline function vapor_diffusivity(model::PowerLawVaporDiffusivity, T, p)
    D₀ = model.reference_diffusivity
    T₀ = model.reference_temperature
    p₀ = model.reference_pressure
    n = model.temperature_exponent
    return D₀ * (T / T₀)^n * (p₀ / p)
end

"""
$(TYPEDSIGNATURES)

Return the thermal conductivity of air ``Kᵀ`` (W m⁻¹ K⁻¹) at temperature `T` and pressure `p`
according to `model`.
"""
@inline function thermal_conductivity(model::LinearThermalConductivity, T, p)
    K₀ = model.reference_conductivity
    K₁ = model.temperature_coefficient
    T₀ = model.reference_temperature
    return K₀ + K₁ * (T - T₀)
end

"""
$(TYPEDSIGNATURES)

Return the dynamic viscosity of air ``μ`` (Pa s) at temperature `T` according to `model`.
"""
@inline function dynamic_viscosity(model::SutherlandViscosity, T)
    μ₀ = model.reference_viscosity
    T₀ = model.reference_temperature
    Tˢ = model.sutherland_temperature
    return μ₀ * (T / T₀) * sqrt(T / T₀) * (T₀ + Tˢ) / (T + Tˢ)
end

@inline function dynamic_viscosity(model::LinearViscosity, T)
    μ₀ = model.reference_viscosity
    μ₁ = model.temperature_coefficient
    T₀ = model.reference_temperature
    return μ₀ + μ₁ * (T - T₀)
end

#####
##### Gas-kinetic corrections, ventilation, solution activity
#####

"""
$(TYPEDEF)

Gas-kinetic (non-continuum) corrections to the vapor diffusivity and the thermal conductivity
near a droplet of radius ``r``
([Fukuta and Walter, 1970](@cite FukutaWalter1970); [Pruppacher and Klett, 2010](@cite pruppacher2010microphysics), section 13.1.3),

```math
Dᵛ / D̃ᵛ = ψᵛ(r) = \\frac{r}{r + Δᵛ} + \\frac{Dᵛ}{αᶜ r} \\sqrt{\\frac{2π}{Rᵛ T}} , \\qquad
Kᵀ / K̃ᵀ = ψᵀ(r) = \\frac{r}{r + Δᵀ} + \\frac{Kᵀ}{αᵀ r ρ cᵖᵈ} \\sqrt{\\frac{2π}{Rᵈ T}} ,
```

where ``D̃ᵛ`` and ``K̃ᵀ`` are the corrected coefficients, ``αᶜ`` is the condensation
(mass accommodation) coefficient, ``αᵀ`` the thermal accommodation coefficient, and ``Δᵛ``
and ``Δᵀ`` the vapor and thermal jump lengths.
"""
struct KineticCorrection{FT}
    condensation_coefficient :: FT
    thermal_accommodation_coefficient :: FT
    vapor_jump_length :: FT
    thermal_jump_length :: FT
end

"""
$(TYPEDSIGNATURES)

Return a [`KineticCorrection`](@ref). The defaults use unit condensation and thermal accommodation
coefficients and the jump lengths ``Δᵛ = 0.104`` μm and ``Δᵀ = 0.216`` μm of
[Pruppacher and Klett (2010)](@cite pruppacher2010microphysics). The NCAR parcel-DNS code of
[Chen et al. (2020)](@cite ChenEtAl2020) uses `condensation_coefficient = 0.036` and
`thermal_accommodation_coefficient = 0.7`.

```jldoctest
using Breeze

KineticCorrection(condensation_coefficient=0.036)

# output
KineticCorrection{Float64}(αᶜ=0.036, αᵀ=1.0, Δᵛ=1.04e-7, Δᵀ=2.16e-7)
```
"""
function KineticCorrection(FT = Oceananigans.defaults.FloatType;
                           condensation_coefficient = 1,
                           thermal_accommodation_coefficient = 1,
                           vapor_jump_length = 0.104e-6,
                           thermal_jump_length = 0.216e-6)

    return KineticCorrection{FT}(convert(FT, condensation_coefficient),
                                 convert(FT, thermal_accommodation_coefficient),
                                 convert(FT, vapor_jump_length),
                                 convert(FT, thermal_jump_length))
end

"""
$(TYPEDEF)

Continuum diffusion of vapor and heat: no gas-kinetic correction, ``ψᵛ = ψᵀ = 1``.
"""
struct NoKineticCorrection end

"""
$(TYPEDEF)

No ventilation: the ventilation factor is ``fᵛ = 1``.
"""
struct NoVentilation end

"""
$(TYPEDEF)

Ventilation factor for the diffusional growth of falling drops,
of [Beard and Pruppacher (1971)](@cite BeardPruppacher1971) (see also
[Pruppacher and Klett, 2010](@cite pruppacher2010microphysics), equations 13.60–13.61),

```math
fᵛ = \\begin{cases} 1 + 0.108 X^2 & X < 1.4 \\\\ 0.78 + 0.308 X & X ≥ 1.4 \\end{cases} , \\qquad
X = \\mathrm{Sc}^{1/3} \\mathrm{Re}^{1/2} ,
```

where ``\\mathrm{Re} = 2 r ρ |\\boldsymbol{u} - \\boldsymbol{v}| / μ`` is the droplet Reynolds number based on the
slip speed and ``\\mathrm{Sc} = μ / (ρ Dᵛ)`` is the Schmidt number of vapor in air.
"""
struct BeardPruppacherVentilation end

"""
$(TYPEDEF)

Water activity of a solution droplet following the κ-Köhler theory of
[Petters and Kreidenweis (2007)](@cite PettersKreidenweis2007),

```math
aʷ = \\frac{r^3 - rᵈ^3}{r^3 - rᵈ^3 (1 - κ)} = \\frac{1 - z}{1 - z + κ z} , \\qquad z = (rᵈ / r)^3 .
```
"""
struct KappaKoehlerActivity end

"""
$(TYPEDEF)

Water activity of a dilute solution droplet, the exponential form of Raoult's law
used by many parcel models (for example the NCAR parcel-DNS code of [Chen et al., 2020](@cite ChenEtAl2020)),

```math
\\log aʷ = - \\frac{κ rᵈ^3}{r^3 - rᵈ^3} = - \\frac{κ z}{1 - z} .
```

It agrees with [`KappaKoehlerActivity`](@ref) to first order in ``κ z``, and the two differ
appreciably only for concentrated haze droplets.
"""
struct DiluteSolutionActivity end

#####
##### Terminal velocity laws
#####

"""
$(TYPEDEF)

Stokes terminal velocity ``𝕎 = 2 ρᴸ g r^2 / (9 μ)``, valid for ``r ≲ 30`` μm.
"""
struct StokesTerminalVelocity end

"""
$(TYPEDEF)

Piecewise power-law terminal velocity of [Rogers and Yau (1989)](@cite rogers1989short),
taken as the smallest of the three power laws so that it is continuous,

```math
𝕎 = \\min \\left [ k₁ r^2, \\; k₃ r, \\; k₂ (ρ₀ / ρ)^{1/2} r^{1/2} \\right ] ,
```

with ``k₁ = 1.19 × 10⁸`` m⁻¹ s⁻¹, ``k₃ = 8 × 10³`` s⁻¹, ``k₂ = 201`` m^{1/2} s⁻¹ and
``ρ₀ = 1.2`` kg m⁻³.
"""
struct RogersYauTerminalVelocity{FT}
    quadratic_coefficient :: FT
    linear_coefficient :: FT
    square_root_coefficient :: FT
    reference_density :: FT
end

"""
$(TYPEDSIGNATURES)

Return a [`RogersYauTerminalVelocity`](@ref).

```jldoctest
using Breeze

RogersYauTerminalVelocity()

# output
RogersYauTerminalVelocity{Float64}(k₁=1.19e8, k₃=8000.0, k₂=201.0, ρ₀=1.2)
```
"""
function RogersYauTerminalVelocity(FT = Oceananigans.defaults.FloatType;
                                   quadratic_coefficient = 1.19e8,
                                   linear_coefficient = 8e3,
                                   square_root_coefficient = 201,
                                   reference_density = 1.2)

    return RogersYauTerminalVelocity{FT}(convert(FT, quadratic_coefficient),
                                         convert(FT, linear_coefficient),
                                         convert(FT, square_root_coefficient),
                                         convert(FT, reference_density))
end

"""
$(TYPEDEF)

Terminal velocity of water drops in air from the three-regime fit of [Beard (1976)](@cite Beard1976),
valid for drop diameters from 0.5 μm to 7 mm at arbitrary temperature and pressure.
Regime 1 (``d < 19`` μm) is Stokes flow with a slip correction; regime 2 (``19`` μm ``≤ d < 1.07`` mm)
is a polynomial fit of ``\\log \\mathrm{Re}`` in the logarithm of the Davies number; regime 3
(``d ≥ 1.07`` mm) fits ``\\log \\mathrm{Re}`` in the logarithm of the Bond number times the
physical-property number to the sixth root, accounting for drop deformation. Diameters beyond
7 mm are evaluated at 7 mm.
"""
struct BeardTerminalVelocity{FT}
    reference_mean_free_path :: FT
    reference_viscosity :: FT
    reference_pressure :: FT
    reference_temperature :: FT
    regime_2_coefficients :: NTuple{7, FT}
    regime_3_coefficients :: NTuple{6, FT}
end

"""
$(TYPEDSIGNATURES)

Return a [`BeardTerminalVelocity`](@ref) with the coefficients of [Beard (1976)](@cite Beard1976).
The mean free path of air molecules, which enters the slip correction, is
``λ = λ₀ (μ / μ₀) (p₀ / p) (T / T₀)^{1/2}`` with ``λ₀ = 6.62 × 10⁻⁸`` m at
``μ₀ = 1.818 × 10⁻⁵`` Pa s, ``p₀ = 101325`` Pa and ``T₀ = 293.15`` K.

```jldoctest
using Breeze

BeardTerminalVelocity()

# output
BeardTerminalVelocity{Float64}(λ₀=6.62e-8, μ₀=1.818e-5, p₀=101325.0, T₀=293.15)
```
"""
function BeardTerminalVelocity(FT = Oceananigans.defaults.FloatType;
                               reference_mean_free_path = 6.62e-8,
                               reference_viscosity = 1.818e-5,
                               reference_pressure = 101325,
                               reference_temperature = 293.15)

    b² = (-0.318657e1, 0.992696, -0.153193e-2, -0.987059e-3, -0.578878e-3, 0.855176e-4, -0.327815e-5)
    b³ = (-0.500015e1, 0.523778e1, -0.204914e1, 0.475294, -0.542819e-1, 0.238449e-2)

    return BeardTerminalVelocity{FT}(convert(FT, reference_mean_free_path),
                                     convert(FT, reference_viscosity),
                                     convert(FT, reference_pressure),
                                     convert(FT, reference_temperature),
                                     convert.(FT, b²),
                                     convert.(FT, b³))
end

#####
##### Parameters
#####

"""
$(TYPEDEF)

Parameters of the per-droplet physics: surface tension, the transport-property models of air,
gas-kinetic corrections, ventilation, solution activity, the terminal velocity law,
and the scalar solver used by the implicit condensational growth step.

Liquid density, gas constants, heat capacities, the latent heat ``ℒˡ(T)`` and the saturation
vapor pressure ``pᵛ⁺(T)`` are taken from Breeze's [`ThermodynamicConstants`](@ref),
which every function receives alongside the parameters.
"""
struct DropletPhysicsParameters{FT, D, K, M, C, V, A, W, S}
    surface_tension :: FT
    vapor_diffusivity :: D
    thermal_conductivity :: K
    dynamic_viscosity :: M
    kinetic_correction :: C
    ventilation :: V
    solution_activity :: A
    terminal_velocity :: W
    solver :: S
end

"""
$(TYPEDSIGNATURES)

Return `DropletPhysicsParameters` with float type `FT`.

Keyword arguments
=================

- `surface_tension`: surface tension of the droplet-air interface ``σ`` (N m⁻¹). Default: 0.072.
- `vapor_diffusivity`: model for ``Dᵛ(T, p)``. Default: [`PowerLawVaporDiffusivity`](@ref).
- `thermal_conductivity`: model for ``Kᵀ(T, p)``. Default: [`LinearThermalConductivity`](@ref).
- `dynamic_viscosity`: model for ``μ(T)``. Default: [`SutherlandViscosity`](@ref).
- `kinetic_correction`: [`KineticCorrection`](@ref) (default) or [`NoKineticCorrection`](@ref).
- `ventilation`: [`BeardPruppacherVentilation`](@ref) (default) or [`NoVentilation`](@ref).
- `solution_activity`: [`KappaKoehlerActivity`](@ref) (default) or [`DiluteSolutionActivity`](@ref).
- `terminal_velocity`: [`BeardTerminalVelocity`](@ref) (default), [`RogersYauTerminalVelocity`](@ref)
  or [`StokesTerminalVelocity`](@ref).
- `solver`: [`NewtonSolver`](@ref) for the safeguarded Newton iterations, with tolerances on
  nondimensional unknowns. Default: `reltol = √ε`, `abstol = 10ε`, `maxiter = 50`, with ``ε = `` `eps(FT)`.

```jldoctest
using Breeze

DropletPhysicsParameters()

# output
DropletPhysicsParameters{Float64}:
├── surface_tension: 0.072
├── vapor_diffusivity: PowerLawVaporDiffusivity{Float64}(D₀=2.11e-5, T₀=273.15, p₀=101325.0, n=1.94)
├── thermal_conductivity: LinearThermalConductivity{Float64}(K₀=0.0238, K₁=7.11e-5, T₀=273.15)
├── dynamic_viscosity: SutherlandViscosity{Float64}(μ₀=1.716e-5, T₀=273.15, Tˢ=110.4)
├── kinetic_correction: KineticCorrection{Float64}(αᶜ=1.0, αᵀ=1.0, Δᵛ=1.04e-7, Δᵀ=2.16e-7)
├── ventilation: BeardPruppacherVentilation()
├── solution_activity: KappaKoehlerActivity()
├── terminal_velocity: BeardTerminalVelocity{Float64}(λ₀=6.62e-8, μ₀=1.818e-5, p₀=101325.0, T₀=293.15)
└── solver: NewtonSolver(reltol=1.49012e-8, abstol=2.22045e-15, maxiter=50)
```
"""
function DropletPhysicsParameters(FT = Oceananigans.defaults.FloatType;
                                  surface_tension = 0.072,
                                  vapor_diffusivity = PowerLawVaporDiffusivity(FT),
                                  thermal_conductivity = LinearThermalConductivity(FT),
                                  dynamic_viscosity = SutherlandViscosity(FT),
                                  kinetic_correction = KineticCorrection(FT),
                                  ventilation = BeardPruppacherVentilation(),
                                  solution_activity = KappaKoehlerActivity(),
                                  terminal_velocity = BeardTerminalVelocity(FT),
                                  solver = NewtonSolver(FT; reltol=sqrt(eps(FT)), abstol=10eps(FT), maxiter=50))

    return DropletPhysicsParameters(convert(FT, surface_tension),
                                    vapor_diffusivity,
                                    thermal_conductivity,
                                    dynamic_viscosity,
                                    kinetic_correction,
                                    ventilation,
                                    solution_activity,
                                    terminal_velocity,
                                    solver)
end

#####
##### Show methods
#####

Base.summary(m::PowerLawVaporDiffusivity{FT}) where FT =
    string("PowerLawVaporDiffusivity{", FT, "}(D₀=", prettysummary(m.reference_diffusivity),
           ", T₀=", prettysummary(m.reference_temperature),
           ", p₀=", prettysummary(m.reference_pressure),
           ", n=", prettysummary(m.temperature_exponent), ")")

Base.summary(m::LinearThermalConductivity{FT}) where FT =
    string("LinearThermalConductivity{", FT, "}(K₀=", prettysummary(m.reference_conductivity),
           ", K₁=", prettysummary(m.temperature_coefficient),
           ", T₀=", prettysummary(m.reference_temperature), ")")

Base.summary(m::SutherlandViscosity{FT}) where FT =
    string("SutherlandViscosity{", FT, "}(μ₀=", prettysummary(m.reference_viscosity),
           ", T₀=", prettysummary(m.reference_temperature),
           ", Tˢ=", prettysummary(m.sutherland_temperature), ")")

Base.summary(m::LinearViscosity{FT}) where FT =
    string("LinearViscosity{", FT, "}(μ₀=", prettysummary(m.reference_viscosity),
           ", μ₁=", prettysummary(m.temperature_coefficient),
           ", T₀=", prettysummary(m.reference_temperature), ")")

Base.summary(c::KineticCorrection{FT}) where FT =
    string("KineticCorrection{", FT, "}(αᶜ=", prettysummary(c.condensation_coefficient),
           ", αᵀ=", prettysummary(c.thermal_accommodation_coefficient),
           ", Δᵛ=", prettysummary(c.vapor_jump_length),
           ", Δᵀ=", prettysummary(c.thermal_jump_length), ")")

Base.summary(w::RogersYauTerminalVelocity{FT}) where FT =
    string("RogersYauTerminalVelocity{", FT, "}(k₁=", prettysummary(w.quadratic_coefficient),
           ", k₃=", prettysummary(w.linear_coefficient),
           ", k₂=", prettysummary(w.square_root_coefficient),
           ", ρ₀=", prettysummary(w.reference_density), ")")

Base.summary(w::BeardTerminalVelocity{FT}) where FT =
    string("BeardTerminalVelocity{", FT, "}(λ₀=", prettysummary(w.reference_mean_free_path),
           ", μ₀=", prettysummary(w.reference_viscosity),
           ", p₀=", prettysummary(w.reference_pressure),
           ", T₀=", prettysummary(w.reference_temperature), ")")

const SingletonDropletPhysics = Union{NoKineticCorrection, NoVentilation, BeardPruppacherVentilation,
                                      KappaKoehlerActivity, DiluteSolutionActivity, StokesTerminalVelocity}

Base.summary(s::SingletonDropletPhysics) = string(nameof(typeof(s)), "()")

const DropletPhysicsComponent = Union{SingletonDropletPhysics, PowerLawVaporDiffusivity, LinearThermalConductivity,
                                      SutherlandViscosity, LinearViscosity, KineticCorrection,
                                      RogersYauTerminalVelocity, BeardTerminalVelocity}

Base.show(io::IO, c::DropletPhysicsComponent) = print(io, summary(c))

Base.summary(::DropletPhysicsParameters{FT}) where FT = "DropletPhysicsParameters{$FT}"

function Base.show(io::IO, p::DropletPhysicsParameters)
    print(io, summary(p), ":", '\n',
          "├── surface_tension: ", prettysummary(p.surface_tension), '\n',
          "├── vapor_diffusivity: ", summary(p.vapor_diffusivity), '\n',
          "├── thermal_conductivity: ", summary(p.thermal_conductivity), '\n',
          "├── dynamic_viscosity: ", summary(p.dynamic_viscosity), '\n',
          "├── kinetic_correction: ", summary(p.kinetic_correction), '\n',
          "├── ventilation: ", summary(p.ventilation), '\n',
          "├── solution_activity: ", summary(p.solution_activity), '\n',
          "├── terminal_velocity: ", summary(p.terminal_velocity), '\n',
          "└── solver: ", summary(p.solver))
end

#####
##### κ-Köhler equilibrium
#####

"""
$(TYPEDSIGNATURES)

Return the curvature (Kelvin) coefficient ``A = 2σ / (Rᵛ ρᴸ T)`` (m) at temperature `T`.
"""
@inline function curvature_coefficient(T, constants, parameters)
    σ = parameters.surface_tension
    Rᵛ = vapor_gas_constant(constants)
    ρᴸ = constants.liquid.density
    return 2 * σ / (Rᵛ * ρᴸ * T)
end

# The logarithm of the water activity, log aʷ, and the logarithmic derivative of the activity,
# aʷh = ∂aʷ/∂log r, as functions of the dry-to-wet volume ratio z = (rᵈ/r)³ ∈ [0, 1].
# Both are finite at z = 0 (no solute, aʷ = 1); at z = 1 (r = rᵈ), log aʷ = -∞ for κ > 0.
# The logarithm is computed without cancellation so that 𝒮ᵉ = expm1(A/r + log aʷ) keeps
# full relative precision for nearly dilute droplets, also in single precision.
@inline function water_activity(::KappaKoehlerActivity, z, κ)
    Q = 1 - z + κ * z
    valid = Q > 0
    Q₊ = ifelse(valid, Q, one(Q))
    log_aʷ = ifelse(valid, log1p(-κ * z / Q₊), zero(Q)) # aʷ = (1 - z) / Q = 1 - κz / Q
    aʷh = ifelse(valid, 3 * κ * z / Q₊^2, zero(Q))
    return log_aʷ, aʷh
end

@inline function water_activity(::DiluteSolutionActivity, z, κ)
    ζ = 1 - z
    valid = ζ > 0
    ζ₊ = ifelse(valid, ζ, one(ζ))
    log_aʷ = ifelse(valid, -κ * z / ζ₊, ifelse(κ > 0, typemin(ζ), zero(ζ)))
    aʷ = exp(log_aʷ)
    aʷh = ifelse(aʷ > 0, aʷ * (3 * κ * z / ζ₊) / ζ₊, zero(ζ))
    return log_aʷ, aʷh
end

# The logarithmic solute slope h(z) = ∂log aʷ/∂log r and its derivative ∂h/∂z,
# used to locate the maximum of the Köhler curve, where h(z) = A / r.
@inline function solute_slope(::KappaKoehlerActivity, z, κ)
    Q = 1 - z + κ * z
    P = (1 - z) * Q
    ∂P = (1 - z) * (κ - 1) - Q
    h = 3 * κ * z / P
    ∂h = 3 * κ * (P - z * ∂P) / P^2
    return h, ∂h
end

@inline function solute_slope(::DiluteSolutionActivity, z, κ)
    ζ = 1 - z
    h = 3 * κ * z / ζ^2
    ∂h = 3 * κ * (1 + z) / ζ^3
    return h, ∂h
end

@inline dry_to_wet_volume_ratio(r, rᵈ) = min(1, (rᵈ / r)^3)

# 𝒮ᵉ = aʷ exp(A/r) - 1 and ∂𝒮ᵉ/∂r = exp(A/r) (aʷh - aʷ A/r) / r
@inline function equilibrium_supersaturation_and_derivative(activity, r, rᵈ, κ, A)
    z = dry_to_wet_volume_ratio(r, rᵈ)
    log_aʷ, aʷh = water_activity(activity, z, κ)
    𝒮ᵉ = expm1(A / r + log_aʷ)
    ∂𝒮ᵉ = exp(A / r) * (aʷh - exp(log_aʷ) * A / r) / r
    return 𝒮ᵉ, ∂𝒮ᵉ
end

"""
$(TYPEDSIGNATURES)

Return the equilibrium supersaturation ``𝒮ᵉ`` of a droplet with wet radius `r`, dry radius `rᵈ`
and hygroscopicity `κ` at temperature `T`,

```math
1 + 𝒮ᵉ = aʷ(r, rᵈ, κ) \\exp(A / r) , \\qquad A = \\frac{2σ}{Rᵛ ρᴸ T} ,
```

where the water activity ``aʷ`` is given by `parameters.solution_activity`
([`KappaKoehlerActivity`](@ref) or [`DiluteSolutionActivity`](@ref)).

```jldoctest
using Breeze

constants = ThermodynamicConstants()
parameters = DropletPhysicsParameters()
r, rᵈ, κ, T = 1e-6, 5e-8, 0.6, 283.0
𝒮ᵉ = equilibrium_supersaturation(r, rᵈ, κ, T, constants, parameters)
round(𝒮ᵉ, sigdigits=4)

# output
0.001028
```
"""
@inline function equilibrium_supersaturation(r, rᵈ, κ, T, constants, parameters)
    A = curvature_coefficient(T, constants, parameters)
    𝒮ᵉ, _ = equilibrium_supersaturation_and_derivative(parameters.solution_activity, r, rᵈ, κ, A)
    return 𝒮ᵉ
end

# Critical radius in units of the dry radius, sᶜ = rᶜ / rᵈ, the unique root of
# F(s) = s h(z) - A / rᵈ (z = s⁻³). F decreases monotonically from +∞ at s = 1 to -A/rᵈ,
# so the Köhler curve has a single maximum. With s₀ = (3κ rᵈ / A)^{1/2}, the classical
# dilute (r ≫ rᵈ) estimate, F(max(s₀, 1)) ≥ 0 and F(max(2s₀, 2^{1/3})) ≤ 0 for both activities.
@inline function critical_radius_ratio(activity, rᵈ, κ, A, solver)
    a = A / rᵈ
    s₀ = sqrt(3 * κ / a)
    s⁻ = max(s₀, one(s₀))
    s⁺ = max(2 * s₀, cbrt(2 * one(s₀)))
    s★ = s₀ * sqrt(1 + 2 / s₀^3) # first-order correction for finite rᵈ

    F = s -> begin
        z = 1 / s^3
        h, ∂h = solute_slope(activity, z, κ)
        (s * h - a, h - 3 * z * ∂h)
    end

    # F(s⁺) ≤ 0 ≤ F(s⁻)
    return bracketed_newton_solve(F, solver, s★, s⁺, s⁻)
end

"""
$(TYPEDSIGNATURES)

Return the critical radius ``rᶜ`` of a droplet with dry radius `rᵈ > 0` and hygroscopicity `κ`
at temperature `T`: the radius at which the equilibrium supersaturation (the Köhler curve)
is maximum. Droplets with ``r > rᶜ`` are activated. For [`DiluteSolutionActivity`](@ref) and
``rᶜ ≫ rᵈ``, ``rᶜ ≈ (3 κ rᵈ^3 / A)^{1/2}``.

```jldoctest
using Breeze

constants = ThermodynamicConstants()
parameters = DropletPhysicsParameters()
rᶜ = critical_radius(5e-8, 0.6, 283.0, constants, parameters)
round(rᶜ, sigdigits=4)

# output
4.522e-7
```
"""
@inline function critical_radius(rᵈ, κ, T, constants, parameters)
    A = curvature_coefficient(T, constants, parameters)
    sᶜ = critical_radius_ratio(parameters.solution_activity, rᵈ, κ, A, parameters.solver)
    return rᵈ * sᶜ
end

"""
$(TYPEDSIGNATURES)

Return the critical supersaturation ``𝒮ᶜ = 𝒮ᵉ(rᶜ)`` of a droplet with dry radius `rᵈ > 0`
and hygroscopicity `κ` at temperature `T`, the maximum of the Köhler curve.
For [`DiluteSolutionActivity`](@ref) and ``rᶜ ≫ rᵈ``,
``\\log(1 + 𝒮ᶜ) ≈ (4 A^3 / (27 κ rᵈ^3))^{1/2}``.

```jldoctest
using Breeze

constants = ThermodynamicConstants()
parameters = DropletPhysicsParameters()
𝒮ᶜ = critical_supersaturation(5e-8, 0.6, 283.0, constants, parameters)
round(𝒮ᶜ, sigdigits=4)

# output
0.001628
```
"""
@inline function critical_supersaturation(rᵈ, κ, T, constants, parameters)
    rᶜ = critical_radius(rᵈ, κ, T, constants, parameters)
    return equilibrium_supersaturation(rᶜ, rᵈ, κ, T, constants, parameters)
end

"""
$(TYPEDSIGNATURES)

Return the stable (haze) equilibrium radius of a droplet with dry radius `rᵈ > 0` and
hygroscopicity `κ` in air with supersaturation `𝒮` at temperature `T`: the root of
``𝒮ᵉ(r) = 𝒮`` on the rising branch ``rᵈ < r ≤ rᶜ`` of the Köhler curve.
If `𝒮` exceeds the critical supersaturation, no equilibrium exists and the critical radius
``rᶜ`` is returned.

```jldoctest
using Breeze

constants = ThermodynamicConstants()
parameters = DropletPhysicsParameters()
r = equilibrium_radius(5e-8, 0.6, -0.1, 283.0, constants, parameters)
round(r, sigdigits=4)

# output
8.987e-8
```
"""
@inline function equilibrium_radius(rᵈ, κ, 𝒮, T, constants, parameters)
    activity = parameters.solution_activity
    solver = parameters.solver
    A = curvature_coefficient(T, constants, parameters)
    a = A / rᵈ
    sᶜ = critical_radius_ratio(activity, rᵈ, κ, A, solver)
    𝒮ᶜ, _ = equilibrium_supersaturation_and_derivative(activity, sᶜ, one(sᶜ), κ, a)

    # Solute-only (flat-surface, κ-Köhler) estimate; NaN for 𝒮 ≥ 0, in which case
    # the solver starts from the bracket midpoint.
    aʷ = 1 + 𝒮
    z★ = (1 - aʷ) / (1 - aʷ + aʷ * κ)
    s★ = 1 / cbrt(z★)

    # Lengths in units of rᵈ: 𝒮ᵉ(s) - 𝒮 rises from -1 - 𝒮 ≤ 0 at s = 1 to 𝒮ᶜ - 𝒮 at s = sᶜ
    f = s -> begin
        𝒮ᵉ, ∂𝒮ᵉ = equilibrium_supersaturation_and_derivative(activity, s, one(s), κ, a)
        (𝒮ᵉ - 𝒮, ∂𝒮ᵉ)
    end

    s = bracketed_newton_solve(f, solver, s★, one(sᶜ), sᶜ)
    return rᵈ * ifelse(𝒮 < 𝒮ᶜ, s, sᶜ)
end

#####
##### Diffusional growth
#####

# Gas-kinetic factors ψ(r) = r / (r + Δ) + ℓ / r, their derivatives, and the lower bound
# ψᵇ = r / (r + Δ), which also bounds ψ(r′) from below for every r′ ≥ r.
@inline function kinetic_factor(r, Δ, ℓ)
    ψ = r / (r + Δ) + ℓ / r
    ∂ψ = Δ / (r + Δ)^2 - ℓ / r^2
    ψᵇ = r / (r + Δ)
    return ψ, ∂ψ, ψᵇ
end

@inline vapor_kinetic_factor(::NoKineticCorrection, r, ℓᵛ) = (one(r), zero(r), one(r))
@inline thermal_kinetic_factor(::NoKineticCorrection, r, ℓᵀ) = (one(r), zero(r), one(r))
@inline vapor_kinetic_factor(c::KineticCorrection, r, ℓᵛ) = kinetic_factor(r, c.vapor_jump_length, ℓᵛ)
@inline thermal_kinetic_factor(c::KineticCorrection, r, ℓᵀ) = kinetic_factor(r, c.thermal_jump_length, ℓᵀ)

# Gas-kinetic lengths ℓᵛ = (Dᵛ / αᶜ) √(2π / (Rᵛ T)) and ℓᵀ = Kᵀ / (αᵀ ρ cᵖᵈ) √(2π / (Rᵈ T))
@inline kinetic_lengths(::NoKineticCorrection, Dᵛ, Kᵀ, ρ, cᵖᵈ, Rᵛ, Rᵈ, T) = (zero(Dᵛ), zero(Kᵀ))

@inline function kinetic_lengths(c::KineticCorrection, Dᵛ, Kᵀ, ρ, cᵖᵈ, Rᵛ, Rᵈ, T)
    αᶜ = c.condensation_coefficient
    αᵀ = c.thermal_accommodation_coefficient
    ℓᵛ = Dᵛ / αᶜ * sqrt(π / (Rᵛ * T) * 2)
    ℓᵀ = Kᵀ / (αᵀ * ρ * cᵖᵈ) * sqrt(π / (Rᵈ * T) * 2)
    return ℓᵛ, ℓᵀ
end

# Radius-independent coefficients of the growth law at fixed T and p:
# Fᵛ = Cᵛ ψᵛ(r), Fᵀ = Cᵀ ψᵀ(r) with Cᵛ = ρᴸ Rᵛ T / (Dᵛ pᵛ⁺), Cᵀ = (ℒˡ / (Rᵛ T) - 1) ℒˡ ρᴸ / (Kᵀ T).
@inline function diffusional_growth_coefficients(T, p, constants, parameters)
    Rᵛ = vapor_gas_constant(constants)
    Rᵈ = dry_air_gas_constant(constants)
    ρᴸ = constants.liquid.density
    cᵖᵈ = constants.dry_air.heat_capacity
    ℒˡ = liquid_latent_heat(T, constants)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, PlanarLiquidSurface())
    Dᵛ = vapor_diffusivity(parameters.vapor_diffusivity, T, p)
    Kᵀ = thermal_conductivity(parameters.thermal_conductivity, T, p)
    ρ = p / (Rᵈ * T)

    Cᵛ = ρᴸ * Rᵛ * T / (Dᵛ * pᵛ⁺)
    Cᵀ = (ℒˡ / (Rᵛ * T) - 1) * ℒˡ * ρᴸ / (Kᵀ * T)
    ℓᵛ, ℓᵀ = kinetic_lengths(parameters.kinetic_correction, Dᵛ, Kᵀ, ρ, cᵖᵈ, Rᵛ, Rᵈ, T)
    A = curvature_coefficient(T, constants, parameters)

    return (; A, Cᵛ, Cᵀ, ℓᵛ, ℓᵀ)
end

# Φ = r dr/dt = fᵛ G(r) (𝒮 - 𝒮ᵉ(r)) with G = 1 / (Cᵛ ψᵛ + Cᵀ ψᵀ), and ∂Φ/∂r.
@inline function growth_rate_and_derivative(r, rᵈ, κ, 𝒮, fᵛ, coefficients, parameters)
    (; A, Cᵛ, Cᵀ, ℓᵛ, ℓᵀ) = coefficients
    correction = parameters.kinetic_correction
    ψᵛ, ∂ψᵛ, _ = vapor_kinetic_factor(correction, r, ℓᵛ)
    ψᵀ, ∂ψᵀ, _ = thermal_kinetic_factor(correction, r, ℓᵀ)
    G = 1 / (Cᵛ * ψᵛ + Cᵀ * ψᵀ)
    ∂G = - G^2 * (Cᵛ * ∂ψᵛ + Cᵀ * ∂ψᵀ)
    𝒮ᵉ, ∂𝒮ᵉ = equilibrium_supersaturation_and_derivative(parameters.solution_activity, r, rᵈ, κ, A)
    Φ = fᵛ * G * (𝒮 - 𝒮ᵉ)
    ∂Φ = fᵛ * (∂G * (𝒮 - 𝒮ᵉ) - G * ∂𝒮ᵉ)
    return Φ, ∂Φ, 𝒮ᵉ
end

"""
$(TYPEDSIGNATURES)

Return the diffusional growth rate ``r \\, dr/dt`` (m² s⁻¹) of a droplet with wet radius `r`,
dry radius `rᵈ` and hygroscopicity `κ`, in air with supersaturation `𝒮` (over a planar liquid
surface), temperature `T` and pressure `p`, with ventilation factor `fᵛ`
([Rogers and Yau, 1989](@cite rogers1989short); [Pruppacher and Klett, 2010](@cite pruppacher2010microphysics)):

```math
r \\frac{dr}{dt} = \\frac{fᵛ (𝒮 - 𝒮ᵉ(r))}{Fᵀ + Fᵛ} , \\qquad
Fᵀ = \\left(\\frac{ℒˡ}{Rᵛ T} - 1\\right) \\frac{ℒˡ ρᴸ}{K̃ᵀ T} , \\qquad
Fᵛ = \\frac{ρᴸ Rᵛ T}{D̃ᵛ pᵛ⁺(T)} ,
```

where ``𝒮ᵉ`` is the [`equilibrium_supersaturation`](@ref), ``ℒˡ(T)`` the latent heat of
condensation, ``pᵛ⁺(T)`` the saturation vapor pressure over a planar liquid surface, and
``D̃ᵛ = Dᵛ / ψᵛ(r)`` and ``K̃ᵀ = Kᵀ / ψᵀ(r)`` the vapor diffusivity and thermal conductivity
with the gas-kinetic corrections of `parameters.kinetic_correction`.
"""
@inline function diffusional_growth_rate(r, rᵈ, κ, 𝒮, fᵛ, T, p, constants, parameters)
    coefficients = diffusional_growth_coefficients(T, p, constants, parameters)
    Φ, _, _ = growth_rate_and_derivative(r, rᵈ, κ, 𝒮, fᵛ, coefficients, parameters)
    return Φ
end

"""
$(TYPEDSIGNATURES)

Advance the radius of a droplet with wet radius `r`, dry radius `rᵈ` and hygroscopicity `κ`
over the time step `Δt` by diffusional growth at fixed ambient supersaturation `𝒮`,
temperature `T`, pressure `p` and ventilation factor `fᵛ`, using the backward-Euler
discretization of the growth law ``d(r^2)/dt = 2 Φ(r)``, ``Φ = r \\, dr/dt``
(see [`diffusional_growth_rate`](@ref)):

```math
r_{n+1}^2 = r_n^2 + 2 Δt \\, Φ(r_{n+1}) .
```

Returns the named tuple `(radius, condensed_mass)`, where `condensed_mass` is the mass of
water condensed on the droplet during the step (negative for evaporation),
``(4π/3) ρᴸ (r_{n+1}^3 - r_n^3)``, computed from the solved increment of ``r^2``
without cancellation so that it remains accurate in single precision.

The implicit equation is solved for ``y = \\log(r_{n+1}^2 / r_n^2)`` with the safeguarded
Newton–bisection iteration [`Breeze.Solvers.bracketed_newton_solve`](@ref) and
`parameters.solver`, so that the solver tolerances are relative tolerances on ``r^2``.
With ``ζ = r_{n+1}^2 / r_n^2 - 1 = e^y - 1``, the bracket is chosen from the shape of the
Köhler curve, which has a single maximum ``𝒮ᶜ`` at ``rᶜ`` (see [`critical_radius`](@ref)):

* **Evaporation** (``Φ(r_n) < 0``): ``ζ ∈ [(rᵈ / r_n)^2 - 1, 0]``. The residual is
  negative at the dry radius because ``𝒮ᵉ(rᵈ) = -1 < 𝒮``, so the new radius
  never falls below ``rᵈ``, and it lies between ``r_n`` and the stable equilibrium below ``r_n``.
* **Growth** (``Φ(r_n) ≥ 0``): ``ζ ∈ [0, ζ⁺]``, with ``ζ⁺`` the smaller of
  (i) ``2 Δt fᵛ Gᵇ (𝒮 - \\min(𝒮ᵉ(r_n), 0)) / r_n^2``, a bound on the growth over ``Δt`` in which
  ``Gᵇ`` bounds the growth coefficient for all ``r ≥ r_n`` and the Köhler curve
  bounds ``𝒮ᵉ(r) ≥ \\min(𝒮ᵉ(r_n), 0)`` for ``r ≥ r_n``; and (ii), for a haze droplet
  (``r_n < rᶜ`` and ``𝒮 < 𝒮ᶜ``), the Köhler barrier ``(rᶜ / r_n)^2 - 1``, where ``Φ < 0``.
  The barrier keeps the solution on the haze branch for any ``Δt``, so that large time steps
  relax haze droplets to their equilibrium radius instead of spuriously activating them.

The solution is therefore unconditionally stable, free of NaN, and bounded below by `rᵈ`,
for any `Δt`. The input radius is first clamped to `max(r, rᵈ)`. Pure water droplets
(`rᵈ = 0`) require `r > 0`; they shrink by at most a factor `eps(r)` per step.

```jldoctest
using Breeze

constants = ThermodynamicConstants()
parameters = DropletPhysicsParameters()
step = condensational_growth_step(1e-6, 5e-8, 0.6, 0.005, 1.0, 283.0, 9e4, 1.0, constants, parameters)
round(step.radius, sigdigits=4), round(step.condensed_mass, sigdigits=3)

# output
(1.33e-6, 5.66e-15)
```
"""
@inline function condensational_growth_step(r, rᵈ, κ, 𝒮, fᵛ, T, p, Δt, constants, parameters)
    activity = parameters.solution_activity
    correction = parameters.kinetic_correction
    solver = parameters.solver
    coefficients = diffusional_growth_coefficients(T, p, constants, parameters)
    (; A, Cᵛ, Cᵀ, ℓᵛ, ℓᵀ) = coefficients

    r⁰ = max(r, rᵈ)
    x⁰ = r⁰^2
    Φ⁰, _, 𝒮ᵉ⁰ = growth_rate_and_derivative(r⁰, rᵈ, κ, 𝒮, fᵛ, coefficients, parameters)

    # The unknown is y = log(r²/r⁰²), so that the solver tolerances are relative tolerances on r²
    # for any growth or shrinkage factor.

    # Evaporation bracket: the dry radius (pure-water droplets shrink by at most a factor of 1/ε per step)
    yᵈ = 2 * log(max(rᵈ, eps(typeof(r⁰)) * r⁰) / r⁰)

    # Growth bracket (i): bound on the growth over Δt
    _, _, ψᵛᵇ = vapor_kinetic_factor(correction, r⁰, ℓᵛ)
    _, _, ψᵀᵇ = thermal_kinetic_factor(correction, r⁰, ℓᵀ)
    Gᵇ = 1 / (Cᵛ * ψᵛᵇ + Cᵀ * ψᵀᵇ)
    ζᵇ = 2 * Δt * fᵛ * Gᵇ * (𝒮 - min(𝒮ᵉ⁰, 0)) / x⁰

    # Growth bracket (ii): the Köhler barrier for haze droplets
    sᶜ = critical_radius_ratio(activity, rᵈ, κ, A, solver)
    rᶜ = rᵈ * sᶜ
    𝒮ᶜ, _ = equilibrium_supersaturation_and_derivative(activity, rᶜ, rᵈ, κ, A)
    haze = (r⁰ < rᶜ) & (𝒮 < 𝒮ᶜ)
    ζᶜ = (rᶜ / r⁰)^2 - 1
    yᵍ = log1p(max(0, ifelse(haze, min(ζᵇ, ζᶜ), ζᵇ))) # ζᵇ < 0 when evaporating; yᵍ is then unused

    growing = Φ⁰ ≥ 0
    y⁻ = ifelse(growing, zero(yᵈ), yᵈ) # residual ≤ 0
    y⁺ = ifelse(growing, yᵍ, zero(yᵈ)) # residual ≥ 0
    ζᴱ = 2 * Δt * Φ⁰ / x⁰              # forward-Euler first guess, unless it evaporates the droplet
    yᴱ = log1p(ifelse(ζᴱ > -1, ζᴱ, zero(ζᴱ)))

    # R(y) = (r² - r⁰²) / r⁰² - 2 Δt Φ(r) / r⁰², with r = r⁰ exp(y / 2)
    residual = y -> begin
        rⁿ = r⁰ * exp(y / 2)
        Φ, ∂Φ, _ = growth_rate_and_derivative(rⁿ, rᵈ, κ, 𝒮, fᵛ, coefficients, parameters)
        (expm1(y) - 2 * Δt * Φ / x⁰, exp(y) - Δt * ∂Φ * rⁿ / x⁰)
    end

    y = bracketed_newton_solve(residual, solver, yᴱ, y⁻, y⁺)
    y = max(y, yᵈ)
    ζ = expm1(y) # r⁺² / r⁰² - 1
    r⁺ = max(r⁰ * exp(y / 2), rᵈ)

    # r⁺³ - r⁰³ = (r⁺² - r⁰²) (r⁺² + r⁺ r⁰ + r⁰²) / (r⁺ + r⁰)
    Δr³ = ζ * x⁰ * (r⁺^2 + r⁺ * r⁰ + x⁰) / (r⁺ + r⁰)
    ρᴸ = constants.liquid.density
    Δm = π * ρᴸ * Δr³ * 4 / 3

    return (radius = r⁺, condensed_mass = Δm)
end

"""
$(TYPEDSIGNATURES)

Return the mass of liquid water (kg) in a droplet of wet radius `r` and dry radius `rᵈ`,
``(4π/3) ρᴸ (r^3 - rᵈ^3)``, assuming additive volumes of water and solute.
"""
@inline function droplet_water_mass(r, rᵈ, constants)
    ρᴸ = constants.liquid.density
    return π * ρᴸ * (r^3 - rᵈ^3) * 4 / 3
end

#####
##### Ventilation
#####

"""
$(TYPEDSIGNATURES)

Return the ventilation factor ``fᵛ`` for the Reynolds number `Re` and Schmidt number `Sc`
according to `ventilation` ([`BeardPruppacherVentilation`](@ref) or [`NoVentilation`](@ref)).
"""
@inline ventilation_factor(::NoVentilation, Re, Sc) = one(Re)

@inline function ventilation_factor(::BeardPruppacherVentilation, Re, Sc)
    FT = typeof(Re)
    X = cbrt(Sc) * sqrt(Re)
    fᵛ₁ = 1 + convert(FT, 0.108) * X^2
    fᵛ₂ = convert(FT, 0.78) + convert(FT, 0.308) * X
    return ifelse(X < convert(FT, 1.4), fᵛ₁, fᵛ₂)
end

"""
$(TYPEDSIGNATURES)

Return the ventilation factor ``fᵛ`` of a droplet of radius `r` moving with speed
`slip_speed` relative to air at temperature `T` and pressure `p`, with
``\\mathrm{Re} = 2 r ρ |\\boldsymbol{u} - \\boldsymbol{v}| / μ``, ``\\mathrm{Sc} = μ / (ρ Dᵛ)``
and ``ρ = p / (Rᵈ T)``.
"""
@inline function ventilation_factor(r, slip_speed, T, p, constants, parameters)
    Rᵈ = dry_air_gas_constant(constants)
    ρ = p / (Rᵈ * T)
    μ = dynamic_viscosity(parameters.dynamic_viscosity, T)
    Dᵛ = vapor_diffusivity(parameters.vapor_diffusivity, T, p)
    Re = 2 * r * ρ * abs(slip_speed) / μ
    Sc = μ / (ρ * Dᵛ)
    return ventilation_factor(parameters.ventilation, Re, Sc)
end

#####
##### Droplet motion
#####

"""
$(TYPEDSIGNATURES)

Return the [Schiller and Naumann (1933)](@cite SchillerNaumann1933) correction to Stokes drag,
``fᴰ = 1 + 0.15 \\mathrm{Re}^{0.687}``, at droplet Reynolds number `Re`.
"""
@inline function drag_correction(Re)
    FT = typeof(Re)
    return 1 + convert(FT, 0.15) * Re^convert(FT, 0.687)
end

"""
$(TYPEDSIGNATURES)

Return the response time ``τ = τᵖ / fᴰ(\\mathrm{Re})`` (s) of a droplet of radius `r` moving with
speed `slip_speed` relative to air at temperature `T` and pressure `p`: the Stokes response
time ``τᵖ = 2 ρᴸ r^2 / (9 μ)`` reduced by the [`drag_correction`](@ref) at the slip Reynolds
number ``\\mathrm{Re} = 2 r ρ |\\boldsymbol{u} - \\boldsymbol{v}| / μ``.

```jldoctest
using Breeze

constants = ThermodynamicConstants()
parameters = DropletPhysicsParameters()
τ = droplet_response_time(1e-5, 0.0, 293.15, 101325.0, constants, parameters)
round(τ, sigdigits=4)

# output
0.001225
```
"""
@inline function droplet_response_time(r, slip_speed, T, p, constants, parameters)
    Rᵈ = dry_air_gas_constant(constants)
    ρ = p / (Rᵈ * T)
    ρᴸ = constants.liquid.density
    μ = dynamic_viscosity(parameters.dynamic_viscosity, T)
    τᵖ = 2 * ρᴸ * r^2 / (9 * μ)
    Re = 2 * r * ρ * abs(slip_speed) / μ
    return τᵖ / drag_correction(Re)
end

"""
$(TYPEDSIGNATURES)

Advance one component of the velocity `v` of a droplet with response time `τ`
(see [`droplet_response_time`](@ref)) over the time step `Δt`, under drag toward the
fluid velocity `u` and the constant body acceleration `a` (for example ``-g`` for
the vertical component):

```math
\\frac{dv}{dt} = \\frac{u - v}{τ} + a .
```

With `u`, `a` and `τ` frozen over the step, the exact solution relaxes ``v`` toward
``v^∞ = u + a τ``,

```math
v_{n+1} = v^∞ + (v_n - v^∞) e^{-Δt/τ} , \\qquad
Δx = v^∞ Δt + (v_n - v^∞) τ (1 - e^{-Δt/τ}) .
```

Returns the tuple `(v_{n+1}, Δx)` of the new velocity and the displacement over the step.
The update is exact for constant `u`, stable for any `Δt / τ`, and
computed with `expm1` so that it remains accurate as ``Δt / τ → 0``.

```jldoctest
using Breeze

v, Δx = exponential_drag_step(0.0, 1.0, -9.81, 1e-3, 10.0)
round(v, digits=6), round(Δx, digits=6)

# output
(0.99019, 9.90091)
```
"""
@inline function exponential_drag_step(v, u, a, τ, Δt)
    v∞ = u + a * τ
    E = expm1(-Δt / τ) # e^{-Δt/τ} - 1
    v⁺ = v + (v - v∞) * E
    Δx = v∞ * Δt - (v - v∞) * τ * E
    return v⁺, Δx
end

#####
##### Terminal velocity
#####

"""
$(TYPEDSIGNATURES)

Return the terminal fall speed ``𝕎`` (m s⁻¹, positive downward) of a water drop of radius `r`
in still air at temperature `T` and pressure `p`, according to `parameters.terminal_velocity`
([`BeardTerminalVelocity`](@ref), [`RogersYauTerminalVelocity`](@ref) or
[`StokesTerminalVelocity`](@ref)).

```jldoctest
using Breeze

constants = ThermodynamicConstants()
parameters = DropletPhysicsParameters()
𝕎 = droplet_terminal_velocity(5e-4, 293.15, 101325.0, constants, parameters)
round(𝕎, digits=3)

# output
4.012
```
"""
@inline droplet_terminal_velocity(r, T, p, constants, parameters) =
    droplet_terminal_velocity(parameters.terminal_velocity, r, T, p, constants, parameters)

@inline function droplet_terminal_velocity(::StokesTerminalVelocity, r, T, p, constants, parameters)
    g = constants.gravitational_acceleration
    ρᴸ = constants.liquid.density
    μ = dynamic_viscosity(parameters.dynamic_viscosity, T)
    return 2 * ρᴸ * g * r^2 / (9 * μ)
end

@inline function droplet_terminal_velocity(law::RogersYauTerminalVelocity, r, T, p, constants, parameters)
    Rᵈ = dry_air_gas_constant(constants)
    ρ = p / (Rᵈ * T)
    k₁ = law.quadratic_coefficient
    k₃ = law.linear_coefficient
    k₂ = law.square_root_coefficient
    ρ₀ = law.reference_density
    return min(k₁ * r^2, k₃ * r, k₂ * sqrt(ρ₀ / ρ * r))
end

@inline function droplet_terminal_velocity(law::BeardTerminalVelocity, r, T, p, constants, parameters)
    FT = typeof(r)
    g = constants.gravitational_acceleration
    ρᴸ = constants.liquid.density
    Rᵈ = dry_air_gas_constant(constants)
    ρ = p / (Rᵈ * T)
    Δρ = ρᴸ - ρ
    μ = dynamic_viscosity(parameters.dynamic_viscosity, T)
    σ = parameters.surface_tension

    d = min(2 * r, convert(FT, 7e-3))

    # Slip correction with the mean free path of air molecules
    λ = law.reference_mean_free_path * (μ / law.reference_viscosity) *
        (law.reference_pressure / p) * sqrt(T / law.reference_temperature)
    Cˢ = 1 + convert(FT, 2.51) * λ / d

    # Regime 1: Stokes flow with slip correction, d < 19 μm
    𝕎₁ = Δρ * g * d^2 * Cˢ / (18 * μ)

    # Regime 2: 19 μm ≤ d < 1.07 mm, fit in the logarithm of the Davies number
    Da = 4 * ρ * Δρ * g * d^3 / (3 * μ^2)
    X₂ = log(Da)
    Y₂ = evalpoly(X₂, law.regime_2_coefficients)
    Re₂ = Cˢ * exp(Y₂)
    𝕎₂ = μ * Re₂ / (ρ * d)

    # Regime 3: 1.07 mm ≤ d ≤ 7 mm, fit in the logarithm of Bond × (physical property number)^{1/6}
    Bo = 4 * Δρ * g * d^2 / (3 * σ)
    Np⅙ = cbrt(sqrt(σ^3 * ρ^2 / (μ^4 * Δρ * g)))
    X₃ = log(Bo * Np⅙)
    Y₃ = evalpoly(X₃, law.regime_3_coefficients)
    Re₃ = Np⅙ * exp(Y₃)
    𝕎₃ = μ * Re₃ / (ρ * d)

    return ifelse(d < convert(FT, 19e-6), 𝕎₁, ifelse(d < convert(FT, 1.07e-3), 𝕎₂, 𝕎₃))
end
