#####
##### Convective Boundary Layer (CBL) benchmark case
#####
##### Based on Section 4.2 of Sauer & Munoz-Esparza (2020):
##### "The FastEddy® resident-GPU accelerated large-eddy simulation framework"
##### https://doi.org/10.1029/2020MS002100
#####
##### This case represents the dry convective boundary layer at the SWiFT facility
##### near Lubbock, Texas on 4 July 2012 during peak convection (18Z-20Z).
#####

using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: SmagorinskyLilly

using Breeze
using Breeze: CompressibleDynamics

"""
    convective_boundary_layer(arch = CPU();
                              float_type = Float64,
                              Nx = 64, Ny = 64, Nz = 64,
                              dynamics = nothing,
                              advection = WENO(order=5),
                              closure = SmagorinskyLilly())

Create an `AtmosphereModel` for the dry convective boundary layer benchmark case
from [Sauer and Munoz-Esparza (2020)](@cite Sauer2020fasteddy), Section 4.2.

# Arguments
- `arch`: Architecture to run on (`CPU()` or `GPU()`)

# Keyword Arguments
- `float_type`: Floating point precision (`Float32` or `Float64`)
- `Nx, Ny, Nz`: Grid resolution
- `dynamics`: Dynamics formulation. `nothing` defaults to `AnelasticDynamics`.
  Pass a `CompressibleDynamics` instance for compressible formulations.
- `advection`: Advection scheme (default: `WENO(order=5)`)
- `closure`: Turbulence closure (default: `SmagorinskyLilly()`)

# Physical parameters (from Sauer & Munoz-Esparza 2020, Section 4.2)
- Domain: 12 km × 12 km × 3 km
- Geostrophic wind: (Uᵍ, Vᵍ) = (9, 0) m/s
- Latitude: 33.5° N (f ≈ 8.0 × 10⁻⁵ s⁻¹)
- Surface potential temperature: θ₀ = 309 K
- Surface heat flux: 0.35 K⋅m/s
- Initial stratification: neutral below 600 m, dθ/dz = 0.004 K/m above
- Initial perturbations: ±0.25 K in lowest 400 m
"""
function convective_boundary_layer(arch = CPU();
                                   float_type = Float64,
                                   Nx = 64, Ny = 64, Nz = 64,
                                   dynamics = nothing,
                                   advection = WENO(order=5),
                                   closure = SmagorinskyLilly())

    # Set floating point precision
    Oceananigans.defaults.FloatType = float_type

    # Domain size (from paper: 12.0 × 11.9 × 3.0 km, simplified to 12 × 12 × 3 km)
    Lx = 12kilometers
    Ly = 12kilometers
    Lz = 3kilometers

    grid = RectilinearGrid(arch;
        size = (Nx, Ny, Nz),
        x = (0, Lx),
        y = (0, Ly),
        z = (0, Lz),
        halo = (5, 5, 5),
        topology = (Periodic, Periodic, Bounded)
    )

    # Reference state
    # Surface pressure: standard atmosphere
    # Surface potential temperature: 309 K (from paper)
    p₀ = 101325  # Pa
    θ₀ = 309     # K

    constants = ThermodynamicConstants()

    # Build dynamics formulation
    if isnothing(dynamics)
        # Default: anelastic dynamics with reference state
        reference_state = ReferenceState(grid, constants;
            surface_pressure = p₀,
            potential_temperature = θ₀
        )
        dynamics = AnelasticDynamics(reference_state)
    elseif dynamics isa CompressibleDynamics
        # CompressibleDynamics is passed in pre-constructed;
        # set reference_potential_temperature for acoustic substepping if not already set
        dynamics = CompressibleDynamics(dynamics.time_discretization;
            surface_pressure = p₀,
            reference_potential_temperature = θ₀
        )
    end

    # Coriolis parameter for latitude 33.5° N
    # f = 2Ω sin(φ) where Ω = 7.2921 × 10⁻⁵ s⁻¹
    latitude = 33.5
    f = 2 * 7.2921e-5 * sind(latitude)  # ≈ 8.0 × 10⁻⁵ s⁻¹
    coriolis = FPlane(; f)

    # Geostrophic wind: (Uᵍ, Vᵍ) = (9, 0) m/s
    Uᵍ = 9.0  # m/s
    Vᵍ = 0.0  # m/s
    geostrophic = geostrophic_forcings(z -> Uᵍ, z -> Vᵍ)

    # Surface fluxes
    # Surface heat flux: 0.35 K⋅m/s (kinematic)
    # Convert to mass flux: multiply by surface density
    FT = eltype(grid)
    q₀ = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
    ρ₀ = Breeze.Thermodynamics.density(FT(θ₀), FT(p₀), q₀, constants)

    w′θ′ = FT(0.35)  # K⋅m/s (kinematic heat flux)
    ρθ_flux = ρ₀ * w′θ′
    ρθ_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρθ_flux))

    # Surface momentum flux (drag)
    # Using bulk drag with roughness length z₀ = 0.05 m
    # For simplicity, use a friction velocity approach similar to BOMEX
    u★ = FT(0.4)  # m/s (estimated from typical CBL conditions)
    ϵ = FT(1e-10)  # small number to avoid division by zero
    @inline ρu_drag(x, y, t, ρu, ρv, p) = -p.ρ₀ * p.u★^2 * ρu / sqrt(ρu^2 + ρv^2 + p.ϵ)
    @inline ρv_drag(x, y, t, ρu, ρv, p) = -p.ρ₀ * p.u★^2 * ρv / sqrt(ρu^2 + ρv^2 + p.ϵ)

    ρu_drag_bc = FluxBoundaryCondition(ρu_drag, field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★, ϵ))
    ρv_drag_bc = FluxBoundaryCondition(ρv_drag, field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★, ϵ))
    ρu_bcs = FieldBoundaryConditions(bottom = ρu_drag_bc)
    ρv_bcs = FieldBoundaryConditions(bottom = ρv_drag_bc)

    # Forcings
    forcing = (; ρu = geostrophic.ρu, ρv = geostrophic.ρv)

    # Build the model
    model = AtmosphereModel(grid;
        dynamics,
        advection,
        closure,
        coriolis,
        forcing,
        boundary_conditions = (ρθ = ρθ_bcs, ρu = ρu_bcs, ρv = ρv_bcs)
    )

    # Set initial conditions
    # Potential temperature profile:
    #   θ = θ₀ for z ≤ 600 m
    #   θ = θ₀ + 0.004 * (z - 600) for z > 600 m
    # Plus random perturbations ±0.25 K in lowest 400 m
    z_inv = 600.0   # m, inversion height
    Γ = 0.004       # K/m, lapse rate above inversion
    δθ = 0.25       # K, perturbation amplitude
    z_pert = 400.0  # m, depth of perturbations

    θᵢ(x, y, z) = θ₀ + max(0, z - z_inv) * Γ + δθ * (2 * rand() - 1) * (z < z_pert)
    uᵢ(x, y, z) = Uᵍ
    vᵢ(x, y, z) = Vᵍ

    set!(model, θ = θᵢ, u = uᵢ, v = vᵢ)

    return model
end
