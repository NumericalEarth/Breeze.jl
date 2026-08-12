#####
##### Shared utilities for clear-sky and all-sky RRTMGP radiation
#####

using Oceananigans.Operators: ℑzᵃᵃᶠ, Δzᶜᶜᶜ
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: ConstantField
using Oceananigans.Utils: launch!

using RRTMGP: RRTMGPSolver
using RRTMGP.AtmosphericStates: AtmosphericState

using Breeze.AtmosphereModels: BackgroundAtmosphere, specific_humidity

#####
##### Gas state update (shared by clear-sky and all-sky)
#####

function update_rrtmgp_gas_state!(as::AtmosphericState, model, surface_temperature,
                                  background_atmosphere::BackgroundAtmosphere, params)
    grid = model.grid
    arch = architecture(grid)

    # RRTMGP assumes level pressures are positive and monotonically decreasing with height. That
    # holds for the anelastic hydrostatic reference exactly, and for the compressible diagnosed
    # pressure in practice, whose hydrostatic part dominates dynamic/acoustic perturbations by
    # orders of magnitude.
    p = dynamics_pressure(model.dynamics)
    T = model.temperature
    qᵛ = specific_humidity(model)

    g = params.grav
    mᵈ = params.molmass_dryair
    mᵛ = params.molmass_water
    ℕᴬ = params.avogad
    O₃ = background_atmosphere.O₃  # Can be ConstantField or Field

    launch!(arch, grid, :xyz, _update_rrtmgp_gas_state!, as, grid, p, T, qᵛ, surface_temperature, g, mᵈ, mᵛ, ℕᴬ, O₃)
    return nothing
end

@kernel function _update_rrtmgp_gas_state!(as, grid, p, T, qᵛ, surface_temperature, g, mᵈ, mᵛ, ℕᴬ, O₃)
    i, j, k = @index(Global, NTuple)

    Nz = size(grid, 3)
    c = rrtmgp_column_index(i, j, grid.Nx)

    layerdata = as.layerdata
    pᶠ = as.p_lev
    Tᶠ = as.t_lev
    T₀ = as.t_sfc

    vmr_h2o = as.vmr.vmr_h2o
    vmr_o3 = as.vmr.vmr_o3

    @inbounds begin
        # Layer (cell-centered) values
        pᶜ = p[i, j, k]
        qᵛₖ = max(qᵛ[i, j, k], zero(eltype(qᵛ)))

        # Face values at k and k+1 (needed for column dry air mass and level temperatures)
        pᶠₖ = ℑzᵃᵃᶠ(i, j, k, grid, p)
        pᶠₖ₊₁ = ℑzᵃᵃᶠ(i, j, k+1, grid, p)
        Tᶠₖ = ℑzᵃᵃᶠ(i, j, k, grid, T)
        Tᶠₖ₊₁ = ℑzᵃᵃᶠ(i, j, k+1, grid, T)

        # Use face-averaged temperature for the RRTMGP layer temperature.
        # This ensures consistency between RRTMGP's layer and level Planck sources,
        # preventing 2Δz oscillations in the radiative heating rate that arise when
        # RRTMGP's linear-in-tau source correction amplifies lay_source − lev_source
        # mismatches at the grid Nyquist frequency.
        Tᶜ = (Tᶠₖ + Tᶠₖ₊₁) / 2

        # RRTMGP Planck/source lookup tables are defined over a finite temperature range.
        # Clamp temperatures to avoid extrapolation that can yield tiny negative source values
        # and trigger DomainErrors in geometric means.
        # TODO: This clamping should ideally be done internally in RRTMGP.jl.
        Tmin = 160
        Tmax = 355
        Tᶜ = clamp(Tᶜ, Tmin, Tmax)
        Tᶠₖ = clamp(Tᶠₖ, Tmin, Tmax)

        # Store level values
        pᶠ[k, c] = pᶠₖ
        Tᶠ[k, c] = Tᶠₖ

        # Topmost level (once)
        if k == 1
            pᶠ[Nz+1, c] = ℑzᵃᵃᶠ(i, j, Nz+1, grid, p)
            Tᴺ⁺¹ = ℑzᵃᵃᶠ(i, j, Nz+1, grid, T)
            Tᶠ[Nz+1, c] = clamp(Tᴺ⁺¹, Tmin, Tmax)
            T₀[c] = clamp(surface_temperature[i, j, 1], Tmin, Tmax)
        end

        # Column dry air mass: molecules / cm² of dry air
        Δp = max(pᶠₖ - pᶠₖ₊₁, zero(pᶠₖ))
        dry_mass_fraction = 1 - qᵛₖ
        dry_mass_per_area = (Δp / g) * dry_mass_fraction
        m⁻²_to_cm⁻² = convert(eltype(pᶜ), 1e4)
        column_dry = dry_mass_per_area / mᵈ * ℕᴬ / m⁻²_to_cm⁻² # (molecules / m²) -> (molecules / cm²)

        # Populate layerdata: (column_dry, pᶜ, Tᶜ, relative_humidity)
        layerdata[1, k, c] = column_dry
        layerdata[2, k, c] = pᶜ
        layerdata[3, k, c] = Tᶜ
        layerdata[4, k, c] = zero(eltype(Tᶜ))

        # H₂O volume mixing ratio from specific humidity
        r = qᵛₖ / dry_mass_fraction
        vmr_h2o[k, c] = r * (mᵈ / mᵛ)

        # O₃ volume mixing ratio - index into field (works for ConstantField or Field)
        vmr_o3[k, c] = O₃[i, j, k]
    end
end

#####
##### Surface boundary conditions (shared by gray, clear-sky and all-sky)
#####

"""
$(TYPEDSIGNATURES)

Throw an `ArgumentError` for any keyword whose value is a `Number` outside ``[0, 1]``.

Emissivity and albedo are fractions, so a scalar outside the unit interval is a user error — an albedo
given in percent, say — worth rejecting at construction rather than carrying into the solver. Values
that are not `Number`s (a field, a dataset, `nothing`) pass through: a field is validated by whatever
built it.
"""
function validate_surface_fractions(; kw...)
    for (name, value) in kw
        value isa Number && !(0 <= value <= 1) &&
            throw(ArgumentError("`$name` must lie in [0, 1]; received $value."))
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Wrap a scalar surface property in a `ConstantField` of the working precision, passing anything
already field-valued through unchanged, so that emissivity and both albedos are uniformly
field-valued whether the user supplied a number, a field, or a dataset.
"""
constant_field_property(x::Number, FT) = ConstantField(convert(FT, x))
constant_field_property(x, FT) = x

"""
$(TYPEDSIGNATURES)

Copy the surface emissivity `ε` and the direct and diffuse albedos `αᵈ`, `αˢ` from
`surface_properties` into RRTMGP's band-by-column boundary-condition arrays `ε₀`, `αᵈ₀`, `αˢ₀`.

Nothing else writes those arrays, so a spatially varying emissivity or albedo would otherwise never
reach the solver, which would read whatever the allocation happened to contain. Call once at
construction and again before every solve, so a property that evolves is picked up rather than frozen.

Breeze treats all three properties as spectrally grey: every band receives the same value.
"""
function update_rrtmgp_surface_boundary_conditions!(ε₀, αᵈ₀, αˢ₀, surface_properties, grid)
    arch = architecture(grid)

    launch!(arch, grid, :xy, _update_rrtmgp_surface_boundary_conditions!,
            ε₀, αᵈ₀, αˢ₀,
            surface_properties.surface_emissivity,
            surface_properties.direct_surface_albedo,
            surface_properties.diffuse_surface_albedo,
            grid)

    return nothing
end

# Full-spectrum (clear-sky and all-sky) models keep both RTE solvers inside one `RRTMGPSolver`,
# reached through RRTMGP's own accessors rather than its internal field nesting.
update_rrtmgp_surface_boundary_conditions!(solver::RRTMGPSolver, surface_properties, grid) =
    update_rrtmgp_surface_boundary_conditions!(RRTMGP.surface_emissivity(solver),
                                               RRTMGP.direct_sw_surface_albedo(solver),
                                               RRTMGP.diffuse_sw_surface_albedo(solver),
                                               surface_properties, grid)

@kernel function _update_rrtmgp_surface_boundary_conditions!(ε₀, αᵈ₀, αˢ₀, ε, αᵈ, αˢ, grid)
    i, j = @index(Global, NTuple)

    c = rrtmgp_column_index(i, j, grid.Nx)

    @inbounds begin
        εᵢⱼ = ε[i, j, 1]
        αᵈᵢⱼ = αᵈ[i, j, 1]
        αˢᵢⱼ = αˢ[i, j, 1]

        for b in 1:size(ε₀, 1)
            ε₀[b, c] = εᵢⱼ
        end

        for b in 1:size(αᵈ₀, 1)
            αᵈ₀[b, c] = αᵈᵢⱼ
            αˢ₀[b, c] = αˢᵢⱼ
        end
    end
end

#####
##### Copy fluxes to Oceananigans fields (shared by clear-sky and all-sky)
#####

function copy_rrtmgp_fluxes_to_fields!(rtm, solver, grid)
    arch = architecture(grid)

    lw_flux_up = solver.lws.flux.flux_up
    lw_flux_dn = solver.lws.flux.flux_dn
    sw_flux_up = solver.sws.flux.flux_up
    sw_flux_dn = solver.sws.flux.flux_dn  # Total SW (direct + diffuse)

    ℐ_lw_up = rtm.upwelling_longwave_flux
    ℐ_lw_dn = rtm.downwelling_longwave_flux
    ℐ_sw_up = rtm.upwelling_shortwave_flux
    ℐ_sw_dn = rtm.downwelling_shortwave_flux

    Nx, Ny, Nz = size(grid)
    launch!(arch, grid, (Nx, Ny, Nz+1), _copy_rrtmgp_fluxes!,
            ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn,
            lw_flux_up, lw_flux_dn, sw_flux_up, sw_flux_dn, grid)

    return nothing
end

@kernel function _copy_rrtmgp_fluxes!(ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn,
                                      lw_flux_up, lw_flux_dn, sw_flux_up, sw_flux_dn, grid)
    i, j, k = @index(Global, NTuple)

    c = rrtmgp_column_index(i, j, grid.Nx)

    @inbounds begin
        ℐ_lw_up[i, j, k] = lw_flux_up[k, c]
        ℐ_lw_dn[i, j, k] = -lw_flux_dn[k, c]
        ℐ_sw_up[i, j, k] = sw_flux_up[k, c]
        ℐ_sw_dn[i, j, k] = -sw_flux_dn[k, c]
    end
end

#####
##### Compute radiation flux divergence from radiative fluxes
#####

function compute_radiation_flux_divergence!(rtm, grid)
    arch = architecture(grid)
    ℐ_lw_up = rtm.upwelling_longwave_flux
    ℐ_lw_dn = rtm.downwelling_longwave_flux
    ℐ_sw_up = rtm.upwelling_shortwave_flux
    ℐ_sw_dn = rtm.downwelling_shortwave_flux
    flux_div = rtm.flux_divergence
    launch!(arch, grid, :xyz, _compute_radiation_flux_divergence!,
            flux_div, ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn, grid)
    return nothing
end

@kernel function _compute_radiation_flux_divergence!(flux_div, ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn, grid)
    i, j, k = @index(Global, NTuple)
    # Net flux at faces k and k+1 (positive upward)
    @inbounds begin
        F_k  = ℐ_lw_up[i, j, k]   + ℐ_lw_dn[i, j, k]   + ℐ_sw_up[i, j, k]   + ℐ_sw_dn[i, j, k]
        F_k1 = ℐ_lw_up[i, j, k+1] + ℐ_lw_dn[i, j, k+1] + ℐ_sw_up[i, j, k+1] + ℐ_sw_dn[i, j, k+1]
    end
    Δz = Δzᶜᶜᶜ(i, j, k, grid)
    # Flux divergence: -dF/dz (positive when flux convergence warms)
    @inbounds flux_div[i, j, k] = -(F_k1 - F_k) / Δz
end

# The constructors accept `surface_temperature = nothing` so that a coupled model can bind
# its interface surface temperature after construction; solving without one is an error.
function assert_bound_surface_temperature(rtm)
    isnothing(rtm.surface_properties.surface_temperature) && throw(ArgumentError(
        "This RadiativeTransferModel has no surface temperature: construct it with " *
        "`surface_temperature = ...`, or bind one before the first radiation update " *
        "(coupled models wire their interface surface temperature automatically)."))
    return nothing
end
