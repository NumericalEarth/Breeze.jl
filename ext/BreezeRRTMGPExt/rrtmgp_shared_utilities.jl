#####
##### Shared utilities for gray, clear-sky and all-sky RRTMGP radiation
#####

using Oceananigans.Operators: ℑzᵃᵃᶠ
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: ConstantField
using Oceananigans.Utils: launch!

using Breeze.AtmosphereModels: SurfaceRadiation, BackgroundAtmosphere, specific_humidity

using RRTMGP: RRTMGPSolver
using RRTMGP.AtmosphericStates: AtmosphericState
using RRTMGP.VolumeMixingRatios: VmrGM

#####
##### Volume mixing ratio initialization (shared by clear-sky and all-sky)
#####

# Zero-initialized global-mean volume mixing ratios: H₂O and O₃ vary per layer
# and column, all other gases are well-mixed scalars.
function initialize_global_mean_vmr(Ngas, Nz, Nc, FT, ArrayType)
    vmr_h2o = ArrayType{FT}(undef, Nz, Nc)
    vmr_o3 = ArrayType{FT}(undef, Nz, Nc)
    global_mean_vmr = ArrayType{FT}(undef, Ngas)

    fill!(vmr_h2o, zero(FT))
    fill!(vmr_o3, zero(FT))
    fill!(global_mean_vmr, zero(FT))

    return VmrGM(vmr_h2o, vmr_o3, global_mean_vmr)
end

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

# RRTMGP Planck/source lookup tables are defined over a finite temperature range. Atmospheric and
# surface temperatures are clamped into it to avoid extrapolation that can yield tiny negative
# source values and trigger DomainErrors in geometric means.
# TODO: This clamping should ideally be done internally in RRTMGP.jl.
const rrtmgp_temperature_bounds = (160, 355)

@kernel function _update_rrtmgp_gas_state!(as, grid, p, T, qᵛ, surface_temperature, g, mᵈ, mᵛ, ℕᴬ, O₃)
    i, j, k = @index(Global, NTuple)

    Nz = size(grid, 3)
    c = column_index(i, j, grid.Nx)

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

        Tmin, Tmax = rrtmgp_temperature_bounds
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
            T₀[c] = solved_surface_temperature(surface_temperature[i, j, 1], rrtmgp_temperature_bounds)
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

Validate and materialize the user's surface properties into a `SurfaceRadiation`.

Albedo is given either as `surface_albedo`, used for both the direct and diffuse albedo, or as
*both* `direct_surface_albedo` and `diffuse_surface_albedo`. Scalar emissivity, albedos and surface
temperature become `ConstantField`s of the grid's float type; `surface_temperature = nothing` stays
`nothing` until a coupled model binds one.
"""
function materialize_surface_radiation(grid, solar_position; surface_temperature, surface_emissivity,
                                       surface_albedo, direct_surface_albedo, diffuse_surface_albedo)

    validate_surface_fractions(; surface_emissivity, surface_albedo,
                                 direct_surface_albedo, diffuse_surface_albedo)

    direct_surface_albedo, diffuse_surface_albedo =
        resolve_surface_albedos(surface_albedo, direct_surface_albedo, diffuse_surface_albedo, grid, solar_position)

    surface_emissivity = materialize_surface_property(surface_emissivity, grid, solar_position)

    FT = eltype(grid)
    surface_temperature = constant_field_property(surface_temperature, FT)
    surface_emissivity = constant_field_property(surface_emissivity, FT)
    direct_surface_albedo = constant_field_property(direct_surface_albedo, FT)
    diffuse_surface_albedo = constant_field_property(diffuse_surface_albedo, FT)

    return SurfaceRadiation(surface_temperature, surface_emissivity,
                            direct_surface_albedo, diffuse_surface_albedo)
end

# Seed RRTMGP's surface temperature from a constant one; any other is written at each radiation update.
function initialize_surface_temperature!(T₀, surface_temperature::ConstantField)
    T₀ .= surface_temperature.constant
    return nothing
end

initialize_surface_temperature!(T₀, surface_temperature) = nothing

"""
$(TYPEDSIGNATURES)

Copy the surface emissivity `ε` and the direct and diffuse albedos `αᵈ`, `αˢ` from
`surface_radiation` into RRTMGP's band-by-column boundary-condition arrays `ε₀`, `αᵈ₀`, `αˢ₀`.

Nothing else writes those arrays, so a spatially varying emissivity or albedo would otherwise never
reach the solver, which would read whatever the allocation happened to contain. Call once at
construction and again before every solve, so a property that evolves is picked up rather than frozen.

Breeze treats all three properties as spectrally grey: every band receives the same value.
"""
function update_rrtmgp_surface_boundary_conditions!(ε₀, αᵈ₀, αˢ₀, surface_radiation, grid)
    arch = architecture(grid)

    launch!(arch, grid, :xy, _update_rrtmgp_surface_boundary_conditions!,
            ε₀, αᵈ₀, αˢ₀,
            surface_radiation.surface_emissivity,
            surface_radiation.direct_surface_albedo,
            surface_radiation.diffuse_surface_albedo,
            grid)

    return nothing
end

# Full-spectrum (clear-sky and all-sky) models keep both RTE solvers inside one `RRTMGPSolver`,
# reached through RRTMGP's own accessors rather than its internal field nesting.
update_rrtmgp_surface_boundary_conditions!(solver::RRTMGPSolver, surface_radiation, grid) =
    update_rrtmgp_surface_boundary_conditions!(RRTMGP.surface_emissivity(solver),
                                               RRTMGP.direct_sw_surface_albedo(solver),
                                               RRTMGP.diffuse_sw_surface_albedo(solver),
                                               surface_radiation, grid)

@kernel function _update_rrtmgp_surface_boundary_conditions!(ε₀, αᵈ₀, αˢ₀, ε, αᵈ, αˢ, grid)
    i, j = @index(Global, NTuple)

    c = column_index(i, j, grid.Nx)

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

# `batch_rows` and `j_offset` name the slab of latitude rows just solved. The presentation
# buffers, (Nz+1, Nc) views refreshed by update_lw_fluxes!/update_sw_fluxes!, are batch-width,
# so the kernel reads them batch-locally and writes the global row.
function copy_rrtmgp_fluxes_to_fields!(rtm, solver, grid, batch_rows, j_offset)
    copy_rrtmgp_longwave_fluxes!(rtm.upwelling_longwave_flux, rtm.downwelling_longwave_flux,
                                 solver, grid, batch_rows, j_offset)

    # Total SW (direct + diffuse)
    copy_rrtmgp_flux_pair!(rtm.upwelling_shortwave_flux, rtm.downwelling_shortwave_flux,
                           RRTMGP.sw_flux_up(solver), RRTMGP.sw_flux_dn(solver), grid, batch_rows, j_offset)

    return nothing
end

# The longwave fluxes of the last longwave solve, into `ℐ_lw_up` and `ℐ_lw_dn`
copy_rrtmgp_longwave_fluxes!(ℐ_lw_up, ℐ_lw_dn, solver, grid, batch_rows, j_offset) =
    copy_rrtmgp_flux_pair!(ℐ_lw_up, ℐ_lw_dn, RRTMGP.lw_flux_up(solver), RRTMGP.lw_flux_dn(solver),
                           grid, batch_rows, j_offset)

function copy_rrtmgp_flux_pair!(ℐ_up, ℐ_dn, flux_up, flux_dn, grid, batch_rows, j_offset)
    Nx, _, Nz = size(grid)
    launch!(architecture(grid), grid, (Nx, batch_rows, Nz+1), _copy_rrtmgp_flux_pair!,
            ℐ_up, ℐ_dn, flux_up, flux_dn, grid, j_offset)
    return nothing
end

@kernel function _copy_rrtmgp_flux_pair!(ℐ_up, ℐ_dn, flux_up, flux_dn, grid, j_offset)
    i, j, k = @index(Global, NTuple)

    c = column_index(i, j, grid.Nx)  # batch-local column
    jᵍ = j + j_offset                # global latitude row

    @inbounds begin
        ℐ_up[i, jᵍ, k] = flux_up[k, c]
        ℐ_dn[i, jᵍ, k] = -flux_dn[k, c]
    end
end
