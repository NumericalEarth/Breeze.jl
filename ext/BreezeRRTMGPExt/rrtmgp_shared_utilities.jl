#####
##### Shared utilities for clear-sky and all-sky RRTMGP radiation
#####

using Oceananigans.Operators: ℑzᵃᵃᶠ, Δzᶜᶜᶜ
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: launch!

using RRTMGP.AtmosphericStates: AtmosphericState

using Breeze.AtmosphereModels: BackgroundAtmosphere, specific_humidity

#####
##### Gas state update (shared by clear-sky and all-sky)
#####

function update_rrtmgp_gas_state!(as::AtmosphericState, model, surface_temperature,
                                  background_atmosphere::BackgroundAtmosphere, params)
    grid = model.grid
    arch = architecture(grid)

    pᵣ = model.dynamics.reference_state.pressure
    T = model.temperature
    qᵛ = specific_humidity(model)

    g = params.grav
    mᵈ = params.molmass_dryair
    mᵛ = params.molmass_water
    ℕᴬ = params.avogad
    O₃ = background_atmosphere.O₃  # Can be ConstantField or Field

    launch!(arch, grid, :xyz, _update_rrtmgp_gas_state!, as, grid, pᵣ, T, qᵛ, surface_temperature, g, mᵈ, mᵛ, ℕᴬ, O₃)
    return nothing
end

@kernel function _update_rrtmgp_gas_state!(as, grid, pᵣ, T, qᵛ, surface_temperature, g, mᵈ, mᵛ, ℕᴬ, O₃)
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
        pᶜ = pᵣ[i, j, k]
        qᵛₖ = max(qᵛ[i, j, k], zero(eltype(qᵛ)))

        # Face values at k and k+1 (needed for column dry air mass and level temperatures)
        pᶠₖ = ℑzᵃᵃᶠ(i, j, k, grid, pᵣ)
        pᶠₖ₊₁ = ℑzᵃᵃᶠ(i, j, k+1, grid, pᵣ)
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
            pᶠ[Nz+1, c] = ℑzᵃᵃᶠ(i, j, Nz+1, grid, pᵣ)
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
##### Copy fluxes to Oceananigans fields (shared by clear-sky and all-sky)
#####

function copy_rrtmgp_fluxes_to_fields!(rtm, solver, grid)
    arch = architecture(grid)
    Nz = size(grid, 3)

    lw_flux_up = solver.lws.flux.flux_up
    lw_flux_dn = solver.lws.flux.flux_dn
    sw_flux_dn = solver.sws.flux.flux_dn  # Total SW (direct + diffuse)

    ℐ_lw_up = rtm.upwelling_longwave_flux
    ℐ_lw_dn = rtm.downwelling_longwave_flux
    ℐ_sw_dn = rtm.downwelling_shortwave_flux

    Nx, Ny, Nz = size(grid)
    launch!(arch, grid, (Nx, Ny, Nz+1), _copy_rrtmgp_fluxes!,
            ℐ_lw_up, ℐ_lw_dn, ℐ_sw_dn, lw_flux_up, lw_flux_dn, sw_flux_dn, grid)

    return nothing
end

@kernel function _copy_rrtmgp_fluxes!(ℐ_lw_up, ℐ_lw_dn, ℐ_sw_dn,
                                      lw_flux_up, lw_flux_dn, sw_flux_dn, grid)
    i, j, k = @index(Global, NTuple)

    c = rrtmgp_column_index(i, j, grid.Nx)

    @inbounds begin
        ℐ_lw_up[i, j, k] = lw_flux_up[k, c]
        ℐ_lw_dn[i, j, k] = -lw_flux_dn[k, c]
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
    ℐ_sw_dn = rtm.downwelling_shortwave_flux
    flux_div = rtm.flux_divergence
    launch!(arch, grid, :xyz, _compute_radiation_flux_divergence!, flux_div, ℐ_lw_up, ℐ_lw_dn, ℐ_sw_dn, grid)
    return nothing
end

@kernel function _compute_radiation_flux_divergence!(flux_div, ℐ_lw_up, ℐ_lw_dn, ℐ_sw_dn, grid)
    i, j, k = @index(Global, NTuple)
    # Net flux at faces k and k+1 (positive upward)
    @inbounds begin
        F_k  = ℐ_lw_up[i, j, k]   + ℐ_lw_dn[i, j, k]   + ℐ_sw_dn[i, j, k]
        F_k1 = ℐ_lw_up[i, j, k+1] + ℐ_lw_dn[i, j, k+1] + ℐ_sw_dn[i, j, k+1]
    end
    Δz = Δzᶜᶜᶜ(i, j, k, grid)
    # Flux divergence: -dF/dz (positive when flux convergence warms)
    @inbounds flux_div[i, j, k] = -(F_k1 - F_k) / Δz
end
