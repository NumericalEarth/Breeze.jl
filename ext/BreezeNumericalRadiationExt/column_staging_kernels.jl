#####
##### Column staging kernels: grid → spectral columns → grid
#####
##### Kernel A stages every grid cell into its column layer and every grid face into its column
##### interface. Kernel B stacks the extension layers above the grid top, hydrostatically. Kernel D
##### copies the solved fluxes back onto the grid's faces with Breeze's sign convention.
#####

"""
$(TYPEDSIGNATURES)

Stage the grid's cells and faces of `model` into `columns` (kernel A): layer pressure and
temperature, the molar amounts of dry air, water vapor and ozone, and the cloud water paths,
plus interface pressure and temperature — interior faces by interpolation, the bottom and top
faces by extrapolation from the adjacent cells (never from the halo).
"""
function stage_spectral_columns!(columns::SpectralColumns, model, background_atmosphere)
    grid = model.grid
    arch = architecture(grid)
    constants = model.thermodynamic_constants

    g = constants.gravitational_acceleration
    Mᵈ = constants.dry_air.molar_mass
    Mᵛ = constants.vapor.molar_mass

    launch!(arch, grid, :xyz, _stage_spectral_columns!,
            columns, grid,
            dynamics_pressure(model.dynamics), model.temperature, total_density(model.dynamics),
            specific_prognostic_moisture(model), model.microphysics, model.microphysical_fields,
            background_atmosphere.O₃, g, Mᵈ, Mᵛ)

    return nothing
end

@kernel function _stage_spectral_columns!(columns, grid, p, T, ρ, qᵛᵉ, microphysics, microphysical_fields, O₃, g, Mᵈ, Mᵛ)
    i, j, k = @index(Global, NTuple)

    Nz = size(grid, 3)
    N = number_of_layers(columns)
    c = column_index(i, j, grid.Nx)
    kᶜ = N + 1 - k  # column layer of cell k
    kᶠ = N + 2 - k  # column interface of face k (the bottom face of cell k)

    @inbounds begin
        ρᵢ = ρ[i, j, k]
        Δz = Δzᶜᶜᶜ(i, j, k, grid)

        q = grid_moisture_fractions(i, j, k, grid, microphysics, ρᵢ, qᵛᵉ[i, j, k], microphysical_fields)
        qᵛ = max(0, q.vapor)
        qˡ = max(0, q.liquid)
        qⁱ = max(0, q.ice)

        # Molar column amounts (mol m⁻²) from the mass of each constituent in the cell
        n_dry = ρᵢ * (1 - qᵛ - qˡ - qⁱ) * Δz / Mᵈ
        n_h2o = ρᵢ * qᵛ * Δz / Mᵛ

        columns.pressure_layers[c, kᶜ] = p[i, j, k]
        columns.temperature_layers[c, kᶜ] = T[i, j, k]
        columns.dry_air[c, kᶜ] = n_dry
        columns.water_vapor[c, kᶜ] = n_h2o
        columns.ozone[c, kᶜ] = O₃[i, j, k] * n_dry
        columns.liquid_water_path[c, kᶜ] = ρᵢ * qˡ * Δz
        columns.ice_water_path[c, kᶜ] = ρᵢ * qⁱ * Δz

        if k == 1
            columns.pressure_interfaces[c, kᶠ] = bottom_face_pressure(i, j, grid, p, ρ, g)
            columns.temperature_interfaces[c, kᶠ] = bottom_face_temperature(i, j, grid, T)
        else
            columns.pressure_interfaces[c, kᶠ] = ℑzᵃᵃᶠ(i, j, k, grid, p)
            columns.temperature_interfaces[c, kᶠ] = ℑzᵃᵃᶠ(i, j, k, grid, T)
        end

        # The top face of the grid, column interface N + 1 - Nz
        if k == Nz
            columns.pressure_interfaces[c, kᶠ - 1] = top_face_pressure(i, j, grid, p, ρ, g)
            columns.temperature_interfaces[c, kᶠ - 1] = top_face_temperature(i, j, grid, T)
        end
    end
end

"""
$(TYPEDSIGNATURES)

Stack the extension layers of `columns.extension` above the grid top of every column (kernel B):
the temperature profile anchored to the grid's top face, hydrostatic interface pressures with the
virtual temperature, log-mean layer pressures, and the moist-molar-mass gas amounts
`n_dry = Δp / (g (Mᵈ + Mᵛ χ))`, `n_h2o = χ n_dry`, `n_o3 = χ_o3 n_dry`. Extension layers are clear.
"""
function extend_spectral_columns!(columns::SpectralColumns, extension::MaterializedColumnExtension, model)
    grid = model.grid
    arch = architecture(grid)
    constants = model.thermodynamic_constants

    g = constants.gravitational_acceleration
    Mᵈ = constants.dry_air.molar_mass
    Mᵛ = constants.vapor.molar_mass
    Rᵈ = constants.molar_gas_constant / Mᵈ

    launch!(arch, grid, :xy, _extend_spectral_columns!, columns, extension, grid, g, Rᵈ, Mᵈ, Mᵛ)

    return nothing
end

extend_spectral_columns!(columns, ::Nothing, model) = nothing

@kernel function _extend_spectral_columns!(columns, extension, grid, g, Rᵈ, Mᵈ, Mᵛ)
    i, j = @index(Global, NTuple)

    c = column_index(i, j, grid.Nx)
    Nₑ = number_of_extension_layers(extension)
    h = extension.blending_height
    base = extension.base

    @inbounds begin
        # The grid's top face is extension interface m = 1, column interface Nₑ + 1
        T_top = columns.temperature_interfaces[c, Nₑ+1]
        p_below = columns.pressure_interfaces[c, Nₑ+1]
        anchor = T_top - extension.join_temperature

        for m in 1:Nₑ
            kᶜ = Nₑ + 1 - m  # column layer of extension layer m; also the interface above it

            Δz = extension.Δz[m]
            z = extension.z_layer[m]
            zᶠ = z + Δz / 2

            # Anchor decaying away from the grid top; `h = 0` disables it
            w = ifelse(h > 0, exp(-(z - base) / h), zero(h))
            wᶠ = ifelse(h > 0, exp(-(zᶠ - base) / h), zero(h))
            T = extension.temperature_layers[m] + anchor * w
            Tᶠ = extension.temperature_interfaces[m+1] + anchor * wᶠ

            # Specific humidity q → mole fraction relative to dry air χ and virtual temperature
            q = extension.specific_humidity[m]
            χ = q / (1 - q) * Mᵈ / Mᵛ
            Tᵥ = T * (1 + (Mᵈ / Mᵛ - 1) * q)

            p_above = p_below * exp(-g * Δz / (Rᵈ * Tᵥ))
            Δp = p_below - p_above
            p_layer = Δp / log(p_below / p_above)
            n_dry = Δp / (g * (Mᵈ + Mᵛ * χ))

            columns.pressure_layers[c, kᶜ] = p_layer
            columns.temperature_layers[c, kᶜ] = T
            columns.dry_air[c, kᶜ] = n_dry
            columns.water_vapor[c, kᶜ] = χ * n_dry
            columns.ozone[c, kᶜ] = extension.ozone[m] * n_dry
            columns.liquid_water_path[c, kᶜ] = 0
            columns.ice_water_path[c, kᶜ] = 0

            columns.pressure_interfaces[c, kᶜ] = p_above
            columns.temperature_interfaces[c, kᶜ] = Tᶠ

            p_below = p_above
        end

        # The top of the atmosphere radiates at the temperature of the top layer
        columns.temperature_interfaces[c, 1] = columns.temperature_layers[c, 1]
    end
end

"""
$(TYPEDSIGNATURES)

Copy the solved column fluxes onto the four `ZFaceField`s of `rtm` (kernel D): grid face `k` is
column interface `N + 2 - k`, and the downwelling fluxes change sign to Breeze's positive-upward
convention.
"""
function copy_spectral_fluxes!(rtm, grid)
    arch = architecture(grid)
    columns = rtm.atmospheric_state
    Nx, Ny, Nz = size(grid)

    launch!(arch, grid, (Nx, Ny, Nz+1), _copy_spectral_fluxes!,
            rtm.upwelling_longwave_flux, rtm.downwelling_longwave_flux,
            rtm.upwelling_shortwave_flux, rtm.downwelling_shortwave_flux,
            columns, grid)

    return nothing
end

@kernel function _copy_spectral_fluxes!(ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn, columns, grid)
    i, j, k = @index(Global, NTuple)

    N = number_of_layers(columns)
    c = column_index(i, j, grid.Nx)
    kᶠ = N + 2 - k

    @inbounds begin
        ℐ_lw_up[i, j, k] = columns.flux_up_lw[c, kᶠ]
        ℐ_lw_dn[i, j, k] = -columns.flux_down_lw[c, kᶠ]
        ℐ_sw_up[i, j, k] = columns.flux_up_sw[c, kᶠ]
        ℐ_sw_dn[i, j, k] = -columns.flux_down_sw[c, kᶠ]
    end
end
