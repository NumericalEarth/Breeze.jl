#####
##### Halos of the water of moist HydrostaticDynamics across the seam of a Yin-Yang grid
#####
#
# A `YinYangGrid` fills the horizontal halos of a panel by Lagrange interpolation from the interior
# of the other panel, with `stencil_width × stencil_width` stencils (4 × 4 by default: cubic). The
# outer weights of a cubic stencil are negative, so next to a sharp edge of a field that is
# nowhere negative (the edge of a rain shaft, a jump of the humidity) the interpolated halo value
# can be negative. A transport scheme that keeps the owned cells within their bounds given
# neighbors within those bounds then carries the negative halo value into the owned cells of the
# seam: measured, with first-order upwind water, as negative rain that appears in the seam band
# and nowhere else, and that does not appear with 2 × 2 (bilinear, convex) stencils.
#
# The winds and the thermodynamics keep the interpolation of the grid. For the water only, the
# halos are filled again here, after every diagnosis of the state (every stage of a time step):
#
#   1. the mixing ratio per unit dry mass r = ρc / ρ, which is what the model transports, is
#      interpolated with the stencils and weights of the grid, and the result is limited to the
#      range of the mixing ratios of its own stencil. A value between the smallest and the largest
#      of its donors is not negative if they are not, and a uniform mixing ratio is reproduced
#      exactly, whatever the dry mass does;
#   2. the halo of the water density is that mixing ratio times the dry density of the halo cell,
#      ρc = ρ r, the halo of the total density is ρ + Σ ρc, and the halos of the mass fractions are
#      ρc / (ρ + Σ ρc), so that the mixing ratio that `MixingRatio` forms in a halo cell is r again.
#
# Only halo cells are written. No owned cell is changed, so nothing is clipped and no mass is
# moved: what the limiter changes is the value a panel sees of the other panel's water where the
# unlimited interpolant leaves the range of the data it interpolates. Where the interpolant stays
# within that range, which is everywhere the field is smooth on the scale of the stencil, the
# halo is the cubic one. The exchange through the seam of an overset grid is not conservative
# with either interpolation.
#
# TODO (Oceananigans): a shape-preserving option of the Yin-Yang halo interpolation, per field,
# belongs in `Oceananigans.MultiRegion` (`interpolate_donor`). This file reads the tables of the
# grid (`yin_yang_interpolation`, `interpolation_table`) and launches its own kernel over their
# receivers with `launch_yin_yang_interpolation!`.

using Oceananigans.MultiRegion: YinYangGrid, YinYangInterpolationTable, yin_yang_interpolation, interpolation_table,
                                launch_yin_yang_interpolation!
using Oceananigans.Utils: getregion

@inline safe_ratio(a, b) = ifelse(b == 0, zero(a), a / ifelse(b == 0, one(b), b))

# The Lagrange interpolant of r = ρc / ρ at receiver `n`, limited to the range of r over its stencil
@inline function limited_mixing_ratio_interpolation(ρc, ρ, table::YinYangInterpolationTable{S}, n, k) where S
    FT = eltype(ρc)
    @inbounds begin
        i₀, j₀ = table.donors.i[n], table.donors.j[n]
        wx, wy = table.weights.x, table.weights.y
        value = zero(FT)
        lowest = convert(FT, Inf)
        highest = convert(FT, -Inf)

        for b in 1:S
            row = zero(FT)
            for a in 1:S
                r = safe_ratio(ρc[i₀ + a - 1, j₀ + b - 1, k], ρ[i₀ + a - 1, j₀ + b - 1, k])
                row += wx[n, a] * r
                lowest = min(lowest, r)
                highest = max(highest, r)
            end
            value += wy[n, b] * row
        end
    end

    return clamp(value, lowest, highest)
end

@kernel function _fill_water_density_seam_halo!(ρc, ρ, donor_ρc, donor_ρ, table)
    n, k = @index(Global, NTuple)
    @inbounds begin
        i = table.receivers.i[n]
        j = table.receivers.j[n]
        ρc[i, j, k] = ρ[i, j, k] * limited_mixing_ratio_interpolation(donor_ρc, donor_ρ, table, n, k)
    end
end

@inline set_mass_fractions_at!(i, j, k, ::Tuple{}, ::Tuple{}, ρᵗ) = nothing

@inline function set_mass_fractions_at!(i, j, k, q::Tuple, ρq::Tuple, ρᵗ)
    @inbounds first(q)[i, j, k] = safe_ratio(first(ρq)[i, j, k], ρᵗ)
    return set_mass_fractions_at!(i, j, k, Base.tail(q), Base.tail(ρq), ρᵗ)
end

@kernel function _fill_total_density_seam_halo!(ρᵗ, q, ρ, ρq, table)
    n, k = @index(Global, NTuple)
    @inbounds begin
        i = table.receivers.i[n]
        j = table.receivers.j[n]
        ρᵗ[i, j, k] = ρ[i, j, k] + sum_at(i, j, k, ρq)
        set_mass_fractions_at!(i, j, k, q, ρq, ρᵗ[i, j, k])
    end
end

# The mass fractions that `MixingRatio` reads for the prognostic densities of water, in their order
function water_mass_fractions(model)
    specific = specific_prognostic_scalar_fields(model)
    names = (AtmosphereModels.moisture_prognostic_name(model.microphysics),
             AtmosphereModels.prognostic_field_names(model.microphysics)...)
    return map(name -> specific[name], names)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Fill the horizontal halos of the water of `model`, an `AtmosphereModel` with moist
`HydrostaticDynamics`, across the seam of a `YinYangGrid`: the densities of water from an
interpolation of the mixing ratio per unit dry mass that is limited to the range of its donors,
and the total density and the mass fractions consistently with them. On any other grid the halos
are those of the grid, and nothing is done. The halos of the dry density must be filled.
"""
fill_water_seam_halos!(model) = fill_water_seam_halos!(model, model.grid)
fill_water_seam_halos!(model, grid) = nothing

function fill_water_seam_halos!(model, grid::YinYangGrid)
    dynamics = model.dynamics
    ρ = dynamics.density
    ρᵗ = dynamics.moisture.total_density
    ρq = water_densities(model)
    q = water_mass_fractions(model)
    table = interpolation_table(yin_yang_interpolation(grid), Center(), Center())

    for receiver in 1:2
        donor = 3 - receiver
        panel = getregion(grid, receiver)
        data(field, r) = getregion(field.data, r)
        ρʳ = data(ρ, receiver)
        ks = axes(ρʳ, 3)

        for ρc in ρq
            launch_yin_yang_interpolation!(panel, ks, table, _fill_water_density_seam_halo!,
                                           data(ρc, receiver), ρʳ, data(ρc, donor), data(ρ, donor), table)
        end

        launch_yin_yang_interpolation!(panel, ks, table, _fill_total_density_seam_halo!,
                                       data(ρᵗ, receiver), map(f -> data(f, receiver), q), ρʳ,
                                       map(f -> data(f, receiver), ρq), table)
    end

    return nothing
end
