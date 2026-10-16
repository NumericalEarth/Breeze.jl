# Reference-convention adapter for testing the P3 equations themselves. This does
# not change Breeze's production thermodynamics: it supplies upstream's dry-air
# saturation ratio, constant latent heats and dry cp at the microphysics boundary.
using Breeze
using Breeze.Thermodynamics: Thermodynamics as TD
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticleProperties as PPP
using DelimitedFiles
using Printf

struct ReferenceSaturation{FT, F}
    pressure :: FT
    fit :: F
end
const ReferenceConstants{FT, C, I, S<:ReferenceSaturation} = TD.ThermodynamicConstants{FT, C, I, S}
const reference_Lv = 3.1484e6 - 2370 * 273.15
const reference_Ls = reference_Lv + 0.3337e6
TD.liquid_latent_heat(T, c::ReferenceConstants) = reference_Lv
TD.ice_latent_heat(T, c::ReferenceConstants) = reference_Ls
function TD.saturation_vapor_pressure(T, c::ReferenceConstants, surface)
    fit = c.saturation_vapor_pressure.fit
    coefficients = surface isa TD.PlanarIceSurface && T < 273.15 ? fit.ice_coefficients : fit.liquid_coefficients
    return evalpoly(T - 273.15, coefficients)
end
function TD.saturation_specific_humidity(T, ρ, c::ReferenceConstants, surface)
    e = TD.saturation_vapor_pressure(T, c, surface)
    return 0.622 * e / (c.saturation_vapor_pressure.pressure - e)
end
# The limiter and melting balance also need the dry-air convention. Matching
# only the ordinary saturation diagnosis would leave a hidden moist/dry mismatch.
TD.adjustment_saturation_specific_humidity(T, P, qt, c::ReferenceConstants,
    surface::Union{TD.PlanarLiquidSurface, TD.PlanarIceSurface}) =
    TD.saturation_specific_humidity(T, 1., c, surface)
PPP.freezing_point_saturation_mass_fraction(c::ReferenceConstants, T0, rho) =
    TD.saturation_specific_humidity(T0, rho, c, TD.PlanarLiquidSurface())

struct ReferenceProcessState{FT} <: TD.AbstractReferencePressureState{FT}
    T :: FT
    moisture_mass_fractions :: TD.MoistureMassFractions{FT}
    reference_pressure :: FT
end
TD.temperature(s::ReferenceProcessState, constants) = s.T
function TD.with_moisture(s::ReferenceProcessState, q)
    ΔT = ((q.liquid - s.moisture_mass_fractions.liquid) * reference_Lv +
          (q.ice - s.moisture_mass_fractions.ice) * reference_Ls) / 1005
    return ReferenceProcessState(s.T + ΔT, q, s.reference_pressure)
end
TD.with_temperature(s::ReferenceProcessState, T, constants) = ReferenceProcessState(T, s.moisture_mass_fractions, s.reference_pressure)


function reference_constants(P)
    constants = ThermodynamicConstants(;
        dry_air_molar_mass=8.314462618/287.15, vapor_molar_mass=8.314462618/461.51,
        gravitational_acceleration=9.816,
        liquid=TD.CondensedPhase(reference_latent_heat=reference_Lv, heat_capacity=4218, density=1000),
        saturation_vapor_pressure=ReferenceSaturation(P, TD.FlatauPolynomial(reference_temperature=273.15)))
    return constants
end

function reference_microphysics(P; dt=1.)
    constants = reference_constants(P)
    process_rates = PPP.ProcessRate(; thermodynamic_constants=constants, predict_supersaturation=true,
        sink_limiting_timescale=dt, ice_nucleation_timescale=dt, homogeneous_freezing_timescale=dt,
        refreezing_timescale=dt, rime_densification_timescale=dt, rain_evaporation_timescale=dt,
        shed_drop_mass_liqfrac=1/1.928e6)
    aerosol = PPP.AerosolActivation(PPP.AerosolMode(molecular_weight_water=0.018);
        prognostic=false, activation_timescale=dt, molecular_weight_water=0.018, universal_gas_constant=8.3145)
    p3 = P3Microphysics(; thermodynamic_constants=constants, process_rates, aerosol)
    return p3, constants
end

# The upstream routine includes a hard PSD/trace cleanup after Euler stepping.
# Breeze normally relaxes those constraints through its RHS. For comparing the
# process operator independently of time-integration policy, apply the same
# end-of-step projection using Breeze's existing diagnoses (no fitted factors).
function reference_projection!(values, p3, ρ, constants)
    T, qv, qc, nc, qr, nr, qi, ni, qf, bf, qli, ssat = values
    floor = p3.minimum_mass_mixing_ratio
    if qc < floor
        qv += qc; T -= qc * reference_Lv / 1005; qc = 0.; nc = 0.
    else
        nc = PPP.diagnose_cloud_dsd(p3, qc, nc, ρ).nᶜˡ
    end
    if qr < floor
        qv += qr; T -= qr * reference_Lv / 1005; qr = 0.; nr = 0.
    else
        nr = PPP.bounded_rain_number(nr, qr, p3.process_rates)
    end
    if qi + qli < floor
        qv += qi + qli; T -= (qi * reference_Ls + qli * reference_Lv) / 1005
        qi = 0.; ni = 0.; qf = 0.; bf = 0.; qli = 0.
    else
        micro = PPP.P3MicrophysicalState(qc, nc, qr, nr, qi, ni, qf, bf, qli, ssat, 3e8, 0.)
        properties = PPP.p3_process_properties(p3, ρ, micro)
        ni = properties.nⁱ; qf = properties.qᶠ; bf = properties.bᶠ
        qli = qli < floor ? 0. : qli
    end
    ssat = qv - TD.saturation_specific_humidity(T, ρ, constants, TD.PlanarLiquidSurface())
    values .= (T, qv, qc, nc, qr, nr, qi, ni, qf, bf, qli, ssat)
    return values
end

function reference_convention_step(values, p3, constants, P, dt; project=true, thermodynamic_state=nothing)
    T, qv, qc, nc, qr, nr, qi, ni, qf, bf, qli, ssat = values
    ρ = P / (287.15 * T)
    q = TD.MoistureMassFractions(qv, qc + qr + qli, qi)
    state = isnothing(thermodynamic_state) ? ReferenceProcessState(T, q, P) : thermodynamic_state
    micro = PPP.P3MicrophysicalState(qc, nc, qr, nr, qi, ni, qf, bf, qli, ssat, 3e8, 0.)
    result = PPP.p3_state_tendencies(p3, ρ, micro, state, constants)
    updated = copy(values)
    names = (:ρqᵛ, :ρqᶜˡ, :ρnᶜˡ, :ρqʳ, :ρnʳ, :ρqⁱ, :ρnⁱ, :ρqᶠ, :ρbᶠ, :ρqʷⁱ, :ρsᵛ⁺ˡ)
    for (index, name) in enumerate(names)
        updated[index+1] += dt * getproperty(result, Symbol(:tendency_, name)) / ρ
    end
    q_after = TD.MoistureMassFractions(updated[2], updated[3]+updated[5]+updated[11], updated[7])
    updated[1] = TD.temperature(TD.with_moisture(state, q_after), constants)
    project && reference_projection!(updated, p3, ρ, constants)
    return updated
end
