include("reference_conventions.jl")
using Statistics
using Test
using CairoMakie

root = normpath(joinpath(@__DIR__, "../../validation_output/p3_processes"))
dt = parse(Float64, get(ARGS, 1, "0.1"))
liquid_ice = get(ARGS, 2, "reference") == "liquid_ice"
name = dt == 1 ? "trajectories" : "trajectories_dt$(dt)"
directory = joinpath(root, liquid_ice ? name * "_liquid_ice" : name)
cases = readdlm(joinpath(directory, "cases.dat"))
reference = readdlm(joinpath(directory, "reference_final.dat"))
breeze = similar(reference)
function liquid_ice_state(values, pressure, constants)
    T, rv, rc, nc, rr, nr, ri, ni, rf, bf, rli, ssat = values
    dry_fraction = 1 / (1 + rv + rc + rr + ri + rli)
    q = TD.MoistureMassFractions(rv * dry_fraction, (rc + rr + rli) * dry_fraction, ri * dry_fraction)
    state = TD.LiquidIcePotentialTemperatureState(0., q, 1e5, pressure)
    return TD.with_temperature(state, T, constants)
end

function liquid_ice_temperature(values, pressure, constants, theta_li)
    state = liquid_ice_state(values, pressure, constants)
    state = TD.LiquidIcePotentialTemperatureState(theta_li, state.moisture_mass_fractions, 1e5, pressure)
    return TD.temperature(state, constants)
end

# P3 reference rates use dry mixing ratios. Convert those to moist mass
# fractions only at the actual Breeze liquid–ice thermodynamic-state boundary.
struct DryBasisLiquidIceState{FT} <: TD.AbstractReferencePressureState{FT}
    state :: TD.LiquidIcePotentialTemperatureState{FT}
    moisture_mass_fractions :: TD.MoistureMassFractions{FT}
    reference_pressure :: FT
end
TD.temperature(state::DryBasisLiquidIceState, constants) = TD.temperature(state.state, constants)
function TD.with_moisture(state::DryBasisLiquidIceState, r)
    dry_fraction = 1 / (1 + r.vapor + r.liquid + r.ice)
    q = TD.MoistureMassFractions(r.vapor * dry_fraction, r.liquid * dry_fraction, r.ice * dry_fraction)
    return DryBasisLiquidIceState(TD.with_moisture(state.state, q), r, state.reference_pressure)
end
TD.with_temperature(state::DryBasisLiquidIceState, T, constants) =
    DryBasisLiquidIceState(TD.with_temperature(state.state, T, constants), state.moisture_mass_fractions, state.reference_pressure)

function run_trajectories!(breeze, cases, dt; liquid_ice=false)
    row = 0
    for test in axes(cases, 1)
        T, P, RH, qc, nc, qr, nr, qi, ni, qf, ρf, qli = cases[test, :]
        p3, constants = reference_microphysics(P; dt)
        qsl = TD.saturation_specific_humidity(T, P/(287.15*T), constants, TD.PlanarLiquidSurface())
        qv = test >= 13 ? 0.015 / (1 - 0.015) : RH * qsl
        values = [T, qv, qc, nc, qr, nr, qi, ni, qf, qf/ρf, qli, qv-qsl]
        theta_li = liquid_ice_state(values, P, constants).potential_temperature
        steps = round(Int, (test >= 13 ? 5400 : 600) / dt)
        for step in 1:steps
            if test >= 13
                pressure = 101325 * (1 - 9.816 * (test - 12) * step * dt / (1005 * 300))^(1005/287.15)
                values[1] = liquid_ice ? liquid_ice_temperature(values, pressure, constants, theta_li) :
                                        values[1] * (pressure/P)^(287.15/1005)
                P = pressure
                constants = reference_constants(P)
                values[12] = values[2] - TD.saturation_specific_humidity(values[1], P/(287.15*values[1]), constants, TD.PlanarLiquidSurface())
            end
            thermodynamic_state = if liquid_ice
                r = TD.MoistureMassFractions(values[2], values[3] + values[5] + values[11], values[7])
                DryBasisLiquidIceState(liquid_ice_state(values, P, constants), r, P)
            else
                nothing
            end
            values = reference_convention_step(values, p3, constants, P, dt; thermodynamic_state)
            if liquid_ice
                # Replace the microphysics temperature update with the conserved host θ_li.
                values[1] = liquid_ice_temperature(values, P, constants, theta_li)
                values[12] = values[2] - TD.saturation_specific_humidity(values[1], 1., constants, TD.PlanarLiquidSurface())
            end
            if step % round(Int, 10/dt) == 0
                row += 1
                breeze[row, :] .= [test; step*dt; P; values]
            end
        end
    end
    return breeze
end
run_trajectories!(breeze, cases, dt; liquid_ice)
writedlm(joinpath(directory, "breeze.dat"), breeze)
labels = ("Temperature", "Vapor", "Cloud liquid", "Cloud number", "Rain", "Rain number",
          "Frozen ice", "Ice number", "Rime", "Rime volume", "Liquid on ice", "Supersaturation")
open(joinpath(directory, "metrics.txt"), "w") do io
    for test in axes(cases, 1)
        rows = reference[:, 1] .== test
        @printf(io, "Case %d: max temperature error %.6g K\n", test,
                maximum(abs.(breeze[rows, 4] - reference[rows, 4])))
        for column in 5:15
            a, b = breeze[rows, column], reference[rows, column]
            scale = maximum(abs.(b))
            error = maximum(abs.(a-b))
            @printf(io, "  %-18s max_abs=%10.4g normalized=%10.4g L1=%10.4g\n",
                    labels[column-3], error, error/max(scale,1e-30), sum(abs.(a-b))/max(sum(abs.(b)),1e-30))
        end
    end
end
print(read(joinpath(directory, "metrics.txt"), String))
set_theme!(Theme(fontsize=20, linewidth=3))
for test in (13, 14)
    figure = Figure(size=(1500, 1100))
    rows = reference[:, 1] .== test
    times = reference[rows, 2] / 60
    second_panel = test == 13 ? (2, 5, 1e3, "Vapor (g/kg dry air)") :
                               (2, 11, 1., "Ice number (kg⁻¹ dry air)")
    panels = ((1, 4, 1., "Temperature (K)"), second_panel,
              (3, 6, 1e3, "Cloud liquid (g/kg dry air)"), (4, 8, 1e3, "Rain (g/kg dry air)"),
              (5, 10, 1e3, "Frozen ice (g/kg dry air)"), (6, 15, 1e3, "Supersaturation (g/kg dry air)"))
    for (panel, column, scale, ylabel) in panels
        axis = Axis(figure[1+(panel-1)÷2,1+(panel-1)%2]; xlabel="Time (min)", ylabel)
        lines!(axis, times, reference[rows, column]*scale; label="Upstream P3", color=:black)
        lines!(axis, times, breeze[rows, column]*scale; label="Breeze P3 · matched driver", color=:dodgerblue, linestyle=:dash)
        panel == 1 && axislegend(axis; position=:rt)
    end
    speed = test - 12
    Label(figure[0, :], "Direct P3 comparison · $(liquid_ice ? "θ_li host" : "reference host") · $speed m/s ascent · 90 minutes · Δt=$dt s"; fontsize=25)
    name = test == 13 ? "ascent_comparison" : "cold_ascent_comparison"
    save(joinpath(directory, "$name.svg"), figure)
    save(joinpath(directory, "$name.png"), figure; px_per_unit=1.5)
end

@testset "Direct P3 trajectory agreement" begin
    @test all(isfinite, breeze)
    @test breeze[:, 1:3] ≈ reference[:, 1:3]
    @test minimum(breeze[:, 5:14]) >= -1e-12
    if liquid_ice
        for test in axes(cases, 1), data in (breeze, reference)
            rows = findall(data[:, 1] .== test)
            theta = [liquid_ice_state(data[row, 4:15], data[row, 3], reference_constants(data[row, 3])).potential_temperature for row in rows]
            @test maximum(theta) - minimum(theta) < 1e-8
        end
    end
    for test in axes(cases, 1)
        rows = reference[:, 1] .== test
        @test maximum(abs.(breeze[rows, 4] - reference[rows, 4])) < 0.001
        for column in 5:15
            a, b = breeze[rows, column], reference[rows, column]
            @test maximum(abs.(a-b)) <= 0.005 * maximum(abs.(b)) + 1e-10
        end
        # Closed parcels conserve all water; rime is already included in ice.
        T, P, RH, qc, nc, qr, nr, qi, ni, qf, rhof, qli = cases[test, :]
        constants = reference_constants(P)
        qv = test >= 13 ? 0.015 / 0.985 : RH * TD.saturation_specific_humidity(T, 1., constants, TD.PlanarLiquidSurface())
        total = qv + qc + qr + qi + qli
        for values in (breeze, reference)
            water = vec(sum(values[rows, [5, 6, 8, 10, 14]]; dims=2))
            @test maximum(abs.(water .- total)) < 1e-10
        end
    end
end
