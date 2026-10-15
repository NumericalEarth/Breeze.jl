include("reference_conventions.jl")
using Test

directory = normpath(joinpath(@__DIR__, "../../validation_output/p3_processes"))
initial = readdlm(joinpath(directory, "initial.dat"))
reference_rates = readdlm(joinpath(directory, "reference_rates.dat"))
reference_final = readdlm(joinpath(directory, "reference_final.dat"))
names = Symbol.(readlines(joinpath(directory, "process_names.txt")))
@assert size(reference_rates) == (12, length(names))
breeze_rates = similar(reference_rates)
breeze_final = similar(reference_final)
projected_final = similar(reference_final)
for test in axes(initial, 1)
    T, P, ρ, qv, qsl, qsi, qc, nc, qr, nr, qi, ni, qf, ρf, qli = initial[test, :]
    p3, constants = reference_microphysics(P)
    @assert TD.saturation_specific_humidity(T, ρ, constants, TD.PlanarLiquidSurface()) ≈ qsl rtol=1e-11
    @assert TD.saturation_specific_humidity(T, ρ, constants, TD.PlanarIceSurface()) ≈ qsi rtol=1e-11
    q = TD.MoistureMassFractions(qv, qc + qr + qli, qi)
    state = ReferenceProcessState(T, q, P)
    micro = PPP.P3MicrophysicalState(qc, nc, qr, nr, qi, ni, qf, qf/ρf, qli, qv-qsl, 3e8, 0.)
    properties = PPP.p3_process_properties(p3, ρ, micro)
    rates = PPP.compute_p3_process_rates(p3, ρ, micro, state, constants, properties, T, 0., 0.)
    for (column, name) in enumerate(names)
        value = getproperty(rates, name)
        if name == :rain_self_collection
            value -= rates.rain_breakup
        end
        breeze_rates[test, column] = value
    end
    tendencies = PPP.p3_tendency_compute(p3, ρ, micro, state, constants, properties, T, 0., 0.)
    # Extract the same prognostic tendencies used by grid and parcel callers.
    updated = [qv, qc, nc, qr, nr, qi, ni, qf, qf/ρf, qli, qv-qsl]
    tendency_names = (:ρqᵛ, :ρqᶜˡ, :ρnᶜˡ, :ρqʳ, :ρnʳ, :ρqⁱ, :ρnⁱ, :ρqᶠ, :ρbᶠ, :ρqʷⁱ, :ρsᵛ⁺ˡ)
    for (index, name) in enumerate(tendency_names)
        updated[index] += getproperty(tendencies, Symbol(:tendency_, name)) / ρ
    end
    q_after = TD.MoistureMassFractions(updated[1], updated[2]+updated[4]+updated[10], updated[6])
    breeze_final[test, :] .= [TD.temperature(TD.with_moisture(state, q_after), constants); updated]
    projected_final[test, :] .= reference_projection!(copy(breeze_final[test, :]), p3, ρ, constants)
end
writedlm(joinpath(directory, "breeze_rates.dat"), breeze_rates)
writedlm(joinpath(directory, "breeze_final.dat"), breeze_final)
writedlm(joinpath(directory, "breeze_projected_final.dat"), projected_final)
open(joinpath(directory, "rate_comparison.txt"), "w") do io
    for test in axes(initial, 1)
        @printf(io, "\nCase %d T=%.2f K\n", test, initial[test, 1])
        for (column, name) in enumerate(names)
            a, b = breeze_rates[test, column], reference_rates[test, column]
            scale = max(abs(a), abs(b))
            if scale > 1e-16
                @printf(io, "%-28s Breeze=%12.5e reference=%12.5e relative=%9.3g%%\n", name, a, b, 100abs(a-b)/scale)
            end
        end
    end
end
print(read(joinpath(directory, "rate_comparison.txt"), String))

@testset "Direct upstream P3 rates and projected updates" begin
    @test all(isfinite, breeze_rates)
    for test in axes(reference_rates, 1), column in axes(reference_rates, 2)
        # Upstream retains a 1.51e-9 rain evaporation flux at exactly liquid
        # saturation; Breeze correctly zeros the complete evaporation budget.
        trace = test == 7 && names[column] == :rain_evaporation
        @test isapprox(breeze_rates[test, column], reference_rates[test, column];
                       rtol=1e-3, atol=trace ? 2e-9 : 1e-15)
    end
    @test maximum(abs.(projected_final[:, 1] - reference_final[:, 1])) < 2e-5
    for column in 2:12
        atol = column in (4, 6, 8) ? 0.05 : 1e-10
        @test all(isapprox.(projected_final[:, column], reference_final[:, column]; rtol=1e-3, atol))
    end
end
