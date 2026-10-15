# Read-only evaluation of trajectories advanced by ParcelModel's native RK3.
using DelimitedFiles
using Printf
using CairoMakie
using Test
root = normpath(joinpath(@__DIR__, "../../validation_output"))
parcel_directory = joinpath(root, "p3_parcel_direct", get(ARGS, 1, "verified_reference_constants_dt0.0025_tau0.025"))
reference_dt = get(ARGS, 2, "0.0015625")
reference_directory = joinpath(root, "p3_processes", "trajectories_dt$(reference_dt)_liquid_ice")
reference = readdlm(joinpath(reference_directory, "reference_final.dat"))
labels = ("Vapor", "Cloud liquid", "Cloud number", "Rain", "Rain number", "Frozen ice", "Ice number", "Rime", "Rime volume", "Liquid on ice", "Supersaturation")
function moist_reference(reference, speed)
    ref = copy(reference[reference[:,1] .== speed+12, :])
    dry_fraction = 1 ./ (1 .+ vec(sum(ref[:,[5,6,8,10,14]]; dims=2)))
    saturation_ratio = ref[:,5] - ref[:,15]
    ref[:,15] .*= 0.622 ./ (0.622 .+ saturation_ratio)
    ref[:,5:15] .*= dry_fraction
    return ref
end
set_theme!(fontsize=20, linewidth=3)
open(joinpath(parcel_directory, "evaluation_reference_dt$(reference_dt).txt"), "w") do io
    println(io, "Actual ParcelModel, native SSP RK3, no prognostic resets or comparison adapter.")
    println(io, "Upstream timestep: $reference_dt s. Both use a liquid–ice potential-temperature host.")
    for speed in (1,2)
        data = readdlm(joinpath(parcel_directory, "parcel_$(speed).dat"))
        data = data[data[:,2] .> 0, :]
        ref = moist_reference(reference, speed)
        @testset "ParcelModel physical checks: $speed m/s" begin
            @test all(isfinite, data)
            @test maximum(abs.(data[:,2]-ref[:,2])) < 1e-7
            @test maximum(abs.(data[:,3]-ref[:,3])) < 0.01
            @test maximum(abs.(data[:,16] .- data[1,16])) < 1e-8
            @test maximum(abs.(data[:,17] .- .015)) < 1e-12
            @test minimum(data[:,5:14]) >= -1e-12
            @test maximum(abs.(data[:,15]-data[:,20])) < 1e-7
        end
        @printf(io, "\n%d m/s: max temperature error %.8g K; RMS %.8g K\n", speed,
                maximum(abs.(data[:,4]-ref[:,4])), sqrt(sum(abs2,data[:,4]-ref[:,4])/size(data,1)))
        for col in 5:15
            a,b = data[:,col], ref[:,col]
            @printf(io, "%-16s max/peak=%10.6f%% L1=%10.6f%% final=%12.6g reference=%12.6g\n", labels[col-4],
                    100maximum(abs.(a-b))/max(maximum(abs.(b)),1e-30),
                    100sum(abs.(a-b))/max(sum(abs.(b)),1e-30), a[end],b[end])
        end
        @printf(io, "Maximum prognostic/diagnosed supersaturation difference: %.8g kg/kg\n", maximum(abs.(data[:,15]-data[:,20])))
        fig = Figure(size=(1500,1400))
        Label(fig[0,:], "ParcelModel RK3 vs upstream P3 · θ_li · $speed m/s · 90 minutes"; fontsize=26)
        panels=((4,"Temperature (K)"),(6,"Cloud liquid (kg/kg)"),(8,"Rain (kg/kg)"),(10,"Frozen ice (kg/kg)"),
                (7,"Cloud number (kg⁻¹)"),(11,"Ice number (kg⁻¹)"),(15,"Supersaturation (kg/kg)"))
        for (index,(col,label)) in enumerate(panels)
            ax=Axis(fig[1+(index-1)÷2,1+(index-1)%2]; xlabel="Time (min)", ylabel=label)
            lines!(ax,ref[:,2]/60,ref[:,col];color=:black,label="Upstream P3")
            lines!(ax,data[:,2]/60,data[:,col];color=:dodgerblue,linestyle=:dash,label="Breeze ParcelModel · RK3")
            index==1 && axislegend(ax;position=:rt)
        end
        ax=Axis(fig[4,2];xlabel="Time (min)",ylabel="Temperature difference (K)")
        lines!(ax,data[:,2]/60,data[:,4]-ref[:,4];color=:magenta)
        hlines!(ax,[0];color=:gray,linestyle=:dash)
        save(joinpath(parcel_directory,"parcel_upstream_$(speed).png"),fig;px_per_unit=1.5)
        save(joinpath(parcel_directory,"parcel_upstream_$(speed).svg"),fig)
    end
end
print(read(joinpath(parcel_directory, "evaluation_reference_dt$(reference_dt).txt"),String))
