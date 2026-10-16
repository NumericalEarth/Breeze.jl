# Read-only diagnosis and plots. Intervened upstream runs are explicitly labeled
# sensitivity experiments; the unmodified reference remains the baseline.
include("parcel_compare.jl")
using CairoMakie
using Test

root = normpath(joinpath(@__DIR__, "../../validation_output/p3_matrix"))
directory = joinpath(root,"outliers")
cases = readdlm(joinpath(root,"cases.dat"))
identifiers = (3,9,18)
variants = ("baseline","nucleation_relaxation","retain_trace_ice","retain_trace_rime",
            "retain_trace_ice_and_rime","consistent_saturation")

function moist_basis(data)
    data = copy(data)
    data[:,5:14] ./= 1 .+ sum(data[:,[5,6,8,10,14]];dims=2)
    return data
end

breeze = Dict(id=>readdlm(joinpath(directory,"breeze_matched_constants/case_$id.dat")) for id in identifiers)
reference = Dict(name=>readdlm(joinpath(directory,name,"reference_final.dat")) for name in variants)
rates = Dict(name=>readdlm(joinpath(directory,name,"reference_rates.dat")) for name in variants)

@testset "Outlier trajectory integrity" begin
    @test reference["baseline"] == readdlm(joinpath(root,"reference_dt0.00078125/reference_final.dat"))
    for id in identifiers
        data = breeze[id]
        @test size(data) == (54001,20)
        @test all(isfinite,data)
        @test maximum(abs.(data[:,16].-data[1,16])) < 1e-8
        @test maximum(abs.(sum(data[:,[5,6,8,10,14]];dims=2).-cases[id,5])) < 1e-10
        @test minimum(data[:,5:14]) >= -1e-12
        @test maximum(abs.(data[:,15]-data[:,20])) < 1e-7
    end
    for name in variants
        data = reference[name]
        @test size(data) == (1620,15)
        @test all(isfinite,data)
        @test minimum(data[:,5:14]) >= -1e-12
        for id in identifiers
            state = moist_basis(data[data[:,1].==id,:])
            @test maximum(abs.(sum(state[:,[5,6,8,10,14]];dims=2).-cases[id,5])) < 1e-10
        end
    end
end

open(joinpath(directory,"diagnosis.txt"),"w") do io
    println(io,"All sensitivities change ONLY the upstream diagnostic copy; they are not official reference runs.")
    println(io,"Breeze: production ParcelModel, dt=0.0025, tau=0.025, aerosol R=8.3145, saved every 0.1 s.")
    println(io,"Upstream: dt=0.00078125, full states every 10 s; process-entry diagnostics every 0.1 s around events.")
    println(io,"Case  variant  max_T_difference_at_10s_K  max_Ni_difference_percent_upstream_peak  final_Ni_Breeze  final_Ni_upstream")
    for id in identifiers
        a = breeze[id][101:100:end,:]
        original = readdlm(joinpath(root,"breeze_dt0.0025/case_$(lpad(id,3,'0')).dat"))[2:end,:]
        @printf(io,"Aerosol-constant correction, case %d: max T change %.8g K; max Ni change %.6g /kg\n",
                id,maximum(abs.(a[:,4]-original[:,4])),maximum(abs.(a[:,11]-original[:,11])))
        for name in variants
            b = moist_basis(reference[name][reference[name][:,1].==id,:])
            @assert maximum(abs.(a[:,2]-b[:,2])) < 1e-6
            @printf(io,"%d %-26s %.9g %.9g %.9g %.9g\n",id,name,
                maximum(abs.(a[:,4]-b[:,4])),100maximum(abs.(a[:,11]-b[:,11]))/maximum(b[:,11]),
                a[end,11],b[end,11])
        end
    end
    for id in (3,18), name in variants
        b = rates[name][rates[name][:,1].==id,:]
        a = breeze[id][round.(Int,b[:,2]*10).+1,:]
        difference = abs.(a[:,4]-b[:,3])
        peak = argmax(difference)
        @printf(io,"0.1 s samples, case %d %-26s max T difference %.9g K at %.3f s\n",id,name,difference[peak],b[peak,2])
        for threshold in (1e-8,1e-4,1e-3)
            j = findfirst(b[:,8]*(1-cases[id,5]).>=threshold)
            k = findfirst(a[:,10]+a[:,14].>=threshold)
            if !isnothing(j) && !isnothing(k)
                @printf(io,"  ice+coating %.1e: Breeze %.3f s, upstream %.3f s\n",threshold,a[k,2],b[j,2])
            end
        end
    end
end
print(read(joinpath(directory,"diagnosis.txt"),String))

set_theme!(fontsize=21,linewidth=3)
fig = Figure(size=(1650,1150))
Label(fig[0,1:3],"P3 outliers · controlled diagnosis";fontsize=28)
colors = (:dodgerblue,:black,:darkorange)
styles = (:dash,:solid,:dot)

ax = Axis(fig[1,1];title="288 K · 60% RH · 2 m/s",xlabel="Time (min)",ylabel="Temperature (K)")
ax_rime = Axis(fig[2,1];xlabel="Time (min)",ylabel="Rime mass (kg/kg)",yscale=log10)
for (method,color,style,label) in zip(("breeze","baseline","retain_trace_ice_and_rime"),colors,styles,
    ("Breeze","Upstream","Upstream: retain trace ice/rime"))
    if method == "breeze"
        a = breeze[3]
        a = a[(1400 .<= a[:,2]).&(a[:,2].<=1580),:]
        lines!(ax,a[:,2]/60,a[:,4];color,linestyle=style,label)
        lines!(ax_rime,a[:,2]/60,max.(a[:,12],1e-18);color,linestyle=style)
    else
        a = rates[method]
        a = a[(a[:,1].==3).&(1400 .<= a[:,2]).&(a[:,2].<=1580),:]
        lines!(ax,a[:,2]/60,a[:,3];color,linestyle=style,label)
        b = moist_basis(reference[method][reference[method][:,1].==3,:])
        b = b[(1400 .<= b[:,2]).&(b[:,2].<=1580),:]
        lines!(ax_rime,b[:,2]/60,max.(b[:,12],1e-18);color,linestyle=style)
    end
end
Legend(fig[3,1],ax;labelsize=18,tellwidth=false,tellheight=true)

ax = Axis(fig[1,2];title="288 K · 100% RH · 2 m/s",xlabel="Time (min)",ylabel="Ice number (10³ kg⁻¹)")
ax_saturation = Axis(fig[2,2];xlabel="Time (min)",ylabel="Ice supersaturation (%)")
constants = parcel_thermodynamic_constants("reference_constants")
for (method,color,style,label) in zip(("breeze","baseline","nucleation_relaxation"),colors,styles,
    ("Breeze","Upstream","Upstream: nucleation τ=0.025 s"))
    a = method == "breeze" ? breeze[9] : moist_basis(reference[method][reference[method][:,1].==9,:])
    a = a[a[:,2].>=4500,:]
    lines!(ax,a[:,2]/60,a[:,11]/1000;color,linestyle=style,label)
    if method == "breeze"
        a = a[(4700 .<= a[:,2]).&(a[:,2].<=4800),:]
        supersaturation = map(eachrow(a)) do row
            q = TD.MoistureMassFractions(row[5],row[6]+row[8]+row[14],row[10])
            rho = TD.density(row[4],row[3],q,constants)
            qsi = TD.saturation_specific_humidity(row[4],rho,constants,TD.PlanarIceSurface())
            100(row[5]/qsi-1)
        end
        lines!(ax_saturation,a[:,2]/60,supersaturation;color,linestyle=style)
    else
        a = rates[method][rates[method][:,1].==9,:]
        lines!(ax_saturation,a[:,2]/60,100a[:,10];color,linestyle=style)
    end
end
hlines!(ax_saturation,[5.];color=:gray,linestyle=:dash,linewidth=1)
Legend(fig[3,2],ax;labelsize=18,tellwidth=false,tellheight=true)

ax = Axis(fig[1,3];title="300 K · 100% RH · 2 m/s",xlabel="Time − 3579 s (s)",ylabel="Temperature (K)")
ax_ice = Axis(fig[2,3];xlabel="Time − 3579 s (s)",ylabel="Frozen + coating mass (g/kg)")
for (method,color,style,label) in zip(("breeze","baseline","consistent_saturation"),colors,styles,
    ("Breeze","Upstream","Upstream: saturation ε=Rd/Rv"))
    if method == "breeze"
        a = breeze[18]
        a = a[(3578 .<= a[:,2]).&(a[:,2].<=3583),:]
        lines!(ax,a[:,2].-3579,a[:,4];color,linestyle=style,label)
        lines!(ax_ice,a[:,2].-3579,1000(a[:,10]+a[:,14]);color,linestyle=style)
    else
        a = rates[method]
        a = a[(a[:,1].==18).&(3578 .<= a[:,2]).&(a[:,2].<=3583),:]
        lines!(ax,a[:,2].-3579,a[:,3];color,linestyle=style,label)
        # Process-entry diagnostic records total ice including coating.
        lines!(ax_ice,a[:,2].-3579,1000a[:,8]*(1-cases[18,5]);color,linestyle=style)
    end
end
Legend(fig[3,3],ax;labelsize=18,tellwidth=false,tellheight=true)
Label(fig[4,1:3],"Orange: isolated changes to upstream for diagnosis, not replacement reference data.\nUpstream close-ups use process-entry states; their time staggering is less than 0.001 s.";fontsize=20)
save(joinpath(directory,"outlier_diagnosis.png"),fig;px_per_unit=1.5)
save(joinpath(directory,"outlier_diagnosis.svg"),fig)
