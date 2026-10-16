# Identical-state, single-call process probes. The reference-convention adapter
# is used ONLY to compare formulas here; it never advances a parcel trajectory.
include("reference_conventions.jl")
using Test

root = normpath(joinpath(@__DIR__, "../../validation_output"))
directory = joinpath(root,"p3_matrix/outliers/process_probes")
mkpath(directory)
reference = readdlm(joinpath(root,"p3_matrix/reference_dt0.00078125/reference_final.dat"))
rows = Vector{Float64}[]
labels = String[]
for (identifier,times) in ((3,(1300,1400,1500,1530,1550)), (9,(4720,4730)), (18,(3570,3580)))
    for time in times
        row = only(eachrow(reference[(reference[:,1].==identifier).&(reference[:,2].==time),:]))
        T,P = row[4],row[3]
        constants = reference_constants(P)
        qsl = TD.saturation_specific_humidity(T,1.,constants,TD.PlanarLiquidSurface())
        rime_density = row[13] > 0 ? row[12]/row[13] : 400.
        push!(rows,[T,P,row[5]/qsl,row[6:12]...,rime_density,row[14]])
        push!(labels,"case=$identifier time=$time")
    end
end
# Probe just above the Cooper threshold with the pre-event case-9 population.
row = copy(rows[6])
c = reference_constants(row[2])
row[3] = 1.05001 * TD.saturation_specific_humidity(row[1],1.,c,TD.PlanarIceSurface()) /
                  TD.saturation_specific_humidity(row[1],1.,c,TD.PlanarLiquidSurface())
push!(rows,row)
push!(labels,"case=9 pre-event state with S_ice=0.05001")
# Probe the initial rain-freezing threshold without advancing any state.
row = copy(rows[8]); row[1] = 269.149
push!(rows,row)
push!(labels,"case=18 pre-event condensate with T=269.149 K")

dt = 0.00078125
writedlm(joinpath(directory,"cases.dat"),permutedims(hcat(rows...)))
write(joinpath(directory,"labels.txt"),join(labels,'\n')*"\n")
source_directory = joinpath(root,"p3_processes")
cp(joinpath(source_directory,"microphy_p3.f90"),joinpath(directory,"microphy_p3.f90");force=true)
driver = replace(read(joinpath(source_directory,"process_probe.f90"),String),
                 "dt=1."=>"dt=$dt", "do test=1,12"=>"do test=1,$(length(rows))")
write(joinpath(directory,"process_probe.f90"),driver)
ispath(joinpath(directory,"lookup_tables")) || symlink(realpath(joinpath(source_directory,"lookup_tables")),joinpath(directory,"lookup_tables"))
cd(directory) do
    run(`gfortran -O2 -cpp -ffree-line-length-none -fdefault-real-8 -fdefault-double-8 -fcheck=bounds -fbacktrace microphy_p3.f90 process_probe.f90 -o process_probe`)
    run(`./process_probe`)
end
initial = readdlm(joinpath(directory,"initial.dat"))
upstream = readdlm(joinpath(directory,"reference_rates.dat"))
names = Symbol.(readlines(joinpath(source_directory,"process_names.txt")))
breeze = similar(upstream)
for index in axes(initial,1)
    T,P,rho,qv,qsl,qsi,qc,nc,qr,nr,qi,ni,qf,rhof,qli = initial[index,:]
    p3,constants = reference_microphysics(P;dt)
    state = ReferenceProcessState(T,TD.MoistureMassFractions(qv,qc+qr+qli,qi),P)
    micro = PPP.P3MicrophysicalState(qc,nc,qr,nr,qi,ni,qf,qf/rhof,qli,qv-qsl,3e8,0.)
    properties = PPP.p3_process_properties(p3,rho,micro)
    rates = PPP.compute_p3_process_rates(p3,rho,micro,state,constants,properties,T,0.,0.)
    for (column,name) in enumerate(names)
        breeze[index,column] = getproperty(rates,name) - (name == :rain_self_collection ? rates.rain_breakup : 0.)
    end
end
writedlm(joinpath(directory,"breeze_rates.dat"),breeze)
open(joinpath(directory,"comparison.txt"),"w") do io
    for index in axes(initial,1)
        println(io,"\n",labels[index])
        for (column,name) in enumerate(names)
            a,b = breeze[index,column],upstream[index,column]
            scale = max(abs(a),abs(b))
            scale > 1e-15 || continue
            @printf(io,"%-27s Breeze=%12.5e upstream=%12.5e relative=%9.3g%%\n",name,a,b,100abs(a-b)/scale)
        end
    end
end
print(read(joinpath(directory,"comparison.txt"),String))
@testset "Outlier probe validity" begin
    @test size(breeze) == size(upstream) == (length(rows),length(names))
    @test all(isfinite,breeze) && all(isfinite,upstream)
end
@testset "Outlier nucleation and freezing formulas" begin
    # Allow upstream's rounded mathematical constants; this tests the formulas
    # at identical states/timescales, not trajectory agreement.
    for name in (:cloud_freezing_mass,:cloud_freezing_number,:rain_freezing_mass,:rain_freezing_number)
        column = findfirst(==(name),names)
        for index in axes(initial,1)
            @test isapprox(breeze[index,column],upstream[index,column];rtol=1e-6,atol=1e-25)
        end
    end
    for name in (:nucleation_mass,:nucleation_number)
        column = findfirst(==(name),names)
        @test isapprox(breeze[10,column],upstream[10,column];rtol=1e-6)
    end
    for name in (:ccn_activation_mass,:ccn_activation_number)
        column = findfirst(==(name),names)
        @test isapprox(breeze[8,column],upstream[8,column];rtol=1e-6)
    end
end
