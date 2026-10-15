# Read-only aggregation of the production ParcelModel / upstream P3 matrix.
using DelimitedFiles
using Statistics
using Printf
using CairoMakie
using Test

function upstream_moist_basis(data)
    data = copy(data)
    dry_fraction = 1 ./ (1 .+ vec(sum(data[:,[5,6,8,10,14]]; dims=2)))
    saturation_ratio = data[:,5] - data[:,15]
    data[:,15] .*= 0.622 ./ (0.622 .+ saturation_ratio)
    data[:,5:15] .*= dry_fraction
    return data
end

function trajectory_errors(a, b, total_water)
    mass_errors = [100mean(abs.(a[:,column]-b[:,column])) / total_water for column in (6,8,10)]
    number_error = maximum(b[:,11]) > 1e-3 ?
        100maximum(abs.(a[:,11]-b[:,11])) / maximum(b[:,11]) : NaN
    return [maximum(abs.(a[:,4]-b[:,4])); mass_errors; number_error]
end

root = normpath(joinpath(@__DIR__, "../../validation_output/p3_matrix"))
breeze_dt = get(ARGS, 1, "0.01")
reference_dt = get(ARGS, 2, "0.003125")
directory = joinpath(root, "comparison_$(breeze_dt)_$(reference_dt)")
mkpath(directory)
cases = readdlm(joinpath(root, "cases.dat"))
raw_reference = readdlm(joinpath(root, "reference_dt$(reference_dt)", "reference_final.dat"))
reference = Dict{Int, Matrix{Float64}}()
breeze = Dict{Int, Matrix{Float64}}()
final_reference = fill(NaN, size(cases,1), 15)
final_breeze = similar(final_reference)
fill!(final_breeze, NaN)
errors = fill(NaN, size(cases,1), 5)
checks = Vector{Float64}[]
for row in eachrow(cases)
    identifier = Int(row[1])
    a = readdlm(joinpath(root, "breeze_dt$(breeze_dt)", "case_$(lpad(identifier,3,'0')).dat"))
    a = a[a[:,2] .> 0,:]
    b = upstream_moist_basis(raw_reference[raw_reference[:,1] .== identifier,:])
    @assert size(a,1) == size(b,1) == 540 "Incomplete case $identifier"
    @assert maximum(abs.(a[:,2]-b[:,2])) < 1e-6
    if !all(isfinite,a) || !all(isfinite,b)
        @warn "Nonfinite matrix case; retain NaN tiles" identifier
        continue
    end
    breeze[identifier] = a
    reference[identifier] = b
    final_breeze[identifier,:] = a[end,1:15]
    final_reference[identifier,:] = b[end,:]
    errors[identifier,:] = trajectory_errors(a,b,row[5])
    push!(checks, [identifier, maximum(abs.(a[:,3]-b[:,3])),
        maximum(abs.(a[:,16] .- a[1,16])), maximum(abs.(a[:,17] .- row[5])),
        maximum(abs.(vec(sum(b[:,[5,6,8,10,14]];dims=2)) .- row[5])),
        maximum(abs.(a[:,15]-a[:,20])), min(minimum(a[:,5:14]),minimum(b[:,5:14])),
        maximum(abs.(vec(sum(a[:,[5,6,8,10,14]];dims=2)) .- row[5]))])
end
writedlm(joinpath(directory,"final_breeze.dat"),final_breeze)
writedlm(joinpath(directory,"final_upstream.dat"),final_reference)
writedlm(joinpath(directory,"errors.dat"),hcat(cases,errors))
writedlm(joinpath(directory,"physical_checks.dat"),permutedims(hcat(checks...)))
write(joinpath(directory,"physical_check_columns.txt"),
      "case_id max_pressure_difference_Pa theta_drift_K water_drift_Breeze water_drift_upstream supersaturation_identity_error minimum_water_or_number diagnosed_water_drift_Breeze\n")
write(joinpath(directory,"error_columns.txt"),
      "case_id T0_K RH0 ascent_m_s qtotal max_dT_K cloud_error_percent_water rain_error_percent_water ice_error_percent_water ice_number_error_percent_peak\n")

set_theme!(fontsize=22, linewidth=3)
temperatures = unique(cases[:,2])
humidities = unique(cases[:,3])
speeds = unique(cases[:,4])
function matrix_tile(values, T)
    result = fill(NaN,length(humidities),length(speeds))
    for row in eachrow(cases)
        row[2] == T || continue
        h = findfirst(==(row[3]), humidities)
        w = findfirst(==(row[4]), speeds)
        result[h,w] = values[Int(row[1])]
    end
    return result
end
function add_matrix_panel!(position, values, T; title, colorrange, colormap, signed=false)
    ax = Axis(position; xlabel="Initial relative humidity (%)", ylabel="Ascent rate (m/s)", title,
              xticks=(1:length(humidities),string.(Int.(100humidities))),
              yticks=(1:length(speeds),string.(speeds)), xgridvisible=false, ygridvisible=false)
    tile = matrix_tile(values,T)
    hm = heatmap!(ax,1:length(humidities),1:length(speeds),tile;
                  colorrange,colormap,nan_color=:lightgray)
    for h in eachindex(humidities), w in eachindex(speeds)
        value = tile[h,w]
        fraction = (value-colorrange[1])/(colorrange[2]-colorrange[1])
        color = signed ? (abs(fraction-0.5)>0.3 ? :white : :black) : (fraction<0.5 ? :white : :black)
        text!(ax,h,w;text=isfinite(value) ? @sprintf("%.3g",value) : "—",align=(:center,:center),color,fontsize=23)
    end
    return hm
end

# The final-state pages use a common state scale for the two implementations,
# and a separate symmetric scale for their signed difference.
for (column,label,scale,name) in ((4,"Temperature (K)",1.,"temperature"),
    (6,"Cloud liquid (g/kg)",1000.,"cloud"),(8,"Rain (g/kg)",1000.,"rain"),
    (10,"Frozen ice (g/kg)",1000.,"ice"),(11,"Ice number (10³ kg⁻¹)",0.001,"ice_number"))
    a,b = final_breeze[:,column]*scale, final_reference[:,column]*scale
    finite_states = filter(isfinite,vcat(a,b))
    lower,upper = extrema(finite_states)
    upper = max(upper,lower+1e-12)
    difference = a-b
    bound = max(maximum(abs,filter(isfinite,difference)),1e-12)
    fig = Figure(size=(1350,1450))
    Label(fig[0,1:3],"Final $label · after 90 min of ascent";fontsize=27)
    for (row,(values,method)) in enumerate(((b,"Upstream P3"),(a,"Breeze ParcelModel"),(difference,"Breeze − upstream")))
        for (col,T) in enumerate(temperatures)
            hm = add_matrix_panel!(fig[row,col],values,T;title="$method · T₀=$(Int(T)) K",
                colorrange=row==3 ? (-bound,bound) : (lower,upper),
                colormap=row==3 ? :balance : :viridis,signed=row==3)
            col==length(temperatures) && Colorbar(fig[row,3],hm;label)
        end
    end
    save(joinpath(directory,"final_$name.png"),fig;px_per_unit=1.5)
    save(joinpath(directory,"final_$name.svg"),fig)
end

fig = Figure(size=(1350,1700))
Label(fig[0,1:3],"Trajectory errors · Breeze ParcelModel vs upstream P3";fontsize=27)
for (row,label) in enumerate(("Maximum |ΔT| (K)","Mean |Δ cloud| (% initial water)",
                              "Mean |Δ rain| (% initial water)","Mean |Δ ice| (% initial water)"))
    values = errors[:,row]
    bound = max(maximum(filter(isfinite,values)),1e-12)
    for (col,T) in enumerate(temperatures)
        hm = add_matrix_panel!(fig[row,col],values,T;title="$label · T₀=$(Int(T)) K",
                               colorrange=(0,bound),colormap=:viridis)
        col==length(temperatures) && Colorbar(fig[row,3],hm)
    end
end
save(joinpath(directory,"trajectory_errors.png"),fig;px_per_unit=1.5)
save(joinpath(directory,"trajectory_errors.svg"),fig)

fig = Figure(size=(1350,1050))
Label(fig[0,1:3],"Adiabatic parcel matrix · 90-minute ascents";fontsize=27)
for (row,(data,label)) in enumerate(((1000final_breeze[:,10],"Breeze final ice (g/kg)"),
                                    (errors[:,1],"Maximum |Breeze − upstream T| (K)")))
    bound = max(maximum(filter(isfinite,data)),1e-12)
    for (col,T) in enumerate(temperatures)
        hm = add_matrix_panel!(fig[row,col],data,T;title="$label · T₀=$(Int(T)) K",
                               colorrange=(0,bound),colormap=:viridis)
        col==length(temperatures) && Colorbar(fig[row,3],hm)
    end
end
Label(fig[3,1:3],"18 cases · no entrainment or fallout · conserved θ_li\nTemperature differences include timestep sensitivity; see refinement plots.";fontsize=20)
save(joinpath(directory,"matrix_overview.png"),fig;px_per_unit=1.5)
save(joinpath(directory,"matrix_overview.svg"),fig)

fig = Figure(size=(1350,600))
Label(fig[0,1:3],"Maximum ice-number error / upstream peak (%)";fontsize=27)
number_errors = errors[:,5]
number_bound = max(maximum(filter(isfinite,number_errors)),1e-12)
for (col,T) in enumerate(temperatures)
    hm = add_matrix_panel!(fig[1,col],number_errors,T;title="T₀=$(Int(T)) K",
                           colorrange=(0,number_bound),colormap=:viridis)
    col==length(temperatures) && Colorbar(fig[1,3],hm;label="% of upstream peak")
end
Label(fig[2,1:3],"Gray: no appreciable upstream ice number; relative error is undefined.";fontsize=19)
save(joinpath(directory,"ice_number_errors.png"),fig;px_per_unit=1.5)
save(joinpath(directory,"ice_number_errors.svg"),fig)

# Envelopes describe this finite design, not confidence or probability intervals.
fig = Figure(size=(1350,1100))
Label(fig[0,1:2],"Adiabatic parcel matrix · $(length(breeze))/$(size(cases,1)) finite paired cases";fontsize=27)
for (index,(column,label,scale)) in enumerate(((4,"Temperature (K)",1.),
    (6,"Cloud liquid (g/kg)",1000.),(8,"Rain (g/kg)",1000.),(10,"Frozen ice (g/kg)",1000.)))
    ax = Axis(fig[1+(index-1)÷2,1+(index-1)%2];xlabel="Time (min)",ylabel=label)
    for (collection,name,color,style) in ((reference,"Upstream P3",:black,:solid),(breeze,"Breeze",:dodgerblue,:dash))
        matrix = hcat([collection[id][:,column]*scale for id in sort(collect(keys(collection)))]...)
        times = first(Base.values(collection))[:,2]/60
        band!(ax,times,vec(minimum(matrix;dims=2)),vec(maximum(matrix;dims=2));color=(color,0.13))
        lines!(ax,times,[median(row) for row in eachrow(matrix)];color,linestyle=style,label=name*" median")
    end
    index==1 && axislegend(ax;position=:lb)
end
Label(fig[3,1:2],"Shading: full min–max range over the selected cases; lines: medians.\nFixed duration means final heights of 2.7, 5.4, and 10.8 km.";fontsize=20)
save(joinpath(directory,"trajectory_envelopes.png"),fig;px_per_unit=1.5)
save(joinpath(directory,"trajectory_envelopes.svg"),fig)

open(joinpath(directory,"summary.txt"),"w") do io
    println(io,"18-case factorial matrix: T0 = 288/300 K; RH0 = 60/80/100%; w = 0.5/1/2 m/s.")
    println(io,"Breeze dt=$breeze_dt s; upstream dt=$reference_dt s; fixed process tau=0.025 s.")
    println(io,"Initial RH is vapor partial pressure / Flatau saturation pressure; epsilon=0.622 for the common initial water.")
    println(io,"No entrainment/fallout; fixed aerosol 3e8 per kg dry air; conserved liquid–ice potential temperature.")
    println(io,"Finite cases: $(length(breeze))/$(size(cases,1)). No acceptance threshold is fitted to the observed differences.")
    println(io,"Water error is the time mean absolute difference / initial total water; not a relative error against tiny species.")
    println(io,"case T0 RH0 w qtotal max_dT_K cloud_error_%water rain_error_%water ice_error_%water ice_number_error_%peak")
    for row in eachrow(hcat(cases,errors))
        println(io,join(row,' '))
    end
    println(io,"Physical check maxima by column: ",maximum(permutedims(hcat(checks...));dims=1))
end
print(read(joinpath(directory,"summary.txt"),String))

@testset "Adiabatic matrix physical consistency" begin
    @test length(breeze) == size(cases,1)
    for check in checks
        @test check[2] < 0.01 # Common pressure forcing, Pa.
        @test check[3] < 1e-8 # Conserved theta_li, K.
        @test check[4] < 1e-10 # Conserved total water, kg/kg.
        @test check[5] < 1e-10 # Upstream water independently summed.
        @test check[6] < 1e-7 # Prognostic versus diagnosed supersaturation.
        @test check[7] >= -1e-12 # Positivity in both implementations.
        @test check[8] < 1e-10 # Breeze water independently summed.
    end
end

# Optional refinement data: retain each resolution and report self-convergence
# separately from cross-implementation differences. These are not pass/fail tests.
open(joinpath(directory,"refinement.txt"),"w") do io
    println(io,"Metrics: maximum |dT| K; mean |d cloud/rain/ice| as % initial water; maximum |dNi| / second trajectory peak (%).")
    for (coarse_breeze,fine_breeze,coarse_reference,fine_reference) in
        (("0.01","0.005","0.003125","0.0015625"),
         ("0.005","0.0025","0.0015625","0.00078125"))
        reference_path = joinpath(root,"reference_dt$fine_reference","reference_final.dat")
        isfile(reference_path) || continue
        raw_fine = readdlm(reference_path)
        raw_coarse = readdlm(joinpath(root,"reference_dt$coarse_reference","reference_final.dat"))
        for identifier in Int.(unique(raw_fine[:,1]))
            coarse_path = joinpath(root,"breeze_dt$coarse_breeze","case_$(lpad(identifier,3,'0')).dat")
            fine_path = joinpath(root,"breeze_dt$fine_breeze","case_$(lpad(identifier,3,'0')).dat")
            isfile(fine_path) || continue
            a = readdlm(coarse_path)[2:end,:]
            b = readdlm(fine_path)[2:end,:]
            u = upstream_moist_basis(raw_coarse[raw_coarse[:,1].==identifier,:])
            v = upstream_moist_basis(raw_fine[raw_fine[:,1].==identifier,:])
            @assert size(a,1) == size(b,1) == size(u,1) == size(v,1) == 540
            @assert all(isfinite,b) && all(isfinite,v)
            @assert maximum(abs.(b[:,2]-v[:,2])) < 1e-6
            water = cases[identifier,5]
            println(io,"Case $identifier: Breeze $coarse_breeze → $fine_breeze; upstream $coarse_reference → $fine_reference")
            println(io,"  Breeze self: ",trajectory_errors(a,b,water))
            println(io,"  Upstream self: ",trajectory_errors(u,v,water))
            println(io,"  Fine cross: ",trajectory_errors(b,v,water))
            peak = argmax(abs.(b[:,4]-v[:,4]))
            println(io,"  Largest fine T discrepancy: time=$(b[peak,2]) s; T_Breeze=$(b[peak,4]) K; T_upstream=$(v[peak,4]) K")
        end
    end
end
print(read(joinpath(directory,"refinement.txt"),String))

fig = Figure(size=(1450,1250))
Label(fig[0,1:3],"Outliers · trajectory differences and timestep sensitivity";fontsize=27)
for (col,identifier) in enumerate((3,9,18))
    _,T,humidity,speed,_ = cases[identifier,:]
    title = "T₀=$(Int(T)) K · RH₀=$(Int(100humidity))% · w=$speed m/s"
    ax_temperature = Axis(fig[1,col];title,xlabel="Time (min)",ylabel="Breeze − upstream T (K)")
    ax_number = Axis(fig[2,col];xlabel="Time (min)",ylabel="Ice number (10³ kg⁻¹)")
    for (breeze_step,reference_step,color,style) in
        (("0.01","0.003125",:gray,:solid),("0.005","0.0015625",:darkorange,:dash),
         ("0.0025","0.00078125",:dodgerblue,:dot))
        path = joinpath(root,"reference_dt$reference_step","reference_final.dat")
        breeze_path = joinpath(root,"breeze_dt$breeze_step","case_$(lpad(identifier,3,'0')).dat")
        isfile(path) && isfile(breeze_path) || continue
        raw = readdlm(path)
        b = upstream_moist_basis(raw[raw[:,1].==identifier,:])
        a = readdlm(breeze_path)[2:end,:]
        @assert size(a,1) == size(b,1) == 540
        times = a[:,2]/60
        lines!(ax_temperature,times,a[:,4]-b[:,4];color,linestyle=style,
               label="Δt = $breeze_step / $reference_step s")
        lines!(ax_number,times,b[:,11]/1000;color,linestyle=:solid,label="Upstream, $reference_step s")
        lines!(ax_number,times,a[:,11]/1000;color,linestyle=:dash,label="Breeze, $breeze_step s")
    end
    if col == 1
        Legend(fig[3,1:3],ax_temperature;orientation=:horizontal,labelsize=16)
        Legend(fig[4,1:3],ax_number;orientation=:horizontal,nbanks=2,labelsize=16)
    end
end
Label(fig[5,1:3],"Top: paired temperature differences. Bottom: upstream solid, Breeze dashed; colors indicate refinement level.\nAll resolutions retain the same Breeze process timescale τ = 0.025 s.";fontsize=20)
save(joinpath(directory,"outlier_refinement.png"),fig;px_per_unit=1.5)
save(joinpath(directory,"outlier_refinement.svg"),fig)
