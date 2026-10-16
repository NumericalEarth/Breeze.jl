# Re-run outliers through production ParcelModel with finer read-only output.
include("parcel_compare.jl")

root = normpath(joinpath(@__DIR__, "../../validation_output/p3_matrix"))
cases = readdlm(joinpath(root,"cases.dat"))
directory = joinpath(root,"outliers/breeze_matched_constants")
mkpath(directory)
identifiers = isempty(ARGS) ? (3,9,18) : parse.(Int,split(ARGS[1],','))
for identifier in identifiers
    row = cases[identifier,:]
    data = run_production_parcel(row[4];dt=0.0025,timescale=0.025,configuration="reference_constants",
        initial_temperature=row[2],total_water=row[5],case_identifier=identifier,output_interval=0.1)
    writedlm(joinpath(directory,"case_$identifier.dat"),data)
end
write(joinpath(directory,"configuration.txt"),
      "Production ParcelModel, dt=0.0025 s, fixed tau=0.025 s; output every 0.1 s.\n" *
      "Upstream aerosol gas constant corrected to 8.3145 J/mol/K; original matrix used 8.314.\n")
