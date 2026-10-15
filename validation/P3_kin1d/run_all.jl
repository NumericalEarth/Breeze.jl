# Run from the repository root after developing Breeze into this study's environment.
# julia --project=validation/P3_kin1d validation/P3_kin1d/run_all.jl
project = dirname(Base.active_project())
julia = Base.julia_cmd()
output_directory = get(ENV, "P3_COMPARISON_OUTPUT",
                       normpath(joinpath(@__DIR__, "../../validation_output/p3_kin1d")))
mkpath(output_directory)
repository = normpath(joinpath(@__DIR__, "../.."))
open(joinpath(output_directory, "breeze_working_tree.patch"), "w") do io
    run(pipeline(`git -C $repository diff HEAD`; stdout=io))
end
open(joinpath(output_directory, "environment.txt"), "w") do io
    println(io, "Julia: ", VERSION)
    println(io, "Project: ", Base.active_project())
    print(io, "Breeze base commit: ", read(`git -C $repository rev-parse HEAD`, String))
    print(io, read(`gfortran --version`, String))
end
for (case, dt) in (("fixed", "1"), ("fixed", "0.5"), ("upwind", "1"),
                   ("reservoir", "1"), ("reference_options", "1"))
    run(`$julia --project=$project $(joinpath(@__DIR__, "run_breeze.jl")) $case $dt`)
end
for dt in ("1", "0.5")
    run(`$julia --project=$project $(joinpath(@__DIR__, "prepare_reference.jl")) $dt`)
end
dry_output = joinpath(output_directory, "dry")
run(addenv(`$julia --project=$project $(joinpath(@__DIR__, "run_breeze.jl")) dry 1`,
           "P3_COMPARISON_OUTPUT" => dry_output))
run(addenv(`$julia --project=$project $(joinpath(@__DIR__, "prepare_reference.jl")) 1`,
           "P3_COMPARISON_OUTPUT" => dry_output))
run(`$julia --project=$project $(joinpath(@__DIR__, "compare.jl"))`)
