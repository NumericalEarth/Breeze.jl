# Reproduce direct upstream P3 validation without kin1d or CM1.
using Pkg
using Dates
root = normpath(joinpath(@__DIR__, "../.."))
directory = joinpath(root, "validation_output/p3_processes")
mkpath(directory)
open(joinpath(directory, "breeze_provenance.txt"), "w") do io
    println(io, "Run UTC: ", now(UTC))
    println(io, "Breeze commit: ", readchomp(`git -C $root rev-parse HEAD`))
    println(io, "Julia: ", VERSION)
    Pkg.status(; io, mode=Pkg.PKGMODE_MANIFEST)
end
write(joinpath(directory, "breeze.patch"), read(`git -C $root diff`, String))
for script in ("process_reference.jl", "process_compare.jl", "trajectory_reference.jl", "trajectory_compare.jl")
    run(`$(Base.julia_cmd()) --project=$(@__DIR__) $(joinpath(@__DIR__, script))`)
end

# Independently implemented liquid–ice potential-temperature host in both drivers.
for dt in (0.05, 0.025), script in ("trajectory_reference.jl", "trajectory_compare.jl")
    run(`$(Base.julia_cmd()) --project=$(@__DIR__) $(joinpath(@__DIR__, script)) $dt liquid_ice`)
end
