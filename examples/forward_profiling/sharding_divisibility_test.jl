# Sharding Divisibility Test
#
# Demonstrates that when an array axis is not evenly divisible by the
# number of devices in a sharding mesh, Reactant/XLA only distributes
# across the largest device subset that divides the axis — not all devices.
#
# We request 4-device sharding for 1D arrays of length:
#   128  (÷4 ✓)  → expect sharding across 4 devices
#   126  (÷2 ✓, ÷4 ✗) → expect sharding across ≤2 devices
#   125  (odd)   → expect no sharding (replicated on 1 device)
#
# For each case we compile `x .+ 1` and save the HLO so the actual
# sharding annotations can be compared.
#
# Launch (4 GPUs):
#   julia --project=test examples/forward_profiling/sharding_divisibility_test.jl

using Reactant

devices = Reactant.devices()
ndev = length(devices)
mesh = Reactant.Sharding.Mesh(reshape(devices[1:4], 4, 1), (:x, :y))
sharding = Reactant.Sharding.DimsSharding(mesh, (1,), (:x,))

func(x) = x

l = 125

x = Reactant.to_rarray(ones(Float32, l); sharding)
hlo = Reactant.@code_hlo raise=true func(x)

outdir = mkpath(joinpath(@__DIR__, "sharding_divisibility_results"))

path = joinpath(outdir, "hlo_$(l).txt")
open(path, "w") do io
    show(io, hlo)
end
