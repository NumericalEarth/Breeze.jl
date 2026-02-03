# MWE: DelinearizeIndexingPass segfault with ReactantBackend + check-bounds
# Run: julia --project=test --check-bounds=yes test/delinearize/test_reactant.jl

using Reactant, Enzyme, KernelAbstractions, CUDA
using Statistics: mean
using KernelAbstractions: StaticSize

Reactant.set_default_backend("cpu")
Reactant.allowscalar(true)

H, N = 2, 4
total = N + 2*H

@kernel function halo_kernel!(c, H, N)
    j = @index(Global, Linear)
    @inbounds for i = 1:H
        c[i, j] = c[N+i, j]
    end
end

# Toggle between these two lines to see the difference:
const ReactantBackend = Base.get_extension(Reactant, :ReactantKernelAbstractionsExt).ReactantBackend
dev = ReactantBackend()  # ← FAILS with segfault
# dev = KernelAbstractions.CPU()  # ← WORKS

kernel! = halo_kernel!(dev, StaticSize((8,8)), StaticSize((total,total)))

loss(c) = (kernel!(c, H, N); mean(c.^2))
grad(c, dc) = (dc .= 0; Enzyme.autodiff(Enzyme.ReverseWithPrimal, loss, Active, Duplicated(c, dc)))

c  = Reactant.to_rarray(zeros(total, total))
dc = Reactant.to_rarray(zeros(total, total))

Reactant.@compile raise=true raise_first=true sync=true grad(c, dc)
