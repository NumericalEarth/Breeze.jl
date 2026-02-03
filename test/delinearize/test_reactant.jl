# MWE: KernelAbstractions kernel compilation failure with ReactantBackend
# Run: julia --project=test --check-bounds=yes test/delinearize/test_reactant.jl
#
# Behavior (Reactant v0.2.211+):
#   --check-bounds=yes: "failed to raise func" (MLIR affine expressions)
#   --check-bounds=no:  StableHLO shape error (dynamic_update_slice mismatch)
#
# Behavior (Reactant < v0.2.211):
#   --check-bounds=yes: Segfault (process crashes)
#
# Root cause: Loop-dependent indices (for i = 1:H where H > 1) in KA kernels
# generate complex expressions that Reactant cannot compile.

using Reactant, Enzyme, KernelAbstractions, CUDA
using Statistics: mean
using KernelAbstractions: StaticSize

@info "Versions" Reactant=pkgversion(Reactant) Enzyme=pkgversion(Enzyme) KernelAbstractions=pkgversion(KernelAbstractions)

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
dev = ReactantBackend()  # ← FAILS with "failed to raise func" (v0.2.211+)
# dev = KernelAbstractions.CPU()  # ← WORKS

kernel! = halo_kernel!(dev, StaticSize((8,8)), StaticSize((total,total)))

loss(c) = (kernel!(c, H, N); mean(c.^2))
grad(c, dc) = (dc .= 0; Enzyme.autodiff(Enzyme.ReverseWithPrimal, loss, Active, Duplicated(c, dc)))

c  = Reactant.to_rarray(zeros(total, total))
dc = Reactant.to_rarray(zeros(total, total))

Reactant.@compile raise=true raise_first=true sync=true grad(c, dc)
