# MWE: Non-zero offset in kernel index causes Reactant raising to fail

using KernelAbstractions
using Reactant, Enzyme, CUDA

Reactant.set_default_backend("cpu")

offset = 0  # fails; change to 0 and all tests pass
len = 11
size = (len, len)

@kernel function _k!(B, A, off)
    i, j = @index(Global, NTuple)
    @inbounds B[i + off, j + off] = A[i + off, j + off] * 2
end

function loss(B, A)
    kernel = _k!(get_backend(A))
    kernel(B, A, offset; ndrange=size)
    sum(B)
end

function grad_loss(B, dB, A, dA)
    _, l = Enzyme.autodiff(Enzyme.ReverseWithPrimal, loss, Enzyme.Active,
                    Enzyme.Duplicated(B, dB), Enzyme.Duplicated(A, dA))
    return dA, l
end

# Test 1: Direct Julia execution
println("Test 1: loss(B, A)")
B1, A1 = zeros(size...), ones(size...)
println("  result: ", loss(B1, A1))

# Test 2: Reactant forward compilation
println("Test 2: Reactant.@compile loss")
A2 = Reactant.to_rarray(ones(size...))
B2 = Reactant.to_rarray(zeros(size...))
compiled_loss = Reactant.@compile loss(B2, A2)
println("  result: ", compiled_loss(B2, A2))

# Test 3: Reactant autodiff compilation
println("Test 3: Reactant.@compile grad_loss")
A3 = Reactant.to_rarray(ones(size...))
B3 = Reactant.to_rarray(zeros(size...))
compiled_grad = Reactant.@compile raise=true raise_first=true grad_loss(B3, similar(B3), A3, similar(A3))
println("  result: ", compiled_grad(B3, similar(B3), A3, similar(A3)))
