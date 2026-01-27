# Minimal Working Example: Autodiff through kernel_call fails
#
# Error: "could not compute the adjoint for this operation enzymexla.kernel_call"

using KernelAbstractions # version 0.9.0
using Reactant # version 0.2.203   
using Enzyme # version 0.13.118
using CUDA

Reactant.set_default_backend("cpu")

@kernel function _add_kernel!(output, input)
    i, j = @index(Global, NTuple)
    @inbounds output[i, j] = output[i, j] + input[i, j]
end

function kernel_loss(output, input)
    backend = get_backend(output)
    kernel = _add_kernel!(backend)
    kernel(output, input; ndrange=size(output))
    return sum(output)
end

function grad_kernel_loss(output, doutput, input, dinput)
    doutput .= 0
    dinput .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        kernel_loss, Enzyme.Active,
        Enzyme.Duplicated(output, doutput),
        Enzyme.Duplicated(input, dinput))
    return dinput, loss_value
end

output = Reactant.to_rarray(zeros(16, 16))
doutput = Reactant.to_rarray(zeros(16, 16))
input = Reactant.to_rarray(rand(16, 16))
dinput = Reactant.to_rarray(zeros(16, 16))

compiled = Reactant.@compile raise=true raise_first=true grad_kernel_loss(
    output, doutput, input, dinput)