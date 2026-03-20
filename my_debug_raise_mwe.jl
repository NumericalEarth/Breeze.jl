using CUDA, Reactant, KernelAbstractions
using Reactant: ConcreteRNumber
using Adapt: Adapt

mutable struct Clock{TT, S}
    time :: TT
    stage :: S
end

Adapt.adapt_structure(to, clock::Clock) = (time          = clock.time,
                                           stage         = clock.stage)

@kernel function _fill!(c, args)
    i = @index(Global)
    @inbounds c[i] = c[i]
end

N = 16
c = Reactant.to_rarray(zeros(N))
clock = Clock(ConcreteRNumber(0.0), 1)

function loss(c, clock)
    backend = KernelAbstractions.get_backend(c)
    _fill!(backend)(c, clock; ndrange=size(c))
    KernelAbstractions.synchronize(backend)
    return 0.0
end

compiled = Reactant.@compile raise=true raise_first=true loss(c, clock)
