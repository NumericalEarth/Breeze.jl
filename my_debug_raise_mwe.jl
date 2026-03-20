using CUDA, Reactant, Enzyme, KernelAbstractions, OffsetArrays
using Reactant: ConcreteRNumber
using Adapt: Adapt

Reactant.Compiler.DUMP_LLVMIR[] = false
Reactant.set_default_backend("cpu")

# ── Clock (from Oceananigans src/TimeSteppers/clock.jl) ──

mutable struct Clock{TT, S}
    time :: TT
    stage :: S
end

Adapt.adapt_structure(to, clock::Clock) = (time          = clock.time,
                                           stage         = clock.stage)

@kernel function _fill_halo!(c, N, H, args)
    i = @index(Global)
    @inbounds parent(c)[i] = parent(c)[i]
end

# ── Setup ──

Nx, H = 16, 3
raw = Reactant.to_rarray(zeros(Nx + 2H))
c   = OffsetArray(raw, 1-H:Nx+H)
clock = Clock(ConcreteRNumber(0.0), 1)
mf    = ()

function loss(c, clock, mf)
    backend = KernelAbstractions.get_backend(parent(c))
    _fill_halo!(backend)(c, Nx, H, (mf, clock); ndrange=size(c))
    KernelAbstractions.synchronize(backend)
    return 0.0
end

@info "Compiling..."
@time compiled = Reactant.@compile raise=true raise_first=true loss(c, clock, mf)
