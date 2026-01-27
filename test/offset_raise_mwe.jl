# MWE: Negative offset indices fail to raise in Reactant autodiff

using KernelAbstractions
using KernelAbstractions: Kernel, ndrange, workgroupsize, CompilerMetadata
using KernelAbstractions.NDIteration: NDIteration, NDRange, workitems, _Size
using Reactant
using Enzyme
using CUDA
using Base: @pure

import KernelAbstractions: get, expand, StaticSize, partition
import KernelAbstractions: __ndrange, __groupsize

Reactant.set_default_backend("cpu")

struct OffsetStaticSize{S} <: _Size
    function OffsetStaticSize{S}() where S
        new{S::Tuple{Vararg}}()
    end
end

@pure OffsetStaticSize(s::Tuple{Vararg{UnitRange{Int}}}) = OffsetStaticSize{s}()
@pure OffsetStaticSize(s::Tuple{Vararg{Int}}) = OffsetStaticSize{s}()
@pure get(::Type{OffsetStaticSize{S}}) where {S} = S
@pure Base.length(::OffsetStaticSize{S}) where {S} = prod(length, S)

@inline offsets_from_ranges(ranges::NTuple{2, UnitRange}) = (ranges[1].start - 1, ranges[2].start - 1)
@inline getrange(::Type{OffsetStaticSize{S}}) where {S} = map(length, S), offsets_from_ranges(S)

const OffsetNDRange{N, S} = NDRange{N, <:StaticSize, <:StaticSize, <:Any, <:OffsetStaticSize{S}} where {N, S}

@inline function expand(ndrange::OffsetNDRange{N, S}, groupidx::CartesianIndex{N}, idx::CartesianIndex{N}) where {N, S}
    nI = ntuple(Val(N)) do I
        stride = size(workitems(ndrange), I)
        (groupidx.I[I] - 1) * stride + idx.I[I] + S[I]
    end
    return CartesianIndex(nI)
end

@inline __ndrange(::CompilerMetadata{NDRange}) where {NDRange<:OffsetStaticSize} = CartesianIndices(get(NDRange))
@inline __groupsize(cm::CompilerMetadata{NDRange}) where {NDRange<:OffsetStaticSize} = size(__ndrange(cm))

const OffsetKernel = Kernel{<:Any, <:StaticSize, <:OffsetStaticSize}

function partition(kernel::OffsetKernel, ::Nothing, ::Nothing)
    range, offs = getrange(ndrange(kernel))
    groupsize = get(workgroupsize(kernel))
    blocks, groupsize, dynamic = NDIteration.partition(range, groupsize)
    iterspace = NDRange{length(range), StaticSize{blocks}, StaticSize{groupsize}}(blocks, OffsetStaticSize(offs))
    return iterspace, dynamic
end

@kernel function _kernel!(B, A)
    i, j = @index(Global, NTuple)
    @inbounds B[i, j] = A[i, j] * 2
end

offset = -2:14

function loss(B, A)
    backend = get_backend(A)
    kernel = _kernel!(backend, StaticSize((16, 16)), OffsetStaticSize((offset, offset)))
    kernel(B, A)
    return sum(B)
end

function grad_loss(B, dB, A, dA)
    dB .= 0
    dA .= 0
    Enzyme.autodiff(Enzyme.ReverseWithPrimal, loss, Enzyme.Active,
                    Enzyme.Duplicated(B, dB), Enzyme.Duplicated(A, dA))
end

A = Reactant.to_rarray(rand(17, 17))
B = Reactant.to_rarray(zeros(17, 17))
dA = Reactant.to_rarray(zeros(17, 17))
dB = Reactant.to_rarray(zeros(17, 17))

# FAILS: negative offset (-2:14)
# WORKS: positive offset (1:17)
# compiled = Reactant.@compile raise=true raise_first=true grad_loss(B, dB, A, dA)
Reactant.@compile loss(B, A)
