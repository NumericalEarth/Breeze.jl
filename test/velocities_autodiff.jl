# MWE: Kernel with interpolation + offset iteration fails to raise
#
# This isolates the issue where a kernel using interpolation operators
# with offset iteration fails with "failed to raise func" on large grids.

using Oceananigans
using Oceananigans.Architectures: ReactantState, device
using Oceananigans.Grids: topology, halo_size, Periodic, Flat
using KernelAbstractions
using KernelAbstractions: Kernel, ndrange, workgroupsize, CompilerMetadata
using KernelAbstractions.NDIteration: NDIteration, NDRange, workitems, _Size
using Reactant
using Enzyme
using Statistics: mean
using CUDA
using Base: @pure

import KernelAbstractions: get, expand, StaticSize, partition
import KernelAbstractions: __ndrange, __groupsize

Reactant.set_default_backend("cpu")

#####
##### KernelParameters (from Oceananigans)
#####

struct OffsetKernelParams{S, O} end

function OffsetKernelParams(r1::AbstractUnitRange, r2::AbstractUnitRange, r3::AbstractUnitRange)
    sz = (length(r1), length(r2), length(r3))
    off = (first(r1) - 1, first(r2) - 1, first(r3) - 1)
    return OffsetKernelParams{sz, off}()
end

#####
##### OffsetStaticSize (from Oceananigans)
#####

struct OffsetStaticSize{S} <: _Size
    function OffsetStaticSize{S}() where S
        new{S::Tuple{Vararg}}()
    end
end

@pure OffsetStaticSize(s::Tuple{Vararg{UnitRange{Int}}}) = OffsetStaticSize{s}()
@pure OffsetStaticSize(s::Tuple{Vararg{Int}}) = OffsetStaticSize{s}()
@pure get(::Type{OffsetStaticSize{S}}) where {S} = S
@pure get(::OffsetStaticSize{S}) where {S} = S
@pure Base.length(::OffsetStaticSize{S}) where {S} = prod(length, S)

@inline ka_offsets(ranges::NTuple{3, UnitRange}) = (ranges[1].start - 1, ranges[2].start - 1, ranges[3].start - 1)
@inline getrange(::Type{OffsetStaticSize{S}}) where {S} = map(length, S), ka_offsets(S)

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
    iterspace = NDRange{length(range), StaticSize{blocks}, StaticSize{groupsize}}(blocks, OffsetStaticSize(Tuple(o for o in offs)))
    return iterspace, dynamic
end

#####
##### Minimal launch with offset
#####

function offset_launch!(arch, grid, kp::OffsetKernelParams{S, O}, kernel!, args...) where {S, O}
    workgroup = (16, 16, 1)  # heuristic_workgroup for 3D
    range = Tuple(1+o:s+o for (s, o) in zip(S, O))
    worksize = OffsetStaticSize(range)

    loop! = kernel!(device(arch), StaticSize(workgroup), worksize)
    loop!(args...)
    return nothing
end

#####
##### Interpolation operators
#####

@inline interp_x(i, j, k, ρ) = @inbounds 0.5 * (ρ[i-1, j, k] + ρ[i, j, k])
@inline interp_y(i, j, k, ρ) = @inbounds 0.5 * (ρ[i, j-1, k] + ρ[i, j, k])
@inline interp_z(i, j, k, ρ) = @inbounds 0.5 * (ρ[i, j, k-1] + ρ[i, j, k])

#####
##### Offset iteration helper
#####

offset_range(::Periodic, N, H) = -H+1:N+H
offset_range(::Flat, N, H) = 1:N

#####
##### Kernel with interpolation (neighbor access)
#####

@kernel function _velocity_kernel!(u, v, w, ρ, ρu, ρv, ρw)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρ_x = interp_x(i, j, k, ρ)
        ρ_y = interp_y(i, j, k, ρ)
        ρ_z = interp_z(i, j, k, ρ)

        u[i, j, k] = ρu[i, j, k] / ρ_x
        v[i, j, k] = ρv[i, j, k] / ρ_y
        w[i, j, k] = ρw[i, j, k] / ρ_z
    end
end

#####
##### Launch with offset iteration
#####

function compute_vels!(u, v, w, ρ, ρu, ρv, ρw, grid)
    arch = grid.architecture
    TX, TY, TZ = topology(grid)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    ii = offset_range(TX(), Nx, Hx)
    jj = offset_range(TY(), Ny, Hy)
    kk = offset_range(TZ(), Nz, Hz)

    kp = OffsetKernelParams(ii, jj, kk)

    offset_launch!(arch, grid, kp, _velocity_kernel!, u, v, w, ρ, ρu, ρv, ρw)
    return nothing
end

#####
##### Test setup
#####

grid = RectilinearGrid(ReactantState(); size=(11, 11), extent=(1000, 1000),
                       halo=(3, 3), topology=(Periodic, Periodic, Flat))

u = XFaceField(grid)
v = YFaceField(grid)
w = ZFaceField(grid)
ρ = CenterField(grid)
ρu = XFaceField(grid)
ρv = YFaceField(grid)
ρw = ZFaceField(grid)

set!(ρ, 1.0)
set!(ρu, 0.1)
set!(ρv, 0.1)
set!(ρw, 0.0)

du = Enzyme.make_zero(u)
dv = Enzyme.make_zero(v)
dw = Enzyme.make_zero(w)
dρ = Enzyme.make_zero(ρ)
dρu = Enzyme.make_zero(ρu)
dρv = Enzyme.make_zero(ρv)
dρw = Enzyme.make_zero(ρw)

#####
##### Loss and gradient
#####

function velocity_loss(u, v, w, ρ, ρu, ρv, ρw, grid)
    compute_vels!(u, v, w, ρ, ρu, ρv, ρw, grid)
    return mean(interior(u))
end

function grad_velocity_loss(u, du, v, dv, w, dw, ρ, dρ, ρu, dρu, ρv, dρv, ρw, dρw, grid)
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        velocity_loss, Enzyme.Active,
        Enzyme.Duplicated(u, du),
        Enzyme.Duplicated(v, dv),
        Enzyme.Duplicated(w, dw),
        Enzyme.Duplicated(ρ, dρ),
        Enzyme.Duplicated(ρu, dρu),
        Enzyme.Duplicated(ρv, dρv),
        Enzyme.Duplicated(ρw, dρw),
        Enzyme.Const(grid))
    return loss_value
end

#####
##### Compile
#####

compiled = Reactant.@compile raise=true raise_first=true grad_velocity_loss(
    u, du, v, dv, w, dw, ρ, dρ, ρu, dρu, ρv, dρv, ρw, dρw, grid)
