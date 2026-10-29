module BreezeCUDAExt

using Breeze.Utils: Utils
using Oceananigans: Oceananigans
using Oceananigans.Architectures: GPU, convert_to_device
using CUDA: CUDA
using CUDA.CUDAKernels: CUDABackend
using KernelAbstractions: KernelAbstractions as KA

#####
##### Register-capped launches (`Utils.launch_with_maxregs!`; handoff 582e072d C9c, e490c5ed I1b, fc4445df)
#####
##### The arguments are put in their device form with Oceananigans' `convert_to_device`, as `launch!` does, so these
##### launches compile for the same argument types as every `launch!` kernel. With `Utils.CACHED_MAXREGS_LAUNCH[]`
##### a static-workgroup kernel object whose arguments are all in device form is compiled once per
##### (context, function, argument types, maxthreads, always_inline, maxregs) with exactly the `cufunction` call of
##### the uncached launch and launched with `convert = Val(false)`: same machine code, arguments and launch shape.
##### `Utils.LAUNCH_MAXREGS[] = false` ignores every cap.
#####

const cached_maxregs_kernels = Dict{Any, Any}()
const cached_maxregs_lock = ReentrantLock()
const cached_maxregs_counts = zeros(Int, 2)   # cached launches / uncached launches

const OceananigansCUDA = Base.get_extension(Oceananigans, :OceananigansCUDAExt)
const fork_device_form_test = !isnothing(OceananigansCUDA) && isdefined(OceananigansCUDA, :all_device_form)

@inline all_device_form(kernel_args) = fork_device_form_test ? OceananigansCUDA.all_device_form(kernel_args) : false

function cached_maxregs_kernel(f::F, ::Type{TT}, maxthreads, always_inline, maxregs) where {F, TT}
    key = (CUDA.context(), f, TT, maxthreads, always_inline, maxregs)
    kernel = @lock cached_maxregs_lock get(cached_maxregs_kernels, key, nothing)
    if kernel === nothing
        kernel = maxregs > 0 ? CUDA.cufunction(f, TT; always_inline, maxthreads, maxregs) :
                               CUDA.cufunction(f, TT; always_inline, maxthreads)
        @lock cached_maxregs_lock cached_maxregs_kernels[key] = kernel
    end
    return kernel::CUDA.HostKernel{F, TT}
end

# `true` when launched (or empty), `false` when the launch must take the uncached path
@inline cached_maxregs_launch!(obj, maxregs, ndrange, args...) = false
@inline function cached_maxregs_launch!(obj::KA.Kernel{CUDABackend, <:KA.StaticSize}, maxregs, ndrange, args...)
    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, nothing)
    ctx = KA.mkcontext(obj, ndrange, iterspace)
    kernel_args = (ctx, args...)
    all_device_form(kernel_args) || return false
    threads = length(KA.workitems(iterspace))
    blocks = length(KA.blocks(iterspace))
    cached_maxregs_counts[1] += 1
    blocks == 0 && return true
    maxthreads = prod(KA.get(KA.workgroupsize(obj)))
    always_inline = KA.backend(obj).always_inline
    kernel = cached_maxregs_kernel(obj.f, typeof(kernel_args), maxthreads, always_inline, maxregs)
    kernel(kernel_args...; threads, blocks, convert = Val(false))
    return true
end

function Utils.launch_with_maxregs!(arch::GPU{<:CUDABackend}, obj, maxregs, host_args...; ndrange)
    maxregs = Utils.LAUNCH_MAXREGS[] ? maxregs : 0
    GC.@preserve host_args begin
        args = convert_to_device(arch, host_args)
        Utils.CACHED_MAXREGS_LAUNCH[] && cached_maxregs_launch!(obj, maxregs, ndrange, args...) && return nothing
        cached_maxregs_counts[2] += 1
        if maxregs > 0
            ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, nothing)
            ctx = KA.mkcontext(obj, ndrange, iterspace)
            threads = length(KA.workitems(iterspace))
            blocks = length(KA.blocks(iterspace))
            kernel = CUDA.@cuda launch=false always_inline=KA.backend(obj).always_inline maxthreads=threads maxregs=maxregs obj.f(ctx, args...)
            kernel(ctx, args...; threads, blocks)
        else
            obj(args...; ndrange)
        end
    end
    return nothing
end

end # module
