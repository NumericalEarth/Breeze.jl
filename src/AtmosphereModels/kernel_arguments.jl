using Oceananigans.Architectures: convert_to_device, GPU

#####
##### Device form of the model's field tuples, converted once per argument epoch (host cost)
#####
##### Every launch whose arguments include the model's field tuple (`fields(model)`) or the
##### boundary-condition arguments (`boundary_condition_args(model)`: the clock, the field tuple and the
##### thermodynamic fields) converts the tuple to its device form again (`Architectures.convert_to_device`)
##### — at the halo fills with boundary-condition arguments, the flux boundary conditions, the tendency
##### launches and the implicit solves. The fields are persistent objects whose arrays are never
##### reallocated, so the device form is the same at every launch. `kernel_fields(model)` and
##### `kernel_boundary_condition_args(model)` return it from a per-model cache instead. That is bit-identical
##### by construction: converting the device form again is the identity, so every kernel compiles to, and
##### runs with, the same argument types and values as with the host tuples.
#####
##### An *argument epoch* ends at every `update_model_field_time_series!` (the start of every
##### `update_state!`) and at the start of every `time_step!` of the acoustic RK3 stepper; the first use in
##### the next epoch rebuilds the cache. `KERNEL_ARGUMENT_CACHE[] = false` converts the tuples at every call;
##### `CHECK_KERNEL_ARGUMENT_CACHE[] = true` compares the cached device form with a fresh conversion on
##### every use (`===`, bitwise for the `isbits` device form) and throws if they differ (both also set at
##### load from the environment, see `__init__`). Only GPU
##### architectures use the cache (on the CPU the conversion is cheap and the host tuples are passed).
#####

const KERNEL_ARGUMENT_CACHE = Ref(true)
const CHECK_KERNEL_ARGUMENT_CACHE = Ref(false)
const kernel_argument_epoch = Ref(0)

# The switches at load, from the environment: `BREEZE_KERNEL_ARGUMENT_CACHE=false`,
# `BREEZE_CHECK_KERNEL_ARGUMENT_CACHE=true`.
function __init__()
    KERNEL_ARGUMENT_CACHE[] = lowercase(get(ENV, "BREEZE_KERNEL_ARGUMENT_CACHE", "true")) != "false"
    CHECK_KERNEL_ARGUMENT_CACHE[] = lowercase(get(ENV, "BREEZE_CHECK_KERNEL_ARGUMENT_CACHE", "false")) == "true"
    return nothing
end

mutable struct KernelArgumentCache{F, T}
    epoch :: Int
    fields :: F
    thermodynamic_fields :: T
end

const kernel_argument_caches = WeakKeyDict{AtmosphereModel, KernelArgumentCache}()
const kernel_argument_lock = ReentrantLock()

"""
$(TYPEDSIGNATURES)

End the current kernel-argument epoch: the next `kernel_fields` / `kernel_boundary_condition_args`
of every model converts its field tuples afresh.
"""
end_kernel_argument_epoch!() = (kernel_argument_epoch[] += 1; nothing)

device_model_fields(model) = convert_to_device(model.architecture, fields(model))
device_thermodynamic_fields(model) = convert_to_device(model.architecture, dynamics_thermodynamic_fields(model.dynamics))

function kernel_argument_cache(model)
    epoch = kernel_argument_epoch[]
    cache = @lock kernel_argument_lock get(kernel_argument_caches, model, nothing)
    if cache === nothing || cache.epoch != epoch
        cache = KernelArgumentCache(epoch, device_model_fields(model), device_thermodynamic_fields(model))
        @lock kernel_argument_lock kernel_argument_caches[model] = cache
    end
    return cache
end

@noinline function check_kernel_argument(cached, fresh, name)
    cached === fresh || error("kernel-argument cache: the cached device form of $name differs from a fresh conversion")
    return nothing
end

"""
$(TYPEDSIGNATURES)

The model's field tuple for kernel arguments: on a GPU the device form of `fields(model)`, from the cache
(see `kernel_argument_cache`) or converted afresh when `KERNEL_ARGUMENT_CACHE[]` is `false`; on the CPU
`fields(model)` itself.
"""
# (On a GPU both branches below return the device form, so the result is concretely typed whatever the
# switch says: with the cache off the tuple is converted afresh at every call.)
@inline function kernel_fields(model)
    model.architecture isa GPU || return fields(model)
    KERNEL_ARGUMENT_CACHE[] || return device_model_fields(model)
    F = Core.Compiler.return_type(device_model_fields, Tuple{typeof(model)})
    f = kernel_argument_cache(model).fields::F
    CHECK_KERNEL_ARGUMENT_CACHE[] && check_kernel_argument(f, device_model_fields(model), "fields(model)")
    return f
end

@inline function kernel_thermodynamic_fields(model)
    model.architecture isa GPU || return dynamics_thermodynamic_fields(model.dynamics)
    KERNEL_ARGUMENT_CACHE[] || return device_thermodynamic_fields(model)
    T = Core.Compiler.return_type(device_thermodynamic_fields, Tuple{typeof(model)})
    f = kernel_argument_cache(model).thermodynamic_fields::T
    CHECK_KERNEL_ARGUMENT_CACHE[] && check_kernel_argument(f, device_thermodynamic_fields(model), "the thermodynamic fields")
    return f
end

"""
$(TYPEDSIGNATURES)

`boundary_condition_args(model)` for kernel launches: the clock and the cached device forms of the
field tuples (see `kernel_fields`).
"""
@inline kernel_boundary_condition_args(model::AtmosphereModel) =
    (model.clock, kernel_fields(model), kernel_thermodynamic_fields(model))
