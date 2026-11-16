# Bytes allocated by the acoustic time steppers, by component and by prognostic field.
#
#   CUDA_VISIBLE_DEVICES=-1 julia --project=<env with Breeze and CloudMicrophysics> \
#       validation/mis_memory/account_timestepper_memory.jl [N] [coefficient sets...]
#
# Builds Float32 models on an N³ CPU grid (default 32, halo 5) and walks the time stepper and the
# model for arrays, counting each array once. A "field" is the storage of one Float32 cell-centered
# 3D field with halos. Totals are scaled to 512×512×256 (halo 5) and to 10⁸ interior cells of that grid.

using Breeze
using Oceananigans
using Oceananigans: prognostic_fields
using CloudMicrophysics: CloudMicrophysics
using Printf

const FT = Float32
const N = isempty(ARGS) ? 32 : parse(Int, ARGS[1])
const H = 5
const requested = length(ARGS) > 1 ? Symbol.(ARGS[2:end]) :
                  [:WickerSkamarock3, :TVDMISB, :MIS4a, :SSPMIS43, :ConvexMIS42, :ConvexMIS52, :ConvexMIS52r20]

const CMExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)

# Every array reachable from `x`, keyed by `objectid` so that aliased storage is counted once.
function collect_arrays!(arrays, seen, x)
    (isbits(x) || x isa Union{Symbol, String, Type, Function, Module}) && return arrays
    id = objectid(x)
    id ∈ seen && return arrays
    push!(seen, id)

    if x isa Array
        if isbitstype(eltype(x))
            arrays[id] = x
        else
            foreach(i -> isassigned(x, i) && collect_arrays!(arrays, seen, x[i]), eachindex(x))
        end
    elseif x isa AbstractArray && !(x isa Oceananigans.Fields.AbstractField) && parent(x) !== x
        collect_arrays!(arrays, seen, parent(x))
    elseif x isa Union{Tuple, NamedTuple}
        foreach(y -> collect_arrays!(arrays, seen, y), x)
    elseif x isa Oceananigans.Grids.AbstractGrid
        nothing # grids are shared by every field and counted with the model
    else
        for name in fieldnames(typeof(x))
            isdefined(x, name) && collect_arrays!(arrays, seen, getfield(x, name))
        end
    end

    return arrays
end

collect_arrays(x) = collect_arrays!(Dict{UInt, Array}(), Set{UInt}(), x)
bytes(arrays) = sum(sizeof, values(arrays); init = 0)

# The arrays of each component of the time stepper that no earlier component and no array of
# `exclude` (the model without its time stepper) holds. The substepper comes first, so storage
# that another component shares with it is counted as the substepper's.
function component_arrays(timestepper, exclude)
    counted = Set(keys(exclude))
    components = Pair{String, Dict{UInt, Array}}[]

    function add!(label, x)
        arrays = filter(p -> p.first ∉ counted, collect_arrays(x))
        union!(counted, keys(arrays))
        push!(components, label => arrays)
    end

    add!("substepper", timestepper.substepper)

    for name in fieldnames(typeof(timestepper))
        name === :substepper && continue
        x = getfield(timestepper, name)
        x isa Union{Number, Base.RefValue} && continue
        if name === :accumulators
            for (k, pair) in enumerate(x), part in keys(pair)
                add!("accumulators[$k].$part", pair[part])
            end
        else
            add!(string(name), x)
        end
    end

    return components
end

function build_model(microphysics, timestepper, timestepper_kwargs)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(CPU(); size = (N, N, N), halo = (H, H, H),
                           x = (0, 12e3), y = (0, 12e3), z = (0, 3e3),
                           topology = (Periodic, Periodic, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                    base_pressure = 101325, reference_potential_temperature = 309)
    return AtmosphereModel(grid; dynamics, microphysics, advection = WENO(order=5), timestepper, timestepper_kwargs)
end

function one_moment_microphysics()
    categories = CMExt.one_moment_cloud_microphysics_categories(FT)
    cloud_formation = NonEquilibriumCloudFormation(nothing, categories.parameters.cloud.ice)
    return CMExt.OneMomentCloudMicrophysics(FT; cloud_formation, categories)
end

microphysics_cases = ["dry" => () -> nothing,
                      "1M"  => one_moment_microphysics,
                      "P3"  => () -> PredictedParticlePropertiesMicrophysics(FT)]

available(name) = hasmethod(Breeze.TimeSteppers.multirate_infinitesimal_step_tableau, Tuple{Val{name}, Type{FT}})

steppers = Any[("WSRK3", :AcousticRungeKutta3, NamedTuple())]
for name in requested
    available(name) || (@warn "coefficient set $name is not defined on this branch"; continue)
    push!(steppers, (string(name), :MultirateInfinitesimalStep,
                     (; coefficients = name, post_loop_scalar_tendencies = false)))
    push!(steppers, (string(name, " post-loop"), :MultirateInfinitesimalStep,
                     (; coefficients = name, post_loop_scalar_tendencies = true)))
end
push!(steppers, ("WSRK3 post-loop", :AcousticRungeKutta3, (; post_loop_scalar_tendencies = true)))

field_bytes = sizeof(FT) * (N + 2H)^3
target_field_bytes = sizeof(FT) * (512 + 2H) * (512 + 2H) * (256 + 2H)
field_bytes_per_1e8 = target_field_bytes * 1e8 / (512 * 512 * 256) # with the halo fraction of the 512×512×256 grid
GiB = 2.0^30

summary_rows = []

for (case, make_microphysics) in microphysics_cases
    reference_total = Ref(0.0)
    for (label, timestepper, timestepper_kwargs) in steppers
        model = build_model(make_microphysics(), timestepper, timestepper_kwargs)
        ts = model.timestepper
        names = keys(prognostic_fields(model))

        # The model without its time stepper
        model_arrays = Dict{UInt, Array}()
        seen = Set{UInt}()
        for name in fieldnames(typeof(model))
            name === :timestepper && continue
            collect_arrays!(model_arrays, seen, getfield(model, name))
        end
        collect_arrays!(model_arrays, seen, model.grid)
        components = component_arrays(ts, model_arrays)

        stepper_bytes = sum(bytes(last(c)) for c in components)
        total = stepper_bytes + bytes(model_arrays)
        label == "WSRK3" && (reference_total[] = total)

        println("\n=== $case | $label | $(length(names)) prognostics $(names) ===")
        @printf("%-34s %9s %12s\n", "component", "fields", "bytes")
        for (component, arrays) in components
            b = bytes(arrays)
            b == 0 && continue
            @printf("%-34s %9.2f %12d\n", component, b / field_bytes, b)
        end
        @printf("%-34s %9.2f %12d\n", "time stepper total", stepper_bytes / field_bytes, stepper_bytes)
        @printf("%-34s %9.2f %12d\n", "model without time stepper", bytes(model_arrays) / field_bytes, bytes(model_arrays))
        @printf("%-34s %9.2f %12d\n", "total", total / field_bytes, total)

        # Per prognostic field: the full-state sets that hold one array per prognostic
        sets = 0
        for (component, arrays) in components
            startswith(component, "substepper") && continue
            length(arrays) == length(names) && (sets += 1)
        end

        push!(summary_rows, (; case, label, prognostics = length(names), sets,
                             stepper_fields = stepper_bytes / field_bytes,
                             model_fields = bytes(model_arrays) / field_bytes,
                             total_fields = total / field_bytes,
                             ratio = total / reference_total[],
                             GiB_512 = total / field_bytes * target_field_bytes / GiB,
                             GB_per_1e8 = total / field_bytes * field_bytes_per_1e8 / 1e9))
    end
end

println("\n=== Summary (Float32, halo $H; fields = cell-centered 3D field equivalents) ===")
@printf("%-5s %-26s %5s %5s %9s %8s %8s %8s %12s %12s\n", "case", "stepper", "prog", "sets", "stepper", "model",
        "total", "/WSRK3", "GiB@512²×256", "GB/1e8cells")
for r in summary_rows
    @printf("%-5s %-26s %5d %5d %9.1f %8.1f %8.1f %8.2f %12.2f %12.2f\n", r.case, r.label, r.prognostics, r.sets,
            r.stepper_fields, r.model_fields, r.total_fields, r.ratio, r.GiB_512, r.GB_per_1e8)
end
