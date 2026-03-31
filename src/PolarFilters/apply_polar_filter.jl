using Oceananigans: prognostic_fields, fields, Callback, add_callback!, IterationInterval
using Oceananigans.BoundaryConditions: fill_halo_regions!

"""
$(TYPEDSIGNATURES)

Apply `filter` to a single `field`, damping high-wavenumber zonal modes
at latitudes poleward of the filter's threshold.

The algorithm:

1. Gather filtered latitude rows into a contiguous buffer.
2. Batched forward `rfft` along the zonal dimension.
3. Multiply by the pre-computed spectral mask.
4. Batched inverse `brfft` (with `1/Nλ` normalization).
5. Scatter the filtered data back into the field.
"""
function apply_polar_filter!(filter::PolarFilter, field)
    Nλ = filter.grid.Nx
    Nz = filter.grid.Nz
    N_filtered = length(filter.filtered_indices)

    N_filtered == 0 && return nothing

    buf_r = filter.buffer_real
    buf_c = filter.buffer_complex
    data = interior(field)

    ## Bring interior data to CPU for FFT processing
    data_cpu = Array(data)

    ## Gather filtered rows into buffer_real
    for (row, j) in enumerate(filter.filtered_indices)
        for k in 1:Nz
            batch = (row - 1) * Nz + k
            @views buf_r[:, batch] .= data_cpu[:, j, k]
        end
    end

    ## Forward rfft (batched along dim 1)
    buf_c .= filter.forward_plan * buf_r

    ## Apply spectral mask
    for (row, _) in enumerate(filter.filtered_indices)
        for k in 1:Nz
            batch = (row - 1) * Nz + k
            @views buf_c[:, batch] .*= filter.spectral_mask[:, row]
        end
    end

    ## Inverse brfft (batched along dim 1) — brfft is unnormalized, apply 1/Nλ
    buf_r .= filter.inverse_plan * buf_c
    buf_r .*= 1 / Nλ

    ## Scatter filtered rows back into the CPU copy
    for (row, j) in enumerate(filter.filtered_indices)
        for k in 1:Nz
            batch = (row - 1) * Nz + k
            @views data_cpu[:, j, k] .= buf_r[:, batch]
        end
    end

    ## Copy the full array back to the device
    copyto!(data, data_cpu)

    return nothing
end

"""
$(TYPEDSIGNATURES)

Apply `filter` to every field in the iterable `fields_tuple`.
"""
function apply_polar_filter!(filter::PolarFilter, fields_tuple::Union{NamedTuple, Tuple})
    for field in fields_tuple
        apply_polar_filter!(filter, field)
    end
    return nothing
end

# Callback function invoked by Oceananigans' run! loop
function _polar_filter_callback!(simulation, filter)
    model = simulation.model
    apply_polar_filter!(filter, prognostic_fields(model))
    fill_halo_regions!(prognostic_fields(model), model.clock, fields(model))
    return nothing
end

"""
$(TYPEDSIGNATURES)

Add a polar filter callback to `simulation` that damps high-wavenumber zonal
modes poleward of `threshold_latitude` after each time step.

This is the recommended entry point for using the polar filter. It constructs
a [`PolarFilter`](@ref), wraps it in an Oceananigans `Callback`, and registers
it with the simulation.

Keyword Arguments
=================

- `threshold_latitude`: latitude in degrees poleward of which filtering is active (default `60`).
- `filter_mode`: [`SharpTruncation`](@ref) or [`ExponentialRolloff`](@ref)`(order)` (default `ExponentialRolloff(8)`).
- `schedule`: callback schedule (default `IterationInterval(1)` = every time step).

Returns the constructed [`PolarFilter`](@ref).
"""
function add_polar_filter!(simulation;
                           threshold_latitude = 60,
                           filter_mode = ExponentialRolloff(8),
                           schedule = IterationInterval(1))

    grid = simulation.model.grid
    filter = PolarFilter(grid; threshold_latitude, filter_mode)

    callback = Callback(_polar_filter_callback!, schedule; parameters=filter)
    add_callback!(simulation, callback)

    return filter
end
