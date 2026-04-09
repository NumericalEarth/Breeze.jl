using Oceananigans: architecture
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Utils: launch!

#####
##### Gather kernels: copy intensive form f = ρf/ρ into the filter buffer
#####

@inline function _interpolate_density(i, j, k, grid, ρ, ::Val{:xface})
    return ℑxᶠᵃᵃ(i, j, k, grid, ρ)
end

@inline function _interpolate_density(i, j, k, grid, ρ, ::Val{:yface})
    return ℑyᵃᶠᵃ(i, j, k, grid, ρ)
end

@inline function _interpolate_density(i, j, k, grid, ρ, ::Val{:center})
    return @inbounds ρ[i, j, k]
end

@kernel function _gather_intensive!(buffer, ρf, ρ, filtered_indices, grid, ::Val{loc}) where loc
    i, j_local, k = @index(Global, NTuple)
    j = @inbounds filtered_indices[j_local]
    ρ_local = _interpolate_density(i, j, k, grid, ρ, Val(loc))
    @inbounds buffer[i, j_local, k] = ρf[i, j, k] / ρ_local
end

#####
##### Shapiro 1-2-1 smoother pass
#####

@kernel function _shapiro_pass!(out, in_, passes_per_row, current_pass, Nλ)
    i, j_local, k = @index(Global, NTuple)
    i_left  = ifelse(i == 1,  Nλ, i - 1)
    i_right = ifelse(i == Nλ, 1,  i + 1)
    @inbounds begin
        do_smooth = current_pass <= passes_per_row[j_local]
        smoothed = (in_[i_left, j_local, k] + 2 * in_[i, j_local, k] + in_[i_right, j_local, k]) / 4
        out[i, j_local, k] = ifelse(do_smooth, smoothed, in_[i, j_local, k])
    end
end

#####
##### Scatter kernels: write filtered intensive form back as ρf = f_filtered · ρ
#####

@kernel function _scatter_intensive!(ρf, buffer, ρ, filtered_indices, grid, ::Val{loc}) where loc
    i, j_local, k = @index(Global, NTuple)
    j = @inbounds filtered_indices[j_local]
    ρ_local = _interpolate_density(i, j, k, grid, ρ, Val(loc))
    @inbounds ρf[i, j, k] = buffer[i, j_local, k] * ρ_local
end

#####
##### Public entry point for a single field
#####

"""
$(TYPEDSIGNATURES)

Apply the polar filter to a single conservation-form field `ρf` by filtering
its intensive form `f = ρf / ρ` (with appropriate stagger interpolation),
then writing back `ρf = f_filtered · ρ`.

`loc` should be `Val(:xface)`, `Val(:yface)`, or `Val(:center)`.
"""
function apply_polar_filter_intensive!(filter::PolarFilter, ρf, ρ, loc::Val)
    grid = filter.grid
    arch = architecture(grid)
    Nλ = grid.Nx
    Nz = grid.Nz
    N_filtered = length(filter.filtered_indices)

    N_filtered == 0 && return nothing

    worksize = (Nλ, N_filtered, Nz)

    ## Gather: compute intensive form into buffer_a
    launch!(arch, grid, worksize,
            _gather_intensive!, filter.buffer_a, ρf, ρ, filter.filtered_indices, grid, loc)

    ## Smooth: ping-pong between buffer_a and buffer_b
    passes_per_row = filter.passes_per_row
    max_N = maximum(Array(passes_per_row))

    buf_in  = filter.buffer_a
    buf_out = filter.buffer_b

    for pass in 1:max_N
        launch!(arch, grid, worksize,
                _shapiro_pass!, buf_out, buf_in, passes_per_row, pass, Nλ)
        ## Swap buffers
        buf_in, buf_out = buf_out, buf_in
    end

    ## After the loop, buf_in holds the final result (last write was to what is now buf_in)

    ## Scatter: write back as ρf = f_filtered · ρ
    launch!(arch, grid, worksize,
            _scatter_intensive!, ρf, buf_in, ρ, filter.filtered_indices, grid, loc)

    return nothing
end

#####
##### Dynamics-level entry point (called from the time stepper)
#####

#####
##### Direct field filtering (for fields already in the right form)
#####

@kernel function _gather_field!(buffer, field, filtered_indices)
    i, j_local, k = @index(Global, NTuple)
    j = @inbounds filtered_indices[j_local]
    @inbounds buffer[i, j_local, k] = field[i, j, k]
end

@kernel function _scatter_field!(field, buffer, filtered_indices)
    i, j_local, k = @index(Global, NTuple)
    j = @inbounds filtered_indices[j_local]
    @inbounds field[i, j, k] = buffer[i, j_local, k]
end

"""
$(TYPEDSIGNATURES)

Apply the polar filter directly to a field (no intensive-form conversion).
Use this for fields that are already in the form you want to smooth
(e.g., velocity `u`, `v`, or perturbation `ρθ″`).
"""
function apply_polar_filter_field!(filter::PolarFilter, field)
    grid = filter.grid
    arch = architecture(grid)
    Nλ = grid.Nx
    N_filtered = length(filter.filtered_indices)

    N_filtered == 0 && return nothing

    ## Use the field's actual z-extent (Nz for center, Nz+1 for z-face)
    Nk = size(interior(field), 3)
    worksize = (Nλ, N_filtered, Nk)

    ## Gather
    launch!(arch, grid, worksize,
            _gather_field!, filter.buffer_a, field, filter.filtered_indices)

    ## Smooth: ping-pong
    passes_per_row = filter.passes_per_row
    max_N = maximum(Array(passes_per_row))

    buf_in  = filter.buffer_a
    buf_out = filter.buffer_b

    for pass in 1:max_N
        launch!(arch, grid, worksize,
                _shapiro_pass!, buf_out, buf_in, passes_per_row, pass, Nλ)
        buf_in, buf_out = buf_out, buf_in
    end

    ## Scatter
    launch!(arch, grid, worksize,
            _scatter_field!, field, buf_in, filter.filtered_indices)

    return nothing
end

#####
##### Dynamics-level entry points
#####

"""
$(TYPEDSIGNATURES)

Apply the polar filter to the intensive forms of `ρu`, `ρv`, and `ρθ`. Density
and vertical momentum `ρw` are deliberately not filtered, following the WRF and
CAM-FV convention. Called at the end of each WS-RK3 stage.
"""
function _apply_polar_filter!(filter::PolarFilter, model)
    ρ = model.dynamics.density
    ρu = model.momentum.ρu
    ρv = model.momentum.ρv
    ρθ = model.formulation.potential_temperature_density

    apply_polar_filter_intensive!(filter, ρu, ρ, Val(:xface))
    apply_polar_filter_intensive!(filter, ρv, ρ, Val(:yface))
    apply_polar_filter_intensive!(filter, ρθ, ρ, Val(:center))

    return nothing
end

_apply_polar_filter!(::Nothing, model) = nothing

"""
$(TYPEDSIGNATURES)

Apply the polar filter to velocity and perturbation fields inside the acoustic
substep loop. Called after the horizontal forward step. Filters `u`, `v` (already
intensive), and optionally the perturbation momentum `ρu″`, `ρv″`.
"""
function _apply_polar_filter_substep!(filter::PolarFilter, u, v, ρu″, ρv″)
    apply_polar_filter_field!(filter, u)
    apply_polar_filter_field!(filter, v)
    apply_polar_filter_field!(filter, ρu″)
    apply_polar_filter_field!(filter, ρv″)
    return nothing
end

_apply_polar_filter_substep!(::Nothing, u, v, ρu″, ρv″) = nothing

"""
$(TYPEDSIGNATURES)

Apply the polar filter to perturbation scalar fields after the acoustic solve.
Matches WRF's `pxft(flag_t=1, flag_mu=1)` call after the θ/μ update.
"""
function _apply_polar_filter_scalar_substep!(filter::PolarFilter, ρθ″, ρ″, ρw″)
    apply_polar_filter_field!(filter, ρθ″)
    apply_polar_filter_field!(filter, ρ″)
    apply_polar_filter_field!(filter, ρw″)
    return nothing
end

_apply_polar_filter_scalar_substep!(::Nothing, ρθ″, ρ″, ρw″) = nothing

"""
$(TYPEDSIGNATURES)

Apply the polar filter to the recovered state after the acoustic substep loop.
Filters velocities u, v, w and scalars ρ, ρθ as direct fields (no
intensive-form conversion) to prevent high-k content from leaking across
RK3 stages.
"""
function _apply_polar_filter_recovered!(filter::PolarFilter, u, v, w, ρ, ρθ, momentum)
    apply_polar_filter_field!(filter, u)
    apply_polar_filter_field!(filter, v)
    apply_polar_filter_field!(filter, w)
    apply_polar_filter_field!(filter, ρ)
    apply_polar_filter_field!(filter, ρθ)
    ## Also filter conservation-form momentum: ρu = ρ·u reintroduces high-k
    ## content through the nonlinear product even when ρ and u are individually
    ## filtered. Filtering ρu/ρv prevents high-k leakage into WENO advection.
    apply_polar_filter_field!(filter, momentum.ρu)
    apply_polar_filter_field!(filter, momentum.ρv)
    apply_polar_filter_field!(filter, momentum.ρw)
    return nothing
end

_apply_polar_filter_recovered!(::Nothing, u, v, w, ρ, ρθ, momentum) = nothing
