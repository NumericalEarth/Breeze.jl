#####
##### Shared helpers for validation/substepping drivers
#####
##### Each driver builds two models at the same Δt — AnelasticDynamics and
##### CompressibleDynamics(SplitExplicitTimeDiscretization()) — runs both for
##### `stop_iteration` outer steps, times them, writes JLD2 output, and drops
##### a quick side-by-side figure.
#####

using Breeze
using Oceananigans: Oceananigans, interior
using Oceananigans.Units
using Printf
using CairoMakie
using JLD2

"""
    timed_run!(simulation; label)

Run `simulation` with wall-clock timing; return (elapsed seconds, ran_to_completion::Bool).
If an exception fires mid-run, we catch it, log the hypothesis candidates, and return
(elapsed, false) so the caller can keep going.
"""
function timed_run!(simulation; label::String = "run")
    t0 = time()
    ok = true
    err_msg = ""
    try
        Oceananigans.run!(simulation)
    catch e
        ok = false
        err_msg = sprint(showerror, e)
        @warn "[$label] simulation crashed" error = err_msg
    end
    elapsed = time() - t0
    return (; elapsed, ok, err_msg)
end

"""
    summarize_result(label, result, model)

Return a NamedTuple holding the timing and terminal state that we feed into
the markdown report.
"""
function summarize_result(label, result, model)
    u, v, w = model.velocities
    wmax = maximum(abs, interior(w))
    umax = maximum(abs, interior(u))
    nan_w = any(isnan, parent(w))
    return (; label,
              elapsed = result.elapsed,
              ok = result.ok,
              error = result.err_msg,
              iteration = Int(model.clock.iteration),
              time_sim = Float64(model.clock.time),
              wmax = Float64(wmax),
              umax = Float64(umax),
              has_nan = nan_w)
end

"""
    two_column_figure(path, field_a, field_b; title_a, title_b, colormap=:balance)

Shared 1×2 figure that dumps field_a on the left (anelastic) and field_b on the right
(compressible) at the last recorded snapshot. Handles 2D RectilinearGrids out of the box.
"""
function _to_finite_array(f)
    # Strip halos and replace non-finite entries with 0 so CairoMakie doesn't die.
    a = Array(interior(f))
    if any(!isfinite, a)
        a = copy(a)
        a[.!isfinite.(a)] .= 0
    end
    return a
end

function two_column_figure(path, field_a, field_b; title_a = "anelastic", title_b = "compressible",
                           label = "", colormap = :balance, colorrange = nothing)
    fig = Figure(size = (1200, 500))
    ax1 = Axis(fig[1, 1]; title = title_a)
    ax2 = Axis(fig[1, 2]; title = title_b)
    a = _to_finite_array(field_a)
    b = _to_finite_array(field_b)
    a2 = ndims(a) >= 2 ? dropdims(a; dims = tuple(findall(s -> s == 1, size(a))...)) : a
    b2 = ndims(b) >= 2 ? dropdims(b; dims = tuple(findall(s -> s == 1, size(b))...)) : b
    kw = (; colormap)
    cr = if colorrange === nothing
        vmax = max(maximum(abs, a2), maximum(abs, b2))
        vmax = isfinite(vmax) && vmax > 0 ? vmax : 1
        (-vmax, vmax)
    else
        colorrange
    end
    hm1 = heatmap!(ax1, a2; colorrange = cr, kw...)
    heatmap!(ax2, b2; colorrange = cr, kw...)
    Colorbar(fig[1, 3], hm1; label)
    save(path, fig)
    return path
end

"""
    report_case(io, case_name, setup_desc, anelastic_summary, compressible_summary)

Append a `##` section to a markdown IO handle.
"""
function report_case(io, case_name, setup_desc, a, c)
    println(io, "## ", case_name)
    println(io)
    println(io, setup_desc)
    println(io)
    println(io, "| run | elapsed | iters reached | sim time | max|u| | max|w| | NaN? | ok |")
    println(io, "|-----|--------:|--------------:|---------:|-------:|-------:|:-----|:---|")
    for s in (a, c)
        ok_str = s.ok ? "✓" : "✗"
        @printf(io, "| %s | %.2fs | %d | %.1fs | %.3g | %.3g | %s | %s |\n",
                s.label, s.elapsed, s.iteration, s.time_sim, s.umax, s.wmax,
                s.has_nan ? "yes" : "no", ok_str)
    end
    if c.ok && a.ok && a.elapsed > 0
        @printf(io, "\n**Slowdown factor (compressible/anelastic): %.2f×**\n\n", c.elapsed / a.elapsed)
    end
    if !c.ok
        println(io, "\n**Compressible run crashed:** ", c.error, "\n")
    end
    if !a.ok
        println(io, "\n**Anelastic run crashed:** ", a.error, "\n")
    end
    return nothing
end
