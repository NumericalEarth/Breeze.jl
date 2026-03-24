# # Advection–Diffusion of a Gaussian: AD Verification
#
# We verify reverse-mode AD through a Breeze forward model that exercises
# **both WENO advection and isotropic diffusion**.  A Gaussian tracer is
# advected by a uniform wind ``U_0`` and simultaneously diffused with
# diffusivity ``\kappa``.  The continuous PDE admits a closed-form
# solution, giving us exact analytical gradients of a scalar objective
# ``J`` with respect to the initial-condition parameters ``A``, ``\sigma_0``
# and the advection velocity ``U_0``.
#
# Under grid refinement the AD gradients (exact for the *discrete* system)
# must converge to these analytical values at the order of accuracy of
# the spatial discretisation.

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.TurbulenceClosures: ScalarDiffusivity
using Oceananigans.Grids: xnodes, ynodes
using CUDA
using Reactant
using Reactant: @trace, @allowscalar
using Enzyme
using CairoMakie
using Printf

Reactant.set_default_backend("cpu")
Reactant.allowscalar(true)

# ## 1. The continuous problem
#
# On a doubly periodic square ``\Omega = [-L/2,\,L/2]^2`` we solve
#
# ```math
# \frac{\partial T}{\partial t}
#   + U_0 \frac{\partial T}{\partial x}
#   = \kappa\,\nabla^2 T,
# \qquad
# T^0(x,y) = A\exp\!\left(-\frac{x^2 + y^2}{2\sigma_0^2}\right).
# ```
#
# The exact solution is a translating, spreading Gaussian with
# ``s(t) = \sigma_0^2 + 2\kappa\,t``:
#
# ```math
# T(x,y,t) = A\,\frac{\sigma_0^2}{s(t)}\,
#   \exp\!\left(-\frac{(x - U_0 t)^2 + y^2}{2\,s(t)}\right).
# ```

A   = 2.0       # amplitude
σ₀  = 20.0      # initial width               [m]
κ   = 40.0      # isotropic diffusivity        [m²/s]
U₀  = 10.0      # advection velocity           [m/s]
L   = 200.0     # domain side length           [m]
Nₛ₀ = 100       # time steps at coarsest grid
Δt₀ = 0.01      # time step at coarsest grid   [s]

# At the coarsest resolution (``N = 16``), ``t_f = N_s\Delta t = 1`` s,
# ``s_f = \sigma_0^2 + 2\kappa t_f = 480``, and the Gaussian translates
# ``U_0 t_f = 10`` m — half a grid cell.  That is enough to exercise WENO
# nontrivially while keeping the pulse well inside the domain.

# ## 2. The objective and its analytical sensitivities
#
# We define the squared ``L^2`` norm at ``t_f = N_s\Delta t``:
#
# ```math
# J = \int_\Omega T(x,y,t_f)^2\;\mathrm{d}x\,\mathrm{d}y
#   = \frac{\pi A^2 \sigma_0^4}{s_f},
# \qquad s_f = \sigma_0^2 + 2\kappa\,t_f.
# ```
#
# Differentiating with respect to each parameter (note that both the
# numerator ``\sigma_0^4`` and the denominator ``s_f`` depend on
# ``\sigma_0``):
#
# ```math
# \frac{\partial J}{\partial A}
#   = \frac{2\pi A\,\sigma_0^4}{s_f} = \frac{2J}{A},
# \qquad
# \frac{\partial J}{\partial \sigma_0}
#   = \frac{2\pi A^2 \sigma_0^3(\sigma_0^2 + 4\kappa\,t_f)}{s_f^2},
# \qquad
# \frac{\partial J}{\partial U_0} = 0.
# ```
#
# The last identity holds because the ``L^2`` norm is translation-invariant.
# At finite resolution the WENO stencil introduces numerical dissipation,
# so the discrete ``\partial J/\partial U_0`` is nonzero but should
# decrease as ``O(\Delta x^p)``.  A large or non-convergent
# ``\partial J/\partial U_0`` indicates a bug in the advection adjoint.
#
# !!! note "``\partial J/\partial \kappa``"
#     The diffusivity ``\kappa`` enters through `ScalarDiffusivity` at
#     model construction, which lives **outside** the AD tape.
#     Differentiating ``J`` with respect to ``\kappa`` requires placing
#     the closure inside the traced path — a harder test left for future
#     work.  Analytically,
#     ``\partial J/\partial\kappa = -2\pi A^2\sigma_0^4\,t_f/s_f^2``.

function analytical_values(A, σ₀, κ, t_f)
    s_f    = σ₀^2 + 2κ * t_f
    J      = π * A^2 * σ₀^4 / s_f
    ∂J_∂A  =  2π * A   * σ₀^4 / s_f
    ∂J_∂σ₀ =  2π * A^2 * σ₀^3 * (σ₀^2 + 4κ * t_f) / s_f^2
    ∂J_∂U₀ =  0.0
    return (; J, ∂J_∂A, ∂J_∂σ₀, ∂J_∂U₀)
end

# ## 3. The adjoint field ``\partial J / \partial T^0`` (reference)
#
# For completeness: the continuous adjoint at ``t = 0`` is a Gaussian
# centred at the origin with variance ``\sigma_0^2 + 4\kappa\,t_f``:
#
# ```math
# \frac{\partial J}{\partial T^0}(x,y)
#   = \frac{2A\,\sigma_0^2}{\sigma_0^2 + 4\kappa\,t_f}\,
#     \exp\!\left(-\frac{x^2 + y^2}
#       {2(\sigma_0^2 + 4\kappa\,t_f)}\right).
# ```
#
# We do not verify this field-gradient directly here; instead, we verify
# the scalar sensitivities ``\partial J/\partial\theta`` where
# ``\theta = (A,\,\sigma_0,\,U_0)``.

# ## 4. Parameter vector ``\theta``
#
# We bundle the differentiable parameters into a vector
# ``\theta = [A,\;\sigma_0,\;U_0]`` and pass it as `Duplicated`.
# Inside `loss`, `@allowscalar` unpacks the traced parameters, and
# the initial condition is built via coordinate broadcasts on
# pre-extracted `xc`, `yc` arrays (passed as `Const` RArrays).

# ## 5. Mapping onto Breeze
#
# We embed the problem as a density-weighted passive tracer ``\rho c``
# in an [`AtmosphereModel`](@ref) with [`CompressibleDynamics`](@ref).
# Setting ``\rho \equiv 1``, ``\theta_{\mathrm{air}} \equiv 300`` K,
# ``u \equiv U_0``, ``v = w = 0`` gives
# ``\partial_t(\rho c) + U_0\partial_x(\rho c) = \kappa\nabla^2(\rho c)``.
#
# The grid, model, coordinate arrays, and Enzyme shadows are constructed
# **outside** the differentiated code path.

function build_case(N, L, κ)
    Δx = L / N

    grid = RectilinearGrid(ReactantState();
        size = (N, N), x = (-L/2, L/2), y = (-L/2, L/2),
        topology = (Periodic, Periodic, Flat))

    model = AtmosphereModel(grid;
        dynamics  = CompressibleDynamics(),
        advection = WENO(order = 5),
        closure   = ScalarDiffusivity(κ = Float64(κ)),
        tracers   = :ρc)

    T⁰  = CenterField(grid)
    dT⁰ = CenterField(grid)
    set!(dT⁰, 0.0)
    dmodel = Enzyme.make_zero(model)

    xc = Reactant.to_rarray(Array(xnodes(grid, Center())))
    yc = Reactant.to_rarray(Array(ynodes(grid, Center())))

    return model, dmodel, T⁰, dT⁰, xc, yc, Δx
end

# ## 6. The loss function
#
# `loss` unpacks ``A = \theta_1``, ``\sigma_0 = \theta_2``,
# ``U_0 = \theta_3`` via `@allowscalar`, constructs the Gaussian IC
# through coordinate broadcasts (identical pattern to the acoustic-wave
# AD example), sets the model state, integrates, and returns
#
# ```math
# J_{\mathrm{disc}} = \Delta x^2 \sum_{ij} T_{ij}(t_f)^2.
# ```

function loss(model, T⁰, θ, xc, yc, Δt, Nₛ, Δx)
    A_  = @allowscalar θ[1]
    σ₀_ = @allowscalar θ[2]
    U₀_ = @allowscalar θ[3]

    X = reshape(xc, :, 1)
    Y = reshape(yc, 1, :)
    r² = X .^ 2 .+ Y .^ 2
    T_vals = A_ .* exp.(-r² ./ (2 * σ₀_^2))
    interior(T⁰) .= reshape(T_vals, size(interior(T⁰)))

    set!(model; ρc = T⁰, ρ = 1.0, θ = 300.0, u = U₀_, v = 0.0, w = 0.0,
               enforce_mass_conservation = false)
    @trace track_numbers = false mincut = true checkpointing = false for _ in 1:Nₛ
        time_step!(model, Δt)
    end
    return Δx^2 * sum(interior(model.tracers.ρc) .^ 2)
end

# ## 7. The gradient wrapper
#
# ``\theta`` is `Duplicated`; the shadow `d\theta` receives
# ``(\partial J/\partial A,\;\partial J/\partial\sigma_0,\;
# \partial J/\partial U_0)``.  Coordinate arrays are `Const`.

function grad_loss(model, dmodel, T⁰, dT⁰, θ, dθ, xc, yc, Δt, Nₛ, Δx)
    parent(dT⁰) .= 0
    dθ .= 0
    _, J = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(T⁰,   dT⁰),
        Enzyme.Duplicated(θ, dθ),
        Enzyme.Const(xc),
        Enzyme.Const(yc),
        Enzyme.Const(Δt),
        Enzyme.Const(Nₛ),
        Enzyme.Const(Δx))
    return dθ, J
end

# ## 8. Grid-refinement sweep
#
# We hold the final time ``t_f`` constant across resolutions by scaling
# ``\Delta t \propto \Delta x`` and adjusting ``N_s`` accordingly.
# At each resolution we build, compile, execute, and compare against
# the analytical sensitivities.

N_list = [16, 32, 64]

@info "Advection–diffusion AD verification: N ∈ $N_list"

results = []

for (ℓ, N) in enumerate(N_list)
    Δx  = L / N
    Δt  = Δt₀ * (Δx / (L / N_list[1]))
    Nₛ  = round(Int, (Nₛ₀ * Δt₀) / Δt)
    t_f = Nₛ * Δt
    @info "Pass $ℓ/$(length(N_list)):  N=$N, Δx=$(round(Δx; digits=3)) m, " *
          "Δt=$(round(Δt; sigdigits=3)) s, Nₛ=$Nₛ, t_f=$(round(t_f; digits=4)) s"

    @info "  Building grid, model, and shadow objects …"
    model, dmodel, T⁰, dT⁰, xc, yc, Δx = build_case(N, L, κ)

    θ  = Reactant.to_rarray(Float64[A, σ₀, U₀])
    dθ = Reactant.to_rarray(zeros(3))

    @info "  Compiling grad_loss …"
    compiled = Reactant.@compile raise_first = true raise = true sync = true grad_loss(
        model, dmodel, T⁰, dT⁰, θ, dθ, xc, yc, Δt, Nₛ, Δx)

    @info "  Running compiled grad_loss …"
    dθ_result, J_ad = compiled(model, dmodel, T⁰, dT⁰, θ, dθ, xc, yc, Δt, Nₛ, Δx)

    J_ad      = Float64(J_ad)
    grads     = Array(dθ_result)
    ∂J_∂A_ad  = grads[1]
    ∂J_∂σ₀_ad = grads[2]
    ∂J_∂U₀_ad = grads[3]

    exact = analytical_values(A, σ₀, κ, t_f)

    rel_J  = abs(J_ad      - exact.J)      / abs(exact.J)
    rel_A  = abs(∂J_∂A_ad  - exact.∂J_∂A)  / abs(exact.∂J_∂A)
    rel_σ₀ = abs(∂J_∂σ₀_ad - exact.∂J_∂σ₀) / abs(exact.∂J_∂σ₀)
    abs_U₀ = abs(∂J_∂U₀_ad)

    push!(results, (; N, Δx, t_f, J_ad,
                      rel_J, rel_A, rel_σ₀, abs_U₀,
                      ∂J_∂A_ad, ∂J_∂σ₀_ad, ∂J_∂U₀_ad, exact))

    @info "  Rel errors:  J = $(round(rel_J; sigdigits=3)),  " *
          "∂J/∂A = $(round(rel_A; sigdigits=3)),  " *
          "∂J/∂σ₀ = $(round(rel_σ₀; sigdigits=3)),  " *
          "|∂J/∂U₀| = $(round(abs_U₀; sigdigits=3))"
end

# ## 9. Convergence plot
#
# All errors should decrease as ``O(\Delta x^2)`` or better.
# ``|\partial J/\partial U_0|`` is plotted as an absolute value
# (its analytical target is zero).

Δx_vec = [r.Δx for r in results]
safe(v) = max.(v, eps(Float64))

fig = Figure(size = (820, 520), fontsize = 14)
ax  = Axis(fig[1, 1];
    xscale = log10, yscale = log10,
    xlabel = L"\Delta x\;[\mathrm{m}]",
    ylabel = "relative error",
    title  = L"AD gradient convergence ($J = \Delta x^2\Sigma\, T_{ij}^2$)")

err_J_vec  = safe([r.rel_J  for r in results])
err_A_vec  = safe([r.rel_A  for r in results])
err_σ₀_vec = safe([r.rel_σ₀ for r in results])
err_U₀_vec = safe([r.abs_U₀ for r in results])

scatterlines!(ax, Δx_vec, err_J_vec,  marker = :rect,      linewidth = 2,
              label = L"J")
scatterlines!(ax, Δx_vec, err_A_vec,  marker = :utriangle, linewidth = 2,
              label = L"\partial J/\partial A")
scatterlines!(ax, Δx_vec, err_σ₀_vec, marker = :diamond,   linewidth = 2,
              label = L"\partial J/\partial \sigma_0")
scatterlines!(ax, Δx_vec, err_U₀_vec, marker = :star5,     linewidth = 2,
              label = L"|\partial J/\partial U_0|")

ref₂ = err_J_vec[1] .* (Δx_vec ./ Δx_vec[1]) .^ 2
lines!(ax, Δx_vec, ref₂, linestyle = :dash, color = :black,
       label = L"O(\Delta x^2)")

axislegend(ax, position = :rb)

outfile = "advection_diffusion_gradient_convergence.png"
save(outfile, fig)
@info "Saved convergence plot → $outfile"

nothing #hide
