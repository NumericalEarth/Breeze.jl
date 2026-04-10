# # Advection–Diffusion of a Gaussian: AD Verification
#
# This example verifies reverse-mode automatic differentiation (AD)
# through a Breeze forward model that exercises **both WENO advection
# and isotropic diffusion**.
#
# A Gaussian tracer is advected by a uniform wind ``U_0`` and
# simultaneously diffused with diffusivity ``\kappa``.  Because the
# PDE has a closed-form solution, we can derive *exact* analytical
# gradients of a scalar objective ``J`` with respect to the
# initial-condition parameters ``A``, ``\sigma_0`` and the advection
# velocity ``U_0``.
#
# **Strategy:**  We run AD on a sequence of progressively finer grids.
# The AD gradients are exact for the *discrete* system and must converge
# to the analytical (continuous) values at the order of accuracy of the
# spatial discretisation.

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.TurbulenceClosures: ScalarDiffusivity
using Oceananigans.Grids: xnodes, ynodes
using CUDA
using Reactant
using Reactant: @trace
using GPUArraysCore: @allowscalar
using Enzyme
using CairoMakie
using Printf

# ## 1. The continuous PDE
#
# On a doubly periodic square ``\Omega = [-L/2,\,L/2]^2`` we solve the
# advection–diffusion equation
#
# ```math
# \frac{\partial T}{\partial t}
#   + U_0 \frac{\partial T}{\partial x}
#   = \kappa\,\nabla^2 T,
# \qquad
# T^0(x,y) = A\exp\!\Bigl(-\frac{x^2 + y^2}{2\sigma_0^2}\Bigr).
# ```
#
# The exact solution at time ``t`` is a translating, spreading Gaussian
# whose variance grows linearly with ``s(t) = \sigma_0^2 + 2\kappa t``:
#
# ```math
# T(x,y,t) = A\,\frac{\sigma_0^2}{s(t)}\,
#   \exp\!\Bigl(-\frac{(x - U_0 t)^2 + y^2}{2\,s(t)}\Bigr).
# ```

A   = 2.0       # amplitude
σ₀  = 20.0      # initial width               [m]
κ   = 40.0      # isotropic diffusivity        [m²/s]
U₀  = 10.0      # advection velocity           [m/s]
L   = 200.0     # domain side length           [m]
t_f = 1.0       # integration time             [s]
CFL  = 0.5       # CFL number  (Δt = CFL × Δx / max_speed)
c_s  = sqrt(1.4 * 287.0 * 300.0)  # approximate sound speed [m/s] for θ = 300 K

# With these values ``s_f = \sigma_0^2 + 2\kappa t_f = 480`` m² and
# the Gaussian translates ``U_0 t_f = 10`` m — less than one coarse
# grid cell.  This is enough to exercise WENO nontrivially while
# keeping the pulse well inside the domain.
#
# Because `CompressibleDynamics` with `ExplicitTimeStepping` resolves
# acoustic modes, the time step must satisfy the **acoustic** CFL
# condition ``\Delta t \le \mathrm{CFL}\;\Delta x / c_s`` (with
# ``c_s \approx 347`` m/s for dry air at ``\theta = 300`` K).
# This is far more restrictive than the advective CFL and dominates
# the step-size selection.  The number of steps is
# ``N_s = \lceil t_f / \Delta t \rceil``.

# ## 2. Objective and analytical sensitivities
#
# We measure the squared ``L^2`` norm of the tracer at ``t_f``:
#
# ```math
# J \;=\; \int_\Omega T(x,y,t_f)^2\;\mathrm{d}x\,\mathrm{d}y
#   \;=\; \frac{\pi A^2 \sigma_0^4}{s_f},
# \qquad s_f = \sigma_0^2 + 2\kappa\,t_f.
# ```
#
# Differentiating analytically:
#
# ```math
# \frac{\partial J}{\partial A}
#   = \frac{2J}{A},
# \qquad
# \frac{\partial J}{\partial \sigma_0}
#   = \frac{2\pi A^2 \sigma_0^3\,(\sigma_0^2 + 4\kappa\,t_f)}{s_f^2},
# \qquad
# \frac{\partial J}{\partial U_0} = 0.
# ```
#
# The last identity holds because ``J`` is an ``L^2`` norm and hence
# translation-invariant.  At finite resolution the WENO stencil
# introduces numerical dissipation that depends on the advection
# velocity, so the *discrete* ``\partial J/\partial U_0`` is nonzero but
# should shrink as ``O(\Delta x^p)``.
#
# !!! note "``∂J/∂κ``"
#     The diffusivity ``\kappa`` enters through `ScalarDiffusivity` at
#     model construction time, which lives **outside** the AD tape.
#     Differentiating through it requires placing the closure inside
#     the traced path — a harder test left for future work.
#     Analytically, ``\partial J/\partial\kappa = -2\pi A^2\sigma_0^4\,t_f / s_f^2``.

function analytical_values(A, σ₀, κ, t_f)
    s_f    = σ₀^2 + 2κ * t_f
    J      = π * A^2 * σ₀^4 / s_f
    ∂J_∂A  =  2π * A   * σ₀^4 / s_f
    ∂J_∂σ₀ =  2π * A^2 * σ₀^3 * (σ₀^2 + 4κ * t_f) / s_f^2
    ∂J_∂U₀ =  0.0
    return (; J, ∂J_∂A, ∂J_∂σ₀, ∂J_∂U₀)
end

# ## 3. Mapping onto Breeze
#
# We embed the problem as a density-weighted passive tracer ``\rho c``
# in an [`AtmosphereModel`](@ref) with [`CompressibleDynamics`](@ref).
# With ``\rho \equiv 1``, ``\theta_{\mathrm{air}} \equiv 300`` K,
# ``u \equiv U_0``, ``v = w = 0`` the tracer equation reduces to
# ``\partial_t(\rho c) + U_0\,\partial_x(\rho c) = \kappa\,\nabla^2(\rho c)``.
#
# **What lives outside the AD tape:**  grid construction, model
# allocation, `Enzyme.make_zero`, and the coordinate arrays ``x_c``,
# ``y_c`` (used to build the IC via broadcasts).
#
# **What lives inside the AD tape (i.e. `loss`):**  stamping the
# Gaussian IC from ``\theta``, setting the model state, time-stepping,
# and evaluating ``J``.

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

# ## 4. Loss and gradient functions
#
# We bundle the differentiable parameters into a vector
# ``\theta = [A,\;\sigma_0,\;U_0]`` stored as a `ConcreteRArray` and
# annotated `Duplicated` so that Enzyme accumulates
# ``\nabla_\theta J`` in the shadow ``d\theta``.
#
# Inside `loss` the three parameters are unpacked with `@allowscalar`
# (needed because `θ[i]` on a traced array is a scalar read).
# The Gaussian IC is then built entirely via **broadcasts** on the
# coordinate arrays `xc`, `yc` — no per-element `set!` calls are
# required.
#
# The coordinate arrays and all remaining scalar arguments
# (``\Delta t``, ``N_s``, ``\Delta x``) are passed as `Const`.

function loss(model, T⁰, θ, xc, yc, Δt, Nₛ, Δx)
    A_  = @allowscalar θ[1]
    σ₀_ = @allowscalar θ[2]
    U₀_ = @allowscalar θ[3]

    X = reshape(xc, :, 1)
    Y = reshape(yc, 1, :)
    r² = X .^ 2 .+ Y .^ 2
    T_vals = A_ .* exp.(-r² ./ (2 * σ₀_^2))
    interior(T⁰) .= reshape(T_vals, size(interior(T⁰)))

    set!(model; ρc = T⁰, ρ = 1.0, θ = 300.0, u = U₀_, v = 0.0, w = 0.0)
    @trace track_numbers = false mincut = true checkpointing = false for _ in 1:Nₛ
        time_step!(model, Δt)
    end
    return Δx^2 * sum(interior(model.tracers.ρc) .^ 2)
end

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

# ## 5. Grid-refinement loop
#
# For each resolution we pick ``\Delta t`` from the CFL condition and
# ``N_s = \lceil t_f / \Delta t \rceil``, then adjust ``\Delta t`` so
# that ``N_s \Delta t = t_f`` exactly.  At each resolution we:
# (1) build a fresh model, (2) compile `grad_loss` via
# `Reactant.@compile`, (3) execute the compiled function to obtain
# ``(\nabla_\theta J,\; J)`` in a single AD call, and (4) compare
# against the analytical reference.

N_list = [16, 32, 64, 128]

@info "Advection–diffusion AD verification: N ∈ $N_list"

results = []

for (ℓ, N) in enumerate(N_list)
    Δx  = L / N
    Δt  = CFL * Δx / (c_s + U₀)
    Nₛ  = ceil(Int, t_f / Δt)
    Δt  = t_f / Nₛ                  # adjust so Nₛ × Δt = t_f exactly
    @info "Pass $ℓ/$(length(N_list)):  N=$N, Δx=$(round(Δx; digits=3)) m, " *
          "Δt=$(round(Δt; sigdigits=3)) s, Nₛ=$Nₛ, t_f=$t_f s"

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
    rel_U₀ = abs(∂J_∂U₀_ad) / abs(exact.J / U₀)

    push!(results, (; N, Δx, t_f, J_ad,
                      rel_J, rel_A, rel_σ₀, rel_U₀,
                      ∂J_∂A_ad, ∂J_∂σ₀_ad, ∂J_∂U₀_ad, exact))

    @info "  Rel errors:  J = $(round(rel_J; sigdigits=3)),  " *
          "∂J/∂A = $(round(rel_A; sigdigits=3)),  " *
          "∂J/∂σ₀ = $(round(rel_σ₀; sigdigits=3)),  " *
          "∂J/∂U₀ = $(round(rel_U₀; sigdigits=3))"
end

# ## 6. Convergence plot
#
# All errors should decrease as ``O(\Delta x^2)`` or better where x is the grid spacing.
# Since the analytical ``\partial J/\partial U_0 = 0``, its relative error
# is undefined; instead we normalise by ``J/U_0`` (the natural scale for
# a velocity-derivative of the objective) to obtain a dimensionless
# quantity comparable to the other relative errors.

Δx_vec = [r.Δx for r in results]
safe(v) = max.(v, eps(Float64))

fig = Figure(size = (820, 520), fontsize = 14)
ax  = Axis(fig[1, 1];
    xscale = log2, yscale = log2,
    xlabel = L"\Delta x\;[\mathrm{m}]",
    ylabel = "relative error",
    title  = L"AD gradient convergence ($J = \Delta x^2\Sigma\, T_{ij}^2$)")

err_J_vec  = safe([r.rel_J  for r in results])
err_A_vec  = safe([r.rel_A  for r in results])
err_σ₀_vec = safe([r.rel_σ₀ for r in results])
err_U₀_vec = safe([r.rel_U₀ for r in results])

scatterlines!(ax, Δx_vec, err_J_vec,  marker = :rect,      linewidth = 2,
              label = L"J")
scatterlines!(ax, Δx_vec, err_A_vec,  marker = :utriangle, linewidth = 2,
              label = L"\partial J/\partial A")
scatterlines!(ax, Δx_vec, err_σ₀_vec, marker = :diamond,   linewidth = 2,
              label = L"\partial J/\partial \sigma_0")
scatterlines!(ax, Δx_vec, err_U₀_vec, marker = :star5,     linewidth = 2,
              label = L"|\partial J/\partial U_0|\, /\, (J/U_0)")

ref₂ = err_J_vec[1] .* (Δx_vec ./ Δx_vec[1]) .^ 2
lines!(ax, Δx_vec, ref₂, linestyle = :dash, color = :black,
       label = L"O(\Delta x^2)")

axislegend(ax, position = :rb)

save("advection_diffusion_gradient_convergence.png", fig) #src
fig
