# # Verifying AD Through a Breeze Model: Gaussian Diffusion
#
# When we differentiate through a numerical model with automatic
# differentiation (AD), how do we know the gradient is correct?
# For general nonlinear simulations we can only fall back on
# finite-difference checks—but for *linear* dynamics on a *periodic*
# grid we can do much better: every Fourier mode is an exact
# eigenfunction of the discrete operators, so the exact gradient
# of the discrete program can be written in closed form.
#
# This example sets up the simplest such case—isotropic diffusion
# of a Gaussian—and verifies the Enzyme reverse-mode gradient
# against this exact reference to **machine precision**. If the
# AD correctly differentiates through the spatial stencil, the
# time integrator, and ``N_t`` compositions of the time-step map,
# the two must agree to ``\sim\!10^{-14}``.
#
# ## The physics
#
# On a doubly periodic domain ``\Omega = [-L/2,\, L/2]^2`` we solve
#
# ```math
# \frac{\partial T}{\partial t} = \kappa \, \nabla^2 T, \qquad
# T^0(x,y) = A \exp\!\Bigl(-\frac{x^2+y^2}{2\sigma_0^2}\Bigr).
# ```
#
# ## How Breeze solves it
#
# We introduce a density-weighted passive tracer ``\rho c`` in an
# [`AtmosphereModel`](@ref) with [`CompressibleDynamics`](@ref).
# With `advection = nothing` and `closure = ScalarDiffusivity(κ = κ)`,
# the only active tendency is diffusion.  Keeping ``\rho \equiv 1``
# and ``\theta \equiv 300`` K everywhere, every field except ``\rho c``
# remains static and the tracer satisfies
# ``\partial_t(\rho c) = \kappa\,\nabla^2(\rho c)``.
#
# ## Exact gradient of the discrete system
#
# On a uniform periodic grid of ``N \times N`` cells with spacing
# ``\Delta x``, Fourier mode ``(m,n)`` is an exact eigenfunction of the
# centred-difference Laplacian with *modified wavenumber*
#
# ```math
# \tilde{k}^2_{mn}
#   = \frac{4}{\Delta x^2}\!\left[
#       \sin^2\!\Bigl(\frac{\pi m}{N}\Bigr)
#     + \sin^2\!\Bigl(\frac{\pi n}{N}\Bigr)
#   \right].
# ```
#
# The SSP-RK3 time integrator applied to ``\dot{T}=\lambda T``
# is the cubic stability polynomial
# ``g(z) = 1 + z + z^2/2 + z^3/6``,
# so after ``N_t`` steps the Fourier coefficient evolves as
#
# ```math
# \hat{T}^{N_t}_{mn}
#   = G_{mn}\;\hat{T}^{\,0}_{mn}, \qquad
# G_{mn} = g\!\bigl(-\kappa\,\tilde{k}^2_{mn}\,\Delta t\bigr)^{N_t}.
# ```
#
# For the objective ``J = \sum_{ij}(T^{N_t}_{ij})^2``, Parseval's
# identity and the chain rule give the exact discrete gradient
#
# ```math
# \frac{\partial J}{\partial T^0_{ij}}
#   = 2\;\texttt{ifft}\!\bigl(G_{mn}^2 \;\hat{T}^{\,0}_{mn}\bigr)_{ij}.
# ```
#
# No PDE approximation enters.  This is the true gradient of the
# program, computed through an independent (non-AD) route.

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.TurbulenceClosures: ScalarDiffusivity
using CUDA
using Reactant
using Enzyme
using GPUArraysCore: @allowscalar
using FFTW
using Printf

# ## Parameters

A  = 1.0        # amplitude
σ₀ = 20.0       # initial Gaussian width   [m]
κ  = 40.0       # isotropic diffusivity     [m²/s]
L  = 200.0      # domain side length         [m]  (L/σ₀ = 10)
N  = 128        # grid cells per direction
Δt = 0.01       # time step                  [s]
Nₜ = 1          # number of compiled steps
Δx = L / N

# ## Model

grid = RectilinearGrid(ReactantState();
                       size     = (N, N),
                       x        = (-L/2, L/2),
                       y        = (-L/2, L/2),
                       topology = (Periodic, Periodic, Flat))

model = AtmosphereModel(grid;
                        dynamics  = CompressibleDynamics(),
                        advection = nothing,
                        closure   = ScalarDiffusivity(κ = κ),
                        tracers   = :ρc)

# ## Initial condition

T⁰(x, y) = A * exp(-(x^2 + y^2) / (2σ₀^2))

# ## Objective and its Enzyme gradient
#
# We define ``J = \sum_{ij} T_{ij}^2`` and differentiate through
# the full forward model with Enzyme reverse mode, compiled by Reactant.

function objective(model, c⁰, Δt, Nₜ)
    set!(model; ρc = c⁰, θ = 300.0, ρ = 1.0)
    @trace mincut = true checkpointing = true track_numbers = false for _ in 1:Nₜ
        time_step!(model, Δt)
    end
    return sum(interior(model.tracers.ρc) .^ 2)
end

function grad_objective(model, dmodel, c⁰, dc⁰, Δt, Nₜ)
    f(model, c⁰) = objective(model, c⁰, Δt, Nₜ)
    parent(dc⁰) .= 0
    _, Jval = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        f, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(c⁰, dc⁰))
    return dc⁰, Jval
end

# ### Compile and execute

c⁰  = CenterField(grid); set!(c⁰,  T⁰)
dc⁰ = CenterField(grid); set!(dc⁰, 0)
dmodel = Enzyme.make_zero(model)

@info "Compiling grad_objective via Reactant (Nₜ = $Nₜ) …"
compiled_∇J = Reactant.@compile raise = true raise_first = true sync = true grad_objective(
    model, dmodel, c⁰, dc⁰, Δt, Nₜ)

dc, Jᵃᵈ = compiled_∇J(model, dmodel, c⁰, dc⁰, Δt, Nₜ)
∂J_∂T⁰_ad = @allowscalar Array(interior(dc))[:, :, 1]

@info @sprintf("AD:  J = %.8e", Jᵃᵈ)

# ## Continuous analytical gradient
#
# The continuous adjoint of the heat equation gives a Gaussian with
# variance ``\sigma_0^2 + 4\kappa\,t``.  This is the gradient of the
# continuous ``L^2`` objective ``\int T^2\,dx\,dy``; to compare with our
# discrete sum ``\sum T^2`` we scale by the cell area ``\Delta x^2``.
# The error converges as ``O(\Delta x^2)`` under grid refinement.

xᶜ = [(-L/2 + (i - 0.5) * Δx) for i in 1:N]
yᶜ = [(-L/2 + (j - 0.5) * Δx) for j in 1:N]

t      = Nₜ * Δt
σ²_adj = σ₀^2 + 4κ * t

∂J_∂T⁰_cont = [Δx^2 * (2A * σ₀^2 / σ²_adj) * exp(-(x^2 + y^2) / (2σ²_adj))
                for x in xᶜ, y in yᶜ]

ε_cont = maximum(abs, ∂J_∂T⁰_ad .- ∂J_∂T⁰_cont) / maximum(abs, ∂J_∂T⁰_cont)
@info @sprintf("Continuous ∂J/∂T⁰:  rel err = %.2e  (expected ~ Δx²/σ₀² ≈ %.1e)", ε_cont, Δx^2 / σ₀^2)

# ## Exact discrete gradient via Fourier analysis
#
# We now build the same gradient from scratch.  Because every Fourier
# mode is an eigenfunction of both the stencil and the time integrator,
# the discrete propagator is diagonal in Fourier space and the gradient
# follows by the chain rule — no approximation whatsoever.

T⁰_grid = [T⁰(x, y) for x in xᶜ, y in yᶜ]

# Discrete Fourier transform of the initial condition.
T̂⁰ = fft(T⁰_grid)

# Modified wavenumbers of the second-order centred Laplacian.
k̃² = [(4 / Δx^2) * (sin(π * m / N)^2 + sin(π * n / N)^2)
       for m in 0:N-1, n in 0:N-1]

# SSP-RK3 stability polynomial ``g(z) = 1 + z + z²/2 + z³/6``.
g(z) = 1 + z + z^2/2 + z^3/6

# Per-mode amplification factor after ``Nₜ`` steps.
Gₘₙ = [g(-κ * k̃²[m, n] * Δt)^Nₜ for m in 1:N, n in 1:N]

# Exact discrete objective: ``J = N^{-2} \sum_{mn} G_{mn}^2\,|\hat{T}^0_{mn}|^2``.
Jᵈⁱˢᶜ = (1 / N^2) * sum(Gₘₙ .^ 2 .* abs2.(T̂⁰))

# Exact discrete adjoint: ``\partial J/\partial T^0 = 2\,\texttt{ifft}(G^2 \hat{T}^0)``.
∂J_∂T⁰_exact = 2 * real.(ifft(Gₘₙ .^ 2 .* T̂⁰))

# ### Comparison — should agree to ``\sim\!10^{-14}``

ε_abs = maximum(abs, ∂J_∂T⁰_ad .- ∂J_∂T⁰_exact)
ε_rel = ε_abs / maximum(abs, ∂J_∂T⁰_exact)

@info @sprintf("‖∂J/∂T⁰‖_∞:  AD = %.8e,  exact = %.8e", maximum(abs, ∂J_∂T⁰_ad), maximum(abs, ∂J_∂T⁰_exact))
@info @sprintf("J:           AD = %.8e,  exact = %.8e,  rel err = %.2e", Jᵃᵈ, Jᵈⁱˢᶜ, abs(Jᵃᵈ - Jᵈⁱˢᶜ) / abs(Jᵈⁱˢᶜ))
@info @sprintf("∂J/∂T⁰:     max abs err = %.2e,  max rel err = %.2e", ε_abs, ε_rel)

if ε_rel < 1e-10
    @info "AD gradient agrees with exact discrete gradient to near machine precision. ✓"
elseif ε_rel < 1e-4
    @warn "Moderate error — possible roundoff accumulation or wrong stability polynomial."
else
    @error "AD gradient does not match discrete reference.  Likely an AD or compilation bug."
end

nothing #hide
