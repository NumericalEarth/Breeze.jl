#####
##### The scalar second moments θˡⁱ′², qᵗ′² and θˡⁱ′qᵗ′ from their steady budget with vertical transport
#####
##### The local diagnosis of the distribution, X = ℂᵛ τ Kᶜ ∂z a ∂z b, is the steady budget without transport,
##### X / τᵈ = 2 ℂᵛ Kᶜ ∂z a ∂z b with the dissipation time τᵈ = τ / 2 (`DissipationTimeScale`: τ = 2 τᵠ). With
##### the transport, at each interior face,
#####
#####     X / τᵈ - ∂z (Kᵛ ∂z X) + ∂z T = 2 ℂᵛ Kᶜ ∂z a ∂z b,    Kᵛ = ℂᵛᵗ Kᶜ,
#####
##### that is X - τᵈ ∂z (Kᵛ ∂z X) = X_local - τᵈ ∂z T, with no flux through the bottom and top faces. The three
##### moments share the matrix: one Thomas elimination per column, three right-hand sides. T is the
##### third moment w′a′b′, or zero.
#####
##### The matrix has a positive diagonal, non-positive off-diagonals and is strictly diagonally dominant
##### by rows: it is an M-matrix, its inverse G is non-negative, so θˡⁱ′² and qᵗ′² are non-negative where
##### their productions are (T = 0). The local moments have correlation ±1 (θˡⁱ′qᵗ′² = θˡⁱ′² qᵗ′²); with
##### G ≥ 0, Cauchy–Schwarz on the sums Σₖ Gⱼₖ sₖ ∂θₖ ∂qₖ gives (θˡⁱ′qᵗ′)² ≤ θˡⁱ′² qᵗ′² after the solve,
##### with the correlation below 1 where gradients of different ratio mix. With T ≠ 0 neither holds by
##### construction.
#####

"""
$(TYPEDEF)

The vertical transport of the scalar second moments `θˡⁱ′²`, `qᵗ′²` and `θˡⁱ′qᵗ′` of a
[`DoubleGaussian`](@ref) (`variance_transport`). At each interior face the moments solve the
steady budget

    X / τᵈ - ∂z (ℂᵛᵗ Kᶜ ∂z X) + ∂z T = X_local / τᵈ,    τᵈ = τ / 2,

with `X_local` the down-gradient diagnosis, `τ` the moment time scale and no flux of `X` through
the bottom and top faces; one tridiagonal elimination per column serves the three moments.
`T` is zero, or with `component_third_moment = true` the third moment `w′a′b′` of the components
of the double-Gaussian PDF, whose means carry the fluxes,

    w′a′b′ = w′³ w′a′ w′b′ / ((1 - ℂᵍ(Sk)) w′²)²,

evaluated at the centers with the third moment `w′³` of the local moments (one fixed-point pass).
It transports variance against the gradient where the skewness and the fluxes are positive.

$(TYPEDFIELDS)
"""
struct ScalarVarianceTransport{FT, TM}
    "Ratio of the diffusivity of the scalar second moments to the closure's `Kᶜ`"
    ℂᵛᵗ :: FT
    "`Val(true)` for the third moment of the PDF components, `Val(false)` for none"
    component_third_moment :: TM
end

Adapt.@adapt_structure ScalarVarianceTransport

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`ScalarVarianceTransport`](@ref).
"""
function ScalarVarianceTransport(FT::DataType = Oceananigans.defaults.FloatType; ℂᵛᵗ = 1, component_third_moment = false)
    ℂᵛᵗ ≥ 0 || throw(ArgumentError("ℂᵛᵗ must be non-negative, got $ℂᵛᵗ"))
    return ScalarVarianceTransport(convert(FT, ℂᵛᵗ), Val(component_third_moment))
end

convert_eltype(::Type{FT}, t::ScalarVarianceTransport) where FT = ScalarVarianceTransport(convert(FT, t.ℂᵛᵗ), t.component_third_moment)

scalar_variance_transport_fields(grid, ::Nothing) = NamedTuple()
scalar_variance_transport_fields(grid, ::ScalarVarianceTransport) = (variance_transport_work = ZFaceField(grid),)

# The third moments w′θˡⁱ′², w′qᵗ′², w′θˡⁱ′qᵗ′ at the center k
@inline component_scalar_third_moments(::Val{false}, d, m, cl, i, j, k) = (zero(eltype(cl.w′²)), zero(eltype(cl.w′²)), zero(eltype(cl.w′²)))

@inline function component_scalar_third_moments(::Val{true}, d, m, cl, i, j, k)
    @inbounds begin
        w′² = cl.w′²[i, j, k]
        w′³ = cl.w′³[i, j, k]
        Sk = w′³ / (w′² * sqrt(w′²))
        w′²ᵇ = (1 - within_component_share(d.assumed_pdf, Sk)) * w′²
        r = w′³ / (w′²ᵇ * w′²ᵇ)
        w′θ = (m.w′θˡⁱ′[i, j, k] + m.w′θˡⁱ′[i, j, k+1]) / 2
        w′q = (m.w′qᵗ′[i, j, k] + m.w′qᵗ′[i, j, k+1]) / 2
    end
    return r * w′θ * w′θ, r * w′q * w′q, r * w′θ * w′q
end

# The third moment w′³ at the centers from the local moments, for the third-moment flux of the solve
@inline local_center_third_moments!(::Val{false}, args...) = nothing

@inline local_center_third_moments!(::Val{true}, m, cl, i, j, grid, d, closure, closure_fields, c³, θ, e) =
    third_moment_profile!(m, cl, i, j, grid, d, closure, closure_fields, c³, θ, e)

@inline transport_scalar_variances!(::Nothing, m, cl, i, j, grid, d, closure, closure_fields, c³, θ, e) = nothing

@inline function transport_scalar_variances!(t::ScalarVarianceTransport, m, cl, i, j, grid, d, closure, closure_fields, c³, θ, e)
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    local_center_third_moments!(t.component_third_moment, m, cl, i, j, grid, d, closure, closure_fields, c³, θ, e)
    work = m.variance_transport_work
    Kᶜ = closure_fields.Kᶜ
    @inbounds begin
        # Thomas algorithm over the faces 2 … Nz; the rows are diagonally dominant
        c′ = zero(FT)
        dθ = zero(FT)
        dq = zero(FT)
        dθq = zero(FT)
        a⁻ = zero(FT) # Kᵛ / Δz at the center below the face (zero below face 2: no flux through the bottom)
        Tθ⁻ = zero(FT)
        Tq⁻ = zero(FT)
        Tθq⁻ = zero(FT)
        for k in 2:Nz
            a = zero(FT)
            Tθ = zero(FT)
            Tq = zero(FT)
            Tθq = zero(FT)
            if k < Nz # the center k, between the faces k and k + 1; none above face Nz: no flux through the top
                a = t.ℂᵛᵗ * (Kᶜ[i, j, k] + Kᶜ[i, j, k+1]) / 2 / Δzᶜᶜᶜ(i, j, k, grid)
                Tθ, Tq, Tθq = component_scalar_third_moments(t.component_third_moment, d, m, cl, i, j, k)
            end
            r = m.τ[i, j, k] / 2 / Δzᶜᶜᶠ(i, j, k, grid)
            lower = -r * a⁻
            upper = -r * a
            pivot = 1 + r * (a + a⁻) - lower * c′
            dθ = (m.θˡⁱ′²[i, j, k] - r * (Tθ - Tθ⁻) - lower * dθ) / pivot
            dq = (m.qᵗ′²[i, j, k] - r * (Tq - Tq⁻) - lower * dq) / pivot
            dθq = (m.θˡⁱ′qᵗ′[i, j, k] - r * (Tθq - Tθq⁻) - lower * dθq) / pivot
            c′ = upper / pivot
            work[i, j, k] = c′
            m.θˡⁱ′²[i, j, k] = dθ
            m.qᵗ′²[i, j, k] = dq
            m.θˡⁱ′qᵗ′[i, j, k] = dθq
            a⁻ = a
            Tθ⁻ = Tθ
            Tq⁻ = Tq
            Tθq⁻ = Tθq
        end
        for k in Nz-1:-1:2
            m.θˡⁱ′²[i, j, k] -= work[i, j, k] * m.θˡⁱ′²[i, j, k+1]
            m.qᵗ′²[i, j, k] -= work[i, j, k] * m.qᵗ′²[i, j, k+1]
            m.θˡⁱ′qᵗ′[i, j, k] -= work[i, j, k] * m.θˡⁱ′qᵗ′[i, j, k+1]
        end
    end
    return nothing
end
