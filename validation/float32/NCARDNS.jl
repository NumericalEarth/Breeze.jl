"""
    NCARDNS

Tools to compare Breeze with the NCAR parcel-DNS code (Chen, Xue & Yau 2020):

- readers for the NCAR outputs (`.para`, `.eng`, `.spc`, `.dsd`, `.nc`, `Zk*.ncf`, `drop*.ncf`);
- an importer that evaluates the NCAR spectral fields exactly at the staggered
  locations of an Oceananigans C-grid with any resolution `M ≥ N`;
- diagnostics (KE, u′, ε, λ, Re_λ, kmax η, E(k), velocity-gradient skewness) computed with
  one code path for both codes, from Fourier coefficients of the velocity field.

Conventions of the NCAR code (see `parcel/DNS/{main,initturbdim,fftw3,diagdim,outputting}.F90`):

- Grid spacing `h = π η / 1.3` with `η = (ν³/ε)^(1/4)`, `ν = 1.6e-5`, `ε = 0.05`; domain `L = N h`.
  Real-space grid points are at `x = (j-1) h`, `j = 1, …, N`.
- Spectral arrays have dimensions `(KX, KY, KZ) = (N/2+1, N, N)` (Fortran order; ncdump shows
  `(T, RI, KZ, KY, KX)`), stored as separate real and imaginary parts along `RI`.
  `KX` holds `kx = 0, …, N/2`; `KY` and `KZ` are in FFT order `0, 1, …, N/2-1, [N/2 → unused], -N/2+1, …, -1`.
  Physical wavenumber `k = dimk × integer`, `dimk = 2π/L`.
- The forward FFT is normalized by `1/N³` (`fftwrk`), the backward FFT is unnormalized, so the stored
  coefficients are Fourier-series coefficients: `f(x) = Σ_k f̂_k exp(i k⋅x)`, with
  `⟨f²⟩ = Σ_{all k} |f̂_k|²`.
- Only the half space selected by the mask `L` is retained (kx > 0, or kx = 0 and ky > 0, or kx = ky = 0
  and kz > 0), truncated spherically at `|k| ≲ N/2` dimk; the other half follows from `f̂_{-k} = conj(f̂_k)`.
- Stored fields: vorticity `ZXK, ZYK, ZZK` (ω = ∇ × u), temperature perturbation `TTK` (T′), and vapor
  perturbation `QVK` (qv′). Velocity: `û = i k × ω̂ / |k|²` (`vandr.F90:VELO`).
- Diagnostics: `.spc` shell `j = INT(|k|/dimk + 0.5)` holds `Σ_{half space, shell j} |ω̂|²/k²`
  (= KE in the shell, so that `Σ_j E_j = ½⟨|u|²⟩`), the enstrophy `Σ|ω̂|²` (= ½⟨|ω|²⟩), `Σ|T̂|²`, `Σ|q̂|²`,
  and twice the number of retained modes. `.para` columns: time, u′ = √(2KE/3), ε = 2ν Σ_{j<N/2} Ω_j, λ,
  Re_λ, eddy time, kmax η, Σ|ω̂|², Σ_{j<N/2} Ω_j, λ (Vincent & Meneguzzi), Re_λ (idem), √(2 KE).
  `.para` and `.spc` totals exclude the last shell `j = N/2`; `.eng` (time, KE, PE, VVK, V, H, S2, KKOL)
  sums all retained modes.
"""
module NCARDNS

using FFTW
using NCDatasets
using Statistics
using Printf
using DelimitedFiles

export NCARParameters, ncar_parameters,
       read_para, read_eng, read_spc, read_dsd, read_nc, read_zk, read_drop, read_real_space,
       ZkField, retained_mask, full_coefficients, velocity_coefficients,
       evaluate_on_grid, staggered_velocities, centered_scalar,
       FourierVelocity, fourier_velocity_from_arrays, fourier_velocity_from_zk,
       turbulence_statistics, shell_spectrum, finite_difference_statistics,
       IncrementStatistics, accumulate_increments!, injection_power,
       write_csv,
       gradient_moments, finite_difference_gradient_moments, staggering_offsets, collocated,
       truncated, field_difference, difference_spectrum, save_fourier_velocity, load_fourier_velocity,
       scalar_statistics

#####
##### Parameters
#####

const ν_ncar = 1.6e-5
const ε_target_ncar = 0.05
const κ_temperature_ncar = 2.22e-5
const κ_vapor_ncar = 2.55e-5

struct NCARParameters
    N :: Int
    viscosity :: Float64
    target_dissipation :: Float64
    kolmogorov_length :: Float64
    spacing :: Float64
    domain_length :: Float64
    dimk :: Float64
end

"""
    ncar_parameters(N; ν=1.6e-5, ε=0.05, box_N=N)

Return the geometry of an `N³` NCAR DNS: `h = π η / 1.3`, `L = N h`, `dimk = 2π / L`.
`ε` is converted to single precision, as in the Fortran (`edr` is `real`).
`box_N` is the `BOXN` build option of the resolution study (`~/ncar-dns/runs/R`): a fixed box
`L = box_N π η / 1.3` resolved with `N³` points, i.e. `h = (π η / 1.3) box_N / N`.
"""
function ncar_parameters(N; ν=ν_ncar, ε=ε_target_ncar, box_N=N)
    ε = Float64(Float32(ε))
    η = (ν^3 / ε)^(1/4)
    h = 3.1415926535 / 1.3 * η * (box_N / N)  # exactly the original h when box_N = N
    L = N * h
    return NCARParameters(N, ν, ε, η, h, L, 2π / L)
end

Base.show(io::IO, p::NCARParameters) =
    print(io, @sprintf("NCARParameters(N=%d, ν=%.3g, ε=%.3g, η=%.4g m, h=%.4g m, L=%.5g m, dimk=%.5g /m)",
                       p.N, p.viscosity, p.target_dissipation, p.kolmogorov_length, p.spacing,
                       p.domain_length, p.dimk))

#####
##### Text readers
#####

parse_float(s) = parse(Float64, replace(strip(s), r"[dD]" => "e"))

function numeric_rows(path)
    rows = Vector{Vector{Float64}}()
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        push!(rows, parse_float.(split(s)))
    end
    return rows
end

const para_columns = (:time, :rms_velocity, :dissipation, :taylor_microscale, :taylor_reynolds,
                      :eddy_time, :kmax_eta, :enstrophy_all, :enstrophy, :taylor_microscale_vm,
                      :taylor_reynolds_vm, :sqrt_2ke)

"""
    read_para(path)

Read a `.para` file (one row per output: $(join(para_columns, ", "))). Returns a `NamedTuple` of vectors.
"""
function read_para(path)
    rows = numeric_rows(path)
    return NamedTuple{para_columns}(Tuple([r[i] for r in rows] for i in 1:length(para_columns)))
end

const eng_columns = (:time, :kinetic_energy, :temperature_variance, :vvk, :enstrophy, :helicity,
                     :vapor_variance, :kolmogorov_index)

# Fortran format 5044: 1x,F8.3,2x,3(e10.3,2x),e10.3,2x,f8.3,2x,e10.3,2x,f4.1
const eng_widths = [(2, 9), (12, 21), (24, 33), (36, 45), (48, 57), (60, 67), (70, 79), (82, 85)]

"""
    read_eng(path)

Read an `.eng` file (columns $(join(eng_columns, ", "))). KE = ½⟨|u|²⟩ and the enstrophy is
½⟨|ω|²⟩, both summed over all retained modes. Values have three significant digits.
"""
function read_eng(path)
    rows = Vector{Vector{Float64}}()
    for line in eachline(path)
        isempty(strip(line)) && continue
        parts = split(strip(line))
        if length(parts) == length(eng_columns)
            push!(rows, parse_float.(parts))
        else # fixed-width fallback for fields that run together
            push!(rows, [parse_float(line[a:min(b, end)]) for (a, b) in eng_widths])
        end
    end
    return NamedTuple{eng_columns}(Tuple([r[i] for r in rows] for i in 1:length(eng_columns)))
end

"""
    read_spc(path)

Read a `.spc` file. Returns a vector of `NamedTuple`s, one per output time, with fields
`time`, `k` (shell index j), `energy` (KE in shell), `enstrophy`, `temperature`, `vapor`, `modes`
(the Fortran count, which is twice the number of retained half-space modes in the shell).
"""
function read_spc(path)
    blocks = []
    time = NaN
    cols = [Float64[] for _ in 1:6]
    function flush!()
        if !isempty(cols[1])
            push!(blocks, (; time, k=copy(cols[1]), energy=copy(cols[2]), enstrophy=copy(cols[3]),
                             temperature=copy(cols[4]), vapor=copy(cols[5]), modes=Int.(cols[6])))
        end
        foreach(empty!, cols)
    end
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        if occursin("TIME", s)
            flush!()
            time = parse_float(first(split(s)))
        else
            vals = parse_float.(split(s))
            for i in 1:6
                push!(cols[i], vals[i])
            end
        end
    end
    flush!()
    return blocks
end

"""
    read_dsd(path)

Read a `.dsd` (or `.dsdlog`) file: one row per output with the time and the droplet counts in
100 radius bins (bin `i` counts droplets with `nint(r / 1 µm) = i`; radii below 0.5 µm go to bin 1).
Returns `(; time, counts)` with `counts[i, n]`.
"""
function read_dsd(path)
    rows = numeric_rows(path)
    isempty(rows) && return (; time=Float64[], counts=zeros(Int, 100, 0))
    time = [r[1] for r in rows]
    counts = reduce(hcat, [Int.(r[2:end]) for r in rows])
    return (; time, counts)
end

const nc_columns = (:time, :supersaturation, :pressure, :temperature, :potential_temperature,
                    :vapor_mixing_ratio, :liquid_water_content, :saturation_mixing_ratio, :deltaqp, :air_density)

"""
    read_nc(path)

Read the parcel time series `.nc` (a text file, not netCDF): $(join(nc_columns, ", ")).
"""
function read_nc(path)
    rows = numeric_rows(path)
    return NamedTuple{nc_columns}(Tuple([r[i] for r in rows] for i in 1:length(nc_columns)))
end

#####
##### Spectral fields (Zk*.ncf)
#####

"""
    ZkField

Spectral fields read from a `Zk*.ncf` file, as complex arrays of size `(N/2+1, N, N)` in the NCAR
layout, together with the time, the geometry, and the retained-mode mask.
"""
struct ZkField
    time :: Float64
    parameters :: NCARParameters
    ωx :: Array{ComplexF64, 3}
    ωy :: Array{ComplexF64, 3}
    ωz :: Array{ComplexF64, 3}
    T :: Union{Nothing, Array{ComplexF64, 3}}
    qv :: Union{Nothing, Array{ComplexF64, 3}}
    mask :: BitArray{3}
end

Base.show(io::IO, z::ZkField) = print(io, "ZkField(t=", z.time, ", N=", z.parameters.N, ", ",
                                       count(z.mask), " retained modes)")

function read_complex(ds, name)
    haskey(ds, name) || return nothing
    a = Array{Float32}(ds[name][:, :, :, :, 1])
    return complex.(Float64.(a[:, :, :, 1]), Float64.(a[:, :, :, 2]))
end

"""
    read_zk(path; ν=1.6e-5, ε=0.05, box_N=nothing)

Read a `Zk*.ncf` file. `N` is inferred from the `KY` dimension. `box_N` (default `N`) is the
`BOXN` of a fixed-box build of the resolution study (`L = box_N π η / 1.3`).
"""
function read_zk(path; ν=ν_ncar, ε=ε_target_ncar, box_N=nothing)
    NCDataset(path) do ds
        N = ds.dim["KY"]
        @assert ds.dim["KX"] == N ÷ 2 + 1
        box_N = something(box_N, N)
        time = Float64(ds["TIMES"][1])
        p = ncar_parameters(N; ν, ε, box_N)
        ωx = read_complex(ds, "ZXK")
        ωy = read_complex(ds, "ZYK")
        ωz = read_complex(ds, "ZZK")
        T = read_complex(ds, "TTK")
        qv = read_complex(ds, "QVK")
        return ZkField(time, p, ωx, ωy, ωz, T, qv, retained_mask(N; box_N))
    end
end

"Integer wavenumber of NCAR index `i` along a full (`KY` or `KZ`) dimension; the N/2 slot maps to 0."
function ncar_wavenumber(i, N)
    kt = N ÷ 2
    j = i - 1
    i > kt && (j -= 2kt)
    i == kt + 1 && (j = 0)
    return j
end

"""
    retained_mask(N; box_N=N)

The NCAR mask `L` (`initturbdim.F90`): retained half-space modes inside the spherical truncation.
The truncation test is evaluated in single precision, as in the Fortran.
"""
function retained_mask(N; box_N=N)
    p = ncar_parameters(N; box_N)
    dimk = Float32(2 * 3.1415926535 / (N * p.spacing))
    ktrunc = Float32(N / 2)
    threshold = Float32(floor(Int, dimk * ktrunc + 0.5f0)) - 0.5f0
    mask = falses(N ÷ 2 + 1, N, N)
    for ikz in 1:N, iky in 1:N, ikx in 1:N÷2+1
        kx = ikx - 1
        ky = ncar_wavenumber(iky, N)
        kz = ncar_wavenumber(ikz, N)
        keep = true
        iky == N ÷ 2 + 1 && (keep = false)
        ikz == N ÷ 2 + 1 && (keep = false)
        kx == 0 && ky < 0 && (keep = false)
        kx == 0 && ky == 0 && kz < 0 && (keep = false)
        wk = sqrt(Float32(kx * dimk)^2 + Float32(ky * dimk)^2 + Float32(kz * dimk)^2)
        (wk == 0 || wk > threshold) && (keep = false)
        mask[ikx, iky, ikz] = keep
    end
    return mask
end

"""
    retained_modes(z::ZkField)

Return a vector of `(kx, ky, kz)` integer wavenumbers and the linear indices of the retained modes.
"""
function retained_modes(mask)
    N = size(mask, 2)
    modes = Tuple{Int, Int, Int}[]
    indices = Int[]
    for I in CartesianIndices(mask)
        mask[I] || continue
        ikx, iky, ikz = Tuple(I)
        push!(modes, (ikx - 1, ncar_wavenumber(iky, N), ncar_wavenumber(ikz, N)))
        push!(indices, LinearIndices(mask)[I])
    end
    return modes, indices
end

"""
    velocity_coefficients(z::ZkField)

Return the half-space velocity coefficients `(modes, û, v̂, ŵ)` with `û = i k × ω̂ / |k|²`
(in physical wavenumbers), for the retained modes only.
"""
function velocity_coefficients(z::ZkField)
    modes, indices = retained_modes(z.mask)
    dimk = z.parameters.dimk
    n = length(modes)
    û = zeros(ComplexF64, n); v̂ = similar(û); ŵ = similar(û)
    for (m, (mode, I)) in enumerate(zip(modes, indices))
        kx, ky, kz = dimk .* mode
        k² = kx^2 + ky^2 + kz^2
        ωx, ωy, ωz = z.ωx[I], z.ωy[I], z.ωz[I]
        û[m] = im * (ky * ωz - kz * ωy) / k²
        v̂[m] = im * (kz * ωx - kx * ωz) / k²
        ŵ[m] = im * (kx * ωy - ky * ωx) / k²
    end
    return modes, û, v̂, ŵ
end

scalar_coefficients(z::ZkField, f) = f[retained_modes(z.mask)[2]]

#####
##### Evaluation on (staggered) grids
#####

"""
    evaluate_on_grid(modes, coefficients, M, offset)

Evaluate the real field `f(x) = Σ_{half space} [f̂_k e^{i k⋅x} + c.c.]` exactly at the points
`x_i = (i - 1 + offset_i) Δ`, `i = 1, …, M`, of a uniform `M³` grid with `Δ = L / M`
(so that `offset = 0` is a face and `offset = 1/2` a cell center), by zero-padding the
coefficients into an `M³` real-to-complex array with phase shifts and an inverse FFT.
`modes` holds integer wavenumbers (units of 2π/L); all must satisfy `|k_i| < M/2`.
"""
function evaluate_on_grid(modes, coefficients, M, offset)
    C = zeros(ComplexF64, M ÷ 2 + 1, M, M)
    ox, oy, oz = offset
    for ((kx, ky, kz), c) in zip(modes, coefficients)
        @assert max(abs(kx), abs(ky), abs(kz)) < M ÷ 2 "mode $((kx, ky, kz)) does not fit on a $M³ grid"
        # phase of the offset: exp(i 2π k⋅offset / M)
        phase = cispi(2 * (kx * ox + ky * oy + kz * oz) / M)
        put!(C, kx, ky, kz, c * phase, M)
        if kx == 0 # the conjugate mode -k is also stored in the kx = 0 plane
            put!(C, 0, -ky, -kz, conj(c * phase), M)
        end
    end
    return brfft(C, M)  # unnormalized: Σ_k C_k e^{+ikx}, Hermitian completion implicit
end

@inline wrap(k, M) = mod(k, M) + 1

function put!(C, kx, ky, kz, value, M)
    C[kx + 1, wrap(ky, M), wrap(kz, M)] += value
    return nothing
end

"""
    staggered_velocities(z::ZkField, M)

Return `(u, v, w)` evaluated exactly at the C-grid locations of a uniform `M³` Oceananigans grid
with `x ∈ [0, L)`: `u` at (Face, Center, Center), `v` at (Center, Face, Center), `w` at (Center, Center, Face).
"""
function staggered_velocities(z::ZkField, M)
    modes, û, v̂, ŵ = velocity_coefficients(z)
    u = evaluate_on_grid(modes, û, M, (0, 1/2, 1/2))
    v = evaluate_on_grid(modes, v̂, M, (1/2, 0, 1/2))
    w = evaluate_on_grid(modes, ŵ, M, (1/2, 1/2, 0))
    return u, v, w
end

"""
    centered_scalar(z::ZkField, name, M)

Return the scalar `name ∈ (:T, :qv)` evaluated at the cell centers of a uniform `M³` grid.
"""
function centered_scalar(z::ZkField, name, M)
    f = getfield(z, name)
    modes, indices = retained_modes(z.mask)
    return evaluate_on_grid(modes, f[indices], M, (1/2, 1/2, 1/2))
end

#####
##### Fourier representation used by the common diagnostics
#####

"""
    FourierVelocity

Fourier-series coefficients of a velocity field on a `Mx × My × Mz` periodic grid, in the rfft layout
(`(M÷2+1, M, M)`), normalized so that `u(x) = Σ_k û_k e^{i k⋅x}` (sum over the full spectrum),
together with the physical wavenumber unit `dk = 2π/L` and the viscosity.
Phase offsets of staggered grids are irrelevant to all diagnostics computed here.
"""
struct FourierVelocity
    û :: Array{ComplexF64, 3}
    v̂ :: Array{ComplexF64, 3}
    ŵ :: Array{ComplexF64, 3}
    M :: Int
    dk :: Float64
    viscosity :: Float64
end

"""
    fourier_velocity_from_arrays(u, v, w, L; ν=1.6e-5)

Fourier coefficients of real-space velocity arrays (each `M³`) on a periodic cube of side `L`.
"""
function fourier_velocity_from_arrays(u, v, w, L; ν=ν_ncar)
    M = size(u, 1)
    @assert size(u) == size(v) == size(w) == (M, M, M)
    n = M^3
    return FourierVelocity(rfft(Float64.(u)) ./ n, rfft(Float64.(v)) ./ n, rfft(Float64.(w)) ./ n,
                           M, 2π / L, ν)
end

"""
    fourier_velocity_from_zk(z::ZkField)

Velocity coefficients of an NCAR spectral field, placed in the `N³` rfft layout.
"""
function fourier_velocity_from_zk(z::ZkField)
    N = z.parameters.N
    modes, û, v̂, ŵ = velocity_coefficients(z)
    arrays = map((û, v̂, ŵ)) do c
        C = zeros(ComplexF64, N ÷ 2 + 1, N, N)
        for ((kx, ky, kz), cₖ) in zip(modes, c)
            put!(C, kx, ky, kz, cₖ, N)
            kx == 0 && put!(C, 0, -ky, -kz, conj(cₖ), N)
        end
        C
    end
    return FourierVelocity(arrays..., N, z.parameters.dimk, z.parameters.viscosity)
end

"Signed integer wavenumber of index `i` along a full FFT dimension of length `M`."
@inline signed_wavenumber(i, M) = i - 1 <= M ÷ 2 ? i - 1 : i - 1 - M

"Weight of an rfft-layout mode in a sum over the full spectrum."
@inline rfft_weight(ikx, M) = (ikx == 1 || (iseven(M) && ikx == M ÷ 2 + 1)) ? 1 : 2

"""
    shell_spectrum(F::FourierVelocity; kmax_index=nothing)

Shell-summed spectra with the NCAR binning `j = floor(|k|/dk + 0.5)`: returns
`(; k, energy, dissipation, modes)` for shells `j = 1, …`, where `energy[j] = ½ Σ_{shell} |û|²`
(so that `Σ_j energy[j] = KE = ½⟨|u|²⟩`, as in NCAR's `.spc`) and `dissipation[j] = 2ν k_j² energy`-like
exact shell sums `ν Σ_{shell} |k|² |û|²`.
"""
function shell_spectrum(F::FourierVelocity)
    M = F.M
    jmax = floor(Int, sqrt(3) * M / 2 + 0.5) + 1
    E = zeros(jmax); D = zeros(jmax); Z = zeros(jmax); n = zeros(Int, jmax)
    for ikz in 1:M, iky in 1:M, ikx in 1:M÷2+1
        kx = ikx - 1
        ky = signed_wavenumber(iky, M)
        kz = signed_wavenumber(ikz, M)
        k² = kx^2 + ky^2 + kz^2
        k² == 0 && continue
        j = floor(Int, sqrt(k²) + 0.5)
        w = rfft_weight(ikx, M)
        e = w * (abs2(F.û[ikx, iky, ikz]) + abs2(F.v̂[ikx, iky, ikz]) + abs2(F.ŵ[ikx, iky, ikz]))
        E[j] += e / 2
        D[j] += F.viscosity * k² * F.dk^2 * e
        n[j] += w
    end
    last = findlast(>(0), n)
    return (; k=collect(1:last), energy=E[1:last], dissipation=D[1:last], modes=n[1:last])
end

"""
    spectral_gradient(F::FourierVelocity, component, direction; pad=3//2)

The derivative `∂_direction u_component` evaluated on a uniform grid of size `pad × M`
(padding avoids aliasing of the cubic moments in the skewness).
"""
function spectral_gradient(F::FourierVelocity, c, d; pad=3//2)
    M = F.M
    P = Int(ceil(pad * M / 2) * 2)
    Ĉ = (F.û, F.v̂, F.ŵ)[c]
    G = zeros(ComplexF64, P ÷ 2 + 1, P, P)
    for ikz in 1:M, iky in 1:M, ikx in 1:M÷2+1
        kx = ikx - 1
        ky = signed_wavenumber(iky, M)
        kz = signed_wavenumber(ikz, M)
        k = (kx, ky, kz)[d]
        # Drop Nyquist modes (|kᵢ| = M/2 in any direction), whose sign is ambiguous;
        # NCAR fields have none, and they carry negligible energy in resolved Breeze fields.
        iseven(M) && M ÷ 2 ∈ (kx, abs(ky), abs(kz)) && continue
        G[kx + 1, mod(ky, P) + 1, mod(kz, P) + 1] += im * k * F.dk * Ĉ[ikx, iky, ikz]
        # brfft assumes Hermitian symmetry in the kx = 0 plane, which the source already has.
    end
    return brfft(G, P)
end

"""
    turbulence_statistics(F::FourierVelocity; skewness=true)

Statistics computed identically for both codes from the velocity Fourier coefficients:

- `kinetic_energy` KE = ½⟨|u|²⟩, `rms_velocity` u′ = √(2KE/3)
- `dissipation` ε = 2ν⟨SᵢⱼSᵢⱼ⟩ = ν Σ_k |k|²|û_k|² (exact spectral derivatives; equal to ν⟨|ω|²⟩)
- `taylor_microscale` λ = √(15 ν u′² / ε), `taylor_reynolds` Re_λ = u′λ/ν
- `kolmogorov_length` η = (ν³/ε)^(1/4), `kmax_eta` = (M/2) dk η
- `skewness` S = mean over i of ⟨(∂ᵢuᵢ)³⟩/⟨(∂ᵢuᵢ)²⟩^(3/2) (spectral derivatives, 3/2-padded grid)
- `integral_scale` L_int = 2π Σ_j E_j / (j dk) / KE and `eddy_time` T_eddy = L_int / √(2 KE), with NCAR's
  definitions (`diagdim.F90`; NCAR sums shells 1…N/2−1, here all shells, which differs by < 1e-4 for NCAR fields)
"""
function turbulence_statistics(F::FourierVelocity; skewness=true)
    spectrum = shell_spectrum(F)
    KE = sum(spectrum.energy)
    ε = sum(spectrum.dissipation)
    ν = F.viscosity
    u′ = sqrt(2KE / 3)
    λ = sqrt(15ν * u′^2 / ε)
    Reλ = u′ * λ / ν
    η = (ν^3 / ε)^(1/4)
    kmax_eta = F.M / 2 * F.dk * η
    integral_scale = 2π * sum(spectrum.energy ./ (spectrum.k .* F.dk)) / KE
    eddy_time = integral_scale / sqrt(2KE)
    S = NaN
    if skewness
        S = mean(1:3) do i
            g = spectral_gradient(F, i, i)
            mean(g .^ 3) / mean(g .^ 2)^(3/2)
        end
    end
    return (; kinetic_energy=KE, rms_velocity=u′, dissipation=ε, taylor_microscale=λ,
              taylor_reynolds=Reλ, kolmogorov_length=η, kmax_eta, skewness=S, integral_scale, eddy_time)
end

#####
##### Velocity increments and forcing power
#####

"""
    IncrementStatistics(shifts, separations; edges=range(-12, 12, length=193))

Accumulator for longitudinal velocity increments δu(r) = u(x + r e₁) − u(x) (and likewise v along y,
w along z) of C-grid velocity arrays, pooled over the three components and over snapshots.
`shifts[n]` is the separation in grid cells of the arrays that will be accumulated, `separations[n]`
its physical value. For each separation the accumulator holds the number of samples, the sums of
δu², δu³, δu⁴, and a histogram of δu/σ(r) with bin `edges`, where σ(r) is the standard deviation
of the snapshot (so the PDF is normalized snapshot by snapshot, identically for both codes).
"""
struct IncrementStatistics
    shifts :: Vector{Int}
    separations :: Vector{Float64}
    edges :: Vector{Float64}
    samples :: Vector{Float64}
    moments :: Matrix{Float64}   # (separation, power 2:4)
    counts :: Matrix{Float64}    # (bin, separation)
    snapshots :: Base.RefValue{Int}
end

function IncrementStatistics(shifts, separations; edges=range(-12, 12, length=193))
    n = length(shifts)
    return IncrementStatistics(collect(shifts), collect(Float64, separations), collect(Float64, edges),
                               zeros(n), zeros(n, 3), zeros(length(edges) - 1, n), Ref(0))
end

"""
    accumulate_increments!(stats::IncrementStatistics, u, v, w)

Add the longitudinal increments of one snapshot of periodic C-grid arrays `u, v, w` to `stats`.
"""
function accumulate_increments!(stats::IncrementStatistics, u, v, w)
    edges = stats.edges
    nbins = length(edges) - 1
    Δb = edges[2] - edges[1]
    for (n, s) in enumerate(stats.shifts)
        δ = vcat(vec(circshift(u, (-s, 0, 0)) .- u), vec(circshift(v, (0, -s, 0)) .- v),
                 vec(circshift(w, (0, 0, -s)) .- w))
        σ = sqrt(mean(abs2, δ))
        stats.samples[n] += length(δ)
        for p in 2:4
            stats.moments[n, p - 1] += sum(x -> x^p, δ)
        end
        for x in δ
            b = floor(Int, (x / σ - edges[1]) / Δb) + 1
            1 ≤ b ≤ nbins && (stats.counts[b, n] += 1)
        end
    end
    stats.snapshots[] += 1
    return stats
end

"""
    injection_power(F::FourierVelocity, wavevectors, amplitudes, L)

Power ⟨u ⋅ f⟩ of a band forcing `f = 2 Re Σₘ Aₘ exp(i kₘ ⋅ x)` (`amplitudes[m, c]`, `wavevectors[m]`
integer half-space wavevectors, evaluated at the C-grid location of each component as in Breeze's band
forcings) on the velocity `F`, computed from the C-grid arrays with `fourier_velocity_from_arrays`.
The index-based rfft coefficients are shifted to physical positions (u at x faces, etc.).
"""
function injection_power(F::FourierVelocity, wavevectors, amplitudes, L)
    M = F.M
    Δ = L / M
    offsets = ((0, Δ / 2, Δ / 2), (Δ / 2, 0, Δ / 2), (Δ / 2, Δ / 2, 0))
    coefficients = (F.û, F.v̂, F.ŵ)
    P = 0.0
    for c in 1:3, (m, n) in enumerate(wavevectors)
        k = 2π .* n ./ L
        ũ = coefficients[c][n[1] + 1, mod(n[2], M) + 1, mod(n[3], M) + 1]
        û = ũ * cis(-sum(k .* offsets[c]))
        P += 2 * real(conj(û) * amplitudes[m, c])
    end
    return P
end

"""
    finite_difference_statistics(u, v, w, L; ν=1.6e-5)

Second-order finite-difference diagnostics of staggered C-grid velocity arrays (`u` at faces in x,
etc.): ε_FD = ν ⟨Σᵢⱼ (δⱼuᵢ/Δ)²⟩ (equal to 2ν⟨SᵢⱼSᵢⱼ⟩ for a discretely divergence-free periodic field),
the FD velocity-gradient skewness of δᵢuᵢ/Δ at cell centers, and the maximum discrete divergence.
"""
function finite_difference_statistics(u, v, w, L; ν=ν_ncar)
    M = size(u, 1)
    Δ = L / M
    # forward difference from a face to a center (or center to face): δf[i] = f[i+1] - f[i]
    δ(f, d) = (circshift(f, ntuple(n -> n == d ? -1 : 0, 3)) .- f) ./ Δ
    G = ((δ(u, 1), δ(u, 2), δ(u, 3)), (δ(v, 1), δ(v, 2), δ(v, 3)), (δ(w, 1), δ(w, 2), δ(w, 3)))
    ε = ν * sum(mean(g .^ 2) for row in G for g in row)
    S = mean(mean(G[i][i] .^ 3) / mean(G[i][i] .^ 2)^(3/2) for i in 1:3)
    divergence = G[1][1] .+ G[2][2] .+ G[3][3]
    gradient_scale = sqrt(mean(G[1][1] .^ 2))
    return (; dissipation=ε, skewness=S, max_divergence=maximum(abs, divergence),
              rms_divergence=sqrt(mean(divergence .^ 2)), rms_gradient=gradient_scale,
              kinetic_energy=(mean(u .^ 2) + mean(v .^ 2) + mean(w .^ 2)) / 2)
end

#####
##### Droplets
#####

"""
    read_drop(path; npe)

Read a `drop*.ncf` droplet file written by `npe` MPI ranks. Returns a `NamedTuple` with the parcel state
(`time, pressure, air_density, potential_temperature, supersaturation, vapor_mixing_ratio`) and per-droplet
arrays (`id, x, y, z, radius, dry_radius, dr3, u, v, w`) for the valid droplets only.

The file holds `npe` consecutive per-rank blocks of `NDROPPE/npe` slots; in each block only the first
`ndropreal` slots are droplets (stale copies follow). The last slot of each block stores `ndropreal`
in `IDP` and the second-to-last stores `nddone` (`outputting.F90:ncdumpdrop`). Droplet times are
microphysics times (from 0), unlike the flow time of the Zk files.
"""
function read_drop(path; npe)
    NCDataset(path) do ds
        get1(name) = haskey(ds, name) ? Float64(ds[name][1]) : NaN
        getv(name) = haskey(ds, name) ? vec(Array(ds[name][:, 1])) : nothing
        idp = getv("IDP")
        nb = length(idp) ÷ npe
        valid = Int[]
        for r in 0:npe-1
            nreal = round(Int, idp[r * nb + nb])
            append!(valid, r * nb .+ (1:nreal))
        end
        sel(a) = a === nothing ? nothing : Float64.(a[valid])
        return (; time=get1("TIMES"), pressure=get1("PP"), air_density=get1("rhoa"),
                  potential_temperature=get1("thetapp"), supersaturation=get1("sp"),
                  vapor_mixing_ratio=get1("qvpp"),
                  id=round.(Int, idp[valid]), x=sel(getv("XDROP")), y=sel(getv("YDROP")), z=sel(getv("ZDROP")),
                  radius=sel(getv("R")), dry_radius=sel(getv("R_CCN")), dr3=sel(getv("DR3")),
                  u=sel(getv("UDROP")), v=sel(getv("VDROP")), w=sel(getv("WDROP")))
    end
end

#####
##### Real-space fields (out.ncf, written when rsflag = 1)
#####

"""
    read_real_space(path; variables=("U", "V", "W"))

Read the real-space fields of `out.ncf`. NCAR stores real-space arrays with dimensions `(X, Z, Y, T)`;
the arrays returned here are permuted to `(x, y, z)`, one per output time, with the grid points
at `x = (i-1) h`. Returns `(; times, fields)` where `fields[name][n]` is the `n`-th snapshot.
"""
function read_real_space(path; variables=("U", "V", "W"))
    NCDataset(path) do ds
        times = Float64.(ds["TIMES"][:])
        fields = Dict(name => [permutedims(Float64.(ds[name][:, :, :, n]), (1, 3, 2)) for n in eachindex(times)]
                      for name in variables if haskey(ds, name))
        return (; times, fields)
    end
end

#####
##### Output
#####

"""
    write_csv(path, table::NamedTuple)

Write a `NamedTuple` of equal-length vectors as a CSV file with a header.
"""
function write_csv(path, table::NamedTuple)
    open(path, "w") do io
        println(io, join(string.(keys(table)), ","))
        n = length(first(values(table)))
        for i in 1:n
            println(io, join((string(c[i]) for c in values(table)), ","))
        end
    end
    return path
end

#####
##### Resolution study: velocity-gradient moments, pointwise differences, snapshots
#####

"Sums of g², g³, g⁴ over an array (one pass, Float64 accumulators)."
function power_sums(g)
    s2 = s3 = s4 = 0.0
    @inbounds @simd for x in g
        x2 = Float64(x)^2
        s2 += x2
        s3 += x2 * x
        s4 += x2 * x2
    end
    n = length(g)
    return s2 / n, s3 / n, s4 / n
end

"""
    gradient_moments(F::FourierVelocity; pad=2)

Moments of the velocity-gradient tensor with exact (spectral) derivatives of the field represented by `F`,
evaluated on a `(pad M)³` grid. With `pad = 2` (default) the means of products of up to four
band-limited fields are free of aliasing (Nyquist modes are dropped). Returns

- `skewness` = mean over i of ⟨(∂ᵢuᵢ)³⟩/⟨(∂ᵢuᵢ)²⟩^(3/2) (longitudinal; negative), and its components;
- `flatness` = mean over i of ⟨(∂ᵢuᵢ)⁴⟩/⟨(∂ᵢuᵢ)²⟩² (longitudinal), and its components;
- `transverse_flatness` = mean over i ≠ j of ⟨(∂ⱼuᵢ)⁴⟩/⟨(∂ⱼuᵢ)²⟩²;
- `longitudinal_variance` and `transverse_variance` (mean ⟨(∂ᵢuᵢ)²⟩ and mean ⟨(∂ⱼuᵢ)²⟩, i ≠ j; isotropy
  gives transverse/longitudinal = 2).
"""
function gradient_moments(F::FourierVelocity; pad=2)
    Sc = zeros(3); Fc = zeros(3); FTc = Float64[]
    m2L = 0.0; m2T = 0.0
    for i in 1:3
        m2, m3, m4 = power_sums(spectral_gradient(F, i, i; pad))
        Sc[i] = m3 / m2^1.5
        Fc[i] = m4 / m2^2
        m2L += m2 / 3
    end
    for i in 1:3, j in 1:3
        i == j && continue
        m2, _, m4 = power_sums(spectral_gradient(F, i, j; pad))
        push!(FTc, m4 / m2^2)
        m2T += m2 / 6
    end
    return (; skewness=mean(Sc), flatness=mean(Fc), transverse_flatness=mean(FTc),
              skewness_components=Sc, flatness_components=Fc,
              longitudinal_variance=m2L, transverse_variance=m2T)
end

"""
    finite_difference_gradient_moments(u, v, w, L)

The same moments as [`gradient_moments`](@ref) for Breeze's own second-order differences of the C-grid
arrays: longitudinal δᵢuᵢ/Δ at cell centers, transverse δⱼuᵢ/Δ at cell edges.
"""
function finite_difference_gradient_moments(u, v, w, L)
    M = size(u, 1)
    Δ = L / M
    δ(f, d) = (circshift(f, ntuple(n -> n == d ? -1 : 0, 3)) .- f) ./ Δ
    U = (u, v, w)
    Sc = zeros(3); Fc = zeros(3); FTc = Float64[]
    m2L = 0.0; m2T = 0.0
    for i in 1:3
        m2, m3, m4 = power_sums(δ(U[i], i))
        Sc[i] = m3 / m2^1.5; Fc[i] = m4 / m2^2; m2L += m2 / 3
    end
    for i in 1:3, j in 1:3
        i == j && continue
        m2, _, m4 = power_sums(δ(U[i], j))
        push!(FTc, m4 / m2^2); m2T += m2 / 6
    end
    return (; skewness=mean(Sc), flatness=mean(Fc), transverse_flatness=mean(FTc),
              skewness_components=Sc, flatness_components=Fc,
              longitudinal_variance=m2L, transverse_variance=m2T)
end

"Physical positions of the first u, v, w points of a C-grid with spacing Δ (relative to the origin x = 0)."
staggering_offsets(Δ) = ((0, Δ / 2, Δ / 2), (Δ / 2, 0, Δ / 2), (Δ / 2, Δ / 2, 0))

"""
    collocated(F::FourierVelocity, offsets)

Fourier coefficients relative to the physical origin of arrays whose first points sit at `offsets[c]`
(component c): `û_k = c̃_k exp(−i k⋅x₀)`. For Breeze C-grid arrays use `staggering_offsets(L / M)`;
NCAR coefficients (`fourier_velocity_from_zk`) are already relative to the origin. Needed only for
pointwise comparisons; all statistics are phase independent.
"""
function collocated(F::FourierVelocity, offsets)
    M = F.M
    out = map(1:3) do c
        Ĉ = copy((F.û, F.v̂, F.ŵ)[c])
        x₀ = offsets[c]
        for ikz in 1:M, iky in 1:M, ikx in 1:M÷2+1
            k = F.dk .* (ikx - 1, signed_wavenumber(iky, M), signed_wavenumber(ikz, M))
            Ĉ[ikx, iky, ikz] *= cis(-sum(k .* x₀))
        end
        Ĉ
    end
    return FourierVelocity(out..., M, F.dk, F.viscosity)
end

"""
    truncated(F::FourierVelocity, Mt)

Embed `F` in an `Mt³` rfft layout: modes with all |kᵢ| < min(M, Mt)/2 are copied (Nyquist modes dropped),
so `Mt > M` zero-pads and `Mt < M` truncates.
"""
function truncated(F::FourierVelocity, Mt)
    M = F.M
    K = min(M, Mt) ÷ 2
    out = map((F.û, F.v̂, F.ŵ)) do Ĉ
        T = zeros(ComplexF64, Mt ÷ 2 + 1, Mt, Mt)
        for ikz in 1:M, iky in 1:M, ikx in 1:M÷2+1
            kx = ikx - 1; ky = signed_wavenumber(iky, M); kz = signed_wavenumber(ikz, M)
            (kx < K && abs(ky) < K && abs(kz) < K) || continue
            T[kx + 1, mod(ky, Mt) + 1, mod(kz, Mt) + 1] = Ĉ[ikx, iky, ikz]
        end
        T
    end
    return FourierVelocity(out..., Mt, F.dk, F.viscosity)
end

"""
    field_difference(A::FourierVelocity, B::FourierVelocity)

Pointwise difference of two (collocated) velocity fields in the same box, B being the reference:
relative L² errors of the velocity, `‖u_A − u_B‖/‖u_B‖`, and of the velocity gradient (k²-weighted),
and the correlation coefficient of the gradient fields. Modes present in only one field count fully.
"""
function field_difference(A::FourierVelocity, B::FourierVelocity)
    M = max(A.M, B.M)
    a = truncated(A, M); b = truncated(B, M)
    e0 = n0 = e2 = n2 = cAB = nA = 0.0
    for ikz in 1:M, iky in 1:M, ikx in 1:M÷2+1
        k² = (ikx - 1)^2 + signed_wavenumber(iky, M)^2 + signed_wavenumber(ikz, M)^2
        w = rfft_weight(ikx, M)
        for (x, y) in ((a.û, b.û), (a.v̂, b.v̂), (a.ŵ, b.ŵ))
            d = abs2(x[ikx, iky, ikz] - y[ikx, iky, ikz]); r = abs2(y[ikx, iky, ikz])
            e0 += w * d; n0 += w * r; e2 += w * k² * d; n2 += w * k² * r
            cAB += w * k² * real(x[ikx, iky, ikz] * conj(y[ikx, iky, ikz]))
            nA += w * k² * abs2(x[ikx, iky, ikz])
        end
    end
    return (; velocity_error=sqrt(e0 / n0), gradient_error=sqrt(e2 / n2), gradient_correlation=cAB / sqrt(nA * n2))
end

"""
    difference_spectrum(A::FourierVelocity, B::FourierVelocity)

Shell spectrum (NCAR binning) of ½|û_A − û_B|² for two collocated fields; divided by the shell spectrum of B it
is the fraction of the energy in each shell that differs (0: identical, 2: uncorrelated with equal energy).
"""
function difference_spectrum(A::FourierVelocity, B::FourierVelocity)
    M = max(A.M, B.M)
    a = truncated(A, M); b = truncated(B, M)
    D = FourierVelocity(a.û .- b.û, a.v̂ .- b.v̂, a.ŵ .- b.ŵ, M, A.dk, A.viscosity)
    return shell_spectrum(D)
end

"""
    save_fourier_velocity(path, F::FourierVelocity, time; Mt=min(F.M, 128))

Write the coefficients of `F` truncated to `Mt` (single precision) to a netCDF file (for pointwise comparisons).
"""
function save_fourier_velocity(path, F::FourierVelocity, time; Mt=min(F.M, 128))
    T = truncated(F, Mt)
    isfile(path) && rm(path)
    NCDataset(path, "c") do ds
        ds.attrib["M_original"] = F.M; ds.attrib["dk"] = F.dk; ds.attrib["viscosity"] = F.viscosity
        ds.attrib["time"] = time
        defDim(ds, "KX", Mt ÷ 2 + 1); defDim(ds, "KY", Mt); defDim(ds, "KZ", Mt); defDim(ds, "RI", 2)
        for (name, C) in (("U", T.û), ("V", T.v̂), ("W", T.ŵ))
            defVar(ds, name, cat(Float32.(real(C)), Float32.(imag(C)); dims=4), ("KX", "KY", "KZ", "RI"))
        end
    end
    return path
end

function load_fourier_velocity(path)
    NCDataset(path) do ds
        Mt = ds.dim["KY"]
        C = [complex.(Float64.(ds[n][:, :, :, 1]), Float64.(ds[n][:, :, :, 2])) for n in ("U", "V", "W")]
        return FourierVelocity(C..., Mt, ds.attrib["dk"], ds.attrib["viscosity"]), ds.attrib["time"]
    end
end

"""
    scalar_statistics(f, L, κ; pad=2)

Statistics of a periodic scalar array `f` (cell centers, `M³`, box `L`) computed spectrally: variance,
dissipation χ = 2κ⟨|∇f|²⟩ (exact derivatives), shell spectrum of ½|f̂|², and the skewness and flatness of
∂f/∂x (pad-ed grid), averaged over directions; plus the second-order finite-difference χ_FD.
"""
function scalar_statistics(f, L, κ; pad=2)
    M = size(f, 1)
    f̂ = rfft(Float64.(f)) ./ M^3
    dk = 2π / L
    Z = zeros(ComplexF64, size(f̂))
    G = FourierVelocity(f̂, Z, Z, M, dk, κ)
    spectrum = shell_spectrum(G)                    # energy = ½ Σ|f̂|² per shell
    variance = 2 * sum(spectrum.energy)              # mean removed (k = 0 excluded)
    χ = 2 * sum(spectrum.dissipation)                # 2κ Σ k²|f̂|²
    Sd = Float64[]; Fd = Float64[]
    for d in 1:3
        m2, m3, m4 = power_sums(spectral_gradient(G, 1, d; pad))
        push!(Sd, m3 / m2^1.5); push!(Fd, m4 / m2^2)
    end
    Δ = L / M
    χ_FD = 2κ * sum(mean(((circshift(f, ntuple(n -> n == d ? -1 : 0, 3)) .- f) ./ Δ) .^ 2) for d in 1:3)
    return (; variance, dissipation=χ, dissipation_fd=χ_FD, gradient_skewness=mean(Sd), gradient_flatness=mean(Fd),
              spectrum=(k=spectrum.k, variance=2 .* spectrum.energy))
end

end # module
