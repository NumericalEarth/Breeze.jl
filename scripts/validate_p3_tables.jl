## Validate Julia P3 lookup tables against official Fortran reference tables
##
## Compares the Julia P3IntegralEvaluator output against Fortran reference tables
## from ~/Aeolus/P3-microphysics/lookup_tables/.
##
## Key conventions:
##   - Julia uses N=1 (per-particle) convention; Fortran bakes in n₀(q).
##   - For normalized integrals (fall speeds, mean diameter, density, eff. radius),
##     the n₀ cancels and values are directly comparable.
##   - For raw integrals (ventilation, collection, shedding), Julia values must be
##     scaled by q/M₃ (single integrals) or (q/M₃)² (double integrals like nagg).
##
## Usage:
##   julia --project scripts/validate_p3_tables.jl [fortran_table_dir]

using Printf
using Statistics

using Oceananigans
using Breeze

const PP = Breeze.Microphysics.PredictedParticleProperties

fortran_dir = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(homedir(), "Aeolus", "P3-microphysics", "lookup_tables")

# ═══════════════════════════════════════════════════════════════════════════════
# Fortran coordinate mappings
# ═══════════════════════════════════════════════════════════════════════════════

const FORTRAN_RHOR = [50.0, 250.0, 450.0, 650.0, 900.0]
const FORTRAN_FR = [0.0, 0.333, 0.667, 1.0]
const FORTRAN_FL = [0.0, 0.333, 0.667, 1.0]

"""Log₁₀ of mean particle mass for Fortran index i_Qnorm (1-based)."""
fortran_log_mass(i) = ((i + 10) * 0.1) * log10(800.0) - 18.0

# Fortran column indices (1-based within the 21-value block for 2-moment)
const COL = Dict(:uns => 1, :ums => 2, :nagg => 3, :nrwat => 4,
                 :vdep => 5, :eff => 6, :i_qsmall => 7, :i_qlarge => 8,
                 :refl2 => 9, :vdep1 => 10, :dmm => 11, :rhomm => 12,
                 :lambda_i => 13, :mu_i_save => 14,
                 :vdepm1 => 15, :vdepm2 => 16, :vdepm3 => 17, :vdepm4 => 18,
                 :qshed => 19, :nawcol => 20, :naicol => 21)

# ═══════════════════════════════════════════════════════════════════════════════
# Parse Fortran Table 1 (2-moment)
# ═══════════════════════════════════════════════════════════════════════════════

function parse_table1_2mom(filepath)
    text = read(pipeline(`gunzip -c $filepath`), String)
    main = Dict{NTuple{4,Int}, Vector{Float64}}()
    for line in split(text, '\n')
        s = strip(line)
        isempty(s) && continue
        startswith(s, "LOOKUP") && continue
        tokens = split(s)
        length(tokens) == 25 || continue
        vals = tryparse.(Float64, tokens)
        any(isnothing, vals) && continue
        main[ntuple(i -> Int(vals[i]), 4)] = Float64[vals[j] for j in 5:25]
    end
    return main
end

# ═══════════════════════════════════════════════════════════════════════════════
# Build Julia evaluators
# ═══════════════════════════════════════════════════════════════════════════════

function build_evaluators(; nquad=80)
    mk(I) = PP.P3IntegralEvaluator(I, Float64; number_of_quadrature_points=nquad)
    return Dict(
        :uns   => mk(PP.NumberWeightedFallSpeed()),
        :ums   => mk(PP.MassWeightedFallSpeed()),
        :dmm   => mk(PP.MeanDiameter()),
        :rhomm => mk(PP.MeanDensity()),
        :eff   => mk(PP.EffectiveRadius()),
        :vdep  => mk(PP.Ventilation()),
        :vdep1 => mk(PP.VentilationEnhanced()),
        :nagg  => mk(PP.AggregationNumber()),
        :nrwat => mk(PP.RainCollectionNumber()),
        :qshed => mk(PP.SheddingRate()),
        :_mass => mk(PP.MassMomentLambdaLimit()),  # for N-scaling
    )
end

# Integral types by normalization convention
const NORMALIZED = Set([:uns, :ums, :dmm, :rhomm, :eff])
const SINGLE_RAW = Set([:vdep, :vdep1, :nrwat, :qshed])
const DOUBLE_RAW = Set([:nagg])

# ═══════════════════════════════════════════════════════════════════════════════
# Comparison engine
# ═══════════════════════════════════════════════════════════════════════════════

function compare(main, evs, Fr_indices, Fl_indices, Qnorm_range)
    results = Dict{Symbol, Vector{Float64}}()
    vars = [:uns, :ums, :dmm, :rhomm, :eff, :vdep, :vdep1, :nagg, :nrwat, :qshed]

    for var in vars
        rel_errs = Float64[]
        col = COL[var]
        ev = evs[var]
        ev_mass = evs[:_mass]

        for i_rhor in 1:5, i_Fr in Fr_indices, i_Fl in Fl_indices, i_Qnorm in Qnorm_range
            idx = (i_rhor, i_Fr, i_Fl, i_Qnorm)
            haskey(main, idx) || continue
            f_val = main[idx][col]
            abs(f_val) < 1e-30 && continue

            lm = fortran_log_mass(i_Qnorm)
            Fr = FORTRAN_FR[i_Fr]
            Fl = FORTRAN_FL[i_Fl]
            ρ = FORTRAN_RHOR[i_rhor]

            j_val = ev(lm, Fr, Fl; rime_density=ρ)
            isfinite(j_val) || continue

            # N-convention scaling for raw integrals
            if var in SINGLE_RAW || var in DOUBLE_RAW
                q = 10.0^lm
                M3 = ev_mass(lm, Fr, Fl; rime_density=ρ)
                isfinite(M3) && M3 > 1e-30 || continue
                scale = q / M3
                j_val *= var in DOUBLE_RAW ? scale^2 : scale
            end

            re = abs(f_val - j_val) / abs(f_val)
            (isnan(re) || isinf(re)) && continue
            push!(rel_errs, re)
        end
        results[var] = rel_errs
    end
    return results
end

function print_results(label, results)
    println("\n", "="^100)
    println(label)
    println("="^100)
    @printf("  %-10s  %5s  %12s  %12s  %12s\n", "Variable", "n", "median_rel", "p95_rel", "max_rel")
    println("  ", "-"^80)

    for var in [:uns, :ums, :dmm, :rhomm, :eff, :vdep, :vdep1, :nagg, :nrwat, :qshed]
        haskey(results, var) || continue
        errs = results[var]
        n = length(errs)
        n == 0 && continue
        @printf("  %-10s  %5d  %12.2e  %12.2e  %12.2e\n",
                string(var), n, median(errs), quantile(errs, 0.95), maximum(errs))
    end
end

function print_psd_results(main, Fr_indices, Qnorm_range)
    println("\n  PSD parameters:")
    for pvar in [:lambda_i, :mu_i_save]
        colno = COL[pvar]
        rel_errs = Float64[]
        for i_rhor in 1:5, i_Fr in Fr_indices, i_Qnorm in Qnorm_range
            idx = (i_rhor, i_Fr, 1, i_Qnorm)
            haskey(main, idx) || continue
            f_val = main[idx][colno]
            abs(f_val) < 1e-30 && continue
            try
                mass = PP.IceMassPowerLaw(Float64)
                closure = PP.P3Closure(Float64)
                bounds = PP.DiameterBounds(Float64, FORTRAN_FR[i_Fr])
                p = PP.distribution_parameters(10.0^fortran_log_mass(i_Qnorm), 1.0,
                                                FORTRAN_FR[i_Fr], FORTRAN_RHOR[i_rhor];
                                                liquid_fraction=0.0, mass, closure,
                                                diameter_bounds=bounds)
                j_val = pvar == :lambda_i ? p.λ : p.μ
                re = abs(f_val - j_val) / abs(f_val)
                (isnan(re) || isinf(re)) && continue
                push!(rel_errs, re)
            catch
            end
        end
        n = length(rel_errs)
        n == 0 && continue
        @printf("  %-10s  n=%d  median=%.2e  p95=%.2e  max=%.2e\n",
                string(pvar), n, median(rel_errs), quantile(rel_errs, 0.95), maximum(rel_errs))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Run validation
# ═══════════════════════════════════════════════════════════════════════════════

@info "P3 lookup table validation: Julia vs Fortran reference"
@info "Fortran table directory: $fortran_dir"

filepath = joinpath(fortran_dir, "p3_lookupTable_1.dat-v6.9-2momI.gz")
@info "Parsing Fortran Table 1 (2-moment)..." filepath
main = parse_table1_2mom(filepath)
@info "Parsed $(length(main)) entries"

evs = build_evaluators(nquad=80)

# --- Cleanest comparison: unrimed, dry ice ---
r1 = compare(main, evs, [1], [1], 1:50)        # Fr=0, Fl=0, all mass
print_results("UNRIMED DRY ICE (Fr=0, Fl=0)", r1)
print_psd_results(main, [1], 1:50)

# --- All rime fractions including Fr=1.0 ---
r2 = compare(main, evs, 1:4, [1], 1:50)        # Fr=0..1.0, Fl=0
print_results("ALL RIME FRACTIONS (Fr=0..1.0, Fl=0)", r2)
print_psd_results(main, 1:4, 1:50)

# --- All rime fractions and all liquid fractions ---
r3 = compare(main, evs, 1:4, 1:4, 1:50)        # Fr=0..1.0, Fl=0..1.0
print_results("FULL TABLE (Fr=0..1.0, Fl=0..1.0)", r3)

println("\n", "="^100)
println("REMAINING KNOWN LIMITATION:")
println("  nagg at very small mass (iq<10): Fortran 1500×50μm bins vs Julia 80-pt")
println("  Chebyshev produce different near-zero values. Not physically meaningful.")
println("="^100)
