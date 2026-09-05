#####
##### P3 lookup table format
#####
##### Layout of the P3 ice ASCII lookup tables and the parser that reads them.
##### The tables ship with the reference P3 distribution as whitespace-separated
##### text; nothing about the format depends on the language that wrote them.
##### A single file holds two blocks, indexed by the coordinates its rows carry:
#####
#####   Table 1: ice integrals,          axes (Q̄, Fᶠ, Fˡ, ρᶠ index)
#####   Table 2: rain-ice collection,    axes (Q̄, λ_r, Fᶠ, Fˡ, ρᶠ index)
#####
##### The rime-density axis is stored as an index 1..5 over the non-uniform grid
##### {50, 250, 450, 650, 900} kg/m³; `RimeDensityIndexedTable4D`/`5D` apply the
##### coordinate transform on evaluation.
#####

using Oceananigans.Architectures: CPU, on_architecture
using Oceananigans.Utils: TabulatedFunction

#####
##### Table dimension constants
#####

const N_NORMALIZED_MASS = 50
const N_RIME_FRACTION = 4
const N_LIQUID_FRACTION = 4
const N_RIME_DENSITY = 5
const N_RAIN_SLOPE = 30

#####
##### Axis range constants (matching the reference P3 grid generation)
#####
##### Table 1 mass axis: log_m = (i * 0.1 * log10(800)) - 18  for i = 11..60
##### Table 2 rain axis: log_lambda_r = -log10(1.24^j * 10e-6) for j = 1..30
#####

const LOG_MASS_MIN = 11 * 0.1 * log10(800) - 18
const LOG_MASS_MAX = 60 * 0.1 * log10(800) - 18
const LOG_LAMBDA_R_MIN = -log10(1.24^30 * 10e-6)
const LOG_LAMBDA_R_MAX = -log10(1.24^1 * 10e-6)

#####
##### Line parsing
#####

"""
$(TYPEDSIGNATURES)

Parse one whitespace-separated data line of a table file into a vector of
Float64. Handles the `E`-exponent scientific notation the files are written in
(e.g. `0.12345E+06`); integer fields are parsed as Float64 too.
"""
function parse_table_line(line::AbstractString)
    tokens = split(strip(line))
    return [parse(Float64, t) for t in tokens]
end

#####
##### Table file parser
#####

"""
$(TYPEDSIGNATURES)

Parse the P3 ice ASCII table file, which carries both table blocks.

Returns two dictionaries:
- `table1_fields`: Dict of Symbol => Array{FT,4} for ice integrals, with axes
  (normalized mass, Fᶠ, Fˡ, rime-density index)
- `table2_fields`: Dict of Symbol => Array{FT,5} for rain-ice collection, with axes
  (normalized mass, λʳ reversed into ascending order, Fᶠ, Fˡ, rime-density index)
"""
function parse_lookup_table_file(filepath::AbstractString, FT::Type)
    lines = readlines(filepath)

    Nq = N_NORMALIZED_MASS
    NFᶠ = N_RIME_FRACTION
    NFˡ = N_LIQUID_FRACTION
    Nρᶠ = N_RIME_DENSITY
    Nλʳ = N_RAIN_SLOPE

    # Column names for ice data.
    # Column 4 (`cloud_collection`) is the ice-cloud-water sweep-out integral
    # ∫ 𝕎(D) A(D) N'(D) dD. Ice-*rain* collection is not in the 4D ice block: it
    # needs the rain slope parameter as an extra coordinate and lives in the 5D
    # rain-ice block embedded later in the same Table 1 file
    # (`rain_number` / `rain_mass`).
    col_names = [
        :number_weighted, :mass_weighted, :aggregation, :cloud_collection,
        :ventilation, :effective_radius, :small_q, :large_q,
        :reflectivity, :ventilation_enhanced, :mean_diameter, :mean_density,
        :slope_parameter, :shape_parameter,
        :small_ice_ventilation_constant, :small_ice_ventilation_reynolds,
        :large_ice_ventilation_constant, :large_ice_ventilation_reynolds,
        :shedding,
        :cloud_aerosol_collection, :ice_aerosol_collection
    ]

    # Allocate arrays for ice integrals: (Qnorm, Fr, Fl, rhor)
    table1_fields = Dict{Symbol, Array{FT, 4}}()
    for name in col_names
        table1_fields[name] = zeros(FT, Nq, NFᶠ, NFˡ, Nρᶠ)
    end

    # Allocate arrays for rain-ice collection: (Qnorm, Drscale, Fr, Fl, rhor)
    rain_names = [:rain_number, :rain_mass]

    table2_fields = Dict{Symbol, Array{FT, 5}}()
    for name in rain_names
        table2_fields[name] = zeros(FT, Nq, Nλʳ, NFᶠ, NFˡ, Nρᶠ)
    end

    # Parse data lines (skip header line 1 and blank line 2)
    line_idx = 3  # 1-indexed; line 3 is first data line

    # Each row starts with its own coordinate indices, which are re-derived from the loop
    # counters instead of being read back, so they are skipped.
    ice_index_columns = 4
    rain_index_columns = 4

    # The file is one flat sequence of rows, and the loop nest below is not a choice: it
    # has to walk the axes in exactly the order the rows were written, outermost ρᶠ to
    # innermost λʳ, with the 50 ice rows of a (ρᶠ, Fᶠ, Fˡ) triple followed by its 50 × 30
    # rain-ice rows. Reordering any level silently mis-assigns every value.
    #
    #   ρᶠ index (1..5) → Fᶠ (1..4) → Fˡ (1..4) → { 50 ice rows; 50 × 30 rain-ice rows }
    for i_ρᶠ in 1:Nρᶠ
        for i_Fᶠ in 1:NFᶠ
            for i_Fˡ in 1:NFˡ
                for i_q in 1:Nq
                    vals = parse_table_line(lines[line_idx])
                    line_idx += 1
                    for (col_idx, name) in enumerate(col_names)
                        v = vals[ice_index_columns + col_idx]
                        table1_fields[name][i_q, i_Fᶠ, i_Fˡ, i_ρᶠ] = FT(v)
                    end
                end

                for i_q in 1:Nq
                    for i_λʳ in 1:Nλʳ
                        vals = parse_table_line(lines[line_idx])
                        line_idx += 1
                        # CRITICAL: reverse the λʳ axis. File order runs from largest λʳ
                        # to smallest, so it is reversed into ascending λʳ order here.
                        j_λʳ = Nλʳ - i_λʳ + 1
                        for (col_idx, name) in enumerate(rain_names)
                            v = vals[rain_index_columns + col_idx]
                            # Rain number and mass are stored as log10 in the file
                            table2_fields[name][i_q, j_λʳ, i_Fᶠ, i_Fˡ, i_ρᶠ] = FT(v)
                        end
                    end
                end
            end
        end
    end

    return table1_fields, table2_fields
end

#####
##### TabulatedFunction construction from raw arrays
#####

"""
$(TYPEDSIGNATURES)

Build a one- through five-dimensional `TabulatedFunction` directly from a
pre-computed data array and axis ranges.
"""
function make_lookup_table(data::Array{FT, N}, ranges, arch) where {FT, N}
    # Oceananigans owns the call methods for one- through five-dimensional
    # `TabulatedFunction` objects; nothing evaluates a table beyond that.
    1 ≤ N ≤ 5 || throw(ArgumentError("lookup tables support 1 to 5 dimensions, received $N"))
    points = size(data)
    inv_delta = map(ranges, points) do (lo, hi), n
        ifelse(n == 1, zero(FT), FT(1) / ((FT(hi) - FT(lo)) / (n - 1)))
    end
    gpu_data = on_architecture(arch, data)
    return TabulatedFunction{N, Nothing, typeof(gpu_data), typeof(ranges), typeof(inv_delta)}(
        nothing, gpu_data, ranges, inv_delta)
end

function ice_integrals_axes(FT)
    # Axes: (log_mass, Fᶠ, Fˡ, rime-density index). The two mass fractions span
    # [0, 1], and the rime-density index runs 1..5 over the non-uniform ρᶠ grid
    # (the wrapper applies the transform).
    return (
        (FT(LOG_MASS_MIN), FT(LOG_MASS_MAX)),
        (FT(0), FT(1)),
        (FT(0), FT(1)),
        (FT(1), FT(5))
    )
end

function rain_ice_collection_axes(FT)
    # Axes: (log_mass, log_lambda_r, Fᶠ, Fˡ, rime-density index); the ice
    # coordinates match `ice_integrals_axes`.
    return (
        (FT(LOG_MASS_MIN), FT(LOG_MASS_MAX)),
        (FT(LOG_LAMBDA_R_MIN), FT(LOG_LAMBDA_R_MAX)),
        (FT(0), FT(1)),
        (FT(0), FT(1)),
        (FT(1), FT(5))
    )
end
