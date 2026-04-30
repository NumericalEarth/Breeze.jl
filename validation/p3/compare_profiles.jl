using Printf

function load_data(filepath)
    lines = readlines(filepath)
    ncols = length(parse.(Float64, split(strip(lines[1]))))
    nrows = length(lines)
    data = zeros(ncols, nrows)
    for (i, line) in enumerate(lines)
        vals = parse.(Float64, split(strip(line)))
        data[:, i] .= vals
    end
    return data
end

ref = load_data("validation/p3/reference_out_p3_1TT.dat")
breeze = load_data("validation/p3/out_breeze_1TT.dat")

nk = 41
for t in [40, 60]
    offset = (t - 1) * nk
    println("\n=== PROFILE at t=$t min ===")
    println("  k    z(m)   Fort_qi  Bz_qi   Fort_qr  Bz_qr   Fort_T  Bz_T    Fort_ni   Bz_ni    F_Ff   B_Ff")
    for k in 1:nk
        ri = offset + k
        qi_r = ref[11, ri] * 1000
        qr_r = ref[8, ri] * 1000
        T_r = ref[6, ri]
        ni_r = ref[12, ri]
        Ff_r = ref[13, ri]
        qi_b = breeze[11, ri] * 1000
        qr_b = breeze[8, ri] * 1000
        T_b = breeze[6, ri]
        ni_b = breeze[12, ri]
        Ff_b = breeze[13, ri]
        z = ref[1, ri]
        if qi_r > 0.01 || qi_b > 0.01 || qr_r > 0.01 || qr_b > 0.01
            @printf("  %2d %6.0f  %7.3f  %7.3f  %7.3f  %7.3f  %6.1f  %6.1f  %8.0f  %8.0f  %5.2f  %5.2f\n",
                    k, z, qi_r, qi_b, qr_r, qr_b, T_r, T_b, ni_r, ni_b, Ff_r, Ff_b)
        end
    end
end
