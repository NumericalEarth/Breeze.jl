#!/usr/bin/env python3
"""Grade the moist bubble runs against CM1 using the tolerances in CRITERIA.md. numpy only; login-node safe."""
import glob, os, re, sys
import numpy as np

W = os.path.expanduser("~/rce-energy/moist-bubble-work")
Rd, Rv, cpd, cpv, cl, L, Tr, g = 287.04, 461.5, 1005.7, 1870.0, 4190.0, 2501000.0, 273.15, 9.81
eps = Rd / Rv
p00 = 1e5
Ttr, ptr = 273.16, 611.657
L0 = L - (cpv - cl) * Tr


def es_breeze(T):  # Breeze ClausiusClapeyron: integrates dln(e)/dT = (L0 + (cpv - cl) T) / (Rv T^2) from the triple point

    return ptr * (T / Ttr) ** ((cpv - cl) / Rv) * np.exp(L0 / Rv * (1 / Ttr - 1 / T))


def es_bolton(T):
    return 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))


def theta_e(T, p, rv, rt, es):
    e = p * rv / (eps + rv)
    cpt = cpd + cl * rt
    Lv = L + (cpv - cl) * (T - Tr)
    return T * (p00 / (p - e)) ** (Rd / cpt) * (e / es(T)) ** (-rv * Rv / cpt) * np.exp(Lv * rv / (cpt * T))


class Frame:
    pass


def derive(f, es, background=None):
    qd = 1 - f.qv - f.ql
    f.rv, f.rt = f.qv / qd, (f.qv + f.ql) / qd
    f.the = theta_e(f.T, f.p, f.rv, f.rt, es)
    f.qd = qd
    return f


def read_breeze(label, Nx, Nz):
    a = np.fromfile(f"{W}/results/{label}.bin", dtype=np.float64)
    nf = a.size // (7 * Nx * Nz)
    a = a[: nf * 7 * Nx * Nz].reshape(nf, 7, Nz, Nx)
    frames = []
    for n in range(nf):
        f = Frame()
        f.T, f.p, f.rho, f.qv, f.ql, f.u, f.w = [a[n, m] for m in range(7)]
        frames.append(derive(f, es_breeze))
    return frames


def read_cm1(run):
    d = f"{W}/cm1/{run}"
    ctl = open(f"{d}/cm1out_s.ctl").read()
    Nx = int(re.search(r"xdef\s+(\d+)", ctl).group(1))
    Nz = int(re.search(r"zdef\s+(\d+)", ctl).group(1))
    a = np.fromfile(f"{d}/cm1out_s.dat", dtype=np.float32)
    rec = Nx * (7 + 7 * Nz)
    nf = a.size // rec
    uf = np.fromfile(f"{d}/cm1out_u.dat", dtype=np.float32).reshape(nf, Nz, Nx + 1)
    wf = np.fromfile(f"{d}/cm1out_w.dat", dtype=np.float32).reshape(nf, Nz + 1, Nx)
    frames = []
    for n in range(nf):
        b = a[n * rec : (n + 1) * rec][7 * Nx :].reshape(7, Nz, Nx).astype(np.float64)
        f = Frame()
        th, f.p, rv, rc, f.u, _, f.w = b
        f.T = th * (f.p / p00) ** (Rd / cpd)
        rt = rv + rc
        f.qv, f.ql = rv / (1 + rt), rc / (1 + rt)
        f.rho = f.p / (Rd * f.T * (1 + rv / eps) / (1 + rt))
        derive(f, es_bolton)
        uf2 = (uf[n, :, :-1].astype(float) ** 2 + uf[n, :, 1:].astype(float) ** 2) / 2
        wf2 = (wf[n, :-1, :].astype(float) ** 2 + wf[n, 1:, :].astype(float) ** 2) / 2
        f.K = (uf2 + wf2) / 2
        f.wface = wf[n]
        frames.append(f)
    return frames, Nx, Nz


def energy(f, Δ, Lx=20e3):
    cvm = f.qd * (cpd - Rd) + f.qv * (cpv - Rv) + f.ql * cl
    Nz = f.T.shape[0]
    z = (np.arange(Nz) + 0.5) * Δ
    return (f.rho * (cvm * f.T + L0 * f.qv + f.K + g * z[:, None])).sum() * Δ * Δ / Lx, (f.rho * f.K).sum() * Δ * Δ / Lx


def top(the1, Δ, level=1.0):
    col = the1.max(axis=1)
    idx = np.where(col >= level)[0]
    if idx.size == 0:
        return float("nan")
    k = idx[-1]
    z = (np.arange(col.size) + 0.5) * Δ
    if k == col.size - 1:
        return z[k]
    return z[k] + Δ * (col[k] - level) / (col[k] - col[k + 1])


def metrics(frames, Δ):
    b = frames[0].the[:, 0:1]
    a0 = frames[0].the - b
    a = frames[-1].the - b
    m = dict(the_max=a.max(), the_min=a.min(), the0_max=a0.max(), top=top(a, Δ), sym=np.abs(a - a[:, ::-1]).max())
    i0 = (frames[0].qd * frames[0].rho * a0).sum()
    i1 = (frames[-1].qd * frames[-1].rho * a).sum()
    m["cons"] = (i1 - i0) / i0
    m["a"] = a
    return m


def csv(label):
    rows = open(f"{W}/results/{label}.csv").read().strip().split("\n")
    names = rows[0].split(",")
    data = np.array([[float(x) for x in r.split(",")] for r in rows[1:]])
    return {n: data[:, i] for i, n in enumerate(names)}


def checks(label):
    out = {}
    for line in open(f"{W}/results/{label}.txt"):
        k, v = line.split(" ", 1)
        out[k] = np.array([float(x) for x in v.split(",")]) if "," in v else float(v)
    return out


def pf(ok):
    return "pass" if ok else "**FAIL**"


def cm1_summary(run, Δ):
    frames, Nx, Nz = read_cm1(run)
    m = metrics(frames, Δ)
    m["wmax"], m["wmin"] = frames[-1].wface.max(), frames[-1].wface.min()
    E = [energy(f, Δ) for f in frames]
    m["dE"] = E[-1][0] - E[0][0]
    m["KEmax"] = max(e[1] for e in E)
    m["w"] = frames[-1].w
    m["frames"] = frames
    return m


def grade(label, Δ, ref, null_label=None):
    Nx, Nz = round(20e3 / Δ), round(10e3 / Δ)
    frames = read_breeze(label, Nx, Nz)
    if len(frames) < 11:
        print(f"| {label} | incomplete: {len(frames)} frames |")
        return None
    m = metrics(frames, Δ)
    ts = csv(label)
    ck = checks(label)
    comp = "_C-" in label
    r = dict(label=label)
    wmax = ts["wmax"][-1]
    a = m["a"]
    r["A1"] = (ck["env_θₑ_max"] - ck["env_θₑ_min"], abs((ck["env_θₑ_max"] + ck["env_θₑ_min"]) / 2 - 320))
    r["A2"], r["A3"], r["A4"] = ck["T_error"], ck["ql_error"], ck["p_error"] if comp else 0.0
    cm0 = ref["frames"][0]
    r["A5"] = (np.abs(ck["T_env"] - cm0.T[:, 0]).max(), np.abs(ck["qv_env"] - cm0.qv[:, 0]).max())
    if null_label and os.path.exists(f"{W}/results/{null_label}.csv"):
        tn = csv(null_label)
        r["A6"] = max(tn["wmax"].max(), -tn["wmin"].min())
        r["A7"] = max(tn["θₑ′max"].max(), -tn["θₑ′min"].min())
    else:
        r["A6"] = r["A7"] = float("nan")
    r["B1"] = (wmax, wmax / ref["wmax"] - 1)
    r["B2"] = (m["the_max"], m["the_max"] / ref["the_max"] - 1)
    r["B3"] = (m["the_min"], m["the_min"] - ref["the_min"])
    r["B4"] = (m["top"], m["top"] - ref["top"])
    r["B5"] = np.linalg.norm(a - ref["a"]) / np.linalg.norm(ref["a"])
    wc = frames[-1].w
    r["B6"] = np.linalg.norm(wc - ref["w"]) / np.linalg.norm(ref["w"])
    r["B7"] = m["cons"]
    r["B8"] = m["sym"]
    KEmax = ts["KE"].max()
    r["C"] = (ts["E"][-1] - ts["E"][0]) / KEmax
    r["C4"] = (ts["W"][-1] - ts["W"][0]) / ts["W"][0]
    r["KEmax"] = KEmax
    r["θ0"] = m["the0_max"]
    r["S"] = ts["S"][-1] - ts["S"][0]
    Ctol = 1e-6 if "_C-E" in label else 0.05
    tests = dict(
        A1=r["A1"][0] <= 0.01 and r["A1"][1] <= 0.01, A2=r["A2"] <= 1e-3, A3=r["A3"] <= 1e-6, A4=r["A4"] <= 1e-5,
        A5=r["A5"][0] <= 0.5 and r["A5"][1] <= 3e-4, A6=r["A6"] <= 0.05, A7=r["A7"] <= 0.05,
        B1=abs(r["B1"][1]) <= 0.10, B2=abs(r["B2"][1]) <= 0.15, B3=r["B3"][0] >= -0.5 and abs(r["B3"][1]) <= 0.25,
        B4=abs(r["B4"][1]) <= 200, B5=r["B5"] <= 0.25, B6=r["B6"] <= 0.25, B7=abs(r["B7"]) <= 0.05, B8=r["B8"] <= 0.1,
        C=abs(r["C"]) <= Ctol, C4=abs(r["C4"]) <= 1e-10)
    r["tests"] = tests
    r["fails"] = [k for k, v in tests.items() if not v]
    return r


if __name__ == "__main__":
    print("## CM1\n")
    print("| run | Δ | θₑ′max(0) | θₑ′max | θₑ′min | w max | w min | top (θₑ′=1) | conservation | sym | ΔE/KEmax | KEmax |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    refs = {}
    for run in sorted(os.listdir(f"{W}/cm1"), key=lambda s: (s[:2], int(re.search(r"_(\d+)m", s).group(1)), s)):
        Δ = float(re.search(r"_(\d+)m", run).group(1))
        try:
            m = cm1_summary(run, Δ)
        except Exception as e:
            print(f"| {run} | failed {e} |")
            continue
        refs[run] = m
        print(f"| {run} | {Δ:.0f} | {m['the0_max']:.3f} | {m['the_max']:.3f} | {m['the_min']:.3f} | {m['wmax']:.2f} | {m['wmin']:.2f} | {m['top']:.0f} | {m['cons']:+.3%} | {m['sym']:.3f} | {m['dE']/m['KEmax']:+.3f} | {m['KEmax']:.3e} |")
    print("\n## Breeze\n")
    print("| run | θₑ′max(0) | w max (vs CM1) | θₑ′max (vs CM1) | θₑ′min (diff) | top (diff m) | L2 θₑ′ | L2 w | conservation | sym | ΔE/KEmax | ΔW/W | null w | null θₑ′ | fails |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for path in sorted(glob.glob(f"{W}/results/*.bin")):
        label = os.path.basename(path)[:-4]
        if "-null" in label:
            continue
        Δ = float(re.search(r"_(\d+)m", label).group(1))
        ref = refs.get(f"e2_{Δ:.0f}m")
        if ref is None:
            continue
        null_label = label.replace("equilibrium_", "equilibrium-null_").replace(f"_{Δ:.0f}m", "_100m") if Δ == 100 else None
        r = grade(label, Δ, ref, null_label)
        if r is None:
            continue
        print(f"| {label} | {r['θ0']:.3f} | {r['B1'][0]:.2f} ({r['B1'][1]:+.1%}) | {r['B2'][0]:.3f} ({r['B2'][1]:+.1%}) | {r['B3'][0]:.3f} ({r['B3'][1]:+.3f}) | {r['B4'][0]:.0f} ({r['B4'][1]:+.0f}) | {r['B5']:.3f} | {r['B6']:.3f} | {r['B7']:+.2%} | {r['B8']:.3f} | {r['C']:+.2e} | {r['C4']:+.1e} | {r['A6']:.2e} | {r['A7']:.2e} | {','.join(r['fails']) or 'none'} |")
