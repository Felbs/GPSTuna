#!/usr/bin/env python3
"""relativity.py - decode Einstein from a 120 s GPS capture.

Pipeline (built on measure.py's acquisition/tracking):
  1. acquire + track the strongest PRN, model its code phase over the capture
  2. 1 ms prompt correlations with an absolute-time carrier wipe
  3. residual-chirp removal (per-second FFT of p^2 - squaring wipes the BPSK)
  4. phase reference from the unwrapped p^2 phase -> BPSK on the real axis
  5. 20 ms bit integration on the bit_tent grid -> ~5700 nav bits
  6. frame sync: TLM preamble 10001011 (both polarities) + IS-GPS-200
     (32,26) parity on every word; parse subframes 1-3
  7. THE PHYSICS: from the decoded orbit (e, sqrt(A)) and clock terms:
       - relativistic eccentricity clock wobble  dt_r = F * e * sqrtA * sin(E)
       - SR + GR budget from the decoded semi-major axis vs the factory
         clock offset -4.4647e-10
       - the "GPS drifts ~11 km/day without Einstein" number

  python relativity.py --iq ../captures/gps_l1_navbits.cs16
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from measure import (CODE_RATE, C_LIGHT, FL1, THRESH, acquire, bit_tent,
                     load_seg, sampled_code, track_sv)

MU = 3.986005e14                 # WGS-84 earth GM (m^3/s^2) - the IS-GPS value
F_REL = -4.442807633e-10         # s / sqrt(m)  (IS-GPS-200 relativistic constant)
OMEGA_E = 7.2921151467e-5
PREAMBLE = np.array([1, 0, 0, 0, 1, 0, 1, 1], dtype=np.int8)

# IS-GPS-200 20.3.5.2 parity: D25..D30 as XOR masks over d1..d24 (+ D29*/D30*)
PAR = [
    (29, [1, 2, 3, 5, 6, 10, 11, 12, 13, 14, 17, 18, 20, 23]),
    (30, [2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 18, 19, 21, 24]),
    (29, [1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 19, 20, 22]),
    (30, [2, 4, 5, 6, 8, 9, 13, 14, 15, 16, 17, 20, 21, 23]),
    (30, [1, 3, 5, 6, 7, 9, 10, 14, 15, 16, 17, 18, 21, 22, 24]),
    (29, [3, 5, 6, 8, 9, 10, 11, 13, 15, 19, 22, 23, 24]),
]


# ------------------------------------------------------------ bits from RF
def prompt_stream(path, fs, tr, t_start, dur_s):
    """Phase-continuous 1 ms prompts over dur_s using the track's code-phase
    model (slope/icpt) and a fixed-fd carrier wipe on ABSOLUTE time (so the
    carrier reference never resets between chunks)."""
    n1 = int(round(fs * 1e-3))
    out = []
    for chunk in range(int(dur_s * 10)):
        t0 = t_start + chunk * 0.1
        x = load_seg(path, fs, t0, 0.1)
        ci = int(round(tr["slope"] * t0 + tr["icpt"])) % n1
        code = np.roll(sampled_code(tr["prn"], fs, n1), ci)
        t_abs = t0 + np.arange(len(x)) / fs
        xw = x * np.exp(-2j * np.pi * tr["fd"] * t_abs)
        out.append((xw.reshape(100, n1) * code[None, :]).sum(axis=1))
    return np.concatenate(out)


def clean_carrier(p):
    """Remove the residual Doppler chirp + phase so BPSK lands on the real axis.
    Per-second: FFT of p^2 gives 2*(residual f); integrate into a phase model;
    then the unwrapped low-passed p^2 phase (halved) is the carrier reference."""
    n_blk = len(p) // 1000
    # 1) block frequency estimates from squared prompts (BPSK removed)
    freqs = np.zeros(n_blk)
    for i in range(n_blk):
        z = p[i * 1000:(i + 1) * 1000] ** 2
        Z = np.fft.fftshift(np.fft.fft(z * np.hanning(len(z)), 8192))
        fax = np.fft.fftshift(np.fft.fftfreq(8192, d=1e-3))
        freqs[i] = fax[np.argmax(np.abs(Z))] / 2.0
    # 2) integrate the piecewise-constant frequency into a phase correction
    t = np.arange(len(p)) * 1e-3
    f_of_t = np.repeat(freqs, 1000)[: len(p)]
    if len(f_of_t) < len(p):
        f_of_t = np.pad(f_of_t, (0, len(p) - len(f_of_t)), mode="edge")
    phi = 2 * np.pi * np.cumsum(f_of_t) * 1e-3
    p1 = p * np.exp(-1j * phi)
    # 3) residual phase reference from smoothed p^2 (halved, unwrapped)
    z = p1 ** 2
    k = 51
    zs = np.convolve(z, np.ones(k) / k, mode="same")
    ref = 0.5 * np.unwrap(np.angle(zs))
    return p1 * np.exp(-1j * ref)


def costas(p, bn=20.0, zeta=0.707):
    """Classic 2nd-order Costas PLL on the 1 ms prompts (1 kHz update).
    err = atan(Q/I) is BPSK-data-insensitive; the closed loop tracks the
    phase wander that defeated every open-loop batch correction."""
    dt = 1e-3
    wn = bn / (zeta + 1 / (4 * zeta)) * 2  # ~ Bn -> wn
    k1 = 2 * zeta * wn * dt
    k2 = (wn * dt) ** 2
    ph = 0.0
    fr = 0.0
    out = np.empty_like(p)
    for k in range(len(p)):
        v = p[k] * np.exp(-1j * ph)
        out[k] = v
        if abs(v.real) > 1e-12:
            err = np.arctan(v.imag / v.real)
        else:
            err = 0.0
        fr += k2 * err
        ph += fr + k1 * err
    return out


def bit_sums(pc, tent_off):
    """20 ms complex bit sums on the tent grid."""
    p = pc[tent_off:]
    nb = len(p) // 20
    s = p[: nb * 20].reshape(nb, 20).sum(axis=1)
    bits = (s.real > 0).astype(np.int8)
    conf = np.abs(s.real) / (np.abs(s.imag) + 1e-9)
    return s, bits, float(np.median(conf))


def subframe_bits(p_raw, tent_off, i):
    """Bits for ONE subframe from a LOCAL carrier model: over this ~6 s region,
    fit unwrap(angle(smoothed p^2)) with a cubic - a smooth per-region phase
    reference (no pi-jump seams possible inside a region, unlike a global
    stitch). Then 20 ms sums, preamble-anchored transition decode (safe under
    GPS's complement convention: flipped subframes parity-check to the same d)."""
    a0 = tent_off + (i - 2) * 20
    a1 = tent_off + (i + 300) * 20
    if a0 < 0 or a1 > len(p_raw):
        return None, 0, 0
    p = p_raw[a0:a1]
    z = p ** 2
    k = 41
    zs = np.convolve(z, np.ones(k) / k, mode="same")
    ph = np.unwrap(np.angle(zs))
    t = np.arange(len(p))
    ref = np.polyval(np.polyfit(t, ph, 3), t) / 2.0
    pr = p * np.exp(-1j * ref)
    s = pr.reshape(-1, 20).sum(axis=1)                       # 302 bit sums
    same = (s[1:] * np.conj(s[:-1])).real > 0
    b = np.zeros(len(s), dtype=np.int8)
    b[2:10] = PREAMBLE                                       # anchor at the preamble
    for k2 in range(10, len(b)):
        b[k2] = b[k2 - 1] ^ (0 if same[k2 - 1] else 1)
    for k2 in range(1, -1, -1):
        b[k2] = b[k2 + 1] ^ (0 if same[k2] else 1)
    return b[2:302], b[0], b[1]                              # 300 bits, D29*, D30*


# ------------------------------------------------------- frame sync + parity
def parity_ok(word, d29s, d30s):
    """word = 30 bits (transmitted). Returns (ok, data24) with the complement
    convention: d_i = D_i XOR D30* for the 24 data bits."""
    D = word.astype(np.int8)
    d = D[:24] ^ d30s
    for k, (star, mask) in enumerate(PAR):
        acc = d29s if star == 29 else d30s
        for m in mask:
            acc ^= d[m - 1]
        if acc != D[24 + k]:
            return False, d
    return True, d


def find_grid(bits):
    """Locate the 300-bit subframe grid via the preamble's mod-300 histogram
    (both polarities). Returns (polarity, [subframe start indices])."""
    best = (0, 0, [])
    for pol in (0, 1):
        b = bits ^ pol
        pos = [i for i in range(len(b) - 300) if np.array_equal(b[i:i + 8], PREAMBLE)]
        if not pos:
            continue
        mods = np.bincount([i % 300 for i in pos], minlength=300)
        phase = int(np.argmax(mods))
        starts = [i for i in range(phase, len(b) - 300, 300)]
        score = int(mods[phase])
        if score > best[0]:
            best = (score, pol, starts)
    return best[1], best[2]


def harvest_words(p_raw, tent_off, starts):
    """Word-level harvest from locally-decoded, preamble-anchored subframes:
    chain D29*/D30* through each; keep every word that passes parity."""
    out = []
    n_ok = n_tot = 0
    for i in starts:
        sb, d29s, d30s = subframe_bits(p_raw, tent_off, i)
        if sb is None:
            continue
        words = {}
        for w in range(10):
            word = sb[w * 30:(w + 1) * 30]
            if len(word) < 30:
                break
            ok, d = parity_ok(word, d29s, d30s)
            n_tot += 1
            if ok:
                words[w] = d
                n_ok += 1
            d29s, d30s = word[28], word[29]
        if 1 in words:                       # need a clean HOW for the subframe ID
            sfid = ubits(words[1], 20, 22)
            tow = ubits(words[1], 1, 17)
            if 1 <= sfid <= 5:
                out.append((sfid, tow, words))
    print(f"[rel] word parity: {n_ok}/{n_tot} clean "
          f"({100*n_ok/max(n_tot,1):.0f}%) across {len(starts)} subframes")
    return out


def ubits(d, a, b):
    """unsigned int from data-bit positions a..b (1-indexed) of a 24-bit word."""
    v = 0
    for k in range(a - 1, b):
        v = (v << 1) | int(d[k])
    return v


def sbits(d, a, b):
    v = ubits(d, a, b)
    n = b - a + 1
    return v - (1 << n) if v >= (1 << (n - 1)) else v


def cat(hi, hi_n, lo, lo_n, signed):
    v = (hi << lo_n) | lo
    n = hi_n + lo_n
    if signed and v >= (1 << (n - 1)):
        v -= 1 << n
    return v


def parse_harvest(harvest):
    """Assemble ephemeris from parity-clean words (fields need specific words;
    a field is taken from the FIRST subframe repeat where its words are clean)."""
    eph = {"tows": [(s, t) for s, t, _ in harvest]}

    def put(key, val):
        if key not in eph:
            eph[key] = val

    for sfid, tow, W in harvest:
        if sfid == 1:
            if 2 in W:
                put("WN", ubits(W[2], 1, 10))
            if 8 in W:
                put("af2", sbits(W[8], 1, 8) * 2 ** -55)
                put("af1", sbits(W[8], 9, 24) * 2 ** -43)
            if 9 in W:
                put("af0", sbits(W[9], 1, 22) * 2 ** -31)
            if 7 in W:
                put("toc", ubits(W[7], 9, 24) * 16)
        elif sfid == 2:
            if 2 in W:
                put("IODE2", ubits(W[2], 1, 8))
                put("Crs", sbits(W[2], 9, 24) * 2 ** -5)
            if 3 in W:
                put("dn", sbits(W[3], 1, 16) * 2 ** -43 * np.pi)
            if 3 in W and 4 in W:
                put("M0", cat(ubits(W[3], 17, 24), 8, ubits(W[4], 1, 24), 24, True) * 2 ** -31 * np.pi)
            if 5 in W:
                put("Cuc", sbits(W[5], 1, 16) * 2 ** -29)
            if 5 in W and 6 in W:
                put("e", cat(ubits(W[5], 17, 24), 8, ubits(W[6], 1, 24), 24, False) * 2 ** -33)
            if 7 in W:
                put("Cus", sbits(W[7], 1, 16) * 2 ** -29)
            if 7 in W and 8 in W:
                put("sqrtA", cat(ubits(W[7], 17, 24), 8, ubits(W[8], 1, 24), 24, False) * 2 ** -19)
            if 9 in W:
                put("toe", ubits(W[9], 1, 16) * 16)
        elif sfid == 3:
            if 2 in W:
                put("Cic", sbits(W[2], 1, 16) * 2 ** -29)
            if 2 in W and 3 in W:
                put("Omega0", cat(ubits(W[2], 17, 24), 8, ubits(W[3], 1, 24), 24, True) * 2 ** -31 * np.pi)
            if 4 in W:
                put("Cis", sbits(W[4], 1, 16) * 2 ** -29)
            if 4 in W and 5 in W:
                put("i0", cat(ubits(W[4], 17, 24), 8, ubits(W[5], 1, 24), 24, True) * 2 ** -31 * np.pi)
            if 6 in W:
                put("Crc", sbits(W[6], 1, 16) * 2 ** -5)
            if 6 in W and 7 in W:
                put("omega", cat(ubits(W[6], 17, 24), 8, ubits(W[7], 1, 24), 24, True) * 2 ** -31 * np.pi)
            if 8 in W:
                put("OmegaDot", sbits(W[8], 1, 24) * 2 ** -43 * np.pi)
            if 9 in W:
                put("IDOT", sbits(W[9], 9, 22) * 2 ** -43 * np.pi)
    return eph


# ----------------------------------------------------------------- physics
def physics_report(eph, prn):
    e = eph["e"]
    sqrtA = eph["sqrtA"]
    A = sqrtA ** 2
    n0 = np.sqrt(MU / A ** 3)
    T = 2 * np.pi / n0
    amp = abs(F_REL) * e * sqrtA
    lines = []
    lines.append(f"PRN{prn} DECODED EPHEMERIS (from OUR antenna's bits):")
    lines.append(f"  eccentricity e        = {e:.6e}")
    lines.append(f"  sqrt(A)               = {sqrtA:.6f} sqrt(m)  ->  a = {A/1e3:,.1f} km")
    lines.append(f"  orbital period        = {T/3600:.3f} h (should be ~11.97 h)")
    if "af0" in eph:
        lines.append(f"  clock af0/af1/af2     = {eph['af0']:.3e} s, {eph['af1']:.3e}, {eph['af2']:.3e}")
    lines.append("")
    lines.append("RELATIVISTIC ECCENTRICITY WOBBLE  dt_r = F*e*sqrtA*sin(E):")
    lines.append(f"  amplitude = {amp*1e9:.2f} ns  (peak-to-peak {2*amp*1e9:.2f} ns each orbit)")
    lines.append(f"  = the satellite clock visibly speeds up/slows down as its")
    lines.append(f"    altitude+speed change around the ellipse. Receivers MUST")
    lines.append(f"    apply this correction; the parameters came from our own decode.")
    lines.append("")
    # SR + GR budget from the decoded orbit
    v2 = MU / A                              # circular-orbit v^2
    sr = -v2 / (2 * C_LIGHT ** 2)
    r_geoid = 6.3781363e6                    # effective earth radius (geoid)
    # earth-surface potential incl. rotation (the clock we compare against)
    phi_surf = -MU / r_geoid - 0.5 * (OMEGA_E * r_geoid) ** 2
    phi_sat = -MU / A
    gr = (phi_sat - phi_surf) / C_LIGHT ** 2
    net = sr + gr
    lines.append("SR + GR CLOCK BUDGET (from the DECODED semi-major axis):")
    lines.append(f"  special relativity (speed {np.sqrt(v2):,.0f} m/s): {sr:+.4e}  ({sr*86400*1e6:+.2f} us/day SLOW)")
    lines.append(f"  general relativity (altitude):                    {gr:+.4e}  ({gr*86400*1e6:+.2f} us/day FAST)")
    lines.append(f"  net predicted offset:                             {net:+.4e}")
    lines.append(f"  GPS factory clock detune (IS-GPS-200):            -4.4647e-10")
    lines.append(f"  agreement: {net/-4.4647e-10*100:.1f}% of the designed value")
    lines.append("")
    drift_us = net * 86400 * 1e6
    lines.append(f"WITHOUT EINSTEIN: clocks drift {drift_us:+.1f} us/day ->")
    lines.append(f"  ranging error {abs(drift_us)*1e-6*C_LIGHT/1e3:,.1f} km/day. Maps dead by lunch.")
    return "\n".join(lines), (e, sqrtA, T, amp)


def wobble_plot(eph, prn, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e, sqrtA = eph["e"], eph["sqrtA"]
    A = sqrtA ** 2
    n0 = np.sqrt(MU / A ** 3)
    T = 2 * np.pi / n0
    t = np.linspace(0, T, 2000)
    M = eph.get("M0", 0.0) + n0 * t
    E = M.copy()
    for _ in range(12):
        E = M + e * np.sin(E)
    dtr = F_REL * e * sqrtA * np.sin(E)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(t / 3600, dtr * 1e9, lw=2, color="#0072B2")
    ax.axhline(0, color="#888", lw=0.7)
    ax.set_xlabel("time into orbit (hours)")
    ax.set_ylabel("relativistic clock correction  (ns)")
    ax.set_title(f"GPS PRN{prn}: time dilation wobble decoded from our own antenna\n"
                 f"$\\Delta t_r = F\\,e\\,\\sqrt{{A}}\\,\\sin E$   "
                 f"e={e:.2e}, $\\sqrt{{A}}$={sqrtA:.1f} -> amplitude {abs(F_REL)*e*sqrtA*1e9:.1f} ns")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iq", default=str(HERE.parent / "captures" / "gps_l1_navbits.cs16"))
    ap.add_argument("--fs", type=float, default=2.048e6)
    a = ap.parse_args()
    fs = a.fs
    dur_s = Path(a.iq).stat().st_size / 4 / fs
    print(f"[rel] capture {Path(a.iq).name}: {dur_s:.0f} s @ {fs/1e6} MHz")
    x = load_seg(a.iq, fs, 0.5, 0.310)
    acq = acquire(x, fs, list(range(1, 33)), np.arange(-7000, 7001, 250.0), 300)
    prn, r = max(acq.items(), key=lambda kv: kv[1]["metric"])
    if r["metric"] < THRESH:
        print("[rel] no PRN above threshold - blocker logged")
        return 1
    print(f"[rel] strongest: PRN{prn} metric {r['metric']:.2f} dopp {r['dopp']:+.0f}")
    tr = track_sv(a.iq, fs, prn, r["dopp"], dur_s)
    print(f"[rel] track: fd {tr['fd']:+.1f} Hz  keys={sorted(tr.keys())}")
    tent = bit_tent(a.iq, fs, tr, t_start=5.0, dur_s=10.0)
    tent_off = int(np.argmax(tent))
    print(f"[rel] bit grid offset {tent_off} ms (tent max/min {tent.max()/max(tent.min(),1e-9):.2f})")
    p = prompt_stream(a.iq, fs, tr, 1.0, min(dur_s - 2.0, 118.0))
    pc = costas(clean_carrier(p))
    s, bits, conf = bit_sums(pc, tent_off)
    print(f"[rel] {len(bits)} nav bits, median |I/Q| confidence {conf:.1f}")
    pol, starts = find_grid(bits)
    print(f"[rel] subframe grid: {len(starts)} starts")
    # UNION harvest: absolute-sign bits (global stitch) AND local anchored decode
    harvest = harvest_words(p, tent_off, starts)
    b = bits ^ pol
    n_ok = 0
    for i in starts:
        d29s, d30s = (b[i - 2], b[i - 1]) if i >= 2 else (0, 0)
        words = {}
        for w in range(10):
            word = b[i + w * 30: i + (w + 1) * 30]
            if len(word) < 30:
                break
            ok, d = parity_ok(word, d29s, d30s)
            if ok:
                words[w] = d
                n_ok += 1
            d29s, d30s = word[28], word[29]
        if 1 in words:
            sfid = ubits(words[1], 20, 22)
            tow = ubits(words[1], 1, 17)
            if 1 <= sfid <= 5:
                harvest.append((sfid, tow, words))
    print(f"[rel] union harvest: +{n_ok} clean words from global-stitch decode")
    # ---- MAJORITY VOTE across subframe repeats (ephemeris repeats every 30 s;
    # 4% BER -> ~0.5% after a 3-4 way vote; words 1-2 change per repeat (TOW)
    # so they come from a clean instance, words 3-10 get voted).
    anchors = {}
    for k, i in enumerate(starts):
        d29s, d30s = (b[i - 2], b[i - 1]) if i >= 2 else (0, 0)
        w1 = b[i:i + 30]
        ok, _ = parity_ok(w1, d29s, d30s)
        d29s2, d30s2 = w1[28], w1[29]
        ok2, d2 = parity_ok(b[i + 30:i + 60], d29s2, d30s2)
        if ok2:
            sf = ubits(d2, 20, 22)
            if 1 <= sf <= 5:
                anchors[k] = sf
    if anchors:
        k0, s0 = next(iter(anchors.items()))
        consistent = all((s0 - 1 + (k - k0)) % 5 + 1 == s for k, s in anchors.items())
        print(f"[rel] sfid anchors {anchors} cycle-consistent={consistent}")
        groups = {}
        for k, i in enumerate(starts):
            sf = (s0 - 1 + (k - k0)) % 5 + 1
            if i + 300 <= len(b):
                groups.setdefault(sf, []).append(i)
        for sf, idxs in sorted(groups.items()):
            if sf > 3 or len(idxs) < 2:
                continue
            stack = np.stack([b[i:i + 300] for i in idxs])
            voted = (stack.mean(axis=0) > 0.5).astype(np.int8)
            inst = idxs[0] if k0 not in range(len(starts)) else idxs[0]
            hyb = voted.copy()
            hyb[:60] = b[inst:inst + 60]              # TLM+HOW from an instance
            d29s, d30s = (b[inst - 2], b[inst - 1]) if inst >= 2 else (0, 0)
            words = {}
            nok = 0
            for w in range(10):
                word = hyb[w * 30:(w + 1) * 30]
                ok, d = parity_ok(word, d29s, d30s)
                if ok:
                    words[w] = d
                    nok += 1
                d29s, d30s = word[28], word[29]
            print(f"[rel] VOTE sfid{sf}: {len(idxs)} repeats -> {nok}/10 words clean")
            if 1 in words:
                harvest.append((sf, ubits(words[1], 1, 17), words))
            else:
                harvest.append((sf, 0, words))
    if not harvest:
        print("[rel] BLOCKER: no parity-clean HOW words")
        return 1
    eph = parse_harvest(harvest)
    ids = [s for s, _ in eph.get("tows", [])]
    print(f"[rel] subframe IDs decoded: {sorted(set(ids))}  TOWs: "
          f"{[t for _, t in eph['tows'][:6]]}")
    need = {"e", "sqrtA"}
    if not need.issubset(eph):
        print(f"[rel] BLOCKER: got {sorted(set(eph) - {'tows'})} - need subframe 2 "
              f"(e, sqrtA). May need a longer capture window.")
        return 1
    report, _ = physics_report(eph, prn)
    print("\n" + "=" * 66)
    print(report)
    print("=" * 66)
    png = wobble_plot(eph, prn, str(HERE / "figures" / "relativity_wobble.png"))
    print(f"[rel] wobble plot -> {png}")
    (HERE / "RELATIVITY.md").write_text(
        "# Einstein, decoded from a 120-second GPS capture\n\n"
        "Pipeline: acquire/track PRN -> 1 ms prompts -> carrier cleanup -> 50 bps\n"
        "nav bits -> preamble+parity frame sync -> ephemeris -> relativity.\n\n"
        "```\n" + report + "\n```\n\n"
        "![wobble](figures/relativity_wobble.png)\n", encoding="utf-8")
    print("[rel] RELATIVITY.md written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
