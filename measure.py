#!/usr/bin/env python3
"""Dig GPS L1 C/A satellites out of thermal noise with a TV antenna.

GPS L1 (1575.42 MHz) arrives at about -130 dBm — roughly 20 dB BELOW
the thermal noise floor of a 2 MHz channel. You cannot see it on any
spectrum display. What rescues it is the grid: every satellite
transmits a known 1023-chip Gold code at exactly 1.023 Mchip/s
(one code epoch per millisecond), BPSK-spread, with 50 bps navigation
data on top. Correlating 1 ms of signal against a local code replica
concentrates the signal 30 dB (10*log10(1023)); noncoherent stacking
of a few hundred epochs buys the rest.

This script derives, from raw wideband IQ (no LNA, no GPS antenna):
  1. C/A code generator self-check vs published IS-GPS-200 octals
  2. parallel code-phase acquisition: all 32 PRNs x Doppler bins
     (FFT circular correlation, 1 ms coherent x N noncoherent)
  3. per-SV track over the capture: code-phase drift must equal
     -carrier_doppler/1540 (the code clock and carrier are locked to
     the same atomic standard 1540 half-cycles apart)
  4. measured C/A epoch length vs the value the carrier predicts
  5. the 50 bps navigation-data bit grid (20 ms alignment energy)
  6. optional: TLE cross-check (which SVs were overhead, predicted
     Doppler per SV, common receiver clock offset)

Usage:
  python measure.py --iq gps_l1_rabbit.cs16 --fs 2048000
        [--iq2 gps_l1_discone.cs16]   # second capture for the acq figure
        [--tle gps_ops.tle]           # celestrak gps-ops TLE file
        [--t0 "2026-07-20T12:14:27Z"] # capture start UTC (for TLE check)
        [--lat <deg> --lon <deg>]     # optional, only for --tle sky-check
        [--selftest]                  # synthetic -20 dB proof, then exit

Privacy note: the default lat/lon is a deliberately coarse central-
Virginia grid point. Elevation and Doppler predictions change by less
than the TLE error over ~100 km, so a vague point is all this needs.
This script never computes a position fix.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FL1 = 1575.42e6
CODE_RATE = 1.023e6
C_LIGHT = 299792458.0
THRESH = 2.5          # peak / second-peak detection threshold

# 4-series categorical palette (Okabe-Ito subset, CVD-validated)
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]

# ---------------------------------------------------------------- C/A codes
# IS-GPS-200 G2 phase-selector taps (1-indexed) for PRN 1..32
G2_TAPS = {
    1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9), 6: (2, 10),
    7: (1, 8), 8: (2, 9), 9: (3, 10), 10: (2, 3), 11: (3, 4), 12: (5, 6),
    13: (6, 7), 14: (7, 8), 15: (8, 9), 16: (9, 10), 17: (1, 4), 18: (2, 5),
    19: (3, 6), 20: (4, 7), 21: (5, 8), 22: (6, 9), 23: (1, 3), 24: (4, 6),
    25: (5, 7), 26: (6, 8), 27: (7, 9), 28: (8, 10), 29: (1, 6), 30: (2, 7),
    31: (3, 8), 32: (4, 9),
}
# first 10 chips as octal, from IS-GPS-200 (generator must reproduce these)
PUBLISHED_OCTAL = {1: "1440", 2: "1620", 3: "1710", 4: "1744", 5: "1133"}


def ca_code(prn):
    """1023-chip C/A code, +1/-1 floats. G1: x^10+x^3+1; G2: x^10+x^9+x^8+x^6+x^3+x^2+1."""
    s1, s2 = G2_TAPS[prn]
    g1 = np.ones(10, dtype=int)
    g2 = np.ones(10, dtype=int)
    out = np.empty(1023, dtype=int)
    for i in range(1023):
        out[i] = g1[9] ^ g2[s1 - 1] ^ g2[s2 - 1]
        fb1 = g1[2] ^ g1[9]
        fb2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = np.concatenate(([fb1], g1[:9]))
        g2 = np.concatenate(([fb2], g2[:9]))
    return 1.0 - 2.0 * out


def generator_selfcheck():
    ok = True
    for prn, want in PUBLISHED_OCTAL.items():
        chips = (1.0 - ca_code(prn)[:10]) / 2.0
        got = format(int("".join(str(int(b)) for b in chips), 2), "04o")
        ok &= got == want
        print(f"  PRN{prn:2d} first-10-chips octal: {got} (published {want})"
              f" {'OK' if got == want else 'FAIL'}")
    if not ok:
        raise SystemExit("C/A generator failed the published check values -- stop.")


def sampled_code(prn, fs, n_samp):
    idx = (np.arange(n_samp) * CODE_RATE / fs).astype(np.int64) % 1023
    return ca_code(prn)[idx]


# ------------------------------------------------------------- acquisition
def acquire(x, fs, prns, dopplers, n_noncoh):
    """Parallel code-phase search. 1 ms coherent x n_noncoh noncoherent.
    Returns {prn: dict(metric, dopp, code_phase, peak_over_floor)} where
    metric = peak / second peak (2nd peak excludes +-1 chip at all Dopplers)."""
    n1 = int(round(fs * 1e-3))
    blocks = x[: n1 * n_noncoh].reshape(n_noncoh, n1)
    t = np.arange(n1) / fs
    code_f = {p: np.conj(np.fft.fft(sampled_code(p, fs, n1))) for p in prns}
    excl = int(round(fs / CODE_RATE))
    maps = {p: [] for p in prns}
    for fd in dopplers:
        BF = np.fft.fft(blocks * np.exp(-2j * np.pi * fd * t)[None, :], axis=1)
        for p in prns:
            corr = np.fft.ifft(BF * code_f[p][None, :], axis=1)
            maps[p].append((np.abs(corr) ** 2).sum(axis=0))
    out = {}
    for p in prns:
        m = np.vstack(maps[p])
        di, ci = np.unravel_index(np.argmax(m), m.shape)
        mask = np.ones(m.shape[1], dtype=bool)
        for off in range(-excl, excl + 1):
            mask[(ci + off) % m.shape[1]] = False
        out[p] = dict(metric=float(m[di, ci] / m[:, mask].max()),
                      dopp=float(dopplers[di]), code_phase=int(ci),
                      peak_over_floor=float(m[di, ci] / np.median(m)))
    return out


def load_seg(path, fs, t0_s, dur_s):
    n0 = int(t0_s * fs) * 2
    n = int(dur_s * fs) * 2
    raw = np.memmap(path, dtype=np.int16, mode="r")[n0:n0 + n].astype(np.float32)
    x = raw[0::2] + 1j * raw[1::2]
    return x - x.mean()


def noise_stats(path, fs):
    x = load_seg(path, fs, 5.0, 1.0)
    i, q = x.real, x.imag
    clip = max(np.mean(np.abs(i) >= 32000), np.mean(np.abs(q) >= 32000))
    print(f"  RMS {np.sqrt(np.mean(np.abs(x)**2)):.0f} counts, clipped "
          f"{clip*100:.3f}% -- {'AGC left headroom' if clip < 1e-4 else 'CLIPPING'}")


# ----------------------------------------------------------------- tracking
def prompts_ms(x, fs, prn, fd, code_phase, n_ms):
    n1 = int(round(fs * 1e-3))
    code = np.roll(sampled_code(prn, fs, n1), code_phase)
    t = np.arange(n1 * n_ms) / fs
    xw = x[: n1 * n_ms] * np.exp(-2j * np.pi * fd * t)
    return (xw.reshape(n_ms, n1) * code[None, :]).sum(axis=1)


def track_sv(path, fs, prn, fd0, dur_s):
    """Re-acquire in 1 s steps; fine Doppler from FFT of squared prompts."""
    n1 = int(round(fs * 1e-3))
    times, phases, fds, metrics, pofs = [], [], [], [], []
    for t0 in np.arange(0.5, dur_s - 0.7, 1.0):
        x = load_seg(path, fs, t0, 0.110)
        dops = fd0 + np.arange(-375, 376, 125.0)
        r = acquire(x, fs, [prn], dops, n_noncoh=100)[prn]
        p = prompts_ms(x, fs, prn, r["dopp"], r["code_phase"], 100)
        F = np.fft.fftshift(np.fft.fft(p ** 2, 65536))       # squaring wipes data bits
        fax = np.fft.fftshift(np.fft.fftfreq(65536, 1e-3))
        fds.append(r["dopp"] + fax[np.argmax(np.abs(F))] / 2.0)
        times.append(t0); phases.append(r["code_phase"])
        metrics.append(r["metric"]); pofs.append(r["peak_over_floor"])
    ph = np.array(phases, float)
    for i in range(1, len(ph)):                              # unwrap mod 1 code period
        while ph[i] - ph[i - 1] > n1 / 2: ph[i] -= n1
        while ph[i] - ph[i - 1] < -n1 / 2: ph[i] += n1
    t = np.array(times)
    slope, icpt = np.polyfit(t, ph, 1)
    fd = float(np.median(fds))
    spc = fs / CODE_RATE
    lam = np.mean(pofs) - 1.0
    return dict(prn=prn, times=t, phases=ph, fd=fd, fd_std=float(np.std(fds)),
                slope=float(slope), icpt=float(icpt),
                pred_slope=float(-fd / 1540.0 * spc),
                resid_rms=float(np.std(ph - (slope * t + icpt))),
                epoch_ms=float(1023.0 / (CODE_RATE - slope / spc) * 1000.0),
                epoch_pred_ms=float(1.0 / (1.0 + fd / FL1)),
                cn0=float(10 * np.log10(max(lam, 1e-9)) + 30.0),
                metric_mean=float(np.mean(metrics)))


def bit_tent(path, fs, tr, t_start=5.0, dur_s=10.0):
    """50 bps bit-edge search: energy of 20-prompt coherent sums vs alignment."""
    n1 = int(round(fs * 1e-3))
    ps = []
    for chunk in range(int(dur_s * 10)):
        t0 = t_start + chunk * 0.1
        x = load_seg(path, fs, t0, 0.1)
        ci = int(round(tr["slope"] * t0 + tr["icpt"])) % n1
        code = np.roll(sampled_code(tr["prn"], fs, n1), ci)
        t = np.arange(len(x)) / fs
        xw = x * np.exp(-2j * np.pi * tr["fd"] * t)
        ps.append((xw.reshape(100, n1) * code[None, :]).sum(axis=1))
    p = np.concatenate(ps)
    nb = len(p) // 20 - 1
    e = np.array([np.mean(np.abs(p[off:off + nb * 20].reshape(nb, 20)
                                 .sum(axis=1)) ** 2) for off in range(20)])
    return e / e.max()


# ---------------------------------------------------------------- TLE check
def sky_check(tle_path, when_utc, lat_deg, lon_deg):
    import re
    from datetime import datetime, timezone
    from sgp4.api import Satrec, jday
    lines = [ln for ln in open(tle_path).read().splitlines() if ln.strip()]
    dt = datetime.fromisoformat(when_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    jd0, fr0 = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                    dt.second + dt.microsecond / 1e6)
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    a, f = 6378.137, 1 / 298.257223563
    e2 = f * (2 - f)
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    obs = np.array([N * np.cos(lat) * np.cos(lon), N * np.cos(lat) * np.sin(lon),
                    N * (1 - e2) * np.sin(lat)])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

    def gmst(jd, fr):
        d = jd + fr - 2451545.0
        return np.deg2rad((280.46061837 + 360.98564736629 * d) % 360.0)

    sky = {}
    for i in range(0, len(lines), 3):
        m = re.search(r"PRN (\d+)", lines[i])
        if not m:
            continue
        sat = Satrec.twoline2rv(lines[i + 1], lines[i + 2])
        rr = []
        for ddt in (0.0, 1.0):
            jd, fr = jd0, fr0 + ddt / 86400.0
            e, r, v = sat.sgp4(jd, fr)
            if e:
                break
            th = gmst(jd, fr)
            R = np.array([[np.cos(th), np.sin(th), 0],
                          [-np.sin(th), np.cos(th), 0], [0, 0, 1]])
            d = R @ np.asarray(r) - obs
            rr.append(np.linalg.norm(d))
            if ddt == 0.0:
                el = np.rad2deg(np.arcsin(np.dot(d, up) / rr[0]))
        if len(rr) == 2:
            sky[int(m.group(1))] = dict(el=float(el),
                                        fd=float(-(rr[1] - rr[0]) * 1000.0 / C_LIGHT * FL1))
    return sky


# ----------------------------------------------------------------- selftest
def selftest(fs):
    print("synthetic self-test: PRN7 at -20 dB SNR "
          f"(C/N0 {(-20 + 10*np.log10(fs)):.1f} dB-Hz), Doppler +1830 Hz")
    rng = np.random.default_rng(42)
    n = int(fs * 1e-3) * 40
    t = np.arange(n) / fs
    cr = CODE_RATE * (1 + 1830.0 / FL1)
    sig = ca_code(7)[(np.floor(411.25 + t * cr).astype(np.int64)) % 1023] \
        * np.exp(2j * np.pi * 1830.0 * t)
    x = sig * 10 ** (-20 / 20) + (rng.standard_normal(n)
                                  + 1j * rng.standard_normal(n)) / np.sqrt(2)
    res = acquire(x, fs, list(range(1, 33)), np.arange(-5000, 5001, 250.0), 20)
    best = max(res, key=lambda p: res[p]["metric"])
    r = res[best]
    print(f"  found PRN{best} metric {r['metric']:.2f} at {r['dopp']:+.0f} Hz "
          f"({'PASS' if best == 7 and r['metric'] > THRESH else 'FAIL'})")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iq", help="CS16 interleaved IQ at L1")
    ap.add_argument("--fs", type=float, default=2.048e6)
    ap.add_argument("--iq2", help="optional second capture (comparison panel)")
    ap.add_argument("--tle", help="celestrak gps-ops TLE file (optional)")
    ap.add_argument("--t0", default=None, help="capture start UTC ISO time")
    ap.add_argument("--lat", type=float, default=0.0,
                help="observer latitude - ONLY needed for the optional --tle "
                     "sky-check; left at 0 so no location ships in the repo")
    ap.add_argument("--lon", type=float, default=0.0,
                help="observer longitude - see --lat")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    print("C/A generator self-check vs IS-GPS-200:")
    generator_selfcheck()
    if a.selftest:
        selftest(a.fs)
        return
    if not a.iq:
        raise SystemExit("--iq required (or --selftest)")

    fs = a.fs
    dur_s = Path(a.iq).stat().st_size / 4 / fs
    dop = np.arange(-7000, 7001, 250.0)

    caps = [("capture", a.iq)] + ([("capture 2", a.iq2)] if a.iq2 else [])
    acq = {}
    for label, path in caps:
        print(f"\n{label}: {Path(path).name}")
        noise_stats(path, fs)
        x = load_seg(path, fs, 0.5, 0.310)
        acq[label] = acquire(x, fs, list(range(1, 33)), dop, n_noncoh=300)
        det = {p: r for p, r in acq[label].items() if r["metric"] > THRESH}
        print(f"  acquisition 1 ms x 300, +-7 kHz / 250 Hz, threshold {THRESH}:")
        for p, r in sorted(det.items(), key=lambda kv: -kv[1]["metric"]):
            print(f"    PRN{p:2d}  metric {r['metric']:.2f}  Doppler {r['dopp']:+5.0f} Hz"
                  f"  code phase {r['code_phase']} samp   DETECTED")
        if not det:
            print("    no PRN above threshold "
                  f"(best {max(r['metric'] for r in acq[label].values()):.2f})")

    main_label, main_path = caps[0]
    det = {p: r for p, r in acq[main_label].items() if r["metric"] > THRESH}

    # ---- track every detected SV
    tracks = []
    for p, r in sorted(det.items()):
        tr = track_sv(main_path, fs, p, r["dopp"], dur_s)
        tracks.append(tr)
        print(f"\nPRN{p} track over {dur_s:.0f} s:")
        print(f"  carrier Doppler {tr['fd']:+.1f} Hz (std {tr['fd_std']:.1f})"
              f"   C/N0 ~{tr['cn0']:.1f} dB-Hz")
        print(f"  code-phase drift {tr['slope']:+.3f} samp/s, "
              f"carrier predicts {tr['pred_slope']:+.3f} (-fd/1540): "
              f"ratio {tr['slope']/tr['pred_slope']:.3f}, fit rms {tr['resid_rms']:.2f} samp")
        print(f"  C/A epoch {tr['epoch_ms']:.9f} ms, carrier predicts "
              f"{tr['epoch_pred_ms']:.9f} ms "
              f"(diff {abs(tr['epoch_ms']-tr['epoch_pred_ms'])*1e9:.1f} ps)")

    # ---- 50 bps bit grid on the two strongest
    tents = []
    for tr in sorted(tracks, key=lambda t: -t["cn0"])[:2]:
        e = bit_tent(main_path, fs, tr, t_start=5.0, dur_s=min(10.0, dur_s - 6))
        tents.append((tr["prn"], e))
        print(f"\nPRN{tr['prn']} nav-bit alignment (20 ms blocks over 10 s): "
              f"tent peak at {int(np.argmax(e))} ms, max/min {e.max()/e.min():.2f} "
              f"-- 50 bps bit grid {'FOUND' if e.max()/e.min() > 1.2 else 'not seen'}")

    # ---- TLE cross-check
    if a.tle and a.t0:
        sky = sky_check(a.tle, a.t0, a.lat, a.lon)
        vis = sorted(p for p, s in sky.items() if s["el"] > 0)
        print(f"\nTLE check ({len(vis)} SVs above horizon): {vis}")
        dps = [tr["fd"] - sky[tr["prn"]]["fd"] for tr in tracks if tr["prn"] in sky]
        off = float(np.mean(dps))
        print(f"  common receiver clock offset {off:+.0f} Hz "
              f"({off/FL1*1e6:+.3f} ppm of the LO)")
        for tr in tracks:
            if tr["prn"] in sky:
                s = sky[tr["prn"]]
                print(f"  PRN{tr['prn']:2d} el {s['el']:+5.1f}  meas {tr['fd']:+7.1f} Hz"
                      f"  TLE {s['fd']:+6.0f}  resid {tr['fd']-s['fd']-off:+5.0f} Hz")
        # sub-threshold corroboration: best cell within 150 Hz of prediction?
        print("  sub-threshold PRNs whose best acquisition cell lands on the "
              "TLE-predicted Doppler (chance: ~1 in 19 each):")
        for p, r in sorted(acq[main_label].items()):
            if p in det or p not in sky or sky[p]["el"] < 5:
                continue
            resid = r["dopp"] - sky[p]["fd"] - off
            if abs(resid) <= 150:
                print(f"    PRN{p:2d} el {sky[p]['el']:+5.1f}  cell {r['dopp']:+6.0f} Hz"
                      f"  predicted {sky[p]['fd']+off:+6.0f}  (metric {r['metric']:.2f})")

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = HERE / "figures"
    figdir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, len(caps), figsize=(11, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, path) in zip(axes, caps):
        prns = sorted(acq[label])
        met = [acq[label][p]["metric"] for p in prns]
        cols = [COLORS[0] if m > THRESH else "#b0b0b0" for m in met]
        ax.bar(prns, met, color=cols, width=0.8)
        ax.axhline(THRESH, color="#444444", lw=1, ls="--")
        ax.text(32.3, THRESH, f" threshold {THRESH}", va="center", fontsize=8,
                color="#444444")
        for p, m in zip(prns, met):
            if m > THRESH:
                ax.annotate(f"PRN{p}", (p, m), textcoords="offset points",
                            xytext=(0, 4), ha="center", fontsize=9, color="#222222")
        ax.set_title(f"{Path(path).stem}", fontsize=11)
        ax.set_xlabel("PRN")
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("acquisition metric\n(peak / 2nd peak)")
    fig.suptitle("GPS L1 C/A acquisition -- 1 ms x 300 noncoherent, 32 PRNs x +-7 kHz",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(figdir / "gps_acq.png", dpi=140)
    print(f"\nwrote {figdir/'gps_acq.png'}")

    if tracks:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
        for k, tr in enumerate(sorted(tracks, key=lambda t: -t["cn0"])):
            c = COLORS[k % len(COLORS)]
            ax1.plot(tr["times"], tr["phases"] - tr["phases"][0], "o", ms=3.5,
                     color=c)
            tt = np.array([tr["times"][0], tr["times"][-1]])
            ax1.plot(tt, tr["pred_slope"] * (tt - tr["times"][0]), "--", lw=1.2,
                     color=c)
            ax1.annotate(f"PRN{tr['prn']}  {tr['fd']:+.0f} Hz",
                         (tr["times"][-1], tr["phases"][-1] - tr["phases"][0]),
                         textcoords="offset points", xytext=(6, 0), fontsize=9,
                         color=c, va="center")
        ax1.set_xlim(right=ax1.get_xlim()[1] + 6)
        ax1.set_xlabel("time into capture (s)")
        ax1.set_ylabel("code-phase drift (samples)")
        ax1.set_title("measured code phase (dots) vs carrier-Doppler\nprediction "
                      "-fd/1540 (dashed)", fontsize=10)
        ax1.grid(alpha=0.25, lw=0.5); ax1.set_axisbelow(True)
        for prn_tent, (p, e) in zip(range(len(tents)), tents):
            ax2.plot(range(20), e, "-o", ms=4, lw=1.5, color=COLORS[prn_tent])
            side = -1 if prn_tent == 0 else 1
            ax2.annotate(f"PRN{p}", (int(np.argmax(e)), e.max()),
                         textcoords="offset points", xytext=(8 * side, 6),
                         ha="right" if side < 0 else "left",
                         fontsize=9, color=COLORS[prn_tent])
        ax2.set_xlabel("alignment offset (ms mod 20)")
        ax2.set_ylabel("20 ms coherent-sum energy (norm.)")
        ax2.set_title("navigation-data bit grid: energy peaks when\n20 ms sums "
                      "align with the 50 bps bits", fontsize=10)
        ax2.set_xticks(range(0, 20, 2))
        ax2.grid(alpha=0.25, lw=0.5); ax2.set_axisbelow(True)
        for ax in (ax1, ax2):
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(figdir / "gps_track.png", dpi=140)
        print(f"wrote {figdir/'gps_track.png'}")


if __name__ == "__main__":
    main()
