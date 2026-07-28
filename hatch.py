#!/usr/bin/env python3
"""hatch.py - carrier-phase (Hatch) smoothing of the joint GPS+Galileo fix,
plus the inter-system-bias slow-trend forensics.

WHY: joint_fix.py's remaining ~15 m epoch scatter is code noise. The L1
carrier is a 19 cm ruler riding on the exact same clock as the code
(1540 carrier cycles per chip, by construction). Integrated carrier
Doppler therefore predicts the code-phase CHANGE between any two times
to millimetres; only the code measurement's absolute value is noisy.
Carrier-smoothing = propagate every code measurement in a window to the
epoch along the carrier, then average:

    phi_smooth(T) = P(T) + mean_{t_i in (T-W, T]} [ phi(t_i) - P(t_i) ]

with P(t) = -Theta(t) / (1540 * 1023)  [ms of code delay], Theta(t) the
continuous carrier phase in cycles. phi - P is the code-minus-carrier
(CMC) observable: constant up to code noise, code multipath and 2x the
ionosphere rate (iono delays code, advances carrier). CMC jumps are the
cycle-slip / outlier detector. This is algebraically the fixed-window
Hatch filter, written in its batch form.

CARRIER SOURCES (continuity is everything):
  * GPS: one phase-continuous 1 ms prompt stream across the whole clean
    section (cubic code-phase model + exact polynomial-integral carrier
    wipe -- the linear-chirp trick of joint_fix generalized, because over
    470 s the Doppler-rate itself drifts). Residual carrier = per-second
    FFT of p^2 (squaring wipes the BPSK) integrated into a phase model,
    plus the unwrapped smoothed-p^2 phase. Theta = wipe + residual.
  * Galileo: the pilot PLL of gal_inav.track_prn now records its NCO
    phase at every 4 ms block start (sample-stamped by ptrs) -- that IS
    the integrated Doppler, from a genuine closed-loop track.

MEASUREMENT-TIME CONVENTION: an N-block noncoherent acquisition smears
the drifting code phase; the summed peak sits at the phase of the
buffer CENTROID, not the buffer start. joint_fix's snapshots (GPS 300 x
1 ms -> +0.1495 s, Galileo 40 x 4 ms -> +0.078 s) are kept EXACTLY, so
smoothed-vs-raw differs only by the smoothing. The centroid asynchrony
between the two systems (70 ms) x the per-bird pseudorange rate is also
a deliberate ISB suspect: --stage isb projects it through the epoch
geometry (hypothesis c), alongside the fs-error grid-slide prediction
(hypothesis d), anchor-fit drifts (a, b) and the CMC divergence rates
(e).

Stages (caches in lab_local/, safe to delete):
  --stage gps [--prn N]     GPS code series @ 2 s + continuous carrier
  --stage gal [--prn N]     Galileo PLL re-track (phase) + code series
  --stage smooth            CMC, slips, Hatch -> joint_snap_smoothed.json
  --stage solve             joint_fix solve on the smoothed snapshots
                            -> lab_local/fix_result_smoothed.json
  --stage isb               ISB slow-trend hypothesis shoot-out

PRIVACY: coordinates only inside gitignored lab_local/. This tool
prints rms / scatter / ns / slip counts -- never latitude/longitude.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joint_fix
from joint_fix import EPOCHS, FS, GAL_T0, LAB, PATH, _load_cache
from measure import CODE_RATE, FL1, load_seg, sampled_code
from measure import acquire as gps_acquire
import gal_e1
from gal_e1 import T_COH, boc_replica, load_codes
from gal_inav import boc_chips, track_prn
from fix import C, clock_corr, sat_ecef

TA, TB = 293.0, 767.4            # series span: clean zone through last epoch
CAD = 2.0                        # code-measurement cadence [s]
W_HATCH = 100.0                  # smoothing window [s], causal
N_NONCOH_GPS = 300               # = joint_fix.stage_snap, keeps conventions
SEG_GPS = 0.310
D_EFF_GPS = 0.1495               # noncoherent centroid: mean block start,
D_EFF_GAL = 0.078                # 300x1 ms and 40x4 ms respectively
K_CY_MS = 1540.0 * 1023.0        # carrier cycles per ms of code delay
MS_M = C * 1e-3                  # metres per ms of delay
LAM_L1 = C / FL1                 # 0.1903 m
TCXO_PPB = 796.7216401063106     # lab_local/tcxo_cal.json (2026-07-28T01:00)


def _grid_times():
    g = list(np.arange(TA, TB - 0.5, CAD)) + [float(T) for T in EPOCHS]
    return np.array(sorted(set(g)))


# ------------------------------------------------------ GPS carrier + code
def prompt_stream_poly(prn, code_poly, fd_poly, t_start, dur_s):
    """Phase-continuous 1 ms prompts with a cubic code-phase model and an
    EXACT polynomial-integral carrier wipe on absolute time. relativity's
    prompt_stream tops out at a linear chirp; over 470 s the Doppler rate
    itself drifts (~0.1 Hz/s change), which bends the code phase by more
    than a chip -- hence the cubic."""
    n1 = int(round(FS * 1e-3))
    ipoly = np.polyint(fd_poly)                    # cycles(t), exact
    base = sampled_code(prn, FS, n1)
    out = []
    for chunk in range(int(dur_s * 10)):
        t0 = t_start + chunk * 0.1
        x = load_seg(PATH, FS, t0, 0.1)
        ci = int(round(np.polyval(code_poly, t0))) % n1
        code = np.roll(base, ci)
        t_abs = t0 + np.arange(len(x)) / FS
        xw = x * np.exp(-2j * np.pi * np.polyval(ipoly, t_abs))
        out.append((xw.reshape(100, n1) * code[None, :]).sum(axis=1))
    return np.concatenate(out)


def carrier_resid_cycles(p):
    """Residual carrier phase of a prompt stream, in cycles, continuous.
    Stage 1: per-second FFT of p^2 (BPSK squared away) -> piecewise
    frequency, integrated to phase. Stage 2: 51 ms smoothed p^2 phase,
    unwrapped and halved. Also returns an unwrap-stress count: adjacent
    smoothed-p^2 phase steps > 2 rad, i.e. spots where a half-cycle
    ambiguity could have slipped in."""
    n_blk = len(p) // 1000
    freqs = np.zeros(n_blk)
    for i in range(n_blk):
        z = p[i * 1000:(i + 1) * 1000] ** 2
        Z = np.fft.fftshift(np.fft.fft(z * np.hanning(len(z)), 8192))
        fax = np.fft.fftshift(np.fft.fftfreq(8192, d=1e-3))
        freqs[i] = fax[np.argmax(np.abs(Z))] / 2.0
    f_of_t = np.repeat(freqs, 1000)
    if len(f_of_t) < len(p):
        f_of_t = np.pad(f_of_t, (0, len(p) - len(f_of_t)), mode="edge")
    phi = 2 * np.pi * np.cumsum(f_of_t[: len(p)]) * 1e-3
    p1 = p * np.exp(-1j * phi)
    z = p1 ** 2
    k = 51
    zs = np.convolve(z, np.ones(k) / k, mode="same")
    dstep = np.angle(zs[1:] * np.conj(zs[:-1]))
    stress = int(np.sum(np.abs(dstep) > 2.0))
    ref = 0.5 * np.unwrap(np.angle(zs))
    return (phi + ref) / (2 * np.pi), stress


def stage_gps(only_prn=None):
    birds = _load_cache("joint_gps_prn*.json")
    n1 = int(round(FS * 1e-3))
    spc = FS / CODE_RATE
    times = _grid_times()
    for b in birds:
        prn = b["prn"]
        if only_prn and prn != only_prn:
            continue
        t0w = time.time()
        ctr = b["fd"]
        phis, mets, dopps = [], [], []
        for t0 in times:
            x = load_seg(PATH, FS, t0, SEG_GPS)
            r = gps_acquire(x, FS, [prn],
                            np.arange(ctr - 375, ctr + 376, 125.0),
                            N_NONCOH_GPS)[prn]
            phis.append(r.get("code_phase_f", r["code_phase"]))
            mets.append(r["metric"])
            dopps.append(r["dopp"])
            if r["metric"] > 2.0:
                ctr = r["dopp"]
        phis = np.array(phis, float)
        mets = np.array(mets)
        dopps = np.array(dopps)
        good = mets > 1.8
        # GLOBAL-GRID FRAME: phi is measured against its own buffer start
        # (floor(t*FS)); that reference is only phase-coherent on the file's
        # integer-millisecond comb. Adding the buffer start's offset within
        # the code-period grid gives a coordinate that moves at the true
        # code drift rate for ANY sample time (the fractional snapshot
        # epochs jumped by the sub-ms comb offset otherwise).
        off = np.floor(times * FS).astype(np.int64) % n1
        ph = phis + off
        # unwrap mod 1 code period toward the Doppler-predicted drift
        for i in range(1, len(ph)):
            pred = ph[i - 1] + (-dopps[i] / 1540.0 * spc) \
                * (times[i] - times[i - 1])
            ph[i] -= np.round((ph[i] - pred) / n1) * n1
        code_poly = np.polyfit(times[good], ph[good], 3)
        fd_poly = np.polyfit(times[good], dopps[good], 2)
        code_rms = float(np.std(ph[good] - np.polyval(code_poly,
                                                      times[good])))
        print(f"[gps] PRN{prn}: {good.sum()}/{len(times)} code points, "
              f"cubic fit rms {code_rms:.2f} samp, fd(mid) "
              f"{np.polyval(fd_poly, 530.0):+.1f} Hz", flush=True)
        p = prompt_stream_poly(prn, code_poly, fd_poly, TA, TB - TA - 0.4)
        resid, stress = carrier_resid_cycles(p)
        t_th = TA + (np.arange(len(p)) + 0.5) * 1e-3
        theta = np.polyval(np.polyint(fd_poly), t_th) + resid
        amp = np.abs(p)
        print(f"[gps] PRN{prn}: carrier {len(p)} ms, unwrap-stress steps "
              f"{stress}, prompt |p| mean/std {np.mean(amp):.0f}/"
              f"{np.std(amp):.0f} [{time.time()-t0w:.0f} s]", flush=True)
        np.savez(LAB / f"hatch_gps_prn{prn}.npz",
                 times=times, phi_samp=phis, mets=mets, dopps=dopps,
                 code_poly=code_poly, fd_poly=fd_poly, t_th=t_th[::1],
                 theta=theta, stress=stress, code_fit_rms=code_rms)
    return 0


# -------------------------------------------------------- Galileo carrier
def stage_gal(only_prn=None):
    birds = _load_cache("joint_gal_prn*.json")
    codes_b, codes_c = load_codes()
    mm = np.memmap(PATH, dtype=np.int16, mode="r")
    times = _grid_times()
    for b in birds:
        prn = b["prn"]
        if only_prn and prn != only_prn:
            continue
        t0w = time.time()
        rep = {prn: boc_replica(codes_c[prn], FS)}
        x = load_seg(PATH, FS, GAL_T0, 40 * T_COH + 0.01)
        r = gal_e1.acquire(x, FS, rep, np.arange(-5000, 5001, 125.0),
                           40)[prn]
        print(f"[gal] PRN{prn}: acq metric {r['metric']:.2f} "
              f"dopp {r['dopp']:+.0f} Hz", flush=True)
        tr = track_prn(mm, FS, prn, r["dopp"],
                       int(GAL_T0 * FS) + r["code_phase"],
                       TB + 0.8 - GAL_T0,
                       boc_chips(codes_b[prn]), boc_chips(codes_c[prn]))
        t_th = tr["ptrs"] / FS
        theta = tr["phs"] / (2 * np.pi)
        # code snapshots on the same grid/conventions as joint_fix's snap
        fdp = b["fd_poly"]
        phis, mets, dopps = [], [], []
        for t0 in times:
            xa = load_seg(PATH, FS, t0, 40 * T_COH + 0.01)
            fd = float(np.polyval(fdp, t0))
            rr = gal_e1.acquire(xa, FS, rep,
                                np.arange(fd - 250, fd + 251, 125.0),
                                40)[prn]
            phis.append(rr.get("code_phase_f", rr["code_phase"]))
            mets.append(rr["metric"])
            dopps.append(rr["dopp"])
        np.savez(LAB / f"hatch_gal_prn{prn}.npz",
                 times=times, phi_samp=np.array(phis, float),
                 mets=np.array(mets), dopps=np.array(dopps),
                 fd_poly=np.array(fdp), t_th=t_th, theta=theta,
                 lock=tr["lock"], cn0=tr["cn0"])
        print(f"[gal] PRN{prn}: {sum(m > 1.8 for m in mets)}/{len(times)} "
              f"code points, PLL lock min {tr['lock'][2:].min():.3f} "
              f"[{time.time()-t0w:.0f} s]", flush=True)
    return 0


# ------------------------------------------------------------- smoothing
T_MID = 530.0                    # reference time for the global-rate term


def _cmc_series(times, phi_ms, good, t_th, theta, per, d_eff, mu_ms=0.0):
    """Code-minus-carrier, branch-unwrapped mod the code period at the
    measurement's effective (centroid) time. mu_ms [ms of delay per s] is
    the GLOBAL code-vs-carrier clock-scale split (see stage_smooth): the
    capture's code clock and carrier clock scales differ by ~20 ppb, so
    raw CMC drifts by c*20e-9 ~ 6 m/s identically on every bird. That
    common instrumental rate is folded into the carrier prediction."""
    t_eff = times + d_eff
    P = -np.interp(t_eff, t_th, theta) / K_CY_MS + mu_ms * (t_eff - T_MID)
    cmc = np.mod(phi_ms - P, per)
    last = None
    for i in range(len(cmc)):
        if not good[i]:
            continue
        if last is not None:
            cmc[i] -= np.round((cmc[i] - last) / per) * per
        last = cmc[i]
    return cmc, P


def _flag_cmc(times, cmc, good):
    """Outliers vs a rolling median; remaining big steps = slips (segment
    boundaries). Returns (outlier mask, seg id per point, sigma_m, n_slip)."""
    gi = np.where(good)[0]
    c = cmc[gi]
    n = len(c)
    rmed = np.array([np.median(c[max(0, i - 10):min(n, i + 11)])
                     for i in range(n)])
    dev = c - rmed
    sig = 1.4826 * np.median(np.abs(dev - np.median(dev))) + 1e-12
    out_g = np.abs(dev - np.median(dev)) > 5 * sig
    outlier = np.zeros(len(cmc), bool)
    outlier[gi[out_g]] = True
    # slips: steps between consecutive clean points
    ok = gi[~out_g]
    seg = np.zeros(len(cmc), int)
    sid = 0
    n_slip = 0
    for a, bb in zip(ok, ok[1:]):
        if np.abs(cmc[bb] - cmc[a]) > max(8 * sig, 1e-12):
            sid += 1
            n_slip += 1
        seg[bb] = sid
    for i in range(1, len(seg)):                 # propagate to flagged pts
        if seg[i] == 0 and i not in ok:
            seg[i] = seg[i - 1]
    return outlier, seg, float(sig * MS_M), n_slip


def _smooth_at(T, times, cmc, P_at, good, outlier, seg, w=W_HATCH):
    """Hatch value at epoch T: carrier prediction + causal window-mean CMC
    from the segment T lives in. Returns (phi_ms mod-free, n_used)."""
    use = good & ~outlier & (times <= T + 1e-6) & (times > T - w)
    if not use.any():
        return None, 0
    sT = seg[np.where(good & ~outlier & (times <= T + 1e-6))[0][-1]]
    use &= seg == sT
    return float(P_at + np.mean(cmc[use])), int(use.sum())


def stage_smooth():
    snap_raw = json.loads((LAB / "joint_snap.json").read_text())
    snap = {}
    stats = {"per_bird": {}, "window_s": W_HATCH, "cadence_s": CAD}
    for ei in sorted(snap_raw, key=int):
        snap[ei] = {"T": snap_raw[ei]["T"], "gps": {}, "gal": {}}

    birds = []
    for sysname, per, d_eff in (("gps", 1.0, D_EFF_GPS),
                                ("gal", 4.0, D_EFF_GAL)):
        n_per = int(round(FS * per * 1e-3))
        for f in sorted(LAB.glob(f"hatch_{sysname}_prn*.npz")):
            d = np.load(f)
            times = d["times"]
            # global-grid frame (see stage_gps): buffer-relative phi plus
            # the buffer start's offset within the code-period sample grid
            off = np.floor(times * FS).astype(np.int64) % n_per
            birds.append(dict(sysname=sysname, per=per, d_eff=d_eff,
                              n_per=n_per,
                              prn=int(f.stem.split("prn")[1]),
                              times=times,
                              phi_ms=(d["phi_samp"] + off) / FS * 1e3,
                              good=d["mets"] > 1.8,
                              t_th=d["t_th"], theta=d["theta"],
                              stress=int(d["stress"]) if "stress" in d
                              else None))
    # ---- pass 1: raw CMC drift per bird -> the GLOBAL clock-scale split.
    # Every bird (both constellations) shows the same c * ~20 ppb CMC rate:
    # the capture's code-clock scale vs carrier-clock scale disagree by a
    # constant. Bird-independent = instrumental, so calibrate it ONCE.
    drifts = []
    for b in birds:
        cmc, _ = _cmc_series(b["times"], b["phi_ms"], b["good"],
                             b["t_th"], b["theta"], b["per"], b["d_eff"])
        outl, _, _, _ = _flag_cmc(b["times"], cmc, b["good"])
        gg = b["good"] & ~outl
        b["raw_drift"] = float(np.polyfit(b["times"][gg], cmc[gg], 1)[0])
        drifts.append(b["raw_drift"])
    mu_ms = float(np.median(drifts))
    stats["cmc_common_rate_m_per_s"] = mu_ms * MS_M
    stats["code_vs_carrier_scale_split_ppb"] = -mu_ms * 1e6
    print(f"[smooth] GLOBAL code-vs-carrier rate: {mu_ms*MS_M:+.3f} m/s "
          f"(= {-mu_ms*1e6:+.2f} ppb clock-scale split), bird spread "
          f"{np.std(drifts)*MS_M*1000:.1f} mm/s", flush=True)

    # ---- pass 2: calibrated CMC -> slips/outliers -> Hatch at the epochs
    for b in birds:
        sysname, per, d_eff, prn = (b["sysname"], b["per"], b["d_eff"],
                                    b["prn"])
        times, good = b["times"], b["good"]
        cmc, P = _cmc_series(times, b["phi_ms"], good, b["t_th"], b["theta"],
                             per, d_eff, mu_ms)
        outlier, seg, sig_m, n_slip = _flag_cmc(times, cmc, good)
        gg = good & ~outlier
        dr = np.polyfit(times[gg], cmc[gg], 1)[0] * MS_M   # residual m/s
        det = cmc[gg] - np.polyval(np.polyfit(times[gg], cmc[gg], 1),
                                   times[gg])
        stats["per_bird"][f"{sysname.upper()}{prn}"] = {
            "n_points": int(good.sum()), "n_outlier": int(outlier.sum()),
            "n_slip": n_slip, "code_sigma_m": sig_m,
            "cmc_drift_m_per_s": float(dr),
            "cmc_detrended_rms_m": float(np.std(det) * MS_M),
            "unwrap_stress": b["stress"]}
        for ei in snap:
            T = snap[ei]["T"]
            t_eff = T + d_eff
            P_at = -np.interp(t_eff, b["t_th"], b["theta"]) / K_CY_MS \
                + mu_ms * (t_eff - T_MID)
            val, n_used = _smooth_at(T, times, cmc, P_at, good,
                                     outlier, seg)
            raw_ent = snap_raw[ei][sysname].get(str(prn))
            if val is None or raw_ent is None:
                continue
            # back to the snapshot convention: subtract the epoch buffer's
            # comb offset (raw snap phi is buffer-relative at floor(T*FS))
            off_T = float((np.floor(T * FS).astype(np.int64) % b["n_per"])
                          / FS * 1e3)
            # metric gate stays the RAW epoch metric -> same bird set
            snap[ei][sysname][str(prn)] = {
                "metric": raw_ent["metric"],
                "phi_ms": float(np.mod(val - off_T, per)),
                "n_smooth": n_used,
                "phi_raw_ms": raw_ent["phi_ms"]}
        s = stats["per_bird"][f"{sysname.upper()}{prn}"]
        print(f"[smooth] {sysname.upper()}{prn}: {s['n_points']} pts, "
              f"code sigma {s['code_sigma_m']:.1f} m, outliers "
              f"{s['n_outlier']}, slips {s['n_slip']}, resid CMC drift "
              f"{s['cmc_drift_m_per_s']*1000:+.1f} mm/s, detrended rms "
              f"{s['cmc_detrended_rms_m']:.1f} m", flush=True)
    (LAB / "joint_snap_smoothed.json").write_text(json.dumps(snap, indent=1))
    (LAB / "hatch_stats.json").write_text(json.dumps(stats, indent=1))
    print("[smooth] -> lab_local/joint_snap_smoothed.json, hatch_stats.json")
    return 0


# ------------------------------------------------------------- ISB forensics
def _isb_trend(res, tag):
    rows = [(r["T"], r[tag]["isb_ns"]) for r in res["per_epoch"]
            if tag in r and r[tag].get("isb_ns") is not None]
    t = np.array([x for x, _ in rows])
    y = np.array([x for _, x in rows])
    A = np.vstack([t - t.mean(), np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    sig = np.sqrt(np.sum(resid ** 2) / max(len(t) - 2, 1)
                  / np.sum((t - t.mean()) ** 2))
    return float(coef[0]), float(sig), float(coef[1]), t, y


def stage_isb():
    raw = json.loads((LAB / "fix_result_joint.json").read_text())
    smf = LAB / "fix_result_smoothed.json"
    sm = json.loads(smf.read_text()) if smf.exists() else None
    gps = _load_cache("joint_gps_prn*.json")
    gal = _load_cache("joint_gal_prn*.json")
    snap = json.loads((LAB / "joint_snap.json").read_text())

    print("=" * 72)
    print("ISB SLOW-TREND FORENSICS")
    print("=" * 72)
    sl, ssig, mean_isb, tt, yy = _isb_trend(raw, "healthy")
    print(f"observed (raw, healthy 6G+2E):  {sl:+.4f} +- {ssig:.4f} ns/s, "
          f"mean {mean_isb:+.1f} ns")
    sl14, ssig14, m14, *_ = _isb_trend(raw, "with14")
    print(f"observed (raw, incl PRN14):     {sl14:+.4f} +- {ssig14:.4f} ns/s,"
          f" mean {m14:+.1f} ns")
    if sm:
        sls, ssigs, ms, *_ = _isb_trend(sm, "healthy")
        print(f"observed (SMOOTHED, healthy):   {sls:+.4f} +- {ssigs:.4f} "
              f"ns/s, mean {ms:+.1f} ns")
        sls4, ssigs4, ms4, *_ = _isb_trend(sm, "with14")
        print(f"observed (SMOOTHED, incl 14):   {sls4:+.4f} +- {ssigs4:.4f} "
              f"ns/s, mean {ms4:+.1f} ns")

    # ---- (a)/(b): anchor-fit slopes + residual drifts per bird
    print("\n(a)/(b) anchor fits (file->SV time): slope-1 and residual trend")
    fits = {}
    for b in gps + gal:
        sysname = "GAL" if "fd_poly" in b else "GPS"
        ft = np.array(b["anchors_ft"])
        st = np.array(b["anchors_st"])
        aa, bb = np.polyfit(ft, st, 1)
        fits[(sysname, b["prn"])] = (aa, bb)
        resid = st - (aa * ft + bb)
        m = len(ft) // 2                     # split-half slope drift: the
        dhalf = (np.polyfit(ft[m:], st[m:], 1)[0]   # honest drift metric (a
                 - np.polyfit(ft[:m], st[:m], 1)[0])  # linear fit's residual
        print(f"  {sysname}{b['prn']:<3d} slope-1 {aa-1.0:+11.3e}   fit rms "
              f"{np.std(resid)*1e6:7.2f} us   half-slope drift "
              f"{dhalf:+9.2e} over {ft.max()-ft.min():.0f} s")
    print("  -> BUT: fits only pick the integer code-period count N0;"
          "\n     t_tx = N0*T_code + measured phase. A coarse-time drift"
          "\n     cannot move the pseudorange until it jumps a whole period.")
    # N0 margin: worst distance of coarse error from the per/2 cliff
    print("\n  N0 rounding margins at the far epochs (slope-1-fit "
          "extrapolation x -fd/f):")
    for b in gps:
        fd = b["fd"]
        err_ms = abs(fd) / FL1 * (765.0 - 424.0) * 1e3
        print(f"  GPS{b['prn']:<3d} |coarse err| at T=765: {err_ms:6.3f} ms "
              f"vs 0.500 ms cliff "
              f"{'-> integer hop, absorbed by offs search' if err_ms > 0.5 else '(safe)'}")

    # ---- (d): fs-error grid slide
    print(f"\n(d) fs error {TCXO_PPB:.2f} ppb (TCXO cal):")
    naive = TCXO_PPB                  # ppb == ns/s of code-clock slide
    print(f"  code-clock slide = {naive:.2f} ns/s = "
          f"{naive*1e-3:.3f} us per 1000 s of capture")
    print(f"  if that slide entered the ISB whole: {naive:+.2f} ns/s; "
          f"observed {sl:+.4f} ns/s -> off by x{naive/max(abs(sl),1e-9):.0f}")
    print(f"  AND mechanism-blocked: fs error rides identically in BOTH "
          f"systems' phase\n  measurements (same sampler) -> receiver-clock "
          f"unknown; anchors couple N0-only.")
    # only surviving fs path: phase-scale error delta*phi (phi < 4 ms)
    dphi_gal = np.mean([abs(np.polyval(b["fd_poly"], 535.0)) for b in gal]) \
        / K_CY_MS                     # ms of code phase per s
    print(f"  surviving path (delta * d(phi)/dt, 4 ms vs 1 ms ranges): "
          f"~{TCXO_PPB*1e-9*dphi_gal*1e6:.1e} ns/s -- negligible")

    # ---- (c): centroid-asynchrony projection through the epoch geometry
    print("\n(c) noncoherent-centroid asynchrony projection "
          f"(GPS +{D_EFF_GPS*1e3:.1f} ms vs GAL +{D_EFF_GAL*1e3:.1f} ms):")
    fdmods = {}
    for b in gps:
        f = LAB / f"hatch_gps_prn{b['prn']}.npz"
        if f.exists():
            fdmods[("GPS", b["prn"])] = np.load(f)["fd_poly"]
    for b in gal:
        fdmods[("GAL", b["prn"])] = np.array(b["fd_poly"])
    ephs = {("GPS", b["prn"]): b["eph"] for b in gps}
    ephs.update({("GAL", b["prn"]): b["eph"] for b in gal})
    pred = []
    for r_ep in raw["per_epoch"]:
        if "healthy" not in r_ep:
            continue
        T = r_ep["T"]
        x0 = np.array(r_ep["healthy"]["x5"])
        ei = [k for k in snap if abs(snap[k]["T"] - T) < 1e-6][0]
        rows_A, rows_e = [], []
        for sysname, d_eff, isgal in (("gps", D_EFF_GPS, 0.0),
                                      ("gal", D_EFF_GAL, 1.0)):
            for sp_prn, ent in snap[ei][sysname].items():
                prn = int(sp_prn)
                key = ("GAL" if isgal else "GPS", prn)
                if ent["metric"] < 2.0 or key not in fdmods:
                    continue
                if isgal and ephs[key].get("E1BHS", 0) != 0:
                    continue
                aa, bb = fits[key]
                sp = sat_ecef(ephs[key], aa * T + bb
                              - clock_corr(ephs[key], aa * T + bb))
                u = (x0 - sp) / np.linalg.norm(x0 - sp)
                fd = float(np.polyval(fdmods[key], T))
                rows_A.append(list(u) + [1.0, isgal])
                rows_e.append(-LAM_L1 * fd * d_eff)
        A = np.array(rows_A)
        e = np.array(rows_e)
        est, *_ = np.linalg.lstsq(A, e, rcond=None)
        pred.append((T, -est[4] / C * 1e9))
    pt = np.array([x for x, _ in pred])
    py = np.array([x for _, x in pred])
    psl = np.polyfit(pt, py, 1)[0]
    print(f"  predicted ISB offset from asynchrony: mean {py.mean():+.1f} ns,"
          f" trend {psl:+.4f} ns/s")
    print(f"  (includes the TCXO clock-rate term: c*delta*dT = "
          f"{C*TCXO_PPB*1e-9*(D_EFF_GAL-D_EFF_GPS):+.1f} m = "
          f"{TCXO_PPB*1e-9*(D_EFF_GAL-D_EFF_GPS)*1e9:+.1f} ns before "
          f"geometry projection)")

    # ---- (e): code-vs-carrier divergence (iono 2x + code multipath)
    hs = LAB / "hatch_stats.json"
    if hs.exists():
        st = json.loads(hs.read_text())["per_bird"]
        gald = [v["cmc_drift_m_per_s"] for k, v in st.items()
                if k.startswith("GAL") and k != "GAL14"]
        gpsd = [v["cmc_drift_m_per_s"] for k, v in st.items()
                if k.startswith("GPS")]
        dd = np.mean(gald) - np.mean(gpsd)
        print(f"\n(e) CMC divergence rates (code-specific drift = 2x iono + "
              f"multipath):")
        for k, v in sorted(st.items()):
            print(f"  {k:<6s} {v['cmc_drift_m_per_s']*1000:+7.2f} mm/s   "
                  f"detrended rms {v['cmc_detrended_rms_m']:5.1f} m")
        print(f"  mean GAL(healthy) - mean GPS = {dd*1000:+.2f} mm/s = "
              f"{dd/C*1e9:+.4f} ns/s of code-specific ISB drift")
    print("=" * 72)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["gps", "gal", "smooth", "solve", "isb"])
    ap.add_argument("--prn", type=int, default=None)
    a = ap.parse_args()
    if a.stage == "gps":
        return stage_gps(a.prn)
    if a.stage == "gal":
        return stage_gal(a.prn)
    if a.stage == "smooth":
        return stage_smooth()
    if a.stage == "solve":
        return joint_fix.stage_solve("joint_snap_smoothed.json",
                                     "fix_result_smoothed.json")
    return stage_isb()


if __name__ == "__main__":
    sys.exit(main())
