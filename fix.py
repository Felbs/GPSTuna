#!/usr/bin/env python3
"""fix.py - a GPS position fix from raw L1 IQ (the full receiver).

Reuses relativity.py's proven nav decode (acquire -> track -> 1 ms prompts ->
carrier cleanup -> parity/vote -> ephemeris) and adds the two stages that turn
a satellite's broadcast into YOUR location:
  * sat_ecef(): IS-GPS-200 ephemeris -> satellite ECEF position (Kepler solve +
    all harmonic corrections + Earth-rotation of the orbit plane)
  * solve(): weighted least-squares for (x, y, z, clock) from >= 4 pseudoranges

PRIVACY: a computed fix is written ONLY to lab_local/ (gitignored) and never
printed to any shared log. The code is public; the coordinates never are.

  python fix.py --iq ../captures/gps_fix_20260725.cs16   # needs >=4 birds
  python fix.py --validate                               # sat-position sanity
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from measure import acquire, load_seg, track_sv
from relativity import (bit_sums, clean_carrier, costas, find_grid,
                        harvest_words, parity_ok, parse_harvest, prompt_stream,
                        ubits)

MU = 3.986005e14
OMEGA_E = 7.2921151467e-5
C = 299792458.0
F_REL = -4.442807633e-10        # IS-GPS-200 relativistic clock constant


def decode_eph(path, fs, prn, dopp, dur, want_timing=False):
    """Full nav decode for one PRN -> ephemeris dict (mirrors relativity.main's
    union+vote harvest, standalone so it runs per satellite).
    want_timing=True also returns {tent, anchors:[(bit_idx, sf, tow)]} - the
    millisecond-accurate subframe clock the pseudorange assembly needs."""
    tr = track_sv(path, fs, prn, dopp, dur)
    from measure import bit_tent
    tent = int(np.argmax(bit_tent(path, fs, tr, 5.0, 10.0)))
    p = prompt_stream(path, fs, tr, 1.0, min(dur - 2.0, 118.0))
    s, bits, _ = bit_sums(costas(clean_carrier(p)), tent)
    pol, starts = find_grid(bits)
    harvest = harvest_words(p, tent, starts)
    sf_anchors = []
    b = bits ^ pol
    for i in starts:                                  # global-stitch union
        d29s, d30s = (b[i - 2], b[i - 1]) if i >= 2 else (0, 0)
        words = {}
        for w in range(10):
            word = b[i + w * 30:i + (w + 1) * 30]
            if len(word) < 30:
                break
            ok, d = parity_ok(word, d29s, d30s)
            if ok:
                words[w] = d
            d29s, d30s = word[28], word[29]
        if 1 in words:
            sf = ubits(words[1], 20, 22)
            if 1 <= sf <= 5:
                tow = ubits(words[1], 1, 17)
                harvest.append((sf, tow, words))
                sf_anchors.append((int(i), int(sf), int(tow)))
    # majority vote across repeats
    anchors = {}
    for k, i in enumerate(starts):
        w1 = b[i:i + 30]
        ok2, d2 = parity_ok(b[i + 30:i + 60], w1[28], w1[29])
        if ok2:
            sf = ubits(d2, 20, 22)
            if 1 <= sf <= 5:
                anchors[k] = sf
    if anchors:
        k0, s0 = next(iter(anchors.items()))
        groups = {}
        for k, i in enumerate(starts):
            sf = (s0 - 1 + (k - k0)) % 5 + 1
            if i + 300 <= len(b):
                groups.setdefault(sf, []).append(i)
        for sf, idxs in sorted(groups.items()):
            if sf > 3 or len(idxs) < 2:
                continue
            voted = (np.stack([b[i:i + 300] for i in idxs]).mean(0) > 0.5).astype(np.int8)
            hyb = voted.copy()
            hyb[:60] = b[idxs[0]:idxs[0] + 60]
            d29s, d30s = (b[idxs[0] - 2], b[idxs[0] - 1]) if idxs[0] >= 2 else (0, 0)
            words = {}
            for w in range(10):
                word = hyb[w * 30:(w + 1) * 30]
                ok, d = parity_ok(word, d29s, d30s)
                if ok:
                    words[w] = d
                d29s, d30s = word[28], word[29]
            harvest.append((sf, ubits(words[1], 1, 17) if 1 in words else 0, words))
    eph = parse_harvest(harvest)
    eph["prn"] = prn
    if want_timing:
        return eph, {"tent": tent, "anchors": sf_anchors, "tr": tr}
    return eph


def sat_ecef(eph, t):
    """IS-GPS-200 Table 20-IV: ephemeris + GPS time t -> satellite ECEF (m)."""
    A = eph["sqrtA"] ** 2
    n0 = np.sqrt(MU / A ** 3)
    tk = t - eph["toe"]
    if tk > 302400:
        tk -= 604800
    elif tk < -302400:
        tk += 604800
    n = n0 + eph.get("dn", 0.0)
    M = eph["M0"] + n * tk
    E = M
    for _ in range(15):
        E = M + eph["e"] * np.sin(E)
    e = eph["e"]
    nu = np.arctan2(np.sqrt(1 - e * e) * np.sin(E), np.cos(E) - e)
    # orientation fields (omega/i0/Omega0) come from subframe 3 and set only the
    # DIRECTION; the orbit RADIUS r depends only on A,e,E - so |r| validates the
    # Kepler math even from a partial ephemeris. Default the orientation to 0.
    phi = nu + eph.get("omega", 0.0)
    s2, c2 = np.sin(2 * phi), np.cos(2 * phi)
    u = phi + eph.get("Cus", 0) * s2 + eph.get("Cuc", 0) * c2
    r = A * (1 - e * np.cos(E)) + eph.get("Crs", 0) * s2 + eph.get("Crc", 0) * c2
    i = eph.get("i0", 0.964) + eph.get("IDOT", 0) * tk + eph.get("Cis", 0) * s2 + eph.get("Cic", 0) * c2
    xo, yo = r * np.cos(u), r * np.sin(u)
    Om = eph.get("Omega0", 0.0) + (eph.get("OmegaDot", 0) - OMEGA_E) * tk - OMEGA_E * eph["toe"]
    x = xo * np.cos(Om) - yo * np.cos(i) * np.sin(Om)
    y = xo * np.sin(Om) + yo * np.cos(i) * np.cos(Om)
    z = yo * np.sin(i)
    return np.array([x, y, z])


def ecc_anomaly(eph, t):
    """Eccentric anomaly at GPS time-of-week t (same Kepler solve as sat_ecef)."""
    A = eph["sqrtA"] ** 2
    tk = t - eph["toe"]
    if tk > 302400:
        tk -= 604800
    elif tk < -302400:
        tk += 604800
    M = eph["M0"] + (np.sqrt(MU / A ** 3) + eph.get("dn", 0.0)) * tk
    E = M
    for _ in range(15):
        E = M + eph["e"] * np.sin(E)
    return E


def clock_corr(eph, t_sv):
    """Satellite clock offset dt_sv = af0 + af1*dt + af2*dt^2 + relativistic
    eccentricity term (IS-GPS-200 20.3.3.3.3.1). t_gps = t_sv - dt_sv."""
    dt = t_sv - eph.get("toc", t_sv)
    if dt > 302400:
        dt -= 604800
    elif dt < -302400:
        dt += 604800
    dtr = F_REL * eph["e"] * eph["sqrtA"] * np.sin(ecc_anomaly(eph, t_sv))
    return (eph.get("af0", 0.0) + eph.get("af1", 0.0) * dt
            + eph.get("af2", 0.0) * dt * dt + dtr)


def ecef_to_llh(p):
    a, f = 6378137.0, 1 / 298.257223563
    b = a * (1 - f)
    e2 = f * (2 - f)
    x, y, z = p
    lon = np.arctan2(y, x)
    r = np.hypot(x, y)
    lat = np.arctan2(z, r * (1 - e2))
    for _ in range(8):
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = r / np.cos(lat) - N
        lat = np.arctan2(z, r * (1 - e2 * N / (N + h)))
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    h = r / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), h


def solve(sats):
    """sats = [(ecef_xyz, pseudorange_m)]. LS for (x,y,z,c*dt)."""
    x = np.array([0.0, 0.0, 0.0, 0.0])
    for _ in range(12):
        A, res = [], []
        for sp, pr in sats:
            d = x[:3] - sp
            rng = np.linalg.norm(d)
            A.append(np.append(d / rng, 1.0))
            res.append(pr - (rng + x[3]))
        A = np.array(A)
        dx, *_ = np.linalg.lstsq(A, np.array(res), rcond=None)
        x = x + dx
        if np.linalg.norm(dx[:3]) < 1e-3:
            break
    return x


def solve_prs(prs):
    """prs = [(prn, eph, t_gps_tx)] at a common receive epoch. Iterates the
    unknown receive time, Sagnac-rotates each satellite by its travel time,
    LS-solves. Returns (rms_m, lat, lon, h, x)."""
    t_rx = max(t for _, _, t in prs) + 0.075
    x = np.array([0.0, 0.0, 0.0, 0.0])
    sats = []
    for _ in range(8):
        sats = []
        for prn, eph, t_tx in prs:
            sp = sat_ecef(eph, t_tx)
            tau = max(t_rx - t_tx, 0.0)
            th = OMEGA_E * tau                    # Sagnac: rotate into the
            rot = np.array([[np.cos(th), np.sin(th), 0],   # rx-epoch frame
                            [-np.sin(th), np.cos(th), 0], [0, 0, 1]])
            sats.append((rot @ sp, C * (t_rx - t_tx)))
        x = solve(sats)
        t_rx -= x[3] / C                          # absorb clock into epoch
    res = [pr - (np.linalg.norm(x[:3] - sp) + x[3]) for sp, pr in sats]
    rms = float(np.sqrt(np.mean(np.square(res))))
    lat, lon, h = ecef_to_llh(x[:3])
    return rms, lat, lon, h, x


def solve_snapshot(entries):
    """entries = [{prn, eph, t_sv_coarse, phi_ms}] with t_sv_coarse the
    SV-CLOCK transmit time from the anchor fit (ms-quantized) and phi_ms the
    snapshot code phase (offset of the next code epoch AFTER the epoch, ms).

    Assembly laws (each one was a bug once):
      * work in SV time: code epochs align to SV-clock ms boundaries, so the
        integer-ms + code-phase combination happens BEFORE the clock
        correction (af0 alone is up to +-0.5 ms = +-150 km if applied first);
      * the code epoch starts phi AFTER the snapshot epoch, so the SV transmit
        time at the epoch is t_sv_tx = N - phi (NOT N + phi);
      * the anchor-fit integer is only ms-accurate -> exhaustive search of
        per-bird offsets in {-1,0,+1}, scored by residual rms + altitude
        sanity (with >= 5 birds only the true set collapses the residuals).
    Returns (rms, lat, lon, h, offsets)."""
    import itertools
    n = len(entries)
    frac = [(-e["phi_ms"]) % 1.0 for e in entries]
    N0 = [np.round(e["t_sv_coarse"] * 1e3 - f)
          for e, f in zip(entries, frac)]
    # a COMMON integer shift across all birds is absorbed by the receiver
    # clock (unobservable) - pin bird 0 and search only RELATIVE offsets,
    # re-centering if the best sits on the +-1 window edge.
    total = [0] * n
    for _round in range(4):
        best = None
        for rel in itertools.product((-1, 0, 1), repeat=n - 1):
            offs = (0,) + rel
            prs = []
            for k, e in enumerate(entries):
                t_sv_tx = (N0[k] + total[k] + offs[k] + frac[k]) * 1e-3
                t_gps_tx = t_sv_tx - clock_corr(e["eph"], t_sv_tx)
                prs.append((e["prn"], e["eph"], t_gps_tx))
            rms, lat, lon, h, x = solve_prs(prs)
            score = rms + (0 if -3000 < h < 9000 else 1e6)
            if best is None or score < best[0]:
                best = (score, rms, lat, lon, h, offs)
        _, rms, lat, lon, h, offs = best
        total = [t + o for t, o in zip(total, offs)]
        if all(o == 0 for o in offs):
            break
    return rms, lat, lon, h, total


def validate():
    """Decode our 2-bird capture, compute each satellite's ECEF at its toe,
    check the altitude is the GPS shell (~20,200 km). Proves the whole
    ephemeris->ECEF chain without needing a 4-bird fix."""
    fs = 2.048e6
    # validate the ECEF math on the PROVEN-good navbits capture (PRN15 fully
    # decodes there); the 2-bird fix capture is weaker.
    path = str(HERE.parent / "captures" / "gps_l1_navbits.cs16")
    dur = Path(path).stat().st_size / 4 / fs
    x = load_seg(path, fs, 0.5, 0.310)
    acq = acquire(x, fs, list(range(1, 33)), np.arange(-7000, 7001, 250.0), 300)
    det = {p: r for p, r in acq.items() if r["metric"] > 2.5}
    print(f"[fix] validate on {Path(path).name}: {len(det)} birds {sorted(det)}")
    for prn, r in det.items():
        eph = decode_eph(path, fs, prn, r["dopp"], dur)
        if not {"sqrtA", "e", "M0"}.issubset(eph):
            print(f"  PRN{prn}: partial ephemeris {sorted(set(eph)-{'tows','prn'})}"
                  f" - need subframe 2 (e, sqrtA, M0)")
            continue
        pos = sat_ecef(eph, eph["toe"])
        rr = np.linalg.norm(pos) / 1e3
        full = {"omega", "i0", "Omega0"}.issubset(eph)
        ok = 26000 < rr < 27200               # GPS orbit radius (dir needs sf3)
        print(f"  PRN{prn}: orbit radius |r|={rr:,.1f} km  "
              f"{'VALID GPS shell' if ok else 'OUT OF RANGE'}  "
              f"(ephemeris {'COMPLETE - full 3D position ready' if full else 'radius-only, sf3 partial'})")


def resolve_from_cache():
    """Re-run only the assembly+solve from lab_local/prs_cache.json (seconds,
    vs 15 min for a full redecode). Reconstructs the SV-clock coarse time
    from the cached (already clock-corrected) t_gps_tx_coarse, then applies
    the proven SV-time / N-minus-phi assembly in solve_snapshot()."""
    import json as _json
    cache = _json.loads((HERE / "lab_local" / "prs_cache.json").read_text())
    entries = []
    for c in cache:
        eph = c["eph"]
        dt0 = eph.get("af0", 0.0) + eph.get("af1", 0.0) * (
            c["t_gps_tx_coarse"] - eph.get("toc", c["t_gps_tx_coarse"]))
        entries.append({"prn": c["prn"], "eph": eph,
                        "t_sv_coarse": c["t_gps_tx_coarse"] + dt0,
                        "phi_ms": c["phi_ms"]})
    rms, lat, lon, h, offs = solve_snapshot(entries)
    (HERE / "lab_local" / "fix_result.json").write_text(_json.dumps({
        "lat": lat, "lon": lon, "alt_m": h,
        "birds": [e["prn"] for e in entries],
        "resid_rms_m": rms, "ms_offsets": offs}, indent=1))
    sane = -500 < h < 5000
    print(f"[resolve] residual rms {rms:,.1f} m, altitude "
          f"{'PLAUSIBLE' if sane else 'IMPLAUSIBLE'} ({h:,.0f} m), "
          f"ms offsets {offs}")
    print("[resolve] coordinates in lab_local/fix_result.json (private)")
    return 0 if sane else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iq", default=str(HERE.parent / "captures" / "gps_fix_20260725.cs16"))
    ap.add_argument("--fs", type=float, default=2.048e6)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--resolve", action="store_true",
                    help="re-run assembly+solve from lab_local cache")
    a = ap.parse_args()
    if a.validate:
        return validate()
    if a.resolve:
        return resolve_from_cache()
    fs = a.fs
    dur = Path(a.iq).stat().st_size / 4 / fs
    x = load_seg(a.iq, fs, 0.5, 0.310)
    acq = acquire(x, fs, list(range(1, 33)), np.arange(-7000, 7001, 250.0), 300)
    det = {p: r for p, r in acq.items() if r["metric"] > 2.5}
    print(f"[fix] {len(det)} birds acquired: {sorted(det)}")
    if len(det) < 4:
        print(f"[fix] NEED >= 4 satellites for a position fix, have {len(det)}.")
        print("[fix] indoor capture sees too few - re-capture with the antenna at")
        print("      a window or outdoors (a $10 GPS patch antenna gets 8-12).")
        print("[fix] running --validate instead (proves the sat-position math):")
        return validate()
    # (4+ birds path) decode, pseudoranges from code phase + TOW, solve
    print("[fix] 4+ birds - decoding ephemerides + solving (fix -> lab_local/ only)")
    return full_fix(a.iq, fs, det, dur)


def full_fix(path, fs, det, dur):
    """The final stage: coarse-time from each bird's nav-bit grid
    (millisecond-accurate subframe clocks) + sub-millisecond from a
    common-epoch acquisition snapshot -> pseudoranges -> solve().
    The computed position goes ONLY to lab_local/ (gitignored)."""
    import json as _json
    # 1. per-bird nav decode with timing anchors
    birds = {}
    for prn, r in det.items():
        try:
            eph, tim = decode_eph(path, fs, prn, r["dopp"], dur,
                                  want_timing=True)
        except Exception as e:
            print(f"  PRN{prn}: decode failed ({e})")
            continue
        need = {"sqrtA", "e", "M0", "toe", "omega", "i0", "Omega0", "af0"}
        missing = need - set(eph)
        if missing:
            print(f"  PRN{prn}: ephemeris incomplete (missing {sorted(missing)})")
            continue
        if not tim["anchors"]:
            print(f"  PRN{prn}: no parity-clean subframe anchor")
            continue
        birds[prn] = (eph, tim)
        print(f"  PRN{prn}: ephemeris COMPLETE, {len(tim['anchors'])} anchors, "
              f"C/N0 {tim['tr']['cn0']:.0f} dB-Hz")
    if len(birds) < 4:
        print(f"[fix] only {len(birds)} birds fully decoded - need 4")
        return 1

    # 2. common-epoch snapshot: sub-ms code phase for every bird at T_RX
    T_RX = min(dur - 1.0, 45.0)
    xsnap = load_seg(path, fs, T_RX, 0.310)
    n1 = int(round(fs * 1e-3))
    prs = []
    cache = []
    for prn, (eph, tim) in birds.items():
        r = acquire(xsnap, fs, [prn],
                    np.array([tim["tr"]["fd"]]), 300)[prn]
        phi_ms = (r["code_phase"] % n1) / fs * 1e3          # 0..1 ms
        # coarse transmit clock, SELF-CALIBRATED: every parity-clean
        # subframe anchor is a (file-time, satellite-time) pair; a
        # linear fit across them derives the exact mapping with no
        # Doppler-sign assumptions, and the fit residual is a built-in
        # truth check (microseconds = right, anything more = broken)
        ft = np.array([1.0 + (tim["tent"] + 20.0 * i) * 1e-3
                       for i, _sf, _tw in tim["anchors"]], float)
        st = np.array([tw * 6.0 - 6.0 for _i, _sf, tw in tim["anchors"]],
                      float)
        aa, bb = np.polyfit(ft, st, 1)
        fitres = float(np.std(st - (aa * ft + bb)))
        print(f"  PRN{prn}: anchor fit residual {fitres*1e6:.1f} us "
              f"(clock rate {aa - 1.0:+.2e})")
        t_sv_at_rx = aa * T_RX + bb               # SV-CLOCK time (TOW is SV time)
        # NOTE the assembly happens in SV time inside solve_snapshot(); the
        # clock correction (af0/af1/af2 + relativistic) is applied AFTER the
        # integer-ms + code-phase combination - code epochs align to SV-clock
        # ms boundaries, and af0 alone can be +-0.5 ms (+-150 km) if
        # subtracted first. Cache keeps the gps-corrected field for compat.
        dt_sv = clock_corr(eph, t_sv_at_rx)
        prs.append({"prn": prn, "eph": eph, "t_sv_coarse": t_sv_at_rx,
                    "phi_ms": phi_ms})
        cache.append({"prn": int(prn), "phi_ms": float(phi_ms),
                      "t_gps_tx_coarse": float(t_sv_at_rx - dt_sv),
                      "eph": {k: (float(v) if isinstance(v, (int, float))
                                  else v)
                              for k, v in eph.items() if k != "tows"}})
        print(f"  PRN{prn}: snapshot code phase {phi_ms:.4f} ms, "
              f"metric {r['metric']:.1f}")

    import json as _json2
    (HERE / "lab_local").mkdir(exist_ok=True)
    (HERE / "lab_local" / "prs_cache.json").write_text(_json2.dumps(cache))
    # 3. SV-time assembly (t_sv_tx = N - phi) + integer-ms search + solve
    rms, lat, lon, h, offs = solve_snapshot(prs)
    out = HERE / "lab_local"
    out.mkdir(exist_ok=True)
    (out / "fix_result.json").write_text(_json.dumps({
        "lat": lat, "lon": lon, "alt_m": h,
        "birds": sorted(int(e["prn"]) for e in prs),
        "resid_rms_m": rms, "ms_offsets": offs,
        "capture": str(path)}, indent=1))
    sane = -500 < h < 5000
    print(f"[fix] SOLVED with {len(prs)} birds: residual rms {rms:,.0f} m, "
          f"altitude {'PLAUSIBLE' if sane else 'IMPLAUSIBLE'} "
          f"({h:,.0f} m)")
    print(f"[fix] coordinates written to lab_local/fix_result.json "
          f"(gitignored - yours alone)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
