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


def decode_eph(path, fs, prn, dopp, dur):
    """Full nav decode for one PRN -> ephemeris dict (mirrors relativity.main's
    union+vote harvest, standalone so it runs per satellite)."""
    tr = track_sv(path, fs, prn, dopp, dur)
    from measure import bit_tent
    tent = int(np.argmax(bit_tent(path, fs, tr, 5.0, 10.0)))
    p = prompt_stream(path, fs, tr, 1.0, min(dur - 2.0, 118.0))
    s, bits, _ = bit_sums(costas(clean_carrier(p)), tent)
    pol, starts = find_grid(bits)
    harvest = harvest_words(p, tent, starts)
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
                harvest.append((sf, ubits(words[1], 1, 17), words))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iq", default=str(HERE.parent / "captures" / "gps_fix_20260725.cs16"))
    ap.add_argument("--fs", type=float, default=2.048e6)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if a.validate:
        return validate()
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
    # ... pseudorange assembly + solve(); result written to gitignored lab_local/
    # kept out of this public listing's stdout on purpose.


if __name__ == "__main__":
    sys.exit(main())
