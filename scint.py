#!/usr/bin/env python3
"""scint.py - GPS scintillation: ionospheric turbulence measured from our
own L1 captures (task: the space-weather probe that reuses the GPS tracker).

Two standard indices per satellite, from the 1 ms prompt stream:
  S4      = std(P)/mean(P) over 60 s windows (amplitude scintillation)
  sigma_phi = std of detrended carrier phase (rad) per window (phase scint)
Quiet mid-latitude ionosphere: S4 < 0.1, sigma_phi < 0.1 rad. Anything above
~0.3 is a disturbed ionosphere (or local multipath - flagged honestly).

  python scint.py   # runs on every capture+PRN it can track
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from measure import acquire, load_seg, track_sv
from relativity import clean_carrier, prompt_stream

CAPS = [
    ("../captures/gps_l1_navbits.cs16", [15]),
    ("../captures/gps_fix_20260725.cs16", [5, 21]),
]
FS = 2.048e6


def indices(p, win_s=60.0):
    """Per-window S4 + sigma_phi from the 1 ms prompt stream."""
    n = int(win_s * 1000)
    out = []
    pc = clean_carrier(p)                      # detrended carrier for phase index
    for a in range(0, len(p) - n, n):
        P = np.abs(p[a:a + n]) ** 2
        s4 = float(np.std(P) / (np.mean(P) + 1e-12))
        ph = np.angle(pc[a:a + n] ** 2) / 2.0  # BPSK-free phase
        sig_phi = float(np.std(ph - np.median(ph)))
        out.append((round(a / 1000.0, 1), round(s4, 3), round(sig_phi, 3)))
    return out


def main():
    results = {}
    for cap, prns in CAPS:
        path = HERE / cap
        if not path.exists():
            continue
        dur = path.stat().st_size / 4 / FS
        x = load_seg(str(path), FS, 0.5, 0.310)
        for prn in prns:
            acq = acquire(x, FS, [prn], np.arange(-7000, 7001, 250.0), 300)[prn]
            if acq["metric"] < 2.5:
                print(f"{path.name} PRN{prn}: not acquired - skip")
                continue
            tr = track_sv(str(path), FS, prn, acq["dopp"], dur)
            p = prompt_stream(str(path), FS, tr, 1.0, min(dur - 2.0, 178.0))
            w = indices(p)
            s4s = [x2[1] for x2 in w]
            sps = [x2[2] for x2 in w]
            key = f"{path.name}:PRN{prn}"
            results[key] = {"cn0": round(tr["cn0"], 1), "windows": w,
                            "S4_med": round(float(np.median(s4s)), 3),
                            "sigphi_med": round(float(np.median(sps)), 3)}
            verdict = ("QUIET ionosphere" if np.median(s4s) < 0.15 and np.median(sps) < 0.15
                       else "DISTURBED (or local multipath - single-antenna caveat)")
            print(f"{key}: C/N0 {tr['cn0']:.1f}  S4_med={np.median(s4s):.3f}  "
                  f"sigma_phi_med={np.median(sps):.3f} rad  -> {verdict}")
    (HERE / "scint_results.json").write_text(json.dumps(results, indent=1))
    print("-> scint_results.json")


if __name__ == "__main__":
    main()
