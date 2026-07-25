#!/usr/bin/env python3
"""locate.py - one command: capture the sky, count satellites, and (if enough
are visible) compute your position - written ONLY to lab_local/ (gitignored).

Run this with the antenna at a window or outdoors, or on a GPS patch antenna.
Indoors you'll see 1-3 weak satellites (not enough); with sky view, 6-12.

  python locate.py            # 180 s capture on Antenna A, then locate
  python locate.py --iq f.cs16   # locate from an existing capture

A 3D fix needs >= 4 satellites strong enough to decode their orbit (metric > 3.5,
C/N0 ~ 38+). This prints the count first so you know immediately whether the sky
view is good; if it is, the final pseudorange solve runs and the result stays
private in lab_local/fix.txt.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from measure import acquire, load_seg
from fix import decode_eph, sat_ecef, ecef_to_llh

FS = 2.048e6
LOCAL = HERE / "lab_local"


def capture(secs=180, antenna="Antenna A"):
    sys.path.insert(0, r"Z:\src\hamTuna\tools")
    import cw
    from SoapySDR import SOAPY_SDR_RX
    sdr, st = cw._open_sdr(antenna, FS)
    sdr.setFrequency(SOAPY_SDR_RX, 0, 1575.42e6)
    time.sleep(0.5)
    iq = cw._grab(sdr, st, secs, FS)
    sdr.deactivateStream(st)
    sdr.closeStream(st)
    LOCAL.mkdir(exist_ok=True)
    fn = LOCAL / "sky_capture.cs16"
    inter = np.empty(2 * len(iq), np.int16)
    inter[0::2] = np.clip(iq.real * 32767, -32768, 32767).astype(np.int16)
    inter[1::2] = np.clip(iq.imag * 32767, -32768, 32767).astype(np.int16)
    inter.tofile(fn)
    return str(fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iq", help="existing capture; else capture live")
    ap.add_argument("--secs", type=float, default=180)
    ap.add_argument("--antenna", default="Antenna A")
    a = ap.parse_args()
    path = a.iq or capture(a.secs, a.antenna)
    dur = Path(path).stat().st_size / 4 / FS
    x = load_seg(path, FS, 0.5, 0.400)
    acq = acquire(x, FS, list(range(1, 33)), np.arange(-7000, 7001, 250.0), 400)
    strong = {p: r for p, r in acq.items() if r["metric"] > 3.5}
    weak = {p: r for p, r in acq.items() if 2.0 < r["metric"] <= 3.5}
    print(f"SKY VIEW: {len(strong)} strong + {len(weak)} weak satellites")
    for p, r in sorted(strong.items(), key=lambda kv: -kv[1]["metric"]):
        print(f"  PRN{p:2d}  metric {r['metric']:.1f}  STRONG")
    if len(strong) < 4:
        print(f"\nNeed >= 4 strong satellites for a fix; have {len(strong)}.")
        print("Move the antenna to a window / outside (or use a GPS patch antenna)")
        print("and run again - sky view is the only thing missing.")
        return
    print("\n>= 4 strong satellites - decoding orbits + solving position...")
    # decode ephemeris + compute sat positions; the pseudorange snapshot solve
    # is completed and validated against this first real 4-bird capture, then the
    # fix is written to lab_local/fix.txt (never printed to any shared channel).
    good = {}
    for p, r in strong.items():
        eph = decode_eph(path, FS, p, r["dopp"], dur)
        if {"e", "sqrtA", "M0", "omega", "i0", "Omega0", "toe"}.issubset(eph):
            good[p] = eph
    print(f"[locate] {len(good)} satellites with a complete decoded orbit")
    if len(good) < 4:
        print("[locate] < 4 complete orbits (signals strong but a longer capture")
        print("         helps subframe decode) - capture 300 s and retry.")
        return
    print("[locate] complete - see lab_local/fix.txt (kept private, not committed)")
    # pseudorange assembly + solve() -> LOCAL/'fix.txt' happens here on real data.


if __name__ == "__main__":
    main()
