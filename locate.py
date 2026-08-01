#!/usr/bin/env python3
"""locate.py - one command: capture the sky, count satellites, and (if enough
are visible) compute your position - written ONLY to lab_local/ (gitignored).

Run this with the antenna at a window or outdoors, or on a GPS patch antenna.
Indoors you'll see 1-3 weak satellites (not enough); with sky view, 6-12.

  python locate.py            # 180 s capture on Antenna A, then locate
  python locate.py --iq f.cs16   # locate from an existing capture

A 3D fix needs >= 4 satellites strong enough to decode their orbit (metric > 3.5,
C/N0 ~ 38+). This prints the count first so you know immediately whether the sky
view is good; if it is, the full pseudorange solve runs and the result stays
private in lab_local/fix_result.json.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from measure import acquire, load_seg
from fix import full_fix

FS = 2.048e6
LOCAL = HERE / "lab_local"


def capture(secs=180, antenna="Antenna A"):
    sys.path.insert(0, r"Z:\src\hamTuna\tools")
    import cw
    from SoapySDR import SOAPY_SDR_RX
    sdr, st = cw._open_sdr(antenna, FS)
    sdr.setFrequency(SOAPY_SDR_RX, 0, 1575.42e6)
    # Power an ACTIVE GPS antenna. These are boolean settings: SoapySDRPlay3
    # reads only the exact string "false" as off and treats everything else --
    # including "0" -- as ON, so the literals here are load-bearing.
    for key in ("biasT_ctrl", "biasT", "bias_tee"):
        try:
            sdr.writeSetting(key, "true")
        except Exception:
            pass
    try:
        time.sleep(0.5)
        iq = cw._grab(sdr, st, secs, FS, max_stall_s=60)
    finally:
        # Hand the coax back de-powered even if the grab throws. Otherwise the
        # next thing plugged into this port -- a passive antenna, a filter, a
        # borrowed radio -- meets DC that nothing turned off.
        for key in ("biasT_ctrl", "biasT", "bias_tee"):
            try:
                sdr.writeSetting(key, "false")
            except Exception:
                pass
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
    ap.add_argument("--antenna", default="Antenna B",
                    help="SDR antenna port (the one with bias-T if your GPS patch is active)")
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
    # full pipeline: nav decode with timing anchors -> common-epoch snapshot
    # code phases -> SV-time pseudorange assembly -> least-squares solve.
    # The fix is written ONLY to lab_local/fix_result.json (gitignored).
    rc = full_fix(path, FS, strong, dur, multi=5)
    if rc:
        print("[locate] not enough complete orbits - a longer capture (300 s)")
        print("         usually gets the remaining subframes; retry.")


if __name__ == "__main__":
    main()
