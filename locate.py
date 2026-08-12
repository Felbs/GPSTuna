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

# A 300 s capture followed by a long decode looks HUNG when stdout is a file
# rather than a terminal, because Python block-buffers it and nothing appears
# until the process exits. Anyone redirecting this to a log -- which is the
# natural thing to do with a run this long -- deserves to see it working.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:                                                # noqa: BLE001
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from measure import acquire, load_seg, require_capture
from fix import full_fix

FS = 2.048e6
LOCAL = HERE / "lab_local"


def _open_sdr(antenna, fs):
    """Open the first SoapySDR device at `fs`, return (device, active stream).

    Self-contained on purpose: this used to import a helper from a sibling
    project by absolute path, which meant `locate.py` could only ever run on
    one machine.
    """
    try:
        import SoapySDR
        from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
    except ImportError:
        raise SystemExit(
            "\nCapturing from a radio needs SoapySDR, which is not importable "
            "from this Python.\n\n"
            "  * install SoapySDR plus a driver for your radio, or\n"
            "  * record 1575.42 MHz yourself with any SDR tool and pass the "
            "file:\n"
            "        python locate.py --iq your_capture.cs16\n\n"
            "See the README (\"Getting a capture\") for the recording "
            "settings.\n")

    devices = SoapySDR.Device.enumerate()
    if not devices:
        raise SystemExit(
            "\nSoapySDR is installed but found no radio.\n"
            "Check that the device is plugged in and not held open by another "
            "program.\n")
    sdr = SoapySDR.Device(devices[0])
    sdr.setSampleRate(SOAPY_SDR_RX, 0, fs)
    if antenna:
        # Read it BACK. A silently-swallowed setAntenna means capturing on the
        # wrong port and blaming the sky.
        try:
            sdr.setAntenna(SOAPY_SDR_RX, 0, antenna)
        except Exception as e:
            print(f"[locate] could not select antenna {antenna!r}: {e}")
        got = sdr.getAntenna(SOAPY_SDR_RX, 0)
        if got != antenna:
            ports = list(sdr.listAntennas(SOAPY_SDR_RX, 0))
            print(f"[locate] WARNING: asked for antenna {antenna!r} but the "
                  f"radio reports {got!r}. Available: {ports}")
    st = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(st)
    return sdr, st


def _grab(sdr, st, secs, fs, max_stall_s=60):
    """Read `secs` of complex baseband. -> np.complex64"""
    want = int(secs * fs)
    out = np.empty(want, np.complex64)
    chunk = np.empty(8192, np.complex64)
    got = 0
    last_progress = time.time()
    while got < want:
        sr = sdr.readStream(st, [chunk], len(chunk), timeoutUs=1000000)
        n = getattr(sr, "ret", sr if isinstance(sr, int) else -1)
        if n > 0:
            n = min(n, want - got)
            out[got:got + n] = chunk[:n]
            got += n
            last_progress = time.time()
        elif time.time() - last_progress > max_stall_s:
            raise SystemExit(
                f"\nThe radio stopped delivering samples after {got/fs:.1f} s "
                f"({got}/{want}).\nThis is usually USB: try a short, direct "
                f"USB 3.0 cable and no hub.\n")
    return out


def capture(secs=180, antenna="Antenna A"):
    # _open_sdr carries the friendly ImportError, so open FIRST and only then
    # import the constants -- otherwise this line raises the raw traceback.
    sdr, st = _open_sdr(antenna, FS)
    from SoapySDR import SOAPY_SDR_RX
    sdr.setFrequency(SOAPY_SDR_RX, 0, 1575.42e6)
    # Power an ACTIVE GPS antenna. These are boolean settings: SoapySDRPlay3
    # reads only the exact string "false" as off and treats everything else --
    # including "0" -- as ON, so the literals here are load-bearing.
    powered = False
    for key in ("biasT_ctrl", "biasT", "bias_tee"):
        try:
            sdr.writeSetting(key, "true")
            # An unverified writeSetting is a HOPE. On this driver only
            # biasT_ctrl exists; the other two accept the write and read back
            # empty, which would look like success and quietly leave an active
            # antenna's LNA unpowered.
            if str(sdr.readSetting(key)).lower() == "true":
                powered = True
                print(f"[locate] bias-T ON via {key} (readback confirmed)")
        except Exception:
            pass
    if not powered:
        print("[locate] WARNING: could not confirm bias-T power. An ACTIVE GPS "
              "patch antenna needs it;\n"
              "         without it expect to acquire a few birds but fail to "
              "decode their orbits.")
    try:
        time.sleep(0.5)
        iq = _grab(sdr, st, secs, FS, max_stall_s=60)
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
    require_capture(path)
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
