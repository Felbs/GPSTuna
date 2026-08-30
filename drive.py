#!/usr/bin/env python3
"""drive.py - the road recorder: capture GPS raw data in a loop, no network,
no screen, no one watching.

The war-drive car has no hotspot, so there is no dashboard and no SSH once
it leaves the driveway. This records instead of solves: a loop of stamped
captures with a quick acquisition check between them, everything written to
lab_local/, until the disk guard says stop, --max-hours runs out, a signal
arrives, or a file named lab_local/STOP appears. Solving happens at home,
from the recordings, where the bugs can be fixed against the data that
showed them.

  python3 drive.py                     # 90 s captures until the disk is full
  python3 drive.py --secs 120 --max-hours 2
  python3 drive.py --secs 45           # half the road per capture, fewer birds
  python3 drive.py --no-check          # trust the antenna, record more

Capture length is the road-smear lever: at 50 mph a 90 s capture covers
2.0 km, and every epoch inside it is somewhere else. fix.py fits a track
through the epochs so that stretch becomes a trace rather than an error --
which is the cheaper fix, because shortening the capture also costs
satellites (a complete ephemeris needs subframes 1-3 to arrive clean).

Start it BEFORE leaving (while still on wifi); stop it from the driveway on
return, or just let it finish. Every cycle appends one line to
lab_local/drive_log.jsonl -- PRN counts and file sizes only, never
coordinates, so the log is safe to read anywhere.
"""
import argparse
import json
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from locate import FS, LOCAL, capture          # noqa: E402

EPH_SECS = 45.0          # below this, complete ephemerides start going missing

STOP_FILE = LOCAL / "STOP"
MISSION = LOCAL / "MISSION"
LOG = LOCAL / "drive_log.jsonl"


def sky_check(path):
    """One cheap acquisition pass on 0.4 s of the capture: how many strong
    birds, which ones. The point is the drive log saying 'antenna was alive
    at 19:42' -- or not -- before two more hours get recorded."""
    import numpy as np
    from measure import acquire, load_seg
    x = load_seg(path, FS, 0.5, 0.400)
    acq = acquire(x, FS, list(range(1, 33)), np.arange(-7000, 7001, 250.0),
                  400)
    strong = sorted(p for p, r in acq.items() if r["metric"] > 3.5)
    return strong


def hold_load(until, should_stop):
    """Wait out a pacing gap WITHOUT going idle. Drive #2 (8/28) died in
    the first --every gap, 2 min in: radio closed, bias-T off, CPU asleep,
    and the USB bank saw a load small enough to call 'charged' and shut
    off. Back-to-back recording on the same bank had survived 33 min the
    week before. So the gap keeps one core doing FFTs -- about a watt --
    which is enough draw to look like a computer, not a phone that has
    finished charging."""
    import numpy as np
    while time.time() < until and not should_stop():
        x = np.random.standard_normal(1 << 16).astype(np.complex64)
        for _ in range(50):
            x = np.fft.fft(x) / 256.0        # keep magnitudes bounded
            if time.time() >= until:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=90,
                    help="capture length per cycle (90 s ~ 0.74 GB)")
    ap.add_argument("--antenna", default="Antenna B")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly after this long, for a known drive")
    ap.add_argument("--every", type=float, default=0,
                    help="extra seconds to wait between cycles (default "
                         "none: back-to-back)")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the acquisition check between captures")
    ap.add_argument("--until", type=float, default=None,
                    help="absolute unix time to stop at; overrides "
                         "--max-hours. Set by the reboot-resume path so a "
                         "restarted mission keeps its original end time")
    a = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:                                            # noqa: BLE001
        pass

    # Between captures the handlers are ours (capture() reclaims them for
    # itself while the radio is open, then restores). Either way a signal
    # means the same thing: finish the bookkeeping and stop.
    stop = {"why": None}
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda sig, _f: stop.update(why=f"signal {sig}"))

    # Cap the CPU before anything else. Measured 8/23 in the backyard: the
    # battery bank's 5 V rail collapses under full 2.7 GHz 4-core load
    # (firmware latched under-voltage 0x50000; the Pi hard-reset twice),
    # and at 1.5 GHz it holds 4.84 V minimum through a whole solve. A road
    # recorder that browns out its own computer records nothing.
    try:
        import subprocess
        for c in range(4):
            subprocess.run(
                ["sudo", "-n", "tee",
                 f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_max_freq"],
                input=b"1500000", capture_output=True, timeout=5)
        print("[drive] CPU capped at 1.5 GHz (battery brownout guard)",
              flush=True)
    except Exception:                                            # noqa: BLE001
        print("[drive] WARNING: could not cap CPU frequency -- on battery "
              "the 5 V rail may sag", flush=True)

    STOP_FILE.unlink(missing_ok=True)          # a stale STOP is not an order
    t0 = time.time()
    deadline = (a.until if a.until
                else t0 + a.max_hours * 3600 if a.max_hours else None)

    # The mission marker: proof to the next boot that a drive was in
    # progress. On the road there is no screen and no network, so a
    # brownout reboot would otherwise end the recording in silence and the
    # first anyone knows is an empty log at the destination. crontab's
    # @reboot hook reads this file and relaunches with the SAME end time.
    # Written before the first capture and deleted on every clean stop, so
    # its mere existence after a boot means "you were interrupted".
    relaunch = [sys.executable, str(HERE / "drive.py"),
                "--secs", f"{a.secs:g}", "--antenna", a.antenna]
    if a.every:
        relaunch += ["--every", f"{a.every:g}"]
    if a.no_check:
        relaunch += ["--no-check"]
    if deadline:
        relaunch += ["--until", f"{deadline:.0f}"]
    MISSION.write_text(json.dumps(
        {"started": t0, "deadline": deadline, "argv": relaunch}) + "\n")
    per_cycle_gb = a.secs * FS * 4 / 1e9
    print(f"[drive] road recorder: {a.secs:.0f} s captures "
          f"({per_cycle_gb:.2f} GB each)"
          + (f", stopping after {a.max_hours:g} h" if a.max_hours else
             ", until the disk guard says stop"), flush=True)
    # What a capture length COSTS, both ways, so the choice is made before
    # the drive rather than discovered in the numbers afterwards.
    print("[drive] at speed one capture is a stretch of road, not a point: "
          + ", ".join(f"{mph:.0f} mph = {a.secs * mph * 0.44704 / 1000:.1f} km"
                      for mph in (30, 50, 70))
          + f" per {a.secs:.0f} s capture", flush=True)
    if a.secs < EPH_SECS:
        # Drive #3 (8/29): 3 of 28 ninety-second captures could not complete
        # four ephemerides ("only 3 birds fully decoded - need 4"); the
        # solver's own advice on those was "a longer capture (300 s) usually
        # gets the remaining subframes". Shorter trades birds for smear, and
        # a capture with three birds is worth nothing at any smear.
        print(f"[drive] WARNING: {a.secs:.0f} s is below {EPH_SECS:.0f} s -- "
              f"a full ephemeris needs subframes 1-3 (18 s) to arrive "
              f"parity-clean, and 3 of 28 captures on the 8/29 drive missed "
              f"a 4th bird even at 90 s. fix.py now fits a track through the "
              f"epochs, which removes the smear WITHOUT costing birds; "
              f"shorten the capture only if you would rather have more, "
              f"shorter stretches of road.", flush=True)

    n = 0
    while stop["why"] is None:
        if deadline and time.time() >= deadline:
            stop["why"] = "max-hours reached"
            break
        if STOP_FILE.exists():
            stop["why"] = "STOP file"
            break
        n += 1
        cyc_t0 = time.time()
        try:
            path = capture(a.secs, a.antenna)
        except SystemExit as e:
            # capture() exits with a message, not a code: the disk guard's
            # refusal, "found no radio", a stream that never started. Pass
            # it through verbatim -- guessing "disk guard" for all of them
            # is how a knocked-out antenna cable reads as a full card, and
            # this log is the only witness the drive has.
            stop["why"] = ("aborted mid-capture" if e.code == 130
                           else " ".join(str(e.code).split()))
            n -= 1                     # it was counted before it was taken
            break
        entry = {"t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "cycle": n,
                 "file": Path(path).name,
                 "secs": a.secs,
                 "gb": round(Path(path).stat().st_size / 1e9, 3)}
        if not a.no_check:
            try:
                strong = sky_check(path)
                entry["strong"] = len(strong)
                entry["prns"] = strong
                print(f"[drive] cycle {n}: {len(strong)} strong bird(s) "
                      f"{strong}", flush=True)
                if not strong:
                    print("[drive] WARNING: zero strong birds -- check the "
                          "antenna if this persists", flush=True)
            except Exception as e:                               # noqa: BLE001
                # a broken check must not stop the recording
                entry["check_error"] = str(e)
        entry["cycle_s"] = round(time.time() - cyc_t0, 1)
        with LOG.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        if a.every and stop["why"] is None:
            hold_load(time.time() + a.every,
                      lambda: stop["why"] is not None or STOP_FILE.exists())

    STOP_FILE.unlink(missing_ok=True)
    MISSION.unlink(missing_ok=True)            # a clean stop is not a crash
    hrs = (time.time() - t0) / 3600
    print(f"[drive] stopped ({stop['why'] or 'done'}): {n} capture(s) in "
          f"{hrs:.2f} h. Solve at home with\n"
          f"        python3 locate.py --iq lab_local/<capture>.cs16",
          flush=True)


if __name__ == "__main__":
    main()
