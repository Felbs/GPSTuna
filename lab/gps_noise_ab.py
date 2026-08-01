"""GPS re-acquisition experiment: is it LNA power, or is it host noise?

BACKGROUND. GPS acquired 7 satellites from an attic capture on 7/27 and zero
today. Eliminated by measurement: the code (today's fix.py still recovers 7
birds from the archived capture), the port (SAW-filter sweep peaks 8.68x on
Antenna B), the sample rate and format (2.048 MHz cs16), and the signal level.

What was NOT eliminated, because the instrument was broken: LNA power. Every
bias-T toggle today wrote "1"/"0", and SoapySDRPlay3 parses that setting as
`if (value=="false") off; else ON;` -- so "0" meant ON and both halves of every
comparison were powered. We have never actually observed this antenna unpowered.
Fixed and pushed today (STVT 98a57ca), which is what makes this run possible.

HYPOTHESIS. The failure is conducted host noise at L1, not LNA power. The SDR
changed machines today; the spectrum went from 6 spur bins to 638 (1.95%) with
a 46.7 dB DC spike, on all three ports, with bias-T off -- the signature of
interference arriving through the host rather than the antenna. USB 3.0 is the
prime suspect.

PREDICTION, stated before the run so it can be wrong:
  * If LNA power was the problem -> bias-T ON acquires satellites, OFF does not.
  * If host noise is the problem -> BOTH fail, and both show the elevated spur
    count against the known-good baseline (median 84.5 dB, peak-median 18.6 dB,
    6 spur bins).
A third outcome -- ON acquires and the spurs are gone -- would mean something
today was transient, and that is worth knowing too.

Coordinates, if a fix ever solves, go to lab_local/ only. Never to a log.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import SoapySDR
SoapySDR.setLogLevel(SoapySDR.SOAPY_SDR_FATAL)
from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent / "lab_local"
FS = 2_048_000                 # the rate the working 7/27 capture used
L1 = 1_575_420_000
SECS = 90.0

# The known-good 7/27 attic capture, for comparison. These are the numbers the
# spectrum must approach for the front end to be considered healthy again.
BASELINE = dict(median_db=84.5, peak_minus_median_db=18.6, spur_bins=6)


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(HERE / "gps_noise_ab_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_biast(sdr, on):
    """Write the bias-T and CONFIRM it. An unverified write is a hope."""
    want = "true" if on else "false"
    sdr.writeSetting("biasT_ctrl", want)
    try:
        got = str(sdr.readSetting("biasT_ctrl")).strip().lower()
    except Exception:
        return None
    if got != want:
        log(f"  !! bias-T did not latch: wrote {want!r}, reports {got!r}")
    return got


def spectrum_stats(x, nfft=8192):
    """Noise-floor and spur census -- the discriminator for this experiment."""
    n = (len(x) // nfft) * nfft
    if n < nfft:
        return None
    seg = x[:n].reshape(-1, nfft)
    win = np.hanning(nfft).astype(np.float32)
    acc = np.zeros(nfft)
    for i in range(0, len(seg), max(1, len(seg) // 64)):     # subsample: 64 frames
        acc += np.abs(np.fft.fftshift(np.fft.fft(seg[i] * win))) ** 2
    psd_db = 10 * np.log10(acc / max(1, len(range(0, len(seg), max(1, len(seg) // 64)))) + 1e-30)
    med = float(np.median(psd_db))
    peak = float(np.max(psd_db))
    spurs = int(np.sum(psd_db > med + 20.0))
    return dict(median_db=med, peak_db=peak,
                peak_minus_median_db=peak - med,
                spur_bins=spurs, spur_frac=spurs / nfft,
                rms=float(np.sqrt(np.mean(np.abs(x) ** 2)) * 32768))


def grab(sdr, secs):
    st = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(st)
    need = int(FS * secs)
    out = np.empty(need, np.complex64)
    buf = np.empty(1 << 16, np.complex64)
    got = 0
    t0 = time.time()
    try:
        while got < need and time.time() - t0 < secs * 3 + 20:
            sr = sdr.readStream(st, [buf], len(buf), timeoutUs=int(1e6))
            if sr.ret > 0:
                k = min(sr.ret, need - got)
                out[got:got + k] = buf[:k]
                got += k
    finally:
        sdr.deactivateStream(st)
        sdr.closeStream(st)
    return out[:got]


def write_cs16(x, path):
    inter = np.empty(2 * len(x), np.int16)
    inter[0::2] = np.clip(x.real * 32767, -32768, 32767).astype(np.int16)
    inter[1::2] = np.clip(x.imag * 32767, -32768, 32767).astype(np.int16)
    inter.tofile(path)


def main():
    LOCAL.mkdir(exist_ok=True)
    log("=" * 68)
    log("GPS A/B: bias-T verified ON vs verified OFF, Antenna B, 2.048 MHz cs16")
    log(f"baseline to beat (7/27 known-good): {BASELINE}")

    sys.path.insert(0, "Z:/src/gr-radiotuna/tools")
    import radio_lock
    holder = radio_lock.Holder("gps_noise_ab",
                               "GPS acquisition A/B (bias-T verified)", 60,
                               wait_s=600)
    holder.__enter__()
    results = {}
    sdr = SoapySDR.Device("driver=sdrplay")
    try:
        sdr.setSampleRate(SOAPY_SDR_RX, 0, FS)
        sdr.setAntenna(SOAPY_SDR_RX, 0, "Antenna B")
        sdr.setFrequency(SOAPY_SDR_RX, 0, L1)
        try:
            sdr.setGainMode(SOAPY_SDR_RX, 0, False)
            sdr.setGain(SOAPY_SDR_RX, 0, "IFGR", 40)
            sdr.writeSetting("rfgain_sel", "4")
        except Exception:
            pass

        for on in (True, False):
            state = "ON" if on else "OFF"
            got = set_biast(sdr, on)
            log(f"\n--- bias-T {state}  (device reports {got!r}) ---")
            time.sleep(1.5)                       # let the LNA settle / collapse
            x = grab(sdr, SECS)
            log(f"  captured {len(x)/FS:.1f} s ({len(x)} samples)")
            st = spectrum_stats(x)
            if st is None:
                log("  !! too few samples to analyse")
                continue
            log(f"  rms {st['rms']:.1f}   median {st['median_db']:.1f} dB   "
                f"peak-median {st['peak_minus_median_db']:.1f} dB   "
                f"spur bins {st['spur_bins']} ({st['spur_frac']*100:.2f}%)")
            verdict = ("CLEAN - comparable to the known-good capture"
                       if st["spur_bins"] <= 60 else
                       "DIRTY - spur census far above the known-good 6 bins")
            log(f"  {verdict}")
            path = LOCAL / f"sky_biast_{state.lower()}.cs16"
            write_cs16(x, path)
            log(f"  wrote {path.name}")
            results[state] = dict(stats=st, capture=str(path), verdict=verdict)
    finally:
        try:
            set_biast(sdr, False)                 # never leave DC on the coax
        except Exception:
            pass
        del sdr
        holder.__exit__(None, None, None)
        log("\nbias-T off, device released, lock released")

    (LOCAL / "gps_noise_ab.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    log("\n" + "=" * 68)
    log("ACQUISITION: run locate.py against each capture (offline, no radio):")
    for state in ("ON", "OFF"):
        if state in results:
            log(f"  python locate.py --iq \"{results[state]['capture']}\"")
    log("=" * 68)


if __name__ == "__main__":
    main()
