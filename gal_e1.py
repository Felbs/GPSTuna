#!/usr/bin/env python3
"""Galileo E1 first light: find Galileo satellites in a GPS L1 capture.

Galileo E1 shares the 1575.42 MHz carrier with GPS L1, but everything
else is different: the E1-B (data) and E1-C (pilot) primary spreading
codes are 4092-chip *memory codes* (no LFSR generates them -- they are
published as hex tables), the code period is 4 ms (not 1 ms), and the
chips carry a BOC(1,1) square-wave subcarrier at 1.023 MHz that splits
the spectrum into two lobes at +-1.023 MHz.

KNOWN HANDICAP, stated up front: this tool runs on a 2.048 MS/s capture,
so Nyquist is +-1.024 MHz -- the BOC(1,1) main lobes sit exactly at the
band edge. The SDR's anti-alias filter shaves several dB off each lobe,
and the 1/11-power CBOC(6,1) component is gone entirely (~0.4 dB more).
With only 2 samples/chip the sharp BOC correlation peak also suffers up
to a few dB of sub-sample scalloping. Detection of strong birds still
works; weak ones will not show. A wider-bandwidth capture fixes this.

Code tables: parsed at runtime from gnss-sdr's system-parameters header
(Galileo_E1.h, GPL-3.0-or-later, https://github.com/gnss-sdr/gnss-sdr),
auto-downloaded into lab_local/ (gitignored). The tables are NOT
embedded here so this MIT repo does not redistribute GPL code tables.
PocketSDR (python/sdr_code.py) carries the same tables as an alternate
source. The values originate from the Galileo OS SIS ICD, Annex C.

Method (adapted from measure.py's GPS C/A machinery):
  * 4 ms coherent FFT circular correlation (one full primary code),
    x N noncoherent blocks. Noncoherent summing makes the E1-B data
    symbols (250 sps) and the E1-C 25-chip secondary code irrelevant:
    both only flip the sign of whole 4 ms blocks.
  * replicas carry the BOC(1,1) subcarrier and are built 8x oversampled
    then band-limited to fs by FFT truncation (exact for one circular
    code period), so the replica matches what the front end could pass.
  * detection metric = peak / second peak (2nd peak excludes +-1 chip
    around the max across all Dopplers), same as measure.py.
  * cross-check: re-acquire in a second window later in the file; a
    real bird reappears at a consistent Doppler.

Usage:
  python gal_e1.py --iq Z:/src/grid-atlas/captures/gps_attic.cs16
        [--fs 2048000] [--nblocks 40] [--t0 1.0] [--t1 70.0]
        [--selftest]     # synthetic -15 dB proof + sensitivity floor

Privacy: acquisition only -- prints PRN / Doppler / metric. No position
solving, nothing written outside the repo tree.
"""
import argparse
import re
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE / "lab_local"
FL1 = 1575.42e6
CODE_RATE = 1.023e6          # chips/s
CODE_LEN = 4092              # chips per primary code
T_COH = CODE_LEN / CODE_RATE  # 4 ms
THRESH = 2.5                 # peak / second-peak detection threshold
N_PRN = 36                   # Galileo constellation PRN slots

GNSS_SDR_URL = ("https://raw.githubusercontent.com/gnss-sdr/gnss-sdr/"
                "main/src/core/system_parameters/Galileo_E1.h")
# first 16 hex chars of PRN 1, E1-B / E1-C -- generator checksum
# (tiny excerpt from the Galileo OS SIS ICD Annex C tables, cited above)
CHECK_B1 = "F5D710130573541B"
CHECK_C1 = "B39340CA1C817D81"


# ------------------------------------------------------- code table loading
def _parse_table(text, name):
    """Extract one 50 x 1023-hex-char table from the gnss-sdr header."""
    m = re.search(name + r"\[[^\]]*\]\[\d+\]\s*=\s*\{(.*?)\};", text, re.S)
    if not m:
        raise SystemExit(f"could not find {name} in Galileo_E1.h")
    allhex = "".join(re.findall(r'"([0-9A-Fa-f]+)"', m.group(1)))
    if len(allhex) != 50 * 1023:
        raise SystemExit(f"{name}: expected {50*1023} hex chars, "
                         f"got {len(allhex)}")
    return [allhex[i * 1023:(i + 1) * 1023] for i in range(50)]


def _hex_to_chips(h):
    """1023 hex chars -> 4092 chips as +1/-1 (bit 1 -> -1)."""
    bits = np.unpackbits(np.frombuffer(bytes.fromhex("0" + h), np.uint8))
    return (1.0 - 2.0 * bits[4:]).astype(np.float64)   # drop pad nibble


def load_codes():
    """Return (codes_b, codes_c): {prn: 4092 chips of +-1}. Caches an npz."""
    npz = LAB / "galileo_e1_codes.npz"
    if npz.exists():
        d = np.load(npz)
        return ({p: d[f"b{p}"] for p in range(1, N_PRN + 1)},
                {p: d[f"c{p}"] for p in range(1, N_PRN + 1)})
    hdr = LAB / "Galileo_E1.h"
    if not hdr.exists():
        LAB.mkdir(exist_ok=True)
        print(f"downloading Galileo E1 code tables from gnss-sdr:\n  {GNSS_SDR_URL}")
        urllib.request.urlretrieve(GNSS_SDR_URL, hdr)
    text = hdr.read_text()
    tab_b = _parse_table(text, "GALILEO_E1_B_PRIMARY_CODE")
    tab_c = _parse_table(text, "GALILEO_E1_C_PRIMARY_CODE")
    if tab_b[0][:16].upper() != CHECK_B1 or tab_c[0][:16].upper() != CHECK_C1:
        raise SystemExit("code-table checksum failed vs ICD excerpt -- stop.")
    print("  code tables parsed, PRN 1 checksum vs OS SIS ICD Annex C: OK")
    codes_b = {p: _hex_to_chips(tab_b[p - 1]) for p in range(1, N_PRN + 1)}
    codes_c = {p: _hex_to_chips(tab_c[p - 1]) for p in range(1, N_PRN + 1)}
    np.savez_compressed(npz, **{f"b{p}": codes_b[p] for p in codes_b},
                        **{f"c{p}": codes_c[p] for p in codes_c})
    return codes_b, codes_c


# ------------------------------------------------------------------ replicas
def boc_replica(chips, fs, osf=8):
    """One 4 ms code period with BOC(1,1) subcarrier, band-limited to fs.

    Built at osf*fs, then band-limited by keeping only the central FFT
    bins (exact circular decimation for one code period). CBOC's 1/11-
    power BOC(6,1) component is ignored: its lobes at +-6.14 MHz are far
    outside a 2 MHz capture.
    """
    n_lo = int(round(fs * T_COH))
    n_hi = n_lo * osf
    t = np.arange(n_hi) / (fs * osf)
    ci = np.minimum((t * CODE_RATE).astype(np.int64), CODE_LEN - 1)
    half = (t * 2 * CODE_RATE).astype(np.int64) & 1     # sine-phased BOC(1,1)
    sig = chips[ci] * (1.0 - 2.0 * half)
    S = np.fft.fft(sig)
    S_lo = np.concatenate((S[: n_lo // 2], S[-n_lo // 2:])) / osf
    return np.fft.ifft(S_lo)


# --------------------------------------------------------------- acquisition
def acquire(x, fs, replicas, dopplers, n_noncoh):
    """Parallel code-phase search, 4 ms coherent x n_noncoh noncoherent.
    replicas: {prn: band-limited 4 ms replica}. Same metric as measure.py."""
    n4 = int(round(fs * T_COH))
    blocks = x[: n4 * n_noncoh].reshape(n_noncoh, n4)
    t = np.arange(n4) / fs
    rep_f = {p: np.conj(np.fft.fft(r)) for p, r in replicas.items()}
    excl = int(np.ceil(fs / CODE_RATE))                 # +-1 chip
    maps = {p: [] for p in replicas}
    for fd in dopplers:
        BF = np.fft.fft(blocks * np.exp(-2j * np.pi * fd * t)[None, :], axis=1)
        for p in rep_f:
            corr = np.fft.ifft(BF * rep_f[p][None, :], axis=1)
            maps[p].append((np.abs(corr) ** 2).sum(axis=0))
    out = {}
    for p in replicas:
        m = np.vstack(maps[p])
        di, ci = np.unravel_index(np.argmax(m), m.shape)
        mask = np.ones(m.shape[1], dtype=bool)
        for off in range(-excl, excl + 1):
            mask[(ci + off) % m.shape[1]] = False
        lam = m[di, ci] / np.median(m) - 1.0            # ~coherent SNR
        out[p] = dict(metric=float(m[di, ci] / m[:, mask].max()),
                      dopp=float(dopplers[di]), code_phase=int(ci),
                      peak_over_floor=float(m[di, ci] / np.median(m)),
                      cn0=float(10 * np.log10(max(lam, 1e-9))
                                + 10 * np.log10(1.0 / T_COH)))
    return out


def load_seg(path, fs, t0_s, dur_s):
    n0 = int(t0_s * fs) * 2
    n = int(dur_s * fs) * 2
    raw = np.memmap(path, dtype=np.int16, mode="r")[n0:n0 + n].astype(np.float32)
    x = raw[0::2] + 1j * raw[1::2]
    return x - x.mean()


# ------------------------------------------------------------------ selftest
def selftest(fs, codes_c, n_noncoh=40):
    """Synthetic E1-C at -15 dB SNR, then walk the SNR down to the floor."""
    prn_true, fd_true, ph_true = 11, +1830.0, 1234      # arbitrary truth
    n4 = int(round(fs * T_COH))
    rng = np.random.default_rng(42)
    rep = boc_replica(codes_c[prn_true], fs)
    one = np.roll(rep, ph_true)
    sec = 1.0 - 2.0 * rng.integers(0, 2, n_noncoh)      # secondary-code flips
    sig = np.concatenate([s * one for s in sec])
    t = np.arange(len(sig)) / fs
    sig = sig * np.exp(2j * np.pi * fd_true * t)
    sig /= np.sqrt(np.mean(np.abs(sig) ** 2))
    noise = (rng.standard_normal(len(sig))
             + 1j * rng.standard_normal(len(sig))) / np.sqrt(2)
    dops = np.arange(-7000, 7001, 125.0)
    reps = {p: boc_replica(codes_c[p], fs) for p in range(1, N_PRN + 1)}
    print(f"synthetic self-test: E1-C PRN{prn_true}, Doppler {fd_true:+.0f} Hz, "
          f"code phase {ph_true} samp, random 4 ms sign flips,")
    print(f"  4 ms x {n_noncoh} noncoherent, 36 PRNs x +-7 kHz / 125 Hz:")
    floor = None
    for snr_db in (-15, -18, -21, -24, -27, -30):
        x = sig * 10 ** (snr_db / 20) + noise
        res = acquire(x, fs, reps, dops, n_noncoh)
        best = max(res, key=lambda p: res[p]["metric"])
        r = res[best]
        right = (best == prn_true and abs(r["dopp"] - fd_true) <= 63
                 and min((r["code_phase"] - ph_true) % n4,
                         (ph_true - r["code_phase"]) % n4) <= 2)
        tag = "PASS" if right and r["metric"] > THRESH else \
              ("found but sub-threshold" if right else "LOST")
        print(f"  SNR {snr_db:+3d} dB (C/N0 {snr_db + 10*np.log10(fs):.1f} dB-Hz): "
              f"best PRN{best:2d} metric {r['metric']:5.2f} at {r['dopp']:+5.0f} Hz "
              f"phase {r['code_phase']:4d}  {tag}")
        if right and r["metric"] > THRESH:
            floor = snr_db
        else:
            break
    if floor is not None:
        print(f"  sensitivity floor (this integration): about {floor} dB SNR "
              f"= C/N0 ~{floor + 10*np.log10(fs):.0f} dB-Hz")
    print("  NOTE: the digital floor above excludes the front-end band-edge "
          "loss a real BOC signal suffers in this 2.048 MS/s capture.")


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iq", help="CS16 interleaved IQ at 1575.42 MHz")
    ap.add_argument("--fs", type=float, default=2.048e6)
    ap.add_argument("--nblocks", type=int, default=40,
                    help="noncoherent 4 ms blocks per window (default 40 = 160 ms)")
    ap.add_argument("--t0", type=float, default=1.0, help="first window start [s]")
    ap.add_argument("--t1", type=float, default=70.0,
                    help="cross-check window start [s]")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    codes_b, codes_c = load_codes()
    if a.selftest:
        selftest(a.fs, codes_c, a.nblocks)
        return
    if not a.iq:
        raise SystemExit("--iq required (or --selftest)")

    fs, nb = a.fs, a.nblocks
    dops = np.arange(-7000, 7001, 125.0)
    dur = nb * T_COH
    reps_c = {p: boc_replica(codes_c[p], fs) for p in range(1, N_PRN + 1)}
    reps_b = {p: boc_replica(codes_b[p], fs) for p in range(1, N_PRN + 1)}

    print(f"\nGalileo E1 acquisition on {Path(a.iq).name}: "
          f"4 ms x {nb} noncoherent ({dur*1e3:.0f} ms), "
          f"36 PRNs x +-7 kHz / 125 Hz, threshold {THRESH}")
    print("bandwidth caveat: BOC(1,1) lobes at +-1.023 MHz sit on this "
          "capture's Nyquist edge -- expect several dB of loss.\n")

    x = load_seg(a.iq, fs, a.t0, dur + 0.01)
    results = {}
    for comp, reps in (("E1-C pilot", reps_c), ("E1-B data", reps_b)):
        acq = acquire(x, fs, reps, dops, nb)
        results[comp] = acq
        det = {p: r for p, r in acq.items() if r["metric"] > THRESH}
        print(f"{comp} (window at t={a.t0:.1f} s):")
        for p, r in sorted(det.items(), key=lambda kv: -kv[1]["metric"]):
            print(f"  PRN{p:2d}  metric {r['metric']:5.2f}  "
                  f"Doppler {r['dopp']:+5.0f} Hz  code phase {r['code_phase']:4d} "
                  f"samp  C/N0 ~{r['cn0']:.1f} dB-Hz   DETECTED")
        if not det:
            best = max(acq, key=lambda p: acq[p]["metric"])
            print(f"  no PRN above threshold (best PRN{best} "
                  f"metric {acq[best]['metric']:.2f} at "
                  f"{acq[best]['dopp']:+.0f} Hz)")
        print()

    # ---- cross-check: detected PRNs must reappear at consistent Doppler
    det_c = {p: r for p, r in results["E1-C pilot"].items()
             if r["metric"] > THRESH}
    if det_c:
        print(f"cross-check window at t={a.t1:.1f} s (same PRNs, "
              "Doppler should move only slowly):")
        x2 = load_seg(a.iq, fs, a.t1, dur + 0.01)
        reps2 = {p: reps_c[p] for p in det_c}
        acq2 = acquire(x2, fs, reps2, dops, nb)
        for p in sorted(det_c):
            r1, r2 = det_c[p], acq2[p]
            dfd = r2["dopp"] - r1["dopp"]
            ok = r2["metric"] > THRESH and abs(dfd) <= 250
            print(f"  PRN{p:2d}  metric {r2['metric']:5.2f}  "
                  f"Doppler {r2['dopp']:+5.0f} Hz (moved {dfd:+4.0f} Hz in "
                  f"{a.t1 - a.t0:.0f} s)  "
                  f"{'CONFIRMED' if ok else 'NOT confirmed'}")
    else:
        print("no E1-C detections to cross-check. Given the band-edge "
              "handicap this is a possible honest outcome; run --selftest "
              "to verify the machinery.")


if __name__ == "__main__":
    main()
