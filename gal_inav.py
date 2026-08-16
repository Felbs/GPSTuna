#!/usr/bin/env python3
"""gal_inav.py - read Galileo's clock over the air: E1-B I/NAV page decode.

Sequel to gal_e1.py (which only *detected* Galileo birds). This tool
closes the loop from raw IQ to CRC-verified navigation pages:

  1. acquire E1-C (pilot) with gal_e1's BOC(1,1) machinery
  2. scalar tracking, pilot-aided: DLL (noncoherent early/late on E1-C)
     + Costas PLL, then find the 25-chip secondary code CS25 (100 ms
     period), wipe it off, and promote to a pure 4-quadrant PLL --
     the dataless pilot gives a clean carrier the data channel rides on
  3. E1-B prompt correlations at the same code timing -> 250 sym/s
     soft symbols (one symbol per 4 ms primary-code period)
  4. I/NAV frame chain per the OS SIS ICD (conventions cross-checked
     against gnss-sdr's telemetry decoder):
       - 10-symbol sync pattern 0101100000 every 250 symbols; the
         preamble correlation sign fixes each part's BPSK polarity
       - 240 symbols deinterleave (8 rows x 30 columns: received
         row-wise, read out column-wise: out[c*8+r] = in[r*30+c])
       - Viterbi k=7 r=1/2, G1=171o G2=133o, second branch inverted
         on air (so the received odd-index symbols get negated),
         64-state soft-decision trellis in numpy, tail-terminated
       - even(114) + odd(114) -> 228-bit page; CRC-24Q (0x1864CFB)
         over the first 196 bits vs the 24-bit CRC field
  5. parse words 0..6: IODnav, ephemeris scales, health, and the
     headline -- GST week number + time of week, read off the air,
     cross-checked against position in the capture across satellites.

Usage:
  python gal_inav.py --iq your_capture.cs16 --fs 4096000
        [--fs 4096000] [--prns 29,27,14] [--t0 1.0] [--dur 777]
        [--selftest]   # synthesize a full E1 signal end-to-end and
                       # require CRC-clean pages back out of it

Privacy: no position computation. Time/ephemeris parameters printed.
The capture is opened strictly read-only (np.memmap mode "r") and
streamed in 4 ms blocks -- the 12.8 GB file is never loaded whole.
"""
from measure import require_capture
import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gal_e1 import (CODE_LEN, CODE_RATE, FL1, T_COH, acquire, boc_replica,
                    load_codes, load_seg)

LAB = HERE / "lab_local"
CS25 = "0011100000001010110110010"        # E1-C secondary code (ICD 3.8.2)
SYNC = "0101100000"                        # I/NAV page-part sync pattern
G1_OCT, G2_OCT = 0o171, 0o133              # k=7 rate-1/2 FEC polynomials
CRC24_POLY = 0x1864CFB                     # CRC-24Q (RTCM/Galileo)
SPP = 250                                  # symbols per page part (1 s)
# GST epoch (WN=0, TOW=0) = 1999-08-22 00:00:00 GST = 23:59:47 UTC the
# 21st (GPS-UTC was 13 s then). UTC = epoch + WN*604800 + TOW - (dtLS-13).
GST_EPOCH_UTC = datetime(1999, 8, 21, 23, 59, 47, tzinfo=timezone.utc)

SEC = np.array([1.0 - 2.0 * int(c) for c in CS25])       # +-1 secondary
PRE = np.array([1.0 if c == "1" else -1.0 for c in SYNC])  # bit1 -> +1


# ------------------------------------------------------------ FEC machinery
def _trellis():
    """64-state tables: for each new state s', its two predecessor states
    and the expected symbol signs (bit1 -> +1) of each branch."""
    sp = np.arange(64)
    p0 = 2 * (sp & 31)                    # predecessor with LSB 0
    p1 = p0 + 1
    b = (sp >> 5) & 1                     # input bit that leads to s'
    v0 = (b << 6) | p0                    # 7-bit register [b_k .. b_k-6]
    v1 = (b << 6) | p1
    pop = np.array([bin(v).count("1") & 1 for v in range(128)])
    s1 = 2.0 * pop[np.arange(128) & G1_OCT] - 1.0   # sign of c1 per register
    s2 = 2.0 * pop[np.arange(128) & G2_OCT] - 1.0
    return p0, p1, v0, v1, s1, s2


TP0, TP1, TV0, TV1, TS1, TS2 = _trellis()


def viterbi120(soft240):
    """Soft-decision Viterbi over one page part (120 info+tail bits).
    soft240: deinterleaved symbols, second branch ALREADY un-inverted,
    mapping bit1 -> +1. Returns (bits[120], path_metric)."""
    r = soft240.reshape(120, 2)
    pm = np.full(64, -1e12)
    pm[0] = 0.0                                       # encoder starts flushed
    dec = np.zeros((120, 64), dtype=bool)
    for k in range(120):
        m0 = pm[TP0] + TS1[TV0] * r[k, 0] + TS2[TV0] * r[k, 1]
        m1 = pm[TP1] + TS1[TV1] * r[k, 0] + TS2[TV1] * r[k, 1]
        take1 = m1 > m0
        pm = np.where(take1, m1, m0)
        dec[k] = take1
    s = 0                                             # tail-terminated
    bits = np.zeros(120, dtype=np.int8)
    for k in range(119, -1, -1):
        bits[k] = (s >> 5) & 1
        s = TP1[s] if dec[k, s] else TP0[s]
    return bits, float(pm[0])


def conv_encode(bits):
    """Transmit-side encoder (selftest): returns 2N symbols c1,c2 with the
    Galileo second-branch inversion applied."""
    out = np.zeros(2 * len(bits), dtype=np.int8)
    s = 0
    pop = [bin(v).count("1") & 1 for v in range(128)]
    for k, b in enumerate(bits):
        v = (int(b) << 6) | s
        out[2 * k] = pop[v & G1_OCT]
        out[2 * k + 1] = pop[v & G2_OCT] ^ 1          # inverted branch
        s = (int(b) << 5) | (s >> 1)
    return out


def deinterleave(sym240):
    """8 rows x 30 cols: received row-wise, read column-wise."""
    return sym240.reshape(8, 30).T.ravel()


def interleave(sym240):
    """Transmit side (selftest): inverse of deinterleave."""
    return sym240.reshape(30, 8).T.ravel()


def crc24q(bits):
    r = 0
    for b in bits:
        r = ((r << 1) ^ (CRC24_POLY if ((r >> 23) ^ int(b)) & 1 else 0)) \
            & 0xFFFFFF
    return r


# ---------------------------------------------------------------- tracking
def boc_chips(code):
    """4092-chip code -> 8184 half-chips with sine-phased BOC(1,1)."""
    b = np.empty(2 * CODE_LEN)
    b[0::2] = code
    b[1::2] = -code
    return b


def track_prn(mm, fs, prn, fd0, start_samp, dur_s, bocB, bocC, d_el=0.25,
              bn_pll=12.0, verbose=True):
    """Pilot-aided scalar tracking. Returns dict with the E1-B soft-symbol
    stream (one per 4 ms block), per-second C/N0 and lock stats."""
    n_total = mm.size // 2
    dt = T_COH
    zeta = 0.707
    wn = bn_pll / (zeta + 1 / (4 * zeta)) * 2
    k1 = 2 * zeta * wn * dt
    k2 = (wn * dt) ** 2
    kd = 0.05                                        # DLL gain [chips/update]
    n2 = 2 * CODE_LEN

    # ---- open-loop fine frequency: 0.5 s of pilot prompts, FFT of p^2
    fd = fd0
    for _ in range(2):                               # two refinement passes
        ptr, chip_off, phase = start_samp, 0.0, 0.0
        cr = CODE_RATE * (1 + fd / FL1)
        ps = []
        for k in range(125):
            n = int(round((CODE_LEN - chip_off) / cr * fs))
            raw = mm[2 * ptr: 2 * (ptr + n)]
            x = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
            tt = np.arange(n) / fs
            y = x * np.exp(-1j * (phase + 2 * np.pi * fd * tt))
            ct = chip_off + tt * cr
            ps.append((y * bocC[(ct * 2).astype(np.int64) % n2]).sum())
            phase += 2 * np.pi * fd * n / fs
            chip_off = chip_off + n * cr / fs - CODE_LEN
            ptr += n
        z = np.array(ps) ** 2
        Z = np.fft.fftshift(np.fft.fft(z * np.hanning(len(z)), 8192))
        fax = np.fft.fftshift(np.fft.fftfreq(8192, d=dt))
        df = fax[np.argmax(np.abs(Z))] / 2.0
        fd += df
    if verbose:
        print(f"  fine freq: {fd0:+.0f} -> {fd:+.1f} Hz")

    # ---- closed loop
    ptr, chip_off, phase = start_samp, 0.0, 0.0
    n_blocks = int(dur_s / dt)
    pB = np.zeros(n_blocks, dtype=np.complex128)
    pCw = np.zeros(n_blocks, dtype=np.complex128)    # secondary-wiped pilot
    ephi = np.zeros(n_blocks)
    fds = np.zeros(n_blocks)
    ptrs = np.zeros(n_blocks, dtype=np.int64)        # file sample @ block start
    phs = np.zeros(n_blocks)                         # carrier NCO phase (rad)
                                                     # @ block start: continuous
                                                     # integrated Doppler for
                                                     # carrier-smoothing use
    sec_off, sec_sign, sec_frac = None, 1.0, 0.0
    N_COSTAS = 375                                   # 1.5 s of Costas first
    pilot_raw = np.zeros(N_COSTAS, dtype=np.complex128)
    t_wall = time.time()
    k = 0
    while k < n_blocks:
        cr = CODE_RATE * (1 + fd / FL1)
        n = int(round((CODE_LEN - chip_off) / cr * fs))
        if ptr + n > n_total:
            break
        ptrs[k] = ptr                                # timing anchor for pseudoranges
        phs[k] = phase                               # carrier phase at this sample
        raw = mm[2 * ptr: 2 * (ptr + n)]
        x = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
        tt = np.arange(n) / fs
        y = x * np.exp(-1j * (phase + 2 * np.pi * fd * tt))
        ct = chip_off + tt * cr
        hc = (ct * 2).astype(np.int64) % n2
        pc = (y * bocC[hc]).sum()
        pb = (y * bocB[hc]).sum()
        E = np.abs((y * bocC[((ct + d_el) * 2).astype(np.int64) % n2]).sum())
        L = np.abs((y * bocC[((ct - d_el + CODE_LEN) * 2).astype(np.int64)
                             % n2]).sum())
        # advance NCOs
        phase += 2 * np.pi * fd * n / fs
        chip_off = chip_off + n * cr / fs - CODE_LEN
        ptr += n
        # DLL
        if E + L > 0:
            chip_off += kd * (E - L) / (E + L)
        # PLL
        if k < N_COSTAS:                             # Costas (sign-blind)
            pilot_raw[k] = pc
            e = np.arctan(pc.imag / pc.real) if abs(pc.real) > 1e-12 else 0.0
            w = sec_sign
        else:                                        # pure PLL, sec wiped
            w = sec_sign * SEC[(k + sec_off) % 25]
            pw = pc * w
            e = np.arctan2(pw.imag, pw.real)
        phase += k1 * e
        fd += k2 * e / (2 * np.pi * dt)
        ephi[k] = e
        fds[k] = fd
        pB[k] = pb
        pCw[k] = pc * (w if sec_off is not None else 1.0)
        k += 1
        # ---- secondary-code sync at the end of the Costas stage
        if k == N_COSTAS:
            seg = pilot_raw[125:N_COSTAS].real       # skip pull-in transient
            best = (0.0, 0, 1.0)
            for off in range(25):
                sc = (seg * SEC[(np.arange(125, N_COSTAS) + off) % 25]).sum()
                if abs(sc) > abs(best[0]):
                    best = (sc, off, 1.0 if sc > 0 else -1.0)
            sec_off, sec_sign = best[1], best[2]
            match = np.mean(np.sign(seg) == sec_sign
                            * SEC[(np.arange(125, N_COSTAS) + sec_off) % 25])
            sec_frac = float(match)
            if verbose:
                print(f"  secondary code CS25: offset {sec_off}, polarity "
                      f"{int(sec_sign):+d}, sign match {match*100:.1f}% "
                      f"{'SYNC' if match > 0.85 else 'WEAK -- pilot not clean'}")
    n_blocks = k
    pB, pCw, ephi, fds = pB[:k], pCw[:k], ephi[:k], fds[:k]
    ptrs = ptrs[:k]
    phs = phs[:k]

    # ---- per-second quality
    nsec = n_blocks // SPP
    cn0 = np.zeros(nsec)
    lock = np.zeros(nsec)
    for i in range(nsec):
        w = pCw[i * SPP:(i + 1) * SPP]
        ps = np.mean(w.real) ** 2
        pn = 2 * np.var(w.imag) + 1e-12
        cn0[i] = 10 * np.log10(max(ps / pn, 1e-9) / dt)
        lock[i] = np.mean(np.cos(2 * ephi[i * SPP:(i + 1) * SPP]))
    c2, l2 = (cn0[2:], lock[2:]) if nsec > 3 else (cn0, lock)  # skip pull-in
    if verbose:
        print(f"  tracked {n_blocks} blocks ({n_blocks*dt:.0f} s) in "
              f"{time.time()-t_wall:.0f} s wall; "
              f"C/N0 mean {c2.mean():.1f} dB-Hz (min {c2.min():.1f}), "
              f"PLL lock mean {l2.mean():.3f} (min {l2.min():.3f})")
        print(f"  Doppler {fds[0]:+.1f} -> {fds[-1]:+.1f} Hz over the track")
    return dict(prn=prn, sym=pB.real, pB=pB, cn0=cn0, lock=lock, fds=fds,
                sec_off=sec_off, sec_frac=sec_frac, n_blocks=n_blocks,
                ptrs=ptrs, phs=phs)


# ------------------------------------------------------------- frame chain
def decode_part(sym250):
    """One page part: 10-sym preamble already stripped by caller? No --
    caller hands 250 symbols starting at the sync pattern, polarity fixed.
    Returns (bits120, metric_eff, tail_ok)."""
    d = deinterleave(sym250[10:250].copy())
    d[1::2] *= -1.0                                  # undo branch inversion
    bits, pm = viterbi120(d)
    eff = pm / (np.abs(d).sum() + 1e-12)
    return bits, eff, bool((bits[114:] == 0).all())


def find_parts(sym):
    """Preamble search. Returns (grid_phase, list of (start, polarity,
    strength), stats dict)."""
    c = np.correlate(sym, PRE, mode="valid")         # c[i] = sum sym[i+j]*PRE[j]
    scale = 10 * np.median(np.abs(sym)) + 1e-12
    cn = c / scale
    mods = np.zeros(SPP)
    for ph in range(SPP):
        mods[ph] = np.abs(cn[ph::SPP]).sum()
    ph0 = int(np.argmax(mods))
    starts = list(range(ph0, len(sym) - SPP + 1, SPP))
    strong = sum(1 for i in starts if abs(cn[i]) > 0.5)
    parts = [(i, 1.0 if cn[i] > 0 else -1.0, float(abs(cn[i])))
             for i in starts]
    # off-grid preamble-grade hits = spacing instability indicator
    # (0.9 = 10/10 symbol agreement; random data reaches 0.6 routinely)
    allhits = np.where(np.abs(cn) > 0.9)[0]
    offgrid = int(np.sum((allhits - ph0) % SPP != 0))
    return ph0, parts, dict(n_slots=len(starts), n_strong=strong,
                            offgrid_strong=offgrid,
                            grid_contrast=float(mods[ph0] /
                                                (np.median(mods) + 1e-12)))


def harvest_pages(sym):
    """Full chain: preamble grid -> per-part Viterbi -> even/odd pairing ->
    CRC-24Q. Returns (pages, stats). pages = list of dicts."""
    ph0, parts, pstats = find_parts(sym)
    decoded = []
    for i, pol, strength in parts:
        if i + SPP > len(sym):
            continue
        bits, eff, tail_ok = decode_part(pol * sym[i:i + SPP])
        decoded.append(dict(i=i, bits=bits, eff=eff, tail_ok=tail_ok,
                            eo=int(bits[0]), ptype=int(bits[1]),
                            strength=strength))
    pages = []
    n_pair = n_crc = 0
    for a, b in zip(decoded, decoded[1:]):
        if b["i"] - a["i"] != SPP or a["eo"] != 0 or b["eo"] != 1:
            continue
        n_pair += 1
        page = np.concatenate([a["bits"][:114], b["bits"][:114]])
        ok = crc24q(page[:196]) == int("".join(str(x) for x in page[196:220]),
                                       2)
        if ok:
            n_crc += 1
            pages.append(dict(i=a["i"], page=page, alert=a["ptype"],
                              eff=(a["eff"] + b["eff"]) / 2))
    tails = np.mean([d["tail_ok"] for d in decoded]) if decoded else 0.0
    effs = np.mean([d["eff"] for d in decoded]) if decoded else 0.0
    stats = dict(**pstats, grid_phase=ph0, n_parts=len(decoded),
                 tail_ok_frac=float(tails), viterbi_eff=float(effs),
                 n_pairs=n_pair, n_crc=n_crc)
    return pages, stats


# ------------------------------------------------------------ word parsing
def _u(b, i, n):
    v = 0
    for k in range(i, i + n):
        v = (v << 1) | int(b[k])
    return v


def _s(b, i, n):
    v = _u(b, i, n)
    return v - (1 << n) if v >= (1 << (n - 1)) else v


def parse_word(page):
    """228-bit page -> (word_type, dict of fields). Word = Data(1/2)+Data(2/2)."""
    w = np.concatenate([page[2:114], page[116:132]])  # 128 bits
    wt = _u(w, 0, 6)
    f = {}
    if wt == 0:
        f["time_flag"] = _u(w, 6, 2)
        if f["time_flag"] == 2:
            f["WN"] = _u(w, 96, 12)
            f["TOW"] = _u(w, 108, 20)
    elif wt == 1:
        f["IODnav"] = _u(w, 6, 10)
        f["toe"] = _u(w, 16, 14) * 60
        f["M0_sc"] = _s(w, 30, 32) * 2**-31
        f["e"] = _u(w, 62, 32) * 2**-33
        f["sqrtA"] = _u(w, 94, 32) * 2**-19
    elif wt == 2:
        f["IODnav"] = _u(w, 6, 10)
        f["Omega0_sc"] = _s(w, 16, 32) * 2**-31
        f["i0_sc"] = _s(w, 48, 32) * 2**-31
        f["omega_sc"] = _s(w, 80, 32) * 2**-31
        f["IDOT_sc"] = _s(w, 112, 14) * 2**-43      # semicircles/s
    elif wt == 3:
        f["IODnav"] = _u(w, 6, 10)
        f["OmegaDot_sc"] = _s(w, 16, 24) * 2**-43   # semicircles/s
        f["dn_sc"] = _s(w, 40, 16) * 2**-43         # semicircles/s
        f["Cuc"] = _s(w, 56, 16) * 2**-29           # rad
        f["Cus"] = _s(w, 72, 16) * 2**-29           # rad
        f["Crc"] = _s(w, 88, 16) * 2**-5            # m
        f["Crs"] = _s(w, 104, 16) * 2**-5           # m
        f["SISA"] = _u(w, 120, 8)
    elif wt == 4:
        f["IODnav"] = _u(w, 6, 10)
        f["SVID"] = _u(w, 16, 6)
        f["Cic"] = _s(w, 22, 16) * 2**-29           # rad
        f["Cis"] = _s(w, 38, 16) * 2**-29           # rad
        f["toc"] = _u(w, 54, 14) * 60
        f["af0"] = _s(w, 68, 31) * 2**-34
        f["af1"] = _s(w, 99, 21) * 2**-46
        f["af2"] = _s(w, 120, 6) * 2**-59
    elif wt == 5:
        f["ai0"] = _u(w, 6, 11) * 2**-2
        f["ai1"] = _s(w, 17, 11) * 2**-8
        f["ai2"] = _s(w, 28, 14) * 2**-15
        f["BGD_E1E5a"] = _s(w, 47, 10) * 2**-32     # s
        f["BGD_E1E5b"] = _s(w, 57, 10) * 2**-32     # s (I/NAV E1 users: Eq 15)
        f["E5bHS"] = _u(w, 67, 2)
        f["E1BHS"] = _u(w, 69, 2)
        f["E5bDVS"] = _u(w, 71, 1)
        f["E1BDVS"] = _u(w, 72, 1)
        f["WN"] = _u(w, 73, 12)
        f["TOW"] = _u(w, 85, 20)
    elif wt == 6:
        f["A0"] = _s(w, 6, 32) * 2**-30
        f["A1"] = _s(w, 38, 24) * 2**-50
        f["dtLS"] = _s(w, 62, 8)
        f["TOW"] = _u(w, 105, 20)
    elif wt == 10:
        # GST-GPS conversion (GGTO), OS SIS ICD 5.1.8 Eq 21:
        # dt_systems = t_Galileo - t_GPS
        #            = A0G + A1G*(TOW - t0G + 604800*((WN - WN0G) mod 64))
        # all-ones in all four fields = GGTO not valid (ICD 5.1.8)
        f["ggto_valid"] = not (_u(w, 86, 16) == 0xFFFF
                               and _u(w, 102, 12) == 0xFFF
                               and _u(w, 114, 8) == 0xFF
                               and _u(w, 122, 6) == 0x3F)
        f["A0G"] = _s(w, 86, 16) * 2**-35           # s
        f["A1G"] = _s(w, 102, 12) * 2**-51          # s/s
        f["t0G"] = _u(w, 114, 8) * 3600             # s
        f["WN0G"] = _u(w, 122, 6)                   # GST week mod 64
    return wt, f


def gst_to_utc(wn, tow, dtls=18):
    return GST_EPOCH_UTC + timedelta(seconds=wn * 604800 + tow - (dtls - 13))


# ---------------------------------------------------------------- selftest
def selftest(fs):
    """Synthesize 8 s of E1 (pilot+data, BOC(1,1), CS25, real I/NAV framing
    with CRC) at 45 dB-Hz, write CS16, and demand CRC-clean pages back."""
    print("selftest: synthetic E1 PRN11, fd +1830 Hz, 45 dB-Hz per component")
    rng = np.random.default_rng(7)
    codes_b, codes_c = load_codes()
    prn, fd, wn_true, tow_true = 11, 1830.0, 1234, 333330

    def make_word(wt):
        w = np.zeros(128, dtype=np.int8)
        for k in range(6):
            w[k] = (wt >> (5 - k)) & 1
        if wt == 0:
            w[6:8] = [1, 0]                          # time flag = 2
            for k in range(12):
                w[96 + k] = (wn_true >> (11 - k)) & 1
            for k in range(20):
                w[108 + k] = (tow_true >> (19 - k)) & 1
        else:
            w[8:96] = rng.integers(0, 2, 88)
        return w

    def make_page(word):
        even = np.concatenate([[0, 0], word[:112]]).astype(np.int8)   # 114
        odd = np.zeros(114, dtype=np.int8)
        odd[0], odd[1] = 1, 0
        odd[2:18] = word[112:128]
        odd[18:82] = rng.integers(0, 2, 64)          # osnma+sar+spare junk
        crc = crc24q(np.concatenate([even, odd[:82]]))
        for k in range(24):
            odd[82 + k] = (crc >> (23 - k)) & 1
        return even, odd

    def part_symbols(half114):
        coded = conv_encode(np.concatenate([half114, np.zeros(6, np.int8)]))
        tx = interleave(1.0 - 2.0 * coded.astype(float))  # bit1 -> -1 on air
        pre = np.array([1.0 - 2.0 * int(c) for c in SYNC])
        return np.concatenate([pre, tx])

    syms = [part_symbols(h) for wt in (0, 2, 5, 0)
            for h in make_page(make_word(wt))]
    stream = np.concatenate([np.sign(rng.standard_normal(37))] + syms)

    n = int(7.9 * fs)
    tt = np.arange(n) / fs
    cr = CODE_RATE * (1 + fd / FL1)
    ct = tt * cr + 100.0                             # arbitrary code phase
    hc = (ct * 2).astype(np.int64)
    blk = hc // (2 * CODE_LEN)
    hci = hc % (2 * CODE_LEN)
    bocB, bocC = boc_chips(codes_b[prn]), boc_chips(codes_c[prn])
    symb = stream[np.minimum(blk, len(stream) - 1)]
    secc = SEC[blk % 25]
    a = np.sqrt(10 ** 4.5 / fs)
    sig = a * (bocB[hci] * symb - bocC[hci] * secc) * np.exp(2j * np.pi * fd * tt)
    iq = sig + (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    iq16 = np.empty(2 * n, dtype=np.int16)
    iq16[0::2] = np.clip(iq.real * 1000, -32767, 32767).astype(np.int16)
    iq16[1::2] = np.clip(iq.imag * 1000, -32767, 32767).astype(np.int16)

    x = iq[: int(fs * (40 * T_COH + 0.01))].astype(np.complex64)
    rep = {prn: boc_replica(codes_c[prn], fs)}
    r = acquire(x, fs, rep, np.arange(-5000, 5001, 125.0), 40)[prn]
    print(f"  acq: metric {r['metric']:.2f} dopp {r['dopp']:+.0f} "
          f"phase {r['code_phase']}")
    tr = track_prn(iq16, fs, prn, r["dopp"], r["code_phase"], 7.5, bocB, bocC)
    pages, st = harvest_pages(tr["sym"])
    print(f"  parts {st['n_parts']} tail_ok {st['tail_ok_frac']*100:.0f}% "
          f"viterbi_eff {st['viterbi_eff']:.2f} pairs {st['n_pairs']} "
          f"CRC pass {st['n_crc']}")
    got = [parse_word(p["page"]) for p in pages]
    wt0 = [f for wt, f in got if wt == 0 and "WN" in f]
    ok = (st["n_crc"] >= 3 and wt0
          and wt0[0]["WN"] == wn_true and wt0[0]["TOW"] == tow_true)
    print(f"  word types {[wt for wt, _ in got]}; "
          f"WT0 gives WN {wt0[0]['WN']} TOW {wt0[0]['TOW']}" if wt0 else
          "  no WT0 decoded")
    print(f"selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iq", default=None,
                    help="raw L1 IQ capture, interleaved int16 (required "
                         "unless --selftest)")
    ap.add_argument("--fs", type=float, default=4.096e6)
    ap.add_argument("--prns", default="29,27,14")
    ap.add_argument("--t0", type=float, default=1.0)
    ap.add_argument("--dur", type=float, default=777.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if not a.selftest and not a.iq:
        ap.error("--iq CAPTURE is required (or --selftest)")
    if a.selftest:
        return selftest(a.fs)

    fs = a.fs
    prns = [int(p) for p in a.prns.split(",")]
    codes_b, codes_c = load_codes()
    require_capture(a.iq)
    mm = np.memmap(a.iq, dtype=np.int16, mode="r")
    file_dur = mm.size / 2 / fs
    dur = min(a.dur, file_dur - a.t0 - 1.5)
    print(f"\nGalileo E1-B I/NAV decode: {Path(a.iq).name} "
          f"({file_dur:.0f} s @ {fs/1e6:.3f} MS/s), PRNs {prns}, "
          f"t0 {a.t0:.1f} s, track {dur:.0f} s")

    # ---- acquisition (E1-C), 160 ms noncoherent
    x = load_seg(a.iq, fs, a.t0, 40 * T_COH + 0.01)
    reps = {p: boc_replica(codes_c[p], fs) for p in prns}
    acq = acquire(x, fs, reps, np.arange(-5000, 5001, 125.0), 40)
    for p in prns:
        r = acq[p]
        print(f"  acq PRN{p:2d}: metric {r['metric']:5.2f}  "
              f"Doppler {r['dopp']:+5.0f} Hz  phase {r['code_phase']:5d} samp  "
              f"C/N0 ~{r['cn0']:.1f} dB-Hz")

    all_pages = {}
    all_stats = {}
    tracks = {}
    for p in prns:
        r = acq[p]
        if r["metric"] < 2.0:
            print(f"\nPRN{p}: acquisition metric {r['metric']:.2f} too weak, "
                  "skipping")
            continue
        print(f"\nPRN{p} tracking:")
        tr = track_prn(mm, fs, p, r["dopp"],
                       int(a.t0 * fs) + r["code_phase"], dur,
                       boc_chips(codes_b[p]), boc_chips(codes_c[p]))
        tracks[p] = tr
        pages, st = harvest_pages(tr["sym"])
        all_pages[p], all_stats[p] = pages, st
        sym = tr["sym"]
        eye = np.mean(np.abs(sym)) / (np.std(np.abs(sym)) + 1e-12)
        print(f"  symbols: {len(sym)} @ 250/s, eye |mean|/std {eye:.1f}")
        print(f"  preambles: grid phase {st['grid_phase']}, "
              f"{st['n_strong']}/{st['n_slots']} strong on a 250-sym grid, "
              f"{st['offgrid_strong']} strong off-grid, "
              f"grid contrast {st['grid_contrast']:.1f}x")
        print(f"  parts: {st['n_parts']} decoded, tail-bits clean "
              f"{st['tail_ok_frac']*100:.1f}%, Viterbi metric eff "
              f"{st['viterbi_eff']:.3f}")
        print(f"  PAGES: {st['n_pairs']} even/odd pairs -> {st['n_crc']} "
              f"CRC-24Q PASS ({100*st['n_crc']/max(st['n_pairs'],1):.1f}%)")

    # ---- parse + cross-checks
    print("\n" + "=" * 70)
    tow_anchor = []
    for p, pages in all_pages.items():
        if not pages:
            continue
        hist = {}
        gst = []
        info = {}
        for pg in pages:
            wt, f = parse_word(pg["page"])
            hist[wt] = hist.get(wt, 0) + 1
            if "TOW" in f and "WN" in f:
                t_cap = a.t0 + pg["i"] * T_COH       # part start, s into file
                gst.append((f["WN"], f["TOW"], t_cap, wt))
            for key in ("IODnav", "SISA", "E1BHS", "E1BDVS", "dtLS",
                        "A0", "e", "sqrtA", "toe", "af0"):
                if key in f:
                    info.setdefault(key, f[key])
        print(f"PRN{p}: word-type histogram {dict(sorted(hist.items()))}")
        if info:
            keys = ", ".join(f"{k}={v:.6g}" if isinstance(v, float)
                             else f"{k}={v}" for k, v in sorted(info.items()))
            print(f"  fields: {keys}")
        for wn, tow, t_cap, wt in gst[:3]:
            utc = gst_to_utc(wn, tow, int(info.get("dtLS", 18)))
            print(f"  GST from WT{wt}: WN {wn}  TOW {tow}  -> "
                  f"{utc:%Y-%m-%d %H:%M:%S} UTC  (page at t={t_cap:.0f} s)")
        tow_anchor += [(p, tow - t_cap) for wn, tow, t_cap, wt in gst]
    if tow_anchor:
        offs = np.array([o for _, o in tow_anchor])
        print(f"\nGST consistency: TOW minus capture-time across "
              f"{len(offs)} time-stamped pages / {len(set(p for p, _ in tow_anchor))} "
              f"PRNs: spread {offs.max()-offs.min():.3f} s "
              f"(same clock on every bird = decode is real)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
