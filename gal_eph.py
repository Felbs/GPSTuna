#!/usr/bin/env python3
"""gal_eph.py - assemble full Galileo ephemerides from decoded I/NAV words.

Takes the CRC-clean pages that gal_inav.py harvests and turns them into
fix.py-style ephemeris dicts ready for sat_ecef()/clock_corr():

  * word types 1-4 matched by IODnav (ICD 5.1.9.2: one IOD identifies the
    ephemeris + clock + SISA batch; a field set is only used if all four
    words carry the SAME IODnav)
  * angles arrive in semicircles (2^-31 / 2^-43 scale factors) and are
    converted here to radians (* pi), matching fix.py's GPS conventions
  * mu = 3.986004418e14 m^3/s^2 (Galileo OS SIS ICD 5.1.1) rides inside
    the dict so fix.sat_ecef uses the Galileo constant
  * WT5 adds BGD + health + GST context. Single-frequency E1 users of
    I/NAV must apply BGD(E1,E5b) to the clock (ICD 5.1.5 Eq 15):
        dt_sv(E1) = dt_sv(E1,E5b) - BGD(E1,E5b)
    -> af0_e1 = af0 - BGD(E1,E5b), folded by fold_bgd() (NOT automatic,
    so the raw broadcast values stay inspectable).
  * WT10 gives the GST-GPS offset (GGTO, ICD 5.1.8 Eq 21):
        dt_systems = t_Galileo - t_GPS
                   = A0G + A1G*(TOW - t0G + 604800*((WN - WN0G) mod 64))

Scale factors verified against the Galileo OS SIS ICD Issue 2.0 (Jan 2021)
tables 60-69 and cross-checked against gnss-sdr's Galileo_INAV.h.
"""
import numpy as np

MU_GAL = 3.986004418e14         # ICD 5.1.1 geocentric gravitational constant
PI = 3.1415926535898            # ICD value of pi (semicircle conversions)

# fields each word type must contribute for a complete ephemeris
_WT_KEYS = {
    1: ("toe", "M0_sc", "e", "sqrtA"),
    2: ("Omega0_sc", "i0_sc", "omega_sc", "IDOT_sc"),
    3: ("OmegaDot_sc", "dn_sc", "Cuc", "Cus", "Crc", "Crs", "SISA"),
    4: ("toc", "af0", "af1", "af2", "Cic", "Cis"),
}


def assemble_eph(parsed, prn):
    """parsed = [(wt, fields_dict), ...] from gal_inav.parse_word over the
    CRC-clean pages of ONE satellite. Returns a fix.py-style eph dict or
    None if no IODnav-matched WT1-4 set exists."""
    by_iod = {}
    for wt, f in parsed:
        if wt in _WT_KEYS and "IODnav" in f:
            by_iod.setdefault(f["IODnav"], {})[wt] = f
    # newest complete batch (highest IODnav with all four words)
    complete = [iod for iod, ws in by_iod.items()
                if all(wt in ws for wt in (1, 2, 3, 4))]
    if not complete:
        return None
    iod = max(complete)
    ws = by_iod[iod]
    eph = {
        "prn": prn, "sys": "GAL", "mu": MU_GAL, "IODnav": iod,
        # WT1
        "toe": float(ws[1]["toe"]),
        "M0": ws[1]["M0_sc"] * PI,
        "e": float(ws[1]["e"]),
        "sqrtA": float(ws[1]["sqrtA"]),
        # WT2 (semicircles -> rad)
        "Omega0": ws[2]["Omega0_sc"] * PI,
        "i0": ws[2]["i0_sc"] * PI,
        "omega": ws[2]["omega_sc"] * PI,
        "IDOT": ws[2]["IDOT_sc"] * PI,
        # WT3
        "OmegaDot": ws[3]["OmegaDot_sc"] * PI,
        "dn": ws[3]["dn_sc"] * PI,
        "Cuc": float(ws[3]["Cuc"]), "Cus": float(ws[3]["Cus"]),
        "Crc": float(ws[3]["Crc"]), "Crs": float(ws[3]["Crs"]),
        "SISA": int(ws[3]["SISA"]),
        # WT4 clock
        "toc": float(ws[4]["toc"]),
        "af0": float(ws[4]["af0"]),
        "af1": float(ws[4]["af1"]),
        "af2": float(ws[4]["af2"]),
        "Cic": float(ws[4]["Cic"]), "Cis": float(ws[4]["Cis"]),
    }
    # WT5 extras: BGD, health, iono, GST week context (no IODnav on WT5)
    for wt, f in parsed:
        if wt == 5:
            for k in ("BGD_E1E5a", "BGD_E1E5b", "E1BHS", "E1BDVS",
                      "E5bHS", "ai0", "ai1", "ai2", "WN"):
                if k in f and k not in eph:
                    eph[k] = f[k]
    return eph


def fold_bgd(eph):
    """ICD 5.1.5 Eq 15: single-frequency E1 user clock. Returns a COPY with
    af0 -> af0 - BGD(E1,E5b) so fix.clock_corr() yields dt_sv(E1)."""
    e2 = dict(eph)
    e2["af0"] = eph["af0"] - eph.get("BGD_E1E5b", 0.0)
    e2["bgd_folded"] = True
    return e2


def extract_ggto(parsed):
    """First valid WT10 -> GGTO dict, or None."""
    for wt, f in parsed:
        if wt == 10 and f.get("ggto_valid"):
            return {k: f[k] for k in ("A0G", "A1G", "t0G", "WN0G")}
    return None


def ggto_eval(ggto, wn_gst, tow_gst):
    """ICD 5.1.8 Eq 21: dt_systems = t_Galileo - t_GPS (seconds) at GST
    (wn, tow). Week difference is mod-64 (WN0G is 6 bits)."""
    dw = (wn_gst - ggto["WN0G"]) % 64
    if dw > 31:                          # ICD: |WN - WN0G| <= 31 at broadcast
        dw -= 64
    return ggto["A0G"] + ggto["A1G"] * (tow_gst - ggto["t0G"] + 604800.0 * dw)


def anchors_from_pages(pages, parse_word, ptrs, fs):
    """Timing anchors: (file_time_s, gst_tow_s) at each time-stamped page.
    ICD 5.1.2: the broadcast TOW marks the leading edge of the first chip of
    the first symbol of its page - i.e. the start of the EVEN part, which is
    pages[k]['i'] in symbol index = ptrs[i] in file samples. TOW is each
    satellite's own GST realization, so this maps file time -> SV time
    (exactly the anchor-fit convention fix.py uses for GPS)."""
    ft, st = [], []
    for pg in pages:
        wt, f = parse_word(pg["page"])
        if "TOW" in f and pg["i"] < len(ptrs):
            ft.append(ptrs[pg["i"]] / fs)
            st.append(float(f["TOW"]))
    return np.array(ft), np.array(st)
