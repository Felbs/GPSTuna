#!/usr/bin/env python3
"""drive_map.py -- draw a war drive's fixes on a map so the eye can judge them.

Reads the stamped fix results locate.py keeps in lab_local/ (one per solved
stop, each naming the capture that made it) plus drive.py's drive_log.jsonl,
and writes ONE self-contained HTML file back into lab_local/: numbered stops
in capture order, a circle per stop whose radius is that stop's epoch scatter
(the honest accuracy figure -- with 4 birds the residual rms is meaningless),
the route line between them, OpenStreetMap and satellite-imagery layers so a
fix can be checked against the car park it was actually taken in, and a table.

Coordinates go into the HTML only, and the HTML lives in lab_local/, which is
gitignored: nothing this prints or commits contains a position. The map tiles
and the Leaflet library load from the web when the page is opened, like any
map page; the fixes themselves stay in the file on your disk.

    python3 drive_map.py                      # every stamped fix
    python3 drive_map.py --drive 20260825     # captures named sky_capture_20260825_*
    python3 drive_map.py --out my_drive.html  # inside lab_local/ unless absolute
"""
import argparse
import glob
import html
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = HERE / "lab_local"
STAMP_RE = re.compile(r"sky_capture_(\d{8})_(\d{6})Z")


def capture_stamp(path):
    """'sky_capture_20260825_002140Z.cs16' -> ('20260825', '002140') or None."""
    m = STAMP_RE.search(os.path.basename(str(path)))
    return (m.group(1), m.group(2)) if m else None


def load_fixes(drive):
    """Latest valid stamped fix per capture, in capture order."""
    by_capture = {}
    for f in sorted(glob.glob(str(LOCAL / "fix_result_*Z.json"))):
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        if not d.get("valid") or "lat" not in d or "lon" not in d:
            continue
        st = capture_stamp(d.get("capture", ""))
        if st is None or (drive and st[0] != drive):
            continue
        d["_stamp"] = st
        d["_solved"] = os.path.basename(f)
        by_capture[st] = d                      # sorted() => last solve wins
    return [by_capture[k] for k in sorted(by_capture)]


def load_drive_log():
    """drive.py's per-cycle log, keyed by capture stamp (may be absent)."""
    out = {}
    p = LOCAL / "drive_log.jsonl"
    if not p.is_file():
        return out
    for line in p.read_text().splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        st = capture_stamp(d.get("file", ""))
        if st:
            out[st] = d
    return out


def quality(scatter_m, n_birds):
    if scatter_m < 40 and n_birds >= 5:
        return "#2e9e44", "good"
    if scatter_m < 120:
        return "#d9a400", "fair"
    return "#d13b2e", "poor"


def build_html(fixes, dlog, title):
    stops = []
    for i, d in enumerate(fixes, 1):
        day, hms = d["_stamp"]
        when = f"{day[:4]}-{day[4:6]}-{day[6:]} {hms[:2]}:{hms[2:4]}:{hms[4:]}Z"
        birds = d.get("birds", [])
        scatter = float(d.get("scatter_m", float("nan")))
        rms = float(d.get("resid_rms_m", 0.0))
        dof = max(0, len(birds) - 4)
        color, grade = quality(scatter, len(birds))
        log = dlog.get(d["_stamp"], {})
        stops.append({
            "n": i, "when": when, "lat": d["lat"], "lon": d["lon"],
            "alt": float(d.get("alt_m", float("nan"))),
            "birds": birds, "el": d.get("el_deg", []),
            "epochs": int(d.get("epochs_used", 0)),
            "scatter": scatter, "rms": rms, "dof": dof,
            "color": color, "grade": grade,
            "cycle": log.get("cycle"), "strong_at_capture": log.get("strong"),
            "capture": os.path.basename(str(d.get("capture", ""))),
            "solved": d["_solved"],
        })
    rows = []
    for s in stops:
        rms_txt = "&mdash;" if s["dof"] == 0 else f"{s['rms']:.1f} m"
        birds_txt = ", ".join(str(b) for b in s["birds"])
        rows.append(
            f"<tr style='border-left:6px solid {s['color']}'><td>{s['n']}</td>"
            f"<td>{s['when']}</td><td>{len(s['birds'])} <small>({birds_txt})</small></td>"
            f"<td>{s['epochs']}</td><td>{s['scatter']:.0f} m</td><td>{rms_txt}</td>"
            f"<td>{s['alt']:.0f} m</td><td>{s['grade']}</td></tr>")
    rows = "\n".join(rows)
    data = json.dumps(stops)
    t = html.escape(title)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{t}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 html,body{{margin:0;height:100%;font:14px/1.4 system-ui,sans-serif;background:#111;color:#ddd}}
 #map{{height:68vh}}
 #panel{{padding:10px 14px;overflow:auto;height:calc(32vh - 20px)}}
 table{{border-collapse:collapse;width:100%}} th,td{{padding:3px 8px;text-align:left;border-bottom:1px solid #333}}
 th{{color:#9ab}} small{{color:#889}}
 .stop{{background:#fff;border:2px solid #333;border-radius:50%;width:22px;height:22px;
        line-height:20px;text-align:center;font-weight:700;color:#000;font-size:12px}}
 .leaflet-popup-content{{font:13px/1.35 system-ui,sans-serif}}
 .legend{{position:absolute;right:10px;top:10px;z-index:1000;background:rgba(17,17,17,.85);
          padding:8px 10px;border-radius:6px;font-size:12px}}
 .sw{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}}
</style></head><body>
<div id="map"></div>
<div class="legend"><b>{t}</b><br>
 <span class="sw" style="background:#2e9e44"></span>good: scatter &lt; 40 m, &ge; 5 birds<br>
 <span class="sw" style="background:#d9a400"></span>fair: scatter &lt; 120 m<br>
 <span class="sw" style="background:#d13b2e"></span>poor<br>
 circle radius = epoch scatter of that stop</div>
<div id="panel">
<table><thead><tr><th>#</th><th>capture (UTC)</th><th>birds</th><th>epochs</th>
<th>scatter</th><th>rms</th><th>alt</th><th>grade</th></tr></thead>
<tbody>{rows}</tbody></table>
<p><small>rms is shown only when the solve had degrees of freedom (more than 4 birds);
with exactly 4 the residual is zero by construction and only the epoch scatter says anything.
Stops are numbered in capture order. Open the satellite layer (top-right of the map) to check
a fix against the ground you were actually parked on.</small></p>
</div>
<script>
const stops = {data};
const map = L.map('map');
const osm = L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}});
const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{maxZoom: 19, attribution: 'Imagery &copy; Esri'}});
osm.addTo(map);
L.control.layers({{'Map': osm, 'Satellite': sat}}).addTo(map);
const pts = [];
for (const s of stops) {{
  const ll = [s.lat, s.lon]; pts.push(ll);
  L.circle(ll, {{radius: s.scatter, color: s.color, weight: 1.5, fillOpacity: 0.12}}).addTo(map);
  const rms = s.dof === 0 ? '&mdash; (0 DOF)' : s.rms.toFixed(1) + ' m';
  L.marker(ll, {{icon: L.divIcon({{className: '', html: `<div class="stop" style="border-color:${{s.color}}">${{s.n}}</div>`,
                                   iconSize: [22, 22], iconAnchor: [11, 11]}})}})
   .bindPopup(`<b>Stop ${{s.n}}</b> &middot; ${{s.when}}<br>` +
              `${{s.lat.toFixed(6)}}, ${{s.lon.toFixed(6)}} &middot; alt ${{s.alt.toFixed(0)}} m<br>` +
              `birds ${{s.birds.join(', ')}} (el ${{s.el.map(e => e.toFixed(0)).join('/')}}&deg;)<br>` +
              `epochs ${{s.epochs}} &middot; scatter ${{s.scatter.toFixed(0)}} m &middot; rms ${{rms}}<br>` +
              `<small>${{s.capture}}</small>`).addTo(map);
}}
if (pts.length > 1) L.polyline(pts, {{color: '#5aa9ff', weight: 2, dashArray: '6 6'}}).addTo(map);
if (pts.length) map.fitBounds(L.latLngBounds(pts).pad(0.25)); else map.setView([0, 0], 2);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--drive", help="YYYYMMDD of the captures to draw (default: all)")
    ap.add_argument("--out", help="output HTML (relative paths land in lab_local/)")
    a = ap.parse_args()
    fixes = load_fixes(a.drive)
    if not fixes:
        sys.exit("no stamped fix results in lab_local/ match -- run locate.py first")
    out = Path(a.out) if a.out else Path(f"drive_map_{a.drive or 'all'}.html")
    if not out.is_absolute():
        out = LOCAL / out
    title = f"GPSTuna war drive {a.drive}" if a.drive else "GPSTuna fixes"
    out.write_text(build_html(fixes, load_drive_log(), title), encoding="utf-8")
    grades = [quality(float(f.get("scatter_m", 1e9)), len(f.get("birds", [])))[1]
              for f in fixes]
    print(f"{len(fixes)} stops -> {out}  "
          f"(good {grades.count('good')}, fair {grades.count('fair')}, "
          f"poor {grades.count('poor')})")
    print("open it in a browser; coordinates are in that file only")


if __name__ == "__main__":
    main()
