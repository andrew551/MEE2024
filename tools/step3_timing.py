"""The totality time budget: where the 106.8 s of Leon totality actually went.

Reads every FITS header under the eclipse tree and accounts for totality second by
second -- C2 to first shutter, the SCI_ladder window, the slew, the CAL_piLeo window to
C3 -- so the four segments sum to C3 - C2 exactly.

Two conventions the numbers depend on, both established elsewhere and both load-bearing:

  * `DATE-OBS` is the frame START ("System Clock:Est. Frame Start"), not the mid-point.
    `DATE-AVG` and `DATE-END` are derived from it by adding EXPTIME/2 and EXPTIME, so on
    a trap frame they inherit the wrong exposure and must not be used here.
  * Exposure comes from the FOLDER, not from `EXPTIME` -- the EXPTIME trap (STEP3_2026.md
    S0): the first frames after a SET EXPOSURE change carry the new header on the old
    exposure. This script re-confirms it a second way, by cadence: median frame-to-frame
    start spacing tracks the folder exposure, not the header.

Usage:  step3_timing.py [eclipse_tree]
"""
import os
import re
import sys
import glob
import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta

from astropy.io import fits

# leon_eclipse_v1.17.scs, one entry per CAPTURE command, in order. The block identity is the
# capture folder, so intended-vs-captured is a per-block comparison, not a per-tier one.
SCI_INTENT = [(0.1, 24), (0.3, 12), (0.6, 6), (1.2, 6), (0.6, 6), (0.3, 12), (0.1, 24)]
CAL_INTENT = [(0.3, 6), (1.0, 6), (2.0, 8)]   # x LOOP 8 TIMES

ROOT = sys.argv[1] if len(sys.argv) > 1 else r"G:\Leon Aug 2026\2026-08-12\Eclipse"

# LEON_2026-08-11.md 3.1, from the eclipse circumstances record. The tabulated contacts
# give 106.8 s; the same record quotes 1 m 46.4 s limb-corrected. The 0.4 s difference is
# left standing -- the budget below has to close against the contacts, not the summary.
C2 = datetime(2026, 8, 12, 18, 28, 7, 100000)
MAX = datetime(2026, 8, 12, 18, 29, 0, 600000)
C3 = datetime(2026, 8, 12, 18, 29, 53, 900000)


def hms(t):
    return t.strftime("%H:%M:%S.%f")[:-4]


def load(root):
    """One dict per frame: start time, folder-truth exposure, header exposure."""
    frames = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.fits"), recursive=True)):
        h = fits.getheader(path)
        parts = os.path.relpath(path, root).split(os.sep)
        folder_exp = parts[1] if len(parts) > 2 else parts[1] if len(parts) > 1 else ""
        hdr_exp = float(h["EXPTIME"])
        # "0.6s" -> 0.6; "discard" has no folder truth, so fall back to the header.
        try:
            exp = float(folder_exp.rstrip("s"))
        except ValueError:
            exp = hdr_exp
        t0 = datetime.strptime(h["DATE-OBS"][:26], "%Y-%m-%dT%H:%M:%S.%f")
        frames.append(dict(obj=parts[0], folder_exp=folder_exp,
                           capture=parts[-2] if len(parts) > 2 else "",
                           rel=os.path.relpath(path, root),
                           t0=t0, t1=t0 + timedelta(seconds=exp),
                           exp=exp, hdr_exp=hdr_exp,
                           ra=h.get("RA"), dec=h.get("DEC")))
    frames.sort(key=lambda f: f["t0"])
    return frames


def integrated(frames):
    return sum(f["exp"] for f in frames)


def blocks(frames):
    """Capture folders, in time order, with the cadence that proves the exposure."""
    g = defaultdict(list)
    for f in frames:
        g[(f["obj"], f["folder_exp"], f["capture"])].append(f)
    return sorted(g.items(), key=lambda kv: min(x["t0"] for x in kv[1]))


def blocks_vs_intent(root, frames):
    """Per capture block: what the script asked for vs what SharpCap actually wrote.

    A capture folder is one CAPTURE command, so the folders in time order line up with the
    script's CAPTURE lines and give the commanded frame count. Counting what arrived needs
    care in two places, both of them the EXPTIME trap:

      * frames re-sorted into another tier keep their original file number and pick up a
        " (2)" suffix on the name collision, and the rogue first frame was moved to
        `discard/`. Those files still ARRIVED -- they must be credited back to the block
        that wrote them, or the block looks short when it is not.
      * a relocated file's header EXPTIME is the trap value, which is precisely the
        exposure its ORIGINAL block was commanded. That, plus nearest block start,
        identifies the origin uniquely on this dataset.

    What is left after crediting those back is the real loss: a file number the block
    allocated and never wrote.
    """
    by_dir = defaultdict(list)
    for f in frames:
        by_dir[os.path.dirname(os.path.join(root, f["rel"]))].append(f)

    blocks, orphans = [], []
    for d, fs in by_dir.items():
        exp, ts = None, None
        for s in glob.glob(os.path.join(d, "*.CameraSettings.txt")):
            txt = open(s, encoding="utf-8", errors="ignore").read()
            m = re.search(r"^Exposure=(.+)$", txt, re.M)
            t = re.search(r"^TimeStamp=(.+)$", txt, re.M)
            if m:
                raw = m.group(1).strip()
                exp = float(raw[:-2]) / 1000 if raw.endswith("ms") else float(raw[:-1])
            if t:
                ts = datetime.strptime(t.group(1).strip()[:26], "%Y-%m-%dT%H:%M:%S.%f")
        own, moved = [], []
        for f in fs:
            (moved if ("(2)" in f["rel"] or os.path.basename(d) == "discard") else
             own).append(f)
        orphans += moved
        if own:
            blocks.append(dict(dir=os.path.relpath(d, root), exp=exp,
                               ts=ts or min(f["t0"] for f in own),
                               nums={int(re.search(r"_(\d{5})", f["rel"]).group(1))
                                     for f in own}))
    blocks.sort(key=lambda b: b["ts"])
    for f in orphans:                       # credit each relocated file back to its origin
        cand = [b for b in blocks if b["exp"] is not None
                and abs(b["exp"] - f["hdr_exp"]) < 1e-6]
        if cand:
            min(cand, key=lambda b: abs((b["ts"] - f["t0"]).total_seconds()))["nums"].add(
                int(re.search(r"_(\d{5})", f["rel"]).group(1)))
    return blocks


def slew(frames):
    """Mount-reported pointing either side of the field change (JNOW, from the headers)."""
    sci = [f for f in frames if f["obj"] == "SCI_ladder"]
    cal = [f for f in frames if f["obj"] == "CAL_piLeo"]
    a = max(sci, key=lambda f: f["t0"])
    b = min(cal, key=lambda f: f["t0"])
    d1, d2 = math.radians(a["dec"]), math.radians(b["dec"])
    dra = b["ra"] - a["ra"]
    sep = math.degrees(math.acos(math.sin(d1) * math.sin(d2)
                                 + math.cos(d1) * math.cos(d2) * math.cos(math.radians(dra))))
    return a, b, dra, b["dec"] - a["dec"], sep


def main():
    frames = load(ROOT)
    sci = [f for f in frames if f["obj"] == "SCI_ladder"]
    cal = [f for f in frames if f["obj"] == "CAL_piLeo"]
    cal_in = [f for f in cal if f["t0"] < C3]

    print(f"{len(frames)} frames under {ROOT}\n")

    print("=== blocks: cadence confirms the folder exposure, not the header ===")
    print(f"{'block':<30} {'n':>3} {'hdr EXPTIME':<14} {'first start':<14} {'med dt':>7}")
    for (obj, fexp, cap), g in blocks(frames):
        g = sorted(g, key=lambda f: f["t0"])
        dts = [(g[i + 1]["t0"] - g[i]["t0"]).total_seconds() for i in range(len(g) - 1)]
        med = f"{statistics.median(dts):.3f}" if dts else "-"
        exps = sorted({f["hdr_exp"] for f in g})
        print(f"{obj + '/' + fexp + '/' + cap:<30} {len(g):>3} {str(exps):<14} "
              f"{hms(g[0]['t0']):<14} {med:>7}")

    s_open = min(f["t0"] for f in sci)
    s_close = max(f["t1"] for f in sci)
    c_open = min(f["t0"] for f in cal_in)
    c_close = max(f["t1"] for f in cal_in)

    seg = [("C2 -> first shutter", C2, s_open, []),
           ("SCI_ladder window", s_open, s_close, sci),
           ("slew + settle", s_close, c_open, []),
           ("CAL_piLeo window to C3", c_open, C3, cal_in)]

    print("\n=== the budget (it must sum to C3 - C2) ===")
    print(f"{'segment':<24} {'from':<14} {'to':<14} {'s':>7} {'open s':>7} "
          f"{'dead s':>7} {'duty':>6}")
    total = open_total = 0.0
    for name, a, b, fs in seg:
        d = (b - a).total_seconds()
        o = integrated(fs)
        total += d
        open_total += o
        duty = f"{100 * o / d:5.1f}%" if fs else "     -"
        print(f"{name:<24} {hms(a):<14} {hms(b):<14} {d:7.2f} {o:7.2f} "
              f"{d - o:7.2f} {duty:>6}")
    print(f"{'TOTAL':<24} {hms(C2):<14} {hms(C3):<14} {total:7.2f} {open_total:7.2f} "
          f"{total - open_total:7.2f} {100 * open_total / total:5.1f}%")
    print(f"totality (C3 - C2) = {(C3 - C2).total_seconds():.2f} s; "
          f"MAX at {hms(MAX)}, {(MAX - C2).total_seconds():.1f} s in")

    print("\n=== integrated time per tier ===")
    for label, fs in (("SCI_ladder (all)", sci), ("CAL_piLeo (inside totality)", cal_in)):
        per = defaultdict(lambda: [0, 0.0])
        for f in fs:
            per[f["folder_exp"]][0] += 1
            per[f["folder_exp"]][1] += f["exp"]
        print(f"  {label}: {len(fs)} frames, {integrated(fs):.2f} s")
        for k in sorted(per, key=lambda k: per[k][1], reverse=True):
            print(f"      {k:>8}: {per[k][0]:>3} frames, {per[k][1]:6.2f} s")

    # The C3 cut lands in a gap, not in an exposure -- worth asserting rather than
    # assuming, because if it ever straddled, the CAL total would need clipping.
    straddle = [f for f in cal if f["t0"] < C3 < f["t1"]]
    print(f"\nframes straddling C3: {len(straddle)}")
    print(f"  last close inside : {hms(c_close)}  ({(C3 - c_close).total_seconds():.2f} s "
          f"before C3)")
    after = [f for f in cal if f["t0"] >= C3]
    if after:
        first_after = min(after, key=lambda f: f["t0"])
        print(f"  first open after  : {hms(first_after['t0'])}  "
              f"({(first_after['t0'] - C3).total_seconds():.2f} s after C3)")
    kept_after = [f for f in after if f["folder_exp"] != "discard"]
    print(f"  CAL past C3: {len(after)} frames, {integrated(after):.2f} s "
          f"(kept tiers: {len(kept_after)} frames, {integrated(kept_after):.2f} s)")

    print("\n=== intended vs captured, per capture block ===")
    intent = [n for _, n in SCI_INTENT] + [n for _, n in CAL_INTENT] * 8
    census = blocks_vs_intent(ROOT, frames)
    print(f"{'block':<32} {'cmd exp':>8} {'start':>13} {'cmd n':>6} {'arrived':>8} "
          f"{'missing':>8}")
    lost = 0
    for b, want in zip(census, intent):
        got = len(b["nums"])
        miss = sorted(set(range(1, want + 1)) - b["nums"])
        lost += len(miss)
        note = f"  <-- no #{','.join(map(str, miss))}" if miss else ""
        print(f"{b['dir']:<32} {b['exp'] if b['exp'] else '-':>8} "
              f"{b['ts'].strftime('%H:%M:%S.%f')[:-4]:>13} {want:>6} {got:>8} "
              f"{len(miss):>8}{note}")
    print(f"frames commanded and never written: {lost}")

    a, b, dra, ddec, sep = slew(frames)
    print("\n=== the field change, from the mount's own reported pointing (JNOW) ===")
    print(f"  SCI last frame : RA {a['ra']:9.4f}  Dec {a['dec']:+8.4f}")
    print(f"  CAL first frame: RA {b['ra']:9.4f}  Dec {b['dec']:+8.4f}")
    print(f"  RA axis {dra:+.4f} deg ({dra*60:+.1f}')   Dec axis {ddec:+.4f} deg "
          f"({ddec*60:+.1f}')")
    print(f"  on-sky great circle {sep:.4f} deg; longer axis "
          f"{'Dec' if abs(ddec) > abs(dra) else 'RA'} at {max(abs(dra), abs(ddec)):.4f} deg")
    gap = (b["t0"] - max(f["t1"] for f in frames if f["obj"] == "SCI_ladder")).total_seconds()
    motion = gap - 10.0 - 0.75 - 0.30   # 5 s SharpCap settle + DELAY 5; measured overheads
    print(f"  gap {gap:.2f} s - 10 s settling - ~1.05 s overhead -> motion ~{motion:.1f} s "
          f"= {abs(ddec)/motion:.2f} deg/s on the Dec axis, {sep/motion:.2f} deg/s on sky")


if __name__ == "__main__":
    main()
