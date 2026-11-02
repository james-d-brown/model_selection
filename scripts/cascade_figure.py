"""
Draw docs/cascade.svg, Figure 1 of docs/methods.md (The cascade, Procedure):
forward selection in each tier and the checks that lead to the next tier.
The layout is set by hand so that labels and lines do not overlap.

    python scripts/cascade_figure.py [output path]
"""
import sys
from pathlib import Path

FONT = "Helvetica, Arial, sans-serif"
C = {"box": "#f4f6f8", "start": "#e6eef7", "trial": "#fde8e8", "check": "#fff6e0", "acc": "#e7f4ea",
     "line": "#4f5b63", "esc": "#b03a3a", "frame": "#9aa7b0"}
out = []

def text(x, y, s, size=13, anchor="middle", color="#1d1d1f", weight="normal"):
    out.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" text-anchor="{anchor}" '
               f'fill="{color}" font-weight="{weight}">{s}</text>')

def box(cx, cy, w, h, lines, fill, bold=False):
    out.append(f'<rect x="{cx-w/2}" y="{cy-h/2}" width="{w}" height="{h}" rx="9" fill="{C[fill]}" '
               f'stroke="{C["line"]}" stroke-width="{1.8 if bold else 1.2}"/>')
    n = len(lines); lh = 16
    for i, s in enumerate(lines):
        text(cx, cy - (n - 1) * lh / 2 + i * lh + 4.5, s, weight="bold" if bold else "normal")
    return (cx, cy, w, h)

def path(points, color="line", arrow=True, dashed=False):
    d = "M " + " L ".join(f"{x},{y}" for x, y in points)
    marker = f' marker-end="url(#arrow-{color})"' if arrow else ""
    dash = ' stroke-dasharray="6,4"' if dashed else ""
    out.append(f'<path d="{d}" fill="none" stroke="{C[color]}" stroke-width="1.4"{dash}{marker}/>')

def label(x, y, s, color="line", anchor="start"):
    text(x, y, s, size=12, anchor=anchor, color=C[color] if color != "line" else "#333")

def top(b): return (b[0], b[1] - b[3] / 2)
def bot(b): return (b[0], b[1] + b[3] / 2)
def left(b): return (b[0] - b[2] / 2, b[1])
def right(b): return (b[0] + b[2] / 2, b[1])

CW = 440
GAP = 110  # wide enough for the escalation labels between the tiers
W, H = 40 + 3 * CW + 2 * GAP, 840
cols = {"lin": 20 + CW / 2, "non": 20 + CW + GAP + CW / 2, "ff": 20 + 2 * (CW + GAP) + CW / 2}  # column frame width
rows = {"start": 92, "score": 170, "best": 240, "c1": 315, "test": 405, "res": 520}
BW, BH, BH3 = 230, 44, 60
FRAME_H = 655
bus_y, end_y = 725, 795
FF = {"cal": 92, "start": 170, "score": 240, "best": 315, "test": 405, "cal2": 520, "acc": 615}

# frames and titles
for key, title in (("lin", "Linear tier"), ("non", "Nonlinear tier"), ("ff", "Free-form tier")):
    x = cols[key]
    out.append(f'<rect x="{x-CW/2}" y="28" width="{CW}" height="{FRAME_H}" rx="14" fill="none" stroke="{C["frame"]}" stroke-width="1.2"/>')
    text(x, 52, title, size=15, weight="bold")

boxes = {}
L, N, F = cols["lin"], cols["non"], cols["ff"]
boxes["lin", "start"] = box(L, rows["start"], BW, BH, ["Start: empty selection"], "start", bold=True)
boxes["non", "start"] = box(N, rows["start"], BW, BH, ["New trial: empty selection,", "or the selection so far"], "trial")
boxes["ff", "cal"] = box(F, FF["cal"], BW + 40, BH, ["Calibration test: is the free-form", "tier calibrated on the sample?"], "check")
boxes["ff", "start"] = box(F, FF["start"], BW, BH, ["Continue from the", "selection so far"], "trial")
for key, r in (("lin", rows), ("non", rows), ("ff", FF)):
    x = cols[key]
    boxes[key, "score"] = box(x, r["score"], BW, BH, ["Score each remaining", "candidate"], "box")
    boxes[key, "best"] = box(x, r["best"], BW, BH, ["Take the highest-scoring", "candidate"], "box")
    boxes[key, "test"] = box(x, r["test"], BW + 20, BH3, ["Acceptance test: does it add", "significant information, at", "least the acceptance floor?"], "box")
boxes["lin", "acc"] = box(L - 95, rows["res"], 170, BH3, ["Add it to the", "selection.", "Candidates left?"], "acc")
boxes["non", "acc"] = box(N - 95, rows["res"], 170, BH3, ["Add it to the", "selection.", "Candidates left?"], "acc")
boxes["ff", "cal2"] = box(F - 95, FF["cal2"], 170, BH3, ["Calibration test: is the", "tier calibrated, given the", "current selection?"], "check")
boxes["ff", "acc"] = box(F - 95, FF["acc"], 170, BH3, ["Add it to the", "selection.", "Candidates left?"], "acc")
boxes["lin", "c1"] = box(L, rows["c1"], BW + 80, BH, ["Linear adjustment test: can the dependence", "on Z and V be removed linearly?"], "check")
boxes["lin", "c2"] = box(L + 100, rows["res"] + 8, 190, 76, ["Each remaining candidate:", "does it relate nonlinearly", "to the observations, or vary", "across states?"], "check")
boxes["non", "c3"] = box(N + 100, rows["res"] + 8, 190, 76, ["Each remaining candidate:", "does its relationship with", "the observations vary", "across states?"], "check")
end = box(N, end_y, 300, BH, ["End: the selected formulations"], "start", bold=True)


# vertical main flow
for key in cols:
    path([bot(boxes[key, "start"]), top(boxes[key, "score"])])
    path([bot(boxes[key, "score"]), top(boxes[key, "best"])])
path([bot(boxes["ff", "cal"]), top(boxes["ff", "start"])]); label(F + 8, (FF["cal"] + FF["start"]) / 2 + 4, "yes")
path([bot(boxes["lin", "best"]), top(boxes["lin", "c1"])])
path([bot(boxes["lin", "c1"]), top(boxes["lin", "test"])]); label(L + 6, rows["c1"] + 44, "yes")
for key in ("non", "ff"):
    path([bot(boxes[key, "best"]), top(boxes[key, "test"])])

def down(src, dst, lab):
    """Arrow from the bottom of src straight down into dst (which sits below a part of src)."""
    sb, db = boxes[src], boxes[dst]
    path([(db[0], sb[1] + sb[3] / 2), top(db)]); label(db[0] + 6, sb[1] + sb[3] / 2 + 22, lab)

down(("lin", "test"), ("lin", "acc"), "yes"); down(("lin", "test"), ("lin", "c2"), "no")
down(("non", "test"), ("non", "acc"), "yes"); down(("non", "test"), ("non", "c3"), "no")
down(("ff", "test"), ("ff", "cal2"), "yes")
path([bot(boxes["ff", "cal2"]), top(boxes["ff", "acc"])]); label(boxes["ff", "acc"][0] + 6, bot(boxes["ff", "cal2"])[1] + 22, "yes")

# accept -> next round (yes): out of the left side, up the column's left margin, into the score box
for key in cols:
    ab = boxes[key, "acc"]; sb = boxes[key, "score"]; gx = cols[key] - CW / 2 + 12
    path([left(ab), (gx, ab[1]), (gx, sb[1]), left(sb)])
    label((gx + left(ab)[0]) / 2, ab[1] - 6, "yes", anchor="middle")
    out.append(f'<text x="{gx + 13}" y="{(sb[1] + ab[1]) / 2}" font-family="{FONT}" font-size="12" fill="#333" '
               f'text-anchor="middle" transform="rotate(-90 {gx + 13} {(sb[1] + ab[1]) / 2})">next round</text>')

# merge bus to End: every "no" that ends selection drops onto one line
xs = []
def drop(x, y_from, y_label):
    path([(x, y_from), (x, bus_y)], arrow=False); label(x + 6, y_label, "no"); xs.append(x)
for key in cols:
    ab = boxes[key, "acc"]; drop(ab[0], bot(ab)[1], bot(ab)[1] + 18)
for key, ck in (("lin", "c2"), ("non", "c3")):
    cb = boxes[key, ck]; drop(cb[0], bot(cb)[1], bot(cb)[1] + 18)
# free-form: not calibrated after accepting -> end (from the right side of the box, down to the bus)
c2b = boxes["ff", "cal2"]; x2 = F + 20
path([right(c2b), (x2, c2b[1]), (x2, bus_y)], arrow=False); label(right(c2b)[0] + 6, c2b[1] - 7, "no"); xs.append(x2)
# free-form: test not passed -> end
tb = boxes["ff", "test"]; drop(F + 100, tb[1] + tb[3] / 2, tb[1] + tb[3] / 2 + 22)
# free-form tier not calibrated on entry -> end: the nonlinear tier's selection stands (the departure concerns power only)
cal = boxes["ff", "cal"]; cx_end = F + CW / 2 - 14
path([right(cal), (cx_end, cal[1]), (cx_end, bus_y)], arrow=False); label(right(cal)[0] + 8, cal[1] - 7, "no")
xs.append(cx_end)
out.append(f'<path d="M {min(xs)},{bus_y} L {max(xs)},{bus_y}" stroke="{C["line"]}" stroke-width="1.4"/>')
path([(end[0], bus_y), top(end)])

# escalation arrows (red): out of the right side, along a gutter, into the next tier
def escalate(src, dst, edge, gx, y_in, lab, entry, entry_below=False):
    """Red arrow from src to dst. lab sits above the line just outside the
    source tier's frame (edge); entry (restart or continue) sits just inside
    the destination tier's frame, above the incoming line, or below it, clear
    of the arrowhead."""
    sb = src; db = dst
    path([right(sb), (gx, sb[1]), (gx, y_in), (db[0] - db[2] / 2, y_in)], color="esc")
    label(edge + 6, sb[1] - 6, lab, color="esc")
    label(edge + GAP + 8, y_in + 14 if entry_below else y_in - 6, entry, color="esc")
edge1 = L + CW / 2
escalate(boxes["lin", "c1"], boxes["non", "start"], edge1, edge1 + GAP / 3, rows["start"] - 14, "no", "restart")
escalate(boxes["lin", "c2"], boxes["non", "start"], edge1, edge1 + 2 * GAP / 3, rows["start"] + 14, "yes",
         "continue", entry_below=True)
edge2 = N + CW / 2
escalate(boxes["non", "c3"], boxes["ff", "cal"], edge2, edge2 + GAP / 2, FF["cal"], "yes", "continue")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
       f'aria-label="Flow of the cascade across the three tiers">'
       '<defs>' + "".join(
           f'<marker id="arrow-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
           f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{C[k]}"/></marker>' for k in ("line", "esc")) + '</defs>'
       f'<rect width="100%" height="100%" fill="white"/>' + "\n".join(out) + '</svg>')
target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "docs" / "cascade.svg"
target.write_text(svg, encoding="utf-8")
print(f"wrote {target}")
