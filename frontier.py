"""
frontier -- the risk/return curve each agent can reach, and whether A-S earns its k.

WHY A FRONTIER AND NOT A HEAD-TO-HEAD. Comparing one Avellaneda-Stoikov agent
against one naive agent compares two arbitrary parameter choices, not two
strategies. The earlier run had A-S quoting ~1.3 ticks against naive's 4: it
filled twice as often, ate the adverse selection, and lost. That says nothing
about inventory control -- it says a tight quoter gets picked off.

So sweep BOTH families over their own risk knob (gamma for A-S, half-spread for
naive) and plot what each can actually reach in (inventory risk, P&L) space. A
strategy is better only if its curve sits above the other's at matched risk.

CALIBRATING k, WHICH WAS PREVIOUSLY A GUESS. The A-S spread carries a term
(2/gamma)*ln(1 + gamma/k), where k is the decay rate of fill intensity in
distance from the mid. The model ASSUMES

    lambda(delta) = A * exp(-k * delta)

and k was set to 1.5 out of the air, which is what put the spread at 1.3 ticks.
`calibrate_k` measures it instead: quote at a fixed distance, count fills per
step, repeat across distances, and regress log-intensity on distance. The slope
is -k.

That measurement is also a TEST OF THE FLOW. If log-intensity is not close to
linear in distance, this simulator does not satisfy the assumption A-S is derived
under, and the closed form is being applied outside its own model. The R^2 is
reported for exactly that reason rather than quietly discarded.

WHAT THIS STILL DOES NOT PROVE. The flow and both agents were written by the same
person. A frontier drawn here describes these assumptions, not any real market.
What it legitimately shows is the SHAPE of the tradeoff and whether the closed
form reaches a better one than a fixed spread.
"""

from __future__ import annotations

import json
import math
import os
import statistics as st
from dataclasses import dataclass

from agent import AvellanedaStoikov, NaiveMM
from evaluate import decompose, simulate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reports")

SEEDS = (11, 12, 13, 14)
STEPS = 6000


# ------------------------------------------------------------------ calibration
def calibrate_k(distances=(1, 2, 3, 4, 6, 8), seeds=(21, 22, 23),
                steps=3000) -> dict:
    """Measure the decay rate of fill intensity in distance from the mid.

    Quote symmetrically at `delta` ticks, count filled units per step, and
    regress log(intensity) on delta by ordinary least squares. Slope = -k.
    """
    rows = []
    for d in distances:
        rates = []
        for s in seeds:
            res = simulate(lambda b, d=d: NaiveMM(b, half_spread=d),
                           seed=s, steps=steps)
            units = sum(q for (_t, _side, _px, q) in res.mm.fills)
            rates.append(units / steps)
        rows.append((d, st.mean(rates)))

    out = fit_exponential(rows)
    out["rows"] = rows
    return out


def fit_exponential(rows) -> dict:
    """Fit lambda(d) = A * exp(-k*d) by OLS on log-intensity. Returns k, A, R^2.

    Split out from `calibrate_k` so the arithmetic can be tested against data
    with a known k, rather than only against whatever the simulator happens to
    produce -- a calibration that is only ever checked on its own output is not
    checked.
    """
    pts = [(d, r) for d, r in rows if r > 0]
    if len(pts) < 3:
        return {"k": 1.5, "A": float("nan"), "r2": float("nan"),
                "usable": len(pts)}

    xs = [d for d, _ in pts]
    ys = [math.log(r) for _, r in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return {"k": 1.5, "A": float("nan"), "r2": float("nan"), "usable": n}
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx

    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return {"k": -slope, "A": math.exp(intercept), "r2": r2, "usable": n}


# --------------------------------------------------------------------- frontier
@dataclass
class Point:
    label: str
    param: float
    inv_sd: float
    pnl: float
    spread: float
    adverse: float
    inventory: float
    fills: float


def _measure(label, param, make, seeds=SEEDS, steps=STEPS) -> Point:
    sds, pnls, sp, ad, iv, fl = [], [], [], [], [], []
    for s in seeds:
        res = simulate(make, seed=s, steps=steps)
        d = decompose(res)
        sds.append(st.pstdev(res.inventory_path))
        pnls.append(d["total"])
        sp.append(d["spread"])
        ad.append(d["adverse"])
        iv.append(d["inventory"])
        fl.append(d["fills"])
    return Point(label, param, st.mean(sds), st.mean(pnls), st.mean(sp),
                 st.mean(ad), st.mean(iv), st.mean(fl))


def sweep(k: float) -> list[Point]:
    pts = []
    for g in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        pts.append(_measure("A-S", g,
                            lambda b, g=g: AvellanedaStoikov(b, gamma=g, k=k)))
    for hs in (1, 2, 3, 4, 6, 8, 12):
        pts.append(_measure("naive", hs, lambda b, hs=hs: NaiveMM(b, half_spread=hs)))
    return pts


def dominates(a: list[Point], b: list[Point]) -> dict:
    """At each of b's risk levels, is a's best P&L at no more risk higher?"""
    wins = 0
    checked = 0
    for pb in b:
        cands = [p for p in a if p.inv_sd <= pb.inv_sd]
        if not cands:
            continue
        checked += 1
        wins += max(c.pnl for c in cands) > pb.pnl
    return {"checked": checked, "wins": wins}


def main():
    os.makedirs(OUT, exist_ok=True)

    print("CALIBRATING k -- the A-S arrival-intensity decay, previously a guess")
    print("=" * 68)
    cal = calibrate_k()
    print(f"  {'distance':>10}{'fills/step':>14}")
    for d, r in cal["rows"]:
        print(f"  {d:>10}{r:>14.4f}")
    print(f"\n  fitted lambda(d) = {cal.get('A', float('nan')):.3f} * "
          f"exp(-{cal['k']:.3f} * d)")
    print(f"  k = {cal['k']:.3f}   R^2 = {cal['r2']:.3f}   (was guessed at 1.5)")
    if cal["r2"] < 0.90:
        print("  ** R^2 below 0.90: this flow does not cleanly satisfy the")
        print("     exponential-arrival assumption A-S is derived under. The")
        print("     closed form is being applied outside its own model, and that")
        print("     belongs in the write-up, not in a footnote.")
    else:
        print("  ** The exponential-arrival assumption holds well enough to use.")

    k = cal["k"]
    print("\n\nFRONTIER -- each family swept over its own risk knob")
    print("=" * 68)
    pts = sweep(k)
    print(f"  {'agent':>7}{'param':>8}{'inv sd':>9}{'P&L':>10}"
          f"{'spread':>10}{'adverse':>10}{'invent':>10}{'fills':>7}")
    for p in pts:
        print(f"  {p.label:>7}{p.param:>8g}{p.inv_sd:>9.1f}{p.pnl:>10.0f}"
              f"{p.spread:>10.0f}{p.adverse:>10.0f}{p.inventory:>10.0f}"
              f"{p.fills:>7.0f}")

    a_s = [p for p in pts if p.label == "A-S"]
    naive = [p for p in pts if p.label == "naive"]
    d1 = dominates(a_s, naive)
    d2 = dominates(naive, a_s)
    print(f"\n  A-S beats naive at matched-or-lower risk: "
          f"{d1['wins']}/{d1['checked']} of naive's risk levels")
    print(f"  naive beats A-S at matched-or-lower risk: "
          f"{d2['wins']}/{d2['checked']} of A-S's risk levels")

    with open(os.path.join(OUT, "frontier.json"), "w", encoding="utf-8") as fh:
        json.dump({"calibration": {kk: vv for kk, vv in cal.items() if kk != "rows"},
                   "calibration_rows": cal["rows"],
                   "points": [p.__dict__ for p in pts],
                   "as_beats_naive": d1, "naive_beats_as": d2,
                   "seeds": list(SEEDS), "steps": STEPS}, fh, indent=2)
    print(f"\n  wrote reports/frontier.json")
    return cal, pts


if __name__ == "__main__":
    main()
