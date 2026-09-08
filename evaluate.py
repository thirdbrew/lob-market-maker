"""
evaluate -- decompose a market maker's P&L into where it actually came from.

MOST PUBLIC LOB PROJECTS PLOT A P&L CURVE AND STOP. That number answers nothing:
a maker can be profitable because it captured spread, or because its inventory
happened to drift the right way, and those have opposite implications for whether
the strategy is any good. This module splits the total into three terms that sum
EXACTLY to it.

THE ALGEBRA, because "approximately decomposed" is not a decomposition. For fill
i with signed quantity `s_i` (positive bought, negative sold) at price `p_i`, the
realised P&L marked at a final value `V` is

    total = SUM s_i * (V - p_i)

Insert the mid at fill time `m_i` and the value a short horizon later
`v_i = v(t_i + h)`, and the sum telescopes:

    total = SUM s_i * (m_i - p_i)      <- SPREAD CAPTURE
          + SUM s_i * (v_i - m_i)      <- ADVERSE SELECTION
          + SUM s_i * (V  - v_i)       <- INVENTORY

because (m_i - p_i) + (v_i - m_i) + (V - v_i) = V - p_i for every i. The identity
is exact, which makes it testable -- `test_evaluate.py` asserts the three terms
reconstruct the total to floating point.

WHAT EACH TERM MEANS:

  SPREAD CAPTURE      what you were paid to provide liquidity. Bought below the
                      mid or sold above it. This is the maker's revenue.

  ADVERSE SELECTION   what it cost you to have been RIGHT THERE when someone
                      informed arrived. Negative when the fills you got were the
                      ones you should not have wanted -- you bought and the value
                      fell. This is the maker's cost of doing business, and it is
                      the term the naive simulator has none of.

  INVENTORY           everything after the horizon: the mark-to-market on
                      whatever position you were still carrying. This is risk you
                      chose to hold, not a cost of quoting, and it is what the
                      Avellaneda-Stoikov skew exists to control.

THE HORIZON IS A CHOICE, AND IT MOVES THE SPLIT. Adverse selection has no
canonical measurement window. A short horizon attributes less to adverse
selection and more to inventory; a long one does the reverse. `horizon_sweep`
reports the split across several so the reader can see the sensitivity rather
than trust one number.

THE ANALYST SEES FAIR VALUE. THE AGENT DOES NOT. This module reads `FairValue`
deliberately -- measuring adverse selection requires knowing what the value did
after the fill. `agent.py` has no access to it and a test enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent import MarketMaker
from book import Book, Side
from flow import FairValue, Flow, FlowParams

DEFAULT_HORIZON = 40


@dataclass
class RunResult:
    """One simulation, with everything the decomposition needs."""
    mm: MarketMaker
    v_path: list[int] = field(default_factory=list)      # latent fair value
    mid_path: list[float] = field(default_factory=list)  # observed mid
    inventory_path: list[int] = field(default_factory=list)

    @property
    def steps(self) -> int:
        return len(self.v_path)


def simulate(make_agent, seed: int = 11, steps: int = 6000, warmup: int = 400,
             flow_params: FlowParams | None = None) -> RunResult:
    """Warm a book with noise flow, then run one agent against it."""
    book = Book()
    fair = FairValue(sigma_ticks=1.0)
    flow = Flow(book, fair, flow_params or FlowParams(seed=seed))
    for _ in range(warmup):
        flow.step()

    mm = make_agent(book)
    res = RunResult(mm=mm)
    for t in range(steps):
        mm.step(t, steps)
        flow.step()
        res.v_path.append(fair.price)
        res.mid_path.append(book.mid if book.mid is not None else fair.price)
        res.inventory_path.append(mm.inventory)
    mm._collect_fills()
    return res


def decompose(res: RunResult, horizon: int = DEFAULT_HORIZON) -> dict:
    """Split P&L into spread capture, adverse selection and inventory.

    The three terms sum to `total` exactly; see the module docstring for why.
    """
    v = res.v_path
    if not v:
        return {"total": 0.0, "spread": 0.0, "adverse": 0.0, "inventory": 0.0,
                "fills": 0, "horizon": horizon}

    V = float(v[-1])
    spread = adverse = inventory = 0.0
    volume = 0

    for (t, side, px, qty) in res.mm.fills:
        s = qty if side is Side.BID else -qty
        t = min(t, len(v) - 1)
        m = res.mid_path[t]
        v_h = float(v[min(t + horizon, len(v) - 1)])

        spread += s * (m - px)
        adverse += s * (v_h - m)
        inventory += s * (V - v_h)
        volume += qty

    total = spread + adverse + inventory
    return {
        "total": total,
        "spread": spread,
        "adverse": adverse,
        "inventory": inventory,
        "fills": len(res.mm.fills),
        "volume": volume,
        "spread_per_unit": spread / volume if volume else 0.0,
        "adverse_per_unit": adverse / volume if volume else 0.0,
        "horizon": horizon,
        "final_inventory": res.mm.inventory,
    }


def realised_pnl(res: RunResult) -> float:
    """Cash plus final inventory marked at the final FAIR value.

    Marked at fair value, not the observed mid, because that is the economic
    result. `MarketMaker.mark_to_market` uses the mid because that is all the
    AGENT can see; the two differ by the book's tracking error.
    """
    return res.mm.cash + res.mm.inventory * float(res.v_path[-1])


def horizon_sweep(res: RunResult, horizons=(10, 20, 40, 80, 160)) -> list[dict]:
    """The same run, split at several horizons. The total never moves; the
    boundary between adverse selection and inventory does."""
    return [decompose(res, h) for h in horizons]


def report(label: str, res: RunResult, horizon: int = DEFAULT_HORIZON) -> dict:
    d = decompose(res, horizon)
    print(f"\n{label}")
    print(f"  {'total P&L':<22}{d['total']:>12,.1f}")
    print(f"  {'  spread capture':<22}{d['spread']:>12,.1f}"
          f"   ({d['spread_per_unit']:+.3f} /unit)")
    print(f"  {'  adverse selection':<22}{d['adverse']:>12,.1f}"
          f"   ({d['adverse_per_unit']:+.3f} /unit)")
    print(f"  {'  inventory':<22}{d['inventory']:>12,.1f}")
    print(f"  {'fills / volume':<22}{d['fills']:>6,} /{d['volume']:>6,}"
          f"   final inventory {d['final_inventory']:+d}")
    return d


def main():
    from agent import AvellanedaStoikov, NaiveMM

    print("P&L DECOMPOSITION -- three terms that sum exactly to the total")
    print("=" * 64)

    runs = {
        "naive, half-spread 2": lambda b: NaiveMM(b, half_spread=2),
        "naive, half-spread 4": lambda b: NaiveMM(b, half_spread=4),
        "A-S, gamma=0.1": lambda b: AvellanedaStoikov(b, gamma=0.1),
        "A-S, gamma=1.0": lambda b: AvellanedaStoikov(b, gamma=1.0),
    }
    for label, make in runs.items():
        res = simulate(make)
        report(label, res)

    print("\n" + "=" * 64)
    print("HORIZON SENSITIVITY -- naive, half-spread 2")
    print("The total is fixed. Where the boundary between adverse selection and")
    print("inventory falls is a measurement choice, so it is shown, not hidden.")
    res = simulate(runs["naive, half-spread 2"])
    print(f"\n  {'horizon':>8}{'spread':>12}{'adverse':>12}{'inventory':>12}{'total':>12}")
    for d in horizon_sweep(res):
        print(f"  {d['horizon']:>8}{d['spread']:>12,.0f}{d['adverse']:>12,.0f}"
              f"{d['inventory']:>12,.0f}{d['total']:>12,.0f}")

    print("\n" + "=" * 64)
    print("WHAT THIS P&L DOES NOT PROVE. The order flow and the agent were both")
    print("written by the same person. Any profit here is a property of those")
    print("assumptions, not evidence of an edge in any real market. What the")
    print("decomposition IS good for is showing WHERE a result comes from -- and")
    print("whether a strategy that looks profitable is being paid for liquidity")
    print("or is simply long a drifting inventory.")


if __name__ == "__main__":
    main()
