"""
test_agent -- accounting, the closed form, and the one property that validates it.

THE TEST THAT MATTERS is `test_gamma_controls_inventory_variance`. Avellaneda-
Stoikov predicts that raising risk aversion tightens inventory control, and that
is a *falsifiable* statement about this implementation. If inventory variance
does not fall as gamma rises, what is in `agent.py` is a spread rule wearing the
model's name. Measured here: 63.6 at gamma=1e-4 down to 7.1 at gamma=3.

Everything else is accounting and sign conventions -- unglamorous, and the place
where a market-making P&L quietly books the spread twice.

TWO REGRESSION TESTS AT THE BOTTOM pin volatility-estimator failures that each
silently disabled inventory control while every run still looked healthy.
"""

import statistics as st

import pytest

from agent import MM_OWNER, AvellanedaStoikov, MarketMaker, NaiveMM
from book import Book, Side
from flow import FairValue, Flow, FlowParams


def _run(make, seed=11, steps=4000, warm=400):
    b = Book()
    fv = FairValue(sigma_ticks=1.0)
    f = Flow(b, fv, FlowParams(seed=seed))
    for _ in range(warm):
        f.step()
    mm = make(b)
    invs, sigmas = [], []
    for t in range(steps):
        mm.step(t, steps)
        f.step()
        invs.append(mm.inventory)
        if t % 100 == 0:
            sigmas.append(mm.sigma)
    mm._collect_fills()
    return mm, invs, sigmas


# ------------------------------------------------------------------ accounting

def test_a_buy_fill_moves_inventory_and_cash():
    b = Book()
    mm = NaiveMM(b, half_spread=2, size=10)
    b.limit(Side.BID, 90, 5, owner=1)     # resident book kept WIDE on purpose:
    b.limit(Side.ASK, 110, 5, owner=1)    # at the same price we queue BEHIND it
    mm.step(0, 100)                       # mid 100 -> we quote 98/102, alone

    b.market(Side.ASK, 4, owner=2)        # someone sells into our bid
    mm._collect_fills()
    assert mm.inventory == 4, "a bought fill must increase inventory"
    assert mm.cash == -4 * mm.resting_price(Side.BID), "cash must fall by qty * RESTING price"


def test_a_sell_fill_moves_inventory_and_cash():
    b = Book()
    mm = NaiveMM(b, half_spread=2, size=10)
    b.limit(Side.BID, 90, 5, owner=1)
    b.limit(Side.ASK, 110, 5, owner=1)
    mm.step(0, 100)
    b.market(Side.BID, 3, owner=2)        # someone buys from our ask
    mm._collect_fills()
    assert mm.inventory == -3
    assert mm.cash == 3 * mm.resting_price(Side.ASK)


def test_round_trip_captures_the_spread_and_nothing_more():
    """Buy at the bid, sell at the ask, flat: P&L is exactly the spread."""
    b = Book()
    mm = NaiveMM(b, half_spread=3, size=10)
    b.limit(Side.BID, 90, 5, owner=1)
    b.limit(Side.ASK, 110, 5, owner=1)
    mm.step(0, 100)
    bid_px, ask_px = mm.resting_price(Side.BID), mm.resting_price(Side.ASK)
    b.market(Side.ASK, 2, owner=2)        # we buy 2 at our bid
    b.market(Side.BID, 2, owner=3)        # we sell 2 at our ask
    mm._collect_fills()
    assert mm.inventory == 0
    assert mm.cash == pytest.approx(2 * (ask_px - bid_px))


# ----------------------------------------------------------- the closed form

def test_long_inventory_skews_quotes_down():
    """r = s - q*gamma*sigma^2*tau. Long inventory makes the maker want to sell."""
    b = Book()
    b.limit(Side.BID, 9_998, 20, owner=1)
    b.limit(Side.ASK, 10_002, 20, owner=1)
    mm = AvellanedaStoikov(b, gamma=1.0)
    import random                                   # a RAMP has zero variance and
    rng = random.Random(0)                          # floors sigma, which rounds the
    px = 10_000.0                                   # skew away entirely
    for _ in range(120):
        px += rng.gauss(0, 2.0)
        mm._mids.append(px)

    mm.inventory = 0
    flat_bid, flat_ask = mm.quotes(0, 100)
    mm.inventory = 50
    long_bid, long_ask = mm.quotes(0, 100)
    mm.inventory = -50
    short_bid, short_ask = mm.quotes(0, 100)

    assert long_bid < flat_bid and long_ask < flat_ask, "long inventory quotes DOWN"
    assert short_bid > flat_bid and short_ask > flat_ask, "short inventory quotes UP"


def test_risk_terms_vanish_at_the_horizon():
    """As tau -> 0 there is no inventory risk left to price."""
    b = Book()
    mm = AvellanedaStoikov(b, gamma=1.0)
    import random
    rng = random.Random(0)
    px = 10_000.0
    for _ in range(120):
        px += rng.gauss(0, 2.0)
        mm._mids.append(px)
    mm.inventory = 40

    near = abs(mm.reservation_price(10_000, tau=0.01) - 10_000)
    far = abs(mm.reservation_price(10_000, tau=1.00) - 10_000)
    assert near < far, "skew must shrink as the horizon closes"
    assert mm.optimal_spread(0.01) < mm.optimal_spread(1.00), "so must the width"


def test_spread_has_the_arrival_term_even_with_no_risk():
    """At tau=0 the risk term is 0 and the spread is (2/gamma)*ln(1+gamma/k)."""
    import math
    b = Book()
    mm = AvellanedaStoikov(b, gamma=0.5, k=1.5)
    expected = (2.0 / 0.5) * math.log(1.0 + 0.5 / 1.5)
    assert mm.optimal_spread(0.0) == pytest.approx(expected)


def test_gamma_controls_inventory_variance():
    """THE VALIDATION. Theory says higher risk aversion tightens inventory.

    If this fails, what is implemented is not Avellaneda-Stoikov.
    """
    low = [st.pstdev(_run(lambda b: AvellanedaStoikov(b, gamma=0.001), seed=s)[1])
           for s in (11, 12, 13)]
    high = [st.pstdev(_run(lambda b: AvellanedaStoikov(b, gamma=3.0), seed=s)[1])
            for s in (11, 12, 13)]
    assert st.mean(high) < st.mean(low) * 0.35, (
        f"inventory sd must fall materially with gamma: "
        f"{st.mean(low):.1f} at gamma=0.001 vs {st.mean(high):.1f} at gamma=3.0")


# ------------------------------------------------------ estimator regressions

def test_sigma_is_neither_floored_nor_capped_in_normal_operation():
    """REGRESSION x2. Both prior volatility estimators silently disabled
    inventory control while every run still looked healthy.

    The sample standard deviation OVERFLOWED a float: the observed mid gaps when
    the touch empties, and squaring a 120-tick liquidity event is not a
    volatility measurement.

    Then MAD was DEGENERATE: the mid is unchanged on ~2 of 3 steps, so the median
    diff is 0, the MAD is 0, and sigma pinned to its floor on 49 of 49 samples --
    the skew collapsed to 0.28 ticks and A-S became a fixed-spread quoter.

    Winsorising keeps a standard deviation's sensitivity while capping what one
    gap contributes.
    """
    mm, _, sigmas = _run(lambda b: AvellanedaStoikov(b, gamma=0.1), steps=3000)
    at_floor = sum(1 for s in sigmas if s <= mm.SIGMA_FLOOR + 1e-9)
    at_cap = sum(1 for s in sigmas if s >= mm.SIGMA_CAP - 1e-9)
    assert at_floor == 0, f"sigma floored on {at_floor}/{len(sigmas)} samples"
    assert at_cap == 0, f"sigma capped on {at_cap}/{len(sigmas)} samples"
    assert 0.1 < st.mean(sigmas) < 3.0, f"sigma {st.mean(sigmas):.3f} is implausible"


def test_agent_cannot_see_fair_value():
    """Structural: the agent module may not IMPORT or reference the latent value.

    Reads the parsed AST, not the source text -- the first version grepped the
    raw source and tripped on the module docstring, which explains at length that
    the agent cannot see FairValue.
    """
    import ast

    tree = ast.parse(open("agent.py", encoding="utf-8").read())
    imported, attrs = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Attribute):
            attrs.add(node.attr)
    assert "FairValue" not in imported, "agent must not import the latent value"
    assert "flow" not in imported, "agent must not depend on the flow module"
    assert "fair" not in attrs and "price" not in (attrs & {"fair"}), (
        "agent must not read a fair-value attribute")


def test_book_stays_healthy_with_an_agent_quoting():
    mm, _, _ = _run(lambda bk: AvellanedaStoikov(bk, gamma=0.1), steps=3000)
    mm.book.check()
    assert mm.book.best_bid is not None and mm.book.best_ask is not None


def test_sigma_survives_a_liquidity_gap():
    """Closes a mutation gap: removing the winsorisation changed nothing, because
    a healthy run has no gaps to clip. Feed it one on purpose.

    A 120-tick jump is a liquidity event, not volatility. A sample standard
    deviation squares it; the winsorised estimator must not.
    """
    b = Book()
    mm = AvellanedaStoikov(b, gamma=0.1)
    for i in range(120):                       # calm: +/- 1 tick
        mm._mids.append(10_000 + (i % 3) - 1)
    calm = mm.sigma

    mm._mids.append(10_120)                    # one 120-tick gap
    mm._mids.append(10_000)
    gapped = mm.sigma

    assert gapped < calm * 3.0, (
        f"one 120-tick gap moved sigma {calm:.3f} -> {gapped:.3f}. A single "
        f"liquidity event must not dominate the volatility estimate -- that is "
        f"the feedback loop that ran this simulator away.")
    assert gapped < mm.SIGMA_CAP, "sigma should not need the cap to stay sane"
