"""
test_flow -- does the simulated market actually behave like one?

A synthetic flow generator is trivially "correct" -- it runs, it emits orders,
nothing raises. The question that matters is whether it produces the *mechanism*
the rest of the project needs, and that is testable.

The load-bearing property is ADVERSE SELECTION -- that a passive maker gets
picked off, and more so the more informed flow there is. If that collapses,
`evaluate.py`'s decomposition has an empty term and the whole project measures
nothing.

It is deliberately measured DIRECTLY rather than through tracking. Tracking was
the first proxy and it was a good one until the noise makers were given their own
noisy view of value; after that the makers did the tracking work themselves and
the proxy went quiet while the mechanism was untouched. A proxy that stops
tracking its target is worse than no proxy, because it keeps reporting.

The regression test at the bottom is the one that earns its keep. It pins a bug
that produced a perfectly healthy-looking run with **zero uninformed trades in
it**.
"""

import statistics as st

import pytest

from book import Book, Side
from flow import (INFORMED_OWNER, MAKER_OWNER, TAKER_OWNER, FairValue, Flow,
                  FlowParams)


def _run(steps=4000, warmup=400, **kw):
    b = Book()
    fv = FairValue(sigma_ticks=1.0)
    f = Flow(b, fv, FlowParams(**kw))
    errs = []
    for i in range(steps):
        f.step()
        if i > warmup and b.mid is not None:
            errs.append(b.mid - fv.price)
    return b, fv, f, errs


# ------------------------------------------------------------ the mechanism

def _adverse_drift(informed_frac, seed, steps=6000, horizon=40):
    """Mean drift AGAINST a passive maker's fills, in ticks x size.

    Positive means the maker was picked off: it bought and the value fell, or
    sold and the value rose.
    """
    from agent import NaiveMM

    b = Book()
    fv = FairValue(sigma_ticks=1.0)
    f = Flow(b, fv, FlowParams(seed=seed, informed_frac=informed_frac))
    for _ in range(400):
        f.step()
    mm = NaiveMM(b, half_spread=2)
    vpath = []
    for t in range(steps):
        mm.step(t, steps)
        f.step()
        vpath.append(fv.price)
    mm._collect_fills()

    d = []
    for (t, side, px, qty) in mm.fills:
        if t + horizon < len(vpath):
            later = vpath[t + horizon]
            d.append(qty * ((px - later) if side is Side.BID else (later - px)))
    return st.mean(d) if d else 0.0


def test_informed_flow_creates_adverse_selection():
    """THE PROPERTY THE SIMULATOR EXISTS TO HAVE, measured directly.

    An earlier version of this test measured TRACKING -- how closely the mid
    followed fair value -- as a proxy, and it was a good proxy right up until
    the noise makers were given their own noisy view of value. Once makers
    anchored to `v` themselves, they did the tracking work and informed flow
    barely moved it (sd 5.63 -> 5.18, ~8%). The proxy stopped measuring the
    mechanism while the mechanism was still there.

    So measure the mechanism. Adverse selection is the maker being picked off:
    it bought just before the value fell, or sold just before it rose. That is
    what informed flow *is*, and it scales cleanly with how much of it there is:

        informed_frac   drift per fill
             0.00            1.49
             0.15            2.88
             0.35            4.76
             0.60           10.77

    If this collapses, `evaluate.py`'s three-way decomposition has an empty term.
    """
    low = st.mean([_adverse_drift(0.0, s) for s in (1, 2, 3)])
    high = st.mean([_adverse_drift(0.6, s) for s in (1, 2, 3)])
    assert high > low * 3.0, (
        f"informed flow must materially increase adverse selection: "
        f"{low:.2f} at frac=0 vs {high:.2f} at frac=0.6. If these are close there "
        f"is nothing for the P&L decomposition to attribute.")
    assert low > 0, "even uninformed flow should show some drift against a maker"


def test_informed_traders_take_the_mispriced_side():
    """An informed buyer appears only when the ask is BELOW fair value."""
    b = Book()
    fv = FairValue(price=10_000, sigma_ticks=0.0)
    f = Flow(b, fv, FlowParams(seed=2, informed_edge=2))

    b.limit(Side.ASK, 9_990, 50, owner=MAKER_OWNER)   # ask 10 ticks too cheap
    b.limit(Side.BID, 9_960, 50, owner=MAKER_OWNER)
    assert f._informed_side() is Side.BID, "cheap ask must attract an informed BUY"

    b2 = Book()
    f2 = Flow(b2, fv, FlowParams(seed=2, informed_edge=2))
    b2.limit(Side.BID, 10_010, 50, owner=MAKER_OWNER)  # bid 10 ticks too rich
    b2.limit(Side.ASK, 10_040, 50, owner=MAKER_OWNER)
    assert f2._informed_side() is Side.ASK, "rich bid must attract an informed SELL"


def test_no_informed_trade_when_the_quote_is_fair():
    b = Book()
    fv = FairValue(price=10_000, sigma_ticks=0.0)
    f = Flow(b, fv, FlowParams(seed=3, informed_edge=2))
    b.limit(Side.BID, 9_999, 20, owner=MAKER_OWNER)
    b.limit(Side.ASK, 10_001, 20, owner=MAKER_OWNER)
    assert f._informed_side() is None, "a fair quote must not be picked off"


# --------------------------------------------------------- the regression test

def test_uninformed_market_orders_actually_fill():
    """REGRESSION. The noise makers and the uninformed takers once shared an
    owner id, so the book's self-trade prevention silently refused EVERY
    uninformed market order -- 0 fills over 6,000 steps, while every other
    statistic in the run looked healthy.

    A flow with no uninformed trades in it has no spread to capture, which would
    have made the market maker's P&L pure adverse selection and the result
    meaningless.
    """
    _, _, f, _ = _run(steps=3000, informed_frac=0.35, seed=4)
    assert f.uninformed_fills > 0, (
        "no uninformed order ever filled -- check that MAKER_OWNER, TAKER_OWNER "
        "and INFORMED_OWNER are distinct")
    assert f.informed_fills > 0, "no informed order ever filled"
    assert MAKER_OWNER != TAKER_OWNER != INFORMED_OWNER != MAKER_OWNER


def test_both_flow_types_are_material():
    """Neither population may be a rounding error, or the mix is not a mix."""
    _, _, f, _ = _run(steps=4000, informed_frac=0.35, informed_edge=1, seed=5)
    total = f.informed_fills + f.uninformed_fills
    share = f.informed_fills / total
    assert 0.10 < share < 0.70, f"informed share {share:.0%} is degenerate"


# -------------------------------------------------------------- book integrity

def test_book_survives_a_long_run():
    b, _, _, _ = _run(steps=5000, informed_frac=0.35, seed=6)
    b.check()
    assert b.best_bid is not None and b.best_ask is not None, "book emptied"
    assert b.spread > 0


def test_flow_is_deterministic_given_a_seed():
    a = _run(steps=800, seed=7)[3]
    c = _run(steps=800, seed=7)[3]
    assert a == c, "same seed must reproduce the same path"


def test_book_does_not_grow_without_bound():
    """Cancels must keep pace with posts, or the book is a memory leak."""
    b, _, _, _ = _run(steps=6000, informed_frac=0.35, seed=8)
    resting = len(b.open_order_ids())
    assert resting < 400, f"{resting} resting orders -- cancels are not keeping up"


if __name__ == "__main__":
    b, fv, f, errs = _run(steps=6000, informed_frac=0.35, informed_edge=1, seed=1)
    print(f"mid - fair : mean {st.mean(errs):+.2f}  sd {st.pstdev(errs):.2f}  "
          f"|max| {max(abs(e) for e in errs):.1f}")
    print(f"fills      : informed {f.informed_fills}  uninformed {f.uninformed_fills}")
    print(f"book       : {len(b.open_order_ids())} resting, spread {b.spread}")
