"""
test_flow -- does the simulated market actually behave like one?

A synthetic flow generator is trivially "correct" -- it runs, it emits orders,
nothing raises. The question that matters is whether it produces the *mechanism*
the rest of the project needs, and that is testable.

The load-bearing property is TRACKING. The agent never sees fair value; it sees
a book, and the book is only informative because informed traders drag it toward
fair value. If informed flow does not measurably improve tracking, there is no
adverse selection in the simulator, and `evaluate.py`'s decomposition has two of
its three terms pinned at zero by construction.

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

def test_informed_flow_improves_tracking():
    """The property the whole simulator exists to have.

    Without informed traders nothing connects the book to fair value: noise
    makers post around the observed mid, which is self-referential, so the book
    is free to wander. Informed flow is the only force pulling it back.
    """
    _, _, _, without = _run(informed_frac=0.0, seed=1)
    _, _, _, with_ = _run(informed_frac=0.35, informed_edge=1, seed=1)

    sd_without, sd_with = st.pstdev(without), st.pstdev(with_)
    assert sd_with < sd_without * 0.85, (
        f"informed flow must materially improve tracking: "
        f"sd {sd_without:.2f} without vs {sd_with:.2f} with. If these are close, "
        f"there is no adverse selection in this simulator and the P&L "
        f"decomposition downstream is measuring nothing.")


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
