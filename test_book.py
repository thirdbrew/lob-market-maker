"""
test_book -- the matching engine, by invariant rather than by example.

WHY PROPERTY TESTS HERE SPECIFICALLY. A matching engine is easy to write so that
it passes every example you think to write down. The bugs live in the sequences
nobody types by hand: a limit order sweeping three levels and resting the
remainder, a cancel arriving for an order that a fill already exhausted, a
partial fill at the front of a queue with a cancelled order behind it. Hypothesis
generates those sequences; the invariants below are what must hold after every
one of them.

Four invariants carry the engine:

  NEVER CROSSED     best_bid < best_ask, always. If this breaks, liquidity that
                    should have traded is sitting in the book.
  VOLUME HONEST     tracked depth equals the sum of live resting quantity. This
                    is the one lazy cancellation can silently break.
  CONSERVATION      every unit that leaves an aggressor arrives at a resting
                    order. Fills are not created or destroyed.
  PRICE-TIME        at one price, the earlier order fills first. This is the
                    rule the whole structure exists to enforce.
"""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from book import Book, Side

# --------------------------------------------------------------- named cases


def test_resting_then_crossing():
    b = Book()
    b.limit(Side.ASK, 101, 10)
    _, trades = b.limit(Side.BID, 101, 4)
    assert [(t.price, t.qty) for t in trades] == [(101, 4)]
    assert b.depth(Side.ASK, 101) == 6
    assert b.best_bid is None, "fully-filled aggressor must not rest"
    b.check()


def test_partial_fill_rests_remainder():
    b = Book()
    b.limit(Side.ASK, 100, 3)
    oid, trades = b.limit(Side.BID, 100, 10)
    assert sum(t.qty for t in trades) == 3
    assert b.depth(Side.BID, 100) == 7, "unfilled remainder must rest"
    assert b.best_bid == 100 and b.best_ask is None
    b.check()


def test_sweeps_multiple_levels_best_price_first():
    b = Book()
    b.limit(Side.ASK, 102, 5)
    b.limit(Side.ASK, 100, 5)
    b.limit(Side.ASK, 101, 5)
    _, trades = b.limit(Side.BID, 102, 12)
    assert [t.price for t in trades] == [100, 101, 102], "best price first"
    assert [t.qty for t in trades] == [5, 5, 2]
    assert b.depth(Side.ASK, 102) == 3
    b.check()


def test_price_time_priority_within_a_level():
    b = Book()
    first, _ = b.limit(Side.ASK, 100, 5)
    second, _ = b.limit(Side.ASK, 100, 5)
    _, trades = b.limit(Side.BID, 100, 5)
    assert [t.resting_id for t in trades] == [first], "earlier order fills first"
    b.check()


def test_limit_does_not_trade_through_its_price():
    b = Book()
    b.limit(Side.ASK, 105, 10)
    oid, trades = b.limit(Side.BID, 100, 10)
    assert trades == [], "a bid at 100 must not lift an offer at 105"
    assert b.depth(Side.BID, 100) == 10
    b.check()


def test_market_order_never_rests():
    b = Book()
    b.limit(Side.ASK, 100, 3)
    _, trades = b.market(Side.BID, 10)
    assert sum(t.qty for t in trades) == 3
    assert b.best_bid is None, "unfilled market remainder is discarded, not rested"
    b.check()


def test_cancel_removes_liquidity():
    b = Book()
    oid, _ = b.limit(Side.BID, 99, 7)
    assert b.depth(Side.BID, 99) == 7
    assert b.cancel(oid) is True
    assert b.depth(Side.BID, 99) == 0
    assert b.best_bid is None
    assert b.cancel(oid) is False, "double cancel is False, not an error"
    b.check()


def test_cancelled_order_is_skipped_not_filled():
    b = Book()
    dead, _ = b.limit(Side.ASK, 100, 5)
    live, _ = b.limit(Side.ASK, 100, 5)
    b.cancel(dead)
    _, trades = b.limit(Side.BID, 100, 5)
    assert [t.resting_id for t in trades] == [live], "cancelled order must not fill"
    b.check()


def test_cancel_after_full_fill_is_false():
    b = Book()
    oid, _ = b.limit(Side.ASK, 100, 5)
    b.limit(Side.BID, 100, 5)
    assert b.cancel(oid) is False, "a cancel racing a fill is normal, not an error"
    b.check()


def test_zero_and_negative_quantities_rejected():
    b = Book()
    for q in (0, -1):
        with pytest.raises(ValueError):
            b.limit(Side.BID, 100, q)
        with pytest.raises(ValueError):
            b.market(Side.BID, q)


# ------------------------------------------------------- the state machine

class BookMachine(RuleBasedStateMachine):
    """Random op sequences; the invariants must survive all of them."""

    def __init__(self):
        super().__init__()
        self.book = Book()
        self.live: list[int] = []
        self.aggressor_filled = 0
        self.resting_filled = 0

    def _record(self, trades):
        for t in trades:
            self.aggressor_filled += t.qty
            self.resting_filled += t.qty

    @rule(side=st.sampled_from(Side), price=st.integers(95, 105),
          qty=st.integers(1, 20))
    def submit_limit(self, side, price, qty):
        oid, trades = self.book.limit(side, price, qty)
        self._record(trades)
        self.live.append(oid)

    @rule(side=st.sampled_from(Side), qty=st.integers(1, 20))
    def submit_market(self, side, qty):
        _, trades = self.book.market(side, qty)
        self._record(trades)

    @rule(data=st.data())
    def cancel_one(self, data):
        if not self.live:
            return
        oid = data.draw(st.sampled_from(self.live))
        self.book.cancel(oid)

    @invariant()
    def never_crossed(self):
        b, a = self.book.best_bid, self.book.best_ask
        assert b is None or a is None or b < a, f"crossed book: {b} >= {a}"

    @invariant()
    def volume_is_honest(self):
        self.book.check()

    @invariant()
    def fills_are_conserved(self):
        assert self.aggressor_filled == self.resting_filled

    @invariant()
    def resting_orders_have_a_price(self):
        for oid in self.book.open_order_ids():
            assert self.book._orders[oid].price is not None, (
                "a market order must never be resting")


TestBookMachine = BookMachine.TestCase
TestBookMachine.settings = settings(
    max_examples=250, stateful_step_count=60, deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much])


# --------------------------------------------------------- conservation, bulk

@given(ops=st.lists(
    st.tuples(st.sampled_from(Side), st.integers(95, 105), st.integers(1, 15)),
    min_size=1, max_size=120))
@settings(max_examples=200, deadline=None)
def test_total_traded_matches_book_depletion(ops):
    """Units in equal units resting plus units traded. Nothing evaporates."""
    b = Book()
    submitted = {Side.BID: 0, Side.ASK: 0}
    traded = 0
    for side, price, qty in ops:
        _, trades = b.limit(side, price, qty)
        submitted[side] += qty
        traded += sum(t.qty for t in trades)
    b.check()

    resting = {Side.BID: 0, Side.ASK: 0}
    for oid in b.open_order_ids():
        o = b._orders[oid]
        resting[o.side] += o.remaining

    for side in (Side.BID, Side.ASK):
        assert submitted[side] == resting[side] + traded, (
            f"{side.name}: submitted {submitted[side]} != "
            f"resting {resting[side]} + traded {traded}")


# ------------------------------------------- self-trade prevention (owner-level)

def test_participant_does_not_trade_with_itself():
    """A market maker quoting both sides must not fill its own resting order.

    The guard this tests replaced one comparing ORDER IDS, which was dead code --
    a taker is matched before it rests, so no resting order can share its id. A
    mutation test caught that: deleting the guard changed nothing.
    """
    b = Book()
    mine, _ = b.limit(Side.ASK, 100, 5, owner=7)
    _, trades = b.limit(Side.BID, 100, 5, owner=7)
    assert trades == [], "a participant must not trade with itself"
    assert b.depth(Side.ASK, 100) == 5, "own resting order stays put"
    assert b.best_bid is None, (
        "cancel-newest: the blocked remainder must NOT rest, or the book crosses "
        "against the participant's own order -- which is how the first version "
        "of this failed")
    b.check()


def test_self_trade_prevention_steps_over_to_a_stranger():
    """Skip policy: step over own order, fill the next one at that level."""
    b = Book()
    mine, _ = b.limit(Side.ASK, 100, 5, owner=7)
    theirs, _ = b.limit(Side.ASK, 100, 5, owner=9)
    _, trades = b.limit(Side.BID, 100, 5, owner=7)
    assert [t.resting_id for t in trades] == [theirs]
    assert b.depth(Side.ASK, 100) == 5, "own order untouched and still resting"
    b.check()


def test_different_owners_trade_normally():
    b = Book()
    b.limit(Side.ASK, 100, 5, owner=1)
    _, trades = b.limit(Side.BID, 100, 5, owner=2)
    assert sum(t.qty for t in trades) == 5
    b.check()


def test_ownerless_orders_are_unaffected():
    """owner=None means anonymous flow; two None owners still trade."""
    b = Book()
    b.limit(Side.ASK, 100, 5)
    _, trades = b.limit(Side.BID, 100, 5)
    assert sum(t.qty for t in trades) == 5
    b.check()
