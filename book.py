"""
book -- a price-time-priority limit order book.

THE DATA STRUCTURE IS THE DESIGN DECISION, so it is made deliberately here
rather than inherited from whatever was convenient.

    _levels    (side, price) -> deque of order ids      FIFO within a price level
    _orders    order_id      -> Order                   O(1) lookup for cancel
    _volume    (side, price) -> resting quantity        O(1) depth, no scanning
    _bids      max-heap of negated prices               best bid in O(1) peek
    _asks      min-heap of prices                       best ask in O(1) peek

Why this and not the alternatives:

  A sorted list of price levels gives O(n) insertion on every new level, and a
  book that walks away from its old prices creates levels constantly.

  A balanced tree (or `sortedcontainers.SortedDict`) is asymptotically fine and
  reads well, but pays constant factors on the hot path -- best-price lookup --
  that a heap does not, and best-price lookup happens on every single event.

  A flat array indexed by tick is the fastest thing possible and is what a real
  exchange does, but it assumes a dense, bounded price range. Under a random
  walk with no bounds that is a memory leak with extra steps.

  The heaps carry STALE PRICES on purpose. Removing a price from a binary heap
  is O(n); instead an emptied level is left in place and skipped when it
  surfaces (`_prune`). Every price is pushed once and popped at most once, so
  the amortised cost stays O(log n) and the heap never exceeds the number of
  distinct prices ever touched.

CANCELS ARE LAZY, AND THAT IS A CORRECTNESS DECISION AS MUCH AS A SPEED ONE. A
deque has no O(1) removal by value, so a cancelled order is marked cancelled and
skipped when it reaches the front of its level. The alternative -- an intrusive
doubly-linked list per level -- is genuinely O(1) and is what production engines
use, but it is more code to get wrong, and the resting quantity has to be
decremented at cancel time either way (see `_volume`), which is what actually
keeps depth honest.

PRICES ARE INTEGER TICKS. NEVER FLOATS. A float book compares 10.1 against
10.099999999999999 and puts an order on the wrong side of the spread for reasons
that will not show up in any test written on round numbers. Convert at the edge.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum


class Side(IntEnum):
    BID = 0
    ASK = 1

    @property
    def opposite(self) -> "Side":
        return Side.ASK if self is Side.BID else Side.BID


@dataclass(slots=True)
class Order:
    id: int
    side: Side
    price: int          # integer ticks; None-priced market orders never rest
    qty: int            # original size
    remaining: int
    ts: int
    owner: int | None = None
    cancelled: bool = False

    @property
    def is_open(self) -> bool:
        return self.remaining > 0 and not self.cancelled


@dataclass(slots=True, frozen=True)
class Trade:
    """One fill. `aggressor` took liquidity; `resting` provided it."""
    price: int
    qty: int
    aggressor_id: int
    resting_id: int
    aggressor_side: Side
    ts: int


@dataclass(slots=True)
class Book:
    _orders: dict[int, Order] = field(default_factory=dict)
    _levels: dict[tuple[Side, int], deque] = field(default_factory=dict)
    _volume: dict[tuple[Side, int], int] = field(default_factory=dict)
    _bids: list[int] = field(default_factory=list)      # max-heap via negation
    _asks: list[int] = field(default_factory=list)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))
    ts: int = 0

    # ---------------------------------------------------------------- queries
    def _prune(self, side: Side) -> None:
        """Drop heap entries whose level is empty. Lazy deletion, paid on read."""
        heap = self._bids if side is Side.BID else self._asks
        while heap:
            price = -heap[0] if side is Side.BID else heap[0]
            if self._volume.get((side, price), 0) > 0:
                return
            heapq.heappop(heap)
            self._levels.pop((side, price), None)
            self._volume.pop((side, price), None)

    def best(self, side: Side) -> int | None:
        self._prune(side)
        heap = self._bids if side is Side.BID else self._asks
        if not heap:
            return None
        return -heap[0] if side is Side.BID else heap[0]

    @property
    def best_bid(self) -> int | None:
        return self.best(Side.BID)

    @property
    def best_ask(self) -> int | None:
        return self.best(Side.ASK)

    @property
    def spread(self) -> int | None:
        b, a = self.best_bid, self.best_ask
        return None if b is None or a is None else a - b

    @property
    def mid(self) -> float | None:
        b, a = self.best_bid, self.best_ask
        return None if b is None or a is None else (a + b) / 2.0

    def depth(self, side: Side, price: int) -> int:
        return self._volume.get((side, price), 0)

    def open_order_ids(self) -> list[int]:
        return [i for i, o in self._orders.items() if o.is_open]

    # ------------------------------------------------------------------ write
    def _rest(self, order: Order) -> None:
        key = (order.side, order.price)
        if key not in self._levels:
            self._levels[key] = deque()
            self._volume[key] = 0
            if order.side is Side.BID:
                heapq.heappush(self._bids, -order.price)
            else:
                heapq.heappush(self._asks, order.price)
        self._levels[key].append(order.id)
        self._volume[key] += order.remaining

    def _crosses(self, side: Side, price: int | None, book_price: int) -> bool:
        """A market order (price None) crosses anything. A limit order crosses
        only through its own limit."""
        if price is None:
            return True
        return price >= book_price if side is Side.BID else price <= book_price

    def _match(self, taker: Order) -> list[Trade]:
        """Walk the opposite side, best price first, FIFO within each level."""
        trades: list[Trade] = []
        other = taker.side.opposite

        while taker.remaining > 0:
            book_price = self.best(other)
            if book_price is None or not self._crosses(taker.side, taker.price,
                                                       book_price):
                break

            key = (other, book_price)
            queue = self._levels[key]
            skipped: list[int] = []
            while queue and taker.remaining > 0:
                maker = self._orders[queue[0]]
                if not maker.is_open:
                    queue.popleft()            # cancelled or exhausted; skip it
                    continue
                if (taker.owner is not None
                        and maker.owner == taker.owner):
                    # SELF-TRADE PREVENTION, at the PARTICIPANT level.
                    #
                    # The first version of this compared ORDER IDS, which is dead
                    # code: a taker is given a fresh id and matched before it
                    # rests, so no resting order can ever share its id. A
                    # mutation test caught it -- deleting the guard changed
                    # nothing, because it never fired.
                    #
                    # The real case is a market maker quoting both sides: if the
                    # fair value drifts, its own bid can cross its own ask. That
                    # is a participant colliding with itself, not an order.
                    #
                    # Policy is SKIP-THEN-CANCEL-NEWEST. The taker steps over
                    # its own resting order and matches the next one at that
                    # level. If it is still unfilled and its price would cross
                    # the opposite best -- which can now only be its own order --
                    # the REMAINDER IS CANCELLED rather than rested (see
                    # `limit`).
                    #
                    # Pure skip is not an option, and the first version of this
                    # tried it: the taker rested at its limit, sitting at the
                    # same price as its own order on the other side, and the
                    # book came back CROSSED (bid 100 >= ask 100). A test caught
                    # it immediately. Cancel-newest is what CME does for the
                    # same reason -- it is the only one of the three policies
                    # that never removes liquidity the participant did not
                    # withdraw AND never leaves the book crossed.
                    skipped.append(queue.popleft())
                    continue

                qty = min(taker.remaining, maker.remaining)
                maker.remaining -= qty
                taker.remaining -= qty
                self._volume[key] -= qty
                trades.append(Trade(book_price, qty, taker.id, maker.id,
                                    taker.side, self.ts))
                if maker.remaining == 0:
                    queue.popleft()

            for oid in reversed(skipped):      # put own orders back, in order
                queue.appendleft(oid)
            if skipped and taker.remaining > 0:
                break                          # nothing left here but our own

            if self._volume.get(key, 0) <= 0:
                self._prune(other)
            elif queue and not taker.remaining:
                break
            elif not queue:
                break
        return trades

    def limit(self, side: Side, price: int, qty: int,
              owner: int | None = None) -> tuple[int, list[Trade]]:
        """Submit a limit order. Crosses what it can, rests the remainder.

        `owner` identifies the participant. Orders sharing a non-None owner
        never trade with each other -- see the note in `_match`.
        """
        if qty <= 0:
            raise ValueError("qty must be positive")
        self.ts += 1
        oid = next(self._ids)
        order = Order(oid, side, price, qty, qty, self.ts, owner)
        self._orders[oid] = order
        trades = self._match(order)
        if order.remaining > 0:
            opposite_best = self.best(side.opposite)
            if opposite_best is not None and self._crosses(side, price,
                                                           opposite_best):
                # Still crossing after matching means the only thing in the way
                # is our own resting order. Resting here would cross the book.
                order.cancelled = True
            else:
                self._rest(order)
        return oid, trades

    def market(self, side: Side, qty: int,
               owner: int | None = None) -> tuple[int, list[Trade]]:
        """Marketable order with no limit. Any unfilled remainder is DISCARDED,
        not rested -- a market order that rests is a limit order, and conflating
        them is how a book quietly accumulates phantom liquidity."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        self.ts += 1
        oid = next(self._ids)
        order = Order(oid, side, None, qty, qty, self.ts, owner)
        self._orders[oid] = order
        trades = self._match(order)
        order.remaining = 0
        return oid, trades

    def cancel(self, order_id: int) -> bool:
        """Cancel a resting order. False if unknown, already filled, or already
        cancelled -- a cancel racing a fill is normal, not an error."""
        order = self._orders.get(order_id)
        if order is None or not order.is_open:
            return False
        order.cancelled = True
        key = (order.side, order.price)
        if key in self._volume:
            self._volume[key] -= order.remaining
            if self._volume[key] <= 0:
                self._prune(order.side)
        return True

    # ------------------------------------------------------------- invariants
    def check(self) -> None:
        """Assert the structure is self-consistent. Tests call this after every
        operation; nothing in the hot path does."""
        for (side, price), queue in self._levels.items():
            live = sum(self._orders[i].remaining for i in queue
                       if self._orders[i].is_open)
            assert self._volume[(side, price)] == live, (
                f"volume desync at {side.name} {price}: "
                f"tracked {self._volume[(side, price)]} vs actual {live}")
        b, a = self.best_bid, self.best_ask
        assert b is None or a is None or b < a, (
            f"crossed book: bid {b} >= ask {a}")
        for price in (-p for p in self._bids):
            assert self._volume.get((Side.BID, price), 0) >= 0
