"""
flow -- synthetic order flow, and the latent value the book is trying to track.

THE ONE DESIGN DECISION THAT MATTERS HERE IS INFORMED FLOW, and it is easy to
leave out.

A simulator whose market orders are all uninformed coin flips has **no adverse
selection in it**. A market maker quoting a symmetric spread around the true mid
then captures half the spread on every round trip and cannot lose except through
inventory, which makes it profitable by construction. Every "my market maker made
money" project on GitHub is that simulator, and its P&L number means nothing:
the decomposition in `evaluate.py` would have two of its three terms pinned at
zero by design.

So a fraction of market orders here are INFORMED: they arrive because the quote
is stale relative to the latent fair value, and they take the side that is
mispriced. That is the whole mechanism of adverse selection -- the fills a maker
regrets are exactly the ones it got because its price was wrong -- and it has to
be in the flow before it can be measured.

WHAT IS LATENT AND WHAT IS OBSERVED. `FairValue` is the true price. The agent
never sees it. It sees the book, and the book only tracks fair value because
informed traders drag it there. Handing the agent `v` directly would remove the
problem the agent exists to solve.

THE FLOW IS NOT CALIBRATED TO ANY REAL MARKET. Arrival rates, the depth profile
and the informed fraction are chosen to produce a book that behaves plausibly,
not to match an instrument. Every number that comes out of this simulator is a
property of these assumptions -- see the note in `evaluate.py` about what the
resulting P&L does and does not prove.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from book import Book, Side

# DISTINCT PARTICIPANTS, and this is not cosmetic. The book's self-trade
# prevention blocks orders sharing an owner, so when the noise makers and the
# uninformed takers were both owner 0, EVERY uninformed market order was silently
# refused -- 0 fills across 6,000 steps, while the run still looked healthy. The
# liquidity providers and the uninformed takers are different people; give them
# different ids or the flow has no uninformed trades in it at all.
MAKER_OWNER = 0          # noise liquidity providers, posting limits
TAKER_OWNER = 1          # uninformed market orders
INFORMED_OWNER = 3       # market orders that fire only on a stale quote


@dataclass(slots=True)
class FairValue:
    """Latent value as an integer-tick random walk. Never shown to an agent."""
    price: int = 10_000
    sigma_ticks: float = 1.0
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    def step(self) -> int:
        self.price += round(self.rng.gauss(0.0, self.sigma_ticks))
        return self.price


@dataclass
class FlowParams:
    limit_rate: float = 1.20        # expected new limit orders per step
    market_rate: float = 0.28       # expected market orders per step
    cancel_rate: float = 0.06       # per resting noise order, per step
    informed_frac: float = 0.35     # share of market orders that are informed
    informed_edge: int = 1          # ticks of mispricing an informed trader needs
    # Both tuned by measuring TRACKING, not by taste. Over 6,000 steps, mid-vs-fair
    # standard deviation: 15.73 ticks with no informed flow, 10.87 at edge=2,
    # 9.29 at edge=1, and back up to 11.29 at informed_frac=0.50 -- too much
    # informed flow thins the book faster than the makers replace it. 0.35/1 is
    # the measured floor, and it leaves a 31/69 informed/uninformed fill mix.
    depth_scale: float = 3.0        # mean distance of a new limit from the touch
    max_size: int = 8
    seed: int = 0


class Flow:
    """Emits one step of order flow against a book.

    Uninformed limit orders populate the book around the *observed* mid (or fair
    value when the book is empty). Uninformed market orders pick a side at
    random. Informed market orders fire only when the touch is stale by at least
    `informed_edge` ticks, and always take the mispriced side.
    """

    def __init__(self, book: Book, fair: FairValue, params: FlowParams | None = None):
        self.book = book
        self.fair = fair
        self.p = params or FlowParams()
        self.rng = random.Random(self.p.seed)
        self.noise_ids: list[int] = []
        self.informed_fills = 0
        self.uninformed_fills = 0

    # ------------------------------------------------------------------ pieces
    def _poisson(self, rate: float) -> int:
        """Knuth. Rates here are small, so the loop is short."""
        import math
        l, k, p = math.exp(-rate), 0, 1.0
        while True:
            p *= self.rng.random()
            if p <= l:
                return k
            k += 1

    def _reference(self) -> int:
        mid = self.book.mid
        return int(round(mid)) if mid is not None else self.fair.price

    def _post_limits(self, n: int) -> None:
        ref = self._reference()
        for _ in range(n):
            side = Side.BID if self.rng.random() < 0.5 else Side.ASK
            off = 1 + int(self.rng.expovariate(1.0 / self.p.depth_scale))
            price = ref - off if side is Side.BID else ref + off
            qty = self.rng.randint(1, self.p.max_size)
            oid, _ = self.book.limit(side, price, qty, owner=MAKER_OWNER)
            self.noise_ids.append(oid)

    def _cancels(self) -> None:
        if not self.noise_ids:
            return
        keep: list[int] = []
        for oid in self.noise_ids:
            order = self.book._orders.get(oid)
            if order is None or not order.is_open:
                continue
            if self.rng.random() < self.p.cancel_rate:
                self.book.cancel(oid)
            else:
                keep.append(oid)
        self.noise_ids = keep

    def _informed_side(self) -> Side | None:
        """Which side is mispriced enough to attract an informed taker?

        If the best ask sits below fair value, an informed trader BUYS it. If the
        best bid sits above fair value, they SELL into it. Ties and thin books
        produce nothing.
        """
        v = self.fair.price
        ask, bid = self.book.best_ask, self.book.best_bid
        buy_edge = (v - ask) if ask is not None else -1
        sell_edge = (bid - v) if bid is not None else -1
        if buy_edge >= self.p.informed_edge and buy_edge >= sell_edge:
            return Side.BID
        if sell_edge >= self.p.informed_edge:
            return Side.ASK
        return None

    def _markets(self, n: int) -> list:
        trades = []
        for _ in range(n):
            informed = self.rng.random() < self.p.informed_frac
            side = self._informed_side() if informed else None
            if informed and side is None:
                continue                      # nothing stale enough to pick off
            if side is None:
                side = Side.BID if self.rng.random() < 0.5 else Side.ASK
            qty = self.rng.randint(1, self.p.max_size)
            owner = INFORMED_OWNER if informed else TAKER_OWNER
            _, t = self.book.market(side, qty, owner=owner)
            if t:
                filled = sum(x.qty for x in t)
                if informed:
                    self.informed_fills += filled
                else:
                    self.uninformed_fills += filled
            trades.extend(t)
        return trades

    # -------------------------------------------------------------------- step
    def step(self) -> list:
        """Advance fair value, then post, cancel and take. Returns this step's trades."""
        self.fair.step()
        self._cancels()
        self._post_limits(self._poisson(self.p.limit_rate))
        return self._markets(self._poisson(self.p.market_rate))
