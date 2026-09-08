"""
agent -- two market makers, and the accounting that makes them comparable.

WHAT THE AGENT IS AND IS NOT ALLOWED TO SEE. It sees the book. It never sees
`FairValue`. Volatility is estimated from observed mid changes, the way a real
maker would have to. Handing it `v` would delete the problem it exists to solve
and turn adverse selection into a rounding error.

TWO AGENTS, ONE INTERFACE, SO THE COMPARISON IS LIKE FOR LIKE:

  NaiveMM     fixed symmetric half-spread around the observed mid. Ignores
              inventory entirely. This is the baseline almost every public LOB
              project stops at.

  AvellanedaStoikov   the closed form from Avellaneda & Stoikov (2008):

                  reservation price   r = s - q*gamma*sigma^2*(T-t)
                  optimal spread      d = gamma*sigma^2*(T-t)
                                          + (2/gamma)*ln(1 + gamma/k)

              and quotes r +/- d/2. Two distinct effects, and they are worth
              separating when explaining it:

              THE SKEW. `r` shifts away from the mid in proportion to inventory.
              Long inventory pushes both quotes DOWN -- the maker becomes eager
              to sell and reluctant to buy. This is inventory control, and it is
              what the naive agent lacks.

              THE WIDTH. `d` has a risk term that grows with gamma, sigma^2 and
              time remaining, plus a term from the order-arrival intensity k
              that prices how much edge you give up by quoting wider.

              As t -> T the risk terms vanish: with no time left there is no
              inventory risk to price, so the agent quotes tight and flat. That
              terminal behaviour is the easiest way to check the implementation
              is really the closed form and not a spread rule wearing its name.

FILL ACCOUNTING. A market maker is a MAKER: its fills arrive when somebody else's
order crosses into its resting quote, so they cannot be read off the return value
of its own submissions. `_collect_fills` diffs each resting order's remaining
quantity against what it was when posted. Cash and inventory move at the RESTING
price, never the mid -- getting that wrong silently books the spread twice.
"""

from __future__ import annotations

import math
from collections import deque

from book import Book, Side

MM_OWNER = 9


class MarketMaker:
    """Order lifecycle, inventory and cash. Subclasses only choose quotes."""

    def __init__(self, book: Book, size: int = 5, owner: int = MM_OWNER,
                 vol_window: int = 200):
        self.book = book
        self.size = size
        self.owner = owner
        self.inventory = 0
        self.cash = 0.0
        self.resting: dict[int, tuple[Side, int, int]] = {}   # oid -> side, px, rem
        self.fills: list[tuple[int, Side, int, int]] = []     # step, side, px, qty
        self._mids: deque = deque(maxlen=vol_window)
        self.step_idx = 0

    # -------------------------------------------------------------- estimation
    def observe(self) -> None:
        mid = self.book.mid
        if mid is not None:
            self._mids.append(mid)

    SIGMA_FLOOR = 0.05
    SIGMA_CAP = 3.0

    @property
    def sigma(self) -> float:
        """Robust per-step volatility of the OBSERVED mid, in ticks.

        MAD-based, not the sample standard deviation, and the reason is a real
        failure rather than a preference. The observed mid GAPS when the touch
        empties -- 50 to 120 ticks in a step -- and a sample variance squares
        those gaps. That is not a volatility estimate, it is a liquidity event
        being read as volatility, and it feeds straight back: inflated sigma ->
        wider A-S quotes -> thinner touch -> bigger gaps. The sample estimator
        overflowed a float in this simulator at gamma=0.1.

        THE FIX IS WINSORISED, NOT MAD, AND THE FIRST ATTEMPT WAS MAD. `1.4826 *
        MAD` is the textbook robust estimator and it is DEGENERATE here: the mid
        is unchanged on ~2 of every 3 steps, so the median diff is 0, the MAD is
        0, and sigma pinned to its floor on 49 of 49 samples. The skew term
        collapsed to 0.28 ticks and inventory control was silently off -- which
        is exactly the failure the floor was supposed to prevent, arriving
        through the estimator instead.

        Winsorising at the 98th percentile of |diff| keeps the sensitivity of a
        standard deviation while capping what one gap can contribute. The floor
        and cap remain as a backstop: sigma is never 0 and never large enough to
        run away.

        Never the true sigma -- the agent cannot see FairValue.
        """
        if len(self._mids) < 20:
            return 1.0
        d = [self._mids[i + 1] - self._mids[i] for i in range(len(self._mids) - 1)]
        mag = sorted(abs(x) for x in d)
        cap = mag[int(0.98 * (len(mag) - 1))] or 1.0
        w = [max(-cap, min(cap, x)) for x in d]
        m = sum(w) / len(w)
        var = sum((x - m) ** 2 for x in w) / max(len(w) - 1, 1)
        return min(max(math.sqrt(var), self.SIGMA_FLOOR), self.SIGMA_CAP)

    # ----------------------------------------------------------------- lifecycle
    def _collect_fills(self) -> None:
        """Diff resting quantity to find what was taken from us since last step."""
        for oid, (side, px, rem_at_post) in list(self.resting.items()):
            order = self.book._orders.get(oid)
            if order is None:
                continue
            filled = rem_at_post - order.remaining
            if filled > 0:
                signed = filled if side is Side.BID else -filled
                self.inventory += signed
                self.cash -= signed * px          # buy pays cash, sell receives
                self.fills.append((self.step_idx, side, px, filled))
                self.resting[oid] = (side, px, order.remaining)

    def resting_price(self, side: Side) -> int | None:
        """Price of our live quote on `side`, or None if we have none there."""
        for oid, (s, px, _) in self.resting.items():
            order = self.book._orders.get(oid)
            if s is side and order is not None and order.is_open:
                return px
        for _oid, (s, px, _) in self.resting.items():   # filled but still recorded
            if s is side:
                return px
        return None

    def _cancel_all(self) -> None:
        for oid in list(self.resting):
            self.book.cancel(oid)
        self.resting.clear()

    def quotes(self, t: int, horizon: int) -> tuple[int, int] | None:
        raise NotImplementedError

    def step(self, t: int, horizon: int) -> None:
        self.step_idx = t
        self.observe()
        self._collect_fills()
        self._cancel_all()

        q = self.quotes(t, horizon)
        if q is None:
            return
        bid, ask = q
        if bid >= ask:                 # never quote a crossed pair of our own
            return
        for side, price in ((Side.BID, bid), (Side.ASK, ask)):
            oid, _ = self.book.limit(side, price, self.size, owner=self.owner)
            order = self.book._orders[oid]
            if order.is_open:
                self.resting[oid] = (side, price, order.remaining)

    # ------------------------------------------------------------------ results
    def mark_to_market(self) -> float:
        """Cash plus inventory marked at the observed mid. The only P&L an agent
        can compute for itself -- it does not know fair value."""
        mid = self.book.mid
        if mid is None:
            mid = self._mids[-1] if self._mids else 0.0
        return self.cash + self.inventory * mid


class NaiveMM(MarketMaker):
    """Fixed half-spread around the mid. No inventory control at all."""

    def __init__(self, book: Book, half_spread: int = 2, **kw):
        super().__init__(book, **kw)
        self.half_spread = half_spread

    def quotes(self, t: int, horizon: int):
        mid = self.book.mid
        if mid is None:
            return None
        m = int(round(mid))
        return m - self.half_spread, m + self.half_spread


class AvellanedaStoikov(MarketMaker):
    """The 2008 closed form. gamma is risk aversion; k is arrival intensity."""

    def __init__(self, book: Book, gamma: float = 0.10, k: float = 1.5, **kw):
        super().__init__(book, **kw)
        self.gamma = gamma
        self.k = k

    def reservation_price(self, mid: float, tau: float) -> float:
        return mid - self.inventory * self.gamma * (self.sigma ** 2) * tau

    def optimal_spread(self, tau: float) -> float:
        risk = self.gamma * (self.sigma ** 2) * tau
        edge = (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)
        return risk + edge

    def quotes(self, t: int, horizon: int):
        mid = self.book.mid
        if mid is None:
            return None
        tau = max((horizon - t) / horizon, 0.0)      # normalised time remaining
        r = self.reservation_price(mid, tau)
        half = self.optimal_spread(tau) / 2.0
        bid, ask = int(math.floor(r - half)), int(math.ceil(r + half))
        if bid >= ask:                               # degenerate at tau -> 0
            bid, ask = int(math.floor(r)) - 1, int(math.ceil(r)) + 1
        return bid, ask
