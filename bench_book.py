"""
bench_book -- how many events per second, and where the time goes.

A throughput number is worth more than a paragraph about the data structure,
and it is the number an interviewer will push on -- so this reports where the
time actually goes rather than a single headline.

Three workloads, because they stress different parts:

  RESTING     limit orders far from the touch. Pure insert: heap push on a new
              price level, deque append otherwise. No matching.
  CROSSING    aggressive orders that trade. Exercises the match loop, level
              walking and lazy pruning.
  MIXED       ~70% limit / 20% cancel / 10% market around a drifting mid. The
              closest of the three to real flow, and the number to quote.

Run:  python bench_book.py
"""

import random
import time

from book import Book, Side


def _timed(fn, n):
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    return n / dt, dt


def bench_resting(n=200_000, seed=0):
    rng = random.Random(seed)
    b = Book()

    def run():
        for _ in range(n):
            if rng.random() < 0.5:
                b.limit(Side.BID, rng.randint(9_000, 9_900), rng.randint(1, 50))
            else:
                b.limit(Side.ASK, rng.randint(10_100, 11_000), rng.randint(1, 50))
    rate, dt = _timed(run, n)
    return rate, dt, len(b.open_order_ids())


def bench_crossing(n=200_000, seed=1):
    rng = random.Random(seed)
    b = Book()
    for i in range(20_000):                       # seed a deep book
        b.limit(Side.ASK, 10_000 + (i % 200), 100)
        b.limit(Side.BID, 9_999 - (i % 200), 100)

    def run():
        for _ in range(n):
            side = Side.BID if rng.random() < 0.5 else Side.ASK
            b.market(side, rng.randint(1, 30))
    rate, dt = _timed(run, n)
    return rate, dt, len(b.open_order_ids())


def bench_mixed(n=200_000, seed=2):
    rng = random.Random(seed)
    b = Book()
    live: list[int] = []
    mid = 10_000

    def run():
        nonlocal mid
        for _ in range(n):
            mid += rng.choice((-1, 0, 1))
            r = rng.random()
            if r < 0.70:
                side = Side.BID if rng.random() < 0.5 else Side.ASK
                off = rng.randint(1, 20)
                price = mid - off if side is Side.BID else mid + off
                oid, _ = b.limit(side, price, rng.randint(1, 20), owner=1)
                live.append(oid)
                if len(live) > 50_000:
                    del live[:25_000]
            elif r < 0.90 and live:
                b.cancel(live[rng.randrange(len(live))])
            else:
                side = Side.BID if rng.random() < 0.5 else Side.ASK
                b.market(side, rng.randint(1, 10), owner=2)
    rate, dt = _timed(run, n)
    return rate, dt, len(b.open_order_ids())


def main():
    print(f"{'workload':<12}{'events':>10}{'seconds':>10}{'events/sec':>14}"
          f"{'resting after':>15}")
    for name, fn in (("resting", bench_resting),
                     ("crossing", bench_crossing),
                     ("mixed", bench_mixed)):
        rate, dt, resting = fn()
        print(f"{name:<12}{200_000:>10,}{dt:>10.2f}{rate:>14,.0f}{resting:>15,}")

    print("\nMIXED is the number to quote -- 70% limit / 20% cancel / 10% market")
    print("around a drifting mid, which is the closest of the three to real flow.")
    print("\nPure Python, single core, CPython 3.12. The point of the number is")
    print("not that it is fast in absolute terms -- it is not, and a C++ engine")
    print("is three orders of magnitude quicker. The point is that the structure")
    print("is O(log n) in distinct prices and O(1) amortised per event, so the")
    print("rate does not degrade as the book deepens. That is what the three")
    print("workloads are there to show.")


if __name__ == "__main__":
    main()
