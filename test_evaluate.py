"""
test_evaluate -- the decomposition must be an identity, not an estimate.

If spread + adverse + inventory only approximately equals the total, the split is
a story rather than an accounting, and any conclusion drawn from the shares is
unfalsifiable. So the first two tests are exactness tests, and they are the ones
that matter.

The sign tests come second: an identity that reconstructs the total while
attributing everything to the wrong term is still wrong. Adverse selection must
respond to how much informed flow exists, and spread capture must respond to how
wide the maker quotes.
"""

import statistics as st

import pytest

from agent import AvellanedaStoikov, NaiveMM
from evaluate import (decompose, horizon_sweep, realised_pnl, report,
                      simulate)
from flow import FlowParams

STEPS = 3000


def _naive(hs=2):
    return lambda b: NaiveMM(b, half_spread=hs)


# ------------------------------------------------------------------ exactness

def test_the_three_terms_sum_to_the_total():
    res = simulate(_naive(), steps=STEPS)
    d = decompose(res)
    assert d["spread"] + d["adverse"] + d["inventory"] == pytest.approx(
        d["total"], rel=1e-9, abs=1e-6), "the decomposition must be an identity"


def test_the_total_equals_the_realised_pnl():
    """The decomposition must reconstruct the P&L that actually happened --
    cash plus final inventory marked at the final fair value."""
    res = simulate(_naive(), steps=STEPS)
    assert decompose(res)["total"] == pytest.approx(realised_pnl(res), abs=1e-6)


def test_the_total_is_invariant_to_the_horizon():
    """The horizon is a measurement choice. It moves the boundary between
    adverse selection and inventory; it must not move the total."""
    res = simulate(_naive(), steps=STEPS)
    totals = [d["total"] for d in horizon_sweep(res)]
    assert max(totals) - min(totals) < 1e-6, f"total moved with horizon: {totals}"


def test_the_horizon_does_move_the_split():
    """Guard the guard: if the horizon changed nothing, the sweep would be
    reporting a constant and the sensitivity claim would be empty."""
    res = simulate(_naive(), steps=STEPS)
    sweep = horizon_sweep(res, horizons=(10, 160))
    assert abs(sweep[0]["adverse"] - sweep[1]["adverse"]) > 1.0, (
        "adverse selection should depend on the window it is measured over")


def test_an_agent_that_never_quotes_decomposes_to_zero():
    """An agent that posts NOTHING, not one that posts far away.

    The first version of this used half_spread=100_000 and still took a fill: a
    market order carries no price limit, so it sweeps to whatever depth it needs.
    Quoting far away is not the same as not quoting, which is worth knowing
    before designing a "stay out of the way" agent.
    """
    class Silent(NaiveMM):
        def quotes(self, t, horizon):
            return None

    res = simulate(lambda b: Silent(b), steps=500)
    d = decompose(res)
    assert d["fills"] == 0
    assert d["total"] == d["spread"] == d["adverse"] == d["inventory"] == 0.0


# ---------------------------------------------------------------- attribution

def test_adverse_selection_tracks_informed_flow():
    """The term must respond to the thing it claims to measure."""
    def adverse(frac, seed):
        res = simulate(_naive(), seed=seed, steps=STEPS,
                       flow_params=FlowParams(seed=seed, informed_frac=frac))
        d = decompose(res)
        return d["adverse_per_unit"]

    quiet = st.mean([adverse(0.0, s) for s in (11, 12, 13)])
    noisy = st.mean([adverse(0.6, s) for s in (11, 12, 13)])
    assert noisy < quiet, (
        f"more informed flow must cost the maker more: {quiet:.3f} at frac=0 vs "
        f"{noisy:.3f} at frac=0.6 (both per unit, negative is a cost)")


def test_quoting_wider_earns_more_spread_per_unit():
    """Spread capture is what you are paid for liquidity, so it must scale with
    how far from the mid you are willing to quote."""
    tight = decompose(simulate(_naive(2), steps=STEPS))["spread_per_unit"]
    wide = decompose(simulate(_naive(6), steps=STEPS))["spread_per_unit"]
    assert wide > tight, (
        f"a wider quote must earn more per unit: {tight:.3f} at hs=2 vs "
        f"{wide:.3f} at hs=6")
    assert tight > 0, "a passive maker quoting around the mid should capture spread"


def test_adverse_selection_is_a_cost():
    """It is a cost of doing business, not an occasional accident. With informed
    flow present it must be negative."""
    res = simulate(_naive(), steps=STEPS,
                   flow_params=FlowParams(seed=11, informed_frac=0.35))
    assert decompose(res)["adverse"] < 0


def test_profit_can_be_inventory_rather_than_spread():
    """The finding this module exists to surface.

    A naive maker can post a positive total while its spread capture is dwarfed
    by an inventory swing -- it did not get paid for liquidity, it got lucky on a
    drifting position. A bare P&L curve calls that a profitable market maker.
    """
    res = simulate(_naive(2), steps=6000)
    d = decompose(res)
    assert d["fills"] > 100
    assert abs(d["inventory"]) > abs(d["spread"]), (
        "expected the inventory term to dominate spread capture for a naive "
        "maker; if this flips, re-read the run before trusting the total")


def test_decomposition_holds_for_the_avellaneda_stoikov_agent():
    res = simulate(lambda b: AvellanedaStoikov(b, gamma=0.5), steps=STEPS)
    d = decompose(res)
    assert d["spread"] + d["adverse"] + d["inventory"] == pytest.approx(
        d["total"], abs=1e-6)
    assert d["total"] == pytest.approx(realised_pnl(res), abs=1e-6)


def test_adverse_selection_is_measured_against_the_LATENT_value():
    """Closes a mutation gap: swapping fair value for the observed mid in the
    adverse term survived every other test, because the identity telescopes
    regardless and the mid normally tracks value closely.

    The distinction is semantic and it matters. Against the MID you measure "did
    the market move against me", which includes your own impact. Against the
    LATENT VALUE you measure "was my counterparty informed", which is what
    adverse selection means.

    So construct a case where they disagree: value falls, the mid never moves.
    """
    from book import Side
    from evaluate import RunResult

    class _Stub:
        cash = 0.0
        inventory = 0
        fills = [(0, Side.BID, 999, 10)]        # bought 10 at 999 at t=0

    res = RunResult(mm=_Stub())
    res.v_path = [1000] * 50 + [900] * 51       # latent value drops at t=50
    res.mid_path = [1000.0] * 101               # observed mid never moves

    d = decompose(res, horizon=60)              # v[60] = 900, mid[0] = 1000

    assert d["adverse"] == pytest.approx(10 * (900 - 1000)), (
        "adverse selection must use the LATENT value at the horizon; measured "
        f"{d['adverse']:.1f}, expected -1000. A mid-based term gives 0 here.")
    assert d["spread"] == pytest.approx(10 * (1000 - 999))
    assert d["spread"] + d["adverse"] + d["inventory"] == pytest.approx(d["total"])
