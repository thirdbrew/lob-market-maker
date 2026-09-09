"""
test_frontier -- the calibration arithmetic and the dominance rule.

The project's headline conclusion is "naive dominates A-S at 7/7 of its risk
levels", and that sentence is only worth anything if `dominates` means what it
says and `fit_exponential` recovers a k it did not invent. Both are tested here
against hand-made inputs with known answers, because a calibration checked only
against its own simulator output is not checked.
"""

import math

import pytest

from frontier import Point, dominates, fit_exponential


# ----------------------------------------------------------------- calibration

def test_fit_recovers_a_known_decay_rate():
    """Generate lambda(d) = 3.5 * exp(-0.8*d) exactly; the fit must return it."""
    rows = [(d, 3.5 * math.exp(-0.8 * d)) for d in (1, 2, 3, 4, 6, 8)]
    out = fit_exponential(rows)
    assert out["k"] == pytest.approx(0.8, abs=1e-9)
    assert out["A"] == pytest.approx(3.5, rel=1e-9)
    assert out["r2"] == pytest.approx(1.0, abs=1e-9)


def test_fit_reports_a_poor_r2_when_the_shape_is_wrong():
    """R^2 is reported so a broken assumption is visible, not so it can be
    ignored. Feed it something that is not exponential."""
    rows = [(d, 1.0 / d) for d in (1, 2, 3, 4, 6, 8)]     # power law, not exp
    out = fit_exponential(rows)
    assert out["r2"] < 0.97, (
        f"a power law should not fit an exponential this well (R^2={out['r2']:.3f})")


def test_fit_ignores_zero_intensity_rows():
    """A distance that never filled contributes no information and must not
    become log(0)."""
    rows = [(1, 0.5), (2, 0.2), (3, 0.08), (9, 0.0)]
    out = fit_exponential(rows)
    assert out["usable"] == 3
    assert math.isfinite(out["k"])


def test_fit_falls_back_when_there_is_too_little_data():
    out = fit_exponential([(1, 0.5), (2, 0.0), (3, 0.0)])
    assert out["k"] == 1.5, "should fall back rather than fit two points"
    assert out["usable"] < 3


def test_calibrated_k_differs_from_the_guess_it_replaced():
    """Pins the finding: k was guessed at 1.5 and measured at ~0.63, so the
    guess put the A-S spread at less than half its calibrated width."""
    rows = [(1, 0.6546), (2, 0.2006), (3, 0.0847),
            (4, 0.0377), (6, 0.0117), (8, 0.0077)]
    out = fit_exponential(rows)
    assert out["k"] == pytest.approx(0.630, abs=0.01)
    assert out["r2"] > 0.90, "the exponential assumption should hold on this flow"
    assert abs(out["k"] - 1.5) > 0.5, "the guess was materially wrong"


# ------------------------------------------------------------------- dominance

def _pt(label, param, inv_sd, pnl):
    return Point(label, param, inv_sd, pnl, 0.0, 0.0, 0.0, 0.0)


def test_dominance_requires_no_more_risk():
    """A higher P&L at HIGHER inventory risk is not dominance."""
    a = [_pt("a", 1, 50.0, 1000.0)]          # better P&L, but far riskier
    b = [_pt("b", 1, 10.0, 500.0)]
    assert dominates(a, b) == {"checked": 0, "wins": 0}, (
        "a point at more risk must not be counted as beating one at less")


def test_dominance_counts_a_genuine_win():
    a = [_pt("a", 1, 8.0, 900.0)]            # less risk AND more P&L
    b = [_pt("b", 1, 10.0, 500.0)]
    assert dominates(a, b) == {"checked": 1, "wins": 1}


def test_dominance_counts_a_genuine_loss():
    a = [_pt("a", 1, 8.0, 100.0)]            # less risk, but worse P&L
    b = [_pt("b", 1, 10.0, 500.0)]
    assert dominates(a, b) == {"checked": 1, "wins": 0}


def test_dominance_uses_the_best_available_candidate():
    a = [_pt("a", 1, 9.0, 100.0), _pt("a", 2, 5.0, 900.0)]
    b = [_pt("b", 1, 10.0, 500.0)]
    assert dominates(a, b)["wins"] == 1, "should take a's best point at <= risk"


def test_dominance_is_not_symmetric_by_construction():
    """Guard the guard: if the rule returned the same answer both ways it would
    be measuring nothing."""
    a = [_pt("a", 1, 5.0, 900.0)]
    b = [_pt("b", 1, 10.0, 100.0)]
    assert dominates(a, b)["wins"] == 1
    assert dominates(b, a)["wins"] == 0
