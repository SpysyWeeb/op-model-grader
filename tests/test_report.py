from opgrader.grading import S_CUTOFF, letter
from opgrader.report import _grade_class, _lettered_score


def test_grade_class_gold_for_perfect_score():
    assert _grade_class(100.0) == "ggold"
    assert _grade_class(S_CUTOFF) == "ggold"
    assert _grade_class(99.9) == "ggood"  # real near-miss, not gold
    assert _grade_class(80) == "ggood"
    assert _grade_class(65) == "gmid"
    assert _grade_class(40) == "gbad"
    assert _grade_class(None) == "gnone"


def test_lettered_score_never_shows_100_unless_truly_s_tier():
    """A score like 99.6 rounds to '100' via plain :.0f -- displayed right
    next to a non-S letter, that reads as a bug ('says 100 but isn't S?').
    '100' must be reserved for genuine S-tier scores."""
    for near_miss in (99.5, 99.6, 99.95, 99.999):
        assert letter(near_miss) != "S"
        assert _lettered_score(near_miss) == "99"
    for perfect in (S_CUTOFF, 100.0):
        assert letter(perfect) == "S"
        assert _lettered_score(perfect) == "100"
    assert _lettered_score(87.4) == "87"  # ordinary case unaffected
