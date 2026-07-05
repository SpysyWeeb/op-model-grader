from opgrader.grading import S_CUTOFF
from opgrader.report import _grade_class


def test_grade_class_gold_for_perfect_score():
    assert _grade_class(100.0) == "ggold"
    assert _grade_class(S_CUTOFF) == "ggold"
    assert _grade_class(99.9) == "ggood"  # real near-miss, not gold
    assert _grade_class(80) == "ggood"
    assert _grade_class(65) == "gmid"
    assert _grade_class(40) == "gbad"
    assert _grade_class(None) == "gnone"
