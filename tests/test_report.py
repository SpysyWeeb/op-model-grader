from opgrader.grading import CategoryResult, METRIC_BY_KEY, S_CUTOFF, MetricResult, letter
from opgrader.report import _category_card, _grade_class, _lettered_score, _metric_rows


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


def test_metric_rows_hides_unscored_diagnostics():
    """scorer='none' rows (cmd_unwind_lead_*, curve_*) are computed but not
    worth a table row -- they'd otherwise clutter the card with 'diagnostic,
    not scored' filler."""
    d = METRIC_BY_KEY["cmd_unwind_lead_left"]
    assert d.scorer == "none"
    assert d.show_unscored is False
    m = MetricResult(definition=d, model_vals=[0.4, 0.5, 0.6], driver_vals=[])
    m.model_agg = 0.5
    assert _metric_rows([m]) == ""


def test_metric_rows_show_unscored_escape_hatch():
    """turn_effort_* is scorer='none' too (no defensible scoring formula
    yet) but IS worth showing -- show_unscored=True keeps it out of the
    blanket hide rule above."""
    d = METRIC_BY_KEY["turn_effort_left"]
    assert d.scorer == "none"
    assert d.show_unscored is True
    m = MetricResult(definition=d, model_vals=[40.0, 60.0, 90.0], driver_vals=[])
    m.model_agg = 60.0
    html = _metric_rows([m])
    assert html != ""
    assert "60.00" in html
    assert "not scored" in html  # still reads as context, not a grade


def test_metric_rows_row_override_replaces_model_you_cells():
    d = METRIC_BY_KEY["resisted_divergence_left"]
    m = MetricResult(definition=d, model_vals=[70.0, 80.0, 90.0], driver_vals=[])
    m.model_agg = 80.0
    m.score = 51.0
    html = _metric_rows([m], {"resisted_divergence_left": {"model_deg": 82.3, "you_deg": 45.1, "n": 12}})
    assert "82.30" in html and "45.10" in html and "(n=12)" in html
    assert "80.00" not in html  # the raw model_agg must not leak through when overridden


def test_metric_rows_no_override_uses_normal_rendering():
    d = METRIC_BY_KEY["resisted_divergence_left"]
    m = MetricResult(definition=d, model_vals=[70.0, 80.0, 90.0], driver_vals=[])
    m.model_agg = 80.0
    m.score = 51.0
    html = _metric_rows([m])
    assert "80.00" in html


def test_category_card_turns_onset_lead_shows_reference_you():
    """cmd_onset_lead's 'You' is always exactly 0.00 by construction (the
    lead/lag is measured relative to when the wheel itself turned in) --
    _category_card must wire that reference text in, not leave the normal
    (empty, since needs_driver=False) You cell showing '-'."""
    d = METRIC_BY_KEY["cmd_onset_lead_left"]
    m = MetricResult(definition=d, model_vals=[0.3, 0.4, 0.5], driver_vals=[])
    m.model_agg, m.score = 0.4, 62.0
    cat = CategoryResult(name="Turns", weight=0.50, metrics=[m])
    html = _category_card(cat)
    assert "0.00" in html and "reference" in html
    assert "0.40" in html  # the real model_agg is untouched


def test_category_card_turns_resisted_angles_wired_from_extra():
    d = METRIC_BY_KEY["resisted_divergence_left"]
    m = MetricResult(definition=d, model_vals=[70.0, 80.0, 90.0], driver_vals=[])
    m.model_agg, m.score = 80.0, 51.0
    cat = CategoryResult(
        name="Turns", weight=0.50, metrics=[m],
        extra={"resisted_angles": {"left": {"model_deg": 82.3, "you_deg": 45.1, "n": 12}}},
    )
    html = _category_card(cat)
    assert "82.30" in html and "45.10" in html and "(n=12)" in html


def test_metric_rows_desc_renders_clickable_row_with_hidden_desc_row():
    """A metric with `desc` set gets a clickable row (hasdesc class, a
    rowinfo cue icon, a data-desc-id) plus a matching hidden descrow
    holding the explanation text -- the click-to-expand mechanism."""
    d = METRIC_BY_KEY["turn_effort_left"]
    assert d.desc  # sanity: this metric is expected to carry a description
    m = MetricResult(definition=d, model_vals=[40.0, 60.0, 90.0], driver_vals=[])
    m.model_agg = 60.0
    html = _metric_rows([m])
    assert 'class="hasdesc insuff"' in html
    assert 'data-desc-id="desc-turn_effort_left"' in html
    assert '<span class="rowinfo">' in html
    assert 'id="desc-turn_effort_left"' in html
    assert "highest % of available steering torque" in html  # the desc text itself, apostrophe-free substring


def test_metric_rows_no_desc_renders_plain_row_no_click_affordance():
    d = METRIC_BY_KEY["rms_jerk"]
    assert d.desc == ""  # sanity: this metric has no description authored
    m = MetricResult(definition=d, model_vals=[0.1, 0.2, 0.3], driver_vals=[0.1, 0.2, 0.3])
    m.model_agg, m.driver_agg, m.score = 0.2, 0.2, 100.0
    html = _metric_rows([m])
    assert "hasdesc" not in html
    assert "rowinfo" not in html
    assert "descrow" not in html


def test_metric_rows_desc_star_ordering_scored_row():
    """The abs-scoring asterisk must read before the click cue (Label*▸),
    not after (Label▸*) -- the star is scoring metadata, the icon is a UI
    affordance, and label*-then-icon is the natural reading order."""
    d = METRIC_BY_KEY["resisted_divergence_left"]
    assert d.desc and d.scorer == "abs"
    m = MetricResult(definition=d, model_vals=[70.0, 80.0, 90.0], driver_vals=[])
    m.model_agg, m.score = 80.0, 51.0
    html = _metric_rows([m])
    star_idx = html.index("*")
    icon_idx = html.index('<span class="rowinfo">')
    assert star_idx < icon_idx
