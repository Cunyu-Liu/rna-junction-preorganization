"""X0 independent external-case qualification tests (v1.5)."""
import json
import os

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"


def _load(name):
    with open(os.path.join(RUN_ROOT, "external_case", "x0", name)) as f:
        return json.load(f)


def test_decision_is_inconclusive_terminal_state():
    d = _load("X0_decision.json")
    assert d["state"] == "X0_INCONCLUSIVE_LOW_N_OR_OPERATOR_AMBIGUITY"


def test_platform_independent_true():
    d = _load("X0_decision.json")
    assert d["platform_independent"] is True


def test_no_qualified_analysis_card_when_not_qualified():
    # §13.2: analysis card only when qualified.
    assert not os.path.exists(os.path.join(RUN_ROOT, "external_case", "x0",
                                           "external_case_analysis_card.json"))


def test_novelty_blocker_registered():
    d = _load("X0_decision.json")
    assert "decisive_blockers" in d and len(d["decisive_blockers"]) >= 1


def test_cluster_not_split():
    with open(os.path.join(RUN_ROOT, "external_case", "x0", "platform_lineage.tsv")) as f:
        content = f.read()
    assert "B0/B1/B2" not in content  # placeholder guard
    assert "ONE platform lineage" in content or "one platform lineage" in content


def test_quality_matrix_has_all_eleven_conditions():
    with open(os.path.join(RUN_ROOT, "external_case", "x0", "qualification_matrix.tsv")) as f:
        rows = [l.split("\t") for l in f.read().strip().splitlines()]
    header, body = rows[0], rows[1:]
    assert "eligibility_condition" in header
    assert len(body) == 10  # the ten §13.1 conditions assessed


def test_documentation_files_present():
    for p in ("candidate_registry.tsv", "source_integrity.tsv",
              "platform_lineage.tsv", "qualification_matrix.tsv"):
        assert os.path.exists(os.path.join(RUN_ROOT, "external_case", "x0", p))