"""Unit tests for the matched no-sequence latent-operator model (R2 core)."""
import numpy as np

from audit.models.no_sequence_latent_operator import (
    make_no_sequence_adapter, NO_SEQUENCE_LATENT_OPERATOR,
)


def _rows():
    # 3 junctions, 2 scaffolds, small censored+measured mix
    rows = []
    for j, seq in enumerate(["AAAA", "CCCC", "GGGG"]):
        for s in (1, 2):
            rows.append({"source_row_id": f"{j}_{s}", "jid": f"j{j}",
                         "junction_seq": seq, "scaf": s, "helix_seq": f"h{j}_{s}",
                         "y": -5.0 - j, "cens": (j == 1)})
    return rows


def test_model_registered():
    assert "no_sequence_latent_operator" in NO_SEQUENCE_LATENT_OPERATOR
    fit, predict = NO_SEQUENCE_LATENT_OPERATOR["no_sequence_latent_operator"]
    assert callable(fit) and callable(predict)


def test_fit_returns_gate_fields_and_predicts():
    fit, predict = make_no_sequence_adapter()
    rows = _rows()
    model = fit(rows)
    assert model["kind"] == "no_sequence_latent_operator"
    assert "success" in model and "nit" in model and "final_grad_norm" in model
    assert model["theta"].shape == (1,)
    # predict on rows with both scaffolds present in train
    mu, sigma, cp, support, abstain = predict(model, rows)
    assert len(mu) == len(rows)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))


def test_predictions_constant_per_scaffold_no_sequence_input():
    """The matched no-sequence model must NOT use junction identity/sequence:
    every junction inside a given scaffold gets the same latent location."""
    fit, predict = make_no_sequence_adapter()
    rows = _rows()
    model = fit(rows)
    mu, sigma, cp, support, abstain = predict(model, rows)
    mu_s1 = [mu[i] for i, r in enumerate(rows) if int(r["scaf"]) == 1]
    mu_s2 = [mu[i] for i, r in enumerate(rows) if int(r["scaf"]) == 2]
    assert abs(max(mu_s1) - min(mu_s1)) <= 1e-12, "must be constant per scaffold"
    assert abs(max(mu_s2) - min(mu_s2)) <= 1e-12, "must be constant per scaffold"


def test_unseen_scaffold_abstains_no_placeholder():
    fit, predict = make_no_sequence_adapter()
    train = _rows()
    test = [{"source_row_id": "x", "jid": "j9", "junction_seq": "UUUU",
             "scaf": 99, "helix_seq": "hx", "y": -6.0, "cens": 0}]
    model = fit(train)
    mu, sigma, cp, support, abstain = predict(model, test)
    assert bool(abstain[0]) is True and bool(support[0]) is False
