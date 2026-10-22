"""Unit tests for AI explanation, confidence probabilities, margins, and context windows."""
import pytest
from src.inference import TokenClassifierPredictor
from src.lexer_utils import CLASSES

@pytest.fixture
def predictor():
    return TokenClassifierPredictor(artifacts_dir="artifacts")

def test_inference_probabilities_and_alternatives(predictor):
    code = "int count = 42;"
    results = predictor.predict_code(code)
    assert len(results) == 5

    for r in results:
        # Probabilities exist for all 5 classes
        assert "probabilities" in r
        probs = r["probabilities"]
        assert len(probs) == len(CLASSES)
        for cls in CLASSES:
            assert cls in probs
            assert 0.0 <= probs[cls] <= 1.0

        # Sum of probabilities is close to 1.0
        prob_sum = sum(probs.values())
        assert 0.98 <= prob_sum <= 1.02

        # Alternative prediction
        assert "alternative_class" in r
        assert r["alternative_class"] in CLASSES
        assert "alternative_confidence" in r
        assert 0.0 <= r["alternative_confidence"] <= r["confidence"]
        assert "margin" in r
        assert r["margin"] >= 0.0

def test_inference_surrounding_context_windows(predictor):
    code = "int a = 1; float b = 2.0;"
    results = predictor.predict_code(code)
    assert len(results) >= 8

    # First token has empty left context
    assert results[0]["context_left"] == []
    assert len(results[0]["context_right"]) == 3
    assert results[0]["context_right"] == ["a", "=", "1"]

    # Middle token has both left and right context
    mid_idx = 4  # ';'
    assert len(results[mid_idx]["context_left"]) == 3
    assert len(results[mid_idx]["context_right"]) == 3
