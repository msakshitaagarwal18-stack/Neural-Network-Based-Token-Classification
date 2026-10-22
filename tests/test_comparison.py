"""Unit tests for Model Comparison Lab."""
import pytest
from src.inference import TokenClassifierPredictor
from src.evaluate import compare_bilstm_vs_lexer

@pytest.fixture
def predictor():
    return TokenClassifierPredictor(artifacts_dir="artifacts")

def test_compare_bilstm_vs_lexer_metrics(predictor):
    code = "int a = 10; float b = 20.0; return a;"
    res = compare_bilstm_vs_lexer(code, predictor)
    assert res["total_tokens"] > 5
    assert res["bilstm_time_ms"] > 0
    assert res["lexer_time_ms"] > 0
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["macro_f1"] <= 1.0
    assert "per_class" in res

def test_compare_bilstm_vs_lexer_empty(predictor):
    res = compare_bilstm_vs_lexer("", predictor)
    assert res == {}
