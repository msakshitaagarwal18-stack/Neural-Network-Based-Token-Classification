import os
import pytest
from src.inference import TokenClassifierPredictor
from src.lexer_utils import CLASSES

@pytest.fixture
def predictor():
    return TokenClassifierPredictor(artifacts_dir="artifacts")

def test_inference_basic(predictor):
    code = "int count = 42;"
    results = predictor.predict_code(code)
    assert len(results) == 5
    
    tokens = [r["token"] for r in results]
    assert tokens == ["int", "count", "=", "42", ";"]

    for r in results:
        assert r["predicted_class"] in CLASSES
        assert 0.0 <= r["confidence"] <= 1.0
        assert r["line"] >= 1
        assert r["col"] >= 1

def test_inference_empty_and_whitespace(predictor):
    assert predictor.predict_code("") == []
    assert predictor.predict_code("   \n\t  ") == []

def test_inference_malformed_code(predictor):
    # Missing closing brace, random operators
    code = "int x = ( * ; float"
    results = predictor.predict_code(code)
    assert len(results) > 0
    for r in results:
        assert r["predicted_class"] in CLASSES

def test_inference_unfamiliar_characters(predictor):
    code = "int @#$ mystery_var = 1;"
    results = predictor.predict_code(code)
    assert len(results) > 0
    for r in results:
        assert r["predicted_class"] in CLASSES

def test_inference_large_input(predictor):
    # Generates a code snippet with hundreds of tokens
    repeated_code = "int a = 1; int b = 2; a = a + b;\n" * 50
    results = predictor.predict_code(repeated_code)
    assert len(results) > 400
    for r in results[:10]:
        assert r["predicted_class"] in CLASSES
