import pytest
from src.evaluate import compute_metrics
from src.lexer_utils import CLASSES

def test_compute_metrics_perfect():
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4]
    metrics = compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["total_tokens"] == 5
    assert len(metrics["confusion_matrix"]) == len(CLASSES)

def test_compute_metrics_imperfect():
    y_true = [0, 1, 2, 3, 4, 1, 2]
    y_pred = [0, 1, 2, 3, 4, 2, 1]
    metrics = compute_metrics(y_true, y_pred)

    assert 0.0 < metrics["accuracy"] < 1.0
    assert 0.0 < metrics["macro_f1"] < 1.0
    assert metrics["total_tokens"] == 7
