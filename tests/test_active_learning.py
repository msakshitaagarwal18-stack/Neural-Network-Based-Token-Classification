"""Unit tests for active learning feedback and safe retraining."""
import os
import pytest
from src.active_learning import (
    save_feedback,
    load_feedback,
    get_feedback_count,
    clear_feedback,
    train_active_learning_model
)

def test_active_learning_feedback_lifecycle():
    clear_feedback()
    assert get_feedback_count() == 0

    entry = save_feedback(
        token="val",
        line=2,
        col=8,
        model_pred="KEYWORD",
        confidence=0.65,
        corrected_class="IDENTIFIER",
        surrounding_tokens=["int", "val", "="]
    )
    assert entry["token"] == "val"
    assert entry["corrected_class"] == "IDENTIFIER"
    assert get_feedback_count() == 1

    all_fb = load_feedback()
    assert len(all_fb) == 1
    assert all_fb[0]["token"] == "val"

def test_active_learning_invalid_class():
    with pytest.raises(ValueError):
        save_feedback("x", 1, 1, "IDENTIFIER", 0.9, "INVALID_CLASS")

def test_active_learning_safe_retraining_preserves_baseline():
    baseline_path = os.path.join("artifacts", "model.pt")
    backup_path = os.path.join("artifacts", "model_baseline_backup.pt")
    target_path = os.path.join("artifacts", "active_learning_model.pt")

    assert os.path.isfile(baseline_path)
    baseline_mtime_before = os.path.getmtime(baseline_path)

    # Run quick 1-epoch fine-tuning
    res = train_active_learning_model(epochs=1, lr=0.001)
    assert res["status"] == "success"
    assert os.path.isfile(target_path)
    assert os.path.isfile(backup_path)

    # Baseline file must NOT have been modified
    baseline_mtime_after = os.path.getmtime(baseline_path)
    assert baseline_mtime_before == baseline_mtime_after
