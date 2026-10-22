"""Evaluation module for Token Classification."""
import os
import json
from typing import Dict, Any, List, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from src.lexer_utils import CLASSES, CLASS_TO_ID, ID_TO_CLASS

def compute_metrics(
    y_true: List[int],
    y_pred: List[int]
) -> Dict[str, Any]:
    """Computes genuine multi-class evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))).tolist()
    
    per_class_report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
        "total_tokens": len(y_true),
        "confusion_matrix": cm,
        "per_class": per_class_report
    }

def plot_confusion_matrix(
    cm: List[List[int]],
    classes: List[str],
    save_path: str,
    title: str = "Confusion Matrix (Test Set)"
):
    """Plots and saves the confusion matrix as a high-resolution PNG image."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm_arr, cmap=plt.cm.Blues, alpha=0.85)

    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            val = cm_arr[i, j]
            color = "white" if val > cm_arr.max() / 2 else "black"
            ax.text(x=j, y=i, s=str(val), va="center", ha="center", size="medium", color=color, weight="bold")

    fig.colorbar(cax)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="left")
    ax.set_yticklabels(classes)
    plt.title(title, pad=20, fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=10, fontweight="bold")
    plt.ylabel("Ground Truth Class", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str
):
    """Plots training and validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss", markersize=3)
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=3)
    if "val_acc" in history and history["val_acc"]:
        ax2.plot(epochs, history["val_acc"], "g--^", label="Val Acc", markersize=3)
    ax2.set_title("Token Accuracy vs. Epochs", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def compare_bilstm_vs_lexer(source_code: str, predictor) -> Dict[str, Any]:
    """
    Compares BiLSTM neural classifier vs. Pygments deterministic lexer on arbitrary C code.
    Measures genuine execution time and computes accuracy, precision, recall, and macro F1.
    """
    import time
    from src.lexer_utils import tokenize_c_code, CLASS_TO_ID

    if not source_code or not source_code.strip():
        return {}

    # 1. Measure Lexer execution
    t0_lexer = time.perf_counter()
    lexer_tokens = tokenize_c_code(source_code, require_labels=True)
    lexer_time_ms = (time.perf_counter() - t0_lexer) * 1000.0

    # 2. Measure BiLSTM execution
    t0_bilstm = time.perf_counter()
    bilstm_preds = predictor.predict_code(source_code)
    bilstm_time_ms = (time.perf_counter() - t0_bilstm) * 1000.0

    # 3. Match ground truth labels from lexer
    y_true = []
    y_pred = []
    for p in bilstm_preds:
        gt = p.get("ground_truth")
        if gt in CLASS_TO_ID:
            y_true.append(CLASS_TO_ID[gt])
            y_pred.append(CLASS_TO_ID[p["predicted_class"]])

    metrics = {}
    if y_true:
        metrics = compute_metrics(y_true, y_pred)

    total_tokens = len(bilstm_preds)
    throughput_bilstm = (total_tokens / (bilstm_time_ms / 1000.0)) if bilstm_time_ms > 0 else 0.0
    throughput_lexer = (total_tokens / (lexer_time_ms / 1000.0)) if lexer_time_ms > 0 else 0.0

    return {
        "total_tokens": total_tokens,
        "labeled_tokens": len(y_true),
        "bilstm_time_ms": round(bilstm_time_ms, 2),
        "lexer_time_ms": round(lexer_time_ms, 2),
        "bilstm_throughput": round(throughput_bilstm, 1),
        "lexer_throughput": round(throughput_lexer, 1),
        "accuracy": metrics.get("accuracy", 0.0),
        "macro_precision": metrics.get("macro_precision", 0.0),
        "macro_recall": metrics.get("macro_recall", 0.0),
        "macro_f1": metrics.get("macro_f1", 0.0),
        "per_class": metrics.get("per_class", {}),
        "confusion_matrix": metrics.get("confusion_matrix", [])
    }
