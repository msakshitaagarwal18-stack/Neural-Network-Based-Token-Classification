"""Training pipeline for Neural Network-Based Token Classification."""
import os
import sys
import json
import argparse
from typing import Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.lexer_utils import CLASSES
from src.dataset import CharVocab, TokenProgramDataset, get_file_lists
from src.model import CharContextBiLSTM
from src.evaluate import compute_metrics, plot_confusion_matrix, plot_training_history

def train_model(
    data_dir: str = "data",
    artifacts_dir: str = "artifacts",
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 0.003,
    seq_len: int = 48,
    max_token_len: int = 24,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Trains the BiLSTM token classification model and saves artifacts."""
    os.makedirs(artifacts_dir, exist_ok=True)
    device_obj = torch.device(device)

    # 1. Load file paths (strict file-level split to prevent data leakage)
    train_files, test_files = get_file_lists(data_dir)
    print(f"Found {len(train_files)} training programs and {len(test_files)} testing programs.")

    # 2. Build and save vocabulary
    vocab = CharVocab()
    vocab_path = os.path.join(artifacts_dir, "vocab.json")
    vocab.save(vocab_path)
    print(f"Vocabulary saved to {vocab_path} (vocab size: {len(vocab)})")

    # 3. Create datasets and dataloaders
    train_dataset = TokenProgramDataset(
        file_paths=train_files,
        vocab=vocab,
        seq_len=seq_len,
        max_token_len=max_token_len,
        stride=24
    )
    test_dataset = TokenProgramDataset(
        file_paths=test_files,
        vocab=vocab,
        seq_len=seq_len,
        max_token_len=max_token_len,
        stride=seq_len  # Non-overlapping windows for test
    )

    print(f"Generated {len(train_dataset)} train windows and {len(test_dataset)} test windows.")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. Instantiate Model
    model = CharContextBiLSTM(
        vocab_size=len(vocab),
        num_classes=len(CLASSES),
        char_emb_dim=32,
        char_hidden_dim=32,
        token_emb_dim=64,
        context_hidden_dim=64,
        dropout=0.2
    ).to(device_obj)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # 5. Training Loop
    print("\n--- Starting Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            char_ids = batch["char_ids"].to(device_obj)
            char_lens = batch["char_lens"].to(device_obj)
            labels = batch["labels"].to(device_obj)

            optimizer.zero_grad()
            logits = model(char_ids, char_lens)

            # Flatten batch and seq dimensions for CrossEntropyLoss
            B, S, C = logits.shape
            loss = criterion(logits.view(B * S, C), labels.view(B * S))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * B

            # Accuracy on non-padding tokens
            valid_mask = labels != -100
            preds = torch.argmax(logits, dim=-1)
            correct += ((preds == labels) & valid_mask).sum().item()
            total += valid_mask.sum().item()

        scheduler.step()

        epoch_loss = total_loss / max(1, len(train_dataset))
        epoch_acc = (correct / total) if total > 0 else 0.0

        # Evaluate on test set
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in test_loader:
                char_ids = batch["char_ids"].to(device_obj)
                char_lens = batch["char_lens"].to(device_obj)
                labels = batch["labels"].to(device_obj)

                logits = model(char_ids, char_lens)
                B, S, C = logits.shape
                loss = criterion(logits.view(B * S, C), labels.view(B * S))
                val_loss += loss.item() * B

                valid_mask = labels != -100
                preds = torch.argmax(logits, dim=-1)
                val_correct += ((preds == labels) & valid_mask).sum().item()
                val_total += valid_mask.sum().item()

        epoch_val_loss = val_loss / max(1, len(test_dataset))
        epoch_val_acc = (val_correct / val_total) if val_total > 0 else 0.0

        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}% | "
                f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%"
            )

    # 6. Final Test Set Evaluation
    print("\n--- Evaluating Final Model on Held-Out Test Set ---")
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for batch in test_loader:
            char_ids = batch["char_ids"].to(device_obj)
            char_lens = batch["char_lens"].to(device_obj)
            labels = batch["labels"].to(device_obj)

            logits = model(char_ids, char_lens)
            preds = torch.argmax(logits, dim=-1)

            valid_mask = labels != -100
            flat_preds = preds[valid_mask].cpu().numpy().tolist()
            flat_targets = labels[valid_mask].cpu().numpy().tolist()

            all_preds.extend(flat_preds)
            all_targets.extend(flat_targets)

    metrics = compute_metrics(all_targets, all_preds)
    print(f"Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Macro Precision: {metrics['macro_precision']*100:.2f}%")
    print(f"Macro Recall:    {metrics['macro_recall']*100:.2f}%")
    print(f"Macro F1-Score:  {metrics['macro_f1']*100:.2f}%")

    # 7. Save Model Weights & Config
    model_save_path = os.path.join(artifacts_dir, "model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "config": model.get_config(),
        "seq_len": seq_len,
        "max_token_len": max_token_len,
        "classes": CLASSES
    }, model_save_path)
    print(f"Model checkpoint saved to {model_save_path}")

    # 8. Save Metrics JSON
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # 9. Save History & Plots
    # Save the numeric epoch history for interactive evaluation charts in the Streamlit UI.
    history_json_path = os.path.join(artifacts_dir, "training_history.json")
    with open(history_json_path, "w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)

    history_plot_path = os.path.join(artifacts_dir, "training_history.png")
    plot_training_history(history, history_plot_path)
    loss_curve_path = os.path.join(artifacts_dir, "loss_curve.png")
    plot_training_history(history, loss_curve_path)
    print(f"Training curves saved to {history_plot_path}")

    cm_plot_path = os.path.join(artifacts_dir, "confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], CLASSES, cm_plot_path)
    print(f"Confusion matrix plot saved to {cm_plot_path}")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Neural Network Token Classifier")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
