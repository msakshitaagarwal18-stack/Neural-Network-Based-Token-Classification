"""Inference engine for Neural Network-Based Token Classification."""
import os
import json
from typing import List, Dict, Any, Optional
import torch

from src.lexer_utils import tokenize_c_code, CLASSES, CLASS_TO_ID, ID_TO_CLASS
from src.dataset import CharVocab
from src.model import CharContextBiLSTM

class TokenClassifierPredictor:
    """Loads a trained BiLSTM token classification model and performs neural inference."""

    def __init__(self, artifacts_dir: str = "artifacts", model_path: Optional[str] = None, device: str = "cpu"):
        self.artifacts_dir = artifacts_dir
        self.device = torch.device(device)

        vocab_path = os.path.join(artifacts_dir, "vocab.json")
        if model_path is None:
            model_path = os.path.join(artifacts_dir, "model.pt")

        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at: {vocab_path}")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        # 1. Load Vocab
        self.vocab = CharVocab.load(vocab_path)

        # 2. Load Checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.seq_len = checkpoint.get("seq_len", 48)
        self.max_token_len = checkpoint.get("max_token_len", 24)
        config = checkpoint["config"]

        # 3. Instantiate and Load Model
        self.model = CharContextBiLSTM(
            vocab_size=config["vocab_size"],
            num_classes=config["num_classes"],
            char_emb_dim=config["char_emb_dim"],
            char_hidden_dim=config["char_hidden_dim"],
            token_emb_dim=config["token_emb_dim"],
            context_hidden_dim=config["context_hidden_dim"]
        ).to(self.device)

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict_code(self, source_code: str) -> List[Dict[str, Any]]:
        """
        Runs neural inference on arbitrary C source code string.
        Safely handles empty, malformed, unfamiliar, and large inputs.
        Provides full class probabilities, runner-up predictions, and context windows.
        """
        if not source_code or not source_code.strip():
            return []

        # Tokenize (lexer handles malformed inputs gracefully without throwing syntax errors)
        try:
            tokens = tokenize_c_code(source_code, require_labels=False)
        except Exception:
            # Fallback if lexer faces strange byte encoding
            tokens = []

        if not tokens:
            return []

        results: List[Dict[str, Any]] = []

        # Chunk into windows to safely handle arbitrarily large programs
        stride = self.seq_len
        n_tokens = len(tokens)

        for start in range(0, n_tokens, stride):
            chunk = tokens[start:start + self.seq_len]
            chunk_len = len(chunk)

            # Build char_ids and char_lens tensors
            char_ids_list = []
            char_lens_list = []

            for tok in chunk:
                c_ids, c_len = self.vocab.encode_token(tok.value, self.max_token_len)
                char_ids_list.append(c_ids)
                char_lens_list.append(c_len)

            # Pad chunk to self.seq_len if smaller
            pad_char_ids = [self.vocab.PAD_IDX] * self.max_token_len
            while len(char_ids_list) < self.seq_len:
                char_ids_list.append(pad_char_ids)
                char_lens_list.append(1)

            char_ids_tensor = torch.tensor([char_ids_list], dtype=torch.long, device=self.device)
            char_lens_tensor = torch.tensor([char_lens_list], dtype=torch.long, device=self.device)

            with torch.no_grad():
                # Forward pass through neural network
                logits = self.model(char_ids_tensor, char_lens_tensor)  # (1, seq_len, num_classes)
                probs = torch.softmax(logits, dim=-1)[0]                 # (seq_len, num_classes)

            # Extract actual predictions for non-padded tokens in chunk
            for idx in range(chunk_len):
                tok = chunk[idx]
                probs_row = probs[idx]

                sorted_indices = torch.argsort(probs_row, descending=True)
                top_idx = sorted_indices[0].item()
                second_idx = sorted_indices[1].item() if len(sorted_indices) > 1 else top_idx

                pred_cls_name = ID_TO_CLASS.get(top_idx, "IDENTIFIER")
                confidence = float(probs_row[top_idx].item())
                alt_cls_name = ID_TO_CLASS.get(second_idx, "IDENTIFIER")
                alt_conf = float(probs_row[second_idx].item())
                margin = confidence - alt_conf

                prob_dict = {
                    ID_TO_CLASS[c_i]: round(float(probs_row[c_i].item()), 4)
                    for c_i in range(len(ID_TO_CLASS))
                }

                results.append({
                    "token": tok.value,
                    "line": tok.line,
                    "col": tok.col,
                    "predicted_class": pred_cls_name,
                    "confidence": confidence,
                    "ground_truth": tok.label,
                    "probabilities": prob_dict,
                    "alternative_class": alt_cls_name,
                    "alternative_confidence": alt_conf,
                    "margin": margin
                })

        # Attach surrounding context windows to each token
        for i in range(len(results)):
            results[i]["context_left"] = [results[j]["token"] for j in range(max(0, i - 3), i)]
            results[i]["context_right"] = [results[j]["token"] for j in range(i + 1, min(len(results), i + 4))]

        return results

    def predict_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Reads a .c file and predicts tokens."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()
        return self.predict_code(code)
