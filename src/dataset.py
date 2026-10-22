"""Dataset, Vocabulary, and Data Loading for Token Classification."""
import os
import json
from typing import List, Dict, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader
from src.lexer_utils import tokenize_c_code, CLASSES, CLASS_TO_ID, ID_TO_CLASS, TokenItem

class CharVocab:
    """Character-level vocabulary for token embeddings."""
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self):
        self.char_to_id: Dict[str, int] = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        self.id_to_char: Dict[int, str] = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}
        
        # Add all printable ASCII characters
        for code in range(32, 127):
            ch = chr(code)
            if ch not in self.char_to_id:
                idx = len(self.char_to_id)
                self.char_to_id[ch] = idx
                self.id_to_char[idx] = ch

    def __len__(self) -> int:
        return len(self.char_to_id)

    def encode_char(self, ch: str) -> int:
        return self.char_to_id.get(ch, self.UNK_IDX)

    def encode_token(self, token_str: str, max_token_len: int = 24) -> Tuple[List[int], int]:
        """Encodes a token into fixed-length character IDs and actual length."""
        chars = list(token_str[:max_token_len])
        actual_len = max(1, len(chars))
        char_ids = [self.encode_char(c) for c in chars]
        # Pad to max_token_len
        if len(char_ids) < max_token_len:
            char_ids += [self.PAD_IDX] * (max_token_len - len(char_ids))
        return char_ids, actual_len

    def save(self, filepath: str):
        data = {
            "char_to_id": self.char_to_id,
            "classes": CLASSES,
            "class_to_id": CLASS_TO_ID
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CharVocab":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls()
        vocab.char_to_id = data["char_to_id"]
        vocab.id_to_char = {int(v): k for k, v in vocab.char_to_id.items()}
        return vocab


class TokenProgramDataset(Dataset):
    """
    Dataset representing sequences of C tokens chunked into fixed-length windows.
    Strictly splits by program/file to prevent data leakage.
    """
    def __init__(
        self,
        file_paths: List[str],
        vocab: CharVocab,
        seq_len: int = 48,
        max_token_len: int = 24,
        stride: int = 24
    ):
        self.vocab = vocab
        self.seq_len = seq_len
        self.max_token_len = max_token_len
        self.samples: List[Dict[str, Any]] = []

        self._process_files(file_paths, stride)

    def _process_files(self, file_paths: List[str], stride: int):
        for path in file_paths:
            if not os.path.isfile(path):
                continue
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()

            tokens = tokenize_c_code(code, require_labels=True)
            if not tokens:
                continue

            # Slide window over program tokens
            n_tokens = len(tokens)
            if n_tokens <= self.seq_len:
                self.samples.append(self._encode_window(tokens))
            else:
                for start in range(0, n_tokens, stride):
                    chunk = tokens[start:start + self.seq_len]
                    if len(chunk) < 6:  # Skip tiny tail fragments
                        continue
                    self.samples.append(self._encode_window(chunk))

    def _encode_window(self, tokens: List[TokenItem]) -> Dict[str, Any]:
        cur_len = len(tokens)
        char_ids_list = []
        char_lens_list = []
        label_ids = []

        for tok in tokens:
            c_ids, c_len = self.vocab.encode_token(tok.value, self.max_token_len)
            char_ids_list.append(c_ids)
            char_lens_list.append(c_len)
            label_ids.append(CLASS_TO_ID.get(tok.label, 0))

        # Pad window to self.seq_len
        pad_char_ids = [self.vocab.PAD_IDX] * self.max_token_len
        while len(char_ids_list) < self.seq_len:
            char_ids_list.append(pad_char_ids)
            char_lens_list.append(1)
            label_ids.append(-100)  # PyTorch ignore_index

        return {
            "char_ids": torch.tensor(char_ids_list, dtype=torch.long),
            "char_lens": torch.tensor(char_lens_list, dtype=torch.long),
            "seq_len": torch.tensor(cur_len, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def get_file_lists(data_dir: str) -> Tuple[List[str], List[str]]:
    """Returns separate lists of train and test file paths."""
    train_dir = os.path.join(data_dir, "raw_c_programs", "train")
    test_dir = os.path.join(data_dir, "raw_c_programs", "test")

    train_files = [
        os.path.join(train_dir, f) for f in os.listdir(train_dir)
        if f.endswith(".c")
    ]
    test_files = [
        os.path.join(test_dir, f) for f in os.listdir(test_dir)
        if f.endswith(".c")
    ]
    return sorted(train_files), sorted(test_files)
