"""PyTorch Neural Network for Token Classification: Character-level and Contextual BiLSTM."""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharContextBiLSTM(nn.Module):
    """
    Two-stage BiLSTM model for compiler token classification:
    1. Character-level BiLSTM: extracts morphological/lexical features from token characters.
    2. Contextual BiLSTM: extracts syntactic and contextual dependencies across token sequences.
    """
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 5,
        char_emb_dim: int = 32,
        char_hidden_dim: int = 32,
        token_emb_dim: int = 64,
        context_hidden_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.char_emb_dim = char_emb_dim
        self.char_hidden_dim = char_hidden_dim
        self.token_emb_dim = token_emb_dim
        self.context_hidden_dim = context_hidden_dim

        # 1. Character-level subnetwork
        self.char_embedding = nn.Embedding(vocab_size, char_emb_dim, padding_idx=0)
        self.char_lstm = nn.LSTM(
            input_size=char_emb_dim,
            hidden_size=char_hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.char_proj = nn.Sequential(
            nn.Linear(char_hidden_dim * 2, token_emb_dim),
            nn.LayerNorm(token_emb_dim),
            nn.ReLU()
        )

        # 2. Contextual subnetwork
        self.dropout = nn.Dropout(dropout)
        self.context_lstm = nn.LSTM(
            input_size=token_emb_dim,
            hidden_size=context_hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # 3. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(context_hidden_dim * 2, context_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_hidden_dim, num_classes)
        )

    def extract_char_features(self, char_ids: torch.Tensor, char_lens: torch.Tensor) -> torch.Tensor:
        """
        char_ids: (Batch_Tokens, Max_Token_Len)
        char_lens: (Batch_Tokens,)
        Returns: (Batch_Tokens, token_emb_dim)
        """
        # Embed chars
        char_embs = self.char_embedding(char_ids)  # (BT, L, char_emb_dim)
        lstm_out, _ = self.char_lstm(char_embs)    # (BT, L, char_hidden_dim * 2)

        # Mask padding characters for max pooling
        mask = (char_ids != 0).unsqueeze(-1)       # (BT, L, 1)
        # Set padding time steps to large negative value before max pool
        masked_lstm_out = lstm_out.masked_fill(~mask, -1e9)
        pooled, _ = torch.max(masked_lstm_out, dim=1)  # (BT, char_hidden_dim * 2)

        # In case all chars were padding, avoid -1e9
        pooled = torch.clamp(pooled, min=-10.0, max=10.0)
        token_vecs = self.char_proj(pooled)        # (BT, token_emb_dim)
        return token_vecs

    def forward(
        self,
        char_ids: torch.Tensor,
        char_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        char_ids: (Batch, Seq_Len, Max_Token_Len)
        char_lens: (Batch, Seq_Len) [optional]
        Returns: logits of shape (Batch, Seq_Len, num_classes)
        """
        B, S, L = char_ids.size()
        flat_char_ids = char_ids.view(B * S, L)

        if char_lens is not None:
            flat_char_lens = char_lens.view(B * S)
        else:
            flat_char_lens = torch.sum(flat_char_ids != 0, dim=-1).clamp(min=1)

        token_vecs = self.extract_char_features(flat_char_ids, flat_char_lens)  # (B*S, token_emb_dim)
        token_seq = token_vecs.view(B, S, self.token_emb_dim)                   # (B, S, token_emb_dim)

        token_seq = self.dropout(token_seq)
        context_out, _ = self.context_lstm(token_seq)                           # (B, S, context_hidden_dim * 2)
        logits = self.classifier(context_out)                                   # (B, S, num_classes)
        return logits

    def get_config(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "num_classes": self.num_classes,
            "char_emb_dim": self.char_emb_dim,
            "char_hidden_dim": self.char_hidden_dim,
            "token_emb_dim": self.token_emb_dim,
            "context_hidden_dim": self.context_hidden_dim,
        }
