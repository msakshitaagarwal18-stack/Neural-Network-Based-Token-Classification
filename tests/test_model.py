import pytest
import torch
from src.dataset import CharVocab
from src.model import CharContextBiLSTM

def test_char_vocab():
    vocab = CharVocab()
    assert len(vocab) > 90
    assert vocab.encode_char("<PAD>") == 0
    assert vocab.encode_char("<UNK>") == 1
    
    encoded, length = vocab.encode_token("int", max_token_len=10)
    assert len(encoded) == 10
    assert length == 3
    assert encoded[3:] == [0] * 7

def test_model_forward():
    vocab = CharVocab()
    model = CharContextBiLSTM(
        vocab_size=len(vocab),
        num_classes=5,
        char_emb_dim=16,
        char_hidden_dim=16,
        token_emb_dim=32,
        context_hidden_dim=32
    )
    model.eval()

    batch_size = 2
    seq_len = 8
    max_token_len = 12

    dummy_input = torch.randint(0, len(vocab), (batch_size, seq_len, max_token_len))
    dummy_lens = torch.randint(1, max_token_len, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(dummy_input, dummy_lens)

    assert logits.shape == (batch_size, seq_len, 5)
    assert not torch.isnan(logits).any()

def test_model_single_token():
    vocab = CharVocab()
    model = CharContextBiLSTM(
        vocab_size=len(vocab),
        num_classes=5,
        char_emb_dim=16,
        char_hidden_dim=16,
        token_emb_dim=32,
        context_hidden_dim=32
    )
    model.eval()

    dummy_input = torch.randint(0, len(vocab), (1, 1, 10))
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (1, 1, 5)
