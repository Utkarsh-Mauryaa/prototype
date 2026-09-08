"""
Bidirectional LSTM (BiLSTM) Classifier for NLP Sequence Classification
====================================================================
Standard PyTorch implementation with:
  - Token embedding layer
  - 2-layer Bidirectional LSTM core
  - Mask-aware Global Average + Max pooling
  - Dropout regularization and dense classification head
"""

import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.25,
        pad_idx: int = 0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Concat of bidirectional states (hidden_dim * 2) after Avg & Max pooling gives hidden_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Tensor of shape (batch_size, seq_len) with 1 for tokens and 0 for padding
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        x = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        x = self.embed_dropout(x)
        
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim * 2)

        if attention_mask is not None:
            # Mask out padding tokens to ensure accurate pooling
            mask = attention_mask.unsqueeze(-1).expand_as(lstm_out).float()
            
            # Masked average pooling
            sum_out = torch.sum(lstm_out * mask, dim=1)
            lengths = torch.clamp(mask.sum(dim=1), min=1e-9)
            avg_pool = sum_out / lengths
            
            # Masked max pooling (set padding positions to large negative value)
            masked_lstm_out = lstm_out.masked_fill(~attention_mask.unsqueeze(-1).bool(), -1e9)
            max_pool = torch.max(masked_lstm_out, dim=1)[0]
        else:
            avg_pool = torch.mean(lstm_out, dim=1)
            max_pool = torch.max(lstm_out, dim=1)[0]

        # Concatenate average and max pooled representations
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (batch_size, hidden_dim * 4)
        logits = self.classifier(pooled)
        return logits
