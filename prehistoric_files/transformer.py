import torch
import torch.nn as nn
import torch.nn.functional as F


class SinPositionEncoding(nn.Module):
    def __init__(self, max_sequence_length, d_model, base=10000):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.base = base

    def forward(self):
        pe = torch.zeros(
            self.max_sequence_length, self.d_model, dtype=torch.float
        )  # size(max_sequence_length, d_model)
        exp_1 = torch.arange(
            self.d_model // 2, dtype=torch.float
        )  # 初始化一半维度，sin位置编码的维度被分为了两部分
        exp_value = exp_1 / (self.d_model / 2)

        alpha = 1 / (self.base**exp_value)  # size(dmodel/2)
        out = (
            torch.arange(self.max_sequence_length, dtype=torch.float)[:, None]
            @ alpha[None, :]
        )  # size(max_sequence_length, d_model/2)
        embedding_sin = torch.sin(out)
        embedding_cos = torch.cos(out)

        pe[:, 0::2] = embedding_sin  # 奇数位置设置为sin
        pe[:, 1::2] = embedding_cos  # 偶数位置设置为cos
        return pe


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Multihead self-attention layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Feedforward network layers
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: Tensor of shape [seq_len, batch, d_model]
        """
        seq_len = x.size(0)
        # Create a causal mask for self-attention;
        i = torch.arange(seq_len, device=x.device).unsqueeze(1)
        j = torch.arange(seq_len, device=x.device).unsqueeze(0)
        # Standard causal mask: allow attending only to previous tokens and itself.
        mask = j > i  # Shape: [seq_len, seq_len]

        # Apply self-attention with the mask
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        # Apply residual connection and normalization after attention
        x = self.norm1(x + self.dropout(attn_out))
        # Feedforward network with residual connection and normalization
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.norm2(x + self.dropout(ff_output))


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_length):
        super().__init__()
        # Word embedding and position embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = SinPositionEncoding(
            max_sequence_length=max_seq_length * 2, d_model=d_model
        ).forward()
        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead) for _ in range(num_layers)]
        )
        # Final fully connected layer to project to vocab size
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x, start_pos):
        """
        x: Tensor of token indices with shape [batch, seq_len]
        start_pos: Tensor of starting positions for positional encoding [batch, pos]
        Returns:
            Tensor of shape [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.size()
        self.pos_embedding = self.pos_embedding.to(x.device)
        _pos_embed = torch.stack(
            [
                self.pos_embedding[start_pos[i] : start_pos[i] + seq_len]
                for i in range(batch_size)
            ],
            dim=0,
        )
        # Scale embedding and add positional encoding
        x = self.embedding(x) * (self.d_model**0.5)
        x = x + _pos_embed
        # Transformer expects input shape [seq_len, batch, d_model]
        x = x.permute(1, 0, 2)
        # Process through the decoder layers
        for layer in self.layers:
            x = layer(x)
        # Project to vocabulary space and return to [batch, seq_len, vocab_size] shape
        return self.fc(x.permute(1, 0, 2))
