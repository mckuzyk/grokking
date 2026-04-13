import torch
from torch import nn
from torch.nn import functional as F
import math
from grokking.config import TransformerConfig


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.attention_heads = nn.ModuleList(
            [
                MultiHeadAttention(d_model=config.d_model, n_heads=config.n_heads)
                for _ in range(config.n_blocks)
            ]
        )

        self.mlps = nn.ModuleList(
            [
                MLP(d_in=config.d_model, d_out=config.d_model, d_hidden=config.d_mlp)
                for _ in range(config.n_blocks)
            ]
        )

        self.w_embed = nn.Embedding(config.P + 1, config.d_model)
        self.w_unembed = nn.Linear(config.d_model, config.P, bias=False)

        # Positional embedding
        self.w_pos_emb = nn.Embedding(3, config.d_model)

    def forward(self, t):
        positions = torch.arange(3, device=t.device)
        x = self.w_embed(t) + self.w_pos_emb(positions)
        for i in range(self.config.n_blocks):
            x = x + self.attention_heads[i](x)
            x = x + self.mlps[i](x)
        return self.w_unembed(x[:, 2, :])


class MLP(nn.Module):
    def __init__(self, d_in, d_out, d_hidden):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out)
        )

    def forward(self, x):
        return self.mlp(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.size()  # Batch size, Sequence length, Embedding dim (d_model)

        # Split into n_heads with view and move the head dimension left one so all batch dimensions come first
        # Resulting tensors have shape (B, n_heads, T, d_head == d_model // n_heads)
        q = self.q_proj(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # (B, n_heads, T, d_head) @ (B, n_heads, d_head, T) -> (B, n_heads, T, T)
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))

        # Softmax is applied to norm weights of a given query dotted w/ all keys, so dim=-1
        att = F.softmax(att, dim=-1)

        # (B, n_heads, T, T) @ (B, n_heads, T, d_head) -> (B, n_heads, T, d_head)
        att = att @ v

        # Concat
        att = att.transpose(1, 2).reshape(B, T, C)
        att = self.o_proj(att)

        return att


if __name__ == "__main__":
    from grokking.config import TransformerConfig
    from grokking.data import ModPDataset, train_test_split

    dset = ModPDataset(13)
    train_inputs, train_labels, test_inputs, test_labels = train_test_split(dset)

    cfg = TransformerConfig(P=13)
    tran = Transformer(cfg)
    out = tran.forward(train_inputs)
    print(out[0])
