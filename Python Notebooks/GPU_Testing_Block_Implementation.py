import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MultiHead Attention (with fused QKV) ---
class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads

        self.c_attn = nn.Linear(emb_dim, 3 * emb_dim)  # fused QKV
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)  # (B, T, 3*emb_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # each is (B, T, emb_dim)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_heads, T, T)
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, emb_dim)
        return self.out_proj(out)


# --- LayerNorm ---
class MyLayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# --- GELU ---
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


# --- Feedforward ---
class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim)
        )
        self.norm = MyLayerNorm(emb_dim)

    def forward(self, x):
        out = self.net(x)
        return self.norm(out), out


# --- Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        emb_dim = config["emb_dim"]
        self.ln1 = MyLayerNorm(emb_dim)
        self.attn = MultiHeadCausalSelfAttention(emb_dim, config["n_heads"])
        self.ln2 = MyLayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim)

    def forward(self, x):
        attn_out = self.attn(self.ln1(x))
        x = x + attn_out
        ff_out, _ = self.ff(self.ln2(x))
        return x + ff_out


# --- GPT Model ---
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.context_length = config["context_length"]
        self.emb_dim = config["emb_dim"]

        self.token_embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.positional_embedding = nn.Embedding(self.context_length, self.emb_dim)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )
        self.final_norm = MyLayerNorm(self.emb_dim)
        self.out_head = nn.Linear(self.emb_dim, self.vocab_size, bias=False)

    def create_input_embeddings(self, token_ids):
        B, T = token_ids.shape
        if T > self.context_length:
            raise ValueError(f"Sequence length {T} exceeds context limit {self.context_length}")
        tok_emb = self.token_embedding(token_ids)
        pos_ids = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, T)
        pos_emb = self.positional_embedding(pos_ids)
        return tok_emb + pos_emb

    def forward(self, token_ids):
        x = self.create_input_embeddings(token_ids)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


# --- Utilities ---
def print_gradients(model, token_ids):
    output = model(token_ids)
    target = torch.zeros_like(output)  # dummy
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item():.6f}")


def predict_next_tokens(logits_batch, tokenizer=None):
    last_logits = logits_batch[:, -1, :]
    probs = F.softmax(last_logits, dim=-1)
    predicted_token_ids = torch.argmax(probs, dim=-1)
    predicted_tokens = [tokenizer.decode([i.item()]) for i in predicted_token_ids] if tokenizer else None
    return predicted_token_ids, predicted_tokens, probs
