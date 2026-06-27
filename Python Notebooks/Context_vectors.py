import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


def basic_concept_attention():
    x1 = np.array([0.4, 0.1, 0.8])
    x2 = np.array([0.5, 0.8, 0.6])
    x3 = np.array([0.5, 0.8, 0.6])

    X = np.stack([x1, x2, x3])
    Q = x1

    # Compute dot products (Q · K)
    scores = np.dot(X, Q)
    print("Dot-product scores:", scores)

    # Apply softmax to scores
    def softmax(x):
        shifted = x - np.max(x)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores)

    attention_weights = softmax(scores)
    print("Attention weights (α):", attention_weights)

    # Compute context vector z²
    # Multiply each token vector by its corresponding weight and sum

    z2 = np.sum(X.T * attention_weights, axis=1)
    print("Context vector z²:", z2)


def basic_self_attention(X):
    B, T, D = X.shape

    scores = torch.matmul(X, X.transpose(1, 2))
    # Applies on each sequence computing dot scores
    # for b in range(Batch):
    #     scores[b] = X[b] @ X[b].T

    mask = torch.tril(torch.ones(T, T, device=X.device)).bool()
    # torch.tril keeps only the lower triangle(including the diagonal values) and sets them to 1 and rest all 0 indicating that only values of 1 are to be preserved

    scores = scores.masked_fill(~mask, float('-inf'))
    # upper triangles values become infinity and neglected at later stages


    # Softmax across token dimension (each token gets weighted sum of others)
    attn_weights = F.softmax(scores, dim=-1)
    # Max values are chosen from each row(subtracting max values , taking exponential and dividing by sum of expoonestial


    # Multiply attention weights with the values but here in simplified self attention they are token vectors only
    context = torch.matmul(attn_weights, X)

    return context, attn_weights,scores


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, embed_dim_1,embed_dim_2):
        super().__init__()
        self.embed_dim_1 = embed_dim_1
        self.embed_dim_2 = embed_dim_2
        self.dropout = torch.nn.Dropout(0.0)

        # Learnable linear projections for Q, K, V by creating initial layer of weight matrices
        self.W_q = torch.nn.Linear(embed_dim_1, embed_dim_2, bias=False)
        self.W_k = torch.nn.Linear(embed_dim_1, embed_dim_2, bias=False)
        self.W_v = torch.nn.Linear(embed_dim_1, embed_dim_2, bias=False)

    def forward(self, X):
        B, T, D = X.shape

        # Linear projections to Q, K, V and part where layer matrix is multiplied by input ( X * W.T)
        Q = self.W_q(X)  # (B, T, D)
        K = self.W_k(X)  # (B, T, D)
        V = self.W_v(X)  # (B, T, D)

        # Scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(1, 2)) / (D ** 0.5)  # To prevent large values from exploding

        # Causal mask (only look backward in time)
        mask = torch.tril(torch.ones(T, T, device=X.device)).bool()
        scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        dropped_attn_weights=self.dropout(attn_weights)

        # Weighted sum of value vectors
        context = torch.matmul(dropped_attn_weights, V)

        return context, attn_weights, scores,dropped_attn_weights



class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # D_H
        self.dropout = torch.nn.Dropout(0.5)

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Final linear layer to combine all heads' outputs
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, X):
        B, T, D = X.shape


        Q = self.W_q(X)  # (B, T, D)
        K = self.W_k(X)
        V = self.W_v(X)

        # Split into heads
        # (B, T, D) -> (B, H, T, D_H)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T, T)

        # Apply causal mask (lower triangle)
        mask = torch.tril(torch.ones(T, T, device=X.device)).bool()
        scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Adding dropouts
        attn_weights = self.dropout(attn_weights)

        # Multiply attention weights with V to get context vectors
        context = torch.matmul(attn_weights, V)  # (B, H, T, D_H)

        # Concatenate heads and apply final projection
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        output = self.out_proj(context)  # (B, T, D)

        return output



