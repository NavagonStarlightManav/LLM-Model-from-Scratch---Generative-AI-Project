import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import tiktoken
import urllib.request
import nltk
nltk.download('punkt_tab')
import nltk.corpus
import re
from nltk import word_tokenize

import sys
sys.path.append("/content/drive/MyDrive/Manav LLM/")
from Class_Tokenizer_from_raw_text import Tokenizer_from_rawtext
from Class_Tokenizer_from_tokenised_vocab import Tokenizer_from_vocab
from Pytorch_Process import *
from Transformer_Block_Implementation import print_gradients, predict_next_tokens, GPTModel



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- GPT Config ---
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

drive_file_path = "/content/drive/MyDrive/Manav LLM/File_path.txt"

with open(drive_file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

#
tokenizer = tiktoken.get_encoding("gpt2")
Tokens_text_byte_encoding = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})


dataloader = create_dataloader_v1(
    raw_text,
    batch_size=5,
    max_length=4,
    stride=4,
    shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print(inputs)
print(targets)

vocab_size = 50257
output_dim = 8
max_Size = 4
torch.manual_seed(123)

def create_input_embeddings(inputs, vocab_size=vocab_size, output_dim=output_dim, max_position_embeddings=max_Size):
    inputs = inputs.to(device)
    batch_size, seq_length = inputs.shape
    token_embedding_layer = nn.Embedding(vocab_size, output_dim).to(device)
    token_embeddings = token_embedding_layer(inputs)

    positional_embedding_layer = nn.Embedding(max_position_embeddings, output_dim).to(device)
    position_ids = torch.arange(seq_length, device=inputs.device).unsqueeze(0).expand(batch_size, seq_length)
    positional_embeddings = positional_embedding_layer(position_ids)

    input_embeddings = token_embeddings + positional_embeddings

    emb_dim = GPT_CONFIG_124M['emb_dim']
    project_to_10 = nn.Linear(output_dim, emb_dim).to(device)
    dimensional_changed_input_embedding = project_to_10(input_embeddings)
    return dimensional_changed_input_embedding

input_embedding = create_input_embeddings(inputs)
print(input_embedding)

g1 = GPTModel(GPT_CONFIG_124M).to(device)


def calc_loss_batch(inputs_batch, target_batch, model=g1):
    inputs_batch = inputs_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(inputs_batch)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_batch.view(-1))
    return loss

def calc_loss_loader(data_loader, num_batches, model=g1):
    total_loss = 0.
    num_batches = len(data_loader) if num_batches is None else num_batches
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model)
        total_loss += loss.item()
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, num_batches=eval_iter, model=model)
        val_loss = calc_loss_loader(val_loader, num_batches=eval_iter, model=model)
    model.train()
    return train_loss, val_loss

def token_ids_to_text(token_ids, tokenizer):
    if token_ids.dim() == 2:
        token_ids = token_ids.squeeze(0)
    return tokenizer.decode(token_ids.tolist())

def text_to_token_ids(text, tokenizer):
    token_ids = tokenizer.encode(text)
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    idx = idx.to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx.cpu()

def generate_and_print_sample_from_dataloader(tokenizer, dataloader, model=g1, top_k=20, temperature=1.0):
    model.eval()
    inputs, targets = next(iter(dataloader))
    inputs = inputs.to(device)

    with torch.no_grad():
        generated_token_ids = generate(
            model=model,
            idx=inputs,
            max_new_tokens=50,
            context_size=GPT_CONFIG_124M["context_length"],
            temperature=temperature,
            top_k=top_k,
            eos_id=None
        )
        decoded = token_ids_to_text(generated_token_ids[0], tokenizer)
        print("Generated Text:\n", decoded.replace("\n", " "))
    model.train()



def train_model_simple(train_loader, val_loader, optimizer, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, model, top_k, temperature):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1
            print(f"Step {global_step}, Loss: {loss.item():.4f}")

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

        print(f"\nEp {epoch + 1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}\n")

        model.eval()
        start_ids = torch.tensor([tokenizer.encode(start_context)], dtype=torch.long).to(device)

        if start_ids.shape[1] > GPT_CONFIG_124M["context_length"]:
            start_ids = start_ids[:, -GPT_CONFIG_124M["context_length"]:]

        with torch.no_grad():
            generated = generate(
                model=model,
                idx=start_ids,
                max_new_tokens=50,
                context_size=GPT_CONFIG_124M["context_length"],
                temperature=temperature,
                top_k=top_k,
                eos_id=None  # or specify your <eos> token ID if available
            )
            print("Sample Generated Text:\n")
            print(tokenizer.decode(generated[0].tolist()).replace("\n", " "))
        model.train()

    return train_losses, val_losses, track_tokens_seen

train_ratio = 0.90
split_idx = int(train_ratio * len(raw_text))

train_loader = create_dataloader_v1(
    raw_text[:split_idx],
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    raw_text[split_idx:],
    batch_size=1,
    max_length=GPT_CONFIG_124M["context_length"]//2,
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

optimizer = torch.optim.AdamW(g1.parameters(), lr=5e-4)
num_epochs = 10
eval_iter = 1
eval_freq = 5



train_losses, val_losses, track_tokens_seen = train_model_simple(
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    num_epochs=num_epochs,
    eval_freq=eval_freq,
    eval_iter=eval_iter,
    start_context="Once there used to be a king and queen across kingdoms par narrow sea",
    tokenizer=tokenizer,
    model=g1,
    top_k=25,
    temperature=1.4,
)

# Process of saving a new Model and loading it into a new model
torch.save(g1.state_dict(),"Final_Model.Manav")

model=GPTModel(GPT_CONFIG_124M)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load("Final_Model.Manav", map_location=device))
model.to(device)


val_loader = create_dataloader_v1(
    raw_text[split_idx:],
    batch_size=1,
    max_length=GPT_CONFIG_124M["context_length"]//2,
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)


generate_and_print_sample_from_dataloader(tokenizer, val_loader, model)

