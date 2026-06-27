import sys
import torch
import tqdm
import numpy as np


print("tqdm version:", tqdm.__version__)

sys.path.append("/content/drive/MyDrive/Manav LLM/")


from GPU_Testing_Block_Implementation import GPTModel
from Concepts_Implementation import generate, tokenizer
from gpt_download import download_and_load_gpt2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
print(f"Model settings: {settings}")
print(f"Parameter keys: {params.keys()}")
print(f"Embedding matrix shape: {params['wte'].shape}")


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})


gpt = GPTModel(NEW_CONFIG).to(device)
gpt.eval()


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: left {left.shape}, right {right.shape}")
    return torch.nn.Parameter(torch.tensor(right, dtype=left.dtype, device=left.device))


def load_weights_into_gpt(gpt, params):
    gpt.token_embedding.weight = assign(gpt.token_embedding.weight, params["wte"])
    gpt.positional_embedding.weight = assign(gpt.positional_embedding.weight, params["wpe"])

    for b in range(len(params["blocks"])):
        block = gpt.trf_blocks[b]
        block_params = params["blocks"][b]

        block.attn.c_attn.weight = assign(block.attn.c_attn.weight, block_params["attn"]["c_attn"]["w"].T)
        block.attn.c_attn.bias = assign(block.attn.c_attn.bias, block_params["attn"]["c_attn"]["b"])

        block.attn.out_proj.weight = assign(block.attn.out_proj.weight, block_params["attn"]["c_proj"]["w"].T)
        block.attn.out_proj.bias = assign(block.attn.out_proj.bias, block_params["attn"]["c_proj"]["b"])

        block.ff.net[0].weight = assign(block.ff.net[0].weight, block_params["mlp"]["c_fc"]["w"].T)
        block.ff.net[0].bias = assign(block.ff.net[0].bias, block_params["mlp"]["c_fc"]["b"])
        block.ff.net[2].weight = assign(block.ff.net[2].weight, block_params["mlp"]["c_proj"]["w"].T)
        block.ff.net[2].bias = assign(block.ff.net[2].bias, block_params["mlp"]["c_proj"]["b"])

        block.ln1.scale = assign(block.ln1.scale, block_params["ln_1"]["g"])
        block.ln1.shift = assign(block.ln1.shift, block_params["ln_1"]["b"])
        block.ln2.scale = assign(block.ln2.scale, block_params["ln_2"]["g"])
        block.ln2.shift = assign(block.ln2.shift, block_params["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])

    # Weight tying
    gpt.out_head.weight = gpt.token_embedding.weight

load_weights_into_gpt(gpt, params)


torch.manual_seed(123)
start_context = "Once there used to be a king and queen across narrow sea"
start_ids = torch.tensor([tokenizer.encode(start_context)], dtype=torch.long).to(device)

if start_ids.shape[1] > GPT_CONFIG_124M["context_length"]:
    start_ids = start_ids[:, -GPT_CONFIG_124M["context_length"]:]


with torch.no_grad():
    generated = generate(
        model=gpt,
        idx=start_ids,
        max_new_tokens=50,
        context_size=GPT_CONFIG_124M["context_length"],
        temperature=1.4,
        top_k=25,
        eos_id=None
    )

print("Sample Generated Text:\n")
print(tokenizer.decode(generated[0].tolist()).replace("\n", " "))
