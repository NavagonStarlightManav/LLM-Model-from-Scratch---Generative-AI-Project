
#  Building GPT-2 From Scratch with PyTorch


> A complete educational implementation of a GPT-style Large Language Model built from scratch using **PyTorch**, covering every major component of the Transformer decoder architecture—from tokenization and embeddings to instruction fine-tuning and text generation.



# Project Preview

<img width="2258" height="1228" alt="image" src="https://github.com/user-attachments/assets/f6920b7f-3d77-456b-bc30-e1dc264980de" />



# Overview

Unlike projects that simply use pretrained language models, this repository focuses on understanding **how GPT works internally** by implementing every core module manually.

The project follows the complete lifecycle of a GPT model:

- Raw text preprocessing
- Tokenization
- Vocabulary construction
- Token & positional embeddings
- Self-attention
- Multi-head attention
- Transformer decoder blocks
- GPT architecture
- Loading pretrained GPT-2 weights
- Instruction fine-tuning
- Text generation

This repository serves as both a learning resource and a practical implementation of modern Transformer-based language models.


#  Features

- ✅ Custom Tokenizer
- ✅ Vocabulary Builder
- ✅ Token Embeddings
- ✅ Positional Embeddings
- ✅ Context Windows
- ✅ Self-Attention
- ✅ Masked Multi-Head Attention
- ✅ Feed Forward Networks
- ✅ Layer Normalization
- ✅ Residual Connections
- ✅ Dropout
- ✅ Transformer Decoder Blocks
- ✅ GPT Architecture
- ✅ GPT-2 Weight Loading
- ✅ Instruction Fine-Tuning
- ✅ Text Generation
- ✅ Modular PyTorch Implementation



# Development Pipeline

```text
Raw Text
    │
    ▼
Tokenization
    │
    ▼
Vocabulary
    │
    ▼
Token IDs
    │
    ▼
Token Embeddings
    │
    ▼
Positional Embeddings
    │
    ▼
Context Representation
    │
    ▼
Self Attention
    │
    ▼
Multi-Head Attention
    │
    ▼
Transformer Block
    │
    ▼
GPT Architecture
    │
    ▼
Load GPT-2 Weights
    │
    ▼
Instruction Fine-Tuning
    │
    ▼
Text Generation
```


# Concepts Implemented

| Module | Status |
|---------|:------:|
| Tokenizer | ✅ |
| Vocabulary Builder | ✅ |
| Token Embeddings | ✅ |
| Positional Embeddings | ✅ |
| Context Windows | ✅ |
| Self-Attention | ✅ |
| Multi-Head Attention | ✅ |
| Feed Forward Network | ✅ |
| Layer Normalization | ✅ |
| Residual Connections | ✅ |
| Dropout | ✅ |
| Transformer Decoder | ✅ |
| GPT Architecture | ✅ |
| GPT-2 Weight Loading | ✅ |
| Instruction Fine-Tuning | ✅ |
| Text Generation | ✅ |


# Implementation Journey

## 1. Tokenization

Convert raw text into meaningful tokens that can be processed by the model.



## 2. Embeddings

Generate dense vector representations using token embeddings and positional embeddings.

<img width="809" height="859" alt="Token Embeddings to Logit Vector" src="https://github.com/user-attachments/assets/61ff1457-06d4-4a88-9c8c-4a2c2569c0b2" />

---

## 3. Context Window

Prepare context windows to enable autoregressive next-token prediction.




## 4. Self & Multi-Head Attention

Implement scaled dot-product attention with causal masking and combine multiple attention heads to capture richer contextual relationships.




## 5. Transformer Decoder Block

Each decoder block combines masked attention, feed-forward layers, residual connections, layer normalization, and dropout.

<img width="1200" height="1188" alt="Transformer Architecture image" src="https://github.com/user-attachments/assets/c9ad8cc2-1c9f-40f3-970c-26ab5e5617e8" />



## 6. GPT Architecture

Stacked multiple Transformer decoder blocks to construct a complete GPT-style language model.

<img width="496" height="254" alt="Implementing a GPT Architecture" src="https://github.com/user-attachments/assets/1aceba6c-682d-4fa5-ad0e-b80664e8a49e" />




## 7. Loading GPT-2 Weights

Initialized the custom architecture with pretrained GPT-2 weights and validate compatibility with the implemented model.




## 8. Instruction Fine-Tuning

Fine-tuned the model on instruction-response datasets to improve its ability to follow prompts and generate helpful responses.



## 9. Final Output

Generate coherent text using the fine-tuned GPT model.

<img width="2258" height="1228" alt="Screenshot 2026-06-27 081607" src="https://github.com/user-attachments/assets/2771b828-cf03-45c3-bc49-10ee75954c04" />



#  Repository Structure

```text
LLM-From-Scratch/
│
├── assets/
├── datasets/
├── models/
├── notebooks/
├── src/
├── requirements.txt
├── README.md
└── LICENSE
```

# Tech Stack

| Category | Technology |
|----------|------------|
| Programming | Python |
| Deep Learning | PyTorch |
| NLP | GPT-2, Tokenization |
| Libraries | NumPy, Matplotlib, tiktoken |
| Development | Jupyter Notebook, PyCharm |
| Version Control | Git & GitHub |


# Key Learning Outcomes

- Built a GPT-style Transformer architecture from scratch.
- Understood the mathematics behind self-attention.
- Implemented modular Transformer decoder blocks.
- Learned how token and positional embeddings interact.
- Loaded pretrained GPT-2 weights into a custom implementation.
- Fine-tuned the model using instruction-response datasets.
- Generated text using an autoregressive decoder.



#  Future Improvements

- Retrieval-Augmented Generation (RAG)
- LoRA / QLoRA Fine-Tuning
- RLHF
- FlashAttention
- Quantization
- FastAPI Deployment
- Streamlit / Gradio Interface
- Hugging Face Integration
- Docker Support


#  Author

**Manav Goyal**
