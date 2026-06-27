import os
import tiktoken
import urllib.request
import torch.nn as nn
import nltk
nltk.download('punkt_tab')
import nltk.corpus
import re
from nltk import word_tokenize
import torch
import Class_Tokenizer_from_raw_text
from Class_Tokenizer_from_raw_text import Tokenizer_from_rawtext
import Class_Tokenizer_from_tokenised_vocab
from Class_Tokenizer_from_tokenised_vocab import Tokenizer_from_vocab
import Pytorch_Process
from Pytorch_Process import *

with open("File_path.txt","r")as f:
    raw_text = f.read()



# Applying Byte encoding on our verdict/raw text
tokenizer_tik = tiktoken.get_encoding("gpt2")
Tokens_text_byte_encoding=tokenizer_tik.encode(raw_text,allowed_special={"<|endoftext|>"})
print(Tokens_text_byte_encoding)


# Putting all tokens in Tokens_text_byte_encoding in sample window containing all tokens in small sets
# sampled_windows = [
#     [101, 102, 103, 104],  # window 1
#     [105, 106, 107, 108],  # window 2
#     [109, 110, 111, 112]   # window 3
# ]

def sliding_window_token_sampling(tokens, window_size, step_size):
    windows = []
    for i in range(0, len(tokens) - window_size + 1, step_size): #Because we want to include the last full window.
        window = Tokens_text_byte_encoding[i:i + window_size]
        windows.append(window)
    return windows

sample_windows=sliding_window_token_sampling(Tokens_text_byte_encoding, window_size=4, step_size=1)
print(sample_windows)

# Predicting tokens after each tokens in incrementally increasing windows
context_size=5
enc_sample=Tokens_text_byte_encoding[50:]

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "----->",desired)
    print(tokenizer_tik.decode(context), "----->",tokenizer_tik.decode([desired]))


# Loading the functionality of Pytorch process (tokenization , creating dataset for input and target , batches)
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=3,
    max_length=4,
    stride=4,
    shuffle=False
)
data_iter = iter(dataloader)
inputs,targets = next(data_iter)

print("Process of token embedding below")
# Process of token embedding and converting words to tokens
vocab_size = 50257
output_dim = 8
max_Size=4

torch.manual_seed(123)
# Same seed = same shuffle = same results

embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

token_embeddings=embedding_layer(inputs)

positional_embedding_layer=torch.nn.Embedding(max_Size, output_dim)
# This max_Size is used for total generations to do meaning how many positions

batch_size,seq_length=inputs.shape
positions_ids=torch.arange(seq_length).unsqueeze(0).expand(batch_size, seq_length)

positional_embeddings=positional_embedding_layer(positions_ids)
input_embedding = token_embeddings + positional_embeddings

print("Now we are printing information of Scores of dot products , attention weights , final context vectors ")
import Context_vectors
from Context_vectors import basic_self_attention

context,attention_weights,scores=basic_self_attention(input_embedding)
print(scores)
print(attention_weights)
print(context)


print("Now context vectors will be for another layer providing deeper insights")
context_vectors_2,attn_weights_2,scores_2=basic_self_attention(context)
print(scores_2)
print(attn_weights_2)
print(context_vectors_2)


print("Now we are proceeding with different method including the concept of casual attention mask")
from Context_vectors import CausalSelfAttention
c1=CausalSelfAttention(8,10)
context_three,attention_weight_3,scores_3,dropped_out=c1.forward(input_embedding)
print(scores_3)
print(attention_weight_3)
print(dropped_out)
print(context_three)


print("Now we will be computing multi dimension results")
from Context_vectors import MultiHeadCausalSelfAttention
m1=MultiHeadCausalSelfAttention(8,2)
context_multi=m1.forward(input_embedding)
print(context_multi)


from Transformer_Block_Implementation import MyLayerNorm,FeedForward
print("Printing the feed forward resutls computed from multi dimension results")
f1=FeedForward(8)
feedforwarded,normalised=f1.forward(context_multi)
print(feedforwarded)
print("Now we will seeing multi dimension results in the normalized form by applying layer norm")
print(normalised)
