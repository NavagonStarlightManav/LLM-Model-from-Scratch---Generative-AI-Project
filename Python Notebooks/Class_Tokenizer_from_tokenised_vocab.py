import os
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize

class Tokenizer_from_vocab:
    def __init__(self,vocab):
        self.vocab = vocab
        self.reverse_tokens = {v: k for k, v in self.vocab.items()}

    def encode(self,text):
        self.tokenized_text=word_tokenize(text)
        self.ids = [
            self.vocab[word] if word in self.vocab else self.vocab["<unk>"]
            for word in self.tokenized_text
        ]
        return self.ids
    def decode(self,ids):
        self.text = " ".join([self.reverse_tokens[i] for i in ids])
        return self.text




