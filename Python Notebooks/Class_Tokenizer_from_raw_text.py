import os
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize

class Tokenizer_from_rawtext:
    def __init__(self,vocab):
        self.vocab = vocab
        self.tokens_text = word_tokenize(vocab)
        self.all_words = sorted(set(self.tokens_text))
        self.special_tokens = ["<|endoftext|>", "<unk>"]
        for token in self.special_tokens:
            if token not in self.all_words:
                self.all_words.append(token)
        self.tokens = {token:integer for integer,token in enumerate(self.all_words)}
        self.reverse_tokens = {v: k for k, v in self.tokens.items()}

    def encode(self,text):
        self.tokenized_text=word_tokenize(text)
        self.ids = [
            self.tokens[word] if word in self.tokens else self.tokens["<unk>"]
            for word in self.tokenized_text
        ]
        return self.ids

    def decode(self,ids):
        self.text = " ".join([self.reverse_tokens[i] for i in ids])
        return self.text




