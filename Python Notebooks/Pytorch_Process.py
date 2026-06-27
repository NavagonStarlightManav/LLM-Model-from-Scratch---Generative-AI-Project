from torch.utils.data import Dataset, DataLoader
import torch
import tiktoken

# Tensors are data units that go into a neural network and CPUs or GPUs for fast computations

class GPTDatasetV1(Dataset): # Its makes class behave like dataset
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        self.tokenizer_decode = tiktoken.get_encoding("gpt2")

        # Use a sliding window to chunk the book into overlapping sequences
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk)) # Store data in form of tensors
            self.target_ids.append(torch.tensor(target_chunk)) # Store data in form of tensors

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
            # return (
            #     self.tokenizer_decode.decode(self.input_ids[idx].tolist()),
            #     self.tokenizer_decode.decode(self.target_ids[idx].tolist())
            # )
            return self.input_ids[idx], self.target_ids[idx]


# Drop last will drop last batch if it doesn't have tokens up to size specified

# num worker provide parallel workers for loading simultaneously if > 0

# dataloader is like conveyor belt that deals with final data delivery from dataset

# max_length specifies context size or number of tokens in one sample

# batch size specifies number of samples at once

# we will see that data loader functionality how helpful for receiving dataset in proper from

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader
