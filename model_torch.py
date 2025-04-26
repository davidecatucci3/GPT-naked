import torch.nn as nn
import torch

#! hyperparameters
vocab_size = 32768
ctx_length = 512
n_layers = 4
d_model = 128

#! GPT functions

#! model
class PreProcessing:
    def __init__(self):
        self.embedding = nn.Embedding(vocab_size, d_model)

    def __call__(self, tokens):
        embed = self.embedding(tokens) # (B, T, C)

        return embed

class Block:
    def __init__(self):
        pass

class PostProcessing:
    def __init__(self):
        pass

class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.pre_processing = PreProcessing()                  
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        self.post_processing = PostProcessing()

    def forward(self, tokens):
        B, T = tokens.size()

        x = self.pre_processing(x) # (B, T) -> (B, T, C)