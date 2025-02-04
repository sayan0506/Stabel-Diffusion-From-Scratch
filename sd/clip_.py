import torch
from torch.nn import functional as F
from torch import nn
from sd.attention_ import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()

        # total vocabulary size of the dictionary
        # embedding dimension
        # seq length or total number of tokens
        self.token_embedding = nn.Embedding(n_vocab, n_embd)

        # a learnable weight matrix which encodes the positional information for the individual token
        # in case of bert the positional embedding is not learnable
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))

    def forward(self, tokens):
        #(batch_size, seq_len) -> (batch_size, seq_len, emb_dim)
        x = self.token_embedding(tokens)

        # (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, emb_dim)
        x += self.position_embedding

        return x
    


class CLIPLayer(nn.Module):
    # just like transformer encoder layer
    def __init__(self, n_head: int, n_embd:int):
        super().__init__()

        # pre attention norm
        self.layernorm1 = nn.LayerNorm(n_embd)
        # self-attention
        self.attention = SelfAttention(n_head, n_embd)
        # pre-norm of FFN
        self.layernorm2 = nn.LayerNorm(n_embd)
        # FFN
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4* n_embd, n_embd)

    def forward(self, x):
        # (Batch_size, seq_len, emb_dim) -> (Batch_size, seq_len, emb_dim)
        residue = x

        ### Self-attention
        x = self.layernorm1(x)

        x = self.attention(x, causal_mask=True)
        x+= residue

        ### Feed forward
        residue = x

        x = self.layernorm2(x)

        # (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, 4*emb_dim
        x = self.linear1(x)        
        x = torch.sigmoid(1.702 * x) # quick gelu activation

        # (batch_size, seq_len, 4*emb_dim) -> (batch_size, seq_len, emb_dim)
        x = self.linear2(x)

        x += residue

        return x



class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # clip has embeddings, which convert token to embeddings
        # the dimension of embeddings vector is 712
        # embedding is representation that holds meaning of the word in the embedding dimension space kind of representation
        
        # the voabulary size is 49408
        # max sequence length 77
        # the embedding dimension is 768
        self.embedding = CLIPEmbedding(49408, 768, 77)

        # define architecture for clip text encoder
        # it uses 12 transformer encoder we call it clip layer
        # as clip used bert

        # for each layer we use 12 attention heads
        self.layers = nn.ModuleList(
            [CLIPLayer(12,768) for _ in range(12)]
        )

        # at the end we use a layernorm, which normalizes the distribution across all the features or channels inside a layer for a single sample
        self.layernorm = nn.LayerNorm(768)
 
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # token is longtensor as the input id is a number
        # which represents the position of tokens inside the vocabulary

        # confirms the token data type to long
        tokens = tokens.type(torch.long)

        # embeddings of the token
        # (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dim)
        state = self.embedding(tokens)

        # apply encoder same as transformer encoder layer
        for layer in self.layers:
            # (Batch_size, seq_len, dim) -> (Batch_size, seq_len, dim)
            state = layer(state)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        output = self.layernorm(state)

        return output