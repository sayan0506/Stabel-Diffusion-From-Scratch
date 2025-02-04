import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    # d_model = embedding_dim
    # in text embeddings are the dim that captures the meaning of word
    # in image embeddings are channels, which contains features that every pixel holds
    # in_proj = input projection layer
    # out_proj = output projection layer
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        # w metrices
        # instead of defining, wq, wk, wv individually
        # defining it as single big linear layer
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias = in_proj_bias)

        # this represents the wo matrix for multi-head attention
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)

        self.n_heads = 3 * d_embed
        self.d_head = d_embed/n_heads

    # in case of encoder of transformer causal mask concept use
    # where the current word can only attend the previous word in the sequence
    # the context of previous sequence is used to predict the next word
    # which prevents information leakage from future sequence
    def forward(self, x, causal_mask = False):
        # the input to self attention is the seq*embed_dim
        # x: (batch_size, seq_len, dim)

        # (batch_size, seq_len, emb_dim)
        input_shape = x.shape

        batch_size, seq_len, d_embed = input_shape

        # (batch_size, seq_len, H, dim/h)
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # obtain tensors for q, k, v
        # create 3 chunks, split across axis = 1
        # (batch_size, seq_len, emb_dim) -> 3 tensors (batch_size, seq_len, emb_dim)
        q, k, v = self.in_proj(x).chunk(3, dim = -1)

        # rehshape q, k, v a matrix which contains q, k, v values for individual head

        # (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, h, embed_dim/h) -> (batch_size, h, seq_len, d_head)
        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        # obtain attention scores
        # @ = matrix multiplication or vector dot product
        # (batch_size, h, seq_len, d_head) @ (batch_size, h, d_head, seq_len)
        weight = q @ k.transpose(-1,-2)

        if causal_mask:
            # mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            # this creates a mask matrix where upper triangle is masked
            # fill the upper triangle with -inf
            weight = weight.masked_fill(mask, -torch.inf)

        # after applying the causal mask, we do softmax
        # for multi_head attention is d_head
        # for single head attention is emb_dim
        weight /= math.sqrt(self.d_head)

        # apply softmax over all the sequence
        # where each query attends to all the keys
        # i,e softmax calculation is done across the seq_len
        # (batch_size, h, seq_len, seq_len) -> (batch_size, h, seq_len, seq_len)
        weight = F.softmax(weight, dim=-1)

        # (batch_size, h, seq_len, seq_len) @ (batch_size, h, seq_len, d_head)
        output = weight @ v

        # (batch_size, h, seq_len, d_head) -> (batch_size, seq_len, h, d_head)
        output = output.transpose(1,2)

        # concatenate attention heads
        # (batch_size, seq_len, h, d_head) -> (batch_size, seq_len, emb_dim)
        output = output.reshape(input_shape)

        # apply output projection layer
        # (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, emb_dim)
        output = self.out_proj(output)

        return output 
    

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias = True):
        super().__init__()

        # 1st query belongs to the feature map
        self.q_proj = nn.Linear(d_embed, d_embed, bias = in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias = in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed//n_heads

    def forward(self, x, y):
        # x(latent): (batch_size, seq_len_Q, dim_Q)
        # y(context): (batch_size, seq_len_KV, dim_KV) = (batch_size, 77, 768)

        # (batch_size, seq_len, dim)
        input_shape = x.shape
        batch_size, seq_len, d_embd = input_shape 

        # (batch_size, seq_len, h, dim/h)
        # divide each embedding of Q into multiple heads where each having dimension dim/n_heads
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # (batch_size, seq_len_Q, dim_Q) -> (batch_size, seq_len_Q, dim_Q)
        q = self.q_proj(x)
        # (batch_size, seq_len_KV, dim_KV) -> (batch_size, seq_len_KV, dim_Q)
        k = self.k_proj(y)
        # (batch_size, seq_len_KV, dim_KV) -> (batch_size, seq_len_KV, dim_Q)
        v = self.v_proj(y)

        # convert the q, k, v for multi-threads
        # (batch_size, seq_len_Q, dim_Q) -> (batch_size, seq_len_Q, n_heads, dim_Q/n_heads)
        q = q.view(interim_shape).transpose(1,2)
        # (batch_size, seq_len_Q, dim_KV) -> (batch_size, seq_len_KV, n_heads, dim_KV/n_heads)
        k = k.view(interim_shape).tranpose(1,2)
        # (batch_size, seq_len_Q, dim_KV) -> (batch_size, seq_len_KV, n_heads, dim_KV/n_heads)
        v = v.view(interim_shape).transpose(1,2)

        # (batch_size, n_heads, seq_len_Q, dim_Q/n_heads) @ (batch_size, n_heads, dim_KV/n_heads, seq_len_KV) -> (batch_size, n_heads, seq_len_Q, seq_len_KV)
        weight = q @ k.transpose(-1, -2)

        # (batch_size, n_heads, seq_len_Q, seq_len_KV) -> (batch_size, seq_len_Q, n_heads, seq_len_KV)
        weight /= math.sqrt(self.d_head)

        # (batch_size, n_heads, seq_len_Q, seq_len_KV)
        weight = F.softmax(weight, dim = -1)

        # (batch_size, n_heads, seq_len_Q, seq_len_KV) @ (batch_size, n_heads, seq_len_KV, dim_KV/n_heads) -> (batch_size, n_heads, seq_len_Q, dim_KV/n_heads) -> (batch_size, seq_len_Q, n_heads, dim_Q/n_heads)
        # matrix multiplication
        output = weight @ v

        # (batch_size, n_heads, seq_len_Q, dim_Q/n_heads) -> (batch_size, seq_len_Q, n_heads, dim_Q/n_heads) -> (batch_size, seq_len_Q, dim_Q) -> (batch_size, seq_len_Q, n_heads, dim_Q/h)
        # the transpose operation changes the view of the tensor
        # but does not rearrange the underlying data in the memory
        # thus the contiguos() operation used to create a new tensor with desired memory layout for the transposed tensor
        output = output.transpose(1, 2).contiguous() 

        # (batch_size, seq_len_Q, h, dim_Q/h) -> (batch_size, seq_len_Q, dim_Q)
        output = output.view(input_shape)

        # (batch_size, seq_len_Q, dim_Q) -> (batch_size, seq_len_Q, dim_Q)
        output = self.out_proj(output)

        return output









        
 