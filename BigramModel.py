import torch
import torch.nn as nn
from attention_head import MultiHeadAttention, Head
from feedforward import FeedForward
from residual_block import Block
import torch.nn.functional as F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 256
droput = 0.2
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
n_embed = 384
n_head = 6
n_layer = 6

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
    self.position_embedding_table = nn.Embedding(block_size,n_embed)
    self.blocks = nn.Sequential(
        Block(n_embed,n_head),
        Block(n_embed,n_head),
        Block(n_embed,n_head),
        Block(n_embed,n_head),
        Block(n_embed,n_head),
        Block(n_embed,n_head),
        nn.LayerNorm((n_embed)))
    self.dropout = nn.Dropout(droput)
    self.lm_head = nn.Linear(n_embed,vocab_size)

  def forward(self,idx,targets=None):
    tok_emb = self.token_embedding_table(idx) # (B,T,C)
    # (B,T,vocab_size)
    pos_emb = self.position_embedding_table(torch.arange(idx.shape[1], device=device)) # (T,C)
    x = tok_emb + pos_emb # (B,T,C)
    x = self.dropout(x)
    x = self.blocks(x)
    logits = self.lm_head(x) 

    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits,targets)
    return logits,loss

  def generate(self,idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:,-block_size:]
      logits,loss = self(idx_cond)
      logits = logits[:,-1,:]
      probs = F.softmax(logits,dim = 1)
      idx_next = torch.multinomial(probs, num_samples = 1)
      idx = torch.cat((idx,idx_next),dim=1)
    return idx