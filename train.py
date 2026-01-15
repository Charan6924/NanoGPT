# Imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
torch.manual_seed(1337)
from attention_head import MultiHeadAttention, Head
from feedforward import FeedForward
from residual_block import Block

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 10000000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
patience = 5       
patience_counter = 0  
best_val_loss = float('inf')
droput = 0.2

# Preprocessing
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : "".join([itos[ix] for ix in l])
data = torch.tensor(encode(text), dtype = torch.long)

# Splits
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Data Loader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Model


# Instantiate model and optimizer
model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training 
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print('New high score!')
            print(f"New best model saved with val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
            

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


