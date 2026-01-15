from BigramModel import BigramLanguageModel
import torch

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}
device = 'cuda' 
decode = lambda l : "".join([itos[ix] for ix in l])

loaded_model = BigramLanguageModel()
loaded_model = loaded_model.to(device)
loaded_model.load_state_dict(torch.load('best_model.pth', weights_only=True))
loaded_model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(loaded_model.generate(context, max_new_tokens=500)[0].tolist()))
