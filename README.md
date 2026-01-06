# ShakespeareGPT

A character-level Generative Pre-trained Transformer (GPT) built from scratch in PyTorch. 
This model is trained on the complete works of William Shakespeare to generate original, authentic-sounding plays and poetry.
This project implements the core architecture of the "Transformer" paper (*Attention Is All You Need*) to learn the statistical structure of Shakespearean English. 

## Model Architecture
The model is a **Decoder only Transformer** featuring:
* **Multi Head Self Attention:** Parallel processing of context to capture relationships between distant characters.
* **Residual Connections:** Deep network training stability (6+ layers).
* **Layer Normalization:** Pre-norm formulation for better convergence.
* **Feed Forward Networks:** Position-wise processing with ReLU activation.

**Hyperparameters:**
* Context Window: 256 characters
* Embedding Dimension: 384
* Layers: 6
* Attention Heads: 6
* Parameters: ~10 Million


## Sample output
Where he is possible castle, that
hadst thou not pursuit, or in it in his rock grave;
The wearing shows faults, to cowards are; and serve,
I bent lies, I chase dishonour'd on Rome.

CORIOLANUS:
Marcius doth not the prove is the morth,
For whatsoever sit to-gao
Of He applot of this bed my friends,
Sending om the bloody deep.

First Coriol:
Anon that I advert in Lord Aumerle to taught:
O pray my with weak, I say never entreaties to agre?
One so Bondaggordon that has get to chese,
That might never 
