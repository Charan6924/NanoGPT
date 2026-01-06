# NanoGPT

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
