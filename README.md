# 🧠 MiniGPT — A GPT-Style Language Model Built From Scratch

> Understanding Transformers by building one, line by line — not just calling an API.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Architecture-Transformer-6f42c1?style=flat-square" alt="Transformer">
  <img src="https://img.shields.io/badge/Status-Educational%20%2F%20In%20Progress-yellow?style=flat-square" alt="Project Status">
</p>

---

## 📌 Overview

**MiniGPT** is a small, from-scratch implementation of a GPT-style autoregressive language model, built using Python and PyTorch. It implements the core building blocks of the Transformer decoder architecture — tokenization, embeddings, self-attention, multi-head attention, feed-forward layers, and autoregressive text generation — without relying on pre-built Transformer libraries like `transformers` or `xformers` for the model internals.

The goal of this project is **not** to compete with production LLMs, but to demonstrate a working, hands-on understanding of how models like GPT-2 and GPT-3 actually work under the hood.

---

## 🎯 Why I Built This Project

I am trying to understand how attention works and also:

- Implementing the Transformer decoder architecture manually instead of importing it
- Understanding *why* self-attention works, not just *that* it works
- Building intuition for the full pipeline — tokenization → training → generation
- Creating a reference project I can revisit, extend, and explain in technical interviews

---

## ✨ Key Features

- 🔤 Custom tokenization pipeline
- 📐 Learned token + positional embeddings
- 🎯 Scaled dot-product self-attention implemented manually
- 🧩 Multi-head attention with causal (masked) attention
- 🏗️ Stackable Transformer decoder blocks with residual connections and layer normalization
- 🔁 Autoregressive next-token generation loop
- 🌡️ Configurable sampling: temperature
- ⚙️ Configurable hyperparameters (layers, heads, embedding size, context length)

---

## 🏛️ GPT Architecture Explanation

MiniGPT follows the **decoder-only Transformer** architecture popularized by GPT-2/GPT-3. Given a sequence of input tokens, the model predicts a probability distribution over the next token, one step at a time. Each Transformer block combines **masked self-attention** (to let each token look at previous tokens only) with a **position-wise feed-forward network**, wrapped in residual connections and layer normalization.

### Architecture Diagram

```
                     ┌────────────────────────────┐
                     │        Input Tokens        │
                     └──────────────┬─────────────┘
                                    ▼
                     ┌────────────────────────────┐
                     │  Token Embedding + Position│
                     │        Embedding (sum)     │
                     └──────────────┬─────────────┘
                                    ▼
              ┌───────────────────────────────────────────┐
              │              Transformer Block ×N         │
              │  ┌─────────────────────────────────────┐  │
              │  │   Layer Norm                        │  │
              │  │        ↓                            │  │
              │  │   Masked Multi-Head Self-Attention  │  │
              │  │        ↓                            │  │
              │  │   Residual Connection (+)           │  │
              │  │        ↓                            │  │
              │  │   Layer Norm                        │  │
              │  │        ↓                            │  │
              │  │   Feed-Forward Network (MLP)        │  │
              │  │        ↓                            │  │
              │  │   Residual Connection (+)           │  │
              │  └─────────────────────────────────────┘  │
              └───────────────────┬───────────────────────┘
                                    ▼
                     ┌────────────────────────────┐
                     │      Final Layer Norm      │
                     └──────────────┬─────────────┘
                                    ▼
                     ┌──────────────────────────────┐
                     │  Language Modeling Head      │
                     │  (Linear → Vocabulary Logits)│
                     └──────────────┬───────────────┘
                                    ▼
                     ┌───────────────────────────────┐
                     │   Softmax → Next-Token Probs  │
                     └───────────────────────────────┘
```

---

## 🧩 Explanation of Major Components

### 1. Tokenization

Converts raw text into a sequence of integer token IDs that the model can process. `[TODO: specify tokenizer type — e.g., character-level, byte-pair encoding, or custom vocabulary]`

### 2. Token Embeddings

Each token ID is mapped to a dense vector via a learned embedding matrix of shape `(vocab_size, embedding_dim)`, giving the model a numerical representation of each token's meaning.

### 3. Positional Embeddings

Since self-attention has no inherent notion of token order, a positional embedding is added to each token embedding to encode **where** a token appears in the sequence.

### 4. Self-Attention

Allows each token to "look at" other tokens in the sequence and weigh their relevance when building its own representation — this is the core mechanism that lets Transformers model context and long-range dependencies.

### 5. Query, Key, Value (Q, K, V)

Each token's embedding is projected into three vectors:

- **Query (Q):** what this token is "looking for"
- **Key (K):** what this token "offers" to others
- **Value (V):** the actual content passed along if attended to

### 6. Scaled Dot-Product Attention

Attention scores are computed as the dot product of queries and keys, scaled to stabilize gradients:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where $d_k$ is the dimensionality of the key vectors.

### 7. Causal / Masked Attention

To preserve the autoregressive property, each token is only allowed to attend to **itself and earlier tokens** — future tokens are masked out (set to $-\infty$ before softmax) so the model cannot "see" the answer during training.

### 8. Multi-Head Attention

Instead of computing a single attention function, the model runs several attention "heads" in parallel, each learning to focus on different types of relationships, then concatenates and projects the results.

### 9. Feed-Forward Network (MLP)

A position-wise fully connected network (typically Linear → activation → Linear) applied independently to each token, adding non-linearity and additional representational capacity.

### 10. Layer Normalization

Normalizes activations across the embedding dimension to stabilize and speed up training.

### 11. Residual Connections

Each sub-layer's input is added back to its output (`x + Sublayer(x)`), helping gradients flow through deep networks and preventing degradation.

### 12. Transformer Blocks

A Transformer block combines masked multi-head attention, a feed-forward network, layer normalization, and residual connections into one repeatable unit. MiniGPT stacks `N` of these blocks.

### 13. Language Modeling Head

A final linear layer projects the output of the last Transformer block onto the vocabulary size, producing logits that are converted into next-token probabilities via softmax.

---

## 🔮 Autoregressive Next-Token Prediction

GPT-style models generate text **one token at a time**, feeding each prediction back in as input for the next step.

**Example:**

```
Input:  "The cat sat on the"
Step 1: model predicts → "mat"
Input:  "The cat sat on the mat"
Step 2: model predicts → "."
Final:  "The cat sat on the mat."
```

At each step, the model outputs a probability distribution over the entire vocabulary, and a sampling strategy (see below) picks the next token.

---

## 🏋️ Training Pipeline

```
 Raw Text Corpus
        │
        ▼
   Tokenization
        │
        ▼
 Sequence Batching (context windows)
        │
        ▼
   Forward Pass (MiniGPT)
        │
        ▼
 Cross-Entropy Loss (predicted vs actual next token)
        │
        ▼
   Backpropagation
        │
        ▼
 Optimizer Step (e.g., AdamW) `[TODO: confirm optimizer]`
        │
        ▼
   Repeat for N epochs / steps
```

---

## ✍️ Text Generation Pipeline

```
 Prompt Text
      │
      ▼
 Tokenize Prompt
      │
      ▼
 ┌───────────────────────────────┐
 │  Loop until max tokens:       │
 │   1. Forward pass             │
 │   2. Get logits for last tok  │
 │   3. Apply temperature/top-k/ │
 │      top-p sampling           │
 │   4. Sample next token        │
 │   5. Append to sequence       │
 └───────────────────────────────┘
      │
      ▼
 Detokenize → Final Generated Text
```

---

## 📦 Requirements

- Python 3.10+
- PyTorch 2.x

Example `requirements.txt`:

```
torch>=2.0
numpy
tqdm
```

---

## 🧪 Example Input/Output

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

---

## 📚 What I Learned

Building MiniGPT from scratch helped me develop a much deeper, practical understanding of:

- How self-attention actually computes relationships between tokens mathematically
- Why causal masking is essential for autoregressive language modeling
- How positional information is injected into a permutation-invariant attention mechanism
- The role of residual connections and layer normalization in stabilizing deep network training
- How training loss connects to next-token prediction via cross-entropy
- The practical differences between greedy decoding and probabilistic sampling strategies
- How architectural choices (layers, heads, embedding size) affect model behavior and compute cost

---

## 📖 References

- Vaswani, A. et al. (2017). *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners (GPT-2).* OpenAI.
- Brown, T. et al. (2020). *Language Models are Few-Shot Learners (GPT-3).* OpenAI.
- Andrej Karpathy — *Let's build GPT: from scratch, in code, spelled out* and related educational materials on neural networks and language models.

---

## Acknowledgements

This project was built for educational purposes as part of my personal journey into deep learning and NLP. Special thanks to the authors and educators whose research papers and teaching materials made it possible to understand and reproduce these ideas at a small scale.

---