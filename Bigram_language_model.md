# Bigram Language Model — What Is It Doing?

This project builds a very simple **Bigram Language Model**.

The main idea is:

> **Given the current token, predict what token is most likely to come next.**

This is one of the simplest ideas behind modern language models such as GPT.

---

## 🧠 The Main Idea

Imagine the training text contains:

```text
hello there
```

The model tries to learn relationships between consecutive tokens:

```text
h → e
e → l
l → l
l → o
o → space
space → t
t → h
...
```

This is called a **bigram** model because it focuses on pairs of consecutive tokens.

The model does not understand the meaning of the text. It learns **statistical relationships between tokens**.

---

# 1. What Does the Model Learn?

The model learns:

```text
Current token → Likely next token
```

For example, after seeing lots of English text, it might learn that:

```text
q → u
```

is very common.

Conceptually, the model has a table like this:

```text
                 Possible next token
              a     b     c     d    ...
Current a    [ ?     ?     ?     ?    ...]
token   b    [ ?     ?     ?     ?    ...]
        c    [ ?     ?     ?     ?    ...]
        d    [ ?     ?     ?     ?    ...]
        ...
```

Each row represents the **current token**.

Each value represents how strongly the model thinks that token should be the **next token**.

---

# 2. Training: Learning Next-Token Prediction

Suppose the training text contains:

```text
hello
```

The model receives examples like:

```text
Current token     Target next token

h                 e
e                 l
l                 l
l                 o
```

The model makes predictions:

```text
h → ?
e → ?
l → ?
l → ?
```

The correct answers are:

```text
h → e
e → l
l → l
l → o
```

The model compares its predictions with the correct answers.

The difference between the prediction and the correct answer is measured using **loss**.

The training process then adjusts the model so that correct next tokens become more likely.

---

# 3. What Does the Model Ultimately Learn?

After training on a large amount of text, the model might learn something like:

```text
Current token: q

Possible next tokens:

u    0.91
a    0.02
e    0.01
i    0.01
o    0.01
...
```

This means:

> If the model sees `q`, it has learned that `u` is usually a very good next token.

For another token, the distribution might look like:

```text
Current token: t

h    0.25
o    0.20
e    0.15
a    0.10
...
```

So the model has learned **statistical patterns from the training text**.

---

# 4. What Are Logits?

The model first produces raw scores called **logits**.

For example:

```text
Input: q

a:  -2.3
b:  -4.1
c:  -3.7
...
u:   5.8
...
```

These numbers represent how strongly the model prefers each possible next token.

However, these numbers are **not probabilities yet**.

---

# 5. From Logits to Probabilities

The model uses **softmax** to convert logits into probabilities.

For example:

```text
Logits:

u: 5.8
a: -2.3
b: -4.1
...
```

After softmax:

```text
Probabilities:

u: 0.91
a: 0.001
b: 0.000
...
```

Now the model has a probability distribution for the next token.

The probabilities tell us:

> Which tokens are more likely to come next?

---

# 6. How Does Text Generation Work?

The `generate()` part is responsible for creating new text.

Suppose we start with:

```text
h
```

The model predicts:

```text
h → e
```

The generated text becomes:

```text
he
```

Now the model considers the newest token:

```text
e
```

and predicts:

```text
e → l
```

Now:

```text
hel
```

Then:

```text
l → l
```

becomes:

```text
hell
```

Then:

```text
l → o
```

becomes:

```text
hello
```

The process continues.

---

# 7. Generation Process

The whole generation process can be visualized as:

```text
             Start
               │
               ▼
        Current token
               │
               ▼
      Predict next token
               │
               ▼
      Convert scores into
         probabilities
               │
               ▼
       Sample a token
               │
               ▼
       Add token to text
               │
               ▼
      Use new token as
       current token
               │
               ▼
            Repeat
```

For example:

```text
h
↓
he
↓
hel
↓
hell
↓
hello
↓
hello ...
```

The model generates the text **one token at a time**.

---

# 8. Why Is Randomness Used?

The model does not necessarily always choose the most probable token.

Imagine the model predicts:

```text
Next-token probabilities:

a → 10%
b → 5%
c → 70%
d → 15%
```

The most likely token is:

```text
c
```

But the model can sample from the entire probability distribution.

Therefore:

```text
c
```

is likely to be selected, but occasionally:

```text
a, b, or d
```

could be selected.

This randomness makes generation less deterministic and can produce different outputs.

---

# 9. What Is the Limitation of a Bigram Model?

A Bigram Language Model is extremely simple.

It essentially considers only the **current token** when predicting the next token.

For example:

```text
The cat sat on the
```

A bigram model mainly asks:

```text
What usually comes after "the"?
```

It does not properly consider the entire sentence:

```text
The cat sat on the
^^^^^^^^^^^^^^^^^^
full context
```

Modern language models consider much more context.

For example, GPT-style models can use information from many previous tokens to predict the next one.

---

# 10. Bigram Model vs Modern GPT

| Bigram Model                        | Modern GPT                          |
| ----------------------------------- | ----------------------------------- |
| Very simple                         | Very complex                        |
| Looks mainly at the current token   | Uses a large context                |
| Learns token-to-token relationships | Learns complex patterns             |
| Small number of parameters          | Billions of parameters can be used  |
| No attention mechanism              | Uses Transformer attention          |
| Limited language understanding      | Much stronger language capabilities |
| Simple lookup table                 | Deep neural network                 |

Despite the huge difference, they share an important fundamental idea:

> **Predict the next token.**

---

# 11. The Big Picture

The entire project can be summarized as:

```text
                    TRAINING
                       │
                       ▼
                 Text Dataset
                       │
                       ▼
             Current → Next Token
                       │
                       ▼
            Learn Token Relationships
                       │
                       ▼
             ┌─────────────────────┐
             │ Bigram Language     │
             │ Model               │
             │                     │
             │ "a" → next token ?  │
             │ "b" → next token ?  │
             │ "c" → next token ?  │
             │ ...                 │
             └─────────────────────┘
                       │
                       ▼
                   GENERATION
                       │
                       ▼
                 Start with "h"
                       │
                       ▼
                Predict "e"
                       │
                       ▼
                Predict "l"
                       │
                       ▼
                Predict "l"
                       │
                       ▼
                Predict "o"
                       │
                       ▼
                 "hello..."
```

---

# 12. The Core Concept

The most important thing to understand is:

```text
                CURRENT TOKEN
                      │
                      ▼
              ┌──────────────┐
              │     MODEL    │
              └──────────────┘
                      │
                      ▼
             NEXT TOKEN SCORES
                      │
                      ▼
                  SOFTMAX
                      │
                      ▼
             NEXT TOKEN PROBS
                      │
                      ▼
              SAMPLE A TOKEN
                      │
                      ▼
              ADD TO SEQUENCE
                      │
                      ▼
                    REPEAT
```

So the model is essentially learning:

> **"When I see this token, what token usually comes next?"**

And during generation:

> **"Based on what I learned, what should I generate next?"**

---

# 13. Why This Is Important for Learning GPT

This Bigram Language Model is a useful starting point because it teaches the fundamental objective of language modeling:

```text
Previous tokens → Predict next token
```

Modern GPT models make this idea much more powerful.

Instead of simply doing:

```text
current token → next token
```

a Transformer-based GPT does something closer to:

```text
previous context
       ↓
token embeddings
       ↓
positional information
       ↓
self-attention
       ↓
multiple neural-network layers
       ↓
logits
       ↓
probabilities
       ↓
next-token prediction
```

The basic goal remains:

> **Predict the next token.**

---

## 🚀 Final Summary

This code creates a very small language model that:

1. Learns relationships between consecutive tokens.
2. Learns which tokens are likely to follow other tokens.
3. Produces logits representing next-token preferences.
4. Converts logits into probabilities.
5. Samples a next token.
6. Adds that token to the generated text.
7. Repeats the process to create a sequence.

In one sentence:

> **A Bigram Language Model learns which token is likely to come after each token, then uses those learned probabilities to generate new text one token at a time.**

This is one of the simplest possible versions of the **next-token prediction** idea used by modern GPT-style language models.
