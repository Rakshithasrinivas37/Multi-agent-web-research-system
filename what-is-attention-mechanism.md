# Attention Mechanisms in Machine Learning

## Executive Summary
Attention mechanisms allow neural models to dynamically focus on the most relevant parts of an input when generating each output token. Originating as **additive (Bahdanau) attention** for neural machine translation, the idea has evolved into **self‑attention** and **multi‑head attention** that power modern Transformers. Attention improves translation quality (e.g., BLEU gains on WMT’14 English‑German) and underlies large language models, but its naïve implementation incurs quadratic time‑ and memory‑costs with sequence length. Recent work proposes more efficient transformations (e.g., Entmax, ALiBi) to mitigate these limits.

---

## 1. High‑Level Concept

Attention augments a sequence‑to‑sequence model by letting the decoder (or any downstream component) **weight different source positions** according to their relevance for the current prediction. Rather than compressing the entire source into a single fixed‑size vector, the model computes a **context vector** as a weighted sum of source annotations, where the weights are learned probabilities that reflect alignment strength [5]. This mechanism “relieves the encoder from the burden of having to encode all information … into a fixed‑length vector” [5].

---

## 2. Original Formulation: Additive (Bahdanau) Attention (2014)

The first widely adopted attention scheme was introduced by Bahdanau et al. (2014) for neural machine translation. For each decoder step *i*:

1. **Alignment scores** *eᵢⱼ* are produced by a feed‑forward network that takes the previous decoder hidden state *sᵢ₋₁* and each encoder annotation *hⱼ*.
2. The scores are normalised with a softmax to obtain **attention weights** *αᵢⱼ* (the probability that output *i* aligns to source position *j*).
3. The **context vector** is the expectation over annotations:

\[
c_i = \sum_j \alpha_{ij}\, h_j
\]

This formulation explicitly models the “importance of the annotation *hⱼ* with respect to the previous hidden states *sᵢ₋₁*” [5].

*Limitations of the evidence*: The exact functional form of the feed‑forward network for *eᵢⱼ* (the “additive” part) is not quoted in the retrieved material.

---

## 3. Scaled Dot‑Product Attention & Self‑Attention

The Transformer replaces additive scoring with **scaled dot‑product attention**. An attention function maps a **query** *q* and a set of **key–value** pairs *(k, v)* to an output computed as a weighted sum of the values, where the weights are the softmax of the dot products between the query and each key, scaled by √dₖ. The retrieved source describes this mapping abstractly:

> “An attention function can be described as mapping a query and a set of key‑value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum … Scaled Dot‑Product Attention” [6].

In **self‑attention**, the queries, keys, and values are linear projections of the *same* input sequence, enabling each position to attend to all others.

---

## 4. Multi‑Head Attention

Transformers employ **multiple parallel attention heads** to capture different relational subspaces:

* The model dimension *d_model* is split into *h* heads (the paper uses *h = 8*).
* Each head works with reduced dimensions *dₖ = dᵥ = d_model / h = 64* (for the base model) [6].
* After independent scaled dot‑product attention, the heads’ outputs are concatenated and linearly projected back to *d_model*.

Multi‑head attention is used in three places: encoder self‑attention, decoder self‑attention (masked), and encoder‑decoder attention where queries come from the decoder and keys/values from the encoder [6].

---

## 5. Attention Variants

| Variant | Description | Evidence |
|---------|-------------|----------|
| **Global (soft) attention** | Each output can attend to all source positions; weights are softmax probabilities. | Implicit in additive and Transformer formulations [5][6]. |
| **Local attention** | Attention is restricted to a window around each position (e.g., window size *D = 10*). | Reported in a local‑attention system achieving BLEU improvements [8]. |
| **Hard attention** | Discrete selection of a single source position (non‑differentiable). | Not covered by the retrieved sources; cannot be detailed. |

The sources discuss global and local attention but do not provide a formal treatment of hard vs. soft attention.

---

## 6. Empirical Impact

* **Machine Translation** – Adding a local‑attention variant (window *D = 10*) to a baseline system reduced perplexity and raised tokenized BLEU from 18.1 to 20.9, with an ensemble reaching 23.0 [8].
* **Large Language Models** – The attention mechanism is a core component of the GPT series, which have demonstrated state‑of‑the‑art performance on language‑model benchmarks, though quantitative scores are not provided in the available material [4].

---

## 7. Computational Limitations

A naïve attention implementation requires **O(n² · d)** time and memory, where *n* is sequence length and *d* the model dimension. This quadratic scaling becomes prohibitive for long sequences, limiting practical deployment [9].

---

## 8. Efficient‑Attention Variants

Recent research proposes alternatives that retain expressive power while reducing cost:

* **Entmax transformation** – Replaces the softmax with an *α‑entmax* function that can produce sparse attention distributions, controlled by a threshold τ [7].
* **ALiBi (Linear Biases)** – Adds a linear positional bias to the scaled‑dot‑product scores, encouraging locality without extra computation [7].

These methods illustrate ongoing efforts to address the quadratic bottleneck.

---

## 9. Conclusion

Attention mechanisms have transformed sequence modeling by enabling dynamic, content‑based weighting of inputs. Starting from additive attention for translation, the field progressed to self‑attention and multi‑head designs that underpin Transformers and large language models. While attention yields substantial quality gains, its quadratic complexity motivates a vibrant line of research into sparse and bias‑augmented variants.

---

## References

- [1] Introduction to Attention Mechanism – erdem.pl (illustrated intuition).
- [4] “Attention Is All You Need” – parseur.com blog (qualitative impact on GPT series).
- [5] Bahdanau, D., Cho, K., & Bengio, Y. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*. https://arxiv.org/pdf/1409.0473
- [6] Vaswani, A. et al. (2017). *Attention Is All You Need*. https://arxiv.org/pdf/1706.03762
- [7] Recent efficient attention paper (Entmax, ALiBi). https://arxiv.org/pdf/2508.08369
- [8] Stanford NLP (2015) – Local attention experiments with BLEU results. https://nlp.stanford.edu/pubs/emnlp15_attn.pdf
- [9] Michael Brenndoerfer – Quadratic scaling of attention. https://mbrenndoerfer.com/writing/attention-complexity-quadratic-scaling-memory-efficient-transformers
