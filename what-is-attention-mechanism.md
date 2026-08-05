# Attention Mechanisms in Neural Networks
*A deep‑dive technical overview*

---

## Executive Summary
Attention mechanisms enable neural models to **selectively focus** on the most relevant parts of their input when constructing a new representation. Originating in neural machine translation (Bahdanau et al., 2014) as an *alignment model*, attention has become the core operation of modern architectures such as the Transformer. The basic operation projects inputs into **queries**, **keys**, and **values**, computes similarity scores (often via dot‑product or an additive network), normalises them with a softmax, and produces a weighted sum of the values. Extensions include **scaled dot‑product self‑attention**, **multi‑head** parallelisation, and numerous variants (additive vs. multiplicative, soft vs. hard, self‑ vs. cross‑attention). Attention underpins state‑of‑the‑art NLP models (BERT, GPT) and vision models (Vision Transformer), but its naïve quadratic cost in sequence length poses scalability challenges.

---

## 1. Definition & Basic Intuition

- **Intuitive view** – Attention lets a model “focus” on the most relevant vectors among a set when generating a new representation.
- **IBM description** – A *query* vector **q** (obtained after a linear projection **Wᵩ**) is added to each *key* vector **k** (projected by **Wᴷ**). The sum passes through a **tanh** activation, is multiplied by a value weight **wᵥ**, and finally normalised by a **softmax** to obtain attention weights. The weighted sum of the *value* vectors yields the context vector [2].
- **VaultFS description** – In the common multiplicative form, attention scores are computed as a **dot product** between a query and every key, indicating “how much focus should be placed on other parts of the input sequence for each element” [3].

> **Formal definition (partial)** – Given a query **q**, a set of keys \(\{k_i\}\) and values \(\{v_i\}\), attention produces weights \(\alpha_i\) and an output vector:
> \[
> \alpha_i = \frac{\exp\!\big(f_{\text{attn}}(k_i, q)\big)}{\sum_j \exp\!\big(f_{\text{attn}}(k_j, q)\big)},
> \qquad
> \text{output} = \sum_i \alpha_i\, v_i
> \]
> where \(f_{\text{attn}}\) is a similarity function (e.g., dot product or an additive network) [5].

---

## 2. Origin: Bahdanau et al. (2014)

Bahdanau, Cho, and Bengio introduced attention as an **alignment model** on top of an encoder‑decoder RNN for neural machine translation [11]. Instead of compressing the entire source sentence into a single fixed‑length vector, the decoder at each time step computes a weighted sum of the encoder hidden states. The weights are produced by a small neural network (the alignment model) that scores each encoder state with respect to the current decoder state (the query). This mechanism allows the decoder to “look at” different source positions dynamically, improving translation quality.

*Evidence limitation*: The retrieved snippet does not contain the full mathematical formulation of the alignment model; only the high‑level description is available.

---

## 3. Scaled Dot‑Product Self‑Attention (Vaswani et al., 2017)

The Transformer paper introduced **self‑attention** where queries, keys, and values all originate from the same sequence [8]. The core operation computes a similarity score by a dot product between query and key vectors and then **scales** the result (by \(1/\sqrt{d_k}\) in the original formulation).

- **Self‑attention** enables each token to attend to every other token in the same sequence.
- The scaling factor mitigates the growth of dot‑product magnitudes with dimensionality, stabilising the softmax.

*Evidence limitation*: The explicit scaled‑dot‑product equation (including the \(1/\sqrt{d_k}\) factor) is not present in the retrieved excerpts; we can only state that the paper introduced the concept.

---

## 4. Multi‑Head Attention

Multi‑head attention runs several independent attention operations (“heads”) in parallel, each with its own learned linear projections for queries, keys, and values. The outputs of all heads are concatenated and linearly transformed to produce the final representation [10]. This design allows the model to capture information from different representation subspaces simultaneously.

*Evidence limitation*: The available excerpt mentions the parallel execution of multiple attention operations but does not provide a detailed component diagram or exact dimensionality rules.

---

## 5. Core Q‑K‑V Computation

1. **Query vector** – Defined as a vector \(q \in \mathbb{R}^{d_q}\) extracted from the current processing step [1].
2. **Linear projections** – Learned matrices \(W^{Q}, W^{K}, W^{V}\) map input features to query, key, and value spaces (as described in the generic attention module) [5].
3. **Similarity function** – Either additive (tanh‑based) [2] or multiplicative (dot product) [3].
4. **Normalisation** – Softmax over the similarity scores yields attention weights \(\alpha_i\) [5].
5. **Output** – Weighted sum of value vectors:
\[
\text{output} = \sum_i \alpha_i \, v_i
\]
(same equation as in Section 1) [5].

---

## 6. Main Variants

| Variant | Description | Evidence |
|---------|-------------|----------|
| **Additive (Bahdanau) attention** | Computes alignment scores via a small feed‑forward network with a **tanh** activation before softmax [2] | [2] |
| **Multiplicative (dot‑product) attention** | Scores are simple dot products between query and key [3] | [3] |
| **Soft attention** | Uses a softmax to produce a dense distribution over all positions [5] | [5] |
| **Hard attention** | Selects a single position (non‑differentiable); not covered by the retrieved evidence | — |
| **Self‑attention** | Queries, keys, values come from the same sequence (Transformer) [8] | [8] |
| **Cross‑attention** | Queries come from one sequence (e.g., decoder) while keys/values come from another (e.g., encoder); concept implied by encoder‑decoder alignment but explicit formulation missing | — |

---

## 7. Applications in NLP and Vision

- **NLP** – Self‑attention is the backbone of large pre‑trained language models such as **BERT** (Devlin et al., 2019) and the **GPT** series (Radford et al., 2018‑2020) [12].
- **Vision** – The **Vision Transformer (ViT)** adapts the same self‑attention mechanism to image patches, achieving competitive performance on ImageNet and other benchmarks [12].

These models demonstrate that attention can replace recurrence and convolution entirely for a wide range of tasks.

---

## 8. Limitations & Efficiency Challenges

- **Quadratic complexity** – Naïve self‑attention requires computing pairwise interactions between all positions, leading to \(O(N^2)\) time and memory where \(N\) is sequence length. This becomes prohibitive for long sequences or high‑resolution images [12].
- **Memory‑intensive Q‑K‑V tensors** – Storing the full query‑key‑value matrices can exceed hardware limits, motivating research into efficient variants (e.g., sparse, low‑rank, or linear‑complexity attention) [10].

---

## References

- [1] https://arxiv.org/pdf/2203.14263
- [2] https://www.ibm.com/think/topics/attention-mechanism
- [3] https://vaultfs.io/global-self-attention-for-transformers-mathematical-explanation-how-is-it-different-from-spatio-temporal-attention
- [5] https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
- [8] https://arxiv.org/pdf/1706.03762
- [10] https://mbrenndoerfer.com/writing/attention-complexity-quadratic-scaling-memory-efficient-transformers
- [11] https://arxiv.org/pdf/1409.0473
- [12] https://arxiv.org/pdf/2010.11929
