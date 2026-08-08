# Attention Mechanism – A Technical Deep‑Dive

---

## Executive Summary
The attention mechanism is a core building block of modern neural networks, especially in natural‑language processing. It maps a **query** vector together with a set of **key‑value** pairs to an output vector that is a weighted sum of the values, where the weights reflect similarity between query and keys. The idea originated in neural machine translation (Bahdanau *et al.*, 2014) as an **additive (concat) attention** function. Subsequent work, most notably the Transformer architecture, generalized attention to **scaled‑dot‑product**, **self‑attention**, and **multi‑head attention**. Evidence confirms the definitions, the original additive formulation, and the architectural variants, while concrete formulae for scaled‑dot‑product attention and detailed application examples are not present in the supplied excerpts. Likewise, systematic discussion of computational or interpretability limitations is absent from the provided material.

---

## 1. Definition and High‑Level Overview

An **attention function** takes three inputs:

* a **query** vector **q**,
* a set of **key** vectors **{k₁,…,kₙ}**, and
* the corresponding **value** vectors **{v₁,…,vₙ}**.

It produces an output **o** that is a weighted sum of the values:

\[
o = \sum_{i=1}^{n} \alpha_i \, v_i,
\]

where the attention weights **αᵢ** are derived from a similarity measure between the query and each key and are normalised (e.g., with softmax) so that they sum to one. This formulation is described in the Transformer paper and summarised in the literature [4][5].

---

## 2. Historical Origin – Additive (Concat) Attention

The first widely cited attention mechanism appeared in the neural machine translation model of Bahdanau *et al.* (2014). Their **additive attention** (also called “concat” attention) computes an alignment score between the decoder hidden state **sₜ** and each encoder hidden state **hᵢ** as

\[
\text{score}(s_t, h_i) = v_{\alpha}^{\top}\,\tanh\!\bigl(W_{\alpha}\,[s_t;h_i]\bigr),
\]

where **[ ; ]** denotes vector concatenation, **Wₐ** and **vₐ** are learnable parameters, and **tanh** provides non‑linearity. The scores are turned into normalised weights that weight the encoder values, yielding the context vector for the decoder. This exact formulation is provided in the seminar notes [1].

---

## 3. Scaled‑Dot‑Product Attention (Transformer)

The Transformer architecture introduced **scaled‑dot‑product attention**, which replaces the additive score with a dot product between queries and keys, scaled by the square root of the key dimension. The retrieved excerpts refer to “Scaled Dot‑Product Attention” but do **not** contain the explicit mathematical expression. Consequently, the report can only state that the mechanism computes similarity via a dot product, applies a scaling factor, passes the result through a softmax to obtain weights, and finally multiplies the weights by the values [4][5]. The precise formula is not reproduced here because it is absent from the supplied evidence.

---

## 4. Self‑Attention

**Self‑attention** applies the attention operation within a single sequence: each token’s query attends to all keys (including its own) derived from the same sequence. This enables every position to incorporate information from every other position, facilitating rich contextual representations. The concept is described in the Transformer paper and illustrated in the “self‑attention sub‑layer” discussion [4] and visualised in the Illustrated Transformer [8].

---

## 5. Multi‑Head Attention

**Multi‑head attention** extends the basic attention block by performing several attention operations (heads) in parallel, each with its own learned linear projections of the queries, keys, and values. The per‑head outputs (each of dimension *dᵥ*) are concatenated and passed through a final linear projection to produce the overall result. This design allows the model to capture information from different representation subspaces at different positions. The mechanism is explained in the seminar material, which notes the projection of Q/K/V *h* times and the subsequent concatenation [1]; the Illustrated Transformer also discusses the “many heads” architecture [8].

---

## 6. Applications

The evidence notes that attention mechanisms are employed across a broad range of tasks in natural‑language processing and beyond, and that dedicated “Applications” sections exist in the literature [5]. However, the retrieved excerpts do not enumerate specific use‑cases (e.g., machine translation, summarisation, image captioning, protein folding). Therefore, while it is clear that attention is widely applied, concrete examples cannot be listed without additional sources.

---

## 7. Known Limitations

The supplied material does not discuss the computational cost of attention (which grows quadratically with sequence length) nor interpretability challenges associated with attention weight analysis. Consequently, this report cannot provide evidence‑based statements on these limitations.

---

## References

- **[1]** Attention and Self‑Attention for NLP – seminar notes, LMU. https://slds-lmu.github.io/seminar_nlp_ss20/attention-and-self-attention-for-nlp.html
- **[4]** “Attention Is All You Need”, Vaswani *et al.*, 2017. https://arxiv.org/pdf/1706.03762
- **[5]** Wikipedia article “Attention (machine learning)”. https://en.wikipedia.org/wiki/Attention_(machine_learning)
- **[8]** The Illustrated Transformer – Jay Alammar. https://jalammar.github.io/illustrated-transformer
