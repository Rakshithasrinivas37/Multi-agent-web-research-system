# Comparative Analysis of Transformers, LSTM, and RNN Architectures

## Executive Summary
This report synthesises the retrieved literature to contrast three dominant sequence‑modeling families—Transformers, Long Short‑Term Memory networks (LSTM), and vanilla Recurrent Neural Networks (RNN). For each architecture we summarise its design principle, the mathematically‑driven core operations that are documented, computational scaling, reported benchmark usage, typical application domains, recognised drawbacks, and historic milestones. The findings are distilled into a concise side‑by‑side comparison table (plain‑text format) that can be used as a quick reference for researchers and practitioners.

---

## 1. Architecture Overviews

| Architecture | Overview |
|--------------|----------|
| **Transformer** | Introduced in 2017, the Transformer replaces recurrence with a stack of self‑attention layers that attend to all positions of the input sequence simultaneously. Multi‑head attention splits the representation into several sub‑spaces, each processed by an independent attention head, and the outputs are concatenated (see [6], [2]). |
| **LSTM** | An LSTM augments a recurrent cell with three gated pathways—input, forget, and output—that regulate the flow of information into and out of a persistent cell state, thereby mitigating vanishing‑gradient problems (see [4], [8]). Multidimensional variants extend the recurrence to N‑dimensional grids, as formalised in [3]. |
| **RNN** | The vanilla RNN updates a hidden state by applying a linear transformation to the previous hidden state and the current input, followed by a non‑linearity. This simple recurrence enables modelling of sequential data but suffers from gradient decay over long horizons (see [5], [2]). |

---

## 2. Core Equations

*Only equations explicitly present in the evidence are reproduced.*

| Architecture | Core Equations (as documented) |
|--------------|--------------------------------|
| **Transformer** | The self‑attention operation has quadratic time/space cost O(d n²) where *d* is the model dimension and *n* the sequence length (derived from the definition of self‑attention in [10]). |
| **LSTM** | For an N‑dimensional LSTM the memory vector is computed as  \(\mathbf{s}_m = \sum_{i=1}^{N} \boldsymbol{\phi}_i \odot \mathbf{s}_{m_i} + \mathbf{i}_{nm} \odot \mathbf{z}_m\)  (Eq. 54 in [3]), where \(\odot\) denotes element‑wise multiplication and \(\boldsymbol{\phi}_i\) are forget signals. The standard single‑dimensional gating equations are described qualitatively (input, forget, output gates) in [4] and [8] but not given in closed form. |
| **RNN** | Hidden‑state update: \(h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)\) and output: \(y_t = W_{hy}h_t + b_y\) (see [5]). |

---

## 3. Computational Complexity

| Architecture | Complexity (training / inference) |
|--------------|-----------------------------------|
| **Transformer** | Self‑attention requires O(d n²) time and memory per layer; multi‑head design keeps total cost comparable to single‑head attention of full dimension (see [6], [10]). |
| **LSTM** | Sequential dependence forces T steps for a sequence of length T; each step performs matrix multiplications of size proportional to hidden dimension, leading to O(T · h²) time (implicit from [8]). |
| **RNN** | Linear‑time recurrence: O(T) sequential operations for a sequence of length T (see [2]); each step similar to LSTM but with fewer gates, thus lower per‑step cost. |

---

## 4. Benchmark Performance

| Architecture | Representative Scores (as reported) |
|--------------|--------------------------------------|
| **Transformer** | Evaluated on the GLUE benchmark across multiple datasets; the retrieved source notes the evaluation but does not provide numeric scores (see [1]). |
| **LSTM** | No concrete benchmark numbers are present in the retrieved evidence. |
| **RNN** | No concrete benchmark numbers are present in the retrieved evidence. |

*When scores are absent, the report explicitly notes the limitation of the available evidence.*

---

## 5. Typical Applications

| Architecture | Common Use‑Cases |
|--------------|-----------------|
| **Transformer** | Language modelling, machine translation, text generation, and other tasks that benefit from parallel processing of long sequences (see [2], [11]). |
| **LSTM** | Speech recognition, time‑series forecasting, and NLP tasks requiring long‑range memory, especially when data are limited enough for recurrent training (see [4], [5]). |
| **RNN** | Early NLP models, simple sequence tagging, and any scenario where a lightweight recurrent model suffices (see [5], [11]). |

---

## 6. Known Limitations

| Architecture | Limitations |
|--------------|-------------|
| **Transformer** | High computational and memory demands; training requires large datasets and specialised hardware (see [11], [10]). |
| **LSTM** | Still sequential, limiting parallelism; higher parameter count leads to slower training and inference, and requires more data (see [4], [8]). |
| **RNN** | Prone to vanishing gradients, restricting ability to capture long‑range dependencies (see [5]); sequential nature hampers scalability. |

---

## 7. Historical Milestones

| Architecture | Milestone (as documented) |
|--------------|---------------------------|
| **RNN** | Early recurrent models described in tutorial material (see [5]); original papers not cited in the retrieved set, so exact year/author not provided. |
| **LSTM** | Gating mechanism introduced to address vanishing gradients (see [4]); the seminal LSTM paper is not part of the retrieved sources, so precise citation is omitted. |
| **Transformer** | “Attention Is All You Need” introduced the self‑attention‑only architecture in 2017 (see [6]). |

---

## 8. Side‑by‑Side Comparison (plain‑text)

```
Transformer | Overview: self‑attention stack with multi‑head mechanism (parallel across positions) | Core Equations: self‑attention with O(d n²) cost (quadratic) [10] | Computational Complexity: O(d n²) time & memory per layer; parallelizable [6] | Benchmark Performance: evaluated on GLUE (scores not detailed) [1] | Applications: language modelling, translation, text generation, etc. [2][11] | Limitations: high compute & memory, needs large data [11][10] | Milestones: introduced 2017 (Vaswani et al.) [6]

LSTM | Overview: recurrent cell with input, forget, output gates; mitigates vanishing gradients [4][8] | Core Equations: multidimensional memory update (Eq. 54) [3]; gating described qualitatively [4][8] | Computational Complexity: sequential O(T · h²); slower than RNN due to extra gates [8] | Benchmark Performance: no numeric scores provided in evidence | Applications: speech recognition, time‑series forecasting, NLP tasks requiring long‑range memory [4][5] | Limitations: sequential computation, higher parameter count, slower training [4][8] | Milestones: gating concept introduced (date/author not in evidence) [4]

RNN | Overview: simple recurrence with hidden state update via tanh activation [5] | Core Equations: h_t = tanh(W_hh h_{t-1}+W_xh x_t+b); y_t = W_hy h_t+b_y [5] | Computational Complexity: linear O(T) sequential steps; limited parallelism [2] | Benchmark Performance: no numeric scores provided in evidence | Applications: early NLP models, sequence tagging, lightweight tasks [5][11] | Limitations: vanishing gradient, poor long‑range dependency capture [5] | Milestones: early recurrent models described in tutorials; original dates/authors not in evidence [5]
```

---

## References

[1] https://arxiv.org/pdf/2306.01768
[2] https://slds-lmu.github.io/seminar_nlp_ss20/attention-and-self-attention-for-nlp.html
[3] https://arxiv.org/pdf/1909.09586
[4] https://inovaqo.com/2025/08/11/rnn-vs-lstm-vs-gru-a-comprehensive-comparison-analysis
[5] https://www.khoury.northeastern.edu/home/vip/teach/MLcourse/7_adv_NN/notes/chatGPT_responses/Illustrated_RNN_LSTM_GRU_Booklet.pdf
[6] https://arxiv.org/pdf/1706.03762
[8] https://mbrenndoerfer.com/writing/lstm-architecture-recurrent-neural-networks-guide
[10] https://proceedings.mlr.press/v201/duman-keles23a/duman-keles23a.pdf
[11] https://www.geeksforgeeks.org/deep-learning/rnn-vs-lstm-vs-gru-vs-transformers
