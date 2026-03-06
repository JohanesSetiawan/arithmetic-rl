# Arithmetic RL — Pure GRPO from Scratch

> A decoder-only transformer (~1M params) trained to learn arithmetic **purely through reinforcement learning** — no supervised pretraining, no labelled answers during training. The model receives only a scalar reward signal and must discover how to compute $a \odot b$ from scratch.

---

## Table of Contents

- [Arithmetic RL — Pure GRPO from Scratch](#arithmetic-rl--pure-grpo-from-scratch)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Architecture](#architecture)
    - [Hyperparameters](#hyperparameters)
    - [Token Embedding](#token-embedding)
    - [Rotary Positional Embedding (RoPE)](#rotary-positional-embedding-rope)
    - [Causal Self-Attention](#causal-self-attention)
    - [SwiGLU Feed-Forward Network](#swiglu-feed-forward-network)
    - [Weight Tying](#weight-tying)
  - [GRPO Algorithm](#grpo-algorithm)
    - [Motivation](#motivation)
    - [Group-Relative Advantage](#group-relative-advantage)
    - [Clipped Surrogate Objective](#clipped-surrogate-objective)
    - [KL Divergence Penalty](#kl-divergence-penalty)
    - [Full Objective](#full-objective)
  - [Reward Design](#reward-design)
  - [Curriculum Learning](#curriculum-learning)
  - [Training Loop](#training-loop)
  - [Anti-Reward-Hacking Mechanisms](#anti-reward-hacking-mechanisms)
  - [References](#references)

---

## Overview

| Property        | Value                                                          |
| --------------- | -------------------------------------------------------------- |
| Model size      | ~1.05M parameters                                              |
| Vocabulary      | 20 tokens (character-level: digits, operators, punctuation)    |
| Task            | Integer arithmetic: $a\ \{+,-,\times,\div\}\ b = ?$            |
| Operand range   | 1–3 digits each                                                |
| RL algorithm    | GRPO (Group Relative Policy Optimization)                      |
| Pretraining     | None — random initialization                                   |
| Training signal | Scalar reward only (no teacher forcing, no cross-entropy loss) |

The model is a standard autoregressive language model. Given a question like `47 + 83 = ?`, it generates an answer token-by-token. The only learning signal is whether its answer is numerically correct.

---

## Architecture

```
Input IDs  [B, T]
    │
    ▼
Embedding  [B, T, 128]
    │
    ├─► RoPE  →  (cos, sin)  [T, 32]
    │
    ▼
DecoderBlock × 4
    │  ┌───────────────────────────────┐
    │  │  RMSNorm                      │
    │  │     └─► CausalSelfAttention   │
    │  │           (RoPE · fused QKV)  │
    │  │  + residual                   │
    │  │                               │
    │  │  RMSNorm                      │
    │  │     └─► SwiGLUFFN             │
    │  │  + residual                   │
    │  └───────────────────────────────┘
    │
    ▼
RMSNorm  [B, T, 128]
    │
    ▼
LM Head  [B, T, 20]   ← weight-tied with Embedding
```

### Hyperparameters

| Symbol           | Name                           | Value |
| ---------------- | ------------------------------ | ----- |
| $V$              | Vocab size                     | 20    |
| $d$              | Model dim ($d_{\text{model}}$) | 128   |
| $H$              | Attention heads                | 4     |
| $d_h$            | Head dim ($d / H$)             | 32    |
| $L$              | Transformer layers             | 4     |
| $d_{\text{ffn}}$ | FFN inner dim                  | 512   |

Total parameters: $\approx 1.05 \times 10^6$

---

### Token Embedding

Each input token $x_t \in \{0, \ldots, V-1\}$ is mapped to a dense vector:

$$\mathbf{e}_t = \mathbf{W}_E[x_t] \in \mathbb{R}^d$$

where $\mathbf{W}_E \in \mathbb{R}^{V \times d}$ is initialized with $\mathcal{N}(0, 0.02^2)$.

There is **no additive positional embedding** — position information is injected via RoPE directly into the attention mechanism.

---

### Rotary Positional Embedding (RoPE)

RoPE (Su et al., 2021) encodes position by **rotating** query and key vectors in 2-D sub-spaces. This gives a natural relative-distance bias and requires no fixed maximum sequence length.

**Inverse frequencies:**

$$\theta_i = \frac{1}{10000^{2i / d_h}}, \quad i = 0, 1, \ldots, \frac{d_h}{2} - 1$$

**Rotation matrix for position $m$, dimension pair $(i, i + d_h/2)$:**

$$R_m^{(i)} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$

**Applied to queries and keys:**

$$\tilde{\mathbf{q}}_{m,h} = \mathbf{q}_{m,h} \odot \cos(m\boldsymbol{\theta}) + \text{rotate}\_\text{half}(\mathbf{q}_{m,h}) \odot \sin(m\boldsymbol{\theta})$$

$$\tilde{\mathbf{k}}_{m,h} = \mathbf{k}_{m,h} \odot \cos(m\boldsymbol{\theta}) + \text{rotate}\_\text{half}(\mathbf{k}_{m,h}) \odot \sin(m\boldsymbol{\theta})$$

where $\text{rotate}\_\text{half}(\mathbf{x}) = [-x_2 \| x_1]$ splits and negates the second half.

**Key property:** The inner product between a query at position $m$ and a key at position $n$ depends only on their relative displacement $m - n$:

$$\langle \tilde{\mathbf{q}}_m, \tilde{\mathbf{k}}_n \rangle = f(\mathbf{q}_m, \mathbf{k}_n, m - n)$$

---

### Causal Self-Attention

**Fused QKV projection** — a single matrix multiplication produces all three:

$$[\mathbf{Q}\ |\ \mathbf{K}\ |\ \mathbf{V}] = \mathbf{X}\mathbf{W}_{QKV}, \quad \mathbf{W}_{QKV} \in \mathbb{R}^{d \times 3d}$$

**Scaled dot-product attention** with causal mask:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\tilde{\mathbf{Q}}\tilde{\mathbf{K}}^\top}{\sqrt{d_h}} + \mathbf{M}\right)\mathbf{V}$$

where $\mathbf{M}_{ij} = -\infty$ if $j > i$ (future tokens), else $0$.

Dispatched to **FlashAttention** on CUDA or an optimized memory-efficient kernel on CPU via `torch.nn.functional.scaled_dot_product_attention`.

---

### SwiGLU Feed-Forward Network

Proposed by Shazeer (2020) and adopted in PaLM, Llama, and most modern LLMs.

$$\text{SwiGLU}(\mathbf{x}) = \mathbf{W}_{\text{down}}\left(\text{SiLU}(\mathbf{W}_{\text{gate}}\mathbf{x}) \odot \mathbf{W}_{\text{up}}\mathbf{x}\right)$$

where $\text{SiLU}(z) = z \cdot \sigma(z)$ (Sigmoid Linear Unit).

Compared to a standard ReLU FFN with the same parameter budget, SwiGLU empirically achieves lower perplexity. The gating mechanism $\text{SiLU}(\mathbf{W}_{\text{gate}}\mathbf{x})$ acts as a learned information filter before the up-projection.

---

### Weight Tying

The LM head and the token embedding share the same weight matrix:

$$\mathbf{W}_{\text{LM}} = \mathbf{W}_E^\top$$

This reduces the parameter count and, more importantly, enforces consistency: the model must use the same geometric representation for input tokens and output token predictions.

---

## GRPO Algorithm

### Motivation

Standard policy gradient methods like REINFORCE suffer from high variance. PPO addresses this with a value function (critic), but training a separate critic adds cost and complexity. GRPO (Shao et al., 2024) achieves variance reduction without a critic by using **group-relative baselines**: the baseline for a question $q$ is estimated directly from the rewards of $G$ rollouts sampled from the same question.

---

### Group-Relative Advantage

For each question $q$, sample $G$ completions $\{o_1, \ldots, o_G\}$ from the old policy $\pi_{\theta_{\text{old}}}$ and compute their rewards $\{r_1, \ldots, r_G\}$.

The group-normalised advantage of completion $i$ is:

$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \varepsilon}$$

where:

$$\mu_r = \frac{1}{G}\sum_{j=1}^{G} r_j, \qquad \sigma_r = \sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j - \mu_r)^2}, \qquad \varepsilon = 10^{-8}$$

**Special case:** If all $G$ completions receive identical rewards ($\sigma_r \approx 0$), then $\hat{A}_i \approx 0$ for all $i$. This is correct — there is no relative preference among rollouts, so no gradient update should occur.

---

### Clipped Surrogate Objective

Identical to PPO-Clip (Schulman et al., 2017). For each answer token at position $t$ in completion $i$:

$$\rho_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,\lt t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,\lt t})} = \exp\!\left(\log\pi_\theta - \log\pi_{\theta_{\text{old}}}\right)$$

$$\mathcal{L}_{\text{clip}}^{(i,t)} = -\min\!\left(\rho_{i,t}\,\hat{A}_i,\ \text{clip}(\rho_{i,t},\, 1-\varepsilon,\, 1+\varepsilon)\,\hat{A}_i\right)$$

The clip prevents excessively large policy updates when the advantage is positive (exploitation), while still allowing the policy to move in the correct direction.

---

### KL Divergence Penalty

To prevent the policy from drifting too far from a fixed reference policy $\pi_{\text{ref}}$ (frozen at initialization), a KL penalty is added per token.

**Reference model:** A frozen deepcopy of the randomly-initialized model. Since there is no pretraining, this anchors the policy to its own starting distribution — preventing collapse to degenerate strategies.

**KL estimator:** Rather than computing the exact KL (which requires a full sum over the vocabulary), we use the **k3 unbiased estimator** (Schulman, 2020):

$$\text{KL}(\pi_\theta \| \pi_{\text{ref}}) \approx \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$$

Let $r = \pi_{\text{ref}} / \pi_\theta = \exp(\log\pi_{\text{ref}} - \log\pi_\theta)$. Then:

$$\widehat{\text{KL}}_{i,t} = r_{i,t} - \log r_{i,t} - 1$$

Properties:
- Always $\geq 0$ (by convexity of $f(x) = x - \log x - 1$)
- Equals $0$ iff $\pi_\theta = \pi_{\text{ref}}$
- Unbiased: $\mathbb{E}[\widehat{\text{KL}}] = \text{KL}(\pi_\theta \| \pi_{\text{ref}})$
- No extra forward pass needed

---

### Full Objective

Combining everything (DeepSeekMath, Eq. 3):

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{\substack{q \sim p(Q) \\ \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(\cdot\mid q)}}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left(\mathcal{L}_{\text{clip}}^{(i,t)} - \beta\,\widehat{\text{KL}}_{i,t}\right)\right]$$

Where:
- $|o_i|$ — length of completion $i$ (normalises for sequence length differences)
- $\beta$ — KL penalty coefficient (default: $0.01$)
- $\varepsilon$ — PPO clipping range (default: $0.2$)

The masked mean over real answer tokens (excluding padding) is used in practice:

$$\mathcal{L} = \frac{1}{G}\sum_{i=1}^G \frac{\sum_t \mathbb{1}[t \in \text{answer}] \cdot \ell_{i,t}}{\sum_t \mathbb{1}[t \in \text{answer}]}$$

---

## Reward Design

The reward function is deliberately simple — no partial credit, no shaped rewards, no learned reward model.

$$r(o, q) = \begin{cases} 1.0 & \text{if } \text{parse}(o) = \text{answer}(q) \quad \text{(exact match)} \\ 0.1 & \text{if } \text{parse}(o) \neq \bot \text{ and } \text{parse}(o) \neq \text{answer}(q) \quad \text{(valid integer)} \\ 0.0 & \text{if } \text{parse}(o) = \bot \quad \text{(unparseable)} \end{cases}$$

**Why a format reward?**

With a randomly initialized model, the probability of generating a correct $k$-digit answer in a single token step is roughly:

$$P(\text{correct}) \approx \left(\frac{1}{V}\right)^k \approx \left(\frac{1}{20}\right)^k$$

For $k = 1$, this is $5\%$. For a group of $G = 16$ rollouts, the expected number of correct answers is $\approx 0.8$ — and when they all happen to be 0, the group advantage is identically 0 and no gradient signal exists.

The format reward ($r = 0.1$) gives a non-zero reward for generating any valid integer, bootstrapping early learning. This mirrors the format + accuracy reward strategy used in DeepSeek-R1-Zero.

Once the model consistently generates valid integers, the accuracy reward ($r = 1.0$) dominates and drives correctness.

---

## Curriculum Learning

Training directly on the full dataset (answers ranging from single-digit to 6-digit numbers) is too hard from a random start. We use a 3-stage curriculum based on **answer complexity** (digit count of the result).

> **Note on design choice:** The dataset uses 1–3 digit operands, but due to multiplication, answers can reach 6 digits. Staging by *operand* digit count would leave Stage 1 nearly empty. Staging by *answer* digit count correctly captures task difficulty.

| Stage | Training pool                    | Advance condition        |
| ----- | -------------------------------- | ------------------------ |
| 1     | Answers with 1 digit             | Val accuracy $\geq 70\%$ |
| 2     | Answers with 1–2 digits          | Val accuracy $\geq 70\%$ |
| 3     | Full dataset (1–6 digit answers) | —                        |

**Data splits** (applied per stage):

- Train: 80% of dataset
- Validation: 10% — used to measure stage advancement
- Held-out: 10% — reserved for final evaluation

Stage advancement is evaluated every 50 steps using greedy decoding on a sample of 200 validation questions.

---

## Training Loop

```
Initialize:
  π_θ         ← random init  (ArithmeticTransformer)
  π_ref       ← deepcopy(π_θ), frozen forever
  curriculum  ← Stage 1

For step = 1 … T:
  ┌─ ROLLOUT PHASE (no_grad) ──────────────────────────────────────────────┐
  │  questions ← curriculum.sample(Q questions)                            │
  │  for each question q with answer a*:                                   │
  │    for i = 1 … G:                                                      │
  │      o_i, log π_θ_old(o_i) ← sample from π_θ with temperature τ        │
  │      r_i ← reward(o_i, a*)                                             │
  │      log π_ref(o_i) ← forward pass through π_ref                       │
  │    Â ← group_advantages(r_1, …, r_G)                                   │
  │    pack into Experience                                                │
  └────────────────────────────────────────────────────────────────────────┘

  ┌─ UPDATE PHASE (with grad) ─────────────────────────────────────────────┐
  │  optimizer.zero_grad()                                                 │
  │  for each Experience:                                                  │
  │    log π_θ(o_i) ← forward pass through π_θ  (answer tokens only)       │
  │    loss, kl ← GRPO_Loss(log π_θ, log π_θ_old, log π_ref, Â)            │
  │    loss.backward()                                                     │
  │  clip_grad_norm_(π_θ, max_norm=1.0)                                    │
  │  optimizer.step()                                                      │
  └────────────────────────────────────────────────────────────────────────┘

  every 50 steps:
    val_acc ← greedy_evaluate(π_θ, val_pool)
    if val_acc ≥ threshold: advance curriculum stage

  every 200 steps:
    save checkpoint
```

**Optimizer:** AdamW with $\text{lr} = 5 \times 10^{-5}$, $\lambda = 0.01$

**Key implementation detail — log-prob indexing:**

During the update, we recompute $\log\pi_\theta(o_{i,t} \mid q, o_{i,\lt t})$ via a single batched forward pass. Given logits $\mathbf{Z} \in \mathbb{R}^{G \times T \times V}$:

$$\log\pi_\theta(o_{i,t}) = \log\text{softmax}(\mathbf{Z}_{i,\, p+t-1,\, :})[o_{i,t}]$$

where $p = \text{prompt}\_\text{len}$. The offset $p - 1$ is critical: $\mathbf{Z}_{i, j, :}$ predicts token at position $j + 1$, so the first answer token at position $p$ is predicted by $\mathbf{Z}_{i, p-1, :}$.

---

## Anti-Reward-Hacking Mechanisms

| Mechanism             | Implementation                                                                        | Purpose                                           |
| --------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------- |
| Exact match reward    | Binary $\{0, 1\}$ for correctness, no partial credit                                  | Prevents gaming via near-misses                   |
| KL penalty            | $\beta \cdot \widehat{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ with $\beta = 0.01$ | Prevents over-optimization                        |
| Format reward cap     | $r_{\text{format}} = 0.1 \ll r_{\text{accuracy}} = 1.0$                               | Discourages exploiting format without correctness |
| Curriculum validation | Stage advancement uses a held-out val split                                           | Prevents memorization of training questions       |
| Gradient clipping     | $\|\nabla\theta\|_2 \leq 1.0$                                                         | Prevents instability from large policy updates    |

---

## References

|                                                                                                                                        |                                                                  |
| -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **[1]** Shao, Z. et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300 | GRPO algorithm (Section 3)                                       |
| **[2]** DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:2501.12948     | Pure RL training, format+accuracy reward, "aha moment" emergence |
| **[3]** Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864                        | RoPE positional encoding                                         |
| **[4]** Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347                                        | PPO clipped surrogate objective                                  |
| **[5]** Schulman, J. (2020). *Approximating KL Divergence.* http://joschu.net/blog/kl-approx.html                                      | k3 unbiased KL estimator                                         |
| **[6]** Shazeer, N. (2020). *GLU Variants Improve Transformer.* arXiv:2002.05202                                                       | SwiGLU feed-forward network                                      |
| **[7]** open-thought. *tiny-grpo.* https://github.com/open-thought/tiny-grpo                                                           | Reference GRPO implementation                                    |