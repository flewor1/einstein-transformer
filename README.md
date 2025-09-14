# Decoder-Only Transformers in Einstein Notation

<div align="center">

[![PDF](https://img.shields.io/badge/PDF-Available-red.svg?style=for-the-badge&logo=adobe-acrobat-reader)](main.pdf)
[![LaTeX](https://img.shields.io/badge/LaTeX-Source-green.svg?style=for-the-badge&logo=latex)](main.tex)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/flewor1/einstein-transformer?style=for-the-badge&logo=github)](https://github.com/flewor1/einstein-transformer/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/flewor1/einstein-transformer?style=for-the-badge&logo=github)](https://github.com/flewor1/einstein-transformer/issues)

**A comprehensive mathematical exposition of modern transformer architectures using Einstein summation notation**

*Rigorous tensor formulations • Clear index semantics • Modern architectural variants*

</div>

---

## 📑 Table of Contents

- [🎯 Overview](#-overview)
- [📋 Abstract](#-abstract)
- [📚 Mathematical Contents](#-mathematical-contents)
- [🔧 Quick Start](#-quick-start)
- [📖 Main Content](#-main-content)
  - [Conventions and Einstein Primer](#conventions-and-einstein-primer)
  - [Pre-Norm Decoder Block](#pre-norm-decoder-block)
  - [Positional Encoding Schemes](#positional-schemes)
  - [KV Caching and Long Context](#kv-caching-and-long-context)
  - [FlashAttention](#flashattention-exact-tiled-online-softmax)
  - [Advanced Architectures](#gqa-mla-moe-mtp)
  - [Implementation Details](#implementation-details)
- [📁 Repository Structure](#-repository-structure)
- [🔬 Mathematical Notation Guidelines](#-mathematical-notation-guidelines)
- [💡 Usage Examples](#-usage-examples)
- [🤝 Contributing](#-contributing)
- [📄 Citation](#-citation)
- [📚 Related Work](#-related-work)
- [⚖️ License](#️-license)

---

## 🎯 Overview

This repository presents a **mathematically rigorous framework** for understanding decoder-only transformer architectures through the lens of **Einstein summation notation**. Unlike conventional presentations that often rely on implicit broadcasting and ambiguous index semantics, this approach provides **crystal-clear tensor operation definitions** that eliminate ambiguity in implementation and theoretical analysis.

### Key Features

- 🎯 **Unambiguous Index Semantics**: Every tensor operation is precisely defined with explicit indices
- 🔄 **Complete Architecture Coverage**: From basic attention to advanced variants (GQA, MLA, MoE)
- ⚡ **Modern Optimizations**: FlashAttention, KV caching, mixed precision training
- 📐 **Positional Encoding**: Comprehensive coverage of RoPE, ALiBi, and YaRN scaling
- 🧮 **Mathematical Rigor**: Lemmas and proofs for key architectural properties

## 📋 Abstract

This document provides a rigorous mathematical framework for understanding decoder-only transformer architectures through Einstein notation. It covers essential components including multi-head attention, positional encoding schemes (RoPE, ALiBi), advanced optimization techniques (FlashAttention, KV caching), and modern architectural variants (GQA, MLA, MoE). The Einstein notation approach offers clear tensor operation semantics and eliminates ambiguity in index handling.

**Target Audience**: Researchers, engineers, and students seeking a deep mathematical understanding of transformer architectures with implementation-ready formulations.

## 📚 Mathematical Contents

<details>
<summary><strong>🔢 Core Components</strong></summary>

- **Einstein Summation Convention** with explicit temporal index handling
- **Pre-Norm Decoder Blocks** with complete tensor formulations  
- **Multi-Head Self-Attention** with rigorous index semantics
- **Feed-Forward Networks** with Einstein notation

</details>

<details>
<summary><strong>📍 Positional Encoding</strong></summary>

- **RoPE (Rotary Position Embedding)** with rotation matrices
- **ALiBi (Attention with Linear Biases)** score modifications
- **YaRN-Style Scaling** for context length extension

</details>

<details>
<summary><strong>⚡ Optimization Techniques</strong></summary>

- **FlashAttention** tiled online-softmax algorithm
- **KV Caching** for efficient autoregressive generation
- **Sliding Window** and **Paged Attention** for long contexts
- **FP8 Mixed Precision** training strategies

</details>

<details>
<summary><strong>🏗️ Advanced Architectures</strong></summary>

- **GQA (Grouped-Query Attention)** for efficient inference
- **MLA (Multi-Head Latent Attention)** with compression
- **MoE (Mixture of Experts)** with learned routing
- **MTP (Multi-Token Prediction)** extensions

</details>

---

## 🔧 Quick Start

### Prerequisites

- **LaTeX Distribution**: TeX Live, MiKTeX, or MacTeX
- **PDF Viewer**: Any modern PDF viewer
- **Mathematical Background**: Linear algebra, tensor operations
- **Optional**: Basic familiarity with transformer architectures

### Viewing Options

| Format | Description | Best For |
|--------|-------------|----------|
| 📖 **[GitHub README](README.md)** | Online with math rendering | Quick reference, sharing |
| 📄 **[PDF Document](main.pdf)** | High-quality typesetting | Printing, offline reading |
| 📝 **[LaTeX Source](main.tex)** | Editable source code | Customization, contribution |

### Compilation

```bash
# Clone the repository
git clone https://github.com/flewor1/einstein-transformer.git
cd einstein-transformer

# Compile LaTeX source
pdflatex main.tex

# Optional: Clean auxiliary files
rm *.aux *.log *.out
```

---

## 📖 Main Content

### Conventions and Einstein Primer

We use Einstein summation: an index repeated once up and once down in a term is summed. **Temporal indices $t,s,u$ are never implicitly summed**; we write $\sum$ or $\text{softmax}_s(\cdot)$ explicitly. Kronecker deltas always pair one up with one down. We use the trivial metric $\delta$ to raise/lower indices only when needed; e.g. $a_k := a^m \delta_{m k}$.

**Index sets.** $t,s\in\{1..T\}$ (time), $u\in\{1..U\}$ (encoder), $r\in\{1..R\}$ (request id), $f\in\{1..d_{\text{model}}\}$, $h\in\{1..H\}$, $d,k\in\{1..d_k\}$, $e\in\{1..d_{\text{ff}}\}$, $\rho\in\{1..r\}$ (latent for MLA), $x\in\{1..E\}$ (experts), $v\in\{1..V\}$ (vocab).

### Pre-Norm Decoder Block

Let $X_t{}^{f}$ be token embeddings. One layer (pre-norm): $\widetilde H_t{}^{f}=\text{LN}(H_t{}^{f})$; self-attention produces $A_t{}^{f}$; residual $H'_t{}^{f}=H_t{}^{f}+A_t{}^{f}$; then $\text{LN}$, FFN:

$$
U_t{}^{e}=H''_t{}^{f} W_{1 f}{}^{e},    
Z_t{}^{e}=\sigma(U_t{}^{e}),    
F_t{}^{f}=Z_t{}^{e} W_{2 e}{}^{f},    
H^{\text{out}}_t{}^{f}=H'_t{}^{f}+F_t{}^{f}.
$$

#### Multi-Head Self-Attention

Projections for head $h$:

$$
Q_{t h}{}^{d}=\widetilde H_t{}^{f} W_{Q h f}{}^{d},    
K_{s h}{}^{d}=\widetilde H_s{}^{f} W_{K h f}{}^{d},    
V_{s h}{}^{d}=\widetilde H_s{}^{f} W_{V h f}{}^{d}.
$$

RoPE rotation matrices have indices *exactly* $R_{t}{}_{d}{}^{k}$ so that

$$
\widehat Q_{t h}{}^{k}=Q_{t h}{}^{d}  R_{t}{}_{d}{}^{k},        
\widehat K_{s h}{}^{k}=K_{s h}{}^{d}  R_{s}{}_{d}{}^{k},
$$

(no implicit sums over $t$ or $s$). Causal logits (we lower $\widehat K$ using $\delta$; no implicit sum over $s$):

$$
L_{t s}^{(h)}=\frac{1}{\sqrt{d_k}}   \widehat Q_{t h}{}^{k}  \widehat K_{s h k},         \text{with } \widehat K_{s h k}:=\widehat K_{s h}{}^{m} \delta_{m k}.
$$

Masked to $s\le t$. With ALiBi, add $-m_h (t-s)$ inside the softmax. Weights:

$$
A_{t s}^{(h)}=\text{softmax}_{s}\big(L_{t s}^{(h)}\big).
$$

Head output and merge:

$$
Y_{t h}{}^{d}=\sum_{s\le t} A_{t s}^{(h)} V_{s h}{}^{d},        
A_t{}^{f}=Y_{t h}{}^{d} W_{O h d}{}^{f}.
$$

### Positional Schemes

#### RoPE (post-projection)
As above, RoPE uses $R_{t}{}_{d}{}^{k}$, yielding a dot-product depending on $(t-s)$ via relative phase.

#### ALiBi (score bias)
Add per-head linear bias:

$$
A_{t s}^{(h)}=\text{softmax}_{s}\!\left(\frac{1}{\sqrt{d_k}} Q_{t h}{}^{d}K_{s h d}-m_h (t-s)\right),     s\le t.
$$

### KV Caching and Long Context

#### KV Cache
At step $t$, reuse cached $K_{s h}{}^{d},V_{s h}{}^{d}$ for $s<t$; compute only $Q_{t h}{}^{d}$ (and $K_{t},V_{t}$) and append.

#### RoPE in cache: store rotated vs unrotated
- **Option A**: store $\widehat K_{s h}{}^{k}$ and use $\widehat Q_{t h}{}^{k}$ directly.
- **Option B**: store unrotated $K_{s h}{}^{d}$ and rotate on-the-fly: $\widehat K_{s h}{}^{k}=K_{s h}{}^{d}R_{s}{}_{d}{}^{k}$ (flexible for scaling, more compute).

#### Paged attention
Partition $\{1..t\}$ into pages of size $P$; attend within page (or with small overlap). Mask equivalent: $L_{t s}=-\infty$ if $s$ outside $t$'s page window.

#### Sliding window
Restrict to $s\in (t-W, \ldots, t]$ by masking $s<t-W$. Drop old cache entries beyond window if desired.

#### Continuous batching (request index)
Use $\delta_{r}{}^{r'}$ to prevent cross-request attention:

$$
L_{r t,  r' s}^{(h)}=\frac{1}{\sqrt{d_k}}  Q_{r t h}{}^{d} K_{r' s h d}  \delta_{r}{}^{r'}.
$$

Then $Y_{r t h}{}^{d}=\sum_{s} A_{r t,  r s}^{(h)} V_{r s h}{}^{d}$.

### FlashAttention: Exact Tiled Online-Softmax

For fixed $(t,h)$, iterate blocks $B$ over $s$. Maintain running max $m_{t}^{(h)}$, partition $z_{t}^{(h)}$, and numerator $N_{t h}{}^{d}$.

Initialize: $m=-\infty$, $z=0$, $N_{t h}{}^{d}=0$.

For a block $B\subseteq\{s\le t\}$, define per-$s$ logits
$\ell_s=\tfrac{1}{\sqrt{d_k}}\widehat Q_{t h}{}^{k}\widehat K_{s h k}+b_{t s}^{(h)}$ (bias includes mask/ALiBi if any; no implicit sum over $s$). Let $m_B=\max_{s\in B}\ell_s$ and $m'=\max(m,m_B)$. Then

$$
\alpha=\exp(m-m'),        
z\leftarrow \alpha  z + \sum_{s\in B}\exp(\ell_s-m'),        
N_{t h}{}^{d}\leftarrow \alpha  N_{t h}{}^{d} + \sum_{s\in B}\exp(\ell_s-m')  V_{s h}{}^{d},        
m\leftarrow m'.
$$

After all blocks, exact output:

$$
Y_{t h}{}^{d}=\frac{N_{t h}{}^{d}}{z}.
$$

This is numerically equivalent to full softmax but never materializes the $t\times t$ matrix.

### GQA, MLA, MoE, MTP

#### Grouped-Query Attention (GQA)
Let $g\in\{1..G\}$ index KV groups ($G<H$). Queries use $h$, K/V use $g$, with a mapping $\pi(h)$:

$$
L_{t s}^{(h)}=\tfrac{1}{\sqrt{d_k}}  Q_{t h}{}^{d} K_{s, \pi(h)}{}_{d},        
Y_{t h}{}^{d}=\sum_{s\le t} A_{t s}^{(h)} V_{s, \pi(h)}{}^{d}.
$$

#### Multi-Head Latent Attention (MLA)
Compress to latent $L_s{}^{\rho}=\widetilde H_s{}^{f} U_{f}{}^{\rho}$, then per-head expand:

$$
K_{s h}{}^{d}=L_s{}^{\rho} P_{h \rho}{}^{d},        
V_{s h}{}^{d}=L_s{}^{\rho} Q_{h \rho}{}^{d}.
$$

Logits become $\tfrac{1}{\sqrt{d_k}}  q_{t h}{}^{\rho} L_s{}^{\rho}$ with $q_{t h}{}^{\rho}=Q_{t h}{}^{d} P_{h \rho d}$; the weighted latent $z_{t h}{}^{\rho}=\sum_{s\le t}A_{t s}^{(h)}L_s{}^{\rho}$; output $Y_{t h}{}^{d}=z_{t h}{}^{\rho} Q_{h \rho}{}^{d}$.

#### MoE with learned router bias
Router logits: $G_t{}^{x}=H''_t{}^{f} W_{\text{gate} f}{}^{x}+b^{x}$; choose top-$k$ experts $\{x_i\}$ and weights $p_{t x_i}=\frac{e^{G_t{}^{x_i}}}{\sum_j e^{G_t{}^{x_j}}}$. Output $F_t{}^{f}=\sum_i p_{t x_i}  F_{(x_i),t}{}^{f}$.

#### Multi-Token Prediction (MTP)
Add $n$ vocab heads:

$$
O^{(j)}_{t}{}^{v}=H^{\text{final}}_{t}{}^{f} W_{O_j f}{}^{v},    
\mathcal{L}=\frac{1}{n}\sum_{j=1}^{n}\text{CE}\big(O^{(j)}_{t},  w_{t+j}\big).
$$

### Implementation Details

#### FP8 Mixed Precision and Pipeline Overlap
FP8 for matmuls with per-tensor scales; keep reductions (LayerNorm/softmax sums) higher precision. Pipeline: split layers across devices; overlap micro-batches to minimize bubble; recompute or checkpoint as needed.

#### YaRN-Style RoPE Scaling
For extension from $L$ to $L'$, scale angles per dimension: $\theta'_{d}(t)=\theta_{d}(t/\alpha)$ (static, $\alpha=L'/L$), or dynamic scale increasing with current length. Use $\tilde R_{t}{}_{d}{}^{k}$ in place of $R_{t}{}_{d}{}^{k}$. **Cache note:** if scale changes mid-generation, keep *unrotated* $K_{s h}{}^{d}$ to re-rotate with new $\tilde R_{s}$.

#### Lemma: Single Bilinear Collapse and Why RoPE/ALiBi Break It
**Lemma.** Without positional terms, if $W_Q^{(h)} (W_K^{(h)})^{\top}=M$ (same $M$ for all $h$), then all heads share logits
$L_{t s}^{(h)}=X_t{}^{f} M_{f g} X_s{}^{g}$ and the layer equals a single-head with attention weights from $M$ and a combined value-projection $U_{g}{}^{f}=\sum_h W_{V g}^{(h) d} W_{O h d}{}^{f}$.

**RoPE counterexample.** Effective bilinear becomes $M^{(h)}(t,s)=W_Q^{(h)} R_{t}^{\top} R_{s} (W_K^{(h)})^{\top}$ (depends on $t,s$). No single global $M$ matches all $(t,s)$.

**ALiBi counterexample.** Head-specific slopes $m_h$ yield $L_{t s}^{(h)}=X_t{}^{f} M_{f g} X_s{}^{g}-m_h (t-s)$; differing $m_h$ produce genuinely different $A^{(h)}$.

#### Index Sanity Checklist
- ✅ Every contraction pairs one up with one down (e.g.\ $Q_{t h}{}^{d}K_{s h d}$).
- ✅ Temporal indices $t,s,u$ are never implicitly summed; sums and $\text{softmax}_s$ are explicit.
- ✅ RoPE matrices use *exact* indices $R_{t}{}_{d}{}^{k}$; $\widehat Q_{t h}{}^{k}=Q_{t h}{}^{d} R_{t}{}_{d}{}^{k}$ and similarly for $\widehat K$.
- ✅ $\delta$ usage: $\delta_{r}{}^{r'}$ (one up, one down) gates cross-request terms; we also lower $\widehat K$ via $\delta$ in dot-products.
- ✅ Shapes check: $A_t{}^{f}=Y_{t h}{}^{d} W_{O h d}{}^{f}$ sums $(h,d)$ to return $f$.

### Appendix: Cross-Attention (Encoder--Decoder)

Encoder states $E_{u}{}^{f}$; cross-attn queries from decoder $\widetilde H_t{}^{f}$:

$$
Q^{\text{x}}_{t h}{}^{d}=\widetilde H_t{}^{f} W^{\text{x}}_{Q h f}{}^{d},    
K^{\text{x}}_{u h}{}^{d}=E_{u}{}^{f} W^{\text{x}}_{K h f}{}^{d},    
V^{\text{x}}_{u h}{}^{d}=E_{u}{}^{f} W^{\text{x}}_{V h f}{}^{d}.
$$

Optionally apply RoPE on ($t,u$) with $R_{t}{}_{d}{}^{k}, R_{u}{}_{d}{}^{k}$, then

$$
L^{\text{x}(h)}_{t u}=\tfrac{1}{\sqrt{d_k}}  \widehat Q^{\text{x}}_{t h}{}^{k}\widehat K^{\text{x}}_{u h k},    
A^{\text{x}(h)}_{t u}=\text{softmax}_{u}(L^{\text{x}(h)}_{t u}),    
Y^{\text{x}}_{t h}{}^{d}=\sum_{u} A^{\text{x}(h)}_{t u} V^{\text{x}}_{u h}{}^{d}.
$$

Finally merge heads with $W_O^{\text{x}}$ and add as a sublayer (pre-norm as usual).

---

## 📁 Repository Structure

```
einstein-transformer/
├── 📄 README.md          # This comprehensive guide
├── 📝 main.tex           # LaTeX source code
├── 📖 main.pdf           # Compiled PDF document
├── 🔧 .gitignore        # Git ignore patterns
└── 📜 LICENSE            # MIT License
```

### File Descriptions

| File | Purpose | Format |
|------|---------|--------|
| `README.md` | Online documentation with math rendering | Markdown + LaTeX |
| `main.tex` | Complete mathematical exposition | LaTeX source |
| `main.pdf` | Publication-ready document | PDF |
| `.gitignore` | Version control exclusions | Text |

---

## 🔬 Mathematical Notation Guidelines

### Einstein Summation Convention
- **Repeated indices** (one upper, one lower) are automatically summed
- **Temporal indices**: `t`, `s`, `u` are never implicitly summed
- **Explicit operations**: Sums and softmax operations are always written out
- **Index placement**: Clear distinction between contravariant (upper) and covariant (lower) indices

### Tensor Operations
- **Unambiguous contractions**: Every operation specifies exact index pairing
- **Shape consistency**: All tensor operations maintain dimensional compatibility
- **Broadcasting avoidance**: No implicit broadcasting—all operations are explicit

### Notation Benefits
- 🎯 **Eliminates ambiguity** in matrix/tensor operations
- 🔍 **Clarifies implementation** requirements
- 📐 **Enables dimensional analysis** for debugging
- 🔄 **Facilitates optimization** reasoning

---

## 💡 Usage Examples

### For Researchers
```latex
% Reference specific equations in papers
\cite{einstein_transformer_2024}
% Use notation for clear mathematical exposition
$$A_{t s}^{(h)}=\text{softmax}_{s}\big(L_{t s}^{(h)}\big)$$
```

### For Engineers
```python
# Implementation guided by Einstein notation
def attention_step(Q_th_d, K_sh_d, V_sh_d):
    # L_{ts}^{(h)} = (1/√d_k) * Q_{th}^k * K_{sh}_k
    logits = torch.einsum('thd,shd->tsh', Q_th_d, K_sh_d) / sqrt(d_k)
    # A_{ts}^{(h)} = softmax_s(L_{ts}^{(h)})
    weights = torch.softmax(logits, dim=1)  # softmax over s
    # Y_{th}^d = Σ_s A_{ts}^{(h)} * V_{sh}^d
    return torch.einsum('tsh,shd->thd', weights, V_sh_d)
```

### For Students
- 📖 **Start with** [Conventions and Einstein Primer](#conventions-and-einstein-primer)
- 🔍 **Focus on** index placement and summation rules
- 💡 **Practice with** simple examples before advanced topics
- 🔄 **Cross-reference** equations with implementation code

---

## 🤝 Contributing

We welcome contributions to improve this mathematical exposition! Here's how you can help:

### Types of Contributions
- 🐛 **Bug Reports**: Mathematical errors, typos, unclear notation
- ✨ **Enhancements**: Additional architectures, optimization techniques
- 📚 **Documentation**: Examples, tutorials, implementation guides
- 🔧 **Tools**: Code implementations, visualization scripts

### Contribution Process
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-addition`)
3. **Make** your changes with clear commit messages
4. **Test** LaTeX compilation (`pdflatex main.tex`)
5. **Submit** a pull request with detailed description

### Mathematical Standards
- ✅ **Follow Einstein notation** conventions consistently
- ✅ **Explicit temporal indices** (no implicit summation over t, s, u)
- ✅ **Clear index semantics** (upper/lower placement)
- ✅ **Dimensional consistency** in all equations
- ✅ **Proper citations** for new techniques/papers

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@misc{decoder_einstein_notation,
  title={Decoder-Only Transformers in Einstein Notation},
  author={Valentin A.},
  year={2024},
  howpublished={\url{https://github.com/flewor1/einstein-transformer}},
  note={A comprehensive mathematical exposition of modern transformer architectures}
}
```

### Academic Use
This document is designed for academic and educational purposes. When using equations or concepts in your work:
- 📚 **Cite appropriately** using the BibTeX above
- 🔍 **Reference specific sections** for detailed citations
- 📧 **Contact author** for collaboration or questions

---

## 📚 Related Work

### Foundational Papers
- **Attention Is All You Need** ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))
- **RoFormer: Enhanced Transformer with Rotary Position Embedding** ([Su et al., 2021](https://arxiv.org/abs/2104.09864))
- **Train Short, Test Long: Attention with Linear Biases** ([Press et al., 2021](https://arxiv.org/abs/2108.12409))

### Optimization Techniques
- **FlashAttention: Fast and Memory-Efficient Exact Attention** ([Dao et al., 2022](https://arxiv.org/abs/2205.14135))
- **GQA: Training Generalized Multi-Query Transformer Models** ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))

### Mathematical Foundations
- **Tensor Notation and Einstein Summation** - Standard references in differential geometry and tensor calculus
- **Matrix Calculus** - For gradient derivations and optimization theory

### Related Repositories
- 🔧 **[Transformer Implementations](https://github.com/topics/transformer)** - Code implementations
- 📊 **[Attention Visualizations](https://github.com/topics/attention-visualization)** - Understanding attention patterns
- 📚 **[Mathematical ML](https://github.com/topics/mathematical-machine-learning)** - Rigorous ML theory

---

## ⚖️ License

This work is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

```
MIT License - Copyright (c) 2024 Valentin A.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

### Usage Rights
- ✅ **Commercial use** permitted
- ✅ **Modification** and distribution allowed
- ✅ **Private use** encouraged
- ⚠️ **Attribution required**

---

<div align="center">

**📬 Questions? Issues? Contributions?**

[Open an Issue](https://github.com/flewor1/einstein-transformer/issues) • [Start a Discussion](https://github.com/flewor1/einstein-transformer/discussions) • [Submit a PR](https://github.com/flewor1/einstein-transformer/pulls)

---

*This document provides a rigorous mathematical foundation for understanding modern transformer architectures through the lens of Einstein notation, enabling clearer reasoning about tensor operations and index semantics.*

**Made with ❤️ for the research community**

</div>
