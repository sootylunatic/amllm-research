**English** | [Magyar](ARCHITECTURE_GUIDE.hu.md)

# Architecture guide — fixed and adaptive computation

**Scope:** ~17M BPE F5 / overnight configurations, not the historical [v0.2.1 byte implementation](01-architecture/ARCHITECTURE.md).

```text
6-layer MoE                        6-hop AM-LLM FixedDepth
Attn_1 → private experts 1          Attn_1 → shared bank (24 experts)
Attn_2 → private experts 2          Attn_2 → same bank
...                                 ...
Attn_6 → private experts 6          Attn_6 → same bank
```

The attention parameter budget is identical: **6 × 591,360** weights; expert parameters **7,096,320** in both. The MoE has 6 layers × 4 experts, selecting two per layer; AM has 6 modules × 4 experts, selects module then expert and reuses weights across hops. **Four factors change together:** expert tying, hierarchical rather than flat routing, one common bank normalization instead of per-layer normalization, and the router's training rule (the MoE control uses `topk_softmax`, every AM-LLM configuration `straight_through`: same forward value, different gradient to the router). The preregistered W1 round separated them ([W1 report](../results/round_v3_w1/W1_REPORT.md)): expert tying and hierarchical routing each cost quality; for the normalisation and the training rule no effect was detected at n = 3 (inconclusive, which is not equivalence). Router storage is ~11,550 parameters (~0.07% of full model).

| Model | Depth | Routing | Analytic MFLOP/token |
|---|---:|---|---:|
| MoE baseline | 6 layers | Top-2 of 4/layer | 22.834 |
| FixedDepth base | 6 hops | 1 module × 2 experts | 22.857 |
| FixedDepth 8 hops | 8 hops | 1 × 2 | 28.379 |
| FixedDepth Top-2/Top-2 | 6 hops | 2 × 2 | 29.954 |
| FixedDepth smaller experts | 6 hops | 1 × 2 narrow | 19.337 |

Common features: d_model=384, 6 attention heads, 8192-vocab BPE, 512-token context, distinct attention weights per hop, SiLU experts; AM-LLM configurations use straight-through top-k router weighting, the MoE control the default `topk_softmax`. Output LM head alone costs ~**6.291 MFLOP/token (~28%)** and cannot be removed by halting. Analytical FLOPs omit some backend overhead and are **not measured generation throughput**.

## Why current halting fails

Soft survival **training executes every hop**, scaling state updates by survival probability. Hard **inference freezes stopped tokens**, which nevertheless remain later attention keys/values. The hard truncated route is not directly optimized. In F5 seed 42 soft NLL ~2.183 became hard NLL ~3.426 at stop threshold 0.5 (~3.47 hops). Threshold tuning alone did not yield a favorable quality/compute balance.

**FixedDepth** is a valuable AM-LLM family member and matched reference but does **not** demonstrate dynamic computation. **AdaptiveDepth** tests that original idea. CPU–GPU expert paging is not implemented. Research needs controlled ablations, per-hop predictions, hard-aware training, intermediate exit losses and selective within-hop operations. [Open questions](OPEN_QUESTIONS.md) · [Results](../results/RESEARCH_REPORT.md).
