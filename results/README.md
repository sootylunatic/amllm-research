**English** | [Magyar](README.hu.md)

# Research results

- [Public evidence report](RESEARCH_REPORT.md): E00–F5 and 36 overnight runs, strengths, failure modes, confidence intervals and disclosure limits.
- [Round V3 W1 report](round_v3_w1/W1_REPORT.md): 48 preregistered runs (anchors, FD01–FD03, FD07, AD02a) with per-seed values, decision rules, per-bank expert statistics and the AD01 diagnostics; [preregistration](round_v3_w1/PREREGISTRATION.md) and amendments, per-experiment `summary.csv`, analysis JSON in `round_v3_w1/analysis/`.
- [Round V3 W2 report](round_v3_w2/W2_REPORT.md): 66 preregistered runs (attention sharing, matched quality–compute points, FDC1, adaptive K) with per-seed values and AD01 diagnostics; [preregistration](round_v3_w2/PREREGISTRATION.md), per-experiment `summary.csv`, analysis JSON in `round_v3_w2/analysis/`.
- [Round V3 W3 report](round_v3_w3/W3_REPORT.md): FDC1 against the equal-size MoE, 2×2 single factors, 18k-step runs, practical measurement of 21 checkpoints (executed work, latency, decode, VRAM, hotset, generation loops) and an exploratory decode-speed benchmark; [preregistration](round_v3_w3/PREREGISTRATION.md), tables `analysis_*.txt`, raw `measure/` and `c_decode/`.
- [Round V3 W4 report](round_v3_w4/W4_REPORT.md): one-time descriptive check on an independent holdout (154 fresh packages, 36 frozen checkpoints, package-level bootstrap); [preregistration](round_v3_w4/PREREGISTRATION.md), comparison and aggregate per-package results in `round_v3_w4/holdout_eval/` (the holdout text is not published); K2 causal router test `k2_causal.json`.
- [summary.csv](summary.csv): machine-readable **E00–E09 only** (byte-level experiments).
- [Research status](../docs/RESEARCH_STATUS.md) · [Open questions](../docs/OPEN_QUESTIONS.md).

Compare summary CSV rows only within one experiment, code version and eval mode. Byte NLL is not comparable to BPE nat/token. A diagnostic-inclusive batch time is not decoded tok/s. Training VRAM is not inference VRAM. The 17M table is a secondary aggregate, **not the raw run archive**.
