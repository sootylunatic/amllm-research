**English** | [Magyar](0002-byte-level-tokenization.hu.md)

# ADR-0002: Use byte-level tokens in the first experiment series

- **Status:** accepted (for the first series)
- **Date:** 2026-10-21
- **Affects:** `ModelConfig.vocab_size = 256`, `amllm.data`

## Context
For an architecture comparison every model must see the same representation. Training and versioning a BPE
tokenizer adds a variable, and for tiny models (~1–2.5M parameters) a large vocabulary matrix would dominate the
parameter count.

## Decision
One byte = one token (vocabulary 256), context 256 bytes. Losses are reported in nat/byte and bits/byte (bpb).

## Consequences
- Simple, tokenizer-independent, identical for all models.
- Perplexity is per byte and not comparable with token-level LLMs.
- Coding ability (e.g. `tsc`, `pass@1`) cannot be evaluated at this level; that is a later step (BPE).
