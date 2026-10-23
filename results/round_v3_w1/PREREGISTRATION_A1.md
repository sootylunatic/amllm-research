---
title: "W1 preregistration — amendment A1 (before the hypothesis runs start)"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "RESULT"
source_of_truth: true
supersedes: []
superseded_by: []
locked: true
---

**English** | [Magyar](PREREGISTRATION_A1.hu.md)

# W1 preregistration — amendment A1 (before the hypothesis runs start)

**Status:** The approved and sealed [W1 preregistration](PREREGISTRATION.md) (`PREREGISTRATION.sha256`) is not rewritten; this supplement is a **dated amendment**, motivated by two verified review findings. The FD/AD hypothesis runs (parts 2–3) start after the amendment is recorded; at this moment no FD/AD result exists and the anchors are not affected. In case of conflict the amendment governs.
**Date:** 2026-10-23. **Code version:** unchanged (`ba81b4fc02cd7b63`).

## A1.1 AD success gate aligned with the main task list (instead of the gate in part 3)

The main task list (RESEARCH_TASKS, AD gate) asks for an advantage in **actually executed** compute. The routed `flops_per_token` is only a screening signal: a halted token's attention is still computed, and bmm padding is work too. Therefore:
- **Screening signal (W1):** if in `hard_surv_0.5` mode the val NLL ≤ 1.03 × the B_moe mean **and** `flops_per_token` ≤ 0.90 × B_moe's, on 3/3 seeds, the verdict is **`PROMISING_EXPLORATORY`**. It is not a success.
- **Success gate (`SUPPORTED_IN_SCOPE`)** may only be given if W3's real execution measurement (a padding-free path that really leaves out the skipped work; the number of rows and attention queries computed, padding counted; decode latency) shows the **executed** work ≤ 0.90 × B_moe's executed work, together with the val NLL condition. W1 alone cannot give it.
- In point 4 of "Scope", the phrase "`SUPPORTED_IN_SCOPE (routed FLOP)`" is void.

## A1.2 Restricting the causal claim of FD01 and a new control (FD01d)

FD01b changes **two** factors against B_fixed at once: it removes bank sharing and it reduces the expert pool available per hop from 24 to 4 (because of the equal total parameters). Therefore:
- The FD01b − B_fixed difference is a **package effect** ("separate small banks per hop vs. one shared large bank, equal total parameters"), not the pure effect of sharing. No such causal statement is made.
- **New control FD01d:** `bank_groups=1 n_modules=2 experts_per_module=2` — a single 4-expert bank used by all 6 hops (about 11.2M parameters). The four cells of the 2×2 layout (sharing × pool per hop): B_fixed (shared, 24), FD01d (shared, 4), FD01b (separate, 4), FD01a (separate, 24).
- **Causal statements only from pairs with an equal per-hop pool:** the effect of sharing = FD01b − FD01d (pool 4) and FD01a − B_fixed (pool 24); the effect of the pool = B_fixed − FD01d (shared) and FD01a − FD01b (separate). None of these pairs has equal total parameters; the report states this on every row. The "equal parameters" verdict (FD01b − B_fixed) remains a package effect.
- The decision rule (directional, |Δ| ≥ 0.010, CI, 3/3) is the same for every pair comparison.
- **Scope:** +3 runs (FD01d × 3 seeds): W1 totals 12 + 24 + 6 = **42 runs**.
