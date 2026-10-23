---
title: "W1 előregisztráció — A2 módosítás (diagnosztikai megjegyzés, bankonkénti expertstatisztika)"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "RESULT"
source_of_truth: true
supersedes: []
superseded_by: []
locked: true
---

[English](PREREGISTRATION_A2.md) | **Magyar**

# W1 előregisztráció — A2 módosítás (diagnosztikai megjegyzés, bankonkénti expertstatisztika)

**Állapot:** A lezárt [W1 előregisztrációt](PREREGISTRATION.hu.md) és az [A1](PREREGISTRATION_A1.hu.md)-et nem írjuk át; ez dátumozott kiegészítés, amelyet egy igazolt felülvizsgálati találat indokol. A W1 kódja (`train_code_sha256` = `ba81b4fc02cd7b63`) ettől **nem változik**, a futások folynak.
**Dátum:** 2026-10-23.

## A2.1 A hiba és a hatóköre
A W1 kódjában az `AdaptiveModularLM` hoponkénti `module_counts` / `expert_counts` számlálói egyetlen `[modul, expert]` táblába összegződnek, a bankazonosító nélkül. Ha `bank_groups > 1` (FD01a, FD01b, FD01c), a **külön bankok azonos indexű expertjei összemosódnak**: pl. 12 különböző expertből a jelentés 4 kategóriát lát.
- **Érintett (diagnosztika):** a `report.json` és a `summary.csv` `expert_counts`, `module_counts`, `dead_experts`, `expert_usage_entropy`, `max_expert_share` mezői az FD01a/b/c futásoknál **érvénytelenek**.
- **Nem érintett:** a validációs veszteség, a FLOP (`flops_per_token`, `flops_per_token_executed`), a paraméterszám és a tanítás. Az FD-döntési szabályok kizárólag a veszteségre épülnek, az AD-kapu a veszteségre és a FLOP-ra; **egyik döntési szabály sem használja az expertstatisztikát**. A `bank_groups = 1` futások (B_fixed, B_moe, A0, FD01d, FD02, FD03, AD02a) számlálói helyesek.

## A2.2 Kezelés
1. Az FD01a/b/c érvénytelen mezőit a jelentések nem idézik; a bankonként helyes értékeket a **`tools/analysis/bank_usage.py`** adja a mentett checkpointokból (route-trace, validációs split, CPU), és az FD01-eredményekhez ez kerül.
2. A javítás (bankonkénti számlálók) a **`w2-prep` ágon** van kész és tesztelt (a következő hullámhoz); a W1 kódja fagyasztott marad, ezért a `main`-en nem változtatunk a tanítókódon.
3. A W1 nem módosul: nincs újrafuttatás, nincs új kapu.
