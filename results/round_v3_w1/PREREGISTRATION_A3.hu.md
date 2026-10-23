---
title: "W1 előregisztráció — A3 módosítás (negyedik összekevert különbség: a router tanítási szabálya)"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "RESULT"
source_of_truth: true
supersedes: []
superseded_by: []
locked: true
---

[English](PREREGISTRATION_A3.md) | **Magyar**

# W1 előregisztráció — A3 módosítás (negyedik összekevert különbség: a router tanítási szabálya)

**Állapot:** A lezárt [W1 előregisztrációt](PREREGISTRATION.hu.md), az [A1](PREREGISTRATION_A1.hu.md)-et és az [A2](PREREGISTRATION_A2.hu.md)-t nem írjuk át; ez dátumozott kiegészítés egy igazolt felülvizsgálati találat miatt. A kód (`ba81b4fc02cd7b63`) változatlan. Az itt hozzáadott futások az A3 rögzítése után indulnak.
**Dátum:** 2026-10-23.

## A3.1 A megállapítás
A mentett konfigurációk szerint a MoE-kontroll (`moe_f5`, `B_moe`) a `route_weighting = topk_softmax` alapértelmezést használja, a FixedDepth AM-LLM (`amllm_f5`, `B_fixed`) a `straight_through`-t. Az előre irányú érték azonos (a kiválasztott pontszámok softmaxa), a **routerhez jutó gradiens különbözik** (a `straight_through` a teljes softmax-valószínűségen is átvezeti). Az AM–MoE különbség leltára ezért **négy** összekevert tényezőt tartalmaz: (1) expert-kötés, (2) hierarchikus routing, (3) közös bank-normalizálás, (4) **a router tanítási szabálya**. A dokumentumok ennek megfelelően javítva.

## A3.2 Új kontrollok (FD07), 2 × 3 seed
| ID | Beállítás | Referencia | Mit különít el |
|---|---|---|---|
| **B_moe_st** (FD07a) | `moe_f5`, `route_weighting=straight_through` | B_moe | a szabály hatása a **MoE oldalon** |
| **B_fixed_fs** (FD07b) | `amllm_f5`, `halt_rule=none route_weighting=full_softmax` | B_fixed | a szabály hatása az **AM oldalon** (routergradienst megtartó, nem straight-through szabály) |

A `topk_softmax` az AM-ben Top-1 modulnál állandó 1 súlyt ad (a modulrouter nem kap LM-gradienst, ADR-0013): ez degenerált cella, ezért az AM oldalon a `full_softmax` a megfelelő kontroll. A döntési szabály (irányos, |Δ| ≥ 0,010, 95%-CI, 3/3 seed) a páronkénti összevetésekre azonos. Az eredmény a 2×2 elrendezés (architektúra × szabály) két sora; **oksági állítás csak azonos architektúrán belüli párból** tehető, és minden összevetés a teljes paraméterszámot kiírja.
- **Terjedelem:** +6 futás, a W1 összesen **48 futás**.
