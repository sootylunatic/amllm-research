---
title: "AM-LLM — architektúra és implementáció (v0.2.1)"
version: "v0.2.1"
created: "2026-10-21"
updated: "2026-10-23"
status: "HISTORICAL"
scope: "VERSION"
source_of_truth: true
supersedes: []
superseded_by: []
---

[English](ARCHITECTURE.md) | **Magyar**

> **Dokumentumhatókör (2026-10-23):** ez a **v0.2.1 történeti kódállapotának pontos leírása**. Nem a későbbi F5 vagy éjszakai variánsok specifikációja: azokat az ARCHITECTURE_F5.hu.md írja le. A projekt aktuális **A–B–C–D céljai**: [../00-goals/RESEARCH_GOALS.hu.md](../00-goals/RESEARCH_GOALS.hu.md). Az eredeti A kérdésben a tényleges aktív számítás megtakarítása a fő bizonyíték; a memória (B), generálás (C) és nagy modellképesség (D) **külön** vizsgálandó. Az alábbi architektúramegállapításokat ettől nem módosítottuk.


# AM-LLM — architektúra és implementáció (v0.2.1)


Ez a dokumentum a `src/amllm/` kód pontos leírása. A döntések indoklása az ADR-ekben van
([`adr/`](../04-decisions/README.hu.md)); a mérési protokoll a [../02-methodology/METHODOLOGY.hu.md](../02-methodology/METHODOLOGY.hu.md)-ben.

## 1. Áttekintés

```text
bájtok ─► embedding + tanult pozíció ─► [ v0.1-topológia: 1× attention ]
                                         │
                ┌────────────────────────┴───── hop h = 1 … H ─────────────────────────┐
                │  w_h = s_h · active_h            (survival-súly; hard módban maszk)    │
                │  x ← x + w_h · Attn_h(LN(x))      (megosztott, vagy hoponként saját)   │
                │  x ← x + w_h · Bank(LN(x))        (csak az aktív tokenekre fut)        │
                │  p_h = σ(halt(x));  s_{h+1} = s_h · (1 − p_h)                          │
                │  hard: active_{h+1} = active_h ∧ (s_{h+1} ≥ τ)                         │
                └───────────────────────────────────────────────────────────────────────┘
                                         │
                               LN ─► lm_head ─► következő bájt logitjai
```

Jelölések: `d` rejtett dimenzió, `t` kontextushossz, `M` modulszám, `E` expert/modul, `Km` modul Top-K,
`Ke` expert Top-K, `H` maximális hop, `h_e` expert rejtett szélessége.

## 2. Komponensek

### 2.1 Bemenet és kimenet (`_ByteLM`)
- `embed: Embedding(256, d)` + `positions: Embedding(max_seq_len, d)`.
- Kimenet: `LayerNorm` → `Linear(d, 256, bias=False)` (nem kötött az embeddinghez).
- `diagnostics` kapcsoló (lásd 2.7).

### 2.2 Kauzális attention (`CausalAttention`)
`qkv: Linear(d, 3d)`, `out: Linear(d, d)`, `n_heads` fej, `F.scaled_dot_product_attention(is_causal=True)`.
Pre-LN: `x + Attn(LN(x))`.

### 2.3 Expert-tár (`ExpertBank`)
`G = M·E` független expert, egymásra rakott súlyokkal:
`w1 ∈ R^{G×d×h_e}`, `b1 ∈ R^{G×h_e}`, `w2 ∈ R^{G×h_e×d}`, `b2 ∈ R^{G×d}`;
`FFN_g(x) = SiLU(x·w1_g + b1_g)·w2_g + b2_g`.

Bemenet: hozzárendelések listája `(token, expert, súly)`. Kimenet: `y[token] = Σ súly · FFN_expert(x[token])`.

| `dispatch` | Működés | Kiszámolt sorok | Host-szinkron |
|---|---|---|---|
| `loop` | expert szerint rendez, expertenként egy matmul-pár a saját soraira | = hozzárendelések száma | 1 / hívás |
| `bmm` | minden expert sorait `cap = max(darabszám)`-ra paddingeli, 2 `baddbmm` | `G · cap` (≥ hozzárendelések) | 1 / hívás |

**Összegzés.** `routes_per_token = Km·Ke` esetén (a modulbankon belül mindig ez a helyzet) a tokenenkénti
expert-kimenetek rögzített sorrendű összegzéssel adódnak össze (`index_copy` vissza tokenrendbe, majd
`view(N, k, d).sum(1)`) atomikus `index_add` helyett, így a CUDA-forward bitre reprodukálható.

A két dispatch-út kimenete és gradiense azonos (CPU-s és CUDA/bf16 tesztek). A `rows_computed` és a
`last_executed` diagnosztikai számlálók.

### 2.4 Hierarchikus modulbank (`HierarchicalModuleBank`)
Bemenet `x ∈ R^{N×d}` (csak a még futó tokenek).

1. Modulpontszám `z = Wm·x + bm ∈ R^{N×M}`; Top-`Km` modul `m`.
2. Expert-pontszám csak a kiválasztott modulokra: `u = x·We[m] + be[m] ∈ R^{N×Km×E}`; Top-`Ke` expert `e`.
3. Útvonal-súlyok (`route_weighting`):
   - `topk_softmax` (alapértelmezett): softmax a **kiválasztott** pontszámokon. `k = 1` esetén a súly azonosan 1,
     és az adott router nem kap nyelvimodell-gradienst (ADR-0013).
   - `full_softmax`: softmax az **összes** pontszámon, a kiválasztott útvonalaknál kiolvasva (Switch-Transformer-stílus);
     minden router minden `k` mellett kap gradienst. A súlyok összege ekkor nem 1.
4. Hozzárendelések `(token n, expert m·E + e, súly w_m·w_e)`: tokenenként `Km·Ke` darab.
5. Terheléskiegyenlítés (Switch-stílus, a `load` leválasztva):
   modul `M · Σ_m mean_n softmax(z)_m · load_m`, `load_m = count_m / (N·Km)`;
   expert `E · Σ_m load_m · Σ_e imp_{m,e} · load_{m,e}` (a modult választó tokenekre átlagolva).
6. `router_mode="random"`: egyenletes véletlen útvonalak egyenletes `1/k` súllyal — tanítási „nem tanult routing”
   kontroll, azonos paraméterekkel és számítással.
7. Kiértékelési diagnosztika: `random_routing` (véletlen választás, a tanult pontszámok mint súly), `last_route` (trace).

A fix MoE kontroll ugyanez az osztály `M = 1, Km = 1` mellett.

### 2.5 AM-LLM rekurzió (`AdaptiveModularLM.forward`)
Tokenenkénti állapot: `s` (survival, kezdetben 1), `active` (hard módban), `counts` (lefutott hopok).

Hoponként (`h = 0 … H−1`):
1. Hard módban `idx` = aktív tokenek; ha üres, a ciklus véget ér (a további hopok **nem futnak**).
2. Súly: `w = s` (`survival`) vagy `w = 1` (`step`, `none`); hard módban `w ← w · active`.
3. `attn_per_hop` esetén `x ← x + w · Attn_h(LN_h(x))`; az `Attn_h` megosztott, vagy `unshared_attention=True`
   mellett a `h ≥ 1` hopokban saját súlyú. Az attention minden tokenre fut (a megállt tokenek kulcsként/értékként szolgálnak).
4. Modulbank: soft módban minden tokenre, hard módban csak `idx`-re: `x[idx] += w[idx] · Bank(LN(x[idx]))`.
5. Halting (`h < H−1`, `halt_rule ≠ none`): `p = σ(halt_head(x))`, `s ← s·(1−p)`;
   hard `survival`: `active ← active ∧ (s ≥ τ)`; hard `step` (v0.1): `active ← active ∧ (p < τ)`.

Kiegészítő veszteség: `aux = λ_ponder · P + λ_balance · mean_h(balance_h)`; soft módban `P = Σ_h mean(s_h)`
(várt hopszám), hard módban a lefutott hopok átlaga. A halting-fej inicializálása: súly 0, bias `halt_bias_init`
(−1 → `p ≈ 0,27`).

**Tesztelt tulajdonságok:** kauzális minden módban (fp64-ben pontosan CPU-n és CUDA-n; bf16-ban kerekítésen belül
CUDA-n); determinisztikus kiértékelés; `τ → 0` esetén hard ≡ soft; `module_counts = Σ hops · Km`,
`expert_counts = Σ hops · Km · Ke`; a kiszámolt FFN-sorok = a routolt hozzárendelések (`loop`); korai kilépésnél
kevesebb FLOP.

### 2.6 Kontrollmodellek (`LayeredLM`)
`n_layers` blokk, mindegyikben saját attention és vagy dense FFN (`d → ffn_hidden → d`, SiLU), vagy MoE-réteg
(`n_experts` × `expert_hidden`, Top-`moe_topk`). Pre-LN, residual. UT-kontroll: `arch="amllm"`, `M=E=Km=Ke=1`
(egy közös blokk H-szor), `halt_rule="none"` mellett (fix mélység) vagy haltinggal (csak-mélység kontroll).

### 2.7 Diagnosztika és a gyors inferencia-út
`diagnostics=True` mellett (tanítás és kiértékelés) minden forward kiszámolja a terheléskiegyenlítő veszteséget,
a modulstatisztikát és egy analitikus FLOP-riportot (ez a darabszámokat visszaolvassa a hostra).
A `set_diagnostics(model, False)` ezeket kihagyja; a kimenet azonos (tesztelve). Hoponként megmaradó host-szinkronok:
a dispatch expert-darabszámai és hard módban az aktív tokenek indexe. A késleltetést mindkét beállítással közöljük.

## 3. Konfiguráció (`ModelConfig`)

| Mező | Alap | Jelentés |
|---|---|---|
| `arch` | `amllm` | `amllm` / `dense` / `moe` |
| `vocab_size`, `max_seq_len` | 256, 128 | byte-tokenek, kontextus (a kísérletekben 256) |
| `d_model`, `n_heads` | 64, 4 | rejtett dimenzió, fejek (a kísérletekben 128) |
| `n_layers`, `ffn_hidden` | 4, 256 | Dense/MoE mélység, dense FFN szélesség |
| `n_experts`, `moe_topk` | 8, 2 | MoE-réteg |
| `n_modules`, `experts_per_module` | 8, 4 | AM-LLM modulbank |
| `module_topk`, `expert_topk` | 2, 2 | Top-K szintenként |
| `expert_hidden` | 128 | expert FFN szélesség (AM-LLM és MoE) |
| `n_hops` | 4 | maximális hop |
| `attn_per_hop` | True | False = v0.1-topológia (egy attention a hopok előtt) |
| `unshared_attention` | False | hoponként saját attention (4 hopnál +3 attention-blokk) |
| `halt_rule`, `halt_threshold` | `survival`, 0,5 | ADR-0004 |
| `halt_bias_init` | −1,0 | halting-fej bias |
| `route_weighting` | `topk_softmax` | `topk_softmax` / `full_softmax` (ADR-0013) |
| `router_mode` | `learned` | `learned` / `random` (kontroll) |
| `ponder_coefficient` | 0,002 | λ (a fő kísérletekben 0,01, előre rögzített szabállyal választva) |
| `balance_coefficient` | 0,01 | terheléskiegyenlítés |

## 4. FLOP-becslés (forward, tokenenként; szorzás-összeadás = 2 FLOP)

| Tag | Képlet |
|---|---|
| attention-alkalmazás | `8d² + 2d(t+1)` (qkv + out projekciók; QKᵀ és AV átlagosan `(t+1)/2` kulccsal) |
| expert / FFN hívás | `4·d·h` |
| modulrouter | `2·d·M` hoponként; expert-router `2·d·E` kiválasztott modulonként |
| halting-fej | `2·d` hoponként (az utolsó kivételével) |
| LM-fej | `2·d·256` |

- **Routolt** (`flops_per_token`, az 1. fázisban „implementált”): attention minden lefutott hopban minden tokenre, és csak
  a *kiválasztott* expert-hozzárendelések.
- **Végrehajtott** (`amllm.profile`): a routolt plusz a `bmm` dispatch által szintén kiszámolt padding-sorok; `loop`
  esetén azonos a routolttal.
- **Ideális:** attention csak az aktív query-tokenekre — még nincs implementálva.
- **Profilozott** (`amllm.profile`): `torch.profiler` matmul-FLOP; az SDPA nincs benne, ezért a különbség a becsléshez
  pontosan `attention-hívások · 2d(t+1)` (ellenőrizve). `bmm` dispatch mellett a profilozott FLOP a paddinget is tartalmazza.

## 5. Modulok

| Fájl | Tartalom |
|---|---|
| `config.py` | `ModelConfig` (validáció, dict-konverzió) |
| `routing.py` | `count_ids`, `ExpertBank`, `HierarchicalModuleBank` |
| `model.py` | `AdaptiveModularLM`, `LayeredLM`, FLOP-függvények, `set_dispatch`, `set_diagnostics`, `set_random_routing` |
| `data.py` | korpuszépítés, csomagonkénti split, batch-mintavétel, bájtosztályok |
| `synth.py` | szintetikus feladatok kategóriacímkékkel |
| `train.py` | tanítás, kiértékelési módok, futáskönyvtár |
| `experiment.py` | preset × seed mátrix, kísérletenkénti `summary.csv` |
| `invariants.py` | helyességi ellenőrzések (CPU és CUDA), kihagyás-mérés, route-trace |
| `profile.py` | késleltetés (diagnosztikával/nélküle), VRAM, profilozott FLOP, kernel- és szinkronszám, aktív paraméterek |
| `router_analysis.py` | routing-statisztika, kölcsönös információ, tanítatlan kontroll |
| `report.py` | globális összesítő, összehasonlító JSON, ábrák (en/hu) |
| `bench.py`, `env.py` | mikrobenchmark, környezetnapló (útvonal-tisztítással) |
