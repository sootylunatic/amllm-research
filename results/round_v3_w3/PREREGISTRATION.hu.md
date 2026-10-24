---
title: "W3 előregisztráció"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "RESULT"
source_of_truth: true
supersedes: []
superseded_by: []
locked: true
---

[English](PREREGISTRATION.md) | **Magyar**

# W3 előregisztráció

**Állapot:** JÓVÁHAGYVA ÉS LEZÁRVA — a tulajdonos 2026-10-24-én jóváhagyta („jóváhagyom, mehet”). Innentől csak dátumozott kiegészítéssel módosítható; a tartalmat `PREREGISTRATION.sha256` rögzíti. A tanítás és a mérés a lezárás után indul.

**Kódverzió:** a tanítási kód **változatlan** a W2-höz képest: `train_code_sha256` = `cd909ea282660f0f`. A W3 csak a nem hash-elt végrehajtási kódot bővíti (`fastpath.py`, `generate.py` — többbankos, hoponkénti normás és adaptív K-s modellek a KV-cache és a statikus/bucketelt kemény úton, fp64/fp32 egyezőségi tesztekkel; új mérőmodul: `w3measure.py`). **Ezért a W2 futásai (B_moe, B_dense, B_fixed, FDC1, AD02a-3, AD03b) a W3-ban közvetlenül összevethetők**, nem futnak újra.
**Adat / protokoll:** a W1/W2-vel azonos — `code_corpus_v2`, batch 32 × 512, lr 1e-3, seed 42/43/44, validációs NLL; teszt-split és holdout nincs használva (a mérőmodul is csak a val splitet tölti be). **Ismert korlát:** a tanító split 53,1M token; 9000 lépés ~2,8, 18 000 lépés ~5,6 epoch ismétlés.
**Döntési szabály (minden páros összevetésre, a W1/W2-vel azonos, irányított):** Δ = változat − referencia; `SUPPORTED` (jobb) = Δ ≤ −0,010 **és** a 95%-os t-CI felső határa < 0 **és** 3/3 seed negatív; `NOT_SUPPORTED_IN_SCOPE` (rosszabb) = tükörképe; minden más `INCONCLUSIVE` (nem egyenértékűség). „Azonos FLOP” = routolt FLOP; a végrehajtott FLOP külön szerepel.

## 1. Tanítás: az FDC1 előnye szerkezeti-e (9000 lépés, 3 seed)

| ID | Preset, `--set` | Paraméter | Routolt FLOP× B_moe | Szerep |
|---|---|---|---|---|
| MoE12 | `moe_f5 n_experts=12` | 31,37M | 1,002 | **az FDC1 paraméterben és routolt FLOP-ban illesztett MoE-kontrollja** (72 expert, tokenenként 2/réteg) |
| B3H | `amllm_f5 halt_rule=none bank_groups=3` | 31,37M | 1,001 | 3 bank, hierarchikus routing (csak a bankszám változik a B_fixedhez képest) |
| B1F | `amllm_f5 halt_rule=none n_modules=1 experts_per_module=24 module_topk=1` | 17,15M | 1,004 | 1 bank, lapos routing (csak a routing változik a B_fixedhez képest) |

A W2-ből: B_fixed (1 bank, hierarchikus, 17,15M), FDC1 (3 bank, lapos, 31,36M), B_moe. Ez egy 2×2 (bankszám × routing) és mellé az azonos méretű MoE.

**Elsődleges kérdés:** FDC1 − MoE12 (azonos paraméter, azonos routolt FLOP).
- `SUPPORTED` → az FDC1 előnye ezen a skálán nem csak a paraméterszámé.
- **A W3b-feltétel (a későbbi ~300M-es kísérlethez, a tulajdonos külön döntésével):** az FDC1 „tartja magát”, ha az FDC1 − MoE12 átlagos Δ ≤ +0,010 (legfeljebb minimálisan rosszabb). Ez **nem** A-bizonyíték, csak a továbblépés feltétele; a 300M-es futás nem része a W3-nak.

**Másodlagos (ugyanazzal a szabállyal):** FDC1 − B3H (a lapos routing hatása 3 bankkal, azonos paraméter); B1F − B_fixed (a lapos routing hatása 1 bankkal, azonos paraméter); B3H − B_fixed és FDC1 − B1F (a bankszám hatása; a paraméter 1,83×, ezt minden sorban feltüntetjük). Kölcsönhatás leíróan: (FDC1 − B3H) − (B1F − B_fixed), páros seedenként, CI-vel.

9 futás.

## 2. Tanítás: megmarad-e az előny hosszabb tanításnál (18 000 lépés, 3 seed)

| ID | Preset, `--set` | Lépés |
|---|---|---|
| FDC1_18k | `amllm_f5 halt_rule=none bank_groups=3 n_modules=1 experts_per_module=24 module_topk=1` | 18 000 |
| MoE12_18k | `moe_f5 n_experts=12` | 18 000 |

Kérdés: FDC1_18k − MoE12_18k (ugyanazzal a szabállyal); leíróan a különbség változása 9k és 18k között. A 18k-s futások ~5,6-szor látják az adatot, ezért a túltanulás (a val NLL emelkedése) lehetséges kimenet; a görbét teljesen jelentjük. 6 futás.

## 3. Mérés (a terv „gyakorlati mérés” hulláma)

**Modellek:** B_moe, B_dense, B_fixed, FDC1, AD02a-3, AD03b (W2) és MoE12 (W3, 9k) — 7 konfiguráció × 3 seed = 21 checkpoint. **Egyetlen gépen** (a `w3a` slot RTX 3090-e, amely nem tanít, így a mérést semmi nem zavarja), egy környezetben, azonos backenddel: bf16, a Dense kivételével Triton grouped dispatch (csak a routolt sorok); a mérőmodul más környezet eredményét nem keveri egy fájlba. Mérések (`amllm.w3measure`):

| Mérés | Tartalom | Cél |
|---|---|---|
| Végrehajtott munka | routolt és végrehajtott FLOP/token, kiszámolt expert-sorok dispatchenként (bmm vs. padding nélküli loop/Triton); halting modellnél a bucketelt kemény út paddingja és az attention-lekérdezések aránya | **A** |
| Késleltetés | teljes szekvenciás forward (8 × 512): eager és CUDA Graph, ahol rögzíthető | **C** |
| Dekódolás | KV-cache, batch 1/8/32: prefill (TTFT), ms/token p50/p95, token/s (eager és CUDA Graph) | **C** |
| VRAM (K5) | súlyok fp32/bf16, KV-cache batch 1/2/4/8 × kontextus 128/256/512, mért csúcsmemória | **B** |
| Expert-hotset (K6) | használati koncentráció bankonként/rétegenként, LRU-cache szimuláció (25/50/75% expert), bájt/token | **B** |
| NLL pozíció szerint | 64 tokenes sávok az 512-es ablakban | minőség hosszban |
| Generálás | 32 val-prompt × 128 token, 384 új token, mohó és mintavételes (T 0,8, top-p 0,95): ismétlődő 4-gram arány, distinct-1/2, hurokba kerülő szekvenciák aránya | hurokba kerülés |

**Kapuk és összevetések:**
- **A-sikerkapu (az A1 szerint, most először mérhető):** egy AD-jelölt (AD02a-3, AD03b) `SUPPORTED_IN_SCOPE`, ha a padding nélküli végrehajtott FLOP-ja/token ≤ 0,90 × a B_moe-é (mindkettő a Triton-dispatch referencia-útján mért `flops_per_token_executed`: csak a ténylegesen kiszámolt expert-sorok; az attention ezen az úton minden lekérdezésre lefut, mert a megállt tokenek kulcsként/értékként megmaradnak — ez a konzervatív szám; a bucketelt út csomagolási aránya leíróan mellette) **és** a kemény θ 0,5 val NLL ≤ 1,03 × B_moe, 3/3 seeden; különben `NOT_SUPPORTED_IN_SCOPE`. Az FDC1-re: „azonos paraméter és azonos végrehajtott számítás mellett jobb” csak akkor mondható, ha az FDC1 − MoE12 `SUPPORTED` (1. rész) **és** az FDC1 padding nélküli végrehajtott FLOP-ja ≤ 1,00 × a MoE12-é (ugyanazon a Triton-úton); egyébként a két számot külön jelentjük, összevont állítás nélkül.
- **C:** decode token/s, TTFT, p50/p95 arány a B_moe-hoz, azonos GPU-n és backenden — leíró, kapu nélkül.
- **B:** VRAM-bontás és hotset — leíró.
- **Pozíció szerinti NLL:** FDC1 − MoE12 és FDC1 − B_moe az utolsó sávban (448–512) a döntési szabállyal (másodlagos); a többi sáv leíró.
- **Generálás:** FDC1, MoE12, B_moe, B_dense hurok- és ismétlésmérői, páros seedenkénti különbséggel és CI-vel — leíró, kapu nélkül.

## 4. Terjedelem, költség, leállás
- Öt RTX 3090 (Community, $0,22/óra): a `w3b`–`w3e` tanít (9 + 6 futás, a W1/W2-ben mért futásidőkkel kiegyensúlyozott sorok: `queues/w3b.txt` 5, `w3c.txt` 4, `w3d.txt` 3, `w3e.txt` 3 futás, ~2–2,3 óra falióra); a `w3a` csak mér: azonnal a W2 18 checkpointját, a MoE12 elkészülte után annak 3 checkpointját (~1–1,5 óra). Összesen ~$2,5–3.
- A sor podoldali, futás közben kiegyensúlyozható; minden kész futás ~2 percen belül a mentőszerverre kerül checkpointtal, SHA-256-tal ellenőrizve. A W2- és a MoE12-checkpointokat a mentőszerver adja a mérőgépnek (a Mac mobilnetje nem érintett).
- Lezárás: döntési tábla, mérési táblák, záró mentés, riport; a gépek sorsáról a tulajdonos dönt.

## 5. Nem része a W3-nak
A ~300M-es FDC1-kísérlet (W3b; az adatkorpusz még nem kész, a tulajdonos külön dönt), holdout és OOD (K7 — a W4-gyel), tanítási kódváltozás, a 2. résznél hosszabb tanítás.
