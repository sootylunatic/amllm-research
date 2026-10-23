---
title: "W2 előregisztráció"
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

# W2 előregisztráció

**Állapot:** JÓVÁHAGYVA ÉS LEZÁRVA — a tulajdonos 2026-10-23-án jóváhagyta („javítás után indulhat a w2 podokon”). Innentől a kapuszámok nem változhatnak (csak dátumozott kiegészítéssel); a fájl tartalmát `PREREGISTRATION.sha256` rögzíti. A horgonyok és a 2–4. rész futásai a lezárás után indulnak.

**Kódverzió (a W2 alatt fagyasztva):** `train_code_sha256` = `cd909ea282660f0f` (a W2-kódjavítások után újraszámolva; a diagnosztikai javítások — `profile.py`, `router_analysis.py`, `tools/analysis/` — nem részei a hashnek). **Nincs összevetés a W1 futásaival** (más hash): minden W2-összevetés W2-n belüli, ezért a horgonyok újra futnak.
**Adat / protokoll:** a W1-gyel azonos — `code_corpus_v2`, 9000 lépés, batch 32 × 512, lr 1e-3, seed 42/43/44, validációs NLL (nat/token) az alapmódban (`fixed`; halting modellnél `hard_surv_0.5`); a teszt-split és a holdout nincs használva (`eval_test=false`). Egy futás csak akkor kész, ha a lépésszám a kért és minden eredmény véges. FLOP két számmal: `flops_per_token` (routolt) és `flops_per_token_executed` (bmm-paddinggel); a paraméterszám minden futásnál.
**Döntési szabály (minden páros összevetésre, a W1-gyel azonos, irányított):** Δ = változat − referencia; `SUPPORTED` (jobb) = Δ ≤ −0,010 **és** a 95%-os t-CI felső határa < 0 **és** 3/3 seed negatív; `NOT_SUPPORTED_IN_SCOPE` (rosszabb) = tükörképe; minden más `INCONCLUSIVE` (nem egyenértékűség).

## 1. Horgonyok (új hash; azonnal indulhatnak)

| ID | Preset / beállítás | Cél |
|---|---|---|
| B_moe | `moe_f5` | MoE-referencia |
| B_dense | `dense_f5` | dense-referencia |
| B_fixed | `amllm_f5 halt_rule=none` | FD-referencia |
| A0 | `amllm_f5` (survival, λ 0,02) | AD-referencia |

Ellenőrzés: a B_moe és B_fixed átlaga a W1-étől legfeljebb ±0,03 nat-tal tér el (a kódváltozás alapértelmezésben bitre semleges, ez a GPU-független reprodukció). 4 × 3 = **12 futás**.

## 2. FD04 — kell-e hoponként külön attention (H02)

A B_fixed hoponként külön attention-t használ (`unshared_attention=true`). Referencia: B_fixed. Minden sor kevesebb paramétert és (FD04d) kevesebb FLOP-ot használ; a kérdés, mennyi minőség vész el.

| ID | `--set` (mind `halt_rule=none`) | FLOP× / param× B_moe | Kérdés |
|---|---|---|---|
| FD04a | `unshared_attention=false` | 1,001 / 0,827 | egy közös attention minden hopnak |
| FD04b | `unshared_attention=false attn_groups=3` | 1,001 / 0,896 | 3 attention-súlykészlet (hopok 2-esével) |
| FD04c | `unshared_attention=false attn_adapter_rank=32` | 1,012 / 0,834 | közös attention + hoponkénti r=32 adapter |
| FD04d | `unshared_attention=false attn_every=2 attn_groups=3` | 0,794 / 0,896 | minden második hop csak expert, az attention-hopoknak saját súly |

Értelmezés: `SUPPORTED` → a megosztott / ritkított változat jobb; `NOT_SUPPORTED_IN_SCOPE` → a hoponként külön attention számít; `INCONCLUSIVE` → ezen a felbontáson nem mutatható ki különbség, ami **nem** egyenértékűség (n = 3 mellett a CI széles). Az FD04d a FLOP-ot ~0,79×-re csökkenti: a B_moe-hoz mért minőségarányát és FLOP-arányát leíróan jelentjük. 4 × 3 = **12 futás**.

## 3. FD05 — illesztett minőség–számítás pontok (H03/H04/H05) és a W1-kombináció

Az illesztés a `tools/fd05_match.py` valós előrefutásaiból (routolt FLOP × B_moe / paraméter × B_moe). **Minden „azonos / illesztett FLOP” ebben a dokumentumban a routolt munkára vonatkozik**; a végrehajtott FLOP (bmm-padding) változatonként eltér, ezért minden összevetésnél külön jelentjük, és egyik ítélet sem állít végrehajtott-munka-egyezést.

| ID | Preset, `--set` | FLOP× / param× | Referencia | Kérdés |
|---|---|---|---|---|
| FD05a | `amllm_f5 halt_rule=none n_hops=8 expert_hidden=160 experts_per_module=8` | 1,003 / 1,001 | B_fixed | H05: 8 hop azonos routolt FLOP-on és paraméteren |
| FD05b | `amllm_f5 halt_rule=none n_hops=4 expert_hidden=832 experts_per_module=2` | 1,000 / 0,965 | B_fixed | H05: 4 hop (moduljain belül nincs expertválasztás) |
| FD05c | `amllm_f5 halt_rule=none module_topk=2 expert_topk=1` | 1,002 / 1,000 | B_fixed | H03: top-2 modul / top-1 expert |
| FD05d | `amllm_f5 halt_rule=none module_topk=2 expert_hidden=192` | 1,002 / 0,793 | B_fixed | H03: top-2/top-2 (paraméter nem illeszthető) |
| FD05e | `amllm_f5 halt_rule=none experts_per_module=8 expert_hidden=192` | 0,847 / 1,001 | **FD05e_moe**, B_fixed | H04: finomabb expertek 0,85×-ön |
| FD05e_moe | `moe_f5 n_experts=8 expert_hidden=192` | 0,846 / 1,001 | — | illesztett MoE-kontroll |
| FD05f | `amllm_f5 halt_rule=none expert_hidden=208` | 0,859 / 0,810 | **FD05f_moe**, B_fixed | H04: kisebb expertek 0,86×-on |
| FD05f_moe | `moe_f5 expert_hidden=208` | 0,858 / 0,811 | — | illesztett MoE-kontroll |
| FD05f_dense | `dense_f5 ffn_hidden=416` | 0,857 / 0,698 | — | illesztett dense-kontroll |
| FDC1 | `amllm_f5 halt_rule=none bank_groups=3 n_modules=1 experts_per_module=24 module_topk=1` | 1,004 / 1,828 | B_fixed, B_moe (leíró) | W1-kombináció: részleges bankmegosztás + lapos routing |

- Az FD05e − FD05e_moe és FD05f − FD05f_moe az **A szempontjából releváns** pár (azonos routolt FLOP, azonos paraméter): ugyanazzal a döntési szabállyal. A végrehajtott FLOP-ot mellé írjuk, A-sikert ez sem ad (az A1 szerint a végrehajtott munka W3-mérést igényel).
- Az FDC1 paramétere 1,83× (külön bankok); ezt minden sorban feltüntetjük, nem azonos paraméterű összevetés.
- 10 × 3 = **30 futás**.

## 4. AD — anytime-súly és adaptív K (AD03)

Mind `amllm_f5`, survival halting, λ 0,02. Referencia: A0 (W2-horgony).

| ID | `--set` | Kérdés |
|---|---|---|
| AD02a-3 (könyvtár: `AD02a_3`) | `anytime_weight=0.5` | köztes súly: kemény út vs. végső minőség |
| AD03a | `anytime_weight=0.5 expert_topp=0.5` | adaptív K (a második expert csak ha az első < 0,5 valószínűségű) |
| AD03b | `anytime_weight=0.5 expert_topp=0.7` | adaptív K, enyhébb |
| AD03c | `anytime_weight=0.5 expert_topk=1` | fix K = 1 kontroll (az adaptív K alsó határa) |

- Elsődleges kapu (az A1 szűrőjele, változatlan): kemény θ 0,5 val NLL ≤ 1,03 × B_moe **és** routolt FLOP ≤ 0,90 × B_moe, 3/3 seeden → `PROMISING_EXPLORATORY`; sikert csak a W3 végrehajtási mérése adhat.
- Páros összevetések: AD02a-3 − A0; AD03a/b − AD02a-3 (mit ad az adaptív K azonos súly mellett); AD03a/b − AD03c **a FLOP-ot is jelentve** (az adaptív K akkor értékes, ha az átlagos expertszám és a FLOP az AD03c-hez közel, a minőség az AD02a-3-hoz közel van).
- Másodlagos: az AD01-diagnosztika (hoponkénti kiolvasás, kemény–soft rés) mind a négy változatra és az A0-ra. A tanult kilépés kontrollja a **valódi véletlen megállítás** (`random_exit_matched_nll_hard`: kemény futás, a tokenek a kemény út kilépési hopjainak összekevert eloszlása szerint állnak meg); a soft nyomból vett kiolvasás (`…_from_soft_trace`) csak közelítés, következtetés nem épül rá.
- Az adaptív K elhagyott útvonalai a nyomban −1-ként szerepelnek; a routerdiagnosztika ezeket kihagyja, a MoE aktívparaméter-száma a ténylegesen kiküldött útvonalakból számol.
- 4 × 3 = **12 futás**.

## 5. Terjedelem, költség, leállás
- 12 horgony + 12 FD04 + 30 FD05 + 12 AD = **66 futás**; a W1 podonként mért futásidejei alapján kiegyensúlyozott sorok (`queues/w2a.txt` 27, `w2b.txt` 27, `w2c.txt` 12 futás, a lassabb A40 kevesebbet kap): becsült falióra **~7,5–8 óra**, ~**$5,5** ($0,68/óra a három podra).
- A gépek a W2 alatt nem állnak le; a sor podoldali (`worker.sh`), futás közben kiegyensúlyozható. A slotok neve w2a/w2b/w2c (külön podlog-könyvtár). Minden kész futás ~2 percen belül a vps2-re mentődik checkpointtal, SHA-256-tal ellenőrizve.
- Lezárás: a kiértékelés (döntési tábla, AD01-diagnosztika) a podokon, utána záró mentés; a gépek sorsáról a tulajdonos dönt.

## 6. Nem része a W2-nek
Holdout, teszt-adat, 18k futások, OOD, decode/VRAM-mérés, padding nélküli végrehajtási mérés (W3), AD04 (routing × kilépés), bármilyen kódváltozás a futások alatt.
