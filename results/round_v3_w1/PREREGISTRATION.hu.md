---
title: "W1 előregisztráció (tervezet, tulajdonosi jóváhagyásra vár)"
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

# W1 előregisztráció (tervezet, tulajdonosi jóváhagyásra vár)

**Állapot:** JÓVÁHAGYVA ÉS LEZÁRVA — a tulajdonos jóváhagyta 2026-10-23-án (a javított döntési szabályokkal és a kódverzióval, ahogy lent áll). A kapuk számértékei ettől kezdve nem módosíthatók; a fájl tartalmát a `PREREGISTRATION.sha256` rögzíti. A horgony-futások (1. rész) kontrollok és a jóváhagyás előtt indultak; a 2–3. rész futásai a jóváhagyás után indultak.

**Kódverzió (befagyasztva W1 alatt):** `train_code_sha256` = `ba81b4fc02cd7b63` (`config.py`, `data.py`, `model.py`, `routing.py`, `train.py`); a tanítókód ettől nem változhat. A korábbi 406c4eef88f4ba69, e361ca5da3bf54ed és 83339a48c73ce5c0 hash alatti futásokkal nem hasonlítunk (R2). Egy futás csak akkor kész, ha a tényleges lépésszám megegyezik a kértel és minden eredmény véges (a NaN-os tanítás hibával áll le, FAILED.json jelzőfájlt hagy és nem ír report.json-t). (Az e361… verzió még nem tanított semmit: a felülvizsgálat után javítottuk a teszt-split kiértékelését és a FLOP-jelentést.)
**Adat/protokoll:** `code_corpus_v2` (split-id `355afd73091925ef`), 9 000 lépés, batch 32 × 512, lr 1e-3, a korábbi F5-protokoll; seed 42, 43, 44; validáció a döntéshez; **a teszt-split és a holdout nem használt: a futás nem értékeli és nem naplózza a teszt-splitet (`eval_test=false`, a `report.json`-ban `test_default_mode: null`; tesztelve)**. Minden futás rögzíti a GPU-t; a GPU-típus a tanítás veszteségét nem befolyásolja (lásd RESEARCH_TASKS), az időmérést itt nem használjuk.
**Elsődleges mutató:** validációs NLL (nat/token) az alapértelmezett módban (`fixed`; halting-modellnél `hard_surv_0.5`), páros (azonos seedű) különbség, 95% t-CI (n = 3). A FLOP két számmal jelentett: `flops_per_token` (az útválasztott munka) és `flops_per_token_executed` (a ténylegesen végrehajtott, a `bmm` dispatch padding-soraival együtt; a tanítás és a kiértékelés `bmm`-et használ). A paraméterszám minden futásnál jelentve. n = 3 mellett a CI széles: ami a 0-t tartalmazza, az **INCONCLUSIVE**, nem ekvivalencia.

## 1. Horgony és kontroll (azonnal indul)

| ID | Preset / beállítás | Seed |
|---|---|---|
| B_moe | `moe_f5` | 42, 43, 44 |
| B_dense | `dense_f5` | 42, 43, 44 |
| B_fixed | `amllm_f5`, `halt_rule=none` (FixedDepth alap) | 42, 43, 44 |
| A0 | `amllm_f5` (jelenlegi survival-halting, λ 0,02) — az L1 reprodukciója az új kódon | 42, 43, 44 |

Cél: új hash alatti horgonyok és az A0 kontroll. Kapu nincs; ellenőrzés: a B_moe és B_fixed átlaga a korábbi 2,1166 / 2,1625 értékektől legfeljebb ±0,03 nat-tal tér el (a kód- és GPU-függetlenség sanity-checkje; nagyobb eltérés esetén a futásokat nem használjuk, hanem okot keresünk).

## 2. FD-sáv (FixedDepth; minden variáns `amllm_f5`, `halt_rule=none`, ha nincs más megadva)

Minden kérdés **egy** tényezőt változtat; a referencia a B_fixed (shared bank, 6×4, Top-1/Top-2, saját hoponkénti attention, közös bank-norm).

| ID | Variáns (`--set`) | Teljes paraméter | MFLOP/token | Kérdés |
|---|---|---|---|---|
| FD01a | `bank_groups=6` | 52,69M | 22,857 | azonos aktív FLOP, 3× paraméter: segít-e a külön bank? |
| FD01b | `bank_groups=6 n_modules=2 experts_per_module=2` | 17,15M | 22,830 | **azonos teljes paraméter** (24 szakértő), külön bank hoponként |
| FD01c | `bank_groups=3` | 31,37M | 22,857 | részleges megosztás |
| FD02a | `n_modules=1 experts_per_module=24 module_topk=1` | 17,15M | 22,926 | lapos 24-es router a hierarchikus ellen |
| FD02b | `router_mode=random` | 17,15M | 22,857 | véletlen routing (tanítási kontroll) |
| FD03a | `bank_norm=per_hop` | 17,15M | 22,857 | hoponkénti bank-LayerNorm |
| FD03b | `bank_norm=hop_affine` | 17,15M | 22,857 | közös LN + hoponkénti skála/eltolás |

**Döntési szabály (kérdésenként, páros Δ = variáns − B_fixed, nat/token):**
Az irány kötött (a kisebb veszteség jobb):
- `SUPPORTED` (a variáns **jobb**): Δ ≤ −0,010 (≈ egy seed szórása) **és** a 95%-CI **felső határa < 0** (a CI teljes egészében nulla alatt) **és** mind a három seed Δ-ja negatív;
- `NOT_SUPPORTED_IN_SCOPE` (a variáns **rosszabb**): Δ ≥ +0,010 **és** a 95%-CI **alsó határa > 0** **és** mind a három seed Δ-ja pozitív;
- egyébként `INCONCLUSIVE`.
Egy szignifikánsan rosszabb variáns tehát soha nem lehet `SUPPORTED`.
Az FD01a nem egyedül döntő, mert paraméterben nem illesztett: az FD01b az azonos paraméterű ítélet. Az FD02b (random) csak a router értékét méri, nem versenyző jelölt. A „javult” állítás nem jelent A-sikert: az A-hoz végrehajtott-FLOP-illesztett összevetés kell (FD05, W2).

## 3. AD-sáv (AdaptiveDepth; `amllm_f5` survival-halting, λ 0,02)

| ID | Variáns (`--set`) | Referencia |
|---|---|---|
| AD02a-1 | `anytime_weight=0.3` | A0 |
| AD02a-2 | `anytime_weight=1.0` | A0 |

**Döntési szabály:**
- Elsődleges (az A-cél felé): a **hard_surv_0.5** módban a val NLL ≤ 1,03 × a B_moe átlaga **és** az útválasztott `flops_per_token` ≤ 0,90 × a B_moe-é (≥ 10% megtakarítás), 3/3 seeden. A `flops_per_token_executed` (bmm-padding-gel) **nem dönt**, de kötelezően jelentett a padding-aránnyal együtt, mert a bmm-padding a dispatch tulajdonsága, nem az architektúráé; a megtakarítás csak akkor számít valódinak, ha létezik padding-mentes végrehajtási út (F3-eredmény, W3-ban ellenőrzendő) — addig az ítélet legfeljebb `SUPPORTED_IN_SCOPE (routed FLOP)`; ez `SUPPORTED_IN_SCOPE` az AD-ágra, nem teljes A-bizonyíték.
- Másodlagos (diagnosztikai): az AD01 mérőszámai újrafuttatva (hoponkénti kiolvasás, kemény–soft rés, küszöb-görbe); javulás, ha az 5. hop kiolvasott NLL-je a végsőhöz közelebb kerül és a θ = 0,3 / 0,5 pont a soft értékhez közelebb marad, mint A0-nál (páros, 3 seed).
- Egyébként `INCONCLUSIVE` / `NOT_SUPPORTED_IN_SCOPE`. Az AD-ág eredménye nem következtet a FixedDepth-re és fordítva.

## 4. Terjedelem, költség, leállítás
- 12 horgony + 21 FD-futás + 6 AD-futás = **39 futás** (≈ 11–12 GPU-óra 3090-en, ≈ $3–4 a mért árakon).
- A gépek a W1 alatt nem állnak le; a sor a pod-oldali worker (`tools/runpod/worker.sh`) fájlsora, amelyhez futás közben is fűzhető. A költség a foglalási idő × óradíj; a W1 plafonja a $20 javasolt körplafonon belül marad.
- A lezárás: minden eredmény a vps2-re és a Macre mentve, SHA ellenőrizve; a további kérdések (FD04, FD05, AD03) a W2-ben.

## 5. Nem része a W1-nek
Holdout, tesztadat, 18k-s futások, FD04/FD05, AD03/AD04, decode/VRAM-mérés, új kódváltozás (W1 alatt a tanítókód fagyasztott).
