---
title: "W1 eredmények — horgonyok, FixedDepth-tényezők (FD01–FD03, FD07), anytime-veszteség (AD02a)"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "RESULT"
source_of_truth: true
supersedes: []
superseded_by: []
---

[English](W1_REPORT.md) | **Magyar**

# W1 eredmények — horgonyok, FixedDepth-tényezők (FD01–FD03, FD07), anytime-veszteség (AD02a)

**Állapot:** a W1 végleges eredménye. Az [előregisztráció](PREREGISTRATION.hu.md) és kiegészítései ([A1](PREREGISTRATION_A1.hu.md), [A2](PREREGISTRATION_A2.hu.md), [A3](PREREGISTRATION_A3.hu.md)) mind a **48 futása** lefutott a kért 9000 lépéssel, minden eredmény véges, egyik sem hibázott. A kód végig `train_code_sha256` = `ba81b4fc02cd7b63`. Minden futáskönyvtár (checkpointtal együtt) a privát mentőszerveren van (futásonként SHA-256-tal ellenőrizve); a riportok, logok és elemzések ebben a könyvtárban. Teszt-split és holdout nincs használva.
**Protokoll:** `code_corpus_v2`, 9000 lépés, batch 32 × 512, seed 42/43/44, validációs NLL (nat/token) az alapmódban (`fixed`; halting modellnél `hard_surv_0.5`). Döntési szabály (előregisztrált, irányított): `SUPPORTED` = Δ ≤ −0,010, a 95%-os t-CI (n = 3, t = 4,303) teljesen 0 alatt és 3/3 seed negatív; `NOT_SUPPORTED_IN_SCOPE` = ennek tükörképe; minden más `INCONCLUSIVE` (ami nem egyenértékűség). Gépek: RunPod Community RTX 3090 Ti / RTX 3090 / RTX A4000 (a GPU-típus futásonként rögzítve, döntésbe nem számít).

## 1. Horgonyok (az ellenőrzés teljesült)

| Konfig | val NLL átlag | seed 42 / 43 / 44 | paraméter | routolt MFLOP/token | végrehajtott MFLOP/token | hop |
|---|---|---|---|---|---|---|
| B_moe (`moe_f5`) | **2,1185** | 2,1070 / 2,1311 / 2,1172 | 17,15M | 22,83 | 25,02 | 6 |
| B_dense (`dense_f5`) | 2,1253 | 2,1320 / 2,1157 / 2,1281 | 13,59M | 22,81 | 22,81 | 6 |
| B_fixed (`amllm_f5`, halting nélkül) | 2,1585 | 2,1574 / 2,1568 / 2,1612 | 17,15M | 22,86 | 31,56 | 6 |
| A0 (`amllm_f5`, survival λ 0,02), kemény θ 0,5 | 3,3821 | 3,4576 / 3,4105 / 3,2782 | 17,15M | 19,94 | 27,22 | 3,54 |

A B_moe és a B_fixed a korábbi értékektől (2,1166 / 2,1625) +0,002 / −0,004 nat-tal tér el, a ±0,03-as tűrésen belül: a horgonyok reprodukálódnak. Az A0 soft-módú NLL-je 2,187; a kemény útja a korábbiakhoz hasonlóan jóval rosszabb.

## 2. FD-ág — melyik összemosott tényező rontja a minőséget

Páros Δ = változat − referencia (nat/token), 3 seed.

| Kérdés | Változat | paraméter | Δ | 95% CI | Ítélet |
|---|---|---|---|---|---|
| FD01a külön bank minden hopnak (hoponként 24-es készlet) | 2,1221 | 52,69M | −0,0363 | [−0,0493; −0,0234] | **SUPPORTED** (jobb) |
| FD01b külön bankok, **azonos össz-paraméter** (hoponként 4-es készlet) | 2,1384 | 17,15M | −0,0200 | [−0,0296; −0,0105] | **SUPPORTED** — csomaghatás (A1) |
| FD01c részleges megosztás (3 bank) | 2,1345 | 31,37M | −0,0240 | [−0,0440; −0,0040] | **SUPPORTED** |
| FD01d kontroll: egy közös 4-expertes bank | 2,2359 | 11,23M | +0,0774 | [+0,0482; +0,1066] | NOT_SUPPORTED_IN_SCOPE (rosszabb) |
| FD02a lapos 24-utas router | 2,1251 | 17,15M | −0,0334 | [−0,0430; −0,0237] | **SUPPORTED** |
| FD02b véletlen routing (a router értékének kontrollja) | 2,2874 | 17,15M | +0,1289 | [+0,1192; +0,1386] | NOT_SUPPORTED_IN_SCOPE (rosszabb) |
| FD03a hoponkénti bank-LayerNorm | 2,1611 | 17,15M | +0,0026 | [−0,0079; +0,0132] | INCONCLUSIVE |
| FD03b közös LN + hoponkénti skála/eltolás | 2,1624 | 17,15M | +0,0040 | [−0,0050; +0,0129] | INCONCLUSIVE |

**FD01 2×2 (A1: oksági állítás csak azonos hoponkénti készleten belül; egyik párnak sem azonos az össz-paramétere):**

| Pár | Mit különít el | Δ | 95% CI | Ítélet | paraméter |
|---|---|---|---|---|---|
| FD01b − FD01d | megosztás, 4-es készlet | −0,0974 | [−0,1353; −0,0596] | **SUPPORTED** (külön jobb) | 17,15M vs 11,23M |
| FD01a − B_fixed | megosztás, 24-es készlet | −0,0363 | [−0,0493; −0,0234] | **SUPPORTED** (külön jobb) | 52,69M vs 17,15M |
| B_fixed − FD01d | készletméret, közös bank | −0,0774 | [−0,1066; −0,0482] | **SUPPORTED** (24 jobb, mint 4) | 17,15M vs 11,23M |
| FD01a − FD01b | készletméret, külön bankok | −0,0163 | [−0,0368; +0,0042] | INCONCLUSIVE | 52,69M vs 17,15M |

**FD07 router-tanítási szabály (A3, egy architektúrán belül):**

| Pár | Δ | 95% CI | Ítélet |
|---|---|---|---|
| FD07a MoE: `straight_through` − `topk_softmax` (B_moe_st 2,1335 − B_moe) | +0,0150 | [−0,0356; +0,0656] | INCONCLUSIVE (seedek: +0,022 / −0,008 / +0,031) |
| FD07b AM: `full_softmax` − `straight_through` (B_fixed_fs 2,1653 − B_fixed) | +0,0068 | [+0,0004; +0,0132] | INCONCLUSIVE (|Δ| a 0,010 alatt) |

**Bankonkénti expert-használat (A2; `tools/analysis/bank_usage.py`, validáció, 3-3 seed)** — az FD01a/b/c riportok `expert_counts`-ből számolt mezői érvénytelenek (összevont bankok), ezért nem idézzük őket; a helyes értékek:

| Futás | bankok | különálló expertek | halott (seedenként) | entrópia (bit) / max | max. részesedés | legkiegyensúlyozatlanabb bank (entrópia / max) |
|---|---|---|---|---|---|---|
| FD01a | 6 | 144 | 2 / 0 / 0 | 6,90 / 7,17 | 0,021 | 0,91 |
| FD01b | 6 | 24 | 0 / 0 / 0 | 4,57 / 4,58 | 0,050 | 0,98 |
| FD01c | 3 | 72 | 0 / 0 / 0 | 5,94 / 6,17 | 0,035 | 0,92 |

**Az FD-ág értelmezése (érvényességi kör: 17M paraméter, ez a korpusz, 9k lépés):**
- A négy összemosott AM–MoE-különbségből kettő mérhetően rontja a minőséget: **az expertek hopok közötti megosztása** (FD01) és **a hierarchikus routing** (FD02a: a lapos 24-utas router 0,033-mal jobb azonos paraméter és azonos **routolt** FLOP mellett; a végrehajtott FLOP nem azonos — bmm-paddinggel 38,45 vs 31,56 MFLOP). A 2×2 szerint rögzített hoponkénti készletnél a külön bankok jobbak, közös banknál pedig a nagyobb készlet jobb; az azonos paraméterű FD01b csomaghatás.
- A másik kettőnél nem mutatható ki hatás: **a bank-normalizálás** (FD03a/b) és **a router-tanítási szabály** (FD07a/b) INCONCLUSIVE. Ez **nem** bizonyítja, hogy nem számítanak: n = 3 mellett a CI-k még −0,008 és +0,013 (FD03a), −0,005 és +0,013 (FD03b), −0,036 és +0,066 (FD07a), illetve +0,0004 és +0,013 (FD07b) nat közötti hatást is megengednek. Az FD07b CI-je 0 fölött van, de a hatás (+0,007) az előregisztrált 0,010-es küszöb alatt marad.
- A tanult routing számít: a véletlen routing 0,129-cel rosszabb (FD02b).
- Az FD01 nyereségét nem a közös bank halott expertjei magyarázzák: a külön bankok minden expertet (a 144-ből kettő kivételével mindet) közel egyenletesen használnak.
- A legjobb FixedDepth-változatok (FD01a 2,1221, FD02a 2,1251) elérik a dense kontrollt (2,1253), de a B_moe-t (2,1185) nem. A „jobb a B_fixednél” nem A-eredmény: az A-hoz végrehajtott-FLOP-illesztett összevetés kell (FD05, W2). A végrehajtott FLOP minden AM-változatnál nagyobb a routoltnál a bmm-padding miatt (pl. FD02a 38,45 vs 22,93 MFLOP).

## 3. AD-ág — anytime (több kilépéses) veszteség

| Konfig | kemény θ 0,5 NLL | seedek | hop | routolt MFLOP | végrehajtott MFLOP | soft NLL | Δ az A0-hoz (kemény) | 95% CI | Ítélet |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 3,3821 | 3,458 / 3,411 / 3,278 | 3,54 | 19,94 | 27,22 | 2,187 | — | — | — |
| AD02a-1 (súly 0,3) | 2,6834 | 2,716 / 2,673 / 2,661 | 3,42 | 19,80 | 27,55 | 2,233 | −0,6987 | [−0,8740; −0,5234] | **SUPPORTED** |
| AD02a-2 (súly 1,0) | 2,5041 | 2,570 / 2,509 / 2,433 | 3,42 | 19,80 | 28,14 | 2,342 | −0,8780 | [−0,9505; −0,8055] | **SUPPORTED** |

**Elsődleges kapu (A1 szűrőjel):** kemény θ 0,5 NLL ≤ 1,03 × B_moe **és** routolt FLOP ≤ 0,90 × B_moe, 3/3 seeden. Minőségarány: AD02a-1 **1,267×**, AD02a-2 **1,182×** (A0 1,596×); routolt FLOP-arány 0,867×. **0/3 seed teljesíti: nem `PROMISING_EXPLORATORY`.** A végrehajtott FLOP (bmm-paddinggel) a B_moe végrehajtottjának 1,10–1,12-szerese, vagyis ezzel a dispatch-csel nincs végrehajtott megtakarítás.

**Másodlagos diagnosztika (az AD01 újrafuttatása, `tools/analysis/ad01_trace.py`, 128 validációs ablak, a nyom mind a 9 checkpointon pontosan egyezik a forwarddal):**

| | soft kiolvasás hoponként 1→6 | oracle\* | kemény θ 0,1 | θ 0,3 | θ 0,5 | θ 0,8 |
|---|---|---|---|---|---|---|
| A0 | 6,17 · 4,70 · 3,99 · 3,05 · 2,45 · **2,17** | 1,86 | 2,17 @ 5,90 | 2,28 @ 5,13 | 3,33 @ 3,54 | 6,17 @ 1,00 |
| AD02a-1 | 4,08 · 3,32 · 2,94 · 2,35 · 2,27 · **2,22** | 1,92 | 2,22 @ 5,59 | 2,28 @ 4,53 | 2,64 @ 3,42 | 3,82 @ 1,35 |
| AD02a-2 | 3,62 · 3,13 · 2,64 · 2,38 · 2,34 · **2,32** | 2,03 | 2,33 @ 5,08 | 2,36 @ 4,10 | 2,47 @ 3,42 | 3,05 @ 2,17 |

\* tokenenként a legjobb hop a célbájt ismeretében: analitikai korlát, nem policy.

Párosan az A0-hoz (3 seed): az 5. hop kiolvasásának távolsága a végső soft értéktől 0,15–0,39-ről 0,05-re (AD02a-1) és 0,01–0,02-re (AD02a-2) csökken — **3/3 seeden**, de az irányított CI-szabály szerint INCONCLUSIVE, mert az A0 seedenként erősen szór. A kemény–soft rés 0 alatti CI-vel csökken θ 0,3-nál (−0,047 / −0,073) és θ 0,5-nél (−0,733 / −1,009): mindkét súlyra **SUPPORTED**. A tanult kilépés kontrollja a **valódi véletlen megállítás**: kemény futás, amelyben minden token a kemény út kilépési hopjainak a tokenek között összekevert értékénél áll meg (azonos kilépési eloszlás; `random_exit_matched_nll_hard` az `analysis/` alatt). A tanult kilépés θ 0,1-nél, 0,3-nál és 0,5-nél jobb az A0-nál, az AD02a-1-nél és az AD02a-2-nél is (párosan, 3 seed, mind **SUPPORTED**; pl. θ 0,5: A0 3,334 vs 3,640, AD02a-1 2,645 vs 2,769, AD02a-2 2,475 vs 2,565); θ 0,8-nál INCONCLUSIVE. A bekezdés korábbi változata a *soft* nyom kiolvasását használta kontrollként a véletlen hopnál; ez nem megállított futás (a későbbi tokenek nem befagyasztott állapotokra figyelnek). A valódi megállított futástól legfeljebb 0,031 nat-tal tér el (A0, θ 0,5), és nem épül rá következtetés.

**Az AD-ág értelmezése:** az anytime-veszteség azt teszi, amire szántuk — a köztes állapotok kiolvashatók, a kemény út nem omlik össze (θ 0,3 az A0 minőségét 5,1 helyett 4,5 hoppal adja). Ennek ára a végső minőség (soft 2,22 / 2,32 vs 2,17), és a B_moe kb. 1,05×-ösén belüli kemény pontokhoz ≥ 5,6 átlagos hop kell (AD02a-1 θ 0,1: 1,048× 5,59 hoppal; A0 θ 0,1: 1,026× 5,90 hoppal), ahol alig marad megtakarítás. Az AD-ág tehát él, de messze van az A céltól; a következő lépés a hopnál finomabb szelektív számítás (AD03) és egy 0,3 és 1,0 közötti súly, amely megtartja a végső minőséget.

## 4. Amit a W1 nem állapít meg
- Nincs A-bizonyíték: még nincs végrehajtott-FLOP-illesztett összevetés (FD05, W2) és padding nélküli végrehajtási mérés (W3).
- Egy skála (17M), egy korpusz, 9k lépés, n = 3: az INCONCLUSIVE sor nem „nincs hatás”.
- Az FD01 azonos paraméterű ítélete (FD01b) csomaghatás; az FD01a háromszoros paramétert használ.

## 5. Következmények a W2-re (javaslat, a W2 előregisztrációja dönt)
- **Az FD04/FD05 a javított FixedDepth-re épül:** az attention-kérdéseket (FD04) és az illesztett minőség–számítás pontokat (FD05: 8 hop 1,00× FLOP-on, top-2, H04 0,85×-ön) az alapon *és* a legjobb W1-tényezőkombináción (külön / részben külön bankok + lapos routing) is mérjük, mindig FLOP- és paraméterillesztett MoE/dense kontrollal szemben.
- **Az FD01c+FD02a kombináció** (részleges bankmegosztás lapos routinggal) kézenfekvő jelölt; hogy a két nyereség összeadódik-e, nincs tesztelve.
- **Az FD03 és az FD07 nem megy tovább** elsődleges kérdésként — ez költségvetési sorrend, nem annak megállapítása, hogy nincs hatásuk (INCONCLUSIVE); nyitva maradnak, eldöntésükhöz több seed kellene.
- **AD:** AD03 (adaptív K, hopnál finomabb szelektív számítás) az anytime-veszteségre építve; 0,5 körüli anytime-súly kompromisszumként a kemény út és a végső minőség között.

## Fájlok
- `*/amllm_f5_s4?/` stb.: futásonként `config.json`, `env.json`, `report.json`, `train.log` (checkpoint csak a vps2-n); `*/summary.csv`.
- `analysis/`: `bank_usage__*.json` (A2), `ad01_trace__*.json` (AD01-újrafuttatás), podonkénti `STATUS_*`.
- `queues/`, `_queue_w1*/`: a lefutott sorok és a worker-logok.
