---
title: "W2 eredmények — attention-megosztás (FD04), illesztett minőség–számítás pontok (FD05), a W1-kombináció (FDC1), anytime-súly és adaptív K (AD02a-3, AD03)"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "RESULT"
source_of_truth: true
supersedes: []
superseded_by: []
---

[English](W2_REPORT.md) | **Magyar**

# W2 eredmények — attention-megosztás (FD04), illesztett minőség–számítás pontok (FD05), a W1-kombináció (FDC1), anytime-súly és adaptív K (AD02a-3, AD03)

**Állapot:** a W2 végleges eredménye. A [lezárt előregisztráció](PREREGISTRATION.hu.md) mind a **66 futása** lefutott 9000 lépéssel, minden eredmény véges, egyik sem hibázott. A kód végig `train_code_sha256` = `cd909ea282660f0f`. Minden futáskönyvtár (checkpointtal együtt) a privát mentőszerveren van (futásonként SHA-256-tal ellenőrizve); a riportok, logok és elemzések ebben a könyvtárban. Teszt-split és holdout nincs használva. Gépek: RunPod Community RTX 3090 Ti, RTX 3090 és RTX A4000 (az előregisztráció a harmadikat „A40”-nek nevezi — **RTX A4000** volt; ez csak a sorok kiegyensúlyozását írja le, döntésbe nem számít).
**Protokoll és szabály:** az előregisztrált — `code_corpus_v2`, 9000 lépés, batch 32 × 512, seed 42/43/44, validációs NLL (nat/token) az alapmódban (`fixed`; halting modellnél `hard_surv_0.5`); `SUPPORTED` = Δ ≤ −0,010, a 95%-os t-CI (n = 3) teljesen 0 alatt és 3/3 seed negatív; `NOT_SUPPORTED_IN_SCOPE` = tükörképe; minden más `INCONCLUSIVE`, ami **nem** bizonyítja a hatás hiányát. **Az „azonos / illesztett FLOP” mindig routolt FLOP**; a végrehajtott FLOP (bmm dispatch, paddinggel) mellette szerepel, és eltér.

## 1. Horgonyok — ±0,0015-ön belül reprodukálják a W1-et

| Konfig | val NLL | seed 42 / 43 / 44 | W1 | paraméter | routolt MFLOP | végrehajtott MFLOP |
|---|---|---|---|---|---|---|
| B_moe | **2,1180** | 2,1120 / 2,1242 / 2,1177 | 2,1185 | 17,15M | 22,83 | 24,87 |
| B_dense | 2,1254 | 2,1328 / 2,1143 / 2,1291 | 2,1253 | 13,59M | 22,81 | 22,81 |
| B_fixed | 2,1599 | 2,1600 / 2,1577 / 2,1620 | 2,1585 | 17,15M | 22,86 | 31,60 |
| A0 (kemény θ 0,5) | 3,3828 | 3,4404 / 3,4393 / 3,2687 | 3,3821 | 17,15M | 19,95 | 27,32 |

## 2. Az összes konfiguráció

Az arányok a B_moe-hoz (NLL, routolt FLOP, végrehajtott FLOP).

| Konfig | val NLL | seedek | paraméter | routolt / végrehajtott MFLOP | NLL× | routolt× | végr.× |
|---|---|---|---|---|---|---|---|
| FD04a közös attention | 2,3337 | 2,3154 / 2,3433 / 2,3425 | 14,19M | 22,86 / 32,27 | 1,102 | 1,001 | 1,298 |
| FD04b 3 attention-készlet | 2,1796 | 2,1830 / 2,1882 / 2,1676 | 15,37M | 22,86 / 32,93 | 1,029 | 1,001 | 1,324 |
| FD04c közös + r=32 adapter | 2,2914 | 2,2871 / 2,2992 / 2,2880 | 14,31M | 23,10 / 32,44 | 1,082 | 1,012 | 1,304 |
| FD04d minden 2. hop csak expert | 2,2738 | 2,2521 / 2,3078 / 2,2615 | 15,37M | 18,14 / 29,22 | 1,074 | 0,794 | 1,175 |
| FD05a 8 hop | 2,1677 | 2,1746 / 2,1757 / 2,1527 | 17,17M | 22,90 / 31,21 | 1,023 | 1,003 | 1,255 |
| FD05b 4 hop | 2,2342 | 2,2543 / 2,2204 / 2,2277 | 16,55M | 22,83 / 29,23 | 1,055 | 1,000 | 1,175 |
| FD05c top-2 modul / top-1 expert | 2,1616 | 2,1647 / 2,1516 / 2,1684 | 17,15M | 22,88 / 31,53 | 1,021 | 1,002 | 1,268 |
| FD05d top-2 / top-2, expert 192 | 2,1984 | 2,2045 / 2,2041 / 2,1865 | 13,61M | 22,88 / 31,72 | 1,038 | 1,002 | 1,276 |
| FD05e finomabb expertek (8/modul, 192) | 2,1817 | 2,1679 / 2,1856 / 2,1916 | 17,17M | 19,34 / 27,26 | 1,030 | 0,847 | 1,096 |
| FD05e_moe illesztett MoE | 2,1280 | 2,1223 / 2,1213 / 2,1403 | 17,17M | 19,31 / 21,30 | 1,005 | 0,846 | 0,857 |
| FD05f kisebb expertek (208) | 2,2061 | 2,2012 / 2,2207 / 2,1965 | 13,90M | 19,61 / 24,14 | 1,042 | 0,859 | 0,970 |
| FD05f_moe illesztett MoE | 2,1622 | 2,1628 / 2,1650 / 2,1588 | 13,91M | 19,59 / 20,71 | 1,021 | 0,858 | 0,833 |
| FD05f_dense illesztett dense | 2,1676 | 2,1724 / 2,1661 / 2,1642 | 11,97M | 19,57 / 19,57 | 1,023 | 0,857 | 0,787 |
| **FDC1** 3 bank + lapos routing | **2,0870** | 2,0886 / 2,0842 / 2,0881 | **31,36M** | 22,93 / 34,23 | **0,985** | 1,004 | 1,376 |
| AD02a-3 anytime 0,5 (kemény θ 0,5) | 2,5237 | 2,4938 / 2,5427 / 2,5347 | 17,15M | 19,72 / 28,48 | 1,192 | 0,864 | 1,145 |
| AD03a + adaptív K, p 0,5 | 2,5593 | 2,5008 / 2,6324 / 2,5446 | 17,15M | 17,80 / 23,42 | 1,208 | 0,779 | 0,942 |
| AD03b + adaptív K, p 0,7 | 2,5409 | 2,4830 / 2,5968 / 2,5428 | 17,15M | 18,28 / 25,54 | 1,200 | 0,800 | 1,027 |
| AD03c + fix K = 1 | 2,6085 | 2,5389 / 2,6649 / 2,6217 | 17,15M | 17,68 / 20,97 | 1,232 | 0,774 | 0,843 |

## 3. FD04 — a hoponként külön attention számít

| Pár (a B_fixedhez) | Δ | 95% CI | Ítélet |
|---|---|---|---|
| FD04a egy közös attention | +0,1738 | [+0,1336; +0,2141] | NOT_SUPPORTED_IN_SCOPE (rosszabb) |
| FD04b 3 attention-készlet | +0,0197 | [−0,0120; +0,0513] | INCONCLUSIVE (mindhárom seed pozitív) |
| FD04c közös + hoponkénti adapter | +0,1315 | [+0,1101; +0,1530] | NOT_SUPPORTED_IN_SCOPE (rosszabb) |
| FD04d minden második hop csak expert | +0,1139 | [+0,0354; +0,1924] | NOT_SUPPORTED_IN_SCOPE (rosszabb) |

Az attention megosztása a hopok között 0,11–0,17 nat-ba kerül, hoponkénti adapterrel is; az attention kihagyása minden második hopon (0,79× routolt FLOP) 0,11 nat-ba — ez a MoE veszteségének 1,07-szerese. A három attention-készlet (hopok párban) n = 3 mellett nem különböztethető meg a hoponként külön attentiontől, de a CI +0,051 nat-ig enged romlást. A hoponkénti attention — amely a MoE-ban rétegenként szintén megvan — hordozza a minőség nagy részét; ez egyezik a korábbi kis modelles eredménnyel.

## 4. FD05 — illesztett minőség–számítás pontok; az A szempontjából releváns párok nem teljesülnek

| Pár | Δ | 95% CI | Ítélet |
|---|---|---|---|
| FD05a 8 hop − B_fixed (azonos routolt FLOP és paraméter) | +0,0078 | [−0,0292; +0,0447] | INCONCLUSIVE |
| FD05b 4 hop − B_fixed | +0,0742 | [+0,0309; +0,1176] | NOT_SUPPORTED_IN_SCOPE (rosszabb) |
| FD05c top-2/top-1 − B_fixed | +0,0016 | [−0,0153; +0,0185] | INCONCLUSIVE |
| FD05d top-2/top-2 − B_fixed (0,79× paraméter) | +0,0385 | [+0,0084; +0,0685] | NOT_SUPPORTED_IN_SCOPE (rosszabb) |
| FD05e finomabb expertek − B_fixed | +0,0218 | [−0,0081; +0,0517] | INCONCLUSIVE |
| FD05f kisebb expertek − B_fixed | +0,0462 | [+0,0094; +0,0831] | NOT_SUPPORTED_IN_SCOPE (rosszabb) |
| **FD05e − FD05e_moe** (A-releváns: 0,85× routolt FLOP, azonos paraméter) | **+0,0537** | [+0,0301; +0,0774] | **NOT_SUPPORTED_IN_SCOPE (rosszabb)** |
| **FD05f − FD05f_moe** (A-releváns: 0,86×, azonos paraméter) | **+0,0440** | [+0,0188; +0,0691] | **NOT_SUPPORTED_IN_SCOPE (rosszabb)** |
| FD05f − FD05f_dense (0,86×) | +0,0386 | [+0,0039; +0,0733] | NOT_SUPPORTED_IN_SCOPE (rosszabb) |

- **0,85× routolt FLOP mellett az AM-LLM 0,044–0,054 nat-tal rosszabb az azonos paraméterű MoE-nál**; az FD05f az illesztett dense kontrollnál is rosszabb. Az illesztett MoE keveset veszít az olcsóbbá tétellel (0,85×-ön 2,1280, 1,00×-on 2,1180); az AM-LLM többet. Ezen a skálán az A hipotézis tehát a 0,85×-es ponton sem támogatott (a H04 ennél a kialakításnál már nem `INCONCLUSIVE`, hanem `NOT_SUPPORTED_IN_SCOPE`).
- Mélység: a 8 hop azonos routolt FLOP és paraméter mellett nem különböztethető meg a 6 hoptól (a CI +0,045-ig enged); a 4 hop egyértelműen rosszabb.
- Top-K: a top-2 modul top-1 experttel nem különböztethető meg az alaptól (CI −0,015…+0,019, ami nem egyenértékűség); a top-2/top-2 rosszabb, de 0,79× paramétere is van, így a kettő nem választható szét.

## 5. FDC1 — a W1-kombinációnak a legkisebb a vesztesége a W2-ben, 1,83× paraméterrel

| Pár | Δ | 95% CI | Ítélet |
|---|---|---|---|
| FDC1 − B_fixed | −0,0729 | [−0,0762; −0,0696] | **SUPPORTED** (jobb) |
| FDC1 − B_moe (leíró, nem illesztett) | −0,0310 | [−0,0519; −0,0101] | 3/3 seeden jobb |

A részleges bankmegosztás (3 bank, mindegyiket két hop használja) lapos 24-utas routerrel **2,0870**-et ér el, a MoE (2,1180) és a dense (2,1254) alatt, azonos **routolt** FLOP mellett (1,004×). Ez nem A-eredmény: **1,83× paramétere** (31,36M) és **1,38× végrehajtott FLOP-ja** van a B_moe-hoz képest (34,23 vs 24,87 MFLOP; bmm-padding 72 experten). A hiányzó kontroll egy azonos paraméterű MoE azonos routolt FLOP-on (`moe_f5 n_experts=12`, top-2: 31,37M paraméter 1,002× routolt FLOP-on, a `tools/fd05_match.py`-jal mérve). Hogy a két W1-tényező összeadódik-e, a W2-ből egymagában nem olvasható ki: az egyes tényezőket a W1 mérte más kódverzióval, és hashek között nincs összevetés.

## 6. AD — 0,5-ös anytime-súly és adaptív K

| Pár | Δ | 95% CI | Ítélet |
|---|---|---|---|
| AD02a-3 − A0 (kemény θ 0,5) | −0,8591 | [−1,1352; −0,5829] | **SUPPORTED** (jobb) |
| AD03a (p 0,5) − AD02a-3 | +0,0355 | [−0,0811; +0,1522] | INCONCLUSIVE (−10% routolt FLOP) |
| AD03b (p 0,7) − AD02a-3 | +0,0171 | [−0,0659; +0,1001] | INCONCLUSIVE (−7% routolt FLOP) |
| AD03a − AD03c (fix K = 1) | −0,0492 | [−0,1096; +0,0111] | INCONCLUSIVE (+0,7% routolt FLOP) |
| **AD03b − AD03c** | **−0,0676** | [−0,0962; −0,0391] | **SUPPORTED** (jobb, +3,4% routolt FLOP) |
| AD03c − AD02a-3 | +0,0848 | [−0,0111; +0,1807] | INCONCLUSIVE |

**Elsődleges kapu (az A1 szűrőjele):** kemény θ 0,5 NLL ≤ 1,03 × B_moe és routolt FLOP ≤ 0,90 × B_moe, 3/3 seeden — **minden AD-változatnál 0/3** (NLL 1,19–1,23× MoE, 0,77–0,86× routolt FLOP mellett). Nem `PROMISING_EXPLORATORY`.

**Másodlagos (AD01, `tools/analysis/ad01_trace.py`, 128 validációs ablak, a nyom mind a 15 checkpointon pontosan egyezik a forwarddal):**

| | soft kiolvasás hoponként 1→6 | kemény θ 0,1 | θ 0,3 | θ 0,5 | θ 0,8 |
|---|---|---|---|---|---|
| A0 | 6,17 · 4,68 · 4,01 · 3,06 · 2,48 · **2,18** | 2,18 @ 5,90 | 2,29 @ 5,14 | 3,34 @ 3,55 | 6,17 @ 1,00 |
| AD02a-3 | 3,84 · 3,22 · 2,66 · 2,33 · 2,28 · **2,26** | 2,26 @ 5,33 | 2,31 @ 4,21 | 2,49 @ 3,35 | 3,42 @ 1,74 |
| AD03a | 3,84 · 3,25 · 2,58 · 2,34 · 2,30 · **2,27** | 2,27 @ 5,32 | 2,32 @ 4,18 | 2,52 @ 3,24 | 3,51 @ 1,57 |
| AD03b | 3,85 · 3,25 · 2,68 · 2,34 · 2,29 · **2,27** | 2,27 @ 5,37 | 2,31 @ 4,28 | 2,51 @ 3,36 | 3,49 @ 1,64 |
| AD03c | 3,87 · 3,25 · 2,64 · 2,39 · 2,32 · **2,29** | 2,30 @ 5,34 | 2,35 @ 4,20 | 2,57 @ 3,24 | 3,60 @ 1,48 |

- Párosan az A0-hoz: az 5. hop kiolvasásának távolsága a végső soft értéktől ~0,28 nat-tal, a kemény–soft rés θ 0,5-nél 0,88–0,93 nat-tal csökken mind a négy változatnál (**SUPPORTED**); θ 0,3-nál az AD02a-3, AD03a és AD03b esetén SUPPORTED, az AD03c-nél INCONCLUSIVE.
- Tanult kilépés a **valódi véletlen megállítással** szemben (kemény futás, a tokenek a kemény út kilépési hopjainak összekevert értékénél állnak meg): a tanult kilépés θ 0,3-nál és 0,5-nél mind az öt konfigurációnál jobb, θ 0,1-nél az AD03c kivételével mindnél (SUPPORTED); θ 0,8-nál mindenhol INCONCLUSIVE.

**Értelmezés:** a 0,5-ös anytime-súly rendbe hozza a kemény utat (−0,86 nat az A0-hoz a W2-n belül; leíróan a kemény és a soft értéke a W1 0,3-as és 1,0-s súlya közé esik, de hashek között nincs összevetés). Az adaptív K számítást takarít meg a teljes K-s modellhez képest (routolt −10% / −7%, végrehajtott −18% / −10% p 0,5 / 0,7 mellett); minőségi eltérése a teljes K-s modelltől INCONCLUSIVE, széles CI-vel (+0,15 nat-ig), ezért költségmentes megtakarítást nem állítunk. p = 0,7 mellett egyértelműen jobb a fix K = 1-nél, 3%-kal több routolt FLOP-ért. Mindez nem viszi a MoE-minőség közelébe az AD-ágat: a legjobb kemény pont θ 0,5-nél még mindig 1,19× MoE.

## 7. Amit a W2 nem állapít meg
- Nincs A-bizonyíték: az A-releváns illesztett párok nem teljesülnek; az FDC1 csak több paraméterrel és több végrehajtott FLOP-pal jobb a MoE-nál; padding nélküli végrehajtási mérés még nincs (W3).
- Egy skála (17M), egy korpusz, 9k lépés, n = 3: az INCONCLUSIVE sorok (FD04b, FD05a, FD05c, FD05e − B_fixed, az AD03 − AD02a-3 párok) nem „nincs hatás”.
- Az FD05d a top-K-t kevesebb paraméterrel keveri; az FD05b moduljaiban nincs expertválasztás.

## 8. Következmények (javaslat; a következő előregisztráció dönt)
- **FixedDepth:** a hoponként külön attention marad (FD04). Az egyetlen MoE-nál jobb konfiguráció az FDC1; a döntő következő próba az **FDC1 egy paraméterben és routolt FLOP-ban illesztett MoE-val szemben** (`moe_f5 n_experts=12`, 31,37M), valamint a végrehajtott költsége padding nélküli úton. Egytényezős kontrollok (csak 3 bank, csak lapos routing) azonos hash alatt megmutatnák, összeadódnak-e a tényezők.
- **A kisebb számításon:** ezzel a kialakítással az AM-LLM 0,85×-ön kikap az illesztett MoE-tól (FD05e/f). Újabb csökkentett számítású pontot csak az FDC1-szerkezetre építve érdemes mérni.
- **AdaptiveDepth:** az anytime-veszteség (0,5) és az adaptív K (p 0,7) a jelenlegi legjobb AD-alap; a MoE-tól való távolság (1,19×) a nyitott probléma. A következő AD-lépés a végső minőséget célozza, ne a további számítás-megtakarítást.

## Fájlok
- `*/<preset>_s4?/`: futásonként `config.json`, `env.json`, `report.json`, `train.log` (checkpoint csak a mentőszerveren); `*/summary.csv`.
- `analysis/`: `ad01_trace__*.json` (AD01 a valódi véletlen-megállásos kontrollal) az A0-ra, AD02a-3-ra, AD03a/b/c-re; podonkénti `STATUS_*`.
- `queues/`, `_queue_w2*/`: a lefutott sorok és worker-logok.
