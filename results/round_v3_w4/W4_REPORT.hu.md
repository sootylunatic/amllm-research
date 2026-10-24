---
title: "W4 eredmények — leíró általánosítási ellenőrzés a független holdout H1-en"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "RESULT"
source_of_truth: true
supersedes: []
superseded_by: []
---

[English](W4_REPORT.md) | **Magyar**

# W4 eredmények — leíró általánosítási ellenőrzés a független holdout H1-en

**Állapot:** végleges. A [lezárt előregisztráció](PREREGISTRATION.hu.md) szerint ez **nem megerősítés** (egyik AM-jelölt sem ment át a kapuján, így a megerősítő N = 0), hanem annak ellenőrzése, hogy a V3 validációs megállapításai átvihetők-e friss csomagokra. A H1-et **egyszer** nyitottuk meg (egy kiértékelő-hívás, hash-láncolt napló: `holdout_eval/calls.log`), a lezárt holdouttal (`SHA256SUMS` `6bc776c0…`) és a 36 checkpoint befagyasztott listájával (`690fd214…`); mindkét hash-t a kiértékelő a futás előtt ellenőrizte. A holdout-fájlokat utána töröltük a GPU-gépről; a H2 lezárva marad. Egy RTX 3090, bf16; az eredmény mentve és SHA-256-tal ellenőrizve.

## 1. Szintek és rangsor

| Konfig | val NLL | holdout H1 NLL | holdout − val |
|---|---|---|---|
| MoE12 | 2,0680 | **2,5084** | +0,440 |
| B_dense | 2,1254 | 2,5173 | +0,392 |
| B_moe | 2,1180 | 2,5244 | +0,406 |
| FDC1 | 2,0870 | 2,5254 | +0,438 |
| B1F | 2,1263 | 2,5270 | +0,401 |
| B3H | 2,1332 | 2,5492 | +0,416 |
| B_fixed | 2,1599 | 2,5567 | +0,397 |
| FDC1_18k | 2,0510 | 2,5934 | +0,542 |
| MoE12_18k | 2,0499 | 2,6128 | +0,563 |
| AD02a-3 | 2,5237 | 2,9994 | +0,476 |
| AD03b | 2,5409 | 3,0142 | +0,473 |
| A0 | 3,3828 | 3,9367 | +0,554 |

A holdout minden modellnek ~0,4 nat-tal nehezebb a validációnál (más csomageloszlás, ahogy várható). A 12 konfiguráció validációs és holdout-átlagának Spearman-korrelációja: **0,52**. A rangsor fő elmozdulása: **a validáción legjobb 18k lépéses modellek friss kódon a leggyengébbek közé kerülnek** — *rosszabbak*, mint a saját 9k-s változatuk (MoE12 2,508 → 2,613, FDC1 2,525 → 2,593). 53M tanító tokennel a 18k lépés (5,6 epoch) a tanító eloszlásra illeszkedik, és rosszabbul általánosít; a W3 18k-s összevetése főleg ezt a tartományt mérte.

## 2. Az előregisztrált párok

Holdout: tokennel súlyozott páros különbség 154 csomagon (seedátlagok), 95% CI csomagszintű bootstrappel (10 000). Validáció: a W2/W3 páros seedes eredménye. Címkék az előregisztráció szerint: *átvihető* = azonos előjel és a holdout-CI nem tartalmazza a 0-t; *nem reprodukálódik* = ellentétes előjel és a CI nem tartalmazza a 0-t; egyébként *nem eldönthető*.

| Pár | val Δ (ítélet) | holdout Δ [95% CI] | csomagok, ahol a jobb | Címke |
|---|---|---|---|---|
| FDC1 − MoE12 | +0,019 (rosszabb) | **+0,017** [+0,014; +0,020] | 20% | átvihető |
| FDC1_18k − MoE12_18k | +0,001 (INCONCLUSIVE) | −0,019 [−0,025; −0,015] | 75% | nem reprodukálódik |
| B1F − B_fixed (lapos routing, 1 bank) | −0,034 (jobb) | **−0,030** [−0,033; −0,026] | 93% | átvihető |
| FDC1 − B3H (lapos routing, 3 bank) | −0,046 (jobb) | **−0,024** [−0,031; −0,016] | 90% | átvihető |
| B3H − B_fixed (3 bank, hierarchikus) | −0,027 (jobb) | −0,008 [−0,013; −0,002] | 73% | átvihető |
| FDC1 − B1F (3 bank, lapos) | −0,039 (jobb) | −0,002 [−0,011; +0,010] | 64% | nem eldönthető |
| MoE12 − B_moe | −0,050 (jobb) | −0,016 [−0,024; −0,007] | 81% | átvihető |
| B_dense − B_moe | +0,007 (INCONCLUSIVE) | −0,007 [−0,016; −0,001] | 51% | nem reprodukálódik |
| FDC1 − B_moe | −0,031 (jobb) | +0,001 [−0,007; +0,010] | 63% | nem eldönthető |
| AD02a-3 − A0 (anytime-veszteség) | −0,859 (jobb) | **−0,937** [−0,975; −0,903] | 100% | átvihető |
| AD03b − AD02a-3 (adaptív K) | +0,017 (INCONCLUSIVE) | +0,015 [+0,012; +0,018] | 14% | átvihető |

## 3. Értelmezés
- **Ami friss kódon is megáll:** a lapos routing jobb a hierarchikusnál (1 és 3 bankkal is, ~0,024–0,030 nat, a csomagok 90%+-ában); az anytime-veszteség rendbe hozza a kemény halting-utat (−0,94 nat, minden csomagon); az FDC1 rosszabb marad az azonos méretű MoE12-nél; a nagyobb MoE jobb marad a kicsinél; az adaptív K kevés minőséget (+0,015) ad a számítás-megtakarításért.
- **Ami zsugorodik vagy eltűnik:** a külön bankok nyeresége friss kódon sokkal kisebb (−0,008 hierarchikusan, lapos routinggal nem eldönthető); az FDC1 validációs előnye a kis MoE-val szemben (−0,031) eltűnik (+0,001, nem eldönthető). A sok paramétert hozó nyereségek a validáción részben a tanító eloszlást tükrözik.
- **A 18k-s eredmény friss kódon megfordul:** a holdouton az FDC1_18k jobb a MoE12_18k-nál (−0,019), de mindkettő rosszabb, mint 9k-nál; ez azt tükrözi, hogyan illeszkednek túl az ismétlődő adatra, nem szerkezeti előnyt, és azt jelzi, hogy ezekhez az összevetésekhez a 9k lépés (vagy nagyobb korpusz) a megfelelő tartomány.
- **A két „nem reprodukálódik” címke** olyan pároknál, ahol a validáció INCONCLUSIVE volt, az előregisztrált szabály mechanikus alkalmazása: a holdouton egyértelmű különbség, ahol a validáció nem látott ilyet.
- Mindebből nem következik A-bizonyíték (nincs végrehajtott-számítási állítás; leíró ellenőrzés).

## Fájlok
`holdout_eval/H1_results.json` (futásonként és csomagonként NLL-összegek és tokenszámok — csak aggregátumok), `holdout_eval/calls.log` (hash-láncolt), `holdout_eval/W4_comparison.json` / `.txt`, `candidates_frozen.json`, `pairs.json`, `val_per_seed.json`. A holdout tartalma nincs a repóban.
