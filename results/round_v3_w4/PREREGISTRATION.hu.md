---
title: "W4 előregisztráció — leíró általánosítási ellenőrzés a független holdouton"
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

# W4 előregisztráció — leíró általánosítási ellenőrzés a független holdouton

**Állapot:** JÓVÁHAGYVA ÉS LEZÁRVA — a tulajdonos 2026-10-24-én jóváhagyta („rendben”). A tartalmat `PREREGISTRATION.sha256` rögzíti; a holdout H1 része csak ezután nyílik meg, egyszer.

## 1. Miért leíró, és nem megerősítő
A holdout-terv előre rögzített szabálya szerint egy ág csak akkor kap jelölthelyet, ha a jelöltje átment a saját előregisztrált kapuján. A V3 végén **egyik AM-jelölt sem ment át** (FixedDepth: az FDC1 kikapott az azonos méretű MoE12-től, W3; AdaptiveDepth: az A-kapu 0/3, W2–W3), így a megerősítő értékelés jelöltszáma **N = 0**. A tulajdonos 2026-10-24-én úgy döntött („2. legyen, tesztek mindig fontosak”), hogy a holdout **H1** részét egy **leíró általánosítási ellenőrzésre** használjuk: átvihetők-e a V3 validációs megállapításai friss, független csomagokra. **Ez nem megerősítés:** egyik modell sem lesz „holdouton igazolt”, és az eredmény nem jut vissza kiválasztásra. A **H2** (lezárt tartalék) érintetlen marad egy jövőbeli megerősítő értékelésre.

## 2. A holdout
H-A, `tools/holdout/build_holdout.py`, épült 2026-10-24-én a privát mentőszerveren: friss npm-csomagok (legutóbbi verzió), a `code_corpus_v2` mindhárom splitjétől független név, scope, névtő és repository szerint, fájlonként pontos és közeli (MinHash, Jaccard ≥ 0,8) duplikátumszűréssel a v2 minden dokumentumával szemben; a v2 tokenizerével; csomagonként legfeljebb 0,2 MB; H1 és H2 családok szerint diszjunkt. **A lezárt holdout:** H1 3 055 222 token, 154 csomag, 1 980 fájl (a legnagyobb csomag a tokenek 3,2%-a, medián 7 121 token/csomag); H2 3 054 781 token, 149 csomag (lezárt tartalék). Kizárva: 299 csomag családi egyezés, 6 csomag szivárgás, 46 fájl pontos és 46 fájl közeli duplikátum miatt (26 350 jelöltből). `SHA256SUMS` SHA-256: `6bc776c0dae7d2fc9d03b9c34c050fadbdf39d28f9012edb022278157b01d379`; eltérő hash esetén a kiértékelő megtagadja a futást. Két korábbi építést lezárás előtt elvetettünk, mert nem feleltek meg a tervnek (egy csomag túl nagy részesedése); azokon modell nem futott.

## 3. Befagyasztott modellek (nem választás: minden V3 fő konfiguráció, 3 seed)
Kódhash `cd909ea282660f0f` (W2/W3). A checkpointok SHA-256-ját a befagyasztott lista (`candidates_frozen.json`, 36 checkpoint; a fájl SHA-256-ja `690fd214c8bd601e08cf834174a85b60b19367fc15431e69a86895f92caa910b`) rögzíti.

| Csoport | Konfigurációk |
|---|---|
| Kontrollok | B_moe, MoE12, B_dense |
| FixedDepth | B_fixed, B3H, B1F, FDC1 |
| Hosszabb tanítás | FDC1_18k, MoE12_18k |
| AdaptiveDepth | A0, AD02a-3, AD03b |

12 konfiguráció × 3 seed = 36 checkpoint.

## 4. Mérés és összevetések
- **Metrika:** holdout NLL (nat/token), az alapmódban (`fixed`; halting modellnél kemény θ 0,5), bf16, azonos, át nem fedő 512-es ablakokon; mellé bits/byte.
- **Statisztika:** páros különbség csomagonként (a seedek átlaga), tokennel súlyozva; 95% CI **csomagszintű bootstrappel** (10 000 újramintavétel) — a csomag a független egység.
- **Párok** (mindegyik a validációs megfelelőjével együtt jelentve):
  1. FDC1 − MoE12 · 2. FDC1_18k − MoE12_18k · 3. B1F − B_fixed · 4. FDC1 − B3H · 5. B3H − B_fixed · 6. FDC1 − B1F · 7. MoE12 − B_moe · 8. B_dense − B_moe · 9. FDC1 − B_moe · 10. AD02a-3 − A0 · 11. AD03b − AD02a-3
- **Átviteli címke (leíró, kapu nélkül):** *átvihető*, ha a holdout-különbség előjele egyezik a validációsé, és a CI nem tartalmazza a 0-t; *nem reprodukálódik*, ha ellentétes előjelű és a CI nem tartalmazza a 0-t; egyébként *nem eldönthető*.
- **Rangsor:** a 12 konfiguráció val- és holdout-átlagának Spearman-korrelációja.
- **Abszolút szint:** a holdout eloszlása eltérhet a validációétól; a következtetés a páros különbségen és a rangsoron alapul, nem a szinten.

## 5. Eljárás, költség
- Egyszeri kiértékelés egy GPU-n (RTX 3090 vagy hasonló) a `python -m amllm.holdout evaluate` paranccsal; minden hívás hash-láncolt naplóba kerül; az eredményt akkor is közöljük, ha kedvezőtlen.
- A holdout-fájlok a kiértékelés idejére kerülnek a GPU-gépre, utána törlődnek; a tartalom nem publikus (csomaglista és hashek igen).
- ~36 checkpoint × ~3M token: néhány perc GPU-idő, < $0,5.

## 6. Nem része a W4-nek
A H2 megnyitása; bármilyen modellválasztás vagy hangolás a holdout alapján; új tanítás.
