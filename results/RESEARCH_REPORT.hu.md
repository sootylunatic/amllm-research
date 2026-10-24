[English](RESEARCH_REPORT.md) | **Magyar**

# Publikus kutatási jelentés — 2026-10-23-i állapot

A lezárt belső kísérletek **másodlagos, auditált összesítője**, nem független újrafuttatás vagy checkpointkiadás. Az előregisztrált V3 kör lezárt W1–W3 hulláma alább szerepel.

| Fázis | Modell/adat | Értelmezés |
|---|---|---|
| E00–E09 | Bájtszintű, ~0,3–2,5M | Tanuló router; hoponként külön attention javít; eredeti kapu **bukott** |
| F1–F4 | Byte és kernel | Vegyes részeredmények, általános decode-előny nincs |
| F5 | ~17M, 8 192 BPE | MoE 2,1166, AM soft ~2,189, hard ~3,345; eredeti Q kapu **bukott** |
| Éjszaka 2026-10-22/23 | 12 konfiguráció × 3 seed | Legjobb FixedDepth MoE-közeli NLL több FLOP mellett |
| V3 W1 2026-10-23 | 16 konfiguráció × 3 seed, előregisztrálva | Az expert-kötés és a hierarchikus routing ront; az anytime-veszteség rendbe hozza a kemény utat, de nem MoE-szintre |
| V3 W2 2026-10-24 | 22 konfiguráció × 3 seed, előregisztrálva | A hoponkénti attention számít; 0,85× routolt FLOP-on az AM kikap az illesztett MoE-tól; az FDC1 csak 1,83× paraméterrel van a MoE alatt |
| V3 W3 2026-10-24 | 17–31M, 5 konfiguráció × 3 seed + mérés, előregisztrálva | Az azonos méretű MoE 9k-nál veri az FDC1-et, 18k-nál nem különböztethetők meg; az A-kapu nem teljesül; a dekódolási rés főleg végrehajtási (feltáró) |

A bájtszintű és BPE nat/token loss **nem összevethető közvetlenül**.

## A 36 futás összesített eredménye

9000 lépés, batch 32, kontextus 512, azonos adat/tokenizer, seed 42/43/44; 512 validációs ablak. A relatív FLOP **analitikus becslés**, nem minden backend tényleges FLOP-ja és nem mért decode sebesség.

| Konfiguráció | Átlag val NLL ↓ | Rel. FLOP ↓ | Paraméter | Mód |
|---|---:|---:|---:|---|
| B_moe_3090 | **2,1166** | 1,00× | 17,15M | fixed |
| H05_h8_fixed | 2,1212 | 1,24× | 18,33M | 8 hop |
| H03_m2e2_fixed | 2,1366 | 1,31× | 17,15M | Top-2/Top-2 |
| H03_m2e1_fixed | 2,1558 | 1,00× | 17,15M | Top-2/Top-1 |
| H01_fixed | 2,1625 | 1,00× | 17,15M | 6 hop |
| H04_e8h192_fixed | 2,1804 | 0,85× | 17,17M | keskeny expertek |
| H01_p005 | 2,1846 | 1,00× | 17,15M | hard survival |
| H04_m12h192_fixed | 2,1886 | 0,85× | 17,17M | több kisebb modul |
| H03_m1e1_fixed | 2,1943 | 0,85× | 17,15M | Top-1/Top-1 |
| H05_h4_fixed | 2,2542 | 0,76× | 15,97M | 4 hop |
| H01_bias_m2 | 3,2753 | 0,88× | 17,15M | hard survival |
| H01_p05 | 6,3677 | 0,49× | 17,15M | hard survival |

Párosított 95% CI (AM−MoE val NLL, n=3): **H05_h8 +0,005 [−0,035;+0,044]**, **H03_m2e2 +0,020 [−0,018;+0,058]**. A nulla jelenléte **nem bizonyít ekvivalenciát**. Egyetlen mért AM sem igazolta a MoE-közeli minőséget kevesebb tényleges számítással. A H04 **INCONCLUSIVE**, mert hiányzik a hasonló FLOP-os MoE/Dense kontroll.

## V3 kör, W1 — előregisztrált ablációk (48 futás, lezárva 2026-10-23)

Azonos korpusz, méret és protokoll (9000 lépés, seed 42/43/44, validációs NLL), új kódverzió (`ba81b4fc02cd7b63`), [lezárt előregisztráció](round_v3_w1/PREREGISTRATION.hu.md) irányított szabállyal: `SUPPORTED` = Δ ≤ −0,010, a 95%-os CI 0 alatt, 3/3 seed. Seedenkénti táblák: [W1 riport](round_v3_w1/W1_REPORT.hu.md).

| Kérdés (Δ a FixedDepth B_fixedhez, 2,1585; MoE 2,1185, Dense 2,1253) | Δ nat | 95% CI | Ítélet |
|---|---:|---|---|
| Külön bank minden hopnak (3× paraméter) | −0,036 | [−0,049, −0,023] | jobb |
| Külön bankok, azonos össz-paraméter (csomaghatás) | −0,020 | [−0,030, −0,011] | jobb |
| Lapos 24-utas router a hierarchikus helyett | −0,033 | [−0,043, −0,024] | jobb |
| Véletlen routing (a router értékének kontrollja) | +0,129 | [+0,119, +0,139] | rosszabb |
| Hoponkénti bank-LayerNorm / hoponkénti skála-eltolás | +0,003 / +0,004 | a CI tartalmazza a 0-t | INCONCLUSIVE |
| Router-szabály: MoE `straight_through`-val / AM `full_softmax`-szal | +0,015 / +0,007 | a CI tartalmazza a 0-t / küszöb alatt | INCONCLUSIVE |
| Anytime-veszteség 0,3 / 1,0 a halting A0-hoz (kemény út) | −0,699 / −0,878 | 0 alatt | jobb; még 1,27× / 1,18× MoE |

Az „azonos FLOP” routolt FLOP-ot jelent; a bmm-paddinggel végrehajtott FLOP eltér (a W1 riportban). Az INCONCLUSIVE sorok nem mutatják a hatás hiányát.

## V3 kör, W2 — attention, illesztett pontok, FDC1, adaptív K (66 futás, lezárva 2026-10-24)

Új kódverzió (`cd909ea282660f0f`); a horgonyok ±0,0015-ön belül reprodukálják a W1-et (MoE 2,1180, Dense 2,1254, FixedDepth B_fixed 2,1599). [Lezárt előregisztráció](round_v3_w2/PREREGISTRATION.hu.md) · [W2 riport](round_v3_w2/W2_REPORT.hu.md) minden seedenkénti értékkel.

| Kérdés | Δ nat | 95% CI | Ítélet |
|---|---:|---|---|
| Egy közös attention a hoponkénti helyett (B_fixed) | +0,174 | [+0,134, +0,214] | rosszabb |
| Közös attention + hoponkénti adapter | +0,132 | [+0,110, +0,153] | rosszabb |
| Minden második hop csak expert (0,79× routolt FLOP) | +0,114 | [+0,035, +0,192] | rosszabb |
| 3 attention-készlet | +0,020 | [−0,012, +0,051] | INCONCLUSIVE |
| 8 hop, azonos routolt FLOP és paraméter | +0,008 | [−0,029, +0,045] | INCONCLUSIVE |
| 4 hop, azonos routolt FLOP | +0,074 | [+0,031, +0,118] | rosszabb |
| **AM finomabb expertek − illesztett MoE, 0,85× routolt FLOP, azonos paraméter** | **+0,054** | [+0,030, +0,077] | **rosszabb** |
| **AM kisebb expertek − illesztett MoE, 0,86×, azonos paraméter** | **+0,044** | [+0,019, +0,069] | **rosszabb** |
| FDC1 (3 bank + lapos routing) − B_fixed | −0,073 | [−0,076, −0,070] | jobb |
| FDC1 − MoE (leíró: 1,83× paraméter, 1,38× végrehajtott FLOP) | −0,031 | [−0,052, −0,010] | kisebb veszteség, nem illesztett |
| Anytime 0,5 a halting A0-hoz (kemény út) | −0,859 | [−1,135, −0,583] | jobb; még 1,19× MoE |
| Adaptív K (p 0,7) − fix K = 1 | −0,068 | [−0,096, −0,039] | jobb |

Az azonos FLOP routolt FLOP-ot jelent. A tanító split 53M token, így minden futás kb. 2,8-szor látja; az FDC1 előnye a MoE-val szemben a tanítás során csökken (2k lépésnél −0,127, 9k-nál −0,031). Az INCONCLUSIVE sorok nem mutatják a hatás hiányát.

## V3 kör, W3 — FDC1 az azonos méretű MoE ellen, hosszabb tanítás, gyakorlati mérés (lezárva 2026-10-24)

A tanítási kód a W2-höz képest változatlan, így a W2-futásokkal közvetlen az összevetés. [Lezárt előregisztráció](round_v3_w3/PREREGISTRATION.hu.md) · [W3 riport](round_v3_w3/W3_REPORT.hu.md) seedenkénti értékekkel, görbékkel és mérési táblákkal.

| Kérdés | Δ nat | 95% CI | Ítélet |
|---|---:|---|---|
| **FDC1 − MoE12** (azonos paraméter 31,4M, azonos routolt FLOP), 9k lépés | **+0,019** | [+0,000, +0,038] | **rosszabb** — a W2-es előny a paraméterekből jött |
| FDC1 − MoE12 18k lépésnél (az adat ~5,6-szor ismétlődik) | +0,001 | [−0,018, +0,020] | INCONCLUSIVE |
| Lapos − hierarchikus routing, 3 bank / 1 bank (azonos paraméter) | −0,046 / −0,034 | 0 alatt | jobb |
| 3 bank − 1 bank, hierarchikus / lapos (1,83× paraméter) | −0,027 / −0,039 | 0 alatt | jobb |
| A-sikerkapu, halting modellek (padding nélküli végrehajtott FLOP ≤ 0,90× MoE **és** veszteség ≤ 1,03× MoE) | 0,80–0,86× FLOP, 1,18–1,23× veszteség | | **nem teljesül** (0/3) |

Feltáró dekódolási mérés (egy RTX 3090, batch 1, nem előregisztrált): egy hívásonkénti expertsúly-konverzió és a rendező/szóró dispatch javítása 3–3,3-szorosára gyorsítja az AM-LLM dekódolását (FDC1 2 179 vs MoE 2 293 token/s), az adaptív mélységű modellt ~23-szorosára (88 → ~2 000 token/s); a validációs veszteség legfeljebb 0,003-mal változik. Batch 32-n a MoE marad gyorsabb.

## Halting, decode és memória

A survival soft tanítás minden hopot végrehajt, a hard inferencia levágja a token frissítéseit. F5 seed 42: soft val ~2,183, hard θ=0,5 ~3,426 és ~3,47 hop. Ez a mechanizmus nem működött a vizsgált skálán, de nem cáfol minden lehetséges adaptív architektúrát.

A korábbi F5 batch-1 KV decode **AM 632 vs MoE 884 tok/s** értéket mutatott; nem az éjszakai legjobb modellek végső L4 mérése. Training VRAM/tok/s nem inference VRAM/decode; a 17M pass@1 nem bizonyít nagy modellképességet.

## Reprodukció és tisztaság

A publikus repóban forráskód, tesztek, toy adat, [E00–E09 summary.csv](summary.csv) és ez az összesítő szerepel. A teljes korpusz, checkpointok, egyedi 36 JSON és privát orchestration **nincs nyilvánosan**, így a tábla **nem számolható újra teljes egészében** a publikus fájlokból. A reprodukcióhoz commit/kódhash, adat/tokenizer-split, optimizer, scheduler, seed és azonos FLOP-keret szükséges. Belső azonosítók: kódhash `406c4eef88f4ba69`, split ID `355afd73091925ef` — ezek nem letöltési linkek.

A régi éjszakai tervező látta a tesztveszteségeket, ezért az a teszt **nem tiszta holdout**. A kísérletválasztás validáción történjen.

[Architektúra](../docs/ARCHITECTURE_GUIDE.hu.md) · [Nyitott kérdések](../docs/OPEN_QUESTIONS.hu.md) · [Közreműködés](../CONTRIBUTING.hu.md).
