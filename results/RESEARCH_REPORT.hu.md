[English](RESEARCH_REPORT.md) | **Magyar**

# Publikus kutatási jelentés — 2026-10-23-i állapot

A lezárt belső kísérletek **másodlagos, auditált összesítője**, nem független újrafuttatás vagy checkpointkiadás. A most futó privát W1 eredményei **nem szerepelnek** benne.

| Fázis | Modell/adat | Értelmezés |
|---|---|---|
| E00–E09 | Bájtszintű, ~0,3–2,5M | Tanuló router; hoponként külön attention javít; eredeti kapu **bukott** |
| F1–F4 | Byte és kernel | Vegyes részeredmények, általános decode-előny nincs |
| F5 | ~17M, 8 192 BPE | MoE 2,1166, AM soft ~2,189, hard ~3,345; eredeti Q kapu **bukott** |
| Éjszaka 2026-10-22/23 | 12 konfiguráció × 3 seed | Legjobb FixedDepth MoE-közeli NLL több FLOP mellett |

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

## Halting, decode és memória

A survival soft tanítás minden hopot végrehajt, a hard inferencia levágja a token frissítéseit. F5 seed 42: soft val ~2,183, hard θ=0,5 ~3,426 és ~3,47 hop. Ez a mechanizmus nem működött a vizsgált skálán, de nem cáfol minden lehetséges adaptív architektúrát.

A korábbi F5 batch-1 KV decode **AM 632 vs MoE 884 tok/s** értéket mutatott; nem az éjszakai legjobb modellek végső L4 mérése. Training VRAM/tok/s nem inference VRAM/decode; a 17M pass@1 nem bizonyít nagy modellképességet.

## Reprodukció és tisztaság

A publikus repóban forráskód, tesztek, toy adat, [E00–E09 summary.csv](summary.csv) és ez az összesítő szerepel. A teljes korpusz, checkpointok, egyedi 36 JSON és privát orchestration **nincs nyilvánosan**, így a tábla **nem számolható újra teljes egészében** a publikus fájlokból. A reprodukcióhoz commit/kódhash, adat/tokenizer-split, optimizer, scheduler, seed és azonos FLOP-keret szükséges. Belső azonosítók: kódhash `406c4eef88f4ba69`, split ID `355afd73091925ef` — ezek nem letöltési linkek.

A régi éjszakai tervező látta a tesztveszteségeket, ezért az a teszt **nem tiszta holdout**. A kísérletválasztás validáción történjen.

[Architektúra](../docs/ARCHITECTURE_GUIDE.hu.md) · [Nyitott kérdések](../docs/OPEN_QUESTIONS.hu.md) · [Közreműködés](../CONTRIBUTING.hu.md).
