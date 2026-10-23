[English](OPEN_QUESTIONS.md) | **Magyar**

# Nyitott kutatási kérdések — közreműködőket várunk

**Állapot: 2026-10-23.** Külső közreműködési ötletek, **nem** a jelenleg futó privát W1 kísérletek eredményei. Előbb ellenőrizd az [Issue-kat](https://github.com/sootylunatic/amllm-research/issues).

| ID | Kérdés | Minimálisan értelmes mérés |
|---|---|---|
| **R-01** | Hasznos a 24 expert hopok közötti súlymegosztása? | Közös vs külön bank, azonos routing/attention, FLOP és paraméter külön |
| **R-02** | Jobb a hierarchikus routing a laposnál? | Lapos 24-expertes kontroll, illesztett Top-K/FLOP és routergradiens |
| **R-03** | Ront-e az egyetlen közös bank-LayerNorm? | Közös vs hoponként külön norm, egyetlen változó |
| **R-04** | Javítható-e a hard halting? | Hard korrekt végrehajtás, köztes NLL, tényleges FLOP, fix kontroll |
| **R-05** | Olcsóbb lehet a csoportos attention vagy adapter? | Hoponként külön attention pozitív kontroll, azonos FLOP/paraméter |
| **R-06** | Miért lassabb a dekódolás a MoE-nál? | Azonos GPU/backend, decode, prefill, TTFT, VRAM, profiler |
| **R-07** | Érdekes Pareto-pont-e a H04 kisebb expertje? | 0,85×-FLOP-os MoE/Dense kontroll, CI |
| **R-08** | Generalizál-e a módszer? | Licencellenőrzött adat, csomagcsalád-deduplikáció, több seed |

Jó PR egy correctness-teszt, CPU-s ellenpélda, előregisztrált pilot vagy többseedes kontrollált kísérlet is. **Negatív eredmény is értékes.** Kérünk hipotézist, kontrollt, rögzített tényezőket, kapukat és költségkeretet. Kereséshez csak validációs adatot használjunk, **nem** a korábbi szennyezett tesztet vagy a lezárt holdoutot. Privát adatot, modellsúlyt, kulcsot és infrastruktúrarészletet tilos közzétenni. [Közreműködés](../CONTRIBUTING.hu.md) · [Architektúra](ARCHITECTURE_GUIDE.hu.md) · [Eredmények](../results/RESEARCH_REPORT.hu.md).
