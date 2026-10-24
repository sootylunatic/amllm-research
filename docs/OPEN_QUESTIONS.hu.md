[English](OPEN_QUESTIONS.md) | **Magyar**

# Nyitott kutatási kérdések — közreműködőket várunk

**Állapot: 2026-10-24 (a V3 W4 után).** Közreműködési irányok a jelenlegi bizonyítékkal. Előbb ellenőrizd az [Issue-kat](https://github.com/sootylunatic/amllm-research/issues).

| ID | Kérdés | Minimálisan értelmes mérés | Állapot (V3) |
|---|---|---|---|
| **R-01** | Hasznos a 24 expert hopok közötti súlymegosztása? | Közös vs külön bank, azonos routing/attention, FLOP és paraméter külön | **W1: a kötés ront** (külön bankok −0,020…−0,036 nat; az azonos paraméterű eredmény csomaghatás). Nyitott: nagyobb skálán, illesztett végrehajtott számítással |
| **R-02** | Jobb a hierarchikus routing a laposnál? | Lapos 24-expertes kontroll, illesztett Top-K/FLOP és routergradiens | **W1: a lapos jobb** (−0,033 nat, azonos paraméter és routolt FLOP; a végrehajtott FLOP nagyobb). Nyitott: miért, és skálán is igaz-e |
| **R-03** | Ront-e az egyetlen közös bank-LayerNorm? | Közös vs hoponként külön norm, egyetlen változó | W1: nem mutatható ki hatás (INCONCLUSIVE; CI −0,008…+0,013 és −0,005…+0,013) — több seed kell |
| **R-04** | Javítható-e a hard halting? | Hard korrekt végrehajtás, köztes NLL, tényleges FLOP, fix kontroll | **W1: az anytime-veszteség erősen szűkíti a rést** (−0,70…−0,88 nat), de ≥ 1,18× MoE. W2: anytime 0,5 és adaptív K (jobb a fix K = 1-nél) mellett is ≥ 1,19× MoE. Nyitott |
| **R-05** | Olcsóbb lehet a csoportos attention vagy adapter? | Hoponként külön attention pozitív kontroll, azonos FLOP/paraméter | **W2: nem** ~17M-en — közös attention +0,17, hoponkénti adapterrel +0,13, minden második hop csak expert +0,11 nat; 3 attention-készlet INCONCLUSIVE (+0,02, CI +0,05-ig) |
| **R-06** | Miért lassabb a dekódolás a MoE-nál? | Azonos GPU/backend, decode, prefill, TTFT, VRAM, profiler | **W3 (feltáró): főleg végrehajtási** — az expert-kernel hívásonkénti súlykonverziója és a rendező/szóró dispatch; a javított utak batch 1-en a MoE ~95%-át érik el. Nyitott: nagy batch, előregisztrált mérés |
| **R-07** | Érdekes Pareto-pont-e a H04 kisebb expertje? | 0,85×-FLOP-os MoE/Dense kontroll, CI | **W2: nem** — 0,85× routolt FLOP-on az AM-LLM 0,044–0,054 nat-tal rosszabb az azonos paraméterű MoE-nál, és az illesztett dense kontrollnál is |
| **R-08** | Generalizál-e a módszer? | Licencellenőrzött adat, csomagcsalád-deduplikáció, több seed | **W4 (részben):** független, friss csomagokból álló holdouton a lapos routing, az anytime-veszteség és az FDC1 < azonos méretű MoE átvihető; a bankok nyeresége zsugorodik; a 18k lépéses modellek rosszabbul általánosítanak. Nyitott: más adat, más méret |
| **R-09** | Megmagyarázza-e a router tanítási szabálya az AM–MoE különbség egy részét? (MoE `topk_softmax`, AM-LLM `straight_through`) | Mindkét szabály mindkét architektúrán, minden más azonos, páros seedekkel | W1: nem mutatható ki hatás (INCONCLUSIVE; MoE CI −0,036…+0,066) — több seed kell |
| **R-10** | Szerkezeti-e az FDC1 előnye (3 bank + lapos routing, 2,087 vs MoE 2,118), vagy csak a több paraméteré? | Paraméterben és routolt FLOP-ban illesztett MoE (12 expert/réteg, 31,4M), egytényezős kontrollok, hosszabb tanítás, végrehajtott költség padding nélkül | **W3: főleg a paraméterek** — az FDC1 9k lépésnél kikap az azonos méretű MoE12-től (+0,019); 18k-nál nem különböztethetők meg (+0,001, INCONCLUSIVE, ismétlődő adat). Nyitott: nagyobb korpusz |

Jó PR egy correctness-teszt, CPU-s ellenpélda, előregisztrált pilot vagy többseedes kontrollált kísérlet is. **Negatív eredmény is értékes.** Kérünk hipotézist, kontrollt, rögzített tényezőket, kapukat és költségkeretet. Kereséshez csak validációs adatot használjunk, **nem** a korábbi szennyezett tesztet vagy a lezárt holdoutot. Privát adatot, modellsúlyt, kulcsot és infrastruktúrarészletet tilos közzétenni. [Közreműködés](../CONTRIBUTING.hu.md) · [Architektúra](ARCHITECTURE_GUIDE.hu.md) · [Eredmények](../results/RESEARCH_REPORT.hu.md).
