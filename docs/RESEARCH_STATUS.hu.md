[English](RESEARCH_STATUS.md) | **Magyar**

# Kutatási állapot — 2026-10-23

| Cél | Jelenlegi bizonyíték | Hiányzó igazolás |
|---|---|---|
| **A: minőség / tényleges FLOP** | **17M-en nem támogatott:** a legjobb AM csak több FLOP-pal vagy paraméterrel éri el a MoE-t; 0,85× routolt FLOP-on 0,044–0,054 nat-tal kikap az azonos paraméterű MoE-tól (W2); E07 kapu bukott | FLOP-illesztett Pareto-előny, független adat |
| **B: inference GPU-memória** | Nem igazolt; training VRAM nem memóriaelőny | Valós paging/offload, VRAM és PCIe adatok |
| **C: dekódolás** | Nem igazolt: korábbi F5 b=1 AM 632 vs MoE 884 tok/s; éjszakai nyertesek nem teljesen mértek | Azonos backendű decode/prefill/TTFT/VRAM |
| **D: nagyobb modellképesség** | Távoli cél, nem bizonyított | Skálázási és kódolási tesztek |

E09: tanult routing relatív 19,9%-kal jobb volt a **véletlen routingnál**, nem a MoE-nál. Hoponként külön attention javította a kis modell minőségét, több kapacitással. A 36 éjszakai futásban (12 × 3 seed) H05_h8 loss 2,1212 vs MoE 2,1166, 1,24× FLOP és +6,9% paraméter mellett. A nullát tartalmazó CI **nem bizonyít ekvivalenciát**. A soft-survival/hard-truncation eltérés súlyos minőségvesztést okoz rövidebb mélységnél. H04 0,85× FLOP mellett **INCONCLUSIVE**.

**V3 kör, W1 (előregisztrált, 48 futás, lezárva 2026-10-23):** a FixedDepth-et a MoE-tól elválasztó négy tényezőből a hopok közötti expert-kötés (külön bankok −0,020…−0,036 nat) és a hierarchikus routing (lapos 24-utas router −0,033 nat azonos paraméter és azonos *routolt* FLOP mellett) egyaránt ront; a közös bank-normalizálásnál és a router tanítási szabályánál nem mutatható ki hatás (INCONCLUSIVE, nem egyenértékűség). A tanult routing 0,129 nat-tal jobb a véletlennél. A legjobb FixedDepth-változatok a dense kontrollt érik el, a MoE-t nem. Az anytime (több kilépéses) veszteség 0,70–0,88 nat-tal javítja a kemény halting-utat, de 1,18–1,27× marad a MoE-hoz képest 0,87× routolt és 1,10–1,12× végrehajtott FLOP mellett — az előregisztrált A-szűrőkapu nem teljesül. [W1 riport](../results/round_v3_w1/W1_REPORT.hu.md).

**V3 kör, W2 (előregisztrált, 66 futás, lezárva 2026-10-24):** a hoponként külön attention számít (közös +0,174, adapterrel +0,132, minden második hop csak expert +0,114 nat). Azonos routolt FLOP és paraméter mellett a 8 hop és a top-2/top-1 nem különböztethető meg az alaptól (INCONCLUSIVE); a 4 hop rosszabb. **Az A-releváns illesztett párok nem teljesülnek:** 0,85× routolt FLOP-on az AM-LLM 0,044–0,054 nat-tal rosszabb az azonos paraméterű MoE-nál. A W1-kombináció, az *FDC1* (3 bank + lapos routing) 2,087-et ér el a MoE 2,118-ával szemben azonos routolt FLOP-on, de 1,83× paraméterrel és 1,38× végrehajtott FLOP-pal; előnye a tanítás során csökken (2k lépésnél −0,127, 9k-nál −0,031), és a paraméterben illesztett MoE-kontroll még nem futott. A halting-ágon a 0,5-ös anytime-súly és az adaptív K (0,068 nat-tal jobb a fix K = 1-nél) mellett is ≥ 1,19× MoE a kemény út. Megjegyzés: a tanító split 53M token, így minden futás ~2,8-szor látja. [W2 riport](../results/round_v3_w2/W2_REPORT.hu.md).

Korlátok: csak 3 seed; bájt- és BPE-loss nem vethető össze; FLOP-becslés ≠ tényleges kernelmunka vagy decode tok/s; az éjszakai tervező látta a tesztértékeket, így a teszt nem tiszta holdout. A teljes korpusz, súly és egyedi JSON nem publikus, tehát ez **auditált másodlagos összefoglaló**, nem független reprodukció. [Jelentés](../results/RESEARCH_REPORT.hu.md) · [Architektúra](ARCHITECTURE_GUIDE.hu.md) · [Nyitott kérdések](OPEN_QUESTIONS.hu.md).
