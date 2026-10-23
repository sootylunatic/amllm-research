[English](RESEARCH_STATUS.md) | **Magyar**

# Kutatási állapot — 2026-10-23

| Cél | Jelenlegi bizonyíték | Hiányzó igazolás |
|---|---|---|
| **A: minőség / tényleges FLOP** | **17M-en nem támogatott:** legjobb AM ≈ MoE loss, +24–31% becsült FLOP; E07 kapu bukott | FLOP-illesztett Pareto-előny, független adat |
| **B: inference GPU-memória** | Nem igazolt; training VRAM nem memóriaelőny | Valós paging/offload, VRAM és PCIe adatok |
| **C: dekódolás** | Nem igazolt: korábbi F5 b=1 AM 632 vs MoE 884 tok/s; éjszakai nyertesek nem teljesen mértek | Azonos backendű decode/prefill/TTFT/VRAM |
| **D: nagyobb modellképesség** | Távoli cél, nem bizonyított | Skálázási és kódolási tesztek |

E09: tanult routing relatív 19,9%-kal jobb volt a **véletlen routingnál**, nem a MoE-nál. Hoponként külön attention javította a kis modell minőségét, több kapacitással. A 36 éjszakai futásban (12 × 3 seed) H05_h8 loss 2,1212 vs MoE 2,1166, 1,24× FLOP és +6,9% paraméter mellett. A nullát tartalmazó CI **nem bizonyít ekvivalenciát**. A soft-survival/hard-truncation eltérés súlyos minőségvesztést okoz rövidebb mélységnél. H04 0,85× FLOP mellett **INCONCLUSIVE**.

Korlátok: csak 3 seed; bájt- és BPE-loss nem vethető össze; FLOP-becslés ≠ tényleges kernelmunka vagy decode tok/s; az éjszakai tervező látta a tesztértékeket, így a teszt nem tiszta holdout. A teljes korpusz, súly és egyedi JSON nem publikus, tehát ez **auditált másodlagos összefoglaló**, nem független reprodukció. [Jelentés](../results/RESEARCH_REPORT.hu.md) · [Architektúra](ARCHITECTURE_GUIDE.hu.md) · [Nyitott kérdések](OPEN_QUESTIONS.hu.md).
