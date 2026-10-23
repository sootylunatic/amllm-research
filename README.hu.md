[English](README.md) | **Magyar**

# AM-LLM — Adaptív moduláris nyelvi modell

**Nyílt kutatási prototípus.** Közös expertbank, hierarchikus routing, ismételt hopok és opcionális adaptív számítás. [Tesztek](https://github.com/sootylunatic/amllm-research/actions) · [MIT licenc](LICENSE)

> **Bizonyítékok — 2026-10-23:** Az eredeti számítási hatékonysági hipotézist a vizsgált ~17M paraméteres skálán **nem támogatják az adatok**. A legjobb FixedDepth változatok MoE-közeli validációs loss-t értek el **24–31%-kal több becsült FLOP** mellett. A jelenlegi halting soft tanítás és hard inferencia között jelentős eltérés van. Ez nem kész nagy nyelvi modell, és nem bizonyított teljesítményelőny.

## Kutatási kérdés

Megtanulhatja-e egy modell, hogy **mely újrafelhasználható modulokat/experteket, milyen sorrendben és hányszor** használjon, Dense/MoE-közeli minőséget elérve **kevesebb tényleges számítással**? A B–D célok: kisebb valós inference GPU-munkakészlet, gyors dekódolás és hasznos nagy modellképesség fogyasztói hardveren. A 16 GB VRAM és ≥60 dekódolt tok/s **hosszú távú cél**, nem bizonyított eredmény. [A–D célok](docs/00-goals/RESEARCH_GOALS.hu.md).

## Mérési eredmények

F5/éjszaka: ~17M paraméter, 8 192 BPE token, 512 kontextus, 12 konfiguráció × 3 seed. Validációs NLL (nat/token); a FLOP **becslés**, nem dekódolási sebesség.

| Konfiguráció | Val NLL ↓ | Becsült FLOP / MoE ↓ |
|---|---:|---:|
| 6 rétegű MoE | **2,1166** | 1,00× |
| FixedDepth, 6 hop | 2,1625 | 1,00× |
| FixedDepth, 8 hop | 2,1212 | 1,24× |
| FixedDepth, Top-2/Top-2 | 2,1366 | 1,31× |
| FixedDepth, kisebb expertek | 2,1804 | 0,85× |

A két legjobb modell párosított 95%-os CI-je tartalmazza a nullát, ami **nem bizonyít ekvivalenciát**. A 0,85×-os változat megfelelő kontroll nélkül nem eldöntött. A régi tesztet a kísérlettervező látta, ezért **nem tiszta holdout**.

[Eredmények és korlátok](results/RESEARCH_REPORT.hu.md) · [Állapot](docs/RESEARCH_STATUS.hu.md) · [E00–E09 CSV](results/summary.csv)

## Két architekturális ág

**FixedDepth:** fix 4/6/8 hop, hoponként külön attention, újrahasznált expertbank. Hasznos kontroll, nem az adaptív számítás igazolása. **AdaptiveDepth:** tanult megállás, ahol a soft/hard eltérés jelenleg minőségromláshoz vezet. A hat hopos FixedDepth szerkezetileg egy **expert-súlymegosztásos MoE-hoz** hasonlít; a bankmegosztás, hierarchikus routing és közös normalizálás hatása egyelőre **nincs külön mérve**. [Architektúra](docs/ARCHITECTURE_GUIDE.hu.md).

## Közreműködés

Várjuk a kontrollált ablációkat, teszteket, profilozást, tudományos kritikát és **negatív eredményeket**. [Nyitott feladatok](docs/OPEN_QUESTIONS.hu.md) · [CONTRIBUTING](CONTRIBUTING.hu.md) · [Issues](https://github.com/sootylunatic/amllm-research/issues).

## Indítás (csak smoke teszt)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[dev,analysis]'
.venv/bin/python -m pytest -q
.venv/bin/python -m amllm.train --data data/toy_typescript.txt --steps 40 --batch 4 --max-seq-len 64 --run-dir runs/smoke
```

A teljes korpusz, checkpointok és egyedi naplók **nem nyilvánosak**, így a 36 futás a publikus fájlokból még nem reprodukálható. Valós nagy modell-VRAM vagy ≥60 tok/s eredményt nem állítunk. A fejlesztésben és auditban Claude és Codex is segített; a megállapítások függetlenül vizsgálhatók.
