[English](README.md) | **Magyar**

# AM-LLM — Adaptive Modular Language Model (kutatási prototípus)

[![tests](https://github.com/sootylunatic/amllm-research/actions/workflows/tests.yml/badge.svg)](https://github.com/sootylunatic/amllm-research/actions/workflows/tests.yml)
![status](https://img.shields.io/badge/results-preliminary%20%E2%80%94%20under%20audit-orange)
![license](https://img.shields.io/badge/license-MIT-blue)

Kis, tanulmányi jellegű kutatási prototípus. A modellek apró, bájtszintű nyelvi modellek, egyetlen fogyasztói GPU-n
tanítva. Ez nem termék, és semmilyen állítást nem tesz nagy modellek teljesítményéről.

> **Előzetes eredmények — audit alatt.** Az első kísérletsorozatot egy külső audit átnézte, és mérési, valamint
> tulajdonítási hibákat talált. Ezek javítása és újramérése folyamatban van. Addig a
> [`results/summary.csv`](results/README.hu.md) előzetes, és nem idézhető megállapításként.

## Kutatási kérdés

Megtanulhatja-e egy nyelvi modell, hogy tokenenként **mely feldolgozó egységeket, milyen sorrendben és hányszor**
futtasson, és így **kevesebb aktív számítással** érje el egy Dense Transformer vagy egy fix mélységű
Mixture-of-Experts minőségét?

A prototípus három mechanizmust kombinál:

1. **megosztott modulbank**, amelyet a modell ismételten („hopokban”) használ;
2. **hierarchikus Top-K routing**: egy router modulokat választ, majd minden kiválasztott modul experteket;
3. **tokenenkénti adaptív mélység**: egy megállási fej dönti el, kell-e további hop.

## Tartalom

```text
src/amllm/   modell, routing, adat, tanítás, kísérletek, profilozás, elemzés
tests/       pytest-tesztek (CPU; GPU esetén további CUDA/bf16 tesztek)
docs/        architektúra, módszertan, technikai döntési rekordok (angolul és magyarul)
data/        csak egy apró mintafájl; a korpuszok helyben épülnek újra (data/README.hu.md)
results/     előzetes összesítő tábla (results/README.hu.md)
```

Dokumentáció: [docs/ARCHITECTURE.hu.md](docs/ARCHITECTURE.hu.md) · [docs/METHODOLOGY.hu.md](docs/METHODOLOGY.hu.md) ·
[docs/adr/](docs/adr/README.hu.md)

## Gyors indítás

```bash
# referencia-környezet: Python 3.12, PyTorch 2.11 (CUDA 12.8); a CPU-s tesztek GPU nélkül is futnak
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e '.[dev,analysis]'
.venv/bin/pytest -q

# 40 lépéses smoke-futás a mellékelt mintafájlon
.venv/bin/python -m amllm.train --data data/toy_typescript.txt --steps 40 --batch 4 --max-seq-len 64 --run-dir runs/smoke
```

## Korlátok

- Bájtszintű tokenek, ~0,3–2,5M paraméter, ~14 MB nyílt forrású TypeScript-kódon tanítva.
- A késleltetés teljes szekvenciás forward. Generált token/s-t nem mérünk, mert még nincs KV-cache.
- Minden súly a GPU-memóriában marad, ezért VRAM-megtakarítást nem állítunk.
- Tanítóadat és modellsúly nincs publikálva (lásd [data/README.hu.md](data/README.hu.md)).

## Megjegyzés

Az implementáció egyes részeinél és a kísérletek átnézésénél Claude és Codex modellek segítettek.

## Licenc

MIT, lásd [LICENSE](LICENSE).
