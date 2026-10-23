[English](CONTRIBUTING.md) | **Magyar**

# Közreműködés az AM-LLM kutatásban

Értékesek a **kontrollált negatív eredmények**, reprodukciós javítások, tesztek, kódauditok és architektúra-ablációk. Kiindulópont: [nyitott kutatási kérdések](docs/OPEN_QUESTIONS.hu.md).

1. Olvasd el a [célokat](docs/00-goals/RESEARCH_GOALS.hu.md), az [eredményeket](results/RESEARCH_REPORT.hu.md) és nézd meg az [Issue-kat](https://github.com/sootylunatic/amllm-research/issues).
2. Nyiss javaslatot ellenőrizhető hipotézissel, Dense/MoE-kontrollal, rögzített változókkal, mérési metrikákkal, helyességi tesztekkel és a legkisebb informatív kísérlettel. GPU-költség előtt egyeztess.
3. A PR tartalmazzon teszteket, kódcommitot/kódhash-t, korpusz/tokenizer/split ID-ket, seedlistát, tanítási lépést/tokenszámot, modellparamétereket, validációs NLL-t, becsült vagy tényleges FLOP-ot megfelelő jelöléssel és bizonytalanságot. **Training és inference** tok/s/VRAM külön értékelendő.
4. Az egyseedes pilot **feltáró**, három seed mellett is lehet széles CI. Történeti sikerkaput utólag nem lehet átírni. A negatív ablációkat is jelentsük.

## Helyi teszt

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[dev,analysis]'
.venv/bin/python -m pytest -q
```

A toy adat nem az F5 teljes korpusza. A 36 futás összesítője nem reprodukálható teljesen a publikus állományból checkpointok és adatforrások nélkül. **Nem szabad a tervező által látott régi teszten vagy lezárt holdouton hangolni.** Kulcsot, nem engedélyezett korpuszt, védett súlyt, infrastruktúrarészletet ne publikálj. Generált kód futtatása izolált offline sandboxban történjen.

**PR-ellenőrzés:** hipotézis/kontrollok; kauzalitás és correctness; kód-/adat-azonosítók és seedek; bizonytalanság/negatív eredmények; megalapozott FLOP/VRAM/decode; holdout-tisztaság; frissített dokumentáció.
