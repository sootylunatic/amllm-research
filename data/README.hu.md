[English](README.md) | **Magyar**

# Adatok

A repó **nem tartalmaz tanítóadatot** — a korpuszok helyben, determinisztikusan újraépíthetők.

| Mappa | Forrás | Előállítás |
|---|---|---|
| `toy_typescript.txt` | kézzel írt, 986 bájtos szintetikus minta (csak smoke-teszt) | a repó része |
| `ts_corpus/` | harmadik féltől származó nyílt forrású npm-csomagok `.ts/.tsx` fájljai egy helyi `node_modules` mappából | `python -m amllm.data build --src <node_modules ...> --out data/ts_corpus --max-mb 32 --val-frac 0.1 --test-frac 0.1` |
| `synthetic/` | generált ismétlés/másolás/összeadás feladatok | `python -m amllm.synth --out data/synthetic` |

A kódkorpusz felosztása **csomagonként** történik (a csomagnév SHA-256 hash-e alapján), így egy könyvtár fájljai
nem kerülhetnek egyszerre a train és a val/test részbe; a fájlok tartalom szerint deduplikáltak. A
`manifest.json` helyi elérési utakat tartalmaz, ezért nincs publikálva. Az egyes csomagok licencei eltérhetnek:
a korpusz kizárólag helyi kutatási célra készül, ne terjeszd.

Saját (nem nyílt forrású) projektek kódját **ne** add a korpuszhoz: titkokat vagy személyes adatot tartalmazhatnak.
