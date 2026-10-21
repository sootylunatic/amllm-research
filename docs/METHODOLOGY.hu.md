[English](METHODOLOGY.md) | **Magyar**

# AM-LLM — kísérleti módszertan és reprodukálás

Kapcsolódó döntések: [ADR-0007](adr/0007-corpus-and-split.hu.md) (adat), [ADR-0008](adr/0008-control-models.hu.md)
(kontrollok), [ADR-0009](adr/0009-measurement-protocol.hu.md) (mérés), [ADR-0014](adr/0014-deterministic-combine-and-diagnostics.hu.md)
(determinisztikusság, diagnosztika nélküli késleltetés). Modellrészletek: [ARCHITECTURE.hu.md](ARCHITECTURE.hu.md).

> **Állapot: előzetes — audit alatt.** Az első kísérletsorozatot egy külső audit átnézte; mérési és tulajdonítási
> hibákat talált, ezek újramérése folyamatban van (E09 szakasz). A [`results/summary.csv`](../results/README.hu.md)-ben
> közölt eredmények előzetesek.

## 1. Szakaszok

| Szakasz | Kérdés | Eszköz |
|---|---|---|
| E00 | Helyes-e a környezet és a kód? | `pytest` (CPU; GPU esetén CUDA/bf16 tesztek is), `amllm.env` |
| E01 | Fut-e a tanítás, menthető és visszatölthető-e a checkpoint, működik-e az inferencia? | `amllm.train` (40 lépés) |
| E02 | Kauzalitás, determinisztikusság, számlálók, a számítás valódi kihagyása | `amllm.invariants` (CPU fp32 és CUDA fp64) |
| E03 | Hogyan cserél a halting minőséget végrehajtott hopokra? | `amllm.experiment` (ponder-sweep) + előre rögzített választási szabály |
| E04 / E05 | Dense / fix mélységű MoE kontrollok | `amllm.experiment` |
| E06 | Egytényezős abláció | `amllm.experiment` |
| E07 | Fő összehasonlítás, 3 seed | `amllm.experiment` |
| E08 | Szintetikus feladatok külön értékelése | `amllm.experiment --data data/synthetic` |
| E09 | Audit-kontrollok: kapacitásillesztett hoponkénti attention, tanult vs. véletlen routing, csak-mélység, gradienst kapó Top-1, 4 vs. 8 expert | `amllm.experiment` (a referenciák ugyanazon a kódon újratanítva) |

## 2. Tanítás (minden modellnél azonos)

| Beállítás | Érték |
|---|---|
| Reprezentáció | bájt (szókincs 256), kontextus 256 |
| Batch × lépés | 32 × 6000 (E08: 3000) → 49,2M bájt-predikció futásonként |
| Optimalizáló | AdamW, β = (0,9; 0,95), weight decay 0,05 (csak ≥ 2 dimenziós tenzorokra) |
| Tanulási ráta | 1e-3, 100 lépés lineáris warmup, koszinusz csökkenés 10%-ig |
| Pontosság | bf16 autocast (CUDA), gradiensvágás 1,0 |
| Seedek | 42, 43, 44 (fő összehasonlítás és E09-kontrollok); 42 (ablációk) |
| Dispatch | tanításhoz `bmm` (numerikusan azonos a `loop`-pal) |
| Hardver | egyetlen GPU (`CUDA_VISIBLE_DEVICES=0`), mérés közben más GPU-terhelés nélkül |

## 3. Kiértékelés

- **Validáció:** 512 rögzített, egyenletesen elosztott 256 bájtos ablak a val splitből (131 072 predikció), minden
  modellnél ugyanaz. **Test:** egyszer, az alapértelmezett inferencia-módban.
- **Alapértelmezett mód:** teljes mélység; adaptív AM-LLM: hard survival halting, τ = 0,5.
- **További AM-LLM módok:** soft; hard survival τ ∈ {0,3; 0,5; 0,65; 0,8}; v0.1 `step` τ ∈ {0,5; 0,65; 0,8};
  véletlen routing azonos súlyokkal.
- **Kategóriánként:** a megjósolandó bájt osztálya (szóköz, újsor, betű, számjegy, írásjel, egyéb) vagy a szintetikus
  feladattípus szerint: veszteség és átlagos hop.

## 4. Mutatók és definíciók

| Mutató | Definíció |
|---|---|
| val/test loss | átlagos keresztentrópia nat/bájtban; `bpb = loss / ln 2`; perplexity = `exp(loss)` (bájtonként) |
| átlagos hop | végrehajtott modulbank-hívások tokenenként (Dense/MoE: rétegszám) |
| becsült FLOP/token | analitikus forward-becslés, „implementált” és „ideális” ([ARCHITECTURE.hu.md 4. pont](ARCHITECTURE.hu.md#4-flop-becslés-forward-tokenenként-szorzás-összeadás--2-flop)) |
| profilozott FLOP/token | `torch.profiler` matmul-FLOP egy forwardon (batch 32), dispatch-utanként |
| késleltetés | teljes szekvenciás forward, CUDA-szinkronnal, 10 warmup + 50 ismétlés, medián és p95, batch 1 és 32, **diagnosztikával és anélkül**, mindhárom seeden; **nem generált token/s** |
| host-szinkronok | szinkronizációk forwardonként a CUDA sync-debug módból (alsó becslés) |
| VRAM | csúcs-allokáció a forward alatt + fp32 súlyok, **dispatch-utanként**; a tanítási csúcs külön |
| aktív paraméter/token | route-trace alapján: egyedi érintett súlyok, valamint paraméter-alkalmazások a **ténylegesen végrehajtott** attention-hívásokkal (az ideális érték külön) |
| router-specializáció | `I(választott modul ; bemeneti bájtosztály) / H(bájtosztály)` tanítatlan kontrollal összevetve; használati entrópia, max. részesedés, halott egységek |
| véletlen-routing Δ | a validációs veszteség növekedése, ha a tanult útvonalakat azonos súlyok mellett véletlenre cseréljük |
| tanítási költség | falióra-idő, token/s, tanítási csúcs-VRAM |

Minden közölt sor **egyetlen** végrehajtási út (dispatch, halting-mód, batchméret, diagnosztika-beállítás) minőségét,
számítását, késleltetését és memóriáját tartalmazza; különböző utak értékei nem keverednek egy sorban.

## 5. Döntési szabályok

- A hiperparamétereket és az értelmezési szabályokat a futtatás előtt írásban rögzítjük, és utólag nem módosítjuk.
- Az első kör célértékei: a validációs veszteség legfeljebb ~5%-kal rosszabb a megfelelő kontrollnál; ≥ ~15%
  számításcsökkenés, vagy azonos számítással jobb minőség; ≤ ~10% latency-regresszió; tanuló router; mérhető
  dinamikus mélység; 3 seeden hasonló következtetés.
- Egy komponensnek tulajdonított hatáshoz olyan kontroll kell, amely csak azt a komponenst változtatja (azonos
  paraméter és számítás); különben az eredményt zavart (confounded) eredményként közöljük.
- Azokat a futásokat, amelyekben a konfiguráció miatt egy router nem tanítható (Top-1 `topk_softmax` mellett,
  ADR-0013), az adott kérdésre érvénytelennek jelöljük.

## 6. Reprodukálás

```bash
# környezet (referencia: Python 3.12, PyTorch 2.11 cu128)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e '.[dev,analysis]'
.venv/bin/pytest -q && .venv/bin/python -m amllm.env

# adatok (lásd data/README.hu.md)
.venv/bin/python -m amllm.data build --src <node_modules mappák> --out data/ts_corpus --max-mb 32 --val-frac 0.1 --test-frac 0.1
.venv/bin/python -m amllm.synth --out data/synthetic

# kísérletek: a presetek a src/amllm/experiment.py-ban
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python -m amllm.experiment --name e07_seeds \
    --presets dense,dense_wide,moe,ut,amllm_fixed,amllm,amllm_m16 --seeds 42,43,44 --set ponder_coefficient=0.01

# elemzés
.venv/bin/python -m amllm.invariants --checkpoint <checkpoint.pt>
.venv/bin/python -m amllm.profile --runs <futáskönyvtárak> --out results/profile.json
.venv/bin/python -m amllm.router_analysis --runs <futáskönyvtárak> --out results/router_analysis.json
.venv/bin/python -m amllm.report
```

Megjegyzések:
- A determinisztikus expert-összegzéssel (v0.2.1) a kiértékelő forwardok bitre reprodukálhatók; a GPU-s tanítás
  futásonként nem bitpontos (atomikus gradiensösszegzés), így ugyanaz a seed kb. ±0,002 validációs veszteség
  eltérést adhat.
- A kódkorpusz a helyi `node_modules` tartalmától függ, ezért bájtra pontosan csak ugyanabból a forrásból
  reprodukálható; a split-azonosító minden futás `config.json`-jában szerepel.
