[English](README.md) | **Magyar**

# Eredmények (első sorozat, auditált)

A `summary.csv` futásonként és kiértékelési módonként egy sort tartalmaz. Ez az egyetlen publikált eredmény:
a futásnaplók, a checkpointok és a részletes jelentés a privát fejlesztési archívumban maradnak.

**A számok használata előtt:**

- A tábla **csak az első sorozatot (E00–E09)** tartalmazza. Az audit mérési és tulajdonítási hibákat talált; a javítások
  és az újramérések benne vannak. Az előre rögzített go/no-go kapu nem teljesült, a későbbi feltáró eredmények
  (F1–F5, éjszakai kör) nem részei a táblának.
- Modelleket **csak azonos `experiment`-en belül** hasonlíts — így azonos `code_version` mellett — és azonos
  `eval_mode`-dal. Az E07 és az E09 eltérő kódverzióval futott; az E09 a saját referenciáit újratanítja.
- A nem üres `validity_note`-tal jelölt sorok nem azt mérik, amit a preset neve sugall. Például Top-1 routing
  `topk_softmax` súlyozással, ahol a router nem kap gradienst (lásd ADR-0013).
- Az `eval_batch_ms_with_diagnostics` a routing-statisztikát és a host-szinkronizációt is tartalmazza. Ez **nem**
  késleltetési eredmény, és nem generált token/s.

| Oszlop | Jelentés |
|---|---|
| `experiment` | E01 smoke, E03 halting-sweep, E06 ablációk, E07 fő összehasonlítás, E07b csökkentett számítás / feltáró, E08 szintetikus feladatok, E09 audit-kontrollok |
| `code_version` | a futást előállító csomagverzió |
| `preset`, `seed`, `arch` | konfiguráció neve (lásd `src/amllm/experiment.py`), véletlen seed, modellcsalád |
| `eval_mode`, `is_default_mode` | kiértékelési mód (pl. `fixed`, `soft`, `hard_surv_0.5`, `random_routes`); az alapértelmezett inferencia-mód jelölve |
| `validity_note` | miért nem olvasható a sor úgy, ahogy a neve sugallja (üres, ha nincs ismert probléma) |
| `params_total`, `params_non_embedding` | tárolt paraméterek |
| `val_loss`, `val_ppl`, `val_bpb` | validációs keresztentrópia (nat/bájt), bájtonkénti perplexity, bit/bájt |
| `mean_hops`, `p95_hops` | végrehajtott hopok tokenenként (Dense/MoE: rétegszám) |
| `est_flops_per_token_impl`, `est_flops_per_token_ideal` | analitikus forward-FLOP tokenenként, ahogy implementálva van / csak az aktív tokenek attentionjével |
| `eval_batch_ms_with_diagnostics` | kiértékelési batch ideje diagnosztikával (nem késleltetési eredmény) |
| `expert_usage_entropy_norm`, `dead_experts`, `max_expert_share` | routing-kiegyensúlyozottság (Dense-nél üres) |
| `test_loss_default_mode` | test split vesztesége, csak az alapértelmezett módban |
| `train_tokens`, `train_seconds`, `train_tokens_per_s`, `train_peak_vram_mb` | tanítási keret és költség |
