[English](0009-measurement-protocol.md) | **Magyar**

# ADR-0009: Mérési protokoll és előre rögzített döntési szabályok

- **Státusz:** elfogadott — kiegészítve: ADR-0014
- **Dátum:** 2026-10-21
- **Érintett kód:** `amllm/train.py`, `amllm/profile.py`, `amllm/router_analysis.py`, `results/e03_halting/PREREGISTRATION.md`

## Kontextus
Kis modelleknél a seedek közti szórás összemérhető a modellek közti különbséggel; a hiperparaméter-választás
utólagos igazítása könnyen „sikert” gyárt.

## Döntés
- Validáció: 512 rögzített, egyenletesen elosztott ablak a val splitből — minden modellnél ugyanaz.
- Alapértelmezett inferencia-mód: teljes mélység; adaptív AM-LLM-nél hard survival halting, τ = 0,5.
- Fő összehasonlítás 3 seeddel (42/43/44), átlag ± szórás és seedenkénti nyers értékek; páros (azonos seedű) összevetés.
- Hiperparaméter-választás (pl. ponder λ) **a futtatás előtt írásban rögzített** szabállyal; az eredmény után nem módosítjuk.
- Latency: CUDA-szinkronnal, warmup + ismétlés, medián és p95, teljes szekvenciás forward —
  **nem** generált token/s. VRAM: csak a ténylegesen mért csúcs; megtakarítást nem állítunk (minden súly a GPU-n).
- Router: véletlen-routing abláció (azonos súlyok) és kölcsönös információ a bájtosztállyal, tanítatlan kontrollal.
- Minden futás: `config.json`, `env.json` (verziók, hardver, forrás-hash), `train.log`, `report.json`.

## Következmények
- A célértékeket (≤5% minőségromlás, ≥15% számításcsökkenés, ≤10% latency-regresszió) nem igazítottuk az eredményhez;
  a jelentés kritériumonként rögzíti, mi teljesült.
