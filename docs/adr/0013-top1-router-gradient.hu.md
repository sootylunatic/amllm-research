[English](0013-top1-router-gradient.md) | **Magyar**

# ADR-0013: A Top-1 router renormált súlyokkal nem kap LM-gradienst; teljes-softmax súlyozás bevezetése

- **Státusz:** elfogadott (felváltja: ADR-0010)
- **Dátum:** 2026-10-21
- **Érintett:** `HierarchicalModuleBank._select`, `ModelConfig.route_weighting`, tesztek: `test_topk_softmax_top1_router_gets_no_lm_gradient`, `test_full_softmax_weighting_gives_top1_routers_gradient`

## Kontextus
Az alapértelmezett útvonal-súly a *kiválasztott* pontszámokon vett softmax. `k = 1` esetén ez azonosan 1, így az adott
szint routere csak a terheléskiegyenlítő veszteségtől kap gradienst. Egy külső audit rámutatott, hogy ez nemcsak a
modul (`Km = 1`), hanem az **expert** szinten (`Ke = 1`) is igaz; mindkét szinten gradiensteszttel ellenőrizve.
Az E06 „modul Top-1” és „expert Top-1” sora így olyan routert mér, amely a nyelvi veszteségből nem tud tanulni —
nem a Top-1 routing képességét.

## Döntés
- Az alapértelmezés marad `route_weighting="topk_softmax"`, hogy az E00–E08 eredmények az őket előállító kódra érvényesek maradjanak.
- Új opció: `route_weighting="full_softmax"` (Switch-Transformer-stílus): a súly az **összes** pontszámon vett softmax
  a kiválasztott útvonalaknál, így minden router minden `k` mellett kap gradienst.
- Az E09 mindkét szinten újraméri a Top-1-et `full_softmax`-szal, `full_softmax` Top-2 referenciához képest.
- Az E06 Top-1 sorai a Top-1 routing méréseként érvénytelennek jelölve.

## Következmények
- A hibát mindkét szinten teszt rögzíti; a javítás gradiensét teszt ellenőrzi.
- A `full_softmax` megváltoztatja a kimenet skáláját (a súlyok összege nem 1), ezért `full_softmax` futásokat csak egymással hasonlítunk.
