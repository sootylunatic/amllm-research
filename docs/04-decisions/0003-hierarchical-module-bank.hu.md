[English](0003-hierarchical-module-bank.md) | **Magyar**

# ADR-0003: Megosztott modulbank, hierarchikus Top-K routing (modul → expert)

- **Státusz:** elfogadott (a kutatási hipotézis tárgya)
- **Dátum:** 2026-10-21
- **Érintett kód:** `amllm/routing.py` (`HierarchicalModuleBank`, `ExpertBank`), `amllm/model.py` (`AdaptiveModularLM`)

## Kontextus
A kutatási kérdés: képes-e egy modell tokenenként és lépésenként kiválasztani, *mely* feldolgozó egységeket
futtassa, és ezzel kevesebb aktív számítással hasonló minőséget elérni. A hagyományos Transformer minden rétegen
végigvisz minden tokent; a MoE rétegenként választ, de a rétegsorrend rögzített.

## Döntés
- Egyetlen, minden hopban **újrahasznosított** modulbank: `M` modul × `E` expert (SiLU FFN).
- Hoponként és tokenenként: modulrouter → Top-`Km` modul (softmax a kiválasztottak pontszámain),
  majd a kiválasztott modul saját expert-routere → Top-`Ke` expert.
  A kimenet: `Σ_m Σ_e w_m · w_e · FFN_{m,e}(x)`.
- Az expert-router csak a modult kiválasztó tokenekre fut (feltételes számítás).
- Switch-stílusú terheléskiegyenlítő veszteség (differenciálható „importance” × leválasztott „load”) mindkét szinten.
- A plain MoE a speciális eset: `M=1, Km=1` — azonos kód, így a kontrollmodell implementációs különbsége minimális.

## Következmények
- Mérve (E07, 3 seed): a router tanul — véletlen routinggal +2,99 nat; MI a bájtosztállyal 2–4× a tanítatlan kontrollé;
  nincs összeomlás (normált entrópia 0,98–0,99).
- Mérve: azonos aktív FLOP mellett rosszabb minőség a Dense (+6,8%) és a MoE (+14,5%) kontrollnál (lásd ADR-0011).
- Az alapértelmezett súlyozás mellett a Top-1 router (modul vagy expert szinten) nem kap nyelvimodell-gradienst (ADR-0013).

## Megfontolt alternatívák
- Laposan indexelt `M·E` expert egy routerrel: elveszítené a „modul” mint feladatszintű egység (később cache-elhető) fogalmát.
- Dense „soft” keverés minden modulra: nem ritka, nincs számítási megtakarítás.
