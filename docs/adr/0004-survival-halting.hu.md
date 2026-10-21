[English](0004-survival-halting.md) | **Magyar**

# ADR-0004: Kumulatív survival-alapú halting a lépésenkénti küszöb helyett

- **Státusz:** elfogadott (felváltja a v0.1 lépésenkénti szabályát, amely összehasonlításra megmaradt)
- **Dátum:** 2026-10-21
- **Érintett kód:** `ModelConfig.halt_rule` (`survival` | `step` | `none`), `AdaptiveModularLM.forward`

## Kontextus
A v0.1 tanításkor kumulatív túlélési valószínűséggel súlyozott (`s_h = Π(1 − p_i)`), inferenciában viszont a
*lépésenkénti* `p_h ≥ 0,65` feltétellel állt meg. Mérve (E01): tanítás közben ~1,8 várt hop, inferenciában **4,0**,
mert a tanult `p_h` ≈ 0,5 sosem éri el a 0,65-öt. A két szabály nem ugyanazt a modellt valósítja meg.

## Döntés
`survival` szabály: a hop frissítése mindkét módban `s_h`-val súlyozott; hard inferenciában egy token a következő
hopot csak akkor futtatja, ha `s_h ≥ τ`. A hard inferencia így a soft tanítás **csonkítása** ott, ahol a maradék
súly kicsi. Ponder-büntetés: `λ · Σ_h mean(s_h)`. Teszt: `τ → 0` esetén hard ≡ soft (`test_hard_with_tiny_threshold_equals_soft`).

## Következmények
- Kalibrálható: nagyobb τ → kevesebb hop (a `step` szabálynál fordítva viselkedett, E03).
- Mérve: λ ≤ 0,01 mellett a halting alig vág (~1%); λ = 0,03 → 1,7 hop, de +8% veszteség és 4,8% soft/hard rés;
  8 hop + τ = 0,8 → összeomlás (1,46 → 2,89). A rés kezelése nyitott feladat (NEXT_EXPERIMENTS N5).
- A megállt tokenek állapota befagy, de attention-kulcsként/értékként továbbra is szolgál; az attention maga nem
  hagyódik ki (ADR-0006, „ideális” FLOP).

## Megfontolt alternatívák
- ACT-féle maradék-súlyozás: hasonló, de bonyolultabb; a csonkítási értelmezés egyszerűbben tesztelhető.
- Hard (straight-through) halting tanítás közben: későbbi kísérlet (N5).
