[English](ARCHITECTURE_GUIDE.md) | **Magyar**

# Architektúra — fix és adaptív számítás

**Hatókör:** ~17M BPE F5/éjszakai változatok; nem a [történeti v0.2.1 bájtmodell](01-architecture/ARCHITECTURE.hu.md).

```text
6 rétegű MoE                        6 hopos AM-LLM FixedDepth
Attn_1 → saját expertek 1           Attn_1 → közös bank (24 expert)
Attn_2 → saját expertek 2           Attn_2 → ugyanaz a bank
...                                 ...
Attn_6 → saját expertek 6           Attn_6 → ugyanaz a bank
```

Mindkettőben **6 × 591 360** attention-paraméter és összesen **7 096 320** expertparaméter van. A MoE hat rétegében négy-négy expertből két expertet választ; az AM 6 modul × 4 expertet használ, előbb modult, majd expertet választ, és minden hopban újrahasználja a bankot. **Három tényező egyszerre változott:** közös expertek, hierarchikus routing, közös bank-LayerNorm. Egyedi okozati hatásuk **még nincs külön mérve**. A router ~11 550 paramétere a teljes modell ~0,07%-a.

| Modell | Mélység | Routing | Becsült MFLOP/token |
|---|---:|---|---:|
| MoE kontroll | 6 réteg | Top-2 a 4-ből/réteg | 22,834 |
| FixedDepth alap | 6 hop | 1 modul × 2 expert | 22,857 |
| FixedDepth 8 hop | 8 hop | 1 × 2 | 28,379 |
| FixedDepth Top-2/Top-2 | 6 hop | 2 × 2 | 29,954 |
| FixedDepth kisebb expertek | 6 hop | 1 × 2 keskeny | 19,337 |

Közös keret: d_model=384, 6 attention fej, 8192 BPE-token, 512-es kontextus, hoponként külön attention, SiLU és straight-through routing. A kimeneti fej **~6,291 MFLOP/token (~28%)**, amit a halting nem spórol meg. A becsült FLOP **nem mért generálási sebesség**.

## A halting problémája

A soft survival tanításkor **minden hopot végrehajt**, a frissítéseket túlélési súllyal skálázza. A hard inferencia **lefagyasztja a leállt tokeneket**, amelyek későbbi attention K/V-ként is jelen vannak. A hard csonkolt útvonalra nem történik közvetlen tanítás. Az F5 seed 42 soft 2,183-as vesztesége hard θ=0,5 esetén ~3,426-ra romlott, ~3,47 hop mellett. Egyszerű küszöbhangolás nem adott jó kompromisszumot.

A **FixedDepth** hasznos kontroll és az AM-LLM architektúracsalád része, de nem igazolja a dinamikus számítást. Az **AdaptiveDepth** ezt az eredeti hipotézist vizsgálja. Valós CPU–GPU expert-paging nincs. Szükséges: kontrollált abláció, hoponkénti loss, hard-aware training, intermediate exit loss és szelektív hopon belüli számítás. [Nyitott kérdések](OPEN_QUESTIONS.hu.md) · [Eredmények](../results/RESEARCH_REPORT.hu.md).
