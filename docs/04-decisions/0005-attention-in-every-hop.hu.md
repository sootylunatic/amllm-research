[English](0005-attention-in-every-hop.md) | **Magyar**

# ADR-0005: Megosztott kauzális attention minden hopban; a hoponként saját attention a következő jelölt

- **Státusz:** elfogadott (megosztott attention); **javasolt** (hoponként saját attention) — kiegészítve: ADR-0016
- **Dátum:** 2026-10-21
- **Érintett kód:** `ModelConfig.attn_per_hop`, `ModelConfig.unshared_attention`

## Kontextus
A v0.1-ben egyetlen attention futott a hopok előtt; a hopok tokenenkénti FFN-rekurziók voltak, a tokenek a hopok
között nem látták egymást. A Dense/MoE kontrollok rétegenként saját attentiont használnak.

## Döntés
1. `attn_per_hop=True` az alapértelmezés: minden hop elején ugyanaz a (megosztott) kauzális attention fut
   (Universal-Transformer-szerű rekurzió). A v0.1-topológia `attn_per_hop=False` ablációként megmaradt.
2. A `unshared_attention=True` opció hoponként saját attention-súlyokat ad (a modulbank közös marad).
   Ez az E07 után, feltáró jelleggel került be.

## Következmények
- Mérve (E06): a v0.1-topológia rosszabb (1,622 vs. 1,526).
- Mérve (feltáró, 3 seed, azonos FLOP): hoponként saját attentionnel 1,526 → **1,347** (MoE: 1,333; Dense: 1,430).
  Feltáró eredmény (az E07 *után* fogalmazódott meg); megerősítő kísérlet kell előre rögzített feltételekkel.
- A hoponként saját attention pontosan **198 912** paramétert ad (három további attention-blokk a LayerNormjaikkal).

**Javítás (2026-10-21, audit):** a jelentés első változata ezt a javulást egyedül a „megosztás megszüntetésének”
tulajdonította („legnagyobb egyedi tényező”). A megosztás megszüntetése és a kapacitásnövekedés együtt változott,
így ez a tulajdonítás nem volt alátámasztva. Az E09 kapacitásillesztett kontrollt ad; az eredmény az ADR-0016-ban.
