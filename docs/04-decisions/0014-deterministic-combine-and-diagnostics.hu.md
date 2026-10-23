[English](0014-deterministic-combine-and-diagnostics.md) | **Magyar**

# ADR-0014: Determinisztikus expert-összegzés, szinkronmentes statisztika és diagnosztika nélküli mérési út

- **Státusz:** elfogadott (kiegészíti: ADR-0006, ADR-0009)
- **Dátum:** 2026-10-21
- **Érintett:** `ExpertBank.forward(routes_per_token=…)`, `count_ids`, `set_diagnostics`, `amllm.profile`, `tests/test_cuda.py`

## Kontextus
Egy külső audit két hiányosságot talált:
1. a késleltetést úgy mértük, hogy minden forwardban futott a routing-statisztika, a terheléskiegyenlítő veszteség és
   egy host-oldali FLOP-riport (CUDA-n minden `float(tensor)` és `torch.bincount` host-szinkron);
2. minden helyességi teszt CPU-n/fp32-ben futott, holott a tanítás és a kiértékelés CUDA-n/bf16-ban.

A CUDA-ellenőrzések hozzáadásakor kiderült:
- az AM-LLM forward CUDA-n nem volt bitre determinisztikus (a tokenenkénti `Km·Ke` expert-kimenet atomikus
  `index_add`-dal összegződött; fp64-ben ~1e-15, az útvonalak azonosak);
- bf16-ban egy prefix-logit 1 ulp-ot változhat, ha a jövőbeli tokenek változnak, mert egy expert eltérő sorszámú
  GEMM-je ugyanazt a sort másképp kerekítheti. fp64-ben az eltérés ~1e-15, tehát ez batch-összetételi numerika,
  nem információáramlás.

## Döntés
- Az expert-kimenetek rögzített sorrendű összegzéssel adódnak össze (`index_copy` vissza tokenrendbe, majd `view(N, k, D).sum(1)`); a CUDA-forward bitre reprodukálható.
- A számlálás `scatter_add`-dal (`count_ids`) történik `torch.bincount` helyett (nincs host-szinkron).
- `set_diagnostics(model, False)`: nincs balance-veszteség, modulstatisztika és FLOP-riport, így csak a szükséges szinkronok maradnak: hoponként egy a dispatch darabszámaihoz, és hard módban hoponként egy az aktív tokenek indexéhez.
- Az `amllm.profile` a késleltetést diagnosztikával **és** anélkül is méri, és számolja a host-szinkronokat (CUDA sync-debug mód, alsó becslés).
- CUDA-tesztek: fp64-ben pontos kauzalitás, bf16-ban kerekítésen belüli kauzalitás, `loop`/`bmm` forward és minden gradiens egyezése, determinisztikusság. Az E02 invariánsok CPU-n és CUDA-n (fp64) is futnak.

## Következmények
- A tanítás/kiértékelés numerikája kerekítési szinten eltér az E00–E08 kódtól, ezért az E09 a referenciákat is újratanítja.
- A késleltetés-összevetés most elkülöníti a modell szükséges végrehajtását a mérőkódtól.
