[English](0002-byte-level-tokenization.md) | **Magyar**

# ADR-0002: Az első kísérletsorozat byte-szintű tokenizálást használ

- **Státusz:** elfogadott (az első kísérletsorozatra)
- **Dátum:** 2026-10-21
- **Érintett kód:** `ModelConfig.vocab_size = 256`, `amllm.data`

## Kontextus
Az architektúra-összehasonlításhoz minden modellnek ugyanazt a reprezentációt kell kapnia. Egy BPE-tokenizer
tanítása és verziózása további változót hozna be, és a kis modellek (~1–2,5M paraméter) számára a nagy
szókincs-mátrix uralná a paraméterszámot.

## Döntés
Bájt = token (szókincs: 256), kontextus 256 bájt. A veszteséget nat/bájtban és bit/bájtban (bpb) jelentjük.

## Következmények
- Egyszerű, tokenizer-független, minden modellnél azonos.
- A perplexity bájt-szintű, nem hasonlítható össze token-szintű LLM-ekével.
- A kódolási képesség (pl. `tsc`, `pass@1`) ezen a szinten nem értékelhető; ez a következő kör feladata (BPE).
