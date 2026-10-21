[English](0008-control-models.md) | **Magyar**

# ADR-0008: Kontrollmodellek azonos aktív FLOP-ra és azonos teljes paraméterre illesztve

- **Státusz:** elfogadott — kiegészítve: ADR-0016
- **Dátum:** 2026-10-21
- **Érintett kód:** `amllm/model.py: LayeredLM`, `amllm/experiment.py: PRESETS`

## Kontextus
Egy ritka modell könnyen „nyer”, ha több paramétere vagy több számítása van. Két tengelyen kell összehasonlítani,
és el kell választani a súlymegosztás hatását a modularitásétól.

## Döntés
Közös: d_model = 128, 4 fej, kontextus 256, byte-tokenek, azonos optimalizáló és tokenkeret.
- **Azonos aktív FLOP (~1,9 MFLOP/token):** Dense (4 réteg, FFN 512); MoE (4 réteg, 8 expert × 256, Top-2);
  UT (1 közös blokk × 4 hop, routing nélkül — a súlymegosztás kontrollja); AM-LLM (8×4 expert × 128, Top-2/Top-2, 4 hop).
- **Azonos teljes paraméter (~2,2–2,4M nem-embedding):** Dense széles (FFN 2048), MoE, AM-LLM 16 modul.
- **Csökkentett számítás (~0,85–1,0 MFLOP/token):** Dense 2 réteg, MoE 2 réteg, erős haltingú AM-LLM.
- FLOP: analitikus forward-becslés (`attention_flops`, `ffn_flops`), „implementált” és „ideális” változatban,
  `torch.profiler`-rel ellenőrizve.

## Következmények
- A maradék különbségek (paraméterszám, attention-megosztás) minden táblában jelölve.

**Javítás (2026-10-21, audit):** az első jelentés a UT → AM-LLM javulást (−11% veszteség) a routingnak tulajdonította.
A UT 297 ezer, az AM-LLM 1,23 millió paraméteres, így a routing és a tárolt kapacitás együtt változott. Az E09
azonos paraméterű és FLOP-ú véletlen-routing kontrollt ad; az eredmény az ADR-0016-ban.
