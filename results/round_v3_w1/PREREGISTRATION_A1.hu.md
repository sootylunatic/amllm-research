---
title: "W1 előregisztráció — A1 módosítás (a hipotézisfutások indítása előtt)"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "RESULT"
source_of_truth: true
supersedes: []
superseded_by: []
locked: true
---

[English](PREREGISTRATION_A1.md) | **Magyar**

# W1 előregisztráció — A1 módosítás (a hipotézisfutások indítása előtt)

**Állapot:** A jóváhagyott és lezárt [W1 előregisztrációt](PREREGISTRATION.hu.md) (`PREREGISTRATION.sha256`) nem írjuk át; ez a kiegészítés **dátumozott módosítás**, amelyet a felülvizsgálat két igazolt találata indokol. Az FD/AD hipotézisfutások (2–3. rész) a módosítás rögzítése után indulnak; ekkor még egyetlen FD/AD eredmény sem létezik, a horgonyok nem érintettek. Ütközés esetén a módosítás az irányadó.
**Dátum:** 2026-10-23. **Kódverzió:** változatlan (`ba81b4fc02cd7b63`).

## A1.1 Az AD sikerkapu és a fő feladatsor összhangja (a 3. rész kapuja helyett)

A fő feladatsor (RESEARCH_TASKS, AD-kapu) a **ténylegesen végrehajtott** számítás előnyét kéri. Az útválasztott `flops_per_token` ugyanakkor csak szűrőjelzés: a megállt token attentionje kiszámolódik, és a bmm-padding is munka. Ezért:
- **Szűrőjelzés (W1):** ha a `hard_surv_0.5` módban a val NLL ≤ 1,03 × a B_moe átlaga **és** a `flops_per_token` ≤ 0,90 × a B_moe-é, 3/3 seeden, az ítélet **`PROMISING_EXPLORATORY`**. Ez nem siker.
- **Sikerkapu (`SUPPORTED_IN_SCOPE`)** csak akkor adható, ha a W3 valódi végrehajtási mérése (padding-mentes, a kihagyott munkát ténylegesen elhagyó út; a számolt sorok és attention-queryk száma, padding beleszámítva; decode-késleltetés) szerint a **végrehajtott** munka ≤ 0,90 × a B_moe végrehajtott munkája, a val NLL feltétellel együtt. A W1 egymaga ezt nem adhatja.
- A „Terjedelem" 4. pontjában a „`SUPPORTED_IN_SCOPE (routed FLOP)`" kifejezés érvénytelen.

## A1.2 Az FD01 oksági állításának korlátozása és új kontroll (FD01d)

Az FD01b egyszerre változtat **két** tényezőt a B_fixed-hez képest: megszünteti a bankmegosztást, és a hoponként elérhető szakértőkészletet 24-ről 4-re csökkenti (az azonos teljes paraméter miatt). Ezért:
- Az FD01b–B_fixed különbség **csomag-hatás** („hoponként külön kis bank vs. egy közös nagy bank, azonos teljes paraméter"), nem a megosztás tiszta hatása. Ilyen okozatot nem állítunk.
- **Új kontroll FD01d:** `bank_groups=1 n_modules=2 experts_per_module=2` — egyetlen, 4 szakértős bank, amelyet mind a 6 hop közösen használ (kb. 11,2M paraméter). A 2×2-es elrendezés négy cellája (megosztás × hoponkénti készlet): B_fixed (megosztott, 24), FD01d (megosztott, 4), FD01b (külön, 4), FD01a (külön, 24).
- **Oksági állítás csak azonos hoponkénti készletű párokból:** a megosztás hatása = FD01b − FD01d (készlet 4) és FD01a − B_fixed (készlet 24); a készlet hatása = B_fixed − FD01d (megosztott) és FD01a − FD01b (külön). Mindegyik páros nem azonos teljes paraméterű, ezt a jelentés minden sornál kiírja. Az „azonos paraméter" ítélet (FD01b−B_fixed) csomag-hatásként marad.
- A döntési szabály (irányos, |Δ| ≥ 0,010, CI, 3/3) minden pár-összevetésre ugyanaz.
- **Terjedelem:** +3 futás (FD01d × 3 seed): a W1 összesen 12 + 24 + 6 = **42 futás**.
