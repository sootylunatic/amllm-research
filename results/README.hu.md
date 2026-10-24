[English](README.md) | **Magyar**

# Kutatási eredmények

- [Publikus kutatási jelentés](RESEARCH_REPORT.hu.md): E00–F5 és a 36 éjszakai futás, hibák, CI-k és reprodukálási korlátok.
- [V3 kör, W1 riport](round_v3_w1/W1_REPORT.hu.md): 48 előregisztrált futás (horgonyok, FD01–FD03, FD07, AD02a) seedenkénti értékekkel, döntési szabályokkal, bankonkénti expertstatisztikával és AD01-diagnosztikával; [előregisztráció](round_v3_w1/PREREGISTRATION.hu.md) és kiegészítései, kísérletenkénti `summary.csv`, elemzési JSON a `round_v3_w1/analysis/` alatt.
- [V3 kör, W2 riport](round_v3_w2/W2_REPORT.hu.md): 66 előregisztrált futás (attention-megosztás, illesztett minőség–számítás pontok, FDC1, adaptív K) seedenkénti értékekkel és AD01-diagnosztikával; [előregisztráció](round_v3_w2/PREREGISTRATION.hu.md), kísérletenkénti `summary.csv`, elemzési JSON a `round_v3_w2/analysis/` alatt.
- [V3 kör, W3 riport](round_v3_w3/W3_REPORT.hu.md): az FDC1 az azonos méretű MoE ellen, 2×2 egytényezős elemzés, 18k lépéses futások, 21 checkpoint gyakorlati mérése (végrehajtott munka, késleltetés, dekódolás, VRAM, hotset, generálási hurkok) és feltáró dekódolási sebességmérés; [előregisztráció](round_v3_w3/PREREGISTRATION.hu.md), táblák `analysis_*.txt`, nyers `measure/` és `c_decode/`.
- [V3 kör, W4 riport](round_v3_w4/W4_REPORT.hu.md): egyszeri leíró ellenőrzés független holdouton (154 friss csomag, 36 befagyasztott checkpoint, csomagszintű bootstrap); [előregisztráció](round_v3_w4/PREREGISTRATION.hu.md), összevetés és csomagonkénti aggregátumok a `round_v3_w4/holdout_eval/` alatt (a holdout szövege nem publikus); K2 kauzális router-teszt: `k2_causal.json`.
- [summary.csv](summary.csv): részletes gépi adatok **csak E00–E09** bájtszintű kísérletekhez.
- [Állapot](../docs/RESEARCH_STATUS.hu.md) · [Nyitott kérdések](../docs/OPEN_QUESTIONS.hu.md).

A summary.csv csak azonos experiment/kódverzió/eval módon belül összehasonlítható; a bájt- és BPE-loss nem azonos, a diagnosztikai batch-idő nem decode tok/s, a training VRAM nem inference VRAM. A 17M-es tábla másodlagos összesítés, **nem nyers futásarchívum**.
