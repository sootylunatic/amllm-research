---
title: "AM-LLM — kutatási célok és elfogadási keretrendszer (A–D)"
version: "1.0"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "GOALS"
source_of_truth: true
supersedes: ["docs/archive/Phase3 backlog §0 goal sentence"]
superseded_by: []
---

[English](RESEARCH_GOALS.md) | **Magyar**

# AM-LLM — kutatási célok és elfogadási keretrendszer (A–D)


**Verzió:** 1.0 — 2026-10-23  
**Státusz:** aktuális, javasolt központi célmeghatározás; nem változtatja meg a korábbi előregisztrációkat.  
**Hatókör:** az AM-LLM hosszú távú kutatási programja. A kísérleti feladatok, kiértékelések és architekturális variánsok erre hivatkozzanak.  
**Elsődleges történeti forrás:** az eredeti első kísérleti megbízás (`goal.md`, belső irat) §2 (első kísérletsorozat, 2026-10-20/21).  
**Eredmények:** `../../results/EXPERIMENT_REPORT.hu.md`, `../../results/overnight_v2_20261022/NIGHT_RESEARCH_REPORT_HU.md`.

## 1. Mi változatlan az eredeti hipotézisből?

Az eredeti, 2026-10-20-i/21-i megbízás vizsgálandó kérdése:

> Egy hierarchikus modulrouterrel, modulon belüli szakértőválasztással és dinamikus feldolgozási mélységgel rendelkező modell képes-e a Dense Transformer és a fix mélységű MoE predikciós minőségét megközelíteni **kevesebb aktív számítással**?

Ez a **tudományos alaphipotézis (A)**. Nem írjuk át a későbbi sikeres vagy sikertelen kísérletek kedvéért. A korai `goal.md` *feladatmegbízás*, nem a teljes jövőbeni program aktuális szabályzata. Az első kör eredeti kapuja: legfeljebb ~5% loss-romlás, legalább ~15% számításmegtakarítás **vagy** azonos számításon jobb minőség, legfeljebb ~10% latency-regresszió, tanuló router, változó mélység, három seed. Ez **csak az első kör korabeli döntési kritériuma**, nem az új fázisokra automatikusan kivetett küszöb.

## 2. Az aktuális A–B–C–D célrendszer

### A — Tudományos alaphipotézis: jobb minőség/számítás hatékonyság

**Kérdés:** a tanulható modul-/expertrouting, a többszörösen felhasználható modulok és a bemenethez igazított feldolgozás képes-e **azonos vagy megfelelően közeli minőséget kevesebb ténylegesen végrehajtott aktív számítással** biztosítani egy hitelesen illesztett Dense/MoE-kontrollnál?

- **Központi mérés:** validációs NLL/perplexity és feladat-alapú minőség vs **végrehajtott** FLOP/token, kernelmunka, FFN-/attention-sorok, kontextus, batch és modellméret szerint.
- **Sikerkövetelmény:** előre rögzített, reprodukált minőség–számítás Pareto-előny; a csupán *ideális* FLOP-csökkenés vagy kevesebb átlagos hop önmagában nem siker.
- **Kulcshipotézisek:** dinamikus mélység; szelektív figyelem-/expert-hívás; Top-K; modul- és hop-specializáció; strukturált blokk-/mélységválasztás.
- **Sikertelen eset:** egy fix, nyolc hopos variáns MoE-közeli loss-t ad, de több számítást igényel — jó képességkontroll, **nem A-szintű igazolás**.

### B — Rendszermemória: alacsony valódi GPU-munkakészlet

**Kérdés:** a nem szükséges modulok/expert-súlyok maradhatnak-e CPU RAM-ban vagy háttértáron úgy, hogy a GPU-n csak a ténylegesen szükséges súlyok, KV-cache és futási pufferek foglalnak helyet, elfogadható cache-miss költséggel?

- **Célkörnyezet:** elsősorban fogyasztói GPU; a **16 GB VRAM** hosszú távú célkorlát a nagy modellre, nem az első 17M-es mikrotesztek sikerkapuja.
- **Mérés:** valódi inference peak/allokált VRAM, rezidens súlyok, KV és scratch külön; cache-hit/miss, host↔GPU bájt/token, p95 miss-latency, teljes munkakészlet.
- **Sikerkövetelmény:** ténylegesen futó szelektív betöltés/offload és hitelesen kisebb VRAM-igény hasonló minőség mellett; csak szimulált caching vagy ritka routing még nem elég.
- **Nem előfeltétel:** a *teljes* modellsúlynak vagy összparaméterszámnak nem kell kisebbnek lennie. A teljes méret mérendő; egy RAM-ban tárolt nagy expertbank indokolt lehet.

### C — Valódi futási sebesség és használhatóság

**Kérdés:** a dinamikus végrehajtás a kernel-, routing-, szinkronizációs és paging-költségekkel együtt képes-e **legalább MoE-közeli**, később jobb valós generálási sebességre?

- **Mérés:** autoregresszív decode token/s batch 1 és többfelhasználós terhelésen, TTFT, prefill token/s, p50/p95 latency, 32K kontextus környezete és saját VRAM/host RAM; azonos GPU, kvantálás, minőségi és backendbeállítású kontroll.
- **Hosszú távú mérnöki cél:** **≥60 dekódolt token/s** hasznos kódolási minőség mellett egy nagyobb modellen a deklarált fogyasztói hardveren. Nem egy 17M-es modellre extrapolált teljesítmény.
- **Sikerkövetelmény:** mért sebességelőny vagy legalább közeli sebesség; a tanítási token/s és a teljes-szekvenciás forward-ms nem dekódolási bizonyíték.

### D — Végső stratégiai cél: nagy modellképesség fogyasztói hardveren

**Vízió:** olyan nagy összkapacitású, feladatfüggően kevés részt aktiváló AM-LLM, amely lényegesen nagyobb hagyományos Dense/MoE-modellek **valós problémamegoldó képességét** megközelíti kódolási, implementációs, hibakeresési és később általános feladatokon, miközben az A–C feltételeket együtt teljesíti.

- **Illusztratív méretpéldák, nem vállalt vagy igazolt eredmények:** ~27B méretű AM-LLM a nála jóval nagyobb modell minőségéhez közel; később akár ~200B összkapacitású modell, amely nagyobb, például ~377B-s MoE-vel versenyez, ha az adatok és képzés ezt indokolják.
- **Egy nagyobb teljes modellméret nem önmagában probléma**, amennyiben az aktív útvonal, a GPU-munkakészlet és a dekódolási késleltetés kezelhető. A kisebb *összsúly* előny, de **nem alapfeltétel**.
- **Képességmérés:** szivárgásmentes, nehézség és kódtípus szerint bontott TypeScript/React/NestJS hibajavítás, implementáció és pass@1; hosszabb távon független nagy-modellel azonos protokollban, azonos promptokkal, több seed-/mintafeltétellel.
- **Státusz:** távoli kutatási cél. Az E00–F5 és a 36 éjszakai kísérlet sem bizonyítja a nagy modellskálán a D-t.

**Kapcsolat:** A = az eredeti kutatási *miért*; B és C = a megvalósíthatóság két önálló mérnöki bizonyítéka; D = a végső *mire akarjuk használni*. Egyik pont sem helyettesíti a másik mérését.

### 2.1 Az architektúracsalád és az eredeti hipotézis (döntés, 2026-10-23)

A fix mélységű, hoponként külön attentiont használó változat az AM-LLM architektúracsalád tagja. Az **architektúracsalád** és az **eredeti teljes hipotézis** (dinamikus számítás) azonban nem ugyanaz.

| Komponens | Meghatározás |
|---|---|
| Újrafelhasználható modulbank | az architektúra alapvető része |
| Hierarchikus modul- és expertrouting | az architektúra alapvető része |
| Több feldolgozási hop, tanult útvonalakkal | az architektúra alapvető része |
| Dinamikus mélység / halting | az eredeti fő hipotézis része, de nem kötelező minden kontrollváltozatban |
| Hoponként külön attention | megengedett architekturális változat |
| Fix 4/6/8 hop | megengedett kontroll- és kutatási változat |

Elnevezés: **AM-LLM-FixedDepth** — hierarchikus routing és újrafelhasználható modulbank, rögzített mélységgel; **AM-LLM-AdaptiveDepth** — ugyanez tanult, bemenetfüggő számítási mélységgel.

- A FixedDepth változat megközelítheti vagy meghaladhatja a MoE minőségét, de **önmagában nem bizonyítja** az eredeti dinamikus számítási hipotézist (A): ahhoz AdaptiveDepth kell.
- Ha a hierarchikus routing és az újrafelhasználható modulbank is megszűnik, az már **másik architektúra**, nem AM-LLM-változat.

## 3. Ami nem cél, és ami csak lehetséges eszköz

- **Nem kötelező:** egy adott összparaméter-szám elérése, a teljes súlyállomány minimálisra csökkentése, pontosan 400 modul × 100 expert, minden tokennél kevesebb hop.
- **Nem tekinthető automatikus sikernek:** alacsonyabb ideális FLOP; gyorsabb tanítás; jobb loss több számítással; kisebb training VRAM inference-mérés nélkül; jobb CUDA backend azonos backendű MoE-kontroll nélkül.
- **Lehetséges eszközök:** hierarchikus routing, Top-1/Top-2 és adaptív K, hard-/soft-halting és tanítás-tudatos exit, per-hop/részben megosztott attention, recurrence, expertbank-méret, strukturált ritkítás, KV-cache, Triton/CUDA Graph, cache-aware routing és RAM–VRAM paging.
- **Nincs kényszer az eredeti v0.1 architektúra változatlan megtartására (az architektúracsaládot a 2.1 pont rögzíti)** az új fázisokban. Az eredeti architekturális ág azonban maradjon elkülönített, megismételhető történeti kontroll.

## 4. Kutatási bizonyítékok és jelenlegi korlátok (2026-10-23)

**Történeti E00–E09:** a tanult routing a random kontrollnál 19,9%-kal jobb volt 3 seeden; a hoponként külön attention MoE-közeli veszteséget adott. Az eredeti E07 go/no-go kapu **nem teljesült** (ADR-0011). Ez nem változik meg a későbbi architekturális javításoktól.

**F5 / BPE 17M:** a korabeli F5 Q kapu **nem teljesült**. A soft és hard halting között súlyos minőségi rés mutatkozott. A pass@1 később elkészült, de a 17M-es modelleknél többnyire nulla vagy közeli, ezért nem bizonyít kódolási kompetenciát.

**Éjszaka 2026-10-22/23:** 36 tanítás (12 konfiguráció × 3 seed) lezárva; fix 8 hopos AM-LLM átlagos val NLL 2,1212 a MoE 2,1166-jával szemben (1,002×), **+6,9% összparaméterrel és több számítással**. A kevesebb hop mellett jó minőségű hard halting nem igazolódott. A dekódolás b=1 mért eredménye ennél a kísérleti családnál AM 632, MoE 884, Dense 2075 tok/s; ez *kis modell* és nem skálázható közvetlenül 27B-re. Teljes jelentés: `../../results/overnight_v2_20261022/NIGHT_RESEARCH_REPORT_HU.md`.

**F6:** az éjszakai eredményjelentésben jelenleg NO-GO saját runtime-ra. A korábbi F6 terve *történeti terv*, nem bizonyíték arra, hogy elindult vagy sikeres.

**A tudományos állapot összefoglalása:** A **jelenleg nem támogatott a 17M-es skálán, a vizsgált architekturális változatokkal**: a két legjobb változat MoE-közeli minőséget ért el, de 24–31%-kal több FLOP-pal, ami az adott tartományban a hatékonysági hipotézis ellen szól. Ez nem bizonyítja, hogy más routing-, attention- vagy adaptív mélységi megoldással A elérhetetlen. B nagy modellen nem mért; C nagy modellen nem igazolt; D hosszú távú ambíció. A router tanulása pozitív rész-eredmény, de nem szabad ebből A–D sikert állítani.

## 5. Új kísérletek szabályai és célhoz rendelése

- Minden új hypothesist rendeljenek az **A / B / C / D** elsődleges célhoz, és másodlagos célokhoz, ha releváns. A „kísérlet sikere” mindig az előre rögzített *adott* kérdésé, nem automatikusan a teljes projekt sikere.
- Az új kontroll és kapu **új preregisztráció**: a `goal.md` első körös 5%/15%/10% küszöbei, a F5-Q és az ADR-0011 visszamenőleg nem módosíthatók.
- Azonos train/val/test split és tokenizer, párosított seed, megfelelő paraméter/FLOP/quality kontroll; diagnosztika nélkül mért valós inferencia külön a becsült FLOP-tól.
- A 2026-10-22/23 éjszakai tervező a tesztveszteséget is látta: **a következő hipotézisek kiválasztása kizárólag validációs adatokon történjen**, és a végső megerősítéshez lehetőleg új, független holdout szükséges.
- A 400-modulos/27B/későbbi nagyobb modell terv külön skálázási hipotézis; csak új minőség-, memória- és decode-kapuk után indítható.
- A szakmai jelentések a negatív, inkomplett és elvetett eredményeket is közöljék. A „csak soft javul” vagy „csak training gyorsul” nem nevezehető általános győzelemnek.

## 6. Források és dokumentumhierarchia

1. **Ez a dokumentum (`docs/00-goals/RESEARCH_GOALS.hu.md`) az aktuális A–D célok kanonikus leírása.** A párja [`RESEARCH_GOALS.md`](RESEARCH_GOALS.md).
2. az eredeti első kísérleti megbízás (`goal.md`, belső irat) — eredeti első kísérleti megbízás, változatlan történeti bizonyíték.
3. `RESEARCH_OVERVIEW.hu.md` — a tudományos kérdés és állapot rövid áttekintése.
4. [`../02-methodology/METHODOLOGY.hu.md`](../02-methodology/METHODOLOGY.hu.md) — a korai protokoll és az új A–D mérési kiterjesztés.
5. [`../01-architecture/ARCHITECTURE.hu.md`](../01-architecture/ARCHITECTURE.hu.md) — a v0.2.1 kód *történeti* technikai leírása, nem a későbbi F5/F6 branch automatikus specifikációja.
6. `../archive/PHASE2_PLAN.hu.md` — F1–F6 eredeti előregisztrált/tervezett sorrendje, történeti dokumentum.
7. `../archive/Phase3/AM_LLM_ADAPTIV_EJSZAKAI_KUTATASI_BACKLOG_V2_HU.md` — 2026-10-22 adaptív backlog, az abban szereplő v2 célmondatot ez a dokumentum **felülírja**.
8. Az ADR-ek és eredményfájlok megőrzik az adott időpontban érvényes döntéseket, futtatási feltételeket és mérési adatokat; az új célrendszer nem módosítja az archív adatokat.
