[English](README.md) | **Magyar**

# AM-LLM — Adaptív moduláris nyelvi modell

![status](https://img.shields.io/badge/status-research%20prototype%20%E2%80%94%20main%20hypothesis%20not%20yet%20supported-orange)
[![tests](https://github.com/sootylunatic/amllm-research/actions/workflows/tests.yml/badge.svg)](https://github.com/sootylunatic/amllm-research/actions) ![license](https://img.shields.io/badge/license-MIT-blue)

**Nyílt kutatási program, nem termék.** Az AM-LLM azt kérdezi, megtanulhatja-e egy nyelvi modell, hogy tokenenként *mely újrafelhasználható feldolgozó egységeket, milyen sorrendben és hányszor* futtassa, és így a Dense Transformer vagy a Mixture-of-Experts (MoE) minőségét megközelítse kevesebb, **ténylegesen végrehajtott** számítással. A repó a prototípus kódját, a teszteket, az út közben hozott döntéseket és annak őszinte beszámolóját tartalmazza, hogy mi igazolódott és mi nem. A negatív eredmények a rekord részei.

## A fő cél

**Nagy, valóban hasznos nyelvi modellt — elsősorban kódoláshoz — futtatni fogyasztói GPU-n**: egy sokkal nagyobb Dense/MoE modell minőségének nagy részét a *végrehajtott* számítás töredékével, kis *inference* memóriaigénnyel (hosszú távú cél: 16 GB VRAM) és gyors helyi generálással (hosszú távú cél: ≥ 60 dekódolt token/s). A hosszú távú, illusztratív méretek (például ~27B tárolt paraméter több száz modulban) **törekvések, nem eredmények**. Minden alább leírt lépés annak tesztje, hogy ez egyáltalán lehetséges-e: kicsiben indulunk, és nem skálázunk fel, amíg a kis modell nem mutat nyereséget.

## A hipotézis

> **H (központi):** egy **közös, újrafelhasználható modulbankkal**, **kétszintű routinggal** (modul, majd expert) és **tanult, tokenenkénti mélységgel** rendelkező modell a Dense Transformer vagy a fix mélységű MoE minőségét **kevesebb, ténylegesen végrehajtott számítással** is elérheti.
>
> **H₀ (nullhipotézis):** a minőségromlás, a routing többletköltsége és a memóriaforgalom kioltja az előnyt. A program ezt az eredményt is elfogadja, ha az adatok ezt mutatják.

Úgy van megfogalmazva, hogy megbukhasson. Az első kört előre rögzített küszöbökhöz mértük: a validációs veszteség legfeljebb ~5%-kal rosszabb a kontrollnál, legalább ~15% kevesebb végrehajtott számítás (vagy azonos számításon jobb minőség), legfeljebb ~10% késleltetés-romlás, bizonyíthatóan tanuló router, működő dinamikus mélység, és ugyanaz a következtetés három seeden.

| A hipotézis része | Állapot |
|---|---|
| **H1** a router stabil, bemenetfüggő útvonalakat tanul | **Alátámasztott** (kis modelleken) |
| **H2** azonos minőség kevesebb végrehajtott számítással | **Nem támogatott ~17M-en** |
| **H3** az egyszerű tokenekhez kevesebb lépés elég (adaptív mélység) | Részben: a mélység tartalomfüggő, de a jelenlegi halting minőséget veszít |
| **H4** a specializáció javítja a kódolási minőséget | Ezen a skálán nem eldönthető |
| **H5** az előrejelezhető modulbetöltés csökkenti a memóriát | Nincs megvalósítva, nem tesztelt |

## Az ötlet

```text
Dense / MoE                          AM-LLM
token ─► réteg 1 ─► réteg 2 ─► …     token ─► [ modulrouter ] ─► modul ─► [ expertrouter ] ─► expertek
         (rétegenként saját súly,           └────────── egy közös bank, minden hopban újra használva ◄──────┘
          rögzített mélység)                ─► hop 1 ─► hop 2 ─► … ─► (opcionális) tokenenkénti megállás: kell még egy hop?
```

- **Közös modulbank:** M modul, mindegyikben néhány kis expert hálózat, amelyeket minden feldolgozási lépésben („hopban”) újra használunk.
- **Kétszintű routing:** tokenenként és hoponként egy router modulokat, azokon belül egy expertrouter experteket választ.
- **Adaptív mélység (opcionális):** egy megállási fej dönti el tokenenként, kell-e további hop.

A családon belül két tagot különböztetünk meg: **FixedDepth** (a bank és a routing rögzített hopszámmal) és **AdaptiveDepth** (ugyanez tanult, bemenetfüggő mélységgel). A FixedDepth jogos kontroll és kutatási tárgy, de önmagában **nem** teszteli a hipotézis dinamikus számítási részét. [Architektúra-útmutató](docs/ARCHITECTURE_GUIDE.hu.md) · [a v0.2.1 implementáció](docs/01-architecture/ARCHITECTURE.hu.md).

## A négy cél: A–B–C–D

A fő célt négy külön bizonyítandó állításra bontjuk, mert az egyik jó eredménye könnyen összetéveszthető a másikéval.

| | A kérdés | Mi számít bizonyítéknak | Mi **nem** számít | Állapot |
|---|---|---|---|---|
| **A** Minőség végrehajtott számításonként | Azonos minőség kevesebb *végrehajtott* számítással, mint egy illesztett Dense/MoE? (az eredeti tudományos hipotézis, H2) | Előregisztrált, reprodukált minőség–számítás előny illesztett kontrollokkal szemben, független adaton | Az *ideális*, de nem végrehajtott FLOP-csökkenés; önmagában a kisebb átlagos hopszám; jobb loss több számítással | **~17M-en nem támogatott**: a legjobb változatok csak több számítással vagy több paraméterrel érik el a MoE veszteségét; 0,85× routolt FLOP-on az AM-LLM kikap az azonos paraméterű MoE-tól (V3 W2) |
| **B** Inference memória | Kisebb valós GPU-munkakészlet, a ritkán használt súlyok RAM-ban vagy lemezen tartásával? | Mért inference VRAM (súly, KV-cache, pufferek) és host→GPU forgalom működő offload mellett | A tanítási memória; a cache *szimulációja*; kevesebb teljes paraméter önmagában | **Nem igazolt** |
| **C** Generálási sebesség | Valós autoregresszív dekódolás legalább MoE-közeli, később jobb? | Mért prefill/decode token/s, TTFT, késleltetési percentilisek, a MoE-kontrollal azonos GPU-n és backenden | A tanítási token/s; a teljes szekvenciás forward ideje | **Nem igazolt**; első mérés: AM-LLM 632 vs MoE 884 tok/s, batch 1 |
| **D** Képesség | Egy sokkal nagyobb modell hasznos kódolási és problémamegoldó képessége fogyasztói hardveren, az A, B és C együttes teljesítésével? | Független, szivárgásmentes kódolási tesztek sokkal nagyobb modellekkel szemben | A 17M-es alacsony loss extrapolálása 27B-re vagy 200B-re | **Hosszú távú cél, nem tesztelt** |

Így viszonyulnak egymáshoz (a kanonikus céldokumentum szerint): **az A az eredeti tudományos *miért*; a B és a C a megvalósíthatóság két, egymástól független mérnöki bizonyítéka; a D a végső *mire* akarjuk használni, és az A, B és C együttes teljesítését igényli.** Egyik cél sem helyettesíti a másik mérését. [Az eredmények megítélése](docs/00-goals/RESEARCH_GOALS.hu.md) · [bizonyítékok állapota](docs/RESEARCH_STATUS.hu.md).

## A kutatási program áttekintése

| Fázis | Skála | Mit kérdeztünk | Mi lett belőle |
|---|---|---|---|
| **v0.1 prototípus** (2026-10-20) | bájtszintű, apró | Egyáltalán működik a mechanizmus? | Igen: routing, halting, tesztek (4). |
| **1. fázis — E00–E09** | bájtszintű, 0,3–2,5M paraméter | Eléri a Dense/MoE szintet kevesebb számítással? Kontrollált összevetés, 3 seed, auditált. | A router **tanul** (19,9%-kal kisebb veszteség a véletlen routingnál). A hoponkénti attention a legnagyobb minőségi tényező. Azonos aktív számításnál az AM-LLM 6,8%-kal rosszabb a Dense-nél és 14,5%-kal a MoE-nál, és lassabb a GPU-n. Az előre rögzített go/no-go kapu **nem teljesült**. |
| **2. fázis — F1–F4** | bájtszintű | A papíron lévő megtakarításból lehet valódi? | A gradienst kapó Top-1 routing működik; a vödrös kihagyás és a kernelmunka (Triton, CUDA Graph, KV-cache dekódolás) részleges eredményt adott, általános dekódolási nyereség nincs. |
| **F5** (feltáró) | ~17M, 8 192 BPE | Megmaradnak a hatások nagyobb modellen és valódi kódon? | A minőségi kapu **nem teljesült**; a tanult halting sokat veszít a minőségből, ha következtetéskor csonkolunk. |
| **Éjszakai kör** (2026-10-22/23) | ~17M | 12 konfiguráció × 3 seed | A legjobb FixedDepth változatok a MoE veszteségét **24–31%-kal több** becsült FLOP mellett közelítik; nincs olyan változat, amely kevesebb számítással érné el a MoE-t. |
| **V3 kör — W0, W1** (lezárva 2026-10-23) | ~17M | Miért bukik a halting? A család melyik része ront: az expert-kötés, a hierarchikus routing, a közös normalizálás, a router tanítási szabálya? | **W1, 48 előregisztrált futás:** a hopok közötti expert-kötés és a hierarchikus routing egyaránt ront (külön bankok −0,020…−0,036 nat, lapos router −0,033 nat a FixedDepth-hez képest); a közös normalizálásnál és a tanítási szabálynál nem mutatható ki hatás (INCONCLUSIVE, nem egyenértékűség). Az anytime (több kilépéses) veszteség rendbe hozza a kemény halting-utat (−0,70…−0,88 nat), de ≥ 1,18× marad a MoE veszteségéhez képest. [W1 riport](results/round_v3_w1/W1_REPORT.hu.md) |
| **V3 kör — W2** (lezárva 2026-10-24) | ~17M | Attention-megosztás, FLOP-illesztett mélység/Top-K/granularitás pontok illesztett MoE/Dense kontrollal, adaptív K | **66 előregisztrált futás:** a hoponként külön attention számít (a megosztás 0,11–0,17 nat-ba kerül). **0,85× routolt FLOP-on az AM-LLM 0,044–0,054 nat-tal kikap az azonos paraméterű MoE-tól.** A W1-kombináció (3 bank + lapos routing, *FDC1*) az egyetlen modell a MoE alatt (2,087 vs 2,118), de 1,83× paraméterrel és 1,38× végrehajtott FLOP-pal; a paraméterben illesztett MoE-kontroll a következő próba. Az adaptív K jobb a fix K = 1-nél; a halting-ág ≥ 1,19× MoE marad. [W2 riport](results/round_v3_w2/W2_REPORT.hu.md) |

A részletes számok, a konfidenciaintervallumok és a korlátok a [kutatási jelentésben](results/RESEARCH_REPORT.hu.md) vannak; az egyes lépések mögötti döntések a [műszaki rekordokban](docs/04-decisions/README.hu.md).

## Mit tanultunk eddig

1. **A routing tanulható és használt.** A kis modelleknél a tanult routing ~20%-kal kisebb veszteséget ér el, mint a véletlen routing azonos paraméterszám és számítás mellett.
2. **A hoponkénti attention számít a legtöbbet** a kis modelleknél; ez nem kapacitás-műtermék. A W2 ~17M-en is megerősítette: az attention megosztása a hopok között 0,11–0,17 nat-ba kerül, hoponkénti adapterrel is.
3. **~17M-en nincs olyan változat, amely kevesebb végrehajtott FLOP-pal éri el a MoE minőségét.** A hat hopos FixedDepth szerkezetileg *expert-kötött MoE*; a kötés, a kétszintű routing, a közös normalizálás és a router tanítási szabálya (`straight_through` vs `topk_softmax`) együtt változik; a W1 kör szétválasztotta őket: a kötés és a hierarchikus routing ront, a normalizálásnál és a tanítási szabálynál nem mutatható ki hatás (n = 3 mellett INCONCLUSIVE). A legjobb FixedDepth-változatok (2,122–2,125) elérik a dense kontrollt, a MoE-t (2,119) nem.
4. **A jelenlegi tanult halting nem ad használható kompromisszumot:** lágyan tanítjuk (minden hop lefut), keményen alkalmazzuk (a tokenek megállnak), és ez az eltérés sokba kerül. Ez e *mechanizmus* ellen szól, nem az adaptív számítás általában. Az anytime (több kilépéses) veszteség nagyrészt rendbe hozza a kemény utat (W1), de a végső minőségből veszít, és alig takarít meg számítást.
5. **A valós sebességet alig mértük:** egy korai, batch 1-es dekódolási mérésben az AM-LLM 632, a MoE 884 token/s volt.

## Amit *nem* állítunk

Nincs nagy modellre vonatkozó eredmény, VRAM-megtakarítás, ≥ 60 tok/s, és nincs fölény a MoE-val vagy a Dense-szel szemben. A régi teszt-splitet egy automatikus kísérlettervező látta, ezért nem érintetlen holdout. Három seeddel a konfidenciaintervallumok szélesek: a nullát tartalmazó intervallum **nem** igazol ekvivalenciát. A becsült FLOP nem mért GPU-munka. A bájtszintű és a BPE veszteség nem összehasonlítható.

## Miben segíthetsz

Kontrollált ablációk, hard-tudatos halting, illesztett kontrollok, profilozás, replikáció és negatív eredmények egyaránt jöhetnek: [nyitott kutatási kérdések](docs/OPEN_QUESTIONS.hu.md) · [közreműködés](CONTRIBUTING.hu.md) · [Issues](https://github.com/sootylunatic/amllm-research/issues).

## A repó szerkezete

```text
src/amllm/   modell, routing, adat, tanítás, kísérletek, profilozás, gyors inferenciaút, generálás, kiértékelés
tests/       pytest-csomag (CPU; GPU megléte esetén további CUDA-tesztek is futnak)
docs/        célok, architektúra-útmutató, módszertan, műszaki döntési rekordok (angolul és magyarul)
results/     kutatási jelentés, E00–E09 összesítő tábla
data/        csak egy apró mintafájl; a korpuszok helyben épülnek újra (data/README.hu.md)
```

## Gyors indítás (smoke teszt, nem a jelentett futások reprodukciója)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[dev,analysis]'
.venv/bin/python -m pytest -q
.venv/bin/python -m amllm.train --data data/toy_typescript.txt --steps 40 --batch 4 --max-seq-len 64 --run-dir runs/smoke
```

A teljes korpusz, a modellsúlyok és a futásonkénti naplók **nem nyilvánosak**, ezért a publikus forrás önmagában nem reprodukálja a jelentett futásokat; a [jelentés](results/RESEARCH_REPORT.hu.md) auditált másodlagos összefoglaló. MIT licenc, lásd: [LICENSE](LICENSE). A Claude és a Codex segített a megvalósításban és az átvizsgálásban; minden megállapítás nyitott a független ellenőrzésre.
