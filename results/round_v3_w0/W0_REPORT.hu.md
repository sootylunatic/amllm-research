---
title: "W0 (CPU) első eredmények — AD01 trace, K1 görbeelemzés, K2 route-trace"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "RESULT"
source_of_truth: true
supersedes: []
superseded_by: []
---

[English](W0_REPORT.md) | **Magyar**

# W0 (CPU) első eredmények — AD01 trace, K1 görbeelemzés, K2 route-trace

**Platform:** helyi Mac (arm64, M4), torch 2.14.1, fp32, CPU. Adat: a `code_corpus_v2` **val** splitje, 128 egyenletesen elosztott ablak × 512 token (65 536 token; a teljes kiértékelés 512 ablakot használ, ezért az abszolút értékek ≈ 0,01 nat-tal eltérhetnek a jelentett val-értékektől). Teszt-adat nincs használva. A fő táblák **seed 42**-ek; az AD01 három seeden is megismételve (alább): ezek diagnosztikák, nem megerősítő eredmények. Eszközök: `tools/analysis/ad01_trace.py`, `k1_curves.py`, `k2_route_trace.py`; nyers kimenet: `AD01/`, `K1/`, `K2/`. Ellenőrzött checkpointok (SHA-256 a vps2-vel egyezik).

## AD01 — hol romlik a minőség a halting modelleknél

**Helyesség:** a nyomkövető a modell tényleges hop-ciklusát futtatja újra, és a végső logitokat összeveti a `model.forward` kimenetével soft és hard módban: **a legnagyobb eltérés 0,0 mind az 5 checkpointon**. A kimenetet tehát nem rejtett eltérés okozza; a soft/hard rés a módszer sajátja, nem implementációs hiba.

**Hoponkénti kiolvasás** (az adott hop utáni állapot a végső normán és fejen át, nat/token):

| Checkpoint | hop 1 | hop 2 | hop 3 | hop 4 | hop 5 | hop 6 (végső) | oracle: legjobb hop tokenenként\* |
|---|---|---|---|---|---|---|---|
| `H01_fixed` (halting nélkül) | 6,24 | 5,78 | 5,20 | 4,20 | 3,57 | **2,14** | 1,86 |
| F5-alap (λ 0,02) | 6,04 | 4,60 | 4,00 | 3,34 | 2,45 | **2,17** | 1,85 |
| `H01_bias_m2` | 6,09 | 4,59 | 3,95 | 3,24 | 2,45 | **2,18** | 1,86 |
| `H01_p005` | 6,32 | 5,65 | 4,72 | 4,29 | 2,71 | **2,15** | 1,86 |
| `H01_p05` | 6,23 | 4,48 | 3,43 | 2,63 | 2,36 | **2,23** | 1,91 |

\* Az oracle tokenenként a legkisebb NLL-ű hopot választja a célbájt ismeretében: **analitikai alsó korlát, nem alkalmazható policy**, és a minimumképzés önmagában optimista.

**Kemény kilépés** (küszöb → val NLL @ átlagos hop; „véletlen” = ugyanaz a hop-eloszlás véletlen tokenekhez rendelve a soft nyomból):

| Checkpoint | θ 0,1 | θ 0,3 | θ 0,5 | θ 0,8 |
|---|---|---|---|---|
| F5-alap | 2,174 @ 5,91 (véletlen 2,21) | 2,291 @ 5,16 (2,59) | 3,383 @ 3,47 (3,68) | 6,04 @ 1,0 |
| `H01_bias_m2` | 2,176 @ 5,91 | 2,290 @ 5,15 | 3,274 @ 3,58 (3,57) | 6,09 @ 1,0 |
| `H01_p005` | 2,153 @ 6,00 | 2,155 @ 5,98 | 2,177 @ 5,90 (2,22) | 3,08 @ 5,06 (2,93) |
| `H01_p05` | 2,255 @ 5,30 (2,37) | 3,487 @ 2,81 (3,76) | 6,21 @ 1,01 | 6,23 @ 1,0 |

**Mit mutat ez (a mért pontokra, 1 seed):**
1. **A köztes állapotok nem használhatók közvetlen kiolvasásra,** halting nélkül sem: a fix mélységű modell a 4. hopnál 4,20, az 5.-nél 3,57 NLL-t ad, a végén ugrik 2,14-re. A modell hopjai nincsenek arra tanítva, hogy korai kimenetet adjanak. Ezért a küszöb csak akkor tud megállítani anélkül, hogy a veszteség megugrana, ha a modell eleve a végén kapja a minőséget — ez a mért kudarc oka. A következtetés az AD05 döntésfa „köztes hop gyenge” ágát támasztja alá, és az **AD02 első tagját (anytime/multi-exit veszteség)** indokolja.
2. **A tanult stop-policy a mért pontok többségén jobb a véletlen kilépésnél** (0,03–0,3 nat; kivétel `H01_p005` θ 0,8: 3,08 a véletlen 2,93 ellen), tehát hordoz információt, de ez nem elég a veszteség kordában tartásához. A „véletlen” kontroll a soft nyomból olvas, nem a kemény út állapotaiból, ezért közelítő.
3. **Kevesebb hop csak a minőség rovására:** a soft értékhez közeli kemény érték (θ 0,1) ≈ 5,9 hoppal jön, vagyis alig van megtakarítás. A `p05` jobb köztes állapotokat tanul (hop 4: 2,63), de a küszöb ott is összeesik.
4. **Az oracle-korlát (≈ 1,85–1,91) a végső 2,14–2,23 alatt van:** tokenek egy részénél egy korábbi hop jobb, mint a végső. Ez csak hipotézis a tartalék-kapacitásra; policyként nem használható és nem bizonyít semmit a deployolható nyereségről.

**Három seeden megismételve** (42/43/44, az éjszakai 12 konfiguráció négy AM-LLM sora; átlag, zárójelben a seedek közti terjedelem; a nyomkövető–`forward` eltérés mindenhol 0,0):

| Konfiguráció | hop 1 → 5 kiolvasás (NLL) | végső | oracle | θ 0,1 | θ 0,3 | θ 0,5 |
|---|---|---|---|---|---|---|
| `H01_fixed` | 6,29 / 5,78 / 5,21 / 4,15 / 3,43 | 2,149 | 1,87 | – | – | – |
| `H01_p005` | 6,31 / 5,66 / 4,63 / 4,03 / 2,99 | 2,144 | 1,85 | 2,144 @ 6,00 | 2,146 @ 5,98 | 2,171 @ 5,91 (véletlen 2,22) |
| `H01_bias_m2` | 6,19 / 4,66 / 3,96 / 3,06 / 2,54 | 2,165 | 1,86 | 2,165 @ 5,88 | 2,268 @ 5,14 | 3,222 @ 3,66 (3,51) |
| `H01_p05` | 6,40 / 4,45 / 3,36 / 2,77 / 2,40 | 2,255 | 1,94 | 2,277 @ 5,24 | 3,368 @ 2,87 | 6,322 @ 1,06 |

A három seed ugyanazt mutatja, mint az egy seedes sor: a köztes kiolvasások gyengék (a halting nélküli modell 4. hopja 4,15, az 5.-é 3,43), a kemény kilépés csak ≈ 5,9 hoppal marad a soft érték közelében, és a tanult policy a véletlen kilépésnél jobb (θ 0,3–0,5: 0,05–0,3 nat), de a veszteség nem tartható meg kevesebb hoppal.

## AD01 — külön gradiensaudit (autograd-tesztek és mérés)

**Autograd-tesztek** (`tests/test_ad01_gradient_paths.py`, pontos, nem közelítés):
1. **Kemény kilépés:** a kilépett token vesztesége a kilépés utáni hopoknak **pontosan nulla** gradienst ad, vagyis a csonkolt jövőbeli állapot nem kap LM-gradienst. A **küszöb-döntés** nem differenciálható, de a ténylegesen lefutott hopok **túlélési súlya** igen: az egy hopot futó token (súly 1) nem ad gradienst a halt-fejnek, a két hopot futó token a második hop súlyán át **nem nulla** LM-gradienst ad neki (két külön teszt).
2. **Soft (tanítási) út:** a halt-fej kap gradienst a túlélési súlyokon át, és a későbbi hopok minden tokentől kapnak gradienst, a túlélési súllyal skálázva.
3. **Ponder-büntetés:** `d(ponder)/d(bias) < 0`, a gradiens-leszállás a bias-t növeli, vagyis korábbi megállás felé tol, az LM-vesztesége nélkül is.
4. **Kilépési veszteség (AD02a):** a hop *h* utáni állapot vesztesége a *≤ h* hopok paramétereihez, a megosztott bankhoz és a megosztott fejhez jut, a *> h* hopokhoz **pontosan nulla**.

**Mérés valódi checkpointokon** (`tools/analysis/ad01_grad.py`; Mac, train-split, 6 × 4 × 512 token, soft/train mód, seed 42, egy checkpoint konfigurációnként; gradiensnorma az LM-veszteségre és külön az aux-veszteségre (ponder + balance)):

| Konfiguráció | halt-fej LM | halt-fej aux | aux/LM | attention hop 1 → 6 (LM) |
|---|---|---|---|---|
| `H01_p005` (λ 0,005) | 0,24 | 0,02 | 0,09 | 0,23 / 0,48 / 0,36 / 0,35 / 0,30 / 0,25 |
| F5-alap (λ 0,02) | 0,79 | 0,37 | 0,46 | 0,35 / 0,43 / 0,29 / 0,22 / 0,25 / 0,18 |
| `H01_bias_m2` (λ 0,02) | 0,76 | 0,37 | 0,48 | 0,38 / 0,46 / 0,29 / 0,23 / 0,24 / 0,17 |
| `H01_p05` (λ 0,05) | 1,29 | 0,80 | 0,62 | 0,48 / 0,41 / 0,26 / 0,23 / 0,14 / 0,12 |

- A halt-fejre ható aux-gradiens a λ-val nő, de a végső checkpointokon az LM-gradiens **nagyobb** (aux/LM < 1): a ponder-büntetés nem uralja a végső állapotot. A λ = 0,05 összeomlása tehát nem egy pillanatképen mért aux-túlsúly eredménye, hanem a tanítás alatti dinamikáé (lásd a K1/AD01 görbéket).
- A későbbi hopok attention-gradiense kisebb, a legerősebb ponder mellett a hop 6 gradiense a hop 1-é negyede (0,12 vs 0,48): a túlélési súly a későbbi hopokat kevésbé tanítja, ami az AD02a (köztes hopok közvetlen tanítása) mellett szól.
- Az aux-gradiens a későbbi hopok attentionjén gyakorlatilag nulla (≤ 0,0007 a hop 6-nál): a ponder csak a halt-fejet és a túlélésen át a korábbi hopokat nyomja.
- Korlát: egy seed, a végső checkpoint, 6 mini-batch; tanítás közbeni gradiensnorma-napló nincs (a W1 futások ezt még nem naplózzák, ez a következő hullám feladata).

## K1 — a MoE 18k-s „romlás” oka (meglévő görbék, F5_2x, seed 42)

| Lépés | 14 000 | 15 000 | 16 000 | 17 000 | végső |
|---|---|---|---|---|---|
| MoE val | 2,046 | 2,056 | **2,254** | 2,253 | 2,318 |
| MoE tanítási veszteség (500 lépéses átlag) | 1,27 | 1,26–1,33 | **1,41 → 1,53** | 1,54 | 1,58 |
| Dense tanítási veszteség | 1,33 | 1,32–1,34 | 1,27–1,31 | 1,29 | 1,25 |

- A MoE validációs vesztesége 15 000 lépésig **javul** (2,046), majd 15 000–16 000 között **+0,2 nat-ot ugrik**, és közben a **tanítási veszteség is megemelkedik** (1,26 → 1,53–1,58). Túltanulásnál a tanítási veszteség esne, miközben a validációs emelkedik; itt mindkettő romlik: ez **tanítási instabilitási esemény** (LR ≈ 1,4·10⁻⁴ körül), nem túltanulás.
- A végső értékelésben **nincs routing-összeomlás** (halott szakértő 0; használati entrópia 3,08 / 3,18, a 9k-s futásé 3,13). A kiváltó ok (pl. attention-logit növekedés, bf16, router) ebből nem azonosítható; a Dense ugyanazon az ütemezésen nem esett ki.
- Egy seed, egy futás: ez **hipotézis**, nem megerősített ok. A K1 új futásai (W2) tudják szétválasztani (azonos konfiguráció több seeddel, gradiensnorma- és logitnorma-naplózás, rövidebb ütemezési horizont).

## K2 — mire támaszkodik a router? (H06/FD06 első CPU-mérés)

**Módszer:** `tools/analysis/k2_route_trace.py`; 256 val-ablak (131 072 token), seedenként 42/43/44, `H01_fixed` (FixedDepth AM-LLM) és `B_moe_3090` (MoE). A route-címke a hop/réteg első választású szakértője. A plug-in kölcsönös információból (bit) levonjuk a permutációs alapszintet (5 keverés), így az „excess” a véletlen fölötti rész. A gyakori tokenek (≥ 30 előfordulás) a tokenek 80%-át fedik; a ritkák egy csoportba kerülnek. Platform: Mac arm64, fp32. A három seed átlaga; a seedek közti szórás ≤ 0,14 bit.

| | Route-entrópia (bit) | I(route; aktuális token), többlet | …ebből a route-entrópia hányada | I(route; előző token) hányada | I(route; pozíció) | I(route; tokenforma) |
|---|---|---|---|---|---|---|
| `H01_fixed` (24 szakértő/hop, max 4,58) | 4,19–4,40 | 1,87–2,35 | **45–54%** | 14–28% | ≈ 0,005–0,009 | 0,48–0,81 |
| MoE (4 szakértő/réteg, max 2,00) | 1,82–1,98 | 1,30–1,51 | **69–78%** | 12–27% | ≈ 0,000–0,001 | 0,14–0,61 |

(Az előző token és az aktuális token egymással korrelál, a hányadok nem összegezhetők. A tokenforma: szóköz/újsor, szám, azonosító-szerű, írásjel, egyéb.)

**Mit mutat (1 mérési család, 3 seed, csak a feltett változókra):**
1. **Mindkét modell routingja erősen az aktuális tokenhez kötődik** (a route-entrópia 45–78%-át megmagyarázza a token azonossága), az előző token további, kisebb részt ad; a pozíció gyakorlatilag semmit.
2. **Az AM-LLM routingja kevésbé token-determinisztikus, mint a MoE-é** a route-entrópia hányadában (45–54% a 69–78% ellen), vagyis több a kontextusfüggő vagy más (nem token-azonosságú) része. A két route-címke ábécéje eltérő (24 vs 4), ezért a hányadok közelítő összevetések, nem bizonyított különbség.
3. **Amit nem mér:** feladat-, szintaxis- vagy szemantikai kontextust (ehhez címkézett kategóriák kellenek), a route kauzális szerepét (shuffle/reinit hatás a veszteségre) és a szakértők funkcióját. A H06 ezért **PARTIAL**, az FD06 többi kérdése (shuffle, másolás) hátra van.

## Következmények a tervre
- AD02 első módszere: **anytime/multi-exit veszteség**; a döntési kapuhoz az AD01 mérőszámai (hoponkénti NLL, kemény–soft rés) újrafuttathatók ugyanezzel az eszközzel.
- A K1 eredménye miatt minden hosszabb (18k) futásnál kötelező a **gradiens- és logitnorma-naplózás**, és a MoE-horgony 18k-s futását több seeddel kell megismételni, mielőtt a hosszú tanítást összehasonlításhoz használjuk.
- Hiányzik: az AD01 több seeden (az éjszakai 36 checkpoint már a Macen van), a route kauzális tesztje (shuffle/reinit) és az arm64–x86 numerikus egyezés ellenőrzése (platformfüggetlenség).
