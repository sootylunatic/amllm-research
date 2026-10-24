---
title: "W3 eredmények — szerkezeti-e az FDC1 előnye, megmarad-e, és mennyibe kerülnek a modellek végrehajtva"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "RESULT"
source_of_truth: true
supersedes: []
superseded_by: []
---

[English](W3_REPORT.md) | **Magyar**

# W3 eredmények — szerkezeti-e az FDC1 előnye, megmarad-e, és mennyibe kerülnek a modellek végrehajtva

**Állapot:** a W3 végleges eredménye. A [lezárt előregisztráció](PREREGISTRATION.hu.md) mind a **15 tanítási futása** a kért lépésszámmal és véges eredménnyel lefutott, egyik sem hibázott. A tanítási kód a W2-höz képest változatlan (`train_code_sha256` = `cd909ea282660f0f`), ezért a W2-futásokkal (B_moe, B_fixed, FDC1, AD02a-3, AD03b, B_dense) közvetlen az összevetés. Minden futáskönyvtár (checkpointtal együtt) a privát mentőszerveren van, futásonként SHA-256-tal ellenőrizve; a riportok, logok és elemzések ebben a könyvtárban. Teszt-split és holdout nincs használva. Gépek: öt RunPod Community RTX 3090; a `w3a` csak mért (nem tanított), a `w3b`–`w3e` tanított.
**Szabály:** az előregisztrált — Δ = változat − referencia (validációs NLL, nat/token, 3 páros seed); `SUPPORTED` = Δ ≤ −0,010, a 95%-os t-CI 0 alatt, 3/3 seed; `NOT_SUPPORTED_IN_SCOPE` = tükörképe; minden más `INCONCLUSIVE` (nem bizonyítja a hatás hiányát). „Azonos FLOP” = routolt FLOP. Döntési tábla: `analysis_decision_table.txt` (a mentett riportokból a `tools/analysis/w3_table.py`-jal).

## 1. Az elsődleges kérdés: FDC1 az azonos méretű MoE ellen

| Konfig | val NLL | seed 42 / 43 / 44 | paraméter | routolt MFLOP | végrehajtott MFLOP (bmm) |
|---|---|---|---|---|---|
| B_moe (W2) | 2,1180 | 2,1120 / 2,1242 / 2,1177 | 17,15M | 22,83 | 24,87 |
| **MoE12** (`moe_f5 n_experts=12`) | **2,0680** | 2,0766 / 2,0572 / 2,0701 | 31,37M | 22,87 | 28,24 |
| **FDC1** (W2; 3 bank + lapos routing) | 2,0870 | 2,0886 / 2,0842 / 2,0881 | 31,36M | 22,93 | 34,23 |

**FDC1 − MoE12 = +0,0190 nat**, 95% CI [+0,0003; +0,0377], mindhárom seed pozitív (+0,012 / +0,027 / +0,018) → **`NOT_SUPPORTED_IN_SCOPE`: az FDC1 rosszabb az azonos paraméterszámú, azonos routolt FLOP-ú MoE-nál.** A W2-es előnye a 17M-es MoE-val szemben a paramétereiből jött, nem a szerkezetéből. Maga a MoE ugyanattól a paraméternöveléstől 0,050 nat-ot nyer (MoE12 − B_moe, leíró).

**A W3b-feltétel** (a későbbi ~300M-es kísérlethez előre rögzítve: átlagos FDC1 − MoE12 ≤ +0,010): +0,019 → 9000 lépésnél **nem teljesül** (de lásd a 3. részt).

## 2. Egyes tényezők (2×2: bankszám × routing)

| Pár | Mi változik | Δ | 95% CI | Ítélet |
|---|---|---|---|---|
| FDC1 − B3H | lapos a hierarchikus routing helyett, 3 bank (azonos paraméter) | −0,0462 | [−0,0682; −0,0242] | **SUPPORTED** (a lapos jobb) |
| B1F − B_fixed | lapos a hierarchikus routing helyett, 1 bank (azonos paraméter) | −0,0336 | [−0,0458; −0,0215] | **SUPPORTED** (a lapos jobb) |
| B3H − B_fixed | 3 bank 1 helyett, hierarchikus (1,83× paraméter) | −0,0267 | [−0,0475; −0,0059] | **SUPPORTED** (3 bank jobb) |
| FDC1 − B1F | 3 bank 1 helyett, lapos (1,83× paraméter) | −0,0393 | [−0,0546; −0,0240] | **SUPPORTED** (3 bank jobb) |
| Kölcsönhatás (FDC1 − B3H) − (B1F − B_fixed) | összeadódik-e a két nyereség? | −0,0126 | [−0,0445; +0,0193] | leíró; a CI tartalmazza a 0-t |

B3H 2,1332 (31,37M), B1F 2,1263 (17,15M). Mindkét W1-es eredmény reprodukálódik ezzel a kóddal: azonos paraméter mellett a lapos routing jobb a hierarchikusnál (0,034–0,046 nat-tal), és a külön bankok segítenek (több paraméterrel). Az adatok összeférnek azzal, hogy a két nyereség összeadódik (a kölcsönhatás CI-je tartalmazza a 0-t; az nincs kimutatva, hogy nulla). Egyik AM-változat sem éri el az azonos méretű MoE-t.

## 3. Hosszabb tanítás (18 000 lépés)

| Konfig | val NLL | seedek | Δ a saját 9k-s futásához |
|---|---|---|---|
| FDC1_18k | 2,0510 | 2,0521 / 2,0600 / 2,0410 | −0,0359 (SUPPORTED) |
| MoE12_18k | 2,0499 | 2,0482 / 2,0529 / 2,0485 | −0,0181 (INCONCLUSIVE) |

**FDC1_18k − MoE12_18k = +0,0012**, 95% CI [−0,0179; +0,0203] → **`INCONCLUSIVE`**: 18k lépés után a különbség már nem mutatható ki (9k-nál +0,019 volt), de egyenértékűség sincs igazolva. Validációs görbék (a seedek átlaga):

| lépés | 2k | 6k | 9k | 12k | 15k | 16k | 17k |
|---|---|---|---|---|---|---|---|
| FDC1_18k | 2,854 | 2,194 | 2,107 | 2,066 | 2,052 | 2,052 | 2,052 |
| MoE12_18k | 2,831 | 2,174 | 2,089 | 2,059 | 2,048 | 2,050 | 2,050 |

Mindkét görbe 15k lépés körül ellaposodik, a MoE12-é utána kicsit emelkedik — 53M tanító tokennel 18k lépés ~5,6-szor látja az adatot, így a görbe vége már az ismétlődő adat tartománya. Az FDC1 9k és 18k között többet javult, mint a MoE12 (−0,036 vs −0,018); hogy ez az FDC1 valódi késői előnye, vagy csak a MoE telítődik előbb az ismétlődő adaton, ezen a korpuszon nem választható szét.

## 4. Mérés (egy RTX 3090, `w3a`, bf16, Triton expert-dispatch; 21 checkpoint)

3 seed átlaga; `analysis_measurement_table.txt`, nyers: `measure/w3a_rtx3090.json`.

| Konfig | végrehajtott MFLOP padding nélkül / bmm | decode tok/s b1 / b8 / b32 | TTFT b1 | súlyok bf16 | LRU-találat, az expertek 25% / 50%-a cache-ben | mohó hurokarány | mintavételes hurokarány |
|---|---|---|---|---|---|---|---|
| B_moe | 22,83 / 24,93 | 875 / 6 532 / 25 750 | 9,4 ms | 32,7 MB | 0,12 / 0,47 | 0,82 | 0,32 |
| MoE12 | 22,87 / 28,80 | 780 / 6 021 / 23 025 | 9,4 ms | 59,8 MB | 0,29 / 0,60 | 0,82 | 0,17 |
| B_dense | 22,81 / 22,81 | 2 077 / 14 972 / 53 017 | 3,7 ms | 25,9 MB | – | 0,80 | 0,41 |
| B_fixed | 22,86 / 31,82 | 653 / 4 881 / 19 253 | 10,1 ms | 32,7 MB | 0,52 / 0,74 | 0,82 | 0,34 |
| FDC1 | 22,93 / 35,71 | 634 / 4 992 / 19 523 | 10,1 ms | 59,8 MB | 0,56 / 0,76 | 0,77 | 0,24 |
| AD02a-3 | 19,71 / 27,61 | 625 / 4 700 / 18 288 | 10,9 ms | 32,7 MB | 0,58 / 0,80 | 0,54 | 0,06 |
| AD03b | 18,30 / 24,96 | 87 / 661 / 2 612 (eager) | 11,8 ms | 32,7 MB | 0,60 / 0,82 | 0,58 | 0,13 |

- **A-sikerkapu (A1):** az AD02a-3 a B_moe padding nélküli FLOP-jának 0,86-szorosát, az AD03b 0,80-szorosát hajtja végre — a számítási feltétel teljesül —, de a kemény úti NLL-jük 1,18–1,23× MoE: mindkettőnél **0/3 seed, `NOT_SUPPORTED_IN_SCOPE`**. FDC1: a padding nélküli végrehajtott FLOP-ja a MoE12-é 1,002-szerese, de a minőségi pár nem teljesül, így összevont állítás nincs. Padding nélküli úton az FDC1/B_fixed végrehajtott többlete eltűnik (a bmm-többlet, ahogy várható volt, a dispatch tulajdonsága).
- **NLL pozíció szerint** (az ablak utolsó 64 tokene): FDC1 − MoE12 +0,028, FDC1 − B_moe −0,022, mindkettő INCONCLUSIVE.
- **Generálás (val-promptok, 384 új token; leíró):** mohó dekódolásnál minden 17–31M-es modell nagyon gyakran hurokba kerül (77–82%; a halting modellek 54–58%, az okot itt nem elemeztük). Mintavétellel a nagyobb modellek ritkábban hurkolnak (MoE12 0,17, FDC1 0,24, szemben a B_moe 0,32-ével és a Dense 0,41-ével); az FDC1 − MoE12 előjele seedenként eltér. Ez az első közvetlen mérése annak a hurokba kerülésnek, amelyet a tulajdonos nagyobb MoE-finomhangolásoknál tapasztalt; ezen a méreten főleg azt mutatja, hogy a több paraméter csökkenti a mintavételes hurkolást.
- **Hotset (B, leíró):** az expertek negyedét cache-elve az AM-modellek 52–60%-os találatot érnek el (a bankok hopok közötti újrahasználata kisebb munkakészletet ad), a MoE-k 12–29%-ot. Ez szimuláció, nem mért offload.

## 5. A C cél feltárása (nem előregisztrált): mennyire gyorsítható a dekódolás?

`c_decode/w3a_rtx3090_v2.json` (`tools/analysis/c_decode_bench.py`, seed 42, RTX 3090, prompt 128, teacher-forced val szöveg). A profil (`c_decode/profile_w3a_console.txt`) megmutatta, hogy a Triton expert-kernel minden hívásnál a teljes expertsúly-tömböt bf16-ra konvertálta (~150 másoló kernel dekódolási lépésenként). Minden modellre azonosan alkalmazott változatok: `bf16` (a súlyok egyszer konvertálva), `gather` (közvetlen expert-kiolvasás rendezés/számlálás/szórás nélkül, adaptív K-val is gráfba rögzíthető; `amllm.fastdecode`, fp64-egyezőségi teszttel), `compile` (gather + `torch.compile`, CUDA Graph-ként visszajátszva).

| Modell | base b1 tok/s | legjobb b1 (változat) | gyorsulás | b32 legjobb (bf16) | val NLL base → legjobb |
|---|---|---|---|---|---|
| Dense | 1 946 | 3 045 (bf16) | 1,6× | 72 191 | 2,1593 → 2,1600 |
| MoE | 911 | 2 293 (compile) | 2,5× | 32 698 | 2,1554 → 2,1553 |
| MoE12 | 825 | 2 158 (compile) | 2,6× | 31 511 | 2,0947 → 2,0947 |
| B_fixed | 673 | 2 013 (compile) | 3,0× | 27 655 | 2,1900 → 2,1876 |
| FDC1 | 668 | 2 179 (compile) | 3,3× | 27 882 | 2,0948 → 2,0945 |
| AD03b | 88 (eager) | 1 989 (compile) | ~23× | 17 374 (compile) | 2,5219 → 2,5222 |

A minőség megmarad: a validációs NLL legfeljebb 0,003-mal változik, és az időzített út a teacher-forced lépések 96–100%-ában ugyanazt jósolja, mint ugyanaz a változat eagerben (a `compile` kis numerikus eltérést mutat). Batch 1-en az AM–MoE sebességrés az FDC1-nél a MoE 0,73-szorosáról 0,95-szörösére szűkül, és az adaptív K használhatóvá válik (88-ról ~2 000 token/s-ra), mert a gather-út CUDA Graph-ba rögzíthető. Batch 32-n csak a bf16 segít (~30%), a gather/compile lassít, és a MoE marad gyorsabb. Korlátok: feltáró, egy seed, egy GPU-típus, kis modellek, ahol a kernelindítás dominál. A mérés két korábbi futása megmaradt: `partial_v1_*` (leállítva, még a minőségellenőrzés előtt) és `partial_v2_compile_fallback_*` (a fordító néhány modell után csendben eagerre esett vissza); csak a `_v2` érvényes.

## 6. Amit a W3 nem állapít meg
- Nincs A-bizonyíték: az A-kapu mindkét AD-jelöltnél elbukik (minőség), és az FDC1 elveszti az illesztett minőségi párt.
- Egy skála (17–31M), egy korpusz (53M tanító token), n = 3; az INCONCLUSIVE sor nem „nincs hatás”.
- A 18k-s összevetés az adatismétlés tartományában van.
- A C-cél számai feltáróak, nem előregisztrált dekódolási eredmény.

## 7. Következmények (javaslat)
- **Az FDC1 nem az a szerkezeti nyertes, amit a W2 sugallt.** Azonos méretnél a sima MoE 9k lépésnél jobb; 18k-nál a kettő nem különböztethető meg. A ~300M-es FDC1-kísérlet előre rögzített feltétele nem teljesül; a tulajdonos dönti el, változtat-e ezen a 18k-s eredmény (az adat egyébként is kell hozzá).
- **Ami a V3-ból megáll:** a lapos routing jobb a hierarchikusnál; a külön (hoponkénti vagy részben megosztott) bankok segítenek; a hoponkénti attention elengedhetetlen; az anytime-veszteség rendbe hozza a kemény halting-utat; az adaptív K számítást takarít meg, a teljes K-hoz mért minőségi különbsége INCONCLUSIVE volt (W2; nincs kimutatva, hogy nulla). Egyik sem viszi a MoE fölé az AM-változatot azonos méreten ezen a skálán.
- **C cél:** a dekódolási sebességrés nagy része végrehajtási probléma volt; a gather-úttal az AM-modellek batch 1-en a MoE sebességének ~95%-ával dekódolnak, és az adaptív mélység gyakorlatban is használhatóvá válik. Egy előregisztrált dekódolási mérés (több seed, második GPU-típus) tenné ezt bizonyítékká.

## Fájlok
- `*/<preset>_s4?/`: futásonként `config.json`, `env.json`, `report.json`, `train.log` (checkpoint csak a mentőszerveren).
- `analysis_decision_table.txt`, `analysis_measurement_table.txt`; `measure/w3a_rtx3090.json`; `c_decode/` (dekódolási benchmark, profil, manifest).
- `queues/`, `_queue_w3*/`: lefutott sorok és worker-logok.
