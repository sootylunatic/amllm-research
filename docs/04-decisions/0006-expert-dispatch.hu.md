[English](0006-expert-dispatch.md) | **Magyar**

# ADR-0006: Egymásra rakott expert-súlyok, kétféle dispatch (`loop` és `bmm`)

- **Státusz:** elfogadott — kiegészítve: ADR-0014
- **Dátum:** 2026-10-21
- **Érintett kód:** `amllm/routing.py: ExpertBank`, `amllm.model.set_dispatch`

## Kontextus
A v0.1 modulonként és expertenként `nonzero()`-t hívott, ami hoponként sok host–GPU szinkronizációt okoz.
Egyúttal bizonyítani kell, hogy a ritka végrehajtás *valóban* kihagy számítást, nem csak a statisztika mutat kevesebbet.

## Döntés
- Az összes modul összes expertje egy `[G, D, H]` / `[G, H, D]` súlytenzorban (`G = M·E`).
- `loop`: a hozzárendeléseket expert szerint rendezzük; expertenként egy matmul-pár csak a kiválasztott sorokra.
  Hoponként egyetlen host-szinkron (a darabszámok). A kiszámolt sorok száma = a routolt hozzárendelések száma.
- `bmm`: minden expert sorait a legnagyobb csoportra paddingeljük és egyetlen batch-matmult futtatunk
  (grouped-GEMM helyettesítő). Kevés kernel, de a padding és az üres expertek is számolódnak.
- A két út numerikusan azonos (kimenet és gradiens, tesztelve); tanításhoz `bmm`, a valódi kihagyás méréséhez `loop`.

## Következmények
- Mérve (E02): 1 vs. 4 hop kikényszerítve: `loop` 32 768 vs. 131 072 FFN-sor, 2,2 vs. 8,2 ms → a kihagyás valós.
- Mérve (profil): egy AM-LLM forward ~1660 kernelt indít (`loop`); a halting −36% profilozott FLOP-ja csak −3,5%
  falióra-időt ad. `bmm`: 2,1–3× gyorsabb a `loop`-nál, de 2,3–3× több FLOP-ot számol.

**Javítás (2026-10-21, audit):** a jelentés „a falióra ~60%-a CPU-oldali indítási overhead” állítását a külön
profilozott kernelidő és egy másik mérés késleltetésének különbségéből vezette le. Ez a különbség az indítási
overheadet, a host-szinkronizációt, a Pythont és a diagnosztikai munkát együtt tartalmazza, nem bontja fel őket.
Az ADR-0014 diagnosztika nélküli futási utat és szinkronszámlálást ad; a jelentés már csak a mért értékeket közli.
- Valós gyorsuláshoz kernel-szintű munka kell (CUDA Graphs, Triton grouped GEMM — NEXT_EXPERIMENTS N3).
