[English](README.md) | **Magyar**

# Technikai döntési rekordok (ADR)

Sablon: [0000-template.hu.md](0000-template.hu.md). Elfogadott rekordot nem írunk át; a változást új rekord rögzíti, a ténybeli
hibát dátumozott *Javítás* bejegyzés. Csak a technikai rekordok nyilvánosak; a hiányzó sorszámok belső projektrekordok. Az ezekre, a `NEXT_EXPERIMENTS` pontjaira és az előregisztrációs fájlokra tett említések belső dokumentumokra utalnak.
A rekordokban idézett mért értékek **előzetesek (audit alatt)**.

| # | Döntés | Státusz |
|---|---|---|
| [0002](0002-byte-level-tokenization.hu.md) | Byte-szintű tokenizálás az első kísérletsorozatban | elfogadott |
| [0003](0003-hierarchical-module-bank.hu.md) | Megosztott modulbank, hierarchikus Top-K routing | elfogadott |
| [0004](0004-survival-halting.hu.md) | Kumulatív survival-alapú halting a lépésenkénti küszöb helyett | elfogadott |
| [0005](0005-attention-in-every-hop.hu.md) | Megosztott attention minden hopban; hoponként saját attention jelöltként | elfogadott / javasolt |
| [0006](0006-expert-dispatch.hu.md) | Egymásra rakott expertek, `loop` és `bmm` dispatch | elfogadott, kiegészítve: 0014 |
| [0007](0007-corpus-and-split.hu.md) | Nyílt forrású, csomagonként szétválasztott korpusz; saját kód kizárva | elfogadott |
| [0008](0008-control-models.hu.md) | Kontrollok azonos aktív FLOP-ra és teljes paraméterre illesztve | elfogadott |
| [0009](0009-measurement-protocol.hu.md) | Mérési protokoll és előre rögzített döntési szabályok | elfogadott, kiegészítve: 0014 |
| [0013](0013-top1-router-gradient.hu.md) | A Top-1 router nem kap LM-gradienst; teljes-softmax súlyozás | elfogadott |
| [0014](0014-deterministic-combine-and-diagnostics.hu.md) | Determinisztikus összegzés, szinkronmentes statisztika, diagnosztika nélküli mérés | elfogadott |
