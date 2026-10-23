[English](0007-corpus-and-split.md) | **Magyar**

# ADR-0007: Nyílt forrású, csomagonként szétválasztott kódkorpusz; saját kód kizárva

- **Státusz:** elfogadott
- **Dátum:** 2026-10-21
- **Érintett kód:** `amllm/data.py`, `amllm/synth.py`, `data/README.md`

## Kontextus
A v0.1 egy 986 bájtos fájlon tanult, train/validation szétválasztás nélkül — ez csak memorizálást mér.
Valódi kódra van szükség, de a saját projektek kódja titkokat és személyes adatot tartalmazhat.

## Döntés
- Forrás: harmadik féltől származó nyílt forrású npm-csomagok `.ts/.tsx` fájljai (nem `.d.ts`) egy helyi
  `node_modules` mappából. Saját repók **nem** kerülhetnek be.
- Deduplikáció tartalom szerinti SHA-256-tal; minifikált fájlok kiszűrése.
- **Csomagonkénti** split a csomagnév hash-e alapján (train/val/test ≈ 80/10/10%), hogy egy könyvtár fájljai ne
  szivárogjanak a kiértékelésbe. A split-azonosító (SHA-256) minden futás `config.json`-jában rögzítve.
- Külön szintetikus készlet (ismétlés / másolás / összeadás) bájtonkénti feladatcímkével; a val/test sorok nem
  szerepelnek a trainben (az ismétlés kivétel: csak 234 változat létezik).
- A korpusz és a manifest nem publikált (helyi útvonalak, eltérő licencek); a generálás determinisztikus.

## Következmények
- A val/test adat sosem került tanításba; a test splitet futásonként egyszer értékeltük.
- A korpusz kicsi (13,6 MB train); a 6000 lépéses tanítás ~3,6 epoch.
