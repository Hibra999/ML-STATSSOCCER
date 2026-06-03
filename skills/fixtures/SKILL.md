---
name: fixtures
description: Predice fixtures futuros desde archivo local o scraping headless.
when_to_use:
  - cuando el usuario quiera predecir fixtures calendario proximos partidos o FootyStats
  - cuando mencione input csv xlsx scraping headless filters selected
arguments:
  - league_id
  - model_id
  - input_or_date
examples:
  - "/skill fixtures epl-2018 rf-result fixtures.csv"
  - "/predict fixtures epl-2018 --model rf-result --input fixtures.csv --filters all --output exports/fixtures.csv"
  - "/predict fixtures epl-2018 --model rf-result --date 2026-08-15 --headless --filters all"
allowed_tools:
  - cli
  - read
  - list
user_invocable: true
disable_model_invocation: false
---

# Fixtures Skill

Usa esta skill para predicciones masivas de fixtures.

Desde archivo local:

`predict fixtures <league_id> --model <model_id> --input fixtures.csv --filters all --output exports/fixtures.csv`

Con scraping headless:

`predict fixtures <league_id> --model <model_id> --date YYYY-MM-DD --headless --filters all`

Formato de archivo requerido:

`Home,Away,1,X,2`

Validacion:

- Si no pasas `--input`, debes pasar `--date`.
- En VPS usa `--headless` y confirma que `storage/network/browser.json` apunta a un navegador instalado.
- `--filters all` usa filtros guardados en evaluacion; si no hay filtros, el CLI avisa y usa filtros de liga.
