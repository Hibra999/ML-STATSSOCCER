---
name: downloadleague
description: Descarga o crea una liga nueva desde el catalogo disponible.
aliases:
  - createleague
  - newleague
when_to_use:
  - cuando el usuario quiera descargar crear nueva liga desde catalogo pais temporada
  - cuando mencione download league create league catalogo league-index start-year
arguments:
  - league_id
  - catalog_or_index
  - start_year
examples:
  - "/downloadleague"
  - "/league list --catalog"
  - "/league create --league-index 6 --id epl-2018 --start-year 2018 --history-window 3 --goal-margin 2 --stats all --yes"
allowed_tools:
  - cli
  - read
user_invocable: true
disable_model_invocation: false
---

# Download League Skill

Usa esta skill para descargar datos historicos y crear una liga guardada.

Flujo recomendado:

- Ver catalogo: `league list --catalog`
- Crear por indice: `league create --league-index <n> --id <league_id> --start-year <year> --yes`
- Crear por pais/liga: `league create --country England --name Premier-League --id epl-2018 --start-year 2018 --yes`

Ejemplo completo:

`league create --league-index 6 --id epl-2018 --start-year 2018 --history-window 3 --goal-margin 2 --stats all --yes`

Validacion:

- `league_id` debe ser unico; si ya existe usa `league show <league_id>`.
- Si no sabes el indice, ejecuta primero `league list --catalog`.
- `start_year` no puede ser anterior al inicio disponible de la liga.
