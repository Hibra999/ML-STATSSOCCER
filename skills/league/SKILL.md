---
name: league
description: Gestiona ligas guardadas y catalogo de ligas descargables.
aliases:
  - leagues
  - manageleague
when_to_use:
  - cuando el usuario quiera crear listar actualizar mostrar o eliminar ligas
  - cuando mencione catalogo pais liga temporada dataset historico o league_id
arguments:
  - action
  - league_id
examples:
  - "/league list --catalog"
  - "/league show epl-2018"
  - "/skill league list"
allowed_tools:
  - cli
  - read
  - grep
user_invocable: true
disable_model_invocation: false
---

# League Skill

Usa esta skill para administrar ligas desde el CLI existente.

Acciones comunes:

- Listar catalogo: `league list --catalog`
- Listar ligas guardadas: `league list`
- Mostrar una liga: `league show <league_id> --rows 20`
- Crear una liga: `league create --league-index <n> --id <league_id> --start-year <year> --yes`
- Actualizar una liga: `league update <league_id>`
- Eliminar una liga: `league delete <league_id>`

Validacion:

- `league_id` debe usar letras, numeros, puntos, guiones o guiones bajos.
- Antes de entrenar o predecir, confirma que la liga existe con `league show <league_id>`.
- `league delete` es destructivo y requiere confirmacion explicita.
