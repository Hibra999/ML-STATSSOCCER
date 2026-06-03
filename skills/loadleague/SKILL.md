---
name: loadleague
description: Carga, lista o muestra una liga guardada y su dataset.
aliases:
  - openleague
  - showleague
  - listleague
when_to_use:
  - cuando el usuario quiera cargar abrir mostrar listar ligas guardadas
  - cuando mencione load league open league show league dataset preview rows
arguments:
  - league_id
examples:
  - "/loadleague epl-2018"
  - "/league list"
  - "/league show epl-2018 --rows 20"
  - "/data show epl-2018 --rows 25 --hide-missing"
allowed_tools:
  - cli
  - read
user_invocable: true
disable_model_invocation: false
---

# Load League Skill

Usa esta skill para revisar ligas ya guardadas y cargar su dataset.

Comandos comunes:

- Listar ligas guardadas: `league list`
- Mostrar resumen: `league show <league_id> --rows 20`
- Ver dataset: `data show <league_id> --rows 25`
- Ver dataset sin filas incompletas: `data show <league_id> --rows 25 --hide-missing`
- Actualizar liga antes de mostrarla: `league show <league_id> --update --rows 20`

Validacion:

- Si no recuerdas el `league_id`, usa `league list`.
- Si no aparece ninguna liga guardada, usa `/downloadleague` para crear una.
