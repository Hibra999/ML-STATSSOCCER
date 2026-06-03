---
name: evaluate
description: Evalua modelos entrenados y administra filtros de odds y percentiles.
aliases:
  - evaluatemodel
  - metrics
when_to_use:
  - cuando el usuario quiera evaluar rendimiento precision recall f1 accuracy o balance
  - cuando mencione odds filtros percentiles store-filter delete-filter dataset train eval
arguments:
  - league_id
  - model_id
examples:
  - "/skill evaluate epl-2018 rf-result"
  - "/model evaluate epl-2018 --model rf-result --dataset eval"
  - "/model evaluate epl-2018 --model rf-result --odd-filter \"1:1.31:1.60\" --p1 70 --px 60 --p2 70 --store-filter"
allowed_tools:
  - cli
  - read
user_invocable: true
disable_model_invocation: false
---

# Evaluate Skill

Usa esta skill para medir un modelo y, si aplica, guardar filtros para predicciones de fixtures.

Comandos relacionados:

- Evaluar todo: `model evaluate <league_id> --model <model_id> --dataset all`
- Evaluar holdout: `model evaluate <league_id> --model <model_id> --dataset eval`
- Aplicar odds: `model evaluate <league_id> --model <model_id> --odd-filter "1:1.31:1.60"`
- Guardar filtro: `model evaluate <league_id> --model <model_id> --dataset eval --p1 70 --px 60 --p2 70 --store-filter`
- Borrar filtro: `model evaluate <league_id> --model <model_id> --odd-filter "1:1.31:1.60" --delete-filter`
- Ver metricas guardadas: `model metrics <league_id> <model_id>`

Validacion:

- `--delete-filter` modifica configuracion del modelo y requiere cuidado.
- Si el target es over-under usa `--pu` y `--po`; para result usa `--p1`, `--px`, `--p2`.
