---
name: train
description: Entrena modelos ML para una liga usando el CLI existente.
when_to_use:
  - cuando el usuario quiera entrenar un modelo
  - cuando mencione random forest xgboost svm dnn optuna cross validation calibracion sampler normalizer
arguments:
  - league_id
  - model_type
  - model_id
examples:
  - "/skill train epl-2018 random-forest"
  - "/model train epl-2018 random-forest --id rf-result"
  - "/model train epl-2018 xgboost --id xgb-uo --target over-under --tune all --trials 25"
allowed_tools:
  - cli
  - read
  - grep
user_invocable: true
disable_model_invocation: false
---

# Train Skill

Usa esta skill para preparar y ejecutar entrenamiento de modelos.

Modelos soportados:

`logistic`, `discriminant`, `decision-tree`, `random-forest`, `xgboost`, `knn`, `naive-bayes`, `svm`, `dnn`.

Comandos relacionados:

- Listar modelos existentes: `model list <league_id>`
- Entrenar rapido: `model train <league_id> random-forest --id <model_id>`
- Entrenar con target: `model train <league_id> xgboost --id <model_id> --target over-under`
- Entrenar con tuning: `model train <league_id> xgboost --id <model_id> --tune all --trials 50 --objective F1`
- Exportar metricas: `model train <league_id> random-forest --id <model_id> --export-metrics outputs/metrics`

Validacion:

- Comprueba que la liga existe: `league show <league_id>`.
- Evita reutilizar un `model_id` ya guardado.
- El entrenamiento puede tardar; para VPS empieza con pocos `--trials`.
