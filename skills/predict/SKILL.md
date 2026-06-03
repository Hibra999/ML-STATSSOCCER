---
name: predict
description: Genera predicciones manuales para partidos individuales.
when_to_use:
  - cuando el usuario quiera predecir Arsenal Chelsea local visitante cuota odd
  - cuando mencione partido manual home away probabilidades
arguments:
  - league_id
  - model_id
  - home
  - away
examples:
  - "/skill predict epl-2018 rf-result Arsenal Chelsea"
  - "/predict manual epl-2018 --model rf-result --home Arsenal --away Chelsea --odd-1 2.10 --odd-x 3.40 --odd-2 3.10"
allowed_tools:
  - cli
  - read
user_invocable: true
disable_model_invocation: false
---

# Predict Skill

Usa esta skill para prediccion manual de un partido con odds conocidas.

Comando principal:

`predict manual <league_id> --model <model_id> --home <home> --away <away> --odd-1 <odd1> --odd-x <oddx> --odd-2 <odd2>`

Ejemplo:

`predict manual epl-2018 --model rf-result --home Arsenal --away Chelsea --odd-1 2.10 --odd-x 3.40 --odd-2 3.10`

Validacion:

- Los equipos deben existir en la liga historica.
- Home y Away no pueden ser iguales.
- Las odds deben ser mayores que 1.00.
- Usa `--output exports/manual.csv` si necesitas guardar el resultado.
