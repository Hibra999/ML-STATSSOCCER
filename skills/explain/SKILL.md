---
name: explain
description: Genera graficos de interpretabilidad para modelos entrenados.
when_to_use:
  - cuando el usuario quiera explicar un modelo shap pdp waterfall boundary decision boundary
  - cuando mencione coeficientes arbol impurity attention interpretabilidad
arguments:
  - league_id
  - model_id
  - plot_type
examples:
  - "/skill explain epl-2018 rf-result shap"
  - "/explain shap epl-2018 rf-result --target H --output outputs/shap.png"
  - "/explain extra epl-2018 rf-result --plot impurity --output outputs/impurity.png"
allowed_tools:
  - cli
  - read
  - list
user_invocable: true
disable_model_invocation: false
---

# Explain Skill

Usa esta skill para explicar modelos ya entrenados.

Comandos relacionados:

- Boundary: `explain boundary <league_id> <model_id> --features "1,HW%" --output outputs/boundary.png`
- PDP: `explain pdp <league_id> <model_id> --feature "1" --target H --output outputs/pdp.png`
- Waterfall SHAP: `explain waterfall <league_id> <model_id> --match-index 0 --target H --output outputs/waterfall.png`
- SHAP bar: `explain shap <league_id> <model_id> --target H --output outputs/shap.png`
- Extra: `explain extra <league_id> <model_id> --plot impurity --output outputs/impurity.png`

Validacion:

- `--target` depende del modelo: H/D/A para result, U/O para over-under.
- SHAP puede tardar mas y consumir memoria.
- Algunas graficas extra solo existen para ciertos tipos de modelo.
