---
name: analysis
description: Ejecuta analisis estadistico y exporta graficos o tablas.
when_to_use:
  - cuando el usuario quiera estadisticas distribuciones varianza correlacion boruta coeficientes impurity reglas
  - cuando mencione feature importance heatmap descriptive analysis
arguments:
  - league_id
  - analysis_type
examples:
  - "/skill analysis epl-2018 variance"
  - "/analysis variance epl-2018 --output outputs/variance.png"
  - "/analysis correlation epl-2018 --method spearman --output outputs/correlation.png"
allowed_tools:
  - cli
  - read
  - list
user_invocable: true
disable_model_invocation: false
---

# Analysis Skill

Usa esta skill para generar salidas estadisticas desde datasets de liga.

Comandos relacionados:

- Descriptivo: `analysis descriptive <league_id> --feature-type home --output outputs/descriptive.png`
- Distribuciones: `analysis distributions <league_id> --column Result --output outputs/result-dist.png`
- Varianza: `analysis variance <league_id> --output outputs/variance.png`
- Correlacion: `analysis correlation <league_id> --method spearman --output outputs/correlation.png`
- Boruta: `analysis boruta <league_id> --target result --output outputs/boruta.png`
- Coeficientes: `analysis coefficients <league_id> --target result --output outputs/coefficients.png`
- Impurity: `analysis impurity <league_id> --target result --output outputs/impurity.png`
- Reglas: `analysis rules <league_id> --target result --depth 4 --output outputs/rules.png`

Validacion:

- Todos los comandos guardan imagenes; define `--output`.
- Para `distributions`, confirma que `--column` existe en el dataset.
