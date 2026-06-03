---
name: troubleshoot
description: Diagnostica errores de CLI, datos, modelos, scraping y entorno.
aliases:
  - debug
  - fix
when_to_use:
  - cuando el usuario reporte errores excepciones fallos de importacion o problemas de Selenium
  - cuando mencione debug compileall pytest browser headless datos faltantes o modelo no existe
arguments:
  - symptom
examples:
  - "/skill troubleshoot error de scraping"
  - "!git status"
  - "/config browser show"
allowed_tools:
  - cli
  - bash
  - read
  - grep
  - git
user_invocable: true
disable_model_invocation: false
---

# Troubleshoot Skill

Usa esta skill para diagnosticar problemas del CLI y del entorno.

Pasos utiles:

- Ver ayuda general: `python cli.py --help`
- Ejecutar con tracebacks: `python cli.py --debug <command>`
- Revisar navegador: `config browser show`
- Comprobar sintaxis: `python -m compileall app.py cli.py install.py src`
- Ejecutar tests: `python -m pytest tests -q`
- Ver cambios locales: `git status`
- Buscar errores: usar `grep` o `rg` sobre `src/` y `tests/`

Validacion:

- No ejecutes comandos destructivos sin confirmacion.
- Para errores de scraping revisa `storage/network/browser.json`, navegador instalado, driver y conectividad.
- Para errores de modelo confirma `league show <league_id>` y `model list <league_id>`.
