---
name: install
description: Prepara el entorno Python y verifica dependencias del CLI.
aliases:
  - setup
  - dependencies
when_to_use:
  - cuando el usuario quiera instalar configurar entorno venv conda requirements
  - cuando mencione Python 3.11 TensorFlow dependencias VPS o app.sh
arguments:
  - environment
examples:
  - "/skill install"
  - "!python --version"
  - "/run resources"
allowed_tools:
  - bash
  - read
  - cli
user_invocable: true
disable_model_invocation: false
---

# Install Skill

Usa esta skill para preparar una instalacion local o VPS.

Comandos recomendados:

- Crear venv: `python3.11 -m venv .venv`
- Activar: `source .venv/bin/activate`
- Instalar: `pip install -r requirements.txt`
- Instalador alternativo: `python install.py --venv .venv`
- Smoke check: `python cli.py --help`
- Compilacion: `python -m compileall app.py cli.py install.py src`

Notas:

- El proyecto recomienda Python 3.11 por TensorFlow.
- En VPS no se requiere GUI.
- Para scraping instala navegador y driver compatible con Selenium.
