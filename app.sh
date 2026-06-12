#!/bin/bash
set -e

# Navigate to the directory of the script
cd "$(dirname "$0")"

if [ -f ".venv/bin/activate" ]; then
  . ".venv/bin/activate"
elif [ -f "venv/bin/activate" ]; then
  . "venv/bin/activate"
else
  echo "No se encontro .venv ni venv. Crea el entorno e instala dependencias con: python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

if ! python -c "import pandas" >/dev/null 2>&1; then
  echo "Faltan dependencias en el entorno virtual activo. Ejecuta: pip install -r requirements.txt" >&2
  exit 1
fi

# Run the local web application and pass any provided arguments.
python app.py "$@"
