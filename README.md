# ML-STATSSOCCER

ML-STATSSOCCER es una aplicacion local para analisis y prediccion de partidos de futbol con modelos de Machine Learning. La interfaz principal es una aplicacion web que se ejecuta solo en `localhost`.

## Nota Para Agentes En Este VPS

Este VPS tiene recursos limitados. No ejecutes pruebas, entrenamientos, builds pesados ni comandos de verificacion largos en este entorno, aunque existan instrucciones de testing en este README. Haz cambios de codigo con revision estatica ligera y deja que las pruebas se ejecuten en una maquina local con mejor hardware.

Para Mundial, el flujo vigente usa solo el dataset internacional `all_matches.csv` desde 2014. El ETL debe dividir por tiempo en 80/10/10: train inicial, validacion intermedia y test final. El entrenamiento de boosting queda en perfil de features `balanced` por defecto: maximo 480 columnas, sin `train.csv/test.csv`, sin familia `kaggle_`, con `history` compacto y ventanas 3/5/10. Si un agente toca esta parte, debe conservar esos defaults salvo que el usuario pida explicitamente modo completo.

Cuando termines cambios en este repositorio, commitea y sube todo a Git:

```bash
git add <archivos modificados>
git commit -m "mensaje claro"
git push origin main
```

## Caracteristicas

- Gestion de ligas historicas.
- Exploracion y exportacion de datasets.
- Entrenamiento y evaluacion de modelos.
- Prediccion automatica de futuros partidos desde scraping.
- Analisis estadistico e interpretabilidad.
- Configuracion local del navegador para scraping.

## Requisitos

- Python 3.11.
- Navegador compatible para scraping, si se usa esa funcion: Chrome, Firefox, Edge o Brave.
- Driver compatible con Selenium, si el navegador lo requiere.

TensorFlow y sus dependencias son sensibles a la version de Python. Se recomienda usar un entorno virtual dedicado.

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### CUDA local opcional

El servidor puede correr sin CUDA. En una PC local con NVIDIA, este proyecto mantiene `numpy==1.26.4` por TensorFlow 2.15 y Numba, asi que no instales `cupy-cuda13x` sin version: puede resolver a CuPy 14 + NumPy 2.x. Para RTX 5070 con driver/CUDA UMD 13.3 usa CuPy CUDA 13 fijado a 13.6.0 y el runtime/NVRTC CUDA 13.3 de PyPI:

```powershell
python -m pip uninstall -y cupy cupy-cuda12x cupy-cuda13x cuda-toolkit cuda-pathfinder nvidia-cublas nvidia-cuda-runtime nvidia-cuda-nvrtc nvidia-cufft nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-nvjitlink numpy ml-dtypes
python -m pip install -r requirements.txt --force-reinstall
python -m pip install -r requirements-gpu-cuda13.txt --force-reinstall
python -m pip check
python -c "import numpy as np, cupy as cp; print('numpy', np.__version__); print('gpus', cp.cuda.runtime.getDeviceCount()); print('runtime', cp.cuda.runtime.runtimeGetVersion()); print('probe', float(cp.sum(cp.arange(8, dtype=cp.float32)).get()))"
```

Para CUDA 12.x usa `requirements-gpu-cuda12.txt`. No mezcles `cupy-cuda12x` y `cupy-cuda13x` en el mismo entorno. Si CuPy no puede cargar NVRTC o alguna DLL CUDA, la app cae automaticamente a CPU/NumPy y lo marca como `CPU fallback` en vez de detener el reporte.

## Ejecucion

Iniciar la aplicacion web:

```bash
python app.py
```

Abrir en el navegador:

```text
http://127.0.0.1:5050
```

Para usar otro puerto:

```bash
python app.py --port 5051
```

Iniciar la aplicacion independiente del Mundial 2026:

```bash
python mundial.py
```

Abrir en:

```text
http://127.0.0.1:5052
```

Features opcionales para Mundial desde API-Football:

```bash
export API_FOOTBALL_KEY="tu_api_key"
python mundial.py
```

La app usa la cache local en `storage/worldcup/api_football/` y solo intenta descargar datos oficiales cuando se refresca el historial/ETL. Las features se construyen con corte temporal por fecha de partido para evitar leakage.

La aplicacion se enlaza a `127.0.0.1`. No esta pensada para exponerse publicamente.

## Uso Basico

1. Crear o cargar una liga desde la seccion **Ligas**.
2. Revisar el dataset desde **Datos**.
3. Entrenar un modelo desde **Modelos**.
4. Evaluar el rendimiento desde **Evaluar**.
5. Generar predicciones desde **Predecir**.
6. Crear graficos desde **Analisis**.

Los procesos largos, como descargas, entrenamientos y graficos pesados, se ejecutan como procesos locales.

## CLI Secundaria

La CLI sigue disponible para automatizacion y tareas puntuales:

```bash
python cli.py --help
python cli.py league list --catalog
python cli.py model list epl-2018
```

Ejemplo de prediccion de fixtures:

```bash
python cli.py predict fixtures epl-2018 \
  --model xgb-result \
  --date 2026-06-05 \
  --filters all \
  --output exports/fixtures.csv
```

## Modelos Soportados

- NGBoost.
- CatBoost.
- LightGBM.
- XGBoost.

Objetivos disponibles:

- `result`: resultado 1/X/2.
- `over-under`: U/O 2.5.

## Configuracion De Scraping

La configuracion del navegador se encuentra en:

```text
storage/network/browser.json
```

Ejemplo:

```json
{
  "application": "chrome",
  "headless": true,
  "brave_binary": ""
}
```

`application` acepta `chrome`, `firefox`, `edge` o `brave`. Si se usa Brave y el sistema no detecta el ejecutable, indique la ruta en `brave_binary` o desde la seccion **Configuracion** de la interfaz web.

Las banderas del catalogo se leen desde:

```text
storage/graphics/countries
```

## Verificacion

```bash
python -m compileall app.py mundial.py cli.py install.py src
python -m pytest tests -q
```

## Notas De Seguridad

- No commitear entornos virtuales.
- No commitear modelos privados, datasets sensibles, cookies ni perfiles de navegador.
- La aplicacion es local y monousuario.
- Los jobs en memoria se pierden al reiniciar el servidor; los archivos ya guardados permanecen en disco.
