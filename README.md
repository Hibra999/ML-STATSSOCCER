# ML-STATSSOCCER

ML-STATSSOCCER es un proyecto de analisis y prediccion de futbol con Python. Incluye una aplicacion web local, una app dedicada al Mundial 2026 y una CLI para automatizar tareas de datos, modelos y predicciones.

## Que Incluye

- Gestion de ligas y datasets historicos.
- Preparacion de datos, seleccion de variables y transformaciones estadisticas.
- Entrenamiento y evaluacion de modelos de Machine Learning.
- Prediccion de fixtures y resultados 1/X/2.
- Analisis exploratorio, graficos e interpretabilidad.
- Aplicacion independiente para simulacion y reportes del Mundial 2026.
- CLI para flujos repetibles desde terminal.

## Mundial 2026

![ML-STATSSOCCER Mundial 2026](src/web/static/img/worldcup-dashboard-bg.webp)

El modulo Mundial 2026 concentra el flujo internacional del proyecto: datos historicos, generacion de features, modelos estadisticos de goles, boosting moderno, backtesting temporal y reportes locales para fixtures del torneo.

Capacidades principales:

- Prediccion de partidos del Mundial 2026.
- Modelos Poisson para goles y probabilidades de resultado.
- Modelos avanzados con LightGBM, XGBoost, CatBoost y NGBoost.
- Flujo xG + LightGBM para evaluacion de rendimiento.
- Backtesting con separacion temporal de train, validacion y test.
- Reportes web con metricas, tablas y simulaciones.

## Stack Tecnico

- Python 3.11.
- FastAPI y Uvicorn para la capa web.
- pandas, NumPy, Polars y SciPy para procesamiento numerico.
- scikit-learn, imbalanced-learn y Boruta para pipelines de ML.
- LightGBM, XGBoost, CatBoost y NGBoost para modelos de boosting.
- statsmodels, PyMC y CmdStanPy para modelos estadisticos.
- TensorFlow/Keras para modelos neuronales.
- Matplotlib, Seaborn y SHAP para analisis e interpretabilidad.
- HTML, CSS y JavaScript para la interfaz.
- pytest para pruebas.

## Estructura Del Proyecto

```text
app.py                 Aplicacion web principal
mundial.py             Aplicacion Mundial 2026
cli.py                 Entrada de la CLI
src/web/               Servidores, servicios y archivos estaticos
src/worldcup/          Datos, features, modelos y simulacion Mundial
src/models/            Modelos y entrenamiento
src/preprocessing/     Preparacion y seleccion de datos
src/network/           Descarga y scraping de datos
src/analysis/          Analisis estadistico y graficos
src/interpretability/  Explicabilidad de modelos
tests/                 Suite de pruebas
storage/               Cache y datos locales del proyecto
screenshots/           Imagenes de referencia de la interfaz
```

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecucion

Aplicacion principal:

```bash
python app.py
```

Abrir:

```text
http://127.0.0.1:5050
```

Aplicacion Mundial 2026:

```bash
python mundial.py
```

Abrir:

```text
http://127.0.0.1:5052
```

CLI:

```bash
python cli.py --help
```

## Uso Basico

1. Crear o cargar una liga.
2. Revisar el dataset disponible.
3. Entrenar un modelo.
4. Evaluar metricas y rendimiento.
5. Generar predicciones para fixtures.
6. Revisar analisis, graficos e interpretabilidad.

## Modelos Soportados

- LightGBM.
- XGBoost.
- CatBoost.
- NGBoost.
- Modelos clasicos de scikit-learn.
- Modelos neuronales con TensorFlow/Keras.
- Modelos estadisticos para goles y resultados.

Objetivos principales:

- `result`: resultado 1/X/2.
- `over-under`: mercado U/O 2.5.

## Verificacion

```bash
python -m compileall app.py mundial.py cli.py install.py src
python -m pytest tests -q
```

## Licencia

Este proyecto usa licencia MIT. Ver [LICENSE.txt](LICENSE.txt).
