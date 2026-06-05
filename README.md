# ML-STATSSOCCER

ML-STATSSOCCER es una aplicacion local para analisis y prediccion de partidos de futbol con modelos de Machine Learning. La interfaz principal es una aplicacion web que se ejecuta solo en `localhost`.

## Caracteristicas

- Gestion de ligas historicas.
- Exploracion y exportacion de datasets.
- Entrenamiento y evaluacion de modelos.
- Prediccion manual de partidos.
- Prediccion de fixtures desde archivo o scraping.
- Analisis estadistico e interpretabilidad.
- Configuracion local del navegador para scraping.

## Requisitos

- Python 3.11.
- Navegador compatible para scraping, si se usa esa funcion.
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

La aplicacion se enlaza a `127.0.0.1`. No esta pensada para exponerse publicamente.

## Uso Basico

1. Crear o cargar una liga desde la seccion **Ligas**.
2. Revisar el dataset desde **Datos**.
3. Entrenar un modelo desde **Modelos**.
4. Evaluar el rendimiento desde **Evaluar**.
5. Generar predicciones desde **Predecir**.
6. Crear graficos desde **Analisis**.

Los procesos largos, como descargas, entrenamientos y graficos pesados, se ejecutan como jobs locales.

## CLI Secundaria

La CLI sigue disponible para automatizacion y tareas puntuales:

```bash
python cli.py --help
python cli.py league list --catalog
python cli.py model list epl-2018
```

Ejemplo de prediccion manual:

```bash
python cli.py predict manual epl-2018 \
  --model rf-result \
  --home Arsenal \
  --away Chelsea \
  --odd-1 2.10 \
  --odd-x 3.40 \
  --odd-2 3.10
```

## Modelos Soportados

- Logistic Regression.
- Discriminant Analysis.
- Decision Tree.
- Random Forest.
- XGBoost.
- KNN.
- Naive Bayes.
- SVM.
- Deep Neural Network.

Targets disponibles:

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
  "headless": true
}
```

Tambien puede editarse desde la seccion **Config** de la interfaz web.

## Verificacion

```bash
python -m compileall app.py cli.py install.py src
python -m pytest tests -q
```

## Notas De Seguridad

- No commitear entornos virtuales.
- No commitear modelos privados, datasets sensibles, cookies ni perfiles de navegador.
- La aplicacion es local y monousuario.
- Los jobs en memoria se pierden al reiniciar el servidor; los archivos ya guardados permanecen en disco.
