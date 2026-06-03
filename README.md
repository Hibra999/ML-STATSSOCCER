# ML-STATSSOCCER CLI

Aplicacion de prediccion de futbol con Machine Learning, ahora enfocada 100% en terminal. No requiere interfaz grafica y puede ejecutarse en local o en un VPS.

El CLI permite:

- Crear, listar, actualizar y eliminar ligas.
- Inspeccionar, buscar y exportar datasets.
- Entrenar, evaluar y administrar modelos.
- Predecir partidos manuales y fixtures.
- Aplicar filtros por odds y percentiles.
- Exportar resultados a CSV/XLSX.
- Ejecutar analisis estadistico.
- Generar graficos de interpretabilidad.
- Usar scraping en modo headless.

La documentacion completa de comandos esta en [CLI.md](CLI.md).

## Instalacion Con Conda

```bash
git clone git@github.com-hibra999:Hibra999/ML-STATSSOCCER.git
cd ML-STATSSOCCER

conda create -n prophitbet python=3.11 -y
conda activate prophitbet

pip install -r requirements.txt
```

Usa Python 3.11 para este entorno. TensorFlow y sus paquetes auxiliares son sensibles a la version de Python.

Comprobar que el CLI carga:

```bash
python cli.py --help
```

`python app.py --help` tambien funciona, pero `cli.py` es el entrypoint recomendado.

## Agent Mode

Ademas de los comandos tradicionales, el proyecto incluye un agente interactivo de terminal. No usa un modelo LLM: funciona como shell CLI con slash commands, skills en disco, memoria de sesion y ejemplos guiados.

```bash
python cli.py agent
python cli.py chat
```

Dentro del agente puedes usar slash commands que delegan al CLI existente:

```text
/help
/skills
/skill train epl-2018 random-forest
/league list --catalog
/model list epl-2018
/predict fixtures epl-2018 --model rf-result --input fixtures.csv
/status
/exit
```

El agente carga skills desde `skills/*/SKILL.md` y tambien desde `.mlstatssoccer/skills/*/SKILL.md` si existen. `/skills` muestra las skills con ejemplos de comandos listos para usar. La memoria de sesion vive en `.mlstatssoccer/sessions/` y no requiere interfaz grafica.

## Comandos Principales

Listar ligas disponibles para descargar:

```bash
python cli.py league list --catalog
```

Crear una liga:

```bash
python cli.py league create \
  --league-index 6 \
  --id epl-2018 \
  --start-year 2018 \
  --history-window 3 \
  --goal-margin 2 \
  --stats all \
  --yes
```

Ver o actualizar una liga:

```bash
python cli.py league show epl-2018 --rows 20
python cli.py league update epl-2018
```

Explorar datos:

```bash
python cli.py data show epl-2018 --rows 30 --hide-missing
python cli.py data search epl-2018 Arsenal --column Home
python cli.py data export epl-2018 --output exports/epl.csv --hide-missing
```

Entrenar un modelo:

```bash
python cli.py model train epl-2018 random-forest \
  --id rf-result \
  --target result \
  --normalizer standard
```

Evaluar un modelo y guardar filtros:

```bash
python cli.py model evaluate epl-2018 \
  --model rf-result \
  --dataset eval \
  --odd-filter "1:1.31:1.60" \
  --p1 70 \
  --store-filter
```

Prediccion manual:

```bash
python cli.py predict manual epl-2018 \
  --model rf-result \
  --home Arsenal \
  --away Chelsea \
  --odd-1 2.10 \
  --odd-x 3.40 \
  --odd-2 3.10
```

Prediccion de fixtures desde archivo:

```bash
python cli.py predict fixtures epl-2018 \
  --model rf-result \
  --input fixtures.csv \
  --filters all \
  --output exports/fixtures.csv
```

Prediccion de fixtures con scraping headless:

```bash
python cli.py predict fixtures epl-2018 \
  --model rf-result \
  --date 2026-08-15 \
  --headless \
  --filters all
```

## Modelos Soportados

```text
logistic
discriminant
decision-tree
random-forest
xgboost
knn
naive-bayes
svm
dnn
```

Targets soportados:

```text
result       -> 1/X/2
over-under   -> U/O 2.5
```

## Analisis E Interpretabilidad

Ejemplos:

```bash
python cli.py analysis variance epl-2018 --output outputs/variance.png
python cli.py analysis correlation epl-2018 --method spearman --output outputs/correlation.png
python cli.py analysis rules epl-2018 --target result --depth 4 --output outputs/rules.png

python cli.py explain shap epl-2018 rf-result --target H --output outputs/shap.png
python cli.py explain extra epl-2018 rf-result --plot impurity --output outputs/impurity.png
```

## Configuracion De Scraping

El navegador se configura en `storage/network/browser.json`. Por defecto usa Chrome en modo headless:

```json
{
  "application": "chrome",
  "headless": true
}
```

Comandos:

```bash
python cli.py config browser show
python cli.py config browser set --application chrome --headless
python cli.py config browser set --application firefox --no-headless
```

Para scraping en VPS necesitas tener instalado el navegador elegido y su driver compatible para Selenium.

## Checks

```bash
python -m compileall app.py cli.py install.py src
python -m pytest tests -q
```

## Notas

- No se deben commitear entornos virtuales, modelos privados, cookies, perfiles de navegador ni datos descargados sensibles.
- Las operaciones destructivas, como eliminar ligas o modelos, piden confirmacion salvo que uses `--yes`.
- Para una guia mas completa de comandos y opciones, revisa [CLI.md](CLI.md).
