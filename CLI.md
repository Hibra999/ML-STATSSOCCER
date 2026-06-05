# ML-STATSSOCCER CLI

La interfaz principal del proyecto es la web local:

```bash
python app.py
```

URL:

```text
http://127.0.0.1:5050
```

La CLI queda como herramienta secundaria para automatizacion y tareas puntuales.

## Entry Points

```bash
python app.py
python app.py --port 5051
python cli.py --help
bash app.sh
```

`python cli.py` sin argumentos ya no abre ningun modo interactivo; muestra ayuda y recomienda la web local.

## League Workflow

```bash
python cli.py league list --catalog

python cli.py league create \
  --league-index 6 \
  --id epl-2018 \
  --start-year 2018 \
  --history-window 3 \
  --goal-margin 2 \
  --stats all \
  --yes

python cli.py league show epl-2018 --rows 20
python cli.py league update epl-2018
python cli.py league delete epl-2018 --yes
```

## Data Tools

```bash
python cli.py data show epl-2018 --rows 30 --hide-missing
python cli.py data search epl-2018 Arsenal --column Home
python cli.py data export epl-2018 --output exports/epl.csv --hide-missing
```

## Model Training

Modelos:

```text
ngboost, catboost, lightgbm, xgboost
```

Ejemplo:

```bash
python cli.py model train epl-2018 xgboost \
  --id epl-xgb-result \
  --target result \
  --normalizer standard \
  --eval-size 20 \
  --tune all \
  --trials 25 \
  --optuna-sampler tpe \
  --cv \
  --sliding-cv
```

## Evaluation

```bash
python cli.py model evaluate epl-2018 --model epl-xgb-result --dataset all

python cli.py model evaluate epl-2018 \
  --model epl-xgb-result \
  --dataset eval \
  --odd-filter "1:1.31:1.60" \
  --p1 70 --px 60 --p2 70 \
  --store-filter
```

## Predictions

```bash
python cli.py predict manual epl-2018 \
  --model epl-xgb-result \
  --home Arsenal \
  --away Chelsea \
  --odd-1 2.10 \
  --odd-x 3.40 \
  --odd-2 3.10

python cli.py predict fixtures epl-2018 \
  --model epl-xgb-result \
  --input fixtures.csv \
  --filters all \
  --output exports/fixtures.csv
```

Fixture input files must include:

```text
Home,Away,1,X,2
```

## Analysis And Interpretability

```bash
python cli.py analysis variance epl-2018 --output outputs/variance.png
python cli.py analysis correlation epl-2018 --method spearman --output outputs/correlation.png
python cli.py analysis rules epl-2018 --target result --depth 4 --output outputs/rules.png

python cli.py explain shap epl-2018 epl-xgb-result --target H --output outputs/shap.png
python cli.py explain extra epl-2018 epl-xgb-result --plot impurity --output outputs/impurity.png
```

## Browser Configuration

```bash
python cli.py config browser show
python cli.py config browser set --application chrome --headless
python cli.py config browser set --application firefox --no-headless
```

## Checks

```bash
python -m compileall app.py cli.py install.py src
python -m pytest tests -q
```
