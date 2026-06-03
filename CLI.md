# ProphitBet CLI

This project is now terminal-first. The application entrypoints are:

```bash
python cli.py
python cli.py --help
python app.py --help
bash app.sh --help
```

The CLI is designed for local machines and VPS environments without a graphical display. It uses Rich terminal tables, guided prompts, confirmations for destructive actions, CSV/XLSX exports, matplotlib's non-GUI backend, and headless browser scraping by default.

## GUI to CLI Map

| Former GUI area | CLI replacement |
| --- | --- |
| File / New League | `league create` |
| File / Load League | `league show`, `league update`, `data show` |
| File / Delete League | `league delete --yes` |
| Table / Find | `data search` |
| Table / Hide Missing | `data show --hide-missing`, `data export --hide-missing` |
| Table / Copy | `data export --output file.csv` |
| Analysis / Descriptions | `analysis descriptive` |
| Analysis / Distributions | `analysis distributions` |
| Analysis / Variances | `analysis variance` |
| Analysis / Correlations | `analysis correlation` |
| Analysis / Boruta Selections | `analysis boruta` |
| Analysis / Coefficients | `analysis coefficients` |
| Analysis / Impurity Analysis | `analysis impurity` |
| Analysis / Rule Extraction | `analysis rules` |
| Models / Train | `model train <league> <model-type>` |
| Models / Evaluate | `model evaluate` |
| Models / Manage Models | `model list`, `model metrics`, `model delete` |
| Models / Interpretability | `explain boundary`, `explain pdp`, `explain waterfall`, `explain shap`, `explain extra` |
| Predict / Predict Manual | `predict manual` |
| Predict / Predict Fixtures | `predict fixtures` |
| View / Theme | Removed; terminal output uses Rich formatting |
| Help / Learn, Update, Bug, Donation | `resources` |

## League Workflow

List downloadable leagues:

```bash
python cli.py league list --catalog
```

Create a league:

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

Inspect or update:

```bash
python cli.py league show epl-2018 --rows 20
python cli.py league update epl-2018
```

Delete, with explicit confirmation:

```bash
python cli.py league delete epl-2018
python cli.py league delete epl-2018 --yes
```

## Data Tools

```bash
python cli.py data show epl-2018 --rows 30 --hide-missing
python cli.py data search epl-2018 Arsenal --column Home
python cli.py data export epl-2018 --output exports/epl.csv --hide-missing
```

## Model Training

Supported model types:

```text
logistic, discriminant, decision-tree, random-forest, xgboost, knn, naive-bayes, svm, dnn
```

Train with GUI-equivalent defaults:

```bash
python cli.py model train epl-2018 random-forest \
  --id epl-rf-result \
  --target result \
  --normalizer standard \
  --eval-size 20 \
  --cv \
  --sliding-cv
```

Train with tuning:

```bash
python cli.py model train epl-2018 xgboost \
  --id epl-xgb-uo \
  --target over-under \
  --normalizer standard \
  --tune n_estimators,max_depth,learning_rate \
  --trials 50 \
  --objective F1
```

View stored metrics or delete a model:

```bash
python cli.py model list epl-2018
python cli.py model metrics epl-2018 epl-rf-result --export-dir exports/metrics
python cli.py model delete epl-2018 epl-rf-result --yes
```

## Evaluation Filters

Evaluate a model on all rows:

```bash
python cli.py model evaluate epl-2018 --model epl-rf-result --dataset all
```

Apply an odds filter and probability percentiles, then store it for fixture prediction:

```bash
python cli.py model evaluate epl-2018 \
  --model epl-rf-result \
  --dataset eval \
  --odd-filter "1:1.31:1.60" \
  --p1 70 --px 60 --p2 70 \
  --store-filter
```

Delete a stored filter:

```bash
python cli.py model evaluate epl-2018 \
  --model epl-rf-result \
  --odd-filter "1:1.31:1.60" \
  --delete-filter
```

## Predictions

Manual prediction:

```bash
python cli.py predict manual epl-2018 \
  --model epl-rf-result \
  --home Arsenal \
  --away Chelsea \
  --odd-1 2.10 \
  --odd-x 3.40 \
  --odd-2 3.10 \
  --output exports/manual.csv
```

Fixture prediction from a local file:

```bash
python cli.py predict fixtures epl-2018 \
  --model epl-rf-result \
  --input fixtures.csv \
  --filters all \
  --output exports/fixtures.csv
```

Fixture prediction from FootyStats in headless mode:

```bash
python cli.py predict fixtures epl-2018 \
  --model epl-rf-result \
  --date 2026-08-15 \
  --headless \
  --filters all
```

Fixture input files must include:

```text
Home,Away,1,X,2
```

## Analysis

All analysis commands save image outputs. Descriptive analysis can also export a table.

```bash
python cli.py analysis descriptive epl-2018 --feature-type home --output outputs/descriptive.png --table-output outputs/descriptive.csv
python cli.py analysis distributions epl-2018 --column Result --output outputs/result-dist.png
python cli.py analysis variance epl-2018 --output outputs/variance.png
python cli.py analysis correlation epl-2018 --method spearman --feature-type away --output outputs/correlation.png
python cli.py analysis boruta epl-2018 --target result --output outputs/boruta.png
python cli.py analysis coefficients epl-2018 --target over-under --output outputs/coefficients.png
python cli.py analysis impurity epl-2018 --target result --output outputs/impurity.png
python cli.py analysis rules epl-2018 --target result --depth 4 --output outputs/rules.png
```

## Interpretability

Common plots:

```bash
python cli.py explain boundary epl-2018 epl-rf-result --features "1,HW%" --output outputs/boundary.png
python cli.py explain pdp epl-2018 epl-rf-result --feature "1" --target H --output outputs/pdp.png
python cli.py explain waterfall epl-2018 epl-rf-result --match-index 0 --target H --output outputs/waterfall.png
python cli.py explain shap epl-2018 epl-rf-result --target H --output outputs/shap.png
```

Model-specific plots:

```bash
python cli.py explain extra epl-2018 epl-rf-result --plot impurity --output outputs/impurity-model.png
python cli.py explain extra epl-2018 epl-rf-result --plot tree --depth 3 --estimator-id 0 --output outputs/tree.png
python cli.py explain extra epl-2018 epl-logistic --plot coefficients --output outputs/logistic-coefficients.png
python cli.py explain extra epl-2018 epl-logistic --plot model --feature "1" --output outputs/logistic-model.png
python cli.py explain extra epl-2018 epl-dnn --plot attention --output outputs/attention.png
```

## Agent Mode

Start an interactive terminal agent on top of the existing CLI. This is a CLI-only assistant: it does not call an LLM, and it works through slash commands, disk skills, session memory and safe tools.

```bash
python cli.py
```

Traditional CLI subcommands remain available, and the default `python cli.py` entrypoint opens the agent. Agent slash commands delegate to the same handlers used by commands such as `python cli.py league list --catalog`.

Useful commands inside the agent:

```text
/help
/skills
/skill train epl-2018 random-forest
/league list --catalog
/league show epl-2018
/model list epl-2018
/predict manual epl-2018 --model rf-result --home Arsenal --away Chelsea --odd-1 2.10 --odd-x 3.40 --odd-2 3.10
/predict fixtures epl-2018 --model rf-result --input fixtures.csv
/analysis variance epl-2018 --output outputs/variance.png
/explain shap epl-2018 rf-result --target H --output outputs/shap.png
/status
/exit
```

Agent context shortcuts:

```text
@CLI.md resume los comandos principales
!git status
/run league list --catalog
```

Skills are loaded from `skills/*/SKILL.md` and optional local skills from `.mlstatssoccer/skills/*/SKILL.md`. Use `/skills` to list available skills with examples, and `/skill <name>` to show the full instructions and command examples. Session history and compact summaries are stored under `.mlstatssoccer/sessions/`.

Risky operations such as `rm`, `git reset`, `git checkout`, `league delete` and `model delete` require explicit confirmation in agent mode.

## Browser Configuration

The scraper reads `storage/network/browser.json`. Headless mode is enabled by default.

```bash
python cli.py config browser show
python cli.py config browser set --application chrome --headless
python cli.py config browser set --application firefox --no-headless
```

VPS requirements for scraping still apply: a supported browser and matching Selenium driver must be available in the environment.

## Checks

Syntax smoke check:

```bash
python -m compileall app.py cli.py install.py src
```

Unit tests:

```bash
python -m pytest tests -q
```
