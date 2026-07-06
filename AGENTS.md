# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.11 soccer analytics and prediction project. Root wrappers include `app.py` for the main web app, `mundial.py` for the World Cup 2026 app, and `cli.py` for automation. Core packages live under `src/`: `src/web` contains FastAPI/static UI code, `src/cli` contains command handlers, `src/worldcup` contains World Cup data, feature, simulation, and training logic, and `src/models`, `src/preprocessing`, `src/network`, `src/analysis`, and `src/interpretability` hold the broader ML pipeline. Tests are in `tests/`. Cached datasets and local configuration live in `storage/`; UI/reference images are in `screenshots/`.

## Build, Test, and Development Commands

Create a local environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the main local web app with `python app.py`, then open `http://127.0.0.1:5050`. Use `python app.py --port 5051` for another port. Run the Mundial app with `python mundial.py` at `http://127.0.0.1:5052`. Use `python cli.py --help` for CLI commands. Static verification is `python -m compileall app.py mundial.py cli.py install.py src`; the full suite is `python -m pytest tests -q`.

## Coding Style & Naming Conventions

Use Python 3.11, 4-space indentation, `snake_case` for modules/functions/variables, and `PascalCase` for classes. Follow the existing direct, module-oriented style before adding abstractions. Keep imports grouped as standard library, third-party, then local modules. No formatter config is committed, so keep edits consistent with nearby files.

## Testing Guidelines

Tests use `pytest` and follow `tests/test_*.py` naming. Add focused tests near the behavior changed, especially for World Cup data preparation, model reports, CLI parsing, and web-local flows. No coverage gate is configured. Prefer small fixtures and cached data under `storage/worldcup/cache`; avoid live scraping or long model training.

## Commit & Pull Request Guidelines

Recent history uses short imperative commit subjects, for example `Use automatic World Cup benchmark window` and `Unify Mundial prediction report flow`. Keep commits focused and use a clear subject without trailing punctuation. Pull requests should describe the user-visible change, note data/model implications, list verification performed, link issues when applicable, and include screenshots for UI changes.

## Security & Agent-Specific Notes

Do not commit virtual environments, private models, sensitive datasets, browser profiles, cookies, or API keys. `API_FOOTBALL_KEY` should remain an environment variable. This VPS has limited resources: avoid heavy tests, training runs, builds, or long verification commands here unless explicitly requested; prefer static review and lightweight checks.
