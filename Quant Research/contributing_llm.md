LLM CONTRIBUTION GUIDELINES
Context
You are writing code, docs, or notebook content for the quant_research_lab project—a provider-plug-in, hexagonal, typed-Python crypto/equities research platform.

1 Global coding rules
Python 3.12+, 100 % type-annotated.

Run through black, isort, flake8, mypy --strict; tests must stay ≥ 85 % coverage.

Never hard-code secrets; read from os.getenv() and reference .env.example.

2 Architecture constraints

Layer	LLM must…
providers/	Implement BaseProvider; expose only fetch_ohlcv, fetch_fundamentals, stream_marketdata. No analytics or storage logic here.
analytics/	Pure functions: pd.DataFrame -> signals.parquet. No network or I/O other than writing returned DataFrame/parquet.
backtest/	Use vectorbt wrappers in engine.py; all new strategies go in src/quant_research/backtest/strategies/ and register via entry-points.
core/	Do not reference crypto-specific strings; rely on models.PriceBar, models.Signal, etc.
storage	Use DuckDB/Parquet exactly as designed in ADR-0001. No Postgres unless the ADR is superseded.
3 Documentation duties
Docstrings: Google-style; every public function/class.

Update ADR index—if a change breaks an accepted ADR, create a new /docs/adr/00NN-*.md (Status: Proposed).

When adding/altering a strategy, parameterise & re-run the validation notebook; commit the new PDF to docs/reports/.

4 Testing & CI
Every new module → matching test file in tests/.

Mock external APIs with pytest-asyncio + respx or local fixtures.

If tests introduce heavy data, stage it under tests/data/ ≤ 2 MB.

5 Commit etiquette (for Copilot-style suggestions)
makefile
Copy
Edit
feat(provider): add Kraken OHLCV websocket
test: coverage for timeout + reconnection
docs(adr): 0005-kraken-provider-added
6 Model-validation workflow
papermill notebooks/validation/template_report.ipynb -p strat new_strategy …

nbconvert --to pdf … → place in docs/reports/.

Add a bullet to docs/README_validation.md summarising Sharpe, max-DD, p-values.

7 Security & ethics
Obey upstream API ToS (see docs/data_governance.md).

No personal data scraping beyond permitted endpoints.

Flag rate-limit risks and include exponential-backoff logic.