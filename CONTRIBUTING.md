# Contributing to KagglePipe

Thanks for improving KagglePipe. Small, focused pull requests are easiest to review.

## Development setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"  # Linux/macOS
# Windows: .venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Before opening a pull request, run:

```bash
python -m ruff check src tests
python -m pytest
python -m compileall -q src
```

Keep user-facing changes covered by tests and document any new or changed CLI behavior in the README or quickstart. Do not include Kaggle credentials, generated data, or `.kagglepipe/` state in a commit.

## Integration tests

The real Kaggle integration test is opt-in because it creates remote resources:

```bash
KAGGLEPIPE_RUN_INTEGRATION=1 python -m pytest -m integration
```

Run it only against credentials and a Kaggle account you control.
