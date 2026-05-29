# GraphAlign library


## Run the example implementation script
```uv run --group dev example/main.py```

## Build package
```uv build```
Check dist folder for wheel files.

## Run linter
```uv run ruff check --output-format=concise```
## Run type checker
```uv run ty check --output-format=concise```

## Tests
Run test suite:
```uv run pytest -n auto```

Run test suite with extra info (sequential):
```uv run pytest -n auto - v```
The above wil run the test suite

## Generate Documentation
```uv run --group docs make -C docs html```

Check ```docs > build``` folder for compiled documentation